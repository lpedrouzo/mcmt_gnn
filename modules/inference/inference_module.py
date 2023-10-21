import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import pandas as pd
import cv2
import motmetrics as mm
from datetime import datetime
from ignite.metrics import Precision, Recall, Accuracy, Loss
from ignite.engine import Engine, Events
from .postprocessing import postprocessing, fix_annotation_frames, remove_non_roi, filter_dets_outside_frame_bounds

class InferenceModule(nn.Module):
    """ A class to perform inference on a given sequence
    and extra methods to generate  and evaluate multi-camera tracks

    Example:
    =========

    dataset = ObjectGraphDataset(...)
    graph, node_df, edge_df, sampled_df = dataset[0]

    inf_module = InferenceModule(model, 
                                 sampled_df, 
                                 "/mcmt_gnn/datasets/AIC20")
    inf_module.predict_tracks((graph, node_df, edge_df))
    """
    def __init__(self, model, data_df, sequence_path, directed_graph=True):
        super().__init__()
        self.model = model
        self.model.eval()
        self.data_df = data_df
        self.num_cameras = len(data_df.camera.unique())
        self.sequence_path = sequence_path
        self.directed_graph = directed_graph

    def forward(self, data):
        """ Generate estimations of graph connectivity
        and post-process to obtain clusters. Each cluster
        represent a single object, and each node within the cluster
        represents a trajectory on a single camera.

        Parameters
        ==========
        data: torch_geometric.data.Data
            A graph to feed the model
        
        Returns
        ==========
        id_pred: torch.tensor
            A tensor with the object ids after performing connected 
            components to cluster nodes.
        predictions: torch.tensor
            Edge predictions from the GNN model.
        """
        with torch.no_grad():
            # Output predictions from GNN            
            output_dict, _ = self.model(data)
            logits = torch.cat(output_dict['classified_edges'], dim=0)
            preds_prob = F.softmax(logits, dim=1).cpu()
            predictions = torch.argmax(preds_prob, dim=1).cpu()

            edge_list = data.edge_index.cpu().numpy()

            # Connected components output
            id_pred, predictions = postprocessing(self.num_cameras, 
                                                  preds_prob,
                                                  predictions,
                                                  edge_list,
                                                  data.cpu(),
                                                  self.directed_graph)
        return id_pred, predictions
    
    def predict_tracks(self, batch, 
                       frame_width=None, 
                       frame_height=None, 
                       filter_frame_bounds=True, 
                       filter_roi=True):
        """ Generate cluster predictions from the GNN and 
        cluster components and generate a detection file 
        with the appropriate object ids.

        Parameters
        ==========
        batch: tuple
            A tuple that contains a graph (torch_geometric.data.Data),
            a DataFrame from nodes (columns: id, camera, sequence),
            and a DataFrame from edges (columns: src_node_id, dst_node_id, 
            src_obj_id, dst_obj_id, edge_labels)
        filter_roi: boolean
            Whether to filter using region of interest or not.
        
        Returns
        ==========
        data_df: pd.DataFrame
            The predictions file
        """
        # Batches from our custom datasets return these vars
        data, node_df, _ = batch

        # Predict ids from CCs
        id_pred, _ = self.forward(data)

        # Generate tracking dataframe
        for n in range(len(data.x)):
            id_new = int(id_pred[n])
            cam_id = node_df.camera[n]
            id_old = data.y[n]

            # Assign the labels from the connected components to the detections df
            self.data_df.loc[(self.data_df['id'] == id_old) & 
                            (self.data_df['camera'] == cam_id),'id'] = id_new

        # If required, remove detections that go beyond the frame limits
        if filter_frame_bounds:
            self.data_df = filter_dets_outside_frame_bounds(self.data_df, frame_width, frame_height)

        # If required, remove detections outside region of interest
        if filter_roi:
            self.data_df = remove_non_roi(self.sequence_path, self.data_df)

        return self.data_df
    
    def evaluate_mtmc(self, gt_df, th):
        """ Use Pymotmetrics to compute multi-camera
        multi-target performance metrics (idf, idp, idf1)
        
        Parameters
        ==========
        gt_df: pd.DataFrame
            The ground truth dataframe with detections from 
            all of the cameras in one single object.
            Both gt_df and the dataframe passed in the constructor of this
            object must have the following columns:
            ('frame', 'id', 'xmin', 'ymin', 'width', 'height')
        th: float
            Threshold

        Returns
        =========
        summary: str
            A summary of the performance metrics provided by
            pymotmetrics.

        Notes
        ========
        pymotmetrics requires columns (FrameId, X, Y, Width, Height, Id)
        Since this is not our convention, we transform the dataframes inside
        this function to keep the main scripts clean and following our own 
        convention ('frame', 'id', 'xmin', 'ymin', 'width', 'height').
        """

        # Avoid colisions in the frame index
        gt_df, pred_df = fix_annotation_frames(gt_df, self.data_df)

        # Setting column names as motmetrics requires it
        gt_df = gt_df.rename(columns={'frame': 'FrameId',
                                    'xmin': 'X', 
                                    'ymin': 'Y', 
                                    'width': 'Width', 
                                    'height': 'Height',
                                    'id': 'Id'})

        pred_df = pred_df.rename(columns={'frame': 'FrameId',
                                    'xmin': 'X', 
                                    'ymin': 'Y', 
                                    'width': 'Width', 
                                    'height': 'Height',
                                    'id': 'Id'})

        # Setting frame id and id as index since motmetrics requires it
        gt_df = gt_df.set_index(['FrameId', 'Id'])
        pred_df = pred_df.set_index(['FrameId', 'Id'])

        # Compute metrics using Pymotmetrics
        mh = mm.metrics.create()
        accumulator = mm.utils.compare_to_groundtruth(gt_df, pred_df, 'iou', distth=th)
        metrics=[*mm.metrics.motchallenge_metrics, *['num_frames','idfp','idfn','idtp']]
        summary = mh.compute(accumulator, metrics=metrics, name='MultiCam')

        return summary
            

class InferenceEngineAbstract(object):
    def __init__(self, model, val_loader, results_path_prefix, metrics):
        self.results_path_prefix = osp.join(results_path_prefix, "metrics_inferece")
        self.results_path = osp.join(self.results_path_prefix, "results_metrics.csv")
        self.val_loader = val_loader
        self.model = model
        self.metrics = {}
        os.makedirs(self.results_path_prefix, exist_ok=True)
        self.setup_validation(metrics)

    def setup_validation(self, metrics):
        """Set up the validation process.

        Parameters:
        ===========
        metrics : list
            List of metric names to be used for evaluation.
        """
 
        # Setting up evaluator engines
        validation_step = self.setup_validation_step()
        self.val_evaluator = Engine(validation_step)
        self.setup_metrics(metrics)


    def setup_metrics(self, metrics):
        """Set up evaluation metrics.
        This function uses the exact strings to instantiate the metric class.
        Valid values are ['Precision', 'Recall', 'Accuracy'] case sensitive.

        Parameters:
        ===========
        metrics : list
            List of metric names to be used for evaluation.
        """

        # Converting strings into ignite.metrics objects and storing in dictionary
        for metric in metrics:
            self.metrics[metric] = getattr(sys.modules[__name__], metric)()

        for name, metric in self.metrics.items():
            metric.attach(self.val_evaluator, name)

        # Preparing empty dataframes for logging
        if os.path.exists(self.results_path):
            self.val_metrics_df = pd.read_csv(self.results_path)
        else:
            self.val_metrics_df = pd.DataFrame(columns=["timestamp", *metrics])


    def run_validation(self):
        
        # Run the evaluator through the
        self.val_evaluator.run(self.val_loader)
        metrics = self.val_evaluator.state.metrics

        # Log current metrics
        row = {"timestamp": datetime.now().strftime('%m/%d/%Y')}
        msg = "Inference Results - "
        for name, _ in self.metrics.items():
            msg += f"{name}: {metrics[name]} - "
            row[name] = metrics[name]
        print(msg)

        # This result logs will besaved to csv once trainer.run() is finished
        row_df = pd.DataFrame([row], columns=self.val_metrics_df.columns)
        self.val_metrics_df = pd.concat([self.val_metrics_df, row_df], ignore_index=True)  

        # Save results
        self.val_metrics_df.to_csv(self.results_path)

    def setup_validation_step(self):
        raise NotImplementedError("You must override this function.")
