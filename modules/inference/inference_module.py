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
from .postprocessing import postprocessing, fix_annotation_frames, remove_dets_with_one_camera
from .preprocessing import remove_non_roi, filter_dets_outside_frame_bounds

class InferenceModule:
    """ A class to perform inference on a given sequence
    and extra methods to generate  and evaluate multi-camera tracks

    Example:
    =========

    dataset = ObjectGraphDataset(...)
    graph, node_df, edge_df, sampled_df = dataset[0]

    inf_module = InferenceModule(model, 
                                 graph,
                                 node_df,
                                 sampled_df, 
                                 "/mcmt_gnn/datasets/AIC20",
                                 device=torch.device('cuda'))
    inf_module.predict_tracks()
    """
    def __init__(self, 
                 model, 
                 data,
                 node_df,
                 data_df, 
                 sequence_path, 
                 gt_df=None,
                 device=torch.device('cpu')):
        """ Constructor

        Parameters
        ==========
        model: torch.nn.Module
            A graph neural network implemented in pytorch
            that uses (x, edge_index, edge_attr, y, edge_labels)
            as input and generates predictions for the edges
        data: torch_geometric.data.Data
            A graph to feed the GNN model.
        node_df: pd.DataFrame
            The information about nodes in the graphs. It must have
            the columns (node_id, object_id, camera)
            This dataframe is produced after using one of the custom PyG datasets
            in this repository.
        data_df: pd.DataFrame
            A dataframe with consolidated tracking annotations
            Requires columns (id, frame, xmin, xmax, ymin, ymax, width, height, camera)
            This should be a single dataframe with information from all
            the sequences that are needed for inference.
        sequence_path: string
            Path to the dataset folder.
        gt_df: pd.DataFrame, optional
            A GROUND TRUTH dataframe with consolidated tracking annotations
            Requires columns (id, frame, xmin, xmax, ymin, ymax, width, height, camera)
            This should be a single dataframe with information from all
            the sequences that are needed for inference.
            **This dataframe MUST be provided if the purpose is to evaluate
            tracking performance**
        device: torch.device, optional
            The device to use for computations.
        """
        self.model = model
        self.model.eval()
        self.data = data
        self.node_df = node_df
        self.data_df = data_df
        self.gt_df = gt_df
        self.num_cameras = len(data_df.camera.unique())
        self.sequence_path = sequence_path
        self.device = device
        
    def predict_edges(self, directed_graph, 
                      remove_unidirectional=False, 
                      allow_pruning=True, 
                      allow_spliting=True):
        """ Generate estimations of graph connectivity
        and post-process to obtain clusters. Each cluster
        represent a single object, and each node within the cluster
        represents a trajectory on a single camera.

        Parameters
        ==========
        directed_graph: bool
            If true, then edges [u, v] != [v, u]. The graph will be 
            treated as directed.
        remove_unidirectional: bool, optional
            If true, it removes edges with just one direction in 
            the postprocessing.

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
            output_dict, _, _ = self.model(self.data.to(self.device))
            logits = output_dict['classified_edges'][-1]
            preds_prob = F.softmax(logits, dim=1).cpu()
            predictions = torch.argmax(logits, dim=1).cpu()

            edge_list = self.data.edge_index.cpu().numpy()

            # Connected components output
            id_pred, predictions = postprocessing(self.num_cameras, 
                                                  preds_prob,
                                                  predictions,
                                                  edge_list,
                                                  self.data.cpu(),
                                                  directed_graph,
                                                  remove_unidirectional,
                                                  allow_pruning,
                                                  allow_spliting)
        return id_pred, predictions, preds_prob
    
    def predict_tracks(self, 
                       frame_width, 
                       frame_height, 
                       directed_graph,
                       allow_pruning,
                       allow_spliting,
                       remove_unidirectional=False,
                       filter_frame_bounds=True, 
                       filter_roi=True,
                       filter_single_camera_nodes=True):
        """ Generate cluster predictions from the GNN and 
        cluster components and generate a detection file 
        with the appropriate object ids.

        Parameters
        ==========
        frame_width: int
            The frame width.
        frame_height: int
            The frame height.
        filter_frame_bounds: bool
            If True, then the detections in the annotations dataframe 
            that go beyond the frame boundaries will be filtered out.
        filter_roi: bool
            Whether to filter using region of interest or not.
        filter_single_camera_nodes: bool
            If true, then IDs that only appears on one camera will be removed
            from detection dataframe (self.data_df)

        Returns
        ==========
        data_df: pd.DataFrame
            The resulting predictions file
        id_pred: np.array
            A 1D array with the predicted vehicle IDs
            that result from the connected components
        edge_pred: torch.tensor
            A 1D tensor with the predictions (positive/negative edges)
        preds_prob: torch.tensor
            A 2D tensor with the same length as edge_pred but with two columns
            The first is the probability of class 0, the second is the probability of \
            class 1.
        """

        # Predict ids from CCs
        id_pred, edge_pred, preds_prob = self.predict_edges(directed_graph, 
                                                            remove_unidirectional, 
                                                            allow_pruning, 
                                                            allow_spliting)

        # Add old IDs as separate column to inspect
        self.data_df['id_old'] = self.data_df.id

        # Generate tracking dataframe
        for n in range(len(self.data.x)):
            id_new = int(id_pred[n])
            cam_id = self.node_df.camera[n]
            id_old = self.node_df.object_id[n]

            # Assign the labels from the connected components to the detections df
            self.data_df.loc[(self.data_df['id_old'] == id_old) & 
                            (self.data_df['camera'] == cam_id),'id'] = id_new

        # If required, remove detections that go beyond the frame limits
        if filter_frame_bounds:
            self.data_df = filter_dets_outside_frame_bounds(self.data_df, 
                                                            frame_width, 
                                                            frame_height)
            if self.gt_df is not None: 
                self.gt_df = filter_dets_outside_frame_bounds(self.gt_df, 
                                                              frame_width, 
                                                              frame_height)
        
        # This is only for the single camera tracking estimates, not gorund truth
        if filter_single_camera_nodes:
            self.data_df = remove_dets_with_one_camera(self.data_df)

        return self.data_df, id_pred, edge_pred, preds_prob
    
    def evaluate_mtmc(self, th):
        """ Use Pymotmetrics to compute multi-camera
        multi-target performance metrics (idf, idp, idf1)
        
        Parameters
        ==========
        th: float
            Intersection over Union (IoU) threshold
            for evaluation of performance

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
        assert self.gt_df is not None, \
            "You must provide a grond truth dataframe to the constructor."
        
        self.data_df.camera = self.data_df.camera.apply(lambda cam: int(cam.replace('c', '')))
        self.gt_df.camera = self.gt_df.camera.apply(lambda cam: int(cam.replace('c', '')))

        # Avoid colisions in the frame index
        gt_df, pred_df = fix_annotation_frames(self.gt_df, self.data_df)

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