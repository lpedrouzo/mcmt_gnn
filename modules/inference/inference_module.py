import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import pandas as pd
from datetime import datetime
from ignite.metrics import Precision, Recall, Accuracy, Loss
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine, create_lr_scheduler_with_warmup, Checkpoint
from .postprocessing import postprocessing

class InferenceModule(nn.Module):
    def __init__(self, model, num_cameras):
        super().__init__()
        self.model = model
        self.model.eval()

        self.num_cameras = num_cameras

    def forward(self, data):
        with torch.no_grad():            
            output_dict, _ = self.model(data)
            logits = torch.cat(output_dict['classified_edges'], dim=0)
            preds_prob = F.softmax(logits, dim=1)
            predictions = torch.argmax(preds_prob, dim=1)

            edge_list = data.edge_index.cpu().numpy()

            id_pred, predictions = postprocessing(self.num_cameras, 
                                                  preds_prob,
                                                  predictions,
                                                  edge_list,
                                                  data)
        
        print(predictions)
        return predictions
    

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
