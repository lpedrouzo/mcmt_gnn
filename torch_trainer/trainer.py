import torch
import torch.nn.functional as F
from .custom_loss import cross_entropy_with_online_weights

import torch.nn as nn
import torch.nn.functional as F
from .trainer_abstract import TrainingEngineAbstract

class TrainingEngineRGNNMulticlass(TrainingEngineAbstract):
    """Subclass for Recurrent Graph Neural Network
    for multiclass classification and no online weights
    """
    def setup_train_step(self):
        def train_step(engine, batch):
            self.gnn_model.train()
            self.optimizer.zero_grad()
            output_dict, latent_nodes_feats = self.gnn_model(batch)

            logits = torch.cat(output_dict['classified_edges'], dim=0)

            loss = self.loss_fn(logits, batch.edge_labels)
            loss.backward()
            self.optimizer.step()

            return loss.item()
        return train_step

    def setup_validation_step(self):
        def validation_step(engine, batch):
            self.gnn_model.eval()
            with torch.no_grad():
                output_dict, latent_nodes_feats = self.gnn_model(batch)
                logits = torch.cat(output_dict['classified_edges'], dim=0)
            return F.softmax(logits, dim=1), batch.edge_labels
        return validation_step


class TrainingEngineRGNNBinary(TrainingEngineAbstract):
    """Subclass for Recurrent Graph Neural Network
    for binary classification and no online weights
    """
    def setup_train_step(self):
        def train_step(engine, batch):
            self.gnn_model.train()
            self.optimizer.zero_grad()
            output_dict, latent_nodes_feats = self.gnn_model(batch)

            logits = output_dict['classified_edges'][0].squeeze()
            
            loss = self.loss_fn(logits, batch.edge_labels)
            loss.backward()
            self.optimizer.step()

            return loss.item()
        return train_step
        
    def setup_validation_step(self):
        def validation_step(engine, batch):
            self.gnn_model.eval()
            with torch.no_grad():
                output_dict, latent_nodes_feats = self.gnn_model(batch)
                logits = output_dict['classified_edges'][0].squeeze()
            return F.sigmoid(logits), batch.edge_labels
        return validation_step

