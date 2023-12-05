import torch
import numpy as np
from sklearn.metrics import (f1_score, 
                             accuracy_score, 
                             precision_score, 
                             recall_score,
                             classification_report)

def train_func(gnn_model, optimizer, train_loader, loss_fn, lr_scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_model.train()

    for batch_idx, graph in enumerate(train_loader):

        graph = graph.to(device)
        optimizer.zero_grad()

        output_dict, latent_nodes_feats, latent_edge_feats = gnn_model(graph)
        logits = output_dict['classified_edges'][-1]
        loss = loss_fn(logits, graph.edge_labels)
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
    
    return loss.cpu().item()

def test_func_multiclass(gnn_model, test_loader, loss_fn, task_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_model.eval()

    # Initialize lists to store true labels and predicted labels
    all_targets = []
    all_predictions = []
    all_logits = []

    with torch.no_grad():
        for batch_idx, graph in enumerate(test_loader):
            graph = graph.to(device)
            output_dict, latent_nodes_feats, latent_edge_feats = gnn_model(graph)
            logits = output_dict['classified_edges'][-1]

            if task_type == 'binary':
                predictions = (logits > 0.5).float()
            else:
                predictions = torch.argmax(logits, dim=1)

            labels = graph.edge_labels

            # Append true labels and predicted labels to the lists
            all_targets.append(labels)
            all_predictions.append(predictions)
            all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    metrics_dict = classification_report(all_targets.cpu().numpy(), 
                                         all_predictions.cpu().numpy(), 
                                         output_dict=True)
    metrics = {
        "loss": loss_fn(all_logits, all_targets.to(device)).cpu().item(),
        "macro_f1_score": metrics_dict['macro avg']['f1-score'],
        "precision": metrics_dict['macro avg']['precision'],
        "recall": metrics_dict['macro avg']['recall'],
        "f1_class0": metrics_dict['0']['f1-score'],
        "precision_class0": metrics_dict['0']['precision'],
        "recall_class0": metrics_dict['0']['recall'],
        "f1_class1": metrics_dict['1']['f1-score'],
        "precision_class1": metrics_dict['1']['precision'],
        "recall_class1": metrics_dict['1']['recall'],
        "accuracy": metrics_dict['accuracy']
        }
    return metrics, graph, latent_nodes_feats, latent_edge_feats
