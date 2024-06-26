import sys
import os
import os.path as osp
sys.path.insert(1, osp.abspath('.'))
import yaml
import torch
import torch_geometric.transforms as T
import mlflow
import optuna
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
from datetime import date
from torch_geometric.loader import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR, LambdaLR

from models.mcmt.rgnn import MOTMPNet
from models.mcmt.gallery_rgnn import GalleryMOTMPNet
from modules.torch_dataset.object_graph_reid_precomp import ObjectGraphREIDPrecompDataset
from modules.torch_dataset.object_graph_gallery_precomp import ObjectGraphGalleryPrecompDataset
from modules.tuning.epoch_functions import test_func_multiclass, train_func
from modules.torch_trainer.custom_loss import CECustom
from modules.plots import tsne2d_scatterplot, draw_pyg_network, plot_tsne_samples, plot_histogram_wrapper
from modules.inference.inference_module import InferenceModule
from modules.data_processor.annotations_processor import AnnotationsProcessor
from modules.tuning.trial_helpers import update_gnn_config, get_or_create_experiment, warmup_lr, define_gallery_gnn_config


def trainable_function(task_config, 
                       experiment_params, 
                       search_space, 
                       gnn_arch, 
                       toggles,
                       device, 
                       experiment_id, 
                       trial):
    """ Trainable function For Optuna hyperparameter search.
    Before passing to Optuna, please use partial() from 
    functools to pass values to the parameter task_config with basic configuration.
    
    This function will:
    - Train a GNN for edge prediction using Sequences S01, S03, S04
    - Evaluate link prediction metrics like f1, precision and recall on seqeuence S02
    - Use the GNN to predict Links on a new dataset on sequence S02
    - Perform Postprocessing using the InferenceModule object
    - Execute evaluation using the InferenceModuleObject
    - Report results
    """

    # Task parameters
    sequence_path = task_config['sequence_path']
    test_sequence = task_config['test_sequence']
    gt_filename = task_config['gt_filename']
    sct_filename = task_config['sct_filename']

    # Search space sampling
    config = {
        param_key: trial.suggest_categorical(param_key, param_space)\
            for param_key, param_space in search_space.items()
    }
    config['eval_metric'] = task_config['eval_metric']
    config['trial_number'] = trial.number

    with mlflow.start_run(experiment_id=experiment_id, 
                          run_name=f"{experiment_params['experiment_name']}"
                                    + f"_trial_{trial.number}"
                                    + f"_{experiment_params['num_trials']}"):
        
        # Logging metrics to mlflow
        mlflow.log_params(config)

        mlflow.set_tags({
            "sct_filename": sct_filename,
            "gt_filename": gt_filename,
            "test_sequence": test_sequence,
            "toggle_gnn_update": toggles['modify_gnn_config'],
            "enable_negative_sampling": toggles['enable_negative_sampling'],
            "enable_dynamic_weights": toggles['enable_dynamic_weights']
        })

        # Set dataset obejct based on the type of input 
        if config['input_format'] == 'gallery':
            ObjectDataset = ObjectGraphGalleryPrecompDataset
        else:
            ObjectDataset = ObjectGraphREIDPrecompDataset

        # Instantiation of the dataset objects
        train_dataset = ObjectDataset(sequence_path_prefix=sequence_path,
                                      sequence_names=["S01", "S03", "S04"],
                                      annotations_filename=gt_filename,
                                      num_ids_per_graph=config['num_ids_per_graph'],
                                      return_dataframes=False,
                                      negative_links_ratio=config['ratio_neg_links_graph'] \
                                        if toggles['enable_negative_sampling'] == True else None,
                                      graph_transform=T.ToUndirected())

        val_dataset = ObjectDataset(sequence_path_prefix=sequence_path,
                                    sequence_names=["S02"],
                                    annotations_filename=gt_filename,
                                    num_ids_per_graph=-1,
                                    return_dataframes=False,
                                    graph_transform=T.ToUndirected())

        # Instantiation of the dataloaders
        train_dataloader = DataLoader(train_dataset, 
                                        batch_size=1, 
                                        shuffle=False)
        val_dataloader = DataLoader(val_dataset, 
                                    batch_size=1, 
                                    shuffle=False)

        # GNN, optimizer, criterion, and LR scheduler

        if toggles['modify_gnn_config'] == 'all':
            gnn_arch = update_gnn_config(gnn_arch, config)
        elif toggles['modify_gnn_config'] == 'gallery_gnn':
            gnn_arch = define_gallery_gnn_config(gnn_arch, config)
            
        if config['input_format'] == 'gallery':
            gnn = GalleryMOTMPNet(gnn_arch).to(device)
        else:
            gnn = MOTMPNet(gnn_arch).to(device)

        optimizer = SGD(gnn.parameters(),
                        lr=config["lr"],
                        momentum=config["optimizer_momentum"])

        criterion = CrossEntropyLoss(reduction='mean') \
            if toggles["enable_dynamic_weights"] == False else CECustom()

        lr_scheduler = StepLR(optimizer, 
                                config["lr_scheduler_step_size"], 
                                0.1) if config["lr_scheduler_step_size"] else None

        warmup_scheduler = LambdaLR(optimizer, partial(warmup_lr, warmup_epochs=config['warmup_duration']))\
            if config['warmup_duration'] > 0 else None

        # Perform training
        for epoch in tqdm(range(0, config['epochs'])):
            loss = train_func(gnn, optimizer, train_dataloader, criterion, lr_scheduler)
            metrics, graph, node_feats, _ = test_func_multiclass(gnn, val_dataloader, criterion, 'multiclass')

            if warmup_scheduler: warmup_scheduler.step()

            # Log performance metrics every epoch
            mlflow.log_metrics({**metrics, "train_loss": loss})
            
            # Prepare for a new epoch if trial is not pruned
            train_dataloader.dataset.on_epoch_end()

        
        # Save the model checkpoint
        checkpoint_path = f"model_checkpoint_trial_{trial.number}.pt"
        torch.save(gnn.state_dict(), checkpoint_path)

        # Log the model checkpoint to MLflow
        mlflow.log_artifact(checkpoint_path)

        # Save artifacts to be picked up by mlflow
        tsne = tsne2d_scatterplot(node_feats.cpu(), 
                                  graph.y.cpu(), 
                                  fig_title="TSNE 2D Plot of Node Embeddings",
                                  save_path=None, show=False)
        pyg_graph = draw_pyg_network(graph.cpu(),
                                     fig_title="Graph Connectivity Plot",
                                     layout='spring', 
                                     save_path=None, show=False)

        # Getting samples of embeddings
        figures = plot_tsne_samples(graph, node_feats, [10,20,30,40,50])
        
        # Start the multi-camera association
        data_df = AnnotationsProcessor(sequence_path=sequence_path, 
                                    annotations_filename=sct_filename)\
                                    .consolidate_annotations([test_sequence], ["frame", "camera"])

        gt_df = AnnotationsProcessor(sequence_path=sequence_path, 
                                    annotations_filename=gt_filename)\
                                    .consolidate_annotations([test_sequence], ["frame", "camera"])

        val_dataset = ObjectDataset(sequence_path_prefix=sequence_path,
                                    sequence_names=[test_sequence],
                                    annotations_filename=sct_filename,
                                    num_ids_per_graph=-1,
                                    return_dataframes=True,
                                    graph_transform=T.ToUndirected())

        graph, node_df, edge_df = val_dataset[0]

        inf_module = InferenceModule(gnn, graph, node_df, data_df,
                                    sequence_path, gt_df, device)
        res_df, id_pred, edge_pred, preds_prob = inf_module.predict_tracks(frame_width=1920,
                                                                           frame_height=1080,
                                                                           directed_graph=True,
                                                                           allow_pruning=config['pruning'],
                                                                           allow_spliting=config['spliting'])
        positive_predicted_edges_prob = preds_prob[:, 1][edge_pred == 1]
        negative_predicted_edges_prob = preds_prob[:, 0][edge_pred == 0]

        mlflow.log_metrics({
            "mean_score_predictions": np.mean(preds_prob[:, 1].cpu().numpy()),
            "var_score_predictions": np.var(preds_prob[:, 1].cpu().numpy()),
            "mean_score_predictions_positive": np.mean(positive_predicted_edges_prob.cpu().numpy()),
            "mean_score_predictions_negative": np.mean(negative_predicted_edges_prob.cpu().numpy()),
            "var_score_predictions_positive": np.var(positive_predicted_edges_prob.cpu().numpy()),
            "var_score_predictions_negative": np.var(negative_predicted_edges_prob.cpu().numpy())
        })

        hist_class0 = plot_histogram_wrapper(preds_prob[:, 0], 
                                             "Softmax Scores for Negative Class Output Neuron",
                                             "Probability",
                                             "Histogram of Softmax Scores for Negative Class Output Neuron",
                                             bins=10, save_path=None, show=False)
        hist_class1 = plot_histogram_wrapper(preds_prob[:, 1], 
                                             "Softmax scores for Positive Class Output Neuron",
                                             "Probability",
                                             "Histogram of Softmax Scores for Positive Class Output Neuron",
                                             bins=10, save_path=None, show=False)
        hist_class0_pred0 = plot_histogram_wrapper(negative_predicted_edges_prob, 
                                                   "Softmax Scores for Negative Predicted Edges",
                                                   "Probability",
                                                   "Histogram of Softmax Scores for Negative Predicted Edges",
                                                   bins=10, save_path=None, show=False)
        hist_class1_pred1 = plot_histogram_wrapper(positive_predicted_edges_prob, 
                                                  "Softmax scores for Positive Predicted edges",
                                                  "Probability",
                                                  "Histogram of Softmax Scores for Positive Predicted Edges",
                                                   bins=10, save_path=None, show=False)

        # Prune trial if there is no multi-camera association
        if res_df.empty or sum(edge_pred) == 0:
            mlflow.set_tag("mc_association", False)
            raise optuna.TrialPruned()
        else:
            mlflow.set_tag("mc_association", True)

        # Evaluate if association was possible
        summary = inf_module.evaluate_mtmc(th=0.8)

        tracking_metrics = {
            "idf1": summary.loc["MultiCam", "idf1"],
            "idp": summary.loc["MultiCam", "idp"],
            "idr": summary.loc["MultiCam", "idr"]
        }

        pred_graph = draw_pyg_network(graph.cpu(),
                                      node_labels=id_pred,
                                      edge_labels=edge_pred.cpu().numpy(),
                                      class_ids=[1],
                                      fig_title="Predicted graph Plot",
                                      layout='spring', 
                                      save_path=None, show=False)
        # Log to MLflow metrics
        mlflow.log_metrics(tracking_metrics)

        # Log figures
        mlflow.log_figure(tsne, "node_embeddings.png")
        for tsne_figure, num_samples in figures:
            mlflow.log_figure(tsne_figure, f"node_embeddings{num_samples}.png")
        mlflow.log_figure(pyg_graph, "graph.png")
        mlflow.log_figure(pred_graph, "pred_graph.png")
        mlflow.log_figure(hist_class0, "hist_class0.png")
        mlflow.log_figure(hist_class1, "hist_class1.png")
        mlflow.log_figure(hist_class0_pred0, "hist_class0_pred0.png")
        mlflow.log_figure(hist_class1_pred1, "hist_class1_pred1.png")

        # Log dataframes
        mlflow.log_table(data_df, "test_tracking_sct_df.txt")
        mlflow.log_table(gt_df, "test_tracking_gt_df.txt")
        mlflow.log_table(node_df, "test_tracking_node_df.txt")
        mlflow.log_table(edge_df, "test_tracking_edge_df.txt")
        mlflow.log_table(pd.DataFrame({
            "negative_link_prob": preds_prob[:, 0].cpu().numpy(),
            "positive_link_prob": preds_prob[:, 1].cpu().numpy(),

        }), "prediction_scores.txt")

    return metrics['macro_f1_score']



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading configuration file")
    with open("config/tuning.yml", "r") as config_file:
        yaml_file = yaml.safe_load(config_file)['06_hyperparameter_tuning']
        task_config = yaml_file['dataset_params']
        experiment_params = yaml_file['experiment_params']
        search_space = yaml_file['search_space']
        toggles = yaml_file['hyperparam_toggles']
    
    print("Loading GNN default config")
    with open("config/training_rgcnn.yml", "r") as config_file:
        config = yaml.safe_load(config_file)
        gnn_arch = config["gnn_arch"]

    mlflow.set_tracking_uri("http://192.168.23.226:5000")
    
    # Set the experiment as the annotations name plus the experiment block id
    experiment_name = task_config['sct_filename'].replace('.txt', '')\
                      + "_" + experiment_params['experiment_name']
    experiment_id = get_or_create_experiment(experiment_name)
    
    # Insert the task config parameters into the trainable function
    trainable = partial(trainable_function, 
                        task_config, 
                        experiment_params,
                        search_space, 
                        gnn_arch, 
                        toggles,
                        device,
                        experiment_id)

    

    print("Loading Optuna")  
    # Initialize the Optuna study
    study = optuna.create_study(direction="maximize", 
                                sampler=optuna.samplers.GridSampler(
                                    search_space=search_space
                                ) if experiment_params["sampler"]=='grid' else None,
                                study_name=experiment_name,
                                storage=f"sqlite:///results/{experiment_name}.db",
                                load_if_exists=True)


    # Execute the hyperparameter optimization trials.
    study.optimize(trainable, n_trials=experiment_params['num_trials'])

