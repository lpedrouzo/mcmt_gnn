import sys
import os.path as osp
sys.path.insert(1, osp.abspath('.'))

import yaml
import torch 
import pandas as pd
from modules.torch_dataset.object_graph_dataset import ObjectGraphDataset
from modules.data_processor.annotations_processor import AnnotationsProcessor
from models.reid.resnet import resnet101_ibn_a
from models.mcmt.rgnn import MOTMPNet
from modules.inference.inference_module import InferenceModule

# Loading configuration files
with open("config/evaluation.yml") as config_file:
    config = yaml.safe_load(config_file)['06_tracking_evaluation']
    path_config = config['paths']
    dataset_config = config['dataset_config']

with open("config/training_rgcnn.yml", "r") as arch_file:
    config = yaml.safe_load(arch_file)
    gnn_arch = config["gnn_arch"]

# Loading REID model to generate embeddings for the test dataset
reid_model = resnet101_ibn_a(model_path=path_config['reid_model_path'], device="cpu")

# Consolidating annotations from all cameras in S02 into a single dataframe
data_df = AnnotationsProcessor(sequence_path=path_config['sequence_path'],
                               annotations_filename=dataset_config['annotations_filename']).consolidate_annotations([config['evaluation_sequence']])

# Instantiating the Graph dataset using the detections of data_df 
dataset = ObjectGraphDataset(data_df, 
                             sequence_path_prefix=path_config['sequence_path'], 
                             sequence_names=dataset_config['evaluation_sequence'], 
                             annotations_filename=dataset_config['annotations_filename'], 
                             reid_model=reid_model,
                             num_ids_per_graph=-1, 
                             embeddings_per_it=dataset_config['embeddings_per_iteration'], 
                             resized_img_shape=dataset_config['resized_img_shape'], 
                             orignal_img_shape=dataset_config['original_img_shape'], 
                             augmentation=None)

# There will be just one graph on this dataset with all the nodes loaded at once
graph, node_df, edge_df, sampled_df = dataset[0]

# Loading the trained GNN model 
model_state = torch.load(path_config['gnn_model_checkpoint'])
model = MOTMPNet(gnn_arch)
model.load_state_dict(model_state['model'])

# Loading the inference module for prediction of track and evaluation
inf_module = InferenceModule(model, sampled_df, path_config['sequence_path'])
_ = inf_module.predict_tracks(data_df)

# Loading ground truth files
gt_df = pd.read_csv(path_config['ground_truth_path'], 
                    sep=' ', 
                    names=['camera','id', 'frame', 'xmin', 'ymin', 'width', 'height', 'xworld', 'yworld'])

# Computing results and generating a DataFrame with the metrics summary
summary = inf_module.evaluate_mtmc(gt_df, th=config['motmetrics_config']['th'])
summary.to_csv(path_config['results_output_path'])