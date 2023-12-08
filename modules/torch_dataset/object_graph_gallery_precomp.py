import os
import os.path as osp
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from .utils import simple_negative_sampling, precompute_samples_for_graph_id
from .object_graph_reid_precomp import ObjectGraphREIDPrecompDataset

class ObjectGraphGalleryPrecompDataset(ObjectGraphREIDPrecompDataset):
    """PyTorch Geometric Dataset for sequence-based graph data.

    Example usage
    ==============
    dataset = SequenceGraphDataset(sequence_path_prefix='./data', 
                                   sequence_names=['sequence1', 'sequence2'], 
                                   annotations_filename='annotations.txt')
    print(len(dataset))  # Number of processed graphs
    data = dataset[0]     # Access the first processed graph
    """
    def __init__(self, sequence_path_prefix, 
                 sequence_names, 
                 annotations_filename, 
                 num_ids_per_graph,
                 return_dataframes=False, 
                 negative_links_ratio=None,
                 graph_transform=None,
                 trajectory_folder_name='gallery'):

        super().__init__(sequence_path_prefix, 
                         sequence_names, 
                         annotations_filename, 
                         num_ids_per_graph,
                         return_dataframes,
                         negative_links_ratio,
                         graph_transform,
                         trajectory_folder_name)

        self.process_initial_nodes()


    def get(self, idx):
        """Get a processed graph by the sequence index.

        Parameters
        ===========
        idx : int
            Index of the graph to retrieve.

        Returns
        ===========
        data : Data
            The processed graph data.
        """
        # Retrieve object id sample using idx and filter detection dataset
        id_sample = self.precomputed_id_samples[idx]
        sampled_node_df = self.node_df[self.node_df.object_id.isin(id_sample)].reset_index(drop=True)

        # Reset the node indices after sampling
        sampled_node_df.node_id = range(len(sampled_node_df))

        # smaple the embeddings and node labels as well
        sampled_node_embeddings = self.node_embeddings[self.node_df.object_id.isin(id_sample)]
        sampled_node_labels = self.node_labels[self.node_df.object_id.isin(id_sample)]

        #print("Generating edges")                                     
        # Loading edge information, embeddings and labels for all cameras 
        edge_df, edge_idx, edge_labels = self.setup_edges(sampled_node_df)

        #print("Creating PyG graph")
        graph = Data(x=sampled_node_embeddings, 
                     edge_index=edge_idx, 
                     y=sampled_node_labels, 
                     edge_labels=edge_labels)
        
        # Do transformation if defined
        if self.graph_transform:
            graph = self.graph_transform(graph)

        if self.return_dataframes:
            return graph.to(self.device), sampled_node_df, edge_df
        else:
            return graph.to(self.device)