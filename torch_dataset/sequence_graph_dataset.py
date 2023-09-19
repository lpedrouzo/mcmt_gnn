import os
import os.path as osp
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset

class SequenceGraphDataset(Dataset):
    """PyTorch Geometric Dataset for sequence-based graph data.

    Example usage
    ==============
    dataset = SequenceGraphDataset(sequence_path_prefix='./data', 
                                   sequence_names=['sequence1', 'sequence2'], 
                                   annotations_filename='annotations.txt')
    print(len(dataset))  # Number of processed graphs
    data = dataset[0]     # Access the first processed graph
    """
    def __init__(self, sequence_path_prefix, sequence_names, annotations_filename, transform=None, pre_transform=None):

        self.sequence_path_prefix = sequence_path_prefix
        graph_root = osp.join(sequence_path_prefix, "graphs")
        self.sequence_names = sequence_names
        self.annotations_filename = annotations_filename
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(SequenceGraphDataset, self).__init__(graph_root, transform, pre_transform)


    @property
    def processed_file_names(self):
        """The names of the processed graph files"""
        return [f'sequence_graph_S{i.zfill(2)}.pt' 
                for i in self.sequence_names]


    def setup_nodes(self, camera_folders_prefix, camera_folders):
        """Load and set up node information and embeddings.
        This funciton will create a pandas DataFrame with all of the
        objects on every camera folder. In each row, it will record the camera name,
        the id of the object and the path to its embeddings. Furthermore, using the column
        with the paths, node embeddings will be loaded.

        Parameters
        ===========
        camera_folders_prefix : str
            Prefix for camera folders containing object embeddings.
        camera_folders : list
            List of camera folders.

        Returns
        ===========
        node_df : pd.DataFrame
            DataFrame with node information (columns=["camera", "object_id", "embeddings_path"])
        node_embeddings : torch.Tensor
            Node embeddings.
        node_labels : torch.Tensor
            Node labels (object IDs).
        """
        # Logging node ingormation
        node_df = pd.DataFrame(columns=[("camera", "string"), ("object_id", "int"), ("embeddings_path", "string")])

        # Recording all objects on all cameras
        for camera_folder in camera_folders:
            print(f"Processing nodes for camera {camera_folder}")

            object_path_prefix = osp.join(camera_folders_prefix, camera_folder)
            object_files = os.listdir(object_path_prefix)

            for object_file in object_files:
                node_df = pd.concat([node_df, pd.DataFrame.from_records([{
                    "camera": camera_folder, 
                    "object_id": int(object_file.replace(".pt", "").split("_")[-1]),
                    "embeddings_path": osp.join(object_path_prefix, object_file)
                    }])], ignore_index=True)
                
        node_df["node_id"] = node_df.index
        node_embeddings = torch.stack([torch.load(filepath) for filepath in node_df.embeddings_path.values]).to(self.device)
        node_labels = torch.tensor(node_df.object_id.values).to(self.device)

        return node_df, node_embeddings, node_labels


    def setup_edges(self, node_df, node_embeddings):
        """Load and set up edge information and embeddings.
        This method iterates over all pairs of nodes in the provided `node_df` and establishes edges 
        between nodes that belong to different cameras. For each edge, it records the source and destination node IDs, 
        the corresponding object IDs, and assigns an edge label based on whether the source and destination nodes 
        belong to the same object. The resulting edge information and labels are stored in the `edge_df`, and the
        edge indices are collected in the `edge_idx` tensor. The edge embeddings are computed using the 
        provided node embeddings and are returned as `edge_embeddings`.

        Parameters
        ===========
        node_df : pd.DataFrame
            DataFrame containing node information (columns=["camera", "object_id", "embeddings_path"]).
        node_embeddings : torch.Tensor
            Node embeddings.

        Returns
        ===========
        edge_df : pd.DataFrame
            DataFrame with edge information.
        edge_idx : torch.Tensor
            Edge indices.
        edge_embeddings : torch.Tensor
            Edge embeddings.
        edge_labels : torch.Tensor
            Edge labels.
        """
        # Logging edge information
        edge_df = pd.DataFrame(columns=[("src_node_id", "int"), ("dst_node_id", "int"), ("src_obj_id", "int"), ("dst_obj_id", "int")])

        cameras_visited = []
        edge_idx = []
        edge_labels = []

        for camera in node_df.camera.unique():
            print(f"Processing edges for camera {camera}")
            nodes_in_camera = node_df[node_df.camera == camera].reset_index(drop=True)
            nodes_not_in_camera = node_df[(node_df.camera != camera) & (~node_df.camera.isin(cameras_visited))].reset_index(drop=True)
            
            for i in range(len(nodes_in_camera)):
                for j in range(len(nodes_not_in_camera)):

                    src_node_id, dst_node_id = nodes_in_camera.loc[i, "node_id"], nodes_not_in_camera.loc[j, "node_id"]
                    src_obj_id, dst_obj_id = nodes_in_camera.loc[i, "object_id"], nodes_not_in_camera.loc[j, "object_id"]

                    edge_df = pd.concat([edge_df, pd.DataFrame.from_records([{
                        "src_node_id": src_node_id, "dst_node_id": dst_node_id,
                        "src_obj_id": src_obj_id, "dst_obj_id": dst_obj_id,
                        "edge_label": int(src_obj_id == dst_obj_id)
                    }])], ignore_index=True)

                    edge_idx.append(torch.tensor([src_node_id, dst_node_id]))
                    edge_labels.append(int(src_obj_id == dst_obj_id))

            cameras_visited.append(camera)
        
        # Convert lists to torch tensors and finally compute edge embeddings
        edge_idx = torch.stack(edge_idx).to(self.device)
        edge_labels = torch.tensor(edge_labels).to(self.device)
        edge_embeddings = self.compute_edge_embeddings(node_embeddings, edge_idx)
        return edge_df, edge_idx, edge_embeddings, edge_labels


    def compute_edge_embeddings(self, node_embeddings, edge_idx):
        return torch.cat((
                F.pairwise_distance(node_embeddings[edge_idx.T[0]], node_embeddings[edge_idx.T[1]]).view(-1, 1),
                1 - F.cosine_similarity(node_embeddings[edge_idx.T[0]], node_embeddings[edge_idx.T[1]]).view(-1, 1)
            ), dim=1)
    

    def process(self):
        """Uses the functions setup_nodes and setup_edges to compute all the necessary information
        to build a graph for each sequence. Then, the node information, as well as the edge information is saved
        as JSON logs.

        Parameters
        ===========
        None

        Returns
        ===========
        None
        """
        graph_idx = 0
        for sequence_name in self.sequence_names:
            print(f"Generating graph for {sequence_name}")

            camera_folders_prefix = osp.join(self.sequence_path_prefix, "embeddings", sequence_name, self.annotations_filename, "avg")
            camera_folders = os.listdir(camera_folders_prefix)
            
            # Loading node information, embeddings, and labels (object ids) for all cameras in the sequence
            node_df, node_embeddings, node_labels = self.setup_nodes(camera_folders_prefix, camera_folders)

            # Loading edge information, embeddings and labels for all cameras in the sequence
            edge_df, edge_idx, edge_embeddings, edge_labels = self.setup_edges(node_df, node_embeddings)

            graph = Data(x=node_embeddings, 
                        edge_index=edge_idx, 
                        y=node_labels, 
                        edge_attr=edge_embeddings, 
                        edge_labels=edge_labels)
            
            # Save each graph separately
            torch.save(graph, self.processed_file_names[graph_idx])

            # Saving log information
            node_df.to_json(self.sequence_path_prefix, "logs", sequence_name, "sequence_graph_nodes.json")
            edge_df.to_json(self.sequence_path_prefix, "logs", sequence_name, "sequence_graph_edges.json")

            # Increment graph index
            graph_idx += 1

            
    def len(self):
        """Get the number of graphs in this dataset"""
        return len(self.processed_file_names)
    
    
    def get(self, idx):
        """Get a processed graph by index.

        Parameters
        ===========
        idx : int
            Index of the graph to retrieve.

        Returns
        ===========
        data : Data
            The processed graph data.
        """
        return torch.load(osp.join(self.processed_file_names[idx]))