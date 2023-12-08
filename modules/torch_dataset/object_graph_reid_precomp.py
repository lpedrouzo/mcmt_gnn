import os
import os.path as osp
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from .utils import simple_negative_sampling, precompute_samples_for_graph_id

class ObjectGraphREIDPrecompDataset(Dataset):
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
                 trajectory_folder_name='avg'):

        self.sequence_path_prefix = sequence_path_prefix
        self.sequence_names = sequence_names
        self.annotations_filename = annotations_filename
        self.num_ids_per_graph = num_ids_per_graph
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.return_dataframes = return_dataframes
        self.negative_links_ratio = negative_links_ratio
        self.graph_transform = graph_transform
        self.trajectory_folder_name = trajectory_folder_name
        super(ObjectGraphREIDPrecompDataset, self).__init__(None, None, None)

        self.process_initial_nodes()


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
        node_df = pd.DataFrame({
            "camera": pd.Series(dtype="str"), 
            "object_id": pd.Series(dtype="int"), 
            "embeddings_path": pd.Series(dtype="str")
            })

        # Recording all objects on all cameras
        for camera_folder in camera_folders:
            #print(f"Processing nodes for camera {camera_folder}")

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
        #print(f"Length of node_df {len(node_df)}")

        return node_df, node_embeddings, node_labels


    def setup_edges(self, node_df):
        """Set up edge information and embeddings efficiently.
        This function efficiently establishes edges between nodes that 
        belong to different cameras in a DataFrame containing node information. 
        For each edge, it records the source and destination node IDs, 
        the corresponding object IDs, and assigns an edge label based on whether
        the source and destination nodes belong to the same object.

        Parameters
        ===========
        node_df: pd.DataFrame
            DataFrame containing node information with 
            columns=["camera", "node_id", "object_id"].

        Returns
        ==========
        edge_df: pd.DataFrame
            DataFrame with edge information including 
            columns=["src_node_id", "dst_node_id", "src_obj_id", "dst_obj_id"].
        edge_idx: torch.Tensor
            Edge indices as a torch tensor.
        edge_embeddings: torch.Tensor
            Edge embeddings computed based on the provided node embeddings.
        edge_labels: torch.Tensor
            Edge labels (0 or 1) indicating whether source and 
            destination nodes belong to the same object.

        Notes:
            - `np.repeat` is used to repeat each element in the source node IDs 
            and object IDs arrays, allowing for the generation of pairs of source 
            nodes that belong to the same camera with every destination node from different cameras.
            - `np.tile` is used to create pairs of destination node IDs and object 
            IDs by repeating the elements from the different camera's nodes 
            for each source node from the same camera.
        """
        cameras_visited = []
        edge_idx = []
        edge_labels = []

        edge_df = pd.DataFrame(columns=["src_node_id", "dst_node_id", "src_obj_id", "dst_obj_id", "edge_labels"])

        for camera in node_df.camera.unique():
            nodes_in_camera = node_df[node_df.camera == camera].reset_index(drop=True)
            nodes_not_in_camera = node_df[
                (node_df.camera != camera) & (~node_df.camera.isin(cameras_visited))
                ].reset_index(drop=True)

            src_node_ids_ = nodes_in_camera["node_id"].values
            dst_node_ids_ = nodes_not_in_camera["node_id"].values
            src_obj_ids_ = nodes_in_camera["object_id"].values
            dst_obj_ids_ = nodes_not_in_camera["object_id"].values

            # See description of repeat and tile in docstring above
            src_node_ids = np.repeat(src_node_ids_, len(dst_node_ids_))
            dst_node_ids = np.tile(dst_node_ids_, len(src_node_ids_))
            src_obj_ids = np.repeat(src_obj_ids_, len(dst_obj_ids_))
            dst_obj_ids = np.tile(dst_obj_ids_, len(src_obj_ids_))
            edge_label = (src_obj_ids == dst_obj_ids).astype(int)

            edge_df = pd.concat([edge_df, pd.DataFrame({
                "src_node_id": src_node_ids,
                "dst_node_id": dst_node_ids,
                "src_obj_id": src_obj_ids,
                "dst_obj_id": dst_obj_ids,
                "edge_labels": edge_label
            })], ignore_index=True)

            edge_idx.extend(np.column_stack((src_node_ids, dst_node_ids)))
            edge_labels.extend(edge_label)

            # Save visited cameras so we only get pairs of edges only once (Edge (1, 2) is equal to Edge (2,1))
            cameras_visited.append(camera)

        edge_idx = np.array(edge_idx, dtype=np.int64)
        edge_labels = np.array(edge_labels, dtype=bool)

        # Sample negative edges according to ratio if there is one
        if self.negative_links_ratio:
            edge_df, edge_idx, edge_labels = simple_negative_sampling(edge_df, edge_idx, edge_labels, self.negative_links_ratio)
        
        # Convert edge indices and labels to torch tensors and compute embeddings
        # For edge_idx we take transpose as the GNN needs edges with shape (2, num_edges)
        edge_idx = torch.tensor(np.array(edge_idx), dtype=torch.int64, device=self.device).T
        edge_labels = torch.tensor(np.array(edge_labels), dtype=torch.int64, device=self.device)   

        return edge_df, edge_idx, edge_labels

    def compute_edge_embeddings(self, node_embeddings, edge_idx):
        """ Compute edge embeddings using l2 and cosine distances between
        pairs of node embeddings given by edges.
        """
        return torch.cat((
                F.pairwise_distance(node_embeddings[edge_idx[0]], 
                                    node_embeddings[edge_idx[1]]).view(-1, 1),
                1 - F.cosine_similarity(node_embeddings[edge_idx[0]], 
                                        node_embeddings[edge_idx[1]]).view(-1, 1)
            ), dim=1).to(self.device)


    def process_initial_nodes(self):
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
        node_dfs, node_embeddings_list, node_labels_list = [], [], []

        for sequence_name in self.sequence_names:
            #print(f"Generating nodes for {sequence_name}")

            # Set the path prefix with the correct epoch folder
            camera_folders_prefix = osp.join(self.sequence_path_prefix, 
                                             "embeddings", 
                                             sequence_name, 
                                             self.annotations_filename, self.trajectory_folder_name)
            
            camera_folders = os.listdir(camera_folders_prefix)
            
            # Loading node information, embeddings, and labels (object ids) for all cameras in the sequence
            node_df, node_embeddings, node_labels = self.setup_nodes(camera_folders_prefix, camera_folders)

            node_dfs.append(node_df)
            node_embeddings_list.append(node_embeddings)
            node_labels_list.append(node_labels)

        # Generating a single set of dfs and embeddings for all sequences
        self.node_df = pd.concat(node_dfs, ignore_index=True)
        self.node_embeddings = torch.cat(node_embeddings_list, dim=0)
        self.node_labels = torch.cat(node_labels_list, dim=0)

        self.initial_object_id_list = self.node_df.object_id.unique()
        len_ids = len(self.initial_object_id_list)

        # If num_ids_per_graph is 0 or less, then load all the objects at once
        self.num_ids_per_graph = self.num_ids_per_graph if self.num_ids_per_graph > 0 else len_ids

        # Calculate the number of possible iterations in this dataset
        self.num_available_iterations = (len_ids//self.num_ids_per_graph) 
        if len_ids % self.num_ids_per_graph:
            self.num_available_iterations += 1

        self.on_epoch_end()
    
    def on_epoch_end(self):
        # Precompute the object ids to enable graph ids by the dataset
        self.precomputed_id_samples = precompute_samples_for_graph_id(
            self.initial_object_id_list,
            self.num_ids_per_graph,
            self.num_available_iterations
            )
    
        
    def len(self):
        """Determine the number of available steps in this dataset"""
        return self.num_available_iterations
    
    
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
        edge_embeddings = self.compute_edge_embeddings(sampled_node_embeddings, edge_idx)

        #print("Creating PyG graph")
        graph = Data(x=sampled_node_embeddings, 
                     edge_index=edge_idx, 
                     y=sampled_node_labels, 
                     edge_attr=edge_embeddings, 
                     edge_labels=edge_labels)
        
        # Do transformation if defined
        if self.graph_transform:
            graph = self.graph_transform(graph)

        if self.return_dataframes:
            return graph.to(self.device), sampled_node_df, edge_df
        else:
            return graph.to(self.device)