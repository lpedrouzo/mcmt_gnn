import os
import os.path as osp
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.utils.negative_sampling import negative_sampling
from ..data_processor.embeddings_processor import EmbeddingsProcessor
from ..data_processor.utils import check_nans_df, check_nans_tensor
from .utils import simple_negative_sampling, get_n_rows_per_group
np.random.seed(11)

class ObjectGraphDataset(Dataset):
    def __init__(self, data_df,
                 sequence_path_prefix, 
                 reid_model, 
                 num_ids_per_graph, 
                 embeddings_per_it, 
                 resized_img_shape, 
                 frames_per_vehicle_cam=False,
                 augmentation=None, 
                 normalize_embedding=False,
                 frames_num_workers=2,
                 return_dataframes=True,
                 negative_sampling=False,
                 num_neg_samples=None,
                 graph_transform=None):

        super(ObjectGraphDataset, self).__init__(None, None, None)

        self.all_annotations_df = data_df
        self.sequence_path_prefix = sequence_path_prefix
        self.frame_dir = osp.join(self.sequence_path_prefix, "frames")
        self.frames_per_vehicle_cam = frames_per_vehicle_cam
        self.normalize_embedding = normalize_embedding
        self.augmentation = augmentation
        self.return_dataframes = return_dataframes
        self.initial_object_id_list = self.all_annotations_df.id.unique()
        self.negative_sampling = negative_sampling
        self.num_neg_samples = num_neg_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph_transform = graph_transform

        # This object will help computing embeddings
        self.emb_proc = EmbeddingsProcessor(img_size=resized_img_shape,
                                            img_batch_size=embeddings_per_it, 
                                            cnn_model=reid_model,
                                            num_workers=frames_num_workers)

        # Determine the number of possible iterations for this dataset
        len_ids = len(self.initial_object_id_list)

        # If num_ids_per_graph is 0 or less, then load all the objects at once
        self.num_ids_per_graph = num_ids_per_graph if num_ids_per_graph > 0 else len_ids

        # Calculate the number of possible iterations in this dataset
        self.num_available_iterations = (len_ids//self.num_ids_per_graph) 
        if len_ids % num_ids_per_graph:
            self.num_available_iterations += 1
        
        # Precompute the object ids to enable graph ids by the dataset
        self.precomputed_id_samples = self.precompute_samples_for_graph_id(
            self.initial_object_id_list,
            self.num_ids_per_graph,
            self.num_available_iterations
            )
        
    def precompute_samples_for_graph_id(self, initial_id_list, num_ids_per_graph, num_iterations):
        """ Computes

        Parameters
        ==========
        None

        Returns 
        ==========
        None
        """
        print("Sampling object ids")
        precomputed_id_samples = []
        remaining_obj_ids = initial_id_list

        for _ in range(num_iterations):
            sampled_ids, remaining_obj_ids = self.sample_obj_ids(remaining_obj_ids,
                                                                 num_ids_per_graph)
            precomputed_id_samples.append(sampled_ids)
        return precomputed_id_samples

    def sample_obj_ids(self, unique_obj_ids, num_ids):
        """ Sample a specified number of unique object IDs from a list of unique IDs.

        Parameters
        ----------
        self : object
            An instance of the class.
        det_df : pd.DataFrame
            A DataFrame containing detection information.
        unique_obj_ids : list
            A list of unique object IDs.
        num_ids : int
            The number of unique object IDs to sample.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing only the sampled object IDs.
        list
            A list of remaining unique object IDs after sampling.
        """
        if len(unique_obj_ids) > num_ids:
            # Getting unique IDs and taking a sample of them
            sampled_obj_ids = np.random.choice(unique_obj_ids, size=num_ids, replace=False)

            # Getting the remaining IDs for further iterations
            remaining_obj_ids = [id for id in unique_obj_ids if id not in sampled_obj_ids]
        else:
            sampled_obj_ids = unique_obj_ids
            remaining_obj_ids = []
            
        # Return the DataFrame with only the sampled objects and the list of remaining IDs
        return sampled_obj_ids, remaining_obj_ids


    def setup_nodes(self, reid_embeds, det_ids, sequence_ids, camera_ids):
        """ Setup node information for object trajectories.
        This function processes object embeddings, detection IDs, sequence IDs, and camera IDs
        to prepare node information for object trajectories. It computes mean trajectory embeddings
        for objects observed in different cameras and sequences, assigns unique node IDs to each
        object trajectory, and retrieves relevant information such as camera, object, and sequence indices.

        Parameters
        ----------
        self : object
            An instance of the class.
        reid_embeds : torch.Tensor
            Tensor containing object embeddings.
        det_ids : torch.Tensor
            Tensor containing detection IDs.
        sequence_ids : torch.Tensor
            Tensor containing sequence IDs.
        camera_ids : torch.Tensor
            Tensor containing camera IDs.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing node information, including camera, object, node, and sequence indices.
        torch.Tensor
            Tensor containing mean trajectory embeddings for objects.
        torch.Tensor
            Tensor containing node labels based on object IDs.
        """
        #check_nans_tensor(reid_embeds, "reid_embeds")

        node_info = {"camera": [], "node_id": [], "object_id": [], "sequence_name": []}
        trajectory_embeddings = []
        node_id = 0
        for cam_id in np.unique(camera_ids):
            for obj_id in np.unique(det_ids):

                if ((det_ids == obj_id) & (camera_ids == cam_id)).any():

                    # Getting all the embeddings for an object obj_id in camera cam_id
                    obj_cam_embeds = reid_embeds[(det_ids == obj_id) & (camera_ids == cam_id)]

                    # Getting the average embedding for object trajectories
                    mean_obj_cam_embed = torch.mean(obj_cam_embeds, dim=0)
                    
                    # Saving cam, obj, node, sequence indices
                    node_info["camera"].append(cam_id)
                    node_info["object_id"].append(obj_id)
                    node_info["node_id"].append(node_id)
                    node_info["sequence_name"].append(sequence_ids[(det_ids == obj_id) & (camera_ids == cam_id)])
                    trajectory_embeddings.append(mean_obj_cam_embed)

                    node_id += 1

        # Saving info into a DataFrame
        node_df = pd.DataFrame(node_info)

        # Convert the list of mean trajectory embeddings into one tensor
        trajectory_embeddings = torch.stack(trajectory_embeddings, dim=0)
        node_labels = torch.tensor(node_df.object_id.values).to(self.device)

        # Sanity checks
        #check_nans_df(node_df, "node_df")
        #check_nans_tensor(trajectory_embeddings, "trajectory_embeddings")
        #check_nans_tensor(node_labels, "node_labels")

        return node_df, trajectory_embeddings, node_labels
        
     
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


    def setup_edges(self, node_df, node_embeddings):
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
        node_embeddings: torch.Tensor
            Node embeddings.

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

        if self.negative_sampling:
            edge_df, edge_idx, edge_labels = simple_negative_sampling(edge_df, edge_idx, edge_labels, self.num_neg_samples)
        
        # Convert edge indices and labels to torch tensors and compute embeddings
        # For edge_idx we take transpose as the GNN needs edges with shape (2, num_edges)
        edge_idx = torch.tensor(np.array(edge_idx), dtype=torch.int64, device=self.device).T
        edge_labels = torch.tensor(np.array(edge_labels), dtype=torch.int64, device=self.device)   
        edge_embeddings = self.compute_edge_embeddings(node_embeddings, edge_idx)

        return edge_df, edge_idx, edge_embeddings, edge_labels
    
    
    def len(self):
        """Determine the number of available steps in this dataset"""
        return self.num_available_iterations
    

    def get(self, idx):
        """ Get a sample graph data object.
        It samples objects from the remaining object IDs, computes embeddings 
        for the chosen objects' detections, and constructs the graph with nodes 
        and edges, including node and edge embeddings and labels.

        Parameters
        ----------
        self : object
            An instance of the class.
        idx : int
            The index of the sample graph. This variable is ommited.
            self.remaining_obj_ids is used to build the graphs. 

        Returns
        -------
        Data
            A PyTorch Geometric (PyG) Data object representing the sample graph.
        """
        # Retrieve object id sample using idx and filter detection dataset
        id_sample = self.precomputed_id_samples[idx]
        sampled_df = self.all_annotations_df[self.all_annotations_df.id.isin(id_sample)]

        # If required, do not get the whole trajectories, get frames_per_vehicle_cam rows instead
        if self.frames_per_vehicle_cam:
            sampled_df = sampled_df.groupby(["id", "camera"]).apply(get_n_rows_per_group, 
                                                                    self.frames_per_vehicle_cam)\
                                                             .reset_index(drop=True)
            
        check_nans_df(sampled_df, "sampled_df")
        
        print("Computing embeddings")
        # Compute embeddings for every detection for the chosen object ids
        results = self.emb_proc.load_embeddings_from_imgs(sampled_df, 
                                                          self.frame_dir,
                                                          fully_qualified_dir=False, 
                                                          mode="train", 
                                                          augmentation=self.augmentation)
        
        node_embeds, reid_embeds, det_ids, frame_nums, sequence_ids, camera_ids = results

        print("Generating nodes")
        # Compute nodes embeddings and node info
        node_df, node_embeddings, node_labels = self.setup_nodes(reid_embeds,
                                                                 det_ids, 
                                                                 sequence_ids, 
                                                                 camera_ids)
        
        if self.normalize_embedding:
            node_embeddings = F.normalize(node_embeddings, p=2, dim=0)

        print("Generating edges")
        # Compute edges embeddings and info
        edge_df, edge_idx, edge_embeddings, edge_labels = self.setup_edges(node_df, node_embeddings)

        print("Creating PyG graph")
        # Generate a PyG Data object (the graph)
        graph = Data(x=node_embeddings, 
                     edge_index=edge_idx, 
                     y=node_labels, 
                     edge_attr=edge_embeddings, 
                     edge_labels=edge_labels)
        
        # Do transformation if defined
        if self.graph_transform:
            graph = self.graph_transform(graph)

        if self.return_dataframes:
            return graph.to(self.device), node_df, edge_df, sampled_df
        else:
            return graph.to(self.device)



class ObjectGraphREIDPrecompDataset(ObjectGraphDataset):
    """ Pytorch Geometric Dataset definition that inherits from
    ObjectGraphDataset defined aboce. This dataset has the same functionality
    as its parent class, but it loads the REID embeddings from the filesystem instead
    of computing them while building the graph.
    """
    def __init__(self, data_df,
                 sequence_path_prefix, 
                 annotations_filename,
                 num_ids_per_graph, 
                 embeddings_per_it, 
                 frames_per_vehicle_cam=False,
                 normalize_embedding=False,
                 frames_num_workers=2,
                 return_dataframes=True,
                 negative_sampling=False,
                 num_neg_samples=None,
                 graph_transform=None):

        # Call the constructor of the grandparent class (PyG Dataset)
        super(ObjectGraphDataset, self).__init__(None, None, None)

        self.all_annotations_df = data_df
        self.sequence_path_prefix = sequence_path_prefix
        self.annotations_filename = annotations_filename
        self.frame_dir = osp.join(self.sequence_path_prefix, "embeddings")
        self.frames_per_vehicle_cam = frames_per_vehicle_cam
        self.normalize_embedding = normalize_embedding
        self.return_dataframes = return_dataframes
        self.initial_object_id_list = self.all_annotations_df.id.unique()
        self.negative_sampling = negative_sampling
        self.num_neg_samples = num_neg_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph_transform = graph_transform

        # This object will help computing embeddings
        self.emb_proc = EmbeddingsProcessor(img_batch_size=embeddings_per_it,
                                            annotations_filename=annotations_filename,
                                            num_workers=frames_num_workers)

        # Determine the number of possible iterations for this dataset
        len_ids = len(self.initial_object_id_list)

        # If num_ids_per_graph is 0 or less, then load all the objects at once
        self.num_ids_per_graph = num_ids_per_graph if num_ids_per_graph > 0 else len_ids

        # Calculate the number of possible iterations in this dataset
        self.num_available_iterations = (len_ids//self.num_ids_per_graph) 
        if len_ids % num_ids_per_graph:
            self.num_available_iterations += 1
        
        # Precompute the object ids to enable graph ids by the dataset
        self.precomputed_id_samples = self.precompute_samples_for_graph_id(
            self.initial_object_id_list,
            self.num_ids_per_graph,
            self.num_available_iterations
            )
        
    def get(self, idx):
        """ Get a sample graph data object.
        It samples objects from the remaining object IDs, computes embeddings 
        for the chosen objects' detections, and constructs the graph with nodes 
        and edges, including node and edge embeddings and labels.

        Parameters
        ==========
        self : object
            An instance of the class.
        idx : int
            The index of the sample graph. This variable is ommited.
            self.remaining_obj_ids is used to build the graphs. 

        Returns
        =========
        Data
            A PyTorch Geometric (PyG) Data object representing the sample graph.
        """
        # Retrieve object id sample using idx and filter detection dataset
        id_sample = self.precomputed_id_samples[idx]
        sampled_df = self.all_annotations_df[self.all_annotations_df.id.isin(id_sample)]

        # If required, do not get the whole trajectories, get frames_per_vehicle_cam rows instead
        if self.frames_per_vehicle_cam:
            sampled_df = sampled_df.groupby(["id", "camera"]).apply(get_n_rows_per_group, 
                                                                    self.frames_per_vehicle_cam)\
                                                             .reset_index(drop=True)
            
        check_nans_df(sampled_df, "sampled_df")
        
        print("Computing embeddings")
        # Compute embeddings for every detection for the chosen object ids
        results = self.emb_proc.load_embeddings_from_filesystem(sampled_df, 
                                                                self.frame_dir,
                                                                fully_qualified_dir=False, 
                                                                mode="train")
        
        node_embeds, reid_embeds, det_ids, frame_nums, sequence_ids, camera_ids = results

        print("Generating nodes")
        # Compute nodes embeddings and node info
        node_df, node_embeddings, node_labels = self.setup_nodes(reid_embeds,
                                                                 det_ids, 
                                                                 sequence_ids, 
                                                                 camera_ids)
        
        if self.normalize_embedding:
            node_embeddings = F.normalize(node_embeddings, p=2, dim=0)

        print("Generating edges")
        # Compute edges embeddings and info
        edge_df, edge_idx, edge_embeddings, edge_labels = self.setup_edges(node_df, node_embeddings)

        print("Creating PyG graph")
        # Generate a PyG Data object (the graph)
        graph = Data(x=node_embeddings, 
                     edge_index=edge_idx, 
                     y=node_labels, 
                     edge_attr=edge_embeddings, 
                     edge_labels=edge_labels)
        
        # Do transformation if defined
        if self.graph_transform:
            graph = self.graph_transform(graph)

        if self.return_dataframes:
            return graph.to(self.device), node_df, edge_df, sampled_df
        else:
            return graph.to(self.device)
