import os
import os.path as osp
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from ..data_processor.utils import get_incremental_folder, get_previous_folder
from ..data_processor.embeddings_processor import EmbeddingsProcessor
from ..data_processor.annotations_processor import AnnotationsProcessor

class ObjectGraphDataset(Dataset):
    def __init__(self, sequence_path_prefix, 
                 sequence_names, 
                 annotations_filename, 
                 reid_model, 
                 num_ids_per_graph, 
                 embeddings_per_it, 
                 resized_img_shape, 
                 orignal_img_shape, 
                 temporal_threshold=None,
                 augmentation=None, 
                 frames_num_workers=2,
                 transform=None, 
                 pre_transform=None):

        super(ObjectGraphDataset, self).__init__(None, transform, pre_transform)

        self.sequence_path_prefix = sequence_path_prefix
        self.frame_dir = osp.join(self.sequence_path_prefix, "frames")
        self.sequence_names = sequence_names
        self.annotations_filename = annotations_filename
        self.num_ids_per_graph = num_ids_per_graph
        self.temporal_threshold = temporal_threshold
        self.augmentation = augmentation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # This object will help computing embeddings
        self.emb_proc = EmbeddingsProcessor(resized_img_shape[0], 
                                            resized_img_shape[1], 
                                            img_size=orignal_img_shape,
                                            img_batch_size=embeddings_per_it, 
                                            cnn_model=reid_model,
                                            num_workers=frames_num_workers)
        
        self.all_annotations_df = self.consolidate_annotations()
        self.remaining_obj_ids = self.all_annotations_df.id.unique()

        # Determine the number of possible iterations for this dataset
        len_ids = len(self.remaining_obj_ids)
        self.num_available_iterations = (len_ids//num_ids_per_graph) 
        if len_ids % num_ids_per_graph:
            self.num_available_iterations += 1


    def consolidate_annotations(self):
        """ Concatenate multiple annotation DataFrames into a single DataFrame.
        This function will effectively take all the annotations
        from all sequences, and all cameras within each sequence
        and will concatenate them.

        For this to work, each annotation file must have the columns:
        
        (frame,id,xmin,ymin,width,height,lost,occluded,generated,label,xmax,ymax,camera,sequence_name)

        Which can be generated using modules.data_processor.annotations_processor

        Parameters
        ==========
        None

        Returns
        ==========
        pd.DataFrame
            A consolidated DataFrame containing annotations from all sequences and cameras.
        """
        annotations_dfs = []

        # iterating through each sequence and each camera within the sequence
        for sequence_name in self.sequence_names:
            sequence_annotations_prefix = osp.join(self.sequence_path_prefix, "annotations", sequence_name)
            for camera_name in os.listdir(sequence_annotations_prefix):

                annotations_path = osp.join(sequence_annotations_prefix, 
                                            camera_name, 
                                            self.annotations_filename)
                
                annotations_dfs.append(pd.read_csv(annotations_path))
        return pd.concat(annotations_dfs, axis=0)
    

    def sample_obj_ids(self, det_df, unique_obj_ids, num_ids):
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
        # Getting unique IDs and taking a sample of them
        sampled_obj_ids = np.random.choice(unique_obj_ids, size=num_ids, replace=False)

        # Getting the remaining IDs for further iterations
        remaining_obj_ids = [id for id in unique_obj_ids if id not in sampled_obj_ids]

        # Return the DataFrame with only the sampled objects and the list of remaining IDs
        return det_df[det_df.id.isin(sampled_obj_ids)], remaining_obj_ids


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
        node_info = {"camera": [], "node_id": [], "object_id": [], "sequence_name": []}
        trajectory_embeddings = []
        node_id = 0
        for cam_id in np.unique(camera_ids):
            for obj_id in np.unique(det_ids):

                # Getting all the embeddings for an object obj_id in camera cam_id
                obj_cam_embeds = reid_embeds[(det_ids == obj_id) & (camera_ids == cam_id)]

                # Temporal threshold for trajectories
                if self.temporal_threshold:
                   obj_cam_embeds = obj_cam_embeds[:self.temporal_threshold]

                # Getting the average embedding for object trajectories
                mean_obj_cam_embed = torch.mean(obj_cam_embeds, dim=0)
                
                # Saving cam, obj, node, sequence indices
                node_info["camera"].append(cam_id)
                node_info["object_id"].append(obj_id)
                node_info["node_id"].append(node_id)
                node_info["sequence_name"].append(sequence_ids[(det_ids == obj_id) & (camera_ids == cam_id)][0])
                trajectory_embeddings.append(mean_obj_cam_embed)

            node_id += 1
        # Convert the list of mean trajectory embeddings into one tensor
        trajectory_embeddings = torch.stack(trajectory_embeddings, dim=0)
        node_labels = torch.tensor(node_df.object_id.values).to(self.device)

        # Saving info into a DataFrame
        node_df = pd.DataFrame(node_info)
        return node_df, trajectory_embeddings, node_labels
        
     
    def compute_edge_embeddings(self, node_embeddings, edge_idx):
        return torch.cat((
                F.pairwise_distance(node_embeddings[edge_idx.T[0]], node_embeddings[edge_idx.T[1]]).view(-1, 1),
                1 - F.cosine_similarity(node_embeddings[edge_idx.T[0]], node_embeddings[edge_idx.T[1]]).view(-1, 1)
            ), dim=1)
    

    def setup_edges(self, node_df, node_embeddings):
        """Set up edge information and embeddings efficiently.
        This function efficiently establishes edges between nodes that belong to different cameras in a DataFrame
        containing node information. For each edge, it records the source and destination node IDs, the corresponding
        object IDs, and assigns an edge label based on whether the source and destination nodes belong to the same object.

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
            - `np.repeat` is used to repeat each element in the source node IDs and object IDs arrays, allowing for the
            generation of pairs of source nodes that belong to the same camera with every destination node from different cameras.
            - `np.tile` is used to create pairs of destination node IDs and object IDs by repeating the elements from the
            different camera's nodes for each source node from the same camera.
        """
        cameras_visited = []
        edge_idx = []
        edge_labels = []

        edge_df = pd.DataFrame(columns=["src_node_id", "dst_node_id", "src_obj_id", "dst_obj_id", "edge_labels"])

        for camera in node_df.camera.unique():
            nodes_in_camera = node_df[node_df.camera == camera].reset_index(drop=True)
            nodes_not_in_camera = node_df[(node_df.camera != camera) & (~node_df.camera.isin(cameras_visited))].reset_index(drop=True)

            src_node_ids_ = nodes_in_camera["node_id"].values
            dst_node_ids_ = nodes_not_in_camera["node_id"].values
            src_obj_ids_ = nodes_in_camera["object_id"].values
            dst_obj_ids_ = nodes_not_in_camera["object_id"].values

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

            print(f"Processed edges for camera {camera}. Shapes: ", 
                  len(src_obj_ids), len(dst_obj_ids), len(src_node_ids), len(dst_node_ids))

        edge_idx = torch.tensor(edge_idx, dtype=torch.int64, device=self.device)
        edge_labels = torch.tensor(edge_labels, dtype=torch.int64, device=self.device)
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
        print("Sampling object ids")
        # Get a sample of objects from available objects in remaining_obj_ids
        sampled_df, self.remaining_obj_ids = self.sample_obj_ids(self.all_annotations_df, 
                                                                 self.remaining_obj_ids,
                                                                 self.num_ids_per_graph)
        print("Computing embeddings")
        # Compute embeddings for every detection for the chosen object ids
        results = self.emb_proc.load_embeddings_from_imgs(sampled_df, 
                                                          self.frame_dir,
                                                          fully_qualified_dir=False, 
                                                          mode="train", 
                                                          augmentation=self.augmentation,
                                                          return_imgs=False)
        
        _, node_embeds, reid_embeds, det_ids, frame_nums, sequence_ids, camera_ids = results

        print("Generating nodes")
        # Compute nodes embeddings and node info
        node_df, node_embeddings, node_labels = self.setup_nodes(reid_embeds,
                                                                 det_ids, 
                                                                 sequence_ids, 
                                                                 camera_ids)
        print("Generating edges")
        # Compute edges embeddings and info
        edge_df, edge_idx, edge_embeddings, edge_labels = self.setup_edges(node_df, node_embeddings)

        print("Creating PyG graph")
        # Generate a PyG Data object (the graph)
        graph = Data(x=node_embeddings, 
                     edge_index=edge_idx.T, 
                     y=node_labels, 
                     edge_attr=edge_embeddings, 
                     edge_labels=edge_labels)
        
        return graph, node_df, edge_df
        

        

        
