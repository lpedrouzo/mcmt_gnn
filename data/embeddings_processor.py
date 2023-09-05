import torch
import os.path as osp
import os
import shutil
import numpy as np
import pandas as pd

from .bounding_box_dataset import BoundingBoxDataset
from torch.utils.data import DataLoader
from time import time

COL_NAMES_EPFL = ('id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated','label')

class EmbeddingsProcessor(object):

    def __init__(self, 
                 inference_mode:str,
                 precomputed_embeddings:bool, 
                 frame_width:int, 
                 frame_height:int, 
                 img_batch_size:int=None, 
                 cnn_model=None, 
                 img_size:tuple=None, 
                 sequence_path:str=None,
                 sequence_name:str=None,
                 num_workers:int=2):
        
        self.inference_mode = inference_mode
        self.precomputed_embeddings = precomputed_embeddings
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.img_batch_size = img_batch_size
        self.sequence_path = sequence_path 
        self.sequence_name = sequence_name
        self.img_size = img_size
        self.cnn_model = cnn_model
        self.num_workers = num_workers

    def load_appearance_data(self, det_df, node_embeds_path, reid_embeds_path):
        """
        Loads embeddings for node features and reid.
        Returns:
            tuple with (reid embeddings, node_feats), both are torch.tensors with shape (num_nodes, embed_dim)
        """
        if self.inference_mode and not self.precomputed_embeddings:
            assert self.cnn_model is not None
            
            # Compute embeddings from scatch using CNN
            _, node_feats, reid_embeds = self._load_embeddings_from_imgs(det_df=det_df,
                                                                        cnn_model=self.cnn_model,
                                                                        return_imgs=False,
                                                                        use_cuda=self.inference_mode)
        else:
            # Load node and reid embeddings from filesystem
            reid_embeds = self._load_precomputed_embeddings(det_df=det_df,
                                                            embeddings_dir=node_embeds_path,
                                                            use_cuda=self.inference_mode)
      
            node_feats = self._load_precomputed_embeddings(det_df=det_df,
                                                           embeddings_dir=reid_embeds_path,
                                                           use_cuda=self.inference_mode)
        return reid_embeds, node_feats


    def _load_embeddings_from_imgs(self, det_df, return_imgs = False, use_cuda=True):
        """
        Computes embeddings for each detection in det_df with a CNN.
        Args:
            det_df: pd.DataFrame with detection coordinates
            dataset_params: A python dictionary with the following keys
                - frame_width
                - frame_height
                - img_size
                - img_batch_size
            cnn_model: CNN to compute embeddings with. It needs to return BOTH node embeddings and reid embeddings
            return_imgs: bool, determines whether RGB images must also be returned
        Returns:
            (bb_imgs for each det or [], torch.Tensor with shape (num_detects, node_embeddings_dim), torch.Tensor with shape (num_detects, reidembeddings_dim))

        """
        device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

        ds = BoundingBoxDataset(det_df, 
                                frame_width=self.frame_width, 
                                frame_height=self.frame_height, 
                                output_size=self.img_size,
                                return_det_ids_and_frame=False)
        
        bb_loader = DataLoader(ds, 
                               batch_size=self.img_batch_size, 
                               pin_memory=True, num_workers=self.num_workers)
        
        cnn_model = self.cnn_model.eval()

        bb_imgs = []
        node_embeds = []
        reid_embeds = []
        with torch.no_grad():
            for bboxes in bb_loader:
                node_out, reid_out = cnn_model(bboxes.cuda())
                node_embeds.append(node_out.to(device))
                reid_embeds.append(reid_out.to(device))

                if return_imgs:
                    bb_imgs.append(bboxes)

        node_embeds = torch.cat(node_embeds, dim=0)
        reid_embeds = torch.cat(reid_embeds, dim=0)

        return bb_imgs, node_embeds, reid_embeds

    def _load_precomputed_embeddings(self, det_df, embeddings_dir, use_cuda):
        """
        Given a sequence's detections, it loads from disk embeddings that have already been computed and stored for its
        detections
        Args:
            det_df: pd.DataFrame with detection coordinates
            seq_info_dict: dict with sequence meta info (we need frame dims)
            embeddings_dir: name of the directory where embeddings are stored

        Returns:
            torch.Tensor with shape (num_detects, embeddings_dim)

        """
        assert self.sequence_name is not None and self.sequence_path is not None, \
               "Make sure both sequence_name and sequence_path are defined"+ \
               "in the constructor before calling load_precomputed_embeddings()"
        assert embeddings_dir is not None, "You need to define an embedings directory to load embeddings."
        
        # Retrieve the embeddings we need from their corresponding locations
        embeddings_path = osp.join(self.sequence_path, 'embeddings', self.sequence_name, 
                                   'generic_detector', embeddings_dir)
        print("Defined embeddings path is ", embeddings_path)

        frames_to_retrieve = sorted(det_df.frame.unique())
        embeddings_list = [torch.load(osp.join(embeddings_path, f"{frame_num}.pt")) for frame_num in frames_to_retrieve]
        embeddings = torch.cat(embeddings_list, dim=0)

        # First column in embeddings is the index. Drop the rows of those that are not present in det_df
        ixs_to_drop = list(set(embeddings[:, 0].int().numpy()) - set(det_df['id']))
        embeddings = embeddings[~np.isin(embeddings[:, 0], ixs_to_drop)]  # Not so clean, but faster than a join

        assert_str = "Problems loading embeddings. Indices between query and stored embeddings do not match. BOTH SHOULD BE SORTED!"
        assert (embeddings[:, 0].numpy() == det_df['id'].values).all(), assert_str

        embeddings = embeddings[:, 1:]  # Get rid of the detection index

        return embeddings.to(torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"))


    def store_embeddings(self):

        assert self.cnn_model is not None, "Embeddings CNN was not properly loaded."
        print(f"Storing embeddings for sequence {self.sequence_name}")

        # Create dirs to store embeddings (node embeddings)
        node_embeds_path = osp.join(self.sequence_path, 'embeddings', self.sequence_name,
                                    'generic_detector', 'node')

        # reid embeddings
        reid_embeds_path = osp.join(self.sequence_path, 'embeddings', self.sequence_name,
                                    'generic_detector', 'reid')
        
        # Frames
        frame_dir = osp.join(self.sequence_path, 'frames', self.sequence_name)
        frame_cameras = os.listdir(frame_dir)

        # Annotations
        annotations_dir = osp.join(self.sequence_path, 'annotations', self.sequence_name)

        # For each of the cameras, compute and store embeddings
        for frame_camera in frame_cameras:
            det_df = pd.read_csv(osp.join(annotations_dir, frame_camera + '.txt'), sep=' ', names=COL_NAMES_EPFL)
            self._store_embeddings_camera(det_df, 
                                          osp.join(node_embeds_path, frame_camera),
                                          osp.join(reid_embeds_path, frame_camera),
                                          osp.join(frame_dir, frame_camera))
        
    def _store_embeddings_camera(self, det_df, node_embeds_path, reid_embeds_path, frame_dir):

        if osp.exists(node_embeds_path):
            print("Found existing stored node embeddings. Deleting them and replacing them for new ones")
            shutil.rmtree(node_embeds_path)

        if osp.exists(reid_embeds_path):
            print("Found existing stored reid embeddings. Deleting them and replacing them for new ones")
            shutil.rmtree(reid_embeds_path)

        os.makedirs(node_embeds_path)
        os.makedirs(reid_embeds_path)

        # Compute and store embeddings
        # If there are more than 100k detections, we split the df into smaller dfs avoid running out of RAM, as it
        # requires storing all embedding into RAM (~6 GB for 100k detections)

        print(f"Computing embeddings for {det_df.shape[0]} detections")

        num_dets = det_df.shape[0]
        max_dets_per_df = int(1e4) # Needs to be larger than the maximum amount of dets possible to have in one frame

        frame_cutpoints = [det_df.frame.iloc[i] for i in np.arange(0, num_dets , max_dets_per_df, dtype=int)]
        frame_cutpoints += [det_df.frame.iloc[-1] + 1]

        for frame_start, frame_end in zip(frame_cutpoints[:-1], frame_cutpoints[1:]):
            sub_df_mask = det_df.frame.between(frame_start, frame_end - 1)
            sub_df = det_df.loc[sub_df_mask]

            print(sub_df.frame.min(), sub_df.frame.max())
            bbox_dataset = BoundingBoxDataset(sub_df, 
                                            frame_dir=frame_dir,
                                            frame_width=self.frame_width, 
                                            frame_height=self.frame_height, 
                                            output_size=self.img_size,
                                            return_det_ids_and_frame=True)
            
            bbox_loader = DataLoader(bbox_dataset, 
                                    batch_size=self.img_batch_size, 
                                    pin_memory=True,
                                    num_workers=self.num_workers)

            # Feed all bboxes to the CNN to obtain node and reid embeddings
            self.cnn_model.eval()
            node_embeds, reid_embeds = [], []
            frame_nums, det_ids = [], []
            
            with torch.no_grad():
                for frame_num, det_id, bboxes in bbox_loader:
                    node_out, reid_out = self.cnn_model(bboxes.cuda())
                    node_embeds.append(node_out.cpu())
                    reid_embeds.append(reid_out.cpu())
                    frame_nums.append(frame_num)
                    det_ids.append(det_id)

            det_ids = torch.cat(det_ids, dim=0)
            frame_nums = torch.cat(frame_nums, dim=0)

            node_embeds = torch.cat(node_embeds, dim=0)
            reid_embeds = torch.cat(reid_embeds, dim=0)

            # Add detection ids as first column of embeddings, to ensure that embeddings are loaded correctly
            node_embeds = torch.cat((det_ids.view(-1, 1).float(), node_embeds), dim=1)
            reid_embeds = torch.cat((det_ids.view(-1, 1).float(), reid_embeds), dim=1)

            # Save embeddings grouped by frame
            for frame in sub_df.frame.unique():
                mask = frame_nums == frame
                frame_node_embeds = node_embeds[mask]
                frame_reid_embeds = reid_embeds[mask]

                frame_node_embeds_path = osp.join(node_embeds_path, f"{frame}.pt")
                frame_reid_embeds_path = osp.join(reid_embeds_path, f"{frame}.pt")

                torch.save(frame_node_embeds, frame_node_embeds_path)
                torch.save(frame_reid_embeds, frame_reid_embeds_path)

        print("Finished computing and storing embeddings")