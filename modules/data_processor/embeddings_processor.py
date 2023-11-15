import torch
import os.path as osp
import os
import shutil
import numpy as np
import pandas as pd
import multiprocessing

from .utils import try_loading_logs, get_incremental_folder, get_previous_folder
from ..torch_dataset.bounding_box_dataset import BoundingBoxDataset
from ..torch_dataset.reid_embedding_dataset import REIDEmbeddingDataset
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm
from functools import partial
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class EmbeddingsProcessor(object):

    def __init__(self, 
                 img_batch_size:int=None, 
                 cnn_model=None, 
                 img_size:tuple=None, 
                 sequence_path:str=None,
                 sequence_name:str=None,
                 annotations_filename:str=None,
                 annotations_sep:str=',',
                 num_workers:int=2):
        
        self.img_batch_size = img_batch_size
        self.sequence_path = sequence_path  
        self.sequence_name = sequence_name
        self.img_size = img_size
        self.cnn_model = cnn_model
        self.num_workers = num_workers
        self.annotations_filename = annotations_filename
        self.annotations_sep = annotations_sep
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def load_embeddings_from_imgs(self, det_df, frame_dir, fully_qualified_dir, mode, augmentation=None):
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
            (bb_imgs for each det or [], torch.Tensor with shape (num_detects, node_embeddings_dim), 
            torch.Tensor with shape (num_detects, reidembeddings_dim))

        """
        ds = BoundingBoxDataset(det_df, 
                                frame_dir=frame_dir,
                                output_size=self.img_size,
                                fully_qualified_dir=fully_qualified_dir,
                                return_det_ids_and_frame=True,
                                mode=mode,
                                augmentation=augmentation)
        
        bb_loader = DataLoader(ds, 
                               batch_size=self.img_batch_size, 
                               num_workers=self.num_workers)
        
        self.cnn_model.eval()
        return self._compute_reid_data(bb_loader, 
                                       lambda bbox_img: self.cnn_model(bbox_img.to(self.device)))


    def load_embeddings_from_filesystem(self, 
                                        det_df, 
                                        frame_dir, 
                                        fully_qualified_dir, 
                                        mode):
        
        assert self.annotations_filename, "You should set the name of the annotations file before loadind embeddings from FS."
        
        ds = REIDEmbeddingDataset(det_df, 
                                  frame_dir=frame_dir, 
                                  annotations_filename=self.annotations_filename,
                                  fully_qualified_dir=fully_qualified_dir, 
                                  return_det_ids_and_emb=True, 
                                  mode=mode)
        
        embedding_loader = DataLoader(ds,
                                      batch_size=self.img_batch_size,
                                      num_workers=self.num_workers)
        
        return self._compute_reid_data(embedding_loader,
                                       lambda bbox_embedding: (bbox_embedding, bbox_embedding) )
        

    def _compute_reid_data(self, loader, reid_fn):
        node_embeds = []
        reid_embeds = []
        frame_nums = []
        det_ids = []
        sequence_ids = []
        camera_ids = []

        with torch.no_grad():
            for frame_num, det_id, camera, sequence, core_bbox_obj in tqdm(loader):

                # Compute REID embeddings, core_bbox_obj can be image patch, or embedding
                node_out, reid_out = reid_fn(core_bbox_obj)

                # Add the info from the bounding box dataset to the lists
                node_embeds.append(node_out.to(self.device))
                reid_embeds.append(reid_out.to(self.device))
                frame_nums.append(frame_num)
                det_ids.append(det_id)
                sequence_ids.append(sequence)
                camera_ids.append(camera)

        # Flatten the arrays and convert to torch.tensor
        det_ids = np.concatenate(det_ids, axis=None)
        frame_nums = np.concatenate(frame_nums, axis=None)
        sequence_ids = np.concatenate(sequence_ids, axis=None)
        camera_ids = np.concatenate(camera_ids, axis=None)

        node_embeds = torch.cat(node_embeds, dim=0)
        reid_embeds = torch.cat(reid_embeds, dim=0)

        return node_embeds, reid_embeds, det_ids, frame_nums, sequence_ids, camera_ids
        

    def store_embeddings(self, max_detections_per_df:int, mode:str='train', augmentation=None):
        """ Compute and store node and reid embeddings for all cameras in the sequence.
        This method processes frames and annotations for each camera in the sequence and computes node
        and reid embeddings for the detected objects in those frames. The embeddings are stored in separate
        directories for node and reid embeddings.

        Parameters:
        -----------
        max_detections_per_df : int
            Maximum number of detections to process at once to avoid running out of memory.
        mode: str
            'train' or 'test' modes. If mode is train, image augmentation can be applied
        augmentation: torchvision.transforms.Compose
            A list of image transformations to augment data at training time.

        Notes:
        ------
        This method processes each camera in the sequence, reading annotation data and frames, and then
        calls the '_store_embeddings_camera' method to compute and store embeddings for each camera.

        The embeddings are organized into the following directory structure:

        - sequence_path_prefix
            |- embeddings
            | |- sequence_name
            | |   |- annotations_filename
            | |   |   |- < node | reid >
            | |   |   |   |- camera_folder_1
            | |   |   |   |  |- <frame_1>.pt
            | |   |   |   |  |- <frame_2>.pt
            | |   |   |   |  |- ...
            | |   |   |   |- camera_folder_2
            | |   |   |   |  |- <frame_1>.pt
            | |   |   |   |  |- <frame_2>.pt
            | |   |   |   |  |- ...
            | |   |   |   |- ...

        Returns:
        --------
        None
        """
        assert self.cnn_model is not None, "Embeddings CNN was not properly loaded."
        print(f"Storing embeddings for sequence {self.sequence_name}")

        # Create dirs to store embeddings (node embeddings)
        node_embeds_path_prefix = osp.join(self.sequence_path, 'embeddings', self.sequence_name,
                                    'generic' if self.annotations_filename is None else self.annotations_filename, "node")
        os.makedirs(node_embeds_path_prefix, exist_ok=True)
            
        # reid embeddings
        reid_embeds_path_prefix = osp.join(self.sequence_path, 'embeddings', self.sequence_name,
                                    'generic' if self.annotations_filename is None else self.annotations_filename, "reid")
        os.makedirs(reid_embeds_path_prefix, exist_ok=True)
            
        # Frames
        frame_dir = osp.join(self.sequence_path, 'frames', self.sequence_name)
        frame_cameras = os.listdir(frame_dir)

        # Annotations
        annotations_dir_prefix = osp.join(self.sequence_path, 'annotations', self.sequence_name)

        # Logs 
        logs_dir_prefix = osp.join(self.sequence_path, 'logs', self.sequence_name)

        # For each of the cameras, compute and store embeddings
        for frame_camera in frame_cameras:

            if self.annotations_filename is not None:
                annotations_dir = osp.join(frame_camera, self.annotations_filename)
            else:
                annotations_dir = frame_camera + '.txt'

            det_df = pd.read_csv(osp.join(annotations_dir_prefix, annotations_dir), sep=self.annotations_sep)
            self._store_embeddings_camera(det_df, 
                                          osp.join(node_embeds_path_prefix, frame_camera),
                                          osp.join(reid_embeds_path_prefix, frame_camera),
                                          osp.join(frame_dir, frame_camera),
                                          osp.join(logs_dir_prefix, frame_camera + '.json'),
                                          max_detections_per_df,
                                          mode=mode,
                                          augmentation=augmentation)
        

    def _store_embeddings_camera(self, det_df, 
                                 node_embeds_path, 
                                 reid_embeds_path, 
                                 frame_dir, 
                                 log_dir, 
                                 max_detections_per_df,
                                 mode='train',
                                 augmentation=None,
                                 add_detection_id=False):
        """Store node and reid embeddings for detections in a camera.
        This method processes detections from a DataFrame, computes node and reid embeddings for each detection,
        and stores the embeddings in separate directories for node and reid embeddings.

        Parameters:
        -----------
        det_df : pandas DataFrame
            DataFrame containing detection information, including bounding boxes and frame numbers.
        node_embeds_path : str
            Path to the directory where node embeddings will be stored.
        reid_embeds_path : str
            Path to the directory where reid embeddings will be stored.
        frame_dir : str
            Directory containing frames where the detections occurred.
        log_dir : str
            Directory containing log data necessary to retrieve image width and height.
        max_detections_per_df : int
            Maximum number of detections to process at once to avoid running out of memory.
        mode: str
            'train' or 'test' modes. If mode is train, image augmentation can be applied
        augmentation: torchvision.transforms.Compose
            A list of image transformations to augment data at training time.

        Notes:
        ------
        - This method processes the detections in chunks to avoid memory issues when processing a large number
        of detections. It computes node and reid embeddings for each detection, and the embeddings are stored
        in separate directories. Detection IDs are added as the first column in the embeddings to ensure correct
        loading.

        - The input DataFrame `det_df` should have columns for frame numbers, bounding box information, and detection IDs.

        - The method uses the provided `log_dir` to retrieve image width and height information.

        Returns:
        --------
        None
        """
        if osp.exists(node_embeds_path):
            print("Found existing stored node embeddings. Deleting them and replacing them for new ones")
            shutil.rmtree(node_embeds_path)

        if osp.exists(reid_embeds_path):
            print("Found existing stored reid embeddings. Deleting them and replacing them for new ones")
            shutil.rmtree(reid_embeds_path)

        os.makedirs(node_embeds_path)
        os.makedirs(reid_embeds_path)

        # We need this log data to retrieve image width and height
        log_data = try_loading_logs(log_dir)

        # Compute and store embeddings
        # If there are more than 100k detections, we split the df into smaller dfs avoid running out of RAM, as it
        # requires storing all embedding into RAM (~6 GB for 100k detections)

        print(f"Computing embeddings for {det_df.shape[0]} detections")

        num_dets = det_df.shape[0]
        frame_cutpoints = [det_df.frame.iloc[i] for i in np.arange(0, num_dets , max_detections_per_df, dtype=int)]
        frame_cutpoints += [det_df.frame.iloc[-1] + 1]

        for frame_start, frame_end in tqdm(zip(frame_cutpoints[:-1], frame_cutpoints[1:])):
            sub_df_mask = det_df.frame.between(frame_start, frame_end - 1)
            sub_df = det_df.loc[sub_df_mask]

            # Computing embeddings from previous function
            result = self.load_embeddings_from_imgs(sub_df, frame_dir, mode, augmentation)
            node_embeds, reid_embeds, det_ids, frame_nums, _, _ = result

            if add_detection_id:
                # Add detection ids as first column of embeddings
                node_embeds = torch.cat((torch.tensor(det_ids).view(-1, 1).float().to(device), node_embeds), dim=1)
                reid_embeds = torch.cat((torch.tensor(det_ids).view(-1, 1).float().to(device), reid_embeds), dim=1)

            # Save embeddings grouped by frame
            for frame in sub_df.frame.unique():
                mask = frame_nums == frame
                frame_node_embeds = node_embeds[mask]
                frame_reid_embeds = reid_embeds[mask]

                frame_node_embeds_path = osp.join(node_embeds_path, f"{str(frame).zfill(6)}.pt")
                frame_reid_embeds_path = osp.join(reid_embeds_path, f"{str(frame).zfill(6)}.pt")

                torch.save(frame_node_embeds, frame_node_embeds_path)
                torch.save(frame_reid_embeds, frame_reid_embeds_path)

        print("Finished computing and storing embeddings")


    def _generate_average_embeddings_single_camera(self, input_path_prefix, output_path_prefix, camera_folder):
        """Processes the embeddings for each frame within the specified camera folder,
        computes the average embedding for each subject, and saves them individually in the
        output directory.

        Parameters
        ===========
        input_path_prefix : str
            The path to the input directory containing frame-wise embeddings.
        output_path_prefix : str
            The path to the output directory where average embeddings will be saved.
        camera_folder : str
            The name of the camera folder being processed.
        
        Returns
        ===========
        None
        """
        camera_dir_prefix_frame = osp.join(input_path_prefix, camera_folder)
        camera_dir_prefix_subject = osp.join(output_path_prefix, camera_folder)
        os.makedirs(camera_dir_prefix_subject)

        subject_sum_dict = {}
        subject_count_dict = {}
        
        for frame in os.listdir(camera_dir_prefix_frame):

            # Load embeddings at current frame
            embeddings_frame = torch.load(osp.join(camera_dir_prefix_frame, frame))

            # Iterate through the frames at the current index and sum the embeddings by subject
            for embedding in embeddings_frame:
                subject_id = int(embedding[0].cpu())

                if subject_id not in subject_sum_dict:
                    subject_sum_dict[subject_id] = embedding[1:]
                    subject_count_dict[subject_id] = 1
                else:
                    subject_sum_dict[subject_id] += embedding[1:]
                    subject_count_dict[subject_id] += 1

        # For every subject, take the sum embedding and divide by count, then save to filesystem
        for subject_id in subject_count_dict:
            avg_subject_embed = (subject_sum_dict[subject_id] / subject_count_dict[subject_id]).cpu()
            torch.save(avg_subject_embed, osp.join(camera_dir_prefix_subject, 'obj_' + str(subject_id) + '.pt'))


    def generate_average_embeddings_single_camera(self, remove_past_iteration_data=True):
        """Process and generate average embeddings for all camera folders in the sequence.
        This method processes all camera folders in the specified sequence, calculates the average
        embeddings for subjects within each camera folder, and saves them in the designated output
        directory. Multiprocessing to parallelize the processing of camera folders is used.

        The input directory structure should follow the format:
        - sequence_path_prefix
            |- embeddings
            | |- sequence_name
            | |   |- annotations_filename
            | |   |   |- epoch_<iteration>
            | |   |   |   |- node <----- IMPORTANT
            | |   |   |   |  |- camera_folder_1
            | |   |   |   |  |  |- <frame_1>.pt
            | |   |   |   |  |  |- <frame_2>.pt
            | |   |   |   |  |  |- ...
            | |   |   |   |  |- camera_folder_2
            | |   |   |   |  |  |- <frame_1>.pt
            | |   |   |   |  |  |- <frame_2>.pt
            | |   |   |   |  |  |- ...
            | |   |   |   |  |- ...

        The output directory structure will be organized as follows:
        - sequence_path_prefix
          |- embeddings
          | |- sequence_name
          | |   |- annotations_filename
          | |   |   |- epoch_{iteration}
          | |   |   |   |- avg <---- IMPORTANT
          | |   |   |   |  |- camera_folder_1
          | |   |   |   |  |  |- obj_<subject_id_1>.pt
          | |   |   |   |  |  |- obj_<subject_id_2>.pt
          | |   |   |   |  |  |- ...
          | |   |   |   |  |- camera_folder_2
          | |   |   |   |  |  |- obj_<subject_id_1>.pt
          | |   |   |   |  |  |- obj_<subject_id_2>.pt
          | |   |   |   |  |  |- ...
          | |   |   |   |  |- ...

        If the output directory already exists, it will be deleted and recreated to store new average
        embeddings.

        Parameters
        ============
        remove_past_iteration_data: boolean
            If true, comptued emebddings from past iteration will be removed, to save space.

        Returns
        ============
        None
        """
        sequence_path_prefix_ = osp.join(self.sequence_path, "embeddings", self.sequence_name, self.annotations_filename)

        if get_incremental_folder(sequence_path_prefix_, next_iteration_name=False):
            sequence_path_prefix = osp.join(sequence_path_prefix_, get_incremental_folder(sequence_path_prefix_, next_iteration_name=False))
        else:
            raise Exception(f"There are no embeddings stored in {sequence_path_prefix_}")
        
        # Remove data from past iteration if needed.
        if remove_past_iteration_data and get_previous_folder(sequence_path_prefix_):
            shutil.rmtree(osp.join(sequence_path_prefix_, get_previous_folder(sequence_path_prefix_)))

        frames_sequence_path_prefix = osp.join(sequence_path_prefix, "node")
        subjects_sequence_path_prefix = osp.join(sequence_path_prefix, "avg")

        camera_folders = os.listdir(frames_sequence_path_prefix)
        partial_generate_function = partial(self._generate_average_embeddings_single_camera, 
                                            frames_sequence_path_prefix, 
                                            subjects_sequence_path_prefix)
        
        # Use multiprocessing to parallelize processing camera folders
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        pool.map(partial_generate_function, camera_folders)
        pool.close()
        pool.join()