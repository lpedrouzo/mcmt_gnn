import numpy as np
import cv2
import pandas as pd
import os.path as osp
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class BoundingBoxDataset(Dataset):
    """
    Class used to process detections. Given a DataFrame (det_df) with detections of a MOT sequence, it returns
    the image patch corresponding to the detection's bounding box coordinates
    """
    def __init__(self, det_df:pd.DataFrame, 
                 frame_height:int, 
                 frame_width:int, 
                 frame_dir:str,
                 fully_qualified_dir=True,
                 output_size:tuple = (128, 64),
                 return_det_ids_and_frame:bool = False,
                 mode='train',
                 augmentation=None):
        
        # Initialization of constructor variables
        self.det_df = det_df
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_dir = frame_dir
        self.fully_qualified_dir = fully_qualified_dir
        
        transform_list = [Resize(output_size), 
                          ToTensor(), 
                          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        
        if augmentation and mode == 'train':
            transform_list.append(augmentation)
        
        self.transforms = Compose(transform_list)

        # Initialize two variables containing the path and img of the frame that is being loaded to avoid loading multiple
        # times for boxes in the same image
        self.curr_img = None
        self.curr_frame_num = None
        self.curr_camera = None
        self.curr_sequence = None

        self.return_det_ids_and_frame = return_det_ids_and_frame

    def __len__(self):
        """ Get the size of the dataset that is defined as the number of rows
        in the annotations dataframe.

        Parameters
        ===========
        None

        Returns
        ===========
        dataset_len: int
            The length of the dataset
        """
        return self.det_df.shape[0]

    def __getitem__(self, ix):
        """ Get a single annotations row using index ix.
        Then, loads an image using opencv that corresponds to the detection defined in the annotation's row,
        and performs the transformations defined in the constructor.

        Parameters
        ===========
        ix: int
            A torch.dataset managed index.
        
        Returns
        ===========
        row_frame: int
            The frame id corresponding to the image patch.
        row_id: int
            The id of the tracked subject corresponding to the image patch.
        bb_img: torch.tensor
            The image patch corresponding to the bounding box detection.
        """
        row = self.det_df.iloc[ix]

        # Load this bounding box' frame img, in case we haven't done it yet
        if [row['frame'], row['camera'], row['sequence_name']] != [self.curr_frame_num, self.curr_camera, self.curr_sequence]:
            
            # If fully qualified, then the path must contain sequence name and camera name
            if self.fully_qualified_dir:
                frame_name = osp.join(self.frame_dir, 
                                      str(row['frame']).zfill(6) + ".jpg")
            else:
                frame_name = osp.join(self.frame_dir, 
                                      str(row['sequence_name']), 
                                      str(row['camera']),
                                      str(row['frame']).zfill(6) + ".jpg")
            # Load the image
            self.curr_img = cv2.imread(frame_name)

            if self.curr_img is None:
                raise Exception(f"Img '{frame_name}' could not be loaded.")
            
            self.curr_frame_num = row['frame']
            self.curr_camera = row['camera']
            self.curr_sequence = row['sequence_name']

        frame_img = self.curr_img

        # Crop the bounding box, and pad it if necessary to
        bb_img = frame_img[int(max(0, row['ymin'])): int(max(0, row['ymax'])),
                   int(max(0, row['xmin'])): int(max(0, row['xmax']))]
        
        # Apply transformations
        bb_img = Image.fromarray(bb_img)
        if self.transforms is not None:
            bb_img = self.transforms(bb_img)

        if self.return_det_ids_and_frame:
            return row['frame'], row['id'], row['camera'], row['sequence_name'], bb_img
        else:
            return bb_img