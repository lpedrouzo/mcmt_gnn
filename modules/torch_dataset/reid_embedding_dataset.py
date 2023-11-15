import numpy as np
import pandas as pd
import os.path as osp
import torch
from torch.utils.data import Dataset

class REIDEmbeddingDataset(Dataset):
    """
    Class used to process REID embeddings. Given a DataFrame (det_df) with detections of a MOT sequence, it returns
    the REID embedding corresponding to the detection's bounding box coordinates.
    """
    def __init__(self, det_df:pd.DataFrame, 
                 frame_dir:str,
                 annotations_filename=None,
                 fully_qualified_dir=True,
                 return_det_ids_and_emb:bool=False,
                 mode='train'):
        
        # Initialization of constructor variables
        self.det_df = det_df
        self.frame_dir = frame_dir
        self.annotations_filename = annotations_filename
        self.fully_qualified_dir = fully_qualified_dir

        self.curr_emb = None
        self.curr_emb_num = None
        self.curr_camera = None
        self.curr_sequence = None

        self.return_det_ids_and_emb = return_det_ids_and_emb

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
        Then, load a REID embedding using torch that corresponds 
        to the detection defined in the annotation's row.

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
        if [row['frame'], row['camera'], row['sequence_name']] != [self.curr_emb_num, self.curr_camera, self.curr_sequence]:
            
            # If fully qualified, then the path must contain sequence name and camera name
            if self.fully_qualified_dir:
                frame_name = osp.join(self.frame_dir, 
                                      str(row['frame']).zfill(6) + ".pt")
            else:
                frame_name = osp.join(self.frame_dir, 
                                      str(row['sequence_name']), 
                                      self.annotations_filename,
                                      "node",
                                      str(row['camera']),
                                      str(row['frame']).zfill(6) + ".pt")
            # Load the image
            self.curr_emb = torch.load(frame_name)

            if self.curr_emb is None:
                raise Exception(f"Img '{frame_name}' could not be loaded.")
            
            self.curr_emb_num = row['frame']
            self.curr_camera = row['camera']
            self.curr_sequence = row['sequence_name']

        reid_emb = self.curr_emb

        if self.return_det_ids_and_emb:
            return row['frame'], row['id'], row['camera'], row['sequence_name'], reid_emb
        else:
            return reid_emb