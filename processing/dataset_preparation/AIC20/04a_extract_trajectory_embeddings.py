import sys
import os
import os.path as osp
sys.path.insert(1, osp.abspath('.'))

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from data_processor.embeddings_processor import EmbeddingsProcessor
from data_processor.utils import load_config

common_config, _ = load_config("config/processing.yml", 
                               "04a_extract_trajectory_embeddings")

if __name__ == "__main__":

    for sequence_name in common_config['sequence_names']:

        emb_proc = EmbeddingsProcessor(inference_mode=False, 
                                       sequence_path=common_config['sequence_path'], 
                                       sequence_name=sequence_name, 
                                       annotations_filename=common_config['annotations_filename'])
        emb_proc.generate_average_embeddings_single_camera()
                
        

    



