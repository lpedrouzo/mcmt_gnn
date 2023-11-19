import sys
import os
import os.path as osp
sys.path.insert(1, osp.abspath('.'))

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from modules.data_processor.embeddings_processor import EmbeddingsProcessor
from modules.data_processor.utils import load_config

common_config, task_config = load_config("config/preprocessing.yml", 
                               "05a_extract_trajectory_embeddings")

if __name__ == "__main__":

    print("Extracting embeddings for training sequences")
    for sequence_name in task_config['train_sequences']:
        print(f"Working on {sequence_name}")

        emb_proc = EmbeddingsProcessor(sequence_path=common_config['sequence_path'], 
                                       sequence_name=sequence_name, 
                                       annotations_filename=task_config['annotations_filename'],
                                       device='cpu')
        emb_proc.generate_average_embeddings_single_camera()
    
    print("Extracting embeddings for testing sequences")
    for sequence_name in task_config['test_sequences']:
        print(f"Working on {sequence_name}")
        
        emb_proc = EmbeddingsProcessor(sequence_path=common_config['sequence_path'], 
                                       sequence_name=sequence_name, 
                                       annotations_filename=task_config['annotations_filename'],
                                       device='cpu')
        emb_proc.generate_average_embeddings_single_camera()