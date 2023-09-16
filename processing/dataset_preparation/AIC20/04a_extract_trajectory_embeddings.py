import sys
import os
import os.path as osp
sys.path.insert(1, osp.abspath('.'))

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from data.embeddings_processor import EmbeddingsProcessor

sequence_path = "datasets/AIC20"
sequence_names = ("S01", "S03", "S04")
annotations_filename = 'gt.txt'

if __name__ == "__main__":

    for sequence_name in sequence_names:

        emb_proc = EmbeddingsProcessor(inference_mode=False, 
                                       sequence_path=sequence_path, 
                                       sequence_name=sequence_name, 
                                       annotations_filename=annotations_filename)

        
        emb_proc.generate_average_embeddings_single_camera()
                
        

    



