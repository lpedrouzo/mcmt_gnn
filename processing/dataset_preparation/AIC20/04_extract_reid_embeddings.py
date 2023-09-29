import sys
import os.path as osp
import os
sys.path.insert(1, osp.abspath('.'))

import torch
import yaml
from models.reid.resnet import resnet101_ibn_a
from models.reid.swin_transformer import load_swin_encoder
from data_processor.embeddings_processor import EmbeddingsProcessor
from data_processor.utils import load_config

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

common_config, task_config = load_config("config/processing.yml", 
                                         "04_extract_reid_embeddings")


if __name__ == '__main__':
    # Instantiate model and load weights (send to GPU if applicable)
    if task_config['model_type'] == 'resnet':
        model = resnet101_ibn_a(model_path=task_config['model_path'], device=device)
    elif task_config['model_type'] == 'swin':
        model = load_swin_encoder(model_size=task_config['model_size'], model_path=task_config['model_path'], device=device)


    # Iterate over all of the sequences
    for sequence_name in common_config['sequence_names']:

        # Instantiate the embeddings processor
        emb_proc = EmbeddingsProcessor(inference_mode=False, 
                                    precomputed_embeddings=False,
                                    frame_width=task_config['original_img_size'][0],
                                    frame_height=task_config['original_img_size'][1],
                                    img_batch_size=task_config['img_batch_size'],
                                    img_size=task_config['cnn_img_size'],
                                    cnn_model=model,
                                    sequence_path=common_config['sequence_path'],
                                    sequence_name=sequence_name,
                                    annotations_filename=common_config['annotations_filename']
                                    )
        
        emb_proc.store_embeddings(max_detections_per_df=task_config['max_detections_per_df'])