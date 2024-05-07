import sys
import os.path as osp
import os
sys.path.insert(1, osp.abspath('.'))

import torch
import yaml
import torchvision.transforms as transforms
from models.reid.resnet import resnet101_ibn_a
from models.reid.swin_transformer import load_swin_encoder
from modules.data_processor.embeddings_processor import EmbeddingsProcessor
from modules.data_processor.utils import load_config
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def main_extract_reid_embeddings(config_filepath:str="config/preprocessing.yml")->None:
    """Runs step 05: extracting ReID embeddings of every detection in the dataset, according to the yml configuration file

    The yml file must have the following configuration parameters:
    - `common_params` block with
        - `sequence_path`
        - `sc_preds_filename`
    - `05_extract_reid_embeddings` block, with the configuration parameters for this step:
        - `max_detections_per_df`
        - `model_type`
        - `model_path`
        - `annotations_filename`
        - `train_sequences`
        - `test_sequences`        
        - `cnn_img_size`
        - `img_batch_size`
        - `augmentation`
        - `add_detection_id`


    Args:
        config_filepath (str, optional): Path to yml configuration file. Defaults to "config/preprocessing.yml".
    """
    common_config, task_config = load_config(config_filepath, "05_extract_reid_embeddings")

    # Instantiate model and load weights (send to GPU if applicable)
    if task_config['model_type'] == 'resnet':
        model = resnet101_ibn_a(model_path=task_config['model_path'], device=device)
    elif task_config['model_type'] == 'swin':
        model = load_swin_encoder(model_size=task_config['model_size'], model_path=task_config['model_path'], device=device)


    augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomErasing(scale=(0.02, 0.15)),
            transforms.RandomRotation(degrees=0.15),
            transforms.RandomPerspective(distortion_scale=0.25)
        ])

    print("Extracting embeddings for training sequences")
    # Iterate over all of the sequences
    for sequence_name in task_config['train_sequences']:
        print(f"Working on {sequence_name}")

        # Instantiate the embeddings processor for train sequences
        emb_proc = EmbeddingsProcessor(img_batch_size=task_config['img_batch_size'],
                                       img_size=task_config['cnn_img_size'],
                                       cnn_model=model,
                                       sequence_path=common_config['sequence_path'],
                                       sequence_name=sequence_name,
                                       annotations_filename=task_config['annotations_filename']
                                    )
        
        emb_proc.store_embeddings(max_detections_per_df=task_config['max_detections_per_df'], 
                                  mode='train', 
                                  augmentation=augmentation if task_config["augmentation"] else None,
                                  add_detection_id=task_config["add_detection_id"])

    print("Extracting embeddings for testing sequences")
    for sequence_name in task_config['test_sequences']:
        print(f"Working on {sequence_name}")

         # Instantiate the embeddings processor for test sequences
        emb_proc = EmbeddingsProcessor(img_batch_size=task_config['img_batch_size'],
                                       img_size=task_config['cnn_img_size'],
                                       cnn_model=model,
                                       sequence_path=common_config['sequence_path'],
                                       sequence_name=sequence_name,
                                       annotations_filename=task_config['annotations_filename'])
        
        emb_proc.store_embeddings(max_detections_per_df=task_config['max_detections_per_df'], 
                                  mode='test',
                                  add_detection_id=task_config['add_detection_id'])


if __name__ == '__main__':
    main_extract_reid_embeddings()