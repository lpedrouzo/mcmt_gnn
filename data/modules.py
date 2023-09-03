import torch.nn as nn
import torch
import numpy as np
from models.reid.swin_transformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224
from torchvision import transforms

class SwinReidModel(object):
    def __init__(self, weights_path:str, device:str, swin_size:str='tiny', semantic_weight:float=1.0):
        super().__init__(self)

        self.device = device

        # Initialize Swin transformer based on size
        if swin_size == 'tiny':
            self.cnn_model = swin_tiny_patch4_window7_224(convert_weights=False, 
                                                          semantic_weight=semantic_weight)
        elif swin_size == 'small':
            self.cnn_model = swin_small_patch4_window7_224(convert_weights=False, 
                                                           semantic_weight=semantic_weight)
        else:
            self.cnn_model = swin_base_patch4_window7_224(convert_weights=False, 
                                                           semantic_weight=semantic_weight)
        
        # Load weights for model
        self.cnn_model.init_weights(weights_path)
        self.cnn_model.to(device)
    
        # Define transformations
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __call__(self, batch):
        batch = self.normalize(batch)
        self.cnn_model(batch.to(self.device))

    def predict(self, img):
        img = self.normalize(img)
        node_embeddings, _ = self.cnn_model(img.unsqueeze(0).to(self.device))
        return node_embeddings, node_embeddings
    