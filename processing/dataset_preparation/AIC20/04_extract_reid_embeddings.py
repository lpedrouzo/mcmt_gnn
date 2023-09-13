import sys
import os.path as osp
import os
sys.path.insert(1, osp.abspath('.'))

import torch

from models.reid.swin_transformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224
from data.embeddings_processor import EmbeddingsProcessor

# Macros related to REID model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_path = 'models/reid/st_reid_weights.pth'
model_size = 'tiny'
semantic_weight = 1.0
swin_builder = None

# Macros related to dataset
sequence_path = "datasets/AIC20/"

# Validate REID model size is valid
if model_size == 'tiny':
    swin_builder = swin_tiny_patch4_window7_224
elif model_size == 'small':
    swin_builder = swin_small_patch4_window7_224
elif model_size == 'base':
    swin_builder = swin_base_patch4_window7_224
else:
    raise Exception(f"The only three valid options are (tiny, small, base). Your input was {model_size}")

# Instantiate model and load weights (send to GPU if applicable)
swin = swin_builder(convert_weights=False, semantic_weight=semantic_weight)
swin.init_weights(model_path)
swin.to(device)
print(swin)


# Iterate over all of the sequences
for sequence_name in os.listdir(osp.join(sequence_path, "annotations")):

    # Instantiate the embeddings processor
    emb_proc = EmbeddingsProcessor(inference_mode=False, 
                                precomputed_embeddings=False,
                                frame_width=360,
                                frame_height=288,
                                img_batch_size=32,
                                img_size=(224,224),
                                cnn_model=swin,
                                sequence_path=sequence_path,
                                sequence_name="basketball"
                                )