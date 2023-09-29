import sys
import os
import os.path as osp
sys.path.insert(1, osp.abspath('.'))

from data_processor.video_processor import VideoProcessor
from data_processor.utils import load_config

common_config, _ = load_config("config/processing.yml",
                                         "02_extract_frames")


if __name__ == "__main__":
    # Instantiate the multi camera video processor
    vp = VideoProcessor(sequence_path=common_config['sequence_path'], 
                        video_filename=common_config['video_filename'],
                        video_format=common_config['video_format'])

    # Iterate over all sequences (the cameras wihtin sequence are handled by the VideoProcessor)
    for sequence in common_config['sequence_names']:
        vp.store_frames(sequence)