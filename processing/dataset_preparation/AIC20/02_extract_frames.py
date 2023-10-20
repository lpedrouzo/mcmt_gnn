import sys
import os
import os.path as osp
sys.path.insert(1, osp.abspath('.'))

from modules.data_processor.video_processor import VideoProcessor
from modules.data_processor.utils import load_config

common_config, task_config = load_config("config/preprocessing.yml",
                                         "02_extract_frames")


if __name__ == "__main__":
    # Instantiate the multi camera video processor
    vp = VideoProcessor(sequence_path=common_config['sequence_path'], 
                        video_filename=task_config['video_filename'],
                        video_format=task_config['video_format'])

    # Iterate over all sequences (the cameras wihtin sequence are handled by the VideoProcessor)
    for sequence in task_config['sequences_to_process']:
        vp.store_frames(sequence)