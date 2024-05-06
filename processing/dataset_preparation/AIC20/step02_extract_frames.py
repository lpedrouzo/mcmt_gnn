import sys
import os
import os.path as osp
sys.path.insert(1, osp.abspath('.'))

from modules.data_processor.video_processor import VideoProcessor
from modules.data_processor.utils import load_config




def main_extract_frames(config_filepath:str="config/preprocessing.yml")->None:
    """Runs step 02: exctracts all the individual frames from each video and stores them according to the yml configuration file.

    The yml file must have the following configuration parameters:
    - `common_params` block with
        - `sequence_path`
        - `sc_preds_filename`
    - `02_extract_frames` block, with the configuration parameters for this step:
        - `sequences_to_process`
        - `video_filename`
        - `video_format`
        

    Args:
        config_filepath (str, optional): Path to yml configuration file. Defaults to "config/preprocessing.yml".
    """
    common_config, task_config = load_config(config_filepath, "02_extract_frames")

    # Instantiate the multi camera video processor
    vp = VideoProcessor(sequence_path=common_config['sequence_path'], 
                        video_filename=task_config['video_filename'],
                        video_format=task_config['video_format'])

    # Iterate over all sequences (the cameras wihtin sequence are handled by the VideoProcessor)
    for sequence in task_config['sequences_to_process']:
        vp.store_frames(sequence)


if __name__ == "__main__":
    main_extract_frames()