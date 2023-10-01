import os
import os.path as osp
import shutil
import sys
sys.path.insert(1, osp.abspath('.'))

from modules.data_processor.utils import load_config

common_config, task_config = load_config("config/preprocessing.yml",
                                         "01_prep_videos_annotations")

entrypoint = task_config['original_aic_dataset_path']
partitions = task_config['dataset_partitions']

output_path_prefix = common_config['sequence_path']
output_folders = ("frames", "videos", "annotations", "embeddings")

def create_output_folder(sequence_name:str, camera_name:str, folder_type:str) -> str:
    """Create the output folder and return the path as a string.
    """
    output_path = osp.join(output_path_prefix, folder_type, sequence_name, camera_name)
    os.makedirs(output_path, exist_ok=True)
    return output_path


def move_data_to_output_folder(raw_sequence_path:str, sequence_name:str, camera_name:str, filename:str, folder_type:str) -> None:
    """Moves the desired file in the raw dataset to the desired folder path
    by performing the following steps:
    1. Getting the path of the raw folder
    2. Creating the output directory
    3. Moving the folder from raw directory to the output directory (shutil)
    """
    # Get the file from the raw folder
    camera_path = osp.join(raw_sequence_path, camera_name, filename)

    # Create the output folder
    camera_output_path = create_output_folder(sequence_name, camera_name, folder_type)

    # Move file to output folder
    shutil.copy(src=camera_path, dst=camera_output_path)

    print(f"Moved from {camera_path} to {camera_path}")


def process_partition(entrypoint, partition, video_path, preds_path=None, annotations_path=None):
    """ Creates destination folders and copies videos and annotations
    to those destination folders.
    """
    # Example of sequence name: "S01"
    sequence_path_prefix = osp.join(entrypoint, partition)
    for sequence_name in os.listdir(sequence_path_prefix):

        # Avoid hidden folders
        if not sequence_name.startswith("."):

            # Example of camera name: "c001"
            sequence_path = osp.join(sequence_path_prefix, sequence_name)
            for camera_name in os.listdir(sequence_path):

                # Avoid hidden folders
                if not camera_name.startswith("."):
                    print(sequence_path, camera_name)
                    # 1. Move video from raw to output folder
                    move_data_to_output_folder(sequence_path, sequence_name, camera_name, video_path, "videos")

                    # 2. Move annotations from raw to output folder
                    if preds_path:
                        move_data_to_output_folder(sequence_path, sequence_name, camera_name, preds_path, "annotations")

                    if annotations_path:
                        move_data_to_output_folder(sequence_path, sequence_name, camera_name, annotations_path, "annotations")
                        
if __name__ == "__main__":

    print("Creating the structure of the data for the project")
    for output_folder in output_folders:
        output_path =  osp.join(output_path_prefix, output_folder)
        if not osp.exists(output_path):
            os.makedirs(output_path)
            print(f"Created folder {output_path}")
        else:
            print(f"Folder {output_path} already exists")

    process_partition(entrypoint, "train", 
                      common_config['video_filename'], 
                      task_config['sc_train_preds_path'], 
                      task_config['annotations_path'])

    process_partition(entrypoint, "validation", 
                      common_config['video_filename'], 
                      task_config['sc_test_preds_path'], 
                      task_config['annotations_path'])
        

    # Example of raw video: "datasets/raw/AIC20/train/S01/c001/vdo.avi"
    # Example of output video: "datasets/AIC20/videos/S01/c001/vdo.avi"