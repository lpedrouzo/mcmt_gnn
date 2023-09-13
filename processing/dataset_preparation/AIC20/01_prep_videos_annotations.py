import os
import os.path as osp
import shutil

entrypoint = "datasets/raw/AIC20"
partitions = ("train","test")

output_path_prefix = "datasets/AIC20"
output_folders = ("frames", "videos", "annotations", "embeddings")


def create_output_folder(folder_type:str) -> str:
    """Create the output folder and return the path as a string.
    """
    output_path = osp.join(output_path_prefix, folder_type, sequence_name, camera_name)
    os.makedirs(output_path)
    return output_path


def move_data_to_output_folder(filename:str, folder_type:str) -> None:
    """Moves the desired file in the raw dataset to the desired folder path
    by performing the following steps:
    1. Getting the path of the raw folder
    2. Creating the output directory
    3. Moving the folder from raw directory to the output directory (shutil)
    """
    # Get the file from the raw folder
    camera_path = osp.join(sequence_path, camera_name, filename)

    # Create the output folder
    camera_output_path = create_output_folder(folder_type)

    # Move file to output folder
    shutil.copy(src=camera_path, dst=camera_output_path)

    print(f"Moved from {camera_path} to {camera_path}")


if __name__ == "__main__":

    print("Creating the structure of the data for the project")
    for output_folder in output_folders:
        output_path =  osp.join(output_path_prefix, output_folder)
        if not osp.exists(output_path):
            os.makedirs(output_path)
            print(f"Created folder {output_path}")
        else:
            print(f"Folder {output_path} already exists")

    # Example of partitions: ("train", "test")
    for partition in partitions:

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
                        move_data_to_output_folder("vdo.avi", "videos")

                        # 2. Move annotations from raw to output folder
                        move_data_to_output_folder("mtsc/mtsc_deepsort_ssd512.txt", "annotations")

    # Example of raw video: "datasets/raw/AIC20/train/S01/c001/vdo.avi"
    # Example of output video: "datasets/AIC20/videos/S01/c001/vdo.avi"