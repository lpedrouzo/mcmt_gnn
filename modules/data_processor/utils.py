import os.path as osp
import os
import json
import yaml
import torch
import numpy as np

def try_loading_logs(log_path:str):
    """ Loads a JSON file with saved video metadata from the frame extraction process
    in 02_extract_frames.py

    Parameters
    ===========
    log_path: str
        The path to the JSON file

    Returns
    ===========
    json_data: dict[str, Any]
        The video metadata.
    """
    if os.path.exists(log_path):
        with open(log_path, 'r') as file:
            return json.load(file)
    else:
        raise Exception("No Logfile Found. Most likely you have not used the VideoProcessor yet.")
    

def load_config(config_path:str, task_name:str):
    """ Loads configuration parameters for the processing tasks (processing folder).

    Parameters
    ==========
    config_path: str
        The path of the YAML files where the task configuration is defined.
    task_name: str
        The name of the task. By convention, it should be equal 
        to the name of the corresponding python file.
        Example: 04_extract_reid_embeddings
    
    Returns
    ==========
    common_config: dict[str, Any]
        The configuration parameters common to all tasks
    task_config: dict[str, Any]
        The configuration parameters that are specific to the task
    
    """
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
        common_config = config['common_params'] # Common for all processing tasks
        if task_name in config:
            task_config = config[task_name] # Task-specific params
        else:
            task_config = None
    return common_config, task_config


def check_nans_df(df, message):
    """ Check if there are NaN values in the DataFrame
    """
    has_nan = df.isna().any().any()

    if has_nan:
        raise Exception(f"The DataFrame has NaN values. {message}")

def check_nans_tensor(tnsr, message):
    """ Check for NaN values in torch.tensor
    """
    has_nan = torch.isnan(tnsr).any().item()

    if has_nan:
        raise Exception(f"The tensor has NaN values. {message}")


def generate_samples_for_galleries(annotations_df, frames_per_gallery):
    """ Generate a sample of frame numbers for each vehicle ID
    in the annotations dataframe to be used in the galleries.
    
    Parameters
    ==========
    annotations_df: pd.DataFrame
        A dataframe with the single camera annotations. Must have the columns
        ("frame", "id")
    frames_per_gallery: int
        The number of frames to have in each gallery.
    
    Returns
    ==========
    frame_samples_per_id: Dict
        The keys correspond to the vehicle IDs and the values
        correspond to a list of frame ids (samples) per vehicle.
    """
    frame_samples_per_id = {}
    # Get a random sample of frames per subject id
    unique_ids = annotations_df["id"].unique()
    for id in unique_ids:
        frame_idx_in_id = annotations_df.loc[annotations_df.id == id, "frame"]
        if len(frame_idx_in_id) > frames_per_gallery:
            # sample without replacement 
            frame_sample_per_id = annotations_df.loc[annotations_df.id == id, "frame"]\
                                                .sample(frames_per_gallery, replace=False).to_list()
        else:
            # Get the whole set of IDs and if neccesary, sample the remaining IDs
            frame_sample_per_id = annotations_df.loc[annotations_df.id == id, "frame"].to_list()

        # Finally save those samples on a dictionary
        frame_samples_per_id[id] = frame_sample_per_id
    return frame_samples_per_id