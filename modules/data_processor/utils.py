import os.path as osp
import os
import json
import yaml

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


def get_incremental_folder(path, next_iteration_name=False):
    """ COmputes the folder name for the next Epoch based on 
    the indices of existing folders in the filesystem.

    Parameters
    ==========
    path: str
        The path where the incrementally named folders will be.
    next_iteration_name: bool
        Whether to return the folder for the next iteration or the current iteration.
        Example: if next_iteration_name == True -> return epoch_{i + 1} else epoch_{i}

    Returns
    ==========
    str
        The name of the folder for the next iteration.
    
    """
    folders = os.listdir(path)
    if len(folders):
        epoch_indices = [int(folder.split("_")[-1]) for folder in folders if folder.startswith("epoch_")]

        if len(epoch_indices):
            return f"epoch_{max(epoch_indices) + (1 if next_iteration_name else 0)}"

    return "epoch_0" if next_iteration_name else None
