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