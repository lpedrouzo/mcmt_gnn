import os.path as osp
import os
import json

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