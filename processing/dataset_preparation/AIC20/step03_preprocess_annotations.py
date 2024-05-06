import sys
import os.path as osp
sys.path.insert(1, osp.abspath('.'))

import os
import numpy as np
from modules.data_processor.annotations_processor import AnnotationsProcessor
from modules.data_processor.utils import load_config

# Data schema definition
annotations_schema = {
    'frame': np.int64, 
    'id': np.int64, 
    'xmin': np.int64, 
    'ymin': np.int64, 
    'width': np.int64, 
    'height': np.int64, 
    'lost': np.int64, 
    'occluded': np.int64, 
    'generated': np.int64, 
    'label': np.int64
}



def main_preprocess_annotations(config_filepath:str="config/preprocessing.yml")->None:
    """Runs step 03: preprocessing annotations using the configuration from the yml file.

    The yml file must have the following configuration parameters:
    - `common_params` block with
        - `sequence_path`
        - `sc_preds_filename`
    - `03_preprocess_annotations` block, with the configuration parameters for this step:
        - `sequences_to_process`
        - `sort_column_name`
        - `gt_filename`
        - `sc_preds_filename`


    Args:
        config_filepath (str, optional): Path to yml configuration file. Defaults to "config/preprocessing.yml".
    """
    common_config, task_config = load_config(config_filepath, "03_preprocess_annotations")

    # Iterate over all of the sequences
    for sequence_name in task_config['sequences_to_process']:

        if task_config['gt_filename']:
            # Instantiate the processor for the ground truth
            det_proc = AnnotationsProcessor(sequence_path=common_config['sequence_path'],
                                            sequence_name=sequence_name,
                                            annotations_filename=task_config['gt_filename'],
                                            delimiter=',')

            # The annotations are loading directly from path and stored back in the backend
            print("Applying schemas")
            det_proc.apply_schema(annotations_schema)

            print("Sorting DataFrames")
            det_proc.sort_annotations_by_column(column=task_config['sort_column_name'])

            print("Standardizing bounding box coordinates")
            det_proc.standardize_bounding_box_columns()
        
        if task_config['sc_preds_filename']:
            # Instantiate the processor for the estimated tracks
            det_proc = AnnotationsProcessor(sequence_path=common_config['sequence_path'],
                                            sequence_name=sequence_name,
                                            annotations_filename=task_config['sc_preds_filename'],
                                            delimiter=',')

            # The annotations are loading directly from path and stored back in the backend
            print("Applying schemas")
            det_proc.apply_schema(annotations_schema)

            print("Sorting DataFrames")
            det_proc.sort_annotations_by_column(column=task_config['sort_column_name'])

            print("Standardizing bounding box coordinates")
            det_proc.standardize_bounding_box_columns()

if __name__ == "__main__":
    main_preprocess_annotations()