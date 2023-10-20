import os
import os.path as osp
import sys
sys.path.insert(1, osp.abspath('.'))
import pandas as pd
import json
from modules.inference.postprocessing import filter_dets_outside_frame_bounds, remove_non_roi
from modules.data_processor.utils import load_config

# Loading configuration
common_config, task_config = load_config("config/preprocessing.yml",
                                         "04_filter_sc_tracking")

if __name__ == "__main__":
    
    # Assigning variables according to the configuration definitions
    sequence_name = task_config['validation_partition']
    sequence_path_prefix = common_config['sequence_path']
    sequence_path = osp.join(sequence_path_prefix, "annotations", sequence_name)
    logs_path = osp.join(sequence_path_prefix, "logs", sequence_name)
    input_annotations_filename = task_config['in_sc_preds_filename']
    out_annotations_filename = task_config['out_sc_preds_filename']

    min_bb_area = task_config['min_bb_area']
    filter_frame_bounds = task_config['filter_frame_bounds']
    filter_roi = task_config['filter_roi']

    for camera_name in os.listdir(sequence_path):

        # Load annotation file 
        in_annotations = osp.join(sequence_path, camera_name, input_annotations_filename)
        out_annotations = osp.join(sequence_path, camera_name, out_annotations_filename)
        annotations_df = pd.read_csv(in_annotations)

        print(f"Filtering SC track {in_annotations}")

        # Load logs from script 02_extract_frames to obtain frame dimensions
        with open(osp.join(logs_path, camera_name + '.json'), 'r') as camera_logs:
            camera_log_data = json.load(camera_logs)
            frame_width = camera_log_data['frame_width']
            frame_height = camera_log_data['frame_height']

        # Filter detections that do not have a minimum area
        if min_bb_area:
            annotations_df = annotations_df[(annotations_df['width'] * annotations_df['height']) >= 750]
            print("Done filtering area")

        # Remove detections that go out of frame boundaries
        if filter_frame_bounds:
            annotations_df = filter_dets_outside_frame_bounds(annotations_df, frame_width, frame_height)
            print("Done filtering detections beyond boundaries")

        # If required, remove detections outside region of interest
        if filter_roi:
            annotations_df = remove_non_roi(sequence_path_prefix, annotations_df)
            print("Done RoI filtering of detections")

        # Save the filtered annotations
        annotations_df.to_csv(out_annotations)
        print(f"Saved processed annotation at {out_annotations}")