import sys
import os.path as osp
sys.path.insert(1, osp.abspath('.'))

import os
import numpy as np

from data.annotations_processor import AnnotationsProcessor


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

sequence_path = "datasets/AIC20/"
annotations_filename = "gt.txt"


# Iterate over all of the sequences
for sequence_name in os.listdir(osp.join(sequence_path, "annotations")):

    # Instantiate the processor for a given sequence
    det_proc = AnnotationsProcessor(sequence_path=sequence_path,
                                    sequence_name=sequence_name,
                                    annotations_filename=annotations_filename,
                                    delimiter=',')

    # The annotations are loading directly from path and stored back in the backend
    print("Applying schemas")
    #det_proc.apply_schema(annotations_schema)

    print("Sorting DataFrames")
    #det_proc.sort_annotations_by_column(column='frame')

    print("Standardizing bounding box coordinates")
    det_proc.standardize_bounding_box_columns()
    