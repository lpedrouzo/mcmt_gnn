import os.path as osp
import os
import pandas as pd

class AnnotationsProcessor(object):
    def __init__(self, sequence_path, sequence_name, annotations_filename=None, delimiter=' '):
        self.sequence_path = sequence_path
        self.sequence_name = sequence_name
        self.delimiter = delimiter
        self.annotations_filename = annotations_filename

        self.annotations_dir = osp.join(self.sequence_path, 'annotations', self.sequence_name)
        self.annotations = os.listdir(self.annotations_dir)

    def apply_schema(self, schema):
        for annotation_path in self.annotations:

            if self.annotations_filename is not None:
                annotation_path = osp.join(annotation_path, self.annotations_filename)
            print(f"Procesing {annotation_path}")

            # Loading detections file and inserting column names
            det_df = pd.read_csv(osp.join(self.annotations_dir, annotation_path), 
                                 sep=self.delimiter, 
                                 names=list(schema.keys()))
            
            # Applying the schema supplied as parameter
            det_df = det_df.astype(schema)

            # Saving detections back to their original path
            det_df.to_csv(osp.join(self.annotations_dir, annotation_path), 
                          sep=self.delimiter, index=False)

    def sort_annotations_by_column(self, column):
        for annotation_path in self.annotations:

            if self.annotations_filename is not None:
                annotation_path = osp.join(annotation_path, self.annotations_filename)
            print(f"Procesing {annotation_path}")

            # Loading detections file and sorting by column
            det_df = pd.read_csv(osp.join(self.annotations_dir, annotation_path), 
                                 sep=self.delimiter)
            det_df = det_df.sort_values(by=column)

            # Saving detections back to their original path
            det_df.to_csv(osp.join(self.annotations_dir, annotation_path), 
                          sep=self.delimiter, index=False)
