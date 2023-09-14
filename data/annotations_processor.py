import os.path as osp
import os
import json
import pandas as pd

class AnnotationsProcessor(object):
    """ This class is used to define pre-processing steps for the annotation files:
    - enforcing data types
    - adding column names
    - sorting by column
    - standardizing bounding box columns regardless of the dataset
    """
    def __init__(self, sequence_path, sequence_name, annotations_filename=None, delimiter=' '):
        self.sequence_path = sequence_path
        self.sequence_name = sequence_name
        self.delimiter = delimiter
        self.annotations_filename = annotations_filename

        self.annotations_path_prefix = osp.join(self.sequence_path, 'annotations', self.sequence_name)
        self.annotations = os.listdir(self.annotations_path_prefix)

    def _try_loading_logs(self, log_path):
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
        

    def apply_schema(self, schema):
        """ Using a schema specification defined as a dictionary,
        Add the column names to the dataframe and enforce the specified
        datatypes in the schema. Finally, save the resulting annotation files
        back to its original place.

        The annotations are loaded based on the sequence_path and sequence_name 
        issued in the constructor. Moreover, the annotations
        file must be in a directory that follows:

        - '<sequence_path>/'annotations'/<self.sequence_name>/<camera>/<annotations_filename>'

        if the constructor parameter annotations_filename is specified, or if not then it should be:
        
        - '<sequence_path>/'annotations'/<self.sequence_name>/<camera>.txt'

        Parameters
        ===========
        schema: dict[str, np.dtype]
            The schema definition for the annotations.

        Returns
        ===========
        None
        """
        for annotation_folder in self.annotations:

            if self.annotations_filename is not None:
                annotation_folder = osp.join(annotation_folder, self.annotations_filename)
            print(f"Procesing {annotation_folder}")

            # Loading detections file and inserting column names
            det_df = pd.read_csv(osp.join(self.annotations_path_prefix, annotation_folder), 
                                 sep=self.delimiter, 
                                 names=list(schema.keys()))
            
            # Applying the schema supplied as parameter
            det_df = det_df.astype(schema)

            # Saving detections back to their original path
            det_df.to_csv(osp.join(self.annotations_path_prefix, annotation_folder), 
                          sep=self.delimiter, index=False)
            

    def standardize_bounding_box_columns(self):
        """ Adds xmax, ymax or width, height columns to the annotations dataframe
        if not present, and saves the dataframe back to its original place.

        The annotations are loaded based on the sequence_path and sequence_name 
        issued in the constructor. Moreover, the annotations
        file must be in a directory that follows:

        - '<sequence_path>/'annotations'/<self.sequence_name>/<camera>/<annotations_filename>'

        if the constructor parameter annotations_filename is specified, or if not then it should be:
        
        - '<sequence_path>/'annotations'/<self.sequence_name>/<camera>.txt'

        Parameters
        ===========
        None

        Returns
        ===========
        None
        """
        for annotation_folder in self.annotations:
            
            log_path = osp.join(self.sequence_path, "logs", self.sequence_name, annotation_folder+'.json')
            log_data = self._try_loading_logs(log_path)

            if self.annotations_filename is not None:
                annotation_folder = osp.join(annotation_folder, self.annotations_filename)
            print(f"Procesing {annotation_folder}")

            # Loading detections file 
            det_df = pd.read_csv(osp.join(self.annotations_path_prefix, annotation_folder), 
                                 sep=self.delimiter)
            
            # Adding xmax and ymax columns if not present in dataset
            if 'xmax' not in det_df.columns or 'ymax' not in det_df.columns:
                xmax, ymax = det_df['xmin'] + det_df['width'], det_df['ymin'] + det_df['height']
                det_df['xmax'] = xmax if log_data["frame_width"] > xmax else log_data["frame_width"]
                det_df['ymax'] = ymax if log_data['frame_height'] > ymax else log_data['frame_height']

            # Adding width and height if not present in dataset
            elif 'width' not in det_df.columns or 'height' not in det_df.columns:
                xmax = det_df['xmax'] if det_df['xmax'] < log_data['frame_width'] else log_data['frame_width']
                ymax = det_df['ymax'] if det_df['ymax'] < log_data['frame_hegiht'] else log_data['frame_height']
                det_df['width'] = xmax - det_df['xmin']
                det_df['height'] = ymax - det_df['ymin']

            # Saving detections back to their original path
            det_df.to_csv(osp.join(self.annotations_path_prefix, annotation_folder), 
                          sep=self.delimiter, index=False)


    def sort_annotations_by_column(self, column):
        """ Sorts the annotation file by a given column and then saved back to
        its original directory.

        The annotations are loaded based on the sequence_path and sequence_name 
        issued in the constructor. Moreover, the annotations
        file must be in a directory that follows:

        - '<sequence_path>/'annotations'/<self.sequence_name>/<camera>/<annotations_filename>'

        if the constructor parameter annotations_filename is specified, or if not then it should be:
        
        - '<sequence_path>/'annotations'/<self.sequence_name>/<camera>.txt'

        Parameters
        ===========
        column: str or list[str]
            The specified column or columns to sort the annotations.

        Returns
        ===========
        None
        """
        for annotation_folder in self.annotations:

            if self.annotations_filename is not None:
                annotation_folder = osp.join(annotation_folder, self.annotations_filename)
            print(f"Procesing {annotation_folder}")

            # Loading detections file and sorting by column
            det_df = pd.read_csv(osp.join(self.annotations_path_prefix, annotation_folder), 
                                 sep=self.delimiter)
            det_df = det_df.sort_values(by=column)

            # Saving detections back to their original path
            det_df.to_csv(osp.join(self.annotations_path_prefix, annotation_folder), 
                          sep=self.delimiter, index=False)
