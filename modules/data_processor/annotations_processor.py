import os.path as osp
import os
import json
import pandas as pd
import numpy as np
from .utils import try_loading_logs

class AnnotationsProcessor(object):
    """ This class is used to define pre-processing steps for the annotation files:
    - enforcing data types
    - adding column names
    - sorting by column
    - standardizing bounding box columns regardless of the dataset
    """
    def __init__(self, sequence_path:str, sequence_name:str=None, annotations_filename:str=None, delimiter:str=' '):
        self.sequence_path = sequence_path
        self.delimiter = delimiter
        self.annotations_filename = annotations_filename

        # If sequence name is issued, then this is a single camera processor
        if sequence_name:
            self.sequence_name = sequence_name
            self.annotations_path_prefix = osp.join(self.sequence_path, 'annotations', self.sequence_name)
            self.annotations = os.listdir(self.annotations_path_prefix)
        

    def apply_schema(self, schema:dict):
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
        In addition it also adds the camera and sequence_name as columns.

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
            log_data = try_loading_logs(log_path)

            if self.annotations_filename is not None:
                annotation_file = osp.join(annotation_folder, self.annotations_filename)
            print(f"Procesing {annotation_file}")

            # Loading detections file 
            det_df = pd.read_csv(osp.join(self.annotations_path_prefix, annotation_file), 
                                 sep=self.delimiter)
            
            # Adding xmax and ymax columns if not present in dataset
            if 'xmax' not in det_df.columns or 'ymax' not in det_df.columns:
                xmax, ymax = det_df['xmin'] + det_df['width'], det_df['ymin'] + det_df['height']
                det_df['xmax'] = np.where(xmax.values < log_data["frame_width"], xmax.values, log_data["frame_width"])
                det_df['ymax'] = np.where(ymax.values < log_data["frame_height"], ymax.values, log_data["frame_height"])

            # Adding width and height if not present in dataset
            elif 'width' not in det_df.columns or 'height' not in det_df.columns:
                xmax = np.where(det_df['xmax'].values < log_data['frame_width'], det_df['xmax'].values, log_data['frame_width'])
                ymax = np.where(det_df['ymax'].values < log_data['frame_height'], det_df['ymax'].values, log_data['frame_height'])
                det_df['width'] = xmax - det_df['xmin'].values
                det_df['height'] = ymax - det_df['ymin'].values

            # Adding camera and sequence information to the dataframe
            det_df['camera'] = annotation_folder
            det_df['sequence_name'] = self.sequence_name

            # Saving detections back to their original path
            det_df.to_csv(osp.join(self.annotations_path_prefix, annotation_file), 
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

    def consolidate_annotations(self, sequence_names:list, sort_column:str):
        """ Concatenate multiple annotation DataFrames into a single DataFrame.
        This function will effectively take all the annotations
        from all sequences, and all cameras within each sequence
        and will concatenate them.

        For this to work, each annotation file must have the columns:
        
        (frame,id,xmin,ymin,width,height,lost,occluded,generated,label,xmax,ymax,camera,sequence_name)

        Which can be generated using modules.data_processor.annotations_processor

        Parameters
        ==========
        sequence_names: list[str]
            The list of sequences to load annotations from.

        Returns
        ==========
        pd.DataFrame
            A consolidated DataFrame containing annotations from all sequences and cameras.
        """
        annotations_dfs = []

        # iterating through each sequence and each camera within the sequence
        for sequence_name in sequence_names:
            sequence_annotations_prefix = osp.join(self.sequence_path, "annotations", sequence_name)
            for camera_name in os.listdir(sequence_annotations_prefix):

                annotations_path = osp.join(sequence_annotations_prefix, 
                                            camera_name, 
                                            self.annotations_filename)
                
                annotations_dfs.append(pd.read_csv(annotations_path))
        return pd.concat(annotations_dfs, axis=0).sort_values(by=sort_column, ignore_index=True)