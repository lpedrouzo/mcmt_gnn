import os.path as osp
import numpy as np
import pandas as pd
import cv2

def remove_duplicated_detections(df):
    """Remove repetition to ensure that all objects are unique for every frame.

    Parameters
    ==========
    df : pandas.DataFrame
        Data that should be filtered

    Returns
    =========
    df : pandas.DataFrame
        Filtered data that all objects are unique for every frame.
    """
    df = df.drop_duplicates(subset=['camera', 'id', 'frame'], keep='first')
    return df


def filter_dets_outside_frame_bounds(det_df, frame_width, frame_height, boundary_percentage=0.1):
    """Filter bounding boxes outside frame boundaries.
    This function filters the rows in the input DataFrame (det_df) based on whether
    the bounding boxes are within the specified frame boundaries. Bounding boxes that
    fall outside the frame boundaries (with a given margin) are removed from the output.

    Parameters
    ==========
    det_df : pandas.DataFrame
        DataFrame containing bounding box information, including 'xmin', 'ymin',
        'width', and 'height' columns.

    frame_width : int
        The width of the frame.

    frame_height : int
        The height of the frame.

    boundary_percentage : float, optional
        Margin as a percentage of the frame dimensions (default is 0.1, representing
        a 10% margin).

    Returns
    ==========
    pandas.DataFrame
        A filtered DataFrame with only the rows containing bounding boxes within the
        specified frame boundaries.

    Example
    ==========
    >>> filtered_df = filter_dets_outside_frame_bounds(bounding_boxes_df, 1920, 1080)
    """
    # Check if bounding box x-axis is inside the frame with a margin
    cond1 = (frame_width * boundary_percentage <= det_df['xmin'] + (det_df['width'] / 2))
    cond2 = det_df['xmin'] + (det_df['width'] / 2) <= frame_width - (frame_width * boundary_percentage)
    idx_x = np.logical_and(cond1, cond2)

    # Check if bounding box y-axis is inside the frame with a margin
    cond1 = (frame_height * boundary_percentage <= det_df['ymin'] + det_df['height'])
    cond2 = det_df['ymin'] + det_df['height'] <= frame_height - (frame_height * boundary_percentage)
    idx_h = np.logical_and(cond1, cond2)

    # Combine both conditions to filter the DataFrame
    idx = np.logical_and(idx_x, idx_h)

    return det_df[idx]

def load_roi(sequence_path, camera_name, sequence_name=None):
    """ Load Region of Interest binary maps for each
    camera.

    Parameters
    ==========
    sequence_path: str
        The path to the processed dataset.
        Example: mcmt_gnn/datasets/AIC22
    camera_name: str
        The camera name. Ex: c001
    sequence_name: str
        The name of the sequence. Ex: S01
    """
    # If sequence_name is not provided, we assume that it is embedded in sequence_path
    if sequence_name:
        roi_dir = osp.join(sequence_path, "roi", sequence_name, camera_name, 'roi.jpg')
    else:
        roi_dir = osp.join(sequence_path, camera_name, 'roi.jpg')

    if not osp.exists(roi_dir):
        raise ValueError(f"Missing ROI image for camera {camera_name}")
    
    # Load grayscale image and convert to binary image
    binary_image = cv2.imread(roi_dir, cv2.IMREAD_GRAYSCALE)
    binary_image = (binary_image == 255)

    # Transpose if scene is in portrait
    if binary_image.shape[0] > binary_image.shape[1]:
        binary_image = binary_image.T

    return binary_image
    

def is_outside_roi(row, roi):
    """ Check if detection is inside region of interest.
    Return False if not.

    Parameter
    =========
    row: 
        The dataframe row that represents a detection and must have
        (xmin, xmax, ymin, ymax) columns.
    roi: np.array (w,h)
        The region of interest image as a binary map.
    
    Returns
    =========
    boolean
        True if detection is inside region of interest.
        False otherwise.
    """
    height = roi.shape[0]
    width = roi.shape[1]

    # If out of bounds
    if (row['xmin'] > width or row['ymin'] > height or
        row['xmin'] < 0 or row['ymin'] < 0):
        return True
    
    y_bottom = min(row['ymin'] + int(row['height']//2), 1079)
    x_bottom = min(row['xmin'] + int(row['width']//2), 1919)

    # roi[row['ymin'], row['xmin']] == true means it is inside roi 
    return not roi[y_bottom, x_bottom]


def remove_non_roi(sequence_path, data_df):
    """ Remove detections that correspond to areas
    outside of the Region of Interest (RoI).

    Parameters
    ==========
    data_df: pd.DataFrame
        The predicitons dataframe.

    Returns
    ==========
    pd.DataFrame
        The predictions dataframe with non-roi detections
        eliminated.
    """
    # Initialize outlier column as all false
    data_df['outlier'] = False

    # Check if detections are outlier and mark them
    for i, row in data_df.iterrows():
        roi = load_roi(sequence_path, row['camera'], row['sequence_name'])
        if is_outside_roi(row, roi):
            data_df.loc[i, 'outlier'] = True

    # Remove outlier detections
    return data_df[~data_df['outlier']].drop(columns=['outlier'])