import os
import cv2
import time
import numpy as np
import pandas as pd
import os.path as osp
import json

class VideoProcessor(object):
    """
    This object is used to load videos and store individual frames on the filesystem.
    """
    def __init__(self, 
                 sequence_path:str,
                 video_filename:str=None,
                 video_format:str='.avi'):
        """
        Object constructor.

        Parameters
        ==========
        sequence_path: str
            The path to the EFPL dataset (example: 'datasets/EFPL')
        video_format: str
            The format of the video (exmaple: '.avi')
        """
        self.sequence_path = sequence_path
        self.video_format = video_format
        self.log_records = []
        self.video_filename = video_filename


    def _log_video_metadata(self, log_dir, single_camera_video, info_dict):
        """ Generate metadata logs about the video with:
        - Image width
        - Image Height
        - frames per second
        - Total number of frames
        - Total time cost of extraction

        Parameters
        ===========
        log_dir: str
            The directory prefix where the log file is going to be.
        single_camera_video: str
            The name of the log filename. It should follow the name of the camera for a given
            sequence.
        info_dict: dict[str, Any]
            The video metadata.

        Returns
        ===========
        None
        """
        os.makedirs(log_dir, exist_ok=True)
        with open(osp.join(log_dir, single_camera_video+'.json'), 'w') as json_file:
            json.dump(info_dict, json_file)
                  
    def store_frames(self, sequence_name) -> None:
        """
        For a list of videos, each representing a different camera in the same scene, 
        Loads the video using OpenCV, then saves all the individual frames on a folder inside 
        the original data path.

        The resulting frame path will follow:

        '<sequence_path>/frames/<sequence_name>/<camera>.<video_format>

        if <video_format> is passed to the constructor and <camera> is a file, not a folder. Else:
        
        '<sequence_path>/frames/<sequence_name>/<camera>/<video_filename>'

        If <video_filename> is passed to the constructor and <camera is a folder, not a file.

        Parameters
        ===========
        sequence_name: str
            The name of the EFPL sequence (ex: "terrace")
        
        Returns
        ==========
        None
        """
        # Define the path where the videos from multiple cameras are
        video_dir = osp.join(self.sequence_path, "videos", sequence_name)
        frame_dir = osp.join(self.sequence_path, "frames", sequence_name)
        log_dir = osp.join(self.sequence_path, "logs", sequence_name)

        cameras_videos = os.listdir(video_dir)

        # Iterate over all cameras in the sequence and store images
        for single_camera_video in cameras_videos:
            
            # Work the paths if the camera names are folders that store fixed filenames or if they are the videos themselves
            if self.video_filename is not None:
                single_camera_video_dir = osp.join(single_camera_video, self.video_filename)
                frame_folder = single_camera_video
            else:
                single_camera_video_dir = single_camera_video
                frame_folder = single_camera_video.replace(self.video_format, '')

            tStart = time.time()
            print('Processing ' + single_camera_video)

            # Load the video using OpenCV
            video = cv2.VideoCapture(osp.join(video_dir, single_camera_video_dir))

            # Define and create the output path 
            output_dir = osp.join(frame_dir, frame_folder)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_counter = 0

            # Read video file and save image frames
            while video.isOpened():

                ret, frame = video.read()
                frame_name = osp.join(output_dir, str(frame_counter).zfill(6) + ".jpg")
                frame_counter += 1

                if not ret:
                    print("End of video file.")
                    break
                
                cv2.imwrite(frame_name, frame)

            tEnd = time.time()

            # Log basic metadata about the video
            self._log_video_metadata(
                log_dir,
                frame_folder,
                {
                    "frame_width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "frame_height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": video.get(cv2.CAP_PROP_FPS),
                    "num_frames": video.get(cv2.CAP_PROP_FRAME_COUNT),
                    "frame_extraction_time": tEnd - tStart
                }
            )

            print("Frame extraction took %f sec" % (tEnd - tStart))

        