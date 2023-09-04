import os
import cv2
import time
import numpy as np
import pandas as pd
import os.path as osp
import json

class VideoProcessor(object):
    """
    This object is used to load videos and store indivual frames on filesystem.
    """
    def __init__(self, 
                 sequence_path:str,
                 video_format:str='.avi'):
        """
        Constructor.

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

    def _log_video_metadata(self, log_path, info_dict):
        with open(log_path, 'w') as json_file:
            json.dump(info_dict, json_file)
                  
    def store_frames(self, sequence_name) -> None:
        """
        For a list of videos, each representing a different camera in the same scene, 
        Loads the video using OpenCV, then saves all the individual frames on a folder inside 
        the original data path

        Parameters
        ==========
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

            tStart = time.time()
            print('Processing ' + single_camera_video)

            # Load the video using OpenCV
            video = cv2.VideoCapture(osp.join(video_dir, single_camera_video))

            # Define the output path 
            output_dir = osp.join(frame_dir, 
                                  single_camera_video.replace(self.video_format, ''))
            
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
                osp.join(log_dir, single_camera_video.replace(self.video_format, '.json')),
                self.log_records.extend({
                    "frame_width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "frame_height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": video.get(cv2.CAP_PROP_FPS),
                    "num_frames": video.get(cv2.CAP_PROP_FRAME_COUNT),
                    "frame_extraction_time": tEnd - tStart
                })
            )

            print("Frame extraction took %f sec" % (tEnd - tStart))

        