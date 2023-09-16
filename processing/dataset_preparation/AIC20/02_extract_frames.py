import sys
import os
import os.path as osp
sys.path.insert(1, osp.abspath('.'))

from data.video_processor import VideoProcessor


sequence_path = "datasets/AIC20/"
video_filename = "vdo.avi"

# Instantiate the multi camera video processor
vp = VideoProcessor(sequence_path=sequence_path, 
                    video_filename=video_filename,
                    video_format='.avi')

# Iterate over all sequences (the cameras wihtin sequence are handled by the VideoProcessor)
for sequence in os.listdir(os.path.join(sequence_path, "videos")):
    vp.store_frames(sequence)