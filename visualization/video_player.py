import cv2
import os
import os.path as osp
import pandas as pd

class VideoPlayer:
    def __init__(self, frames_dir, bbox_dir, annotations_filename, video_filename):
        """
        Initialize the VideoPlayer class.

        Parameters
        ----------
        frames_dir : str
            Directory path containing video frames.
        bbox_dir : str
            Directory path containing bounding box data.
        """
        self.frames_dir = frames_dir
        self.bbox_dir = bbox_dir
        self.annotations_filename = annotations_filename
        self.video_filename = video_filename
        self.video_folders = [folder for folder in os.listdir(self.frames_dir) if os.path.isdir(os.path.join(self.frames_dir, folder))]
        self.current_video_index = 0
        self.current_frame_number = 0
        self.paused = False

        if not self.video_folders:
            print("No video folders found.")
            return

        self.bbox_data, self.video_capture = self.load_video(self.current_video_index)

    def load_video(self, video_index):
        """
        Load a video by setting the current video and frame number.

        Parameters
        ----------
        video_index : int
            Index of the video folder to load.
        """
        video_name = self.video_folders[video_index]
        self.current_video_index = video_index
        self.current_frame_number = 0

        bbox_data = self.load_bounding_box_data(video_name)
        video_capture = self.open_video_capture(video_name)
        return bbox_data, video_capture

    def load_bounding_box_data(self, video_name):
        """
        Load bounding box data for the selected video.

        Parameters
        ----------
        video_name : str
            Name of the video to load bounding box data for.
        """
        bbox_path = os.path.join(self.bbox_dir, video_name, self.annotations_filename)
        if os.path.exists(bbox_path):
            bbox_data = pd.read_csv(bbox_path)
        else:
            bbox_data = None
            print(f"Bounding box file not found for {video_name}.")
        return bbox_data

    def open_video_capture(self, video_name):
        """
        Open the video using VideoCapture.

        Parameters
        ----------
        video_name : str
            Name of the video to open.
        """
        video_path = os.path.join(self.frames_dir, video_name, self.video_filename)
        return cv2.VideoCapture(video_path)

    def play(self):
        """
        Play the video and handle keyboard controls.
        """
        while True:
            if not self.paused:
                ret, frame = self.video_capture.read()
                self.current_frame_number += 1
                if not ret:
                    self.handle_video_end()

                self.display_frame(frame)

            key = cv2.waitKey(1) & 0xFF
            self.handle_keyboard_input(key)

            if key == ord('q'):
                break

        self.release_resources()

    def handle_video_end(self):
        """
        Handle reaching the end of the current video.
        """
        self.current_frame_number = 0
        self.switch_to_next_video()
        self.paused = False

    def switch_to_next_video(self):
        """
        Switch to the next video or loop back to the first.
        """
        self.current_video_index = (self.current_video_index + 1) % len(self.video_folders)
        self.load_video(self.current_video_index)

    def handle_keyboard_input(self, key):
        """
        Handle keyboard input.

        Parameters
        ----------
        key : int
            Key code representing the pressed key.
        """
        if key == ord('p'):
            self.paused = not self.paused

        elif key == ord('d'):
            self.current_frame_number += 1
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number - 1)
            ret, frame = self.video_capture.read()
            self.display_frame(frame)

        elif key == ord('a'):
            self.current_frame_number -= 1
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number - 1)
            ret, frame = self.video_capture.read()
            self.display_frame(frame)

        elif key == ord('n'):  # Backspace key
            self.switch_to_next_video()

    def draw_annotations(self, frame, df):
        """
        Draw object detection annotations (rectangles and object IDs) on a frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Input frame.
        df : pandas.DataFrame
            DataFrame containing object detection data.

        Returns
        -------
        numpy.ndarray
            Frame with annotations.
        """
    
        for _, frame_data in df[df.frame == (self.current_frame_number + 1)].iterrows():
            xmin, xmax, ymin, ymax, subject_id = frame_data[["xmin", "xmax", "ymin", "ymax", "id"]]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"Subject {subject_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
    
    def display_frame(self, frame):
        """
        Display the frame with bounding boxes and subject IDs.

        Parameters
        ----------
        frame : numpy.ndarray
            The frame to display.
        """
        if self.bbox_data is not None and not self.bbox_data.empty:
            if self.current_frame_number < 0:
                self.current_frame_number = 0
            elif self.current_frame_number >= self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT):
                self.current_frame_number = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1

            frame = self.draw_annotations(frame, self.bbox_data)

        cv2.imshow("Video Player", frame)

    def release_resources(self):
        """
        Release video capture and close OpenCV windows.
        """
        if self.video_capture is not None:
            self.video_capture.release()
        cv2.destroyAllWindows()



class VideoGridPlayer(VideoPlayer):
    """ Subclass of Videoplayer, uses its loading functions and drawing of bboxes.
    Create a grid of videos with object detection annotations and lines connecting
    detections with the same 'object_id'.

    Parameters
    ----------
    video_paths : list of str
        List of paths to video files.
    csv_paths : list of str
        List of paths to CSV files containing object detection data.
    """

    def __init__(self, frames_dir, bbox_dir, annotations_filename, video_filename):
        """
        Initialize the VideoGridVisualizer instance.

        Parameters
        ----------
        video_paths : list of str
            List of paths to video files.
        csv_paths : list of str
            List of paths to CSV files containing object detection data.
        """

        self.frames_dir = frames_dir
        self.bbox_dir = bbox_dir
        self.annotations_filename = annotations_filename
        self.video_filename = video_filename
        self.video_folders = [folder for folder in os.listdir(self.frames_dir) if os.path.isdir(os.path.join(self.frames_dir, folder))]
        self.current_frame_number = 0
        self.paused = False
        
        if not self.video_folders:
            print("No video folders found.")
            return

        self.video_data = [self.load_video(i) for i in range(len(self.video_folders))]
        self.video_caps = [video[1] for video in self.video_data]
        self.csv_data = [video[0] for video in self.video_data]


    def draw_lines_between_detections(self, frames, df_list):
        """
        Draw lines connecting detections with the same 'object_id' across videos.

        Parameters
        ----------
        frames : list of numpy.ndarray
            List of frames from different videos.
        df_list : list of pandas.DataFrame
            List of DataFrames containing object detection data for each video.
        """
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                for _, row1 in df_list[i].iterrows():
                    object_id1 = row1['id']
                    for _, row2 in df_list[j].iterrows():
                        object_id2 = row2['id']
                        if object_id1 == object_id2:
                            pt1 = (int(row1['xmin']), int(row1['ymin']))
                            pt2 = (int(row2['xmin']), int(row2['ymin']))
                            cv2.line(frames[i], pt1, pt2, (0, 0, 255), 2)


    def display_video_grid(self):
        """
        Display a grid of videos with object detection annotations and lines connecting
        detections with the same 'object_id'.
        """
        cv2.namedWindow('Video Grid', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video Grid', 1600, 900)
        while True:
            frames = []

            for df, cap in self.video_data:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.draw_annotations(frame, df)
                frames.append(frame)

            #self.draw_lines_between_detections(frames, self.csv_data)

            # Get the dimensions of the first frame
            common_width, common_height = frames[0].shape[1], frames[0].shape[0]

            # Resize and convert frames if necessary
            for i in range(1, len(frames)):
                if frames[i].shape[1] != common_width or frames[i].shape[0] != common_height:
                    frames[i] = cv2.resize(frames[i], (common_width, common_height))
                if frames[i].dtype != frames[0].dtype:
                    frames[i] = cv2.convertScaleAbs(frames[i])

            top_row = cv2.hconcat(frames[:3])
            bottom_row = cv2.hconcat(frames[3:])
            grid = cv2.vconcat([top_row, bottom_row])

            cv2.imshow('Video Grid', grid)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.current_frame_number += 1

        for cap in self.video_caps:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    frames_dir = 'datasets/AIC20/videos/S03'
    bbox_dir = 'datasets/AIC20/annotations/S03'
    annotations_filename = 'gt.txt'
    video_filename = 'vdo.avi'
    mode = 'grid'

    if mode == 'single':
        player = VideoPlayer(frames_dir, bbox_dir, annotations_filename, video_filename)
        player.play()
    elif mode == 'grid':
        visualizer = VideoGridPlayer(frames_dir, bbox_dir, annotations_filename, video_filename)
        visualizer.display_video_grid()

