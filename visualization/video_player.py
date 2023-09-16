import cv2
import os
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
        self.bbox_data = None
        self.video_capture = None

        if not self.video_folders:
            print("No video folders found.")
            return

        self.load_video(self.current_video_index)

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

        self.load_bounding_box_data(video_name)
        self.open_video_capture(video_name)

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
            self.bbox_data = pd.read_csv(bbox_path)
        else:
            self.bbox_data = None
            print(f"Bounding box file not found for {video_name}.")

    def open_video_capture(self, video_name):
        """
        Open the video using VideoCapture.

        Parameters
        ----------
        video_name : str
            Name of the video to open.
        """
        video_path = os.path.join(self.frames_dir, video_name, self.video_filename)
        self.video_capture = cv2.VideoCapture(video_path)

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

            if self.current_frame_number + 1 in self.bbox_data.index:
                for _, frame_data in self.bbox_data[self.bbox_data.frame == (self.current_frame_number + 1)].iterrows():
                    xmin, xmax, ymin, ymax, subject_id = frame_data[["xmin", "xmax", "ymin", "ymax", "id"]]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f"Subject {subject_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Video Player", frame)

    def release_resources(self):
        """
        Release video capture and close OpenCV windows.
        """
        if self.video_capture is not None:
            self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    frames_dir = 'datasets/AIC20/videos/S01'
    bbox_dir = 'datasets/AIC20/annotations/S01'
    annotations_filename = 'gt.txt'
    video_filename = 'vdo.avi'

    player = VideoPlayer(frames_dir, bbox_dir, annotations_filename, video_filename)
    player.play()
