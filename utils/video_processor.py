import cv2
import numpy as np

class VideoProcessor:
    """Process videos and extract pose keypoints from all frames"""

    def __init__(self, detector):
        self.detector = detector

    def process_video(self, video_path, frame_skip=2):
        """
        Extract keypoints from video frames

        Args:
            video_path: path to video file
            frame_skip: process every Nth frame (2 = 30fps -> 15fps)

        Returns:
            all_keypoints: list of (17, 3) keypoint arrays
            fps: original video FPS
            frame_count: number of frames processed
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        all_keypoints = []
        frame_idx = 0
        processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                keypoints = self.detector.detect(frame)
                all_keypoints.append(keypoints)
                processed += 1

            frame_idx += 1

        cap.release()
        print(f"Processed {processed} frames from {frame_idx} total ({fps} FPS)")

        return all_keypoints, fps, processed
