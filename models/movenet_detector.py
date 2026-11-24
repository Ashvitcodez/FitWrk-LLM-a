import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class MoveNetDetector:
    """MoveNet Lightning wrapper for pose detection"""

    def __init__(self, model_variant='lightning'):
        if model_variant == 'lightning':
            model_url = 'https://tfhub.dev/google/movenet/singlepose/lightning/4'
            self.input_size = 192
        else:
            model_url = 'https://tfhub.dev/google/movenet/singlepose/thunder/4'
            self.input_size = 256

        print(f"Loading MoveNet {model_variant} model...")
        self.model = hub.load(model_url)
        self.movenet = self.model.signatures['serving_default']
        print("Model loaded successfully!")

        self.KEYPOINT_NAMES = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        self.KEYPOINT_DICT = {name: idx for idx, name in enumerate(self.KEYPOINT_NAMES)}

    def preprocess_frame(self, frame):
        """Preprocess frame for MoveNet input"""
        img = tf.image.convert_image_dtype(frame, tf.uint8)
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize_with_pad(
            img,
            target_height=self.input_size,
            target_width=self.input_size
        )
        return img

    def detect(self, frame):
        """Detect pose keypoints in a single frame"""
        input_tensor = self.preprocess_frame(frame)
        outputs = self.movenet(input_tensor)
        keypoints = outputs['output_0'].numpy()
        return keypoints[0, 0, :, :]

    def get_keypoint(self, keypoints, name):
        """Get specific keypoint by name"""
        idx = self.KEYPOINT_DICT[name]
        return keypoints[idx]
