import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.keypoint_utils import get_joint_angle, check_body_alignment

class PushupAnalyzer:
    """Analyze pushup form and count reps using MoveNet keypoints"""

    def __init__(self, keypoint_dict):
        self.keypoint_dict = keypoint_dict

    def count_reps(self, all_keypoints):
        """
        Count pushup reps from video keypoints

        Args:
            all_keypoints: list of (17, 3) keypoint arrays

        Returns:
            rep_count: number of reps detected
            angles: list of elbow angles over time
            timestamps: frame indices where reps occurred
        """
        angles = []

        for keypoints in all_keypoints:
            try:
                angle = get_joint_angle(keypoints, 'left_elbow', self.keypoint_dict)
                angles.append(angle)
            except Exception as e:
                # If detection fails, use previous angle
                angles.append(angles[-1] if angles else 180)

        # Smooth the angle data
        angles_smooth = gaussian_filter1d(angles, sigma=2)

        # Find valleys (bottom of pushup = small elbow angle)
        valleys, properties = find_peaks(
            [-a for a in angles_smooth],
            distance=10,  # Minimum frames between reps
            prominence=20  # Minimum angle change for a rep
        )

        rep_count = len(valleys)

        return rep_count, angles, valleys

    def analyze_form(self, all_keypoints):
        """
        Analyze pushup form quality

        Returns:
            form_score: 0-100
            feedback: dict with detailed scores
        """
        scores = {
            'body_alignment': self._check_body_alignment(all_keypoints),
            'range_of_motion': self._check_range_of_motion(all_keypoints),
            'stability': self._check_stability(all_keypoints)
        }

        # Overall score is weighted average
        weights = {
            'body_alignment': 0.40,
            'range_of_motion': 0.40,
            'stability': 0.20
        }

        overall = sum(scores[k] * weights[k] for k in scores)

        return overall, scores

    def _check_body_alignment(self, all_keypoints):
        """Check if body is in straight line"""
        deviations = []

        for kp in all_keypoints:
            try:
                dev = check_body_alignment(
                    kp,
                    self.keypoint_dict,
                    ['left_shoulder', 'left_hip', 'left_ankle']
                )
                deviations.append(dev)
            except:
                pass

        if not deviations:
            return 70  # Default if can't detect

        avg_deviation = np.mean(deviations)

        # Convert to score (lower deviation = higher score)
        score = max(0, 100 - (avg_deviation * 500))

        return score

    def _check_range_of_motion(self, all_keypoints):
        """Check if user goes low enough"""
        min_angles = []

        for kp in all_keypoints:
            try:
                angle = get_joint_angle(kp, 'left_elbow', self.keypoint_dict)
                min_angles.append(angle)
            except:
                pass

        if not min_angles:
            return 50  # Default if can't detect

        min_angle = min(min_angles)

        # Full ROM: elbow should flex to ~90 degrees
        if min_angle <= 90:
            return 100
        elif min_angle <= 110:
            return 75
        else:
            return 50

    def _check_stability(self, all_keypoints):
        """Check movement stability"""
        shoulder_positions = []

        for kp in all_keypoints:
            try:
                shoulder = kp[self.keypoint_dict['left_shoulder']]
                shoulder_positions.append(shoulder[:2])
            except:
                pass

        if not shoulder_positions:
            return 70  # Default

        # Calculate variance in x-position (side-to-side movement)
        x_positions = [p[1] for p in shoulder_positions]
        variance = np.var(x_positions)

        if variance < 0.01:
            return 100
        elif variance < 0.02:
            return 80
        else:
            return 60
