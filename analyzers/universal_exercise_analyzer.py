"""
Universal Exercise Analyzer
Analyzes ANY exercise using MoveNet keypoints
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.keypoint_utils import get_joint_angle, check_body_alignment

class UniversalExerciseAnalyzer:
    """
    Universal analyzer that works for ANY exercise
    Automatically detects movement patterns and counts reps
    """

    def __init__(self, keypoint_dict):
        self.keypoint_dict = keypoint_dict

        # Define exercise configurations
        self.exercise_configs = {
            # Upper body pushing exercises
            'pushup': {
                'primary_joints': ['left_elbow', 'right_elbow'],
                'movement_type': 'flexion_extension',
                'direction': 'vertical',
                'min_angle': 70,
                'max_angle': 160,
                'form_checks': ['body_alignment', 'range_of_motion', 'stability']
            },
            'bench press': {
                'primary_joints': ['left_elbow', 'right_elbow'],
                'movement_type': 'flexion_extension',
                'direction': 'vertical',
                'min_angle': 70,
                'max_angle': 160,
                'form_checks': ['range_of_motion', 'stability', 'bar_path']
            },
            'shoulder press': {
                'primary_joints': ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder'],
                'movement_type': 'flexion_extension',
                'direction': 'vertical',
                'min_angle': 80,
                'max_angle': 170,
                'form_checks': ['range_of_motion', 'stability']
            },
            'overhead press': {
                'primary_joints': ['left_elbow', 'right_elbow'],
                'movement_type': 'flexion_extension',
                'direction': 'vertical',
                'min_angle': 80,
                'max_angle': 170,
                'form_checks': ['range_of_motion', 'stability']
            },

            # Lower body exercises
            'squat': {
                'primary_joints': ['left_knee', 'right_knee', 'left_hip', 'right_hip'],
                'movement_type': 'flexion_extension',
                'direction': 'vertical',
                'min_angle': 80,
                'max_angle': 170,
                'form_checks': ['knee_tracking', 'depth', 'back_angle', 'stability']
            },
            'lunge': {
                'primary_joints': ['left_knee', 'right_knee'],
                'movement_type': 'flexion_extension',
                'direction': 'vertical',
                'min_angle': 70,
                'max_angle': 170,
                'form_checks': ['knee_tracking', 'stability']
            },

            # Pulling exercises
            'pull-up': {
                'primary_joints': ['left_elbow', 'right_elbow'],
                'movement_type': 'flexion_extension',
                'direction': 'vertical',
                'min_angle': 60,
                'max_angle': 160,
                'form_checks': ['range_of_motion', 'stability']
            },
            'pull up': {
                'primary_joints': ['left_elbow', 'right_elbow'],
                'movement_type': 'flexion_extension',
                'direction': 'vertical',
                'min_angle': 60,
                'max_angle': 160,
                'form_checks': ['range_of_motion', 'stability']
            },
            'pullup': {
                'primary_joints': ['left_elbow', 'right_elbow'],
                'movement_type': 'flexion_extension',
                'direction': 'vertical',
                'min_angle': 60,
                'max_angle': 160,
                'form_checks': ['range_of_motion', 'stability']
            },
            'row': {
                'primary_joints': ['left_elbow', 'right_elbow'],
                'movement_type': 'flexion_extension',
                'direction': 'horizontal',
                'min_angle': 70,
                'max_angle': 160,
                'form_checks': ['range_of_motion', 'stability']
            },

            # Arm exercises
            'bicep curl': {
                'primary_joints': ['left_elbow', 'right_elbow'],
                'movement_type': 'flexion_extension',
                'direction': 'vertical',
                'min_angle': 30,
                'max_angle': 160,
                'form_checks': ['range_of_motion', 'elbow_stability']
            },
            'tricep extension': {
                'primary_joints': ['left_elbow', 'right_elbow'],
                'movement_type': 'flexion_extension',
                'direction': 'vertical',
                'min_angle': 40,
                'max_angle': 170,
                'form_checks': ['range_of_motion', 'elbow_stability']
            },

            # Core/ab exercises
            'sit-up': {
                'primary_joints': ['left_hip', 'right_hip'],
                'movement_type': 'flexion_extension',
                'direction': 'vertical',
                'min_angle': 60,
                'max_angle': 140,
                'form_checks': ['range_of_motion']
            },
            'crunch': {
                'primary_joints': ['left_hip', 'right_hip'],
                'movement_type': 'flexion_extension',
                'direction': 'vertical',
                'min_angle': 70,
                'max_angle': 130,
                'form_checks': ['range_of_motion']
            },

            # Deadlift/hip hinge
            'deadlift': {
                'primary_joints': ['left_hip', 'right_hip', 'left_knee', 'right_knee'],
                'movement_type': 'hip_hinge',
                'direction': 'vertical',
                'min_angle': 80,
                'max_angle': 170,
                'form_checks': ['back_angle', 'hip_hinge', 'stability']
            },
        }

    def detect_exercise_type(self, all_keypoints, exercise_hint=None):
        """
        Automatically detect exercise type from movement patterns

        Args:
            all_keypoints: list of keypoint arrays
            exercise_hint: optional hint from user (e.g., "chest", "legs")

        Returns:
            detected exercise type
        """
        if exercise_hint:
            # Try to match hint to known exercises
            hint_lower = exercise_hint.lower()

            # Direct match
            if hint_lower in self.exercise_configs:
                return hint_lower

            # Fuzzy match
            for exercise in self.exercise_configs:
                if hint_lower in exercise or exercise in hint_lower:
                    return exercise

            # Match by body part
            body_part_exercises = {
                'chest': 'pushup',
                'leg': 'squat',
                'legs': 'squat',
                'back': 'row',
                'shoulder': 'shoulder press',
                'shoulders': 'shoulder press',
                'arm': 'bicep curl',
                'arms': 'bicep curl',
                'bicep': 'bicep curl',
                'tricep': 'tricep extension',
            }

            for part, exercise in body_part_exercises.items():
                if part in hint_lower:
                    return exercise

        # Analyze movement to detect exercise
        # Check which joints move the most
        joint_movements = self._analyze_joint_movements(all_keypoints)

        # Upper body (elbows move a lot)
        if joint_movements['left_elbow'] > 30 or joint_movements['right_elbow'] > 30:
            # Check if body is horizontal (pushup) or vertical (other)
            body_orientation = self._detect_body_orientation(all_keypoints)
            if body_orientation == 'horizontal':
                return 'pushup'
            else:
                return 'shoulder press'  # Default upper body

        # Lower body (knees/hips move a lot)
        if joint_movements['left_knee'] > 30 or joint_movements['right_knee'] > 30:
            return 'squat'

        # Default to pushup
        return 'pushup'

    def _analyze_joint_movements(self, all_keypoints):
        """Measure how much each joint moves"""
        movements = {}

        for joint_name in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee', 'left_hip', 'right_hip']:
            try:
                angles = []
                for kp in all_keypoints:
                    angle = get_joint_angle(kp, joint_name, self.keypoint_dict)
                    angles.append(angle)

                # Range of motion
                movement = max(angles) - min(angles)
                movements[joint_name] = movement
            except:
                movements[joint_name] = 0

        return movements

    def _detect_body_orientation(self, all_keypoints):
        """Detect if body is horizontal (plank) or vertical (standing)"""
        try:
            # Average shoulder and hip y-positions
            shoulder_y = []
            hip_y = []

            for kp in all_keypoints[:10]:  # Check first 10 frames
                shoulder_y.append(kp[self.keypoint_dict['left_shoulder']][0])
                hip_y.append(kp[self.keypoint_dict['left_hip']][0])

            avg_shoulder_y = np.mean(shoulder_y)
            avg_hip_y = np.mean(hip_y)

            # If shoulder and hip are at similar height, body is horizontal
            if abs(avg_shoulder_y - avg_hip_y) < 0.15:
                return 'horizontal'
            else:
                return 'vertical'
        except:
            return 'vertical'

    def count_reps(self, all_keypoints, exercise_type=None, exercise_hint=None):
        """
        Universal rep counter that works for ANY exercise

        Args:
            all_keypoints: list of keypoint arrays
            exercise_type: specific exercise (optional)
            exercise_hint: hint about exercise (optional)

        Returns:
            rep_count, angles, valleys
        """
        # Detect exercise if not provided
        if not exercise_type:
            exercise_type = self.detect_exercise_type(all_keypoints, exercise_hint)

        print(f"Analyzing exercise: {exercise_type}")

        # Get exercise configuration
        config = self.exercise_configs.get(exercise_type, self.exercise_configs['pushup'])

        # Track angles for primary joints
        angles = []

        for keypoints in all_keypoints:
            try:
                # Average angle across all primary joints
                joint_angles = []
                for joint in config['primary_joints']:
                    angle = get_joint_angle(keypoints, joint, self.keypoint_dict)
                    joint_angles.append(angle)

                avg_angle = np.mean(joint_angles)
                angles.append(avg_angle)
            except:
                angles.append(angles[-1] if angles else config['max_angle'])

        # Smooth the angle data
        angles_smooth = gaussian_filter1d(angles, sigma=2)

        # Find valleys (bottom of movement = small angle)
        valleys, _ = find_peaks(
            [-a for a in angles_smooth],
            distance=10,
            prominence=15
        )

        rep_count = len(valleys)

        return rep_count, angles, valleys

    def analyze_form(self, all_keypoints, exercise_type=None, exercise_hint=None):
        """
        Universal form analyzer

        Returns:
            form_score: 0-100
            detailed_scores: dict
        """
        # Detect exercise if not provided
        if not exercise_type:
            exercise_type = self.detect_exercise_type(all_keypoints, exercise_hint)

        config = self.exercise_configs.get(exercise_type, self.exercise_configs['pushup'])

        scores = {}

        # Check range of motion (universal)
        scores['range_of_motion'] = self._check_range_of_motion(
            all_keypoints,
            config['primary_joints'],
            config['min_angle'],
            config['max_angle']
        )

        # Check stability (universal)
        scores['stability'] = self._check_stability(all_keypoints)

        # Exercise-specific checks
        if 'body_alignment' in config.get('form_checks', []):
            scores['body_alignment'] = self._check_body_alignment(all_keypoints)

        if 'knee_tracking' in config.get('form_checks', []):
            scores['knee_tracking'] = self._check_knee_tracking(all_keypoints)

        if 'back_angle' in config.get('form_checks', []):
            scores['back_angle'] = self._check_back_angle(all_keypoints)

        # Calculate overall score
        overall = np.mean(list(scores.values()))

        return overall, scores

    def _check_range_of_motion(self, all_keypoints, joints, min_angle, max_angle):
        """Check if user achieves full range of motion"""
        try:
            all_angles = []

            for kp in all_keypoints:
                for joint in joints:
                    try:
                        angle = get_joint_angle(kp, joint, self.keypoint_dict)
                        all_angles.append(angle)
                    except:
                        pass

            if not all_angles:
                return 70

            actual_min = min(all_angles)
            actual_max = max(all_angles)
            actual_rom = actual_max - actual_min
            target_rom = max_angle - min_angle

            # Check if they reach the minimum angle
            if actual_min <= min_angle + 20:
                rom_score = 100
            elif actual_min <= min_angle + 40:
                rom_score = 75
            else:
                rom_score = 50

            # Check range
            if actual_rom >= target_rom * 0.8:
                rom_score = min(100, rom_score + 0)
            else:
                rom_score = rom_score * 0.8

            return rom_score
        except:
            return 70

    def _check_stability(self, all_keypoints):
        """Check movement stability (universal)"""
        try:
            # Track shoulder position variance
            positions = []

            for kp in all_keypoints:
                try:
                    shoulder = kp[self.keypoint_dict['left_shoulder']]
                    positions.append(shoulder[:2])
                except:
                    pass

            if not positions:
                return 70

            x_positions = [p[1] for p in positions]
            variance = np.var(x_positions)

            if variance < 0.01:
                return 100
            elif variance < 0.02:
                return 85
            elif variance < 0.03:
                return 70
            else:
                return 60
        except:
            return 70

    def _check_body_alignment(self, all_keypoints):
        """Check if body is in straight line (for pushups, planks)"""
        try:
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
                return 70

            avg_deviation = np.mean(deviations)
            score = max(0, 100 - (avg_deviation * 500))

            return score
        except:
            return 70

    def _check_knee_tracking(self, all_keypoints):
        """Check if knees track over toes (for squats, lunges)"""
        try:
            # Check if knees go past toes (bad form)
            violations = 0
            total = 0

            for kp in all_keypoints:
                try:
                    knee = kp[self.keypoint_dict['left_knee']]
                    ankle = kp[self.keypoint_dict['left_ankle']]

                    # Knee should not go significantly past ankle
                    if knee[1] > ankle[1] + 0.1:  # x-position
                        violations += 1
                    total += 1
                except:
                    pass

            if total == 0:
                return 70

            violation_rate = violations / total

            if violation_rate < 0.1:
                return 100
            elif violation_rate < 0.3:
                return 80
            else:
                return 60
        except:
            return 70

    def _check_back_angle(self, all_keypoints):
        """Check back angle (for deadlifts, squats)"""
        try:
            # Back should stay relatively straight
            angles = []

            for kp in all_keypoints:
                try:
                    shoulder = kp[self.keypoint_dict['left_shoulder']][:2]
                    hip = kp[self.keypoint_dict['left_hip']][:2]

                    # Calculate torso angle from vertical
                    dy = hip[0] - shoulder[0]
                    dx = hip[1] - shoulder[1]
                    angle = np.degrees(np.arctan2(dy, dx))
                    angles.append(abs(angle))
                except:
                    pass

            if not angles:
                return 70

            max_angle = max(angles)

            # Back should not round excessively
            if max_angle < 30:
                return 100
            elif max_angle < 45:
                return 80
            else:
                return 60
        except:
            return 70
