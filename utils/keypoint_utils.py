import numpy as np

def calculate_angle(point1, point2, point3):
    """
    Calculate angle between three points (point2 is vertex)

    Args:
        point1, point2, point3: (y, x) or (y, x, conf) tuples

    Returns:
        angle in degrees (0-180)
    """
    a = np.array(point1[:2])
    b = np.array(point2[:2])
    c = np.array(point3[:2])

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    return np.degrees(angle)

def get_joint_angle(keypoints, joint_name, keypoint_dict):
    """
    Get angle at specific joint

    Args:
        keypoints: (17, 3) array
        joint_name: 'left_elbow', 'right_knee', etc.
        keypoint_dict: mapping of names to indices

    Returns:
        angle in degrees
    """
    joint_connections = {
        'left_elbow': ('left_shoulder', 'left_elbow', 'left_wrist'),
        'right_elbow': ('right_shoulder', 'right_elbow', 'right_wrist'),
        'left_knee': ('left_hip', 'left_knee', 'left_ankle'),
        'right_knee': ('right_hip', 'right_knee', 'right_ankle'),
        'left_shoulder': ('left_elbow', 'left_shoulder', 'left_hip'),
        'right_shoulder': ('right_elbow', 'right_shoulder', 'right_hip'),
        'left_hip': ('left_shoulder', 'left_hip', 'left_knee'),
        'right_hip': ('right_shoulder', 'right_hip', 'right_knee'),
    }

    if joint_name not in joint_connections:
        raise ValueError(f"Unknown joint: {joint_name}")

    p1_name, p2_name, p3_name = joint_connections[joint_name]
    p1 = keypoints[keypoint_dict[p1_name]]
    p2 = keypoints[keypoint_dict[p2_name]]
    p3 = keypoints[keypoint_dict[p3_name]]

    return calculate_angle(p1, p2, p3)

def check_body_alignment(keypoints, keypoint_dict, points):
    """
    Check if three body points are aligned

    Args:
        keypoints: (17, 3) array
        keypoint_dict: mapping
        points: list of 3 keypoint names

    Returns:
        deviation: how far from straight line (0 = perfect)
    """
    p1 = keypoints[keypoint_dict[points[0]]][:2]
    p2 = keypoints[keypoint_dict[points[1]]][:2]
    p3 = keypoints[keypoint_dict[points[2]]][:2]

    v1 = p2 - p1
    v2 = p3 - p1

    cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
    length = np.linalg.norm(v2)
    deviation = cross / (length + 1e-6)

    return deviation
