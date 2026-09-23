"""
blink_detector.py

Handles everything related to detecting blinks from face landmarks:
- Calculating the Eye Aspect Ratio (EAR)
- Deciding whether an eye is open or closed

Keeping this separate from main.py means the "vision math" is isolated
from the "morse code timing" logic, which makes each piece easier to
explain, test, and modify independently.
"""

import math

# MediaPipe's Face Mesh gives 468 landmark points per face.
# These specific index numbers correspond to 6 points around each eye
# (based on MediaPipe's official face mesh landmark map).
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Below this EAR value, the eye is considered "closed".
# This can vary a little between people/lighting, so it's exposed here
# as a constant that's easy to tune (see calibration.py for auto-tuning).
DEFAULT_EAR_THRESHOLD = 0.21


def _euclidean(p1, p2):
    """Straight-line distance between two (x, y) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(landmarks, eye_points, frame_width, frame_height):
    """
    Calculates the Eye Aspect Ratio (EAR) for one eye.

    The EAR formula compares the vertical distance between the eyelids
    to the horizontal width of the eye. When the eye is open, the
    vertical distance is relatively large, so EAR is higher. When the
    eye closes, the vertical distance shrinks toward zero, so EAR drops.

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Args:
        landmarks: MediaPipe's list of 468 face landmark points.
        eye_points: the 6 landmark indices for one eye (LEFT_EYE or RIGHT_EYE).
        frame_width / frame_height: used to convert normalized (0-1)
            landmark coordinates into actual pixel coordinates.

    Returns:
        A float EAR value. Roughly 0.25-0.35 when open, drops below
        ~0.2 when closed (varies by person/camera).
    """
    coords = [
        (landmarks[i].x * frame_width, landmarks[i].y * frame_height)
        for i in eye_points
    ]
    p1, p2, p3, p4, p5, p6 = coords

    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def average_ear(landmarks, frame_width, frame_height):
    """Averages EAR across both eyes for a more stable single reading."""
    left = eye_aspect_ratio(landmarks, LEFT_EYE, frame_width, frame_height)
    right = eye_aspect_ratio(landmarks, RIGHT_EYE, frame_width, frame_height)
    return (left + right) / 2.0


def is_eye_closed(avg_ear, threshold=DEFAULT_EAR_THRESHOLD):
    """Simple helper: is the eye currently closed, given an EAR value?"""
    return avg_ear < threshold
