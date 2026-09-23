"""
head_gesture.py

Detects head NODS (up/down) and TURNS (left/right) using head pose
estimation, so they can act as extra "gesture channels" alongside
blinks -- e.g. nod = space, turn = backspace.

HOW IT WORKS:
1. We pick 6 stable points on the face (nose tip, chin, eye corners,
   mouth corners) and compare them to a generic 3D face model using
   OpenCV's solvePnP -- this is the standard technique for estimating
   head orientation (pitch/yaw/roll) from a single 2D image.
2. We record the user's "neutral" pitch/yaw when they start.
3. Every frame, we check how far the current pitch/yaw has drifted
   from neutral. A big enough drift = a gesture. We wait for the head
   to return near neutral before allowing the next gesture, so one
   nod doesn't get counted multiple times.
"""

import cv2
import numpy as np

# A generic 3D face model (arbitrary units, not a real measurement --
# just needs to be roughly face-shaped for solvePnP to work).
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye, left corner
    (225.0, 170.0, -135.0),    # Right eye, right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0),   # Right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe landmark indices for those 6 points
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def get_head_pose(landmarks, frame_width, frame_height):
    """
    Estimates head orientation from face landmarks.

    Returns:
        (pitch, yaw) in degrees, or (None, None) if estimation failed.
        Pitch: negative/positive as the head tilts down/up.
        Yaw: negative/positive as the head turns left/right.
    """
    image_points = np.array([
        (landmarks[i].x * frame_width, landmarks[i].y * frame_height)
        for i in LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None, None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    # Build a dummy projection matrix so we can decompose it into
    # Euler angles (pitch, yaw, roll) -- a standard OpenCV trick.
    proj_matrix = np.hstack((rotation_matrix, np.zeros((3, 1))))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, _roll = [float(a) for a in euler_angles]
    return pitch, yaw


class HeadGestureTracker:
    """
    Tracks head pose over time and reports discrete gesture events
    ('nod' or 'turn') instead of raw angles.
    """

    def __init__(self, nod_threshold=15.0, turn_threshold=20.0):
        self.baseline_pitch = None
        self.baseline_yaw = None
        self.nod_threshold = nod_threshold
        self.turn_threshold = turn_threshold
        self.gesture_active = False

    def update(self, pitch, yaw):
        """
        Feed in the current frame's (pitch, yaw). Returns 'nod', 'turn',
        or None if no new gesture happened this frame.
        """
        if pitch is None or yaw is None:
            return None

        if self.baseline_pitch is None:
            # First reading becomes our "neutral" reference position.
            self.baseline_pitch = pitch
            self.baseline_yaw = yaw
            return None

        pitch_delta = pitch - self.baseline_pitch
        yaw_delta = yaw - self.baseline_yaw

        if not self.gesture_active:
            if abs(pitch_delta) > self.nod_threshold:
                self.gesture_active = True
                return "nod"
            if abs(yaw_delta) > self.turn_threshold:
                self.gesture_active = True
                return "turn"
        else:
            # Wait for the head to return close to neutral before
            # allowing the next gesture to be detected.
            if (abs(pitch_delta) < self.nod_threshold / 2
                    and abs(yaw_delta) < self.turn_threshold / 2):
                self.gesture_active = False

        return None
