"""
head_gesture.py  (v2 -- ratio-based, replaces the solvePnP approach)

WHY THE CHANGE:
The original version used OpenCV's solvePnP to estimate full 3D head
rotation, which requires guessing the camera's focal length. On most
webcams that guess is inaccurate enough that real nods/turns barely
moved the calculated angle -- so gestures almost never triggered.

NEW APPROACH:
Instead of full 3D pose, we compare RATIOS of distances between
stable facial landmarks:
- NOD: how far down the nose sits relative to eye level, normalized
  by the distance between your eyes (so it works regardless of how
  close you are to the camera).
- TURN: how the nose's horizontal position compares to your left vs.
  right cheek. Facing forward, it's roughly centered; turning shifts
  it noticeably toward one side.

This is simpler, easier to explain, and responds much more linearly
to real head movement.
"""

import math

NOSE_TIP = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_CHEEK = 234
RIGHT_CHEEK = 454


def _point(landmarks, idx, w, h):
    return (landmarks[idx].x * w, landmarks[idx].y * h)


def _distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_head_metrics(landmarks, frame_width, frame_height):
    """
    Returns (pitch_ratio, yaw_ratio) for the current frame.

    pitch_ratio: how far below eye-level the nose sits, normalized by
        eye distance. Increases when nodding down, decreases when
        tilting the head back/up.
    yaw_ratio: fraction of nose-to-cheek distance on the left side
        (0.5 = facing forward; drifts toward 0 or 1 when turning).

    Returns (None, None) if landmarks look degenerate (shouldn't
    normally happen once a face is detected).
    """
    nose = _point(landmarks, NOSE_TIP, frame_width, frame_height)
    left_eye = _point(landmarks, LEFT_EYE_OUTER, frame_width, frame_height)
    right_eye = _point(landmarks, RIGHT_EYE_OUTER, frame_width, frame_height)
    left_cheek = _point(landmarks, LEFT_CHEEK, frame_width, frame_height)
    right_cheek = _point(landmarks, RIGHT_CHEEK, frame_width, frame_height)

    interocular = _distance(left_eye, right_eye)
    if interocular < 1e-6:
        return None, None

    eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
    pitch_ratio = (nose[1] - eye_center_y) / interocular

    dist_left = abs(nose[0] - left_cheek[0])
    dist_right = abs(right_cheek[0] - nose[0])
    total = dist_left + dist_right
    yaw_ratio = dist_left / total if total > 1e-6 else 0.5

    return pitch_ratio, yaw_ratio


class HeadGestureTracker:
    """
    Tracks pitch/yaw ratios over time and reports discrete gesture
    events ('nod' or 'turn') instead of raw numbers.

    nod_delta / turn_delta: how far the ratio must move away from
    baseline to count as a gesture. These are in the same "ratio"
    units as get_head_metrics() returns -- tune them by watching the
    on-screen debug numbers while nodding/turning naturally.
    """

    def __init__(self, nod_delta=0.20, turn_delta=0.10):
        self.baseline_pitch = None
        self.baseline_yaw = None
        self.nod_delta = nod_delta
        self.turn_delta = turn_delta
        self.gesture_active = False

    def set_baseline(self, pitch_ratio, yaw_ratio):
        self.baseline_pitch = pitch_ratio
        self.baseline_yaw = yaw_ratio

    def update(self, pitch_ratio, yaw_ratio):
        """Feed the current frame's ratios. Returns 'nod', 'turn', or None."""
        if pitch_ratio is None or yaw_ratio is None:
            return None

        if self.baseline_pitch is None:
            self.set_baseline(pitch_ratio, yaw_ratio)
            return None

        pitch_delta = pitch_ratio - self.baseline_pitch
        yaw_delta = yaw_ratio - self.baseline_yaw

        if not self.gesture_active:
            if abs(pitch_delta) > self.nod_delta:
                self.gesture_active = True
                return "nod"
            if abs(yaw_delta) > self.turn_delta:
                self.gesture_active = True
                return "turn"
        else:
            if (abs(pitch_delta) < self.nod_delta / 2
                    and abs(yaw_delta) < self.turn_delta / 2):
                self.gesture_active = False

        return None
