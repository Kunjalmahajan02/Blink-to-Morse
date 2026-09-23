"""
pipeline.py

The CORE detection logic, extracted into a reusable class so both the
real-time webcam mode AND the video-file upload mode can share exactly
the same blink/gesture/morse pipeline. Only the SOURCE of frames
differs between the two modes -- everything else is identical.

This matters for explaining the project: "we built one detection
engine, and it works the same whether the frames come live from a
webcam or from a pre-recorded video file."

TIMING NOTE: this class doesn't call time.time() itself -- the caller
passes in `now`, a time value in seconds. For the webcam, that's
wall-clock time.time(). For a video file, it should be
(frame_index / video_fps) -- the video's OWN internal clock -- so
blink durations are measured correctly regardless of how fast the
computer processes the file.
"""

import cv2
import mediapipe as mp

from blink_detector import average_ear, is_eye_closed
from morse_translator import decode_letter
from head_gesture import get_head_metrics, HeadGestureTracker

SHORT_BLINK_MAX = 0.3
LETTER_PAUSE = 1.2
WORD_PAUSE = 3.0
DEFAULT_EAR_THRESHOLD = 0.21


class BlinkMorsePipeline:
    def __init__(self, ear_threshold=DEFAULT_EAR_THRESHOLD):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.ear_threshold = ear_threshold
        self.gesture_tracker = HeadGestureTracker()

        self.eye_closed = False
        self.close_start_time = None
        self.morse_buffer = ""
        self.decoded_text = ""
        self.last_blink_end_time = None
        self.word_pause_added = False

    def set_calibration(self, ear_threshold, pitch_baseline=None, yaw_baseline=None):
        self.ear_threshold = ear_threshold
        if pitch_baseline is not None:
            self.gesture_tracker.set_baseline(pitch_baseline, yaw_baseline)

    def clear_text(self):
        self.morse_buffer = ""
        self.decoded_text = ""

    def process_frame(self, frame_bgr, now):
        """
        Processes one frame and updates internal state.

        Args:
            frame_bgr: the video frame (BGR, as OpenCV reads it).
            now: current time in seconds (see module docstring).

        Returns a dict describing what happened this frame:
            face_found, ear, pitch_delta, yaw_delta,
            letter (a newly committed letter, or None),
            word_complete (a newly completed word, or None),
            letter_progress (0.0-1.0, for a progress bar UI)
        """
        events = {
            "face_found": False, "ear": None,
            "pitch_delta": None, "yaw_delta": None,
            "letter": None, "word_complete": None,
            "letter_progress": 0.0,
        }

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        h, w, _ = frame_bgr.shape

        if results.multi_face_landmarks:
            events["face_found"] = True
            landmarks = results.multi_face_landmarks[0].landmark
            ear = average_ear(landmarks, w, h)
            events["ear"] = ear

            if is_eye_closed(ear, self.ear_threshold):
                if not self.eye_closed:
                    self.eye_closed = True
                    self.close_start_time = now
            else:
                if self.eye_closed:
                    self.eye_closed = False
                    duration = now - self.close_start_time
                    self.morse_buffer += "." if duration < SHORT_BLINK_MAX else "-"
                    self.last_blink_end_time = now
                    self.word_pause_added = False

            pitch, yaw = get_head_metrics(landmarks, w, h)
            if self.gesture_tracker.baseline_pitch is not None and pitch is not None:
                events["pitch_delta"] = pitch - self.gesture_tracker.baseline_pitch
                events["yaw_delta"] = yaw - self.gesture_tracker.baseline_yaw

            gesture = self.gesture_tracker.update(pitch, yaw)
            if gesture == "nod" and self.decoded_text and not self.decoded_text.endswith(" "):
                self.decoded_text += " "
            elif gesture == "turn" and self.decoded_text:
                self.decoded_text = self.decoded_text[:-1]

        if not self.eye_closed and self.last_blink_end_time is not None:
            gap = now - self.last_blink_end_time

            if self.morse_buffer and gap > LETTER_PAUSE:
                letter = decode_letter(self.morse_buffer)
                self.decoded_text += letter
                events["letter"] = letter
                self.morse_buffer = ""

            if gap > WORD_PAUSE and not self.word_pause_added and self.decoded_text and not self.decoded_text.endswith(" "):
                self.decoded_text += " "
                self.word_pause_added = True
                events["word_complete"] = self.decoded_text.strip().split(" ")[-1]

        if not self.eye_closed and self.morse_buffer and self.last_blink_end_time is not None:
            events["letter_progress"] = min((now - self.last_blink_end_time) / LETTER_PAUSE, 1.0)

        return events
