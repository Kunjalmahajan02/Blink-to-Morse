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
import codebook
from head_gesture import get_head_metrics, HeadGestureTracker
import enhancement
from collections import deque

SHORT_BLINK_MAX = 0.3
LETTER_PAUSE = 1.2
WORD_PAUSE = 3.0
DEFAULT_EAR_THRESHOLD = 0.21

# --- Rolling baseline (robustness to camera angle / head position) ---
# The "eyes open" EAR level depends on viewing angle: a camera above the
# face makes eyes look narrower, a turned head changes proportions, and
# people shift position during a recording. A threshold fixed once at
# calibration then becomes wrong. So the open-eye level is re-measured
# continuously from the last few seconds of readings.
THRESHOLD_RATIO = 0.75          # threshold = 75% of the current open-eye level
BASELINE_WINDOW_S = 6.0         # how many seconds of history to look at
BASELINE_PERCENTILE = 0.85      # "open level" = 85th percentile of recent EAR
# Using a high percentile (not the average) means blinks, even long dashes,
# don't drag the baseline down: eyes are open most of the time, so the
# upper readings represent "open" even while someone blinks Morse.
BASELINE_MIN_SAMPLES_S = 1.5    # need this much history before trusting it
BASELINE_CLAMP = (0.6, 1.5)     # never drift below 60% / above 150% of calibration

# --- Panic pattern: an unmistakable distress signal that bypasses ---
# --- normal spelling entirely, for when a person is injured or    ---
# --- cannot keep up sustained accurate Morse timing.               ---
# Five long, deliberate holds in quick succession. Each hold must be
# well beyond a normal dash (which can already be any length above
# SHORT_BLINK_MAX), so a real message is never mistaken for a panic
# signal, and the holds must come close together so it reads as one
# deliberate act, not five dashes scattered through a long message.
PANIC_BLINK_COUNT = 5
PANIC_MIN_HOLD_S = 1.0
# Measured between the END of one hold and the END of the next, so this
# must cover the next hold's own duration (>= 1.0s) plus a real pause
# between blinks, not just the pause alone.
PANIC_MAX_GAP_S = 3.0
# A deliberate Morse blink (even a dash) lasts well under 2 s. If the eyes
# seem "closed" for longer, the viewing angle or position has most likely
# changed and pushed the open-eye EAR under the old threshold. In that case
# the "closure" is discarded (no fake dash) and the baseline is re-measured
# immediately from the last second of readings.
MAX_CLOSURE_S = 2.0
REBASELINE_RECENT_S = 1.0


class BlinkMorsePipeline:
    def __init__(self, ear_threshold=DEFAULT_EAR_THRESHOLD, enhance_mode="auto",
                 adaptive_threshold=True, gestures_enabled=True):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.ear_threshold = ear_threshold
        self.calibrated_baseline = ear_threshold / THRESHOLD_RATIO
        self.enhance_mode = enhance_mode   # "off", "auto" or "on" (see enhancement.py)
        self.adaptive_threshold = adaptive_threshold
        # Head gestures (nod = space, turn = backspace) are meant for a
        # cooperating live user. In recorded footage, natural head
        # movement would delete letters, so video modes switch them off.
        self.gestures_enabled = gestures_enabled
        self._ear_history = deque()        # (time, ear) pairs for the rolling baseline
        self._panic_streak = deque()       # timestamps of recent long holds, for panic detection
        self.gesture_tracker = HeadGestureTracker()

        self.eye_closed = False
        self.close_start_time = None
        self.morse_buffer = ""
        self.decoded_text = ""
        self.last_blink_end_time = None
        self.word_pause_added = False

    def set_calibration(self, ear_threshold, pitch_baseline=None, yaw_baseline=None):
        self.ear_threshold = ear_threshold
        self.calibrated_baseline = ear_threshold / THRESHOLD_RATIO
        self._ear_history.clear()
        if pitch_baseline is not None:
            self.gesture_tracker.set_baseline(pitch_baseline, yaw_baseline)

    def clear_text(self):
        self.morse_buffer = ""
        self.decoded_text = ""

    def _current_threshold(self, now, ear):
        """Updates the rolling open-eye baseline with this reading and
        returns the threshold to use for this frame."""
        if not self.adaptive_threshold:
            return self.ear_threshold
        self._ear_history.append((now, ear))
        while self._ear_history and now - self._ear_history[0][0] > BASELINE_WINDOW_S:
            self._ear_history.popleft()
        span = now - self._ear_history[0][0]
        if span < BASELINE_MIN_SAMPLES_S or len(self._ear_history) < 10:
            return self.ear_threshold
        values = sorted(e for _, e in self._ear_history)
        open_level = values[int(BASELINE_PERCENTILE * (len(values) - 1))]
        lo, hi = BASELINE_CLAMP
        open_level = min(max(open_level, lo * self.calibrated_baseline), hi * self.calibrated_baseline)
        return THRESHOLD_RATIO * open_level

    def _rebaseline(self, now):
        recent = [e for t, e in self._ear_history if now - t <= REBASELINE_RECENT_S]
        if not recent:
            return
        recent.sort()
        level = recent[len(recent) // 2]   # median of the last second
        lo, hi = BASELINE_CLAMP
        level = min(max(level, lo * self.calibrated_baseline), hi * self.calibrated_baseline)
        self.ear_threshold = THRESHOLD_RATIO * level
        self._ear_history = deque((t, e) for t, e in self._ear_history if now - t <= REBASELINE_RECENT_S)
        self.eye_closed = False
        self.close_start_time = None

    def _update_eye_state(self, now, ear):
        """Core blink logic for one EAR reading.
        Returns (threshold_used, new_symbol_or_None, rebaselined_bool,
        duration_or_None). duration is the blink's length in seconds,
        set only on the frame where a blink just completed."""
        threshold = self._current_threshold(now, ear)
        rebaselined = False
        if (self.adaptive_threshold and self.eye_closed and self.close_start_time is not None
                and now - self.close_start_time > MAX_CLOSURE_S):
            self._rebaseline(now)
            threshold = self.ear_threshold
            rebaselined = True

        symbol = None
        duration = None
        if is_eye_closed(ear, threshold):
            if not self.eye_closed:
                self.eye_closed = True
                self.close_start_time = now
        elif self.eye_closed:
            self.eye_closed = False
            duration = now - self.close_start_time
            symbol = "." if duration < SHORT_BLINK_MAX else "-"
        return threshold, symbol, rebaselined, duration

    def _check_panic(self, now, duration):
        """Feed in a just-completed blink's duration. Returns True the
        moment PANIC_BLINK_COUNT long holds have landed in quick
        succession; otherwise tracks progress toward it."""
        if duration is None or duration < PANIC_MIN_HOLD_S:
            self._panic_streak.clear()   # a short blink breaks the streak
            return False
        if self._panic_streak and now - self._panic_streak[-1] > PANIC_MAX_GAP_S:
            self._panic_streak.clear()   # too slow to be one deliberate act
        self._panic_streak.append(now)
        if len(self._panic_streak) >= PANIC_BLINK_COUNT:
            self._panic_streak.clear()
            return True
        return False

    def detect(self, frame_bgr):
        """Low-light enhancement (if needed) + face landmark detection.
        Every mode (live, upload, calibration, evaluation) goes through
        here, so they all get identical preprocessing.
        Returns (mediapipe_results, light_info, frame_used)."""
        frame_used, light_info = enhancement.enhance(frame_bgr, self.enhance_mode)
        rgb = cv2.cvtColor(frame_used, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb), light_info, frame_used

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
            light (brightness info from enhancement.py)
            frame_used (the enhanced frame, or the original if not enhanced)
        """
        events = {
            "face_found": False, "ear": None,
            "pitch_delta": None, "yaw_delta": None,
            "letter": None, "word_complete": None, "codebook_phrase": False, "panic": False,
            "letter_progress": 0.0,
            "ear_threshold": None,
            "rebaselined": False,
        }

        results, light_info, frame_used = self.detect(frame_bgr)
        events["light"] = light_info
        events["frame_used"] = frame_used
        h, w, _ = frame_bgr.shape

        if results.multi_face_landmarks:
            events["face_found"] = True
            landmarks = results.multi_face_landmarks[0].landmark
            ear = average_ear(landmarks, w, h)
            events["ear"] = ear
            threshold, symbol, rebaselined, duration = self._update_eye_state(now, ear)
            events["ear_threshold"] = threshold
            events["rebaselined"] = rebaselined
            if symbol:
                if self._check_panic(now, duration):
                    # Distress signal overrides whatever was being spelled:
                    # discard it rather than let 5 long holds garble the
                    # message, and reset so a second signal can be sent.
                    events["panic"] = True
                    self.morse_buffer = ""
                    self.eye_closed = False
                    self.close_start_time = None
                    self.last_blink_end_time = None
                    self.word_pause_added = False
                else:
                    self.morse_buffer += symbol
                    self.last_blink_end_time = now
                    self.word_pause_added = False

            pitch, yaw = get_head_metrics(landmarks, w, h)
            if self.gesture_tracker.baseline_pitch is not None and pitch is not None:
                events["pitch_delta"] = pitch - self.gesture_tracker.baseline_pitch
                events["yaw_delta"] = yaw - self.gesture_tracker.baseline_yaw

            gesture = self.gesture_tracker.update(pitch, yaw) if self.gestures_enabled else None
            if gesture == "nod" and self.decoded_text and not self.decoded_text.endswith(" "):
                self.decoded_text += " "
            elif gesture == "turn" and self.decoded_text:
                self.decoded_text = self.decoded_text[:-1]

        if not self.eye_closed and self.last_blink_end_time is not None:
            gap = now - self.last_blink_end_time

            if self.morse_buffer and gap > LETTER_PAUSE:
                letter = codebook.decode_group(self.morse_buffer, decode_letter)
                is_phrase = codebook.is_codebook_pattern(self.morse_buffer)
                self.decoded_text += (" " + letter + " ") if is_phrase else letter
                events["letter"] = letter
                events["codebook_phrase"] = is_phrase
                self.morse_buffer = ""

            if gap > WORD_PAUSE and not self.word_pause_added and self.decoded_text and not self.decoded_text.endswith(" "):
                self.decoded_text += " "
                self.word_pause_added = True
                events["word_complete"] = self.decoded_text.strip().split(" ")[-1]

        if not self.eye_closed and self.morse_buffer and self.last_blink_end_time is not None:
            events["letter_progress"] = min((now - self.last_blink_end_time) / LETTER_PAUSE, 1.0)

        return events
