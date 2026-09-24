"""
calibration.py  (v2)

Auto-calibrates BOTH:
1. Your personal EAR threshold (for blink detection)
2. Your neutral head position (pitch/yaw baseline, for nod/turn detection)

...in one combined 5-second phase at startup, with a clearly visible
on-screen countdown so it's obvious the calibration is happening (and
when it ends).
"""

import cv2
import time

from blink_detector import average_ear
from head_gesture import get_head_metrics

THRESHOLD_RATIO = 0.75  # EAR threshold = this fraction of your open-eye baseline


def calibrate(cap, face_mesh, duration=5.0, window_name="Blink to Morse Translator"):
    """
    Runs the combined calibration phase.

    Returns a dict: {"ear_threshold": float, "pitch": float, "yaw": float}
    or None if no face was detected at all during calibration.
    """
    print(f"[calibration] Starting {duration:.0f}s calibration -- "
          f"look at the camera normally, keep eyes open.")

    ear_readings = []
    pitch_readings = []
    yaw_readings = []
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            ear_readings.append(average_ear(landmarks, w, h))
            pitch, yaw = get_head_metrics(landmarks, w, h)
            if pitch is not None:
                pitch_readings.append(pitch)
                yaw_readings.append(yaw)
            face_status = "Face detected"
            status_color = (0, 255, 0)
        else:
            face_status = "No face -- center yourself in frame!"
            status_color = (0, 0, 255)

        remaining = duration - (time.time() - start_time)

        # Big, unmistakable calibration banner
        cv2.rectangle(frame, (0, 0), (w, 110), (40, 40, 40), -1)
        cv2.putText(frame, "CALIBRATING - LOOK STRAIGHT, KEEP EYES OPEN",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Time remaining: {remaining:.1f}s",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, face_status, (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

        # Also print a countdown to the terminal every ~1 second, so it's
        # obvious there too (helpful if the video window is behind other
        # windows or hard to notice).
        print(f"\r[calibration] {remaining:4.1f}s remaining...", end="", flush=True)

    print()  # newline after the countdown

    if not ear_readings:
        print("[calibration] FAILED -- no face detected during calibration. "
              "Falling back to default settings.")
        return None

    baseline_ear = sum(ear_readings) / len(ear_readings)
    ear_threshold = baseline_ear * THRESHOLD_RATIO

    result = {"ear_threshold": ear_threshold, "pitch": None, "yaw": None}

    if pitch_readings:
        result["pitch"] = sum(pitch_readings) / len(pitch_readings)
        result["yaw"] = sum(yaw_readings) / len(yaw_readings)

    print(f"[calibration] Done. EAR baseline: {baseline_ear:.3f} -> "
          f"threshold: {ear_threshold:.3f}")
    if result["pitch"] is not None:
        print(f"[calibration] Head neutral position -- pitch: "
              f"{result['pitch']:.3f}, yaw: {result['yaw']:.3f}")

    return result
