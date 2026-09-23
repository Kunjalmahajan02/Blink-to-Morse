"""
calibration.py

Auto-calibrates the EAR threshold instead of relying on one fixed number
for everyone. Different people/lighting/cameras produce different EAR
values, so a fixed threshold (like 0.21) isn't equally reliable for
everyone.

HOW IT WORKS:
We ask the user to keep their eyes open normally for a few seconds,
record their EAR during that time, and take the average as their
personal "eyes open" baseline. We then set the threshold a bit below
that baseline -- so a real blink still crosses it, but normal open-eye
fluctuation doesn't.
"""

import cv2
import time

from blink_detector import average_ear

# How far below the open-eye baseline we consider "closed".
# e.g. 0.75 means: threshold = 75% of their normal open-eye EAR.
THRESHOLD_RATIO = 0.75


def calibrate_ear_threshold(cap, face_mesh, duration=3.0):
    """
    Runs a short calibration phase using the live camera feed.

    Args:
        cap: an already-opened cv2.VideoCapture object.
        face_mesh: an already-created MediaPipe FaceMesh object.
        duration: how many seconds to sample for.

    Returns:
        A personalized EAR threshold (float), or None if no face was
        detected during calibration (caller should fall back to a
        default threshold in that case).
    """
    print(f"Calibrating... keep your eyes open normally for {duration:.0f} seconds.")
    readings = []
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
            readings.append(average_ear(landmarks, w, h))

        remaining = duration - (time.time() - start_time)
        cv2.putText(frame, f"Calibrating... keep eyes open ({remaining:.1f}s)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Blink to Morse Translator", frame)
        cv2.waitKey(1)

    if not readings:
        print("Calibration failed (no face detected) -- using default threshold.")
        return None

    baseline = sum(readings) / len(readings)
    threshold = baseline * THRESHOLD_RATIO
    print(f"Calibration done. Open-eye baseline: {baseline:.3f} -> Threshold: {threshold:.3f}")
    return threshold
