"""
main.py

Entry point for the Blink-to-Morse Translator.

PIPELINE OVERVIEW:
    Camera -> Face landmark detection (MediaPipe) -> Eye Aspect Ratio (EAR)
    -> Short/Long blink classification -> Morse buffer -> Letter/word timing
    -> Morse-to-text decoding -> Speech + Logging + On-screen display

    Head NOD  -> insert a space (alternative to waiting for the pause timer)
    Head TURN -> backspace (delete the last character)

RUN: python main.py
CONTROLS:
    'q' - quit (saves session log)
    'c' - clear buffer/text (saves what was cleared to the log first)
    't' - speak the current decoded text out loud
"""

import cv2
import mediapipe as mp
import time

from blink_detector import average_ear, is_eye_closed
from morse_translator import decode_letter
from calibration import calibrate_ear_threshold
from head_gesture import get_head_pose, HeadGestureTracker
from speech import SpeechEngine
from logger import log_text

# --- Timing rules (in seconds) ---
SHORT_BLINK_MAX = 0.3   # blink shorter than this = dot, otherwise = dash
LETTER_PAUSE = 1.2      # pause after last blink -> commit current letter
WORD_PAUSE = 3.0        # pause after last blink -> also insert a space

DEFAULT_EAR_THRESHOLD = 0.21  # fallback if auto-calibration fails


def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)

    # --- Feature 5: Auto-calibration ---
    ear_threshold = calibrate_ear_threshold(cap, face_mesh)
    if ear_threshold is None:
        ear_threshold = DEFAULT_EAR_THRESHOLD

    # --- Feature 1 setup: head gesture tracker ---
    gesture_tracker = HeadGestureTracker()

    # --- Feature 2 setup: text-to-speech engine ---
    speech_engine = SpeechEngine()

    # --- State that changes as the program runs ---
    eye_closed = False
    close_start_time = None
    morse_buffer = ""       # dots/dashes for the letter currently being spelled
    decoded_text = ""       # the full decoded sentence so far
    last_blink_end_time = None
    word_pause_added = False

    print("Ready! Blink to spell words. Nod = space, turn head = backspace.")
    print("Press 'q' to quit, 'c' to clear, 't' to speak the text aloud.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror view, feels more natural
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        h, w, _ = frame.shape
        now = time.time()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            ear = average_ear(landmarks, w, h)

            # --- Blink detection (dot/dash) ---
            if is_eye_closed(ear, ear_threshold):
                if not eye_closed:
                    eye_closed = True
                    close_start_time = now
            else:
                if eye_closed:
                    eye_closed = False
                    duration = now - close_start_time
                    morse_buffer += "." if duration < SHORT_BLINK_MAX else "-"
                    last_blink_end_time = now
                    word_pause_added = False

            # --- Feature 1: head nod/turn gestures ---
            pitch, yaw = get_head_pose(landmarks, w, h)
            gesture = gesture_tracker.update(pitch, yaw)
            if gesture == "nod":
                if decoded_text and not decoded_text.endswith(" "):
                    decoded_text += " "
                    print(f"[nod -> space]  So far: {decoded_text}")
            elif gesture == "turn":
                if decoded_text:
                    decoded_text = decoded_text[:-1]
                    print(f"[turn -> backspace]  So far: {decoded_text}")

            cv2.putText(frame, f"EAR: {ear:.3f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No face found...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- Check pause timing to decide when a letter/word is "done" ---
        if not eye_closed and last_blink_end_time is not None:
            gap = now - last_blink_end_time

            if morse_buffer and gap > LETTER_PAUSE:
                letter = decode_letter(morse_buffer)
                decoded_text += letter
                print(f"Letter: {morse_buffer} -> {letter}  |  So far: {decoded_text}")
                morse_buffer = ""

            if gap > WORD_PAUSE and not word_pause_added and decoded_text and not decoded_text.endswith(" "):
                decoded_text += " "
                word_pause_added = True
                print(f"[space added]  So far: {decoded_text}")
                # Feature 3: log each completed word to the session file
                log_text(decoded_text)
                # Feature 2: speak the word just completed
                last_word = decoded_text.strip().split(" ")[-1]
                speech_engine.speak(last_word)

        # --- Feature 4: UI polish -- countdown bar toward letter commit ---
        if not eye_closed and morse_buffer and last_blink_end_time is not None:
            gap = now - last_blink_end_time
            progress = min(gap / LETTER_PAUSE, 1.0)
            bar_x, bar_y, bar_w, bar_h = 20, 100, 300, 15
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
            fill_w = int(bar_w * progress)
            bar_color = (0, 0, 255) if progress > 0.8 else (0, 200, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

        # --- On-screen display ---
        cv2.putText(frame, f"Buffer: {morse_buffer}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Text: {decoded_text}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, "'q' quit | 'c' clear | 't' speak text", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Blink to Morse Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            log_text(decoded_text)
            break
        elif key == ord('c'):
            log_text(decoded_text)
            morse_buffer = ""
            decoded_text = ""
        elif key == ord('t'):
            speech_engine.speak(decoded_text)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal decoded text: {decoded_text}")


if __name__ == "__main__":
    main()
