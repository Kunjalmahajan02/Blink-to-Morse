"""
main.py  (v3 -- bug fixes for calibration visibility, head gestures, TTS, logging)

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
import calibration
from head_gesture import get_head_metrics, HeadGestureTracker
from speech import SpeechEngine
from logger import log_text, LOG_FILE

SHORT_BLINK_MAX = 0.3
LETTER_PAUSE = 1.2
WORD_PAUSE = 3.0
DEFAULT_EAR_THRESHOLD = 0.21

WINDOW_NAME = "Blink to Morse Translator"


def main():
    print(f"[logger] Session log will be saved to: {LOG_FILE}")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME)

    speech_engine = SpeechEngine()
    # Speak immediately on startup -- an easy way to confirm audio
    # is actually working before you start relying on it.
    speech_engine.speak("System ready. Starting calibration.")

    cal_result = calibration.calibrate(cap, face_mesh, duration=5.0, window_name=WINDOW_NAME)
    if cal_result is None:
        ear_threshold = DEFAULT_EAR_THRESHOLD
        gesture_tracker = HeadGestureTracker()
    else:
        ear_threshold = cal_result["ear_threshold"]
        gesture_tracker = HeadGestureTracker()
        if cal_result["pitch"] is not None:
            gesture_tracker.set_baseline(cal_result["pitch"], cal_result["yaw"])

    eye_closed = False
    close_start_time = None
    morse_buffer = ""
    decoded_text = ""
    last_blink_end_time = None
    word_pause_added = False

    print("Ready! Blink to spell words. Nod = space, turn head = backspace.")
    print("Press 'q' to quit, 'c' to clear, 't' to speak the text aloud.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        h, w, _ = frame.shape
        now = time.time()

        pitch_delta_display = None
        yaw_delta_display = None

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            ear = average_ear(landmarks, w, h)

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

            pitch, yaw = get_head_metrics(landmarks, w, h)
            if gesture_tracker.baseline_pitch is not None and pitch is not None:
                pitch_delta_display = pitch - gesture_tracker.baseline_pitch
                yaw_delta_display = yaw - gesture_tracker.baseline_yaw

            gesture = gesture_tracker.update(pitch, yaw)
            if gesture == "nod":
                if decoded_text and not decoded_text.endswith(" "):
                    decoded_text += " "
                    print(f"[gesture] NOD -> space added. So far: {decoded_text}")
            elif gesture == "turn":
                if decoded_text:
                    decoded_text = decoded_text[:-1]
                    print(f"[gesture] TURN -> backspace. So far: {decoded_text}")

            cv2.putText(frame, f"EAR: {ear:.3f} (threshold {ear_threshold:.3f})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if pitch_delta_display is not None:
                cv2.putText(frame, f"Pitch delta: {pitch_delta_display:+.3f}  Yaw delta: {yaw_delta_display:+.3f}",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
        else:
            cv2.putText(frame, "No face found...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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
                log_text(decoded_text)
                last_word = decoded_text.strip().split(" ")[-1]
                speech_engine.speak(last_word)

        if not eye_closed and morse_buffer and last_blink_end_time is not None:
            gap = now - last_blink_end_time
            progress = min(gap / LETTER_PAUSE, 1.0)
            bar_x, bar_y, bar_w, bar_h = 20, 130, 300, 15
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
            fill_w = int(bar_w * progress)
            bar_color = (0, 0, 255) if progress > 0.8 else (0, 200, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

        cv2.putText(frame, f"Buffer: {morse_buffer}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Text: {decoded_text}", (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, "'q' quit | 'c' clear | 't' speak text", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            log_text(decoded_text)
            break
        elif key == ord('c'):
            log_text(decoded_text)
            morse_buffer = ""
            decoded_text = ""
        elif key == ord('t'):
            print("[main] 't' pressed -- speaking current text")
            speech_engine.speak(decoded_text)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal decoded text: {decoded_text}")


if __name__ == "__main__":
    main()
