"""
gui.py

The desktop application. Gives you two buttons -- "Start Real-Time"
and "Upload Video File" -- both of which feed frames into the exact
same BlinkMorsePipeline (see pipeline.py).

RUN: python gui.py

WHY A SEPARATE FILE FROM main.py:
main.py is still there as the original simple command-line script --
useful if you ever want to show your guide "the bare pipeline" without
GUI code cluttering the explanation. gui.py is the polished version
you'd actually demo.
"""

import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from pipeline import BlinkMorsePipeline
from blink_detector import average_ear
from head_gesture import get_head_metrics
from speech import SpeechEngine
from logger import log_text, LOG_FILE

CALIBRATION_DURATION = 5.0
THRESHOLD_RATIO = 0.75


class BlinkMorseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Blink-to-Morse Translator")
        self.root.geometry("900x740")
        self.root.configure(bg="#1e1e1e")

        self.cap = None
        self.pipeline = None
        self.running = False
        self.mode = None            # "realtime" or "video"
        self.video_fps = 30.0
        self.video_frame_index = 0

        self.calibrating = False
        self.cal_start_time = None
        self.cal_ear_readings = []
        self.cal_pitch_readings = []
        self.cal_yaw_readings = []

        self.speech_engine = SpeechEngine()

        self._build_ui()

    # ---------------- UI LAYOUT ----------------

    def _build_ui(self):
        top = tk.Frame(self.root, bg="#1e1e1e")
        top.pack(pady=10)

        tk.Button(top, text="Start Real-Time (Webcam)", command=self.start_realtime,
                  width=24, height=2, bg="#2e7d32", fg="white").pack(side="left", padx=5)
        tk.Button(top, text="Upload Video File", command=self.start_video_upload,
                  width=20, height=2, bg="#1565c0", fg="white").pack(side="left", padx=5)
        tk.Button(top, text="Stop", command=self.stop, width=10, height=2,
                  bg="#b71c1c", fg="white").pack(side="left", padx=5)

        self.video_label = tk.Label(self.root, bg="black", width=640, height=360)
        self.video_label.pack(pady=10)

        info = tk.Frame(self.root, bg="#1e1e1e")
        info.pack(pady=5, fill="x", padx=20)

        tk.Label(info, text="Buffer:", font=("Arial", 12, "bold"),
                 bg="#1e1e1e", fg="white").grid(row=0, column=0, sticky="w")
        self.buffer_var = tk.StringVar(value="")
        tk.Label(info, textvariable=self.buffer_var, font=("Consolas", 14),
                 bg="#1e1e1e", fg="#4caf50").grid(row=0, column=1, sticky="w")

        tk.Label(info, text="Decoded Text:", font=("Arial", 12, "bold"),
                 bg="#1e1e1e", fg="white").grid(row=1, column=0, sticky="nw")
        self.text_var = tk.StringVar(value="")
        tk.Label(info, textvariable=self.text_var, font=("Arial", 16),
                 bg="#1e1e1e", fg="#ffca28", wraplength=650, justify="left"
                 ).grid(row=1, column=1, sticky="w")

        bottom = tk.Frame(self.root, bg="#1e1e1e")
        bottom.pack(pady=10)
        tk.Button(bottom, text="Speak Text", command=self.speak_text, width=15).pack(side="left", padx=5)
        tk.Button(bottom, text="Clear", command=self.clear_text, width=15).pack(side="left", padx=5)

        self.status_var = tk.StringVar(value=f"Ready. Log file: {LOG_FILE}")
        tk.Label(self.root, textvariable=self.status_var, font=("Arial", 9),
                 bg="#1e1e1e", fg="gray").pack(pady=5)

    # ---------------- MODE START/STOP ----------------

    def start_realtime(self):
        if self.running:
            messagebox.showinfo("Already running", "Stop the current session first.")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Could not open the webcam.")
            return
        self.mode = "realtime"
        self._begin_session()

    def start_video_upload(self):
        if self.running:
            messagebox.showinfo("Already running", "Stop the current session first.")
            return
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("File error", "Could not open that video file.")
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_fps = fps if fps and fps > 1 else 30.0
        self.video_frame_index = 0
        self.mode = "video"
        self.status_var.set(
            "Tip: the video should show the person looking at the camera with "
            "eyes open for the first few seconds, used for calibration."
        )
        self._begin_session()

    def _begin_session(self):
        self.pipeline = BlinkMorsePipeline()
        self.running = True
        self.calibrating = True
        self.cal_start_time = None  # set on first real frame (video mode has no wall clock)
        self.cal_ear_readings = []
        self.cal_pitch_readings = []
        self.cal_yaw_readings = []
        self.buffer_var.set("")
        self.text_var.set("")
        self._update_frame()

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.pipeline and self.pipeline.decoded_text.strip():
            log_text(self.pipeline.decoded_text)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_var.set(f"Stopped. Log file: {LOG_FILE}")

    def clear_text(self):
        if self.pipeline:
            if self.pipeline.decoded_text.strip():
                log_text(self.pipeline.decoded_text)
            self.pipeline.clear_text()
            self.buffer_var.set("")
            self.text_var.set("")

    def speak_text(self):
        if self.pipeline and self.pipeline.decoded_text.strip():
            self.speech_engine.speak(self.pipeline.decoded_text)

    # ---------------- MAIN FRAME LOOP ----------------

    def _current_time(self):
        """Realtime mode uses the wall clock. Video mode uses the video's
        own internal clock (frame index / fps) so timing stays correct
        no matter how fast this computer processes the file."""
        if self.mode == "realtime":
            return time.time()
        return self.video_frame_index / self.video_fps

    def _update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            if self.mode == "video":
                self.status_var.set("Video finished.")
            else:
                self.status_var.set("Camera read failed.")
            self.stop()
            return

        if self.mode == "realtime":
            frame = cv2.flip(frame, 1)
        else:
            self.video_frame_index += 1

        now = self._current_time()

        if self.calibrating:
            self._handle_calibration_frame(frame, now)
        else:
            self._handle_decoding_frame(frame, now)

        self._render_frame(frame)

        delay_ms = 15 if self.mode == "realtime" else max(1, int(1000 / self.video_fps))
        self.root.after(delay_ms, self._update_frame)

    def _handle_calibration_frame(self, frame, now):
        if self.cal_start_time is None:
            self.cal_start_time = now

        elapsed = now - self.cal_start_time
        remaining = CALIBRATION_DURATION - elapsed

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pipeline.face_mesh.process(rgb)
        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            self.cal_ear_readings.append(average_ear(landmarks, w, h))
            pitch, yaw = get_head_metrics(landmarks, w, h)
            if pitch is not None:
                self.cal_pitch_readings.append(pitch)
                self.cal_yaw_readings.append(yaw)
            face_msg = "Face detected"
        else:
            face_msg = "No face -- center yourself in frame!"

        cv2.rectangle(frame, (0, 0), (w, 90), (40, 40, 40), -1)
        cv2.putText(frame, "CALIBRATING - look straight, keep eyes open",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Time remaining: {max(remaining, 0):.1f}s",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, face_msg, (15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        self.status_var.set(f"Calibrating... {max(remaining, 0):.1f}s remaining")

        if elapsed >= CALIBRATION_DURATION:
            self._finish_calibration()

    def _finish_calibration(self):
        if self.cal_ear_readings:
            baseline = sum(self.cal_ear_readings) / len(self.cal_ear_readings)
            ear_threshold = baseline * THRESHOLD_RATIO
        else:
            ear_threshold = 0.21  # fallback default

        pitch_baseline = yaw_baseline = None
        if self.cal_pitch_readings:
            pitch_baseline = sum(self.cal_pitch_readings) / len(self.cal_pitch_readings)
            yaw_baseline = sum(self.cal_yaw_readings) / len(self.cal_yaw_readings)

        self.pipeline.set_calibration(ear_threshold, pitch_baseline, yaw_baseline)
        self.calibrating = False
        self.status_var.set(f"Calibration done (threshold {ear_threshold:.3f}). Ready.")

    def _handle_decoding_frame(self, frame, now):
        events = self.pipeline.process_frame(frame, now)
        h, w, _ = frame.shape

        if events["ear"] is not None:
            cv2.putText(frame, f"EAR: {events['ear']:.3f}", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if events["pitch_delta"] is not None:
            cv2.putText(frame, f"Pitch: {events['pitch_delta']:+.2f}  Yaw: {events['yaw_delta']:+.2f}",
                        (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
        if not events["face_found"]:
            cv2.putText(frame, "No face found...", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Letter-commit progress bar
        progress = events["letter_progress"]
        if progress > 0:
            bar_x, bar_y, bar_w, bar_h = 15, 65, 250, 12
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
            fill_w = int(bar_w * progress)
            color = (0, 0, 255) if progress > 0.8 else (0, 200, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

        self.buffer_var.set(self.pipeline.morse_buffer)
        self.text_var.set(self.pipeline.decoded_text)

        if events["word_complete"]:
            log_text(self.pipeline.decoded_text)
            self.speech_engine.speak(events["word_complete"])

    def _render_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((640, 360))
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # keep a reference so it isn't garbage-collected


def main():
    root = tk.Tk()
    app = BlinkMorseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
