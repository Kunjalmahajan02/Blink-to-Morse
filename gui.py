"""
gui.py  (v2 -- HUD / covert-terminal visual theme)

Same functionality as before (real-time webcam + video-file upload,
both driven by pipeline.py), restyled to look like a covert-ops
monitoring terminal: dark background, neon green monospace text, HUD
corner brackets and a scanning sweep line drawn directly onto the
video feed, and a live REC indicator.

RUN: python gui.py
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

# --- Theme ---
BG = "#000000"
PANEL_BG = "#0a0a0a"
ACCENT = "#00ff66"        # neon green (RGB, for Tk widgets)
ACCENT_DIM = "#0a4d24"
DANGER = "#ff3b3b"
FONT = ("Consolas", 11)
FONT_BOLD = ("Consolas", 12, "bold")
FONT_TITLE = ("Consolas", 16, "bold")
FONT_MONO_TEXT = ("Consolas", 15, "bold")

# BGR colors for drawing directly on OpenCV frames
ACCENT_BGR = (102, 255, 0)
ACCENT_DIM_BGR = (40, 110, 0)
DANGER_BGR = (60, 60, 255)


class BlinkMorseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BLINK-CIPHER :: COVERT COMMS TERMINAL")
        self.root.geometry("920x780")
        self.root.configure(bg=BG)

        self.cap = None
        self.pipeline = None
        self.running = False
        self.mode = None
        self.video_fps = 30.0
        self.video_frame_index = 0
        self.frame_counter = 0
        self.session_start = None

        self.calibrating = False
        self.cal_start_time = None
        self.cal_ear_readings = []
        self.cal_pitch_readings = []
        self.cal_yaw_readings = []

        self._blink_on = False
        self.speech_engine = SpeechEngine()

        self._build_ui()
        self._blink_indicator()

    # ---------------- UI LAYOUT ----------------

    def _build_ui(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", pady=(12, 4))

        tk.Label(header, text="BLINK-CIPHER // COVERT COMMS TERMINAL",
                 font=FONT_TITLE, bg=BG, fg=ACCENT).pack(side="left", padx=20)

        self.indicator_var = tk.StringVar(value="STANDBY")
        self.indicator_label = tk.Label(header, textvariable=self.indicator_var,
                                         font=FONT_BOLD, bg=BG, fg="#666666")
        self.indicator_label.pack(side="right", padx=20)

        controls = tk.Frame(self.root, bg=BG)
        controls.pack(pady=8)

        self._make_button(controls, "[ START REAL-TIME SCAN ]", self.start_realtime, ACCENT).pack(side="left", padx=6)
        self._make_button(controls, "[ UPLOAD INTERCEPT FILE ]", self.start_video_upload, ACCENT).pack(side="left", padx=6)
        self._make_button(controls, "[ TERMINATE ]", self.stop, DANGER).pack(side="left", padx=6)

        video_border = tk.Frame(self.root, bg=ACCENT, padx=2, pady=2)
        video_border.pack(pady=14)
        self.video_label = tk.Label(video_border, bg="black", width=640, height=360)
        self.video_label.pack()

        readout = tk.Frame(self.root, bg=PANEL_BG, highlightbackground=ACCENT_DIM,
                            highlightthickness=1)
        readout.pack(pady=10, fill="x", padx=40)

        tk.Label(readout, text="BUFFER>", font=FONT_BOLD, bg=PANEL_BG, fg=ACCENT
                 ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 2))
        self.buffer_var = tk.StringVar(value="")
        tk.Label(readout, textvariable=self.buffer_var, font=FONT_MONO_TEXT,
                 bg=PANEL_BG, fg="#b6ffb6").grid(row=0, column=1, sticky="w", pady=(10, 2))

        tk.Label(readout, text="DECODED>", font=FONT_BOLD, bg=PANEL_BG, fg=ACCENT
                 ).grid(row=1, column=0, sticky="nw", padx=10, pady=(2, 10))
        self.text_var = tk.StringVar(value="")
        tk.Label(readout, textvariable=self.text_var, font=FONT_MONO_TEXT, bg=PANEL_BG,
                 fg="#ffe066", wraplength=680, justify="left"
                 ).grid(row=1, column=1, sticky="w", pady=(2, 10))

        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(pady=12)
        self._make_button(bottom, "[ SPEAK TRANSMISSION ]", self.speak_text, ACCENT).pack(side="left", padx=6)
        self._make_button(bottom, "[ WIPE BUFFER ]", self.clear_text, ACCENT).pack(side="left", padx=6)

        self.status_var = tk.StringVar(value=f">>> SYSTEM IDLE. LOG: {LOG_FILE}")
        tk.Label(self.root, textvariable=self.status_var, font=("Consolas", 9),
                 bg=BG, fg="#3d8b52").pack(pady=(0, 10))

    def _make_button(self, parent, text, command, color):
        return tk.Button(
            parent, text=text, command=command, font=FONT_BOLD,
            bg="black", fg=color, activebackground="#001a00", activeforeground=color,
            relief="flat", bd=0, highlightbackground=color, highlightthickness=1,
            padx=10, pady=8, cursor="hand2",
        )

    def _blink_indicator(self):
        self._blink_on = not self._blink_on
        if self.running:
            self.indicator_label.configure(fg=ACCENT if self._blink_on else ACCENT_DIM)
            self.indicator_var.set("LIVE")
        else:
            self.indicator_label.configure(fg="#666666")
            self.indicator_var.set("STANDBY")
        self.root.after(500, self._blink_indicator)

    # ---------------- MODE START/STOP ----------------

    def start_realtime(self):
        if self.running:
            messagebox.showinfo("Already running", "Terminate the current session first.")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Could not open the webcam.")
            return
        self.mode = "realtime"
        self._begin_session()

    def start_video_upload(self):
        if self.running:
            messagebox.showinfo("Already running", "Terminate the current session first.")
            return
        path = filedialog.askopenfilename(
            title="Select intercepted video file",
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
        self.status_var.set(">>> TIP: subject should face camera, eyes open, for first ~5s (calibration).")
        self._begin_session()

    def _begin_session(self):
        self.pipeline = BlinkMorsePipeline()
        self.running = True
        self.calibrating = True
        self.cal_start_time = None
        self.cal_ear_readings = []
        self.cal_pitch_readings = []
        self.cal_yaw_readings = []
        self.frame_counter = 0
        self.session_start = time.time()
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
        self.status_var.set(f">>> SESSION TERMINATED. LOG: {LOG_FILE}")

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
        if self.mode == "realtime":
            return time.time()
        return self.video_frame_index / self.video_fps

    def _update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set(">>> VIDEO FINISHED." if self.mode == "video" else ">>> CAMERA READ FAILED.")
            self.stop()
            return

        if self.mode == "realtime":
            frame = cv2.flip(frame, 1)
        else:
            self.video_frame_index += 1

        self.frame_counter += 1
        now = self._current_time()

        if self.calibrating:
            self._handle_calibration_frame(frame, now)
        else:
            self._handle_decoding_frame(frame, now)

        self._draw_hud_chrome(frame)
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
            face_msg = "TARGET LOCKED"
        else:
            face_msg = "NO TARGET -- CENTER IN FRAME"

        cv2.rectangle(frame, (0, 0), (w, 85), (10, 10, 10), -1)
        cv2.putText(frame, "CALIBRATING BIOMETRIC BASELINE", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, ACCENT_BGR, 2)
        cv2.putText(frame, f"T-MINUS {max(remaining, 0):.1f}s", (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, face_msg, (15, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT_BGR, 1)

        self.status_var.set(f">>> CALIBRATING... T-MINUS {max(remaining, 0):.1f}s")

        if elapsed >= CALIBRATION_DURATION:
            self._finish_calibration()

    def _finish_calibration(self):
        if self.cal_ear_readings:
            baseline = sum(self.cal_ear_readings) / len(self.cal_ear_readings)
            ear_threshold = baseline * THRESHOLD_RATIO
        else:
            ear_threshold = 0.21

        pitch_baseline = yaw_baseline = None
        if self.cal_pitch_readings:
            pitch_baseline = sum(self.cal_pitch_readings) / len(self.cal_pitch_readings)
            yaw_baseline = sum(self.cal_yaw_readings) / len(self.cal_yaw_readings)

        self.pipeline.set_calibration(ear_threshold, pitch_baseline, yaw_baseline)
        self.calibrating = False
        self.status_var.set(f">>> BASELINE LOCKED (threshold {ear_threshold:.3f}). DECODING ACTIVE.")

    def _handle_decoding_frame(self, frame, now):
        events = self.pipeline.process_frame(frame, now)

        if events["ear"] is not None:
            cv2.putText(frame, f"EAR {events['ear']:.3f}", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT_BGR, 2)
        if events["pitch_delta"] is not None:
            cv2.putText(frame, f"PITCH {events['pitch_delta']:+.2f}  YAW {events['yaw_delta']:+.2f}",
                        (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        if not events["face_found"]:
            cv2.putText(frame, "NO TARGET", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, DANGER_BGR, 2)

        progress = events["letter_progress"]
        if progress > 0:
            h, w, _ = frame.shape
            bar_x, bar_y, bar_w, bar_h = 15, 62, 220, 10
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1)
            fill_w = int(bar_w * progress)
            color = DANGER_BGR if progress > 0.8 else ACCENT_BGR
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

        self.buffer_var.set(self.pipeline.morse_buffer)
        self.text_var.set(self.pipeline.decoded_text)

        if events["word_complete"]:
            log_text(self.pipeline.decoded_text)
            self.speech_engine.speak(events["word_complete"])

    def _draw_hud_chrome(self, frame):
        """Draws the spy-HUD overlay: corner brackets, a sweeping scan
        line, and a REC indicator. Purely cosmetic -- doesn't affect
        detection, which already ran on the frame before this is called."""
        h, w, _ = frame.shape
        L = 28
        color = ACCENT_BGR
        thick = 2

        cv2.line(frame, (5, 5), (5, 5 + L), color, thick)
        cv2.line(frame, (5, 5), (5 + L, 5), color, thick)
        cv2.line(frame, (w - 5, 5), (w - 5, 5 + L), color, thick)
        cv2.line(frame, (w - 5, 5), (w - 5 - L, 5), color, thick)
        cv2.line(frame, (5, h - 5), (5, h - 5 - L), color, thick)
        cv2.line(frame, (5, h - 5), (5 + L, h - 5), color, thick)
        cv2.line(frame, (w - 5, h - 5), (w - 5, h - 5 - L), color, thick)
        cv2.line(frame, (w - 5, h - 5), (w - 5 - L, h - 5), color, thick)

        scan_y = (self.frame_counter * 3) % h
        overlay = frame.copy()
        cv2.line(overlay, (0, scan_y), (w, scan_y), color, 1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        rec_on = (self.frame_counter // 15) % 2 == 0
        rec_color = DANGER_BGR if rec_on else (60, 60, 60)
        cv2.circle(frame, (w - 90, 20), 5, rec_color, -1)
        cv2.putText(frame, "REC", (w - 78, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        src_label = "WEBCAM" if self.mode == "realtime" else "FILE"
        cv2.putText(frame, f"SRC:{src_label}", (w - 130, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def _render_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((640, 360))
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=photo)
        self.video_label.image = photo


def main():
    root = tk.Tk()
    app = BlinkMorseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
