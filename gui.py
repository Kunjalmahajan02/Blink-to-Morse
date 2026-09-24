"""
gui.py  (PyQt6 version)

A more visually polished desktop application than the earlier Tkinter
version -- same functionality (real-time webcam + video-file upload,
both driven by pipeline.py), but PyQt6 lets us add real neon glow
(drop-shadow effects) and a smooth pulsing "LIVE" animation instead of
a blunt on/off blink.

NOTE: none of the detection logic changed at all -- pipeline.py,
blink_detector.py, head_gesture.py, speech.py, logger.py are reused
exactly as they were. Only this display/window layer is new.

RUN: python gui.py
"""

import time

import cv2
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFrame, QFileDialog, QMessageBox, QGraphicsDropShadowEffect, QGraphicsOpacityEffect,
    QComboBox,
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont

from pipeline import BlinkMorsePipeline
from blink_detector import average_ear
from head_gesture import get_head_metrics
from speech import SpeechEngine
from translator import translate, SUPPORTED_LANGUAGES, INDIAN_LANGUAGE_NAMES
from logger import log_text, LOG_FILE
from forensic_mode import ForensicWindow

CALIBRATION_DURATION = 5.0
THRESHOLD_RATIO = 0.75

ACCENT = "#00ff66"
DANGER = "#ff3b3b"

# BGR colors for drawing directly on OpenCV frames (unchanged from before)
ACCENT_BGR = (102, 255, 0)
DANGER_BGR = (60, 60, 255)

STYLE_SHEET = f"""
QWidget {{
    background-color: #000000;
    color: {ACCENT};
    font-family: Consolas;
}}
QPushButton {{
    background-color: #001a0d;
    color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: bold;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: #00331a;
    border: 1px solid #55ffaa;
}}
QPushButton:pressed {{
    background-color: #004d26;
}}
QFrame#videoFrame {{
    border: 2px solid {ACCENT};
    border-radius: 12px;
    background-color: #000000;
}}
QFrame#readoutPanel {{
    background-color: #0a0a0a;
    border: 1px solid #0a4d24;
    border-radius: 10px;
}}
"""

TERMINATE_STYLE = f"""
QPushButton {{
    background-color: #1a0000;
    color: {DANGER};
    border: 1px solid {DANGER};
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: bold;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: #330000;
}}
"""


def glow(widget, color=ACCENT, radius=20):
    """Applies a neon drop-shadow glow effect to any widget."""
    effect = QGraphicsDropShadowEffect()
    effect.setColor(QColor(color))
    effect.setBlurRadius(radius)
    effect.setOffset(0, 0)
    widget.setGraphicsEffect(effect)
    return effect


class BlinkMorseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BLINK-CIPHER :: COVERT COMMS TERMINAL")
        self.resize(960, 800)
        self.setStyleSheet(STYLE_SHEET)

        self.cap = None
        self.pipeline = None
        self.running = False
        self.mode = None
        self.video_fps = 30.0
        self.video_frame_index = 0
        self.frame_counter = 0

        self.calibrating = False
        self.cal_start_time = None
        self.cal_ear_readings = []
        self.cal_pitch_readings = []
        self.cal_yaw_readings = []

        self.speech_engine = SpeechEngine()
        self._last_translation_time = 0
        self._last_translated_text = ""

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        self._build_ui()

    # ---------------- UI LAYOUT ----------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(24, 20, 24, 20)

        # --- Header ---
        header = QHBoxLayout()
        title = QLabel("BLINK-CIPHER // COVERT COMMS TERMINAL")
        title.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
        glow(title, ACCENT, 25)
        header.addWidget(title)
        header.addStretch()

        self.indicator = QLabel("● STANDBY")
        self.indicator.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        self.indicator.setStyleSheet("color: #666666;")
        header.addWidget(self.indicator)
        root.addLayout(header)

        self._indicator_opacity = QGraphicsOpacityEffect()
        self.indicator.setGraphicsEffect(self._indicator_opacity)
        self._pulse_anim = QPropertyAnimation(self._indicator_opacity, b"opacity")
        self._pulse_anim.setDuration(1000)
        self._pulse_anim.setKeyValueAt(0.0, 1.0)
        self._pulse_anim.setKeyValueAt(0.5, 0.3)
        self._pulse_anim.setKeyValueAt(1.0, 1.0)
        self._pulse_anim.setLoopCount(-1)

        # --- Controls ---
        controls = QHBoxLayout()
        self.start_btn = QPushButton("[ START REAL-TIME SCAN ]")
        self.upload_btn = QPushButton("[ UPLOAD INTERCEPT FILE ]")
        self.forensic_btn = QPushButton("[ FORENSIC ANALYSIS MODE ]")
        self.stop_btn = QPushButton("[ TERMINATE ]")
        self.stop_btn.setStyleSheet(TERMINATE_STYLE)

        for btn in (self.start_btn, self.upload_btn, self.forensic_btn, self.stop_btn):
            glow(btn, ACCENT if btn is not self.stop_btn else DANGER, 15)

        self.start_btn.clicked.connect(self.start_realtime)
        self.upload_btn.clicked.connect(self.start_video_upload)
        self.forensic_btn.clicked.connect(self.open_forensic_mode)
        self.stop_btn.clicked.connect(self.stop)

        controls.addWidget(self.start_btn)
        controls.addWidget(self.upload_btn)
        controls.addWidget(self.forensic_btn)
        controls.addWidget(self.stop_btn)
        root.addLayout(controls)

        # --- Video panel ---
        video_frame = QFrame()
        video_frame.setObjectName("videoFrame")
        glow(video_frame, ACCENT, 30)
        video_layout = QVBoxLayout(video_frame)
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 360)
        self.video_label.setStyleSheet("border: none; background-color: black;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)
        root.addWidget(video_frame, alignment=Qt.AlignmentFlag.AlignCenter)

        # --- Readout panel ---
        readout = QFrame()
        readout.setObjectName("readoutPanel")
        readout_layout = QVBoxLayout(readout)

        buf_row = QHBoxLayout()
        buf_label = QLabel("BUFFER>")
        buf_label.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        self.buffer_val = QLabel("")
        self.buffer_val.setFont(QFont("Consolas", 15, QFont.Weight.Bold))
        self.buffer_val.setStyleSheet("color: #b6ffb6;")
        buf_row.addWidget(buf_label)
        buf_row.addWidget(self.buffer_val)
        buf_row.addStretch()
        readout_layout.addLayout(buf_row)

        text_row = QHBoxLayout()
        text_label = QLabel("DECODED>")
        text_label.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        self.text_val = QLabel("")
        self.text_val.setFont(QFont("Consolas", 15, QFont.Weight.Bold))
        self.text_val.setStyleSheet("color: #ffe066;")
        self.text_val.setWordWrap(True)
        text_row.addWidget(text_label)
        text_row.addWidget(self.text_val, stretch=1)
        readout_layout.addLayout(text_row)

        # --- Language selector + translated text row ---
        lang_row = QHBoxLayout()
        lang_lbl = QLabel("LANGUAGE>")
        lang_lbl.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        self.lang_combo = QComboBox()
        self.lang_combo.setStyleSheet(
            "QComboBox { background-color: #0a0a0a; color: #00ff66; border: 1px solid #0a4d24;"
            " border-radius: 6px; padding: 4px; font-family: Consolas; font-size: 11px; }"
            "QComboBox QAbstractItemView { background-color: #0a0a0a; color: #00ff66; }"
        )
        # Indian languages first, then a divider, then international
        for name in INDIAN_LANGUAGE_NAMES:
            self.lang_combo.addItem(f"🇮🇳 {name}", SUPPORTED_LANGUAGES[name])
        self.lang_combo.insertSeparator(len(INDIAN_LANGUAGE_NAMES))
        for name, code in SUPPORTED_LANGUAGES.items():
            if name not in INDIAN_LANGUAGE_NAMES:
                self.lang_combo.addItem(name, code)
        # Default to English (original) -- no translation
        idx = self.lang_combo.findText("English (original)")
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        lang_row.addWidget(lang_lbl)
        lang_row.addWidget(self.lang_combo)
        lang_row.addStretch()
        readout_layout.addLayout(lang_row)

        trans_row = QHBoxLayout()
        trans_lbl = QLabel("TRANSLATED>")
        trans_lbl.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        self.trans_val = QLabel("")
        self.trans_val.setFont(QFont("Consolas", 15, QFont.Weight.Bold))
        self.trans_val.setStyleSheet("color: #66ccff;")
        self.trans_val.setWordWrap(True)
        trans_row.addWidget(trans_lbl)
        trans_row.addWidget(self.trans_val, stretch=1)
        readout_layout.addLayout(trans_row)

        root.addWidget(readout)

        # --- Bottom buttons ---
        bottom = QHBoxLayout()
        self.speak_btn = QPushButton("[ SPEAK TRANSMISSION ]")
        self.clear_btn = QPushButton("[ WIPE BUFFER ]")
        glow(self.speak_btn, ACCENT, 15)
        glow(self.clear_btn, ACCENT, 15)
        self.speak_btn.clicked.connect(self.speak_text)
        self.clear_btn.clicked.connect(self.clear_text)
        bottom.addWidget(self.speak_btn)
        bottom.addWidget(self.clear_btn)
        root.addLayout(bottom)

        # --- Status bar ---
        self.status_label = QLabel(f">>> SYSTEM IDLE. LOG: {LOG_FILE}")
        self.status_label.setStyleSheet("color: #3d8b52; font-size: 10px;")
        root.addWidget(self.status_label)

    def _set_indicator(self, live):
        if live:
            self.indicator.setText("● LIVE")
            self.indicator.setStyleSheet(f"color: {ACCENT};")
            self._pulse_anim.start()
        else:
            self._pulse_anim.stop()
            self._indicator_opacity.setOpacity(1.0)
            self.indicator.setText("● STANDBY")
            self.indicator.setStyleSheet("color: #666666;")

    # ---------------- MODE START/STOP ----------------

    def open_forensic_mode(self):
        # Keep a reference on self so Python doesn't garbage-collect
        # the window the moment this method returns.
        self._forensic_window = ForensicWindow()
        self._forensic_window.show()

    def start_realtime(self):
        if self.running:
            QMessageBox.information(self, "Already running", "Terminate the current session first.")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera error", "Could not open the webcam.")
            return
        self.mode = "realtime"
        self._begin_session()

    def start_video_upload(self):
        if self.running:
            QMessageBox.information(self, "Already running", "Terminate the current session first.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select intercepted video file", "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)",
        )
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "File error", "Could not open that video file.")
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_fps = fps if fps and fps > 1 else 30.0
        self.video_frame_index = 0
        self.mode = "video"
        self.status_label.setText(">>> TIP: subject should face camera, eyes open, for first ~5s.")
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
        self.buffer_val.setText("")
        self.text_val.setText("")
        self._set_indicator(True)

        interval = 15 if self.mode == "realtime" else max(1, int(1000 / self.video_fps))
        self.timer.start(interval)

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.timer.stop()
        if self.pipeline and self.pipeline.decoded_text.strip():
            log_text(self.pipeline.decoded_text)
        if self.cap:
            self.cap.release()
            self.cap = None
        self._set_indicator(False)
        self.status_label.setText(f">>> SESSION TERMINATED. LOG: {LOG_FILE}")

    def clear_text(self):
        if self.pipeline:
            if self.pipeline.decoded_text.strip():
                log_text(self.pipeline.decoded_text)
            self.pipeline.clear_text()
            self.buffer_val.setText("")
            self.text_val.setText("")

    def _current_lang_code(self):
        return self.lang_combo.currentData() or "en"

    def _on_language_changed(self):
        """Only translate when the user actively picks a language."""
        if self.pipeline and self.pipeline.decoded_text.strip():
            self._update_translation(self.pipeline.decoded_text)
        else:
            self.trans_val.setText("")

    def _update_translation(self, decoded_text):
        """Translates decoded_text and updates the TRANSLATED> row."""
        import time
        lang_code = self._current_lang_code()
        if lang_code in ("en", "en-US"):
            self.trans_val.setText("")
            return
        # Skip if nothing new to translate
        if not decoded_text.strip():
            return
        now = time.time()
        if (decoded_text == self._last_translated_text
                and now - self._last_translation_time < 2.0):
            return
        self._last_translation_time = now
        self._last_translated_text = decoded_text
        self.trans_val.setText("Translating...")
        self.trans_val.setStyleSheet("color: #888888;")
        translated, ok, err = translate(decoded_text, lang_code)
        if ok:
            self.trans_val.setText(translated)
            self.trans_val.setStyleSheet("color: #66ccff;")
        else:
            self.trans_val.setText(f"[Error: {err}]")
            self.trans_val.setStyleSheet("color: #ff9933;")

    def speak_text(self):
        if not (self.pipeline and self.pipeline.decoded_text.strip()):
            return
        lang_code = self._current_lang_code()
        if lang_code != "en":
            # Translate first, then speak
            self._update_translation(self.pipeline.decoded_text)
            text_to_speak = self.trans_val.text()
            if not text_to_speak or text_to_speak.startswith("[") or text_to_speak == "Translating...":
                text_to_speak = self.pipeline.decoded_text
        else:
            text_to_speak = self.pipeline.decoded_text
        self.speech_engine.speak(text_to_speak, lang_code=lang_code)

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
            self.status_label.setText(">>> VIDEO FINISHED." if self.mode == "video" else ">>> CAMERA READ FAILED.")
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

        self.status_label.setText(f">>> CALIBRATING... T-MINUS {max(remaining, 0):.1f}s")

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
        self.status_label.setText(f">>> BASELINE LOCKED (threshold {ear_threshold:.3f}). DECODING ACTIVE.")

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
            bar_x, bar_y, bar_w, bar_h = 15, 62, 220, 10
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1)
            fill_w = int(bar_w * progress)
            color = DANGER_BGR if progress > 0.8 else ACCENT_BGR
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

        self.buffer_val.setText(self.pipeline.morse_buffer)
        self.text_val.setText(self.pipeline.decoded_text)

        if events["word_complete"]:
            log_text(self.pipeline.decoded_text)
            self.speech_engine.speak(events["word_complete"])

    def _draw_hud_chrome(self, frame):
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
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            640, 360, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.stop()
        event.accept()


def main():
    app = QApplication([])
    window = BlinkMorseApp()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
