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
import mediapipe as mp
import orientation
import video_metadata
from blink_detector import LEFT_EYE, RIGHT_EYE
from hud_widgets import (EmblemWidget, LedIndicator, SignalGraph, MorseTape, HudPanel,
                         StatusTicker, make_standby_frame, draw_target_lock, paint_grid_background,
                         StandbyAnimator, TypewriterTicker, make_emblem_icon, apply_cinematic)
from codebook import CODEBOOK
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFrame, QFileDialog, QMessageBox, QGraphicsDropShadowEffect, QGraphicsOpacityEffect,
    QComboBox, QCheckBox,
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont, QPainter

from pipeline import BlinkMorsePipeline
from blink_detector import average_ear
from head_gesture import get_head_metrics
from speech import SpeechEngine
from translator import translate, SUPPORTED_LANGUAGES, INDIAN_LANGUAGE_NAMES
import offline_translator
from logger import log_text, LOG_FILE
from forensic_mode import ForensicWindow

CALIBRATION_DURATION = 3.0
THRESHOLD_RATIO = 0.75

# All colours, fonts and widget styles now come from theme.py (design only).
from theme import (ACCENT, DANGER, ACCENT_BGR, DANGER_BGR, STYLE_SHEET, TERMINATE_STYLE, COMBO_STYLE,
                   TITLE_STYLE, SUBTITLE_STYLE, BANNER_STYLE, TICKER_STYLE)


def glow(widget, color=ACCENT, radius=20):
    """Applies a neon drop-shadow glow effect to any widget."""
    effect = QGraphicsDropShadowEffect()
    effect.setColor(QColor(color))
    effect.setBlurRadius(radius)
    effect.setOffset(0, 0)
    widget.setGraphicsEffect(effect)
    return effect


def _still_face_mesh():
    """Face detector for independent sample frames (orientation check)."""
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                           refine_landmarks=True, min_detection_confidence=0.5)


class BlinkMorseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BLINK-CIPHER :: COVERT COMMS TERMINAL")
        self.resize(1260, 820)
        self.setStyleSheet(STYLE_SHEET)
        self.setWindowIcon(make_emblem_icon())

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

        self._fps = 0.0
        self._last_frame_wall = None
        self.session_start = None
        self._build_ui()
        self._update_offline_badge()
        self._standby = StandbyAnimator()
        self._standby_timer = QTimer(self)
        self._standby_timer.timeout.connect(self._tick_standby)
        self._standby_timer.start(80)
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_clock()

    # ---------------- UI LAYOUT ----------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(24, 12, 24, 12)
        root.addWidget(self._make_banner())

        # --- Header ---
        header = QHBoxLayout()
        header.addWidget(EmblemWidget(58))
        header.addSpacing(8)
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("BLINK-CIPHER // COVERT COMMS TERMINAL")
        title.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
        title.setStyleSheet(TITLE_STYLE)
        glow(title, ACCENT, 25)
        subtitle = QLabel("CLASSIFIED  //  AUTHORIZED PERSONNEL ONLY  //  OCULAR SIGNAL INTERCEPT SYSTEM")
        subtitle.setStyleSheet(SUBTITLE_STYLE)
        self.ticker = TypewriterTicker(self._ticker_messages)
        self.ticker.setStyleSheet(TICKER_STYLE)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        title_box.addWidget(self.ticker)
        header.addLayout(title_box)
        header.addStretch()
        self.clock_label = QLabel("")
        self.clock_label.setStyleSheet("color: #7fdc9a; font-size: 11px;")
        header.addWidget(self.clock_label)
        header.addSpacing(20)

        self.indicator = QLabel("● STANDBY")
        self.indicator.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        self.indicator.setStyleSheet("color: #666666;")
        self.offline_badge = QLabel("")
        self.offline_badge.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        header.addWidget(self.offline_badge)
        header.addSpacing(20)
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

        middle = QHBoxLayout()
        middle.setSpacing(16)
        middle.addWidget(self._build_status_panel())
        middle.addWidget(video_frame, alignment=Qt.AlignmentFlag.AlignCenter)
        middle.addWidget(self._build_reference_panel())
        root.addLayout(middle)

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
        buf_row.addSpacing(16)
        self.morse_tape = MorseTape()
        buf_row.addWidget(self.morse_tape)
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
        self.lang_combo.setStyleSheet(COMBO_STYLE)
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
        lang_row.addSpacing(30)
        light_lbl = QLabel("LOW-LIGHT>")
        light_lbl.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        self.light_combo = QComboBox()
        self.light_combo.setStyleSheet(self.lang_combo.styleSheet())
        self.light_combo.addItem("Auto (enhance when dark)", "auto")
        self.light_combo.addItem("Off", "off")
        self.light_combo.addItem("Always on", "on")
        self.light_combo.currentIndexChanged.connect(self._on_light_mode_changed)
        lang_row.addWidget(light_lbl)
        lang_row.addWidget(self.light_combo)
        lang_row.addSpacing(30)
        self.offline_check = QCheckBox("OFFLINE MODE")
        self.offline_check.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        self.offline_check.setChecked(True)   # secure default: nothing leaves the computer
        self.offline_check.setToolTip(
            "On: translation uses the local phrasebook + transliteration, speech uses local voices.\n"
            "Off: allows online translation (MyMemory) and online speech (Google). "
            "The decoded message is then sent to those servers.")
        self.offline_check.stateChanged.connect(self._on_offline_changed)
        lang_row.addWidget(self.offline_check)
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
        self.status_label = StatusTicker(f">>> SYSTEM IDLE. LOG: {LOG_FILE}")
        self.status_label.setStyleSheet("color: #3d8b52; font-size: 10px;")
        root.addWidget(self.status_label)
        root.addWidget(self._make_banner())

    # ---------------- DECORATIVE / DISPLAY-ONLY ELEMENTS ----------------

    def paintEvent(self, event):
        """Subtle grid behind the whole window (display only)."""
        painter = QPainter(self)
        paint_grid_background(self, painter)
        painter.end()

    def _build_status_panel(self):
        panel = HudPanel("SYSTEM STATUS")
        panel.setFixedWidth(215)
        self.leds = {}
        for key, label in [("camera", "SIGNAL SOURCE"), ("face", "FACE LOCK"), ("eyes", "EYES"),
                           ("night", "NIGHT MODE"), ("offline", "OFFLINE MODE"), ("gestures", "HEAD GESTURES")]:
            led = LedIndicator(label)
            self.leds[key] = led
            panel.body.addWidget(led)
        panel.body.addSpacing(6)
        self.telemetry_label = QLabel("")
        self.telemetry_label.setStyleSheet("color: #7fdc9a; font-size: 10px; background: transparent; border: none;")
        panel.body.addWidget(self.telemetry_label)
        panel.body.addSpacing(6)
        graph_caption = QLabel("// EAR SIGNAL  (dips = blinks)")
        graph_caption.setStyleSheet("color: #00ff66; font-size: 10px; background: transparent; border: none;")
        panel.body.addWidget(graph_caption)
        self.signal_graph = SignalGraph()
        panel.body.addWidget(self.signal_graph)
        panel.body.addStretch()
        self._update_telemetry(None, None, None)
        return panel

    def _build_reference_panel(self):
        panel = HudPanel("MORSE REFERENCE")
        panel.setFixedWidth(215)
        letters = [("A", ".-"), ("B", "-..."), ("C", "-.-."), ("D", "-.."), ("E", "."), ("F", "..-."),
                   ("G", "--."), ("H", "...."), ("I", ".."), ("J", ".---"), ("K", "-.-"), ("L", ".-.."),
                   ("M", "--"), ("N", "-."), ("O", "---"), ("P", ".--."), ("Q", "--.-"), ("R", ".-."),
                   ("S", "..."), ("T", "-"), ("U", "..-"), ("V", "...-"), ("W", ".--"), ("X", "-..-"),
                   ("Y", "-.--"), ("Z", "--..")]
        half = (len(letters) + 1) // 2
        lines = []
        for i in range(half):
            left = f"{letters[i][0]} {letters[i][1]:<5}"
            right = f"{letters[i + half][0]} {letters[i + half][1]}" if i + half < len(letters) else ""
            lines.append(f"{left}   {right}")
        chart = QLabel("\n".join(lines))
        chart.setFont(QFont("Consolas", 9))
        chart.setStyleSheet("color: #b6ffb6; background: transparent; border: none;")
        panel.body.addWidget(chart)

        cb_caption = QLabel("// CODEBOOK (one group)")
        cb_caption.setStyleSheet("color: #00ff66; font-size: 10px; background: transparent; border: none;")
        panel.body.addSpacing(4)
        panel.body.addWidget(cb_caption)
        cb = QLabel("\n".join(f"{pat}  {phrase}" for pat, phrase in CODEBOOK.items()))
        cb.setFont(QFont("Consolas", 8))
        cb.setStyleSheet("color: #ffe066; background: transparent; border: none;")
        panel.body.addWidget(cb)

        panic = QLabel("// DISTRESS: 5 holds of 1-2 s")
        panic.setStyleSheet("color: #ff6b6b; font-size: 10px; background: transparent; border: none;")
        panel.body.addSpacing(4)
        panel.body.addWidget(panic)
        panel.body.addStretch()
        return panel

    def _make_banner(self):
        banner = QLabel("RESTRICTED   ///   OCULAR COMMUNICATION INTERCEPT   ///   AUTHORIZED PERSONNEL ONLY")
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        banner.setStyleSheet(BANNER_STYLE)
        return banner

    def _ticker_messages(self):
        if not hasattr(self, "offline_check"):
            return ["INITIALISING..."]
        return [
            "SIGNAL ACQUIRED // DECODING IN PROGRESS" if self.running
            else "OCULAR MORSE INTERCEPT SYSTEM // STANDING BY",
            "SHORT BLINK = DOT   //   LONG BLINK = DASH",
            "FIVE HOLDS OF 1-2 SECONDS = DISTRESS SIGNAL",
            "SECURE OFFLINE CHANNEL // NO DATA LEAVES THIS TERMINAL" if self.offline_check.isChecked()
            else "WARNING // ONLINE TRANSLATION ENABLED",
        ]

    def _tick_standby(self):
        if not self.running:
            self._render_frame(self._standby.render(time.time()))

    def _timecode(self):
        if self.mode == "video":
            return self.video_frame_index / self.video_fps
        return time.time() - self.session_start if self.session_start else 0.0

    def _source_tag(self):
        return "SRC:WEBCAM" if self.mode == "realtime" else "SRC:FILE"

    def _update_clock(self):
        clock = time.strftime("%d %b %Y  %H:%M:%S")
        if self.session_start is not None:
            elapsed = int(time.time() - self.session_start)
            clock += f"   |   SESSION {elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
        self.clock_label.setText(clock)

    def _update_telemetry(self, ear, threshold, light):
        fps = f"{self._fps:5.1f}" if self.running else "  ---"
        ear_txt = f"{ear:.3f}" if ear is not None else "---"
        thr_txt = f"{threshold:.3f}" if threshold is not None else "---"
        light_txt = f"{light['brightness']:.0f}/255" if light else "---"
        self.telemetry_label.setText(f"FPS    {fps}\nEAR    {ear_txt}\nTHRESH {thr_txt}\nLIGHT  {light_txt}")

    def _update_status_leds(self, face_found=None, light=None):
        """Refresh the status lights from what the app already knows."""
        if not hasattr(self, "leds"):
            return
        leds = self.leds
        if self.running:
            leds["camera"].set_state("on", "WEBCAM" if self.mode == "realtime" else "FILE")
        else:
            leds["camera"].set_state("off", "IDLE")
        if not self.running or face_found is None:
            leds["face"].set_state("off")
            leds["eyes"].set_state("off")
        elif face_found:
            leds["face"].set_state("on", "LOCKED")
            closed = bool(self.pipeline and self.pipeline.eye_closed)
            leds["eyes"].set_state("warn" if closed else "on", "CLOSED" if closed else "OPEN")
        else:
            leds["face"].set_state("alert", "LOST")
            leds["eyes"].set_state("off")
        if light and light.get("applied"):
            leds["night"].set_state("on", "ACTIVE")
        else:
            leds["night"].set_state("off", "STANDBY" if self.running else "")
        if self.offline_check.isChecked():
            leds["offline"].set_state("on", "SECURE")
        else:
            leds["offline"].set_state("warn", "ONLINE")
        if self.running and self.pipeline is not None:
            on = self.pipeline.gestures_enabled
            leds["gestures"].set_state("on" if on else "off", "ENABLED" if on else "OFF")
        else:
            leds["gestures"].set_state("off")

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
        self.video_rotation = None
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
        self.status_label.setText(">>> CHECKING VIDEO ORIENTATION...")
        QApplication.processEvents()
        self.video_rotation, rot_label, _ = orientation.detect_rotation(self.cap, _still_face_mesh)
        rot_msg = f"ROTATED {rot_label} DEG. " if self.video_rotation is not None else ""
        meta = video_metadata.read_metadata(path)
        self.status_label.setText(f">>> {rot_msg}FILE METADATA (unverified): {video_metadata.format_summary(meta)}")
        log_text(f"[VIDEO LOADED] {path} | FILE METADATA (unverified): {video_metadata.format_summary(meta)}")
        self._begin_session()

    def _begin_session(self):
        # Head gestures only for a live, cooperating user: in recorded
        # footage, natural head movement would delete letters.
        self.pipeline = BlinkMorsePipeline(enhance_mode=self.light_combo.currentData(),
                                           gestures_enabled=(self.mode == "realtime"))
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
        self.session_start = time.time()
        self._last_frame_wall = None
        self._fps = 0.0
        self.signal_graph.clear()
        self.morse_tape.set_pattern("")
        self._update_status_leds()

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
        self.session_start = None
        self._update_clock()
        self.morse_tape.set_pattern("")
        self._update_telemetry(None, None, None)
        self._update_status_leds()
        self._standby.subtitle = "SESSION TERMINATED  //  SELECT A SIGNAL SOURCE"

    def clear_text(self):
        if self.pipeline:
            if self.pipeline.decoded_text.strip():
                log_text(self.pipeline.decoded_text)
            self.pipeline.clear_text()
            self.buffer_val.setText("")
            self.text_val.setText("")
            self.morse_tape.set_pattern("")

    def _on_light_mode_changed(self):
        if self.pipeline:
            self.pipeline.enhance_mode = self.light_combo.currentData()

    def _current_lang_code(self):
        return self.lang_combo.currentData() or "en"

    def _is_offline(self):
        return self.offline_check.isChecked()

    def _update_offline_badge(self):
        self._update_status_leds()
        if self._is_offline():
            self.offline_badge.setText("OFFLINE: NOTHING LEAVES THIS COMPUTER")
            self.offline_badge.setStyleSheet(f"color: {ACCENT};")
        else:
            self.offline_badge.setText("ONLINE TRANSLATION ENABLED")
            self.offline_badge.setStyleSheet("color: #ff9933;")

    def _on_offline_changed(self):
        self._update_offline_badge()
        self._last_translated_text = ""   # force a fresh translation in the new mode
        self._on_language_changed()

    def _current_lang_name(self):
        code = self._current_lang_code()
        for name, c in SUPPORTED_LANGUAGES.items():
            if c == code:
                return name
        return None

    def _on_language_changed(self):
        """Only translate when the user actively picks a language."""
        if self.pipeline and self.pipeline.decoded_text.strip():
            self._update_translation(self.pipeline.decoded_text)
        else:
            self.trans_val.setText("")

    def _update_translation(self, decoded_text):
        """Translates decoded_text and updates the TRANSLATED> row.
        OFFLINE (default): phrasebook + transliteration, no network at all.
        ONLINE: MyMemory service (sends the text over the internet)."""
        import time
        lang_code = self._current_lang_code()
        if lang_code in ("en", "en-US") or not decoded_text.strip():
            self.trans_val.setText("")
            return

        if self._is_offline():
            result = offline_translator.translate_offline(decoded_text, self._current_lang_name())
            self.trans_val.setText(result["text"])
            unverified = any(p["source"] == "phrasebook" and not p["verified"] for p in result["pieces"])
            approximate = any(p["source"] != "phrasebook" for p in result["pieces"])
            self.trans_val.setStyleSheet(f"color: {'#ff9933' if (unverified or approximate) else '#66ccff'};")
            self.status_label.setText(f">>> OFFLINE TRANSLATION: {result['summary']}.")
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
            self.status_label.setText(">>> ONLINE TRANSLATION (text was sent to MyMemory).")
        else:
            self.trans_val.setText(f"[Error: {err}]")
            self.trans_val.setStyleSheet("color: #ff9933;")

    def speak_text(self):
        if not (self.pipeline and self.pipeline.decoded_text.strip()):
            return
        lang_code = self._current_lang_code()
        text_to_speak = self.pipeline.decoded_text
        if lang_code not in ("en", "en-US"):
            self._update_translation(self.pipeline.decoded_text)
            candidate = self.trans_val.text()
            if candidate and not candidate.startswith("[") and candidate != "Translating...":
                text_to_speak = candidate
        started, how = self.speech_engine.speak(
            text_to_speak, lang_code=lang_code, offline=self._is_offline(),
            lang_name=self._current_lang_name())
        if not started and lang_code not in ("en", "en-US"):
            # No offline voice for this language: say the original English
            # rather than silently sending anything online.
            self.speech_engine.speak(self.pipeline.decoded_text, lang_code="en")
            self.status_label.setText(f">>> {how.upper()}. SPOKE ORIGINAL ENGLISH INSTEAD.")
        elif started:
            self.status_label.setText(f">>> SPEAKING ({how}).")

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
            frame = orientation.rotate(frame, self.video_rotation)
            self.video_frame_index += 1

        self.frame_counter += 1
        wall = time.time()
        if self._last_frame_wall is not None and wall > self._last_frame_wall:
            inst = 1.0 / (wall - self._last_frame_wall)
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
        self._last_frame_wall = wall
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

        results, light, frame_used = self.pipeline.detect(frame)
        if light["applied"]:
            frame[:] = frame_used   # show the analyst what the detector sees
        apply_cinematic(frame, self.frame_counter, self._timecode(), self._source_tag())
        self._draw_light_badge(frame, light)
        h, w, _ = frame.shape
        face_found = bool(results.multi_face_landmarks)
        self._update_status_leds(face_found=face_found, light=light)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            draw_target_lock(frame, landmarks, False, LEFT_EYE, RIGHT_EYE, locked_text="CALIBRATING")
            cal_ear = average_ear(landmarks, w, h)
            self.signal_graph.add(cal_ear, None)
            self._update_telemetry(cal_ear, None, light)
            self.cal_ear_readings.append(cal_ear)
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
        if events["light"]["applied"]:
            frame[:] = events["frame_used"]   # show the analyst what the detector sees
        apply_cinematic(frame, self.frame_counter, self._timecode(), self._source_tag())
        self._draw_light_badge(frame, events["light"])
        if events.get("landmarks") is not None:
            draw_target_lock(frame, events["landmarks"], bool(self.pipeline.eye_closed), LEFT_EYE, RIGHT_EYE)
        if events["ear"] is not None:
            self.signal_graph.add(events["ear"], events.get("ear_threshold"))
        self._update_telemetry(events["ear"], events.get("ear_threshold"), events["light"])
        self._update_status_leds(face_found=events["face_found"], light=events["light"])
        self.morse_tape.set_pattern(self.pipeline.morse_buffer)

        if events["ear"] is not None:
            thr = events.get("ear_threshold")
            thr_txt = f"  THR {thr:.3f}" if thr is not None else ""
            cv2.putText(frame, f"EAR {events['ear']:.3f}{thr_txt}", (15, 25),
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

        if events.get("panic"):
            self._panic_until = now + 4.0
            self.status_label.setText(">>> !!! DISTRESS SIGNAL TRIGGERED !!!")
            log_text("[PANIC SIGNAL] Distress pattern detected (5 rapid long holds)")
            self.speech_engine.speak("Distress signal triggered", lang_code="en")

        if getattr(self, "_panic_until", 0) > now:
            h, w, _ = frame.shape
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            cv2.rectangle(frame, (5, 5), (w - 5, h - 5), DANGER_BGR, 4)
            text = "!!! DISTRESS SIGNAL !!!"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            cv2.putText(frame, text, ((w - tw) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        if events.get("codebook_phrase"):
            self._codebook_flash_text = events["letter"].strip()
            self._codebook_flash_until = now + 2.5
            self.status_label.setText(f">>> CODEBOOK PHRASE RECOGNIZED: \"{self._codebook_flash_text}\"")
            log_text(f"[CODEBOOK PHRASE] {self._codebook_flash_text}")
            self.speech_engine.speak(self._codebook_flash_text, lang_code="en")

        if getattr(self, "_codebook_flash_until", 0) > now:
            h, w, _ = frame.shape
            text = f"CODEBOOK: {self._codebook_flash_text}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (0, h // 2 - th - 15), (w, h // 2 + 15), (0, 40, 0), -1)
            cv2.putText(frame, text, ((w - tw) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACCENT_BGR, 2)

        self.buffer_val.setText(self.pipeline.morse_buffer)
        self.text_val.setText(self.pipeline.decoded_text)

        if events["word_complete"]:
            log_text(self.pipeline.decoded_text)
            self.speech_engine.speak(events["word_complete"])

    def _draw_light_badge(self, frame, light):
        """Bottom-left HUD line: measured brightness and whether night
        enhancement is active on this frame."""
        h, w, _ = frame.shape
        if light["applied"]:
            text = f"NIGHT MODE ACTIVE  LIGHT {light['brightness']:.0f}/255 ({light['level'].upper()})"
            color = (0, 200, 255)
        else:
            text = f"LIGHT {light['brightness']:.0f}/255 ({light['level'].upper()})"
            color = (150, 150, 150)
        cv2.putText(frame, text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

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
