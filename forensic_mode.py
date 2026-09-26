"""
forensic_mode.py

A human-in-the-loop analysis tool, separate from the live/automatic
GUI. Built for the realistic version of this project's use case:
reviewing a SINGLE piece of already-recorded footage (e.g. hostage,
propaganda, or interrogation video) where automatic detection may not
be fully reliable -- bad lighting, low resolution, re-compression,
unusual camera angle, a subject who isn't blinking in a clean,
cooperative way.

Instead of trying to make automatic detection "just work" on footage
like that (an overclaim), this gives an analyst the tools to verify
and correct it by hand:

1. The ENTIRE video is analyzed once up front, producing one EAR
   value per frame (this is the same Eye Aspect Ratio math used
   everywhere else in the project -- see blink_detector.py).
2. That EAR sequence is plotted as a graph across the whole video,
   so a dip below the threshold is visually obvious even where
   automatic detection might be uncertain.
3. The analyst can scrub through frames, adjust the threshold live,
   and see the effect immediately.
4. Where automatic detection is wrong or missing something, the
   analyst can manually mark a blink's start/end frame directly.
5. Every event -- automatic or manual -- gets an honest confidence
   flag: LOW if its duration sits close to the dot/dash cutoff, or if
   face tracking was lost on a nearby frame (both cheap, explainable
   checks -- not a trained classifier making a black-box judgment).
6. After decoding, the graph shows exactly which blink events were
   grouped into which letter, so the decoding is auditable rather
   than a black box.

RUN: opened from gui.py via the "FORENSIC ANALYSIS MODE" button, or
standalone with: python forensic_mode.py
"""

import cv2
import mediapipe as mp

from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSlider,
    QDoubleSpinBox, QFileDialog, QMessageBox, QApplication,
    QTableWidget, QTableWidgetItem, QCheckBox, QComboBox, QListWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from blink_detector import average_ear
from morse_translator import decode_letter
import segmentation
import enhancement
import orientation
import video_metadata
import alternatives
from pipeline import SHORT_BLINK_MAX, LETTER_PAUSE, WORD_PAUSE
from logger import log_text, LOG_FILE

ACCENT = "#00ff66"
DANGER = "#ff3b3b"
MANUAL_COLOR = "#33aaff"
LOW_CONF_COLOR = "#ff9933"

# How close (as a fraction of SHORT_BLINK_MAX) a blink's duration must be
# to the dot/dash cutoff before we flag it as "borderline" rather than
# silently deciding. e.g. 0.15 = within 15% of the cutoff on either side.
BORDERLINE_TOLERANCE = 0.15
# How many frames before/after a blink to check for lost face tracking.
TRACKING_CHECK_WINDOW = 3


class ForensicWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FORENSIC ANALYSIS MODE :: FRAME-BY-FRAME BLINK REVIEW")
        self.resize(1050, 900)
        self.setStyleSheet(
            "QWidget { background-color: #000000; color: #00ff66; font-family: Consolas; }"
            "QPushButton { background-color: #001a0d; color: #00ff66; border: 1px solid #00ff66;"
            " border-radius: 8px; padding: 8px 12px; font-weight: bold; }"
            "QPushButton:hover { background-color: #00331a; }"
            "QTableWidget { background-color: #0a0a0a; color: #e0ffe0; gridline-color: #0a4d24; }"
            "QHeaderView::section { background-color: #0a2a15; color: #00ff66; }"
        )

        self.cap = None
        self.fps = 30.0
        self.frame_count = 0
        self.ear_series = []
        self.pending_start = None
        self.events = []  # each: {start, end, duration, symbol, manual, include, confidence, confidence_reason}
        self.letter_groups = None  # set only right after a successful decode; cleared by any edit
        self.decoded_text = ""

        self._build_ui()

    # ---------------- UI LAYOUT ----------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self.load_btn = QPushButton("[ LOAD VIDEO FOR ANALYSIS ]")
        self.load_btn.clicked.connect(self.load_video)
        self.info_label = QLabel("No video loaded.")
        top.addWidget(self.load_btn)
        top.addWidget(QLabel("LOW-LIGHT:"))
        self.light_combo = QComboBox()
        self.light_combo.addItem("Auto (enhance when dark)", "auto")
        self.light_combo.addItem("Off", "off")
        self.light_combo.addItem("Always on", "on")
        self.light_combo.setToolTip("Applied when a video is loaded. Reload the video after changing it.")
        top.addWidget(self.light_combo)
        top.addWidget(self.info_label)

        meta_row = QHBoxLayout()
        meta_caption = QLabel("FILE METADATA (from the video file itself, unverified, separate from the decoded message):")
        meta_caption.setStyleSheet("color: #3d8b52; font-size: 10px;")
        self.metadata_label = QLabel("No video loaded.")
        self.metadata_label.setStyleSheet("color: #ffcc66; font-size: 10px;")
        meta_row.addWidget(meta_caption)
        meta_row.addWidget(self.metadata_label, stretch=1)
        top.addStretch()
        layout.addLayout(top)
        layout.addLayout(meta_row)

        self.figure = Figure(figsize=(8, 2.6), facecolor="#0a0a0a")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("EAR THRESHOLD:"))
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.05, 0.5)
        self.thresh_spin.setSingleStep(0.01)
        self.thresh_spin.setValue(0.21)
        self.thresh_spin.valueChanged.connect(self._recompute_auto_events)
        thresh_row.addWidget(self.thresh_spin)
        thresh_row.addStretch()
        layout.addLayout(thresh_row)

        preview_row = QHBoxLayout()
        self.frame_label = QLabel()
        self.frame_label.setFixedSize(340, 190)
        self.frame_label.setStyleSheet("background-color:black; border: 1px solid #00ff66;")
        preview_row.addWidget(self.frame_label)

        scrub_col = QVBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self._on_scrub)
        self.frame_pos_label = QLabel("Frame: 0 / 0   Time: 0.00s")
        step_row = QHBoxLayout()
        self.step_back_btn = QPushButton("<< STEP")
        self.step_fwd_btn = QPushButton("STEP >>")
        self.step_back_btn.clicked.connect(lambda: self._step(-1))
        self.step_fwd_btn.clicked.connect(lambda: self._step(1))
        step_row.addWidget(self.step_back_btn)
        step_row.addWidget(self.step_fwd_btn)
        scrub_col.addWidget(self.frame_slider)
        scrub_col.addWidget(self.frame_pos_label)
        scrub_col.addLayout(step_row)
        preview_row.addLayout(scrub_col)
        layout.addLayout(preview_row)

        mark_row = QHBoxLayout()
        self.mark_start_btn = QPushButton("[ MARK BLINK START HERE ]")
        self.mark_end_btn = QPushButton("[ MARK BLINK END HERE ]")
        self.mark_start_btn.clicked.connect(self._mark_start)
        self.mark_end_btn.clicked.connect(self._mark_end)
        self.pending_label = QLabel("No pending start marked.")
        mark_row.addWidget(self.mark_start_btn)
        mark_row.addWidget(self.mark_end_btn)
        mark_row.addWidget(self.pending_label)
        layout.addLayout(mark_row)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Include", "Start Frame", "End Frame", "Duration (s)", "Symbol", "Confidence"]
        )
        layout.addWidget(self.table)

        decode_row = QHBoxLayout()
        decode_row.addWidget(QLabel("TIMING:"))
        self.timing_combo = QComboBox()
        self.timing_combo.addItem("Adaptive (rhythm + repair)", "adaptive")
        self.timing_combo.addItem("Fixed (1.2s / 3.0s)", "fixed")
        self.timing_combo.setStyleSheet(
            "QComboBox { background-color: #0a0a0a; color: #00ff66; border: 1px solid #0a4d24;"
            " border-radius: 6px; padding: 4px; }"
            "QComboBox QAbstractItemView { background-color: #0a0a0a; color: #00ff66; }"
        )
        self.timing_combo.currentIndexChanged.connect(self._decode)
        decode_row.addWidget(self.timing_combo)
        self.decode_btn = QPushButton("[ DECODE FROM REVIEWED EVENTS ]")
        self.decode_btn.clicked.connect(self._decode)
        self.decoded_label = QLabel("DECODED> ")
        self.decoded_label.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        self.decoded_label.setStyleSheet("color: #ffe066;")
        self.save_btn = QPushButton("[ SAVE TO LOG ]")
        self.save_btn.clicked.connect(self._save_log)
        decode_row.addWidget(self.decode_btn)
        decode_row.addWidget(self.decoded_label, stretch=1)
        decode_row.addWidget(self.save_btn)
        layout.addLayout(decode_row)

        alt_header = QLabel("ALTERNATIVE READINGS  (suggestions only: nothing changes until you apply one)")
        alt_header.setStyleSheet("color: #3d8b52;")
        layout.addWidget(alt_header)
        alt_row = QHBoxLayout()
        self.alt_list = QListWidget()
        self.alt_list.setMaximumHeight(110)
        self.alt_list.setStyleSheet("QListWidget { background-color: #0a0a0a; color: #e0ffe0;"
                                    " border: 1px solid #0a4d24; }")
        self.alt_list.itemDoubleClicked.connect(lambda _item: self._apply_alternative())
        self.apply_alt_btn = QPushButton("[ APPLY SELECTED ]")
        self.apply_alt_btn.clicked.connect(self._apply_alternative)
        alt_row.addWidget(self.alt_list, stretch=1)
        alt_row.addWidget(self.apply_alt_btn)
        layout.addLayout(alt_row)
        self.current_alternatives = []
        self.current_included = []

        self.status_label = QLabel(f"Ready. Log file: {LOG_FILE}")
        self.status_label.setStyleSheet("color: #3d8b52; font-size: 10px;")
        layout.addWidget(self.status_label)

    # ---------------- VIDEO LOADING & ANALYSIS ----------------

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video for forensic analysis", "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)",
        )
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open that video file.")
            return

        self.cap = cap
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.info_label.setText(f"{self.frame_count} frames @ {self.fps:.1f} fps")
        self.video_metadata = video_metadata.read_metadata(path)
        self.metadata_label.setText(video_metadata.format_summary(self.video_metadata))
        self.events = []
        self.letter_groups = None
        self.decoded_text = ""
        self.decoded_label.setText("DECODED> ")

        self.status_label.setText("Checking video orientation...")
        QApplication.processEvents()
        self.video_rotation, rot_label, _ = orientation.detect_rotation(
            self.cap, lambda: mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                              refine_landmarks=True, min_detection_confidence=0.5))
        self.rotation_note = f" Video rotated {rot_label} deg." if self.video_rotation is not None else ""
        self.status_label.setText("Analyzing video frame-by-frame (extracting EAR values)...")
        QApplication.processEvents()
        self._analyze_video()

        self.frame_slider.setRange(0, max(self.frame_count - 1, 0))
        self.frame_slider.setValue(0)
        self._recompute_auto_events()
        self.status_label.setText("Analysis complete. " + self.light_summary + self.rotation_note)

    def _analyze_video(self):
        """Runs face-mesh + EAR once for every frame in the video, up front."""
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.ear_series = []
        self.analysis_light_mode = self.light_combo.currentData()
        brightness_sum, enhanced = 0.0, 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = orientation.rotate(frame, self.video_rotation)
            frame, light = enhancement.enhance(frame, self.analysis_light_mode)
            brightness_sum += light["brightness"]
            enhanced += int(light["applied"])
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = frame.shape
                self.ear_series.append(average_ear(landmarks, w, h))
            else:
                self.ear_series.append(None)  # no face found on this frame
            idx += 1
            if idx % 30 == 0:
                self.status_label.setText(f"Analyzing... frame {idx}/{self.frame_count}")
                QApplication.processEvents()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        n = max(idx, 1)
        lost = sum(1 for v in self.ear_series if v is None)
        self.light_summary = (
            f"Avg brightness {brightness_sum / n:.0f}/255, night enhancement on "
            f"{100 * enhanced / n:.0f}% of frames ({self.analysis_light_mode}), "
            f"face lost on {100 * lost / n:.1f}% of frames."
        )

    # ---------------- PLOTTING ----------------

    def _redraw_plot(self):
        self.ax.clear()
        self.ax.set_facecolor("#0a0a0a")
        self.figure.patch.set_facecolor("#0a0a0a")
        for spine in self.ax.spines.values():
            spine.set_color("#0a4d24")
        self.ax.tick_params(colors="#00ff66", labelsize=7)
        self.ax.set_xlabel("Time (s)", color="#00ff66", fontsize=8)
        self.ax.set_ylabel("EAR", color="#00ff66", fontsize=8)

        xs = [i / self.fps for i in range(len(self.ear_series))]
        ys = [v if v is not None else float("nan") for v in self.ear_series]
        self.ax.plot(xs, ys, color=ACCENT, linewidth=1)
        self.ax.axhline(self.thresh_spin.value(), color=DANGER, linestyle="--", linewidth=1)

        y_top = max((v for v in self.ear_series if v is not None), default=0.4) * 1.15

        for e in self.events:
            if not e["include"]:
                continue
            # Color priority: LOW confidence is flagged regardless of
            # auto/manual, since "uncertain" is the more important fact
            # for an analyst to see at a glance than the source.
            if e.get("confidence") == "LOW":
                color = LOW_CONF_COLOR
            elif e["manual"]:
                color = MANUAL_COLOR
            else:
                color = ACCENT
            self.ax.axvspan(e["start"] / self.fps, (e["end"] + 1) / self.fps, color=color, alpha=0.3)

        # Letter-timeline annotations -- only present right after a
        # successful decode; cleared by any edit (see _invalidate_decode)
        # so a stale label never lingers over changed data.
        if self.letter_groups:
            for group in self.letter_groups:
                if not group["events"]:
                    continue
                start_t = group["events"][0]["start"] / self.fps
                end_t = (group["events"][-1]["end"] + 1) / self.fps
                mid_t = (start_t + end_t) / 2
                # Reconstructed letters (inferred by pattern repair, not
                # directly separated by a pause) are orange and starred.
                col = LOW_CONF_COLOR if group.get("reconstructed") else "#ffe066"
                label = group["letter"] + ("*" if group.get("reconstructed") else "")
                self.ax.plot([start_t, end_t], [y_top, y_top], color=col, linewidth=1.5)
                self.ax.plot([start_t, start_t], [y_top * 0.95, y_top * 1.05], color=col, linewidth=1.5)
                self.ax.plot([end_t, end_t], [y_top * 0.95, y_top * 1.05], color=col, linewidth=1.5)
                self.ax.text(mid_t, y_top * 1.08, label, color=col,
                             fontsize=9, fontweight="bold", ha="center")

        current_t = self.frame_slider.value() / self.fps if self.frame_count else 0
        self.cursor_line = self.ax.axvline(current_t, color="#ffe066", linewidth=1)
        self.canvas.draw()

    # ---------------- CONFIDENCE ASSESSMENT ----------------

    def _compute_confidence(self, start, end, duration):
        """
        Flags a blink event as LOW confidence instead of silently trusting
        it, in either of two honest, cheap-to-check situations:

        1. Its duration sits close to the dot/dash cutoff (SHORT_BLINK_MAX)
           -- a small timing wobble could have flipped which symbol this
           should be.
        2. Face tracking was lost on a nearby frame (just before/after the
           blink) -- meaning this "blink" might actually be the face
           briefly leaving and re-entering frame, not a real eye closure.

        This is deliberately NOT a machine-learned classifier -- it's an
        honest, explainable heuristic, which is a more defensible claim
        for a system reviewing real, unverified footage.
        """
        reasons = []

        boundary_distance = abs(duration - SHORT_BLINK_MAX) / SHORT_BLINK_MAX
        if boundary_distance < BORDERLINE_TOLERANCE:
            reasons.append("duration borderline")

        lo = max(0, start - TRACKING_CHECK_WINDOW)
        hi = min(len(self.ear_series) - 1, end + TRACKING_CHECK_WINDOW)
        if any(self.ear_series[i] is None for i in range(lo, hi + 1)):
            reasons.append("tracking gap nearby")

        if reasons:
            return "LOW", " + ".join(reasons)
        return "HIGH", ""

    def _invalidate_decode(self):
        """Any edit to events makes a previous decode stale -- clear the
        letter-timeline annotations until the analyst re-decodes."""
        self.letter_groups = None

    # ---------------- AUTOMATIC + MANUAL EVENT DETECTION ----------------

    def _recompute_auto_events(self):
        if not self.ear_series:
            return
        threshold = self.thresh_spin.value()
        auto_events = []
        i, n = 0, len(self.ear_series)
        while i < n:
            v = self.ear_series[i]
            if v is not None and v < threshold:
                start = i
                while i < n and self.ear_series[i] is not None and self.ear_series[i] < threshold:
                    i += 1
                end = i - 1
                duration = (end - start + 1) / self.fps
                symbol = "." if duration < SHORT_BLINK_MAX else "-"
                confidence, reason = self._compute_confidence(start, end, duration)
                auto_events.append({
                    "start": start, "end": end, "duration": duration,
                    "symbol": symbol, "manual": False, "include": True,
                    "confidence": confidence, "confidence_reason": reason,
                    "original_symbol": symbol,
                })
            else:
                i += 1

        manual_events = [e for e in self.events if e.get("manual")]
        self.events = sorted(auto_events + manual_events, key=lambda e: e["start"])
        self._invalidate_decode()
        self._refresh_table()
        self._redraw_plot()

    def _mark_start(self):
        if self.cap is None:
            return
        self.pending_start = self.frame_slider.value()
        self.pending_label.setText(f"Pending start: frame {self.pending_start}. Now scrub to the end and mark it.")

    def _mark_end(self):
        if self.pending_start is None:
            QMessageBox.information(self, "No start marked", "Mark a blink START frame first.")
            return
        start = self.pending_start
        end = self.frame_slider.value()
        if end < start:
            start, end = end, start
        duration = (end - start + 1) / self.fps
        symbol = "." if duration < SHORT_BLINK_MAX else "-"
        self.events.append({
            "start": start, "end": end, "duration": duration,
            "symbol": symbol, "manual": True, "include": True,
            "confidence": "MANUAL", "confidence_reason": "analyst-verified by eye",
            "original_symbol": symbol,
        })
        self.events.sort(key=lambda e: e["start"])
        self.pending_start = None
        self.pending_label.setText("No pending start marked.")
        self._invalidate_decode()
        self._refresh_table()
        self._redraw_plot()

    # ---------------- TABLE ----------------

    def _refresh_table(self):
        self.table.setRowCount(len(self.events))
        for row, e in enumerate(self.events):
            include_cb = QCheckBox()
            include_cb.setChecked(e["include"])
            include_cb.stateChanged.connect(lambda _state, r=row: self._on_include_changed(r))
            self.table.setCellWidget(row, 0, include_cb)

            label = "MANUAL" if e["manual"] else "AUTO"
            if not e["manual"] and e["symbol"] != e.get("original_symbol", e["symbol"]):
                label += ", CORRECTED"
            self.table.setItem(row, 1, QTableWidgetItem(f"{e['start']} ({label})"))
            self.table.setItem(row, 2, QTableWidgetItem(str(e["end"])))
            self.table.setItem(row, 3, QTableWidgetItem(f"{e['duration']:.3f}"))

            symbol_combo = QComboBox()
            symbol_combo.addItems([".", "-"])
            symbol_combo.setCurrentText(e["symbol"])
            symbol_combo.currentTextChanged.connect(lambda text, r=row: self._on_symbol_changed(r, text))
            self.table.setCellWidget(row, 4, symbol_combo)

            conf = e.get("confidence", "HIGH")
            reason = e.get("confidence_reason", "")
            conf_item = QTableWidgetItem(f"{conf}" + (f" ({reason})" if reason else ""))
            if conf == "LOW":
                conf_item.setForeground(QColor(LOW_CONF_COLOR))
            elif conf == "MANUAL":
                conf_item.setForeground(QColor(MANUAL_COLOR))
            else:
                conf_item.setForeground(QColor(ACCENT))
            self.table.setItem(row, 5, conf_item)

    def _on_include_changed(self, row):
        cb = self.table.cellWidget(row, 0)
        self.events[row]["include"] = cb.isChecked()
        self._invalidate_decode()
        self._redraw_plot()

    def _on_symbol_changed(self, row, text):
        self.events[row]["symbol"] = text
        self._invalidate_decode()

    # ---------------- SCRUBBING ----------------

    def _step(self, delta):
        new_val = max(0, min(self.frame_slider.value() + delta, self.frame_count - 1))
        self.frame_slider.setValue(new_val)

    def _on_scrub(self, value):
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
        ret, frame = self.cap.read()
        if ret:
            # Show exactly what the detector analysed (rotated/enhanced if applied)
            frame = orientation.rotate(frame, getattr(self, "video_rotation", None))
            frame, _ = enhancement.enhance(frame, getattr(self, "analysis_light_mode", "off"))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(
                340, 190, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.frame_label.setPixmap(pixmap)

        t = value / self.fps if self.fps else 0
        self.frame_pos_label.setText(f"Frame: {value} / {self.frame_count}   Time: {t:.2f}s")

        if hasattr(self, "cursor_line") and self.cursor_line is not None:
            self.cursor_line.set_xdata([t, t])
            self.canvas.draw_idle()

    # ---------------- DECODING ----------------

    def _decode(self):
        if not self.events:
            return
        included = sorted([e for e in self.events if e["include"]], key=lambda e: e["start"])
        if not included:
            self.decoded_label.setText("DECODED> (no reviewed/included blink events)")
            self.letter_groups = None
            self.alt_list.clear()
            self.current_alternatives = []
            self._redraw_plot()
            return

        # Letter/word segmentation lives in segmentation.py so Forensic
        # Mode and evaluate.py use exactly the same logic.
        blinks = [{"t_start": e["start"] / self.fps, "t_end": e["end"] / self.fps,
                   "duration": e["duration"], "symbol": e["symbol"], "src": e,
                   "low_tracking": "tracking gap" in e.get("confidence_reason", "")}
                  for e in included]
        mode = self.timing_combo.currentData()
        result = segmentation.decode(blinks, mode)

        self.letter_groups = [
            {"events": [b["src"] for b in entry["blinks"]],
             "letter": entry["letter"],
             "reconstructed": entry["reconstructed"]}
            for entry in result["letters"]
        ]
        text = result["text"]
        self.decoded_text = text

        uncertain_count = sum(1 for e in included if e.get("confidence") == "LOW")
        recon_count = result["reconstructed_count"]
        warnings = []
        if uncertain_count:
            warnings.append(f"{uncertain_count} UNCERTAIN SYMBOL(S)")
        if recon_count:
            warnings.append(f"{recon_count} RECONSTRUCTED LETTER(S)*")
        warning = ("   \u26a0 " + " + ".join(warnings) + " -- REVIEW") if warnings else ""
        self.decoded_label.setText(f"DECODED> {text}{warning}")
        self.decoded_label.setStyleSheet(
            f"color: {LOW_CONF_COLOR if warnings else '#ffe066'};"
        )

        unit_txt = f", rhythm unit {result['unit']:.2f}s" if result["unit"] else ""
        self.status_label.setText(
            f"Decoded {len(included)} blink(s) in {mode.upper()} mode: letter gap "
            f"{result['letter_gap']:.2f}s, word gap {result['word_gap']:.2f}s{unit_txt}."
        )
        self._redraw_plot()
        self._update_alternatives(blinks, included, mode)

    # ---------------- ALTERNATIVE READINGS + RAW / FINAL LAYERS ----------------

    def _raw_text(self, mode):
        """What the detector produced on its own: every automatic blink,
        with its original symbol, and no analyst changes."""
        auto = sorted([e for e in self.events if not e["manual"]], key=lambda e: e["start"])
        blinks = [{"t_start": e["start"] / self.fps, "t_end": e["end"] / self.fps,
                   "symbol": e.get("original_symbol", e["symbol"])} for e in auto]
        return segmentation.decode(blinks, mode)["text"] if blinks else ""

    def _corrections(self):
        notes = []
        for e in sorted(self.events, key=lambda e: e["start"]):
            t = e["start"] / self.fps
            if e["manual"] and e["include"]:
                notes.append(f"added blink at {t:.2f}s")
            elif not e["manual"] and not e["include"]:
                notes.append(f"excluded blink at {t:.2f}s")
            elif not e["manual"] and e["symbol"] != e.get("original_symbol", e["symbol"]):
                notes.append(f"blink at {t:.2f}s '{e['original_symbol']}' -> '{e['symbol']}'")
        return notes

    def _update_alternatives(self, blinks, included, mode):
        self.alt_list.clear()
        self.current_included = included
        _base, self.current_alternatives = alternatives.reading_alternatives(blinks, mode)
        for k, alt in enumerate(self.current_alternatives, 1):
            hint = f"   [contains known phrase: {', '.join(alt['known_phrases'])}]" if alt["known_phrases"] else ""
            self.alt_list.addItem(f"{k}. {alt['text']}   <- {alt['explanation']}{hint}")
        if not self.current_alternatives:
            self.alt_list.addItem("(no alternative readings)")

        raw = self._raw_text(mode)
        corrections = self._corrections()
        layers = f"RAW: {raw}  |  FINAL: {self.decoded_text}  |  {len(corrections)} analyst correction(s)"
        self.status_label.setText(self.status_label.text() + "   " + layers)

    def _apply_alternative(self):
        row = self.alt_list.currentRow()
        if row < 0 or row >= len(self.current_alternatives):
            QMessageBox.information(self, "Nothing selected", "Select an alternative reading first.")
            return
        for idx, kind, _old, new in self.current_alternatives[row]["changes"]:
            event = self.current_included[idx]
            if kind == "flip":
                event["symbol"] = new
            else:
                event["include"] = False
        self._invalidate_decode()
        self._refresh_table()
        self._decode()

    def _save_log(self):
        if self.decoded_text.strip():
            mode = self.timing_combo.currentData()
            corrections = self._corrections()
            meta = getattr(self, "video_metadata", None)
            meta_txt = video_metadata.format_summary(meta) if meta else "not read"
            log_text(f"[FORENSIC MODE] RAW: {self._raw_text(mode)} | FINAL: {self.decoded_text} | "
                     f"CORRECTIONS: {'; '.join(corrections) if corrections else 'none'} | "
                     f"FILE METADATA (unverified): {meta_txt}")
            self.status_label.setText(f"Saved to log: {LOG_FILE}")
        else:
            QMessageBox.information(self, "Nothing to save", "Decode some text first.")


def main():
    from PyQt6.QtWidgets import QApplication as QApp
    app = QApp([])
    window = ForensicWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
