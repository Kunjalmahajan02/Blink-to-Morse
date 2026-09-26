"""
hud_widgets.py

Purely DECORATIVE / DISPLAY-ONLY widgets for the main window.
Nothing in this file changes how the system works: no widget here is
clickable or accepts input. They only visualise information the app
already has (face found, eyes open/closed, EAR values, the Morse
buffer, and so on).

Widgets:
    EmblemWidget   animated radar-style eye emblem for the header
    LedIndicator   a glowing status light with a label
    SignalGraph    heart-monitor style graph of EAR over time
    MorseTape      the current Morse buffer drawn as glowing dots/dashes
    HudPanel       a titled panel with neon corner brackets
    StatusTicker   a status line that time-stamps every message
    make_standby_frame()   image shown in the video area when idle
"""

import math
import time
from collections import deque

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt6.QtGui import (QPainter, QPen, QColor, QBrush, QFont, QPolygonF,
                         QRadialGradient)
from PyQt6.QtWidgets import QWidget, QLabel, QFrame, QVBoxLayout

ACCENT = QColor("#00ff66")
ACCENT_DIM = QColor("#0a4d24")
DANGER = QColor("#ff3b3b")
WARN = QColor("#ffb020")
OFF = QColor("#2a2a2a")
TEXT = QColor("#b6ffb6")


def _antialias(painter):
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)


def paint_grid_background(widget, painter, spacing=28):
    """Very subtle grid behind the whole window (visible in the gaps
    between panels)."""
    painter.fillRect(widget.rect(), QColor("#000000"))
    pen = QPen(QColor(0, 255, 102, 14))
    pen.setWidth(1)
    painter.setPen(pen)
    w, h = widget.width(), widget.height()
    for x in range(0, w, spacing):
        painter.drawLine(x, 0, x, h)
    for y in range(0, h, spacing):
        painter.drawLine(0, y, w, y)


class EmblemWidget(QWidget):
    """Rotating radar ring around a stylised eye."""

    def __init__(self, size=46, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angle = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)

    def _tick(self):
        self._angle = (self._angle + 4.0) % 360.0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        _antialias(p)
        w, h = self.width(), self.height()
        cx, cy, r = w / 2, h / 2, min(w, h) / 2 - 3

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(ACCENT_DIM, 1))
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.drawEllipse(QPointF(cx, cy), r * 0.62, r * 0.62)

        # rotating sweep arc
        p.setPen(QPen(ACCENT, 2))
        start = int(-self._angle * 16)
        p.drawArc(QRectF(cx - r, cy - r, 2 * r, 2 * r), start, 70 * 16)
        # sweep line
        a = math.radians(self._angle)
        p.setPen(QPen(QColor(0, 255, 102, 150), 1))
        p.drawLine(QPointF(cx, cy), QPointF(cx + r * math.cos(a), cy - r * math.sin(a)))

        # eye in the middle
        eye_w, eye_h = r * 0.9, r * 0.45
        top = QPolygonF([QPointF(cx - eye_w, cy), QPointF(cx, cy - eye_h), QPointF(cx + eye_w, cy),
                         QPointF(cx, cy + eye_h)])
        p.setPen(QPen(ACCENT, 1.5))
        p.drawPolygon(top)
        p.setBrush(QBrush(ACCENT))
        p.drawEllipse(QPointF(cx, cy), r * 0.16, r * 0.16)
        p.end()


class LedIndicator(QWidget):
    """A glowing status light. state: 'on', 'off', 'warn' or 'alert'."""

    COLORS = {"on": ACCENT, "off": OFF, "warn": WARN, "alert": DANGER}

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.label = label
        self.detail = ""
        self.state = "off"
        self.setFixedHeight(22)
        self.setMinimumWidth(170)

    def set_state(self, state, detail=""):
        if state != self.state or detail != self.detail:
            self.state, self.detail = state, detail
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        _antialias(p)
        color = self.COLORS.get(self.state, OFF)
        cx, cy = 10, self.height() / 2
        if self.state != "off":
            glow = QRadialGradient(cx, cy, 10)
            c = QColor(color)
            c.setAlpha(120)
            glow.setColorAt(0.0, c)
            c2 = QColor(color)
            c2.setAlpha(0)
            glow.setColorAt(1.0, c2)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(QPointF(cx, cy), 10, 10)
        p.setPen(QPen(QColor("#111111"), 1))
        p.setBrush(QBrush(color))
        p.drawEllipse(QPointF(cx, cy), 5, 5)

        p.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        p.setPen(QPen(TEXT if self.state != "off" else QColor("#5a5a5a")))
        p.drawText(QPointF(24, cy + 4), self.label)
        if self.detail:
            p.setPen(QPen(color if self.state != "off" else QColor("#5a5a5a")))
            p.drawText(QPointF(self.width() - 8 - 7 * len(self.detail), cy + 4), self.detail)
        p.end()


class SignalGraph(QWidget):
    """Heart-monitor style graph of the Eye Aspect Ratio, with the
    current blink threshold as a dashed red line. Dips below the line
    are blinks."""

    def __init__(self, history=160, parent=None):
        super().__init__(parent)
        self.values = deque(maxlen=history)
        self.thresholds = deque(maxlen=history)
        self.setMinimumHeight(90)
        self.setMinimumWidth(170)

    def add(self, ear, threshold):
        self.values.append(ear)
        self.thresholds.append(threshold)
        self.update()

    def clear(self):
        self.values.clear()
        self.thresholds.clear()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        _antialias(p)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#050805"))

        p.setPen(QPen(QColor(0, 255, 102, 25), 1))
        for i in range(1, 4):
            y = h * i / 4
            p.drawLine(0, int(y), w, int(y))
        for i in range(1, 6):
            x = w * i / 6
            p.drawLine(int(x), 0, int(x), h)

        lo, hi = 0.0, 0.45

        def ypos(v):
            v = min(max(v, lo), hi)
            return h - 4 - (v - lo) / (hi - lo) * (h - 8)

        n = len(self.values)
        if n >= 2:
            step = w / (self.values.maxlen - 1)
            x0 = w - (n - 1) * step
            thr_pts = [QPointF(x0 + i * step, ypos(t)) for i, t in enumerate(self.thresholds) if t is not None]
            if len(thr_pts) >= 2:
                pen = QPen(QColor(255, 59, 59, 180), 1)
                pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.drawPolyline(QPolygonF(thr_pts))
            pts = [QPointF(x0 + i * step, ypos(v)) for i, v in enumerate(self.values)]
            p.setPen(QPen(QColor(0, 255, 102, 60), 4))
            p.drawPolyline(QPolygonF(pts))
            p.setPen(QPen(ACCENT, 1.5))
            p.drawPolyline(QPolygonF(pts))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(ACCENT))
            p.drawEllipse(pts[-1], 3, 3)
        else:
            p.setPen(QPen(QColor("#3d8b52")))
            p.setFont(QFont("Consolas", 8))
            p.drawText(QPointF(8, h / 2 + 3), "NO SIGNAL")
        p.end()


class MorseTape(QWidget):
    """Draws the current Morse buffer as glowing dots and dashes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pattern = ""
        self.setFixedHeight(26)
        self.setMinimumWidth(260)

    def set_pattern(self, pattern):
        if pattern != self.pattern:
            self.pattern = pattern
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        _antialias(p)
        h = self.height()
        cy = h / 2
        x = 6
        for sym in self.pattern[-24:]:
            glow = QColor(0, 255, 102, 70)
            if sym == ".":
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(glow))
                p.drawEllipse(QPointF(x + 5, cy), 8, 8)
                p.setBrush(QBrush(ACCENT))
                p.drawEllipse(QPointF(x + 5, cy), 4.5, 4.5)
                x += 18
            elif sym == "-":
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(glow))
                p.drawRoundedRect(QRectF(x - 2, cy - 7, 30, 14), 7, 7)
                p.setBrush(QBrush(ACCENT))
                p.drawRoundedRect(QRectF(x + 1, cy - 4, 24, 8), 4, 4)
                x += 34
        if not self.pattern:
            p.setPen(QPen(QColor("#2f5f3f")))
            p.setFont(QFont("Consolas", 9))
            p.drawText(QPointF(6, cy + 4), "- - awaiting signal - -")
        p.end()


class HudPanel(QFrame):
    """A titled side panel with neon corner brackets."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("hudPanel")
        self.setStyleSheet("QFrame#hudPanel { background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
                           " stop:0 #07170d, stop:1 #020905); border: 1px solid #0f5c2e;"
                           " border-top: 2px solid #00ff66; border-radius: 4px; }")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(10, 8, 10, 10)
        self.body.setSpacing(4)
        heading = QLabel(f"\u25c6 {title}")
        heading.setStyleSheet("color: #001a0a; font-weight: bold; font-size: 11px; padding: 2px 6px;"
                              " background-color: #00ff66; border: none; border-radius: 2px;")
        self.body.addWidget(heading)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setPen(QPen(ACCENT, 2))
        w, h, L = self.width() - 1, self.height() - 1, 12
        for (x, y, dx, dy) in [(1, 1, 1, 1), (w, 1, -1, 1), (1, h, 1, -1), (w, h, -1, -1)]:
            p.drawLine(x, y, x + dx * L, y)
            p.drawLine(x, y, x, y + dy * L)
        p.end()


class StatusTicker(QLabel):
    """Status line that time-stamps every message, like a system log."""

    def setText(self, text):
        super().setText(f"[{time.strftime('%H:%M:%S')}] {text}")


def make_standby_frame(width=640, height=360, subtitle="SELECT A SIGNAL SOURCE ABOVE"):
    """Image shown in the video area when no camera or file is active."""
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = (6, 10, 6)
    for x in range(0, width, 32):
        cv2.line(img, (x, 0), (x, height), (14, 30, 14), 1)
    for y in range(0, height, 32):
        cv2.line(img, (0, y), (width, y), (14, 30, 14), 1)
    cx, cy = width // 2, height // 2 - 20
    for r in (70, 50, 30):
        cv2.circle(img, (cx, cy), r, (30, 90, 40), 1)
    cv2.line(img, (cx - 90, cy), (cx + 90, cy), (30, 90, 40), 1)
    cv2.line(img, (cx, cy - 90), (cx, cy + 90), (30, 90, 40), 1)
    eye = np.array([[cx - 42, cy], [cx, cy - 20], [cx + 42, cy], [cx, cy + 20]], np.int32)
    cv2.polylines(img, [eye], True, (102, 255, 0), 2)
    cv2.circle(img, (cx, cy), 8, (102, 255, 0), -1)
    for text, y, scale, color in [("AWAITING SIGNAL SOURCE", cy + 110, 0.8, (102, 255, 0)),
                                  (subtitle, cy + 140, 0.45, (60, 140, 70))]:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2 if scale > 0.5 else 1)
        cv2.putText(img, text, ((width - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    2 if scale > 0.5 else 1)
    L = 26
    for (x, y, dx, dy) in [(6, 6, 1, 1), (width - 7, 6, -1, 1), (6, height - 7, 1, -1), (width - 7, height - 7, -1, -1)]:
        cv2.line(img, (x, y), (x + dx * L, y), (102, 255, 0), 2)
        cv2.line(img, (x, y), (x, y + dy * L), (102, 255, 0), 2)
    return img


def draw_target_lock(frame, landmarks, eyes_closed, left_eye, right_eye, locked_text="TARGET LOCK"):
    """
    Draws a targeting overlay on the video frame (in place):
      - corner brackets around the whole face
      - a small crosshair on the nose tip
      - outlines of both eyes: green when open, red when closed
    landmarks: MediaPipe's normalised face landmarks (x, y in 0-1).
    left_eye / right_eye: the 6 landmark indices used for EAR.
    Purely visual; detection has already happened before this is called.
    """
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, y1 = int(max(min(xs) - 12, 0)), int(max(min(ys) - 18, 0))
    x2, y2 = int(min(max(xs) + 12, w - 1)), int(min(max(ys) + 12, h - 1))
    green, red = (102, 255, 0), (60, 60, 255)
    L = max(14, (x2 - x1) // 6)
    for (x, y, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (x, y), (x + dx * L, y), green, 2)
        cv2.line(frame, (x, y), (x, y + dy * L), green, 2)
    cv2.putText(frame, locked_text, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, green, 1)

    nose = landmarks[1]
    nx, ny = int(nose.x * w), int(nose.y * h)
    cv2.circle(frame, (nx, ny), 5, green, 1)
    cv2.line(frame, (nx - 10, ny), (nx - 6, ny), green, 1)
    cv2.line(frame, (nx + 6, ny), (nx + 10, ny), green, 1)
    cv2.line(frame, (nx, ny - 10), (nx, ny - 6), green, 1)
    cv2.line(frame, (nx, ny + 6), (nx, ny + 10), green, 1)

    eye_color = red if eyes_closed else green
    for idx in (left_eye, right_eye):
        # EAR points are ordered p1..p6 as: corner, top, top, corner, bottom, bottom
        order = [idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]]
        pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in order], np.int32)
        cv2.polylines(frame, [pts], True, eye_color, 1, cv2.LINE_AA)


# ======================================================================
#   THEME ADDITIONS (display only): animated standby screen, typewriter
#   ticker, window icon, cinematic video overlay.
# ======================================================================

import random

from PyQt6.QtGui import QIcon, QPixmap


class StandbyAnimator:
    """
    Produces the animated idle screen shown in the video area when no
    camera or file is running:
      1. a short boot sequence (status lines typing out), then
      2. "Morse rain": columns of falling dots and dashes, a rotating
         radar sweep, and the AWAITING SIGNAL SOURCE message.
    Drawn with OpenCV into a normal image, so it is cheap and only runs
    while the app is idle.
    """

    BOOT_LINES = [
        "> INITIALISING INTERFACE ............ OK",
        "> LOADING DECODER MODULES ........... OK",
        "> MORSE TABLE + CODEBOOK ............ OK",
        "> OFFLINE MODE ...................... ENGAGED",
        "> AWAITING SIGNAL SOURCE",
    ]
    BOOT_SECONDS = 2.6

    def __init__(self, width=640, height=360, seed=7):
        self.w, self.h = width, height
        rng = random.Random(seed)
        self.col_w = 18
        self.columns = []
        for x in range(8, width, self.col_w):
            self.columns.append({
                "x": x,
                "speed": rng.uniform(40, 120),          # pixels per second
                "offset": rng.uniform(0, height + 200),
                "symbols": [rng.choice(".-") for _ in range(40)],
                "length": rng.randint(6, 14),
            })
        self.subtitle = "SELECT A SIGNAL SOURCE ABOVE"
        self.boot_start = None
        self._base = self._make_base()

    def restart_boot(self):
        self.boot_start = None

    def _make_base(self):
        img = np.zeros((self.h, self.w, 3), np.uint8)
        img[:] = (4, 8, 4)
        for x in range(0, self.w, 32):
            cv2.line(img, (x, 0), (x, self.h), (12, 26, 12), 1)
        for y in range(0, self.h, 32):
            cv2.line(img, (0, y), (self.w, y), (12, 26, 12), 1)
        return img

    def _draw_symbol(self, img, sym, x, y, color):
        if sym == ".":
            cv2.circle(img, (x + 4, y), 2, color, -1, cv2.LINE_AA)
        else:
            cv2.line(img, (x, y), (x + 9, y), color, 2, cv2.LINE_AA)

    def _corners(self, img):
        L, g = 26, (102, 255, 0)
        w, h = self.w, self.h
        for (x, y, dx, dy) in [(6, 6, 1, 1), (w - 7, 6, -1, 1), (6, h - 7, 1, -1), (w - 7, h - 7, -1, -1)]:
            cv2.line(img, (x, y), (x + dx * L, y), g, 2)
            cv2.line(img, (x, y), (x, y + dy * L), g, 2)

    def render(self, t):
        if self.boot_start is None:
            self.boot_start = t
        img = self._base.copy()
        elapsed = t - self.boot_start

        if elapsed < self.BOOT_SECONDS:
            chars = int(elapsed * 95)
            y = 70
            for line in self.BOOT_LINES:
                shown = line[:max(0, chars)]
                chars -= len(line) + 6
                if shown:
                    ok = shown.endswith("OK") or shown.endswith("ENGAGED")
                    cv2.putText(img, shown, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (160, 255, 190) if ok else (102, 255, 0), 1, cv2.LINE_AA)
                y += 34
            if int(t * 3) % 2 == 0:
                cv2.rectangle(img, (40, y - 12), (50, y), (102, 255, 0), -1)
            self._corners(img)
            return img

        # ---- Morse rain ----
        spacing = 16
        for col in self.columns:
            head = (col["offset"] + elapsed * col["speed"]) % (self.h + col["length"] * spacing + 40)
            for k in range(col["length"]):
                y = int(head - k * spacing)
                if 0 <= y < self.h:
                    fade = 1.0 - k / col["length"]
                    sym = col["symbols"][(int(head / spacing) - k) % len(col["symbols"])]
                    if k == 0:
                        color = (220, 255, 230)
                    else:
                        color = (0, int(60 + 150 * fade), int(20 + 40 * fade))
                    self._draw_symbol(img, sym, col["x"], y, color)

        # ---- dark centre plate so the text stays readable ----
        cx, cy = self.w // 2, self.h // 2 - 16
        plate = img.copy()
        cv2.circle(plate, (cx, cy), 96, (2, 6, 2), -1)
        cv2.rectangle(plate, (cx - 190, cy + 88), (cx + 190, cy + 140), (2, 6, 2), -1)
        cv2.addWeighted(plate, 0.94, img, 0.06, 0, img)

        # ---- radar sweep ----
        angle = (elapsed * 90) % 360
        sweep = img.copy()
        cv2.ellipse(sweep, (cx, cy), (86, 86), 0, angle - 40, angle, (0, 120, 40), -1, cv2.LINE_AA)
        cv2.addWeighted(sweep, 0.35, img, 0.65, 0, img)
        for r in (86, 60, 34):
            cv2.circle(img, (cx, cy), r, (30, 110, 45), 1, cv2.LINE_AA)
        cv2.line(img, (cx - 100, cy), (cx + 100, cy), (30, 90, 40), 1)
        cv2.line(img, (cx, cy - 100), (cx, cy + 100), (30, 90, 40), 1)
        a = np.radians(angle)
        cv2.line(img, (cx, cy), (int(cx + 86 * np.cos(a)), int(cy + 86 * np.sin(a))), (102, 255, 0), 1, cv2.LINE_AA)

        # ---- eye emblem (blinks now and then) ----
        blink = (elapsed % 4.0) < 0.18
        eye_h = 2 if blink else 20
        eye = np.array([[cx - 44, cy], [cx, cy - eye_h], [cx + 44, cy], [cx, cy + eye_h]], np.int32)
        cv2.polylines(img, [eye], True, (102, 255, 0), 2, cv2.LINE_AA)
        if not blink:
            cv2.circle(img, (cx, cy), 8, (102, 255, 0), -1, cv2.LINE_AA)

        # ---- text ----
        title = "AWAITING SIGNAL SOURCE"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(img, title, ((self.w - tw) // 2, cy + 116), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (102, 255, 0), 2, cv2.LINE_AA)
        if int(t * 2) % 2 == 0:
            cv2.rectangle(img, ((self.w + tw) // 2 + 6, cy + 100), ((self.w + tw) // 2 + 16, cy + 118),
                          (102, 255, 0), -1)
        (sw, _), _ = cv2.getTextSize(self.subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(img, self.subtitle, ((self.w - sw) // 2, cy + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (90, 170, 100), 1, cv2.LINE_AA)
        self._corners(img)
        return img


class TypewriterTicker(QLabel):
    """Header ticker that types out short system messages one after
    another. get_messages is a function returning the current list, so
    messages can reflect real state (e.g. offline mode on or off)."""

    def __init__(self, get_messages, parent=None):
        super().__init__(parent)
        self.get_messages = get_messages
        self._msg_index = 0
        self._chars = 0
        self._hold = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(45)

    def _tick(self):
        messages = self.get_messages() or [""]
        msg = messages[self._msg_index % len(messages)]
        if self._chars < len(msg):
            self._chars += 1
        elif self._hold < 55:            # ~2.5 s pause on the full message
            self._hold += 1
        else:
            self._msg_index += 1
            self._chars, self._hold = 0, 0
        cursor = "_" if (self._chars < len(msg) or (self._hold // 8) % 2 == 0) else " "
        super().setText(">> " + msg[:self._chars] + cursor)


def make_emblem_icon(size=64):
    """Window icon: the eye emblem, drawn once."""
    pix = QPixmap(size, size)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    _antialias(p)
    c, r = size / 2, size / 2 - 3
    p.setPen(QPen(ACCENT, 3))
    p.setBrush(QBrush(QColor("#021a0b")))
    p.drawEllipse(QPointF(c, c), r, r)
    eye = QPolygonF([QPointF(c - r * 0.75, c), QPointF(c, c - r * 0.38),
                     QPointF(c + r * 0.75, c), QPointF(c, c + r * 0.38)])
    p.setPen(QPen(ACCENT, 3))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolygon(eye)
    p.setBrush(QBrush(ACCENT))
    p.drawEllipse(QPointF(c, c), r * 0.16, r * 0.16)
    p.end()
    return QIcon(pix)


_vignette_cache = {}


def apply_cinematic(frame, frame_counter, timecode_s, mode_label):
    """
    Cinematic look for the live/recorded video (display only; detection
    has already run on the clean frame before this is called):
      - soft vignette (darker edges), subtle scan-line texture
      - timecode and frame counter
    """
    h, w = frame.shape[:2]
    key = (h, w)
    if key not in _vignette_cache:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((xs - w / 2) / (w / 2)) ** 2 + ((ys - h / 2) / (h / 2)) ** 2)
        mask = np.clip(1.08 - 0.42 * d ** 2, 0.45, 1.0).astype(np.float32)
        _vignette_cache[key] = cv2.merge([mask, mask, mask])
    frame[:] = (frame.astype(np.float32) * _vignette_cache[key]).astype(np.uint8)
    frame[::3] = (frame[::3] * 0.86).astype(np.uint8)

    mins, secs = divmod(max(timecode_s, 0.0), 60)
    hrs, mins = divmod(int(mins), 60)
    tc = f"T+{hrs:02d}:{mins:02d}:{secs:04.1f}"
    cv2.putText(frame, tc, (w - 150, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FRM {frame_counter:06d}  {mode_label}", (w - 215, h - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 190, 130), 1, cv2.LINE_AA)
