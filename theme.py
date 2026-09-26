"""
theme.py

The visual theme for the main window: "Classified Night-Ops
Intelligence Console". DESIGN ONLY: colours, fonts, borders and
gradients. Nothing here changes what any button, option or command does.

Everything visual lives here so the look is consistent and easy to
adjust in one place.
"""

# ---------------- palette ----------------
BG = "#010503"            # near-black with a green tint
PANEL_TOP = "#07170d"     # panel gradient (top)
PANEL_BOTTOM = "#020905"  # panel gradient (bottom)
BORDER = "#0f5c2e"        # dim neon border
ACCENT = "#00ff66"        # primary neon green
ACCENT_SOFT = "#7dffb0"   # hover / highlight
INFO = "#3fe0c5"          # cool cyan for secondary information
AMBER = "#ffb020"         # warnings
DANGER = "#ff2d55"        # crimson for danger / terminate
MUTED = "#3d8b52"         # quiet captions
TEXT = "#c8ffd9"          # body text

# BGR versions for OpenCV drawing on video frames
ACCENT_BGR = (102, 255, 0)
DANGER_BGR = (85, 45, 255)
INFO_BGR = (197, 224, 63)
AMBER_BGR = (32, 176, 255)

# Windows 10/11 ships "Bahnschrift", a condensed DIN-style font with a
# military stencil feel; if it is missing, Qt falls back automatically.
TITLE_FONT = "Bahnschrift"
MONO_FONT = "Consolas"

STYLE_SHEET = f"""
QWidget {{
    background-color: {BG};
    color: {ACCENT};
    font-family: {MONO_FONT};
}}
QToolTip {{
    background-color: #02130a;
    color: {ACCENT};
    border: 1px solid {ACCENT};
    padding: 4px;
}}
QPushButton {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0a3a1d, stop:1 #02130a);
    color: {ACCENT};
    border: 1px solid #13823f;
    border-left: 4px solid {ACCENT};
    border-radius: 3px;
    padding: 11px 16px;
    font-weight: bold;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #11663a, stop:1 #04200f);
    color: #eafff1;
    border: 1px solid {ACCENT_SOFT};
    border-left: 4px solid {ACCENT_SOFT};
}}
QPushButton:pressed {{
    background-color: #00ff66;
    color: #001a0a;
}}
QFrame#videoFrame {{
    border: 2px solid {ACCENT};
    border-radius: 4px;
    background-color: #000000;
}}
QFrame#readoutPanel {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {PANEL_TOP}, stop:1 {PANEL_BOTTOM});
    border: 1px solid {BORDER};
    border-top: 2px solid {ACCENT};
    border-radius: 4px;
}}
QComboBox {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0a2a16, stop:1 #02100a);
    color: {ACCENT};
    border: 1px solid #13823f;
    border-radius: 3px;
    padding: 4px 8px;
    font-family: {MONO_FONT};
    font-size: 11px;
}}
QComboBox:hover {{
    border: 1px solid {ACCENT_SOFT};
}}
QComboBox QAbstractItemView {{
    background-color: #02100a;
    color: {ACCENT};
    selection-background-color: #0f5c2e;
    selection-color: #eafff1;
    border: 1px solid {ACCENT};
}}
QCheckBox {{
    color: {ACCENT};
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {ACCENT};
    border-radius: 2px;
    background-color: #02130a;
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
}}
QMessageBox QLabel {{
    color: {TEXT};
}}
"""

TERMINATE_STYLE = f"""
QPushButton {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3a0010, stop:1 #140005);
    color: {DANGER};
    border: 1px solid #8a1030;
    border-left: 4px solid {DANGER};
    border-radius: 3px;
    padding: 11px 16px;
    font-weight: bold;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6a0020, stop:1 #200008);
    color: #ffe3ea;
    border: 1px solid #ff6b8a;
    border-left: 4px solid #ff6b8a;
}}
QPushButton:pressed {{
    background-color: {DANGER};
    color: #1a0005;
}}
"""

# Empty on purpose: dropdowns now take their look from STYLE_SHEET above.
COMBO_STYLE = ""

TITLE_STYLE = (f"font-family: '{TITLE_FONT}'; font-size: 26px; font-weight: bold;"
               f" color: {ACCENT}; background: transparent;")
SUBTITLE_STYLE = f"color: {MUTED}; font-size: 10px; background: transparent;"
BANNER_STYLE = ("background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2a0008,"
                " stop:0.5 #5c0014, stop:1 #2a0008); color: #ffc9d4; font-size: 10px;"
                " font-weight: bold; padding: 3px; border-top: 1px solid #ff2d55;"
                " border-bottom: 1px solid #ff2d55;")
TICKER_STYLE = f"color: {INFO}; font-size: 11px; background: transparent;"
LABEL_HEADING_STYLE = f"color: {ACCENT}; font-weight: bold; background: transparent;"
