"""
logger.py

Keeps a simple timestamped record of everything decoded during a
session, saved to a text file. Useful for reviewing past sessions or
demonstrating the project's history to your guide.
"""

from datetime import datetime

LOG_FILE = "session_log.txt"


def log_text(text):
    """Appends one timestamped line to the session log file."""
    text = text.strip()
    if not text:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")
