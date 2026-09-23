"""
logger.py  (v2 -- fixes the "file doesn't appear" bug)

THE BUG: the log file was created using a relative path
("session_log.txt"), which gets placed wherever your TERMINAL happens
to be pointed when you run the script -- not necessarily your project
folder. If you ran python from a different directory, the file was
being created there instead (or the write silently went somewhere you
weren't looking).

THE FIX: build an absolute path based on where THIS SCRIPT FILE lives,
so the log always ends up next to your other project files no matter
where you run the command from.
"""

import os
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_log.txt")


def log_text(text):
    """Appends one timestamped line to the session log file."""
    text = text.strip()
    if not text:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")
    print(f"[logger] Saved entry to: {LOG_FILE}")
