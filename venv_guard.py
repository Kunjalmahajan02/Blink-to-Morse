"""
venv_guard.py

Makes the project start correctly however it is launched: typing
`python gui.py` in a terminal, VS Code's Run button, or run_blinkbridge.bat.

WHY
This project runs in its own Python environment (.venv) with the exact
library versions it needs (MediaPipe 0.10.21 needs protobuf below 5).
The computer's shared Python can have different versions installed by
other projects (for example, TensorFlow installs protobuf 7), which makes
MediaPipe crash. Typing `python gui.py` normally uses that shared Python.

HOW
Every runnable script imports this module FIRST, before any other library.
If the script is not already running inside the project's .venv, and a
.venv exists, it restarts the same script (with the same arguments) using
the .venv's Python, waits for it to finish, and exits with its result.
Inside .venv it does nothing.

If there is no .venv, nothing happens and the script runs with whichever
Python started it, so the project still works on a computer set up
without a separate environment.

It only uses Python's built-in modules, so it works even in a Python
where the project's libraries are missing or broken.
"""

import os
import subprocess
import sys

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_VENV_DIR = os.path.join(_PROJECT_DIR, ".venv")
if os.name == "nt":
    _VENV_PYTHON = os.path.join(_VENV_DIR, "Scripts", "python.exe")
else:
    _VENV_PYTHON = os.path.join(_VENV_DIR, "bin", "python")

_MARKER = "BLINKBRIDGE_RELAUNCHED"


def _same_path(a, b):
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def ensure_venv():
    if not os.path.exists(_VENV_PYTHON):
        return  # no project environment: run as-is
    if _same_path(sys.prefix, _VENV_DIR):
        return  # already running inside .venv
    if os.environ.get(_MARKER):
        return  # safety: never restart twice (prevents an endless loop)
    script = getattr(sys.modules.get("__main__"), "__file__", None)
    if not script:
        return  # interactive Python or `python -c`: leave it alone

    print("[BlinkBridge] Switching to the project's own Python environment (.venv)...", flush=True)
    env = dict(os.environ, **{_MARKER: "1"})
    try:
        code = subprocess.call([_VENV_PYTHON, os.path.abspath(script)] + sys.argv[1:], env=env)
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)


ensure_venv()
