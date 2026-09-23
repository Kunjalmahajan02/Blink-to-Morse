"""
speech.py  (v2 -- fixes silent audio on Windows)

THE BUG: pyttsx3 uses Windows' built-in speech engine (SAPI5) through
a library called pywin32, which relies on something called "COM."
COM has to be explicitly initialized on whichever thread is using it.
Our speech runs on a background thread (so it doesn't freeze the
video), but that thread never initialized COM -- so Windows silently
did nothing instead of speaking.

THE FIX: call pythoncom.CoInitialize() at the start of the speaking
thread, and CoUninitialize() when it's done.
"""

import threading
import pyttsx3

try:
    import pythoncom
    HAS_PYTHONCOM = True
except ImportError:
    # Not on Windows, or pywin32 isn't installed -- speech may still
    # work (e.g. on Mac/Linux pyttsx3 uses a different backend), we
    # just skip the COM calls in that case.
    HAS_PYTHONCOM = False


class SpeechEngine:
    def __init__(self, rate=160, volume=1.0):
        self.rate = rate
        self.volume = volume

    def speak(self, text):
        """Speaks the given text without blocking the main program."""
        text = text.strip()
        if not text:
            return
        threading.Thread(target=self._speak_now, args=(text,), daemon=True).start()

    def _speak_now(self, text):
        if HAS_PYTHONCOM:
            pythoncom.CoInitialize()
        try:
            print(f"[speech] Speaking: \"{text}\"")
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            # Printing the real error is important for debugging --
            # silent failures are exactly what caused this bug.
            print(f"[speech] ERROR while speaking: {e}")
        finally:
            if HAS_PYTHONCOM:
                pythoncom.CoUninitialize()
