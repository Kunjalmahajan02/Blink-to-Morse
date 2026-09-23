"""
speech.py

Speaks decoded text out loud using pyttsx3, an offline text-to-speech
library (no internet connection needed, works well on Windows).

Speech runs on a background thread so it doesn't freeze the webcam
video while talking.
"""

import threading
import pyttsx3


class SpeechEngine:
    def __init__(self, rate=160):
        self.rate = rate

    def speak(self, text):
        """Speaks the given text without blocking the main program."""
        text = text.strip()
        if not text:
            return
        threading.Thread(target=self._speak_now, args=(text,), daemon=True).start()

    def _speak_now(self, text):
        # A fresh engine instance per call avoids threading issues that
        # can happen when reusing one pyttsx3 engine across threads.
        engine = pyttsx3.init()
        engine.setProperty('rate', self.rate)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
