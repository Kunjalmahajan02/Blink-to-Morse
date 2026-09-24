"""
speech.py  (v3 -- multilingual TTS)

Two TTS backends:
- pyttsx3 (offline) for English -- same as before, no internet needed.
- gTTS (Google TTS) for all other languages, especially Indian ones.
  gTTS produces dramatically better results for Hindi, Tamil, Telugu,
  etc. than pyttsx3, which on Windows only has English voices by
  default. Requires an internet connection.

Both still run on a background thread so they don't freeze the video.
"""

import threading
import os
import tempfile

try:
    import pythoncom
    HAS_PYTHONCOM = True
except ImportError:
    HAS_PYTHONCOM = False


class SpeechEngine:
    def __init__(self, rate=160, volume=1.0):
        self.rate = rate
        self.volume = volume

    def speak(self, text, lang_code="en"):
        """Speaks text in the given language. lang_code follows BCP-47
        (e.g. 'hi' for Hindi, 'ta' for Tamil, 'en' for English)."""
        text = text.strip()
        if not text:
            return
        if lang_code == "en":
            threading.Thread(
                target=self._speak_pyttsx3, args=(text,), daemon=True
            ).start()
        else:
            threading.Thread(
                target=self._speak_gtts, args=(text, lang_code), daemon=True
            ).start()

    def _speak_pyttsx3(self, text):
        """Offline English TTS via Windows SAPI5."""
        if HAS_PYTHONCOM:
            pythoncom.CoInitialize()
        try:
            import pyttsx3
            print(f"[speech/en] Speaking: \"{text}\"")
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[speech/en] ERROR: {e}")
        finally:
            if HAS_PYTHONCOM:
                pythoncom.CoUninitialize()

    def _speak_gtts(self, text, lang_code):
        """Online Indian/multilingual TTS via Google (gTTS)."""
        tmp_path = None
        try:
            from gtts import gTTS
            import pygame

            print(f"[speech/{lang_code}] Speaking: \"{text}\"")

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name

            gTTS(text=text, lang=lang_code, slow=False).save(tmp_path)

            pygame.mixer.init()
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(50)
            pygame.mixer.quit()

        except ImportError as e:
            print(f"[speech/{lang_code}] Missing library: {e}")
            print("  Run: pip install gTTS pygame")
        except Exception as e:
            print(f"[speech/{lang_code}] ERROR: {e}")
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
