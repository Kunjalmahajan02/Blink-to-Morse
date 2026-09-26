"""
speech.py  (v4 -- offline-first speech)

Speaks decoded text. Backends, in order of preference:

OFFLINE (nothing leaves the computer):
  1. Windows built-in voices (SAPI5, via pyttsx3). English is always
     available; some Indian languages (e.g. Hindi) can be added in
     Windows Settings > Time & language > Speech.
  2. eSpeak NG, if installed (free, offline, supports most Indian
     languages; robotic-sounding but understandable).
  If neither has a voice for the language, it says so instead of
  quietly going online.

ONLINE (only when offline mode is switched off):
  3. Google TTS (gTTS) - best quality for Indian languages, but sends
     the text to Google's servers.

Speech runs on a background thread so the video never freezes.
Windows voices need COM initialised on that thread (pythoncom), which
is what caused silent speech in an earlier version.

Self-test: python speech.py   (lists the offline voices Python can see)
"""

import os
import shutil
import subprocess
import tempfile
import threading

try:
    import pythoncom
    HAS_PYTHONCOM = True
except ImportError:
    HAS_PYTHONCOM = False

ESPEAK_CANDIDATES = [
    "espeak-ng",
    r"C:\Program Files\eSpeak NG\espeak-ng.exe",
    r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
]


def _short(lang_code):
    """'hi-IN' -> 'hi'. Different services use different code styles."""
    return (lang_code or "en").split("-")[0].lower()


class SpeechEngine:
    def __init__(self, rate=160, volume=1.0):
        self.rate = rate
        self.volume = volume
        self._voices = None        # cached list of (id, name, languages) Windows voices
        self._espeak = None        # cached path to espeak-ng, or "" if not found

    # ---------------- discovering offline voices ----------------

    def _windows_voices(self):
        """Lists Windows voices once, on a separate thread (COM rule)."""
        if self._voices is not None:
            return self._voices
        found = []

        def scan():
            if HAS_PYTHONCOM:
                pythoncom.CoInitialize()
            try:
                import pyttsx3
                engine = pyttsx3.init()
                for v in engine.getProperty("voices"):
                    langs = [str(l) for l in (getattr(v, "languages", None) or [])]
                    found.append((v.id, v.name or "", langs))
                engine.stop()
                del engine
            except Exception as e:
                print(f"[speech] Could not list Windows voices: {e}")
            finally:
                if HAS_PYTHONCOM:
                    pythoncom.CoUninitialize()

        t = threading.Thread(target=scan, daemon=True)
        t.start()
        t.join(timeout=8)
        self._voices = found
        return found

    def _espeak_path(self):
        if self._espeak is None:
            self._espeak = ""
            for cand in ESPEAK_CANDIDATES:
                path = shutil.which(cand) or (cand if os.path.isfile(cand) else None)
                if path:
                    self._espeak = path
                    break
        return self._espeak

    def find_offline_voice(self, lang_code, lang_name=None):
        """Returns ("windows", voice_id), ("espeak", path) or None."""
        short = _short(lang_code)
        name = (lang_name or "").lower()
        for vid, vname, langs in self._windows_voices():
            text = f"{vid} {vname} {' '.join(langs)}".lower()
            if (f"{short}-" in text or f"_{short}-" in text or f"{short}_" in text
                    or (name and name in text)):
                return ("windows", vid)
        if short == "en":
            voices = self._windows_voices()
            if voices:
                return ("windows", voices[0][0])
        if self._espeak_path():
            return ("espeak", self._espeak_path())
        return None

    # ---------------- speaking ----------------

    def speak(self, text, lang_code="en", offline=True, lang_name=None):
        """
        Starts speaking in the background.
        Returns (True, description) if speech was started, or
        (False, reason) if it cannot be spoken under current settings.
        """
        text = text.strip()
        if not text:
            return False, "nothing to speak"
        short = _short(lang_code)

        if short == "en":
            threading.Thread(target=self._speak_windows, args=(text, None), daemon=True).start()
            return True, "offline English voice"

        voice = self.find_offline_voice(lang_code, lang_name)
        if voice:
            kind, ref = voice
            if kind == "windows":
                threading.Thread(target=self._speak_windows, args=(text, ref), daemon=True).start()
                return True, "offline Windows voice"
            threading.Thread(target=self._speak_espeak, args=(text, short, ref), daemon=True).start()
            return True, "offline eSpeak NG voice"

        if offline:
            return False, f"no offline voice installed for {lang_name or short}"
        threading.Thread(target=self._speak_gtts, args=(text, short), daemon=True).start()
        return True, "online Google voice"

    def _speak_windows(self, text, voice_id):
        if HAS_PYTHONCOM:
            pythoncom.CoInitialize()
        try:
            import pyttsx3
            print(f"[speech/offline] Speaking: \"{text}\"")
            engine = pyttsx3.init()
            if voice_id:
                engine.setProperty("voice", voice_id)
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[speech/offline] ERROR: {e}")
        finally:
            if HAS_PYTHONCOM:
                pythoncom.CoUninitialize()

    def _speak_espeak(self, text, short, path):
        try:
            print(f"[speech/espeak-{short}] Speaking: \"{text}\"")
            subprocess.run([path, "-v", short, text], check=False,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception as e:
            print(f"[speech/espeak] ERROR: {e}")

    def _speak_gtts(self, text, short):
        tmp_path = None
        try:
            from gtts import gTTS
            import pygame
            print(f"[speech/online-{short}] Speaking: \"{text}\"")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            gTTS(text=text, lang=short, slow=False).save(tmp_path)
            pygame.mixer.init()
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(50)
            pygame.mixer.quit()
        except ImportError as e:
            print(f"[speech/online] Missing library: {e}  (pip install gTTS pygame)")
        except Exception as e:
            print(f"[speech/online] ERROR: {e}")
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass


if __name__ == "__main__":
    eng = SpeechEngine()
    print("Windows voices Python can see:")
    for vid, name, langs in eng._windows_voices():
        print(f"  - {name}  {langs}")
    print("eSpeak NG:", eng._espeak_path() or "not installed")
    for lang, name in [("hi-IN", "Hindi"), ("ta-IN", "Tamil"), ("mr-IN", "Marathi")]:
        print(f"Offline voice for {name}:", eng.find_offline_voice(lang, name) or "none")
