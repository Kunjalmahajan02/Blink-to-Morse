"""
translator.py

Handles translation of decoded Morse text into any target language,
with special emphasis on Indian languages.

WHY THIS MATTERS FOR THE ARMY USE CASE:
Standard Morse code maps dots/dashes to Latin letters only (A-Z, 0-9).
There is no standard Morse encoding for Devanagari, Tamil script, etc.
So the pipeline always decodes to Latin first, and this module then
translates to the analyst's preferred language.

Example:
  Agent blinks: ... --- ...  (SOS in standard Morse)
  System decodes: "SOS"
  Translator renders: "एसओएस" (Hindi) or "எஸ்ஓஎஸ்" (Tamil)

Translation uses Google Translate via the deep-translator library,
which supports 100+ languages including all major Indian languages.
Requires an internet connection for non-English languages.

OFFLINE FALLBACK:
If translation fails (no internet, API limit, etc.), the system
returns the original decoded text unchanged and logs a warning --
it degrades gracefully rather than crashing.
"""

SUPPORTED_LANGUAGES = {
    # --- Indian languages (primary focus) ---
    # MyMemory requires locale codes (e.g. "hi-IN" not just "hi")
    "Hindi": "hi-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Marathi": "mr-IN",
    "Bengali": "bn-IN",
    "Gujarati": "gu-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Punjabi": "pa-IN",
    "Odia": "or-IN",
    "Urdu": "ur-PK",
    "Nepali": "ne-NP",
    "Assamese": "as-IN",
    "Sindhi": "sd-PK",
    # --- International ---
    "English (original)": "en",
    "Arabic": "ar-SA",
    "Russian": "ru-RU",
    "French": "fr-FR",
    "Spanish": "es-ES",
    "German": "de-DE",
    "Chinese (Simplified)": "zh-CN",
    "Japanese": "ja-JP",
    "Persian": "fa-IR",
}

# Group Indian languages first in the dropdown
INDIAN_LANGUAGE_NAMES = [
    "Hindi", "Tamil", "Telugu", "Marathi", "Bengali", "Gujarati",
    "Kannada", "Malayalam", "Punjabi", "Odia", "Urdu",
    "Nepali", "Assamese", "Sindhi",
]


def translate(text, target_lang_code):
    """
    Translates text to the target language using MyMemory API.

    MyMemory is used instead of Google Translate because it has a much
    more generous free tier (1000 words/day, no API key needed) and
    doesn't rate-limit casual demo usage the way Google's unofficial
    API does.

    Args:
        text: decoded Morse text (always in English/Latin initially)
        target_lang_code: ISO 639-1 language code, e.g. "hi" for Hindi

    Returns:
        (translated_text, success_bool, error_message_or_None)
    """
    text = text.strip()
    if not text:
        return text, True, None

    if target_lang_code in ("en", "en-US"):
        return text, True, None

    try:
        from deep_translator import MyMemoryTranslator
        translated = MyMemoryTranslator(
            source="en-US",
            target=target_lang_code,
        ).translate(text)
        return translated, True, None
    except ImportError:
        return text, False, "deep-translator not installed. Run: pip install deep-translator"
    except Exception as e:
        return text, False, str(e)
