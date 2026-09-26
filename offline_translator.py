"""
offline_translator.py

Turns a decoded Morse message into the analyst's language WITHOUT any
internet connection, so a decoded message never leaves the computer.

WHY NOT A FULL OFFLINE TRANSLATION MODEL?
Offline neural translators for Indian languages exist (e.g. AI4Bharat's
IndicTrans2), but they are large and slow on an ordinary laptop. Morse
messages sent by blinking are short and mostly predictable (HELP,
CAPTURED, I AM SAFE...), so two lightweight techniques cover them:

1. PHRASEBOOK (phrasebook.json)
   A local table of common emergency phrases with their translations.
   The message is matched longest-phrase-first, so "I AM SAFE" is
   translated as one phrase rather than word by word.
   Each entry records whether a native speaker has verified it.

2. TRANSLITERATION (for everything not in the phrasebook)
   Names, places and codewords should not be translated at all -
   "ISHA" is a name, not a word to look up. Instead they are written
   in the target script using the offline `indic-transliteration`
   library (e.g. ISHA -> an approximate Devanagari spelling).
   This is approximate phonetics, and is labelled as such.
   Urdu and Sindhi use Arabic script, which this library does not
   support, so those words are left in the original Latin letters.

Every output piece is tagged with its source (phrasebook /
transliterated / original) so the analyst always knows what was looked
up, what was sound-converted, and what was left untouched.
"""

import json
import os

PHRASEBOOK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phrasebook.json")
MAX_PHRASE_WORDS = 4

# Target script for transliteration, per language (names as in translator.py)
SCRIPT_FOR_LANGUAGE = {
    "Hindi": "DEVANAGARI", "Marathi": "DEVANAGARI", "Nepali": "DEVANAGARI",
    "Bengali": "BENGALI", "Assamese": "BENGALI",
    "Gujarati": "GUJARATI", "Punjabi": "GURMUKHI",
    "Kannada": "KANNADA", "Malayalam": "MALAYALAM",
    "Odia": "ORIYA", "Tamil": "TAMIL", "Telugu": "TELUGU",
    # Urdu, Sindhi (Arabic script) and non-Indian languages: no transliteration
}

_phrasebook_cache = None


def load_phrasebook():
    global _phrasebook_cache
    if _phrasebook_cache is None:
        try:
            with open(PHRASEBOOK_PATH, encoding="utf-8") as f:
                _phrasebook_cache = json.load(f)
        except (OSError, ValueError):
            _phrasebook_cache = {"phrases": [], "translations": {}}
    return _phrasebook_cache


def reload_phrasebook():
    global _phrasebook_cache
    _phrasebook_cache = None
    return load_phrasebook()


def english_to_itrans(word):
    """
    Converts an English-style spelling into ITRANS, the romanisation
    scheme the transliteration library reads. ITRANS is not English
    spelling (capital letters mean long vowels, etc.), so a few simple
    rules make common Indian names come out closer to how they sound:
        ee -> I (long i), oo -> U (long u), aa -> A (long a),
        a word-final 'a' -> A (Isha, Priya, Asha end in a long 'aa'),
        ck -> k, ph -> f, c (not ch) -> k, q -> k, w -> v, x -> ks
    Still approximate: English spelling doesn't map exactly to sounds.
    """
    w = word.lower()
    w = w.replace("ck", "k").replace("ph", "f")
    w = w.replace("ee", "I").replace("oo", "U").replace("aa", "A")
    out = []
    for i, ch in enumerate(w):
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if ch == "c" and nxt != "h":
            out.append("k")
        elif ch == "q":
            out.append("k")
        elif ch == "w":
            out.append("v")
        elif ch == "x":
            out.append("ks")
        else:
            out.append(ch)
    w = "".join(out)
    if w.endswith("a") and len(w) > 1:
        w = w[:-1] + "A"
    return w


def transliterate_word(word, language):
    """Returns the word in the language's script, or None if not possible."""
    script = SCRIPT_FOR_LANGUAGE.get(language)
    if not script or not word.isalpha():
        return None
    try:
        from indic_transliteration import sanscript
        return sanscript.transliterate(english_to_itrans(word), sanscript.ITRANS, getattr(sanscript, script))
    except Exception:
        return None


def translate_offline(text, language):
    """
    Returns a dict:
        text:   the rendered message
        pieces: list of {"source": str, "original": str, "output": str, "verified": bool|None}
        summary: short human-readable description of what was done
    """
    words = [w for w in text.upper().replace("?", " ").split() if w]
    table = load_phrasebook().get("translations", {}).get(language, {})
    pieces = []
    i = 0
    while i < len(words):
        match = None
        for n in range(min(MAX_PHRASE_WORDS, len(words) - i), 0, -1):
            phrase = " ".join(words[i:i + n])
            if phrase in table:
                match = (phrase, n)
                break
        if match:
            phrase, n = match
            entry = table[phrase]
            pieces.append({"source": "phrasebook", "original": phrase,
                           "output": entry["text"], "verified": entry.get("verified", False)})
            i += n
            continue
        word = words[i]
        translit = transliterate_word(word, language)
        if translit:
            pieces.append({"source": "transliterated", "original": word, "output": translit, "verified": None})
        else:
            pieces.append({"source": "original", "original": word, "output": word, "verified": None})
        i += 1

    counts = {}
    for p in pieces:
        counts[p["source"]] = counts.get(p["source"], 0) + 1
    parts = []
    if counts.get("phrasebook"):
        unverified = sum(1 for p in pieces if p["source"] == "phrasebook" and not p["verified"])
        note = f", {unverified} unverified" if unverified else ""
        parts.append(f"{counts['phrasebook']} from phrasebook{note}")
    if counts.get("transliterated"):
        parts.append(f"{counts['transliterated']} transliterated (approximate)")
    if counts.get("original"):
        parts.append(f"{counts['original']} left as original")
    return {"text": " ".join(p["output"] for p in pieces), "pieces": pieces,
            "summary": "; ".join(parts) if parts else "nothing to translate"}
