"""
build_phrasebook.py

Fills in the offline phrasebook (phrasebook.json) for every language.
This is the ONLY part of the project that deliberately uses the
internet, and it's run once, by you, while connected. After that the
app translates common messages fully offline.

RUN:  python build_phrasebook.py

- Skips entries that already exist (so hand-written or verified
  entries are never overwritten).
- Saves after every entry, so if it stops (internet drops, or the free
  translation service's daily limit is reached) you just run it again
  later and it continues where it left off.
- Machine translations are saved as "verified": false. Emergency
  phrases should be checked by a native speaker before being relied on;
  once checked, change "verified" to true in phrasebook.json.

The phrase list itself is in phrasebook.json under "phrases" - add your
own phrases there and run this again.
"""

import json
import time

from translator import SUPPORTED_LANGUAGES, translate
from offline_translator import PHRASEBOOK_PATH

DELAY_S = 1.5          # pause between requests, to stay polite to the free service
KEEP_AS_IS = {"SOS"}   # universal signals: never translated


def main():
    with open(PHRASEBOOK_PATH, encoding="utf-8") as f:
        book = json.load(f)
    translations = book.setdefault("translations", {})
    phrases = book.get("phrases", [])

    languages = [name for name, code in SUPPORTED_LANGUAGES.items() if not code.startswith("en")]
    todo = [(lang, ph) for lang in languages for ph in phrases
            if ph not in translations.get(lang, {})]
    print(f"{len(todo)} translations missing across {len(languages)} languages.")

    done = 0
    for lang, phrase in todo:
        code = SUPPORTED_LANGUAGES[lang]
        if phrase in KEEP_AS_IS:
            text, ok, err = phrase, True, None
        else:
            text, ok, err = translate(phrase.lower(), code)
        if not ok:
            print(f"\nStopped at {lang} / {phrase}: {err}")
            print("Progress is saved. Run this script again later to continue.")
            break
        translations.setdefault(lang, {})[phrase] = {"text": text, "source": "machine", "verified": False}
        with open(PHRASEBOOK_PATH, "w", encoding="utf-8") as f:
            json.dump(book, f, ensure_ascii=False, indent=2)
        done += 1
        print(f"[{done}/{len(todo)}] {lang:10s} {phrase:15s} -> {text}")
        if phrase not in KEEP_AS_IS:
            time.sleep(DELAY_S)

    print(f"\nAdded {done} translation(s). Remember: they are unverified until a native speaker checks them.")


if __name__ == "__main__":
    main()
