"""
alternatives.py

Suggests ALTERNATIVE READINGS of a decoded message, based on which
blinks were most likely misread. Nothing is changed automatically:
the analyst decides.

WHY
---
The most common single error is a dot/dash mix-up: a dash held a bit
too briefly reads as a dot (this turned "SOS" into "SGS" in testing),
or a long dot reads as a dash. Blinks whose duration was close to the
dot/dash cutoff (0.3 s) are the likeliest culprits.

HOW
---
1. For every blink, measure its "doubt": how close its duration was to
   the cutoff, relative to the cutoff.
       doubt margin = |duration - 0.3 s| / 0.3 s   (small = doubtful)
2. Flip each blink (dot <-> dash) on its own, and also flip PAIRS of
   the four most doubtful blinks (two mistakes in one message).
3. Blinks flagged LOW confidence because face tracking was lost nearby
   are also tried REMOVED (they may not be real blinks).
4. Re-decode each variant with the same segmentation as the main
   decoder. Keep only variants whose text actually differs.
5. Rank by doubt: the reading that needs the least surprising change
   comes first. No invented percentages: the ranking is just "how
   close were the changed blinks to the cutoff".
6. As a HINT only, note when a reading contains a known emergency
   phrase (from phrasebook.json) that the original reading didn't.
   A known phrase is never used to rank or auto-correct, because real
   messages may contain names, codes or numbers that are not in any
   list, and forcing them towards familiar words would be dangerous.
"""

from itertools import combinations

import segmentation
from pipeline import SHORT_BLINK_MAX
from offline_translator import load_phrasebook

PAIR_POOL = 4        # consider pairs among the N most doubtful blinks
MAX_RESULTS = 5


def _duration(b):
    return b.get("duration", b["t_end"] - b["t_start"])


def doubt_margin(blink):
    """0.0 = exactly on the dot/dash cutoff (maximally doubtful)."""
    return abs(_duration(blink) - SHORT_BLINK_MAX) / SHORT_BLINK_MAX


def _flip(sym):
    return "-" if sym == "." else "."


def _known_phrases():
    return [p for p in load_phrasebook().get("phrases", []) if p]


def _phrases_in(text):
    padded = f" {text} "
    return {p for p in _known_phrases() if f" {p} " in padded}


def reading_alternatives(blinks, mode="adaptive", max_results=MAX_RESULTS):
    """
    blinks: list of dicts with t_start, t_end, symbol, and optionally
            duration and low_tracking (True if tracking was lost nearby).
    Returns (base_text, alternatives) where each alternative is:
        {"text", "changes": [(index, "flip"|"remove", old, new)],
         "score" (lower = more likely), "explanation", "known_phrases"}
    """
    base_text = segmentation.decode(blinks, mode)["text"]
    base_phrases = _phrases_in(base_text)

    def variant(changes):
        v = [dict(b) for b in blinks]
        removed = set()
        for idx, kind, _old, new in changes:
            if kind == "flip":
                v[idx]["symbol"] = new
            else:
                removed.add(idx)
        v = [b for i, b in enumerate(v) if i not in removed]
        return segmentation.decode(v, mode)["text"] if v else ""

    candidates = []

    def consider(changes, score, why):
        text = variant(changes)
        if text and text != base_text:
            candidates.append({
                "text": text, "changes": changes, "score": score, "explanation": why,
                "known_phrases": sorted(_phrases_in(text) - base_phrases),
            })

    # single flips
    for i, b in enumerate(blinks):
        m = doubt_margin(b)
        new = _flip(b["symbol"])
        consider([(i, "flip", b["symbol"], new)], m,
                 f"blink at {b['t_start']:.2f}s lasted {_duration(b):.2f}s "
                 f"(cutoff {SHORT_BLINK_MAX:.2f}s): read as '{b['symbol']}', try '{new}'")

    # pairs of the most doubtful blinks
    pool = sorted(range(len(blinks)), key=lambda i: doubt_margin(blinks[i]))[:PAIR_POOL]
    for i, j in combinations(sorted(pool), 2):
        bi, bj = blinks[i], blinks[j]
        consider([(i, "flip", bi["symbol"], _flip(bi["symbol"])),
                  (j, "flip", bj["symbol"], _flip(bj["symbol"]))],
                 doubt_margin(bi) + doubt_margin(bj),
                 f"two blinks flipped, at {bi['t_start']:.2f}s and {bj['t_start']:.2f}s")

    # removal of blinks next to a tracking gap
    for i, b in enumerate(blinks):
        if b.get("low_tracking"):
            consider([(i, "remove", b["symbol"], None)], 0.5,
                     f"blink at {b['t_start']:.2f}s was next to a face-tracking gap: try removing it")

    # keep the best-scoring way of reaching each distinct text
    best = {}
    for c in candidates:
        if c["text"] not in best or c["score"] < best[c["text"]]["score"]:
            best[c["text"]] = c
    ranked = sorted(best.values(), key=lambda c: c["score"])
    return base_text, ranked[:max_results]
