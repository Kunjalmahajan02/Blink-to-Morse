"""
segmentation.py

Decides where one LETTER ends and the next begins (and where WORDS
end), given a list of detected blinks with their timings.

WHY THIS FILE EXISTS
--------------------
Evaluation showed a real problem: a user blinked S-O-S correctly, but
paused only ~0.5-0.8 s between letters. The original rule ("a letter
ends after a pause longer than 1.2 s") merged everything into one
invalid pattern "...---...", decoded as "?".

People don't keep fixed timings, especially under stress, but they do
tend to keep a RHYTHM. Standard Morse code is defined by ratios, not
seconds:
    gap inside a letter   = 1 unit
    gap between letters   = 3 units
    gap between words     = 7 units

TWO MODES
---------
"fixed"    -> the original rule: letter gap 1.2 s, word gap 3.0 s.
"adaptive" -> 1. Estimate this person's "unit" from their own shortest
                 typical gaps (25th percentile of all gaps).
              2. Letter boundary = gap longer than 2.5 units (but at
                 least 0.6 s, and never stricter than fixed mode).
                 Word boundary   = gap longer than 5 units, and at
                 least 2.5x the letter gap (never stricter than fixed).
              3. PATTERN REPAIR: if a group of blinks still isn't a
                 valid Morse letter, try every way of cutting it into
                 valid letters (A-Z). Choose the reading needing the
                 FEWEST cuts; if there's a tie, choose the one whose
                 cuts fall on the LONGEST pauses. Letters produced this
                 way are marked "reconstructed", so the analyst knows
                 they were inferred, not directly observed.

INPUT FORMAT
------------
A list of blink dicts, in time order, each with:
    "t_start": seconds when the eye closed
    "t_end":   seconds when the eye reopened
    "symbol":  "." or "-"
Any extra keys are kept untouched (so callers can attach their own data).
"""

from morse_translator import MORSE_CODE, decode_letter

FIXED_LETTER_GAP = 1.2
FIXED_WORD_GAP = 3.0

# Morse letter gaps are 3 units and symbol gaps 1 unit. We split letters
# at 2.5 units -- deliberately a little cautious, because testing showed
# that SPLITTING TOO MUCH cannot be undone (the pieces still look like
# valid letters), while MERGING TOO MUCH can be repaired by the
# validity check below (the merged pattern is not a real letter).
LETTER_GAP_UNITS = 2.5
WORD_GAP_UNITS = 5.0
# Never split on pauses shorter than this: gaps between the dashes of a
# single letter were measured at up to ~0.45 s in real test videos.
MIN_LETTER_GAP = 0.6
WORD_GAP_OVER_LETTER = 2.5   # a word gap must be clearly longer than a letter gap
MIN_GAPS_FOR_ADAPTIVE = 3

LETTER_CODES = {code for code, ch in MORSE_CODE.items() if ch.isalpha()}


def _gaps(blinks):
    return [blinks[i + 1]["t_start"] - blinks[i]["t_end"] for i in range(len(blinks) - 1)]


def thresholds(blinks, mode):
    """Returns (letter_gap, word_gap, unit). unit is None in fixed mode
    or when there are too few blinks to estimate a rhythm."""
    if mode != "adaptive":
        return FIXED_LETTER_GAP, FIXED_WORD_GAP, None

    gaps = sorted(g for g in _gaps(blinks) if g > 0)
    if len(gaps) < MIN_GAPS_FOR_ADAPTIVE:
        return FIXED_LETTER_GAP, FIXED_WORD_GAP, None

    unit = gaps[len(gaps) // 4]   # 25th percentile: typical short gap
    letter_gap = min(FIXED_LETTER_GAP, max(MIN_LETTER_GAP, LETTER_GAP_UNITS * unit))
    word_gap = min(FIXED_WORD_GAP, max(WORD_GAP_OVER_LETTER * letter_gap, WORD_GAP_UNITS * unit))
    return letter_gap, word_gap, unit


def _repair(group):
    """Split an invalid blink group into valid letters.
    Dynamic programming over cut positions:
      minimise number of letters, then maximise total length of the
      pauses we cut at. Always succeeds, since any pattern can at worst
      be read as single-symbol letters (E = '.', T = '-')."""
    n = len(group)
    inner_gaps = _gaps(group)
    # best[i] = (num_letters, -sum_of_cut_gaps, list_of_piece_end_indices) for group[:i]
    best = [None] * (n + 1)
    best[0] = (0, 0.0, [])
    for end in range(1, n + 1):
        for start in range(max(0, end - 4), end):   # letters are 1-4 symbols
            if best[start] is None:
                continue
            code = "".join(b["symbol"] for b in group[start:end])
            if code not in LETTER_CODES:
                continue
            cut_gap = inner_gaps[start - 1] if start > 0 else 0.0
            cand = (best[start][0] + 1, best[start][1] - cut_gap, best[start][2] + [end])
            if best[end] is None or cand[:2] < best[end][:2]:
                best[end] = cand
    pieces, prev = [], 0
    for end in best[n][2]:
        pieces.append(group[prev:end])
        prev = end
    return pieces


def decode(blinks, mode="fixed"):
    """
    Returns a dict:
        text         decoded message, e.g. "SOS HELP"
        letters      list of {"blinks", "letter", "reconstructed", "space_after"}
        letter_gap   the letter-boundary threshold used (seconds)
        word_gap     the word-boundary threshold used (seconds)
        unit         estimated rhythm unit (seconds) or None
        reconstructed_count
    """
    result = {"text": "", "letters": [], "letter_gap": None, "word_gap": None,
              "unit": None, "reconstructed_count": 0}
    if not blinks:
        return result

    letter_gap, word_gap, unit = thresholds(blinks, mode)
    result.update(letter_gap=letter_gap, word_gap=word_gap, unit=unit)

    # 1. Group blinks into letters using the gap thresholds
    groups = [[blinks[0]]]
    word_break_after = []   # parallel to groups: True if a word gap follows
    for prev, cur in zip(blinks, blinks[1:]):
        gap = cur["t_start"] - prev["t_end"]
        if gap > letter_gap:
            word_break_after.append(gap > word_gap)
            groups.append([cur])
        else:
            groups[-1].append(cur)
    word_break_after.append(False)

    # 2. Decode each group, repairing invalid ones in adaptive mode
    for group, space_after in zip(groups, word_break_after):
        code = "".join(b["symbol"] for b in group)
        if mode == "adaptive" and code not in MORSE_CODE:
            pieces = _repair(group)
            for i, piece in enumerate(pieces):
                piece_code = "".join(b["symbol"] for b in piece)
                result["letters"].append({
                    "blinks": piece,
                    "letter": decode_letter(piece_code),
                    "reconstructed": True,
                    "space_after": space_after and i == len(pieces) - 1,
                })
            result["reconstructed_count"] += len(pieces)
        else:
            result["letters"].append({
                "blinks": group,
                "letter": decode_letter(code),
                "reconstructed": False,
                "space_after": space_after,
            })

    result["text"] = "".join(
        entry["letter"] + (" " if entry["space_after"] else "") for entry in result["letters"]
    ).strip()
    return result
