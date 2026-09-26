"""
codebook.py

Lets a single continuous group of blinks (no pause in between, exactly
like blinking one letter) stand for a WHOLE PHRASE instead of one
letter. This makes urgent, common messages much faster to send than
spelling them out.

WHY THIS CAN NEVER COLLIDE WITH REAL SPELLING
Standard Morse letters and digits use patterns of 1 to 5 symbols
(checked against morse_translator.MORSE_CODE: max length 5, for the
digits). Every codebook entry below uses 6 OR MORE symbols. A group of
blinks with no internal pause is decoded as ONE unit (see
segmentation.py), so a 6+ symbol group can never be mistaken for a
sequence of real letters either -- it's checked as a whole against this
codebook before anything else is tried.

If a 6+ symbol group is NOT in the codebook, it falls through to normal
pattern-repair (segmentation.py's _repair), which reads it as ordinary
letters. So an accidental long run of blinks is still spelled out
letter by letter, not silently dropped.

CUSTOMISING
This table is deliberately short and easy to edit. Add or change
entries as needed; each pattern must be a string of only '.' and '-',
6 characters or longer.
"""

MIN_CODEBOOK_LENGTH = 6

CODEBOOK = {
    "......": "NEED BACKUP",
    "------": "ENEMY SIGHTED",
    ".-.-.-": "ALL CLEAR",
    "-.-.-.": "UNDER FIRE",
    "..--..": "MISSION COMPLETE",
    "--..--": "REQUEST EXTRACTION",
}

for _pattern in CODEBOOK:
    assert set(_pattern) <= {".", "-"}, f"Bad codebook pattern: {_pattern!r}"
    assert len(_pattern) >= MIN_CODEBOOK_LENGTH, (
        f"Codebook pattern {_pattern!r} is only {len(_pattern)} symbols long; "
        f"must be at least {MIN_CODEBOOK_LENGTH} to avoid colliding with real letters."
    )

PATTERN_BY_PHRASE = {v: k for k, v in CODEBOOK.items()}


def is_codebook_pattern(pattern):
    return pattern in CODEBOOK


def lookup(pattern):
    """Returns the phrase for this exact pattern, or None."""
    return CODEBOOK.get(pattern)


def decode_group(pattern, fallback_decode_letter):
    """
    Decodes one blink group (no internal pause): a codebook phrase if
    the pattern matches exactly, otherwise whatever fallback_decode_letter
    (normally morse_translator.decode_letter) would return.
    """
    phrase = lookup(pattern)
    return phrase if phrase is not None else fallback_decode_letter(pattern)
