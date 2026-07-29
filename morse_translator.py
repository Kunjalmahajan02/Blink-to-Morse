"""
morse_translator.py

Handles everything related to Morse code itself:
- The dot/dash -> letter lookup table
- Decoding a dot/dash string into a letter

Keeping this separate from the camera/blink code means the morse logic
can be tested and explained on its own, without needing a webcam.
"""

# Standard international Morse code table (letters + digits)
MORSE_CODE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
}


def decode_letter(morse_pattern: str) -> str:
    """
    Converts a single dot/dash pattern (e.g. '...') into its letter (e.g. 'S').
    Returns '?' if the pattern doesn't match any known letter.
    """
    return MORSE_CODE.get(morse_pattern, '?')
