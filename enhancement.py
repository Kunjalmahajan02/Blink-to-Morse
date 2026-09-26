"""
enhancement.py

Low-light / night mode preprocessing.

WHY
---
Blink detection depends on MediaPipe first FINDING the face and eye
landmarks. In dark footage the face blends into the background, so
MediaPipe loses the face (or places landmarks badly) and blinks are
missed. Captives are rarely filmed in good light, so this matters for
the project's main scenario.

The Eye Aspect Ratio itself is pure geometry (distances between
points), so we don't need to change EAR at all. We only need to make
the face easier to FIND. That is what this module does, before any
detection happens.

THE PIPELINE
------------
    frame
      -> brightness assessment   (how dark is it?)
      -> if dark: gamma correction  (lift dark tones)
                  + CLAHE           (boost local contrast)
      -> face detection / eye tracking / blink detection (unchanged)

1. BRIGHTNESS ASSESSMENT
   Mean of the L (lightness) channel in LAB colour space, on a 0-255
   scale. LAB separates lightness from colour, so this measures "how
   bright" without being fooled by colourful clothing or walls.
       >= 90   normal
       50-90   low light
       <  50   very low light
   (Thresholds are reasonable starting values, not tuned by data. The
   evaluation tool records the brightness of every video so they can
   be checked against real results.)

2. GAMMA CORRECTION
   Brightens dark tones much more than bright ones (a curve, not a
   flat "+50 brightness"), so shadows open up without washing out
   highlights. The darker the frame, the stronger the curve.

3. CLAHE (Contrast Limited Adaptive Histogram Equalisation)
   Instead of stretching contrast over the whole image at once, it
   works on small tiles (8x8 grid), so a dim face next to a bright
   lamp both get enhanced properly. "Contrast limited" caps how much
   any tile can be boosted, which stops camera noise in dark areas
   being amplified into fake detail. Applied only to the L channel so
   colours are not distorted.

MODES
-----
    "off"  -> never enhance (original behaviour)
    "auto" -> enhance only frames darker than the normal threshold (default)
    "on"   -> always enhance (for experiments)

KNOWN LIMITATION
----------------
Enhancement cannot create information the camera never captured. In
near-total darkness the frame is mostly sensor noise, and brightening
it mostly brightens noise. That is exactly what evaluation should
measure rather than assume.
"""

import cv2
import numpy as np

NORMAL_BRIGHTNESS = 90      # at or above this: no enhancement in "auto"
VERY_LOW_BRIGHTNESS = 50    # below this: labelled "very low light"
TARGET_BRIGHTNESS = 110     # gamma aims to lift the mean towards this

CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID = (8, 8)

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)


def assess_brightness(frame_bgr):
    """Returns (brightness 0-255, level string)."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    brightness = float(np.mean(lab[:, :, 0]))
    if brightness >= NORMAL_BRIGHTNESS:
        level = "normal"
    elif brightness >= VERY_LOW_BRIGHTNESS:
        level = "low"
    else:
        level = "very_low"
    return brightness, level


def _gamma_for(brightness):
    """Choose a gamma that would move the mean brightness towards
    TARGET_BRIGHTNESS. gamma < 1 brightens. Clamped to a safe range so
    very dark frames are not blown out into pure noise."""
    b = max(brightness, 1.0) / 255.0
    target = TARGET_BRIGHTNESS / 255.0
    gamma = np.log(target) / np.log(b)
    return float(np.clip(gamma, 0.35, 1.0))


def _apply_gamma(frame_bgr, gamma):
    table = ((np.arange(256) / 255.0) ** gamma * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.LUT(frame_bgr, table)


def enhance(frame_bgr, mode="auto"):
    """
    Returns (frame_to_use, info).

    frame_to_use: the enhanced frame if enhancement was applied,
                  otherwise the original frame (same object).
    info: {"brightness": float, "level": str, "applied": bool, "gamma": float|None}
    """
    brightness, level = assess_brightness(frame_bgr)
    info = {"brightness": brightness, "level": level, "applied": False, "gamma": None}

    if mode == "off" or (mode == "auto" and level == "normal"):
        return frame_bgr, info

    gamma = _gamma_for(brightness)
    out = _apply_gamma(frame_bgr, gamma) if gamma < 0.99 else frame_bgr

    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    info.update(applied=True, gamma=gamma)
    return out, info
