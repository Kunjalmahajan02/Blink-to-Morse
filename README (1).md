# Blink-to-Morse Translator

A computer vision system that lets a person communicate hands-free by
blinking and moving their head — short blinks become Morse code dots,
long blinks become dashes, head nods insert spaces, and head turns
backspace — all decoded into readable text (and spoken aloud) in real
time, using nothing but a camera.

Includes a desktop application with two modes: **live webcam** and
**pre-recorded video file** analysis, both powered by the same
detection engine.

## How it works

```
Camera (live) OR Video file
  |
  v
Face Landmark Detection (MediaPipe Face Mesh)
  |
  v
Eye Aspect Ratio (EAR) calculation  ---->  Head pose ratios (pitch/yaw)
  |                                              |
  v                                              v
Short blink -> dot (.)                   Nod -> space
Long blink  -> dash (-)                  Turn -> backspace
  |
  v
Letter/word timing (pauses between blinks)
  |
  v
Morse-to-text decoding
  |
  v
Live text display + Text-to-Speech + Session logging
```

### 1. Face landmark detection
[MediaPipe Face Mesh](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
locates 468 points on the face in every frame, including precise
points around the eyes, nose, and cheeks.

### 2. Eye Aspect Ratio (EAR) — blink detection
For each eye, we measure how "open" it is using 6 landmark points:

```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```

EAR is high when the eye is open and drops sharply when it closes. We
time how long it stays low:
- Under 0.3 seconds → **dot**
- 0.3 seconds or more → **dash**

### 3. Auto-calibration
Instead of a fixed threshold for everyone, the system watches the
user's face for 5 seconds at startup — recording their normal
open-eye EAR and neutral head position — and sets personalized
thresholds from that baseline. This makes detection far more reliable
across different people, lighting conditions, and camera setups.

### 4. Head gesture detection (nod / turn)
Rather than full 3D head-pose estimation (which needs an accurate
camera calibration to work well), we compare **ratios of distances**
between stable facial landmarks:
- **Nod**: how far below eye-level the nose sits, normalized by the
  distance between the eyes.
- **Turn**: how the nose's horizontal position compares to the left
  vs. right cheek.

Both are scale-invariant (work regardless of distance from the
camera) and respond reliably to real head movement.

### 5. Letter/word segmentation
The system watches the *pauses* between blinks:
- A short pause means you're still spelling the same letter
- A ~1.2 second pause means the letter is complete → decode it
- A ~3 second pause also inserts a space (new word)

### 6. Morse-to-text decoding
The completed dot/dash pattern (e.g. `...`) is looked up in a
standard Morse code table to get the letter (e.g. `S`).

### 7. Output
- **On-screen**: live buffer, decoded text, and a progress bar showing
  how close the current letter is to being committed.
- **Speech**: each completed word (and the full sentence, on demand)
  is spoken aloud via offline text-to-speech.
- **Logging**: every completed word is saved with a timestamp to
  `session_log.txt`.

## Features

- Real-time webcam mode
- Video file upload mode — run the exact same detection engine on
  pre-recorded footage instead of a live camera
- Auto-calibrating blink threshold (per-user, per-session)
- Head nod (space) and head turn (backspace) as extra input gestures
- Offline text-to-speech output
- Timestamped session logging
- Desktop GUI (PyQt6) with a live video panel, decoded-text readout,
  and on-screen HUD overlay

## Project structure

```
blink-morse-translator/
├── gui.py                 # Desktop application (PyQt6) -- the main way to run this project
├── main.py                # Simpler command-line version of the same pipeline (no GUI)
├── pipeline.py             # Core detection engine, shared by both gui.py and main.py
├── blink_detector.py       # Eye Aspect Ratio (EAR) calculation
├── head_gesture.py         # Nod/turn detection via landmark distance ratios
├── calibration.py          # Auto-calibration used by main.py's command-line flow
├── morse_translator.py     # Morse code lookup table and decoding
├── speech.py                # Offline text-to-speech (with Windows COM fix)
├── logger.py                # Timestamped session logging
├── requirements.txt
└── README.md
```

`pipeline.py` contains the actual detection logic exactly once —
both the GUI and the simpler command-line script call into it, so
there's a single, testable, explainable engine at the core of the
project rather than the same logic duplicated in two places.

## Setup

```bash
pip install -r requirements.txt
```

> **Note on the MediaPipe version:** this project pins
> `mediapipe==0.10.21`. Google removed the legacy "Solutions" API
> (which includes Face Mesh) starting from MediaPipe 0.10.30, in
> favor of the newer Tasks API. Pinning to 0.10.21 keeps the simpler,
> well-documented Solutions API working.
>
> **Note on text-to-speech (Windows):** `pyttsx3` uses Windows' SAPI5
> engine via `pywin32`, which requires COM to be initialized on the
> thread that speaks. `speech.py` handles this explicitly — if audio
> ever goes silent again, that's the first thing to check.

## Usage

### Desktop app (recommended)

```bash
python gui.py
```

- Click **Start Real-Time Scan** to use your webcam, or **Upload
  Intercept File** to analyze a video file instead.
- For video files: the subject should look at the camera with eyes
  open for the first ~5 seconds (used for calibration) before
  starting to blink out a message.
- **Speak Transmission** reads the decoded text aloud on demand.
- **Wipe Buffer** clears the current text (saving it to the log
  first). **Terminate** ends the session.

### Command-line version

```bash
python main.py
```

A simpler, GUI-free version of the same pipeline — useful for walking
through the core logic without GUI code in the way.

- Blink normally for a **dot**, hold your eyes closed a beat longer
  for a **dash**
- Nod to insert a space, turn your head to backspace
- Press `t` to speak the current text, `c` to clear, `q` to quit

## Planned improvements

- [ ] Multi-modal redundancy (e.g. eyebrow raise as a third input
      channel if blinking/head movement is restricted)
- [ ] Adaptive re-calibration if lighting changes mid-session
- [ ] Predefined phrase shortcuts for common messages
- [ ] Encrypted/secure transmission of decoded text to a remote
      endpoint
