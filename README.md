# Blink-to-Morse Translator

A computer vision system that lets a person "type" by blinking — short
blinks become Morse code dots, long blinks become dashes, and the system
decodes the pattern into readable text in real time using just a webcam.

## How it works

```
Camera
  |
  v
Face Landmark Detection (MediaPipe Face Mesh)
  |
  v
Eye Aspect Ratio (EAR) calculation
  |
  v
Short blink -> dot (.)   |   Long blink -> dash (-)
  |
  v
Letter/word timing (pauses between blinks)
  |
  v
Morse-to-text decoding
  |
  v
Live text output on screen
```

### 1. Face landmark detection
[MediaPipe Face Mesh](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
locates 468 points on the face in every video frame, including precise
points around each eye.

### 2. Eye Aspect Ratio (EAR)
For each eye, we measure how "open" it is using 6 landmark points:

```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```

This compares the vertical gap between the eyelids to the eye's
horizontal width. EAR is high when the eye is open and drops sharply
when it closes.

### 3. Blink classification
We time how long EAR stays below a threshold (eye closed):
- Under 0.3 seconds → **dot**
- 0.3 seconds or more → **dash**

### 4. Letter/word segmentation
The system also watches the *pauses* between blinks:
- A short pause means you're still spelling the same letter
- A ~1.2 second pause means the letter is complete → decode it
- A ~3 second pause also inserts a space (new word)

### 5. Morse-to-text decoding
The completed dot/dash pattern (e.g. `...`) is looked up in a standard
Morse code table to get the letter (e.g. `S`).

## Project structure

```
blink-morse-translator/
├── main.py               # Entry point: camera loop, timing logic, UI
├── blink_detector.py      # EAR calculation and eye-open/closed logic
├── morse_translator.py    # Morse code lookup table and decoding
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

> **Note on the MediaPipe version:** this project pins
> `mediapipe==0.10.21`. Google removed the legacy "Solutions" API
> (which includes Face Mesh) starting from MediaPipe 0.10.30, in
> favor of the newer Tasks API. Pinning to 0.10.21 keeps the simpler,
> well-documented Solutions API working. A future improvement would
> be migrating to the MediaPipe Tasks API for long-term support.

## Usage

```bash
python main.py
```

- Blink normally for a **dot**, hold your eyes closed a beat longer for a **dash**
- Pause after a letter to let it lock in
- Press `c` to clear the buffer/text
- Press `q` to quit

## Planned improvements

- [ ] Head nods/turns as additional input gestures (e.g. nod = space, turn = backspace)
- [ ] Text-to-speech output for the decoded sentence
- [ ] Session logging (timestamped text file of everything typed)
- [ ] Auto-calibration of the EAR threshold per-user at startup
- [ ] Polished on-screen UI (progress bar for letter-pause timing)
