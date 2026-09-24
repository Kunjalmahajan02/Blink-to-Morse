# Blink-to-Morse Translator

A proof-of-concept computer vision system for **silent communication
through eye blinks**. A person blinks in Morse code (short blink = dot,
long blink = dash), and the system decodes the message into text from
either a **live webcam** or a **pre-recorded video**, then translates
and speaks it in 14 Indian languages.

## Motivation

Some situations leave a person with no way to speak or use their hands,
but they can still control their eyes: patients with severe motor
impairments, or a captured person filmed on video.

This has real precedent. In 1966, US Navy pilot Jeremiah Denton, held as
a prisoner of war, was forced to appear in a propaganda video. While
answering his captors' questions, he blinked the word **T-O-R-T-U-R-E**
in Morse code, and US intelligence decoded it after the footage was
broadcast.

This project explores whether a hidden blink message in such footage can
be extracted **systematically and verifiably**, rather than relying on
a human happening to notice it.

> This is an academic prototype, not an operational system. Its
> accuracy and limitations are measured and reported below.

## System overview

```
      LIVE WEBCAM            RECORDED VIDEO
           \                      /
            \                    /
         Face landmark detection (MediaPipe Face Mesh, 468 points)
                        |
          Eye Aspect Ratio (EAR) + head pose ratios
                        |
        Blink detection -> dot / dash classification
        Head nod -> space        Head turn -> backspace
                        |
         Letter / word timing (pauses between blinks)
                        |
                Morse-to-text decoding
                        |
      Confidence flags + analyst review (Forensic Mode)
                        |
     Translation (14 Indian languages) + speech + log
```

## How it works

### 1. Face landmarks
MediaPipe Face Mesh locates 468 points on the face in every frame,
including six points around each eye.

### 2. Eye Aspect Ratio (EAR)
```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```
The vertical eyelid gap divided by the eye's width. It stays roughly
constant while the eye is open and drops sharply when it closes.

### 3. Auto-calibration
For the first 5 seconds the system records the person's normal open-eye
EAR and neutral head position. The blink threshold is set to 75% of that
personal baseline, so it adapts to different faces, cameras and lighting.

### 4. Dot / dash classification
- Eye closed under 0.3 s -> **dot**
- Eye closed 0.3 s or longer -> **dash**

### 5. Letter and word segmentation
- Pause over 1.2 s after a blink -> letter complete
- Pause over 3.0 s -> word complete (space)

### 6. Head gestures
Nods and turns are detected by comparing **ratios of distances** between
facial landmarks (nose vs. eye line, nose vs. cheeks). Ratios work at
any distance from the camera. A nod inserts a space; a turn deletes the
last character.

### 7. Video timing
For recorded video, time is measured with the video's own clock
(frame number / frames per second), not the computer's clock. Blink
durations therefore stay correct however fast the computer processes
the file.

## Features

| Feature | Description |
|---|---|
| Real-time mode | Live webcam decoding |
| Video upload mode | Same engine, run on a recorded video file |
| Forensic Analysis Mode | Human-in-the-loop review of recorded footage (below) |
| Confidence flags | Uncertain blinks are flagged instead of silently trusted |
| Multilingual output | Translation into 14 Indian languages + others |
| Speech output | Offline English speech; online speech for Indian languages |
| Session log | Timestamped record of every decoded message |
| Evaluation tool | Measures accuracy on labelled test videos |

## Forensic Analysis Mode

Automatic detection can fail on poor footage: low light, compression,
odd angles, or a tired subject blinking unevenly. When one wrong letter
can change a message's meaning, the output should not be trusted
blindly. Forensic Mode lets an analyst verify it:

- **EAR graph** of the entire video, showing every eye closure as a dip
- **Adjustable threshold** with live feedback on which blinks are detected
- **Frame-by-frame scrubbing** with a video preview
- **Manual blink marking** for blinks the system missed
- **Review table** to exclude false blinks or correct dot/dash labels
- **Confidence flags**: a blink is marked LOW confidence if its duration
  sits close to the dot/dash cutoff, or if face tracking was lost on a
  nearby frame
- **Letter timeline**: after decoding, the graph shows which blinks
  formed which letter, so the result can be audited

Colour key on the graph: green = automatic, confident; orange = low
confidence; blue = manually marked by the analyst.

## Multilingual output

Standard Morse code only covers Latin letters and digits, so messages are
always decoded to Latin text first and then translated.

Supported Indian languages: Hindi, Tamil, Telugu, Marathi, Bengali,
Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu, Nepali, Assamese,
Sindhi.

Translation uses the MyMemory service (via `deep-translator`); Indian
language speech uses Google TTS (`gTTS`). **Both need an internet
connection.** Detection, decoding and English speech work fully offline.
Translation is triggered only when the user changes the language or
presses Speak, to stay within the free service limits.

## Evaluation

### Why
A demo shows that the system *can* work. An evaluation shows *how often*
it works and *where it fails*.

### Related work
Bhatt (2025), *"Blink-to-Code: Real-Time Morse Code Communication via Eye
Blink Detection and Classification"* (arXiv:2508.09344), used a similar
EAR-based method and reported **62% decoding accuracy across 5
participants** in a well-lit, controlled setting. This project adds
recorded-video decoding, calibration, confidence flags and human review,
and tests under harder conditions.

### Protocol
- Participants: _N_ (fill in)
- Messages: SOS, HELP, SAFE
- Conditions: good light, low light, glasses, far from camera (~1.5 m),
  and video re-compressed by sending it through WhatsApp
- Each video starts with 5 seconds of eyes open (calibration)

### Running the evaluation
```bash
python evaluate.py eval_videos     # first run creates labels.csv
# fill in 'expected' and 'condition' for each video in labels.csv
python evaluate.py eval_videos     # second run produces results
```
Metrics: exact message match rate, character accuracy (edit distance),
letter accuracy (ignoring spaces), expected vs. detected blinks, and the
percentage of frames where no face was found.

### Results
_To be filled in from `eval_videos/summary.csv`._

| Condition | Videos | Exact match | Char accuracy | Blinks expected / detected |
|---|---|---|---|---|
| good_light | | | | |
| low_light | | | | |
| glasses | | | | |
| far | | | | |
| whatsapp_compressed | | | | |
| **Overall** | | | | |

## Limitations

- Calibration assumes the first 5 seconds show the subject with eyes
  open; real intercepted footage may not provide this.
- Thresholds (0.3 s dot/dash, 1.2 s letter, 3.0 s word) are fixed; a
  stressed or injured person may not keep consistent timing.
- Natural involuntary blinks can be mistaken for dots.
- Face tracking degrades with low resolution, poor light, or the face
  turned away.
- Translation and Indian-language speech require internet access.

## Future work

- Adaptive timing that learns each person's blink rhythm
- Distinguishing intentional from involuntary blinks
- Video integrity checks (detecting edited or tampered footage)
- Running on low-power edge hardware (e.g. Raspberry Pi)

## Project structure

```
Blink-to-Morse/
├── gui.py              # Desktop app (PyQt6): live, video upload, forensic mode
├── forensic_mode.py    # Forensic Analysis window: graph, review, confidence
├── evaluate.py         # Batch accuracy evaluation on labelled test videos
├── pipeline.py         # Core detection engine shared by every mode
├── blink_detector.py   # Eye Aspect Ratio calculation
├── head_gesture.py     # Nod / turn detection via landmark ratios
├── calibration.py      # Calibration for the command-line version
├── morse_translator.py # Morse code table and decoding
├── translator.py       # Multilingual translation
├── speech.py           # Text-to-speech (offline English, online Indian languages)
├── logger.py           # Timestamped session log
├── main.py             # Simple command-line version (no GUI)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python gui.py
```

Notes:
- `mediapipe` is pinned to **0.10.21**. Versions from 0.10.30 onward
  removed the Face Mesh "Solutions" API this project uses.
- On Windows, `speech.py` initialises COM on the speaking thread; without
  this, offline speech runs silently.

## References

- Bhatt, A. (2025). *Blink-to-Code: Real-Time Morse Code Communication
  via Eye Blink Detection and Classification.* arXiv:2508.09344.
- Soukupova, T. & Cech, J. (2016). *Real-Time Eye Blink Detection using
  Facial Landmarks.* (Origin of the Eye Aspect Ratio.)
- Google MediaPipe Face Mesh.
- Jeremiah Denton, 1966 POW broadcast (historical precedent).
