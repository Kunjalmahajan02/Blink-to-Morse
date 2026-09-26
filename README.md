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
For the first 3 seconds the system records the person's normal open-eye
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

### 7. Adaptive timing and pattern repair (recorded video)
Early evaluation showed a real failure: a user blinked S-O-S correctly
but paused only ~0.5-0.8 s between letters, so the fixed 1.2 s rule
merged everything into `...---...` and decoded it as `?`.

Morse code is defined by **ratios**, not seconds (symbol gap 1 unit,
letter gap 3 units, word gap 7 units). People under stress rarely keep
exact timings, but they usually keep a rhythm. Adaptive mode:

1. Estimates the person's rhythm unit from their own short gaps.
2. Splits letters at 2.5 units (at least 0.6 s) and words at 5 units,
   never stricter than the fixed rule.
3. **Pattern repair:** if a blink group is still not a valid letter, it
   tries every way of cutting it into valid letters, choosing the reading
   with the fewest cuts, placed at the longest pauses.

Letters produced by repair are marked **reconstructed** (orange, with a
`*` on the Forensic Mode graph), so an analyst can see what was inferred
rather than directly observed.

Design note: splitting too much cannot be undone (the pieces still look
like valid letters), while merging too much can be repaired (the merged
pattern is invalid). The adaptive thresholds are therefore deliberately
cautious and rely on repair for the fine work.

Adaptive timing currently applies to recorded video (Forensic Mode and
`evaluate.py`); the live webcam mode uses fixed timing.

### 8. Low-light (night) mode
Blink detection depends on MediaPipe first finding the face. In dark
footage the face blends into the background and landmarks are lost.
The EAR formula is pure geometry, so it doesn't need changing; the face
just needs to be easier to find. Before detection, every frame goes
through:

```
frame -> brightness assessment -> (if dark) gamma correction + CLAHE -> face detection
```

- **Brightness assessment:** mean lightness (L channel of LAB colour
  space, 0-255). Normal >= 90, low 50-90, very low < 50.
- **Gamma correction:** lifts dark tones much more than bright ones;
  stronger for darker frames.
- **CLAHE** (Contrast Limited Adaptive Histogram Equalisation):
  boosts contrast in small tiles so a dim face is enhanced even next to
  a bright light, with a limit that stops dark-area noise being
  amplified. Applied to lightness only, so colours are not distorted.

Modes: **Auto** (enhance only dark frames, default), **Off**, **Always
on**. It applies to every mode (live, upload, Forensic Mode,
evaluation), costs roughly 8 ms per frame, and the video panel shows the
measured brightness and whether night mode is active.

Limitation: enhancement cannot recover detail the camera never
captured. In near-total darkness it mostly brightens sensor noise,
which is why its effect is measured rather than assumed.

### 9. Robustness to camera angle
The system cannot work from *every* angle: if the eyes are not visible
(side profile, back of the head, eyes covered), there is nothing to
measure. Within the range where the face is visible, four measures make
it much less sensitive to angle and position:

- **Visibility-weighted EAR.** When the head is turned, the far eye looks
  narrower and is partly hidden, so its landmarks are unreliable. Each
  eye is weighted by the square of its visible width: facing the camera
  this is a normal average; as the head turns, the near eye dominates.
- **Rolling baseline.** The open-eye EAR level changes with viewing
  angle (a camera above the face makes eyes look narrower). Instead of
  a threshold fixed once at calibration, the open-eye level is
  re-measured continuously as the 85th percentile of the last 6 seconds
  (a high percentile, so blinks and dashes don't drag it down), and the
  threshold is 75% of that.
- **Re-baselining.** A real Morse blink lasts well under 2 s. If the eyes
  seem closed for longer, the view has almost certainly changed, so that
  "closure" is discarded (no fake dash) and the baseline is re-measured
  immediately from the last second.
- **Automatic orientation.** Phone videos filmed sideways or upside down
  are detected by trying 0/90/180/270 degree rotations on sample frames
  and using the one where the face is found.

Head gestures (nod = space, turn = backspace) are enabled only in live
mode; in recorded footage, natural head movement would delete letters.

In a simulated mid-video camera shift (open-eye level 0.30 -> 0.21, 17
blinks), a fixed threshold detected 7 blinks; the rolling baseline
detected 16, with one lost during re-baselining. Real angled footage
should be evaluated with conditions such as `angle_30`, `angle_45`,
`camera_above`, `camera_below`.

### 10. Video timing
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
| Alternative readings | Ranked suggestions for likely misread blinks, RAW/FINAL kept separate |
| Codebook phrases | One blink group = one whole phrase, can't collide with letters |
| Distress signal | 5 rapid long holds trigger an immediate alert |
| HUD interface | "Night-ops" theme: status lights, EAR signal graph, target-lock overlay, Morse-rain idle screen, Morse reference card |
| File metadata | Recording time / GPS from the video file, shown separately as unverified |
| Multilingual output | 14 Indian languages, offline by default |
| Speech output | Offline English speech; online speech for Indian languages |
| Session log | Timestamped record of every decoded message |
| Low-light mode | Automatic brightness check and night enhancement |
| Angle robustness | Visibility-weighted EAR, rolling baseline, auto-rotation |
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

### Alternative readings
The most common error is a dot/dash mix-up (a dash held too briefly
reads as a dot, e.g. SOS decoded as SGS). After decoding, Forensic Mode
lists up to five **alternative readings**:

1. Each blink's "doubt" is how close its duration was to the 0.3 s
   dot/dash cutoff.
2. The most doubtful blinks are flipped (singly and in pairs), and
   blinks next to a face-tracking gap are tried removed.
3. Each variant is re-decoded, and readings are ranked by how small a
   change they need. No confidence percentages are invented.
4. A reading containing a known emergency phrase (from the phrasebook)
   is marked as a **hint only**. It never ranks or auto-corrects,
   because real messages may contain names or codes not in any list.

Nothing changes until the analyst applies a suggestion. The system keeps
three layers separate, shown in the status line and saved to the log:
**RAW** (detector output, untouched), **FINAL** (after review), and the
list of **corrections** made (flips, exclusions, added blinks).

On the test video that decoded as `SGS S`, the top suggestion was
`SOS S`, pointing at the correct blink (a 0.18 s "dash").

## Codebook phrases

Urgent, common messages can be sent as one blink group instead of being
spelled out letter by letter:

| Pattern (one group, no pause) | Meaning |
|---|---|
| `......` (6 dots) | NEED BACKUP |
| `------` (6 dashes) | ENEMY SIGHTED |
| `.-.-.-` | ALL CLEAR |
| `-.-.-.` | UNDER FIRE |
| `..--..` | MISSION COMPLETE |
| `--..--` | REQUEST EXTRACTION |

**Why it can't be confused with normal spelling:** every standard Morse
letter and digit uses 1 to 5 symbols. Every codebook pattern uses 6 or
more, so a codebook group can never be a real letter. A 6+ symbol group
that isn't in the codebook falls back to normal pattern repair and is
read as letters. The table is in `codebook.py` and easy to customise;
the file checks at start-up that every pattern is 6+ symbols long.

## Distress signal (panic pattern)

**Five long holds in quick succession** (each held for 1 to 2 seconds,
each within 3 seconds of the previous one) trigger an immediate distress
alert: a full-screen red warning, a spoken alert, and a log entry. It
bypasses normal decoding, for a person who is injured or cannot keep up
accurate Morse timing.

Holds must stay under 2 seconds because, for angle robustness, any
closure longer than 2 seconds is treated as a change of camera view and
discarded (see "Robustness to camera angle").

It is designed not to trigger by accident: normal dashes are far
shorter than 1 second, any short blink resets the count, and holds
spread far apart don't count as one deliberate act. Tested cases: five
rapid holds trigger it; four holds plus a short blink do not; a normal
SOS never does; five holds spread 10 seconds apart do not.

## File metadata (unverified)

When a recorded video is loaded, the app reads the video file's own
embedded **recording time** and **GPS location**, if present, and shows
them in a separate, clearly labelled panel. They are never mixed into
the decoded message, because they come from a different, less
trustworthy source:

- most videos that were re-encoded, forwarded or uploaded to a
  messaging app have this information stripped;
- a device clock can be wrong, and metadata can be deliberately faked;
- the software does not verify it against anything.

`video_metadata.py` reads MP4/MOV files directly (the standard `mvhd`
header for recording time and the optional `loci` box for GPS) using
only Python's built-in `struct` module, with no external tools. Values
are sanity-checked (plausible date, valid coordinates, and 0,0 treated
as missing) and anything doubtful is reported as "not available" rather
than guessed. AVI/MKV files are not parsed. Tested against real files
with and without metadata, plus missing, unsupported and corrupted
files. The metadata is also written to the log, labelled "unverified".

## Multilingual output and offline mode

Standard Morse code only covers Latin letters and digits, so messages are
always decoded to Latin text first and then rendered in the analyst's
language (14 Indian languages supported).

### Offline mode (default)
Detection and decoding always run entirely on the device. For output,
**OFFLINE MODE is on by default** and the app makes no network calls:

- **Phrasebook** (`phrasebook.json`): common emergency phrases (HELP,
  I AM SAFE, CAPTURED, NEED HELP, ...) with translations, matched
  longest-phrase-first so "I AM SAFE" is translated as one phrase.
  Every entry records whether a native speaker has verified it;
  unverified output is shown in orange.
- **Transliteration** for everything else: names, places and codewords
  are written in the target script (offline `indic-transliteration`
  library), not translated. This is approximate phonetics and is
  labelled as such. Urdu and Sindhi (Arabic script) are not supported
  by the library, so those words stay in Latin letters.
- **Offline speech**: Windows built-in voices (Hindi and some others can
  be added in Windows Settings), or eSpeak NG if installed. If no
  offline voice exists for a language, the app says so and speaks the
  original English instead of going online.

Why this matters: in the motivating scenario, sending a decoded message
to a third-party translation server would leak it. Offline mode means a
decoded message never leaves the computer.

### Online mode (optional)
Unticking OFFLINE MODE allows full-sentence translation via the MyMemory
service and higher-quality speech via Google TTS. The header then shows
"ONLINE TRANSLATION ENABLED" as a warning.

### Building the phrasebook
Hindi is pre-filled by hand. Other languages are filled once, while
online, with:
```bash
python build_phrasebook.py
```
It skips existing entries, saves after every translation, and can be
re-run if it stops at the free service's daily limit. Machine
translations are saved as unverified until checked by a native speaker.

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
- Each video starts with 3-5 seconds of eyes open (calibration)

### Running the evaluation
```bash
python evaluate.py eval_videos                  # first run creates labels.csv
# fill in 'expected' and 'condition' for each video in labels.csv
python evaluate.py eval_videos --enhance off    # without night enhancement
python evaluate.py eval_videos --enhance auto   # with night enhancement (default)
```
Results are saved as `results_enhance-<mode>.csv` and
`summary_enhance-<mode>.csv`, so the two runs can be compared directly.
Metrics: exact message match rate, character accuracy (edit distance),
letter accuracy (ignoring spaces), expected vs. detected blinks, and the
percentage of frames where no face was found. Every accuracy metric is
reported for both **fixed** and **adaptive** timing on the same videos.
The evaluation also reports **recoverable with review**: the share of
videos where the true message was either the decode itself or among the
top five alternative readings.
Each video's measured average brightness is also recorded, so "low light"
is backed by a number, not just a label.

### Results: low-light enhancement
_To be filled in from `summary_enhance-off.csv` and `summary_enhance-auto.csv`._

| Condition | Avg brightness | Face lost (off) | Face lost (auto) | Char acc (off) | Char acc (auto) |
|---|---|---|---|---|---|
| good_light | | | | | |
| low_light | | | | | |
| very_low_light | | | | | |

### Results
_To be filled in from `eval_videos/summary.csv`._

| Condition | Videos | Exact (fixed) | Char acc (fixed) | Exact (adaptive) | Char acc (adaptive) | Blinks exp / det |
|---|---|---|---|---|---|---|
| good_light | | | | | | |
| low_light | | | | | | |
| glasses | | | | | | |
| far | | | | | | |
| whatsapp_compressed | | | | | | |
| **Overall** | | | | | | |

## Limitations

- Calibration assumes the first 3 seconds show the subject with eyes
  open; real intercepted footage may not provide this.
- Thresholds (0.3 s dot/dash, 1.2 s letter, 3.0 s word) are fixed; a
  stressed or injured person may not keep consistent timing.
- Natural involuntary blinks can be mistaken for dots.
- Face tracking degrades with low resolution, poor light, or the face
  turned away.
- Offline translation only covers phrasebook phrases; other words are
  transliterated (approximate) rather than translated.
- Offline speech depends on which voices are installed on the computer.

## Future work

- Adaptive timing in live webcam mode
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
├── segmentation.py     # Fixed vs adaptive letter/word timing + pattern repair
├── alternatives.py     # Alternative readings for doubtful blinks
├── codebook.py         # Whole-phrase shortcuts (6+ symbol patterns)
├── hud_widgets.py      # Display-only interface elements (status lights, EAR graph, overlays, idle animation)
├── theme.py            # Visual theme: colours, fonts and widget styles (design only)
├── video_metadata.py   # Reads recording time / GPS from MP4/MOV files
├── enhancement.py      # Low-light assessment and enhancement (gamma + CLAHE)
├── orientation.py      # Automatic rotation fix for sideways/upside-down videos
├── blink_detector.py   # Eye Aspect Ratio calculation
├── head_gesture.py     # Nod / turn detection via landmark ratios
├── calibration.py      # Calibration for the command-line version
├── morse_translator.py # Morse code table and decoding
├── translator.py       # Online translation (MyMemory), used only when offline mode is off
├── offline_translator.py # Offline phrasebook lookup + transliteration
├── phrasebook.json     # Emergency phrases and their translations
├── build_phrasebook.py # One-time online script to fill the phrasebook
├── speech.py           # Offline-first speech (Windows voices, eSpeak NG; Google only if online)
├── logger.py           # Timestamped session log
├── main.py             # Simple command-line version (no GUI); a deliberately minimal
│                       # teaching script that does not include the newer features
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
