"""
evaluate.py

Measures how accurate the system actually is, instead of just showing
that a demo works.

It runs the EXACT same detection engine used by the app's
"Upload Intercept File" mode (pipeline.py, with the same 5-second
calibration) on a folder of test videos, compares what the system
decoded against what the person was really trying to blink, and
reports accuracy numbers.

HOW TO USE
----------
1. Put your test videos in a folder, e.g. eval_videos/
   Name them clearly, e.g. p1_goodlight_sos.mp4, p2_lowlight_help.mp4

2. Run once to create a labels template:
       python evaluate.py eval_videos
   This creates eval_videos/labels.csv listing every video.

3. Open labels.csv in Excel/Notepad and fill in, for every video:
       expected   -> what the person actually blinked, e.g. SOS
       condition  -> the recording condition, e.g. good_light, low_light,
                     glasses, far, whatsapp_compressed

4. Run again:
       python evaluate.py eval_videos
   Results are printed and saved to:
       eval_videos/results_enhance-auto.csv   (one row per video)
       eval_videos/summary_enhance-auto.csv   (averages overall and per condition)

5. To measure what low-light enhancement actually changes, run the same
   videos with it switched off and compare the two summary files:
       python evaluate.py eval_videos --enhance off
       python evaluate.py eval_videos --enhance auto

METRICS (explained in plain words)
----------------------------------
- exact_match:      did the whole message come out perfectly? (yes/no)
- char_accuracy:    how close the decoded text is to the expected text,
                    counting spaces. 1.0 = perfect. Based on "edit
                    distance": the number of letters you would need to
                    add, remove or change to fix the output.
- letter_accuracy:  same, but ignoring spaces. Useful because a missed
                    word gap is a timing error, not a blink error.
- expected_blinks / detected_blinks: how many dots+dashes the message
                    needs vs. how many blinks the system actually counted.
                    Big differences point to missed or extra blinks.
- face_lost_pct:    % of frames where no face was found at all. High
                    values explain bad results on poor footage.
- avg_brightness:   mean frame brightness (0-255, LAB lightness). Gives
                    the "low light" condition an objective number.
- enhanced_pct:     % of frames where night enhancement was applied.

- alt_rank / recoverable_with_review: whether the true message was the
                    adaptive decode (rank 0) or appeared among the top-5
                    alternative readings (see alternatives.py). This measures
                    how often a human reviewer could recover the right message.

Every metric is reported twice: FIXED timing (the original 1.2 s / 3.0 s
rule, exactly what the live app does) and ADAPTIVE timing (rhythm-based
gaps + pattern repair, see segmentation.py), so the improvement can be
measured on the same videos.
"""

import csv
import os
import sys
import time

import cv2
import mediapipe as mp
import orientation

from pipeline import BlinkMorsePipeline
from blink_detector import average_ear
from head_gesture import get_head_metrics
from morse_translator import MORSE_CODE, decode_letter
import segmentation
import alternatives

CALIBRATION_SECONDS = 3.0   # same as the GUI
THRESHOLD_RATIO = 0.75      # same as the GUI
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

LETTER_TO_MORSE = {letter: code for code, letter in MORSE_CODE.items()}


# ---------------- TEXT COMPARISON HELPERS ----------------

def normalize(text):
    """Uppercase and collapse multiple spaces, so 'sos ' == 'SOS'."""
    return " ".join(text.upper().split())


def edit_distance(a, b):
    """Levenshtein distance: minimum number of single-character
    insertions, deletions or substitutions to turn a into b."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def accuracy(expected, decoded):
    """1 - (edit distance / expected length), floored at 0."""
    if not expected:
        return 1.0 if not decoded else 0.0
    return max(0.0, 1.0 - edit_distance(expected, decoded) / len(expected))


def expected_symbol_count(text):
    """How many dots+dashes are needed to blink this message."""
    return sum(len(LETTER_TO_MORSE.get(ch, "")) for ch in text if ch != " ")


# ---------------- RUNNING ONE VIDEO ----------------

def run_video(path, enhance_mode="auto"):
    """Runs calibration + the full pipeline on one video, exactly the way
    the GUI's video-upload mode does. Returns a dict of raw results."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": "could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    # Same settings as the app's video-upload mode: no head gestures,
    # automatic orientation correction.
    pipeline = BlinkMorsePipeline(enhance_mode=enhance_mode, gestures_enabled=False)
    rotation, rot_label, _ = orientation.detect_rotation(
        cap, lambda: mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                     refine_landmarks=True, min_detection_confidence=0.5))
    cal_frames = int(CALIBRATION_SECONDS * fps)
    cal_ear, cal_pitch, cal_yaw = [], [], []
    calibrated = False
    ear_threshold = None

    frame_idx = 0
    detected_blinks = 0
    blink_log = []
    frames_no_face = 0
    brightness_sum = 0.0
    frames_enhanced = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = orientation.rotate(frame, rotation)
        frame_idx += 1
        now = frame_idx / fps   # the video's own clock, not wall-clock time

        # --- Phase 1: calibration (first CALIBRATION_SECONDS) ---
        if not calibrated:
            results, light, _ = pipeline.detect(frame)
            brightness_sum += light["brightness"]
            frames_enhanced += int(light["applied"])
            h, w, _ = frame.shape
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                cal_ear.append(average_ear(lm, w, h))
                p, y = get_head_metrics(lm, w, h)
                if p is not None:
                    cal_pitch.append(p)
                    cal_yaw.append(y)
            else:
                frames_no_face += 1

            if frame_idx >= cal_frames:
                if cal_ear:
                    ear_threshold = (sum(cal_ear) / len(cal_ear)) * THRESHOLD_RATIO
                else:
                    ear_threshold = 0.21
                pitch_b = sum(cal_pitch) / len(cal_pitch) if cal_pitch else None
                yaw_b = sum(cal_yaw) / len(cal_yaw) if cal_yaw else None
                pipeline.set_calibration(ear_threshold, pitch_b, yaw_b)
                calibrated = True
            continue

        # --- Phase 2: decoding ---
        before = len(pipeline.morse_buffer)
        events = pipeline.process_frame(frame, now)
        brightness_sum += events["light"]["brightness"]
        frames_enhanced += int(events["light"]["applied"])
        if not events["face_found"]:
            frames_no_face += 1
        # A new dot or dash was added this frame -> one blink detected.
        # Record its timing so it can also be decoded with adaptive timing.
        if len(pipeline.morse_buffer) > before:
            detected_blinks += 1
            blink_log.append({"t_start": pipeline.close_start_time,
                              "t_end": pipeline.last_blink_end_time,
                              "symbol": pipeline.morse_buffer[-1]})

    cap.release()

    if not calibrated:
        return {"error": f"video shorter than the {CALIBRATION_SECONDS:.0f}s calibration period"}

    # The video may end before the letter-pause timer fires, so commit
    # whatever is still sitting in the buffer.
    if pipeline.morse_buffer:
        pipeline.decoded_text += decode_letter(pipeline.morse_buffer)
        pipeline.morse_buffer = ""

    adaptive = segmentation.decode(blink_log, "adaptive")
    _base, alts = alternatives.reading_alternatives(blink_log, "adaptive")

    return {
        "decoded": normalize(pipeline.decoded_text),
        "decoded_adaptive": normalize(adaptive["text"]),
        "reconstructed": adaptive["reconstructed_count"],
        "alternative_texts": [normalize(a["text"]) for a in alts],
        "detected_blinks": detected_blinks,
        "ear_threshold": ear_threshold,
        "face_lost_pct": 100.0 * frames_no_face / max(frame_idx, 1),
        "avg_brightness": brightness_sum / max(frame_idx, 1),
        "rotation": rot_label,
        "enhanced_pct": 100.0 * frames_enhanced / max(frame_idx, 1),
        "duration_s": frame_idx / fps,
        "processing_s": time.time() - start,
    }


# ---------------- LABELS FILE ----------------

def create_labels_template(folder, labels_path):
    videos = sorted(f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTS))
    if not videos:
        print(f"No videos found in {folder}. Add some .mp4/.mov/.avi/.mkv files first.")
        return
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "expected", "condition"])
        for v in videos:
            writer.writerow([v, "", ""])
    print(f"Created {labels_path} with {len(videos)} video(s).")
    print("Fill in the 'expected' and 'condition' columns, then run this command again.")


def load_labels(labels_path):
    with open(labels_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------- MAIN ----------------

def main():
    args = sys.argv[1:]
    enhance_mode = "auto"
    if "--enhance" in args:
        i = args.index("--enhance")
        if i + 1 >= len(args) or args[i + 1] not in ("off", "auto", "on"):
            print("--enhance must be followed by off, auto or on")
            sys.exit(1)
        enhance_mode = args[i + 1]
        del args[i:i + 2]
    if not args:
        print("Usage: python evaluate.py <folder_with_videos> [--enhance off|auto|on]")
        sys.exit(1)

    folder = args[0]
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        sys.exit(1)

    labels_path = os.path.join(folder, "labels.csv")
    if not os.path.exists(labels_path):
        create_labels_template(folder, labels_path)
        return

    rows = []
    for label in load_labels(labels_path):
        filename = label.get("filename", "").strip()
        expected = normalize(label.get("expected", ""))
        condition = label.get("condition", "").strip() or "unspecified"

        if not filename:
            continue
        if not expected:
            print(f"[skip] {filename}: no expected text filled in labels.csv")
            continue
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            print(f"[skip] {filename}: file not found")
            continue

        print(f"[run ] {filename} ({condition}) ...", end=" ", flush=True)
        result = run_video(path, enhance_mode)
        if "error" in result:
            print(f"ERROR: {result['error']}")
            continue

        decoded = result["decoded"]
        dec_a = result["decoded_adaptive"]
        row = {
            "filename": filename,
            "condition": condition,
            "expected": expected,
            "decoded": decoded,
            "exact_match": int(decoded == expected),
            "char_accuracy": round(accuracy(expected, decoded), 3),
            "letter_accuracy": round(accuracy(expected.replace(" ", ""), decoded.replace(" ", "")), 3),
            "decoded_adaptive": dec_a,
            "exact_match_adaptive": int(dec_a == expected),
            "char_accuracy_adaptive": round(accuracy(expected, dec_a), 3),
            "letter_accuracy_adaptive": round(accuracy(expected.replace(" ", ""), dec_a.replace(" ", "")), 3),
            "reconstructed_letters": result["reconstructed"],
            # Where the true message appears: 0 = the adaptive decode itself was
            # correct, 1-5 = rank among the suggested alternative readings,
            # blank = not recoverable from the top 5.
            "alt_rank": (0 if dec_a == expected else
                         next((k for k, t in enumerate(result["alternative_texts"], 1) if t == expected), "")),
            "expected_blinks": expected_symbol_count(expected),
            "detected_blinks": result["detected_blinks"],
            "ear_threshold": round(result["ear_threshold"], 3),
            "face_lost_pct": round(result["face_lost_pct"], 1),
            "avg_brightness": round(result["avg_brightness"], 1),
            "rotation_deg": result["rotation"],
            "enhanced_pct": round(result["enhanced_pct"], 1),
            "duration_s": round(result["duration_s"], 1),
            "processing_s": round(result["processing_s"], 1),
        }
        rows.append(row)
        print(f"expected '{expected}' -> fixed '{decoded}' ({row['char_accuracy']:.0%}) | "
              f"adaptive '{dec_a}' ({row['char_accuracy_adaptive']:.0%})")

    if not rows:
        print("No videos were evaluated.")
        return

    results_path = os.path.join(folder, f"results_enhance-{enhance_mode}.csv")
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- Summary: overall + per condition ----
    def summarize(group):
        n = len(group)
        return {
            "videos": n,
            "exact_match_rate": round(sum(r["exact_match"] for r in group) / n, 3),
            "avg_char_accuracy": round(sum(r["char_accuracy"] for r in group) / n, 3),
            "avg_letter_accuracy": round(sum(r["letter_accuracy"] for r in group) / n, 3),
            "exact_match_rate_adaptive": round(sum(r["exact_match_adaptive"] for r in group) / n, 3),
            "recoverable_with_review": round(sum(1 for r in group if r["alt_rank"] != "") / n, 3),
            "avg_char_accuracy_adaptive": round(sum(r["char_accuracy_adaptive"] for r in group) / n, 3),
            "avg_letter_accuracy_adaptive": round(sum(r["letter_accuracy_adaptive"] for r in group) / n, 3),
            "blinks_expected": sum(r["expected_blinks"] for r in group),
            "blinks_detected": sum(r["detected_blinks"] for r in group),
            "avg_face_lost_pct": round(sum(r["face_lost_pct"] for r in group) / n, 1),
            "avg_brightness": round(sum(r["avg_brightness"] for r in group) / n, 1),
        }

    summary_rows = [{"group": "OVERALL", **summarize(rows)}]
    for cond in sorted({r["condition"] for r in rows}):
        summary_rows.append({"group": cond, **summarize([r for r in rows if r["condition"] == cond])})

    summary_path = os.path.join(folder, f"summary_enhance-{enhance_mode}.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n========== SUMMARY  (low-light enhancement: {enhance_mode.upper()}) ==========")
    print(f"{'':<22}{'':>7}{'--- FIXED TIMING ---':>30}{'--- ADAPTIVE TIMING ---':>32}")
    print(f"{'group':<22}{'videos':>7}{'exact':>8}{'char':>8}{'letter':>9}"
          f"{'exact':>11}{'char':>8}{'letter':>9}{'blinks exp/det':>17}{'light':>7}{'face lost':>11}")
    for s_ in summary_rows:
        print(f"{s_['group']:<22}{s_['videos']:>7}"
              f"{s_['exact_match_rate']:>8.0%}{s_['avg_char_accuracy']:>8.0%}{s_['avg_letter_accuracy']:>9.0%}"
              f"{s_['exact_match_rate_adaptive']:>11.0%}{s_['avg_char_accuracy_adaptive']:>8.0%}"
              f"{s_['avg_letter_accuracy_adaptive']:>9.0%}"
              f"{str(s_['blinks_expected']) + '/' + str(s_['blinks_detected']):>17}"
              f"{s_['avg_brightness']:>7.0f}{s_['avg_face_lost_pct']:>10.1f}%")
    overall = summary_rows[0]
    print(f"\nRecoverable with analyst review (true message = adaptive decode or in top-5 "
          f"alternatives): {overall['recoverable_with_review']:.0%} of videos")
    print(f"\nPer-video results: {results_path}")
    print(f"Summary:           {summary_path}")


if __name__ == "__main__":
    main()
