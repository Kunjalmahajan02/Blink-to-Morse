"""
orientation.py

Automatic rotation correction for recorded videos.

WHY
---
Phone videos are often filmed sideways or upside down. Some players fix
this using a rotation tag in the file, but that tag is not always read
(and is often lost when a video is forwarded or re-encoded). The result
is a face lying on its side, which MediaPipe does not detect well.

HOW
---
Before analysing a video, a handful of frames spread across the first
few seconds are tried in each of the four orientations (0, 90, 180,
270 degrees). The orientation in which a face is found most often is
used for the whole video. If the unrotated video already finds a face
in most sample frames, it is used as-is, so normal videos only pay for
one quick check.

This is only for recorded video. A live webcam is always upright.
"""

import cv2

ROTATIONS = [
    (None, "0"),
    (cv2.ROTATE_90_CLOCKWISE, "90"),
    (cv2.ROTATE_180, "180"),
    (cv2.ROTATE_90_COUNTERCLOCKWISE, "270"),
]

SAMPLE_FRAMES = 8
SAMPLE_SPAN_S = 4.0
GOOD_ENOUGH = 0.75   # if upright finds a face in 75%+ of samples, skip the rest


def rotate(frame, rotation):
    return frame if rotation is None else cv2.rotate(frame, rotation)


def detect_rotation(cap, face_mesh_factory):
    """
    cap: an opened cv2.VideoCapture (its position is restored to frame 0).
    face_mesh_factory: a function returning a fresh MediaPipe FaceMesh
        (a fresh one per orientation so tracking from one orientation
        doesn't leak into another).

    Returns (rotation_code_or_None, label, hits_by_label dict).
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(SAMPLE_SPAN_S * fps)
    span = min(total, int(SAMPLE_SPAN_S * fps))
    step = max(1, span // SAMPLE_FRAMES)

    samples = []
    for idx in range(0, span, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            samples.append(frame)
        if len(samples) >= SAMPLE_FRAMES:
            break
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not samples:
        return None, "0", {}

    hits = {}
    best = (None, "0", -1)
    for rotation, label in ROTATIONS:
        mesh = face_mesh_factory()
        found = 0
        for frame in samples:
            rgb = cv2.cvtColor(rotate(frame, rotation), cv2.COLOR_BGR2RGB)
            if mesh.process(rgb).multi_face_landmarks:
                found += 1
        mesh.close()
        hits[label] = found
        if found > best[2]:
            best = (rotation, label, found)
        if label == "0" and found >= GOOD_ENOUGH * len(samples):
            break   # upright works: don't waste time on the other angles
    return best[0], best[1], hits
