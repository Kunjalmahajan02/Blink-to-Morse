"""
video_metadata.py

Reads a video FILE's own embedded metadata (recording time, GPS if
present) -- completely separate from anything decoded from blinks.

WHY KEPT SEPARATE FROM THE DECODED MESSAGE
A decoded blink message and a file's metadata come from different
sources with different trust levels. The blink message is what the
person deliberately signalled. The file metadata is whatever the
recording device (or whoever last re-saved the file) wrote, which:
  - is often stripped entirely when a video is re-encoded, forwarded,
    or uploaded to a messaging app,
  - can be wrong (a camera with an incorrect clock) or deliberately
    falsified,
  - was not verified by this software in any way.
So it is shown, if present, as clearly labelled, unverified file
information -- never merged into or presented as part of the message.

WHAT THIS READS
MP4/MOV files are structured as nested "boxes" (also called atoms).
This is a minimal, dependency-free reader for exactly two of them,
using only Python's built-in `struct` (no ffmpeg or other external
tool required):

  - 'mvhd' (Movie Header): a required, standardised box present in
    every valid MP4/MOV. Its "creation_time" field, in seconds since
    1 January 1904, is set by whatever program wrote the file.
  - 'loci' (QuickTime Location box, inside 'udta'): an OPTIONAL box
    some phones/apps write with GPS coordinates as fixed-point numbers.
    Most videos do not have this box at all.

AVI, MKV, and other container formats are not parsed (this returns
"not available" for them, rather than guessing).

Every returned value is checked for plausibility (a sane date range, a
valid latitude/longitude) before being trusted; anything that fails
those checks is treated as absent rather than reported as a real value.
"""

import datetime
import os
import struct

QT_EPOCH = datetime.datetime(1904, 1, 1, tzinfo=datetime.timezone.utc)
PLAUSIBLE_YEAR_RANGE = (1990, 2100)


def _iter_boxes(data, start, end):
    """Yields (box_type: bytes, payload_start, payload_end) for the
    top-level boxes between start and end. Corrupt/truncated data is
    handled by simply stopping, never by guessing."""
    offset = start
    while offset + 8 <= end:
        size = struct.unpack(">I", data[offset:offset + 4])[0]
        box_type = data[offset + 4:offset + 8]
        header = 8
        if size == 1:
            if offset + 16 > end:
                break
            size = struct.unpack(">Q", data[offset + 8:offset + 16])[0]
            header = 16
        elif size == 0:
            size = end - offset
        if size < header or offset + size > end:
            break
        yield box_type, offset + header, offset + size
        offset += size


def _find_box(data, path, start=0, end=None):
    """Finds the first box at a given path, e.g. [b'moov', b'udta', b'loci'].
    'meta' boxes have an extra 4-byte version/flags header before their
    child boxes, per the ISO base media file format spec."""
    end = len(data) if end is None else end
    box_type = path[0]
    for found_type, p_start, p_end in _iter_boxes(data, start, end):
        if found_type == box_type:
            if len(path) == 1:
                return p_start, p_end
            child_start = p_start + 4 if box_type == b"meta" else p_start
            result = _find_box(data, path[1:], child_start, p_end)
            if result:
                return result
    return None


def _read_creation_time(data):
    box = _find_box(data, [b"moov", b"mvhd"])
    if not box:
        return None
    start, end = box
    if end - start < 4:
        return None
    version = data[start]
    try:
        if version == 1:
            if end - start < 20:
                return None
            raw = struct.unpack(">Q", data[start + 4:start + 12])[0]
        else:
            if end - start < 12:
                return None
            raw = struct.unpack(">I", data[start + 4:start + 8])[0]
        dt = QT_EPOCH + datetime.timedelta(seconds=raw)
    except (struct.error, OverflowError, OSError):
        return None
    if raw == 0 or not (PLAUSIBLE_YEAR_RANGE[0] <= dt.year <= PLAUSIBLE_YEAR_RANGE[1]):
        return None
    return dt


def _read_gps(data):
    box = _find_box(data, [b"moov", b"udta", b"loci"])
    if not box:
        return None
    start, end = box
    payload = data[start:end]
    try:
        i = 6  # skip 4 bytes version/flags + 2 bytes language code
        name_end = payload.index(b"\x00", i)
        i = name_end + 1 + 1  # skip the name string and the 1-byte role field
        lon = struct.unpack(">i", payload[i:i + 4])[0] / 65536.0
        lat = struct.unpack(">i", payload[i + 4:i + 8])[0] / 65536.0
    except (ValueError, struct.error, IndexError):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    if lat == 0.0 and lon == 0.0:
        return None  # (0, 0) is Null Island -- almost always a missing/default value
    return lat, lon


def read_metadata(path):
    """
    Returns a dict, always with these keys:
        available:      True if the file format could be read at all
        format_note:    what was or wasn't found, and why
        creation_time:  a timezone-aware datetime, or None
        gps:            (latitude, longitude) tuple, or None
        verified:       always False -- this is raw file metadata,
                        never checked against any external source
    """
    result = {"available": False, "format_note": "", "creation_time": None,
              "gps": None, "verified": False}

    ext = os.path.splitext(path)[1].lower()
    if ext not in (".mp4", ".mov", ".m4v"):
        result["format_note"] = f"'{ext}' is not an MP4/MOV file; metadata reading is not supported for it."
        return result

    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        result["format_note"] = f"Could not read the file: {e}"
        return result

    result["available"] = True
    result["creation_time"] = _read_creation_time(data)
    result["gps"] = _read_gps(data)

    notes = ["recording time found" if result["creation_time"] else "no recording time found in file",
             "GPS location found" if result["gps"] else "no GPS location found in file"]
    result["format_note"] = "; ".join(notes) + " (most re-encoded or forwarded videos have none of this left)."
    return result


def format_summary(meta):
    """One-line, human-readable summary for display in the UI."""
    if not meta["available"]:
        return meta["format_note"]
    parts = []
    if meta["creation_time"]:
        parts.append(f"Recorded: {meta['creation_time'].strftime('%Y-%m-%d %H:%M:%S UTC')} (unverified)")
    else:
        parts.append("Recording time: not available")
    if meta["gps"]:
        lat, lon = meta["gps"]
        parts.append(f"GPS: {lat:.4f}, {lon:.4f} (unverified)")
    else:
        parts.append("GPS: not available")
    return " | ".join(parts)
