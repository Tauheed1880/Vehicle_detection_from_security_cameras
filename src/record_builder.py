"""
Step 13 - Build the vehicle record.

Input : detection dict (from 2A) + crop + analysis results (from 12 plate)
Output: (record dict, JPEG bytes)

`detector_class` holds the raw YOLO class. Make/model/type/colour/decorations
have no model behind them, so they stay "unknown".
"""
from datetime import datetime
from pathlib import Path

import cv2

# Value used for every attribute the pipeline cannot fill in (no model yet, or a
# vehicle that was never analysed).
UNKNOWN = "unknown"


def _split_time(when):
    """Format a datetime as separate `date` (DD/MM/YYYY) and `time` (HH:MM:SS).

    The value is the instant the vehicle was FIRST tracked (kept on the
    detection dict). Falls back to now() only when no timestamp was recorded.
    """
    when = when or datetime.now()
    return when.strftime("%d/%m/%Y"), when.strftime("%H:%M:%S")


def build_record(det, crop, results, camera_id, jpeg_quality=90):
    """`results` keys: type, type_conf, make, model, make_model_conf, colour,
    colour_conf, decoration, decoration_conf, plate, plate_conf, province."""
    ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    if not ok:
        raise ValueError("Could not encode the snapshot as JPEG.")
    jpg_bytes = buffer.tobytes()

    date, time = _split_time(det.get("first_seen_time"))
    record = {
        "track_id": det["track_id"],
        "first_seen_frame": det.get("first_seen_frame"),
        "camera_id": camera_id,
        "date": date,
        "time": time,
        "type": results["type"],
        "type_conf": round(results["type_conf"], 3),
        "detector_class": det["class_name"],
        "make": results["make"],
        "model": results["model"],
        "make_model_conf": round(results["make_model_conf"], 3),
        "colour": results["colour"],
        "colour_conf": round(results["colour_conf"], 3),
        "decoration": results["decoration"],
        "decoration_conf": round(results["decoration_conf"], 3),
        "plate": results["plate"],
        "plate_conf": round(results["plate_conf"], 3),
        "province": results.get("province"),
        "confidence": round(det["conf"], 3),
        "snapshot_file": f"{det['track_id']}.jpg",
    }
    return record, jpg_bytes


def build_minimal_record(det, crop, camera_id, jpeg_quality=90):
    """Detection-only record for a tracked vehicle that was never analysed.

    Same schema as build_record, but the attribute fields keep their
    "unavailable" values (unknown / 0.0) and there is no plate or province -
    nothing is invented. The snapshot is the best crop seen, or None when no
    usable crop was ever captured.
    """
    jpg_bytes = None
    if crop is not None and getattr(crop, "size", 0):
        ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY,
                                                 int(jpeg_quality)])
        if ok:
            jpg_bytes = buffer.tobytes()

    date, time = _split_time(det.get("first_seen_time"))
    record = {
        "track_id": det["track_id"],
        "first_seen_frame": det.get("first_seen_frame"),
        "camera_id": camera_id,
        "date": date,
        "time": time,
        "type": UNKNOWN,
        "type_conf": 0.0,
        "detector_class": det["class_name"],
        "make": UNKNOWN,
        "model": UNKNOWN,
        "make_model_conf": 0.0,
        "colour": UNKNOWN,
        "colour_conf": 0.0,
        "decoration": UNKNOWN,
        "decoration_conf": 0.0,
        "plate": None,
        "plate_conf": 0.0,
        "province": None,
        "confidence": round(det["conf"], 3),
        "snapshot_file": f"{det['track_id']}.jpg" if jpg_bytes else None,
    }
    return record, jpg_bytes


def save_snapshot(jpg_bytes, track_id, snapshot_dir):
    """Write the JPEG to disk and return its path."""
    folder = Path(snapshot_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{track_id}.jpg"
    path.write_bytes(jpg_bytes)
    return path
