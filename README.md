# Stage 2 - AI Processing (frames -> vehicle properties)

Reads a video or live RTSP stream, finds and tracks vehicles, reads each NEW vehicle's
number plate once and sends a record to the backend (Stage 3).

Models: **YOLO26m** (detection) + **ByteTrack** (tracking, built into Ultralytics) +
**PaddleOCR** (plate text). Make/model/type/colour/decorations have no model behind
them yet, so those fields are recorded as `unknown`.

## Flow

```
Stage 2A (every frame)                        Stage 2B (once per new vehicle)
6  stream_reader.py     OpenCV frame           crop   cropper.py
7  detector_tracker.py  YOLO26m                12     analyzers/plate.py       plate -> PaddleOCR
8  detector_tracker.py  ByteTrack              13     record_builder.py        dict + JPEG
9  vehicle_registry.py  new track ID?          14     backend_client.py        HTTP POST
A  annotator.py         boxes + video out      worker.py runs 12-14 in a background thread
```

## Project structure

```
stage2_ai_worker/
  main.py                 entry point, connects all steps
  config.py               ALL settings (paths, thresholds, URLs)
  requirements.txt        pip packages
  models/                 yolo26m.pt, license_plate_detector.pt
  data/                   test video (traffic.mp4)
  snapshots/              output: one JPEG per vehicle
  src/
    stream_reader.py      step 6
    detector_tracker.py   steps 7 + 8 (tensors -> plain Python)
    vehicle_registry.py   step 9
    cropper.py            crop + "is this box good enough?"
    annotator.py          branch A (draw boxes, FFmpeg -> MediaMTX)
    worker.py             runs stage 2B off the main loop
    record_builder.py     step 13
    backend_client.py     step 14
    analyzers/
      plate.py            step 12
  tests/                  unit tests (no model downloads needed)
```

## Install

```
python -m venv venv
venv\Scripts\activate          # Windows      (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Put a test video at `data/traffic.mp4`. The first run downloads `yolo26m.pt`
(move it into `models/` afterwards).

## Run

```
python main.py                                  # window opens, press q to quit
python main.py --source data/traffic.mp4 --headless
python main.py --source rtsp://localhost:8554/traffic     # live stream from Stage 1
python main.py --send                           # POST records to the backend
python main.py --publish                        # annotated video -> MediaMTX (needs FFmpeg)
```

Records are written to `records.json` (one JSON array of record objects) and each snapshot is
saved in `snapshots/`. With `--send` the same records are POSTed to the backend instead.

## Data formats

| Where | Format |
|---|---|
| frame | NumPy array (H, W, 3), uint8, **BGR** |
| detection | `{"track_id": 12, "box": [x1,y1,x2,y2], "class_name": "car", "conf": 0.91}` |
| crop | NumPy array (h, w, 3), BGR (a copy) |
| plate | `("ABC-123", 0.93, plate_crop)` |
| record | JSON dict (below) + JPEG bytes |
| POST to backend | `multipart/form-data`: field `payload` (JSON string) + file `snapshot` (JPEG) |

```json
{
  "track_id": 12, "first_seen_frame": 318, "camera_id": "cam-01",
  "time": "2026-11-15T10:42:15",
  "type": "Sedan", "type_conf": 0.71, "detector_class": "car",
  "make": "Toyota", "model": "Corolla", "make_model_conf": 0.66,
  "colour": "white", "colour_conf": 0.58,
  "decoration": "none", "decoration_conf": 0.83,
  "plate": "ABC-123", "plate_conf": 0.93,
  "confidence": 0.91, "snapshot_file": "12.jpg"
}
```

The raw YOLO class is kept in `detector_class`; make/model/type/colour/decorations are
`unknown` until a model is added for them. Records are written in detection order
(ascending `first_seen_frame`), not in the order the analysis finished. Agree these field
names with the Stage 3 developer before you start sending.

## Plate accuracy

PaddleOCR works best on a tight plate crop. Train a small YOLO plate detector, save it as
`models/yolo26s_plate.pt`, and it is used automatically. Without it the bottom part of the
vehicle is used (works, but less accurate).

## Tuning (config.py)

| Problem | Setting |
|---|---|
| Vehicles never get analysed | lower `MIN_BOX_AREA` or set `EDGE_MARGIN = 0` |
| Blurry/small crops | raise `MIN_BOX_AREA` |
| Too slow | use `yolo26n.pt` or `yolo26s.pt`, set `DEVICE = "0"` for a GPU |
| Wrong plates | train `yolo26s_plate.pt`, raise `PLATE_MIN_CONF` |

## Tests

```
pip install -r requirements-dev.txt
pytest
```

## Known limits

- A vehicle that is lost and re-found by the tracker gets a new ID and is analysed twice.
- Night, glare and bad angles reduce plate accuracy. Test on the real camera early.
- Ultralytics YOLO is AGPL-3.0: closed-source company use needs an Enterprise License.
