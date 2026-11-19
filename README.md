# Stage 2 - AI Processing (frames -> vehicle properties)

Reads a video or live RTSP stream, finds and tracks vehicles, analyses each NEW vehicle
once (make, model, body type, colour, decorations, number plate) and sends a record to
the backend (Stage 3).

Models: **YOLO26m** (detection) + **ByteTrack** (tracking, built into Ultralytics) +
**CLIP ViT-B/32** zero-shot (make, model, type, colour, decorations) +
**PaddleOCR** (plate text). There are no HSV colour rules and no trained make/model
classifier any more - CLIP does all of that without training.

## Flow

```
Stage 2A (every frame)                        Stage 2B (once per new vehicle)
6  stream_reader.py     OpenCV frame           crop   cropper.py
7  detector_tracker.py  YOLO26m                10+11  analyzers/clip_classifier.py  CLIP zero-shot
8  detector_tracker.py  ByteTrack                     (make, model, type, colour, decorations)
9  vehicle_registry.py  new track ID?          12     analyzers/plate.py       plate -> PaddleOCR
A  annotator.py         boxes + video out      13     record_builder.py        dict + JPEG
                                               14     backend_client.py        HTTP POST
                                               worker.py runs 10-14 in a background thread
```

## Project structure

```
stage2_ai_worker/
  main.py                 entry point, connects all steps
  config.py               ALL settings (paths, thresholds, URLs)
  requirements.txt        pip packages
  models/                 yolo26m.pt, yolo26s_plate.pt (CLIP is downloaded by transformers)
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
      clip_classifier.py  steps 10 + 11 (CLIP zero-shot)
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
(move it into `models/` afterwards) and the CLIP model (cached by `transformers`).

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
| CLIP | make/model/type/colour/decorations, each `("label", score)` |
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

`type` is CLIP's body type (Sedan/SUV/...); the raw YOLO class is kept in
`detector_class`. Records are written in detection order (ascending `first_seen_frame`),
not in the order the analysis finished. Agree these field names with the Stage 3
developer before you start sending.

## About CLIP

`src/analyzers/clip_classifier.py` loads `openai/clip-vit-base-patch32` through Hugging
Face `transformers` (downloaded once, then cached). Every class is a short English text
prompt ("a photo of a Toyota car"); CLIP embeds the text and the vehicle crop into the
same space and the closest prompt wins. It is used in zero-shot mode - nothing is trained.

- The reported score is a **similarity** (a softmax over the label set), **not** a
  calibrated probability. Do not read it as "90% chance".
- Below `CLIP_MIN_CONF` the answer becomes `unknown` instead of a forced guess.
- Make and model are kept consistent: a known model implies its make (Corolla -> Toyota).
- Decorations are looked for on the upper part of the crop (`CLIP_DECORATION_REGION`),
  because small stickers are easy to miss in a full vehicle crop.
- The number plate is **not** handled by CLIP - that stays YOLO + PaddleOCR in `plate.py`.
- If `torch`/`transformers` are missing or the download fails, CLIP degrades to
  `unknown` and the rest of the pipeline keeps running.

Zero-shot CLIP is **not** reliable for exact models or for small stickers. Treat make,
model and decoration as hints that need testing on real footage, not as ground truth.

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
| Many `unknown` labels | lower `CLIP_MIN_CONF`, or use clearer text prompts |
| CLIP on GPU | set `CLIP_DEVICE = "cuda"` |
| CLIP cache location | set `CLIP_CACHE_DIR` |

## Tests

```
pip install -r requirements-dev.txt
pytest
```

## Known limits

- A vehicle that is lost and re-found by the tracker gets a new ID and is analysed twice.
- Night, glare and bad angles reduce CLIP and plate accuracy. Test on the real camera early.
- CLIP predicts a colour label from a fixed list; unusual or two-tone paints fall into `other`.
- Ultralytics YOLO is AGPL-3.0: closed-source company use needs an Enterprise License.
