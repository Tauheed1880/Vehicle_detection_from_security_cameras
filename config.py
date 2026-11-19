"""
config.py - every setting of the Stage 2 AI worker lives here.

When the real camera arrives, change VIDEO_SOURCE and nothing else.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------- Step 6: input
# A video file for development, or the live stream from Stage 1:
#   VIDEO_SOURCE = "rtsp://localhost:8554/traffic"
# Override without editing this file, e.g. in Docker:
#   docker run -e VIDEO_SOURCE=data/myclip.mp4 ...
VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", str(BASE_DIR / "data" / "traffic.mp4"))
CAMERA_ID = "cam-01"
RECONNECT_ATTEMPTS = 5      # RTSP only: retries when the stream drops
LOOP_VIDEO_FILE = True      # file only: endless loop until the process is stopped

# ------------------------------------------------- Steps 7 + 8: YOLO26m + ByteTrack
YOLO_WEIGHTS = str(BASE_DIR / "models" / "yolo26m.pt")   # downloads on first run
VEHICLE_CLASSES = [2, 3, 5, 7]   # COCO ids: car, motorcycle, bus, truck
DETECT_CONF = 0.35               # ignore detections below this confidence
IMAGE_SIZE = 480                 # YOLO input size (reduced from 640 → ~35% faster on CPU)
TRACKER = "bytetrack.yaml"
DEVICE = None                    # None = automatic, or "cpu", "0" (first GPU)

# ---- Priority-2 CPU speed knobs (Phase 2 optimisation) ----
# Torch threads: 0 = let PyTorch choose (uses all cores); baseline was 1 thread
TORCH_THREADS = 0
# Faster NMS: treat all classes the same during suppression.
AGNOSTIC_NMS = True

# ------------------------------------------------------------ Step 9: new vehicle
# A vehicle is analysed ONCE per track, using its BEST crop. Instead of sending
# the first acceptable frame, the loop keeps watching a new track and submits when
# its crop stops improving, the box has been stable for a few frames and the crop
# is sharp enough - or at MAX_OBSERVE_FRAMES / when the track disappears, whichever
# comes first (the best valid crop is used then).
MIN_BOX_AREA = 12000   # pixels (width x height); smaller boxes are too blurry
EDGE_MARGIN = 5        # skip boxes touching the frame edge (vehicle is cut off)
MIN_OBSERVE_FRAMES = 3  # floor before a quality-based submit is allowed (not a trigger)
MAX_OBSERVE_FRAMES = 45 # hard cap: submit the best crop even if quality never settles

# Shape guards: reject incomplete / extremely narrow or short boxes. The band is
# deliberately generous so small, distant vehicles (which keep a normal shape)
# still pass; it only drops degenerate slivers.
MIN_BOX_WIDTH = 32     # pixels; boxes narrower than this are cut off
MIN_BOX_HEIGHT = 32    # pixels; boxes shorter than this are cut off
MIN_ASPECT_RATIO = 0.5 # width / height lower bound
MAX_ASPECT_RATIO = 4.0 # width / height upper bound

# Adaptive submission: compare consecutive frames and the crop's own quality.
STABLE_FRAMES = 3          # consecutive stable frames needed before submitting
STABLE_IOU = 0.80          # IoU vs the previous frame's box that counts as stable
QUALITY_CONVERGENCE_TOL = 0.05  # submit once the score is within 5% of the best seen
SIZE_REF_AREA_RATIO = 0.02 # box area (share of the frame) that scores full size marks
TRUNC_MARGIN = 8           # px band near the frame border used for a truncation penalty

# Crop sharpness (variance of the Laplacian), measured on a width-normalised crop.
CROP_NORM_WIDTH = 128      # resize crops to this width before measuring sharpness
SHARPNESS_REF = 60.0       # sharpness that scores full marks
SHARPNESS_MIN = 25.0       # crops below this are not considered "ready" yet

# Shutdown/expiry recovery of vehicles that never produced a good box. The area
# floor is lower so genuinely distant vehicles are still recorded, but the shape
# guards above still apply, so a sliver is never analysed.
RECOVERY_MIN_BOX_AREA = 4000

# ------------------------------------------------------ Step 12: plate + PaddleOCR
# Primary: license_plate_detector.pt - a single-class ("license_plate") YOLO detector.
# Fallback (if primary absent): bottom half of vehicle crop.
PLATE_DETECTOR_WEIGHTS = str(BASE_DIR / "models" / "license_plate_detector.pt")
PLATE_DETECTOR_CONF = 0.25       # detection confidence for the plate locator
PLATE_MIN_CONF = 0.50            # minimum OCR confidence to accept a plate text
PADDLE_LANG = "en"

# Plate-first frame selection: how many candidate vehicle crops (ranked by the
# EXISTING quality score) to keep per vehicle, so the Stage-2 worker can prefer
# a frame where a number plate is actually visible. Bounded, so the extra
# plate-detector calls stay cheap; they run in the worker thread, never in the
# video loop, and the observation window / submission timing are unchanged.
PLATE_CANDIDATE_FRAMES = 3

# Minimum plate-region sharpness for a detected plate to count as the
# plate-visible candidate. Same measurement as SHARPNESS_MIN above (variance of
# the Laplacian on a width-normalised crop), so it is image clarity, NOT the
# plate detector's confidence. Anchored to the existing SHARPNESS_MIN; plate
# regions are smaller than vehicle crops, so this initial value may need
# calibration on real footage.
PLATE_SHARPNESS_MIN = 25.0

# -------------------------------------------------- Step 13: record + snapshot
SNAPSHOT_DIR = str(BASE_DIR / "snapshots")
JPEG_QUALITY = 90
# Vehicle records are written here as one JSON array of record objects.
# The file is reset at the start of every run, so it holds exactly one record
# per unique track ID for that run (track IDs restart at 1 on each run, so
# appending would otherwise repeat IDs across runs).
RECORDS_FILE = str(BASE_DIR / "records.json")
RESET_RECORDS_ON_START = True    # truncate RECORDS_FILE when a run starts
RESET_SNAPSHOTS_ON_START = True  # delete old *.jpg in SNAPSHOT_DIR when a run starts

# A track that has not been seen for this many frames is treated as gone. If it
# was never analysed, its best crop is analysed late instead of being dropped.
TRACK_TIMEOUT_FRAMES = 30

# ------------------------------------------------------ Step 14: send to backend
SEND_TO_BACKEND = False          # switch to True when the Stage 3 server is ready
BACKEND_URL = "http://localhost:8000/vehicles"
BACKEND_TIMEOUT_SEC = 5
BACKEND_RETRIES = 2

# ------------------------------------- Step 14b: live ingest uploader (Stage 3)
# The pipeline writes records.json live; post_records.Uploader tails that file
# and POSTs every new record as raw JSON (x-api-key auth). It runs as a
# background thread inside the pipeline process, so ONE container runs both the
# pipeline and the uploader together - no second process, no supervisor.
# Override in Docker with env vars: inside a container "localhost" is the
# container, not the host, so use host.docker.internal or the LAN IP.
INGEST_ENABLED = os.environ.get("INGEST_ENABLED", "1").lower() not in ("0", "false", "no")
INGEST_URL = os.environ.get("INGEST_URL", "http://10.1.19.78:3000/api/detections/ingest")
INGEST_API_KEY = os.environ.get("INGEST_API_KEY", "N1UoJkQ1LWn7wPZmR1v3NofdTHbcn0Sx")
INGEST_STATE_FILE = os.environ.get("INGEST_STATE", str(BASE_DIR / "ingest_state.json"))
INGEST_POLL_SEC = float(os.environ.get("INGEST_INTERVAL", "1.0"))

# --------------------------------------------------- Worker (keeps video smooth)
USE_WORKER_THREAD = True         # run the heavy models in a background thread
QUEUE_SIZE = 16                  # max vehicles waiting for analysis

# ------------------------------------------------------ Branch A: annotated video
SHOW_WINDOW = True               # show a local OpenCV window (press q to quit)
PUBLISH_ANNOTATED = False        # push annotated video to MediaMTX (needs FFmpeg)
ANNOTATED_RTSP = "rtsp://localhost:8554/annotated"
