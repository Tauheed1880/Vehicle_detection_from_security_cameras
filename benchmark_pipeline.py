"""
FPS Benchmark for the Stage-2 AI pipeline.

Strategy (fast, non-destructive):
  Phase 1 - Video Read:    pure cv2.read loop over all frames       (~2s)
  Phase 2 - YOLO+Track:   50-frame sample with live progress       (~30-120s on CPU)
  Phase 3 - Plate/OCR:    10-crop sample (slowest stage, CPU OCR)  (~30-120s on CPU)
  Phase 4 - Annotate:     draw_detections on 50 blank frames       (<1s)

Nothing in src/ is modified.
Results are flushed immediately so progress is visible.
"""
import sys, time, statistics, glob, os
import cv2
import numpy as np

sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)   # flush every print

import config
from src.stream_reader import probe_stream
from src.detector_tracker import DetectorTracker
from src.annotator import draw_detections
from src.analyzers.plate import PlateReader

SEP = "=" * 72

def hdr(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def stats(label, times, unit="ms"):
    if not times:
        print(f"  {label:35s}  — no samples —")
        return
    scale = 1000 if unit == "ms" else 1
    avg = statistics.mean(times) * scale
    med = statistics.median(times) * scale
    mn  = min(times) * scale
    mx  = max(times) * scale
    fps = 1.0 / statistics.mean(times) if statistics.mean(times) > 0 else float("inf")
    print(f"  {label:35s}  n={len(times):>4d}  "
          f"avg={avg:7.2f}{unit}  med={med:7.2f}{unit}  "
          f"min={mn:6.2f}{unit}  max={mx:7.2f}{unit}  "
          f"→ {fps:7.2f} fps")

# ── 0. Video metadata ────────────────────────────────────────────────────────
source = config.VIDEO_SOURCE
w, h, input_fps = probe_stream(source)
cap_tmp = cv2.VideoCapture(source)
total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
cap_tmp.release()

hdr("INPUT VIDEO")
print(f"  File        : {source}")
print(f"  Resolution  : {w} × {h}")
print(f"  Native FPS  : {input_fps:.4f}")
print(f"  Total frames: {total_frames}")
print(f"  Duration    : {total_frames / input_fps:.2f} s")

# ── PHASE 1 – Video Read ────────────────────────────────────────────────────
hdr("PHASE 1 — Video Read (pure cv2.read, all frames)")
cap = cv2.VideoCapture(source)
read_times = []
t_phase = time.perf_counter()
while True:
    t0 = time.perf_counter()
    ok, _ = cap.read()
    if not ok:
        break
    read_times.append(time.perf_counter() - t0)
cap.release()
phase1_elapsed = time.perf_counter() - t_phase
print(f"  Frames read : {len(read_times)}")
print(f"  Wall time   : {phase1_elapsed:.3f} s")
stats("cv2.read per frame", read_times)
read_fps = len(read_times) / phase1_elapsed
print(f"  Read FPS    : {read_fps:.2f}")

# ── PHASE 2 – YOLO Detection + ByteTrack ────────────────────────────────────
YOLO_SAMPLE = 50   # sample this many frames
hdr(f"PHASE 2 — YOLO Detection + ByteTrack ({YOLO_SAMPLE}-frame sample)")
print("  Loading YOLO model...", flush=True)
t0 = time.perf_counter()
detector = DetectorTracker(
    config.YOLO_WEIGHTS, config.VEHICLE_CLASSES,
    conf=config.DETECT_CONF, imgsz=config.IMAGE_SIZE,
    tracker=config.TRACKER, device=config.DEVICE,
)
print(f"  Model loaded in {time.perf_counter()-t0:.2f}s")

cap = cv2.VideoCapture(source)
det_times = []
frame_idx = 0
print(f"  Running inference on {YOLO_SAMPLE} frames...", flush=True)
t_phase = time.perf_counter()
while frame_idx < YOLO_SAMPLE:
    ok, frame = cap.read()
    if not ok:
        break
    t0 = time.perf_counter()
    detections = detector.run(frame)
    dt = time.perf_counter() - t0
    det_times.append(dt)
    frame_idx += 1
    if frame_idx % 10 == 0 or frame_idx == 1:
        fps_so_far = frame_idx / (time.perf_counter() - t_phase)
        print(f"    frame {frame_idx:>3d}/{YOLO_SAMPLE}  this={dt*1000:.0f}ms  running avg={fps_so_far:.2f}fps", flush=True)
cap.release()
phase2_elapsed = time.perf_counter() - t_phase
print(f"  Wall time   : {phase2_elapsed:.3f} s")
stats("YOLO detect + ByteTrack per frame", det_times)
yolo_fps = len(det_times) / phase2_elapsed
print(f"  YOLO FPS    : {yolo_fps:.2f}")

# Save last frame + detections for phase 5
last_frame = frame
last_detections = detections

# ── Snapshot crops (loaded once, used by the plate phase below) ──────────────
snap_dir = config.SNAPSHOT_DIR
snap_files = sorted(glob.glob(os.path.join(snap_dir, "*.jpg")))
crops = [cv2.imread(f) for f in snap_files]
crops = [c for c in crops if c is not None]
print(f"  Snapshot crops: {len(crops)} (from {snap_dir})")

# ── PHASE 3 – Plate / PaddleOCR ─────────────────────────────────────────────
PLATE_SAMPLE = min(10, len(snap_files)) if snap_files else 0
hdr(f"PHASE 3 — Plate / PaddleOCR ({PLATE_SAMPLE}-crop sample)")
if PLATE_SAMPLE == 0:
    print("  No snapshot crops — skipping")
    plate_times = []
else:
    print("  Loading PlateReader...", flush=True)
    t0 = time.perf_counter()
    plate_reader = PlateReader(
        config.PLATE_DETECTOR_WEIGHTS,
        min_conf=config.PLATE_MIN_CONF, lang=config.PADDLE_LANG,
    )
    # warm-up: force model load
    dummy = np.zeros((100, 200, 3), dtype=np.uint8)
    plate_reader.read(dummy)
    print(f"  PlateReader loaded in {time.perf_counter()-t0:.2f}s", flush=True)

    plate_times = []
    sample_crops = crops[:PLATE_SAMPLE]
    t_phase = time.perf_counter()
    for i, crop in enumerate(sample_crops, 1):
        t0 = time.perf_counter()
        try:
            plate_reader.read(crop)
        except Exception as e:
            print(f"    crop {i}: ERROR {e}", flush=True)
        dt = time.perf_counter() - t0
        plate_times.append(dt)
        print(f"    crop {i:>2d}/{PLATE_SAMPLE}  {dt*1000:.0f}ms", flush=True)
    phase4_elapsed = time.perf_counter() - t_phase
    stats("Plate (YOLO+PaddleOCR) per crop", plate_times)
    print(f"  Plate FPS   : {len(plate_times)/phase4_elapsed:.2f}")
    if plate_reader._detector is None:
        print("  (no plate detector weights → bottom-half crop fallback used)")

# ── PHASE 4 – Annotation drawing ────────────────────────────────────────────
hdr("PHASE 4 — Annotation / draw_detections (50 synthetic frames)")
ann_times = []
done_ids = set(range(10))
fake_dets = [{"track_id": i, "box": [100, 100, 300, 300], "class_name": "car", "conf": 0.9}
             for i in range(10)]
t_phase = time.perf_counter()
for _ in range(50):
    frame_blank = np.zeros((h, w, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    draw_detections(frame_blank, fake_dets, done_ids)
    ann_times.append(time.perf_counter() - t0)
phase6_elapsed = time.perf_counter() - t_phase
stats("draw_detections per frame", ann_times)
print(f"  Annotate FPS: {len(ann_times)/phase6_elapsed:.2f}")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
hdr("SUMMARY — FPS at each pipeline stage")
print(f"  {'Stage':40s}  {'FPS':>10s}  {'Avg ms':>10s}  {'Note'}")
print(f"  {'-'*40}  {'-'*10}  {'-'*10}  {'-'*30}")

def row(label, times, note=""):
    if not times:
        print(f"  {label:40s}  {'N/A':>10s}  {'N/A':>10s}  {note}")
        return
    avg = statistics.mean(times)
    fps = 1.0 / avg
    print(f"  {label:40s}  {fps:>10.2f}  {avg*1000:>10.2f}  {note}")

row("1. Video Read (cv2.read)",         read_times,   "MP4")
row("2. YOLO detect + ByteTrack",       det_times,    f"{YOLO_SAMPLE}-frame sample")
row("3. Plate / PaddleOCR",             plate_times,  f"{PLATE_SAMPLE}-crop sample")
row("4. Annotation (draw boxes)",       ann_times,    "50 synthetic frames")

print()
print(f"  Input video native FPS : {input_fps:.2f}")
if det_times:
    sync_fps = 1.0 / statistics.mean(det_times)   # YOLO is the sync bottleneck
    ratio = sync_fps / input_fps
    print(f"  Pipeline processing FPS: {sync_fps:.2f}  ({ratio:.2f}× realtime)")
    print(f"  Output (annotated) FPS : {sync_fps:.2f}  (same loop, no extra encode)")

print()
print("  BOTTLENECK RANKING (synchronous per-frame stages):")
all_stages = []
if det_times:   all_stages.append(("YOLO detect + ByteTrack", statistics.mean(det_times)))
if read_times:  all_stages.append(("Video Read",               statistics.mean(read_times)))
if ann_times:   all_stages.append(("Annotation draw",          statistics.mean(ann_times)))
all_stages.sort(key=lambda x: x[1], reverse=True)
for i, (name, avg) in enumerate(all_stages, 1):
    bar = "█" * max(1, int(avg / max(a for _, a in all_stages) * 40))
    print(f"    {i}. {name:35s}  {avg*1000:7.1f}ms  {bar}")

print()
print("  NOTE: Plate/OCR runs in a background thread (AnalysisWorker)")
print("  and does NOT block the main video loop.")
print("  The main-loop bottleneck is YOLO detect+track (runs synchronously).")
print(SEP)
