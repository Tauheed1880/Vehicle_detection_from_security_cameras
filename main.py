"""
Stage 2 - AI Processing: frames -> vehicle properties.

    6  OpenCV reads frame
    7  YOLO26m detects vehicles
    8  ByteTrack gives each vehicle a track ID
    9  New vehicle? (only once per ID)  -> worker: PaddleOCR
    13 Build record, 14 POST to backend
    A  Draw boxes and show / publish the annotated video

Run:  python main.py            (settings are in config.py)
      python main.py --source data/traffic.mp4 --headless --send
"""
import argparse
import logging
import signal
from datetime import datetime
from pathlib import Path

import cv2

import config
from post_records import Uploader
from src.analyzers.plate import PlateReader
from src.annotator import FFmpegPublisher, draw_detections
from src.backend_client import send_record
from src.cropper import crop_quality, crop_reject_reason, crop_sharpness, crop_vehicle
from src.detector_tracker import DetectorTracker
from src.stream_reader import probe_stream, read_frames
from src.vehicle_registry import VehicleRegistry
from src.worker import AnalysisWorker

log = logging.getLogger("main")


def parse_args():
    p = argparse.ArgumentParser(description="Vehicle detection - Stage 2 AI worker")
    p.add_argument("--source", default=config.VIDEO_SOURCE, help="video file or RTSP URL")
    p.add_argument("--headless", action="store_true", help="do not open a video window")
    p.add_argument("--send", action="store_true", help="POST records to the backend")
    p.add_argument("--publish", action="store_true", help="publish annotated video via FFmpeg")
    p.add_argument("--no-ingest", action="store_true",
                   help="do not POST new records to the ingest API")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Container stop (docker stop / compose down -> SIGTERM) should behave like
    # Ctrl+C, so the finally block below can drain the queue and flush every
    # record instead of the process dying mid-analysis. On Windows SIGTERM is
    # never delivered by the OS, but registering it is harmless.
    def _on_stop(_signum, _frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _on_stop)
    except (ValueError, OSError, AttributeError):
        pass                    # not the main thread / signal not supported

    # ── Priority-2: set torch thread count before any model loads ──────────
    import torch
    n_threads = config.TORCH_THREADS if config.TORCH_THREADS > 0 else torch.get_num_threads()
    if config.TORCH_THREADS > 0:
        torch.set_num_threads(config.TORCH_THREADS)
    log.info("Torch CPU threads: %d (was: %d)", torch.get_num_threads(), n_threads)

    # ── Reset the records file so this run starts fresh ────────────────────
    # ByteTrack track IDs restart at 1 on every run, so appending record after
    # record would repeat the same ID across runs. Resetting here (to an empty
    # JSON array) guarantees the file holds exactly one record per unique track
    # ID for this run, and stays valid JSON from the very first frame.
    if config.RESET_RECORDS_ON_START:
        records_path = Path(config.RECORDS_FILE)
        records_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            records_path.write_text("[]", encoding="utf-8")
            log.info("Records file reset for this run: %s", records_path)
        except OSError:
            log.exception("Could not reset the records file %s", records_path)

    # ── Clear last run's snapshots so the folder matches this run ──────────
    # Snapshots are named by track ID and ByteTrack IDs restart at 1 every run,
    # so old .jpg files would otherwise linger next to this run's images.
    if config.RESET_SNAPSHOTS_ON_START:
        snapshot_dir = Path(config.SNAPSHOT_DIR)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        removed = 0
        for old in snapshot_dir.glob("*.jpg"):
            try:
                old.unlink()
                removed += 1
            except OSError:
                log.exception("Could not delete the old snapshot %s", old)
        log.info("Cleared %d old snapshot(s) from %s", removed, snapshot_dir)

    # ── Live ingest: new records -> ingest API (Stage 3) ───────────────────
    # Runs as a background thread in THIS process, so one container runs the
    # pipeline and the uploader together (no second process / supervisor).
    # reset_state is tied to the records reset above: records.json restarts at
    # track_id 1 each run, so the uploader must forget the previous run's IDs or
    # it would treat the new records as already-sent and skip them.
    uploader = None
    if config.INGEST_ENABLED and not args.no_ingest:
        uploader = Uploader(records_file=config.RECORDS_FILE,
                            url=config.INGEST_URL,
                            api_key=config.INGEST_API_KEY,
                            state_file=config.INGEST_STATE_FILE,
                            interval=config.INGEST_POLL_SEC,
                            reset_state=config.RESET_RECORDS_ON_START)
        uploader.start()

    # API-first: har naya record pehle seedha ingest API par jata hai; uske baad
    # records.json backup ke taur par likha jata hai. Uploader ka background
    # sweep wo records uthata hai jo direct POST nahi ho paye (server down etc.).
    direct_send = uploader.send_now if uploader is not None else None

    show_window = config.SHOW_WINDOW and not args.headless
    send = config.SEND_TO_BACKEND or args.send
    publish = config.PUBLISH_ANNOTATED or args.publish

    detector = DetectorTracker(config.YOLO_WEIGHTS, config.VEHICLE_CLASSES,
                               conf=config.DETECT_CONF, imgsz=config.IMAGE_SIZE,
                               tracker=config.TRACKER, device=config.DEVICE,
                               agnostic_nms=config.AGNOSTIC_NMS)
    registry = VehicleRegistry(track_timeout=config.TRACK_TIMEOUT_FRAMES)
    plate_reader = PlateReader(config.PLATE_DETECTOR_WEIGHTS,
                               min_conf=config.PLATE_MIN_CONF, lang=config.PADDLE_LANG,
                               detector_conf=config.PLATE_DETECTOR_CONF)

    send_fn = None
    if send:
        send_fn = lambda record, jpg: send_record(
            record, jpg, config.BACKEND_URL,
            timeout=config.BACKEND_TIMEOUT_SEC, retries=config.BACKEND_RETRIES)

    worker = AnalysisWorker(plate_reader, send_fn,
                            camera_id=config.CAMERA_ID,
                            snapshot_dir=config.SNAPSHOT_DIR,
                            jpeg_quality=config.JPEG_QUALITY,
                            use_thread=config.USE_WORKER_THREAD,
                            queue_size=config.QUEUE_SIZE,
                            records_file=config.RECORDS_FILE,
                            plate_sharpness_min=config.PLATE_SHARPNESS_MIN,
                            direct_send=direct_send)

    publisher = None
    if publish:
        width, height, fps = probe_stream(args.source)
        publisher = FFmpegPublisher(config.ANNOTATED_RTSP, width, height, fps)

    frames = 0
    reject_area = 0          # new vehicles skipped: box too small
    reject_edge = 0          # new vehicles skipped: box touches the frame edge
    reject_shape = 0         # new vehicles skipped: box too narrow/short/misshapen
    reject_ids = set()       # unique ids skipped by the box filter
    queue_full = 0           # times the worker had no room (submit -> False)
    recovered = 0            # vehicles analysed late from their best crop
    try:
        for frame in read_frames(args.source,
                                 reconnect_attempts=config.RECONNECT_ATTEMPTS,
                                 loop=config.LOOP_VIDEO_FILE):         # step 6
            frames += 1

            detections = detector.run(frame)                       # steps 7 + 8

            for det in detections:
                track_id = det["track_id"]
                reason = crop_reject_reason(
                    det["box"], frame.shape, config.MIN_BOX_AREA, config.EDGE_MARGIN,
                    config.MIN_BOX_WIDTH, config.MIN_BOX_HEIGHT,
                    config.MIN_ASPECT_RATIO, config.MAX_ASPECT_RATIO)
                good = reason is None
                if not registry.is_new(track_id):                      # step 9
                    # already handed to the worker: keep a better crop so a plate
                    # missed earlier can be retried with it before the run ends
                    if good:
                        crop = crop_vehicle(frame, det["box"])
                        if crop is not None:
                            det["first_seen_frame"] = registry.first_seen(track_id)
                            det["first_seen_time"] = registry.first_seen_time(track_id)
                            sharp = crop_sharpness(crop, config.CROP_NORM_WIDTH)
                            score = crop_quality(
                                crop, det["box"], frame.shape, sharp,
                                config.CROP_NORM_WIDTH, config.SIZE_REF_AREA_RATIO,
                                config.SHARPNESS_REF, config.TRUNC_MARGIN)
                            registry.remember(track_id, det, crop, frames, True, score)
                    continue
                # First time this vehicle is seen: remember its first detection,
                # frame number and the local wall-clock instant (kept forever and
                # reused for this vehicle's record, in every recovery path).
                registry.mark_first_seen(track_id, frames, det, datetime.now())
                det["first_seen_frame"] = registry.first_seen(track_id)
                det["first_seen_time"] = registry.first_seen_time(track_id)
                crop = crop_vehicle(frame, det["box"])
                if crop is not None:
                    sharp = crop_sharpness(crop, config.CROP_NORM_WIDTH)
                    score = crop_quality(
                        crop, det["box"], frame.shape, sharp,
                        config.CROP_NORM_WIDTH, config.SIZE_REF_AREA_RATIO,
                        config.SHARPNESS_REF, config.TRUNC_MARGIN)
                    # remember the best crop even when the box is rejected, so
                    # the vehicle can be analysed late instead of being dropped
                    registry.observe(track_id, det, crop, frames, good, score, sharp,
                                     config.STABLE_IOU)
                    # also keep a few best-quality candidate crops (ranked by the
                    # same score, no extra work): the worker prefers one with a
                    # visible plate. Cheap, and only used at submit time.
                    registry.remember_candidate(track_id, crop, score,
                                                config.PLATE_CANDIDATE_FRAMES)
                if not good:
                    if reason == "area":
                        reject_area += 1
                    elif reason == "edge":
                        reject_edge += 1
                    else:
                        reject_shape += 1
                    reject_ids.add(track_id)
                    log.debug("track %s skipped (box %s)", track_id, reason)
                    continue
                # adaptive decision: submit once the crop has stopped improving
                # and the box has been stable for a few frames, or at the cap.
                # Never just because a fixed number of frames have passed.
                ready = registry.ready(track_id, config.MIN_OBSERVE_FRAMES,
                                       config.STABLE_FRAMES, config.SHARPNESS_MIN,
                                       config.QUALITY_CONVERGENCE_TOL)
                if not ready and registry.observed(track_id) < config.MAX_OBSERVE_FRAMES:
                    continue
                best_det, best_crop = registry.best_for(track_id)
                if best_crop is None:
                    continue
                # hand the worker the best-first candidate frames so it can
                # analyse one where the plate is visible (else the best overall)
                candidates = registry.candidate_crops(track_id)
                if worker.submit(best_det, best_crop, candidates):     # stage 2B
                    registry.mark_done(track_id)
                else:
                    queue_full += 1
                    log.debug("queue full, track %s will be retried", track_id)

            # recover vehicles whose track disappeared without being analysed
            for track_id in registry.expired(frames):
                det, crop = registry.best_for(track_id)
                if det is None or crop is None or crop.size == 0:
                    # no usable crop at all: the shutdown fallback pass still
                    # writes a detection-only record, so it is not dropped.
                    registry.mark_done(track_id)
                    continue
                det["first_seen_frame"] = registry.first_seen(track_id)
                det["first_seen_time"] = registry.first_seen_time(track_id)
                if worker.submit(det, crop):
                    registry.mark_done(track_id)
                    recovered += 1
                    log.info("Recovered track %s (never analysed) from its best crop.",
                             track_id)

            if frames % 100 == 0:
                log.info("progress: frames=%d seen=%d analysed=%d written=%d queue_full=%d",
                         frames, len(registry.seen_ids()), worker.processed,
                         worker.records_written, queue_full)

            if show_window or publisher is not None:                   # branch A
                annotated = draw_detections(frame, detections, registry.done_ids())
                if publisher is not None:
                    publisher.write(annotated)
                if show_window:
                    cv2.imshow("Stage 2 - vehicle detection (q to quit)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        log.info("Finishing the vehicles still in the queue...")
        worker.wait_until_idle()                    # drain everything submitted
        # nothing queued any more: analyse every vehicle that never got sent.
        # The queue is bounded, so wait for it to drain instead of overflowing it
        # (a rejected submit here would silently drop the vehicle for good).
        pending = registry.pending()
        log.info("Recovering %d vehicle(s) that were never analysed...", len(pending))
        for track_id, det, crop in pending:
            if det is None or crop is None or crop.size == 0:
                # no usable crop: the detection-only fallback pass below still
                # writes a record, so the vehicle is never dropped.
                registry.mark_done(track_id)
                continue
            det["first_seen_frame"] = registry.first_seen(track_id)
            det["first_seen_time"] = registry.first_seen_time(track_id)
            while not worker.submit(det, crop):
                worker.wait_until_idle()             # make room, then retry
            registry.mark_done(track_id)
            recovered += 1
            log.info("Recovered track %s (never analysed) from its best crop.", track_id)
        worker.wait_until_idle()
        # Late plate retry: vehicles whose plate was not read from their first
        # crop get one more chance with the best (largest) crop seen later.
        missing = worker.records_missing_plate()
        if missing:
            log.info("Retrying plate recognition for %d vehicle(s) with their best crop...",
                     len(missing))
            for track_id in missing:
                _det, crop = registry.best_for(track_id)
                if crop is None:
                    continue
                if worker.retry_plate(track_id, crop):
                    log.info("Late plate found for track %s.", track_id)
        # Every tracked vehicle must end up in the file. This runs after all the
        # worker results are collected, so a detection-only record can only be
        # added for a track that has none - it never duplicates an analysed one.
        missing_ids = [tid for tid in registry.all_ids()
                       if tid not in worker.analysed_ids()]
        if missing_ids:
            entries = []
            for track_id in missing_ids:
                det = registry.first_seen_det(track_id)
                if det is None:
                    continue
                _best_det, crop = registry.best_for(track_id)
                entries.append((det, crop))
            added = worker.write_fallback_records(entries)
            log.info("Wrote %d detection-only record(s) for vehicles never analysed.",
                     added)
        # every record is now buffered: write them in first_seen_frame order
        worker.write_records()
        worker.stop()
        if uploader is not None:
            # stop() sets the flag, so the thread flushes anything still unsent
            # before the process (container) exits.
            log.info("Flushing the ingest uploader...")
            uploader.stop()
        if publisher is not None:
            publisher.close()
        if show_window:
            cv2.destroyAllWindows()

        analysed = worker.processed
        written = worker.records_written
        seen = len(registry.all_ids())
        detection_only = written - analysed
        log.info("Summary: frames=%d, unique_tracks=%d, analysed=%d, written=%d",
                 frames, seen, analysed, written)
        log.info("  box skips: area_too_small=%d, touching_edge=%d, bad_shape=%d "
                 "(unique ids=%d); queue_full retries=%d; recovered_late=%d; "
                 "detection_only=%d",
                 reject_area, reject_edge, reject_shape, len(reject_ids),
                 queue_full, recovered, detection_only)
        if written != seen:
            log.warning("written (%d) != unique tracks (%d): some tracked vehicles "
                        "have no record", written, seen)
        log.info("Done. Frames: %d, vehicles analysed: %d", frames, len(registry))


if __name__ == "__main__":
    main()
