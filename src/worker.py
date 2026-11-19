"""
Stage 2B - runs the heavy analysis once per new vehicle.

crop -> [ CLIP (10+11: make, model, type, colour, decorations) | plate (12) ]
      in parallel -> build record (13) -> save snapshot -> send (14)

With use_thread=True the work happens in a background thread, so slow models never
freeze the video loop.
"""
import json
import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from src.analyzers.clip_classifier import empty_result
from src.cropper import crop_sharpness
from src.record_builder import build_minimal_record, build_record, save_snapshot

log = logging.getLogger(__name__)


class AnalysisWorker:
    def __init__(self, clip_classifier, plate_reader, send_fn, camera_id, snapshot_dir,
                 jpeg_quality=90, use_thread=True, queue_size=16, records_file=None,
                 plate_sharpness_min=None, direct_send=None):
        self.clip_classifier = clip_classifier
        self.plate_reader = plate_reader
        self.send_fn = send_fn              # callable(record, jpg_bytes) or None
        self.direct_send = direct_send      # callable(record) -> API-first ingest, or None
        self.camera_id = camera_id
        self.snapshot_dir = snapshot_dir
        self.jpeg_quality = jpeg_quality
        self.use_thread = use_thread
        self.records_file = Path(records_file) if records_file else None
        self.plate_sharpness_min = plate_sharpness_min  # None = no minimum (accept any plate)
        self.records_written = 0
        self._records = []                  # buffered records, flushed in order at exit
        self._by_id = {}                    # track_id -> buffered record (for updates)
        self._lock = threading.Lock()       # guards the record buffer + counter
        self.last_record = None
        self.processed = 0

        self._pool = ThreadPoolExecutor(max_workers=2)   # CLIP + plate in parallel
        self._queue = queue.Queue(maxsize=queue_size)
        self._thread = None
        if use_thread:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    # ---------------------------------------------------------------- analysis
    @staticmethod
    def _safe(name, fn, default):
        try:
            return fn()
        except Exception:
            log.exception("%s failed, using default value", name)
            return default

    def _locate_plate(self, find_plate, crop):
        """Run ONE plate detection on `crop`; returns (plate_crop, checked).

        `checked` is False only when the detection could not run (no detector, or
        an error), so the caller can let the reader locate the plate itself - the
        old behaviour. When `checked` is True the detection did run, and
        `plate_crop` is the located region or None when the crop has no plate.
        """
        try:
            return find_plate(crop), True
        except Exception:
            log.exception("Plate location failed")
            return None, False

    def _pick_crop(self, crop, candidates):
        """Choose the crop to analyse, reusing its plate detection for OCR.

        `candidates` is best-quality first (see VehicleRegistry.candidate_crops).
        EVERY candidate in the bounded set is evaluated: the plate region is
        located once per candidate, and a candidate qualifies as the plate
        candidate only when its region is at least `plate_sharpness_min` sharp.
        The sharpest qualifying plate wins. When no candidate qualifies - no
        visible plate, or every plate below the threshold - the best overall
        vehicle crop (candidates[0]) is used, so a missing or unreadable plate
        never discards the vehicle.

        Plate quality is the existing image-clarity measure (crop_sharpness -
        variance of the Laplacian on a width-normalised crop), NOT the plate
        detector's confidence.

        Returns (crop, plate_crop, located). `located` is True when detection
        already ran for the chosen crop, so `plate_crop` (possibly None = no
        plate found) must be OCR'd directly; it is False when detection was not
        possible and the reader should locate the plate itself.
        """
        if not candidates:
            return crop, None, False
        find_plate = getattr(self.plate_reader, "find_plate", None)
        if find_plate is None:
            return candidates[0], None, False

        best_plate = None            # (sharpness, crop, plate_crop) of the best passing candidate
        overall = None               # (crop, plate_crop, checked) of candidates[0]
        for candidate in candidates:
            plate_crop, checked = self._locate_plate(find_plate, candidate)
            if overall is None:
                overall = (candidate, plate_crop, checked)
            if not checked or plate_crop is None:
                continue
            sharp = crop_sharpness(plate_crop)
            if self.plate_sharpness_min is not None and sharp < self.plate_sharpness_min:
                continue
            if best_plate is None or sharp > best_plate[0]:
                best_plate = (sharp, candidate, plate_crop)

        if best_plate is not None:
            return best_plate[1], best_plate[2], True
        return overall

    def _read_plate(self, crop, plate_crop, located):
        """OCR the chosen crop, reusing an already-located plate region.

        When `located` is True, plate detection already ran for this crop, so
        only OCR runs on the pre-computed `plate_crop` (None = no plate found)
        and the detector is NOT called again. Otherwise the reader locates the
        plate itself (`read`), exactly as before.
        """
        read_with_crop = getattr(self.plate_reader, "read_with_crop", None)
        if located and read_with_crop is not None:
            return read_with_crop(plate_crop)
        return self.plate_reader.read(crop)

    def analyse(self, det, crop, candidates=None):
        crop, plate_crop, located = self._pick_crop(crop, candidates)

        f_clip = self._pool.submit(self._safe, "clip",
                                   lambda: self.clip_classifier.classify(crop),
                                   empty_result())
        f_plate = self._pool.submit(self._safe, "plate",
                                    lambda: self._read_plate(crop, plate_crop, located),
                                    (None, 0.0, None, None))

        clip = f_clip.result()
        plate, plate_conf, _plate_crop, province = f_plate.result()

        results = {
            "make": clip["make"], "model": clip["model"],
            "make_model_conf": clip["make_model_conf"],
            "type": clip["type"], "type_conf": clip["type_conf"],
            "colour": clip["colour"], "colour_conf": clip["colour_conf"],
            "decoration": clip["decoration"], "decoration_conf": clip["decoration_conf"],
            "plate": plate, "plate_conf": plate_conf, "province": province,
        }
        record, jpg = build_record(det, crop, results, self.camera_id, self.jpeg_quality)
        try:
            save_snapshot(jpg, det["track_id"], self.snapshot_dir)
        except Exception:
            log.exception("Could not save the snapshot for track %s", det["track_id"])

        if self.send_fn is not None:
            try:
                self.send_fn(record, jpg)
            except Exception:
                log.exception("send_fn failed for track %s", det["track_id"])
        # API-first: record pehle seedha ingest API par jata hai, uske baad
        # records.json me backup ke taur par likha jata hai. Direct POST fail ho
        # to record file me rehta hai aur Uploader thread use retry karta hai.
        if self.direct_send is not None:
            try:
                self.direct_send(record)
            except Exception:
                log.exception("direct ingest failed for track %s", det["track_id"])
        if self.records_file is not None:
            self._append_record(record)
        elif self.send_fn is None:
            log.info("RECORD %s", json.dumps(record))
        self.last_record = record
        self.processed += 1
        return record

    def _append_record(self, record):
        """Remember one finished record and write it to the JSON file.

        Records finish out of order, so the whole buffer is rewritten oldest
        first after every record. That keeps the file current at all times and
        means nothing is lost if the run is stopped before it ends. Storage is
        keyed by track ID, so a track can never produce two entries.
        """
        if self.records_file is None:
            return
        with self._lock:
            track_id = record["track_id"]
            existing = self._by_id.get(track_id)
            if existing is not None:
                existing.update(record)          # one record per track ID
            else:
                self._records.append(record)
                self._by_id[track_id] = record
            self._flush_records()

    def _flush_records(self):
        """Write the whole buffer to the JSON file, oldest first.

        One JSON array of record objects, sorted by first_seen_frame so the file
        follows detection order, not the order analysis happened to finish. The
        write goes to a temp file and is then os.replace'd over the target, so a
        crash mid-write can never leave a truncated file. Caller holds _lock.
        """
        if self.records_file is None:
            return
        records = sorted(self._records,
                         key=lambda r: r.get("first_seen_frame") or 0)
        if not records:
            return
        tmp = self.records_file.with_name(self.records_file.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(records, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.records_file)
            self.records_written = len(records)
        except Exception:
            log.exception("Could not write the records to %s", self.records_file)

    def write_records(self):
        """Final write of every buffered record, oldest first.

        Records are already written as they finish, so this normally just
        rewrites the same array at shutdown.
        """
        if self.records_file is None:
            return
        with self._lock:
            self._flush_records()

    def analysed_ids(self):
        """Track IDs that already have a buffered (analysed) record."""
        with self._lock:
            return set(self._by_id)

    def write_fallback_records(self, entries):
        """Store a detection-only record for each tracked vehicle still missing.

        `entries` yields (det, crop) for every track ID seen this run. A track
        that already has a record is skipped, so a vehicle is never stored
        twice. The added record uses the normal schema (attributes left
        "unknown", no plate/province) and its snapshot is the best crop seen,
        if any. Returns the number of records added.
        """
        if self.records_file is None:
            return 0
        added = 0
        with self._lock:
            for det, crop in entries:
                track_id = det["track_id"]
                if track_id in self._by_id:
                    continue                     # already analysed: keep it
                record, jpg = build_minimal_record(det, crop, self.camera_id,
                                                   self.jpeg_quality)
                if jpg is not None:
                    try:
                        save_snapshot(jpg, track_id, self.snapshot_dir)
                    except Exception:
                        log.exception("Could not save the fallback snapshot for track %s",
                                      track_id)
                self._records.append(record)
                self._by_id[track_id] = record
                added += 1
            if added:
                self._flush_records()
        return added

    # --------------------------------------------------- late plate retry
    def records_missing_plate(self):
        """Track IDs of buffered records that still have no plate text."""
        if self.records_file is None:
            return []
        with self._lock:
            return [tid for tid, rec in self._by_id.items() if not rec.get("plate")]

    def retry_plate(self, track_id, crop):
        """Re-run plate detection ONLY (no CLIP) on a later, better crop.

        If a plate is found and the buffered record still has none, the record
        is updated in place AND the snapshot is overwritten with this exact
        crop, so the saved image always shows the reading that was stored.
        Returns True when the record was updated.
        """
        if self.records_file is None or crop is None or crop.size == 0:
            return False
        try:
            plate, plate_conf, _plate_crop, province = self.plate_reader.read(crop)
        except Exception:
            log.exception("Plate retry failed for track %s", track_id)
            return False
        if not plate:
            return False
        with self._lock:
            record = self._by_id.get(track_id)
            if record is None or record.get("plate"):
                return False
            record["plate"] = plate
            record["plate_conf"] = round(plate_conf, 3)
            record["province"] = province
            self._flush_records()
        # The snapshot must be the crop that produced this reading, so replace
        # the earlier one (the image the record already points at).
        try:
            ok, buffer = cv2.imencode(
                ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)])
            if ok:
                save_snapshot(buffer.tobytes(), track_id, self.snapshot_dir)
            else:
                log.warning("Could not encode the retry snapshot for track %s", track_id)
        except Exception:
            log.exception("Could not refresh the snapshot for track %s", track_id)
        return True

    # ------------------------------------------------------------------ queue
    def submit(self, det, crop, candidates=None):
        """Hand a new vehicle to the worker. Returns False if the queue is full
        (the main loop then simply tries again on the next frame).

        `candidates` is the optional best-first list of candidate crops used for
        plate-first frame selection inside the worker (None keeps the old
        single-crop behaviour).
        """
        if not self.use_thread:
            self.analyse(det, crop, candidates)
            return True
        try:
            self._queue.put_nowait((det, crop, candidates))
            return True
        except queue.Full:
            return False

    def _loop(self):
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                self.analyse(*job)
            except Exception:
                log.exception("Analysis crashed for a vehicle")
            finally:
                self._queue.task_done()

    def wait_until_idle(self):
        if self.use_thread:
            self._queue.join()

    def stop(self):
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=30)
        self._pool.shutdown(wait=False)
