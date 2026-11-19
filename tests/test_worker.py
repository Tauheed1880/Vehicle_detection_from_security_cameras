"""Runs the whole Stage 2B flow with fake models (no downloads needed)."""
import json
from datetime import datetime

import cv2
import numpy as np

import config
import src.worker as worker_module
from src.worker import AnalysisWorker


class FakeClip:
    def classify(self, crop):
        return {"make": "Toyota", "model": "Corolla", "make_model_conf": 0.9,
                "type": "Sedan", "type_conf": 0.8,
                "colour": "white", "colour_conf": 0.7,
                "decoration": "none", "decoration_conf": 0.95}


class BrokenClip:
    def classify(self, crop):
        raise RuntimeError("CLIP exploded")


class FakePlateReader:
    def read(self, crop):
        return "ABC-123", 0.93, crop, "Punjab"


class BrokenPlateReader:
    def read(self, crop):
        raise RuntimeError("OCR exploded")


class StagedPlateReader:
    """No plate on the first read; a plate on the late retry."""

    def __init__(self):
        self.calls = 0

    def read(self, crop):
        self.calls += 1
        if self.calls == 1:
            return None, 0.0, None, None
        return "GX15-0GJ", 0.908, crop, "Punjab"


DET = {"track_id": 12, "box": [0, 0, 100, 80], "class_name": "car", "conf": 0.91}


def white_crop():
    img = np.zeros((80, 100, 3), dtype=np.uint8)
    img[:] = 255
    return img


def test_full_record(tmp_path):
    sent = []
    worker = AnalysisWorker(FakeClip(), FakePlateReader(),
                            lambda rec, jpg: sent.append((rec, jpg)),
                            camera_id="cam-01", snapshot_dir=tmp_path, use_thread=False)
    assert worker.submit(DET, white_crop())

    record, jpg = sent[0]
    assert record["track_id"] == 12
    assert record["type"] == "Sedan"              # CLIP body type
    assert record["detector_class"] == "car"      # raw YOLO class is kept
    assert (record["make"], record["model"]) == ("Toyota", "Corolla")
    assert record["colour"] == "white"
    assert record["decoration"] == "none"
    assert record["plate"] == "ABC-123"
    assert record["province"] == "Punjab"
    assert record["camera_id"] == "cam-01"
    assert jpg[:2] == b"\xff\xd8"                 # JPEG magic bytes
    assert (tmp_path / "12.jpg").exists()


def test_broken_plate_does_not_stop_the_record(tmp_path):
    worker = AnalysisWorker(FakeClip(), BrokenPlateReader(), None,
                            camera_id="cam-01", snapshot_dir=tmp_path, use_thread=False)
    record = worker.analyse(DET, white_crop())
    assert record["plate"] is None
    assert record["type"] == "Sedan"


def test_broken_clip_does_not_stop_the_record(tmp_path):
    worker = AnalysisWorker(BrokenClip(), FakePlateReader(), None,
                            camera_id="cam-01", snapshot_dir=tmp_path, use_thread=False)
    record = worker.analyse(DET, white_crop())
    assert record["make"] == "unknown"            # CLIP failed -> unknown
    assert record["plate"] == "ABC-123"           # plate still recorded


def test_threaded_worker_processes_jobs(tmp_path):
    worker = AnalysisWorker(FakeClip(), FakePlateReader(), None,
                            camera_id="cam-01", snapshot_dir=tmp_path, use_thread=True)
    for i in range(3):
        assert worker.submit({**DET, "track_id": i}, white_crop())
    worker.wait_until_idle()
    worker.stop()
    assert worker.processed == 3


def test_retry_overwrites_snapshot_and_sets_province(tmp_path):
    records = tmp_path / "records.json"
    reader = StagedPlateReader()
    worker = AnalysisWorker(FakeClip(), reader, None,
                            camera_id="cam-01", snapshot_dir=tmp_path,
                            use_thread=False, records_file=records)

    record = worker.analyse(DET, white_crop())
    assert record["plate"] is None                # nothing read from the first crop
    assert record["province"] is None
    assert worker.records_missing_plate() == [12]

    first = (tmp_path / "12.jpg").read_bytes()

    # a later, larger crop: the retry reads a plate from it
    big = np.zeros((40, 60, 3), dtype=np.uint8)
    big[:] = 200
    assert worker.retry_plate(12, big) is True

    assert record["plate"] == "GX15-0GJ"
    assert record["plate_conf"] == 0.908
    assert record["province"] == "Punjab"

    # the snapshot is replaced by the exact crop that produced the reading
    second = (tmp_path / "12.jpg").read_bytes()
    assert second != first
    saved = cv2.imread(str(tmp_path / "12.jpg"))
    assert saved.shape[:2] == (40, 60)

    # the file on disk reflects the retry too
    data = json.loads(records.read_text(encoding="utf-8"))
    assert data[0]["plate"] == "GX15-0GJ"
    assert data[0]["province"] == "Punjab"


def test_retry_does_not_touch_a_record_that_already_has_a_plate(tmp_path):
    records = tmp_path / "records.json"
    worker = AnalysisWorker(FakeClip(), FakePlateReader(), None,
                            camera_id="cam-01", snapshot_dir=tmp_path,
                            use_thread=False, records_file=records)
    record = worker.analyse(DET, white_crop())
    assert record["plate"] == "ABC-123"
    before = (tmp_path / "12.jpg").read_bytes()

    assert worker.retry_plate(12, white_crop()) is False
    assert record["plate"] == "ABC-123"
    assert (tmp_path / "12.jpg").read_bytes() == before


def test_write_records_writes_one_json_array_in_detection_order(tmp_path):
    records = tmp_path / "records.json"
    worker = AnalysisWorker(FakeClip(), FakePlateReader(), None,
                            camera_id="cam-01", snapshot_dir=tmp_path,
                            use_thread=False, records_file=records)
    worker.analyse({**DET, "track_id": 3, "first_seen_frame": 30}, white_crop())
    worker.analyse({**DET, "track_id": 1, "first_seen_frame": 10}, white_crop())
    worker.write_records()

    data = json.loads(records.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert [r["track_id"] for r in data] == [1, 3]   # sorted by first_seen_frame
    assert worker.records_written == 2


def test_record_is_written_as_soon_as_it_is_analysed(tmp_path):
    records = tmp_path / "records.json"
    worker = AnalysisWorker(FakeClip(), FakePlateReader(), None,
                            camera_id="cam-01", snapshot_dir=tmp_path,
                            use_thread=False, records_file=records)

    worker.analyse({**DET, "track_id": 7, "first_seen_frame": 5}, white_crop())

    # no write_records() / shutdown: the file must already hold the record
    data = json.loads(records.read_text(encoding="utf-8"))
    assert [r["track_id"] for r in data] == [7]


def test_send_also_writes_the_local_file(tmp_path):
    records = tmp_path / "records.json"
    sent = []
    worker = AnalysisWorker(FakeClip(), FakePlateReader(),
                            lambda rec, jpg: sent.append(rec["track_id"]),
                            camera_id="cam-01", snapshot_dir=tmp_path,
                            use_thread=False, records_file=records)

    worker.analyse({**DET, "track_id": 4, "first_seen_frame": 1}, white_crop())

    assert sent == [4]                                    # still POSTed to the backend
    data = json.loads(records.read_text(encoding="utf-8"))
    assert [r["track_id"] for r in data] == [4]           # ...and kept locally


def test_record_has_split_date_and_time(tmp_path):
    worker = AnalysisWorker(FakeClip(), FakePlateReader(), None,
                            camera_id="cam-01", snapshot_dir=tmp_path, use_thread=False)
    when = datetime(2026, 11, 18, 14, 5, 9)
    record = worker.analyse({**DET, "first_seen_frame": 3,
                             "first_seen_time": when}, white_crop())
    assert record["date"] == "18/11/2026"                 # DD/MM/YYYY
    assert record["time"] == "14:05:09"                   # HH:MM:SS, 24-hour
    assert "timestamp_source" not in record               # no extra attributes


def test_analysed_record_is_not_written_twice(tmp_path):
    records = tmp_path / "records.json"
    worker = AnalysisWorker(FakeClip(), FakePlateReader(), None,
                            camera_id="cam-01", snapshot_dir=tmp_path,
                            use_thread=False, records_file=records)
    worker.analyse({**DET, "track_id": 9, "first_seen_frame": 1}, white_crop())
    worker.analyse({**DET, "track_id": 9, "first_seen_frame": 1}, white_crop())
    data = json.loads(records.read_text(encoding="utf-8"))
    assert [r["track_id"] for r in data] == [9]           # one record per track ID


def test_analysed_ids_reports_tracks_with_records(tmp_path):
    records = tmp_path / "records.json"
    worker = AnalysisWorker(FakeClip(), FakePlateReader(), None,
                            camera_id="cam-01", snapshot_dir=tmp_path,
                            use_thread=False, records_file=records)
    worker.analyse({**DET, "track_id": 4, "first_seen_frame": 1}, white_crop())
    assert worker.analysed_ids() == {4}


def test_fallback_records_fill_in_missing_tracks(tmp_path):
    records = tmp_path / "records.json"
    worker = AnalysisWorker(FakeClip(), FakePlateReader(), None,
                            camera_id="cam-01", snapshot_dir=tmp_path,
                            use_thread=False, records_file=records)
    worker.analyse({**DET, "track_id": 12, "first_seen_frame": 10}, white_crop())

    # track 3 was tracked but never analysed -> a detection-only record
    det = {"track_id": 3, "box": [0, 0, 20, 20], "class_name": "bus", "conf": 0.7,
           "first_seen_frame": 20,
           "first_seen_time": datetime(2026, 11, 18, 8, 1, 2)}
    assert worker.write_fallback_records([(det, white_crop())]) == 1

    data = json.loads(records.read_text(encoding="utf-8"))
    by_id = {r["track_id"]: r for r in data}
    assert set(by_id) == {12, 3}                          # both, each once
    fb = by_id[3]
    assert fb["type"] == "unknown" and fb["make"] == "unknown"
    assert fb["plate"] is None and fb["province"] is None
    assert fb["detector_class"] == "bus"
    assert fb["date"] == "18/11/2026" and fb["time"] == "08:01:02"
    assert fb["snapshot_file"] == "3.jpg"
    assert (tmp_path / "3.jpg").exists()
    assert by_id[12]["type"] == "Sedan"                   # analysed one untouched


def test_fallback_skips_a_track_that_was_analysed(tmp_path):
    records = tmp_path / "records.json"
    worker = AnalysisWorker(FakeClip(), FakePlateReader(), None,
                            camera_id="cam-01", snapshot_dir=tmp_path,
                            use_thread=False, records_file=records)
    worker.analyse({**DET, "track_id": 12, "first_seen_frame": 10}, white_crop())

    added = worker.write_fallback_records(
        [({**DET, "track_id": 12, "first_seen_frame": 10}, white_crop())])

    assert added == 0
    data = json.loads(records.read_text(encoding="utf-8"))
    assert [r["track_id"] for r in data] == [12]
    assert data[0]["type"] == "Sedan"                     # analysed record kept


def test_fallback_without_crop_has_no_snapshot(tmp_path):
    records = tmp_path / "records.json"
    worker = AnalysisWorker(FakeClip(), FakePlateReader(), None,
                            camera_id="cam-01", snapshot_dir=tmp_path,
                            use_thread=False, records_file=records)
    det = {"track_id": 5, "box": [0, 0, 20, 20], "class_name": "car", "conf": 0.6,
           "first_seen_frame": 2}
    assert worker.write_fallback_records([(det, None)]) == 1
    data = json.loads(records.read_text(encoding="utf-8"))
    assert data[0]["snapshot_file"] is None


# ------------------------------------------------- plate-first frame selection
def _good_plate():
    """High-frequency region: high Laplacian variance, i.e. clearly in focus."""
    return np.random.RandomState(0).randint(0, 256, (32, 200, 3), dtype=np.uint8)


def _poor_plate():
    """Flat region: zero Laplacian variance, i.e. clearly out of focus."""
    return np.full((32, 200, 3), 128, dtype=np.uint8)


def _plate(tag):
    """A distinct flat region, for tests that stub the sharpness measure."""
    return np.full((32, 200, 3), tag, dtype=np.uint8)


def _ids(seq):
    return [id(item) for item in seq]


class CandidatePlateReader:
    """Plate reader whose candidate crops show a prepared plate region.

    `plates` maps id(vehicle_crop) -> plate region (missing/None = no plate in
    that candidate). find_plate returns that exact region so a test can prove it
    is reused for OCR. It records every detection (`located`) and every OCR call
    (`ocr_crops`).
    """

    def __init__(self, plates=None):
        self.plates = dict(plates or {})
        self.located = []          # crops passed to find_plate (detections)
        self.ocr_crops = []        # plate regions passed to read_with_crop
        self.reads = 0             # times the full read() path was used

    def find_plate(self, crop):
        self.located.append(crop)
        return self.plates.get(id(crop))

    def read(self, crop):
        self.reads += 1
        return None, 0.0, None, None

    def read_with_crop(self, plate_crop):
        self.ocr_crops.append(plate_crop)
        if plate_crop is None:
            return None, 0.0, None, None
        return "ABC-123", 0.93, plate_crop, "Punjab"


class FailingLocateReader:
    """find_plate always fails, but read() still works."""

    def __init__(self):
        self.reads = 0

    def find_plate(self, crop):
        raise RuntimeError("plate detector died")

    def read(self, crop):
        self.reads += 1
        return "LEA-1234", 0.9, None, "Punjab"


def _crop(value):
    return np.full((20, 30, 3), value, dtype=np.uint8)


def _worker(reader, tmp_path):
    return AnalysisWorker(FakeClip(), reader, None,
                          camera_id="cam-01", snapshot_dir=tmp_path,
                          use_thread=False,
                          plate_sharpness_min=config.PLATE_SHARPNESS_MIN)


def test_first_candidate_without_plate_uses_the_next_good_plate(tmp_path):
    first, second = _crop(10), _crop(20)
    good = _good_plate()
    reader = CandidatePlateReader({id(second): good})
    worker = _worker(reader, tmp_path)

    record = worker.analyse(DET, first, [first, second])

    assert _ids(reader.located) == _ids([first, second])   # every candidate checked
    assert len(reader.ocr_crops) == 1
    assert reader.ocr_crops[0] is good                     # the second candidate's region
    assert reader.reads == 0                               # detection not repeated
    assert record["plate"] == "ABC-123"


def test_poor_plate_is_rejected_in_favour_of_a_later_good_one(tmp_path):
    first, second = _crop(10), _crop(20)
    poor, good = _poor_plate(), _good_plate()
    reader = CandidatePlateReader({id(first): poor, id(second): good})
    worker = _worker(reader, tmp_path)

    worker.analyse(DET, first, [first, second])

    assert _ids(reader.located) == _ids([first, second])   # both evaluated
    assert len(reader.ocr_crops) == 1
    assert reader.ocr_crops[0] is good                     # poor plate rejected
    assert reader.reads == 0


def test_best_plate_among_several_passing_is_selected(tmp_path, monkeypatch):
    first, second, third = _crop(10), _crop(20), _crop(30)
    p_first, p_second, p_third = _plate(1), _plate(2), _plate(3)
    reader = CandidatePlateReader({id(first): p_first,
                                   id(second): p_second,
                                   id(third): p_third})
    sharpness = {id(p_first): 40.0, id(p_second): 90.0, id(p_third): 60.0}
    monkeypatch.setattr(worker_module, "crop_sharpness",
                        lambda plate: sharpness[id(plate)])
    worker = _worker(reader, tmp_path)

    worker.analyse(DET, first, [first, second, third])

    assert _ids(reader.located) == _ids([first, second, third])   # all evaluated
    assert len(reader.ocr_crops) == 1
    assert reader.ocr_crops[0] is p_second                 # sharpest passing plate
    assert reader.reads == 0


def test_no_visible_plate_falls_back_to_best_overall(tmp_path):
    first, second = _crop(10), _crop(20)
    reader = CandidatePlateReader()                       # no candidate has a plate
    worker = _worker(reader, tmp_path)

    record = worker.analyse(DET, first, [first, second])

    assert _ids(reader.located) == _ids([first, second])
    assert len(reader.ocr_crops) == 1
    assert reader.ocr_crops[0] is None                    # best overall had no plate region
    assert reader.reads == 0
    assert record["plate"] is None


def test_all_plates_below_threshold_falls_back_to_best_overall(tmp_path):
    first, second = _crop(10), _crop(20)
    poor_first, poor_second = _poor_plate(), _poor_plate()
    reader = CandidatePlateReader({id(first): poor_first, id(second): poor_second})
    worker = _worker(reader, tmp_path)

    chosen, plate_crop, located = worker._pick_crop(first, [first, second])

    assert chosen is first                                # best overall vehicle crop
    assert plate_crop is poor_first                       # its region is still OCR'd
    assert located is True
    assert reader.reads == 0                              # never discarded, never re-detected


def test_plate_detection_runs_once_per_candidate(tmp_path):
    first, second, third = _crop(10), _crop(20), _crop(30)
    reader = CandidatePlateReader({id(third): _good_plate()})
    worker = _worker(reader, tmp_path)

    worker.analyse(DET, first, [first, second, third])

    assert _ids(reader.located) == _ids([first, second, third])   # exactly one each
    assert reader.reads == 0                                      # and no re-detection


def test_selected_plate_region_is_reused_for_ocr(tmp_path):
    first, second = _crop(10), _crop(20)
    good = _good_plate()
    reader = CandidatePlateReader({id(second): good})
    worker = _worker(reader, tmp_path)

    worker.analyse(DET, first, [first, second])

    assert len(reader.ocr_crops) == 1
    assert reader.ocr_crops[0] is good                    # the very object find_plate returned
    assert reader.reads == 0


def test_without_candidates_the_reader_locates_the_plate_itself(tmp_path):
    crop = _crop(10)
    reader = CandidatePlateReader()
    worker = _worker(reader, tmp_path)

    worker.analyse(DET, crop, None)

    assert reader.located == []               # no candidate scan
    assert reader.reads == 1                  # old single-crop read() path
    assert reader.ocr_crops == []


def test_plate_location_failure_falls_back_to_read(tmp_path):
    crop = _crop(10)
    reader = FailingLocateReader()
    worker = _worker(reader, tmp_path)

    record = worker.analyse(DET, crop, [crop])

    assert reader.reads == 1                  # detection left to the reader
    assert record["plate"] == "LEA-1234"      # plate still read


def test_reader_without_find_plate_uses_read(tmp_path):
    crop = _crop(10)
    worker = _worker(FakePlateReader(), tmp_path)

    record = worker.analyse(DET, crop, [crop])

    assert record["plate"] == "ABC-123"       # old behaviour preserved


def test_pick_crop_reports_whether_detection_ran(tmp_path):
    crop = _crop(10)
    worker = _worker(CandidatePlateReader(), tmp_path)

    chosen, plate_crop, located = worker._pick_crop(crop, [])
    assert chosen is crop and plate_crop is None and located is False

    chosen, plate_crop, located = worker._pick_crop(crop, [crop])
    assert chosen is crop and plate_crop is None and located is True
