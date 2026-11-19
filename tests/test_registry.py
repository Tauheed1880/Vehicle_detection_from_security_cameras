from datetime import datetime

from src.vehicle_registry import VehicleRegistry


def test_new_then_done():
    reg = VehicleRegistry()
    assert reg.is_new(12)
    reg.mark_done(12)
    assert not reg.is_new(12)
    assert reg.is_new(13)
    assert len(reg) == 1


def test_not_marked_stays_new():
    reg = VehicleRegistry()
    assert reg.is_new(5)
    assert reg.is_new(5)      # checking does not mark it


def test_first_seen_frame_is_kept_forever():
    reg = VehicleRegistry()
    reg.mark_first_seen(7, 10)
    reg.mark_first_seen(7, 99)         # a later frame must not move it
    assert reg.first_seen(7) == 10


def test_mark_first_seen_keeps_first_det_and_time():
    reg = VehicleRegistry()
    det = {"track_id": 7, "box": [0, 0, 10, 10], "class_name": "car", "conf": 0.5}
    when = datetime(2026, 11, 18, 9, 30, 15)
    reg.mark_first_seen(7, 10, det, when)
    # a later call must not replace the first detection's det or timestamp
    reg.mark_first_seen(7, 20,
                        {"track_id": 7, "box": [1, 1, 2, 2],
                         "class_name": "bus", "conf": 0.9},
                        datetime(2026, 11, 18, 10, 0, 0))
    assert reg.first_seen_det(7) is det
    assert reg.first_seen_time(7) == when


def test_all_ids_lists_every_detected_track():
    reg = VehicleRegistry()
    reg.mark_first_seen(5, 1)
    reg.mark_first_seen(2, 3)
    assert reg.all_ids() == [5, 2]     # in first-detection order


def test_candidates_are_ranked_and_bounded():
    reg = VehicleRegistry()
    for crop, score in (("a", 0.2), ("b", 0.9), ("c", 0.5), ("d", 0.7)):
        reg.remember_candidate(1, crop, score, 2)
    assert reg.candidate_crops(1) == ["b", "d"]      # top-2, best quality first


def test_candidates_are_cleared_once_the_track_is_done():
    reg = VehicleRegistry()
    reg.remember_candidate(1, "a", 0.5, 3)
    reg.mark_done(1)
    assert reg.candidate_crops(1) == []


def test_no_candidates_kept_when_limit_is_zero():
    reg = VehicleRegistry()
    reg.remember_candidate(1, "a", 0.5, 0)
    assert reg.candidate_crops(1) == []
