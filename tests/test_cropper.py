import numpy as np
from src.cropper import clamp_box, crop_vehicle, is_good_box


def test_clamp_keeps_box_inside():
    assert clamp_box([-10, -5, 700, 500], (480, 640, 3)) == (0, 0, 640, 480)


def test_crop_is_a_copy_with_right_shape():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = crop_vehicle(frame, [10, 20, 60, 80])
    assert crop.shape == (60, 50, 3)
    crop[:] = 255
    assert frame.max() == 0          # the original frame is untouched


def test_empty_box_gives_none():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert crop_vehicle(frame, [50, 50, 50, 90]) is None
    assert crop_vehicle(frame, [300, 300, 400, 400]) is None


def test_good_box_rules():
    shape = (480, 640, 3)
    assert is_good_box([100, 100, 300, 300], shape, 12000, 5)
    assert not is_good_box([100, 100, 150, 150], shape, 12000, 5)    # too small
    assert not is_good_box([0, 100, 300, 300], shape, 12000, 5)      # touches left edge
    assert not is_good_box([100, 100, 640, 300], shape, 12000, 5)    # touches right edge
    assert is_good_box([0, 100, 300, 300], shape, 12000, 0)          # margin 0 disables
