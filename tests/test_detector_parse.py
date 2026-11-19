from src.detector_tracker import parse_boxes


class FakeTensor:
    """Mimics the part of a PyTorch tensor that parse_boxes uses."""
    def __init__(self, data):
        self.data = data

    def int(self):
        def to_int(x):
            return [to_int(i) for i in x] if isinstance(x, list) else int(x)
        return FakeTensor(to_int(self.data))

    def tolist(self):
        return self.data


class FakeBoxes:
    def __init__(self, ids):
        self.xyxy = FakeTensor([[10.7, 20.2, 110.9, 220.4], [5.0, 6.0, 50.0, 60.0]])
        self.id = None if ids is None else FakeTensor(ids)
        self.cls = FakeTensor([2.0, 7.0])
        self.conf = FakeTensor([0.91, 0.55])


NAMES = {2: "car", 7: "truck"}


def test_tensors_become_plain_python():
    dets = parse_boxes(FakeBoxes([12.0, 13.0]), NAMES)
    assert dets[0] == {"track_id": 12, "box": [10, 20, 110, 220],
                       "class_name": "car", "conf": 0.91}
    assert dets[1]["class_name"] == "truck"
    assert type(dets[0]["track_id"]) is int
    assert all(type(v) is int for v in dets[0]["box"])


def test_no_ids_means_no_detections():
    assert parse_boxes(FakeBoxes(None), NAMES) == []
    assert parse_boxes(None, NAMES) == []
