"""
Steps 7 + 8 - YOLO26m vehicle detector + ByteTrack tracker.

Input : one BGR frame (NumPy array)
Output: a list of plain-Python dicts, one per tracked vehicle:
        {"track_id": 12, "box": [x1, y1, x2, y2], "class_name": "car", "conf": 0.91}

YOLO returns PyTorch tensors. This file converts them to normal Python ints and
floats, so every other file in the project never has to know about tensors.
"""
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def parse_boxes(boxes, names):
    """Convert an Ultralytics `Boxes` object into a list of plain dicts.

    Returns [] when there are no detections or the tracker has not given IDs yet.
    """
    if boxes is None or getattr(boxes, "id", None) is None:
        return []

    xyxy = boxes.xyxy.int().tolist()      # tensor of floats -> list of int lists
    ids = boxes.id.int().tolist()         # tensor -> list of int track ids
    classes = boxes.cls.int().tolist()
    confs = boxes.conf.tolist()

    detections = []
    for box, track_id, cls, conf in zip(xyxy, ids, classes, confs):
        detections.append({
            "track_id": int(track_id),
            "box": [int(v) for v in box],
            "class_name": names[int(cls)],
            "conf": float(conf),
        })
    return detections


class DetectorTracker:
    def __init__(self, weights, vehicle_classes, conf=0.35, imgsz=640,
                 tracker="bytetrack.yaml", device=None, agnostic_nms=False):
        from ultralytics import YOLO          # imported here so tests do not need it

        weights = str(weights)
        if not Path(weights).exists():
            # Keep the full path, never the bare filename: Ultralytics saves a
            # downloaded asset to exactly the path it is given, so a bare name
            # would land in the current working directory (project root) instead
            # of models/.
            Path(weights).parent.mkdir(parents=True, exist_ok=True)
            log.warning("%s not found. Ultralytics will download '%s' into %s.",
                        weights, Path(weights).name, Path(weights).parent)
        self.model = YOLO(weights)
        self.vehicle_classes = list(vehicle_classes)
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker
        self.device = device
        self.agnostic_nms = agnostic_nms

    def run(self, frame):
        kwargs = dict(
            persist=True,                  # keep track IDs between frames
            classes=self.vehicle_classes,
            conf=self.conf,
            imgsz=self.imgsz,
            tracker=self.tracker,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
        )
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model.track(frame, **kwargs)
        return parse_boxes(results[0].boxes, self.model.names)
