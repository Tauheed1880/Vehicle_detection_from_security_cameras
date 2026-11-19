"""
Crop a vehicle out of the frame (first part of step 13) and decide whether the
box is good enough to analyse.
"""
import cv2


def clamp_box(box, shape):
    """Force a box [x1, y1, x2, y2] to stay inside the frame."""
    height, width = shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def crop_vehicle(frame, box):
    """Return a COPY of the vehicle region (BGR NumPy array), or None if empty.

    A copy is important: the worker thread uses it later, after the main loop
    has already moved on and replaced `frame`.
    """
    x1, y1, x2, y2 = clamp_box(box, frame.shape)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def crop_reject_reason(box, shape, min_area=0, edge_margin=-1,
                       min_width=0, min_height=0, min_ratio=0.0, max_ratio=0.0):
    """Why a box is not usable: 'area', 'small', 'shape', 'edge', or None.

    * 'area'  - fewer pixels than min_area (too blurry to analyse)
    * 'small' - narrower or shorter than the minimum side (cut-off sliver)
    * 'shape' - width/height outside the allowed band (incomplete box)
    * 'edge'  - box touches the frame border (vehicle is cut off)

    Pass edge_margin < 0 to waive the edge check (recovery of cut-off vehicles);
    `shape` may then be None.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return "small"
    if w * h < min_area:
        return "area"
    if w < min_width or h < min_height:
        return "small"
    ratio = w / h
    if (min_ratio > 0.0 and ratio < min_ratio) or (max_ratio > 0.0 and ratio > max_ratio):
        return "shape"
    if edge_margin >= 0 and shape is not None:
        height, width = shape[:2]
        if (x1 < edge_margin or y1 < edge_margin
                or x2 > width - edge_margin or y2 > height - edge_margin):
            return "edge"
    return None


def is_good_box(box, shape, min_area, edge_margin, min_width=0, min_height=0,
                min_ratio=0.0, max_ratio=0.0):
    """True if the box is big enough, well shaped and not cut off by the edge."""
    return crop_reject_reason(box, shape, min_area, edge_margin,
                              min_width, min_height, min_ratio, max_ratio) is None


def box_iou(a, b):
    """Intersection-over-union of two [x1, y1, x2, y2] boxes (0.0 .. 1.0)."""
    ax1, ay1, ax2, ay2 = (int(v) for v in a)
    bx1, by1, bx2, by2 = (int(v) for v in b)
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def crop_sharpness(crop, norm_width=128):
    """Variance of the Laplacian of a crop: higher means sharper.

    The crop is resized to a fixed width first, so the value does not depend on
    how close the vehicle is to the camera. Returns 0.0 for an empty crop.
    """
    if crop is None or crop.size == 0:
        return 0.0
    height, width = crop.shape[:2]
    if width <= 0 or height <= 0:
        return 0.0
    scale = float(norm_width) / width
    small = cv2.resize(crop, (norm_width, max(1, int(round(height * scale)))),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_quality(crop, box, shape, sharp=None, norm_width=128, size_ref=0.02,
                 sharp_ref=60.0, trunc_margin=8):
    """Score a crop for ranking and for the adaptive submit decision.

    Combines size, aspect ratio, sharpness and distance from the frame border, so
    the largest, clearest and most complete crop wins. `sharp` may be passed in
    when the caller already measured it, to avoid doing the work twice.
    """
    height, width = shape[:2]
    frame_area = float(height * width) or 1.0
    x1, y1, x2, y2 = (int(v) for v in box)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    if w == 0 or h == 0:
        return 0.0

    size_factor = 1.0
    if size_ref > 0:
        size_factor = min(1.0, ((w * h) / frame_area) / size_ref)

    ratio = w / h
    if 1.0 <= ratio <= 2.5:
        aspect_factor = 1.0
    elif ratio < 1.0:
        aspect_factor = ratio          # taller than wide -> decays towards 0
    else:
        aspect_factor = 2.5 / ratio    # much wider than tall -> decays

    if sharp is None:
        sharp = crop_sharpness(crop, norm_width)
    sharpness_factor = min(1.0, sharp / sharp_ref) if sharp_ref > 0 else 1.0

    # how far the box sits from the nearest frame border (0 = touches the edge)
    border = min(x1, y1, width - x2, height - y2)
    truncation_factor = 1.0
    if trunc_margin > 0:
        truncation_factor = min(1.0, max(0.0, border) / trunc_margin)

    return size_factor * aspect_factor * sharpness_factor * truncation_factor
