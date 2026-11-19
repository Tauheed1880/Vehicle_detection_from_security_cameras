
"""
Step 12 - Number plate: plate detector -> crop -> PaddleOCR.

Input : vehicle crop (BGR NumPy array)
Output: (plate_text, ocr_confidence, plate_crop, province)

The plate detector is the ONLY thing that locates the plate region - there is no
bottom-half fallback, so OCR can never read text outside a detected plate. A
reading is kept even when it holds fewer characters than a full plate (partial
plates are useful for investigations); OCR confidence is stored, never used to
reject. Province is only reported when its name is visibly printed inside the
detected plate region - it is never inferred from the registration characters.
"""

import logging
import re
from pathlib import Path

import cv2

log = logging.getLogger(__name__)


# -----------------------------------------------------------
# Text normalization
# -----------------------------------------------------------
def normalise_plate_text(text):
    """Convert OCR text to uppercase and clean unwanted characters."""
    t = re.sub(r"\s+", "-", text.upper().strip())
    t = re.sub(r"[^A-Z0-9-]", "", t)
    return t.strip("-")


# -----------------------------------------------------------
# Plausibility: is this OCR line even a candidate plate reading?
# -----------------------------------------------------------
# Checks A + B + C. None of them requires a complete plate format and none
# enforces a minimum length, so genuine partial readings ("VCV", "A", "GX15")
# survive. They only drop text that cannot be a plate fragment.
MIN_RAW_ALNUM_RATIO = 0.5      # A: letters/digits must dominate the RAW string

# C: decoration / bumper / make noise. Only rejected when the WHOLE reading
# equals one of these - a partial that merely contains these letters is kept.
NON_PLATE_TOKENS = {
    "TURBO", "DIESEL", "PETROL", "HYBRID", "AIRBAG", "SRS", "ABS", "4X4",
    "SUV", "SEDAN", "SPORT", "TAXI", "POLICE", "AMBULANCE", "AUTO", "MOTOR",
    "GLASS", "TOYOTA", "HONDA", "SUZUKI", "NISSAN", "KIA",
}


def is_plausible_plate(raw_text):
    """Length-agnostic sanity check on the RAW OCR string (before normalising).

    A - letters/digits must be the clear majority of the raw line, so a line
        dominated by punctuation/garbage (which normalising would silently
        strip into a short token) is rejected.
    B - a line that collapses to one repeated character (AAAA, 1111, ----) is
        noise, not a plate fragment.
    C - a line whose whole text is a known decoration/make word is rejected.
    """
    if not raw_text:
        return False

    alnum = sum(c.isalnum() for c in raw_text)
    if alnum < 1:
        return False
    if alnum / len(raw_text) < MIN_RAW_ALNUM_RATIO:        # A
        return False

    cleaned = normalise_plate_text(raw_text)
    if not cleaned:
        return False

    core = cleaned.replace("-", "")
    if len(core) >= 3 and len(set(core)) < 2:               # B
        return False
    if core in NON_PLATE_TOKENS:                            # C
        return False
    return True


# -----------------------------------------------------------
# Province: only what is visibly printed on the plate
# -----------------------------------------------------------
# Full names may appear as part of a line, so they are matched as substrings.
PROVINCE_NAMES = {
    "Punjab": ("PUNJAB",),
    "Sindh": ("SINDH",),
    "Khyber Pakhtunkhwa": ("KHYBER PAKHTUNKHWA", "KHYBERPAKHTUNKHWA", "KHYBER",
                            "PAKHTUNKHWA"),
    "Balochistan": ("BALOCHISTAN",),
    "Islamabad": ("ISLAMABAD", "ISLAMABAD CAPITAL TERRITORY"),
    "Gilgit-Baltistan": ("GILGIT BALTISTAN", "GILGITBALTISTAN", "GILGIT",
                         "BALTISTAN"),
    "Azad Jammu and Kashmir": ("AZAD JAMMU AND KASHMIR", "AZADJAMMUANDKASHMIR",
                               "AZAD KASHMIR", "KASHMIR"),
}

# Short abbreviations collide with ordinary registration characters, so they are
# matched ONLY when the line is exactly that abbreviation (whole-line equality).
PROVINCE_ABBREVIATIONS = {
    "Khyber Pakhtunkhwa": ("KPK",),
    "Islamabad": ("ICT",),
    "Gilgit-Baltistan": ("GB",),
    "Azad Jammu and Kashmir": ("AJK", "AZAD"),
}


def _letters_only(text):
    """Keep A-Z only, so spacing/hyphens never break a province match."""
    return re.sub(r"[^A-Z]", "", text.upper())


def match_province(text):
    """Canonical province if its FULL name appears in the text, else None."""
    t = _letters_only(text)
    if not t:
        return None
    for name, spellings in PROVINCE_NAMES.items():
        for spelling in spellings:
            if _letters_only(spelling) in t:
                return name
    return None


def match_province_abbreviation(text):
    """Canonical province if the line IS exactly an abbreviation, else None.

    Compared on the normalised text, so e.g. "GB" is a province but "GB 123"
    and "GX15-0GJ" (registration characters) are not.
    """
    norm = normalise_plate_text(text)
    for name, abbreviations in PROVINCE_ABBREVIATIONS.items():
        if norm and norm in abbreviations:
            return name
    return None


def province_of(text):
    """Canonical province named by one OCR line, or None."""
    return match_province(text) or match_province_abbreviation(text)


def detect_province(pairs):
    """First province name visibly present in the OCR lines, or None."""
    for text, _score in pairs:
        name = province_of(text)
        if name:
            return name
    return None


def is_province_line(text):
    """True when the whole line is a province name (so it is not a plate).

    A line that also carries registration characters (e.g. "PUNJAB 123") does
    NOT count, because it still holds a registration number.
    """
    norm = normalise_plate_text(text)
    if not norm:
        return False
    for spellings in PROVINCE_NAMES.values():
        for spelling in spellings:
            if norm == normalise_plate_text(spelling):
                return True
    for abbreviations in PROVINCE_ABBREVIATIONS.values():
        if norm in abbreviations:
            return True
    return False


# -----------------------------------------------------------
# PaddleOCR result parser
# -----------------------------------------------------------
def parse_paddle_result(result):
    """Convert PaddleOCR 2.x or 3.x output into [(text, score), ...]."""
    pairs = []

    if not result:
        return pairs

    for item in result:
        if item is None:
            continue

        # PaddleOCR 3.x
        data = item

        if not isinstance(data, dict):
            data = getattr(item, "json", None)
            if callable(data):
                data = data()

        if isinstance(data, dict):
            data = data.get("res", data)

            if "rec_texts" in data:
                for text, score in zip(
                    data["rec_texts"],
                    data.get("rec_scores", [])
                ):
                    pairs.append((str(text), float(score)))
                continue

        # PaddleOCR 2.x
        if isinstance(item, (list, tuple)):
            for line in item:
                try:
                    text, score = line[1]
                    pairs.append((str(text), float(score)))
                except (TypeError, ValueError, IndexError):
                    continue

    return pairs


# -----------------------------------------------------------
# Main class
# -----------------------------------------------------------
class PlateReader:

    def __init__(
        self,
        detector_weights=None,
        min_conf=0.50,
        lang="en",
        device=None,
        detector_conf=0.25
    ):
        self.detector_weights = (
            Path(detector_weights) if detector_weights else None
        )

        self.min_conf = min_conf
        self.lang = lang
        self.device = device
        self.detector_conf = detector_conf

        self._ocr = None
        self._detector = None
        self._detector_checked = False

    # -------------------------------------------------------
    # Load PaddleOCR
    # -------------------------------------------------------
    def _load_ocr(self):
        if self._ocr is not None:
            return

        from paddleocr import PaddleOCR

        try:
            # PaddleOCR 3.x
            self._ocr = PaddleOCR(
                lang=self.lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )

        except (TypeError, ValueError):
            # PaddleOCR 2.x
            self._ocr = PaddleOCR(
                lang=self.lang,
                use_angle_cls=False
            )

        log.info("PaddleOCR loaded.")

    # -------------------------------------------------------
    # Load YOLO plate detector
    # -------------------------------------------------------
    def _load_detector(self):
        if self._detector_checked:
            return

        self._detector_checked = True

        if self.detector_weights and self.detector_weights.exists():
            from ultralytics import YOLO

            self._detector = YOLO(str(self.detector_weights))

            log.info(
                "Plate detector loaded: %s",
                self.detector_weights
            )

        else:
            log.warning(
                "No plate detector at %s. No plate region can be located, so "
                "no plate text will be read.",
                self.detector_weights
            )

    # -------------------------------------------------------
    # Detect and crop number plate
    # -------------------------------------------------------
    def _find_plate_crop(self, crop):
        """Return the detected plate region, or None.

        Only the plate detector locates the region: there is deliberately no
        bottom-half fallback, so OCR never reads text outside a detected plate.
        """
        self._load_detector()

        if self._detector is None:
            return None

        kwargs = {
            "conf": self.detector_conf,
            "verbose": False
        }

        if self.device is not None:
            kwargs["device"] = self.device

        boxes = self._detector.predict(crop, **kwargs)[0].boxes

        if boxes is not None and len(boxes) > 0:

            best = int(boxes.conf.argmax())

            x1, y1, x2, y2 = (
                boxes.xyxy[best].int().tolist()
            )

            pad = 4
            h, w = crop.shape[:2]

            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            if x2 > x1 and y2 > y1:
                return crop[y1:y2, x1:x2].copy()

        return None

    # -------------------------------------------------------
    # Plate detection (locate the region, no OCR)
    # -------------------------------------------------------
    def find_plate(self, crop):
        """Return the detected plate region of `crop`, or None.

        Detection only - it reuses `_find_plate_crop`, so it needs no new model
        and runs no OCR and no plausibility test. The returned region is exactly
        what OCR should be run on, so a caller that checks several candidate
        crops can keep the region of the one it picks and hand it to
        `read_with_crop`, and the plate detector never runs twice on that crop.
        """
        if crop is None or crop.size == 0:
            return None
        return self._find_plate_crop(crop)

    # -------------------------------------------------------
    # Run PaddleOCR
    # -------------------------------------------------------
    def _run_ocr(self, image):

        if hasattr(self._ocr, "predict"):
            # PaddleOCR 3.x
            return self._ocr.predict(image)

        # PaddleOCR 2.x
        return self._ocr.ocr(image, cls=False)

    # -------------------------------------------------------
    # Main plate reading function
    # -------------------------------------------------------
    def read(self, crop):
        """Locate the plate in `crop` and read it.

        Returns (plate_text, ocr_confidence, plate_crop, province). Detection
        happens here; when the plate region was already located (e.g. while
        choosing between candidate crops) call `read_with_crop` with that region
        instead, so the detector does not run again on the same crop.
        """
        if crop is None or crop.size == 0:
            return None, 0.0, None, None

        return self.read_with_crop(self._find_plate_crop(crop))

    def read_with_crop(self, plate_crop):
        """Read an already-located plate region (no plate detection).

        Returns (plate_text, ocr_confidence, plate_crop, province). A None or
        empty `plate_crop` (no plate was detected) gives (None, 0.0, None, None),
        the same as `read`. plate_text is null when nothing readable sits inside
        the region; the OCR confidence is returned as-is (never used to reject),
        so partial low-confidence readings are preserved.
        """
        if plate_crop is None or plate_crop.size == 0:
            return None, 0.0, None, None

        # Enlarge small plate crops
        _h, w = plate_crop.shape[:2]

        if w < 240:
            scale = 240.0 / w

            plate_crop = cv2.resize(
                plate_crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        # Run OCR
        self._load_ocr()

        pairs = parse_paddle_result(
            self._run_ocr(plate_crop)
        )

        print("PADDLE OCR RESULT:", pairs)

        # Province is a separate reading, taken only from visible text.
        province = detect_province(pairs)

        # A province word on its own is not a registration number.
        candidates = [
            (text, score) for text, score in pairs
            if not is_province_line(text)
        ]

        if not candidates:
            return None, 0.0, plate_crop, province

        # Highest-confidence line, kept even when it is a short partial.
        raw, score = max(
            candidates,
            key=lambda item: item[1]
        )

        if not is_plausible_plate(raw):
            return None, 0.0, plate_crop, province

        text = normalise_plate_text(raw)

        if not text:
            return None, 0.0, plate_crop, province

        return text, score, plate_crop, province
