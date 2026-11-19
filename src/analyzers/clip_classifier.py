"""
Steps 10 + 11 - Zero-shot vehicle classification with OpenAI CLIP (ViT-B/32).

This replaces the old ConvNeXt make/model model (step 10) and the HSV colour
rules (step 11). CLIP is used in "zero-shot" mode: it is never trained here.
Every class is described with a short English text prompt ("a photo of a Toyota
car"). CLIP embeds the text and the vehicle crop into the same space, and the
label whose text is closest to the image wins.

Tasks handled here: make, model, body type, colour and decorations/stickers.
The number plate is NOT handled here - that stays in plate.py (YOLO + PaddleOCR).

IMPORTANT
  * The returned score is a CLIP similarity (a softmax over the label set). It is
    NOT a calibrated probability - do not read it as "90% chance".
  * If the best score is below `min_conf` the answer is "unknown" instead of a
    forced guess.
  * Make and model are kept consistent: a known model implies its make.
  * Zero-shot CLIP is not reliable for exact models or for small stickers. Treat
    those results as hints that need testing on real footage, not as truth.

The model is downloaded once from Hugging Face by `transformers` and cached
locally. The heavy imports live inside `_load()` so the rest of the project
(and the unit tests) never import torch/transformers unless CLIP is really used.
"""
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

UNKNOWN = "unknown"

# --------------------------------------------------------------- prompt tables
# Every table is a list of (label, text prompt). Keep one prompt per label; the
# prompt is what CLIP compares the image against.
MAKE_TABLE = [
    ("Toyota",  "a photo of a Toyota car"),
    ("Honda",   "a photo of a Honda car"),
    ("Suzuki",  "a photo of a Suzuki car"),
    ("Hyundai", "a photo of a Hyundai car"),
    ("Kia",     "a photo of a Kia car"),
    ("Nissan",  "a photo of a Nissan car"),
    ("other",   "a photo of a car of an unknown make"),
]

MODEL_TABLE = [
    ("Corolla",  "a photo of a Toyota Corolla car"),
    ("Civic",    "a photo of a Honda Civic car"),
    ("City",     "a photo of a Honda City car"),
    ("Alto",     "a photo of a Suzuki Alto car"),
    ("Cultus",   "a photo of a Suzuki Cultus car"),
    ("Wagon R",  "a photo of a Suzuki Wagon R car"),
    ("Sportage", "a photo of a Kia Sportage car"),
    ("other",    "a photo of a car of an unknown model"),
]

TYPE_TABLE = [
    ("Sedan",      "a photo of a sedan car"),
    ("SUV",        "a photo of an SUV"),
    ("Hatchback",  "a photo of a hatchback car"),
    ("Pickup",     "a photo of a pickup truck"),
    ("Van",        "a photo of a van"),
    ("Truck",      "a photo of a truck"),
    ("Bus",        "a photo of a bus"),
    ("Motorcycle", "a photo of a motorcycle"),
    ("other",      "a photo of a vehicle"),
]

COLOUR_TABLE = [
    ("white",  "a photo of a white car"),
    ("black",  "a photo of a black car"),
    ("silver", "a photo of a silver car"),
    ("grey",   "a photo of a grey car"),
    ("red",    "a photo of a red car"),
    ("blue",   "a photo of a blue car"),
    ("green",  "a photo of a green car"),
    ("brown",  "a photo of a brown car"),
    ("other",  "a photo of a car of another colour"),
]

DECORATION_TABLE = [
    ("mashaallah",          "a photo of a car with a MashaAllah sticker"),
    ("islamic_calligraphy", "a photo of a car with Islamic calligraphy on the window"),
    ("racing",              "a photo of a car with racing stickers"),
    ("political",           "a photo of a car with political stickers"),
    ("brand",               "a photo of a car with brand stickers"),
    ("rear_window",         "a photo of a car with decorations on the rear window"),
    ("other",               "a photo of a car with other visible stickers or decorations"),
    ("none",                "a photo of a plain car without stickers or decorations"),
]

_TASK_TABLES = {
    "make": MAKE_TABLE,
    "model": MODEL_TABLE,
    "type": TYPE_TABLE,
    "colour": COLOUR_TABLE,
    "decoration": DECORATION_TABLE,
}

# A known model implies its make. Used to keep make/model logically consistent.
MODEL_TO_MAKE = {
    "Corolla": "Toyota",
    "Civic": "Honda",
    "City": "Honda",
    "Alto": "Suzuki",
    "Cultus": "Suzuki",
    "Wagon R": "Suzuki",
    "Sportage": "Kia",
}


def empty_result():
    """A result where every task is unknown (used when CLIP is unavailable)."""
    return {
        "make": UNKNOWN, "model": UNKNOWN, "make_model_conf": 0.0,
        "type": UNKNOWN, "type_conf": 0.0,
        "colour": UNKNOWN, "colour_conf": 0.0,
        "decoration": UNKNOWN, "decoration_conf": 0.0,
    }


def resolve_predictions(make, make_conf, model, model_conf, vtype, type_conf,
                        colour, colour_conf, decoration, decoration_conf,
                        min_conf):
    """Apply the uncertainty threshold and the make/model consistency rule.

    Pure function (no CLIP, no torch) so the rules can be unit-tested.

    A label below `min_conf` becomes "unknown", and the literal label "other"
    (CLIP's way of saying "none of my known classes") also becomes "unknown".
    Decorations keep "other" as a real category.
    """
    def accept(label, conf):
        if label == "other" or conf < min_conf:
            return UNKNOWN, 0.0
        return label, float(conf)

    make, make_conf = accept(make, make_conf)
    model, model_conf = accept(model, model_conf)
    vtype, type_conf = accept(vtype, type_conf)
    colour, colour_conf = accept(colour, colour_conf)

    # decoration has a real "other" category, so only the threshold applies
    if decoration_conf < min_conf:
        decoration, decoration_conf = UNKNOWN, 0.0

    # keep make and model consistent: a known model implies its make
    if model in MODEL_TO_MAKE:
        make = MODEL_TO_MAKE[model]
        make_conf = max(make_conf, model_conf)

    return {
        "make": make, "model": model, "make_model_conf": make_conf,
        "type": vtype, "type_conf": type_conf,
        "colour": colour, "colour_conf": colour_conf,
        "decoration": decoration, "decoration_conf": decoration_conf,
    }


class ClipClassifier:
    """Runs CLIP zero-shot on a vehicle crop and returns all five properties."""

    def __init__(self, model_name="openai/clip-vit-base-patch32", min_conf=0.50,
                 device=None, cache_dir=None, decoration_region=0.6):
        self.model_name = str(model_name)
        self.min_conf = min_conf
        self.device = device
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.decoration_region = decoration_region   # top share used for stickers
        self._loaded = False
        self._available = False
        self._model = None
        self._processor = None
        self._torch = None
        self._text_feats = {}     # task -> (labels, normalised text features)

    # ------------------------------------------------------------------ loading
    def _load(self):
        if self._loaded:
            return
        self._loaded = True

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except Exception:
            log.warning("transformers/torch are not available. Install them "
                        "with 'pip install transformers'. CLIP will return "
                        "'unknown'.")
            return

        self._torch = torch
        try:
            log.info("Loading CLIP '%s' (downloads once, then cached)...",
                     self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name,
                                                    cache_dir=self.cache_dir)
            self._processor = CLIPProcessor.from_pretrained(self.model_name,
                                                            cache_dir=self.cache_dir)
        except Exception:
            log.exception("Could not load CLIP model '%s'. Vehicle properties "
                          "will be 'unknown'.", self.model_name)
            return

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.eval().to(self.device)
        self._precompute_text()
        self._available = True
        log.info("CLIP ready on %s (%d text prompts cached).",
                 self.device, sum(len(t) for t in _TASK_TABLES.values()))

    def _precompute_text(self):
        """Embed every prompt once, so each vehicle only needs image forwards."""
        torch = self._torch
        for task, table in _TASK_TABLES.items():
            labels = [label for label, _ in table]
            prompts = [prompt for _, prompt in table]
            inputs = self._processor(text=prompts, return_tensors="pt",
                                     padding=True).to(self.device)
            with torch.no_grad():
                feats = self._model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            self._text_feats[task] = (labels, feats)

    # -------------------------------------------------------------- inference
    def _image_features(self, crop_bgr):
        from PIL import Image

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=Image.fromarray(rgb),
                                 return_tensors="pt").to(self.device)
        torch = self._torch
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def _best(self, task, image_feats):
        """Return (label, score) of the closest text prompt for this image."""
        labels, text_feats = self._text_feats[task]
        torch = self._torch
        similarity = image_feats @ text_feats.t()           # cosine similarity
        # CLIP was trained with a temperature (logit scale); applying it makes
        # the softmax meaningful. The result is a similarity score, not a
        # calibrated probability.
        logits = similarity * self._model.logit_scale.exp()
        probs = torch.softmax(logits, dim=-1)[0]
        conf, idx = probs.max(dim=0)
        return labels[int(idx)], float(conf)

    def _decoration_region(self, crop):
        """The upper part of the vehicle, where window stickers usually are.

        A small sticker is easy to miss in a full vehicle crop, so decorations
        are looked for on this tighter region.
        """
        h = crop.shape[0]
        cut = int(h * self.decoration_region)
        if cut < 8:
            return crop
        return crop[:cut, :]

    def classify(self, crop):
        """Classify one vehicle crop -> dict of all five properties."""
        self._load()
        if not self._available or crop is None or crop.size == 0:
            return empty_result()

        # one image embedding serves make, model, type and colour
        image_feats = self._image_features(crop)
        make, make_conf = self._best("make", image_feats)
        model, model_conf = self._best("model", image_feats)
        vtype, type_conf = self._best("type", image_feats)
        colour, colour_conf = self._best("colour", image_feats)

        # decorations get their own image embedding from the window region
        dec_feats = self._image_features(self._decoration_region(crop))
        decoration, decoration_conf = self._best("decoration", dec_feats)

        return resolve_predictions(
            make, make_conf, model, model_conf, vtype, type_conf,
            colour, colour_conf, decoration, decoration_conf, self.min_conf)
