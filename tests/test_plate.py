import numpy as np

from src.analyzers.plate import (PlateReader, detect_province, is_plausible_plate,
                                 is_province_line, normalise_plate_text,
                                 parse_paddle_result)


def test_normalise():
    assert normalise_plate_text(" lea 1234 ") == "LEA-1234"
    assert normalise_plate_text("A?B*C-123") == "ABC-123"


def test_parse_paddle_3x_format():
    result = [{"rec_texts": ["ABC-123", "x"], "rec_scores": [0.93, 0.2]}]
    assert parse_paddle_result(result) == [("ABC-123", 0.93), ("x", 0.2)]


def test_parse_paddle_3x_nested_res():
    result = [{"res": {"rec_texts": ["LEA 1234"], "rec_scores": [0.88]}}]
    assert parse_paddle_result(result) == [("LEA 1234", 0.88)]


def test_parse_paddle_2x_format():
    result = [[[[[0, 0], [1, 0], [1, 1], [0, 1]], ("ABC-123", 0.91)]]]
    assert parse_paddle_result(result) == [("ABC-123", 0.91)]


def test_parse_empty():
    assert parse_paddle_result(None) == []
    assert parse_paddle_result([None]) == []


# --------------------------------------------------------------- plausibility
def test_plausible_keeps_partials_and_low_confidence_text():
    assert is_plausible_plate("VCV")         # short partial
    assert is_plausible_plate("A")           # single readable character
    assert is_plausible_plate("GX15 0GJ")    # full plate
    assert is_plausible_plate("GX15-0GJ")
    assert is_plausible_plate("AB")          # two characters


def test_plausible_rejects_noise():
    assert not is_plausible_plate("")
    assert not is_plausible_plate("---")
    assert not is_plausible_plate("...")
    assert not is_plausible_plate("AAAA")    # one repeated character
    assert not is_plausible_plate("1111")
    assert not is_plausible_plate("TURBO")   # decoration word
    assert not is_plausible_plate("4X4")


# ------------------------------------------------------------------- province
def test_province_full_names():
    assert detect_province([("PUNJAB", 0.9)]) == "Punjab"
    assert detect_province([("Sindh", 0.9)]) == "Sindh"
    assert detect_province([("KHYBER PAKHTUNKHWA", 0.9)]) == "Khyber Pakhtunkhwa"
    assert detect_province([("Balochistan", 0.9)]) == "Balochistan"
    assert detect_province([("ISLAMABAD", 0.9)]) == "Islamabad"
    assert detect_province([("GILGIT BALTISTAN", 0.9)]) == "Gilgit-Baltistan"
    assert detect_province([("AZAD KASHMIR", 0.9)]) == "Azad Jammu and Kashmir"


def test_province_abbreviation_needs_the_whole_line():
    assert detect_province([("KPK", 0.9)]) == "Khyber Pakhtunkhwa"
    assert detect_province([("GB", 0.9)]) == "Gilgit-Baltistan"
    # an abbreviation must not match ordinary registration characters
    assert detect_province([("GX15-0GJ", 0.9)]) is None
    assert detect_province([("ABGC 123", 0.9)]) is None
    assert detect_province([("ICT123", 0.9)]) is None


def test_province_absent():
    assert detect_province([("GX15 0GJ", 0.9)]) is None
    assert detect_province([]) is None


def test_province_line_is_not_a_registration_number():
    assert is_province_line("Punjab")
    assert is_province_line("KPK")
    assert not is_province_line("PUNJAB 123")    # still carries registration chars
    assert not is_province_line("GX15-0GJ")


# ------------------------------------------------------------- read with crop
def _fake_ocr_pairs(texts, scores):
    return [{"rec_texts": texts, "rec_scores": scores}]


def _reader_without_models(monkeypatch, pairs):
    """A PlateReader whose OCR is canned and whose models are never loaded."""
    reader = PlateReader()
    monkeypatch.setattr(reader, "_load_ocr", lambda: None)
    monkeypatch.setattr(reader, "_run_ocr", lambda image: pairs)
    return reader


def test_read_with_crop_reads_the_given_region_without_detecting(monkeypatch):
    reader = _reader_without_models(
        monkeypatch, _fake_ocr_pairs(["LEA 1234", "PUNJAB"], [0.9, 0.8]))

    def no_detection(*args, **kwargs):
        raise AssertionError("plate detection must not run on an existing region")

    monkeypatch.setattr(reader, "_find_plate_crop", no_detection)

    plate_crop = np.full((30, 300, 3), 200, dtype=np.uint8)
    text, conf, used, province = reader.read_with_crop(plate_crop)
    assert text == "LEA-1234"
    assert conf == 0.9
    assert used is plate_crop                 # the region OCR actually saw
    assert province == "Punjab"


def test_read_with_crop_handles_a_missing_plate():
    reader = PlateReader()
    assert reader.read_with_crop(None) == (None, 0.0, None, None)
    assert reader.read_with_crop(np.zeros((0, 0, 3), dtype=np.uint8)) == (
        None, 0.0, None, None)


def test_read_detects_once_then_ocrs(monkeypatch):
    plate_crop = np.full((30, 300, 3), 200, dtype=np.uint8)
    reader = _reader_without_models(
        monkeypatch, _fake_ocr_pairs(["GX15 0GJ"], [0.88]))

    calls = {"n": 0}

    def fake_find(crop):
        calls["n"] += 1
        return plate_crop

    monkeypatch.setattr(reader, "_find_plate_crop", fake_find)

    text, conf, used, province = reader.read(np.full((40, 60, 3), 100, dtype=np.uint8))
    assert calls["n"] == 1                    # detection runs exactly once
    assert text == "GX15-0GJ"
    assert conf == 0.88
    assert used is plate_crop
    assert province is None
