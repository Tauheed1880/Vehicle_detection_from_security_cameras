"""Tests the pure rules in clip_classifier (no torch/transformers needed)."""
from src.analyzers.clip_classifier import (
    COLOUR_TABLE, DECORATION_TABLE, MAKE_TABLE, MODEL_TABLE, MODEL_TO_MAKE,
    TYPE_TABLE, empty_result, resolve_predictions,
)


def resolve(make="other", make_conf=0.0, model="other", model_conf=0.0,
            vtype="other", type_conf=0.0, colour="other", colour_conf=0.0,
            decoration="none", decoration_conf=0.9, min_conf=0.5):
    return resolve_predictions(make, make_conf, model, model_conf,
                               vtype, type_conf, colour, colour_conf,
                               decoration, decoration_conf, min_conf)


def test_empty_result_is_all_unknown():
    r = empty_result()
    assert r["make"] == "unknown" and r["make_model_conf"] == 0.0
    assert r["model"] == "unknown"
    assert r["type"] == "unknown" and r["type_conf"] == 0.0
    assert r["colour"] == "unknown" and r["colour_conf"] == 0.0
    assert r["decoration"] == "unknown" and r["decoration_conf"] == 0.0


def test_confident_labels_are_kept():
    r = resolve(make="Toyota", make_conf=0.7, vtype="Sedan", type_conf=0.8,
                colour="white", colour_conf=0.6)
    assert (r["make"], r["make_model_conf"]) == ("Toyota", 0.7)
    assert (r["type"], r["type_conf"]) == ("Sedan", 0.8)
    assert (r["colour"], r["colour_conf"]) == ("white", 0.6)


def test_low_confidence_becomes_unknown():
    r = resolve(make="Toyota", make_conf=0.4, colour="white", colour_conf=0.49,
                min_conf=0.5)
    assert r["make"] == "unknown" and r["make_model_conf"] == 0.0
    assert r["colour"] == "unknown" and r["colour_conf"] == 0.0


def test_other_label_becomes_unknown():
    r = resolve(make="other", make_conf=0.9, vtype="other", type_conf=0.9,
                colour="other", colour_conf=0.9)
    assert r["make"] == "unknown"
    assert r["type"] == "unknown"
    assert r["colour"] == "unknown"


def test_decoration_keeps_other_but_respects_threshold():
    kept = resolve(decoration="other", decoration_conf=0.8)
    assert (kept["decoration"], kept["decoration_conf"]) == ("other", 0.8)

    dropped = resolve(decoration="racing", decoration_conf=0.3, min_conf=0.5)
    assert dropped["decoration"] == "unknown"
    assert dropped["decoration_conf"] == 0.0


def test_known_model_forces_make_consistent():
    # CLIP says Honda + Corolla -> Corolla implies Toyota, so make is corrected
    r = resolve(make="Honda", make_conf=0.6, model="Corolla", model_conf=0.7)
    assert r["make"] == "Toyota"
    assert r["make_model_conf"] == 0.7            # max(make_conf, model_conf)


def test_prompt_tables_have_the_required_labels():
    assert {"Toyota", "Honda", "Suzuki", "Hyundai", "Kia", "Nissan"} <= \
        {label for label, _ in MAKE_TABLE}
    assert {"Corolla", "Civic", "City", "Alto", "Cultus", "Wagon R",
            "Sportage"} <= {label for label, _ in MODEL_TABLE}
    assert {"Sedan", "SUV", "Hatchback", "Pickup", "Van", "Truck", "Bus"} <= \
        {label for label, _ in TYPE_TABLE}
    assert {"white", "black", "silver", "grey", "red", "blue", "green",
            "brown"} <= {label for label, _ in COLOUR_TABLE}
    assert {"mashaallah", "islamic_calligraphy", "racing", "political",
            "brand", "rear_window", "none"} <= {label for label, _ in DECORATION_TABLE}


def test_every_model_maps_to_a_known_make():
    makes = {label for label, _ in MAKE_TABLE}
    for model, make in MODEL_TO_MAKE.items():
        assert make in makes, f"{model} -> {make} is not a known make"
