from __future__ import annotations

import numpy as np
from PIL import Image

from experiments.forensics.phase9_frequency_features import FEATURE_NAMES, extract_frequency_features


def test_features_are_deterministic_and_finite() -> None:
    rng = np.random.default_rng(19)
    image = Image.fromarray(rng.integers(0, 256, (96, 128, 3), dtype=np.uint8))
    first = extract_frequency_features(image)
    second = extract_frequency_features(image)
    assert first["status"] == "ok"
    assert first["features"] == second["features"]
    assert len(first["features"]) == len(FEATURE_NAMES)
    assert np.isfinite(first["features"]).all()


def test_small_image_is_explicitly_unsupported() -> None:
    result = extract_frequency_features(Image.new("RGB", (4, 20), "white"))
    assert result["status"] == "unsupported"
    assert result["features"] is None
    assert "below 8" in result["failure"]


def test_corrupt_and_wrong_type_inputs_are_recorded() -> None:
    corrupt = Image.new("RGB", (16, 16))
    corrupt.close()
    result = extract_frequency_features(corrupt)
    assert result["status"] == "error"
    assert result["failure"]

    wrong_type = extract_frequency_features(b"not an image")  # type: ignore[arg-type]
    assert wrong_type["status"] == "error"
    assert "PIL" in wrong_type["failure"]
