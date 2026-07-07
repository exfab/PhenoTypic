"""Unit tests for shared thresholding registry behavior."""
from __future__ import annotations

import numpy as np
import pytest
from skimage import filters

from phenotypic.detect import HysteresisDetector, RoundPeaksDetector
from phenotypic.detect._thresholding_registry import ThresholdingRegistry


def test_validate_method_accepts_known_method_case_insensitively() -> None:
    """Method names are normalized once at the registry boundary."""
    assert ThresholdingRegistry.validate_method("OTSU") == "otsu"


def test_validate_method_rejects_unknown_method() -> None:
    """Unknown threshold names fail instead of falling back to Otsu."""
    with pytest.raises(ValueError, match="Unknown threshold method"):
        ThresholdingRegistry.validate_method("otsuu")


def test_threshold_value_accepts_numeric_manual_threshold() -> None:
    """Manual thresholds are supported for detector paths that allow them."""
    data = np.array([0.0, 1.0, 2.0])

    assert ThresholdingRegistry.threshold_value(1.25, data) == pytest.approx(1.25)


def test_threshold_value_matches_scalar_skimage_method() -> None:
    """Scalar registry dispatch preserves skimage threshold results."""
    data = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)

    observed = ThresholdingRegistry.threshold_value("mean", data)

    assert observed == pytest.approx(float(filters.threshold_mean(data)))


def test_threshold_value_passes_nbins_to_supported_methods(monkeypatch) -> None:
    """Histogram-based methods receive ``2 ** bit_depth`` when available."""
    calls: dict[str, int] = {}

    def fake_minimum(data: np.ndarray, *, nbins: int) -> float:
        calls["nbins"] = nbins
        return float(data.mean())

    monkeypatch.setitem(
        ThresholdingRegistry.METHOD_MAP,
        "minimum",
        fake_minimum,
    )

    observed = ThresholdingRegistry.threshold_value(
        "minimum",
        np.array([0.0, 1.0]),
        bit_depth=12,
    )

    assert observed == pytest.approx(0.5)
    assert calls == {"nbins": 4096}


def test_threshold_value_rejects_local_method() -> None:
    """Local thresholding produces a map, so it is invalid as a scalar value."""
    data = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="does not produce a scalar threshold"):
        ThresholdingRegistry.threshold_value("local", data)


def test_threshold_mask_matches_local_skimage_method() -> None:
    """Local registry dispatch preserves threshold_local mask semantics."""
    matrix = np.arange(25, dtype=float).reshape(5, 5)
    expected = matrix >= filters.threshold_local(matrix, block_size=3)

    observed = ThresholdingRegistry.threshold_mask(
        matrix, method="local", local_block_size=3
    )

    np.testing.assert_array_equal(observed, expected)


def test_grid_detector_runtime_invalid_threshold_raises() -> None:
    """Runtime invalid grid methods raise even when pydantic is bypassed."""
    detector = RoundPeaksDetector.model_construct(thresh_method="not-a-method")

    with pytest.raises(ValueError, match="Unknown threshold method"):
        detector._thresholding(np.arange(25, dtype=float).reshape(5, 5))


def test_hysteresis_compute_threshold_uses_registry_validation() -> None:
    """Hysteresis string thresholds share registry validation and dispatch."""
    data = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)

    observed = HysteresisDetector._compute_threshold("MEAN", data, bit_depth=8)

    assert observed == pytest.approx(float(filters.threshold_mean(data)))
    with pytest.raises(ValueError, match="Unknown threshold method"):
        HysteresisDetector._compute_threshold("not-a-method", data, bit_depth=8)
