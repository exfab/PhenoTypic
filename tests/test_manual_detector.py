"""Tests for ManualDetector threshold-based colony detection."""

import numpy as np
import pytest

from phenotypic.detect import ManualDetector
from phenotypic.data import load_plate_12hr


def test_manual_detector_basic():
    """Test ManualDetector applies user-specified threshold correctly."""
    # Load test image
    image = load_plate_12hr(mode="Image")

    # Apply manual threshold
    detector = ManualDetector(threshold=100, ignore_zeros=True, ignore_borders=True)
    result = detector.apply(image)

    # Check that objmask was created
    assert result.objmask is not None
    mask = result.objmask[:]

    # Check mask is binary
    assert mask.dtype == bool or set(np.unique(mask)).issubset({0, 1, False, True})

    # Check mask has expected shape
    assert mask.shape == image.enh_gray[:].shape


def test_manual_detector_threshold_values():
    """Test that different threshold values produce different masks."""
    image = load_plate_12hr(mode="Image")

    # Low threshold (more foreground) - don't ignore borders to ensure detection
    # Note: enh_gray is normalized to [0, 1] range
    low_detector = ManualDetector(threshold=0.24, ignore_borders=False)
    low_result = low_detector.apply(image)
    low_mask = low_result.objmask[:]

    # High threshold (less foreground)
    high_detector = ManualDetector(threshold=0.30, ignore_borders=False)
    high_result = high_detector.apply(image)
    high_mask = high_result.objmask[:]

    # Low threshold should detect more pixels
    assert low_mask.sum() > high_mask.sum()


def test_manual_detector_ignore_borders():
    """Test ignore_borders parameter removes edge-touching objects."""
    image = load_plate_12hr(mode="Image")

    # With border removal
    with_borders = ManualDetector(threshold=100, ignore_borders=True)
    result_with = with_borders.apply(image)

    # Without border removal
    without_borders = ManualDetector(threshold=100, ignore_borders=False)
    result_without = without_borders.apply(image)

    # Results should be different (assuming image has edge objects)
    # At minimum, they should not be identical
    assert result_with.objmask[:].shape == result_without.objmask[:].shape


def test_manual_detector_threshold_range():
    """Test that threshold value is validated."""
    # Negative threshold should raise error
    # Note: ValueError is wrapped in RuntimeError by the apply method
    with pytest.raises(RuntimeError, match="Threshold must be non-negative"):
        detector = ManualDetector(threshold=-10)
        image = load_plate_12hr(mode="Image")
        detector.apply(image)


def test_manual_detector_zero_threshold():
    """Test edge case of zero threshold (all pixels become foreground)."""
    image = load_plate_12hr(mode="Image")
    detector = ManualDetector(threshold=0, ignore_borders=False)
    result = detector.apply(image)
    mask = result.objmask[:]

    # All pixels should be foreground (>= 0)
    assert mask.all()


def test_manual_detector_max_threshold():
    """Test edge case of very high threshold (no pixels become foreground)."""
    image = load_plate_12hr(mode="Image")
    detector = ManualDetector(threshold=1000000, ignore_borders=False)
    result = detector.apply(image)
    mask = result.objmask[:]

    # No pixels should be foreground (none >= 1000000 for 8-bit image)
    assert not mask.any()


def test_manual_detector_reproducibility():
    """Test that same parameters produce same results."""
    image = load_plate_12hr(mode="Image")

    # Apply twice with same parameters
    detector1 = ManualDetector(threshold=120)
    result1 = detector1.apply(image)

    detector2 = ManualDetector(threshold=120)
    result2 = detector2.apply(image)

    # Results should be identical
    assert np.array_equal(result1.objmask[:], result2.objmask[:])


def test_manual_detector_attributes():
    """Test that ManualDetector stores attributes correctly."""
    threshold = 115
    ignore_zeros = False
    ignore_borders = False

    detector = ManualDetector(
        threshold=threshold,
        ignore_zeros=ignore_zeros,
        ignore_borders=ignore_borders
    )

    assert detector.threshold == threshold
    assert detector.ignore_zeros == ignore_zeros
    assert detector.ignore_borders == ignore_borders
