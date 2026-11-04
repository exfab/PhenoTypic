"""Tests for :class:`phenotypic.abc_.DetectionClearedWarning`.

An enhancer writes ``detect_mat``, and writing ``detect_mat`` resets the
object map. So an enhancer placed after a detector silently erases every
detected colony. ``ImageEnhancer.apply`` warns when the image it receives
already carries objects, and stays silent otherwise.

``load_synth_yeast_plate()`` ships with its ground-truth colonies already in
the object map (96 of them), so it counts as a detected plate. Tests that need
an undetected plate clear that map first.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from phenotypic import ImagePipeline
from phenotypic.abc_ import DetectionClearedWarning
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss


def _undetected_plate():
    """The synthetic plate with its shipped ground-truth object map cleared."""
    plate = load_synth_yeast_plate()
    plate.objmap.reset()
    assert plate.num_objects == 0
    return plate


def test_warning_is_a_user_warning():
    assert issubclass(DetectionClearedWarning, UserWarning)


def test_enhancer_after_detector_warns():
    plate = _undetected_plate()
    with pytest.warns(DetectionClearedWarning, match="BlurGauss"):
        ImagePipeline(ops=[OtsuDetector(), BlurGauss(sigma=2.0)]).apply(plate)


def test_enhancer_before_detector_is_silent():
    plate = _undetected_plate()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DetectionClearedWarning)
        ImagePipeline(ops=[BlurGauss(sigma=2.0), OtsuDetector()]).apply(plate)


def test_direct_apply_on_a_detected_image_warns_and_says_what_to_do():
    detected = OtsuDetector().apply(_undetected_plate())
    assert detected.num_objects > 0
    with pytest.warns(DetectionClearedWarning) as rec:
        BlurGauss(sigma=2.0).apply(detected)
    caught = [w for w in rec if w.category is DetectionClearedWarning]
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "BlurGauss" in message
    assert "before the detector" in message


def test_the_shipped_synthetic_plate_counts_as_detected():
    """Its ground-truth object map is real detection state, and an enhancer clears it."""
    with pytest.warns(DetectionClearedWarning, match="96 detected objects"):
        BlurGauss(sigma=2.0).apply(load_synth_yeast_plate())


def test_a_resetting_pipeline_on_a_detected_image_is_silent():
    """``reset=True`` clears the object map before the first enhancer runs."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DetectionClearedWarning)
        ImagePipeline(ops=[BlurGauss(sigma=2.0), OtsuDetector()], reset=True).apply(
            load_synth_yeast_plate()
        )


def test_warning_does_not_change_the_result():
    plate = _undetected_plate()
    detected = OtsuDetector().apply(plate)
    with pytest.warns(DetectionClearedWarning):
        after_detection = BlurGauss(sigma=2.0).apply(detected)
    on_fresh_plate = BlurGauss(sigma=2.0).apply(plate)
    np.testing.assert_array_equal(after_detection.detect_mat[:], on_fresh_plate.detect_mat[:])
    assert after_detection.num_objects == 0
