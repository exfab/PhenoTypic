"""Tests for :class:`phenotypic.abc_.DetectionClearedWarning`.

An enhancer writes ``detect_mat``, and writing ``detect_mat`` resets the
object map. So an enhancer placed after a detector silently erases every
detected colony. ``ImagePipeline.apply`` checks the run order of its ``ops``
once per call, nested pipelines included, and warns when an enhancer follows
a detector.

The check is about step order, never about the image: ``load_synth_yeast_plate()``
ships with its 96 ground-truth colonies already in the object map, and an
enhancer applied to it, bare or as a pipeline's first step, stays silent.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from phenotypic import ImagePipeline
from phenotypic.abc_ import DetectionClearedWarning
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import CompositeDetector, OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.measure import MeasureSize


def _cleared(record) -> list[warnings.WarningMessage]:
    return [w for w in record if w.category is DetectionClearedWarning]


def test_warning_is_a_user_warning():
    assert issubclass(DetectionClearedWarning, UserWarning)


def test_enhancer_after_detector_warns_once():
    with pytest.warns(DetectionClearedWarning) as rec:
        ImagePipeline(ops=[OtsuDetector(), BlurGauss(sigma=2.0)]).apply(load_synth_yeast_plate())
    caught = _cleared(rec)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "BlurGauss" in message
    assert "OtsuDetector" in message
    assert "before the detector" in message


def test_enhancer_before_detector_on_the_prelabelled_plate_is_silent():
    plate = load_synth_yeast_plate()
    assert plate.num_objects > 0   # ships with its ground-truth colonies
    with warnings.catch_warnings():
        warnings.simplefilter("error", DetectionClearedWarning)
        ImagePipeline(ops=[BlurGauss(sigma=2.0), OtsuDetector()]).apply(plate)


def test_bare_enhancer_on_the_prelabelled_plate_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DetectionClearedWarning)
        BlurGauss(sigma=2.0).apply(load_synth_yeast_plate())


def test_nested_enhancer_after_outer_detector_warns_once():
    inner = ImagePipeline(ops=[BlurGauss(sigma=2.0)])
    outer = ImagePipeline(ops={"detect": OtsuDetector(), "inner": inner})
    with pytest.warns(DetectionClearedWarning) as rec:
        outer.apply(load_synth_yeast_plate())
    caught = _cleared(rec)
    assert len(caught) == 1
    assert "inner/" in str(caught[0].message)


def test_problem_inside_a_nested_pipeline_warns_once_not_per_level():
    inner = ImagePipeline(ops=[OtsuDetector(), BlurGauss(sigma=2.0)])
    with pytest.warns(DetectionClearedWarning) as rec:
        ImagePipeline(ops={"inner": inner}).apply(load_synth_yeast_plate())
    assert len(_cleared(rec)) == 1


def test_detector_branch_pipeline_is_checked_in_its_own_run_order():
    branch = ImagePipeline(ops=[OtsuDetector(), BlurGauss(sigma=2.0)])
    pipeline = ImagePipeline(ops=[CompositeDetector(ops=[branch])])
    with pytest.warns(DetectionClearedWarning) as rec:
        pipeline.apply(load_synth_yeast_plate())
    assert len(_cleared(rec)) == 1


def test_detector_branch_does_not_order_against_the_outer_chain():
    """A branch runs on its own copy: an outer enhancer after it is a separate question."""
    branch = ImagePipeline(ops=[BlurGauss(sigma=2.0), OtsuDetector()])
    pipeline = ImagePipeline(ops=[BlurGauss(sigma=1.0), CompositeDetector(ops=[branch])])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DetectionClearedWarning)
        pipeline.apply(load_synth_yeast_plate())


def test_apply_and_measure_warns_once_before_the_empty_plate_fails_to_measure():
    """The warning is raised before the steps run, so it explains the failure that follows."""
    pipeline = ImagePipeline(ops=[OtsuDetector(), BlurGauss(sigma=2.0)], meas=[MeasureSize()])
    with pytest.warns(DetectionClearedWarning) as rec:
        with pytest.raises(RuntimeError, match="No objects"):
            pipeline.apply_and_measure(load_synth_yeast_plate())
    assert len(_cleared(rec)) == 1


def test_warning_does_not_change_the_result():
    pipeline = ImagePipeline(ops=[OtsuDetector(), BlurGauss(sigma=2.0)])
    with pytest.warns(DetectionClearedWarning):
        warned = pipeline.apply(load_synth_yeast_plate())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DetectionClearedWarning)
        silenced = pipeline.apply(load_synth_yeast_plate())
    np.testing.assert_array_equal(warned.detect_mat[:], silenced.detect_mat[:])
    np.testing.assert_array_equal(warned.objmap[:], silenced.objmap[:])
    assert warned.num_objects == 0
