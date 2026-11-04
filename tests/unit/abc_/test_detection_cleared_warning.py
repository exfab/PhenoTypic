"""Tests for :class:`phenotypic.abc_.DetectionClearedWarning`.

An enhancer writes ``detect_mat``, and writing ``detect_mat`` resets the
object map. So an enhancer placed after a pipeline's last detector silently
erases every detected colony. Every pipeline run method checks the run order
of ``ops`` once per call, nested pipelines included, and warns when an
enhancer follows the last detector. An enhancer that a later detector follows
is a re-detection and stays silent.

The check is about step order, never about the image: ``load_synth_yeast_plate()``
ships with its 96 ground-truth colonies already in the object map, and an
enhancer applied to it, bare or as a pipeline's first step, stays silent.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from phenotypic import ImagePipeline
from phenotypic._core._pipeline_parts._step_order_check import find_enhancer_after_detector
from phenotypic.abc_ import DetectionClearedWarning, ImageEnhancer
from phenotypic.correction import GridAligner
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import CompositeDetector, OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.grid import GridApply
from phenotypic.measure import MeasureSize, MeasureSymZones


def _cleared(record) -> list[warnings.WarningMessage]:
    return [w for w in record if w.category is DetectionClearedWarning]


def _bad() -> ImagePipeline:
    return ImagePipeline(ops=[OtsuDetector(), BlurGauss(sigma=2.0)])


def _assert_silent(pipeline: ImagePipeline) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DetectionClearedWarning)
        pipeline.apply(load_synth_yeast_plate())


class _FailingEnhancer(ImageEnhancer):
    """An enhancer whose step always raises."""

    def _operate(self, image):
        raise ValueError("deliberate failure")


def test_warning_is_a_user_warning():
    assert issubclass(DetectionClearedWarning, UserWarning)


# ---------------------------------------------------------------------- #
# What triggers it
# ---------------------------------------------------------------------- #
def test_enhancer_after_detector_warns_once_at_the_callers_line():
    with pytest.warns(DetectionClearedWarning) as rec:
        _bad().apply(load_synth_yeast_plate())
    caught = _cleared(rec)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "BlurGauss" in message
    assert "OtsuDetector" in message
    assert "before the detector" in message
    assert caught[0].filename == __file__


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


def test_re_detection_after_the_enhancer_is_silent():
    _assert_silent(ImagePipeline(ops=[OtsuDetector(), BlurGauss(sigma=2.0), OtsuDetector()]))


def test_detect_align_enhance_detect_again_is_silent():
    """The documented GridAligner pattern: the first detection is used, then redone."""
    _assert_silent(
        ImagePipeline(ops=[OtsuDetector(), GridAligner(), BlurGauss(sigma=1.0), OtsuDetector()])
    )


def test_enhancer_after_the_last_detector_warns_even_after_a_re_detection():
    pipeline = ImagePipeline(
        ops={
            "detect": OtsuDetector(),
            "blur": BlurGauss(sigma=1.0),
            "redetect": OtsuDetector(),
            "late blur": BlurGauss(sigma=2.0),
        }
    )
    with pytest.warns(DetectionClearedWarning) as rec:
        pipeline.apply(load_synth_yeast_plate())
    caught = _cleared(rec)
    assert len(caught) == 1
    assert "'late blur'" in str(caught[0].message)
    assert "'redetect'" in str(caught[0].message)


def test_grid_apply_of_a_detector_counts_as_a_detector():
    """Checked on the order alone: ``GridApply(image_op=detector).apply`` hits a
    pre-existing float-slice error on the synthetic plate, unrelated to this check."""
    after = ImagePipeline(ops=[GridApply(image_op=OtsuDetector()), BlurGauss(sigma=2.0)])
    found = find_enhancer_after_detector(after)
    assert found is not None
    assert type(found[1]) is BlurGauss and type(found[3]) is GridApply

    redetected = ImagePipeline(
        ops=[OtsuDetector(), BlurGauss(sigma=2.0), GridApply(image_op=OtsuDetector())]
    )
    assert find_enhancer_after_detector(redetected) is None


# ---------------------------------------------------------------------- #
# Nested pipelines and branches
# ---------------------------------------------------------------------- #
def test_nested_enhancer_after_outer_detector_warns_once():
    inner = ImagePipeline(ops=[BlurGauss(sigma=2.0)])
    outer = ImagePipeline(ops={"detect": OtsuDetector(), "inner": inner})
    with pytest.warns(DetectionClearedWarning) as rec:
        outer.apply(load_synth_yeast_plate())
    caught = _cleared(rec)
    assert len(caught) == 1
    assert "inner/" in str(caught[0].message)


def test_problem_inside_a_nested_pipeline_warns_once_not_per_level():
    with pytest.warns(DetectionClearedWarning) as rec:
        ImagePipeline(ops={"inner": _bad()}).apply(load_synth_yeast_plate())
    assert len(_cleared(rec)) == 1


def test_nested_pipeline_is_judged_inline_so_a_later_outer_detector_re_detects():
    """``{inner: [Otsu, Blur], redetect: Otsu}`` runs ``[Otsu, Blur, Otsu]``: nothing is lost."""
    pipeline = ImagePipeline(ops={"inner": _bad(), "redetect": OtsuDetector()})
    _assert_silent(pipeline)


def test_an_ops_key_spelled_like_a_slot_segment_is_still_checked():
    """Slots are told apart by structure, not by the key's text."""
    branch = ImagePipeline(ops=[OtsuDetector(), BlurGauss(sigma=2.0)])
    pipeline = ImagePipeline(ops={"meas:x": CompositeDetector(ops=[branch])})
    with pytest.warns(DetectionClearedWarning) as rec:
        pipeline.apply(load_synth_yeast_plate())
    caught = _cleared(rec)
    assert len(caught) == 1
    assert "meas:x/" in str(caught[0].message)


def test_a_pipeline_inside_the_meas_slot_is_not_checked():
    probe = MeasureSymZones(center_detector=ImagePipeline(ops=[OtsuDetector(), BlurGauss()]))
    pipeline = ImagePipeline(ops=[BlurGauss(), OtsuDetector()], meas=[probe])
    assert find_enhancer_after_detector(pipeline) is None


def test_detector_branch_that_re_detects_is_silent_after_an_outer_detector():
    """``[Otsu, CompositeDetector([[Blur, Otsu]])]``: the branch's own order is fine."""
    branch = ImagePipeline(ops=[BlurGauss(sigma=2.0), OtsuDetector()])
    _assert_silent(ImagePipeline(ops=[OtsuDetector(), CompositeDetector(ops=[branch])]))


def test_detector_branch_is_not_ordered_against_the_outer_chain():
    """Expanding the branch inline would read ``[Otsu, Blur]`` and warn; it runs on its own copy."""
    branch = ImagePipeline(ops=[BlurGauss(sigma=2.0)])
    _assert_silent(ImagePipeline(ops=[OtsuDetector(), CompositeDetector(ops=[branch])]))


def test_bad_detector_branch_warns_although_the_outer_chain_re_detects():
    """Expanding the branch inline would read ``[Otsu, Blur, Otsu]`` and stay silent."""
    branch = ImagePipeline(ops=[OtsuDetector(), BlurGauss(sigma=2.0)])
    pipeline = ImagePipeline(ops=[CompositeDetector(ops=[branch]), OtsuDetector()])
    with pytest.warns(DetectionClearedWarning) as rec:
        pipeline.apply(load_synth_yeast_plate())
    assert len(_cleared(rec)) == 1


# ---------------------------------------------------------------------- #
# Every run method, once per call
# ---------------------------------------------------------------------- #
def test_apply_and_measure_warns_once_at_the_callers_line():
    """Raised before the steps run, so it explains the empty plate's measurement failure."""
    pipeline = ImagePipeline(ops=[OtsuDetector(), BlurGauss(sigma=2.0)], meas=[MeasureSize()])
    with pytest.warns(DetectionClearedWarning) as rec:
        with pytest.raises(RuntimeError, match="No objects"):
            pipeline.apply_and_measure(load_synth_yeast_plate())
    caught = _cleared(rec)
    assert len(caught) == 1
    assert caught[0].filename == __file__


def test_apply_with_intermediates_warns_once():
    with pytest.warns(DetectionClearedWarning) as rec:
        _bad().apply_with_intermediates(load_synth_yeast_plate())
    caught = _cleared(rec)
    assert len(caught) == 1
    assert caught[0].filename == __file__


def _mock_viewer() -> MagicMock:
    """A napari viewer stand-in whose ``add_*`` calls register the layer by name."""
    viewer = MagicMock()
    viewer.layers = {}

    def _add(data, *, name, **kwargs):
        layer = MagicMock()
        layer.data = data
        viewer.layers[name] = layer

    viewer.add_image.side_effect = _add
    viewer.add_labels.side_effect = _add
    return viewer


@patch(
    "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
    True,
)
def test_apply_napari_warns_once():
    with pytest.warns(DetectionClearedWarning) as rec:
        _bad().apply_napari(load_synth_yeast_plate(), viewer=_mock_viewer())
    caught = _cleared(rec)
    assert len(caught) == 1
    assert caught[0].filename == __file__


def test_a_step_that_raises_does_not_leave_the_check_switched_off():
    failing = ImagePipeline(ops={"inner": ImagePipeline(ops=[_FailingEnhancer()])})
    with pytest.raises(RuntimeError, match="deliberate failure"):
        failing.apply(load_synth_yeast_plate())
    with pytest.warns(DetectionClearedWarning) as rec:
        _bad().apply(load_synth_yeast_plate())
    assert len(_cleared(rec)) == 1


def test_warning_does_not_change_the_result():
    with pytest.warns(DetectionClearedWarning):
        warned = _bad().apply(load_synth_yeast_plate())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DetectionClearedWarning)
        silenced = _bad().apply(load_synth_yeast_plate())
    np.testing.assert_array_equal(warned.detect_mat[:], silenced.detect_mat[:])
    np.testing.assert_array_equal(warned.objmap[:], silenced.objmap[:])
    assert warned.num_objects == 0
