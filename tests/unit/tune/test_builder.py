from __future__ import annotations

import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tune._evaluation._builder import build_pipeline


def _base() -> ImagePipeline:
    return ImagePipeline(ops=[
        GaussianBlur(sigma=2.0),           # position 0
        OtsuDetector(ignore_zeros=False),  # position 1
    ])


def test_overlay_scalar_field_rebuilds_op_and_leaves_base_untouched():
    base = _base()
    candidate = build_pipeline(base, {"1.ignore_zeros": True, "0.sigma": 4.0})
    cops = candidate.get_ops()
    assert cops["OtsuDetector"].ignore_zeros is True
    assert cops["GaussianBlur"].sigma == 4.0
    # base is unmutated
    assert base.get_ops()["OtsuDetector"].ignore_zeros is False
    assert base.get_ops()["GaussianBlur"].sigma == 2.0


def test_no_overlay_yields_equivalent_pipeline():
    base = _base()
    candidate = build_pipeline(base, {})
    assert list(candidate.get_ops().keys()) == ["GaussianBlur", "OtsuDetector"]


def test_presence_false_drops_the_op():
    base = _base()
    candidate = build_pipeline(base, {"0.GaussianBlur.__enabled__": False})
    assert list(candidate.get_ops().keys()) == ["OtsuDetector"]


def test_presence_true_keeps_the_op():
    base = _base()
    candidate = build_pipeline(base, {"0.GaussianBlur.__enabled__": True, "0.sigma": 1.5})
    assert list(candidate.get_ops().keys()) == ["GaussianBlur", "OtsuDetector"]
    assert candidate.get_ops()["GaussianBlur"].sigma == 1.5


def test_presence_class_mismatch_raises():
    base = _base()
    # position 0 is a GaussianBlur, not an OtsuDetector
    with pytest.raises(ValueError, match="OtsuDetector"):
        build_pipeline(base, {"0.OtsuDetector.__enabled__": False})


def test_position_out_of_range_raises():
    base = _base()
    with pytest.raises(IndexError):
        build_pipeline(base, {"5.sigma": 1.0})


def test_nested_key_not_supported_in_phase_1():
    base = _base()
    with pytest.raises(NotImplementedError):
        build_pipeline(base, {"1.detectors[0].block_size": 7})
