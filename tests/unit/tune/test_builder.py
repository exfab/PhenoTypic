from __future__ import annotations

import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import FilamentousFungiDetector, OtsuDetector
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


def test_nested_key_on_non_list_field_raises():
    """A nested key whose parent field is not an operation list is a clear error.

    Phase 3 enables nested-op overlay (see ``test_build_pipeline_nested.py``);
    here position 1 is an ``OtsuDetector`` with no ``detectors`` list field, so
    the apply step rejects the key loudly rather than silently no-op'ing.
    """
    base = _base()
    with pytest.raises(ValueError, match="detectors"):
        build_pipeline(base, {"1.detectors[0].ignore_zeros": True})


def test_filamentous_scene_parent_tuning_recomputes_auto_derived_fields():
    base = ImagePipeline(ops=[FilamentousFungiDetector(max_colony_radius_px=250.0)])

    candidate = build_pipeline(base, {"0.max_colony_radius_px": 500.0})
    tuned = next(iter(candidate.get_ops().values()))

    assert tuned.max_colony_radius_px == 500.0
    assert tuned.gauss_sigma == 600.0


def test_filamentous_explicit_derived_fields_are_preserved_when_parent_tuned():
    base = ImagePipeline(
        ops=[
            FilamentousFungiDetector(
                max_colony_radius_px=250.0,
                gauss_sigma=123.0,
            )
        ]
    )

    candidate = build_pipeline(base, {"0.max_colony_radius_px": 500.0})
    tuned = next(iter(candidate.get_ops().values()))

    assert tuned.gauss_sigma == 123.0
