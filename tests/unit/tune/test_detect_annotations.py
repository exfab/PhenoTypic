"""Per-op annotation checks for the ``detect/`` family (Wave B).

Same two contracts as the enhance suite: Tier-1 ``tune_spec`` resolution +
``tunable=False`` opt-out, pure-metadata construction, and validator→Field
equivalence (assert on the ``ValidationError`` type, not the old message). Mode-
dependent bounds (``CannyDetector`` thresholds when ``use_quantiles=False`` are
unbounded) get a ``TuneSpec`` only — no tight ``Field`` upper bound.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from phenotypic import ImagePipeline
from phenotypic.detect import (
    CannyDetector,
    FilamentousFungiDetector,
    InoculumDetector,
    RoundPeaksDetector,
    SinePeakDetector,
    WatershedDetector,
)
from phenotypic.tune import Categorical, FloatRange, IntRange, infer_search_space


def _knob(op, field_name: str):
    space = infer_search_space(ImagePipeline(ops=[op]))
    return next((k for k in space.knobs if k.key == f"0.{field_name}"), None)


def _excluded(op, field_name: str):
    space = infer_search_space(ImagePipeline(ops=[op]))
    return next((e for e in space.excluded if e.key == f"0.{field_name}"), None)


# --------------------------------------------------------------------------- #
# B.1 — TuneSpec search hints resolve via Tier-1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "op, field_name, expected_domain, expected_bounds",
    [
        (WatershedDetector(), "min_size", IntRange, (20, 200)),
        (WatershedDetector(), "compactness", FloatRange, (0.0001, 0.1, True)),
        (CannyDetector(), "sigma", FloatRange, (0.5, 3.0, False)),
        (CannyDetector(), "min_size", IntRange, (20, 500)),
        (CannyDetector(), "low_threshold", FloatRange, (0.05, 0.2, False)),
        (CannyDetector(), "high_threshold", FloatRange, (0.2, 0.4, False)),
        (RoundPeaksDetector(), "footprint_width", IntRange, (4, 20)),
        (RoundPeaksDetector(), "noise_radius", IntRange, (1, 3)),
        (RoundPeaksDetector(), "smoothing_sigma", FloatRange, (0.0, 5.0)),
        (InoculumDetector(), "min_diameter", FloatRange, (5.0, 80.0, True)),
        (InoculumDetector(), "max_diameter", FloatRange, (50.0, 300.0, True)),
        (FilamentousFungiDetector(), "max_colony_radius_px", FloatRange, (50.0, 500.0, True)),
        (FilamentousFungiDetector(), "min_branch_width_px", IntRange, (2, 10)),
        (SinePeakDetector(), "footprint_width", IntRange, (4, 20)),
        (SinePeakDetector(), "noise_radius", IntRange, (1, 3)),
        (SinePeakDetector(), "correlation_threshold", FloatRange, (0.1, 0.5)),
    ],
)
def test_tune_spec_resolves_tier1(op, field_name, expected_domain, expected_bounds):
    knob = _knob(op, field_name)
    assert knob is not None, f"{field_name} not surfaced as a knob"
    assert knob.source == "tune_spec"
    assert knob.needs_review is False
    assert isinstance(knob.domain, expected_domain)
    assert knob.domain.low == expected_bounds[0]
    assert knob.domain.high == expected_bounds[1]
    if len(expected_bounds) == 3:
        assert knob.domain.log is expected_bounds[2]


def test_watershed_connectivity_categorical():
    """connectivity → TuneSpec(categories=[1, 2]) resolves to a Categorical."""
    knob = _knob(WatershedDetector(), "connectivity")
    assert knob is not None
    assert knob.source == "tune_spec"
    assert isinstance(knob.domain, Categorical)
    assert knob.domain.choices == (1, 2)


def test_canny_connectivity_categorical():
    knob = _knob(CannyDetector(), "connectivity")
    assert knob is not None
    assert isinstance(knob.domain, Categorical)
    assert knob.domain.choices == (1, 2)


@pytest.mark.parametrize(
    "op, field_name",
    [
        # FilamentousFungi auto-derived Optional=None scene params → not tunable.
        (FilamentousFungiDetector(), "gauss_sigma"),
        (FilamentousFungiDetector(), "tile_size"),
        (FilamentousFungiDetector(), "mad_window"),
    ],
)
def test_filamentous_auto_params_excluded(op, field_name):
    knob = _knob(op, field_name)
    assert knob is None
    excluded = _excluded(op, field_name)
    assert excluded is not None
    assert excluded.reason == "tune_spec_off"


# --------------------------------------------------------------------------- #
# B.1 — TuneSpec is pure metadata (out-of-range construction still works)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "factory",
    [
        lambda: WatershedDetector(min_size=9999, compactness=99.0),
        lambda: CannyDetector(sigma=99.0, min_size=9999, low_threshold=5.0, high_threshold=9.0),
        lambda: RoundPeaksDetector(footprint_width=999, noise_radius=99, smoothing_sigma=99.0),
        lambda: InoculumDetector(min_diameter=1.0, max_diameter=9999.0),
        lambda: FilamentousFungiDetector(max_colony_radius_px=9999.0, min_branch_width_px=99),
    ],
)
def test_tunespec_is_pure_metadata(factory):
    op = factory()  # must not raise
    assert op is not None


# --------------------------------------------------------------------------- #
# B.2 — validator→Field migrations keep rejecting the same invalid input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "factory",
    [
        lambda: InoculumDetector(min_diameter=0.0),
        lambda: InoculumDetector(max_diameter=0.0),
        lambda: WatershedDetector(min_size=0),
        lambda: WatershedDetector(compactness=-0.1),
        lambda: WatershedDetector(connectivity=0),
        lambda: WatershedDetector(connectivity=3),
        lambda: CannyDetector(min_size=0),
        lambda: CannyDetector(connectivity=0),
        lambda: CannyDetector(connectivity=3),
        lambda: CannyDetector(sigma=0.0),
        lambda: RoundPeaksDetector(footprint_width=0),
        lambda: RoundPeaksDetector(noise_radius=0),
        lambda: SinePeakDetector(footprint_width=0),
        lambda: SinePeakDetector(noise_radius=0),
    ],
)
def test_invalid_value_still_raises_validation_error(factory):
    with pytest.raises(ValidationError):
        factory()


def test_inoculum_diameter_order_still_enforced():
    """The cross-field _check_diameter_order validator is preserved."""
    with pytest.raises(ValidationError):
        InoculumDetector(min_diameter=100.0, max_diameter=50.0)
