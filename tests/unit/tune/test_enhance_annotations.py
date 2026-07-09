"""Per-op annotation checks for the ``enhance/`` family (Wave A).

Two contracts per migrated field:

- **resolution** — ``infer_search_space`` over a one-op pipeline surfaces the
  field as a ``Knob`` with ``source="tune_spec"`` (Tier-1 wins over Tier-2),
  or excludes it with ``reason="tune_spec_off"`` for a ``tunable=False`` hint.
- **pure metadata** — constructing the op with an out-of-(search-)range value
  still succeeds: ``TuneSpec`` is a search hint, never a validator.

Validity-bound migrations (A.2) are covered by the shared back-compat corpus and
the validator-deletion-equivalence cases at the bottom of this file.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from phenotypic import ImagePipeline
from phenotypic.enhance import (
    EnhanceBlockMatch,
    ContrastStretching,
    FocusBlobLoG,
    FocusEdgeMonogenicPhase,
    FocusEdgePhase,
    EnhanceLocalContrast,
    FlattenIllumination,
    GaussianBlur,
    GrayOpening,
    LocalEdgeDenoise,
    MedianFilter,
    NonLocalMeansDenoiser,
    SharpenEdgeGauss,
    SubtractGaussian,
    SubtractOpening,
    SubtractRollingBall,
)
from phenotypic.tune import FloatRange, IntRange, infer_search_space


def _knob(op, field_name: str):
    """Return the inferred ``Knob`` for ``0.<field_name>`` (or ``None``)."""
    space = infer_search_space(ImagePipeline(ops=[op]))
    key = f"0.{field_name}"
    return next((k for k in space.knobs if k.key == key), None)


def _excluded(op, field_name: str):
    space = infer_search_space(ImagePipeline(ops=[op]))
    key = f"0.{field_name}"
    return next((e for e in space.excluded if e.key == key), None)


# --------------------------------------------------------------------------- #
# A.1 — TuneSpec search hints resolve via Tier-1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
        "op, field_name, expected_domain, expected_bounds",
        [
            (GaussianBlur(), "sigma", FloatRange, (0.5, 5.0, True)),
            (MedianFilter(), "width", IntRange, (3, 9)),
            (EnhanceLocalContrast(), "clip_limit", FloatRange, (0.005, 0.05, True)),
            (SubtractGaussian(), "sigma", FloatRange, (20.0, 100.0, False)),
            (SharpenEdgeGauss(), "radius", FloatRange, (0.5, 15.0, True)),
            (SharpenEdgeGauss(), "amount", FloatRange, (0.3, 2.0, False)),
            (LocalEdgeDenoise(), "sigma_spatial", FloatRange, (1.0, 50.0, True)),
            (FocusEdgePhase(), "n_scale", IntRange, (3, 6)),
            (FocusEdgePhase(), "min_wavelength", FloatRange, (2.0, 10.0, False)),
            (FocusEdgePhase(), "sigma_onf", FloatRange, (0.1, 1.0, False)),
            (FocusEdgePhase(), "k", FloatRange, (0.5, 20.0, False)),
            (FocusEdgeMonogenicPhase(), "n_scale", IntRange, (3, 6)),
            (FocusEdgeMonogenicPhase(), "min_wavelength", FloatRange, (2.0, 10.0, False)),
            (FocusEdgeMonogenicPhase(), "sigma_onf", FloatRange, (0.1, 1.0, False)),
            (FocusEdgeMonogenicPhase(), "k", FloatRange, (0.5, 20.0, False)),
            (FocusEdgeMonogenicPhase(), "deviation_gain", FloatRange, (1.0, 2.0, False)),
            (FlattenIllumination(), "sigma", FloatRange, (40.0, 300.0, True)),
            (EnhanceBlockMatch(), "sigma_psd", FloatRange, (0.01, 0.15, True)),
            (FocusBlobLoG(), "min_radius", FloatRange, (1.0, 5.0)),
            (FocusBlobLoG(), "max_radius", FloatRange, (8.0, 50.0)),
            (FocusBlobLoG(), "num_scales", IntRange, (4, 20)),
            (NonLocalMeansDenoiser(), "patch_size", IntRange, (5, 15)),
            (NonLocalMeansDenoiser(), "h", FloatRange, (0.1, 2.0, True)),
            (GrayOpening(), "width", IntRange, (3, 15)),
            (SubtractOpening(), "width", IntRange, (31, 101)),
            (ContrastStretching(), "lower_percentile", IntRange, (1, 5)),
            (ContrastStretching(), "upper_percentile", IntRange, (95, 99)),
            (SubtractRollingBall(), "radius", IntRange, (50, 200, True)),
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


@pytest.mark.parametrize(
        "op, field_name",
        [
            (GaussianBlur(), "truncate"),
        ],
)
def test_tune_spec_off_excludes(op, field_name):
    knob = _knob(op, field_name)
    assert knob is None
    excluded = _excluded(op, field_name)
    assert excluded is not None
    assert excluded.reason == "tune_spec_off"


# --------------------------------------------------------------------------- #
# A.1 — TuneSpec is pure metadata (out-of-range construction still works)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
        "factory",
        [
            lambda: GaussianBlur(sigma=999.0),
            lambda: GaussianBlur(truncate=0.001),
            lambda: MedianFilter(width=101),
            lambda: EnhanceLocalContrast(clip_limit=0.99),
            lambda: SubtractGaussian(sigma=5000.0),
            lambda: SharpenEdgeGauss(radius=999.0, amount=99.0),
            lambda: LocalEdgeDenoise(sigma_spatial=999.0),
            lambda: FocusEdgePhase(n_scale=99, min_wavelength=999.0, k=999.0),
            lambda: FlattenIllumination(sigma=9999.0),
            lambda: EnhanceBlockMatch(sigma_psd=0.99),
            lambda: FocusBlobLoG(min_radius=0.5, max_radius=999.0, num_scales=99),
        ],
)
def test_tunespec_is_pure_metadata(factory):
    """A value outside the search window still constructs (hint, not validator)."""
    op = factory()  # must not raise
    assert op is not None


# --------------------------------------------------------------------------- #
# A.2 — validator→Field migrations keep rejecting the same invalid input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
        "factory",
        [
            lambda: SubtractGaussian(n_iter=0),
            lambda: SharpenEdgeGauss(radius=0.0),
            lambda: SharpenEdgeGauss(n_iter=0),
            lambda: LocalEdgeDenoise(sigma_spatial=0.0),
            lambda: FocusEdgePhase(n_scale=0),
            lambda: FocusEdgePhase(n_orient=0),
            lambda: FocusEdgePhase(min_wavelength=1.0),
            lambda: FocusEdgePhase(mult=1.0),
            lambda: FocusEdgePhase(k=-1.0),
            lambda: FocusEdgePhase(g=0.0),
            lambda: FocusEdgePhase(sigma_onf=0.05),
            lambda: FocusEdgePhase(sigma_onf=1.5),
            lambda: FocusEdgePhase(cutoff=0.0),
            lambda: FocusEdgePhase(cutoff=1.0),
            lambda: FlattenIllumination(sigma=0.0),
            lambda: EnhanceBlockMatch(sigma_psd=-0.1),
        ],
)
def test_invalid_value_still_raises_validation_error(factory):
    """The migrated bound rejects the same invalid input a validator did before."""
    with pytest.raises(ValidationError):
        factory()
