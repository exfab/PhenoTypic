"""Tests for :class:`phenotypic.analysis.GrowthRateAtBoundWarning`.

A linear-softplus fit bounds the growth rate ``v`` at ``_V_UPPER`` (50 per
time unit). An area series grows far faster than that, so the optimizer
pins ``v`` to the bound and returns it as if it were a measurement. The
warning surfaces that, once per ``analyze`` call, from the parent process —
so it survives ``n_jobs > 1``, where a warning raised inside a joblib worker
never reaches the caller.

It must stay silent on a fit whose ``v`` is far from its bound, and it must
not fire on ``alpha``, which sits at its own bound (50) on a sharp onset.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from phenotypic.analysis import (
    GrowthRateAtBoundWarning,
    LinearCapAndLagModel,
    LinearLagModel,
)
from phenotypic.schema import LINEAR_LAG_MODEL

_TIMES = np.arange(0.0, 115.0, 6.0)   # 0 to 114 h in steps of 6
_LAG = 24.0


def _frame(column: str, curve: np.ndarray, *, strains: int, noise: float) -> pd.DataFrame:
    """Three replicates per strain of ``curve`` with relative Gaussian noise."""
    rng = np.random.default_rng(0)
    rows = []
    for s in range(strains):
        for rep in range(3):
            y = curve * (1.0 + rng.normal(0.0, noise, size=curve.size))
            for t, value in zip(_TIMES, y):
                rows.append({
                    "Metadata_Strain": f"SYN-{s + 1:02d}",
                    "Metadata_Replicate": rep,
                    "Metadata_Time": float(t),
                    column: float(value),
                })
    return pd.DataFrame(rows)


def _series(*, column: str, v: float, s0: float, strains: int = 2) -> pd.DataFrame:
    """``y = s0 + v * softplus(t - 24)``, three replicates per strain."""
    curve = s0 + v * np.logaddexp(0.0, _TIMES - _LAG)
    return _frame(column, curve, strains=strains, noise=0.005)


def _hinge_series(*, column: str, v: float, s0: float = 8.0, strains: int = 2) -> pd.DataFrame:
    """``y = s0 + v * max(0, t - 24)``: a sharp onset that fits ``alpha`` to its bound."""
    curve = s0 + v * np.maximum(0.0, _TIMES - _LAG)
    return _frame(column, curve, strains=strains, noise=0.005)


def _bound_warnings(record) -> list[warnings.WarningMessage]:
    return [w for w in record if w.category is GrowthRateAtBoundWarning]


def test_warning_is_a_user_warning():
    assert issubclass(GrowthRateAtBoundWarning, UserWarning)


def test_capped_area_fit_warns():
    df = _series(column="Size_Area", v=120.0, s0=500.0)
    with pytest.warns(GrowthRateAtBoundWarning, match="upper bound"):
        LinearLagModel(on="Size_Area", groupby=["Metadata_Strain"]).analyze(df)


def test_capped_area_fit_warns_once_and_names_what_to_do():
    df = _series(column="Size_Area", v=120.0, s0=500.0, strains=7)
    with pytest.warns(GrowthRateAtBoundWarning) as rec:
        LinearLagModel(on="Size_Area", groupby=["Metadata_Strain"]).analyze(df)
    caught = _bound_warnings(rec)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "LinearLagModel" in message
    assert "7 of 7" in message
    assert "50" in message
    assert "radius" in message
    assert "time" in message
    # Up to five group keys are named; the remaining two are counted, not listed.
    assert sum(f"SYN-{i:02d}" in message for i in range(1, 8)) == 5


def test_capped_fit_warns_once_with_parallel_workers():
    df = _series(column="Size_Area", v=120.0, s0=500.0, strains=3)
    with pytest.warns(GrowthRateAtBoundWarning) as rec:
        LinearLagModel(on="Size_Area", groupby=["Metadata_Strain"], n_jobs=2).analyze(df)
    assert len(_bound_warnings(rec)) == 1


def test_capped_cap_and_lag_fit_warns():
    df = _series(column="Size_Area", v=120.0, s0=500.0)
    with pytest.warns(GrowthRateAtBoundWarning, match="LinearCapAndLagModel"):
        LinearCapAndLagModel(on="Size_Area", groupby=["Metadata_Strain"]).analyze(df)


def test_warning_does_not_change_the_result():
    df = _series(column="Size_Area", v=120.0, s0=500.0)
    with pytest.warns(GrowthRateAtBoundWarning):
        warned = LinearLagModel(on="Size_Area", groupby=["Metadata_Strain"]).analyze(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GrowthRateAtBoundWarning)
        silenced = LinearLagModel(on="Size_Area", groupby=["Metadata_Strain"]).analyze(df)
    pd.testing.assert_frame_equal(warned, silenced)


def test_alpha_at_its_bound_is_silent():
    df = _hinge_series(column="Size_RobustMeanRadius", v=0.3)   # sharp onset: alpha fits to 50.0 (K0-11)
    model = LinearLagModel(on="Size_RobustMeanRadius", groupby=["Metadata_Strain"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", GrowthRateAtBoundWarning)
        model.analyze(df)
    fit = model._latest_fit_internal
    assert (fit[LINEAR_LAG_MODEL.alpha] >= 50.0 * (1 - 1e-6)).all()
    assert (fit[LINEAR_LAG_MODEL.v] < 1.0).all()


def test_clean_radius_fit_is_silent():
    df = _series(column="Size_RobustMeanRadius", v=0.3, s0=8.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GrowthRateAtBoundWarning)
        LinearLagModel(on="Size_RobustMeanRadius", groupby=["Metadata_Strain"]).analyze(df)
