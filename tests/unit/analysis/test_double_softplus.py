"""Unit tests for :class:`phenotypic.analysis.LinearCapAndLagModel`.

Covers the saturating 4-or-5 parameter linear-softplus fitter — smax /
beta / mode handling, shoulder detection, and the two-way mode
dispatch (``fitted_beta`` vs ``fixed_beta``). Shared σ-resolution and
prior machinery is tested via ``test_linear_softplus.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenotypic.analysis import LinearCapAndLagModel
from phenotypic.schema import (
    LINEAR_CAP_AND_LAG_MODEL,
    MODEL_METRICS,
)


# ---------------------------------------------------------------------- #
# Synthetic fixtures
# ---------------------------------------------------------------------- #
def _build_group(
    t: np.ndarray,
    *,
    v: float,
    s0: float,
    lam: float,
    alpha: float,
    smax: float,
    beta: float = 10.0,
    noise_sigma: float = 0.0,
    n_replicates: int = 3,
    strain: str = "Strain1",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build a saturating replicated measurement DataFrame."""
    rng = rng or np.random.default_rng(0)
    y_clean = LinearCapAndLagModel.model_func(
        t=t, v=v, s0=s0, lam=lam, alpha=alpha, smax=smax, beta=beta,
    )
    rows = []
    for rep in range(n_replicates):
        noise = rng.normal(0.0, noise_sigma, size=len(t)) if noise_sigma > 0 else 0.0
        y = np.asarray(y_clean, dtype=float) + noise
        for ti, yi in zip(t, y):
            rows.append(
                {
                    "Metadata_Time": float(ti),
                    "Shape_Area": float(yi),
                    "Metadata_Dataset": "Test",
                    "Metadata_Strain": strain,
                    "Metadata_Replicate": rep,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def clean_fixture():
    """Two-group noise-free saturating fixture."""
    t = np.linspace(0, 20, 30)
    rng = np.random.default_rng(1)
    g1 = _build_group(
        t, v=5.0, s0=1.0, lam=4.0, alpha=10.0, smax=50.0,
        noise_sigma=0.0, strain="Strain1", rng=rng,
    )
    g2 = _build_group(
        t, v=3.0, s0=2.0, lam=6.0, alpha=10.0, smax=40.0,
        noise_sigma=0.0, strain="Strain2", rng=rng,
    )
    return pd.concat([g1, g2], ignore_index=True)


@pytest.fixture(scope="module")
def noisy_fixture():
    t = np.linspace(0, 20, 30)
    rng = np.random.default_rng(2)
    g1 = _build_group(
        t, v=5.0, s0=1.0, lam=4.0, alpha=10.0, smax=50.0,
        noise_sigma=0.5, strain="Strain1", rng=rng,
    )
    g2 = _build_group(
        t, v=3.0, s0=2.0, lam=6.0, alpha=10.0, smax=40.0,
        noise_sigma=0.5, strain="Strain2", rng=rng,
    )
    return pd.concat([g1, g2], ignore_index=True)


# ---------------------------------------------------------------------- #
# Basic behavior
# ---------------------------------------------------------------------- #
class TestBasics:
    def test_initialization(self):
        m = LinearCapAndLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        assert m.on == "Shape_Area"
        assert m.groupby == ["Metadata_Dataset", "Metadata_Strain"]
        assert m.smax is None
        assert m.beta is None
        assert m.shoulder_slope_ratio == 0.05
        assert m.s0_prior is None
        assert not hasattr(m, "prune_saturated")

    def test_model_func_clamps_to_smax(self):
        t = np.linspace(0, 20, 20)
        y = LinearCapAndLagModel.model_func(
            t=t, v=2.0, s0=0.5, lam=2.0, alpha=10.0, smax=5.0, beta=10.0,
        )
        assert y.shape == t.shape
        assert y[-1] < 5.0 + 1e-6

    def test_schema(self, noisy_fixture):
        m = LinearCapAndLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        results = m.analyze(noisy_fixture)
        expected = [
            LINEAR_CAP_AND_LAG_MODEL.v,
            LINEAR_CAP_AND_LAG_MODEL.s0,
            LINEAR_CAP_AND_LAG_MODEL.lam,
            LINEAR_CAP_AND_LAG_MODEL.alpha,
            LINEAR_CAP_AND_LAG_MODEL.smax,
            LINEAR_CAP_AND_LAG_MODEL.beta,
            LINEAR_CAP_AND_LAG_MODEL.mode,
            MODEL_METRICS.MAE,
            MODEL_METRICS.MSE,
            MODEL_METRICS.RMSE,
            MODEL_METRICS.R2,
            MODEL_METRICS.NUM_SAMPLES,
            MODEL_METRICS.LOSS,
            MODEL_METRICS.STATUS,
        ]
        for col in expected:
            assert col in results.columns, f"missing column: {col}"

    def test_no_pruning_attr(self):
        """LinearCapAndLagModel does not prune — saturation IS the model."""
        m = LinearCapAndLagModel(on="Shape_Area", groupby=["Metadata_Strain"])
        # Sanity: ``_prepare_group`` is the base no-op (returns group unchanged).
        df = pd.DataFrame({
            "Metadata_Time": np.linspace(0, 20, 30),
            "Shape_Area": np.linspace(1, 50, 30),
            "Metadata_Strain": "A",
        })
        out = m._prepare_group(df)
        assert len(out) == len(df)


# ---------------------------------------------------------------------- #
# Parameter recovery
# ---------------------------------------------------------------------- #
class TestParameterRecovery:
    def test_recovers_ground_truth(self, clean_fixture):
        m = LinearCapAndLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        results = m.analyze(clean_fixture).set_index("Metadata_Strain")

        row1 = results.loc["Strain1"]
        assert abs(row1[LINEAR_CAP_AND_LAG_MODEL.v] - 5.0) < 0.3
        assert abs(row1[LINEAR_CAP_AND_LAG_MODEL.s0] - 1.0) < 0.5
        assert abs(row1[LINEAR_CAP_AND_LAG_MODEL.lam] - 4.0) < 0.5

        row2 = results.loc["Strain2"]
        assert abs(row2[LINEAR_CAP_AND_LAG_MODEL.v] - 3.0) < 0.3
        assert abs(row2[LINEAR_CAP_AND_LAG_MODEL.s0] - 2.0) < 0.5
        assert abs(row2[LINEAR_CAP_AND_LAG_MODEL.lam] - 6.0) < 0.5


# ---------------------------------------------------------------------- #
# smax fallback
# ---------------------------------------------------------------------- #
class TestSmaxFallback:
    def test_per_group_observed_max(self, clean_fixture):
        m = LinearCapAndLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        results = m.analyze(clean_fixture).set_index("Metadata_Strain")
        assert abs(results.loc["Strain1", LINEAR_CAP_AND_LAG_MODEL.smax] - 50.0) < 1.0
        assert abs(results.loc["Strain2", LINEAR_CAP_AND_LAG_MODEL.smax] - 40.0) < 1.0

    def test_explicit_smax_overrides(self, clean_fixture):
        m = LinearCapAndLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            smax=100.0,
        )
        results = m.analyze(clean_fixture)
        assert (results[LINEAR_CAP_AND_LAG_MODEL.smax] == 100.0).all()


# ---------------------------------------------------------------------- #
# Per-group mode dispatch (now binary: fitted_beta vs fixed_beta)
# ---------------------------------------------------------------------- #
class TestModeDispatch:
    def test_saturating_curve_uses_fitted_beta(self):
        t = np.linspace(0, 20, 30)
        rng = np.random.default_rng(100)
        df = _build_group(
            t, v=5.0, s0=1.0, lam=4.0, alpha=10.0, smax=50.0, beta=12.0,
            noise_sigma=0.1, strain="Saturated", rng=rng, n_replicates=2,
        )
        m = LinearCapAndLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        res = m.analyze(df)
        assert res[LINEAR_CAP_AND_LAG_MODEL.mode].iloc[0] == "fitted_beta"
        beta_fit = float(res[LINEAR_CAP_AND_LAG_MODEL.beta].iloc[0])
        assert 2.0 <= beta_fit <= 50.0
        assert np.isfinite(float(res[LINEAR_CAP_AND_LAG_MODEL.smax].iloc[0]))

    def test_non_saturating_with_smax_uses_fixed_beta(self):
        """No shoulder + ``beta=None`` → ``fixed_beta`` at module default."""
        t = np.linspace(0, 6, 15)
        rng = np.random.default_rng(102)
        df = _build_group(
            t, v=5.0, s0=1.0, lam=3.0, alpha=10.0, smax=50.0, beta=10.0,
            noise_sigma=0.0, strain="OpenWithSmax", rng=rng, n_replicates=1,
        )
        m = LinearCapAndLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            smax=50.0,
        )
        res = m.analyze(df)
        assert res[LINEAR_CAP_AND_LAG_MODEL.mode].iloc[0] == "fixed_beta"
        assert float(res[LINEAR_CAP_AND_LAG_MODEL.smax].iloc[0]) == 50.0
        assert float(res[LINEAR_CAP_AND_LAG_MODEL.beta].iloc[0]) == 10.0

    def test_explicit_beta_forces_fixed_mode_even_with_shoulder(self):
        t = np.linspace(0, 20, 30)
        rng = np.random.default_rng(103)
        df = _build_group(
            t, v=5.0, s0=1.0, lam=4.0, alpha=10.0, smax=50.0, beta=10.0,
            noise_sigma=0.0, strain="SatButPinned", rng=rng, n_replicates=1,
        )
        m = LinearCapAndLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            beta=7.0,
        )
        res = m.analyze(df)
        assert res[LINEAR_CAP_AND_LAG_MODEL.mode].iloc[0] == "fixed_beta"
        assert float(res[LINEAR_CAP_AND_LAG_MODEL.beta].iloc[0]) == 7.0

    def test_fitted_beta_recovers_distinct_ground_truth(self):
        t = np.linspace(0, 20, 40)
        rng = np.random.default_rng(104)
        g_sharp = _build_group(
            t, v=5.0, s0=1.0, lam=4.0, alpha=10.0, smax=50.0, beta=25.0,
            noise_sigma=0.05, strain="SharpKnee", rng=rng, n_replicates=2,
        )
        g_soft = _build_group(
            t, v=5.0, s0=1.0, lam=4.0, alpha=10.0, smax=50.0, beta=4.0,
            noise_sigma=0.05, strain="SoftKnee", rng=rng, n_replicates=2,
        )
        df = pd.concat([g_sharp, g_soft], ignore_index=True)
        m = LinearCapAndLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        res = m.analyze(df).set_index("Metadata_Strain")
        for strain in ("SharpKnee", "SoftKnee"):
            assert res.loc[strain, LINEAR_CAP_AND_LAG_MODEL.mode] == "fitted_beta"
        beta_sharp = float(res.loc["SharpKnee", LINEAR_CAP_AND_LAG_MODEL.beta])
        beta_soft = float(res.loc["SoftKnee", LINEAR_CAP_AND_LAG_MODEL.beta])
        assert beta_sharp > beta_soft + 3.0
        assert abs(beta_soft - 4.0) < 3.0

    def test_non_positive_beta_raises(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="beta must be None or"):
                LinearCapAndLagModel(
                    on="Shape_Area",
                    groupby=["Metadata_Dataset", "Metadata_Strain"],
                    beta=bad,
                )

    def test_invalid_shoulder_slope_ratio_raises(self):
        for bad in (0.0, -0.1, 1.0, 1.5, float("nan")):
            with pytest.raises(ValueError, match="shoulder_slope_ratio"):
                LinearCapAndLagModel(
                    on="Shape_Area",
                    groupby=["Metadata_Dataset", "Metadata_Strain"],
                    shoulder_slope_ratio=bad,
                )


# ---------------------------------------------------------------------- #
# Shoulder detection
# ---------------------------------------------------------------------- #
class TestShoulderDetection:
    """Unit tests for ``_has_saturation_shoulder`` — the signal driving
    the ``fitted_beta`` vs ``fixed_beta`` branch of mode dispatch.
    """

    @pytest.fixture
    def model(self):
        return LinearCapAndLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )

    def _wrap(self, t, y) -> pd.DataFrame:
        return pd.DataFrame({"Metadata_Time": t, "Shape_Area": y})

    def test_saturating_curve_detected(self, model):
        t = np.linspace(0, 20, 30)
        y = LinearCapAndLagModel.model_func(
            t=t, v=5.0, s0=1.0, lam=4.0, alpha=10.0, smax=50.0, beta=10.0,
        )
        assert model._has_saturation_shoulder(self._wrap(t, y))

    def test_linear_growth_throughout_not_detected(self, model):
        t = np.linspace(0, 20, 30)
        y = 3.0 * t + 1.0
        assert not model._has_saturation_shoulder(self._wrap(t, y))

    def test_flat_noise_not_detected(self, model):
        rng = np.random.default_rng(200)
        t = np.linspace(0, 20, 30)
        y = 1.0 + rng.normal(0, 1e-4, size=t.size)
        assert not model._has_saturation_shoulder(self._wrap(t, y))

    def test_too_few_points_not_detected(self, model):
        t = np.linspace(0, 20, 5)
        y = LinearCapAndLagModel.model_func(
            t=t, v=5.0, s0=1.0, lam=4.0, alpha=10.0, smax=50.0, beta=10.0,
        )
        assert not model._has_saturation_shoulder(self._wrap(t, y))


# ---------------------------------------------------------------------- #
# s0 prior with cv/sigma split (LinearCapAndLagModel inherits the same machinery)
# ---------------------------------------------------------------------- #
class TestInoculumPrior:
    def test_cv_sigma_xor_validation(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            LinearCapAndLagModel(
                on="Shape_Area",
                groupby=["Metadata_Dataset", "Metadata_Strain"],
                s0_prior=1.0,
                s0_prior_cv=0.05,
                s0_prior_sigma=0.5,
            )

    def test_direct_sigma_branch(self):
        m = LinearCapAndLagModel(
            on="Shape_Area",
            groupby=["G"],
            s0_prior=10.0,
            s0_prior_sigma=2.5,
        )
        assert m._prior.cv is None
        assert m._prior.direct_sigma == pytest.approx(2.5)
        stats = m._prior.stats_for(pd.DataFrame({"G": ["x"]}))
        assert stats == pytest.approx((10.0, 2.5))

    def test_cv_branch(self):
        m = LinearCapAndLagModel(
            on="Shape_Area",
            groupby=["G"],
            s0_prior=5.0,
            s0_prior_cv=0.1,
        )
        assert m._prior.cv == 0.1
        stats = m._prior.stats_for(pd.DataFrame({"G": ["x"]}))
        assert stats == pytest.approx((5.0, 0.5))


# ---------------------------------------------------------------------- #
# Degenerate input → NaN row (NaN ``mode`` column behavior)
# ---------------------------------------------------------------------- #
class TestDegenerateInput:
    def test_nan_input_triggers_nan_row(self):
        t = np.linspace(0, 20, 15)
        rows = []
        for ti in t:
            rows.append(
                {
                    "Metadata_Time": float(ti),
                    "Shape_Area": float("nan"),
                    "Metadata_Dataset": "Test",
                    "Metadata_Strain": "Broken",
                }
            )
        df = pd.DataFrame(rows)
        m = LinearCapAndLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        res = m.analyze(df)
        assert len(res) == 1
        assert np.isnan(float(res[LINEAR_CAP_AND_LAG_MODEL.v].iloc[0]))
        assert pd.isna(res[LINEAR_CAP_AND_LAG_MODEL.beta].iloc[0])
        assert pd.isna(res[LINEAR_CAP_AND_LAG_MODEL.mode].iloc[0])
