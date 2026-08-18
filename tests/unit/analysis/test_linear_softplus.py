"""Unit tests for :class:`phenotypic.analysis.LinearLagModel`.

Covers the unclamped 4-parameter linear-softplus fitter and the
shared base-class machinery (σ resolution, s0 prior, aggregation
broadcasting). The 3-mode dispatch is gone in this class; mode-related
tests live in ``test_double_softplus.py`` instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenotypic.analysis import LinearCapAndLagModel, LinearLagModel
from phenotypic.schema import (
    CULTURE,
    EXPERIMENT,
    GENETIC,
    LINEAR_LAG_MODEL,
    MODEL_METRICS,
    SAMPLE,
    qualified_header,
)


def _q(member):
    """Qualified header for tests that fit on ``Shape_Area`` (token ``Area``)."""
    return qualified_header(member, "Area")


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
    smax: float | None = None,
    beta: float = 10.0,
    noise_sigma: float = 0.0,
    n_replicates: int = 3,
    strain: str = "Strain1",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build a replicated measurement DataFrame for a single strain.

    When ``smax`` is given, generates from the saturating
    ``LinearCapAndLagModel.model_func`` for compatibility with ground-truth
    fixtures inherited from the legacy test file. Otherwise generates
    from the unclamped ``LinearLagModel.model_func``.
    """
    rng = rng or np.random.default_rng(0)
    if smax is None:
        y_clean = LinearLagModel.model_func(t=t, v=v, s0=s0, lam=lam, alpha=alpha)
    else:
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
    """Two-group noise-free fixture for parameter-recovery tests.

    Uses ``smax=None`` so the data is purely linear-with-lag — the
    natural fit target for ``LinearLagModel``.
    """
    t = np.linspace(0, 6, 20)  # truncated before any saturation
    rng = np.random.default_rng(1)
    g1 = _build_group(
        t, v=5.0, s0=1.0, lam=2.0, alpha=10.0, smax=None,
        noise_sigma=0.0, strain="Strain1", rng=rng,
    )
    g2 = _build_group(
        t, v=3.0, s0=2.0, lam=3.0, alpha=10.0, smax=None,
        noise_sigma=0.0, strain="Strain2", rng=rng,
    )
    return pd.concat([g1, g2], ignore_index=True)


@pytest.fixture(scope="module")
def noisy_fixture():
    """Two-group noisy fixture for R² / schema tests."""
    t = np.linspace(0, 6, 20)
    rng = np.random.default_rng(2)
    g1 = _build_group(
        t, v=5.0, s0=1.0, lam=2.0, alpha=10.0, smax=None,
        noise_sigma=0.5, strain="Strain1", rng=rng,
    )
    g2 = _build_group(
        t, v=3.0, s0=2.0, lam=3.0, alpha=10.0, smax=None,
        noise_sigma=0.5, strain="Strain2", rng=rng,
    )
    return pd.concat([g1, g2], ignore_index=True)


# ---------------------------------------------------------------------- #
# Basic behavior
# ---------------------------------------------------------------------- #
class TestBasics:
    def test_initialization(self):
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        assert m.on == "Shape_Area"
        assert m.groupby == ["Metadata_Dataset", "Metadata_Strain"]
        assert m.time_label == "Metadata_Time"
        assert m.stderr_label is None
        assert m.s0_prior is None
        assert m.s0_prior_cv is None
        assert m.s0_prior_sigma is None
        assert m.s0_prior_groupby is None
        assert m.prune_saturated is True
        assert m.loss == "huber"
        assert not m.verbose

    def test_metadata_capable_model_references_accept_flat_spellings(self):
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Strain", "Metadata_Dataset"],
            time_label="Metadata_Time",
            s0_prior=True,
            s0_prior_groupby="Metadata_Dataset",
        )

        assert m.groupby == [str(GENETIC.STRAIN), str(EXPERIMENT.DATASET)]
        assert m.time_label == str(CULTURE.TIME)
        assert m.s0_prior_groupby == [str(EXPERIMENT.DATASET)]

    def test_s0_prior_normalizes_only_string_metadata_references(self):
        configured = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
            s0_prior="Metadata_SampleID",
        )

        assert configured.s0_prior == str(SAMPLE.SAMPLE_ID)
        assert LinearLagModel(
            on="Shape_Area", groupby=["Metadata_Strain"], s0_prior=True
        ).s0_prior is True
        assert LinearLagModel(
            on="Shape_Area", groupby=["Metadata_Strain"], s0_prior=1.5
        ).s0_prior == 1.5
        assert LinearLagModel(
            on="Shape_Area", groupby=["Metadata_Strain"], s0_prior=None
        ).s0_prior is None

    def test_no_smax_or_beta_attrs(self):
        """LinearLagModel has no saturation params (those live on LinearCapAndLagModel)."""
        m = LinearLagModel(on="Shape_Area", groupby=["Metadata_Strain"])
        assert not hasattr(m, "smax")
        assert not hasattr(m, "beta")

    def test_class_constants_present(self):
        """Power-user knobs are class constants, not instance kwargs."""
        assert LinearLagModel._V_UPPER == 50.0
        assert LinearLagModel._STDERR_FLOOR_QUANTILE == 0.25
        assert LinearLagModel._PRUNE_SLOPE_RATIO == 0.05
        assert LinearLagModel._PRUNE_BUFFER == 2

    def test_model_func_shape(self):
        t = np.linspace(0, 10, 20)
        y = LinearLagModel.model_func(t=t, v=2.0, s0=0.5, lam=2.0, alpha=10.0)
        assert y.shape == t.shape
        assert y[-1] > y[0]

    def test_schema(self, noisy_fixture):
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        results = m.analyze(noisy_fixture)
        assert isinstance(results, pd.DataFrame)
        assert not results.empty

        expected = [
            LINEAR_LAG_MODEL.v,
            LINEAR_LAG_MODEL.s0,
            LINEAR_LAG_MODEL.lam,
            LINEAR_LAG_MODEL.alpha,
            MODEL_METRICS.MAE,
            MODEL_METRICS.MSE,
            MODEL_METRICS.RMSE,
            MODEL_METRICS.R2,
            MODEL_METRICS.NUM_SAMPLES,
            MODEL_METRICS.LOSS,
            MODEL_METRICS.STATUS,
        ]
        for col in expected:
            assert _q(col) in results.columns, f"missing column: {col}"

        # Schema *must not* carry smax/beta/mode — those moved to LinearCapAndLagModel.
        for forbidden in ("smax", "beta", "mode"):
            for col in results.columns:
                assert forbidden not in str(col), (
                    f"LinearLagModel output should not contain {forbidden!r}, "
                    f"got column {col!r}."
                )

    def test_r2_finite_and_bounded(self, noisy_fixture):
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        results = m.analyze(noisy_fixture)
        r2 = results[_q(MODEL_METRICS.R2)]
        assert r2.notna().all()
        assert np.isfinite(r2).all()
        assert (r2 <= 1.0 + 1e-6).all()
        assert (r2 > 0.8).all()


# ---------------------------------------------------------------------- #
# Parameter recovery
# ---------------------------------------------------------------------- #
class TestParameterRecovery:
    def test_recovers_ground_truth(self, clean_fixture):
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=False,
        )
        results = m.analyze(clean_fixture).set_index("Metadata_Strain")

        # Strain1: v=5, s0=1, lam=2
        row1 = results.loc["Strain1"]
        assert abs(row1[_q(LINEAR_LAG_MODEL.v)] - 5.0) < 0.3
        assert abs(row1[_q(LINEAR_LAG_MODEL.s0)] - 1.0) < 0.5
        assert abs(row1[_q(LINEAR_LAG_MODEL.lam)] - 2.0) < 0.5

        # Strain2: v=3, s0=2, lam=3
        row2 = results.loc["Strain2"]
        assert abs(row2[_q(LINEAR_LAG_MODEL.v)] - 3.0) < 0.3
        assert abs(row2[_q(LINEAR_LAG_MODEL.s0)] - 2.0) < 0.5
        assert abs(row2[_q(LINEAR_LAG_MODEL.lam)] - 3.0) < 0.5


# ---------------------------------------------------------------------- #
# Weighting (exercises shared σ-resolution machinery in the base class)
# ---------------------------------------------------------------------- #
class TestWeighting:
    def test_weighted_vs_unweighted_differ(self):
        """Stderr label injects per-row σ — distinct from auto-SEM."""
        t = np.linspace(0, 6, 25)
        rng = np.random.default_rng(10)
        clean = _build_group(
            t, v=5.0, s0=1.0, lam=2.0, alpha=10.0, smax=None,
            noise_sigma=0.0, strain="Strain1", rng=rng,
        )
        early_times = np.unique(t)[:4]
        mask = clean["Metadata_Time"].isin(early_times)
        clean.loc[mask, "Shape_Area"] += rng.normal(0, 5.0, size=mask.sum())

        clean["Area_SE"] = np.where(mask, 5.0, 0.1)
        clean["Area_SE_uniform"] = 1.0

        m_weighted = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            stderr_label="Area_SE",
            prune_saturated=False,
        )
        m_unweighted = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            stderr_label="Area_SE_uniform",
            prune_saturated=False,
        )

        res_w = m_weighted.analyze(clean)
        res_u = m_unweighted.analyze(clean)

        s0_w = float(res_w[_q(LINEAR_LAG_MODEL.s0)].iloc[0])
        s0_u = float(res_u[_q(LINEAR_LAG_MODEL.s0)].iloc[0])
        assert abs(s0_w - s0_u) > 1e-3

    def test_stderr_label_column_passed_through(self):
        t = np.linspace(0, 6, 15)
        rng = np.random.default_rng(11)
        df = _build_group(
            t, v=4.0, s0=1.0, lam=2.0, alpha=10.0, smax=None,
            noise_sigma=0.2, strain="Strain1", rng=rng, n_replicates=2,
        )
        df["Area_SE"] = 1.0
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            stderr_label="Area_SE",
            prune_saturated=False,
        )
        res = m.analyze(df)
        assert np.isfinite(res[_q(LINEAR_LAG_MODEL.v)].iloc[0])
        assert np.isfinite(res[_q(LINEAR_LAG_MODEL.s0)].iloc[0])

    def test_singleton_replicate_groups_fall_back_to_unweighted(self):
        """Singleton groups should not blow up — auto-SEM is NaN, fit
        falls back to unweighted residual."""
        t = np.linspace(0, 6, 25)
        rng = np.random.default_rng(7)
        df = _build_group(
            t, v=4.0, s0=1.0, lam=2.0, alpha=10.0, smax=None,
            noise_sigma=0.3, strain="Strain1", rng=rng, n_replicates=1,
        )
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=False,
        )
        res = m.analyze(df)
        v_fit = float(res[_q(LINEAR_LAG_MODEL.v)].iloc[0])
        s0_fit = float(res[_q(LINEAR_LAG_MODEL.s0)].iloc[0])
        lam_fit = float(res[_q(LINEAR_LAG_MODEL.lam)].iloc[0])
        assert np.isfinite(v_fit)
        assert np.isfinite(s0_fit)
        assert np.isfinite(lam_fit)
        assert abs(v_fit - 4.0) < 1.0
        assert abs(lam_fit - 2.0) < 2.0

    def test_stderr_floor_quantile_lifts_sub_quantile_sigma(self):
        """The class-constant ``_STDERR_FLOOR_QUANTILE=0.25`` lifts
        sub-quantile σ values up to the 25th-percentile floor.

        Pins the two semantic guarantees the floor provides without
        relying on full-pipeline parameter recovery:

        - The smallest σ in the floored vector equals
          ``np.quantile(σ_input, 0.25)``.
        - The 1/σ² weight ratio between the stiffest and softest point
          is dramatically reduced compared to the unfloored input
          (orders of magnitude on a multi-decade σ fixture).
        """
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
        )
        # σ spanning four orders of magnitude — without a floor the
        # 1/σ² weight ratio between smallest and largest is 1e8.
        sigma_in = np.array([1e-4, 1e-3, 1e-2, 0.1, 1.0, 1.0, 1.0, 1.0])
        group = pd.DataFrame({
            "Shape_Area": np.arange(len(sigma_in), dtype=float),
            "Shape_Area_stderr": sigma_in,
            "Shape_Area_std_pool": np.full(len(sigma_in), np.nan),
            "Metadata_Strain": ["A"] * len(sigma_in),
            "Metadata_Time": np.arange(len(sigma_in), dtype=float),
        })
        sigma_out = m._resolve_y_stderr(group)
        assert sigma_out is not None

        expected_floor = float(np.quantile(sigma_in, 0.25))
        assert sigma_out.min() == pytest.approx(expected_floor)
        # No value lifted *above* the original max.
        assert sigma_out.max() <= sigma_in.max() + 1e-12

        # Weight ratio dropped by at least 3 orders of magnitude vs.
        # raw input. (Exact post-floor ratio depends on the σ spread,
        # but on this fixture: raw ≈ 1e8, floored ≈ 1.6e4.)
        raw_weight_ratio = (sigma_in.max() / sigma_in.min()) ** 2
        floored_weight_ratio = (sigma_out.max() / sigma_out.min()) ** 2
        assert floored_weight_ratio < raw_weight_ratio / 1e3, (
            f"floor failed to reduce weight ratio enough: "
            f"raw={raw_weight_ratio:.2e}, floored={floored_weight_ratio:.2e}"
        )

    def test_stderr_floor_disabled_via_subclass(self):
        """Subclassing and setting ``_STDERR_FLOOR_QUANTILE = None``
        recovers raw inverse-variance weighting — sub-quantile σ
        values are not lifted."""
        class _NoFloor(LinearLagModel):
            _STDERR_FLOOR_QUANTILE = None  # type: ignore[assignment]

        m = _NoFloor(on="Shape_Area", groupby=["Metadata_Strain"])
        sigma_in = np.array([1e-4, 1e-3, 1e-2, 0.1, 1.0, 1.0, 1.0, 1.0])
        group = pd.DataFrame({
            "Shape_Area": np.arange(len(sigma_in), dtype=float),
            "Shape_Area_stderr": sigma_in,
            "Shape_Area_std_pool": np.full(len(sigma_in), np.nan),
            "Metadata_Strain": ["A"] * len(sigma_in),
            "Metadata_Time": np.arange(len(sigma_in), dtype=float),
        })
        sigma_out = m._resolve_y_stderr(group)
        assert sigma_out is not None
        # No floor: the smallest input σ survives.
        np.testing.assert_array_equal(sigma_out, sigma_in)

    def test_mixed_singleton_pool_broadcasts(self):
        """For mixed n=1/n=3 groups, the pool column is finite within
        the group; for fully-singleton groups it is NaN."""
        t = np.linspace(0, 6, 12)
        rng = np.random.default_rng(77)

        df_a = _build_group(
            t, v=5.0, s0=1.0, lam=2.0, alpha=10.0, smax=None,
            noise_sigma=0.3, strain="StrainA", rng=rng, n_replicates=3,
        )
        times = np.sort(df_a["Metadata_Time"].unique())
        singleton_times = times[::2]
        keep_mask = ~(
            df_a["Metadata_Time"].isin(singleton_times)
            & (df_a["Metadata_Replicate"] > 0)
        )
        df_a = df_a[keep_mask].reset_index(drop=True)

        df_b = _build_group(
            t, v=3.0, s0=2.0, lam=3.0, alpha=10.0, smax=None,
            noise_sigma=0.3, strain="StrainB", rng=rng, n_replicates=1,
        )

        df = pd.concat([df_a, df_b], ignore_index=True)

        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=False,
        )
        m.analyze(df)

        cached = m._latest_measurements
        pool_col = "Shape_Area_std_pool"
        assert pool_col in cached.columns

        pool_a = cached.loc[cached["Metadata_Strain"] == "StrainA", pool_col]
        pool_b = cached.loc[cached["Metadata_Strain"] == "StrainB", pool_col]
        assert pool_a.notna().all()
        assert (pool_a > 0).all()
        assert pool_a.nunique() == 1
        assert pool_b.isna().all()


# ---------------------------------------------------------------------- #
# Saturation pruning (LinearLagModel only)
# ---------------------------------------------------------------------- #
class TestSaturationPruning:
    def test_basic_pruning_drops_plateau_and_preserves_growth(self):
        """Pruning drops plateau rows but keeps the growth phase."""
        t_growth = np.linspace(0, 20, 20)
        t_plateau = np.linspace(20.5, 50, 20)
        t_all = np.concatenate([t_growth, t_plateau])
        rng = np.random.default_rng(20)
        df = _build_group(
            t_all, v=5.0, s0=1.0, lam=4.0, alpha=10.0, smax=50.0,
            noise_sigma=0.0, strain="Strain1", rng=rng, n_replicates=1,
        )
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=True,
        )
        group = df.groupby(["Metadata_Dataset", "Metadata_Strain"]).get_group(
            ("Test", "Strain1")
        )
        pruned = m._prepare_group(group)

        assert len(pruned) < len(group)
        y_pruned = pruned[m.on].to_numpy(dtype=float)
        assert y_pruned.min() <= 2.0
        assert y_pruned.max() >= 49.0

    def test_lag_noise_does_not_trim_lag(self):
        t = np.linspace(0, 20, 30)
        rng = np.random.default_rng(21)
        df = _build_group(
            t, v=5.0, s0=1.0, lam=6.0, alpha=10.0, smax=50.0,
            noise_sigma=0.0, strain="Strain1", rng=rng, n_replicates=1,
        )
        lag_mask = df["Metadata_Time"] <= 5.0
        df.loc[lag_mask, "Shape_Area"] += rng.normal(
            0, 0.3, size=int(lag_mask.sum())
        )

        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=True,
        )
        pruned = m._prepare_group(
            df.groupby(["Metadata_Dataset", "Metadata_Strain"]).get_group(
                ("Test", "Strain1")
            )
        )
        lag_rows_kept = (pruned["Metadata_Time"] <= 5.0).sum()
        lag_rows_original = int(lag_mask.sum())
        assert lag_rows_kept == lag_rows_original

    def test_non_saturating_curve_is_noop(self):
        """Curve ending mid-growth must not be pruned."""
        t = np.linspace(0, 6, 15)
        rng = np.random.default_rng(23)
        df = _build_group(
            t, v=5.0, s0=1.0, lam=4.0, alpha=10.0, smax=50.0,
            noise_sigma=0.0, strain="Strain1", rng=rng, n_replicates=1,
        )
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=True,
        )
        group = df.groupby(["Metadata_Dataset", "Metadata_Strain"]).get_group(
            ("Test", "Strain1")
        )
        pruned = m._prepare_group(group)
        assert len(pruned) == len(group)

    def test_prune_disabled(self):
        t = np.linspace(0, 20, 30)
        rng = np.random.default_rng(24)
        df = _build_group(
            t, v=5.0, s0=1.0, lam=4.0, alpha=10.0, smax=50.0,
            noise_sigma=0.0, strain="Strain1", rng=rng, n_replicates=1,
        )
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=False,
        )
        group = df.groupby(["Metadata_Dataset", "Metadata_Strain"]).get_group(
            ("Test", "Strain1")
        )
        pruned = m._prepare_group(group)
        assert len(pruned) == len(group)


# ---------------------------------------------------------------------- #
# Inoculum prior (exercises the shared base + the new cv/sigma split)
# ---------------------------------------------------------------------- #
class TestInoculumPrior:
    def test_column_prior_pulls_s0_toward_group_mean(self):
        t = np.linspace(0, 6, 25)
        rng = np.random.default_rng(30)
        df = _build_group(
            t, v=5.0, s0=3.0, lam=2.0, alpha=10.0, smax=None,
            noise_sigma=0.3, strain="Strain1", rng=rng, n_replicates=2,
        )
        df["Inoc_Size"] = rng.normal(0.5, 0.02, size=len(df))

        m_no_prior = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=False,
        )
        m_prior = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            s0_prior="Inoc_Size",
            prune_saturated=False,
        )

        s0_no = float(m_no_prior.analyze(df)[_q(LINEAR_LAG_MODEL.s0)].iloc[0])
        s0_yes = float(m_prior.analyze(df)[_q(LINEAR_LAG_MODEL.s0)].iloc[0])

        group_mean = float(df["Inoc_Size"].mean())
        assert abs(s0_yes - group_mean) < abs(s0_no - group_mean)

    def test_no_prior_residual_when_disabled(self):
        """``_loss_func`` residual is N without prior kwargs, N+1 with them."""
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=False,
        )
        y = np.asarray([1.0, 2.0, 3.0], dtype=float)
        t_arr = np.asarray([0.0, 1.0, 2.0], dtype=float)
        params = [1.0, 0.5, 1.0, 10.0]

        # Without saturation kwargs (smax=None) and no prior → length N.
        res = m._loss_func(params, t_arr, y)
        assert res.shape == (3,)

        # With prior kwargs → length N+1.
        res_with_prior = m._loss_func(
            params, t_arr, y, s0_prior_mean=0.5, s0_prior_sigma=0.1,
        )
        assert res_with_prior.shape == (4,)

    def test_zero_variance_column_engages_default_cv(self):
        """A scalar inoc column engages the prior at default σ = 0.05 × µ."""
        t = np.linspace(0, 6, 15)
        rng = np.random.default_rng(32)
        df = _build_group(
            t, v=5.0, s0=1.0, lam=2.0, alpha=10.0, smax=None,
            noise_sigma=0.0, strain="Strain1", rng=rng, n_replicates=1,
        )
        df["Inoc_Size"] = 0.5

        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            s0_prior="Inoc_Size",
            s0_prior_cv=0.2,
            prune_saturated=False,
        )

        probe = df.copy()
        m._prior.prepare(probe, m.groupby, m.time_label)
        group = probe.groupby(
            ["Metadata_Dataset", "Metadata_Strain"]
        ).get_group(("Test", "Strain1"))
        stats = m._inoc_stats(group)
        assert stats is not None
        mu, sigma = stats
        assert mu == pytest.approx(0.5)
        assert sigma == pytest.approx(0.2 * 0.5)

    def test_direct_sigma_override(self):
        """``s0_prior_sigma`` overrides the CV-derived σ with an absolute value."""
        t = np.linspace(0, 6, 15)
        rng = np.random.default_rng(41)
        df = _build_group(
            t, v=5.0, s0=1.0, lam=2.0, alpha=10.0, smax=None,
            noise_sigma=0.0, strain="Strain1", rng=rng, n_replicates=1,
        )
        df["Inoc_Size"] = 10.0

        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            s0_prior="Inoc_Size",
            s0_prior_sigma=2.5,
            prune_saturated=False,
        )
        probe = df.copy()
        m._prior.prepare(probe, m.groupby, m.time_label)
        group = probe.groupby(
            ["Metadata_Dataset", "Metadata_Strain"]
        ).get_group(("Test", "Strain1"))
        stats = m._inoc_stats(group)
        assert stats is not None
        mu, sigma = stats
        assert mu == pytest.approx(10.0)
        assert sigma == pytest.approx(2.5)

    def test_default_cv_when_neither_specified(self):
        """If neither ``s0_prior_cv`` nor ``s0_prior_sigma`` is set,
        the helper applies CV=0.05 as a moderately informative default."""
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            s0_prior=10.0,
        )
        assert m._prior.cv == 0.05
        assert m._prior.direct_sigma is None
        stats = m._prior.stats_for(pd.DataFrame({"x": [1]}))
        assert stats == pytest.approx((10.0, 0.5))

    def test_cv_sigma_xor_validation(self):
        """Passing both ``s0_prior_cv`` and ``s0_prior_sigma`` raises."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            LinearLagModel(
                on="Shape_Area",
                groupby=["Metadata_Dataset", "Metadata_Strain"],
                s0_prior=1.0,
                s0_prior_cv=0.05,
                s0_prior_sigma=0.5,
            )

    def test_direct_mean_scalar_prior(self):
        """Positive scalar ``s0_prior`` engages a uniform prior."""
        t = np.linspace(0, 6, 20)
        rng = np.random.default_rng(40)
        df = _build_group(
            t, v=5.0, s0=3.0, lam=2.0, alpha=10.0, smax=None,
            noise_sigma=0.2, strain="Strain1", rng=rng, n_replicates=2,
        )

        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            s0_prior=0.5,
            s0_prior_cv=0.1,
            prune_saturated=False,
        )
        group = df.groupby(
            ["Metadata_Dataset", "Metadata_Strain"]
        ).get_group(("Test", "Strain1"))
        stats = m._inoc_stats(group)
        assert stats == pytest.approx((0.5, 0.05))

        m_no_prior = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=False,
        )
        s0_no = float(m_no_prior.analyze(df)[_q(LINEAR_LAG_MODEL.s0)].iloc[0])
        s0_yes = float(m.analyze(df)[_q(LINEAR_LAG_MODEL.s0)].iloc[0])
        assert abs(s0_yes - 0.5) < abs(s0_no - 0.5)

    def test_s0_prior_groupby_pools_across_fit_groups(self):
        t = np.linspace(0, 6, 15)
        rng = np.random.default_rng(42)
        g1 = _build_group(
            t, v=5.0, s0=1.0, lam=2.0, alpha=10.0, smax=None,
            noise_sigma=0.1, strain="StrainA", rng=rng, n_replicates=2,
        )
        g2 = _build_group(
            t, v=4.0, s0=1.0, lam=2.0, alpha=10.0, smax=None,
            noise_sigma=0.1, strain="StrainB", rng=rng, n_replicates=2,
        )
        df = pd.concat([g1, g2], ignore_index=True)
        rng_inoc = np.random.default_rng(43)
        df["Inoc_Size"] = np.where(
            df["Metadata_Strain"] == "StrainA",
            rng_inoc.normal(0.3, 0.01, size=len(df)),
            rng_inoc.normal(0.9, 0.01, size=len(df)),
        )

        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            s0_prior="Inoc_Size",
            s0_prior_groupby=["Metadata_Dataset"],
            prune_saturated=False,
        )
        probe = df.copy()
        m._prior.prepare(probe, m.groupby, m.time_label)

        t_min = probe["Metadata_Time"].min()
        expected_mu = float(
            probe.loc[probe["Metadata_Time"] == t_min, "Inoc_Size"].median()
        )
        for strain in ("StrainA", "StrainB"):
            group = probe.groupby(
                ["Metadata_Dataset", "Metadata_Strain"]
            ).get_group(("Test", strain))
            stats = m._inoc_stats(group)
            assert stats is not None
            mu, _ = stats
            assert mu == pytest.approx(expected_mu, rel=1e-6)

    def test_s0_prior_groupby_must_be_subset_of_groupby(self):
        t = np.linspace(0, 6, 10)
        rng = np.random.default_rng(44)
        df = _build_group(
            t, v=5.0, s0=1.0, lam=2.0, alpha=10.0, smax=None,
            noise_sigma=0.0, strain="Strain1", rng=rng, n_replicates=1,
        )
        df["Inoc_Size"] = 0.5

        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
            s0_prior="Inoc_Size",
            s0_prior_groupby=["Metadata_Dataset"],
            prune_saturated=False,
        )
        with pytest.raises(ValueError, match="must be a subset"):
            m.analyze(df)

    def test_s0_prior_groupby_without_column_raises(self):
        with pytest.raises(ValueError, match="requires a column-backed"):
            LinearLagModel(
                on="Shape_Area",
                groupby=["Metadata_Dataset", "Metadata_Strain"],
                s0_prior=0.5,
                s0_prior_groupby=["Metadata_Dataset"],
            )

    def test_missing_s0_prior_column_raises(self):
        t = np.linspace(0, 6, 10)
        rng = np.random.default_rng(45)
        df = _build_group(
            t, v=5.0, s0=1.0, lam=2.0, alpha=10.0, smax=None,
            noise_sigma=0.0, strain="Strain1", rng=rng, n_replicates=1,
        )
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            s0_prior="Inoc_Size",
            prune_saturated=False,
        )
        with pytest.raises(ValueError, match="not present in data columns"):
            m.analyze(df)

    @pytest.mark.parametrize("kwarg", ["s0_prior_cv", "s0_prior_sigma"])
    @pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
    def test_non_positive_factor_raises(self, kwarg, bad):
        with pytest.raises(ValueError, match=kwarg):
            LinearLagModel(
                on="Shape_Area",
                groupby=["Metadata_Dataset", "Metadata_Strain"],
                s0_prior="Inoc_Size",
                **{kwarg: bad},
            )

    def test_non_positive_scalar_prior_raises(self):
        for bad_mean in (0.0, -1.0):
            with pytest.raises(
                ValueError, match="s0_prior scalar must be a positive"
            ):
                LinearLagModel(
                    on="Shape_Area",
                    groupby=["Metadata_Dataset", "Metadata_Strain"],
                    s0_prior=bad_mean,
                )

    def test_empty_s0_prior_groupby_raises(self):
        with pytest.raises(ValueError, match="must not be an empty list"):
            LinearLagModel(
                on="Shape_Area",
                groupby=["Metadata_Dataset", "Metadata_Strain"],
                s0_prior="Inoc_Size",
                s0_prior_groupby=[],
            )

    def test_s0_prior_true_grounds_on_on_column(self):
        """``s0_prior=True`` grounds µ on ``self.on`` at ``t_min``."""
        t = np.linspace(0, 6, 20)
        rng = np.random.default_rng(50)
        df = _build_group(
            t, v=5.0, s0=1.0, lam=2.0, alpha=10.0, smax=None,
            noise_sigma=0.1, strain="Strain1", rng=rng, n_replicates=2,
        )

        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            s0_prior=True,
            prune_saturated=False,
        )
        assert m._prior.label == "Shape_Area"
        assert m._prior.direct_mean is None

        probe = df.copy()
        m._prior.prepare(probe, m.groupby, m.time_label)
        group = probe.groupby(
            ["Metadata_Dataset", "Metadata_Strain"]
        ).get_group(("Test", "Strain1"))
        stats = m._inoc_stats(group)
        assert stats is not None
        mu, sigma = stats

        t_min = float(df["Metadata_Time"].min())
        expected_mu = float(
            df.loc[df["Metadata_Time"] == t_min, "Shape_Area"].median()
        )
        assert mu == pytest.approx(expected_mu, rel=1e-6)
        assert sigma == pytest.approx(0.05 * expected_mu)

    def test_s0_prior_true_not_interpreted_as_numeric(self):
        """Guard against the ``isinstance(True, int)`` gotcha."""
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            s0_prior=True,
        )
        assert m._prior.label == "Shape_Area"
        assert m._prior.direct_mean is None

    def test_s0_prior_false_disables_prior(self):
        m_false = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            s0_prior=False,
        )
        m_none = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            s0_prior=None,
        )
        assert m_false._prior.is_configured is False
        assert m_none._prior.is_configured is False

    def test_s0_prior_invalid_type_raises(self):
        for bad in ([1, 2], {"x": 1}, (0.5,)):
            with pytest.raises(TypeError, match="s0_prior must be"):
                LinearLagModel(
                    on="Shape_Area",
                    groupby=["Metadata_Dataset", "Metadata_Strain"],
                    s0_prior=bad,  # type: ignore[arg-type]
                )


# ---------------------------------------------------------------------- #
# Degenerate input → NaN row
# ---------------------------------------------------------------------- #
class TestDegenerateInput:
    def test_all_zero_data_does_not_crash(self):
        t = np.linspace(0, 6, 15)
        rows = []
        for rep in range(2):
            for ti in t:
                rows.append(
                    {
                        "Metadata_Time": float(ti),
                        "Shape_Area": 0.0,
                        "Metadata_Dataset": "Test",
                        "Metadata_Strain": "Dead",
                        "Metadata_Replicate": rep,
                    }
                )
        df = pd.DataFrame(rows)
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        res = m.analyze(df)
        assert len(res) == 1
        assert 0.0 <= float(res[_q(LINEAR_LAG_MODEL.v)].iloc[0]) <= 50.0

    def test_nan_input_triggers_nan_row(self):
        t = np.linspace(0, 6, 15)
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
        m = LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        res = m.analyze(df)
        assert len(res) == 1
        assert np.isnan(float(res[_q(LINEAR_LAG_MODEL.v)].iloc[0]))
        assert np.isnan(float(res[_q(LINEAR_LAG_MODEL.s0)].iloc[0]))
        assert np.isnan(float(res[_q(LINEAR_LAG_MODEL.lam)].iloc[0]))
        assert np.isnan(float(res[_q(LINEAR_LAG_MODEL.alpha)].iloc[0]))


# ---------------------------------------------------------------------- #
# LogGrowthModel regression — hooks must be no-ops by default
# ---------------------------------------------------------------------- #
def test_log_growth_still_works():
    """Sanity check: extending the base class didn't break siblings."""
    from phenotypic.analysis import LogGrowthModel

    np.random.seed(0)
    t = np.arange(0, 10, 1)
    r_true, K_true, N0_true = 0.5, 1000.0, 50.0
    rows = []
    for ti in t:
        y = K_true / (1 + (K_true - N0_true) / N0_true * np.exp(-r_true * ti))
        for _ in range(3):
            rows.append(
                {
                    "Metadata_Time": float(ti),
                    "Shape_Area": float(y),
                    "Metadata_Dataset": "Test",
                    "Metadata_Strain": "Strain1",
                }
            )
    df = pd.DataFrame(rows)
    m = LogGrowthModel(
        on="Shape_Area",
        groupby=["Metadata_Dataset", "Metadata_Strain"],
    )
    res = m.analyze(df)
    assert not res.empty
