"""Unit tests for :class:`phenotypic.analysis.LinearSoftplusModel`.

Fixtures mirror the structure of ``test_log_growth_model.py``: synthetic
two-group time courses built directly from ``model_func`` plus noise so
ground-truth parameters are known. Assertions are tolerance-based to
accommodate the least-squares-with-prior optimizer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenotypic.analysis import LinearSoftplusModel
from phenotypic.tools_.measurement_info_ import (
    LINEAR_SOFTPLUS_MODEL,
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
    smax: float | None,
    beta: float = 10.0,
    noise_sigma: float = 0.0,
    n_replicates: int = 3,
    strain: str = "Strain1",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build a replicated measurement DataFrame for a single strain."""
    rng = rng or np.random.default_rng(0)
    y_clean = LinearSoftplusModel.model_func(
        t=t, v=v, s0=s0, lam=lam, alpha=alpha, smax=smax, beta=beta
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
    """Two-group noise-free fixture for parameter-recovery tests."""
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
    """Two-group noisy fixture for R² / schema tests."""
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
        m = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        assert m.on == "Shape_Area"
        assert m.groupby == ["Metadata_Dataset", "Metadata_Strain"]
        assert m.time_label == "Metadata_Time"
        assert m.smax is None
        assert m.beta == 10
        assert m.stderr_label is None
        assert m.inoc_size_label is None
        assert m.prune_saturated is True
        assert m.saturation_threshold == 0.05
        assert m.saturation_buffer == 2
        assert m.v_upper == 50.0
        assert m.loss == "linear"
        assert not m.verbose

    def test_model_func_shapes(self):
        t = np.linspace(0, 10, 20)
        y_open = LinearSoftplusModel.model_func(
            t=t, v=2.0, s0=0.5, lam=2.0, alpha=10.0, smax=None
        )
        assert y_open.shape == t.shape
        assert y_open[-1] > y_open[0]

        y_sat = LinearSoftplusModel.model_func(
            t=t, v=2.0, s0=0.5, lam=2.0, alpha=10.0, smax=5.0, beta=10.0
        )
        assert y_sat.shape == t.shape
        # saturation ceiling should cap the growth
        assert y_sat[-1] < 5.0 + 1e-6

    def test_schema(self, noisy_fixture):
        m = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        results = m.analyze(noisy_fixture)
        assert isinstance(results, pd.DataFrame)
        assert not results.empty

        expected = [
            LINEAR_SOFTPLUS_MODEL.v,
            LINEAR_SOFTPLUS_MODEL.s0,
            LINEAR_SOFTPLUS_MODEL.lam,
            LINEAR_SOFTPLUS_MODEL.alpha,
            LINEAR_SOFTPLUS_MODEL.smax,
            LINEAR_SOFTPLUS_MODEL.beta,
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

    def test_r2_finite_and_bounded(self, noisy_fixture):
        m = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        results = m.analyze(noisy_fixture)
        r2 = results[MODEL_METRICS.R2]
        assert r2.notna().all()
        assert np.isfinite(r2).all()
        assert (r2 <= 1.0 + 1e-6).all()
        # both groups should fit reasonably well on the noisy fixture
        assert (r2 > 0.8).all()


# ---------------------------------------------------------------------- #
# Parameter recovery
# ---------------------------------------------------------------------- #
class TestParameterRecovery:
    def test_recovers_ground_truth(self, clean_fixture):
        m = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=False,
        )
        results = m.analyze(clean_fixture).set_index("Metadata_Strain")

        # Strain1: v=5, s0=1, lam=4, smax=50
        row1 = results.loc["Strain1"]
        assert abs(row1[LINEAR_SOFTPLUS_MODEL.v] - 5.0) < 0.3
        assert abs(row1[LINEAR_SOFTPLUS_MODEL.s0] - 1.0) < 0.5
        assert abs(row1[LINEAR_SOFTPLUS_MODEL.lam] - 4.0) < 0.5

        # Strain2: v=3, s0=2, lam=6, smax=40
        row2 = results.loc["Strain2"]
        assert abs(row2[LINEAR_SOFTPLUS_MODEL.v] - 3.0) < 0.3
        assert abs(row2[LINEAR_SOFTPLUS_MODEL.s0] - 2.0) < 0.5
        assert abs(row2[LINEAR_SOFTPLUS_MODEL.lam] - 6.0) < 0.5


# ---------------------------------------------------------------------- #
# Weighting
# ---------------------------------------------------------------------- #
class TestWeighting:
    def test_weighted_vs_unweighted_differ(self):
        """Feed a fixture where early-time points are deliberately noisy.

        With the `stderr_label` column flagging high SE at early times,
        the weighted fit should pull ``s0`` closer to the true value
        than the unweighted fit.
        """
        t = np.linspace(0, 20, 25)
        rng = np.random.default_rng(10)
        clean = _build_group(
            t, v=5.0, s0=1.0, lam=4.0, alpha=10.0, smax=50.0,
            noise_sigma=0.0, strain="Strain1", rng=rng,
        )
        # Corrupt the first 4 timepoints (early times, replicated) to be noisy.
        early_times = np.unique(t)[:4]
        mask = clean["Metadata_Time"].isin(early_times)
        clean.loc[mask, "Shape_Area"] += rng.normal(0, 5.0, size=mask.sum())

        # Provide a stderr_label: high for noisy early points, low elsewhere.
        clean["Area_SE"] = np.where(mask, 5.0, 0.1)

        m_weighted = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            stderr_label="Area_SE",
            prune_saturated=False,
        )
        # To make "unweighted" comparable, supply a uniform SE column.
        clean["Area_SE_uniform"] = 1.0
        m_unweighted = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            stderr_label="Area_SE_uniform",
            prune_saturated=False,
        )

        res_w = m_weighted.analyze(clean)
        res_u = m_unweighted.analyze(clean)

        s0_w = float(res_w[LINEAR_SOFTPLUS_MODEL.s0].iloc[0])
        s0_u = float(res_u[LINEAR_SOFTPLUS_MODEL.s0].iloc[0])
        # Weighted should be closer to ground truth s0=1.0 than unweighted.
        assert abs(s0_w - 1.0) < abs(s0_u - 1.0) + 1e-9
        # At minimum the two fits must differ non-trivially.
        assert abs(s0_w - s0_u) > 1e-3

    def test_stderr_label_column_passed_through(self):
        """Given a stderr_label, it must be used — not auto-SEM."""
        t = np.linspace(0, 20, 15)
        rng = np.random.default_rng(11)
        df = _build_group(
            t, v=4.0, s0=1.0, lam=4.0, alpha=10.0, smax=40.0,
            noise_sigma=0.2, strain="Strain1", rng=rng, n_replicates=2,
        )
        df["Area_SE"] = 1.0
        m = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            stderr_label="Area_SE",
            prune_saturated=False,
        )
        res = m.analyze(df)
        # Smoke check: a fit completed with finite parameters.
        assert np.isfinite(res[LINEAR_SOFTPLUS_MODEL.v].iloc[0])
        assert np.isfinite(res[LINEAR_SOFTPLUS_MODEL.s0].iloc[0])


# ---------------------------------------------------------------------- #
# smax fallback
# ---------------------------------------------------------------------- #
class TestSmaxFallback:
    def test_per_group_observed_max(self, clean_fixture):
        m = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=False,
        )
        results = m.analyze(clean_fixture).set_index("Metadata_Strain")
        # Strain1 ground truth smax=50 → observed max ≈ 50
        assert abs(results.loc["Strain1", LINEAR_SOFTPLUS_MODEL.smax] - 50.0) < 1.0
        # Strain2 ground truth smax=40 → observed max ≈ 40
        assert abs(results.loc["Strain2", LINEAR_SOFTPLUS_MODEL.smax] - 40.0) < 1.0

    def test_explicit_smax_overrides(self, clean_fixture):
        m = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            smax=100.0,
            prune_saturated=False,
        )
        results = m.analyze(clean_fixture)
        assert (results[LINEAR_SOFTPLUS_MODEL.smax] == 100.0).all()


# ---------------------------------------------------------------------- #
# Saturation pruning
# ---------------------------------------------------------------------- #
class TestSaturationPruning:
    def test_basic_pruning_drops_plateau_and_preserves_growth(self):
        """On a 20-growth + 20-plateau fixture, ``_prepare_group`` must
        drop real rows (catches a silent no-op) and keep the growth
        phase intact, and the end-to-end fit must still recover ``v``.

        This combines three checks that together pin down the pruning
        behavior deterministically:
          1. ``len(pruned) < len(group)`` — a no-op `_prepare_group`
             (returning the group unchanged) fails here.
          2. ``len(pruned) >= len(t_growth)`` — an over-aggressive
             implementation that trims the growth phase fails here.
          3. The downstream fit recovers ``v`` near ground truth — a
             broken pruner that e.g. drops the wrong rows would fail.
        """
        t_growth = np.linspace(0, 20, 20)
        t_plateau = np.linspace(20.5, 50, 20)
        t_all = np.concatenate([t_growth, t_plateau])
        rng = np.random.default_rng(20)
        df = _build_group(
            t_all, v=5.0, s0=1.0, lam=4.0, alpha=10.0, smax=50.0,
            noise_sigma=0.0, strain="Strain1", rng=rng, n_replicates=1,
        )
        m = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=True,
        )
        group = df.groupby(["Metadata_Dataset", "Metadata_Strain"]).get_group(
            ("Test", "Strain1")
        )
        pruned = m._prepare_group(group)

        # Direct behavioral check — pruning must actually drop rows.
        assert len(pruned) < len(group), (
            f"expected pruning to drop plateau rows; "
            f"got {len(pruned)}/{len(group)}"
        )
        # Dynamic-range check — the pruned slice must span the full
        # growth transition (lag near s0=1 through saturation near
        # smax=50). An over-aggressive pruner that clipped mid-growth
        # would fail here.
        y_pruned = pruned[m.on].to_numpy(dtype=float)
        assert y_pruned.min() <= 2.0, (
            f"pruner dropped the lag phase; min y={y_pruned.min():.2f}"
        )
        assert y_pruned.max() >= 49.0, (
            f"pruner dropped the saturation shoulder; max y={y_pruned.max():.2f}"
        )

        # End-to-end smoke — downstream fit recovers v within tolerance.
        res = m.analyze(df)
        v_fit = float(res[LINEAR_SOFTPLUS_MODEL.v].iloc[0])
        assert abs(v_fit - 5.0) < 0.3, (
            f"pruned fit did not recover v within tolerance: v_fit={v_fit:.4f}"
        )

    def test_lag_noise_does_not_trim_lag(self):
        """Heavy lag-phase noise must NOT trigger early trimming."""
        t = np.linspace(0, 20, 30)
        rng = np.random.default_rng(21)
        df = _build_group(
            t, v=5.0, s0=1.0, lam=6.0, alpha=10.0, smax=50.0,
            noise_sigma=0.0, strain="Strain1", rng=rng, n_replicates=1,
        )
        # Add large noise only to lag-phase points (t <= 5).
        lag_mask = df["Metadata_Time"] <= 5.0
        df.loc[lag_mask, "Shape_Area"] += rng.normal(
            0, 0.3, size=int(lag_mask.sum())
        )

        m = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=True,
        )
        pruned = m._prepare_group(
            df.groupby(["Metadata_Dataset", "Metadata_Strain"]).get_group(
                ("Test", "Strain1")
            )
        )
        # All pre-peak (t <= 8) rows should survive.
        lag_rows_kept = (pruned["Metadata_Time"] <= 5.0).sum()
        lag_rows_original = int(lag_mask.sum())
        assert lag_rows_kept == lag_rows_original

    def test_transient_growth_dip_not_pruned(self):
        """A single noisy low-derivative point mid-growth must NOT trigger."""
        t = np.linspace(0, 20, 40)
        rng = np.random.default_rng(22)
        df = _build_group(
            t, v=5.0, s0=1.0, lam=4.0, alpha=10.0, smax=50.0,
            noise_sigma=0.0, strain="Strain1", rng=rng, n_replicates=1,
        )
        # Inject a transient dip at a mid-growth timepoint.
        mid_time = 8.0
        dip_mask = df["Metadata_Time"] == df["Metadata_Time"].iloc[
            np.argmin(np.abs(df["Metadata_Time"] - mid_time))
        ]
        df.loc[dip_mask, "Shape_Area"] *= 0.95  # small dip

        m = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=True,
        )
        group = df.groupby(["Metadata_Dataset", "Metadata_Strain"]).get_group(
            ("Test", "Strain1")
        )
        pruned = m._prepare_group(group)
        # Ensure the dip location is still present.
        assert pruned["Metadata_Time"].max() >= mid_time + 1.0

    def test_non_saturating_curve_is_noop(self):
        """Curve ending mid-growth must not be pruned."""
        # Stop well before the curve saturates: t ∈ [0, 6] with smax=50.
        t = np.linspace(0, 6, 15)
        rng = np.random.default_rng(23)
        df = _build_group(
            t, v=5.0, s0=1.0, lam=4.0, alpha=10.0, smax=50.0,
            noise_sigma=0.0, strain="Strain1", rng=rng, n_replicates=1,
        )
        m = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=True,
        )
        group = df.groupby(["Metadata_Dataset", "Metadata_Strain"]).get_group(
            ("Test", "Strain1")
        )
        pruned = m._prepare_group(group)
        # Tail-slope guard must short-circuit: nothing pruned.
        assert len(pruned) == len(group)


# ---------------------------------------------------------------------- #
# Inoculum prior
# ---------------------------------------------------------------------- #
class TestInoculumPrior:
    def test_column_prior_pulls_s0_toward_group_mean(self):
        """A tight inoc_size column far from the data's implied s0 must
        bias the fit toward the column's group mean."""
        t = np.linspace(0, 20, 25)
        rng = np.random.default_rng(30)
        df = _build_group(
            t, v=5.0, s0=3.0, lam=4.0, alpha=10.0, smax=50.0,
            noise_sigma=0.3, strain="Strain1", rng=rng, n_replicates=2,
        )
        # Inoculum measurements clustered tightly around 0.5 —
        # far from the curve's implied s0 ≈ 3.
        df["Inoc_Size"] = rng.normal(0.5, 0.02, size=len(df))

        m_no_prior = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=False,
        )
        m_prior = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            inoc_size_label="Inoc_Size",
            prune_saturated=False,
        )

        s0_no = float(m_no_prior.analyze(df)[LINEAR_SOFTPLUS_MODEL.s0].iloc[0])
        s0_yes = float(m_prior.analyze(df)[LINEAR_SOFTPLUS_MODEL.s0].iloc[0])

        group_mean = float(df["Inoc_Size"].mean())
        # The prior should pull s0 closer to the column's group mean
        # than the no-prior fit.
        assert abs(s0_yes - group_mean) < abs(s0_no - group_mean)

    def test_no_inoc_label_omits_prior_residual(self):
        """Verify residual length: N without prior kwargs, N+1 with them.

        Locks in the symmetry of ``_loss_func`` — a regression in either
        direction (always appending, or never appending) would flip one
        of the two assertions below.
        """
        m = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            prune_saturated=False,
        )
        y = np.asarray([1.0, 2.0, 3.0], dtype=float)
        t_arr = np.asarray([0.0, 1.0, 2.0], dtype=float)
        params = [1.0, 0.5, 1.0, 10.0]

        # Negative: no prior kwargs → residual vector matches y shape.
        res = m._loss_func(params, t_arr, y, smax=5.0)
        assert res.shape == (3,)

        # Positive: prior kwargs appended → residual vector is length N+1.
        res_with_prior = m._loss_func(
            params, t_arr, y, smax=5.0, sGMM_mean=0.5, sGMM_sigma=0.1
        )
        assert res_with_prior.shape == (4,)

    def test_inoc_label_with_nan_stats_omits_prior(self):
        """A zero-variance inoculum column must cause ``_inoc_stats`` to
        return ``None`` so the virtual residual is silently skipped —
        not a silent division by zero that corrupts the fit."""
        t = np.linspace(0, 20, 15)
        rng = np.random.default_rng(32)
        df = _build_group(
            t, v=5.0, s0=1.0, lam=4.0, alpha=10.0, smax=50.0,
            noise_sigma=0.0, strain="Strain1", rng=rng, n_replicates=1,
        )
        df["Inoc_Size"] = 0.5  # zero-variance column
        m = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
            inoc_size_label="Inoc_Size",
            prune_saturated=False,
        )

        # Direct check: construct the group the same way analyze() would
        # (broadcasting per-group mean/std via transform) and assert the
        # resolver returns None on the zero-variance input.
        probe = df.copy()
        grouped = probe.groupby(
            ["Metadata_Dataset", "Metadata_Strain"]
        )["Inoc_Size"]
        probe["Inoc_Size_group_mean"] = grouped.transform("mean")
        probe["Inoc_Size_group_sigma"] = grouped.transform("std")
        group = probe.groupby(
            ["Metadata_Dataset", "Metadata_Strain"]
        ).get_group(("Test", "Strain1"))
        assert m._inoc_stats(group) is None

        # Companion check: analyze() completes and doesn't silently
        # corrupt the fit when the column is degenerate.
        res = m.analyze(df)
        assert np.isfinite(res[LINEAR_SOFTPLUS_MODEL.v].iloc[0])


# ---------------------------------------------------------------------- #
# Degenerate input → NaN row
# ---------------------------------------------------------------------- #
class TestDegenerateInput:
    def test_all_zero_data_does_not_crash(self):
        """All-zero measurements must not raise; analyze returns a row.

        The optimizer converges to some trivial parameter vector here
        (residuals are degenerate), so we only assert that the pipeline
        completes and produces a single row — not that the values are
        NaN.
        """
        t = np.linspace(0, 20, 15)
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
        m = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        res = m.analyze(df)
        assert len(res) == 1
        # v must respect the configured upper bound.
        assert 0.0 <= float(res[LINEAR_SOFTPLUS_MODEL.v].iloc[0]) <= 50.0

    def test_nan_input_triggers_nan_row(self):
        """NaN measurements propagate into the optimizer, triggering the
        NaN fit-column path (``_nan_fit_columns``) instead of crashing."""
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
        m = LinearSoftplusModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset", "Metadata_Strain"],
        )
        res = m.analyze(df)
        assert len(res) == 1
        # Every fitted parameter must be NaN.
        assert np.isnan(float(res[LINEAR_SOFTPLUS_MODEL.v].iloc[0]))
        assert np.isnan(float(res[LINEAR_SOFTPLUS_MODEL.s0].iloc[0]))
        assert np.isnan(float(res[LINEAR_SOFTPLUS_MODEL.lam].iloc[0]))
        assert np.isnan(float(res[LINEAR_SOFTPLUS_MODEL.alpha].iloc[0]))


# ---------------------------------------------------------------------- #
# LogGrowthModel regression — hooks must be no-ops by default
# ---------------------------------------------------------------------- #
def test_log_growth_still_works():
    """Sanity check: adding hooks to the base class didn't break siblings."""
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
