"""Unit tests for ``ReplicateAgreement`` standard-error quality check.

Synthetic in-memory ``pd.DataFrame`` fixtures exercise the
per-``(group, time)`` summary statistics, the three NaN guard paths
(under-powered, near-zero mean, degenerate-zero), the time-label
absence fallback, and the Plotly ``dash()`` rendering.
"""

from __future__ import annotations

from math import sqrt

import pandas as pd
import plotly.graph_objects as go
import pytest

from phenotypic.analysis.qc import ReplicateAgreement
from phenotypic.schema import METADATA


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def known_three_replicates() -> pd.DataFrame:
    """Single ``(group, time)`` bin with values [1.0, 2.0, 3.0]."""
    return pd.DataFrame({
        "Plate": ["P1"] * 3,
        "MetadataCulture_Time": [0, 0, 0],
        "Size_Area": [1.0, 2.0, 3.0],
    })


@pytest.fixture
def tight_replicates() -> pd.DataFrame:
    """Replicates within ~1% of a mean of 100 — severity well below warn."""
    return pd.DataFrame({
        "Plate": ["P1"] * 6,
        "MetadataCulture_Time": [0, 0, 0, 1, 1, 1],
        "Size_Area": [100.0, 100.5, 99.5, 200.0, 200.5, 199.5],
    })


@pytest.fixture
def dispersed_replicates() -> pd.DataFrame:
    """Replicates with very high relative dispersion — metric above fail."""
    return pd.DataFrame({
        "Plate": ["P1"] * 3,
        "MetadataCulture_Time": [0, 0, 0],
        "Size_Area": [1.0, 5.0, 9.0],
    })


@pytest.fixture
def warn_band_replicates() -> pd.DataFrame:
    """Replicates whose relative-SE metric falls between warn and fail.

    Values chosen so ``metric = stddev / (sqrt(n) * mean)`` lands
    inside ``[0.10, 0.20)``. For [8, 10, 12] at mean=10, n=3:
    ``stddev = 2``, ``SE = 2/sqrt(3) ~= 1.1547``,
    ``metric = 1.1547 / 10 = 0.1155`` (warn band).
    """
    return pd.DataFrame({
        "Plate": ["P1"] * 3,
        "MetadataCulture_Time": [0, 0, 0],
        "Size_Area": [8.0, 10.0, 12.0],
    })


# --------------------------------------------------------------------------- #
# Statistical correctness
# --------------------------------------------------------------------------- #


class TestStatisticalCorrectness:
    """Exact-value checks for SE / Mean / CV math."""

    def test_exact_se_value(self, known_three_replicates: pd.DataFrame) -> None:
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(known_three_replicates)

        assert result["QC_SE_Mean"].iloc[0] == pytest.approx(2.0)
        assert result["QC_SE_Value"].iloc[0] == pytest.approx(1.0 / sqrt(3))

    def test_num_replicates_count_correct(self) -> None:
        data = pd.DataFrame({
            "Plate": ["P1"] * 9,
            "MetadataCulture_Time": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "Size_Area": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        })
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert (result["QC_SE_NumReplicates"] == 3).all()

    def test_cv_calculation(self, known_three_replicates: pd.DataFrame) -> None:
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(known_three_replicates)
        # stddev=1.0, mean=2.0 -> CV = 0.5
        assert result["QC_SE_CV"].iloc[0] == pytest.approx(0.5)

    def test_cv_nan_when_mean_near_zero(self) -> None:
        # Mean falls below default eps=1e-9.
        data = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "MetadataCulture_Time": [0, 0, 0],
            "Size_Area": [0.0, 1e-12, -1e-12],
        })
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_SE_CV"].isna().all()


# --------------------------------------------------------------------------- #
# Metric tri-state semantics
# --------------------------------------------------------------------------- #


class TestMetricTriState:
    """Verify pass / warn / fail thresholds at the SE check's defaults."""

    def test_metric_below_warn(
        self, tight_replicates: pd.DataFrame
    ) -> None:
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(tight_replicates)
        assert (result["QC_SE_Metric"] < 0.10).all()
        assert (result["QC_SE_Status"] == "pass").all()
        assert (~result["QC_SE_Flag"]).all()

    def test_metric_above_fail(
        self, dispersed_replicates: pd.DataFrame
    ) -> None:
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(dispersed_replicates)
        assert (result["QC_SE_Metric"] >= 0.20).all()
        assert (result["QC_SE_Status"] == "fail").all()
        assert result["QC_SE_Flag"].all()

    def test_metric_in_warn_band(
        self, warn_band_replicates: pd.DataFrame
    ) -> None:
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(warn_band_replicates)
        metric = result["QC_SE_Metric"]
        assert (metric >= 0.10).all() and (metric < 0.20).all()
        assert (result["QC_SE_Status"] == "warn").all()
        assert (~result["QC_SE_Flag"]).all()


# --------------------------------------------------------------------------- #
# NaN guard paths
# --------------------------------------------------------------------------- #


class TestNanGuardPaths:
    """The three documented guard paths must yield NaN metric / pass."""

    def test_min_replicates_nan_guard(self) -> None:
        # One replicate per (group, time); default min_replicates=2.
        data = pd.DataFrame({
            "Plate": ["P1", "P1"],
            "MetadataCulture_Time": [0, 1],
            "Size_Area": [10.0, 20.0],
        })
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_SE_Metric"].isna().all()
        # NaN metric → "pass" status per base class.
        assert (result["QC_SE_Status"] == "pass").all()

    def test_eps_guard_for_near_zero_mean(self) -> None:
        data = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "MetadataCulture_Time": [0, 0, 0],
            "Size_Area": [0.0, 1e-12, -1e-12],
        })
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_SE_Metric"].isna().all()

    def test_degenerate_bin_all_zeros(self) -> None:
        data = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "MetadataCulture_Time": [0, 0, 0],
            "Size_Area": [0.0, 0.0, 0.0],
        })
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_SE_Metric"].isna().all()
        assert (result["QC_SE_Status"] == "pass").all()

    def test_min_replicates_three_with_two_replicates_is_nan(self) -> None:
        data = pd.DataFrame({
            "Plate": ["P1"] * 2,
            "MetadataCulture_Time": [0, 0],
            "Size_Area": [10.0, 12.0],
        })
        chk = ReplicateAgreement(
            on="Size_Area", groupby=["Plate"], min_replicates=3
        )
        result = chk.analyze(data)
        assert result["QC_SE_Metric"].isna().all()


# --------------------------------------------------------------------------- #
# Behavioral edge cases
# --------------------------------------------------------------------------- #


class TestBehavioralEdges:
    """Snapshot data, emitted columns, and ``_exposes_agg_func`` flag."""

    def test_missing_time_label_treats_all_as_one_bin(self) -> None:
        # No "MetadataCulture_Time" column → entire group is one bin.
        data = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "Size_Area": [1.0, 2.0, 3.0],
        })
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        # All three rows share the same per-bin scalars.
        assert result["QC_SE_NumReplicates"].nunique() == 1
        assert result["QC_SE_NumReplicates"].iloc[0] == 3
        assert result["QC_SE_Mean"].iloc[0] == pytest.approx(2.0)
        assert result["QC_SE_Value"].iloc[0] == pytest.approx(1.0 / sqrt(3))

    def test_emitted_columns_present(
        self, known_three_replicates: pd.DataFrame
    ) -> None:
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(known_three_replicates)
        expected = [
            "QC_SE_Value",
            "QC_SE_Mean",
            "QC_SE_CV",
            "QC_SE_NumReplicates",
            "QC_SE_Metric",
            "QC_SE_Flag",
            "QC_SE_Status",
        ]
        for col in expected:
            assert col in result.columns, f"missing column: {col}"

    def test_exposes_agg_func_is_false(self) -> None:
        assert ReplicateAgreement._exposes_agg_func is False

    def test_higher_is_bad_is_true(self) -> None:
        assert ReplicateAgreement._HIGHER_IS_BAD is True

    def test_metric_col_returns_metric_name(self) -> None:
        assert ReplicateAgreement.metric_col() == "QC_SE_Metric"

    def test_threshold_defaults_and_override(self) -> None:
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        assert chk.warn_threshold == 0.10
        assert chk.fail_threshold == 0.20
        custom = ReplicateAgreement(
            on="Size_Area",
            groupby=["Plate"],
            warn_threshold=0.05,
            fail_threshold=0.15,
        )
        assert custom.warn_threshold == 0.05
        assert custom.fail_threshold == 0.15

    def test_flagged_keys_returns_image_file_object_label_pairs(self) -> None:
        # Build dispersed (failing) data carrying ImageFile + Object_Label.
        data = pd.DataFrame({
            str(METADATA.IMAGE_NAME): ["plate1.png"] * 3,
            "Object_Label": [1, 2, 3],
            "Plate": ["P1"] * 3,
            "MetadataCulture_Time": [0, 0, 0],
            "Size_Area": [1.0, 5.0, 9.0],
        })
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        chk.analyze(data)
        flagged = chk.flagged_keys()
        assert flagged == [
            ("plate1.png", 1),
            ("plate1.png", 2),
            ("plate1.png", 3),
        ]


# --------------------------------------------------------------------------- #
# dash()
# --------------------------------------------------------------------------- #


class TestDash:
    """Plotly figure construction and the pre-analyze guard."""

    def test_dash_returns_plotly_figure(
        self, tight_replicates: pd.DataFrame
    ) -> None:
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        chk.analyze(tight_replicates)
        fig = chk.dash()
        assert isinstance(fig, go.Figure)
        # At least one trace per group.
        assert len(fig.data) >= 1

    def test_dash_raises_before_analyze(self) -> None:
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        with pytest.raises(RuntimeError, match="call analyze\\(\\) first"):
            chk.dash()
