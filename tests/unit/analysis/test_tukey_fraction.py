"""Unit tests for ``TukeyOutlierFraction`` detection quality check.

Synthetic in-memory ``pd.DataFrame`` fixtures exercise the per-``(group,
time)`` outlier-fraction metric, the directional pass/warn/fail
thresholds, the NaN guard path (under-powered bins below
``min_replicates=4``), the ``k`` multiplier, the time-label absence
fallback, and the ``summary()`` / ``group_members()`` helpers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from phenotypic.analysis._helper._qc_math import tukey_outlier_fraction
from phenotypic.analysis.qc import TukeyOutlierFraction
from phenotypic.schema import METADATA


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def clean_bin() -> pd.DataFrame:
    """Eight evenly-spread members with no Tukey outliers."""
    return pd.DataFrame({
        "Plate": ["P1"] * 8,
        "Metadata_Time": [0] * 8,
        "Size_Area": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
    })


@pytest.fixture
def many_outliers() -> pd.DataFrame:
    """Twelve members, nine tight and three extreme spikes — fraction 0.25.

    Tukey detects a *minority* of outliers: a tight majority pins the IQR
    so the three isolated high spikes sit beyond the upper fence, giving a
    3/12 = 0.25 outlier fraction that meets the 0.25 fail threshold.
    """
    return pd.DataFrame({
        "Plate": ["P1"] * 12,
        "Metadata_Time": [0] * 12,
        "Size_Area": [
            10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 14.0,
            300.0, 400.0, 500.0,
        ],
    })


# --------------------------------------------------------------------------- #
# Statistical correctness
# --------------------------------------------------------------------------- #


class TestStatisticalCorrectness:
    """Exact-value checks for the outlier-fraction math."""

    def test_metric_matches_helper(
        self, many_outliers: pd.DataFrame
    ) -> None:
        chk = TukeyOutlierFraction(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(many_outliers)
        values = many_outliers["Size_Area"].to_numpy(dtype=float)
        expected = tukey_outlier_fraction(values, 1.5)
        assert result["QC_Tukey_Metric"].iloc[0] == pytest.approx(expected)

    def test_emits_fences_and_counts(
        self, many_outliers: pd.DataFrame
    ) -> None:
        chk = TukeyOutlierFraction(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(many_outliers)
        assert result["QC_Tukey_NumMembers"].iloc[0] == 12
        assert result["QC_Tukey_NumOutliers"].iloc[0] >= 1
        assert (
            result["QC_Tukey_LowerFence"].iloc[0]
            < result["QC_Tukey_UpperFence"].iloc[0]
        )

    def test_clean_bin_zero_fraction(self, clean_bin: pd.DataFrame) -> None:
        chk = TukeyOutlierFraction(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(clean_bin)
        assert result["QC_Tukey_Metric"].iloc[0] == pytest.approx(0.0)
        assert result["QC_Tukey_NumOutliers"].iloc[0] == 0


# --------------------------------------------------------------------------- #
# Metric tri-state semantics (higher-is-bad)
# --------------------------------------------------------------------------- #


class TestMetricTriState:
    """Verify pass / warn / fail thresholds at the Tukey defaults (0.10/0.25)."""

    def test_clean_bin_passes(self, clean_bin: pd.DataFrame) -> None:
        chk = TukeyOutlierFraction(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(clean_bin)
        assert (result["QC_Tukey_Status"] == "pass").all()
        assert (~result["QC_Tukey_Flag"]).all()

    def test_many_outliers_fail(self, many_outliers: pd.DataFrame) -> None:
        chk = TukeyOutlierFraction(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(many_outliers)
        assert (result["QC_Tukey_Metric"] >= 0.25).all()
        assert (result["QC_Tukey_Status"] == "fail").all()
        assert result["QC_Tukey_Flag"].all()

    def test_single_outlier_in_ten_warns(self) -> None:
        # 1/10 = 0.10 outlier fraction -> warn band [0.10, 0.25).
        data = pd.DataFrame({
            "Plate": ["P1"] * 10,
            "Metadata_Time": [0] * 10,
            "Size_Area": [
                10.0, 11.0, 12.0, 13.0, 14.0,
                10.5, 11.5, 12.5, 13.5, 500.0,
            ],
        })
        chk = TukeyOutlierFraction(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_Tukey_Metric"].iloc[0] == pytest.approx(0.10)
        assert (result["QC_Tukey_Status"] == "warn").all()
        assert (~result["QC_Tukey_Flag"]).all()


# --------------------------------------------------------------------------- #
# NaN guard path
# --------------------------------------------------------------------------- #


class TestNanGuardPath:
    """Bins below ``min_replicates`` yield NaN metric / pass."""

    def test_under_powered_bin_is_nan(self) -> None:
        # Three members, default min_replicates=4.
        data = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "Metadata_Time": [0, 0, 0],
            "Size_Area": [10.0, 11.0, 12.0],
        })
        chk = TukeyOutlierFraction(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_Tukey_Metric"].isna().all()
        assert (result["QC_Tukey_Status"] == "pass").all()

    def test_min_replicates_override(self) -> None:
        # Lowering min_replicates lets a 3-member bin compute a metric.
        data = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "Metadata_Time": [0, 0, 0],
            "Size_Area": [10.0, 11.0, 12.0],
        })
        chk = TukeyOutlierFraction(
            on="Size_Area", groupby=["Plate"], min_replicates=3
        )
        result = chk.analyze(data)
        assert result["QC_Tukey_Metric"].notna().all()


# --------------------------------------------------------------------------- #
# Behavioral edges + contract metadata
# --------------------------------------------------------------------------- #


class TestBehavioralEdges:
    """Snapshot data, k multiplier, emitted columns, polarity, helpers."""

    def test_k_multiplier_widens_fences(
        self, many_outliers: pd.DataFrame
    ) -> None:
        # k=3.0 widens the fences so fewer members are flagged.
        default_k = TukeyOutlierFraction(
            on="Size_Area", groupby=["Plate"]
        ).analyze(many_outliers)["QC_Tukey_NumOutliers"].iloc[0]
        wide_k = TukeyOutlierFraction(
            on="Size_Area", groupby=["Plate"], k=3.0
        ).analyze(many_outliers)["QC_Tukey_NumOutliers"].iloc[0]
        assert wide_k <= default_k

    def test_missing_time_label_treats_all_as_one_bin(
        self, clean_bin: pd.DataFrame
    ) -> None:
        data = clean_bin.drop(columns=["Metadata_Time"])
        chk = TukeyOutlierFraction(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_Tukey_NumMembers"].nunique() == 1
        assert result["QC_Tukey_NumMembers"].iloc[0] == 8

    def test_emitted_columns_present(self, clean_bin: pd.DataFrame) -> None:
        chk = TukeyOutlierFraction(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(clean_bin)
        for col in [
            "QC_Tukey_LowerFence",
            "QC_Tukey_UpperFence",
            "QC_Tukey_NumOutliers",
            "QC_Tukey_NumMembers",
            "QC_Tukey_Metric",
            "QC_Tukey_Flag",
            "QC_Tukey_Status",
        ]:
            assert col in result.columns, f"missing column: {col}"

    def test_higher_is_bad_is_true(self) -> None:
        assert TukeyOutlierFraction._HIGHER_IS_BAD is True

    def test_metric_col_name(self) -> None:
        assert TukeyOutlierFraction.metric_col() == "QC_Tukey_Metric"

    def test_exposes_agg_func_is_false(self) -> None:
        assert TukeyOutlierFraction._exposes_agg_func is False

    def test_default_min_replicates_is_four(self) -> None:
        chk = TukeyOutlierFraction(on="Size_Area", groupby=["Plate"])
        assert chk.min_replicates == 4

    def test_summary_worst_metric_is_max(
        self, many_outliers: pd.DataFrame
    ) -> None:
        chk = TukeyOutlierFraction(on="Size_Area", groupby=["Plate"])
        chk.analyze(many_outliers)
        summary = chk.summary()
        assert list(summary.columns) == [
            "Plate",
            "qc_n_members",
            "qc_n_flagged",
            "qc_worst_metric",
            "qc_status",
        ]
        assert summary["qc_status"].iloc[0] == "fail"
        assert summary["qc_worst_metric"].iloc[0] >= 0.25

    def test_group_members_maps_keys_to_member_rows(self) -> None:
        data = pd.DataFrame({
            str(METADATA.IMAGE_NAME): ["plate1.png"] * 4,
            "Object_Label": [1, 2, 3, 4],
            "Plate": ["P1"] * 4,
            "Metadata_Time": [0, 0, 0, 0],
            "Size_Area": [10.0, 11.0, 12.0, 13.0],
        })
        chk = TukeyOutlierFraction(on="Size_Area", groupby=["Plate"])
        chk.analyze(data)
        members = chk.group_members()
        assert members[("P1",)] == [
            ("plate1.png", 1, 10.0),
            ("plate1.png", 2, 11.0),
            ("plate1.png", 3, 12.0),
            ("plate1.png", 4, 13.0),
        ]
