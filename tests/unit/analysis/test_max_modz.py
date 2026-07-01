"""Unit tests for ``MaxModifiedZScore`` outlier quality check.

Synthetic in-memory ``pd.DataFrame`` fixtures exercise the per-``(group,
time)`` maximum modified Z-score metric, the directional pass/warn/fail
thresholds, the NaN guard paths (under-powered bins, perfectly-identical
bins), the time-label absence fallback, and the ``summary()`` /
``group_members()`` helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenotypic.analysis.qc import MaxModifiedZScore
from phenotypic.analysis._helper._qc_math import modified_z_scores
from phenotypic.schema import METADATA


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def clean_bin() -> pd.DataFrame:
    """Tightly-clustered members — every modified Z-score below warn."""
    return pd.DataFrame({
        "Plate": ["P1"] * 6,
        "Metadata_Time": [0] * 6,
        "Size_Area": [10.0, 10.1, 9.9, 10.05, 9.95, 10.0],
    })


@pytest.fixture
def one_extreme_outlier() -> pd.DataFrame:
    """One member far from the rest — max modified Z above fail."""
    return pd.DataFrame({
        "Plate": ["P1"] * 6,
        "Metadata_Time": [0] * 6,
        "Size_Area": [10.0, 10.1, 9.9, 10.05, 9.95, 200.0],
    })


# --------------------------------------------------------------------------- #
# Statistical correctness
# --------------------------------------------------------------------------- #


class TestStatisticalCorrectness:
    """Exact-value checks for the max-modified-Z math."""

    def test_metric_equals_nanmax_of_modified_z(
        self, one_extreme_outlier: pd.DataFrame
    ) -> None:
        chk = MaxModifiedZScore(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(one_extreme_outlier)
        values = one_extreme_outlier["Size_Area"].to_numpy(dtype=float)
        expected = float(np.nanmax(modified_z_scores(values)))
        assert result["QC_ZMax_Metric"].iloc[0] == pytest.approx(expected)

    def test_emits_median_and_mad(
        self, one_extreme_outlier: pd.DataFrame
    ) -> None:
        chk = MaxModifiedZScore(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(one_extreme_outlier)
        values = one_extreme_outlier["Size_Area"].to_numpy(dtype=float)
        assert result["QC_ZMax_Median"].iloc[0] == pytest.approx(
            float(np.median(values))
        )
        assert result["QC_ZMax_NumMembers"].iloc[0] == 6


# --------------------------------------------------------------------------- #
# Metric tri-state semantics (higher-is-bad)
# --------------------------------------------------------------------------- #


class TestMetricTriState:
    """Verify pass / warn / fail thresholds at the ZMax defaults (3.5/5.0)."""

    def test_clean_bin_passes(self, clean_bin: pd.DataFrame) -> None:
        chk = MaxModifiedZScore(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(clean_bin)
        assert (result["QC_ZMax_Status"] == "pass").all()
        assert (~result["QC_ZMax_Flag"]).all()

    def test_extreme_outlier_fails(
        self, one_extreme_outlier: pd.DataFrame
    ) -> None:
        chk = MaxModifiedZScore(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(one_extreme_outlier)
        assert (result["QC_ZMax_Metric"] >= 5.0).all()
        assert (result["QC_ZMax_Status"] == "fail").all()
        assert result["QC_ZMax_Flag"].all()

    def test_threshold_override_moves_band(
        self, one_extreme_outlier: pd.DataFrame
    ) -> None:
        # Raise thresholds so the same outlier lands only in warn.
        metric = MaxModifiedZScore(
            on="Size_Area", groupby=["Plate"]
        ).analyze(one_extreme_outlier)["QC_ZMax_Metric"].iloc[0]
        chk = MaxModifiedZScore(
            on="Size_Area",
            groupby=["Plate"],
            warn_threshold=metric - 0.1,
            fail_threshold=metric + 0.1,
        )
        result = chk.analyze(one_extreme_outlier)
        assert (result["QC_ZMax_Status"] == "warn").all()
        assert (~result["QC_ZMax_Flag"]).all()


# --------------------------------------------------------------------------- #
# NaN guard paths
# --------------------------------------------------------------------------- #


class TestNanGuardPaths:
    """Under-powered and perfectly-identical bins yield NaN metric / pass."""

    def test_min_replicates_nan_guard(self) -> None:
        data = pd.DataFrame({
            "Plate": ["P1", "P1"],
            "Metadata_Time": [0, 1],
            "Size_Area": [10.0, 20.0],
        })
        chk = MaxModifiedZScore(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_ZMax_Metric"].isna().all()
        assert (result["QC_ZMax_Status"] == "pass").all()

    def test_all_identical_is_nan_not_outlier(self) -> None:
        # MAD and mean-AD are both zero -> all scores zero -> NaN metric.
        data = pd.DataFrame({
            "Plate": ["P1"] * 5,
            "Metadata_Time": [0] * 5,
            "Size_Area": [42.0] * 5,
        })
        chk = MaxModifiedZScore(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_ZMax_Metric"].isna().all()
        assert (result["QC_ZMax_Status"] == "pass").all()


# --------------------------------------------------------------------------- #
# Behavioral edges + contract metadata
# --------------------------------------------------------------------------- #


class TestBehavioralEdges:
    """Snapshot data, emitted columns, polarity, helpers."""

    def test_missing_time_label_treats_all_as_one_bin(
        self, one_extreme_outlier: pd.DataFrame
    ) -> None:
        data = one_extreme_outlier.drop(columns=["Metadata_Time"])
        chk = MaxModifiedZScore(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_ZMax_NumMembers"].nunique() == 1
        assert result["QC_ZMax_NumMembers"].iloc[0] == 6

    def test_emitted_columns_present(
        self, clean_bin: pd.DataFrame
    ) -> None:
        chk = MaxModifiedZScore(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(clean_bin)
        for col in [
            "QC_ZMax_Median",
            "QC_ZMax_MAD",
            "QC_ZMax_NumMembers",
            "QC_ZMax_Metric",
            "QC_ZMax_Flag",
            "QC_ZMax_Status",
        ]:
            assert col in result.columns, f"missing column: {col}"

    def test_higher_is_bad_is_true(self) -> None:
        assert MaxModifiedZScore._HIGHER_IS_BAD is True

    def test_metric_col_name(self) -> None:
        assert MaxModifiedZScore.metric_col() == "QC_ZMax_Metric"

    def test_exposes_agg_func_is_false(self) -> None:
        assert MaxModifiedZScore._exposes_agg_func is False

    def test_summary_worst_metric_is_max(self) -> None:
        data = pd.DataFrame({
            "Plate": ["P1"] * 12,
            "Metadata_Time": [0] * 6 + [1] * 6,
            "Size_Area": (
                [10.0, 10.1, 9.9, 10.05, 9.95, 10.0]  # clean
                + [10.0, 10.1, 9.9, 10.05, 9.95, 200.0]  # outlier
            ),
        })
        chk = MaxModifiedZScore(on="Size_Area", groupby=["Plate"])
        chk.analyze(data)
        summary = chk.summary()
        assert list(summary.columns) == [
            "Plate",
            "qc_n_members",
            "qc_n_flagged",
            "qc_worst_metric",
            "qc_status",
        ]
        assert summary["qc_status"].iloc[0] == "fail"
        assert summary["qc_worst_metric"].iloc[0] >= 5.0

    def test_group_members_maps_keys_to_member_rows(self) -> None:
        data = pd.DataFrame({
            str(METADATA.IMAGE_NAME): ["plate1.png"] * 3,
            "Object_Label": [1, 2, 3],
            "Plate": ["P1"] * 3,
            "Metadata_Time": [0, 0, 0],
            "Size_Area": [10.0, 11.0, 12.0],
        })
        chk = MaxModifiedZScore(on="Size_Area", groupby=["Plate"])
        chk.analyze(data)
        members = chk.group_members()
        assert members[("P1",)] == [
            ("plate1.png", 1, 10.0),
            ("plate1.png", 2, 11.0),
            ("plate1.png", 3, 12.0),
        ]


def test_qc_math_moved_to_helper():
    from phenotypic.analysis._helper._qc_math import modified_z_scores
    from phenotypic.analysis._helper import render_error_analysis_report
    import importlib
    import pytest
    # old private path is gone (hard cutover)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("phenotypic.analysis._qc_math")
    assert callable(modified_z_scores)
    assert callable(render_error_analysis_report)
