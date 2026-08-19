"""Unit tests for ``RelativeMAD`` robust replicate-agreement quality check.

Synthetic in-memory ``pd.DataFrame`` fixtures exercise the per-``(group,
time)`` MAD/median statistics, the directional pass/warn/fail thresholds,
the NaN guard paths (under-powered, near-zero median), the time-label
absence fallback, and the ``summary()`` / ``group_members()`` helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenotypic.analysis._helper._qc_math import median_abs_deviation
from phenotypic.analysis.qc import RelativeMAD
from phenotypic.schema import IMAGE


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tight_replicates() -> pd.DataFrame:
    """Replicates within ~0.5% of a median of 100 — metric below warn."""
    return pd.DataFrame({
        "Plate": ["P1"] * 6,
        "Metadata_Time": [0, 0, 0, 1, 1, 1],
        "Size_Area": [100.0, 100.5, 99.5, 200.0, 200.5, 199.5],
    })


@pytest.fixture
def dispersed_replicates() -> pd.DataFrame:
    """Replicates with high relative MAD — metric above fail.

    For [1, 5, 9] the median is 5 and the MAD is 4, so the relative MAD
    is 4/5 = 0.8, well above the 0.20 fail threshold.
    """
    return pd.DataFrame({
        "Plate": ["P1"] * 3,
        "Metadata_Time": [0, 0, 0],
        "Size_Area": [1.0, 5.0, 9.0],
    })


# --------------------------------------------------------------------------- #
# Statistical correctness
# --------------------------------------------------------------------------- #


class TestStatisticalCorrectness:
    """Exact-value checks for the median / MAD / metric math."""

    def test_exact_median_and_mad(
        self, dispersed_replicates: pd.DataFrame
    ) -> None:
        chk = RelativeMAD(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(dispersed_replicates)
        assert result["QC_MAD_Median"].iloc[0] == pytest.approx(5.0)
        assert result["QC_MAD_MAD"].iloc[0] == pytest.approx(4.0)

    def test_metric_is_mad_over_abs_median(
        self, dispersed_replicates: pd.DataFrame
    ) -> None:
        chk = RelativeMAD(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(dispersed_replicates)
        values = np.array([1.0, 5.0, 9.0])
        expected = median_abs_deviation(values) / abs(np.median(values))
        assert result["QC_MAD_Metric"].iloc[0] == pytest.approx(expected)

    def test_num_members_count_correct(self) -> None:
        data = pd.DataFrame({
            "Plate": ["P1"] * 6,
            "Metadata_Time": [0, 0, 0, 1, 1, 1],
            "Size_Area": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        chk = RelativeMAD(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert (result["QC_MAD_NumMembers"] == 3).all()


# --------------------------------------------------------------------------- #
# Metric tri-state semantics (higher-is-bad)
# --------------------------------------------------------------------------- #


class TestMetricTriState:
    """Verify pass / warn / fail thresholds at the MAD check's defaults."""

    def test_metric_below_warn(self, tight_replicates: pd.DataFrame) -> None:
        chk = RelativeMAD(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(tight_replicates)
        assert (result["QC_MAD_Metric"] < 0.10).all()
        assert (result["QC_MAD_Status"] == "pass").all()
        assert (~result["QC_MAD_Flag"]).all()

    def test_metric_above_fail(
        self, dispersed_replicates: pd.DataFrame
    ) -> None:
        chk = RelativeMAD(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(dispersed_replicates)
        assert (result["QC_MAD_Metric"] >= 0.20).all()
        assert (result["QC_MAD_Status"] == "fail").all()
        assert result["QC_MAD_Flag"].all()

    def test_metric_in_warn_band(self) -> None:
        # median 100, MAD 15 -> rel-MAD 0.15 (in [0.10, 0.20) warn band).
        data = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "Metadata_Time": [0, 0, 0],
            "Size_Area": [85.0, 100.0, 115.0],
        })
        chk = RelativeMAD(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        metric = result["QC_MAD_Metric"]
        assert (metric >= 0.10).all() and (metric < 0.20).all()
        assert (result["QC_MAD_Status"] == "warn").all()
        assert (~result["QC_MAD_Flag"]).all()


# --------------------------------------------------------------------------- #
# NaN guard paths
# --------------------------------------------------------------------------- #


class TestNanGuardPaths:
    """Under-powered and near-zero-median bins yield NaN metric / pass."""

    def test_min_replicates_nan_guard(self) -> None:
        # One member per (group, time); default min_replicates=2.
        data = pd.DataFrame({
            "Plate": ["P1", "P1"],
            "Metadata_Time": [0, 1],
            "Size_Area": [10.0, 20.0],
        })
        chk = RelativeMAD(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_MAD_Metric"].isna().all()
        assert (result["QC_MAD_Status"] == "pass").all()

    def test_near_zero_median_nan_guard(self) -> None:
        data = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "Metadata_Time": [0, 0, 0],
            "Size_Area": [0.0, 1e-12, -1e-12],
        })
        chk = RelativeMAD(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_MAD_Metric"].isna().all()
        assert (result["QC_MAD_Status"] == "pass").all()

    def test_degenerate_bin_all_zeros(self) -> None:
        data = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "Metadata_Time": [0, 0, 0],
            "Size_Area": [0.0, 0.0, 0.0],
        })
        chk = RelativeMAD(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_MAD_Metric"].isna().all()


# --------------------------------------------------------------------------- #
# Behavioral edges + contract metadata
# --------------------------------------------------------------------------- #


class TestBehavioralEdges:
    """Snapshot data, emitted columns, polarity, thresholds, helpers."""

    def test_missing_time_label_treats_all_as_one_bin(self) -> None:
        data = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "Size_Area": [1.0, 5.0, 9.0],
        })
        chk = RelativeMAD(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_MAD_NumMembers"].nunique() == 1
        assert result["QC_MAD_NumMembers"].iloc[0] == 3
        assert result["QC_MAD_Median"].iloc[0] == pytest.approx(5.0)

    def test_emitted_columns_present(self) -> None:
        data = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "Metadata_Time": [0, 0, 0],
            "Size_Area": [1.0, 5.0, 9.0],
        })
        chk = RelativeMAD(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        for col in [
            "QC_MAD_Median",
            "QC_MAD_MAD",
            "QC_MAD_NumMembers",
            "QC_MAD_Metric",
            "QC_MAD_Flag",
            "QC_MAD_Status",
        ]:
            assert col in result.columns, f"missing column: {col}"

    def test_higher_is_bad_is_true(self) -> None:
        assert RelativeMAD._HIGHER_IS_BAD is True

    def test_metric_col_name(self) -> None:
        assert RelativeMAD.metric_col() == "QC_MAD_Metric"

    def test_exposes_agg_func_is_false(self) -> None:
        assert RelativeMAD._exposes_agg_func is False

    def test_threshold_defaults_and_override(self) -> None:
        chk = RelativeMAD(on="Size_Area", groupby=["Plate"])
        assert chk.warn_threshold == 0.10
        assert chk.fail_threshold == 0.20
        custom = RelativeMAD(
            on="Size_Area",
            groupby=["Plate"],
            warn_threshold=0.05,
            fail_threshold=0.15,
        )
        assert custom.warn_threshold == 0.05
        assert custom.fail_threshold == 0.15

    def test_summary_worst_metric_is_max(self) -> None:
        # Two timepoints in one group; higher-is-bad -> worst = max metric.
        data = pd.DataFrame({
            "Plate": ["P1"] * 6,
            "Metadata_Time": [0, 0, 0, 1, 1, 1],
            "Size_Area": [
                100.0, 100.5, 99.5,  # tight -> low metric
                1.0, 5.0, 9.0,       # dispersed -> high metric
            ],
        })
        chk = RelativeMAD(on="Size_Area", groupby=["Plate"])
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
        # worst metric should reflect the dispersed bin (~0.8), not the tight.
        assert summary["qc_worst_metric"].iloc[0] > 0.20

    def test_group_members_maps_keys_to_member_rows(self) -> None:
        data = pd.DataFrame({
            str(IMAGE.IMAGE_NAME): ["plate1.png"] * 3,
            "Object_Label": [1, 2, 3],
            "Plate": ["P1"] * 3,
            "Metadata_Time": [0, 0, 0],
            "Size_Area": [1.0, 5.0, 9.0],
        })
        chk = RelativeMAD(on="Size_Area", groupby=["Plate"])
        chk.analyze(data)
        members = chk.group_members()
        assert ("P1",) in members
        assert members[("P1",)] == [
            ("plate1.png", 1, 1.0),
            ("plate1.png", 2, 5.0),
            ("plate1.png", 3, 9.0),
        ]
