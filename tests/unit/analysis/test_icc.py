"""Unit tests for the ``ICC`` replicate-reliability quality check.

Synthetic in-memory ``pd.DataFrame`` fixtures exercise the ICC(2,1)
two-way random absolute-agreement estimator (validated against the
canonical Shrout & Fleiss reference value), the **lower-is-bad**
directional thresholds, the NaN guard paths (missing axis column → LOUD
``unmatched_groups``; incomplete/duplicated matrix; too-few
subjects/raters; zero variance), and the ``summary()`` /
``group_members()`` helpers. ICC defaults to a repeated-measures design —
subjects are ``Metadata_Time``, raters are ``Metadata_Replicate``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenotypic.analysis.qc import ICC


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_group(matrix: np.ndarray, plate: str = "P1") -> pd.DataFrame:
    """Build a long-form QC frame from a subjects x raters matrix.

    Rows are subjects (the default ``Metadata_Time`` repeated-measure axis),
    columns are raters (``Metadata_Replicate``).
    """
    rows = []
    n_subjects, n_raters = matrix.shape
    for s in range(n_subjects):
        for r in range(n_raters):
            rows.append({
                "Plate": plate,
                "Metadata_ImageFile": f"{plate}_t{s}.png",
                "Object_Label": s * n_raters + r + 1,
                "Metadata_Time": s,
                "Metadata_Replicate": r + 1,
                "Size_Area": float(matrix[s, r]),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def high_agreement() -> pd.DataFrame:
    """Three subjects x three raters with near-perfect agreement (ICC ~1)."""
    matrix = np.array([
        [10.0, 10.1, 9.9],
        [20.0, 20.2, 19.8],
        [40.0, 40.1, 39.9],
    ])
    return _make_group(matrix)


@pytest.fixture
def low_agreement() -> pd.DataFrame:
    """Subjects with raters that disagree wildly (ICC well below 0.50)."""
    matrix = np.array([
        [10.0, 90.0, 30.0],
        [80.0, 15.0, 70.0],
        [25.0, 60.0, 5.0],
    ])
    return _make_group(matrix)


# --------------------------------------------------------------------------- #
# Statistical correctness
# --------------------------------------------------------------------------- #


class TestStatisticalCorrectness:
    """Exact-value checks for the ICC(2,1) estimator."""

    def test_shrout_fleiss_reference_value(self) -> None:
        # Shrout & Fleiss (1979) Table 2; pingouin reports ICC2 = 0.2898.
        matrix = np.array([
            [9, 2, 5, 8],
            [6, 1, 3, 2],
            [8, 4, 6, 8],
            [7, 1, 2, 6],
            [10, 5, 6, 9],
            [6, 2, 4, 7],
        ], dtype=float)
        assert ICC._icc_2_1(matrix) == pytest.approx(0.2898, abs=1e-4)

    def test_high_agreement_icc_near_one(
        self, high_agreement: pd.DataFrame
    ) -> None:
        chk = ICC(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(high_agreement)
        assert result["QC_ICC_Metric"].iloc[0] > 0.99

    def test_metric_is_broadcast_to_all_rows(
        self, high_agreement: pd.DataFrame
    ) -> None:
        chk = ICC(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(high_agreement)
        assert result["QC_ICC_Metric"].nunique() == 1
        assert len(result) == len(high_agreement)

    def test_emits_subject_and_rater_counts(
        self, high_agreement: pd.DataFrame
    ) -> None:
        chk = ICC(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(high_agreement)
        assert result["QC_ICC_NumSubjects"].iloc[0] == 3
        assert result["QC_ICC_NumRaters"].iloc[0] == 3
        assert result["QC_ICC_NumMembers"].iloc[0] == 9


# --------------------------------------------------------------------------- #
# Lower-is-bad tri-state semantics
# --------------------------------------------------------------------------- #


class TestLowerIsBadTriState:
    """ICC is an agreement score: smaller is worse (warn <=0.75, fail <=0.50)."""

    def test_high_agreement_passes(
        self, high_agreement: pd.DataFrame
    ) -> None:
        chk = ICC(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(high_agreement)
        assert (result["QC_ICC_Status"] == "pass").all()
        assert (~result["QC_ICC_Flag"]).all()

    def test_low_agreement_fails(self, low_agreement: pd.DataFrame) -> None:
        chk = ICC(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(low_agreement)
        assert (result["QC_ICC_Metric"] <= 0.50).all()
        assert (result["QC_ICC_Status"] == "fail").all()
        assert result["QC_ICC_Flag"].all()

    def test_polarity_is_lower_is_bad(self) -> None:
        assert ICC._HIGHER_IS_BAD is False


# --------------------------------------------------------------------------- #
# NaN guard paths
# --------------------------------------------------------------------------- #


class TestNanGuardPaths:
    """Degenerate two-way designs yield NaN metric / pass."""

    def test_missing_rater_column_is_nan_and_loud(self) -> None:
        data = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "Metadata_Time": [0, 1, 2],
            "Size_Area": [10.0, 20.0, 40.0],
        })
        chk = ICC(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_ICC_Metric"].isna().all()
        assert (result["QC_ICC_Status"] == "pass").all()
        # LOUD: a not-evaluated check records the group key.
        assert chk.unmatched_groups == [("P1",)]

    def test_missing_subject_column_is_nan_and_loud(self) -> None:
        data = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "Metadata_Replicate": [1, 2, 3],
            "Size_Area": [10.0, 20.0, 40.0],
        })
        chk = ICC(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_ICC_Metric"].isna().all()
        assert chk.unmatched_groups == [("P1",)]

    def test_unmatched_groups_reset_between_runs(self) -> None:
        missing = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "Metadata_Time": [0, 1, 2],
            "Size_Area": [10.0, 20.0, 40.0],
        })
        chk = ICC(on="Size_Area", groupby=["Plate"])
        chk.analyze(missing)
        assert chk.unmatched_groups == [("P1",)]
        # A subsequent complete frame must clear the stale unmatched entry.
        complete = _make_group(np.array([[10.0, 10.1], [20.0, 20.2]]))
        chk.analyze(complete)
        assert chk.unmatched_groups == []

    def test_incomplete_matrix_is_nan_not_loud(self) -> None:
        # Duplicate (Time=0, Replicate=1) makes the design ambiguous.
        data = pd.DataFrame({
            "Plate": ["P1"] * 5,
            "Metadata_Time": [0, 0, 1, 1, 0],
            "Metadata_Replicate": [1, 2, 1, 2, 1],
            "Size_Area": [10.0, 11.0, 20.0, 21.0, 99.0],
        })
        chk = ICC(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_ICC_Metric"].isna().all()
        # Both axis columns are present -> genuine insufficient-data, NOT
        # an unmatched (axis-missing) case.
        assert chk.unmatched_groups == []

    def test_missing_cell_is_nan(self) -> None:
        # 2 subjects x 2 raters but one cell absent.
        data = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "Metadata_Time": [0, 0, 1],
            "Metadata_Replicate": [1, 2, 1],
            "Size_Area": [10.0, 11.0, 20.0],
        })
        chk = ICC(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_ICC_Metric"].isna().all()
        assert chk.unmatched_groups == []

    def test_single_rater_is_nan(self) -> None:
        # 3 subjects but only 1 rater -> k < 2.
        data = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "Metadata_Time": [0, 1, 2],
            "Metadata_Replicate": [1, 1, 1],
            "Size_Area": [10.0, 20.0, 40.0],
        })
        chk = ICC(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_ICC_Metric"].isna().all()
        assert chk.unmatched_groups == []

    def test_single_subject_is_nan(self) -> None:
        # 1 subject but 3 raters -> n < 2.
        data = pd.DataFrame({
            "Plate": ["P1"] * 3,
            "Metadata_Time": [0, 0, 0],
            "Metadata_Replicate": [1, 2, 3],
            "Size_Area": [10.0, 20.0, 40.0],
        })
        chk = ICC(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_ICC_Metric"].isna().all()
        assert chk.unmatched_groups == []

    def test_zero_variance_is_nan(self) -> None:
        matrix = np.full((3, 3), 42.0)
        data = _make_group(matrix)
        chk = ICC(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        assert result["QC_ICC_Metric"].isna().all()
        assert (result["QC_ICC_Status"] == "pass").all()


# --------------------------------------------------------------------------- #
# Disagreement / absolute-agreement semantics
# --------------------------------------------------------------------------- #


class TestDisagreementSemantics:
    """Genuine disagreement flags; ICC(2,1) is absolute agreement."""

    def test_anticorrelated_raters_negative_icc_fails(self) -> None:
        # Raters anti-correlate across subjects -> negative ICC (valid),
        # which must flag as fail.
        data = pd.DataFrame({
            "Plate": ["P1"] * 6,
            "Metadata_Time": [0, 0, 1, 1, 2, 2],
            "Metadata_Replicate": [1, 2, 1, 2, 1, 2],
            "Size_Area": [10.0, 40.0, 40.0, 10.0, 25.0, 25.0],
        })
        chk = ICC(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        metric = float(result["QC_ICC_Metric"].iloc[0])
        assert np.isfinite(metric)
        assert metric < 0.0  # negative ICC is valid, not an error
        assert (result["QC_ICC_Status"] == "fail").all()

    def test_systematic_offset_low_abs_agreement_fails(self) -> None:
        # Raters track the same subject trend but rater 2 sits ~100 units
        # higher. Consistency would be ~1, but ICC(2,1) is ABSOLUTE
        # agreement, so a fixed offset is correctly near-zero and fails.
        data = pd.DataFrame({
            "Plate": ["P1"] * 6,
            "Metadata_Time": [0, 0, 1, 1, 2, 2],
            "Metadata_Replicate": [1, 2, 1, 2, 1, 2],
            "Size_Area": [10.0, 110.0, 20.0, 120.0, 30.0, 130.0],
        })
        chk = ICC(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(data)
        metric = float(result["QC_ICC_Metric"].iloc[0])
        assert np.isfinite(metric)
        assert metric < 0.5  # below the 0.50 fail threshold
        assert (result["QC_ICC_Status"] == "fail").all()


# --------------------------------------------------------------------------- #
# Realistic data — packaged all_meas.csv
# --------------------------------------------------------------------------- #


class TestRealDataAllMeas:
    """ICC against the packaged ``all_meas.csv`` (real plate measurements).

    The CSV carries ``Metadata_Time`` (5 timepoints), ``Metadata_Replicate``,
    and ``Metadata_Strain`` with ``Shape_Area`` measurements. Its per-strain
    Time x Replicate grids are **sparse** (missing cells, no usable complete
    grid), so the default-axis check correctly NaNs them — the incomplete
    guard, not a silent green pass. Restricting to a complete sub-grid of the
    same data yields a finite ICC, demonstrating the estimator on real values.
    """

    def test_sparse_real_grids_are_nan_not_silent_pass(self) -> None:
        from phenotypic.data import load_meas

        df = load_meas()
        chk = ICC(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
            subject_label="Metadata_Time",
            rater_label="Metadata_Replicate",
        )
        result = chk.analyze(df)
        # Both axis columns are present, so this is the incomplete-design
        # guard (NaN), not the missing-axis (unmatched) path.
        assert result["QC_ICC_Metric"].isna().all()
        assert chk.unmatched_groups == []

    def test_complete_subgrid_yields_finite_icc(self) -> None:
        from phenotypic.data import load_meas

        df = load_meas()
        strain = "CBS11445"
        sub = df[df["Metadata_Strain"] == strain]
        # Keep only replicate columns observed at every timepoint -> a
        # complete (well-populated) Time x Replicate sub-grid.
        pivot = sub.pivot_table(
            index="Metadata_Time",
            columns="Metadata_Replicate",
            values="Shape_Area",
            aggfunc="first",
        )
        complete_reps = pivot.columns[~pivot.isna().any(axis=0)]
        assert len(complete_reps) >= 2, "fixture data lost its complete grid"
        well_populated = sub[sub["Metadata_Replicate"].isin(complete_reps)]

        chk = ICC(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
            subject_label="Metadata_Time",
            rater_label="Metadata_Replicate",
        )
        result = chk.analyze(well_populated)
        metric = float(result["QC_ICC_Metric"].iloc[0])
        assert np.isfinite(metric)
        assert result["QC_ICC_NumSubjects"].iloc[0] == 5
        assert result["QC_ICC_NumRaters"].iloc[0] == len(complete_reps)


# --------------------------------------------------------------------------- #
# Behavioral edges + contract metadata
# --------------------------------------------------------------------------- #


class TestBehavioralEdges:
    """Emitted columns, custom axis labels, thresholds, helpers."""

    def test_emitted_columns_present(
        self, high_agreement: pd.DataFrame
    ) -> None:
        chk = ICC(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(high_agreement)
        for col in [
            "QC_ICC_NumSubjects",
            "QC_ICC_NumRaters",
            "QC_ICC_NumMembers",
            "QC_ICC_Metric",
            "QC_ICC_Flag",
            "QC_ICC_Status",
        ]:
            assert col in result.columns, f"missing column: {col}"

    def test_metric_col_name(self) -> None:
        assert ICC.metric_col() == "QC_ICC_Metric"

    def test_exposes_agg_func_is_false(self) -> None:
        assert ICC._exposes_agg_func is False

    def test_default_axis_labels(self) -> None:
        chk = ICC(on="Size_Area", groupby=["Plate"])
        assert chk.subject_label == "Metadata_Time"
        assert chk.rater_label == "Metadata_Replicate"

    def test_threshold_defaults(self) -> None:
        chk = ICC(on="Size_Area", groupby=["Plate"])
        assert chk.warn_threshold == 0.75
        assert chk.fail_threshold == 0.50

    def test_custom_axis_labels(self) -> None:
        matrix = np.array([
            [10.0, 10.1],
            [20.0, 20.2],
            [40.0, 40.1],
        ])
        data = pd.DataFrame({
            "Plate": ["P1"] * 6,
            "Day": [0, 0, 1, 1, 2, 2],
            "Rep": [1, 2, 1, 2, 1, 2],
            "Size_Area": matrix.reshape(-1),
        })
        chk = ICC(
            on="Size_Area",
            groupby=["Plate"],
            subject_label="Day",
            rater_label="Rep",
        )
        result = chk.analyze(data)
        assert result["QC_ICC_NumSubjects"].iloc[0] == 3
        assert result["QC_ICC_NumRaters"].iloc[0] == 2
        assert result["QC_ICC_Metric"].notna().all()

    def test_summary_worst_metric_is_min(
        self, high_agreement: pd.DataFrame, low_agreement: pd.DataFrame
    ) -> None:
        # Two plates: one high, one low; lower-is-bad -> worst = min metric.
        low2 = low_agreement.copy()
        low2["Plate"] = "P2"
        data = pd.concat([high_agreement, low2], ignore_index=True)
        chk = ICC(on="Size_Area", groupby=["Plate"])
        chk.analyze(data)
        summary = chk.summary().set_index("Plate")
        assert list(summary.columns) == [
            "qc_n_members",
            "qc_n_flagged",
            "qc_worst_metric",
            "qc_status",
        ]
        assert summary.loc["P1", "qc_status"] == "pass"
        assert summary.loc["P2", "qc_status"] == "fail"

    def test_group_members_maps_keys_to_member_rows(
        self, high_agreement: pd.DataFrame
    ) -> None:
        chk = ICC(on="Size_Area", groupby=["Plate"])
        chk.analyze(high_agreement)
        members = chk.group_members()
        assert ("P1",) in members
        assert len(members[("P1",)]) == len(high_agreement)
        image_file, label, value = members[("P1",)][0]
        assert image_file == "P1_t0.png"
        assert isinstance(label, int)
