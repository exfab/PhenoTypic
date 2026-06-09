"""Degenerate-case tests for ICC(2,1) — snapshot, subject=StrainID model.

Encodes the devil's-advocate "ICC degenerate guards" scenarios from the
smart-QC design spec (Phase A) against the FINAL ICC model:

* **Snapshot** design (no time axis): each group's two-way matrix is built
  with the *subject* axis indexed by ``Metadata_StrainID`` and the *rater*
  axis indexed by ``Metadata_Replicate``.
* Every under-powered or undefined design yields ``metric = NaN`` (= ``"pass"``,
  "insufficient data" — *not* "good agreement").
* A genuinely-bad design (replicates disagreeing across strains) yields a
  low/negative ICC that flags as ``"fail"``; a negative ICC is valid (worse
  than chance agreement) and NaN is never treated as good.

Two classes of guard, asserted as *distinct*:

* **Loud missing-axis** — the subject or rater column is absent entirely. The
  group key is recorded in :attr:`unmatched_groups` (mirroring
  :class:`ExpectedVsDetectedCount`'s convention: a list of ``groupby`` key
  tuples, reset each ``analyze``) AND the metric is NaN. This is a
  configuration error the GUI should surface, not a quiet insufficient-data
  bin.
* **Quiet insufficient-data** — too few subjects/raters, an
  incomplete/duplicated cell, or zero variance. The metric is NaN but the
  group is NOT recorded in ``unmatched_groups`` (it is genuine missing signal,
  not a misconfiguration).

This file is intentionally focused on the **degenerate matrix**; the realistic
``all_meas.csv`` case lives in qc-engine's ``test_icc.py`` to avoid overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from phenotypic.analysis.qc import ICC

# Final ICC axis model: snapshot, subject = StrainID, rater = Replicate.
_SUBJECT = "Metadata_StrainID"
_RATER = "Metadata_Replicate"


def _make_check() -> ICC:
    """ICC configured for the snapshot subject=StrainID / rater=Replicate model."""
    return ICC(
        on="Size_Area",
        groupby=["Plate"],
        subject_label=_SUBJECT,
        rater_label=_RATER,
    )


def _analyze_single_group(chk: ICC, df: pd.DataFrame) -> tuple[float, str, bool]:
    """Run ``chk`` on a single-group frame; return (metric, status, flag)."""
    result = chk.analyze(df)
    metric = float(result[chk.metric_col()].iloc[0])
    status = str(result[chk.status_col()].iloc[0])
    flag = bool(result[chk.flag_col()].iloc[0])
    return metric, status, flag


class TestICCLoudMissingAxis:
    """Absent subject/rater column → recorded in unmatched_groups AND NaN.

    This is the LOUD guard: a missing axis is a misconfiguration, distinct
    from the quiet insufficient-data guards which never touch
    ``unmatched_groups``.
    """

    def test_subject_column_absent_is_unmatched_and_nan(self) -> None:
        # No Metadata_StrainID (subject axis): two raters, no subject index.
        df = pd.DataFrame(
            {
                "Plate": ["P1"] * 4,
                _RATER: [1, 2, 1, 2],
                "Size_Area": [10.0, 11.0, 20.0, 21.0],
            }
        )
        chk = _make_check()
        metric, status, flag = _analyze_single_group(chk, df)
        assert np.isnan(metric)
        assert status == "pass"
        assert flag is False
        # Loud: the group key tuple is recorded for the GUI to surface.
        assert ("P1",) in chk.unmatched_groups

    def test_rater_column_absent_is_unmatched_and_nan(self) -> None:
        # No Metadata_Replicate (rater axis).
        df = pd.DataFrame(
            {
                "Plate": ["P1"] * 3,
                _SUBJECT: ["A", "B", "C"],
                "Size_Area": [10.0, 20.0, 40.0],
            }
        )
        chk = _make_check()
        metric, status, flag = _analyze_single_group(chk, df)
        assert np.isnan(metric)
        assert status == "pass"
        assert flag is False
        assert ("P1",) in chk.unmatched_groups

    def test_unmatched_groups_resets_on_reanalyze(self) -> None:
        # The list must not carry over between analyze() calls.
        missing = pd.DataFrame(
            {
                "Plate": ["P1"] * 3,
                _SUBJECT: ["A", "B", "C"],
                "Size_Area": [10.0, 20.0, 40.0],
            }
        )
        chk = _make_check()
        chk.analyze(missing)
        assert ("P1",) in chk.unmatched_groups

        # A subsequent clean run with both axes present must clear it.
        clean = _clean_two_strain_frame()
        chk.analyze(clean)
        assert chk.unmatched_groups == []


class TestICCQuietInsufficientData:
    """Under-powered / undefined designs → NaN, but NOT recorded as unmatched."""

    def test_n_subjects_lt_2_is_nan_not_unmatched(self) -> None:
        # One subject (single StrainID), three raters.
        df = pd.DataFrame(
            {
                "Plate": ["P1"] * 3,
                _SUBJECT: ["A", "A", "A"],
                _RATER: [1, 2, 3],
                "Size_Area": [10.0, 11.0, 12.0],
            }
        )
        chk = _make_check()
        metric, status, flag = _analyze_single_group(chk, df)
        assert np.isnan(metric)
        assert status == "pass"
        assert flag is False
        # Quiet: insufficient data is NOT a misconfiguration.
        assert ("P1",) not in chk.unmatched_groups

    def test_n_raters_lt_2_single_rater_is_nan_not_unmatched(self) -> None:
        # Three subjects, a single rater -> n_raters < 2.
        df = pd.DataFrame(
            {
                "Plate": ["P1"] * 3,
                _SUBJECT: ["A", "B", "C"],
                _RATER: [1, 1, 1],
                "Size_Area": [10.0, 20.0, 40.0],
            }
        )
        chk = _make_check()
        metric, status, flag = _analyze_single_group(chk, df)
        assert np.isnan(metric)
        assert status == "pass"
        assert flag is False
        assert ("P1",) not in chk.unmatched_groups

    def test_incomplete_cell_nans_entire_group_rows_preserved(self) -> None:
        # 3 strains x 2 raters but the (C, rater=2) cell is missing.
        df = pd.DataFrame(
            {
                "Plate": ["P1"] * 5,
                _SUBJECT: ["A", "A", "B", "B", "C"],
                _RATER: [1, 2, 1, 2, 1],
                "Size_Area": [10.0, 10.1, 20.0, 20.1, 40.0],
            }
        )
        chk = _make_check()
        result = chk.analyze(df)
        # The whole group is NaN'd rather than silently completing the design.
        assert bool(result[chk.metric_col()].isna().all())
        assert (result[chk.status_col()] == "pass").all()
        assert not result[chk.flag_col()].any()
        # Rows are NOT dropped — all 5 input rows survive.
        assert len(result) == 5
        # An incomplete design is insufficient data, not a missing axis.
        assert ("P1",) not in chk.unmatched_groups

    def test_duplicated_cell_nans_entire_group(self) -> None:
        # Two observations for (A, rater=1) makes the pivot ambiguous.
        df = pd.DataFrame(
            {
                "Plate": ["P1"] * 5,
                _SUBJECT: ["A", "A", "A", "B", "B"],
                _RATER: [1, 1, 2, 1, 2],
                "Size_Area": [10.0, 10.5, 11.0, 20.0, 21.0],
            }
        )
        chk = _make_check()
        metric, status, flag = _analyze_single_group(chk, df)
        assert np.isnan(metric)
        assert status == "pass"
        assert flag is False
        assert ("P1",) not in chk.unmatched_groups

    def test_zero_variance_is_nan_not_perfect(self) -> None:
        # All-identical values → total variance 0 → ICC undefined → NaN.
        df = pd.DataFrame(
            {
                "Plate": ["P1"] * 6,
                _SUBJECT: ["A", "A", "B", "B", "C", "C"],
                _RATER: [1, 2, 1, 2, 1, 2],
                "Size_Area": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
            }
        )
        chk = _make_check()
        metric, status, flag = _analyze_single_group(chk, df)
        # Zero variance is "insufficient signal", explicitly NOT a perfect-1.
        assert np.isnan(metric)
        assert status == "pass"
        assert flag is False
        assert ("P1",) not in chk.unmatched_groups


def _clean_two_strain_frame() -> pd.DataFrame:
    """Three well-separated strains, tight replicate agreement → ICC≈1."""
    return pd.DataFrame(
        {
            "Plate": ["P1"] * 9,
            _SUBJECT: ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            _RATER: [1, 2, 3] * 3,
            "Size_Area": [
                10.0, 10.1, 9.9,
                20.0, 20.2, 19.8,
                40.0, 40.1, 39.9,
            ],
        }
    )


class TestICCAgreementDirection:
    """Genuinely-poor agreement → low/negative ICC → fail; clean → ≈1 → pass."""

    def test_disagreeing_replicates_low_or_negative_icc_fails(self) -> None:
        # Replicates contradict each other across strains: no consistent
        # between-strain signal -> ICC collapses to low/negative -> fail.
        df = pd.DataFrame(
            {
                "Plate": ["P1"] * 6,
                _SUBJECT: ["A", "A", "B", "B", "C", "C"],
                _RATER: [1, 2, 1, 2, 1, 2],
                "Size_Area": [10.0, 40.0, 40.0, 10.0, 25.0, 25.0],
            }
        )
        chk = _make_check()
        metric, status, flag = _analyze_single_group(chk, df)
        assert np.isfinite(metric)
        # Low/negative absolute-agreement ICC at or below the 0.50 fail line.
        assert metric <= chk.fail_threshold
        assert status == "fail"
        assert flag is True
        assert ("P1",) not in chk.unmatched_groups

    def test_clean_separated_strains_icc_near_one_passes(self) -> None:
        df = _clean_two_strain_frame()
        chk = _make_check()
        metric, status, flag = _analyze_single_group(chk, df)
        assert np.isfinite(metric)
        assert metric > chk.warn_threshold  # well above 0.75
        assert status == "pass"
        assert flag is False
        assert ("P1",) not in chk.unmatched_groups
