"""Adversarial / risk-regression tests for the QC engine contract.

These encode the devil's-advocate scenarios from the smart-QC design spec
(``docs/superpowers/specs/2026-05-29-smart-qc-design.md``, "Risk-driven
refinements" → Phase A). They exercise the *agreed contract* rather than
any one check's internals:

* the directional ``model_validator`` that rejects mis-ordered thresholds,
* inclusive directional flag/status boundaries in *both* ``_HIGHER_IS_BAD``
  polarities,
* NaN-metric → ``"pass"`` / ``Flag=False`` for under-powered bins,
* ``summary()`` column naming that must not collide with a ``groupby`` column
  literally named ``status``,
* ``group_members()`` shape and its missing-key-column guard,
* discoverability of every shipped check via the public ``phenotypic.analysis``
  surface and via ``OperationRegistry.get_by_category("quality_check")``.

The shipped check used for the *higher-is-bad* path is
:class:`~phenotypic.analysis._replicate_agreement.ReplicateAgreement`,
imported from its concrete private module to be robust to ``__init__`` import
timing. The *lower-is-bad* path is exercised here by a minimal in-file
:class:`_LowerIsBadCheck` fixture so the abstract contract (validator,
inclusive boundary, summary-min) is pinned independently of any one concrete
lower-is-bad check; ICC's own degenerate/agreement coverage lives in
``test_icc_degenerate.py``. The discoverability test reaches for the public
surface and asserts all six shipped checks — including ICC (final roster = 6)
— are importable and registry-discoverable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from phenotypic.analysis._replicate_agreement import ReplicateAgreement
from phenotypic.analysis.abc_._quality_check import QualityCheck

# Renamed, prefixed summary columns per the contract (collision-proof).
_SUMMARY_COLS = [
    "qc_n_members",
    "qc_n_flagged",
    "qc_worst_metric",
    "qc_status",
]


class _LowerIsBadCheck(QualityCheck):
    """Minimal ``_HIGHER_IS_BAD=False`` check for exercising the agreement path.

    A deterministic lower-is-bad stand-in that pins the *abstract* contract
    (threshold validator, inclusive ``<=`` boundary, summary-min) without
    depending on any one concrete lower-is-bad check's numerics. The metric is
    simply the mean of ``self.on`` across the group, broadcast to every member
    row, so a test can construct a frame whose group mean lands exactly on a
    threshold and assert the inclusive lower-is-bad boundary behavior. Lower
    mean ⇒ worse, matching an agreement score where smaller is worse.
    """

    name = "LoBad"
    _HIGHER_IS_BAD = False

    warn_threshold: float = 0.75
    fail_threshold: float = 0.50

    def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
        out = group.copy()
        out[self.metric_col()] = float(group[self.on].mean())
        return out


class TestThresholdValidator:
    """A ``model_validator(mode="after")`` enforces threshold ordering.

    Higher-is-bad checks require ``warn_threshold <= fail_threshold``;
    lower-is-bad checks require ``warn_threshold >= fail_threshold``.
    Equality is allowed (no warn band). Mis-order must raise
    ``pydantic.ValidationError`` so a lower-is-bad check with mis-ordered
    thresholds cannot silently invert pass/fail.
    """

    def test_higher_is_bad_misorder_raises(self) -> None:
        # ReplicateAgreement is _HIGHER_IS_BAD=True: warn must be <= fail.
        with pytest.raises(ValidationError):
            ReplicateAgreement(
                on="Size_Area",
                groupby=["Plate"],
                warn_threshold=0.2,
                fail_threshold=0.1,
            )

    def test_lower_is_bad_misorder_raises(self) -> None:
        # Lower-is-bad: warn must be >= fail. warn<fail would invert pass/fail.
        with pytest.raises(ValidationError):
            _LowerIsBadCheck(
                on="Size_Area",
                groupby=["Plate"],
                warn_threshold=0.5,
                fail_threshold=0.75,
            )

    def test_higher_is_bad_ordered_ok(self) -> None:
        chk = ReplicateAgreement(
            on="Size_Area",
            groupby=["Plate"],
            warn_threshold=0.1,
            fail_threshold=0.2,
        )
        assert chk.warn_threshold == pytest.approx(0.1)
        assert chk.fail_threshold == pytest.approx(0.2)

    def test_lower_is_bad_ordered_ok(self) -> None:
        chk = _LowerIsBadCheck(
            on="Size_Area",
            groupby=["Plate"],
            warn_threshold=0.75,
            fail_threshold=0.50,
        )
        assert chk.warn_threshold == pytest.approx(0.75)
        assert chk.fail_threshold == pytest.approx(0.50)

    def test_equal_thresholds_allowed_higher_is_bad(self) -> None:
        # Equality = no warn band; must not raise.
        chk = ReplicateAgreement(
            on="Size_Area",
            groupby=["Plate"],
            warn_threshold=0.15,
            fail_threshold=0.15,
        )
        assert chk.warn_threshold == chk.fail_threshold

    def test_equal_thresholds_allowed_lower_is_bad(self) -> None:
        chk = _LowerIsBadCheck(
            on="Size_Area",
            groupby=["Plate"],
            warn_threshold=0.6,
            fail_threshold=0.6,
        )
        assert chk.warn_threshold == chk.fail_threshold


class TestDirectionalBoundaries:
    """Flag/status boundaries are inclusive in both polarities.

    Higher-is-bad: ``metric >= fail_threshold`` → "fail" (Flag True);
    ``metric >= warn_threshold`` → "warn". Lower-is-bad inverts to ``<=``.
    The boundary value itself (metric == threshold) is part of the bad band.
    """

    def _status_for(self, chk, df: pd.DataFrame, group_key) -> tuple[str, bool]:
        """Run ``chk`` on ``df`` and return (status, flag) for one group key."""
        result = chk.analyze(df)
        groupby = list(chk.groupby)
        mask = pd.Series(True, index=result.index)
        for col, val in zip(groupby, group_key):
            mask &= result[col] == val
        rows = result.loc[mask]
        status = rows[chk.status_col()].iloc[0]
        flag = bool(rows[chk.flag_col()].iloc[0])
        return status, flag

    # --- Higher-is-bad: ReplicateAgreement (relative SE) ---

    def _se_frame_with_metric(self, target_rel_se: float) -> pd.DataFrame:
        """Two-replicate single-time frame whose relative SE == target.

        For n=2 with values (m-d, m+d): std(ddof=1)=d*sqrt(2),
        SE=std/sqrt(2)=d, mean=m, relative SE = d/m. Pick m=100, d=m*target.
        """
        mean = 100.0
        d = mean * target_rel_se
        return pd.DataFrame(
            {
                "Plate": ["P1", "P1"],
                "Metadata_Time": [0, 0],
                "Size_Area": [mean - d, mean + d],
            }
        )

    def test_higher_is_bad_at_fail_threshold_is_fail(self) -> None:
        chk = ReplicateAgreement(
            on="Size_Area",
            groupby=["Plate"],
            warn_threshold=0.10,
            fail_threshold=0.20,
        )
        df = self._se_frame_with_metric(0.20)  # metric == fail_threshold
        result = chk.analyze(df)
        metric = result[chk.metric_col()].iloc[0]
        assert metric == pytest.approx(0.20)
        status, flag = self._status_for(chk, df, ("P1",))
        assert status == "fail"
        assert flag is True

    def test_higher_is_bad_at_warn_threshold_is_warn(self) -> None:
        chk = ReplicateAgreement(
            on="Size_Area",
            groupby=["Plate"],
            warn_threshold=0.10,
            fail_threshold=0.20,
        )
        df = self._se_frame_with_metric(0.10)  # metric == warn_threshold
        result = chk.analyze(df)
        assert result[chk.metric_col()].iloc[0] == pytest.approx(0.10)
        status, flag = self._status_for(chk, df, ("P1",))
        assert status == "warn"
        assert flag is False

    # --- Lower-is-bad: _LowerIsBadCheck (metric == group mean) ---

    def _lobad_frame(self, mean_value: float) -> pd.DataFrame:
        """Single group whose ``Size_Area`` mean equals ``mean_value``."""
        return pd.DataFrame(
            {
                "Plate": ["P1", "P1"],
                "Size_Area": [mean_value - 0.1, mean_value + 0.1],
            }
        )

    def test_lower_is_bad_at_fail_threshold_is_fail(self) -> None:
        chk = _LowerIsBadCheck(
            on="Size_Area",
            groupby=["Plate"],
            warn_threshold=0.75,
            fail_threshold=0.50,
        )
        df = self._lobad_frame(0.50)  # metric == fail_threshold
        result = chk.analyze(df)
        assert result[chk.metric_col()].iloc[0] == pytest.approx(0.50)
        status, flag = self._status_for(chk, df, ("P1",))
        assert status == "fail"
        assert flag is True

    def test_lower_is_bad_at_warn_threshold_is_warn(self) -> None:
        chk = _LowerIsBadCheck(
            on="Size_Area",
            groupby=["Plate"],
            warn_threshold=0.75,
            fail_threshold=0.50,
        )
        df = self._lobad_frame(0.75)  # metric == warn_threshold (> fail)
        result = chk.analyze(df)
        assert result[chk.metric_col()].iloc[0] == pytest.approx(0.75)
        status, flag = self._status_for(chk, df, ("P1",))
        assert status == "warn"
        assert flag is False


class TestNaNMetricIsPass:
    """A NaN metric (under-powered / degenerate bin) is pass / Flag False.

    Degenerate bins must never gate curation regardless of polarity.
    """

    def test_higher_is_bad_nan_is_pass(self) -> None:
        # Single replicate per (group, time) with min_replicates=2 -> NaN.
        df = pd.DataFrame(
            {
                "Plate": ["P1", "P1"],
                "Metadata_Time": [0, 1],
                "Size_Area": [10.0, 20.0],
            }
        )
        chk = ReplicateAgreement(
            on="Size_Area", groupby=["Plate"], min_replicates=2
        )
        result = chk.analyze(df)
        assert bool(result[chk.metric_col()].isna().all())
        assert (result[chk.status_col()] == "pass").all()
        assert not result[chk.flag_col()].any()

    def test_lower_is_bad_nan_is_pass(self) -> None:
        # NaN measurement -> NaN mean metric -> pass even though lower-is-bad.
        df = pd.DataFrame(
            {
                "Plate": ["P1", "P1"],
                "Size_Area": [np.nan, np.nan],
            }
        )
        chk = _LowerIsBadCheck(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(df)
        assert bool(result[chk.metric_col()].isna().all())
        assert (result[chk.status_col()] == "pass").all()
        assert not result[chk.flag_col()].any()


class TestSummaryNameCollision:
    """``summary()`` must not collide with a groupby column named ``status``.

    The contract renames summary outputs to a prefixed set so a groupby
    column literally named ``status`` (or ``num_rows`` etc.) does not crash
    ``reset_index``. ``qc_worst_metric`` is the max for higher-is-bad and the
    min for lower-is-bad.
    """

    def test_groupby_named_status_does_not_raise(self) -> None:
        df = pd.DataFrame(
            {
                # groupby column deliberately named "status" to collide.
                "status": ["a", "a", "b", "b"],
                "Metadata_Time": [0, 0, 0, 0],
                "Size_Area": [10.0, 20.0, 100.0, 100.5],
            }
        )
        chk = ReplicateAgreement(on="Size_Area", groupby=["status"])
        chk.analyze(df)
        summary = chk.summary()  # must not raise on reset_index collision
        # The groupby key column is preserved, plus the prefixed outputs.
        assert "status" in summary.columns
        for col in _SUMMARY_COLS:
            assert col in summary.columns, f"missing prefixed summary col {col}"

    def test_worst_metric_is_max_for_higher_is_bad(self) -> None:
        # Two timepoints in one group; worst (max) relative SE should win.
        df = pd.DataFrame(
            {
                "Plate": ["P1"] * 4,
                "Metadata_Time": [0, 0, 1, 1],
                # t=0 rel-SE small; t=1 rel-SE large.
                "Size_Area": [100.0, 101.0, 100.0, 140.0],
            }
        )
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(df)
        chk_max = float(np.nanmax(result[chk.metric_col()]))
        summary = chk.summary()
        row = summary.loc[summary["Plate"] == "P1"].iloc[0]
        assert row["qc_worst_metric"] == pytest.approx(chk_max)

    def test_worst_metric_is_min_for_lower_is_bad(self) -> None:
        # Two groups with different means; lower-is-bad worst == per-group min.
        # _LowerIsBadCheck broadcasts the group mean to every row, so the
        # worst (min) for a group is that single broadcast metric.
        df = pd.DataFrame(
            {
                "Plate": ["P1", "P1", "P2", "P2"],
                "Size_Area": [0.40, 0.42, 0.90, 0.92],
            }
        )
        chk = _LowerIsBadCheck(on="Size_Area", groupby=["Plate"])
        result = chk.analyze(df)
        summary = chk.summary()
        for plate in ("P1", "P2"):
            group_metric = float(
                result.loc[result["Plate"] == plate, chk.metric_col()].iloc[0]
            )
            row = summary.loc[summary["Plate"] == plate].iloc[0]
            # Single broadcast metric per group => min == that metric.
            assert row["qc_worst_metric"] == pytest.approx(group_metric)


class TestGroupMembers:
    """``group_members()`` maps group keys to member (file, label, value)."""

    def test_returns_member_tuples_keyed_by_group(self) -> None:
        df = pd.DataFrame(
            {
                "Plate": ["P1", "P1", "P2"],
                "Metadata_Time": [0, 0, 0],
                "Metadata_ImageFile": ["a.png", "a.png", "b.png"],
                "Object_Label": [1, 2, 1],
                "Size_Area": [10.0, 11.0, 50.0],
            }
        )
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        chk.analyze(df)
        members = chk.group_members()
        assert isinstance(members, dict)
        # Keys are always tuples, even for a single groupby column.
        assert ("P1",) in members
        assert ("P2",) in members
        p1 = members[("P1",)]
        assert len(p1) == 2
        # Each member is (Metadata_ImageFile, Object_Label, member_value).
        files = {m[0] for m in p1}
        labels = {m[1] for m in p1}
        values = {m[2] for m in p1}
        assert files == {"a.png"}
        assert labels == {1, 2}
        assert values == {10.0, 11.0}

    def test_returns_empty_when_key_columns_absent(self) -> None:
        # No Metadata_ImageFile / Object_Label -> empty mapping, no raise.
        df = pd.DataFrame(
            {
                "Plate": ["P1", "P1"],
                "Metadata_Time": [0, 0],
                "Size_Area": [10.0, 11.0],
            }
        )
        chk = ReplicateAgreement(on="Size_Area", groupby=["Plate"])
        chk.analyze(df)
        assert chk.group_members() == {}


class TestDiscoverability:
    """The 6 shipped QualityCheck subclasses are import- and registry-visible.

    Guards the design's "no registry edits" claim: a check added to
    ``analysis/__init__.py`` auto-appears in
    ``OperationRegistry.get_by_category("quality_check")``. ICC is part of the
    v1 roster (final decision: roster = 6), so it must be present on both the
    public import surface and in the registry category.
    """

    EXPECTED = {
        "ReplicateAgreement",
        "ExpectedVsDetectedCount",
        "RelativeMAD",
        "MaxModifiedZScore",
        "ICC",
        "TukeyOutlierFraction",
    }

    def test_all_checks_importable_from_public_surface(self) -> None:
        import phenotypic.analysis as analysis

        for name in self.EXPECTED:
            assert hasattr(analysis, name), (
                f"{name} not exported from phenotypic.analysis"
            )
            assert name in analysis.__all__, f"{name} missing from __all__"

    def test_all_checks_in_registry_category(self) -> None:
        try:
            from phenotypic.gui._operation_registry import OperationRegistry
        except Exception as exc:  # pragma: no cover - GUI import guard
            pytest.skip(f"OperationRegistry unavailable: {exc}")

        registry = OperationRegistry()
        registry.discover()
        discovered = {
            info.name for info in registry.get_by_category("quality_check")
        }
        missing = self.EXPECTED - discovered
        assert not missing, (
            f"checks not discovered by registry quality_check category: "
            f"{missing}"
        )
