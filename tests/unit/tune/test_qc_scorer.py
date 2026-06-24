from __future__ import annotations

import pandas as pd
import pytest

from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.tune.score._qc_scorer import QCScorer, _threshold_anchored


def test_threshold_anchored_anchors():
    assert _threshold_anchored(0.0, 0.10) == pytest.approx(1.0)        # perfect
    assert _threshold_anchored(0.10, 0.10) == pytest.approx(0.5)       # at fail boundary
    assert _threshold_anchored(0.20, 0.10) == pytest.approx(0.25)      # 2× boundary
    assert _threshold_anchored(float("inf"), 0.10) == 0.0             # unmatched group
    # monotone decreasing in the metric
    assert _threshold_anchored(0.05, 0.10) > _threshold_anchored(0.15, 0.10)


def _layout(n: int, name: str = "p1") -> pd.DataFrame:
    return pd.DataFrame(
        {"Metadata_ImageName": [name] * n, "Object_Label": list(range(n))}
    )


def _measurements(n: int, name: str = "p1") -> pd.DataFrame:
    return pd.DataFrame(
        {"Metadata_ImageName": [name] * n, "Object_Label": list(range(n))}
    )


def test_score_image_perfect_match_is_zero_cost():
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout(96), groupby=["Metadata_ImageName"]
        )
    )
    out = scorer.score_image(None, _measurements(96))
    assert set(out) == {"Count"}
    assert out["Count"] == pytest.approx(0.0)  # perfect match = zero cost


def test_score_image_at_fail_threshold_is_half():
    # expected 100, detected 90 → metric 0.10 == fail_threshold → goodness 0.5 → cost 0.5
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout(100), groupby=["Metadata_ImageName"]
        )
    )
    assert scorer.score_image(None, _measurements(90))["Count"] == pytest.approx(0.5)


def test_score_image_unmatched_group_is_worst_cost():
    # measurement group "p2" has no metadata counterpart → metric inf → goodness 0 → cost 1
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout(96, "p1"), groupby=["Metadata_ImageName"]
        )
    )
    assert scorer.score_image(None, _measurements(10, "p2"))["Count"] == pytest.approx(1.0)


def test_score_image_empty_measurements_is_worst_cost():
    # empty frame floors goodness to 0 → cost 1
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout(96), groupby=["Metadata_ImageName"]
        )
    )
    assert scorer.score_image(None, pd.DataFrame())["Count"] == pytest.approx(1.0)


def test_availability_reflects_metadata():
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout(96), groupby=["Metadata_ImageName"]
        )
    )
    assert scorer.availability() is True


def test_path_configured_scorer_round_trips(tmp_path):
    # Configure the check from a CSV path so the layout path persists through JSON.
    csv = tmp_path / "layout.csv"
    _layout(96).to_csv(csv, index=False)
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=str(csv), groupby=["Metadata_ImageName"]
        )
    )
    reloaded = QCScorer.model_validate_json(scorer.model_dump_json())
    assert reloaded.check.metadata == str(csv)
    # the reloaded scorer scores identically (re-read the layout from disk)
    assert reloaded.score_image(None, _measurements(96))["Count"] == pytest.approx(0.0)
