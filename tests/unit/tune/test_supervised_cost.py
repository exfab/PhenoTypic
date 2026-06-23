from __future__ import annotations

import pandas as pd
import pytest

from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.tune.score import (
    GroundTruthMasks,
    SupervisedScorer,
)
from phenotypic.tune.score._orient import Sense


def _counts_csv(tmp_path):
    csv = tmp_path / "counts.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["p1"] * 96, "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    return csv


def _measured(n: int, name: str = "p1") -> pd.DataFrame:
    return pd.DataFrame(
        {"Metadata_ImageName": [name] * n, "Object_Label": list(range(n))}
    )


def test_supervised_declares_higher_better():
    assert SupervisedScorer._TERM_SENSE is Sense.HIGHER_BETTER


def test_count_tier_perfect_match_is_zero_cost(tmp_path):
    csv = _counts_csv(tmp_path)
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=csv),
        count_check=ExpectedVsDetectedCount(
            metadata=str(csv), groupby=["Metadata_ImageName"]
        ),
    )
    # score_image orients the goodness fold to cost: perfect 96-vs-96 -> 0.0.
    assert scorer.score_image(None, _measured(96)) == {
        "CountMAE": pytest.approx(0.0)
    }


def test_count_tier_under_detect_is_higher_cost(tmp_path):
    csv = _counts_csv(tmp_path)
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=csv),
        count_check=ExpectedVsDetectedCount(
            metadata=str(csv), groupby=["Metadata_ImageName"]
        ),
    )
    faithful = scorer.score_image(None, _measured(96))["CountMAE"]
    under = scorer.score_image(None, _measured(24))["CountMAE"]
    assert faithful < under
