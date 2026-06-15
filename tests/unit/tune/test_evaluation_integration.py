from __future__ import annotations

import math

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.tune import Evaluator, QCScorer


def _layout_csv(tmp_path, n: int, image_name: str = "Synthetic96PlateWithObjects"):
    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {
            "Metadata_ImageName": [image_name] * n,
            "Object_Label": list(range(n)),
        }
    ).to_csv(csv, index=False)
    return str(csv)


def _measured_count(pipeline: ImagePipeline) -> int:
    measured = pipeline.apply_and_measure(
        load_synth_yeast_plate(), inplace=False, apply_post=False
    )
    return len(measured)


def test_perfect_count_scores_one(tmp_path):
    base = ImagePipeline(ops=[OtsuDetector()])
    expected_count = _measured_count(base)
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout_csv(tmp_path, expected_count),
            groupby=["Metadata_ImageName"],
        )
    )
    result = Evaluator().evaluate(base, scorer, {}, [load_synth_yeast_plate()])
    assert result.n_images == 1
    assert result.terms["Count"] == pytest.approx(0.0)
    assert result.score == pytest.approx(0.0)


def test_count_mismatch_scores_below_one(tmp_path):
    base = ImagePipeline(ops=[OtsuDetector()])
    detected_count = _measured_count(base)
    expected_count = max(detected_count + 1, round(detected_count * 1.25))
    metric = abs(detected_count - expected_count) / expected_count
    check = ExpectedVsDetectedCount(
        metadata=_layout_csv(tmp_path, expected_count),
        groupby=["Metadata_ImageName"],
    )
    expected_cost = 1.0 - math.exp(-math.log(2.0) * metric / check.fail_threshold)
    scorer = QCScorer(
        check=check
    )
    result = Evaluator().evaluate(base, scorer, {}, [load_synth_yeast_plate()])
    assert result.score == pytest.approx(expected_cost, abs=1e-6)
