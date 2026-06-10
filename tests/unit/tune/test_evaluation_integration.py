from __future__ import annotations

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.tune import Evaluator, QCScorer


def _layout_csv(tmp_path, n: int):
    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {
            "Metadata_ImageName": ["Synthetic96PlateWithObjects"] * n,
            "Object_Label": list(range(n)),
        }
    ).to_csv(csv, index=False)
    return str(csv)


def test_perfect_count_scores_one(tmp_path):
    base = ImagePipeline(ops=[OtsuDetector()])
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout_csv(tmp_path, 96), groupby=["Metadata_ImageName"]
        )
    )
    result = Evaluator().evaluate(base, scorer, {}, [load_synth_yeast_plate()])
    assert result.n_images == 1
    assert result.terms["Count"] == pytest.approx(0.0)
    assert result.score == pytest.approx(0.0)


def test_count_mismatch_scores_below_one(tmp_path):
    # layout expects 120, detector finds 96 → metric 24/120 = 0.2 → goodness = exp(-ln2*2) = 0.25 → cost = 0.75
    base = ImagePipeline(ops=[OtsuDetector()])
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout_csv(tmp_path, 120), groupby=["Metadata_ImageName"]
        )
    )
    result = Evaluator().evaluate(base, scorer, {}, [load_synth_yeast_plate()])
    assert result.score == pytest.approx(0.75, abs=1e-6)
