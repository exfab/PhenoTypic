from __future__ import annotations

import pandas as pd

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tune import (
    Categorical,
    Evaluator,
    GridConfig,
    Knob,
    QCScorer,
    SearchSpace,
)
from phenotypic.tune._spec import Budget, TuningSpec


def _spec(tmp_path) -> TuningSpec:
    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["p"] * 96, "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()]),
        search_space=SearchSpace(knobs=(
            Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=str(csv), groupby=["Metadata_ImageName"]
            )
        ),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )


def test_budget_defaults():
    b = Budget()
    assert b.n_trials is None  # grid → run until exhausted


def test_spec_round_trips_pipeline_and_scorer(tmp_path):
    spec = _spec(tmp_path)
    back = TuningSpec.model_validate_json(spec.model_dump_json())
    # embedded pipeline reconstructed (polymorphic ops survive)
    assert [type(o).__name__ for o in back.pipeline.get_ops().values()] == [
        "GaussianBlur", "OtsuDetector",
    ]
    assert back.pipeline.get_ops()["GaussianBlur"].sigma == 2.0
    # polymorphic scorer reconstructed; path-configured check still scores
    assert isinstance(back.scorer, QCScorer)
    assert back.scorer.score_image(
        None, pd.DataFrame({"Metadata_ImageName": ["p"] * 96,
                            "Object_Label": list(range(96))})
    )["Count"] == 1.0
    assert isinstance(back.strategy, GridConfig)
