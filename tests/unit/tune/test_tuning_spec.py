from __future__ import annotations

import json
from pathlib import Path

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
    RandomConfig,
    SearchSpace,
)
from phenotypic.tune._spec import Budget, TuningSpec

#: The frozen Phase-1 ``tuning_spec.json`` (strategy block in the original
#: discriminated-union form ``{"seed": 0, "kind": "grid"}``) — proves that
#: widening ``strategy`` to a ``polymorphic_field`` keeps old recipes loadable.
_PHASE1_FIXTURE = Path(__file__).parent / "fixtures" / "phase1_tuning_spec.json"


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


def test_strategy_field_round_trips_grid_via_registry(tmp_path):
    # Widened ``strategy`` (a polymorphic_field) still round-trips a GridConfig:
    # the concrete subclass survives a JSON dump/load via the class registry.
    spec = _spec(tmp_path)
    back = TuningSpec.model_validate_json(spec.model_dump_json())
    assert isinstance(back.strategy, GridConfig)
    assert back.strategy.kind == "grid"

    # And a RandomConfig (carrying extra fields) round-trips too.
    spec2 = _spec(tmp_path)
    spec2 = spec2.model_copy(update={"strategy": RandomConfig(n_trials=4, seed=2)})
    back2 = TuningSpec.model_validate_json(spec2.model_dump_json())
    assert isinstance(back2.strategy, RandomConfig)
    assert back2.strategy.n_trials == 4
    assert back2.strategy.seed == 2


def test_phase1_grid_spec_json_still_loads(tmp_path):
    # A frozen Phase-1 tuning_spec.json (strategy block in the original
    # discriminated-union form, no "class" wrapper) must still deserialize after
    # the field is widened to a polymorphic_field.
    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["p"] * 96, "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    raw = json.loads(_PHASE1_FIXTURE.read_text())
    # Confirm the frozen fixture really carries the Phase-1 discriminator form.
    assert raw["strategy"] == {"seed": 0, "kind": "grid"}
    raw["scorer"]["params"]["check"]["metadata_source"] = str(csv)
    back = TuningSpec.model_validate(raw)
    assert isinstance(back.strategy, GridConfig)
    assert back.strategy.kind == "grid"
    assert [type(o).__name__ for o in back.pipeline.get_ops().values()] == [
        "GaussianBlur", "OtsuDetector",
    ]
