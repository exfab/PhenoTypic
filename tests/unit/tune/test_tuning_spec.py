from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tune import (
    Categorical,
    Evaluator,
    FloatRange,
    Knob,
    SearchSpace,
)
from phenotypic.tune.score import QCScorer
from phenotypic.tune.strategy import (
    GridConfig,
    RandomConfig,
)
from phenotypic.tune._search_space._targets import Param
from phenotypic.tune._spec import Budget, TuningSpec
from phenotypic.sdk_ import CONFIG_SUFFIX_TUNING, ensure_typed_json_suffix

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
    )["Count"] == 0.0
    assert isinstance(back.strategy, GridConfig)


def test_spec_to_json_returns_string(tmp_path):
    spec = _spec(tmp_path)
    payload = spec.to_json()

    assert isinstance(payload, str)
    assert json.loads(payload)["phenotypic_version"]
    back = TuningSpec.model_validate_json(payload)
    assert isinstance(back.strategy, GridConfig)
    assert back.phenotypic_version == spec.phenotypic_version


def test_missing_phenotypic_version_warns_and_defaults(tmp_path):
    payload = json.loads(_spec(tmp_path).model_dump_json())
    payload.pop("phenotypic_version")

    with pytest.warns(UserWarning, match="phenotypic_version"):
        back = TuningSpec.model_validate(payload)

    assert back.phenotypic_version


def test_spec_to_json_file_uses_typed_suffix(tmp_path):
    spec = _spec(tmp_path)
    filepath = tmp_path / "tuning_spec.json"
    typed_filepath = ensure_typed_json_suffix(filepath, CONFIG_SUFFIX_TUNING)

    spec.to_json(filepath)

    assert not filepath.exists()
    assert typed_filepath.exists()
    back = TuningSpec.model_validate_json(typed_filepath.read_text())
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
    raw["scorer"]["params"]["check"]["metadata"] = str(csv)
    back = TuningSpec.model_validate(raw)
    assert isinstance(back.strategy, GridConfig)
    assert back.strategy.kind == "grid"
    assert [type(o).__name__ for o in back.pipeline.get_ops().values()] == [
        "GaussianBlur", "OtsuDetector",
    ]


def _qc(tmp_path) -> QCScorer:
    csv = tmp_path / "qc_layout.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["p1"] * 96, "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    return QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=str(csv), groupby=["Metadata_ImageName"]
        )
    )


def _spec_with(knobs, tmp_path) -> TuningSpec:
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()]),
        search_space=SearchSpace(knobs=knobs),
        scorer=_qc(tmp_path),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )


def test_valid_targets_pass(tmp_path):
    _spec_with(
        (Knob(target=Param(op=0, field="sigma"), domain=FloatRange(low=0.5, high=8.0)),),
        tmp_path,
    )


def test_out_of_range_op_rejected(tmp_path):
    with pytest.raises(Exception, match="op 5"):
        _spec_with(
            (Knob(target=Param(op=5, field="sigma"),
                  domain=FloatRange(low=0.5, high=8.0)),),
            tmp_path,
        )


def test_op_class_mismatch_rejected(tmp_path):
    with pytest.raises(Exception, match="OtsuDetector"):
        _spec_with(
            (Knob(target=Param(op=0, field="sigma", op_class="OtsuDetector"),
                  domain=FloatRange(low=0.5, high=8.0)),),
            tmp_path,
        )


def test_missing_field_suggests(tmp_path):
    with pytest.raises(Exception, match="did you mean 'sigma'"):
        _spec_with(
            (Knob(target=Param(op=0, field="sigam"),
                  domain=FloatRange(low=0.5, high=8.0)),),
            tmp_path,
        )
