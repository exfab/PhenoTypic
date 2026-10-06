from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.tune import Categorical, Evaluator, Knob, SearchSpace
from phenotypic.tune._spec import Budget, TuningSpec, _refuse_reference_metadata
from phenotypic.tune.score import QCScorer
from phenotypic.tune.strategy import GridConfig
from tests.unit.abc_.test_ref_metadata import _ReadsStrain


def test_tune_refuses_reference_metadata_pipelines():
    with pytest.raises(ValueError, match="reference metadata"):
        _refuse_reference_metadata(ImagePipeline(ops={"r": _ReadsStrain(), "d": OtsuDetector()}))


def test_tune_accepts_ordinary_pipelines():
    _refuse_reference_metadata(ImagePipeline(ops={"b": BlurGauss(), "d": OtsuDetector()}))


def _spec(tmp_path, first_op) -> TuningSpec:
    csv = tmp_path / "counts.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["p"] * 96, "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    return TuningSpec(
        pipeline=ImagePipeline(ops={"r": first_op, "d": OtsuDetector()}),
        search_space=SearchSpace(knobs=(
            Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(metadata=str(csv), groupby=["Metadata_ImageName"])
        ),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )


def test_tuning_spec_refuses_a_reference_pipeline_at_construction(tmp_path):
    """The validator, not only the helper: a spec that would fail on every
    trial is refused once, naming the op's path."""
    with pytest.raises(ValidationError) as info:
        _spec(tmp_path, _ReadsStrain())
    message = str(info.value)
    assert "reference metadata" in message
    assert "r ('Metadata_Strain', 'Metadata_BlankImage')" in message


def test_tuning_spec_accepts_the_same_shape_without_reference_ops(tmp_path):
    """Control: the spec above is otherwise valid."""
    _spec(tmp_path, BlurGauss())
