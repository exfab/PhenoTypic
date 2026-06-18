import json

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.gui.tune._export import (
    export_best_from_run,
    export_pareto_pipeline,
    export_winning_pipeline,
)
from phenotypic.sdk_ import (
    CONFIG_SUFFIX_PIPELINE,
    best_params_path,
    best_pipeline_path,
    pareto_best_pipeline_path,
    tuning_spec_path,
)
from phenotypic.tune import Categorical, Evaluator, GridConfig, Knob, QCScorer, SearchSpace
from phenotypic.tune._spec import Budget, TuningSpec


def _base() -> ImagePipeline:
    return ImagePipeline(ops=[GaussianBlur(sigma=1.0)])


def _spec(tmp_path) -> TuningSpec:
    csv = tmp_path / "layout.csv"
    csv.write_text(
        "Metadata_ImageName,Object_Label\n"
        + "\n".join(f"plate,{i}" for i in range(96))
    )
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()]),
        search_space=SearchSpace(
            knobs=(Knob(key="1.ignore_zeros", domain=Categorical(choices=(True,))),)
        ),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=str(csv), groupby=["Metadata_ImageName"]
            )
        ),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )


def test_export_winning_writes_typed_pipeline(tmp_path):
    out = export_winning_pipeline(_base(), {"0.sigma": 2.5}, tmp_path)
    assert out == best_pipeline_path(tmp_path)
    assert str(out).endswith(CONFIG_SUFFIX_PIPELINE)
    assert out.exists()
    reloaded = ImagePipeline.from_json(out.read_text())
    ops = list(reloaded.get_ops().values())
    assert ops[0].sigma == 2.5


def test_export_pareto_writes_per_objective(tmp_path):
    out = export_pareto_pipeline(
        _base(), {"0.sigma": 3.0}, tmp_path, objective="s0"
    )
    assert out == pareto_best_pipeline_path(tmp_path, "s0")
    assert str(out).endswith(CONFIG_SUFFIX_PIPELINE)
    assert out.exists()
    reloaded = ImagePipeline.from_json(out.read_text())
    ops = list(reloaded.get_ops().values())
    assert ops[0].sigma == 3.0


def test_export_best_from_run_reads_best_params_json(tmp_path):
    out = tmp_path / "run"
    tuning_spec_path(out).parent.mkdir(parents=True)
    tuning_spec_path(out).write_text(_spec(tmp_path).model_dump_json())
    best_params_path(out).write_text(
        json.dumps(
            {
                "trial_number": 7,
                "score": 0.91,
                "objectives": {},
                "params": {"0.sigma": 4.0},
                "selection": "single_best",
            }
        )
    )

    written = export_best_from_run(out)

    assert written == best_pipeline_path(out)
    restored = ImagePipeline.from_json(written.read_text())
    ops = list(restored.get_ops().values())
    assert ops[0].sigma == 4.0
