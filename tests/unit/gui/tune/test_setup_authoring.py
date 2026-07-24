from pathlib import Path

import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.tune._setup_authoring import (
    build_authored_setup_spec,
    resolve_setup_path,
    setup_path_payload,
    write_authored_setup_spec,
)
from phenotypic.tune import (
    Evaluator,
    Fixed,
    Knob,
    SearchSpace,
    TuningSpec,
    infer_search_space,
)
from phenotypic.tune.score import QCScorer
from phenotypic.tune.strategy import GridConfig, RandomConfig
from phenotypic.tune._spec import Budget
from phenotypic.tune.targets import Param


def _metadata(path: Path) -> Path:
    path.write_text(
        "MetadataImage_ImageName,Object_Label\n"
        + "\n".join(f"plate,{i}" for i in range(96)),
        encoding="utf-8",
    )
    return path


def test_write_authored_setup_spec_uses_path_backed_qc_scorer(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.json.pht-pipe"
    pipeline_path.write_text(
        ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()]).to_json(),
        encoding="utf-8",
    )
    metadata_path = _metadata(tmp_path / "layout.csv")

    written = write_authored_setup_spec(
        sandbox_root=tmp_path,
        pipeline_or_spec_path=pipeline_path,
        metadata_path=metadata_path,
    )

    assert written.name.endswith(".json.pht-tune")
    assert ".phenotypic-gui" in str(written)
    reloaded = TuningSpec.model_validate_json(written.read_text(encoding="utf-8"))
    assert isinstance(reloaded.scorer, QCScorer)
    assert reloaded.scorer.availability() is True
    assert reloaded.scorer.check.metadata == str(metadata_path)
    assert reloaded.search_space.knobs


def test_write_authored_setup_spec_requires_metadata_file(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.json.pht-pipe"
    pipeline_path.write_text(
        ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()]).to_json(),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        write_authored_setup_spec(
            sandbox_root=tmp_path,
            pipeline_or_spec_path=pipeline_path,
            metadata_path=tmp_path / "missing.csv",
        )


def test_write_authored_setup_spec_preserves_existing_spec_search_space(
    tmp_path: Path,
):
    metadata_path = _metadata(tmp_path / "layout.csv")
    original = TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()]),
        search_space=SearchSpace(
            knobs=(
                Knob(
                    target=Param(op=0, field="sigma"),
                    domain=Fixed(value=2.5),
                ),
            )
        ),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=str(metadata_path),
                groupby=["MetadataImage_ImageName"],
            )
        ),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )
    spec_path = tmp_path / "existing.json.pht-tune"
    spec_path.write_text(original.model_dump_json(), encoding="utf-8")

    written = write_authored_setup_spec(
        sandbox_root=tmp_path,
        pipeline_or_spec_path=spec_path,
        metadata_path=None,
    )

    reloaded = TuningSpec.model_validate_json(written.read_text(encoding="utf-8"))
    assert reloaded.search_space == original.search_space
    assert type(reloaded.scorer) is type(original.scorer)
    assert reloaded.scorer.check.metadata == str(metadata_path)
    assert reloaded.scorer.check.groupby == ["MetadataImage_ImageName"]
    assert reloaded.strategy == original.strategy
    assert reloaded.budget == original.budget
    assert reloaded.evaluator == original.evaluator


def test_existing_spec_replaces_scorer_only_when_requested(tmp_path: Path):
    original_metadata = _metadata(tmp_path / "original.csv")
    replacement_metadata = _metadata(tmp_path / "replacement.csv")
    original = TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()]),
        search_space=SearchSpace(
            knobs=(
                Knob(
                    target=Param(op=0, field="sigma"),
                    domain=Fixed(value=2.5),
                ),
            )
        ),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=str(original_metadata),
                groupby=["MetadataImage_ImageName"],
            )
        ),
        evaluator=Evaluator(),
        strategy=RandomConfig(seed=7, n_trials=13),
        budget=Budget(n_trials=11, max_failures=2),
    )
    spec_path = tmp_path / "existing.json.pht-tune"
    spec_path.write_text(original.model_dump_json(), encoding="utf-8")

    written = write_authored_setup_spec(
        sandbox_root=tmp_path,
        pipeline_or_spec_path=spec_path,
        metadata_path=replacement_metadata,
        replace_scorer=True,
    )

    reloaded = TuningSpec.model_validate_json(written.read_text(encoding="utf-8"))
    assert reloaded.scorer.check.metadata == str(replacement_metadata)
    assert reloaded.scorer.check.metadata != str(original_metadata)
    assert reloaded.search_space == original.search_space
    assert reloaded.strategy == original.strategy
    assert reloaded.budget == original.budget


def test_build_authored_setup_spec_reports_all_validation_issues(tmp_path: Path):
    pipeline = ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()])
    pipeline_path = tmp_path / "pipeline.json.pht-pipe"
    pipeline_path.write_text(pipeline.to_json(), encoding="utf-8")
    first_knob = infer_search_space(pipeline).knobs[0]

    result = build_authored_setup_spec(
        pipeline_or_spec_path=pipeline_path,
        metadata_path=None,
        edits={
            first_knob.key: {
                "low": 10,
                "high": 1,
                "tunable": True,
            }
        },
    )

    assert result.spec is None
    assert any("Metadata is required" in issue for issue in result.issues)
    assert any("high" in issue or "greater" in issue for issue in result.issues)


def test_setup_path_precedence_and_same_path_reselection(tmp_path: Path):
    sandbox = SandboxRoot.from_path(tmp_path)
    typed = tmp_path / "typed.json.pht-pipe"
    picked = tmp_path / "picked.json.pht-pipe"
    shared = tmp_path / "shared.json.pht-pipe"
    for path in (typed, picked, shared):
        path.write_text(ImagePipeline(ops=[]).to_json(), encoding="utf-8")

    first_payload = setup_path_payload(sandbox, picked, kind="pipeline")
    second_payload = setup_path_payload(sandbox, picked, kind="pipeline")
    assert first_payload is not None
    assert second_payload is not None
    assert first_payload["selected_at"] != second_payload["selected_at"]

    typed_resolution = resolve_setup_path(
        sandbox=sandbox,
        kind="pipeline",
        typed_path=str(typed),
        picker_payload=first_payload,
        shared_payload=str(shared),
    )
    assert typed_resolution.path == typed
    assert typed_resolution.source == "typed"

    picker_resolution = resolve_setup_path(
        sandbox=sandbox,
        kind="pipeline",
        typed_path="",
        picker_payload=first_payload,
        shared_payload=str(shared),
    )
    assert picker_resolution.path == picked
    assert picker_resolution.source == "picker"
