from pathlib import Path

import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.gui.tune._setup_authoring import write_authored_setup_spec
from phenotypic.tune import (
    Evaluator,
    Fixed,
    Knob,
    SearchSpace,
    TuningSpec,
)
from phenotypic.tune.score import QCScorer
from phenotypic.tune.strategy import GridConfig
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
        metadata_path=metadata_path,
    )

    reloaded = TuningSpec.model_validate_json(written.read_text(encoding="utf-8"))
    assert reloaded.search_space == original.search_space
    assert isinstance(reloaded.scorer, QCScorer)
