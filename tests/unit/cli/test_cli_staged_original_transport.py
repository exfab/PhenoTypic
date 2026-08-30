"""Staged-worker transport for full-forward original retention."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def test_staged_worker_transports_drop_originals_to_stage1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from phenotypic._cli import _cli_staged_slurm_worker as worker

    pipeline_identity = {
        "source_path": str((tmp_path / "source.json").resolve()),
        "sha256": "a" * 64,
    }
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        worker, "preload_custom_operation_modules", lambda: None
    )
    monkeypatch.setattr(
        worker, "load_staged_manifest", lambda path: [("ds", "image")]
    )

    def _stage1_spy(*args: Any, **kwargs: Any) -> None:
        del args
        seen.update(kwargs)

    monkeypatch.setattr(worker, "run_stage1_step", _stage1_spy)

    result = worker.main(
        [
            "--stage",
            "1",
            "--pipeline",
            str(tmp_path / "pipeline.json"),
            "--output-dir",
            str(tmp_path / "out"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--index",
            "0",
            "--epoch",
            "epoch-1",
            "--provenance-pipeline-source-path",
            pipeline_identity["source_path"],
            "--provenance-pipeline-sha256",
            pipeline_identity["sha256"],
            "--drop-originals",
        ]
    )

    assert result == 0
    assert seen["drop_originals"] is True
    assert seen["pipeline_identity"] == pipeline_identity


def test_staged_slurm_emits_drop_originals_only_for_stage1(
    tmp_path: Path,
) -> None:
    from phenotypic._cli._cli_staged_orchestration import StagedManifestEntry
    from phenotypic._cli._cli_staged_slurm import generate_staged_scripts

    pipeline_identity = {
        "source_path": str((tmp_path / "source.json").resolve()),
        "sha256": "b" * 64,
    }
    scripts = generate_staged_scripts(
        pipeline_path=tmp_path / "pipeline.json",
        datasets_manifest=[
            StagedManifestEntry(
                "ds",
                "image.tif",
                "image",
                str(tmp_path / "image.tif"),
            )
        ],
        output_dir=tmp_path / "out",
        image_type="Image",
        cpu_slurm_args={"slurm_partition": "cpu"},
        gpu_slurm_args={"slurm_partition": "gpu"},
        n_shards=1,
        array_limit=10,
        epoch="epoch-1",
        drop_originals=True,
        pipeline_identity=pipeline_identity,
    )

    stage1 = scripts["stage1"][0].read_text(encoding="utf-8")
    stage2 = scripts["stage2"].read_text(encoding="utf-8")
    stage3 = scripts["stage3"][0].read_text(encoding="utf-8")
    assert "--drop-originals" in stage1
    assert "--provenance-pipeline-source-path" in stage1
    assert pipeline_identity["source_path"] in stage1
    assert pipeline_identity["sha256"] in stage1
    assert "--drop-originals" not in stage2
    assert "--drop-originals" not in stage3
    assert "--provenance-pipeline-source-path" not in stage2
    assert "--provenance-pipeline-source-path" not in stage3
