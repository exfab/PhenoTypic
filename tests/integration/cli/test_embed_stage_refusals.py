"""``--embed-stage`` refusals that need a real run on the stub backbone (#282 P4 T1). [torch]

Run in the foundation venv (``TPT``); in the shared env they skip.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tests._tiny_run import run_cli, with_stage4_dino_objects  # noqa: E402


def test_r3_refuses_a_switched_continuation(tmp_path, tiny_inputs, stage4_cpu):
    out = tmp_path / "out"
    pipe = with_stage4_dino_objects(tmp_path)
    first = run_cli(tiny_inputs, out, pipeline=pipe)                         # local auto: inline, recorded
    assert first.exit_code == 0, first.output
    result = run_cli(tiny_inputs, out, pipeline=pipe, extra=["--embed-stage", "gpu"])
    assert result.exit_code == 2 and "started with --embed-stage inline" in result.output, result.output


def test_no_cuda_stops_before_stage1(tmp_path, tiny_inputs, param_stub_backbone, monkeypatch):
    from phenotypic._cli._cli_failure_tracker import read_terminal_failures

    monkeypatch.delenv("PHENOTYPIC_STAGE4_DEVICE", raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, pipeline=with_stage4_dino_objects(tmp_path), extra=["--embed-stage", "gpu"])
    assert result.exit_code == 1 and "CUDA" in result.output, result.output
    errors = [line for line in result.output.splitlines() if line.startswith("Error:")]
    assert len(errors) == 1, result.output
    assert not list(out.rglob("*.ome.zarr")) and read_terminal_failures(out) == []


def test_a_slurm_detector_dino_run_resolves_inline_and_finishes(tmp_path, tiny_inputs, stage4_cpu, monkeypatch):
    """Review C1 scenario 1: the most common production shape, all defaults, on SLURM.

    A ``GpuDetector`` plus a DINO embedder used to resolve ``gpu``, so every
    Stage-3 task deferred to a Stage 4 no SLURM path runs, and the run could
    never finish. Under the interim gate ``auto`` resolves ``inline``, with one
    note; the staged submitter records that in ``job_metadata.json``, which is
    where its workers read deferral; and Stages 1-3, run here as the SLURM
    workers run them, publish every image's record with its embeddings.

    P5 T2 flips this test when it lifts SLURM in ``stage4_supported``.
    """
    from datetime import datetime
    from pathlib import Path

    from phenotypic._cli import _cli_staged_slurm_worker as worker
    from phenotypic._cli._cli_staged_slurm import _write_staged_job_metadata, staged_manifest_entry
    from phenotypic._cli._cli_types import ExecutionResults
    from phenotypic.sdk_ import RunExitCode
    from tests._tiny_run import all_records_valid, run_items, store_has_embeddings

    seen: dict = {}

    class _Submitted:
        def execute(self, datasets, output_dir):
            seen["datasets"] = datasets
            now = datetime.now()
            return ExecutionResults(datasets={}, total_images=2, total_completed=0, total_failed=0,
                                    execution_mode="slurm", start_time=now, end_time=now, submitted=True)

    def _strategy(config, output_manager):
        seen["config"] = config
        return _Submitted()

    monkeypatch.setattr("phenotypic.phenotypicCLI.create_execution_strategy", _strategy)
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    result = run_cli(tiny_inputs, out, pipeline=pipe, slurm=True, extra=["--skip-validation"])
    assert result.exit_code == RunExitCode.SUBMITTED, result.output
    assert "Note: Stage 4 on SLURM is not yet available, so --embed-stage auto resolves to inline." in result.output
    config = seen["config"]
    assert config.embed_stage == "inline"

    _write_staged_job_metadata(
        datasets=seen["datasets"], output_dir=out, config=config,
        scripts={"stage1": [Path("s1.sh")], "stage2": Path("s2.sh"), "stage3": [Path("s3.sh")],
                 "finalizer": Path("f.sh"), "controller": Path("c.sh")},
        job_ids=[], start_time=datetime.now(), epoch="unfenced",
    )
    manifest = [staged_manifest_entry(config, ds.name, image) for ds in seen["datasets"] for image in ds.images]
    assert len(manifest) == 2
    common = {"pipeline_path": pipe, "output_dir": out, "image_type": "Image", "manifest": manifest}
    for index in range(len(manifest)):
        worker.run_stage1_step(**common, index=index)
    worker.run_stage2_shard(**common, shard_index=0, n_shards=1)
    for index in range(len(manifest)):
        worker.run_stage3_step(**common, index=index)

    assert all_records_valid(out)
    assert all(store_has_embeddings(out, item) for item in run_items(out))
