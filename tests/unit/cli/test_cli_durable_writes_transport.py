"""``--no-durable-writes`` must survive the trip into a fresh worker process.

Spec §3.7 / Phase 3 Task 3.7. Every SLURM execution path spawns workers that
build their own :class:`OutputManager` from their own command line. An *unset*
flag resolves correctly on its own there -- ``durable_writes_enabled`` re-reads
``SLURM_JOB_ID`` in the worker. But ``--no-durable-writes`` is the case a user
reaches for *precisely because* they are on fast local scratch inside a job,
and that value exists only in the submitting process. If the submitter does not
emit it, every worker silently re-enables fsync and the flag appears to do
nothing on the one execution path where it costs the most.

So each test here asserts against ``SLURM_JOB_ID`` **set**: the interesting
failure is not "the value was dropped", it is "the value was dropped and the
environment then supplied the opposite one".

Two transports, one per submission path:

* ``_cli_slurm_array_scripts`` -> ``_cli_process_single`` (the ordinary
  per-image SLURM array),
* ``_cli_staged_slurm`` -> ``_cli_staged_slurm_worker`` (staged GPU Stages 1/3).

A third case, ``_cli_slurm_scripts.generate_image_processing_script``, had a
test here covering the flag its standalone per-image script emitted. That
generator and its only caller ``generate_all_image_scripts`` were an
unreachable subtree, deleted in Phase 6 of the OME-Zarr store change; the test
went with them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from phenotypic.sdk_.ngff_ import durable_writes_enabled


@pytest.fixture(autouse=True)
def _inside_a_slurm_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every assertion below is about beating the auto-detection."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)


# ---------------------------------------------------------------------------
# Transport 1: the ordinary per-image worker (``_cli_process_single``)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], None),
        (["--durable-writes"], True),
        (["--no-durable-writes"], False),
    ],
)
def test_process_single_worker_builds_a_manager_carrying_the_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    argv: list[str],
    expected: bool | None,
) -> None:
    from click.testing import CliRunner

    from phenotypic._cli import _cli_process_single

    seen: dict[str, Any] = {}

    def _spy(**kwargs: Any) -> None:
        seen["manager"] = kwargs["output_manager"]

    monkeypatch.setattr(
        _cli_process_single, "process_single_image_core", _spy
    )
    image = next(synth_one_level_input.rglob("*.tif"))
    result = CliRunner().invoke(
        _cli_process_single.main,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--image",
            str(image),
            "--output-dir",
            str(tmp_path / "out"),
            "--dataset-name",
            "ds",
            "--image-type",
            "Image",
            *argv,
        ],
    )

    assert "manager" in seen, result.output
    assert seen["manager"].durable_writes is expected
    # The point of the flag, not merely its transport: with SLURM_JOB_ID set,
    # only an explicit False changes what the promote does.
    assert durable_writes_enabled(seen["manager"].durable_writes) is (
        expected is not False
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, ""), (True, "--durable-writes"), (False, "--no-durable-writes")],
)
def test_slurm_array_script_emits_the_flag(
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    make_exec_config: Callable[..., Any],
    value: bool | None,
    expected: str,
) -> None:
    from phenotypic._cli._cli_slurm_array_scripts import (
        generate_array_job_script,
    )
    from phenotypic._cli._cli_types import Dataset

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    image = next(synth_one_level_input.rglob("*.tif"))
    config = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=synth_one_level_input,
        output_dir=output_dir,
        force_local=False,
        slurm_args={"slurm_partition": "short"},
        durable_writes=value,
    )
    dataset = Dataset("ds", [image], synth_one_level_input, output_dir)

    script = generate_array_job_script(
        dataset, (0, 1), config, output_dir
    ).read_text()

    if expected:
        assert expected in script
    else:
        assert "durable-writes" not in script


# ---------------------------------------------------------------------------
# Transport 2: the staged GPU worker (``_cli_staged_slurm_worker``)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], None),
        (["--durable-writes"], True),
        (["--no-durable-writes"], False),
    ],
)
@pytest.mark.parametrize("stage", [1, 3])
def test_staged_worker_argv_reaches_the_stage_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argv: list[str],
    expected: bool | None,
    stage: int,
) -> None:
    """The argparse surface is the only way the value can cross the process."""
    from phenotypic._cli import _cli_staged_slurm_worker as worker

    seen: dict[str, Any] = {}
    target = "run_stage1_step" if stage == 1 else "run_stage3_step"

    def _spy(*args: Any, **kwargs: Any) -> None:
        seen["durable_writes"] = kwargs.get("durable_writes", "MISSING")

    monkeypatch.setattr(worker, target, _spy)
    monkeypatch.setattr(worker, "preload_custom_operation_modules", lambda: None)
    monkeypatch.setattr(worker, "load_staged_manifest", lambda _p: [])

    worker.main(
        [
            "--stage",
            str(stage),
            "--pipeline",
            str(tmp_path / "pipe.json"),
            "--output-dir",
            str(tmp_path / "out"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--index",
            "0",
            "--epoch",
            "e1",
            *argv,
        ]
    )

    assert seen["durable_writes"] is expected


@pytest.mark.parametrize("stage", [1, 3])
@pytest.mark.parametrize("value", [None, True, False])
def test_staged_stage_step_hands_the_flag_to_its_output_manager(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: int,
    value: bool | None,
) -> None:
    """A transported value that stops at the stage function is still inert."""
    from phenotypic._cli import _cli_staged_slurm_worker as worker
    from phenotypic._cli._cli_staged_slurm_worker import StagedManifestEntry

    class _Stop(Exception):
        """Abort the step once its manager exists; the rest needs real data."""

    seen: dict[str, Any] = {}
    real_from_config = worker.OutputManager.from_config

    def _spy(*args: Any, **kwargs: Any):
        seen["manager"] = real_from_config(*args, **kwargs)
        raise _Stop()

    monkeypatch.setattr(worker.OutputManager, "from_config", _spy)
    # Stage 1 loads and splits the pipeline before it builds its manager; the
    # split is irrelevant to durability, so stub it rather than ship a GPU
    # pipeline fixture into a transport test.
    monkeypatch.setattr(
        worker.ImagePipeline, "from_json", staticmethod(lambda _p: None)
    )
    monkeypatch.setattr(worker, "split_pipeline_at_gpu", lambda _p: object())
    entry = StagedManifestEntry(
        dataset="ds",
        image_name="img.tiff",
        stem="img",
        input_path=str(tmp_path / "img.tiff"),
    )
    step = worker.run_stage1_step if stage == 1 else worker.run_stage3_step

    with pytest.raises(_Stop):
        step(
            tmp_path / "pipe.json",
            tmp_path / "out",
            "Image",
            [entry],
            0,
            ".tiff",
            epoch=None,
            durable_writes=value,
        )

    assert seen["manager"].durable_writes is value


@pytest.mark.parametrize("value", [None, True, False])
def test_staged_submitter_hands_the_flag_to_the_script_generator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: bool | None
) -> None:
    """The staged strategy is the only caller of ``generate_staged_scripts``.

    Everything below it is already pinned, but a submitter that never passes
    the value makes all of that unreachable: the scripts come out without the
    flag and every staged worker re-detects SLURM instead.
    """
    from phenotypic._cli import _cli_staged_slurm as staged

    captured: dict[str, Any] = {}

    def _fake_generate(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "stage1": [tmp_path / "s1.sh"],
            "stage2": tmp_path / "s2.sh",
            "stage3": [tmp_path / "s3.sh"],
            "finalizer": tmp_path / "finalizer.sh",
            "controller": tmp_path / "controller.sh",
            "controller_config": tmp_path / "controller.json",
        }

    monkeypatch.setattr(staged, "get_slurm_array_limit", lambda: 1000)
    monkeypatch.setattr(staged, "get_slurm_max_submit_jobs", lambda: 5)
    monkeypatch.setattr(staged, "generate_staged_scripts", _fake_generate)
    monkeypatch.setattr(
        staged, "submit_with_intent", lambda *a, **k: "100"
    )

    config = type(
        "Config",
        (),
        {
            "pipeline_json": tmp_path / "pipeline.json",
            "image_type": "Image",
            "slurm_args": {},
            "gpu_slurm_args": {},
            "gpu_shards": 1,
            "ext": None,
            "overlay_alpha": 0.3,
            "include_dataset_column": False,
            "metadata_csv": None,
            "no_qc": False,
            "input_path": tmp_path / "inputs",
            "resume": False,
            "restart": False,
            "staged_resume_phase": None,
            "staged_finalizer_only": False,
            "staged_stage3_markers": True,
            "wait": False,
            "full_dataset_inventory": {},
            "nrows": None,
            "ncols": None,
            "durable_writes": value,
        },
    )()
    strategy = object.__new__(staged.StagedSlurmStrategy)
    strategy.config = config

    strategy.execute([], tmp_path)

    assert captured["durable_writes"] is value


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, ""), (True, "--durable-writes"), (False, "--no-durable-writes")],
)
def test_staged_worker_body_emits_the_flag(
    tmp_path: Path, value: bool | None, expected: str
) -> None:
    from phenotypic._cli._cli_staged_slurm import _stage_worker_body

    body = _stage_worker_body(
        "python",
        1,
        tmp_path / "pipe.json",
        tmp_path,
        "Image",
        tmp_path / "manifest.json",
        ".tiff",
        "e1",
        False,
        durable_writes=value,
    )

    if expected:
        assert expected in body
    else:
        assert "durable-writes" not in body
