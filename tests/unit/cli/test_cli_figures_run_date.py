"""The run's initial CLI call: minted once, recorded, and read back by every
worker -- never carried on a command line.

Figures spec §1a: the call's UTC date is the figure-folder ``{date}``, and its
UTC timestamp and pid go into each store's run entry. The CLI records all
three in ``state.config`` before it submits or launches any worker; a resume
reuses them. Measure mode keeps no state, so a measure-mode SLURM invocation
records them in ``job_metadata.json`` before fan-out. In-process workers get
them on the CLI's ``OutputManager``; a worker in another process reads them
back (``recorded_run_initiation`` / ``metadata_run_initiation``). User
decision: nothing here travels on a command line, hidden or not.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from phenotypic.sdk_._image_figures import RunInitiation

DATE = "2026-09-22"
CALL = RunInitiation(date=DATE, at_utc="2026-09-22T23:59:58.123Z", pid=4242)


@pytest.fixture
def recorded_state(
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    make_exec_config: Callable[..., Any],
) -> Callable[..., Path]:
    """Write a real processing state under an output root, as the CLI does."""

    def _write(out: Path, initiation: RunInitiation | None = CALL) -> Path:
        from phenotypic._cli._cli_state_management import (
            create_initial_state,
            save_processing_state,
        )
        from phenotypic._cli._cli_types import Dataset

        config = make_exec_config(
            pipeline_json=simple_pipeline_json,
            input_path=synth_one_level_input,
            output_dir=out,
            run_initiation=initiation,
        )
        image = next(synth_one_level_input.rglob("*.tif"))
        state = create_initial_state(
            config,
            [Dataset("ds", [image], synth_one_level_input, out)],
            out,
            identity=SimpleNamespace(processing_generation="g", restart_epoch=0),
        )
        return save_processing_state(state, out)

    return _write


# ---------------------------------------------------------------------------
# Minting and reuse
# ---------------------------------------------------------------------------


def test_one_instant_gives_the_date_the_timestamp_and_the_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    from phenotypic.sdk_ import _image_figures

    instant = datetime(2026, 9, 22, 23, 59, 59, 999000, tzinfo=timezone.utc)
    monkeypatch.setattr(_image_figures, "_utc_now", lambda: instant)
    call = _image_figures.mint_run_initiation()
    assert call == RunInitiation(
        date="2026-09-22", at_utc="2026-09-22T23:59:59.999Z", pid=os.getpid()
    )


def test_a_resume_reuses_the_recorded_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from phenotypic import phenotypicCLI
    from phenotypic.sdk_ import _image_figures

    monkeypatch.setattr(
        _image_figures, "_utc_now",
        lambda: datetime(2026, 10, 30, tzinfo=timezone.utc),
    )
    state = SimpleNamespace(config={
        "figures_run_date": CALL.date,
        "initiated_at_utc": CALL.at_utc,
        "initiated_pid": CALL.pid,
    })
    assert phenotypicCLI._run_initiation(state) == CALL


@pytest.mark.parametrize(
    "state",
    [
        None,
        SimpleNamespace(config={}),
        SimpleNamespace(config={"figures_run_date": None}),
        SimpleNamespace(config={"figures_run_date": "2026-13-45"}),
    ],
    ids=["fresh-restart-or-measure", "state-before-the-call", "unrecorded", "malformed"],
)
def test_anything_else_is_a_new_call(monkeypatch: pytest.MonkeyPatch, state: Any) -> None:
    import os

    from phenotypic import phenotypicCLI
    from phenotypic.sdk_ import _image_figures

    monkeypatch.setattr(
        _image_figures, "_utc_now",
        lambda: datetime(2026, 10, 30, 1, 2, 3, tzinfo=timezone.utc),
    )
    assert phenotypicCLI._run_initiation(state) == RunInitiation(
        "2026-10-30", "2026-10-30T01:02:03.000Z", os.getpid()
    )


def test_the_state_the_cli_writes_is_what_a_worker_reads(
    tmp_path: Path, recorded_state: Callable[..., Path]
) -> None:
    from phenotypic._cli._cli_state_management import (
        load_processing_state,
        recorded_run_initiation,
    )

    out = tmp_path / "out"
    recorded_state(out)
    config = load_processing_state(out).config
    assert (
        config["figures_run_date"], config["initiated_at_utc"], config["initiated_pid"]
    ) == (CALL.date, CALL.at_utc, CALL.pid)
    assert recorded_run_initiation(out) == CALL


def test_no_recorded_call_reads_as_none(
    tmp_path: Path, recorded_state: Callable[..., Path]
) -> None:
    from phenotypic._cli._cli_state_management import recorded_run_initiation

    assert recorded_run_initiation(tmp_path / "absent") is None
    recorded_state(tmp_path / "unrecorded", None)
    assert recorded_run_initiation(tmp_path / "unrecorded") is None
    corrupt = recorded_state(tmp_path / "corrupt")
    corrupt.write_text("{not json", encoding="utf-8")
    assert recorded_run_initiation(tmp_path / "corrupt") is None


def _edit_recorded_call(state_file: Path, **values: Any) -> None:
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["config"].update(values)
    state_file.write_text(json.dumps(state), encoding="utf-8")


@pytest.mark.parametrize(
    "date", ["2026-13-45", "20260922", "2026-9-22", "../escape", 20260922]
)
def test_a_malformed_recorded_date_reads_as_unrecorded(
    tmp_path: Path, recorded_state: Callable[..., Path], date: Any
) -> None:
    """MINOR-2: the date names a folder, so a corrupt one never reaches a
    worker -- which then names its run as an unrecorded state would."""
    from phenotypic._cli._cli_state_management import (
        metadata_run_initiation,
        recorded_run_initiation,
    )
    from phenotypic.sdk_ import JobMetadataKey, job_metadata_path

    out = tmp_path / "out"
    _edit_recorded_call(recorded_state(out), figures_run_date=date)
    assert recorded_run_initiation(out) is None
    job_metadata_path(out).parent.mkdir(parents=True, exist_ok=True)
    job_metadata_path(out).write_text(
        json.dumps({JobMetadataKey.FIGURES_RUN_DATE: date}), encoding="utf-8"
    )
    assert metadata_run_initiation(out) is None


@pytest.mark.parametrize(
    ("at_utc", "pid"),
    [("yesterday", -1), ("2026-09-22T12:00:00", 0), (None, True), (123, "4242")],
)
def test_a_malformed_timestamp_or_pid_reads_as_absent(
    tmp_path: Path, recorded_state: Callable[..., Path], at_utc: Any, pid: Any
) -> None:
    from phenotypic._cli._cli_state_management import recorded_run_initiation

    out = tmp_path / "out"
    _edit_recorded_call(recorded_state(out), initiated_at_utc=at_utc, initiated_pid=pid)
    assert recorded_run_initiation(out) == RunInitiation(date=DATE)


def test_the_call_is_not_part_of_the_work_id(
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    make_exec_config: Callable[..., Any],
) -> None:
    """A new day must not invalidate continuation (spec §1a)."""
    from phenotypic._cli._cli_failure_tracker import (
        processing_configuration_digest,
    )

    digests = {
        processing_configuration_digest(
            make_exec_config(
                pipeline_json=simple_pipeline_json,
                input_path=synth_one_level_input,
                run_initiation=value,
            )
        )
        for value in (None, CALL, RunInitiation("2026-10-30", "x", 1))
    }
    assert len(digests) == 1


# ---------------------------------------------------------------------------
# No command line carries it
# ---------------------------------------------------------------------------


def test_the_slurm_array_script_never_mentions_the_call(
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    make_exec_config: Callable[..., Any],
) -> None:
    from phenotypic._cli._cli_slurm_array_scripts import generate_array_job_script
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
        run_initiation=CALL,
    )
    dataset = Dataset("ds", [image], synth_one_level_input, output_dir)
    script = generate_array_job_script(dataset, (0, 1), config, output_dir).read_text()
    for word in ("figures-run-date", "figures_run_date", "initiat", DATE, CALL.at_utc):
        assert word not in script


def test_no_staged_script_mentions_the_call(tmp_path: Path) -> None:
    """Every script ``generate_staged_scripts`` writes, controller included."""
    from phenotypic._cli._cli_stage2_token import detector_slot
    from phenotypic._cli._cli_staged_orchestration import StagedManifestEntry
    from phenotypic._cli._cli_staged_slurm import generate_staged_scripts

    scripts = generate_staged_scripts(
        pipeline_path=tmp_path / "pipeline.json",
        detector_slot=detector_slot(("FakeGpuDetector",)),
        datasets_manifest=[
            StagedManifestEntry("ds", "image.tif", "image", str(tmp_path / "image.tif"))
        ],
        output_dir=tmp_path / "out",
        image_type="Image",
        cpu_slurm_args={"slurm_partition": "cpu"},
        gpu_slurm_args={"slurm_partition": "gpu"},
        n_shards=1,
        array_limit=10,
        epoch="epoch-1",
    )
    written = [
        path
        for value in scripts.values()
        for path in (value if isinstance(value, list) else [value])
        if isinstance(path, Path) and path.is_file()
    ]
    assert written
    for path in written:
        text = path.read_text(encoding="utf-8")
        for word in ("figures-run-date", "figures_run_date", "initiat"):
            assert word not in text, path


def test_neither_worker_accepts_the_call_as_an_argument(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from phenotypic._cli import _cli_process_single
    from phenotypic._cli import _cli_staged_slurm_worker as worker

    result = CliRunner().invoke(_cli_process_single.main, ["--figures-run-date", DATE])
    assert "No such option: --figures-run-date" in result.output
    with pytest.raises(SystemExit):
        worker.main([
            "--stage", "1", "--pipeline", "p", "--output-dir", str(tmp_path),
            "--manifest", "m", "--index", "0", "--epoch", "e",
            "--figures-run-date", DATE,
        ])


# ---------------------------------------------------------------------------
# Workers in another process read the recorded call
# ---------------------------------------------------------------------------


def _invoke_worker(image: Path, pipeline: Path, out: Path, *argv: str):
    from click.testing import CliRunner

    from phenotypic._cli import _cli_process_single

    return CliRunner().invoke(
        _cli_process_single.main,
        [
            "--pipeline", str(pipeline),
            "--image", str(image),
            "--output-dir", str(out),
            "--dataset-name", "ds",
            "--image-type", "Image",
            *argv,
        ],
    )


def test_the_array_worker_builds_its_manager_from_the_recorded_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    recorded_state: Callable[..., Path],
) -> None:
    from phenotypic._cli import _cli_process_single

    seen: dict[str, Any] = {}

    def _spy(**kwargs: Any) -> None:
        seen["manager"] = kwargs["output_manager"]

    monkeypatch.setattr(_cli_process_single, "process_single_image_core", _spy)
    out = tmp_path / "out"
    recorded_state(out)
    image = next(synth_one_level_input.rglob("*.tif"))
    result = _invoke_worker(image, simple_pipeline_json, out)
    assert "manager" in seen, result.output
    assert seen["manager"].run_initiation == CALL


def test_the_process_worker_hands_the_recorded_call_to_its_core(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    recorded_state: Callable[..., Path],
) -> None:
    from phenotypic._cli import _cli_process_single

    seen: dict[str, Any] = {}

    def _spy(**kwargs: Any) -> None:
        seen["call"] = kwargs.get("run_initiation", "MISSING")
        raise RuntimeError("stop after the call")

    monkeypatch.setattr(_cli_process_single, "process_single_apply_only_core", _spy)
    out = tmp_path / "out"
    recorded_state(out)
    image = next(synth_one_level_input.rglob("*.tif"))
    _invoke_worker(
        image, simple_pipeline_json, out,
        "--mode", "process", "--layer", "rgb",
        "--input-root", str(synth_one_level_input),
    )
    assert seen["call"] == CALL


def _write_job_metadata(out: Path, payload: dict | str) -> None:
    from phenotypic.sdk_ import job_metadata_path

    path = job_metadata_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
    )


def _measure_manager(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pipeline: Path, out: Path
):
    from phenotypic._cli import _cli_process_single

    seen: dict[str, Any] = {}

    def _spy(**kwargs: Any) -> None:
        seen["manager"] = kwargs["output_manager"]

    monkeypatch.setattr(_cli_process_single, "process_single_store_measure_core", _spy)
    store = tmp_path / "plate.ome.zarr"
    store.mkdir(exist_ok=True)
    result = _invoke_worker(store, pipeline, out, "--mode", "measure")
    assert "manager" in seen, result.output
    return seen["manager"]


def test_a_measure_worker_reads_its_call_from_the_job_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    recorded_state: Callable[..., Path],
) -> None:
    """Not from the state: the state under a measured tree is the earlier run's."""
    from phenotypic.sdk_ import JobMetadataKey

    out = tmp_path / "out"
    recorded_state(out, RunInitiation("2026-01-01", "2026-01-01T00:00:00.000Z", 1))
    _write_job_metadata(out, {
        JobMetadataKey.START_TIME: "2026-09-22T16:59:58.123",
        JobMetadataKey.FIGURES_RUN_DATE: CALL.date,
        JobMetadataKey.INITIATED_AT_UTC: CALL.at_utc,
        JobMetadataKey.INITIATED_PID: CALL.pid,
    })
    manager = _measure_manager(monkeypatch, tmp_path, simple_pipeline_json, out)
    assert manager.run_initiation == CALL


@pytest.mark.parametrize("metadata", [None, "{not json", {"start_time": "x"}],
                         ids=["absent", "corrupt", "no-call"])
def test_a_measure_worker_without_a_recorded_call_has_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    recorded_state: Callable[..., Path],
    metadata: Any,
) -> None:
    """The measure core then falls back to today in UTC."""
    out = tmp_path / "out"
    recorded_state(out)
    if metadata is not None:
        _write_job_metadata(out, metadata)
    manager = _measure_manager(monkeypatch, tmp_path, simple_pipeline_json, out)
    assert manager.run_initiation is None


@pytest.mark.parametrize("stage", [1, 3])
def test_a_staged_step_builds_its_manager_from_the_recorded_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: int,
    recorded_state: Callable[..., Path],
) -> None:
    from phenotypic._cli import _cli_staged_slurm_worker as worker
    from phenotypic._cli._cli_staged_slurm_worker import StagedManifestEntry

    class _Stop(Exception):
        """Abort the step once its manager exists."""

    seen: dict[str, Any] = {}
    real_from_config = worker.OutputManager.from_config

    def _spy(*args: Any, **kwargs: Any):
        seen["manager"] = real_from_config(*args, **kwargs)
        raise _Stop()

    monkeypatch.setattr(worker.OutputManager, "from_config", _spy)
    monkeypatch.setattr(worker.ImagePipeline, "from_json", staticmethod(lambda _p: None))
    monkeypatch.setattr(worker, "split_pipeline_at_gpu", lambda _p: object())
    out = tmp_path / "out"
    recorded_state(out)
    entry = StagedManifestEntry(
        dataset="ds", image_name="img.tiff", stem="img",
        input_path=str(tmp_path / "img.tiff"),
    )
    step = worker.run_stage1_step if stage == 1 else worker.run_stage3_step
    with pytest.raises(_Stop):
        step(tmp_path / "pipe.json", out, "Image", [entry], 0, ".tiff", epoch=None)
    assert seen["manager"].run_initiation == CALL


# ---------------------------------------------------------------------------
# The measure-mode SLURM submitter records the call before fan-out
# ---------------------------------------------------------------------------


def test_a_measure_submission_records_the_call_before_fan_out(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    make_exec_config: Callable[..., Any],
) -> None:
    from phenotypic._cli import _cli_execution_strategies as strategies
    from phenotypic._cli import _cli_finalize_fanout
    from phenotypic._cli._cli_output_manager import OutputManager
    from phenotypic._cli._cli_state_management import metadata_run_initiation
    from phenotypic._cli._cli_types import Dataset

    class _Stop(Exception):
        """End the run at submission."""

    seen: dict[str, Any] = {}

    def _at_submission(**kwargs: Any):
        seen["call"] = metadata_run_initiation(out)
        raise _Stop()

    def _fanout(*args: Any, **kwargs: Any) -> None:
        seen["fanout_saw"] = metadata_run_initiation(out)

    out = tmp_path / "out"
    out.mkdir()
    store = tmp_path / "plate.ome.zarr"
    store.mkdir()
    script = tmp_path / "chunk.sh"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    monkeypatch.setattr(strategies, "get_slurm_array_limit", lambda: 1000)
    monkeypatch.setattr(
        strategies, "generate_all_array_job_scripts",
        lambda datasets, config, output_dir, limit: {"ds": [script]},
    )
    monkeypatch.setattr(strategies, "generate_terminal_finalizer_script", lambda *a, **k: script)
    monkeypatch.setattr(strategies, "submit_slurm_script_chain", _at_submission)
    monkeypatch.setattr(_cli_finalize_fanout, "begin_aggregation_fanout", _fanout)
    config = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=tmp_path,
        output_dir=out,
        force_local=False,
        slurm_args={"slurm_partition": "short"},
        measure_only=True,
        run_initiation=CALL,
    )
    strategy = strategies.AutonomousSLURMStrategy(config, OutputManager.from_config(out, ".tiff"))
    with pytest.raises(_Stop):
        strategy.execute([Dataset("ds", [store], tmp_path, out)], out)
    assert seen["fanout_saw"] == CALL
    assert seen["call"] == CALL


# ---------------------------------------------------------------------------
# In-process workers: the CLI's own manager, or the strategy, hands it on
# ---------------------------------------------------------------------------


def test_a_local_process_run_hands_the_call_to_its_core(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
    make_exec_config: Callable[..., Any],
) -> None:
    """Process mode has no OutputManager to carry it; the strategy passes it."""
    from phenotypic._cli import _cli_process_only
    from phenotypic._cli._cli_execution_strategies import LocalParallelStrategy
    from phenotypic._cli._cli_output_manager import OutputManager
    from phenotypic._cli._cli_types import Dataset

    seen: dict[str, Any] = {}

    def _spy(**kwargs: Any) -> bool:
        seen["call"] = kwargs.get("run_initiation", "MISSING")
        raise RuntimeError("stop after the call")

    monkeypatch.setattr(_cli_process_only, "process_single_apply_only_core", _spy)
    out = tmp_path / "out"
    config = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=synth_one_level_input,
        output_dir=out,
        process_only_layer="rgb",
        run_initiation=CALL,
    )
    strategy = LocalParallelStrategy(config, OutputManager.from_config(out, ".tiff"))
    image = next(synth_one_level_input.rglob("*.tif"))
    out.mkdir()
    strategy._process_single_local_apply_only(
        Dataset("ds", [image], synth_one_level_input, out), image, out, out / "events.jsonl"
    )
    assert seen["call"] == CALL
