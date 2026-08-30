"""Two-generation races at real ordinary and staged publication boundaries."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image as PILImage

from phenotypic import Image, ImagePipeline
from phenotypic._cli import _cli_process_single as ordinary_worker
from phenotypic._cli import _cli_staged_slurm_worker as staged_worker
from phenotypic._cli._cli_failure_tracker import (
    PerImageScientificError,
    read_terminal_failures,
)
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_slurm_lifecycle import (
    SlurmGenerationInactiveError,
    deactivate_generation,
    initialize_slurm_lifecycle,
    lifecycle_lock_path,
    slurm_generation_inactive_cause,
)
from phenotypic._cli._cli_stage2_token import (
    load_stage2_raw,
    read_stage2_token,
    stage2_token_path,
    write_stage2_raw,
    write_stage2_token,
)
from phenotypic._cli._cli_staged_orchestration import (
    initialize_orchestration,
)
from phenotypic._cli._cli_types import Dataset
from phenotypic._cli._cli_update_state import SLURM_GENERATION_ENV_VAR
from phenotypic.enhance import BlurGauss
from phenotypic.detect import OtsuDetector
from phenotypic.sdk_ import (
    atomic_write_json,
    event_log_path,
    progress_dir,
    zarr_store_path,
)
from phenotypic.sdk_._file_locking import (
    ArtifactLockTimeout,
    exclusive_path_lock,
)


_STALE = "stale-generation"
_SUCCESSOR = "successor-generation"
_STALE_THREAD = "stale-publication"


def _ordinary_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    pixels = np.zeros((72, 60, 3), dtype=np.uint8)
    pixels[20:50, 20:40, :] = 200
    image_path = tmp_path / "plate.tiff"
    PILImage.fromarray(pixels).save(image_path)
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        ImagePipeline(
            ops={
                "prepare": BlurGauss(sigma=1.0),
                "detect": OtsuDetector(),
            }
        ).to_json()
        or "",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    manager = OutputManager.from_config(
        output_dir, ".tiff", save_overlays=False
    )
    manager.create_structure(
        [Dataset("ds", [image_path], tmp_path, output_dir)]
    )
    return image_path, pipeline_path, output_dir


def _ordinary_args(
    image_path: Path,
    pipeline_path: Path,
    output_dir: Path,
    *,
    event_log: Path | None = None,
) -> list[str]:
    args = [
        "--pipeline",
        str(pipeline_path),
        "--image",
        str(image_path),
        "--output-dir",
        str(output_dir),
        "--dataset-name",
        "ds",
        "--image-type",
        "Image",
        "--no-save-overlays",
    ]
    if event_log is not None:
        args.extend(["--event-log", str(event_log)])
    return args


def _use_harness_plan(
    monkeypatch: pytest.MonkeyPatch,
    run: Any,
) -> None:
    """Keep actual worker routing while bypassing test-only op discovery."""
    serialized = SimpleNamespace(from_json=lambda _path: run.plan)
    monkeypatch.setattr(staged_worker, "ImagePipeline", serialized)
    monkeypatch.setattr(staged_worker, "split_pipeline_at_gpu", lambda plan: plan)


def _pause_stale_replace(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    at_commit: threading.Event,
    release: threading.Event,
) -> None:
    original_replace = os.replace

    def _replace(source: Any, destination: Any) -> None:
        if (
            threading.current_thread().name == _STALE_THREAD
            and Path(destination) == target
        ):
            at_commit.set()
            if not release.wait(30):
                raise TimeoutError("test did not release stale replacement")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", _replace)


def _pause_stale_unlink(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    at_commit: threading.Event,
    release: threading.Event,
) -> None:
    original_unlink = Path.unlink

    def _unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if (
            threading.current_thread().name == _STALE_THREAD
            and path == target
        ):
            at_commit.set()
            if not release.wait(30):
                raise TimeoutError("test did not release stale deletion")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _unlink)


def _lifecycle_lock_is_held(output_dir: Path) -> bool:
    try:
        with exclusive_path_lock(lifecycle_lock_path(output_dir), timeout=0.1):
            return False
    except ArtifactLockTimeout:
        return True


def _run_two_generation_race(
    *,
    output_dir: Path,
    at_commit: threading.Event,
    release: threading.Event,
    stale_call: Callable[[], None],
    successor_call: Callable[[], None],
) -> tuple[bool, list[BaseException]]:
    stale_errors: list[BaseException] = []
    successor_errors: list[BaseException] = []

    def _run_stale() -> None:
        try:
            stale_call()
        except BaseException as exc:
            stale_errors.append(exc)

    stale_thread = threading.Thread(
        target=_run_stale,
        name=_STALE_THREAD,
        daemon=True,
    )
    stale_thread.start()
    if not at_commit.wait(30):
        release.set()
        stale_thread.join(30)
        pytest.fail(
            "stale worker never reached the canonical commit point: "
            f"{stale_errors!r}"
        )

    lock_was_held = _lifecycle_lock_is_held(output_dir)
    successor_done = threading.Event()

    def _run_successor() -> None:
        try:
            successor_call()
        except BaseException as exc:
            successor_errors.append(exc)
        finally:
            successor_done.set()

    successor_thread = threading.Thread(
        target=_run_successor,
        name="successor-publication",
        daemon=True,
    )
    successor_thread.start()
    if lock_was_held:
        assert not successor_done.wait(0.1), (
            "successor published while the stale atomic commit held the lock"
        )
    else:
        assert successor_done.wait(30), (
            "unguarded successor did not publish before stale resumed"
        )
    release.set()
    stale_thread.join(30)
    successor_thread.join(30)

    assert not stale_thread.is_alive()
    assert not successor_thread.is_alive()
    assert successor_errors == []
    unexpected_stale = [
        error
        for error in stale_errors
        if slurm_generation_inactive_cause(error) is None
    ]
    assert unexpected_stale == []
    return lock_was_held, stale_errors


def _supersede_ordinary(output_dir: Path) -> None:
    assert deactivate_generation(output_dir, _STALE) is True
    initialize_slurm_lifecycle(
        output_dir,
        generation=_SUCCESSOR,
        mode="ordinary",
    )


def _supersede_staged(output_dir: Path) -> None:
    assert deactivate_generation(output_dir, _STALE) is True
    initialize_orchestration(
        output_dir,
        epoch=_SUCCESSOR,
        mode="full",
        controller_config_path=output_dir / "successor-controller.json",
    )


def _stamp_successor_root(root: Path) -> None:
    payload = json.loads(root.read_text(encoding="utf-8"))
    payload.setdefault("attributes", {})["race_owner"] = _SUCCESSOR
    atomic_write_json(root, payload, sort_keys=False)


def test_ordinary_root_replace_holds_generation_lock_until_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing the guard lets stale root JSON replace successor metadata."""
    image_path, pipeline_path, output_dir = _ordinary_inputs(tmp_path)
    initialize_slurm_lifecycle(
        output_dir, generation=_STALE, mode="ordinary"
    )
    monkeypatch.setenv("SLURM_JOB_ID", "101")
    monkeypatch.setenv(SLURM_GENERATION_ENV_VAR, _STALE)
    root = zarr_store_path(output_dir, "ds", image_path.stem) / "zarr.json"
    at_commit = threading.Event()
    release = threading.Event()
    _pause_stale_replace(monkeypatch, root, at_commit, release)
    results: list[Any] = []

    def _stale_call() -> None:
        results.append(
            CliRunner().invoke(
                ordinary_worker.main,
                _ordinary_args(image_path, pipeline_path, output_dir),
            )
        )

    def _successor_call() -> None:
        _supersede_ordinary(output_dir)
        _stamp_successor_root(root)

    held, _ = _run_two_generation_race(
        output_dir=output_dir,
        at_commit=at_commit,
        release=release,
        stale_call=_stale_call,
        successor_call=_successor_call,
    )

    assert json.loads(root.read_text(encoding="utf-8"))["attributes"][
        "race_owner"
    ] == _SUCCESSOR
    assert held is True
    assert results and results[0].exit_code == 1


def test_ordinary_store_build_is_unlocked_but_promotion_is_locked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pyramid construction stays concurrent; only promotion holds the fence."""
    image_path, pipeline_path, output_dir = _ordinary_inputs(tmp_path)
    initialize_slurm_lifecycle(
        output_dir, generation=_STALE, mode="ordinary"
    )
    monkeypatch.setenv("SLURM_JOB_ID", "101")
    monkeypatch.setenv(SLURM_GENERATION_ENV_VAR, _STALE)
    store = zarr_store_path(output_dir, "ds", image_path.stem)

    preparing = threading.Event()
    release_preparation = threading.Event()
    at_promotion = threading.Event()
    release_promotion = threading.Event()
    original_write_series = Image._write_series
    original_replace = os.replace

    def _write_series(
        subject: Image, *args: Any, **kwargs: Any
    ) -> None:
        if (
            threading.current_thread().name == _STALE_THREAD
            and not preparing.is_set()
        ):
            preparing.set()
            if not release_preparation.wait(30):
                raise TimeoutError("test did not release store preparation")
        original_write_series(subject, *args, **kwargs)

    def _replace(source: Any, destination: Any) -> None:
        if (
            threading.current_thread().name == _STALE_THREAD
            and Path(destination) == store
            and not at_promotion.is_set()
        ):
            at_promotion.set()
            if not release_promotion.wait(30):
                raise TimeoutError("test did not release store promotion")
        original_replace(source, destination)

    monkeypatch.setattr(Image, "_write_series", _write_series)
    monkeypatch.setattr(os, "replace", _replace)
    results: list[Any] = []

    def _worker_call() -> None:
        results.append(
            CliRunner().invoke(
                ordinary_worker.main,
                _ordinary_args(image_path, pipeline_path, output_dir),
            )
        )

    worker = threading.Thread(
        target=_worker_call,
        name=_STALE_THREAD,
        daemon=True,
    )
    worker.start()
    try:
        assert preparing.wait(30), "worker did not begin pyramid construction"
        assert _lifecycle_lock_is_held(output_dir) is False
        release_preparation.set()
        assert at_promotion.wait(30), "worker did not reach store promotion"
        assert _lifecycle_lock_is_held(output_dir) is True
    finally:
        release_preparation.set()
        release_promotion.set()
        worker.join(60)

    assert not worker.is_alive()
    assert results and results[0].exit_code == 0, results[0].output


def test_stage1_root_replace_holds_generation_lock_until_commit(
    staged_run_with_provenance: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage1's operation checkpoint cannot replace a successor root."""
    run = staged_run_with_provenance
    _use_harness_plan(monkeypatch, run)
    initialize_orchestration(
        run.output_dir,
        epoch=_STALE,
        mode="full",
        controller_config_path=run.output_dir / "stale-controller.json",
    )
    root = run.store() / "zarr.json"
    at_commit = threading.Event()
    release = threading.Event()
    _pause_stale_replace(monkeypatch, root, at_commit, release)

    def _stale_call() -> None:
        staged_worker.run_stage1_step(
            pipeline_path=run.pipeline_path,
            output_dir=run.output_dir,
            image_type="Image",
            manifest=[("ds", "img", str(run.image_path))],
            index=0,
            epoch=_STALE,
        )

    def _successor_call() -> None:
        _supersede_staged(run.output_dir)
        _stamp_successor_root(root)

    held, _ = _run_two_generation_race(
        output_dir=run.output_dir,
        at_commit=at_commit,
        release=release,
        stale_call=_stale_call,
        successor_call=_successor_call,
    )

    assert json.loads(root.read_text(encoding="utf-8"))["attributes"][
        "race_owner"
    ] == _SUCCESSOR
    assert held is True


def test_stage2_token_replace_holds_generation_lock_until_commit(
    staged_run: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage2 cannot replace a successor token after losing ownership."""
    run = staged_run
    run.run_stage1()
    _use_harness_plan(monkeypatch, run)
    initialize_orchestration(
        run.output_dir,
        epoch=_STALE,
        mode="full",
        controller_config_path=run.output_dir / "stale-controller.json",
    )
    token = stage2_token_path(run.output_dir, "ds", "img")
    at_commit = threading.Event()
    release = threading.Event()
    _pause_stale_replace(monkeypatch, token, at_commit, release)

    def _stale_call() -> None:
        staged_worker.run_stage2_shard(
            pipeline_path=run.pipeline_path,
            output_dir=run.output_dir,
            image_type="Image",
            manifest=[("ds", "img", str(run.image_path))],
            shard_index=0,
            n_shards=1,
            epoch=_STALE,
        )

    def _successor_call() -> None:
        _supersede_staged(run.output_dir)
        write_stage2_raw(
            run.output_dir,
            "ds",
            "img",
            np.full((600, 800), 7, dtype=np.uint16),
        )
        write_stage2_token(
            run.output_dir,
            "ds",
            "img",
            objmap_shape=(600, 800),
            detector_duration_seconds=777.0,
        )

    held, _ = _run_two_generation_race(
        output_dir=run.output_dir,
        at_commit=at_commit,
        release=release,
        stale_call=_stale_call,
        successor_call=_successor_call,
    )

    assert read_stage2_token(run.output_dir, "ds", "img")[
        "detector_duration_seconds"
    ] == 777.0
    assert np.all(load_stage2_raw(run.output_dir, "ds", "img") == 7)
    assert held is True


def test_stage3_token_delete_holds_generation_lock_until_commit(
    staged_run: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage3 cannot delete the successor's newly published Stage2 token."""
    run = staged_run
    run.run_stage1()
    run.run_stage2()
    _use_harness_plan(monkeypatch, run)
    stage3_output = OutputManager.from_config(
        run.output_dir, ".tiff", save_overlays=True
    )
    stage3_output.get_output_path(
        "ds", "overlays", "img"
    ).parent.mkdir(parents=True, exist_ok=True)
    initialize_orchestration(
        run.output_dir,
        epoch=_STALE,
        mode="full",
        controller_config_path=run.output_dir / "stale-controller.json",
    )
    token = stage2_token_path(run.output_dir, "ds", "img")
    at_commit = threading.Event()
    release = threading.Event()
    _pause_stale_unlink(monkeypatch, token, at_commit, release)

    def _stale_call() -> None:
        staged_worker.run_stage3_step(
            pipeline_path=run.pipeline_path,
            output_dir=run.output_dir,
            image_type="Image",
            manifest=[("ds", "img", str(run.image_path))],
            index=0,
            epoch=_STALE,
        )

    def _successor_call() -> None:
        _supersede_staged(run.output_dir)
        write_stage2_raw(
            run.output_dir,
            "ds",
            "img",
            np.full((600, 800), 9, dtype=np.uint16),
        )
        write_stage2_token(
            run.output_dir,
            "ds",
            "img",
            objmap_shape=(600, 800),
            detector_duration_seconds=999.0,
        )

    held, _ = _run_two_generation_race(
        output_dir=run.output_dir,
        at_commit=at_commit,
        release=release,
        stale_call=_stale_call,
        successor_call=_successor_call,
    )

    assert read_stage2_token(run.output_dir, "ds", "img")[
        "detector_duration_seconds"
    ] == 999.0
    assert np.all(load_stage2_raw(run.output_dir, "ds", "img") == 9)
    assert held is True


def test_ordinary_cli_stale_exit_writes_no_failure_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chained inactive cause must bypass every ordinary failure sink."""
    image_path, pipeline_path, output_dir = _ordinary_inputs(tmp_path)
    initialize_slurm_lifecycle(
        output_dir, generation=_STALE, mode="ordinary"
    )
    assert deactivate_generation(output_dir, _STALE) is True
    monkeypatch.setenv("SLURM_JOB_ID", "101")
    monkeypatch.setenv(SLURM_GENERATION_ENV_VAR, _STALE)

    def _raise_stale(**_kwargs: Any) -> bool:
        inactive = SlurmGenerationInactiveError("stale during publication")
        try:
            raise inactive
        except SlurmGenerationInactiveError as cause:
            raise PerImageScientificError("full", cause) from cause

    monkeypatch.setattr(
        ordinary_worker,
        "process_single_image_core",
        _raise_stale,
    )
    event_log = tmp_path / "events.jsonl"
    result = CliRunner().invoke(
        ordinary_worker.main,
        _ordinary_args(
            image_path,
            pipeline_path,
            output_dir,
            event_log=event_log,
        ),
    )

    assert result.exit_code == 1
    statuses = [
        line.split("|")[3]
        for line in event_log.read_text(encoding="utf-8").splitlines()
    ]
    assert statuses == ["started"]
    assert not (progress_dir(output_dir) / "failures.jsonl").exists()
    assert read_terminal_failures(output_dir) == []


def test_stage2_shard_reraises_inactive_without_failure_outcome(
    staged_run: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale single-item shard is infrastructure failure, not science."""
    run = staged_run
    run.run_stage1()
    _use_harness_plan(monkeypatch, run)
    initialize_orchestration(
        run.output_dir,
        epoch=_STALE,
        mode="full",
        controller_config_path=run.output_dir / "stale-controller.json",
    )

    def _raise_inactive(*_args: Any, **_kwargs: Any) -> None:
        raise SlurmGenerationInactiveError("stage2 ownership revoked")

    monkeypatch.setattr(
        staged_worker,
        "stage2_detect_core",
        _raise_inactive,
    )

    with pytest.raises(
        SlurmGenerationInactiveError, match="stage2 ownership revoked"
    ):
        staged_worker.run_stage2_shard(
            pipeline_path=run.pipeline_path,
            output_dir=run.output_dir,
            image_type="Image",
            manifest=[("ds", "img", str(run.image_path))],
            shard_index=0,
            n_shards=1,
            epoch=_STALE,
        )

    assert read_terminal_failures(run.output_dir) == []
    events = event_log_path(run.output_dir)
    if events.exists():
        assert '"status": "failed"' not in events.read_text(encoding="utf-8")
