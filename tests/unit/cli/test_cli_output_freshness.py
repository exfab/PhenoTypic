"""Tests for the CLI's exact reserved GUI-log freshness exception."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from phenotypic.phenotypicCLI import (
    _has_prior_scientific_artifacts,
    is_safe_gui_launch_state_entry,
    is_safe_gui_log_entry,
)
from phenotypic.sdk_ import (
    GUI_LAUNCH_OWNER_JSON,
    RUN_LOG_DIRNAME,
    STDOUT_LOG,
    atomic_write_json,
    progress_dir,
)


def test_stdout_only_gui_log_directory_is_safe(tmp_path: Path) -> None:
    log_dir = tmp_path / RUN_LOG_DIRNAME
    log_dir.mkdir()
    (log_dir / STDOUT_LOG).write_text("", encoding="utf-8")

    assert is_safe_gui_log_entry(log_dir)


def test_empty_real_gui_log_directory_is_safe(tmp_path: Path) -> None:
    log_dir = tmp_path / RUN_LOG_DIRNAME
    log_dir.mkdir()

    assert is_safe_gui_log_entry(log_dir)


def test_restart_artifact_scan_ignores_only_machine_and_safe_gui_entries(
    tmp_path: Path,
) -> None:
    """Top-level process outputs are science; machine-only roots are fresh."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not _has_prior_scientific_artifacts(empty)

    machine_only = tmp_path / "machine-only"
    (machine_only / ".phenotypic" / "progress").mkdir(parents=True)
    (machine_only / ".phenotypic" / "progress" / "state.json").write_bytes(
        b"machine state"
    )
    assert not _has_prior_scientific_artifacts(machine_only)

    launch_only = tmp_path / "launch-only"
    _write_launch_owner(launch_only)
    assert not _has_prior_scientific_artifacts(launch_only)

    fake_machine_state = tmp_path / "fake-machine-state"
    fake_machine_state.mkdir()
    (fake_machine_state / ".phenotypic").write_bytes(b"not a directory")
    assert _has_prior_scientific_artifacts(fake_machine_state)

    gui_only = tmp_path / "gui-only"
    gui_log = gui_only / RUN_LOG_DIRNAME
    gui_log.mkdir(parents=True)
    (gui_log / STDOUT_LOG).write_bytes(b"log")
    assert not _has_prior_scientific_artifacts(gui_only)

    process_store = tmp_path / "process-store"
    (process_store / "plate.ome.zarr").mkdir(parents=True)
    assert _has_prior_scientific_artifacts(process_store)

    process_file = tmp_path / "process-file"
    process_file.mkdir()
    (process_file / "plate.tiff").write_bytes(b"pixels")
    assert _has_prior_scientific_artifacts(process_file)


@pytest.mark.parametrize("child_name", ["stderr.log", "notes.txt"])
def test_unrecognized_gui_log_file_is_not_safe(
    tmp_path: Path, child_name: str
) -> None:
    log_dir = tmp_path / RUN_LOG_DIRNAME
    log_dir.mkdir()
    (log_dir / child_name).write_text("", encoding="utf-8")

    assert not is_safe_gui_log_entry(log_dir)


def test_nested_gui_log_directory_is_not_safe(tmp_path: Path) -> None:
    log_dir = tmp_path / RUN_LOG_DIRNAME
    (log_dir / "nested").mkdir(parents=True)

    assert not is_safe_gui_log_entry(log_dir)


def test_wrong_directory_name_is_not_safe(tmp_path: Path) -> None:
    log_dir = tmp_path / ".other-log"
    log_dir.mkdir()
    (log_dir / STDOUT_LOG).write_text("", encoding="utf-8")

    assert not is_safe_gui_log_entry(log_dir)


def test_symlinked_gui_log_directory_is_not_safe(tmp_path: Path) -> None:
    target = tmp_path / "real-log"
    target.mkdir()
    (target / STDOUT_LOG).write_text("", encoding="utf-8")
    link = tmp_path / RUN_LOG_DIRNAME
    try:
        link.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable")

    assert not is_safe_gui_log_entry(link)


def test_symlinked_allowed_log_file_is_not_safe(tmp_path: Path) -> None:
    target = tmp_path / "real.log"
    target.write_text("", encoding="utf-8")
    log_dir = tmp_path / RUN_LOG_DIRNAME
    log_dir.mkdir()
    link = log_dir / STDOUT_LOG
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("file symlinks are unavailable")

    assert not is_safe_gui_log_entry(log_dir)


def _write_launch_owner(output_dir: Path) -> Path:
    owner = progress_dir(output_dir) / GUI_LAUNCH_OWNER_JSON
    atomic_write_json(
        owner,
        {
            "version": 1,
            "run_id": "fresh",
            "generation": str(uuid4()),
            "mode": "local",
            "output_dir": str(output_dir),
            "rel_path": "fresh",
            "status": "running",
            "command_digest": "sha256:test",
        },
    )
    return owner


def test_exact_prelaunch_owner_state_is_safe(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    owner = _write_launch_owner(output_dir)
    owner.with_suffix(".lock").touch()

    assert is_safe_gui_launch_state_entry(output_dir / ".phenotypic")


def test_generation_scoped_submitter_logs_are_safe(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    owner = _write_launch_owner(output_dir)
    generation = json.loads(owner.read_text(encoding="utf-8"))["generation"]
    token = UUID(generation).hex
    log_dir = output_dir / ".phenotypic" / "logs" / "gui"
    log_dir.mkdir(parents=True)
    for stream in ("stdout", "stderr"):
        (log_dir / f"submitter.{token}.{stream}.log").touch()

    assert is_safe_gui_launch_state_entry(output_dir / ".phenotypic")


@pytest.mark.parametrize(
    "filename",
    [
        "notes.txt",
        "submitter.wrong.stdout.log",
    ],
)
def test_unrecognized_submitter_log_is_not_safe(
    tmp_path: Path,
    filename: str,
) -> None:
    output_dir = tmp_path / "output"
    _write_launch_owner(output_dir)
    log_dir = output_dir / ".phenotypic" / "logs" / "gui"
    log_dir.mkdir(parents=True)
    (log_dir / filename).touch()

    assert not is_safe_gui_launch_state_entry(output_dir / ".phenotypic")


@pytest.mark.parametrize(
    ("relative_path", "contents"),
    [
        ("progress/job_metadata.json", "{}"),
        ("processing_state.json", "{}"),
        ("progress/run_completion.json", "{}"),
    ],
)
def test_other_machine_state_is_not_fresh(
    tmp_path: Path,
    relative_path: str,
    contents: str,
) -> None:
    output_dir = tmp_path / "output"
    _write_launch_owner(output_dir)
    extra = output_dir / ".phenotypic" / relative_path
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text(contents, encoding="utf-8")

    assert not is_safe_gui_launch_state_entry(output_dir / ".phenotypic")


def test_owner_for_different_output_is_not_safe(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    owner = _write_launch_owner(output_dir)
    payload = json.loads(owner.read_text(encoding="utf-8"))
    payload["output_dir"] = str(tmp_path / "other")
    atomic_write_json(owner, payload)

    assert not is_safe_gui_launch_state_entry(output_dir / ".phenotypic")


def test_symlinked_machine_state_is_not_safe(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    real_output = tmp_path / "real-output"
    _write_launch_owner(real_output)
    output_dir.mkdir()
    link = output_dir / ".phenotypic"
    try:
        link.symlink_to(real_output / ".phenotypic", target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable")

    assert not is_safe_gui_launch_state_entry(link)
