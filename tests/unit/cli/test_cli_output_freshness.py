"""Tests for the CLI's exact reserved GUI-log freshness exception."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from phenotypic.phenotypicCLI import (
    _OUTPUT_ROOT_UNREADABLE,
    _has_prior_scientific_artifacts,
    _prior_scientific_artifact_names,
    is_safe_gui_launch_state_entry,
    is_safe_gui_log_entry,
)
from phenotypic.sdk_ import (
    DIR_PHENOTYPIC,
    DIR_PROGRESS,
    GUI_LAUNCH_OWNER_JSON,
    PROCESSING_EVENTS_LOG,
    PROCESSING_STATE_JSON,
    RUN_LOG_DIRNAME,
    STDOUT_LOG,
    STORE_SUFFIX,
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


def _build_empty(root: Path) -> None:
    root.mkdir()


def _build_missing(root: Path) -> None:
    """Leave ``root`` absent: a not-yet-created output dir is fresh."""


def _build_machine_only(root: Path) -> None:
    (root / DIR_PHENOTYPIC / DIR_PROGRESS).mkdir(parents=True)
    (root / DIR_PHENOTYPIC / DIR_PROGRESS / "state.json").write_bytes(
        b"machine state"
    )


def _build_legacy_machine_only(root: Path) -> None:
    """Pre-migration root-level state -- exactly what restart deletes."""
    (root / DIR_PROGRESS).mkdir(parents=True)
    (root / DIR_PROGRESS / "img001.json").write_bytes(b"progress")
    (root / PROCESSING_STATE_JSON).write_bytes(b"{}")
    (root / PROCESSING_EVENTS_LOG).write_bytes(b"event\n")


def _build_launch_only(root: Path) -> None:
    _write_launch_owner(root)


def _build_gui_log_only(root: Path) -> None:
    gui_log = root / RUN_LOG_DIRNAME
    gui_log.mkdir(parents=True)
    (gui_log / STDOUT_LOG).write_bytes(b"log")


def _build_fake_machine_state(root: Path) -> None:
    root.mkdir()
    (root / DIR_PHENOTYPIC).write_bytes(b"not a directory")


def _build_machine_state_holding_a_store(root: Path) -> None:
    """A dataset named ``.phenotypic`` mirrors its exports into the cache.

    ``scan_directory_structure`` does not skip dot-directories, so this is
    reachable -- and calling it machine-state would feed published stores to
    ``clear_machine_state``'s ``rmtree``.
    """
    store = root / DIR_PHENOTYPIC / f"img001{STORE_SUFFIX}"
    store.mkdir(parents=True)
    (store / "zarr.json").write_bytes(b"scientific")


def _build_process_store(root: Path) -> None:
    (root / f"plate{STORE_SUFFIX}").mkdir(parents=True)


def _build_process_file(root: Path) -> None:
    root.mkdir()
    (root / "plate.tiff").write_bytes(b"pixels")


def _build_output_is_a_file(root: Path) -> None:
    root.write_bytes(b"not a directory")


def _build_scaffolding_only(root: Path) -> None:
    """Exactly what every run creates before writing a single image."""
    (root / "results" / "plate1" / "zarr").mkdir(parents=True)
    (root / "results" / "plate2" / "zarr").mkdir(parents=True)
    (root / "deliverables" / "overlays" / "plate1").mkdir(parents=True)


def _build_scaffolding_with_one_result(root: Path) -> None:
    store = root / "results" / "plate1" / "zarr" / f"img001{STORE_SUFFIX}"
    store.mkdir(parents=True)
    (store / "zarr.json").write_bytes(b"scientific")


def _build_nested_empty_store(root: Path) -> None:
    """An empty store directory is still a published store identity."""
    (root / "results" / "plate1" / "zarr" / f"img001{STORE_SUFFIX}").mkdir(
        parents=True
    )


@pytest.mark.parametrize(
    ("build", "expected"),
    (
        pytest.param(_build_empty, False, id="empty-dir"),
        pytest.param(_build_missing, False, id="missing-dir"),
        pytest.param(_build_machine_only, False, id="machine-state-only"),
        pytest.param(
            _build_legacy_machine_only, False, id="legacy-machine-state"
        ),
        pytest.param(_build_launch_only, False, id="gui-launch-state"),
        pytest.param(_build_gui_log_only, False, id="gui-log-only"),
        pytest.param(
            _build_fake_machine_state, True, id="phenotypic-is-a-file"
        ),
        pytest.param(
            _build_machine_state_holding_a_store,
            True,
            id="phenotypic-holds-a-store",
        ),
        pytest.param(_build_process_store, True, id="process-store-at-root"),
        pytest.param(_build_process_file, True, id="process-file-at-root"),
        pytest.param(_build_output_is_a_file, True, id="output-is-a-file"),
        pytest.param(_build_scaffolding_only, False, id="empty-scaffolding"),
        pytest.param(
            _build_scaffolding_with_one_result, True, id="one-real-result"
        ),
        pytest.param(
            _build_nested_empty_store, True, id="nested-empty-store"
        ),
    ),
)
def test_restart_artifact_scan_ignores_only_machine_and_safe_gui_entries(
    tmp_path: Path,
    build: Callable[[Path], None],
    expected: bool,
) -> None:
    """Top-level process outputs are science; machine-only roots are fresh."""
    root = tmp_path / "out"
    build(root)

    assert _has_prior_scientific_artifacts(root) is expected


def test_symlinked_machine_state_is_prior_science(tmp_path: Path) -> None:
    """A ``.phenotypic`` symlink cannot launder an arbitrary directory.

    The target holds only plausible machine-state, so nothing *except* the
    symlink check can refuse it -- and accepting it would hand a directory
    outside the output tree to ``clear_machine_state``'s ``rmtree``.
    """
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / DIR_PROGRESS).mkdir(parents=True)
    (elsewhere / DIR_PROGRESS / "state.json").write_bytes(b"machine state")
    root = tmp_path / "out"
    root.mkdir()
    try:
        (root / DIR_PHENOTYPIC).symlink_to(elsewhere, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable")

    assert _has_prior_scientific_artifacts(root)


def test_an_unreadable_output_root_is_reported_as_prior_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed: an unreadable root is never certified as fresh.

    Driven through ``iterdir`` rather than ``chmod`` so the clause stays
    pinned when the suite runs as root, which bypasses directory permissions.
    """
    root = tmp_path / "out"
    root.mkdir()
    real_iterdir = Path.iterdir

    def refuse(self: Path):
        if self == root:
            raise PermissionError(13, "Permission denied")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", refuse)

    assert _prior_scientific_artifact_names(root) == [_OUTPUT_ROOT_UNREADABLE]
    assert _has_prior_scientific_artifacts(root)


def test_blocking_entries_are_named_for_the_error_message(
    tmp_path: Path,
) -> None:
    """The refusal can say what to remove instead of only that it refused."""
    root = tmp_path / "out"
    (root / DIR_PHENOTYPIC).mkdir(parents=True)
    (root / "notes.txt").write_bytes(b"stray")
    (root / f"plate{STORE_SUFFIX}").mkdir()

    assert _prior_scientific_artifact_names(root) == [
        "notes.txt",
        f"plate{STORE_SUFFIX}",
    ]


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


def test_a_root_symlink_is_prior_science(tmp_path: Path) -> None:
    """A link's target is outside the tree and cannot be vouched for."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    root = tmp_path / "out"
    root.mkdir()
    try:
        (root / "results").symlink_to(elsewhere, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable")

    assert _has_prior_scientific_artifacts(root)


def test_the_runs_own_pipeline_copy_is_not_prior_science(
    tmp_path: Path,
) -> None:
    """``_copy_pipeline_to_output`` writes the source basename at the root.

    Without this, the guard fires on every run that ever started, because the
    copy is made before any image is processed.
    """
    root = tmp_path / "out"
    root.mkdir()
    (root / "my_pipeline.json").write_bytes(b"{}")

    assert _prior_scientific_artifact_names(root) == ["my_pipeline.json"]
    assert (
        _prior_scientific_artifact_names(
            root, pipeline_snapshot_name="my_pipeline.json"
        )
        == []
    )
    # Only the file is exempt -- a directory of that name is not the copy.
    other = tmp_path / "other"
    (other / "my_pipeline.json" / f"p{STORE_SUFFIX}").mkdir(parents=True)
    assert _prior_scientific_artifact_names(
        other, pipeline_snapshot_name="my_pipeline.json"
    ) == ["my_pipeline.json"]
