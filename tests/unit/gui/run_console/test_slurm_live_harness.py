"""Safety tests for the opt-in live SLURM acceptance harness."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from phenotypic import Image, ImagePipeline
from phenotypic._cli._cli_slurm_lifecycle import (
    append_lifecycle_entry,
    initialize_slurm_lifecycle,
    lifecycle_state_path,
)
from phenotypic.gui.run_console._slurm_observer import SchedulerQueryResult
from phenotypic.sdk_ import (
    atomic_write_json,
    job_metadata_path,
    processing_state_path,
)
from tests._support import live_slurm as live

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason=(
        "live SLURM cleanup safety requires POSIX directory handles, "
        "inode identity, dir_fd, and O_NOFOLLOW"
    ),
)


def _case(root: Path) -> tuple[Path, Path]:
    case_generation = uuid4().hex
    case_root = root / f"{live.CASE_PREFIX}{case_generation}"
    output_dir = case_root / f"output-{case_generation}"
    output_dir.mkdir(parents=True)
    return case_root, output_dir


def _completed_scancel(job_id: str, *, returncode: int = 0):
    return subprocess.CompletedProcess(
        ["scancel", job_id],
        returncode,
        "",
        "already terminal" if returncode else "",
    )


def _comment_matches(generation: UUID, job_id: str, token: str) -> dict:
    return {f"phenotypic:{generation.hex}:{token}": {job_id}}


def _configure_terminal_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    generation: UUID,
    job_id: str,
    *,
    token: str,
    scancel_returncode: int = 0,
) -> list[list[str]]:
    """Return recorded subprocess commands for one terminal scheduler job."""
    commands: list[list[str]] = []
    monkeypatch.setattr(
        live,
        "query_scheduler_comments",
        lambda **_kwargs: _comment_matches(generation, job_id, token),
    )
    monkeypatch.setattr(
        live,
        "active_generation_comment_ids",
        lambda _generation: set(),
    )
    monkeypatch.setattr(
        live,
        "query_known_job_states",
        lambda _job_ids: SchedulerQueryResult(
            states={job_id: "CANCELLED"},
            available=True,
        ),
    )

    def record_command(command, **_kwargs):
        commands.append(list(command))
        return _completed_scancel(
            job_id,
            returncode=scancel_returncode,
        )

    monkeypatch.setattr(live.subprocess, "run", record_command)
    monkeypatch.setattr(live, "POLL_SECONDS", 0.0)
    monkeypatch.setattr(live, "RECONCILIATION_GRACE_SECONDS", 0.0)
    return commands


def _forbid_scheduler_access(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Fail if an unsafe path reaches a scheduler query or cancellation."""
    calls: list[str] = []

    def forbidden_call(*_args, **_kwargs):
        calls.append("scheduler")
        raise AssertionError("scheduler access attempted")

    monkeypatch.setattr(live, "query_scheduler_comments", forbidden_call)
    monkeypatch.setattr(
        live,
        "active_generation_comment_ids",
        forbidden_call,
    )
    monkeypatch.setattr(live, "query_known_job_states", forbidden_call)
    monkeypatch.setattr(live.subprocess, "run", forbidden_call)
    return calls


def _forbid_following_path(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Instrument following stat/resolve calls against one symlink pathname."""
    calls: list[str] = []
    real_stat = live.os.stat
    real_resolve = Path.resolve

    def guarded_stat(raw_path, *args, **kwargs):
        if (
            kwargs.get("dir_fd") is None
            and kwargs.get("follow_symlinks", True)
            and Path(raw_path) == path
        ):
            calls.append("stat")
            raise AssertionError("symlink target stat attempted")
        return real_stat(raw_path, *args, **kwargs)

    def guarded_resolve(candidate: Path, *args, **kwargs):
        if candidate == path:
            calls.append("resolve")
            raise AssertionError("symlink target resolve attempted")
        return real_resolve(candidate, *args, **kwargs)

    monkeypatch.setattr(live.os, "stat", guarded_stat)
    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    return calls


def _cleanup_evidence(case_root: Path) -> tuple[Path, ...]:
    return tuple(case_root.glob(".live-slurm-cleanup-*.json"))


def _create_retained_evidence(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    job_id: str = "799",
) -> tuple[Path, Path, UUID, str]:
    """Create one valid retained case using only the fake terminal scheduler."""
    case_root, output_dir = _case(root)
    generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir,
        generation=generation.hex,
        mode="ordinary",
    )
    append_lifecycle_entry(
        output_dir,
        generation=generation.hex,
        token="finalizer",
        role="finalizer",
        status="submitted",
        job_id=job_id,
    )
    _configure_terminal_scheduler(
        monkeypatch,
        generation,
        job_id,
        token="finalizer",
    )
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(root))
    evidence_name = live.cleanup_case(
        case_root,
        output_dir,
        generation,
        iter((job_id,)),
        forbidden=(),
    )
    return case_root, output_dir, generation, evidence_name


def test_cleanup_preflight_rejects_case_symlink_without_following(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    case_root, output_dir = _case(root)
    real_case = tmp_path / "relocated-case"
    case_root.rename(real_case)
    case_root.symlink_to(real_case, target_is_directory=True)
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(root))
    scheduler_calls = _forbid_scheduler_access(monkeypatch)
    follow_calls = _forbid_following_path(case_root, monkeypatch)

    with pytest.raises(AssertionError, match="unsafe live cleanup case"):
        live.cleanup_case(
            case_root,
            output_dir,
            uuid4(),
            iter(("700",)),
            forbidden=(),
        )

    assert scheduler_calls == []
    assert follow_calls == []
    assert case_root.is_symlink()


def test_cleanup_preflight_rejects_root_symlink_without_following(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_root = tmp_path / "real-root"
    case_root, output_dir = _case(real_root)
    root = tmp_path / "root-link"
    root.symlink_to(real_root, target_is_directory=True)
    linked_case = root / case_root.name
    linked_output = linked_case / output_dir.name
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(root))
    scheduler_calls = _forbid_scheduler_access(monkeypatch)
    follow_calls = _forbid_following_path(root, monkeypatch)

    with pytest.raises(AssertionError, match="unsafe live cleanup root"):
        live.cleanup_case(
            linked_case,
            linked_output,
            uuid4(),
            iter(("700",)),
            forbidden=(),
        )

    assert scheduler_calls == []
    assert follow_calls == []


def test_cleanup_preflight_rejects_output_symlink_before_scheduler_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    case_root, output_dir = _case(root)
    relocated_output = tmp_path / "relocated-output"
    output_dir.rename(relocated_output)
    output_dir.symlink_to(relocated_output, target_is_directory=True)
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(root))
    calls = _forbid_scheduler_access(monkeypatch)

    with pytest.raises(AssertionError, match="unsafe live cleanup output"):
        live.cleanup_case(
            case_root,
            output_dir,
            uuid4(),
            iter(("700",)),
            forbidden=(),
        )

    assert calls == []
    assert output_dir.is_symlink()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("protected", "overlaps protected path"),
        ("malformed", "malformed identity"),
        ("nested", "unsafe live cleanup case"),
        ("wrong-output", "unsafe live cleanup output"),
    ],
)
def test_cleanup_preflight_rejects_wrong_identity_before_scheduler_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    root = tmp_path / "root"
    case_root, output_dir = _case(root)
    forbidden: tuple[Path, ...] = ()
    if mutation == "protected":
        forbidden = (output_dir,)
    elif mutation == "malformed":
        malformed = case_root.with_name(f"{case_root.name}-extra")
        case_root.rename(malformed)
        output_dir = malformed / output_dir.name
        case_root = malformed
    elif mutation == "nested":
        nested = root / "nested"
        nested.mkdir()
        moved_case = nested / case_root.name
        case_root.rename(moved_case)
        case_root = moved_case
        output_dir = case_root / output_dir.name
    else:
        output_dir = case_root / "output-wrong"
        output_dir.mkdir()
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(root))
    calls = _forbid_scheduler_access(monkeypatch)

    with pytest.raises(AssertionError, match=message):
        live.cleanup_case(
            case_root,
            output_dir,
            uuid4(),
            iter(("700",)),
            forbidden=forbidden,
        )

    assert calls == []
    assert case_root.is_dir()


@pytest.mark.parametrize(
    ("symlink_level", "message"),
    [
        ("cache", "unsafe scheduler-state directory"),
        ("progress", "unsafe scheduler-state directory"),
        ("lifecycle", "unsafe scheduler-state file"),
    ],
)
def test_nested_scheduler_symlink_is_never_read_or_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_level: str,
    message: str,
) -> None:
    """Nested scheduler symlinks fail before file content or scheduler access."""
    root = tmp_path / "root"
    case_root, output_dir = _case(root)
    target = tmp_path / "protected-machine-state"
    target.mkdir()
    (target / "sentinel.txt").write_text("do not read", encoding="utf-8")
    cache = output_dir / ".phenotypic"
    if symlink_level == "cache":
        cache.symlink_to(target, target_is_directory=True)
    elif symlink_level == "progress":
        cache.mkdir()
        (cache / "progress").symlink_to(
            target,
            target_is_directory=True,
        )
    else:
        progress = cache / "progress"
        progress.mkdir(parents=True)
        (progress / "slurm_lifecycle.json").symlink_to(
            target / "sentinel.txt"
        )
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(root))
    scheduler_calls = _forbid_scheduler_access(monkeypatch)
    file_reads: list[str] = []

    def forbidden_read(*_args, **_kwargs):
        file_reads.append("read")
        raise AssertionError("nested scheduler file read attempted")

    monkeypatch.setattr(live.os, "read", forbidden_read)

    with pytest.raises(AssertionError, match=message):
        live.cleanup_case(
            case_root,
            output_dir,
            uuid4(),
            iter(("700",)),
            forbidden=(),
        )

    assert scheduler_calls == []
    assert file_reads == []
    assert (target / "sentinel.txt").read_text(encoding="utf-8") == "do not read"


def test_terminal_cleanup_retains_exact_case_without_path_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup preserves the exact case and writes fd-bound recovery evidence."""
    root = tmp_path / "root"
    case_root, output_dir = _case(root)
    generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir,
        generation=generation.hex,
        mode="ordinary",
    )
    append_lifecycle_entry(
        output_dir,
        generation=generation.hex,
        token="finalizer",
        role="finalizer",
        status="submitted",
        job_id="702",
    )
    protected = tmp_path / "protected"
    protected.mkdir()
    protected_file = protected / "keep.txt"
    protected_file.write_text("keep", encoding="utf-8")
    (case_root / "protected-link").symlink_to(
        protected,
        target_is_directory=True,
    )
    commands = _configure_terminal_scheduler(
        monkeypatch,
        generation,
        "702",
        token="finalizer",
        scancel_returncode=1,
    )
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(root))
    path_mutations: list[str] = []

    def forbid_path_mutation(*_args, **_kwargs):
        path_mutations.append("mutation")
        raise AssertionError("pathname mutation attempted")

    monkeypatch.setattr(live.os, "rename", forbid_path_mutation)
    monkeypatch.setattr(live.os, "unlink", forbid_path_mutation)
    monkeypatch.setattr(live.os, "rmdir", forbid_path_mutation)

    live.cleanup_case(
        case_root,
        output_dir,
        generation,
        iter(("702",)),
        forbidden=(protected,),
    )

    assert case_root.is_dir()
    assert output_dir.is_dir()
    assert (case_root / "protected-link").is_symlink()
    assert protected_file.read_text(encoding="utf-8") == "keep"
    evidence = _cleanup_evidence(case_root)
    assert len(evidence) == 1
    payload = json.loads(evidence[0].read_text(encoding="utf-8"))
    assert payload["status"] == "retained-after-scheduler-cleanup"
    assert payload["case_path"] == str(case_root)
    assert payload["scheduler_generation"] == generation.hex
    assert payload["scheduler_job_ids"] == ["702"]
    assert payload["forbidden_paths"][0]["current_status"] == "unchanged"
    assert commands == [["scancel", "702"]]
    assert path_mutations == []


def test_quiescence_reconciles_child_hidden_for_multiple_empty_scans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child hidden for multiple scans resets the grace and is terminalized."""
    root = tmp_path / "root"
    case_root, output_dir = _case(root)
    generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir,
        generation=generation.hex,
        mode="staged",
    )
    append_lifecycle_entry(
        output_dir,
        generation=generation.hex,
        token="controller-initial",
        role="controller-initial",
        status="submitted",
        job_id="710",
    )
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(root))
    monkeypatch.setattr(
        live,
        "query_scheduler_comments",
        lambda **_kwargs: _comment_matches(
            generation,
            "710",
            "controller-initial",
        ),
    )
    active_responses = iter(
        (
            set(),
            set(),
            set(),
            {"711"},
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
        )
    )
    comment_scans: list[set[str]] = []

    def delayed_child(_generation: UUID) -> set[str]:
        response = next(active_responses)
        comment_scans.append(response)
        return response

    monkeypatch.setattr(
        live,
        "active_generation_comment_ids",
        delayed_child,
    )
    queried_ids: list[set[str]] = []
    child_state_queries = 0

    def terminal_states(job_ids: set[str]) -> SchedulerQueryResult:
        nonlocal child_state_queries
        queried_ids.append(set(job_ids))
        states = {job_id: "CANCELLED" for job_id in job_ids}
        if "711" in job_ids:
            child_state_queries += 1
            if child_state_queries == 1:
                states["711"] = "RUNNING"
        return SchedulerQueryResult(
            states=states,
            available=True,
        )

    monkeypatch.setattr(live, "query_known_job_states", terminal_states)
    commands: list[list[str]] = []

    def record_scancel(command, **_kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(live.subprocess, "run", record_scancel)
    fake_now = 0.0

    def monotonic() -> float:
        return fake_now

    def sleep(seconds: float) -> None:
        nonlocal fake_now
        fake_now += seconds

    monkeypatch.setattr(live.time, "monotonic", monotonic)
    monkeypatch.setattr(live.time, "sleep", sleep)

    live.cleanup_case(
        case_root,
        output_dir,
        generation,
        iter(("710",)),
        forbidden=(),
    )

    assert commands == [["scancel", "710"], ["scancel", "711"]]
    assert comment_scans[:3] == [set(), set(), set()]
    assert comment_scans[3] == {"711"}
    assert len(comment_scans) == 11
    assert queried_ids[:3] == [{"710"}, {"710"}, {"710"}]
    assert queried_ids[3] == {"710", "711"}
    assert queried_ids[-2:] == [{"710", "711"}, {"710", "711"}]
    (evidence,) = _cleanup_evidence(case_root)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["scheduler_job_ids"] == ["710", "711"]


def test_protected_inode_moved_into_case_is_retained_and_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A moved protected directory is retained and recorded without traversal."""
    root = tmp_path / "root"
    case_root, output_dir = _case(root)
    generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir,
        generation=generation.hex,
        mode="ordinary",
    )
    append_lifecycle_entry(
        output_dir,
        generation=generation.hex,
        token="chunk-0",
        role="chunk",
        status="submitted",
        job_id="706",
    )
    protected = tmp_path / "protected-results"
    protected.mkdir()
    (protected / "keep.txt").write_text("keep", encoding="utf-8")
    _configure_terminal_scheduler(
        monkeypatch,
        generation,
        "706",
        token="chunk-0",
    )
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(root))
    original_write = live.write_retained_case_evidence

    def move_protected_then_write(target, **kwargs):
        protected.rename(case_root / "moved-protected")
        return original_write(target, **kwargs)

    monkeypatch.setattr(
        live,
        "write_retained_case_evidence",
        move_protected_then_write,
    )

    live.cleanup_case(
        case_root,
        output_dir,
        generation,
        iter(("706",)),
        forbidden=(protected,),
    )

    assert case_root.is_dir()
    assert (case_root / "moved-protected" / "keep.txt").read_text(
        encoding="utf-8"
    ) == "keep"
    (evidence,) = _cleanup_evidence(case_root)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["forbidden_paths"][0]["current_status"] == "missing"


@pytest.mark.parametrize("swap_target", ["case", "output"])
def test_path_swap_before_evidence_writes_only_to_original_open_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_target: str,
) -> None:
    """A swapped visible path cannot redirect fd-relative cleanup evidence."""
    root = tmp_path / "root"
    case_root, output_dir = _case(root)
    generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir,
        generation=generation.hex,
        mode="ordinary",
    )
    append_lifecycle_entry(
        output_dir,
        generation=generation.hex,
        token="finalizer",
        role="finalizer",
        status="submitted",
        job_id="707",
    )
    _configure_terminal_scheduler(
        monkeypatch,
        generation,
        "707",
        token="finalizer",
    )
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(root))
    relocated = tmp_path / f"relocated-{swap_target}-before-evidence"
    original_write = live.write_retained_case_evidence

    def swap_then_write(target, **kwargs):
        if swap_target == "case":
            case_root.rename(relocated)
            case_root.mkdir()
        else:
            output_dir.rename(relocated)
            output_dir.mkdir()
        return original_write(target, **kwargs)

    monkeypatch.setattr(
        live,
        "write_retained_case_evidence",
        swap_then_write,
    )

    live.cleanup_case(
        case_root,
        output_dir,
        generation,
        iter(("707",)),
        forbidden=(),
    )

    assert case_root.is_dir()
    assert relocated.is_dir()
    evidence_root = relocated if swap_target == "case" else case_root
    if swap_target == "case":
        assert _cleanup_evidence(case_root) == ()
    (evidence,) = _cleanup_evidence(evidence_root)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    if swap_target == "case":
        assert payload["case_ino"] == relocated.stat().st_ino
    else:
        assert payload["output_ino"] == relocated.stat().st_ino
        assert payload["output_ino"] != output_dir.stat().st_ino


def test_evidence_validation_rejects_case_swap_before_open_without_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A case swapped after lstat fails its fd identity check before any read."""
    case_root, _output_dir, generation, evidence_name = (
        _create_retained_evidence(tmp_path, monkeypatch)
    )
    relocated = tmp_path / "relocated-validation-case"
    original_open_directory = live._open_nofollow_directory
    swapped = False

    def swap_case_then_open(name, *, dir_fd=None):
        nonlocal swapped
        if Path(name) == case_root and dir_fd is None and not swapped:
            swapped = True
            case_root.rename(relocated)
            case_root.mkdir()
        return original_open_directory(name, dir_fd=dir_fd)

    read_calls: list[int] = []
    original_read = live.os.read

    def record_read(fd: int, size: int) -> bytes:
        read_calls.append(fd)
        return original_read(fd, size)

    monkeypatch.setattr(
        live,
        "_open_nofollow_directory",
        swap_case_then_open,
    )
    monkeypatch.setattr(live.os, "read", record_read)

    with pytest.raises(AssertionError, match="retained case identity changed"):
        live.validate_retained_case_evidence(
            case_root,
            evidence_name,
            scheduler_generation=generation,
        )

    assert swapped
    assert read_calls == []
    assert _cleanup_evidence(case_root) == ()
    assert tuple(path.name for path in _cleanup_evidence(relocated)) == (
        evidence_name,
    )


def test_evidence_validation_rejects_case_symlink_without_following(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A visible case symlink is rejected before opening or reading its target."""
    case_root, _output_dir, generation, evidence_name = (
        _create_retained_evidence(tmp_path, monkeypatch)
    )
    relocated = tmp_path / "relocated-validation-symlink-case"
    case_root.rename(relocated)
    case_root.symlink_to(relocated, target_is_directory=True)
    follow_calls = _forbid_following_path(case_root, monkeypatch)
    directory_open_calls: list[Path] = []
    original_open_directory = live._open_nofollow_directory

    def record_directory_open(name, *, dir_fd=None):
        if Path(name) == case_root and dir_fd is None:
            directory_open_calls.append(Path(name))
        return original_open_directory(name, dir_fd=dir_fd)

    read_calls: list[int] = []
    monkeypatch.setattr(
        live,
        "_open_nofollow_directory",
        record_directory_open,
    )
    monkeypatch.setattr(
        live.os,
        "read",
        lambda fd, _size: read_calls.append(fd) or b"",
    )

    with pytest.raises(AssertionError, match="unsafe live cleanup retained case"):
        live.validate_retained_case_evidence(
            case_root,
            evidence_name,
            scheduler_generation=generation,
        )

    assert follow_calls == []
    assert directory_open_calls == []
    assert read_calls == []


def test_evidence_validation_rejects_evidence_swap_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An evidence file swapped after no-follow stat fails before payload reads."""
    case_root, _output_dir, generation, evidence_name = (
        _create_retained_evidence(tmp_path, monkeypatch)
    )
    evidence_path = case_root / evidence_name
    relocated = case_root / "relocated-cleanup-evidence.json"
    replacement_payload = evidence_path.read_bytes()
    original_open = live.os.open
    swapped = False

    def swap_evidence_then_open(name, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if name == evidence_name and dir_fd is not None and not swapped:
            swapped = True
            evidence_path.rename(relocated)
            evidence_path.write_bytes(replacement_payload)
        return original_open(name, flags, mode, dir_fd=dir_fd)

    read_calls: list[int] = []
    original_read = live.os.read

    def record_read(fd: int, size: int) -> bytes:
        read_calls.append(fd)
        return original_read(fd, size)

    monkeypatch.setattr(live.os, "open", swap_evidence_then_open)
    monkeypatch.setattr(live.os, "read", record_read)

    with pytest.raises(
        AssertionError,
        match="retained cleanup evidence .* identity changed",
    ):
        live.validate_retained_case_evidence(
            case_root,
            evidence_name,
            scheduler_generation=generation,
        )

    assert swapped
    assert read_calls == []
    assert relocated.read_bytes() == replacement_payload


def test_evidence_validation_rejects_evidence_symlink_without_following(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A symlinked evidence entry is rejected before open or payload reads."""
    case_root, _output_dir, generation, evidence_name = (
        _create_retained_evidence(tmp_path, monkeypatch)
    )
    evidence_path = case_root / evidence_name
    relocated = tmp_path / "relocated-cleanup-evidence.json"
    evidence_path.rename(relocated)
    evidence_path.symlink_to(relocated)
    original_open = live.os.open
    evidence_open_calls: list[str] = []

    def record_open(name, flags, mode=0o777, *, dir_fd=None):
        if name == evidence_name and dir_fd is not None:
            evidence_open_calls.append(name)
        return original_open(name, flags, mode, dir_fd=dir_fd)

    read_calls: list[int] = []
    monkeypatch.setattr(live.os, "open", record_open)
    monkeypatch.setattr(
        live.os,
        "read",
        lambda fd, _size: read_calls.append(fd) or b"",
    )

    with pytest.raises(
        AssertionError,
        match="unsafe scheduler-state file: retained cleanup evidence",
    ):
        live.validate_retained_case_evidence(
            case_root,
            evidence_name,
            scheduler_generation=generation,
        )

    assert evidence_open_calls == []
    assert read_calls == []
    assert relocated.is_file()


def test_partial_setup_failure_is_retained_with_forbidden_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-submission failures retain their partial fixture for root disposal."""

    def fail_image_write(_input_dir: Path) -> Path:
        raise RuntimeError("fixture write failed")

    protected = tmp_path / "protected"
    protected.mkdir()
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(live, "write_one_small_image", fail_image_write)

    with pytest.raises(RuntimeError, match="fixture write failed"):
        with live.prepared_case(
            tmp_path,
            (protected,),
        ):
            pytest.fail("setup failure must occur before yield")

    retained = tuple(tmp_path.glob(f"{live.CASE_PREFIX}*"))
    assert len(retained) == 1
    assert (retained[0] / "input").is_dir()
    (evidence,) = _cleanup_evidence(retained[0])
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["status"] == "retained-pre-submit-failure"
    assert payload["forbidden_paths"][0]["path"] == str(protected)


def test_prepared_case_pipeline_detects_and_measures_one_colony(
    tmp_path: Path,
) -> None:
    """The live fixture must produce real work for the generated pipeline."""
    with live.prepared_case(tmp_path, ()) as (
        case_root,
        pipeline_path,
        _output_dir,
    ):
        (image_path,) = (case_root / "input").glob("*.tiff")
        pipeline = ImagePipeline.from_json(
            pipeline_path.read_text(encoding="utf-8"),
        )
        detected = pipeline.apply(Image.imread(str(image_path)))
        measurements = pipeline.measure(
            detected,
            include_metadata=False,
            apply_post=False,
        )

    assert detected.num_objects == 1
    assert len(measurements) == 1
    assert measurements["Size_Area"].iloc[0] > 0


def test_cleanup_retains_case_when_queue_quiescence_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_root, output_dir = _case(tmp_path)
    generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir,
        generation=generation.hex,
        mode="ordinary",
    )
    append_lifecycle_entry(
        output_dir,
        generation=generation.hex,
        token="chunk-0",
        role="chunk",
        status="submitted",
        job_id="701",
    )
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(
        live,
        "query_scheduler_comments",
        lambda **_kwargs: _comment_matches(generation, "701", "chunk-0"),
    )
    monkeypatch.setattr(
        live,
        "active_generation_comment_ids",
        lambda _generation: set(),
    )
    monkeypatch.setattr(
        live,
        "query_known_job_states",
        lambda _job_ids: SchedulerQueryResult(
            states={},
            available=False,
            detail="permission denied",
        ),
    )
    monkeypatch.setattr(live, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(live, "POLL_SECONDS", 0.0)
    monkeypatch.setattr(
        live.subprocess,
        "run",
        lambda *_args, **_kwargs: _completed_scancel("701", returncode=1),
    )

    with pytest.raises(AssertionError, match="quiescence was not proven"):
        live.cleanup_case(
            case_root,
            output_dir,
            generation,
            iter(("701",)),
            forbidden=(),
        )

    assert case_root.is_dir()
    assert _cleanup_evidence(case_root) == ()


def test_empty_comment_query_cannot_hide_known_running_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_root, output_dir = _case(tmp_path)
    generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir,
        generation=generation.hex,
        mode="ordinary",
    )
    append_lifecycle_entry(
        output_dir,
        generation=generation.hex,
        token="chunk-0",
        role="chunk",
        status="submitted",
        job_id="703",
    )
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(live, "query_scheduler_comments", lambda **_kwargs: {})
    monkeypatch.setattr(
        live,
        "active_generation_comment_ids",
        lambda _generation: set(),
    )
    monkeypatch.setattr(
        live,
        "query_known_job_states",
        lambda _job_ids: SchedulerQueryResult(
            states={"703": "RUNNING"},
            available=True,
        ),
    )
    monkeypatch.setattr(live, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(live, "POLL_SECONDS", 0.0)
    monkeypatch.setattr(
        live.subprocess,
        "run",
        lambda *_args, **_kwargs: _completed_scancel("703", returncode=1),
    )

    with pytest.raises(AssertionError, match="quiescence was not proven"):
        live.cleanup_case(
            case_root,
            output_dir,
            generation,
            iter(("703",)),
            forbidden=(),
        )

    assert case_root.is_dir()


def test_corrupt_lifecycle_recovers_metadata_job_but_retains_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_root, output_dir = _case(tmp_path)
    generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir,
        generation=generation.hex,
        mode="ordinary",
    )
    lifecycle_state_path(output_dir).write_text("{corrupt", encoding="utf-8")
    atomic_write_json(
        job_metadata_path(output_dir),
        {
            "slurm_generation": generation.hex,
            "slurm_job_ids": {
                "chunk-0": {
                    "job_id": "704",
                    "role": "chunk",
                    "generation": generation.hex,
                }
            },
        },
    )
    commands = _configure_terminal_scheduler(
        monkeypatch,
        generation,
        "704",
        token="chunk-0",
    )
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(tmp_path))

    with pytest.raises(AssertionError, match="slurm lifecycle is malformed"):
        live.cleanup_case(
            case_root,
            output_dir,
            None,
            iter(()),
            forbidden=(),
        )

    assert commands == [["scancel", "704"]]
    assert case_root.is_dir()


def test_partial_ledger_only_submission_is_recovered_and_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_root, output_dir = _case(tmp_path)
    generation = uuid4()
    append_lifecycle_entry(
        output_dir,
        generation=generation.hex,
        token="finalizer",
        role="finalizer",
        status="submitted",
        job_id="705",
    )
    _configure_terminal_scheduler(
        monkeypatch,
        generation,
        "705",
        token="finalizer",
    )
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(tmp_path))

    live.cleanup_case(
        case_root,
        output_dir,
        None,
        iter(()),
        forbidden=(),
    )

    assert case_root.is_dir()
    (evidence,) = _cleanup_evidence(case_root)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["scheduler_job_ids"] == ["705"]


def test_retain_partial_case_rejects_prefix_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    malformed = root / f"{live.CASE_PREFIX}{uuid4().hex}-extra"
    malformed.mkdir()
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(root))

    with pytest.raises(AssertionError, match="refusing unsafe"):
        live.retain_partial_case(malformed, forbidden=())

    assert malformed.is_dir()


def test_retain_partial_case_lstats_symlink_without_following_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    case_root = root / f"{live.CASE_PREFIX}{uuid4().hex}"
    target = tmp_path / "real-case"
    target.mkdir()
    case_root.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(root))
    follow_calls = _forbid_following_path(case_root, monkeypatch)

    with pytest.raises(AssertionError, match="refusing unsafe"):
        live.retain_partial_case(case_root, forbidden=())

    assert follow_calls == []
    assert case_root.is_symlink()
    assert target.is_dir()


def test_environment_rejects_duplicate_active_and_latest_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    protected = tmp_path / "protected"
    protected.mkdir()
    state_path = processing_state_path(protected)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(live, "require_exact_clean_source", lambda: None)
    monkeypatch.setenv(live.LIVE_ROOT_ENV, str(root))
    monkeypatch.setenv(live.PARTITION_ENV, "short")
    monkeypatch.setenv(live.ACTIVE_OUTPUT_ENV, str(protected))
    monkeypatch.setenv(live.LATEST_RESULTS_ENV, str(protected))
    monkeypatch.delenv(live.NO_ACTIVE_OUTPUT_SENTINEL_ENV, raising=False)

    with pytest.raises(pytest.fail.Exception, match="must be distinct"):
        live.require_live_environment()


def test_no_active_output_sentinel_rejects_placeholder_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = tmp_path / "inspection.json"
    sentinel.write_text(
        json.dumps({"status": "no-active-output"}),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        live.NO_ACTIVE_OUTPUT_SENTINEL_ENV,
        str(sentinel),
    )

    with pytest.raises(pytest.fail.Exception, match="does not match"):
        live.require_no_active_output_sentinel()


def test_exact_source_gate_rejects_dirty_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "a" * 40
    monkeypatch.setenv(live.EXPECTED_SHA_ENV, expected)

    def fake_git(command, **_kwargs):
        if command[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, expected + "\n", "")
        return subprocess.CompletedProcess(command, 0, "?? unexpected.txt\n", "")

    monkeypatch.setattr(live.subprocess, "run", fake_git)

    with pytest.raises(pytest.fail.Exception, match="not clean"):
        live.require_exact_clean_source()
