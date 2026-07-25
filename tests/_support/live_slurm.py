"""Shared fail-closed safety helpers for opt-in live SLURM tests."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from phenotypic import Image, ImagePipeline
from phenotypic._cli._cli_slurm_lifecycle import (
    SchedulerQueryUnavailable,
    lifecycle_state_path,
    query_scheduler_comments,
    read_lifecycle_ledger,
)
from phenotypic.detect import OtsuDetector
from phenotypic.gui.run_console._slurm_observer import (
    SchedulerQueryResult,
    SlurmCommandScheduler,
)
from phenotypic.measure import MeasureSize
from phenotypic.sdk_ import (
    JobMetadataKey,
    gui_launch_owner_path,
    job_metadata_path,
    processing_state_path,
)

LIVE_ROOT_ENV = "PHENOTYPIC_LIVE_SLURM_ROOT"
ACTIVE_OUTPUT_ENV = "PHENOTYPIC_LIVE_SLURM_ACTIVE_OUTPUT"
NO_ACTIVE_OUTPUT_SENTINEL_ENV = (
    "PHENOTYPIC_LIVE_SLURM_NO_ACTIVE_OUTPUT_SENTINEL"
)
LATEST_RESULTS_ENV = "PHENOTYPIC_LIVE_SLURM_LATEST_RESULTS"
PARTITION_ENV = "PHENOTYPIC_TEST_SLURM_PARTITION"
ACCOUNT_ENV = "PHENOTYPIC_TEST_SLURM_ACCOUNT"
EXPECTED_SHA_ENV = "PHENOTYPIC_LIVE_SLURM_EXPECTED_SHA"
CASE_PREFIX = "gui-v1-live-"
CASE_NAME_RE = re.compile(rf"{re.escape(CASE_PREFIX)}([0-9a-f]{{32}})")
POLL_SECONDS = 2.0
CLEANUP_TIMEOUT_SECONDS = 60.0
RECONCILIATION_GRACE_SECONDS = 10.0
MAX_CLEANUP_EVIDENCE_BYTES = 64 * 1024
REPO_ROOT = Path(__file__).resolve().parents[2]
TERMINAL_SCHEDULER_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)


@dataclass(frozen=True)
class AnchoredCleanupTarget:
    """Open directory handles binding cleanup to validated filesystem objects."""

    root: Path
    case_root: Path
    output_dir: Path
    case_generation: str
    root_fd: int
    case_fd: int
    output_fd: int
    root_identity: tuple[int, int]
    case_identity: tuple[int, int]
    output_identity: tuple[int, int]
    forbidden_identities: tuple[tuple[Path, tuple[int, int]], ...]
    scheduler_snapshot: SchedulerDurableSnapshot


@dataclass(frozen=True)
class SchedulerDurableSnapshot:
    """Minimal scheduler identity state read without following path components."""

    lifecycle: Mapping[str, object] | None
    metadata: Mapping[str, object] | None
    ledger_rows: tuple[Mapping[str, object], ...]
    errors: tuple[str, ...]


def require_live_environment() -> tuple[Path, str, tuple[Path, ...]]:
    """Validate source identity, shared root, and inspected path exclusions."""
    require_exact_clean_source()
    raw_root = os.environ.get(LIVE_ROOT_ENV, "").strip()
    if not raw_root:
        pytest.fail(f"{LIVE_ROOT_ENV} must name a shared test-only directory")
    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        pytest.fail(f"{LIVE_ROOT_ENV} must be an existing absolute directory")
    try:
        root_lstat = os.lstat(root)
    except OSError:
        pytest.fail(f"{LIVE_ROOT_ENV} must be an existing absolute directory")
    if stat.S_ISLNK(root_lstat.st_mode):
        pytest.fail(f"{LIVE_ROOT_ENV} must be a canonical non-symlink path")
    if not stat.S_ISDIR(root_lstat.st_mode):
        pytest.fail(f"{LIVE_ROOT_ENV} must be an existing absolute directory")
    if root.resolve() != root:
        pytest.fail(f"{LIVE_ROOT_ENV} must be a canonical non-symlink path")

    partition = os.environ.get(PARTITION_ENV, "").strip()
    if not partition:
        pytest.fail(
            f"{PARTITION_ENV} must name a partition verified by read-only sinfo"
        )

    latest_results = require_existing_canonical_path(
        LATEST_RESULTS_ENV,
        expect_directory=True,
    )
    raw_active = os.environ.get(ACTIVE_OUTPUT_ENV, "").strip()
    raw_sentinel = os.environ.get(
        NO_ACTIVE_OUTPUT_SENTINEL_ENV,
        "",
    ).strip()
    if bool(raw_active) == bool(raw_sentinel):
        pytest.fail(
            f"set exactly one of {ACTIVE_OUTPUT_ENV} or "
            f"{NO_ACTIVE_OUTPUT_SENTINEL_ENV}"
        )
    active_guard = (
        require_active_output_path()
        if raw_active
        else require_no_active_output_sentinel()
    )
    if latest_results == active_guard:
        pytest.fail(
            "active-output evidence and latest-results path must be distinct"
        )
    forbidden = (active_guard, latest_results)
    for active_path in forbidden:
        if paths_overlap(root, active_path):
            pytest.fail(
                f"live test root {root} overlaps active result path {active_path}"
            )
    return root, partition, forbidden


def require_exact_clean_source() -> None:
    """Refuse live submission from a dirty or unexpected source checkout."""
    expected = os.environ.get(EXPECTED_SHA_ENV, "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        pytest.fail(f"{EXPECTED_SHA_ENV} must contain the reviewed 40-char SHA")
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip().lower()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    if head != expected:
        pytest.fail(f"reviewed SHA {expected} does not match checkout HEAD {head}")
    if status:
        pytest.fail("live SLURM source checkout is not clean:\n" + status)
    print(f"LIVE_SLURM_SOURCE head={head} clean=true root={REPO_ROOT}")


def require_existing_canonical_path(
    env_name: str,
    *,
    expect_directory: bool,
) -> Path:
    """Read one independently labeled canonical inspected path."""
    raw_path = os.environ.get(env_name, "").strip()
    if not raw_path:
        pytest.fail(f"{env_name} is required")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        pytest.fail(f"{env_name} must be an existing absolute path")
    try:
        path_lstat = os.lstat(path)
    except OSError:
        pytest.fail(f"{env_name} must be an existing absolute path")
    if stat.S_ISLNK(path_lstat.st_mode):
        pytest.fail(f"{env_name} must be canonical and non-symlinked")
    if path.resolve() != path:
        pytest.fail(f"{env_name} must be canonical and non-symlinked")
    if expect_directory and not stat.S_ISDIR(path_lstat.st_mode):
        pytest.fail(f"{env_name} must name a directory")
    return path


def require_no_active_output_sentinel() -> Path:
    """Validate a read-only scheduler inspection record when no output is active."""
    path = require_existing_canonical_path(
        NO_ACTIVE_OUTPUT_SENTINEL_ENV,
        expect_directory=False,
    )
    if not path.is_file() or path.suffix != ".json":
        pytest.fail(
            f"{NO_ACTIVE_OUTPUT_SENTINEL_ENV} must name an inspection JSON file"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        pytest.fail(f"invalid no-active-output inspection record: {exc}")
    required = {
        "status": "no-active-output",
        "host": "bluejay",
        "user": "anguy344",
        "squeue_job_count": 0,
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in required.items()
    ):
        pytest.fail(
            "no-active-output inspection record does not match the required "
            f"read-only scheduler evidence: {required}"
        )
    if not isinstance(payload.get("inspected_at"), str):
        pytest.fail("no-active-output inspection record lacks inspected_at")
    return path


def require_active_output_path() -> Path:
    """Require durable run evidence under an explicitly protected active output."""
    path = require_existing_canonical_path(
        ACTIVE_OUTPUT_ENV,
        expect_directory=True,
    )
    evidence = (
        gui_launch_owner_path(path),
        job_metadata_path(path),
        lifecycle_state_path(path),
        processing_state_path(path),
    )
    if not any(item.is_file() for item in evidence):
        pytest.fail(f"{ACTIVE_OUTPUT_ENV} has no durable GUI/CLI scheduler evidence")
    return path


def paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either canonical path contains the other."""
    first = first.resolve(strict=False)
    second = second.resolve(strict=False)
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


@contextmanager
def prepared_case(
    root: Path,
    forbidden: tuple[Path, ...],
) -> Iterator[tuple[Path, Path, Path]]:
    """Yield one exact case, retaining incomplete setup with fd-bound evidence."""
    generation = uuid4().hex
    case_root = root / f"{CASE_PREFIX}{generation}"
    input_dir = case_root / "input"
    output_dir = case_root / f"output-{generation}"
    pipeline_path = case_root / "pipeline.json.pht-pipe"
    setup_complete = False
    try:
        case_root.mkdir(mode=0o700)
        input_dir.mkdir()

        canonical_case = case_root.resolve()
        canonical_input = input_dir.resolve()
        canonical_output = output_dir.resolve(strict=False)
        assert case_root.name == f"{CASE_PREFIX}{generation}"
        assert canonical_case.parent == root
        assert not output_dir.exists()
        assert not output_dir.is_symlink()
        assert not paths_overlap(canonical_input, canonical_output)
        for active_path in forbidden:
            assert not paths_overlap(canonical_input, active_path)
            assert not paths_overlap(canonical_output, active_path)

        write_one_small_image(input_dir)
        pipeline_path.write_text(
            ImagePipeline(
                ops=[OtsuDetector()],
                meas=[MeasureSize()],
            ).to_json(),
            encoding="utf-8",
        )
        assert len(tuple(input_dir.glob("*.tiff"))) == 1
        setup_complete = True
        yield case_root, pipeline_path, output_dir
    finally:
        if not setup_complete:
            try:
                os.lstat(case_root)
            except FileNotFoundError:
                pass
            else:
                retain_partial_case(
                    case_root,
                    forbidden=forbidden,
                )


def write_one_small_image(input_dir: Path) -> Path:
    """Write exactly one deterministic 96 x 96 RGB colony image."""
    rows, cols = np.ogrid[:96, :96]
    colony = (rows - 48) ** 2 + (cols - 48) ** 2 <= 20**2
    rgb = np.full((96, 96, 3), 228, dtype=np.uint8)
    rgb[colony] = (38, 72, 41)
    image_path = input_dir / "single-small-colony.tiff"
    Image(arr=rgb, name=image_path.stem).rgb.imsave(filepath=image_path)
    return image_path


def jobs_by_role(
    output_dir: Path,
    generation: UUID,
) -> dict[str, tuple[str, ...]]:
    """Read the latest submitted/recovered jobs grouped by durable role."""
    grouped: dict[str, list[str]] = {}
    for row in read_lifecycle_ledger(
        output_dir,
        generation=generation.hex,
    ):
        job_id = row.get("job_id")
        if row.get("status") not in {"submitted", "recovered"} or not job_id:
            continue
        role = str(row.get("role", "unknown"))
        grouped.setdefault(role, [])
        if str(job_id) not in grouped[role]:
            grouped[role].append(str(job_id))
    return {role: tuple(ids) for role, ids in sorted(grouped.items())}


def generation_comment_ids(generation: UUID) -> set[str]:
    """Return all queue/accounting ids bearing the exact generation prefix."""
    matches = query_scheduler_comments(
        prefix=f"phenotypic:{generation.hex}:",
    )
    return {job_id for ids in matches.values() for job_id in ids}


def active_generation_comment_ids(generation: UUID) -> set[str]:
    """Return currently queued/running ids bearing the generation prefix."""
    matches = query_scheduler_comments(
        prefix=f"phenotypic:{generation.hex}:",
        include_accounting=False,
    )
    return {job_id for ids in matches.values() for job_id in ids}


def parse_generation(value: object) -> UUID | None:
    """Parse one scheduler generation emitted with or without hyphens."""
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


def recover_scheduler_generation(
    snapshot: SchedulerDurableSnapshot,
    supplied: UUID | None,
) -> tuple[UUID | None, tuple[str, ...]]:
    """Recover identity from the no-follow durable scheduler snapshot."""
    errors = list(snapshot.errors)
    lifecycle = snapshot.lifecycle
    lifecycle_generation = parse_generation(
        (
            lifecycle.get("generation", lifecycle.get("epoch"))
            if lifecycle
            else None
        )
    )
    metadata = snapshot.metadata
    metadata_generation = (
        parse_generation(
            metadata.get("slurm_generation")
            or metadata.get(JobMetadataKey.ORCHESTRATION_EPOCH)
        )
        if isinstance(metadata, dict)
        else None
    )
    ledger_generations: set[UUID] = set()
    for row in snapshot.ledger_rows:
        parsed = parse_generation(
            row.get("generation", row.get("epoch"))
        )
        if parsed is not None:
            ledger_generations.add(parsed)

    authoritative = supplied or lifecycle_generation or metadata_generation
    if authoritative is None and len(ledger_generations) == 1:
        authoritative = next(iter(ledger_generations))
    if authoritative is None and len(ledger_generations) > 1:
        errors.append("multiple ledger generations prevent safe recovery")
    for label, candidate in (
        ("lifecycle", lifecycle_generation),
        ("metadata", metadata_generation),
    ):
        if (
            authoritative is not None
            and candidate is not None
            and candidate != authoritative
        ):
            errors.append(
                f"{label} generation {candidate.hex} conflicts with "
                f"{authoritative.hex}"
            )
    if authoritative is not None and ledger_generations:
        if ledger_generations != {authoritative}:
            errors.append(
                "durable ledger generation set conflicts with recovered "
                f"generation {authoritative.hex}"
            )
    return authoritative, tuple(errors)


def recover_durable_jobs(
    snapshot: SchedulerDurableSnapshot,
    generation: UUID,
) -> tuple[
    dict[str, tuple[str, ...]],
    set[str],
    set[str],
    tuple[str, ...],
]:
    """Merge no-follow metadata/ledger jobs and unresolved submission intents."""
    errors: list[str] = []
    grouped: dict[str, list[str]] = {}
    job_ids: set[str] = set()
    submitted_tokens: set[str] = set()
    intent_tokens: set[str] = set()
    generation_texts = {str(generation), generation.hex}

    def add(job_id: object, *, role: str, token: str) -> None:
        canonical = str(job_id)
        if not canonical.isdigit():
            return
        job_ids.add(canonical)
        grouped.setdefault(role or "unknown", [])
        if canonical not in grouped[role or "unknown"]:
            grouped[role or "unknown"].append(canonical)
        if token:
            submitted_tokens.add(token)

    metadata = snapshot.metadata or {}
    metadata_generation = parse_generation(
        metadata.get("slurm_generation")
        or metadata.get(JobMetadataKey.ORCHESTRATION_EPOCH)
    )
    if metadata_generation is None or metadata_generation == generation:
        raw_jobs = metadata.get(JobMetadataKey.SLURM_JOB_IDS)
        if isinstance(raw_jobs, Mapping):
            for raw_token, raw_value in raw_jobs.items():
                token = str(raw_token)
                if isinstance(raw_value, Mapping):
                    row_generation = raw_value.get("generation")
                    if (
                        row_generation is not None
                        and str(row_generation) not in generation_texts
                    ):
                        continue
                    add(
                        raw_value.get("job_id"),
                        role=str(raw_value.get("role", "unknown")),
                        token=token,
                    )
                else:
                    add(raw_value, role="unknown", token=token)
        raw_chunks = metadata.get(JobMetadataKey.CHUNK_JOB_IDS)
        if isinstance(raw_chunks, Mapping):
            for raw_index, raw_id in raw_chunks.items():
                add(
                    raw_id,
                    role="chunk",
                    token=f"chunk-{raw_index}",
                )

    for row in snapshot.ledger_rows:
        row_generation = row.get("generation", row.get("epoch"))
        if str(row_generation) not in generation_texts:
            continue
        token = str(row.get("token", ""))
        status = str(row.get("status", ""))
        if status in {"intent", "blocked"} and token:
            intent_tokens.add(token)
        if status in {"submitted", "recovered", "terminal"}:
            add(
                row.get("job_id"),
                role=str(row.get("role", "unknown")),
                token=token,
            )
        if status == "reconciled-no-job" and token:
            submitted_tokens.add(token)
    roles = {
        role: tuple(ids)
        for role, ids in sorted(grouped.items())
    }
    return (
        roles,
        job_ids,
        intent_tokens - submitted_tokens,
        tuple(errors),
    )


def query_known_job_states(job_ids: set[str]) -> SchedulerQueryResult:
    """Query each known id through independent squeue and sacct commands."""
    return SlurmCommandScheduler(timeout_seconds=15.0).query(
        tuple(sorted(job_ids, key=int))
    )


def inactive_known_job_ids(
    job_ids: set[str],
    result: SchedulerQueryResult,
) -> tuple[bool, str]:
    """Classify terminal/not-found separately from scheduler query failure."""
    if not result.available:
        return False, result.detail or "scheduler state query unavailable"
    nonterminal = {
        job_id: result.states[job_id]
        for job_id in job_ids
        if job_id in result.states
        and str(result.states[job_id]).split("+", 1)[0].upper()
        not in TERMINAL_SCHEDULER_STATES
    }
    if nonterminal:
        detail = ", ".join(
            f"{job_id}={state}"
            for job_id, state in sorted(
                nonterminal.items(),
                key=lambda item: int(item[0]),
            )
        )
        return False, f"known scheduler jobs remain nonterminal: {detail}"
    return True, "every known job is terminal or absent from scheduler sources"


def _identity(file_stat: os.stat_result) -> tuple[int, int]:
    """Return the stable device/inode identity for an opened object."""
    return file_stat.st_dev, file_stat.st_ino


def _require_nofollow_directory(path: Path, *, label: str) -> os.stat_result:
    """Validate one absolute directory with lstat before any following lookup."""
    if not path.is_absolute():
        raise AssertionError(f"unsafe live cleanup {label}: {path}")
    try:
        path_stat = os.lstat(path)
    except OSError as exc:
        raise AssertionError(
            f"unsafe live cleanup {label}: {path}: {exc}"
        ) from exc
    if stat.S_ISLNK(path_stat.st_mode):
        raise AssertionError(f"unsafe live cleanup {label}: {path}")
    if not stat.S_ISDIR(path_stat.st_mode):
        raise AssertionError(f"unsafe live cleanup {label}: {path}")
    if path.resolve() != path:
        raise AssertionError(f"unsafe live cleanup {label}: {path}")
    return path_stat


def _open_nofollow_directory(
    name: str | Path,
    *,
    dir_fd: int | None = None,
) -> int:
    """Open one directory without following its final path component."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise AssertionError("live SLURM cleanup requires O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_DIRECTORY | nofollow
    return os.open(name, flags, dir_fd=dir_fd)


def _require_same_identity(
    actual: os.stat_result,
    expected: os.stat_result | tuple[int, int],
    *,
    label: str,
) -> None:
    """Reject any object swapped between no-follow validation and opening."""
    expected_identity = (
        _identity(expected)
        if isinstance(expected, os.stat_result)
        else expected
    )
    if _identity(actual) != expected_identity:
        raise AssertionError(f"live cleanup {label} identity changed")


def _open_optional_child_directory(
    parent_fd: int,
    name: str,
    *,
    label: str,
) -> int | None:
    """Open one optional child directory without following any symlink."""
    try:
        child_stat = os.stat(
            name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(child_stat.st_mode) or not stat.S_ISDIR(
        child_stat.st_mode
    ):
        raise AssertionError(f"unsafe scheduler-state directory: {label}")
    child_fd = _open_nofollow_directory(name, dir_fd=parent_fd)
    try:
        _require_same_identity(
            os.fstat(child_fd),
            child_stat,
            label=label,
        )
    except BaseException:
        os.close(child_fd)
        raise
    return child_fd


def _read_optional_regular_file(
    parent_fd: int,
    name: str,
    *,
    label: str,
    max_bytes: int = 8 * 1024 * 1024,
) -> bytes | None:
    """Read one bounded regular file after a no-follow stat/open identity check."""
    try:
        file_stat = os.stat(
            name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(
        file_stat.st_mode
    ):
        raise AssertionError(f"unsafe scheduler-state file: {label}")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise AssertionError("live SLURM cleanup requires O_NOFOLLOW")
    file_fd = os.open(
        name,
        os.O_RDONLY | nofollow,
        dir_fd=parent_fd,
    )
    try:
        _require_same_identity(
            os.fstat(file_fd),
            file_stat,
            label=label,
        )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(file_fd, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > max_bytes:
            raise AssertionError(
                f"scheduler-state file exceeds {max_bytes} bytes: {label}"
            )
        return payload
    finally:
        os.close(file_fd)


def _parse_json_mapping(
    payload: bytes | None,
    *,
    label: str,
    errors: list[str],
) -> Mapping[str, object] | None:
    """Parse one optional JSON object while retaining malformed-file evidence."""
    if payload is None:
        return None
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{label} is malformed: {exc}")
        return None
    if not isinstance(parsed, dict):
        errors.append(f"{label} must contain a JSON object")
        return None
    return parsed


def read_scheduler_snapshot(output_fd: int) -> SchedulerDurableSnapshot:
    """Read minimal scheduler files through fd-relative no-follow operations."""
    errors: list[str] = []
    cache_fd = _open_optional_child_directory(
        output_fd,
        ".phenotypic",
        label=".phenotypic",
    )
    if cache_fd is None:
        return SchedulerDurableSnapshot(None, None, (), ())
    progress_fd = -1
    try:
        progress_fd = _open_optional_child_directory(
            cache_fd,
            "progress",
            label=".phenotypic/progress",
        )
        if progress_fd is None:
            return SchedulerDurableSnapshot(None, None, (), ())
        lifecycle = _parse_json_mapping(
            _read_optional_regular_file(
                progress_fd,
                "slurm_lifecycle.json",
                label=".phenotypic/progress/slurm_lifecycle.json",
            ),
            label="slurm lifecycle",
            errors=errors,
        )
        metadata = _parse_json_mapping(
            _read_optional_regular_file(
                progress_fd,
                "job_metadata.json",
                label=".phenotypic/progress/job_metadata.json",
            ),
            label="job metadata",
            errors=errors,
        )
        rows: list[Mapping[str, object]] = []
        for filename in ("slurm_jobs.jsonl", "staged_jobs.jsonl"):
            raw_ledger = _read_optional_regular_file(
                progress_fd,
                filename,
                label=f".phenotypic/progress/{filename}",
            )
            if raw_ledger is None:
                continue
            try:
                lines = raw_ledger.decode("utf-8").splitlines()
            except UnicodeDecodeError as exc:
                errors.append(f"{filename} is not UTF-8: {exc}")
                continue
            for line_number, raw_line in enumerate(lines, start=1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    errors.append(
                        f"{filename}:{line_number} is malformed: {exc}"
                    )
                    continue
                if not isinstance(row, dict):
                    errors.append(
                        f"{filename}:{line_number} is not a JSON object"
                    )
                    continue
                rows.append(row)
        return SchedulerDurableSnapshot(
            lifecycle,
            metadata,
            tuple(rows),
            tuple(errors),
        )
    finally:
        if progress_fd >= 0:
            os.close(progress_fd)
        os.close(cache_fd)


@contextmanager
def anchored_cleanup_target(
    case_root: Path,
    output_dir: Path,
    forbidden: Sequence[Path],
) -> Iterator[AnchoredCleanupTarget]:
    """Open and hold exact cleanup directories before scheduler access."""
    raw_root = os.environ.get(LIVE_ROOT_ENV, "").strip()
    root = Path(raw_root).expanduser()
    if not raw_root:
        raise AssertionError(
            f"unsafe live cleanup root from {LIVE_ROOT_ENV}: {root}"
        )
    root_lstat = _require_nofollow_directory(root, label="root")
    if (
        not case_root.is_absolute()
        or case_root.parent != root
    ):
        raise AssertionError(f"unsafe live cleanup case: {case_root}")
    case_lstat = _require_nofollow_directory(case_root, label="case")
    match = CASE_NAME_RE.fullmatch(case_root.name)
    if match is None:
        raise AssertionError(
            f"live cleanup case has malformed identity: {case_root.name}"
        )
    case_generation = match.group(1)
    if UUID(hex=case_generation).hex != case_generation:
        raise AssertionError(
            f"live cleanup case has invalid UUID identity: {case_root.name}"
        )

    expected_output = case_root / f"output-{case_generation}"
    if (
        not output_dir.is_absolute()
        or output_dir != expected_output
        or output_dir.parent != case_root
    ):
        raise AssertionError(f"unsafe live cleanup output: {output_dir}")
    output_lstat = _require_nofollow_directory(output_dir, label="output")

    protected_paths: list[tuple[Path, os.stat_result]] = []
    for protected in forbidden:
        if not protected.is_absolute():
            raise AssertionError(
                f"unsafe protected-path evidence during cleanup: {protected}"
            )
        try:
            protected_lstat = os.lstat(protected)
        except OSError as exc:
            raise AssertionError(
                f"unsafe protected-path evidence during cleanup: {protected}"
            ) from exc
        if stat.S_ISLNK(protected_lstat.st_mode):
            raise AssertionError(
                f"unsafe protected-path evidence during cleanup: {protected}"
            )
        if protected.resolve() != protected:
            raise AssertionError(
                f"unsafe protected-path evidence during cleanup: {protected}"
            )
        protected_paths.append((protected, protected_lstat))
    for candidate in (root, case_root, output_dir):
        for protected, _protected_lstat in protected_paths:
            if paths_overlap(candidate, protected):
                raise AssertionError(
                    f"live cleanup target {candidate} overlaps protected "
                    f"path {protected}"
                )

    root_fd = case_fd = output_fd = -1
    try:
        root_fd = _open_nofollow_directory(root)
        _require_same_identity(
            os.fstat(root_fd),
            root_lstat,
            label="root",
        )
        case_entry = os.stat(
            case_root.name,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(case_entry.st_mode):
            raise AssertionError(f"unsafe live cleanup case: {case_root}")
        _require_same_identity(case_entry, case_lstat, label="case")
        case_fd = _open_nofollow_directory(
            case_root.name,
            dir_fd=root_fd,
        )
        _require_same_identity(
            os.fstat(case_fd),
            case_lstat,
            label="case",
        )
        output_entry = os.stat(
            output_dir.name,
            dir_fd=case_fd,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(output_entry.st_mode):
            raise AssertionError(f"unsafe live cleanup output: {output_dir}")
        _require_same_identity(output_entry, output_lstat, label="output")
        output_fd = _open_nofollow_directory(
            output_dir.name,
            dir_fd=case_fd,
        )
        _require_same_identity(
            os.fstat(output_fd),
            output_lstat,
            label="output",
        )

        for fixture_name, kind in (
            ("input", "input"),
            ("pipeline.json.pht-pipe", "pipeline"),
        ):
            try:
                fixture_stat = os.stat(
                    fixture_name,
                    dir_fd=case_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(fixture_stat.st_mode):
                raise AssertionError(
                    f"unsafe live cleanup {kind} fixture: "
                    f"{case_root / fixture_name}"
                )

        for protected, protected_lstat in protected_paths:
            try:
                current = os.lstat(protected)
            except OSError as exc:
                raise AssertionError(
                    "protected-path identity disappeared during cleanup "
                    f"preflight: {protected}"
                ) from exc
            _require_same_identity(
                current,
                protected_lstat,
                label=f"protected path {protected}",
            )
            if _identity(current) in {
                _identity(case_lstat),
                _identity(output_lstat),
            }:
                raise AssertionError(
                    f"protected path aliases live cleanup target: {protected}"
                )

        yield AnchoredCleanupTarget(
            root=root,
            case_root=case_root,
            output_dir=output_dir,
            case_generation=case_generation,
            root_fd=root_fd,
            case_fd=case_fd,
            output_fd=output_fd,
            root_identity=_identity(root_lstat),
            case_identity=_identity(case_lstat),
            output_identity=_identity(output_lstat),
            forbidden_identities=tuple(
                (protected, _identity(protected_lstat))
                for protected, protected_lstat in protected_paths
            ),
            scheduler_snapshot=read_scheduler_snapshot(output_fd),
        )
    finally:
        for fd in (output_fd, case_fd, root_fd):
            if fd >= 0:
                os.close(fd)


def cleanup_case(
    case_root: Path,
    output_dir: Path,
    generation: UUID | None,
    initially_known_ids: Iterator[str],
    *,
    forbidden: Sequence[Path],
) -> str:
    """Cancel every role and retain the case after proven quiescence."""
    with anchored_cleanup_target(
        case_root,
        output_dir,
        forbidden,
    ) as target:
        return _cleanup_anchored_case(
            target,
            generation,
            initially_known_ids,
        )


def _cleanup_anchored_case(
    target: AnchoredCleanupTarget,
    generation: UUID | None,
    initially_known_ids: Iterator[str],
) -> str:
    """Clean one case from a no-follow snapshot, then record retained evidence."""
    snapshot = target.scheduler_snapshot
    cleanup_errors: list[str] = []
    job_ids = {str(item) for item in initially_known_ids if str(item).isdigit()}
    roles: Mapping[str, tuple[str, ...]] = {}
    lifecycle = snapshot.lifecycle
    generation, recovery_errors = recover_scheduler_generation(
        snapshot,
        generation,
    )
    cleanup_errors.extend(recovery_errors)
    unresolved_tokens: set[str] = set()
    if generation is not None:
        (
            roles,
            recovered_ids,
            unresolved_tokens,
            job_errors,
        ) = recover_durable_jobs(
            snapshot,
            generation,
        )
        job_ids.update(recovered_ids)
        cleanup_errors.extend(job_errors)
        try:
            comment_matches = query_scheduler_comments(
                prefix=f"phenotypic:{generation.hex}:",
            )
        except SchedulerQueryUnavailable as exc:
            cleanup_errors.append(f"comment discovery unavailable: {exc}")
        else:
            for comment, ids in comment_matches.items():
                job_ids.update(ids)
                unresolved_tokens.discard(comment.rsplit(":", 1)[-1])
        if unresolved_tokens:
            cleanup_errors.append(
                "scheduler submission intents remain unresolved: "
                + ", ".join(sorted(unresolved_tokens))
            )

    scancel_warnings: list[str] = []

    def scancel_ids(ids: set[str]) -> None:
        for job_id in sorted(ids, key=int):
            try:
                completed = subprocess.run(
                    ["scancel", job_id],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                scancel_warnings.append(f"scancel {job_id} unavailable: {exc}")
                continue
            if completed.returncode != 0:
                scancel_warnings.append(
                    f"scancel {job_id} failed: {completed.stderr.strip()}"
                )

    scancel_ids(job_ids)
    active_after: set[str] = set()
    scheduler_detail = ""
    durable_scheduler_files = (
        lifecycle is not None
        or snapshot.metadata is not None
        or bool(snapshot.ledger_rows)
        or bool(snapshot.errors)
    )
    queue_quiescence_proven = (
        generation is None
        and not job_ids
        and not durable_scheduler_files
    )
    if generation is not None:
        deadline = time.monotonic() + CLEANUP_TIMEOUT_SECONDS
        quiet_since: float | None = None
        while time.monotonic() < deadline:
            try:
                active_after = active_generation_comment_ids(generation)
            except SchedulerQueryUnavailable as exc:
                cleanup_errors.append(
                    f"queue cleanup verification unavailable: {exc}"
                )
                break
            if active_after:
                job_ids.update(active_after)
                scancel_ids(active_after)
            states = query_known_job_states(job_ids)
            known_inactive, scheduler_detail = inactive_known_job_ids(
                job_ids,
                states,
            )
            if active_after or not known_inactive:
                quiet_since = None
            else:
                now = time.monotonic()
                if quiet_since is None:
                    quiet_since = now
                elif now - quiet_since >= RECONCILIATION_GRACE_SECONDS:
                    try:
                        final_active = active_generation_comment_ids(
                            generation
                        )
                    except SchedulerQueryUnavailable as exc:
                        cleanup_errors.append(
                            "final queue cleanup verification unavailable: "
                            f"{exc}"
                        )
                        break
                    if final_active:
                        job_ids.update(final_active)
                        scancel_ids(final_active)
                        active_after = final_active
                        quiet_since = None
                        time.sleep(POLL_SECONDS)
                        continue
                    final_states = query_known_job_states(job_ids)
                    final_known_inactive, scheduler_detail = (
                        inactive_known_job_ids(
                            job_ids,
                            final_states,
                        )
                    )
                    if final_known_inactive:
                        active_after = set()
                        queue_quiescence_proven = True
                        break
                    quiet_since = None
            time.sleep(POLL_SECONDS)
        if active_after:
            cleanup_errors.append(
                "generation jobs remained active after cleanup: "
                + ", ".join(sorted(active_after, key=int))
            )
    elif job_ids or lifecycle is not None:
        cleanup_errors.append(
            "scheduler generation is unknown; queue quiescence cannot be proven"
        )

    print(
        "LIVE_SLURM_CLEANUP "
        f"generation={generation.hex if generation else '<unknown>'} "
        f"roles={json.dumps(roles, sort_keys=True)} "
        f"scancelled={','.join(sorted(job_ids, key=int)) or '<none>'} "
        f"active_after={','.join(sorted(active_after, key=int)) or '<none>'} "
        f"scancel_warnings={json.dumps(scancel_warnings)} "
        f"quiescent={queue_quiescence_proven} "
        f"scheduler_detail={scheduler_detail!r}"
    )
    if cleanup_errors or not queue_quiescence_proven:
        print(
            "LIVE_SLURM_MANUAL_CLEANUP_REQUIRED "
            f"case={target.case_root} output={target.output_dir} "
            f"jobs={','.join(sorted(job_ids, key=int)) or '<unknown>'}"
        )
        raise AssertionError(
            "; ".join(cleanup_errors)
            or "scheduler queue quiescence was not proven"
        )
    return write_retained_case_evidence(
        target,
        scheduler_generation=generation,
        job_ids=job_ids,
    )


def _write_cleanup_evidence(
    case_fd: int,
    *,
    evidence_name: str,
    payload: Mapping[str, object],
) -> None:
    """Write one recoverable evidence record through the held case descriptor."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise AssertionError("live SLURM cleanup requires O_NOFOLLOW")
    evidence_fd = os.open(
        evidence_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
        0o600,
        dir_fd=case_fd,
    )
    try:
        encoded = (
            json.dumps(payload, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        written = 0
        while written < len(encoded):
            written += os.write(evidence_fd, encoded[written:])
        os.fsync(evidence_fd)
    finally:
        os.close(evidence_fd)


def _forbidden_identity_evidence(
    forbidden_identities: Sequence[tuple[Path, tuple[int, int]]],
) -> list[dict[str, object]]:
    """Describe inspected protected identities without traversing or moving them."""
    evidence: list[dict[str, object]] = []
    for protected, expected_identity in forbidden_identities:
        try:
            current = os.lstat(protected)
        except OSError:
            evidence.append(
                {
                    "path": str(protected),
                    "inspected_dev": expected_identity[0],
                    "inspected_ino": expected_identity[1],
                    "current_status": "missing",
                }
            )
            continue
        current_identity = _identity(current)
        evidence.append(
            {
                "path": str(protected),
                "inspected_dev": expected_identity[0],
                "inspected_ino": expected_identity[1],
                "current_dev": current_identity[0],
                "current_ino": current_identity[1],
                "current_status": (
                    "unchanged"
                    if current_identity == expected_identity
                    else "changed"
                ),
                "current_is_symlink": stat.S_ISLNK(current.st_mode),
            }
        )
    return evidence


def _inspect_forbidden_paths(
    forbidden: Sequence[Path],
) -> tuple[tuple[Path, tuple[int, int]], ...]:
    """Capture no-follow identities for partial-setup evidence."""
    inspected: list[tuple[Path, tuple[int, int]]] = []
    for protected in forbidden:
        try:
            protected_stat = os.lstat(protected)
        except OSError:
            continue
        inspected.append((protected, _identity(protected_stat)))
    return tuple(inspected)


def _write_open_case_evidence(
    *,
    case_fd: int,
    case_path: Path,
    case_identity: tuple[int, int],
    status: str,
    output_identity: tuple[int, int] | None,
    forbidden_identities: Sequence[tuple[Path, tuple[int, int]]],
    scheduler_generation: UUID | None = None,
    job_ids: Iterable[str] = (),
) -> str:
    """Write cleanup evidence only to the original held case inode."""
    _require_same_identity(
        os.fstat(case_fd),
        case_identity,
        label="open retained case",
    )
    evidence_id = uuid4().hex
    evidence_name = f".live-slurm-cleanup-{evidence_id}.json"
    try:
        _write_cleanup_evidence(
            case_fd,
            evidence_name=evidence_name,
            payload={
                "schema_version": 1,
                "status": status,
                "case_path": str(case_path),
                "case_dev": case_identity[0],
                "case_ino": case_identity[1],
                "output_dev": (
                    output_identity[0] if output_identity is not None else None
                ),
                "output_ino": (
                    output_identity[1] if output_identity is not None else None
                ),
                "scheduler_generation": (
                    scheduler_generation.hex
                    if scheduler_generation is not None
                    else None
                ),
                "scheduler_job_ids": sorted(
                    {str(job_id) for job_id in job_ids},
                    key=int,
                ),
                "forbidden_paths": _forbidden_identity_evidence(
                    forbidden_identities
                ),
                "recorded_at_epoch": time.time(),
            },
        )
    except BaseException:
        print(
            "LIVE_SLURM_MANUAL_CLEANUP_REQUIRED "
            f"case={case_path} dev={case_identity[0]} "
            f"ino={case_identity[1]} evidence_write=failed"
        )
        raise
    print(
        "LIVE_SLURM_RETAINED "
        f"case={case_path} dev={case_identity[0]} ino={case_identity[1]} "
        f"evidence_name={evidence_name}"
    )
    return evidence_name


def write_retained_case_evidence(
    target: AnchoredCleanupTarget,
    *,
    scheduler_generation: UUID | None,
    job_ids: Iterable[str],
) -> str:
    """Record a quiescent live case without changing any pathname."""
    return _write_open_case_evidence(
        case_fd=target.case_fd,
        case_path=target.case_root,
        case_identity=target.case_identity,
        status="retained-after-scheduler-cleanup",
        output_identity=target.output_identity,
        forbidden_identities=target.forbidden_identities,
        scheduler_generation=scheduler_generation,
        job_ids=job_ids,
    )


def validate_retained_case_evidence(
    case_root: Path,
    evidence_name: str,
    *,
    scheduler_generation: UUID | None,
) -> Mapping[str, object]:
    """Validate retained evidence with bounded descriptor-relative reads."""
    if not re.fullmatch(
        r"\.live-slurm-cleanup-[0-9a-f]{32}\.json",
        evidence_name,
    ):
        raise AssertionError(f"invalid cleanup evidence name: {evidence_name}")
    case_stat = _require_nofollow_directory(
        case_root,
        label="retained case",
    )
    try:
        case_fd = _open_nofollow_directory(case_root)
    except OSError as exc:
        raise AssertionError(
            f"retained case cannot be opened safely: {case_root}"
        ) from exc
    try:
        _require_same_identity(
            os.fstat(case_fd),
            case_stat,
            label="retained case",
        )
        raw_payload = _read_optional_regular_file(
            case_fd,
            evidence_name,
            label=f"retained cleanup evidence {evidence_name}",
            max_bytes=MAX_CLEANUP_EVIDENCE_BYTES,
        )
    finally:
        os.close(case_fd)
    if raw_payload is None:
        raise AssertionError(f"cleanup evidence is missing: {evidence_name}")
    try:
        payload = json.loads(raw_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssertionError("cleanup evidence is malformed") from exc
    if not isinstance(payload, dict):
        raise AssertionError("cleanup evidence must be a JSON object")
    if payload.get("status") != "retained-after-scheduler-cleanup":
        raise AssertionError("cleanup evidence has the wrong retained status")
    if (
        payload.get("case_dev") != case_stat.st_dev
        or payload.get("case_ino") != case_stat.st_ino
    ):
        raise AssertionError("cleanup evidence does not match retained case inode")
    expected_generation = (
        scheduler_generation.hex
        if scheduler_generation is not None
        else None
    )
    if payload.get("scheduler_generation") != expected_generation:
        raise AssertionError("cleanup evidence generation does not match")
    return payload


def retain_partial_case(
    case_root: Path,
    *,
    forbidden: Sequence[Path],
) -> str:
    """Retain exact partial setup and record evidence through its case fd."""
    raw_root = os.environ.get(LIVE_ROOT_ENV, "").strip()
    root = Path(raw_root).expanduser()
    match = CASE_NAME_RE.fullmatch(case_root.name)
    if (
        not raw_root
        or not case_root.is_absolute()
        or case_root.parent != root
        or match is None
        or UUID(hex=match.group(1)).hex != match.group(1)
    ):
        raise AssertionError(f"refusing unsafe live cleanup target {case_root}")
    try:
        root_lstat = _require_nofollow_directory(root, label="root")
        case_lstat = _require_nofollow_directory(case_root, label="case")
    except AssertionError as exc:
        raise AssertionError(
            f"refusing unsafe live cleanup target {case_root}"
        ) from exc

    forbidden_identities = _inspect_forbidden_paths(forbidden)
    root_fd = case_fd = -1
    try:
        root_fd = _open_nofollow_directory(root)
        _require_same_identity(
            os.fstat(root_fd),
            root_lstat,
            label="partial cleanup root",
        )
        case_entry = os.stat(
            case_root.name,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
        _require_same_identity(
            case_entry,
            case_lstat,
            label="partial cleanup case",
        )
        case_fd = _open_nofollow_directory(
            case_root.name,
            dir_fd=root_fd,
        )
        _require_same_identity(
            os.fstat(case_fd),
            case_lstat,
            label="open partial cleanup case",
        )
        return _write_open_case_evidence(
            case_fd=case_fd,
            case_path=case_root,
            case_identity=_identity(case_lstat),
            status="retained-pre-submit-failure",
            output_identity=None,
            forbidden_identities=forbidden_identities,
        )
    finally:
        for fd in (case_fd, root_fd):
            if fd >= 0:
                os.close(fd)
