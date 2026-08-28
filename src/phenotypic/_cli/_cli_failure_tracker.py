"""
Structured failure tracking for PhenoTypic CLI processing.

Writes structured JSONL records to ``progress/failures.jsonl`` for both local
and SLURM execution modes.  Each line is a self-contained JSON object that can
be read by the live dashboard or post-hoc analysis tools.
"""

from __future__ import annotations

import json
import hashlib
import logging
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, TYPE_CHECKING

from ._cli_file_locking import atomic_append, atomic_read, FileLockTimeout
from phenotypic.sdk_ import (
    CommitGuard,
    FAILURES_JSONL,
    STORE_SUFFIX,
    publication_commit,
    terminal_failures_jsonl_path,
)
from phenotypic.sdk_.typing_ import FailureSource

if TYPE_CHECKING:
    from ._cli_types import ExecutionConfig

logger = logging.getLogger(__name__)

TERMINAL_FAILURE_SCHEMA_VERSION = 1
WORK_ID_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TerminalFailureRecord:
    """Durable terminal outcome for one exact image computation."""

    version: int
    work_id: str
    dataset: str
    relative_image_path: str
    failed_stage: str
    exception_type: str
    exception_message: str
    attempt_id: str
    lifecycle_epoch: str
    timestamp: str
    traceback: str = ""
    slurm_job_id: str = ""


class PerImageScientificError(Exception):
    """Wrap an exception raised inside a per-image scientific boundary."""

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


class TerminalFailureJournalError(RuntimeError):
    """Raised when authoritative terminal-failure state cannot be read."""


def is_terminal_scientific_exception(exception: Exception) -> bool:
    """Return whether a caught scientific exception may be terminal."""
    if isinstance(exception, (MemoryError, TimeoutError)):
        return False
    torch_module = sys.modules.get("torch")
    cuda_module = getattr(torch_module, "cuda", None)
    oom_type = getattr(cuda_module, "OutOfMemoryError", None)
    return not (
        isinstance(oom_type, type) and isinstance(exception, oom_type)
    )


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of *path* without retaining file contents.

    A ``*.ome.zarr`` input is a **directory**, so the streaming read raises
    ``IsADirectoryError``. A store is digested over its whole tree: every
    member's store-relative path and content, in sorted path order.

    **Not the root ``zarr.json`` alone.** An earlier version did that, on the
    reasoning that the promote protocol writes the root last so it fingerprints
    completeness, and that it "changes whenever any published content does".
    The first half is true; the second is false, and verified so by execution.
    The root carries the schema version, the series map, the pyramid geometry,
    the metadata sections and the provenance journal -- none of which move when
    pixels do. Two stores whose images differ entirely produced one digest::

        pixels genuinely differ : True (mean 0.640 vs 0.500)
        shard bytes differ      : True
        root zarr.json identical: True
        file_sha256 differs     : False

    That silently breaks content-change detection for a store input: the
    digest feeds :func:`work_id_for_image` and the SLURM identity ledger, so
    an edited store would keep its work ID and continuation would reuse stale
    output. The flat-file path digests every pixel byte; the store path must
    not be weaker.

    The walk is also not the cost the earlier reasoning assumed. A store holds
    roughly a dozen files whose bytes are the same bytes an equivalent TIFF
    would carry, so digesting the tree reads about as much as digesting that
    TIFF -- plus a directory walk, which is what buys the guarantee.

    Paths are folded in alongside content so that moving a chunk between
    members, or renaming one, changes the digest. Sorting makes it independent
    of filesystem iteration order.

    A directory that is not a store still raises ``IsADirectoryError``. It has
    no meaningful content fingerprint, and inventing one would let a
    mis-specified ``--input`` produce a stable work ID for something that is
    not an image.

    Args:
        path: An input image file, or a ``*.ome.zarr`` store directory.

    Returns:
        The hex digest.

    Raises:
        IsADirectoryError: If *path* is a directory that is not an OME-Zarr
            store.
    """
    target = Path(path)
    digest = hashlib.sha256()

    if target.is_dir():
        if not target.name.endswith(STORE_SUFFIX):
            raise IsADirectoryError(
                f"{target} is a directory but not an OME-Zarr store; "
                f"it has no content fingerprint"
            )
        members = sorted(
            (p for p in target.rglob("*") if p.is_file()),
            key=lambda p: p.relative_to(target).as_posix(),
        )
        for member in members:
            digest.update(member.relative_to(target).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with member.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def processing_configuration_digest_from_values(
    *,
    image_type: str,
    nrows: int | None,
    ncols: int | None,
    bit_depth: int | None,
    detect_mode: str,
    process_only_layer: str | None,
    ext: str,
    include_dataset_column: bool,
    overlay_alpha: float,
    save_overlays: bool,
    drop_originals: bool = False,
) -> str:
    """Hash explicit settings that affect one image's scientific outputs."""
    payload: dict[str, object] = {
        "image_type": image_type,
        "nrows": nrows,
        "ncols": ncols,
        "bit_depth": bit_depth,
        "detect_mode": detect_mode,
        "drop_originals": drop_originals,
    }
    if process_only_layer is not None:
        payload.update(
            {
                "process_only_layer": process_only_layer,
                "ext": ext,
            }
        )
    else:
        payload.update(
            {
                "include_dataset_column": include_dataset_column,
                "overlay_alpha": overlay_alpha,
                "save_overlays": save_overlays,
            }
        )
    return _canonical_digest(payload)


def processing_configuration_digest(config: "ExecutionConfig") -> str:
    """Hash settings that affect per-image science or required artifacts."""
    return processing_configuration_digest_from_values(
        image_type=config.image_type,
        nrows=config.nrows,
        ncols=config.ncols,
        bit_depth=config.bit_depth,
        detect_mode=config.detect_mode,
        process_only_layer=config.process_only_layer,
        ext=config.ext,
        include_dataset_column=config.include_dataset_column,
        overlay_alpha=config.overlay_alpha,
        save_overlays=config.save_overlays,
        drop_originals=config.drop_originals,
    )


def compute_work_id(
    *,
    dataset: str,
    relative_image_path: str,
    input_sha256: str,
    pipeline_fingerprint: str,
    processing_config_digest: str,
    mode: str,
) -> str:
    """Return the stable identity of one exact per-image computation."""
    return _canonical_digest(
        {
            "schema_version": WORK_ID_SCHEMA_VERSION,
            "dataset": dataset,
            "relative_image_path": Path(relative_image_path).as_posix(),
            "input_sha256": input_sha256,
            "pipeline_fingerprint": pipeline_fingerprint,
            "processing_configuration_digest": processing_config_digest,
            "mode": mode,
        }
    )


def work_id_for_image(
    config: "ExecutionConfig", dataset: str, image_path: Path
) -> tuple[str, str]:
    """Return ``(work_id, normalized_relative_path)`` for an input image.

    A ``*.ome.zarr`` store is a directory, so ``--input`` naming one directly
    does not take the ``is_file`` branch. It falls through to ``relative_to``,
    which yields ``Path(".")`` when the two paths are the same -- see the
    degenerate-path recovery below.
    """
    if config.input_path.is_file():
        relative_path = Path(image_path.name)
    else:
        try:
            relative_path = image_path.relative_to(config.input_path)
        except ValueError:
            relative_path = Path(image_path.name)
        if relative_path == Path("."):
            # `--input` names the image itself, so `relative_to` yields `.`
            # and every such input shares one relative path. Two stores with
            # identical content under one dataset would then produce the same
            # work ID -- the same collapse `process_only_output_path` guards
            # against, and the same recovery. Pre-existing on the flat-file
            # path; fixed here because a single store input is exactly what
            # spec 7 makes routine.
            relative_path = Path(image_path.name)
    mode = (
        "measure"
        if config.measure_only
        else "process"
        if config.process_only_layer is not None
        else "full"
    )
    pipeline_fingerprint = file_sha256(config.pipeline_json)
    return (
        compute_work_id(
            dataset=dataset,
            relative_image_path=relative_path.as_posix(),
            input_sha256=file_sha256(image_path),
            pipeline_fingerprint=pipeline_fingerprint,
            processing_config_digest=processing_configuration_digest(config),
            mode=mode,
        ),
        relative_path.as_posix(),
    )


def append_terminal_failure(
    output_dir: Path,
    *,
    work_id: str,
    dataset: str,
    relative_image_path: str,
    failed_stage: str,
    exception: Exception,
    attempt_id: str,
    lifecycle_epoch: str,
    traceback: str = "",
    slurm_job_id: str = "",
    commit_guard: CommitGuard | None = None,
) -> bool:
    """Durably append one terminal failure without an unlocked fallback.

    Returns:
        ``True`` only after the complete JSON line has been flushed and
        ``fsync``ed. A false return leaves the image pending.
    """
    if not is_terminal_scientific_exception(exception):
        return False
    from ._cli_completion import valid_image_success

    if valid_image_success(
        output_dir,
        dataset=dataset,
        image_stem=Path(relative_image_path).stem,
        work_id=work_id,
    ):
        return False
    record = TerminalFailureRecord(
        version=TERMINAL_FAILURE_SCHEMA_VERSION,
        work_id=work_id,
        dataset=dataset,
        relative_image_path=Path(relative_image_path).as_posix(),
        failed_stage=failed_stage,
        exception_type=type(exception).__name__,
        exception_message=str(exception),
        attempt_id=attempt_id,
        lifecycle_epoch=lifecycle_epoch,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        traceback=traceback,
        slurm_job_id=slurm_job_id,
    )
    line = json.dumps(
        asdict(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ) + "\n"
    try:
        atomic_append(
            terminal_failures_jsonl_path(output_dir),
            line,
            timeout=60.0,
            durable=True,
            repair_incomplete_line=True,
            commit_guard=commit_guard,
        )
    except (FileLockTimeout, OSError, ValueError):
        logger.error("Failed to commit terminal failure", exc_info=True)
        return False
    return True


def read_terminal_failures(output_dir: Path) -> List[TerminalFailureRecord]:
    """Read valid terminal records, ignoring malformed or incomplete lines."""
    path = terminal_failures_jsonl_path(output_dir)

    def _parse(content: str) -> List[TerminalFailureRecord]:
        records: List[TerminalFailureRecord] = []
        for line_number, raw_line in enumerate(content.splitlines(), 1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
                if not isinstance(row, dict):
                    raise TypeError("terminal record is not an object")
                normalized = {
                    field: row[field]
                    for field in (
                        "work_id",
                        "dataset",
                        "relative_image_path",
                        "failed_stage",
                        "exception_type",
                        "exception_message",
                        "attempt_id",
                        "lifecycle_epoch",
                        "timestamp",
                    )
                }
                normalized["version"] = row.get(
                    "version", row.get("schema_version")
                )
                normalized["traceback"] = row.get("traceback", "")
                normalized["slurm_job_id"] = row.get("slurm_job_id", "")
                record = TerminalFailureRecord(**normalized)
            except (json.JSONDecodeError, TypeError, KeyError):
                logger.warning(
                    "Ignoring malformed terminal-failure line %d in %s",
                    line_number,
                    path,
                )
                continue
            if record.version != TERMINAL_FAILURE_SCHEMA_VERSION:
                logger.warning(
                    "Ignoring unsupported terminal-failure schema %s in %s",
                    record.version,
                    path,
                )
                continue
            records.append(record)
        return records

    try:
        return atomic_read(path, _parse, timeout=60.0)
    except (FileLockTimeout, OSError) as exc:
        raise TerminalFailureJournalError(
            f"Cannot read authoritative terminal-failure journal: {path}"
        ) from exc


def terminal_failure_index(
    output_dir: Path,
) -> Dict[str, TerminalFailureRecord]:
    """Return the latest valid terminal record for each ``work_id``."""
    return {record.work_id: record for record in read_terminal_failures(output_dir)}


def migrate_legacy_terminal_failures(
    output_dir: Path,
    *,
    valid_work_ids: set[str],
) -> int:
    """Durably import only fully identified legacy scientific failures.

    Ambiguous dashboard rows, scheduler-derived status, and the old
    ``DatasetState.failed`` projection are deliberately ignored.
    """
    from phenotypic.sdk_ import failures_jsonl_path, progress_dir

    sources = (
        failures_jsonl_path(output_dir),
        progress_dir(output_dir) / "stage2_terminal_failures.jsonl",
    )
    existing = set(terminal_failure_index(output_dir))
    migrated = 0
    excluded_types = {"MemoryError", "OutOfMemoryError", "TimeoutError"}
    excluded_terms = {
        "cancel",
        "preempt",
        "node_loss",
        "node loss",
        "timeout",
        "out_of_memory",
        "oom",
        "lock",
        "fence",
        "staging",
        "artifact",
        "publication",
        "aggregate",
        "finaliz",
        "missing_prereq",
    }
    for source in sources:
        try:
            content = atomic_read(source, lambda raw: raw, timeout=60.0)
        except (FileLockTimeout, OSError):
            continue
        for raw_line in content.splitlines():
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            work_id = row.get("work_id")
            classification = row.get(
                "failure_classification", row.get("classification")
            )
            exception_type = row.get("exception_type", row.get("error_type"))
            trusted_boundary = bool(
                row.get("caught_per_image_exception") is True
                or row.get("failure_boundary") == "per_image_scientific"
            )
            diagnostic_text = " ".join(
                str(row.get(key, "")).lower()
                for key in (
                    "failed_stage",
                    "exception_type",
                    "exception_message",
                    "error_message",
                    "failure_source",
                )
            )
            if (
                not isinstance(work_id, str)
                or work_id not in valid_work_ids
                or work_id in existing
                or classification not in {"per_image_scientific", "scientific"}
                or exception_type in excluded_types
                or not trusted_boundary
                or any(term in diagnostic_text for term in excluded_terms)
            ):
                continue
            required = (
                "dataset",
                "relative_image_path",
                "failed_stage",
                "attempt_id",
                "lifecycle_epoch",
                "timestamp",
            )
            if any(not isinstance(row.get(key), str) for key in required):
                continue
            message = row.get("exception_message", row.get("error_message"))
            if not isinstance(message, str) or not isinstance(exception_type, str):
                continue
            exception_class = type(exception_type, (Exception,), {})
            if append_terminal_failure(
                output_dir,
                work_id=work_id,
                dataset=row["dataset"],
                relative_image_path=row["relative_image_path"],
                failed_stage=row["failed_stage"],
                exception=exception_class(message),
                attempt_id=row["attempt_id"],
                lifecycle_epoch=row["lifecycle_epoch"],
                traceback=str(row.get("traceback", "")),
                slurm_job_id=str(row.get("slurm_job_id", "")),
            ):
                existing.add(work_id)
                migrated += 1
    return migrated


def append_failure(
    progress_dir: Path,
    *,
    dataset: str,
    image: str,
    error_type: str,
    error_message: str,
    traceback: str = "",
    slurm_job_id: str = "",
    failure_source: FailureSource = "python",
    commit_guard: CommitGuard | None = None,
) -> None:
    """
    Atomically append a structured failure record to ``failures.jsonl``.

    Args:
        progress_dir: Directory containing ``failures.jsonl``.
        dataset: Dataset name.
        image: Image filename.
        error_type: Exception class name or SLURM failure category
            (e.g. ``"OUT_OF_MEMORY"``).
        error_message: Human-readable error summary.
        traceback: Full Python traceback string (empty for SLURM-detected failures).
        slurm_job_id: SLURM job ID including array index (e.g. ``"12345_42"``).
        failure_source: ``"python"`` for caught exceptions, ``"slurm"`` for
            failures detected via ``sacct``.
    """
    progress_dir.mkdir(parents=True, exist_ok=True)
    failures_path = progress_dir / FAILURES_JSONL

    record: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "dataset": dataset,
        "image": image,
        "error_type": error_type,
        "error_message": error_message,
        "traceback": traceback,
        "slurm_job_id": slurm_job_id,
        "failure_source": failure_source,
    }

    line = json.dumps(record, ensure_ascii=False) + "\n"

    # Use file-locked atomic append for thread/process safety, consistent
    # with how append_event() works for the event log.
    try:
        atomic_append(
            failures_path, line, timeout=10.0, commit_guard=commit_guard
        )
    except FileLockTimeout:
        # Fallback: best-effort direct append (still safe for single writes
        # on most POSIX systems).
        try:
            with publication_commit(commit_guard):
                with open(failures_path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
        except OSError as exc:
            logger.error("Failed to write failure record: %s", exc)


def read_failures(progress_dir: Path) -> List[Dict[str, Any]]:
    """
    Read all failure records from ``failures.jsonl``.

    Args:
        progress_dir: Directory containing ``failures.jsonl``.

    Returns:
        List of failure record dicts, one per JSONL line.  Malformed lines
        are skipped with a warning.
    """
    failures_path = progress_dir / FAILURES_JSONL
    if not failures_path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with open(failures_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed JSONL line %d in %s", line_num, failures_path
                )
    return records


def categorize_failures(failures: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Group failures by ``error_type`` and return counts.

    Args:
        failures: List of failure record dicts (as returned by :func:`read_failures`).

    Returns:
        Mapping of error type to occurrence count, sorted descending by count.
    """
    counts = Counter(
        f.get("error_type") or f.get("exception_type") or "Unknown"
        for f in failures
    )
    return dict(counts.most_common())
