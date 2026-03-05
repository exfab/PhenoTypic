"""
Structured failure tracking for PhenoTypic CLI processing.

Writes structured JSONL records to ``progress/failures.jsonl`` for both local
and SLURM execution modes.  Each line is a self-contained JSON object that can
be read by the live dashboard or post-hoc analysis tools.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ._cli_file_locking import atomic_append, FileLockTimeout

logger = logging.getLogger(__name__)


def append_failure(
    progress_dir: Path,
    *,
    dataset: str,
    image: str,
    error_type: str,
    error_message: str,
    traceback: str = "",
    slurm_job_id: str = "",
    failure_source: str = "python",
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
    failures_path = progress_dir / "failures.jsonl"

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
        atomic_append(failures_path, line, timeout=10.0)
    except FileLockTimeout:
        # Fallback: best-effort direct append (still safe for single writes
        # on most POSIX systems).
        try:
            with open(failures_path, "a", encoding="utf-8") as f:
                f.write(line)
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
    failures_path = progress_dir / "failures.jsonl"
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
    counts = Counter(f.get("error_type", "Unknown") for f in failures)
    return dict(counts.most_common())
