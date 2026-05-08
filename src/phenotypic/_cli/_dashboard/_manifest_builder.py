"""
Manifest builder for the PhenoTypic CLI live dashboard.

Builds ``progress/manifest.json`` from the append-only event log and failure
records so that the dashboard can display up-to-date processing status.
Also provides ``sacct`` integration for detecting OOM kills and other
silent SLURM failures that would otherwise go unreported.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .._cli_failure_tracker import append_failure, categorize_failures, read_failures
from .._cli_update_state import aggregate_state_from_events, append_event
from phenotypic.tools_ import (
    PROCESSING_EVENTS_LOG,
    MANIFEST_JSON,
    DashboardManifestKey,
    DashboardManifestSlurmInfoKey,
    analysis_scatter_json_path,
    analysis_full_parquet_path,
    chunks_dir,
    master_measurements_parquet_path,
)
from phenotypic.tools_.typing_ import ExecutionMode

logger = logging.getLogger(__name__)

# SLURM states that indicate a terminal (finished) job.
_TERMINAL_STATES = frozenset({
    "COMPLETED",
    "FAILED",
    "OUT_OF_MEMORY",
    "TIMEOUT",
    "CANCELLED",
    "NODE_FAIL",
    "PREEMPTED",
})

# SLURM states indicating a failure of some kind.
_FAILURE_STATES = frozenset({
    "FAILED",
    "OUT_OF_MEMORY",
    "TIMEOUT",
    "CANCELLED",
    "NODE_FAIL",
    "PREEMPTED",
})

# Module-level cache for jobs that have reached a terminal state.
# Once a job is COMPLETED/FAILED/etc. it will never change, so we avoid
# re-querying sacct for it on subsequent manifest builds.
_terminal_job_cache: Dict[str, Dict[str, str]] = {}


# ---------------------------------------------------------------------------
# sacct helpers
# ---------------------------------------------------------------------------


def query_sacct_job_states(job_id: str) -> Optional[Dict[str, str]]:
    """Query sacct for array task states.

    Runs ``sacct`` with parsable output and returns a mapping from
    ``JobID`` to ``State`` for every array task in the given job.
    Sub-step entries (e.g. ``12345_0.batch``) are skipped.

    Args:
        job_id: The SLURM job ID to query (may be an array job ID).

    Returns:
        Mapping of ``"jobid_taskid"`` to SLURM state string, or ``None``
        if ``sacct`` is unavailable or the command fails.
    """
    try:
        result = subprocess.run(
            [
                "sacct",
                "-j", str(job_id),
                "--noheader",
                "--parsable2",
                "--format=JobID,State,ExitCode,MaxRSS",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        logger.debug("sacct not found on this system")
        return None
    except PermissionError:
        logger.debug("sacct: permission denied")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("sacct timed out after 30s for job %s", job_id)
        return None

    if result.returncode != 0:
        logger.debug(
            "sacct returned non-zero exit code %d for job %s: %s",
            result.returncode,
            job_id,
            result.stderr.strip(),
        )
        return None

    states: Dict[str, str] = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue

        raw_job_id = parts[0]
        state = parts[1]

        # Skip sub-step lines like "12345_0.batch"
        if "." in raw_job_id:
            continue

        # Normalise "12345_0" -> "12345_0", plain "12345" stays as-is.
        states[raw_job_id] = state

    return states


def query_sacct_batch(
    job_ids: List[str],
) -> Optional[Dict[str, Dict[str, str]]]:
    """Query sacct for multiple jobs in a single call.

    Args:
        job_ids: SLURM job IDs to query.

    Returns:
        Mapping of job ID to a dict of ``"jobid_taskid"`` -> SLURM state
        string for that job, or ``None`` if ``sacct`` is unavailable or
        the command fails.
    """
    if not job_ids:
        return {}

    try:
        result = subprocess.run(
            [
                "sacct",
                "-j", ",".join(str(jid) for jid in job_ids),
                "--noheader",
                "--parsable2",
                "--format=JobID,State,ExitCode,MaxRSS",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        logger.debug("sacct not found on this system")
        return None
    except PermissionError:
        logger.debug("sacct: permission denied")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("sacct timed out after 30s for batch query (%d jobs)", len(job_ids))
        return None

    if result.returncode != 0:
        logger.debug(
            "sacct returned non-zero exit code %d for batch query: %s",
            result.returncode,
            result.stderr.strip(),
        )
        return None

    per_job: Dict[str, Dict[str, str]] = {jid: {} for jid in job_ids}
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue

        raw_job_id = parts[0]
        state = parts[1]

        if "." in raw_job_id:
            continue

        # Determine which parent job this line belongs to.
        # raw_job_id is either "12345" or "12345_0"; the parent is the
        # portion before the underscore (or the whole string).
        parent = raw_job_id.split("_")[0]
        if parent in per_job:
            per_job[parent][raw_job_id] = state

    return per_job


def query_sacct_chunk_states(
    chunk_job_ids: Dict[str, str],
) -> Tuple[List[int], List[int], List[int]]:
    """Determine active, completed, and pending chunks from sacct.

    For each chunk job ID, queries ``sacct`` to determine the aggregate
    state of all its array tasks.

    Args:
        chunk_job_ids: Mapping of chunk index (as string) to SLURM job ID.

    Returns:
        Tuple of ``(active_chunks, completed_chunks, pending_chunks)`` as
        lists of chunk indices (integers).
    """
    active: List[int] = []
    completed: List[int] = []
    pending: List[int] = []

    # Separate cached (terminal) jobs from those needing a fresh query.
    uncached_job_ids: Dict[str, str] = {}
    cached_results: Dict[str, Dict[str, str]] = {}

    for chunk_idx_str, job_id in chunk_job_ids.items():
        if job_id in _terminal_job_cache:
            cached_results[job_id] = _terminal_job_cache[job_id]
        else:
            uncached_job_ids[chunk_idx_str] = job_id

    # Batch-query all uncached jobs in a single sacct call.
    fresh_results: Dict[str, Dict[str, str]] = {}
    if uncached_job_ids:
        unique_ids = list(dict.fromkeys(uncached_job_ids.values()))
        batch = query_sacct_batch(unique_ids)
        if batch is None:
            # sacct unavailable -- fall back to cached results only
            pass
        else:
            fresh_results = batch
            # Cache any jobs that have reached a terminal state.
            for job_id, task_states in fresh_results.items():
                if task_states:
                    state_values = set(task_states.values())
                    if state_values <= _TERMINAL_STATES:
                        _terminal_job_cache[job_id] = task_states

    # Merge cached and fresh results, then classify each chunk.
    all_results: Dict[str, Dict[str, str]] = {**fresh_results, **cached_results}

    for chunk_idx_str, job_id in chunk_job_ids.items():
        chunk_idx = int(chunk_idx_str)
        task_states = all_results.get(job_id)

        if task_states is None:
            continue

        if not task_states:
            pending.append(chunk_idx)
            continue

        state_values = set(task_states.values())

        if "RUNNING" in state_values:
            active.append(chunk_idx)
        elif state_values <= _TERMINAL_STATES:
            completed.append(chunk_idx)
        elif state_values == {"PENDING"}:
            pending.append(chunk_idx)
        elif "PENDING" in state_values and state_values <= (_TERMINAL_STATES | {"PENDING"}):
            active.append(chunk_idx)
        else:
            active.append(chunk_idx)

    active.sort()
    completed.sort()
    pending.sort()
    return active, completed, pending


# ---------------------------------------------------------------------------
# Silent-failure detection
# ---------------------------------------------------------------------------


def detect_silent_failures(
    output_dir: Path,
    progress_dir: Path,
    slurm_job_ids: Dict[str, str],
    image_task_mapping: Dict[str, List[str]],
) -> List[dict]:
    """Detect images killed silently by SLURM (OOM, timeout, etc.).

    Cross-references images that were started but never completed/failed
    against ``sacct`` job states.  When the associated SLURM task is in a
    terminal failure state, a synthetic ``"failed"`` event is appended to
    the event log and a failure record is written to ``failures.jsonl``.

    Args:
        output_dir: Root output directory (contains ``processing_events.log``).
        progress_dir: Progress directory (contains ``failures.jsonl``).
        slurm_job_ids: Mapping of ``"chunk_index"`` to SLURM job ID.
        image_task_mapping: Mapping of ``"jobid_taskid"`` to
            ``[dataset, image_filename]``.

    Returns:
        List of failure record dicts for newly detected silent failures.
        Returns an empty list if ``sacct`` is unavailable.
    """
    event_log = output_dir / PROCESSING_EVENTS_LOG
    dataset_states = aggregate_state_from_events(event_log)

    # Collect all in-progress images across datasets.
    in_progress_images: Dict[str, List[str]] = {}  # dataset -> [images]
    for ds_name, ds_state in dataset_states.items():
        ip = ds_state.in_progress
        if ip:
            in_progress_images[ds_name] = list(ip)

    if not in_progress_images:
        return []

    # Build a reverse lookup: (dataset, image) -> job_task_id
    image_to_task: Dict[Tuple[str, str], str] = {}
    for task_id, mapping in image_task_mapping.items():
        if len(mapping) >= 2:
            image_to_task[(mapping[0], mapping[1])] = task_id

    # Query sacct for all relevant job IDs.
    all_task_states: Dict[str, str] = {}
    queried_jobs: set = set()
    for job_id in slurm_job_ids.values():
        if job_id in queried_jobs:
            continue
        queried_jobs.add(job_id)
        states = query_sacct_job_states(job_id)
        if states is None:
            # sacct unavailable
            return []
        all_task_states.update(states)

    detected_failures: List[dict] = []

    for ds_name, images in in_progress_images.items():
        for image in images:
            task_id = image_to_task.get((ds_name, image))
            if task_id is None:
                continue

            state = all_task_states.get(task_id)
            if state is None:
                continue

            if state in ("RUNNING", "PENDING"):
                # Still in progress -- not a silent failure.
                continue

            if state not in _FAILURE_STATES:
                # COMPLETED but we never saw a completion event -- unusual
                # but not necessarily a silent failure we can categorise.
                continue

            # Determine error type from SLURM state.
            error_type = state  # e.g. "OUT_OF_MEMORY", "TIMEOUT", etc.
            error_message = (
                f"SLURM task {task_id} terminated with state {state}"
            )

            # Write a synthetic "failed" event to the event log.
            append_event(
                event_log,
                dataset=ds_name,
                image=image,
                status="failed",
                error_msg=error_message,
            )

            # Write a failure record to failures.jsonl.
            append_failure(
                progress_dir,
                dataset=ds_name,
                image=image,
                error_type=error_type,
                error_message=error_message,
                slurm_job_id=task_id,
                failure_source="slurm",
            )

            record = {
                "dataset": ds_name,
                "image": image,
                "error_type": error_type,
                "error_message": error_message,
                "slurm_task_id": task_id,
            }
            detected_failures.append(record)
            logger.warning(
                "Detected silent SLURM failure: %s/%s (%s via task %s)",
                ds_name,
                image,
                error_type,
                task_id,
            )

    return detected_failures


def _get_analysis_data_version(progress_dir: Path) -> int:
    """Return the maximum modification time across analysis data sources.

    Checks ``analysis_scatter.json``, ``analysis_full.parquet``,
    the ``progress/chunks/`` directory, and ``master_measurements.parquet``,
    returning the most recent mtime so the dashboard detects when new
    data is available.
    """
    candidates = [
        analysis_scatter_json_path(progress_dir),
        analysis_full_parquet_path(progress_dir),
        chunks_dir(progress_dir),
        master_measurements_parquet_path(progress_dir.parent),
    ]
    max_mtime = 0
    for path in candidates:
        try:
            max_mtime = max(max_mtime, int(path.stat().st_mtime))
        except OSError:
            continue
    return max_mtime


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def build_manifest(
    output_dir: Path,
    progress_dir: Path,
    datasets: Dict[str, int],
    execution_mode: ExecutionMode,
    start_time: str,
    slurm_job_ids: Optional[Dict[str, str]] = None,
    chunk_scripts: Optional[List[str]] = None,
    input_path: Optional[str] = None,
) -> None:
    """Build ``progress/manifest.json`` for the live dashboard.

    Aggregates processing state from the event log, failure records, and
    (when running under SLURM) ``sacct`` job states into a single JSON
    file that the dashboard reads.

    Args:
        output_dir: Root output directory containing
            ``processing_events.log``.
        progress_dir: Progress directory where ``manifest.json`` and
            ``failures.jsonl`` reside.
        datasets: Mapping of dataset name to total image count.
        execution_mode: ``"slurm"`` or ``"local"``.
        start_time: ISO-formatted start timestamp.
        slurm_job_ids: Mapping of chunk index (as string) to SLURM job ID.
            Required for SLURM chunk-state queries and silent-failure
            detection.  Ignored when *execution_mode* is ``"local"``.
        chunk_scripts: List of generated SLURM chunk script paths.
        input_path: Display name for the input (e.g. folder stem or
            image filename).  Stored in the manifest so the dashboard
            can show which input is being processed.
    """
    event_log = output_dir / PROCESSING_EVENTS_LOG

    # 1. Aggregate state from the event log.
    dataset_states = aggregate_state_from_events(event_log)

    # 2. Read and categorise failures.
    failures = read_failures(progress_dir)
    failure_categories = categorize_failures(failures)

    # 3-4. SLURM-specific queries.
    is_slurm = execution_mode == "slurm"
    active_chunks: List[int] = []
    completed_chunks: List[int] = []
    pending_chunks: List[int] = []

    if is_slurm and slurm_job_ids:
        # Early exit: skip sacct entirely when event log already shows all
        # images have reached a terminal state (completed or failed).
        total_images_for_check = sum(datasets.values())
        event_completed = sum(
            len(ds.completed) for ds in dataset_states.values()
        )
        event_failed = sum(
            len(ds.failed) for ds in dataset_states.values()
        )
        if (event_completed + event_failed) >= total_images_for_check:
            completed_chunks = sorted(int(k) for k in slurm_job_ids)
        else:
            active_chunks, completed_chunks, pending_chunks = (
                query_sacct_chunk_states(slurm_job_ids)
            )

        # Detect OOM / silent failures (needs image_task_mapping -- build
        # from event log SLURM fields if possible).  The caller may not
        # always provide a mapping; in that case we skip detection.
        # detect_silent_failures is called by the orchestrator which has
        # the mapping available; here we only do chunk-state queries.

    # 5. Build the manifest dict.
    total_images = sum(datasets.values())
    global_completed = 0
    global_failed = 0
    global_in_progress = 0

    per_dataset: Dict[str, dict] = {}
    for ds_name, ds_total in datasets.items():
        ds_state = dataset_states.get(ds_name)
        if ds_state is None:
            per_dataset[ds_name] = {
                "total": ds_total,
                "completed": 0,
                "failed": 0,
                "started": 0,
                "pending": ds_total,
            }
            global_in_progress += 0
        else:
            ds_completed = len(ds_state.completed)
            ds_failed = len(ds_state.failed)
            ds_in_progress = len(ds_state.in_progress)
            ds_pending = ds_total - ds_completed - ds_failed - ds_in_progress

            per_dataset[ds_name] = {
                "total": ds_total,
                "completed": ds_completed,
                "failed": ds_failed,
                "started": ds_in_progress,
                "pending": max(ds_pending, 0),
            }
            global_completed += ds_completed
            global_failed += ds_failed
            global_in_progress += ds_in_progress

    global_pending = total_images - global_completed - global_failed - global_in_progress

    if (global_completed + global_failed) > 0:
        success_rate = global_completed / (global_completed + global_failed)
    else:
        success_rate = 0.0

    is_complete = (global_completed + global_failed) == total_images

    manifest: dict = {
        DashboardManifestKey.VERSION: 1,
        DashboardManifestKey.LAST_UPDATED: datetime.now().isoformat(timespec="milliseconds"),
        DashboardManifestKey.EXECUTION_MODE: execution_mode,
        DashboardManifestKey.TOTAL_IMAGES: total_images,
        DashboardManifestKey.COMPLETED: global_completed,
        DashboardManifestKey.FAILED: global_failed,
        DashboardManifestKey.STARTED: global_in_progress,
        DashboardManifestKey.PENDING: max(global_pending, 0),
        DashboardManifestKey.SUCCESS_RATE: round(success_rate, 6),
        DashboardManifestKey.IS_COMPLETE: is_complete,
        DashboardManifestKey.START_TIME: start_time,
        DashboardManifestKey.INPUT_PATH: input_path,
        DashboardManifestKey.DATASETS: per_dataset,
        DashboardManifestKey.FAILURE_CATEGORIES: failure_categories,
        DashboardManifestKey.ANALYSIS_DATA_VERSION: _get_analysis_data_version(progress_dir),
    }

    # Add SLURM info when in SLURM mode.
    if is_slurm:
        manifest[DashboardManifestKey.SLURM_INFO] = {
            DashboardManifestSlurmInfoKey.CHUNK_SCRIPTS: chunk_scripts or [],
            DashboardManifestSlurmInfoKey.TOTAL_CHUNKS: len(slurm_job_ids) if slurm_job_ids else 0,
            DashboardManifestSlurmInfoKey.CHUNK_JOB_IDS: slurm_job_ids or {},
            DashboardManifestSlurmInfoKey.ACTIVE_CHUNKS: active_chunks,
            DashboardManifestSlurmInfoKey.COMPLETED_CHUNKS: completed_chunks,
            DashboardManifestSlurmInfoKey.PENDING_CHUNKS: pending_chunks,
        }

    # 6. Write manifest atomically.
    progress_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = progress_dir / MANIFEST_JSON

    fd = tempfile.NamedTemporaryFile(
        mode="w",
        dir=progress_dir,
        prefix=".manifest_",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    )
    try:
        json.dump(manifest, fd, indent=2, ensure_ascii=False)
        fd.write("\n")
        fd.flush()
        os.fsync(fd.fileno())
        fd.close()
        os.replace(fd.name, manifest_path)
    except BaseException:
        fd.close()
        # Clean up temp file on failure.
        try:
            os.unlink(fd.name)
        except OSError:
            pass
        raise

    logger.debug(
        "Wrote manifest: %d/%d completed, %d failed, %d pending",
        global_completed,
        total_images,
        global_failed,
        max(global_pending, 0),
    )
