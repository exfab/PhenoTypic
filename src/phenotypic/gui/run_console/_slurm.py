"""Subprocess shell-out around the CLI's SLURM submitter (Phase 6).

The Run console submits SLURM jobs by spawning ``python -m phenotypic
<argv> --slurm k=v ...`` as a subprocess and waiting for it to exit. The
CLI handles all SLURM-specific work: dispatcher chain generation,
``sbatch`` calls, manifest writing. The GUI layer's only job is to capture
the resulting array job ID, which it reads from
``<output_dir>/progress/job_metadata.json::chunk_job_ids`` after the
subprocess returns. **Rich-formatted stdout is intentionally not parsed**:
locale and terminal-width fragility make it unreliable. The metadata file
is the structured contract.

This module has no Dash dependency; importing it from the UI must not
trigger any UI imports either. ``state.to_argv`` (in
:mod:`phenotypic.gui.run_console._state`) is the single source of truth
for argv shape across local + SLURM modes.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from phenotypic.gui.run_console._state import RunConsoleState, to_argv


__all__ = [
    "SlurmSubmitError",
    "SlurmSubmitResult",
    "submit_slurm",
    "wait_for_job_id",
]

logger = logging.getLogger(__name__)


# Keys recognised inside ``RunConsoleState.slurm_args`` that the CLI's
# ``--slurm k=v`` syntax accepts directly. ``cpus_per_task`` is mapped
# below to the CLI's ``cpus_per_task`` SLURM key.
_SLURM_DIRECT_KEYS: tuple[str, ...] = (
    "partition",
    "time",
    "mem",
    "cpus_per_task",
    "gpus",
)


class SlurmSubmitError(RuntimeError):
    """Raised when SLURM submission fails for any reason.

    Surfaced by :func:`submit_slurm` when the CLI subprocess fails, times
    out, or when ``progress/job_metadata.json`` is missing/malformed
    afterwards. Callers should display ``str(err)`` to the user — the
    captured stderr is included in the message body.
    """


@dataclass(frozen=True)
class SlurmSubmitResult:
    """Outcome of a successful SLURM submission shell-out.

    Attributes:
        job_id: Array primary id (e.g. ``"45678901"``). Computed from the
            first value of ``chunk_job_ids``: the suffix after ``"_"`` is
            stripped so callers get the array id, not the array task id.
        output_dir: Absolute path to the run's output directory (mirrors
            ``state.output_dir`` for caller convenience).
        stdout: Captured stdout from the CLI subprocess.
        stderr: Captured stderr from the CLI subprocess.
        returncode: Subprocess exit code (always ``0`` on success;
            non-zero exits raise :class:`SlurmSubmitError` instead of
            being returned).
    """

    job_id: str
    output_dir: Path
    stdout: str
    stderr: str
    returncode: int


# ---------------------------------------------------------------------------
# argv assembly
# ---------------------------------------------------------------------------


def _slurm_argv_extension(slurm_args: dict[str, object]) -> list[str]:
    """Build the ``--slurm k=v`` repeats for ``slurm_args``.

    Args:
        slurm_args: Dict pulled from :class:`RunConsoleState.slurm_args`.
            Recognised top-level keys (``partition``, ``time``, ``mem``,
            ``cpus_per_task``, ``gpus``) are emitted as ``--slurm key=value``
            repeats. The ``extra`` sub-dict (``dict[str, str]``) is merged
            in, also as ``--slurm key=value`` repeats. Unknown top-level
            keys are ignored.

    Returns:
        Flat list of argv tokens, e.g.
        ``["--slurm", "partition=compute", "--slurm", "mem=16G"]``.
    """

    if not slurm_args:
        return []

    pairs: list[tuple[str, str]] = []

    for key in _SLURM_DIRECT_KEYS:
        value = slurm_args.get(key)
        if value is None or value == "":
            continue
        # Map GUI key → CLI ``--slurm`` key. The CLI's k=v parser accepts
        # both ``slurm_partition=...`` and ``partition=...`` shapes; we
        # forward the shorter form so the user sees what they typed.
        pairs.append((key, str(value)))

    extra = slurm_args.get("extra") or {}
    if isinstance(extra, dict):
        for k, v in extra.items():
            if k is None or v is None or str(k) == "" or str(v) == "":
                continue
            pairs.append((str(k), str(v)))

    argv: list[str] = []
    for key, value in pairs:
        argv.extend(["--slurm", f"{key}={value}"])
    return argv


def _build_subprocess_argv(state: RunConsoleState) -> list[str]:
    """Assemble the full argv list for the CLI subprocess.

    Args:
        state: Form state. Must have ``output_dir`` etc. populated; the
            check is performed by :func:`to_argv`.

    Returns:
        Full argv list, ready to pass to :func:`subprocess.run`. Starts
        with ``[sys.executable, "-m", "phenotypic", ...]``.

    Raises:
        ValueError: Propagated from :func:`to_argv` if required state
            fields are missing.
    """

    base = to_argv(state)
    slurm_extra = _slurm_argv_extension(state.slurm_args or {})
    return [sys.executable, "-m", "phenotypic", *base, *slurm_extra]


# ---------------------------------------------------------------------------
# Job-id resolution
# ---------------------------------------------------------------------------


def _read_chunk_job_ids(output_dir: Path) -> dict[str, str]:
    """Read ``chunk_job_ids`` from ``<output_dir>/progress/job_metadata.json``.

    Args:
        output_dir: Run output directory whose ``progress/`` subdir contains
            the structured submission metadata.

    Returns:
        Dict mapping chunk index (string) → array task id (e.g.
        ``"45678901_0"``). Empty dict when the file is missing, unreadable,
        malformed, or contains no ``chunk_job_ids`` entry.
    """

    metadata_path = output_dir / "progress" / "job_metadata.json"
    if not metadata_path.is_file():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw = payload.get("chunk_job_ids")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            out[str(key)] = value
    return out


def _primary_job_id(chunk_job_ids: dict[str, str]) -> str:
    """Resolve the array primary job id from ``chunk_job_ids``.

    Mirrors the helper logic in
    :meth:`phenotypic.gui.shell._runs_registry.RunRegistry._read_status_from_manifest`:
    each value looks like ``"45678901_0"`` (array id + ``_`` + task index);
    splitting on ``"_"`` and taking the first segment yields the primary
    array id used as the user-facing job identifier.

    Args:
        chunk_job_ids: Dict from :func:`_read_chunk_job_ids`.

    Returns:
        The primary array job id (e.g. ``"45678901"``).

    Raises:
        SlurmSubmitError: If ``chunk_job_ids`` is empty or the first value
            is unparseable.
    """

    if not chunk_job_ids:
        raise SlurmSubmitError(
            "SLURM metadata is missing chunk_job_ids; "
            "the CLI did not record any submitted jobs."
        )
    first_value = next(iter(chunk_job_ids.values()))
    if not isinstance(first_value, str) or not first_value:
        raise SlurmSubmitError(
            f"SLURM metadata chunk_job_ids has unexpected first value: "
            f"{first_value!r}"
        )
    return first_value.split("_")[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def submit_slurm(
    state: RunConsoleState,
    *,
    sandbox_root: Path,
    timeout: float = 60.0,
) -> SlurmSubmitResult:
    """Spawn the CLI submitter subprocess and wait for it to exit cleanly.

    Runs ``python -m phenotypic <argv> --slurm k=v ...`` synchronously.
    After the subprocess returns, parses
    ``<output_dir>/progress/job_metadata.json`` for the array job id.
    Rich-formatted stdout is intentionally not parsed (locale / terminal
    width fragile); the metadata file is the structured contract.

    Args:
        state: Form state with ``output_dir``, ``pipeline_path``,
            ``input_dir``, etc. populated. Filesystem-existence checks are
            the caller's responsibility — :func:`to_argv` only enforces
            that the slots are non-empty.
        sandbox_root: Sandbox root, used as the subprocess CWD. The CLI
            resolves relative paths against this directory.
        timeout: How long to wait for the CLI subprocess (default 60s).
            ``sbatch`` can be slow on a busy controller; 60s is generous.

    Raises:
        SlurmSubmitError: If the subprocess fails, times out, or
            ``progress/job_metadata.json`` is missing/malformed.

    Returns:
        :class:`SlurmSubmitResult` with the array primary id, captured
        stdout/stderr, the output directory, and the (zero) returncode.
    """

    if state.output_dir is None:
        raise SlurmSubmitError(
            "RunConsoleState.output_dir is required for SLURM submission."
        )

    try:
        argv = _build_subprocess_argv(state)
    except ValueError as err:
        raise SlurmSubmitError(str(err)) from err

    output_dir = Path(state.output_dir)
    logger.info(
        "submit_slurm: spawning subprocess (cwd=%s, argv=%s)",
        sandbox_root,
        argv,
    )

    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(sandbox_root),
            check=False,
        )
    except subprocess.TimeoutExpired as err:
        # ``stdout``/``stderr`` may be ``None`` on TimeoutExpired; coerce
        # them so the error message is always informative.
        stdout = err.stdout if isinstance(err.stdout, str) else ""
        stderr = err.stderr if isinstance(err.stderr, str) else ""
        raise SlurmSubmitError(
            f"SLURM submission timed out after {timeout:.1f}s. "
            f"stderr:\n{stderr or '<empty>'}\n"
            f"stdout:\n{stdout or '<empty>'}"
        ) from err
    except FileNotFoundError as err:
        # ``python -m phenotypic`` not invocable (e.g. wrong sys.executable).
        raise SlurmSubmitError(
            f"Failed to launch SLURM submitter subprocess: {err}"
        ) from err

    if completed.returncode != 0:
        raise SlurmSubmitError(
            f"SLURM submission subprocess exited with code "
            f"{completed.returncode}. "
            f"stderr:\n{completed.stderr or '<empty>'}\n"
            f"stdout:\n{completed.stdout or '<empty>'}"
        )

    chunk_job_ids = _read_chunk_job_ids(output_dir)
    if not chunk_job_ids:
        raise SlurmSubmitError(
            "SLURM submission subprocess exited cleanly but "
            f"{output_dir / 'progress' / 'job_metadata.json'} is missing or "
            f"empty. stderr:\n{completed.stderr or '<empty>'}"
        )

    job_id = _primary_job_id(chunk_job_ids)

    return SlurmSubmitResult(
        job_id=job_id,
        output_dir=output_dir,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        returncode=completed.returncode,
    )


def wait_for_job_id(
    output_dir: Path,
    *,
    timeout: float = 5.0,
    poll_interval: float = 0.25,
) -> str | None:
    """Poll ``progress/job_metadata.json`` until a job id appears.

    Useful for callers that want to wait on the metadata file without
    blocking the submitter subprocess (e.g. tests, or a UI thread that
    spawned the submitter via a worker pool). On success, returns the same
    primary array id that :func:`submit_slurm` would surface.

    Args:
        output_dir: Run output directory containing ``progress/``.
        timeout: Maximum number of seconds to wait.
        poll_interval: Sleep duration between filesystem checks.

    Returns:
        The primary array job id, or ``None`` if the timeout expires
        before the metadata file appears with a parseable
        ``chunk_job_ids`` entry.
    """

    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        chunk_job_ids = _read_chunk_job_ids(output_dir)
        if chunk_job_ids:
            try:
                return _primary_job_id(chunk_job_ids)
            except SlurmSubmitError:
                return None
        if time.monotonic() >= deadline:
            return None
        time.sleep(max(0.0, poll_interval))
