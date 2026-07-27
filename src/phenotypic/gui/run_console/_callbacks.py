"""Run console Dash callbacks.

Wires the Run console's UI affordances into the LocalRunner / SLURM
submitter / RunRegistry plumbing. Validate and Run intentionally fan into one
generation-allocating callback; all later lifecycle callbacks consume the
resulting run-id + generation receipt.

Wired effects (per ``GUI_SPEC_V1.md`` section 5):

    * Pipeline picker modal — open / close / navigate / file-click.
    * Input picker modal — open / close / navigate / confirm.
    * Output picker modal — open / close / navigate / type / confirm.
    * Form-state sync — every form input writes back to
      :data:`ids.RC_STORE_FORM_STATE` via the
      :class:`RunConsoleState` round-trip.
    * Run (Local) — :class:`LocalRunner.start`, registers a
      :class:`RunRecord`, schedules dashboard polling, enables the log
      interval, lights up the iframe ``src``.
    * Run (SLURM) — :func:`._slurm.submit_slurm`, registers a
      :class:`RunRecord`, sets the iframe ``src`` immediately (the
      submitter writes ``dashboard.html`` up-front).
    * Validate (dry-run) — runs the same Local path with ``--dry-run``;
      log only.
    * Cancel — :class:`LocalRunner.stop`; updates registry status.
    * Save preset — writes ``<root>/.phenotypic-gui/presets/<name>.json``.
    * Load preset — populates the form from a preset file.
    * Recent Runs row click — re-points the iframe ``src`` to the
      clicked run's ``/runs/<rel>/deliverables/dashboard.html``.
    * Log tail — :class:`LocalRunner.snapshot_log` poll on a
      :class:`dcc.Interval`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from uuid import UUID

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import ALL, Input, Output, State, ctx, html, no_update

from phenotypic.gui._config import (
    DASHBOARD_FILENAME,
    DEFAULT_URL_PREFIX,
    DELIVERABLES_DIRNAME,
    IMAGE_EXTS,
    RUN_ACTION_CALLBACK_TIMEOUT_MS,
    RUNS_BLUEPRINT_PREFIX,
    SANDBOX_GUI_DIRNAME,
    SANDBOX_PRESETS_SUBDIR,
    THREAD_NAME_PREFIX,
    join_url_prefix,
)
from phenotypic.gui.run_console import _ids as ids
from phenotypic.gui.run_console._directory_picker import (
    render_output_dir_tree,
)
from phenotypic.gui.run_console._form import (
    render_input_tree,
    render_pipeline_tree,
)
from phenotypic.gui.run_console._layout import render_recents_table
from phenotypic.gui.run_console._recent_runs import scan_recent_runs
from phenotypic.gui.run_console._request_safety import (
    MetadataPreflight,
    RunRequestSafetyError,
    build_metadata_preflight,
    confirm_output_target,
    recheck_metadata_selection,
    validate_output_confirmation,
)
from phenotypic.gui.run_console._runner import LocalRunner
from phenotypic.gui.run_console._slurm_observer import (
    IncrementalLogReader,
    SchedulerGenerationBinding,
    SlurmLifecycleObserver,
)
from phenotypic.gui.shell._ids import (
    SHELL_METADATA_CSV_STORE,
    SHELL_SOURCE_IMAGE_ROOT_STORE,
)
from phenotypic.gui.shell._metadata_context import (
    resolve_metadata_csv,
)
from phenotypic.gui.shell._runs_registry import RunMode, RunRecord, RunRegistry
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._source_context import (
    resolve_source_image_root,
)
from phenotypic.sdk_ import PIPELINE_CONFIG_SUFFIXES, matches_any_suffix

logger = logging.getLogger(__name__)

__all__ = ["register_callbacks"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _toast(
    message: str, *, ok: bool = True, header: Optional[str] = None
) -> Tuple[bool, str, str, str]:
    """Build the four toast outputs (``is_open``, ``children``, ``icon``, ``header``)."""
    icon = "primary" if ok else "danger"
    h = header if header is not None else ("Run console" if ok else "Error")
    return True, message, icon, h


def _format_exception(exc: BaseException) -> str:
    """Pretty single-line summary for toast display."""
    return f"{type(exc).__name__}: {exc}"


def _trigger_kind_path(
    triggered: Any, expected_type: str
) -> Optional[Tuple[str, str]]:
    """Validate a directory-tree click and return ``(kind, path)``."""
    if not isinstance(triggered, dict):
        return None
    if triggered.get("type") != expected_type:
        return None
    if not ctx.triggered or not ctx.triggered[0].get("value"):
        return None
    kind = triggered.get("kind")
    path = triggered.get("path")
    if not isinstance(kind, str) or not isinstance(path, str):
        return None
    return kind, path


def _shorten_path(path_str: Optional[str], sandbox: SandboxRoot) -> str:
    """Return a sandbox-relative display string for ``path_str``."""
    if not path_str:
        return "(none)"
    p = Path(path_str)
    try:
        return str(p.relative_to(sandbox.root))
    except ValueError:
        return path_str


def _input_dir_from_shared_source(
    sandbox: SandboxRoot,
    shared_payload: object,
    current_input_dir: Optional[str],
) -> str | None:
    """Return a Run input dir from shared source when the field is empty."""
    if current_input_dir:
        return None
    resolved = resolve_source_image_root(sandbox, shared_payload)
    return str(resolved) if resolved is not None else None


def _fingerprint_label(value: str | None) -> str:
    """Return a compact visible fingerprint without hiding its algorithm."""
    if value is None:
        return "unavailable"
    return value if len(value) <= 24 else f"{value[:20]}…"


def _metadata_preflight_children(report: MetadataPreflight) -> Any:
    """Render the exact ambient descriptor and source compatibility."""
    metadata_label = report.metadata_path or f"({report.metadata_state})"
    source_label = report.source_path or f"({report.source_state})"
    color = {
        "compatible": "success",
        "warning": "warning",
        "blocked": "danger",
        "pending": "secondary",
        "absent": "secondary",
    }[report.compatibility]
    details: list[Any] = [
        html.Div(
            [
                html.Strong("Status: "),
                report.compatibility,
            ]
        ),
        html.Div([html.Strong("Source: "), source_label]),
        html.Div(
            [
                html.Strong("Source inventory: "),
                f"{report.source_image_count} image(s), ",
                _fingerprint_label(report.source_fingerprint),
            ]
        ),
        html.Div([html.Strong("Ambient metadata: "), metadata_label]),
    ]
    if report.metadata_path is not None:
        details.extend(
            [
                html.Div(
                    [
                        html.Strong("Metadata descriptor: "),
                        f"{report.metadata_row_count} row(s), preflight join "
                        "keys ",
                        ", ".join(report.join_columns) or "none",
                        ", ",
                        _fingerprint_label(report.metadata_fingerprint),
                    ]
                ),
                html.Div(
                    [
                        html.Strong("Compatibility: "),
                        f"{report.matched_source_count}/"
                        f"{report.source_image_count} input image(s) matched; "
                        f"{report.metadata_only_count} metadata-only row(s); "
                        f"{report.duplicate_key_count} duplicate key row(s).",
                    ]
                ),
            ]
        )
    if report.warnings:
        details.append(
            html.Ul([html.Li(message) for message in report.warnings])
        )
    return dbc.Alert(details, color=color, className="mb-0 py-2")


def _looks_like_pipeline_json(path: Path) -> bool:
    """Cheap "is this a pipeline JSON?" probe — read first 4 KB.

    Returns ``True`` if the file is readable and contains the
    ``"operations"`` token in its first 4 KB. False positives are
    acceptable (the user gets a warning toast either way); we just want
    to catch the obvious "wrong file" case.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(4096)
    except OSError:
        return False
    return b'"operations"' in head


def _pipeline_uses_staged_gpu(path_value: object) -> bool:
    """Return whether the selected pipeline requires staged GPU execution."""
    if not isinstance(path_value, str) or not path_value:
        return False
    try:
        from phenotypic._cli._cli_validation import pipeline_requires_gpu

        return pipeline_requires_gpu(Path(path_value))
    except (OSError, ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Stream B seams — RunConsoleState + SLURM submitter.
# ---------------------------------------------------------------------------

from phenotypic.gui.run_console._slurm import (  # noqa: E402
    SlurmSubmitError,
    SlurmSubmitPending,
    SlurmSubmitResult,
    read_submitted_job_set,
    submit_slurm,
)
from phenotypic.gui.run_console._state import (  # noqa: E402
    RunConsoleState,
    run_state_from_json,
    run_state_to_json,
    state_from_controls,
    to_argv as state_to_argv_tail,
)


# ---------------------------------------------------------------------------
# Argv construction
# ---------------------------------------------------------------------------


def _local_argv_for(state: RunConsoleState) -> list[str]:
    """Build the full local-run ``argv`` from ``state``.

    Wraps Stream B's :func:`state_to_argv_tail` (which returns the tail
    starting with the pipeline path) and prepends ``[sys.executable,
    "-m", "phenotypic"]`` so :class:`subprocess.Popen` can spawn it.

    Args:
        state: Run-console state with required slots populated.

    Returns:
        Full argv list starting with ``sys.executable``.

    Raises:
        ValueError: Propagated from :func:`state_to_argv_tail` when a
            required slot is missing.
    """
    return [sys.executable, "-m", "phenotypic", *state_to_argv_tail(state)]


def _action_control_states() -> tuple[State, ...]:
    """Return every visible run-form control in authoritative action order."""
    return (
        State(ids.RC_STORE_PIPELINE_PATH, "data"),
        State(ids.RC_STORE_INPUT_DIR, "data"),
        State(ids.RC_INPUT_OUTPUT_PATH, "value"),
        State(ids.RC_RADIO_MODE, "value"),
        State(ids.RC_CHECKS_FLAGS, "value"),
        State(ids.RC_INPUT_SAMPLE, "value"),
        State(ids.RC_INPUT_NROWS, "value"),
        State(ids.RC_INPUT_NCOLS, "value"),
        State(ids.RC_INPUT_IMAGE_TYPE, "value"),
        State(ids.RC_INPUT_WORKERS, "value"),
        State(ids.RC_INPUT_LOG_LEVEL, "value"),
        State(ids.RC_INPUT_SLURM_PARTITION, "value"),
        State(ids.RC_INPUT_SLURM_TIME, "value"),
        State(ids.RC_INPUT_SLURM_MEM, "value"),
        State(ids.RC_INPUT_SLURM_CPUS, "value"),
        State(ids.RC_INPUT_SLURM_GPUS, "value"),
        State(ids.RC_INPUT_SLURM_EXTRA, "value"),
        State(ids.RC_INPUT_GPU_SLURM, "value"),
        State(ids.RC_INPUT_GPU_SHARDS, "value"),
        State(SHELL_METADATA_CSV_STORE, "data"),
        State(ids.RC_METADATA_CHOICE, "value"),
        State(ids.RC_METADATA_ACKNOWLEDGEMENT, "value"),
        State(ids.RC_STORE_METADATA_PREFLIGHT, "data"),
        State(ids.RC_STORE_OUTPUT_CONFIRMATION, "data"),
    )


def _action_control_outputs() -> tuple[Output, ...]:
    """Return preset-load controls without writing shared shell authority."""
    return (
        Output(ids.RC_STORE_PIPELINE_PATH, "data", allow_duplicate=True),
        Output(ids.RC_STORE_INPUT_DIR, "data", allow_duplicate=True),
        Output(ids.RC_INPUT_OUTPUT_PATH, "value", allow_duplicate=True),
        Output(ids.RC_RADIO_MODE, "value", allow_duplicate=True),
        Output(ids.RC_CHECKS_FLAGS, "value", allow_duplicate=True),
        Output(ids.RC_INPUT_SAMPLE, "value", allow_duplicate=True),
        Output(ids.RC_INPUT_NROWS, "value", allow_duplicate=True),
        Output(ids.RC_INPUT_NCOLS, "value", allow_duplicate=True),
        Output(ids.RC_INPUT_IMAGE_TYPE, "value", allow_duplicate=True),
        Output(ids.RC_INPUT_WORKERS, "value", allow_duplicate=True),
        Output(ids.RC_INPUT_LOG_LEVEL, "value", allow_duplicate=True),
        Output(ids.RC_INPUT_SLURM_PARTITION, "value", allow_duplicate=True),
        Output(ids.RC_INPUT_SLURM_TIME, "value", allow_duplicate=True),
        Output(ids.RC_INPUT_SLURM_MEM, "value", allow_duplicate=True),
        Output(ids.RC_INPUT_SLURM_CPUS, "value", allow_duplicate=True),
        Output(ids.RC_INPUT_SLURM_GPUS, "value", allow_duplicate=True),
        Output(ids.RC_INPUT_SLURM_EXTRA, "value", allow_duplicate=True),
        Output(ids.RC_INPUT_GPU_SLURM, "value", allow_duplicate=True),
        Output(ids.RC_INPUT_GPU_SHARDS, "value", allow_duplicate=True),
        Output(ids.RC_METADATA_CHOICE, "value", allow_duplicate=True),
        Output(ids.RC_STORE_OUTPUT_DIR, "data", allow_duplicate=True),
        Output(
            ids.RC_STORE_OUTPUT_CONFIRMATION,
            "data",
            allow_duplicate=True,
        ),
    )


def _state_from_action_controls(
    values: tuple[Any, ...],
    *,
    sandbox: SandboxRoot,
) -> RunConsoleState:
    """Build authoritative state from one action callback's raw controls."""
    if len(values) != 24:
        raise ValueError(
            f"expected 24 raw run controls, received {len(values)}"
        )
    output_dir = validate_output_confirmation(
        sandbox,
        values[2],
        values[23],
    )
    metadata_csv = recheck_metadata_selection(
        sandbox,
        input_dir=values[1],
        metadata_payload=values[19],
        choice=values[20],
        acknowledgement=values[21],
        preflight_payload=values[22],
    )
    state = state_from_controls(
        pipeline_path=values[0],
        input_dir=values[1],
        output_dir=str(output_dir),
        mode=values[3],
        flags=values[4],
        sample=values[5],
        nrows=values[6],
        ncols=values[7],
        image_type=values[8],
        workers=values[9],
        log_level=values[10],
        slurm_partition=values[11],
        slurm_time=values[12],
        slurm_mem=values[13],
        slurm_cpus=values[14],
        slurm_gpus=values[15],
        slurm_extra=values[16],
        gpu_slurm=values[17],
        gpu_shards=values[18],
        metadata_payload=None,
        sandbox=sandbox,
    )
    state.metadata_csv = str(metadata_csv) if metadata_csv is not None else None
    return state


def _controls_from_run_state(
    state: RunConsoleState,
    *,
    sandbox: SandboxRoot,
) -> tuple[Any, ...]:
    """Restore preset fields while requiring fresh output/metadata confirmation."""
    flags: list[str] = []
    if state.dry_run:
        flags.append("dry_run")
    if state.resume:
        flags.append("resume")

    advanced = state.advanced_args or {}
    slurm = state.slurm_args or {}
    raw_extra = slurm.get("extra")
    extra_lines = (
        "\n".join(f"{key}={value}" for key, value in raw_extra.items())
        if isinstance(raw_extra, dict)
        else ""
    )
    gpu_lines = "\n".join(state.gpu_slurm_args)
    del sandbox

    return (
        state.pipeline_path,
        state.input_dir,
        state.output_dir,
        state.mode,
        flags,
        advanced.get("sample"),
        advanced.get("nrows"),
        advanced.get("ncols"),
        advanced.get("image_type"),
        advanced.get("workers"),
        advanced.get("log_level"),
        slurm.get("partition"),
        slurm.get("time"),
        slurm.get("mem"),
        slurm.get("cpus_per_task"),
        slurm.get("gpus"),
        extra_lines or None,
        gpu_lines or None,
        state.gpu_shards,
        "omit",
        None,
        None,
    )


def _command_digest(state: RunConsoleState) -> str:
    """Return a stable digest for the exact validated launch state."""
    payload = json.dumps(
        run_state_to_json(state),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _resolved_output_identity(
    state: RunConsoleState,
    *,
    sandbox: SandboxRoot,
) -> tuple[Path, str]:
    """Return the contained output path and canonical registry run id."""
    if state.output_dir is None:
        raise ValueError("output_dir is required")
    output_dir = sandbox.resolve(state.output_dir)
    return output_dir, str(output_dir.relative_to(sandbox.root))


def _run_receipt(record: RunRecord) -> dict[str, str]:
    """Return the browser receipt for one durable registry generation."""
    if record.generation is None:
        raise ValueError("a launch receipt requires a durable generation")
    return {
        "run_id": record.run_id,
        "generation": str(record.generation),
        "rel_path": record.rel_path,
        "mode": record.mode,
    }


def _record_for_receipt(
    registry: RunRegistry,
    payload: object,
) -> RunRecord | None:
    """Resolve ``payload`` only when both run id and generation still match."""
    if not isinstance(payload, dict):
        return None
    run_id = payload.get("run_id")
    raw_generation = payload.get("generation")
    if not isinstance(run_id, str):
        return None
    try:
        generation = UUID(str(raw_generation))
    except (TypeError, ValueError):
        return None
    record = registry.get(run_id)
    if record is None or record.generation != generation:
        return None
    return record


def _action_result(
    action: str,
    click: int,
    *,
    message: str,
    record: RunRecord | None = None,
) -> dict[str, Any]:
    """Build a click-correlated server acknowledgement for the watchdog."""
    result: dict[str, Any] = {
        "action": action,
        "click": click,
        "outcome": "accepted" if record is not None else "error",
        "message": message,
    }
    if record is not None:
        result["receipt"] = _run_receipt(record)
    return result


def _require_slurm_request(state: RunConsoleState) -> None:
    """Reject any request that could reach the submitter without SLURM flags."""
    if state.mode != "slurm":
        raise ValueError("SLURM submitter requires mode='slurm'")
    if not state.slurm_args:
        raise ValueError("SLURM mode requires a nonempty CPU SLURM profile")


def _parse_slurm_extra(text: Optional[str]) -> dict[str, str]:
    """Parse a ``key=value`` textarea into a flat dict.

    Empty lines and lines that don't contain ``=`` are skipped silently
    — we don't want a typo to block submission, just to drop the bad
    line.
    """
    if not text:
        return {}
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key:
            out[key] = value
    return out


def _form_inputs_to_state(
    pipeline_path: Optional[str],
    input_dir: Optional[str],
    output_dir: Optional[str],
    mode: Optional[str],
    flags: Optional[List[str]],
    sample: Optional[Any],
    nrows: Optional[Any],
    ncols: Optional[Any],
    image_type: Optional[str],
    workers: Optional[Any],
    log_level: Optional[str],
    slurm_partition: Optional[str],
    slurm_time: Optional[str],
    slurm_mem: Optional[str],
    slurm_cpus: Optional[Any],
    slurm_gpus: Optional[Any],
    slurm_extra: Optional[str],
    *,
    metadata_payload: object = None,
    metadata_choice: object = "omit",
    sandbox: SandboxRoot | None = None,
) -> dict[str, Any]:
    """Bundle every form input into a flat state dict.

    This is the inverse of the Stream-B ``RunConsoleState.from_dict``
    contract. We avoid importing it directly here so the form callback
    fan-in does not blow up if Stream B's module is mid-rebase.
    """
    flag_set = set(flags or [])
    advanced = {
        "sample": sample,
        "nrows": nrows,
        "ncols": ncols,
        "image_type": image_type,
        "workers": workers,
        "log_level": log_level,
    }
    # ``RunConsoleState`` recognised SLURM keys (per Stream B):
    # ``partition``, ``time``, ``mem``, ``cpus_per_task``, ``gpus``.
    # Free-form ``k=v`` lines from the textarea go into ``extra``.
    slurm_typed: dict[str, Any] = {
        "partition": slurm_partition,
        "time": slurm_time,
        "mem": slurm_mem,
        "cpus_per_task": slurm_cpus,
        "gpus": slurm_gpus,
    }
    slurm_typed = {k: v for k, v in slurm_typed.items() if v not in (None, "")}
    slurm_args: dict[str, Any] = dict(slurm_typed)
    extra = _parse_slurm_extra(slurm_extra)
    if extra:
        slurm_args["extra"] = extra

    metadata_csv: str | None = None
    if sandbox is not None and metadata_choice == "include":
        resolved_metadata = resolve_metadata_csv(sandbox, metadata_payload)
        metadata_csv = (
            str(resolved_metadata) if resolved_metadata is not None else None
        )

    return {
        "pipeline_path": pipeline_path,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "metadata_csv": metadata_csv,
        "mode": mode or "local",
        "dry_run": "dry_run" in flag_set,
        "resume": "resume" in flag_set,
        "advanced_args": {
            k: v for k, v in advanced.items() if v not in (None, "")
        },
        "slurm_args": slurm_args,
    }


# ---------------------------------------------------------------------------
# Preset I/O
# ---------------------------------------------------------------------------


def _presets_dir(sandbox: SandboxRoot) -> Path:
    """Return ``<sandbox>/.phenotypic-gui/presets``, creating it if needed."""
    path = sandbox.root / SANDBOX_GUI_DIRNAME / SANDBOX_PRESETS_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _list_preset_options(sandbox: SandboxRoot) -> List[dict[str, str]]:
    """Build the ``options`` list for the load-preset dropdown."""
    presets_dir = _presets_dir(sandbox)
    out: List[dict[str, str]] = []
    for entry in sorted(presets_dir.glob("*.json")):
        name = entry.stem
        out.append({"label": name, "value": str(entry)})
    return out


# ---------------------------------------------------------------------------
# Concurrency cap
# ---------------------------------------------------------------------------


def _local_run_active(runner: LocalRunner, registry: RunRegistry) -> bool:
    """Return ``True`` iff any registered local run is still alive.

    Considers only ``mode="local"`` records — ``mode="validate"`` (dry-run
    probes) is intentionally excluded so a long validation does not block
    the Run button.
    """
    for record in registry.list():
        if record.mode != "local":
            continue
        if runner.is_running(record.run_id, generation=record.generation):
            return True
    return False


# ---------------------------------------------------------------------------
# Async SLURM submission (non-blocking)
# ---------------------------------------------------------------------------

# ``submit_slurm`` shells out to ``sbatch`` synchronously; on a contended
# SLURM controller this can block for tens of seconds. Running it directly
# inside a Dash callback would freeze the entire UI thread (log-tail
# polling, Cancel button, etc.). Instead we offload to a small dedicated
# thread pool and let the existing log-tail Interval (which runs once per
# second) drive a follow-up callback that resolves the future.
#
# The pool is module-level so a single executor backs every Run console
# instance in the process. Two workers are plenty — ``--max-local-runs``
# is 1 and SLURM submissions queue serially in practice.
_SLURM_EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix=f"{THREAD_NAME_PREFIX}-slurm",
)
@dataclass(frozen=True)
class _SlurmCompletion:
    """Presentation event emitted after durable registry reconciliation."""

    result: SlurmSubmitResult | None = None
    error: str | None = None
    pending: str | None = None


_PENDING_SLURM: dict[tuple[str, UUID], Future[Any]] = {}
_COMPLETED_SLURM: dict[tuple[str, UUID], _SlurmCompletion] = {}
_MAX_SLURM_COMPLETIONS = 128
_MAX_SLURM_LOG_GENERATIONS = 64
_PENDING_SLURM_LOCK = threading.Lock()
_TERMINAL_RUN_STATUSES = frozenset({"complete", "failed", "cancelled"})


class _SlurmLogTailCache:
    """Globally bound incremental readers to current nonterminal generations."""

    def __init__(
        self,
        registry: RunRegistry | None = None,
        *,
        max_generations: int = _MAX_SLURM_LOG_GENERATIONS,
    ) -> None:
        if max_generations < 1:
            raise ValueError("max_generations must be at least 1")
        self._registry = registry
        self._max_generations = max_generations
        self._readers: dict[tuple[str, UUID], IncrementalLogReader] = {}
        self._text: dict[tuple[str, UUID], str] = {}
        self._lock = threading.Lock()

    def read(self, record: RunRecord) -> str:
        """Read one generation while pruning all inactive cached generations."""
        if record.generation is None:
            return "(waiting for lifecycle generation...)"
        key = (record.run_id, record.generation)
        active_generations = self._active_registry_generations()
        cacheable = (
            record.status not in _TERMINAL_RUN_STATUSES
            and (
                active_generations is None
                or key in active_generations
            )
        )
        labelled_paths = {
            (
                f"{'GUI submitter' if 'gui' in path.parts else 'SLURM'}: "
                f"{path.name}"
            ): path
            for path in record.log_paths
        }
        with self._lock:
            self._prune_inactive_locked(active_generations)
            reader = self._readers.pop(key, None) or IncrementalLogReader()
            if cacheable:
                self._readers[key] = reader
            batch = reader.read(labelled_paths)
            if batch.text:
                prior = self._text.get(key, "")
                self._text[key] = (prior + "\n" + batch.text)[-128 * 1024 :]
            text = self._text.get(
                key,
                "(waiting for submission or scheduler output...)",
            )
            if not cacheable:
                self._readers.pop(key, None)
                self._text.pop(key, None)
            else:
                self._enforce_bound_locked()
            return text

    def _active_registry_generations(
        self,
    ) -> set[tuple[str, UUID]] | None:
        """Return current nonterminal generations, when a registry is bound."""
        if self._registry is None:
            return None
        return {
            (record.run_id, record.generation)
            for record in self._registry.list()
            if (
                record.generation is not None
                and record.status not in _TERMINAL_RUN_STATUSES
            )
        }

    def _prune_inactive_locked(
        self,
        active_generations: set[tuple[str, UUID]] | None,
    ) -> None:
        """Evict terminal or superseded generations across every run."""
        if active_generations is None:
            return
        for key in tuple(self._readers):
            if key not in active_generations:
                self._readers.pop(key, None)
                self._text.pop(key, None)

    def _enforce_bound_locked(self) -> None:
        """Evict least-recently-read generations above the global cap."""
        while len(self._readers) > self._max_generations:
            oldest = next(iter(self._readers))
            self._readers.pop(oldest, None)
            self._text.pop(oldest, None)

    @property
    def tracked_generations(self) -> int:
        """Return the number of generation-scoped readers retained."""
        with self._lock:
            return len(self._readers)


def _persist_scheduler_binding(
    registry: RunRegistry,
    run_id: str,
    record_generation: UUID,
    output_dir: Path,
    scheduler_generation: UUID,
) -> bool:
    """Persist an exact GUI-owner to scheduler-lifecycle binding."""
    from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle

    lifecycle = load_slurm_lifecycle(output_dir)
    raw_epoch = lifecycle.get("generation") if lifecycle else None
    try:
        parsed_epoch = UUID(str(raw_epoch))
    except (TypeError, ValueError):
        return False
    if parsed_epoch != scheduler_generation:
        return False
    return registry.compare_and_set(
        run_id,
        record_generation,
        expected_statuses={
            "submitting",
            "queued",
            "running",
            "reconciling",
            "cancelling",
            "unknown",
        },
        lifecycle_epoch=str(raw_epoch),
    )


def _persist_and_bind_scheduler_generation(
    *,
    registry: RunRegistry,
    observer: SlurmLifecycleObserver,
    run_id: str,
    record_generation: UUID,
    output_dir: Path,
    scheduler_generation: UUID,
) -> SchedulerGenerationBinding | None:
    """Bind in memory only after the exact durable association is persisted."""
    if not _persist_scheduler_binding(
        registry,
        run_id,
        record_generation,
        output_dir,
        scheduler_generation,
    ):
        return None
    try:
        return observer.bind_generation(
            run_id=run_id,
            record_generation=record_generation,
            scheduler_generation=scheduler_generation,
        )
    except ValueError:
        logger.exception("Could not bind scheduler generation for %s", run_id)
        return None


def _complete_slurm_submission(
    future: Future[Any],
    *,
    registry: RunRegistry,
    run_id: str,
    generation: UUID,
    observer: SlurmLifecycleObserver,
) -> None:
    """Reconcile one submitter future independently of browser page state."""
    key = (run_id, generation)
    try:
        result = future.result()
        if not isinstance(result, SlurmSubmitResult):
            raise TypeError(
                "SLURM submitter returned an unexpected result "
                f"{type(result).__name__}"
            )
    except SlurmSubmitPending as exc:
        jobs = exc.submitted_jobs
        binding = _persist_and_bind_scheduler_generation(
            registry=registry,
            observer=observer,
            run_id=run_id,
            record_generation=generation,
            output_dir=exc.output_dir,
            scheduler_generation=exc.generation,
        )
        record = registry.get(run_id)
        if record is None or record.generation != generation:
            completion = None
        elif record.status == "cancelling" and binding is not None:
            _cancel_bound_generation(record, binding)
            completion = _SlurmCompletion(pending=str(exc))
        elif record.status == "cancelling":
            completion = _SlurmCompletion(pending=str(exc))
        else:
            scheduler_ids = jobs.all_ids if jobs is not None else ()
            primary = jobs.primary_id if jobs is not None else None
            registry.compare_and_set(
                run_id,
                generation,
                expected_statuses={"submitting", "unknown", "reconciling"},
                status=(
                    "submitting" if exc.scheduler_available else "unknown"
                ),
                scheduler_ids=scheduler_ids,
                primary_scheduler_id=primary,
                returncode=exc.returncode,
                status_detail=str(exc),
            )
            completion = _SlurmCompletion(pending=str(exc))
    except Exception as exc:  # noqa: BLE001
        detail = (
            str(exc)
            if isinstance(exc, SlurmSubmitError)
            else _format_exception(exc)
        )
        record = registry.get(run_id)
        cancelled_before_submission = bool(
            record is not None
            and record.generation == generation
            and record.status == "cancelling"
            and isinstance(exc, SlurmSubmitError)
        )
        updated = registry.compare_and_set(
            run_id,
            generation,
            expected_statuses=(
                {"cancelling"}
                if cancelled_before_submission
                else {"submitting", "reconciling", "unknown"}
            ),
            status=("cancelled" if cancelled_before_submission else "failed"),
            terminal_at=(
                datetime.now(timezone.utc)
                if cancelled_before_submission
                else None
            ),
            status_detail=detail,
        )
        completion = _SlurmCompletion(error=detail) if updated else None
    else:
        jobs = result.submitted_jobs or read_submitted_job_set(
            result.output_dir
        )
        if jobs is None:
            detail = "SLURM submission returned no generation-bound job set"
            updated = registry.compare_and_set(
                run_id,
                generation,
                expected_statuses={"submitting", "reconciling", "unknown"},
                status="unknown",
                status_detail=detail,
            )
            completion = _SlurmCompletion(error=detail) if updated else None
        else:
            binding = _persist_and_bind_scheduler_generation(
                registry=registry,
                observer=observer,
                run_id=run_id,
                record_generation=generation,
                output_dir=result.output_dir,
                scheduler_generation=jobs.generation,
            )
            if binding is None:
                detail = (
                    "Could not durably bind the GUI run to the scheduler "
                    "generation"
                )
                updated = registry.compare_and_set(
                    run_id,
                    generation,
                    expected_statuses={
                        "submitting",
                        "reconciling",
                        "unknown",
                    },
                    status="unknown",
                    status_detail=detail,
                )
                completion = (
                    _SlurmCompletion(error=detail) if updated else None
                )
            else:
                record = registry.get(run_id)
                if (
                    record is not None
                    and record.generation == generation
                    and record.status == "cancelling"
                ):
                    _cancel_bound_generation(record, binding)
                    completion = _SlurmCompletion(result=result)
                else:
                    updated = registry.compare_and_set(
                        run_id,
                        generation,
                        expected_statuses={
                            "submitting",
                            "reconciling",
                            "unknown",
                        },
                        status="queued",
                        scheduler_ids=jobs.all_ids,
                        primary_scheduler_id=jobs.primary_id,
                        submitted_at=datetime.now(timezone.utc),
                        returncode=result.returncode,
                        status_detail=None,
                    )
                    completion = (
                        _SlurmCompletion(result=result) if updated else None
                    )

    with _PENDING_SLURM_LOCK:
        if _PENDING_SLURM.get(key) is future:
            _PENDING_SLURM.pop(key, None)
        if completion is not None:
            _COMPLETED_SLURM[key] = completion
            while len(_COMPLETED_SLURM) > _MAX_SLURM_COMPLETIONS:
                oldest_key = next(iter(_COMPLETED_SLURM))
                _COMPLETED_SLURM.pop(oldest_key)


def _track_pending_slurm(
    run_id: str,
    generation: UUID,
    future: Future[Any],
    *,
    registry: RunRegistry,
    observer: SlurmLifecycleObserver,
) -> None:
    """Track a future and immediately attach its generation-matched callback."""
    key = (run_id, generation)
    with _PENDING_SLURM_LOCK:
        _PENDING_SLURM[key] = future
    future.add_done_callback(
        lambda completed: _complete_slurm_submission(
            completed,
            registry=registry,
            run_id=run_id,
            generation=generation,
            observer=observer,
        )
    )


def _take_slurm_completion(
    run_id: str,
    generation: UUID,
) -> _SlurmCompletion | None:
    """Pop one presentation event after lifecycle reconciliation."""
    with _PENDING_SLURM_LOCK:
        return _COMPLETED_SLURM.pop((run_id, generation), None)


def _has_pending_slurm() -> bool:
    """True iff at least one pending submission is registered."""
    with _PENDING_SLURM_LOCK:
        return bool(_PENDING_SLURM)


def _cancel_pending_slurm(run_id: str, generation: UUID) -> bool:
    """Cancel a submit future that has not started running yet."""
    with _PENDING_SLURM_LOCK:
        future = _PENDING_SLURM.get((run_id, generation))
    return bool(future is not None and future.cancel())


def _cancel_bound_generation(
    record: RunRecord,
    binding: SchedulerGenerationBinding,
) -> tuple[str, ...]:
    """Fence and cancel every scheduler job while retaining ``cancelling``."""
    from phenotypic._cli._cli_slurm_lifecycle import cancel_generation

    if (
        record.generation is None
        or binding.run_id != record.run_id
        or binding.record_generation != record.generation
        or record.lifecycle_epoch != binding.scheduler_epoch
    ):
        return ()
    result = cancel_generation(
        record.output_dir,
        binding.scheduler_epoch,
    )
    return result.job_ids



# ---------------------------------------------------------------------------
# Dashboard polling helper
# ---------------------------------------------------------------------------


def _dashboard_url(
    rel_path: str, *, url_prefix: str = DEFAULT_URL_PREFIX
) -> str:
    """Build the iframe ``src`` for ``rel_path``.

    The shell mounts ``/runs/<rel>/<file>`` regardless of the Dash sub-app's
    mount prefix. ``url_prefix`` is the optional browser-visible base path
    supplied for reverse proxies such as Open OnDemand ``/node`` or ``/rnode``.
    The dashboard now lives under the run's ``deliverables/`` subdirectory.
    """
    safe_rel = rel_path.strip("/").replace("\\", "/")
    path = (
        f"{RUNS_BLUEPRINT_PREFIX}/{safe_rel}/"
        f"{DELIVERABLES_DIRNAME}/{DASHBOARD_FILENAME}"
    )
    return join_url_prefix(url_prefix, path)


# ---------------------------------------------------------------------------
# register_callbacks
# ---------------------------------------------------------------------------


def register_callbacks(
    app: dash.Dash,
    sandbox: SandboxRoot,
    *,
    registry: RunRegistry,
    runner: LocalRunner,
    slurm_observer: SlurmLifecycleObserver,
    slurm_submitter: Callable[..., SlurmSubmitResult] = submit_slurm,
    server_url_prefix: str = DEFAULT_URL_PREFIX,
) -> None:
    """Register every Run console callback on ``app``.

    Idempotent within one app instance. The leader's ``_app.py`` factory
    calls this exactly once after assigning ``app.layout =
    build_run_console_layout(...)``.

    Args:
        app: The :class:`dash.Dash` instance.
        sandbox: Frozen-at-launch sandbox; passed into modal renderers
            and the output-picker confirm path.
        registry: Process-wide :class:`RunRegistry`. Mutated by the
            run-start, cancel, and recents-refresh callbacks.
        runner: Process-wide :class:`LocalRunner`. Owns subprocess
            lifecycle for Local runs.
        slurm_observer: Process-wide lifecycle observer. Submission callbacks
            bind exact scheduler epochs to GUI record generations.
        slurm_submitter: Submission dependency. Production uses
            :func:`submit_slurm`; browser tests inject a no-scheduler seam.
        server_url_prefix: Browser-visible base prefix for shell-level
            Flask routes such as ``/runs``.
    """
    slurm_log_cache = _SlurmLogTailCache(registry)

    app.clientside_callback(
        """
        function(validateClicks, runClicks) {
            const context = window.dash_clientside.callback_context;
            const trigger = context.triggered_id;
            const action = trigger === "rc-btn-validate" ? "validate"
                : trigger === "rc-btn-run" ? "run" : null;
            if (!action) {
                return window.dash_clientside.no_update;
            }
            const click = action === "validate" ? validateClicks : runClicks;
            if (!click) {
                return window.dash_clientside.no_update;
            }
            return {
                action: action,
                click: click,
                started_at_ms: Date.now()
            };
        }
        """,
        Output(ids.RC_STORE_ACTION_ATTEMPT, "data"),
        Input(ids.RC_BTN_VALIDATE, "n_clicks"),
        Input(ids.RC_BTN_RUN, "n_clicks"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        f"""
        function(attempt, result, _tick) {{
            const base = "run-console-action-feedback";
            if (!attempt || !attempt.action || !attempt.click) {{
                return ["No action requested.", base, {{display: "none"}}];
            }}
            const matched = result
                && result.action === attempt.action
                && result.click === attempt.click;
            if (matched && result.outcome === "accepted" && result.receipt) {{
                const receipt = result.receipt;
                return [
                    `${{attempt.action}} accepted | run_id=${{receipt.run_id}}`
                    + ` | generation=${{receipt.generation}}`,
                    base + " run-console-action-feedback--ok",
                    {{display: "block"}}
                ];
            }}
            if (matched) {{
                return [
                    `${{attempt.action}} error | ${{result.message}}`,
                    base + " run-console-action-feedback--error",
                    {{display: "block"}}
                ];
            }}
            if (Date.now() - Number(attempt.started_at_ms || 0)
                    >= {RUN_ACTION_CALLBACK_TIMEOUT_MS}) {{
                return [
                    `${{attempt.action}} error | callback request failed before`
                    + " a durable generation was confirmed; retry the action.",
                    base + " run-console-action-feedback--error",
                    {{display: "block"}}
                ];
            }}
            return [
                `${{attempt.action}} request pending…`,
                base + " run-console-action-feedback--pending",
                {{display: "block"}}
            ];
        }}
        """,
        Output(ids.RC_ACTION_FEEDBACK, "children"),
        Output(ids.RC_ACTION_FEEDBACK, "className"),
        Output(ids.RC_ACTION_FEEDBACK, "style"),
        Input(ids.RC_STORE_ACTION_ATTEMPT, "data"),
        Input(ids.RC_STORE_ACTION_RESULT, "data"),
        Input(ids.RC_INTERVAL_ACTION_WATCHDOG, "n_intervals"),
    )

    # ----------------------------------------------------------------------
    # 1. Picker buttons → open / cancel modals
    # ----------------------------------------------------------------------
    # Each picker button (Pipeline / Input / Output) opens its modal; each
    # Cancel button closes it. The wiring is mechanical — register the six
    # callbacks in a loop so the per-modal IDs read top-to-bottom.

    _modal_buttons: list[tuple[str, str, str]] = [
        (
            ids.RC_MODAL_PIPELINE,
            ids.RC_BTN_PICK_PIPELINE,
            ids.RC_BTN_PIPELINE_CANCEL,
        ),
        (ids.RC_MODAL_INPUT, ids.RC_BTN_PICK_INPUT, ids.RC_BTN_INPUT_CANCEL),
        (
            ids.RC_MODAL_OUTPUT,
            ids.RC_BTN_PICK_OUTPUT,
            ids.RC_BTN_OUTPUT_CANCEL,
        ),
    ]

    def _register_modal_toggle(
        modal_id: str, button_id: str, *, open_value: bool
    ) -> None:
        """Register an open/close callback for a single picker button."""

        @app.callback(
            Output(modal_id, "is_open", allow_duplicate=True),
            Input(button_id, "n_clicks"),
            prevent_initial_call=True,
        )
        def _toggle_modal(n_clicks: Optional[int]) -> Any:
            if not n_clicks:
                return no_update
            return open_value

    for modal_id, open_btn_id, cancel_btn_id in _modal_buttons:
        _register_modal_toggle(modal_id, open_btn_id, open_value=True)
        _register_modal_toggle(modal_id, cancel_btn_id, open_value=False)

    # ----------------------------------------------------------------------
    # 2. Pipeline-tree navigation + file confirm
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.RC_STORE_BROWSE_DIR_PIPELINE, "data", allow_duplicate=True),
        Output(ids.RC_STORE_PIPELINE_PATH, "data", allow_duplicate=True),
        Output(ids.RC_MODAL_PIPELINE, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "children", allow_duplicate=True),
        Output(ids.RC_TOAST, "icon", allow_duplicate=True),
        Output(ids.RC_TOAST, "header", allow_duplicate=True),
        Input(
            {
                "type": ids.RC_DIR_ENTRY_TYPE_PIPELINE_JSON,
                "kind": ALL,
                "path": ALL,
            },
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def click_pipeline_entry(_clicks: List[int]) -> Tuple[Any, ...]:
        """Handle a click in the pipeline-JSON tree (navigate or pick)."""
        match = _trigger_kind_path(
            ctx.triggered_id, ids.RC_DIR_ENTRY_TYPE_PIPELINE_JSON
        )
        if match is None:
            return (no_update,) * 7
        kind, path_str = match
        if kind in {"dir", "parent"}:
            return (path_str, *((no_update,) * 6))
        if kind == "file":
            chosen = Path(path_str)
            if not _looks_like_pipeline_json(chosen):
                # Allow selection but warn — the file may still be a
                # valid pipeline (truncated probe, custom serialiser).
                return (
                    no_update,
                    str(chosen),
                    False,
                    *_toast(
                        f"Picked {chosen.name} (no 'operations' token in"
                        " first 4 KB -- may not be a pipeline).",
                        ok=True,
                        header="Run console (warning)",
                    ),
                )
            return (
                no_update,
                str(chosen),
                False,
                *_toast(f"Pipeline: {chosen.name}", ok=True),
            )
        return (no_update,) * 7

    def _register_tree_renderer(
        body_id: str,
        store_id: str,
        renderer: Callable[[SandboxRoot, Optional[Path]], Any],
    ) -> None:
        """Register a body-rerender callback for one picker tree."""

        @app.callback(
            Output(body_id, "children"),
            Input(store_id, "data"),
            prevent_initial_call=True,
        )
        def _render_body(dir_value: Optional[str]) -> Any:
            current = Path(dir_value) if dir_value else None
            return renderer(sandbox, current)

    _register_tree_renderer(
        ids.RC_MODAL_PIPELINE_BODY,
        ids.RC_STORE_BROWSE_DIR_PIPELINE,
        render_pipeline_tree,
    )

    # ----------------------------------------------------------------------
    # 3. Input-tree navigation + confirm
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.RC_STORE_BROWSE_DIR_INPUT, "data", allow_duplicate=True),
        Input(
            {
                "type": ids.RC_DIR_ENTRY_TYPE_INPUT_DIR,
                "kind": ALL,
                "path": ALL,
            },
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def navigate_input_tree(_clicks: List[int]) -> Any:
        """Update the browse-dir store when a folder is clicked."""
        match = _trigger_kind_path(
            ctx.triggered_id, ids.RC_DIR_ENTRY_TYPE_INPUT_DIR
        )
        if match is None:
            return no_update
        _, path_str = match
        return path_str

    _register_tree_renderer(
        ids.RC_MODAL_INPUT_BODY,
        ids.RC_STORE_BROWSE_DIR_INPUT,
        render_input_tree,
    )

    @app.callback(
        Output(ids.RC_STORE_INPUT_DIR, "data", allow_duplicate=True),
        Output(ids.RC_MODAL_INPUT, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "children", allow_duplicate=True),
        Output(ids.RC_TOAST, "icon", allow_duplicate=True),
        Output(ids.RC_TOAST, "header", allow_duplicate=True),
        Input(ids.RC_BTN_INPUT_CONFIRM, "n_clicks"),
        State(ids.RC_STORE_BROWSE_DIR_INPUT, "data"),
        prevent_initial_call=True,
    )
    def confirm_input_dir(
        n_clicks: Optional[int], dir_value: Optional[str]
    ) -> Tuple[Any, ...]:
        """Confirm the currently-browsed input directory."""
        if not n_clicks:
            return (no_update,) * 6
        if not dir_value:
            return (
                no_update,
                no_update,
                *_toast("Pick a folder first", ok=False),
            )
        chosen = Path(dir_value)
        # Cheap "any image files?" probe — count entries with an image-ish
        # extension in the depth-1 listing.
        try:
            sample = [
                p
                for p in chosen.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTS
            ]
        except OSError:
            sample = []
        toast_text = (
            f"Input: {chosen.name}"
            if sample
            else f"Input: {chosen.name} (no obvious image files --"
            " set --image-type if needed)"
        )
        return (
            str(chosen),
            False,
            *_toast(toast_text, ok=True),
        )

    # ----------------------------------------------------------------------
    # 4. Output-tree navigation + confirm + path-input typing
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.RC_STORE_BROWSE_DIR_OUTPUT, "data", allow_duplicate=True),
        Output(ids.RC_INPUT_OUTPUT_PATH, "value", allow_duplicate=True),
        Input(
            {
                "type": ids.RC_DIR_ENTRY_TYPE_OUTPUT_DIR,
                "kind": ALL,
                "path": ALL,
            },
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def navigate_output_tree(_clicks: List[int]) -> Tuple[Any, Any]:
        """Update browse-dir + path input when a folder is clicked."""
        match = _trigger_kind_path(
            ctx.triggered_id, ids.RC_DIR_ENTRY_TYPE_OUTPUT_DIR
        )
        if match is None:
            return no_update, no_update
        _, path_str = match
        return path_str, path_str

    _register_tree_renderer(
        ids.RC_MODAL_OUTPUT_BODY,
        ids.RC_STORE_BROWSE_DIR_OUTPUT,
        render_output_dir_tree,
    )

    @app.callback(
        Output(ids.RC_STORE_OUTPUT_DIR, "data", allow_duplicate=True),
        Output(
            ids.RC_STORE_OUTPUT_CONFIRMATION,
            "data",
            allow_duplicate=True,
        ),
        Output(ids.RC_MODAL_OUTPUT, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "children", allow_duplicate=True),
        Output(ids.RC_TOAST, "icon", allow_duplicate=True),
        Output(ids.RC_TOAST, "header", allow_duplicate=True),
        Input(ids.RC_BTN_OUTPUT_CONFIRM, "n_clicks"),
        State(ids.RC_INPUT_OUTPUT_PATH, "value"),
        prevent_initial_call=True,
    )
    def confirm_output_dir(
        n_clicks: Optional[int],
        typed_value: Optional[str],
    ) -> Tuple[Any, ...]:
        """Confirm only the typed path and preserve a new canonical target."""
        if not n_clicks:
            return (no_update,) * 7
        try:
            confirmation = confirm_output_target(sandbox, typed_value)
        except RunRequestSafetyError as exc:
            return (
                no_update,
                no_update,
                no_update,
                *_toast(
                    str(exc),
                    ok=False,
                ),
            )
        return (
            confirmation.canonical_path,
            confirmation.to_json(),
            False,
            *_toast(
                f"Output confirmed: {confirmation.relative_path}",
                ok=True,
            ),
        )

    # ----------------------------------------------------------------------
    # 5. Picker labels (mirror selected paths to display)
    # ----------------------------------------------------------------------
    # Each picker store mirrors its sandbox-relative display string into the
    # adjacent label.

    def _register_picker_label(label_id: str, store_id: str) -> None:
        """Register a label-mirroring callback for one picker store."""

        @app.callback(
            Output(label_id, "children"),
            Input(store_id, "data"),
        )
        def _update_label(path_str: Optional[str]) -> str:
            return _shorten_path(path_str, sandbox)

    _register_picker_label(ids.RC_LABEL_PIPELINE, ids.RC_STORE_PIPELINE_PATH)
    _register_picker_label(ids.RC_LABEL_INPUT, ids.RC_STORE_INPUT_DIR)
    _register_picker_label(ids.RC_LABEL_OUTPUT, ids.RC_STORE_OUTPUT_DIR)

    # ----------------------------------------------------------------------
    # 5b. Shared source consumption + visible metadata preflight
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.RC_STORE_INPUT_DIR, "data", allow_duplicate=True),
        Input(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
        State(ids.RC_STORE_INPUT_DIR, "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _initialize_input_dir_from_shared_source(
        shared_payload: object,
        current_input_dir: Optional[str],
    ) -> Any:
        input_dir = _input_dir_from_shared_source(
            sandbox, shared_payload, current_input_dir
        )
        return input_dir if input_dir is not None else no_update

    @app.callback(
        Output(ids.RC_METADATA_PREFLIGHT, "children"),
        Output(ids.RC_STORE_METADATA_PREFLIGHT, "data"),
        Output(ids.RC_METADATA_CHOICE, "value", allow_duplicate=True),
        Output(
            ids.RC_METADATA_ACKNOWLEDGEMENT,
            "value",
            allow_duplicate=True,
        ),
        Output(f"{ids.RC_METADATA_ACKNOWLEDGEMENT}-wrapper", "style"),
        Input(ids.RC_STORE_INPUT_DIR, "data"),
        Input(SHELL_METADATA_CSV_STORE, "data"),
        prevent_initial_call="initial_duplicate",
    )
    def refresh_metadata_preflight(
        input_dir: object,
        metadata_payload: object,
    ) -> tuple[Any, ...]:
        """Publish a current snapshot and reset request authority to Omit."""
        report = build_metadata_preflight(
            sandbox,
            input_dir,
            metadata_payload,
        )
        acknowledgement_style = {
            "display": "block"
            if report.requires_acknowledgement
            else "none"
        }
        return (
            _metadata_preflight_children(report),
            report.to_json(),
            "omit",
            [],
            acknowledgement_style,
        )

    # ----------------------------------------------------------------------
    # 6. Advanced + SLURM collapses
    # ----------------------------------------------------------------------
    # Each toggle button flips its collapse open/closed.

    def _register_collapse_toggle(collapse_id: str, button_id: str) -> None:
        """Register an open/closed flip for one collapse button."""

        @app.callback(
            Output(collapse_id, "is_open"),
            Input(button_id, "n_clicks"),
            State(collapse_id, "is_open"),
            prevent_initial_call=True,
        )
        def _toggle(n_clicks: Optional[int], is_open: bool) -> bool:
            if not n_clicks:
                return is_open
            return not is_open

    _register_collapse_toggle(
        ids.RC_COLLAPSE_ADVANCED, ids.RC_BTN_TOGGLE_ADVANCED
    )
    _register_collapse_toggle(ids.RC_COLLAPSE_SLURM, ids.RC_BTN_TOGGLE_SLURM)

    @app.callback(
        Output(ids.RC_STAGED_GPU_SECTION, "style"),
        Input(ids.RC_STORE_PIPELINE_PATH, "data"),
        Input(ids.RC_RADIO_MODE, "value"),
    )
    def show_staged_gpu_controls(
        pipeline_path: object,
        mode: object,
    ) -> dict[str, str]:
        """Show GPU-stage resources only for a SLURM GPU pipeline."""
        visible = mode == "slurm" and _pipeline_uses_staged_gpu(pipeline_path)
        return {"display": "block" if visible else "none"}

    # ----------------------------------------------------------------------
    # 7. Form-state sync — every input writes back to the form state store.
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.RC_STORE_FORM_STATE, "data"),
        Input(ids.RC_STORE_PIPELINE_PATH, "data"),
        Input(ids.RC_STORE_INPUT_DIR, "data"),
        Input(ids.RC_STORE_OUTPUT_DIR, "data"),
        Input(ids.RC_RADIO_MODE, "value"),
        Input(ids.RC_CHECKS_FLAGS, "value"),
        Input(ids.RC_INPUT_SAMPLE, "value"),
        Input(ids.RC_INPUT_NROWS, "value"),
        Input(ids.RC_INPUT_NCOLS, "value"),
        Input(ids.RC_INPUT_IMAGE_TYPE, "value"),
        Input(ids.RC_INPUT_WORKERS, "value"),
        Input(ids.RC_INPUT_LOG_LEVEL, "value"),
        Input(ids.RC_INPUT_SLURM_PARTITION, "value"),
        Input(ids.RC_INPUT_SLURM_TIME, "value"),
        Input(ids.RC_INPUT_SLURM_MEM, "value"),
        Input(ids.RC_INPUT_SLURM_CPUS, "value"),
        Input(ids.RC_INPUT_SLURM_GPUS, "value"),
        Input(ids.RC_INPUT_SLURM_EXTRA, "value"),
        Input(ids.RC_INPUT_GPU_SLURM, "value"),
        Input(ids.RC_INPUT_GPU_SHARDS, "value"),
        Input(SHELL_METADATA_CSV_STORE, "data"),
        Input(ids.RC_METADATA_CHOICE, "value"),
    )
    def sync_form_state(  # noqa: PLR0913
        pipeline_path: Optional[str],
        input_dir: Optional[str],
        output_dir: Optional[str],
        mode: Optional[str],
        flags: Optional[List[str]],
        sample: Optional[Any],
        nrows: Optional[Any],
        ncols: Optional[Any],
        image_type: Optional[str],
        workers: Optional[Any],
        log_level: Optional[str],
        slurm_partition: Optional[str],
        slurm_time: Optional[str],
        slurm_mem: Optional[str],
        slurm_cpus: Optional[Any],
        slurm_gpus: Optional[Any],
        slurm_extra: Optional[str],
        gpu_slurm: Optional[str],
        gpu_shards: Optional[Any],
        metadata_payload: object,
        metadata_choice: object,
    ) -> dict[str, Any]:
        """Bundle all form fields into the run-state store on any change."""
        payload = _form_inputs_to_state(
            pipeline_path,
            input_dir,
            output_dir,
            mode,
            flags,
            sample,
            nrows,
            ncols,
            image_type,
            workers,
            log_level,
            slurm_partition,
            slurm_time,
            slurm_mem,
            slurm_cpus,
            slurm_gpus,
            slurm_extra,
            metadata_payload=metadata_payload,
            metadata_choice=metadata_choice,
            sandbox=sandbox,
        )
        payload["gpu_slurm_args"] = [
            line.strip()
            for line in (gpu_slurm or "").splitlines()
            if line.strip()
        ]
        payload["gpu_shards"] = gpu_shards or 1
        return payload

    # ----------------------------------------------------------------------
    # 8. Validate / Run / Cancel
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.RC_TOAST, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "children", allow_duplicate=True),
        Output(ids.RC_TOAST, "icon", allow_duplicate=True),
        Output(ids.RC_TOAST, "header", allow_duplicate=True),
        Output(ids.RC_STORE_ACTIVE_RUN_ID, "data", allow_duplicate=True),
        Output(ids.RC_STORE_ACTIVE_REL_PATH, "data", allow_duplicate=True),
        Output(ids.RC_STORE_ACTIVE_RUN_RECEIPT, "data", allow_duplicate=True),
        Output(ids.RC_STORE_ACTION_RESULT, "data"),
        Output(ids.RC_INTERVAL_LOG, "disabled", allow_duplicate=True),
        Output(
            ids.RC_INTERVAL_DASHBOARD_POLL, "disabled", allow_duplicate=True
        ),
        Output(
            ids.RC_INTERVAL_DASHBOARD_POLL, "n_intervals", allow_duplicate=True
        ),
        Output(ids.RC_STORE_RECENTS_REFRESH, "data", allow_duplicate=True),
        Input(ids.RC_BTN_VALIDATE, "n_clicks"),
        Input(ids.RC_BTN_RUN, "n_clicks"),
        *_action_control_states(),
        State(ids.RC_STORE_RECENTS_REFRESH, "data"),
        prevent_initial_call=True,
    )
    def click_action(
        validate_clicks: Optional[int],
        run_clicks: Optional[int],
        *args: Any,
    ) -> Tuple[Any, ...]:
        """Handle Validate and Run through one generation-recording seam."""
        try:
            triggered = ctx.triggered_id
        except dash.exceptions.MissingCallbackContextException:
            triggered = None
        if triggered == ids.RC_BTN_VALIDATE:
            action = "validate"
            click = int(validate_clicks or 0)
        elif triggered == ids.RC_BTN_RUN:
            action = "run"
            click = int(run_clicks or 0)
        elif run_clicks and not validate_clicks:
            # Direct callback invocation in integration tests has no Dash
            # callback context. This fallback is not used by browser traffic.
            action = "run"
            click = int(run_clicks)
        elif validate_clicks and not run_clicks:
            action = "validate"
            click = int(validate_clicks)
        else:
            return (no_update,) * 12
        if not click:
            return (no_update,) * 12

        control_values = tuple(args[:-1])
        refresh_count = args[-1] if args else None
        try:
            state = _state_from_action_controls(
                control_values, sandbox=sandbox
            )
            output_dir, rel_path = _resolved_output_identity(
                state, sandbox=sandbox
            )
        except Exception as exc:  # noqa: BLE001
            detail = _format_exception(exc)
            return (
                *_toast(detail, ok=False),
                no_update,
                no_update,
                no_update,
                _action_result(action, click, message=detail),
                no_update,
                no_update,
                no_update,
                no_update,
            )

        new_refresh = (refresh_count or 0) + 1
        record: RunRecord | None = None

        if action == "validate":
            try:
                state.dry_run = True
                argv = _local_argv_for(state)
                record = registry.allocate(
                    mode="validate",
                    output_dir=output_dir,
                    rel_path=rel_path,
                    command_digest=_command_digest(state),
                    status="queued",
                )
                generation = record.generation
                if generation is None:  # pragma: no cover
                    raise RuntimeError(
                        "allocated validation has no generation"
                    )
                validation_run_id = record.run_id

                def _observe_validation_exit(
                    _handle: Any,
                    returncode: int,
                ) -> None:
                    registry.observe_local_exit(
                        validation_run_id,
                        generation,
                        returncode,
                    )

                handle = runner.start(
                    record.run_id,
                    argv,
                    output_dir=output_dir,
                    generation=generation,
                    on_exit=_observe_validation_exit,
                )
                registry.compare_and_set(
                    record.run_id,
                    generation,
                    expected_statuses={"queued"},
                    status="running",
                    pid=handle.process.pid,
                    log_paths=(handle.stdout_log_path,),
                )
                message = "Validation (dry-run) started"
                return (
                    *_toast(message, ok=True),
                    record.run_id,
                    None,
                    _run_receipt(record),
                    _action_result(
                        action,
                        click,
                        message=message,
                        record=record,
                    ),
                    False,
                    True,
                    0,
                    new_refresh,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Validate failed")
                detail = _format_exception(exc)
                if record is not None and record.generation is not None:
                    registry.compare_and_set(
                        record.run_id,
                        record.generation,
                        expected_statuses={"queued", "running"},
                        status="failed",
                        status_detail=detail,
                    )
                    failed = registry.get(record.run_id) or record
                    return (
                        *_toast(detail, ok=False),
                        record.run_id,
                        None,
                        _run_receipt(record),
                        _action_result(
                            action,
                            click,
                            message=detail,
                            record=failed,
                        ),
                        False,
                        True,
                        0,
                        new_refresh,
                    )
                return (
                    *_toast(detail, ok=False),
                    no_update,
                    no_update,
                    no_update,
                    _action_result(action, click, message=detail),
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )

        if state.mode == "local":
            if _local_run_active(runner, registry):
                detail = "A local run is already active; Cancel it first."
                return (
                    *_toast(detail, ok=False),
                    no_update,
                    no_update,
                    no_update,
                    _action_result(action, click, message=detail),
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )
            try:
                argv = _local_argv_for(state)
                record = registry.allocate(
                    mode="local",
                    output_dir=output_dir,
                    rel_path=rel_path,
                    command_digest=_command_digest(state),
                    status="queued",
                )
                local_generation = record.generation
                if local_generation is None:  # pragma: no cover
                    raise RuntimeError("allocated local run has no generation")
                run_id = record.run_id

                def _observe_local_exit(_handle: Any, returncode: int) -> None:
                    registry.observe_local_exit(
                        run_id,
                        local_generation,
                        returncode,
                    )

                handle = runner.start(
                    run_id,
                    argv,
                    output_dir=output_dir,
                    generation=local_generation,
                    on_exit=_observe_local_exit,
                )
                registry.compare_and_set(
                    run_id,
                    local_generation,
                    expected_statuses={"queued"},
                    status="running",
                    pid=handle.process.pid,
                    log_paths=(handle.stdout_log_path,),
                )
                message = f"Local run started: {rel_path}"
                return (
                    *_toast(message, ok=True),
                    run_id,
                    rel_path,
                    _run_receipt(record),
                    _action_result(
                        action,
                        click,
                        message=message,
                        record=record,
                    ),
                    False,
                    False,
                    0,
                    new_refresh,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Local run start failed")
                detail = _format_exception(exc)
                if record is not None and record.generation is not None:
                    registry.compare_and_set(
                        record.run_id,
                        record.generation,
                        expected_statuses={"queued", "running"},
                        status="failed",
                        status_detail=detail,
                    )
                    failed = registry.get(record.run_id) or record
                    return (
                        *_toast(detail, ok=False),
                        record.run_id,
                        rel_path,
                        _run_receipt(record),
                        _action_result(
                            action,
                            click,
                            message=detail,
                            record=failed,
                        ),
                        False,
                        True,
                        0,
                        new_refresh,
                    )
                return (
                    *_toast(detail, ok=False),
                    no_update,
                    no_update,
                    no_update,
                    _action_result(action, click, message=detail),
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )

        record = None
        try:
            _require_slurm_request(state)
            record = registry.allocate(
                mode="slurm",
                output_dir=output_dir,
                rel_path=rel_path,
                command_digest=_command_digest(state),
                status="submitting",
            )
            slurm_generation = record.generation
            if slurm_generation is None:  # pragma: no cover
                raise RuntimeError("allocated SLURM run has no generation")
            future = _SLURM_EXECUTOR.submit(
                slurm_submitter,
                state,
                sandbox_root=sandbox.root,
                record_generation=slurm_generation,
            )
            _track_pending_slurm(
                record.run_id,
                slurm_generation,
                future,
                registry=registry,
                observer=slurm_observer,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("SLURM submitter startup failed")
            detail = _format_exception(exc)
            if record is not None and record.generation is not None:
                registry.compare_and_set(
                    record.run_id,
                    record.generation,
                    expected_statuses={"submitting"},
                    status="failed",
                    status_detail=detail,
                )
                failed = registry.get(record.run_id) or record
                return (
                    *_toast(detail, ok=False),
                    record.run_id,
                    rel_path,
                    _run_receipt(record),
                    _action_result(
                        action,
                        click,
                        message=detail,
                        record=failed,
                    ),
                    False,
                    True,
                    0,
                    new_refresh,
                )
            return (
                *_toast(detail, ok=False),
                no_update,
                no_update,
                no_update,
                _action_result(action, click, message=detail),
                no_update,
                no_update,
                no_update,
                no_update,
            )

        message = f"SLURM submitting: {rel_path}"
        return (
            *_toast(
                message,
                ok=True,
                header="Submitting…",
            ),
            record.run_id,
            rel_path,
            _run_receipt(record),
            _action_result(
                action,
                click,
                message=message,
                record=record,
            ),
            False,
            True,
            0,
            new_refresh,
        )

    @app.callback(
        Output(ids.RC_TOAST, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "children", allow_duplicate=True),
        Output(ids.RC_TOAST, "icon", allow_duplicate=True),
        Output(ids.RC_TOAST, "header", allow_duplicate=True),
        Output(ids.RC_STORE_RECENTS_REFRESH, "data", allow_duplicate=True),
        Input(ids.RC_BTN_CANCEL, "n_clicks"),
        State(ids.RC_STORE_ACTIVE_RUN_RECEIPT, "data"),
        State(ids.RC_STORE_RECENTS_REFRESH, "data"),
        prevent_initial_call=True,
    )
    def click_cancel(
        n_clicks: Optional[int],
        receipt_payload: object,
        refresh_count: Optional[int],
    ) -> Tuple[Any, ...]:
        """Fence local or scheduler work and wait for verified quiescence."""
        if not n_clicks:
            return (no_update,) * 5
        try:
            record = _record_for_receipt(registry, receipt_payload)
            if record is None or record.generation is None:
                return (
                    *_toast(
                        "The selected run generation is no longer current",
                        ok=False,
                    ),
                    no_update,
                )
            run_id = record.run_id
            new_refresh = (refresh_count or 0) + 1
            if record.status in _TERMINAL_RUN_STATUSES:
                return (
                    *_toast(
                        f"{run_id} is already {record.status}",
                        ok=False,
                    ),
                    no_update,
                )
            if record.mode != "slurm":
                stopped = runner.stop(
                    run_id,
                    generation=record.generation,
                )
                if stopped:
                    registry.compare_and_set(
                        run_id,
                        record.generation,
                        expected_statuses={"queued", "running", "unknown"},
                        status="cancelled",
                        terminal_at=datetime.now(timezone.utc),
                    )
                    return (
                        *_toast(f"Cancelled {run_id}", ok=True),
                        new_refresh,
                    )
                return (
                    *_toast(f"No live run for {run_id}", ok=False),
                    no_update,
                )

            marked_cancelling = registry.compare_and_set(
                run_id,
                record.generation,
                expected_statuses={
                    "submitting",
                    "queued",
                    "running",
                    "reconciling",
                    "unknown",
                },
                status="cancelling",
                status_detail="cancellation requested; awaiting scheduler quiescence",
            )
            if not marked_cancelling:
                return (
                    *_toast(f"No live run for {run_id}", ok=False),
                    no_update,
                )
            record = registry.get(run_id)
            binding = (
                slurm_observer.proven_binding(record)
                if record is not None
                else None
            )
            if record is not None and binding is not None:
                cancelled_jobs = _cancel_bound_generation(
                    record,
                    binding,
                )
                return (
                    *_toast(
                        f"Cancellation fenced for {run_id}; "
                        f"{len(cancelled_jobs)} scheduler job(s) signalled",
                        ok=True,
                        header="Cancelling…",
                    ),
                    new_refresh,
                )
            if (
                record is not None
                and record.generation is not None
                and _cancel_pending_slurm(run_id, record.generation)
            ):
                registry.compare_and_set(
                    run_id,
                    record.generation,
                    expected_statuses={"cancelling"},
                    status="cancelled",
                    terminal_at=datetime.now(timezone.utc),
                    status_detail="submission cancelled before it started",
                )
                return (
                    *_toast(
                        f"Cancelled {run_id} before submission",
                        ok=True,
                    ),
                    new_refresh,
                )
            return (
                *_toast(
                    "Cancellation requested; waiting for the submitter "
                    "to publish its scheduler epoch",
                    ok=True,
                    header="Cancelling…",
                ),
                new_refresh,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Cancel failed")
            return (
                *_toast(_format_exception(exc), ok=False),
                no_update,
            )

    @app.callback(
        Output(ids.RC_BTN_CANCEL, "disabled"),
        Input(ids.RC_INTERVAL_LOG, "n_intervals"),
        Input(ids.RC_STORE_ACTIVE_RUN_RECEIPT, "data"),
    )
    def update_cancel_disabled(
        _n: Optional[int],
        receipt_payload: object,
    ) -> bool:
        """Disable Cancel unless the exact displayed generation is live."""
        record = _record_for_receipt(registry, receipt_payload)
        return (
            record is None
            or record.generation is None
            or record.status in _TERMINAL_RUN_STATUSES
        )

    # ----------------------------------------------------------------------
    # 9. Dashboard polling — toggle iframe src once dashboard.html exists.
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.RC_IFRAME, "src", allow_duplicate=True),
        Output(ids.RC_IFRAME, "style", allow_duplicate=True),
        Output(ids.RC_IFRAME_PLACEHOLDER, "style", allow_duplicate=True),
        Output(ids.RC_IFRAME_PLACEHOLDER, "children", allow_duplicate=True),
        Output(
            ids.RC_INTERVAL_DASHBOARD_POLL, "disabled", allow_duplicate=True
        ),
        Input(ids.RC_INTERVAL_DASHBOARD_POLL, "n_intervals"),
        Input(ids.RC_BTN_REFRESH_DASHBOARD, "n_clicks"),
        State(ids.RC_STORE_ACTIVE_REL_PATH, "data"),
        State(ids.RC_STORE_ACTIVE_RUN_RECEIPT, "data"),
        prevent_initial_call=True,
    )
    def poll_dashboard(
        _n: Optional[int],
        _refresh_clicks: Optional[int],
        rel_path: Optional[str],
        receipt_payload: object,
    ) -> Tuple[Any, ...]:
        """Wait for ``dashboard.html`` to land then point the iframe at it."""
        if not rel_path:
            return (no_update,) * 5
        record = _record_for_receipt(registry, receipt_payload)
        if record is None or record.rel_path != rel_path:
            return (no_update,) * 5
        try:
            target = sandbox.resolve(
                Path(rel_path) / DELIVERABLES_DIRNAME / DASHBOARD_FILENAME
            )
        except ValueError:
            return (no_update,) * 5
        if not target.is_file():
            if record.status in _TERMINAL_RUN_STATUSES:
                detail = record.status_detail or "No dashboard was published."
                return (
                    no_update,
                    {"display": "none"},
                    {"display": "flex"},
                    (
                        f"Run status: {record.status}. {detail} "
                        "Use Refresh to check again."
                    ),
                    True,
                )
            return (no_update,) * 5
        return (
            _dashboard_url(rel_path, url_prefix=server_url_prefix),
            {"display": "block"},
            {"display": "none"},
            no_update,
            True,
        )

    # ----------------------------------------------------------------------
    # 10. Log tail polling
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.RC_LOG_TAIL, "children"),
        Output(ids.RC_STATUS_BANNER, "children"),
        Input(ids.RC_INTERVAL_LOG, "n_intervals"),
        State(ids.RC_STORE_ACTIVE_RUN_RECEIPT, "data"),
    )
    def update_log_tail(
        _n: Optional[int], receipt_payload: object
    ) -> Tuple[Any, str]:
        """Render the last N log lines + a status banner string."""
        if not receipt_payload:
            return "(no log yet)", "(no active run)"
        record = _record_for_receipt(registry, receipt_payload)
        if record is None or record.generation is None:
            return (
                "(generation is no longer current)",
                "(selected generation is no longer current)",
            )
        run_id = record.run_id
        generation_label = str(record.generation)
        if record.mode == "slurm":
            text = slurm_log_cache.read(record)
            banner = (
                f"slurm | {record.rel_path} | "
                f"generation={generation_label} | status={record.status}"
            )
            if record.status_detail:
                banner += f" | {record.status_detail}"
            return text, banner
        lines = runner.snapshot_log(
            run_id,
            generation=record.generation,
            tail=200,
        )
        text = "".join(lines) if lines else "(waiting for first output...)"
        running = runner.is_running(
            run_id,
            generation=record.generation,
        )
        status = "running" if running else record.status
        banner = (
            f"{record.mode} | {record.rel_path} | "
            f"generation={generation_label} | status={status}"
        )
        if record.status_detail:
            banner += f" | {record.status_detail}"
        return text, banner

    # ----------------------------------------------------------------------
    # 10b. Resolve pending SLURM submissions (non-blocking submit follow-up)
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.RC_TOAST, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "children", allow_duplicate=True),
        Output(ids.RC_TOAST, "icon", allow_duplicate=True),
        Output(ids.RC_TOAST, "header", allow_duplicate=True),
        Output(ids.RC_STORE_RECENTS_REFRESH, "data", allow_duplicate=True),
        Input(ids.RC_INTERVAL_LOG, "n_intervals"),
        State(ids.RC_STORE_ACTIVE_RUN_RECEIPT, "data"),
        State(ids.RC_STORE_RECENTS_REFRESH, "data"),
        prevent_initial_call=True,
    )
    def resolve_pending_slurm(
        _n: Optional[int],
        receipt_payload: object,
        refresh_count: Optional[int],
    ) -> Tuple[Any, ...]:
        """Surface a lifecycle result already committed by the future callback."""
        record = _record_for_receipt(registry, receipt_payload)
        if (
            record is None
            or record.mode != "slurm"
            or record.generation is None
        ):
            return (no_update,) * 5
        completion = _take_slurm_completion(
            record.run_id,
            record.generation,
        )
        if completion is None:
            return (no_update,) * 5
        new_refresh = (refresh_count or 0) + 1
        if completion.error is not None:
            return (
                *_toast(completion.error, ok=False),
                new_refresh,
            )
        if completion.pending is not None:
            return (
                *_toast(
                    completion.pending,
                    ok=True,
                    header="Reconciling submission…",
                ),
                new_refresh,
            )
        result = completion.result
        if result is None:  # pragma: no cover - dataclass invariant
            return (no_update,) * 5
        return (
            *_toast(
                f"SLURM submitted ({result.job_id}): {record.rel_path}",
                ok=True,
            ),
            new_refresh,
        )

    # ----------------------------------------------------------------------
    # 11. Recent Runs panel — refresh + row click
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.RC_STORE_RECENTS_REFRESH, "data", allow_duplicate=True),
        Input(ids.RC_INTERVAL_LOG, "n_intervals"),
        State(ids.RC_STORE_RECENTS_REFRESH, "data"),
        prevent_initial_call=True,
    )
    def publish_registry_revision(
        _n: Optional[int],
        current_revision: object,
    ) -> Any:
        """Publish lifecycle revisions without scanning the sandbox."""
        revision = registry.revision
        return revision if current_revision != revision else no_update

    @app.callback(
        Output(ids.RC_RECENTS_BODY, "children"),
        Output(ids.RC_DROPDOWN_LOAD_PRESET, "options"),
        Input(ids.RC_STORE_RECENTS_REFRESH, "data"),
        prevent_initial_call=True,
    )
    def refresh_recents(_refresh: Any) -> Tuple[List, List[dict[str, str]]]:
        """Rebuild the Recent Runs table (and preset dropdown options).

        Triggered ONLY by ``RC_STORE_RECENTS_REFRESH`` bumps from
        Run/Validate/Cancel. Was previously also wired to
        ``RC_INTERVAL_LOG``, which forced a sandbox walk every second
        regardless of whether anything changed; large sandboxes burn
        real CPU on that. The layout pre-renders one table at boot
        (``_layout._recents_panel``), so dropping the interval input
        does not leave the panel empty.
        """
        rows = scan_recent_runs(sandbox, registry=registry)
        return render_recents_table(rows), _list_preset_options(sandbox)

    @app.callback(
        Output(ids.RC_IFRAME, "src", allow_duplicate=True),
        Output(ids.RC_IFRAME, "style", allow_duplicate=True),
        Output(ids.RC_IFRAME_PLACEHOLDER, "style", allow_duplicate=True),
        Output(ids.RC_STORE_ACTIVE_REL_PATH, "data", allow_duplicate=True),
        Input({"type": "rc-recents-row", "rel_path": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def click_recents_row(_clicks: List[int]) -> Tuple[Any, ...]:
        """Re-point the iframe at a clicked recent-run dashboard."""
        triggered = ctx.triggered_id
        if (
            not isinstance(triggered, dict)
            or triggered.get("type") != "rc-recents-row"
        ):
            return (no_update,) * 4
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return (no_update,) * 4
        rel_path = triggered.get("rel_path")
        if not isinstance(rel_path, str):
            return (no_update,) * 4
        # Only point at a real dashboard.html — clicking a row whose run
        # never produced a dashboard does nothing visible.
        try:
            target = sandbox.resolve(
                Path(rel_path) / DELIVERABLES_DIRNAME / DASHBOARD_FILENAME
            )
        except ValueError:
            return (no_update,) * 4
        if not target.is_file():
            return (no_update,) * 4
        return (
            _dashboard_url(rel_path, url_prefix=server_url_prefix),
            {"display": "block"},
            {"display": "none"},
            rel_path,
        )

    # ----------------------------------------------------------------------
    # 12. Save preset / Load preset
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.RC_TOAST, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "children", allow_duplicate=True),
        Output(ids.RC_TOAST, "icon", allow_duplicate=True),
        Output(ids.RC_TOAST, "header", allow_duplicate=True),
        Input(ids.RC_BTN_SAVE_PRESET, "n_clicks"),
        State(ids.RC_INPUT_PRESET_NAME, "value"),
        *_action_control_states(),
        prevent_initial_call=True,
    )
    def click_save_preset(
        n_clicks: Optional[int],
        name: Optional[str],
        *control_values: Any,
    ) -> Tuple[Any, ...]:
        """Write the current form state to ``presets/<name>.json``."""
        if not n_clicks:
            return (no_update,) * 4
        if not name or not name.strip():
            return _toast("Name the preset first", ok=False)
        safe_name = "".join(
            c for c in name.strip() if c.isalnum() or c in "-_"
        )
        if not safe_name:
            return _toast("Invalid preset name", ok=False)
        try:
            target = _presets_dir(sandbox) / f"{safe_name}.json"
            state = _state_from_action_controls(
                tuple(control_values), sandbox=sandbox
            )
            payload = run_state_to_json(state)
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return _toast(f"Saved preset {safe_name}", ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Save preset failed")
            return _toast(_format_exception(exc), ok=False)

    @app.callback(
        *_action_control_outputs(),
        Output(ids.RC_TOAST, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "children", allow_duplicate=True),
        Output(ids.RC_TOAST, "icon", allow_duplicate=True),
        Output(ids.RC_TOAST, "header", allow_duplicate=True),
        Input(ids.RC_DROPDOWN_LOAD_PRESET, "value"),
        prevent_initial_call=True,
    )
    def click_load_preset(preset_path: Optional[str]) -> Tuple[Any, ...]:
        """Restore every authoritative visible control from a preset file."""
        if not preset_path:
            return (no_update,) * 26
        try:
            payload = json.loads(Path(preset_path).read_text(encoding="utf-8"))
            state = run_state_from_json(payload)
            return (
                *_controls_from_run_state(state, sandbox=sandbox),
                *_toast(f"Loaded preset {Path(preset_path).stem}", ok=True),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Load preset failed")
            return (
                *((no_update,) * 22),
                *_toast(_format_exception(exc), ok=False),
            )

    # ----------------------------------------------------------------------
    # 13. Run button concurrency cap
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.RC_BTN_RUN, "disabled"),
        Input(ids.RC_INTERVAL_LOG, "n_intervals"),
        Input(ids.RC_STORE_ACTIVE_RUN_ID, "data"),
        Input(ids.RC_RADIO_MODE, "value"),
    )
    def update_run_disabled(
        _n: Optional[int],
        _active: Optional[str],
        mode: Optional[RunMode],
    ) -> bool:
        """Disable Run while a Local run is active (SLURM is unconstrained)."""
        if mode == "slurm":
            return False
        return _local_run_active(runner, registry)

    # ----------------------------------------------------------------------
    # 14. Sidebar hand-off banner — consume SHELL_SIDEBAR_SELECTION_STORE.
    # ----------------------------------------------------------------------

    from phenotypic.gui.shell._ids import SHELL_SIDEBAR_SELECTION_STORE

    @app.callback(
        Output(ids.RC_HANDOFF_BANNER, "style"),
        Output(ids.RC_HANDOFF_LABEL, "children"),
        Output(ids.RC_HANDOFF_USE_PIPELINE, "disabled"),
        Output(ids.RC_HANDOFF_USE_INPUT, "disabled"),
        Output(ids.RC_HANDOFF_USE_OUTPUT, "disabled"),
        Input(SHELL_SIDEBAR_SELECTION_STORE, "data"),
        prevent_initial_call=True,
    )
    def populate_handoff_banner(
        selection: Optional[dict[str, Any]],
    ) -> Tuple[Any, ...]:
        """Reflect the sidebar selection in the hand-off banner.

        Buttons are enabled/disabled based on the selection's
        capabilities:

        * **Set as pipeline** — enabled when the path looks like a pipeline
          config (``has_pipeline_json`` capability OR a matching config suffix).
        * **Set as input dir** — enabled for any directory.
        * **Set as output dir** — enabled for any directory (a CLI
          output dir or a fresh path the user typed).

        The banner hides itself when the selection is empty.
        """
        hidden_style = {"display": "none"}
        if not selection or not isinstance(selection, dict):
            return hidden_style, "(none)", True, True, True
        path = selection.get("path") or ""
        if not path:
            return hidden_style, "(none)", True, True, True

        caps = selection.get("capabilities") or {}
        is_dir = bool(selection.get("is_dir"))
        looks_like_json = caps.get(
            "has_pipeline_json", False
        ) or matches_any_suffix(path, PIPELINE_CONFIG_SUFFIXES)

        return (
            {"display": "flex"},
            path,
            not looks_like_json,
            not is_dir,
            not is_dir,
        )

    @app.callback(
        Output(ids.RC_STORE_PIPELINE_PATH, "data", allow_duplicate=True),
        Output(ids.RC_STORE_INPUT_DIR, "data", allow_duplicate=True),
        Output(ids.RC_INPUT_OUTPUT_PATH, "value", allow_duplicate=True),
        Output(ids.RC_STORE_OUTPUT_DIR, "data", allow_duplicate=True),
        Output(
            ids.RC_STORE_OUTPUT_CONFIRMATION,
            "data",
            allow_duplicate=True,
        ),
        Output(ids.RC_MODAL_OUTPUT, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "children", allow_duplicate=True),
        Output(ids.RC_TOAST, "icon", allow_duplicate=True),
        Output(ids.RC_TOAST, "header", allow_duplicate=True),
        Output(SHELL_SIDEBAR_SELECTION_STORE, "data", allow_duplicate=True),
        Input(ids.RC_HANDOFF_USE_PIPELINE, "n_clicks"),
        Input(ids.RC_HANDOFF_USE_INPUT, "n_clicks"),
        Input(ids.RC_HANDOFF_USE_OUTPUT, "n_clicks"),
        Input(ids.RC_HANDOFF_DISMISS, "n_clicks"),
        State(SHELL_SIDEBAR_SELECTION_STORE, "data"),
        prevent_initial_call=True,
    )
    def click_handoff_button(
        _pipe: Optional[int],
        _input: Optional[int],
        _output: Optional[int],
        _dismiss: Optional[int],
        selection: Optional[dict[str, Any]],
    ) -> Tuple[Any, ...]:
        """Route the sidebar selection into a form field.

        ``ctx.triggered_id`` distinguishes which button fired. Output
        selections populate the typed field and reopen confirmation; they
        never become an accepted output target directly.
        """
        triggered = ctx.triggered_id
        if triggered is None:
            return (no_update,) * 11

        if triggered == ids.RC_HANDOFF_DISMISS:
            # ``no_update`` for the four toast outputs so an unrelated
            # toast (e.g. a still-visible "Saved preset" confirmation)
            # is not torn down as a side-effect of dismissing the
            # hand-off banner.
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                None,
            )

        path = (selection or {}).get("path") if selection else None
        abs_path = (selection or {}).get("abs_path") if selection else None
        if not path:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                *_toast("No sidebar selection", ok=False),
                no_update,
            )

        # The form stores carry absolute paths so the CLI gets a fully-
        # resolved argv tail; fall back to the rel path if the chrome
        # didn't stamp the absolute one.
        target = abs_path or path

        if triggered == ids.RC_HANDOFF_USE_PIPELINE:
            return (
                target,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                *_toast(f"Set as pipeline: {path}", ok=True),
                None,
            )
        if triggered == ids.RC_HANDOFF_USE_INPUT:
            return (
                no_update,
                target,
                no_update,
                no_update,
                no_update,
                no_update,
                *_toast(f"Set as input dir: {path}", ok=True),
                None,
            )
        if triggered == ids.RC_HANDOFF_USE_OUTPUT:
            return (
                no_update,
                no_update,
                target,
                None,
                None,
                True,
                *_toast(
                    f"Review and confirm output path: {path}",
                    ok=True,
                ),
                None,
            )
        return (no_update,) * 11
