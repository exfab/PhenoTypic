"""Run console Dash callbacks.

Wires the Run console's UI affordances into the LocalRunner / SLURM
submitter / RunRegistry plumbing. One callback per logical effect — fan-in
via ``ctx.triggered_id`` is used only for the dir-tree pattern-matching
inputs and for the action-button row that needs to feed multiple outputs.

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

import json
import logging
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import dash
from dash import ALL, Input, Output, State, ctx, no_update

from phenotypic.gui._config import (
    DASHBOARD_FILENAME,
    DELIVERABLES_DIRNAME,
    RUNS_BLUEPRINT_PREFIX,
    SANDBOX_GUI_DIRNAME,
    SANDBOX_PRESETS_SUBDIR,
    THREAD_NAME_PREFIX,
)
from phenotypic.gui.run_console import _ids as ids
from phenotypic.gui.run_console._directory_picker import (
    ensure_output_dir,
    render_output_dir_tree,
)
from phenotypic.gui.run_console._form import (
    render_input_tree,
    render_pipeline_tree,
)
from phenotypic.gui.run_console._layout import render_recents_table
from phenotypic.gui.run_console._recent_runs import scan_recent_runs
from phenotypic.gui.run_console._runner import LocalRunner
from phenotypic.gui.shell._ids import (
    SHELL_METADATA_CSV_STORE,
    SHELL_SOURCE_IMAGE_ROOT_STORE,
)
from phenotypic.gui.shell._metadata_context import resolve_metadata_csv
from phenotypic.gui.shell._runs_registry import RunMode, RunRecord, RunRegistry
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._source_context import (
    SourcePayload,
    resolve_source_image_root,
    source_payload_from_path,
)
from phenotypic.tools_ import PIPELINE_CONFIG_SUFFIXES, matches_any_suffix

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


def _trigger_kind_path(triggered: Any, expected_type: str) -> Optional[Tuple[str, str]]:
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


def _source_payload_for_input_dir(
    sandbox: SandboxRoot,
    input_dir: Optional[str],
    current_payload: object,
) -> SourcePayload | None:
    """Build a shared source payload from the Run input directory."""
    if not input_dir:
        return None
    payload = source_payload_from_path(
        sandbox, input_dir, source="run-console"
    )
    if payload is None:
        return None
    if (
        isinstance(current_payload, dict)
        and current_payload.get("abs_path") == payload["abs_path"]
    ):
        return None
    return payload


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


# ---------------------------------------------------------------------------
# Stream B seams — RunConsoleState + SLURM submitter.
# ---------------------------------------------------------------------------

from phenotypic.gui.run_console._slurm import (  # noqa: E402
    SlurmSubmitError,
    SlurmSubmitResult,
    submit_slurm,
)
from phenotypic.gui.run_console._state import (  # noqa: E402
    RunConsoleState,
    run_state_from_json,
    run_state_to_json,
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
    if sandbox is not None:
        resolved_metadata = resolve_metadata_csv(sandbox, metadata_payload)
        metadata_csv = str(resolved_metadata) if resolved_metadata is not None else None

    return {
        "pipeline_path": pipeline_path,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "metadata_csv": metadata_csv,
        "mode": mode or "local",
        "dry_run": "dry_run" in flag_set,
        "resume": "resume" in flag_set,
        "save_inspect": "save_inspect" in flag_set,
        "advanced_args": {k: v for k, v in advanced.items() if v not in (None, "")},
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
        if runner.is_running(record.run_id):
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
_PENDING_SLURM: dict[str, Future[Any]] = {}
_PENDING_SLURM_LOCK = threading.Lock()


def _stash_pending_slurm(transient_id: str, future: "Future[Any]") -> None:
    """Record a pending SLURM submission keyed by its transient run id."""
    with _PENDING_SLURM_LOCK:
        _PENDING_SLURM[transient_id] = future


def _take_pending_slurm(transient_id: str) -> "Future[Any] | None":
    """Pop a pending submission; return ``None`` if unknown or still in flight.

    Returns the future ONLY when it has completed (success or exception).
    Callers should call ``future.result()`` afterwards to surface the
    outcome.
    """
    with _PENDING_SLURM_LOCK:
        future = _PENDING_SLURM.get(transient_id)
        if future is None or not future.done():
            return None
        # Pop only after we've decided to handle it — avoids losing
        # the future on a transient race with a parallel poll.
        return _PENDING_SLURM.pop(transient_id, None)


def _has_pending_slurm() -> bool:
    """True iff at least one pending submission is registered (any state)."""
    with _PENDING_SLURM_LOCK:
        return bool(_PENDING_SLURM)


# ---------------------------------------------------------------------------
# Dashboard polling helper
# ---------------------------------------------------------------------------


def _dashboard_url(rel_path: str) -> str:
    """Build the iframe ``src`` for ``rel_path``.

    The shell mounts ``/runs/<rel>/<file>`` regardless of the Dash sub-app's
    ``url_prefix`` so we always use the absolute ``/runs/...`` path. The
    dashboard now lives under the run's ``deliverables/`` subdirectory.
    """
    safe_rel = rel_path.strip("/").replace("\\", "/")
    return f"{RUNS_BLUEPRINT_PREFIX}/{safe_rel}/{DELIVERABLES_DIRNAME}/{DASHBOARD_FILENAME}"


# ---------------------------------------------------------------------------
# register_callbacks
# ---------------------------------------------------------------------------


def register_callbacks(
    app: dash.Dash,
    sandbox: SandboxRoot,
    *,
    registry: RunRegistry,
    runner: LocalRunner,
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
    """

    # ----------------------------------------------------------------------
    # 1. Picker buttons → open / cancel modals
    # ----------------------------------------------------------------------
    # Each picker button (Pipeline / Input / Output) opens its modal; each
    # Cancel button closes it. The wiring is mechanical — register the six
    # callbacks in a loop so the per-modal IDs read top-to-bottom.

    _modal_buttons: list[tuple[str, str, str]] = [
        (ids.RC_MODAL_PIPELINE, ids.RC_BTN_PICK_PIPELINE, ids.RC_BTN_PIPELINE_CANCEL),
        (ids.RC_MODAL_INPUT, ids.RC_BTN_PICK_INPUT, ids.RC_BTN_INPUT_CANCEL),
        (ids.RC_MODAL_OUTPUT, ids.RC_BTN_PICK_OUTPUT, ids.RC_BTN_OUTPUT_CANCEL),
    ]

    def _register_modal_toggle(modal_id: str, button_id: str, *, open_value: bool) -> None:
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
            return (no_update, no_update, *_toast("Pick a folder first", ok=False))
        chosen = Path(dir_value)
        # Cheap "any image files?" probe — count entries with an image-ish
        # extension in the depth-1 listing.
        try:
            sample = [
                p
                for p in chosen.iterdir()
                if p.is_file()
                and p.suffix.lower()
                in {
                    ".png",
                    ".tif",
                    ".tiff",
                    ".jpg",
                    ".jpeg",
                    ".raw",
                    ".nef",
                    ".cr2",
                    ".arw",
                    ".dng",
                }
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
        Output(ids.RC_MODAL_OUTPUT, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "children", allow_duplicate=True),
        Output(ids.RC_TOAST, "icon", allow_duplicate=True),
        Output(ids.RC_TOAST, "header", allow_duplicate=True),
        Input(ids.RC_BTN_OUTPUT_CONFIRM, "n_clicks"),
        State(ids.RC_INPUT_OUTPUT_PATH, "value"),
        State(ids.RC_STORE_BROWSE_DIR_OUTPUT, "data"),
        prevent_initial_call=True,
    )
    def confirm_output_dir(
        n_clicks: Optional[int],
        typed_value: Optional[str],
        browsed_value: Optional[str],
    ) -> Tuple[Any, ...]:
        """Resolve + create the output directory and close the modal."""
        if not n_clicks:
            return (no_update,) * 6
        candidate = (typed_value or browsed_value or "").strip()
        if not candidate:
            return (
                no_update,
                no_update,
                *_toast("Type or browse a directory first", ok=False),
            )
        resolved = ensure_output_dir(sandbox, candidate)
        if resolved is None:
            return (
                no_update,
                no_update,
                *_toast(
                    f"Refused: {candidate} escapes sandbox or cannot be created",
                    ok=False,
                ),
            )
        return (
            str(resolved),
            False,
            *_toast(f"Output: {resolved.name}", ok=True),
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
    # 5b. Shared source-image-root sync
    # ----------------------------------------------------------------------

    @app.callback(
        Output(SHELL_SOURCE_IMAGE_ROOT_STORE, "data", allow_duplicate=True),
        Input(ids.RC_STORE_INPUT_DIR, "data"),
        State(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
        prevent_initial_call=True,
    )
    def _mirror_input_dir_to_shared_source(
        input_dir: Optional[str],
        current_payload: object,
    ) -> Any:
        payload = _source_payload_for_input_dir(
            sandbox, input_dir, current_payload
        )
        return payload if payload is not None else no_update

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

    _register_collapse_toggle(ids.RC_COLLAPSE_ADVANCED, ids.RC_BTN_TOGGLE_ADVANCED)
    _register_collapse_toggle(ids.RC_COLLAPSE_SLURM, ids.RC_BTN_TOGGLE_SLURM)

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
        Input(SHELL_METADATA_CSV_STORE, "data"),
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
        metadata_payload: object,
    ) -> dict[str, Any]:
        """Bundle all form fields into the run-state store on any change."""
        return _form_inputs_to_state(
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
            sandbox=sandbox,
        )

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
        Output(ids.RC_INTERVAL_LOG, "disabled", allow_duplicate=True),
        Input(ids.RC_BTN_VALIDATE, "n_clicks"),
        State(ids.RC_STORE_FORM_STATE, "data"),
        prevent_initial_call=True,
    )
    def click_validate(
        n_clicks: Optional[int], form_state: dict[str, Any]
    ) -> Tuple[Any, ...]:
        """Run the pipeline with ``--dry-run`` for validation.

        Clears ``RC_STORE_ACTIVE_REL_PATH`` so the dashboard-poll callback
        does not try to fetch a stale dashboard.html from a previous run
        while validation is in flight (validate runs do not produce a
        dashboard).
        """
        if not n_clicks:
            return (no_update,) * 7
        try:
            state_dict = dict(form_state or {})
            state_dict["dry_run"] = True
            state = run_state_from_json(state_dict)
            argv = _local_argv_for(state)
            if state.output_dir is None:
                raise ValueError("output_dir is required")
            output_dir = Path(state.output_dir)
            run_id = f"validate-{int(time.time() * 1000)}"
            runner.start(run_id, argv, output_dir=output_dir)
            # ``mode="validate"`` is intentionally distinct from ``"local"``
            # so the run-button concurrency cap (``_local_run_active``)
            # does NOT block a real run while a dry-run probe is alive.
            registry.register(
                RunRecord(
                    run_id=run_id,
                    mode="validate",
                    output_dir=output_dir,
                    rel_path=str(
                        output_dir.relative_to(sandbox.root)
                        if sandbox.contains(output_dir)
                        else output_dir
                    ),
                    status="running",
                )
            )
            return (
                *_toast("Validation (dry-run) started", ok=True),
                run_id,
                None,  # Clear active rel_path so dashboard poll stays idle.
                False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Validate failed")
            return (
                *_toast(_format_exception(exc), ok=False),
                no_update,
                no_update,
                no_update,
            )

    @app.callback(
        Output(ids.RC_TOAST, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "children", allow_duplicate=True),
        Output(ids.RC_TOAST, "icon", allow_duplicate=True),
        Output(ids.RC_TOAST, "header", allow_duplicate=True),
        Output(ids.RC_STORE_ACTIVE_RUN_ID, "data", allow_duplicate=True),
        Output(ids.RC_STORE_ACTIVE_REL_PATH, "data", allow_duplicate=True),
        Output(ids.RC_INTERVAL_LOG, "disabled", allow_duplicate=True),
        Output(ids.RC_INTERVAL_DASHBOARD_POLL, "disabled", allow_duplicate=True),
        Output(ids.RC_INTERVAL_DASHBOARD_POLL, "n_intervals", allow_duplicate=True),
        Output(ids.RC_BTN_CANCEL, "disabled", allow_duplicate=True),
        Output(ids.RC_STORE_RECENTS_REFRESH, "data", allow_duplicate=True),
        Input(ids.RC_BTN_RUN, "n_clicks"),
        State(ids.RC_STORE_FORM_STATE, "data"),
        State(ids.RC_STORE_RECENTS_REFRESH, "data"),
        prevent_initial_call=True,
    )
    def click_run(
        n_clicks: Optional[int],
        form_state: dict[str, Any],
        refresh_count: Optional[int],
    ) -> Tuple[Any, ...]:
        """Spawn a Local or SLURM run from the form state."""
        if not n_clicks:
            return (no_update,) * 11
        if not form_state:
            return (
                *_toast("Form is empty", ok=False),
                *((no_update,) * 7),
            )
        state = run_state_from_json(form_state)
        if state.output_dir is None:
            return (
                *_toast("Output directory not set", ok=False),
                *((no_update,) * 7),
            )
        output_dir = Path(state.output_dir)

        try:
            rel_path = str(output_dir.relative_to(sandbox.root))
        except ValueError:
            return (
                *_toast(
                    f"Refused: output {output_dir} escapes sandbox",
                    ok=False,
                ),
                *((no_update,) * 7),
            )

        new_refresh = (refresh_count or 0) + 1

        if state.mode == "local":
            if _local_run_active(runner, registry):
                return (
                    *_toast(
                        "A local run is already active -- Cancel it first.",
                        ok=False,
                    ),
                    *((no_update,) * 7),
                )
            try:
                argv = _local_argv_for(state)
                run_id = rel_path
                # Reap any stale handle from a previous completed run on
                # the same output dir; otherwise ``runner.start`` would
                # raise "run_id already running" (the runner's reap is
                # caller-driven — Phase 4 left it that way deliberately).
                runner.reap(run_id)
                runner.start(run_id, argv, output_dir=output_dir)
                handle = runner.get(run_id)
                pid = handle.process.pid if handle is not None else None
                registry.register(
                    RunRecord(
                        run_id=run_id,
                        mode="local",
                        output_dir=output_dir,
                        rel_path=rel_path,
                        status="running",
                        pid=pid,
                    )
                )
                return (
                    *_toast(f"Local run started: {rel_path}", ok=True),
                    run_id,
                    rel_path,
                    False,
                    False,
                    0,
                    False,
                    new_refresh,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Local run start failed")
                return (
                    *_toast(_format_exception(exc), ok=False),
                    *((no_update,) * 7),
                )

        # SLURM path. ``submit_slurm`` shells out to ``sbatch`` and can
        # block up to its 60s timeout; running it inline would freeze
        # every other callback. Offload to the module-level executor and
        # let ``resolve_pending_slurm`` (driven by the log-tail interval)
        # surface the outcome.
        transient_id = f"slurm-pending-{uuid.uuid4().hex[:8]}"
        registry.register(
            RunRecord(
                run_id=transient_id,
                mode="slurm",
                output_dir=output_dir,
                rel_path=rel_path,
                status="submitting",
            )
        )
        future = _SLURM_EXECUTOR.submit(
            submit_slurm, state, sandbox_root=sandbox.root
        )
        _stash_pending_slurm(transient_id, future)
        return (
            *_toast(
                f"SLURM submitting: {rel_path}",
                ok=True,
                header="Submitting…",
            ),
            transient_id,
            rel_path,
            False,
            True,
            0,
            False,
            new_refresh,
        )

    @app.callback(
        Output(ids.RC_TOAST, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "children", allow_duplicate=True),
        Output(ids.RC_TOAST, "icon", allow_duplicate=True),
        Output(ids.RC_TOAST, "header", allow_duplicate=True),
        Output(ids.RC_BTN_CANCEL, "disabled", allow_duplicate=True),
        Output(ids.RC_INTERVAL_LOG, "disabled", allow_duplicate=True),
        Input(ids.RC_BTN_CANCEL, "n_clicks"),
        State(ids.RC_STORE_ACTIVE_RUN_ID, "data"),
        prevent_initial_call=True,
    )
    def click_cancel(
        n_clicks: Optional[int], run_id: Optional[str]
    ) -> Tuple[Any, ...]:
        """Send SIGTERM to the active local run and update registry."""
        if not n_clicks or not run_id:
            return (no_update,) * 6
        try:
            stopped = runner.stop(run_id)
            if stopped:
                registry.update_status(run_id, "cancelled")
                return (
                    *_toast(f"Cancelled {run_id}", ok=True),
                    True,
                    True,
                )
            return (
                *_toast(f"No live run for {run_id}", ok=False),
                True,
                True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Cancel failed")
            return (
                *_toast(_format_exception(exc), ok=False),
                no_update,
                no_update,
            )

    # ----------------------------------------------------------------------
    # 9. Dashboard polling — toggle iframe src once dashboard.html exists.
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.RC_IFRAME, "src", allow_duplicate=True),
        Output(ids.RC_IFRAME, "style", allow_duplicate=True),
        Output(ids.RC_IFRAME_PLACEHOLDER, "style", allow_duplicate=True),
        Output(ids.RC_INTERVAL_DASHBOARD_POLL, "disabled", allow_duplicate=True),
        Input(ids.RC_INTERVAL_DASHBOARD_POLL, "n_intervals"),
        State(ids.RC_STORE_ACTIVE_REL_PATH, "data"),
        prevent_initial_call=True,
    )
    def poll_dashboard(
        _n: Optional[int], rel_path: Optional[str]
    ) -> Tuple[Any, ...]:
        """Wait for ``dashboard.html`` to land then point the iframe at it."""
        if not rel_path:
            return (no_update,) * 4
        try:
            target = sandbox.resolve(
                Path(rel_path) / DELIVERABLES_DIRNAME / DASHBOARD_FILENAME
            )
        except ValueError:
            return (no_update,) * 4
        if not target.is_file():
            return (no_update,) * 4
        return (
            _dashboard_url(rel_path),
            {"display": "block"},
            {"display": "none"},
            True,
        )

    # ----------------------------------------------------------------------
    # 10. Log tail polling
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.RC_LOG_TAIL, "children"),
        Output(ids.RC_STATUS_BANNER, "children"),
        Input(ids.RC_INTERVAL_LOG, "n_intervals"),
        State(ids.RC_STORE_ACTIVE_RUN_ID, "data"),
    )
    def update_log_tail(
        _n: Optional[int], run_id: Optional[str]
    ) -> Tuple[Any, str]:
        """Render the last N log lines + a status banner string."""
        if not run_id:
            return "(no log yet)", "(no active run)"
        # SLURM submissions live as ``slurm-pending-<uuid>`` until the
        # async submit resolves. Show the registry status (``submitting``)
        # in the banner; there is no log yet.
        if run_id.startswith("slurm-pending-"):
            record = registry.get(run_id)
            banner = (
                f"slurm | {record.rel_path} | status=submitting"
                if record is not None
                else f"run_id={run_id} (submitting)"
            )
            return "(SLURM submission in flight…)", banner
        lines = runner.snapshot_log(run_id, tail=200)
        text = "".join(lines) if lines else "(waiting for first output...)"
        record = registry.get(run_id)
        if record is None:
            banner = f"run_id={run_id} (not in registry)"
        else:
            running = runner.is_running(run_id)
            status = "running" if running else record.status
            banner = (
                f"{record.mode} | {record.rel_path} | status={status}"
            )
        return text, banner

    # ----------------------------------------------------------------------
    # 10b. Resolve pending SLURM submissions (non-blocking submit follow-up)
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.RC_TOAST, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "children", allow_duplicate=True),
        Output(ids.RC_TOAST, "icon", allow_duplicate=True),
        Output(ids.RC_TOAST, "header", allow_duplicate=True),
        Output(ids.RC_STORE_ACTIVE_RUN_ID, "data", allow_duplicate=True),
        Output(ids.RC_STORE_RECENTS_REFRESH, "data", allow_duplicate=True),
        Input(ids.RC_INTERVAL_LOG, "n_intervals"),
        State(ids.RC_STORE_ACTIVE_RUN_ID, "data"),
        State(ids.RC_STORE_RECENTS_REFRESH, "data"),
        prevent_initial_call=True,
    )
    def resolve_pending_slurm(
        _n: Optional[int],
        run_id: Optional[str],
        refresh_count: Optional[int],
    ) -> Tuple[Any, ...]:
        """Promote a completed pending SLURM future to a real RunRecord.

        The :func:`click_run` SLURM path returns immediately with a
        transient ``slurm-pending-<uuid>`` run id and stashes the future
        in :data:`_PENDING_SLURM`. This callback (driven once per
        ``RC_INTERVAL_LOG`` tick) checks whether the future for the
        currently-active transient id has completed and, if so, replaces
        its registry record with the real ``slurm-{job_id}`` entry and
        toasts the outcome.
        """
        if not run_id or not run_id.startswith("slurm-pending-"):
            return (no_update,) * 6
        future = _take_pending_slurm(run_id)
        if future is None:
            # Either still in flight or already handled by a prior tick.
            return (no_update,) * 6

        prior = registry.get(run_id)
        rel_path = prior.rel_path if prior is not None else ""
        output_dir = prior.output_dir if prior is not None else None
        new_refresh = (refresh_count or 0) + 1

        try:
            result: SlurmSubmitResult = future.result()
        except SlurmSubmitError as exc:
            registry.remove(run_id)
            return (
                *_toast(str(exc), ok=False),
                None,  # clear active run id
                new_refresh,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("SLURM submit (async) raised")
            registry.remove(run_id)
            return (
                *_toast(_format_exception(exc), ok=False),
                None,
                new_refresh,
            )

        real_run_id = f"slurm-{result.job_id}"
        # Drop the transient and register the real record.
        registry.remove(run_id)
        registry.register(
            RunRecord(
                run_id=real_run_id,
                mode="slurm",
                output_dir=(
                    output_dir if output_dir is not None else result.output_dir
                ),
                rel_path=rel_path,
                status="running",
                slurm_job_id=result.job_id,
            )
        )
        return (
            *_toast(
                f"SLURM submitted ({result.job_id}): {rel_path}",
                ok=True,
            ),
            real_run_id,
            new_refresh,
        )

    # ----------------------------------------------------------------------
    # 11. Recent Runs panel — refresh + row click
    # ----------------------------------------------------------------------

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
        if not isinstance(triggered, dict) or triggered.get("type") != "rc-recents-row":
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
            _dashboard_url(rel_path),
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
        State(ids.RC_STORE_FORM_STATE, "data"),
        prevent_initial_call=True,
    )
    def click_save_preset(
        n_clicks: Optional[int],
        name: Optional[str],
        form_state: dict[str, Any],
    ) -> Tuple[Any, ...]:
        """Write the current form state to ``presets/<name>.json``."""
        if not n_clicks:
            return (no_update,) * 4
        if not name or not name.strip():
            return _toast("Name the preset first", ok=False)
        safe_name = "".join(c for c in name.strip() if c.isalnum() or c in "-_")
        if not safe_name:
            return _toast("Invalid preset name", ok=False)
        try:
            target = _presets_dir(sandbox) / f"{safe_name}.json"
            state = run_state_from_json(form_state or {})
            payload = run_state_to_json(state)
            target.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            return _toast(f"Saved preset {safe_name}", ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Save preset failed")
            return _toast(_format_exception(exc), ok=False)

    @app.callback(
        Output(ids.RC_STORE_PIPELINE_PATH, "data", allow_duplicate=True),
        Output(ids.RC_STORE_INPUT_DIR, "data", allow_duplicate=True),
        Output(ids.RC_STORE_OUTPUT_DIR, "data", allow_duplicate=True),
        Output(ids.RC_RADIO_MODE, "value"),
        Output(ids.RC_CHECKS_FLAGS, "value"),
        Output(ids.RC_TOAST, "is_open", allow_duplicate=True),
        Output(ids.RC_TOAST, "children", allow_duplicate=True),
        Output(ids.RC_TOAST, "icon", allow_duplicate=True),
        Output(ids.RC_TOAST, "header", allow_duplicate=True),
        Input(ids.RC_DROPDOWN_LOAD_PRESET, "value"),
        prevent_initial_call=True,
    )
    def click_load_preset(preset_path: Optional[str]) -> Tuple[Any, ...]:
        """Populate the form from a preset file."""
        if not preset_path:
            return (no_update,) * 9
        try:
            payload = json.loads(
                Path(preset_path).read_text(encoding="utf-8")
            )
            state = run_state_from_json(payload)
            flags: List[str] = []
            if state.dry_run:
                flags.append("dry_run")
            if state.resume:
                flags.append("resume")
            if state.save_inspect:
                flags.append("save_inspect")
            return (
                state.pipeline_path,
                state.input_dir,
                state.output_dir,
                state.mode,
                flags,
                *_toast(
                    f"Loaded preset {Path(preset_path).stem}", ok=True
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Load preset failed")
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
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
        looks_like_json = (
            caps.get("has_pipeline_json", False)
            or matches_any_suffix(path, PIPELINE_CONFIG_SUFFIXES)
        )

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
        Output(ids.RC_STORE_OUTPUT_DIR, "data", allow_duplicate=True),
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
        """Route the sidebar selection into the form's per-field stores.

        ``ctx.triggered_id`` distinguishes which button fired. Each button
        writes to one of the three RC stores and clears the selection
        store so the banner closes.
        """
        triggered = ctx.triggered_id
        if triggered is None:
            return (no_update,) * 8

        if triggered == ids.RC_HANDOFF_DISMISS:
            # ``no_update`` for the four toast outputs so an unrelated
            # toast (e.g. a still-visible "Saved preset" confirmation)
            # is not torn down as a side-effect of dismissing the
            # hand-off banner.
            return (
                no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
                None,
            )

        path = (selection or {}).get("path") if selection else None
        abs_path = (selection or {}).get("abs_path") if selection else None
        if not path:
            return (
                no_update, no_update, no_update,
                *_toast("No sidebar selection", ok=False),
                no_update,
            )

        # The form stores carry absolute paths so the CLI gets a fully-
        # resolved argv tail; fall back to the rel path if the chrome
        # didn't stamp the absolute one.
        target = abs_path or path

        if triggered == ids.RC_HANDOFF_USE_PIPELINE:
            return (
                target, no_update, no_update,
                *_toast(f"Set as pipeline: {path}", ok=True),
                None,
            )
        if triggered == ids.RC_HANDOFF_USE_INPUT:
            return (
                no_update, target, no_update,
                *_toast(f"Set as input dir: {path}", ok=True),
                None,
            )
        if triggered == ids.RC_HANDOFF_USE_OUTPUT:
            return (
                no_update, no_update, target,
                *_toast(f"Set as output dir: {path}", ok=True),
                None,
            )
        return (no_update,) * 8

