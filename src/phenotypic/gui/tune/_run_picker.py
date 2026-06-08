"""Run picker — sandbox-bounded tune-output binding (Chunk C).

The hub mounts ``/tune/`` before a run is bound (``create_app(root=None)``), so
the user binds a tune output directory **at runtime** from the page itself: a
sandbox-bounded directory picker plus a "Bind this run" action. On confirm the
chosen directory is validated by :meth:`~phenotypic.gui.tune.TuneRunRoot.discover`
and, on success, its path is written into
:data:`~phenotypic.gui.tune._ids.TUNE_RUN_ROOT_STORE` — the store the Monitor /
Curate / Space / Launch callbacks all re-read — which makes the loaded views
render. A non-tune directory yields a clear note, never a 500.

This mirrors the Curate :mod:`~phenotypic.gui.tune._image_source` picker exactly
(the same builder ``directory_tree`` folder-only listing inside a
:class:`dbc.Modal`, the same sandbox boundary), so the two pickers coexist on the
unified hub with distinct pattern-matching ``id_type`` values. Binding only
**reads** the run directory — it never mutates it (the read-only invariant).

Like the rest of :mod:`phenotypic.gui.tune`, importing this module must never
drag ``optuna`` into ``sys.modules`` (it imports only Dash + the sandbox / tree
primitives + :class:`TuneRunRoot`, none of which touch optuna).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.gui.builder._directory_browser import directory_tree
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.tune import _ids as ids
from phenotypic.gui.tune._run_root import TuneRunRoot, TuneRunRootError

logger = logging.getLogger(__name__)

__all__ = [
    "resolve_run_dir",
    "discover_run_payload",
    "render_run_picker_tree",
    "build_run_picker_modal",
    "build_run_picker_row",
]

#: The label shown next to the picker button before a run is bound.
_NO_RUN_LABEL: str = "no run bound"


def resolve_run_dir(sandbox: SandboxRoot, candidate: str) -> Path | None:
    """Resolve ``candidate`` to an in-sandbox directory, or ``None``.

    The run directory must be a real directory inside the sandbox boundary. Any
    path that escapes the sandbox (``..`` traversal or an out-of-root absolute
    path) or is not an existing directory is refused — the caller surfaces the
    refusal in the picker note rather than binding a run from an unbounded
    location.

    Args:
        sandbox: The frozen-at-launch containment primitive.
        candidate: The path string from the picker (typed or browsed).

    Returns:
        The absolute, sandbox-contained directory path, or ``None`` when the
        candidate escapes the sandbox or is not an existing directory.
    """
    try:
        resolved = sandbox.resolve(candidate)
    except ValueError:
        logger.warning("Run directory escapes sandbox: %r", candidate)
        return None
    if not resolved.is_dir():
        logger.warning("Run directory is not a directory: %s", resolved)
        return None
    return resolved


def discover_run_payload(
    sandbox: Optional[SandboxRoot], candidate: str
) -> "tuple[Optional[dict], str]":
    """Validate ``candidate`` as a tune output and return its store payload.

    The pure bind logic, unit-testable without Dash. Resolves ``candidate``
    inside the sandbox (when one is bound), runs
    :meth:`~phenotypic.gui.tune.TuneRunRoot.discover`, and on success returns the
    JSON-serialisable run-root payload the
    :data:`~phenotypic.gui.tune._ids.TUNE_RUN_ROOT_STORE` carries (a ``{"path":
    <abs>}`` descriptor the views re-discover from). On failure it returns
    ``(None, <note>)`` with a clear human-readable reason — an out-of-sandbox
    escape, a missing directory, or a non-tune directory — so the caller renders
    a note instead of a 500.

    Binding only **reads** the directory (``discover`` reads the markers / spec
    only); it never writes to it.

    Args:
        sandbox: The frozen-at-launch sandbox, or ``None`` for a standalone
            launch without a sandbox (the candidate is then trusted as-is).
        candidate: The directory string from the picker.

    Returns:
        ``(payload, note)``: the run-root store payload (``{"path": <abs>}``) and
        an empty note on success, or ``(None, <reason>)`` on any failure.
    """
    if not candidate:
        return None, "Pick a tune output directory to bind."

    if sandbox is not None:
        resolved = resolve_run_dir(sandbox, candidate)
        if resolved is None:
            return (
                None,
                f"Refused: {candidate} escapes the sandbox or is not a directory.",
            )
    else:
        resolved = Path(candidate)
        if not resolved.is_dir():
            return None, f"Not a directory: {candidate}."

    try:
        root = TuneRunRoot.discover(resolved)
    except TuneRunRootError:
        # A real directory that simply is not a tune output — the single
        # expected "this is not a tune run" rejection. Surface a short, clear
        # message rather than the full marker-precedence explanation.
        logger.info("Run bind rejected (not a tune output): %s", resolved)
        return (
            None,
            f"Not a tune output: {resolved} has no tune run markers "
            "(.pht-tune-cache/run.json, tuning_spec.json, or trials.parquet).",
        )
    except Exception:  # noqa: BLE001 - any other failure degrades to a note
        logger.warning("Run bind discovery failed for %s", resolved, exc_info=True)
        return None, f"Could not read {resolved} as a tune output (see the log)."

    return {"path": str(root.path)}, ""


def render_run_picker_tree(
    sandbox: SandboxRoot, current: Path | None = None
) -> html.Div:
    """Render the folder-only run-picker tree for ``current``.

    Reuses the builder's :func:`directory_tree` (no file selection — the user
    picks the run output *directory*) with a Tune-run-specific ``id_type`` so the
    pattern-matching navigation callback never collides with the builder /
    run-console / Curate Image Source trees on the unified hub.

    Args:
        sandbox: Containment primitive — the listing never crosses
            ``sandbox.root``.
        current: Directory currently being viewed (defaults to the sandbox root).

    Returns:
        The depth-1 directory listing as an :class:`html.Div`.
    """
    return directory_tree(
        sandbox.root,
        current=current,
        extensions=None,
        select_files=False,
        id_type=ids.TUNE_DIR_ENTRY_RUN,
    )


def build_run_picker_modal(sandbox: SandboxRoot) -> dbc.Modal:
    """Build the sandbox-bounded run-directory picker modal.

    Folder-only tree (mirrors the Curate Image Source picker and the run-console
    output picker). The selected directory is committed via the "Bind this run"
    footer button so an accidental click while exploring doesn't bind the wrong
    directory.

    Args:
        sandbox: Containment primitive used as the security boundary.

    Returns:
        A :class:`dbc.Modal` with id :data:`ids.TUNE_RUN_PICKER_MODAL`, starting
        closed.
    """
    body = html.Div(
        [
            dcc.Store(
                id=ids.TUNE_RUN_PICKER_BROWSE_DIR,
                data=str(sandbox.root),
            ),
            html.Div(
                id=ids.TUNE_RUN_PICKER_MODAL_BODY,
                children=[render_run_picker_tree(sandbox, sandbox.root)],
            ),
        ]
    )
    footer = dbc.ModalFooter(
        [
            dbc.Button(
                "Cancel",
                id=ids.TUNE_BTN_RUN_PICKER_CANCEL,
                color="secondary",
                outline=True,
                n_clicks=0,
            ),
            dbc.Button(
                "Bind this run",
                id=ids.TUNE_BTN_RUN_PICKER_CONFIRM,
                color="primary",
                n_clicks=0,
            ),
        ]
    )
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Pick a tune output directory")),
            dbc.ModalBody(body),
            footer,
        ],
        id=ids.TUNE_RUN_PICKER_MODAL,
        is_open=False,
        size="lg",
        scrollable=True,
        backdrop=True,
        centered=True,
    )


def build_run_picker_row(
    sandbox: Optional[SandboxRoot], *, bound_path: Optional[str] = None
) -> html.Div:
    """Build the run-picker row (Bind-run button + bound-path label + note).

    Lives in the page header so the picker is reachable in BOTH the empty state
    (no run bound) and the loaded state (re-bind a different run). When
    ``sandbox`` is ``None`` (a standalone launch without a sandbox) the picker is
    omitted and a short note explains a run cannot be picked — the page is still
    usable if it was constructed with a pre-bound ``root``.

    Args:
        sandbox: The frozen-at-launch sandbox; ``None`` degrades the row to a
            note.
        bound_path: The currently-bound run path (shown in the label), or
            ``None`` when no run is bound yet.

    Returns:
        The picker row :class:`html.Div`.
    """
    if sandbox is None:
        return html.Div(
            "Run picker unavailable (no sandbox bound).",
            className="tune-runpicker-note",
        )
    label = bound_path if bound_path else _NO_RUN_LABEL
    return html.Div(
        [
            dbc.Button(
                "Bind run...",
                id=ids.TUNE_BTN_PICK_RUN,
                color="primary",
                outline=True,
                size="sm",
                n_clicks=0,
            ),
            html.Span(
                label,
                id=ids.TUNE_RUN_PICKER_LABEL,
                className="tune-runpicker-value",
            ),
            html.Span(
                "",
                id=ids.TUNE_RUN_PICKER_NOTE,
                className="tune-runpicker-note",
            ),
        ],
        className="tune-runpicker-row",
    )
