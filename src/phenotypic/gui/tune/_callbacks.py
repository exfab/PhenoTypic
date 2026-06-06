"""Dash callbacks for the ``/tune/`` co-pilot.

Two concerns live here, each a thin Dash adapter around a pure, headless-
testable helper:

* **Sub-tab switching** — :func:`active_view` maps a clicked sub-tab button's
  ID to its view name (falling back to the default Monitor view for an unknown
  or absent trigger); the registered callback toggles which view container is
  visible and which button carries the active class.
* **Monitor poll** — :func:`read_study_for_monitor` reads the run's study every
  poll tick (live ``OptunaStudyStore`` when the ``tune`` extra is importable AND
  the storage URL connects within a short timeout, else the finished
  ``trials.parquet``); the registered poll callback re-builds the objective /
  importance figures, the gap badge, and the trials table from it.

The module never imports ``optuna`` at import time. The live store is imported
**inside** :func:`read_study_for_monitor` (gated on
``importlib.util.find_spec("optuna")``), so this module stays in the package's
optuna-free import surface.
"""
from __future__ import annotations

import importlib.util
import logging
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import TYPE_CHECKING, Optional

from dash import ctx

from phenotypic.gui._config import THREAD_NAME_PREFIX
from phenotypic.gui.tune import _ids as ids

if TYPE_CHECKING:
    from phenotypic.gui.tune._study_read import _ReadableStore
    from phenotypic.gui.tune._run_root import TuneRunRoot

logger = logging.getLogger(__name__)

#: The default view shown when no (or an unknown) sub-tab is active.
_DEFAULT_VIEW: ids.SubTabName = "monitor"

#: Hard cap on how long a live-study open may block the 3-second poll. An
#: unreachable Postgres would otherwise stall the constructor for ~30 s
#: (libpq's default connect timeout), starving every other poll. The bound is
#: enforced at TWO levels (see :func:`read_study_for_monitor` and
#: :func:`_ensure_connect_timeout`): a libpq ``connect_timeout`` merged into the
#: storage URL so the constructor itself returns fast, AND a non-re-blocking
#: worker-thread wait so even a slow connect can't freeze the poll.
_LIVE_CONNECT_TIMEOUT_S: float = 3.0

#: Schemes (SQLAlchemy backend names) whose driver honors a libpq-style
#: ``connect_timeout`` query param. SQLite is local (no network hang), so it is
#: deliberately absent — :func:`_ensure_connect_timeout` no-ops on it.
_CONNECT_TIMEOUT_BACKENDS: frozenset[str] = frozenset({"postgresql"})

#: A process-wide, single-worker pool for the live-study open. Shared (not a
#: per-call ``with`` block) so a still-connecting worker is NEVER re-joined:
#: when :func:`read_study_for_monitor` times out it returns the parquet
#: fallback immediately and leaves the orphaned worker to finish (and be
#: discarded) on its own. The single worker naturally coalesces — a poll tick
#: that arrives while a previous open is still in flight simply queues behind
#: it and will itself time out, falling back to parquet, rather than spawning
#: an unbounded fan of connect attempts against a dead host.
_LIVE_OPEN_POOL: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix=f"{THREAD_NAME_PREFIX}-tune-live-open"
)

#: The degrade note shown when a live read was attempted but the storage could
#: not be reached in time — points the user at the usual culprits.
_NOTE_LIVE_UNREACHABLE: str = (
    "couldn't reach the live study — check network / ~/.pgpass. "
    "Showing the last finished trials."
)

#: The note shown when the run wants a live study but the ``tune`` extra (and so
#: ``optuna``) is not installed.
_NOTE_MISSING_EXTRA: str = (
    "install the tune extra for live monitoring (pip install 'phenotypic[tune]')."
)

#: Reverse map: a sub-tab button ID -> its view name. Built once from the
#: ordered sub-tab names so the helper never re-derives strings at call time.
_BUTTON_ID_TO_VIEW: dict[str, ids.SubTabName] = {
    ids.subtab_button_id(name): name for name in ids.SUBTAB_ORDER
}


def active_view(trigger_id: str | None) -> ids.SubTabName:
    """Resolve which sub-tab view a click on ``trigger_id`` should show.

    Pure routing logic, unit-testable without Dash: a known sub-tab button ID
    (``tune-subtab-<name>``) resolves to its view name; ``None``, an empty
    string, or any unrecognised ID falls back to the default Monitor view (so
    the initial render and any stray trigger land somewhere valid).

    Args:
        trigger_id: The ID of the component that fired the callback (Dash's
            ``ctx.triggered_id``), or ``None`` on the initial call.

    Returns:
        The resolved view name (one of :data:`ids.SUBTAB_ORDER`).
    """
    if not trigger_id:
        return _DEFAULT_VIEW
    return _BUTTON_ID_TO_VIEW.get(trigger_id, _DEFAULT_VIEW)


def _view_container_class(name: ids.SubTabName, active: ids.SubTabName) -> str:
    """The class string for a view container (non-active gets the hidden class)."""
    classes = ["tune-view"]
    if name != active:
        classes.append("tune-view-hidden")
    return " ".join(classes)


def _subtab_button_class(name: ids.SubTabName, active: ids.SubTabName) -> str:
    """The class string for a sub-tab button (active gets the highlight class)."""
    classes = ["tune-subtab"]
    if name == active:
        classes.append("tune-subtab-active")
    return " ".join(classes)


# ---------------------------------------------------------------------------
# Monitor — graceful live read (OQ4)
# ---------------------------------------------------------------------------

def _load_journal(root: "TuneRunRoot") -> "Optional[_ReadableStore]":
    """Load the finished ``trials.parquet`` journal, or ``None`` when absent.

    The fall-back read path: a parquet-only run, or a live run whose study could
    not be reached. Returns ``None`` (rather than raising) when no journal has
    been written yet, so a brand-new run degrades to "no trials yet" cleanly.
    """
    from phenotypic.tune._study_store import JournalStudyStore

    trials_path = root.trials_path
    if trials_path is None or not trials_path.exists():
        return None
    try:
        return JournalStudyStore.from_parquet(trials_path)
    except Exception:  # noqa: BLE001 - poll must never raise; degrade to empty
        logger.warning("Could not load tune journal at %s", trials_path, exc_info=True)
        return None


def _ensure_connect_timeout(storage_url: str) -> str:
    """Merge a libpq ``connect_timeout`` into a postgres storage URL.

    The real fix for the OQ4 stall: the live ``OptunaStudyStore`` constructor
    connects eagerly, and libpq's default connect timeout is ~30 s (or never,
    on a black-holed host). Merging ``connect_timeout=<N>`` (N =
    :data:`_LIVE_CONNECT_TIMEOUT_S`) into the URL makes the *constructor itself*
    return fast — SQLAlchemy's psycopg dialect passes URL query params straight
    through to the driver, and libpq honors ``connect_timeout``.

    Only ``postgresql*`` schemes are touched (the backend name covers
    ``postgresql``, ``postgresql+psycopg``, ``postgresql+psycopg2``, …); SQLite
    is local and never network-hangs, so it is returned unchanged. A
    user-supplied ``connect_timeout`` is preserved (never overwritten), so an
    operator who deliberately set a longer bound keeps it. ``connect_timeout``
    is not a secret, so this respects the password-in-``.pgpass`` rule.

    Args:
        storage_url: The run's resolved Optuna storage URL.

    Returns:
        The URL with ``connect_timeout`` ensured for a postgres backend, or the
        original URL unchanged for any other backend (or an unparseable URL).
    """
    from sqlalchemy.engine import make_url
    from sqlalchemy.exc import ArgumentError

    try:
        url = make_url(storage_url)
    except (ArgumentError, ValueError):
        # An unparseable URL: leave it alone and let the constructor surface the
        # real error (still bounded by the non-re-blocking worker wait).
        return storage_url

    if url.get_backend_name() not in _CONNECT_TIMEOUT_BACKENDS:
        return storage_url
    if "connect_timeout" in url.query:
        return storage_url

    bounded = url.update_query_dict(
        {"connect_timeout": str(int(_LIVE_CONNECT_TIMEOUT_S))}, append=False
    )
    return bounded.render_as_string(hide_password=False)


def _open_live_study(root: "TuneRunRoot") -> "_ReadableStore":
    """Open the live ``OptunaStudyStore`` for ``root`` (called on a worker thread).

    Imports optuna lazily HERE (never at module import). The constructor opens
    the RDB study eagerly, so the storage URL is first passed through
    :func:`_ensure_connect_timeout` to bound the connect at the source.
    """
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    assert root.storage_url is not None  # guarded by the caller
    return OptunaStudyStore(
        storage_url=_ensure_connect_timeout(root.storage_url),
        study_name=root.study_name,
        directions=root.directions,
    )


def read_study_for_monitor(
    root: "TuneRunRoot",
) -> "tuple[Optional[_ReadableStore], str]":
    """Read the run's study for the Monitor view, degrading gracefully (OQ4).

    Resolution order:

    1. **Parquet-only run** (no ``storage_url``): read the finished
       ``trials.parquet`` directly — no live attempt, no note.
    2. **Live run, ``tune`` extra missing**: skip the live read, fall back to
       the journal, and return the "install the tune extra" note.
    3. **Live run, extra present**: open the live ``OptunaStudyStore`` on a
       worker thread with a short connect timeout
       (:data:`_LIVE_CONNECT_TIMEOUT_S`). On success → the live store, no note.
       On timeout / connection error → the journal + the "couldn't reach the
       live study" note.

    The store is read-only and the poll must never raise, so every failure path
    degrades to the journal (or ``None`` when no journal exists yet).

    Args:
        root: The bound tune output handle.

    Returns:
        ``(store, note)``: the resolved store (or ``None`` when nothing is
        readable yet) and a degrade / status note (``""`` when the read was
        clean).
    """
    # 1. Parquet-only run — no live study to reach.
    if root.storage_url is None:
        return _load_journal(root), ""

    # 2. Live run, but the tune extra (optuna) is not installed.
    if importlib.util.find_spec("optuna") is None:
        return _load_journal(root), _NOTE_MISSING_EXTRA

    # 3. Live run, extra present — open with a bounded, non-re-blocking wait so
    #    an unreachable storage can't stall the poll. The connect is bounded at
    #    the source (``_ensure_connect_timeout`` merges a libpq
    #    ``connect_timeout`` into the URL), and the wait here NEVER re-joins a
    #    still-connecting worker: we submit to the shared single-worker pool and,
    #    on timeout, return the parquet fallback immediately, leaving the
    #    orphaned future to finish (and be discarded) on its own. Using a
    #    ``with``-managed pool here would re-introduce the bug — its ``__exit__``
    #    calls ``shutdown(wait=True)``, re-joining the stuck worker and blocking
    #    the full connect duration regardless of the ``result`` timeout.
    future: "Future[_ReadableStore]" = _LIVE_OPEN_POOL.submit(_open_live_study, root)
    try:
        return future.result(timeout=_LIVE_CONNECT_TIMEOUT_S), ""
    except FutureTimeout:
        logger.warning(
            "Live tune study open timed out after %.1fs (url=%s); "
            "degrading to journal.",
            _LIVE_CONNECT_TIMEOUT_S,
            root.storage_url,
        )
        return _load_journal(root), _NOTE_LIVE_UNREACHABLE
    except Exception:  # noqa: BLE001 - any open/connect error degrades
        logger.warning(
            "Live tune study open failed (url=%s); degrading to journal.",
            root.storage_url,
            exc_info=True,
        )
        return _load_journal(root), _NOTE_LIVE_UNREACHABLE


# ---------------------------------------------------------------------------
# Monitor — render helpers
# ---------------------------------------------------------------------------

#: The trials-table columns, in display order.
_TRIALS_COLUMNS: tuple[str, ...] = ("number", "score", "n_images", "failed")


def _build_trials_table(store: "Optional[_ReadableStore]"):  # type: ignore[no-untyped-def]
    """Render the trials ``DataTable`` from ``store`` (a placeholder when empty)."""
    from dash import dash_table, html

    if store is None or not store.trials:
        return html.P("No trials yet.", className="tune-monitor-note")

    rows = [
        {
            "number": t.number,
            "score": round(t.score, 4),
            "n_images": t.n_images,
            "failed": "yes" if t.failed else "",
        }
        for t in store.trials
    ]
    return dash_table.DataTable(  # type: ignore[attr-defined]
        data=rows,
        columns=[{"name": col, "id": col} for col in _TRIALS_COLUMNS],
        page_size=10,
        sort_action="native",
    )


def _gap_badge_outputs(store: "Optional[_ReadableStore]") -> tuple[str, str]:
    """The gap-badge ``(label, className)`` for ``store`` (empty → stable)."""
    from phenotypic.gui.tune._study_read import gap_badge

    if store is None:
        return "—", "tune-gap-badge tune-gap-badge-stable"
    label, flagged = gap_badge(store)
    variant = "unstable" if flagged else "stable"
    return label, f"tune-gap-badge tune-gap-badge-{variant}"


def register_callbacks(app, *, sandbox=None) -> None:  # type: ignore[no-untyped-def]
    """Register the tune sub-app's Dash callbacks on ``app``.

    Wires:

    * **Sub-tab switch** — a click on any of the four sub-tab buttons re-resolves
      the active view via :func:`active_view`, toggles each view container's
      visibility and each button's active class, and mirrors the active view
      name into :data:`ids.TUNE_ACTIVE_VIEW_STORE`.
    * **Monitor poll** — every 3 s the poll re-reads the study
      (:func:`read_study_for_monitor`) and re-renders the objective / importance
      figures, the gap badge, the trials table, and the degrade note.
    * **Curate** (when ``sandbox`` is bound) — the sandbox-bounded Image Source
      picker, the A/B pin + overlay render/poll, the mode toggle, and the
      "Set as winner" write (see :func:`_register_curate_callbacks`).

    Args:
        app: The :class:`dash.Dash` instance whose layout is assigned.
        sandbox: The frozen-at-launch sandbox root. When ``None`` the Curate
            callbacks are skipped (the Curate view degrades to a note).
    """
    from dash import Input, Output, State

    @app.callback(
        Output(ids.TUNE_ACTIVE_VIEW_STORE, "data"),
        *[
            Output(ids.view_container_id(name), "className")
            for name in ids.SUBTAB_ORDER
        ],
        *[
            Output(ids.subtab_button_id(name), "className")
            for name in ids.SUBTAB_ORDER
        ],
        *[
            Input(ids.subtab_button_id(name), "n_clicks")
            for name in ids.SUBTAB_ORDER
        ],
        prevent_initial_call=True,
    )
    def _switch_subtab(*_n_clicks: int) -> tuple[str, ...]:
        active = active_view(ctx.triggered_id)
        container_classes = [
            _view_container_class(name, active) for name in ids.SUBTAB_ORDER
        ]
        button_classes = [
            _subtab_button_class(name, active) for name in ids.SUBTAB_ORDER
        ]
        return (active, *container_classes, *button_classes)

    @app.callback(
        Output(ids.TUNE_OBJECTIVE_FIGURE, "figure"),
        Output(ids.TUNE_IMPORTANCE_FIGURE, "figure"),
        Output(ids.TUNE_GAP_BADGE, "children"),
        Output(ids.TUNE_GAP_BADGE, "className"),
        Output(ids.TUNE_TRIALS_TABLE, "children"),
        Output(ids.TUNE_MONITOR_NOTE, "children"),
        Input(ids.TUNE_STUDY_POLL, "n_intervals"),
        State(ids.TUNE_RUN_ROOT_STORE, "data"),
    )
    def _poll_study(_n_intervals: int, run_root_data: "Optional[dict]"):  # type: ignore[no-untyped-def]
        from phenotypic.gui.tune._run_root import TuneRunRoot
        from phenotypic.gui.tune._study_read import (
            build_importance_figure,
            build_objective_figure,
        )

        # Re-discover the bound run from its path each tick (cheap: reads the
        # markers, never optuna). A missing/invalid store renders empty figures.
        store: "Optional[_ReadableStore]" = None
        note = ""
        if run_root_data and run_root_data.get("path"):
            from pathlib import Path

            try:
                root = TuneRunRoot.discover(Path(run_root_data["path"]))
                store, note = read_study_for_monitor(root)
            except Exception:  # noqa: BLE001 - poll must never raise
                logger.warning("Monitor poll re-discovery failed", exc_info=True)

        trials = list(store.trials) if store is not None else []
        objective_fig = build_objective_figure(trials)

        importances = _param_importances(store)
        importance_fig = build_importance_figure(importances)

        badge_label, badge_class = _gap_badge_outputs(store)
        table = _build_trials_table(store)
        return objective_fig, importance_fig, badge_label, badge_class, table, note

    if sandbox is not None:
        _register_curate_callbacks(app, sandbox)


def _param_importances(store: "Optional[_ReadableStore]") -> dict[str, float]:
    """The study's param importances, or ``{}`` when unavailable.

    Only the live ``OptunaStudyStore`` exposes ``param_importances`` (fANOVA);
    the parquet journal does not, so the importance figure is empty on the
    degraded path. Any failure degrades to ``{}`` so the poll never raises.
    """
    if store is None:
        return {}
    getter = getattr(store, "param_importances", None)
    if getter is None:
        return {}
    try:
        importances = getter()
    except Exception:  # noqa: BLE001 - poll must never raise
        logger.warning("param_importances read failed", exc_info=True)
        return {}
    return dict(importances) if importances else {}


# ---------------------------------------------------------------------------
# Curate — pure pin / mode helpers (no Dash; unit-tested headless)
# ---------------------------------------------------------------------------

#: The A/B pin store slot names, in fill order.
_AB_SLOTS: tuple[str, str] = ("a", "b")


def pinned_pair(clicked_trial: int, store: "Optional[dict]") -> dict:
    """Pin ``clicked_trial`` into the A/B pair, assign-A-then-B-then-re-pin.

    The pure pin logic, unit-testable without Dash. Given the clicked trial
    number and the current ``{"a": <n|None>, "b": <n|None>}`` store:

    * an empty slot A takes the trial;
    * else an empty slot B takes it;
    * else (both full) the trial re-pins into slot A — the oldest pin cycles
      out so the user can keep comparing against a held B side.

    Clicking the trial already pinned in a slot is idempotent (it does not
    duplicate the trial into the other slot).

    Args:
        clicked_trial: The shortlist-card trial number the user clicked.
        store: The current pin store (``{"a", "b"}``), or ``None`` / partial on
            first render — missing keys are treated as empty slots.

    Returns:
        The updated pin store ``{"a": <n|None>, "b": <n|None>}``.
    """
    base = store if isinstance(store, dict) else {}
    a = base.get("a")
    b = base.get("b")
    # Idempotent: clicking an already-pinned trial leaves the pair unchanged.
    if clicked_trial in (a, b):
        return {"a": a, "b": b}
    if a is None:
        return {"a": clicked_trial, "b": b}
    if b is None:
        return {"a": a, "b": clicked_trial}
    return {"a": clicked_trial, "b": b}


#: The valid Curate overlay modes (side-by-side ↔ difference).
_CURATE_MODES: frozenset[str] = frozenset({"side", "difference"})

#: The default Curate mode when the trigger is unknown / absent.
_DEFAULT_CURATE_MODE: str = "side"


def curate_mode(trigger: "Optional[str]") -> str:
    """Resolve the Curate overlay mode from the toggle's value.

    Pure, unit-testable: a known mode (``"side"`` / ``"difference"``) passes
    through; any unknown / absent value falls back to ``"side"`` so a stray
    trigger lands on the side-by-side default.

    Args:
        trigger: The mode toggle's value.

    Returns:
        One of ``"side"`` / ``"difference"``.
    """
    if trigger in _CURATE_MODES:
        return trigger  # type: ignore[return-value]
    return _DEFAULT_CURATE_MODE


# ---------------------------------------------------------------------------
# Curate — sandbox-bounded Image Source picker (B-IMG)
# ---------------------------------------------------------------------------

#: The pre-selection prompt shown above the (empty) overlay area until an
#: Image Source is bound.
_IMAGE_SOURCE_PLACEHOLDER: str = "no Image Source selected"


def _register_curate_callbacks(app, sandbox) -> None:  # type: ignore[no-untyped-def]
    """Register the Curate-view callbacks (Image Source picker; B-IMG).

    Args:
        app: The :class:`dash.Dash` instance.
        sandbox: The frozen-at-launch sandbox bounding plate loads.
    """
    from dash import ALL, Input, Output, State, no_update

    from phenotypic.gui.tune._image_source import (
        render_image_source_tree,
        resolve_image_source,
    )

    # --- Open / cancel the picker modal -----------------------------------
    @app.callback(
        Output(ids.TUNE_IMAGE_SOURCE_MODAL, "is_open", allow_duplicate=True),
        Input(ids.TUNE_BTN_PICK_IMAGE_SOURCE, "n_clicks"),
        Input(ids.TUNE_BTN_IMAGE_SOURCE_CANCEL, "n_clicks"),
        prevent_initial_call=True,
    )
    def _toggle_image_source_modal(open_clicks, cancel_clicks):  # type: ignore[no-untyped-def]
        # The open button opens; cancel closes. Dispatch on the trigger id.
        if ctx.triggered_id == ids.TUNE_BTN_PICK_IMAGE_SOURCE and open_clicks:
            return True
        if ctx.triggered_id == ids.TUNE_BTN_IMAGE_SOURCE_CANCEL and cancel_clicks:
            return False
        return no_update

    # --- Navigate the tree (folder click → browse-dir store) --------------
    @app.callback(
        Output(ids.TUNE_IMAGE_SOURCE_BROWSE_DIR, "data", allow_duplicate=True),
        Input(
            {"type": ids.TUNE_DIR_ENTRY_IMAGE_SOURCE, "kind": ALL, "path": ALL},
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def _navigate_image_source_tree(_clicks):  # type: ignore[no-untyped-def]
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update
        if triggered.get("type") != ids.TUNE_DIR_ENTRY_IMAGE_SOURCE:
            return no_update
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return no_update
        path = triggered.get("path")
        return path if isinstance(path, str) else no_update

    # --- Re-render the tree body on browse-dir change ---------------------
    @app.callback(
        Output(ids.TUNE_IMAGE_SOURCE_MODAL_BODY, "children"),
        Input(ids.TUNE_IMAGE_SOURCE_BROWSE_DIR, "data"),
        prevent_initial_call=True,
    )
    def _render_image_source_body(dir_value):  # type: ignore[no-untyped-def]
        from pathlib import Path

        current = Path(dir_value) if dir_value else None
        return render_image_source_tree(sandbox, current)

    # --- Confirm → resolve + commit the Image Source ----------------------
    @app.callback(
        Output(ids.TUNE_IMAGE_SOURCE_STORE, "data", allow_duplicate=True),
        Output(ids.TUNE_IMAGE_SOURCE_LABEL, "children", allow_duplicate=True),
        Output(ids.TUNE_IMAGE_SOURCE_MODAL, "is_open", allow_duplicate=True),
        Output(ids.TUNE_CURATE_TOAST, "is_open", allow_duplicate=True),
        Output(ids.TUNE_CURATE_TOAST, "children", allow_duplicate=True),
        Input(ids.TUNE_BTN_IMAGE_SOURCE_CONFIRM, "n_clicks"),
        State(ids.TUNE_IMAGE_SOURCE_BROWSE_DIR, "data"),
        prevent_initial_call=True,
    )
    def _confirm_image_source(n_clicks, browsed):  # type: ignore[no-untyped-def]
        if not n_clicks or not browsed:
            return no_update, no_update, no_update, no_update, no_update
        resolved = resolve_image_source(sandbox, browsed)
        if resolved is None:
            return (
                no_update,
                no_update,
                no_update,
                True,
                f"Refused: {browsed} escapes the sandbox or is not a directory.",
            )
        return str(resolved), str(resolved), False, no_update, no_update

    # --- Mirror the Image Source store → prompt visibility ----------------
    @app.callback(
        Output(ids.TUNE_CURATE_PROMPT, "style"),
        Input(ids.TUNE_IMAGE_SOURCE_STORE, "data"),
        prevent_initial_call=True,
    )
    def _toggle_curate_prompt(image_source):  # type: ignore[no-untyped-def]
        return {"display": "none"} if image_source else {}

    _register_curate_overlay_callbacks(app)


# ---------------------------------------------------------------------------
# Curate — shortlist pin + non-blocking overlay render/poll (B4)
# ---------------------------------------------------------------------------


#: Image extensions the plate picker surfaces (mirrors the builder image picker).
_PLATE_EXTS: frozenset[str] = frozenset(
    {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".nef", ".cr2", ".arw", ".dng"}
)


def _list_plate_names(image_source: "Optional[str]") -> list[str]:
    """List image file names directly under ``image_source`` (sorted).

    Returns ``[]`` for an unset / unreadable source so the picker degrades to
    empty rather than raising. Only depth-1 files with a known image extension
    are surfaced.
    """
    if not image_source:
        return []
    from pathlib import Path

    directory = Path(image_source)
    try:
        names = [
            entry.name
            for entry in directory.iterdir()
            if entry.is_file() and entry.suffix.lower() in _PLATE_EXTS
        ]
    except OSError:
        return []
    return sorted(names)


def _shortlist_card_class(trial: "Optional[int]", pinned: dict) -> str:
    """The class string for a shortlist card (highlight its A / B pin slot)."""
    classes = ["tune-shortlist-card"]
    if trial is not None and trial == pinned.get("a"):
        classes.append("tune-shortlist-card-a")
    elif trial is not None and trial == pinned.get("b"):
        classes.append("tune-shortlist-card-b")
    return " ".join(classes)


def _register_curate_overlay_callbacks(app) -> None:  # type: ignore[no-untyped-def]
    """Register the pin + render/poll + mode + winner Curate callbacks.

    Split out from the Image Source picker so the picker can register even when
    no shortlist exists yet (a brand-new live run). These callbacks render
    overlays **on demand** and stay non-blocking: the render callback submits to
    the :class:`OverlayCache` singleton and returns a spinner immediately; the
    ``dcc.Interval`` poll swaps in the real figure once the future resolves.
    """
    import uuid

    from dash import ALL, Input, Output, State, no_update

    from phenotypic.gui.tune import _curate_overlays as ov

    # --- Session id: a fresh uuid on first paint --------------------------
    @app.callback(
        Output(ids.TUNE_SESSION_ID, "data"),
        Input(ids.TUNE_SESSION_ID, "data"),
        prevent_initial_call=False,
    )
    def _init_session_id(current):  # type: ignore[no-untyped-def]
        return current if current else uuid.uuid4().hex

    # --- Populate the plate picker from the Image Source directory --------
    @app.callback(
        Output(ids.TUNE_PLATE_PICKER, "options"),
        Output(ids.TUNE_PLATE_PICKER, "value"),
        Input(ids.TUNE_IMAGE_SOURCE_STORE, "data"),
        State(ids.TUNE_PLATE_PICKER, "value"),
        prevent_initial_call=False,
    )
    def _populate_plates(image_source, current):  # type: ignore[no-untyped-def]
        names = _list_plate_names(image_source)
        options = [{"label": n, "value": n} for n in names]
        value = current if current in names else (names[0] if names else None)
        return options, value

    # --- Pin a shortlist card into the A/B pair ---------------------------
    @app.callback(
        Output(ids.TUNE_AB_STORE, "data"),
        Output({"type": ids.TUNE_SHORTLIST_CARD, "trial": ALL}, "className"),
        Input({"type": ids.TUNE_SHORTLIST_CARD, "trial": ALL}, "n_clicks"),
        State(ids.TUNE_AB_STORE, "data"),
        State({"type": ids.TUNE_SHORTLIST_CARD, "trial": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _pin_card(_clicks, store, ids_list):  # type: ignore[no-untyped-def]
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or "trial" not in triggered:
            return no_update, no_update
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return no_update, no_update
        pinned = pinned_pair(int(triggered["trial"]), store)
        classes = [
            _shortlist_card_class(card_id.get("trial"), pinned)
            for card_id in ids_list
        ]
        return pinned, classes

    # --- Mode toggle → mode store + container visibility ------------------
    @app.callback(
        Output(ids.TUNE_CURATE_MODE_STORE, "data"),
        Output(ids.TUNE_SIDE_BY_SIDE, "className"),
        Output(ids.TUNE_DIFFERENCE, "className"),
        Input(ids.TUNE_CURATE_MODE_TOGGLE, "value"),
        prevent_initial_call=True,
    )
    def _switch_curate_mode(value):  # type: ignore[no-untyped-def]
        mode = curate_mode(value)
        side_cls = "tune-curate-sidebyside"
        diff_cls = "tune-curate-difference"
        if mode == "difference":
            side_cls += " tune-view-hidden"
        else:
            diff_cls += " tune-view-hidden"
        return mode, side_cls, diff_cls

    # --- Render (NON-BLOCKING): submit overlays, return spinners ----------
    @app.callback(
        Output(ids.TUNE_GRAPH_A, "figure", allow_duplicate=True),
        Output(ids.TUNE_GRAPH_B, "figure", allow_duplicate=True),
        Output(ids.TUNE_GRAPH_DIFF, "figure", allow_duplicate=True),
        Input(ids.TUNE_AB_STORE, "data"),
        Input(ids.TUNE_PLATE_PICKER, "value"),
        Input(ids.TUNE_CURATE_MODE_STORE, "data"),
        State(ids.TUNE_SESSION_ID, "data"),
        State(ids.TUNE_IMAGE_SOURCE_STORE, "data"),
        State(ids.TUNE_RUN_ROOT_STORE, "data"),
        prevent_initial_call=True,
    )
    def _render_overlays(  # type: ignore[no-untyped-def]
        pinned, plate, mode, session_id, image_source, run_root_data
    ):
        return _submit_curate_overlays(
            ov,
            pinned=pinned,
            plate=plate,
            mode=mode,
            session_id=session_id,
            image_source=image_source,
            run_root_data=run_root_data,
        )

    # --- Overlay-readiness poll: swap spinners for resolved figures -------
    @app.callback(
        Output(ids.TUNE_GRAPH_A, "figure", allow_duplicate=True),
        Output(ids.TUNE_GRAPH_B, "figure", allow_duplicate=True),
        Output(ids.TUNE_GRAPH_DIFF, "figure", allow_duplicate=True),
        Input(ids.TUNE_OVERLAY_POLL, "n_intervals"),
        State(ids.TUNE_AB_STORE, "data"),
        State(ids.TUNE_PLATE_PICKER, "value"),
        State(ids.TUNE_CURATE_MODE_STORE, "data"),
        State(ids.TUNE_SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def _poll_overlays(  # type: ignore[no-untyped-def]
        _n, pinned, plate, mode, session_id
    ):
        return _poll_curate_overlays(
            ov,
            pinned=pinned,
            plate=plate,
            mode=mode,
            session_id=session_id,
        )

    _register_linked_zoom(app)


def _register_linked_zoom(app) -> None:  # type: ignore[no-untyped-def]
    """Wire the clientside A<->B linked-zoom mirror (Task B4).

    Two clientside callbacks (A->B and B->A), each passing its OWN graph's
    relayout prop-id as the third arg so the JS ``mirrorRange`` propagates only
    from the user-driven graph (the triggered-prop guard that stops the
    A->B->A infinite relayout). The partner graph's current ``figure`` is a
    ``State`` so the JS can clone it with the synced axis range.
    """
    from dash import Input, Output, State

    app.clientside_callback(
        f"function(r, fig) {{ return window.dash_clientside.tune_sync.mirrorRange("
        f"r, fig, '{ids.TUNE_GRAPH_A}.relayoutData'); }}",
        Output(ids.TUNE_GRAPH_B, "figure", allow_duplicate=True),
        Input(ids.TUNE_GRAPH_A, "relayoutData"),
        State(ids.TUNE_GRAPH_B, "figure"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        f"function(r, fig) {{ return window.dash_clientside.tune_sync.mirrorRange("
        f"r, fig, '{ids.TUNE_GRAPH_B}.relayoutData'); }}",
        Output(ids.TUNE_GRAPH_A, "figure", allow_duplicate=True),
        Input(ids.TUNE_GRAPH_B, "relayoutData"),
        State(ids.TUNE_GRAPH_A, "figure"),
        prevent_initial_call=True,
    )


def _submit_curate_overlays(  # type: ignore[no-untyped-def]
    ov,
    *,
    pinned,
    plate,
    mode,
    session_id,
    image_source,
    run_root_data,
):
    """Submit the needed overlays (non-blocking) and return spinner figures.

    Pure orchestration over the overlay module ``ov`` so it is testable without
    Dash: resolves the run + base pipeline, and for each needed slot submits a
    render future (or returns a guidance figure when prerequisites are missing).
    """
    from pathlib import Path

    from phenotypic.gui.tune import _curate as curate
    from phenotypic.gui.tune._overlays import get_overlay_cache
    from phenotypic.gui.tune._run_root import TuneRunRoot

    spinner = curate.placeholder_figure("rendering…")
    no_source = curate.placeholder_figure("pick an Image Source")
    no_plate = curate.placeholder_figure("pick a plate")

    if not image_source:
        return no_source, no_source, no_source
    if not plate:
        return no_plate, no_plate, no_plate
    if not (run_root_data and run_root_data.get("path")):
        return no_update_triple()

    pinned = pinned if isinstance(pinned, dict) else {}
    a_trial, b_trial = pinned.get("a"), pinned.get("b")
    mode = curate_mode(mode)
    session = session_id or "default"

    try:
        root = TuneRunRoot.discover(Path(run_root_data["path"]))
    except Exception:  # noqa: BLE001 - render must degrade, never raise
        logger.warning("Curate render re-discovery failed", exc_info=True)
        return no_update_triple()

    base = ov.read_base_pipeline(root)
    if base is None:
        unavailable = curate.placeholder_figure("base pipeline unavailable")
        return unavailable, unavailable, unavailable

    cache = get_overlay_cache(root.path)
    trials = _trials_by_number(root)

    fig_a = _submit_one_candidate(
        ov, cache, base, trials, session, a_trial, plate, image_source, spinner
    )
    fig_b = _submit_one_candidate(
        ov, cache, base, trials, session, b_trial, plate, image_source, spinner
    )
    fig_diff = _submit_difference(
        ov, cache, base, trials, session, a_trial, b_trial, plate, image_source,
        spinner,
    )
    return fig_a, fig_b, fig_diff


def no_update_triple():  # type: ignore[no-untyped-def]
    """Three ``no_update`` sentinels for the three Curate graph outputs."""
    from dash import no_update

    return no_update, no_update, no_update


def _trials_by_number(root: "TuneRunRoot") -> "dict[int, object]":
    """Map ``trial.number -> Trial`` from the run's journal (``{}`` when none)."""
    store = _load_journal(root)
    if store is None:
        return {}
    return {t.number: t for t in store.trials}


def _submit_one_candidate(  # type: ignore[no-untyped-def]
    ov, cache, base, trials, session, trial_number, plate, image_source, spinner
):
    """Submit one candidate overlay; return its spinner / guidance figure."""
    from phenotypic.gui.tune import _curate as curate

    if trial_number is None:
        return curate.placeholder_figure("pin a candidate")
    trial = trials.get(trial_number)
    if trial is None:
        return curate.placeholder_figure(f"trial {trial_number} not in journal")
    key = (session, int(trial_number), str(plate), "candidate")

    def _render():  # type: ignore[no-untyped-def]
        from phenotypic.gui.tune._overlays import render_candidate_overlay

        grid = ov.load_plate_grid(image_source, plate)
        return render_candidate_overlay(base, trial.params, grid)

    ov.request_overlay(cache, key, _render)
    return spinner


def _submit_difference(  # type: ignore[no-untyped-def]
    ov, cache, base, trials, session, a_trial, b_trial, plate, image_source, spinner
):
    """Submit the A-vs-B difference overlay; return its spinner / guidance."""
    from phenotypic.gui.tune import _curate as curate

    if a_trial is None or b_trial is None:
        return curate.placeholder_figure("pin A and B to diff")
    trial_a = trials.get(a_trial)
    trial_b = trials.get(b_trial)
    if trial_a is None or trial_b is None:
        return curate.placeholder_figure("a pinned trial is not in the journal")
    key = (session, int(a_trial), f"{plate}|{b_trial}", "difference")

    def _render():  # type: ignore[no-untyped-def]
        from phenotypic.gui.tune._overlays import render_difference
        from phenotypic.tune._evaluation._builder import build_pipeline

        grid = ov.load_plate_grid(image_source, plate)
        seg_a = build_pipeline(base, trial_a.params).apply(grid.copy())
        seg_b = build_pipeline(base, trial_b.params).apply(grid.copy())
        return render_difference(
            grid.rgb[:], seg_a.objmap[:], seg_b.objmap[:]
        )

    ov.request_overlay(cache, key, _render)
    return spinner


def _poll_curate_overlays(  # type: ignore[no-untyped-def]
    ov, *, pinned, plate, mode, session_id
):
    """Swap any resolved overlay future into its figure; else ``no_update``."""
    from phenotypic.gui.tune import _curate as curate

    pinned = pinned if isinstance(pinned, dict) else {}
    a_trial, b_trial = pinned.get("a"), pinned.get("b")
    session = session_id or "default"
    error_fig = curate.placeholder_figure("render failed — see logs")

    def _swap(key):  # type: ignore[no-untyped-def]
        from dash import no_update

        if key is None or not ov.overlay_ready(key):
            return no_update
        array = ov.take_overlay(key)
        return ov.overlay_figure(array) if array is not None else error_fig

    key_a = (session, int(a_trial), str(plate), "candidate") if a_trial is not None and plate else None
    key_b = (session, int(b_trial), str(plate), "candidate") if b_trial is not None and plate else None
    key_diff = (
        (session, int(a_trial), f"{plate}|{b_trial}", "difference")
        if a_trial is not None and b_trial is not None and plate
        else None
    )
    return _swap(key_a), _swap(key_b), _swap(key_diff)


__all__ = ["active_view", "read_study_for_monitor", "register_callbacks"]
