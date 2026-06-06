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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import TYPE_CHECKING, Optional

from dash import ctx

from phenotypic.gui.tune import _ids as ids

if TYPE_CHECKING:
    from phenotypic.gui.tune._study_read import _ReadableStore
    from phenotypic.gui.tune._run_root import TuneRunRoot

logger = logging.getLogger(__name__)

#: The default view shown when no (or an unknown) sub-tab is active.
_DEFAULT_VIEW: ids.SubTabName = "monitor"

#: Hard cap on how long a live-study open may block the 3-second poll. An
#: unreachable Postgres would otherwise stall the constructor for ~30 s
#: (psycopg's default connect retry), starving every other poll. We open the
#: live store on a worker thread and abandon it past this deadline, degrading
#: to the finished ``trials.parquet`` (OQ4).
_LIVE_CONNECT_TIMEOUT_S: float = 3.0

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


def _open_live_study(root: "TuneRunRoot") -> "_ReadableStore":
    """Open the live ``OptunaStudyStore`` for ``root`` (called on a worker thread).

    Imports optuna lazily HERE (never at module import). The constructor opens
    the RDB study eagerly, so this is the call the connect-timeout guards.
    """
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    assert root.storage_url is not None  # guarded by the caller
    return OptunaStudyStore(
        storage_url=root.storage_url,
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

    # 3. Live run, extra present — open with a short connect timeout so an
    #    unreachable storage can't stall the poll. The constructor opens the
    #    RDB study eagerly, so we run it on a worker thread and abandon it past
    #    the deadline.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_open_live_study, root)
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


def register_callbacks(app) -> None:  # type: ignore[no-untyped-def]
    """Register the tune sub-app's Dash callbacks on ``app``.

    Wires two callbacks:

    * **Sub-tab switch** — a click on any of the four sub-tab buttons re-resolves
      the active view via :func:`active_view`, toggles each view container's
      visibility and each button's active class, and mirrors the active view
      name into :data:`ids.TUNE_ACTIVE_VIEW_STORE`.
    * **Monitor poll** — every 3 s the poll re-reads the study
      (:func:`read_study_for_monitor`) and re-renders the objective / importance
      figures, the gap badge, the trials table, and the degrade note.

    Args:
        app: The :class:`dash.Dash` instance whose layout is assigned.
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


__all__ = ["active_view", "read_study_for_monitor", "register_callbacks"]
