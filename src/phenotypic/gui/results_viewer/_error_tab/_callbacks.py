"""Callbacks for the Error-analysis tab.

Reactivity is debounced on the existing curation store. Per R2 the Error
tab has **no marking affordance** — every mark/unmark/relabel happens on
the Colony/QC/viewer-card tabs — so the ``active_tab → TAB_ERROR_ID``
Input is the effective trigger: returning to the Error tab always
recomputes :func:`_recompute`, reading ``filtered_state.labels`` fresh
under the lock, which reflects relabels (category reassignments) even
though ``STORE_REMOVED_KEYS`` is byte-identical for a relabel. The off-tab
``PreventUpdate`` gate prevents running the finder on every colony-view
mark (a real perf win) while still satisfying §8's live intent on tab
activation.

The load-bearing body lives in the module-level :func:`_recompute` helper
(Dash callback bugs only fire on ``/_dash-update-component``, so the body
is extracted for direct unit testing); the registered callbacks are thin
adapters around it.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from dash import Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from phenotypic.analysis import (
    ErrorCutoffFinder,
    filter_spec_json,
    filter_spec_query,
    render_error_analysis_html,
)
from phenotypic.analysis._error_cutoffs import RESULT_COLUMNS
from phenotypic.gui._design import category_color
from phenotypic.gui.results_viewer import _ids as viewer_ids
from phenotypic.gui.results_viewer._curation_labels import OTHER_CATEGORY
from phenotypic.gui.results_viewer._error_tab import _ids as ids
from phenotypic.gui.results_viewer._error_tab._data import (
    GoodMode,
    build_good_error_frames,
    category_counts,
    classify_at_cutoff,
    default_category,
    verified_good_keys,
)
from phenotypic.gui.results_viewer._error_tab._figure import build_distribution_figure
from phenotypic.tools_ import (
    error_analysis_csv_path,
    error_analysis_html_path,
    error_analysis_parquet_path,
    verified_parquet_path,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from phenotypic.gui.results_viewer._curation_labels import CurationLabels
    from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)

#: ``error_analysis.parquet`` column order: the engine's category-free
#: result columns prefixed with a leading ``category`` (R3) so the on-disk
#: shape matches Phase 5's all-category emit.
_PERSIST_COLUMNS: tuple[str, ...] = ("category", *RESULT_COLUMNS)

#: Pattern-matching id ``type`` for the per-category chip buttons. Single-sourced
#: so :func:`_chip_id` and the category-select callback's trigger check agree.
_CHIP_TYPE = "error-category-chip"


# ---------------------------------------------------------------------------
# Recompute result seam
# ---------------------------------------------------------------------------


@dataclass
class RecomputeResult:
    """The full output of one :func:`_recompute` pass.

    Attributes:
        category: The focused category token (may be ``None`` when there
            are no labels at all).
        empty_state: ``True`` when the finder had insufficient data and no
            table/figure/parquet was produced.
        table_data: ``DataTable`` row dicts (empty in the empty state).
        table_columns: ``DataTable`` column specs.
        figure: The good-vs-error distribution figure.
        focus: The focus-store payload (measurement value arrays + cutoff).
        good_n: Good-baseline object count actually screened.
        error_n: Error object count actually screened.
        verified_count: Verified-good object count (verified mode only).
        good_mode: The good-baseline mode this pass ran in.
        empty_message: The reason shown in the empty-state card (verified
            mode swaps in a "review more QC groups" prompt; R5).
    """

    category: str | None
    empty_state: bool
    table_data: list[dict[str, Any]] = field(default_factory=list)
    table_columns: list[dict[str, Any]] = field(default_factory=list)
    figure: go.Figure = field(default_factory=go.Figure)
    focus: dict[str, Any] = field(default_factory=dict)
    good_n: int = 0
    error_n: int = 0
    verified_count: int = 0
    good_mode: str = "all_unlabeled"
    empty_message: str = ""


#: Empty-state copy by reason. The verified-good message points the user at
#: the QC tab (the only place reviewed groups grow); R5.
_MSG_NO_LABELS = (
    "No error labels yet — mark objects with a category on the Colony or "
    "QC tab to begin."
)
_MSG_FEW_VERIFIED = (
    "Too few verified-good objects. Mark more QC groups reviewed on the QC "
    "tab to grow the verified baseline, or switch to All-unlabeled."
)
_MSG_NO_SEPARATION = (
    "No measurement separates good from this category yet — the labeled "
    "errors don't differ from the baseline on any measurement."
)


def _few_errors_message(category: str) -> str:
    """Empty-state copy when the error class is the limiting one."""
    return (
        f"Label more '{category}' objects (and keep a good baseline) before "
        "the cutoff finder can rank measurements reliably."
    )


# ---------------------------------------------------------------------------
# Module-level helpers (unit-testable without Dash)
# ---------------------------------------------------------------------------


def _recompute(
    output_root: "OutputRoot",
    filtered_state: "CurationLabels",
    category: str | None,
    good_mode: str,
) -> RecomputeResult:
    """Run the cutoff finder for one category and persist the result.

    Reads ``filtered_state.labels`` under the lock, builds the good/error
    frames in the chosen mode, runs :class:`ErrorCutoffFinder`, and — when
    there is enough data — persists ``error_analysis.{parquet,csv}`` for
    the focused category (R3) and, in verified mode, ``verified.parquet``
    (R4, only when the good frame is non-degenerate).

    Args:
        output_root: The active output root.
        filtered_state: The shared curation store.
        category: The focused category token (or ``None``).
        good_mode: ``"all_unlabeled"`` or ``"verified"``.

    Returns:
        A :class:`RecomputeResult` describing the table / figure / focus
        and whether the empty (need-more-labels) state applies.
    """
    with filtered_state._lock:
        labels = dict(filtered_state.labels)

    if category is None or category not in set(labels.values()):
        # The focused category may be stale (relabel removed it); fall back
        # to the default so the tab focuses *something* when labels exist.
        category = default_category(category_counts(labels), OTHER_CATEGORY)
    if category is None:
        return RecomputeResult(
            category=None,
            empty_state=True,
            good_mode=good_mode,
            empty_message=_MSG_NO_LABELS,
        )

    good_pdf, error_pdf = build_good_error_frames(
        output_root, labels, category, cast(GoodMode, good_mode)
    )
    finder = ErrorCutoffFinder()

    verified_count = 0
    if good_mode == "verified":
        verified_count = len(verified_good_keys(output_root, set(labels.keys())))

    if not finder.enough_data(good_pdf, error_pdf):
        # R5: in verified mode, a short good class means "review more QC
        # groups"; otherwise the error class is the limiting one.
        if good_mode == "verified" and len(good_pdf) < finder.min_good_n:
            message = _MSG_FEW_VERIFIED
        else:
            message = _few_errors_message(category)
        return RecomputeResult(
            category=category,
            empty_state=True,
            verified_count=verified_count,
            good_mode=good_mode,
            empty_message=message,
        )

    res = finder.analyze(good_pdf, error_pdf)
    if res.empty:
        return RecomputeResult(
            category=category,
            empty_state=True,
            verified_count=verified_count,
            good_mode=good_mode,
            empty_message=_MSG_NO_SEPARATION,
        )

    top = res.iloc[0]
    measurement = str(top["measurement"])
    direction = str(top["direction"])
    cutoff = float(top["cutoff"])
    good_values = _measurement_values(good_pdf, measurement)
    error_values = _measurement_values(error_pdf, measurement)

    figure = build_distribution_figure(
        good_values, error_values, measurement, category, cutoff
    )
    focus = {
        "category": category,
        "measurement": measurement,
        "direction": direction,
        "cutoff": cutoff,
        "good_values": good_values.tolist(),
        "error_values": error_values.tolist(),
    }

    _persist(output_root, category, res)
    if good_mode == "verified":
        _persist_verified(output_root, good_pdf)

    return RecomputeResult(
        category=category,
        empty_state=False,
        table_data=cast("list[dict[str, Any]]", res.to_dict("records")),
        table_columns=_table_columns(res),
        figure=figure,
        focus=focus,
        good_n=int(len(good_pdf)),
        error_n=int(len(error_pdf)),
        verified_count=verified_count,
        good_mode=good_mode,
    )


def _measurement_values(frame: pd.DataFrame, measurement: str) -> np.ndarray:
    """Return ``measurement``'s float values from a frame (empty if absent)."""
    if measurement not in frame.columns:
        return np.empty(0, dtype=float)
    return pd.to_numeric(frame[measurement], errors="coerce").to_numpy(dtype=float)


def _table_columns(res: pd.DataFrame) -> list[dict[str, Any]]:
    """Build ``DataTable`` column specs (numeric columns get 4-sig rounding)."""
    columns: list[dict[str, Any]] = []
    for col in res.columns:
        spec: dict[str, Any] = {"name": col, "id": col}
        if pd.api.types.is_numeric_dtype(res[col]):
            spec["type"] = "numeric"
            spec["format"] = {"specifier": ".4g"}
        columns.append(spec)
    return columns


def _persist(output_root: "OutputRoot", category: str, res: pd.DataFrame) -> None:
    """Write ``error_analysis.{parquet,csv}`` for the focused category (R3).

    Prepends a leading ``category`` column so the on-disk shape matches
    Phase 5's all-category concatenation. The GUI writes the focused
    category's rows (transient, last-viewed-in-session); Phase 5 finalize
    owns the authoritative all-category emit.
    """
    out = res.copy()
    out.insert(0, "category", category)
    out = out[list(_PERSIST_COLUMNS)]
    root = Path(output_root.root)
    _atomic_write_parquet(out, error_analysis_parquet_path(root))
    _atomic_write_csv(out, error_analysis_csv_path(root))


def _persist_verified(output_root: "OutputRoot", good_pdf: pd.DataFrame) -> None:
    """Write ``verified.parquet`` from the verified-good frame (R4).

    Only called from the non-degenerate verified branch (the caller has
    already confirmed ``enough_data``), so an empty ``verified.parquet`` is
    never written.
    """
    if good_pdf.empty:
        return
    _atomic_write_parquet(good_pdf, verified_parquet_path(Path(output_root.root)))


def _render_chips(
    counts: dict[str, int],
    focused: str | None,
    custom_categories: list[str],
) -> list[Any]:
    """Build the per-category chip row (selected chip carries ``is-selected``).

    Args:
        counts: Per-category tallies from :func:`category_counts`.
        focused: The currently-focused category token, if any.
        custom_categories: Ordered registered custom tokens, so a custom
            chip's swatch uses its **registration** index (matching the
            tile badge / radial wedge), not its sorted-row position.

    Returns:
        A list of chip components (one per labeled category).
    """
    custom_index_of = {tok: i for i, tok in enumerate(custom_categories)}
    chips: list[Any] = []
    for token, count in sorted(counts.items()):
        color = category_color(token, custom_index_of.get(token, 0))
        selected = token == focused
        chips.append(
            html.Button(
                [
                    html.Span(
                        "",
                        className="error-chip-dot",
                        style={
                            "display": "inline-block",
                            "width": "0.6rem",
                            "height": "0.6rem",
                            "borderRadius": "50%",
                            "background": color,
                            "marginRight": "0.3rem",
                        },
                    ),
                    html.Span(token),
                    html.Span(f" ({count})", className="error-chip-count"),
                ],
                id=_chip_id(token),
                n_clicks=0,
                type="button",
                className="error-category-chip"
                + (" is-selected" if selected else ""),
            )
        )
    return chips


def _chip_id(token: Any) -> dict[str, Any]:
    """Pattern-matching id for a category chip button.

    ``token`` is normally a category string; ``dash.ALL`` is also accepted
    so the category-select callback's ``Input`` can match every chip.
    """
    return {"type": _CHIP_TYPE, "token": token}


def _render_readout(metrics: dict[str, float]) -> list[Any]:
    """Build the recall / specificity / good-flagged readout pills."""
    return [
        html.Span(
            f"recall {metrics['recall']:.2f}", className="error-pill"
        ),
        html.Span(
            f"specificity {metrics['specificity']:.2f}", className="error-pill"
        ),
        html.Span(
            f"good flagged {int(metrics['good_flagged'])}", className="error-pill"
        ),
    ]


def _filter_spec_text(measurement: str, direction: str, cutoff: float) -> str:
    """Compose the copy-able filter spec (query line + JSON block)."""
    return (
        f"{filter_spec_query(measurement, direction, cutoff)}\n"
        f"{filter_spec_json(measurement, direction, cutoff)}"
    )


def _parse_drag_cutoff(relayout: dict[str, Any] | None) -> float | None:
    """Extract the dragged cutoff from ``relayoutData`` (R8).

    Plotly emits an editable-shape drag as flat dotted-string keys
    (``{"shapes[0].y0": …, "shapes[0].y1": …}``), not a nested dict.
    Returns ``None`` when no shape-y key is present (drags can emit
    partial/other keys).
    """
    if not isinstance(relayout, dict):
        return None
    for key in ("shapes[0].y0", "shapes[0].y1"):
        if key in relayout:
            try:
                return float(relayout[key])
            except (TypeError, ValueError):
                return None
    return None


def _stale_banner_text(filtered_state: "CurationLabels") -> tuple[bool, str]:
    """Return ``(is_open, message)`` for the re-key / stale banner."""
    report = filtered_state.rekey_report
    messages: list[str] = []
    if report.rekeyed or report.dropped:
        messages.append(
            f"{report.rekeyed} label(s) re-keyed, {report.dropped} dropped "
            "against the current master."
        )
    if filtered_state.stale:
        messages.append(
            "The measurements mirror changed on disk since this session "
            "loaded; reload to curate against the fresh master."
        )
    if not messages:
        return False, ""
    return True, " ".join(messages)


# ---------------------------------------------------------------------------
# Atomic writers (local equivalents of the curation store's, R4)
# ---------------------------------------------------------------------------


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet via a sibling temp file + ``os.replace`` (atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pl.from_pandas(df).write_parquet(tmp)
    os.replace(tmp, path)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV via a sibling temp file + ``os.replace`` (atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pl.from_pandas(df).write_csv(tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_error_callbacks(
    app: dash.Dash,
    output_root: "OutputRoot",
    filtered_state: "CurationLabels",
) -> None:
    """Register the Error-analysis tab's callbacks on *app*.

    Args:
        app: The Dash application owning the tab's layout.
        output_root: The active output root (read server-side).
        filtered_state: The shared :class:`CurationLabels` store.
    """

    @app.callback(
        Output(ids.ERROR_CATEGORY_CHIPS_ID, "children"),
        Output(ids.ERROR_VERIFIED_COUNT_ID, "children"),
        Output(ids.ERROR_VERIFIED_COUNT_ID, "style"),
        Output(ids.ERROR_TABLE_ID, "data"),
        Output(ids.ERROR_TABLE_ID, "columns"),
        Output(ids.ERROR_FIGURE_ID, "figure"),
        Output(ids.STORE_ERROR_FOCUS_ID, "data"),
        Output(ids.ERROR_EMPTY_STATE_ID, "style"),
        Output(ids.ERROR_EMPTY_STATE_MSG_ID, "children"),
        Output(ids.ERROR_CONTENT_ID, "style"),
        Output(ids.ERROR_STALE_BANNER_ID, "children"),
        Output(ids.ERROR_STALE_BANNER_ID, "is_open"),
        Input(viewer_ids.STORE_REMOVED_KEYS, "data"),
        Input(viewer_ids.TABS_ID, "active_tab"),
        Input(ids.ERROR_GOOD_MODE_TOGGLE_ID, "value"),
        Input(ids.STORE_ERROR_CATEGORY_ID, "data"),
        prevent_initial_call=False,
    )
    def _recompute_cb(
        _removed_keys: list[Any] | None,
        active_tab: str | None,
        good_mode: str | None,
        focused_category: str | None,
    ) -> tuple[Any, ...]:
        # Off-tab: don't run the finder on every colony-view mark (R2). The
        # tab refreshes on activation, satisfying §8's live intent.
        if active_tab != viewer_ids.TAB_ERROR_ID:
            raise PreventUpdate

        mode = good_mode or "all_unlabeled"
        result = _recompute(output_root, filtered_state, focused_category, mode)
        with filtered_state._lock:
            counts = category_counts(dict(filtered_state.labels))
            custom = list(filtered_state.custom_categories)
        chips = _render_chips(counts, result.category, custom)
        verified_style = (
            {"display": "inline-block"} if mode == "verified" else {"display": "none"}
        )
        verified_text = f"verified good: {result.verified_count}"
        stale_open, stale_text = _stale_banner_text(filtered_state)

        if result.empty_state:
            return (
                chips,
                verified_text,
                verified_style,
                [],
                [],
                go.Figure(),
                {},
                {"display": "block", "margin": "1rem"},
                result.empty_message,
                {"display": "none"},
                stale_text,
                stale_open,
            )
        return (
            chips,
            verified_text,
            verified_style,
            result.table_data,
            result.table_columns,
            result.figure,
            result.focus,
            {"display": "none"},
            no_update,
            {"padding": "0.5rem 1rem"},
            stale_text,
            stale_open,
        )

    @app.callback(
        Output(ids.STORE_ERROR_CATEGORY_ID, "data"),
        Input(_chip_id(dash.ALL), "n_clicks"),
        prevent_initial_call=True,
    )
    def _select_category(_clicks: list[int | None]) -> Any:
        triggered = ctx.triggered_id
        if isinstance(triggered, dict) and triggered.get("type") == _CHIP_TYPE:
            return triggered.get("token")
        return no_update

    @app.callback(
        Output(ids.ERROR_READOUT_ID, "children"),
        Output(ids.ERROR_CUTOFF_INPUT_ID, "value"),
        Output(ids.ERROR_FILTER_SPEC_ID, "value"),
        Input(ids.ERROR_FIGURE_ID, "relayoutData"),
        Input(ids.ERROR_CUTOFF_INPUT_ID, "value"),
        Input(ids.STORE_ERROR_FOCUS_ID, "data"),
        prevent_initial_call=True,
    )
    def _update_cutoff(
        relayout: dict[str, Any] | None,
        numeric_value: float | None,
        focus: dict[str, Any] | None,
    ) -> tuple[Any, Any, Any]:
        if not focus or "measurement" not in focus:
            raise PreventUpdate

        measurement = str(focus["measurement"])
        direction = str(focus["direction"])
        good_values = np.asarray(focus.get("good_values", []), dtype=float)
        error_values = np.asarray(focus.get("error_values", []), dtype=float)

        trigger = ctx.triggered_id
        if trigger == ids.ERROR_FIGURE_ID:
            cutoff = _parse_drag_cutoff(relayout)
            if cutoff is None:
                raise PreventUpdate
        elif trigger == ids.ERROR_CUTOFF_INPUT_ID:
            if numeric_value is None:
                raise PreventUpdate
            cutoff = float(numeric_value)
        else:
            # Focus store changed (new measurement): reset to suggested cutoff.
            cutoff = float(focus.get("cutoff", 0.0))

        metrics = classify_at_cutoff(good_values, error_values, cutoff, direction)
        readout = _render_readout(metrics)
        spec = _filter_spec_text(measurement, direction, cutoff)
        return readout, cutoff, spec

    @app.callback(
        Output(ids.ERROR_SAVE_TOAST_ID, "is_open"),
        Input(ids.ERROR_SAVE_REPORT_BTN_ID, "n_clicks"),
        State(ids.STORE_ERROR_FOCUS_ID, "data"),
        State(ids.ERROR_GOOD_MODE_TOGGLE_ID, "value"),
        prevent_initial_call=True,
    )
    def _save_report(
        n_clicks: int | None,
        focus: dict[str, Any] | None,
        good_mode: str | None,
    ) -> bool:
        if not n_clicks or not focus or "category" not in focus:
            raise PreventUpdate
        category = str(focus["category"])
        with filtered_state._lock:
            labels = dict(filtered_state.labels)
        good_pdf, error_pdf = build_good_error_frames(
            output_root, labels, category, cast(GoodMode, good_mode or "all_unlabeled")
        )
        res = ErrorCutoffFinder().analyze(good_pdf, error_pdf)
        html_str = render_error_analysis_html(category, res)
        path = error_analysis_html_path(Path(output_root.root))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html_str, encoding="utf-8")
        return True


__all__ = ["register_error_callbacks", "RecomputeResult"]
