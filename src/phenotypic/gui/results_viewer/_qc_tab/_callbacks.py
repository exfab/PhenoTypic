"""Callbacks for the QC tab.

Five primary callbacks plus the modal open/save/cancel flow:

* :func:`_render_card_shells` — card-list render. Fires on
  :data:`STORE_QC_RECIPE_REVISION` only. Owns the cards-container
  ``children`` list atomically so card count tracks the recipe.
* :func:`_refresh_qc_card_bodies` — card-body refresh. Fires on
  :data:`STORE_REMOVED_KEYS` (and ``STORE_QC_RECIPE_REVISION`` for
  ordering). Updates every card's figure, summary strip, and status
  badge; writes the merged :data:`CFG_QC_AUGMENTED_FRAME`; bumps
  :data:`STORE_QC_AUGMENTED_REVISION` for the heatmap subscriber.
* :func:`_on_modal_open` — opens the shared modal in add / edit /
  duplicate mode.
* :func:`_on_modal_class_change` — re-renders the param form when the
  class dropdown changes.
* :func:`_on_modal_submit` — persists the modal's state via
  :meth:`QcRecipe.add` or :meth:`QcRecipe.update`; bumps the recipe
  revision.
* :func:`_on_card_action` — single fan-in for delete/duplicate/toggle
  buttons.
* :func:`_mark_flagged_for_removal` — push a card's flagged keys onto
  :data:`STORE_REMOVED_KEYS`.
* :func:`_on_export_click` — write ``qc.parquet`` +
  ``qc_summary.json`` into ``deliverables/qc/`` (via
  ``OutputRoot.layout.qc_dir``); show a toast.

See spec lines 842-987 for the UX, lines 893-911 for the callback
split, and lines 818-840 for the export format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import dash
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from dash import ALL, Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate
from flask import current_app

from phenotypic.gui._config import (
    CFG_FILTERED_STATE,
    CFG_OPERATION_REGISTRY,
    CFG_OUTPUT_ROOT,
    CFG_QC_AUGMENTED_FRAME,
    CFG_QC_RECIPE,
)
from phenotypic.gui._design import COLOR_MUTED, OI_VERMILION, OI_VERMILION_TEXT
from phenotypic.sdk_.viz.figures import apply_theme
from phenotypic.gui._operation_registry import OperationRegistry
from phenotypic.gui._param_forms import param_form, parse_widget_value
from phenotypic.gui.results_viewer import _ids as viewer_ids
from phenotypic.gui.results_viewer._filtered_state import (
    KEY_COLUMNS,
    get_curated_frame,
)
from phenotypic.gui.results_viewer._qc_tab import _ids as ids
from phenotypic.gui.results_viewer._qc_tab._check_card import build_check_card
from phenotypic.gui.results_viewer._qc_tab._layout import (
    _banner_style,
    _render_load_warnings,
)
from phenotypic.gui.results_viewer._qc_tab.review import (
    _ids as review_ids,
    register_review_callbacks,
)
from phenotypic.sdk_._qc_recipe import QcRecipe

logger = logging.getLogger(__name__)


#: Bootstrap colour names used by status badges. Mapped from the
#: tri-state QC status emitted by :meth:`QualityCheck.summary`.
_BADGE_COLOR_BY_STATUS: dict[str, str] = {
    "pass": "success",
    "warn": "warning",
    "fail": "danger",
}

#: Status rank for "worst-status wins" reductions across a summary frame.
_STATUS_RANK: dict[str, int] = {"pass": 0, "warn": 1, "fail": 2}
_INV_STATUS_RANK: dict[int, str] = {v: k for k, v in _STATUS_RANK.items()}


# ---------------------------------------------------------------------------
# Module-level helpers (pure)
# ---------------------------------------------------------------------------


def _empty_figure(message: str) -> go.Figure:
    """Build an "empty"-state figure with a centred message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 12},
    )
    apply_theme(fig)
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 10, "b": 10},
        height=320,
    )
    return fig


def _error_figure(*, check_name: str, message: str) -> go.Figure:
    """Build an error-state figure surfacing a per-card analyze failure."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"{check_name} failed:<br>{message}",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"color": OI_VERMILION, "size": 12},
        align="center",
    )
    apply_theme(fig)
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 10, "b": 10},
        height=320,
    )
    return fig


def _render_summary_strip(summary_df: pd.DataFrame) -> str:
    """Render the summary strip text from a check's per-group summary.

    Example output: ``"groups: 4 | flagged: 1 | worst metric: 0.12"``
    (spec line 1218).

    Args:
        summary_df: One-row-per-group frame as produced by
            :meth:`QualityCheck.summary`. Must carry the columns
            ``qc_n_flagged``, ``qc_worst_metric`` and ``qc_status``.

    Returns:
        A single string for display in the per-card summary chip. NaN
        ``qc_worst_metric`` is rendered as ``"nan"``.
    """
    groups = int(len(summary_df))
    if groups == 0 or "qc_n_flagged" not in summary_df.columns:
        return "groups: 0 | flagged: 0 | worst metric: nan"

    flagged = int(summary_df["qc_n_flagged"].fillna(0).astype(int).sum())
    worst_metric_raw = pd.to_numeric(
        summary_df["qc_worst_metric"], errors="coerce"
    )
    if worst_metric_raw.empty or worst_metric_raw.dropna().empty:
        metric_str = "nan"
    else:
        metric_str = f"{float(worst_metric_raw.max()):.2f}"
    return f"groups: {groups} | flagged: {flagged} | worst metric: {metric_str}"


def _worst_status(summary_df: pd.DataFrame) -> Literal["pass", "warn", "fail"]:
    """Return the worst status across a check's summary frame.

    ``"fail"`` wins over ``"warn"`` which wins over ``"pass"``. An empty
    frame or missing column is treated as ``"pass"`` so a degenerate
    check never spuriously alarms.
    """
    if summary_df.empty or "qc_status" not in summary_df.columns:
        return "pass"
    statuses = summary_df["qc_status"].astype(str).tolist()
    if not statuses:
        return "pass"
    worst_rank = max((_STATUS_RANK.get(s, 0) for s in statuses), default=0)
    return _INV_STATUS_RANK[worst_rank]  # type: ignore[return-value]


def _badge_color_for_status(status: str) -> str:
    """Map a QC status string to a Bootstrap badge ``color`` name."""
    return _BADGE_COLOR_BY_STATUS.get(status, "secondary")


def _left_join_qc_columns(
    left: pl.DataFrame,
    right: pd.DataFrame,
    *,
    on: tuple[str, str] = KEY_COLUMNS,
) -> pl.DataFrame:
    """Left-join a check's analyze() output onto the augmented frame.

    Args:
        left: The accumulator (polars) being augmented across cards.
        right: One check's :meth:`QualityCheck.analyze` output
            (pandas). Carries QC columns plus whatever rows the check
            iterated over.
        on: Join key columns. Defaults to
            :data:`KEY_COLUMNS` (``("Metadata_ImageFile", "Object_Label")``)
            — the curation key used by ``STORE_REMOVED_KEYS``.

    Returns:
        A new polars DataFrame with the same row count as ``left`` and
        the QC columns from ``right`` joined in. When either side lacks
        a join key column the join is skipped and ``left`` is returned
        unchanged so the augmented frame still has the rows the heatmap
        expects.
    """
    if right.empty:
        return left
    missing_left = [c for c in on if c not in left.columns]
    missing_right = [c for c in on if c not in right.columns]
    if missing_left or missing_right:
        logger.debug(
            "Skipping QC left-join: missing keys (left=%s, right=%s)",
            missing_left,
            missing_right,
        )
        return left

    # Restrict ``right`` to the join keys + the columns that are new vs.
    # ``left`` so the join doesn't clobber pre-existing measurement
    # columns when two checks emit overlapping severities.
    extra_cols = [c for c in right.columns if c not in left.columns]
    keep_cols = list(on) + extra_cols
    right_subset = right[keep_cols].drop_duplicates(subset=list(on), keep="first")

    right_pl = pl.from_pandas(right_subset)
    try:
        return left.join(right_pl, on=list(on), how="left")
    except Exception as exc:  # noqa: BLE001
        logger.warning("QC left-join failed: %s", exc, exc_info=True)
        return left


def _merge_removed_keys(
    current: Iterable[Any],
    new: Iterable[tuple[str, int]],
) -> list[list[Any]]:
    """Union two ``STORE_REMOVED_KEYS`` payloads, preserving order.

    Args:
        current: The existing payload as received from Dash. Typically a
            list of ``[image_file, label]`` two-element lists with
            possibly stringified labels.
        new: New keys to merge in.

    Returns:
        A list of ``[image_file, label]`` two-element lists with
        no duplicates, with ``current`` order preserved followed by
        any new entries.
    """
    seen: set[tuple[str, int]] = set()
    out: list[list[Any]] = []
    for entry in current or []:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        try:
            key = (str(entry[0]), int(entry[1]))
        except (TypeError, ValueError):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append([key[0], key[1]])
    for image_file, label in new:
        key = (str(image_file), int(label))
        if key in seen:
            continue
        seen.add(key)
        out.append([key[0], key[1]])
    return out


def _gather_modal_raw_values(
    *,
    prefix_marker: str,
    simple: Iterable[tuple[Any, Any]] = (),
    multi_tags: tuple[Any, Any] = ([], []),
    multi_values: tuple[Any, Any] = ([], []),
    column_scalars: tuple[Any, Any] = ([], []),
    column_modes: tuple[Any, Any] = ([], []),
) -> dict[str, Any]:
    """Collect ``{param_name: raw_value}`` from the modal's widget state.

    Reduces the param form's pattern-matched ``(values, ids)`` lists to a
    flat ``{name: raw}`` mapping, keeping only widgets whose ``prefix``
    starts with ``prefix_marker`` (so the shared param-widget id types don't
    leak in from another modal/tool). Single-widget kinds map their value
    directly. Two-id widgets are repacked into the tuple that
    :func:`parse_widget_value` dispatches on, mirroring the analysis
    sub-app's ``_apply_param_edit`` multi-component handling:

    * **multi-type unions** — a ``param-multi-tag`` selector + a
      ``param-multi-value`` input sharing one ``(prefix, name)`` → packed as
      ``(tag, value)``. Without this a genuine multi-primitive union param
      (e.g. ``bool | float``) would be silently dropped on save.
    * **column-with-alt** (``ColumnRef | None``) — a ``param-column-mode``
      toggle + a ``param-column-scalar`` dropdown → packed as
      ``(mode, scalar)`` so a "None" selection round-trips as ``None``. A
      *plain* ``ColumnRef`` scalar has no mode toggle, so it stays a bare
      scalar value.

    Args:
        prefix_marker: Form-id prefix that scopes this modal's widgets
            (e.g. ``"qc-modal-"``).
        simple: Iterable of ``(values, ids)`` pairs for the single-widget
            kinds (bool/num/str/enum/list/tuple/column-multi).
        multi_tags: The ``(values, ids)`` pair for ``param-multi-tag``.
        multi_values: The ``(values, ids)`` pair for ``param-multi-value``.
        column_scalars: The ``(values, ids)`` pair for
            ``param-column-scalar`` (plain or column-with-alt).
        column_modes: The ``(values, ids)`` pair for ``param-column-mode``
            (only emitted for column-with-alt params).

    Returns:
        Mapping of parameter name to its raw widget value — a scalar for
        single widgets, a ``(tag, value)`` tuple for multi-union widgets,
        and a ``(mode, scalar)`` tuple for column-with-alt widgets.
    """
    def _in_scope(pair: tuple[Any, Any]) -> Iterator[tuple[str, str, Any]]:
        """Yield ``(prefix, name, value)`` for this modal's widgets in a pair.

        Filters a single ``(values, ids)`` pattern-match pair down to the
        widgets whose id ``prefix`` belongs to this modal, skipping any
        malformed or out-of-scope ids.
        """
        values, ids_state = pair
        for value, id_dict in zip(values or [], ids_state or []):
            if not isinstance(id_dict, dict):
                continue
            prefix = id_dict.get("prefix", "")
            name = id_dict.get("name")
            if not isinstance(prefix, str) or not prefix.startswith(prefix_marker):
                continue
            if isinstance(name, str):
                yield prefix, name, value

    def _selector_by_key(pair: tuple[Any, Any]) -> dict[tuple[str, str], Any]:
        """Index a two-id widget's selector values by ``(prefix, name)``."""
        return {(prefix, name): value for prefix, name, value in _in_scope(pair)}

    raw_by_name: dict[str, Any] = {}
    for pair in simple:
        for _prefix, name, value in _in_scope(pair):
            raw_by_name[name] = value

    # Multi-type unions: (param-multi-tag, param-multi-value) → (tag, value).
    tag_by_key = _selector_by_key(multi_tags)
    for prefix, name, value in _in_scope(multi_values):
        raw_by_name[name] = (tag_by_key.get((prefix, name)), value)

    # Column-with-alt: (param-column-mode, param-column-scalar) → (mode,
    # scalar). A plain ColumnRef scalar has no mode and stays a bare value.
    mode_by_key = _selector_by_key(column_modes)
    for prefix, name, value in _in_scope(column_scalars):
        key = (prefix, name)
        raw_by_name[name] = (mode_by_key[key], value) if key in mode_by_key else value

    return raw_by_name


# ---------------------------------------------------------------------------
# Module-level helpers (Flask-app-aware)
# ---------------------------------------------------------------------------


def _get_recipe() -> QcRecipe:
    """Read the active :class:`QcRecipe` off the Flask app config."""
    recipe = current_app.config.get(CFG_QC_RECIPE)
    if not isinstance(recipe, QcRecipe):  # pragma: no cover - defensive
        raise RuntimeError(
            "QC tab callbacks fired but CFG_QC_RECIPE is not a QcRecipe."
        )
    return recipe


def _get_registry() -> OperationRegistry | None:
    """Return the operation registry, or ``None`` when unavailable.

    The viewer's ``create_app`` builds an :class:`OperationRegistry` and
    stashes it on *this* server's ``app.server.config`` (each sub-app has
    its own Flask server under the hub's ``DispatcherMiddleware``, so the
    builder's registry is not visible here). The ``None`` fallback is kept
    as a defensive guard for partially-initialized apps (e.g. the
    empty-state pathway): when it triggers the class picker renders an
    empty dropdown and the modal effectively becomes read-only.
    """
    registry = current_app.config.get(CFG_OPERATION_REGISTRY)
    if isinstance(registry, OperationRegistry):
        return registry
    return None


def _columns_provider(source: str) -> list[str]:
    """Resolve a ``ColumnSource`` to the live column list.

    Used by the modal's param form so column-aware widgets (e.g.
    ``groupby``, ``on``) populate from the live measurements schema.
    """
    from phenotypic.gui._config import CFG_MEASUREMENT_SCHEMA

    schema = current_app.config.get(CFG_MEASUREMENT_SCHEMA)
    if schema is None:
        return []
    try:
        return list(schema.columns_for(source))
    except Exception:  # noqa: BLE001 - defensive
        logger.warning("Schema lookup failed", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Registration entry point
# ---------------------------------------------------------------------------


def register_qc_callbacks(app: dash.Dash) -> None:
    """Register the QC tab's callbacks on *app*.

    The Flask app config must already carry:

    * :data:`CFG_QC_RECIPE` — the loaded :class:`QcRecipe`.
    * :data:`CFG_OUTPUT_ROOT` — the validated ``OutputRoot``.
    * :data:`CFG_FILTERED_STATE` — the loaded curation state.

    Args:
        app: The Dash application that will own the callbacks.
    """
    # -----------------------------------------------------------------
    # Callback A: card-list render
    # -----------------------------------------------------------------
    @app.callback(
        Output(ids.QC_CARDS_CONTAINER_ID, "children"),
        Output(ids.QC_EXPORT_BTN_ID, "disabled"),
        Output(ids.QC_LOAD_WARNING_BANNER_ID, "children"),
        Output(ids.QC_LOAD_WARNING_BANNER_ID, "style"),
        Input(viewer_ids.STORE_QC_RECIPE_REVISION, "data"),
    )
    def _render_card_shells(_revision: int | None) -> tuple[Any, bool, Any, dict[str, str]]:
        """Rebuild the cards-container children list on every revision tick."""
        recipe = _get_recipe()
        shells = [build_check_card(entry) for entry in recipe.entries if entry.enabled]
        export_disabled = not any(e.enabled for e in recipe.entries)
        return (
            shells,
            export_disabled,
            _render_load_warnings(recipe.load_warnings),
            _banner_style(recipe.load_warnings),
        )

    # -----------------------------------------------------------------
    # Callback B: card-body refresh
    # -----------------------------------------------------------------
    @app.callback(
        Output({"type": "qc-card-figure", "index": ALL}, "figure"),
        Output({"type": "qc-card-summary", "index": ALL}, "children"),
        Output({"type": "qc-card-status-badge", "index": ALL}, "children"),
        Output({"type": "qc-card-status-badge", "index": ALL}, "color"),
        Output(viewer_ids.STORE_QC_AUGMENTED_REVISION, "data"),
        Input(viewer_ids.STORE_REMOVED_KEYS, "data"),
        Input(viewer_ids.STORE_QC_RECIPE_REVISION, "data"),
        State({"type": "qc-card-root", "index": ALL}, "id"),
        State(viewer_ids.STORE_QC_AUGMENTED_REVISION, "data"),
    )
    def _refresh_qc_card_bodies(
        _removed_keys: list[Any] | None,
        _recipe_revision: int | None,
        ids_list: list[dict[str, str]] | None,
        aug_rev: int | None,
    ) -> tuple[list[go.Figure], list[str], list[str], list[str], int]:
        """Re-run every enabled check and refresh card bodies.

        Per-card error isolation: a single check raising never halts the
        callback; the offending card surfaces the error inline.
        """
        recipe = _get_recipe()
        filtered = current_app.config.get(CFG_FILTERED_STATE)
        output_root = current_app.config.get(CFG_OUTPUT_ROOT)
        ids_list = ids_list or []

        if filtered is None or output_root is None or not ids_list:
            # No frame available or no cards mounted — return a coherent
            # length-matched empty payload so Dash's ALL contract holds.
            return (
                [_empty_figure("No data")] * len(ids_list),
                [""] * len(ids_list),
                ["pass"] * len(ids_list),
                ["secondary"] * len(ids_list),
                (aug_rev or 0) + 1,
            )

        augmented = get_curated_frame(filtered, output_root)
        try:
            pandas_frame = augmented.to_pandas()
        except Exception as exc:  # noqa: BLE001
            logger.warning("QC frame conversion failed: %s", exc, exc_info=True)
            return (
                [_error_figure(check_name="(frame)", message=str(exc))]
                * len(ids_list),
                [f"error: {exc!s}"] * len(ids_list),
                ["error"] * len(ids_list),
                ["danger"] * len(ids_list),
                (aug_rev or 0) + 1,
            )

        instances = dict(recipe.instantiate())
        figures: list[go.Figure] = []
        summaries: list[str] = []
        badge_text: list[str] = []
        badge_color: list[str] = []
        has_any = False

        for component_id in ids_list:
            instance_id = component_id.get("index", "")
            check = instances.get(instance_id)
            if check is None:
                figures.append(_empty_figure("(removed)"))
                summaries.append("")
                badge_text.append("?")
                badge_color.append("secondary")
                continue
            try:
                result = check.analyze(pandas_frame)
                has_any = True
            except Exception as exc:  # noqa: BLE001 - per-card isolation
                logger.warning(
                    "QC check %s (%s) analyze failed: %s",
                    instance_id,
                    type(check).__name__,
                    exc,
                    exc_info=True,
                )
                figures.append(
                    _error_figure(check_name=type(check).__name__, message=str(exc))
                )
                summaries.append(f"error: {exc!s}")
                badge_text.append("error")
                badge_color.append("danger")
                continue

            try:
                figure = check.dash()
                # Stamp the shared theme so QC check figures carry the
                # Okabe-Ito colorway, mono numeric axes, and brand chrome.
                apply_theme(figure)
            except Exception as exc:  # noqa: BLE001
                logger.warning("QC dash() failed: %s", exc, exc_info=True)
                figure = _error_figure(check_name=type(check).__name__, message=str(exc))

            summary = check.summary()
            summaries.append(_render_summary_strip(summary))
            worst = _worst_status(summary)
            badge_text.append(worst)
            badge_color.append(_badge_color_for_status(worst))
            figures.append(figure)

            augmented = _left_join_qc_columns(augmented, result)

        current_app.config[CFG_QC_AUGMENTED_FRAME] = augmented if has_any else None

        return figures, summaries, badge_text, badge_color, (aug_rev or 0) + 1

    # -----------------------------------------------------------------
    # Callback C: modal open (add / edit / duplicate)
    # -----------------------------------------------------------------
    @app.callback(
        Output(ids.QC_MODAL_ID, "is_open"),
        Output(ids.QC_MODAL_TITLE_ID, "children"),
        Output(ids.QC_MODAL_CLASS_PICKER_ID, "options"),
        Output(ids.QC_MODAL_CLASS_PICKER_ID, "value"),
        Output(ids.STORE_QC_EDITING_INSTANCE, "data"),
        Input(ids.QC_ADD_CHECK_BTN_ID, "n_clicks"),
        Input({"type": "qc-card-edit", "index": ALL}, "n_clicks"),
        Input(ids.QC_MODAL_CANCEL_BTN_ID, "n_clicks"),
        State(ids.QC_MODAL_ID, "is_open"),
        prevent_initial_call=True,
    )
    def _on_modal_open(
        _add_clicks: int | None,
        _edit_clicks: list[int | None] | None,
        _cancel_clicks: int | None,
        _is_open: bool,
    ) -> tuple[bool, str, Any, Any, str | None]:
        """Open the modal in add / edit mode; cancel closes it."""
        triggered = ctx.triggered_id
        if triggered is None:
            raise PreventUpdate

        # Pattern-matching callbacks always fire once at boot with all
        # ``n_clicks`` at ``None`` (the ``qc-card-edit`` ALL set populates
        # when the cards first render); skip when no actual click is
        # recorded so the modal never opens on initial layout.
        if not any(item.get("value") for item in ctx.triggered or []):
            raise PreventUpdate

        # Cancel -> just close.
        if triggered == ids.QC_MODAL_CANCEL_BTN_ID:
            return False, "Add QC check", no_update, no_update, None

        # Build class-picker options from the live registry.
        registry = _get_registry()
        if registry is not None:
            options = [
                {"label": info.name, "value": info.name}
                for info in registry.get_by_category("quality_check")
            ]
        else:
            options = []

        # Edit -> pre-populate from the recipe.
        if isinstance(triggered, dict) and triggered.get("type") == "qc-card-edit":
            instance_id = str(triggered["index"])
            recipe = _get_recipe()
            entry = next(
                (e for e in recipe.entries if e.instance_id == instance_id), None
            )
            if entry is None:  # pragma: no cover - defensive
                raise PreventUpdate
            return True, "Edit QC check", options, entry.cls.__name__, instance_id

        # Default branch: Add. STORE_QC_EDITING_INSTANCE clears to None
        # so the submit callback dispatches to QcRecipe.add.
        return True, "Add QC check", options, None, None

    # -----------------------------------------------------------------
    # Callback D: modal class picker -> param form re-render
    # -----------------------------------------------------------------
    @app.callback(
        Output(ids.QC_MODAL_PARAMS_REGION_ID, "children"),
        Input(ids.QC_MODAL_CLASS_PICKER_ID, "value"),
        State(ids.STORE_QC_EDITING_INSTANCE, "data"),
    )
    def _on_modal_class_change(
        class_name: str | None,
        editing_instance_id: str | None,
    ) -> Any:
        """Re-render the param form when the class dropdown changes."""
        if not class_name:
            return html.Div(
                "Pick a check class to configure its parameters.",
                style={"color": COLOR_MUTED, "fontStyle": "italic"},
            )
        registry = _get_registry()
        if registry is None:
            return html.Div(
                "OperationRegistry unavailable -- cannot render params.",
                style={"color": OI_VERMILION_TEXT},
            )
        info = registry.get(class_name)
        if info is None:
            return html.Div(
                f"Unknown class {class_name!r} in registry.",
                style={"color": OI_VERMILION_TEXT},
            )

        # When editing, seed with the entry's current params; otherwise
        # use the empty dict so each param's default flows in.
        current_values: dict[str, Any] = {}
        if editing_instance_id:
            recipe = _get_recipe()
            entry = next(
                (e for e in recipe.entries if e.instance_id == editing_instance_id),
                None,
            )
            if entry is not None and entry.cls.__name__ == class_name:
                current_values = dict(entry.params)

        return param_form(
            info,
            current_values=current_values,
            form_id_prefix=f"qc-modal-{class_name}",
            columns_provider=_columns_provider,
        )

    # -----------------------------------------------------------------
    # Callback E: modal submit
    # -----------------------------------------------------------------
    @app.callback(
        Output(viewer_ids.STORE_QC_RECIPE_REVISION, "data", allow_duplicate=True),
        Output(ids.QC_MODAL_ID, "is_open", allow_duplicate=True),
        Output(ids.STORE_QC_EDITING_INSTANCE, "data", allow_duplicate=True),
        Input(ids.QC_MODAL_SUBMIT_BTN_ID, "n_clicks"),
        State(ids.QC_MODAL_CLASS_PICKER_ID, "value"),
        State(ids.STORE_QC_EDITING_INSTANCE, "data"),
        State(viewer_ids.STORE_QC_RECIPE_REVISION, "data"),
        State({"type": "param-bool", "prefix": ALL, "name": ALL}, "value"),
        State({"type": "param-bool", "prefix": ALL, "name": ALL}, "id"),
        State({"type": "param-num", "prefix": ALL, "name": ALL}, "value"),
        State({"type": "param-num", "prefix": ALL, "name": ALL}, "id"),
        State({"type": "param-str", "prefix": ALL, "name": ALL}, "value"),
        State({"type": "param-str", "prefix": ALL, "name": ALL}, "id"),
        State({"type": "param-enum", "prefix": ALL, "name": ALL}, "value"),
        State({"type": "param-enum", "prefix": ALL, "name": ALL}, "id"),
        State({"type": "param-list", "prefix": ALL, "name": ALL}, "value"),
        State({"type": "param-list", "prefix": ALL, "name": ALL}, "id"),
        State({"type": "param-tuple", "prefix": ALL, "name": ALL}, "value"),
        State({"type": "param-tuple", "prefix": ALL, "name": ALL}, "id"),
        State({"type": "param-column-scalar", "prefix": ALL, "name": ALL}, "value"),
        State({"type": "param-column-scalar", "prefix": ALL, "name": ALL}, "id"),
        State({"type": "param-column-multi", "prefix": ALL, "name": ALL}, "value"),
        State({"type": "param-column-multi", "prefix": ALL, "name": ALL}, "id"),
        State({"type": "param-column-mode", "prefix": ALL, "name": ALL}, "value"),
        State({"type": "param-column-mode", "prefix": ALL, "name": ALL}, "id"),
        State({"type": "param-multi-tag", "prefix": ALL, "name": ALL}, "value"),
        State({"type": "param-multi-tag", "prefix": ALL, "name": ALL}, "id"),
        State({"type": "param-multi-value", "prefix": ALL, "name": ALL}, "value"),
        State({"type": "param-multi-value", "prefix": ALL, "name": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _on_modal_submit(  # noqa: PLR0913 - state-rich gathering matches form widget set
        n_clicks: int | None,
        class_name: str | None,
        editing_instance_id: str | None,
        revision: int | None,
        bool_values: list[Any],
        bool_ids: list[dict[str, str]],
        num_values: list[Any],
        num_ids: list[dict[str, str]],
        str_values: list[Any],
        str_ids: list[dict[str, str]],
        enum_values: list[Any],
        enum_ids: list[dict[str, str]],
        list_values: list[Any],
        list_ids: list[dict[str, str]],
        tuple_values: list[Any],
        tuple_ids: list[dict[str, str]],
        col_scalar_values: list[Any],
        col_scalar_ids: list[dict[str, str]],
        col_multi_values: list[Any],
        col_multi_ids: list[dict[str, str]],
        col_mode_values: list[Any],
        col_mode_ids: list[dict[str, str]],
        multi_tag_values: list[Any],
        multi_tag_ids: list[dict[str, str]],
        multi_value_values: list[Any],
        multi_value_ids: list[dict[str, str]],
    ) -> tuple[int, bool, str | None]:
        """Persist the modal's state via :meth:`QcRecipe.add` / :meth:`update`."""
        if not n_clicks or not class_name:
            raise PreventUpdate

        registry = _get_registry()
        if registry is None:
            raise PreventUpdate
        info = registry.get(class_name)
        if info is None:
            raise PreventUpdate

        # Build {param_name: raw_value} from the pattern-matched state
        # tuples, filtered to widgets belonging to this modal's prefix.
        raw_by_name = _gather_modal_raw_values(
            prefix_marker="qc-modal-",
            simple=(
                (bool_values, bool_ids),
                (num_values, num_ids),
                (str_values, str_ids),
                (enum_values, enum_ids),
                (list_values, list_ids),
                (tuple_values, tuple_ids),
                (col_multi_values, col_multi_ids),
            ),
            multi_tags=(multi_tag_values, multi_tag_ids),
            multi_values=(multi_value_values, multi_value_ids),
            column_scalars=(col_scalar_values, col_scalar_ids),
            column_modes=(col_mode_values, col_mode_ids),
        )

        # Convert raw widget values to typed params using each ParamInfo.
        new_params: dict[str, Any] = {}
        for name, p in info.parameters.items():
            if name not in raw_by_name:
                continue
            try:
                new_params[name] = parse_widget_value(raw_by_name[name], p)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "QC modal: failed to parse %s=%r for %s",
                    name,
                    raw_by_name[name],
                    class_name,
                    exc_info=True,
                )

        recipe = _get_recipe()
        if editing_instance_id:
            ok = recipe.update(editing_instance_id, params=new_params)
            if not ok:
                logger.warning(
                    "QC modal: update failed for %s", editing_instance_id
                )
                raise PreventUpdate
        else:
            try:
                recipe.add(info.cls, new_params)
            except Exception as exc:  # noqa: BLE001
                logger.warning("QC modal: add failed: %s", exc, exc_info=True)
                raise PreventUpdate from exc

        return (revision or 0) + 1, False, None

    # -----------------------------------------------------------------
    # Callback F: card actions (delete / duplicate / toggle)
    # -----------------------------------------------------------------
    @app.callback(
        Output(viewer_ids.STORE_QC_RECIPE_REVISION, "data", allow_duplicate=True),
        Input({"type": "qc-card-delete", "index": ALL}, "n_clicks"),
        Input({"type": "qc-card-duplicate", "index": ALL}, "n_clicks"),
        Input({"type": "qc-card-toggle", "index": ALL}, "n_clicks"),
        State(viewer_ids.STORE_QC_RECIPE_REVISION, "data"),
        prevent_initial_call=True,
    )
    def _on_card_action(
        _delete_clicks: list[int | None] | None,
        _duplicate_clicks: list[int | None] | None,
        _toggle_clicks: list[int | None] | None,
        revision: int | None,
    ) -> int:
        """Single fan-in for the three per-card lifecycle buttons."""
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        # Pattern-matching callbacks always fire once at boot with all
        # ``n_clicks`` at ``None``; skip when no actual click is recorded.
        if not any(item.get("value") for item in ctx.triggered or []):
            raise PreventUpdate

        action_type = triggered.get("type")
        instance_id = str(triggered.get("index", ""))
        if not instance_id:
            raise PreventUpdate

        recipe = _get_recipe()

        if action_type == "qc-card-delete":
            recipe.remove(instance_id)
            return (revision or 0) + 1

        if action_type == "qc-card-toggle":
            entry = next(
                (e for e in recipe.entries if e.instance_id == instance_id), None
            )
            if entry is None:
                raise PreventUpdate
            recipe.update(instance_id, enabled=not entry.enabled)
            return (revision or 0) + 1

        if action_type == "qc-card-duplicate":
            entry = next(
                (e for e in recipe.entries if e.instance_id == instance_id), None
            )
            if entry is None:
                raise PreventUpdate
            try:
                recipe.add(entry.cls, dict(entry.params), enabled=entry.enabled)
            except Exception as exc:  # noqa: BLE001
                logger.warning("QC duplicate failed: %s", exc, exc_info=True)
                raise PreventUpdate from exc
            return (revision or 0) + 1

        raise PreventUpdate

    # -----------------------------------------------------------------
    # Callback G: Mark-flagged-for-removal
    # -----------------------------------------------------------------
    @app.callback(
        Output(viewer_ids.STORE_REMOVED_KEYS, "data", allow_duplicate=True),
        Input({"type": "qc-card-mark-flag", "index": ALL}, "n_clicks"),
        State(viewer_ids.STORE_REMOVED_KEYS, "data"),
        prevent_initial_call=True,
    )
    def _mark_flagged_for_removal(
        n_clicks_list: list[int | None] | None,
        current: list[Any] | None,
    ) -> list[list[Any]]:
        """Push a card's flagged keys into ``STORE_REMOVED_KEYS``."""
        if not n_clicks_list or not any(n_clicks_list):
            raise PreventUpdate
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        instance_id = str(triggered.get("index", ""))
        if not instance_id:
            raise PreventUpdate

        recipe = _get_recipe()
        filtered = current_app.config.get(CFG_FILTERED_STATE)
        output_root = current_app.config.get(CFG_OUTPUT_ROOT)
        if filtered is None or output_root is None:
            raise PreventUpdate

        # Re-instantiate + re-run analyze() so flagged_keys reflects the
        # current curation state, not a stale state.
        instances = dict(recipe.instantiate())
        check = instances.get(instance_id)
        if check is None:
            raise PreventUpdate
        frame = get_curated_frame(filtered, output_root).to_pandas()
        try:
            check.analyze(frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "QC mark-flag: analyze failed for %s: %s",
                instance_id,
                exc,
                exc_info=True,
            )
            raise PreventUpdate from exc

        new_keys = check.flagged_keys()
        if not new_keys:
            raise PreventUpdate
        return _merge_removed_keys(current or [], new_keys)

    # -----------------------------------------------------------------
    # Callback H: Export QC report
    # -----------------------------------------------------------------
    @app.callback(
        Output(ids.QC_EXPORT_TOAST_ID, "is_open"),
        Output(ids.QC_EXPORT_TOAST_ID, "children"),
        Output(ids.QC_EXPORT_TOAST_ID, "icon"),
        Input(ids.QC_EXPORT_BTN_ID, "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_export_click(
        n_clicks: int | None,
    ) -> tuple[bool, Any, str]:
        """Write ``qc.parquet`` + ``qc_summary.json`` and surface a toast."""
        if not n_clicks:
            raise PreventUpdate
        recipe = _get_recipe()
        filtered = current_app.config.get(CFG_FILTERED_STATE)
        output_root = current_app.config.get(CFG_OUTPUT_ROOT)
        if filtered is None or output_root is None:
            return True, "Output root unavailable.", "danger"

        try:
            parquet_path, summary_path = _export_qc_report(
                recipe=recipe,
                filtered=filtered,
                output_root=output_root,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("QC export failed: %s", exc, exc_info=True)
            return True, f"Export failed: {exc!s}", "danger"

        body = html.Div(
            [
                html.Div(f"Wrote {parquet_path}"),
                html.Div(f"Wrote {summary_path}"),
            ]
        )
        return True, body, "success"

    # -----------------------------------------------------------------
    # Callback G: Configure | Review sub-view toggle
    # -----------------------------------------------------------------
    @app.callback(
        Output(review_ids.QC_CONFIGURE_VIEW_ID, "style"),
        Output(review_ids.QC_REVIEW_VIEW_ID, "style"),
        Output(review_ids.STORE_QC_SUBVIEW, "data"),
        Input(review_ids.QC_SUBVIEW_TOGGLE_ID, "value"),
    )
    def _switch_subview(
        subview: str | None,
    ) -> tuple[dict[str, str], dict[str, str], str]:
        """Show the selected sub-view via ``style.display`` (no rebuild)."""
        review = (subview or review_ids.QC_SUBVIEW_CONFIGURE) == (
            review_ids.QC_SUBVIEW_REVIEW
        )
        configure_style = {"display": "none" if review else "block"}
        # Plain block, no height cap: the Review view sizes to its content
        # so the gallery flows down and the ``qc-tab-root`` wrapper (which
        # has ``overflow: auto``) scrolls the whole page. The Review view's
        # sticky header + sidebar keep the nav pinned during that scroll.
        review_style = {"display": "block" if review else "none"}
        return (
            configure_style,
            review_style,
            review_ids.QC_SUBVIEW_REVIEW if review else review_ids.QC_SUBVIEW_CONFIGURE,
        )

    # Review sub-view owns its own callback bundle (worklist, detail,
    # curation, recompute, review-state). Registered here so the QC tab's
    # single ``register_qc_callbacks`` entry wires both sub-views.
    register_review_callbacks(app)


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------


def _export_qc_report(
    *,
    recipe: QcRecipe,
    filtered: Any,
    output_root: Any,
) -> tuple[Path, Path]:
    """Write ``qc.parquet`` and ``qc_summary.json`` into ``deliverables/qc/``.

    Resolves the target via ``output_root.layout.qc_dir`` so a standalone
    deliverables bundle writes inside the bundle (review C4).

    Args:
        recipe: The active QC recipe.
        filtered: The :class:`FilteredMeasurements` curation state.
        output_root: The :class:`OutputRoot` exposing ``master_df`` and
            ``layout``.

    Returns:
        ``(parquet_path, summary_path)`` — absolute file paths written
        to disk.
    """
    # Route through the resolved qc dir so a standalone deliverables bundle
    # (root == deliverables folder) writes inside the bundle, never a raw
    # ``output_root.root`` join (review C4).
    qc_dir = output_root.layout.qc_dir
    qc_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = qc_dir / "qc.parquet"
    summary_path = qc_dir / "qc_summary.json"

    pandas_frame = get_curated_frame(filtered, output_root).to_pandas()
    instances = dict(recipe.instantiate())

    parts: list[pd.DataFrame] = []
    summary_entries: list[dict[str, Any]] = []
    for entry in recipe.entries:
        if not entry.enabled:
            continue
        check = instances.get(entry.instance_id)
        if check is None:
            continue
        try:
            result = check.analyze(pandas_frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "QC export: analyze failed for %s (%s): %s",
                entry.instance_id,
                entry.cls.__name__,
                exc,
            )
            continue
        out = result.copy()
        # Leading discriminator columns — spec lines 820-829.
        out.insert(0, "QC_Check_Class", entry.cls.__name__)
        out.insert(1, "QC_Check_Instance_Id", entry.instance_id)
        parts.append(out)

        summary_frame = check.summary()
        status_counts: dict[str, int] = {"pass": 0, "warn": 0, "fail": 0}
        if "qc_status" in summary_frame.columns:
            for s, cnt in summary_frame["qc_status"].value_counts().items():
                key = str(s)
                if key in status_counts:
                    status_counts[key] = int(cnt)
        summary_entries.append(
            {
                "instance_id": entry.instance_id,
                "class": entry.cls.__name__,
                "params": dict(entry.params),
                "num_rows": int(len(result)),
                "num_flagged": int(
                    summary_frame["qc_n_flagged"].fillna(0).astype(int).sum()
                )
                if "qc_n_flagged" in summary_frame.columns
                else 0,
                "max_severity": (
                    float(
                        pd.to_numeric(
                            summary_frame.get(
                                "qc_worst_metric", pd.Series(dtype=float)
                            ),
                            errors="coerce",
                        ).max()
                    )
                    if "qc_worst_metric" in summary_frame.columns
                    and not summary_frame.empty
                    else float("nan")
                ),
                "status_counts": status_counts,
            }
        )

    if parts:
        combined = pd.concat(parts, axis=0, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame(
            columns=["QC_Check_Class", "QC_Check_Instance_Id"]
        )

    # Write parquet via polars to keep the writer footprint consistent
    # with the rest of the viewer's IO surface.
    pl.from_pandas(combined).write_parquet(parquet_path)
    summary_path.write_text(
        json.dumps(summary_entries, indent=2, sort_keys=False),
        encoding="utf-8",
    )

    return parquet_path, summary_path


# Re-exports for test access.
__all__ = [
    "register_qc_callbacks",
    "_left_join_qc_columns",
    "_render_summary_strip",
    "_worst_status",
    "_badge_color_for_status",
    "_merge_removed_keys",
    "_gather_modal_raw_values",
    "_empty_figure",
    "_error_figure",
]
