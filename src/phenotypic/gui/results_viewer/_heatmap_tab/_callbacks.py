"""Callbacks owned by the Heatmap tab.

Two callbacks:

1. :func:`_render_heatmap` - rebuilds the figure on any picker change
   or store-revision tick. Reads the QC-augmented frame when
   available so the color picker can target QC metric columns;
   falls back to the plain filtered frame otherwise (spec lines
   1074-1090).
2. :func:`_refresh_heatmap_controls` - repopulates the picker /
   slider options when the recipe revision bumps or the user curates.

The ordering edge that prevents the heatmap from reading a stale
QC-augmented frame is enforced by subscribing to
``STORE_QC_AUGMENTED_REVISION`` (bumped by the Wave E QC tab callback
*after* it has finished writing ``CFG_QC_AUGMENTED_FRAME``). See spec
lines 775-798.
"""
from __future__ import annotations

import logging
import math
from typing import Any, cast

import dash
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from dash import Input, Output, State, ctx, no_update
from flask import current_app

from phenotypic.gui._config import (
    CFG_FILTERED_STATE,
    CFG_MEASUREMENT_SCHEMA,
    CFG_OUTPUT_ROOT,
    CFG_QC_AUGMENTED_FRAME,
)
from phenotypic.sdk_.viz.figures import apply_theme

from phenotypic.gui.results_viewer import _ids as viewer_ids
from phenotypic.gui.results_viewer._filtered_state import get_curated_frame
from phenotypic.gui.results_viewer._heatmap_tab import _ids as ids
from phenotypic.gui.results_viewer._heatmap_tab._figure import (
    AggregatorName,
    build_heatmap_figure,
)
from phenotypic.gui.results_viewer._metadata import normalize_viewer_frame
from phenotypic.schema import CULTURE, IMAGE
from phenotypic.gui.results_viewer._picker_navigation import (
    picker_button_disabled_states,
    step_picker_value,
)

_TIME_COL: str = str(CULTURE.TIME)

logger = logging.getLogger(__name__)


# Style returned by the controls-refresh callback when the time slider
# should be visible / hidden. Pre-built so the callback doesn't
# allocate a dict on every fire.
_TIME_WRAPPER_VISIBLE: dict[str, str] = {"display": "block"}
_TIME_WRAPPER_HIDDEN: dict[str, str] = {"display": "none"}


def register_heatmap_callbacks(app: dash.Dash) -> None:
    """Register the Heatmap tab's two callbacks on *app*.

    Args:
        app: The Dash application that will own the callbacks. The
            tab body must have been mounted via
            :func:`._layout.build_heatmap_tab_body` before this is
            called so every Input / Output exists in the layout tree.
    """

    @app.callback(
        Output(ids.HEATMAP_FIGURE_ID, "figure"),
        Input(ids.HEATMAP_COLOR_PICKER_ID, "value"),
        Input(ids.HEATMAP_IMAGE_PICKER_ID, "value"),
        Input(ids.HEATMAP_TIME_SLIDER_ID, "value"),
        Input(ids.HEATMAP_AGGREGATOR_PICKER_ID, "value"),
        Input(ids.STORE_QC_AUGMENTED_REVISION, "data"),
        Input(viewer_ids.STORE_REMOVED_KEYS, "data"),
    )
    def _render_heatmap(  # noqa: PLR0913 - signature mirrors the Input list
        color: str | None,
        image: str | None,
        time_value: float | int | None,
        aggregator: str | None,
        augmented_revision: int | None,  # noqa: ARG001 - subscribed for ordering
        removed_keys: list[Any] | None,
    ) -> go.Figure:
        """Render the heatmap from the current picker state.

        Args:
            color: Selected color column. ``None`` short-circuits to an
                empty-state figure.
            image: Selected ``Metadata_ImageName`` value. ``None``
                short-circuits to an empty-state figure.
            time_value: Slider value or ``None``. Compared as a float;
                see :func:`._figure.build_heatmap_figure` for the
                coercion semantics.
            aggregator: Polars ``GroupBy.agg`` aggregator literal.
                Falls back to ``"mean"`` when ``None`` (e.g. on first
                load before the dropdown emits a value).
            augmented_revision: Tick from the QC tab's writer; only
                subscribed for ordering.
            removed_keys: Curated-removal payload from
                ``STORE_REMOVED_KEYS``.

        Returns:
            A :class:`plotly.graph_objects.Figure`.
        """
        if not color or not image:
            return _empty_state_figure(
                "Select a color column and an image to render the heatmap."
            )
        frame = _resolve_frame()
        if frame is None:
            return _empty_state_figure(
                "No measurements available. Run a pipeline first."
            )
        return build_heatmap_figure(
            frame=frame,
            color_col=color,
            image_file=image,
            time_value=time_value,
            aggregator=cast(AggregatorName, aggregator or "mean"),
            removed_keys=_as_key_set(removed_keys or []),
        )

    @app.callback(
        Output(ids.HEATMAP_COLOR_PICKER_ID, "options"),
        Output(ids.HEATMAP_IMAGE_PICKER_ID, "options"),
        Output(ids.HEATMAP_TIME_SLIDER_ID, "marks"),
        Output(ids.HEATMAP_TIME_SLIDER_ID, "min"),
        Output(ids.HEATMAP_TIME_SLIDER_ID, "max"),
        Output(ids.HEATMAP_TIME_SLIDER_ID, "value"),
        Output(ids.HEATMAP_TIME_SLIDER_WRAPPER_ID, "style"),
        Output(ids.HEATMAP_TIME_NON_NUMERIC_CAPTION_ID, "children"),
        Input(viewer_ids.STORE_QC_RECIPE_REVISION, "data"),
        Input(viewer_ids.STORE_QC_AUGMENTED_REVISION, "data"),
        Input(viewer_ids.STORE_REMOVED_KEYS, "data"),
    )
    def _refresh_heatmap_controls(
        _recipe_revision: int | None,
        _augmented_revision: int | None,
        _removed_keys: list[Any] | None,
    ) -> tuple[
        list[dict[str, str]],
        list[dict[str, str]],
        dict[float, str] | dict[str, str],
        float,
        float,
        float | None,
        dict[str, str],
        str,
    ]:
        """Repopulate dropdown options and the time slider on each tick.

        Reads the augmented frame when present so the color picker can
        include any ``QC_*_Metric`` columns the QC tab emitted; falls
        back to the plain filtered frame + schema otherwise.

        Returns:
            Tuple of:

            * Color column options.
            * Image picker options.
            * Time slider marks dict.
            * Time slider ``min``.
            * Time slider ``max``.
            * Time slider ``value``.
            * Wrapper style toggling slider visibility.
            * Non-numeric caption text.
        """
        frame = _resolve_frame()
        schema = current_app.config.get(CFG_MEASUREMENT_SCHEMA)

        # Color options: schema columns union QC_*_Metric columns in
        # the active frame. Schema may be ``None`` when the analysis
        # sub-app hasn't seeded it (e.g. the viewer is mounted on a
        # freshly-discovered output root without QC checks); we
        # degrade to the frame's own column list.
        column_names: list[str] = []
        if schema is not None:
            try:
                column_names = list(schema.columns_for("measurements"))
            except Exception:  # noqa: BLE001 - defensive
                logger.warning("Schema lookup failed", exc_info=True)
                column_names = []
        if frame is not None:
            for c in frame.columns:
                if c.startswith("QC_") and c.endswith("_Metric") and c not in column_names:
                    column_names.append(c)
            if not column_names:
                column_names = list(frame.columns)
        color_options = [{"label": c, "value": c} for c in column_names]

        # Image options: unique ``Metadata_ImageName`` in the active
        # frame, or empty when the frame is missing.
        image_options: list[dict[str, str]] = []
        if frame is not None and str(IMAGE.IMAGE_NAME) in frame.columns:
            unique = sorted(
                v for v in frame[str(IMAGE.IMAGE_NAME)].unique().to_list() if v is not None
            )
            image_options = [{"label": str(f), "value": str(f)} for f in unique]

        # Time slider state.
        marks, t_min, t_max, t_value, wrapper_style, caption = _build_time_slider_state(frame)
        return (
            color_options,
            image_options,
            marks,
            t_min,
            t_max,
            t_value,
            wrapper_style,
            caption,
        )

    @app.callback(
        Output(ids.HEATMAP_IMAGE_PICKER_ID, "value", allow_duplicate=True),
        Input(ids.HEATMAP_IMAGE_PREV_ID, "n_clicks"),
        Input(ids.HEATMAP_IMAGE_NEXT_ID, "n_clicks"),
        State(ids.HEATMAP_IMAGE_PICKER_ID, "value"),
        State(ids.HEATMAP_IMAGE_PICKER_ID, "options"),
        prevent_initial_call=True,
    )
    def _step_heatmap_image(
        _prev_clicks: int | None,
        _next_clicks: int | None,
        current: str | None,
        options: list[dict[str, Any]] | None,
    ) -> str | Any:
        """Step the Heatmap image picker from the icon-only buttons."""
        triggered = ctx.triggered_id
        if triggered == ids.HEATMAP_IMAGE_PREV_ID:
            return step_picker_value(current, options, "previous") or no_update
        if triggered == ids.HEATMAP_IMAGE_NEXT_ID:
            return step_picker_value(current, options, "next") or no_update
        return no_update

    @app.callback(
        Output(ids.HEATMAP_IMAGE_PREV_ID, "disabled"),
        Output(ids.HEATMAP_IMAGE_NEXT_ID, "disabled"),
        Input(ids.HEATMAP_IMAGE_PICKER_ID, "value"),
        Input(ids.HEATMAP_IMAGE_PICKER_ID, "options"),
    )
    def _sync_heatmap_image_nav_disabled(
        current: str | None,
        options: list[dict[str, Any]] | None,
    ) -> tuple[bool, bool]:
        """Disable Heatmap image navigation buttons at picker bounds."""
        return picker_button_disabled_states(current, options)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_frame() -> pl.DataFrame | None:
    """Pick the QC-augmented frame if present, else the filtered frame.

    Returns ``None`` only when neither path is available, which in
    practice means the viewer was constructed without an output root
    (the empty-state path). Production callbacks should never see this
    case because the layout itself is not mounted in empty-state mode.
    """
    augmented_obj = current_app.config.get(CFG_QC_AUGMENTED_FRAME)
    augmented = _coerce_polars(augmented_obj) if augmented_obj is not None else None
    if augmented is not None:
        return augmented

    filtered = current_app.config.get(CFG_FILTERED_STATE)
    output_root = current_app.config.get(CFG_OUTPUT_ROOT)
    if filtered is None or output_root is None:
        return None
    try:
        return get_curated_frame(filtered, output_root)
    except ValueError:
        # Metadata conflicts are actionable compatibility blockers, not an
        # empty-state condition.
        raise
    except Exception:  # noqa: BLE001 - defensive: stale config keys.
        logger.warning("filtered_df lookup failed", exc_info=True)
        return None


def _coerce_polars(obj: object) -> pl.DataFrame | None:
    """Coerce a config-stashed frame into polars, or None if not a frame.

    Wave E may stash either a polars or pandas frame depending on the
    QC writer's implementation choice. Keeping the coercion local
    means the figure callback doesn't depend on which it ended up
    being.
    """
    if isinstance(obj, pl.DataFrame):
        return normalize_viewer_frame(obj)
    if isinstance(obj, pd.DataFrame):
        return normalize_viewer_frame(pl.from_pandas(obj))
    return None


def _empty_state_figure(message: str) -> go.Figure:
    """Local empty-state figure used by :func:`_render_heatmap`.

    Mirrors the shape of the figure builder's own empty-state path so
    consumers see consistent chrome regardless of which layer rejected
    the request.
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
    )
    apply_theme(fig)
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return fig


def _as_key_set(payload: list[Any]) -> set[tuple[str, int]]:
    """Coerce a ``STORE_REMOVED_KEYS`` payload into a typed key set.

    Mirrors :func:`phenotypic.gui.results_viewer._filtered_state
    .decode_removed_keys_payload` semantics but returns a hash-set
    rather than a list since the figure builder treats curated keys
    as a membership lookup.
    """
    out: set[tuple[str, int]] = set()
    for entry in payload:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        try:
            out.add((str(entry[0]), int(entry[1])))
        except (TypeError, ValueError):
            continue
    return out


def _build_time_slider_state(
    frame: pl.DataFrame | None,
) -> tuple[
    dict[float, str] | dict[str, str],
    float,
    float,
    float | None,
    dict[str, str],
    str,
]:
    """Compute time slider marks, range, value, visibility, caption.

    Time slider visibility rules (spec lines 1021-1028):

    * Hidden when ``Metadata_Time`` is absent.
    * Hidden when there's only one time point.
    * Hidden when coercion to numeric is all-NaN (e.g. labels like
      ``"baseline"`` exclusively).
    * Otherwise visible with marks at every unique numeric value.
    * Partial-NaN coercion (some numeric, some not) shows the slider
      plus a "skipping N non-numeric values" caption.
    """
    empty: tuple[
        dict[float, str] | dict[str, str], float, float, float | None, dict[str, str], str
    ] = ({}, 0.0, 1.0, None, _TIME_WRAPPER_HIDDEN, "")
    if frame is None or _TIME_COL not in frame.columns:
        return empty

    raw = frame[_TIME_COL].to_list()
    coerced = pd.to_numeric(pd.Series(raw), errors="coerce")
    numeric = coerced.dropna()
    if numeric.empty:
        return empty

    unique_values = sorted(set(numeric.tolist()))
    if len(unique_values) <= 1:
        return empty

    nan_count = int(coerced.isna().sum())
    caption = (
        f"Skipping {nan_count} non-numeric Metadata_Time values."
        if nan_count > 0
        else ""
    )

    # Marks: one mark per unique numeric value. Plotly's dcc.Slider
    # marks dict keys are the slider positions; values are the
    # rendered labels. We render labels via ``str()`` so integer
    # timepoints look clean and floats keep their representation.
    marks: dict[float, str] = {}
    for v in unique_values:
        # Plotly's slider rejects non-finite marks; skip them defensively.
        if not math.isfinite(v):
            continue
        marks[v] = _format_time_label(v)

    if not marks:
        return empty

    t_min = float(unique_values[0])
    t_max = float(unique_values[-1])
    t_value = float(unique_values[0])
    return marks, t_min, t_max, t_value, _TIME_WRAPPER_VISIBLE, caption


def _format_time_label(value: float) -> str:
    """Render an integer-valued float without the trailing ``.0``.

    Plotly slider marks render the dict value verbatim; ``"4.0"`` is
    visually noisy when the source data was ``int``-typed. The rest of
    the codebase coerces to float for the comparison only, so this
    helper exists purely for cosmetic mark labels.
    """
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


__all__ = ["register_heatmap_callbacks"]
