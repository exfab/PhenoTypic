"""Dash wiring for the Scatter tab.

Every callback body here is a thin wrapper over a module-level pure
function. That is not tidiness: Dash wraps a registered callback in its
own context manager, so a test that reaches into ``app.callback_map`` to
call one back gets the wrapper rather than the logic. Keeping the logic
outside means the ordering rules below are asserted against the code that
actually runs, not against a re-implementation of it in a test.

Four properties of this module are correctness requirements, and each of
them is invisible when broken:

* **The click index is stamped on ``master_df``, before any filtering.**
  :func:`~.._inspector.index_frame` is the only sanctioned producer, and
  a call site reaching for ``with_row_index`` itself has reintroduced the
  defect Gate 0 found: every click opens a real but *wrong* colony, with
  a real crop and nothing raising.
* **The click fingerprint compared at click time is the one stored when
  the figure was drawn.** Only the *expected* side is read live. Reading
  ``output_root.consumed_state_fingerprint`` on both sides leaves a guard
  that still exists, still reads correctly, and can never fire.
* **Curation is joined after indexing and before the phantom filter.**
  Indexing first keeps every carried index master-anchored; filtering
  after the join means a phantom is dropped by ``plottable`` rather than
  reaching the figure.
* **The filter store is an Input, not a State.** Decision Q4 shares the
  viewer's filters with this tab, so a filter edit must rebuild the
  figure. As a State it would leave the figure showing rows the sidebar
  says are excluded, with no error and no visible staleness.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

import dash
import plotly.graph_objects as go
import polars as pl
from dash import Input, Output, State, ctx, dcc, html, no_update
from dash.development.base_component import Component
from flask import current_app

from phenotypic.gui._config import (
    CFG_URL_PREFIX,
    MOUNT_HOME,
    SCATTER_CROPS_URL_SEGMENT,
)
from phenotypic.gui._design import (
    COLOR_BORDER,
    COLOR_MUTED,
    COLOR_NAVY,
    FONT_FAMILY_MONO,
    FONT_SIZE_CAPTION,
)
from phenotypic.gui.results_viewer import _ids as rv_ids
from phenotypic.gui.results_viewer._filter_state import FilterSpec
from phenotypic.gui.results_viewer._filtered_state import (
    KEY_DATASET,
    KEY_IMAGE_FILE,
    KEY_OBJECT_LABEL,
    decode_removed_keys_payload,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer._scatter_tab import _ids as ids
from phenotypic.gui.results_viewer._scatter_tab._facets import (
    COMPUTED_FRAME_INDEX,
    FacetPlan,
    derive_frame_index,
    plan_facets,
    sort_facet_values,
)
from phenotypic.gui.results_viewer._scatter_tab._figure import (
    REMOVED_COL,
    build_scatter_figure,
)
from phenotypic.gui.results_viewer._scatter_tab._grouping import group_columns
from phenotypic.gui.results_viewer._scatter_tab._inspector import (
    ColonyRef,
    index_frame,
    resolve_click,
)
from phenotypic.gui.results_viewer._scatter_tab._layout import (
    LEGEND_CORNER_DEFAULT,
)
from phenotypic.gui.results_viewer._scatter_tab._pdf import export_sections_pdf
from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec, plottable

logger = logging.getLogger(__name__)

#: Side length of the inspector's crop, in px. Spec section 7 costs the
#: server-side composite against two client-side Viv layers at exactly
#: this size -- 3 inner chunks and ~4 MB against a ~100 KB PNG -- so it
#: is the size that argument was made at, not a number carried in from
#: the colony grid's tile stepper.
INSPECTOR_CROP_SIZE = 256

#: Plotly legend placement per corner token. Inset rather than flush at
#: 0/1 so the legend does not sit on the axis line.
_LEGEND_ANCHORS: dict[str, dict[str, Any]] = {
    "top-left": {"x": 0.01, "y": 0.99, "xanchor": "left", "yanchor": "top"},
    "top-right": {"x": 0.99, "y": 0.99, "xanchor": "right", "yanchor": "top"},
    "bottom-left": {"x": 0.01, "y": 0.01, "xanchor": "left", "yanchor": "bottom"},
    "bottom-right": {
        "x": 0.99,
        "y": 0.01,
        "xanchor": "right",
        "yanchor": "bottom",
    },
}

#: Scratch columns the curation join needs so it can compare on one
#: dtype. Underscore-prefixed like ``CUSTOMDATA_COL`` and dropped again
#: before the frame is returned, so nothing downstream sees them.
_JOIN_IMAGE = "_scatter_join_image"
_JOIN_LABEL = "_scatter_join_label"

#: What the inspector says when it cannot open a colony. A refusal must
#: be visible: the alternative to saying so is a panel that quietly does
#: nothing, which reads as a broken click rather than a stale one.
_STALE_CLICK_MESSAGE = (
    "This point was drawn before the run changed. Refresh the snapshot "
    "and click again."
)
_UNRESOLVED_CLICK_MESSAGE = "That point does not name a colony."


# ---------------------------------------------------------------------------
# Curation join -- ``show_removed``'s producer half (plan section 13.1)
# ---------------------------------------------------------------------------


def join_curation_flags(df: pl.DataFrame, removed_payload: object) -> pl.DataFrame:
    """Add the Boolean ``REMOVED_COL`` from the curation store.

    Curation lives only in a Dash store and in ``CurationLabels``; it is
    never a column of the measurements mirror. The figure builder reads
    it as an optional Boolean column, so this is where that column comes
    into being -- in memory, per render.

    **Boolean, and null-free.** ``_figure._split_on_curation`` treats a
    non-Boolean column as *absent* rather than coercing it, so a wrong
    dtype here would silently stop drawing the grey series instead of
    raising. Every row therefore gets a real ``True``/``False``.

    The join compares on cast scratch columns rather than on the frame's
    own dtypes: ``Object_Label`` is ``Int64`` in the mirror but narrower
    in some frames, and ``Metadata_ImageName`` can be ``Categorical``, so
    a direct join is a dtype mismatch waiting to happen. The label cast
    carries a stated precondition -- see the comment at the call site,
    and ``test_a_categorical_label_is_out_of_scope_and_here_is_why`` for
    the trap it declines to guard. A null label -- every phantom has one
    -- casts to null and matches nothing, which is the right answer: a
    phantom is not a curatable object.

    The key frame is de-duplicated before the join. A store holding the
    same key twice would otherwise duplicate the row it matches, drawing
    one colony as two points.

    Args:
        df: The indexed frame, before filtering.
        removed_payload: Raw ``STORE_REMOVED_KEYS`` data.

    Returns:
        ``df`` with a Boolean ``REMOVED_COL``, or ``df`` unchanged when
        the frame cannot carry curation keys at all -- absence is how
        the figure builder is told "nothing is removed".
    """
    if KEY_IMAGE_FILE not in df.columns or KEY_OBJECT_LABEL not in df.columns:
        logger.debug(
            "scatter: no curation keys in the frame; skipping the join"
        )
        return df

    keys = decode_removed_keys_payload(removed_payload)
    if not keys:
        return df.with_columns(pl.lit(False).alias(REMOVED_COL))

    removed = pl.DataFrame(
        {
            _JOIN_IMAGE: [image for image, _ in keys],
            _JOIN_LABEL: [label for _, label in keys],
        },
        schema={_JOIN_IMAGE: pl.String, _JOIN_LABEL: pl.Int64},
    ).unique(subset=[_JOIN_IMAGE, _JOIN_LABEL])
    removed = removed.with_columns(pl.lit(True).alias(REMOVED_COL))

    return (
        df.with_columns(
            pl.col(KEY_IMAGE_FILE).cast(pl.String).alias(_JOIN_IMAGE),
            # PRECONDITION: Object_Label is an integer-valued dtype, never
            # Categorical AND NEVER ENUM. Both cast to their physical
            # CATEGORY CODES rather than their labels -- ["1", "2"]
            # becomes [0, 1] for each -- so every colony would join
            # against the wrong curation row, with no exception and no
            # null. Enum is named explicitly because it was measured, not
            # assumed: it is the half a later reader would drop as an
            # unused case, and it fails identically. Three things say the
            # state is not reachable, and they are not equally strong:
            #   * GUARANTEED on the key side: CurationLabels writes the
            #     store with a declared schema pinning this column to
            #     Int64 (`_curation_labels.py:829-836`).
            #   * EMPIRICAL on the master side: the verification run
            #     carries Object_Label as Int64 and not one Categorical
            #     column in 149. No counterexample, and no mechanism --
            #     which is weaker than the schema above, and is why this
            #     is written down rather than assumed.
            #   * PRECEDENT: `_curation_labels.py:638` already casts this
            #     same column on this same frame, unguarded and without
            #     even `strict=False`. This join inherits an assumption
            #     the codebase already makes; guarding it here alone would
            #     diverge from that line for no reason.
            # `strict=False` is still right: it makes an unreadable label
            # null, which matches nothing, rather than raising.
            pl.col(KEY_OBJECT_LABEL)
            .cast(pl.Int64, strict=False)
            .alias(_JOIN_LABEL),
        )
        .join(removed, on=[_JOIN_IMAGE, _JOIN_LABEL], how="left")
        .with_columns(pl.col(REMOVED_COL).fill_null(False))
        .drop([_JOIN_IMAGE, _JOIN_LABEL])
    )


# ---------------------------------------------------------------------------
# The render pipeline, shared by the figure and the export
# ---------------------------------------------------------------------------


def prepare_frame(
    output_root: OutputRoot,
    *,
    x_col: str | None,
    y_col: str | None,
    filter_payload: object,
    removed_payload: object,
) -> tuple[pl.DataFrame, int]:
    """Build the plottable frame for one render, in the one legal order.

    Index, then join curation, then filter, then drop phantoms, then
    derive the capture-order index if it is what X selects, then drop
    rows with no coordinates. Each step's position is load-bearing:

    * indexing first is what anchors every carried index to
      ``master_df`` rather than to a filtered slice (see the module
      docstring);
    * joining curation before ``plottable`` means a phantom is removed by
      the phantom filter rather than reaching the figure;
    * deriving the frame index after filtering ranks the timestamps the
      user can actually see, so a filtered-out image does not leave a
      gap in the capture order.

    Args:
        output_root: The bound run.
        x_col: Column plotted on X, or ``COMPUTED_FRAME_INDEX``.
        y_col: Column plotted on Y.
        filter_payload: Raw ``STORE_FILTER_SPEC`` data.
        removed_payload: Raw ``STORE_REMOVED_KEYS`` data.

    Returns:
        ``(frame, dropped)`` -- the plottable frame, and how many rows
        were dropped for having no X or Y value. The count is surfaced
        rather than swallowed: spec section 10 requires the excluded rows
        be reported, not silently omitted.
    """
    base = index_frame(output_root.master_df)
    base = join_curation_flags(base, removed_payload)
    spec = FilterSpec.from_store(_as_filter_payload(filter_payload))
    frame = plottable(spec.apply_to(base))
    if x_col == COMPUTED_FRAME_INDEX:
        frame = derive_frame_index(frame)
    coordinate_columns = [
        column
        for column in (x_col, y_col)
        if column and column in frame.columns
    ]
    if not coordinate_columns:
        return frame, 0
    before = frame.height
    frame = frame.drop_nulls(subset=coordinate_columns)
    return frame, before - frame.height


def _as_filter_payload(payload: object) -> list[dict] | None:
    """Narrow a Dash store's ``Any`` to what ``FilterSpec`` accepts."""
    return cast("list[dict] | None", payload if isinstance(payload, list) else None)


def section_values(frame: pl.DataFrame, section_col: str | None) -> list[str]:
    """Ordered section values, or ``[]`` when nothing groups the frame.

    Nulls are dropped rather than becoming a ``"(none)"`` page -- spec
    section 10's rule for every grouping role.

    Args:
        frame: The plottable frame.
        section_col: The section-group column, or ``None``.

    Returns:
        Section values in page order. Empty means one undivided section.
    """
    if not section_col or section_col not in frame.columns:
        return []
    return sort_facet_values(
        frame[section_col].drop_nulls().unique().cast(pl.String).to_list()
    )


def store_int(value: object) -> int | None:
    """Read a Dash store payload as a whole number, or refuse it.

    ``bool`` is rejected first, and that ordering is the point rather
    than a formality: ``bool`` is an ``int`` subclass and ``int(True)``
    is ``1``, so a truthy flag arriving where a count is expected would
    silently become "section 1" or "1 px wide". This is the trap
    :func:`~.._inspector._row_position` guards with ``SupportsIndex``,
    reached from a different direction -- there it would open the wrong
    colony, here it is visible, and in both places a whitelist of one
    type is not what does the work.

    The two checks are separable on purpose. Drop the ``bool`` branch and
    ``True`` becomes ``1``; drop the type branch and ``None`` raises
    ``TypeError`` where it should return ``None``. Neither shadows the
    other, so each is pinned by a test that fails when it alone is
    removed.

    ``SupportsIndex`` is deliberately NOT reused here. A width really can
    arrive as a float -- it comes from ``getBoundingClientRect()`` -- and
    a float implements no ``__index__``, so the inspector's width would
    silently reset on every re-render.

    Args:
        value: Whatever the Dash store returned.

    Returns:
        The value as an ``int``, or None when it cannot be one.
    """
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (ValueError, OverflowError):
        # OverflowError is what `int(float("inf"))` raises, where NaN
        # raises ValueError -- an asymmetry in `int`, not a distinction
        # this function wants to make. Both mean "not a whole number".
        return None


def clamp_section_index(index: object, count: int) -> int:
    """Clamp a stored section index into range.

    A live run can retire the section the pager was on, so the index is
    clamped rather than trusted. An unreadable payload reads as 0.

    Args:
        index: Raw ``STORE_SCATTER_SECTION_INDEX`` data.
        count: How many sections the current frame has.

    Returns:
        A position in ``[0, max(count - 1, 0)]``.
    """
    position = store_int(index)
    if position is None:
        position = 0
    return max(0, min(position, max(count - 1, 0)))


def paged_section_index(triggered_id: object, index: object, count: int) -> int:
    """Step the section index one page, in the direction that was clicked.

    Takes the raw ``ctx.triggered_id`` rather than a ``forward`` flag on
    purpose. The risky part of a pager is not the arithmetic, it is which
    button means which way -- a swapped pair walks backwards from a
    button labelled "next" and nothing raises. Passing a bool would move
    that mapping back into the callback closure, where no test can reach
    it, and leave this function testing the half that was never in doubt.

    Args:
        triggered_id: ``ctx.triggered_id`` from the pager callback.
        index: The current section index, from its store.
        count: How many sections the current frame has.

    Returns:
        The new index, clamped into range at both ends.
    """
    step = -1 if triggered_id == ids.SCATTER_PREV_BTN else 1
    return clamp_section_index(clamp_section_index(index, count) + step, count)


def legend_layout(payload: object) -> dict[str, Any]:
    """Translate the legend store into a Plotly ``legend`` update.

    Args:
        payload: Raw ``STORE_SCATTER_LEGEND`` data.

    Returns:
        Keyword arguments for ``fig.update_layout``. A collapsed legend
        hides the legend itself rather than emptying it, so the traces
        keep their names for hover.
    """
    corner = LEGEND_CORNER_DEFAULT
    collapsed = False
    if isinstance(payload, dict):
        corner = str(payload.get("corner") or LEGEND_CORNER_DEFAULT)
        collapsed = bool(payload.get("collapsed"))
    anchors = _LEGEND_ANCHORS.get(corner, _LEGEND_ANCHORS[LEGEND_CORNER_DEFAULT])
    return {"showlegend": not collapsed, "legend": dict(anchors)}


def _empty_figure(message: str) -> go.Figure:
    """A figure that says why it is empty, rather than an empty figure."""
    figure = go.Figure()
    figure.add_annotation(
        text=message, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5
    )
    figure.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return figure


def _figure_spec(
    *,
    x_col: str,
    y_col: str,
    section_col: str | None,
    row_col: str | None,
    col_col: str | None,
    hue_col: str | None,
    shape_col: str | None,
    show_removed: object,
) -> FigureSpec:
    """Bind the role controls into the spec both destinations share.

    One constructor for the screen and the export, so a role cannot be
    carried on one path and dropped on the other -- the same reason
    :func:`prepare_frame` is shared. ``show_removed`` arrives from a Dash
    control as an untyped payload and is coerced here, once.

    Args:
        x_col: Column plotted on X, or ``COMPUTED_FRAME_INDEX``.
        y_col: Column plotted on Y.
        section_col: Column whose values become sections.
        row_col: Column whose values become facet rows.
        col_col: Column whose values become facet columns.
        hue_col: Column mapped to marker colour.
        shape_col: Column mapped to marker symbol.
        show_removed: The curation toggle's value.

    Returns:
        The :class:`FigureSpec` for this render.
    """
    return FigureSpec(
        x_col=x_col,
        y_col=y_col,
        section_col=section_col,
        row_col=row_col,
        col_col=col_col,
        hue_col=hue_col,
        shape_col=shape_col,
        show_removed=bool(show_removed),
    )


def build_render_state(
    output_root: OutputRoot,
    *,
    section_col: str | None,
    row_col: str | None,
    col_col: str | None,
    x_col: str | None,
    y_col: str | None,
    hue_col: str | None,
    shape_col: str | None,
    show_removed: object,
    section_index: object,
    filter_payload: object,
    removed_payload: object,
    legend_payload: object,
) -> tuple[go.Figure, str, str]:
    """Build one section's figure, its pager label and its fingerprint.

    The returned fingerprint is the value the click callback will later
    compare against a freshly-read one. It is captured **here**, when the
    figure is drawn, which is the only thing that makes that comparison
    able to fail.

    Args:
        output_root: The bound run.
        section_col: Column whose values become sections.
        row_col: Column whose values become facet rows.
        col_col: Column whose values become facet columns.
        x_col: Column plotted on X, or ``COMPUTED_FRAME_INDEX``.
        y_col: Column plotted on Y.
        hue_col: Column mapped to marker colour.
        shape_col: Column mapped to marker symbol.
        show_removed: The curation toggle's value.
        section_index: Raw section-index store data.
        filter_payload: Raw filter-spec store data.
        removed_payload: Raw removed-keys store data.
        legend_payload: Raw legend store data.

    Returns:
        ``(figure, pager_label, fingerprint)``.
    """
    fingerprint = output_root.consumed_state_fingerprint
    if not x_col or not y_col:
        return (
            _empty_figure("Choose an X and a Y column in Plot settings."),
            "",
            fingerprint,
        )

    frame, dropped = prepare_frame(
        output_root,
        x_col=x_col,
        y_col=y_col,
        filter_payload=filter_payload,
        removed_payload=removed_payload,
    )
    spec = _figure_spec(
        x_col=x_col,
        y_col=y_col,
        section_col=section_col,
        row_col=row_col,
        col_col=col_col,
        hue_col=hue_col,
        shape_col=shape_col,
        show_removed=show_removed,
    )

    sections = section_values(frame, section_col)
    position = clamp_section_index(section_index, len(sections))
    current = sections[position] if sections else ""
    page = (
        frame.filter(pl.col(section_col).cast(pl.String) == current)
        if section_col and sections
        else frame
    )

    plan = plan_facets(page, spec)
    figure = build_scatter_figure(page, spec, plan)
    figure.update_layout(**legend_layout(legend_payload))
    return figure, _pager_label(current, position, sections, plan, dropped), fingerprint


def _pager_label(
    current: str,
    position: int,
    sections: list[str],
    plan: FacetPlan,
    dropped: int,
) -> str:
    """Compose the pager chip's text.

    Both notices are recomputed per render rather than cached with the
    figure: a live run adds facet values and images over time, so a
    truncation notice or an excluded-row count held from an earlier
    render would describe a figure nobody is looking at.

    Args:
        current: The section value on screen.
        position: Its zero-based position in ``sections``.
        sections: Every section value, in page order.
        plan: The ``FacetPlan`` the figure was drawn from.
        dropped: Rows excluded for having no X or Y value.

    Returns:
        The chip text.
    """
    total = len(sections) or 1
    label = f"{current}  ({position + 1} / {total})"
    if plan.truncated:
        shown = len(plan.rows) * len(plan.cols)
        label += f" — showing first {shown} of {plan.total} facets"
    if dropped:
        label += f" — {dropped} rows excluded, no value to plot"
    return label


# ---------------------------------------------------------------------------
# Click resolution and the inspector
# ---------------------------------------------------------------------------


def click_index(click_data: object) -> object | None:
    """Pull the carried row index out of a Plotly click payload.

    Returns ``None`` for anything that is not a point carrying
    customdata -- a click on empty axis space, a payload shape Plotly
    changes under us -- so the caller can tell "no click" from "a click
    that cannot be resolved". The value itself is handed on **unchecked**:
    :func:`~.._inspector.resolve_click` owns deciding what may be a row
    index, and a second check here would be a guard no test could kill.

    Args:
        click_data: The graph's ``clickData`` property.

    Returns:
        The raw customdata value, or ``None``.
    """
    if not isinstance(click_data, dict):
        return None
    points = click_data.get("points")
    if not isinstance(points, list) or not points:
        return None
    first = points[0]
    if not isinstance(first, dict):
        return None
    custom = first.get("customdata")
    if isinstance(custom, (list, tuple)):
        return custom[0] if custom else None
    return custom


def resolve_inspector_click(
    output_root: OutputRoot, click_data: object, fingerprint: object
) -> ColonyRef | None:
    """Resolve a click into a colony, or refuse it.

    ``fingerprint`` is the value the figure callback stored when it drew
    the figure; the value compared against it is read live, here. That
    asymmetry is the whole guard -- reading the live fingerprint on both
    sides yields a comparison that always passes.

    Args:
        output_root: The bound run.
        click_data: The graph's ``clickData`` property.
        fingerprint: The fingerprint stored beside the drawn figure.

    Returns:
        The colony, or ``None`` when the click carries no index, the
        index is stale, or it lands on a row that names no colony.
    """
    index = click_index(click_data)
    if index is None:
        return None
    return resolve_click(
        output_root.master_df,
        index,  # type: ignore[arg-type]
        str(fingerprint),
        output_root.consumed_state_fingerprint,
    )


def inspector_payload(
    output_root: OutputRoot, click_data: object, fingerprint: object
) -> tuple[Any, Any, Any, Any]:
    """Decide what one click does to the inspector.

    Three outcomes, and the difference between the last two matters to a
    user: a click that is not on a point leaves the panel exactly as it
    was, a click whose figure predates a snapshot change says so, and a
    click that resolves opens the colony.

    Args:
        output_root: The bound run.
        click_data: The graph's ``clickData`` property.
        fingerprint: The fingerprint stored beside the drawn figure.

    Returns:
        ``(is_open, title, colony_store, measurement_children)``, each
        possibly :data:`dash.no_update`.
    """
    if click_index(click_data) is None:
        # Not a click on a point. Leaving the panel as it was beats
        # closing it: an errant click on the axis would otherwise discard
        # whatever the user was reading.
        return no_update, no_update, no_update, no_update

    colony = resolve_inspector_click(output_root, click_data, fingerprint)
    if colony is None:
        stale = str(fingerprint) != output_root.consumed_state_fingerprint
        message = _STALE_CLICK_MESSAGE if stale else _UNRESOLVED_CLICK_MESSAGE
        return True, message, None, []

    return (
        True,
        f"{colony.dataset} / {colony.stem} / label {colony.label}",
        {"dataset": colony.dataset, "stem": colony.stem, "label": colony.label},
        measurement_children(output_root, colony, _measurement_config(output_root)),
    )


def crop_url(prefix: str, colony: object, *, contours: object) -> str:
    """Build the inspector's crop ``<img>`` src.

    Points at the Scatter crop route mounted under
    :data:`SCATTER_CROPS_URL_SEGMENT`. ``contours`` rides the URL rather
    than being resolved server-side per click, so moving the
    Contours/Raw control re-requests the same colony instead of
    re-resolving a click that may by then be stale.

    Args:
        prefix: The app's mount-point prefix.
        colony: The ``STORE_SCATTER_COLONY`` payload.
        contours: The Contours/Raw control's value.

    Returns:
        A URL, or ``""`` when no colony is selected -- an ``<img>`` with
        an empty src renders nothing, which is what an empty inspector
        should show.
    """
    if not isinstance(colony, dict):
        return ""
    dataset = colony.get("dataset")
    stem = colony.get("stem")
    label = colony.get("label")
    if dataset is None or stem is None or label is None:
        return ""
    flag = 1 if contours else 0
    return (
        f"{prefix}{SCATTER_CROPS_URL_SEGMENT}/{dataset}/{stem}/"
        f"{label}.png?size={INSPECTOR_CROP_SIZE}&contours={flag}"
    )


def _format_value(value: object) -> str:
    """Render one measurement value for the inspector's rows."""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def measurement_children(
    output_root: OutputRoot, colony: ColonyRef, meas_cfg: dict[str, dict]
) -> list[Component]:
    """Build the inspector's grouped measurement rows for one colony.

    Grouping is delegated to :func:`~.._grouping.group_columns`, which
    resolves each column to the ``MeasureFeatures`` class that emitted it
    from the run's own recorded parameters. Null-valued columns are
    dropped: a phantom-adjacent column with nothing in it is noise in a
    panel whose job is to say what this colony measured.

    Args:
        output_root: The bound run.
        colony: The resolved colony.
        meas_cfg: The run's ``"meas"`` config block.

    Returns:
        One block per group, or a single "no measurements" line.
    """
    row = output_root.master_df.filter(
        (pl.col(KEY_DATASET).cast(pl.String) == colony.dataset)
        & (pl.col(KEY_IMAGE_FILE).cast(pl.String) == colony.stem)
        & (pl.col(KEY_OBJECT_LABEL).cast(pl.Int64, strict=False) == colony.label)
    ).head(1)
    if row.is_empty():
        return [html.Div("No measurements for this colony.")]

    values = row.row(0, named=True)
    # ``master_df`` carries no in-memory column of ours, so the only
    # filter needed is the null one: a column with nothing in it for this
    # colony is noise in a panel whose job is to say what it measured.
    columns = [
        column for column, value in values.items() if value is not None
    ]
    groups = group_columns(columns, meas_cfg)

    blocks: list[Component] = []
    for group in sorted(groups):
        rows = [
            html.Div(
                [
                    html.Span(
                        column,
                        style={"color": COLOR_MUTED, "marginRight": "0.5rem"},
                    ),
                    html.Span(
                        _format_value(values[column]),
                        style={"fontFamily": FONT_FAMILY_MONO},
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "fontSize": FONT_SIZE_CAPTION,
                },
            )
            for column in groups[group]
        ]
        blocks.append(
            html.Div(
                [
                    html.Div(
                        group,
                        style={
                            "color": COLOR_NAVY,
                            "fontWeight": 600,
                            "fontSize": FONT_SIZE_CAPTION,
                            "borderBottom": f"1px solid {COLOR_BORDER}",
                            "margin": "0.5rem 0 0.25rem",
                        },
                    ),
                    *rows,
                ]
            )
        )
    return blocks


def export_payload(
    output_root: OutputRoot,
    *,
    n_clicks: object,
    section_col: str | None,
    row_col: str | None,
    col_col: str | None,
    x_col: str | None,
    y_col: str | None,
    hue_col: str | None,
    shape_col: str | None,
    show_removed: object,
    filter_payload: object,
    removed_payload: object,
) -> tuple[Any, str]:
    """Render every section to a PDF, or say why it could not.

    The export consumes exactly the frame the screen does, through
    :func:`prepare_frame`, so a document cannot describe a different
    selection from the one the user is looking at.

    Args:
        output_root: The bound run.
        n_clicks: The export button's click count.
        section_col: Column whose values become pages.
        row_col: Column whose values become facet rows.
        col_col: Column whose values become facet columns.
        x_col: Column plotted on X.
        y_col: Column plotted on Y.
        hue_col: Column mapped to marker colour.
        shape_col: Column mapped to marker symbol.
        show_removed: The curation toggle's value.
        filter_payload: Raw filter-spec store data.
        removed_payload: Raw removed-keys store data.

    Returns:
        ``(download, status)``. ``download`` is :data:`dash.no_update`
        whenever no document was produced, and ``status`` carries the
        reason -- kaleido's missing-Chrome error is the one failure a
        user will actually hit, and a button that silently does nothing
        reads as a broken export rather than a missing prerequisite.
    """
    if not n_clicks:
        return no_update, ""
    if not x_col or not y_col:
        return no_update, "Choose an X and a Y column first."
    frame, _ = prepare_frame(
        output_root,
        x_col=x_col,
        y_col=y_col,
        filter_payload=filter_payload,
        removed_payload=removed_payload,
    )
    spec = _figure_spec(
        x_col=x_col,
        y_col=y_col,
        section_col=section_col,
        row_col=row_col,
        col_col=col_col,
        hue_col=hue_col,
        shape_col=shape_col,
        show_removed=show_removed,
    )
    try:
        pdf = export_sections_pdf(frame, spec, section_values(frame, section_col))
    except RuntimeError as exc:
        logger.warning("Scatter PDF export failed", exc_info=True)
        return no_update, str(exc)
    return dcc.send_bytes(lambda buffer: buffer.write(pdf), "scatter.pdf"), ""


def _measurement_config(output_root: OutputRoot) -> dict[str, dict]:
    """Read the run's ``"meas"`` block from its own pipeline config.

    Resolved through ``layout.resolved_pipeline_config_path`` rather than
    hand-joined (spec section 8). A run whose config is missing or
    unreadable still opens an inspector -- every column simply lands
    under ``Unattributed`` -- so this degrades rather than raises.

    Args:
        output_root: The bound run.

    Returns:
        The ``"meas"`` mapping, or ``{}``.
    """
    path = output_root.layout.resolved_pipeline_config_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        logger.debug("scatter: no readable pipeline config at %s", path)
        return {}
    meas = payload.get("meas") if isinstance(payload, dict) else None
    return cast("dict[str, dict]", meas) if isinstance(meas, dict) else {}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_callbacks(app: dash.Dash, output_root: OutputRoot) -> None:
    """Register every Scatter callback. Called once per app.

    Args:
        app: The Dash application owning the mounted Scatter body.
        output_root: Validated handle on the CLI output directory,
            captured by closure.
    """

    @app.callback(
        Output(ids.SCATTER_GRAPH, "figure"),
        Output(ids.SCATTER_PAGER_LABEL, "children"),
        Output(ids.STORE_SCATTER_FINGERPRINT, "data"),
        Input(ids.SCATTER_SECTION_COL, "value"),
        Input(ids.SCATTER_ROW_COL, "value"),
        Input(ids.SCATTER_COL_COL, "value"),
        Input(ids.SCATTER_X_COL, "value"),
        Input(ids.SCATTER_Y_COL, "value"),
        Input(ids.SCATTER_HUE_COL, "value"),
        Input(ids.SCATTER_SHAPE_COL, "value"),
        Input(ids.SCATTER_SHOW_REMOVED, "value"),
        Input(ids.STORE_SCATTER_SECTION_INDEX, "data"),
        Input(ids.STORE_SCATTER_LEGEND, "data"),
        # Inputs, NOT States. The refresh revision moves every surface
        # together (spec 16.4); the filter store is shared with the
        # sidebar (Q4) so a filter edit must rebuild rather than leave
        # the figure stale; and the curation store is what draws the
        # grey series at all.
        Input(rv_ids.STORE_PLOT_REFRESH_REVISION, "data"),
        Input(rv_ids.STORE_FILTER_SPEC, "data"),
        Input(rv_ids.STORE_REMOVED_KEYS, "data"),
    )
    def _render(  # noqa: PLR0913 - signature mirrors the Input list
        section_col: str | None,
        row_col: str | None,
        col_col: str | None,
        x_col: str | None,
        y_col: str | None,
        hue_col: str | None,
        shape_col: str | None,
        show_removed: object,
        section_index: object,
        legend_payload: object,
        _revision: object,
        filter_payload: object,
        removed_payload: object,
    ) -> tuple[go.Figure, str, str]:
        return build_render_state(
            output_root,
            section_col=section_col,
            row_col=row_col,
            col_col=col_col,
            x_col=x_col,
            y_col=y_col,
            hue_col=hue_col,
            shape_col=shape_col,
            show_removed=show_removed,
            section_index=section_index,
            filter_payload=filter_payload,
            removed_payload=removed_payload,
            legend_payload=legend_payload,
        )

    @app.callback(
        Output(ids.STORE_SCATTER_SECTION_INDEX, "data"),
        Input(ids.SCATTER_PREV_BTN, "n_clicks"),
        Input(ids.SCATTER_NEXT_BTN, "n_clicks"),
        State(ids.STORE_SCATTER_SECTION_INDEX, "data"),
        State(ids.SCATTER_SECTION_COL, "value"),
        State(ids.SCATTER_X_COL, "value"),
        State(ids.SCATTER_Y_COL, "value"),
        State(rv_ids.STORE_FILTER_SPEC, "data"),
        State(rv_ids.STORE_REMOVED_KEYS, "data"),
        prevent_initial_call=True,
    )
    def _page(  # noqa: PLR0913 - signature mirrors the Input/State list
        _prev: object,
        _next: object,
        section_index: object,
        section_col: str | None,
        x_col: str | None,
        y_col: str | None,
        filter_payload: object,
        removed_payload: object,
    ) -> int:
        # The section list is re-derived rather than remembered: a live
        # run adds sections and the shared filter removes them, so a
        # count cached with the figure would clamp against a list that
        # no longer exists.
        frame, _ = prepare_frame(
            output_root,
            x_col=x_col,
            y_col=y_col,
            filter_payload=filter_payload,
            removed_payload=removed_payload,
        )
        sections = section_values(frame, section_col)
        return paged_section_index(
            ctx.triggered_id, section_index, len(sections)
        )

    @app.callback(
        Output(ids.SCATTER_INSPECTOR, "is_open"),
        Output(ids.SCATTER_INSPECTOR_TITLE, "children"),
        Output(ids.STORE_SCATTER_COLONY, "data"),
        Output(ids.SCATTER_INSPECTOR_MEASUREMENTS, "children"),
        Input(ids.SCATTER_GRAPH, "clickData"),
        State(ids.STORE_SCATTER_FINGERPRINT, "data"),
        prevent_initial_call=True,
    )
    def _open_inspector(
        click_data: object, fingerprint: object
    ) -> tuple[Any, Any, Any, Any]:
        return inspector_payload(output_root, click_data, fingerprint)

    @app.callback(
        Output(ids.SCATTER_INSPECTOR_CROP, "src"),
        Input(ids.STORE_SCATTER_COLONY, "data"),
        Input(ids.SCATTER_CONTOUR_TOGGLE, "value"),
    )
    def _crop_src(colony: object, contours: object) -> str:
        prefix = current_app.config.get(CFG_URL_PREFIX, MOUNT_HOME)
        return crop_url(prefix, colony, contours=contours)

    @app.callback(
        Output(ids.STORE_SCATTER_LEGEND, "data"),
        Input(ids.SCATTER_LEGEND_CORNER, "value"),
        Input(ids.SCATTER_LEGEND_COLLAPSE, "value"),
    )
    def _legend_state(corner: object, collapsed: object) -> dict[str, Any]:
        return {
            "corner": str(corner or LEGEND_CORNER_DEFAULT),
            "collapsed": bool(collapsed),
        }

    @app.callback(
        Output(ids.SCATTER_INSPECTOR, "style"),
        Input(ids.STORE_SCATTER_INSPECTOR_WIDTH, "data"),
    )
    def _inspector_width(width: object) -> dict[str, str]:
        # The drag itself has already clamped and applied the width; this
        # re-applies it so a Dash re-render of the offcanvas does not
        # snap the pane back to its mounted default.
        pixels = store_int(width)
        if pixels is None:
            return {}
        return {"width": f"{pixels}px"}

    @app.callback(
        Output(ids.SCATTER_DOWNLOAD, "data"),
        Output(ids.SCATTER_EXPORT_STATUS, "children"),
        Input(ids.SCATTER_EXPORT_BTN, "n_clicks"),
        State(ids.SCATTER_SECTION_COL, "value"),
        State(ids.SCATTER_ROW_COL, "value"),
        State(ids.SCATTER_COL_COL, "value"),
        State(ids.SCATTER_X_COL, "value"),
        State(ids.SCATTER_Y_COL, "value"),
        State(ids.SCATTER_HUE_COL, "value"),
        State(ids.SCATTER_SHAPE_COL, "value"),
        State(ids.SCATTER_SHOW_REMOVED, "value"),
        State(rv_ids.STORE_FILTER_SPEC, "data"),
        State(rv_ids.STORE_REMOVED_KEYS, "data"),
        prevent_initial_call=True,
    )
    def _export(  # noqa: PLR0913 - signature mirrors the State list
        n_clicks: object,
        section_col: str | None,
        row_col: str | None,
        col_col: str | None,
        x_col: str | None,
        y_col: str | None,
        hue_col: str | None,
        shape_col: str | None,
        show_removed: object,
        filter_payload: object,
        removed_payload: object,
    ) -> tuple[Any, str]:
        return export_payload(
            output_root,
            n_clicks=n_clicks,
            section_col=section_col,
            row_col=row_col,
            col_col=col_col,
            x_col=x_col,
            y_col=y_col,
            hue_col=hue_col,
            shape_col=shape_col,
            show_removed=show_removed,
            filter_payload=filter_payload,
            removed_payload=removed_payload,
        )


__all__ = [
    "build_render_state",
    "clamp_section_index",
    "click_index",
    "crop_url",
    "export_payload",
    "inspector_payload",
    "join_curation_flags",
    "legend_layout",
    "measurement_children",
    "paged_section_index",
    "prepare_frame",
    "register_callbacks",
    "resolve_inspector_click",
    "section_values",
    "store_int",
]
