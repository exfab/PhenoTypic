"""Single-pane viewer card for the results viewer.

A "card" is one independent OpenSeadragon viewport on a particular
``(dataset, image_stem)`` pair, with a header, info chips, a remove
button, and a collapsible per-object ``DataTable`` underneath. Cards
share the filter sidebar and image list but pick images independently;
many cards can be open at once for side-by-side comparison.

This module owns:

* :func:`layout` -- the per-card component tree, keyed by a hex UUID.
* :func:`register_callbacks` -- every callback that mutates card state
  (spawn / remove / picker options / per-card state / info chips /
  details toggle / details DataTable). The clientside OpenSeadragon
  mount/dispose is the responsibility of Wave 3C and the integrator
  (Wave 4) -- this module only renders an empty ``html.Div`` with the
  agreed pattern-matching id where the JS layer will mount.

Picker option-value encoding
-----------------------------
Each picker option's ``value`` is a JSON-encoded
``{"dataset": str, "stem": str}`` string. JSON is preferred over a
``"<ds>||<stem>"`` separator because dataset/stem strings can in
principle contain arbitrary characters; JSON round-trips through
:mod:`json` without escaping pitfalls.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
import polars as pl
from dash import ALL, MATCH, Input, Output, State, ctx, dash_table, dcc, html, no_update

from phenotypic.gui.results_viewer._filter_state import FilterSpec
from phenotypic.gui.results_viewer._ids import (
    BTN_ADD_CARD,
    CARDS_CONTAINER_ID,
    INITIAL_CARD_TRIGGER_ID,
    STORE_CARD_LIST,
    STORE_FILTER_SPEC,
    STORE_IMAGE_PAIRS,
    card_details_collapse_id,
    card_details_table_id,
    card_details_toggle_id,
    card_id,
    card_info_chip_count_id,
    card_info_chip_dataset_id,
    card_info_chip_stem_id,
    card_osd_div_id,
    card_picker_id,
    card_remove_id,
    card_state_store_id,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)


# Hard cap on rows projected into the per-object DataTable. DataTable
# paginates natively, but very large frames slow client-side rendering.
_MAX_DETAILS_ROWS = 5000

#: Prefix used to recognise metadata columns (``Metadata_Dataset`` etc.).
_METADATA_PREFIX = "Metadata_"

#: Default OSD canvas height; can be overridden by Wave 3C CSS.
_OSD_CANVAS_STYLE: dict[str, str] = {"height": "600px", "width": "100%"}


# ---------------------------------------------------------------------------
# Picker option-value encoding helpers
# ---------------------------------------------------------------------------


def _encode_picker_value(dataset: str, stem: str) -> str:
    """Encode a ``(dataset, stem)`` pair as a JSON dropdown ``value``.

    Args:
        dataset: Dataset name (matches ``Metadata_Dataset``).
        stem: Image stem (matches ``Metadata_ImageFile``).

    Returns:
        A JSON string of shape ``{"dataset": ..., "stem": ...}``.
    """
    return json.dumps({"dataset": dataset, "stem": stem}, separators=(",", ":"))


def _decode_picker_value(value: str | None) -> tuple[str, str] | None:
    """Decode a picker ``value`` back into ``(dataset, stem)``.

    Args:
        value: The picker's ``value`` (a JSON string previously produced
            by :func:`_encode_picker_value`) or ``None`` for an unset
            picker.

    Returns:
        Tuple ``(dataset, stem)`` on success, ``None`` if ``value`` is
        falsy or unparseable. Malformed payloads log at ``WARNING`` and
        return ``None`` rather than raising -- the picker should never
        crash the app on a stale store entry.
    """
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Could not decode picker value: %r", value)
        return None
    if not isinstance(parsed, dict):
        return None
    dataset = parsed.get("dataset")
    stem = parsed.get("stem")
    if not isinstance(dataset, str) or not isinstance(stem, str):
        return None
    return dataset, stem


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def layout(card_id: str, output_root: OutputRoot) -> Any:
    """Build the component tree for a single viewer card.

    The card is a self-contained ``dbc.Card`` with:

    1. A header row holding the image-picker dropdown, dataset/stem/
       n-objects info chips, and a remove (``x``) button.
    2. A full-bleed OpenSeadragon canvas div (the JS layer in
       Wave 3C mounts an OSD viewer here when an image is selected).
    3. A details toggle button and a collapsible per-object
       :class:`dash_table.DataTable`.
    4. A per-card :class:`dcc.Store` holding the selected
       ``(dataset, stem)`` plus the columns active in the current
       filter spec.

    Args:
        card_id: Hex UUID identifying this card. Used as the ``index``
            of every pattern-matching component id below.
        output_root: Validated CLI output handle. Currently unused at
            layout time -- options/values are populated by callbacks --
            but accepted so the module signature stays compatible
            with the dispatching ``register_callbacks`` flow.

    Returns:
        A :class:`dash_bootstrap_components.Card` ready to drop into
        the cards container. Typed ``Any`` because Dash / dbc do not
        ship complete stub coverage.
    """
    del output_root  # currently only callbacks need it; future-proof signature.

    picker = dcc.Dropdown(
        id=card_picker_id(card_id),
        options=[],
        value=None,
        placeholder="Select image...",
        clearable=True,
        searchable=True,
        style={"flex": "1 1 auto", "minWidth": "12rem"},
    )

    info_chips = html.Div(
        [
            dbc.Badge(
                "--",
                id=card_info_chip_dataset_id(card_id),
                color="secondary",
                className="card-info-chip me-1",
            ),
            dbc.Badge(
                "--",
                id=card_info_chip_stem_id(card_id),
                color="info",
                className="card-info-chip me-1",
            ),
            dbc.Badge(
                "-- objects",
                id=card_info_chip_count_id(card_id),
                color="light",
                text_color="dark",
                className="card-info-chip",
            ),
        ],
        className="card-info-chips d-flex align-items-center",
        style={"gap": "0.25rem"},
    )

    remove_btn = dbc.Button(
        "x",
        id=card_remove_id(card_id),
        color="danger",
        outline=True,
        size="sm",
        className="card-remove-btn ms-2",
        title="Remove this card",
    )

    header = dbc.CardHeader(
        html.Div(
            [
                html.Div(picker, style={"flex": "1 1 auto", "minWidth": "10rem"}),
                html.Div(
                    info_chips,
                    className="card-info-chips-wrap",
                    style={"flex": "0 0 auto"},
                ),
                remove_btn,
            ],
            className="d-flex align-items-center",
            style={"gap": "0.5rem", "flexWrap": "wrap"},
        )
    )

    osd_div = html.Div(
        id=card_osd_div_id(card_id),
        className="osd-canvas",
        style=_OSD_CANVAS_STYLE,
    )

    details_toggle = dbc.Button(
        "> Details",
        id=card_details_toggle_id(card_id),
        color="link",
        outline=False,
        size="sm",
        className="card-details-toggle p-0",
    )

    details_table = dash_table.DataTable(  # type: ignore[attr-defined]
        id=card_details_table_id(card_id),
        columns=[],
        data=[],
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "DM Mono, monospace",
            "fontSize": "12px",
            "padding": "4px 8px",
            "textAlign": "left",
        },
        style_header={
            "fontFamily": "DM Mono, monospace",
            "fontWeight": "500",
            "textTransform": "uppercase",
            "letterSpacing": "0.08em",
            "color": "#8892a4",
            "borderBottom": "2px solid #003660",
        },
    )

    details_collapse = dbc.Collapse(
        html.Div(details_table, className="card-details-table-wrap"),
        id=card_details_collapse_id(card_id),
        is_open=False,
        className="card-details-collapse",
    )

    state_store = dcc.Store(
        id=card_state_store_id(card_id),
        data=None,
        storage_type="memory",
    )

    body = dbc.CardBody(
        [
            osd_div,
            html.Div(
                details_toggle,
                className="card-details-toggle-wrap mt-2",
            ),
            details_collapse,
            state_store,
        ],
        className="card-body p-3",
    )

    return dbc.Card(
        [header, body],
        id=card_id_pattern(card_id),
        className="viewer-card mb-3",
    )


def card_id_pattern(idx: str) -> dict[str, str]:
    """Wrap :func:`card_id` for clarity at layout call sites.

    Kept separate from the imported :func:`card_id` so the local
    ``card_id`` parameter name in :func:`layout` is unambiguous.

    Args:
        idx: The card's hex UUID.

    Returns:
        Pattern-matching id dict.
    """
    return card_id(idx)


# ---------------------------------------------------------------------------
# Helpers used by callbacks
# ---------------------------------------------------------------------------


def _build_picker_options(
    pairs: list[tuple[str, str]] | list[list[str]] | list[Any],
    output_root: OutputRoot,
) -> list[dict[str, Any]]:
    """Translate a ``[(ds, stem), ...]`` list into Dash dropdown options.

    Pairs whose overlay PNG is missing are kept (so the user still sees
    them) but rendered as ``disabled`` with a leading warning marker so
    the missing-overlay state is visually obvious.

    Args:
        pairs: Iterable of ``(dataset, stem)`` tuples (or 2-element
            lists, since Dash stores round-trip tuples as lists).
        output_root: Validated handle on the output root, used to
            answer :meth:`OutputRoot.has_overlay`.

    Returns:
        A list of dicts shaped
        ``{"label": str, "value": str, "disabled": bool, "title": str}``.
    """
    options: list[dict[str, Any]] = []
    for raw in pairs or []:
        if not raw:
            continue
        try:
            dataset = str(raw[0])
            stem = str(raw[1])
        except (IndexError, TypeError):
            logger.debug("Skipping malformed image pair entry: %r", raw)
            continue
        present = output_root.has_overlay(dataset, stem)
        if present:
            label = f"{dataset} / {stem}"
            tooltip = f"{dataset} / {stem}"
        else:
            label = f"(no overlay) {dataset} / {stem}"
            tooltip = (
                f"{dataset} / {stem} -- overlay PNG missing on disk; "
                "the viewer cannot render this image."
            )
        options.append(
            {
                "label": label,
                "value": _encode_picker_value(dataset, stem),
                "disabled": not present,
                "title": tooltip,
            }
        )
    return options


def _filter_active_columns(filter_spec_payload: list[dict] | None) -> list[str]:
    """Return the column names that have at least one accepted value.

    Args:
        filter_spec_payload: Raw payload from
            :data:`STORE_FILTER_SPEC` (a list of
            ``{"column": str, "values": list[str]}`` dicts) or
            ``None``.

    Returns:
        Sorted list of column names whose filter rows currently
        constrain the master frame.
    """
    spec = FilterSpec.from_store(filter_spec_payload)
    columns = sorted({row.column for row in spec.rows if row.column and row.values})
    return columns


def _slice_for_image(
    output_root: OutputRoot, dataset: str, stem: str
) -> pl.DataFrame:
    """Return master rows for a single ``(dataset, stem)`` pair.

    Both columns are cast to ``pl.String`` for comparison so numeric
    stems (or older runs that wrote integer-typed dataset ids) match
    cleanly against the JSON-decoded picker values.
    """
    return output_root.master_df.filter(
        (pl.col("Metadata_Dataset").cast(pl.String) == dataset)
        & (pl.col("Metadata_ImageFile").cast(pl.String) == stem)
    )


def _project_details_columns(
    df: pl.DataFrame, filter_columns: list[str]
) -> list[str]:
    """Pick which columns to show in the per-object DataTable.

    The projection is: every ``Metadata_*`` column, plus every column
    referenced in the active filter spec, deduplicated and intersected
    with the columns actually present in *df*. This lets the user see
    why a row passed/failed each clause without dragging in the full
    measurement payload.
    """
    available = set(df.columns)
    metadata_cols = [c for c in df.columns if c.startswith(_METADATA_PREFIX)]
    extra = [c for c in filter_columns if c in available and c not in metadata_cols]
    return metadata_cols + extra


# ---------------------------------------------------------------------------
# Layout injection helpers
# ---------------------------------------------------------------------------


def _ensure_initial_card_trigger(app: dash.Dash) -> None:
    """Append the one-shot initial-card ``dcc.Interval`` to ``app.layout``.

    The interval fires exactly once (``max_intervals=1``) shortly after
    page load, which kicks the seeding callback to push a fresh UUID
    into :data:`STORE_CARD_LIST` if it's empty. Injection is
    best-effort: if the host layout already includes an interval with
    :data:`INITIAL_CARD_TRIGGER_ID`, we leave it alone.

    Args:
        app: Dash application whose ``layout`` (if already set) will
            be wrapped to include the interval.
    """
    interval = dcc.Interval(
        id=INITIAL_CARD_TRIGGER_ID, interval=200, max_intervals=1, n_intervals=0
    )
    layout_obj = getattr(app, "layout", None)
    if layout_obj is None:
        # Layout not yet assigned -- expose the interval via a children
        # attribute on the app so the layout owner can opt-in. We don't
        # raise: callbacks remain valid even without the trigger
        # because suppress_callback_exceptions is the project default.
        return
    if _layout_already_has_trigger(layout_obj):
        return
    try:
        app.layout = html.Div([layout_obj, interval])
    except Exception:
        logger.debug("Could not inject initial-card trigger into app.layout", exc_info=True)


def _layout_already_has_trigger(layout_obj: Any) -> bool:
    """Walk a Dash component tree looking for the initial-card interval id.

    The walker is best-effort: it inspects the ``id`` attribute of the
    root and recurses into ``children`` (handling lists, tuples, and
    single components). On any unexpected shape it falls back to
    ``False`` so the caller will inject a fresh interval rather than
    raise.
    """

    def visit(node: Any) -> bool:
        if node is None:
            return False
        if getattr(node, "id", None) == INITIAL_CARD_TRIGGER_ID:
            return True
        children = getattr(node, "children", None)
        if children is None:
            return False
        if isinstance(children, (list, tuple)):
            return any(visit(child) for child in children)
        return visit(children)

    try:
        return visit(layout_obj)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def register_callbacks(app: dash.Dash, output_root: OutputRoot) -> None:
    """Register every viewer-card callback on *app*.

    Owned callbacks (in registration order):

    1. Render the cards container from :data:`STORE_CARD_LIST`.
    2. Append a fresh card on :data:`BTN_ADD_CARD` clicks.
    3. Seed one card on first page load via a one-shot
       :class:`dcc.Interval`.
    4. Drop a card on its remove-button click (pattern-matching).
    5. Populate every picker's ``options`` from
       :data:`STORE_IMAGE_PAIRS` (pattern-matching).
    6. Persist each card's ``(dataset, stem)`` selection plus the
       active filter columns into its private state store
       (pattern-matching, MATCH).
    7. Update the three info chips (dataset / stem / n_objects) per
       card (pattern-matching, MATCH).
    8. Toggle the details ``dbc.Collapse`` per card.
    9. Populate the per-object DataTable from the card's state.

    Args:
        app: The Dash application that will own the callbacks.
        output_root: Validated handle on the CLI output directory;
            passed by closure into every callback that needs to slice
            ``master_df`` or call :meth:`OutputRoot.has_overlay`.

    Notes:
        The seeding callback fires from a ``dcc.Interval`` with
        ``max_intervals=1`` that this function injects into
        ``app.layout`` (when one is set). With
        ``suppress_callback_exceptions=True`` (the project default),
        callbacks whose component anchors are absent from the
        committed layout are simply skipped, so the injection is a
        best-effort convenience for layouts that haven't already
        included the interval themselves.
    """
    _ensure_initial_card_trigger(app)

    # 1. Render the cards container from STORE_CARD_LIST.
    @app.callback(
        Output(CARDS_CONTAINER_ID, "children"),
        Input(STORE_CARD_LIST, "data"),
    )
    def _render_cards(card_ids: list[str] | None) -> list[Any]:
        if not card_ids:
            return []
        return [layout(cid, output_root) for cid in card_ids if cid]

    # 2. Append a fresh card on every "+ Add card" click.
    @app.callback(
        Output(STORE_CARD_LIST, "data", allow_duplicate=True),
        Input(BTN_ADD_CARD, "n_clicks"),
        State(STORE_CARD_LIST, "data"),
        prevent_initial_call=True,
    )
    def _add_card(n_clicks: int | None, current: list[str] | None) -> Any:
        if not n_clicks:
            return no_update
        next_list = list(current or [])
        next_list.append(uuid.uuid4().hex)
        return next_list

    # 3. Seed an initial card on first page load if the list is empty.
    @app.callback(
        Output(STORE_CARD_LIST, "data", allow_duplicate=True),
        Input(INITIAL_CARD_TRIGGER_ID, "n_intervals"),
        State(STORE_CARD_LIST, "data"),
        prevent_initial_call=True,
    )
    def _seed_initial_card(
        n_intervals: int | None, current: list[str] | None
    ) -> Any:
        if not n_intervals:
            return no_update
        if current:
            return no_update
        return [uuid.uuid4().hex]

    # 4. Remove a card. Pattern-match on the remove buttons.
    @app.callback(
        Output(STORE_CARD_LIST, "data", allow_duplicate=True),
        Input({"type": "card-remove", "index": ALL}, "n_clicks"),
        State(STORE_CARD_LIST, "data"),
        prevent_initial_call=True,
    )
    def _remove_card(
        n_clicks_list: list[int | None], current: list[str] | None
    ) -> Any:
        triggered = ctx.triggered_id
        if not triggered or not isinstance(triggered, dict):
            return no_update
        # Dash fires the callback once with all-zero clicks on initial
        # registration; gate on a real click value.
        triggered_value = ctx.triggered[0].get("value") if ctx.triggered else None
        if not triggered_value:
            return no_update
        target = triggered.get("index")
        if not target or not current:
            return no_update
        next_list = [cid for cid in current if cid != target]
        if next_list == current:
            return no_update
        return next_list

    # 5. Populate every card picker's options from STORE_IMAGE_PAIRS.
    @app.callback(
        Output({"type": "card-picker", "index": ALL}, "options"),
        Input(STORE_IMAGE_PAIRS, "data"),
        State({"type": "card-picker", "index": ALL}, "id"),
    )
    def _populate_picker_options(
        pairs: list[Any] | None, picker_ids: list[dict[str, str]]
    ) -> list[list[dict[str, Any]]]:
        options = _build_picker_options(pairs or [], output_root)
        return [options for _ in picker_ids]

    # 6. Persist per-card state (selected pair + active filter columns).
    @app.callback(
        Output({"type": "card-state", "index": MATCH}, "data"),
        Input({"type": "card-picker", "index": MATCH}, "value"),
        State(STORE_FILTER_SPEC, "data"),
    )
    def _persist_card_state(
        picker_value: str | None, filter_payload: list[dict] | None
    ) -> dict[str, Any] | None:
        decoded = _decode_picker_value(picker_value)
        active_columns = _filter_active_columns(filter_payload)
        if decoded is None:
            return {
                "dataset": None,
                "stem": None,
                "filter_columns": active_columns,
            }
        dataset, stem = decoded
        return {
            "dataset": dataset,
            "stem": stem,
            "filter_columns": active_columns,
        }

    # 7. Update info chips (dataset / stem / n_objects) for a card.
    @app.callback(
        Output({"type": "card-info-dataset", "index": MATCH}, "children"),
        Output({"type": "card-info-stem", "index": MATCH}, "children"),
        Output({"type": "card-info-count", "index": MATCH}, "children"),
        Input({"type": "card-state", "index": MATCH}, "data"),
        State(STORE_FILTER_SPEC, "data"),
    )
    def _update_info_chips(
        state: dict[str, Any] | None, filter_payload: list[dict] | None
    ) -> tuple[str, str, str]:
        if not state:
            return "--", "--", "-- objects"
        dataset = state.get("dataset")
        stem = state.get("stem")
        if not dataset or not stem:
            return "--", "--", "-- objects"
        try:
            slice_df = _slice_for_image(output_root, dataset, stem)
            spec = FilterSpec.from_store(filter_payload)
            filtered = spec.apply_to(slice_df)
            n_objects = filtered.height
        except Exception:
            logger.exception(
                "Failed to compute info-chip counts for %s/%s",
                dataset,
                stem,
            )
            n_objects = 0
        return dataset, stem, f"{n_objects} objects"

    # 8. Toggle the details collapse.
    @app.callback(
        Output({"type": "card-details-collapse", "index": MATCH}, "is_open"),
        Input({"type": "card-details-toggle", "index": MATCH}, "n_clicks"),
        State({"type": "card-details-collapse", "index": MATCH}, "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_details(n_clicks: int | None, is_open: bool | None) -> bool:
        if not n_clicks:
            return bool(is_open)
        return not bool(is_open)

    # 9. Populate the per-object DataTable.
    @app.callback(
        Output({"type": "card-details-table", "index": MATCH}, "columns"),
        Output({"type": "card-details-table", "index": MATCH}, "data"),
        Input({"type": "card-state", "index": MATCH}, "data"),
        State(STORE_FILTER_SPEC, "data"),
    )
    def _populate_details_table(
        state: dict[str, Any] | None, filter_payload: list[dict] | None
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        if not state:
            return [], []
        dataset = state.get("dataset")
        stem = state.get("stem")
        if not dataset or not stem:
            return [], []
        try:
            slice_df = _slice_for_image(output_root, dataset, stem)
            spec = FilterSpec.from_store(filter_payload)
            filtered = spec.apply_to(slice_df)
            filter_columns = [
                row.column for row in spec.rows if row.column and row.values
            ]
            project = _project_details_columns(filtered, filter_columns)
            if not project:
                return [], []
            projected = filtered.select(project)
            if projected.height > _MAX_DETAILS_ROWS:
                logger.info(
                    "Capping details DataTable for %s/%s at %d rows (had %d)",
                    dataset,
                    stem,
                    _MAX_DETAILS_ROWS,
                    projected.height,
                )
                projected = projected.head(_MAX_DETAILS_ROWS)
            columns = [{"name": col, "id": col} for col in project]
            data = projected.to_dicts()
        except Exception:
            logger.exception(
                "Failed to populate details table for %s/%s", dataset, stem
            )
            return [], []
        return columns, data


__all__ = [
    "layout",
    "register_callbacks",
]
