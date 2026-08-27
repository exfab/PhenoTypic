"""Single-pane viewer card for the results viewer.

A "card" is one independent full-canvas Viv stage on a particular
``(dataset, image_stem)`` pair. Every control floats OVER the stage --
image stepper and picker top-left, Layers panel top-right, zoom and the
served-level readout along the bottom -- with a provenance note and a
collapsible per-object ``DataTable`` underneath. Cards share the filter
offcanvas and image list but pick images independently; many can be open
at once for side-by-side comparison, and "Lock views" mirrors one stage's
viewState onto the rest.

This module owns:

* :func:`layout` -- the per-card component tree, keyed by a hex UUID.
* :func:`register_callbacks` -- every callback that mutates card state
  (spawn / remove / picker options / per-card state / info chips /
  details toggle / details DataTable / source spec / Layers panel).
  The deck.gl mount and teardown are owned by the JS layer in
  ``_assets/results_viewer.js`` driving ``window.phenotypicViv``; this
  module renders an empty ``html.Div`` with the agreed pattern-matching
  id where the stage mounts, and resolves the SPEC that stage reads.

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
from typing import Any, cast

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
import polars as pl
from dash import (
    ALL,
    MATCH,
    Input,
    Output,
    Patch,
    State,
    ctx,
    dash_table,
    dcc,
    html,
    no_update,
)

from phenotypic.gui._config import CFG_FILTERED_STATE
from phenotypic.gui._design import (
    COLOR_MUTED,
    COLOR_NAVY,
    FONT_FAMILY_MONO,
    FONT_SIZE_BODY_SM,
    FONT_SIZE_CAPTION,
    OI_GREEN,
    OKABE_ITO,
    OI_VERMILION_TEXT,
)
from phenotypic.gui.results_viewer._filter_state import FilterSpec
from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._filtered_state import (
    KEY_DATASET,
    KEY_IMAGE_FILE,
    KEY_OBJECT_LABEL,
    decode_removed_keys_payload,
)
from phenotypic.gui.results_viewer._mutation_guard import (
    OutputMutationBlocked,
    output_mutations_disabled,
    require_output_mutation,
)
from phenotypic.gui.results_viewer._ids import (
    BTN_ADD_CARD,
    CARDS_CONTAINER_ID,
    INITIAL_CARD_TRIGGER_ID,
    STORE_CARD_LIST,
    STORE_FILTER_SPEC,
    STORE_IMAGE_PAIRS,
    STORE_REMOVED_KEYS,
    card_details_collapse_id,
    card_details_table_id,
    card_details_toggle_id,
    card_display_state_id,
    card_id,
    card_info_chip_count_id,
    card_info_chip_dataset_id,
    card_info_chip_stem_id,
    card_layer_eye_id,
    card_layer_opacity_id,
    card_layers_panel_id,
    card_picker_next_id,
    card_picker_prev_id,
    card_picker_id,
    card_pyramid_readout_id,
    card_remove_id,
    card_source_note_id,
    card_source_store_id,
    card_stage_id,
    card_state_store_id,
    card_zoom_readout_id,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui._shared.tiles import StoreUnreadable
from phenotypic.gui.results_viewer._store_source import build_source_spec
from phenotypic.gui.results_viewer._zarr_routes import (
    store_generation_token,
    zarr_store_url,
)
from phenotypic.gui.results_viewer._picker_navigation import (
    picker_button_disabled_states,
    step_picker_value,
)
from phenotypic.sdk_ import is_metadata_header

logger = logging.getLogger(__name__)


# Hard cap on rows projected into the per-object DataTable. DataTable
# paginates natively, but very large frames slow client-side rendering.
_MAX_DETAILS_ROWS = 5000

#: Name of the objmap row in the Layers panel. It is the LABEL image, not a
#: series, and the panel tags it as such -- the store's ``series`` list never
#: contains it (``_save_store`` keeps the two namespaces apart on purpose).
_OBJMAP_LAYER = "objmap"

#: Facade layer ids. Mirrors ``IMAGE_LAYER_ID`` / ``LABEL_LAYER_ID`` in
#: ``_assets/viv_viewer.js``; the display state addresses layers by these.
_FACADE_IMAGE_LAYER = "image"
_FACADE_LABEL_LAYER = "labels"

#: Default opacity per facade layer, matching the facade's own defaults.
_DEFAULT_OPACITY: dict[str, float] = {
    _FACADE_IMAGE_LAYER: 1.0,
    _FACADE_LABEL_LAYER: 0.5,
}

#: Column id and human-facing name for the curation Status column injected
#: as the leftmost column of the per-object DataTable. Clicking a Status
#: cell toggles whether that ``(Metadata_ImageName, Object_Label)`` row is
#: marked as removed.
_STATUS_COLUMN_ID = "Status"

#: Cell value rendered when the row is *not* in
#: :attr:`FilteredMeasurements.removed_keys`.
_STATUS_ACTIVE = "Active"

#: Cell value rendered when the row *is* in
#: :attr:`FilteredMeasurements.removed_keys`.
_STATUS_REMOVED = "Removed"

#: Soft-vermilion row tint (Okabe-Ito vermilion at 10% alpha) applied to
#: rows whose Status cell reads "Removed".
_REMOVED_ROW_BG = "rgba(213, 94, 0, 0.10)"

#: Darkened vermilion text used for the row text when the row is removed;
#: keeps contrast acceptable against :data:`_REMOVED_ROW_BG`.
_REMOVED_ROW_FG = OI_VERMILION_TEXT


# ---------------------------------------------------------------------------
# Picker option-value encoding helpers
# ---------------------------------------------------------------------------


def _encode_picker_value(dataset: str, stem: str) -> str:
    """Encode a ``(dataset, stem)`` pair as a JSON dropdown ``value``.

    Args:
        dataset: Dataset name (matches ``Metadata_Dataset``).
        stem: Image stem (matches ``Metadata_ImageName``).

    Returns:
        A JSON string of shape ``{"dataset": ..., "stem": ...}``.
    """
    return json.dumps(
        {"dataset": dataset, "stem": stem}, separators=(",", ":")
    )


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


def layout(
    idx: str,
    output_root: OutputRoot,
    *,
    mutations_disabled: bool = False,
) -> Any:
    """Build the component tree for a single viewer card.

    The card is a self-contained ``dbc.Card`` with:

    1. A header row holding the image-picker dropdown, dataset/stem/
       n-objects info chips, and a remove (``x``) button.
    2. A full-canvas Viv stage with every control floating over it
       (the JS layer mounts deck.gl into it when an image is selected).
    3. A details toggle button and a collapsible per-object
       :class:`dash_table.DataTable`.
    4. A per-card :class:`dcc.Store` holding the selected
       ``(dataset, stem)`` plus the columns active in the current
       filter spec.

    Args:
        idx: Hex UUID identifying this card. Used as the ``index``
            of every pattern-matching component id below.
        output_root: Validated CLI output handle. Currently unused at
            layout time -- options/values are populated by callbacks --
            but accepted so the module signature stays compatible
            with the dispatching ``register_callbacks`` flow.
        mutations_disabled: Whether persistent table curation must be
            unavailable for this bound output.

    Returns:
        A :class:`dash_bootstrap_components.Card` ready to drop into
        the cards container. Typed ``Any`` because Dash / dbc do not
        ship complete stub coverage.
    """
    picker = dcc.Dropdown(
        id=card_picker_id(idx),
        options=[],
        value=None,
        placeholder="Select image...",
        clearable=True,
        searchable=True,
        style={"flex": "1 1 auto", "minWidth": "12rem"},
    )
    stepper_pair = html.Div(
        [
            html.Button(
                "‹",
                id=card_picker_prev_id(idx),
                n_clicks=0,
                title="Previous image",
                className=(
                    "btn btn-outline-secondary btn-sm "
                    "browse-step-button card-picker-nav-btn"
                ),
                type="button",
                **cast(Any, {"aria-label": "Previous image"}),
            ),
            html.Button(
                "›",
                id=card_picker_next_id(idx),
                n_clicks=0,
                title="Next image",
                className=(
                    "btn btn-outline-secondary btn-sm "
                    "browse-step-button card-picker-nav-btn"
                ),
                type="button",
                **cast(Any, {"aria-label": "Next image"}),
            ),
        ],
        className="btn-group",
        role="group",
        **cast(Any, {"aria-label": "Step through images"}),
    )
    picker_group = html.Div(
        [
            stepper_pair,
            html.Div(picker, style={"flex": "1 1 auto", "minWidth": "10rem"}),
        ],
        className="d-flex align-items-center",
        style={"gap": "0.35rem", "flex": "1 1 auto", "minWidth": "12rem"},
    )

    info_chips = html.Div(
        [
            dbc.Badge(
                "--",
                id=card_info_chip_dataset_id(idx),
                color="dark",
                className="card-info-chip",
            ),
            dbc.Badge(
                "--",
                id=card_info_chip_stem_id(idx),
                color="dark",
                className="card-info-chip",
            ),
            dbc.Badge(
                "-- objects",
                id=card_info_chip_count_id(idx),
                color="dark",
                className="card-info-chip",
            ),
        ],
        className="card-info-chips plate-float__chip",
    )

    remove_btn = dbc.Button(
        "x",
        id=card_remove_id(idx),
        color="danger",
        outline=True,
        size="sm",
        className="card-remove-btn",
        title="Remove this card",
    )

    # The Viv mount point. Empty at layout time and never written by Python:
    # the clientside bridge owns everything inside it.
    stage_canvas = html.Div(
        id=card_stage_id(idx),
        className="plate-stage__canvas",
    )

    layers_panel = html.Div(
        [
            html.Div(
                "Layers",
                className="plate-layers__title",
            ),
            html.Div(
                [],
                id=card_layers_panel_id(idx),
                className="plate-layers__rows",
            ),
        ],
        className="plate-layers",
    )

    # Bottom-right, over the stage. Written ONLY by the clientside bridge
    # from the facade's `onLevelChange`; see `card_pyramid_readout_id`.
    pyramid_readout = html.Div(
        "no image",
        id=card_pyramid_readout_id(idx),
        className="plate-float plate-float--bottom-right plate-readout",
    )
    zoom_readout = html.Div(
        "",
        id=card_zoom_readout_id(idx),
        className="plate-float plate-float--bottom-left plate-readout",
    )

    stage = html.Div(
        [
            stage_canvas,
            html.Div(
                [picker_group, info_chips],
                className="plate-float plate-float--top-left",
            ),
            html.Div(
                [remove_btn, layers_panel],
                className="plate-float plate-float--top-right",
            ),
            zoom_readout,
            pyramid_readout,
        ],
        className="plate-stage",
    )

    source_note = html.Div(
        "",
        id=card_source_note_id(idx),
        className="plate-source-note",
    )

    details_toggle = dbc.Button(
        "> Details",
        id=card_details_toggle_id(idx),
        color="link",
        outline=False,
        size="sm",
        className="card-details-toggle p-0",
    )

    details_table = dash_table.DataTable(  # type: ignore[attr-defined]
        id=card_details_table_id(idx),
        columns=[],
        data=[],
        page_size=20,
        sort_action="native",
        filter_action="native",
        # ``cell_selectable=True`` is required so ``active_cell`` events
        # fire on click; ``editable=False`` keeps Dash from opening an
        # in-place editor for the Status cell.
        cell_selectable=not mutations_disabled,
        editable=False,
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": FONT_FAMILY_MONO,
            "fontSize": FONT_SIZE_BODY_SM,
            "padding": "4px 8px",
            "textAlign": "left",
        },
        style_header={
            "fontFamily": FONT_FAMILY_MONO,
            "fontSize": FONT_SIZE_CAPTION,
            "fontWeight": "500",
            "textTransform": "uppercase",
            "letterSpacing": "0.08em",
            "color": COLOR_MUTED,
            "borderBottom": f"2px solid {COLOR_NAVY}",
        },
        # Tint rows whose Status cell reads "Removed" with a soft
        # vermilion background plus a darkened vermilion text color
        # (Okabe-Ito vermilion at 10% alpha; see ``_REMOVED_ROW_BG``).
        style_data_conditional=[
            {
                "if": {
                    "filter_query": f'{{{_STATUS_COLUMN_ID}}} = "{_STATUS_REMOVED}"'
                },
                "backgroundColor": _REMOVED_ROW_BG,
                "color": _REMOVED_ROW_FG,
            },
        ],
    )

    details_collapse = dbc.Collapse(
        html.Div(details_table, className="card-details-table-wrap"),
        id=card_details_collapse_id(idx),
        is_open=False,
        className="card-details-collapse",
    )

    return html.Div(
        [
            stage,
            html.Div(
                [source_note, details_toggle],
                className="plate-underbar",
            ),
            details_collapse,
            dcc.Store(
                id=card_state_store_id(idx),
                data=None,
                storage_type="session",
            ),
            # The resolved source spec, handed to the facade unmodified.
            dcc.Store(id=card_source_store_id(idx), data=None),
            # What the Layers panel has been set to. Kept apart from the
            # spec so a re-source (a promote, a stepped image) does not
            # silently reset the user's layer choices.
            dcc.Store(id=card_display_state_id(idx), data=None),
        ],
        id=card_id(idx),
        className="viewer-card plate-card",
    )


def build_layer_rows(
    idx: str, spec: dict[str, Any] | None, display: dict[str, Any] | None
) -> list[Any]:
    """Build the Layers-panel rows for one card from the store's REAL series.

    The list comes from ``spec["series"]`` -- the store's own
    ``attributes.phenotypic.series`` -- never from a literal
    ``{rgb, gray, detect_mat}``. ``_write_store_part`` appends ``original``
    whenever the image carries one, and an rgb-less store has no ``rgb``
    row at all; a hard-coded set would offer a layer the byte route 404s
    and hide one it serves.

    A series row SELECTS which series the image layer shows (Viv holds one
    image source at a time), so the eye reads as "displayed". The objmap row
    is different in kind -- it is the label image, drawn over the series --
    and its eye is a true visibility toggle.

    Args:
        idx: Owning card's ``index``.
        spec: The card's source spec, or ``None`` for an unselected card.
        display: The card's display state, or ``None`` for its defaults.

    Returns:
        One row component per readable layer; empty when *spec* is ``None``.
    """
    if not spec:
        return []
    state = display or {}
    active_series = state.get("seriesPath") or spec.get("seriesPath")
    label_visible = state.get("labelVisible", True)
    opacity = {**_DEFAULT_OPACITY, **(state.get("opacity") or {})}

    rows: list[Any] = []
    for position, name in enumerate(spec.get("series", [])):
        shown = name == active_series
        rows.append(
            _layer_row(
                idx,
                name=name,
                kind="series",
                swatch=OKABE_ITO[position % len(OKABE_ITO)],
                shown=shown,
                opacity=opacity[_FACADE_IMAGE_LAYER] if shown else 0.0,
                opacity_enabled=shown,
            )
        )
    if spec.get("labelPath"):
        rows.append(
            _layer_row(
                idx,
                name=_OBJMAP_LAYER,
                # An absent `tables` descriptor means Stage 3 has not run,
                # so the in-store objmap is still zeros. Saying so is the
                # difference between a user waiting and a user filing a bug
                # about a detector that "found nothing".
                kind="label image"
                if spec.get("measured")
                else "measurement pending",
                swatch=OI_GREEN,
                shown=bool(label_visible),
                opacity=opacity[_FACADE_LABEL_LAYER],
                opacity_enabled=True,
            )
        )
    return rows


def _layer_row(
    idx: str,
    *,
    name: str,
    kind: str,
    swatch: str,
    shown: bool,
    opacity: float,
    opacity_enabled: bool,
) -> Any:
    """Build one Layers-panel row: eye, swatch, name, kind tag, opacity."""
    eye = html.Button(
        "\u25c9" if shown else "\u25cc",
        id=card_layer_eye_id(idx, name),
        n_clicks=0,
        type="button",
        title=f"Show {name}" if not shown else f"Hide {name}",
        className=(
            "plate-layer__eye"
            + ("" if shown else " plate-layer__eye--off")
        ),
        **cast(Any, {"aria-label": f"Toggle {name}"}),
    )
    return html.Div(
        [
            eye,
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                className="plate-layer__swatch",
                                style={"background": swatch},
                            ),
                            html.Span(name, className="plate-layer__name"),
                            html.Span(kind, className="plate-layer__kind"),
                        ],
                        className="plate-layer__head",
                    ),
                    dcc.Slider(
                        id=card_layer_opacity_id(idx, name),
                        min=0,
                        max=1,
                        step=0.05,
                        value=opacity,
                        marks=None,
                        tooltip={"placement": "bottom"},
                        disabled=not opacity_enabled,
                        className="plate-layer__opacity",
                    ),
                ],
                className="plate-layer__body",
            ),
        ],
        className="plate-layer",
    )
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
        if isinstance(raw, dict):
            dataset_value = raw.get("dataset")
            stem_value = raw.get("stem")
            if dataset_value is None or stem_value is None:
                logger.debug("Skipping malformed image pair entry: %r", raw)
                continue
            dataset = str(dataset_value)
            stem = str(stem_value)
        else:
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


def _filter_active_columns(
    filter_spec_payload: list[dict] | None,
) -> list[str]:
    """Return the column names whose filter rows currently constrain the frame.

    Args:
        filter_spec_payload: Raw payload from
            :data:`STORE_FILTER_SPEC` (a list of per-row dicts carrying a
            ``column``, a ``method``, and the method's payload) or ``None``.

    Returns:
        Sorted list of column names whose filter rows are *active* under any
        method (list, range, compare, or contains) — i.e. rows for which
        :meth:`FilterRow.to_expr` yields a predicate rather than ``None``.
    """
    spec = FilterSpec.from_store(filter_spec_payload)
    columns = sorted(
        {
            row.column
            for row in spec.rows
            if row.column and row.to_expr() is not None
        }
    )
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
        (pl.col(KEY_DATASET).cast(pl.String) == dataset)
        & (pl.col(KEY_IMAGE_FILE).cast(pl.String) == stem)
    )


def _decode_removed_keys_payload(
    payload: list[list[Any]] | None,
) -> set[tuple[str, int]]:
    """Decode :data:`STORE_REMOVED_KEYS` into a hash-set for membership tests.

    Thin wrapper around the shared :func:`decode_removed_keys_payload`
    helper that turns the list-of-tuples result into a set so the
    per-row Status check is O(1).
    """
    return set(decode_removed_keys_payload(payload))


def _row_status(
    image_file: Any,
    object_label: Any,
    removed_keys_set: set[tuple[str, int]],
) -> str:
    """Resolve the Status cell value for a single DataTable row.

    Args:
        image_file: Value of ``Metadata_ImageName`` from the row dict.
            Coerced to ``str`` to match the lookup-set dtype.
        object_label: Value of ``Object_Label`` from the row dict.
            Coerced to ``int`` to match the lookup-set dtype.
        removed_keys_set: Lookup set produced by
            :func:`_decode_removed_keys_payload`.

    Returns:
        ``"Removed"`` when the row's key is in the lookup set, else
        ``"Active"``. Rows whose key cannot be coerced -- e.g.
        ``Object_Label`` is ``None`` -- default to ``"Active"`` rather
        than raising; this is conservative because misclassifying a row
        as removed would silently strip it from exports.
    """
    if image_file is None or object_label is None:
        return _STATUS_ACTIVE
    try:
        key = (str(image_file), int(object_label))
    except (TypeError, ValueError):
        return _STATUS_ACTIVE
    return _STATUS_REMOVED if key in removed_keys_set else _STATUS_ACTIVE


def _project_details_columns(
    df: pl.DataFrame, filter_columns: list[str]
) -> list[str]:
    """Pick which columns to show in the per-object DataTable.

    The projection is: every ``Metadata_*`` column, plus ``Object_Label``
    (so the per-row Status toggle callback can resolve the curation key),
    plus every column referenced in the active filter spec, deduplicated
    and intersected with the columns actually present in *df*. This
    lets the user see why a row passed/failed each clause without
    dragging in the full measurement payload.
    """
    available = set(df.columns)
    metadata_cols = [c for c in df.columns if is_metadata_header(c)]
    object_label_cols = (
        [KEY_OBJECT_LABEL] if KEY_OBJECT_LABEL in available else []
    )
    seen = set(metadata_cols) | set(object_label_cols)
    extra = [c for c in filter_columns if c in available and c not in seen]
    return metadata_cols + object_label_cols + extra


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
        id=INITIAL_CARD_TRIGGER_ID,
        interval=200,
        max_intervals=1,
        n_intervals=0,
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
        logger.debug(
            "Could not inject initial-card trigger into app.layout",
            exc_info=True,
        )


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
    10. Resolve the card's Viv source spec from the selected image's
        OME-Zarr store, and render the Layers panel from the store's own
        series list.
    11. Fold Layers-panel clicks into the card's display state.

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

    # Capture the curation backend at registration time. Dash callbacks
    # don't run inside a Flask request, so reaching for
    # ``flask.current_app`` later would fail. ``None`` is tolerated so
    # tests / harnesses that don't seed the config still register
    # callbacks (the toggle becomes a no-op in that case).
    filtered_state: CurationLabels | None = app.server.config.get(
        CFG_FILTERED_STATE
    )

    # Diff the desired card list against the card IDs rendered in the
    # requesting browser. The client-specific State keeps reloads and
    # concurrent sessions independent while Patch preserves existing stage
    # viewers during sibling add/remove operations.
    @app.callback(
        Output(CARDS_CONTAINER_ID, "children"),
        Input(STORE_CARD_LIST, "data"),
        State({"type": "card", "index": ALL}, "id"),
    )
    def _render_cards(
        card_ids: list[str] | None,
        rendered_card_ids: list[dict[str, Any]] | None,
    ) -> Any:
        target = [cid for cid in (card_ids or []) if cid]
        rendered_ids = [
            str(rendered_id["index"])
            for rendered_id in (rendered_card_ids or [])
            if isinstance(rendered_id, dict) and rendered_id.get("index")
        ]

        # First-time render: build the container from scratch.
        if not rendered_ids:
            if not target:
                return []
            return [
                layout(
                    cid,
                    output_root,
                    mutations_disabled=output_mutations_disabled(output_root),
                )
                for cid in target
            ]

        # No-op if nothing actually changed.
        if target == rendered_ids:
            return no_update

        # Diff existing -> target.
        target_set = set(target)
        patch = Patch()
        # Remove deleted cards from highest index first to keep indices stable.
        for idx in range(len(rendered_ids) - 1, -1, -1):
            if rendered_ids[idx] not in target_set:
                del patch[idx]
        # Append cards that are in target but not currently rendered.
        existing_set = set(rendered_ids)
        for cid in target:
            if cid not in existing_set:
                patch.append(
                    layout(
                        cid,
                        output_root,
                        mutations_disabled=output_mutations_disabled(output_root),
                    )
                )
        return patch

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
        triggered_value = (
            ctx.triggered[0].get("value") if ctx.triggered else None
        )
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

    # 6. Step an individual card picker from the icon-only navigation buttons.
    @app.callback(
        Output(
            {"type": "card-picker", "index": MATCH},
            "value",
            allow_duplicate=True,
        ),
        Input({"type": "card-picker-prev", "index": MATCH}, "n_clicks"),
        Input({"type": "card-picker-next", "index": MATCH}, "n_clicks"),
        State({"type": "card-picker", "index": MATCH}, "value"),
        State({"type": "card-picker", "index": MATCH}, "options"),
        prevent_initial_call=True,
    )
    def _step_card_picker(
        _prev_clicks: int | None,
        _next_clicks: int | None,
        current: str | None,
        options: list[dict[str, Any]] | None,
    ) -> str | Any:
        triggered = ctx.triggered_id
        if (
            isinstance(triggered, dict)
            and triggered.get("type") == "card-picker-prev"
        ):
            return step_picker_value(current, options, "previous") or no_update
        if (
            isinstance(triggered, dict)
            and triggered.get("type") == "card-picker-next"
        ):
            return step_picker_value(current, options, "next") or no_update
        return no_update

    # 7. Disable per-card navigation buttons at picker bounds.
    @app.callback(
        Output({"type": "card-picker-prev", "index": MATCH}, "disabled"),
        Output({"type": "card-picker-next", "index": MATCH}, "disabled"),
        Input({"type": "card-picker", "index": MATCH}, "value"),
        Input({"type": "card-picker", "index": MATCH}, "options"),
    )
    def _sync_card_picker_nav_disabled(
        current: str | None,
        options: list[dict[str, Any]] | None,
    ) -> tuple[bool, bool]:
        return picker_button_disabled_states(current, options)

    # 8. Persist per-card state (selected pair + active filter columns).
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

    # 9. Card payload: info chips + per-object DataTable, computed in
    #    one pass so the (slice + filter) work isn't duplicated across
    #    two callbacks that fire on the same trigger. ``STORE_REMOVED_KEYS``
    #    is an ``Input`` rather than a ``State`` because flipping a row's
    #    Status must immediately re-render the affected card -- e.g. when
    #    a colony-grid bulk-remove updates the store, every open card
    #    table should pick up the new tint.
    @app.callback(
        Output({"type": "card-info-dataset", "index": MATCH}, "children"),
        Output({"type": "card-info-stem", "index": MATCH}, "children"),
        Output({"type": "card-info-count", "index": MATCH}, "children"),
        Output({"type": "card-details-table", "index": MATCH}, "columns"),
        Output({"type": "card-details-table", "index": MATCH}, "data"),
        Input({"type": "card-state", "index": MATCH}, "data"),
        Input(STORE_REMOVED_KEYS, "data"),
        State(STORE_FILTER_SPEC, "data"),
    )
    def _update_card_payload(
        state: dict[str, Any] | None,
        removed_keys_payload: list[list[Any]] | None,
        filter_payload: list[dict] | None,
    ) -> tuple[str, str, str, list[dict[str, str]], list[dict[str, Any]]]:
        empty_chips = ("--", "--", "-- objects")
        if not state:
            return *empty_chips, [], []
        dataset = state.get("dataset")
        stem = state.get("stem")
        if not dataset or not stem:
            return *empty_chips, [], []

        try:
            slice_df = _slice_for_image(output_root, dataset, stem)
            spec = FilterSpec.from_store(filter_payload)
            filtered = spec.apply_to(slice_df)
        except Exception:
            logger.exception(
                "Failed to slice/filter master for %s/%s", dataset, stem
            )
            return dataset, stem, "0 objects", [], []

        n_objects = filtered.height
        filter_columns = _filter_active_columns(filter_payload)
        project = _project_details_columns(filtered, filter_columns)
        if not project:
            return dataset, stem, f"{n_objects} objects", [], []
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
        # Project columns: Status leftmost, then metadata + filter columns.
        columns: list[dict[str, str]] = [
            {"name": _STATUS_COLUMN_ID, "id": _STATUS_COLUMN_ID}
        ]
        columns.extend({"name": col, "id": col} for col in project)

        removed_keys_set = _decode_removed_keys_payload(removed_keys_payload)
        rows = projected.to_dicts()
        for row in rows:
            image_file = row.get(KEY_IMAGE_FILE)
            label = row.get(KEY_OBJECT_LABEL)
            row[_STATUS_COLUMN_ID] = _row_status(
                image_file, label, removed_keys_set
            )
        return dataset, stem, f"{n_objects} objects", columns, rows

    # 10. Toggle the details collapse.
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

    # 9. Click-to-toggle on a Status cell. Pattern-matches on every
    #    card's details DataTable; the matching ``data`` State is read
    #    in lock-step so we can resolve the clicked cell's
    #    ``(Metadata_ImageName, Object_Label)`` without re-querying the
    #    master frame.
    @app.callback(
        Output(STORE_REMOVED_KEYS, "data", allow_duplicate=True),
        Input({"type": "card-details-table", "index": ALL}, "active_cell"),
        State({"type": "card-details-table", "index": ALL}, "data"),
        State({"type": "card-details-table", "index": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _toggle_status_cell(
        active_cells: list[dict[str, Any] | None],
        data_list: list[list[dict[str, Any]] | None],
        id_list: list[dict[str, str]],
    ) -> Any:
        """Flip the curation state when a Status cell is clicked.

        Args:
            active_cells: Per-table ``active_cell`` payloads, one entry
                per matched DataTable (Dash pads with ``None`` for tables
                with no active cell).
            data_list: Per-table ``data`` lists. Used to recover the
                clicked row's ``Metadata_ImageName`` / ``Object_Label``.
            id_list: Per-table ``id`` dicts. Matched index-aligned with
                ``active_cells`` so the triggered card can be located.

        Returns:
            The new :data:`STORE_REMOVED_KEYS` payload from
            :meth:`FilteredMeasurements.removed_keys_payload` after the
            toggle, or :func:`dash.no_update` if the click should not
            mutate state (curation backend missing, non-Status column,
            no row available, etc.).
        """
        if filtered_state is None:
            return no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update
        triggered_index = triggered.get("index")
        if triggered_index is None:
            return no_update

        # Locate the table whose id matches the trigger. ``id_list`` is
        # index-aligned with ``active_cells`` and ``data_list``.
        table_pos: int | None = None
        for pos, id_obj in enumerate(id_list):
            if id_obj.get("index") == triggered_index:
                table_pos = pos
                break
        if table_pos is None:
            return no_update

        active_cell = active_cells[table_pos]
        if not active_cell:
            return no_update
        if active_cell.get("column_id") != _STATUS_COLUMN_ID:
            return no_update

        rows = data_list[table_pos] or []
        row_idx = active_cell.get("row")
        if not isinstance(row_idx, int) or row_idx < 0 or row_idx >= len(rows):
            return no_update
        row = rows[row_idx]
        image_file_raw = row.get(KEY_IMAGE_FILE)
        label_raw = row.get(KEY_OBJECT_LABEL)
        if image_file_raw is None or label_raw is None:
            return no_update
        try:
            image_file = str(image_file_raw)
            object_label = int(label_raw)
        except (TypeError, ValueError):
            logger.debug(
                "Could not coerce Status toggle key (%r, %r)",
                image_file_raw,
                label_raw,
            )
            return no_update

        try:
            require_output_mutation("Plate details curation")
            payload = filtered_state.mutate_and_payload(
                lambda s: s.toggle(image_file, object_label)
            )
        except OutputMutationBlocked as exc:
            logger.warning("%s", exc)
            return no_update
        except Exception:
            logger.exception(
                "Failed to toggle curation state for %s / %d",
                image_file,
                object_label,
            )
            return no_update
        # ``STORE_REMOVED_KEYS`` is an ``allow_duplicate`` (multi-mode) output
        # whose value is itself a list. Restoring the LAST removed object yields
        # an empty payload ``[]``; a bare ``[]`` makes Dash's multi-mode response
        # validator see *zero* output values and 500. Wrap in a 1-tuple so Dash
        # sees exactly one value (the list) regardless of its length (matches
        # ``colony_view._callbacks._mark_colony_category``).
        return (payload,)

    # 10. Resolve the selected image's store into a source spec, and render
    #     the Layers panel from the store's REAL series list.
    #
    #     The spec crosses to `window.phenotypicViv.setSource` unmodified,
    #     which is why it is built at `build_source_spec`'s own key names
    #     rather than repacked here.
    @app.callback(
        Output({"type": "card-source-spec", "index": MATCH}, "data"),
        Output({"type": "card-layers-panel", "index": MATCH}, "children"),
        Output({"type": "card-source-note", "index": MATCH}, "children"),
        Output(
            {"type": "card-display-state", "index": MATCH},
            "data",
            allow_duplicate=True,
        ),
        Input({"type": "card-state", "index": MATCH}, "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _resolve_card_source(
        state: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, list[Any], str, dict[str, Any] | None]:
        dataset = (state or {}).get("dataset")
        stem = (state or {}).get("stem")
        if not dataset or not stem:
            return None, [], "", None
        store = output_root.store_path(str(dataset), str(stem))
        if store is None:
            # A standalone deliverables bundle ships overlays and no
            # `results/` stores. There is nothing for the pixel client to
            # read, and saying so beats an empty dark rectangle.
            return (
                None,
                [],
                f"no OME-Zarr store for {dataset}/{stem}",
                None,
            )
        try:
            token = store_generation_token(store)
            spec = build_source_spec(
                store,
                zarr_store_url(
                    app.config.requests_pathname_prefix,
                    str(dataset),
                    str(stem),
                    token,
                ),
            )
        except StoreUnreadable as exc:
            # Same condition the byte route answers 422 on. Surfacing the
            # store's own message here means the user reads it before the
            # canvas starts throwing on every chunk.
            logger.error("Unreadable store for %s/%s: %s", dataset, stem, exc)
            return None, [], str(exc), None
        except (OSError, KeyError, ValueError):
            logger.exception(
                "Failed to resolve a Viv source for %s/%s", dataset, stem
            )
            return None, [], f"could not read {dataset}/{stem}", None

        display = {
            "seriesPath": spec["seriesPath"],
            "labelVisible": True,
            "opacity": dict(_DEFAULT_OPACITY),
        }
        note = f"served directly from {store.name} - no tile cache"
        # MATCH callbacks do not receive the matched index as an
        # argument; it is read off the resolved output list.
        idx_ = str(ctx.outputs_list[0]["id"]["index"])
        return spec, build_layer_rows(idx_, spec, display), note, display

    # 11. Fold Layers-panel interaction into the card's display state.
    #
    #     Two separate gestures land here. An eye click on a SERIES row
    #     selects which series the image layer shows -- Viv holds one image
    #     source at a time, so this is a radio, not a checkbox. An eye click
    #     on the objmap row toggles the label layer's visibility. Sliders
    #     set the corresponding facade layer's opacity.
    @app.callback(
        Output(
            {"type": "card-display-state", "index": MATCH},
            "data",
            allow_duplicate=True,
        ),
        Output({"type": "card-layers-panel", "index": MATCH}, "children", allow_duplicate=True),
        Input({"type": "card-layer-eye", "index": MATCH, "layer": ALL}, "n_clicks"),
        Input(
            {"type": "card-layer-opacity", "index": MATCH, "layer": ALL},
            "value",
        ),
        State({"type": "card-display-state", "index": MATCH}, "data"),
        State({"type": "card-source-spec", "index": MATCH}, "data"),
        prevent_initial_call=True,
    )
    def _apply_layer_controls(
        _eye_clicks: list[int | None],
        _opacities: list[float | None],
        display: dict[str, Any] | None,
        spec: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | Any, list[Any] | Any]:
        trigger = ctx.triggered_id
        if not spec or not isinstance(trigger, dict):
            return no_update, no_update
        layer = str(trigger.get("layer") or "")
        if not layer:
            return no_update, no_update
        state = dict(display or {})
        state.setdefault("seriesPath", spec["seriesPath"])
        state.setdefault("labelVisible", True)
        state["opacity"] = {**_DEFAULT_OPACITY, **(state.get("opacity") or {})}

        if trigger.get("type") == "card-layer-eye":
            # A freshly rendered button fires this callback with
            # ``n_clicks == 0``; only a real click carries a count.
            if not ctx.triggered[0]["value"]:
                return no_update, no_update
            if layer == _OBJMAP_LAYER:
                state["labelVisible"] = not state["labelVisible"]
            elif layer == state["seriesPath"]:
                # Clicking the displayed series is a no-op rather than a
                # way to show nothing: Viv has one image source, and an
                # empty stage reads as a broken store.
                return no_update, no_update
            else:
                state["seriesPath"] = layer
        else:
            value = ctx.triggered[0]["value"]
            if value is None:
                return no_update, no_update
            facade_layer = (
                _FACADE_LABEL_LAYER
                if layer == _OBJMAP_LAYER
                else _FACADE_IMAGE_LAYER
            )
            state["opacity"][facade_layer] = float(value)
        return state, build_layer_rows(str(trigger["index"]), spec, state)



__all__ = [
    "build_layer_rows",
    "layout",
    "register_callbacks",
]
