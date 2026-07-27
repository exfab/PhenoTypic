"""Shared radial category-menu component for colony and QC-review tiles.

One radial implementation serves both tile surfaces. Each tile renders a
small ▾ trigger button (or a colored category badge when a category is
already assigned) anchoring a lazily-populated ``dbc.Popover``. The
popover body — an absolutely-positioned ring of wedge buttons — is only
populated on first open (following the ``_build_stack_popover`` pattern in
``colony_view/_grid.py``), keeping the DOM light across grids of many tiles.

Pattern-matched id factories include a ``surface`` key so the colony-view
callbacks (``"colony-cat-wedge"`` type) and QC-review callbacks
(``"qc-cat-wedge"`` type) never collide.

Module-level constant
---------------------
``RADIAL_RESTORE_SENTINEL``
    The special ``category`` value placed on the center "restore / close"
    wedge. The wedge-click callback checks for it to call
    ``CurationLabels.unmark`` rather than ``mark``.

A wedge click maps to one ``CurationLabels.mark(image_file, label, category)``.
Category colors come from :func:`phenotypic.gui._design.category_color` (core
categories = fixed Okabe-Ito slots; custom = cycled palette + a
``radial-badge--custom`` class). Colony-view wiring lives in
``colony_view/_callbacks.py``; QC-review wiring in ``_qc_tab/review/_callbacks.py``.
"""

from __future__ import annotations

import math
from typing import Any

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html
from dash.development.base_component import Component

from phenotypic.gui._design import category_color
from phenotypic.schema import ErrorCategory

# ---------------------------------------------------------------------------
# Sentinel constant
# ---------------------------------------------------------------------------

#: Category value for the center "restore / close" wedge.  The mark callback
#: in Task 4 checks ``cat == RADIAL_RESTORE_SENTINEL`` to call ``unmark``.
RADIAL_RESTORE_SENTINEL: str = "__restore__"

#: Category value for the ``Custom ▸`` folder wedge.  It is a UI affordance
#: (it expands the custom-category section), not a real category, so the mark
#: callbacks short-circuit on ``cat == RADIAL_CUSTOM_FOLDER_SENTINEL`` and
#: never mark a colony with it.
RADIAL_CUSTOM_FOLDER_SENTINEL: str = "__custom_folder__"

# ---------------------------------------------------------------------------
# Id factories (pattern-matched dict ids)
# ---------------------------------------------------------------------------


def radial_wedge_id(
    surface: str, image_file: str, label: int, category: str
) -> dict[str, Any]:
    """Return the pattern-matched Dash id for a radial wedge button.

    Args:
        surface: Tile surface, either ``"colony"`` or ``"qc"``, so the two
            tabs' callbacks don't collide.
        image_file: ``Metadata_ImageName`` value identifying the plate image.
        label: ``Object_Label`` integer identifying the colony.
        category: Category token string (e.g. ``"debris"``) or
            :data:`RADIAL_RESTORE_SENTINEL` for the center restore wedge.

    Returns:
        A ``{"type": "<surface>-cat-wedge", ...}`` dict id for pattern
        matching in Dash callbacks.
    """
    return {
        "type": f"{surface}-cat-wedge",
        "image_file": image_file,
        "label": label,
        "category": category,
    }


def radial_trigger_id(
    surface: str, image_file: str, label: int
) -> dict[str, Any]:
    """Return the pattern-matched id for the per-tile radial trigger button.

    Args:
        surface: Tile surface, either ``"colony"`` or ``"qc"``.
        image_file: ``Metadata_ImageName`` of the colony.
        label: ``Object_Label`` of the colony.

    Returns:
        A ``{"type": "<surface>-radial-trigger", ...}`` dict id.
    """
    return {
        "type": f"{surface}-radial-trigger",
        "image_file": image_file,
        "label": label,
    }


def radial_popover_body_id(
    surface: str, image_file: str, label: int
) -> dict[str, Any]:
    """Return the pattern-matched id for the radial popover body container.

    The popover body starts empty; a lazy-populate callback (Task 4) fills
    it on first open via :func:`build_radial_body`.

    Args:
        surface: Tile surface, either ``"colony"`` or ``"qc"``.
        image_file: ``Metadata_ImageName`` of the colony.
        label: ``Object_Label`` of the colony.

    Returns:
        A ``{"type": "<surface>-radial-popover-body", ...}`` dict id.
    """
    return {
        "type": f"{surface}-radial-popover-body",
        "image_file": image_file,
        "label": label,
    }


def radial_store_id(
    surface: str, image_file: str, label: int
) -> dict[str, Any]:
    """Return the pattern-matched id for the co-located radial data store.

    The store carries ``{image_file, label, surface}`` so the lazy-populate
    callback can resolve the colony and surface without additional context.

    Args:
        surface: Tile surface, either ``"colony"`` or ``"qc"``.
        image_file: ``Metadata_ImageName`` of the colony.
        label: ``Object_Label`` of the colony.

    Returns:
        A ``{"type": "<surface>-radial-store", ...}`` dict id.
    """
    return {
        "type": f"{surface}-radial-store",
        "image_file": image_file,
        "label": label,
    }


def radial_custom_input_id(
    surface: str, image_file: str, label: int
) -> dict[str, Any]:
    """Return the pattern-matched id for the radial's custom-category text input.

    The ``＋ Add custom`` affordance in the Custom folder section of the
    radial body (Task 7). The submit callback reads this input's value to
    register a new custom category.

    Args:
        surface: Tile surface, either ``"colony"`` or ``"qc"``.
        image_file: ``Metadata_ImageName`` of the colony.
        label: ``Object_Label`` of the colony.

    Returns:
        A ``{"type": "<surface>-radial-custom-input", ...}`` dict id.
    """
    return {
        "type": f"{surface}-radial-custom-input",
        "image_file": image_file,
        "label": label,
    }


def radial_custom_submit_id(
    surface: str, image_file: str, label: int
) -> dict[str, Any]:
    """Return the pattern-matched id for the radial's custom-category submit button.

    Args:
        surface: Tile surface, either ``"colony"`` or ``"qc"``.
        image_file: ``Metadata_ImageName`` of the colony.
        label: ``Object_Label`` of the colony.

    Returns:
        A ``{"type": "<surface>-radial-custom-submit", ...}`` dict id.
    """
    return {
        "type": f"{surface}-radial-custom-submit",
        "image_file": image_file,
        "label": label,
    }


def radial_custom_msg_id(
    surface: str, image_file: str, label: int
) -> dict[str, Any]:
    """Return the pattern-matched id for the radial's custom-category message slot.

    Inline area beneath the ``＋ Add custom`` input where the submit callback
    surfaces a validation error (empty / collision) or a success hint.

    Args:
        surface: Tile surface, either ``"colony"`` or ``"qc"``.
        image_file: ``Metadata_ImageName`` of the colony.
        label: ``Object_Label`` of the colony.

    Returns:
        A ``{"type": "<surface>-radial-custom-msg", ...}`` dict id.
    """
    return {
        "type": f"{surface}-radial-custom-msg",
        "image_file": image_file,
        "label": label,
    }


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

#: Default radius in pixels for the wedge-button ring.
_DEFAULT_RADIUS: int = 60

#: Side length of each wedge button in pixels.
_WEDGE_SIZE: int = 36

#: Side length of the center restore/close node in pixels.
_CENTER_SIZE: int = 32

#: Total container side length containing the ring (diameter + button size).
_CONTAINER_SIZE: int = _DEFAULT_RADIUS * 2 + _WEDGE_SIZE


def _wedge_positions(
    n: int, radius: int = _DEFAULT_RADIUS
) -> list[tuple[float, float]]:
    """Compute absolute (left, top) pixel positions for ``n`` wedges on a circle.

    Wedges are placed evenly around a circle of the given ``radius``, starting
    at the top (12 o'clock) and proceeding clockwise.  Coordinates are the
    top-left corner of each wedge button, centered on the circle's
    circumference point.

    The container is assumed to be a square of side
    ``2 * radius + button_size`` pixels.  The center of the container is at
    ``(radius + button_size/2, radius + button_size/2)``.

    Args:
        n: Number of wedge positions to compute.  Must be positive.
        radius: Radius of the circle in pixels.

    Returns:
        A list of ``(left_px, top_px)`` float tuples, one per wedge, in
        clockwise order starting from the top.  The values are the
        CSS ``left`` and ``top`` values for absolutely-positioned elements
        inside the ring container.

    Examples:
        >>> positions = _wedge_positions(4, 60)
        >>> len(positions)
        4
        >>> # Top wedge should be near the top-center.
        >>> abs(positions[0][0] - 60.0) < 1  # left ≈ center
        True
        >>> abs(positions[0][1]) < 2  # top ≈ 0
        True
    """
    button_half = _WEDGE_SIZE / 2
    center = radius + button_half
    positions: list[tuple[float, float]] = []
    for i in range(n):
        angle = math.radians(-90 + i * 360 / n)  # start at top, clockwise
        cx = center + radius * math.cos(angle)
        cy = center + radius * math.sin(angle)
        positions.append((cx - button_half, cy - button_half))
    return positions


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

#: Tokens for the core categories excluding OTHER (which has its own wedge slot).
_CORE_TOKENS: list[str] = [
    m.label for m in ErrorCategory if m.label != "other"
]

#: Short display labels for the long segmentation tokens so the wedge text fits
#: inside the small circular button. The persisted category **token** is
#: unchanged (still ``"oversegmented"`` / ``"undersegmented"``) — this only
#: affects the rendered wedge/badge text.
_DISPLAY_LABELS: dict[str, str] = {
    "oversegmented": "OverS",
    "undersegmented": "UnderS",
}


def _wedge_label(token: str) -> str:
    """Return the short display label for a category token.

    Args:
        token: The bare category token (e.g. ``"oversegmented"``).

    Returns:
        The short label (``"OverS"``/``"UnderS"``) for the long segmentation
        tokens, else the token with underscores rendered as spaces.
    """
    return _DISPLAY_LABELS.get(token, token.replace("_", " "))


def build_radial_trigger(
    surface: str,
    image_file: str,
    label: int,
    current_category: str | None = None,
    is_custom: bool = False,
    disabled: bool = False,
) -> list[Component]:
    """Build the per-tile radial trigger + empty popover + data store.

    Returns a list of three siblings — the trigger button, the empty
    ``dbc.Popover``, and a ``dcc.Store`` — to be spliced into the tile's
    ``extra_children`` or ``remove_button`` slot via
    :func:`~phenotypic.gui._shared.tiles.build_tile_cell`.

    The popover ships with an empty ``dbc.PopoverBody`` (id
    :func:`radial_popover_body_id`).  A lazy-populate callback (Task 4)
    fills the body on first open by calling :func:`build_radial_body`.
    The ``dcc.Store`` (id :func:`radial_store_id`) carries
    ``{image_file, label, surface}`` for that callback.

    When ``current_category`` is set the trigger renders as a **colored
    category badge** (via :func:`~phenotypic.gui._design.category_color`).
    When ``is_custom`` is ``True`` the badge gains the
    ``radial-badge--custom`` CSS modifier so it never reads identically
    to a core-category badge (decision D).

    Args:
        surface: Tile surface — ``"colony"`` or ``"qc"``.
        image_file: ``Metadata_ImageName`` of the colony.
        label: ``Object_Label`` integer.
        current_category: The currently assigned category token, or
            ``None`` when no category has been set.  Controls badge vs.
            neutral ▾ rendering.
        is_custom: ``True`` when ``current_category`` is a user-defined
            custom category (not a member of :class:`~phenotypic.schema.ErrorCategory`).
            Adds ``radial-badge--custom`` to the badge class list.

    Returns:
        A three-element list ``[trigger_button, popover, store]`` ready to
        splice into a tile's component tree.

    Examples:
        >>> from phenotypic.gui._shared._radial import build_radial_trigger
        >>> components = build_radial_trigger("colony", "plate_A.tif", 3)
        >>> len(components)
        3
    """
    trigger_id = radial_trigger_id(surface, image_file, label)
    body_id = radial_popover_body_id(surface, image_file, label)
    store_id = radial_store_id(surface, image_file, label)

    # Build the trigger button.
    if current_category is not None:
        color = category_color(current_category)
        classes = ["radial-badge"]
        if is_custom:
            classes.append("radial-badge--custom")
        trigger_button = dbc.Button(
            _wedge_label(current_category),
            id=trigger_id,
            title=current_category,
            className=" ".join(classes),
            style={
                "backgroundColor": color,
                "borderColor": color,
                "color": "#fff",
            },
            size="sm",
            n_clicks=0,
            disabled=disabled,
        )
    else:
        trigger_button = dbc.Button(
            "▾",
            id=trigger_id,
            className="radial-badge radial-badge--neutral",
            style={},
            size="sm",
            n_clicks=0,
            disabled=disabled,
        )

    popover = dbc.Popover(
        dbc.PopoverBody(
            [],
            id=body_id,
        ),
        target=trigger_id,
        trigger="legacy",
        placement="right",
        hide_arrow=True,
        style={"zIndex": "1090"},
        className="radial-popover",
    )

    store = dcc.Store(
        id=store_id,
        data={
            "image_file": image_file,
            "label": label,
            "surface": surface,
        },
    )

    return [trigger_button, popover, store]


def build_radial_body(
    surface: str,
    image_file: str,
    label: int,
    custom_categories: list[str],
    current_category: str | None = None,
) -> Component:
    """Build the wedge layout for the radial popover body.

    Produces absolutely-positioned wedge buttons arranged in a ring around a
    center restore/close node.  Layout:

    * One colored wedge per core :class:`~phenotypic.schema.ErrorCategory`
      token (excluding ``"other"``).
    * A grey ``Other`` wedge.
    * A ``Custom ▸`` folder wedge (expands to custom categories in Task 7).
    * A center restore/close node (``category = RADIAL_RESTORE_SENTINEL``).

    All primary wedges are capped at 7 to keep the ring readable.

    The wedge ring is rendered inside a fixed-size square container using
    absolute positioning derived from :func:`_wedge_positions`.  The caller
    (lazy-populate callback, Task 4) drops this component into the
    ``dbc.PopoverBody`` identified by :func:`radial_popover_body_id`.

    Args:
        surface: Tile surface — ``"colony"`` or ``"qc"``.
        image_file: ``Metadata_ImageName`` of the colony.
        label: ``Object_Label`` integer.
        custom_categories: List of registered custom category tokens (e.g.
            ``["halo", "ghost"]``).  Each renders as a chip in the expanded
            Custom folder section beneath the ring (with the
            ``radial-badge--custom`` discriminator), alongside an
            ``＋ Add custom`` input that registers a new one.  The core ring
            itself is built from the fixed :data:`ErrorCategory` tokens — no
            active-category list is needed.
        current_category: Currently assigned category, or ``None``.  The
            matching wedge is highlighted with an ``active`` modifier.

    Returns:
        An ``html.Div`` container carrying all wedge and center-node
        components.  Drop it directly into a ``dbc.PopoverBody``.

    Examples:
        >>> from phenotypic.gui._shared._radial import build_radial_body
        >>> body = build_radial_body("colony", "p.tif", 1, [])
        >>> body.className
        'radial-body-wrap'
    """
    # Primary wedge tokens: core (excluding other) + other + custom-folder.
    # Max 7 primary wedges per plan.
    primary_tokens: list[str] = list(_CORE_TOKENS)  # up to 5 core tokens
    primary_tokens.append("other")  # the catch-all slot
    primary_tokens.append(RADIAL_CUSTOM_FOLDER_SENTINEL)  # folder placeholder

    # Clamp to 7 primaries.
    primary_tokens = primary_tokens[:7]

    n = len(primary_tokens)
    positions = _wedge_positions(n, _DEFAULT_RADIUS)
    container_size = _CONTAINER_SIZE
    center_offset = container_size / 2 - _CENTER_SIZE / 2

    wedges: list[Component] = []
    for i, token in enumerate(primary_tokens):
        left, top = positions[i]
        is_active = token == current_category
        is_custom_folder = token == RADIAL_CUSTOM_FOLDER_SENTINEL

        if is_custom_folder:
            color = "#888888"
            label_text = "Custom ▸"
            btn_id = radial_wedge_id(
                surface, image_file, label, RADIAL_CUSTOM_FOLDER_SENTINEL
            )
            classes = "radial-wedge radial-wedge--folder"
        else:
            color = category_color(token)
            label_text = _wedge_label(token)
            btn_id = radial_wedge_id(surface, image_file, label, token)
            classes = "radial-wedge"
            if is_active:
                classes += " radial-wedge--active"

        wedges.append(
            dbc.Button(
                label_text,
                id=btn_id,
                className=classes,
                style={
                    "position": "absolute",
                    "left": f"{left:.1f}px",
                    "top": f"{top:.1f}px",
                    "width": f"{_WEDGE_SIZE}px",
                    "height": f"{_WEDGE_SIZE}px",
                    # Inline 50% beats Bootstrap's ``.btn-sm`` radius so the
                    # wedge is a true circle.
                    "borderRadius": "50%",
                    "backgroundColor": color,
                    "borderColor": color,
                    "color": "#fff",
                    "fontSize": "0.55rem",
                    "lineHeight": "1.1",
                    "padding": "2px",
                    "fontWeight": "600",
                },
                title=label_text,
                size="sm",
                n_clicks=0,
            )
        )

    # Center restore/close node.
    center_node = dbc.Button(
        "✕",
        id=radial_wedge_id(
            surface, image_file, label, RADIAL_RESTORE_SENTINEL
        ),
        className="radial-center",
        style={
            "position": "absolute",
            "left": f"{center_offset:.1f}px",
            "top": f"{center_offset:.1f}px",
            "width": f"{_CENTER_SIZE}px",
            "height": f"{_CENTER_SIZE}px",
            "borderRadius": "50%",
            "backgroundColor": "var(--color-surface, #f8f9fa)",
            "borderColor": "var(--color-border, #dee2e6)",
            "color": "var(--color-body, #333)",
            "fontSize": "0.75rem",
            "padding": "2px",
        },
        title="Restore (clear category)",
        size="sm",
        n_clicks=0,
    )

    ring = html.Div(
        children=wedges + [center_node],
        className="radial-ring-container",
        style={
            "position": "relative",
            "width": f"{container_size}px",
            "height": f"{container_size}px",
        },
    )

    custom_section = _build_custom_section(
        surface, image_file, label, custom_categories, current_category
    )

    return html.Div(
        [ring, custom_section],
        className="radial-body-wrap",
    )


def _build_custom_section(
    surface: str,
    image_file: str,
    label: int,
    custom_categories: list[str],
    current_category: str | None,
) -> Component:
    """Build the expanded Custom folder: custom wedges + an ＋ Add affordance.

    Renders, beneath the core ring:

    * one chip per registered custom category token, colored by
      :func:`~phenotypic.gui._design.category_color` (cycling the custom
      palette by registration index) and carrying the
      ``radial-badge--custom`` discriminator (decision D) so a custom chip
      never reads identically to a core badge;
    * an inline ``dcc.Input`` + confirm button (``＋ Add custom``) plus an
      empty message slot the submit callback (Task 7) fills with a validation
      error or success hint.

    Each custom chip is a wedge id (``radial_wedge_id``) so clicking it marks
    the colony with that custom category through the SAME mark callback as the
    core wedges.

    Args:
        surface: Tile surface — ``"colony"`` or ``"qc"``.
        image_file: ``Metadata_ImageName`` of the colony.
        label: ``Object_Label`` integer.
        custom_categories: Registered custom category tokens, in registration
            order (the order drives the custom-palette color cycle).
        current_category: Currently assigned category, or ``None`` — the
            matching custom chip gets the ``radial-wedge--active`` modifier.

    Returns:
        An ``html.Div`` for the custom folder section.
    """
    chips: list[Component] = []
    for index, token in enumerate(custom_categories):
        color = category_color(token, custom_index=index)
        classes = "radial-custom-chip radial-badge--custom"
        if token == current_category:
            classes += " radial-wedge--active"
        chips.append(
            dbc.Button(
                token.replace("_", " "),
                id=radial_wedge_id(surface, image_file, label, token),
                className=classes,
                style={
                    "backgroundColor": color,
                    "borderColor": color,
                    "color": "#fff",
                    "fontSize": "0.6rem",
                    "padding": "1px 6px",
                    "margin": "2px",
                },
                title=token,
                size="sm",
                n_clicks=0,
            )
        )

    add_row = html.Div(
        [
            dcc.Input(
                id=radial_custom_input_id(surface, image_file, label),
                type="text",
                placeholder="New category…",
                debounce=True,
                className="radial-custom-input",
                style={"fontSize": "0.7rem", "width": "8rem"},
            ),
            dbc.Button(
                "＋ Add",
                id=radial_custom_submit_id(surface, image_file, label),
                className="radial-custom-submit",
                color="secondary",
                outline=True,
                size="sm",
                n_clicks=0,
            ),
        ],
        className="radial-custom-add-row",
        style={
            "display": "flex",
            "gap": "0.25rem",
            "alignItems": "center",
            "marginTop": "0.25rem",
        },
    )

    msg = html.Div(
        "",
        id=radial_custom_msg_id(surface, image_file, label),
        className="radial-custom-msg",
        style={"fontSize": "0.6rem", "minHeight": "0.9rem"},
    )

    return html.Div(
        [
            html.Div(
                "Custom",
                className="radial-custom-header",
                style={
                    "fontSize": "0.6rem",
                    "fontWeight": 600,
                    "color": "var(--color-muted, #888)",
                    "marginTop": "0.35rem",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.06em",
                },
            ),
            html.Div(
                chips,
                className="radial-custom-chips",
                style={"display": "flex", "flexWrap": "wrap"},
            ),
            add_row,
            msg,
        ],
        className="radial-custom-section",
        style={
            "borderTop": "1px solid var(--color-border, #dee2e6)",
            "marginTop": "0.4rem",
            "paddingTop": "0.25rem",
        },
    )
