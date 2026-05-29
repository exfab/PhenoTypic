"""Shared tile-imaging primitives for every GUI tab that renders colony crops.

Two GUI surfaces render fixed-size, centroid-aligned crops of the
overlay PNGs the CLI writes at
``<root>/results/<dataset>/overlays/<stem>.png``:

* the **colony-view** tab — a 2D axis-header grid (see
  :mod:`phenotypic.gui.results_viewer.colony_view._grid`), and
* the **QC review** tab — a flat, faceted gallery (Phase D).

Rather than duplicating the cropper, the path-traversal guard, the
crop-serving Flask route, and the per-tile chrome across both, this
module owns the single implementation and both consumers import it:

* :func:`crop_overlay` / :func:`_load_overlay_rgb` — slice a square crop
  out of an overlay PNG, padded so the result is always ``size`` x
  ``size``, with a small LRU cache over the decoded source.
* :func:`is_safe_path_component` — the URL-capture path-traversal guard.
* :func:`register_crop_route` — a factory that mounts the centered-crop
  Flask route under any URL segment (``crops`` for the colony view, a QC
  segment later), so both tabs serve crops the same way.
* :func:`build_tile_cell` — the per-tile chrome (img/placeholder,
  multi-select checkbox, remove/restore button) — **the dedup boundary**.
* :func:`build_tile_grid` — a flat gallery of tiles plus the row-major
  key list both tabs reuse for shift+click range selection.
* :func:`expand_range` — resolve a shift+click slice over a row-major key
  list (re-exported from ``colony_view/_grid.py`` for back-compat).
"""

from __future__ import annotations

import functools
import io
import logging
import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dash
from dash import html
from dash.development.base_component import Component
from flask import Blueprint, Response
from PIL import Image as PILImage

if TYPE_CHECKING:
    from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Overlay cropping
# ---------------------------------------------------------------------------

#: Number of distinct overlay PNGs to keep decoded in memory. A typical
#: grid pulls from 1–8 plates per render, so this comfortably covers a
#: full grid without holding more than a few hundred MB of pixel data.
_OVERLAY_CACHE_SIZE = 8


@functools.lru_cache(maxsize=_OVERLAY_CACHE_SIZE)
def _load_overlay_rgb(path: str, mtime_ns: int) -> PILImage.Image:
    """Decode an overlay PNG to RGB and cache the result.

    Args:
        path: Absolute path to the overlay PNG, as a string so the
            cache key is hashable.
        mtime_ns: ``st_mtime_ns`` at lookup time. Including it in the
            cache key invalidates the cached frame when the overlay is
            regenerated under a running viewer.
    """
    del mtime_ns  # Cache-key only.
    with PILImage.open(path) as img:
        return img.convert("RGB")


def crop_overlay(
    png_path: Path,
    center_rr: float,
    center_cc: float,
    size: int,
    pad_value: tuple[int, int, int] = (0, 0, 0),
) -> bytes:
    """Crop an overlay PNG to a fixed ``size`` x ``size`` window centered on a colony.

    Computes ``(top, left) = (round(center_rr) - size // 2,
    round(center_cc) - size // 2)``, clamps the requested window to the
    image bounds, and pastes the clamped region onto a freshly-allocated
    canvas filled with ``pad_value``. The output therefore always has
    the exact requested dimensions, even when the colony sits near an
    edge of the source image.

    Args:
        png_path: Path to the overlay PNG written by the CLI (typically
            ``<root>/results/<dataset>/overlays/<stem>.png``). RGB and
            RGBA sources are both accepted; the result is always RGB.
        center_rr: Row coordinate (Y) of the colony centroid, in image
            pixels. Read from ``master_measurements.parquet``.
        center_cc: Column coordinate (X) of the colony centroid, in
            image pixels.
        size: Side length of the square crop, in pixels. Must be
            positive.
        pad_value: RGB fill colour used for any portion of the crop that
            falls outside the source image. Defaults to black.

    Returns:
        PNG-encoded bytes of the ``size`` x ``size`` crop in RGB mode.

    """
    # TODO(future): mirror this with crop_hdf_rgb(h5_path, ...) that
    # loads the raw RGB layer via Image.load_hdf5 (see
    # src/phenotypic/_core/_image_parts/_image_io_handler.py:944) for
    # overlay-free crops.
    mtime_ns = os.stat(png_path).st_mtime_ns
    source = _load_overlay_rgb(str(png_path), mtime_ns)

    src_width, src_height = source.size

    half = size // 2
    left_unclamped = round(center_cc) - half
    top_unclamped = round(center_rr) - half
    right_unclamped = left_unclamped + size
    bottom_unclamped = top_unclamped + size

    left_clamped = max(0, left_unclamped)
    top_clamped = max(0, top_unclamped)
    right_clamped = min(src_width, right_unclamped)
    bottom_clamped = min(src_height, bottom_unclamped)

    result = PILImage.new("RGB", (size, size), pad_value)

    # If the clamped window has positive area, paste it onto the padded
    # canvas at the offset that re-aligns it with the unclamped origin.
    if right_clamped > left_clamped and bottom_clamped > top_clamped:
        region = source.crop(
            (left_clamped, top_clamped, right_clamped, bottom_clamped)
        )
        paste_x = max(0, -left_unclamped)
        paste_y = max(0, -top_unclamped)
        result.paste(region, (paste_x, paste_y))

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Path-traversal guard
# ---------------------------------------------------------------------------

#: Allow only filesystem-safe identifiers (alphanumeric, dot, underscore,
#: dash) — same character class :func:`werkzeug.utils.secure_filename` is
#: comfortable with, but applied before any path math.
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def is_safe_path_component(name: str) -> bool:
    """Return ``True`` if ``name`` is safe to use as a single path component.

    Validates both the ``<dataset>`` and ``<stem>`` URL captures before
    feeding them into filesystem paths.

    Args:
        name: Candidate identifier from the URL.

    Returns:
        ``True`` only if ``name`` is non-empty, contains no path
        separators or parent-directory tokens, does not start with a
        leading dot, and matches ``[A-Za-z0-9._-]+``.
    """
    if not name or name.startswith("."):
        return False
    if "/" in name or "\\" in name or ".." in name:
        return False
    return bool(_NAME_RE.match(name))


# ---------------------------------------------------------------------------
# Crop-serving Flask route factory
# ---------------------------------------------------------------------------

#: Lower bound on the ``?size=`` query parameter. Smaller crops would not
#: hold a useful colony preview; rejecting them early avoids confused
#: callers.
_MIN_CROP_SIZE = 16

#: Upper bound on the ``?size=`` query parameter. Anything larger is
#: almost certainly the result of a bug in the caller (the colony-view
#: grid picks crops on the order of 64-512 px); cap it to avoid a 4k+
#: PNG allocation per request.
_MAX_CROP_SIZE = 4096

#: Sanity ceiling on the parsed ``<label>`` URL component. Real
#: ``Object_Label`` values are dense small integers; anything beyond a
#: billion is almost certainly malformed input.
_MAX_OBJECT_LABEL = 10**9


def register_crop_route(
    app: dash.Dash, output_root: OutputRoot, segment: str
) -> None:
    """Mount a per-colony centered-crop route under ``/<segment>`` on ``app.server``.

    Exposes one route under the ``/<segment>`` URL prefix:

    * ``GET /<segment>/<dataset>/<stem>/<label>.png?size=<int>`` — returns
      a PNG-encoded ``size`` x ``size`` crop of the dataset's overlay PNG
      centered on the colony with ``Object_Label == label`` in image
      ``stem``.

    The route never mutates state and never writes to disk; the on-disk
    cache used by the DZI tile route is intentionally not reused here
    because per-colony crops are tiny and inexpensive to recompute.

    Both the colony-view tab (``segment="crops"``) and the QC review tab
    mount this factory so they serve crops identically; the ``segment``
    keeps their blueprint names and URL namespaces distinct.

    Args:
        app: The Dash application whose Flask server should be extended.
        output_root: Validated handle on the CLI output directory.
            Captured by closure and used to resolve overlay PNGs and the
            master measurements DataFrame.
        segment: URL path segment the route mounts under (e.g. ``crops``).
            Also seeds the blueprint name so multiple segments can coexist
            on one server.
    """
    # Imported here rather than at module scope:
    #  * ``flask.request`` is a Werkzeug ``LocalProxy``; exposing it as a
    #    module attribute makes ``phenotypic.gui._shared.tiles.request``
    #    raise ``RuntimeError: Working outside of request context`` the
    #    moment anything reads ``__name__`` off it — which the public
    #    namespace walk in ``tests/unit/test_pickleable.py`` does during
    #    parametrize-ID generation, aborting collection of all of
    #    ``tests/unit``. Keeping it function-local hides it from that walk.
    #  * the ``results_viewer`` submodules would otherwise form an import
    #    cycle (``results_viewer.__init__`` eagerly imports ``_app`` →
    #    ``_tile_routes`` → this module for the path guard). They are only
    #    needed once a route is actually mounted, well after both packages
    #    have finished importing.
    import polars as pl
    from flask import request

    from phenotypic.gui.results_viewer._filtered_state import (
        KEY_IMAGE_FILE,
        KEY_OBJECT_LABEL,
    )

    bp = Blueprint(
        f"results_viewer_crops_{segment}", __name__, url_prefix=f"/{segment}"
    )

    @bp.route("/<dataset>/<stem>/<label>.png")
    def crop_endpoint(dataset: str, stem: str, label: str) -> Response | tuple[str, int]:
        """Serve a single PNG colony crop for ``(dataset, stem, label)``."""
        # --- 1. Path-component validation --------------------------------
        if not is_safe_path_component(dataset) or not is_safe_path_component(stem):
            logger.warning(
                "Rejected crop request with unsafe identifiers: "
                "dataset=%r stem=%r",
                dataset,
                stem,
            )
            return ("bad request: invalid dataset or stem", 400)

        # --- 2. Label parsing --------------------------------------------
        try:
            label_int = int(label)
        except (TypeError, ValueError):
            logger.warning("Rejected crop request with non-numeric label: %r", label)
            return ("bad request: label must be an integer", 400)
        if label_int < 0 or label_int > _MAX_OBJECT_LABEL:
            logger.warning("Rejected crop request with out-of-range label: %d", label_int)
            return ("bad request: label out of range", 400)

        # --- 3. Size parsing ---------------------------------------------
        size = request.args.get("size", type=int)
        if size is None:
            return ("bad request: missing required ?size=<int>", 400)
        if size < _MIN_CROP_SIZE or size > _MAX_CROP_SIZE:
            return (
                f"bad request: size must be between {_MIN_CROP_SIZE} and "
                f"{_MAX_CROP_SIZE} (got {size})",
                400,
            )

        # --- 4. Lookup ----------------------------------------------------
        # Cast key columns explicitly so the comparison still matches when
        # the master frame stores Metadata_ImageFile as Categorical or
        # Object_Label as a narrower int type.
        try:
            row = (
                output_root.master_df.filter(
                    (pl.col(KEY_IMAGE_FILE).cast(pl.String) == stem)
                    & (pl.col(KEY_OBJECT_LABEL).cast(pl.Int64) == label_int)
                )
                .select(["Bbox_CenterRR", "Bbox_CenterCC"])
                .head(1)
            )
        except Exception:
            logger.exception(
                "Master DataFrame lookup failed for dataset=%s stem=%s label=%d",
                dataset,
                stem,
                label_int,
            )
            return ("internal error: master measurements lookup failed", 500)

        if row.is_empty():
            return (
                f"not found: no row for stem={stem!r} label={label_int}",
                404,
            )

        center_rr = float(row.get_column("Bbox_CenterRR")[0])
        center_cc = float(row.get_column("Bbox_CenterCC")[0])

        # --- 5. Overlay path ---------------------------------------------
        if not output_root.has_overlay(dataset, stem):
            return (
                f"not found: overlay not found for {dataset!r}/{stem!r}",
                404,
            )
        overlay_png = output_root.overlay_path(dataset, stem)

        # --- 6. Crop ------------------------------------------------------
        try:
            png_bytes = crop_overlay(overlay_png, center_rr, center_cc, size)
        except Exception:
            logger.exception(
                "Crop generation failed for dataset=%s stem=%s label=%d size=%d",
                dataset,
                stem,
                label_int,
                size,
            )
            return ("internal error: crop generation failed", 500)

        # --- 7. Response --------------------------------------------------
        response = Response(png_bytes, mimetype="image/png")
        response.headers["Cache-Control"] = "no-cache"
        return response

    app.server.register_blueprint(bp)
    logger.debug(
        "Registered results viewer crop routes under /%s for root=%s",
        segment,
        output_root.root,
    )


# ---------------------------------------------------------------------------
# Per-tile chrome — the dedup boundary
# ---------------------------------------------------------------------------


def build_tile_cell(
    *,
    image_file: str,
    label: int,
    dataset: str,
    crop_size: int,
    display_size: int,
    has_overlay: bool,
    is_removed: bool,
    is_selected: bool,
    url_builder: Callable[[str, str, int, int], str],
    remove_button: Component,
    extra_children: Iterable[Component] | None = None,
    outer_height: int | None = None,
) -> Component:
    """Render the chrome + crop for a single tile — shared across tabs.

    Builds the framed ``display_size`` x ``display_size`` card carrying

    * the crop ``<img>`` (or a striped placeholder when the overlay is
      missing),
    * a CSS-styled multi-select checkbox carrying a
      ``data-key="<image_file>::<label>"`` attribute (the JS layer reads
      it to drive shift+click selection), and
    * the caller-supplied ``remove_button`` (each tab owns its own
      pattern-matched id so the remove/restore callback resolves
      correctly).

    Any tab-specific siblings (e.g. the colony view's ``N=k`` stack badge
    and popover) are appended after the frame via ``extra_children``.

    Args:
        image_file: ``Metadata_ImageFile`` of the represented colony.
        label: ``Object_Label`` of the represented colony.
        dataset: ``Metadata_Dataset`` of the represented colony.
        crop_size: Server crop side length, in pixels — passed to
            ``url_builder`` so the PNG is generated at full resolution
            covering the colony's bbox.
        display_size: CSS render size, in pixels. The browser scales the
            ``<img>`` to this size; ``object-fit: cover`` keeps colonies
            centered without distortion.
        has_overlay: Whether the source overlay PNG exists on disk; if
            not, a striped placeholder is rendered instead of an ``<img>``.
        is_removed: Whether the colony is in the curated removal set.
            Dims the crop and toggles the ``is-removed`` modifier.
        is_selected: Whether the tile is in the active multi-select.
            Toggles the ``is-selected`` modifier + the checkbox's
            ``is-checked`` state.
        url_builder: Callable ``(dataset, image_file, label, crop_size) ->
            str`` returning the crop ``<img>`` src. Each tab supplies a
            builder bound to its own crop-route segment + url prefix.
        remove_button: The tab's remove/restore ``dbc.Button`` (already
            built with the tab's pattern-matched id and removed/active
            styling).
        extra_children: Optional siblings appended after the frame inside
            the outer cell ``<div>`` (e.g. a stack badge + popover).
        outer_height: Outer cell height, in pixels. Defaults to
            ``display_size`` (no extra vertical room). Pass a larger value
            to reserve space for a sibling that peeks out below the frame.

    Returns:
        A component ready to drop into a tile grid or gallery container.
    """
    classes = ["colony-cell"]
    if is_selected:
        classes.append("is-selected")
    if is_removed:
        classes.append("is-removed")

    if has_overlay:
        crop_url = url_builder(dataset, image_file, label, crop_size)
        crop_node: Component = html.Img(
            src=crop_url,
            className="colony-cell-img",
            style={
                "width": f"{display_size}px",
                "height": f"{display_size}px",
                "display": "block",
                "opacity": "0.3" if is_removed else "1",
                "objectFit": "cover",
            },
        )
    else:
        crop_node = html.Div(
            className="colony-cell-placeholder",
            style={
                "width": f"{display_size}px",
                "height": f"{display_size}px",
                "backgroundImage": (
                    "repeating-linear-gradient(45deg, "
                    "rgba(0,54,96,0.05) 0px, rgba(0,54,96,0.05) 8px, "
                    "rgba(0,54,96,0.10) 8px, rgba(0,54,96,0.10) 16px)"
                ),
                "border": "1px dashed rgba(0,54,96,0.25)",
            },
        )

    # CSS-styled span playing the role of a checkbox. We don't use a real
    # <input> because Dash 4 doesn't expose html.Input, and the JS layer
    # only needs `data-key` + the class name to wire up the click event.
    # Visual checked state is driven by the `is-checked` modifier class.
    checkbox_class = "colony-cell-checkbox"
    if is_selected:
        checkbox_class += " is-checked"
    # `data-*` HTML attributes can't be expressed in Dash's typed Span
    # kwargs, so unpack via an Any-typed dict to bypass the stub mismatch.
    extra_props: dict[str, Any] = {"data-key": f"{image_file}::{label}"}
    checkbox_inner = html.Span(
        "",
        className=checkbox_class,
        **extra_props,
    )
    checkbox = html.Span(
        checkbox_inner,
        className="colony-cell-checkbox-wrap",
        style={
            "position": "absolute",
            "top": "4px",
            "left": "4px",
            "zIndex": "2",
        },
    )

    # Image card: the framed display_size×display_size area carrying the
    # crop, checkbox, and remove button. Sits in front of any sibling
    # (e.g. the stack tab) via z-index so the sibling can peek out from
    # beneath the bottom edge.
    frame = html.Div(
        [crop_node, checkbox, remove_button],
        className="colony-cell-frame",
    )

    children: list[Component] = [frame]
    if extra_children is not None:
        children.extend(extra_children)

    if outer_height is None:
        outer_height = display_size

    return html.Div(
        children,
        className=" ".join(classes),
        style={
            "position": "relative",
            "width": f"{display_size}px",
            "height": f"{outer_height}px",
            "overflow": "visible",
        },
    )


# ---------------------------------------------------------------------------
# Flat tile gallery
# ---------------------------------------------------------------------------

#: Default CSS gap between tiles in a flat gallery, in pixels.
_GALLERY_GAP_PX = 8


def build_tile_grid(
    keys: list[tuple[str, str, int]],
    url_builder: Callable[[str, str, int, int], str],
    *,
    selected: set[tuple[str, int]],
    removed: set[tuple[str, int]],
    crop_size: int,
    display_size: int,
    has_overlay: Callable[[str, str], bool],
    remove_button_builder: Callable[[str, int, bool], Component],
    gap_px: int = _GALLERY_GAP_PX,
) -> tuple[Component, list[tuple[str, int]]]:
    """Render a flat gallery of tiles and its row-major key order.

    Unlike :func:`phenotypic.gui.results_viewer.colony_view._grid.build_grid`
    (a 2D axis-header table), this lays the tiles out left-to-right in a
    wrapping flex row in the iteration order of ``keys`` — the layout the
    QC review gallery uses. The returned ``grid_order`` mirrors that
    visible order so :func:`expand_range` can resolve shift+click slices.

    Args:
        keys: ``(dataset, image_file, label)`` tuples in the order tiles
            should appear.
        url_builder: Callable ``(dataset, image_file, label, crop_size) ->
            str`` returning each tile's crop ``<img>`` src.
        selected: ``(image_file, label)`` keys currently multi-selected.
        removed: ``(image_file, label)`` keys currently in the curated
            removal set.
        crop_size: Server crop side length, in pixels.
        display_size: CSS render size, in pixels, for each tile.
        has_overlay: Callable ``(dataset, image_file) -> bool`` answering
            whether the overlay PNG exists (typically
            :meth:`OutputRoot.has_overlay`).
        remove_button_builder: Callable ``(image_file, label, is_removed)
            -> Component`` returning the tile's remove/restore button so
            the gallery owner controls the button id + styling.
        gap_px: CSS gap between tiles, in pixels. Defaults to 8.

    Returns:
        A tuple ``(component, grid_order)``. ``grid_order`` is the
        row-major flat list of ``(image_file, label)`` keys in the same
        order as the rendered tiles.
    """
    children: list[Component] = []
    grid_order: list[tuple[str, int]] = []
    for dataset, image_file, label in keys:
        key = (image_file, label)
        grid_order.append(key)
        is_removed = key in removed
        children.append(
            build_tile_cell(
                image_file=image_file,
                label=label,
                dataset=dataset,
                crop_size=crop_size,
                display_size=display_size,
                has_overlay=has_overlay(dataset, image_file),
                is_removed=is_removed,
                is_selected=key in selected,
                url_builder=url_builder,
                remove_button=remove_button_builder(image_file, label, is_removed),
            )
        )

    gallery = html.Div(
        children,
        className="colony-grid tile-gallery",
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "gap": f"{gap_px}px",
            "padding": "0.5rem",
            "alignItems": "flex-start",
        },
    )
    return gallery, grid_order


# ---------------------------------------------------------------------------
# Range expansion
# ---------------------------------------------------------------------------


def expand_range(
    grid_order: list[tuple[str, int]],
    anchor: tuple[str, int],
    target: tuple[str, int],
) -> list[tuple[str, int]]:
    """Return the contiguous slice of ``grid_order`` between two keys.

    Direction-agnostic: the slice always runs from the lower of the two
    indices to the higher (inclusive) regardless of which key was
    originally clicked first.

    Args:
        grid_order: Row-major flat list of cell keys, as returned by
            :func:`build_tile_grid` or
            :func:`phenotypic.gui.results_viewer.colony_view._grid.build_grid`.
        anchor: Key of the cell that started the range (the most recent
            non-shift click).
        target: Key of the shift-clicked cell.

    Returns:
        Inclusive slice of ``grid_order`` between the two keys.

    Raises:
        ValueError: If either ``anchor`` or ``target`` is not in
            ``grid_order``.
    """
    try:
        a = grid_order.index(anchor)
    except ValueError as exc:
        raise ValueError(f"anchor {anchor!r} is not in grid_order") from exc
    try:
        b = grid_order.index(target)
    except ValueError as exc:
        raise ValueError(f"target {target!r} is not in grid_order") from exc
    lo, hi = (a, b) if a <= b else (b, a)
    return grid_order[lo : hi + 1]


__all__ = [
    "crop_overlay",
    "is_safe_path_component",
    "register_crop_route",
    "build_tile_cell",
    "build_tile_grid",
    "expand_range",
]
