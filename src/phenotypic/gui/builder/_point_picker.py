"""Modal layout, Flask blueprint, and Dash callbacks for the in-builder point picker.

Owns:

* :func:`build_point_picker_modal` — the ``dbc.Modal`` shell with stores,
  channel radio, action buttons, and the OSD mount div.
* :func:`register_point_picker_routes` — Flask blueprint mounted at
  ``/builder/tiles`` that lazily tiles per-session preview PNGs into DZI
  pyramids using :mod:`phenotypic.gui.results_viewer._dzi_tiler`.
* :func:`register_point_picker_callbacks` — Dash callbacks for modal open,
  channel toggle, clear / undo, count label, cancel, and confirm.

Click capture is delegated to the clientside JS layer at
``builder/assets/point_picker.js``, which pushes new points into
``PICKER_STAGED_STORE`` via ``dash_clientside.set_props``.

The module reuses :func:`_dzi_tiler.tile`, :data:`_TILE_NAME_RE`,
:func:`_is_safe_path_component`, and :func:`_json_error` from the results
viewer's tile-routes module so this blueprint behaves like the public
viewer's tile route — same on-disk layout, same path-traversal hardening.
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
import numpy as np
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from flask import Blueprint, Response, send_from_directory
from PIL import Image as PILImage
from werkzeug.utils import secure_filename

from phenotypic.gui._config import (
    BUILDER_TILES_PREFIX,
    CFG_IMAGE_ROOT,
    SANDBOX_BUILDER_TILES_SUBDIR,
    SANDBOX_GUI_DIRNAME,
)
from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._image_renderer import _normalize_to_uint8
from phenotypic.gui.builder._session import get_cache
from phenotypic.gui.builder._state import (
    PIPELINE_CLASS_NAME,
    current_scope,
    state_from_json,
)
from phenotypic.gui.results_viewer import _dzi_tiler
from phenotypic.gui.results_viewer._tile_routes import (
    _TILE_NAME_RE,
    _is_safe_path_component,
    _json_error,
)

logger = logging.getLogger(__name__)


_VALID_SOURCES = ("rgb", "intermediate")

#: Session ids in the URL must look UUID-shaped (lowercase hex + hyphens).
#: We reuse :func:`_is_safe_path_component` for the basic charset check and
#: layer this regex on top to reject anything obviously wrong (too short to
#: be a UUID, missing the hyphen pattern, etc.).
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SESSION_ID_MIN_LEN = 8


# ---------------------------------------------------------------------------
# Path / cache helpers
# ---------------------------------------------------------------------------


def _safe_session_id(session_id: str) -> bool:
    """Return ``True`` if *session_id* is safe to embed in a filesystem path.

    Validates two things on top of :func:`_is_safe_path_component`:

    * The id is at least :data:`_SESSION_ID_MIN_LEN` characters (UUIDs are
      36; we accept the looser bound to avoid breaking tests that pass
      shorter synthetic ids).
    * The id passes the same charset check as a tile filename.

    Args:
        session_id: Per-tab uuid pulled from :data:`STORE_SESSION_ID`.

    Returns:
        ``True`` only when *session_id* is non-empty, long enough, and
        composed of safe path-component characters.
    """
    if not session_id or len(session_id) < _SESSION_ID_MIN_LEN:
        return False
    if not _is_safe_path_component(session_id):
        return False
    return bool(_SESSION_ID_RE.match(session_id))


def _builder_cache_root(image_root: Optional[Path]) -> Path:
    """Resolve the per-builder DZI cache root directory.

    Returns ``<image_root>/.phenotypic-gui/builder_tiles/`` when
    *image_root* is provided, else ``<system tmp>/phenotypic_builder_tiles/``.
    The directory is created lazily.

    Args:
        image_root: Optional sandbox root passed to ``create_app``.

    Returns:
        Absolute path to the cache root (created if missing).
    """
    if image_root is not None:
        root = Path(image_root) / SANDBOX_GUI_DIRNAME / SANDBOX_BUILDER_TILES_SUBDIR
    else:
        root = Path(tempfile.gettempdir()) / "phenotypic_builder_tiles"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _session_cache_dir(image_root: Optional[Path], session_id: str) -> Path:
    """Return the per-session sub-directory of the builder cache root.

    Args:
        image_root: Optional sandbox root.
        session_id: Per-tab uuid (must already be validated).

    Returns:
        Absolute path to ``<root>/<session_id>``.
    """
    return _builder_cache_root(image_root) / session_id


def _channel_png_path(cache_dir: Path, source: str) -> Path:
    """Return the on-disk PNG path for a given channel source.

    Args:
        cache_dir: Per-session cache directory.
        source: One of :data:`_VALID_SOURCES`.

    Returns:
        Absolute path to ``<cache_dir>/<source>.png``.

    Raises:
        ValueError: If *source* is not a recognised channel name.
    """
    if source not in _VALID_SOURCES:
        raise ValueError(f"Unknown source: {source!r}")
    return cache_dir / f"{source}.png"


def _read_first_non_empty(image: Any, channels: Tuple[str, ...]) -> Optional[np.ndarray]:
    """Try each channel in turn; return the first that yields a non-empty array.

    Args:
        image: A :class:`phenotypic.Image` (or :class:`GridImage`).
        channels: Channel names to try in priority order.

    Returns:
        ``np.ndarray`` for the first non-empty channel, or ``None`` if
        every channel was empty / missing.
    """
    for ch in channels:
        accessor = getattr(image, ch, None)
        if accessor is None:
            continue
        # ``isempty`` is the canonical empty check on the rgb accessor;
        # gray / detect_mat don't expose it but their slice is always
        # populated once an image is loaded.
        try:
            if hasattr(accessor, "isempty") and accessor.isempty():
                continue
            arr = np.asarray(accessor[:])
        except Exception:  # noqa: BLE001 - defensive against half-loaded ops
            continue
        if arr.size == 0:
            continue
        return arr
    return None


def _dump_image_to_png(image: Any, source: str, png_path: Path) -> bool:
    """Write ``image.<source>[:]`` to *png_path* as a uint8 PNG.

    For ``source == "rgb"``: try :data:`image.rgb` first, fall back to
    :data:`image.gray` if rgb is empty.

    For ``source == "intermediate"``: try :data:`image.detect_mat`,
    falling back through gray and rgb. This mirrors the napari point
    picker's preference order from
    :file:`tools_/napari_/_point_picker_widget.py:48`.

    The array is normalised to uint8 before saving.

    Args:
        image: A :class:`phenotypic.Image` (or :class:`GridImage`).
        source: One of :data:`_VALID_SOURCES`.
        png_path: Destination PNG path. Parent dirs are created lazily.

    Returns:
        ``True`` if the PNG was written, ``False`` if every candidate
        channel was empty (in which case nothing is written).

    Raises:
        ValueError: If *source* is not a recognised channel name.
    """
    if source == "rgb":
        priority: Tuple[str, ...] = ("rgb", "gray")
    elif source == "intermediate":
        priority = ("detect_mat", "gray", "rgb")
    else:
        raise ValueError(f"Unknown source: {source!r}")

    arr = _read_first_non_empty(image, priority)
    if arr is None:
        return False

    u8 = _normalize_to_uint8(arr)
    if u8.ndim == 2:
        pil = PILImage.fromarray(u8, mode="L")
    elif u8.ndim == 3 and u8.shape[-1] == 4:
        pil = PILImage.fromarray(u8, mode="RGBA")
    elif u8.ndim == 3 and u8.shape[-1] == 3:
        pil = PILImage.fromarray(u8, mode="RGB")
    else:
        # Last-ditch: collapse extra channels into a single grayscale plane.
        pil = PILImage.fromarray(u8.reshape(u8.shape[:2]), mode="L")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(png_path, format="PNG")
    return True


# ---------------------------------------------------------------------------
# Modal layout
# ---------------------------------------------------------------------------


def build_point_picker_modal() -> dbc.Modal:
    """Return the ``dbc.Modal`` shell mounted by :mod:`builder._layout`.

    Body contains:

    * A :class:`dbc.RadioItems` for channel selection (RGB / intermediate).
    * A small help line that gets populated by the open-modal callback
      when the intermediate channel is unavailable.
    * Clear-all / Remove-last button group + count label.
    * The OSD mount ``html.Div`` (id :data:`PICKER_OSD_DIV`).
    * Hidden ``dcc.Store`` instances backing every piece of modal state.

    Footer is a Cancel + Confirm button pair.

    Returns:
        A :class:`dbc.Modal` ready to be added to the builder layout.
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Pick points on the image")),
            dbc.ModalBody(
                [
                    dbc.RadioItems(
                        id=ids.PICKER_CHANNEL_RADIO,
                        options=[
                            {"label": "Original RGB", "value": "rgb"},
                            {"label": "Input to this op", "value": "intermediate"},
                        ],
                        value="rgb",
                        inline=True,
                        className="mb-2",
                    ),
                    html.Small(
                        id=ids.PICKER_CHANNEL_HELP,
                        className="text-muted d-block mb-2",
                        children="",
                    ),
                    html.Div(
                        [
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        "Clear all",
                                        id=ids.BTN_PICKER_CLEAR,
                                        color="secondary",
                                        outline=True,
                                        size="sm",
                                    ),
                                    dbc.Button(
                                        "Remove last",
                                        id=ids.BTN_PICKER_UNDO,
                                        color="secondary",
                                        outline=True,
                                        size="sm",
                                    ),
                                ],
                                className="me-3",
                            ),
                            html.Span(
                                id=ids.PICKER_COUNT_LABEL,
                                children="0 points",
                                className="text-muted small",
                            ),
                        ],
                        className="d-flex align-items-center",
                    ),
                    html.Div(
                        id=ids.PICKER_OSD_DIV,
                        className="point-picker-osd",
                        style={
                            "height": "70vh",
                            "width": "100%",
                            "marginTop": "0.5rem",
                            "background": "#111",
                        },
                        **{"data-testid": "point-picker-osd-canvas"},  # type: ignore[arg-type]
                    ),
                    dcc.Store(id=ids.PICKER_STAGED_STORE, data=[]),
                    dcc.Store(id=ids.PICKER_TARGET_STORE, data=None),
                    dcc.Store(id=ids.PICKER_DZI_URL_STORE, data=None),
                    dcc.Store(
                        id=ids.PICKER_CHANNEL_AVAIL_STORE,
                        data={"rgb": True, "intermediate": False},
                    ),
                    # Output sink for the clientside mount/redraw/dispose
                    # callbacks. Never read; only written to satisfy Dash's
                    # "every callback needs an Output" rule.
                    dcc.Store(id=ids.PICKER_OSD_MOUNT_TRIGGER, data=None),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Cancel",
                        id=ids.BTN_PICKER_CANCEL,
                        color="secondary",
                        outline=True,
                    ),
                    dbc.Button(
                        "Confirm",
                        id=ids.BTN_PICKER_CONFIRM,
                        color="primary",
                    ),
                ]
            ),
        ],
        id=ids.MODAL_POINT_PICKER,
        is_open=False,
        size="xl",
        backdrop="static",
        scrollable=False,
    )


# ---------------------------------------------------------------------------
# Flask blueprint
# ---------------------------------------------------------------------------


def _validate_picker_url(session_id: str, source: str) -> Optional[Response]:
    """Return a 404 :class:`Response` for invalid session/source, else ``None``."""
    if _safe_session_id(session_id) and source in _VALID_SOURCES:
        return None
    logger.warning(
        "Rejected picker tile request: session_id=%r source=%r",
        session_id,
        source,
    )
    return _json_error("invalid session or source", 404)


def register_point_picker_routes(
    app: dash.Dash, image_root: Optional[Path]
) -> None:
    """Mount the per-session DZI tile blueprint on ``app.server``.

    The blueprint is intentionally dumb: it tiles whatever PNG is already
    on disk under ``<cache_dir>/<source>.png``. Dumping the source PNG
    is the responsibility of the modal-open / channel-toggle Dash
    callbacks (which know which session and which preview image to use).

    Two routes are exposed under ``/builder/tiles``:

    * ``GET /builder/tiles/<session_id>/<source>.dzi`` — DZI XML manifest.
    * ``GET /builder/tiles/<session_id>/<source>_files/<level>/<filename>``
      — a single tile PNG.

    Args:
        app: The :class:`dash.Dash` instance whose Flask server should be
            extended.
        image_root: Optional sandbox root passed through from
            :func:`create_app`. Captured by closure so route handlers can
            resolve their cache directories without re-reading the Dash
            server's config dict.
    """
    bp = Blueprint("builder_point_picker", __name__, url_prefix=BUILDER_TILES_PREFIX)

    @bp.route("/<session_id>/<source>.dzi")
    def manifest(session_id: str, source: str) -> Response:
        """Serve the DZI XML manifest, tiling lazily on first request."""
        err = _validate_picker_url(session_id, source)
        if err is not None:
            return err

        cache_dir = _session_cache_dir(image_root, session_id)
        png_path = _channel_png_path(cache_dir, source)
        if not png_path.exists():
            return _json_error("source not staged", 404)

        try:
            _dzi_tiler.tile(png_path, cache_dir)
        except Exception:
            logger.exception("DZI tile generation failed: %s", png_path)
            return _json_error("tile generation failed", 500)

        return send_from_directory(
            cache_dir,
            f"{source}.dzi",
            mimetype="application/xml",
        )

    @bp.route("/<session_id>/<source>_files/<int:level>/<filename>")
    def tile_endpoint(
        session_id: str, source: str, level: int, filename: str
    ) -> Response:
        """Serve an individual tile PNG from the per-session cache."""
        err = _validate_picker_url(session_id, source)
        if err is not None:
            return err

        secured = secure_filename(filename)
        if secured != filename or not _TILE_NAME_RE.match(filename):
            logger.warning(
                "Rejected picker tile request with unsafe filename: %r",
                filename,
            )
            return _json_error("invalid tile filename", 404)

        tile_dir = (
            _session_cache_dir(image_root, session_id)
            / f"{source}_files"
            / str(level)
        )
        if not tile_dir.is_dir():
            return _json_error("tile cache missing", 404)

        return send_from_directory(tile_dir, filename, mimetype="image/png")

    app.server.register_blueprint(bp)
    logger.debug(
        "Registered builder point-picker tile routes under /builder/tiles "
        "(image_root=%s)",
        image_root,
    )


# ---------------------------------------------------------------------------
# State helpers used by the callbacks
# ---------------------------------------------------------------------------


def _resolve_node_and_predecessor(
    state_data: Dict[str, Any], node_id: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Locate *node_id* in the visible scope and return ``(node, prev_id)``.

    ``prev_id`` is the immediately-preceding node in the same scope, or
    ``None`` if *node_id* is the first node (no predecessor in this scope).

    Args:
        state_data: Serialized :class:`BuilderState` from
            :data:`STORE_BUILDER_STATE`.
        node_id: ``StepNode.node_id`` of the node whose picker was opened.

    Returns:
        ``(node_dict, predecessor_node_id)``. Both elements are ``None``
        if *node_id* was not found in the visible scope.
    """
    try:
        state = state_from_json(state_data)
    except Exception:  # noqa: BLE001
        return (None, None)

    try:
        scope = current_scope(state)
    except Exception:  # noqa: BLE001
        return (None, None)

    nodes = scope.nodes
    for idx, node in enumerate(nodes):
        if node.node_id == node_id:
            prev_id = nodes[idx - 1].node_id if idx > 0 else None
            return (
                {
                    "node_id": node.node_id,
                    "class_name": node.class_name,
                    "is_pipeline": node.class_name == PIPELINE_CLASS_NAME,
                },
                prev_id,
            )
    return (None, None)


#: Per-(session, source) memo of the last image object whose pixels were
#: dumped to disk. Lets :func:`_stage_png_for_session` skip re-encoding
#: when the in-memory image is the same instance as the one already on
#: disk — avoiding cascading tile-pyramid invalidation in
#: :func:`_dzi_tiler.tile`, which mtime-compares manifest vs. source PNG.
_LAST_DUMPED: Dict[Tuple[str, str], int] = {}


def _stage_png_for_session(
    session_id: str,
    source: str,
    image_root: Optional[Path],
    image: Any,
) -> bool:
    """Dump *image*'s ``<source>`` channel into the per-session cache.

    Skips the write when the same image object was already dumped for this
    ``(session_id, source)`` and the PNG still exists on disk — keeps the
    DZI tile pyramid valid across modal-open round-trips.

    Returns ``True`` if a PNG is on disk afterwards, ``False`` otherwise.
    """
    if not _safe_session_id(session_id):
        return False
    cache_dir = _session_cache_dir(image_root, session_id)
    png_path = _channel_png_path(cache_dir, source)

    cache_key = (session_id, source)
    if (
        png_path.exists()
        and _LAST_DUMPED.get(cache_key) == id(image)
    ):
        return True

    try:
        ok = _dump_image_to_png(image, source, png_path)
    except Exception:
        logger.exception(
            "Failed to dump %s PNG for session %s", source, session_id
        )
        return False

    if ok:
        _LAST_DUMPED[cache_key] = id(image)
    return ok


def _stage_intermediate_png_bytes(
    session_id: str,
    image_root: Optional[Path],
    png_bytes: bytes,
) -> bool:
    """Write a pre-baked intermediate PNG into the per-session cache.

    The builder's preview run encodes each ops node's intermediate to a
    PNG once via ``render_node_preview`` (see ``builder/_image_renderer.py``);
    the bytes live in :class:`IntermediatesCache`. The picker re-uses that
    pre-baked output verbatim — no re-encoding — so the per-stage render
    rule (overlay for detector/refiner, detect_mat for enhancer, rgb for
    corrector) survives into the modal's DZI source.

    Skips the write when the same bytes object was already staged for this
    session — keeps the DZI tile pyramid valid across modal-open round-trips.
    """
    if not _safe_session_id(session_id):
        return False
    cache_dir = _session_cache_dir(image_root, session_id)
    png_path = _channel_png_path(cache_dir, "intermediate")

    cache_key = (session_id, "intermediate")
    if (
        png_path.exists()
        and _LAST_DUMPED.get(cache_key) == id(png_bytes)
    ):
        return True

    try:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.write_bytes(png_bytes)
    except Exception:
        logger.exception(
            "Failed to stage intermediate PNG for session %s", session_id
        )
        return False

    _LAST_DUMPED[cache_key] = id(png_bytes)
    return True


def _dzi_url(session_id: str, source: str) -> str:
    """Build the DZI manifest URL the JS layer should mount.

    Args:
        session_id: Per-tab uuid (already validated).
        source: One of :data:`_VALID_SOURCES`.

    Returns:
        URL string of the form ``/builder/tiles/<session_id>/<source>.dzi``.
    """
    return f"{BUILDER_TILES_PREFIX}/{session_id}/{source}.dzi"


# ---------------------------------------------------------------------------
# Dash callbacks
# ---------------------------------------------------------------------------


def register_point_picker_callbacks(app: dash.Dash) -> None:
    """Wire the modal's open / toggle / clear / undo / cancel / confirm callbacks.

    Click capture itself happens clientside in
    ``builder/assets/point_picker.js``; that layer pushes new points into
    :data:`PICKER_STAGED_STORE` via ``dash_clientside.set_props``. The
    server callbacks here observe that store, recompute the count label,
    and write the staged data back out into the matching node's picker
    store on Confirm.
    """

    # ------------------------------------------------------------------
    # 1. Open modal
    # ------------------------------------------------------------------
    @app.callback(
        Output(ids.MODAL_POINT_PICKER, "is_open", allow_duplicate=True),
        Output(ids.PICKER_TARGET_STORE, "data"),
        Output(ids.PICKER_STAGED_STORE, "data", allow_duplicate=True),
        Output(ids.PICKER_DZI_URL_STORE, "data", allow_duplicate=True),
        Output(ids.PICKER_CHANNEL_AVAIL_STORE, "data", allow_duplicate=True),
        Output(ids.PICKER_CHANNEL_RADIO, "value", allow_duplicate=True),
        Output(ids.PICKER_CHANNEL_HELP, "children", allow_duplicate=True),
        Input(
            {"type": "param-point-picker-btn", "prefix": ALL, "name": ALL},
            "n_clicks",
        ),
        State(ids.STORE_BUILDER_STATE, "data"),
        State(ids.STORE_SESSION_ID, "data"),
        State(
            {"type": "param-point-picker-store", "prefix": ALL, "name": ALL},
            "data",
        ),
        State(
            {"type": "param-point-picker-store", "prefix": ALL, "name": ALL},
            "id",
        ),
        prevent_initial_call=True,
    )
    def open_picker_modal(  # noqa: PLR0913
        click_payloads: List[Optional[int]],
        state_data: Optional[Dict[str, Any]],
        session_id: Optional[str],
        store_payloads: List[Any],
        store_ids: List[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        """Open the modal in response to a "Pick on image…" click."""
        noop = (no_update,) * 7
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return noop
        if triggered.get("type") != "param-point-picker-btn":
            return noop
        # Suppress the firing that happens at registration time when the
        # picker button first mounts (its ``n_clicks`` defaults to 0 -> 0,
        # but Dash still triggers the callback once with all-None values).
        if not click_payloads or all(c in (None, 0) for c in click_payloads):
            return noop

        node_id = triggered.get("prefix")
        param_name = triggered.get("name")
        if not node_id or not param_name:
            return noop

        # Seed the staged store from the existing param value (so the
        # modal opens "where you left off").
        staged_initial: List[List[float]] = []
        for component_id, val in zip(store_ids, store_payloads):
            if (
                component_id.get("prefix") == node_id
                and component_id.get("name") == param_name
            ):
                if isinstance(val, list):
                    staged_initial = [list(v) for v in val if v is not None]
                break

        # Resolve the predecessor + dump the rgb PNG straight away. The
        # blueprint will tile lazily on the first manifest request.
        _node_info, predecessor_id = _resolve_node_and_predecessor(
            state_data or {}, node_id
        )

        cache = get_cache()
        image, _path = cache.get_image(session_id or "")

        rgb_ok = False
        intermediate_ok = False
        sandbox_root = _resolve_image_root(app)

        if image is not None and session_id:
            rgb_ok = _stage_png_for_session(
                session_id, "rgb", sandbox_root, image
            )
            if predecessor_id is not None:
                pred_value = cache.get_intermediate(session_id, predecessor_id)
                if isinstance(pred_value, (bytes, bytearray)):
                    intermediate_ok = _stage_intermediate_png_bytes(
                        session_id, sandbox_root, bytes(pred_value)
                    )

        # Build the initial DZI URL (default: rgb). If rgb couldn't be
        # staged, leave the URL empty so the JS layer doesn't request a
        # 404'd manifest.
        dzi_url = (
            _dzi_url(session_id, "rgb")
            if (rgb_ok and session_id and _safe_session_id(session_id))
            else None
        )

        target = {"node_id": node_id, "param_name": param_name}
        avail = {"rgb": rgb_ok, "intermediate": intermediate_ok}

        help_msg = ""
        if not intermediate_ok:
            if predecessor_id is None:
                help_msg = "No predecessor — only the original RGB is available."
            else:
                help_msg = (
                    "Run preview first to populate the input-to-this-op view."
                )

        return (
            True,
            target,
            staged_initial,
            dzi_url,
            avail,
            "rgb",
            help_msg,
        )

    # ------------------------------------------------------------------
    # 2. Channel toggle — recompute DZI URL + lazy-stage intermediate
    # ------------------------------------------------------------------
    @app.callback(
        Output(ids.PICKER_DZI_URL_STORE, "data", allow_duplicate=True),
        Output(ids.PICKER_CHANNEL_AVAIL_STORE, "data", allow_duplicate=True),
        Input(ids.PICKER_CHANNEL_RADIO, "value"),
        State(ids.STORE_SESSION_ID, "data"),
        State(ids.STORE_BUILDER_STATE, "data"),
        State(ids.PICKER_TARGET_STORE, "data"),
        State(ids.PICKER_CHANNEL_AVAIL_STORE, "data"),
        prevent_initial_call=True,
    )
    def on_channel_toggle(
        value: Optional[str],
        session_id: Optional[str],
        state_data: Optional[Dict[str, Any]],
        target: Optional[Dict[str, Any]],
        avail: Optional[Dict[str, bool]],
    ) -> Tuple[Any, Any]:
        """Push a new DZI URL when the channel radio changes."""
        if value not in _VALID_SOURCES:
            return (no_update, no_update)
        if not session_id or not _safe_session_id(session_id):
            return (no_update, no_update)

        avail_out = dict(avail or {"rgb": False, "intermediate": False})

        # Lazy-dump the intermediate PNG the first time the user toggles
        # to it. Skip the cache lookup when intermediate is already staged.
        if (
            value == "intermediate"
            and not avail_out.get("intermediate")
            and target
            and state_data is not None
        ):
            target_node = target.get("node_id")
            if target_node:
                _node, predecessor_id = _resolve_node_and_predecessor(
                    state_data, target_node
                )
                if predecessor_id is not None:
                    pred_value = get_cache().get_intermediate(
                        session_id, predecessor_id
                    )
                    if isinstance(pred_value, (bytes, bytearray)):
                        avail_out["intermediate"] = _stage_intermediate_png_bytes(
                            session_id,
                            _resolve_image_root(app),
                            bytes(pred_value),
                        )

        return (_dzi_url(session_id, value), avail_out)

    # ------------------------------------------------------------------
    # 3. Clear all
    # ------------------------------------------------------------------
    @app.callback(
        Output(ids.PICKER_STAGED_STORE, "data", allow_duplicate=True),
        Input(ids.BTN_PICKER_CLEAR, "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_staged(n_clicks: Optional[int]) -> List[Any]:
        """Reset the staged store to an empty list."""
        if not n_clicks:
            return no_update  # type: ignore[return-value]
        return []

    # ------------------------------------------------------------------
    # 4. Remove last
    # ------------------------------------------------------------------
    @app.callback(
        Output(ids.PICKER_STAGED_STORE, "data", allow_duplicate=True),
        Input(ids.BTN_PICKER_UNDO, "n_clicks"),
        State(ids.PICKER_STAGED_STORE, "data"),
        prevent_initial_call=True,
    )
    def undo_last(
        n_clicks: Optional[int], staged: Optional[List[Any]]
    ) -> List[Any]:
        """Drop the most-recently-added staged point."""
        if not n_clicks:
            return no_update  # type: ignore[return-value]
        if not staged:
            return []
        return list(staged)[:-1]

    # ------------------------------------------------------------------
    # 5. Update count label
    # ------------------------------------------------------------------
    @app.callback(
        Output(ids.PICKER_COUNT_LABEL, "children", allow_duplicate=True),
        Input(ids.PICKER_STAGED_STORE, "data"),
        prevent_initial_call=True,
    )
    def update_count(data: Optional[List[Any]]) -> str:
        """Refresh the count label whenever the staged store changes."""
        n = len(data or [])
        return f"{n} point{'s' if n != 1 else ''}"

    # ------------------------------------------------------------------
    # 6. Cancel — close modal, clear staged
    # ------------------------------------------------------------------
    @app.callback(
        Output(ids.MODAL_POINT_PICKER, "is_open", allow_duplicate=True),
        Output(ids.PICKER_STAGED_STORE, "data", allow_duplicate=True),
        Input(ids.BTN_PICKER_CANCEL, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_cancel(n_clicks: Optional[int]) -> Tuple[Any, Any]:
        """Close the modal without writing anything back to the param store."""
        if not n_clicks:
            return (no_update, no_update)
        return (False, [])

    # ------------------------------------------------------------------
    # 7. Confirm — write staged points into the matching node's store
    # ------------------------------------------------------------------
    @app.callback(
        Output(ids.MODAL_POINT_PICKER, "is_open", allow_duplicate=True),
        Output(
            {"type": "param-point-picker-store", "prefix": ALL, "name": ALL},
            "data",
            allow_duplicate=True,
        ),
        Input(ids.BTN_PICKER_CONFIRM, "n_clicks"),
        State(ids.PICKER_STAGED_STORE, "data"),
        State(ids.PICKER_TARGET_STORE, "data"),
        State(
            {"type": "param-point-picker-store", "prefix": ALL, "name": ALL},
            "id",
        ),
        prevent_initial_call=True,
    )
    def on_confirm(
        n_clicks: Optional[int],
        staged: Optional[List[Any]],
        target: Optional[Dict[str, Any]],
        store_ids: List[Dict[str, Any]],
    ) -> Tuple[Any, List[Any]]:
        """Fan-out the staged points into the addressed picker store."""
        if not n_clicks or not target:
            # Match Dash's expected output cardinality even on early-return.
            return (no_update, [no_update for _ in store_ids])

        target_node = target.get("node_id")
        target_param = target.get("param_name")
        payload = list(staged or [])

        out_data: List[Any] = []
        for component_id in store_ids:
            if (
                component_id.get("prefix") == target_node
                and component_id.get("name") == target_param
            ):
                out_data.append(payload)
            else:
                out_data.append(no_update)

        return (False, out_data)

    # ------------------------------------------------------------------
    # 8. Channel-availability → radio options
    # ------------------------------------------------------------------
    @app.callback(
        Output(ids.PICKER_CHANNEL_RADIO, "options"),
        Input(ids.PICKER_CHANNEL_AVAIL_STORE, "data"),
        prevent_initial_call=True,
    )
    def toggle_radio_options(
        avail: Optional[Dict[str, bool]],
    ) -> List[Dict[str, Any]]:
        """Disable the *intermediate* radio option when no preview is staged."""
        avail = avail or {"rgb": True, "intermediate": False}
        return [
            {
                "label": "Original RGB",
                "value": "rgb",
                "disabled": not bool(avail.get("rgb", True)),
            },
            {
                "label": "Input to this op",
                "value": "intermediate",
                "disabled": not bool(avail.get("intermediate", False)),
            },
        ]


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _resolve_image_root(app: dash.Dash) -> Optional[Path]:
    """Return the ``image_root`` stashed on ``app.server.config`` by ``create_app``.

    Args:
        app: The Dash app instance.

    Returns:
        The configured ``Path`` or ``None`` if absent.
    """
    try:
        root = app.server.config.get(CFG_IMAGE_ROOT)
    except Exception:  # noqa: BLE001
        return None
    if root is None:
        return None
    return Path(root)


__all__ = [
    "build_point_picker_modal",
    "register_point_picker_routes",
    "register_point_picker_callbacks",
]
