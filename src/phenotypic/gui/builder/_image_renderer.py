"""Render :class:`phenotypic.Image` channels as base64 PNG data URIs for Dash.

The pipeline-builder inspector pane embeds previews via plain ``<img src=...>``
tags rather than streaming endpoints — base64-encoded PNGs avoid spinning up a
per-session asset route. Channels supported: ``rgb``, ``gray``, ``detect_mat``,
and ``objmap``. Detector / refiner nodes get a separate overlay renderer that
alpha-blends the objmap onto the post-op detect_mat
(:func:`to_overlay_png_bytes`).

Large source arrays are downscaled with ``cv2.resize(INTER_AREA)`` *before*
encoding so a 4000×6000 uint16 plate doesn't pay PIL's slow per-pixel thumbnail
cost.

The builder pre-bakes one PNG per intermediate at preview-run time so the
inspector never re-encodes on selection (see :func:`render_node_preview`).
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from dash import dash_table
from PIL import Image as PILImage

from phenotypic.gui._config import ChannelName
from phenotypic.gui._design import FONT_FAMILY_MONO, FONT_SIZE_LABEL
from phenotypic.gui._operation_registry import get_registry
from phenotypic.gui.builder._state import stage_of

if TYPE_CHECKING:  # pragma: no cover - import only used for type hints
    import pandas as pd  # type: ignore[import-untyped]
    import phenotypic
    from phenotypic.gui._operation_registry import OperationInfo


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_channel(image: "phenotypic.Image", channel: ChannelName) -> np.ndarray:
    """Pull the requested channel from an :class:`Image` via the accessor API.

    Args:
        image: PhenoTypic :class:`Image` (or :class:`GridImage`).
        channel: One of ``rgb``, ``gray``, ``detect_mat``, ``objmap``.

    Returns:
        A NumPy array (the accessor's ``[:]`` slice).

    Raises:
        ValueError: If ``channel`` is not a supported name.
    """
    if channel == "rgb":
        return np.asarray(image.rgb[:])
    if channel == "gray":
        return np.asarray(image.gray[:])
    if channel == "detect_mat":
        return np.asarray(image.detect_mat[:])
    if channel == "objmap":
        return np.asarray(image.objmap[:])
    raise ValueError(
        f"Unknown channel {channel!r}; expected one of "
        "'rgb', 'gray', 'detect_mat', 'objmap'."
    )


def _downscale(arr: np.ndarray, max_dim: int) -> np.ndarray:
    """Resize ``arr`` so its longest spatial side ≤ ``max_dim`` pixels.

    Uses ``cv2.resize`` with ``INTER_AREA`` (best for shrinking). Handles
    integer-label maps by coercing to a cv2-supported dtype before resizing
    and casting back so label IDs survive the resampling pass.

    Args:
        arr: 2-D or 3-D NumPy array.
        max_dim: Maximum allowed length of the longer spatial side.

    Returns:
        Either ``arr`` unchanged (when already small enough) or a resized copy.
    """
    if arr.ndim < 2:
        return arr
    h, w = arr.shape[:2]
    longer = max(h, w)
    if longer <= max_dim:
        return arr
    scale = max_dim / float(longer)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    src = arr
    cast_back: np.dtype | None = None
    interp = cv2.INTER_AREA

    # cv2.resize doesn't accept uint16 multi-channel directly in all builds;
    # the safe path is to widen integer label maps to int32 then restore.
    if np.issubdtype(arr.dtype, np.integer) and arr.dtype not in (
        np.uint8, np.uint16, np.int16, np.int32,
    ):
        cast_back = arr.dtype
        src = arr.astype(np.int32)

    resized = cv2.resize(src, (new_w, new_h), interpolation=interp)
    if cast_back is not None:
        resized = resized.astype(cast_back)
    return resized


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Scale an arbitrary-range array to uint8 in [0, 255].

    Floats are clipped to [0, 1] when their max ≤ 1, otherwise rescaled by
    their global max. Integer dtypes are rescaled by their global max so that
    e.g. uint16 RGB lands in 0..255 with full dynamic range preserved.
    """
    if arr.dtype == np.uint8:
        return arr

    a = arr.astype(np.float64, copy=False)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    amax = float(finite.max())
    amin = float(finite.min())

    if amax <= 1.0 and amin >= 0.0:
        scaled = a * 255.0
    elif amax > 0:
        # Min-max stretch keeps low-contrast detect_mat readable.
        denom = amax - amin if amax > amin else amax
        scaled = (a - amin) / denom * 255.0
    else:
        scaled = np.zeros_like(a)

    return np.clip(scaled, 0, 255).astype(np.uint8)


def _label_map_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert an integer label map to an 8-bit RGB visualisation.

    Tries :func:`skimage.color.label2rgb` first (perceptually distinct colours,
    background preserved). Falls back to a tab20 colormap on the
    ``(label % 20)`` index when scikit-image is unavailable.
    """
    try:
        from skimage.color import label2rgb

        rgb = label2rgb(arr, bg_label=0)
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    except Exception:
        try:
            import matplotlib

            cmap = matplotlib.colormaps["tab20"]
            idx = (arr.astype(np.int64) % 20).astype(np.int64)
            rgba = cmap(idx)
            rgba[arr == 0] = (0.0, 0.0, 0.0, 1.0)
            return (rgba[..., :3] * 255.0).astype(np.uint8)
        except Exception:
            # Last-ditch fallback: greyscale modulus.
            return np.stack([(arr % 256).astype(np.uint8)] * 3, axis=-1)


def _channel_to_rgb_uint8(arr: np.ndarray, channel: ChannelName) -> np.ndarray:
    """Project an arbitrary channel onto a displayable HxWx3 uint8 array.

    Args:
        arr: Source array straight from the channel accessor.
        channel: Channel name driving the projection rule.

    Returns:
        ``(H, W, 3)`` uint8 array suitable for ``PIL.Image.fromarray(..., 'RGB')``.
    """
    if channel == "objmap":
        return _label_map_to_rgb(arr)

    if channel == "rgb":
        u8 = _normalize_to_uint8(arr)
        if u8.ndim == 2:
            return np.stack([u8] * 3, axis=-1)
        if u8.shape[-1] == 4:
            return u8[..., :3]
        return u8

    # gray / detect_mat → grey-as-RGB.
    u8 = _normalize_to_uint8(arr)
    if u8.ndim == 3:
        # Already multi-channel — drop alpha if present.
        return u8[..., :3]
    return np.stack([u8] * 3, axis=-1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _encode_png(rgb_uint8: np.ndarray) -> bytes:
    """PIL-encode an ``(H, W, 3)`` uint8 array as PNG bytes."""

    pil = PILImage.fromarray(rgb_uint8, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def to_png_bytes(
    image: "phenotypic.Image",
    channel: ChannelName = "rgb",
    *,
    max_dim: int = 512,
) -> bytes:
    """Render an :class:`Image` channel as raw PNG bytes (no base64 wrap).

    The image is downscaled with ``cv2.resize(INTER_AREA)`` before PNG encoding
    so multi-megapixel raw scans encode in a fraction of a second. Use
    :func:`bytes_to_data_uri` to wrap the result for an HTML ``<img>`` ``src``.

    Args:
        image: PhenoTypic :class:`Image` (or :class:`GridImage`) instance.
        channel: One of ``"rgb"``, ``"gray"``, ``"detect_mat"``, ``"objmap"``.
            Defaults to ``"rgb"``.
        max_dim: Maximum length of the longer spatial side after resizing,
            in pixels. Defaults to ``512``.

    Returns:
        Raw PNG-encoded bytes (starting with the ``\\x89PNG`` magic).

    Raises:
        ValueError: If ``channel`` is not one of the supported names.

    Examples:
        >>> from phenotypic.data._synthetic_data import load_synth_yeast_plate
        >>> img = load_synth_yeast_plate()
        >>> blob = to_png_bytes(img, channel="rgb")
        >>> blob[:8] == b"\\x89PNG\\r\\n\\x1a\\n"
        True
    """
    arr = _read_channel(image, channel)
    arr = _downscale(arr, max_dim=max_dim)
    rgb_uint8 = _channel_to_rgb_uint8(arr, channel)
    return _encode_png(rgb_uint8)


def bytes_to_data_uri(blob: bytes) -> str:
    """Wrap raw PNG bytes as a ``data:image/png;base64,...`` URI string.

    Args:
        blob: Raw PNG-encoded bytes.

    Returns:
        A URI suitable for direct use as the ``src`` of an HTML ``<img>``
        element.
    """
    encoded = base64.b64encode(bytes(blob)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def to_data_uri(
    image: "phenotypic.Image",
    channel: ChannelName = "rgb",
    *,
    max_dim: int = 512,
) -> str:
    """Render an :class:`Image` channel as a base64 PNG data URI.

    Thin wrapper over :func:`to_png_bytes` + :func:`bytes_to_data_uri`. Kept
    for callers that want one-shot rendering; the builder cache pre-bakes
    raw bytes via :func:`to_png_bytes` directly.

    Args:
        image: PhenoTypic :class:`Image` (or :class:`GridImage`) instance.
        channel: One of ``"rgb"``, ``"gray"``, ``"detect_mat"``, ``"objmap"``.
        max_dim: Maximum length of the longer spatial side, in pixels.

    Returns:
        A string of the form ``"data:image/png;base64,<...>"``.

    Examples:
        >>> from phenotypic.data._synthetic_data import load_synth_yeast_plate
        >>> img = load_synth_yeast_plate()
        >>> uri = to_data_uri(img, channel="rgb")
        >>> uri.startswith("data:image/png;base64,")
        True
    """
    return bytes_to_data_uri(to_png_bytes(image, channel, max_dim=max_dim))


def to_overlay_rgb_array(
    image: "phenotypic.Image",
    *,
    max_dim: int = 512,
    alpha: float = 0.4,
) -> np.ndarray:
    """Composite the objmap (alpha-blended) over the post-op detect_mat as RGB.

    The array-returning core shared by the builder's overlay PNG renderer
    (:func:`to_overlay_png_bytes`) and the ``/tune/`` Curate candidate overlay
    (``phenotypic.gui.tune._overlays.render_candidate_overlay``). Both pass the
    same arguments through this one ``skimage.color.label2rgb`` call so the
    builder preview and the tune overlay stay pixel-for-pixel identical;
    only the final encoding (raw array vs. PNG bytes) differs at the call site.

    Falls back to a plain :func:`_label_map_to_rgb` colormap on the objmap when
    ``scikit-image`` isn't importable, matching :func:`_label_map_to_rgb`.

    Args:
        image: PhenoTypic :class:`Image` whose ``detect_mat`` and ``objmap``
            accessors should both be valid (post-detector / post-refiner).
        max_dim: Maximum length of the longer spatial side after resizing.
        alpha: Label-overlay opacity in ``[0, 1]``. Higher = more colored.

    Returns:
        An ``(H, W, 3)`` uint8 overlay array (RGB, ready for ``go.Image`` or
        :func:`_encode_png`).
    """
    detect = _read_channel(image, "detect_mat")
    objmap = _read_channel(image, "objmap")

    detect = _downscale(detect, max_dim=max_dim)
    objmap = _downscale(objmap, max_dim=max_dim)

    base_u8 = _normalize_to_uint8(detect)
    if base_u8.ndim == 3:
        base_u8 = base_u8[..., :3]
    else:
        base_u8 = np.stack([base_u8] * 3, axis=-1)

    try:
        from skimage.color import label2rgb

        rgb = label2rgb(
            objmap,
            image=base_u8,
            bg_label=0,
            alpha=float(alpha),
            image_alpha=1.0,
            kind="overlay",
        )
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    except Exception:
        return _label_map_to_rgb(objmap)


def to_overlay_png_bytes(
    image: "phenotypic.Image",
    *,
    max_dim: int = 512,
    alpha: float = 0.4,
) -> bytes:
    """Composite the objmap (alpha-blended) over the post-op detect_mat.

    Used for detector / refiner intermediates so the inspector shows
    *which colonies were segmented at this step* against the same grayscale
    background the detector saw. Thin PNG-encoding wrapper over
    :func:`to_overlay_rgb_array` — the shared core that the ``/tune/`` Curate
    overlay also renders through, so the two stay visually identical.

    Args:
        image: PhenoTypic :class:`Image` whose ``detect_mat`` and ``objmap``
            accessors should both be valid (post-detector / post-refiner).
        max_dim: Maximum length of the longer spatial side after resizing.
        alpha: Label-overlay opacity in ``[0, 1]``. Higher = more colored.

    Returns:
        Raw PNG bytes encoding an ``(H, W, 3)`` uint8 overlay.
    """
    return _encode_png(to_overlay_rgb_array(image, max_dim=max_dim, alpha=alpha))


# ---------------------------------------------------------------------------
# Per-stage dispatcher
# ---------------------------------------------------------------------------


def _registry_info_for(class_name: str) -> "OperationInfo | None":
    """Return the :class:`OperationInfo` for *class_name*, or ``None``."""

    try:
        return get_registry().get(class_name)
    except Exception:  # noqa: BLE001
        return None


def _channel_for_class(
    class_name: str, info: "OperationInfo | None"
) -> ChannelName:
    """Pick the channel best showing *class_name*'s output (non-overlay).

    Fallback channel selector for stages that don't get the overlay renderer
    (correctors, nested pipelines, unknown classes, measurement nodes that
    fell through here). ``info`` is passed in so callers don't pay the
    registry lookup twice.
    """

    try:
        stage = stage_of(class_name)
    except KeyError:
        return "rgb"
    if stage != "ops" or info is None:
        return "rgb"
    if info.category == "Enhancer":
        return "detect_mat"
    if info.category in {"Detector", "Refiner"}:
        # Safety-net channel for the case where the overlay renderer raises
        # and the caller falls back to a single-channel render.
        return "objmap"
    return "rgb"


def render_node_preview(
    image: "phenotypic.Image",
    class_name: str,
    *,
    max_dim: int = 512,
) -> bytes:
    """Render the node-appropriate PNG for a pipeline intermediate.

    Per-stage rule (matches the GUI's "show what this op did" intent):

    * Enhancers → :func:`to_png_bytes` on ``detect_mat``.
    * Detectors / Refiners → :func:`to_overlay_png_bytes` (objmap on
      detect_mat).
    * Correctors / nested ``Pipeline`` / unknown → :func:`to_png_bytes` on
      ``rgb``.

    Pre-baked once at preview-run time and cached as bytes; the inspector
    only base64-wraps for display.

    Args:
        image: Post-op intermediate :class:`Image`.
        class_name: Operation class name from ``StepNode.class_name``.
        max_dim: Maximum length of the longer spatial side after resizing.

    Returns:
        Raw PNG bytes.
    """

    info = _registry_info_for(class_name)
    if info is not None and info.category in {"Detector", "Refiner"}:
        return to_overlay_png_bytes(image, max_dim=max_dim)

    channel = _channel_for_class(class_name, info)
    return to_png_bytes(image, channel, max_dim=max_dim)


def dataframe_to_table(
    df: "pd.DataFrame",
    max_rows: int = 50,
    *,
    table_id: str | dict[str, Any] | None = None,
) -> Any:
    """Render the head of a DataFrame as a Dash ``DataTable`` for the inspector.

    The table is sortable, has horizontal overflow scrolling, and renders no
    pagination controls — callers slice to ``max_rows`` rows up front to keep
    payloads small.

    Args:
        df: Source DataFrame. ``None`` / empty frames render an empty table.
        max_rows: Maximum number of rows to include. Defaults to 50.
        table_id: Optional component id (string or pattern-match dict). When
            omitted, Dash assigns an auto id.

    Returns:
        A configured :class:`dash_table.DataTable` ready to drop into a layout.
    """
    if df is None or len(df) == 0:
        rows: list[dict[str, Any]] = []
        columns: list[dict[str, Any]] = []
    else:
        head = df.head(max_rows)
        rows = head.to_dict("records")
        columns = [{"name": str(c), "id": str(c)} for c in head.columns]

    kwargs: dict[str, Any] = {
        "data": rows,
        "columns": columns,
        "sort_action": "native",
        "page_action": "none",
        "style_table": {"overflowX": "auto"},
        "style_cell": {
            "fontFamily": FONT_FAMILY_MONO,
            "fontSize": FONT_SIZE_LABEL,
            "padding": "4px 8px",
            "textAlign": "left",
        },
        "style_header": {"fontWeight": "bold"},
    }
    if table_id is not None:
        kwargs["id"] = table_id

    return dash_table.DataTable(**kwargs)  # type: ignore[attr-defined]


__all__ = [
    "to_data_uri",
    "to_png_bytes",
    "bytes_to_data_uri",
    "to_overlay_rgb_array",
    "to_overlay_png_bytes",
    "render_node_preview",
    "dataframe_to_table",
]
