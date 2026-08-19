"""OME-Zarr (NGFF 0.5 / Zarr format v3) store layout, geometry, and commit protocol.

Single source of truth for everything about the on-disk shape of a per-image
store: the directory layout, the pyramid geometry, the chunk/shard/codec
policy, the ``attributes.phenotypic`` contract, the write-only OME projection,
and the rename-promote commit primitive.

Nothing here reads or writes an :class:`~phenotypic.Image`; the layer that does
is :mod:`phenotypic._core._image_parts._image_io_handler`. Keeping the geometry
free of the image model is what lets the committed logic-validation script
(``docs/superpowers/logic_validation_scripts/2026-08-18-ome-zarr-image-store/``)
re-derive every numeric claim from numpy alone.

See also:
    ``docs/superpowers/specs/2026-08-18-ome-zarr-image-store/design.md``
"""

from __future__ import annotations

import math
from typing import Final, Literal

import numpy as np

# ``zarr`` alone stays function-local: it is a new heavy dependency and this
# module is re-exported through ``sdk_/__init__.py``, so deferring it keeps
# ``import phenotypic.sdk_`` cheap. Everything else is stdlib -- hoisted so
# public signatures can annotate ``Path`` directly instead of quoting it.
# Each task adds the stdlib names it actually uses: the promote primitive's
# ``errno``/``os``/``shutil``/``time``/``uuid4``/``logging``/``hashlib`` land
# with that primitive, because importing them ahead of their first use is a
# ruff F401 in this repo's default rule set (E4, E7, E9, F).

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

#: NGFF specification version written into every ``ome`` block.
NGFF_VERSION: Final[str] = "0.5"

#: ``bioformats2raw.layout`` marker on the root group (named-series collection).
BIOFORMATS2RAW_LAYOUT: Final[int] = 3

#: Version of the PhenoTypic *group and array* layout. Distinct from
#: ``metadata_schema_version``, which versions the header namespace.
STORE_SCHEMA_VERSION: Final[int] = 3

#: Directory suffix for one per-image store.
STORE_SUFFIX: Final[str] = ".ome.zarr"

#: Halve pyramid levels until ``max(H, W) <= PYRAMID_STOP_PX``.
PYRAMID_STOP_PX: Final[int] = 512

OME_GROUP: Final[str] = "OME"
OME_XML_NAME: Final[str] = "METADATA.ome.xml"
LABELS_GROUP: Final[str] = "labels"
OBJMAP_LABEL: Final[str] = "objmap"

#: Canonical series order. ``rgb`` is omitted from a store when empty; the
#: remaining names keep this relative order.
SERIES_ORDER: Final[tuple[str, str, str]] = ("rgb", "gray", "detect_mat")

AXES_3D: Final[tuple[str, str, str]] = ("c", "y", "x")
AXES_2D: Final[tuple[str, str]] = ("y", "x")

#: NGFF axis ``type`` per dimension name.
AXIS_TYPES: Final[dict[str, str]] = {"c": "channel", "y": "space", "x": "space"}

#: Downscaling method per array kind, with its human-readable description.
#: Single source for BOTH the public ``multiscales[].type``/``metadata`` (2.4
#: SHOULD) and the private ``attributes.phenotypic.pyramid.downsample`` record.
#: Every writer of either MUST read it from here -- see
#: ``build_phenotypic_attributes``, which is the other reader.
DOWNSAMPLE_METHODS: Final[dict[str, tuple[str, str]]] = {
    "image": ("mean", "2x block mean over an edge-replicated pad"),
    "label": ("nearest", "2x nearest-neighbour (top-left of each block)"),
}

#: The bare ``kind -> method`` view, which is what the private pyramid record
#: stores.
DOWNSAMPLE_KINDS: Final[dict[str, str]] = {
    kind: method for kind, (method, _) in DOWNSAMPLE_METHODS.items()
}


def axes_for(series: str) -> tuple[str, ...]:
    """Return the ``dimension_names`` tuple for one series or label name.

    Args:
        series: ``"rgb"``, ``"gray"``, ``"detect_mat"``, or ``"objmap"``.

    Returns:
        ``("c", "y", "x")`` for ``rgb``; ``("y", "x")`` otherwise.
    """
    return AXES_3D if series == "rgb" else AXES_2D


# ---------------------------------------------------------------------------
# Pyramid geometry
# ---------------------------------------------------------------------------


def pyramid_level_count(
    height: int, width: int, *, stop_px: int = PYRAMID_STOP_PX
) -> int:
    """Number of pyramid levels when halving until ``max(H, W) <= stop_px``.

    ``ceil``, not ``floor``: a floor-based formula terminates one level early
    and leaves a 4000x3000 plate's smallest level at 1000x750.

    Args:
        height: Level-0 height in pixels.
        width: Level-0 width in pixels.
        stop_px: Longest-edge threshold at which halving stops.

    Returns:
        A level count of at least 1.
    """
    longest = max(height, width)
    if longest <= stop_px:
        return 1
    return int(math.ceil(math.log2(longest / stop_px))) + 1


def pyramid_level_shapes(
    shape: tuple[int, ...], levels: int
) -> tuple[tuple[int, ...], ...]:
    """Explicit shape per pyramid level, ceil-halving the two spatial axes.

    A leading channel axis (3-D input) is carried through unchanged.

    Args:
        shape: Level-0 shape, ``(y, x)`` or ``(c, y, x)``.
        levels: Number of levels to emit, including level 0.

    Returns:
        A tuple of ``levels`` shapes, starting with *shape*.
    """
    shapes: list[tuple[int, ...]] = [tuple(shape)]
    for _ in range(levels - 1):
        previous = shapes[-1]
        lead, (h, w) = previous[:-2], previous[-2:]
        shapes.append((*lead, max(1, (h + 1) // 2), max(1, (w + 1) // 2)))
    return tuple(shapes)


def level_scale_vector(
    level0: tuple[int, ...], level_n: tuple[int, ...]
) -> list[float]:
    """Per-axis downsample factor from the *actual* level shapes.

    NGFF requires ``coordinateTransformations.scale`` to describe the real
    relationship between levels. Odd extents make the true ratio diverge from
    ``2 ** n``, so this is derived from shapes and never from the level index.

    Args:
        level0: Level-0 shape.
        level_n: Shape of the level being described.

    Returns:
        One float per axis, in axis order. Any leading channel axis is 1.0.
    """
    return [float(a) / float(b) for a, b in zip(level0, level_n, strict=True)]


def downsample_image(array: np.ndarray) -> np.ndarray:
    """2x block-mean downsample with edge replication, preserving dtype.

    Edge replication (rather than zero padding) is what keeps an odd trailing
    row or column at its own brightness instead of darkening it toward zero.
    The spatial axes are the last two; any leading channel axis is preserved.

    Args:
        array: 2-D ``(y, x)`` or 3-D ``(c, y, x)`` array.

    Returns:
        An array whose spatial extents are ``(n + 1) // 2``.
    """
    h, w = array.shape[-2:]
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        pad_width = [(0, 0)] * (array.ndim - 2) + [(0, pad_h), (0, pad_w)]
        array = np.pad(array, pad_width, mode="edge")
    lead = array.shape[:-2]
    ph, pw = array.shape[-2:]
    blocks = array.astype(np.float64).reshape(*lead, ph // 2, 2, pw // 2, 2)
    reduced = blocks.mean(axis=(-3, -1))
    if np.issubdtype(array.dtype, np.integer):
        return np.rint(reduced).astype(array.dtype)
    return reduced.astype(array.dtype)


def downsample_label(array: np.ndarray) -> np.ndarray:
    """2x nearest-neighbour downsample (top-left of each 2x2 block).

    A label map must never be mean-downsampled: averaging fabricates label
    values present at no level-0 pixel. Verified by claim C5 of the committed
    logic-validation script.

    Args:
        array: 2-D ``(y, x)`` integer label array.

    Returns:
        An array whose extents are ``(n + 1) // 2``, with dtype preserved and
        no label value absent from *array*.
    """
    return array[..., ::2, ::2]


def build_pyramid(
    array: np.ndarray, levels: int, *, kind: Literal["image", "label"]
) -> list[np.ndarray]:
    """Materialise every pyramid level for one array.

    Args:
        array: Level-0 array.
        levels: Level count, including level 0.
        kind: ``"image"`` downsamples by local mean; ``"label"`` by
            nearest-neighbour.

    Returns:
        A list of ``levels`` arrays, starting with *array*.
    """
    reduce = downsample_image if kind == "image" else downsample_label
    out = [array]
    for _ in range(levels - 1):
        out.append(reduce(out[-1]))
    return out


# ---------------------------------------------------------------------------
# Chunk / shard / codec policy
# ---------------------------------------------------------------------------

#: Inner chunk extent on the two spatial axes.
CHUNK_YX: Final[tuple[int, int]] = (1024, 1024)

#: Shard extent on the two spatial axes. A shard is the write-buffer unit.
SHARD_YX: Final[tuple[int, int]] = (4096, 4096)

#: Compression codec, replacing the HDF path's gzip-4.
CODEC_NAME: Final[str] = "zstd"

#: Chunk-key separator. ``"."`` makes a chunk key one path segment (``c.0.0.0``)
#: rather than four nested directories -- a Windows MAX_PATH measure that MUST
#: be uniform store-wide.
CHUNK_KEY_SEPARATOR: Final[str] = "."


def chunk_shape_for(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Inner chunk shape for one array level.

    Clamped to the level's own extent so a small pyramid level is never given a
    chunk larger than itself.

    Args:
        shape: Level shape, ``(y, x)`` or ``(c, y, x)``.

    Returns:
        ``(1, cy, cx)`` for a 3-D array, ``(cy, cx)`` for 2-D.
    """
    h, w = shape[-2:]
    spatial = (min(CHUNK_YX[0], h), min(CHUNK_YX[1], w))
    return (*(1 for _ in shape[:-2]), *spatial)


def shard_shape_for(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Shard shape for one array level.

    Spans the **full** channel extent, so per-channel chunks collapse into one
    file. On the spatial axes it is the fixed ``SHARD_YX``, **not** clamped to
    the level extent: the Zarr v3 sharding codec constrains shard-vs-chunk
    divisibility only, never shard-vs-array, and partial edge shards are normal.
    Clamping to the extent and rounding down to a chunk multiple would turn a
    4000x4096-shard level into four shard files instead of one, contradicting
    the committed logic-validation script's file counts.

    A level below one chunk collapses to ``chunk == shard == extent``, which
    keeps divisibility trivially true and is one chunk and one shard either way.

    Args:
        shape: Level shape, ``(y, x)`` or ``(c, y, x)``.

    Returns:
        A shard shape that is an exact multiple of :func:`chunk_shape_for`.
    """
    chunk = chunk_shape_for(shape)
    lead = tuple(int(extent) for extent in shape[:-2])  # full channel extent
    spatial = tuple(
        chunk[len(shape) - 2 + axis] if extent < CHUNK_YX[axis] else SHARD_YX[axis]
        for axis, extent in enumerate(shape[-2:])
    )
    return (*lead, *spatial)


def array_create_kwargs(
    shape: tuple[int, ...], dtype: np.dtype, series: str
) -> dict:
    """Keyword arguments for ``zarr.create_array`` for one level of one series.

    Args:
        shape: Level shape.
        dtype: Array dtype.
        series: ``"rgb"``, ``"gray"``, ``"detect_mat"``, or ``"objmap"`` --
            selects the axis names.

    Returns:
        A kwargs mapping carrying ``shape``, ``dtype``, ``chunks``, ``shards``,
        ``compressors``, ``dimension_names``, and ``chunk_key_encoding``.
    """
    from zarr.codecs import ZstdCodec

    return {
        "shape": tuple(shape),
        "dtype": dtype,
        "chunks": chunk_shape_for(shape),
        "shards": shard_shape_for(shape),
        "compressors": (ZstdCodec(),),
        "dimension_names": list(axes_for(series)),
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": CHUNK_KEY_SEPARATOR},
        },
    }
