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

import errno
import json
import math
import os
import re as _re
import shutil
import time
from pathlib import Path
from typing import Final, Literal, Sequence
from uuid import uuid4

import numpy as np

# ``zarr`` alone stays function-local: it is a new heavy dependency and this
# module is re-exported through ``sdk_/__init__.py``, so deferring it keeps
# ``import phenotypic.sdk_`` cheap. Everything else is stdlib -- hoisted so
# public signatures can annotate ``Path`` directly instead of quoting it.
# Each task adds the stdlib names it actually uses: ``errno``/``os``/``shutil``/
# ``time``/``uuid4`` arrived with the promote primitive, because importing them
# ahead of their first use is a ruff F401 in this repo's default rule set
# (E4, E7, E9, F).

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

#: Zarr v3's root metadata document, at the top of every store. Written
#: **last** by the promote protocol, which is what lets a reader treat its
#: presence as "this store is complete" and lets a completion marker
#: fingerprint the store by this file alone.
STORE_ROOT_JSON: Final[str] = "zarr.json"

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

    An integer result is rounded with ``np.rint`` -- **banker's rounding**, so
    an exact ``.5`` mean goes to the nearest EVEN value, not always upward.
    Rounding rather than truncating is what keeps the pyramid unbiased: a plain
    ``astype`` drops the fraction at every level, so a uniform uint8 plate
    drifts 127.52 -> 126.02 over four levels and every thumbnail darkens.

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


# ---------------------------------------------------------------------------
# attributes.phenotypic -- the source of truth on read
# ---------------------------------------------------------------------------

# `METADATA_SCHEMA_VERSION` is deliberately NOT defined here (user ruling,
# 2026-08-19; spec 2.3). Writing it would be a hard-coded constant asserting
# something about metadata this same writer stores "verbatim and unvalidated" --
# no code path enforces it. That also inverts the HDF contract, where the
# attribute is written only AFTER a successful rewrite
# (`sdk_/_metadata_migration.py:1401`) and its absence or mismatch is precisely
# what marks a target migratable. With header-only store migration cut, nothing
# reads it either. The HDF-side `_METADATA_SCHEMA_VERSION_*` constants are a
# different contract and stay where they are.


class PhenotypicAttr:
    """Keys inside the namespaced ``attributes.phenotypic`` block.

    Spelled out here so a renamed key fails at type-check time rather than
    silently at runtime, matching the ``JobMetadataKey`` pattern
    already used in :mod:`phenotypic.sdk_._io_constants`.
    """

    ROOT: Final[str] = "phenotypic"
    STORE_SCHEMA_VERSION: Final[str] = "store_schema_version"
    PHENOTYPIC_VERSION: Final[str] = "phenotypic_version"
    IMAGE_CLASS: Final[str] = "image_class"
    WORK_ID: Final[str] = "work_id"
    SERIES: Final[str] = "series"
    LABELS: Final[str] = "labels"
    PYRAMID: Final[str] = "pyramid"
    DETECT_MODE: Final[str] = "detect_mode"
    ILLUMINANT: Final[str] = "illuminant"
    GAMMA: Final[str] = "gamma"
    GRID: Final[str] = "grid"
    METADATA: Final[str] = "metadata"
    PROTECTED: Final[str] = "protected"
    PUBLIC: Final[str] = "public"
    IMPORTED: Final[str] = "imported"


def primary_series(series_names: Sequence[str]) -> str:
    """Return the series a generic viewer should show, and labels attach to.

    Args:
        series_names: Series present in the store.

    Returns:
        ``"rgb"`` when present, otherwise ``"gray"``.

    Raises:
        ValueError: If neither ``rgb`` nor ``gray`` is present.
    """
    for candidate in ("rgb", "gray"):
        if candidate in series_names:
            return candidate
    raise ValueError(f"no primary series among {list(series_names)!r}")


def objmap_path(primary: str) -> str:
    """Return the store-relative path of the objmap label image.

    Readers MUST take this from ``phenotypic.labels.objmap`` rather than
    hard-coding ``rgb/labels/objmap``: when ``rgb`` is empty the primary series
    is ``gray`` and the label lives under it instead.
    """
    return f"{primary}/{LABELS_GROUP}/{OBJMAP_LABEL}"


def build_phenotypic_attributes(
    *,
    image_class: str,
    series_names: Sequence[str],
    pyramid_levels: int,
    metadata_sections: dict[str, dict],
    detect_mode: str | None,
    illuminant: str | None,
    gamma: str | None,
    has_labels: bool = True,
    grid: dict | None = None,
    work_id: str | None = None,
    phenotypic_version: str | None = None,
) -> dict:
    """Build the ``attributes.phenotypic`` block for one store.

    Args:
        image_class: ``"Image"`` or ``"GridImage"`` -- drives loader dispatch.
            Distinct from ``Metadata_ImageType``, which is user-visible schema
            metadata and lives in *metadata_sections*.
        series_names: Series actually written, in canonical order.
        pyramid_levels: Resolved level count, uniform across the store.
        metadata_sections: ``{"protected": …, "public": …, "imported": …}``
            with canonical flat ``Metadata_<Label>`` keys.
        detect_mode: Detection-matrix mode, or ``None``.
        illuminant: Colour illuminant, or ``None``.
        gamma: Gamma encoding name, or ``None``.
        has_labels: Whether the store carries a label image. ``False`` omits
            the ``labels`` key **entirely** rather than emitting an empty
            mapping -- see the note below.
        grid: ``{"nrows": …, "ncols": …, "grid_finder": …}`` for a GridImage.
        work_id: CLI work id, written here at write time and never patched in
            afterwards -- the root ``zarr.json`` is written last, so a post-hoc
            patch would violate the ordering invariant.
        phenotypic_version: Package version; resolved from the installed
            package when omitted.

    Note:
        Metadata values are stored **verbatim and unvalidated**. Real images
        legitimately carry both ``Metadata_PlateNum`` (which
        ``metadata_member_for_header`` does not resolve) and bare public keys
        that ``_remap_legacy_metadata_key`` deliberately preserves. A
        write-time canonicality gate would abort most production runs; the HDF
        writer has none either. See OPEN-QUESTIONS D3.

    Returns:
        A JSON-serialisable mapping.
    """
    import phenotypic

    primary = primary_series(series_names)
    block: dict = {
        PhenotypicAttr.STORE_SCHEMA_VERSION: STORE_SCHEMA_VERSION,
        PhenotypicAttr.PHENOTYPIC_VERSION: (
            phenotypic_version or phenotypic.__version__
        ),
        PhenotypicAttr.IMAGE_CLASS: image_class,
        PhenotypicAttr.SERIES: {name: name for name in series_names},
        PhenotypicAttr.PYRAMID: {
            "levels": int(pyramid_levels),
            "stop_px": PYRAMID_STOP_PX,
            # NOT a literal (ledger GEN-27 / SIMP-16). An earlier draft
            # hard-coded the dict here while DOWNSAMPLE_METHODS' docstring
            # claimed to be the single source -- the constant had exactly one
            # reader, so the two could drift precisely as before.
            "downsample": dict(DOWNSAMPLE_KINDS),
        },
        PhenotypicAttr.DETECT_MODE: detect_mode,
        PhenotypicAttr.ILLUMINANT: illuminant,
        PhenotypicAttr.GAMMA: gamma,
        PhenotypicAttr.METADATA: {
            PhenotypicAttr.PROTECTED: dict(
                metadata_sections.get(PhenotypicAttr.PROTECTED, {})
            ),
            PhenotypicAttr.PUBLIC: dict(
                metadata_sections.get(PhenotypicAttr.PUBLIC, {})
            ),
            PhenotypicAttr.IMPORTED: dict(
                metadata_sections.get(PhenotypicAttr.IMPORTED, {})
            ),
        },
    }
    # Omitted entirely when the store carries no label image. An earlier draft
    # emitted this unconditionally, so a preview store written by
    # `save_intermediate_zarr(layers=("gray",))` DECLARED
    # `labels.objmap = "gray/labels/objmap"` for a group that does not exist
    # -- and `assert_store_conforms` then FileNotFoundError'd walking it. A
    # guard added downstream tested for an EMPTY mapping, which nothing
    # produced; the key has to be absent at the source. Ledger C3.
    if has_labels:
        block[PhenotypicAttr.LABELS] = {OBJMAP_LABEL: objmap_path(primary)}
    if work_id is not None:
        block[PhenotypicAttr.WORK_ID] = work_id
    if grid is not None:
        block[PhenotypicAttr.GRID] = grid
    return block


def read_root_attributes(store_path: Path) -> dict:
    """Read ``<store>/zarr.json``'s ``attributes`` mapping without opening zarr.

    Args:
        store_path: Path to a ``*.ome.zarr`` directory.

    Returns:
        The ``attributes`` mapping.

    Raises:
        FileNotFoundError: If the root ``zarr.json`` does not exist. An
            interrupted write has no root, so this is the normal "absent" path.
        json.JSONDecodeError: If the root is present but unparseable.
    """
    payload = json.loads(
        (Path(store_path) / STORE_ROOT_JSON).read_text(encoding="utf-8")
    )
    return payload.get("attributes", {})


def read_phenotypic_attributes(store_path: Path) -> dict:
    """Read the ``attributes.phenotypic`` block from a store root.

    Args:
        store_path: Path to a ``*.ome.zarr`` directory.

    Returns:
        The ``phenotypic`` block.

    Raises:
        FileNotFoundError: If the root ``zarr.json`` does not exist.
        KeyError: If the root exists but carries no ``phenotypic`` block.
    """
    attributes = read_root_attributes(store_path)
    return attributes[PhenotypicAttr.ROOT]


def require_readable_store(store_path: Path) -> dict:
    """Read ``attributes.phenotypic``, refusing a store this build cannot decode.

    The gate is by **value**, not presence: a future store opened under
    today's semantics is exactly what the 2026-08-19 ruling exists to
    prevent, and presence alone would let it through.

    Every path that decodes store **content** goes through here, so the two
    halves of that guarantee -- the check and its wording -- cannot drift
    apart. :func:`read_phenotypic_attributes` stays ungated for the callers
    that must *classify* a store rather than read it: ``valid_staged_store``
    answers False on a mismatch instead of raising, and that is what routes a
    stale store back to Stage 1 rather than aborting a run.

    Args:
        store_path: Path to a ``*.ome.zarr`` directory.

    Returns:
        The ``phenotypic`` block.

    Raises:
        FileNotFoundError: If the root ``zarr.json`` does not exist.
        KeyError: If the root exists but carries no ``phenotypic`` block.
        ValueError: If ``store_schema_version`` is not this build's.
    """
    block = read_phenotypic_attributes(store_path)
    found = block.get(PhenotypicAttr.STORE_SCHEMA_VERSION)
    if found != STORE_SCHEMA_VERSION:
        raise ValueError(
                f"Cannot read {store_path}: store_schema_version is {found!r}, "
                f"but this build of PhenoTypic reads {STORE_SCHEMA_VERSION}. "
                f"The store was written by a newer PhenoTypic -- upgrade the "
                f"package to read it."
        )
    return block


# ---------------------------------------------------------------------------
# Write-only OME projection (never read back)
# ---------------------------------------------------------------------------


def build_multiscales(
    *,
    series: str,
    level_shapes: Sequence[tuple[int, ...]],
    name: str | None = None,
) -> dict:
    """Build the ``ome.multiscales`` block for one series.

    ``coordinateTransformations`` is derived from the actual level shapes, not
    from ``2 ** n``: odd extents make the two diverge and NGFF requires the
    scale vector to describe the real relationship between levels.

    **Physical resolution is deliberately not projected.** Scale vectors are
    pure level ratios and ``unit`` is omitted, which §2.1 permits.

    Args:
        series: Series name, selecting the axes.
        level_shapes: Shape per level, level 0 first.
        name: ``multiscales[].name``, typically ``Metadata_ImageName``.

    Returns:
        ``{"multiscales": [ … ]}``.
    """
    names = axes_for(series)
    axes = [{"name": axis, "type": AXIS_TYPES[axis]} for axis in names]

    base = tuple(level_shapes[0])
    datasets = [
        {
            "path": str(index),
            "coordinateTransformations": [
                {"type": "scale", "scale": level_scale_vector(base, tuple(shape))}
            ],
        }
        for index, shape in enumerate(level_shapes)
    ]

    kind = "label" if series == OBJMAP_LABEL else "image"
    multiscale: dict = {
        "axes": axes,
        "datasets": datasets,
        # §2.4 SHOULD: name the downscaling method and describe it. We compute
        # exactly these values, so emitting them costs nothing -- and driving
        # both from one constant is what actually stops the public record from
        # diverging from the private
        # `attributes.phenotypic.pyramid.downsample` one, which reads
        # DOWNSAMPLE_KINDS off the same dict.
        "type": DOWNSAMPLE_METHODS[kind][0],
        "metadata": {"description": DOWNSAMPLE_METHODS[kind][1]},
    }
    if name is not None:
        multiscale["name"] = name
    return {"multiscales": [multiscale]}

# NOTE (ledger ALGO-R2B-16): 2.4 -- "Each 'multiscales' dictionary SHOULD
# contain the field 'name'." Pass `name` at the LABEL call site too
# (phase-2 Task 2.2), not just the three image series, or the label block
# silently skips a SHOULD that every sibling honours.


#: Per-channel display colours for the ``rgb`` series.
_RGB_CHANNEL_COLORS: Final[tuple[tuple[str, str], ...]] = (
    ("R", "FF0000"),
    ("G", "00FF00"),
    ("B", "0000FF"),
)


def build_omero(
    *,
    series: str,
    dtype: "np.dtype",
    bit_depth: int,
    name: str | None = None,
) -> dict:
    """Build the ``ome.omero`` rendering block for one image series.

    NGFF is conditionally strict here: if ``omero`` is present at all, every
    channel MUST carry a 6-hex-digit ``color`` and a ``window`` containing all
    four of ``min``, ``max``, ``start``, ``end``. A partial projection fails the
    conformance gate on the first store written, so this emits the block
    completely or the caller omits it entirely.

    ``omero`` is never emitted on a label group, and never on a **float**
    series. Both ``gray`` and ``detect_mat`` are float, typically in
    ``[0, 1]``, so a ``2**bit_depth - 1`` window over them puts ``[0, 255]``
    across ``[0, 1]`` data and any viewer honouring ``omero`` renders them
    near-black. ``gray`` is the primary series in every rgb-less store, which
    makes it the worst place for that defect. In practice ``rgb`` is the only
    series that carries a block, and an rgb-less store carries none -- which is
    fine: §2.5 makes ``omero`` optional, and the whole-or-nothing rule is per
    group. This supersedes the spec's §2.2, which applies the window to every
    series.

    The test is the **dtype**, not the series name. Keying on dtype means that
    if the deferred integer conversion ever lands, the affected series get their
    block back automatically.

    Args:
        series: ``"rgb"``, ``"gray"``, or ``"detect_mat"``.
        dtype: Level-0 dtype. Float dtypes get no block.
        bit_depth: Source bit depth; ``max``/``end`` are ``2**bit_depth - 1``.
        name: ``omero.name``, typically ``Metadata_ImageName``.

    Returns:
        ``{"omero": {"channels": [ … ]}}``, or ``{}`` for a float series.
    """
    if np.issubdtype(dtype, np.floating):
        return {}
    ceiling = (2 ** int(bit_depth)) - 1
    palette = (
        _RGB_CHANNEL_COLORS if series == "rgb" else ((series, "FFFFFF"),)
    )
    channels = [
        {
            "label": label,
            "color": color,
            "active": True,
            "family": "linear",
            "coefficient": 1,
            "inverted": False,
            "window": {"min": 0, "max": ceiling, "start": 0, "end": ceiling},
        }
        for label, color in palette
    ]
    block: dict = {"channels": channels}
    if name is not None:
        block["name"] = name
    return {"omero": block}


def build_image_label() -> dict:
    """Build the ``ome.image-label`` block for the objmap label image.

    Always emitted: ``label.schema``'s **``properties.ome.required``** is
    ``["image-label", "version"]`` -- note the path. What is required is
    ``ome.version``, *not* a ``version`` inside the ``image-label`` object.
    The inner ``image-label.version`` emitted below is **also** specified --
    NGFF 0.5 2.6: *"That image-label object SHOULD contain the following keys:
    first, a colors key... Second, a version key, whose value MUST be a string
    specifying the version of the OME-Zarr image-label schema."* Both are
    emitted; neither is redundant (ledger **ALGO-6**, corrected by **ALGO-12**
    -- an earlier draft called the inner one "a 0.4-ism", which invites a
    future reader to delete a documented SHOULD).

    **Takes no arguments, deliberately.** ``colors`` carries only the
    transparent background entry rather than one entry per unique label value.
    ``$defs/image-label`` sets no ``required`` list, so ``colors`` is optional
    and a background-only entry conforms. A per-value palette would be a
    function of the array contents, which is what makes it able to go stale;
    this one cannot. Nothing in PhenoTypic reads ``colors`` (the GUI colourises
    through ``skimage.color.label2rgb``); only the conformance gate and external
    viewers do, and external viewers fall back to their own palette. This
    supersedes the spec's §2.3.

    ``properties`` is deliberately not emitted -- parquet remains the only
    measurement surface (locked decision #10).

    Returns:
        ``{"image-label": {…}}``, constant size regardless of colony count.
    """
    return {
        "image-label": {
            "version": NGFF_VERSION,
            "source": {"image": "../../"},
            "colors": [{"label-value": 0, "rgba": [0, 0, 0, 0]}],
        }
    }


def _ome_xml_modules(metadata_sections: dict[str, dict]) -> dict[str, dict]:
    """Group metadata headers by REMBI module for the OME-XML annotation block.

    Note the API: ``header_to_module()`` takes **no arguments** and returns the
    whole ``{header: REMBI_MODULE}`` mapping (``schema/_rembi.py:29``, lru-cached).
    """
    # An earlier draft called it as `header_to_module(key)`, which raises
    # TypeError on the first key -- and because `build_ome_xml` caught
    # everything and returned None, that one-line mistake would have made every
    # store ship with no OME/ group at all, silently. That is why the failure
    # path is now fatal.
    from phenotypic.schema import header_to_module

    mapping = header_to_module()
    grouped: dict[str, dict] = {}
    for section, payload in metadata_sections.items():
        for key, value in payload.items():
            module = mapping.get(key) or section
            # `.value`, NOT str(). REMBI_MODULE is a str-mixin Enum with no
            # __str__ override, so str(REMBI_MODULE.BIOSAMPLE) is the Python
            # repr 'REMBI_MODULE.BIOSAMPLE', not 'Biosample' -- which would
            # ship a Python-internal name as the MapAnnotation Namespace, mixed
            # with plain section fallbacks like "imported". A legal anyURI, so
            # ome.xsd cannot catch it. Ledger ALGO-10.
            grouped.setdefault(getattr(module, "value", module), {})[key] = value
    return grouped


#: Characters XML 1.0 permits at all, per the ``Char`` production. Note **1.0,
#: not 1.1**: `#x7F` (DEL) and the C1 block `#x80-#x9F` are *discouraged* by
#: 1.0 2.2 but NOT forbidden, and they sit inside `[#x20-#xD7FF]` -- only XML
#: 1.1 restricts them, via `RestrictedChar`. Do not "tighten" this to strip
#: them; that would silently drop legitimate MakerNote bytes. `#xFFFD` is
#: deliberately RETAINED -- it is the top of `[#xE000-#xFFFD]` and it is what
#: `decode(errors="replace")` emits, so stripping it would delete the very
#: marks that record a repair.
#: ``#x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]``.
#: Everything else is forbidden OUTRIGHT -- not even as a character reference --
#: so no amount of escaping rescues it.
_XML_FORBIDDEN = _re.compile(
    "[^\u0009\u000A\u000D\u0020-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]"
)


def _xml_text(value: object) -> str:
    """Coerce to a string containing only characters XML 1.0 permits.

    Sanitizes rather than raises, deliberately. A NUL-padded EXIF string is
    *legitimate* camera input, and OME-XML failure is fatal by user ruling
    (ALGO-1) -- so raising here would abort a real run over a ``MakerNote``.
    Dropping the offending code points loses nothing a reader could have used.

    Args:
        value: Any metadata key or value.

    Returns:
        The string form with forbidden code points removed.
    """
    return _XML_FORBIDDEN.sub("", str(value))


#: numpy dtype kind+itemsize -> OME-XML PixelType. A closed enumeration in
#: ``ome.xsd``; an unmapped dtype is a hard error, not a fallback.
#:
#: Deliberately a SUBSET: ``bit`` (numpy ``b1``), ``complex`` (``c8``), and
#: ``double-complex`` (``c16``) are legal PixelType values but unreachable --
#: this function only ever sees rgb/gray/detect_mat. ``objmap`` never reaches
#: it because **labels get no ``<Image>`` element at all** -- not because its
#: dtype is unmappable (``uint16`` maps fine); stating the real reason keeps
#: someone from "fixing" a non-problem. ``int64``/``uint64``/``float16`` have NO OME equivalent at all,
#: so raising on them is correct, not a gap. Do not "complete" this map.
_OME_PIXEL_TYPES: Final[dict[str, str]] = {
    "u1": "uint8", "u2": "uint16", "u4": "uint32",
    "i1": "int8", "i2": "int16", "i4": "int32",
    "f4": "float", "f8": "double",
}


def _ome_pixel_type(dtype: "np.dtype") -> str:
    """Map a numpy dtype to an OME-XML ``PixelType``.

    Raises:
        ValueError: If the dtype has no OME equivalent. Loud by design -- a
            silent fallback here is what let an invalid document ship.
    """
    key = np.dtype(dtype).str[1:]
    if key not in _OME_PIXEL_TYPES:
        raise ValueError(
            f"no OME PixelType for dtype {np.dtype(dtype)!r}; "
            f"supported: {sorted(_OME_PIXEL_TYPES.values())}"
        )
    return _OME_PIXEL_TYPES[key]


def build_ome_xml(
    *,
    series_names: Sequence[str],
    series_shapes: dict[str, tuple[int, ...]],
    series_dtypes: dict[str, "np.dtype"],
    metadata_sections: dict[str, dict],
) -> str:
    """Build the ``MetadataOnly`` OME-XML document. **Raises on failure.**

    §2.2.3 makes this a conditional MUST: the document *"MUST adhere to the
    OME-XML specification but MUST use ``<MetadataOnly/>`` elements"*. An
    earlier draft emitted ``<Pixels />`` with no attributes and no
    ``<MetadataOnly/>`` child, and put ``<M>`` entries directly under
    ``<MapAnnotation>`` instead of inside a ``<Value>`` -- all three invalid
    against ``ome.xsd`` 2016-06, so every store's ``METADATA.ome.xml`` would
    have been rejected by exactly the Bio-Formats/OME tooling that
    ``bioformats2raw.layout: 3`` exists to serve.

    **Failure is fatal, deliberately** (user ruling; OPEN-QUESTIONS
    **PRE-G2** / **ALGO-3**). This is string formatting over already-validated
    data, so the realistic failure modes are an unmapped dtype and a genuine
    bug -- and a bug is exactly what the old ``except Exception: return None``
    hid: it would have swallowed a one-line API mistake and shipped **every**
    store with no ``OME/`` group, silently and forever. The spec's
    "consecutive-integer fallback" is withdrawn with it: keeping named groups
    while dropping ``series`` satisfies neither arm of §2.2.3 and is strictly
    less conformant than either.

    Args:
        series_names: Series in canonical order; one ``<Image>`` each.
        series_shapes: Level-0 shape per series, for the ``Size*`` attributes.
        series_dtypes: Level-0 dtype per series, for ``Type``.
        metadata_sections: Metadata to project as structured annotations.

    Returns:
        A conformant OME-XML document.

    Raises:
        ValueError: On an unmapped dtype.
        KeyError: If a named series has no shape or dtype entry.
    """
    from xml.sax.saxutils import escape, quoteattr

    def _image(index: int, series: str) -> str:
        shape = series_shapes[series]
        size_c = shape[0] if len(shape) == 3 else 1
        size_y, size_x = shape[-2], shape[-1]
        return (
            f'    <Image ID="Image:{index}" Name={quoteattr(series)}>\n'
            f'      <Pixels ID="Pixels:{index}" DimensionOrder="XYZCT" '
            f'Type="{_ome_pixel_type(series_dtypes[series])}" '
            f'SizeX="{size_x}" SizeY="{size_y}" SizeZ="1" '
            f'SizeC="{size_c}" SizeT="1">\n'
            f"        <MetadataOnly/>\n"
            f"      </Pixels>\n"
            f"    </Image>"
        )

    def _annotation(index: int, module: str, payload: dict) -> str:
        entries = "\n".join(
            f"          <M K={quoteattr(_xml_text(key))}>"
            f"{escape(_xml_text(value))}</M>"
            for key, value in sorted(payload.items())
        )
        return (
            f'    <MapAnnotation ID="Annotation:{index}" '
            # `module` is a REMBI_MODULE.value or one of the three literal
            # section names, so it cannot carry a control character -- but that
            # provenance is 200 lines away in `_ome_xml_modules`, and the
            # asymmetry with the sanitized K/text above reads as an oversight.
            # Wrapped for symmetry, at zero cost (ledger algo-r3).
            f"Namespace={quoteattr(_xml_text(module))}>\n"
            f"      <Value>\n{entries}\n      </Value>\n"
            f"    </MapAnnotation>"
        )

    modules = _ome_xml_modules(metadata_sections)
    images = "\n".join(
        _image(index, series) for index, series in enumerate(series_names)
    )
    annotations = "\n".join(
        _annotation(index, module, payload)
        for index, (module, payload) in enumerate(sorted(modules.items()))
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">\n'
        f"{images}\n"
        "  <StructuredAnnotations>\n"
        f"{annotations}\n"
        "  </StructuredAnnotations>\n"
        "</OME>\n"
    )


# ---------------------------------------------------------------------------
# Commit protocol: uuid part, move-aside promote, orphan sweep
# ---------------------------------------------------------------------------

PART_SUFFIX: Final[str] = ".part"
TRASH_SUFFIX: Final[str] = ".trash"

#: Retry budget for the two move-aside renames. On Windows a rename fails with
#: ERROR_SHARING_VIOLATION while any of the store's ~40 files is held open by a
#: running GUI, an antivirus scan, or the search indexer. Same shape as
#: ``_open_hdf_with_recovery`` in :mod:`phenotypic.sdk_.hdf_`.
PROMOTE_RETRY_ATTEMPTS: Final[int] = 5
PROMOTE_RETRY_BASE_SECONDS: Final[float] = 0.1


def _resolve_durability(override: bool | None) -> tuple[bool, str]:
    """Return ``(enabled, reason)`` for the durability decision.

    One function so the flag and the sentence describing it cannot drift.

    Args:
        override: ``--durable-writes`` / ``--no-durable-writes``, or ``None``.

    Returns:
        ``(True, "SLURM")`` / ``(False, "local")`` / ``(True, "--durable-writes")``
        / ``(False, "--no-durable-writes")``.
    """
    if override is True:
        return True, "--durable-writes"
    if override is False:
        return False, "--no-durable-writes"
    on_slurm = bool(
        os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_PER_TASK")
    )
    return (True, "SLURM") if on_slurm else (False, "local")


def durable_writes_enabled(override: bool | None = None) -> bool:
    """Resolve whether the promote fsyncs before renaming.

    ``write()`` returns once data is in the page cache. Without ``fsync`` the
    kernel may flush the root ``zarr.json`` *before* the chunk data it
    describes, so a node crash can leave a store that passes
    :func:`valid_staged_store` -- metadata parses, shapes agree -- while
    reading ``fill_value``. That is silent wrong data, not a visible failure,
    and no amount of metadata validation catches it.

    The dominant failure mode does not need it: a SLURM timeout kills the
    process, and the kernel survives and flushes normally. ``fsync`` buys
    protection only against node loss, power failure, and filesystem crash --
    which is exactly what a cluster job is exposed to and a laptop run is not.

    Args:
        override: ``--durable-writes`` / ``--no-durable-writes``, or ``None``
            to auto-detect.

    Returns:
        ``True`` when the promote should fsync.

    Note:
        This checks ``SLURM_JOB_ID`` **as well as** ``SLURM_CPUS_PER_TASK``.
        ``resolve_worker_count`` (``_cli_utils.py:65-72``) reads only the
        latter, so this is deliberately broader -- not "exactly as" that helper
        does, which is what the spec's §3.7 claims. A job that sets
        ``SLURM_JOB_ID`` without a per-task CPU count still gets durable writes.
    """
    return _resolve_durability(override)[0]


def describe_durability(override: bool | None = None) -> str:
    """One-line description of the resolved durability mode, for the start log.

    The same command carries different guarantees in different places, which is
    a genuinely surprising thing to debug. Logging the resolved mode at run
    start is a required mitigation, not a nicety.

    Shares :func:`_resolve_durability` with :func:`durable_writes_enabled`, so
    the flag and the sentence describing it cannot drift apart.
    """
    enabled, reason = _resolve_durability(override)
    return f"durable writes: {'on' if enabled else 'off'} ({reason})"


def long_path(path: Path) -> str:
    """Return an OS-appropriate path string, ``\\\\?\\``-prefixed on Windows.

    An output root, dataset name, and image stem plus a store-internal path can
    exceed Windows' 260-character ``MAX_PATH``. The ``"."`` chunk-key separator
    keeps a chunk key to one segment; this prefix covers the rest.

    **Apply it at every filesystem entry point, not only array I/O.** Route
    every path through this helper so a new site cannot forget.

    On POSIX this is a true passthrough. ``resolve()`` is confined to the
    Windows branch, which is the only one that needs it: ``\\\\?\\`` disables
    path normalization, so the prefix is only legal on an already fully
    qualified path. Resolving on POSIX too would rewrite any symlinked root --
    macOS' ``/var`` -> ``/private/var``, a symlinked scratch or project mount --
    and hand the caller back a path it never passed in.
    """
    if os.name != "nt":
        return str(path)
    text = str(Path(path).resolve())
    return text if text.startswith("\\\\?\\") else "\\\\?\\" + text


def new_part_path(final: Path) -> Path:
    """Return a fresh, uuid-suffixed ``.part`` sibling of *final*.

    The uuid -- matching the ``attempt_id = uuid4().hex`` convention already
    used in ``_cli_staged_strategy.py`` (lines 148, 192, 225, 359) -- is what
    keeps two concurrent writers from interleaving chunks into one directory.
    It is NOT what makes the promote itself benign; that is the retry loop in
    :func:`promote_store`. An un-suffixed ``.part`` would let two concurrent
    SLURM tasks interleave chunks into one directory and produce a store that
    *validates*. A PID is not enough: PIDs are reused.
    """
    final = Path(final)
    return final.parent / f".{final.name}.{uuid4().hex}{PART_SUFFIX}"


def _fsync_path(path: Path) -> None:
    """``fsync`` one already-existing file or directory."""
    handle = os.open(long_path(path), os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def fsync_tree(root: Path) -> None:
    """``fsync`` every regular file under *root*, then **every** directory.

    Both halves matter. On POSIX a durable file does **not** imply a durable
    directory entry, so flushing files plus the root alone would leave the
    nested ``gray/0/`` and ``rgb/labels/objmap/0/`` dirents unflushed -- exactly
    the silent wrong-data mode §3.7 exists to close. Directories are flushed
    deepest-first so a parent's entry is never made durable before the child it
    points at.

    All directory flushes are POSIX-guarded: Windows cannot open a directory
    handle for flushing and relies on NTFS journaling instead.
    """
    root = Path(root)
    directories: list[Path] = [root]
    for path in sorted(root.rglob("*")):
        if path.is_file():
            _fsync_path(path)
        elif path.is_dir():
            directories.append(path)
    if os.name == "posix":
        for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
            _fsync_path(directory)


#: errno / winerror values worth retrying. Everything else fails fast: retrying
#: a genuine ENOSPC five times with exponential backoff burns 3.1 s per image
#: before surfacing, which at 10k images is an hour of sleeping.
_RETRYABLE_WINERROR: Final[frozenset[int]] = frozenset(
    {32, 33}
)  # SHARING_VIOLATION, LOCK_VIOLATION


def _is_retryable(exc: OSError) -> bool:
    """Whether *exc* is a transient contention error rather than a hard failure.

    Windows refuses to rename a directory while any file inside it is held open
    (``ERROR_SHARING_VIOLATION``); with ~40 files per store instead of one
    ``.h5``, that exposure is 40x larger. On POSIX, ``ENOTEMPTY``/``ENOENT`` on
    the target mean a concurrent promoter moved under us, which the retry loop
    resolves by re-evaluating.
    """
    if getattr(exc, "winerror", None) in _RETRYABLE_WINERROR:
        return True
    return exc.errno in {errno.ENOTEMPTY, errno.ENOENT, errno.EEXIST}


def promote_store(part: Path, final: Path, *, fsync: bool) -> Path:
    """Atomically promote a fully written ``.part`` directory to *final*.

    The caller is responsible for the write **order** inside *part*: all arrays
    and chunks first, then ``OME/zarr.json``, then the root ``zarr.json`` last.
    An interrupted store therefore has no valid root and reads as absent. This
    function does **not** write the root ``zarr.json`` itself.

    The move-aside is mandatory, not an optimization: ``os.replace`` onto a
    non-empty directory raises ``OSError`` (``ENOTEMPTY``) on POSIX, and on
    Windows ``MoveFileEx``'s ``MOVEFILE_REPLACE_EXISTING`` cannot name a
    directory at all.

    The whole ``exists -> move-aside -> replace`` sequence sits inside one
    retry loop and re-evaluates existence on every attempt. That is what makes
    duplicate execution benign: a uuid ``.part`` prevents two writers
    *interleaving chunks*, but it does nothing for the promote itself, where a
    check-then-act done once lets writer B skip the move-aside because A had not
    yet renamed, then hit ``ENOTEMPTY`` on a now-non-empty target.

    On failure after a successful move-aside, the previous store is **rolled
    back** into place before retrying or raising. Deleting it in a ``finally``
    would leave no copy at any path -- a data-loss mode the single-file HDF
    rename never had, since a failed ``os.replace(tmp, final)`` left ``final``
    untouched.

    Known weakening versus the single-file rename: the two renames are still not
    one atomic step, so a crash *between* them (as opposed to a raised error)
    leaves the image absent plus an orphaned ``.trash``. Both are recoverable --
    absence reclassifies to the rebuilding stage, and :func:`sweep_orphan_parts`
    clears the leftovers.

    Args:
        part: Fully written ``.part`` directory.
        final: Target store path.
        fsync: Whether to flush *part* before renaming
            (see :func:`durable_writes_enabled`).

    Returns:
        *final*.
    """
    part, final = Path(part), Path(final)
    if fsync:
        fsync_tree(part)

    trash = final.parent / f"{part.name[: -len(PART_SUFFIX)]}{TRASH_SUFFIX}"
    last: OSError | None = None
    for attempt in range(PROMOTE_RETRY_ATTEMPTS):
        moved_aside = False
        try:
            # Re-evaluate existence EVERY attempt. A concurrent promoter can
            # create or remove `final` between the check and either rename, so
            # a check-then-act done once outside the loop turns a benign
            # duplicate execution into a hard failure.
            if final.exists():
                os.replace(long_path(final), long_path(trash))
                moved_aside = True
            os.replace(long_path(part), long_path(final))
        except OSError as exc:
            last = exc
            if not _is_retryable(exc):
                if moved_aside and trash.exists() and not final.exists():
                    os.replace(long_path(trash), long_path(final))
                raise
            if moved_aside and trash.exists() and not final.exists():
                # Roll back. Without this the previous store is already in
                # `trash` and is about to be deleted, leaving NO copy at any
                # path -- a data-loss mode the single-file HDF rename never had
                # (a failed os.replace(tmp, final) left `final` untouched).
                os.replace(long_path(trash), long_path(final))
            time.sleep(PROMOTE_RETRY_BASE_SECONDS * (2**attempt))
            continue
        # Success: only now is the previous store safe to discard.
        if trash.exists():
            shutil.rmtree(long_path(trash), ignore_errors=True)
        if fsync and os.name == "posix":
            # The rename itself is a directory-entry change in final.parent; a
            # durable store whose dirent is not durable is still a lost store.
            _fsync_path(final.parent)
        return final
    assert last is not None
    raise last


#: A `.part` younger than this may still be being written. The sweep never
#: touches one. Generous by design: the cost of skipping a genuine orphan is one
#: stale directory until the next run; the cost of deleting a live one is a
#: destroyed in-flight image.
SWEEP_MIN_AGE_SECONDS: Final[float] = 6 * 60 * 60


def discard_parts_for(final: Path) -> int:
    """Remove every ``.part`` sibling belonging to one target store.

    Shares the naming convention with :func:`new_part_path` and
    :func:`sweep_orphan_parts` so no caller re-encodes it. The CLI's
    save-failure path would otherwise hand-roll the dot-prefix + uuid + suffix
    glob outside this module (ledger **SIMP-6**).

    Scoped to *one* store by anchoring the glob on ``final.name``: a sibling
    store's in-flight ``.part`` belongs to another writer and is not ours to
    remove.

    Args:
        final: The target store path whose parts should be discarded.

    Returns:
        Number of directories removed.
    """
    final = Path(final)
    removed = 0
    for path in final.parent.glob(f".{final.name}.*{PART_SUFFIX}"):
        if path.is_dir():
            shutil.rmtree(long_path(path), ignore_errors=True)
            removed += 1
    return removed


def sweep_orphan_parts(
    results_root: Path, *, min_age_seconds: float = SWEEP_MIN_AGE_SECONDS
) -> int:
    """Remove *stale* orphaned ``.part`` / ``.trash`` directories.

    **A uuid identifies the attempt, not whether its process is alive.** The
    staged SLURM engine explicitly assumes stale workers can still be running --
    that is what ``assert_active_epoch`` exists for -- and under an array the
    tasks share one output root and start at different times. A sweep with no
    liveness signal would ``rmtree`` the ``.part`` directories its siblings are
    actively filling, which is the same defect a PID-based sweep has.

    Two guards, both required:

    * **age**: only directories whose mtime is older than *min_age_seconds* are
      removed;
    * **placement**: the caller must run this from the controller before any
      worker is submitted, not from each worker's start-up (see Phase 3).

    The scan is bounded to ``results/<dataset>/zarr/`` rather than recursive:
    ``rglob`` would descend into every store, which is the same ~400k-stat
    pathology the spec flags for the GUI's discovery path.

    Args:
        results_root: The run's ``results/`` directory.
        min_age_seconds: Minimum age before a leftover is considered orphaned.

    Returns:
        Number of directories removed.
    """
    removed = 0
    root = Path(results_root)
    if not root.is_dir():
        return 0
    cutoff = time.time() - min_age_seconds
    for dataset_dir in root.iterdir():
        zarr_dir = dataset_dir / "zarr"
        if not zarr_dir.is_dir():
            continue
        for path in zarr_dir.iterdir():
            if not path.is_dir():
                continue
            if not (
                path.name.endswith(PART_SUFFIX) or path.name.endswith(TRASH_SUFFIX)
            ):
                continue
            if STORE_SUFFIX not in path.name:
                continue
            if os.stat(long_path(path)).st_mtime > cutoff:
                continue  # may still be in flight
            shutil.rmtree(long_path(path), ignore_errors=True)
            removed += 1
    return removed


# ---------------------------------------------------------------------------
# Resume validity
# ---------------------------------------------------------------------------


def store_level0_shape(store_path: Path, member_path: str) -> tuple[int, ...] | None:
    """Return the level-0 shape of one member array, or ``None`` if absent.

    Args:
        store_path: Store root.
        member_path: Store-relative group path, e.g. ``"gray"`` or
            ``"rgb/labels/objmap"``.

    Returns:
        The level-0 array shape, or ``None`` when the level-0 array is missing.
    """
    import zarr

    level0 = Path(store_path) / member_path / "0"
    if not level0.is_dir():
        return None
    return tuple(zarr.open_array(store=long_path(level0), mode="r").shape)


def valid_staged_store(path: Path) -> bool:
    """Return whether *path* holds the image layers Stage 2 requires.

    Mirrors ``valid_staged_hdf`` case for case:

    * the root ``zarr.json`` parses and carries ``store_schema_version``;
    * every entry in ``phenotypic.series`` **and** ``phenotypic.labels`` opens
      as a Zarr array group -- objmap included, which Stage 1's zeros write
      guarantees;
    * level-0 ``(y, x)`` extents agree across all of them and are non-zero. A
      zero-size Zarr array is legal and must not pass.

    The exception set is the HDF version's ``(OSError, TypeError,
    ValueError)`` **plus ``KeyError``** -- which the attribute lookups need and
    the HDF version did not -- **plus ``AttributeError``**. The root
    ``zarr.json`` is arbitrary JSON written by anyone, so ``phenotypic``,
    ``phenotypic.series``, and ``phenotypic.labels`` can each come back as a
    list rather than a mapping (another tool's store, or a future schema); the
    ``.get``/``.values()`` calls below then raise ``AttributeError``, which is
    a rejected store, not a crash in resume classification.

    It does **not** need ``zarr.errors.BaseZarrError``. The spec's §3.6 argues
    the opposite ("none of zarr's error types are ``ValueError`` subclasses");
    that is inverted. ``BaseZarrError`` inherits **directly from
    ``ValueError``** (https://zarr.readthedocs.io/en/stable/api/zarr/errors/),
    as do ``MetadataValidationError`` and every other zarr error except the four
    ``IndexError`` ones, none of which this function can raise.
    ``json.JSONDecodeError`` is likewise a ``ValueError`` and
    ``FileNotFoundError`` an ``OSError``, so both are already covered. Keeping
    the shorter tuple also avoids importing ``zarr.errors`` in a function the
    resume planner calls once per image.

    Args:
        path: Candidate ``*.ome.zarr`` directory.

    Returns:
        ``True`` only for a store Stage 2 can consume.
    """
    try:
        store = Path(path)
        if not store.is_dir():
            return False
        block = read_phenotypic_attributes(store)
        # By VALUE, not presence (user ruling, 2026-08-19). A presence-only
        # check would let a store written by a future v4 be read under v3
        # semantics with no error at all. `.get` covers absence in the same
        # comparison. This predicate RETURNS FALSE rather than raising -- the
        # explicit "written by a newer PhenoTypic" error belongs in the loader
        # (Phase 2 `load_zarr`), which is the path a user actually invokes.
        if block.get(PhenotypicAttr.STORE_SCHEMA_VERSION) != STORE_SCHEMA_VERSION:
            return False
        members = [
            *block[PhenotypicAttr.SERIES].values(),
            # `.get`: a label-less store OMITS the key (Task 1.3, ledger C3).
            # This is a validity predicate -- it must RETURN FALSE on a store it
            # does not accept, never raise. Indexing would make a label-less
            # store a KeyError propagating out of resume classification and
            # migration, both of which call this to decide what to do next.
            *block.get(PhenotypicAttr.LABELS, {}).values(),
        ]
        if not members:
            return False
        shapes: list[tuple[int, ...]] = []
        for member in members:
            shape = store_level0_shape(store, member)
            if shape is None:
                return False
            shapes.append(shape)
        spatial = [shape[-2:] for shape in shapes]
        if any(len(yx) < 2 or yx[0] <= 0 or yx[1] <= 0 for yx in spatial):
            return False
        return all(yx == spatial[0] for yx in spatial[1:])
    except (AttributeError, OSError, KeyError, TypeError, ValueError):
        return False
