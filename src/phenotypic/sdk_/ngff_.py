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
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Mapping, NamedTuple, Sequence
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

from ._atomic_io import CommitGuard, publication_commit

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

#: Embedded object-measurement table namespace. The sub-schema is versioned
#: independently from the store group/array schema, which remains version 3.
TABLES_GROUP: Final[str] = "tables"
MEASUREMENT_TABLE_GROUP: Final[str] = "measurements"
MEASUREMENT_TABLE_FILENAME: Final[str] = "table.parquet"
MEASUREMENT_TABLE_RELATIVE_PATH: Final[Path] = Path(
    TABLES_GROUP, MEASUREMENT_TABLE_GROUP, MEASUREMENT_TABLE_FILENAME
)
MEASUREMENT_TABLE_SCHEMA_VERSION: Final[int] = 1


class EmbeddedMeasurementParquetMetadataKeys(NamedTuple):
    """Stable Parquet key/value metadata names for join provenance."""

    JOIN_STATUS: str = "phenotypic.join.status"
    JOIN_KIND: str = "phenotypic.join.kind"
    JOIN_LEFT: str = "phenotypic.join.left"
    JOIN_RIGHT: str = "phenotypic.join.right"
    JOIN_KEYS: str = "phenotypic.join.keys"
    METADATA_SNAPSHOT_SHA256: str = "phenotypic.metadata.snapshot_sha256"
    MEASUREMENT_COLUMNS: str = "phenotypic.measurement_columns"


EMBEDDED_MEASUREMENT_PARQUET_METADATA_KEYS: Final = (
    EmbeddedMeasurementParquetMetadataKeys()
)

#: Canonical series order. ``rgb`` is omitted from a store when empty; the
#: remaining names keep this relative order.
SERIES_ORDER: Final[tuple[str, str, str]] = ("rgb", "gray", "detect_mat")

AXES_3D: Final[tuple[str, str, str]] = ("c", "y", "x")
AXES_2D: Final[tuple[str, str]] = ("y", "x")

#: NGFF axis ``type`` per dimension name.
AXIS_TYPES: Final[dict[str, str]] = {
    "c": "channel",
    "y": "space",
    "x": "space",
}

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
    level0: tuple[int, ...], level_index: int
) -> list[float]:
    """Per-axis sampling factor after repeated 2x spatial reductions.

    Ceil-halving changes the stored array extent but not the sampling operation:
    an odd 1025-pixel axis becomes 513 pixels after one 2x reduction, so its
    sampling factor is 2 rather than the shape ratio ``1025 / 513``. An axis
    saturates once it reaches one sample; leading channel axes are never sampled.

    Args:
        level0: Level-0 shape.
        level_index: Zero-based pyramid level.

    Returns:
        One float per axis, in axis order. Any leading channel axis is 1.0.
    """
    leading = [1.0] * max(0, len(level0) - 2)
    spatial = [
        float(2 ** min(level_index, (int(size) - 1).bit_length()))
        for size in level0[-2:]
    ]
    return [*leading, *spatial]


def level_coordinate_transformations(
    level0: tuple[int, ...], level_index: int
) -> list[dict]:
    """Map a pyramid level's sample centers into level-0 coordinates.

    A repeated 2x block reduction with total sampling factor ``scale`` maps a
    level coordinate ``u`` to ``scale * u + (scale - 1) / 2``. Translation is
    zero for channel axes and spatial axes whose effective scale is 1, and is
    omitted entirely when the vector is all zero.

    Args:
        level0: Level-0 shape.
        level_index: Zero-based pyramid level.

    Returns:
        A scale transformation followed by a nonzero translation, if any.
    """
    scale = level_scale_vector(level0, level_index)
    leading = [0.0] * max(0, len(level0) - 2)
    translation = [*leading, *((factor - 1.0) / 2.0 for factor in scale[-2:])]
    transformations = [{"type": "scale", "scale": scale}]
    if any(offset != 0.0 for offset in translation):
        transformations.append(
            {"type": "translation", "translation": translation}
        )
    return transformations


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
        chunk[len(shape) - 2 + axis]
        if extent < CHUNK_YX[axis]
        else SHARD_YX[axis]
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
    PROVENANCE: Final[str] = "provenance"
    TABLES: Final[str] = "tables"
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
    image_class: str | None,
    series_names: Sequence[str],
    pyramid_levels: int,
    metadata_sections: dict[str, dict],
    detect_mode: str | None,
    illuminant: str | None,
    gamma: str | None,
    has_labels: bool = True,
    grid: dict | None = None,
    work_id: str | None = None,
    provenance: dict | None = None,
    phenotypic_version: str | None = None,
) -> dict:
    """Build the ``attributes.phenotypic`` block for one store.

    Args:
        image_class: ``"Image"`` or ``"GridImage"`` -- drives loader dispatch.
            Distinct from ``Metadata_ImageType``, which is user-visible schema
            metadata and lives in *metadata_sections*. ``None`` omits the key
            entirely, which is what marks a store as **not** a run bundle:
            :meth:`Image.load_zarr` refuses a store without it. Only the
            ``--mode process`` writer passes ``None``.
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
        provenance: Versioned image-operation journal, when owned by an Image.
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
    # Absence, not a null: `load_zarr`'s guard tests key membership, so
    # writing `image_class: None` would defeat it. Key insertion order puts
    # `image_class` after `metadata` rather than between `phenotypic_version`
    # and `series`; nothing reads the block positionally -- it is JSON.
    if image_class is not None:
        block[PhenotypicAttr.IMAGE_CLASS] = image_class
    if provenance is not None:
        block[PhenotypicAttr.PROVENANCE] = provenance
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
# Reading an arbitrary NGFF store as plain pixels (spec 4)
# ---------------------------------------------------------------------------


def project_ngff_axes(
    axes: Sequence[Mapping[str, object]],
    shape: Sequence[int],
    *,
    t: int | None = None,
    z: int | None = None,
    c: int | None = None,
) -> tuple[tuple[int | slice, ...], bool]:
    """Map an NGFF array's axes onto PhenoTypic's 2-D image model.

    ``Image`` is 2-D, optionally with three colour channels. NGFF permits 2 to
    5 axes. This is the total mapping between them, and it **refuses rather
    than guesses**: silently reading ``t=0`` of a timelapse, or channel 0 of a
    five-channel acquisition, yields a plausible image and a wrong result that
    nothing downstream can detect.

    Args:
        axes: The ``multiscales[].axes`` list.
        shape: The level's array shape; same length and order as *axes*.
        t: Index to take on a ``time`` axis of size > 1. ``None`` refuses.
        z: Index to take on the third ``space`` axis when its size is > 1.
            ``None`` refuses.
        c: Index to take on a ``channel`` axis whose size is neither 1 nor 3.
            ``None`` refuses.

    Returns:
        ``(index, is_rgb)`` -- an index tuple to apply to the array, and whether
        the result carries three colour channels. When *is_rgb* is ``True`` the
        caller must still move the channel axis last; NGFF stores it first.

    Raises:
        ValueError: If *axes* and *shape* disagree in length, if an axis of
            size > 1 has no override, if a ``channel`` axis is neither 1 nor 3
            without an explicit *c*, or if an override is out of range.

    Examples:
        A plain 2-D plate image passes through untouched:

        >>> from phenotypic.sdk_ import ngff_
        >>> axes = [{'name': 'y', 'type': 'space'}, {'name': 'x', 'type': 'space'}]
        >>> ngff_.project_ngff_axes(axes, (40, 30))
        ((slice(None, None, None), slice(None, None, None)), False)
    """
    if len(axes) != len(shape):
        raise ValueError(
            f"axes/shape mismatch: {len(axes)} axes for a {len(shape)}-D array"
        )

    def _pick(
        kind: str, name: str, override: int | None, size: int, flag: str
    ) -> int:
        # Both the TYPE and the name are in the message. NGFF constrains
        # `axes[].type` but leaves `axes[].name` free, so a store may call its
        # time axis anything; the type is the half a reader can act on, and
        # naming only the name would make the error unreadable on any store
        # that does not use the conventional single letters.
        # The range check comes FIRST, before the size-1 shortcut. An
        # out-of-range override is a caller error at any size, and a size-1
        # axis is where it is least visible: `c=7` on a 1-channel store
        # silently returned channel 0, contradicting the comment below on
        # the explicit-c branch -- "an explicit c= wins ... quietly returning
        # RGB instead would ignore an instruction rather than honour it".
        if override is not None and not 0 <= override < size:
            raise ValueError(
                f"{flag}={override} is out of range for the {kind} axis "
                f"{name!r} of size {size}"
            )
        if size == 1:
            return 0
        if override is None:
            raise ValueError(
                f"this store's {kind} axis {name!r} has size {size}; "
                f"PhenoTypic's Image is 2-D. Pass {flag}=<index> to choose "
                f"one, or use zarr directly to read the whole array."
            )
        return override

    index: list[int | slice] = []
    is_rgb = False
    seen_space = 0
    n_space = sum(1 for a in axes if a.get("type") == "space")
    # The docstring above claims this is the TOTAL mapping from NGFF axes onto
    # a 2-D image. That claim only holds for the 2 or 3 space axes NGFF itself
    # requires: with 0 or 1 there is no plane to read, and with 4+ the
    # "first of three is z" rule below has no meaning.
    if not 2 <= n_space <= 3:
        raise ValueError(
            f"this store declares {n_space} space axes; PhenoTypic reads 2 "
            f"(yx) or 3 (zyx, choosing one z). It is not a 2-D image."
        )

    for axis, size in zip(axes, shape):
        raw_kind = axis.get("type")
        kind = str(raw_kind) if raw_kind else "untyped"
        name = str(axis.get("name", kind))
        if raw_kind == "time":
            index.append(_pick(kind, name, t, size, "t"))
        elif raw_kind == "channel":
            if size == 3 and c is None:
                is_rgb = True
                index.append(slice(None))
            elif size == 1 and c is None:
                index.append(0)
            elif c is None:
                raise ValueError(
                    f"this store's channel axis {name!r} has size {size}; "
                    f"PhenoTypic reads 1 (grayscale) or 3 (RGB). Pass "
                    f"c=<index> to choose one channel."
                )
            else:
                # An explicit c= wins even at size 3: the caller has said
                # "this one channel", and quietly returning RGB instead would
                # ignore an instruction rather than honour it.
                index.append(_pick(kind, name, c, size, "c"))
        elif raw_kind == "space":
            seen_space += 1
            # Three space axes means the first is the stacking (z) axis.
            if n_space == 3 and seen_space == 1:
                index.append(_pick(kind, name, z, size, "z"))
            else:
                index.append(slice(None))
        else:
            # A custom or null axis type. NGFF permits it; we cannot map it.
            # Size 1 squeezes; anything larger refuses. Its own message, NOT
            # `_pick`: there is no override to name here, and _pick's
            # "Pass (no override)=<index>" reads as a flag that does not exist.
            if size != 1:
                raise ValueError(
                    f"this store's axis {name!r} has type {raw_kind!r}, "
                    f"which PhenoTypic cannot map onto a 2-D image, and "
                    f"size {size}. Only a size-1 axis of an unrecognised "
                    f"type can be squeezed; use zarr directly to read it."
                )
            index.append(0)

    return tuple(index), is_rgb


@dataclass(frozen=True)
class NgffImageSpec:
    """One NGFF image, projected onto PhenoTypic's 2-D image model.

    Attributes:
        array: Level pixels as ``(H, W)`` or ``(H, W, 3)``.
        series: Resolved series path, relative to the store root.
        level: Pyramid level actually read.
        bit_depth: From ``phenotypic.metadata.protected[Metadata_BitDepth]``
            when present, else inferred from an integer dtype, else ``None``.
            There is no ``phenotypic.bit_depth`` key and never has been.
        phenotypic: The ``attributes.phenotypic`` block; ``{}`` when absent.
    """

    array: np.ndarray
    series: str
    level: int
    bit_depth: int | None
    phenotypic: dict


def _zarr_v2_marker(store_path: Path) -> str | None:
    """Return the Zarr v2 group marker present at *store_path*, or ``None``.

    A v2 group is spelled ``.zgroup``/``.zattrs`` where v3 writes
    ``zarr.json``, so a v2 store has no root by :func:`read_root_attributes`'s
    reckoning and surfaces as ``FileNotFoundError`` -- which is this
    codebase's established signal for "interrupted write, store absent". The
    two must not be confused: ``bioformats2raw``'s default output and QuPath's
    export are NGFF 0.4 / Zarr v2 today (spec 3.1 case C).

    Args:
        store_path: Directory to inspect.

    Returns:
        The marker filename found, or ``None`` if the directory is not a
        Zarr v2 group.
    """
    for marker in (".zgroup", ".zattrs"):
        if (Path(store_path) / marker).is_file():
            return marker
    return None


def _declared_series(store_path: Path, attributes: dict) -> list[str]:
    """List the series paths a store declares, for an error message.

    Args:
        store_path: Path to a ``*.ome.zarr`` directory.
        attributes: The root ``attributes`` mapping.

    Returns:
        Declared series paths; ``[]`` when the store declares none.
    """
    ome_json = Path(store_path) / OME_GROUP / STORE_ROOT_JSON
    if ome_json.is_file():
        payload = json.loads(ome_json.read_text(encoding="utf-8"))
        declared = payload.get("attributes", {}).get("ome", {}).get("series")
        if declared:
            return [str(entry) for entry in declared]
    block = attributes.get(PhenotypicAttr.ROOT, {})
    series = block.get(PhenotypicAttr.SERIES)
    if isinstance(series, dict):
        return [str(value) for value in series.values()]
    return []


def _resolve_series_path(store_path: Path, attributes: dict) -> str:
    """Pick the series a generic reader should open. See spec 4.1.

    The ``ome.plate`` check is FIRST, and deliberately so -- see the comment
    at that branch and spec 4.1 step 1.
    """
    ome = attributes.get("ome", {})
    # STEP 1, before the declared-series list. A `bioformats2raw` plate
    # carries BOTH a root `ome.plate` and an `OME/zarr.json` series list, so
    # checking the series list first would return a well field from a store
    # that must be refused.
    if "plate" in ome:
        raise ValueError(
            f"{store_path} is an HCS plate, which is a collection of wells "
            f"rather than one image. Pass series=<row>/<col>/<field> to read "
            f"a single field."
        )

    ome_json = Path(store_path) / OME_GROUP / STORE_ROOT_JSON
    if ome_json.is_file():
        payload = json.loads(ome_json.read_text(encoding="utf-8"))
        declared = payload.get("attributes", {}).get("ome", {}).get("series")
        if declared:
            return str(declared[0])

    if "multiscales" in ome:
        return ""  # the root group is itself the image

    if (Path(store_path) / "0" / STORE_ROOT_JSON).is_file():
        return "0"  # NGFF 2.2.3 consecutive-integer form

    raise ValueError(
        f"{store_path} declares no OME series, no multiscales at its root, "
        f"and no group '0'. It is not an OME-Zarr image."
    )


def read_ngff_image_spec(
    store_path: Path,
    *,
    series: str | None = None,
    level: int = 0,
    t: int | None = None,
    z: int | None = None,
    c: int | None = None,
) -> NgffImageSpec:
    """Read any OME-Zarr store as plain pixels.

    The read path behind :meth:`phenotypic.Image.imread` for a store. It reads
    NGFF **structure** only and treats ``attributes.phenotypic`` as optional
    enrichment, so a napari, QuPath, or ``bioformats2raw`` export works.

    It deliberately does **not** call :func:`require_readable_store`: that
    raises ``KeyError`` when the ``phenotypic`` block is absent, which is the
    normal condition for every third-party store -- the exact case this
    function exists to serve. A store written by a newer PhenoTypic is readable
    here, and correctly so: its NGFF geometry is still NGFF.

    Args:
        store_path: A ``*.ome.zarr`` directory.
        series: Series to read. ``None`` resolves it per spec 4.1.
        level: Pyramid level. ``0`` is the highest resolution; NGFF requires
            ``datasets`` to be ordered largest first.
        t: Index on a ``time`` axis of size > 1.
        z: Index on the stacking ``space`` axis when its size is > 1.
        c: Index on a ``channel`` axis that is neither 1 nor 3.

    Returns:
        An :class:`NgffImageSpec`.

    Raises:
        FileNotFoundError: If the store has no root ``zarr.json`` and is not
            a Zarr v2 group either -- an interrupted write reads as absent.
        ValueError: If the store is Zarr v2 (NGFF 0.4), is an HCS plate,
            declares no readable image, has no series named by *series*, has
            no such pyramid *level*, or cannot be projected onto a 2-D image
            (see :func:`project_ngff_axes`).

    Examples:
        Read back the colony plate a run wrote, as plain pixels:

        >>> import tempfile
        >>> from pathlib import Path
        >>> from phenotypic import Image
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.sdk_ import ngff_
        >>> plate = Image(load_synth_yeast_plate())
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     store = plate.save2zarr(Path(tmp) / 'plate.ome.zarr')
        ...     spec = ngff_.read_ngff_image_spec(store)
        ...     spec.array.shape == plate.rgb[:].shape
        True
    """
    import zarr

    from phenotypic.schema import IMAGE

    store_path = Path(store_path)
    try:
        attributes = read_root_attributes(store_path)
    except FileNotFoundError:
        # NOT in `_resolve_series_path`, which never runs for a v2 store:
        # `read_root_attributes` reads `zarr.json` and raises first. This is
        # the only place the v2 case is reachable.
        marker = _zarr_v2_marker(store_path)
        if marker is None:
            raise
        raise ValueError(
            f"{store_path} carries {marker} and no {STORE_ROOT_JSON}, so it "
            f"is a Zarr v2 store (NGFF 0.4 or earlier). PhenoTypic reads "
            f"NGFF 0.5 / Zarr v3. Convert the store to Zarr v3 before "
            f"reading it."
        ) from None
    phenotypic = attributes.get(PhenotypicAttr.ROOT, {})

    resolved = (
        _resolve_series_path(store_path, attributes) if series is None else series
    )

    group_path = store_path / resolved if resolved else store_path
    group_json = group_path / STORE_ROOT_JSON
    if not group_json.is_file():
        # A bad `series=` is a caller error, not a missing store. Letting
        # `read_text` raise reported `FileNotFoundError` on an internal path,
        # which reads as "the store is gone".
        declared = _declared_series(store_path, attributes)
        available = (
            "It declares: " + ", ".join(repr(name) for name in declared)
            if declared
            else "It declares no series."
        )
        raise ValueError(
            f"{store_path} has no series {resolved!r}. {available}"
        )
    payload = json.loads(group_json.read_text(encoding="utf-8"))
    multiscales = payload["attributes"]["ome"]["multiscales"][0]
    axes = multiscales["axes"]
    datasets = multiscales["datasets"]
    if not 0 <= level < len(datasets):
        raise ValueError(
            f"level {level} is out of range; {store_path} has "
            f"{len(datasets)} pyramid level(s)"
        )

    # `long_path`, matching `load_layer_zarr`: a store path plus a series plus
    # a level segment is long enough to hit Windows' MAX_PATH, and every other
    # array open in the codebase goes through this helper.
    array = zarr.open_array(
        store=long_path(group_path / datasets[level]["path"]), mode="r"
    )
    index, is_rgb = project_ngff_axes(axes, array.shape, t=t, z=z, c=c)
    data = np.asarray(array[index])
    if is_rgb:
        data = np.moveaxis(data, 0, -1)  # NGFF stores channels first

    # `metadata.protected`, NOT `phenotypic.bit_depth` -- no writer emits the
    # latter and none ever has. This is the key `_load_from_store` reads, and
    # it is the ONLY source for a float series, where dtype inference has no
    # answer at all.
    bit_depth = (
        phenotypic.get(PhenotypicAttr.METADATA, {})
        .get(PhenotypicAttr.PROTECTED, {})
        .get(IMAGE.BIT_DEPTH)
    )
    if bit_depth is None:
        bit_depth = {np.uint8: 8, np.uint16: 16}.get(data.dtype.type)
    try:
        resolved_bit_depth = int(bit_depth) if bit_depth is not None else None
    except (TypeError, ValueError):
        # A third-party store may put anything in that key. An unparseable
        # value is "unknown", which the Image constructor's default handles --
        # not a read failure.
        resolved_bit_depth = None

    return NgffImageSpec(
        array=data,
        series=resolved,
        level=level,
        bit_depth=resolved_bit_depth,
        phenotypic=dict(phenotypic),
    )


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

    ``coordinateTransformations`` records the repeated 2x sampling operation,
    not the ratio between stored level shapes. A block-center translation keeps
    the downsampled samples registered to level 0.

    **Physical resolution is deliberately not projected.** Scale vectors are
    pure sampling factors and ``unit`` is omitted, which §2.1 permits.

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
            "coordinateTransformations": level_coordinate_transformations(
                base, index
            ),
        }
        for index in range(len(level_shapes))
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

    ``rdefs.model`` is the only field in NGFF that states the rendering model
    outright (§2.5: exactly ``"color"`` or ``"greyscale"``), and OMERO and
    Vizarr read it. It is emitted only where ``omero`` itself is emitted, so
    the whole-or-nothing rule per group is unaffected.

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
    palette = _RGB_CHANNEL_COLORS if series == "rgb" else ((series, "FFFFFF"),)
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
    block: dict = {
        "channels": channels,
        "rdefs": {"model": "color" if series == "rgb" else "greyscale"},
    }
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
            grouped.setdefault(getattr(module, "value", module), {})[key] = (
                value
            )
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
    "[^\u0009\u000a\u000d\u0020-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]"
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
    "u1": "uint8",
    "u2": "uint16",
    "u4": "uint32",
    "i1": "int8",
    "i2": "int16",
    "i4": "int32",
    "f4": "float",
    "f8": "double",
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
        for directory in sorted(
            directories, key=lambda p: len(p.parts), reverse=True
        ):
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


def _reconcile_attempt_trash(trash: Path, final: Path) -> Path | None:
    """Restore the previous store or identify trash safe to discard.

    If no concurrent writer has published, restoring *trash* to *final*
    preserves the previous store. If another writer has published meanwhile,
    its non-empty *final* wins and this attempt's superseded trash is returned
    for deletion after the publication guard is released. Any other rollback
    failure is surfaced with the trash intact.

    Returns:
        Superseded *trash* safe to delete, or ``None`` if no cleanup remains.
    """
    if not trash.exists():
        return None
    try:
        os.replace(long_path(trash), long_path(final))
    except OSError:
        if not final.exists():
            raise
        return trash
    return None


def promote_store(
    part: Path,
    final: Path,
    *,
    fsync: bool,
    commit_guard: CommitGuard | None = None,
) -> Path:
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

    On failure after a successful move-aside, that attempt's trash is
    reconciled before retrying or raising. The previous store is rolled back
    when *final* is absent; if a concurrent writer has already published a new
    *final*, that winner remains authoritative and only the attempt's
    superseded trash is removed. Every retry uses a fresh UUID trash path, so
    no attempt can collide with its predecessor's move-aside directory.

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

    last: OSError | None = None
    for attempt in range(PROMOTE_RETRY_ATTEMPTS):
        trash = final.parent / f".{final.name}.{uuid4().hex}{TRASH_SUFFIX}"
        moved_aside = False
        discard_trash: Path | None = None
        try:
            with publication_commit(commit_guard):
                try:
                    # Re-evaluate existence EVERY attempt while the commit
                    # fence is held. Preparation and fsync_tree stay outside.
                    if final.exists():
                        os.replace(long_path(final), long_path(trash))
                        moved_aside = True
                    os.replace(long_path(part), long_path(final))
                except OSError:
                    if moved_aside:
                        discard_trash = _reconcile_attempt_trash(trash, final)
                    raise
                if fsync and os.name == "posix":
                    # The rename is a directory-entry change in final.parent.
                    _fsync_path(final.parent)
        except OSError as exc:
            if discard_trash is not None:
                # Recursive cleanup stays outside the output-wide lifecycle lock.
                shutil.rmtree(long_path(discard_trash), ignore_errors=True)
            last = exc
            if not _is_retryable(exc):
                raise
            # Never retain the lifecycle lock during retry backoff.
            time.sleep(PROMOTE_RETRY_BASE_SECONDS * (2**attempt))
            continue
        # The uniquely named prior store can be reclaimed after releasing the
        # output-wide lifecycle lock; no later publication can adopt it.
        if trash.exists():
            shutil.rmtree(long_path(trash), ignore_errors=True)
        return final
    assert last is not None
    raise last


#: A `.part` younger than this may still be being written. The sweep never
#: touches one. Generous by design: the cost of skipping a genuine orphan is one
#: stale directory until the next run; the cost of deleting a live one is a
#: destroyed in-flight image.
SWEEP_MIN_AGE_SECONDS: Final[float] = 6 * 60 * 60


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
                path.name.endswith(PART_SUFFIX)
                or path.name.endswith(TRASH_SUFFIX)
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


def store_level0_shape(
    store_path: Path, member_path: str
) -> tuple[int, ...] | None:
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
    * processed level-0 ``(y, x)`` extents agree and every extent is non-zero;
      the full decoded ``original`` may differ after geometry-changing pre-ops. A
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
        if (
            block.get(PhenotypicAttr.STORE_SCHEMA_VERSION)
            != STORE_SCHEMA_VERSION
        ):
            return False
        series = block[PhenotypicAttr.SERIES]
        labels = block.get(PhenotypicAttr.LABELS, {})
        members = [
            *series.values(),
            # `.get`: a label-less store OMITS the key (Task 1.3, ledger C3).
            # This is a validity predicate -- it must RETURN FALSE on a store it
            # does not accept, never raise. Indexing would make a label-less
            # store a KeyError propagating out of resume classification and
            # migration, both of which call this to decide what to do next.
            *labels.values(),
        ]
        if not members:
            return False
        shapes: dict[str, tuple[int, ...]] = {}
        for member in members:
            shape = store_level0_shape(store, member)
            if shape is None:
                return False
            shapes[member] = shape
        spatial = [shape[-2:] for shape in shapes.values()]
        if any(len(yx) < 2 or yx[0] <= 0 or yx[1] <= 0 for yx in spatial):
            return False
        # The retained decoded source is a full-forward collection member, not
        # a Stage-2 input layer. Geometry-changing pre-ops may legitimately
        # make it larger than every processed series and label.
        aligned = [
            shapes[member]
            for key, member in series.items()
            if key != "original"
        ] + [shapes[member] for member in labels.values()]
        aligned_spatial = [shape[-2:] for shape in aligned]
        return bool(aligned_spatial) and all(
            yx == aligned_spatial[0] for yx in aligned_spatial[1:]
        )
    except (AttributeError, OSError, KeyError, TypeError, ValueError):
        return False
