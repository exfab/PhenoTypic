# Phase 1 — `sdk_/ngff_.py`: geometry, attributes, projection, promote, validity

> Global Constraints live in [`README.md`](README.md#global-constraints) and apply to
> every task here. Spec: [`design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md) §1, §2, §3.2, §3.6, §3.7, §3.8.

**Depends on:** Phase 0 (needs `import zarr`).
**Blocks:** Phase 2 and everything downstream.

This phase builds one new module and nothing else. It touches no existing behaviour, so
the whole phase can land while the HDF path still works. Every task is pure-function or
filesystem-local and testable without a pipeline.

**Before starting, run the logic validation script and confirm it passes:**

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py
```

Expected: `All store-geometry claims hold.` and exit 0. The geometry helpers you write in
Task 1.1 must agree with `level_count` and `level_shapes` in that script — Task 1.1's test
imports the script and asserts equality against it, so the script is the reference, not a
parallel implementation.

---

### Task 1.1: Layout constants and pyramid geometry

**Files:**
- Create: `src/phenotypic/sdk_/ngff_.py`
- Test: `tests/unit/sdk_/test_ngff_geometry.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  ```python
  NGFF_VERSION: Final[str] = "0.5"
  BIOFORMATS2RAW_LAYOUT: Final[int] = 3
  STORE_SCHEMA_VERSION: Final[int] = 3
  STORE_SUFFIX: Final[str] = ".ome.zarr"
  PYRAMID_STOP_PX: Final[int] = 512
  OME_GROUP: Final[str] = "OME"
  OME_XML_NAME: Final[str] = "METADATA.ome.xml"
  LABELS_GROUP: Final[str] = "labels"
  OBJMAP_LABEL: Final[str] = "objmap"
  SERIES_ORDER: Final[tuple[str, str, str]] = ("rgb", "gray", "detect_mat")
  AXES_3D: Final[tuple[str, str, str]] = ("c", "y", "x")
  AXES_2D: Final[tuple[str, str]] = ("y", "x")

  def pyramid_level_count(height: int, width: int, *, stop_px: int = PYRAMID_STOP_PX) -> int
  def pyramid_level_shapes(shape: tuple[int, ...], levels: int) -> tuple[tuple[int, ...], ...]
  def level_scale_vector(level0: tuple[int, ...], level_n: tuple[int, ...]) -> list[float]
  def downsample_image(array: np.ndarray) -> np.ndarray
  def downsample_label(array: np.ndarray) -> np.ndarray
  def build_pyramid(array: np.ndarray, levels: int, *, kind: Literal["image", "label"]) -> list[np.ndarray]
  def axes_for(series: str) -> tuple[str, ...]
  ```

**Constraints specific to this task:**
- `pyramid_level_count` must be `ceil(log2(max(H,W)/stop_px)) + 1`, and `1` when
  `max(H,W) <= stop_px`. **`ceil`, never `floor`.**
- `pyramid_level_shapes` uses ceil-halving `(h+1)//2` with a floor of 1 per axis, and
  leaves any leading channel axis unchanged.
- `level_scale_vector` divides level-0 extent by level-n extent **per axis, from the
  actual shapes** — never `2 ** n`. The channel axis, if present, gets scale `1.0`.
- `downsample_label` is `array[::2, ::2]` (top-left of each block). Never mean.
- `downsample_image` is a 2×2 block mean over an **edge-replicated** pad, so an odd
  extent yields `(h+1)//2` without a zero-padded darkened edge. It preserves dtype:
  integer inputs are rounded with `np.rint` and cast back.
- **The pyramid depth is fixed, not tunable.** `pyramid_level_count(h, w)` is the whole
  policy: a pure function of the level-0 shape, with no user lever and no stored choice to
  disagree with. The spec's `--pyramid-levels auto|N` (§1.3) is **descoped** — see
  OPEN-QUESTIONS **P3**. A single-level store is still reachable internally (builder node
  previews, Phase 2 Task 2.4) via the private `levels=` argument on `_save_store`; it is
  simply not a CLI surface.

  Descoping it also dissolves P3 outright: with geometry a pure function of image shape,
  two stores in one tree cannot disagree, so `valid_staged_store` needs no level check and
  a resumed run cannot produce mixed geometry.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/test_ngff_geometry.py`:

```python
"""Pyramid geometry, checked against the committed logic-validation script.

The script under docs/superpowers/logic_validation_scripts/ is the reference
implementation for level counts and level shapes; it depends only on numpy and
has already refuted a floor-based formula. These tests assert the shipped
helpers agree with it, so the two can never drift.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from phenotypic.sdk_ import ngff_

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "superpowers"
    / "logic_validation_scripts"
    / "2026-08-18-ome-zarr-image-store"
    / "ngff_store_geometry.py"
)


def _load_reference():
    spec = importlib.util.spec_from_file_location("ngff_store_geometry", _SCRIPT)
    assert spec is not None and spec.loader is not None, _SCRIPT
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE = _load_reference()

PLATES = [(2048, 2048), (4000, 3000), (6000, 4000), (512, 512), (300, 200), (513, 100)]


@pytest.mark.parametrize(("height", "width"), PLATES)
def test_level_count_matches_reference(height: int, width: int) -> None:
    assert ngff_.pyramid_level_count(height, width) == REFERENCE.level_count(
        height, width
    )


def test_level_count_uses_ceil_not_floor() -> None:
    """floor(log2(4000/512)) + 1 == 3, which stops one level early at 1000x750."""
    assert ngff_.pyramid_level_count(4000, 3000) == 4


def test_single_level_at_or_below_stop_px() -> None:
    assert ngff_.pyramid_level_count(512, 512) == 1
    assert ngff_.pyramid_level_count(100, 100) == 1


@pytest.mark.parametrize(("height", "width"), PLATES)
def test_level_shapes_match_reference(height: int, width: int) -> None:
    levels = ngff_.pyramid_level_count(height, width)
    shapes = ngff_.pyramid_level_shapes((height, width), levels)
    assert [tuple(s) for s in shapes] == [
        tuple(s) for s in REFERENCE.level_shapes(height, width)
    ]


def test_level_shapes_ceil_halve_odd_extents() -> None:
    assert ngff_.pyramid_level_shapes((1025, 7), 3) == ((1025, 7), (513, 4), (257, 2))


def test_level_shapes_leave_channel_axis_alone() -> None:
    assert ngff_.pyramid_level_shapes((3, 1025, 7), 2) == ((3, 1025, 7), (3, 513, 4))


def test_scale_vector_comes_from_actual_shapes_not_powers_of_two() -> None:
    """1025 -> 513 is a ratio of 1025/513, which is NOT 2.0."""
    scale = ngff_.level_scale_vector((1025, 7), (513, 4))
    assert scale == pytest.approx([1025 / 513, 7 / 4])
    assert scale[0] != pytest.approx(2.0)


def test_scale_vector_pins_channel_axis_to_one() -> None:
    assert ngff_.level_scale_vector((3, 1024, 1024), (3, 512, 512)) == pytest.approx(
        [1.0, 2.0, 2.0]
    )


def test_label_downsample_invents_no_new_values() -> None:
    rng = np.random.default_rng(20260818)
    labels = rng.choice(np.array([0, 3, 7, 11, 40], dtype=np.uint16), size=(64, 64))
    small = ngff_.downsample_label(labels)
    assert set(np.unique(small)).issubset(set(np.unique(labels)))
    assert small.shape == (32, 32)
    assert small.dtype == labels.dtype


def test_mean_downsample_would_invent_values() -> None:
    """Guards C5: proves the rejected method really is wrong, not merely unchosen."""
    labels = np.array([[0, 40], [40, 40]], dtype=np.uint16)
    meaned = ngff_.downsample_image(labels)
    assert set(np.unique(meaned)) - set(np.unique(labels))


def test_image_downsample_odd_extent_uses_edge_pad_not_zero_pad() -> None:
    array = np.full((3, 3), 100, dtype=np.uint8)
    small = ngff_.downsample_image(array)
    assert small.shape == (2, 2)
    assert (small == 100).all(), "a zero pad would darken the trailing row/column"


def test_image_downsample_preserves_dtype() -> None:
    array = np.arange(16, dtype=np.uint16).reshape(4, 4)
    assert ngff_.downsample_image(array).dtype == np.uint16
    assert ngff_.downsample_image(array.astype(np.float64)).dtype == np.float64


def test_build_pyramid_shapes_and_count() -> None:
    array = np.zeros((1025, 7), dtype=np.uint16)
    levels = ngff_.build_pyramid(array, 3, kind="label")
    assert [lvl.shape for lvl in levels] == [(1025, 7), (513, 4), (257, 2)]


def test_build_pyramid_channel_first_rgb() -> None:
    array = np.zeros((3, 1024, 1024), dtype=np.uint8)
    levels = ngff_.build_pyramid(array, 2, kind="image")
    assert [lvl.shape for lvl in levels] == [(3, 1024, 1024), (3, 512, 512)]


def test_axes_for_series() -> None:
    assert ngff_.axes_for("rgb") == ("c", "y", "x")
    assert ngff_.axes_for("gray") == ("y", "x")
    assert ngff_.axes_for("detect_mat") == ("y", "x")
    assert ngff_.axes_for("objmap") == ("y", "x")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/sdk_/test_ngff_geometry.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'phenotypic.sdk_.ngff_'`.

- [ ] **Step 3: Write the module**

Create `src/phenotypic/sdk_/ngff_.py`:

```python
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
import hashlib
import json
import logging
import math
import os
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/unit/sdk_/test_ngff_geometry.py -v
uv run python docs/superpowers/logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py
```

Expected: all tests PASS; script exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/ngff_.py tests/unit/sdk_/test_ngff_geometry.py
git commit -m "feat(sdk): add NGFF layout constants and pyramid geometry

Level count is ceil(log2(max(H,W)/512)) + 1; level shapes ceil-halve; the
coordinateTransformations scale vector is derived from actual level shapes
rather than 2**n, because odd extents make the two diverge. Labels
downsample nearest-neighbour, images by an edge-padded block mean. The
tests import the committed logic-validation script and assert equality
against it, so the shipped helpers and the numeric reference cannot drift."
```

---

### Task 1.2: Chunk, shard, and codec policy

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py`
- Test: `tests/unit/sdk_/test_ngff_array_policy.py`

**Interfaces:**
- Consumes: `axes_for`, `pyramid_level_shapes` (Task 1.1).
- Produces:
  ```python
  CHUNK_YX: Final[tuple[int, int]] = (1024, 1024)
  SHARD_YX: Final[tuple[int, int]] = (4096, 4096)
  CODEC_NAME: Final[str] = "zstd"
  CHUNK_KEY_SEPARATOR: Final[str] = "."

  def chunk_shape_for(shape: tuple[int, ...]) -> tuple[int, ...]
  def shard_shape_for(shape: tuple[int, ...]) -> tuple[int, ...]
  def array_create_kwargs(shape: tuple[int, ...], dtype: np.dtype, series: str) -> dict
  ```

**Constraints specific to this task:**
- Chunks `(1, 1024, 1024)` for `rgb`; `(1024, 1024)` for 2-D arrays.
- Shards `(C, 4096, 4096)`: the **full channel extent**, so per-channel chunks collapse
  into one file. The shard shape must be an exact multiple of the chunk shape **in every
  dimension** — `3 % 1 == 0` on the channel axis is part of the claim, not a triviality.
- **A level's shard is `(C, 4096, 4096)` whenever the level is at least one chunk wide; a
  smaller level collapses to `chunk == shard == extent`.** Partial edge shards are normal —
  zarr constrains shard-vs-chunk divisibility only, never shard-vs-array
  (`zarr-python/design/chunk-grid.md`: "Validation ensures edge lengths are divisible by
  subchunk sizes"). So a 4000×3000 level gets chunk `(1024, 1024)` and shard `(4096, 4096)`,
  which is one shard file — exactly what `ngff_store_geometry.py`'s `data_files` (`:204-207`)
  counts with `ceil(h / 4096) * ceil(w / 4096)`.

  An earlier draft clamped the shard to the level extent and then rounded down to a multiple
  of the chunk. That returns `(3072, 2048)` for a 4000×3000 level — **four** shard files, not
  one — which fails three of this task's own tests and makes spec §1.4's "40 files at auto"
  wrong by construction. Recorded as OPEN-QUESTIONS **P11/P13**. Do not reintroduce it.

  Below one chunk, `chunk = shard = extent` keeps divisibility trivially true and matches the
  script's `ceil` tiling (a 257×2 level is one chunk and one shard either way).
- `chunk_key_encoding` uses `{"name": "default", "configuration": {"separator": "."}}`,
  uniformly store-wide.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/test_ngff_array_policy.py`:

```python
"""Chunk/shard/codec policy. Divisibility is claim C3 of the validation script."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.sdk_ import ngff_


def test_rgb_chunk_is_one_channel_by_1024_square() -> None:
    assert ngff_.chunk_shape_for((3, 4000, 3000)) == (1, 1024, 1024)


def test_two_d_chunk_is_1024_square() -> None:
    assert ngff_.chunk_shape_for((4000, 3000)) == (1024, 1024)


def test_rgb_shard_spans_the_full_channel_axis() -> None:
    assert ngff_.shard_shape_for((3, 4000, 3000)) == (3, 4096, 4096)


def test_two_d_shard() -> None:
    assert ngff_.shard_shape_for((4000, 3000)) == (4096, 4096)


@pytest.mark.parametrize(
    "shape", [(3, 4000, 3000), (4000, 3000), (3, 2048, 2048), (6000, 4000), (257, 2)]
)
def test_shard_is_an_exact_multiple_of_chunk_in_every_dimension(shape) -> None:
    chunk = ngff_.chunk_shape_for(shape)
    shard = ngff_.shard_shape_for(shape)
    assert len(chunk) == len(shard) == len(shape)
    for c, s in zip(chunk, shard, strict=True):
        assert s % c == 0, (shape, chunk, shard)


def test_small_level_clamps_chunk_and_shard_to_its_own_shape() -> None:
    """A 257x2 pyramid level must not carry a 1024x1024 chunk."""
    assert ngff_.chunk_shape_for((257, 2)) == (257, 2)
    assert ngff_.shard_shape_for((257, 2)) == (257, 2)


def test_create_kwargs_carry_dimension_names_matching_axes() -> None:
    kwargs = ngff_.array_create_kwargs((3, 4000, 3000), np.dtype("uint8"), "rgb")
    assert tuple(kwargs["dimension_names"]) == ("c", "y", "x")
    kwargs2d = ngff_.array_create_kwargs((4000, 3000), np.dtype("float64"), "detect_mat")
    assert tuple(kwargs2d["dimension_names"]) == ("y", "x")


def test_create_kwargs_use_the_dot_chunk_key_separator() -> None:
    """A Windows MAX_PATH measure; must be uniform store-wide."""
    kwargs = ngff_.array_create_kwargs((4000, 3000), np.dtype("uint16"), "objmap")
    encoding = kwargs["chunk_key_encoding"]
    assert encoding["configuration"]["separator"] == "."


def test_create_kwargs_use_zstd() -> None:
    kwargs = ngff_.array_create_kwargs((4000, 3000), np.dtype("uint16"), "gray")
    assert "zstd" in repr(kwargs["compressors"]).lower()


def test_shard_write_buffer_is_bounded_and_documented() -> None:
    """96 MB for rgb uint16, 128 MB for a float64 detect_mat (spec 1.4)."""
    rgb = np.prod(ngff_.shard_shape_for((3, 4000, 3000))) * 2
    detect = np.prod(ngff_.shard_shape_for((4000, 3000))) * 8
    assert rgb == 3 * 4096 * 4096 * 2
    assert detect == 4096 * 4096 * 8
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/sdk_/test_ngff_array_policy.py -v
```

Expected: FAIL with `AttributeError: module 'phenotypic.sdk_.ngff_' has no attribute
'chunk_shape_for'`.

- [ ] **Step 3: Append to `ngff_.py`**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/unit/sdk_/test_ngff_array_policy.py -v
```

Expected: all PASS. If `ZstdCodec` is not importable from `zarr.codecs` in the resolved
zarr version, find the correct import with
`uv run python -c "import zarr.codecs as c; print(dir(c))"` and fix the import — do not
fall back to a different codec.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/ngff_.py tests/unit/sdk_/test_ngff_array_policy.py
git commit -m "feat(sdk): add NGFF chunk, shard, and codec policy

Chunks (1,1024,1024) for rgb and (1024,1024) for 2-D; shards span the full
channel extent at (C,4096,4096) so per-channel chunks collapse into one
file. Shard shape is rounded to an exact multiple of the chunk shape in
every dimension including the channel axis, which the v3 sharding codec
requires. Chunk keys use the '.' separator so one key is one path segment,
keeping Windows paths under MAX_PATH."
```

---

### Task 1.3: The `attributes.phenotypic` contract

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py`
- Test: `tests/unit/sdk_/test_ngff_attributes.py`

**Interfaces:**
- Consumes: `STORE_SCHEMA_VERSION`, `SERIES_ORDER`, `LABELS_GROUP`, `OBJMAP_LABEL`.
- Produces:
  ```python
  class PhenotypicAttr:
      ROOT: Final[str] = "phenotypic"
      STORE_SCHEMA_VERSION: Final[str] = "store_schema_version"
      METADATA_SCHEMA_VERSION: Final[str] = "metadata_schema_version"
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

  def primary_series(series_names: Sequence[str]) -> str
  def objmap_path(primary: str) -> str
  def build_phenotypic_attributes(*, image_class, series_names, pyramid_levels,
                                  metadata_sections, detect_mode, illuminant, gamma,
                                  has_labels: bool = True, grid=None, work_id=None,
                                  phenotypic_version=None) -> dict
  def read_root_attributes(store_path: Path) -> dict
  def read_phenotypic_attributes(store_path: Path) -> dict
  ```

**Constraints specific to this task:**
- `series` and `labels` are **separate keys**: `series` maps a logical layer name to a
  group name, `labels` maps a label name to a nested path.
- **`has_labels=False` omits the `labels` key entirely**, for a store written without an
  objmap — which is what a GUI builder preview of a node that changed no labels is
  (`save_intermediate_zarr`, Phase 2 Task 2.4). Defaults to `True` so every existing caller
  is unchanged.

  > **Added (ledger C3).** An earlier draft emitted `labels` unconditionally, so a preview
  > store **declared** `labels.objmap = "gray/labels/objmap"` for a group that was never
  > written — and `assert_store_conforms`'s own loop then raised `FileNotFoundError` walking
  > to it. A guard added downstream in `_assert_reader_level_musts` did not help: it tested
  > for an *empty* mapping, which nothing produced. The key has to be omitted at the source.
- `primary_series` returns `"rgb"` when `rgb` is present, `"gray"` otherwise, and
  `objmap_path(primary)` returns `f"{primary}/{LABELS_GROUP}/{OBJMAP_LABEL}"`.
- `work_id` is a constructor argument, never patched afterwards.
- **Metadata section values are stored verbatim, and are NOT validated.** An earlier draft
  of this task asserted every non-`imported` key resolved through
  `metadata_member_for_header()`. That is wrong and would abort `save2zarr` on most
  production runs — verified by execution in this worktree:

  ```text
  'Metadata_Strain'    | member: Metadata_Strain | is_metadata_header: True
  'Metadata_PlateNum'  | member: None            | is_metadata_header: True
  'MyColumn'           | member: None            | is_metadata_header: False
  ```

  `metadata_member_for_header` is a **semantic-ownership resolver**, not a format check:
  it returns `None` for `Metadata_PlateNum`, a real column in this project's canonical
  Results matrix. And a legitimately loaded image really does carry bare public keys —
  an HDF round-trip yields `public: {..., 'Metadata_PlateNum': 3, 'MyColumn': 'x'}`,
  because `_remap_legacy_metadata_key` (`_image_io_handler.py:100-106`) deliberately
  preserves unknown names verbatim: "public and imported image metadata historically
  round-trip arbitrary names verbatim."

  The HDF writer has no equivalent check, so adding one here is a regression, not a
  hardening. Recorded as OPEN-QUESTIONS **D3**. Ownership questions elsewhere still go
  through `metadata_owner_for_header()` and never through `startswith("Metadata_")` —
  that rule is unchanged; it simply is not a write-time gate.
- `read_root_attributes` reads `<store>/zarr.json` directly with `json.loads` rather than
  opening a zarr group, so it stays cheap and usable from `valid_staged_store`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/test_ngff_attributes.py`:

```python
"""The attributes.phenotypic block is the sole source of truth on read."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic.sdk_ import ngff_
from phenotypic.sdk_.ngff_ import PhenotypicAttr


def _sections() -> dict[str, dict]:
    return {
        "protected": {
            "Metadata_ImageName": "plate_01",
            "Metadata_ImageType": "Grid",
            "Metadata_BitDepth": 16,
        },
        "public": {"Metadata_Strain": "BY4741"},
        "imported": {"TIFF:XResolution": 300.0},
    }


def test_primary_series_prefers_rgb() -> None:
    assert ngff_.primary_series(["rgb", "gray", "detect_mat"]) == "rgb"


def test_primary_series_falls_back_to_gray() -> None:
    assert ngff_.primary_series(["gray", "detect_mat"]) == "gray"


def test_objmap_path_is_relative_to_the_primary_series() -> None:
    assert ngff_.objmap_path("gray") == "gray/labels/objmap"
    assert ngff_.objmap_path("rgb") == "rgb/labels/objmap"


def test_series_and_labels_are_separate_keys() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="GridImage",
        series_names=["rgb", "gray", "detect_mat"],
        pyramid_levels=4,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant="D65",
        gamma="sRGB",
        grid={"nrows": 8, "ncols": 12, "grid_finder": {"class": "X", "params": {}}},
        work_id="w-1",
    )
    assert set(block[PhenotypicAttr.SERIES]) == {"rgb", "gray", "detect_mat"}
    assert block[PhenotypicAttr.LABELS] == {"objmap": "rgb/labels/objmap"}
    assert PhenotypicAttr.SERIES != PhenotypicAttr.LABELS


def test_two_version_markers_are_both_present_and_distinct() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant=None,
        gamma=None,
    )
    assert block[PhenotypicAttr.STORE_SCHEMA_VERSION] == 3
    assert block[PhenotypicAttr.METADATA_SCHEMA_VERSION] == 2


def test_image_class_and_image_type_stay_distinct() -> None:
    """A GridSection is not a GridImage; collapsing them loses information."""
    sections = _sections()
    sections["protected"]["Metadata_ImageType"] = "GridSection"
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections=sections,
        detect_mode="gray",
        illuminant=None,
        gamma=None,
    )
    assert block[PhenotypicAttr.IMAGE_CLASS] == "Image"
    assert (
        block[PhenotypicAttr.METADATA]["protected"]["Metadata_ImageType"]
        == "GridSection"
    )


def test_downsample_methods_pins_the_actual_values() -> None:
    """One literal assertion, or nothing pins "mean"/"nearest" anywhere.

    Both other tests now compare a produced value against the constant the
    producer reads, so they can no longer fail on a wrong value (ledger GEN-43).
    """
    assert ngff_.DOWNSAMPLE_METHODS == {
        "image": ("mean", "2x block mean over an edge-replicated pad"),
        "label": ("nearest", "2x nearest-neighbour (top-left of each block)"),
    }
    assert ngff_.DOWNSAMPLE_KINDS == {"image": "mean", "label": "nearest"}


def test_pyramid_block_records_levels_stop_and_downsample_methods() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=4,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant=None,
        gamma=None,
    )
    pyramid = block[PhenotypicAttr.PYRAMID]
    assert pyramid == {
        "levels": 4,
        "stop_px": 512,
        "downsample": ngff_.DOWNSAMPLE_KINDS,
    }


def test_work_id_is_a_constructor_argument_not_a_patch() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant=None,
        gamma=None,
        work_id="abc123",
    )
    assert block[PhenotypicAttr.WORK_ID] == "abc123"


def test_arbitrary_metadata_keys_are_stored_verbatim() -> None:
    """Real images carry Metadata_PlateNum (member=None) and bare public keys.

    A write-time canonicality gate would abort save2zarr on most production
    runs. See OPEN-QUESTIONS D3.
    """
    sections = _sections()
    sections["public"]["Metadata_PlateNum"] = 3
    sections["public"]["MyColumn"] = "x"
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections=sections,
        detect_mode="gray",
        illuminant=None,
        gamma=None,
    )
    stored = block[PhenotypicAttr.METADATA]["public"]
    assert stored["Metadata_PlateNum"] == 3
    assert stored["MyColumn"] == "x"


def test_block_is_json_serialisable() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="GridImage",
        series_names=["rgb", "gray", "detect_mat"],
        pyramid_levels=4,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant="D65",
        gamma="sRGB",
        grid={"nrows": 8, "ncols": 12, "grid_finder": None},
    )
    assert json.loads(json.dumps(block)) == block


def test_read_phenotypic_attributes_round_trips(tmp_path: Path) -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant=None,
        gamma=None,
    )
    store = tmp_path / "x.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {"version": "0.5", "bioformats2raw.layout": 3},
                    "phenotypic": block,
                },
            }
        ),
        encoding="utf-8",
    )
    assert ngff_.read_phenotypic_attributes(store) == block


def test_read_phenotypic_attributes_raises_on_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ngff_.read_phenotypic_attributes(tmp_path / "absent.ome.zarr")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/sdk_/test_ngff_attributes.py -v
```

Expected: FAIL with `ImportError: cannot import name 'PhenotypicAttr'`.

- [ ] **Step 3: Append to `ngff_.py`**

```python
# ---------------------------------------------------------------------------
# attributes.phenotypic -- the source of truth on read
# ---------------------------------------------------------------------------

#: Version of the flat ``Metadata_<Label>`` header namespace. Distinct from
#: :data:`STORE_SCHEMA_VERSION`, which versions groups and arrays.
METADATA_SCHEMA_VERSION: Final[int] = 2


class PhenotypicAttr:
    """Keys inside the namespaced ``attributes.phenotypic`` block.

    Spelled out here so a renamed key fails at type-check time rather than
    silently at runtime, matching the ``HdfAttr`` / ``JobMetadataKey`` pattern
    already used in :mod:`phenotypic.sdk_._io_constants`.
    """

    ROOT: Final[str] = "phenotypic"
    STORE_SCHEMA_VERSION: Final[str] = "store_schema_version"
    METADATA_SCHEMA_VERSION: Final[str] = "metadata_schema_version"
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
        PhenotypicAttr.METADATA_SCHEMA_VERSION: METADATA_SCHEMA_VERSION,
        PhenotypicAttr.PHENOTYPIC_VERSION: (
            phenotypic_version or phenotypic.__version__
        ),
        PhenotypicAttr.IMAGE_CLASS: image_class,
        PhenotypicAttr.SERIES: {name: name for name in series_names},
        # Omitted entirely when the store carries no label image. An earlier
        # draft emitted this unconditionally, so a preview store written by
        # `save_intermediate_zarr(layers=("gray",))` DECLARED
        # `labels.objmap = "gray/labels/objmap"` for a group that does not exist
        # -- and `assert_store_conforms` then FileNotFoundError'd walking it.
        # Ledger C3.
        PhenotypicAttr.LABELS: (
            {OBJMAP_LABEL: objmap_path(primary)} if has_labels else {}
        ),
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
    if work_id is not None:
        block[PhenotypicAttr.WORK_ID] = work_id
    if grid is not None:
        block[PhenotypicAttr.GRID] = grid
    return block


def read_root_attributes(store_path: "Path") -> dict:
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
    import json
    from pathlib import Path as _Path

    payload = json.loads(
        (_Path(store_path) / "zarr.json").read_text(encoding="utf-8")
    )
    return payload.get("attributes", {})


def read_phenotypic_attributes(store_path: "Path") -> dict:
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
```

Add `from pathlib import Path` and `import json` to the module header imports rather than
leaving them function-local once more than one function needs them.

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/unit/sdk_/test_ngff_attributes.py -v
```

Expected: all PASS. If `metadata_member_for_header` is not importable from
`phenotypic.schema`, confirm the correct name with
`uv run python -c "import phenotypic.schema as s; print([n for n in dir(s) if 'metadata' in n])"`
and use that — do **not** substitute a `startswith("Metadata_")` check.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/ngff_.py tests/unit/sdk_/test_ngff_attributes.py
git commit -m "feat(sdk): add the attributes.phenotypic contract

series and labels are separate keys, so a reader never has to special-case
which values are series names and which are nested paths. store_schema_version
and metadata_schema_version stay two markers. image_class (loader dispatch)
and Metadata_ImageType (user-visible schema metadata) stay distinct. work_id
is a constructor argument because the root zarr.json is written last and a
post-hoc patch would violate the ordering invariant."
```

---

### Task 1.4: The write-only OME projection

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py`
- **Create: `tests/_ngff_conformance.py`** — the OME-XML half only: `_ome_xsd()` and
  `assert_ome_xml_valid`. Phase 2 Task 2.5 extends the same file with the JSON-schema half
  and `assert_store_conforms`.
- Test: `tests/unit/sdk_/test_ngff_projection.py`

> **Why the harness starts here and not in Phase 2 (ledger ALGO-R2B-10).** This task's
> `test_ome_xml_validates_against_the_vendored_xsd` imports `assert_ome_xml_valid`, and an
> earlier draft created that module in Phase 2 — so Phase 1's own exit criterion
> (`uv run pytest tests/unit/sdk_/test_ngff_*.py -q`) would have failed with
> `ModuleNotFoundError` on the two tests that import it — the imports are function-local, so
> collection itself succeeds (ledger **GEN-50**). The XSD half needs only Phase 0's vendored `ome.xsd` fixture
> and `xmlschema`, both already in place, and this is the ALGO-1 remediation — it belongs in
> the same commit as the builder it certifies, not one phase later where a red gate invites
> deleting the assertion.
>
> **The two function bodies are given in full below, in Step 3a — this is their only
> definition in the plan.** Phase 2 Task 2.5's code block opens with a
> `# (unchanged from Task 1.4)` marker instead of repeating them: copying is what *causes*
> drift, not what prevents it, and nothing would check the two copies agreed. (This repo's
> one precedent for a deliberate duplicate — the two vendored `timeline.js` files — carries a
> CI byte-equality guard. Ledger **GEN-52**.)

**Interfaces:**
- Consumes: `axes_for`, `pyramid_level_shapes`, `level_scale_vector`, `AXIS_TYPES`;
  plus **Task 0.2's vendored `tests/fixtures/ome/2016-06/ome.xsd`** and **Task 0.1's
  `xmlschema` dependency**, both required by Step 3a (ledger **GEN-51**).
- Produces:
  ```python
  # NOTE (PRE-G1 / ALGO-5, user ruling): an earlier draft took a
# `resolution=(x_res, y_res)` argument, derived `axes[].unit` and a scaled
# level-0 vector from imported TIFF tags, and hard-coded "micrometer". Removed:
# the DSLR captures this project ingests carry no resolution tags, so the branch
# had no live input, no caller ever passed it, and no test covered it. It also
# carried a latent 25400x error -- `1.0 / x_res` treats a TIFF XResolution,
# which is px/INCH by default, as px/micrometre.
def build_multiscales(*, series, level_shapes, name=None) -> dict
  # NOTE: key on dtype, never on the series name. An earlier fix special-cased
# `detect_mat` by name and missed `gray`, which has an identical dtype and
# range.
def build_omero(*, series, dtype, bit_depth, name=None) -> dict
  def build_image_label() -> dict
  def build_ome_xml(*, series_names, series_shapes, series_dtypes, metadata_sections) -> str
  ```

**Constraints specific to this task:**
- The projection is **derived on every write and never read back**.
- `coordinateTransformations` carries exactly one `scale` entry per dataset, computed
  from actual level shapes.
- **`omero` is emitted completely or not at all.** Each channel carries a 6-hex-digit
  `color` and a `window` with all four of `min`, `max`, `start`, `end`, plus `active`,
  `family`, `coefficient`, `inverted`, `label`. `max`/`end` are `2**bit_depth - 1`.
  `rgb` emits three channels; `gray` emits one white channel.
- **`omero` is omitted from every FLOAT series, and from label groups.** In practice that
  means `rgb` is the only series that carries an `omero` block, and an rgb-less store
  carries none at all — which is fine: NGFF makes `omero` conditional (§2.5, *"The 'omero'
  metadata is optional"*) and the whole-or-nothing rule is **per group**.

  Both `gray` and `detect_mat` are float. Verified by execution on
  `load_synth_yeast_plate()`: `gray` is `float32` with range `[0.545, 0.955]` while
  `bit_depth` is 8 — so a `2**bit_depth - 1` window puts `[0,255]` over `[0,1]` data and
  every viewer honouring `omero` renders it near-black. `gray` is the **primary series in
  every rgb-less store**, i.e. the layer an external reader opens by default, which makes it
  the worst possible place for that defect. The spec's §2.2 applies the bit-depth window to
  every series; that is **superseded**. See OPEN-QUESTIONS **P2** and **ALGO-2**.

  **Keyed on dtype, not on the series name.** An earlier fix special-cased `detect_mat` by
  name and therefore missed `gray`, which has identical dtype and range. Keying on dtype
  makes the rule self-maintaining: any future float layer is handled correctly without
  anyone remembering to add it to a list.

  **Deferred, deliberately (user ruling):** making these layers *render* — by converting
  them to an integer dtype, or by deriving the window from the actual range — is postponed
  until there is data on the effect on analysis quality. Note what NGFF does and does not
  require here: integer pixels are mandated only for **label** images (§2.6, *"The pixels of
  the label images MUST be integer data types"*); image series are unconstrained, so
  omitting `omero` is fully conformant and nothing is being worked around. Converting the
  data would additionally break the bit-exact round-trip §7 requires, quantize an analysis
  input, and run into `detect_mat` values that are not bounded to `[0, 1]` — which is the
  `Image` data-model change spec §10 explicitly defers to its own design.
- **`image-label` is always emitted**, with `version`, `source: {"image": "../../"}`, and
  a `colors` list carrying **only the background entry** `{"label-value": 0, "rgba":
  [0, 0, 0, 0]}`.

  The spec's §2.2 requires one entry per unique label value. That is **superseded**
  (OPEN-QUESTIONS **P1**): `label.schema`'s `$defs/image-label` carries **no `required`
  list**, so `colors` is optional and a background-only entry conforms. Nothing in PhenoTypic
  reads it (the GUI colourises via `skimage.color.label2rgb`,
  `gui/builder/_image_renderer.py:155-166`); only the conformance gate and external viewers
  do, and external viewers fall back to their own palette. A background-only list can never go
  stale — it is not a function of the array contents at all — and drops the ~60 KB per-plate
  JSON §2.3 budgeted for.

  > **The reason changed; the conclusion did not (ledger GEN-22).** An earlier draft justified
  > this by *"Stage 2 overwrites the objmap in place without re-promoting, so a per-value
  > palette written at Stage 1 describes a zeros array."* Round 2 removed Stage 2's in-store
  > write entirely, so that premise is **false** — and a reader who noticed would reasonably
  > reinstate the per-value palette on the strength of the reason having evaporated. PRE-P1's
  > justification above never depended on the in-place write and still holds.
- `properties` is never emitted (locked decision #10).
- **`build_ome_xml` raises on failure; there is no fallback** (user ruling; ledger
  **PRE-G2** / **ALGO-3**). §2.2.3's alternative to a `series` list is *consecutively
  numbered groups*, and the plan's earlier "drop `series`, keep named groups" behaviour
  satisfies neither arm — it is strictly **less** conformant than either. Since the builder
  is string formatting over already-validated data, a failure means a bug or an unmapped
  dtype, and both should stop the run. The previous `except Exception: return None` is
  exactly what would have hidden the `header_to_module(key)` API mistake and shipped every
  store without an `OME/` group, silently.
- **The document must be valid against `ome.xsd` 2016-06** (user ruling; ledger **ALGO-1**):
  `<Pixels>` carries `ID`, `DimensionOrder`, `Type`, `SizeX/Y/Z/C/T` and a `<MetadataOnly/>`
  child, and `<M>` entries sit inside a `<Value>` wrapper. The XSD is vendored in Phase 0
  Task 0.2 and the harness validates against it (Phase 2 Task 2.5).
- The level-0 `scale` is the level-ratio vector and `unit`
  is omitted.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/test_ngff_projection.py`:

```python
"""The write-only OME projection. Never read back; validated on write."""

from __future__ import annotations

import re

import pytest

from phenotypic.sdk_ import ngff_


def test_multiscales_scale_comes_from_actual_level_shapes() -> None:
    shapes = ngff_.pyramid_level_shapes((1025, 7), 3)
    block = ngff_.build_multiscales(series="gray", level_shapes=shapes, name="plate")
    scales = [
        transform["scale"]
        for dataset in block["multiscales"][0]["datasets"]
        for transform in dataset["coordinateTransformations"]
        if transform["type"] == "scale"
    ]
    assert scales[0] == pytest.approx([1.0, 1.0])
    assert scales[1] == pytest.approx([1025 / 513, 7 / 4])
    assert scales[1][0] != pytest.approx(2.0)


def test_multiscales_axes_are_ordered_channel_then_space() -> None:
    shapes = ngff_.pyramid_level_shapes((3, 1024, 1024), 2)
    block = ngff_.build_multiscales(series="rgb", level_shapes=shapes)
    axes = block["multiscales"][0]["axes"]
    assert [axis["name"] for axis in axes] == ["c", "y", "x"]
    assert [axis["type"] for axis in axes] == ["channel", "space", "space"]


def test_multiscales_dataset_paths_are_level_indices() -> None:
    shapes = ngff_.pyramid_level_shapes((2048, 2048), 3)
    block = ngff_.build_multiscales(series="gray", level_shapes=shapes)
    assert [d["path"] for d in block["multiscales"][0]["datasets"]] == ["0", "1", "2"]


def test_omero_emits_every_required_channel_field() -> None:
    """NGFF is conditionally strict: partial omero fails the conformance gate."""
    block = ngff_.build_omero(
        series="rgb", dtype=np.dtype("uint16"), bit_depth=16, name="plate"
    )
    channels = block["omero"]["channels"]
    assert len(channels) == 3
    for channel in channels:
        assert re.fullmatch(r"[0-9A-F]{6}", channel["color"]), channel
        assert set(channel["window"]) == {"min", "max", "start", "end"}
        assert channel["window"]["max"] == 65535
        assert channel["window"]["end"] == 65535
        for key in ("label", "active", "family", "coefficient", "inverted"):
            assert key in channel


def test_omero_window_max_tracks_bit_depth() -> None:
    block = ngff_.build_omero(
        series="rgb", dtype=np.dtype("uint8"), bit_depth=8, name=None
    )
    assert block["omero"]["channels"][0]["window"]["max"] == 255


@pytest.mark.parametrize("series", ["gray", "detect_mat"])
def test_omero_is_omitted_for_every_float_series(series: str) -> None:
    """A float layer in [0,1] under a bit-depth window renders near-black.

    `gray` matters most: it is the PRIMARY series in every rgb-less store, so
    it is the layer an external reader opens by default. Verified by execution
    that it is float32 in [0.545, 0.955] while bit_depth is 8 -- identical to
    detect_mat, which is why keying on the series NAME missed it (ALGO-2).
    """
    assert (
        ngff_.build_omero(
            series=series, dtype=np.dtype("float32"), bit_depth=8, name=None
        )
        == {}
    )


def test_omero_is_keyed_on_dtype_not_on_the_series_name() -> None:
    """Self-maintaining: a future float layer needs no list entry, and an
    integer `gray` would get its block back automatically if the deferred
    conversion ever lands."""
    assert (
        ngff_.build_omero(series="gray", dtype=np.dtype("uint8"), bit_depth=8, name=None)
        != {}
    )
    assert (
        ngff_.build_omero(
            series="rgb", dtype=np.dtype("float32"), bit_depth=8, name=None
        )
        == {}
    )


def test_image_label_is_always_emitted_with_version_and_source() -> None:
    block = ngff_.build_image_label()
    assert block["image-label"]["version"] == "0.5"
    assert block["image-label"]["source"] == {"image": "../../"}


def test_image_label_colors_is_background_only() -> None:
    """`colors` is optional -- $defs/image-label has no `required` list -- and
    nothing in PhenoTypic reads it (P1)."""
    block = ngff_.build_image_label()
    assert block["image-label"]["colors"] == [{"label-value": 0, "rgba": [0, 0, 0, 0]}]


def test_image_label_takes_no_label_values() -> None:
    """It must not depend on array contents; that is what keeps it constant-size."""
    import inspect

    assert inspect.signature(ngff_.build_image_label).parameters == {}


def test_properties_is_never_emitted() -> None:
    """Locked decision #10: parquet stays the only measurement surface."""
    assert "properties" not in ngff_.build_image_label()["image-label"]


def test_image_label_is_constant_size_regardless_of_colony_count() -> None:
    """Drops the ~60 KB per-plate JSON the spec's OQ9 budgeted for."""
    import json

    assert len(json.dumps(ngff_.build_image_label())) < 500


def _xml_kwargs() -> dict:
    return {
        "series_names": ["rgb", "gray", "detect_mat"],
        "series_shapes": {
            "rgb": (3, 64, 48),
            "gray": (64, 48),
            "detect_mat": (64, 48),
        },
        "series_dtypes": {
            "rgb": np.dtype("uint8"),
            "gray": np.dtype("float32"),
            "detect_mat": np.dtype("float32"),
        },
        "metadata_sections": {
            "protected": {"Metadata_ImageName": "plate_01"},
            "public": {},
            "imported": {"TIFF:XResolution": 300.0},
        },
    }


def test_ome_xml_names_every_series_in_order() -> None:
    xml = ngff_.build_ome_xml(**_xml_kwargs())
    assert xml.count("<Image ") == 3
    assert xml.index("rgb") < xml.index("gray") < xml.index("detect_mat")


def test_map_annotation_namespaces_are_rembi_module_VALUES() -> None:
    """Not `str(enum)`, which is the Python repr since 3.11 (ledger ALGO-10).

    ome.xsd cannot catch this -- Annotation/@Namespace is xsd:anyURI, which
    accepts 'REMBI_MODULE.IMAGE_DATA' happily. Only an explicit assertion does.
    """
    xml = ngff_.build_ome_xml(**_xml_kwargs())
    assert "REMBI_MODULE" not in xml, "a Python-internal name leaked into OME-XML"
    assert 'Namespace="ImageData"' in xml


def test_control_characters_in_imported_metadata_do_not_break_the_document() -> None:
    """Real EXIF carries NUL-padded strings; XML 1.0 forbids them outright.

    `xml.sax.saxutils.escape` handles only & < >, so a NUL survives it and the
    document is not well-formed -- and `build_ome_xml` is pure string
    formatting, so nothing raises. This project's inputs are DSLR/raw captures
    read through `exiftool -json -n`, and `_normalize_metadata_value` decodes
    bytes with errors="replace", which fixes invalid UTF-8 but leaves \x00
    intact. Ledger ALGO-R2B-11.
    """
    from tests._ngff_conformance import assert_ome_xml_valid

    kwargs = _xml_kwargs()
    kwargs["metadata_sections"] = {
        "imported": {"EXIF:Make": "Canon\x00\x00 EOS", "EXIF:\x0bBad": "ok"}
    }
    xml = ngff_.build_ome_xml(**kwargs)
    # assert_ome_xml_valid catches XMLSchemaException, which covers a
    # well-formedness failure as well as a schema violation -- so this one call
    # is the whole gate. (Do not add a bare `ElementTree.fromstring` probe: the
    # stdlib parser has no billion-laughs guard, and this text comes from user
    # image files.)
    assert_ome_xml_valid(xml)
    assert "\x00" not in xml and "\x0b" not in xml


def test_ome_xml_validates_against_the_vendored_xsd() -> None:
    """The whole point of bioformats2raw.layout: 3 (ALGO-1).

    An earlier draft emitted a bare `<Pixels />` with none of its eight required
    attributes and no `<MetadataOnly/>`, plus `<M>` entries directly under
    `<MapAnnotation>` -- three separate violations, none of which the
    `xml.count("<Image ")` assertion could see.
    """
    from tests._ngff_conformance import assert_ome_xml_valid

    assert_ome_xml_valid(ngff_.build_ome_xml(**_xml_kwargs()))


def test_every_pixels_element_is_metadata_only() -> None:
    """§2.2.3: MUST use <MetadataOnly/>, never BinData/BinaryOnly/TiffData."""
    xml = ngff_.build_ome_xml(**_xml_kwargs())
    assert xml.count("<MetadataOnly/>") == 3
    for forbidden in ("<BinData", "<BinaryOnly", "<TiffData"):
        assert forbidden not in xml


def test_pixel_type_follows_the_dtype() -> None:
    xml = ngff_.build_ome_xml(**_xml_kwargs())
    assert 'Type="uint8"' in xml       # rgb
    assert 'Type="float"' in xml       # gray / detect_mat, float32


def test_an_unmapped_dtype_raises_rather_than_degrading() -> None:
    """PixelType is a closed enum; a silent fallback ships an invalid file."""
    kwargs = _xml_kwargs()
    kwargs["series_dtypes"]["gray"] = np.dtype("float16")
    with pytest.raises(ValueError, match="PixelType"):
        ngff_.build_ome_xml(**kwargs)


# NOTE (ledger SIMP-18): the propagation test lives in Phase 2 Task 2.2 as
# `test_a_failed_ome_xml_build_aborts_the_write`, which asserts the same
# propagation PLUS the consequence that matters -- that no store is left behind.
# It is strictly stronger and guards the same regression (someone re-adding
# `except Exception: return None`), so a unit-level twin here would be pure
# duplication. The neighbouring tests are NOT redundant with the XSD and stay:
# the OME content model permits BinData/TiffData (only NGFF 2.2.3 forbids them),
# and the XSD validates `Type` against its enum, not against the array's actual
# dtype -- so a float32 -> "uint8" mapping bug passes XSD cleanly.
```

- [ ] **Step 2: Run it to verify it fails**

Two distinct failures are expected here, and both are correct (ledger **GEN-48**): the
`build_*` tests fail with `AttributeError` on `ngff_`, and the two OME-XML tests fail with
`ModuleNotFoundError: No module named 'tests._ngff_conformance'` because Step 3a has not run
yet. Under this plan's TDD discipline an unexpected error message is a stop-and-investigate
signal, so both are named.

```bash
uv run pytest tests/unit/sdk_/test_ngff_projection.py -v
```

Expected: FAIL with `AttributeError: … has no attribute 'build_multiscales'`.

- [ ] **Step 3a: Create `tests/_ngff_conformance.py`**

The OME-XML half of the conformance harness. Phase 2 Task 2.5 extends this same file with the
JSON-schema half; **these two functions are defined here and nowhere else** (ledger
**GEN-52**).

```python
"""Conformance harness for written NGFF stores."""

from __future__ import annotations

import functools
from pathlib import Path


@functools.lru_cache(maxsize=1)
def _ome_xsd() -> "xmlschema.XMLSchema":
    """Load the vendored OME-XML schema.

    NGFF 2.2.3 makes the document a conditional MUST, and JSON-schema
    validation says nothing about it.
    """
    import xmlschema

    path = Path(__file__).resolve().parent / "fixtures" / "ome" / "2016-06" / "ome.xsd"
    if not path.is_file():
        raise AssertionError(
            f"vendored OME schema missing: {path}. A conformance check that "
            "cannot run must fail, never skip."
        )
    return xmlschema.XMLSchema(str(path))


def assert_ome_xml_valid(xml: str) -> None:
    """Validate an OME-XML document against the vendored `ome.xsd`.

    Catches ``XMLSchemaException``, not ``XMLSchemaValidationError``: the
    narrower class does not cover a well-formedness failure, which raises
    ``XMLResourceParseError`` -- and that is the most likely real failure, since
    a control character in an imported EXIF tag breaks well-formedness rather
    than schema conformance (ledger **ALGO-R2B-11**).

    Args:
        xml: The document as a string.

    Raises:
        AssertionError: On any schema violation OR malformed document.
    """
    import xmlschema

    try:
        _ome_xsd().validate(xml)
    except xmlschema.XMLSchemaException as exc:
        # Name the exception type: the widened catch also takes
        # XMLSchemaOSError/XMLResourceOSError, and a disk error reported as
        # "not valid against ome.xsd" would be misread as a conformance bug.
        raise AssertionError(
            f"OME-XML failed validation against ome.xsd "
            f"[{type(exc).__name__}]: {exc}"
        ) from exc
```

> **This is the FINAL body of both functions — Phase 2 Task 2.5 does not restate them**
> (ledger **M2**). An earlier draft gave them here *without* the `type(exc).__name__` rider
> and again, with it, in Task 2.5's "extend" block: two bodies for one pair of functions,
> differing in exactly the detail one of them was added to fix. A literal executor keeps the
> first and silently loses the rider — which is the drift GEN-52 forbade, manufactured inside
> the plan. There is now one definition, here.

- [ ] **Step 3: Append to `ngff_.py`**

Add `import re as _re` to the module header in this step — **not in Task 1.1**. Its only user
is `_XML_FORBIDDEN` below, three tasks later, and CLAUDE.md prescribes `ruff check --fix` at
each completion boundary, where F401 would remove an unused import and break this task
(ledger **GEN-51**).

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/unit/sdk_/test_ngff_projection.py -v
```

Expected: all PASS. If `header_to_module` is not importable from `phenotypic.schema`,
confirm the name with
`uv run python -c "import phenotypic.schema as s; print([n for n in dir(s) if 'module' in n])"`.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/ngff_.py tests/_ngff_conformance.py tests/unit/sdk_/test_ngff_projection.py
git commit -m "feat(sdk): add the write-only OME projection

multiscales scale vectors come from actual level shapes. omero is emitted
completely or not at all -- NGFF requires a 6-hex color and all four window
bounds per channel, and a partial projection would fail the conformance gate
on the first store written. image-label is always emitted (label.schema
requires it despite the prose saying SHOULD) with colors carrying ONLY the
transparent background entry -- $defs/image-label has no required list, so
colors is optional, and a background-only list is not a function of the array
contents and so cannot go stale -- and no properties block. build_ome_xml RAISES
rather than returning None: it is string formatting over validated data, so a
failure is a bug, and the old swallow-and-degrade would have shipped every
store without an OME/ group."
```

---

### Task 1.5: The promote primitive, durability policy, and orphan sweep

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py`
- Test: `tests/unit/sdk_/test_ngff_promote.py`

**Interfaces:**
- Consumes: `STORE_SUFFIX`.
- Produces:
  ```python
  PART_SUFFIX: Final[str] = ".part"
  TRASH_SUFFIX: Final[str] = ".trash"
  PROMOTE_RETRY_ATTEMPTS: Final[int] = 5
  PROMOTE_RETRY_BASE_SECONDS: Final[float] = 0.1

  def durable_writes_enabled(override: bool | None = None) -> bool
  def describe_durability(override: bool | None = None) -> str
  # NOTE (PRE-G3): an earlier draft used this in three places while
# `_write_group_json`, `promote_store`'s renames, `fsync_tree`,
# `read_root_attributes`, and `sweep_orphan_parts` went without -- which was
# most of the paths that actually approach MAX_PATH.
def long_path(path: Path) -> str
  def new_part_path(final: Path) -> Path
  def fsync_tree(root: Path) -> None
  def promote_store(part: Path, final: Path, *, fsync: bool) -> Path
  DOWNSAMPLE_METHODS: Final[dict[str, tuple[str, str]]]   # kind -> (method, description)
  DOWNSAMPLE_KINDS: Final[dict[str, str]]                # the bare kind -> method view
  SWEEP_MIN_AGE_SECONDS: Final[float] = 6 * 60 * 60
  def discard_parts_for(final: Path) -> int
def sweep_orphan_parts(results_root: Path, *, min_age_seconds: float = SWEEP_MIN_AGE_SECONDS) -> int
  ```

**Constraints specific to this task:**
- `new_part_path(final)` returns `final.parent / f".{final.name}.{uuid4().hex}{PART_SUFFIX}"`
  — a **sibling**, with a uuid4 hex, never a PID. Two concurrent SLURM tasks must get
  distinct directories.
- `promote_store` order: (1) if `final` exists, `os.replace(final, trash)`;
  (2) `os.replace(part, final)`; (3) `rmtree(trash)`. Steps 1 and 2 are each wrapped in
  retry-with-backoff, reusing the shape of `_open_hdf_with_recovery` (`sdk_/hdf_.py:34`).
- `promote_store` does **not** write the root `zarr.json` — the caller writes arrays, then
  `OME/zarr.json`, then the root, then calls this. Document that contract on the function.
- `fsync_tree` fsyncs every regular file, then the directory itself; the directory step is
  **POSIX-guarded** (`os.name == "posix"`), because Windows cannot open a directory handle.
- `durable_writes_enabled` returns `override` when it is not `None`; otherwise detects
  SLURM from `SLURM_CPUS_PER_TASK` / `SLURM_JOB_ID` exactly as `resolve_worker_count`
  (`_cli/_cli_utils.py:65`) does.
- `sweep_orphan_parts` removes `.part` and `.trash` directories **by suffix match on the
  uuid-bearing name**, never by PID, and returns the count removed.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/test_ngff_promote.py`:

```python
"""Rename-promote commit protocol: uuid parts, move-aside, sweep, durability."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from phenotypic.sdk_ import ngff_


def _fake_store(root: Path, marker: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "0").mkdir(exist_ok=True)
    (root / "0" / "c.0.0.0").write_bytes(b"chunk")
    (root / "zarr.json").write_text(f'{{"marker": "{marker}"}}', encoding="utf-8")
    return root


def test_part_path_is_a_sibling_hidden_directory(tmp_path: Path) -> None:
    final = tmp_path / "plate_01.ome.zarr"
    part = ngff_.new_part_path(final)
    assert part.parent == final.parent
    assert part.name.startswith(".plate_01.ome.zarr.")
    assert part.name.endswith(".part")


def test_part_paths_are_distinct_across_concurrent_writers(tmp_path: Path) -> None:
    """A PID can be reused; a uuid4 cannot. Two writers must never share a dir."""
    final = tmp_path / "plate_01.ome.zarr"
    parts = {ngff_.new_part_path(final) for _ in range(64)}
    assert len(parts) == 64


def test_part_name_carries_no_pid(tmp_path: Path) -> None:
    part = ngff_.new_part_path(tmp_path / "plate_01.ome.zarr")
    assert str(os.getpid()) not in part.name.replace(".part", "")


def test_discard_parts_for_is_scoped_to_one_store(tmp_path: Path) -> None:
    """A sibling store's in-flight .part must survive (ledger GEN-26).

    The glob is anchored on `final.name`, and this is the assertion that keeps
    the CLI save-failure path from wiping a concurrent writer's work.
    """
    mine = tmp_path / "a.ome.zarr"
    theirs = tmp_path / "b.ome.zarr"
    stale = ngff_.new_part_path(mine)
    stale.mkdir(parents=True)
    live = ngff_.new_part_path(theirs)
    live.mkdir(parents=True)

    assert ngff_.discard_parts_for(mine) == 1
    assert not stale.exists()
    assert live.is_dir(), "a sibling store's .part is not ours to remove"


def test_promote_onto_absent_target(tmp_path: Path) -> None:
    final = tmp_path / "plate_01.ome.zarr"
    part = _fake_store(ngff_.new_part_path(final), "new")
    result = ngff_.promote_store(part, final, fsync=False)
    assert result == final
    assert (final / "zarr.json").read_text(encoding="utf-8") == '{"marker": "new"}'
    assert not part.exists()


def test_promote_replaces_a_non_empty_existing_store(tmp_path: Path) -> None:
    """os.replace onto a non-empty directory raises ENOTEMPTY; the move-aside
    is what makes the promote work at all, on POSIX and on Windows alike."""
    final = _fake_store(tmp_path / "plate_01.ome.zarr", "old")
    part = _fake_store(ngff_.new_part_path(final), "new")
    ngff_.promote_store(part, final, fsync=False)
    assert (final / "zarr.json").read_text(encoding="utf-8") == '{"marker": "new"}'


def test_promote_leaves_no_trash_behind(tmp_path: Path) -> None:
    final = _fake_store(tmp_path / "plate_01.ome.zarr", "old")
    part = _fake_store(ngff_.new_part_path(final), "new")
    ngff_.promote_store(part, final, fsync=False)
    assert [p.name for p in tmp_path.iterdir()] == ["plate_01.ome.zarr"]


def test_bare_os_replace_onto_a_non_empty_directory_still_fails(tmp_path: Path) -> None:
    """Pins the reason the two-step move-aside is mandatory, not defensive."""
    src = _fake_store(tmp_path / "src", "a")
    dst = _fake_store(tmp_path / "dst", "b")
    with pytest.raises(OSError):
        os.replace(src, dst)


def test_sweep_removes_orphan_parts_and_trash(tmp_path: Path) -> None:
    """`min_age_seconds=0` because the fixtures are microseconds old.

    The production default is 6 h — see
    `test_the_sweep_spares_a_young_leftover`, which is the behaviour the age
    guard was added for.
    """
    dataset = tmp_path / "results" / "ds" / "zarr"
    dataset.mkdir(parents=True)
    _fake_store(dataset / "keep.ome.zarr", "keep")
    _fake_store(dataset / ".keep.ome.zarr.deadbeef.part", "orphan")
    _fake_store(dataset / ".keep.ome.zarr.cafef00d.trash", "orphan")
    removed = ngff_.sweep_orphan_parts(tmp_path / "results", min_age_seconds=0)
    assert removed == 2
    assert (dataset / "keep.ome.zarr").is_dir()
    assert list(dataset.glob("*.part")) == []
    assert list(dataset.glob("*.trash")) == []


def test_sweep_is_idempotent_on_a_clean_tree(tmp_path: Path) -> None:
    dataset = tmp_path / "results" / "ds" / "zarr"
    dataset.mkdir(parents=True)
    _fake_store(dataset / "keep.ome.zarr", "keep")
    assert ngff_.sweep_orphan_parts(tmp_path / "results", min_age_seconds=0) == 0


def test_the_sweep_spares_a_young_leftover(tmp_path: Path) -> None:
    """The whole point of the age guard: a uuid gives no liveness signal, so
    under a SLURM array a sibling task may be mid-write into this directory."""
    dataset = tmp_path / "results" / "ds" / "zarr"
    dataset.mkdir(parents=True)
    live = _fake_store(dataset / ".keep.ome.zarr.deadbeef.part", "in flight")
    assert ngff_.sweep_orphan_parts(tmp_path / "results") == 0
    assert live.is_dir()


def test_durable_writes_honour_an_explicit_override(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert ngff_.durable_writes_enabled(True) is True
    assert ngff_.durable_writes_enabled(False) is False


def test_durable_writes_default_off_locally(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert ngff_.durable_writes_enabled(None) is False


def test_durable_writes_default_on_under_slurm(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert ngff_.durable_writes_enabled(None) is True


def test_durability_is_describable_for_the_run_start_log(monkeypatch) -> None:
    """The same command carries different guarantees in different places, so
    the resolved mode must be loggable, not merely resolvable."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert ngff_.describe_durability(None) == "durable writes: on (SLURM)"
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert ngff_.describe_durability(None) == "durable writes: off (local)"
    assert ngff_.describe_durability(True) == "durable writes: on (--durable-writes)"
    assert (
        ngff_.describe_durability(False) == "durable writes: off (--no-durable-writes)"
    )


def test_fsync_tree_runs_without_error_on_a_real_store(tmp_path: Path) -> None:
    store = _fake_store(tmp_path / "s.ome.zarr", "x")
    ngff_.fsync_tree(store)


@pytest.mark.skipif(os.name != "nt", reason="Windows path-prefix behaviour")
def test_long_path_prefixes_on_windows(tmp_path: Path) -> None:
    assert ngff_.long_path(tmp_path).startswith("\\\\?\\")


@pytest.mark.skipif(os.name == "nt", reason="POSIX passthrough")
def test_long_path_is_a_passthrough_on_posix(tmp_path: Path) -> None:
    assert ngff_.long_path(tmp_path) == str(tmp_path)


def test_store_path_segments_have_no_case_only_collisions() -> None:
    """NTFS is case-insensitive; asserted by test rather than by inspection."""
    segments = [
        ngff_.OME_GROUP,
        ngff_.LABELS_GROUP,
        ngff_.OBJMAP_LABEL,
        *ngff_.SERIES_ORDER,
    ]
    assert len({s.lower() for s in segments}) == len(segments)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/sdk_/test_ngff_promote.py -v
```

Expected: FAIL with `AttributeError: … has no attribute 'new_part_path'`.

- [ ] **Step 3: Append to `ngff_.py`**

```python
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
    import os as _os

    if override is True:
        return True, "--durable-writes"
    if override is False:
        return False, "--no-durable-writes"
    on_slurm = bool(
        _os.environ.get("SLURM_JOB_ID") or _os.environ.get("SLURM_CPUS_PER_TASK")
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
    """
    import os as _os
    from pathlib import Path as _Path

    resolved = _Path(path).resolve()
    if _os.name != "nt":
        return str(resolved)
    text = str(resolved)
    return text if text.startswith("\\\\?\\") else "\\\\?\\" + text


def new_part_path(final: "Path") -> "Path":
    """Return a fresh, uuid-suffixed ``.part`` sibling of *final*.

    The uuid -- matching the ``attempt_id = uuid4().hex`` convention already
    used in ``_cli_staged_strategy.py`` (lines 148, 192, 225, 359) -- is what
    keeps two concurrent writers from interleaving chunks into one directory.
    It is NOT what makes the promote itself benign; that is the retry loop in
    :func:`promote_store`. An un-suffixed ``.part`` would let two concurrent SLURM tasks
    interleave chunks into one directory and produce a store that *validates*.
    A PID is not enough: PIDs are reused.
    """
    from pathlib import Path as _Path
    from uuid import uuid4

    final = _Path(final)
    return final.parent / f".{final.name}.{uuid4().hex}{PART_SUFFIX}"


def _fsync_path(path: Path) -> None:
    """``fsync`` one already-existing file or directory."""
    handle = os.open(path, os.O_RDONLY)
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
_RETRYABLE_WINERROR: Final[frozenset[int]] = frozenset({32, 33})  # SHARING_VIOLATION, LOCK_VIOLATION


def _is_retryable(exc: OSError) -> bool:
    """Whether *exc* is a transient contention error rather than a hard failure.

    Windows refuses to rename a directory while any file inside it is held open
    (``ERROR_SHARING_VIOLATION``); with ~40 files per store instead of one
    ``.h5``, that exposure is 40x larger. On POSIX, ``ENOTEMPTY``/``ENOENT`` on
    the target mean a concurrent promoter moved under us, which the retry loop
    resolves by re-evaluating.
    """
    import errno

    if getattr(exc, "winerror", None) in _RETRYABLE_WINERROR:
        return True
    return exc.errno in {errno.ENOTEMPTY, errno.ENOENT, errno.EEXIST}


def promote_store(part: "Path", final: "Path", *, fsync: bool) -> "Path":
    """Atomically promote a fully written ``.part`` directory to *final*.

    The caller is responsible for the write **order** inside *part*: all arrays
    and chunks first, then ``OME/zarr.json``, then the root ``zarr.json`` last.
    An interrupted store therefore has no valid root and reads as absent.

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
    import os as _os
    import shutil
    import time
    from pathlib import Path as _Path

    part, final = _Path(part), _Path(final)
    if fsync:
        fsync_tree(part)

    trash = final.parent / f"{part.name[:-len(PART_SUFFIX)]}{TRASH_SUFFIX}"
    last: OSError | None = None
    for attempt in range(PROMOTE_RETRY_ATTEMPTS):
        moved_aside = False
        try:
            # Re-evaluate existence EVERY attempt. A concurrent promoter can
            # create or remove `final` between the check and either rename, so
            # a check-then-act done once outside the loop turns a benign
            # duplicate execution into a hard failure.
            if final.exists():
                _os.replace(final, trash)
                moved_aside = True
            _os.replace(part, final)
        except OSError as exc:
            last = exc
            if not _is_retryable(exc):
                if moved_aside and trash.exists() and not final.exists():
                    _os.replace(trash, final)
                raise
            if moved_aside and trash.exists() and not final.exists():
                # Roll back. Without this the previous store is already in
                # `trash` and is about to be deleted, leaving NO copy at any
                # path -- a data-loss mode the single-file HDF rename never had
                # (a failed os.replace(tmp, final) left `final` untouched).
                _os.replace(trash, final)
            time.sleep(PROMOTE_RETRY_BASE_SECONDS * (2**attempt))
            continue
        # Success: only now is the previous store safe to discard.
        if trash.exists():
            shutil.rmtree(trash, ignore_errors=True)
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


def discard_parts_for(final: "Path") -> int:
    """Remove every ``.part`` sibling belonging to one target store.

    Shares the naming convention with :func:`new_part_path` and
    :func:`sweep_orphan_parts` so no caller re-encodes it. The CLI's
    save-failure path would otherwise hand-roll the dot-prefix + uuid + suffix
    glob outside this module (ledger **SIMP-6**).

    Args:
        final: The target store path whose parts should be discarded.

    Returns:
        Number of directories removed.
    """
    import shutil

    removed = 0
    for path in final.parent.glob(f".{final.name}.*{PART_SUFFIX}"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


def sweep_orphan_parts(
    results_root: "Path", *, min_age_seconds: float = SWEEP_MIN_AGE_SECONDS
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
    import os as _os
    import shutil
    import time
    from pathlib import Path as _Path

    removed = 0
    root = _Path(results_root)
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
            if not (path.name.endswith(PART_SUFFIX) or path.name.endswith(TRASH_SUFFIX)):
                continue
            if STORE_SUFFIX not in path.name:
                continue
            if _os.stat(path).st_mtime > cutoff:
                continue  # may still be in flight
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/unit/sdk_/test_ngff_promote.py -v
```

Expected: all PASS on Linux, with the two `long_path` tests split by `skipif`.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/ngff_.py tests/unit/sdk_/test_ngff_promote.py
git commit -m "feat(sdk): add the rename-promote commit primitive

.part directories carry a uuid4, not a PID, so two concurrent SLURM tasks
can never interleave chunks into one directory and produce a store that
validates. The promote is a two-step move-aside because os.replace onto a
non-empty directory raises ENOTEMPTY on POSIX and MOVEFILE_REPLACE_EXISTING
cannot name a directory on Windows; both renames retry with backoff for
ERROR_SHARING_VIOLATION. fsync is on under SLURM and off locally, with an
explicit override and a describable mode for the run-start log."
```

---

### Task 1.6: `valid_staged_store`

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py`
- Test: `tests/unit/sdk_/test_ngff_validity.py`

**Interfaces:**
- Consumes: `read_phenotypic_attributes`, `PhenotypicAttr`.
- Produces:
  ```python
  def store_level0_shape(store_path: Path, member_path: str) -> tuple[int, ...] | None
  def valid_staged_store(path: Path) -> bool
  ```

**Constraints specific to this task:**
`valid_staged_store` mirrors `valid_staged_hdf` (`_cli/_cli_staged_resume.py:69`) case for
case.

> **The spec's §3.6 exception argument is inverted — do not copy it.** §3.6 says
> "none of zarr's error types are `ValueError` subclasses". The opposite is true:
> `zarr.errors.BaseZarrError` inherits **directly from `ValueError`**
> (https://zarr.readthedocs.io/en/stable/api/zarr/errors/), as do
> `MetadataValidationError` and every other zarr error except the four
> `IndexError` ones, which this function cannot raise. `json.JSONDecodeError` is
> likewise a `ValueError` and `FileNotFoundError` an `OSError`. The tuple is
> therefore the HDF set **plus `KeyError`** — which the attribute lookups need
> and the HDF version did not — and importing `zarr.errors` here is unnecessary
> in a function the resume planner calls once per image.

The cases:

- the root `zarr.json` parses and carries `phenotypic.store_schema_version`;
- **every** entry in `phenotypic.series` **and** `phenotypic.labels` opens as a Zarr array
  group — objmap included, which Stage 1's zeros write guarantees;
- level-0 `(y, x)` extents agree across all of them **and are non-zero** — a zero-size
  Zarr array is legal and must not pass;
- it catches `OSError`, `KeyError`, `ValueError`, `TypeError`, `json.JSONDecodeError`,
  `FileNotFoundError`, **and `zarr.errors.BaseZarrError`**. The HDF version's
  `(OSError, TypeError, ValueError)` set is insufficient — none of zarr's error types are
  `ValueError` subclasses.

`staged_store_matches_work_id` is **not** defined here; it stays in
`_cli_staged_resume.py` beside the classifier, mirroring today's placement of
`staged_hdf_matches_work_id` (Task 3.4).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/test_ngff_validity.py`:

```python
"""valid_staged_store mirrors valid_staged_hdf case for case."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from phenotypic.sdk_ import ngff_


def _write_store(
    root: Path,
    *,
    shapes: dict[str, tuple[int, ...]],
    series: list[str],
    with_root: bool = True,
    store_schema_version: int | None = 3,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    primary = ngff_.primary_series(series) if series else "gray"
    members = {name: name for name in series}
    labels = {"objmap": ngff_.objmap_path(primary)}
    for name, path in [*members.items(), *labels.items()]:
        if name not in shapes:
            continue
        array = np.zeros(shapes[name], dtype=np.uint16)
        zarr.create_array(
            store=str(root / path / "0"),
            **ngff_.array_create_kwargs(array.shape, array.dtype, name),
        )
    if with_root:
        block = {
            ngff_.PhenotypicAttr.SERIES: members,
            ngff_.PhenotypicAttr.LABELS: labels,
        }
        if store_schema_version is not None:
            block[ngff_.PhenotypicAttr.STORE_SCHEMA_VERSION] = store_schema_version
        (root / "zarr.json").write_text(
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {"ome": {"version": "0.5"}, "phenotypic": block},
                }
            ),
            encoding="utf-8",
        )
    return root


def test_complete_store_is_valid(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is True


def test_missing_store_is_invalid(tmp_path: Path) -> None:
    assert ngff_.valid_staged_store(tmp_path / "absent.ome.zarr") is False


def test_missing_root_zarr_json_is_invalid(tmp_path: Path) -> None:
    """Interrupted after chunks, before the root: reads as absent, by design."""
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
        with_root=False,
    )
    assert ngff_.valid_staged_store(store) is False


def test_root_without_store_schema_version_is_invalid(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
        store_schema_version=None,
    )
    assert ngff_.valid_staged_store(store) is False


def test_missing_objmap_is_invalid(tmp_path: Path) -> None:
    """Stage 1 writes a zeros objmap, so its absence means an incomplete write."""
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48)},
        series=["gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is False


def test_missing_detect_mat_is_invalid(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is False


def test_disagreeing_extents_are_invalid(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 47), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is False


def test_zero_extent_is_invalid(tmp_path: Path) -> None:
    """A zero-size Zarr array is legal; it must not pass validity."""
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (0, 48), "detect_mat": (0, 48), "objmap": (0, 48)},
        series=["gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is False


def test_rgb_store_attaches_labels_under_rgb(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={
            "rgb": (3, 64, 48),
            "gray": (64, 48),
            "detect_mat": (64, 48),
            "objmap": (64, 48),
        },
        series=["rgb", "gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is True
    block = ngff_.read_phenotypic_attributes(store)
    assert block[ngff_.PhenotypicAttr.LABELS]["objmap"] == "rgb/labels/objmap"


def test_corrupt_root_json_is_invalid_not_raising(tmp_path: Path) -> None:
    store = tmp_path / "a.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{not json", encoding="utf-8")
    assert ngff_.valid_staged_store(store) is False


def test_a_file_where_a_store_should_be_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "a.ome.zarr"
    path.write_bytes(b"not a directory")
    assert ngff_.valid_staged_store(path) is False


def test_a_malformed_array_metadata_is_invalid(tmp_path: Path) -> None:
    """A reachable zarr error, not a monkeypatched one.

    Replaces an earlier `test_zarr_errors_are_caught_not_propagated`, which
    could not fail: `BaseZarrError` subclasses `ValueError`, so the assertion
    held with or without it in the tuple.
    """
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
    )
    (store / "gray" / "0" / "zarr.json").write_text(
        '{"zarr_format": 3, "node_type": "array"}', encoding="utf-8"
    )
    assert ngff_.valid_staged_store(store) is False
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/sdk_/test_ngff_validity.py -v
```

Expected: FAIL with `AttributeError: … has no attribute 'valid_staged_store'`.

- [ ] **Step 3: Append to `ngff_.py`**

```python
# ---------------------------------------------------------------------------
# Resume validity
# ---------------------------------------------------------------------------


def store_level0_shape(store_path: "Path", member_path: str) -> tuple[int, ...] | None:
    """Return the level-0 shape of one member array, or ``None`` if absent.

    Args:
        store_path: Store root.
        member_path: Store-relative group path, e.g. ``"gray"`` or
            ``"rgb/labels/objmap"``.

    Returns:
        The level-0 array shape, or ``None`` when the level-0 array is missing.
    """
    import zarr
    from pathlib import Path as _Path

    level0 = _Path(store_path) / member_path / "0"
    if not level0.is_dir():
        return None
    return tuple(zarr.open_array(store=str(level0), mode="r").shape)


def valid_staged_store(path: "Path") -> bool:
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
    the HDF version did not.

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
    from pathlib import Path as _Path

    try:
        store = _Path(path)
        if not store.is_dir():
            return False
        block = read_phenotypic_attributes(store)
        if PhenotypicAttr.STORE_SCHEMA_VERSION not in block:
            return False
        members = [
            *block[PhenotypicAttr.SERIES].values(),
            *block[PhenotypicAttr.LABELS].values(),
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
    except (OSError, KeyError, TypeError, ValueError):
        return False
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/unit/sdk_/test_ngff_validity.py -v
```

Expected: all PASS. If `zarr.errors.BaseZarrError` does not exist in the resolved version,
find the actual base with
`uv run python -c "import zarr.errors as e; print([n for n in dir(e) if n.endswith('Error')])"`
and use the true base class — do **not** widen the handler to bare `Exception`.

> **`write_objmap_in_place` is NOT part of this plan.** An earlier draft added it here so
> Stage 2 could overwrite the promoted store's label array. That write was removed entirely
> (user ruling): only the **final** store needs third-party interop, so Stage 2 has no reason
> to touch the store at all. It writes its raw output to
> `.phenotypic/progress/stage2_raw/` and drops the token; the objmap enters the store exactly
> once, at Stage 3's promote, already post-refined.
>
> Removing it dissolves four open concerns rather than fixing one: **FLOW-5** (the uncached
> crop route serving raw pre-`drop_frame_background` labels), **FLOW-12** (a torn
> cross-level objmap, since there is no multi-level in-place write), **D11**
> (`--mode process --layer objmap` leaving raw output published — the residue is now
> Stage 1's zeros, exactly as the HDF path leaves today), and the **B10** cross-phase
> dependency, since the shared symbol no longer exists.
>
> The store therefore holds Stage 1's zeros objmap until Stage 3 publishes — **byte-for-byte
> the behaviour the HDF path has today**, where the detector output lived in the sidecar and
> never in the `.h5`.

- [ ] **Step 5: Export the public surface from `sdk_/__init__.py`**

Add `valid_staged_store`, `promote_store`, `new_part_path`, `sweep_orphan_parts`,
`durable_writes_enabled`, `describe_durability`, `PhenotypicAttr`, and `STORE_SUFFIX` to
`src/phenotypic/sdk_/__init__.py`'s imports and `__all__`, beside the existing `HDF`
export. Keep the list alphabetised as the file already is.

- [ ] **Step 6: Run the whole new module's suite plus the type gate**

```bash
uv run pytest tests/unit/sdk_/test_ngff_*.py -v
uv run ruff check --fix src/phenotypic/sdk_/ngff_.py src/phenotypic/sdk_/__init__.py tests/unit/sdk_
uv run mypy src/phenotypic/sdk_/ngff_.py
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/sdk_/ngff_.py src/phenotypic/sdk_/__init__.py tests/unit/sdk_/test_ngff_validity.py
git commit -m "feat(sdk): add valid_staged_store

Mirrors valid_staged_hdf case for case: root parses, store_schema_version
matches by VALUE (not merely present, so a future v4 store fails loudly
rather than being read under v3 semantics), every series AND label opens,
level-0 extents agree and are non-zero (a zero-size Zarr array is legal and
must not pass).

The exception set is the HDF version's plus KeyError. The spec's §3.6 claim
that 'none of zarr's error types are ValueError subclasses' is inverted --
BaseZarrError inherits directly from ValueError -- so no zarr import is
needed here, and the test that asserted otherwise could not fail."
```

---

## Phase 1 exit criteria

- [ ] `uv run pytest tests/unit/sdk_/test_ngff_*.py -q` is all green.
- [ ] `uv run python docs/superpowers/logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py` exits 0.
- [ ] `uv run mypy src/phenotypic/sdk_/ngff_.py` passes.
- [ ] `grep -n "2 \*\* n\|2\*\*n" src/phenotypic/sdk_/ngff_.py` finds nothing in scale computation.
- [ ] `grep -n "getpid" src/phenotypic/sdk_/ngff_.py` finds nothing.
- [ ] No existing test changed behaviour — the HDF path is untouched in this phase.
