# Pseudo-cropping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `CropImage` keeps the original frame. Images record a composed `CropFrame`, stores are written zero-padded to the original H×W, and every `info()` table gains `Frame_OffsetRR`/`Frame_OffsetCC`. No op or accessor changes behaviour.

**Architecture:** In memory, arrays stay physically cropped, exactly as today. The image carries a small frozen `CropFrame(canvas_shape, offset)` plus a protected `_pad_on_save` flag. The frame composes through every rectangular `Image.__getitem__` and survives `copy`/`set_image`/the provenance apply wrapper. Padding to the canvas happens only at write boundaries (`save2zarr`, `_save_store`, `save_intermediate_zarr`, accessor `imsave`), and `load_zarr` slices it back. Padded stores write `store_schema_version = 4`, every other store writes 3, and this build reads both.

**Tech Stack:** Python 3, numpy, pandas, zarr v3 / OME-NGFF 0.5, pydantic v2 operations, pytest. Run everything with `uv run`.

**Spec:** `docs/superpowers/specs/2026-10-05-pseudo-cropping/design.md`. Read it first; this plan argues from it. Its independent numeric witness is `docs/superpowers/logic_validation_scripts/2026-10-05-pseudo-cropping/crop_frame_invariants.py`.

## Global Constraints

- Work in the worktree `/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/pseudo-cropping` (branch `worktree-pseudo-cropping`). Never `cd` to the main checkout.
- `uv` is the only runner. Never call bare `python` or `pip`. In a fresh worktree, run `uv sync --group dev --group test-qt --extra gui --extra napari` once before the first test.
- **Focused test command** (use it everywhere below):
  `QT_QPA_PLATFORM=offscreen uv run pytest <paths> -q --capture=fd -p no:cacheprovider`
  (`--capture=fd` overrides the repo's `--capture=no` addopt). **Never `-n auto`, never `-x`** for a run whose result you will quote.
- Lint only the paths you changed: `uv run ruff check --fix <paths>`. Never run a bare `ruff check --fix`.
- Every existing `Bbox_*`, `Grid_*` and centroid value stays **byte-identical** to `main`.
- Unpadded stores (no frame, or `pad_on_save` resolved `False`) stay **bit-identical** to today's output: `store_schema_version` 3 and no new keys. The exception is the `crop_frame` key, written when a valid frame exists.
- Schema authoring: author `label` and `desc` only. Leave `bio_desc=""` and `image=None`.
- Operations are keyword-only pydantic models; `op.apply(image)`, never `op(image)`.
- Keep entry points lazy. `_core/_crop_frame.py` imports only `numpy` (and `pandas` inside a function).
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD
  ```
- Any run over a couple of minutes (the full suite, Task 17) is a Slurm job. Use the `run-phenotypic-test` and `slurm-job` skills.

## Review Focus

These are the inputs the spec implies but doesn't spell out, most likely to bite first. Each one has a pinning test in the task named.

1. **The Results Colony tab on a padded store.** Each colony's Viv camera is placed from table centroids, which are ROI-relative, onto a canvas-sized store. Expected: the view centres on the colony. → Task 15, `test_colony_cell_centroid_is_shifted_into_the_canvas`.
2. **A crop store written before this change, re-measured with `--mode measure`.** It has no `crop_frame` but was cropped. Expected: `Frame_Offset*` is null (not 0), with one warning. → Task 10, `test_pre_change_crop_store_measures_null_offsets` (the in-memory rule is pinned in Task 7).
3. **A 16-bit cropped plate's colony crops.** Expected: the zero margin does not drag the display range to 0 and wash crops out. → Task 15, `test_display_range_ignores_the_zero_margin`.
4. **A grayscale-only image (no RGB) that is cropped and saved.** Expected: `gray`, `detect_mat` and objmap are padded under the `gray` primary, and the round trip is exact. → Task 10, `test_grayscale_only_crop_round_trips`.
5. **A store whose `crop_frame` disagrees with its arrays or its version** (hand-edited, or written by a buggy tool). Expected: the loader refuses it and `valid_staged_store` returns False, so the image is re-derived instead of being windowed into a truncated image. → Task 8 `test_valid_staged_store_rejects_inconsistent_crop_frames`, and Task 10 `test_crop_frame_disagreeing_with_the_arrays_is_refused`.

The old item about pickling to workers was removed: CLI workers receive paths, not Images, so the frame crosses processes only through the store. Task 2's pickle test stays as a cheap regression pin.

---

## File map

| File | Change |
|---|---|
| `src/phenotypic/sdk_/_crop_frame.py` | **Create.** `CropFrame`, `crop_frame_to_attribute`, `crop_frame_from_attribute` (stdlib only, so `ngff_` can use them on light paths) |
| `src/phenotypic/_core/_crop_frame.py` | **Create.** `pad_to_canvas`, `rect_origin_from_key`, `journal_records_crop`, `append_frame_offsets`; re-exports the `sdk_` names |
| `src/phenotypic/_core/_image_parts/_image_data_manager.py` | frame attrs, `_valid_crop_frame`, `_save_padding_frame`, `_pad_layer_for_save`, `_assign_child_frame`, carry-through, reshape drop |
| `src/phenotypic/_core/_image_parts/_image_handler.py` | `__getitem__` composes the frame |
| `src/phenotypic/_core/_image_parts/_grid_image_handler.py` | `GridImage.__getitem__` composes the frame |
| `src/phenotypic/correction/_image_cropper.py` | sets `_pad_on_save = True` |
| `src/phenotypic/correction/_image_padder.py` | frame-aware shift / drop |
| `src/phenotypic/_core/_provenance.py` | frame carry-over in `_carry_logical_image_state` |
| `src/phenotypic/schema/_frame.py` | **Create.** `FRAME` enum |
| `src/phenotypic/schema/__init__.py` | export `FRAME` |
| `src/phenotypic/sdk_/_metadata_helpers.py` | `Frame_` joins the info block |
| `src/phenotypic/_core/_image_parts/accessors/_objects_accessor.py`, `_grid_accessor.py` | append offsets in `info()` |
| `src/phenotypic/sdk_/ngff_.py` | `CROP_FRAME` attr, v4 constant, readable set, gate helper, `padded_crop_offset` |
| `src/phenotypic/_gui/browse/_tile_routes.py` | gate uses the readable set |
| `src/phenotypic/_core/_image_parts/_image_io_handler.py` | writer padding, `pad_on_save`, reader window + frame restore |
| accessor `imsave` ×3 | `pad_on_save` |
| `src/phenotypic/_cli/_cli_process_only.py` | canvas level count |
| `src/phenotypic/_cli/_cli_failure_tracker.py`, `_cli_process_single.py` | revision bump; crop work-id key |
| `src/phenotypic/_gui/_shared/tiles.py` | crops windowed in ROI coordinates against padded stores; display range from the ROI window |
| `src/phenotypic/_gui/results_viewer/_store_source.py`, `colony_view/_grid.py` | `cropOffset` in the Viv source spec; Colony cell centroids shifted |
| `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py` | builder previews pass `pad_on_save=False` |
| `src/phenotypic/_cli/_cli_readme_generator.py` | `FRAME` table in the deliverables README |
| docs (Task 16) | as listed there |

---

## Phase 1: In-memory frame model

### Task 1: `CropFrame` value type and pure helpers

**Files:**
- Create: `src/phenotypic/sdk_/_crop_frame.py` (`CropFrame` + attribute codec; light)
- Create: `src/phenotypic/_core/_crop_frame.py` (pixel helpers; re-exports the `sdk_` names)
- Test: `tests/unit/core/test_crop_frame_helpers.py`

**Interfaces:**
- Produces:
  - `CropFrame(canvas_shape: tuple[int,int], offset: tuple[int,int])`, frozen, with `.fits(shape2d) -> bool`, `.shifted(d_row: int, d_col: int) -> CropFrame`, `.window(shape2d) -> tuple[slice, slice]`
  - `pad_to_canvas(arr: np.ndarray, frame: CropFrame, *, row_axis: int = 0) -> np.ndarray`
  - `rect_origin_from_key(key: Any, shape2d: tuple[int,int]) -> tuple[int,int] | None`
  - `crop_frame_to_attribute(frame: CropFrame, roi_shape: tuple[int,int], padded: bool) -> dict`
  - `crop_frame_from_attribute(value: Any) -> tuple[CropFrame, tuple[int,int], bool] | None`
  - (`journal_records_crop` and `append_frame_offsets` are added to this module in Task 7, once `FRAME` exists)

- [ ] **Step 1: Write the failing tests**

```python
"""Pure helpers behind the crop frame (spec 2026-10-05-pseudo-cropping §3)."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic._core._crop_frame import (
    CropFrame,
    crop_frame_from_attribute,
    crop_frame_to_attribute,
    pad_to_canvas,
    rect_origin_from_key,
)


def test_fits_checks_both_edges():
    frame = CropFrame((40, 60), (5, 10))
    assert frame.fits((35, 50))
    assert not frame.fits((36, 50))
    assert not frame.fits((35, 51))
    assert not CropFrame((40, 60), (-1, 0)).fits((1, 1))


def test_shifted_and_window():
    frame = CropFrame((40, 60), (5, 10)).shifted(2, 3)
    assert frame == CropFrame((40, 60), (7, 13))
    assert frame.window((4, 5)) == (slice(7, 11), slice(13, 18))


def test_values_normalise_to_python_ints():
    frame = CropFrame((np.int64(4), np.int64(5)), (np.int32(1), np.int32(2)))
    assert frame == CropFrame((4, 5), (1, 2))
    assert all(type(v) is int for v in (*frame.canvas_shape, *frame.offset))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_pad_to_canvas_places_roi_and_zeros_elsewhere(dtype):
    roi = (np.arange(12).reshape(3, 4) + 1).astype(dtype)
    frame = CropFrame((6, 9), (2, 3))
    out = pad_to_canvas(roi, frame)
    assert out.shape == (6, 9) and out.dtype == dtype
    np.testing.assert_array_equal(out[2:5, 3:7], roi)
    out[2:5, 3:7] = 0
    assert not out.any()


def test_pad_to_canvas_channel_first_and_channel_last():
    roi_last = np.ones((3, 4, 3), dtype=np.uint8)
    roi_first = np.ones((3, 3, 4), dtype=np.uint8)
    frame = CropFrame((6, 9), (1, 2))
    assert pad_to_canvas(roi_last, frame).shape == (6, 9, 3)
    assert pad_to_canvas(roi_first, frame, row_axis=1).shape == (3, 6, 9)
    assert pad_to_canvas(roi_first, frame, row_axis=1)[:, 1:4, 2:6].all()


def test_pad_to_canvas_refuses_a_roi_that_does_not_fit():
    with pytest.raises(ValueError, match="does not fit"):
        pad_to_canvas(np.ones((5, 5)), CropFrame((6, 6), (2, 2)))


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (slice(5, 20), (5, 0)),
        ((slice(5, 20), slice(10, 50)), (5, 10)),
        ((slice(-10, None), slice(-20, None)), (30, 40)),
        ((slice(None), slice(3, 9)), (0, 3)),
        ((slice(5, 20, 1), slice(None, None, None)), (5, 0)),
    ],
)
def test_rect_origin_for_rectangular_keys(key, expected):
    assert rect_origin_from_key(key, (40, 60)) == expected


@pytest.mark.parametrize(
    "key",
    [
        5,
        (5, slice(None)),
        slice(None, None, 2),
        (slice(0, 10), slice(0, 10), 0),
        Ellipsis,
        np.array([1, 2]),
        (slice(10, 10), slice(None)),
    ],
)
def test_rect_origin_rejects_everything_else(key):
    assert rect_origin_from_key(key, (40, 60)) is None


def test_attribute_round_trip():
    frame = CropFrame((40, 60), (5, 10))
    value = crop_frame_to_attribute(frame, (30, 40), True)
    assert value == {
        "canvas_shape": [40, 60],
        "offset": [5, 10],
        "roi_shape": [30, 40],
        "padded": True,
    }
    assert crop_frame_from_attribute(value) == (frame, (30, 40), True)
    assert crop_frame_from_attribute(None) is None


@pytest.mark.parametrize(
    "value",
    [
        {"canvas_shape": [40, 60], "offset": [5, 10], "padded": True},
        {"canvas_shape": [40], "offset": [5, 10], "roi_shape": [1, 1], "padded": True},
        {"canvas_shape": [40, 60], "offset": [35, 10], "roi_shape": [10, 10], "padded": True},
        ["not", "a", "mapping"],
    ],
)
def test_malformed_attribute_raises(value):
    with pytest.raises(ValueError, match="crop_frame"):
        crop_frame_from_attribute(value)
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core/test_crop_frame_helpers.py -q --capture=fd -p no:cacheprovider`
Expected: collection error, `ModuleNotFoundError: No module named 'phenotypic._core._crop_frame'`.

- [ ] **Step 3: Write the implementation (two modules)**

`CropFrame` and its store-attribute codec live in **`sdk_`**, because `ngff_` (an `sdk_` module) needs them on light paths: GUI chunk routes, `valid_staged_store`, and the run-state probes. Importing anything under `phenotypic._core` runs `phenotypic/_core/__init__.py`, which loads the whole image stack. The pixel helpers live in `_core` and re-export the `sdk_` names, so every `_core` caller and test imports from one place. (Round-2 review R2-m1.)

`src/phenotypic/sdk_/_crop_frame.py`:

```python
"""The crop frame value type and its OME-Zarr attribute codec.

Spec: docs/superpowers/specs/2026-10-05-pseudo-cropping/design.md (§3.1, §5.2).

Light on purpose (stdlib only): ``ngff_`` calls this from GUI chunk routes and
store-validity probes that must not load the image stack. The pixel helpers
that build on it are in ``phenotypic._core._crop_frame``, which re-exports
these names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CropFrame:
    """The original canvas an image's pixels were cut from.

    Attributes:
        canvas_shape: ``(H, W)`` of the original image.
        offset: ``(row, col)`` of this image's top-left pixel in the canvas.
    """

    canvas_shape: tuple[int, int]
    offset: tuple[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "canvas_shape",
            (int(self.canvas_shape[0]), int(self.canvas_shape[1])),
        )
        object.__setattr__(
            self, "offset", (int(self.offset[0]), int(self.offset[1]))
        )

    def fits(self, shape2d: tuple[int, int]) -> bool:
        """Whether an image of 2-D *shape2d* lies inside the canvas at offset."""
        row, col = self.offset
        height, width = self.canvas_shape
        return (
            row >= 0
            and col >= 0
            and row + int(shape2d[0]) <= height
            and col + int(shape2d[1]) <= width
        )

    def shifted(self, d_row: int, d_col: int) -> CropFrame:
        """Return the frame with its offset moved by ``(d_row, d_col)``."""
        return CropFrame(
            self.canvas_shape, (self.offset[0] + d_row, self.offset[1] + d_col)
        )

    def window(self, shape2d: tuple[int, int]) -> tuple[slice, slice]:
        """Canvas slices covering an image of 2-D *shape2d* at this offset."""
        row, col = self.offset
        return (
            slice(row, row + int(shape2d[0])),
            slice(col, col + int(shape2d[1])),
        )


def crop_frame_to_attribute(
    frame: CropFrame, roi_shape: tuple[int, int], padded: bool
) -> dict:
    """Serialise a frame into the store's ``attributes.phenotypic.crop_frame``."""
    return {
        "canvas_shape": [int(v) for v in frame.canvas_shape],
        "offset": [int(v) for v in frame.offset],
        "roi_shape": [int(roi_shape[0]), int(roi_shape[1])],
        "padded": bool(padded),
    }


def crop_frame_from_attribute(
    value: Any,
) -> tuple[CropFrame, tuple[int, int], bool] | None:
    """Parse ``crop_frame``; ``None`` when absent.

    Raises:
        ValueError: If the value is present but malformed or self-inconsistent.
            Reading a padded store's canvas as the image would be silently
            wrong, so a broken record is refused rather than ignored.
    """
    if value is None:
        return None
    try:
        if not isinstance(value, Mapping):
            raise TypeError("not a mapping")
        canvas = value["canvas_shape"]
        offset = value["offset"]
        roi = value["roi_shape"]
        if len(canvas) != 2 or len(offset) != 2 or len(roi) != 2:
            raise TypeError("expected three length-2 sequences")
        frame = CropFrame((canvas[0], canvas[1]), (offset[0], offset[1]))
        roi_shape = (int(roi[0]), int(roi[1]))
        padded = bool(value["padded"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed crop_frame attribute {value!r}: {exc}") from exc
    if not frame.fits(roi_shape):
        raise ValueError(
            f"malformed crop_frame attribute {value!r}: roi_shape does not fit"
        )
    return frame, roi_shape, padded
```

`src/phenotypic/_core/_crop_frame.py`:

```python
"""Pixel-side helpers for an image's crop frame.

Spec: docs/superpowers/specs/2026-10-05-pseudo-cropping/design.md (§3, §5).

A cropped :class:`~phenotypic.Image` keeps compact in-memory arrays and records
a :class:`CropFrame`: the original ``(H, W)`` canvas and the ``(row, col)`` of
its own top-left pixel inside it. Writers zero-pad layers back into the canvas;
the reader slices them out again. ``CropFrame`` and its attribute codec are
defined in :mod:`phenotypic.sdk_._crop_frame` (light, for ``ngff_``) and are
re-exported here, so ``_core`` code imports everything from this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from phenotypic.sdk_._crop_frame import (
    CropFrame,
    crop_frame_from_attribute,
    crop_frame_to_attribute,
)

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "CropFrame",
    "crop_frame_from_attribute",
    "crop_frame_to_attribute",
    "pad_to_canvas",
    "rect_origin_from_key",
]


def pad_to_canvas(
    arr: np.ndarray, frame: CropFrame, *, row_axis: int = 0
) -> np.ndarray:
    """Zero-pad *arr* into *frame*'s canvas.

    Args:
        arr: Layer whose spatial axes are ``row_axis`` and ``row_axis + 1``.
        frame: Destination canvas and offset.
        row_axis: Index of the row axis -- 0 for in-memory ``(H, W[, C])``
            layers, 1 for a store's channel-first ``(C, H, W)`` rgb.

    Returns:
        A new array of the canvas's spatial size with *arr* at the offset and
        0 everywhere else, in *arr*'s dtype.

    Raises:
        ValueError: If *arr* does not fit inside the canvas at the offset.
    """
    roi = (arr.shape[row_axis], arr.shape[row_axis + 1])
    if not frame.fits(roi):
        raise ValueError(
            f"layer of spatial shape {roi} does not fit {frame} -- the crop "
            f"frame is stale"
        )
    shape = list(arr.shape)
    shape[row_axis], shape[row_axis + 1] = frame.canvas_shape
    canvas = np.zeros(shape, dtype=arr.dtype)
    index: list[slice] = [slice(None)] * arr.ndim
    index[row_axis], index[row_axis + 1] = frame.window(roi)
    canvas[tuple(index)] = arr
    return canvas


def rect_origin_from_key(
    key: Any, shape2d: tuple[int, int]
) -> tuple[int, int] | None:
    """Return the ``(row, col)`` origin of a unit-step rectangular window.

    Only a slice, or a tuple of one or two slices, with ``step in (None, 1)``
    and a non-empty extent qualifies. Every other key (an int, an ellipsis, any
    third key element -- even a full channel slice ``[:, :, :]``, which
    ``Image.__getitem__`` cannot apply to the 2-D gray layer anyway -- fancy or
    boolean indexing, a stride) returns ``None``, and the subimage it produced
    records no frame (spec §3.2).
    """
    if isinstance(key, slice):
        key = (key,)
    if not isinstance(key, tuple) or not 1 <= len(key) <= 2:
        return None
    if not all(isinstance(part, slice) for part in key):
        return None
    if len(key) == 1:
        key = (key[0], slice(None))
    origin: list[int] = []
    for part, dim in zip(key, shape2d):
        start, stop, step = part.indices(int(dim))
        if step != 1 or stop <= start:
            return None
        origin.append(start)
    return origin[0], origin[1]
```

(`Mapping` and the `pd` type-checking import are used by the Task 7 additions to this module. If ruff flags them as unused at this point, drop them now and re-add them in Task 7.)

Add a one-line guard to the test file, so the light module stays light:

```python
def test_sdk_codec_module_does_not_import_the_image_stack():
    import subprocess
    import sys

    code = (
        "import sys, phenotypic.sdk_._crop_frame; "
        "print(any(m.startswith('phenotypic._core') for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
```

If this guard fails because `phenotypic.sdk_`'s own package `__init__` already loads `_core`, the test documents that `sdk_` is not a light import root. Report it and drop the guard; don't move the codec again.

- [ ] **Step 4: Run the tests and check they pass**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core/test_crop_frame_helpers.py -q --capture=fd -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/sdk_/_crop_frame.py src/phenotypic/_core/_crop_frame.py tests/unit/core/test_crop_frame_helpers.py
git add src/phenotypic/sdk_/_crop_frame.py src/phenotypic/_core/_crop_frame.py tests/unit/core/test_crop_frame_helpers.py
git commit -m "feat(core): CropFrame value type and pad/slice helpers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD"
```

---

### Task 2: Frame state on `ImageDataManager`

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_image_data_manager.py`
- Test: `tests/unit/core/test_crop_frame_state.py`

**Interfaces:**
- Consumes: `CropFrame`, `pad_to_canvas`, `rect_origin_from_key` (Task 1).
- Produces (on every `Image`):
  - class-level defaults `_crop_frame: CropFrame | None = None` and `_pad_on_save: bool = False`. Class-level, so old pickles and old objects without the attributes still read them.
  - `_valid_crop_frame() -> CropFrame | None`. On a stale frame it warns once (`UserWarning`) and clears both attributes.
  - `_save_padding_frame(pad_on_save: bool | None) -> CropFrame | None`. Resolves `None` to `self._pad_on_save` and returns the valid frame only when padding is in effect.
  - `_pad_layer_for_save(arr: np.ndarray, pad_on_save: bool | None) -> np.ndarray` for in-memory `(H, W[, C])` layers.
  - `_assign_child_frame(child, key) -> None`, used by Task 3.

- [ ] **Step 1: Write the failing tests**

```python
"""Frame state on Image: validity guard, carry-through, reshape drop (spec §3.1, §3.4)."""

from __future__ import annotations

import copy
import pickle
import warnings

import numpy as np
import pytest

from phenotypic import Image
from phenotypic._core._crop_frame import CropFrame


def _image(h: int = 40, w: int = 60) -> Image:
    rng = np.random.default_rng(0)
    return Image(rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8))


def test_new_image_has_no_frame():
    img = _image()
    assert img._crop_frame is None
    assert img._pad_on_save is False
    assert img._valid_crop_frame() is None


def test_valid_frame_is_returned_unchanged():
    img = _image()
    frame = CropFrame((80, 90), (10, 20))
    img._crop_frame = frame
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert img._valid_crop_frame() == frame


def test_stale_frame_is_dropped_with_a_warning():
    img = _image()
    img._crop_frame = CropFrame((10, 10), (5, 5))
    img._pad_on_save = True
    with pytest.warns(UserWarning, match="no longer fits"):
        assert img._valid_crop_frame() is None
    assert img._crop_frame is None
    assert img._pad_on_save is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert img._valid_crop_frame() is None  # warns only once


def test_save_padding_frame_resolution():
    img = _image()
    frame = CropFrame((80, 90), (10, 20))
    img._crop_frame = frame
    img._pad_on_save = False
    assert img._save_padding_frame(None) is None
    assert img._save_padding_frame(True) == frame
    img._pad_on_save = True
    assert img._save_padding_frame(None) == frame
    assert img._save_padding_frame(False) is None


def test_pad_layer_for_save():
    img = _image()
    img._crop_frame = CropFrame((50, 70), (3, 4))
    padded = img._pad_layer_for_save(np.asarray(img.rgb[:]), True)
    assert padded.shape == (50, 70, 3)
    np.testing.assert_array_equal(padded[3:43, 4:64], img.rgb[:])
    unpadded = img._pad_layer_for_save(np.asarray(img.gray[:]), False)
    assert unpadded.shape == (40, 60)


def test_frame_survives_pickle_and_copy():
    img = _image()
    img._crop_frame = CropFrame((80, 90), (10, 20))
    img._pad_on_save = True
    for other in (pickle.loads(pickle.dumps(img)), img.copy(), Image(img), copy.deepcopy(img)):
        assert other._crop_frame == img._crop_frame
        assert other._pad_on_save is True


def test_set_image_from_image_carries_frame():
    src = _image()
    src._crop_frame = CropFrame((80, 90), (10, 20))
    src._pad_on_save = True
    dst = _image(10, 10)
    dst.set_image(src)
    assert dst._crop_frame == src._crop_frame and dst._pad_on_save is True


def test_reshaping_set_image_drops_the_frame():
    img = _image()
    img._crop_frame = CropFrame((80, 90), (10, 20))
    img._pad_on_save = True
    img.set_image(np.zeros((12, 12, 3), dtype=np.uint8))
    assert img._crop_frame is None and img._pad_on_save is False


def test_same_shape_writes_keep_the_frame():
    img = _image()
    img._crop_frame = CropFrame((80, 90), (10, 20))
    img._pad_on_save = True
    img.rgb[0:2, 0:2] = 0
    img.set_image(np.zeros((40, 60, 3), dtype=np.uint8))
    img.rotate(15)
    assert img._crop_frame == CropFrame((80, 90), (10, 20))
    assert img._pad_on_save is True


def test_clear_resets_the_frame():
    img = _image()
    img._crop_frame = CropFrame((80, 90), (10, 20))
    img._pad_on_save = True
    img.clear()
    assert img._crop_frame is None and img._pad_on_save is False
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core/test_crop_frame_state.py -q --capture=fd -p no:cacheprovider`
Expected: FAIL with `AttributeError: ... '_crop_frame'` / `'_valid_crop_frame'`.

- [ ] **Step 3: Write the implementation** in `_image_data_manager.py`

3a. Add the import after the existing `from phenotypic.sdk_.constants_ import ...` line:

```python
from phenotypic._core._crop_frame import CropFrame, pad_to_canvas, rect_origin_from_key
```

3b. In `class ImageDataManager`, directly under `_OBJMAP_DTYPE = np.uint16`:

```python
    #: Where this image's pixels sit in the original canvas, or ``None`` when
    #: the image was never cropped (spec 2026-10-05-pseudo-cropping §3.1).
    #: Class-level defaults so images pickled before this attribute existed,
    #: and partially constructed ones, still read a value. Always read it
    #: through :meth:`_valid_crop_frame`, never directly.
    _crop_frame: CropFrame | None = None
    #: Whether writers pad layers back into the canvas when their
    #: ``pad_on_save`` argument is ``None``. Only ``CropImage`` sets it.
    _pad_on_save: bool = False
```

3c. In `clear()`, before `return`:

```python
        self._crop_frame = None
        self._pad_on_save = False
```

3d. In `_set_from_class_instance`, after the `self._original = (...)` assignment and before `return`:

```python
        # After `_set_from_array` above, whose reshape rule may have cleared a
        # frame this image held: the source's frame is the one that describes
        # the pixels just copied.
        self._crop_frame = input_cls._crop_frame
        self._pad_on_save = bool(input_cls._pad_on_save)
```

3e. Replace `_set_from_array` with:

```python
    def _set_from_array(self, arr: np.ndarray) -> None:
        """Initialize all components from an array.

        A write that changes the 2-D shape drops the crop frame (spec §3.4): the
        frame described the old pixels, and an offset that no longer fits is
        worse than none. A same-shape write keeps it.

        Args:
            arr (np.ndarray): Input image array.
        """
        previous = (
            None if self._data.gray is None else tuple(self._data.gray.shape[:2])
        )
        # Guess format from array shape
        format_enum = self._guess_image_format(arr)
        self._allocate_data(shape=arr.shape)

        # Process based on detected format
        match format_enum:
            case IMAGE_MODE.GRAYSCALE | IMAGE_MODE.GRAYSCALE_SINGLE_CHANNEL:
                self._set_from_matrix(arr if arr.ndim == 2 else arr[:, :, 0])

            case IMAGE_MODE.RGB | IMAGE_MODE.RGB_OR_BGR:
                self._set_from_rgb(arr)

            case IMAGE_MODE.LINEAR_RGB:
                self._set_from_rgb(arr)

            case IMAGE_MODE.RGBA | IMAGE_MODE.RGBA_OR_BGRA:
                self._set_from_rgb(rgba2rgb(arr))

            case _:
                raise ValueError(f"Unsupported image format: {format_enum}")

        if (
            self._crop_frame is not None
            and tuple(self._data.gray.shape[:2]) != previous
        ):
            self._crop_frame = None
            self._pad_on_save = False
        return
```

3f. Add these methods after `_retain_original`:

```python
    def _valid_crop_frame(self) -> CropFrame | None:
        """Return the crop frame if it still fits this image, else drop it.

        The single read path for the frame (spec §3.1). A frame that no longer
        fits (a custom op reshaped ``_data`` behind it) is cleared, together
        with ``_pad_on_save``, after one ``UserWarning``. A wrong offset written
        to a measurement table or a store is worse than no offset.
        """
        frame = self._crop_frame
        if frame is None:
            return None
        shape2d = tuple(self._data.gray.shape[:2])
        if frame.fits(shape2d):
            return frame
        warnings.warn(
            f"Discarding crop frame {frame} of image "
            f"{self._metadata.protected.get(IMAGE.IMAGE_NAME)!r}: it no longer "
            f"fits the image shape {shape2d}. Original-frame offsets are "
            f"unavailable for this image and it will save at its own size.",
            UserWarning,
            stacklevel=2,
        )
        self._crop_frame = None
        self._pad_on_save = False
        return None

    def _save_padding_frame(self, pad_on_save: bool | None) -> CropFrame | None:
        """Frame to pad into for one save, or ``None`` to save unpadded.

        Args:
            pad_on_save: The writer's argument. ``None`` defers to this image's
                ``_pad_on_save``; ``True``/``False`` override it for this call.
        """
        effective = self._pad_on_save if pad_on_save is None else bool(pad_on_save)
        return self._valid_crop_frame() if effective else None

    def _pad_layer_for_save(
        self, arr: np.ndarray, pad_on_save: bool | None
    ) -> np.ndarray:
        """Pad one in-memory ``(H, W[, C])`` layer for a save, if padding applies."""
        frame = self._save_padding_frame(pad_on_save)
        return arr if frame is None else pad_to_canvas(arr, frame, row_axis=0)

    def _assign_child_frame(self, child: Any, key: Any) -> None:
        """Give a subimage cut by *key* its composed frame (spec §3.2).

        A unit-step rectangle composes onto this image's frame (or onto this
        image's own shape when it has none). Any other key leaves the child
        with no frame. Plain slicing never turns padding on.
        """
        child._pad_on_save = False
        shape2d = tuple(self._data.gray.shape[:2])
        origin = rect_origin_from_key(key, shape2d)
        if origin is None:
            child._crop_frame = None
            return
        parent = self._valid_crop_frame()
        base = parent if parent is not None else CropFrame(shape2d, (0, 0))
        child._crop_frame = base.shifted(*origin)
```

- [ ] **Step 4: Run the tests and check they pass**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core/test_crop_frame_state.py tests/unit/core/test_image.py tests/unit/core/test_image_pickle.py -q --capture=fd -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/_core/_image_parts/_image_data_manager.py tests/unit/core/test_crop_frame_state.py
git add src/phenotypic/_core/_image_parts/_image_data_manager.py tests/unit/core/test_crop_frame_state.py
git commit -m "feat(core): crop-frame state, validity guard and carry-through on Image

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD"
```

---

### Task 3: Slicing composes the frame (`Image` and `GridImage`)

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_image_handler.py:88-115` (`__getitem__`)
- Modify: `src/phenotypic/_core/_image_parts/_grid_image_handler.py:258-273` (`__getitem__`)
- Test: `tests/unit/core/test_crop_frame_slicing.py`

**Interfaces:**
- Consumes: `ImageDataManager._assign_child_frame(child, key)` (Task 2).
- Produces: every `image[key]` / `grid_image[key]` / `image.objects[i]` carries a composed frame.

- [ ] **Step 1: Write the failing tests**

```python
"""Rectangular slicing composes the crop frame (spec §3.2)."""

from __future__ import annotations

import numpy as np

from phenotypic import GridImage, Image
from phenotypic._core._crop_frame import CropFrame
from phenotypic.data import load_synth_yeast_plate


def _image() -> Image:
    arr = np.arange(40 * 60 * 3, dtype=np.uint32).reshape(40, 60, 3) % 251
    return Image(arr.astype(np.uint8))


def test_plain_slice_records_offset_into_own_shape():
    sub = _image()[5:25, 10:50]
    assert sub._crop_frame == CropFrame((40, 60), (5, 10))
    assert sub._pad_on_save is False


def test_nested_slices_compose_onto_the_root_canvas():
    img = _image()
    sub = img[5:25, 10:50][2:10, 3:20]
    assert sub._crop_frame == CropFrame((40, 60), (7, 13))
    np.testing.assert_array_equal(sub.rgb[:], img.rgb[7:15, 13:30])


def test_row_only_and_negative_slices():
    img = _image()
    assert img[5:20]._crop_frame == CropFrame((40, 60), (5, 0))
    assert img[-10:, -20:]._crop_frame == CropFrame((40, 60), (30, 40))


def test_non_rectangular_keys_record_no_frame():
    img = _image()
    assert img[::2]._crop_frame is None


def test_grid_image_slice_composes():
    grid = GridImage(load_synth_yeast_plate(), nrows=8, ncols=12)
    sub = grid[10:110, 20:220]
    assert sub._crop_frame == CropFrame(tuple(grid.shape[:2]), (10, 20))


def test_object_crops_carry_plate_offset_plus_bbox():
    plate = Image(load_synth_yeast_plate())
    sub = plate[7:-11, 13:-17]
    first = sub.objects[0]
    min_rr, min_cc = sub.objects.props[0].bbox[:2]
    assert first._crop_frame == CropFrame(tuple(plate.shape[:2]), (7 + min_rr, 13 + min_cc))
    assert first._pad_on_save is False
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core/test_crop_frame_slicing.py -q --capture=fd -p no:cacheprovider`
Expected: FAIL, `assert None == CropFrame(...)`.

- [ ] **Step 3: Write the implementation**

In `ImageHandler.__getitem__` (`_image_handler.py`), insert before `return subimage`:

```python
        self._assign_child_frame(subimage, key)
```

and add to its docstring `Note:` list:

```
            - A unit-step rectangular *key* records where the subimage sits in
              this image's original canvas (``_crop_frame``); other keys record
              none. Slicing never sets ``_pad_on_save``.
```

In `GridImageHandler.__getitem__` (`_grid_image_handler.py`), insert before `return subimage`:

```python
        self._assign_child_frame(subimage, key)
```

- [ ] **Step 4: Run the tests and check they pass**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core/test_crop_frame_slicing.py tests/unit/core/test_image.py tests/unit/grid -q --capture=fd -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/_core/_image_parts/_image_handler.py src/phenotypic/_core/_image_parts/_grid_image_handler.py tests/unit/core/test_crop_frame_slicing.py
git add src/phenotypic/_core/_image_parts/_image_handler.py src/phenotypic/_core/_image_parts/_grid_image_handler.py tests/unit/core/test_crop_frame_slicing.py
git commit -m "feat(core): rectangular slicing composes the crop frame

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD"
```

---

### Task 4: `CropImage` turns padding on; `PadImage` is frame-aware

**Files:**
- Modify: `src/phenotypic/correction/_image_cropper.py` (`_operate`, class docstring)
- Modify: `src/phenotypic/correction/_image_padder.py` (`_operate`, class docstring)
- Test: `tests/unit/correction/test_crop_pad_frames.py`

**Interfaces:**
- Consumes: `_valid_crop_frame`, `_crop_frame`, `_pad_on_save` (Task 2); slicing composition (Task 3).
- Produces: after `CropImage.apply`, `result._pad_on_save is True` and the frame is composed. `PadImage` shifts the frame or drops it with a `UserWarning`.

- [ ] **Step 1: Write the failing tests**

```python
"""CropImage records a padding frame; PadImage shifts or drops it (spec §3.3-3.4)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from phenotypic import GridImage, Image
from phenotypic._core._crop_frame import CropFrame
from phenotypic.correction import CropImage, PadImage
from phenotypic.data import load_synth_yeast_plate


def _image() -> Image:
    return Image(np.random.default_rng(1).integers(0, 255, (40, 60, 3), dtype=np.uint8))


def test_crop_records_frame_and_turns_padding_on():
    out = CropImage(top=3, bottom=5, left=4, right=6).apply(_image())
    assert out.shape[:2] == (32, 50)
    assert out._crop_frame == CropFrame((40, 60), (3, 4))
    assert out._pad_on_save is True


def test_noop_crop_records_identity_frame():
    out = CropImage().apply(_image())
    assert out._crop_frame == CropFrame((40, 60), (0, 0))
    assert out._pad_on_save is True


def test_crop_then_crop_sums_offsets():
    out = CropImage(top=2, left=1).apply(CropImage(top=3, left=4).apply(_image()))
    assert out._crop_frame == CropFrame((40, 60), (5, 5))


def test_grid_crop_keeps_frame_and_grid():
    grid = GridImage(load_synth_yeast_plate(), nrows=8, ncols=12)
    out = CropImage(top=10, left=20).apply(grid)
    assert isinstance(out, GridImage) and out.nrows == 8
    assert out._crop_frame == CropFrame(tuple(grid.shape[:2]), (10, 20))
    assert out._pad_on_save is True


def test_pad_inside_the_canvas_shifts_the_frame():
    cropped = CropImage(top=5, bottom=5, left=5, right=5).apply(_image())
    out = PadImage(top=2, left=3, bottom=1, right=1).apply(cropped)
    assert out.shape[:2] == (33, 54)
    assert out._crop_frame == CropFrame((40, 60), (3, 2))
    assert out._pad_on_save is True


def test_pad_beyond_the_canvas_drops_the_frame_with_a_warning():
    cropped = CropImage(top=2).apply(_image())
    with pytest.warns(UserWarning, match="beyond its original frame"):
        out = PadImage(top=5).apply(cropped)
    assert out._crop_frame is None and out._pad_on_save is False


def test_pad_without_a_frame_is_unchanged_and_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = PadImage(top=5).apply(_image())
    assert out._crop_frame is None and out.shape[:2] == (45, 60)


def test_crop_input_is_untouched():
    img = _image()
    CropImage(top=3).apply(img)
    assert img._crop_frame is None and img.shape[:2] == (40, 60)
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/correction/test_crop_pad_frames.py -q --capture=fd -p no:cacheprovider`
Expected: FAIL. `_pad_on_save` is False after a crop, and the pad tests fail.

- [ ] **Step 3: Write the implementation**

`_image_cropper.py`, in `_operate`, after `result.name = original_name`:

```python
        # The slice above composed where this ROI sits in the original canvas
        # (spec 2026-10-05-pseudo-cropping §3.3); a crop is the one operation
        # whose output saves padded back into that canvas by default.
        result._pad_on_save = True
```

Class docstring: add a paragraph after the first paragraph:

```
    The removed margins are not forgotten: the result records where it sits in
    the original image (its crop frame), every ``info()``/measurement table
    carries ``Frame_OffsetRR``/``Frame_OffsetCC`` so original-frame coordinates
    are ``Bbox_* + Frame_Offset*``, and ``save2zarr`` writes every layer
    zero-padded back to the original size by default
    (``save2zarr(..., pad_on_save=False)`` writes the ROI only).
```

`_image_padder.py`: add `import warnings` to the imports. In `_operate`, insert at the very top of the body:

```python
        frame = image._valid_crop_frame()
        frame_pads_on_save = image._pad_on_save
```

and directly before `# Handle GridImage type preservation`:

```python
        # Frame-aware (spec 2026-10-05-pseudo-cropping §3.4): `_data` was padded
        # directly above, so nothing else updates the frame. Shift it by the
        # top/left margins; keep it only if the padded image still lies inside
        # the original canvas.
        image._crop_frame = None
        image._pad_on_save = False
        if frame is not None:
            shifted = frame.shifted(-(self.top or 0), -(self.left or 0))
            if shifted.fits(image._data.gray.shape[:2]):
                image._crop_frame = shifted
                image._pad_on_save = frame_pads_on_save
            else:
                warnings.warn(
                    f"PadImage extends {image.name!r} beyond its original frame "
                    f"{frame.canvas_shape}; the crop frame is discarded and the "
                    f"image will save at its padded size.",
                    UserWarning,
                    stacklevel=2,
                )
```

Class docstring: add under `Consider Also` or as a final paragraph:

```
    On a cropped image the padding is tracked against the original frame:
    padding that stays inside the cropped-away margin keeps the frame (shifted),
    while padding past the original edge discards it with a warning. When the
    frame is kept, the pad pixels -- whatever ``mode``/``constant_value``
    produced -- are part of the analysed image and are saved at their true
    positions in a padded store; only never-analysed area is zero.
```

- [ ] **Step 4: Run the tests and check they pass**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/correction/test_crop_pad_frames.py tests/unit/correction/test_image_cropper.py tests/unit/correction/test_image_padder.py -q --capture=fd -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/correction/_image_cropper.py src/phenotypic/correction/_image_padder.py tests/unit/correction/test_crop_pad_frames.py
git add src/phenotypic/correction/_image_cropper.py src/phenotypic/correction/_image_padder.py tests/unit/correction/test_crop_pad_frames.py
git commit -m "feat(correction): CropImage records a padding frame; PadImage shifts or drops it

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD"
```

---

### Task 5: Frame carry-over through the apply wrapper

**Files:**
- Modify: `src/phenotypic/_core/_provenance.py:484-523` (`_carry_logical_image_state`) and its single call site (`~:684`) plus the capture block (`~:645`)
- Test: `tests/unit/core/test_crop_frame_provenance_carry.py`

**Interfaces:**
- Consumes: `_valid_crop_frame`, `_pad_on_save` (Task 2).
- Produces: `_carry_logical_image_state(result, source_journal, source_original, nested_operations, *, source_frame=None, source_pad_on_save=False, source_shape=None)`.

- [ ] **Step 1: Write the failing test**

```python
"""A same-shape op that rebuilds its result keeps the input's frame (spec §3.5)."""

from __future__ import annotations

import numpy as np

from phenotypic import Image
from phenotypic.abc_ import ImageCorrector
from phenotypic.correction import CropImage


class _RebuildFresh(ImageCorrector):
    """Returns a brand-new Image with identical pixels (no frame of its own)."""

    def _operate(self, image: Image) -> Image:
        return Image(np.asarray(image.rgb[:]).copy())


class _HalveWidth(ImageCorrector):
    """Returns a brand-new, narrower Image."""

    def _operate(self, image: Image) -> Image:
        return Image(np.asarray(image.rgb[:, : image.shape[1] // 2]).copy())


def _cropped() -> Image:
    rng = np.random.default_rng(2)
    return CropImage(top=3, left=4).apply(Image(rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)))


def test_same_shape_rebuild_inherits_the_frame():
    cropped = _cropped()
    out = _RebuildFresh().apply(cropped)
    assert out._crop_frame == cropped._crop_frame
    assert out._pad_on_save is True


def test_shape_changing_rebuild_does_not_inherit():
    out = _HalveWidth().apply(_cropped())
    assert out._crop_frame is None and out._pad_on_save is False


def test_a_result_with_its_own_frame_is_left_alone():
    cropped = _cropped()
    out = CropImage(top=1).apply(cropped)
    assert out._crop_frame.offset == (4, 4)


def test_capture_does_not_mutate_the_callers_stale_input():
    from phenotypic._core._crop_frame import CropFrame

    img = Image(np.zeros((40, 60, 3), dtype=np.uint8))
    img._crop_frame = CropFrame((5, 5), (0, 0))  # stale: does not fit 40x60
    img._pad_on_save = True
    out = _RebuildFresh().apply(img)
    assert img._crop_frame == CropFrame((5, 5), (0, 0))
    assert img._pad_on_save is True
    assert out._crop_frame is None
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core/test_crop_frame_provenance_carry.py -q --capture=fd -p no:cacheprovider`
Expected: `test_same_shape_rebuild_inherits_the_frame` FAILS (`None == CropFrame(...)`). The other three pass already; that's fine, they are regression pins. If `_RebuildFresh().apply` fails *before* the assertion (an `ImageCorrector` integrity check rejecting a fresh object), switch the base to `phenotypic.abc_.ImageOperation` and record the substitution in the commit message.

- [ ] **Step 3: Write the implementation**

In the apply wrapper, next to `source_original = logical_input._original` (`~:645`), add:

```python
        # Spec 2026-10-05-pseudo-cropping §3.5: captured before the op runs, so
        # an in-place op cannot overwrite what is being carried. Read WITHOUT
        # `_valid_crop_frame()`: that guard clears a stale frame, and with
        # inplace=False this is the caller's object, which must stay untouched.
        source_shape = tuple(logical_input._data.gray.shape[:2])
        source_frame = logical_input._crop_frame
        if source_frame is not None and not source_frame.fits(source_shape):
            source_frame = None
        source_pad_on_save = bool(logical_input._pad_on_save)
```

Change the call (`~:684`) to:

```python
            _carry_logical_image_state(
                result,
                source_journal,
                source_original,
                frame.nested_records,
                source_frame=source_frame,
                source_pad_on_save=source_pad_on_save,
                source_shape=source_shape,
            )
```

Change the signature and append to the end of `_carry_logical_image_state`:

```python
def _carry_logical_image_state(
    result: "Image",
    source_journal: Mapping[str, Any],
    source_original: Any,
    nested_operations: list[dict[str, Any]],
    *,
    source_frame: Any = None,
    source_pad_on_save: bool = False,
    source_shape: tuple[int, ...] | None = None,
) -> None:
```

```python
    # A same-shape op that rebuilt its result as a fresh Image would otherwise
    # lose the crop frame silently (spec 2026-10-05-pseudo-cropping §3.5). A
    # result that set its own frame (CropImage) or changed shape is left alone.
    if (
        result._crop_frame is None
        and source_frame is not None
        and tuple(result._data.gray.shape[:2]) == source_shape
    ):
        result._crop_frame = source_frame
        result._pad_on_save = source_pad_on_save
```

- [ ] **Step 4: Run the tests and check they pass**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core/test_crop_frame_provenance_carry.py tests/unit/core/test_provenance_v2.py tests/unit/core/test_provenance_step_descent.py tests/unit/core/test_provenance_checkpoint.py -q --capture=fd -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Phase 1 affected surface (run once)**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core tests/unit/correction tests/unit/grid tests/unit/abc_ tests/unit/prefab tests/smoke -q --capture=fd -p no:cacheprovider`
(`tests/unit/prefab` covers `prefab/_spimager_pipeline.py:18`, the shipped `CropImage(left=650, right=650, top=600, bottom=600)` user.)
Expected: no failures beyond those already failing on `main`. Run any failing test on its own before you attribute it to this change.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/_core/_provenance.py tests/unit/core/test_crop_frame_provenance_carry.py
git add src/phenotypic/_core/_provenance.py tests/unit/core/test_crop_frame_provenance_carry.py
git commit -m "feat(core): carry the crop frame through same-shape rebuilding ops

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD"
```

---

## Phase 2: Offsets in `info()` and the schema

### Task 6: `FRAME` schema enum and column ordering

**Files:**
- Create: `src/phenotypic/schema/_frame.py`
- Modify: `src/phenotypic/schema/__init__.py` (import after `from ._bbox import BBOX`; `"FRAME"` in `__all__` after `"BBOX"`; mention in the module docstring's example list if it enumerates enums)
- Modify: `src/phenotypic/sdk_/_metadata_helpers.py:140` (info-block detection)
- Modify: `tests/unit/schema/test_classification.py:173-179` (identity list)
- Test: `tests/unit/schema/test_frame_schema.py`

**Interfaces:**
- Produces: `phenotypic.schema.FRAME` with `FRAME.OFFSET_RR == "Frame_OffsetRR"` and `FRAME.OFFSET_CC == "Frame_OffsetCC"`, kind `identity`.

- [ ] **Step 1: Write the failing tests**

```python
"""FRAME offsets are public identity columns that trail in the info block."""

from __future__ import annotations

from phenotypic.schema import FRAME
from phenotypic.sdk_ import order_measurement_columns


def test_headers():
    assert str(FRAME.OFFSET_RR) == "Frame_OffsetRR"
    assert str(FRAME.OFFSET_CC) == "Frame_OffsetCC"
    assert FRAME.get_headers() == ["Frame_OffsetRR", "Frame_OffsetCC"]


def test_identity_kind_and_no_bio_desc():
    for member in FRAME:
        assert member.resolved_kind == "identity"
        assert member.bio_desc == ""


def test_frame_columns_sort_into_the_info_block():
    cols = ["Frame_OffsetRR", "Size_Area", "Bbox_CenterRR", "Object_Label", "Frame_OffsetCC"]
    ordered = order_measurement_columns(cols)
    assert ordered[0] == "Size_Area"
    assert ordered[1] == "Object_Label"
    assert set(ordered[1:]) == {"Object_Label", "Bbox_CenterRR", "Frame_OffsetRR", "Frame_OffsetCC"}
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/schema/test_frame_schema.py -q --capture=fd -p no:cacheprovider`
Expected: `ImportError: cannot import name 'FRAME'`.

- [ ] **Step 3: Write the implementation**

`src/phenotypic/schema/_frame.py`:

```python
"""Original-frame offsets of an image analysed inside a crop."""

from ._measurement_info import Entry
from ._tiers import IdentityInfo


class FRAME(IdentityInfo):
    """Locate the analysed region inside the original image frame.

    ``CropImage`` keeps measuring in the cropped region's own pixel
    coordinates, so every ``Bbox_*``/``Grid_*`` value is relative to the
    region's top-left corner. These two columns record where that corner sat in
    the original image, so original-frame coordinates can be regenerated
    without changing any existing value. They are 0 when no crop is recorded
    and null when a crop is recorded but its offset is not (e.g. a store
    written before crop frames existed), so every table carries the columns.
    """

    @classmethod
    def category(cls) -> str:
        return "Frame"

    OFFSET_RR = Entry(
        "OffsetRR",
        "Row offset of the analysed region's top-left pixel within the original "
        "image frame; 0 when no crop is recorded, null when a crop is recorded "
        "but its offset is not. Original-frame row = Bbox_*RR + Frame_OffsetRR.",
    )
    OFFSET_CC = Entry(
        "OffsetCC",
        "Column offset of the analysed region's top-left pixel within the "
        "original image frame; 0 when no crop is recorded, null when a crop is "
        "recorded but its offset is not. Original-frame column = "
        "Bbox_*CC + Frame_OffsetCC.",
    )
```

`schema/__init__.py`: `from ._frame import FRAME` after `from ._bbox import BBOX`, and `"FRAME",` after `"BBOX",` in `__all__`.

`_metadata_helpers.py:140`, change to:

```python
        elif c == label or c.startswith(("Bbox_", "Grid_", "Frame_")):
```

and update the docstring sentence `The per-object info block (``Object_Label`` + ``Bbox_*`` / ``Grid_*``)` to read `(``Object_Label`` + ``Bbox_*`` / ``Grid_*`` / ``Frame_*``)`.

`tests/unit/schema/test_classification.py` (`test_identity_enums_resolve_identity`): add `FRAME` to the import and to the tuple.

- [ ] **Step 4: Run the tests and check they pass**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/schema tests/unit/core/test_metadata_cluster_order.py -q --capture=fd -p no:cacheprovider`
Expected: all PASS. `test_rembi_coverage.py::test_resolved_module_is_total` must pass for `FRAME` too. If a schema test pins the number of exported enums, increment it.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/schema/_frame.py src/phenotypic/schema/__init__.py src/phenotypic/sdk_/_metadata_helpers.py tests/unit/schema/test_frame_schema.py tests/unit/schema/test_classification.py
git add src/phenotypic/schema/_frame.py src/phenotypic/schema/__init__.py src/phenotypic/sdk_/_metadata_helpers.py tests/unit/schema/test_frame_schema.py tests/unit/schema/test_classification.py
git commit -m "feat(schema): FRAME identity enum (Frame_OffsetRR/CC) in the info block

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD"
```

---

### Task 7: Offsets on both `info()` paths (with the null rule), Bbox golden, real merge site, README

**Files:**
- Modify: `src/phenotypic/_core/_crop_frame.py` (add `journal_records_crop`, `append_frame_offsets`)
- Modify: `src/phenotypic/_core/_image_parts/_image_data_manager.py` (add `_frame_offsets_for_info`, `_unknown_frame_warned`)
- Modify: `src/phenotypic/_core/_image_parts/accessors/_objects_accessor.py:~708` (`info`)
- Modify: `src/phenotypic/_core/_image_parts/accessors/_grid_accessor.py:~170` (`info`)
- Modify: `src/phenotypic/_cli/_cli_readme_generator.py:~175` (`_generate_measurements_section`)
- Test: `tests/unit/core/test_crop_frame_info.py`, `tests/unit/cli/test_readme_frame_section.py`

**Interfaces:**
- Consumes: `FRAME` (Task 6); `_valid_crop_frame` (Task 2). It also relies on the journal layout (`_provenance.py:302-312`): v1 is a flat `operations` list; v2 has `applications[*].operations`. Each entry carries `operation_class` (e.g. `"phenotypic.correction._image_cropper.CropImage"`) and `operation_name`. It deliberately does **not** call `readonly_operations`, which validates and deep-freezes the whole journal on every call.
- `_image_data_manager.py` needs a module logger if it has none: `logger = logging.getLogger(__name__)` (add `import logging`).
- Produces:
  - `journal_records_crop(journal: Mapping) -> bool`
  - `append_frame_offsets(info: pd.DataFrame, offsets: tuple[int, int] | None) -> pd.DataFrame`. `None` gives null `Int64` columns; otherwise `int64`.
  - `ImageDataManager._frame_offsets_for_info() -> tuple[int, int] | None`, implementing spec §4.2:
    - a valid frame gives its offset;
    - no frame and no crop in the journal gives `(0, 0)`;
    - no frame but a crop in the journal gives `None`, plus one `UserWarning` per image instance.

- [ ] **Step 1: Write the failing tests** (`tests/unit/core/test_crop_frame_info.py`)

```python
"""Both info() paths carry Frame_* offsets; Bbox values are unchanged (spec §4)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from phenotypic import GridImage, Image, ImagePipeline
from phenotypic.correction import CropImage
from phenotypic.data import load_synth_yeast_plate
from phenotypic.measure import MeasureSize
from phenotypic.refine import GridOversizedObjectRemover, KeepNearestCenter, KeepSectionLargest
from phenotypic.refine._merge_within_section import MergeWithinSection  # not exported
from phenotypic.schema import BBOX, FRAME

OFFSETS = [str(FRAME.OFFSET_RR), str(FRAME.OFFSET_CC)]
CROP = dict(top=7, bottom=11, left=13, right=17)
_SUFFIXES = ("_x", "_y", "_merged")


def _bbox_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("Bbox_")]


def _cropped_grid() -> GridImage:
    return CropImage(**CROP).apply(GridImage(load_synth_yeast_plate(), nrows=8, ncols=12))


def test_uncropped_info_carries_zero_offsets():
    plate = Image(load_synth_yeast_plate())  # loaded OUTSIDE the error filter
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        info = plate.objects.info(include_metadata=False)
    assert (info[OFFSETS] == 0).all().all()
    assert all(info[c].dtype == np.int64 for c in OFFSETS)


def test_cropped_object_info_carries_offsets_and_unchanged_bbox():
    cropped = CropImage(**CROP).apply(Image(load_synth_yeast_plate()))
    info = cropped.objects.info(include_metadata=False)
    assert set(info[str(FRAME.OFFSET_RR)]) == {7}
    assert set(info[str(FRAME.OFFSET_CC)]) == {13}
    # Golden: the Bbox block is exactly what a plain ROI image measures.
    reference = Image(np.asarray(cropped.rgb[:]))
    reference.objmap[:] = cropped.objmap[:]
    expected = reference.objects.info(include_metadata=False)
    pd.testing.assert_frame_equal(info[_bbox_cols(info)], expected[_bbox_cols(expected)])


def test_original_coordinates_regenerate():
    plate = Image(load_synth_yeast_plate())
    info = CropImage(**CROP).apply(plate).objects.info(include_metadata=False)
    rr = (info[str(BBOX.CENTER_RR)] + info[str(FRAME.OFFSET_RR)]).round().astype(int)
    cc = (info[str(BBOX.CENTER_CC)] + info[str(FRAME.OFFSET_CC)]).round().astype(int)
    hits = plate.objmap[:][rr.to_numpy(), cc.to_numpy()] == info["Object_Label"].to_numpy()
    assert hits.mean() > 0.9


def test_grid_info_carries_offsets():
    info = _cropped_grid().grid.info(include_metadata=False)
    assert set(info[str(FRAME.OFFSET_RR)]) == {7}
    assert set(info[str(FRAME.OFFSET_CC)]) == {13}


def test_a_cropped_image_whose_frame_was_lost_gets_null_offsets_and_warns_once():
    cropped = CropImage(**CROP).apply(Image(load_synth_yeast_plate()))
    cropped._crop_frame = None  # e.g. dropped by PadImage overflow or the stale guard
    with pytest.warns(UserWarning, match="unknown"):
        info = cropped.objects.info(include_metadata=False)
    assert info[OFFSETS].isna().all().all()
    assert all(str(info[c].dtype) == "Int64" for c in OFFSETS)
    assert not info[_bbox_cols(info)].isna().any().any()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cropped.objects.info(include_metadata=False)  # this instance warned already


def test_unknown_offset_warning_text_is_constant_across_images():
    """Constant text lets Python's warning registry collapse a 6,000-image
    re-measure into one line; the image name goes to the log instead."""
    messages = []
    for name in ("plate_a", "plate_b"):
        cropped = CropImage(**CROP).apply(Image(load_synth_yeast_plate()))
        # Set AFTER construction: Image(other) copies other's protected
        # metadata, name included, so a constructor `name=` would be lost and
        # both images would share one name -- making this test vacuous.
        cropped.name = name
        assert cropped.name == name
        cropped._crop_frame = None
        with pytest.warns(UserWarning, match="unknown") as record:
            cropped.objects.info(include_metadata=False)
        messages.append(str(record[0].message))
    assert messages[0] == messages[1]
    assert "plate_a" not in messages[0]


def test_pipeline_measurements_carry_offsets_once():
    df = ImagePipeline(ops={"crop": CropImage(**CROP)}, meas=[MeasureSize()]).apply_and_measure(
        GridImage(load_synth_yeast_plate(), nrows=8, ncols=12)
    )
    assert set(OFFSETS) <= set(df.columns)
    assert not [c for c in df.columns if c.endswith(_SUFFIXES)]


def test_measure_features_include_meta_merges_info_without_collision():
    """The one real info()-merge site (abc_/_measure_features.py:451-456)."""
    df = MeasureSize().measure(_cropped_grid(), include_meta=True)
    assert set(OFFSETS) <= set(df.columns)
    assert not [c for c in df.columns if c.endswith(_SUFFIXES)]


@pytest.mark.parametrize(
    "op",
    [KeepSectionLargest(), KeepNearestCenter(), GridOversizedObjectRemover(), MergeWithinSection()],
    ids=lambda op: type(op).__name__,
)
def test_info_consuming_refiners_keep_working_on_a_cropped_grid(op):
    """Regression pins. The audit found every refiner/measurer merge site safe
    (none merges two info() frames), so these pass before and after."""
    grid = _cropped_grid()
    out = op.apply(grid)
    assert out._crop_frame == grid._crop_frame


def test_aggregation_of_old_and_new_tables_keeps_frame_nullable(tmp_path, monkeypatch):
    """A non-crop run resumed across the upgrade mixes embedded tables with and
    without Frame_* (spec §4.2). The FORWARD master path is
    aggregate_embedded_measurement_tables: each store's table is projected onto
    its own recorded columns, then concatenated. The projection is stubbed to
    return each store's (already projected) table, so this pins the real
    concatenation, not the legacy external-Parquet reader."""
    from pathlib import Path

    import polars as pl

    from phenotypic._cli import _cli_parquet_agg as agg

    old = tmp_path / "old.ome.zarr" / "tables" / "measurements" / "table.parquet"
    new = tmp_path / "new.ome.zarr" / "tables" / "measurements" / "table.parquet"
    projected = {
        old: pl.DataFrame({"Object_Label": [1], "Size_Area": [10.0]}),
        new: pl.DataFrame(
            {"Object_Label": [1], "Size_Area": [11.0], "Frame_OffsetRR": [0], "Frame_OffsetCC": [0]}
        ),
    }
    monkeypatch.setattr(
        agg, "project_embedded_measurement_table", lambda path, *a, **k: projected[Path(path)]
    )
    df, aggregated = agg.aggregate_embedded_measurement_tables({old: "ds", new: "ds"})
    assert set(aggregated) == {old, new}
    assert {"Frame_OffsetRR", "Frame_OffsetCC"} <= set(df.columns)
    assert df["Frame_OffsetRR"].null_count() == 1
```

(Before writing this test, read `_cli_parquet_agg.py:385-420`. If the projection is called with a different first argument than the table path, or its result is post-processed in a way that needs more columns (e.g. `filename`), adapt the stub to match. Keep it a stub of the projection only, so `aggregate_embedded_measurement_tables`' own concatenation is what's under test.)

`tests/unit/cli/test_readme_frame_section.py`:

```python
"""The deliverables README documents the Frame_* columns every table carries."""

from types import SimpleNamespace

from phenotypic import ImagePipeline
from phenotypic._cli._cli_readme_generator import READMEGenerator
from phenotypic.measure import MeasureSize


def test_measurements_section_documents_frame_offsets():
    # A measurer is required: with none, the section returns early
    # ("No measurements configured") before any identity table is built.
    gen = READMEGenerator(
        config=SimpleNamespace(image_type="Image"), pipeline=ImagePipeline(meas=[MeasureSize()])
    )
    section = gen._generate_measurements_section()
    assert "Frame_OffsetRR" in section and "Frame_OffsetCC" in section
    assert section.index("Bbox_CenterRR") < section.index("Frame_OffsetRR")
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core/test_crop_frame_info.py tests/unit/cli/test_readme_frame_section.py -q --capture=fd -p no:cacheprovider`
Expected:
- The offset tests FAIL with `KeyError: "['Frame_OffsetRR', 'Frame_OffsetCC'] not in index"`.
- The README test FAILS.
- The refiner pins and the aggregation pin PASS. That is expected; they pin existing behaviour. If the aggregation pin fails, stop and report: the spec's aggregation claim would then be wrong.

- [ ] **Step 3: Write the implementation**

Append to `_core/_crop_frame.py`:

```python
#: Operation classes whose presence in a journal means "this image was cropped".
#: ``ImageCropper`` is the retired alias (``sdk_/_class_aliases.py``).
_CROP_OPERATION_NAMES = frozenset({"CropImage", "ImageCropper"})


def _journal_operation_entries(journal: Any) -> list[Any]:
    """Operation entries of a v1 (flat) or v2 (per-application) journal.

    A plain walk, with no validation and no copy: this runs on every ``info()``
    of every uncropped image (``readonly_operations`` would validate and
    deep-freeze the whole journal each time), and it must never raise.
    """
    if not isinstance(journal, Mapping):
        return []
    flat = journal.get("operations")
    if isinstance(flat, list):  # schema v1
        return flat
    entries: list[Any] = []
    applications = journal.get("applications")
    if isinstance(applications, list):
        for application in applications:
            if isinstance(application, Mapping) and isinstance(
                application.get("operations"), list
            ):
                entries.extend(application["operations"])
    return entries


def journal_records_crop(journal: Any) -> bool:
    """Whether a provenance journal records a crop operation.

    Matches the last dotted component of ``operation_class`` (and
    ``operation_name``, for entries that lack a class). Anything malformed
    reads as "no crop" -- this only decides between ``0`` and null offsets.
    """
    for operation in _journal_operation_entries(journal):
        if not isinstance(operation, Mapping):
            continue
        for field in ("operation_class", "operation_name"):
            value = operation.get(field)
            if isinstance(value, str) and value.rsplit(".", 1)[-1] in _CROP_OPERATION_NAMES:
                return True
    return False


def append_frame_offsets(
    info: "pd.DataFrame", offsets: tuple[int, int] | None
) -> "pd.DataFrame":
    """Return *info* with ``Frame_OffsetRR``/``Frame_OffsetCC`` appended.

    Always present (spec §4.2): ``int64`` offsets when known (``0, 0`` for a
    never-cropped image), and null ``Int64`` when *offsets* is ``None`` -- an
    image that was cropped but whose offset was never recorded. Existing
    columns are untouched.
    """
    import pandas as pd

    from phenotypic.schema import FRAME

    if offsets is None:
        values = {
            str(FRAME.OFFSET_RR): pd.array([pd.NA] * len(info), dtype="Int64"),
            str(FRAME.OFFSET_CC): pd.array([pd.NA] * len(info), dtype="Int64"),
        }
    else:
        values = {
            str(FRAME.OFFSET_RR): np.int64(offsets[0]),
            str(FRAME.OFFSET_CC): np.int64(offsets[1]),
        }
    return info.assign(**values)
```

In `ImageDataManager`, add a class attribute next to `_pad_on_save`:

```python
    #: Whether this instance already warned that its crop offset is unknown.
    _unknown_frame_warned: bool = False
```

and a method after `_assign_child_frame`:

```python
    def _frame_offsets_for_info(self) -> tuple[int, int] | None:
        """Offsets for the ``Frame_*`` columns (spec §4.2).

        Returns the valid frame's offset; ``(0, 0)`` when the image has no
        frame and its journal records no crop (never cropped); ``None`` -- null
        columns -- when the journal records a crop but no frame survives
        (a store written before crop frames existed, or a frame dropped by
        ``PadImage`` overflow or the stale-frame guard). ``0`` would claim
        "not cropped", which is false, so the offset is reported as unknown.
        """
        frame = self._valid_crop_frame()
        if frame is not None:
            return frame.offset
        from phenotypic._core._crop_frame import journal_records_crop

        if not journal_records_crop(self._metadata.provenance_journal):
            return (0, 0)
        if not self._unknown_frame_warned:
            # Constant text on purpose: Python's warning registry de-duplicates
            # by message, so a re-measure over thousands of pre-change crop
            # stores prints one line, not one per image. The image name goes
            # to the log.
            logger.info(
                "Crop offset unknown for image %r; Frame_OffsetRR/CC are null.",
                self._metadata.protected.get(IMAGE.IMAGE_NAME),
            )
            warnings.warn(
                "A cropped image's offset into the original image is unknown "
                "(it predates recorded crop frames, or its frame was "
                "discarded); its Frame_OffsetRR/CC are null. Re-processing the "
                "source images with --mode full records them.",
                UserWarning,
                stacklevel=2,
            )
            self._unknown_frame_warned = True
        return None
```

`ObjectsAccessor.info`, after `info = MeasureBounds().measure(self._root_image)`, and `GridAccessor.info`, after `info = self._root_image.grid_finder.measure(self._root_image)`:

```python
        from phenotypic._core._crop_frame import append_frame_offsets

        info = append_frame_offsets(info, self._root_image._frame_offsets_for_info())
```

(`grid_finder.measure` itself calls `objects.info()` (`abc_/_grid_finder.py:419-420`), and `assign` overwrites, so there are no duplicate columns.) Add `- Frame_OffsetRR, Frame_OffsetCC: where the analysed region sits in the original image (0 when uncropped, null when cropped but unrecorded)` to both `Returns:` column lists.

`_cli_readme_generator.py`, in `_generate_measurements_section`, after the `bbox_table` block:

```python
        from phenotypic.schema import FRAME

        frame_table = self._generate_measurement_table(FRAME)
        if frame_table:
            sections.append(frame_table)
```

- [ ] **Step 4: Run the tests and check they pass**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core/test_crop_frame_info.py tests/unit/cli/test_readme_frame_section.py tests/unit/cli/test_readme_model_section.py -q --capture=fd -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Phase 2 affected surface: find column-set pins**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core tests/unit/measure tests/unit/refine tests/unit/schema tests/unit/sdk_ tests/unit/cli/test_cli_output_manager.py tests/unit/cli/test_finalize_run.py tests/unit/analysis/test_error_cutoffs.py tests/unit/gui/results_viewer/test_measurement_source.py tests/unit/gui/run_console/test_request_safety.py tests/smoke -q --capture=fd -p no:cacheprovider`

Expected failures come only from tests that pin an exact info/measurement column list or a golden table. For each one, run it in isolation, confirm the only diff is the two `Frame_*` columns, and update the pinned list or regenerate the golden. **Do not** loosen an assertion to `issubset`. Any other diff is a bug in this task.

- [ ] **Step 6: Lint and commit**

Lint and stage the four source files, the readme generator, both new test files, and each pinned test file updated in Step 5, all by explicit path. Commit message:

```
feat(core): Frame_OffsetRR/CC on both info() paths; null when a crop is unrecorded

Audit (spec §4.3): no refiner or grid measurer merges two info() frames; the
one info() merge is MeasureFeatures.measure(include_meta=True), pinned.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD
```

---

## Phase 3: Persistence

### Task 8: Store attributes, v4 for padded stores, readable-version gate

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py` (`:57` constants, `:464` `PhenotypicAttr`, `:522` `build_phenotypic_attributes`, `:696-703` `require_readable_store`, `:2003-2006` `valid_staged_store`; new `is_readable_store_schema_version`, `padded_crop_offset`)
- Modify: `src/phenotypic/_gui/browse/_tile_routes.py:215-219`
- Modify (existing tests that use 4 as the "newer, unreadable" version, so they break the moment 4 is readable): `tests/unit/core/test_image_zarr_roundtrip.py:308-360`, `tests/unit/sdk_/test_ngff_validity.py:~112-120`
- Test: `tests/unit/sdk_/test_padded_store_version.py`

**Interfaces:**
- Produces:
  - `ngff_.PADDED_STORE_SCHEMA_VERSION: Final[int] = 4`
  - `ngff_.READABLE_STORE_SCHEMA_VERSIONS: Final[frozenset[int]] = frozenset({3, 4})`
  - `ngff_.is_readable_store_schema_version(found: object) -> bool`
  - `ngff_.PhenotypicAttr.CROP_FRAME = "crop_frame"`
  - `build_phenotypic_attributes(..., crop_frame: Mapping[str, Any] | None = None)`
  - `ngff_.padded_crop_window(block: Mapping[str, Any]) -> tuple[tuple[int,int], tuple[int,int], tuple[int,int]] | None`, returning `(canvas_shape, offset, roi_shape)` for a padded store and `None` otherwise (including malformed)
  - `ngff_.padded_crop_offset(block: Mapping[str, Any]) -> tuple[int, int]`
  - `ngff_.check_crop_frame_consistency(block: Mapping[str, Any]) -> None`. Raises `ValueError` naming `crop_frame` when the attribute is malformed or when version 4 ⇔ `padded` is violated (spec §5.2 read invariant).
  - `STORE_SCHEMA_VERSION` keeps its value 3 and now means "written by every unpadded store".

- [ ] **Step 1: Write the failing tests**

```python
"""Padded stores are v4; this build reads {3, 4}; older builds refuse v4 (spec §5.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic.sdk_ import ngff_
from phenotypic.sdk_.ngff_ import PhenotypicAttr

_FRAME = {"canvas_shape": [40, 60], "offset": [5, 10], "roi_shape": [30, 40]}


def _attrs(crop_frame=None) -> dict:
    return ngff_.build_phenotypic_attributes(
        image_class="Image",
        series_names=["gray"],
        pyramid_levels=1,
        metadata_sections={},
        detect_mode=None,
        illuminant=None,
        gamma=None,
        crop_frame=crop_frame,
    )


def _store(tmp_path: Path, version, **block) -> Path:
    store = tmp_path / f"v{version}.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "group",
        "attributes": {"phenotypic": {"store_schema_version": version, **block}},
    }))
    return store


def _padded_v4(tmp_path: Path) -> Path:
    """A v4 root that satisfies the read invariant (v4 <=> padded crop_frame)."""
    return _store(tmp_path, 4, crop_frame={**_FRAME, "padded": True})


def test_no_frame_writes_v3_and_no_key():
    block = _attrs()
    assert block[PhenotypicAttr.STORE_SCHEMA_VERSION] == 3
    assert PhenotypicAttr.CROP_FRAME not in block


def test_unpadded_frame_writes_v3_with_the_key():
    block = _attrs({**_FRAME, "padded": False})
    assert block[PhenotypicAttr.STORE_SCHEMA_VERSION] == 3
    assert block[PhenotypicAttr.CROP_FRAME]["padded"] is False


def test_padded_frame_writes_v4():
    block = _attrs({**_FRAME, "padded": True})
    assert block[PhenotypicAttr.STORE_SCHEMA_VERSION] == ngff_.PADDED_STORE_SCHEMA_VERSION == 4


@pytest.mark.parametrize(("found", "ok"), [(3, True), (4, True), (5, False), (None, False), ("3", False), ([3], False)])
def test_is_readable(found, ok):
    assert ngff_.is_readable_store_schema_version(found) is ok


def test_require_readable_store_accepts_v4_refuses_v5(tmp_path):
    ngff_.require_readable_store(_padded_v4(tmp_path))
    with pytest.raises(ValueError, match="newer PhenoTypic"):
        ngff_.require_readable_store(_store(tmp_path, 5))  # version gate fires first


def test_an_older_build_refuses_a_padded_store(tmp_path, monkeypatch):
    monkeypatch.setattr(ngff_, "READABLE_STORE_SCHEMA_VERSIONS", frozenset({3}))
    with pytest.raises(ValueError, match="store_schema_version is 4"):
        ngff_.require_readable_store(_padded_v4(tmp_path))


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ({}, (0, 0)),
        ({"crop_frame": {**_FRAME, "padded": False}}, (0, 0)),
        ({"crop_frame": {**_FRAME, "padded": True}}, (5, 10)),
        ({"crop_frame": "garbage"}, (0, 0)),
    ],
)
def test_padded_crop_offset(block, expected):
    assert ngff_.padded_crop_offset(block) == expected


def test_padded_crop_window():
    block = {"crop_frame": {**_FRAME, "padded": True}}
    assert ngff_.padded_crop_window(block) == ((40, 60), (5, 10), (30, 40))
    assert ngff_.padded_crop_window({"crop_frame": {**_FRAME, "padded": False}}) is None


@pytest.mark.parametrize(
    "block",
    [
        {"store_schema_version": 4},  # v4 without a crop_frame
        {"store_schema_version": 3, "crop_frame": {**_FRAME, "padded": True}},  # v3 claiming padded
        {"store_schema_version": 4, "crop_frame": {**_FRAME, "padded": False}},
        {"store_schema_version": 4, "crop_frame": {"padded": True}},  # malformed
    ],
)
def test_inconsistent_crop_frames_are_refused(block):
    with pytest.raises(ValueError, match="crop_frame"):
        ngff_.check_crop_frame_consistency(block)


@pytest.mark.parametrize(
    "block",
    [
        {"store_schema_version": 3},
        {"store_schema_version": 3, "crop_frame": {**_FRAME, "padded": False}},
        {"store_schema_version": 4, "crop_frame": {**_FRAME, "padded": True}},
    ],
)
def test_consistent_crop_frames_pass(block):
    ngff_.check_crop_frame_consistency(block)


def _rewrite_root(store: Path, edit) -> None:
    root = json.loads((store / "zarr.json").read_text())
    edit(root["attributes"]["phenotypic"])
    (store / "zarr.json").write_text(json.dumps(root))


@pytest.mark.parametrize(
    "edit",
    [
        lambda b: b.update(store_schema_version=4),
        lambda b: b.update(crop_frame={**_FRAME, "padded": True}),
        lambda b: b.update(crop_frame="garbage"),
        # Consistent v4 + padded, but canvas_shape disagrees with the arrays' extent.
        lambda b: b.update(
            store_schema_version=4,
            crop_frame={"canvas_shape": [10_000, 10_000], "offset": [0, 0],
                        "roi_shape": [10, 10], "padded": True},
        ),
    ],
)
def test_valid_staged_store_rejects_inconsistent_crop_frames(tmp_path, edit):
    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate

    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "s.ome.zarr")
    assert ngff_.valid_staged_store(store) is True
    _rewrite_root(store, edit)
    assert ngff_.valid_staged_store(store) is False
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/sdk_/test_padded_store_version.py -q --capture=fd -p no:cacheprovider`
Expected: FAIL with `TypeError: build_phenotypic_attributes() got an unexpected keyword argument 'crop_frame'`.

- [ ] **Step 3: Write the implementation** in `ngff_.py`

Under `STORE_SCHEMA_VERSION` (`:57`):

```python
#: Written *instead of* :data:`STORE_SCHEMA_VERSION` by a store whose image
#: layers are zero-padded into a crop frame's canvas (spec
#: 2026-10-05-pseudo-cropping §5.2). A reader that does not know ``crop_frame``
#: would load the padded canvas as the image while every ``Bbox_*`` stays in ROI
#: coordinates; bumping only these stores makes such a reader refuse them, while
#: every other store keeps writing 3 and every existing tree stays readable.
PADDED_STORE_SCHEMA_VERSION: Final[int] = 4

#: Every ``store_schema_version`` this build decodes.
READABLE_STORE_SCHEMA_VERSIONS: Final[frozenset[int]] = frozenset(
    {STORE_SCHEMA_VERSION, PADDED_STORE_SCHEMA_VERSION}
)


def is_readable_store_schema_version(found: object) -> bool:
    """Whether *found* is a ``store_schema_version`` this build decodes.

    By value. Unhashable or wrongly typed values (another tool's JSON) are
    simply unreadable, never an exception -- the validity predicates that call
    this must return False, not raise.
    """
    if isinstance(found, bool) or not isinstance(found, int):
        return False
    return found in READABLE_STORE_SCHEMA_VERSIONS
```

`PhenotypicAttr`: next to `STORE_SCHEMA_VERSION`:

```python
    CROP_FRAME: Final[str] = "crop_frame"
```

`build_phenotypic_attributes`: add the keyword `crop_frame: Mapping[str, Any] | None = None` after `phenotypic_version`. Document it in `Args:` ("Serialised crop frame (`canvas_shape`, `offset`, `roi_shape`, `padded`), or `None` to omit the key. `padded: true` also writes :data:`PADDED_STORE_SCHEMA_VERSION`."). Then, before `return block`:

```python
    if crop_frame is not None:
        block[PhenotypicAttr.CROP_FRAME] = dict(crop_frame)
        if crop_frame.get("padded"):
            block[PhenotypicAttr.STORE_SCHEMA_VERSION] = PADDED_STORE_SCHEMA_VERSION
```

(Add `Any` and `Mapping` to the imports if they are missing.)

`require_readable_store`:

```python
    found = block.get(PhenotypicAttr.STORE_SCHEMA_VERSION)
    if not is_readable_store_schema_version(found):
        readable = " or ".join(str(v) for v in sorted(READABLE_STORE_SCHEMA_VERSIONS))
        raise ValueError(
            f"Cannot read {store_path}: store_schema_version is {found!r}, "
            f"but this build of PhenoTypic reads {readable}. "
            f"The store was written by a newer PhenoTypic -- upgrade the "
            f"package to read it."
        )
```

and, as the last statement before `return block` in `require_readable_store`:

```python
    check_crop_frame_consistency(block)
```

`valid_staged_store` (`:2003`): replace the version comparison with

```python
        if not is_readable_store_schema_version(
            block.get(PhenotypicAttr.STORE_SCHEMA_VERSION)
        ):
            return False
        # Spec §5.2 read invariant. ValueError is caught below -> False, so an
        # inconsistent store routes back to Stage 1 instead of aborting the
        # run at Stage 2's load_zarr.
        check_crop_frame_consistency(block)
```

and, just before the final `return bool(aligned_spatial) and all(...)`:

```python
        window = padded_crop_window(block)
        if window is not None and (
            not aligned_spatial or tuple(aligned_spatial[0]) != tuple(window[0])
        ):
            return False
```

New functions, after `require_readable_store`. They import the codec from the sibling `phenotypic.sdk_._crop_frame` (stdlib-only), **never** from `phenotypic._core`: that package's `__init__` loads the whole image stack onto GUI chunk routes and store probes (round-2 review R2-m1). The import is inside each function only to keep `ngff_`'s module import graph unchanged.

```python
def padded_crop_window(
    block: Mapping[str, Any],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    """``(canvas_shape, offset, roi_shape)`` of a padded store, else ``None``.

    ``None`` for an unpadded store, a store with no ``crop_frame``, and a
    malformed one -- display paths stay lenient; decoding paths go through
    :func:`check_crop_frame_consistency` (via ``require_readable_store``)
    and refuse.
    """
    from phenotypic.sdk_._crop_frame import crop_frame_from_attribute

    try:
        parsed = crop_frame_from_attribute(block.get(PhenotypicAttr.CROP_FRAME))
    except ValueError:
        return None
    if parsed is None or not parsed[2]:
        return None
    frame, roi_shape, _ = parsed
    return frame.canvas_shape, frame.offset, roi_shape


def padded_crop_offset(block: Mapping[str, Any]) -> tuple[int, int]:
    """``(row, col)`` to add to ROI coordinates to address this store's pixels.

    The store is the source of truth (spec §7): a padded store's layers sit at
    ``crop_frame.offset`` inside the canvas, so a measurement-table coordinate
    (always ROI-relative) must be shifted by it. Anything else returns
    ``(0, 0)``.
    """
    window = padded_crop_window(block)
    return (0, 0) if window is None else window[1]


def check_crop_frame_consistency(block: Mapping[str, Any]) -> None:
    """Refuse a store whose ``crop_frame`` cannot be trusted (spec §5.2).

    Raises:
        ValueError: If ``crop_frame`` is present but malformed, or if
            ``store_schema_version == 4`` and ``crop_frame.padded`` disagree.
            The message names ``crop_frame``.
    """
    from phenotypic.sdk_._crop_frame import crop_frame_from_attribute

    parsed = crop_frame_from_attribute(block.get(PhenotypicAttr.CROP_FRAME))
    padded = parsed is not None and parsed[2]
    version = block.get(PhenotypicAttr.STORE_SCHEMA_VERSION)
    if padded != (version == PADDED_STORE_SCHEMA_VERSION):
        raise ValueError(
            f"crop_frame padded={padded} disagrees with store_schema_version "
            f"{version!r}: a padded store must be version "
            f"{PADDED_STORE_SCHEMA_VERSION} and only a padded store may be"
        )
```

`_gui/browse/_tile_routes.py:215-219`:

```python
        if not isinstance(block, dict) or not ngff_.is_readable_store_schema_version(
            block.get(ngff_.PhenotypicAttr.STORE_SCHEMA_VERSION)
        ):
            raise TypeError("store schema is not readable")
```

Then `grep -rn "STORE_SCHEMA_VERSION" src/phenotypic` and confirm no `!=`/`==` comparison against it remains (other than writers).

**Migrate the existing "future version" tests.** They use `STORE_SCHEMA_VERSION + 1` (= 4) as the unreadable sentinel, and 4 is now readable. In `tests/unit/core/test_image_zarr_roundtrip.py` (`test_load_zarr_raises_on_a_newer_store_schema_version`, `test_load_layer_zarr_raises_…`, `:308-360`) and `tests/unit/sdk_/test_ngff_validity.py:~118`:
- replace `STORE_SCHEMA_VERSION + 1` with `max(READABLE_STORE_SCHEMA_VERSIONS) + 1`, importing it from `phenotypic.sdk_.ngff_`;
- update any message assertion to the new wording ("reads 3 or 4"). The assertion must still check that the found version appears in the message.

Then `grep -rn "STORE_SCHEMA_VERSION *+ *1" tests` must return nothing. **Do not** loosen the gate to make these pass.

- [ ] **Step 4: Run the tests and check they pass**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/sdk_/test_padded_store_version.py tests/unit/sdk_/test_load_zarr_guard.py tests/unit/sdk_/test_ngff_validity.py tests/unit/core/test_image_zarr_roundtrip.py tests/unit/gui/shared/test_tiles_zarr.py tests/gui/browse/test_tile_routes.py -q --capture=fd -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/sdk_/ngff_.py src/phenotypic/_gui/browse/_tile_routes.py tests/unit/sdk_/test_padded_store_version.py
git add src/phenotypic/sdk_/ngff_.py src/phenotypic/_gui/browse/_tile_routes.py tests/unit/sdk_/test_padded_store_version.py
git commit -m "feat(sdk): crop_frame store attribute; v4 for padded stores; readable {3,4}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD"
```

---

### Task 9: Writer padding (`save2zarr`, `_save_store`, `save_intermediate_zarr`)

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py`: `_build_store_attributes` (`:1016`), `save2zarr` (`:1083`), `_save_store` (`:1151`, which only allocates the part and calls `_write_store_part` at `:1224`), **`_write_store_part` (`:1243`, which holds the `arrays` dict, the series/objmap writes, the OME metadata, the XML and the attribute build; every padding edit goes here)**, `save_intermediate_zarr` (`:1487`)
- Modify: `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py:1040-1092` (`apply_with_intermediates`: builder previews stay ROI-sized, a spec §7 decision)
- Test: `tests/unit/core/test_crop_frame_zarr_write.py`

**Interfaces:**
- Consumes: `_save_padding_frame`, `_valid_crop_frame` (Task 2); `pad_to_canvas`, `crop_frame_to_attribute` (Task 1); `build_phenotypic_attributes(crop_frame=)` (Task 8).
- Produces:
  - `Image._canvas_shape_for_save(pad_on_save: bool | None) -> tuple[int, int]`
  - `save2zarr(path, *, ..., pad_on_save: bool | None = None)`
  - `_save_store(..., pad_on_save: bool | None = None)`
  - `save_intermediate_zarr(path, layers, *, pad_on_save: bool | None = None)`

- [ ] **Step 1: Write the failing tests**

```python
"""Padded stores hold the ROI at its offset and zeros elsewhere (spec §5.1-5.2)."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.correction import CropImage
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_.ngff_ import read_phenotypic_attributes

CROP = dict(top=7, bottom=11, left=13, right=17)


@pytest.fixture
def plate() -> Image:
    return Image(load_synth_yeast_plate())


@pytest.fixture
def cropped(plate) -> Image:
    return CropImage(**CROP).apply(plate)


def _outside_is_zero(canvas: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    mask = np.ones(canvas.shape[:2], dtype=bool)
    mask[r:r + h, c:c + w] = False
    return not canvas[mask].any()


def test_padded_store_layers_are_canvas_sized(tmp_path, plate, cropped):
    store = cropped.save2zarr(tmp_path / "c.ome.zarr")
    H, W = plate.shape[:2]
    h, w = cropped.shape[:2]
    for layer, roi in (
        ("rgb", cropped.rgb[:]),
        ("gray", cropped.gray[:]),
        ("detect_mat", cropped.detect_mat[:]),
        ("objmap", cropped.objmap[:]),
    ):
        canvas = Image.load_layer_zarr(store, layer)
        assert canvas.shape[:2] == (H, W), layer
        np.testing.assert_array_equal(canvas[7:7 + h, 13:13 + w], roi, err_msg=layer)
        assert _outside_is_zero(canvas, 7, 13, h, w), layer


def test_padded_store_attributes(tmp_path, plate, cropped):
    block = read_phenotypic_attributes(cropped.save2zarr(tmp_path / "c.ome.zarr"))
    assert block["store_schema_version"] == 4
    assert block["crop_frame"] == {
        "canvas_shape": list(plate.shape[:2]),
        "offset": [7, 13],
        "roi_shape": list(cropped.shape[:2]),
        "padded": True,
    }


def test_original_series_stays_canvas_sized(tmp_path, plate):
    plate._retain_original()  # what the CLI does before any op runs
    cropped = CropImage(**CROP).apply(plate)
    store = cropped.save2zarr(tmp_path / "c.ome.zarr")
    block = read_phenotypic_attributes(store)
    assert "original" in block["series"]
    # load_layer_zarr moves the channel axis last only for layer == "rgb"; the
    # "original" series is stored channel-first like rgb, so move it here.
    original = np.moveaxis(Image.load_layer_zarr(store, "original"), 0, -1)
    assert original.shape[:2] == plate.shape[:2]
    np.testing.assert_array_equal(original, plate.rgb[:])


def test_noop_crop_writes_padded_v4(tmp_path, plate):
    """Decided 2026-10-05: every CropImage result saves padded, no special case."""
    block = read_phenotypic_attributes(CropImage().apply(plate).save2zarr(tmp_path / "n.ome.zarr"))
    assert block["store_schema_version"] == 4
    assert block["crop_frame"]["offset"] == [0, 0]
    assert block["crop_frame"]["padded"] is True


def test_builder_intermediates_stay_roi_sized(tmp_path, plate, cropped):
    from phenotypic import ImagePipeline

    ImagePipeline(ops={"crop": CropImage(**CROP)}).apply_with_intermediates(plate, output_dir=tmp_path)
    # A corrector modifies every layer, so its snapshot is a full `base_NN`
    # store, not a `{NN}_{key}` delta (_image_pipeline_core.py:120-121,
    # 1076-1080). `base_00` is the pre-crop input and is legitimately canvas-sized.
    stores = sorted(p for p in tmp_path.glob("*.ome.zarr") if p.name != "base_00.ome.zarr")
    assert stores, "apply_with_intermediates wrote no post-crop snapshot"
    for store in stores:
        assert Image.load_layer_zarr(store, "gray").shape == cropped.shape[:2]


def test_pad_on_save_false_overrides(tmp_path, cropped):
    store = cropped.save2zarr(tmp_path / "c.ome.zarr", pad_on_save=False)
    block = read_phenotypic_attributes(store)
    assert block["store_schema_version"] == 3
    assert block["crop_frame"]["padded"] is False
    assert Image.load_layer_zarr(store, "gray").shape == cropped.shape[:2]


def test_pad_on_save_true_pads_an_object_crop(tmp_path, plate, cropped):
    colony = cropped.objects[0]
    store = colony.save2zarr(tmp_path / "o.ome.zarr", pad_on_save=True)
    assert Image.load_layer_zarr(store, "gray").shape == plate.shape[:2]
    assert read_phenotypic_attributes(colony.save2zarr(tmp_path / "o2.ome.zarr"))["store_schema_version"] == 3


def test_uncropped_store_has_no_crop_frame(tmp_path, plate):
    block = read_phenotypic_attributes(plate.save2zarr(tmp_path / "p.ome.zarr"))
    assert block["store_schema_version"] == 3
    assert "crop_frame" not in block


def test_intermediate_store_is_padded(tmp_path, plate, cropped):
    store = cropped.save_intermediate_zarr(tmp_path / "i.ome.zarr", layers=("gray",))
    assert Image.load_layer_zarr(store, "gray").shape == plate.shape[:2]


def test_stale_frame_saves_unpadded(tmp_path, cropped):
    from phenotypic._core._crop_frame import CropFrame

    cropped._crop_frame = CropFrame((5, 5), (0, 0))
    with pytest.warns(UserWarning, match="no longer fits"):
        store = cropped.save2zarr(tmp_path / "s.ome.zarr")
    block = read_phenotypic_attributes(store)
    assert block["store_schema_version"] == 3 and "crop_frame" not in block
```

The `"original"` test retains the snapshot **before** cropping, as the CLI does. The apply wrapper carries `_original` across the op (`_provenance.py:523`), so the store's `original` series is the full canvas and must equal the plate's pixels.

- [ ] **Step 2: Run the tests and check they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core/test_crop_frame_zarr_write.py -q --capture=fd -p no:cacheprovider`
Expected: FAIL with `assert (H-18, W-30) == (H, W)` and `TypeError: ... 'pad_on_save'`.

- [ ] **Step 3: Write the implementation** in `_image_io_handler.py`

3a. New method on the IO handler, placed before `save2zarr`:

```python
    def _canvas_shape_for_save(self, pad_on_save: bool | None) -> tuple[int, int]:
        """2-D extent a save will write: the crop canvas when padding applies."""
        frame = self._save_padding_frame(pad_on_save)
        if frame is not None:
            return frame.canvas_shape
        height, width = self.shape[:2]
        return int(height), int(width)
```

3b. `_build_store_attributes`: add the keyword `crop_frame: dict | None = None` (document it) and pass `crop_frame=crop_frame` to `ngff_.build_phenotypic_attributes(...)`.

3c. `save2zarr`: add the keyword `pad_on_save: bool | None = None`. In `Args:` write: "`pad_on_save`: Pad image layers back into the crop frame's original canvas with zeros. `None` (default) uses the image's own setting, which `CropImage` turns on; `True`/`False` override it for this call." Replace the two lines

```python
        gray = self.gray[:]
        pyramid_height, pyramid_width = gray.shape[:2]
```

with

```python
        pyramid_height, pyramid_width = self._canvas_shape_for_save(pad_on_save)
```

and pass `pad_on_save=pad_on_save` to `self._save_store(...)`.

3d. Add the keyword `pad_on_save: bool | None = None` (documented as in 3c) to **both** `_save_store` and `_write_store_part`, and pass `pad_on_save=pad_on_save` in `_save_store`'s `self._write_store_part(...)` call (`:1224`). Every remaining edit in this step is inside **`_write_store_part`**. Directly after its `arrays` dict (and its `"original"` entry) is built (`~:1290-1305`), add:

```python
        # Spec 2026-10-05-pseudo-cropping §5.1: image layers are written into
        # the crop canvas with zeros outside the ROI. `arrays` keeps the compact
        # ROI arrays; each layer is padded only as it is written, so at most one
        # canvas-sized layer exists at a time. `original` is already canvas-sized.
        pad_frame = self._save_padding_frame(pad_on_save)
        valid_frame = self._valid_crop_frame()
        roi_shape = tuple(int(v) for v in self.shape[:2])

        def _as_written(series_name: str, array: np.ndarray) -> np.ndarray:
            if pad_frame is None or series_name == "original":
                return array
            return pad_to_canvas(array, pad_frame, row_axis=array.ndim - 2)

        def _written_shape(series_name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
            if pad_frame is None or series_name == "original":
                return tuple(shape)
            return (*shape[:-2], *pad_frame.canvas_shape)
```

Add `from phenotypic._core._crop_frame import crop_frame_to_attribute, pad_to_canvas` to the function's local imports.

Then make these edits in the same function:
- In the `# 1. arrays and chunks` loop, pass `_as_written(name, arrays[name])` instead of `arrays[name]` to `self._write_series`.
- Objmap write: `self._write_series(part, ngff_.objmap_path(primary), _as_written("objmap", objmap), levels)`.
- In the `# 2.` loop: `shapes = ngff_.pyramid_level_shapes(_written_shape(series_name, arrays[series_name].shape), levels)`.
- `label_shapes = ngff_.pyramid_level_shapes(_written_shape("objmap", objmap.shape), levels)`.
- `build_ome_xml(series_shapes={name_: _written_shape(name_, arrays[name_].shape) for name_ in series_names}, ...)`.
- `_build_store_attributes(..., crop_frame=None if valid_frame is None else crop_frame_to_attribute(valid_frame, roi_shape, pad_frame is not None))`.

3e. `save_intermediate_zarr`: add `*, pad_on_save: bool | None = None` after `layers`, document it, and pass it to `_save_store`.

3f. `apply_with_intermediates` (`_image_pipeline_core.py:1040-1092`): pass `pad_on_save=False` to each of its five snapshot writes (the two `save2zarr(...)` and three `save_intermediate_zarr(...)` calls). Add one comment at the first:

```python
                # Builder previews show the cropped region itself (spec
                # 2026-10-05-pseudo-cropping §7); padding is for run outputs.
```

- [ ] **Step 4: Run the tests and check they pass**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core/test_crop_frame_zarr_write.py tests/unit/core/test_image_zarr_roundtrip.py tests/unit/core/test_save_intermediate_zarr.py tests/unit/core/test_ngff_conformance.py tests/unit/core/test_image_provenance_original_zarr.py tests/unit/core/test_full_layers_intermediates.py tests/unit/core/test_delta_intermediates.py -q --capture=fd -p no:cacheprovider`
Expected: all PASS. The existing round-trip tests prove an unpadded image's store is unchanged.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/_core/_image_parts/_image_io_handler.py tests/unit/core/test_crop_frame_zarr_write.py
git add src/phenotypic/_core/_image_parts/_image_io_handler.py tests/unit/core/test_crop_frame_zarr_write.py
git commit -m "feat(io): pad_on_save — zero-pad cropped layers into the original canvas on save

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD"
```

---

### Task 10: Reader: slice padded layers back and restore the frame

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py`: `_load_from_store` (`~:1671-1790`), `_read_store_array` (`~:1792`), `_imread_store` (`~:880-905`), `save2pickle` (`~:2076`), `load_pickle` (`~:2147-2215`)
- Test: `tests/unit/core/test_crop_frame_zarr_read.py`

**Interfaces:**
- Consumes: `crop_frame_from_attribute` (Task 1); `PhenotypicAttr.CROP_FRAME` (Task 8); writer (Task 9).
- Produces: `_read_store_array(path, member, *, layer="", window: tuple[slice, slice] | None = None)`. `load_zarr` returns an ROI-sized image with `_crop_frame`/`_pad_on_save` restored.

- [ ] **Step 1: Write the failing tests**

```python
"""load_zarr(save2zarr(img)) == img, frame included (spec §5.3)."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import GridImage, Image
from phenotypic.correction import CropImage
from phenotypic.data import load_synth_yeast_plate

CROP = dict(top=7, bottom=11, left=13, right=17)


def _round_trip(img, path, cls=Image, **save_kwargs):
    return cls.load_zarr(img.save2zarr(path, **save_kwargs))


def test_padded_round_trip_is_exact(tmp_path):
    cropped = CropImage(**CROP).apply(Image(load_synth_yeast_plate()))
    loaded = _round_trip(cropped, tmp_path / "c.ome.zarr")
    assert loaded == cropped
    assert loaded._crop_frame == cropped._crop_frame
    assert loaded._pad_on_save is True
    assert loaded.shape == cropped.shape


def test_unpadded_round_trip_restores_frame_without_padding(tmp_path):
    cropped = CropImage(**CROP).apply(Image(load_synth_yeast_plate()))
    loaded = _round_trip(cropped, tmp_path / "c.ome.zarr", pad_on_save=False)
    assert loaded == cropped
    assert loaded._crop_frame == cropped._crop_frame
    assert loaded._pad_on_save is False


def test_grid_round_trip(tmp_path):
    grid = CropImage(**CROP).apply(GridImage(load_synth_yeast_plate(), nrows=8, ncols=12))
    loaded = _round_trip(grid, tmp_path / "g.ome.zarr", cls=GridImage)
    assert isinstance(loaded, GridImage) and loaded.nrows == 8
    assert loaded == grid and loaded._crop_frame == grid._crop_frame


def test_grayscale_only_crop_round_trips(tmp_path):
    gray = Image(np.random.default_rng(3).random((50, 70)).astype(np.float32))
    cropped = CropImage(top=4, left=6, bottom=2, right=3).apply(gray)
    store = cropped.save2zarr(tmp_path / "g.ome.zarr")
    assert Image.load_layer_zarr(store, "gray").shape == (50, 70)
    loaded = Image.load_zarr(store)
    assert loaded == cropped and loaded._crop_frame == cropped._crop_frame


def test_recrop_after_load_composes_onto_the_original_canvas(tmp_path):
    plate = Image(load_synth_yeast_plate())
    loaded = _round_trip(CropImage(**CROP).apply(plate), tmp_path / "c.ome.zarr")
    again = CropImage(top=2, left=3).apply(loaded)
    assert again._crop_frame.offset == (9, 16)
    assert again._crop_frame.canvas_shape == tuple(plate.shape[:2])


def test_imread_reads_the_canvas(tmp_path):
    plate = Image(load_synth_yeast_plate())
    store = CropImage(**CROP).apply(plate).save2zarr(tmp_path / "c.ome.zarr")
    assert Image.imread(store).shape[:2] == plate.shape[:2]


def test_malformed_crop_frame_is_refused(tmp_path):
    import json

    store = CropImage(**CROP).apply(Image(load_synth_yeast_plate())).save2zarr(tmp_path / "c.ome.zarr")
    root = json.loads((store / "zarr.json").read_text())
    root["attributes"]["phenotypic"]["crop_frame"] = {"padded": True}
    (store / "zarr.json").write_text(json.dumps(root))
    with pytest.raises(ValueError, match="crop_frame"):
        Image.load_zarr(store)


def test_crop_frame_disagreeing_with_the_arrays_is_refused(tmp_path):
    """zarr truncates out-of-bounds slices silently; the loader must not."""
    import json

    store = CropImage(**CROP).apply(Image(load_synth_yeast_plate())).save2zarr(tmp_path / "c.ome.zarr")
    root = json.loads((store / "zarr.json").read_text())
    frame = root["attributes"]["phenotypic"]["crop_frame"]
    frame["canvas_shape"] = [frame["canvas_shape"][0] + 50, frame["canvas_shape"][1] + 50]
    (store / "zarr.json").write_text(json.dumps(root))
    with pytest.raises(ValueError, match="crop_frame"):
        Image.load_zarr(store)


def test_pre_change_crop_store_measures_null_offsets(tmp_path):
    """A crop store written before crop frames existed: ROI-sized, v3, no crop_frame,
    but its journal records CropImage. Re-measuring must not claim 'not cropped'."""
    import json

    cropped = CropImage(**CROP).apply(Image(load_synth_yeast_plate()))
    store = cropped.save2zarr(tmp_path / "old.ome.zarr", pad_on_save=False)
    root = json.loads((store / "zarr.json").read_text())
    del root["attributes"]["phenotypic"]["crop_frame"]  # what a pre-change writer produced
    (store / "zarr.json").write_text(json.dumps(root))
    loaded = Image.load_zarr(store)
    assert loaded._crop_frame is None
    with pytest.warns(UserWarning, match="unknown"):
        info = loaded.objects.info(include_metadata=False)
    assert info[["Frame_OffsetRR", "Frame_OffsetCC"]].isna().all().all()


def test_imread_of_a_padded_store_reports_zero_offsets(tmp_path):
    """Process output is valid CLI input: imread returns the canvas, so the
    offset is 0 -- not null, even though the journal records the crop."""
    import warnings

    from phenotypic._core._crop_frame import CropFrame

    plate = Image(load_synth_yeast_plate())
    read = Image.imread(CropImage(**CROP).apply(plate).save2zarr(tmp_path / "c.ome.zarr"))
    assert read._crop_frame == CropFrame(tuple(plate.shape[:2]), (0, 0))
    assert read._pad_on_save is False
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert read._frame_offsets_for_info() == (0, 0)


def test_imread_of_an_unpadded_store_restores_the_recorded_frame(tmp_path):
    cropped = CropImage(**CROP).apply(Image(load_synth_yeast_plate()))
    read = Image.imread(cropped.save2zarr(tmp_path / "u.ome.zarr", pad_on_save=False))
    assert read._crop_frame == cropped._crop_frame
    assert read._pad_on_save is False
    assert read._frame_offsets_for_info() == (7, 13)


def test_save2pickle_round_trips_the_frame(tmp_path):
    cropped = CropImage(**CROP).apply(Image(load_synth_yeast_plate()))
    cropped.save2pickle(tmp_path / "c.pkl")
    loaded = Image.load_pickle(tmp_path / "c.pkl")
    assert loaded._crop_frame == cropped._crop_frame
    assert loaded._pad_on_save is True
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core/test_crop_frame_zarr_read.py -q --capture=fd -p no:cacheprovider`
Expected: FAIL. The loaded shape is the canvas, not the ROI.

- [ ] **Step 3: Write the implementation**

`_read_store_array`: add the keyword `window: tuple[slice, slice] | None = None` (document: "Canvas rows/cols to read; `None` reads the whole level"), and replace the read with:

```python
        array_handle = zarr.open_array(
            store=ngff_.long_path(Path(path) / member / "0"), mode="r"
        )
        selection = (
            (Ellipsis,) if window is None else (Ellipsis, window[0], window[1])
        )
        array = np.asarray(array_handle[selection])
```

`_load_from_store`: right after `sections = block.get(...)`, add:

```python
        from phenotypic._core._crop_frame import crop_frame_from_attribute

        # Spec 2026-10-05-pseudo-cropping §5.3: a padded store's image layers
        # are canvas-sized; read only the ROI window so the image is
        # bit-identical to the one saved. `original` is read whole.
        crop = crop_frame_from_attribute(block.get(ngff_.PhenotypicAttr.CROP_FRAME))
        window = (
            crop[0].window(crop[1]) if crop is not None and crop[2] else None
        )
```

and, directly after it, the extent check (spec §5.3). zarr truncates an out-of-bounds slice silently, so a canvas that disagrees with the arrays would otherwise yield a truncated image:

```python
        if window is not None:
            level0 = ngff_.store_level0_shape(Path(path), series["gray"])
            if level0 is None or tuple(level0[-2:]) != crop[0].canvas_shape:
                raise ValueError(
                    f"{path}: crop_frame canvas_shape {crop[0].canvas_shape} "
                    f"disagrees with the stored extent {level0}"
                )
```

Pass `window=window` to the `_read_store_array` calls for `series["gray"]`, `series["rgb"]`, `series["detect_mat"]` and `labels[ngff_.OBJMAP_LABEL]`. **Not** to `series["original"]`. Then, immediately before the final `return cast("Image", img)`:

```python
        if crop is not None:
            if tuple(img.shape[:2]) != crop[1]:
                raise ValueError(
                    f"{path}: crop_frame roi_shape {crop[1]} disagrees with the "
                    f"layers read ({tuple(img.shape[:2])})"
                )
            img._crop_frame = crop[0]
            img._pad_on_save = crop[2]
```

(The version ⇔ `padded` invariant is already enforced: `load_zarr` and `_io_constants.py:2787` both pass through `require_readable_store`, which calls `check_crop_frame_consistency` (Task 8).)

**`imread` of a store (spec §5.3, round-2 review R2-M4).** In `_imread_store` (`_image_io_handler.py:~880-905`), directly after the provenance-journal block:

```python
        # The journal just copied may record a CropImage; without a frame the
        # Frame_* rule would then report null offsets for pixels whose offset
        # is known. Give the image the frame that describes what was read.
        from phenotypic.sdk_._crop_frame import CropFrame, crop_frame_from_attribute

        try:
            recorded = crop_frame_from_attribute(
                spec.phenotypic.get(ngff_.PhenotypicAttr.CROP_FRAME)
            )
        except ValueError:
            recorded = None
        if recorded is not None:
            frame, roi_shape, padded = recorded
            shape2d = tuple(image.shape[:2])
            if padded and shape2d == frame.canvas_shape:
                image._crop_frame = CropFrame(shape2d, (0, 0))  # the pixels ARE the canvas
            elif not padded and shape2d == roi_shape:
                image._crop_frame = frame
            # imread reads plain pixels; re-saving them must not re-pad.
            image._pad_on_save = False
```

A level > 0 read has a different shape, so it gets no frame; that is honest. If `spec.phenotypic` can be `None` for a third-party store, guard with `(spec.phenotypic or {})`. Check `read_ngff_image_spec`'s return type.

**Pickle files (spec §5.3).** In `save2pickle`, add to `data2save`:

```python
                # Spec 2026-10-05-pseudo-cropping §5.3: plain tuples, so the
                # file never pickles a phenotypic class reference for the frame.
                "crop_frame": (
                    None
                    if self._crop_frame is None
                    else (self._crop_frame.canvas_shape, self._crop_frame.offset)
                ),
                "pad_on_save": bool(self._pad_on_save),
```

In `load_pickle`, immediately before `return instance`:

```python
        # `.get`: pickle files written before crop frames existed carry neither key.
        stored_frame = loaded.get("crop_frame")
        if stored_frame is not None:
            from phenotypic._core._crop_frame import CropFrame

            instance._crop_frame = CropFrame(*stored_frame)
            instance._pad_on_save = bool(loaded.get("pad_on_save", False))
```

- [ ] **Step 4: Run the tests and check they pass**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core/test_crop_frame_zarr_read.py tests/unit/core/test_crop_frame_zarr_write.py tests/unit/core/test_image_zarr_roundtrip.py tests/unit/core/test_grid_image_zarr_roundtrip.py tests/unit/core/test_image_pickle.py tests/unit/sdk_/test_load_zarr_guard.py -q --capture=fd -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/_core/_image_parts/_image_io_handler.py tests/unit/core/test_crop_frame_zarr_read.py
git add src/phenotypic/_core/_image_parts/_image_io_handler.py tests/unit/core/test_crop_frame_zarr_read.py
git commit -m "feat(io): load_zarr slices padded stores back to the ROI and restores the frame

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD"
```

---

### Task 11: Accessor `imsave` honours `pad_on_save`

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_io_handler.py:456` (`imsave`)
- Modify: `src/phenotypic/_core/_image_parts/accessor_abstracts/_multichannel_accessor.py:44` (`imsave`)
- Modify: `src/phenotypic/_core/_image_parts/accessors/_objmap_accessor.py:571` (`imsave`)
- Test: `tests/unit/core/test_crop_frame_imsave.py`

**Interfaces:**
- Consumes: `_pad_layer_for_save` (Task 2).
- Produces: `<accessor>.imsave(filepath, bit_depth=None, *, pad_on_save: bool | None = None)`. `ObjectMap.imsave` keeps `use_label2rgb` and adds `pad_on_save`.

- [ ] **Step 1: Write the failing tests**

```python
"""Flat layer exports follow the same pad rule as stores (spec §6.2)."""

from __future__ import annotations

import numpy as np
import tifffile
from PIL import Image as PILImage

from phenotypic import Image
from phenotypic.correction import CropImage
from phenotypic.data import load_synth_yeast_plate

CROP = dict(top=7, bottom=11, left=13, right=17)


def test_flat_exports_are_canvas_sized_by_default(tmp_path):
    plate = Image(load_synth_yeast_plate())
    cropped = CropImage(**CROP).apply(plate)
    cropped.gray.imsave(tmp_path / "g.tiff")
    cropped.detect_mat.imsave(tmp_path / "d.tiff")
    cropped.rgb.imsave(tmp_path / "r.tiff")
    cropped.objmap.imsave(tmp_path / "o.png")
    H, W = plate.shape[:2]
    assert tifffile.imread(tmp_path / "g.tiff").shape[:2] == (H, W)
    assert tifffile.imread(tmp_path / "d.tiff").shape[:2] == (H, W)
    assert tifffile.imread(tmp_path / "r.tiff").shape[:2] == (H, W)
    objmap = np.asarray(PILImage.open(tmp_path / "o.png"))
    assert objmap.shape[:2] == (H, W)
    assert not objmap[:7].any() and not objmap[:, :13].any()


def test_flat_export_override(tmp_path):
    cropped = CropImage(**CROP).apply(Image(load_synth_yeast_plate()))
    cropped.gray.imsave(tmp_path / "g.tiff", pad_on_save=False)
    assert tifffile.imread(tmp_path / "g.tiff").shape[:2] == cropped.shape[:2]


def test_uncropped_export_unchanged(tmp_path):
    plate = Image(load_synth_yeast_plate())
    plate.gray.imsave(tmp_path / "g.tiff")
    assert tifffile.imread(tmp_path / "g.tiff").shape[:2] == plate.shape[:2]
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core/test_crop_frame_imsave.py -q --capture=fd -p no:cacheprovider`
Expected: FAIL. The exports are ROI-sized, and the override raises `TypeError`.

- [ ] **Step 3: Write the implementation**

`AccessorIOHandler.imsave`: change the signature to
`def imsave(self, filepath: str | Path | None = None, bit_depth: Literal[8, 16] | None = None, *, pad_on_save: bool | None = None) -> None:`. Add to `Args:` "pad_on_save: Zero-pad into the crop frame's original canvas; `None` uses the image's setting (on after `CropImage`)." Replace `arr2save = self._subject_arr` with:

```python
        arr2save = self._root_image._pad_layer_for_save(self._subject_arr, pad_on_save)
```

`MultiChannelAccessor.imsave`: same signature change and docstring line. Replace `arr = self._subject_arr.copy()` with:

```python
        arr = self._root_image._pad_layer_for_save(self._subject_arr, pad_on_save).copy()
```

`ObjectMap.imsave`: add `*, pad_on_save: bool | None = None` after `use_label2rgb` and add the docstring line. Change `super().imsave(filepath=filepath, bit_depth=bit_depth)` to `super().imsave(filepath=filepath, bit_depth=bit_depth, pad_on_save=pad_on_save)`, and in the `else` branch replace `label2rgb(self._subject_arr, bg_label=0)` with `label2rgb(self._root_image._pad_layer_for_save(self._subject_arr, pad_on_save), bg_label=0)`.

Check other overrides: run `grep -rn "def imsave" src/phenotypic/_core`. Any other **image-layer** override that calls `super().imsave(...)` or reads `_subject_arr` gets the same treatment. The colour-space accessor's `imsave` (`_color_space_accessor.py:120`) is not a layer and does not call the layer `imsave`, so leave it **unchanged**; derived colour spaces are not part of the "full-frame layers on disk" contract.

- [ ] **Step 4: Run the tests and check they pass**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core/test_crop_frame_imsave.py tests/unit/core -k "imsave or save" -q --capture=fd -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_io_handler.py src/phenotypic/_core/_image_parts/accessor_abstracts/_multichannel_accessor.py src/phenotypic/_core/_image_parts/accessors/_objmap_accessor.py tests/unit/core/test_crop_frame_imsave.py
git add -u src/phenotypic/_core
git add tests/unit/core/test_crop_frame_imsave.py
git commit -m "feat(io): accessor imsave honours pad_on_save

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD"
```

- [ ] **Step 6: Phase 3 affected surface (run once)**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/core tests/unit/sdk_ tests/unit/gui/shared tests/unit/gui/results_viewer tests/unit/gui/builder tests/gui/browse tests/unit/test_ome_zarr_invariants.py tests/unit/test_ngff_schema_fixtures.py -q --capture=fd -p no:cacheprovider`
Expected: no new failures against `main`. Run each failing test on its own before you attribute it.

---

## Phase 4: CLI

### Task 12: `--mode process` canvas levels and the revision bump

**Files:**
- Modify: `src/phenotypic/_cli/_cli_process_only.py:~210` (zarr-branch level count)
- Modify: `src/phenotypic/_cli/_cli_failure_tracker.py:209` (revision 3 → 4 + history line)
- Modify: `tests/integration/cli/test_figures_in_store.py:268` (`== 3` → `== 4`, comment updated)
- Test: `tests/unit/cli/test_process_only_crop_frame.py`

**Interfaces:**
- Consumes: `_canvas_shape_for_save` (Task 9); accessor padding (Task 11).

- [ ] **Step 1: Write the failing tests**

```python
"""Process-mode exports of a cropped image are canvas-sized (spec §6.2)."""

from __future__ import annotations

import tifffile

from phenotypic import Image
from phenotypic._cli import _cli_failure_tracker as tracker
from phenotypic._cli._cli_process_only import write_process_only_layer
from phenotypic.correction import CropImage
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_.ngff_ import read_phenotypic_attributes


def test_revision_bumped_for_padded_exports():
    assert tracker.PROCESS_LAYER_SEMANTICS_REVISION == 4


def test_zarr_export_is_padded(tmp_path):
    plate = Image(load_synth_yeast_plate())
    cropped = CropImage(top=7, left=13).apply(plate)
    out = tmp_path / "x.ome.zarr"
    write_process_only_layer(cropped, "rgb", out, fmt="zarr")
    block = read_phenotypic_attributes(out)
    assert block["store_schema_version"] == 4
    assert Image.imread(out).shape[:2] == plate.shape[:2]


def test_tiff_export_is_padded(tmp_path):
    plate = Image(load_synth_yeast_plate())
    cropped = CropImage(top=7, left=13).apply(plate)
    out = tmp_path / "x.tiff"
    write_process_only_layer(cropped, "detect_mat", out, fmt="tiff")
    assert tifffile.imread(out).shape[:2] == plate.shape[:2]
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_process_only_crop_frame.py -q --capture=fd -p no:cacheprovider`
Expected: `test_revision_bumped_for_padded_exports` FAILS (3 ≠ 4). `test_zarr_export_is_padded` may fail on the level count or conformance.

- [ ] **Step 3: Write the implementation**

`_cli_process_only.py`, in the zarr branch:

```python
        # The canvas, not `image.shape`, when the export is padded back into a
        # crop frame (spec 2026-10-05-pseudo-cropping §6.2); otherwise the same
        # two numbers as before.
        height, width = image._canvas_shape_for_save(None)
```

(Replace the existing `height, width = image.shape[:2]` and update the comment above it.)

`_cli_failure_tracker.py`:

```python
#: 3 -> 4: a cropped image's layers export zero-padded into the original frame
#:         (spec 2026-10-05-pseudo-cropping §6.2), so a process tree from before
#:         the change is re-derived rather than mixed.
PROCESS_LAYER_SEMANTICS_REVISION = 4
```

`tests/integration/cli/test_figures_in_store.py:268`: change to `assert tracker.PROCESS_LAYER_SEMANTICS_REVISION == 4`, and update its comment to say the figures change took it to 3 and padded crop exports to 4. Keep it exact; don't loosen it.

- [ ] **Step 4: Run the tests and check they pass**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_process_only_crop_frame.py tests/unit/cli/test_process_only_zarr.py tests/unit/cli/test_cli_process_only.py tests/unit/cli/test_work_id_semantics_revision.py tests/unit/cli/test_process_only_consolidated.py -q --capture=fd -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_process_only.py src/phenotypic/_cli/_cli_failure_tracker.py tests/unit/cli/test_process_only_crop_frame.py tests/integration/cli/test_figures_in_store.py
git add src/phenotypic/_cli/_cli_process_only.py src/phenotypic/_cli/_cli_failure_tracker.py tests/unit/cli/test_process_only_crop_frame.py tests/integration/cli/test_figures_in_store.py
git commit -m "feat(cli): process-mode exports pad cropped layers; layer semantics revision 4

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD"
```

---

### Task 13: Crop-only work-id key

**Files:**
- Modify: `src/phenotypic/_cli/_cli_failure_tracker.py` (`compute_work_id` `:304`, `work_id_for_image` `:390`; new constant and helpers)
- Modify: `src/phenotypic/_cli/_cli_process_single.py:148` (`_worker_work_identity`)
- Test: `tests/unit/cli/test_work_id_crop_frame.py`

**Interfaces:**
- Produces:
  - `CROP_FRAME_SEMANTICS_REVISION = 1`
  - `pipeline_uses_crop_frame(pipeline_json: Path) -> bool`
  - `compute_work_id(..., pipeline_json: Path | None = None)`, keyword-only and optional. Callers that don't pass it get exactly today's digest.

- [ ] **Step 1: Write the failing tests** (pattern: `tests/unit/cli/test_work_id_raw_revision.py`)

```python
"""Only pipelines containing CropImage change their work id (spec §6.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from phenotypic import ImagePipeline
from phenotypic._cli import _cli_failure_tracker as tracker
from phenotypic._cli._cli_failure_tracker import (
    compute_work_id,
    pipeline_uses_crop_frame,
    work_id_for_image,
)
from phenotypic._cli._cli_process_single import _worker_work_identity
from phenotypic.correction import CropImage
from phenotypic.detect import OtsuDetector
from tests.unit.cli._preflight_support import make_config

_FIXED = dict(
    dataset="plate1",
    relative_image_path="plate1/img001.tiff",
    input_sha256="a" * 64,
    pipeline_fingerprint="b" * 64,
    processing_config_digest="c" * 64,
    mode="full",
)
#: Same value test_work_id_raw_revision pins (computed at 81d19ec).
TIFF_DIGEST_AT_81D19EC = "0397ca238f767b76d109685e63db2f55a5cb128dc3f1829a0aa19a1716b7df51"


def _write(tmp_path: Path, name: str, pipeline: ImagePipeline) -> Path:
    path = tmp_path / name
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return path


@pytest.fixture
def otsu(tmp_path):
    return _write(tmp_path, "otsu.json", ImagePipeline(ops={"d": OtsuDetector()}))


@pytest.fixture
def cropping(tmp_path):
    return _write(tmp_path, "crop.json", ImagePipeline(ops={"c": CropImage(top=1), "d": OtsuDetector()}))


@pytest.fixture
def nested_cropping(tmp_path):
    inner = ImagePipeline(ops={"c": CropImage(top=1)})
    return _write(tmp_path, "nested.json", ImagePipeline(ops={"inner": inner, "d": OtsuDetector()}))


def test_detection(otsu, cropping, nested_cropping):
    assert pipeline_uses_crop_frame(otsu) is False
    assert pipeline_uses_crop_frame(cropping) is True
    assert pipeline_uses_crop_frame(nested_cropping) is True


def test_a_pipeline_without_a_crop_keeps_its_digest(otsu):
    assert compute_work_id(**_FIXED, pipeline_json=otsu) == TIFF_DIGEST_AT_81D19EC
    assert compute_work_id(**_FIXED) == TIFF_DIGEST_AT_81D19EC


def test_a_crop_pipeline_changes_its_digest(cropping, nested_cropping):
    assert compute_work_id(**_FIXED, pipeline_json=cropping) != TIFF_DIGEST_AT_81D19EC
    assert compute_work_id(**_FIXED, pipeline_json=nested_cropping) != TIFF_DIGEST_AT_81D19EC


def test_both_producers_agree_for_a_crop_pipeline(cropping, tmp_path):
    root = tmp_path / "images"
    image = root / "plate1" / "img001.tiff"
    image.parent.mkdir(parents=True)
    tifffile.imwrite(image, np.zeros((4, 4), dtype=np.uint8))
    config = make_config(pipeline_json=cropping, input_path=root)
    selected, _ = work_id_for_image(config, "plate1", image)
    worker, _ = _worker_work_identity(
        pipeline=cropping, image=image, input_root=root, dataset_name="plate1",
        image_type=config.image_type, nrows=config.nrows, ncols=config.ncols,
        bit_depth=config.bit_depth, detect_mode=config.detect_mode, layer=None,
        ext=config.ext, process_format=config.process_format,
        include_dataset_column=config.include_dataset_column,
        overlay_alpha=config.overlay_alpha, save_overlays=config.save_overlays,
        drop_originals=config.drop_originals, mode="full",
    )
    assert selected == worker
    # Equality alone passes if NEITHER producer forwards the pipeline. Rebuild
    # the same identity with every input identical but no pipeline_json: it
    # must differ, which proves the crop revision reached the producer output.
    rebuilt_without_key = compute_work_id(
        dataset="plate1",
        relative_image_path="plate1/img001.tiff",
        input_sha256=tracker.file_sha256(image),
        pipeline_fingerprint=tracker.file_sha256(cropping),
        processing_config_digest=tracker.processing_configuration_digest(config),
        mode="full",
    )
    assert selected != rebuilt_without_key
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_work_id_crop_frame.py -q --capture=fd -p no:cacheprovider`
Expected: `ImportError: cannot import name 'pipeline_uses_crop_frame'`.

- [ ] **Step 3: Write the implementation** in `_cli_failure_tracker.py`

After `RAW_DECODE_REVISION`:

```python
#: What a pipeline containing ``CropImage`` writes. Revision 1: cropped stores
#: are zero-padded into the original frame (spec 2026-10-05-pseudo-cropping
#: §6.3), so a tree resumed across the change re-derives rather than mixing
#: ROI-sized and canvas-sized stores. Folded into the work id ONLY for such
#: pipelines, inside :func:`compute_work_id` (like RAW_DECODE_REVISION), so
#: every other in-flight continuation keeps its digest.
CROP_FRAME_SEMANTICS_REVISION = 1

#: Serialized class names that produce a padding crop frame. ``ImageCropper``
#: is the retired alias (``sdk_/_class_aliases.py``). A user subclass of
#: CropImage under another name is not detected -- it then simply resumes with
#: its old work id, which is today's behaviour.
_CROP_FRAME_CLASS_NAMES = frozenset({"CropImage", "ImageCropper"})


def _json_names_crop_class(node: Any) -> bool:
    """Depth-first search for an operation whose ``class`` is a crop.

    Every serialized operation is a plain ``{"class": ..., ...}`` dict: top-level
    ops, ``OperationField`` values (``sdk_/typing_.py:285-312``) and nested
    pipelines' ``pipeline_operation``/``config`` envelopes
    (``_serializable_pipeline.py:463-470``) alike, so a dict/list walk sees all
    of them.
    """
    if isinstance(node, dict):
        if node.get("class") in _CROP_FRAME_CLASS_NAMES:
            return True
        return any(_json_names_crop_class(value) for value in node.values())
    if isinstance(node, list):
        return any(_json_names_crop_class(value) for value in node)
    return False


def pipeline_uses_crop_frame(pipeline_json: Path) -> bool:
    """Whether the serialized pipeline contains a ``CropImage`` anywhere.

    Uncached on purpose: every caller already reads the whole file for
    ``file_sha256`` per image, so one more small JSON parse is noise.
    """
    return _json_names_crop_class(
        json.loads(Path(pipeline_json).read_text(encoding="utf-8"))
    )
```

(Add `import json` if it is missing.)

`compute_work_id`: add `pipeline_json: Path | None = None` as the last keyword. Document it: "The serialized pipeline. When given and it contains `CropImage`, :data:`CROP_FRAME_SEMANTICS_REVISION` joins the payload." Before `return canonical_digest(payload)`:

```python
    if pipeline_json is not None and pipeline_uses_crop_frame(pipeline_json):
        payload["crop_frame_semantics"] = CROP_FRAME_SEMANTICS_REVISION
```

`work_id_for_image`: pass `pipeline_json=config.pipeline_json` to `compute_work_id`.
`_cli_process_single._worker_work_identity`: pass `pipeline_json=pipeline` to `compute_work_id`.

- [ ] **Step 4: Run the tests and check they pass**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_work_id_crop_frame.py tests/unit/cli/test_work_id_raw_revision.py tests/unit/cli/test_work_id_semantics_revision.py tests/unit/cli/test_store_work_identity.py -q --capture=fd -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_failure_tracker.py src/phenotypic/_cli/_cli_process_single.py tests/unit/cli/test_work_id_crop_frame.py
git add src/phenotypic/_cli/_cli_failure_tracker.py src/phenotypic/_cli/_cli_process_single.py tests/unit/cli/test_work_id_crop_frame.py
git commit -m "feat(cli): crop-only work-id revision so resumed crop runs re-derive

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD"
```

---

### Task 14: Staged GPU crop round-trip

**Files:**
- Modify: `tests/integration/cli/test_staged_store_stages.py` (new test plus any existing assertions on the `staged_run_with_provenance` store shape)
- No production change is expected. If this test fails, the bug is in Tasks 8–10. Fix it there.

**Interfaces:**
- Consumes: the `staged_run_with_provenance` fixture (`tests/integration/cli/conftest.py:303`), which runs `CropImage(left=1, right=1, top=1, bottom=1)` before `_FixedBlobDetector`.

- [ ] **Step 1: Write the test**

```python
def test_staged_crop_publishes_a_padded_store_and_replays_on_the_roi(
    staged_run_with_provenance,
) -> None:
    """Spec 2026-10-05-pseudo-cropping §6.1: Stage 1 and Stage 3 write padded
    stores; Stage 2 and Stage 3 read the ROI through load_zarr."""
    from phenotypic._cli._cli_stage2_token import load_stage2_raw
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.sdk_.ngff_ import read_phenotypic_attributes

    run = staged_run_with_provenance
    canvas = tuple(load_synth_yeast_plate().shape[:2])
    roi = (canvas[0] - 2, canvas[1] - 2)

    run.run_stage1()
    assert valid_staged_store(run.store()) is True
    assert read_phenotypic_attributes(run.store())["store_schema_version"] == 4
    run.run_stage2()
    raw = load_stage2_raw(run.output_dir, "ds", "img", run.slot)
    assert raw.shape == roi
    run.run_stage3()

    block = read_phenotypic_attributes(run.store())
    assert block["store_schema_version"] == 4
    assert block["crop_frame"] == {
        "canvas_shape": list(canvas), "offset": [1, 1],
        "roi_shape": list(roi), "padded": True,
    }
    objmap = Image.load_layer_zarr(run.store(), "objmap")
    assert objmap.shape == canvas
    assert not objmap[0].any() and not objmap[:, 0].any()
    assert objmap[1:-1, 1:-1].any()
    assert Image.load_zarr(run.store()).shape[:2] == roi
    table = run.read_measurements()
    assert set(table["Frame_OffsetRR"].to_list()) == {1}
    assert set(table["Frame_OffsetCC"].to_list()) == {1}
```

- [ ] **Step 2: Run the file**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/integration/cli/test_staged_store_stages.py -q --capture=fd -p no:cacheprovider`
Expected: the new test PASSES. Any **existing** test using `staged_run_with_provenance` that asserted ROI-sized store layers now sees canvas-sized ones. Update each such assertion to the canvas, and name each one in the commit message. Assertions about the `"original"` series stay exactly as they are.

Also run `tests/integration/cli/test_provenance_fencing.py`, which uses `staged_run_with_provenance`, and `tests/unit/cli/test_cli_provenance_durability.py`, which runs a crop pipeline through the worker (`:88`). Update their store-shape assertions the same way.

- [ ] **Step 3: Phase 4 affected surface (run once)**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli tests/integration/cli -q --capture=fd -p no:cacheprovider`
This may exceed a couple of minutes. If so, run it as a Slurm job per the `run-phenotypic-test` skill.
Expected: no new failures against `main`.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/cli/test_staged_store_stages.py
git commit -m "test(cli): staged GPU run publishes a padded store and replays on the ROI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD"
```

---

## Phase 5: GUI

### Task 15: GUI: Colony Viv cells, server-side crops, and display range on padded stores

**Files:**
- Modify: `src/phenotypic/_gui/results_viewer/_store_source.py` (`build_source_spec` return dict: add `cropOffset`)
- Modify: `src/phenotypic/_gui/results_viewer/colony_view/_grid.py` (`_build_cell`, `~:632-641`: extract `_viv_cell_payload`, add the offset)
- Modify: `src/phenotypic/_gui/_shared/tiles.py` (`_crop_store_layer_window` `~:775`; `image_display_range` `~:546-607`)
- Test: `tests/unit/gui/shared/test_tiles_crop_frame.py`, `tests/unit/gui/results_viewer/test_colony_viv_crop_frame.py`

**Interfaces:**
- Consumes: `ngff_.padded_crop_offset(block)`, `ngff_.padded_crop_window(block)` (Task 8); padded writer (Task 9).
- Produces:
  - `build_source_spec(...)["cropOffset"] == [row, col]`, which is `[0, 0]` for an unpadded store.
  - `_viv_cell_payload(*, dataset, image_file, label, centroid_rr, centroid_cc, viv_spec) -> str | None`.
- Background (spec §7): table coordinates are ROI-relative, and the store is canvas-sized. The store's own `crop_frame` is the only source of the shift, so old masters need no special case and the overlay-PNG fallback (no store) is never shifted. Plate-view contrast is dtype-based (`viv_viewer.js` `dtypeDomain`) and is unaffected.

- [ ] **Step 1: Write the failing tests**

`tests/unit/gui/shared/test_tiles_crop_frame.py`:

```python
"""Server-side crops and display range on padded stores (spec §7)."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import Image
from phenotypic._gui._shared import tiles
from phenotypic.correction import CropImage
from phenotypic.data import load_synth_yeast_plate

CROP = dict(top=7, bottom=11, left=13, right=17)


@pytest.fixture
def stores(tmp_path):
    cropped = CropImage(**CROP).apply(Image(load_synth_yeast_plate()))
    padded = cropped.save2zarr(tmp_path / "p.ome.zarr")
    plain = cropped.save2zarr(tmp_path / "u.ome.zarr", pad_on_save=False)
    return cropped, padded, plain


@pytest.mark.parametrize("layer", ["rgb", "detect_mat", "objmap"])
def test_padded_and_unpadded_stores_give_identical_crops(stores, layer):
    """Every colony, including those touching the ROI edge: the window is
    computed against the ROI's extent, so the zero margin is never read."""
    cropped, padded, plain = stores
    info = cropped.objects.info(include_metadata=False)
    for _, row in info.iterrows():
        args = (layer, float(row["Bbox_CenterRR"]), float(row["Bbox_CenterCC"]), 48)
        kwargs = dict(
            dim_alpha=0.5,
            bbox=(row["Bbox_MinRR"], row["Bbox_MaxRR"], row["Bbox_MinCC"], row["Bbox_MaxCC"]),
            contours=int(row["Object_Label"]) if layer == "rgb" else None,
        )
        assert tiles.crop_store_rgb(padded, *args, **kwargs) == tiles.crop_store_rgb(plain, *args, **kwargs)


def test_display_range_ignores_the_zero_margin(tmp_path):
    """16-bit plates: a padded store's zero margin must not drag lo to 0."""
    rng = np.random.default_rng(4)
    arr = rng.integers(17_000, 48_000, size=(600, 800, 3), dtype=np.uint16)
    cropped = CropImage(top=50, bottom=50, left=60, right=60).apply(Image(arr))
    lo_padded, hi_padded = tiles.image_display_range(cropped.save2zarr(tmp_path / "p.ome.zarr"), "rgb")
    lo_plain, hi_plain = tiles.image_display_range(
        cropped.save2zarr(tmp_path / "u.ome.zarr", pad_on_save=False), "rgb"
    )
    assert lo_padded > 10_000
    # Different pyramids (canvas vs ROI) downsample differently, so equal-ish, not equal.
    assert abs(lo_padded - lo_plain) < 2_000 and abs(hi_padded - hi_plain) < 2_000


def test_display_range_with_odd_offsets_on_a_deep_pyramid(tmp_path):
    """Odd offsets scale to non-integer level coordinates; inward rounding keeps
    straddling (part-zero) mean pixels out of the range."""
    arr = np.full((1400, 1600, 3), 30_000, dtype=np.uint16)
    cropped = CropImage(top=51, left=61, bottom=37, right=43).apply(Image(arr))
    store = cropped.save2zarr(tmp_path / "p.ome.zarr")
    from phenotypic.sdk_.ngff_ import read_phenotypic_attributes

    assert read_phenotypic_attributes(store)["pyramid"]["levels"] >= 3
    assert tiles.image_display_range(store, "rgb")[0] == 30_000


@pytest.mark.parametrize(
    ("offset", "roi", "canvas", "level", "expected"),
    [
        ((650, 650), (100, 100), (2000, 2000), (250, 250), (82, 93, 82, 93)),  # inward
        ((50, 60), (500, 680), (600, 800), (300, 400), (25, 275, 30, 370)),  # exact
        ((3, 3), (2, 2), (16, 16), (2, 2), (0, 1, 0, 1)),  # inward empty -> outward
    ],
)
def test_roi_window_at_level(offset, roi, canvas, level, expected):
    assert tiles._roi_window_at_level(offset, roi, canvas, level) == expected
```

(The constant-valued plate makes the expected `lo` exact: any zero-margin pixel that leaks in lowers it. Check the third case's arithmetic by hand when implementing: start 3·2/16 = 0.375, so inward is `[1, 0)`, which is empty, and outward is `[0, 1)`.)

`tests/unit/gui/results_viewer/test_colony_viv_crop_frame.py`:

```python
"""Colony Viv cells target store pixels, so they carry the crop offset (spec §7)."""

from __future__ import annotations

import json

from phenotypic import Image
from phenotypic._gui.results_viewer._store_source import build_source_spec
from phenotypic._gui.results_viewer.colony_view._grid import _viv_cell_payload
from phenotypic.correction import CropImage
from phenotypic.data import load_synth_yeast_plate


def test_source_spec_carries_the_store_crop_offset(tmp_path):
    cropped = CropImage(top=7, left=13).apply(Image(load_synth_yeast_plate()))
    assert build_source_spec(cropped.save2zarr(tmp_path / "p.ome.zarr"), "/zarr/p")["cropOffset"] == [7, 13]
    assert build_source_spec(
        cropped.save2zarr(tmp_path / "u.ome.zarr", pad_on_save=False), "/zarr/u"
    )["cropOffset"] == [0, 0]


def test_colony_cell_centroid_is_shifted_into_the_canvas():
    payload = json.loads(
        _viv_cell_payload(
            dataset="ds", image_file="img", label=3,
            centroid_rr=40.5, centroid_cc=60.25,
            viv_spec={"storeUrl": "/zarr/x", "cropOffset": [7, 13]},
        )
    )
    assert (payload["centroidRr"], payload["centroidCc"]) == (47.5, 73.25)


def test_colony_cell_without_offset_is_unchanged():
    payload = json.loads(
        _viv_cell_payload(
            dataset="ds", image_file="img", label=3,
            centroid_rr=40.5, centroid_cc=60.25, viv_spec={"storeUrl": "/zarr/x"},
        )
    )
    assert (payload["centroidRr"], payload["centroidCc"]) == (40.5, 60.25)


def test_no_payload_without_spec_or_centroid():
    assert _viv_cell_payload(dataset="d", image_file="i", label=1,
                             centroid_rr=None, centroid_cc=1.0, viv_spec={}) is None
    assert _viv_cell_payload(dataset="d", image_file="i", label=1,
                             centroid_rr=1.0, centroid_cc=1.0, viv_spec=None) is None
```

- [ ] **Step 2: Run the tests and check they fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/shared/test_tiles_crop_frame.py tests/unit/gui/results_viewer/test_colony_viv_crop_frame.py -q --capture=fd -p no:cacheprovider`
Expected:
- The crop tests FAIL: the padded crop is shifted by `(7, 13)`.
- The display-range test FAILS with `lo_padded == 0`.
- The Viv tests FAIL with an `ImportError` (`_viv_cell_payload`) or `KeyError` (`cropOffset`).

- [ ] **Step 3: Write the implementation**

`_store_source.py`, in `build_source_spec`'s returned dict, add:

```python
        # Table coordinates are ROI-relative; a padded store's pixels sit at
        # this offset inside the canvas (spec 2026-10-05-pseudo-cropping §7).
        "cropOffset": list(ngff_.padded_crop_offset(block)),
```

`colony_view/_grid.py`: extract the payload from `_build_cell` into a module-level helper and apply the offset:

```python
def _viv_cell_payload(
    *,
    dataset: str,
    image_file: str,
    label: int,
    centroid_rr: float | None,
    centroid_cc: float | None,
    viv_spec: Mapping[str, Any] | None,
) -> str | None:
    """Serialise one Colony cell's Viv payload, or ``None`` if it has none.

    ``viv_viewer.js`` treats ``centroidRr/Cc`` as STORE pixel coordinates; the
    table's ``Bbox_Center*`` are ROI-relative, so a padded store's
    ``cropOffset`` (from ``build_source_spec``) is added here.
    """
    if viv_spec is None or centroid_rr is None or centroid_cc is None:
        return None
    d_row, d_col = viv_spec.get("cropOffset", (0, 0))
    return json.dumps(
        {
            "id": f"{dataset}:{image_file}:{label}",
            "centroidRr": float(centroid_rr) + d_row,
            "centroidCc": float(centroid_cc) + d_col,
            "spec": dict(viv_spec),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
```

and in `_build_cell` replace the `if viv_spec is not None and centroid_rr ...: outer_props["data-colony-viv-cell"] = json.dumps(...)` block with:

```python
    payload = _viv_cell_payload(
        dataset=dataset,
        image_file=image_file,
        label=label,
        centroid_rr=centroid_rr,
        centroid_cc=centroid_cc,
        viv_spec=viv_spec,
    )
    if payload is not None:
        outer_props["data-colony-viv-cell"] = payload
```

`tiles.py`, `_crop_store_layer_window`: compute the window in ROI coordinates against the ROI's extent, then shift the **read** into the canvas. Replace

```python
    level0 = _level_shape(store_path, member, 0)
    src_height, src_width = level0[-2:]
    window = _crop_window(center_rr, center_cc, size, src_width, src_height)
    arr: np.ndarray | None = None
    read_window: tuple[int, int, int, int] | None = None
    if window.has_area:
        read_window = (window.top, window.bottom, window.left, window.right)
```

with

```python
    # Spec 2026-10-05-pseudo-cropping §7: table coordinates are ROI-relative.
    # Against a padded store, size the window by the ROI (not the canvas) and
    # shift only the READ by the store's own offset. The crop is then
    # byte-identical to one from an unpadded store, edges included, and the
    # zero margin never enters a crop or its per-window normalisation.
    padded = ngff_.padded_crop_window(block)
    if padded is None:
        level0 = _level_shape(store_path, member, 0)
        src_height, src_width = level0[-2:]
        d_row, d_col = 0, 0
    else:
        (src_height, src_width), (d_row, d_col) = padded[2], padded[1]
    window = _crop_window(center_rr, center_cc, size, src_width, src_height)
    arr: np.ndarray | None = None
    read_window: tuple[int, int, int, int] | None = None
    if window.has_area:
        read_window = (
            window.top + d_row,
            window.bottom + d_row,
            window.left + d_col,
            window.right + d_col,
        )
```

(`_read_objmap_window(store_path, read_window)` then reads the same canvas window. If `ngff_` isn't in scope in that function, import it locally: `from phenotypic.sdk_ import ngff_`.)

`tiles.py`, `image_display_range`: replace `smallest = _read_store_level(store_path, layer, levels - 1)` with:

```python
    # A padded store's zero margin would drag `lo` to 0 and wash out every
    # 16-bit crop (spec 2026-10-05-pseudo-cropping §7): read only the ROI's
    # window of the smallest level, scaled by that level's size.
    padded = ngff_.padded_crop_window(block)
    window = None
    if padded is not None:
        (canvas_h, canvas_w), (row, col), (roi_h, roi_w) = padded
        member = _store_member_path(block, store_path, layer)
        level_h, level_w = _level_shape(store_path, member, levels - 1)[-2:]
        window = _roi_window_at_level(
            (row, col), (roi_h, roi_w), (canvas_h, canvas_w), (level_h, level_w)
        )
    smallest = _read_store_level(store_path, layer, levels - 1, window=window)
```

and add the helper beside `image_display_range`:

```python
def _roi_window_at_level(
    offset: tuple[int, int],
    roi_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
    level_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """``(top, bottom, left, right)`` of the ROI at one pyramid level, rounded INWARD.

    The pyramid is a 2x block mean (``ngff_.py:140-142``), so a level pixel
    straddling the ROI edge averages ROI pixels with the zero margin (e.g. the
    spimager offset 650 at level 3 is 81.25 -- pixel 81 is 2/8 zero). Inward
    rounding (ceil start, floor end) keeps only pixels wholly inside the ROI;
    if that leaves nothing, fall back to outward rounding.
    """
    bounds = []
    for start, extent, canvas, level in zip(offset, roi_shape, canvas_shape, level_shape):
        inner_lo = -(-(start * level) // canvas)
        inner_hi = ((start + extent) * level) // canvas
        if inner_hi > inner_lo:
            bounds.append((inner_lo, inner_hi))
        else:
            outer_lo = (start * level) // canvas
            outer_hi = max(outer_lo + 1, -(-((start + extent) * level) // canvas))
            bounds.append((outer_lo, outer_hi))
    (top, bottom), (left, right) = bounds
    return top, bottom, left, right
```

- [ ] **Step 4: Run the tests and check they pass**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/shared tests/unit/gui/results_viewer tests/gui/_shared tests/gui/results_viewer -q --capture=fd -p no:cacheprovider`
Expected: all PASS. If `test_store_source.py::test_the_spec_is_a_valid_facade_source_spec` pins the exact key set, add `cropOffset` to it. Don't loosen it.

- [ ] **Step 5: Ledger check.** `cropOffset` is not chrome, so no `FEATURES.md`/`WORKFLOWS.md` row is required. Run `grep -n "build_source_spec" src/phenotypic/_gui/FEATURES.md`. If that row's prose enumerates the spec's keys, add `cropOffset` there too (gui-tutorial-capture skill).

- [ ] **Step 6: Lint and commit**

Lint and stage the three source files and the two new test files by explicit path. Commit message:

```
feat(gui): colony views, crops and display range honour padded crop stores

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD
```

---

## Phase 6: Docs and final verification

### Task 16: Documentation

**Files:**
- Modify: `docs/source/how_to/pages/zarr_storage.md`: a "Cropped images" section covering padded layers, `crop_frame`, v4, `pad_on_save`, `imread` vs `load_zarr`.
- Modify: `docs/source/how_to/notebooks/crop_and_pad.ipynb`: add **one markdown cell** after the first crop example explaining the frame, `Frame_Offset*`, and `pad_on_save`. Don't add code cells and don't re-execute the notebook.
- Modify: `src/phenotypic/_core/CLAUDE.md`: a "Crop frame" section covering the attributes, the composition rule, the `_valid_crop_frame()` read rule, the PadImage/rotate rules, and the apply-wrapper carry-over.
- Modify: `src/phenotypic/schema/CLAUDE.md`: the enum-module count, 32 → 33 (verify with `ls src/phenotypic/schema/_*.py`).
- Modify: root `CLAUDE.md`: one bullet under **Gotchas**, "**Cropped images save padded.**", covering what a padded store is, v4 on padded stores only (with the v4 ⇔ `padded` read invariant), `pad_on_save`, `Frame_Offset*` (including null for re-measured pre-change crop stores), that `load_layer_zarr`/`imread` return the canvas, and that builder previews stay ROI-sized.
- Modify: `src/phenotypic/_gui/CLAUDE.md`, "Pixel paths": table coordinates are ROI-relative, so any new code placing them onto store pixels must use `ngff_.padded_crop_offset`/`padded_crop_window`. The Colony grid and `tiles.py` are the existing examples.
- Modify: `.claude/skills/working-with-ome-zarr/SKILL.md` **if it exists in the repo** (`ls .claude/skills`). Add the `crop_frame` attribute and the {3,4} readable set.

- [ ] **Step 1: Write the doc changes.** Each statement must match the implemented code: name the real function, attribute or key. For "Bbox + offset" text, cite `FRAME.OFFSET_RR`.

- [ ] **Step 2: Doc tests**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/docs tests/unit/test_docs_foundation.py tests/unit/test_docs_myst_fences.py -q --capture=fd -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 3: Docstring doctests for touched modules**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest --doctest-modules src/phenotypic/_core/_crop_frame.py src/phenotypic/correction/_image_cropper.py src/phenotypic/correction/_image_padder.py src/phenotypic/schema/_frame.py -q --capture=fd -p no:cacheprovider`
Expected: PASS (`+SKIP` lines excepted).

- [ ] **Step 4: Rendering check.** Docs build as a Slurm job, per the global rules: `sphinx-build -j "$SLURM_CPUS_PER_TASK" -D nbsphinx_execute=never docs/source <out>`. Then read the generated HTML for `zarr_storage` and `crop_and_pad`. An exit code of 0 does not prove the pages rendered.

- [ ] **Step 5: Commit**

```bash
git add docs/source/how_to/pages/zarr_storage.md docs/source/how_to/notebooks/crop_and_pad.ipynb src/phenotypic/_core/CLAUDE.md src/phenotypic/schema/CLAUDE.md src/phenotypic/_gui/CLAUDE.md CLAUDE.md
git commit -m "docs: crop frames, padded stores and Frame_Offset columns

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016S1bCicbf2LpbH5s5S65AD"
```

### Task 17: Final verification

- [ ] **Step 1:** `QT_QPA_PLATFORM=offscreen uv run --no-project --with numpy python docs/superpowers/logic_validation_scripts/2026-10-05-pseudo-cropping/crop_frame_invariants.py`. Expected: 4 × `ok`, exit 0.
- [ ] **Step 2:** `uv run mypy src/phenotypic/_core/_crop_frame.py src/phenotypic/_core/_image_parts src/phenotypic/sdk_/ngff_.py src/phenotypic/_cli/_cli_failure_tracker.py src/phenotypic/_cli/_cli_process_only.py`. Expect no new errors against `main` (compare with the same command on `main`).
- [ ] **Step 3:** `uv run ruff check <every file changed on the branch>`. List them with `git diff --name-only main...HEAD -- '*.py'`.
- [ ] **Step 4:** Guards for lazy imports: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/ci/test_startup_imports.py tests/unit/ci/test_deferred_imports.py tests/unit/gui/shell/test_hub_startup_imports.py -q --capture=fd -p no:cacheprovider`.
- [ ] **Step 5: Full sharded regression, once.** Use the `run-phenotypic-test` skill and the `slurm-job` skill. Run it in a worktree detached at the branch HEAD SHA, sharded across a Slurm array, with an `afterany` cleanup finalizer. Compare the results with the latest `main` baseline (memory: `phenotypic-regression-baseline`). Run every new failure in isolation before attributing it, and record the result.
