# Source Timeline View — Phase 1: Shared Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the source-agnostic engine (`gui/_shared/timeline/`) that both the Browse and Results timeline surfaces will consume — a matrix model, a thumbnail-serving Flask route factory, and a virtualization-ready grid renderer.

**Architecture:** Pure helpers + one Flask route factory, mirroring how `gui/_shared/tiles.py` is shared across the colony-view and QC tabs. `build_matrix` turns flat records into an ordered `(row × time)` matrix; `register_thumbnail_route` serves cached, downscaled thumbnails via a per-surface resolver; `build_timeline_grid` renders a CSS grid of **placeholder** cells (no `<img>` yet) plus a row-major key list. The browser-side virtualization JS and the two surfaces are later phases that depend on this one.

**Tech Stack:** Python 3, polars (not needed here — records are plain dicts), Pillow (PIL) for downscaling, Dash/`dash.html` for the grid component, Flask for the route, pytest.

## Global Constraints

- **`uv` is the sole runner.** Every command is `uv run …`; never bare `python`/`pip`.
- **Do NOT import `dash` from `gui/_config.py`** — it must stay cheap to import from blueprint/test code. New `_config.py` additions are pure-Python (ints, tuples, a pure function).
- **Single-source constants:** new shared constants live in `gui/_config.py` (Python identifiers); no re-spelled literals across files. Tile-size stepper constants mirror the existing `COLONY_TILE_SIZE_*` values.
- **FEATURES.md gate:** the `gui-checks` `features-md-gate` job rejects any PR touching `src/phenotypic/gui/` without modifying `src/phenotypic/gui/FEATURES.md`. Phase 1 has no user-visible affordance, so it adds one `🧪 internal` row (the engine is tested but not user-facing) with a real `Test ref`. **Do not use `🚧 in progress`** — the merge gate rejects any row left in that status.
- **Self-invalidating + atomic thumbnail cache** (spec §15.6): cache filename embeds the source `st_mtime_ns`; writes are `tempfile` + `os.replace`.
- **Time ordering is coerce-at-sort** (spec §15.3): try numeric → datetime → lexical; never trust the raw stored dtype.
- **Focus-and-navigate model (spec §16):** the engine feeds a focus-navigate controller (Phase 2), **not** a scroll-virtualized grid. Phase 1 impact is small and confined to two tasks: the constants task adds `TIMELINE_FOCUS_MARGIN` and **drops** the scroll-era `TIMELINE_WINDOW_MARGIN_SCREENS` (§16.7); the grid task emits per-cell `data-row-index`/`data-col-index` so the JS can address cells by grid coordinate, and marks the ⤢ button hover-revealable (§16.8). `build_matrix`, the thumbnail route, and the placeholder-grid contract are otherwise unchanged.
- **Test collection (decided 2026-06-18):** the timeline unit tests live under `tests/gui/…`, and **`tests/gui` is added to `pyproject.toml` `testpaths`** (Task 0) so CI's default `pytest` lane collects them — `tests/gui` was previously orphaned from auto-CI (memory: *gui-test-collection-and-route-fixtures*). Dash Flask-route tests must set `app.layout` (else `NoLayoutException` → 500). This is a repo-wide collection change: Task 0 runs the whole existing `tests/gui` tree first and must leave it green before any timeline task builds on it.

---

### Task 0: Add `tests/gui` to pytest `testpaths` (pre-flight, repo-wide)

**Files:**
- Modify: `pyproject.toml` (the `[tool.pytest.ini_options]` `testpaths` list)

**Why:** Every timeline unit test lands under `tests/gui/…`, but CI's default
`pytest` lane honors `testpaths`, which today is
`["tests/unit", "tests/smoke", "tests/integration"]` — so `tests/gui` (and the
existing colony pure-tests there) are never auto-collected. The user decided to
make CI collect `tests/gui` rather than relocate the tests
(memory: *gui-test-collection-and-route-fixtures*).

- [ ] **Step 1: Confirm the existing tree is green first.** Before changing
  collection, run the whole current `tests/gui` tree to establish a clean baseline:
  `uv run pytest tests/gui -q`. It collects cleanly today (350/351, 1 deselected).
  If anything fails, **STOP and report** — a pre-existing failure must be triaged
  (fix or `-m "not <marker>"`/`xfail` with a note) before `tests/gui` joins the
  default lane, or it will turn CI red for reasons unrelated to this feature.

- [ ] **Step 2: Add `tests/gui` to `testpaths`.** Edit `pyproject.toml`:

  ```toml
  testpaths = ["tests/unit", "tests/smoke", "tests/integration", "tests/gui"]
  ```

- [ ] **Step 3: Verify default collection now includes the tree.** Run
  `uv run pytest --collect-only -q | tail -5` (no path arg, so it uses `testpaths`)
  and confirm `tests/gui/...` items now appear; then `uv run pytest tests/gui -q`
  stays green.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "test(gui): collect tests/gui in the default pytest lane"
```

> **Note:** `tests/gui/_shared/timeline/` does not exist yet — Task 1 creates it.
> Task 0 only widens collection and proves the *existing* `tests/gui` tree is green;
> the new timeline subtree is added by Tasks 1–8 and is collected automatically
> thereafter.

---

### Task 1: Timeline constants + bucket snapping

**Files:**
- Modify: `src/phenotypic/gui/_config.py` (append a "Timeline view" block + extend `__all__`)
- Create: `tests/gui/_shared/timeline/__init__.py` (empty)
- Test: `tests/gui/_shared/timeline/test_constants.py`

**Interfaces:**
- Consumes: nothing (foundational).
- Produces: `THUMB_SIZE_BUCKETS: tuple[int, ...]`, `TIMELINE_TILE_SIZE_DEFAULT/STEP/MIN/MAX: int`, `TIMELINE_GRID_GAP_PX: int`, `TIMELINE_FOCUS_MARGIN: int` (focus-navigate mount-ring distance in cells, §16.3 — replaces the scroll-era `TIMELINE_WINDOW_MARGIN_SCREENS`), `TIMELINE_MOUNT_CAP: int`, `TIMELINE_WARM_CONCURRENCY: int`, `BROWSE_THUMB_URL_SEGMENT: str`, `VIEWER_THUMB_URL_SEGMENT: str`, and `snap_thumb_bucket(size: int) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/_shared/timeline/__init__.py` (empty file), then `tests/gui/_shared/timeline/test_constants.py`:

```python
"""Timeline-view shared constants + thumbnail bucket snapping."""
from __future__ import annotations

import pytest

from phenotypic.gui._config import (
    THUMB_SIZE_BUCKETS,
    TIMELINE_FOCUS_MARGIN,
    TIMELINE_TILE_SIZE_DEFAULT,
    TIMELINE_TILE_SIZE_MAX,
    TIMELINE_TILE_SIZE_MIN,
    TIMELINE_TILE_SIZE_STEP,
    snap_thumb_bucket,
)


def test_focus_margin_is_a_positive_int() -> None:
    # Focus-navigate mount-ring distance in cells (spec §16.3).
    assert isinstance(TIMELINE_FOCUS_MARGIN, int)
    assert TIMELINE_FOCUS_MARGIN >= 1


def test_buckets_are_sorted_ascending_ints() -> None:
    assert THUMB_SIZE_BUCKETS == tuple(sorted(THUMB_SIZE_BUCKETS))
    assert all(isinstance(b, int) for b in THUMB_SIZE_BUCKETS)
    assert THUMB_SIZE_BUCKETS[0] == 64
    assert THUMB_SIZE_BUCKETS[-1] == 256


def test_tile_size_stepper_bounds_mirror_colony() -> None:
    # Mirrors COLONY_TILE_SIZE_* (default 150, step 16, range 64..400).
    assert (
        TIMELINE_TILE_SIZE_DEFAULT,
        TIMELINE_TILE_SIZE_STEP,
        TIMELINE_TILE_SIZE_MIN,
        TIMELINE_TILE_SIZE_MAX,
    ) == (150, 16, 64, 400)


@pytest.mark.parametrize(
    "requested,expected",
    [
        (10, 64),    # below min → smallest bucket
        (64, 64),    # exact
        (65, 96),    # snap up
        (100, 128),  # snap up
        (192, 192),  # exact
        (300, 256),  # above max → largest bucket
    ],
)
def test_snap_thumb_bucket_snaps_up_and_clamps(requested: int, expected: int) -> None:
    assert snap_thumb_bucket(requested) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/_shared/timeline/test_constants.py -v`
Expected: FAIL with `ImportError: cannot import name 'THUMB_SIZE_BUCKETS'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/phenotypic/gui/_config.py` (after the existing colony-tile-size block; keep the file's `from __future__ import annotations` and style). Add a section:

```python
# ---------------------------------------------------------------------------
# Timeline view (Browse + Results) — shared engine constants
# ---------------------------------------------------------------------------

#: Server thumbnail size buckets (longest-edge px). The requested display
#: size snaps UP to the nearest bucket so per-tile bytes track display size
#: without ever upscaling. Bounded set → bounded thumbnail-cache key space.
THUMB_SIZE_BUCKETS: tuple[int, ...] = (64, 96, 128, 192, 256)

#: Timeline tile-size stepper — mirrors COLONY_TILE_SIZE_* so the two grids
#: feel identical. Display size (CSS) steps in this range; the fetched
#: thumbnail snaps to THUMB_SIZE_BUCKETS.
TIMELINE_TILE_SIZE_DEFAULT: int = 150
TIMELINE_TILE_SIZE_STEP: int = 16
TIMELINE_TILE_SIZE_MIN: int = 64
TIMELINE_TILE_SIZE_MAX: int = 400

#: CSS gap between timeline tiles, in pixels.
TIMELINE_GRID_GAP_PX: int = 8

#: Focus-and-navigate mount ring (spec §16.3): mount the focused cell's visible
#: window PLUS this many cells in every direction (off-screen pre-mount for smooth
#: stepping); offload cells farther than this. Replaces the scroll-era
#: TIMELINE_WINDOW_MARGIN_SCREENS.
TIMELINE_FOCUS_MARGIN: int = 2

#: Hard LRU ceiling on mounted <img> elements regardless of viewport/margin.
TIMELINE_MOUNT_CAP: int = 400

#: Background-warm concurrent fetches. Low because Browse warm triggers RAW
#: normalize_to_png; raise for the (cheaper) Results overlay path if needed.
TIMELINE_WARM_CONCURRENCY: int = 2

#: URL segments for the per-surface thumbnail routes (mounted by
#: register_thumbnail_route). Browse lives on the browse server; the viewer
#: segment is distinct so the two blueprints never collide on one server.
BROWSE_THUMB_URL_SEGMENT: str = "thumb"
VIEWER_THUMB_URL_SEGMENT: str = "timeline-thumb"


def snap_thumb_bucket(size: int) -> int:
    """Return the smallest :data:`THUMB_SIZE_BUCKETS` value >= ``size``.

    Sizes at or below the smallest bucket return the smallest bucket; sizes
    at or above the largest bucket return the largest (never upscale, never
    exceed the cap).

    Args:
        size: Requested display size, in pixels.

    Returns:
        The chosen bucket edge length, in pixels.
    """
    for bucket in THUMB_SIZE_BUCKETS:
        if size <= bucket:
            return bucket
    return THUMB_SIZE_BUCKETS[-1]
```

Then add each new name to the module's `__all__` list.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/_shared/timeline/test_constants.py -v`
Expected: PASS (3 plain + 6 parametrized cases = 9 passing).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_config.py tests/gui/_shared/timeline/__init__.py tests/gui/_shared/timeline/test_constants.py
git commit -m "feat(gui-timeline): shared constants + thumbnail bucket snapping"
```

---

### Task 2: Natural sort key (`_matrix._natural_sort_key`)

**Files:**
- Create: `src/phenotypic/gui/_shared/timeline/__init__.py` (empty for now; populated in Task 8)
- Create: `src/phenotypic/gui/_shared/timeline/_matrix.py`
- Test: `tests/gui/_shared/timeline/test_matrix.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `_natural_sort_key(value: object) -> tuple[int, object]` — a sort key trying numeric (rank 0), then ISO datetime (rank 1), then lexical (rank 2). Used to order both axes so a String-dtype `"1","2","10"` sorts `1<2<10` (spec §15.3).

- [ ] **Step 1: Write the failing test**

Create `tests/gui/_shared/timeline/test_matrix.py`:

```python
"""Pure matrix-model helpers for the timeline engine."""
from __future__ import annotations

from phenotypic.gui._shared.timeline._matrix import _natural_sort_key


def test_numeric_strings_sort_numerically_not_lexically() -> None:
    values = ["10", "2", "1"]
    assert sorted(values, key=_natural_sort_key) == ["1", "2", "10"]


def test_numerics_sort_before_plain_strings() -> None:
    values = ["b", "10", "a", "2"]
    assert sorted(values, key=_natural_sort_key) == ["2", "10", "a", "b"]


def test_iso_datetimes_sort_chronologically() -> None:
    values = ["2024-01-10", "2024-01-02", "2024-01-01"]
    assert sorted(values, key=_natural_sort_key) == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-10",
    ]


def test_non_finite_floats_fall_through_to_lexical() -> None:
    # nan/inf must NOT enter the numeric bucket (rank 0): nan breaks sort
    # determinism and inf has no axis position. They fall to lexical (rank 2).
    assert _natural_sort_key("nan")[0] == 2
    assert _natural_sort_key("inf")[0] == 2
    assert _natural_sort_key("-inf")[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/_shared/timeline/test_matrix.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'phenotypic.gui._shared.timeline'`.

- [ ] **Step 3: Write minimal implementation**

Create the empty package init:

```python
# src/phenotypic/gui/_shared/timeline/__init__.py
"""Source-agnostic timeline-view engine (matrix model, thumbnail route, grid)."""
```

Create `src/phenotypic/gui/_shared/timeline/_matrix.py`:

```python
"""Pure matrix model for the timeline view.

Turns flat ``records`` (each a mapping with a row value, a time value, and an
opaque cell reference) into an ordered ``(row × time)`` matrix. Both axes sort
via :func:`_natural_sort_key`, which coerces values at sort time (numeric →
datetime → lexical) because the stored dtype is unreliable — ``join_metadata``
casts join-key columns to ``pl.String``, so a conceptually-numeric
``Metadata_Time`` arrives as strings (spec §15.3).
"""
from __future__ import annotations

import math
from datetime import datetime


def _natural_sort_key(value: object) -> tuple[int, object]:
    """Return a sort key that orders numerics, then datetimes, then strings.

    Coercion is attempted on ``str(value)`` so String-dtype numerics sort
    numerically. The leading rank int keeps the three families segregated and
    comparable (Python compares the rank first, the coerced value only within
    a rank, where the types match).

    Args:
        value: Any axis value (typically a ``str``).

    Returns:
        ``(0, float)`` for numerics, ``(1, datetime)`` for ISO datetimes,
        ``(2, str)`` otherwise.
    """
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        pass
    else:
        # Reject non-finite floats (nan/inf): nan breaks sort determinism
        # (all comparisons False) and inf has no meaningful axis position.
        # Fall through to datetime/lexical instead.
        if math.isfinite(number):
            return (0, number)
    try:
        return (1, datetime.fromisoformat(text))
    except ValueError:
        return (2, text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/_shared/timeline/test_matrix.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_shared/timeline/__init__.py src/phenotypic/gui/_shared/timeline/_matrix.py tests/gui/_shared/timeline/test_matrix.py
git commit -m "feat(gui-timeline): natural (coerce-at-sort) axis ordering"
```

---

### Task 3: `build_matrix` + dataclasses

**Files:**
- Modify: `src/phenotypic/gui/_shared/timeline/_matrix.py`
- Test: `tests/gui/_shared/timeline/test_matrix.py` (append)

**Interfaces:**
- Consumes: `_natural_sort_key` (Task 2).
- Produces:
  - `TimelineCell(row_value: str, time_value: str, representative: object, members: tuple[object, ...], count: int)` — frozen dataclass.
  - `TimelineMatrix(columns: list[str], rows: list[str], cells: dict[tuple[str, str], TimelineCell])` — frozen dataclass.
  - `build_matrix(records: Iterable[Mapping[str, object]], *, row_key: str = "row_value", time_key: str = "time_value", ref_key: str = "cell_ref") -> TimelineMatrix`. Row/time values are stringified via `str()`; the representative is the member with the smallest `str(cell_ref)` (deterministic); empty `(row, time)` pairs are simply absent from `cells`.

- [ ] **Step 1: Write the failing test**

Append to `tests/gui/_shared/timeline/test_matrix.py`:

```python
from phenotypic.gui._shared.timeline._matrix import TimelineMatrix, build_matrix


def _records() -> list[dict[str, object]]:
    return [
        {"row_value": "plateA", "time_value": "1", "cell_ref": "a1"},
        {"row_value": "plateA", "time_value": "10", "cell_ref": "a10"},
        {"row_value": "plateA", "time_value": "2", "cell_ref": "a2"},
        {"row_value": "plateB", "time_value": "1", "cell_ref": "b1"},
        {"row_value": "plateA", "time_value": "1", "cell_ref": "a1b"},  # collide
    ]


def test_build_matrix_orders_axes_numerically() -> None:
    m = build_matrix(_records())
    assert isinstance(m, TimelineMatrix)
    assert m.columns == ["1", "2", "10"]
    assert m.rows == ["plateA", "plateB"]


def test_build_matrix_aggregates_collisions_with_deterministic_representative() -> None:
    m = build_matrix(_records())
    cell = m.cells[("plateA", "1")]
    assert cell.count == 2
    assert set(cell.members) == {"a1", "a1b"}
    assert cell.representative == "a1"  # smallest str(cell_ref)


def test_build_matrix_omits_empty_cells() -> None:
    m = build_matrix(_records())
    assert ("plateB", "2") not in m.cells
    assert ("plateB", "1") in m.cells
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/_shared/timeline/test_matrix.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_matrix'`.

- [ ] **Step 3: Write minimal implementation**

Add to the top of `_matrix.py` (imports) and below `_natural_sort_key`:

```python
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class TimelineCell:
    """One ``(row, time)`` cell of the matrix.

    Attributes:
        row_value: The cell's row (group) value, stringified.
        time_value: The cell's time (column) value, stringified.
        representative: The opaque cell-ref rendered as the tile (smallest
            ``str(cell_ref)`` among ``members`` — deterministic).
        members: Every cell-ref that maps to this cell (length == ``count``).
        count: Number of members (drives the ``N=k`` badge).
    """

    row_value: str
    time_value: str
    representative: object
    members: tuple[object, ...]
    count: int


@dataclass(frozen=True)
class TimelineMatrix:
    """An ordered ``(row × time)`` matrix of cells.

    Attributes:
        columns: Time values, naturally ordered (the X axis).
        rows: Row/group values, naturally ordered (the Y axis).
        cells: ``(row_value, time_value) -> TimelineCell``. Missing pairs are
            absent (empty cells render as placeholders downstream).
    """

    columns: list[str]
    rows: list[str]
    cells: dict[tuple[str, str], TimelineCell]


def build_matrix(
    records: Iterable[Mapping[str, object]],
    *,
    row_key: str = "row_value",
    time_key: str = "time_value",
    ref_key: str = "cell_ref",
) -> TimelineMatrix:
    """Build a :class:`TimelineMatrix` from flat records.

    Args:
        records: Iterable of mappings, each carrying a row value, a time
            value, and an opaque cell reference under the given keys.
        row_key: Mapping key for the row (group) value.
        time_key: Mapping key for the time (column) value.
        ref_key: Mapping key for the opaque per-image cell reference.

    Returns:
        A matrix with naturally-ordered ``columns``/``rows`` and a
        ``cells`` map whose representative is the smallest ``str(cell_ref)``
        in each cell.
    """
    grouped: dict[tuple[str, str], list[object]] = {}
    row_set: set[str] = set()
    col_set: set[str] = set()
    for record in records:
        rv = str(record[row_key])
        tv = str(record[time_key])
        row_set.add(rv)
        col_set.add(tv)
        grouped.setdefault((rv, tv), []).append(record[ref_key])

    cells: dict[tuple[str, str], TimelineCell] = {}
    for (rv, tv), refs in grouped.items():
        ordered = tuple(sorted(refs, key=lambda r: str(r)))
        cells[(rv, tv)] = TimelineCell(
            row_value=rv,
            time_value=tv,
            representative=ordered[0],
            members=ordered,
            count=len(ordered),
        )

    return TimelineMatrix(
        columns=sorted(col_set, key=_natural_sort_key),
        rows=sorted(row_set, key=_natural_sort_key),
        cells=cells,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/_shared/timeline/test_matrix.py -v`
Expected: PASS (7 tests total in the file).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_shared/timeline/_matrix.py tests/gui/_shared/timeline/test_matrix.py
git commit -m "feat(gui-timeline): build_matrix with representative + empty-cell handling"
```

---

### Task 4: Thumbnail cache naming (`thumb_cache_name`) + `ThumbUnavailable`

**Files:**
- Create: `src/phenotypic/gui/_shared/timeline/_thumbnail.py`
- Test: `tests/gui/_shared/timeline/test_thumbnail.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class ThumbUnavailable(RuntimeError)` — raised by a resolver when a source can't be decoded (e.g. RAW on Windows); the route maps it to 422.
  - `thumb_cache_name(identity: str, bucket: int, mtime_ns: int) -> str` — a flat, filesystem-safe, **self-invalidating** filename (`<b64url(identity)>_<bucket>_<mtime_ns>.png`).

- [ ] **Step 1: Write the failing test**

Create `tests/gui/_shared/timeline/test_thumbnail.py`:

```python
"""Thumbnail cache naming, downscaling, and the route factory."""
from __future__ import annotations

import re

from phenotypic.gui._shared.timeline._thumbnail import (
    ThumbUnavailable,
    thumb_cache_name,
)


def test_thumb_cache_name_is_flat_safe_and_self_invalidating() -> None:
    name = thumb_cache_name("d1/img-1", 128, 1234567890)
    # No path separators; ends in the bucket + mtime + .png.
    assert "/" not in name and "\\" not in name
    assert name.endswith("_128_1234567890.png")
    assert re.fullmatch(r"[A-Za-z0-9_-]+\.png", name)


def test_thumb_cache_name_distinguishes_mtime() -> None:
    a = thumb_cache_name("d1/img-1", 128, 111)
    b = thumb_cache_name("d1/img-1", 128, 222)
    assert a != b  # a regenerated source yields a fresh cache file


def test_thumb_unavailable_is_runtime_error() -> None:
    assert issubclass(ThumbUnavailable, RuntimeError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/_shared/timeline/test_thumbnail.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Create `src/phenotypic/gui/_shared/timeline/_thumbnail.py`:

```python
"""Whole-image → cached downscaled thumbnail, plus a Flask route factory.

Mirrors ``gui/_shared/tiles.register_crop_route``: a surface supplies a
``resolve_source`` callable mapping a URL identity to an on-disk source PNG,
and this module owns the shared downscale + self-invalidating disk cache +
serving route. The cache filename embeds the source ``st_mtime_ns`` so a
regenerated source is served fresh without a stat-then-compare (spec §15.6);
writes are atomic (tempfile + os.replace).
"""
from __future__ import annotations

import base64
import logging

logger = logging.getLogger(__name__)


class ThumbUnavailable(RuntimeError):
    """Raised by a resolver when a source cannot be decoded on this platform.

    The common case is camera RAW on Windows. The route maps this to 422 + a
    fixed client message.
    """


def thumb_cache_name(identity: str, bucket: int, mtime_ns: int) -> str:
    """Return a flat, safe, self-invalidating cache filename.

    Args:
        identity: The URL identity (may contain ``/``).
        bucket: The snapped thumbnail size bucket.
        mtime_ns: Source ``st_mtime_ns`` — embedded so a regenerated source
            maps to a new filename (self-invalidating).

    Returns:
        ``<base64url(identity)>_<bucket>_<mtime_ns>.png`` (no path separators).
    """
    token = base64.urlsafe_b64encode(identity.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{token}_{bucket}_{mtime_ns}.png"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/_shared/timeline/test_thumbnail.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_shared/timeline/_thumbnail.py tests/gui/_shared/timeline/test_thumbnail.py
git commit -m "feat(gui-timeline): self-invalidating thumbnail cache naming + ThumbUnavailable"
```

---

### Task 5: `downscale_to_thumb`

**Files:**
- Modify: `src/phenotypic/gui/_shared/timeline/_thumbnail.py`
- Test: `tests/gui/_shared/timeline/test_thumbnail.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces: `downscale_to_thumb(src_png: Path, size: int) -> bytes` — opens the source, converts to RGB, downscales so the **longest edge == `size`** (aspect preserved, never upscaled past the source), returns PNG bytes.

- [ ] **Step 1: Write the failing test**

Append to `tests/gui/_shared/timeline/test_thumbnail.py`:

```python
import io
from pathlib import Path

from PIL import Image as PILImage

from phenotypic.gui._shared.timeline._thumbnail import downscale_to_thumb


def test_downscale_preserves_aspect_with_longest_edge(tmp_path: Path) -> None:
    src = tmp_path / "wide.png"
    PILImage.new("RGB", (200, 100), (255, 0, 0)).save(src, format="PNG")

    data = downscale_to_thumb(src, 64)

    out = PILImage.open(io.BytesIO(data))
    assert out.format == "PNG"
    assert out.size == (64, 32)  # 200x100 → longest edge 64


def test_downscale_outputs_rgb(tmp_path: Path) -> None:
    src = tmp_path / "rgba.png"
    PILImage.new("RGBA", (50, 50), (0, 255, 0, 128)).save(src, format="PNG")

    out = PILImage.open(io.BytesIO(downscale_to_thumb(src, 32)))
    assert out.mode == "RGB"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/_shared/timeline/test_thumbnail.py -v`
Expected: FAIL with `ImportError: cannot import name 'downscale_to_thumb'`.

- [ ] **Step 3: Write minimal implementation**

Add to `_thumbnail.py` (imports + function):

```python
import io
from pathlib import Path

from PIL import Image as PILImage


def downscale_to_thumb(src_png: Path, size: int) -> bytes:
    """Downscale ``src_png`` so its longest edge is ``size`` px; return PNG bytes.

    Aspect ratio is preserved (``PILImage.thumbnail``). The source is converted
    to RGB so palette/RGBA inputs serve consistently. ``thumbnail`` never
    upscales, so a source smaller than ``size`` is returned at its own size.

    Args:
        src_png: Path to the source PNG (already normalized for Browse; the
            overlay PNG for Results).
        size: Target longest-edge length in pixels.

    Returns:
        PNG-encoded bytes of the downscaled RGB image.
    """
    with PILImage.open(src_png) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail((size, size), PILImage.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="PNG")
        return buf.getvalue()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/_shared/timeline/test_thumbnail.py -v`
Expected: PASS (5 tests in the file).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_shared/timeline/_thumbnail.py tests/gui/_shared/timeline/test_thumbnail.py
git commit -m "feat(gui-timeline): downscale_to_thumb (longest-edge, aspect-preserving)"
```

---

### Task 6: `register_thumbnail_route` (Flask factory)

**Files:**
- Modify: `src/phenotypic/gui/_shared/timeline/_thumbnail.py`
- Test: `tests/gui/_shared/timeline/test_thumbnail_route.py`

**Interfaces:**
- Consumes: `snap_thumb_bucket` (Task 1, from `_config`), `thumb_cache_name` + `downscale_to_thumb` + `ThumbUnavailable` (Tasks 4–5).
- Produces: `register_thumbnail_route(app: dash.Dash, *, segment: str, resolve_source: Callable[[str], Path], cache_base: Path) -> None`. Mounts `GET /<segment>/<path:identity>?size=<int>`: snaps the bucket, resolves the source, serves a cached/atomic-written downscaled PNG. Errors: 400 (missing/invalid `size`), 404 (resolver raises `FileNotFoundError` / returns a non-file), 422 (resolver raises `ThumbUnavailable`).

- [ ] **Step 1: Write the failing test**

Create `tests/gui/_shared/timeline/test_thumbnail_route.py`:

```python
"""Flask-test-client smoke tests for register_thumbnail_route."""
from __future__ import annotations

import io
from pathlib import Path

import dash
import pytest
from PIL import Image as PILImage

from phenotypic.gui._shared.timeline._thumbnail import (
    ThumbUnavailable,
    register_thumbnail_route,
)


@pytest.fixture()
def client(tmp_path: Path):
    src = tmp_path / "src.png"
    PILImage.new("RGB", (200, 100), (0, 0, 255)).save(src, format="PNG")

    def resolve_source(identity: str) -> Path:
        if identity == "raw":
            raise ThumbUnavailable("no rawpy")
        if identity == "missing":
            raise FileNotFoundError(identity)
        return src

    app = dash.Dash(__name__)
    app.layout = dash.html.Div()  # REQUIRED: a layout-less Dash app 500s on first
    # request (before_request → validate_layout → NoLayoutException). Matches the
    # established idiom in tests/gui/browse/test_tile_routes.py:30.
    register_thumbnail_route(
        app, segment="thumb", resolve_source=resolve_source, cache_base=tmp_path / "cache"
    )
    return app.server.test_client()


def test_happy_path_returns_bucketed_png(client) -> None:
    resp = client.get("/thumb/img-1?size=100")  # snaps to bucket 128
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    out = PILImage.open(io.BytesIO(resp.data))
    assert max(out.size) == 128  # longest edge == snapped bucket


def test_missing_size_is_400(client) -> None:
    assert client.get("/thumb/img-1").status_code == 400


def test_thumb_unavailable_is_422(client) -> None:
    assert client.get("/thumb/raw?size=128").status_code == 422


def test_missing_source_is_404(client) -> None:
    assert client.get("/thumb/missing?size=128").status_code == 404


def test_second_request_is_served_from_cache(client, tmp_path: Path) -> None:
    assert client.get("/thumb/img-1?size=128").status_code == 200
    cache_files = list((tmp_path / "cache").glob("*.png"))
    assert len(cache_files) == 1
    # A second identical request reuses the cached file (no new file written).
    assert client.get("/thumb/img-1?size=128").status_code == 200
    assert list((tmp_path / "cache").glob("*.png")) == cache_files
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/_shared/timeline/test_thumbnail_route.py -v`
Expected: FAIL with `ImportError: cannot import name 'register_thumbnail_route'`.

- [ ] **Step 3: Write minimal implementation**

Add to `_thumbnail.py` (imports + function). Note `dash`, `flask`, and `os`/`tempfile` imports:

```python
import os
import tempfile
from collections.abc import Callable

import dash
from flask import Blueprint, Response, jsonify, request, send_file

from phenotypic.gui._config import snap_thumb_bucket


def register_thumbnail_route(
    app: dash.Dash,
    *,
    segment: str,
    resolve_source: Callable[[str], Path],
    cache_base: Path,
) -> None:
    """Mount ``GET /<segment>/<identity>?size=`` serving cached thumbnails.

    Args:
        app: The Dash app whose Flask server is extended.
        segment: URL path segment to mount under (e.g. ``"thumb"``). Also
            seeds the blueprint name so multiple segments coexist on one server.
        resolve_source: ``identity -> Path`` to the source PNG. Raise
            ``ThumbUnavailable`` for an undecodable source (→ 422) or
            ``FileNotFoundError`` for a missing one (→ 404). The resolver owns
            all path/sandbox validation.
        cache_base: Directory for the self-invalidating thumbnail cache
            (created on demand).
    """
    bp = Blueprint(f"timeline_thumb_{segment}", __name__, url_prefix=f"/{segment}")
    cache_base = Path(cache_base)

    @bp.route("/<path:identity>")
    def thumb_endpoint(identity: str) -> Response | tuple[str, int]:
        size = request.args.get("size", type=int)
        if size is None or size <= 0:
            return ("bad request: missing or invalid ?size=<int>", 400)
        bucket = snap_thumb_bucket(size)
        try:
            source = resolve_source(identity)
        except ThumbUnavailable as exc:
            logger.info("thumb unavailable for %s: %s", identity, exc)
            return _json_error("source cannot be rendered on this platform", 422)
        except FileNotFoundError:
            return _json_error("source not found", 404)
        source = Path(source)
        if not source.is_file():
            return _json_error("source not found", 404)

        cache_base.mkdir(parents=True, exist_ok=True)
        cache_file = cache_base / thumb_cache_name(
            identity, bucket, source.stat().st_mtime_ns
        )
        if not cache_file.exists():
            try:
                data = downscale_to_thumb(source, bucket)
            except Exception:
                logger.exception("thumb generation failed for %s", identity)
                return _json_error("thumbnail generation failed", 500)
            _atomic_write_bytes(cache_file, data)
        return send_file(cache_file, mimetype="image/png")

    app.server.register_blueprint(bp)
    logger.debug("Registered timeline thumb route under /%s", segment)


def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    """Write ``data`` to ``dest`` atomically (tempfile in the same dir + os.replace)."""
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _json_error(message: str, status: int) -> Response:
    """Build a small JSON error ``Response`` with the given status code."""
    response = jsonify({"error": message})
    response.status_code = status
    return response
```

**Per-source render lock (spec §4.2/§15.6 — include now, decided 2026-06-18).** Wrap the
`if not cache_file.exists(): … downscale_to_thumb(…) … _atomic_write_bytes(…)` block in a
**per-cache-key lock** so that when Phase 2's background-warm sweep fires many concurrent
fetches, two requests for the *same* thumbnail don't both decode+downscale. Mirror the real
repo pattern — `results_viewer/_dzi_tiler._get_lock` (verified 2026-06-18):

```python
_LOCK_CACHE_SIZE = 512

@functools.lru_cache(maxsize=_LOCK_CACHE_SIZE)
def _get_lock(key: str) -> threading.Lock:
    return threading.Lock()
```

Key it by `cache_file.name` (the self-invalidating `<identity>_<bucket>_<mtime>.png` name —
so the lock is per distinct thumbnail). `with _get_lock(cache_file.name):` then re-check
`cache_file.exists()` inside the lock before rendering (double-checked locking). Atomic write
already prevents *corruption* and covers the rare lru-eviction-while-held race (spec §15.6);
the lock prevents *duplicate work*. Add a test asserting two concurrent same-key requests
produce exactly one cache file and one render (e.g. monkeypatch `downscale_to_thumb` with a
call-counter + a small sleep, fire two threads, assert the counter == 1).
**NOTE:** there is no `browse/_dzi_tiler.py`; the only `_get_lock` in the tree is the
`results_viewer` one above — mirror *that*.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/_shared/timeline/test_thumbnail_route.py -v`
Expected: PASS (6 tests: happy-path, missing-size 400, 422, 404, cache-reuse, single-render-under-concurrency).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_shared/timeline/_thumbnail.py tests/gui/_shared/timeline/test_thumbnail_route.py
git commit -m "feat(gui-timeline): register_thumbnail_route (cached, atomic, 422/404)"
```

---

### Task 7: `build_timeline_grid`

**Files:**
- Create: `src/phenotypic/gui/_shared/timeline/_grid.py`
- Test: `tests/gui/_shared/timeline/test_grid.py`

**Interfaces:**
- Consumes: `TimelineMatrix`/`TimelineCell` (Task 3); `TIMELINE_GRID_GAP_PX` (Task 1).
- Produces: `build_timeline_grid(matrix: TimelineMatrix, *, url_builder: Callable[[object, int], str], display_size: int, fetch_size: int, gap_px: int = TIMELINE_GRID_GAP_PX, ref_builder: Callable[[object], str] | None = None) -> tuple[Component, list[tuple[str, str]]]`. Returns the CSS-grid component **and** the row-major list of non-empty `(row_value, time_value)` keys (the testable invariant, mirroring `colony_view.build_grid`'s `grid_order`). Each non-empty cell is a placeholder `html.Div` (no `<img>`) carrying `data-src` (from `url_builder(representative, fetch_size)`), `data-ref` (from `ref_builder(representative)`, default `str(representative)` — the surface's opaque pop-out identity), `data-row`, `data-col`, `data-key="row::time"`, **`data-row-index`/`data-col-index`** (0-based positions in `matrix.rows`/`matrix.columns`, for the focus-navigate controller's grid-coordinate math — spec §16.8), plus a hover-revealed pop-out button (`timeline-cell-popout` styled to appear on `:hover`, spec §16.4) and an `N=k` badge when `count > 1`. **Empty cells also carry `data-row-index`/`data-col-index`** (so every grid coordinate is addressable) under a `timeline-cell timeline-cell--empty` placeholder. **Axis labels** carry their value + index too: the `--x` time headers get `data-col`/`data-col-index`, the `--y` row headers get `data-row`/`data-row-index`, so a header click resolves to its row/column of cells by attribute (the Compare-strip row-header trigger, §7) rather than fragile `textContent` matching.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/_shared/timeline/test_grid.py`:

```python
"""build_timeline_grid: row-major key order + per-cell URL generation."""
from __future__ import annotations

from dash import html

from phenotypic.gui._shared.timeline._grid import build_timeline_grid
from phenotypic.gui._shared.timeline._matrix import build_matrix


def _matrix():
    records = [
        {"row_value": "plateA", "time_value": "1", "cell_ref": "a1"},
        {"row_value": "plateA", "time_value": "2", "cell_ref": "a2"},
        {"row_value": "plateA", "time_value": "10", "cell_ref": "a10"},
        {"row_value": "plateB", "time_value": "1", "cell_ref": "b1"},
        {"row_value": "plateA", "time_value": "1", "cell_ref": "a1b"},  # count=2
    ]
    return build_matrix(records)


def test_grid_order_is_row_major_over_nonempty_cells() -> None:
    calls: list[tuple[object, int]] = []

    def url_builder(ref: object, fetch: int) -> str:
        calls.append((ref, fetch))
        return f"/thumb/{ref}?size={fetch}"

    component, order = build_timeline_grid(
        _matrix(), url_builder=url_builder, display_size=120, fetch_size=128
    )

    assert isinstance(component, html.Div)
    # Row-major (Y outer, X inner); ("plateB","2") is empty and excluded.
    assert order == [
        ("plateA", "1"),
        ("plateA", "2"),
        ("plateA", "10"),
        ("plateB", "1"),
    ]


def test_url_builder_called_once_per_nonempty_cell_with_fetch_size() -> None:
    calls: list[tuple[object, int]] = []

    def url_builder(ref: object, fetch: int) -> str:
        calls.append((ref, fetch))
        return "x"

    build_timeline_grid(
        _matrix(), url_builder=url_builder, display_size=120, fetch_size=128
    )

    assert len(calls) == 4  # one per non-empty cell
    assert all(fetch == 128 for _ref, fetch in calls)
    # The (plateA,1) cell aggregates 2 members; its representative is "a1".
    assert ("a1", 128) in calls


def test_ref_builder_called_once_per_nonempty_cell() -> None:
    refs: list[object] = []

    def ref_builder(ref: object) -> str:
        refs.append(ref)
        return f"TOKEN::{ref}"

    build_timeline_grid(
        _matrix(),
        url_builder=lambda ref, fetch: "x",
        display_size=120,
        fetch_size=128,
        ref_builder=ref_builder,
    )
    assert len(refs) == 4  # one per non-empty cell
    assert "a1" in refs  # representative of the aggregated (plateA,1) cell


def test_cells_carry_grid_coordinate_indices() -> None:
    # The focus-navigate controller addresses cells by 0-based grid coordinate
    # (spec §16.8); every cell — empty or not — must expose both indices.
    # In _matrix(), (plateB, "2") and (plateB, "10") are EMPTY cells — the test
    # must prove THOSE carry coordinates (the new §16.8 requirement), not just
    # the always-attributed non-empty cells.
    component, _ = build_timeline_grid(
        _matrix(), url_builder=lambda ref, fetch: "x", display_size=120, fetch_size=128
    )

    def _walk(node):
        yield node
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                yield from _walk(child)
        elif children is not None:
            yield from _walk(children)

    empties = [
        n for n in _walk(component)
        if "timeline-cell--empty" in (getattr(n, "className", "") or "")
    ]
    assert empties, "expected at least one empty placeholder cell"
    for cell in empties:
        props = cell.to_plotly_json().get("props", {})
        assert "data-row-index" in props and "data-col-index" in props
    # plateB row is index 1; its empty time-columns "2"/"10" are col-index 1/2.
    empty_coords = {
        (cell.to_plotly_json()["props"]["data-row-index"],
         cell.to_plotly_json()["props"]["data-col-index"])
        for cell in empties
    }
    assert ("1", "1") in empty_coords and ("1", "2") in empty_coords
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/_shared/timeline/test_grid.py -v`
Expected: FAIL with `ModuleNotFoundError: ..._grid`.

- [ ] **Step 3: Write minimal implementation**

Create `src/phenotypic/gui/_shared/timeline/_grid.py`:

```python
"""Pure renderer for the timeline grid (placeholders + row-major key order).

Renders a CSS grid sized to the full matrix (corner + time-column headers +
per-row [row-header, cells…]). Every data cell is a SIZE-MATCHED PLACEHOLDER
``html.Div`` carrying ``data-src`` (the thumbnail URL) and identity data-attrs
— NO ``<img>`` enters the DOM here; the virtualization JS (a later phase)
mounts/unmounts the image on scroll. Returns the component plus the row-major
list of non-empty ``(row_value, time_value)`` keys, mirroring
``colony_view.build_grid``'s ``grid_order`` so selection ranges resolve the
same way.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dash import html
from dash.development.base_component import Component

from phenotypic.gui._config import TIMELINE_GRID_GAP_PX
from phenotypic.gui._shared.timeline._matrix import TimelineCell, TimelineMatrix


def build_timeline_grid(
    matrix: TimelineMatrix,
    *,
    url_builder: Callable[[object, int], str],
    display_size: int,
    fetch_size: int,
    gap_px: int = TIMELINE_GRID_GAP_PX,
    ref_builder: Callable[[object], str] | None = None,
) -> tuple[Component, list[tuple[str, str]]]:
    """Render the timeline grid component and its row-major key order.

    Args:
        matrix: The ordered matrix from :func:`build_matrix`.
        url_builder: ``(representative_ref, fetch_size) -> thumbnail URL``,
            written into each placeholder's ``data-src``.
        display_size: CSS tile size (px) — the rendered placeholder size.
        fetch_size: Snapped thumbnail bucket (px) passed to ``url_builder``.
        gap_px: CSS gap between tiles.
        ref_builder: Optional ``representative_ref -> str`` written into each
            cell's ``data-ref`` (the surface's opaque identity for pop-out /
            deep-zoom — Browse encodes a token, Results a ``"dataset/stem"``).
            Defaults to ``str(representative)``.

    Returns:
        ``(component, grid_order)`` where ``grid_order`` is the row-major list
        of non-empty ``(row_value, time_value)`` keys.
    """
    children: list[Component] = [html.Div(className="timeline-grid-corner")]
    for col_index, time_value in enumerate(matrix.columns):
        children.append(
            html.Div(
                time_value,
                className="timeline-axis-label timeline-axis-label--x",
                # Axis labels carry their value + index so the JS can match a
                # header click to its column/row of cells without fragile
                # textContent matching (Compare strip row/column triggers, §7).
                **{"data-col": time_value, "data-col-index": str(col_index)},
            )
        )

    grid_order: list[tuple[str, str]] = []
    for row_index, row_value in enumerate(matrix.rows):
        children.append(
            html.Div(
                row_value,
                className="timeline-axis-label timeline-axis-label--y",
                **{"data-row": row_value, "data-row-index": str(row_index)},
            )
        )
        for col_index, time_value in enumerate(matrix.columns):
            cell = matrix.cells.get((row_value, time_value))
            if cell is None:
                children.append(
                    html.Div(
                        className="timeline-cell timeline-cell--empty",
                        style={"width": f"{display_size}px", "height": f"{display_size}px"},
                        # Every grid coordinate is addressable by the focus controller.
                        **{
                            "data-row-index": str(row_index),
                            "data-col-index": str(col_index),
                        },
                    )
                )
                continue
            grid_order.append((row_value, time_value))
            children.append(
                _build_cell(
                    cell,
                    url_builder,
                    display_size,
                    fetch_size,
                    ref_builder,
                    row_index=row_index,
                    col_index=col_index,
                )
            )

    grid = html.Div(
        children,
        className="timeline-grid",
        style={
            "display": "grid",
            "gridTemplateColumns": (
                f"minmax(0, {display_size}px) "
                + " ".join([f"{display_size}px"] * len(matrix.columns))
            ),
            "gap": f"{gap_px}px",
            "width": "max-content",
        },
    )
    return grid, grid_order


def _build_cell(
    cell: TimelineCell,
    url_builder: Callable[[object, int], str],
    display_size: int,
    fetch_size: int,
    ref_builder: Callable[[object], str] | None,
    *,
    row_index: int,
    col_index: int,
) -> Component:
    """Render one placeholder cell (no <img>; data-src drives focus-window mount)."""
    ref = ref_builder(cell.representative) if ref_builder else str(cell.representative)
    data_props: dict[str, Any] = {
        "data-src": url_builder(cell.representative, fetch_size),
        "data-ref": ref,
        "data-row": cell.row_value,
        "data-col": cell.time_value,
        "data-key": f"{cell.row_value}::{cell.time_value}",
        # Grid coordinates for the focus-navigate controller (spec §16.8).
        "data-row-index": str(row_index),
        "data-col-index": str(col_index),
    }
    inner: list[Component] = [
        # Hover-revealed via CSS (.timeline-cell:hover .timeline-cell-popout);
        # focus + Enter also opens the pop-out (spec §16.4).
        html.Button(
            "⤢",
            className="timeline-cell-popout",
            title="Open full-resolution view",
            type="button",
            n_clicks=0,
        )
    ]
    if cell.count > 1:
        inner.append(html.Span(f"N={cell.count}", className="timeline-cell-badge"))
    return html.Div(
        inner,
        className="timeline-cell",
        style={
            "width": f"{display_size}px",
            "height": f"{display_size}px",
            "position": "relative",
        },
        **data_props,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/_shared/timeline/test_grid.py -v`
Expected: PASS (4 tests: grid order, url_builder, ref_builder, grid-coordinate indices).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_shared/timeline/_grid.py tests/gui/_shared/timeline/test_grid.py
git commit -m "feat(gui-timeline): build_timeline_grid (placeholder cells + grid_order)"
```

---

### Task 8: Package exports + FEATURES.md infra row

**Files:**
- Modify: `src/phenotypic/gui/_shared/timeline/__init__.py`
- Modify: `src/phenotypic/gui/FEATURES.md`
- Test: `tests/gui/_shared/timeline/test_public_api.py`

**Interfaces:**
- Consumes: every public symbol from Tasks 2–7.
- Produces: the package's public API surface (`from phenotypic.gui._shared.timeline import build_matrix, TimelineMatrix, TimelineCell, downscale_to_thumb, register_thumbnail_route, ThumbUnavailable, thumb_cache_name, build_timeline_grid`).

- [ ] **Step 1: Write the failing test**

Create `tests/gui/_shared/timeline/test_public_api.py`:

```python
"""The timeline engine package exposes its public API from __init__."""
from __future__ import annotations

import phenotypic.gui._shared.timeline as timeline


def test_public_api_is_exported() -> None:
    expected = {
        "build_matrix",
        "TimelineMatrix",
        "TimelineCell",
        "downscale_to_thumb",
        "register_thumbnail_route",
        "ThumbUnavailable",
        "thumb_cache_name",
        "build_timeline_grid",
    }
    assert expected.issubset(set(timeline.__all__))
    for name in expected:
        assert hasattr(timeline, name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/_shared/timeline/test_public_api.py -v`
Expected: FAIL with `AssertionError` (or `AttributeError: module ... has no attribute '__all__'`).

- [ ] **Step 3: Write minimal implementation**

Replace `src/phenotypic/gui/_shared/timeline/__init__.py` with:

```python
"""Source-agnostic timeline-view engine (matrix model, thumbnail route, grid).

Consumed by the Browse and Results timeline surfaces (later phases). Mirrors
``gui/_shared/tiles.py`` as the single owner of the matrix model, the cached
thumbnail route factory, and the placeholder-grid renderer.
"""
from __future__ import annotations

from phenotypic.gui._shared.timeline._grid import build_timeline_grid
from phenotypic.gui._shared.timeline._matrix import (
    TimelineCell,
    TimelineMatrix,
    build_matrix,
)
from phenotypic.gui._shared.timeline._thumbnail import (
    ThumbUnavailable,
    downscale_to_thumb,
    register_thumbnail_route,
    thumb_cache_name,
)

__all__ = [
    "build_matrix",
    "TimelineMatrix",
    "TimelineCell",
    "downscale_to_thumb",
    "register_thumbnail_route",
    "ThumbUnavailable",
    "thumb_cache_name",
    "build_timeline_grid",
]
```

Add one row to the **`## Cross-cutting infrastructure`** table in `src/phenotypic/gui/FEATURES.md`, matching its exact 6-column header (verified at `FEATURES.md:430`): `| Feature | Element | Expected behaviour | Status | Test layer | Test ref |`. (`check_features_md.py` only enforces cell-count == header-count, but match the real column names so the row reads correctly.) Use `🧪 internal` (tested, not user-facing) — **never `🚧 in progress`** (the merge gate rejects it). Append this row under the `## Cross-cutting infrastructure` heading:

```markdown
| Timeline shared engine | `gui/_shared/timeline/` | Source-agnostic matrix model (`build_matrix`), cached self-invalidating thumbnail route factory (`register_thumbnail_route`), and placeholder grid (`build_timeline_grid`) consumed by the Browse/Results timeline surfaces. No user-facing affordance yet (Phase 1). | 🧪 internal | unit | tests/gui/_shared/timeline/test_public_api.py::test_public_api_is_exported |
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/_shared/timeline/test_public_api.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Run the full Phase 1 suite + lint**

Run: `uv run pytest tests/gui/_shared/timeline/ -v`
Expected: PASS (all tasks' tests green).
Run: `uv run ruff check src/phenotypic/gui/_shared/timeline src/phenotypic/gui/_config.py`
Run: `uv run mypy src/phenotypic/gui/_shared/timeline`
Expected: clean (fix any reported issues before committing).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/_shared/timeline/__init__.py src/phenotypic/gui/FEATURES.md tests/gui/_shared/timeline/test_public_api.py
git commit -m "feat(gui-timeline): export shared-engine public API + FEATURES infra row"
```

---

## Phase 1 deliverable

A tested, importable `gui/_shared/timeline/` engine: `build_matrix` (ordered matrix + representative + empty cells), `register_thumbnail_route` (cached/atomic/422/404 thumbnail serving), `downscale_to_thumb`, and `build_timeline_grid` (placeholder cells + `grid_order`). No user-visible surface yet — that is **Phase 2 (Browse)** and **Phase 3 (Results)**, which consume these interfaces, plus `timeline.js` (virtualization + warm) introduced in Phase 2 where a live page can drive it under Playwright.

## Subsequent phases (separate plans, written when reached)

- **Phase 2 — Browse surface:** view-mode toggle, per-axis source picker (folder / `{plate}`·`{time}` pattern / CSV column), folder-scoped CSV join, plate-identity pattern + live preview + nudge, token-keyed thumbnail route (reusing `normalize_to_png`), `timeline.js` virtualization + background warm, single-image pop-out (reuse the browse DZI route). FEATURES.md + WORKFLOWS.md + screenshots.
- **Phase 3 — Results surface:** `Timeline` tab, `(dataset, stem)` thumbnail route over overlays, Y dropdown (`selectable_axis_columns(df, column_value_sets)`), uncapped time-column predicate + empty state, filter-sidebar aware.
- **Phase 4 — Synced Compare strip:** row-header + multi-select triggers, ≤12 viewport-synced OSD viewers, shared-viewport feedback guard, accepted DZI spike.
- **Phase 5 — CLI `deliverables/metadata.csv` copy:** best-effort copy in `finalize_post_master_outputs` + `sdk_` path helper.
- **Phase 6 — Docs/CI:** tutorial pages, screenshot capture, final code-simplifier pass + regression run.
