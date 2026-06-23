# Source Timeline View — Phase 2: Browse Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Timeline matrix view to the Browse tab — a folder/EXIF filmstrip by default, upgradeable to a per-axis source picker (folder / `{plate}`·`{time}` pattern / metadata-CSV column), rendered through the Phase 1 shared engine with a browser-side **focus-and-navigate** controller (spec §16) and a single-image deep-zoom pop-out.

**Architecture:** A `Single | Timeline` view-mode toggle swaps the Browse body (both stay mounted, CSS visibility). The Timeline body's controls feed a pure record-builder (`build_browse_records`) → `build_matrix` → `build_timeline_grid` (Phase 1). A token-keyed thumbnail route reuses `_source_render.normalize_to_png` then downscales. `timeline.js` is a **focus-and-navigate controller** (spec §16): the matrix is **not scrollable** — a no-scroll viewport renders a centered window around one **focused cell**, ←/→/↑/↓ (and four on-edge ◀▶▲▼ buttons) move focus, the focused neighborhood plus a margin ring mounts `<img>` (cells beyond offload), and the cache warms neighborhood-first in the background. Pop-out reuses the existing browse DZI route + OpenSeadragon (Enter/Space on the focused cell, or the hover-revealed ⤢ on any visible tile).

**Tech Stack:** Dash + dash-bootstrap-components, Flask blueprint, Pillow (via Phase 1), `exifread` (lightweight EXIF), OpenSeadragon (vendored, already present), pytest + Playwright (`tests/e2e/gui`).

## Global Constraints

- **`uv` is the sole runner.** Every command is `uv run …`; never bare `python`/`pip`.
- **Phase 1 must be merged/available.** This plan consumes `phenotypic.gui._shared.timeline`: `build_matrix`, `TimelineMatrix`, `build_timeline_grid`, `register_thumbnail_route`, `ThumbUnavailable`, and `phenotypic.gui._config` constants `THUMB_SIZE_BUCKETS`, `snap_thumb_bucket`, `TIMELINE_TILE_SIZE_DEFAULT/STEP/MIN/MAX`, `BROWSE_THUMB_URL_SEGMENT`, `TIMELINE_WARM_CONCURRENCY`, `TIMELINE_MOUNT_CAP`, `TIMELINE_FOCUS_MARGIN`. (Note: `TIMELINE_WINDOW_MARGIN_SCREENS` was **removed** by Phase 1 — it was a scroll-era concept superseded by `TIMELINE_FOCUS_MARGIN`, spec §16.7.)
- **Focus-and-navigate model (spec §16 — binding; supersedes the scroll model in D4/§4.4/§15.1).** The Browse Timeline is **not scrollable**. A no-scroll viewport renders a **centered window** of tiles around one **focused cell**; ←/→/↑/↓ and the four on-edge ◀▶▲▼ buttons move focus (clamped at matrix bounds, no wrap); the focused neighborhood + a `TIMELINE_FOCUS_MARGIN`-cell margin ring mounts `<img>` and everything beyond offloads. The engine cells (Phase 1 `build_timeline_grid`) carry `data-row-index`/`data-col-index` (0-based positions in `matrix.rows`/`matrix.columns`) so `timeline.js` can address cells by grid coordinate; the `BROWSE_TL_GRID` container exposes `data-focus-margin` (= `str(TIMELINE_FOCUS_MARGIN)`), `data-mount-cap`, and `data-warm-concurrency` as **static** data-attrs (the scroll-era `data-margin-screens` is gone). The ⤢ pop-out button is **hover-revealed via CSS** (Phase 1); Enter/Space on the focused cell opens the same pop-out.
- **Single-source constants** in `_config.py` / `_design.py`; new Browse component ids in `browse/_ids.py`. Don't re-spell literals; don't import `dash` from `_config.py`/`_design.py`.
- **FEATURES.md + WORKFLOWS.md gates:** any `src/phenotypic/gui/` change must modify `FEATURES.md`; the Timeline is a tutorial-worthy flow so it also needs a `WORKFLOWS.md` row + a `_capture_<id>` in `scripts/capture_gui_tutorial_screenshots.py` + a tutorial page. `✅ shipping` rows need a resolvable `path::test`; never leave a row `🚧 in progress` (merge gate rejects it).
- **Per-axis source picker (spec §5.2, §15.4):** row ∈ {folder | `{plate}` pattern | CSV column}; time ∈ {EXIF | `{time}` pattern | CSV column}. Default = folder + EXIF. CSV joins **folder-scoped by image stem** (no path column).
- **Folder-separated rows (spec §15.5):** pattern row key = `(dataset_folder, {plate})`; identical `{plate}` across folders are separate rows.
- **Lightweight EXIF (spec §15.11):** read capture time via `exifread` directly — never `Image.imread(...).rgb[:]` (full decode).
- **Verify Dash wiring in a live browser (project rule):** callback/JS tasks carry Playwright e2e tests, not only unit tests.

---

### Task 1: Browse Timeline component ids

**Files:**
- Modify: `src/phenotypic/gui/browse/_ids.py`
- Test: `tests/gui/browse/test_ids.py` (append; file already exists)

**Interfaces:**
- Consumes: nothing.
- Produces: new `BROWSE_*` id constants used by every later Browse task. Exact names below. The four `BROWSE_TL_NAV_*` edge-button ids and `BROWSE_TL_POSITION` (focus-navigate, spec §16) are **DOM targets for `timeline.js`** — they carry no Dash callback (the controller binds their clicks + sets the readout text in JS).

- [ ] **Step 1: Write the failing test**

Append to `tests/gui/browse/test_ids.py`:

```python
def test_timeline_ids_present_and_unique() -> None:
    from phenotypic.gui.browse import _ids

    timeline_ids = [
        _ids.BROWSE_VIEW_MODE_TOGGLE,
        _ids.BROWSE_SINGLE_BODY,
        _ids.BROWSE_TIMELINE_BODY,
        _ids.BROWSE_TL_ROW_SOURCE,
        _ids.BROWSE_TL_TIME_SOURCE,
        _ids.BROWSE_TL_ROW_CSV_COL,
        _ids.BROWSE_TL_TIME_CSV_COL,
        _ids.BROWSE_TL_CSV_IMAGE_COL,
        _ids.BROWSE_TL_PATTERN_INPUT,
        _ids.BROWSE_TL_PATTERN_ADVANCED,
        _ids.BROWSE_TL_PATTERN_PREVIEW,
        _ids.BROWSE_TL_TILE_SIZE_MINUS,
        _ids.BROWSE_TL_TILE_SIZE_PLUS,
        _ids.BROWSE_TL_TILE_SIZE_READOUT,
        _ids.BROWSE_TL_NAV_UP,
        _ids.BROWSE_TL_NAV_DOWN,
        _ids.BROWSE_TL_NAV_LEFT,
        _ids.BROWSE_TL_NAV_RIGHT,
        _ids.BROWSE_TL_POSITION,
        _ids.BROWSE_TL_NUDGE,
        _ids.BROWSE_TL_GRID,
        _ids.BROWSE_TL_STORE_TILE_SIZE,
        _ids.BROWSE_TL_STORE_WARNINGS,
        _ids.BROWSE_TL_POPOUT_MODAL,
        _ids.BROWSE_TL_POPOUT_OSD,
        _ids.BROWSE_TL_POPOUT_STORE,
        _ids.BROWSE_TL_POPOUT_INPUT,
    ]
    assert len(timeline_ids) == len(set(timeline_ids))  # all unique
    assert all(isinstance(i, str) and i for i in timeline_ids)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_ids.py::test_timeline_ids_present_and_unique -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'BROWSE_VIEW_MODE_TOGGLE'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/phenotypic/gui/browse/_ids.py` (before `__all__`, then add each to `__all__`):

```python
# --- Timeline view (Phase 2) ---------------------------------------------
BROWSE_VIEW_MODE_TOGGLE = "browse-view-mode-toggle"   # "single" | "timeline"
BROWSE_SINGLE_BODY = "browse-single-body"             # existing OSD pane wrapper
BROWSE_TIMELINE_BODY = "browse-timeline-body"         # timeline matrix wrapper
BROWSE_TL_ROW_SOURCE = "browse-tl-row-source"         # folder|pattern|csv
BROWSE_TL_TIME_SOURCE = "browse-tl-time-source"       # exif|pattern|csv
BROWSE_TL_ROW_CSV_COL = "browse-tl-row-csv-col"
BROWSE_TL_TIME_CSV_COL = "browse-tl-time-csv-col"
BROWSE_TL_CSV_IMAGE_COL = "browse-tl-csv-image-col"
BROWSE_TL_PATTERN_INPUT = "browse-tl-pattern-input"
BROWSE_TL_PATTERN_ADVANCED = "browse-tl-pattern-advanced"
BROWSE_TL_PATTERN_PREVIEW = "browse-tl-pattern-preview"
BROWSE_TL_TILE_SIZE_MINUS = "browse-tl-tile-size-minus"
BROWSE_TL_TILE_SIZE_PLUS = "browse-tl-tile-size-plus"
BROWSE_TL_TILE_SIZE_READOUT = "browse-tl-tile-size-readout"
# Focus-and-navigate (spec §16). The four on-edge directional buttons and the
# focused-cell position readout are DOM targets driven by timeline.js — they
# need NO Dash callbacks (the controller wires clicks + keyboard in JS).
BROWSE_TL_NAV_UP = "browse-tl-nav-up"                 # ▲ move focus up a row
BROWSE_TL_NAV_DOWN = "browse-tl-nav-down"             # ▼ move focus down a row
BROWSE_TL_NAV_LEFT = "browse-tl-nav-left"             # ◀ move focus back in time
BROWSE_TL_NAV_RIGHT = "browse-tl-nav-right"           # ▶ move focus forward in time
BROWSE_TL_POSITION = "browse-tl-position"             # "row 1/74 · time 1/24" readout
BROWSE_TL_NUDGE = "browse-tl-nudge"                   # "add a CSV" banner
BROWSE_TL_GRID = "browse-tl-grid"                     # grid container (timeline.js target)
BROWSE_TL_STORE_TILE_SIZE = "browse-tl-store-tile-size"
BROWSE_TL_STORE_WARNINGS = "browse-tl-store-warnings"
BROWSE_TL_POPOUT_MODAL = "browse-tl-popout-modal"
BROWSE_TL_POPOUT_OSD = "browse-tl-popout-osd"
BROWSE_TL_POPOUT_STORE = "browse-tl-popout-store"     # {token,label} for the pop-out
BROWSE_TL_POPOUT_INPUT = "browse-tl-popout-input"     # hidden dcc.Input; JS→Dash bridge for ⤢ clicks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/browse/test_ids.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/browse/_ids.py tests/gui/browse/test_ids.py
git commit -m "feat(gui-timeline): Browse Timeline component ids"
```

---

### Task 2: Lightweight EXIF capture-time helper

**Files:**
- Create: `src/phenotypic/gui/browse/_capture_time.py`
- Create: `tests/gui/browse/fixtures/with_datetimeoriginal.jpg` (committed binary fixture — see note)
- Test: `tests/gui/browse/test_capture_time.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `read_capture_time(path: Path) -> str | None` — reads EXIF `DateTimeOriginal` (falling back to `Image DateTime`) via `exifread` directly (no pixel decode), cached per `(path, mtime_ns)`. Returns `None` when absent/unreadable.

- [ ] **Step 1: Write the failing test**

The thing under test is `read_capture_time` — the **reader**, not a writer — so the test reads a tiny **committed** JPEG fixture that carries `DateTimeOriginal`, with **no `piexif` (or any EXIF-writer) dependency** in the test itself (OQ-3). Create `tests/gui/browse/test_capture_time.py`:

```python
"""Lightweight EXIF capture-time reader (no full image decode)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage

from phenotypic.gui.browse._capture_time import read_capture_time

# Committed fixture: an 8x8 JPEG whose EXIF DateTimeOriginal is
# "2024:01:02 03:04:05" (authored once; see the authoring note below).
_FIXTURE = Path(__file__).parent / "fixtures" / "with_datetimeoriginal.jpg"


def test_reads_datetimeoriginal() -> None:
    assert read_capture_time(_FIXTURE) == "2024:01:02 03:04:05"


def test_returns_none_without_exif(tmp_path: Path) -> None:
    img = tmp_path / "plain.png"
    PILImage.new("RGB", (8, 8), (0, 0, 0)).save(img, format="PNG")
    assert read_capture_time(img) is None
```

> **Authoring the committed fixture (one-time, OQ-3 — no runtime/dev `piexif` dependency).** The test depends only on the committed binary, never on an EXIF writer. Author `tests/gui/browse/fixtures/with_datetimeoriginal.jpg` once with a throwaway one-liner (do NOT add `piexif` to `pyproject.toml`; run it ad-hoc with `uv run --with piexif python - <<'PY'` so the dep is ephemeral and never lands in the lockfile), then `git add` the resulting JPEG:
>
> ```python
> # one-time author script (ephemeral piexif via `uv run --with piexif`):
> import piexif
> from PIL import Image as PILImage
> exif = {"Exif": {piexif.ExifIFD.DateTimeOriginal: b"2024:01:02 03:04:05"}}
> PILImage.new("RGB", (8, 8), (10, 20, 30)).save(
>     "tests/gui/browse/fixtures/with_datetimeoriginal.jpg",
>     format="JPEG", exif=piexif.dump(exif),
> )
> ```
>
> The committed JPEG is ~1 KB. Create the `tests/gui/browse/fixtures/` directory (it does not exist yet — verified). Confirm the fixture round-trips before relying on it: `uv run python -c "from phenotypic.gui.browse._capture_time import read_capture_time; print(read_capture_time('tests/gui/browse/fixtures/with_datetimeoriginal.jpg'))"` should print `2024:01:02 03:04:05` once Step 3 lands.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_capture_time.py -v`
Expected: FAIL with `ModuleNotFoundError: ..._capture_time`.

- [ ] **Step 3: Write minimal implementation**

Create `src/phenotypic/gui/browse/_capture_time.py`:

```python
"""Read EXIF capture time without decoding pixels.

The Browse Timeline orders the time axis (no-CSV / EXIF path) by capture
time over potentially hundreds of source images. Routing that through
``Image.imread(...).rgb[:]`` (as ``browse/_metadata.read`` does) would decode
every image — far too expensive (spec §15.11). ``exifread`` parses only the
EXIF block, so this is cheap, and results are cached per ``(path, mtime_ns)``.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path

import exifread

logger = logging.getLogger(__name__)

__all__ = ["read_capture_time"]


@functools.lru_cache(maxsize=4096)
def _read_capture_time_cached(path_str: str, mtime_ns: int) -> str | None:
    del mtime_ns  # cache-key only (invalidates when the file changes)
    try:
        with open(path_str, "rb") as handle:
            tags = exifread.process_file(
                handle, details=False, stop_tag="DateTimeOriginal"
            )
    except Exception:  # noqa: BLE001 - capture time is best-effort
        logger.debug("EXIF read failed for %s", path_str, exc_info=True)
        return None
    for key in ("EXIF DateTimeOriginal", "Image DateTime"):
        value = tags.get(key)
        if value is not None:
            return str(value)
    return None


def read_capture_time(path: Path) -> str | None:
    """Return the EXIF capture-time string for ``path``, or ``None``.

    Prefers ``DateTimeOriginal`` (true capture) over the bare ``DateTime``
    (often the file write/scan time), mirroring ``browse/_metadata``'s
    ordering. Best-effort: any failure returns ``None``.
    """
    path = Path(path)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return None
    return _read_capture_time_cached(str(path), mtime_ns)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/browse/test_capture_time.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/browse/_capture_time.py tests/gui/browse/test_capture_time.py tests/gui/browse/fixtures/with_datetimeoriginal.jpg
git commit -m "feat(gui-timeline): lightweight exifread capture-time reader"
```

> (No `pyproject.toml`/`uv.lock` change — `exifread` is already a dependency and
> the EXIF-writer `piexif` is **not** added; the committed JPEG fixture replaces
> the runtime writer, OQ-3.)

---

### Task 3: Plate-identity pattern parser

**Files:**
- Create: `src/phenotypic/gui/browse/_plate_pattern.py`
- Test: `tests/gui/browse/test_plate_pattern.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class PlateMatch(stem: str, plate: str | None, time: str | None)` — frozen dataclass.
  - `class PatternError(ValueError)`.
  - `parse_plate_identity(stems: Iterable[str], pattern: str, *, advanced: bool = False) -> list[PlateMatch]`. Placeholder syntax `{plate}` (required), optional `{time}`, `*` wildcard, literals; compiles to an anchored non-greedy regex. Advanced mode: raw regex requiring a named `plate` group. Non-matching stems → `PlateMatch(stem, None, None)`. Invalid pattern → `PatternError`.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/browse/test_plate_pattern.py`:

```python
"""Plate-identity pattern compilation + matching."""
from __future__ import annotations

import pytest

from phenotypic.gui.browse._plate_pattern import (
    PatternError,
    PlateMatch,
    parse_plate_identity,
)


def test_placeholder_extracts_plate_and_time() -> None:
    out = parse_plate_identity(
        ["Exp1_PlateA_t03", "Exp1_PlateB_t10"], "{plate}_t{time}"
    )
    assert out == [
        PlateMatch("Exp1_PlateA_t03", "Exp1_PlateA", "03"),
        PlateMatch("Exp1_PlateB_t10", "Exp1_PlateB", "10"),
    ]


def test_nonmatching_stem_yields_none() -> None:
    out = parse_plate_identity(["junk"], "{plate}_t{time}")
    assert out == [PlateMatch("junk", None, None)]


def test_plate_only_pattern_leaves_time_none() -> None:
    out = parse_plate_identity(["plateA"], "{plate}")
    assert out == [PlateMatch("plateA", "plateA", None)]


def test_missing_plate_token_raises() -> None:
    with pytest.raises(PatternError):
        parse_plate_identity(["x"], "t{time}")


def test_duplicate_token_raises() -> None:
    with pytest.raises(PatternError):
        parse_plate_identity(["x"], "{plate}_{plate}")


def test_advanced_regex_requires_named_plate_group() -> None:
    with pytest.raises(PatternError):
        parse_plate_identity(["x"], r"(.+)", advanced=True)
    out = parse_plate_identity(["A-1"], r"(?P<plate>[A-Z]+)-(?P<time>\d+)", advanced=True)
    assert out == [PlateMatch("A-1", "A", "1")]


def test_invalid_regex_raises_pattern_error() -> None:
    with pytest.raises(PatternError):
        parse_plate_identity(["x"], r"(?P<plate>", advanced=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_plate_pattern.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `src/phenotypic/gui/browse/_plate_pattern.py`:

```python
"""Compile a plate-identity pattern over filename stems (spec §5.3, §15.5).

Placeholder syntax (primary): ``{plate}`` (required), optional ``{time}``,
``*`` wildcard, literal text. Compiled to an anchored, non-greedy regex.
Advanced mode: a raw regex with a named ``plate`` group (``time`` optional).
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["PlateMatch", "PatternError", "parse_plate_identity"]


@dataclass(frozen=True)
class PlateMatch:
    """One stem's extracted plate identity + time (``None`` when unmatched)."""

    stem: str
    plate: str | None
    time: str | None


class PatternError(ValueError):
    """Raised for an invalid plate-identity pattern (placeholder or regex)."""


_TOKEN = re.compile(r"\{plate\}|\{time\}|\*")
_TOKEN_REGEX = {
    "{plate}": "(?P<plate>.+?)",
    "{time}": "(?P<time>.+?)",
    "*": ".*?",
}


def _compile(pattern: str, *, advanced: bool) -> re.Pattern[str]:
    if advanced:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise PatternError(f"invalid regex: {exc}") from exc
        if "plate" not in compiled.groupindex:
            raise PatternError("pattern must contain a (?P<plate>...) group")
        return compiled

    if "{plate}" not in pattern:
        raise PatternError("pattern must contain {plate}")
    if pattern.count("{plate}") > 1 or pattern.count("{time}") > 1:
        raise PatternError("duplicate {plate}/{time} token")

    parts: list[str] = []
    pos = 0
    for match in _TOKEN.finditer(pattern):
        parts.append(re.escape(pattern[pos : match.start()]))
        parts.append(_TOKEN_REGEX[match.group()])
        pos = match.end()
    parts.append(re.escape(pattern[pos:]))
    try:
        return re.compile("^" + "".join(parts) + "$")
    except re.error as exc:  # pragma: no cover - tokens are well-formed
        raise PatternError(f"could not compile pattern: {exc}") from exc


def parse_plate_identity(
    stems: Iterable[str], pattern: str, *, advanced: bool = False
) -> list[PlateMatch]:
    """Match each stem against ``pattern``; return per-stem plate/time captures.

    Args:
        stems: Filename stems (no directory, no extension).
        pattern: Placeholder (default) or raw-regex (``advanced=True``) pattern.
        advanced: When ``True``, ``pattern`` is a raw regex with a named
            ``plate`` group (``time`` optional).

    Returns:
        One :class:`PlateMatch` per stem (``plate``/``time`` are ``None`` when
        the stem does not match).

    Raises:
        PatternError: When the pattern is structurally invalid.
    """
    compiled = _compile(pattern, advanced=advanced)
    out: list[PlateMatch] = []
    for stem in stems:
        m = compiled.match(stem)
        if m is None:
            out.append(PlateMatch(stem, None, None))
            continue
        groups = m.groupdict()
        out.append(PlateMatch(stem, groups.get("plate"), groups.get("time")))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/browse/test_plate_pattern.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/browse/_plate_pattern.py tests/gui/browse/test_plate_pattern.py
git commit -m "feat(gui-timeline): plate-identity pattern parser ({plate}/{time} + regex)"
```

---

### Task 4: Browse record builder (per-axis source picker + folder-scoped CSV)

**Files:**
- Create: `src/phenotypic/gui/browse/_timeline_records.py`
- Test: `tests/gui/browse/test_timeline_records.py`

**Interfaces:**
- Consumes: `read_capture_time` (Task 2), `parse_plate_identity` (Task 3). It uses a **local** `_sandbox_rel` helper (below) rather than importing `sandbox_rel` from `browse/_callbacks.py` — `_callbacks` imports from this module (Task 8), so importing back would create a cycle. The two-line path-join is duplicated intentionally to keep `_timeline_records` import-cycle-free and pure.
- Produces:
  - `BrowseAxisConfig(row_source, time_source, pattern="", advanced_pattern=False, csv_image_col=None, row_csv_col=None, time_csv_col=None)` — frozen dataclass. `row_source ∈ {"folder","pattern","csv"}`, `time_source ∈ {"exif","pattern","csv"}`.
  - `build_browse_records(datasets, src_root_rel, config, *, csv_rows=None, capture_time_of=read_capture_time_for_rel) -> tuple[list[dict], list[str]]` → `(records, warnings)`. Each record is `{"row_value": str, "time_value": str, "cell_ref": <sandbox-rel path str>}`. `capture_time_of` maps a sandbox-rel path str → capture-time str or `None`.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/browse/test_timeline_records.py`:

```python
"""Per-axis source resolution for the Browse Timeline (pure)."""
from __future__ import annotations

from phenotypic.gui.browse._timeline_records import (
    BrowseAxisConfig,
    build_browse_records,
)


def _datasets() -> dict[str, list[str]]:
    # Two timepoint folders, each holding the same two plate filenames.
    return {
        "2024-01-01": ["plateA.tif", "plateB.tif"],
        "2024-01-02": ["plateA.tif", "plateB.tif"],
    }


def test_default_folder_rows_exif_time() -> None:
    # row=folder, time=exif (fallback to filename when EXIF missing).
    config = BrowseAxisConfig(row_source="folder", time_source="exif")
    records, warnings = build_browse_records(
        _datasets(), "src", config, capture_time_of=lambda rel: None
    )
    rows = {r["row_value"] for r in records}
    assert rows == {"2024-01-01", "2024-01-02"}
    assert warnings == []
    # cell_ref is the sandbox-relative POSIX path.
    refs = {r["cell_ref"] for r in records}
    assert "src/2024-01-01/plateA.tif" in refs


def test_pattern_rows_are_folder_scoped() -> None:
    # Flat-style names inside each folder; row=pattern {plate}, time=pattern {time}.
    datasets = {
        "runX": ["plateA_t01.tif", "plateA_t02.tif", "plateB_t01.tif"],
    }
    config = BrowseAxisConfig(
        row_source="pattern", time_source="pattern", pattern="{plate}_t{time}"
    )
    records, _ = build_browse_records(
        datasets, "src", config, capture_time_of=lambda rel: None
    )
    rows = {r["row_value"] for r in records}
    # Folder-scoped: row key is "<folder>/<plate>".
    assert rows == {"runX/plateA", "runX/plateB"}
    times = {r["time_value"] for r in records}
    assert times == {"01", "02"}


def test_csv_source_joins_by_stem_and_warns_on_cross_folder_collision() -> None:
    csv_rows = [
        {"image": "plateA", "media": "YPD", "tp": "0h"},
        {"image": "plateB", "media": "SD", "tp": "0h"},
    ]
    config = BrowseAxisConfig(
        row_source="csv",
        time_source="csv",
        csv_image_col="image",
        row_csv_col="media",
        time_csv_col="tp",
    )
    records, warnings = build_browse_records(
        _datasets(), "src", config, csv_rows=csv_rows, capture_time_of=lambda rel: None
    )
    rows = {r["row_value"] for r in records}
    assert rows == {"YPD", "SD"}
    # plateA/plateB stems each appear in TWO folders while a CSV axis is active
    # → collision warning (same stem can't disambiguate per-folder rows).
    assert any("plateA" in w or "stem" in w.lower() for w in warnings)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_timeline_records.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `src/phenotypic/gui/browse/_timeline_records.py`:

```python
"""Resolve each Browse source image to a (row, time) matrix record.

The per-axis source picker (spec §5.2): row ∈ {folder | {plate} pattern |
CSV column}; time ∈ {EXIF | {time} pattern | CSV column}. CSV joins are
folder-scoped by image **stem** (no path column, spec §15.4); pattern rows
are folder-scoped (spec §15.5). Pure — Dash wiring lives in the callbacks.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from phenotypic.gui.browse._plate_pattern import parse_plate_identity

__all__ = ["BrowseAxisConfig", "build_browse_records"]


@dataclass(frozen=True)
class BrowseAxisConfig:
    """Which source feeds each Timeline axis."""

    row_source: str          # "folder" | "pattern" | "csv"
    time_source: str         # "exif" | "pattern" | "csv"
    pattern: str = ""
    advanced_pattern: bool = False
    csv_image_col: str | None = None
    row_csv_col: str | None = None
    time_csv_col: str | None = None


_UNMATCHED = "unmatched"


def _sandbox_rel(src_root_rel: str, folder: str, filename: str) -> str:
    parts = [p for p in (src_root_rel, folder) if p and p != "."]
    return PurePosixPath(*parts, filename).as_posix() if parts else filename


def build_browse_records(
    datasets: Mapping[str, Sequence[str]],
    src_root_rel: str,
    config: BrowseAxisConfig,
    *,
    csv_rows: Sequence[Mapping[str, object]] | None = None,
    capture_time_of: Callable[[str], str | None],
) -> tuple[list[dict[str, object]], list[str]]:
    """Build matrix records + warnings for the Browse Timeline.

    Args:
        datasets: ``{dataset_folder: [filename, ...]}`` (from ``list_datasets``).
        src_root_rel: Source root relative to the sandbox (POSIX).
        config: The per-axis source selection.
        csv_rows: Parsed metadata-CSV rows (dicts), or ``None``. Required when
            either axis source is ``"csv"``.
        capture_time_of: ``sandbox_rel_path -> capture-time str | None`` used
            for the EXIF time source.

    Returns:
        ``(records, warnings)``. Each record is
        ``{"row_value": str, "time_value": str, "cell_ref": sandbox_rel}``.
        Warnings are human-readable strings for the UI (e.g. CSV stem
        collisions).
    """
    warnings: list[str] = []
    uses_csv = "csv" in (config.row_source, config.time_source)

    # CSV lookup, keyed by image STEM (matches the existing stem-join convention).
    csv_by_stem: dict[str, Mapping[str, object]] = {}
    if uses_csv and csv_rows and config.csv_image_col:
        for row in csv_rows:
            raw = row.get(config.csv_image_col)
            if raw is None:
                continue
            csv_by_stem[Path(str(raw)).stem] = row

    # Pattern matches, computed per folder so rows stay folder-scoped.
    pattern_by_folder: dict[str, dict[str, tuple[str | None, str | None]]] = {}
    uses_pattern = "pattern" in (config.row_source, config.time_source)
    if uses_pattern:
        for folder, files in datasets.items():
            stems = [Path(f).stem for f in files]
            matches = parse_plate_identity(
                stems, config.pattern, advanced=config.advanced_pattern
            )
            pattern_by_folder[folder] = {
                pm.stem: (pm.plate, pm.time) for pm in matches
            }

    # Cross-folder stem collision check (only meaningful when CSV drives an axis).
    if uses_csv:
        seen: dict[str, set[str]] = {}
        for folder, files in datasets.items():
            for filename in files:
                seen.setdefault(Path(filename).stem, set()).add(folder)
        collided = sorted(s for s, folders in seen.items() if len(folders) > 1)
        if collided:
            warnings.append(
                "CSV axis: stem(s) appear in multiple folders and cannot be "
                f"disambiguated per folder: {', '.join(collided)}"
            )

    records: list[dict[str, object]] = []
    for folder, files in datasets.items():
        for filename in files:
            stem = Path(filename).stem
            rel = _sandbox_rel(src_root_rel, folder, filename)
            plate, ptime = pattern_by_folder.get(folder, {}).get(stem, (None, None))
            csv_row = csv_by_stem.get(stem)

            row_value = _resolve_row(config, folder, plate, csv_row)
            time_value = _resolve_time(config, ptime, csv_row, rel, capture_time_of, filename)
            records.append(
                {"row_value": row_value, "time_value": time_value, "cell_ref": rel}
            )
    return records, warnings


def _resolve_row(
    config: BrowseAxisConfig,
    folder: str,
    plate: str | None,
    csv_row: Mapping[str, object] | None,
) -> str:
    if config.row_source == "folder":
        return folder
    if config.row_source == "pattern":
        if plate is None:
            return _UNMATCHED
        return plate if folder == "." else f"{folder}/{plate}"
    # csv
    if csv_row is None or config.row_csv_col is None:
        return _UNMATCHED
    return str(csv_row.get(config.row_csv_col, _UNMATCHED))


def _resolve_time(
    config: BrowseAxisConfig,
    ptime: str | None,
    csv_row: Mapping[str, object] | None,
    rel: str,
    capture_time_of: Callable[[str], str | None],
    filename: str,
) -> str:
    if config.time_source == "exif":
        return capture_time_of(rel) or filename
    if config.time_source == "pattern":
        return ptime if ptime is not None else (capture_time_of(rel) or filename)
    # csv
    if csv_row is None or config.time_csv_col is None:
        return ""
    return str(csv_row.get(config.time_csv_col, ""))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/browse/test_timeline_records.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/browse/_timeline_records.py tests/gui/browse/test_timeline_records.py
git commit -m "feat(gui-timeline): Browse per-axis record builder (folder/pattern/csv)"
```

---

### Task 5: Browse thumbnail route (token resolver via normalize_to_png)

**Files:**
- Create: `src/phenotypic/gui/browse/_thumb_routes.py`
- Test: `tests/gui/browse/test_thumb_routes.py`

**Interfaces:**
- Consumes: `register_thumbnail_route`, `ThumbUnavailable` (Phase 1); `BROWSE_THUMB_URL_SEGMENT` (Phase 1 `_config`); `_source_render` (`decode_token`, `cache_png_path`, `normalize_to_png`, `browse_cache_base`, `SourceRenderUnavailable`); `SandboxRoot`.
- Produces: `register(app: dash.Dash, sandbox: SandboxRoot) -> None` mounting `GET /<BROWSE_THUMB_URL_SEGMENT>/<token>?size=` on the browse server. The resolver decodes the token, resolves it through the sandbox, normalizes it to the cached PNG (→ `ThumbUnavailable` on RAW-undecodable), and returns that PNG path for downscaling.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/browse/test_thumb_routes.py`:

```python
"""Browse thumbnail route smoke tests (Flask test client)."""
from __future__ import annotations

import io
from pathlib import Path

import dash
from PIL import Image as PILImage

from phenotypic.gui.browse import _source_render, _thumb_routes
from phenotypic.gui.shell._sandbox import SandboxRoot


def _client(monkeypatch, tmp_path: Path):
    # Redirect the ephemeral cache into tmp_path so init_cache() never wipes
    # the real system temp dir (the established browse-test idiom).
    monkeypatch.setattr(
        _source_render.tempfile, "gettempdir", lambda: str(tmp_path / "cache")
    )
    # A source image inside the sandbox root.
    (tmp_path / "imgs").mkdir()
    src = tmp_path / "imgs" / "plateA.png"
    PILImage.new("RGB", (200, 100), (0, 128, 0)).save(src, format="PNG")

    sandbox = SandboxRoot.from_path(tmp_path)  # canonical constructor (resolves)
    _source_render.init_cache()
    app = dash.Dash(__name__)
    _thumb_routes.register(app, sandbox)
    token = _source_render.encode_token("imgs/plateA.png")
    return app.server.test_client(), token


def test_thumb_happy_path(monkeypatch, tmp_path: Path) -> None:
    client, token = _client(monkeypatch, tmp_path)
    resp = client.get(f"/thumb/{token}?size=100")  # snaps to bucket 128
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    out = PILImage.open(io.BytesIO(resp.data))
    assert max(out.size) == 128


def test_thumb_unknown_token_is_404(monkeypatch, tmp_path: Path) -> None:
    client, _token = _client(monkeypatch, tmp_path)
    resp = client.get("/thumb/not-a-real-token?size=128")
    assert resp.status_code == 404
```

> **Verified:** `SandboxRoot.from_path(root)` is the canonical constructor (`shell/_sandbox.py:42`; frozen dataclass, `root: Path`, `.resolve(candidate)` raises `ValueError` on escape). `init_cache()` → `wipe_cache()` → `shutil.rmtree(browse_cache_base())` wipes the whole `phenotypic/browse/` subtree including the `thumb/` subdir, so each test starts clean.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_thumb_routes.py -v`
Expected: FAIL with `ModuleNotFoundError: ..._thumb_routes`.

- [ ] **Step 3: Write minimal implementation**

Create `src/phenotypic/gui/browse/_thumb_routes.py`:

```python
"""Mount the Browse Timeline thumbnail route.

Thin adapter over the Phase 1 ``register_thumbnail_route`` factory: the
resolver decodes the base64url token, resolves it through the sandbox (the
sole security boundary), and normalizes it to the cached 8-bit PNG that the
factory then downscales to the requested bucket. RAW that cannot be decoded
on this platform maps to ``ThumbUnavailable`` (→ 422), mirroring the DZI route.
"""
from __future__ import annotations

import logging
from pathlib import Path

import dash

from phenotypic.gui._config import BROWSE_THUMB_URL_SEGMENT
from phenotypic.gui._shared.timeline import ThumbUnavailable, register_thumbnail_route
from phenotypic.gui.browse import _source_render
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register(app: dash.Dash, sandbox: SandboxRoot) -> None:
    """Mount the token-keyed thumbnail route on ``app.server``."""

    def resolve_source(token: str) -> Path:
        try:
            rel = _source_render.decode_token(token)
            resolved = sandbox.resolve(rel)
        except Exception as exc:  # noqa: BLE001 - malformed/escaping token → 404
            raise FileNotFoundError(token) from exc
        if not resolved.is_file():
            raise FileNotFoundError(token)
        cache_png = _source_render.cache_png_path(token)
        try:
            _source_render.normalize_to_png(resolved, cache_png)
        except _source_render.SourceRenderUnavailable as exc:
            raise ThumbUnavailable(str(exc)) from exc
        return cache_png

    register_thumbnail_route(
        app,
        segment=BROWSE_THUMB_URL_SEGMENT,
        resolve_source=resolve_source,
        cache_base=_source_render.browse_cache_base() / "thumb",
    )
    logger.debug("Registered Browse thumbnail route under /%s", BROWSE_THUMB_URL_SEGMENT)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/browse/test_thumb_routes.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/browse/_thumb_routes.py tests/gui/browse/test_thumb_routes.py
git commit -m "feat(gui-timeline): Browse thumbnail route (token → normalize → downscale)"
```

---

### Task 6: Timeline body layout + view-mode toggle

**Files:**
- Modify: `src/phenotypic/gui/browse/_layout.py`
- Test: `tests/gui/browse/test_layout.py` (append)

**Interfaces:**
- Consumes: Task 1 ids; `TIMELINE_TILE_SIZE_DEFAULT`, `TIMELINE_MOUNT_CAP`, `TIMELINE_FOCUS_MARGIN`, `TIMELINE_WARM_CONCURRENCY` (Phase 1 `_config`).
- Produces:
  - `build_timeline_body() -> Component` — the Timeline controls + **no-scroll focus-window viewport** (grid container + the four edge nav buttons + position readout) + stores + pop-out modal.
  - `build_browse_layout()` (modified) wraps the existing single-pane body in `BROWSE_SINGLE_BODY`, adds the `BROWSE_VIEW_MODE_TOGGLE` to the header, and appends `build_timeline_body()` in a `BROWSE_TIMELINE_BODY` wrapper (hidden by default).

- [ ] **Step 1: Write the failing test**

Append to `tests/gui/browse/test_layout.py`:

```python
def _walk_ids(component) -> set[str]:
    found: set[str] = set()
    stack = [component]
    while stack:
        node = stack.pop()
        cid = getattr(node, "id", None)
        if isinstance(cid, str):
            found.add(cid)
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return found


def test_layout_has_view_mode_toggle_and_both_bodies() -> None:
    from phenotypic.gui.browse._layout import build_browse_layout
    from phenotypic.gui.browse import _ids

    ids = _walk_ids(build_browse_layout())
    assert _ids.BROWSE_VIEW_MODE_TOGGLE in ids
    assert _ids.BROWSE_SINGLE_BODY in ids
    assert _ids.BROWSE_TIMELINE_BODY in ids
    assert _ids.BROWSE_TL_GRID in ids
    assert _ids.BROWSE_TL_PATTERN_INPUT in ids
    assert _ids.BROWSE_TL_TILE_SIZE_READOUT in ids
    # Focus-and-navigate chrome (spec §16): four edge buttons + position readout.
    assert _ids.BROWSE_TL_NAV_UP in ids
    assert _ids.BROWSE_TL_NAV_DOWN in ids
    assert _ids.BROWSE_TL_NAV_LEFT in ids
    assert _ids.BROWSE_TL_NAV_RIGHT in ids
    assert _ids.BROWSE_TL_POSITION in ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_layout.py::test_layout_has_view_mode_toggle_and_both_bodies -v`
Expected: FAIL (ids not present).

- [ ] **Step 3: Write minimal implementation**

In `src/phenotypic/gui/browse/_layout.py`: **add `import dash_bootstrap_components as dbc`** — it is NOT currently imported in this file (the file imports only `from dash import dcc, html` + `_design` constants); `build_timeline_body()` uses `dbc.Modal`, so the import is required or it `NameError`s at import time. Also import the new ids and `TIMELINE_TILE_SIZE_DEFAULT`, `TIMELINE_MOUNT_CAP`, `TIMELINE_FOCUS_MARGIN`, `TIMELINE_WARM_CONCURRENCY` from `_config` (**not** `TIMELINE_WINDOW_MARGIN_SCREENS` — that scroll-era constant was removed by Phase 1, spec §16.7). Add `build_timeline_body()` and wire it into `build_browse_layout()` (exact structural change in Step 3b).

```python
def build_timeline_body() -> Any:
    """Build the Timeline matrix body (hidden until the view toggle selects it)."""
    row_source = dcc.Dropdown(
        id=ids.BROWSE_TL_ROW_SOURCE,
        options=[
            {"label": "Folder", "value": "folder"},
            {"label": "Filename pattern", "value": "pattern"},
            {"label": "CSV column", "value": "csv"},
        ],
        value="folder",
        clearable=False,
        style={"minWidth": "10rem"},
    )
    time_source = dcc.Dropdown(
        id=ids.BROWSE_TL_TIME_SOURCE,
        options=[
            {"label": "EXIF capture time", "value": "exif"},
            {"label": "Filename pattern", "value": "pattern"},
            {"label": "CSV column", "value": "csv"},
        ],
        value="exif",
        clearable=False,
        style={"minWidth": "10rem"},
    )
    csv_cols = [
        dcc.Dropdown(id=ids.BROWSE_TL_ROW_CSV_COL, options=[], placeholder="Row column…"),
        dcc.Dropdown(id=ids.BROWSE_TL_TIME_CSV_COL, options=[], placeholder="Time column…"),
        dcc.Dropdown(id=ids.BROWSE_TL_CSV_IMAGE_COL, options=[], placeholder="Image-name column…"),
    ]
    pattern_controls = html.Div(
        [
            dcc.Input(
                id=ids.BROWSE_TL_PATTERN_INPUT,
                type="text",
                placeholder="{plate}_t{time}",
                debounce=True,
                style={"minWidth": "14rem"},
            ),
            dcc.Checklist(
                id=ids.BROWSE_TL_PATTERN_ADVANCED,
                options=[{"label": "regex", "value": "advanced"}],
                value=[],
            ),
            html.Div(id=ids.BROWSE_TL_PATTERN_PREVIEW, className="browse-tl-pattern-preview"),
        ],
        className="d-flex align-items-center",
        style={"gap": "0.5rem", "flexWrap": "wrap"},
    )
    tile_stepper = html.Div(
        [
            html.Button("−", id=ids.BROWSE_TL_TILE_SIZE_MINUS, n_clicks=0,
                        className="btn btn-outline-secondary btn-sm", type="button"),
            html.Span(f"{TIMELINE_TILE_SIZE_DEFAULT} px", id=ids.BROWSE_TL_TILE_SIZE_READOUT),
            html.Button("+", id=ids.BROWSE_TL_TILE_SIZE_PLUS, n_clicks=0,
                        className="btn btn-outline-secondary btn-sm", type="button"),
        ],
        className="d-flex align-items-center",
        style={"gap": "0.25rem"},
    )
    nudge = html.Div(
        "Add a metadata CSV (Settings → Metadata) for richer time × group axes.",
        id=ids.BROWSE_TL_NUDGE,
        className="alert alert-info py-1 px-2",
        style={"display": "none"},
    )
    controls = html.Div(
        [
            html.Span("Rows"), row_source,
            html.Span("Time"), time_source,
            *csv_cols,
            pattern_controls,
            tile_stepper,
        ],
        className="d-flex align-items-center",
        style={"gap": "0.5rem", "flexWrap": "wrap", "marginBottom": "0.5rem"},
    )
    # The focus-navigate constants ride as STATIC data-* on the container div
    # (they never change at runtime). timeline.js reads them off the grid
    # container; the render callback only replaces this div's *children*, so it
    # cannot set the container's own attributes — they must live here.
    # NOTE (spec §16.7): data-focus-margin REPLACES the scroll-era
    # data-margin-screens. The inner grid is positioned by the JS via a CSS
    # transform to centre the focused cell.
    # SURFACE-AGNOSTIC class `.timeline-grid-container`: `timeline.js` is
    # vendored byte-for-byte into both Browse and Results, so the controller
    # locates the grid container (and re-attaches to it) by this stable class,
    # NOT by the surface-specific id — the id stays only for the Dash render
    # callback's Output target. Each surface's clientside callback passes its
    # own container id into attach(containerId).
    grid = html.Div(
        id=ids.BROWSE_TL_GRID,
        className="timeline-grid-container",
        **{
            "data-mount-cap": str(TIMELINE_MOUNT_CAP),
            "data-focus-margin": str(TIMELINE_FOCUS_MARGIN),
            "data-warm-concurrency": str(TIMELINE_WARM_CONCURRENCY),
        },
    )
    # No-scroll focus-window VIEWPORT (spec §16.1): overflow hidden (no
    # scrollbar), bounded height, position:relative so the four edge buttons +
    # the focus position readout anchor to its edges. timeline.js centres the
    # inner grid on the focused cell via a CSS transform.
    # SURFACE-AGNOSTIC classes: the four nav buttons + readout + bridge input
    # carry stable `timeline-*` classes that `timeline.js` queries (scoped to
    # the timeline body). The Dash ids stay (server callbacks + Browse e2e
    # selectors target them); the controller never reads the ids. The
    # `browse-tl-*` classes remain for Browse-only CSS styling.
    nav_up = html.Button(
        "▲", id=ids.BROWSE_TL_NAV_UP, type="button", n_clicks=0,
        className="timeline-nav-up browse-tl-nav browse-tl-nav--up",
        **{"aria-label": "Move focus up one row"},
    )
    nav_down = html.Button(
        "▼", id=ids.BROWSE_TL_NAV_DOWN, type="button", n_clicks=0,
        className="timeline-nav-down browse-tl-nav browse-tl-nav--down",
        **{"aria-label": "Move focus down one row"},
    )
    nav_left = html.Button(
        "◀", id=ids.BROWSE_TL_NAV_LEFT, type="button", n_clicks=0,
        className="timeline-nav-left browse-tl-nav browse-tl-nav--left",
        **{"aria-label": "Move focus back one time step"},
    )
    nav_right = html.Button(
        "▶", id=ids.BROWSE_TL_NAV_RIGHT, type="button", n_clicks=0,
        className="timeline-nav-right browse-tl-nav browse-tl-nav--right",
        **{"aria-label": "Move focus forward one time step"},
    )
    # timeline.js sets this readout's text (e.g. "row 1/74 · time 1/24").
    position = html.Div(id=ids.BROWSE_TL_POSITION, className="timeline-position browse-tl-position")
    grid_viewport = html.Div(
        [grid, nav_up, nav_down, nav_left, nav_right, position],
        # `.timeline-viewport` is the surface-agnostic anchor the controller
        # walks to from the grid container (`.closest(".timeline-viewport")`);
        # `.browse-tl-viewport` stays for Browse-only CSS.
        className="timeline-viewport browse-tl-viewport",
        # tabIndex makes the viewport focusable so its scoped keyboard handler
        # (arrow keys / Enter / Space) receives events; overflow:hidden = no
        # scrollbar (focus-and-navigate, not scroll).
        tabIndex=0,
        style={
            "overflow": "hidden",
            "position": "relative",
            "height": "75vh",
            "border": "1px solid var(--color-border)",
        },
    )
    popout = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("")),
            dbc.ModalBody(html.Div(id=ids.BROWSE_TL_POPOUT_OSD, style={"height": "70vh"})),
        ],
        id=ids.BROWSE_TL_POPOUT_MODAL,
        is_open=False,
        size="xl",
    )
    return html.Div(
        [
            nudge,
            controls,
            grid_viewport,
            popout,
            # Hidden JS→Dash bridge: timeline.js sets .value to the clicked
            # cell's data-ref (token) + dispatches an input event (Task 9c).
            # SURFACE-AGNOSTIC class `.timeline-popout-bridge`: the controller
            # finds this input by class (scoped to the timeline body), so the
            # vendored timeline.js works identically on Browse + Results. The id
            # stays for the Dash server callback's Input target.
            dcc.Input(
                id=ids.BROWSE_TL_POPOUT_INPUT,
                value="",
                className="timeline-popout-bridge",
                style={"display": "none"},
            ),
            dcc.Store(id=ids.BROWSE_TL_STORE_TILE_SIZE, data=TIMELINE_TILE_SIZE_DEFAULT),
            dcc.Store(id=ids.BROWSE_TL_STORE_WARNINGS, data=[]),
            dcc.Store(id=ids.BROWSE_TL_POPOUT_STORE, data=None),
        ],
        id=ids.BROWSE_TIMELINE_BODY,
        # SURFACE-AGNOSTIC class `.timeline-body`: the controller scopes its
        # sibling-control queries (nav buttons, readout, bridge input) to the
        # enclosing element carrying this class, so the vendored timeline.js
        # never reads a surface-specific id.
        className="timeline-body",
        style={"display": "none"},  # toggled on by the view-mode callback
    )
```

**Step 3b — exact `build_browse_layout()` change.** The current function ends with:

```python
    return html.Div(
        [
            header,
            empty_hint,
            osd_stage,
            metadata_panel,
            csv_metadata_panel,
            dcc.Store(id=ids.BROWSE_DATASETS_STORE, data={}),
            dcc.Store(id=ids.BROWSE_CURRENT_IMAGE_STORE, data=None),
            dcc.Store(id=ids.BROWSE_OSD_SYNC, data=None),
        ],
        className="browse-page",
        style={"padding": "1rem"},
    )
```

Change it to (toggle appended into the existing `header` row so it's always visible; the four single-view children move into `BROWSE_SINGLE_BODY`; `build_timeline_body()` appended; the three existing stores stay as **siblings**, ids unchanged, so all existing browse callbacks keep resolving):

```python
    view_toggle = dcc.RadioItems(
        id=ids.BROWSE_VIEW_MODE_TOGGLE,
        options=[
            {"label": "Single", "value": "single"},
            {"label": "Timeline", "value": "timeline"},
        ],
        value="single",
        inline=True,
        className="browse-view-mode",
    )
    header.children.append(view_toggle)  # header is the html.Div built above

    single_body = html.Div(
        [empty_hint, osd_stage, metadata_panel, csv_metadata_panel],
        id=ids.BROWSE_SINGLE_BODY,
    )

    return html.Div(
        [
            header,
            single_body,
            build_timeline_body(),
            dcc.Store(id=ids.BROWSE_DATASETS_STORE, data={}),
            dcc.Store(id=ids.BROWSE_CURRENT_IMAGE_STORE, data=None),
            dcc.Store(id=ids.BROWSE_OSD_SYNC, data=None),
        ],
        className="browse-page",
        style={"padding": "1rem"},
    )
```

(`header.children` is the list `[dataset_row, picker_group]` built earlier in the function; appending is safe. If the local is named differently, append the toggle to that list.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/browse/test_layout.py -v`
Expected: PASS (existing layout tests + the new one).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/browse/_layout.py tests/gui/browse/test_layout.py
git commit -m "feat(gui-timeline): Browse view-mode toggle + Timeline body layout"
```

---

### Task 7: `timeline.js` — focus-and-navigate controller (spec §16)

**Files:**
- Create: `src/phenotypic/gui/browse/_assets/timeline.js`
- Test: `tests/e2e/gui/test_browse_timeline.py` (Playwright)

**Interfaces:**
- Consumes: the grid DOM produced by `build_timeline_grid` (cells under the `.timeline-grid-container` carrying `data-src`, `data-ref`, and **`data-row-index`/`data-col-index`** — both empty and populated cells, spec §16.8); the sibling controls located **by stable class scoped to the enclosing `.timeline-body`** — the four edge buttons (`.timeline-nav-up/down/left/right`), the position readout (`.timeline-position`), the no-scroll viewport (`.timeline-viewport`), and the hidden pop-out bridge input (`.timeline-popout-bridge`, used for both ⤢ clicks and Enter/Space); `window.__phenotypicAppPrefix`; Phase 1 constants surfaced to JS via static `data-*` on the grid container (`data-focus-margin`, `data-mount-cap`, `data-warm-concurrency`) — written in `build_timeline_body()` (Task 6). **No surface-specific id is read by the controller** — the only surface-specific input is the container id passed into `attach(containerId)`.
- Produces: `window.__phenotypicTimeline.attach(containerId)` — a **focus-and-navigate controller** (spec §16), idempotent: maintains focus state, computes the centered window, mounts the focused neighborhood + margin ring (offloads beyond, LRU-capped), wires keyboard + edge-button navigation, opens the pop-out on Enter/Space, and runs a generation-guarded neighborhood-first background-warm loop. Re-attaches when Dash replaces the container (mirrors `results_viewer.js`'s re-attach idiom, spec §15.7). **Surface-agnostic** so the file is vendored byte-for-byte into Results (Phase 3) — see the "Phase 3 forward-note" near the bottom.

> **Note (cross-surface portability):** the controller reads its sibling controls by **class** (`.timeline-nav-*`, `.timeline-position`, `.timeline-popout-bridge`, `.timeline-viewport`, `.timeline-grid-container`), never by a `browse-tl-*` id, so the same file runs on Browse and Results. **The Browse e2e below still selects by Browse-specific id** (`#browse-tl-nav-right`, `#browse-tl-grid`, `.browse-tl-viewport`) — that is intentional and unbroken, because Browse's elements carry **both** their Dash id and the surface-agnostic `timeline-*` class (Task 6). No e2e selector change is needed.

- [ ] **Step 1: Write the failing test**

Create `tests/e2e/gui/test_browse_timeline.py`:

```python
"""Playwright e2e: Browse Timeline focus-and-navigate controller (spec §16).

Requires PLAYWRIGHT=1 (enforced by the conftest module-skip).
"""
from __future__ import annotations

import pytest

# Tight DOM-poll budget on a fresh Werkzeug server: stochastically slow on GHA.
pytestmark = pytest.mark.ci_flaky


def test_focus_starts_on_first_populated_cell(live_browse_timeline) -> None:
    page = live_browse_timeline  # fixture: server up, Browse open, Timeline mode on
    # Cells exist immediately (placeholders carry data-src + grid coordinates)…
    page.wait_for_selector(".timeline-cell[data-src]")
    # …and exactly one cell is focused — the first populated cell (top-left of
    # the ordered matrix: smallest row-index, then col-index).
    page.wait_for_selector(".timeline-cell--focused")
    focused = page.eval_on_selector_all(".timeline-cell--focused", "els => els.length")
    assert focused == 1
    # That focused cell is the smallest-coordinate populated cell.
    coord = page.eval_on_selector(
        ".timeline-cell--focused",
        "el => el.getAttribute('data-row-index') + ',' + el.getAttribute('data-col-index')",
    )
    assert coord == "0,0"  # seeded fixture has a populated (0,0) cell


def test_arrow_right_moves_focus_and_mounts_new_neighborhood(live_browse_timeline) -> None:
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell--focused")
    # Focus the viewport, then press ArrowRight → focus advances one column.
    # Click the focusable viewport wrapper (tabIndex=0) so its scoped keydown
    # listener receives the event (the inner #browse-tl-grid is not focusable).
    page.click(".browse-tl-viewport")
    page.keyboard.press("ArrowRight")
    page.wait_for_function(
        "document.querySelector('.timeline-cell--focused')"
        ".getAttribute('data-col-index') === '1'"
    )
    # The newly-near neighborhood mounts <img> (focus + margin ring).
    page.wait_for_function(
        "document.querySelectorAll('#browse-tl-grid .timeline-cell img').length > 0"
    )


def test_edge_button_right_moves_focus(live_browse_timeline) -> None:
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell--focused")
    page.click("#browse-tl-nav-right")
    page.wait_for_function(
        "document.querySelector('.timeline-cell--focused')"
        ".getAttribute('data-col-index') === '1'"
    )


def test_far_cell_is_not_mounted_window_is_bounded(live_browse_timeline) -> None:
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell--focused")
    total = page.eval_on_selector_all(".timeline-cell[data-src]", "els => els.length")
    mounted = page.eval_on_selector_all(
        "#browse-tl-grid .timeline-cell img", "els => els.length"
    )
    # Bounded window: only the focused neighborhood + margin ring mounts, NEVER
    # every cell. (Seed enough cells that total > the visible window + margin.)
    assert 0 < mounted < total


def test_offscreen_margin_ring_is_pre_mounted(live_browse_timeline) -> None:
    # User-required smooth-UX behaviour (spec §16.3): a cell JUST OUTSIDE the
    # visible window but WITHIN data-focus-margin must already carry an <img>
    # (pre-mounted), so a quick step into it is instant.
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell--focused")
    # A cell within the focus margin but off-screen is mounted; assert at least
    # one mounted <img> sits outside the viewport's visible rectangle.
    off_screen_mounted = page.evaluate(
        """() => {
            const vp = document.querySelector('.browse-tl-viewport').getBoundingClientRect();
            const imgs = document.querySelectorAll('#browse-tl-grid .timeline-cell img');
            let n = 0;
            imgs.forEach((img) => {
                const r = img.getBoundingClientRect();
                const visible = r.right > vp.left && r.left < vp.right
                    && r.bottom > vp.top && r.top < vp.bottom;
                if (!visible) { n += 1; }
            });
            return n;
        }"""
    )
    assert off_screen_mounted >= 1  # the margin ring pre-mounted at least one
```

> **Fixture recipe (no existing browse e2e to copy — write it fresh; source-store seeding via the PROVEN sidebar-tree-click idiom).** `tests/e2e/gui/conftest.py` provides `fake_sandbox`, `live_server`, and `hub_url` (verified — there are NO `live_browse_*` fixtures and no browse e2e yet). The shared source root is **not** seedable by raw localStorage injection: the `dcc.Store(storage_type="local")` keys localStorage by the **id VALUE** `"shell-source-image-root-store"` (`shell/_ids.py:47`), NOT the Python constant name, and `resolve_source_image_root` (`shell/_source_context.py:91`) **rejects** any payload lacking `version == SOURCE_PAYLOAD_VERSION`, `validated is True`, and a string `abs_path` — it ignores `rel_path` entirely. So a `{"rel_path": "imgs"}` blob under the wrong key is silently discarded. Drive the **sidebar tree click** instead — the one proven idiom in the repo (`tests/e2e/gui/test_shared_source_root.py::_select_plate1_source`: visit a page, wait for `#shell-sidebar-tree`, click the seeded folder's `shell-sidebar-entry` button, and assert `#shell-source-image-root-label` updates). The shared `fake_sandbox` already seeds `plate1/image.tif` (`conftest.py` `_build_sandbox`); seed the timeline matrix **under that same `plate1` root** so the existing sidebar entry selects it:
>
> ```python
> @pytest.fixture()
> def live_browse_timeline(fake_sandbox, live_server, hub_url, page):
>     # 1. Seed ≥3 sub-folders × 3 PNGs UNDER the shared fixture's `plate1`
>     #    source root, so the matrix exceeds the centered focus window + margin
>     #    ring (far cells stay unmounted; the bounded-window assertions are
>     #    meaningful). At the default tile size a 3×3 matrix is small; seed more
>     #    (e.g. 6 folders × 6 names) if the window swallows the whole matrix —
>     #    confirm `total > mounted` holds. `fake_sandbox` is a Path (the sandbox
>     #    root) per conftest `_build_sandbox`.
>     from PIL import Image as PILImage
>     plate1 = fake_sandbox / "plate1"
>     for folder in ("t0", "t1", "t2"):
>         d = plate1 / folder
>         d.mkdir(parents=True, exist_ok=True)
>         for name in ("plateA.png", "plateB.png", "plateC.png"):
>             PILImage.new("RGB", (300, 200), (40, 80, 120)).save(d / name)
>     # 2. Select `plate1` as the shared source via the sidebar tree (the proven
>     #    idiom — see test_shared_source_root._select_plate1_source). This sets
>     #    the source-image-root store + label through the real UI path, so the
>     #    Browse dataset callback sees a validated payload.
>     page.goto(hub_url + "/run/")
>     page.wait_for_selector("#shell-sidebar-tree", timeout=10_000)
>     page.click('button[id*="\\"path\\":\\"plate1\\""][id*="shell-sidebar-entry"]')
>     # 3. Open the Browse mount (it reads the shared source from the store).
>     page.goto(hub_url + "/browse/")
>     # 4. Switch to Timeline mode.
>     page.click("text=Timeline")
>     # 5. Validate the fixture renders a non-empty grid before any test relies
>     #    on it.
>     page.wait_for_selector(".timeline-cell[data-src]", timeout=10_000)
>     return page
> ```
>
> **Optional injection fallback** (only if the UI click proves brittle): set localStorage under the correct key `"shell-source-image-root-store"` with a payload built by `source_payload_from_path(sandbox, plate1, source="manual")` (`shell/_source_context.py:42` — returns the `{version, validated, abs_path, rel_path, source, …}` dict the resolver accepts), JSON-encode it, then `page.reload()`. The UI-click path above is **primary**; this is the documented escape hatch, not the default.

- [ ] **Step 2: Run test to verify it fails**

Run: `PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_browse_timeline.py -v`
Expected: FAIL (no `timeline.js`; no cell is focused, arrows/buttons move nothing, no `<img>` ever mounts).

- [ ] **Step 3: Write minimal implementation**

Create `src/phenotypic/gui/browse/_assets/timeline.js`. The controller is a
focus-and-navigate model (spec §16): the matrix is **not scrollable**; one cell
is always focused; a centered window of tiles renders around it (the inner grid
is shifted by a CSS `transform`); ←/→/↑/↓ and the four edge buttons move focus
(clamped, no wrap); the focused neighborhood + a `data-focus-margin` ring mounts
`<img>` (everything beyond offloads, LRU-capped); Enter/Space opens the pop-out
for the focused cell; the background warm sweeps outward in expanding rings from
the focus.

```javascript
/*
 * timeline.js — focus-and-navigate controller (spec §16) for the timeline
 * matrix. The matrix is NOT scrollable: a no-scroll viewport renders a
 * CENTERED window around exactly one FOCUSED cell, the inner grid shifted by a
 * CSS transform. ←/→/↑/↓ and the four on-edge ◀▶▲▼ buttons move focus
 * (clamped to matrix bounds, no wrap). The focused neighborhood + a margin ring
 * (data-focus-margin) mounts <img>; cells beyond offload (img.remove) to bound
 * decoded-image memory, with data-mount-cap as the absolute LRU ceiling.
 * Enter/Space opens the pop-out for the focused cell. A generation-guarded warm
 * loop pre-fetches thumbnails NEIGHBORHOOD-FIRST (expanding rings from focus).
 * Cells are addressed by [data-row-index][data-col-index]; every grid
 * coordinate (empty or populated) is addressable (spec §16.8).
 *
 * SURFACE-AGNOSTIC: this file is vendored BYTE-FOR-BYTE into both the Browse
 * (`browse/_assets/`) and Results (`results_viewer/_assets/`) Dash apps — which
 * are SEPARATE apps mounted via DispatcherMiddleware (shell/_app.py::compose_hub),
 * each with its own assets_folder, so window.__phenotypicTimeline never collides
 * across surfaces. The controller locates its sibling controls — the four nav
 * buttons, the position readout, the hidden pop-out bridge input — by STABLE
 * CLASS scoped to the enclosing `.timeline-body`, NEVER by a surface-specific
 * id. The ONLY surface-specific input is the container id, passed as the
 * attach(containerId) parameter by each surface's clientside callback. A CI
 * byte-equality guard enforces the two vendored copies never drift.
 */
(function () {
    "use strict";
    const ns = (window.__phenotypicTimeline = window.__phenotypicTimeline || {});
    ns._generation = ns._generation || 0;
    ns._mounted = ns._mounted || [];          // LRU order of mounted cells
    ns._focus = ns._focus || { rowIndex: 0, colIndex: 0 };

    function num(el, attr, dflt) {
        const v = parseFloat(el.getAttribute(attr));
        return Number.isFinite(v) ? v : dflt;
    }

    // --- Surface-agnostic sibling-control lookup ---------------------------
    // The controls live as siblings inside the enclosing `.timeline-body`; find
    // them by stable class scoped to that body so this file is portable across
    // Browse and Results (never a surface-specific id). `body(container)` walks
    // up to the enclosing timeline body (falls back to document if absent).
    function body(container) {
        return (container && container.closest)
            ? (container.closest(".timeline-body") || document) : document;
    }
    function ctrl(container, selector) {
        return body(container).querySelector(selector);
    }

    // --- Grid geometry -----------------------------------------------------
    // Cells (populated AND empty) carry data-row-index/data-col-index. The
    // inner grid carries `.timeline-grid-container`; the no-scroll viewport is
    // its enclosing `.timeline-viewport`. Cells are addressed by coordinate,
    // not DOM order.
    function cellAt(container, r, c) {
        return container.querySelector(
            '.timeline-cell[data-row-index="' + r + '"][data-col-index="' + c + '"]'
        );
    }
    function bounds(container) {
        let maxRow = 0, maxCol = 0;
        container.querySelectorAll("[data-row-index]").forEach(function (el) {
            maxRow = Math.max(maxRow, parseInt(el.getAttribute("data-row-index"), 10) || 0);
            maxCol = Math.max(maxCol, parseInt(el.getAttribute("data-col-index"), 10) || 0);
        });
        return { maxRow: maxRow, maxCol: maxCol };
    }
    function firstPopulatedCell(container) {
        // Smallest row-index, then col-index, among populated (data-src) cells.
        const cells = Array.from(container.querySelectorAll(".timeline-cell[data-src]"));
        let best = null;
        cells.forEach(function (el) {
            const r = parseInt(el.getAttribute("data-row-index"), 10) || 0;
            const c = parseInt(el.getAttribute("data-col-index"), 10) || 0;
            if (!best || r < best.rowIndex || (r === best.rowIndex && c < best.colIndex)) {
                best = { rowIndex: r, colIndex: c };
            }
        });
        return best || { rowIndex: 0, colIndex: 0 };
    }

    // visibleHalfCols/Rows: how many cells fit each side of centre at the
    // current rendered tile size. Read the focused cell's box (incl. CSS gap)
    // and the viewport box; fall back to a small default if unmeasurable.
    function visibleHalf(container, viewport) {
        const sample = container.querySelector(".timeline-cell");
        const vp = viewport.getBoundingClientRect();
        if (!sample) { return { halfCols: 2, halfRows: 2 }; }
        const box = sample.getBoundingClientRect();
        const w = box.width || 1, h = box.height || 1;
        return {
            halfCols: Math.max(1, Math.floor(vp.width / w / 2)),
            halfRows: Math.max(1, Math.floor(vp.height / h / 2)),
        };
    }

    // --- Mount / offload ---------------------------------------------------
    function mount(cell) {
        if (!cell || cell.querySelector("img")) { return; }
        const src = cell.getAttribute("data-src");
        if (!src) { return; }            // empty placeholder — nothing to mount
        const img = document.createElement("img");
        img.src = src;
        img.className = "timeline-cell-img";
        img.loading = "lazy";
        cell.insertBefore(img, cell.firstChild);
        ns._mounted.push(cell);
    }
    function offload(cell) {
        const img = cell && cell.querySelector("img");
        if (img) { img.remove(); }
    }

    // Mount the focus window + margin ring; offload everything outside it.
    function syncWindow(container, viewport, focusMargin, cap) {
        const { halfCols, halfRows } = visibleHalf(container, viewport);
        const colReach = halfCols + focusMargin;
        const rowReach = halfRows + focusMargin;
        const f = ns._focus;
        container.querySelectorAll(".timeline-cell[data-src]").forEach(function (cell) {
            const r = parseInt(cell.getAttribute("data-row-index"), 10) || 0;
            const c = parseInt(cell.getAttribute("data-col-index"), 10) || 0;
            const inWindow = Math.abs(r - f.rowIndex) <= rowReach
                && Math.abs(c - f.colIndex) <= colReach;
            if (inWindow) { mount(cell); } else { offload(cell); }
        });
        // LRU ceiling: never exceed data-mount-cap, even if the window does.
        while (ns._mounted.length > cap) {
            const old = ns._mounted.shift();
            offload(old);
        }
    }

    // Position the inner grid via a CSS transform (no scrollbar —
    // overflow:hidden on the viewport). CLAMP-TRANSLATE (spec §16.1, OQ-1
    // default): translate to centre the focused cell, BUT clamp so the grid
    // never pulls past its own edges — keeps the window full (no empty gutters)
    // including at the default focus (0,0) and at matrix corners; the focused
    // cell then sits off-centre near edges, which the .timeline-cell--focused
    // highlight keeps unambiguous.
    function recenter(container, viewport) {
        const focused = cellAt(container, ns._focus.rowIndex, ns._focus.colIndex);
        if (!focused) { return; }
        const vp = viewport.getBoundingClientRect();
        const gr = container.getBoundingClientRect();
        const fb = focused.getBoundingClientRect();
        // Focused-cell centre relative to the grid's own box. fb/gr both carry
        // the current transform, so (fb - gr) is transform-invariant.
        const cellCenterX = (fb.left + fb.width / 2) - gr.left;
        const cellCenterY = (fb.top + fb.height / 2) - gr.top;
        // Ideal "centre the cell" translation.
        let dx = vp.width / 2 - cellCenterX;
        let dy = vp.height / 2 - cellCenterY;
        // Clamp so the grid edge never crosses into the viewport interior.
        // Allowed dx range: [vp.width - gridWidth, 0] (right edge ≥ viewport
        // right, left edge ≤ viewport left). When the grid is smaller than the
        // viewport, the range collapses and we pin to 0 (top-left aligned).
        const gridWidth = gr.width, gridHeight = gr.height;
        const minDx = Math.min(0, vp.width - gridWidth);
        const minDy = Math.min(0, vp.height - gridHeight);
        dx = Math.max(minDx, Math.min(0, dx));
        dy = Math.max(minDy, Math.min(0, dy));
        container.style.transform = "translate(" + dx + "px, " + dy + "px)";
    }

    function highlight(container) {
        container.querySelectorAll(".timeline-cell--focused").forEach(function (el) {
            el.classList.remove("timeline-cell--focused");
        });
        const focused = cellAt(container, ns._focus.rowIndex, ns._focus.colIndex);
        if (focused) { focused.classList.add("timeline-cell--focused"); }
    }

    function updateReadout(container) {
        const readout = ctrl(container, ".timeline-position");
        if (!readout) { return; }
        const b = bounds(container);
        const focused = cellAt(container, ns._focus.rowIndex, ns._focus.colIndex);
        const rowVal = focused ? (focused.getAttribute("data-row") || "") : "";
        const colVal = focused ? (focused.getAttribute("data-col") || "") : "";
        readout.textContent =
            "row " + (ns._focus.rowIndex + 1) + "/" + (b.maxRow + 1)
            + " · time " + (ns._focus.colIndex + 1) + "/" + (b.maxCol + 1)
            + (rowVal || colVal ? "  (" + rowVal + " · " + colVal + ")" : "");
    }

    function toggleEdgeButtons(container) {
        const b = bounds(container);
        const f = ns._focus;
        const set = function (cls, disabled) {
            const el = ctrl(container, cls);   // by class, scoped to the body
            if (el) { el.disabled = !!disabled; }
        };
        set(".timeline-nav-up", f.rowIndex <= 0);
        set(".timeline-nav-down", f.rowIndex >= b.maxRow);
        set(".timeline-nav-left", f.colIndex <= 0);
        set(".timeline-nav-right", f.colIndex >= b.maxCol);
    }

    // One place that re-renders everything after a focus change.
    function applyFocus(container, viewport, focusMargin, cap) {
        syncWindow(container, viewport, focusMargin, cap);
        recenter(container, viewport);
        highlight(container);
        updateReadout(container);
        toggleEdgeButtons(container);
    }

    function moveFocus(container, viewport, focusMargin, cap, dRow, dCol) {
        const b = bounds(container);
        const f = ns._focus;
        f.rowIndex = Math.min(b.maxRow, Math.max(0, f.rowIndex + dRow));   // clamp
        f.colIndex = Math.min(b.maxCol, Math.max(0, f.colIndex + dCol));   // no wrap
        applyFocus(container, viewport, focusMargin, cap);
    }

    // Enter/Space → open the pop-out for the focused cell, using the SAME hidden
    // bridge input the ⤢ click uses (Task 9): set the bridge `.value` to the
    // focused cell's data-ref + dispatch an 'input' event. The bridge is found
    // by class (`.timeline-popout-bridge`, scoped to the body), so this path is
    // identical on Browse + Results.
    function openFocusedPopout(container) {
        const focused = cellAt(container, ns._focus.rowIndex, ns._focus.colIndex);
        if (!focused) { return; }
        const ref = focused.getAttribute("data-ref");
        const bridge = ctrl(container, ".timeline-popout-bridge");
        if (!ref || !bridge) { return; }
        bridge.value = ref;
        bridge.dispatchEvent(new Event("input", { bubbles: true }));
    }

    // --- Background warm — neighborhood-first (expanding rings from focus) ---
    function warm(container, viewport, generation) {
        const concurrency = num(container, "data-warm-concurrency", 2);
        const b = bounds(container);
        const f = ns._focus;
        // Order populated cells by Chebyshev distance from the focus, so the
        // cells the user is most likely to reach next warm first (spec §16.3).
        const cells = Array.from(container.querySelectorAll(".timeline-cell[data-src]"));
        cells.sort(function (a, e) {
            const da = Math.max(
                Math.abs((parseInt(a.getAttribute("data-row-index"), 10) || 0) - f.rowIndex),
                Math.abs((parseInt(a.getAttribute("data-col-index"), 10) || 0) - f.colIndex)
            );
            const de = Math.max(
                Math.abs((parseInt(e.getAttribute("data-row-index"), 10) || 0) - f.rowIndex),
                Math.abs((parseInt(e.getAttribute("data-col-index"), 10) || 0) - f.colIndex)
            );
            return da - de;
        });
        void b;
        let i = 0;
        function pump() {
            if (generation !== ns._generation) { return; }   // matrix rebuilt → abort
            while (i < cells.length) {
                const src = cells[i++].getAttribute("data-src");
                if (!src) { continue; }
                fetch(src, { credentials: "same-origin" })
                    .catch(function () {})
                    .then(function () { if (generation === ns._generation) pump(); });
                return;
            }
        }
        for (let k = 0; k < concurrency; k++) { pump(); }
    }

    // --- Attach ------------------------------------------------------------
    ns.attach = function (containerId) {
        const container = document.getElementById(containerId);
        if (!container) { return; }
        // Surface-agnostic: resolve the no-scroll viewport by stable class.
        const viewport = container.closest(".timeline-viewport") || container.parentNode;
        // First-paint resilience (OQ-2/W2): the timeline body starts
        // display:none, so on the first attach getBoundingClientRect() can read
        // 0 (window mis-sizes, transform clamps to nothing). Re-schedule via
        // requestAnimationFrame until the viewport has a non-zero width, so the
        // controller self-corrects regardless of whether the toggle callback
        // fired attach before or after the body was shown (belt-and-suspenders
        // with the re-attach observer below).
        if (viewport.getBoundingClientRect().width === 0) {
            window.requestAnimationFrame(function () { ns.attach(containerId); });
            return;
        }
        ns._generation += 1;
        const generation = ns._generation;
        const cap = num(container, "data-mount-cap", 400);
        const focusMargin = num(container, "data-focus-margin", 2);
        ns._mounted = [];
        ns._focus = firstPopulatedCell(container);   // start at first populated cell

        applyFocus(container, viewport, focusMargin, cap);

        // Keyboard: scoped to the viewport, but IGNORED when a text input /
        // select / textarea holds focus (so typing a pattern never navigates).
        if (!viewport.dataset.tlKeysBound) {
            viewport.dataset.tlKeysBound = "1";
            viewport.addEventListener("keydown", function (ev) {
                const ae = document.activeElement;
                const tag = ae && ae.tagName ? ae.tagName.toUpperCase() : "";
                if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") { return; }
                const cur = document.getElementById(containerId);
                if (!cur) { return; }
                if (ev.key === "ArrowLeft") { moveFocus(cur, viewport, focusMargin, cap, 0, -1); ev.preventDefault(); }
                else if (ev.key === "ArrowRight") { moveFocus(cur, viewport, focusMargin, cap, 0, 1); ev.preventDefault(); }
                else if (ev.key === "ArrowUp") { moveFocus(cur, viewport, focusMargin, cap, -1, 0); ev.preventDefault(); }
                else if (ev.key === "ArrowDown") { moveFocus(cur, viewport, focusMargin, cap, 1, 0); ev.preventDefault(); }
                else if (ev.key === "Enter" || ev.key === " ") { openFocusedPopout(cur); ev.preventDefault(); }
            });
        }

        // Edge buttons — found by stable class scoped to the body (idempotent
        // bind via a dataset flag on each button), so the binding is portable
        // across surfaces.
        const bindNav = function (cls, dRow, dCol) {
            const btn = ctrl(container, cls);
            if (!btn || btn.dataset.tlNavBound === "1") { return; }
            btn.dataset.tlNavBound = "1";
            btn.addEventListener("click", function () {
                const cur = document.getElementById(containerId);
                if (cur) { moveFocus(cur, viewport, focusMargin, cap, dRow, dCol); }
            });
        };
        bindNav(".timeline-nav-up", -1, 0);
        bindNav(".timeline-nav-down", 1, 0);
        bindNav(".timeline-nav-left", 0, -1);
        bindNav(".timeline-nav-right", 0, 1);

        // Hover-revealed ⤢ click bridge (shares the same hidden input as
        // Enter/Space — see Task 9). Delegated, idempotent.
        if (!container.dataset.tlPopoutBound) {
            container.dataset.tlPopoutBound = "1";
            container.addEventListener("click", function (ev) {
                const btn = ev.target && ev.target.closest
                    ? ev.target.closest(".timeline-cell-popout") : null;
                if (!btn) { return; }
                const cell = btn.closest(".timeline-cell");
                const ref = cell && cell.getAttribute("data-ref");
                const bridge = ctrl(container, ".timeline-popout-bridge");
                if (ref && bridge) {
                    bridge.value = ref;
                    bridge.dispatchEvent(new Event("input", { bubbles: true }));
                }
            });
        }

        warm(container, viewport, generation);
    };

    // Cancel any in-flight background warm (W4): the view-mode toggle calls this
    // when switching AWAY from Timeline so an in-flight neighborhood-first warm
    // stops asking the server to render thumbnails the user no longer sees.
    // Bumping the generation makes the running pump() loops bail (they guard on
    // `generation !== ns._generation`). A later attach() bumps again and
    // restarts warm, so this is safe to call repeatedly.
    ns.cancelWarm = function () {
        ns._generation += 1;
    };

    // Re-attach when Dash replaces the container (tab/body re-render), mirroring
    // results_viewer.js: poll-until-present + a <body> MutationObserver, both
    // idempotent (the dataset flags above make a re-attach cheap). spec §15.7.
    // Surface-agnostic: discover the grid container by the stable class
    // `.timeline-grid-container` and re-attach using its OWN id — so the
    // byte-identical vendored copy finds Browse's `browse-tl-grid` and Results'
    // `timeline-grid` (or whatever id that surface assigns) without hardcoding.
    function startReattachObserver() {
        if (!document.body || ns._reattachBound) { return; }
        ns._reattachBound = true;
        const obs = new MutationObserver(function () {
            const grid = document.querySelector(".timeline-grid-container");
            if (grid && grid.id) {
                ns.attach(grid.id);
            }
        });
        obs.observe(document.body, { childList: true, subtree: true });
    }
    startReattachObserver();
})();
```

The render callback (Task 8) calls `window.__phenotypicTimeline.attach("browse-tl-grid")` via a clientside callback after each grid render (passing **Browse's** container id — the one surface-specific input); the re-attach observer above also fires when Dash replaces the container, discovering it by the surface-agnostic `.timeline-grid-container` class.

- [ ] **Step 4: Run test to verify it passes**

Run: `PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_browse_timeline.py -v`
Expected: PASS (focus starts on the first populated cell; ArrowRight + the right
edge button advance the focus highlight and mount the new neighborhood; a far
cell stays unmounted while the off-screen margin ring is pre-mounted). If the
timing budget flakes on CI only, keep the `ci_flaky` marker per `tests/CLAUDE.md`.

> **Invariant dependency (C3):** the controller's focus clamping
> (`bounds()` → `[0, maxRow]`/`[0, maxCol]`) and grid-coordinate addressing
> (`cellAt`, `firstPopulatedCell`) are correct **only because Phase 1 emits
> `data-row-index`/`data-col-index` on EVERY cell — empty cells included**
> (spec §16.8). If empty cells lacked coordinates, `bounds()` would under-count
> the matrix extent and focus could not land on (or pass through) an empty
> `(row, time)` cell. The guarding test is Phase 1's
> `tests/gui/_shared/timeline/test_grid.py::test_cells_carry_grid_coordinate_indices`
> (it asserts empty placeholders carry both indices). If that Phase 1 test is
> ever weakened to non-empty cells only, this controller breaks — keep them coupled.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/browse/_assets/timeline.js tests/e2e/gui/test_browse_timeline.py tests/e2e/gui/conftest.py
git commit -m "feat(gui-timeline): timeline.js focus-and-navigate controller"
```

---

### Task 8: Browse Timeline callbacks (render grid, pattern preview, toggle, tile-size)

**Files:**
- Modify: `src/phenotypic/gui/browse/_callbacks.py`
- Test: `tests/gui/browse/test_timeline_callbacks_helpers.py` (pure helpers) + extend `tests/e2e/gui/test_browse_timeline.py` (wiring)

**Interfaces:**
- Consumes: Tasks 1–7; `build_browse_records`/`BrowseAxisConfig` (Task 4), `read_capture_time` (Task 2), `parse_plate_identity` (Task 3), `build_matrix`/`build_timeline_grid` (Phase 1), `step_colony_tile_size`-style stepping (reuse `phenotypic.gui._config` pattern — add `step_timeline_tile_size` mirroring it), `snap_thumb_bucket` (Phase 1), `BROWSE_THUMB_URL_SEGMENT`.
- Produces: pure helpers `timeline_thumb_url(prefix, token, fetch_size) -> str`, `render_timeline_grid(records, *, display_size, prefix) -> Component`, `pattern_preview_rows(datasets, pattern, advanced) -> Component`; plus the Dash callbacks (view-mode toggle visibility, grid render, pattern live preview, tile-size stepper, CSV-column option population, nudge visibility) registered in `register_callbacks`.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/browse/test_timeline_callbacks_helpers.py`:

```python
"""Pure helpers behind the Browse Timeline callbacks."""
from __future__ import annotations

from phenotypic.gui.browse._callbacks import (
    render_timeline_grid,
    timeline_thumb_url,
)


def test_thumb_url_targets_browse_thumb_segment_with_bucket() -> None:
    url = timeline_thumb_url("/browse/", "TOKEN", 128)
    assert url == "/browse/thumb/TOKEN?size=128"


def test_render_timeline_grid_returns_component_for_records() -> None:
    records = [
        {"row_value": "r1", "time_value": "1", "cell_ref": "TOKEN_A"},
        {"row_value": "r1", "time_value": "2", "cell_ref": "TOKEN_B"},
    ]
    component = render_timeline_grid(records, display_size=120, prefix="/browse/")
    # build_timeline_grid returns (component, grid_order); render_* returns the
    # component only, ready to drop into BROWSE_TL_GRID.
    assert component is not None
    assert hasattr(component, "children")
```

> **Note:** `cell_ref` here is the **thumbnail identity** (the base64url token). `render_timeline_grid` must convert each record's sandbox-rel `cell_ref` to a token via `_source_render.encode_token` before building the URL — confirm whether `build_browse_records` should emit the token directly (simpler) or the sandbox-rel path (then encode here). **Pick one and keep it consistent with Task 4's `cell_ref` contract**; this plan assumes `build_browse_records` emits the **sandbox-rel path** and `render_timeline_grid` encodes it. Update the Task 4 records doc if you change this.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_timeline_callbacks_helpers.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Add the pure helpers + `step_timeline_tile_size` (mirror `step_colony_tile_size` in `_config.py` using the `TIMELINE_TILE_SIZE_*` constants — **and add `step_timeline_tile_size` to `_config.py`'s `__all__`**).

**CSV reading (resolves the "from the CSV header" gap).** `SHELL_METADATA_CSV_STORE` holds a path *payload* (`MetadataCsvPayload`), not rows, and the only reader (`shell/_metadata_context._read_rows`) is private and returns rows only — there is no public column accessor. Add a public helper to `shell/_metadata_context.py`:

```python
def read_metadata_csv_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return ``(column_names, rows)`` for a metadata CSV.

    Decoded with ``utf-8-sig`` so an Excel-authored BOM is stripped and never
    prefixes a ``﻿`` onto the first column name (which would silently
    break the ``csv_image_col`` join). Matches ``_read_rows`` (:206).
    """
    import csv
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
        return columns, rows
```

> **`utf-8-sig`, not `utf-8` (C4/W5):** the existing private reader
> `_metadata_context._read_rows` opens with `encoding="utf-8-sig"`
> (`_metadata_context.py:206`) precisely because Excel CSV exports carry a
> UTF-8 BOM; plain `utf-8` leaves a `﻿` on the first header name, so a
> `csv_image_col="image"` lookup misses (`"﻿image" != "image"`) and the
> CSV join silently yields all-`unmatched` rows. Keep the new public helper
> aligned with the private one.

The CSV-sourced callbacks resolve the path via the existing `resolve_metadata_csv(sandbox, payload)` (`_metadata_context.py:106`), then call `read_metadata_csv_table(path)` — `columns` populate the `BROWSE_TL_ROW_CSV_COL`/`BROWSE_TL_TIME_CSV_COL`/`BROWSE_TL_CSV_IMAGE_COL` dropdown options, and `rows` feed `build_browse_records(..., csv_rows=rows)`. Add a 2-test unit check for `read_metadata_csv_table`: (a) a plain tmp CSV → expected columns + rows; (b) a **BOM-bearing** CSV (write the file with `encoding="utf-8-sig"`, or prepend `﻿` to the header) and assert the first column name has **no** leading `﻿` — this is the regression guard for the Excel-BOM join break.

Helpers in `_callbacks.py`:

```python
from functools import partial

from phenotypic.gui._config import BROWSE_THUMB_URL_SEGMENT, snap_thumb_bucket
from phenotypic.gui._shared.timeline import build_matrix, build_timeline_grid
from phenotypic.gui.browse import _source_render


def timeline_thumb_url(prefix: str, token: str, fetch_size: int) -> str:
    """Build a thumbnail ``<img>`` URL for the Browse thumb route."""
    return f"{prefix}{BROWSE_THUMB_URL_SEGMENT}/{token}?size={fetch_size}"


def render_timeline_grid(records, *, display_size: int, prefix: str):
    """Build matrix → grid component (encoding each cell_ref to a thumb token)."""
    fetch_size = snap_thumb_bucket(display_size)

    def _url_builder(cell_ref, fetch):
        token = _source_render.encode_token(str(cell_ref))
        return timeline_thumb_url(prefix, token, fetch)

    matrix = build_matrix(records)
    component, _grid_order = build_timeline_grid(
        matrix, url_builder=_url_builder, display_size=display_size, fetch_size=fetch_size
    )
    return component
```

The callbacks (registered in `register_callbacks(app, sandbox)`): a server callback that, on any control/store change while in Timeline mode, resolves the source root + datasets (reuse the existing `_src_root_rel` + `_source_lister.list_datasets`), reads the CSV (when a CSV source is selected) from `SHELL_METADATA_CSV_STORE`, builds `BrowseAxisConfig`, calls `build_browse_records` (passing `capture_time_of` = a closure resolving each rel via `sandbox.resolve` + `read_capture_time`), and outputs `render_timeline_grid(...)` into `BROWSE_TL_GRID`'s **children** plus the warnings into `BROWSE_TL_STORE_WARNINGS`; a clientside callback that calls `window.__phenotypicTimeline.attach("browse-tl-grid")` after the grid updates (the controller reads the focus-margin/mount-cap/warm-concurrency off the container, resets focus to the first populated cell, and re-renders the centered window); the view-mode toggle callback — show/hide `BROWSE_SINGLE_BODY`/`BROWSE_TIMELINE_BODY`, and (W4) make it a **clientside** callback (or pair the server visibility callback with a clientside companion) so that, when the toggle moves **away** from `"timeline"`, it calls `window.__phenotypicTimeline.cancelWarm()` to bump the warm generation and stop any in-flight background warm (and when it moves **to** `"timeline"`, re-`attach` so the window re-renders after the body is shown — which dovetails with the first-paint `requestAnimationFrame` guard); the tile-size stepper (reusing the `step_timeline_tile_size` helper, mirroring `_step_colony_tile_size`); the pattern live-preview callback (`pattern_preview_rows`); CSV-column dropdown population from the CSV header; and the nudge visibility (shown only when no CSV is loaded). **Do not** set `data-focus-margin`/`data-mount-cap`/`data-warm-concurrency` here — they are **static** attrs on the `BROWSE_TL_GRID` container, written once in `build_timeline_body()` (Task 6); the render callback only replaces the container's children, so it cannot (and must not) set the container's own attributes.

> Keep each callback body thin and delegate to a module-level pure helper (project rule: GUI callbacks must be unit-testable; see memory `gui_review_verify_with_browser`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/gui/browse/test_timeline_callbacks_helpers.py -v`
Expected: PASS.
Run: `PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_browse_timeline.py -v`
Expected: PASS (now also exercises the toggle + render wiring).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/browse/_callbacks.py src/phenotypic/gui/_config.py tests/gui/browse/test_timeline_callbacks_helpers.py tests/e2e/gui/test_browse_timeline.py
git commit -m "feat(gui-timeline): Browse Timeline callbacks + grid render + pattern preview"
```

---

### Task 9: Single-image deep-zoom pop-out

**Files:**
- Modify: `src/phenotypic/gui/browse/_assets/timeline.js` (pop-out trigger) + `src/phenotypic/gui/browse/_callbacks.py` (modal open + OSD mount)
- Test: extend `tests/e2e/gui/test_browse_timeline.py`

**Interfaces:**
- Consumes: the per-cell **hover-revealed** `.timeline-cell-popout` button (rendered by Phase 1 `build_timeline_grid`, CSS `:hover`); the **Enter/Space → pop-out** path already wired by the Task 7 controller (`openFocusedPopout`); the existing browse DZI route (`/tiles/<token>.dzi`) + `browse.js` OSD mount.
- Produces: opening the pop-out from either path — clicking a *visible* tile's hover-revealed ⤢, OR pressing **Enter/Space** on the **focused** cell (spec §16.4) — writes `{token, label}` to `BROWSE_TL_POPOUT_STORE` via the shared hidden bridge input, opens `BROWSE_TL_POPOUT_MODAL`, and mounts a single OSD viewer (reuse the browse DZI deep-zoom) into `BROWSE_TL_POPOUT_OSD`. Both paths share the **same** `BROWSE_TL_POPOUT_INPUT` bridge (the controller in Task 7 already dispatches it for ⤢ clicks and Enter/Space), so Task 9 adds the Dash modal-open + OSD-mount wiring + the e2e assertions, not a second bridge.

- [ ] **Step 1: Write the failing test**

Append to `tests/e2e/gui/test_browse_timeline.py`:

```python
def test_hover_reveals_popout_button(live_browse_timeline) -> None:
    # The ⤢ button is hidden by default and revealed on tile hover (CSS :hover,
    # Phase 1 / spec §16.4). Hovering a populated tile makes it visible.
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell[data-src]")
    cell = page.query_selector(".timeline-cell[data-src]")
    cell.hover()
    btn = cell.query_selector(".timeline-cell-popout")
    assert btn is not None
    page.wait_for_function(
        "() => { const b = document.querySelector("
        "'.timeline-cell[data-src]:hover .timeline-cell-popout'); "
        "return b && getComputedStyle(b).visibility !== 'hidden' "
        "&& getComputedStyle(b).display !== 'none'; }"
    )


def test_popout_opens_deep_zoom_on_hover_click(live_browse_timeline) -> None:
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell[data-src]")
    cell = page.query_selector(".timeline-cell[data-src]")
    cell.hover()  # reveal the ⤢ before clicking it
    page.click(".timeline-cell[data-src] .timeline-cell-popout")
    page.wait_for_selector("#browse-tl-popout-modal.show", timeout=10000)
    # OSD mounts its canvas inside the modal body.
    page.wait_for_function(
        "document.querySelector('#browse-tl-popout-osd canvas') !== null"
    )


def test_enter_opens_popout_for_focused_cell(live_browse_timeline) -> None:
    # The keyboard path (spec §16.4): Enter on the focused cell opens the same
    # pop-out modal + OSD canvas. The controller (Task 7) wires this via the
    # shared #browse-tl-popout-input bridge.
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell--focused")
    page.click(".browse-tl-viewport")  # focus the viewport (not a text input)
    page.keyboard.press("Enter")
    page.wait_for_selector("#browse-tl-popout-modal.show", timeout=10000)
    page.wait_for_function(
        "document.querySelector('#browse-tl-popout-osd canvas') !== null"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_browse_timeline.py -k popout -v`
Expected: FAIL (no modal opens on ⤢ click or Enter; no OSD canvas).

- [ ] **Step 3: Write minimal implementation**

Three concrete pieces:

**(a) `browse.js` — fix the hardcoded OSD div (BLOCKER).** `browse.js` hardcodes `OSD_DIV_ID = "browse-osd-div"` (line 12) and `applyImage` mounts into it (line 78). The pop-out must mount into a *different* div (`browse-tl-popout-osd`). Refactor: extract the OSD-mount body into a `_mountOSD(divId, payload)` helper, have the existing `applyImage` call `_mountOSD(OSD_DIV_ID, payload)`, and add `ns.applyPopoutImage = function (payload) { _mountOSD("browse-tl-popout-osd", payload); }`. Keep `applyImage`'s observable behavior byte-identical (the existing browse tests must still pass). `_mountOSD` reuses the same `/tiles/<token>.dzi` source + loading/`open` handlers, parameterized only by `divId`.

**(b) Cell identity — use Phase 1's `data-ref` (decided; resolves open item #2).** Parsing the token out of `data-src` is fragile, so `build_timeline_grid` (Phase 1) emits a generic **`data-ref`** attribute per cell from a `ref_builder` callback (already in the committed Phase 1 plan — see "Phase 1 prerequisites already in the committed Phase 1 plan" near the bottom). Browse passes `ref_builder=lambda ref: _source_render.encode_token(str(ref))`, so each cell carries `data-ref="<token>"`.

**(c) Click + keyboard plumbing — the shared JS→Dash bridge (already wired in Task 7).** The grid cells are server-rendered children of `#browse-tl-grid` without per-cell Dash ids, so the pop-out uses the same JS→Dash bridge the colony view uses to bridge tile clicks to a store (study `results_viewer/_assets/results_viewer.js` — the `attachListener`/`dispatch` idiom — + `colony_view/_callbacks.py`). **The Task 7 controller already owns both trigger paths on the shared hidden bridge `BROWSE_TL_POPOUT_INPUT`:** (1) a delegated click on `.timeline-cell-popout` reads the parent cell's `data-ref` (the token), sets `#browse-tl-popout-input.value`, and dispatches an `input` event; (2) `openFocusedPopout` does the same for the **focused** cell on Enter/Space. So Task 9 does **not** add a second listener — it adds the Dash side:

- **Server callback** `Input(BROWSE_TL_POPOUT_INPUT,"value") → Output(BROWSE_TL_POPOUT_MODAL,"is_open"), Output(BROWSE_TL_POPOUT_STORE,"data")` opens the modal + stores `{token,label}`. **It MUST `raise PreventUpdate` when `value` is falsy/empty** (C2/OQ-5): the hidden bridge `dcc.Input(value="")` fires the callback on first page load with `""`, which would otherwise flicker the modal open with an empty payload. Guard at the top: `if not value: raise dash.exceptions.PreventUpdate`.
- **Clientside callback** `Input(BROWSE_TL_POPOUT_STORE,"data")` calls `window.__phenotypicBrowse.applyPopoutImage(payload)`. **Dash requires every clientside callback to declare an `Output`** (a sink), and `applyPopoutImage` itself returns nothing useful — so mirror the existing browse OSD-mount idiom (`app.clientside_callback(fn, Output(ids.BROWSE_OSD_SYNC,"data"), Input(ids.BROWSE_CURRENT_IMAGE_STORE,"data"))` at `_callbacks.py:334`, whose JS `return "";`): write to a **dedicated throwaway sink** — add a `dcc.Store(id=BROWSE_TL_POPOUT_OSD_SYNC)` (a new id) or reuse `Output(BROWSE_TL_POPOUT_OSD,"children")` — and have the JS `return ""` / `return window.dash_clientside.no_update`. Do NOT leave it sink-less or Dash raises at callback registration.

(`BROWSE_TL_POPOUT_INPUT` is already in `browse/_ids.py` from Task 1 and in the timeline body from Task 6; if you add a `BROWSE_TL_POPOUT_OSD_SYNC` sink store, add it to `_ids.py` + the timeline body + the Task 1 `timeline_ids` uniqueness test.) The hidden-bridge click listener is the only edit `timeline.js` needs in this task if it was not already added in Task 7 — keep it in one place (Task 7's `attach`) and have Task 9 verify it via e2e only.

- [ ] **Step 4: Run test to verify it passes**

Run: `PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_browse_timeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/browse/_assets/timeline.js src/phenotypic/gui/browse/_callbacks.py tests/e2e/gui/test_browse_timeline.py
git commit -m "feat(gui-timeline): Browse Timeline single-image deep-zoom pop-out"
```

---

### Task 10: Wire into create_app + FEATURES/WORKFLOWS/screenshots

**Files:**
- Modify: `src/phenotypic/gui/browse/_app.py` (register thumb route + ship timeline.js — assets folder already mounted)
- Modify: `src/phenotypic/gui/FEATURES.md`, `src/phenotypic/gui/WORKFLOWS.md`
- Modify: `scripts/capture_gui_tutorial_screenshots.py` (add `_capture_browse_timeline`)
- Create: `docs/source/tutorials/gui/browse_timeline.md` (or the repo's tutorial doc format — match siblings)
- Test: `tests/gui/browse/test_app.py` (append a thumb-route smoke) + run the workflows gate

**Interfaces:**
- Consumes: Task 5 `_thumb_routes.register`.
- Produces: a fully wired Browse Timeline served by `create_app`; passing CI gates.

- [ ] **Step 1: Write the failing test**

Append to `tests/gui/browse/test_app.py`:

```python
def test_create_app_serves_thumbnail_route(monkeypatch, tmp_path) -> None:
    import io
    from PIL import Image as PILImage
    from phenotypic.gui.browse._app import create_app
    from phenotypic.gui.browse import _source_render
    from phenotypic.gui.shell._sandbox import SandboxRoot

    monkeypatch.setattr(
        _source_render.tempfile, "gettempdir", lambda: str(tmp_path / "cache")
    )
    (tmp_path / "imgs").mkdir()
    PILImage.new("RGB", (120, 60), (1, 2, 3)).save(tmp_path / "imgs" / "p.png")
    app = create_app(SandboxRoot.from_path(tmp_path))
    client = app.server.test_client()
    token = _source_render.encode_token("imgs/p.png")
    resp = client.get(f"/thumb/{token}?size=64")
    assert resp.status_code == 200
    assert PILImage.open(io.BytesIO(resp.data)).size[0] <= 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_app.py::test_create_app_serves_thumbnail_route -v`
Expected: FAIL (thumb route not registered in `create_app`).

- [ ] **Step 3: Write minimal implementation**

In `browse/_app.py`, import `_thumb_routes` and call `_thumb_routes.register(app, sandbox)` next to the existing `_tile_routes.register(app, sandbox)`. (The `_assets` folder is already mounted, so `timeline.js` ships automatically.) Then:
- Add FEATURES.md rows (under `## Browse tab (source image viewer)`), in the focus-and-navigate vocabulary (spec §16 — **no** scroll-virtualization wording), for: view-mode toggle; row-source selector; time-source selector; CSV column/image dropdowns; plate-identity pattern input; advanced-regex toggle; pattern preview; tile-size stepper; CSV nudge banner; thumbnail route; the **focus-and-navigate matrix** (one focused cell, centered no-scroll window, focused neighborhood + margin-ring mount with bounded offload); the **four edge nav buttons** (◀▶▲▼); **keyboard navigation** (arrow keys move focus, clamped/no-wrap, ignored while a text input holds focus); the **focused-cell position readout** (`row N/M · time N/M`); and the **deep-zoom pop-out** opened by the hover-revealed ⤢ on any visible tile OR by Enter/Space on the focused cell — each `✅ shipping` with a resolvable `path::test` (point unit-testable ones at the Task 2–6/8 tests; point the focus-navigate / edge-button / keyboard / hover-⤢ / Enter-pop-out rows at the Task 7/9 e2e tests in `tests/e2e/gui/test_browse_timeline.py`).
- Add a WORKFLOWS.md row for "Browse — find ideal starting time" (the flow: switch to Timeline, navigate one plate's time-course with ←/→ to find the ideal starting time, ↑/↓ to compare plates, Enter to deep-zoom); add `_capture_browse_timeline` to `scripts/capture_gui_tutorial_screenshots.py`; add the tutorial page. Run `uv run python scripts/check_workflows_md.py` to confirm the round-trip.

- [ ] **Step 4: Run tests + gates**

Run: `uv run pytest tests/gui/browse -v`
Expected: PASS.
Run: `uv run python scripts/check_workflows_md.py`
Expected: pass (workflow row ↔ capture fn ↔ tutorial page reconciled).
Run: `uv run python scripts/capture_gui_tutorial_screenshots.py` and commit the regenerated PNG set wholesale (per CLAUDE.md — do not cherry-pick).
Run: `uv run ruff check src/phenotypic/gui/browse && uv run mypy src/phenotypic/gui/browse`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(gui-timeline): wire Browse Timeline into create_app + docs/CI gates"
```

---

## Phase 2 deliverable

A working Browse Timeline: `Single | Timeline` toggle; folder/EXIF filmstrip by default; per-axis source picker (folder / `{plate}`·`{time}` pattern / CSV column) with folder-scoped CSV join + collision warnings + live pattern preview + nudge; cached token-keyed thumbnails; a **focus-and-navigate matrix** (spec §16) — one focused cell in a centered no-scroll window, arrow-key + four-edge-button navigation, a position readout, focused-neighborhood + margin-ring `<img>` mounting with bounded offload, and neighborhood-first background warm; colony-style tile-size stepper; single-image deep-zoom pop-out (hover ⤢ or Enter/Space). FEATURES/WORKFLOWS/screenshots updated.

## Phase 1 prerequisites already in the committed Phase 1 plan

The focus-and-navigate controller (Task 7) and the pop-out (Task 9) depend on
three things that the **committed Phase 1 plan already provides** (verified
against `2026-06-18-source-timeline-view-phase1-shared-engine.md`) — no Phase 1
amendment is needed:

- **`ref_builder` + `data-ref`** — `build_timeline_grid` (Phase 1 Task 7) takes an
  optional `ref_builder: Callable[[object], str] | None = None` and `_build_cell`
  emits `data-ref` = `ref_builder(cell.representative)` (default `str(...)`).
  Browse passes `ref_builder=lambda ref: _source_render.encode_token(str(ref))`;
  Results (Phase 3) will pass a `"dataset/stem"` encoder.
- **`data-row-index`/`data-col-index` on every cell** (populated AND empty) — Phase 1
  Task 7 emits the 0-based grid coordinates the controller addresses cells by
  (`cellAt`, `firstPopulatedCell`, `bounds`), spec §16.8.
- **Hover-revealed ⤢** — Phase 1 marks the `.timeline-cell-popout` button
  hover-revealable (CSS class only), spec §16.4/§16.8.

If executing Phase 2 against an older Phase 1 build that predates §16, port these
three back into Phase 1 first; otherwise they are already in place.

## Review resolutions (applied 2026-06-18)

A `plan-reviewer` pass verified this plan against the code. Resolutions folded in:

- **`dbc` import (C2):** Task 6 now explicitly adds `import dash_bootstrap_components as dbc` to `_layout.py` (it was absent).
- **`browse.js` hardcoded OSD div (C1):** Task 9(a) extracts `_mountOSD(divId, payload)` + adds `applyPopoutImage`.
- **`live_browse_timeline` fixture (C3, first review):** Task 7 carries a concrete fixture recipe (no browse e2e existed to copy). **Superseded by the second-review C1 below** — the source-store seeding now uses the proven sidebar-tree-click idiom (the earlier localStorage injection was wrong on both key and payload).
- **Static `data-*` placement (W1):** the focus-navigate constants (`data-focus-margin`, `data-mount-cap`, `data-warm-concurrency` — spec §16.7; the scroll-era `data-margin-screens` is gone) are static attrs on the `BROWSE_TL_GRID` container in `build_timeline_body()` (Task 6), not callback outputs.
- **`data-ref` decision (W2):** resolved via the Phase 1 `data-ref` (already in the committed Phase 1 plan; not URL-parsing).
- **tempdir monkeypatch + `SandboxRoot.from_path` (W3/W4):** applied to Task 5 + Task 10 tests (verified idiom: `tests/gui/browse/test_tile_routes.py`).
- **CSV column/row reader (W5):** Task 8 adds a public `read_metadata_csv_table(path) -> (columns, rows)` to `shell/_metadata_context.py` (the store holds a path payload; `_read_rows` is private).
- **`step_timeline_tile_size` `__all__` (W6):** Task 8 adds it to `_config.py`'s `__all__`.
- **`_sandbox_rel` duplication (S3):** Task 4 keeps a local helper (importing from `_callbacks` would cycle); interface note corrected.
- **`cell_ref` contract:** confirmed — `build_browse_records` emits the **sandbox-rel path**; `render_timeline_grid` encodes it to a token (Task 8). `data-ref` carries the token (Phase 1 `data-ref`, already in the committed Phase 1 plan).

Verified non-issues from the review: `create_app` signature + `_tile_routes.register` call site, `_assets` auto-mount, `window.__phenotypicAppPrefix` injection, `sandbox_rel`/`_src_root_rel`/`list_datasets` availability, `exifread` is already a dependency, `dcc.Input(debounce=True)`/`dcc.RadioItems(inline=True)` valid in Dash 4.1, `check_features_md.py` skips `🧪 internal` rows, the warm-loop bounded-concurrency `pump()` correctness, and that wrapping the body in `BROWSE_SINGLE_BODY` is transparent to existing callbacks.

## Second review resolutions (focus-navigate plan, applied 2026-06-18)

A second `plan-reviewer` pass audited the focus-navigate revision; all fixes folded in (each verified against the real code first):

- **C1 — e2e fixture source-store seeding (blocker, fixed):** the localStorage injection was wrong on **both** the key (the `dcc.Store(storage_type="local")` keys by the id-VALUE `"shell-source-image-root-store"`, `shell/_ids.py:47`, not the constant name) **and** the payload (`resolve_source_image_root`, `_source_context.py:91`, requires `version`/`validated is True`/string `abs_path`; it ignores `rel_path`). The `live_browse_timeline` recipe now seeds the matrix **under the shared `plate1` source root** and selects it via the **proven sidebar-tree-click idiom** (`test_shared_source_root._select_plate1_source`), with a correct-key/`source_payload_from_path` injection only as a documented fallback. The "validate localStorage key early" caveat is dropped.
- **C2/OQ-5 — pop-out clientside callback sink + empty-value guard (fixed):** Task 9(c) now specifies a dedicated throwaway Output sink for the `applyPopoutImage` clientside callback (mirroring `Output(BROWSE_OSD_SYNC,"data")` at `_callbacks.py:334`, `return ""`), and `raise PreventUpdate` on the empty `""` first-load value of the bridge input.
- **C3 — `bounds()` coordinate-0 coupling (documented):** Task 7 Step 4 now cites Phase 1's `test_cells_carry_grid_coordinate_indices` as the guarding invariant (focus math depends on every cell — empty included — emitting `data-row-index`/`data-col-index`).
- **C4/W5 — `read_metadata_csv_table` encoding (fixed):** uses `encoding="utf-8-sig"` (matching `_read_rows`, `:206`) so an Excel BOM doesn't break the `csv_image_col` join; the unit test gains a BOM-bearing-CSV regression case.
- **OQ-1 — corner behavior (implemented as clamp-translate):** `recenter()` now clamps the centering transform so the grid never pulls past its own edges (full window, no empty gutters, even at the default (0,0) focus). Pending a final user confirm, but clamp-translate is the implemented default.
- **OQ-2/W2 — first-paint timing (fixed):** `attach` re-schedules via `requestAnimationFrame` while `viewport.getBoundingClientRect().width === 0`, self-correcting regardless of toggle-callback ordering.
- **OQ-3 — `piexif` test dep (removed):** Task 2 now reads a **committed** JPEG fixture (`tests/gui/browse/fixtures/with_datetimeoriginal.jpg`); no `piexif` runtime/dev dependency (it's authored once via ephemeral `uv run --with piexif`).
- **W4 — abort warm on toggle-to-Single (added):** `timeline.js` exposes `ns.cancelWarm()` (bumps the generation); the view-mode toggle calls it when switching away from Timeline.
- **Surface-agnostic controller (third review, cross-surface):** the vendored `timeline.js` is now portable — it locates its sibling controls by stable class scoped to the enclosing `.timeline-body` (`.timeline-nav-*`, `.timeline-position`, `.timeline-popout-bridge`, `.timeline-viewport`, `.timeline-grid-container`), never by a hardcoded `browse-tl-*` id. See the Phase 3 forward-note below.

### Phase 3 forward-note — separate apps + byte-identical `timeline.js`

**Verified architecture (`shell/_app.py::compose_hub`):** Browse and Results are **separate Dash apps** mounted via `DispatcherMiddleware`, each with its **own `assets_folder`** (`MOUNT_BROWSE.rstrip("/"): browse_app.server`, etc.). They run in distinct page loads, so `window.__phenotypicTimeline`'s focus / mounted-LRU / generation state **does NOT collide across surfaces** — **no per-container state keying is needed** (this supersedes the earlier "must key by container id" note, which assumed a shared page).

Because Results (Phase 3) **vendors this exact `timeline.js` byte-for-byte** (a CI byte-equality guard enforces the two `_assets/timeline.js` copies never drift), the controller must be **surface-agnostic** — which the Task 6 + Task 7 edits make it:

- Task 6 puts stable `timeline-*` classes on Browse's elements (alongside their Dash ids): `.timeline-grid-container`, `.timeline-viewport`, `.timeline-nav-up/down/left/right`, `.timeline-position`, `.timeline-popout-bridge`, and `.timeline-body` on the wrapper. Phase 3 puts the **same classes** on its `timeline-*`-id'd elements.
- Task 7's controller reads only those classes (scoped to `.timeline-body`) + the container id passed into `attach(containerId)`. The container id is the **only** surface-specific input, and it's a parameter, not hardcoded — each surface's clientside callback passes its own (`browse-tl-grid` for Browse, e.g. `timeline-grid` for Results).

So the Phase 3 plan: vendor `timeline.js` verbatim, add the `timeline-*` classes to its layout elements, and pass its container id into `attach` — the controller runs unchanged on `/results/`. **No Phase 2 code change** beyond the class additions already made here.

## Focus-and-navigate retarget (spec §16, applied 2026-06-18)

This plan was originally written for a scrollable, `IntersectionObserver`-virtualized
matrix. Spec §16 (locked, user-directed) replaced that with a **focus-and-navigate**
model; the changes folded in here:

- **Task 1:** added `BROWSE_TL_NAV_UP/DOWN/LEFT/RIGHT` (the four on-edge ◀▶▲▼ buttons)
  + `BROWSE_TL_POSITION` (focused-cell readout) — DOM targets for `timeline.js`, no Dash
  callback.
- **Task 6:** replaced the scrollable `grid_scroll` with a **no-scroll focus-window
  viewport** (`overflow:hidden`, bounded height, `position:relative`, focusable
  `tabIndex=0`); added the four edge buttons + the position readout; changed the grid
  container's static `data-*` (dropped `data-margin-screens`, added `data-focus-margin`);
  fixed the `_config` import to `TIMELINE_FOCUS_MARGIN` (not the removed
  `TIMELINE_WINDOW_MARGIN_SCREENS`).
- **Task 7:** **rewrote** `timeline.js` from IntersectionObserver scroll-virtualization to
  the focus-navigate controller (focus state init to the first populated cell; centered
  window via CSS `transform`; margin-ring mount/offload addressed by
  `[data-row-index][data-col-index]`; arrow-key + edge-button navigation with bound
  clamping and text-input guard; Enter/Space → pop-out via the shared bridge;
  neighborhood-first generation-guarded warm; idempotent `<body>`-observer re-attach).
  The e2e tests were rewritten to the §16.9 assertions (focus on first populated cell,
  arrow/edge-button moves focus + mounts neighborhood, far cell unmounted, off-screen
  margin-ring pre-mounted).
- **Task 9:** the ⤢ button is hover-revealed (CSS); Enter/Space on the focused cell opens
  the same pop-out (controller in Task 7); Task 9 adds the e2e assertions + the Dash
  modal-open/OSD-mount side and shares the single `BROWSE_TL_POPOUT_INPUT` bridge.
- **Task 10:** FEATURES/WORKFLOWS rows use the focus-navigate vocabulary (edge buttons,
  keyboard nav, position readout, hover ⤢ / Enter pop-out); scroll-virtualization wording
  dropped; the "find ideal starting time" WORKFLOWS flow retained.

Tasks 2–5 (data layer: ids-unrelated parts, capture-time, plate-pattern, record-builder,
thumbnail route) are unaffected.

## Remaining open items

- **Corner behavior is clamp-translate (OQ-1) — pending final user confirm only.** §16.1 says the focused cell renders "at the viewport center." Strictly centering leaves a half-empty viewport at edges (incl. the default (0,0) focus), so `recenter()` implements **clamp-translate**: center the focused cell but clamp so the grid never pulls past its own edges (full window, no empty gutters; the focused cell sits off-center near matrix corners, kept unambiguous by the `.timeline-cell--focused` highlight). This is a UX choice, not a correctness one — implemented as the default; flag for a final human nod.

> **Resolved in the second review (no longer open):** the `piexif` test dep (now a committed JPEG fixture, OQ-3); the centered-window corner behavior is now implemented (clamp-translate); the `visibleHalf`/first-paint timing is now handled by the `requestAnimationFrame` guard in `attach` (OQ-2/W2). See "Second review resolutions" above.
