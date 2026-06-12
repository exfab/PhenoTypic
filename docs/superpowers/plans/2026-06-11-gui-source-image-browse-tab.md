# Source Image Browse Tab — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new top-level **Browse** tab to the GUI hub that lists every image under the selected source root (two cascading dropdowns: dataset folder → image, with ‹/› stepper) and renders any one in an OpenSeadragon deep-zoom viewport with a metadata panel — working over an SSH port-forward on an offline cluster.

**Architecture:** A new eager Dash sub-app `gui/browse/` mounted at `/browse/` via the existing `DispatcherMiddleware`. It reuses the Results Viewer's DZI tiler + vendored OpenSeadragon. Each source file is lazily normalized to an 8-bit RGB PNG (via `phenotypic.Image.imread` + `skimage.util.img_as_ubyte`) and tiled into an **ephemeral** temp cache (`tempfile.gettempdir()/phenotypic/browse/<token>`), wiped on launch + `atexit`. The tile URL carries the image's sandbox-relative path as a slash-free **base64url token**, resolved against the frozen `SandboxRoot`.

**Tech Stack:** Python 3, Dash + dash-bootstrap-components, Flask blueprints, Pillow, scikit-image, `phenotypic.Image` (rawpy for RAW), OpenSeadragon (vendored JS), pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-06-11-gui-source-image-browse-tab-design.md` (authoritative for decisions F1/B1/A1/C1/C2/U1/R1).

---

## Conventions for every task

- Run everything through `uv`. Tests: `uv run pytest <path> -v`. Never bare `python`/`pip`.
- The Browse Qt-free; plain pytest (no `qt-test` group needed for these unit tests). Ensure the env has the GUI extra: `uv sync --group dev --extra gui` (one-time, Task 0).
- Commit after each task with the shown message. Branch is already `worktree-gui-source-image-viewer`.
- Source lives in `src/phenotypic/gui/browse/`; tests in `tests/gui/browse/`.

---

## File Structure

**New package `src/phenotypic/gui/browse/`:**
| File | Responsibility |
|------|----------------|
| `__init__.py` | Exports `create_app`. |
| `_ids.py` | All component IDs (static strings). |
| `_source_render.py` | base64url token encode/decode; ephemeral temp-cache paths + wipe/init lifecycle; `normalize_to_png` (any format → faithful 8-bit PNG); `SourceRenderUnavailable`. |
| `_source_lister.py` | `list_datasets(source_root) -> {dataset_rel: [filename,…]}`. |
| `_metadata.py` | `read(original) -> {width,height,bytes,exif}`; `_extract_exif`. |
| `_tile_routes.py` | Flask blueprint: `/<token>.dzi` + `/<token>_files/<level>/<file>`. |
| `_layout.py` | Single-pane layout (dataset dropdown + image picker + ‹/› + OSD div + metadata panel + stores). |
| `_callbacks.py` | Dataset options + map store + flat-hide; cascade dataset→image; ‹/› step + bounds-disable; current-image token store; metadata panel; clientside OSD mount. |
| `_app.py` | `create_app(sandbox, *, url_prefix)`; index-string prefix inject; cache init; route + callback registration. |
| `__main__.py` | Standalone launcher (`python -m phenotypic.gui.browse`). |
| `_assets/browse.js` | Vendored OSD bootstrap + single-viewport mount/dispose + `applyImage`. |
| `_assets/browse.css` | Viewport + metadata panel styling (design tokens only). |
| `_assets/openseadragon/` | Vendored OSD (copied from results_viewer assets). |

**Modified shared files:**
| File | Change |
|------|--------|
| `gui/_config.py` | Add `IMAGE_EXTS`, `RAW_IMAGE_EXTS`, `MOUNT_BROWSE`, `BROWSE_TILES_PREFIX`, `BROWSE_CACHE_TMP_SUBPATH`, `TITLE_BROWSE`. |
| `gui/builder/_directory_browser.py` | Re-export `IMAGE_EXTS` from `_config` (back-compat). |
| `gui/_shared/_picker_navigation.py` | New home for `enabled_picker_values`/`step_picker_value`/`picker_button_disabled_states`. |
| `gui/results_viewer/_picker_navigation.py` | Re-export from `_shared` (back-compat). |
| `gui/shell/_ids.py` | Add `SHELL_TAB_BROWSE`. |
| `gui/shell/_layout.py` | `_TAB_HREFS`/`_TAB_LABELS`/`NAV_MODEL` add Browse leaf after Home. |
| `gui/shell/_app.py` | `compose_hub` builds + mounts the Browse app. |
| `gui/FEATURES.md`, `gui/WORKFLOWS.md`, `scripts/capture_gui_tutorial_screenshots.py`, `docs/source/tutorials/gui/` | Ledgers + tutorial + screenshots. |

---

## Task 0: Environment + scaffolding

**Files:**
- Create: `src/phenotypic/gui/browse/__init__.py`
- Create: `tests/gui/browse/__init__.py`

- [ ] **Step 1: Sync the GUI dev env**

Run: `uv sync --group dev --extra gui`
Expected: completes; `uv run python -c "import dash, PIL, skimage; print('ok')"` prints `ok`.

- [ ] **Step 2: Create empty package + test package markers**

`src/phenotypic/gui/browse/__init__.py`:
```python
"""PhenoTypic GUI — Source Image Browse tab.

A deep-zoom viewer for the raw input images under the selected source
root. Mounted at ``/browse/`` in the unified hub. See
``docs/superpowers/specs/2026-06-11-gui-source-image-browse-tab-design.md``.
"""
from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str):  # lazy to avoid importing dash at package import
    if name == "create_app":
        from phenotypic.gui.browse._app import create_app

        return create_app
    raise AttributeError(name)
```

`tests/gui/browse/__init__.py`:
```python
```

- [ ] **Step 3: Verify import**

Run: `uv run python -c "import phenotypic.gui.browse as b; print(b.__doc__.splitlines()[0])"`
Expected: prints the docstring's first line.

- [ ] **Step 4: Commit**

```bash
git add src/phenotypic/gui/browse/__init__.py tests/gui/browse/__init__.py
git commit -m "feat(gui): scaffold browse sub-app package"
```

---

## Task 1: Lift `IMAGE_EXTS` + add RAW subset to `_config.py`

**Files:**
- Modify: `src/phenotypic/gui/_config.py`
- Modify: `src/phenotypic/gui/builder/_directory_browser.py:30-43`
- Test: `tests/gui/browse/test_config_image_exts.py`

- [ ] **Step 1: Write the failing test**

`tests/gui/browse/test_config_image_exts.py`:
```python
from phenotypic.gui._config import IMAGE_EXTS, RAW_IMAGE_EXTS
from phenotypic.gui.builder._directory_browser import IMAGE_EXTS as BUILDER_IMAGE_EXTS


def test_image_exts_cover_standard_and_raw():
    assert {".png", ".tif", ".tiff", ".jpg", ".jpeg"} <= IMAGE_EXTS
    assert {".raw", ".nef", ".cr2", ".arw", ".dng"} <= IMAGE_EXTS


def test_raw_subset_of_image_exts():
    assert RAW_IMAGE_EXTS <= IMAGE_EXTS
    assert ".png" not in RAW_IMAGE_EXTS


def test_builder_reexports_the_same_object():
    # Back-compat: builder must keep exporting the identical frozenset.
    assert BUILDER_IMAGE_EXTS is IMAGE_EXTS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_config_image_exts.py -v`
Expected: FAIL with `ImportError: cannot import name 'IMAGE_EXTS' from 'phenotypic.gui._config'`.

- [ ] **Step 3: Add the constants to `_config.py`**

In `src/phenotypic/gui/_config.py`, add near the other path/string constants (after the `SANDBOX_*` block):
```python
# ---------------------------------------------------------------------------
# Image file extensions (shared by the directory browser, classifier, and the
# Browse tab). Lifted here so neither browse nor the classifier imports the
# builder package. ``builder/_directory_browser`` re-exports IMAGE_EXTS.
# ---------------------------------------------------------------------------
IMAGE_EXTS: frozenset[str] = frozenset(
    {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".raw", ".nef", ".cr2", ".arw", ".dng"}
)

#: Camera-RAW subset of :data:`IMAGE_EXTS`. These require rawpy (absent on
#: Windows) and decode through ``phenotypic.Image.imread``.
RAW_IMAGE_EXTS: frozenset[str] = frozenset({".raw", ".nef", ".cr2", ".arw", ".dng"})
```
Add `"IMAGE_EXTS"`, `"RAW_IMAGE_EXTS"` to the module `__all__` list.

- [ ] **Step 4: Re-export from the builder (back-compat)**

In `src/phenotypic/gui/builder/_directory_browser.py`, replace the literal frozenset (lines ~30-43) with a re-export. Change:
```python
IMAGE_EXTS: FrozenSet[str] = frozenset(
    {
        ".png",
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
        ".raw",
        ".nef",
        ".cr2",
        ".arw",
        ".dng",
    }
)
```
to:
```python
from phenotypic.gui._config import IMAGE_EXTS  # re-exported for back-compat
```
(Place the import with the other top-of-file imports; delete the literal. Keep the `FrozenSet` import only if still used elsewhere in the file — run ruff to confirm.)

- [ ] **Step 5: Run tests + lint**

Run: `uv run pytest tests/gui/browse/test_config_image_exts.py -v && uv run ruff check --fix src/phenotypic/gui/_config.py src/phenotypic/gui/builder/_directory_browser.py`
Expected: PASS (3 tests); ruff clean.

- [ ] **Step 6: Guard against regressions in the classifier import**

Run: `uv run python -c "from phenotypic.gui.shell._classifier import classify; print('ok')"`
Expected: `ok` (classifier imports `IMAGE_EXTS` from builder, which now re-exports).

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/gui/_config.py src/phenotypic/gui/builder/_directory_browser.py tests/gui/browse/test_config_image_exts.py
git commit -m "refactor(gui): lift IMAGE_EXTS to _config, add RAW subset"
```

---

## Task 2: Lift picker-navigation helpers to `gui/_shared`

**Files:**
- Create: `src/phenotypic/gui/_shared/_picker_navigation.py`
- Modify: `src/phenotypic/gui/results_viewer/_picker_navigation.py`
- Test: `tests/gui/browse/test_shared_picker_navigation.py`

- [ ] **Step 1: Write the failing test**

`tests/gui/browse/test_shared_picker_navigation.py`:
```python
from phenotypic.gui._shared._picker_navigation import (
    enabled_picker_values,
    picker_button_disabled_states,
    step_picker_value,
)
from phenotypic.gui.results_viewer import _picker_navigation as rv_nav


def _opts(*values):
    return [{"label": v, "value": v} for v in values]


def test_step_next_and_prev():
    opts = _opts("a", "b", "c")
    assert step_picker_value("a", opts, "next") == "b"
    assert step_picker_value("c", opts, "previous") == "b"
    assert step_picker_value("c", opts, "next") == "c"  # clamp at end


def test_bounds_disabled_states():
    opts = _opts("a", "b", "c")
    assert picker_button_disabled_states("a", opts) == (True, False)
    assert picker_button_disabled_states("c", opts) == (False, True)


def test_results_viewer_reexports_same_callables():
    assert rv_nav.step_picker_value is step_picker_value
    assert rv_nav.picker_button_disabled_states is picker_button_disabled_states
    assert rv_nav.enabled_picker_values is enabled_picker_values
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_shared_picker_navigation.py -v`
Expected: FAIL with `ModuleNotFoundError: ...gui._shared._picker_navigation`.

- [ ] **Step 3: Create the shared module (move the implementation verbatim)**

Create `src/phenotypic/gui/_shared/_picker_navigation.py` with the exact current contents of `results_viewer/_picker_navigation.py` (the three functions `enabled_picker_values`, `step_picker_value`, `picker_button_disabled_states` plus the `PickerDirection` alias). Update the module docstring to:
```python
"""Dropdown previous/next helpers shared by GUI image pickers."""
```
(Keep the function bodies identical.)

- [ ] **Step 4: Turn the results_viewer module into a re-export**

Replace the body of `src/phenotypic/gui/results_viewer/_picker_navigation.py` with:
```python
"""Back-compat shim: picker-navigation helpers moved to ``gui/_shared``."""
from __future__ import annotations

from phenotypic.gui._shared._picker_navigation import (
    PickerDirection,
    enabled_picker_values,
    picker_button_disabled_states,
    step_picker_value,
)

__all__ = [
    "PickerDirection",
    "enabled_picker_values",
    "step_picker_value",
    "picker_button_disabled_states",
]
```

- [ ] **Step 5: Run new + existing picker-nav tests**

Run: `uv run pytest tests/gui/browse/test_shared_picker_navigation.py -v && uv run pytest tests/gui -k picker_nav -v`
Expected: new test PASS; any existing picker-navigation tests still PASS (they import from the re-export).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/_shared/_picker_navigation.py src/phenotypic/gui/results_viewer/_picker_navigation.py tests/gui/browse/test_shared_picker_navigation.py
git commit -m "refactor(gui): lift picker-navigation helpers to _shared"
```

---

## Task 3: New Browse constants + shell tab id

**Files:**
- Modify: `src/phenotypic/gui/_config.py`
- Modify: `src/phenotypic/gui/shell/_ids.py`
- Test: `tests/gui/browse/test_browse_constants.py`

- [ ] **Step 1: Write the failing test**

`tests/gui/browse/test_browse_constants.py`:
```python
from phenotypic.gui._config import (
    BROWSE_CACHE_TMP_SUBPATH,
    BROWSE_TILES_PREFIX,
    MOUNT_BROWSE,
    TITLE_BROWSE,
)
from phenotypic.gui.shell._ids import SHELL_TAB_BROWSE


def test_browse_mount_and_prefixes():
    assert MOUNT_BROWSE == "/browse/"
    assert BROWSE_TILES_PREFIX == "/tiles"
    assert BROWSE_CACHE_TMP_SUBPATH == ("phenotypic", "browse")
    assert TITLE_BROWSE == "PhenoTypic Source Browser"


def test_shell_tab_browse():
    assert SHELL_TAB_BROWSE == "shell-tab-browse"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_browse_constants.py -v`
Expected: FAIL with `ImportError: cannot import name 'MOUNT_BROWSE'`.

- [ ] **Step 3: Add the `_config.py` constants**

In `src/phenotypic/gui/_config.py`:
- Next to `MOUNT_TUNE`: `MOUNT_BROWSE: str = "/browse/"`
- Next to `VIEWER_TILES_PREFIX`: `BROWSE_TILES_PREFIX: str = "/tiles"`
- Next to the `SANDBOX_*` block: `BROWSE_CACHE_TMP_SUBPATH: tuple[str, str] = ("phenotypic", "browse")`
- Next to `TITLE_VIEWER`: `TITLE_BROWSE: str = "PhenoTypic Source Browser"`

Add `"MOUNT_BROWSE"`, `"BROWSE_TILES_PREFIX"`, `"BROWSE_CACHE_TMP_SUBPATH"`, `"TITLE_BROWSE"` to `__all__`.

- [ ] **Step 4: Add the shell tab id**

In `src/phenotypic/gui/shell/_ids.py`, next to `SHELL_TAB_TUNE`:
```python
SHELL_TAB_BROWSE = "shell-tab-browse"
```
Add `"SHELL_TAB_BROWSE"` to that file's `__all__`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/gui/browse/test_browse_constants.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/_config.py src/phenotypic/gui/shell/_ids.py tests/gui/browse/test_browse_constants.py
git commit -m "feat(gui): add Browse tab constants + shell tab id"
```

---

## Task 4: `_source_render.py` — token + ephemeral cache + faithful render

**Files:**
- Create: `src/phenotypic/gui/browse/_source_render.py`
- Test: `tests/gui/browse/test_source_render.py`

- [ ] **Step 1: Write the failing test**

`tests/gui/browse/test_source_render.py`:
```python
import numpy as np
import pytest
from PIL import Image as PILImage

from phenotypic.gui.browse import _source_render as sr


def test_token_round_trip_is_slash_free():
    rel = "plates/batch7/day3/A1_scan.nef"
    token = sr.encode_token(rel)
    assert "/" not in token and "=" not in token
    assert sr.decode_token(token) == rel


def test_cache_base_under_tempdir(monkeypatch, tmp_path):
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path))
    assert sr.browse_cache_base() == tmp_path / "phenotypic" / "browse"
    assert sr.cache_png_path("tok") == tmp_path / "phenotypic" / "browse" / "tok.png"


def test_normalize_standard_png(monkeypatch, tmp_path):
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path))
    src = tmp_path / "src.png"
    PILImage.fromarray(np.full((8, 8, 3), 200, dtype=np.uint8)).save(src)
    out = sr.normalize_to_png(src, sr.cache_png_path("t1"))
    assert out.exists()
    arr = np.asarray(PILImage.open(out).convert("RGB"))
    assert arr.dtype == np.uint8 and arr.shape == (8, 8, 3)


def test_normalize_is_mtime_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path))
    src = tmp_path / "src.png"
    PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(src)
    out = sr.normalize_to_png(src, sr.cache_png_path("t2"))
    first_mtime = out.stat().st_mtime_ns
    out2 = sr.normalize_to_png(src, sr.cache_png_path("t2"))  # cache hit
    assert out2.stat().st_mtime_ns == first_mtime


def test_raw_unavailable_raises_typed(monkeypatch, tmp_path):
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path))
    raw = tmp_path / "shot.nef"
    raw.write_bytes(b"not really a raw file")

    def _boom(*a, **k):
        raise ImportError("rawpy not installed")

    monkeypatch.setattr(sr.Image, "imread", _boom)
    with pytest.raises(sr.SourceRenderUnavailable):
        sr.normalize_to_png(raw, sr.cache_png_path("t3"))


def test_wipe_and_init_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path))
    base = sr.browse_cache_base()
    base.mkdir(parents=True)
    (base / "stale.png").write_bytes(b"x")
    sr.init_cache()
    assert base.is_dir()
    assert not (base / "stale.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_source_render.py -v`
Expected: FAIL with `ModuleNotFoundError: ...browse._source_render`.

- [ ] **Step 3: Implement `_source_render.py`**

`src/phenotypic/gui/browse/_source_render.py`:
```python
"""Source-file → faithful 8-bit RGB PNG, with an ephemeral temp tile cache.

The Browse tab never tiles a source file directly; it first normalizes any
supported format (standard *or* camera RAW) to an 8-bit RGB PNG via
``phenotypic.Image.imread`` + ``skimage.util.img_as_ubyte`` (a faithful
full-range downcast — no auto-contrast), then hands that PNG to the shared
DZI tiler. The cache lives under ``tempfile.gettempdir()/phenotypic/browse``,
keyed by a slash-free base64url token of the image's sandbox-relative path,
and is wiped on launch + at process exit.
"""
from __future__ import annotations

import atexit
import base64
import logging
import shutil
import tempfile
from pathlib import Path

from PIL import Image as PILImage
from skimage.util import img_as_ubyte

from phenotypic import Image
from phenotypic.gui._config import BROWSE_CACHE_TMP_SUBPATH, RAW_IMAGE_EXTS

logger = logging.getLogger(__name__)

__all__ = [
    "SourceRenderUnavailable",
    "encode_token",
    "decode_token",
    "browse_cache_base",
    "cache_png_path",
    "wipe_cache",
    "init_cache",
    "normalize_to_png",
]

_atexit_registered = False


class SourceRenderUnavailable(RuntimeError):
    """Raised when a source file cannot be decoded on this platform.

    The common case is camera RAW on Windows, where ``rawpy`` is excluded.
    The tile route maps this to a 422 + an inline viewer notice.
    """


def encode_token(sandbox_rel: str) -> str:
    """Encode a sandbox-relative POSIX path as a slash-free base64url token."""
    raw = base64.urlsafe_b64encode(sandbox_rel.encode("utf-8")).decode("ascii")
    return raw.rstrip("=")


def decode_token(token: str) -> str:
    """Inverse of :func:`encode_token`. Raises on malformed input."""
    pad = "=" * (-len(token) % 4)
    return base64.urlsafe_b64decode((token + pad).encode("ascii")).decode("utf-8")


def browse_cache_base() -> Path:
    """The ephemeral cache root (recomputed each call so ``$TMPDIR`` is honoured)."""
    return Path(tempfile.gettempdir()).joinpath(*BROWSE_CACHE_TMP_SUBPATH)


def cache_png_path(token: str) -> Path:
    """Path to the normalized PNG the DZI tiler consumes for ``token``."""
    return browse_cache_base() / f"{token}.png"


def wipe_cache() -> None:
    """Best-effort recursive delete of the cache base. Never raises."""
    shutil.rmtree(browse_cache_base(), ignore_errors=True)


def init_cache() -> None:
    """Wipe stale tiles on launch and register an ``atexit`` cleanup (idempotent)."""
    global _atexit_registered
    wipe_cache()
    browse_cache_base().mkdir(parents=True, exist_ok=True)
    if not _atexit_registered:
        atexit.register(wipe_cache)
        _atexit_registered = True


def normalize_to_png(original: Path, cache_png: Path) -> Path:
    """Render ``original`` to a faithful 8-bit RGB PNG at ``cache_png``.

    Idempotent: returns the existing PNG when it is at least as new as the
    source. RAW that cannot be decoded raises :class:`SourceRenderUnavailable`;
    a decode failure on a standard format re-raises the original error.
    """
    original = Path(original)
    if cache_png.exists() and cache_png.stat().st_mtime >= original.stat().st_mtime:
        return cache_png
    try:
        rgb = Image.imread(str(original)).rgb[:]
    except Exception as exc:  # noqa: BLE001 - classify by extension below
        if original.suffix.lower() in RAW_IMAGE_EXTS:
            raise SourceRenderUnavailable(
                f"cannot decode RAW source on this platform: {original.name}"
            ) from exc
        raise
    rgb8 = img_as_ubyte(rgb)
    cache_png.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(rgb8).save(cache_png, format="PNG")
    return cache_png
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/gui/browse/test_source_render.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check --fix src/phenotypic/gui/browse/_source_render.py && uv run mypy src/phenotypic/gui/browse/_source_render.py`
Expected: ruff clean; mypy clean (or pre-existing-only errors).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/browse/_source_render.py tests/gui/browse/test_source_render.py
git commit -m "feat(gui): browse _source_render — token + ephemeral cache + faithful render"
```

---

## Task 5: `_source_lister.py` — recursive grouped listing

**Files:**
- Create: `src/phenotypic/gui/browse/_source_lister.py`
- Test: `tests/gui/browse/test_source_lister.py`

- [ ] **Step 1: Write the failing test**

`tests/gui/browse/test_source_lister.py`:
```python
from phenotypic.gui.browse._source_lister import list_datasets


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_flat_source_uses_dot_key(tmp_path):
    _touch(tmp_path / "a.png")
    _touch(tmp_path / "b.jpg")
    assert list_datasets(tmp_path) == {".": ["a.png", "b.jpg"]}


def test_nested_groups_by_relative_parent(tmp_path):
    _touch(tmp_path / "plates" / "batch7" / "A1.png")
    _touch(tmp_path / "plates" / "batch7" / "A2.png")
    _touch(tmp_path / "plates" / "batch8" / "B1.tif")
    result = list_datasets(tmp_path)
    assert result == {
        "plates/batch7": ["A1.png", "A2.png"],
        "plates/batch8": ["B1.tif"],
    }


def test_non_image_and_hidden_skipped(tmp_path):
    _touch(tmp_path / "keep.png")
    _touch(tmp_path / "notes.txt")
    _touch(tmp_path / ".phenotypic" / "view" / "cached.png")
    assert list_datasets(tmp_path) == {".": ["keep.png"]}


def test_empty_dir(tmp_path):
    assert list_datasets(tmp_path) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_source_lister.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `_source_lister.py`**

`src/phenotypic/gui/browse/_source_lister.py`:
```python
"""List image files under a source root, grouped by relative subfolder.

Produces the ``{dataset_rel: [filename, ...]}`` map that drives the Browse
tab's two cascading dropdowns. ``dataset_rel`` is the image's parent
directory relative to the source root (``"."`` for files directly under
the root), so arbitrary nesting collapses to one flat set of dataset keys.
"""
from __future__ import annotations

import logging
from pathlib import Path

from phenotypic.gui._config import IMAGE_EXTS

logger = logging.getLogger(__name__)

__all__ = ["list_datasets"]


def list_datasets(source_root: Path) -> dict[str, list[str]]:
    """Return an ordered ``{dataset_rel: [filename, ...]}`` map.

    Hidden files/dirs (leading ``.``) and symlinks whose target escapes the
    source root are skipped. Keys and filename lists are sorted.
    """
    source_root = Path(source_root)
    try:
        root_resolved = source_root.resolve(strict=False)
    except (OSError, RuntimeError):
        return {}
    out: dict[str, list[str]] = {}
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            rel = path.relative_to(source_root)
        except ValueError:
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue  # hidden dotfile / dot-dir (e.g. .phenotypic cache)
        try:
            path.resolve(strict=False).relative_to(root_resolved)
        except ValueError:
            continue  # symlink escaping the source root
        out.setdefault(rel.parent.as_posix(), []).append(path.name)
    for files in out.values():
        files.sort()
    return dict(sorted(out.items()))
```

- [ ] **Step 4: Run tests + lint**

Run: `uv run pytest tests/gui/browse/test_source_lister.py -v && uv run ruff check --fix src/phenotypic/gui/browse/_source_lister.py`
Expected: PASS (4 tests); ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/browse/_source_lister.py tests/gui/browse/test_source_lister.py
git commit -m "feat(gui): browse _source_lister — recursive grouped image listing"
```

---

## Task 6: `_metadata.py` — dims + size + EXIF

**Files:**
- Create: `src/phenotypic/gui/browse/_metadata.py`
- Test: `tests/gui/browse/test_metadata.py`

- [ ] **Step 1: Write the failing test**

`tests/gui/browse/test_metadata.py`:
```python
import numpy as np
from PIL import Image as PILImage

from phenotypic.gui.browse._metadata import _extract_exif, read


def test_extract_exif_from_exifread_keys():
    imported = {
        "EXIF DateTimeOriginal": "2024:03:01 14:22:05",
        "Image Make": "NIKON CORPORATION",
        "Image Model": "NIKON D850",
        "EXIF ExposureTime": "1/200",
    }
    assert _extract_exif(imported) == {
        "captured": "2024:03:01 14:22:05",
        "make": "NIKON CORPORATION",
        "model": "NIKON D850",
    }


def test_extract_exif_empty_when_absent():
    assert _extract_exif({}) == {}


def test_read_dims_and_size_no_exif(tmp_path):
    src = tmp_path / "plate.png"
    PILImage.fromarray(np.zeros((12, 20, 3), dtype=np.uint8)).save(src)
    info = read(src)
    assert info["width"] == 20 and info["height"] == 12
    assert info["bytes"] == src.stat().st_size
    assert info["exif"] == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_metadata.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `_metadata.py`**

`src/phenotypic/gui/browse/_metadata.py`:
```python
"""Read display metadata (dims, file size, EXIF) from a source image file.

EXIF is pulled from ``phenotypic.Image``'s imported metadata, which is
populated by ``exifread`` for both JPEG and TIFF-based RAW (NEF/CR2). Any
field that is absent or unreadable is silently omitted — the panel degrades
gracefully rather than raising.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from phenotypic import Image

logger = logging.getLogger(__name__)

__all__ = ["read"]


def _extract_exif(imported: dict[str, Any]) -> dict[str, str]:
    """Pull capture-time + camera make/model from an exifread-style dict."""

    def _find(*needles: str) -> str | None:
        for key, value in imported.items():
            key_lower = key.lower()
            if any(needle in key_lower for needle in needles):
                return str(value)
        return None

    out: dict[str, str] = {}
    captured = _find("datetimeoriginal", "datetime")
    make = _find("make")
    model = _find("model")
    if captured:
        out["captured"] = captured
    if make:
        out["make"] = make
    if model:
        out["model"] = model
    return out


def read(original: Path) -> dict[str, Any]:
    """Return ``{width, height, bytes, exif}`` for ``original`` (best-effort)."""
    original = Path(original)
    info: dict[str, Any] = {"width": None, "height": None, "bytes": None, "exif": {}}
    try:
        info["bytes"] = original.stat().st_size
    except OSError:
        pass
    try:
        img = Image.imread(str(original))
        arr = img.rgb[:]
        info["height"], info["width"] = int(arr.shape[0]), int(arr.shape[1])
        imported = dict(getattr(img._metadata, "imported", {}) or {})
    except Exception:  # noqa: BLE001 - metadata is best-effort
        logger.debug("metadata read failed for %s", original, exc_info=True)
        return info
    info["exif"] = _extract_exif(imported)
    return info
```

- [ ] **Step 4: Run tests + lint**

Run: `uv run pytest tests/gui/browse/test_metadata.py -v && uv run ruff check --fix src/phenotypic/gui/browse/_metadata.py`
Expected: PASS (3 tests); ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/browse/_metadata.py tests/gui/browse/test_metadata.py
git commit -m "feat(gui): browse _metadata — dims, size, EXIF"
```

---

## Task 7: `_tile_routes.py` — token-keyed DZI blueprint

**Files:**
- Create: `src/phenotypic/gui/browse/_tile_routes.py`
- Test: `tests/gui/browse/test_tile_routes.py`

- [ ] **Step 1: Write the failing test**

`tests/gui/browse/test_tile_routes.py`:
```python
import dash
import numpy as np
import pytest
from PIL import Image as PILImage

from phenotypic.gui.browse import _source_render as sr
from phenotypic.gui.browse import _tile_routes
from phenotypic.gui.shell._sandbox import SandboxRoot


@pytest.fixture
def app_and_root(monkeypatch, tmp_path):
    # Redirect the ephemeral cache into the test's tmp dir.
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(cache))
    # Sandbox root with one image.
    sandbox_root = tmp_path / "sandbox"
    (sandbox_root / "plates" / "b7").mkdir(parents=True)
    img = sandbox_root / "plates" / "b7" / "A1.png"
    PILImage.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(img)
    sandbox = SandboxRoot.from_path(sandbox_root)
    app = dash.Dash(__name__)
    _tile_routes.register(app, sandbox)
    return app.server.test_client(), "plates/b7/A1.png"


def test_manifest_then_tile(app_and_root):
    client, rel = app_and_root
    token = sr.encode_token(rel)
    manifest = client.get(f"/tiles/{token}.dzi")
    assert manifest.status_code == 200
    assert b"<Image" in manifest.data
    tile = client.get(f"/tiles/{token}_files/0/0_0.png")
    assert tile.status_code == 200
    assert tile.mimetype == "image/png"


def test_malformed_token_404(app_and_root):
    client, _ = app_and_root
    assert client.get("/tiles/not%20a%20token.dzi").status_code == 404


def test_escape_token_404(app_and_root):
    client, _ = app_and_root
    token = sr.encode_token("../../etc/passwd")
    assert client.get(f"/tiles/{token}.dzi").status_code == 404


def test_raw_unavailable_422(app_and_root, monkeypatch, tmp_path):
    client, _ = app_and_root

    def _boom(original, cache_png):
        raise sr.SourceRenderUnavailable("nope")

    monkeypatch.setattr(_tile_routes._source_render, "normalize_to_png", _boom)
    # A token that resolves to the existing image so render is attempted.
    token = sr.encode_token("plates/b7/A1.png")
    assert client.get(f"/tiles/{token}.dzi").status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_tile_routes.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `_tile_routes.py`**

`src/phenotypic/gui/browse/_tile_routes.py`:
```python
"""Flask blueprint serving token-keyed DZI manifests + tiles for Browse.

The frontend points OpenSeadragon at ``/tiles/<token>.dzi`` where ``<token>``
is a slash-free base64url encoding of the image's path relative to the frozen
``SandboxRoot``. The blueprint validates + decodes the token, resolves the
original file through ``sandbox.resolve`` (the sole security boundary),
normalizes it to a cached 8-bit PNG, and lazily tiles it with the shared DZI
tiler. Mirrors ``results_viewer/_tile_routes.py`` with one token segment in
place of ``<dataset>/<stem>``.
"""
from __future__ import annotations

import logging
import re

import dash
from flask import Blueprint, Response, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from phenotypic.gui._config import BROWSE_TILES_PREFIX
from phenotypic.gui.browse import _source_render
from phenotypic.gui.results_viewer import _dzi_tiler
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

#: DZI tile filenames are ``<col>_<row>.png`` per the OpenSeadragon spec.
_TILE_NAME_RE = re.compile(r"^\d+_\d+\.png$")
#: base64url alphabet (no padding) — what ``encode_token`` produces.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")

__all__ = ["register"]


def register(app: dash.Dash, sandbox: SandboxRoot) -> None:
    """Mount the token-keyed DZI routes on ``app.server``."""
    bp = Blueprint("browse_tiles", __name__, url_prefix=BROWSE_TILES_PREFIX)

    def _resolve_original(token: str):
        if not _TOKEN_RE.match(token):
            return None
        try:
            rel = _source_render.decode_token(token)
        except Exception:  # noqa: BLE001 - malformed token
            return None
        try:
            resolved = sandbox.resolve(rel)
        except ValueError:
            return None
        if not resolved.is_file():
            return None
        return resolved

    @bp.route("/<token>.dzi")
    def manifest(token: str) -> Response:
        original = _resolve_original(token)
        if original is None:
            return _json_error("invalid or unknown image", 404)
        cache_png = _source_render.cache_png_path(token)
        try:
            _source_render.normalize_to_png(original, cache_png)
        except _source_render.SourceRenderUnavailable as exc:
            return _json_error(str(exc), 422)
        except Exception:
            logger.exception("source render failed for token=%s", token)
            return _json_error("render failed", 500)
        try:
            _dzi_tiler.tile(cache_png, _source_render.browse_cache_base())
        except Exception:
            logger.exception("DZI tiling failed for token=%s", token)
            return _json_error("tile generation failed", 500)
        return send_from_directory(
            _source_render.browse_cache_base(),
            f"{token}.dzi",
            mimetype="application/xml",
        )

    @bp.route("/<token>_files/<int:level>/<filename>")
    def tile_endpoint(token: str, level: int, filename: str) -> Response:
        if not _TOKEN_RE.match(token):
            return _json_error("invalid token", 404)
        secured = secure_filename(filename)
        if secured != filename or not _TILE_NAME_RE.match(filename):
            return _json_error("invalid tile filename", 404)
        tile_dir = _source_render.browse_cache_base() / f"{token}_files" / str(level)
        if not tile_dir.is_dir():
            return _json_error("tile cache missing", 404)
        return send_from_directory(tile_dir, filename, mimetype="image/png")

    app.server.register_blueprint(bp)
    logger.debug("Registered Browse tile routes under %s", BROWSE_TILES_PREFIX)


def _json_error(message: str, status: int) -> Response:
    response = jsonify({"error": message})
    response.status_code = status
    return response
```

- [ ] **Step 4: Run tests + lint + types**

Run: `uv run pytest tests/gui/browse/test_tile_routes.py -v && uv run ruff check --fix src/phenotypic/gui/browse/_tile_routes.py && uv run mypy src/phenotypic/gui/browse/_tile_routes.py`
Expected: PASS (4 tests); ruff clean; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/browse/_tile_routes.py tests/gui/browse/test_tile_routes.py
git commit -m "feat(gui): browse _tile_routes — token-keyed DZI blueprint"
```

---

## Task 8: `_ids.py` — component ids

**Files:**
- Create: `src/phenotypic/gui/browse/_ids.py`
- Test: `tests/gui/browse/test_ids.py`

- [ ] **Step 1: Write the failing test**

`tests/gui/browse/test_ids.py`:
```python
from phenotypic.gui.browse import _ids


def test_ids_are_unique_nonempty_strings():
    values = [
        _ids.BROWSE_DATASET_ROW,
        _ids.BROWSE_DATASET_PICKER,
        _ids.BROWSE_IMAGE_PICKER,
        _ids.BROWSE_PREV_BTN,
        _ids.BROWSE_NEXT_BTN,
        _ids.BROWSE_OSD_DIV,
        _ids.BROWSE_CURRENT_IMAGE_STORE,
        _ids.BROWSE_DATASETS_STORE,
        _ids.BROWSE_OSD_SYNC,
        _ids.BROWSE_META_DIMS,
        _ids.BROWSE_META_SIZE,
        _ids.BROWSE_META_CAPTURED,
        _ids.BROWSE_META_CAMERA,
        _ids.BROWSE_EMPTY_HINT,
    ]
    assert all(isinstance(v, str) and v for v in values)
    assert len(set(values)) == len(values)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_ids.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `_ids.py`**

`src/phenotypic/gui/browse/_ids.py`:
```python
"""Static Dash component ids for the Browse tab (single source of truth)."""
from __future__ import annotations

BROWSE_DATASET_ROW = "browse-dataset-row"          # wrapper, hidden when flat
BROWSE_DATASET_PICKER = "browse-dataset-picker"
BROWSE_IMAGE_PICKER = "browse-image-picker"
BROWSE_PREV_BTN = "browse-prev-btn"
BROWSE_NEXT_BTN = "browse-next-btn"
BROWSE_OSD_DIV = "browse-osd-div"                  # OSD mounts here (.osd-canvas)
BROWSE_CURRENT_IMAGE_STORE = "browse-current-image-store"  # {token, label}
BROWSE_DATASETS_STORE = "browse-datasets-store"    # {dataset_rel: [filename,...]}
BROWSE_OSD_SYNC = "browse-osd-sync"                # dummy clientside-callback sink
BROWSE_META_DIMS = "browse-meta-dims"
BROWSE_META_SIZE = "browse-meta-size"
BROWSE_META_CAPTURED = "browse-meta-captured"
BROWSE_META_CAMERA = "browse-meta-camera"
BROWSE_EMPTY_HINT = "browse-empty-hint"            # shown when no source root

__all__ = [
    "BROWSE_DATASET_ROW",
    "BROWSE_DATASET_PICKER",
    "BROWSE_IMAGE_PICKER",
    "BROWSE_PREV_BTN",
    "BROWSE_NEXT_BTN",
    "BROWSE_OSD_DIV",
    "BROWSE_CURRENT_IMAGE_STORE",
    "BROWSE_DATASETS_STORE",
    "BROWSE_OSD_SYNC",
    "BROWSE_META_DIMS",
    "BROWSE_META_SIZE",
    "BROWSE_META_CAPTURED",
    "BROWSE_META_CAMERA",
    "BROWSE_EMPTY_HINT",
]
```

- [ ] **Step 4: Run test + commit**

Run: `uv run pytest tests/gui/browse/test_ids.py -v`
Expected: PASS.
```bash
git add src/phenotypic/gui/browse/_ids.py tests/gui/browse/test_ids.py
git commit -m "feat(gui): browse _ids — component id constants"
```

---

## Task 9: `_callbacks.py` — pure helpers (TDD) + registration

**Files:**
- Create: `src/phenotypic/gui/browse/_callbacks.py`
- Test: `tests/gui/browse/test_callbacks_helpers.py`

Callbacks are thin Dash adapters around the pure helpers below; we unit-test the helpers (per memory `gui_review_verify_with_browser`, live wiring is checked in Task 13's browser pass).

- [ ] **Step 1: Write the failing test**

`tests/gui/browse/test_callbacks_helpers.py`:
```python
from phenotypic.gui.browse import _callbacks as cb


def test_dataset_options_sorted_labels():
    datasets = {".": ["a.png"], "plates/b7": ["A1.png"]}
    opts = cb.dataset_options(datasets)
    assert opts == [
        {"label": "(root)", "value": "."},
        {"label": "plates/b7", "value": "plates/b7"},
    ]


def test_image_options_for_selected_dataset():
    datasets = {"plates/b7": ["A1.png", "A2.png"]}
    assert cb.image_options(datasets, "plates/b7") == [
        {"label": "A1.png", "value": "A1.png"},
        {"label": "A2.png", "value": "A2.png"},
    ]
    assert cb.image_options(datasets, "missing") == []


def test_dataset_row_hidden_when_flat():
    assert cb.dataset_row_hidden({".": ["a.png"]}) is True
    assert cb.dataset_row_hidden({"plates": ["a.png"]}) is False
    assert cb.dataset_row_hidden({}) is True


def test_sandbox_rel_joins_src_dataset_filename():
    assert cb.sandbox_rel("plates/b7", "day3", "A1.png") == "plates/b7/day3/A1.png"
    assert cb.sandbox_rel("plates/b7", ".", "A1.png") == "plates/b7/A1.png"
    assert cb.sandbox_rel(".", ".", "A1.png") == "A1.png"


def test_current_image_payload_round_trips_token():
    from phenotypic.gui.browse._source_render import decode_token

    payload = cb.current_image_payload("plates/b7", ".", "A1.png")
    assert decode_token(payload["token"]) == "plates/b7/A1.png"
    assert payload["label"] == "plates/b7/A1.png"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_callbacks_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `_callbacks.py`**

`src/phenotypic/gui/browse/_callbacks.py`:
```python
"""Browse-tab callbacks + the pure helpers they wrap.

The helpers (``dataset_options``/``image_options``/``dataset_row_hidden``/
``sandbox_rel``/``current_image_payload``) are unit-tested; the Dash
callbacks are thin adapters so the live wiring is the only thing that needs
a browser smoke check.
"""
from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import Any

import dash
from dash import Input, Output, State, ctx, html, no_update

from phenotypic.gui._shared._picker_navigation import (
    picker_button_disabled_states,
    step_picker_value,
)
from phenotypic.gui.browse import _ids as ids
from phenotypic.gui.browse import _metadata, _source_lister, _source_render
from phenotypic.gui.shell._ids import SHELL_SOURCE_IMAGE_ROOT_STORE
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._source_context import resolve_source_image_root

logger = logging.getLogger(__name__)

__all__ = ["register_callbacks"]

_ROOT_LABEL = "(root)"


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
def dataset_options(datasets: dict[str, list[str]]) -> list[dict[str, str]]:
    """Dropdown options for the dataset picker (``.`` shown as ``(root)``)."""
    return [
        {"label": _ROOT_LABEL if key == "." else key, "value": key}
        for key in datasets
    ]


def image_options(datasets: dict[str, list[str]], dataset: str | None) -> list[dict[str, str]]:
    """Dropdown options for the image picker within ``dataset``."""
    return [{"label": name, "value": name} for name in datasets.get(dataset or "", [])]


def dataset_row_hidden(datasets: dict[str, list[str]]) -> bool:
    """True when the dataset dropdown should be hidden (flat or empty source)."""
    return set(datasets.keys()) in ({"."}, set())


def sandbox_rel(src_root_rel: str, dataset_rel: str, filename: str) -> str:
    """Join the image's path relative to the sandbox root (POSIX)."""
    parts = [p for p in (src_root_rel, dataset_rel) if p and p != "."]
    return PurePosixPath(*parts, filename).as_posix() if parts else filename


def current_image_payload(
    src_root_rel: str, dataset_rel: str, filename: str
) -> dict[str, str]:
    """Build the ``{token, label}`` current-image store payload."""
    rel = sandbox_rel(src_root_rel, dataset_rel, filename)
    return {"token": _source_render.encode_token(rel), "label": rel}


def _src_root_rel(sandbox: SandboxRoot, payload: Any) -> str | None:
    """Resolve the source root and return its path relative to the sandbox."""
    resolved = resolve_source_image_root(sandbox, payload)
    if resolved is None:
        return None
    try:
        return resolved.relative_to(sandbox.root).as_posix()
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Callback registration
# --------------------------------------------------------------------------
def register_callbacks(app: dash.Dash, sandbox: SandboxRoot) -> None:
    """Register every Browse callback on ``app``."""

    @app.callback(
        Output(ids.BROWSE_DATASETS_STORE, "data"),
        Output(ids.BROWSE_DATASET_PICKER, "options"),
        Output(ids.BROWSE_DATASET_PICKER, "value"),
        Output(ids.BROWSE_DATASET_ROW, "style"),
        Output(ids.BROWSE_EMPTY_HINT, "style"),
        Input(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
    )
    def _load_datasets(payload: Any):
        resolved = resolve_source_image_root(sandbox, payload)
        if resolved is None:
            return {}, [], None, {"display": "none"}, {"display": "block"}
        datasets = _source_lister.list_datasets(resolved)
        options = dataset_options(datasets)
        value = options[0]["value"] if options else None
        row_style = {"display": "none"} if dataset_row_hidden(datasets) else {}
        hint_style = {"display": "block"} if not datasets else {"display": "none"}
        return datasets, options, value, row_style, hint_style

    @app.callback(
        Output(ids.BROWSE_IMAGE_PICKER, "options"),
        Output(ids.BROWSE_IMAGE_PICKER, "value"),
        Input(ids.BROWSE_DATASET_PICKER, "value"),
        State(ids.BROWSE_DATASETS_STORE, "data"),
    )
    def _cascade_images(dataset: str | None, datasets: dict | None):
        options = image_options(datasets or {}, dataset)
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output(ids.BROWSE_IMAGE_PICKER, "value", allow_duplicate=True),
        Input(ids.BROWSE_PREV_BTN, "n_clicks"),
        Input(ids.BROWSE_NEXT_BTN, "n_clicks"),
        State(ids.BROWSE_IMAGE_PICKER, "value"),
        State(ids.BROWSE_IMAGE_PICKER, "options"),
        prevent_initial_call=True,
    )
    def _step_image(_p, _n, value, options):
        triggered = ctx.triggered_id
        direction = "previous" if triggered == ids.BROWSE_PREV_BTN else "next"
        return step_picker_value(value, options, direction) or no_update

    @app.callback(
        Output(ids.BROWSE_PREV_BTN, "disabled"),
        Output(ids.BROWSE_NEXT_BTN, "disabled"),
        Input(ids.BROWSE_IMAGE_PICKER, "value"),
        Input(ids.BROWSE_IMAGE_PICKER, "options"),
    )
    def _bounds(value, options):
        return picker_button_disabled_states(value, options)

    @app.callback(
        Output(ids.BROWSE_CURRENT_IMAGE_STORE, "data"),
        Input(ids.BROWSE_IMAGE_PICKER, "value"),
        State(ids.BROWSE_DATASET_PICKER, "value"),
        State(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
    )
    def _current_image(filename, dataset, payload):
        if not filename:
            return None
        src_root_rel = _src_root_rel(sandbox, payload)
        if src_root_rel is None:
            return None
        return current_image_payload(src_root_rel, dataset or ".", filename)

    @app.callback(
        Output(ids.BROWSE_META_DIMS, "children"),
        Output(ids.BROWSE_META_SIZE, "children"),
        Output(ids.BROWSE_META_CAPTURED, "children"),
        Output(ids.BROWSE_META_CAMERA, "children"),
        Input(ids.BROWSE_CURRENT_IMAGE_STORE, "data"),
    )
    def _metadata_panel(payload: dict | None):
        if not payload or not payload.get("token"):
            return "—", "—", "—", "—"
        try:
            rel = _source_render.decode_token(payload["token"])
            original = sandbox.resolve(rel)
        except (ValueError, Exception):  # noqa: BLE001
            return "—", "—", "—", "—"
        info = _metadata.read(original)
        dims = (
            f"{info['width']} × {info['height']} px"
            if info["width"] and info["height"]
            else "—"
        )
        size = _humanize_bytes(info["bytes"]) if info["bytes"] else "—"
        exif = info.get("exif", {})
        captured = exif.get("captured", "—")
        camera = " ".join(p for p in (exif.get("make"), exif.get("model")) if p) or "—"
        return dims, size, captured, camera

    # Clientside: mount/replace the single OSD viewer on image change.
    app.clientside_callback(
        """
        function(payload) {
            if (window.__phenotypicBrowse) {
                window.__phenotypicBrowse.applyImage(payload);
            }
            return "";
        }
        """,
        Output(ids.BROWSE_OSD_SYNC, "data"),
        Input(ids.BROWSE_CURRENT_IMAGE_STORE, "data"),
    )


def _humanize_bytes(n: int) -> str:
    """Compact human-readable file size."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# Re-exported for symmetry with other tools (unused placeholder import guard).
_ = html
```

- [ ] **Step 4: Run helper tests + lint**

Run: `uv run pytest tests/gui/browse/test_callbacks_helpers.py -v && uv run ruff check --fix src/phenotypic/gui/browse/_callbacks.py`
Expected: PASS (5 tests); ruff clean. (Remove the `_ = html` guard if ruff flags `html` as unused — it is only kept if a later edit needs it.)

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/browse/_callbacks.py tests/gui/browse/test_callbacks_helpers.py
git commit -m "feat(gui): browse _callbacks — cascade picker, token store, metadata"
```

---

## Task 10: `_layout.py` — single-pane layout

**Files:**
- Create: `src/phenotypic/gui/browse/_layout.py`
- Test: `tests/gui/browse/test_layout.py`

- [ ] **Step 1: Write the failing test**

`tests/gui/browse/test_layout.py`:
```python
from dash import dcc, html

from phenotypic.gui.browse import _ids as ids
from phenotypic.gui.browse._layout import build_browse_layout


def _ids_in_tree(node, found):
    cid = getattr(node, "id", None)
    if cid:
        found.add(cid)
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for c in children:
            _ids_in_tree(c, found)
    elif children is not None:
        _ids_in_tree(children, found)


def test_layout_contains_core_ids():
    found: set[str] = set()
    _ids_in_tree(build_browse_layout(), found)
    for required in (
        ids.BROWSE_DATASET_PICKER,
        ids.BROWSE_IMAGE_PICKER,
        ids.BROWSE_PREV_BTN,
        ids.BROWSE_NEXT_BTN,
        ids.BROWSE_OSD_DIV,
        ids.BROWSE_CURRENT_IMAGE_STORE,
        ids.BROWSE_DATASETS_STORE,
        ids.BROWSE_OSD_SYNC,
        ids.BROWSE_META_DIMS,
        ids.BROWSE_EMPTY_HINT,
    ):
        assert required in found
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_layout.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `_layout.py`**

`src/phenotypic/gui/browse/_layout.py`:
```python
"""Single-pane Browse layout: dataset + image pickers, OSD canvas, metadata."""
from __future__ import annotations

from typing import Any, cast

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.gui._design import COLOR_MUTED, FONT_SIZE_CAPTION
from phenotypic.gui.browse import _ids as ids

__all__ = ["build_browse_layout"]

_OSD_STYLE = {"height": "70vh", "width": "100%"}


def _meta_chip(label: str, value_id: str) -> Any:
    return html.Div(
        [
            html.Span(
                label,
                style={
                    "color": COLOR_MUTED,
                    "fontSize": FONT_SIZE_CAPTION,
                    "textTransform": "uppercase",
                    "letterSpacing": "0.06em",
                    "marginRight": "0.4rem",
                },
            ),
            html.Span("—", id=value_id),
        ],
        className="browse-meta-chip",
        style={"marginRight": "1.25rem"},
    )


def build_browse_layout() -> Any:
    """Build the Browse page body (chrome is applied by ``wrap_in_chrome``)."""
    dataset_picker = dcc.Dropdown(
        id=ids.BROWSE_DATASET_PICKER,
        options=[],
        value=None,
        placeholder="Dataset…",
        clearable=False,
        searchable=True,
        style={"minWidth": "12rem"},
    )
    dataset_row = html.Div(
        dataset_picker,
        id=ids.BROWSE_DATASET_ROW,
        style={"marginRight": "0.75rem", "flex": "0 0 auto"},
    )

    image_picker = dcc.Dropdown(
        id=ids.BROWSE_IMAGE_PICKER,
        options=[],
        value=None,
        placeholder="Select image…",
        clearable=False,
        searchable=True,
        style={"flex": "1 1 auto", "minWidth": "12rem"},
    )
    picker_group = html.Div(
        [
            html.Button(
                "‹",
                id=ids.BROWSE_PREV_BTN,
                n_clicks=0,
                title="Previous image",
                className="btn btn-outline-secondary btn-sm",
                type="button",
                **cast(Any, {"aria-label": "Previous image"}),
            ),
            html.Div(image_picker, style={"flex": "1 1 auto"}),
            html.Button(
                "›",
                id=ids.BROWSE_NEXT_BTN,
                n_clicks=0,
                title="Next image",
                className="btn btn-outline-secondary btn-sm",
                type="button",
                **cast(Any, {"aria-label": "Next image"}),
            ),
        ],
        className="d-flex align-items-center",
        style={"gap": "0.35rem", "flex": "1 1 auto"},
    )

    header = html.Div(
        [dataset_row, picker_group],
        className="d-flex align-items-center",
        style={"gap": "0.5rem", "flexWrap": "wrap", "marginBottom": "0.75rem"},
    )

    empty_hint = html.Div(
        "No source image root selected. Pick one from the top bar "
        "(“source:”) to browse its images.",
        id=ids.BROWSE_EMPTY_HINT,
        className="text-muted",
        style={"display": "none", "padding": "2rem 0"},
    )

    osd_div = html.Div(
        id=ids.BROWSE_OSD_DIV,
        className="osd-canvas browse-osd-canvas",
        style=_OSD_STYLE,
    )

    metadata_panel = html.Div(
        [
            _meta_chip("Dimensions", ids.BROWSE_META_DIMS),
            _meta_chip("Size", ids.BROWSE_META_SIZE),
            _meta_chip("Captured", ids.BROWSE_META_CAPTURED),
            _meta_chip("Camera", ids.BROWSE_META_CAMERA),
        ],
        className="browse-meta-panel d-flex flex-wrap",
        style={"marginTop": "0.75rem"},
    )

    return html.Div(
        [
            header,
            empty_hint,
            osd_div,
            metadata_panel,
            dcc.Store(id=ids.BROWSE_DATASETS_STORE, data={}),
            dcc.Store(id=ids.BROWSE_CURRENT_IMAGE_STORE, data=None),
            dcc.Store(id=ids.BROWSE_OSD_SYNC, data=None),
        ],
        className="browse-page",
        style={"padding": "1rem"},
    )
```

- [ ] **Step 4: Run test + lint**

Run: `uv run pytest tests/gui/browse/test_layout.py -v && uv run ruff check --fix src/phenotypic/gui/browse/_layout.py`
Expected: PASS; ruff clean. (If `COLOR_MUTED`/`FONT_SIZE_CAPTION` names differ, confirm against `gui/_design.py` and adjust the import.)

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/browse/_layout.py tests/gui/browse/test_layout.py
git commit -m "feat(gui): browse _layout — single-pane pickers + OSD + metadata"
```

---

## Task 11: `_app.py` + `__main__.py` — Dash factory

**Files:**
- Create: `src/phenotypic/gui/browse/_app.py`
- Create: `src/phenotypic/gui/browse/__main__.py`
- Test: `tests/gui/browse/test_app.py`

- [ ] **Step 1: Write the failing test**

`tests/gui/browse/test_app.py`:
```python
import numpy as np
import pytest
from PIL import Image as PILImage

from phenotypic.gui.browse import _source_render as sr
from phenotypic.gui.browse._app import create_app
from phenotypic.gui.shell._sandbox import SandboxRoot


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path / "cache"))
    root = tmp_path / "imgs"
    root.mkdir()
    PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(root / "a.png")
    return SandboxRoot.from_path(root)


def test_create_app_serves_layout_and_tiles(sandbox):
    app = create_app(sandbox, url_prefix="/")
    client = app.server.test_client()
    # Dash layout endpoint responds.
    assert client.get("/_dash-layout").status_code == 200
    # Tile blueprint is mounted.
    token = sr.encode_token("a.png")
    assert client.get(f"/tiles/{token}.dzi").status_code == 200


def test_create_app_injects_app_prefix(sandbox):
    app = create_app(sandbox, url_prefix="/browse/")
    assert "window.__phenotypicAppPrefix" in app.index_string
    assert "/browse/" in app.index_string
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `_app.py`**

`src/phenotypic/gui/browse/_app.py`:
```python
"""Dash app factory for the Browse tab.

Eager, lightweight (no heavy parquet load → no ToolSession). Mounts the
token-keyed tile blueprint, wipes + initialises the ephemeral cache, injects
``window.__phenotypicAppPrefix`` (so ``browse.js`` builds hub-aware tile +
OSD-asset URLs), builds the layout, and registers callbacks.
"""
from __future__ import annotations

import logging
from pathlib import Path

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]

from phenotypic.gui._config import CFG_URL_PREFIX, MOUNT_HOME, TITLE_BROWSE
from phenotypic.gui._design import inject_design_tokens
from phenotypic.gui._shared import register_shared_static
from phenotypic.gui.browse import _source_render, _tile_routes
from phenotypic.gui.browse._callbacks import register_callbacks
from phenotypic.gui.browse._layout import build_browse_layout
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

__all__ = ["create_app"]


def _index_string_with_prefix(url_prefix: str) -> str:
    """Dash ``index_string`` that exposes ``window.__phenotypicAppPrefix``."""
    safe_prefix = (
        url_prefix.replace("\\", "\\\\").replace('"', '\\"').replace("</", "<\\/")
    )
    return (
        "<!DOCTYPE html>\n<html>\n    <head>\n"
        "        {%metas%}\n        <title>{%title%}</title>\n"
        "        {%favicon%}\n        {%css%}\n"
        f'        <script>window.__phenotypicAppPrefix = "{safe_prefix}";</script>\n'
        "    </head>\n    <body>\n        {%app_entry%}\n        <footer>\n"
        "            {%config%}\n            {%scripts%}\n            {%renderer%}\n"
        "        </footer>\n    </body>\n</html>"
    )


def create_app(sandbox: SandboxRoot, *, url_prefix: str = MOUNT_HOME) -> dash.Dash:
    """Build the Browse Dash app.

    Args:
        sandbox: Frozen-at-launch sandbox root (security boundary + path base).
        url_prefix: Mount prefix. ``"/"`` standalone; hub passes ``"/browse/"``.
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder=str(Path(__file__).parent / "_assets"),
        title=TITLE_BROWSE,
        requests_pathname_prefix=url_prefix,
        routes_pathname_prefix=MOUNT_HOME,
    )
    app.index_string = _index_string_with_prefix(url_prefix)
    inject_design_tokens(app)
    register_shared_static(app.server)
    app.server.config[CFG_URL_PREFIX] = url_prefix

    _source_render.init_cache()  # wipe stale tiles + register atexit cleanup
    _tile_routes.register(app, sandbox)
    app.layout = build_browse_layout()
    register_callbacks(app, sandbox)

    logger.debug("Browse app built: sandbox=%s url_prefix=%s", sandbox.root, url_prefix)
    return app
```

- [ ] **Step 4: Implement `__main__.py`**

`src/phenotypic/gui/browse/__main__.py`:
```python
"""Standalone launcher: ``python -m phenotypic.gui.browse --root ./images``."""
from __future__ import annotations

import argparse

from phenotypic.gui._config import (
    MOUNT_HOME,
    add_launcher_args,
    configure_launcher_logging,
    print_launcher_banner,
)
from phenotypic.gui.browse._app import create_app
from phenotypic.gui.shell._sandbox import SandboxRoot


def main() -> None:
    parser = argparse.ArgumentParser(description="PhenoTypic Source Browser")
    parser.add_argument("--root", default=".", help="Sandbox / source image root")
    add_launcher_args(parser)
    args = parser.parse_args()
    configure_launcher_logging(debug=args.debug)
    sandbox = SandboxRoot.from_path(args.root)
    app = create_app(sandbox, url_prefix=MOUNT_HOME)
    print_launcher_banner("PhenoTypic Source Browser", host=args.host, port=args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
```
(If `print_launcher_banner`'s signature differs, match `run_console/__main__.py`'s call exactly.)

- [ ] **Step 5: Run tests + lint + types**

Run: `uv run pytest tests/gui/browse/test_app.py -v && uv run ruff check --fix src/phenotypic/gui/browse/_app.py src/phenotypic/gui/browse/__main__.py && uv run mypy src/phenotypic/gui/browse`
Expected: PASS (2 tests); ruff clean; mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/browse/_app.py src/phenotypic/gui/browse/__main__.py tests/gui/browse/test_app.py
git commit -m "feat(gui): browse _app + standalone __main__"
```

---

## Task 12: Frontend — vendored OSD + `browse.js` + `browse.css`

**Files:**
- Create: `src/phenotypic/gui/browse/_assets/openseadragon/` (copied)
- Create: `src/phenotypic/gui/browse/_assets/browse.js`
- Create: `src/phenotypic/gui/browse/_assets/browse.css`

No unit test (browser-only); verified live in Task 13.

- [ ] **Step 1: Copy the vendored OpenSeadragon assets**

Run:
```bash
cp -R src/phenotypic/gui/results_viewer/_assets/openseadragon \
      src/phenotypic/gui/browse/_assets/openseadragon
ls src/phenotypic/gui/browse/_assets/openseadragon
```
Expected: lists `openseadragon.min.js` + an `images/` dir.
(Optimization noted in spec — a single shared static route — is deferred; copying is the robust v1 choice.)

- [ ] **Step 2: Write `browse.js`**

`src/phenotypic/gui/browse/_assets/browse.js`:
```javascript
/*
 * browse.js — single-viewport OpenSeadragon lifecycle for the Browse tab.
 * Loads OSD from the vendored copy (no CDN; offline-safe over a tunnel) and
 * exposes window.__phenotypicBrowse.applyImage({token,label}), invoked by a
 * Dash clientside callback when the current-image store changes.
 */
(function () {
    "use strict";
    const appPrefix = (typeof window.__phenotypicAppPrefix === "string"
        && window.__phenotypicAppPrefix.length > 0)
        ? window.__phenotypicAppPrefix : "/";
    const OSD_DIV_ID = "browse-osd-div";

    const ns = window.__phenotypicBrowse = window.__phenotypicBrowse || {};
    ns.viewer = ns.viewer || null;

    function loadOSD() {
        return new Promise(function (resolve, reject) {
            if (window.OpenSeadragon) { resolve(); return; }
            const tag = document.createElement("script");
            tag.src = appPrefix + "assets/openseadragon/openseadragon.min.js";
            tag.async = true;
            tag.onload = function () { resolve(); };
            tag.onerror = function () { reject(new Error("OSD vendored load failed")); };
            document.head.appendChild(tag);
        });
    }
    ns.osdReady = ns.osdReady || loadOSD();

    ns.applyImage = async function (payload) {
        await ns.osdReady;
        const el = document.getElementById(OSD_DIV_ID);
        if (!el) { return; }
        if (!payload || !payload.token) {
            if (ns.viewer) { try { ns.viewer.destroy(); } catch (e) {} ns.viewer = null; }
            return;
        }
        const dziUrl = appPrefix + "tiles/" + encodeURIComponent(payload.token) + ".dzi";
        if (ns.viewer && ns.viewer._phenotypicDziUrl === dziUrl) { return; }
        if (ns.viewer) { try { ns.viewer.destroy(); } catch (e) {} ns.viewer = null; }
        const viewer = window.OpenSeadragon({
            element: el,
            prefixUrl: appPrefix + "assets/openseadragon/images/",
            tileSources: dziUrl,
            showNavigator: false,
            showRotationControl: false,
            animationTime: 0.4,
            constrainDuringPan: true,
            visibilityRatio: 0.5,
            minZoomLevel: 0.4,
            maxZoomPixelRatio: 4,
            immediateRender: false,
        });
        viewer._phenotypicDziUrl = dziUrl;
        ns.viewer = viewer;
    };
})();
```

- [ ] **Step 3: Write `browse.css`**

`src/phenotypic/gui/browse/_assets/browse.css`:
```css
.browse-osd-canvas {
    background: var(--color-bg, #0b0b0b);
    border: 1px solid var(--color-border, #ccc);
    border-radius: var(--radius, 6px);
}
.browse-meta-panel {
    font-size: var(--font-size-caption);
    color: var(--color-body);
}
.browse-meta-chip span:last-child {
    font-family: var(--font-mono);
}
```

- [ ] **Step 4: Sanity-check the assets are discoverable**

Run:
```bash
uv run python -c "
from phenotypic.gui.browse._app import create_app
from phenotypic.gui.shell._sandbox import SandboxRoot
import tempfile, pathlib
app = create_app(SandboxRoot.from_path(tempfile.gettempdir()), url_prefix='/browse/')
c = app.server.test_client()
print('js', c.get('/browse/assets/browse.js').status_code)
print('osd', c.get('/browse/assets/openseadragon/openseadragon.min.js').status_code)
"
```
Expected: both print `200`.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/browse/_assets
git commit -m "feat(gui): browse frontend — vendored OSD, browse.js single-viewport, css"
```

---

## Task 13: Hub composition + nav + live smoke

**Files:**
- Modify: `src/phenotypic/gui/shell/_layout.py`
- Modify: `src/phenotypic/gui/shell/_app.py`
- Test: `tests/gui/browse/test_hub_mount.py`

- [ ] **Step 1: Write the failing test**

`tests/gui/browse/test_hub_mount.py`:
```python
from phenotypic.gui.shell._app import compose_hub
from phenotypic.gui.shell._sandbox import SandboxRoot


def test_browse_tab_in_nav_model():
    from phenotypic.gui.shell._ids import SHELL_TAB_BROWSE, SHELL_TAB_HOME
    from phenotypic.gui.shell._layout import NAV_MODEL

    # Browse is a leaf immediately after Home.
    assert NAV_MODEL[0] == SHELL_TAB_HOME
    assert NAV_MODEL[1] == SHELL_TAB_BROWSE


def test_hub_serves_browse_mount(tmp_path):
    (tmp_path / "imgs").mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    app, _viewer_session = compose_hub(sandbox, start_idle_thread=False)
    client = app.server.test_client()
    resp = client.get("/browse/")
    assert resp.status_code == 200
    assert b"PhenoTypic" in resp.data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/browse/test_hub_mount.py -v`
Expected: FAIL — `NAV_MODEL[1]` is the Pipeline group, and `/browse/` 404s.

- [ ] **Step 3: Wire the nav (`shell/_layout.py`)**

In `src/phenotypic/gui/shell/_layout.py`:
- Add to the `from phenotypic.gui._config import (...)` block: `MOUNT_BROWSE`.
- Add to the `from phenotypic.gui.shell._ids import (...)` block: `SHELL_TAB_BROWSE`.
- Add to `_TAB_HREFS`: `SHELL_TAB_BROWSE: MOUNT_BROWSE,`
- Add to `_TAB_LABELS`: `SHELL_TAB_BROWSE: "Browse",`
- Change `NAV_MODEL` to insert the Browse leaf after Home:
```python
NAV_MODEL: tuple["str | _NavGroup", ...] = (
    SHELL_TAB_HOME,
    SHELL_TAB_BROWSE,
    _NavGroup(
        "Pipeline",
        SHELL_TAB_GROUP_PIPELINE,
        (SHELL_TAB_BUILDER, SHELL_TAB_TUNE, SHELL_TAB_RUN),
    ),
    _NavGroup(
        "Results",
        SHELL_TAB_GROUP_RESULTS,
        (SHELL_TAB_VIEWER, SHELL_TAB_ANALYSIS),
    ),
)
```

- [ ] **Step 4: Wire the mount (`shell/_app.py`)**

In `src/phenotypic/gui/shell/_app.py`:
- Add `MOUNT_BROWSE` to the `from phenotypic.gui._config import (...)` block.
- Add `SHELL_TAB_BROWSE` to the `from phenotypic.gui.shell._ids import (...)` block.
- Inside `compose_hub`, add `browse` to the local import:
  `from phenotypic.gui import analysis, browse, builder, results_viewer, run_console, tune`
- After the tune app block (step 4b), add:
```python
    # 4c. Browse Dash (eager — lightweight source-image viewer). No
    #     ToolSession: it loads no heavy parquet, just lists files + serves
    #     ephemeral tiles.
    browse_app = browse.create_app(sandbox, url_prefix=MOUNT_BROWSE)
    wrap_in_chrome(browse_app, active_tab=SHELL_TAB_BROWSE, sandbox=sandbox)
```
- Add the mount to the `DispatcherMiddleware` map (alongside the others):
```python
            MOUNT_BROWSE.rstrip("/"): browse_app.server,
```
- Add `MOUNT_BROWSE` to the `logger.info("GUI hub composed: ...")` mounts list (extend the format string + args).

- [ ] **Step 5: Run tests + lint + types**

Run: `uv run pytest tests/gui/browse/test_hub_mount.py -v && uv run ruff check --fix src/phenotypic/gui/shell/_layout.py src/phenotypic/gui/shell/_app.py && uv run mypy src/phenotypic/gui/shell/_app.py`
Expected: PASS (2 tests); ruff clean; mypy clean.

- [ ] **Step 6: Full browse test sweep + regression on shell**

Run: `uv run pytest tests/gui/browse tests/unit/gui/shell tests/gui/results_viewer/test_dzi_tiler.py -v`
Expected: all PASS (browse suite + shell unit tests + the reused tiler test).

- [ ] **Step 7: Live browser smoke (manual / Playwright MCP)**

Per memory `gui_review_verify_with_browser`, callback-wiring bugs only surface on `/_dash-update-component`. Launch and drive it:
```bash
uv run phenotypic-gui --root ./images --port 8050   # background; use a dir with nested + flat image folders
```
Then in a browser (or Playwright MCP), verify, tailing the launch log for 500s:
1. Click the **Browse** tab → loads; with no source set, the empty hint shows.
2. Set a source root from the top bar → dataset dropdown populates (hidden if the folder is flat); image dropdown auto-selects the first image; OSD renders it.
3. ‹/› step images; buttons disable at the first/last image.
4. Switch dataset → image dropdown re-populates + first image renders.
5. Zoom/pan works; metadata panel shows dims/size (+ EXIF for a JPEG/RAW).

Expected: no `/_dash-update-component` 500s in the log; all five behaviours work.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/gui/shell/_layout.py src/phenotypic/gui/shell/_app.py tests/gui/browse/test_hub_mount.py
git commit -m "feat(gui): mount Browse tab in the hub nav + dispatcher"
```

---

## Task 14: Ledgers, tutorial, screenshots (CI-gated)

**Files:**
- Modify: `src/phenotypic/gui/FEATURES.md`
- Modify: `src/phenotypic/gui/WORKFLOWS.md`
- Modify: `scripts/capture_gui_tutorial_screenshots.py`
- Create: `docs/source/tutorials/gui/browse.rst` (match the existing tutorial format in that dir)

- [ ] **Step 1: Add FEATURES.md rows**

Open `src/phenotypic/gui/FEATURES.md`, read the table header + an existing `✅ shipping` row to match the exact column format, and add one row per Browse affordance with a `Test ref` pointing at the Task 4-13 tests:
- Browse tab anchor (nav) — `tests/gui/browse/test_hub_mount.py`
- Dataset dropdown + flat-hide — `tests/gui/browse/test_callbacks_helpers.py`
- Image dropdown + ‹/› stepper + bounds-disable — `tests/gui/browse/test_shared_picker_navigation.py`
- OSD viewport (token tile route) — `tests/gui/browse/test_tile_routes.py`
- Metadata panel — `tests/gui/browse/test_metadata.py`
- Ephemeral temp cache — `tests/gui/browse/test_source_render.py`
- Current-image store + clientside mount — `tests/gui/browse/test_app.py`

- [ ] **Step 2: Add a WORKFLOWS.md row + capture function**

Add one row to `src/phenotypic/gui/WORKFLOWS.md` for "Browse source images" with id `browse` (match the existing column format), then add a matching `_capture_browse` function in `scripts/capture_gui_tutorial_screenshots.py` (model it on an existing `_capture_*` function — set the source root, open `/browse/`, select an image, screenshot the viewport + metadata panel).

- [ ] **Step 3: Add the tutorial page**

Create `docs/source/tutorials/gui/browse.rst` following the format of a sibling page (e.g. the results-viewer tutorial): intro, "select a source", "browse + zoom", "metadata", and a note that the tile cache is ephemeral (`tempfile.gettempdir()/phenotypic/browse`, wiped each session).

- [ ] **Step 4: Validate the gates locally**

Run:
```bash
uv run python scripts/check_workflows_md.py
```
Expected: passes (the `browse` row has a matching `_capture_browse` + tutorial page).

- [ ] **Step 5: Regenerate the full screenshot set + commit everything**

Run: `uv run python scripts/capture_gui_tutorial_screenshots.py`
Expected: regenerates the full PNG set (unrelated tutorials shift a few bytes — commit them all, do not cherry-pick).

```bash
git add src/phenotypic/gui/FEATURES.md src/phenotypic/gui/WORKFLOWS.md \
        scripts/capture_gui_tutorial_screenshots.py docs/source/tutorials/gui/browse.rst \
        docs/source/tutorials/gui/_static  # or wherever PNGs land
git add -A docs/source/tutorials   # capture collateral
git commit -m "docs(gui): Browse tab FEATURES/WORKFLOWS rows, tutorial, screenshots"
```

---

## Final verification

- [ ] **Step 1: Full Browse suite + lint + types**

Run:
```bash
uv run pytest tests/gui/browse -v
uv run ruff check src/phenotypic/gui/browse src/phenotypic/gui/_config.py src/phenotypic/gui/shell
uv run mypy src/phenotypic/gui/browse
```
Expected: all green.

- [ ] **Step 2: Regression sweep on touched shared code**

Run: `uv run pytest tests/gui/results_viewer tests/unit/gui/shell tests/gui/builder -q`
Expected: PASS (the `IMAGE_EXTS` + picker-navigation lifts and the shell nav/mount change don't regress existing tools).

- [ ] **Step 3: Branch wrap-up**

Use `superpowers:finishing-a-development-branch` to decide merge/PR.

---

## Self-Review

**Spec coverage** (each spec section → task):
- F1 OSD+DZI viewport → Tasks 7, 12. ✅
- B1 normalize→PNG + reuse `_dzi_tiler` → Task 4 (`normalize_to_png`) + Task 7. ✅
- A1 stateless token URL + eager app → Tasks 4 (token), 7 (route), 11 (eager `create_app`). ✅
- C1 ephemeral temp cache → Task 4 (`browse_cache_base`, `cache_png_path`). ✅
- C2 wipe-on-start + atexit → Task 4 (`init_cache`/`wipe_cache`) + Task 11 (called in `create_app`). ✅
- U1 two cascading dropdowns + flat-hide + single pane → Tasks 9 (helpers + callbacks), 10 (layout). ✅
- R1 faithful render + bounds-stop stepping → Task 4 (`img_as_ubyte`, no stretch) + Task 9 (`step_picker_value` clamps). ✅
- Mixed formats (standard + RAW) → Task 4 (`Image.imread` uniform + `SourceRenderUnavailable`). ✅
- Metadata panel (dims/size/EXIF) → Tasks 6, 9, 10. ✅
- New top-level Browse tab after Home → Task 13. ✅
- Offline/vendored OSD → Task 12. ✅
- Reuse lifts (`IMAGE_EXTS`, picker-nav) → Tasks 1, 2. ✅
- CI gates (FEATURES/WORKFLOWS/screenshots) → Task 14. ✅

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Two flagged spots are *verification* directions (confirm `_design.py` token names in Task 10; match `print_launcher_banner` signature in Task 11) — these are real "check against the codebase" steps, not missing code.

**Type/name consistency:** `encode_token`/`decode_token`/`browse_cache_base`/`cache_png_path`/`normalize_to_png`/`init_cache`/`SourceRenderUnavailable` (Task 4) are used identically in Tasks 7, 9, 11. `list_datasets` (Task 5) → consumed in Task 9 callbacks. `dataset_options`/`image_options`/`dataset_row_hidden`/`sandbox_rel`/`current_image_payload` (Task 9) tested in Task 9 + used in callbacks. Ids in Task 8 match every reference in Tasks 9–10. `register` (Task 7) / `register_callbacks` (Task 9) / `create_app` (Task 11) signatures match their call sites in Task 11/13.
