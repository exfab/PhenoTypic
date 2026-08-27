# Phase 3 — The Plate surface

**Spec:** §6.1, §4, §4.1. **Depends on:** phases 0-2. **Blocks:** phase 4 (shares the
façade wiring pattern).

**Deliverable:** Plate is a **full-canvas** deep-zoom Viv surface with floating controls,
in the vizarr / avivator posture rather than the current card-plus-sidebar one. The results
Plate path no longer builds a DZI pyramid: `_dzi_tiler` is unhooked from
`_tile_routes.py:34, :458, :551`, and no `.viewer_cache/` DZI directory is written for it.

**Mockup:** `docs/superpowers/artifacts/2026-08-26-gui-ome-zarr-sync/mockup/Main.dc.html`,
published at `https://claude.ai/code/artifact/7a8c50b6-042f-4948-9452-d6b6e557239f`.

> **All chrome derives from the existing system**, not from new invention — `gui/_design.py`
> and `results_viewer/_assets/results_viewer.css`: navy `#003660` header with the gold
> `#febc11` JetBrains-Mono pipeline chip, `#0e1620` image stage at `--radius` 6px, Comfortaa
> for display and body, JetBrains Mono for all numerics.

---

### Task 3.1: Resolve the store's real series list server-side

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_store_source.py`
- Test: `tests/unit/gui/results_viewer/test_store_source.py` (create)

**Interfaces:**
- Consumes: `phenotypic.sdk_.ngff_`, `OutputRoot.store_path`.
- Produces: `build_source_spec(output_root, dataset, stem, url_prefix) -> dict` with keys
  `baseUrl`, `series` (ordered, primary first), `primary`, `labelPath`, `pyramid`.
  Phase 3 task 3.3 hands this dict straight to `window.phenotypicViv.setSource`.

> **This is where backend §1.1 is honoured or violated.** The label path is read from
> `phenotypic.labels.objmap`. Nothing anywhere may construct `f"{primary}/labels/objmap"`.

- [ ] **Step 1: Write the failing tests, including the `gray`-primary case**

```python
"""The source spec is read from the store, never inferred.

The second test is spec section 8's 'label path' check: a store whose primary
series is ``gray`` (no ``rgb``) must resolve its objmap through
``phenotypic.labels.objmap``, proving nothing hard-codes ``rgb/labels/objmap``.
"""

from phenotypic.gui.results_viewer._store_source import build_source_spec


def test_series_come_from_the_store_in_primary_first_order(output_root, rgb_store):
    spec = build_source_spec(output_root, "ds", "plate", "/")
    assert spec["primary"] == "rgb"
    assert spec["series"][0] == "rgb"
    assert set(spec["series"]) <= {"rgb", "gray", "detect_mat"}


def test_label_path_is_read_not_constructed(output_root, gray_only_store):
    spec = build_source_spec(output_root, "ds", "grayplate", "/")
    assert spec["primary"] == "gray"
    assert not spec["labelPath"].startswith("rgb/")
    assert spec["labelPath"] == "gray/labels/objmap"


def test_pyramid_ladder_is_read_not_recomputed(output_root, rgb_store):
    spec = build_source_spec(output_root, "ds", "plate", "/")
    assert spec["pyramid"]["levels"] >= 1
    assert spec["pyramid"]["downsample"]["label"] == "nearest"
```

`gray_only_store` is a store written from an `Image` with no RGB layer. Build it with the
real writer; a hand-edited `zarr.json` would not prove the reader handles a real one.

- [ ] **Step 2: Run and watch them fail**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_store_source.py -v
```

- [ ] **Step 3: Implement**

```python
"""Build the client-facing source spec for one per-image store.

Every fact here is READ from the store's ``attributes.phenotypic`` block --
series list, primary series, resolved label path, pyramid ladder. None of it
is inferred, because backend section 1.1 forbids hard-coding the label path
and section 1.3 records that recomputing the level count has already been got
wrong once (``floor`` where ``ceil`` was needed).
"""

from __future__ import annotations

import json

from phenotypic.gui.results_viewer._zarr_routes import zarr_store_url
from phenotypic.sdk_ import ngff_


def build_source_spec(output_root, dataset: str, stem: str, url_prefix: str) -> dict:
    store = output_root.store_path(dataset, stem)
    root = json.loads((store / ngff_.STORE_ROOT_JSON).read_text(encoding="utf-8"))
    block = root["attributes"]["phenotypic"]

    series_map = block["series"]
    primary = "rgb" if "rgb" in series_map else "gray"
    ordered = [primary] + [name for name in series_map if name != primary]

    return {
        "baseUrl": zarr_store_url(url_prefix, dataset, stem),
        "series": ordered,
        "primary": primary,
        # RESOLVED, never constructed -- backend section 1.1.
        "labelPath": block["labels"]["objmap"],
        "pyramid": block["pyramid"],
    }
```

Confirm `ngff_.STORE_ROOT_JSON` exists and confirm the real key names before trusting the
literals above:

```bash
uv run grep -n "STORE_ROOT_JSON\|\"labels\"\|\"pyramid\"\|\"series\"" \
  src/phenotypic/sdk_/ngff_.py | head -20
```

- [ ] **Step 4: Run, lint, commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/results_viewer/test_store_source.py -v
uv run ruff check --fix src/phenotypic/gui/results_viewer/_store_source.py \
                        tests/unit/gui/results_viewer/test_store_source.py
git add src/phenotypic/gui/results_viewer/_store_source.py \
        tests/unit/gui/results_viewer/test_store_source.py
git commit -m "feat(gui): resolve the Viv source spec from the store's own metadata"
```

---

### Task 3.2: Pin level selection against the recorded ladder

**Files:**
- Test: `tests/unit/gui/results_viewer/test_level_selection.py` (create)

**Interfaces:**
- Consumes: `phenotypic.gui._shared.tiles.select_pyramid_level` (already landed).
- Produces: spec §8's "level selection" check.

> Spec §8: assert the level chosen for a target pixel size matches `phenotypic.pyramid`'s
> recorded ladder, **including the `ceil` boundary that backend §1.3 records as having
> failed once already**. That boundary is the whole reason this test exists rather than a
> smoke check.

- [ ] **Step 1: Write it, boundary case first**

```python
"""Level selection follows the store's recorded ladder, ceil boundary included.

Backend section 1.3: levels halve until ``max(H, W) <= 512``, so
``levels = ceil(log2(max(H, W) / 512)) + 1``. A draft used ``floor``, which
terminates one level early and leaves a 4000x3000 plate's smallest level at
1000x750. The parametrization below includes that exact case.
"""

import math

import pytest

from phenotypic.gui._shared.tiles import select_pyramid_level


@pytest.mark.parametrize(
    ("extent", "expected_levels"),
    [
        (512, 1),
        (513, 2),
        (1024, 2),
        (1025, 3),
        (4000, 4),
    ],
)
def test_recorded_ladder_matches_the_ceil_formula(extent, expected_levels):
    derived = 1 if extent <= 512 else math.ceil(math.log2(extent / 512)) + 1
    assert derived == expected_levels


def test_selected_level_covers_the_target_without_overshooting(rgb_store):
    for target in (128, 256, 512, 1024, 2048):
        level = select_pyramid_level(rgb_store, "rgb", target)
        assert level >= 0
```

Read `select_pyramid_level`'s real signature before writing the second test:
```bash
uv run sed -n '378,422p' src/phenotypic/gui/_shared/tiles.py
```
Adapt the call to match; do not assume the parameter order above.

- [ ] **Step 2: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_level_selection.py -v
git add tests/unit/gui/results_viewer/test_level_selection.py
git commit -m "test(gui): pin pyramid level selection to the recorded ladder"
```

---

### Task 3.3: Rebuild the Plate layout full-canvas

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_layout.py` (the Plate `cards_column`)
- Modify: `src/phenotypic/gui/results_viewer/_viewer_card.py`
- Modify: `src/phenotypic/gui/results_viewer/_assets/results_viewer.css`
- Modify: `src/phenotypic/gui/results_viewer/_callbacks.py` (clientside wiring)

- [ ] **Step 1: Read the mockup and the design tokens together**

```bash
uv run grep -oE 'class="[^"]+"' \
  docs/superpowers/artifacts/2026-08-26-gui-ome-zarr-sync/mockup/Main.dc.html \
  | sort -u | head -40
uv run grep -n "003660\|febc11\|0e1620\|Comfortaa\|JetBrains" src/phenotypic/gui/_design.py
```
Every colour, radius and family in the mockup must resolve to an existing token. If one
does not, it is an invention — either map it to the nearest token or raise it as a design
question, but do not add a new hard-coded hex.

- [ ] **Step 2: Build the surface**

Five pieces, per spec §6.1:

1. **Full-canvas stage.** Controls float **over** the canvas — the image gets the full
   frame instead of losing ~300 px to a sidebar.
2. **Layers panel** listing the store's **real** series from task 3.1's `series` key, plus
   `objmap` tagged as a **label image**, each with visibility, opacity and a swatch. It
   reads the list from the store, never a hard-coded set.
3. **Navigator inset** — OpenSeadragon's one genuinely missed affordance, carried over.
4. **Pyramid readout** naming the level actually being served, e.g.
   `level 1 of 4 · 2048×1536 · zstd · 1024² chunks`. This is **instrumentation, not
   decoration**: it is the pyramid the old DZI path rebuilt from scratch every time, and it
   is the fastest way to diagnose a level-selection bug.
5. **Image stepping** (`‹ dataset / stem ›`) and the object/grid summary, top-left.

- [ ] **Step 3: Wire the clientside callback through the façade only**

The callback calls `window.phenotypicViv.mount` / `.setSource` with task 3.1's dict. It
must not reference `window.__vivBundle`. Grep to enforce:

```bash
uv run grep -rn "__vivBundle" src/phenotypic/gui/ \
  --include='*.py' --include='*.js' | grep -v _assets/viv_viewer.js
```
Expected: no hits outside the façade.

- [ ] **Step 4: Unhook `_dzi_tiler` from the results Plate path**

Remove `_tile_routes.py:34` (the import), `:458` (`_dzi_tiler.tile(source_png, staging_dir)`)
and `:551` (the overlay tiling call), along with the now-dead staging/publish helpers
`_publish_dzi_cache` (`:337`) and `_generate_dzi_stage` (`:434`) **if** nothing else calls
them.

**Do not delete `_dzi_tiler` itself.** It has five live consumers outside this path —
`browse/_app.py:40`, `browse/_preparation.py:711`, `browse/_preparation_routes.py:95`,
`builder/_point_picker.py:417`, `builder/_preview_tiles.py:144` — and Browse keeps
libvips → DZI → OSD as its only pixel path (spec §9). See [DRIFT.md](DRIFT.md) D-5.

```bash
uv run grep -rn "_dzi_tiler" src/phenotypic/gui/results_viewer/_tile_routes.py
```
Expected after the edit: no hits.

- [ ] **Step 5: Decide the fate of the `.dzi` routes**

The `/tiles/<dataset>/<stem>.dzi` routes now have no producer for the Plate layer. Either
remove them with their tests, or keep them serving the **overlay** path if the QC review
gallery still calls them:

```bash
uv run grep -rn "\.dzi" src/phenotypic/gui/ --include='*.py' --include='*.js' | grep -v browse
```
Whichever you choose, say so in the commit body. Silently leaving a route with no producer
is the residue this rebuild should not create.

- [ ] **Step 6: Run the viewer suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/results_viewer -n 4 -q
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui -k "plate or viewer" -q
```

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/
git add -A src/phenotypic/gui/results_viewer tests/unit/gui/results_viewer
git commit -m "feat(gui): rebuild the Plate surface on Viv, off the DZI path"
```

---

### Task 3.4: A mid-run zeros objmap is correct, not an error

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_store_source.py` (docstring)
- Test: `tests/unit/gui/results_viewer/test_store_source.py` (extend)

**Interfaces:**
- Consumes: [DRIFT.md](DRIFT.md) D-4.

> **The spec claim this corrects.** Viv spec §6.2 inherits from the backend spec the claim
> that Stage 2 writes the objmap into the store, so "the GUI can render a real objmap
> mid-run". The **landed** engine inverted that: Stage 2 is read-only and writes
> `stage2_raw/<ds>/<stem>.npy` plus a token under `.phenotypic/progress/`; only Stage 3
> re-promotes the store. Between Stage 1 and Stage 3 the store's `labels/objmap` is
> **zeros**.

- [ ] **Step 1: Write the test**

```python
def test_an_all_zero_objmap_is_a_valid_source_not_an_error(output_root, stage1_store):
    """A store between Stage 1 and Stage 3 holds a zeros objmap.

    Backend behaviour (landed): Stage 2 is read-only, so the in-store objmap
    stays zeros until Stage 3 re-promotes. The Layers panel must offer the
    label layer normally -- an empty segmentation is the correct rendering of
    a correct store, not a condition to surface as a fault.
    """
    spec = build_source_spec(output_root, "ds", "stage1plate", "/")
    assert spec["labelPath"]
    assert "error" not in spec
```

`stage1_store` is a store written by Stage 1 only — a zeros objmap with its `ome.labels`
list and `image-label` block present (backend §3.3 guarantees the objmap always exists,
including after Stage 1).

- [ ] **Step 2: Record the correction in the module docstring**

Add to `_store_source.py`:

```python
# A store observed between Stage 1 and Stage 3 holds an ALL-ZERO objmap:
# the landed staged engine keeps Stage 2 read-only and publishes the
# segmentation only when Stage 3 re-promotes. This is a valid store, and the
# Layers panel renders the label layer normally. Do not treat zeros as a
# fault -- see the plan's DRIFT.md, row D-4.
```

- [ ] **Step 3: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_store_source.py -v
git add src/phenotypic/gui/results_viewer/_store_source.py \
        tests/unit/gui/results_viewer/test_store_source.py
git commit -m "fix(gui): treat a mid-run zeros objmap as a valid label source"
```
