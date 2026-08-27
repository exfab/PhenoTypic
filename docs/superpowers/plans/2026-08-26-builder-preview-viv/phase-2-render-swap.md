# Phase 2 — Bundle reuse and the render swap

**Spec:** §1, §4. **Depends on:** phase 1, viv-rebuild phases 2-3. **Blocks:** phase 3.

**Deliverable:** the builder app serves the *same* vendored Viv artifact as the results
viewer, its preview pane renders through the shared façade, and `_dzi_tiler` is off the
preview path (`_preview_tiles.py:30, :144`). The point picker is untouched.

---

### Task 2.1: Mount the shared bundle in the builder

**Files:**
- Modify: `src/phenotypic/gui/builder/_app.py`
- Test: `tests/unit/gui/builder/test_shared_viv_asset.py` (create)

**Interfaces:**
- Consumes: `results_viewer/_assets/viv/viv-bundle.min.js` and `_assets/viv_viewer.js` from
  viv-rebuild phase 2.
- Produces: `window.phenotypicViv` available on the builder page.

> **One artifact, two mounts.** Committing a second ~1 MB copy into `builder/assets/` would
> put two artifacts under one build recipe, drifting independently — and nothing would fail
> when they did, because there is no npm in CI to rebuild and compare. Spec §1.

- [ ] **Step 1: Establish which mechanism actually works**

Spec §7 open question 1 records this as unverified. Determine it before writing code:

```bash
uv run grep -n "assets_folder\|assets_url_path\|serve_locally" \
  src/phenotypic/gui/builder/_app.py \
  src/phenotypic/gui/results_viewer/_app.py \
  src/phenotypic/gui/shell/_app.py
```

Two candidates, in preference order:
1. Dash's `assets_folder` / `assets_url_path` on the builder app pointed at the results
   viewer's `_assets/viv/` — no file duplication, no symlink.
2. A small Flask route on the builder serving the two files from the results-viewer package
   directory.

Pick whichever the existing app construction supports without restructuring, and **record
which and why** in the commit body. Either satisfies decision A.

- [ ] **Step 2: Write the test**

```python
"""The builder serves the results viewer's Viv artifact, not a copy of it.

The second assertion is the one that matters: two artifacts under one build
recipe drift independently, and with no npm in CI nothing would fail when they
did.
"""

from pathlib import Path

import phenotypic.gui.builder as builder_pkg
import phenotypic.gui.results_viewer as rv_pkg

RV_BUNDLE = Path(rv_pkg.__file__).parent / "_assets" / "viv" / "viv-bundle.min.js"


def test_builder_page_exposes_the_facade(builder_client):
    resp = builder_client.get("/assets/viv/viv-bundle.min.js")
    assert resp.status_code == 200
    assert len(resp.data) == RV_BUNDLE.stat().st_size


def test_builder_does_not_carry_its_own_bundle_copy():
    stray = list((Path(builder_pkg.__file__).parent / "assets").rglob("viv-bundle*.js"))
    assert not stray, f"builder has its own bundle copy: {stray}"
```

Adjust the URL in the first test to whatever mechanism step 1 chose.

- [ ] **Step 3: Implement, run, commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/builder/test_shared_viv_asset.py -v
git add src/phenotypic/gui/builder/_app.py \
        tests/unit/gui/builder/test_shared_viv_asset.py
git commit -m "feat(gui): serve one shared Viv artifact to the builder"
```

---

### Task 2.2: Swap the preview pane's renderer

**Files:**
- Modify: `src/phenotypic/gui/builder/_preview_tiles.py` — remove `:30` (import) and `:144` (`_dzi_tiler.tile`)
- Modify: `src/phenotypic/gui/builder/_preview_callbacks.py` — clientside wiring
- Modify: `src/phenotypic/gui/builder/assets/preview.js`
- Test: `tests/e2e/gui/test_builder_preview_viv.py` (create)

**Interfaces:**
- Consumes: `preview_zarr_url` from phase 1, `window.phenotypicViv` from task 2.1.
- Produces: a preview pane with no DZI behind it.

- [ ] **Step 1: Write the failing e2e test**

```python
"""The builder preview renders through Viv, reading store chunks directly.

The network assertion is the substance: a preview that renders correctly while
still fetching ``.dzi`` has not been swapped, only decorated.
"""

import pytest


@pytest.mark.e2e
def test_preview_reads_zarr_chunks_not_dzi(page, live_builder_url, seeded_pipeline):
    requests: list[str] = []
    page.on("request", lambda r: requests.append(r.url))

    page.goto(live_builder_url)
    page.wait_for_function("() => window.phenotypicViv !== undefined")
    page.click("[data-testid='preview-node-0']")
    page.wait_for_selector("[data-testid='preview-viv-stage']")

    assert any("/preview-zarr/" in url for url in requests)
    assert not any(url.endswith(".dzi") for url in requests)
```

- [ ] **Step 2: Run and watch it fail**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/e2e/gui/test_builder_preview_viv.py -v
```

- [ ] **Step 3: Build the source spec for a preview node**

The preview's store carries the same `attributes.phenotypic` block a run store does, so
reuse the results viewer's resolver rather than writing a second one:

```bash
uv run grep -n "def build_source_spec" src/phenotypic/gui/results_viewer/_store_source.py
```

It takes an `output_root`; the preview has none. Refactor it to take a **store path and a
base URL** and have the results viewer's caller supply those — one resolver, two callers.
That refactor belongs here rather than in viv-rebuild phase 3, because this is the phase
that discovers the second caller.

**The label-path rule carries over unchanged:** resolve through `phenotypic.labels.objmap`,
never construct `f"{series}/labels/objmap"`. A preview node's store may well be
`gray`-primary, since `save_intermediate_zarr` writes only the requested layers.

- [ ] **Step 4: Rewrite `preview.js`'s render path**

Replace the OpenSeadragon mount with `window.phenotypicViv.mount` / `.setSource`. Talk only
to the façade:

```bash
uv run grep -rn "__vivBundle" src/phenotypic/gui/builder/
```
Expected: no hits.

- [ ] **Step 5: Unhook `_dzi_tiler`**

Remove `_preview_tiles.py:30` (`from phenotypic.gui.results_viewer import _dzi_tiler`) and
`:144` (`_dzi_tiler.tile(png_path, sdir / "dzi")`). Then decide the fate of
`stage_channel_png` (`:74`) and `_channel_to_rgb_uint8` (`:52`): with Viv reading the store
directly, the intermediate PNG has no consumer for the preview pane.

```bash
uv run grep -rn "stage_channel_png\|_channel_to_rgb_uint8\|preview_dzi_url" src/ tests/
```
Remove them if nothing else calls them; keep them if the point picker or a test does. Say
which in the commit body.

**Do not delete `_dzi_tiler`.** After this phase it still has four consumers —
`browse/_app.py:40`, `browse/_preparation.py:711`, `browse/_preparation_routes.py:95`,
`builder/_point_picker.py:417`.

- [ ] **Step 6: Prove the point picker is untouched**

```bash
git diff --stat src/phenotypic/gui/builder/_point_picker.py
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/builder -k "point_picker" -q
```
Expected: **empty diff**, tests PASS. Spec §4 makes this the executable statement that the
picker stays on DZI; a diff here means the phase overreached.

- [ ] **Step 7: Run the builder suites**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/builder -n 4 -q
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui -k builder -q
```

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/gui/builder/ \
                        src/phenotypic/gui/results_viewer/_store_source.py
git add -A src/phenotypic/gui/builder src/phenotypic/gui/results_viewer/_store_source.py \
           tests/unit/gui/builder tests/e2e/gui
git commit -m "feat(gui): render builder node previews through Viv, off the DZI path"
```
