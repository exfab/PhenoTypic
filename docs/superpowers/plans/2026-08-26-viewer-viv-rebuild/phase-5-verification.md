# Phase 5 — Verification, ledgers and docs

**Spec:** §8, and the `gui-checks` obligations inherited from the removals plan.
**Depends on:** phases 0-4.

**Deliverable:** all five of spec §8's checks green, the three CI gates passing, FEATURES.md
and WORKFLOWS.md updated for the rebuilt surfaces, and the bundle-staleness mitigation
wired up.

---

### Task 5.1: Close out spec §8's five checks

**Files:**
- Verify only, except where a check has no test yet.

Spec §8 names five. Four already have homes; confirm each and fill the gap.

- [ ] **Step 1: Walk the checklist**

| Spec §8 check | Where it lives | Action |
|---|---|---|
| Codec ordering — open a **CLI-written** store in a real browser, "not 'the codec registered' — the actual read" | phase 2 task 2.3 | run it |
| Level selection matches `phenotypic.pyramid`'s ladder, `ceil` boundary included | phase 3 task 3.2 | run it — and assert against the **browser's** choice if task 3.2 retired the server-side stack |
| Staleness — a rewritten nested chunk must invalidate **without moving the token** | phase 1 task 1.2 **step 4b** | run it |
| Curation regression — colony curation tests pass **unmodified** | phase 4 task 4.3 step 3 | run it, **and run the three tests that actually prove the chain** (below) |
| Label path — a `gray`-primary store resolves its objmap through `phenotypic.labels.objmap` | phase 3 task 3.1 | run it |

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/shared/test_resolve_within_root.py \
  tests/unit/gui/results_viewer/test_zarr_routes.py \
  tests/unit/gui/results_viewer/test_store_source.py \
  tests/unit/gui/results_viewer/test_level_selection.py \
  tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py \
  -n 4 -v
PLAYWRIGHT=1 QT_QPA_PLATFORM=offscreen xvfb-run -a \
  uv run pytest tests/e2e/gui -k "viv or colony_shared or builder" -v
```

> **`PLAYWRIGHT=1` and `xvfb-run` are both required, and omitting either is silent.**
> Without `PLAYWRIGHT=1` the conftest (`tests/e2e/gui/conftest.py:49`) skips the whole
> module — the command exits 0 having tested **nothing**, and the checklist gets ticked.
> Without `xvfb-run` the rendering tests launch `chromium_headless_shell`, which has no GL
> stack, and fail with `Failed to create WebGL context` — a red that looks like a rendering
> bug and is not one. A verification step that passes by skipping is worse than one that
> fails.
>
> **Measured 2026-08-27:** `15 passed in 81.83s` across
> `test_viv_codec_reads_a_real_store.py` (5), `test_viv_facade_renders.py` (4),
> `test_colony_shared_camera.py` (3), `test_builder_preview_viv.py` (3).

**The curation chain is not proved by `test_colony_callbacks_helpers.py`.** Its 15 tests
drive pure helpers against hand-built `ctx.triggered` dicts — it would pass unmodified while
phase 4's deck.gl rewrite removed the radial entirely. Run the three that do:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/gui/results_viewer/colony_view/test_grid.py::test_build_grid_tiles_carry_radial_trigger_not_old_remove_button \
  tests/integration/gui/test_triage_callbacks.py::test_colony_wedge_mark_writes_category_parquet_and_drops_mirror \
  tests/unit/cli/test_cli_error_outputs.py -v
```

The first is the **only** assertion anywhere that a cell carries `colony-radial-trigger`,
which is exactly what phase 4 endangers. It lives under `tests/gui/`, so no
`tests/unit/gui` invocation reaches it.

Expected: all PASS.

- [ ] **Step 1b: Re-run the colony budget script — it is the only measurement gate left**

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-26-viewer-viv-rebuild/colony_view_budget.py
```
Expected: **exit 0**. A non-zero exit means `RECORDED_CAP` / `RECORDED_FRAME_MS` were never
filled in, and therefore that `COLONY_VIEW_CELL_CAP` in `_grid.py` is an invented number.

With the other two scripts deleted (user ruling, 2026-08-26) this is the **only** surviving
logic-validation script and the only measured number that lands in shipped code — and until
now nothing outside the phase that produces it checked that it was filled in.

**Skip this step if phase 4 was cut.** Phase 4 is optional and marked the first thing to
cut; if it did not land there is no cap to assert, and the script should not be run. Same
shape as the shared-camera FEATURES row in task 5.3 step 1.

- [ ] **Step 2: Confirm the spike gate's findings were actually acted on**

```bash
cat docs/superpowers/plans/2026-08-26-viewer-viv-rebuild/spike/README.md
```
Every question from phase 0 has an answer with evidence; the chunk-size decision carries
its measurement or an explicit "not measured, risk left open". **A gate whose findings were
never revisited is a gate that did not gate anything.**

---

### Task 5.2: Wire the bundle-staleness mitigation

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_app.py` (log the version at startup)
- Test: `tests/unit/gui/results_viewer/test_viv_bundle_version.py` (create)

> **Spec §10, open question 3, is unresolved and stays unresolved.** The build recipe is
> committed and the version logged, but *nothing fails* when the bundle drifts from the
> lockfile. A CI check that rebuilds and compares hashes would need npm in CI, which
> decision A exists to avoid. The version string is a **mitigation, not an answer** — this
> task implements the mitigation and says so plainly rather than dressing it up as a fix.

- [ ] **Step 1: Write the test**

```python
"""``tools/viv-bundle/VERSION`` agrees with the committed artifact.

This does NOT prove the artifact was built from the committed lockfile -- only
a rebuild could, and there is no npm in CI by design (spec section 3). It
catches the common case: bumping one and forgetting the other.
"""

import re
from pathlib import Path

import phenotypic.gui.results_viewer as rv

REPO = Path(rv.__file__).resolve().parents[4]
BUNDLE = Path(rv.__file__).parent / "_assets" / "viv" / "viv-bundle.min.js"
VERSION_FILE = REPO / "tools" / "viv-bundle" / "VERSION"


def test_bundle_embeds_the_recorded_version():
    recorded = VERSION_FILE.read_text(encoding="utf-8").strip()
    assert recorded, "tools/viv-bundle/VERSION is empty"
    head = BUNDLE.read_text(encoding="utf-8", errors="replace")[:4096]
    assert re.search(re.escape(recorded), head), (
        f"bundle does not embed VERSION {recorded!r}; rebuild it via "
        f"tools/viv-bundle/README.md or correct VERSION"
    )
```

Confirm `REPO` resolves correctly on this layout before trusting `parents[4]`:
```bash
uv run python -c "
import pathlib, phenotypic.gui.results_viewer as rv
print(pathlib.Path(rv.__file__).resolve().parents[4])"
```

- [ ] **Step 2: Log the version at startup**

In `_app.py`'s `create_app`, log `viv bundle: <VERSION>` alongside the existing startup
lines. Spec §3 requires the GUI to log it — with no npm in CI, nothing else will tell you
the bundle is stale.

- [ ] **Step 3: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_viv_bundle_version.py -v
git add src/phenotypic/gui/results_viewer/_app.py \
        tests/unit/gui/results_viewer/test_viv_bundle_version.py
git commit -m "chore(viv): log and pin the vendored bundle version"
```

---

### Task 5.3: Ledgers, tutorial and CLAUDE.md

**Files:**
- Modify: `src/phenotypic/gui/FEATURES.md`, `src/phenotypic/gui/WORKFLOWS.md`
- Modify: `scripts/capture_gui_tutorial_screenshots.py`
- Modify: `docs/source/tutorials/gui/06_view_results.md`
- Modify: `src/phenotypic/gui/CLAUDE.md`

- [ ] **Step 1: Update the FEATURES.md rows for the rebuilt surfaces**

The Plate and Colony rows now describe a Viv surface, not an OpenSeadragon one. Update the
capability text and any implementation refs that pointed at `_dzi_tiler` or the `.dzi`
routes. Add rows for the new affordances: the Layers panel, the navigator inset and the
pyramid readout.

**Add the shared-camera lock row only if phase 4 actually landed.** Phase 4 is optional and
marked the first thing to cut; a `✅ shipping` row for it makes `check_features_md.py
--strict` resolve refs for an affordance that does not exist. If phase 4 was cut, either
omit the row or file it as `🔭 planned`.

```bash
uv run grep -n "OpenSeadragon\|DZI\|dzi\|Plate\|Colony" src/phenotypic/gui/FEATURES.md | head -30
```

- [ ] **Step 2: Refresh the results-viewer tutorial and its screenshots**

`06_view_results.md` shows the old card-plus-sidebar Plate. Update the prose and re-capture:

```bash
uv run grep -n "_capture_view_results\|06_view_results" \
  scripts/capture_gui_tutorial_screenshots.py src/phenotypic/gui/WORKFLOWS.md
```

Per the **`gui-tutorial-capture`** skill, the ledger ↔ capture-function ↔ tutorial-page
round trip must stay closed.

- [ ] **Step 3: Update `gui/CLAUDE.md`**

Record: the Plate/Colony pixel path is Viv over `/zarr/...`, and the builder preview is Viv
over `/preview-zarr/...` (phase 6); Browse remains libvips → DZI → `BrowseCache` → OSD;
`_dzi_tiler` survives for **Browse and the point picker** — four consumers, not five, once
phase 6 has landed; `_tile_routes.py` survives as a module even with its `.dzi` routes gone,
because the builder imports `_TILE_NAME_RE` and `_json_error` from it; and the façade at
`_assets/viv_viewer.js` is the only thing that may touch `window.__vivBundle`.

**Run this task last, after phase 6.** Both phases edit the same four files
(`FEATURES.md`, `WORKFLOWS.md`, `gui/CLAUDE.md`, the capture script) through the same three
CI gates; doing it once is the point of folding phase 6 in.

- [ ] **Step 4: Run all three gates**

```bash
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py -v
QT_QPA_PLATFORM=offscreen uv run python scripts/capture_gui_tutorial_screenshots.py --skip-cli
```
Expected: all exit 0.

- [ ] **Step 5: Full suite as a Slurm job**

Per the **`run-phenotypic-test`** skill — the full `tests/unit` suite is a ~65-minute Slurm
job, not a local invocation:

```bash
sbatch docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch
```
Report it as "green except the known baseline failure", and re-confirm it is still *that*
test failing for *that* reason.

- [ ] **Step 6: Commit**

```bash
git add -A src/phenotypic/gui/FEATURES.md src/phenotypic/gui/WORKFLOWS.md \
           src/phenotypic/gui/CLAUDE.md scripts/capture_gui_tutorial_screenshots.py \
           docs/source
git commit -m "docs(gui): record the Viv-backed Plate and Colony surfaces"
```

---

### Task 5.4: File the spec amendments this plan earned

**Files:**
- Modify: `docs/superpowers/specs/2026-08-26-viewer-viv-rebuild/design.md`
- Modify: `docs/superpowers/specs/2026-08-18-ome-zarr-image-store/design.md` (only if the
  chunk-size measurement demanded it)

> Spec changes are the user's call, not the executor's. This task **drafts** them and stops.

- [ ] **Step 1: Draft the §1 correction**

Spec §1 opens by saying the backend is "specification only — there is no zarr code in
`src/`". That is no longer true. Draft a revision pointing at
[DRIFT.md](DRIFT.md) and marking §4.1, §4.2 and §6.2's D3 as landed.

- [ ] **Step 2: Draft the D-4 correction**

Spec §6.2's rationale inherits the backend's claim that Stage 2 writes the objmap in place,
so "the GUI can render a real objmap mid-run". The landed engine keeps Stage 2 read-only.
Draft the correction, and note that the *backend* spec §3.4 needs the same amendment.

- [ ] **Step 3: Draft the §4 route narrowing**

Spec §4 sketches an unrestricted path tail. Phase 1 restricts it to
`_READABLE_ROOTS` because measurements now live inside the store at
`tables/measurements/table.parquet` ([DRIFT.md](DRIFT.md) D-6). Draft the amendment with
that reasoning.

- [ ] **Step 4: Present all three to the user; do not self-approve**

Report the drafts and wait. Amending a spec on the executor's own authority is how a design
record stops being a record.
