# Results Viewer Scatter Tab — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third results-viewer tab that renders faceted, clickable scatter plots from run measurements and exports them as a multi-page PDF.

**Architecture:** A pure `FigureSpec` is built from the shared filtered frame, becomes one Plotly figure with `Scattergl` traces for the screen, and the same figure is re-rendered with `Scatter` traces for kaleido export. Clicking a point carries an `int32` index into `OutputRoot.master_df` (not the filtered frame), resolved server-side into `(dataset, stem, Object_Label)` for the existing crop route and Viv stage. Tasks 1–3 are a prerequisite fix to the shared crop path and ship as their own commits before any Scatter code.

**Tech Stack:** Python 3.12, Dash 4, Plotly 6.6, polars, zarr v3, kaleido 1.2, pypdf (new), skimage 0.25, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-results-scatter-tab/design.md` (revision 3)

**Worktree:** `.worktrees/results-scatter-gui`, branch `feat/results-scatter-gui`, stacked on `ome-zarr-merged`. Run `uv sync --group dev --extra gui` before Task 1 if `.venv` is absent.

## Global Constraints

- **`uv` is the sole runner.** Never bare `python` or `pip`. Tests: `uv run pytest`.
- **Lint with explicit paths only:** `uv run ruff check --fix <paths you changed>`. A bare `ruff check --fix` rewrites the whole repo.
- **Escalate any plan defect that changes the PUBLIC interface — do not decide
  it yourself.** CLAUDE.md defines that boundary: *only `__init__.py` exports
  are public*. Everything under a `_`-prefixed module or package is internal,
  and correcting it is expected and encouraged — just say what you changed.

  Escalate when a fix would change:
  - a name in any `__init__.py` `__all__` — for this branch that is
    `gui/__init__.py` and `results_viewer/__init__.py` (`create_app`,
    `launch_results_viewer`, `OutputRoot`, …);
  - anything in `phenotypic.schema` — the public measurement schema;
  - a **URL route or query parameter**, which users and browsers call directly
    and no `_` prefix protects;
  - a **CLI flag** or its accepted values;
  - what an existing surface **renders or offers to a user** — a tab, a
    dropdown's option list, a rendered image — even with no signature change;
  - the **dependency set** (`pyproject.toml`).

  Do NOT escalate for internals: `_scatter_tab/*`, `_shared/tiles.py`,
  `colony_view/_grid.py` and every other `_`-prefixed module. Changing a
  parameter there from optional to required, adding a helper, or renaming a
  private function is ordinary work — make the change, say so, and I will
  reconcile the plan.

  **Correction, since I set this bar wrongly a moment ago.** I cited
  `_store_layer_array_to_rgb`'s required `store_path` and
  `register_crop_route`'s `default_contours` as changes that should have been
  escalated. Under the project's own definition neither is public: the first
  is `_`-prefixed, and the second lives in `gui/_shared/`, which no
  `__init__.py` exports. Both were correctly settled in-flight. What *would*
  have qualified from Task 3 is the `?contours=` **query parameter**, because a
  URL is a public interface regardless of the module it is defined in.
- **A number is only valid where it was measured. Carrying one across a
  boundary is this branch's single most common defect — five instances, none
  catchable by reading.** In every case the value was correct in its original
  context and wrong in the one it was pasted into, and in every case only
  running something exposed it:
  | pasted value | correct where measured | wrong because |
  |---|---|---|
  | `100`, then `72`, page multiplier | 1600x1200 px of PNG raster; a PDF point is 1/72 in | kaleido reads plotly `width` as px at **96** DPI; the px→pt conversion is its own |
  | `TIMELINE_COMPARE_CAP`'s WebGL rationale | 12 independent OpenSeadragon viewers, 12 contexts | Plotly pools every gl trace into **one** context |
  | `TextureGray_` in `_MEASUREMENT_PREFIXES` | — | no schema declares that category |
  | `rgb/4` display range | that pyramid level | quoted as level 0's |
  | ink threshold `2000` dark px | 5,000 **opaque** markers | applied to 60 at opacity 0.5, luminance ~149, which score **zero** below 128 |
  Before reusing any measured number, ask what was being measured and whether
  this context still matches it. Where possible prefer a **differential**
  against a control over an absolute threshold — it cannot drift when the
  fixture changes, which is the coupling that produced two of the five.
- **A mutation harness must verify its PRECONDITION, not only its
  postcondition — and the orchestrator must snapshot independently.** Two
  harnesses on this branch left a dirty tree. Cluster D's was killed by a
  foreground timeout between applying a mutation and restoring it, so its own
  check never ran; only the orchestrator's separate sha256 set caught it.
  Cluster F's keyed backups on `basename`, and two of its four targets were
  both `_layout.py` — so the backup overwrote one with the other and the
  restore copied the wrong module into a source path, replacing it wholesale.
  Its `sha256sum -c` did fire, but only *after* the evidence was gone.
  The generalisation, in cluster F's words: restore and verification shared not
  a mechanism but a **premise** — both trusted that the backup held four
  distinct files. Verifying a postcondition cannot save you when the flaw is in
  a precondition you never stated. So: mirror full relative paths (in a package
  tree `_layout.py`, `_ids.py` and `__init__.py` repeat by construction, and
  basename is never a unique key over source paths); assert the backup *count*
  equals the target count, which a collision fails even when every surviving
  comparison passes; verify before touching anything and exit non-zero having
  changed nothing; and restore-and-verify after **each** mutation rather than
  letting a corrupted file propagate into the next.
  Run long harnesses in a background slot — a foreground timeout is a SIGTERM
  mid-mutation — and never edit a file a harness has open: an `atexit` restore
  reverts your edit, and a digest check cannot distinguish that from a clean
  restore.
- **Two guards for one property means the property is untested.** Found by
  cluster E while *writing* a mutation, not running it: the phantom check had
  both an explicit `label is None` and a `try/except` around `int(label)`.
  Removing either left the other, so no mutation could kill the test and
  neither guard was pinned — while the redundancy read to a later maintainer
  as evidence the case was covered. Collapse to the single guard a mutation
  can kill. More defensive is not more tested; it is often less.
- **A test that touches the real mirror needs a filter that genuinely drops
  rows.** The index-equals-row-position coincidence is live on the fixture, so
  a round-trip test over an unfiltered frame passes against the very defect it
  names. Assert the degeneracy is absent *before* the assertion that depends
  on it (`assert carried != list(range(height))`), so a fixture that drifts
  back into the coincidence fails as a broken test rather than passing
  vacuously.
- **The click fingerprint must be the one captured WHEN THE FIGURE WAS DRAWN,
  never re-read at click time.** `resolve_click(master_df, index, fingerprint,
  expected_fingerprint)` compares a value stored with the figure against the
  binding's current value. Reading `OutputRoot.snapshot.consumed_state_
  fingerprint` on *both* sides makes the comparison a tautology that always
  passes — the guard still exists, still looks right, and stops nothing. The
  stored side comes from the Dash store the figure callback wrote; only the
  expected side is read live. This cannot be caught inside `_inspector.py`,
  because the defect lives at the call site, and it is the same shape as B1.
- **`index_frame` is idempotent — a re-call is correct, not a 500.** It
  returns an already-indexed frame unchanged, because the index it carries is
  already master-anchored. Do NOT add a "has this been indexed?" check at a
  call site, and do NOT catch `DuplicateError` and re-stamp: re-stamping a
  filtered frame *is* the B1 defect, manufactured by the guard meant to
  prevent a crash.
- **Never spell a measurement header — ask the schema for it.** This branch has
  produced the same defect three times: `TextureGray_AvgContrast` pinned in the
  existing suite, a `TextureGray` category in `_MEASUREMENT_PREFIXES` that no
  schema declares, and `ColorXYZ_X` invented for a Gate 0 test. Each time a
  plausible header was written from the *shape* of a category name, and each
  time it read correctly and tested nothing — the column matched no measurer,
  so the assertion passed or failed for a reason unrelated to its subject.
  Reading cannot catch it, because the string looks right.
  In a test, write `ColorXYZ.get_headers()[0]`, not `"ColorXYZ_X"`. Real
  examples for reference: `Texture_Contrast-deg000-scale05`,
  `ColorXYZ_CieXMean`, `Intensity_MeanIntensity`, `Shape_Area`. If a header
  must be literal, verify it against `get_headers()` first.
- **Ask the schema for the spelling, then assert the column is in the frame.**
  A header the schema defines may belong to a measurer this run never
  configured. `SIZE.AREA` is a real member and `Size_Area` is CLAUDE.md's own
  worked example, yet the verification fixture has no `Size_*` column at all --
  its measurers are `MeasureNeighborDist`/`MeasureShape`/`MeasureIntensity`/
  `MeasureColor`/`MeasureTexture`. Spellability and presence are different
  questions and only one of them is a property of the run. (Found by cluster H;
  wording is theirs.)
- **A count is a summary of evidence, not evidence.** Three defects on this
  branch came from reasoning about a number -- "19 checks, 1 failed", "30 rows",
  "23 mutations" -- instead of reading what it summarized. When a run's summary
  and your expectation disagree, read the failure list. When the disk and your
  model of a file disagree, `grep` costs one command and settles it. This is the
  same failure as "a number is only valid where it was measured", arriving from
  the direction of the summary rather than the boundary.
- **Prefer the operation that cannot do the wrong thing.** `rmdir` refuses a
  non-empty directory; `rm -rf` with a guard around it relies on the guard.
  Applies equally to a differential test whose noise floor is measured rather
  than assumed, and to a required argument that makes a scope error unwriteable
  rather than a comment that asks the reader not to make it.
- **Never `git add -A`, and never stage a directory.** Stage exactly the paths
  your task's **Files:** block names. Clusters share one worktree, so a
  directory-wide add sweeps another agent's half-finished work into your
  commit — and the resulting commit passes its own tests while containing
  changes nobody described. Found by cluster A during Task 3, when
  `colony_view/_grid.py` was dirty from cluster B's concurrent Task 4.
- **A task's commit boundary is notional when tasks share files.** Tasks 1-3
  all edit `tiles.py` and `test_tiles_zarr.py`; once written, no sequence of
  `git add` can separate them, and `git add -p` is unavailable (interactive
  git flags are unsupported here). Land them as one accurate commit rather
  than a split whose messages misdescribe their own contents. Splitting is
  only honest if each commit owns distinct files.
- **Any pytest run wider than a single file goes through Slurm, never inline.**
  Use `docs/superpowers/plans/2026-09-01-results-scatter-tab/run_scatter_tests.sbatch`,
  which shards the tree across a 16-task array:
  ```bash
  sbatch --export=ALL,SCOPE=gui  docs/superpowers/plans/2026-09-01-results-scatter-tab/run_scatter_tests.sbatch
  sbatch --export=ALL,SCOPE=full docs/superpowers/plans/2026-09-01-results-scatter-tab/run_scatter_tests.sbatch
  ```
  Targeted runs (`-k`, one file) stay inline — they finish in seconds. A
  module-wide run does not, and a head node kills it. Three traps the script
  already handles and an inline run does not: `-n auto` reads the node's core
  count rather than the allocation's and manufactures timeouts; the repo's
  `--capture=no` triples runtime against shared storage; and a missing
  `QT_QPA_PLATFORM=offscreen` aborts the interpreter with no summary.
  **Submission is not execution** — always report `StartTime` and `Reason` from
  `scontrol show job <id>`, never "submitted" as if it were "passed".
- **Verification fixture:** `/rhome/anguy344/bigdata_exfab/projects/ucr_029_e_d_Maresca/data/results/2026-08-11-migration-test/` — 36 migrated stores, 844 mirror rows, 723 plottable, 121 phantoms, **22 strains** (23 distinct `Metadata_Strain` values only if the 82 nulls are counted as one -- the null-drop rule below means the pager shows 22), 28 plates, 36 images. Never write to it.
- **Never use `startswith("Metadata_")`** as a semantic metadata check. Use `is_metadata_header()` (`sdk_/_metadata_helpers.py:281`).
- **Never hard-code a store's series or label path.** Resolve from `attributes.phenotypic` (`labels` → `{"objmap": "rgb/labels/objmap"}`, `series`, `pyramid`).
- **Resolve deliverables paths via `phenotypic.sdk_` helpers**, never hand-joined names. The pipeline config is `pipeline.json.pht-pipe`, reached through `layout.pipeline_config_path`.
- **Google-style docstrings** on every new public function.
- **Feed the mirror, not the master:** the tab reads `OutputRoot.master_df`, which is `deliverables/measurements.parquet`.
- **GUI chrome changes require ledger updates** — `FEATURES.md`, `WORKFLOWS.md` and a tutorial capture. See the `gui-tutorial-capture` skill. Task 14 owns this.

---

## File Structure

**Prerequisite (Tasks 1–3), modifying the shared crop path:**
- `src/phenotypic/gui/_shared/tiles.py` — add `image_display_range`, `scale_to_uint8`, `composite_contours`; fix `_store_layer_array_to_rgb`; thread `contours` through `crop_store_rgb`, `crop_colony`, `register_crop_route`.
- `src/phenotypic/gui/_config.py` — `SCATTER_FACET_CAP`, `SECTION_GROUP_CAP`, `SCATTER_CROPS_URL_SEGMENT`.
- `tests/unit/gui/shared/test_tiles_zarr.py` — extend; it already owns the crop-path store fixture.

**Shared-helper fix (Task 4):**
- `src/phenotypic/gui/results_viewer/colony_view/_grid.py` — derive `_MEASUREMENT_PREFIXES` from schema categories.

**Scatter tab (Tasks 5–13):**
- `src/phenotypic/gui/results_viewer/_scatter_tab/_ids.py` — element ids.
- `.../_spec.py` — `FigureSpec` dataclass, the pure config object.
- `.../_facets.py` — facet planning, ordering, caps, phantom predicate.
- `.../_grouping.py` — column → measurer-name grouping.
- `.../_figure.py` — pure figure builder, no Dash imports.
- `.../_pdf.py` — kaleido per page → pypdf merge.
- `.../_layout.py` — tab body, config popover, floating legend.
- `.../_inspector.py` — offcanvas, click resolution.
- `.../_callbacks.py` — Dash wiring.
- `.../__init__.py` — public factory + registrar.
- `src/phenotypic/gui/results_viewer/_ids.py` — add `TAB_SCATTER_ID`.
- `src/phenotypic/gui/results_viewer/_layout.py` — mount the tab.
- `src/phenotypic/gui/results_viewer/_app.py` — register callbacks and the crop route.
- `src/phenotypic/gui/results_viewer/_assets/results_viewer.js` — generalize the splitter.
- Tests under `tests/unit/gui/results_viewer/`.

---

## Task 1: Scale uint16 store crops instead of truncating them

Fixes the bug where every colony crop from a migrated store renders as mod-256 noise. Spec §2.1–2.3.

**Files:**
- Modify: `src/phenotypic/gui/_shared/tiles.py` — `_store_layer_array_to_rgb` at line 459; add two helpers above it.
- Test: `tests/unit/gui/shared/test_tiles_zarr.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `image_display_range(store_path: Path, layer: LayerName) -> tuple[int, int]`
  - `scale_to_uint8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/gui/shared/test_tiles_zarr.py`:

```python
def test_a_uint16_ramp_renders_monotonically(tmp_path: Path) -> None:
    """The regression pin for the mod-256 truncation.

    ``arr.astype(np.uint8)`` is a modular reduction: 18175 -> 255 and
    18176 -> 0, so a monotonic source becomes a sawtooth. Any correct
    scaling is monotonic non-decreasing. Measured on a real store, the
    truncated path produced 75 descending steps where scaling produces 0.
    """
    from phenotypic.gui._shared.tiles import scale_to_uint8

    ramp = np.arange(19061, 38171, dtype=np.uint16)
    out = scale_to_uint8(ramp, 20511, 44047).astype(np.int16)

    assert (np.diff(out) >= 0).all(), "scaling must never descend"
    assert out.max() > out.min(), "the ramp must not collapse to one value"


def test_values_above_the_range_clip_rather_than_wrap() -> None:
    """Clipping is what makes a per-image range safe for a crop window."""
    from phenotypic.gui._shared.tiles import scale_to_uint8

    over = np.array([44047 + 5000], dtype=np.uint16)
    under = np.array([20511 - 5000], dtype=np.uint16)

    assert int(scale_to_uint8(over, 20511, 44047)[0]) == 255
    assert int(scale_to_uint8(under, 20511, 44047)[0]) == 0


def test_uint8_stores_are_passed_through_unchanged() -> None:
    """An 8-bit store must not be contrast-stretched by the new path."""
    from phenotypic.gui._shared.tiles import scale_to_uint8

    arr = np.array([0, 7, 128, 255], dtype=np.uint8)
    assert np.array_equal(scale_to_uint8(arr, 0, 255), arr)


def test_the_display_range_works_for_every_layer_not_just_rgb() -> None:
    """``objmap``'s member is ``rgb/labels/objmap``, not ``objmap``.

    Passing a member path where the reader wants a layer name appears to
    work for ``rgb`` -- the series map is the identity -- and raises
    KeyError for ``objmap`` while returning a silent ``(0, 0)`` for
    ``detect_mat`` and ``gray``. That asymmetry is why this asserts on
    every layer the store carries.
    """
    from phenotypic.gui._shared.tiles import image_display_range

    if not FIXTURE_STORE.exists():
        pytest.skip("migration-test fixture absent")

    for layer in ("rgb", "objmap", "detect_mat", "gray"):
        lo, hi = image_display_range(FIXTURE_STORE, layer)
        assert hi > lo, f"{layer} produced a degenerate range ({lo}, {hi})"


def test_a_brighter_window_renders_brighter_than_a_dim_one() -> None:
    """The pin against a PER-CROP stretch, which is the naive fix.

    Under a per-crop min-max stretch every window is expanded to fill
    0-255, so a bright region and a dim region render with the SAME mean
    and the ordering is destroyed. Under one per-image range the ordering
    survives. ``f(x) == f(x)`` would prove nothing; this asserts a property
    only the correct implementation has.
    """
    import os

    from phenotypic import Image
    from phenotypic.gui._shared.tiles import crop_store_rgb

    # FIXTURE_STORE, not the synthetic `store`: that one is uint8 (max 250),
    # so scale_to_uint8 returns early and this test would exercise nothing.
    # It passed before the implementation existed -- that is the proof.
    if not FIXTURE_STORE.exists():
        pytest.skip("migration-test fixture absent")
    store = FIXTURE_STORE

    full = Image.load_layer_zarr(store, "rgb", level=0)
    # axis=-1: load_layer_zarr returns rgb CHANNEL-LAST (H, W, 3), pinned by
    # test_rgb_crop_returns_channel_last_pixels. axis=0 would average rows.
    plane = full.mean(axis=-1) if full.ndim == 3 else full
    row_means = plane.mean(axis=1)
    brightest, dimmest = int(np.argmax(row_means)), int(np.argmin(row_means))
    centre_col = float(plane.shape[1] // 2)

    mtime = os.stat(store).st_mtime_ns
    bright = _decode_rgb(
        crop_store_rgb(store, "rgb", float(brightest), centre_col, 64, mtime)
    ).mean()
    dim = _decode_rgb(
        crop_store_rgb(store, "rgb", float(dimmest), centre_col, 64, mtime)
    ).mean()

    assert bright > dim, (
        f"a brighter region rendered no brighter ({bright:.1f} vs {dim:.1f}) "
        "-- the range is being taken from the crop, not the image"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/unit/gui/shared/test_tiles_zarr.py -k "monotonic or clip or passed_through or crop_window" -v
```

Expected: FAIL — `ImportError: cannot import name 'scale_to_uint8'`.

- [ ] **Step 3: Implement the two helpers**

In `src/phenotypic/gui/_shared/tiles.py`, above `_store_layer_array_to_rgb`:

```python
def scale_to_uint8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale an integer array to uint8 against a fixed range, then clip.

    A narrowing ``astype`` is a modular reduction (``value & 0xFF``) and is
    therefore non-monotonic: a +1 step across a 256 boundary becomes -255.
    This scales instead, so the mapping is monotonic non-decreasing, and
    clips rather than wrapping when a value falls outside ``[lo, hi]``.

    Args:
        arr: Source array. ``uint8`` input is returned unchanged, so an
            8-bit store is never contrast-stretched.
        lo: Value mapped to 0.
        hi: Value mapped to 255. Must exceed ``lo``.

    Returns:
        A ``uint8`` array of the same shape.
    """
    if arr.dtype == np.uint8:
        return arr
    span = float(hi) - float(lo)
    if span <= 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr.astype(np.float32) - float(lo)) / span * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def image_display_range(store_path: Path, layer: LayerName) -> tuple[int, int]:
    """Return the ``(lo, hi)`` display range for one image's layer.

    Read from the SMALLEST pyramid level, whole. Deliberately **not
    cached**: the computation costs ~4 ms against the ~165 ms level-0 crop
    read the same request already performs, and no available key
    (``st_mtime_ns`` on the store directory, or ``store_generation_token``)
    moves when a nested chunk is rewritten in place -- so a cache here would
    invalidate on a full re-publish but not on a chunk rewrite, and serve a
    silently wrong brightness.

    Uses min/max, not a percentile: on a real store a 0.5/99.5 percentile
    gives 21,644-25,993 against a true range of 17,912-45,344, clipping the
    colonies -- the brightest thing in the frame and the subject of the
    picture.

    Args:
        store_path: Path to a ``*.ome.zarr`` store directory.
        layer: The layer whose range is wanted (normally ``"rgb"``).

    Returns:
        ``(lo, hi)`` as ints. Falls back to ``(0, 255)`` when the layer has
        no pyramid levels to read, and when the level is degenerate
        (``lo == hi``), so a constant layer cannot produce a zero-width
        range that divides to infinity.
    """
    # Two things this must NOT do.
    #
    # 1. Pass a member path where a layer is wanted. _read_store_level takes
    #    a LAYER and resolves the member itself; a member path "works" for
    #    rgb only because the series map is the identity, raises KeyError
    #    for objmap (member rgb/labels/objmap), and returns (0, 0) silently
    #    for detect_mat and gray.
    # 2. Scan the directory for level names. gui/CLAUDE.md:58 -- "Never
    #    recompute the pyramid. The level count is recorded in
    #    phenotypic.pyramid." An iterdir()/isdigit() scan is that
    #    recomputation. Depth is uniform across layers (verified on the
    #    fixture: rgb, gray, detect_mat and rgb/labels/objmap each hold
    #    levels 0..4 against the block's "levels": 5).
    block = _readable_block(store_path)
    levels = int(block[PhenotypicAttr.PYRAMID]["levels"])
    if levels < 1:
        return (0, 255)
    smallest = _read_store_level(store_path, layer, levels - 1)
    finite = smallest[np.isfinite(smallest)] if smallest.dtype.kind == "f" else smallest
    if finite.size == 0:
        return (0, 255)
    lo, hi = int(finite.min()), int(finite.max())
    # (0, 255) on a degenerate level, never (0, 0): scale_to_uint8's
    # span <= 0 branch returns np.zeros, so a zero-width range renders an
    # entirely black crop with no error anywhere.
    return (lo, hi) if hi > lo else (0, 255)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/unit/gui/shared/test_tiles_zarr.py -k "monotonic or clip or passed_through or crop_window" -v
```

Expected: 4 passed. If `image_display_range` raises, check `_read_store_level`'s actual signature at `tiles.py:407` and adapt the call — that helper's parameter names are the one thing this task must read rather than assume.

- [ ] **Step 5: Wire the fix into the render path**

Replace the `rgb` branch of `_store_layer_array_to_rgb` (line 466). The function needs the store path, so change its signature and its **one**
call site (`tiles.py:571` — the plan originally said two; there is one):

```python
def _store_layer_array_to_rgb(
    arr: np.ndarray, layer: str, store_path: Path
) -> np.ndarray:
    # store_path is REQUIRED, not defaulted. A None default falls back to a
    # per-window stretch -- exactly the "brightness ordering destroyed
    # between crops" bug this fix exists to prevent. One call site, so
    # requiring it costs nothing and closes the trap.
    """Convert a decoded store layer array to an RGB uint8 array."""
    from phenotypic.gui.builder._image_renderer import (
        _label_map_to_rgb,
        _normalize_to_uint8,
    )

    if layer == "rgb":
        # Check the dtype BEFORE asking for a range. scale_to_uint8 returns
        # uint8 arrays untouched, so computing a range for one is waste --
        # and on a store small enough to have a single pyramid level, "the
        # smallest level" IS the full layer, which trips
        # test_store_crop_reads_window_without_full_layer_loader. Every
        # store written before the migration is uint8, so this is also the
        # common path.
        if arr.dtype == np.uint8:
            return arr
        lo, hi = image_display_range(store_path, "rgb")
        return scale_to_uint8(arr, lo, hi)
    if layer == "objmap":
        return _label_map_to_rgb(arr)
    gray = _normalize_to_uint8(arr)
    return np.stack([gray] * 3, axis=-1)
```

- [ ] **Step 6: Run the whole crop-path suite**

```bash
uv run pytest tests/unit/gui/shared/test_tiles_zarr.py tests/gui/_shared/test_tiles.py -v
```

Expected: all pass. `test_crop_matches_the_full_resolution_slice` compares against `_normalize_to_uint8`; if it now fails on the `rgb` layer, update that test to compare against `scale_to_uint8` with the same range — the assertion's intent is "the crop equals the full-res slice", not "it equals `_normalize_to_uint8`".

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/gui/_shared/tiles.py tests/unit/gui/shared/test_tiles_zarr.py
uv run mypy src/phenotypic/gui/_shared/tiles.py
git add src/phenotypic/gui/_shared/tiles.py tests/unit/gui/shared/test_tiles_zarr.py
git commit -m "fix(gui): scale uint16 store crops instead of truncating them

_store_layer_array_to_rgb cast uint16 to uint8 with a bare astype, which
is a modular reduction, so every colony crop from a migrated store
rendered as mod-256 noise. Measured on object 24 of d000466_280_003:
mean horizontal neighbour delta 85.3 with 36.8% of adjacent pixels
differing by more than 100, where smooth imagery reads 0-5.

Scales against a per-image range read from the smallest pyramid level.
Not cached: 4 ms against a 165 ms crop read, and no available key sees an
in-place chunk rewrite."
```

---

## Task 2: Verify the fix against a real migrated store

Task 1's tests use a synthetic store. This proves the fix on the fixture, and is the task that would have caught the bug.

**Files:**
- Test: `tests/unit/gui/shared/test_tiles_zarr.py`

**Interfaces:**
- Consumes: `scale_to_uint8`, `image_display_range` from Task 1.
- Produces: nothing.

- [ ] **Step 1: Write the failing test**

```python
FIXTURE_STORE = Path(
    "/rhome/anguy344/bigdata_exfab/projects/ucr_029_e_d_Maresca/data/results"
    "/2026-08-11-migration-test/results/7-24-26_redo_full/zarr"
    "/d000466_280_003_2026-07-26_06-34-47.ome.zarr"
)


@pytest.mark.skipif(not FIXTURE_STORE.exists(), reason="migration-test fixture absent")
def test_a_real_colony_crop_is_smooth_not_noise() -> None:
    """The end-to-end pin: a crop of a real colony must read as an image.

    Object 24 is the largest in this image (9,182 px) at (1783.2, 342.7).
    Truncation gave a mean horizontal neighbour delta of 85.3; smooth
    imagery reads 0-5. The threshold is 20 -- far from both, so this is
    not a delicate test.
    """
    import os

    png = crop_store_rgb(
        FIXTURE_STORE, "rgb", 1783.158135, 342.748203, 256,
        os.stat(FIXTURE_STORE).st_mtime_ns,
    )
    a = _decode_rgb(png).astype(np.int16)
    delta = np.abs(np.diff(a[:, :, 0], axis=1))

    assert delta.mean() < 20.0, f"crop reads as noise: mean delta {delta.mean():.1f}"
    assert (delta > 100).mean() < 0.01
```

- [ ] **Step 2: Run it**

```bash
uv run pytest tests/unit/gui/shared/test_tiles_zarr.py -k real_colony_crop -v
```

Expected: PASS (Task 1 already fixed it). To confirm the test has teeth, temporarily restore `return arr.astype(np.uint8)` in the `rgb` branch, re-run, see it fail with a mean delta near 85, then restore the fix.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/gui/shared/test_tiles_zarr.py
git commit -m "test(gui): pin the crop fix against a real migrated store"
```

---

## Task 3: Composite objmap contours into the crop server-side

Spec §2.4. Without this, "overlay crop by default" has nothing to show: `layer=rgb` is bare pixels and `layer=objmap` is a label map with no plate under it.

**Files:**
- Modify: `src/phenotypic/gui/_shared/tiles.py` — add `composite_contours`; add a `contours` keyword to `crop_store_rgb`, `crop_colony`, and the route in `register_crop_route`.
- Modify: `src/phenotypic/gui/_config.py` — add `SCATTER_CROPS_URL_SEGMENT`.
- Test: `tests/unit/gui/shared/test_tiles_zarr.py`

**Interfaces:**
- Consumes: `image_display_range`, `scale_to_uint8` (Task 1).
- Produces:
  - `composite_contours(rgb: np.ndarray, labels: np.ndarray, focal: int) -> np.ndarray`
  - `crop_store_rgb(..., contours: int | None = None)` — `contours` is the focal `Object_Label`, or `None` for no compositing.
  - `SCATTER_CROPS_URL_SEGMENT: str = "scatter-crops"`

- [ ] **Step 1: Write the failing test**

```python
def test_contours_draw_a_boundary_around_the_focal_label() -> None:
    """Boundaries are drawn for the focal label and dimmed for neighbours."""
    from phenotypic.gui._design import OI_ORANGE
    from phenotypic.gui._shared.tiles import composite_contours

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    labels = np.zeros((32, 32), dtype=np.uint16)
    labels[8:16, 8:16] = 7      # focal
    labels[20:28, 20:28] = 9    # neighbour

    out = composite_contours(rgb, labels, focal=7)

    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert (out != 0).any(), "no contour was drawn"
    assert not np.array_equal(out, rgb)


def test_contours_are_a_no_op_when_no_label_is_present() -> None:
    from phenotypic.gui._shared.tiles import composite_contours

    rgb = np.full((16, 16, 3), 40, dtype=np.uint8)
    labels = np.zeros((16, 16), dtype=np.uint16)
    assert np.array_equal(composite_contours(rgb, labels, focal=3), rgb)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/gui/shared/test_tiles_zarr.py -k contours -v
```

Expected: FAIL — `cannot import name 'composite_contours'`.

- [ ] **Step 3: Implement `composite_contours`**

```python
def composite_contours(
    rgb: np.ndarray, labels: np.ndarray, focal: int
) -> np.ndarray:
    """Draw object boundaries from a label map onto an RGB crop.

    The focal object is tinted distinctly from its neighbours so a crowded
    crop still says which colony the point refers to. Cheap: a store's
    ``objmap`` level costs effectively nothing on disk, and the boundary of
    a typical colony is ~0.5% of the crop's pixels.

    Args:
        rgb: ``(H, W, 3)`` uint8 image to draw on. Not mutated.
        labels: ``(H, W)`` integer label map, same shape as ``rgb``.
        focal: The ``Object_Label`` to emphasise.

    Returns:
        A new ``(H, W, 3)`` uint8 array with boundaries drawn.
    """
    from skimage.segmentation import find_boundaries

    from phenotypic.gui._design import OI_ORANGE, OI_SKY

    def _rgb(hex_: str) -> tuple[int, int, int]:
        h = hex_.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    out = rgb.copy()
    if labels.shape != rgb.shape[:2]:
        return out

    others = (labels > 0) & (labels != focal)
    if others.any():
        out[find_boundaries(others, mode="outer")] = _rgb(OI_SKY)
    focal_mask = labels == focal
    if focal_mask.any():
        out[find_boundaries(focal_mask, mode="outer")] = _rgb(OI_ORANGE)
    return out
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/gui/shared/test_tiles_zarr.py -k contours -v
```

Expected: 2 passed.

- [ ] **Step 5: Thread `contours` through the crop chain**

Add `contours: int | None = None` as a keyword-only parameter to `crop_store_rgb` and `crop_colony`, defaulting to `None`. Inside `crop_store_rgb`, after the rgb window is decoded and scaled, when `contours is not None` and `layer == "rgb"`: resolve the objmap member path from `attributes.phenotypic.labels["objmap"]` (never hard-coded), read the same window from it, and pass both through `composite_contours`.

In `register_crop_route`'s handler, read the flag from the query string and pass the row's own label as the focal id:

```python
        contours_on = request.args.get("contours", type=int, default=default_contours)
        ...
        png_bytes = crop_colony(
            output_root, dataset, stem, layer,
            center_rr, center_cc, size,
            dim_alpha=dim, bbox=bbox,
            contours=label_int if contours_on else None,
        )
```

Give `register_crop_route` a `default_contours: int = 0` parameter. Scatter passes `1`; the two existing segments keep `0`, so the Colony grid and QC gallery are visually unchanged and P0 owes no ledger update.

- [ ] **Step 6: Add the URL segment constant**

In `src/phenotypic/gui/_config.py`, beside `COLONY_CROPS_URL_SEGMENT`:

```python
#: URL segment for the Scatter inspector's crop route. Distinct from the
#: colony and QC segments so the three can differ in their contour default
#: without a shared flag.
SCATTER_CROPS_URL_SEGMENT: str = "scatter-crops"
```

- [ ] **Step 7: Run the full crop suite and commit**

```bash
sbatch --export=ALL,SCOPE=gui docs/superpowers/plans/2026-09-01-results-scatter-tab/run_scatter_tests.sbatch
uv run ruff check --fix src/phenotypic/gui/_shared/tiles.py src/phenotypic/gui/_config.py tests/unit/gui/shared/test_tiles_zarr.py
git add <exactly the files this task's **Files:** block names>   # never -A
git commit -m "feat(gui): composite objmap contours into store crops

The store crop path served no contours at all: layer=rgb is bare pixels
and layer=objmap is a label map with no plate under it. Contours only
existed in the baked-overlay fallback, which is dead whenever a store
exists. Adds an opt-in ?contours= flag, default off for the colony and QC
segments so their appearance is unchanged."
```

---

## Task 4: Derive `_MEASUREMENT_PREFIXES` from the schema

Spec §16.3. The tuple lists `TextureGray`, which is not a real category, and omits 31 that are.

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/colony_view/_grid.py:93-100`
- Test: `tests/unit/gui/results_viewer/test_measurement_prefixes.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `_MEASUREMENT_PREFIXES: tuple[str, ...]` — same name, now derived.

- [ ] **Step 1: Write the failing test**

```python
"""The measurement-prefix exclusion list must come from the schema."""
from __future__ import annotations

from phenotypic.gui.results_viewer.colony_view._grid import _MEASUREMENT_PREFIXES


def test_texture_is_excluded_and_texturegray_is_not_invented() -> None:
    """``TEXTURE.category()`` is ``Texture``; ``TextureGray`` is nothing."""
    assert "Texture_" in _MEASUREMENT_PREFIXES
    assert "TextureGray_" not in _MEASUREMENT_PREFIXES


def test_continuous_measurement_families_are_excluded() -> None:
    for prefix in ("Shape_", "Intensity_", "Size_", "Bbox_", "ColorLab_"):
        assert prefix in _MEASUREMENT_PREFIXES, prefix


def test_grouping_families_stay_selectable_as_axes() -> None:
    """Metadata, Grid, Object and Curation are what an axis IS."""
    for prefix in ("Metadata_", "Grid_", "Object_", "Curation_"):
        assert prefix not in _MEASUREMENT_PREFIXES, prefix
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/gui/results_viewer/test_measurement_prefixes.py -v
```

Expected: FAIL on `Texture_` (the tuple has `TextureGray_`).

- [ ] **Step 3: Replace the hard-coded tuple**

```python
#: Families excluded from axis pickers: continuous per-object measurements.
#: DERIVED from schema ownership rather than hand-maintained -- the previous
#: literal listed ``TextureGray_``, which no schema declares (so ``Texture_``
#: was never excluded), and omitted 26 real categories including ``Size_`` (31 is the
#: size of the derived set; 5 of the old 6 entries survive it).
#: The grouping families below stay selectable because they are what an axis
#: IS, not what it measures.
_AXIS_ELIGIBLE_CATEGORIES: frozenset[str] = frozenset(
    {"Metadata", "Grid", "Object", "Curation", "Status"}
)


def _derive_measurement_prefixes() -> tuple[str, ...]:
    """Every MeasurementInfo category except the grouping families."""
    from phenotypic.schema import MeasurementInfo

    def leaves(cls):
        subs = cls.__subclasses__()
        if not subs:
            yield cls
        for sub in subs:
            yield from leaves(sub)

    # NOT `if hasattr(c, "category")` -- that guard is always true, because
    # category() is defined on the MeasurementInfo base and RAISES rather
    # than being absent. This runs at module scope, so a leaf without an
    # override would fail the import of _grid.py and take the Colony tab
    # with it. Implement the guard's actual intent.
    cats = set()
    for c in leaves(MeasurementInfo):
        try:
            cats.add(c.category())
        except NotImplementedError:
            continue
    return tuple(sorted(f"{c}_" for c in cats - _AXIS_ELIGIBLE_CATEGORIES))


_MEASUREMENT_PREFIXES: tuple[str, ...] = _derive_measurement_prefixes()
```

- [ ] **Step 4: Run the test and the colony-grid suite**

```bash
uv run pytest tests/unit/gui/results_viewer/test_measurement_prefixes.py -v   # one file: inline
sbatch --export=ALL,SCOPE=gui docs/superpowers/plans/2026-09-01-results-scatter-tab/run_scatter_tests.sbatch
```
The single new file runs inline; the colony/axis/grid regression scope is
wide enough to belong on Slurm.

Expected: the new file passes. If a colony-grid test asserts a specific option list, update it — the option list legitimately changes, and that is the point of the fix.

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/colony_view/_grid.py tests/unit/gui/results_viewer/test_measurement_prefixes.py
git add <exactly the files this task's **Files:** block names>   # never -A
git commit -m "fix(gui): derive _MEASUREMENT_PREFIXES from schema categories

The literal listed TextureGray_, which no schema declares, so Texture_
columns were never excluded from axis pickers; and it omitted 26 real
categories including Size_. Derives from MeasurementInfo categories minus
the metadata/identity/grouping families, which must stay selectable."
```

---

## Task 5: `FigureSpec` and the phantom predicate

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_scatter_tab/__init__.py` (empty for now), `_spec.py`
- Test: `tests/unit/gui/results_viewer/test_scatter_spec.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `FigureSpec` — frozen dataclass with fields `section_col: str | None`, `row_col: str | None`, `col_col: str | None`, `x_col: str`, `y_col: str`, `hue_col: str | None`, `shape_col: str | None`, `share_axes: bool`, `show_removed: bool`, `sizes: dict[str, int]`, `marker_size: int`, `marker_opacity: float`
  - `plottable(df: pl.DataFrame) -> pl.DataFrame`

- [ ] **Step 1: Write the failing test**

```python
"""FigureSpec is a pure config object; plottable drops metadata phantoms."""
from __future__ import annotations

import polars as pl

from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec, plottable


def test_plottable_drops_phantom_rows() -> None:
    df = pl.DataFrame({
        "Object_Label": [1, 2, None, None],
        "QC_MetadataOnly": [False, False, True, True],
        "Shape_Area": [10.0, 20.0, None, None],
    })
    out = plottable(df)
    assert out.height == 2
    assert out["Object_Label"].to_list() == [1, 2]


def test_plottable_is_a_no_op_without_the_curation_column() -> None:
    """A per-store table carries no QC_MetadataOnly; it must not crash."""
    df = pl.DataFrame({"Object_Label": [1, 2], "Shape_Area": [10.0, 20.0]})
    assert plottable(df).height == 2


def test_figure_spec_is_frozen() -> None:
    spec = FigureSpec(x_col="Metadata_FrameIndex", y_col="Shape_Area")
    assert spec.share_axes is True
    assert spec.hue_col is None
    try:
        spec.x_col = "other"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("FigureSpec must be frozen")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/gui/results_viewer/test_scatter_spec.py -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `_spec.py`**

```python
"""The pure configuration object the Scatter tab's figures are built from.

Both destinations -- the on-screen ``dcc.Graph`` and the kaleido export --
consume one ``FigureSpec``, so the PDF cannot drift from the screen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

#: The curation column that marks a metadata-only row. Written by the CLI
#: into the measurements mirror; absent from per-store tables.
CURATION_PHANTOM_COL = "QC_MetadataOnly"


@dataclass(frozen=True)
class FigureSpec:
    """Every role and size that defines one Scatter figure.

    Args:
        x_col: Column plotted on X.
        y_col: Column plotted on Y.
        section_col: Column whose values become sections (PDF pages).
        row_col: Column whose values become facet rows.
        col_col: Column whose values become facet columns.
        hue_col: Column mapped to marker colour.
        shape_col: Column mapped to marker symbol.
        share_axes: Whether all facets share one X and Y range.
        show_removed: Whether curation-removed colonies render as grey x.
        sizes: Type sizes in px, keyed by role.
        marker_size: Marker area in points squared.
        marker_opacity: Marker alpha in ``[0, 1]``.
    """

    x_col: str
    y_col: str
    section_col: str | None = None
    row_col: str | None = None
    col_col: str | None = None
    hue_col: str | None = None
    shape_col: str | None = None
    share_axes: bool = True
    show_removed: bool = True
    sizes: dict[str, int] = field(
        default_factory=lambda: {
            "section": 14, "facet": 9, "axis": 8, "tick": 7, "legend": 8
        }
    )
    marker_size: int = 6
    marker_opacity: float = 0.5


def plottable(df: pl.DataFrame) -> pl.DataFrame:
    """Drop metadata-only phantom rows, which cannot become points.

    A phantom has no ``Object_Label``, no coordinates and no crop. In the
    verification fixture 121 of 844 rows are phantoms; in the full run it is
    117,415 of 231,229. The proportion varies, the rule does not.

    Args:
        df: A viewer frame, normally ``OutputRoot.master_df``.

    Returns:
        The subset that can be plotted. Returned unchanged when the frame
        carries no curation column.
    """
    if CURATION_PHANTOM_COL not in df.columns:
        return df
    # DO NOT reach for `strict=False` here. Measured on polars 1.41:
    # Utf8 -> Boolean is unsupported outright, and BOTH strict=True and
    # strict=False raise InvalidOperationError -- even on clean values like
    # ["false", "true"] or ["0", "1"]. So `strict=False` is not a lenient
    # cast that yields nulls; it is the same unsupported operation, and a
    # string-typed flag still 500s the tab.
    #
    # Branch on the dtype instead. Whatever shape is chosen, keep the
    # failure DIRECTION: a malformed flag must show too many points, never
    # silently hide real colonies.
    col = df[CURATION_PHANTOM_COL]
    if col.dtype != pl.Boolean:
        return df
    return df.filter(~pl.col(CURATION_PHANTOM_COL).fill_null(False))
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/gui/results_viewer/test_scatter_spec.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/_scatter_tab/ tests/unit/gui/results_viewer/test_scatter_spec.py
git add <exactly the files this task's **Files:** block names>   # never -A
git commit -m "feat(gui): add FigureSpec and the phantom-row predicate"
```

---

## Task 6: Facet planning, ordering and caps

Spec §5.2, §5.3, §1.3.

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_scatter_tab/_facets.py`
- Modify: `src/phenotypic/gui/_config.py`
- Test: `tests/unit/gui/results_viewer/test_scatter_facets.py`

**Interfaces:**
- Consumes: `FigureSpec` (Task 5).
- Produces:
  - `sort_facet_values(values: list[str]) -> list[str]`
  - `plan_facets(df, spec, cap) -> FacetPlan` where `FacetPlan` has `rows: list[str]`, `cols: list[str]`, `truncated: bool`, `total: int`
  - `SCATTER_FACET_CAP: int`, `SECTION_GROUP_CAP: int`

- [ ] **Step 1: Write the failing test**

```python
"""Facet ordering and the two caps that bound a grid."""
from __future__ import annotations

import polars as pl

from phenotypic.gui.results_viewer._scatter_tab._facets import (
    plan_facets,
    sort_facet_values,
)
from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec


def test_numeric_looking_values_sort_numerically() -> None:
    """Grid_ColNum is a String column with values 0..11.

    A plain string sort gives 0, 1, 10, 11, 2 -- which renders as a
    scrambled grid and reads like a rendering bug rather than a sort bug.
    """
    assert sort_facet_values(["10", "2", "0", "11", "1"]) == ["0", "1", "2", "10", "11"]


def test_non_numeric_values_sort_lexically() -> None:
    assert sort_facet_values(["b", "a", "c"]) == ["a", "b", "c"]


def test_mixed_values_fall_back_to_lexical() -> None:
    """If any value fails to parse, every value sorts as a string."""
    assert sort_facet_values(["10", "a", "2"]) == ["10", "2", "a"]


def test_the_grid_is_capped_by_the_product_not_per_axis() -> None:
    """A 12x12 selection is 144 panels; no context budget survives it."""
    df = pl.DataFrame({
        "r": [str(i) for i in range(12) for _ in range(12)],
        "c": [str(j) for _ in range(12) for j in range(12)],
    })
    spec = FigureSpec(x_col="x", y_col="y", row_col="r", col_col="c")
    plan = plan_facets(df, spec, cap=16)

    assert len(plan.rows) * len(plan.cols) <= 16
    assert plan.truncated is True
    assert plan.total == 144


def test_an_uncapped_grid_is_not_marked_truncated() -> None:
    df = pl.DataFrame({"r": ["0", "1"], "c": ["0", "1"]})
    spec = FigureSpec(x_col="x", y_col="y", row_col="r", col_col="c")
    plan = plan_facets(df, spec, cap=16)
    assert plan.truncated is False and plan.total == 4
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/gui/results_viewer/test_scatter_facets.py -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `_facets.py`**

```python
"""Facet planning: which values become rows and columns, in what order."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from phenotypic.gui._config import SCATTER_FACET_CAP
from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec


@dataclass(frozen=True)
class FacetPlan:
    """The grid a figure will draw.

    Args:
        rows: Ordered row values, already capped.
        cols: Ordered column values, already capped.
        truncated: Whether the cap removed any panel.
        total: Panels the uncapped selection would have produced.
    """

    rows: list[str]
    cols: list[str]
    truncated: bool
    total: int


def sort_facet_values(values: list[str]) -> list[str]:
    """Order facet values numerically when every value parses.

    Grid and many metadata columns are ``String`` even when their values
    are numbers, so a plain sort orders ``Grid_ColNum`` as 0, 1, 10, 11, 2.
    Falls back to a lexical sort the moment any value is non-numeric, so a
    mixed column is ordered consistently rather than half-numerically.

    Args:
        values: Distinct facet values as strings.

    Returns:
        The same values, ordered.
    """
    try:
        return sorted(values, key=lambda v: (float(v), v))
    except (TypeError, ValueError):
        return sorted(values, key=str)


def plan_facets(
    df: pl.DataFrame, spec: FigureSpec, cap: int = SCATTER_FACET_CAP
) -> FacetPlan:
    """Choose the grid's rows and columns, capped by their product.

    The cap bounds ``rows * cols``, not either axis alone: a 12-value row
    axis crossed with a 12-value column axis is 144 panels. Over-cap keeps
    the first panels in facet-value order -- deterministic, and independent
    of how the data happens to be distributed -- and flags ``truncated`` so
    the caller can surface "showing first N of M" rather than silently
    dropping panels.

    Args:
        df: The plottable frame.
        spec: The figure's configuration.
        cap: Maximum number of panels.

    Returns:
        A :class:`FacetPlan`.
    """
    def _values(col: str | None) -> list[str]:
        if col is None or col not in df.columns:
            return [""]
        raw = df[col].drop_nulls().unique().cast(pl.String).to_list()
        return sort_facet_values(raw) or [""]

    rows, cols = _values(spec.row_col), _values(spec.col_col)
    total = len(rows) * len(cols)
    if total <= cap:
        return FacetPlan(rows=rows, cols=cols, truncated=False, total=total)

    kept_rows = rows
    kept_cols = cols
    while len(kept_rows) * len(kept_cols) > cap:
        if len(kept_rows) >= len(kept_cols) and len(kept_rows) > 1:
            kept_rows = kept_rows[:-1]
        elif len(kept_cols) > 1:
            kept_cols = kept_cols[:-1]
        else:
            break
    return FacetPlan(rows=kept_rows, cols=kept_cols, truncated=True, total=total)
```

- [ ] **Step 4: Add the two caps to `_config.py`**

```python
#: Maximum panels in one Scatter facet grid, as ``rows * cols``.
#: NOT a WebGL-context bound -- Plotly pools every gl trace into one shared
#: gl-container (measured: 1 container at 1, 4, 16 and 36 subplots), so
#: facet count does not consume contexts the way TIMELINE_COMPARE_CAP's
#: independent OpenSeadragon viewers do. This bounds legibility (below
#: ~200 px a panel stops being readable), point count per figure, and axis
#: and tick DOM. Over-cap renders the first N in facet-value order plus a
#: visible "showing first N of M" notice, never a silent truncation.
SCATTER_FACET_CAP: int = 24

#: Maximum distinct values a column may have to be offered as a section
#: group. Each section is one PDF page and one pager step, so an unbounded
#: control lets a continuous column ask for one page per colony.
SECTION_GROUP_CAP: int = 60
```

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/unit/gui/results_viewer/test_scatter_facets.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/_scatter_tab/ src/phenotypic/gui/_config.py tests/unit/gui/results_viewer/test_scatter_facets.py
git add <exactly the files this task's **Files:** block names>   # never -A
git commit -m "feat(gui): add Scatter facet planning, ordering and caps"
```

---

## Task 7: Group measurement columns by their measurer

Spec §8. The naive implementation throws on `MeasureTexture`.

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_scatter_tab/_grouping.py`
- Test: `tests/unit/gui/results_viewer/test_scatter_grouping.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `group_columns(columns: list[str], meas_cfg: dict) -> dict[str, list[str]]`

- [ ] **Step 1: Write the failing test**

```python
"""Columns group by the measurer that emits them, from the run's own config."""
from __future__ import annotations

from phenotypic.gui.results_viewer._scatter_tab._grouping import group_columns

MEAS = {
    "MeasureShape": {"class": "MeasureShape", "params": {}},
    "MeasureIntensity": {"class": "MeasureIntensity", "params": {}},
    "MeasureColor": {
        "class": "MeasureColor",
        "params": {"include_XYZ": False, "include_xy": False},
    },
    "MeasureTexture": {"class": "MeasureTexture", "params": {"scale": [5]}},
    "MeasureNeighborDist": {"class": "MeasureNeighborDist", "params": {}},
}


def test_exact_headers_group_by_measurer() -> None:
    """Also asserts the NEGATIVE, or a total-failure bug passes.

    If the implementation calls ``get_measurement_infoclasses`` on the class
    rather than an instance it raises, every measurer is skipped, and every
    column lands in Unattributed. Asserting only "Shape_Area is in
    MeasureShape" would KeyError, but asserting membership without
    asserting absence let three sibling tests pass against exactly that bug.
    """
    groups = group_columns(["Shape_Area", "Intensity_MeanIntensity"], MEAS)
    assert "Shape_Area" in groups["MeasureShape"]
    assert "Intensity_MeanIntensity" in groups["MeasureIntensity"]
    assert "Unattributed" not in groups, (
        "no column here is unclaimed; an Unattributed group means the "
        "measurers were never successfully constructed"
    )


def test_parameterized_schemas_fall_back_to_category() -> None:
    """TEXTURE.get_headers requires a `scale` argument.

    Naively this raises TypeError and dumps every Texture_ column into
    Unattributed -- 65 of 148 columns on the verification fixture.
    """
    groups = group_columns(["Texture_Contrast-deg000-scale05"], MEAS)
    assert groups["MeasureTexture"] == ["Texture_Contrast-deg000-scale05"]


def test_measurer_params_change_the_claimed_headers() -> None:
    """The same column must flip groups when the run's params change.

    Asserting only the "off" case passes against an implementation that
    claims NOTHING -- everything is Unattributed, so the column is
    correctly absent from MeasureColor for entirely the wrong reason. The
    "on" case is what makes this discriminate.
    """
    off = group_columns(["ColorXYZ_X"], MEAS)
    assert "ColorXYZ_X" not in off.get("MeasureColor", [])
    assert "ColorXYZ_X" in off["Unattributed"]

    on_cfg = dict(MEAS)
    on_cfg["MeasureColor"] = {
        "class": "MeasureColor",
        "params": {"include_XYZ": True, "include_xy": True},
    }
    on = group_columns(["ColorXYZ_X"], on_cfg)
    assert "ColorXYZ_X" in on["MeasureColor"], (
        "with include_XYZ=True the column must be claimed -- if it is still "
        "Unattributed the measurer is not being constructed from its params"
    )


def test_metadata_is_one_flat_group_and_curation_is_its_own() -> None:
    groups = group_columns(
        ["Metadata_Strain", "Metadata_PlateID", "QC_MetadataOnly"], MEAS
    )
    assert set(groups["Metadata"]) == {"Metadata_Strain", "Metadata_PlateID"}
    assert groups["Curation"] == ["QC_MetadataOnly"]


def test_unclaimed_columns_land_in_unattributed() -> None:
    """Mixes claimed and unclaimed, so "everything is unattributed" fails."""
    groups = group_columns(
        ["Object_Label", "Bbox_CenterRR", "Grid_RowNum", "Shape_Area"], MEAS
    )
    assert set(groups["Unattributed"]) == {
        "Object_Label", "Bbox_CenterRR", "Grid_RowNum"
    }
    assert groups["MeasureShape"] == ["Shape_Area"]
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/gui/results_viewer/test_scatter_grouping.py -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `_grouping.py`**

```python
"""Attribute measurement columns to the measurer that emitted them."""
from __future__ import annotations

import logging

from phenotypic.sdk_ import is_metadata_header

logger = logging.getLogger(__name__)

#: The curation column gets its own heading rather than falling into
#: Unattributed: no measurer claims it and ``is_metadata_header`` rejects
#: it, but it is the column the phantom predicate depends on, so burying it
#: in a bucket named "Unattributed" is the worst of the options.
CURATION_COLUMNS = ("QC_MetadataOnly",)


def group_columns(
    columns: list[str], meas_cfg: dict[str, dict]
) -> dict[str, list[str]]:
    """Group columns under the ``MeasureFeatures`` class that emits them.

    Measurers are instantiated **from their recorded params**, not used as
    classes: ``get_measurement_infoclasses`` is an instance method and its
    result depends on parameters -- ``MeasureColor()`` yields ColorLab and
    ColorHSV, while ``MeasureColor(include_XYZ=True, include_xy=True)``
    yields four schemas.

    ``get_headers()`` is not uniformly zero-argument. ``TEXTURE`` takes a
    ``scale`` because its column names carry the offset, so a bare call
    raises ``TypeError``. Rather than special-casing each schema, such a
    class falls back to matching the frame's columns against its
    ``category()`` -- which generalizes to schemas that do not exist yet.

    Args:
        columns: Column names to group.
        meas_cfg: The ``"meas"`` block of the run's pipeline config, mapping
            a key to ``{"class": str, "params": dict}``.

    Returns:
        Group name to column names. Always includes ``"Metadata"``,
        ``"Curation"`` and ``"Unattributed"`` keys when non-empty.
    """
    import phenotypic.measure as measure_mod

    owner: dict[str, str] = {}
    for cfg in meas_cfg.values():
        name = cfg.get("class")
        cls = getattr(measure_mod, name, None)
        if cls is None:
            logger.debug("scatter grouping: unknown measurer %r", name)
            continue
        try:
            op = cls(**cfg.get("params", {}))
        except Exception:
            logger.debug("scatter grouping: could not construct %r", name)
            continue
        for info in op.get_measurement_infoclasses():
            try:
                headers = list(info.get_headers())
            except TypeError:
                prefix = f"{info.category()}_"
                headers = [c for c in columns if c.startswith(prefix)]
            for header in headers:
                owner.setdefault(header, name)

    groups: dict[str, list[str]] = {}
    for col in columns:
        if col in CURATION_COLUMNS:
            key = "Curation"
        elif is_metadata_header(col):
            key = "Metadata"
        else:
            key = owner.get(col, "Unattributed")
        groups.setdefault(key, []).append(col)
    return groups
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/gui/results_viewer/test_scatter_grouping.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Verify against the fixture**

```bash
uv run python -c "
import json, polars as pl
from phenotypic.gui.results_viewer._scatter_tab._grouping import group_columns
D='/rhome/anguy344/bigdata_exfab/projects/ucr_029_e_d_Maresca/data/results/2026-08-11-migration-test/deliverables'
cfg=json.load(open(D+'/pipeline.json.pht-pipe'))['meas']
cols=pl.read_parquet(D+'/measurements.parquet').columns
g=group_columns(cols,cfg)
for k in sorted(g): print(f'{k:22s} {len(g[k])}')
print('total', sum(len(v) for v in g.values()), 'of', len(cols))
"
```

Expected exactly: MeasureShape 17, MeasureColor 15, MeasureIntensity 12, MeasureNeighborDist 8, MeasureTexture 65, Metadata 16, Curation 1, Unattributed 15 — total 149 of 149.

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/_scatter_tab/ tests/unit/gui/results_viewer/test_scatter_grouping.py
git add <exactly the files this task's **Files:** block names>   # never -A
git commit -m "feat(gui): group measurement columns by their measurer

Instantiates each measurer from the run's recorded params, since the
emitted header set depends on them. Parameterized schemas whose
get_headers takes an argument (TEXTURE takes scale) fall back to matching
on category rather than being special-cased."
```

---

## Task 8: Derived frame index

Spec §10.

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_scatter_tab/_facets.py`
- Test: `tests/unit/gui/results_viewer/test_scatter_frame_index.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `derive_frame_index(df, plate_col="Metadata_PlateID", time_col="Metadata_ImageDatetime") -> pl.DataFrame` — adds a `Computed_FrameIndex` Int32 column.

- [ ] **Step 1: Write the failing test**

```python
"""Frame index ranks images chronologically within a plate."""
from __future__ import annotations

import polars as pl

from phenotypic.gui.results_viewer._scatter_tab._facets import derive_frame_index


def test_index_ranks_within_each_plate_independently() -> None:
    df = pl.DataFrame({
        "Metadata_PlateID": ["A", "A", "A", "B", "B"],
        "Metadata_ImageDatetime": [
            "2026-07-26T06:00:00", "2026-07-26T18:00:00", "2026-07-27T06:00:00",
            "2026-07-26T09:00:00", "2026-07-26T21:00:00",
        ],
    })
    out = derive_frame_index(df)
    assert out["Computed_FrameIndex"].to_list() == [0, 1, 2, 0, 1]


def test_repeated_timestamps_share_one_index() -> None:
    """Two colonies in the same image are the same frame."""
    df = pl.DataFrame({
        "Metadata_PlateID": ["A", "A", "A"],
        "Metadata_ImageDatetime": [
            "2026-07-26T06:00:00", "2026-07-26T06:00:00", "2026-07-26T18:00:00",
        ],
    })
    assert derive_frame_index(df)["Computed_FrameIndex"].to_list() == [0, 0, 1]


def test_null_datetimes_get_a_null_index_not_zero() -> None:
    """The fixture has 81 such rows; they must be excluded, not ranked 0."""
    df = pl.DataFrame({
        "Metadata_PlateID": ["A", "A"],
        "Metadata_ImageDatetime": ["2026-07-26T06:00:00", None],
    })
    assert derive_frame_index(df)["Computed_FrameIndex"].to_list() == [0, None]
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/gui/results_viewer/test_scatter_frame_index.py -v
```

Expected: FAIL — `cannot import name 'derive_frame_index'`.

- [ ] **Step 3: Implement it**

Append to `_facets.py`:

```python
#: The column name the derived frame index is written to.
COMPUTED_FRAME_INDEX = "Computed_FrameIndex"


def derive_frame_index(
    df: pl.DataFrame,
    plate_col: str = "Metadata_PlateID",
    time_col: str = "Metadata_ImageDatetime",
) -> pl.DataFrame:
    """Rank each image chronologically within its plate, zero-based.

    Needed because ``Metadata_FrameIndex`` is often unpopulated and
    ``Metadata_Timepoint`` can be a constant. Ranks distinct timestamps, so
    every colony in one image shares a frame. Rows with a null timestamp
    get a null index and are excluded from the plot rather than ranked
    zero -- the verification fixture has 81 of them.

    Args:
        df: A plottable frame.
        plate_col: Column identifying the plate.
        time_col: Column carrying the capture timestamp.

    Returns:
        ``df`` with a ``Computed_FrameIndex`` Int32 column appended.
    """
    if plate_col not in df.columns or time_col not in df.columns:
        return df.with_columns(
            pl.lit(None, dtype=pl.Int32).alias(COMPUTED_FRAME_INDEX)
        )
    ranked = (
        df.select([plate_col, time_col])
        .unique()
        .drop_nulls()
        .sort([plate_col, time_col])
        .with_columns(
            pl.col(time_col).cum_count().over(plate_col).sub(1)
            .cast(pl.Int32).alias(COMPUTED_FRAME_INDEX)
        )
    )
    return df.join(ranked, on=[plate_col, time_col], how="left")
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/gui/results_viewer/test_scatter_frame_index.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/_scatter_tab/_facets.py tests/unit/gui/results_viewer/test_scatter_frame_index.py
git add <exactly the files this task's **Files:** block names>   # never -A
git commit -m "feat(gui): derive a frame index from capture order within plate"
```

---

## Task 9: The pure figure builder

Spec §4, §5, §11.1.

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_scatter_tab/_figure.py`
- Test: `tests/unit/gui/results_viewer/test_scatter_figure.py`

**Interfaces:**
- Consumes: `FigureSpec`, `plottable` (Task 5); `plan_facets` (Task 6).
- Produces:
  - `build_scatter_figure(df, spec, plan, *, for_export=False) -> go.Figure`
  - `CUSTOMDATA_COL = "_scatter_row_index"`

- [ ] **Step 1: Write the failing test**

```python
"""The Scatter figure builder is pure: no Dash, no I/O."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl

from phenotypic.gui.results_viewer._scatter_tab._facets import plan_facets
from phenotypic.gui.results_viewer._scatter_tab._figure import build_scatter_figure
from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec


def _frame(n: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame({
        "x": rng.integers(0, 8, n).tolist(),
        "y": rng.normal(10, 2, n).tolist(),
        "r": ["0" if i % 2 else "1" for i in range(n)],
        "c": ["0" if i % 3 else "1" for i in range(n)],
        "hue": ["a" if i % 2 else "b" for i in range(n)],
        "_scatter_row_index": list(range(n)),
    })


def _spec() -> FigureSpec:
    return FigureSpec(x_col="x", y_col="y", row_col="r", col_col="c", hue_col="hue")


def test_the_screen_figure_uses_webgl_traces() -> None:
    """SVG go.Scatter cannot render at this project's point counts."""
    df, spec = _frame(), _spec()
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))
    assert fig.data, "no traces were added"
    assert all(t.type == "scattergl" for t in fig.data)


def test_the_export_figure_uses_svg_traces() -> None:
    """kaleido renders Scattergl as blank axes -- 624 non-white px against
    46,886 for SVG, with no warning and exit code 0. The export pass must
    substitute the trace type or every PDF is empty."""
    df, spec = _frame(), _spec()
    fig = build_scatter_figure(df, spec, plan_facets(df, spec), for_export=True)
    assert fig.data
    assert all(t.type == "scatter" for t in fig.data)


def test_a_series_absent_from_the_first_cell_still_reaches_the_legend() -> None:
    """``showlegend=first_cell`` drops any series missing from cell 1.

    On a sparse frame -- 22 strains over 36 images in the fixture -- that
    is the common case, not a corner: a hue that happens not to appear in
    the top-left facet vanishes from the legend while its points are drawn.
    Track which series have been given a legend entry instead.
    """
    df = pl.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [1.0, 2.0, 3.0, 4.0],
        "r": ["0", "0", "1", "1"],
        "c": ["0", "0", "0", "0"],
        "hue": ["a", "a", "b", "b"],      # 'b' never appears in cell (0,0)
        "_scatter_row_index": [0, 1, 2, 3],
    })
    spec = FigureSpec(x_col="x", y_col="y", row_col="r", col_col="c", hue_col="hue")
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))

    drawn = {t.name for t in fig.data}
    in_legend = {t.name for t in fig.data if t.showlegend}
    assert drawn == in_legend, f"drawn {drawn} but only {in_legend} in the legend"


def test_every_point_carries_the_index_of_its_own_row() -> None:
    """Correspondence, not membership. This is the assertion that let B1 pass.

    The original asserted ``seen <= set(range(df.height))`` -- that every
    index falls in a plausible range. A filtered-frame index satisfies that
    perfectly while pointing at the wrong row, which is exactly how the
    filtered-vs-master defect survived a careful read. Assert that the index
    on a point resolves to the row whose x and y that point actually carries.
    """
    df, spec = _frame(), _spec()
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))

    n_points = 0
    for trace in fig.data:
        for x, y, cd in zip(trace.x, trace.y, trace.customdata):
            row = df.row(int(cd[0]), named=True)
            assert row["x"] == x and row["y"] == y, (
                f"index {int(cd[0])} points at ({row['x']}, {row['y']}) but "
                f"the marker is at ({x}, {y})"
            )
            n_points += 1
    assert n_points == df.height, "every row must be drawn exactly once"


def test_an_empty_facet_still_occupies_its_cell() -> None:
    """A missing (row, col) combination must not collapse the geometry.

    Four cells are planned, three carry data. The grid must still be 2x2 --
    a facet that silently disappears shifts every panel after it and
    misreads as data rather than as absence.
    """
    df = pl.DataFrame({
        "x": [1, 2, 3],
        "y": [1.0, 2.0, 3.0],
        "r": ["0", "0", "1"],
        "c": ["0", "1", "0"],      # (r=1, c=1) is absent
        "_scatter_row_index": [0, 1, 2],
    })
    spec = FigureSpec(x_col="x", y_col="y", row_col="r", col_col="c")
    plan = plan_facets(df, spec)
    assert len(plan.rows) == 2 and len(plan.cols) == 2

    fig = build_scatter_figure(df, spec, plan)
    axes = {k for k in fig.layout.to_plotly_json() if k.startswith("xaxis")}
    assert len(axes) == 4, f"expected a 2x2 grid of axes, got {len(axes)}"
    assert len(fig.data) == 3, "one trace per non-empty cell"


def test_shared_axes_give_every_facet_one_range() -> None:
    df, spec_ = _frame(), _spec()
    fig = build_scatter_figure(df, spec_, plan_facets(df, spec_))
    ranges = {
        tuple(v["range"]) for k, v in fig.layout.to_plotly_json().items()
        if k.startswith("yaxis") and isinstance(v, dict) and v.get("range")
    }
    # `<= 1` would pass on ZERO ranges too, which is exactly what
    # share_axes=False produces -- so the assertion must be `== 1` or it
    # cannot detect the regression it is named for.
    assert len(ranges) == 1, f"expected one shared y range, got {ranges}"


def test_unshared_axes_do_not_set_one_range() -> None:
    """The negative half. Without this, the test above proves nothing."""
    df = _frame()
    spec = FigureSpec(
        x_col="x", y_col="y", row_col="r", col_col="c", share_axes=False
    )
    fig = build_scatter_figure(df, spec, plan_facets(df, spec))
    ranges = {
        tuple(v["range"]) for k, v in fig.layout.to_plotly_json().items()
        if k.startswith("yaxis") and isinstance(v, dict) and v.get("range")
    }
    assert len(ranges) != 1
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/gui/results_viewer/test_scatter_figure.py -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `_figure.py`**

```python
"""Pure Plotly figure construction for the Scatter tab.

Side-effect free and Dash-free so it can be unit-tested against synthetic
frames without booting a server, following ``_heatmap_tab/_figure.py``.
"""
from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from phenotypic.gui._design import OKABE_ITO
from phenotypic.gui.results_viewer._scatter_tab._facets import FacetPlan
from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec

#: Column carrying each point's positional index into ``master_df``.
CUSTOMDATA_COL = "_scatter_row_index"

#: Marker symbols in the order the shape channel consumes them.
_SYMBOLS: tuple[str, ...] = ("circle", "square", "triangle-up", "diamond", "x")


def _axis_range(df: pl.DataFrame, col: str) -> tuple[float, float] | None:
    """Padded (min, max) over the whole frame, or None if not numeric."""
    try:
        series = df[col].cast(pl.Float64, strict=False).drop_nulls()
    except Exception:
        return None
    if series.len() == 0:
        return None
    lo, hi = float(series.min()), float(series.max())
    pad = (hi - lo) * 0.05 or 1.0
    return (lo - pad, hi + pad)


def build_scatter_figure(
    df: pl.DataFrame,
    spec: FigureSpec,
    plan: FacetPlan,
    *,
    for_export: bool = False,
) -> go.Figure:
    """Build one section's faceted scatter figure.

    Args:
        df: The plottable frame for ONE section, carrying ``CUSTOMDATA_COL``.
        spec: Roles, sizes and scales.
        plan: The facet grid to draw.
        for_export: When True, use SVG ``go.Scatter`` traces. kaleido
            renders ``Scattergl`` as blank axes with exit code 0 and no
            warning, so the export pass MUST substitute the trace type.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    trace_cls = go.Scatter if for_export else go.Scattergl
    n_rows, n_cols = max(len(plan.rows), 1), max(len(plan.cols), 1)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=spec.share_axes,
        shared_yaxes=spec.share_axes,
        subplot_titles=[
            f"{spec.col_col}={c}" if spec.col_col else ""
            for _ in plan.rows for c in plan.cols
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.04,
    )

    hues = (
        sorted(df[spec.hue_col].drop_nulls().unique().cast(pl.String).to_list())
        if spec.hue_col and spec.hue_col in df.columns else [None]
    )
    shapes = (
        sorted(df[spec.shape_col].drop_nulls().unique().cast(pl.String).to_list())
        if spec.shape_col and spec.shape_col in df.columns else [None]
    )

    # Track which series already have a legend entry. Keying on the series
    # rather than on "is this the first cell" is what stops a hue absent
    # from cell (0,0) from vanishing out of the legend on a sparse frame.
    legended: set[str] = set()
    for r_i, r_val in enumerate(plan.rows or [""], start=1):
        for c_i, c_val in enumerate(plan.cols or [""], start=1):
            cell = df
            if spec.row_col and spec.row_col in df.columns and r_val != "":
                cell = cell.filter(pl.col(spec.row_col).cast(pl.String) == r_val)
            if spec.col_col and spec.col_col in df.columns and c_val != "":
                cell = cell.filter(pl.col(spec.col_col).cast(pl.String) == c_val)

            for h_i, hue in enumerate(hues):
                for s_i, shape in enumerate(shapes):
                    part = cell
                    if hue is not None:
                        part = part.filter(
                            pl.col(spec.hue_col).cast(pl.String) == hue
                        )
                    if shape is not None:
                        part = part.filter(
                            pl.col(spec.shape_col).cast(pl.String) == shape
                        )
                    if part.height == 0:
                        continue
                    label = " · ".join(
                        p for p in (
                            f"{spec.hue_col}={hue}" if hue is not None else "",
                            f"{spec.shape_col}={shape}" if shape is not None else "",
                        ) if p
                    ) or spec.y_col
                    fig.add_trace(
                        trace_cls(
                            x=part[spec.x_col].to_list(),
                            y=part[spec.y_col].to_list(),
                            mode="markers",
                            name=label,
                            legendgroup=label,
                            showlegend=label not in legended,
                            customdata=[[i] for i in part[CUSTOMDATA_COL].to_list()],
                            marker=dict(
                                size=spec.marker_size,
                                opacity=spec.marker_opacity,
                                color=OKABE_ITO[h_i % len(OKABE_ITO)],
                                symbol=_SYMBOLS[s_i % len(_SYMBOLS)],
                                line=dict(width=0),
                            ),
                        ),
                        row=r_i,
                        col=c_i,
                    )
                    legended.add(label)

    if spec.share_axes:
        x_rng, y_rng = _axis_range(df, spec.x_col), _axis_range(df, spec.y_col)
        if x_rng:
            fig.update_xaxes(range=list(x_rng))
        if y_rng:
            fig.update_yaxes(range=list(y_rng))

    fig.update_layout(
        font=dict(size=spec.sizes["axis"]),
        legend=dict(font=dict(size=spec.sizes["legend"])),
        margin=dict(l=60, r=20, t=40, b=50),
    )
    fig.update_xaxes(tickfont=dict(size=spec.sizes["tick"]))
    fig.update_yaxes(tickfont=dict(size=spec.sizes["tick"]))
    return fig
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/gui/results_viewer/test_scatter_figure.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/_scatter_tab/_figure.py tests/unit/gui/results_viewer/test_scatter_figure.py
git add <exactly the files this task's **Files:** block names>   # never -A
git commit -m "feat(gui): add the pure Scatter figure builder

Scattergl for the screen, Scatter for export -- kaleido renders a gl layer
as blank axes, silently."
```

---

## Task 10: PDF export

Spec §11.

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_scatter_tab/_pdf.py`
- Modify: `pyproject.toml` — add `pypdf`
- Test: `tests/unit/gui/results_viewer/test_scatter_pdf.py`

**Interfaces:**
- Consumes: `build_scatter_figure` (Task 9).
- Produces: `export_sections_pdf(df, spec, sections, *, width_in=16, height_in=12) -> bytes`

- [ ] **Step 1: Add the one dependency**

```bash
uv add pypdf
```

**No rasterizer is needed, and none should be added.** The ink assertion
renders a single page to **PNG** rather than to PDF: `write_fig_sync` takes
its format from the path extension, and Pillow and numpy are already
dependencies. That is also the spec's own methodology — §11.1's evidence
(`gl.png` 624 non-white against `svg.png` 46,886) was measured on PNGs, so
the regression test becomes the measurement rather than an approximation of
it.

Two rejected alternatives, recorded so they are not re-proposed:

- **pypdf text or image extraction.** A kaleido chart is vector, so
  `page.images` is empty, and `extract_text()` returns the axis labels —
  which are **identical** on a blank and a rendered page. §11.1's failure
  mode is precisely that the axes render and the markers do not, so a text
  assertion passes against the broken export.
- **Adding pymupdf.** No rasterizer exists in the tree (fitz, pymupdf,
  pypdfium2 and pdf2image are all absent), and adding a dependency to test
  a property already measurable on a PNG is not worth it.

If a future change must assert on the merged PDF itself, use
`page.get_contents().get_data()` length — 5,000 markers are 5,000 path
operators — rather than reaching for a rasterizer.

- [ ] **Step 2: Write the failing test**

```python
"""Export must produce a PDF with visible ink, not merely a valid file."""
from __future__ import annotations

import os

import numpy as np
import polars as pl
import pytest

from phenotypic.gui.results_viewer._scatter_tab._pdf import export_sections_pdf
from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec

pytest.importorskip("pypdf")


@pytest.fixture(scope="module")
def chrome_or_skip() -> None:
    """Skip, loudly, when kaleido has no browser to drive.

    Both export tests need Chrome. Without this they do not skip -- they
    hard-fail with kaleido's RuntimeError, which reads as a broken export
    rather than a missing prerequisite.
    """
    import shutil
    from pathlib import Path

    if not (
        any(shutil.which(b) for b in ("google-chrome", "chromium", "chrome"))
        or Path.home().joinpath(".cache/kaleido").exists()
        or os.environ.get("BROWSER_PATH")
    ):
        pytest.skip("no Chrome for kaleido; run `uv run plotly_get_chrome`")


def _frame(n: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    return pl.DataFrame({
        "x": rng.integers(0, 8, n).tolist(),
        "y": rng.normal(10, 2, n).tolist(),
        "s": ["A" if i % 2 else "B" for i in range(n)],
        "_scatter_row_index": list(range(n)),
    })


def test_one_page_is_written_per_section(chrome_or_skip) -> None:
    import io
    from pypdf import PdfReader

    df = _frame()
    spec = FigureSpec(x_col="x", y_col="y", section_col="s")
    out = export_sections_pdf(df, spec, ["A", "B"])
    assert PdfReader(io.BytesIO(out)).get_num_pages() == 2


def test_the_exported_page_contains_ink_not_just_axes(
    chrome_or_skip, tmp_path
) -> None:
    """The regression pin for the silent-blank-export failure.

    NOT marked ``slow``. ``addopts`` carries ``-m 'not slow'``, so a slow
    marker here would deselect the only defence against a blank export on
    every default run -- a green suite over 23 empty pages.

    kaleido renders Scattergl as blank axes with exit code 0 and no
    warning, so a test that checks the file exists passes against an empty
    PDF. Measured separation: 289 dark pixels for a blank page against
    36,608 for a rendered one, so 2,000 is nowhere near either boundary.
    """
    import io

    import io

    from PIL import Image as PILImage

    import kaleido

    from phenotypic.gui.results_viewer._scatter_tab._facets import plan_facets
    from phenotypic.gui.results_viewer._scatter_tab._figure import (
        build_scatter_figure,
    )

    df = _frame()
    spec = FigureSpec(x_col="x", y_col="y", section_col="s")
    fig = build_scatter_figure(df, spec, plan_facets(df, spec), for_export=True)
    fig.update_layout(width=800, height=600, paper_bgcolor="white",
                      plot_bgcolor="white", showlegend=False)

    png = tmp_path / "page.png"
    kaleido.write_fig_sync(fig, png)   # format follows the extension
    gray = np.asarray(PILImage.open(png).convert("L"))

    # Measured separation, from spec 11.1: a blank page is 289 dark pixels
    # (axis furniture only) and a rendered one is 36,608. 2,000 sits far
    # from both, so this is not a delicate threshold.
    assert (gray < 128).sum() > 2000, (
        f"exported page has no ink beyond axes: {(gray < 128).sum()} dark px"
    )
```

- [ ] **Step 2b: Install Chrome for kaleido, once**

```bash
uv run plotly_get_chrome
```

Verified absent on this machine: no `google-chrome`, `chromium` or
`chromium-browser` on `PATH`, no `~/.cache/kaleido`, `BROWSER_PATH` unset.
Without it every kaleido call raises `RuntimeError: Kaleido requires Google
Chrome`. Ask the orchestrator to run this — it is a ~150 MB download and
may need approval. If HPCC egress refuses it, stop and report rather than
falling back to the Playwright-vendored browser: that couples PDF export to
the e2e browser cache, which was considered and rejected in spec §16.1.

- [ ] **Step 3: Run it to verify it fails**

```bash
uv run pytest tests/unit/gui/results_viewer/test_scatter_pdf.py -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 4: Implement `_pdf.py`**

```python
"""Render each section to a PDF page and merge them into one document."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import polars as pl

from phenotypic.gui.results_viewer._scatter_tab._facets import plan_facets
from phenotypic.gui.results_viewer._scatter_tab._figure import build_scatter_figure
from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec

_CHROME_HINT = (
    "PDF export needs Chrome for kaleido. Install it once with "
    "`uv run plotly_get_chrome`, then retry."
)


def export_sections_pdf(
    df: pl.DataFrame,
    spec: FigureSpec,
    sections: list[str],
    *,
    width_in: int = 16,
    height_in: int = 12,
) -> bytes:
    """Render one page per section and merge them.

    Every page is built with ``for_export=True`` so its traces are SVG
    ``go.Scatter``: kaleido renders ``Scattergl`` as blank axes, silently.

    Args:
        df: The plottable frame across all sections.
        spec: The figure configuration.
        sections: Section values, in page order.
        width_in: Page width in inches.
        height_in: Page height in inches.

    Returns:
        The merged PDF as bytes.

    Raises:
        RuntimeError: If kaleido cannot find Chrome.
    """
    import kaleido
    from pypdf import PdfWriter

    writer = PdfWriter()
    with tempfile.TemporaryDirectory() as tmp:
        for n, value in enumerate(sections):
            page_df = df
            if spec.section_col and spec.section_col in df.columns:
                page_df = df.filter(
                    pl.col(spec.section_col).cast(pl.String) == str(value)
                )
            fig = build_scatter_figure(
                page_df, spec, plan_facets(page_df, spec), for_export=True
            )
            fig.update_layout(
                title=dict(
                    text=f"{spec.section_col}: {value}"
                    if spec.section_col else str(value),
                    font=dict(size=spec.sizes["section"]),
                ),
                width=width_in * 100,
                height=height_in * 100,
            )
            out = Path(tmp) / f"page_{n:04d}.pdf"
            try:
                kaleido.write_fig_sync(fig, out)
            except RuntimeError as exc:  # kaleido's missing-Chrome error
                if "chrome" in str(exc).lower():
                    raise RuntimeError(_CHROME_HINT) from exc
                raise
            writer.append(str(out))

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()
```

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/unit/gui/results_viewer/test_scatter_pdf.py -v
```

Expected: both pass once Chrome is installed (Step 2b), and both **skip
loudly** without it rather than failing. Neither is marked `slow`: `addopts`
carries `-m 'not slow'`, so a slow marker would deselect the ink assertion on
every default run — a green suite over a blank export.

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/_scatter_tab/_pdf.py tests/unit/gui/results_viewer/test_scatter_pdf.py
git add <exactly the files this task's **Files:** block names>   # never -A
git commit -m "feat(gui): export Scatter sections as a multi-page PDF

One kaleido page per section, merged with pypdf. The ink assertion is the
only defence against kaleido's silent blank render."
```

---

## Task 11: Click resolution against `master_df`

Spec §6, §6.1.

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_scatter_tab/_inspector.py`
- Test: `tests/unit/gui/results_viewer/test_scatter_click.py`

**Interfaces:**
- Consumes: `CUSTOMDATA_COL` (Task 9).
- Produces:
  - `ColonyRef` — frozen dataclass `dataset: str`, `stem: str`, `label: int`
  - `index_frame(master_df: pl.DataFrame) -> pl.DataFrame` — **the producer**
  - `resolve_click(master_df, index, fingerprint, expected_fingerprint) -> ColonyRef | None` — **the consumer**

Both sides of the index contract live in this one module on purpose. Gate 0
found the plan building the index against the *filtered* frame in Task 13 while
resolving it against `master_df` here — every click would have opened a real but
wrong colony, silently. The producer and the consumer were in different tasks in
different clusters, which is exactly how that survived review. Co-locating them
removes the bug class rather than this instance: Task 13 must call
`index_frame`, never `with_row_index` itself.

- [ ] **Step 1: Write the failing test**

```python
"""A click index resolves against master_df, and a stale one is refused."""
from __future__ import annotations

import polars as pl

from phenotypic.gui.results_viewer._scatter_tab._inspector import (
    ColonyRef,
    resolve_click,
)


def _master() -> pl.DataFrame:
    return pl.DataFrame({
        "Metadata_Dataset": ["ds", "ds", "ds"],
        "Metadata_ImageName": ["a", "a", "b"],
        "Object_Label": [1, 2, 1],
    })


def test_an_index_resolves_to_its_colony() -> None:
    assert resolve_click(_master(), 1, "fp", "fp") == ColonyRef("ds", "a", 2)


def test_a_stale_fingerprint_is_refused_not_resolved() -> None:
    """The race this prevents: the user changes a filter, clicks the
    still-rendered old figure, and the index resolves against a new frame.
    It would open a real colony -- the wrong one -- silently."""
    assert resolve_click(_master(), 1, "old", "new") is None


def test_a_phantom_row_is_refused_not_crashed_on() -> None:
    """master_df is the mirror, so it carries metadata-only phantoms.

    ``int(None)`` raises TypeError and 500s the callback; the point is that
    a phantom has no colony to open, so the answer is None.
    """
    master = pl.DataFrame({
        "Metadata_Dataset": ["ds"],
        "Metadata_ImageName": ["a"],
        "Object_Label": [None],
    })
    assert resolve_click(master, 0, "fp", "fp") is None


def test_an_out_of_range_index_is_refused() -> None:
    assert resolve_click(_master(), 99, "fp", "fp") is None
    assert resolve_click(_master(), -1, "fp", "fp") is None


def test_an_index_survives_filtering_and_sorting() -> None:
    # The filter MUST drop rows and the sort MUST reorder them. Measured on
    # the fixture: with all 844 rows kept, the carried index equals the row
    # position exactly -- the coincidence B1 hid behind -- so a round-trip
    # test over an unfiltered frame passes against the defect it names.
    """The round-trip Gate 0 found missing, and the bug it would have caught.

    The index must be assigned to master_df BEFORE any filtering, so that a
    point drawn from a filtered, re-sorted frame still addresses the right
    row of the unfiltered one. Indexing the filtered frame instead passes
    every other test in this module -- none of them exercise the producer --
    and opens the wrong colony in production.
    """
    from phenotypic.gui.results_viewer._scatter_tab._inspector import index_frame
    from phenotypic.gui.results_viewer._scatter_tab._figure import CUSTOMDATA_COL

    master = _master()
    indexed = index_frame(master)

    # Whatever the figure does downstream -- filter, sort, drop rows --
    # the carried index must still resolve to the same colony.
    scrambled = (
        indexed.filter(pl.col("Metadata_ImageName") != "a")
        .sort("Object_Label", descending=True)
    )
    for row in scrambled.iter_rows(named=True):
        idx = int(row[CUSTOMDATA_COL])
        assert resolve_click(master, idx, "fp", "fp") == ColonyRef(
            row["Metadata_Dataset"], row["Metadata_ImageName"],
            int(row["Object_Label"]),
        )
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/gui/results_viewer/test_scatter_click.py -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the resolver**

```python
"""Resolve a clicked point back to the colony it represents."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from phenotypic.gui.results_viewer._filtered_state import (
    KEY_DATASET,
    KEY_IMAGE_FILE,
    KEY_OBJECT_LABEL,
)


@dataclass(frozen=True)
class ColonyRef:
    """The key the crop route, the Viv stage and curation all take."""

    dataset: str
    stem: str
    label: int


def index_frame(master_df: pl.DataFrame) -> pl.DataFrame:
    """Stamp each row with its positional index into ``master_df``.

    **Call this before filtering, never after.** The index this writes is
    positional into the frame passed in, and :func:`resolve_click` reads it
    positionally out of the same frame. Indexing a filtered frame instead
    produces indices that address the wrong rows of ``master_df`` -- and
    because every one of them is a real colony with a real crop, nothing
    errors and the inspector simply shows the wrong colony.

    ``FilterSpec.apply_to`` (``_filter_state.py:230``) is **not** a bare
    ``.filter()`` -- it calls ``normalize_viewer_frame`` first, which renames
    every non-metadata column to a shield name, coalesces, and renames back,
    so this index makes that round trip on every filter application.
    Measured on the real mirror rather than assumed: the column survives
    ``apply_to`` as ``UInt64``, 844 rows in and out, and **0** rows resolve
    to a different colony. ``normalize_viewer_frame`` alone also preserves
    both the column and its values.

    Args:
        master_df: The frozen run frame from ``OutputRoot``.

    Returns:
        ``master_df`` with a ``CUSTOMDATA_COL`` index column prepended.
    """
    from phenotypic.gui.results_viewer._scatter_tab._figure import CUSTOMDATA_COL

    return master_df.with_row_index(CUSTOMDATA_COL)


def resolve_click(
    master_df: pl.DataFrame,
    index: int,
    fingerprint: str,
    expected_fingerprint: str,
) -> ColonyRef | None:
    """Resolve a point's row index into a colony, or refuse it.

    The index is positional into ``master_df``, which ``OutputRoot`` freezes
    at ``discover()`` -- not into the filtered frame, which is re-derived on
    every filter and sort change. A positional index into a moving frame has
    a race with no error path: a click on a stale figure resolves against
    the new frame and opens the wrong colony, silently and plausibly.

    ``master_df`` is stable within one binding but not across a refresh, and
    curation can be written while the tab is open, so the caller passes the
    fingerprint captured when the figure was drawn. A mismatch is refused.

    Args:
        master_df: The frozen run frame.
        index: Positional row index carried as the point's customdata.
        fingerprint: Snapshot fingerprint captured with the figure.
        expected_fingerprint: The binding's current fingerprint.

    Returns:
        The colony, or ``None`` when the index is stale or out of range.
    """
    if fingerprint != expected_fingerprint:
        return None
    # Not decorative. Measured: polars `row(-1)` returns the LAST row
    # Python-style, so without this a negative index silently resolves to a
    # real colony from the wrong end of the frame -- the same shape as B1.
    # `bool` subclasses `int`, so a stray True would otherwise resolve as
    # index 1; refuse it rather than resolve it.
    if isinstance(index, bool) or index < 0 or index >= master_df.height:
        return None
    row = master_df.row(index, named=True)
    label = row[KEY_OBJECT_LABEL]
    if label is None:
        # master_df is the mirror, which carries metadata-only phantoms --
        # 121 of 844 in the fixture, 117,415 of 231,229 in the full run. A
        # phantom has no colony to open, and int(None) would 500 the
        # callback. Measured: Object_Label is Int64 (not float), so a
        # phantom is a true None and `is None` is the right test -- a float
        # dtype would make it NaN, where `is None` is False and int(nan)
        # raises ValueError instead.
        return None
    return ColonyRef(
        dataset=str(row[KEY_DATASET]),
        stem=str(row[KEY_IMAGE_FILE]),
        label=int(label),
    )
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/gui/results_viewer/test_scatter_click.py -v
```

Expected: 3 passed. If the `KEY_*` imports fail, confirm the names in `results_viewer/_filtered_state.py` — that module is constants-only and is the correct source.

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/_scatter_tab/_inspector.py tests/unit/gui/results_viewer/test_scatter_click.py
git add <exactly the files this task's **Files:** block names>   # never -A
git commit -m "feat(gui): resolve Scatter clicks against master_df

Anchors to the frozen frame rather than the filtered one, and refuses a
stale index rather than mis-resolving it."
```

---

## Task 12: Generalize the splitter and mount the tab

Spec §7.

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_assets/results_viewer.js` — section F
- Create: `src/phenotypic/gui/results_viewer/_scatter_tab/_ids.py`, `_layout.py`
- Modify: `src/phenotypic/gui/results_viewer/_ids.py`, `_layout.py`
- Test: `tests/unit/gui/results_viewer/test_layout_tab_shape.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `TAB_SCATTER_ID = "tab-scatter"`; `build_scatter_tab_body(output_root) -> Component`

- [ ] **Step 1: Write the failing test**

Extend `tests/unit/gui/results_viewer/test_layout_tab_shape.py`:

The module already has a `_tab_ids(layout)` helper (`:13`) and a
`built_results_layout` fixture — use them, do not invent a helper.

Note there is an **existing** test, `test_results_tabs_expose_exactly_the
_mounted_surfaces` (`:33`), asserting the tab set is *exactly* Plate and
Colony. Mounting a third tab must make that test fail first — that is the
red step — then update its expected list. Do not delete it: it is what stops
Heatmap or QC being remounted by accident.

```python
def test_scatter_is_mounted_and_the_deprecated_tabs_are_not(
    built_results_layout,
) -> None:
    """Scatter joins Plate and Colony; Heatmap and QC stay unmounted."""
    from phenotypic.gui.results_viewer import _ids as ids

    tab_ids = _tab_ids(built_results_layout)
    assert ids.TAB_SCATTER_ID in tab_ids
    assert ids.TAB_HEATMAP_ID not in tab_ids
    assert ids.TAB_QC_ID not in tab_ids
```

Then extend the existing exact-match test's expected list to
`(TAB_PLATE_ID, TAB_COLONY_ID, TAB_SCATTER_ID)`.

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/gui/results_viewer/test_layout_tab_shape.py -v
```

Expected: FAIL — `AttributeError: TAB_SCATTER_ID`. If `_tab_ids_in_layout` does not exist, write the equivalent walk over `build_app_layout(...)` in the test file first.

- [ ] **Step 3: Generalize the splitter**

In section F of `results_viewer.js`, replace the four hard-coded identifiers with a data-attribute contract, mirroring section (F)'s tile-bridge `BRIDGES` table, which already
parameterizes two surfaces in this same file (`timeline.js` was deleted in
`fe74d832`; the precedent this originally cited no longer exists): the handle carries `data-splitter-target` (the id of the pane to resize) and `data-splitter-store` (the Dash store id to persist to). Keep `clampSidebarWidth` exported. Attach to every `[data-splitter-target]`, not to one id, and clear the poll once attached so an unmounted surface does not leave a timer running for the session.

- [ ] **Step 4: Add the id and mount the tab**

Add `TAB_SCATTER_ID = "tab-scatter"` to `results_viewer/_ids.py` and its `__all__`, then add a third `dbc.Tab` in `_layout.py`'s `dbc.Tabs` block (currently lines 560–577), after Colony.

- [ ] **Step 5: Run the layout tests**

```bash
uv run pytest tests/unit/gui/results_viewer/test_layout_tab_shape.py \
             tests/unit/gui/results_viewer/test_callback_output_ids.py -v
```
Two named files — inline is correct here.

Expected: pass.

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/ tests/unit/gui/results_viewer/
git add <exactly the files this task's **Files:** block names>   # never -A
git commit -m "feat(gui): mount the Scatter tab and generalize the splitter"
```

---

## Task 13: Callbacks, inspector chrome and the crop route

Spec §7, §9, §16.4.

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_scatter_tab/_callbacks.py`
- Modify: `_scatter_tab/_layout.py`, `_scatter_tab/__init__.py`, `results_viewer/_app.py`
- Test: `tests/unit/gui/results_viewer/test_scatter_callbacks.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `register_callbacks(app, output_root) -> None`; `build_scatter_tab_body(output_root) -> Component`

- [ ] **Step 1: Write the failing test**

```python
"""Callback registration and the tab's refresh contract."""
from __future__ import annotations

import dash

from phenotypic.gui.results_viewer import _ids as ids


def test_scatter_subscribes_to_the_shared_refresh_revision(dash_app_and_root) -> None:
    """One Refresh must move every surface together.

    A Scatter-local refresh button would let the tab disagree with Plate
    and Colony about which snapshot it is showing.
    """
    from phenotypic.gui.results_viewer._scatter_tab import register_callbacks

    app, output_root = dash_app_and_root
    register_callbacks(app, output_root)

    # Dash 4.1 stores inputs as plain dicts keyed "id"/"property", not as
    # objects -- the repo's working example is test_filter_panel.py:216-219.
    inputs = {
        spec["id"]
        for entry in app.callback_map.values()
        for spec in entry["inputs"]
    }
    assert ids.STORE_PLOT_REFRESH_REVISION in inputs
    assert ids.STORE_FILTER_SPEC in inputs, (
        "the filter store must be an Input, not a State -- decision Q4 is "
        "that Scatter shares filters, so the figure must rebuild on a filter "
        "edit rather than going stale"
    )
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/gui/results_viewer/test_scatter_callbacks.py -v
```

Expected: FAIL — no `register_callbacks`. Reuse the app/output-root fixture from `tests/unit/gui/results_viewer/conftest.py`; add `dash_app_and_root` there if it is absent.

- [ ] **Step 3: Implement the callbacks**

Six callbacks. The figure callback is the one that must be right — note that
`STORE_PLOT_REFRESH_REVISION` is an Input, not a State, so a Refresh rebuilds
the plan and the figure rather than reusing a cached one.

```python
def register_callbacks(app: dash.Dash, output_root: OutputRoot) -> None:
    """Register every Scatter callback. Called once from ``_app.py``."""

    @app.callback(
        Output(ids.SCATTER_GRAPH, "figure"),
        Output(ids.SCATTER_PAGER_LABEL, "children"),
        Output(ids.STORE_SCATTER_FINGERPRINT, "data"),
        Input(ids.SCATTER_SECTION_COL, "value"),
        Input(ids.SCATTER_ROW_COL, "value"),
        Input(ids.SCATTER_COL_COL, "value"),
        Input(ids.SCATTER_X_COL, "value"),
        Input(ids.SCATTER_Y_COL, "value"),
        Input(ids.SCATTER_HUE_COL, "value"),
        Input(ids.SCATTER_SHAPE_COL, "value"),
        Input(ids.SCATTER_SHOW_REMOVED, "value"),
        Input(ids.STORE_SCATTER_SECTION_INDEX, "data"),
        Input(rv_ids.STORE_PLOT_REFRESH_REVISION, "data"),
        # Input, NOT State: decision Q4 shares filters, so a filter edit must
        # rebuild the figure rather than leave it stale.
        Input(rv_ids.STORE_FILTER_SPEC, "data"),
    )
    def _render(section_col, row_col, col_col, x_col, y_col, hue, shape,
                show_removed, section_index, _revision, filter_payload):
        # Index BEFORE filtering. resolve_click reads this positionally out
        # of master_df, so an index stamped on the filtered frame would
        # address the wrong row -- and every wrong row is a real colony, so
        # nothing would error. index_frame is the only sanctioned producer.
        base = index_frame(output_root.master_df)
        frame = plottable(FilterSpec.from_store(filter_payload).apply_to(base))
        if x_col == COMPUTED_FRAME_INDEX:
            frame = derive_frame_index(frame)
        frame = frame.drop_nulls(subset=[c for c in (x_col, y_col) if c])

        spec = FigureSpec(
            x_col=x_col, y_col=y_col, section_col=section_col,
            row_col=row_col, col_col=col_col, hue_col=hue, shape_col=shape,
            show_removed=bool(show_removed),
        )
        sections = (
            sort_facet_values(
                frame[section_col].drop_nulls().unique().cast(pl.String).to_list()
            ) if section_col else [""]
        )
        # A live run can retire the section we were on; fall back rather than
        # render an empty page.
        idx = min(int(section_index or 0), max(len(sections) - 1, 0))
        current = sections[idx] if sections else ""
        page = (
            frame.filter(pl.col(section_col).cast(pl.String) == current)
            if section_col and current else frame
        )
        plan = plan_facets(page, spec)
        fig = build_scatter_figure(page, spec, plan)
        label = f"{current}  ({idx + 1} / {len(sections)})"
        if plan.truncated:
            label += f" — showing first {len(plan.rows) * len(plan.cols)} of {plan.total} facets"
        return fig, label, output_root.snapshot.consumed_state_fingerprint
```

### 13.0 Required: prove the id contract in both directions

Task 12 declares ids and mounts components; Task 13 binds callbacks to them.
Each half can be correct alone while the pair disagrees — B1's shape, and the
reason the post-G gate exists.

Task 12 already proves one direction with
`test_every_declared_id_is_actually_mounted`: every id in `_scatter_tab/_ids.__all__`
is present in the built body. **Task 13 must prove the other**: every id its
callbacks bind is one Task 12 declares *and mounts*.

Why neither rename is the failure to look for, established by cluster F rather
than assumed:

- renaming an id's **string value** stays in sync, because `_layout.py` and the
  tests both reference the symbol;
- renaming the **symbol** raises `AttributeError` at import — loud;
- **binding an `Input` to an id nothing mounts** is the silent one. Depending on
  `suppress_callback_exceptions` you get a registration error or, worse, a
  callback that simply never fires.

Walk `app.callback_map`, collect every input and output id, and assert the set
is a subset of the mounted ids. Dash 4.1 stores those as plain dicts keyed
`"id"` — `{spec["id"] for entry in app.callback_map.values() for spec in entry["inputs"]}`
(`test_filter_panel.py:215-219`); `dep.component_id` raises.

### 13.1 The curation join — `show_removed`'s producer half

`show_removed` was wired to nothing in the original plan: Task 13 set it and
no task read it. The user chose the **in-memory join**, cluster D built the
render half in `_figure.py`, and this is the producer. Without it the toggle
renders nothing again, in the opposite direction.

**Contract, fixed by `_figure.py` — do not redesign it:**

- Column name `REMOVED_COL` = `"_scatter_removed"`. **Import the constant from
  `_figure.py`; never re-spell the string.**
- **Boolean dtype. Null is read as False.** A non-Boolean column is treated as
  *absent*, not coerced — the column is produced here, in memory, so a wrong
  dtype is a defect in this callback and guessing at it would hide colonies on
  the strength of a guess.
- Absent column ⇒ `_figure.py` behaves exactly as if nothing were removed. So
  a run with no curation state needs no special case here.

**Source:** `STORE_REMOVED_KEYS` (`_ids.py:808`), the Dash store the curation
surfaces already write. `_viewer_card.py` consumes it as an `Input` — follow
that precedent. Join it onto the indexed frame as a Boolean column.

**Ordering, and why it matters:** join curation **after** `index_frame` and
**before** `plottable`. Indexing first keeps every carried index
master-anchored (§6.1); joining before the phantom filter means a removed
*phantom* is dropped by `plottable` rather than drawn as a grey x for a colony
that does not exist.

**Test it against a mutation:** dropping the join must fail a test asserting
the grey series appears for a frame whose keys are in the store. A join that
silently produces an all-False column is the failure to guard — it renders
identically to "nothing is removed", which is exactly how the toggle read
before this section existed.

The remaining five are mechanical and follow the same shape:

1. **Pager** — `Input(PREV, "n_clicks")` / `Input(NEXT, "n_clicks")` →
   `Output(STORE_SCATTER_SECTION_INDEX, "data")`, clamped to the section count.
2. **Click** — `Input(SCATTER_GRAPH, "clickData")` +
   `State(STORE_SCATTER_FINGERPRINT, "data")` → `resolve_click(...)` →
   offcanvas `is_open`, header text, crop `src`, and the grouped
   measurement rows from `group_columns`.
3. **Contours** — the segmented control toggles the `?contours=` value in the
   crop `src`, so it re-requests rather than re-resolving the click.
4. **Legend** — corner and collapsed state into a store the layout reads.
5. **Export** — `Input(EXPORT_BTN, "n_clicks")` → `export_sections_pdf(...)` →
   `dcc.send_bytes(lambda b: b.write(pdf), "scatter.pdf")`.

- [ ] **Step 4: Mount the crop route**

In `results_viewer/_app.py`, beside the existing two crop-route registrations:

```python
    register_crop_route(
        app, output_root, SCATTER_CROPS_URL_SEGMENT, default_contours=1
    )
```

- [ ] **Step 5: Run the full results-viewer suite**

```bash
sbatch --export=ALL,SCOPE=gui docs/superpowers/plans/2026-09-01-results-scatter-tab/run_scatter_tests.sbatch
```

Expected: every array task exits 0. Ask the orchestrator to submit and to
report the per-task exit statuses from the logs.

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/
uv run mypy src/phenotypic/gui/results_viewer/_scatter_tab/
git add <exactly the files this task's **Files:** block names>   # never -A
git commit -m "feat(gui): wire the Scatter tab's callbacks and crop route"
```

---

## Task 14: Ledgers, tutorial capture and a fixture smoke run

Spec §13.

**Files:**
- Modify: `FEATURES.md`, `WORKFLOWS.md`, the tutorial capture script, `docs/source/.../19_scatter.md`

**Interfaces:**
- Consumes: the mounted tab.
- Produces: nothing.

- [ ] **Step 1: Read the skill**

```bash
cat ~/.claude/skills/gui-tutorial-capture/SKILL.md
```

Both ledgers are CI-gated; `check_workflows_md.py` requires a `_capture_<id>` function that is defined **and** dispatched, a non-empty `docs/source/_static/gui_images/<id>/`, and a tutorial page. The highest existing page is `18_browse.md`, so this is `19`.

- [ ] **Step 2: Add the ledger rows**

Add a Scatter row under the `## Results Viewer integration` heading in `FEATURES.md`, and a workflow entry in `WORKFLOWS.md` with a matching capture id.

- [ ] **Step 3: Add the capture function and tutorial page**

Define `_capture_scatter` and dispatch it from `capture_workflow_screenshots`. Write `19_scatter.md`.

- [ ] **Step 4: Run the gates**

```bash
uv run python scripts/check_features_md.py
uv run python scripts/check_workflows_md.py
```

Expected: both pass. If the script paths differ, find them with `ls scripts/ | grep -i check`.

- [ ] **Step 5: Smoke-run against the fixture**

```bash
uv run phenotypic-gui --root /rhome/anguy344/bigdata_exfab/projects/ucr_029_e_d_Maresca/data/results/2026-08-11-migration-test --port 8051
```

Open the Scatter tab. Confirm: 723 of 844 rows reported plottable; the section pager shows **22** strains (NOT 23 -- 82 rows have a null `Metadata_Strain`, and a null is dropped rather than becoming its own page; if you see 23, the null-drop regressed); a facet grid renders; clicking a point opens the inspector with a crop that looks like a colony, not noise; Contours/Raw toggles; the splitter drags; Export PDF produces a file whose pages are not blank.

- [ ] **Step 6: Commit**

```bash
# Explicit paths only -- this worktree is shared and other clusters have
# files in flight. `git add -A` would sweep them into your commit.
git add FEATURES.md WORKFLOWS.md \
        docs/source/tutorials/19_scatter.md \
        <the capture script you edited> \
        docs/source/_static/gui_images/scatter/
git commit -m "docs(gui): add Scatter to the feature ledgers and tutorial"
```

---

## Task 15: Full suite and branch check

**Files:**
- Modify: none — verification only.

**Interfaces:**
- Consumes: the complete branch.
- Produces: nothing.

- [ ] **Step 1: Run the unit suite as a Slurm job**

The suite is ~65 minutes, not two, and `-n auto` reads the node's core count rather than the allocation's, manufacturing timeout failures. Use the committed batch script:

```bash
sbatch --export=ALL,SCOPE=full \
       docs/superpowers/plans/2026-09-01-results-scatter-tab/run_scatter_tests.sbatch
```

16 tasks x 8 CPUs = 128 CPU peak, sized against the 384-CPU account cap with
an interactive session already holding ~170. Check headroom before raising
either number:
`squeue -u "$USER" -t RUNNING -h -o "%C" | paste -sd+ | bc`

Follow the `slurm-job` skill to verify it actually starts (`scontrol show job <id> | grep -E 'StartTime|Reason'`).

- [ ] **Step 2: Compare against the known-failure baseline**

Four pre-existing failures are expected, three of which fail only on compute nodes. Any *new* failure is this branch's.

- [ ] **Step 3: Type-check the changed tree**

```bash
uv run mypy src/phenotypic/gui
```

- [ ] **Step 4: Verify the validation script still reproduces**

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-09-01-results-scatter-tab/crop_uint16_scaling.py
```

Expected: exit 0, all claims reproduced.

---

## Execution Orchestration

Derived from each task's `Files` / `Interfaces` blocks. Version-controlled beside
the plan because it is a view of the plan, not a separate artefact.

### Command protocol (applies to every agent)

Subagents in this session **must not run shell commands themselves**. Approval
prompts raised by a subagent do not surface in the user's Remote Control app, so
a command that auto-mode declines to auto-approve stalls invisibly; the
orchestrator's prompts do surface. Every agent is briefed to:

- do its own file reading and editing (Read / Edit / Write / Grep / Glob);
- send **batched** shell commands to the orchestrator via
  `SendMessage({to: "main", ...})` and wait for the output;
- never `git commit` itself — hand the orchestrator the exact message;
- stop and ask rather than run anything directly.

**Spikes round-trip too — and spikes are encouraged.** Ad-hoc investigation is
where this project's most valuable findings came from: rendering a figure through
kaleido and counting ink pixels proved the export path was silently blank;
counting WebGL canvases in a real browser withdrew a whole contingency from the
spec. Reviewers and implementers should keep writing them. The flow is:

1. **Write** the script to the session scratchpad — file writes need no approval.
2. Send `RUN: uv run python <path>` and wait.
3. The orchestrator executes from the worktree root and returns stdout+stderr
   verbatim; iterate by editing and asking for a re-run.

The same applies to a targeted `pytest -k`, a `python -c` one-liner, `git show`,
`ruff` and `mypy`. A lone spike run need not wait to fill a batch — do not stall
an investigation for the sake of the protocol.

**Large output goes to a file, not a message.** An agent expecting a big result
says so and the orchestrator writes it to a path the agent can `Read`. Message
payloads truncate, which cost this project a complete review report once already.

This trades round-trips for the user's ability to approve from a phone. Agents
batch 3–6 commands per request to keep the trade cheap.

### Dependency DAG

```
1 ─→ 2                     (tiles.py, then its fixture pin)
1 ─→ 3                     (contours build on the scaled rgb path)
4                          (independent — shared helper)
5 ─→ 6 ─→ 8                (spec → facets → frame index; 6 and 8 share _facets.py)
5 ─→ 9                     
6 ─→ 9 ─→ 10               (figure → pdf)
     9 ─→ 11               (figure → click index)
7                          (independent — grouping)
12 ─→ 13 ─→ 14 ─→ 15       (mount → wire → ledgers → suite)
5,6,7,8,9,10,11 ─→ 13
```

**Shared files** (the parallelism constraint): `tiles.py` (1, 3);
`test_tiles_zarr.py` (1, 2, 3); `_config.py` (3, 6); `_facets.py` (6, 8);
`_scatter_tab/_ids.py` + `_layout.py` (12, 13).

### Clusters

| # | Tasks | Shape | Model / effort | Runs |
|---|---|---|---|---|
| A | 1, 2, 3 | Keystone — one intent (fix the crop path), one file pair | Opus / high | parallel with B |
| B | 4 | Sweep — mechanical, but changes Colony's axis options | Opus / medium | parallel with A |
| C | 5, 6, 7, 8 | Keystone — the pure, Dash-free data layer | Opus / high | after A |
| D | 9, 10 | Keystone — the render pipeline | Opus / high | after C |
| E | 11 | Seam — the silent-wrong-colony hazard; owns BOTH sides of the index contract | Opus / high | after D |
| F | 12 | Seam — shared JS + layout mount | Opus / high | after E |
| G | 13 | Seam — the largest integration point | Opus / high | after F |
| H | 14 | Sweep — CI-gated ledgers | Sonnet / medium | after G |
| — | 15 | Verification — orchestrator runs it (Slurm) | — | last |

**Why A and B parallelise:** disjoint file sets (`tiles.py` + its test versus
`colony_view/_grid.py` + a new test). A owns *all three* `_config.py` constants,
including the two Task 6 needs, so C never touches that file and the seam stays
in one cluster.

**Why E, F and G are alone despite being small:** each is a single risky wiring
point. Risk is not size. E resolves a click to a colony and a mistake opens the
*wrong* colony plausibly and silently; F edits JS two other surfaces depend on;
G is where every prior cluster meets.

**Gate 0 changed E's boundary.** The plan originally put the index *producer* in
Task 13 (cluster G) and the *consumer* in Task 11 (cluster E), and they disagreed:
the producer indexed the filtered frame, the consumer read `master_df`
positionally. Every click would have opened a real but wrong colony, with nothing
raising. Splitting a contract across two clusters is how that survived a careful
read, so E now owns both halves — `index_frame` and `resolve_click` live in one
module, and G is briefed that calling `with_row_index` itself is a defect.

### Gate 0 findings (resolved 2026-09-02)

Ten defects, five from reading and five from spikes the reviewer wrote and
the orchestrator ran. All are fixed above; recorded here because several
would have been invisible until runtime.

| # | Defect | How it would have shown up |
|---|---|---|
| B1 | Click index built on the filtered frame, resolved against `master_df` | Every click opens a real but **wrong** colony. Nothing raises |
| B2 | `apply_filters` and `STORE_FILTER_STATE` invented | ImportError at first run |
| B3 | `callback_map` inputs are dicts, not objects | Test raises before *and* after — gates nothing |
| B4 | "Empty facet" test planned a 1x1 grid with no empty facet | Vacuous |
| B5 | Display-range test asserted `f(x) == f(x)` | Vacuous; the signature cannot express the property |
| D1 | `_read_store_level` takes a **layer**, not a member path | `rgb` works by accident; `objmap` raises; `detect_mat`/`gray` return `(0, 0)` silently |
| C1 | Shared-axes test asserted `<= 1`; `share_axes=False` yields **0** | Cannot detect its own regression |
| C2 | `showlegend=first_cell` drops any series absent from cell (0,0) | On a sparse frame, points drawn with no legend entry |
| C3 | `plottable` casts the curation flag to Boolean | A string-typed flag 500s the tab. **The Gate 0 fix (`strict=False`) was itself wrong** — polars 1.41 rejects `Utf8 -> Boolean` under both strict modes, so the cast must be avoided rather than softened. Found by cluster C spiking rather than assuming |
| E1 | `gui/_assets/results_viewer.js` does not exist | Wrong path; the file is under `gui/results_viewer/_assets/` |
| B6 | Export tests had no Chrome guard | Hard-fail with kaleido's RuntimeError, reading as a broken export rather than a missing prerequisite |
| B7 | The ink assertion was `@pytest.mark.slow` **and** needed an absent `pymupdf` | `addopts` carries `-m 'not slow'`, so §15's "only defence" against a blank PDF ran nowhere |
| F1 | Four of Task 7's five tests passed against a class-not-instance bug | Reported coverage that did not exist — found by mutation, not by reading |
| F2 | `resolve_click` did `int(None)` on a phantom row | `master_df` is the mirror; a phantom click 500s the callback |
| F3 | Task 9's index test asserted membership in a range, not correspondence to a row | **This is why B1 survived.** A filtered-frame index satisfies it perfectly while pointing at the wrong row |
| F4 | `image_display_range` scanned the store directory for level names | Violates `gui/CLAUDE.md:58`, "never recompute the pyramid" |

`GATE0.md`'s vacuity table marks Task 7's tests sound — that column is a
hand-trace, and `spike_f`'s mutation run overrides it. Read the outcomes here.

Two spec figures were also corrected by the spikes: level-0's true `rgb`
range is **(17290, 47898)**, not the (17912, 45344) §2.3(c) quotes, so the
`rgb/4` under-coverage is worse than stated — about 11% low and 8% high.
And `image_display_range` now falls back on a degenerate level rather than
returning a zero-width range.

Confirmed sound and needing no change: the schema-prefix derivation (no
`category()` raises; not import-order dependent), `derive_frame_index` (all
three tests pass; left-join preserves order at n=2000), and Task 7's
grouping, which reproduces 17/15/12/8/65/16/1/15 = 149 of 149 against the
real fixture.

### Gates

- **Gate 0 — before any implementation:** `plan-reviewer` over this plan.
  **Run 2026-09-02; ten defects found and fixed. See the table above.**
- **After each cluster (light):** orchestrator reads the diff, runs tests and
  lint, and pauses to surface any design question the review raises.
- **After D and after G (deep):** `implementation-test-reviewer` over the
  combined diff — it checks the new tests can actually *fail*, which matters most
  for the export ink assertion and the monotonic-ramp pin.
- **After H (simplify):** one quality-only pass, then re-run affected tests.

---

## Follow-ups — found by this change, deliberately NOT done in it

All four are in `src/phenotypic/gui/_shared/tiles.py` unless noted, all
pre-existing rather than introduced here, and none is a behaviour bug. They are
listed in the order worth doing, not the order found.

1. **A docstring guarantees a mechanism that is not in its call path.**
   `crop_store_rgb` says *"Geometry is byte-identical because both croppers share
   `_crop_pil_source`."* It does not share it — `crop_store_rgb` reaches
   `_crop_store_layer_window`, which re-implements the geometry. The outcome is
   true today only because the two copies happen to agree, while the sentence
   tells the next reader they *cannot* diverge. That is worse than silence, and
   it is why (2) is worth fixing rather than tolerating.

2. **The spotlight-dim block is duplicated verbatim** between `_crop_pil_source`
   and `_crop_store_layer_window` — 11 lines each (unpack `bbox`, four `_clamp`
   calls against the same unclamped origin, `_dim_outside_bbox`,
   `PILImage.fromarray`), plus a ~10-line unclamped/clamped geometry preamble.
   Change the padding in one cropper and the crops silently diverge.

3. **A hex→RGB parser exists twice, decoding the same two tokens.**
   `composite_contours`'s nested `_rgb(hex_)` (`tiles.py:572`) and
   `_hex_to_rgb` (`gui/tune/_overlays.py:53`) both decode `OI_SKY` and
   `OI_ORANGE`. `_design.py` owns the tokens and is the natural home. Crosses
   into `gui/tune/`, which this branch does not touch.

4. **A dead parameter costing a syscall per crop.** `crop_store_rgb(..., mtime_ns,
   ...)` opens with `del mtime_ns` ("accepted for caller/API compatibility"), and
   its only in-repo caller `crop_colony` runs `os.stat(store).st_mtime_ns` on
   every crop to supply it. `crop_store_rgb` is in `__all__`, so removing it is a
   public API change and needs its own commit.

Also open, from the spec rather than the code: §7's Viv plate-context stage and
the inspector's prev/next point walking appear in no task and were left out
whole rather than half-built; and §16.1 still specifies `plotly_get_chrome`,
which the Playwright browser under `BROWSER_PATH` has made unnecessary —
`verify_scatter_fixture.py` now discovers it unaided.
