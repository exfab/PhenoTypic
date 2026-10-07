# Phase gate A review: plot subfolders, Tasks 1-3 (implementation and tests)

Reviewer: implementation-test-reviewer (phase gate A). Analysis only. No source, test, spec or plan file was edited.

**Scope:**
- `git diff ba725001..7cd96ea4 -- src tests`: commits e067c481 (T1), 285eec2c (T2) and 7cd96ea4 (T3).
- Checked against the spec `docs/superpowers/specs/2026-09-30-plot-subfolders/design.md`, the plan (Tasks 1-3, Global Constraints, Review Focus) and `plan-review.md`.

## Summary verdict: implementation correct, tests have holes

**Implementation.** I found no bug in the T1-T3 code:
- Every version gate goes through the readable set.
- Path splitting refuses every shape the spec lists.
- Build, keep and carry handle both layouts.
- No new nondeterministic ordering.

**Tests.** The suite has four real holes, and each was **proven by mutation**: 8 mutations of the source, all 8 green on the current tests, with a red control. In three of the holes, the property the spec or the Review Focus calls out has no test that would fail if it broke:
- `plot` stamped on page failures (Review Focus 5, the field T4 matches on);
- the kept plot folder taken from the stored path;
- adding a run to a version 1 store through the real write paths (`save2zarr`, `replace_image_tables`).

The fourth hole is the `(plot_name, key)` uniqueness rule. It is the only thing that stops two pages from writing the same file.

Each hole has a concrete killing test below. All four are cheap and should land before T4: T4's `failed_only` matching rests on hole I1.

Counts (from this file): **BLOCKER 0, IMPORTANT 4, MINOR 9.**

---

## Mutation evidence (Slurm job 29304001, run by main)

Setup:
- Scratch copy from `git archive 7cd96ea4`, with each mutation applied alone and then restored and verified.
- The worktree's fingerprint was identical before and after the run (7cd96ea4, dirty=0).
- BASELINE (unmutated, the same 5 files): 119 passed.

| ID | Mutation (src) | Tests run | Result |
|---|---|---|---|
| M1 | `_output.py:95` uniqueness on `(page.plot, key)` instead of `(page.plot_name, key)` | `test_plot_page_plot.py` | **11 passed (survived)** |
| M2 | `_store_figures.py:433,480` failures stamped `plot=page.key` | `test_store_figures_build.py`, `test_staged_figures_keep.py` | **49 passed (survived)** |
| M3 | `_store_figures.py:480` per-format failure loses `plot=` | same | **49 passed (survived)** |
| M4 | `_store_figures.py:388` refuse only "plot given, path flat" | same | **49 passed (survived)** |
| M5 | `_store_figures.py:388` drop `len(page_directories) > 1 or` | same | **49 passed (survived)** |
| M6 | `_store_figures.py:397` kept `directory=plot` instead of from the path | same | **49 passed (survived)** |
| M9 | `_measurement_tables.py:787` gate back to `!= FIGURES_SCHEMA_VERSION` | `sdk_/test_image_figures*.py`, `test_store_figures_build.py`, `test_staged_figures_keep.py` | **108 passed (survived)** |
| M11 | `_image_figures.py:389` carry gate back to `!= FIGURES_SCHEMA_VERSION` | `sdk_/test_image_figures*.py` | **59 passed (survived)** |
| C1 (control) | `_image_figures.py:556` drop `type(version) is int and` | `test_image_figures.py` | 1 failed, 41 passed: `test_a_boolean_schema_version_is_not_a_known_one` (red, as predicted) |

---

## BLOCKER

None.

---

## IMPORTANT

### I1. Failure `plot` stamping is unpinned: a failure stamped with the key passes (M2, M3)

**Where:**
- `src/phenotypic/plotting/_pipeline/_store_figures.py:433` (whole-page failure);
- `src/phenotypic/plotting/_pipeline/_store_figures.py:480` (per-format failure);
- test `tests/unit/plotting/test_store_figures_build.py:702-710`.

**Problem.** `test_a_page_failure_records_its_plot_through_the_descriptor` uses `HandBuiltPages`. Every page there is bare, so `plot_name == key`, and the test cannot tell `plot=page.plot_name` from `plot=page.key`:
- M2 (stamp the key) passes all 49 tests.
- The per-format path (`:480`) is not exercised for `plot` at all. Both `HandBuiltPages` failures are raised before the format loop (an unsupported figure type, and numpy metadata). M3 (drop `plot=` at `:480`) also passes.
- No test anywhere builds a failing page whose `plot` differs from its key. Yet that is the calibration case: key `roi_1`, plot `tiles`.

**Why it matters now.** The plan's Task 4 matches failures to pages by `(key, plot)` (`partial = [... if (f.get("page"), f.get("plot")) == (page["key"], page.get("plot"))]` and `failed_only` keyed by `(page, plot)`). If `plot` were stamped wrong, a partially failed page would lose its errors, and the manifest's `failed` would list a page that was in fact published. Review Focus 5 claims this chain is pinned. At the builder end, it is not.

**Fix: a test that kills M2 and M3.** Add it to `test_store_figures_build.py`. `serialize_store_format` is a module global in `_store_figures`, so the monkeypatch reaches the code under test.

```python
class PlottedFailures(BaseModel, PlotImage):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        from matplotlib.figure import Figure

        def fig():
            f = Figure()
            f.subplots().plot([0, 1])
            return f

        return PlotOutput(pages=(
            PlotPage(key="roi_0", plot="tiles", figure=fig()),
            PlotPage(key="roi_1", plot="tiles", figure=object()),   # whole-page failure
            PlotPage(key="roi_2", plot="tiles", figure=fig()),      # per-format failure, below
        ))


def test_failures_of_plotted_pages_record_the_plot_not_the_key(tmp_path, monkeypatch):
    from phenotypic.plotting._pipeline import _store_figures

    real = _store_figures.serialize_store_format

    def flaky(fmt, figure, *, binding_id, page_key):
        if page_key == "roi_2":
            raise OSError("disk full")
        return real(fmt, figure, binding_id=binding_id, page_key=page_key)

    monkeypatch.setattr(_store_figures, "serialize_store_format", flaky)
    stored = _build(PlottedFailures())
    assert {(f.page, f.format, f.plot) for f in stored.failed} == {
        ("roi_1", None, "tiles"), ("roi_2", "png", "tiles"),
    }
    (tmp_path / "s").mkdir()
    failed = write_image_figures(tmp_path / "s", stored)["figures"]["runs"][TEST_RUN.run_id]["failed"]
    assert {(f["page"], f["plot"]) for f in failed} == {("roi_1", "tiles"), ("roi_2", "tiles")}
```

### I2. "Kept plot folder taken from its own path" is unpinned (M6)

**Where:**
- `src/phenotypic/plotting/_pipeline/_store_figures.py:397`;
- tests `test_store_figures_build.py:644-653` and `:454-465`.

**Problem.**
- The spec §3 table says `_kept_binding` rebuilds "each page with its stored plot folder taken from its own path".
- Every keep test uses plot names that clean to themselves (`tiles`, `delta_e`; `ApplyState`'s bare `tiles`), so a keeper that sets `directory=plot` (M6) is indistinguishable.
- Under M6, a kept plot named `Tile overlay`, or one that took a digest suffix (`tiles-1a2b3c4d`), would be rewritten into a **different** folder (`Tile overlay/`, `tiles/`). That breaks the byte-identity of §3a keeps and changes descriptor paths.
- M6 passed all 49 tests.

**Fix: a test that kills M6.**

```python
class SpacedPlot(BaseModel, PlotImage):
    """A plot name that does not clean to itself: folder `Tile-overlay`."""

    mode: str = "draw"

    def inspect(self, subject=None, *, for_save=False, **overrides):
        from matplotlib.figure import Figure

        if self.mode == "gone":
            raise FigureInputUnavailable("gone")
        f = Figure()
        f.subplots().plot([0, 1])
        return PlotOutput(pages=(PlotPage(key="roi_0", plot="Tile overlay", figure=f),))


def test_a_kept_page_keeps_the_folder_its_path_names_not_its_plot_name(tmp_path):
    first = _build(SpacedPlot())
    assert first.bindings[0].pages[0].directory == "Tile-overlay"
    store = figure_store(tmp_path / "first", first)
    kept = _keep(store, SpacedPlot(mode="gone"))
    assert kept == first
    assert _files(figure_store(tmp_path / "again", kept)) == _files(store)
```

Under M6, `kept.bindings[0].pages[0].directory == "Tile overlay"`, so `kept != first`.

### I3. Version 1 to version 2 is not tested through either real write path (M9, M11)

**Where:**
- `src/phenotypic/sdk_/_measurement_tables.py:787` (measure rewrite gate);
- `src/phenotypic/sdk_/_image_figures.py:389` (carry gate);
- test `tests/unit/sdk_/test_image_figures.py:337-362`.

**Problem.** Spec §5 requires: "Adding a run to a version 1 store writes version 2 and leaves the version 1 run's files and entries byte-identical". The lead's brief singles out exactly these gates as regression points. But:
- **Measure path.** No test runs `replace_image_tables` over a version 1 store. The only "other version" fixture in `test_image_figures_store.py` is `_relabel_as_newer` (version 3). Reverting the measure gate to the 0.19 form, `!= FIGURES_SCHEMA_VERSION` (M9), passes 108 tests. In production, the first `--mode measure` after an upgrade would then **silently add no figure run** to every existing store.
- **Save path.** `test_a_v1_store_gains_a_v2_run_and_keeps_its_v1_run_byte_for_byte` hand-assembles the `save2zarr` sequence (carry, apply, write) and skips the `_image_io_handler.py:1469` gate. Its version 1 store has no run equal to `exclude`.
  - So reverting the carry gate (M11) still passes: the version 1 descriptor is carried whole, as "unknown", and the end state looks the same.
  - In production, M11 carries the excluded run's old folder too. A same-day re-save after an upgrade (the AutoConvertRaw-GC reprocess case) would then leave the old flat `sym/default.plotly.json` in the store beside the new `sym/default/default.plotly.json`. No descriptor would describe it.

**Fix (a), unit level: kills M11.**
- In `test_a_v1_store_gains_a_v2_run_and_keeps_its_v1_run_byte_for_byte`, build the version 1 store from `_stored(_OTHER)` **and** `_stored()` (both runs, both stripped to the version 1 shape).
- After the carry, add:

```python
    assert list(carried["figures"]["runs"]) == [_OTHER.run_id]
    assert not (part / "figures" / _RUN.run_id).exists()   # the excluded run is never carried
```

**Fix (b), measure path: kills M9.** Add this to `tests/unit/sdk_/test_image_figures_store.py`:

```python
def _relabel_as_v1(store) -> dict:
    """The shape 0.19 wrote: schema_version 1, no `plot` on pages or failures."""
    root_path = store / "zarr.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    figures = root["attributes"]["phenotypic"]["figures"]
    figures["schema_version"] = 1
    for run in figures["runs"].values():
        for binding in run["bindings"].values():
            for page in binding["pages"]:
                page.pop("plot", None)
        for failure in run["failed"]:
            failure.pop("plot", None)
    root_path.write_text(json.dumps(root), encoding="utf-8")
    return json.loads(json.dumps(figures["runs"]))


def test_a_measure_rewrite_adds_a_v2_run_to_a_v1_store(tmp_path, plate):
    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures(b"old"))
    v1_runs = _relabel_as_v1(store)
    before = _snapshot(store, RUN)
    replace_image_tables(
        store, _tables(), objmap_target=ngff_.objmap_path("rgb"),
        figures=_figures(b"new", run=LATER),
    )
    descriptor = read_image_figures_descriptor(store)
    assert descriptor["schema_version"] == 2
    assert set(descriptor["runs"]) == {RUN.run_id, LATER.run_id}
    assert descriptor["runs"][RUN.run_id] == v1_runs[RUN.run_id]
    assert {k: d for k, (d, _i) in _snapshot(store, RUN).items()} == {
        k: d for k, (d, _i) in before.items()
    }
```

Optionally, add the save-path twin of (b): re-save `RUN` with `plate.save2zarr` over a version 1 store holding `RUN` and `LATER`, then assert that the set of files under `figures/<RUN>/` equals the descriptor's paths plus group documents.

### I4. `PlotOutput` uniqueness over `plot_name` is unpinned (M1)

**Where:**
- `src/phenotypic/abc_/plotting/_output.py:95`;
- test `tests/unit/abc_/plotting/test_plot_page_plot.py`.

**Problem.**
- A bare page `key="tiles"` is the page `(tiles, tiles)`, and it lands at `tiles/tiles.<ext>`, exactly where `PlotPage(key="tiles", plot="tiles")` lands.
- `PlotOutput` is the only guard. `plot_page_paths` gives both pages the same stem, because `unique_page_stems` sees the same key. `write_image_figures` would then **unlink the first page's file and write the second's**, leaving a descriptor that lists two pages at one path, one with a wrong sha256.
- The T1 tests cannot tell `(page.plot_name, key)` from `(page.plot, key)`. M1 passes all 11. `test_a_bare_page_and_a_plotted_page_with_one_key_coexist` uses a *different* plot, so it passes under both rules.

**Fix: a test that kills M1.**

```python
def test_a_bare_page_and_a_plotted_page_naming_one_file_are_refused():
    # Bare "tiles" is (plot "tiles", key "tiles"): both would be tiles/tiles.<ext>.
    with pytest.raises(ValueError, match="duplicate page keys"):
        PlotOutput(pages=(PlotPage(key="tiles", figure=object()),
                          PlotPage(key="tiles", plot="tiles", figure=object())))
```

---

## MINOR

1. **Review Focus 3's test proves less than its docstring** (`tests/unit/sdk_/test_image_figures.py:374-383`).
   - **What the test does.** It runs the *new* code with *old* constants, and it checks only an in-memory dict through `apply_image_figures_attributes`.
   - **What it does not reach.** It does not reach the carry or the `_image_io_handler.py:1469` gate. It also does not check that the store's files stay intact, which spec §5 requires ("leaves its figures intact").
   - **The guarantee itself holds**, by reading the code at `ba725001` (the `-` lines of the src diff):
     - `known_figures_schema` was `descriptor.get("schema_version") == ngff_.FIGURES_SCHEMA_VERSION`;
     - `read_figure_run` was `version != ngff_.FIGURES_SCHEMA_VERSION`.
     - So 0.19 treats version 2 as unknown: it carries the tree whole and adds no run, which `test_a_save_over_an_unknown_figures_schema_carries_it_untouched` already pins with version 3.
   - **Fix.** Either rename the docstring to what the test proves, or extend it under the same monkeypatch: `carry_figure_runs(v2_store, part)` returns the version 2 descriptor whole, every file under `figures/` is linked byte for byte, and `known_figures_schema` of it is `False`.

2. **The keeper's layout refusal is pinned in one direction only (M4, M5 survived)** (`_store_figures.py:388`).
   - Only "plot given, flat path" is tested (`test_a_page_whose_plot_disagrees_with_its_path_is_refused`).
   - **Untested: "no plot, foldered path".** Kill M4: build `PerRoiPages` into a store, `pages[0].pop("plot")` in the descriptor, keep, and assert `kept.bindings == ()` with `"not laid out"`.
   - **Untested: "one page over two plot folders".** Under M5 this also makes `directory=next(iter(set))` nondeterministic. Kill M5: append `dict(pages[2]["files"][0])` (`delta_e/delta_e.png`, a real file with a valid sha256) to `pages[0]["files"]`, keep, and assert refusal.

3. **The refusal test's `match="plot"` is too loose** (`test_plot_page_plot.py:21-24`). The key check's message, "plot page key must be…", also matches `"plot"`. Use `match="plot page plot"`.

4. **The "stable" assertion in `test_plot_folders_that_collide_get_distinct_names` compares the function with itself in one process** (`test_store_figures_build.py:617`).
   - Within one interpreter, set-ordered iteration repeats, so a `set()` in place of `dict.fromkeys` would not be caught by this line.
   - First-appearance order is pinned only by `folders[0] == "Tiles"`, and only for one input order.
   - **Fix.** Assert the reversed input too (`[("tiles",…), ("Tiles",…)]` gives `"tiles"`, then `"Tiles-<digest>"`). The cross-process byte identity is T7's job.

5. **`StoredFigurePage` does not enforce `(plot is None) == (directory is None)`** (`src/phenotypic/sdk_/_image_figures.py:153-170`, `:292-296`).
   - A page with `plot="tiles", directory=None` is written flat, with `"plot": "tiles"`.
   - The keeper refuses it only later, at the next re-measure, as a binding failure.
   - **Fix.** One check in `__post_init__`, or in the writer, would surface a builder bug at the write instead.

6. **The keeper accepts a binding that mixes flat and foldered pages**, and two pages of one `plot` stored in different folders. The spec states "never mixed" as a fact, not a check. Only hand-edited descriptors can produce either case. Noted; no fix needed.

7. **The GUI keys tabs by `page.key`** (`src/phenotypic/_gui/analysis/_render.py:155-160`, `dcc.Tab(value=page.key)`).
   - T1 now allows one key in two plots, which gives two tabs the same `value`.
   - Latent: calibration's keys are unique, and the spec excludes GUI work.
   - **Fix.** Use `f"{page.plot_name}/{page.key}"` as the tab value in a follow-up.

8. **No CLI-path test keeps a foldered page.** `tests/unit/cli/test_staged_figures_keep.py:57-69` seeds only a flat page (`directory=None`), so the staged Stage 3 keep is proven only for flat pages. Parametrize the seed with `plot="tiles", directory="tiles"`.

9. **For T5: a bare aggregate page's folder comes from its key**, and a `canonical_group_key` whose string value contains `/` fails `safe_path_component`. It then falls back to the folder `page`, then `page-<digest>` (`_writer.py:290-294`).
   - Before this change, the same fallback hid only a file stem. Now it names a folder.
   - Plan-review minor 3 accepted key-named folders. This is the extra case to keep in mind when T5 reviews `test_plot_meas_time_series.py`.

---

## Notes for T4 and T5 (not counted)

- **T4 depends on I1.** `failed_only` and the per-page `partial` lookup match on `(key, plot)`. Land I1's test first, or T4's manifest tests inherit the hole.
- **T4 must take the folder from `split_figure_file_path(entry["path"])`, never from `page["plot"]`.** `plot` is the logical, unsanitized name. The plan's Step 3 does this, and its `_page` fixture (`directory=plot.replace(" ", "-")`) can tell the two apart. Keep at least one copy-out test whose plot contains a space, or the I2 hole reappears in T4.
- **Between T3 and T4, the old copy-out still names files by `unique_page_stems` over `(key, label)` across the whole binding.** Two pages with one key in different plots would therefore overwrite each other in deliverables. This is expected at gate A only, and T4 removes it.
- **`plot_page_paths` already reserves `manifest.json`**, which is what T5's direct-publish directory needs. The Plotly bundle and `.failures.jsonl` live at `plots_base`. `.publication.lock` cannot collide, because `safe_path_component` strips a leading `.`.

---

## Validated (no finding)

- **Version gates.** Every gate goes through `_readable_version` (`type(version) is int and version in READABLE_FIGURES_SCHEMA_VERSIONS`):
  - `known_figures_schema`, `read_figure_run`, `carry_figure_runs` (via `known_figures_schema`), `apply_image_figures_attributes`, `_measurement_tables.py:787` and `_image_io_handler.py:1469`.
  - `grep -rn "FIGURES_SCHEMA_VERSION" src` shows the only remaining use is `_fragment`'s stamp (`_image_figures.py:493`). No `==` or `!=` gate remains. `_cli_process_single.py` uses only `latest_run_date`, which is version-agnostic by design.
  - The bool refusal is pinned (C1 went red).
- **Upgrade path.** `_fragment` stamps 2 on carried and new runs, and `apply_image_figures_attributes` takes `incoming["schema_version"]`. So a version 1 descriptor gaining a run becomes version 2, and the carried entries are untouched (checked by reading; I3 is about the tests, not this logic).
- **`split_figure_file_path`.** It accepts 4 and 5 parts and refuses `.`, `..`, empty components, absolute paths, a wrong root and any other depth. Each refusal branch is pinned by at least one case:
  - `figures/r/b/./f.png` only by the `len(raw)` check;
  - `figures/r/../p/f.png` only by the `..` check;
  - `/figures/…` and `tables/…` only by `parts[0]`.
- **Writer.**
  - It writes a group document per plot folder through `_ensure_group`.
  - It writes no folder for a page that stored nothing, because the builder omits it.
  - Pages and failures carry `"plot"`, null for flat pages and binding-level failures.
  - Descriptor key order is fixed.
- **Carry.** Both shapes are carried. The plot-folder group is created before the link, so dropping it would fail loudly: the link would hit a missing directory.
- **`plot_page_paths`.**
  - Plot folders are unique per binding (case-folded digest) in first-appearance order, and stems are unique per folder.
  - The reserved names keyed `""` work: a plot named exactly `zarr.json` still gets the suffix. `test_reserved_names_never_become_plot_folders` would fail if they were keyed by name, as the plan had them.
  - A global, rather than per-plot, stem pass would be caught by `test_hand_built_pages_use_backend_defaults_and_collision_safe_names`.
- **`_build_pages`.** It closes every figure if `plot_page_paths` or the spec lookup raises. Path computation sits inside the binding's failure boundary.
- **`_kept_binding`.**
  - The single-binding-folder check now applies to the binding part only. `test_a_binding_spread_over_two_binding_folders_is_refused` is genuine: it asserts the 5-part path, moves a real file, and checks the error.
  - A check over binding plus plot would break `test_a_foldered_v2_binding_is_kept_byte_identical`, so both directions are pinned.
- **Sweep rule (plan-review I1) applied in this commit range.**
  - `_files` asserts non-empty.
  - The tamper loop in `test_a_drawable_binding_is_redrawn_not_kept` asserts it found PNGs before tampering.
  - `[png] = ….rglob("*.png")` fails loudly.
  - The flat negative checks left in `test_image_figures_store.py` (`:89`, `:133`, `:188`) run against flat pages that test writes itself, so they are not vacuous.
- **Review Focus status.**
  - **1, collisions:** pinned, by `plot_page_paths` unit tests and the folder digest through `_build`.
  - **2, version 1 kept flat:** pinned at the keeper and write-back level (`test_a_flat_v1_binding_is_kept_flat`, `test_a_kept_flat_page_is_written_flat_with_a_null_plot`), but not through the real measure path (I3).
  - **3, older writer:** true by inspection of `ba725001`. The test is weaker than claimed (minor 1).

## Verification

- **Read in full:** `_image_figures.py`, `_store_figures.py`, the `_writer.py` helpers, `_output.py`, the `_measurement_tables.py` gate and the `_image_io_handler.py` gate.
- **Read with the diffs:** every new or edited test, using main's `git log`, test diff and src diff (`/bigdata/exfab/anguy344/PhenoTypic/.worktrees/phaseA-*.{txt,diff}`).
- **Executed:** only the mutation batch above (Slurm 29304001, run by main).
- **Not executed:** the killing tests proposed here have not been run. Each one's expected red under its mutation is derived from reading the code.

---

## Resolution (orchestrator, 2026-09-30)

- **Fix-up commit:** `21b2c696`. It adds the killing tests for I1-I4 and minor items 2 and 8, the minor 1/3/4 test changes, and the minor 5 `StoredFigurePage` check. It also fixes the missed `test_process_only_zarr.py:720` path.
- **Runs:**
  - red phase (Slurm 29304075): 2 failed, both minor-5 params, as predicted;
  - green (29304086): 179 passed.
- **Mutation rerun against `21b2c696`** (Slurm 29304097): every mutant is now killed.
  - BASELINE: 129 passed.
  - M1: `test_a_bare_page_and_a_plotted_page_naming_one_file_are_refused`.
  - M2, M3: `test_failures_of_plotted_pages_record_the_plot_not_the_key`.
  - M4: `test_a_page_with_no_plot_over_a_plot_folder_is_refused`.
  - M5: `test_a_page_spread_over_two_plot_folders_is_refused`.
  - M6: `test_a_kept_page_keeps_the_folder_its_path_names_not_its_plot_name`.
  - M9: `test_a_measure_rewrite_adds_a_v2_run_to_a_v1_store`.
  - M11: `test_a_v1_store_gains_a_v2_run_and_keeps_its_v1_run_byte_for_byte` and `test_a_resave_over_a_v1_store_leaves_no_stale_file_in_its_run`.
  - Control C1: still red.
  - The worktree fingerprint was identical before and after (`21b2c696`, dirty=0).
- **Deferred:**
  - Minor 6: no fix needed.
  - Minor 7: GUI tab value `plot/key`, a follow-up outside this spec.
  - Minor 9: carried into Task 5's brief.
