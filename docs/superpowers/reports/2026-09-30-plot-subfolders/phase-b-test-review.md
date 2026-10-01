# Phase gate B review: plot subfolders, Tasks 4-5 (implementation and tests)

Reviewer: implementation-test-reviewer (phase gate B). Analysis only. No source, test, spec or plan file was edited.

**Scope:**
- `git diff 5ea81a69..3d0ecb50 -- src tests`: commits 77cb67e2 (T4, the store copy-out) and 3d0ecb50 (T5, the direct publisher and the GUI guard-count test).
- Checked against:
  - the spec `docs/superpowers/specs/2026-09-30-plot-subfolders/design.md` (§3, D4, D6);
  - the plan (decision P1, Tasks 4-5, Global Constraints including the sweep rule, Review Focus 5);
  - `plan-review.md` and `phase-a-test-review.md`.

## Summary verdict: implementation correct, tests have three holes

**Implementation.** I found no bug in the T4-T5 code that a reader would see:
- Names and folders come from the stored path.
- Both writers ask the guard immediately before creating a plot folder.
- Failures are matched on `(key, plot)`.
- Manifest v3 has the same shape from both writers.
- Leftover and stale-sibling removal are scoped to the page's folder.
- The HTML `plotly.min.js` depth is right at both depths.

Two small edge cases are MINOR: an empty plot folder left by a failed copy-out, and a figure left open on one refusal path.

**Tests.** The batch ran 18 mutants and a red control. Every mutant I predicted killed was killed. The four I predicted to survive did survive, and each one is a real hole:
- **C5b, C6:** decision P1, "names come from the stored path", is pinned only where the stored name equals what `plot` or `key` would give. A copy-out that re-derives either one passes.
- **B1:** Review Focus 5 end to end. The builder could store the cleaned folder as the page's `plot`, and every test would pass. That would split a page from its failures under `(key, plot)` matching.
- **W4:** the direct publisher's `failed[].plot` could be the key.

Each hole has a killing test below.

Counts (from this file): **BLOCKER 0, IMPORTANT 3, MINOR 7.**

---

## Mutation evidence (Slurm job 29304592, run by main)

**Setup:**
- A scratch copy from `git archive 3d0ecb50`, with its own `uv sync`, including the Qt group for G.
- Each mutant was applied alone, then restored from git and verified.
- Worktree fingerprint: `3d0ecb50 dirty=0` both before and after.

**Test sets:**
- **U:** `test_store_copyout.py`, `test_coordinator.py`, `test_output_adapter.py`.
- **S:** U plus `test_store_figures_build.py`, `tests/unit/cli/test_staged_figures_keep.py`, `tests/unit/sdk_/test_image_figures.py` and `test_image_figures_store.py`.
- **G:** `tests/gui/results_viewer/test_mutation_guard.py::test_real_plot_writer_rechecks_after_render_and_preserves_generation`.

**Baseline (unmutated, S+G):** 209 passed.

| ID | Mutation (src) | Tests | Result | Killed by |
|---|---|---|---|---|
| W1 | `_writer.py`: drop the guard before the plot-folder `mkdir` | U | 1 failed (**killed**) | `test_the_guard_is_asked_before_a_plot_folder_is_created` |
| W1b | `_writer.py`: guard moved after the `mkdir` | U | 1 failed (**killed**) | same |
| W2 | `_writer.py`: `page_directory.rmdir()` -> `pass` | U | 2 failed (**killed**) | `test_unsupported_page_fails_without_suppressing_sibling`, `test_every_page_failing_yields_an_explanatory_manifest` |
| W3 | `_writer.py`: `_remove_stale_sibling(directory, …)` | U | 2 failed (**killed**) | `test_an_aggregate_rerun_without_chrome_removes_the_previous_png`, `test_a_rerun_removes_the_old_rendering_inside_its_plot_folder` |
| **W4** | `_writer.py`: failed entry `"plot": page.key` | U | **91 passed (survived)** | none |
| W5 | `_writer.py`: `created = True`, `mkdir(exist_ok=True)` (guard asked even when the folder exists) | U+G | 1 failed (**killed by G only**) | `test_real_plot_writer_rechecks_after_render_and_preserves_generation` |
| C1 | `_store_copyout.py`: drop the guard before the plot-folder `mkdir` | U | 2 failed (**killed**) | `test_a_refused_guard_creates_no_plot_folder`, `test_the_flat_path_rechecks_the_publication_guard_before_commit` |
| C1b | `_store_copyout.py`: guard moved after the `mkdir` | U | 1 failed (**killed**) | `test_a_refused_guard_creates_no_plot_folder` |
| C2 | `_store_copyout.py`: `failed_only` matched on key only | U | 1 failed (**killed**) | `test_a_failure_belongs_to_the_page_of_its_key_and_plot` |
| C3 | `_store_copyout.py`: `partial` matched on key only | U | 1 failed (**killed**) | same |
| C4 | `_store_copyout.py`: "no stored file" entry without `plot` | U | 1 failed (**killed**) | `test_a_page_with_no_copyable_file_is_failed_with_its_plot` |
| C5 | `_store_copyout.py`: folder = raw `page["plot"]` | U | 1 failed (**killed**) | `test_the_plot_folder_comes_from_the_stored_path_not_the_plot_name` |
| **C5b** | `_store_copyout.py`: folder = `safe_path_component(page["plot"])` | U | **91 passed (survived)** | none |
| **C6** | `_store_copyout.py`: file name = `page["key"] + extension` | U | **91 passed (survived)** | none |
| C7 | `_store_copyout.py`: `_remove_leftovers(directory, …)` | U | 3 failed (**killed**) | `test_a_republished_page_loses_its_leftover_renderings`, `test_a_rerun_as_matplotlib_removes_the_previous_html`, `test_a_multi_page_image_rerun_without_chrome_removes_the_previous_pngs` |
| C8 | `_store_copyout.py`: `_discard_page` unlinks `directory / Path(written).name` | U | 1 failed (**killed**) | `test_a_refused_guard_mid_page_leaves_no_half_page` |
| **B1** | `_store_figures.py:489`: `plot=plot_directory` (cleaned folder as the page's logical plot) | S | **208 passed (survived)** | none |
| RC (control) | `_store_copyout.py`: manifest `"schema_version": 2` | U | 2 failed (**red**, as predicted) | `test_a_single_default_page_lands_in_its_plot_folder_with_generated_html`, `test_multi_page_writes_a_directory_and_manifest_v3` |

**Gate B test job** (Slurm 29304475, tree 3d0ecb50, run by main): 350 passed, 8 failed. Main checked each failure. All 8 are Task 7 pins of the old layout:
- `test_calibration_figure_in_store.py` ×3;
- `test_figures_in_store.py:138`;
- `test_publication_end_to_end.py:127`, `:151`, `:170` and `:218`.

The out-of-set files gave 35 passed.

---

## BLOCKER

None.

---

## IMPORTANT

### I1. Decision P1 is pinned only where the store's names equal `plot` and `key` (C5b, C6 survived)

**Where:**
- `src/phenotypic/plotting/_pipeline/_store_copyout.py:241-242`;
- test `tests/unit/plotting/test_store_copyout.py` (the `_page` fixture and every copy-out test).

**Problem.**
- P1 and D4 say the copy-out copies each stored file under the store's own relative path. The store itself chose that path through `plot_page_paths`, which:
  - adds a digest suffix to a folder or stem that collides case-folded or is reserved;
  - and cleans the key.
- In every copy-out test, though, the stored folder equals `plot.replace(" ", "-")`, and the stored stem equals the key. The `_page` fixture builds them that way, and no test key contains a character that cleaning changes.
- So two wrong copy-outs pass all 91 U tests:
  - one that re-derives the folder as `safe_path_component(page["plot"])` (C5b);
  - one that names the file `key + extension` (C6).
- `test_the_plot_folder_comes_from_the_stored_path_not_the_plot_name` kills only the *raw* `page["plot"]` (C5), because `safe_path_component("Tile overlay") == "Tile-overlay"` is exactly what the fixture stored.

**Why it matters.**
- Under C5b, two plots `Tiles` and `tiles` (folders `Tiles/` and `tiles-<digest>/` in the store) would both copy out to... `Tiles/` and `tiles/`. Those names are distinct on Linux. But they no longer mirror the store, and on a case-insensitive filesystem they merge.
- Under C6, a key with a space, or two keys that collide case-folded, would publish under a name the store never wrote. Two such pages in one plot folder would overwrite each other.
- The phase A review warned about this exact re-derivation for T4 ("keep at least one copy-out test whose plot contains a space, or the I2 hole reappears"). The space test was added, but its fixture cleans to what re-derivation produces.

**Fix: a test that kills C5b and C6.** Add to `test_store_copyout.py`. The names are taken from production `plot_page_paths`, so the test cannot drift from the builder.

```python
def test_a_disambiguated_folder_and_file_are_copied_as_the_store_named_them(tmp_path):
    """P1 / D4: folder and file come from the stored path, including where the
    store had to disambiguate them, so neither can be re-derived from
    `plot` or `key`."""
    from phenotypic.plotting._pipeline._writer import plot_page_paths

    ids = [("Tiles", "roi 0"), ("tiles", "roi 0")]
    paths = plot_page_paths([(plot, key, key) for plot, key in ids])
    # Premise: the second folder carries the digest suffix; the stem is not the key.
    assert paths[0] == ("Tiles", "roi-0") and paths[1][0].startswith("tiles-")
    pages = tuple(
        StoredFigurePage(
            key, None, "plotly", {"k": 1},
            (StoredFigureFile("plotly-json", "application/vnd.plotly.v1+json",
                              f"{stem}.plotly.json", _plotly_json()),),
            plot=plot, directory=folder,
        )
        for (plot, key), (folder, stem) in zip(ids, paths)
    )
    plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(*pages)))
    directory = plots / "sym" / "ds-1" / _STEM
    assert [p["files"] for p in _manifest(plots)["pages"]] == [
        {"plotly-json": f"{folder}/{stem}.plotly.json", "html": f"{folder}/{stem}.html"}
        for folder, stem in paths
    ]
    for folder, stem in paths:
        assert (directory / folder / f"{stem}.html").is_file()
```

**How each mutant fails it:**
- **C5b:** the second folder becomes `tiles`, so the manifest no longer matches.
- **C6:** the file becomes `roi 0.plotly.json`, so the manifest no longer matches.

### I2. Review Focus 5 is not pinned end to end: the builder's page `plot` can be the cleaned folder (B1 survived)

**Where:**
- `src/phenotypic/plotting/_pipeline/_store_figures.py:489` (`plot=page.plot_name, directory=plot_directory`);
- the matching in `_store_copyout.py:174-179` and `:301-303`.

**Problem.**
- The spec (§2) says a descriptor page's `plot` is the *logical, unsanitized* name, and so are failures (`_store_figures.py:433, 480`, pinned by phase A I1).
- The copy-out matches a page to its failures by `(key, plot)`. So the chain is only right if the builder writes the same logical name on both sides.
- B1 changes the page side to the cleaned folder name. It passes all 208 S tests:
  - Every builder test whose page `plot` matters uses a name that cleans to itself (`tiles`, `delta_e`).
  - Phase A's `SpacedPlot` test compares `kept == first`, which holds under B1 because both sides read the mutated value back.
  - Every T4 copy-out test that exercises `(key, plot)` uses a hand-built `StoredFigureFailure`. So the chain is pinned only piecewise: builder to failure, and descriptor to manifest. No test runs builder to manifest with a plot name that does not clean to itself.

**Why it matters.** Under B1, take a plot named `Tile overlay`, as in a labelled calibration plot:
- Its page is stored as `plot: "Tile-overlay"` and its failures as `plot: "Tile overlay"`.
- In deliverables, a partial failure no longer attaches to its page.
- `failed` lists a "failed" page beside the published one with the same key.

This is exactly the Review Focus 5 property: "keeps its `plot` through the descriptor and into the deliverables manifest".

**Fix: a test that kills B1.** Add to `test_coordinator.py`. It goes through the real builder, store and copy-out (`emit_image_via_store`).

```python
class _SpacedPlotWithAFailure(BaseModel, PlotImage):
    def inspect(self, subject=None, *, for_save=False, **overrides):
        from matplotlib.figure import Figure

        good = Figure()
        good.subplots().plot([0, 1])
        return PlotOutput(pages=(
            PlotPage(key="roi_0", plot="Tile overlay", figure=good),
            PlotPage(key="roi_1", plot="Tile overlay", figure=object()),   # fails outright
        ))


def test_a_plot_name_survives_builder_store_and_copy_out(tmp_path) -> None:
    """Review Focus 5 end to end, with a plot name that does not clean to
    itself, so the logical name and the folder differ."""
    import json

    pipeline = ImagePipeline(plots=[PlotBinding(id="cal", plot=_SpacedPlotWithAFailure())])
    emit_image_via_store(PlotCoordinator(pipeline, tmp_path))

    [directory] = (plots_dir(tmp_path) / "cal" / "ds").iterdir()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert [(p["key"], p["plot"], p["files"]) for p in manifest["pages"]] == [
        ("roi_0", "Tile overlay", {"png": "Tile-overlay/roi_0.png"}),
    ]
    assert [(f["key"], f["plot"]) for f in manifest["failed"]] == [("roi_1", "Tile overlay")]
    assert (directory / "Tile-overlay" / "roi_0.png").is_file()
```

Under B1, the page's `plot` is `"Tile-overlay"`. Optionally, add a per-format failure on `roi_0` (monkeypatch `_store_figures.serialize_store_format` as phase A's I1 test does, with a Plotly page storing two formats). That would also pin `partial` across the same boundary.

### I3. The direct publisher's `failed[].plot` is unpinned (W4 survived)

**Where:**
- `src/phenotypic/plotting/_pipeline/_writer.py:466-469`;
- tests `test_output_adapter.py:116-128` and `:308-339`.

**Problem.**
- Spec §3 makes `plot` part of every `failed[]` entry, for both writers.
- Both direct-publisher failure tests use bare pages, where `plot_name == key`. So `"plot": page.key` (W4) passes all 91 U tests.
- The manifest's *page* entry is pinned by `test_pages_land_in_plot_folders_with_a_v3_manifest`. The failed entry is not.

**Fix: a test that kills W4.** It also pins the folder shared by two pages, which `created` must leave in place.

```python
def test_a_failed_plotted_page_records_its_plot_and_keeps_the_shared_folder(tmp_path) -> None:
    output = PlotOutput(pages=(
        PlotPage(key="good", plot="tiles", figure=plt.figure()),
        PlotPage(key="bad", plot="tiles", figure=object()),
    ))
    manifest = publish_plot_output(output, tmp_path, plot_id="demo")
    assert [(f["key"], f["plot"]) for f in manifest["failed"]] == [("bad", "tiles")]
    # The folder holds the good page, so the failed page does not remove it.
    assert sorted(p.name for p in (tmp_path / "tiles").iterdir()) == ["good.png"]
```

---

## MINOR

1. **The copy-out leaves an empty plot folder when a copy fails after its `mkdir`** (`_store_copyout.py:246-258`).
   - **Cause.** The folder is created after a successful read, but before `_atomic_write`. A non-guard error from the copy (for example `OSError`) leaves `tiles/` behind, empty.
   - **Why it matters.** The direct publisher removes it in the same case (`_writer.py:463-465`). The store writes no folder for a page that stored nothing. So the copy-out stops mirroring the store, and the two writers disagree.
   - **Fix.** Track `created` per page directory, as the writer does. When the page ends with no `files`, and this page created the folder, and it is empty, `rmdir` it. A plot folder shared with a sibling page that published stays.
   - **Killing test.** It is red today. `_commit_manifest` does not go through the module's `_atomic_write`, so the manifest is still written:

   ```python
   def test_a_page_whose_copy_fails_leaves_no_empty_plot_folder(tmp_path, monkeypatch):
       from phenotypic.plotting._pipeline import _store_copyout

       def _disk_full(*args, **kwargs):
           raise OSError("disk full")

       monkeypatch.setattr(_store_copyout, "_atomic_write", _disk_full)
       plots = _publish(tmp_path, figure_store(tmp_path / "s", _one(_page("roi_0", plot="tiles"))))
       # Premise: the read passed, so the copy (after the mkdir) is what failed.
       assert [(r["format"], r["error"]) for r in _lines(plots)] == [("plotly-json", "OSError: disk full")]
       manifest = _manifest(plots)
       assert manifest["pages"] == []
       assert [(f["key"], f["plot"]) for f in manifest["failed"]] == [("roi_0", "tiles")]
       assert not (plots / "sym" / "ds-1" / _STEM / "tiles").exists()
   ```

2. **In the direct publisher, the plot-folder guard and `mkdir` sit outside the page's `try`** (`_writer.py:423-438`). This has two consequences:
   - **A refusal there leaks the page's figure.** The `except PlotPublicationBlocked: FigureAdapter.close(page.figure); raise` arm only covers `_render_page`. Before T5, every in-loop refusal came from inside that `try`.
   - **A `mkdir` error aborts the whole publication.** A `PermissionError`, for example, now ends the publication with no manifest, where a render error is recorded per page. The copy-out's `mkdir` is inside its per-file `try`.

   **Fix.** Move the `created`/`mkdir` block inside the `try`. Its `PlotPublicationBlocked` arm already closes and re-raises. Decide separately whether a `mkdir` `OSError` should become a page failure.

   **Killing test** (red today; `FigureAdapter.close` closes pyplot figures, as `test_the_flat_path_closes_its_matplotlib_figure` relies on):

   ```python
   def test_a_refused_plot_folder_closes_the_page_figure(tmp_path) -> None:
       from phenotypic.plotting._pipeline import PlotPublicationBlocked

       before = set(plt.get_fignums())
       calls = iter([True, True, False])   # entry, inside the lock, then the plot folder
       with pytest.raises(PlotPublicationBlocked):
           publish_plot_output(
               PlotOutput(pages=(PlotPage(key="a", plot="tiles", figure=plt.figure()),)),
               tmp_path / "agg", plot_id="agg", publication_guard=lambda: next(calls, False),
           )
       assert set(plt.get_fignums()) == before
   ```

3. **A stale docstring:** `_writer.py:249-254` (`_remove_stale_sibling`). It still explains the rule "on the flat path because nothing records which generation a file belongs to". No deliverables writer has a flat path any more. State it for the manifest directory and plot folder only.

4. **Test names still say "flat"** in `test_coordinator.py`:
   - `test_the_flat_path_commits_through_the_commit_guard`, `test_the_flat_path_rechecks_the_publication_guard_before_commit`, `test_a_partial_flat_render_publishes_what_it_can_and_records_once` and `test_the_flat_path_closes_its_matplotlib_figure`;
   - the helper `_emit_image_flat`, and the `flat-page` parameter id.

   They now test a single-page binding in a manifest directory. Rename them to `single_page`, so a later reader does not look for a flat case that no longer exists.

5. **`test_matplotlib_pages_publish_with_collision_safe_names` has no collision** (`test_output_adapter.py:70-87`).
   - Its labels are `"A/B"` and `"A B"`. `safe_path_component("A/B")` raises, so the first label becomes `page`, and the second becomes `A-B`. Those are distinct with or without the per-folder uniqueness pass.
   - The plan's I2 ("keep the collision tests meaningful") is carried only by `test_hash_suffix_is_rechecked_for_page_filename_collision`.
   - **Fix.** Use labels that really collide case-folded, for example `"A b"` and `"a-b"`. Those become `A-b` and `a-b-<digest>`. Then assert `files[1]` matches `r"same/a-b-[0-9a-f]{8}\.png"`.

6. **No test runs a real `OutputMutationGuard` against a fresh plot folder.**
   - The GUI test pre-creates `default/`, so it is blind to where the guard sits relative to `mkdir` (W1/W1b do not reach it). It catches only a guard asked when the folder already exists (W5, killed by G alone). That is the right job for it, and the deviation is acceptable.
   - Placement is proven by the unit test (W1, W1b killed).
   - **Optional.** Add a GUI variant without the pre-publish, perturbing at check 3 (then the plot-folder question). Assert `after_dirs == before_dirs` and `not (plot_dir / "default").exists()`.

7. **T8 misses a docs section.** `docs/source/extending/pages/custom_plotter.md:419-450` ("The plot manifest") still documents `schema_version: 2`, a label-named root file (`"files": {"html": "Colony-area.html"}`), and `failed` without `plot`.
   - `:385-395` still speaks of "single-figure image plots".
   - T8's file list names only `:178` and `:227-229`. Add these ranges, or the user docs will describe manifest v2 after the release.

---

## Edge cases the orchestrator asked about

- **Direct publisher: an empty plot folder after a blocked commit.** No finding; the implementer's call is right.
  - After `PlotPublicationBlocked`, the contract is to mutate nothing further. An `rmdir` there would be an unguarded mutation of the tree the guard just protected.
  - Existing behaviour matches. `publish_plot_output` and `_publish_binding` already leave their binding or image folder after a later refusal. `test_the_flat_path_rechecks_the_publication_guard_before_commit` pins the same outcome for the copy-out: "the plot folder exists, its file does not".
  - The folder holds no file and the manifest is not replaced, so no reader is misled.
- **Copy-out: no empty-folder `rmdir`.** MINOR 1 above.
- **The GUI test's count stays 3.** It is sound for what it claims, as MINOR 6 explains:
  - The two premises (`default/default.png` exists before, and exactly one pending `.default.png.*.tmp` at check 3) do bite: W5 was killed by G alone.
  - It cannot see guard placement relative to `mkdir`, by construction. `test_the_guard_is_asked_before_a_plot_folder_is_created` carries that (W1, W1b).

---

## Notes for T6 and T7 (not counted)

- **T6 (per-ROI calibration).**
  - Calibration's names, plot `tiles` and keys `roi_<i>`, clean to themselves. So T6's tests will not exercise holes I1 or I2. Land those killing tests here, before T6.
  - Calibration's labels ("Tile overlay, ROI i") do not reach deliverables file names, under P1. The copy-out names files by the store's key-derived stems. T7's `PAGES` (`tiles/roi_0.png`, …) is consistent with that.
  - A failing overlay fails the whole binding ("all pages or none"). That is a binding-level failure (`page: null`), so it is a record only, with no manifest directory (`test_a_binding_level_failure_is_a_record_only`).
- **T7 (integration and determinism).**
  - The 8 gate-B failures are the expected old-layout pins.
  - The image-figure HTML src at the copy-out depth is `../../../../plotly.min.js`: `plots/<binding>/<dataset>/<stem>/<plot>/`.
  - A version 1 flat page copied into the image folder is `../../../`. That is pinned by `test_a_flat_v1_page_is_copied_into_the_image_folder`.
  - `_deliverable`'s `glob("plate-*")` now matches only the image folder, because no flat `plate-<hash>.<ext>` file is written any more.
  - The determinism test is in process mode, so it covers no deliverables. Manifest bytes are deterministic by construction: `sort_keys=True`, and page order is descriptor order.
- **Upgrade leftovers.** A pre-upgrade deliverables tree keeps its flat `<dataset>/<stem>.<ext>` files and its label-named root files beside the new folders. Spec §6's release note covers this, and spec §3 puts sweeping out of scope.
- **Key with `/` (phase A minor 9).** In the direct publisher, a time-series group key containing `/` gets the folder `page`, then `page-<digest>`. The folders are distinct, so nothing is overwritten. It is only an unhelpful name. Plan-review minor 3 accepted this.
- **The copy-out trusts a stored path's run and binding parts.** It uses only the plot folder and the name from `split_figure_file_path`. That follows from P1, and the trust was there before this change: the containment and sha256 checks are what gate a stored file.

---

## Validated (no finding)

- **Guard immediately before every directory creation.**
  - Copy-out: `_store_copyout.py:246-250`. Direct publisher: `_writer.py:423-427`.
  - Asked only when the folder does not exist; the folder-creating branch is the only new one.
  - Dropping the guard or moving it after the `mkdir` is killed in both writers (W1, W1b, C1, C1b).
  - A guard asked for a folder that already exists is killed (W5).
- **`(key, plot)` matching** in `failed_only` and `partial`: C2 and C3 killed. A version 1 descriptor matches on `(key, None)` on both sides (`test_a_flat_v1_page_is_copied_into_the_image_folder` pins `partial`).
- **The "no stored file" entry carries `plot`:** C4 killed. A tampered file is refused before the `mkdir`, so no folder is created for it (`:391`).
- **Folder from the stored path, not the raw `plot`:** C5 killed. The weaker re-derivation is I1.
- **A version 1 flat page lands in the image folder**, `"plot": null`, at HTML depth `../../../`.
- **`_remove_leftovers` and `_remove_stale_sibling` are scoped to the page folder** (C7, W3 killed). `_remove_leftovers` receives base names, since manifest values are relative paths. `stem` and `page_directory` are always bound from the current page whenever `files` is non-empty, because they are reassigned by any entry that reached the copy.
- **`_discard_page`** still resolves the relative manifest values (`directory / "tiles/roi_0.plotly.json"`); C8 killed.
- **Empty-folder cleanup in the direct publisher:**
  - The cleanup itself: W2 killed.
  - A shared folder survives a later failing page, because `created` is `False` for it (I3's test pins this).
- **Manifest v3 shape is the same in both writers:**
  - pages: `{key, plot, label, files, backend, metadata, partial?}`;
  - failed: `{key, plot, label, error}`;
  - `files` relative to the manifest directory;
  - written with `sort_keys=True` through the shared `_commit_manifest`.
- **HTML src depth** comes from `plotlyjs_src_for(page_directory, …)` in both writers. It is pinned at `../../` (aggregate), `../../../` (version 1 flat) and `../../../../` (image plot folder).
- **Nothing in `src/` still assumes manifest v2 or the flat deliverables layout.** I grepped `manifest.json`, `schema_version`, `unique_page_stems`, `flat` and `plots_dir` across `src/`, including `_gui`, `_cli` and `sdk_`:
  - No source file reads the deliverables manifest.
  - `unique_page_stems` is used only inside `plot_page_paths`.
  - The README generator mentions only `plots/`.
  - The only leftover is the docstring in MINOR 3.
- **Sweep rule.** I re-ran the grep over the four touched test files:
  - Every negative check is at the new depth or uses `rglob`.
  - The globbed positive checks assert non-empty, or unpack (`[html] = …`, `[plot_folder] = …`).
  - `first == second` asserts `first` is non-empty.
  - `_assert_manifest_matches_disk` asserts the manifest names files.
- **GUI test deviation:** see the edge cases above.

## Verification

- **Read in full:** `_store_copyout.py`, `_writer.py` (lines 40-570) and `_store_figures.py` (`_build_pages`, `_build_page`).
- **Read with the diffs:** every new or edited test, using `phaseB-src.diff` and `phaseB-tests.diff`, plus the surrounding fixtures (`_store_fixtures.py`, the `_page` fixture, the coordinator plot classes).
- **Executed:** only the mutation batch above (Slurm 29304592, run by main). The gate-B counts are main's (Slurm 29304475).
- **Not executed:** the killing tests proposed here. For I1-I3 and MINOR 1 and 2, the expected red under each mutant, or against today's code, is derived from reading the code.
