# Phase gate C review: plot subfolders, Tasks 6-7 (implementation and tests)

Reviewer: implementation-test-reviewer (phase gate C). Analysis only. No source, test, spec or plan file was edited. The only other file written is the mutation module `/bigdata/exfab/anguy344/software/AutoConvertRaw-GC/logs/mut/mutations_c.py`.

**Scope:**
- `git diff cd630b61..288b6f66`: commits cd62c7dc (T6, per-ROI calibration pages) and 288b6f66 (T7, integration expectations and the determinism test).
- Checked against:
  - the spec `docs/superpowers/specs/2026-09-30-plot-subfolders/design.md` (§2 Determinism, §3 by mode, §4, §5);
  - the plan (Tasks 6-7, Review Focus 4, the sweep rule);
  - `phase-a-test-review.md` and `phase-b-test-review.md`, including B's "Notes for T6 and T7".

## Summary verdict: implementation correct; the determinism test does not test "same day"

**Implementation (T6).** I found no bug.
- The shallow `model_copy` is sound for the renderer.
- Zero ROIs cannot occur.
- Every image of one operation gets the same page set.
- `show_tiles()` is untouched and still pinned.

The details are under "Task 6 correctness" below.

**Tests (T7).** One IMPORTANT finding and four MINOR ones:
- **I1:** the new determinism test gives both runs the same timestamp and pid, in one interpreter. So it cannot see the values a real same-day rerun differs by. Mutant D1 survived it (job 29305200).
- **M1-M4:**
  - the per-ROI oracle is the implementation's own expression;
  - the refused frame is not exercised through `inspect()`;
  - Review Focus 4's "no stray folder" is not exercised;
  - `_deliverable` checks only the paths it expects.

Counts (from this file): **BLOCKER 0, IMPORTANT 1, MINOR 5.**

**Mutation batch.** `mutations_c.py` ran as Slurm job 29305200 (run by main): 9 mutants and a baseline.
- The baseline was green: `104 passed`.
- Every mutant failed at least one test somewhere in the suite.
- Two survived the test that is meant to pin them, as predicted:
  - **D1** survived the determinism test (I1);
  - **S1** survived `_deliverable` (M4).
- One prediction was open: S1 against the copy-out unit file. It was killed there, incidentally (see M4).

Results are quoted as measured under "Mutation evidence".

---

## BLOCKER

None.

## IMPORTANT

### I1. The determinism test holds the call's timestamp and pid equal, so "same day" is untested

**Where:** `tests/integration/cli/test_calibration_figure_in_store.py:170-190`, `test_two_same_day_process_runs_write_byte_identical_stores`.

**What it does:**
- Both runs receive `RunInitiation(DAY, f"{DAY}T12:00:00.000Z", 7)`. That is the same date, the **same timestamp** and the **same pid**.
- Both run in the pytest process, so they share one `PYTHONHASHSEED`.

**What the contract says.** Spec §2 "Determinism" and figures spec §1a: *two identical process runs on the same UTC day* are byte-identical. Two real runs on one day differ in exactly the call's `at_utc` and `pid` (`mint_run_initiation`, `sdk_/_image_figures.py:91-100`). Process mode is byte-stable only because it omits them:
- `_cli_process_only.py:358-365` passes `date=` alone to `name_figure_run`;
- `_store_figures.py:94-95` writes them into the run entry whenever an initiation is passed.

**Failure scenario.** A refactor makes process mode pass `initiation=run_initiation`, as full mode does. The descriptor then carries `initiated_at_utc` and `initiated_pid`, and two real same-day runs differ. This test still passes, because both of its runs carry identical values (mutant **D1**: **survived** it in job 29305200).

**Mitigation.** The hole is closed elsewhere today, so this is a false green in this test's claim, not an open suite hole:
- `test_figures_in_store.py:260` asserts both keys are absent from the run entry;
- `tests/unit/cli/test_process_only_zarr.py:668` runs two subprocesses with `datetime.now()`, `os.getpid()` and different `PYTHONHASHSEED`.

Both killed D1 in job 29305200. But both cover a Plotly `sym` binding with one page. Nothing else checks a multi-page, matplotlib binding against differing call values. That is exactly what this test was added for (spec §5 "determinism").

**Positive control.** The test is not vacuous for wall-clock data produced *inside* the run. Mutant **D2** (`reproducible_provenance=False`) puts the journal's millisecond `applied_at_utc` and `duration_seconds` into the store, and was **killed** (job 29305200). A run that writes no figures at all is also caught: line 190 requires `.../cal/tiles/roi_1.png`.

**Fix (two lines):**
```python
for name, at, pid in (("a", f"{DAY}T00:00:01.000Z", 7), ("b", f"{DAY}T23:59:59.999Z", 8)):
    ...
    run_initiation=RunInitiation(DAY, at, pid),
```
Also require all three files, not only `roi_1`:
```python
figure_files = {p for p in first if p.startswith("figures/")}
assert {p.split("/cal/", 1)[1] for p in figure_files if "/cal/" in p} >= set(PAGES.values())
```
For hash-seed coverage too, run the two calls as subprocesses with different `PYTHONHASHSEED`, as the unit test at `test_process_only_zarr.py:668` does. That is optional: the calibration path has no set iteration that reaches bytes. `plot_page_paths` orders by `dict.fromkeys` (`_writer.py:325`).

---

## MINOR

### M1. The per-ROI oracle is the implementation's own expression

**Where:** `tests/unit/correction/test_calibration_plot_image.py:75-77`.

```python
alone = render_calibration_overlay(record.model_copy(update={"rois": [roi]}))
assert _png(page.figure) == _png(alone)
```

This kills every wrong *selection*:
- the whole record (R1);
- always the first ROI (R2);
- reversed order (R3);
- a dropped frame-level field (R4).

All four were killed in job 29305200. But it does not independently pin two things spec §4 states:
- *"each overlay figure carries one ROI panel"* (spec §5 `CalibrateColorRpcc` row);
- *"the title still names ... the frame-level `n_fitted` / `n_expected`"*.

Say `render_calibration_overlay` regressed for single-ROI records, or a later change computed per-ROI counts in a helper that both the test and `inspect()` call. Then both sides would move together.

**Fix:** add structural checks per tiles page:
```python
image_axes = [ax for ax in page.figure.axes if ax.images]
assert len(image_axes) == 1
assert f"{record.n_fitted}/{record.n_expected} patches fitted" in page.figure.texts[0].get_text()
assert_no_overlap_or_clipping(page.figure)  # spec §4: no-overlap sizing per figure
```
`assert_no_overlap_or_clipping` is the helper `test_calibration_overlay.py` already uses.

### M2. A refused frame, and a ROI with every patch missing, never go through `inspect()`

**Where:** `test_calibration_plot_image.py:85-96` covers `skipped` only. There the lattices are found and tiles are identified.

**What is unexercised.** Spec §4 says *"A refused frame still draws every per-ROI overlay, each carrying its own reasons"*. No test draws a lone ROI with `lattice_found=False` and `tiles == []` as its own figure. Until T6, such a ROI was only ever drawn beside another one (`test_calibration_overlay.py:295-305`).

**Code reading.** It should work:
- `_plan_roi` is per ROI;
- `need_h` uses `max(..., default=0.0)` (`_calibration_overlay.py:557-561`).

It is still unpinned.

**Fix:** take the setup of `test_no_usable_tiles_keeps_a_record_with_no_lattice`, but with `operation.apply(frame, inplace=True)` under `pytest.raises`. `quietly()` applies to a copy, whose weakref dies, so `inspect()` would report unavailable. Then:
- `_ids(operation.inspect(frame)) == [("tiles","roi_0"), ("tiles","roi_1"), ("delta_e","delta_e")]`;
- each tiles page has one image axis and no patches, and passes `assert_no_overlap_or_clipping`.

### M3. Review Focus 4's "no empty or stray folder" is not exercised

**Where:** `test_calibration_plot_image.py:99-110`, `test_one_roi_stores_one_overlay`.

It asserts the `inspect()` page ids only. Review Focus 4 says the one-ROI pipeline *stores* `tiles/roi_0` and `delta_e/delta_e` "with no empty or stray folder". Nothing builds or writes it.

The risk is low: the builder is generic, and phase B pinned the empty-folder cleanup.

**Fix:** reuse the `build_image_figures` pattern from `test_an_image_pipeline_listing_it_under_plots_stores_every_figure_in_its_plot_folder`, and assert:
```python
[(p.directory, p.key) for p in binding.pages] == [("tiles","roi_0"), ("delta_e","delta_e")]
```

### M4. `_deliverable` reads only the paths it expects

**Where:** `tests/integration/cli/test_calibration_figure_in_store.py:118-123`.

It reads `directory / name` for each `PAGES` value and checks `manifest["schema_version"] == 3`. It does not check two things:
- **Extra files:** for example, a stray copy at the image folder's root, or a leftover label-named `Tile-overlay-ROI-0.png`.
- **The manifest's own content:** its pages and files, and `failed`.

**Failure scenario.** The copy-out also writes `<stem>/roi_0.png` beside `<stem>/tiles/roi_0.png`. This test still passes: mutant **S1** survived it in job 29305200. S1 was killed by two other tests:
- `test_figures_in_store.py::test_full_mode_stores_figures_and_copies_them_out`, whose `rglob("*.plotly.json")` list must equal exactly one path.
- `test_store_copyout.py::test_a_refused_guard_mid_page_leaves_no_half_page`, through its `rglob("*.plotly.json") == []` at `:148`. That kill is incidental. The stray write is a plain `write_bytes`, so it escapes `_discard_page` when the guard refuses mid-page. On the normal path, no copy-out unit test lists the image folder's root (`:68` and `:185` list inside the plot folder; `:395` lists the root only for a flat version 1 page).

So the narrow check is backed for Plotly bindings, but not for the matplotlib calibration binding.

**Fix:**
```python
on_disk = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
           if p.is_file() and not p.name.startswith(".")}
assert on_disk == {"manifest.json", *PAGES.values()}
assert {(p["plot"], p["key"]): p["files"]["png"] for p in manifest["pages"]} == PAGES
assert manifest["failed"] == []
```

### M5. The aggregate manifest test compares a folder name with a logical plot name

**Where:** `tests/integration/plotting/test_publication_end_to_end.py:206-208`.

`Path(name).parent.as_posix() == page["plot"]` is true only because this plot's name cleans to itself.
- `page["plot"]` is the unsanitized logical name (spec §2, P2).
- The folder is the cleaned one.

A future aggregate plot named with a space would fail this test spuriously. That is a false red, not a false green.

**Fix:** compare against `safe_path_component(page["plot"])`, or note in a comment why the two are equal here.

---

## Task 6 correctness (Questions 1a-1c)

**1a. Is the shallow `model_copy` sound? Yes.**
- **What the renderer reads from the record.** `render_calibration_overlay` reads only `image_name`, `verdict`, `degree`, `n_fitted` and `n_expected`, all in `_figure_title` (`_calibration_overlay.py:486-490`). It also reads `rois`.
- **Everything else is per ROI.** `_plan_roi(roi, meter)` and `_draw_roi(roi, plan, ...)` take the ROI alone (`:557`, `:578-580`). There is no lookup by `roi_index` into the record, and no frame-level patch count drawn per panel.
- **Why the frame-level title is correct on every page.** The fit is frame-wide (spec §4), so the title is meant to repeat across pages, and it does.
- **The panel title names the ROI.** It is `"ROI {roi.roi_index}"` (`:448`).
- **Sharing is safe:**
  - The copy shares the frozen `RoiOverlay` objects and their read-only `crop` arrays (`build_overlay_record`, `:224`).
  - The renderer mutates neither.
  - `model_copy(update=...)` skips validation, but the value is a `list[RoiOverlay]`, so the copy stays well-typed.
- **`render_delta_e_bars` is unaffected.** It reads `refusal` and every ROI's tiles, and it is still called with the full record, through `show_delta_bar_plot()`.

**1b. Zero ROIs, and a ROI whose patches are all missing.**
- **Zero ROIs cannot occur.**
  - `rois: list[CheckerRoi] = Field(min_length=1)` (`_calibrate_color_rpcc.py:175`).
  - The base model sets `validate_assignment=True` (`abc_/_base_operation.py:180`), so `op.rois = []` is refused too.
- **The page count is fixed per operation.** Every `keep_record` call is after the ROI loop (`:616-686`). Each ROI appends its draft before any `continue` (`:490-497`). So `len(record.rois) == len(self.rois)` on every exit that keeps a record. An exception inside the loop keeps no record: `inspect()` raises `FigureInputUnavailable`, and the binding is listed unavailable. So the docstring's "every image stores the same pages" holds for one operation instance.
- **A ROI with every patch missing** gets `lattice_found=False`, `tiles=[]`, and its refusal flags. It renders as a lone panel with its reasons. I read this as correct, but it is untested (M2).
- **`roi_index` is the loop position** (`:493-494`). So spec §4's "not renumbered after a refusal" holds by construction. A position-based mutant would be equivalent, so none is proposed.

**1c. Does `show_tiles()` still behave for its other callers?**
- **It is unchanged:** `:281-286`.
- **Callers.** No source file calls it. It appears only in docs: `correction/CLAUDE.md:46`, and the docstrings at `:157` and `:267`. `inspect()` no longer uses it.
- **It stays pinned** by `test_calibration_overlay.py:737-753`. `test_show_tiles_renders_the_last_apply` asserts two image axes, so it would catch a `show_tiles()` that drew one ROI (mutant **R5**: killed there, and green in `test_calibration_plot_image.py`, in job 29305200).

**"All pages or none."** Overlays are rendered eagerly inside `inspect()`, so a failure drawing any ROI raises out of `inspect()`. That fails the whole binding (`page: null`, a record only), as phase B noted.

---

## Can the T7 tests fail? (Question 2)

| Assertion | Plausible regression it catches | Verdict |
|---|---|---|
| `test_full_mode_…:135-137` pages `(plot, key, backend)` | per-ROI split lost; `plot` not recorded; page order changed | live |
| `:138-141` stored path `figures/<run>/cal/<plot>/<key>.png` | flat store layout; folder named by label; key vs. stem confusion | live |
| `:143-144` each tile ≠ `delta_e` | a tiles page that holds the ΔE chart. It does **not** check `roi_0 ≠ roi_1`; the unit test at `test_calibration_plot_image.py:79` and `:194` does | weak but backed |
| `:145` `_deliverable(out) == data` | flat or label-named copy-out (the file is missing, so the read fails); a bytes mix-up between pages; manifest v2. The `[directory] = glob("plate-*")` unpack also fails if a flat `plate-<hash>.png` sibling is written | live for the expected paths; blind to extra files (M4, S1) |
| `test_process_mode_…:162` `list(data) == list(PAGES)` | page set or order in the descriptor | live |
| `:165-166` files at `figures/<run>/cal/<plot>/<key>.png` | flat or missing store files (largely duplicates `_overlay`'s reads) | live |
| `:167` `not (out / "deliverables").exists()` | process mode starts copying out. `out` is the `output_dir` the core receives, and the deliverables root is `output_dir/deliverables`, so the check is at the right place (mutant **P1**: killed) | live |
| `test_two_same_day_…:189` `first == second` | wall-clock data generated inside the run (D2: killed); absolute output paths in the store (the two output dirs differ). **Not** call timestamp or pid leaks (I1, D1) | partial |
| `:190` `roi_1.png` present | figures not written at all; a flat layout | live; only one of three files required |
| `test_staged_…:272` `list(stage1[1]) == list(PAGES)` | Stage 1 page set | live |
| measure-keep `:210-211` | kept pages not byte-identical; the copy-out after a keep | live (unchanged logic, new keys) |
| `test_figures_in_store.py:137-142` | `default` still flat (the `glob("plate-*")` unpack catches a flat sibling); wrong folder or name; a stray extra `.plotly.json` or `.html` anywhere under the image folder (exact `rglob` lists) | live |
| `test_publication_end_to_end.py:131-155` | flat HTML; wrong bundle depth (`../../../../plotly.min.js` is correct for `plots/<b>/<ds>/<stem>/<plot>/`); a manifest naming files that do not exist, or the reverse; a stored PNG published | live |
| `:166-167` mpl `rglob("*.png")` count 1 | missing PNG; a duplicate PNG | live |
| `:186-208` aggregate manifest v3, on-disk ⇔ manifest, files inside the plot folder | direct publisher still flat; manifest/disk drift | live (M5: brittle on unclean names) |
| `:238-240` neighbour HTML; `ExplodingImagePlot` absent | a failure stopping its neighbour; partial output from a failing plot (checked at the binding level, so depth does not matter) | live |

**The `rglob` replacements.** Every positive `rglob` is rooted at one image directory, through `_image_directory` or the `[image_directory]` unpack. Every negative `rglob` is rooted at `tmp_path` or `out`, the whole tree. None can reach into a neighbouring image or binding. The only blind spot is a file of a *different* name at the dataset level, outside the image folder. That is negligible, and the unpack already catches the flat `plate_01-*` form.

**The sweep rule** (grep run over the three files; hits listed in the table) holds:
- every negative check is at the new location or is whole-tree;
- every positive `glob` asserts non-empty, by unpack, exact list or count;
- `first == second` is backed by line 190.

## Tests that can't fail

- **`test_two_same_day_process_runs_write_byte_identical_stores`** against a timestamp or pid leak into a process store. The two runs carry identical values (I1, mutant D1).
- **`_deliverable`** against any extra file in the image folder, or a manifest whose `pages` or `failed` are wrong while the files are right (M4, mutant S1).
- **`test_one_roi_stores_one_overlay`** against a stray or empty folder. It never builds or writes (M3).
- **`test_inspect_draws_one_overlay_per_roi_…:75-77`** against a single-ROI rendering regression inside `render_calibration_overlay`. The oracle calls the same function the same way (M1). It is not a false green for the selection logic: R1-R4 were all killed.

## Spec coverage (Question 3)

| Spec item touched by T6-T7 | Pinned by | Gap |
|---|---|---|
| §4: 2 ROIs give `tiles/roi_0`, `tiles/roi_1`, `delta_e/delta_e` | unit `_ids`; integration paths; store build | none |
| §4: each overlay carries one ROI panel | equality with the same expression only | M1 |
| §4: the title keeps the verdict and frame-level counts | same | M1 |
| §4: no-overlap sizing per figure | none on the per-ROI pages | M1 |
| §4: a refused frame draws N overlays | `skipped` only | M2 |
| §4: 1 ROI gives `tiles/roi_0` | `inspect` ids | storage side M3 |
| §4: `show_tiles()` unchanged | `test_calibration_overlay.py:737-753` | none |
| §4: all pages or none | by construction; phase B's binding-level failure test | none |
| §2/§5: same-day determinism | `test_process_only_zarr.py:668` (Plotly, varied call values); new test (calibration, identical call values) | I1 |
| §3 by mode: process zarr stores, no deliverables | `:162-167`; `test_figures_in_store.py:245-264` | none |
| §5 end to end: process store holds three files, descriptor lists them | `:160-166` | none |
| D4 mirror: the deliverables file set | expected paths only | M4 |

---

## Mutation batch (Question 4)

**Module:** `/bigdata/exfab/anguy344/software/AutoConvertRaw-GC/logs/mut/mutations_c.py`, in the `mutations_b2.py` format.

**Anchors.** Each one was checked to occur exactly once in its file at 288b6f66 (`grep -cF` printed `1` for each):
- `R`: `src/phenotypic/correction/_color_correction/_calibrate_color_rpcc.py`
- `P`: `src/phenotypic/_cli/_cli_process_only.py`
- `C`: `src/phenotypic/plotting/_pipeline/_store_copyout.py`

**Test sets:**
- `UT`: `test_calibration_plot_image.py`
- `OV`: `test_calibration_overlay.py`
- `CU`: `tests/unit/plotting/test_store_copyout.py`
- integration node ids as named in the module

### Mutation evidence (Slurm job 29305200, run by main)

**How it ran:**
- `harness.py` on a `git archive` copy at 288b6f66, with `mutations_c`.
- `COMPLETED 00:18:30`, exit `0:0`.
- Each mutant was applied alone. Every mutant line was followed by `restored+verified 1 file(s)`.
- Worktree fingerprint: `288b6f66 dirty=1` before and after. The `dirty=1` is this untracked report.

**Baseline:** `104 passed in 347.85s` over 9 test files and node ids.

| ID | File | Mutation | Tests | Predicted | Measured | Failing tests |
|---|---|---|---|---|---|---|
| R1 | R | each page drawn from the whole `record` | UT | KILLED | 2 failed, 10 passed (**killed**) | `test_inspect_draws_one_overlay_per_roi_and_the_delta_e_chart`, `test_an_image_pipeline_listing_it_under_plots_stores_every_figure_in_its_plot_folder` |
| R2 | R | each page drawn from `record.rois[0]` | UT | KILLED | 2 failed, 10 passed (**killed**) | same two |
| R3 | R | `for roi in reversed(record.rois)` | UT | KILLED | 3 failed, 9 passed (**killed**) | same two, plus `test_a_skipped_frame_still_draws_every_roi` |
| R4 | R | per-ROI copy also sets `n_fitted=None` (title drift) | UT | KILLED | 1 failed, 11 passed (**killed**) | `test_inspect_draws_one_overlay_per_roi_and_the_delta_e_chart` |
| R5 | R | `show_tiles()` draws only the first ROI | OV + UT | OV killed, UT green | 1 failed, 64 passed (**as predicted**) | `test_calibration_overlay.py::test_show_tiles_renders_the_last_apply` |
| D1 | P | process mode passes `initiation=run_initiation` | DET, PROC_SYM, DET_SYM | DET survives; the other two kill it | 2 failed, 2 passed (**DET survived**) | `test_figures_in_store.py::test_process_mode_carries_figures_only_in_a_store[zarr]`, `test_process_only_zarr.py::test_two_processes_with_a_figure_binding_write_byte_identical_stores` |
| D2 | P | `reproducible_provenance=False` (positive control) | DET | KILLED | 1 failed (**killed**) | `test_two_same_day_process_runs_write_byte_identical_stores` |
| P1 | P | process mode creates `output_dir/deliverables` | PROC | KILLED | 1 failed (**killed**) | `test_process_mode_zarr_stores_the_overlay` |
| S1 | C | copy-out also writes each file flat at the image folder's root | CU, FULL, FIS | FULL survives; CU uncertain; FIS kills it | 2 failed, 32 passed (**FULL survived**; CU and FIS killed) | `test_store_copyout.py::test_a_refused_guard_mid_page_leaves_no_half_page`, `test_figures_in_store.py::test_full_mode_stores_figures_and_copies_them_out` |

**Reading the D1 line.** Its 4 collected tests are DET, PROC_SYM `[zarr]` and `[tiff]`, and DET_SYM. Two failed, both named above. The two that passed are therefore DET and PROC_SYM `[tiff]`, which writes no store.

**Reading the S1 line.** FULL is not among the failures, so it survived. The CU kill is incidental (see M4).

**Outcome.** Every prediction held. The one open prediction (S1 vs. CU) resolved to killed. The fixes for I1 and M4 should make DET kill D1 and FULL kill S1.


---

## Verification

- **Read-only commands run by this reviewer:** `git diff`, `git log`, `grep`, `sed -n` over the worktree at 288b6f66.
- **No pytest, ruff or sbatch was run by this reviewer.**
- **The gate test job**, Slurm job 29305106, was run by main. Its output, as relayed verbatim:
  - `COMPLETED   00:12:12      0:0`
  - tree `288b6f66  dirty=0`, node `i29`, imported from the worktree's `src/phenotypic/__init__.py`
  - the 17-file figure set: **`367 passed in 701.12s (0:11:41)`**
  - the out-of-set files: **`36 passed, 2 warnings in 3.98s`**
  - So the plan's gate C condition, 0 failures on the 17-file set, is met at 288b6f66.
- **The mutation batch**, Slurm job 29305200, was run by main. Its results are quoted in "Mutation evidence" from the output main relayed verbatim.

---

## Addendum: the two assertions C6 flagged (requested by main)

### A1. The single-`default` Plotly image plot now asserts a manifest is present

`test_publication_end_to_end.py:145-154`, `test_a_plotly_image_plot_publishes_html_and_one_hoisted_bundle`.

**Confirmed against the spec: the change of intent is the spec's own.**
- Spec §3 "Deliverables: mirror the store" (`design.md:251-253`): *"The flat special case is removed. A binding whose only page is `default` is published as `<stem>/default/default.<ext>`, like every other plot."* Its layout block shows `manifest.json  schema_version 3` in every `<stem>/` folder.
- Plan Global Constraints set deliverables manifest `schema_version` to `3`.
- The plan's T7 sweep asked for `:139` to be "restated at the new depth by its intent".

**The code agrees.**
- This test publishes through `build_image_figures` -> `save2zarr` -> `PlotCoordinator.publish_store_figures` (`:66-85`), which is the store copy-out.
- `_store_copyout._publish_binding` reaches `_commit_manifest(...)` (`_store_copyout.py:204`) with no flat branch. Every published binding gets a manifest.
- The unit test `test_store_copyout.py:65-79` pins the same thing for a single `default` page.

**The new assertion is stronger than the old one.** The old check, `not manifest.exists()`, only proved absence at the old depth. The new one requires all of:
- `schema_version == 3`;
- `failed == []`;
- exactly one page, with `files` keys `{"html", "plotly-json"}`;
- the manifest's files equal the HTML found plus every `.plotly.json` on disk.

So it catches each of these:
- a reintroduced flat path, which writes no manifest;
- a manifest naming a file that is missing;
- a stray extra `.plotly.json`.

A stray extra `.html` is caught as well: `pages` comes from `rglob("*.html")`, and `len(pages) == 1` (`:133`) fails if a second one exists. No further finding.

### A2. Integration checks each tile differs from `delta_e`, but not `roi_0` from `roi_1`

`test_calibration_figure_in_store.py:143-144`.

**Is "drew the same ROI twice" caught today? Yes, at unit level, twice over:**
- **Passing `record` instead of the per-ROI copy (mutant R1).** Both pages become the combined figure. This fails:
  - `test_calibration_plot_image.py:76-77`, where each page must equal its single-ROI rendering;
  - `:79`, `pages[0] != pages[1]`.
- **Passing a fixed ROI, `[record.rois[0]]` (mutant R2).** Page 1 no longer equals its `roi_1` rendering (`:77`), and the two pages are equal (`:79`).
- **The build layer.** `test_an_image_pipeline_listing_it_under_plots_stores_every_figure_in_its_plot_folder` asserts `binding.pages[0].files[0].data != binding.pages[1].files[0].data` (`:194`). So the bytes leaving `build_image_figures` are pinned distinct for the two ROIs.

All of these run in the 17-file set. In job 29305200, R1 and R2 each failed two tests: `test_inspect_draws_one_overlay_per_roi_and_the_delta_e_chart` and `test_an_image_pipeline_listing_it_under_plots_stores_every_figure_in_its_plot_folder`.

**What the integration level adds, and what it misses.** The CLI calls the same `inspect()`, so a drawing regression is caught by the unit tests above. The only regression the unit tests cannot see is one *after* the build: a store writer or copy-out that writes one page's bytes to both ROI paths. The two halves of that are covered differently:
- **The copy-out half is caught here.** `_deliverable(out) == data` compares each copied file with its own stored file.
- **The store-writer half is not caught by this test.** `_overlay` reads whatever bytes sit at each path, and the descriptor's `sha256` is not compared.

**Recommendation (MINOR, folded into M4's fix):** add one line after `:144`:
```python
assert data[("tiles", "roi_0")] != data[("tiles", "roi_1")]  # two ROIs, two drawings
```
It is safe because the fixture's two bands hold different patches. The unit test at `:79` already shows their renderings differ. Optionally also check each stored file against its descriptor `sha256`.
