# Plan review: plot subfolders (plan ef6c62db, spec 9b832789)

Reviewer: plan-reviewer. Analysis only; no source, spec or plan file was edited.
Base checked: worktree at `ef6c62db` (code identical to `ba725001`).

## Summary verdict: FEASIBLE WITH CONCERNS

The design is sound and the code it names exists with the signatures it assumes. I found no blocker. The main risk is test fallout, not production logic. Changing the layout makes several existing assertions pass vacuously (`glob("*.png") == []`, `not (dir / "x").exists()`, `first == second` over empty lists). The plan's instruction to "update any remaining assertion that pinned the flat layout" only catches assertions that fail loudly. Beyond that, the plan misses several tests outside its 17-file set and some inside it, and a few spec §5 requirements have no test.

Counts (from this file): BLOCKER 0, IMPORTANT 8, MINOR 14.

Probe results are in "Verification".

---

## BLOCKER

None.

---

## IMPORTANT

### I1. Existing assertions that go vacuously green (Tasks 3, 4, 5; plan gives no sweep)
After the change every page lives one level deeper, so flat-depth negative and equality checks keep passing while testing nothing. The plan only says to update assertions that pin "flat filenames".

Concrete hits (all verified by reading):
- `tests/unit/plotting/test_store_figures_build.py:505`: `test_a_drawable_binding_is_redrawn_not_kept` does `for png in (...ApplyState).glob("*.png"): write tampered`. PNGs are now in `ApplyState/tiles/`, so the loop body never runs. `ApplyState(mode="draw")` is redrawn either way and equals `first`, so the test passes and no longer distinguishes redraw from keep. The sibling at `:468` (`[png] = ....glob("*.png")`) fails loudly, so the executor will fix that one and may fix only it.
- `tests/unit/plotting/test_store_copyout.py` `test_a_refused_guard_mid_page_leaves_no_half_page`: asserts `not (base / f"{_STEM}.plotly.json").exists()`. That path no longer exists in any layout, so `_discard_page` could be deleted and it would pass. It must look under the plot folder (`rglob`).
- `tests/unit/plotting/test_coordinator.py:126-130` (`..._stable_for_reruns`): `list(dir.glob("*.png"))` is `[]` both times, so `first == second` passes. Also `:717`, `:963` (`glob("*.png") == []`) and `:718` (`not (directory / "manifest.json").exists()`) are vacuous now. `:97,115,899` fail loudly.
- `tests/integration/plotting/test_publication_end_to_end.py:135` (`glob("*.png") == []`) and `:139` (`not (directory / "manifest.json").exists()`) are vacuous; the lines above them fail loudly, so an executor fixing the loud lines leaves these green and empty. Its `src="../../plotly.min.js"` at `:~108` becomes `../../../../`.
- `tests/unit/plotting/test_output_adapter.py:221` (`not (sym/"Only.png").exists()`) and `:242` (`not (plots_base/"sym"/"plotly.min.js").exists()`): vacuous / aimed at the wrong directory. The "no bundle per directory" intent now concerns `sym/<plot>/` too.
- Fix: add a mechanical sweep step to Tasks 3, 4 and 5: `grep -rn 'glob(\|exists()\|iterdir' <touched test files>`. Rule: every negative assertion becomes `rglob`; every `glob`-based positive assertion also asserts the result is non-empty. Name the hits above in the plan. Same for `first == second` comparisons.

### I2. Direct-publisher collision tests lose their meaning (Task 5)
`test_output_adapter.py:70-87` (`..._collision_safe_names`) and `:89-110` (`test_hash_suffix_is_rechecked_for_page_filename_collision`) use bare pages (`PlotPage("first", ..., label="A")`). Each bare page is now its own plot folder, so file names `first/A.png`, `preempted-suffix/A-xxxx.png`, `third/A.png` are unique without any digest logic. Both tests pass with `unique_page_stems` deleted. They need `plot="same"` on the pages to keep testing file-level collisions. The same gap exists for the store: Task 3 moves the collision test to folder level but no test has two pages of one plot whose stems collide (e.g. keys `"A b"` and `"a-b"` in plot `"tiles"`). `plot_page_paths`' per-plot loop is only tested with distinct keys.

### I3. Blast radius outside the plan's 17-file set, and grep that misses forms
Not mentioned anywhere in the plan; surfaced only at Task 8's full run (or never, for gui):
- `tests/unit/plotting/test_plot_meas_time_series.py:334-338` asserts `files["png"] == ["BY4741.png", ...]` and `destination / "BY4741.png"`. Breaks in Task 5. (`:127-147` uses `tmp_path/"manifest.json"`; still fine.)
- `tests/gui/results_viewer/test_mutation_guard.py:556-600` counts publication-guard calls (`checks == 3`, perturbation at the third). Task 5 adds a `_require_plot_publication` before each plot-folder `mkdir`, which shifts what the third call is (the mkdir guard instead of the PNG commit). The test stays green but stops exercising the commit-time guard. Either drop the extra guard call or update the count/trigger. This is a gui test the baseline never ran.
- Task 2 Step 4's grep (`"schema_version": 1\|...== 1\|"schema_version": 2`) does not match assignment forms. Missed: `tests/unit/sdk_/test_image_figures_store.py:260` (`_relabel_as_newer` sets `["schema_version"] = 2` as the "newer, unknown" layout; used by the tests at `:282` and `:295`, which will fail once 2 is known), and `tests/unit/plotting/test_store_copyout.py:341` (`d.update(schema_version=2)`) with `:346` (`"schema_version 2"`). Task 4 never mentions `:73` (`manifest["schema_version"] == 2`), `:279` (a v2 manifest literal), or the three `run_path("sym/default.plotly.json")` uses at `:125`, `:164`, `:352` (the first two write to a path that no longer exists, so the "tamper" is a no-op creating a stray file; `:352` fails loudly).
- Spec "Background" says 3 test files hard-code flat paths. Reality is at least 6 (add `test_image_figures_store.py` and `test_calibration_figure_in_store.py` is counted separately from `test_figures_in_store.py`). Not harmful to the plan, but the plan's inventory should be grep-driven, not spec-driven.
- Fix: widen the grep to `schema_version` generally (`grep -rn schema_version tests/unit/sdk_ tests/unit/plotting tests/unit/cli tests/integration`), add `tests/unit/plotting/test_plot_meas_time_series.py` and `tests/gui/results_viewer/test_mutation_guard.py` to Task 5, and add `tests/gui/results_viewer/test_mutation_guard.py::<plot refresh test>` to a gate run.

### I4. Copy-out `failed` entry for a page with no copyable file lacks `plot` (Task 4)
In `_store_copyout.py:277-279` the `if not files:` branch appends `{"key", "label", "error": "no stored file could be copied out"}`. Plan Step 3 adds `plot` to the `failed_only` entries and the page entries, but not to this branch. The spec requires `"plot"` on every `failed[]` entry. Add `"plot": page.get("plot")` and a test (tampered file in a foldered page).

### I5. Copy-out creates plot folders without a guard check (Task 4)
Plan: `page_directory.mkdir(exist_ok=True)` inside the per-file `try`, before `_atomic_write`. The existing code calls `_require_plot_publication` before `directory.mkdir`; the docstring of `publish_plot_output` states the contract "immediately before directory creation". With a guard that flips mid-publish, copy-out leaves an empty plot folder (the temp file is cleaned, the folder is not). The direct publisher in Task 5 does call the guard before its mkdir; copy-out should match. Pin with a test that asserts no directory remains after a refused guard (this is also the fixed form of the vacuous test in I1).

### I6. Spec §5 / Review Focus items with no real test
- Review Focus 5 claims the failure keeps its `plot` "through the descriptor and into the manifest". The Task 4 test starts from a hand-built `StoredFigureFailure(plot="tiles")`. Nothing asserts that `_build_pages`/`_build_page` stamp `plot=page.plot_name` (Task 3 changes the code; `test_hand_built_pages...` keeps its failure assertions "unchanged"). Deleting those `plot=` arguments passes every test. Add `failure.plot == "odd"` to the hand-built test, and a build -> `write_image_figures` -> `read_figure_run` check that the descriptor's failure carries it.
- Spec §5 "a rerun that switches a page's format removes that page's old sibling inside its plot folder" (copy-out `_remove_leftovers`, and direct `_remove_stale_sibling`): the plan rewrites neither `test_a_republished_page_loses_its_leftover_renderings` (it seeds `base/<stem>.png`, which now would simply remain and fail the test) nor adds a foldered equivalent, and adds nothing for the direct publisher.
- Spec §5 `_kept_binding` "still refuses a binding spread over two binding folders": no existing test (`grep "one directory"` finds none) and no new one.
- P3 (`"plot": null` for kept flat pages) is asserted only on the in-memory page, not on what `write_image_figures` writes for it. Round-trip it through `figure_store` and assert the file is flat and `plot` is `null`.

### I7. Spec §6 changelog has no task; docs left stale (Task 8)
The spec coverage table maps §6 to Task 8, but Task 8 has no changelog step, and the repo has no changelog file (`find` for changelog/NEWS/release notes returns nothing). Decide where it goes or drop it from the spec. In `docs/source/extending/pages/custom_plotter.md` the plan replaces the table rows (227-229 are indeed the three data rows) but leaves false statements: `:235` ("`PlotColonyArea` publishes `plots/PlotColonyArea/default.html`" becomes `default/default.html`), the sentence just above it ("A page's filename comes from its `label`, or its `key`", false for store copy-outs under P1), `:332-333` ("stores `PlotColonySizes/default.plotly.json`" becomes `default/default.plotly.json`), and the manifest remarks around `:395`/`:421`.

### I8. Task 6 `test_one_roi_stores_one_overlay` is conditional (Task 6)
The plan says "If one ROI is refused by `min_patches`, pass `on_qc_fail="skip"`". `min_patches` defaults to 20 and `on_qc_fail` to `"raise"` (`_calibrate_color_rpcc.py:188,190`); one band is 12 patches. The plan should state which form the test uses rather than leave the executor to discover it. Probe result: with the default `on_qc_fail="raise"`, a one-ROI `apply` raises `RuntimeError`, and `on_qc_fail="skip"` does not avoid it. The cause is not a QC refusal: `_checker_qc.py:271` `require_rank` raises `ValueError` ("a degree-3 root-polynomial fit needs at least 13 patches but only 12 were measured"), re-raised as `RuntimeError`. The plan's hedge (`on_qc_fail="skip"`) is therefore wrong. The one-ROI test needs `degree` lowered (e.g. `degree=2`, if 12 patches suffice) or a different fixture, and the plan should say which. The replacement test also drops the existing `_page_pngs(operation.inspect()) == expected` "no subject: the held image" assertion; keep it.

---

## MINOR

1. Reserved-name collisions at the new level. A plot folder named `zarr.json` collides with the binding's group document (the write would raise `FileExistsError`/`NotADirectoryError` out of `write_image_figures`, which can fail the whole store write, not just a figure). A plot named `manifest.json` collides with the manifest file in deliverables. Cheap fix: pre-reserve both names (casefolded) in `plot_page_paths`' folder pass so they get the digest suffix.
2. `safe_path_component` raises for `/` and `unique_page_stems` falls back to `"page"`, so plots `"a/b"` and `"c/d"` become `page` and `page-<digest>`. Safe but surprising; consider refusing `/` in `PlotPage.plot` (the spec only requires non-empty).
3. Direct publisher, time-series: the folder is derived from the page key (a canonical-group-key JSON string, e.g. `strain-str-BY4741`) while the file keeps the label (`BY4741.png`), giving `strain-str-BY4741/BY4741.png`. Conforms to D2/spec, but the folder is the less readable of the two. Question for the user; no change needed to satisfy the spec.
4. P1 verdict: accept. Mirroring the store file for file is the literal reading of D4, and D3 already says labels do not name files. Costs to document: (a) label-named deliverables from earlier runs (`Tile-overlay.png`) remain beside the new `tiles/roi_0.png` on a re-copy of the same image (same "dropped files are not swept" limitation the spec accepts, but P1 renames files of unchanged plots too); (b) the two writers now use different naming rules inside one `deliverables/plots/` tree; say so in `custom_plotter.md`.
5. A v1 lone `default` page used to publish as `<dataset>/<stem>.<ext>` and now publishes under `<dataset>/<stem>/default.<ext>` with a manifest (flat special case removed for every page). The spec says "copied into the image folder, as today", which is true only for multi-page v1 bindings. Worth one sentence in the spec/doc.
6. The test that models "as 0.19 wrote it" (`test_a_v1_store_gains_a_v2_run...`) starts from a store written by the new writer, so its entries carry `"plot": null` keys that 0.19 never wrote. Strip the `plot` keys from pages and failures so the carried v1 entry has the real v1 shape (`_flat_v1_store` already does this for pages).
7. `known_figures_schema` and `carry_figure_runs` warnings still print `ngff_.FIGURES_SCHEMA_VERSION` as "what this writer knows" (`_image_figures.py:368,493`, `_measurement_tables.py:790` wording). With readable {1,2} the message should print the readable set.
8. `.failures.jsonl` records carry `page` but no `plot`, so a failure for `roi_0` is ambiguous when two plots share a key. Acceptable; note it.
9. Task 5: a page that renders nothing leaves an empty plot folder (`page_directory.mkdir` precedes the render). Review Focus 4 only covers the store; consider creating the folder lazily.
10. Task 7 Step 2 duplicates `test_figures_in_store.py::test_process_mode_carries_figures_only_in_a_store` (zarr and tiff, asserts no `deliverables`). Fold the per-ROI assertions into the existing calibration process test instead of adding a third.
11. Task 3 leaves integration tests red until Task 7 (`test_figures_in_store.py:136` pins `sym/default.plotly.json`). Fine for gate A as the plan lists it, but it makes the Task 3 commit un-bisectable; updating that one path in Task 3 is one line.
12. P2: `plot_name` is a new public property on a public dataclass; the spec called for a private helper function. Harmless; note the deviation in the PR.
13. Windows: plot folders add one path component to `plots/<binding>/<dataset>/<stem>/<plot>/<file>`; the store writer uses `long_path`, copy-out does not. Low risk, flag for the cross-platform checklist.
14. Task 3's new test block repeats `import json` and a module-level `figure_store` import beside the file's function-local import style; ruff may flag the duplicate `json`.

---

## Validated aspects

- Names exist as assumed: `_SameRunKeeper.keep`, `_kept_binding`, `_read_stored_file` (`_store_figures.py:317,355`, `_store_copyout.py:316`), `_publish_binding`, `_publish_pages`, `_remove_leftovers`, `_write_html_from_json`, `_atomic_write`, `_commit_manifest`, `_render_page` returning `(files, errors, backend)`, `plotlyjs_src_for(page_dir, bundle)` (lexical relpath, so the deeper HTML `src` is computed correctly: copy-out gives `../../../../plotly.min.js`, direct publish `../../plotly.min.js`), `figure_store`, `emit_image_via_store`, tests' `_build`, `_keep`, `_files`, `ApplyState`, `TEST_RUN`, `_write_inputs`, `_run_of`, `DAY`, `band_rois`, `band_prior`, `frozen_op`, `RoiOverlay.roi_index`, `CalibrationOverlayRecord` fields. Dataclass field order matches the plan's positional constructions (`StoredFigureBinding(binding_id, plot_class, directory, pages)`, appended defaulted `plot`/`directory`).
- `n_fitted`/`n_expected` are explicit record fields, so `record.model_copy(update={"rois": [roi]})` keeps the frame-level title. The model is frozen but `model_copy(update=...)` is the supported route.
- `split_figure_file_path`: reasoning from `PurePosixPath` semantics agrees with the plan's table: `./` and `//` and a trailing slash make `len(raw) != len(parts)`, an absolute path has `parts[0] == "/"`, `..` is caught by the explicit check. All nine cases confirmed by the probe if `main` returns it.
- Version gate: `type(version) is int` is needed, since `True == 1` and `True in frozenset({1, 2})` are both true. The boolean test would fail without it. Upgrade path `carry_figure_runs` -> `_fragment` (stamps the writer's version) -> `apply_image_figures_attributes` (takes `incoming["schema_version"]`) correctly turns a v1 descriptor into v2 and leaves carried entries untouched. A fragment of `None` leaves a v1 descriptor v1, which is fine.
- Older-writer safety is real: the shipped gate is `descriptor.get("schema_version") == FIGURES_SCHEMA_VERSION` (`_image_figures.py:517`) and also guards `_measurement_tables.py:783` and `_image_io_handler.py:1460`, and `carry_figure_runs` carries an unknown version's tree whole. `test_a_v1_only_writer_adds_no_run_to_a_v2_store` would fail before the change (the monkeypatched `READABLE_FIGURES_SCHEMA_VERSIONS` attribute does not exist) and the function-scope `from . import ngff_` lookups make the monkeypatch effective.
- `(plot is None) != (None in page_directories)` truth table checked: it refuses a foldered path with a null plot and a flat path with a string plot, accepts both consistent forms; a page with no files cannot occur (writer omits it).
- `_kept_binding` keeping the single-binding-folder check, and `read_figure_run` raising for version 3 (copy-out records `<store>`; keeper turns it into a binding failure), cover the "unknown version 3" and "mixed v1/v2 run" cases. Mixed runs in one descriptor work because each run's pages carry their own paths.
- Task order: Task 3's new layout does not break the old copy-out (it reads by descriptor path and names files itself), so gate A's expected-failure list is accurate. Gate B's list is right.
- Nothing in `src/` reads plot manifests or `schema_version` of a deliverables manifest; `_gui` only passes `plots_dir` to the coordinator. `zarr.consolidate_metadata` recurses, so group documents in each plot folder are what the consolidated root needs; the plan creates them. AutoConvertRaw-GC's `verify_store` (`src/acr_core/process.py:125-142`) is descriptor-path driven and ignores `schema_version`, so it needs no change; its failed-entry check iterates every run, v1 runs included.
- Concurrency: no new shared state. Plot folders are created inside the existing `.publication.lock`; writer temp-file naming is per-call unique.
- `PlotOutput` uniqueness over `(plot_name, key)` and the field-not-key-convention choice (keys from `canonical_group_key` may contain `/`) are right. Task 1 tests would each fail for the wrong implementation.

## Verification

Read and checked: spec, plan, `_writer.py`, `_store_figures.py`, `_store_copyout.py`, `_image_figures.py`, `_output.py` (both), `_calibration_overlay.py`, `_calibrate_color_rpcc.py`, `_measurement_tables.py:640-800`, and the tests named above, plus greps across `src/`, `tests/` (including `tests/gui`) and the AutoConvertRaw-GC checkout.

Probe run by `main` (40 s, warning lines filtered):
- `split_figure_file_path` logic: all nine cases behave as the plan intends (the two layouts accepted; `./`, `//`, trailing slash, absolute, `..`, depth 3 and 6 refused).
- `model_copy(update={"rois": [roi]})` renders; `n_fitted`/`n_expected` stay 24/24; two renders of the same record give identical PNG bytes; a one-ROI overlay differs from the combined one. Task 6's byte-equality assertions are sound.
- One-ROI `apply` raises `RuntimeError` under both default and `on_qc_fail="skip"`; root cause is the degree-3 rank check (12 patches < 13). See I8.
- Whole-store determinism: two process runs into different output dirs give identical 13-file trees, and no `deliverables/`. Task 7's determinism test is therefore valid as written (my earlier concern is withdrawn).
- `unique_page_stems([("zarr.json","zarr.json")])` returns `zarr.json` unchanged, so minor 1 (plot folder colliding with the group document) is real; `a/b`, `c/d` give `page`, `page-e5fb6071` (minor 2 confirmed); the time-series key sanitizes to `strain-str-BY4741` (minor 3 as stated).

Not verified by execution: any pytest result; the claim that vacuous tests pass (derived from reading; each is a direct consequence of the file paths).

## Questions for clarification

1. Time-series (and any bare-page aggregate) plots: is a folder named from the key and a file named from the label what you want (minor 3), or should a bare page's folder prefer its label in the direct publisher?
2. Where should the changelog entry go (I7)? There is no changelog file in the repo.
3. Should `PlotPage.plot` refuse `/` (minor 2)?

## Possible bugs in existing code (not fixed)

- `unique_page_stems` swallows every exception from `safe_path_component` with a bare `except Exception` and substitutes `"page"`, so a key like `a/b` silently loses its identity in the file name (pre-existing; the new folder level reuses it).
