# Plan review: per-image figures in the OME-Zarr store

- **Reviewed:** `plans/2026-09-22-figures-in-ome-zarr/plan.md` (commit `e3df9c42`) against the spec and the worktree code.
- **Method:** I read every file the plan cites, plus its callers. Anything I did not verify is marked **UNVERIFIED**. The probe I requested separately (consolidation with non-Zarr files under a nested group; `prepare_image_tables` on a one-column frame; the binding id for `plots=[sym]`) is not reflected in this report.
- **Tally:** 1 BLOCKER, 8 MAJOR, 14 MINOR.

## What checks out (verified)

- **Cited locations are all correct.** Every function and line reference the plan cites exists as stated, within a few lines:
  - the call sites and store writers: `_cli_process_single.py:350,452`, `_cli_staged_workers.py:583`; `_image_io_handler.py` with `save2zarr` at 1068, `_save_store` at 1128, `_write_store_part` at 1212, the tables block at 1377 and the root at 1391;
  - the measurement-table functions: `_measurement_tables.py` with `_rewrite_store_tables` at 632 and `replace_image_tables` at 698; `_cli_output_manager.py` with `save_image_store` at 1842 and `replace_image_store_measurements` at 1941;
  - the process and continuation code: `write_process_only_layer` at `_cli_process_only.py:152`; `PROCESS_LAYER_SEMANTICS_REVISION` at `_cli_failure_tracker.py:205`;
  - the plotting pipeline: `_declared_backends` at `_backends.py:221`; `_image_output_stem` at `_coordinator.py:530`; `record_plot_failure` at `_failures.py:41`; `figure_backend_of` at `abc_/plotting/_output.py:14`; `ngff_.PhenotypicAttr` at 449 (`TABLES` at 464); `ngff_.read_phenotypic_attributes` at 646; `ngff_.STORE_ROOT_JSON` at 66.
- **`plotly.io.to_html` accepts `div_id`**, and the CDN script comes from `cdn.plot.ly`. The SRI hash is derived from the bundled js, so it is deterministic (plotly 6.6.0, `plotly/io/_html.py:47,143,257-266`). `fig.to_json()` defaults to `remove_uids=True` (`plotly/io/_json.py:175`).
- **`plotly_get_chrome -y` is a real console script and flag** (`plotly/io/_kaleido.py:806-832`).
- **Declaring the spec via `inspect` is consistent with today's classification.** `MeasureSymZones.inspect` (`_measure_symzones.py:515`) and `MeasureOrientationZones.inspect` (`_measure_orientation_zones.py:1982`) are themselves `@figure`-decorated. `declared_figure_spec` step 1 therefore picks up a `store=` declared there, consistent with `_declared_backends`.
- **The cache-hit path is real.** `MeasureFeatures.measure` passes the same image object to `_operate` (`abc_/_measure_features.py:451`), so Task 10's first render does hit the cache.
- **`write_provenance_checkpoint` rewrites only `provenance`/`work_id` in an existing root** (`_core/_provenance.py:1062-1094`). `_mark_failed_checkpoint` after a post-promotion copy-out failure therefore keeps `figures`.
- **Binding ids cannot collide on disk.** `normalize_plot_bindings` rejects ids that collide after `safe_path_component` + casefold (`_bindings.py:230-246`).
- **No other store rewrite path drops figures.** The only writers are the forward writer, the measure-mode replace, migrate's `replace_embedded_measurement_table` (which hard-links `figures/` across), and root-only provenance rewrites.
- **The layering is clean.** No new `sdk_`->`abc_`/`plotting` import is introduced. `abc_/plotting/_store_formats.py` is stdlib-only, as `tests/unit/abc_/plotting/test_imports.py` requires. `_store_copyout`->`_coordinator` is a lazy import, so there is no cycle.
- **The one new staged caller of `write_process_only_layer` is fine.** It is at `_cli_staged_strategy.py:529` and is objmap/tiff only, so it never needs figures.

---

## BLOCKER

### B1. Task 9 Step 8's test-porting instructions fail as written

The plan says to swap each `emit_image` call for `_emit_image_via_store` and "keep the assertions". Checked against `tests/unit/plotting/test_coordinator.py`:

1. **Repeated emits for one stem crash.** `figure_store` calls `store.mkdir()` without `exist_ok`. Any second emit for the same stem raises `FileExistsError`: `test_image_plot_output_name_is_stable_for_reruns` (:137-147) and every rerun test at :1034-1093. The `plate 1`/`plate-1`/`Plate-1` loops (:119-134, :150-166) create `store-plate-1` and `store-Plate-1`, which collide on the case-insensitive macOS nightly lane.
2. **The store lands inside the tree the test asserts is empty or PNG-free.** The helper writes it under `tmp_path`. `test_a_refused_publication_guard_propagates_and_writes_nothing` asserts `sorted(tmp_path.rglob("*")) == []` (:835). `test_the_flat_path_rechecks_the_publication_guard_before_commit` asserts `list(tmp_path.rglob("*.png")) == []` (:943), but the mpl `_ImagePlot` store now holds `figures/image/default.png`. The plan says these "hold unchanged". They do not.
3. **The stale-sibling tests' premises are false now.** `test_a_rerun_without_chrome_removes_the_previous_png` (:1022-1042), `test_a_failed_flat_rerun_keeps_both_previous_renderings` (:1045-1070) and `test_a_multi_page_image_rerun_without_chrome_removes_the_previous_pngs` (:1220-1240) need run 1 to write a PNG when `chrome_available()` is True. An undeclared Plotly page now stores `plotly-json` only. `_failing_html` patches `FigureAdapter.save_html`, which the copy-out never calls; it calls `figure.write_html`. `test_a_partial_flat_render_publishes_what_it_can_and_records_once` (:966-995) expects an `OSError` from a PNG that is never attempted, and it is not in the plan's list. These need a rewrite (seed the leftovers by hand, as Task 7's leftover test does), not a call swap.
4. **The failure-record tests fail on class and prefix.** `test_every_emit_point_records_one_failure_under_a_closed_lifecycle[emit_image]` asserts `entry["plot_class"] == type(plot).__name__` (:612), but the copy-out records `"<stored>"` (M1). `test_a_flat_image_render_failure_records_the_real_exception_class` is told to assert `startswith("<Class>: ")`, but `_located` prefixes `"[page=default] "` (M8).
5. **The plan's line list is wrong in places.** :954 is `test_a_strict_flat_failure_keeps_the_renderer_as_its_cause`, which uses `strict=True`. It is listed as a guard test to route through the helper, but it should be deleted along with :105.

**Fix:**
- Give the helper a unique store directory per call, outside the asserted tree (for example `tmp_path_factory.mktemp`), or narrow the "writes nothing" assertions to `plots_dir(tmp_path)`.
- Enumerate each affected test with its new assertions.
- Resolve M1 and M8 first.

## MAJOR

### M1. The copy-out loses `plot_class` for every stored failure

Descriptor `failed` entries carry no class (spec §1), and failed bindings are absent from `bindings`. So `publish_store_figures` records `plot_class="<stored>"`. Today `.failures.jsonl` records the real class (`_coordinator.py:382-392`), and a test pins it (`test_coordinator.py:612`).

**Fix:** have `PlotCoordinator.publish_store_figures` pass a `{binding_id: class}` map built from `self._pipeline.get_plots()`. This needs no spec change. Adding `class` to the failed entries would work too, but that is a spec change and needs user approval.

### M2. The hard-link test cannot fail for the reason it claims

`test_measure_rebuild_replaces_figures_without_touching_live_bytes` writes the old generation as binding `gone` and the new one as `kept`.
- The held file descriptor points at `figures/gone/...`, which nothing writes to. `os.pread == b"old"` therefore holds with or without the `rmtree`.
- The Step 7 mutation does fail the test, but only through the `gone/` exists assertion, which is the stale-directory property.

The real hazard: `write_image_figures` uses `Path.write_bytes`, which opens with `O_TRUNC` on the inode, while `_rewrite_store_tables` copies with `os.link` (`_measurement_tables.py:597-603,669-672`). A same-name rebuild without the `rmtree` writes into the LIVE store, and nothing tests that.

**Fix:** add a same-binding rebuild (old `sym` -> new `sym`) and assert that the held descriptor still reads the old bytes. Better still, write figure files via temp + `os.replace`.

### M3. That Task 6 test fails on the Windows nightly lane

- `os.pread` does not exist on Windows.
- An open handle held across `promote_store`'s renames raises `PermissionError`.
- `tests/unit/sdk_` is in shard `tune-sdk`, which `run-pytest-full.yml` runs on `windows-latest`.

**Fix:** `skipif(sys.platform == "win32")` on the file-descriptor half.

### M4. Page metadata can abort a store write (the plan's addition #1 needs a guard)

`PlotPage.metadata` is typed JSON-native but not enforced (`abc_/plotting/_output.py:52-54`).
- **Measure mode:** the root is written with `atomic_write_json`, with no `default=` (`_measurement_tables.py:684`, `_atomic_io.py:230-236`). A `numpy.int64` value raises `TypeError`, which fails the replace and the image. Today the same value only fails `emit_image`'s swallowed manifest write.
- **Full and process modes:** `_write_group_json` uses `default=str` (`_image_io_handler.py:975-984`), so the value is silently turned into a string.

**Fix:** in `_build_pages`, `json.dumps(dict(page.metadata))` inside the per-page try, and record a failure.

### M5. Declaring both `plotly-json` and `html` makes the copy-out collide

- The copy-out generates `<stem>.html` (hoisted bundle) from the `plotly-json`, then copies the stored `<stem>.html` (CDN) over it, or the reverse. The winner depends on `store` order, and `files["html"]` is assigned twice.
- The spec's §3 rules ("stored files verbatim" versus "HTML generated from plotly-json") contradict each other here.

**Fix:** pick one rule and test it.

### M6. Spec §5 row "Measure mode: figures written before the root" has no test

Nothing distinguishes "written in the part before the root" from "written after promotion", because the final trees are identical.

**Fix:** spy on `ngff_.promote_store` and assert that the part already holds the figure bytes matching the descriptor's `sha256`.

### M7. The integration-file ports drop the `strict=True` failure signal

`tests/integration/plotting/test_publication_end_to_end.py` :96, :122 and :177 use `strict=True` to surface a failure. Without it, a failed build leaves only a descriptor `failed` entry, and file assertions pass on less output.

**Fix:** each port asserts `stored.failed == ()`.

### M8. Undeclared change to the `.failures.jsonl` `error` format

- `_located()` prepends `"[page=.. format=..] "` to stored and copy-out failures.
- Today `error` always starts with the exception class (`_failures.py:21-37`), and tests rely on that (:776).
- This change is not among the plan's three "additions".

**Fix:** use separate JSONL fields for page and format (a declared schema addition), or keep `error` verbatim.

## MINOR

1. **Measure mode rewrites the root with sorted keys.** `atomic_write_json` defaults to `sort_keys=True` (`_atomic_io.py:214`). After `--mode measure`, `bindings` is in sorted order, not pipeline order, which contradicts spec §1 Ordering. Consider a list, or an explicit `order` key.
2. **Several copy-out raise points sit outside any try.** `_image_output_stem(...)` (`safe_path_component` can raise `ValueError`) and the `failure[...]` key lookups run before any try. After promotion, a raise fails the image through `_mark_failed_checkpoint`, contradicting "never raised".
3. **Part of `build_image_figures` sits outside the per-binding try.** `_build_pages` runs `normalize_plot_output`, `declared_figure_spec` (which catches only `RuntimeError`) and `unique_page_stems` outside it, so an unexpected error fails the image. Old `emit_image` wrapped everything.
4. **`inspect()` returning `None` makes a binding vanish.** It yields zero pages, so the binding is in neither `bindings` nor `failed`, which reads as "not configured". Record it as a `page:null` failure.
5. **Process mode builds figures after the status is closed.** Task 9 builds after `set_provenance_status(image, "complete")`; full mode builds before it. A provider whose `inspect` applies an `ImageOperation` to its subject would, in process mode only, append a `programmatic` application to the published store's journal (`_provenance.py:639-665`). No in-tree provider does this (`PlotDetectModes` uses `mode.compute`). Build before closing, for parity.
6. **The copy-out records descriptor failures before the first `publication_guard` check,** contrary to the rule at `_coordinator.py:112-117`.
7. **The copy-out drops the `.publication.lock`** that `publish_plot_output` takes for manifest directories (`_writer.py:317-319`).
8. **The manifest's `renderers` becomes an outcome field** ("png available when a page carries a PNG"). `_writer.py:420-425` documents it as a capability and warns against exactly this. `partial` is also dropped.
9. **Task 3 names a constant that does not exist.** It says `SHARDS`; the real constant is `MANIFEST` (`test_pytest_shard_manifest.py:15`). The plan hedges with a grep.
10. **Task 6 imports `prepare_image_tables` from the wrong module.** It lives in `phenotypic._cli._embedded_measurement_tables`, not `sdk_._measurement_tables`. The plan hedges. Whether the one-column frame is accepted is UNVERIFIED (probe requested).
11. **The Chrome lane (UNVERIFIED on CI):** GitHub `ubuntu-latest` ships Google Chrome, so the marker may already run there. Kaleido launching under Ubuntu 24.04's AppArmor userns restriction is also unverified; only a CI dry run settles either. `requires_kaleido_chrome` currently has no users (only a docstring at `_backends.py:50`). It is still worth making it strict.
12. **Cross-process determinism of real provider figures is untested.** Task 10's byte-identity test runs both runs in one interpreter, and Task 3's cross-process test uses a trivial scatter. Suggest two subprocesses with different `PYTHONHASHSEED` for the process-store test.
13. **The spec's "KB-sized plotly-json" is false for the zone measurers.** `plotly_imshow` uses `px.imshow(binary_string=True)` (`_accessor_dash_handler.py:125-131`), which embeds a full-resolution PNG data URI, so it is MBs per image, in the store and again in deliverables. Don't repeat the claim in the Task 11 docs.
14. **Smaller undeclared additions and oddities:**
    - collision suffixes in store filenames (beyond spec §1's `safe_path_component(page.key)`);
    - the preflight rework (Task 8) is not in the spec;
    - `store="png"` (a str) splits into `('p','n','g')`, so reject a str explicitly;
    - figure paths skip `ngff_.long_path`, so long names can cross Windows `MAX_PATH`. The tables writer has the same gap.

## The three "additions to the spec"

1. **`metadata` on descriptor pages: justified.** Manifest v2 needs it (`_writer.py:410-416`). Add the M4 guard.
2. **Module-level `build_image_figures`: justified.** Process mode has no `plots_base`.
3. **`PlotBackendUnavailable:` spelling: justified.** The class exists at `_backends.py:159`.

Unlisted additions: the `_located` prefix (M8), the `renderers` semantics (minor 8), the filename collision suffix and the preflight change (minor 14).

## Chrome lane (question 6)

- The serializer test is in `tests/unit/plotting/`, which belongs to shard `plots-post-viz`, and the PR workflow iterates it (`run-pytest.yml:87-137`).
- With `PHENOTYPIC_REQUIRE_CHROME=1`, the marker stops skipping, so the Chrome case runs, or fails, on Linux py3.11 and py3.12.
- The shard keys are read only by `run-pytest.yml`, `run-pytest-full.yml` and `test_pytest_shard_manifest.py`. An added `chrome` key breaks none of them.
- The nightly lane does no install, so the case skips there. Caveats are in minor 11.

## Probe results (run by the orchestrator after this report was drafted)

- Consolidating a process store with `figures/` + `figures/sym` groups holding non-Zarr files: keys `['OME', 'figures', 'figures/sym', 'rgb', 'rgb/0', 'rgb/1']`; no warnings escape.
- `prepare_image_tables(pd.DataFrame({"Object_Label":[1]}), None)` -> `PreparedImageTables` (accepted).
- `ImagePipeline(ops=..., meas={"sym": sym}, plots=[sym])` -> binding ids `['sym']`.
