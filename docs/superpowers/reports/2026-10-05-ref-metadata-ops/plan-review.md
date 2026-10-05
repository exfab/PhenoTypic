# Plan review — ref-metadata-ops (HEAD d121e12d)

Reviewer: `xander-local:plan-reviewer` (independent, analysis-only). Report
transcribed verbatim by the orchestrator from the reviewer's messages; the
first message was truncated in transit after M3 and the remainder was
re-requested.

## Probes the reviewer requested (run by the orchestrator, output verbatim)

```
PNG file does not have exif data.
polars 1.41.2 pydantic 2.12.5
csv [String, String] [('000123', None), ('t04', 't00')]
norm ['Metadata_ImageName', 'Metadata_BlankImage', 'Metadata_Grid_RowNum']
mixin init_subclass reached for Probe
metadata [_ColumnRefMarker('measurements'), <__main__._M object at 0x7f0cf80a9190>]
synth GridImage gray 8 Synthetic96PlateWithObjects True
manifest bytes 2967049 parse ms 19.3736151792109
```

```
$ uv run python -c "import sys, phenotypic._core; print(sorted(m for m in ('pandas','polars','scipy','skimage','pyarrow') if m in sys.modules))"
['pandas', 'polars', 'pyarrow', 'scipy', 'skimage']
```

## Plan Feasibility Analysis: ref-metadata-ops (HEAD d121e12d)

### Verdict
**Not ready** — one blocker and seven majors require plan fixes. The core design (contextvars context, fieldless mixin, `_ColumnRefMarker` reuse, run-level manifest, per-image work-id digest) is sound against the code; the defects are in CLI coverage of staged runs, naming, scale, and several test assertions that will not hold.

### Findings

**B1 — blocker — SPEC CHALLENGE (§5.2 "lands in Stage 1 by construction"); Task 6 Step 5.** A `SubtractBlank` that is a pre-detector sibling *inside the GPU detector's own sequence container* (exactly the ucr_033 placement if the detector is a `GpuDetector`) is moved into `stage2_prefix` by `_branch_prefix` (`_cli_pipeline_split.py:84-97`) and applied in Stage 2 by `_apply_stage2_prefix` (`_cli_staged_workers.py:146-194`), called at `:457-458` — outside any try, with no context → every image raises `RefMetadataUnavailableError` (as an unclassified, non-`PerImageScientificError` exception). The `--mode process --layer objmap` export also re-applies the residual pipeline with no context (`_cli_staged_strategy.py:518-522`). **Fix:** wrap both sites in `worker_reference_context(output_dir, dataset_name)` (both have them in scope), extend the source-inspection tripwire to `stage2_detect_core` and `_export_objmap_layer`, and add a staged test with `SubtractBlank` inside the detector's branch.

**M1 — major — Tasks 4/6: planner keys `Path(p).stem`, runtime looks up `image.name`.** For a `.ome.zarr` input `imread` names the image `store_stem` (`_image_io_handler.py:886-892`): `x.ome.zarr` → `x`, but `Path("x.ome.zarr").stem` → `x.ome`. A tree of stores is documented-valid `--input`; every image is then "unmatched" → preflight escalates `PF-REF-UNMATCHED` to error and refuses, and digests are `"unplanned"`. The CLI everywhere else uses `source_image_stem` (`sdk_/_io_constants.py:1887-1903`; e.g. `_cli_state_management.py:610`, `_cli_process_single.py:225`). `resolve_image` also filters `p.is_file()`, so a blank that is a store never resolves. **Fix:** use `source_image_stem` in `plan_references`, `work_id_for_image`, `_worker_work_identity`; accept `is_zarr_store_name` directories in `resolve_image`.

**M2 — major — Task 6 Step 3: `reference_digest_for` re-reads and re-parses the whole manifest per work-id.** Probe: a 2.97 MB / 34,500-entry manifest parses in 19.4 ms. `work_id_for_image` runs per image in several main-process passes (`phenotypicCLI.py:768, :854, :3072, :3322`; `_cli_state_management.py:593, :673`; `_cli_slurm_array_scripts.py:395`; staged strategy) → ~11 min per pass at that scale, several passes per startup, quadratic growth; workers likewise re-read the full manifest per image. **Fix:** memoise the parsed manifest keyed on `(path, st_mtime_ns, st_size)` (or load once and pass the mapping); optionally split per dataset.

**M3 — major — Task 3: two tests assert the wrong exception type.** `ImagePipelineCore._run_operations` re-raises any op exception as `RuntimeError("[Op] (step …): …") from exc` (`_image_pipeline_core.py:934-941`). `test_refuses_after_an_enhancer` and `test_refuses_after_a_corrector` call `pipe.apply` under `pytest.raises(StaleDetectMatError)` and will fail. **Fix:** assert `RuntimeError` and `isinstance(exc.__cause__, StaleDetectMatError)`; note in the failure catalogue that pipeline users see the wrapped error.


**M4 — major — Task 3 / Phase 1 gate: an existing gate test will go red.** `tests/unit/enhance/test_detect_mat_invariant.py:26-54` constructs every enhancer in `enhance.__all__` with no args and applies it to `load_synth_yeast_plate()` with no context, and explicitly forbids skipping (`test_gate_covers_every_public_enhancer`). `SubtractBlank` will raise there. **Fix:** a step updating that gate to apply `RefMetadata` subclasses inside a `ReferenceContext(images={...})` (blank under a different name) — don't weaken the coverage assertion. Also add `SubtractBlank` to `TAXONOMY` in `test_enhancer_taxonomy.py`.

**M5 — major — SPEC CHALLENGE (§9 startup guard); Task 1: the "attribute access stays light" test cannot pass.** Probe 2: `import phenotypic._core` alone loads pandas/polars/pyarrow/scipy/skimage (`_core/__init__.py` imports `_image_parts` and `_pipeline_parts`), so any module under `_core` makes `phenotypic.ReferenceContext` access heavy — exactly as `phenotypic.Image` access already is. `import phenotypic` itself stays light (no risk to `test_startup_imports`). **Fix:** drop `test_attribute_access_does_not_import_polars_or_pandas` and the matching spec §9 clause (or move the module outside `_core`, not worth it).

**M6 — major — Task 8 Step 7: inspector anchor is wrong.** `builder/_callbacks.py:4017` is inside `_render_views(state)` (`:3943`), a module helper with 11 call sites, each its own callback — not one inspector callback that can take a new `State`. Picking a table also triggers no inspector re-render, so the dropdown never appears until an unrelated edit. **Fix:** read the picked path via a single module/session accessor or add one callback that re-renders the inspector on `STORE_REFERENCE_METADATA_PATH` change; don't edit 11 callbacks.

**M7 — major — tests: spec requirements with no test.** §9 "guard passes inside `CompositeEnhance`/`CompositeDetector` branches" (the ucr_033 placement) — no Task 3 test; staged-path context (see B1); §9 "two datasets" in the CLI run — Task 7 uses one (same-named blanks like `t00` in two dataset dirs are what proves per-dataset scoping); §8/§9 bit-depth refusal — no test.

**Minor**
- Measure mode deletes the manifest: `publish_reference_inputs` removes it when `measure_only`, and measure reaches `_prepare_incremental_startup` (`phenotypicCLI.py:2835`); a measure run beside live forward workers strips their context. Only remove in forward modes whose pipeline needs no refs.
- Process-mode "--metadata is ignored" warning (`phenotypicCLI.py:2064-2079`) is not updated.
- Process-mode bad table + `--skip-validation`: the early table parse is skipped for process (`:2355`), so a bad table fails inside `publish_reference_inputs` after the `--overwrite` rmtree and `mint_run_identity`; the CLI guide lists the `--metadata` parse among un-skippable refusals. Parse the reference table early whenever the pipeline needs it.
- Preflight can approve a snapshot fallback that `--overwrite` then deletes → startup failure.
- Severity escalation is per-code, not overall: half unmatched + half unresolved = every image fails, yet both stay warnings.
- Preflight uses `pipeline.reference_columns()` (whole tree) instead of `operations_in_scope` as the module rule requires (`_cli/CLAUDE.md` "Mode scoping").
- Spec §4.1 says an `images=` `Image` is "returned as-is (copied)"; plan returns the shared instance and tests `is blank`.
- `_IMAGE_CACHE` has no lock (Dash runs threaded Werkzeug); 4 full images per loky worker is sizeable at `--njobs` 32.
- `_RESOLVED` stale record: an entry survives an `_operate` that raises after `_ref_values`, and `_ref_image` uses `setdefault`. Reset at the start of `_ref_values` and pop in a `finally`.
- Corrector check reads only `applications[-1]`: in Stage 2 the probe copy opens a fresh "programmatic" application (`_cli_staged_workers.py:177-193` with `_provenance.py:651-667`), so a Stage-1 corrector is invisible there but visible in Stage 3 — GPU time wasted, then refused.
- State-tracking rule: `_cli/CLAUDE.md` requires updating `docs/source/contrib_guide/tracked_state.md` for a new `.phenotypic/` artifact (the manifest); cite the `_PRESERVED_ON_RESTART` membership rule (`_io_constants.py:1282-1312`) when adding the snapshot.
- Tune refusal is in `TuningEngine.__init__`, spec says "at spec load"; a `TuningSpec` validator would catch it before any fan-out.
- Run console: `click_action`'s repaint-race guard (pattern at `_callbacks.py:1944-1948`) not mirrored; `show_reference_metadata_requirement` re-parses the pipeline on every form-state input.
- Spec §10 "API reference entry for SubtractBlank" is missing from Task 11.

### Verified OK
- No work-id is computed before `_prepare_incremental_startup` within an invocation: every `work_id_for_image`/`_worker_work_identity` call is below the startup calls (`phenotypicCLI.py:2838`/`2920`) or in a strategy/worker; the `--dry-run` exit (`:2731`) computes none.
- Startup runs for every forward path (process, staged, SLURM) via the common main flow, in both the resume and non-resume branches.
- Stale SLURM workers fail closed on a work-id mismatch (`_cli_process_single.py:829-852` raises).
- Continuation re-selects by work-id (`_cli_state_management.py:603-614`), so a changed reference digest re-runs exactly the affected images; `"unplanned"` images stop matching terminal-failure keys once the table is fixed.
- The provenance hook works as planned: `provenance_parameters` is called right after success (`_provenance.py:974-977`, called from `:695`), with no other callers.
- Probe 1 (pydantic 2.12.5): a plain mixin's `__init_subclass__` is reached under `BackgroundSubtraction`, and `FieldInfo.metadata` keeps both markers.
- Probe 1 (polars 1.41.2): `read_csv(infer_schema=False)` keeps `000123`, and an empty cell becomes null.
- Probe 1: `ImageName` and `BlankImage` normalise to their `Metadata_` forms; `Grid_RowNum` becomes `Metadata_Grid_RowNum`, which is harmless to lookups.
- The doctest premise holds: `load_synth_yeast_plate()` is a gray-mode `GridImage` with `bit_depth` 8, and its `detect_mat == get_detection_mode(...).compute(p)`.
- The freshness guard holds after construction (`_image_data_manager.py:425-433`) and after `reset()` (`_detect_mat_accessor.py:151-165`). Stores reload `gray`/`detect_mat` as stored (`_image_io_handler.py:1734-1745`); exact equality after a store round-trip is unverified, and the B1 test covers it.
- Reusing `_ColumnRefMarker` brings pipeline metadata migration of the field (`sdk_/_metadata_migration.py:561-573`) and tune inference exclusion (`tune/_search_space/_infer.py:427`) for free.
- Every corrector is an `ImageCorrector` subclass (CalibrateColorRpcc, DenoiseBlockMatch, CropImage, PadImage, GridAligner, GridApply).
- Code anchors confirmed:
  - `sdk_.__getattr__` resolves absolute module names (`sdk_/__init__.py:61-66`)
  - `get_ops()` (`_image_pipeline_core.py:549`)
  - `find_operations` (`_operation_tree.py:145-149`)
  - the preflight helpers (`tests/unit/cli/_preflight_support.py:19-75`)
  - the `"LabL"` mode name (`_lab_channel_modes.py:72`)
  - `canonical_digest` (`sdk_/_digests.py:46`), `atomic_write_*` (`sdk_/_atomic_io.py:180,209`), `preserved_on_restart_names` (`_io_constants.py:1399`)
  - the run-console ids (`_ids.py:55,75,145`) and `update_run_disabled` (`_callbacks.py:2733-2751`)
  - `compute_scope` (`_preview_cache.py:312`), whose nested scopes inherit the parent fingerprint (`:339`)
