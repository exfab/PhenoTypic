# Context brief — CLI↔GUI state tracking

Written round 0. **Reviewers start here and open source only to verify a specific
claim.** Every `file:line` below was read directly in the worktree at commit
`c9d1fbfc` (plan committed at `4d24c33e`). Discovery is paid once, here.

**Worktree:** `/bigdata/exfab/anguy344/PhenoTypic/.worktrees/cli-gui-state-tracking`
**Branch:** `cli-gui-state-tracking`

---

## 0. Spec anchors (for spec gating and the precedence table)

The spec does not use the literal headings the refinery expects. The mapping:

| Refinery anchor | Spec section |
|---|---|
| **Objective** | §1 Purpose — three coupled problems, plus the organising principle "move state that is tracked to state that is checked" |
| **Non-goals** | §2.2 Out of scope (the SLURM observer's decision tree, grace window, `squeue`/`sacct` reconciliation) **plus** `DEFERRED.md` in full |
| **Performance / NFR** | §1 problem 3 ("the completion predicate is `O(N_images)` in full-file SHA-256, invoked on a 2-second daemon tick and a 5–10-second per-browser-tab poll") and §9.2's target ("the 6,000 unchanged images cost one `stat()` each and the 10 arrivals are deep-verified"). Audit §4 quantifies the current cost: on a 10,000-image run on GPFS, one badge refresh is ~10⁴ marker reads, 2–3 × 10⁴ file hashes, 4 × 10⁴ `stat()` calls, a full event-log replay, and a multi-gigabyte parquet hash — **per tab, every five seconds**. |

**A performance concern must cite one of those.** There is no other stated NFR.

---

## 1. What the change is

Three coupled problems in the state written by the CLI and read by the GUI:

1. **Nine evidence sources** for "is this run done?", cross-checked by four independent
   classifiers that can disagree. Disagreement surfaces as `contradictory` → the output is
   flagged read-only for a reason the user cannot act on.
2. **Fourteen identity tokens**, each written to disk and cross-checked against at least
   one other.
3. **The completion predicate is `O(N)` in full-file SHA-256**, on a 2 s and a 5–10 s timer.

Plus, entangled with all three: the **measurement/metadata layout inversion** — embedded
per-image tables become pure measurements and the metadata join moves to finalization.

**The verdict up front, from the audit:** the problem is *not* string duplication. The
constants layer (`sdk_/_io_constants.py`, 2,669 lines) is unusually well kept, and the
atomicity discipline on everything the GUI polls is genuinely correct. The problem is
**cardinality and cost**.

---

## 2. Three decisions taken with the user that depart from the spec

**These are `settled-by-user` before round 1 and may not be re-raised absent new
evidence.** Full reasoning in `plans/2026-09-03-cli-gui-state-tracking/OPEN-QUESTIONS.md`.

| | Decision | What it cuts |
|---|---|---|
| **D-A** | Per-store metadata is **written at promote time**, not backfilled into stores that already carry a content proof | §6.3 hardlink re-promote, §6.4's *generalisation*, `stages.backfilled`, the backfill half of §8's fan-out, spike S-1, residual risk §15.4 |
| **D-B** | The verification cache is **in process** (what audit S1 actually proposed), not `.phenotypic/verification_cache.json`. Spike S-5 decides whether an on-disk tier ships at all | A new tracked artifact |
| **D-C** | `scientific_config_digest` **is** `processing_configuration_digest`, verbatim | A `work_id` change and its migration |

**D-A's cost, and how the plan keeps it honest:** §7.4's late-metadata guarantee narrows —
stores keep the metadata snapshot they were built against rather than being re-promoted.
The plan surfaces divergence as a **derived** `RunState` advisory, read from each store's
own `phenotypic.metadata.snapshot_sha256`, never as a tracked backfill stage. Advisories
are never gates (§4.3).

**D-C rests on a code fact reviewers should verify:**
`processing_configuration_digest_from_values`'s non-process branch
(`src/phenotypic/_cli/_cli_failure_tracker.py:236-243`) includes `include_dataset_column`,
`overlay_alpha` and `save_overlays`, and is folded into `work_id` at `:265`. Spec §5.4
claims `include_dataset_column` is in **neither** `work_id` nor the generation, while §5.1
says `work_id` is **unchanged**. Those cannot both hold.

---

## 3. Architecture map — where the state lives

### Seven substrates (audit §1)

| # | Substrate | Root | Owner | GUI relationship |
|---|---|---|---|---|
| 1 | Machine state | `<out>/.phenotypic/` | CLI | read-only, except the owner record |
| 2 | Progress / scheduler | `<out>/.phenotypic/progress/` | CLI | read-only, except the owner record |
| 3 | Deliverables | `<out>/deliverables/` | CLI **and** GUI | read + write |
| 4 | Per-image stores | `<out>/results/<ds>/zarr/<stem>.ome.zarr/` | CLI | read-only (bytes served to browser) |
| 5 | GUI sandbox caches | sandbox / user cache / `$TMPDIR` | GUI | read + write |
| 6 | GUI in-process | RAM — 136 `dcc.Store`s, 14 `dcc.Interval`s, ~20 module globals | GUI | — |
| 7 | The scheduler itself | `squeue`/`sacct` `--comment` fields | SLURM | polled |

Substrate 7 is easy to miss: `_cli_slurm_lifecycle.py:291` encodes
`phenotypic:<generation>:<token>` into the sbatch `--comment`, and `query_scheduler_comments`
(`:296`) reads it back. **The scheduler's comment field is a durable state store.**

### The nine sources (audit §4)

`inspect_output_consistency` (`gui/results_viewer/_output_consistency.py:336`) reads all
nine in one pass; `classify_output_consistency` (`:117`) reduces them to
`coherent`/`active`/`incomplete`/`contradictory` through ~14 contradiction rules and ~9
incompleteness rules:

1. `gui_launch_owner.json` → `status`
2. `manifest.json` → `is_complete`, `completed`, `failed`, `total_images`
3. `run_completion.json` → `valid_run_completion`
4. `aggregate_publication.json` → `valid_aggregate_snapshot`
5. per-image `image_complete/` markers → `current_run_is_complete`
6. `processing_state.json` + `processing_events.log` replay → counts
7. `staged_orchestration.json` + `staged_finalization_complete.json`
8. `datasets_needing_migration()` → unconverted `.h5` scan
9. (run console only) `squeue`/`sacct` + `--comment` reconciliation

And the GUI re-derives the same question **three more times, differently**:
`RunRegistry._processing_state_conflict` (`shell/_runs_registry.py:1087`) +
`_publication_evidence_conflict` (`:1202`) + `_orchestration_state_conflict` (`:1264`) for
claimability; `_local_completion_evidence_conflict` (`:591`, 8 refusal strings) for local
exit; `SlurmLifecycleObserver._observe_record` (`run_console/_slurm_observer.py:536`, ~20
branches) for scheduler status.

The event log is replayed by **two independent implementations with different semantics**:
`aggregate_state_from_events` (`_cli/_cli_update_state.py:266`, inventory- and
generation-fenced) and `RunRegistry._latest_event_states` (`_runs_registry.py:1172`,
demotes a non-terminal-stage `completed` to `started`).

---

## 4. Key files and their real signatures

Every line number below was read at `c9d1fbfc`. **Re-grep before relying on one** — the
plan's own P6 Task 7 makes the same demand of its implementer.

### `src/phenotypic/_cli/_cli_completion.py` (1,107 lines) — the module §11 splits

```
39    SUCCESS_MARKER_VERSION = 2
51-52 ARTIFACT_KIND_FILE = "file" / ARTIFACT_KIND_STORE = "store"
114   image_data_artifact(output_dir, output_manager, dataset, image_stem) -> (str, Path)
163   publish_image_success(output_dir, *, work_id, dataset, relative_image_path,
                            image_stem, mode, attempt_id, lifecycle_epoch, artifacts,
                            expected_artifact_descriptors=None, source_provenance=None,
                            commit_guard=None) -> Path
255   valid_image_success(output_dir, *, dataset, image_stem, work_id) -> bool
305   refresh_success_markers_after_metadata_migration(output_dir, *, receipt_paths=()) -> int
487   current_success_inventory(output_dir) -> dict[str, frozenset[str]] | None
534   _walk_current_success(output_dir) -> dict[str, dict[str, bool]] | None
592   manifest_completion_inventory(...)
659   current_success_counts(output_dir) -> tuple[int, int] | None
682   _current_success_work_ids(output_dir, work_ids) -> list[str]
705   current_aggregate_is_current(output_dir) -> bool | None
749   current_run_is_complete(output_dir) -> bool | None
768   authorized_measurement_sources(output_dir) -> dict[Path, str] | None
861   _canonical_digest(value) -> str
868   publish_aggregate_snapshot(output_dir, *, commit_guard=None) -> Path
933   valid_aggregate_snapshot(output_dir) -> dict | None
963   publish_run_completion_evidence(output_dir, *, execution_epoch,
                                      gui_record_generation=None, commit_guard=None) -> Path
1063  valid_run_completion(output_dir) -> dict | None
```

**The double walk the plan removes:** `current_run_is_complete` (`:749`) calls
`current_success_counts` → `_walk_current_success`, then calls
`current_aggregate_is_current` (`:705`), which calls `_current_success_work_ids` (`:682`)
— **a second complete walk, re-hashing everything again.** Both call
`load_processing_state`, which replays the entire event log
(`_cli_state_management.py:121`).

`valid_image_success` (`:255`) re-reads the marker JSON and re-hashes every declared
artifact: `_sha256` of the embedded measurements parquet **and** the overlay PNG, plus the
store root `zarr.json`.

**The `required_outputs` D8 shrinks** — `publish_aggregate_snapshot:888-891` currently
declares four: `master_csv`, `master_parquet`, `measurements_csv`, `measurements_parquet`.

### `src/phenotypic/_cli/_cli_failure_tracker.py` — identity

```
200  processing_configuration_digest_from_values(*, image_type, nrows, ncols, bit_depth,
       detect_mode, process_only_layer, ext, process_format, include_dataset_column,
       overlay_alpha, save_overlays, drop_originals=False) -> str
     # base payload: image_type, nrows, ncols, bit_depth, detect_mode, drop_originals
     # process branch (:225): + process_only_layer, ext, process_format
     # else branch   (:236): + include_dataset_column, overlay_alpha, save_overlays
246  processing_configuration_digest(config) -> str
265  compute_work_id(*, dataset, relative_image_path, input_sha256, pipeline_fingerprint,
                     processing_config_digest, mode) -> str
318  work_id_for_image(config, dataset, image_path) -> (work_id, relative_path)
353  append_terminal_failure(output_dir, *, work_id, dataset, relative_image_path,
                             failed_stage, exception, ...)
```

### `src/phenotypic/_cli/_cli_state_management.py` (521 lines)

```
45   save_processing_state(...)
98   load_processing_state(output_dir) -> ProcessingState | None
     # :121 replays the ENTIRE append-only event log on every load
181  create_initial_state(...)   # :237 "processing_generation": uuid4().hex
244  update_state_from_events(state, output_dir) -> ProcessingState
285  validate_resume_compatibility(state, config) -> (bool, str | None)
     # :337-346 guards bit_depth, detect_mode, include_dataset_column, overlay_alpha,
     #          drop_originals, save_overlays, process_format
378  get_remaining_images_for_datasets(...)
486  exclude_terminal_failures_for_datasets(...)
```

`create_initial_state`'s config block (`:215-238`) is the authoritative list of what
`processing_state.json` carries.

### `src/phenotypic/sdk_/_io_constants.py` (2,669 lines) — the constants layer

```
317/321  MASTER_MEASUREMENTS_CSV / MASTER_MEASUREMENTS_PARQUET
327/332  MEASUREMENTS_CSV / MEASUREMENTS_PARQUET
448      PROCESSING_STATE_JSON
657-791  DIR_RESULTS, DIR_PROGRESS, DIR_IMAGE_COMPLETE, DIR_MEASUREMENTS, DIR_ZARR,
         DIR_OVERLAYS, DIR_CHUNKS, DIR_RECOMPILE, DIR_RECOMPILE_SHARDS, DIR_QC,
         DIR_DELIVERABLES, DIR_PHENOTYPIC, …
895      phenotypic_cache_dir(output_dir)
903/912/917  progress_dir / results_dir / deliverables_dir
949      processing_state_path      980  resolve_processing_state_path
1081     clear_machine_state(output_dir) -> bool
         # deletes every child of .phenotypic/ EXCEPT terminal_failures.jsonl
1132-1147  master_measurements_{csv,parquet}_path / measurements_{csv,parquet}_path
1566     zarr_store_path(output_dir, dataset, stem)
1716     store_publication_token(...)
1942/1947/1952  run_completion_marker_path / aggregate_publication_marker_path /
                image_completion_marker_path
2422     class BundleLayout   # .detect() keys on MASTER_MEASUREMENTS_PARQUET
```

### `src/phenotypic/_cli/_cli_output_manager.py` (2,037 lines) — the finalize seam

```
969   finalize_post_master_outputs(output_dir, master_df, pipeline, metadata_csv=None,
        metadata_join_keys=None, no_qc=False, study_config=None, commit_guard=None)
        -> pl.DataFrame
      # the seam that ALREADY exists and is ALREADY shared by both callers
1351  _aggregate_measurements_unlocked(output_dir, dataset_names, include_dataset_column=True,
        metadata_csv=None, pipeline=None, no_qc=False, study_config=None, commit_guard=None)
        -> Path | None
      # :1526 calls finalize_post_master_outputs; :1537 publishes the aggregate proof
      # DOCSTRING (:1362): "Prefers pre-aggregated _dataset_aggregated.parquet files when
      #   available" -- this is EXACTLY the fast path INV-INPUTS (§7.5) forbids
1545  aggregate_measurements(...)   # wraps the above in .aggregate_publication.lock
```

`_cli_recompile_worker.py:764` `_run_post_master_steps` — the second caller, whose own
comment (`:790`) says it is "matching the forward CLI path". Its
`measurement_sources`-vs-`metadata_join_keys` branch (`:777-787`) exists only because the
two callers arrive with differently-shaped inputs.

### `src/phenotypic/_cli/_embedded_measurement_tables.py`

```
42  prepare_embedded_measurement_table(measurements, metadata_csv)
      -> PreparedEmbeddedMeasurementTable
    # :55  measurement_columns computed from the baseline BEFORE the join
    #      -- this is the intrinsic/user boundary the plan says already has a name
    # :81  warns on duplicate metadata keys, preserving fan-out
    # :88-95 the right-join the plan stops before
```

### `src/phenotypic/sdk_/ngff_.py`

```
96    MEASUREMENT_COLUMNS = "phenotypic.measurement_columns"
642   require_readable_store(store_path) -> dict
1665  fsync_tree(root)
1737  promote_store(part, final, *, fsync, commit_guard=None) -> Path
      # docstring: caller writes arrays/chunks first, then OME/zarr.json, then the root
      #   zarr.json LAST. This function does NOT write the root.
      # move-aside is mandatory (ENOTEMPTY on POSIX; MoveFileEx can't name a dir on Windows)
      # :1790 fsync_tree(part) before the rename
```

### The staged engine

```
_cli_stage2_token.py:42   _STAGE2_DIR = "stage2_done"
                    :45   stage2_token_path / :50 write_stage2_token / :92 stage2_token_exists
                    :108  delete_stage2_token
                    :124  _STAGE2_RAW_DIR = "stage2_raw"   (STAYS -- bulk replay data)
_cli_staged_resume.py:136 stage3_completion_marker_path  (:141 inline "stage3_complete")
                     :148 stage3_completion_exists   (imported by the SLURM observer)
                     :157 write_stage3_completion_marker
                     :197 classify_staged_image   <-- the risk surface in P3
                     :288 build_staged_resume_plan
_cli_recompile_slurm_scripts.py:51-53  TASK_MEASUREMENTS / TASK_OVERLAY / TASK_FINALIZE
_cli_slurm_array_scripts.py:30,32      _CHECKPOINT_SENTINEL / _MANIFEST_SENTINEL
```

### GUI consumers

```
gui/_snapshot_status.py:17   snapshot_refresh_status(output_root, *, refresh_supported)
                       :67   _completion_evidence_status
gui/results_viewer/_output_consistency.py  (617 lines -- P6 deletes the file)
    :42  OutputCompletionEvidence  (33 fields)
    :80  OutputConsistencyReport   (.is_read_only, .cache_reusable, .has_active_owner,
                                    .core_readable)
    :117 classify_output_consistency   :336 inspect_output_consistency
gui/results_viewer/_output_root.py
    :112 class OutputRoot   :178 discover   :533 consumed_state_fingerprint
    :542 snapshot_is_current   :559 refresh_state_is_current   :613 overlay_path
gui/results_viewer/_processing_inventory.py
    :277 _walk_results_without_descending_into_stores  (DO NOT "fix" -- see audit)
    :462 _inventory_is_current  (compares st_ctime_ns -- audit S3)
gui/shell/_runs_registry.py
    :591 _local_completion_evidence_conflict   :773 rehydrate (persist=False -- audit S7)
    :1058 _assert_output_claimable_locked   :1087/:1172/:1202/:1264 the four predicates
    :1306 _persist_record_locked
gui/run_console/_slurm_observer.py:536 _observe_record   :1312 the 2s tick
```

---

## 5. Layering, and why it constrains P1

- **`_cli` imports `sdk_`** at module scope, from 51 files.
- **`sdk_` imports `_cli`** from **16 sites, all lazily, inside function bodies**:
  `sdk_/_hdf_to_zarr.py:605,714,718,761,762`, `sdk_/generate_report.py:19,20,25`,
  `sdk_/monitor_slurm_jobs.py:19`, `sdk_/slurm/_dispatcher.py:283,404`.
- `_cli_completion.py:14` imports `phenotypic.sdk_` at module scope.

So spec §11's "readers → `sdk_/_run_state.py`" is feasible only if the readers do **not**
need `load_processing_state`. The plan's resolution (OPEN-QUESTIONS Q4): read
`processing_state.json` as plain JSON, no event-log replay — possible precisely because
§4.2 demotes the event log out of the evidence set. Pinned by INV-LAYER, an AST test.

**The GUI imports 25 private `phenotypic._cli` symbols across 9 modules today** (audit §7).
That is how the `O(N)`-hashing completion predicate ended up on a 2-second timer.

---

## 6. Conventions a reviewer should hold the plan to

From the project `CLAUDE.md` and `src/phenotypic/_cli/CLAUDE.md`:

- **`uv` is the sole runner.** Never bare `python`/`pip`. `ruff check --fix` **always with
  explicit paths** — bare walks the repo and rewrites untouched files.
- **`deliverables/metadata.csv` is byte-exact provenance and is never rewritten by any
  mode, `--mode migrate` included.** An earlier draft carved out an exception (rewrite
  canonical headers, keep `metadata.original.csv`); it was **withdrawn** (D9/FLOW-4) and
  never implemented. `metadata.original.csv` must not exist. Migrate emits
  `metadata.canonical.csv` **alongside** the untouched snapshot.
- **Array-auxiliary work:** no scheduler sidecar job beside an active ordinary array.
  Ancillary work routes through reserved trigger entries **inside** the array task list
  (`__PHENOTYPIC_CHECKPOINT__` / `__PHENOTYPIC_MANIFEST__` pattern), and every trigger entry
  is counted when sizing chunks against `MaxArraySize`. A terminal `afterany` finalizer is
  **not** a sidecar.
- **Output layout:** `master_measurements.*` is the pre-post concatenation of authorized
  embedded tables; `measurements.*` is the post-applied mirror the GUI reads/curates. **Feed
  analysis and dashboards from the mirror, not the master.** Route any FINAL master write
  through `finalize_post_master_outputs`.
- **Never hand-join a path** — always the `phenotypic.sdk_` helpers.
- **Vendored reference sources under `specs/*/refs/` are read-only.** (None in this spec.)

### Things the audit says are load-bearing and must NOT be "simplified"

- The **marker-last publication protocol** end to end: store root `zarr.json` last, marker
  after artifacts, aggregate after outputs, run-completion after aggregate, `chunk_state.json`
  after chunks, stage-2 token after the raw `.npy`.
- `_walk_results_without_descending_into_stores` (`_processing_inventory.py:277`) — the
  `dirnames` pruning is the optimization, and it says so.
- `image_display_range` (`_shared/tiles.py:544`) being deliberately uncached.
- The `store_publication_token` → HTTP **409** (not 404, not 410) contract.
- `BrowseCache`'s immutable revision-addressed entries.
- `store_schema_version`'s single choke point in `require_readable_store`.

---

## 7. Test environment (matters for every "run the tests" step)

- `tests/gui` **is** in `testpaths`; `tests/e2e` is **not** and needs `PLAYWRIGHT=1`.
- `QT_QPA_PLATFORM=offscreen` is mandatory — without it the interpreter aborts at 79% with
  no summary.
- **`-n auto` is wrong on HPCC**: it reads the node's core count, not the allocation's, and
  manufactures timeout failures.
- **`-x` silently truncates** a run that then gets recorded as a baseline.
- The full `tests/unit` suite is **~65 minutes** — a Slurm job, not a foreground command.
  Committed script: `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`.
- Known baseline: **four pre-existing failures**, three of which fail only on compute nodes.
- Pre-flight blast radius (`grep -rl`, at `c9d1fbfc`): `master_measurements_csv_path`
  6 `src/` + 10 `tests/`; `current_run_is_complete` 7 + 2; `valid_run_completion` 5 + 6;
  `inspect_output_consistency` 4 + 3; `_latest_event_states` 1 + 0.

---

## 8. Cluster facts that constrain P0 and P5

From the user's global `CLAUDE.md`:

- Partitions: `short` (2 h), `batch`/`intel`/`epyc` (30 d), `exfab` (30 d, **1 node**,
  requires `--account=exfab`), `preempt` (7 d, `--account=preempt`).
- **`--account=` with an empty value is an invalid account** and `sbatch` rejects the whole
  submission. Build the flag conditionally.
- **Always set `--mem` and `--time`.** `DefMemPerCPU` is 1 GB/CPU.
- **`sbatch --parsable` returns an empty id on rejection.** Verify `^[0-9]+$`.
- **`MaxArraySize` (2500) caps the *index*.** Highest legal index 2499.
- **`/scratch/<user>/<jobid>` is node-local and per-job** — never a job's `--output` or
  result target. Symptom of getting it wrong: `FAILED` with `ExitCode 0:53` and **no log
  file at all**, intermittently.
- Submission ≠ start: `scontrol show job <id> | grep -E 'StartTime|Reason'`.

---

## 9. Where the plan lives

```
docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/
├── README.md              index, global constraints, invariants, phase DAG
├── OPEN-QUESTIONS.md      D-A/D-B/D-C (user-settled), Q2/Q3/Q4/Q6, O-1/O-2 (deferred)
├── phase-0-spike-gate.md  S-2, S-3, S-4, S-5   (S-1 cut by D-A)
├── phase-1-run-state-sdk.md
├── phase-2-identity-schema.md
├── phase-3-per-image-record.md
├── phase-4-finalize-run.md
├── phase-5-fanout.md
├── phase-6-consumer-migration.md
└── phase-7-migrate-mode.md
```

Five named invariants, each of which the plan says is a test that must be **proved able to
fail**: **INV-CACHE**, **INV-INPUTS**, **INV-IMMUTABLE**, **INV-DEGRADE**, **INV-LAYER**.
