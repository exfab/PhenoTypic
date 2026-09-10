# CLI ↔ GUI state-tracking audit

**Date:** 2026-09-03 · **Scope:** every piece of state tracked between what the CLI
writes and what the GUI reads, plus the file-reading machinery on both sides.
**Method:** direct reading of the contract layer plus five parallel exploration
passes (CLI writers, GUI readers, shared path contract, GUI byte-serving, run-console
bridge). Every claim below carries a `file:line`. Claims marked **[verified]** were
re-checked by hand after the exploration pass reported them.

**Verdict up front.** The problem is *not* string duplication — the constants layer
(`sdk_/_io_constants.py`, 2669 lines) is unusually well kept, and the atomicity
discipline on everything the GUI polls is genuinely correct. The problem is
**cardinality and cost**: nine independent sources of truth for "is this run done?",
thirteen distinct identity tokens, and a completion predicate that is `O(N_images)`
in full-file SHA-256 yet is invoked on a 2-second daemon tick and a 5–10-second
per-browser-tab poll.

---

# PART 1 — The inventory

## §1. The seven state substrates

| # | Substrate | Root | Owner | GUI relationship |
|---|---|---|---|---|
| 1 | **Machine state** | `<out>/.phenotypic/` | CLI | read-only, except the owner record |
| 2 | **Progress / scheduler state** | `<out>/.phenotypic/progress/` | CLI | read-only, except the owner record |
| 3 | **Deliverables** | `<out>/deliverables/` | CLI **and** GUI (dual-owned) | read + write |
| 4 | **Per-image stores** | `<out>/results/<ds>/zarr/<stem>.ome.zarr/` | CLI | read-only (bytes served to browser) |
| 5 | **GUI sandbox caches** | `<sandbox>/.phenotypic-gui/`, user cache dir, `$TMPDIR` | GUI | read + write |
| 6 | **GUI in-process state** | RAM | GUI | 136 `dcc.Store`s, 14 `dcc.Interval`s, ~20 module globals |
| 7 | **The scheduler itself** | `squeue` / `sacct` `--comment` fields | SLURM | polled, reconciled against 1–2 |

Substrate 7 is easy to miss: `_cli_slurm_lifecycle.py:291` encodes
`phenotypic:<generation>:<token>` into the sbatch `--comment`, and
`query_scheduler_comments` ([`_cli_slurm_lifecycle.py:296`](src/phenotypic/_cli/_cli_slurm_lifecycle.py:296))
reads it back. The scheduler's comment field is a durable state store in this design.

---

## §2. Complete artifact inventory

### 2a. `<out>/.phenotypic/` — machine state

| Artifact | Format | Writer | Consumer | Atomic | Authority |
|---|---|---|---|---|---|
| `processing_state.json` | JSON | [`save_processing_state`](src/phenotypic/_cli/_cli_state_management.py:45) | CLI resume, GUI consistency + registry | ✅ lock + `atomic_write_json` | **authoritative** for `work_ids`, `processing_generation`, `success_markers_required`; `datasets` is *derived* (re-aggregated from the event log on every load, [`:121`](src/phenotypic/_cli/_cli_state_management.py:121)) |
| `processing_events.log` | pipe-delimited text, append-only | [`append_event`](src/phenotypic/_cli/_cli_update_state.py:76) | `aggregate_state_from_events`, GUI consistency, GUI registry (a **second, different** replay) | locked append, no fsync | derived; fenced by `generation` + `inventory` |
| `terminal_failures.jsonl` | JSONL | [`append_terminal_failure`](src/phenotypic/_cli/_cli_failure_tracker.py:353) | resume exclusion | ✅ `durable=True` + rollback | **authoritative** terminal failure journal |
| `aggregate_publication.json` | JSON | [`publish_aggregate_snapshot`](src/phenotypic/_cli/_cli_completion.py:868) | `valid_aggregate_snapshot`, GUI consistency | ✅ | **authoritative** aggregate-snapshot integrity |
| `logs/{gui,slurm}/…` | text | CLI + GUI submitter tee | GUI log tail | append | — |
| `slurm_scripts/`, `staged_manifest.json`, `staged_controller.json` | sh + JSON | staged submitter | controller | ✅ (JSON) | authoritative worklist |
| `migration_*`, `worker_status/`, `metadata_migration/` | JSON + framed binary | migrate mode | migrate workers | **mixed — 4 naive writers** | authoritative |
| `.aggregate_publication.lock`, `.migration-attempt.lease` | flock | — | — | — | mutual exclusion |

### 2b. `<out>/.phenotypic/progress/` — progress & scheduler state

| Artifact | Format | Writer | Consumer | Atomic | Authority |
|---|---|---|---|---|---|
| `image_complete/<ds>/<stem>.json` | JSON marker | [`publish_image_success`](src/phenotypic/_cli/_cli_completion.py:163) | [`valid_image_success`](src/phenotypic/_cli/_cli_completion.py:255) | ✅ + `pre_replace` revalidation | **THE per-image authority** |
| `stage2_raw/<ds>/<stem>.npy` | NumPy | [`write_stage2_raw`](src/phenotypic/_cli/_cli_stage2_token.py:132) | Stage 3 replay | ✅ | authoritative Stage-3 input |
| `stage2_done/<ds>/<stem>.json` | JSON, **consumable** | [`write_stage2_token`](src/phenotypic/_cli/_cli_stage2_token.py:45) | Stage 3 (unlinks it) | ✅ | Stage-2 completion |
| `stage3_complete/<ds>/<stem>.json` | JSON marker | [`write_stage3_completion_marker`](src/phenotypic/_cli/_cli_staged_resume.py:157) | GUI observer, aggregation gate | ✅ | Stage-3 completion |
| `run_completion.json` | JSON | [`publish_run_completion_evidence`](src/phenotypic/_cli/_cli_completion.py:963) | GUI observer + registry + consistency | ✅ idempotent | **authoritative terminal** run marker |
| `manifest.json` | JSON | [`build_manifest`](src/phenotypic/_cli/_dashboard/_manifest_builder.py:819) | dashboard JS, GUI registry, GUI observer, GUI consistency, GUI classifier | ✅ (no lock; leader-elected) | **derived/display** — but still load-bearing on the legacy path |
| `job_metadata.json` | JSON | CLI strategies + `mirror_job_to_metadata` | GUI observer, GUI slurm | ✅ lock (**except** the recompile writer, below) | authoritative scheduler sidecar |
| `gui_launch_owner.json` | JSON | **GUI** [`_persist_record_locked`](src/phenotypic/gui/shell/_runs_registry.py:1306) | GUI registry, consistency, QC rebuild, CLI freshness guard | ✅ lock + CAS | **authoritative GUI↔CLI identity binding** |
| `slurm_lifecycle.json` | JSON fence | [`_cli_slurm_lifecycle.py:96`](src/phenotypic/_cli/_cli_slurm_lifecycle.py:96) | GUI observer, CLI publish guard | ✅ always locked | **authoritative active-generation fence** |
| `slurm_jobs.jsonl` (+ legacy `staged_jobs.jsonl`) | JSONL ledger | `append_lifecycle_entry` | GUI observer | locked append | **authoritative** submitted-job idempotency key |
| `staged_orchestration.json` | JSON | [`_cli_staged_orchestration.py:121`](src/phenotypic/_cli/_cli_staged_orchestration.py:121) | controller, GUI observer, GUI registry | ✅ lock | authoritative controller state |
| `staged_epoch_deactivations.jsonl` | JSONL | `deactivate_orchestration` | controller | append | **authoritative epoch fence** — outranks the mutable state above |
| `staged_finalization_complete.json` | JSON marker | `mark_staged_complete` | GUI observer, GUI consistency | ✅ | staged terminal marker |
| `stage2_terminal_failures.jsonl` | JSONL | `append_stage2_terminal_failure` | resume | append | per-epoch Stage-2 terminal set |
| `failures.jsonl` | JSONL | [`append_failure`](src/phenotypic/_cli/_cli_failure_tracker.py:589) | dashboard | locked append, **with an unlocked fallback** ([`:643`](src/phenotypic/_cli/_cli_failure_tracker.py:643)) **[verified]** | derived/display |
| `chunk_state.json`, `chunk_manifest.json`, `chunks/chunk_NNN.parquet`, `analysis_full.parquet` | JSON + Parquet | `_cli_chunk_writer.py` | aggregation | ✅ under `.chunk_lock` | `chunk_state` authoritative, rest derived |
| `recompile/`, `worklists/`, `unpublished_stage3/`, `restart_stale_parquets/`, `migration/` | mixed | recompile / staged / migrate | workers | mixed | see §2a note |

### 2c. `<out>/deliverables/` — the dual-owned zone

| Artifact | CLI writes | GUI writes | Notes |
|---|---|---|---|
| `master_measurements.{csv,parquet}` | ✅ | — | archival full object set; curation re-keys against **this**, not the mirror ([`_curation_labels.py:406`](src/phenotypic/gui/results_viewer/_curation_labels.py:406)) |
| `measurements.{csv,parquet}` (mirror) | ✅ | ✅ (`_filtered_state.py:578`) | display frame; post-ops applied, curated rows removed |
| `metadata.csv` | ✅ byte-exact snapshot | — | provenance; never rewritten (CLAUDE.md rule, enforced through `--mode migrate`) |
| `pipeline.json.pht-pipe` | ✅ under `pipeline_publication_lock` | — | — |
| `analysis_manifest.json` + named tables | ✅ 2-phase w/ journal | — | strongest CLI transaction |
| `overlays/<ds>/<stem>.png` | ✅ | — | marker-bound artifact |
| `qc/qc.duckdb` | ✅ (tmp + swap) | — (rebuild path writes) | — |
| `qc/curation_labels.parquet` | re-emits from it | ✅ **primary writer** | GUI-owned durable label store |
| `qc/custom_categories.json` | — | ✅ | — |
| `qc/review_state.json` | **deletes** it at finalize ([`_cli_output_manager.py:1238`](src/phenotypic/_cli/_cli_output_manager.py:1238)) | ✅ | GUI-owned, CLI-reset per run |
| `errors/<category>.parquet` | ✅ re-emit | ✅ live | **dual-owned**, documented |
| `verified.parquet` | — | ✅ | GUI-only; finalize never writes it |
| `error_analysis.{parquet,csv,html}` | ✅ | ✅ | dual |
| `dashboard.html`, `processing_report.html`, `README.md` | ✅ naive `write_text` | — | no machine reader → tolerable |

### 2d. `<out>/results/` — per-image

`results/<ds>/zarr/<stem>.ome.zarr/` published by directory rename with the root
`zarr.json` written **last** ([`ngff_.promote_store:1737`](src/phenotypic/sdk_/ngff_.py:1737)).
That root document is simultaneously: the store's "complete" signal, its content
fingerprint anchor for the success marker, and the GUI's cache/generation key. It
carries `attributes.phenotypic` — `store_schema_version`, `image_class`, `work_id`,
`publication_protocol`, `provenance` journal, `tables`, `series`, `labels`, `pyramid`.
The embedded `tables/measurements/table.parquet` is the authoritative object table.

### 2e. GUI-owned caches (substrate 5)

| Cache | Location | Key | Invalidation |
|---|---|---|---|
| `BrowseCache` (preview + DZI) | `<sandbox>/.phenotypic-gui/browse_cache/` → user cache → tmp | sha256 of `(sandbox, relpath, size, mtime_ns, ctime_ns, store_revision, render_schema, tile, overlap)` | **immutable entries**, LRU 10 GiB→8 GiB; marker-last publish. *The best-designed cache in the codebase.* |
| Processing-inventory cache | `<cache_root>/<key>.processing-inventory.json` | `source_cache_key(root, "processing-inventory-v1")` | schema v2 + evidence fingerprint + per-entry re-stat; **only persisted for `coherent` outputs** |
| Builder preview cache | `$TMPDIR/phenotypic/pipeline-preview/<session>/<scope>` | scope-graph content hash | wiped at launch **and** `atexit` |
| Builder point-picker DZI | `<image_root>/.phenotypic-gui/builder_tiles/<session>/` | `(session_id, source)` | **never garbage-collected** — no launch wipe, no atexit, no LRU |
| Viewer DZI cache dir | `<cache_root>/<key>/dzi` | source fingerprint | **path computed, never created** — vestigial since the Viv rebuild |
| `browse/_source_render.py` tmp cache | `$TMPDIR/phenotypic/browse/` | token | **dead** — `browse_cache_base`/`cache_png_path`/`init_cache`/`wipe_cache` have zero production callers **[verified]** |

---

## §3. Identity tokens — thirteen of them

Every one of these is written to disk somewhere and cross-checked against at least
one other.

| # | Token | Generated | Lives in | Fences |
|---|---|---|---|---|
| 1 | `processing_generation` | `uuid4().hex` | `processing_state.json:config`, `job_metadata.json`, `manifest.json`, **event-log field 9**; env `PHENOTYPIC_PROCESSING_GENERATION` | event-log replay |
| 2 | `slurm_generation` / lifecycle `generation` | `uuid4().hex` | `slurm_lifecycle.json`, `slurm_jobs.jsonl`, `job_metadata.json`; env `PHENOTYPIC_SLURM_GENERATION`; **sbatch `--comment`** | submission + publication |
| 3 | staged `epoch` | `uuid4().hex` | `staged_orchestration.json`, deactivation journal, `staged_finalization_complete.json` | staged controller |
| 4 | `gui_record_generation` | `uuid4()` | `gui_launch_owner.json`, `job_metadata.json`, `manifest.json`, `run_completion.json`; env `PHENOTYPIC_GUI_RECORD_GENERATION` | GUI↔CLI ownership |
| 5 | `lifecycle_epoch` | = 2 or 3 | `RunRecord`, per-image marker | stale-publication refusal |
| 6 | `execution_epoch` | `"local"` or 2 | `run_completion.json` | — |
| 7 | `attempt_id` | per-attempt | per-image marker, event log, terminal failures | retry accounting |
| 8 | `work_id` | content-derived | `processing_state.config.work_ids`, marker, store attrs, failure journal | **the per-image content identity** |
| 9 | `publication_id` | `uuid4().hex` | `aggregate_publication.json`, `run_completion.json` | aggregate↔run binding |
| 10 | `store_publication_token` / `store_revision_identity` | root-`zarr.json` stat or content | `/zarr/<…>/<token>/` URL segment | HTTP 409 on stale reads |
| 11 | `record_revision` + registry `revision` | monotonic int | `gui_launch_owner.json`, RAM | cross-process CAS |
| 12 | `binding_generation` + `BindingRequestFence` | int | RAM + `dcc.Store` | rejects callbacks from a superseded binding |
| 13 | `SourceRevision.cache_key` / `BROWSE_RENDER_SCHEMA_VERSION` | sha256 | Browse cache paths | cache addressing |

## §4. "Is the run done?" — nine sources, cross-checked

This is the heart of the complexity. [`inspect_output_consistency`](src/phenotypic/gui/results_viewer/_output_consistency.py:336)
reads **all nine** in one pass and [`classify_output_consistency`](src/phenotypic/gui/results_viewer/_output_consistency.py:117)
reduces them to one of `coherent` / `active` / `incomplete` / `contradictory` through
~14 contradiction rules and ~9 incompleteness rules.

1. `gui_launch_owner.json` → `status`
2. `manifest.json` → `is_complete`, `completed`, `failed`, `total_images`
3. `run_completion.json` → validated by `valid_run_completion`
4. `aggregate_publication.json` → validated by `valid_aggregate_snapshot`
5. per-image `image_complete/` markers → `current_run_is_complete`
6. `processing_state.json` + `processing_events.log` replay → counts
7. `staged_orchestration.json` + `staged_finalization_complete.json`
8. `datasets_needing_migration()` → unconverted `.h5` scan
9. (run console only) `squeue` / `sacct` job states + `--comment` reconciliation

The GUI then re-derives the same question **three more times, differently**:
- [`RunRegistry._processing_state_conflict`](src/phenotypic/gui/shell/_runs_registry.py:1087) +
  [`_publication_evidence_conflict`](src/phenotypic/gui/shell/_runs_registry.py:1202) +
  [`_orchestration_state_conflict`](src/phenotypic/gui/shell/_runs_registry.py:1264) — for *claimability*
- [`_local_completion_evidence_conflict`](src/phenotypic/gui/shell/_runs_registry.py:591) — 8 distinct refusal strings, for a local exit
- [`SlurmLifecycleObserver._observe_record`](src/phenotypic/gui/run_console/_slurm_observer.py:536) — a ~20-branch decision tree, for scheduler status

And the event log is replayed by **two independent implementations** with different
semantics: [`aggregate_state_from_events`](src/phenotypic/_cli/_cli_update_state.py:266)
(inventory- and generation-fenced) and
[`RunRegistry._latest_event_states`](src/phenotypic/gui/shell/_runs_registry.py:1172)
(demotes a non-terminal-stage `completed` to `started`).

### The cost of asking

`valid_run_completion(output_dir)` → `current_run_is_complete` → `current_success_counts`
→ [`_walk_current_success`](src/phenotypic/_cli/_cli_completion.py:534), which calls
[`valid_image_success`](src/phenotypic/_cli/_cli_completion.py:255) **once per image**.
Each of those re-reads the marker JSON and re-hashes every declared artifact — for a
full run that is a full SHA-256 of the embedded measurements parquet **and** the
overlay PNG, plus the store root `zarr.json`.

Then `current_run_is_complete` also calls `current_aggregate_is_current`, which calls
[`_current_success_work_ids`](src/phenotypic/_cli/_cli_completion.py:682) — **a second
complete walk, re-hashing everything again**. And every one of those entry points
calls `load_processing_state`, which **replays the entire append-only event log**
([`_cli_state_management.py:121`](src/phenotypic/_cli/_cli_state_management.py:121)).

**Where this is invoked from:**

| Caller | Cadence | Multiplier |
|---|---|---|
| [`_snapshot_status`](src/phenotypic/gui/results_viewer/_app.py:373) | `SNAPSHOT_STATUS_INTERVAL_ID`, **5 000–10 000 ms** | × open viewer tabs |
| `analysis/_app.py:256` | same interval | × open analysis tabs |
| [`_slurm_observer._observe_record`](src/phenotypic/gui/run_console/_slurm_observer.py:1312) | daemon thread, **2 000 ms** | × nonterminal SLURM runs |
| `OutputRoot.discover` | per binding | ×2 (double-read snapshot) |
| `OutputMutationGuard` | per mutation | ×2 |

Alongside it, the same 5–10 s tick runs:
- [`snapshot_is_current()`](src/phenotypic/gui/results_viewer/_output_root.py:542) — one
  `stat()` per inventoried path (~4 per image, plus overlays), comparing
  `size`, `mtime_ns` **and `ctime_ns`**;
- [`refresh_state_is_current()`](src/phenotypic/gui/results_viewer/_output_root.py:559) —
  full-content SHA-256 of `measurements.parquet`, `measurements.csv`,
  `pipeline.json`, `curation_labels.parquet`, `custom_categories.json`,
  `qc.duckdb`, `review_state.json`.

On a 10 000-image run on GPFS, one badge refresh is on the order of 10⁴ marker reads,
2–3 × 10⁴ file hashes, 4 × 10⁴ `stat()` calls, a full event-log replay, and a
multi-gigabyte parquet hash. **Per tab. Every five seconds.**

## §5. GUI in-process state

- **136 `dcc.Store`s** — only 35 declare `storage_type` (7 `local`, 12 `session`,
  16 `memory`); the rest default to `memory` and die on page reload.
- **14 `dcc.Interval`s.** The hot one is `RC_INTERVAL_LOG` at **1 000 ms**, which
  fans out to six server-side callbacks ([`_callbacks.py:2345`, `:2425`, `:2480`,
  `:2536`, `:2680`](src/phenotypic/gui/run_console/_callbacks.py:2345)).
  `RC_INTERVAL_ACTION_WATCHDOG` runs at **500 ms** and is never disabled.
- **261 server-side callbacks**, 34 clientside. **No** Dash `background=True` /
  `long_callback` / Celery / Diskcache anywhere — every long operation is a
  hand-rolled thread plus an interval.
- **Process-wide singletons on `app.server.config`:** `RunRegistry`, `LocalRunner`,
  `SlurmLifecycleObserver`, `ToolSession` ×2, `CFG_RESULTS_BINDING_STATE`,
  `BindingCoordinator`, `ResultsBindJobManager`, `CFG_OUTPUT_ROOT`,
  `CFG_FILTERED_STATE` (`CurationLabels`), `CFG_RECIPE_STATE`, `MeasurementSchema`,
  QC scratch ×4, `OperationRegistry`, `BrowseCache`, `BrowsePreparationManager`.
  **One bound output for the whole process**; every browser tab shares it.
- **Unbounded module globals:** `LocalRunner._instances` (self-documented
  `TODO(perf)`), `_LAST_DUMPED` keyed on `id(obj)` **[verified]**,
  `_terminal_job_cache` (sacct results, never evicted), two never-shut-down
  `ThreadPoolExecutor`s in `tune/`.

## §6. File-reading paths — five byte mechanisms, not two

`gui/CLAUDE.md:37` frames this as "two pixel paths". There are five server-side
mechanisms:

1. `/zarr/<ds>/<stem>.ome.zarr/<token>/<tail>` → Viv/deck.gl ([`_zarr_routes.py:389`](src/phenotypic/gui/results_viewer/_zarr_routes.py:389))
2. `/preview-zarr/…` → same client, own route ([`builder/_preview_zarr_routes.py:180`](src/phenotypic/gui/builder/_preview_zarr_routes.py:180))
3. Browse `/assets/<tok>/<rev>/zarr/<member>` → same Viv facade, **a completely
   separate `dirfd`-anchored implementation** ([`browse/_tile_routes.py:357`](src/phenotypic/gui/browse/_tile_routes.py:357))
4. libvips → DZI → OpenSeadragon (Browse flat images, builder point picker)
5. **Server-side PNG crop rendering** — windowed `zarr.open_array` read in Python,
   contour + spotlight compositing, PNG bytes returned, across three mounted URL
   segments ([`_shared/tiles.py:1144`](src/phenotypic/gui/_shared/tiles.py:1144))

There are consequently **three independent implementations of "authorize a Zarr
member for serving"** — `_readable_roots_for` ([`_zarr_routes.py:166`](src/phenotypic/gui/results_viewer/_zarr_routes.py:166)),
`_is_authorized_image_member` ([`_tile_routes.py:258`](src/phenotypic/gui/browse/_tile_routes.py:258)),
and `resolve_within_root` ([`_shared/tiles.py:970`](src/phenotypic/gui/_shared/tiles.py:970)).
They agree today; nothing enforces that they keep agreeing.

**Scanning hot spots:**
- [`browse/_source_lister.list_datasets:38`](src/phenotypic/gui/browse/_source_lister.py:38) —
  uncached unbounded-depth `os.walk` run **inside a Dash callback** on every sidebar refresh.
- [`BrowseCache.usage()`](src/phenotypic/gui/browse/_cache.py:271) `rglob`s every cache
  entry, and is called on every `/api/browse/dataset/status` poll.
- [`_store_revision_snapshot`](src/phenotypic/sdk_/_io_constants.py:1810) fully descends
  a Zarr chunk tree, **twice**, for any store lacking a `publication_protocol`
  declaration — i.e. every third-party store Browse probes.

**The one thing done right:** [`_walk_results_without_descending_into_stores`](src/phenotypic/gui/results_viewer/_processing_inventory.py:277)
prunes `dirnames` in place so a store contributes 2 entries instead of ~58, with a
"Do not 'fix' this back to a recursive walk" comment. Likewise
[`image_display_range`](src/phenotypic/gui/_shared/tiles.py:544) is *deliberately not
cached*, with a written argument for why no available key is sound.

## §7. The seam — GUI imports of private CLI internals

The GUI imports from `phenotypic._cli` in **9 modules**:

| GUI module | Imports |
|---|---|
| `results_viewer/_output_consistency.py` | `_cli_update_state.aggregate_state_from_events`, `_cli_completion.{valid_aggregate_snapshot, valid_run_completion}` |
| `run_console/_slurm.py` | 7 symbols from `_cli_slurm_lifecycle` |
| `run_console/_slurm_observer.py` | 6 from `_cli_slurm_lifecycle`, 2 from `_cli_staged_orchestration`, `_cli_staged_resume.stage3_completion_exists`, `_cli_completion.{current_success_counts, valid_run_completion}` |
| `shell/_runs_registry.py` | `_cli_gui_lifecycle`, `_cli_staged_orchestration`, `_cli_file_locking.atomic_read`, `_cli_update_state.parse_event_line`, `_stages.STAGED_TERMINAL_STAGE`, `_cli_completion.current_run_is_complete` |
| `run_console/_callbacks.py` | `cancel_generation`, `load_slurm_lifecycle`, `pipeline_requires_gpu` |
| `run_console/_request_safety.py` | `prepare_metadata_join_keys`, `scan_directory_structure` |
| `analysis/_callbacks.py` | `_emit_analysis_outputs` |
| `shell/_app.py` | `_cli_preload.preload_custom_operation_modules` |

This is not inherently wrong — sharing one predicate is better than two — but it
means the GUI's correctness is coupled to ~25 private CLI symbols with no declared
interface, and it is how the O(N)-hashing completion predicate ended up on a 2-second
timer.

---

# PART 2 — Simplification

Ranked by (impact × confidence) ÷ cost. Everything here is a proposal, not a change.

## Tier 1 — structural; these are what "brittle at scale" actually means

### S1. Make completion evidence *cacheable* instead of recomputing it every tick
**The single highest-value change.** `valid_run_completion` is a pure function of a
file set that only ever grows monotonically, yet nothing memoizes it.

- Give `valid_image_success` a process-level cache keyed on the marker file's
  `(st_dev, st_ino, st_size, st_mtime_ns)` — the exact pattern
  [`_generation_token_for`](src/phenotypic/gui/results_viewer/_zarr_routes.py:124)
  already uses. A validated marker whose stat tuple is unchanged cannot have become
  invalid *unless one of its artifacts changed*, so pair it with the existing
  processing-inventory stat sweep rather than re-hashing.
- Have `current_run_is_complete` do **one** `_walk_current_success` and pass the
  result into `current_aggregate_is_current`, instead of each walking independently.
- Memoize `load_processing_state` on the state file's stat tuple + event-log size, so
  a single completion query replays the event log once rather than 3–4 times.

Expected effect: a steady-state badge refresh drops from ~10⁴ hashes to ~10⁴ `stat()`s
on the first tick and ~0 on subsequent ones. **Cost:** moderate; **risk:** the
correctness argument must be written down, because the current design's whole point
is that it never trusts a cache.

### S2. Collapse the two overlapping viewer fingerprints
`OutputRoot`'s frozen [`consumed_state_fingerprint`](src/phenotypic/gui/results_viewer/_output_root.py:882)
and `CurationLabels`' self-updating [`_source_fingerprint`](src/phenotypic/gui/results_viewer/_curation_labels.py:760)
hash overlapping path sets with different lifecycles. Because the `OutputRoot` one is
frozen at discovery and `CurationLabels` writes to paths inside it, **marking one
colony makes the viewer report its own write as external drift** — the header badge
flips to `"Changed on disk"` / `danger` via
[`snapshot_refresh_status`](src/phenotypic/gui/_snapshot_status.py:59). It also
re-hashes the master **and** mirror parquets twice per click (before and after, under
the lock).

Fix: one owner. Let `CurationLabels` publish its post-write fingerprint back to the
binding state, or exclude GUI-owned mutable paths from the snapshot fingerprint the
way `snapshot_is_current()` already deliberately does
([`_output_root.py:545-548`](src/phenotypic/gui/results_viewer/_output_root.py:545)).
**Cost:** low. **Impact:** removes a false alarm on the most-used control and a
multi-GB rehash per curation click.

### S3. Drop `ctime_ns` from the inventory currency check
[`_inventory_is_current`](src/phenotypic/gui/results_viewer/_processing_inventory.py:462)
compares `st_ctime_ns`, which moves on any inode metadata change — a `chmod`, an
ownership fix, a hardlink, or an `rsync -a` that preserves mtime but not ctime. On a
shared HPC filesystem those are routine, and each one makes the whole binding report
"Changed on disk". `size` + `mtime_ns` already covers every write the contract makes.
**Cost:** one line plus a test. **Risk:** low — but confirm no test depends on
ctime-sensitivity.

### S4. Reduce nine completion sources to a layered two
The nine sources exist because each was added to close a real hole (the docstrings
say so — see [`manifest_completion_inventory`](src/phenotypic/_cli/_cli_completion.py:592)
for a documented case where the wrong basis flagged whole runs read-only). But the
marker chain (`image_complete/` → `aggregate_publication.json` → `run_completion.json`)
is now self-sufficient and self-validating. The others are either legacy-compat
(`manifest.json` counts, `processing_state` counts) or scheduler liveness
(`slurm_lifecycle`, `staged_orchestration`, `squeue`).

Proposal: declare **markers authoritative, scheduler state advisory, manifest
display-only**, and demote the manifest/processing-state contradiction rules in
[`classify_output_consistency`](src/phenotypic/gui/results_viewer/_output_consistency.py:117)
from `contradictory` (read-only) to a warning whenever `marker_authority_required` is
true. Legacy runs keep the current path. **Cost:** high. **Impact:** the largest
reduction in conceptual surface available, and it directly removes the "run flagged
read-only for a reason the user cannot act on" class of bug.

### S5. One event-log replay, not two
[`RunRegistry._latest_event_states`](src/phenotypic/gui/shell/_runs_registry.py:1172)
reimplements [`aggregate_state_from_events`](src/phenotypic/_cli/_cli_update_state.py:266)
with different semantics (stage demotion, no inventory fence). Two parsers of one
append-only log will drift. Fold the stage-demotion rule into the CLI aggregator as
an option and delete the GUI copy. **Cost:** low–moderate.

### S6. Collapse the four per-image marker directories
`image_complete/`, `stage2_done/`, `stage2_raw/`, `stage3_complete/` are four parallel
`<ds>/<stem>.*` trees answering four sub-questions about the same image, spelled in
three different places ([`_cli_stage2_token.py:42,124`](src/phenotypic/_cli/_cli_stage2_token.py:42),
the inline `"stage3_complete"` literal at
[`_cli_staged_resume.py:141`](src/phenotypic/_cli/_cli_staged_resume.py:141), and
`DIR_IMAGE_COMPLETE`). At minimum, put all four names in `_io_constants` next to each
other. Better: one per-image JSON with a `stages` object, so a completion query is one
file read instead of four `is_file()` probes. **Cost:** moderate (touches the staged
engine's resume logic). **Note:** `stage2_raw` must stay a separate binary artifact.

### S7. Give the stale owner record a repair path
Nothing in the codebase ever deletes or repairs `gui_launch_owner.json` **[verified]**.
If the GUI is SIGKILLed mid-local-run, the record stays `status: "running"`;
`rehydrate_from_sandbox` downgrades it to `unknown` **in memory only**
([`_runs_registry.py:773`](src/phenotypic/gui/shell/_runs_registry.py:773), `persist=False`),
and [`_assert_output_claimable_locked`](src/phenotypic/gui/shell/_runs_registry.py:1058)
then refuses the output forever, with no UI affordance to clear it. Add either a
liveness check (the record stores `pid` and `started_at`) or an explicit "release this
output" action. **Cost:** low. **Impact:** removes a permanent dead-end.

## Tier 2 — cheap, high-confidence hygiene

| # | Change | Sites |
|---|---|---|
| S8 | Use `aggregate_publication_marker_path()` instead of hand-joining `.phenotypic/aggregate_publication.json` | [`_output_consistency.py:380`](src/phenotypic/gui/results_viewer/_output_consistency.py:380) |
| S9 | Move the ~17 "shadow" state filenames into `_io_constants` — `slurm_lifecycle.json`, `slurm_jobs.jsonl`, `staged_*.json(l)`, `stage2_done`, `stage2_raw`, `stage3_complete`, `table-transitions`, the migrate manifests. Two of them (`staged_orchestration.json`, `staged_finalization_complete.json`) are **double-spelled across the CLI/GUI boundary** | [`_cli_slurm_lifecycle.py:33-36`](src/phenotypic/_cli/_cli_slurm_lifecycle.py:33), [`_cli_staged_orchestration.py:50-54`](src/phenotypic/_cli/_cli_staged_orchestration.py:50), [`_output_consistency.py:37-38`](src/phenotypic/gui/results_viewer/_output_consistency.py:37) |
| S10 | Add `DIR_LOG_GUI` / `DIR_LOG_SLURM` and compose from `logs_dir()`. The GUI hand-joins `.phenotypic/logs/{gui,slurm}` at 3 sites and never imports `logs_dir` (8 CLI callers, 0 GUI) | [`run_console/_slurm.py:438`](src/phenotypic/gui/run_console/_slurm.py:438), [`_slurm_observer.py:909-910`](src/phenotypic/gui/run_console/_slurm_observer.py:909) |
| S11 | Make the four naive control-manifest writers atomic — these **are** polled by concurrently launched SLURM workers | `_cli_recompile_slurm_scripts.py:257`, `_cli_migrate_manifest.py:600`, `_cli_migrate_slurm.py:176,311`, and the unlocked recompile `job_metadata.json` write at [`phenotypicCLI.py:3385`](src/phenotypic/phenotypicCLI.py:3385) **[verified]** |
| S12 | Replace the three GUI re-implementations of atomic write with `atomic_write_json` / `atomic_write_parquet` — they skip the `fsync` the SDK helper performs | `_curation_labels.py:453,953,961`; `_filtered_state.py:578` |
| S13 | Version-gate `DashboardManifestKey.VERSION` (written as `3` at exactly one site, read at **zero** **[verified]**) or delete the field. The owner record already models this correctly with `_OWNER_RECORD_VERSION` | [`_manifest_builder.py:766`](src/phenotypic/_cli/_dashboard/_manifest_builder.py:766) |
| S14 | Use `VERIFIED_PARQUET` and `ERROR_ANALYSIS_{PARQUET,CSV,HTML}` instead of re-spelling / format-string-deriving them | `_error_tab/_publication.py:596,849-851` |
| S15 | Use `DIR_PROGRESS` at the two literal sites in the same file that already imports and uses it | [`phenotypicCLI.py:839,841`](src/phenotypic/phenotypicCLI.py:839) vs [`:943`](src/phenotypic/phenotypicCLI.py:943) **[verified]** |
| S16 | Route the 13 `output_dir / "results"` CLI joins through `results_dir()` (migrate/recompile are the only holdouts) | `_cli_migrate.py` ×7, `_cli_recompile_tables.py` ×2, others |
| S17 | Key the sidebar classifier LRU on the markers it actually reads, not the run-root mtime. All four markers sit at depth ≥ 2, where a POSIX root-dir mtime never moves, so a run finishing under a live GUI leaves stale badges until manual Refresh — contradicting the cache's own docstring **[verified]** | [`shell/_classifier.py:201-207`](src/phenotypic/gui/shell/_classifier.py:201) |

## Tier 3 — deletions (pure subtraction)

| # | Delete | Evidence |
|---|---|---|
| S18 | `sdk_/monitor_slurm_jobs.py` — zero importers in `src/` or `tests/` **[verified]** | grep |
| S19 | `browse/_source_render.py`'s `browse_cache_base` / `cache_png_path` / `init_cache` / `wipe_cache` — zero production callers; only the module's own `__all__` and its test **[verified]**. The module docstring still describes this cache as live | [`_source_render.py:35-38`](src/phenotypic/gui/browse/_source_render.py:35) |
| S20 | The viewer DZI cache dir helper — the path is computed and documented as "not created"; the Plate surface stopped rendering pyramids in the Viv rebuild | [`_output_root.py:751`](src/phenotypic/gui/results_viewer/_output_root.py:751) |
| S21 | Eight zero-caller resolvers in `_io_constants`, two of which claim in their docstrings to replace inline blocks that still exist: `read_run_manifest`, `load_master_measurements`, `resolve_best_pipeline_path`, `resolve_qc_dir`, `recompile_status_dir`, `chunk_parquet_path`, `checkpoint_lock_path`, `chunk_manifest_path` | [`_io_constants.py:2107`](src/phenotypic/sdk_/_io_constants.py:2107) |
| S22 | `_LAST_DUMPED` — an unbounded module dict keyed on `id(image)`, i.e. a CPython object address reusable after GC **[verified]**. Key it on a content or revision token, or drop the memo | [`builder/_point_picker.py:549`](src/phenotypic/gui/builder/_point_picker.py:549) |
| S23 | Bound `_terminal_job_cache` (sacct states, never evicted) and `LocalRunner._instances` (self-documented `TODO(perf)`); garbage-collect `builder_tiles/<session>/` (one dir per browser tab, forever — its sibling `_preview_cache` wipes at launch *and* `atexit`) | `_manifest_builder.py:73`, `_runner.py:129`, `_point_picker.py:106` |

## Tier 4 — worth knowing, lower priority

- **S24.** Consolidate the three Zarr-member authorizers (§6). They agree today;
  nothing tests that they keep agreeing.
- **S25.** `RC_INTERVAL_LOG` at 1 000 ms drives six server-side callbacks. Split the
  cheap ones (cancel-disabled, registry revision) onto a slower interval.
- **S26.** `.gui_log/stdout.log` is append-only with no rotation and **the GUI never
  reads it back** — the UI reads only the in-memory `deque(maxlen=5000)`. Either read
  it (fixing S27) or stop writing it.
- **S27.** A page reload loses `RC_STORE_ACTIVE_RUN_RECEIPT` (memory storage), and no
  path re-attaches Cancel or the live log tail to a running job. Making the receipt
  `session` storage would fix most of this.
- **S28.** `browse/_source_lister.list_datasets` is an uncached unbounded-depth
  `os.walk` inside a Dash callback, with no progress reporting and no cancellation —
  unlike `OutputRoot.discover`, which has both.

## What *not* to change

Several things that look like duplication are load-bearing and documented; leave them:

- The **marker-last publication protocol** end to end (store root `zarr.json` last,
  marker after artifacts, aggregate after outputs, run-completion after aggregate,
  `chunk_state.json` after chunks, stage-2 token after the raw `.npy`).
- [`_walk_results_without_descending_into_stores`](src/phenotypic/gui/results_viewer/_processing_inventory.py:277) —
  the pruning is the optimization, and it says so.
- [`image_display_range`](src/phenotypic/gui/_shared/tiles.py:544) being uncached.
- The `store_publication_token` → HTTP **409** (not 404, not 410) contract.
- `BrowseCache`'s immutable revision-addressed entries — this is the model the rest of
  the caching layer should be measured against.
- `store_schema_version`'s single choke point in `require_readable_store` — the model
  S13 should copy.
