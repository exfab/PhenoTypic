# P4 plan review — `phase-4-finalize-run.md`

**Reviewed at** `869e9dee`, worktree `/bigdata/exfab/anguy344/PhenoTypic/.worktrees/cli-gui-state-tracking`.
**Subject:** `docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/phase-4-finalize-run.md` (5 tasks, 1280 lines).
**Against:** `specs/2026-09-03-cli-gui-state-tracking/design.md` §7 + D8, as amended by
`plans/.../OPEN-QUESTIONS.md#d-a-…`; `EXECUTION.md`; the P3 reports; `document-drift.md`.

**Verdict: do not dispatch.** Eight blocking findings. The dominant mechanism is the one the
brief predicted: **the plan is written against P3's plan, not against P3's shipped code.** Two
of the five rows in its recompile repoint table describe work P3 already did (and following
one would revert it); the load-bearing claim that authorizes its largest simplification is
false against `_cli_image_record.py`; and two of its "this has no home yet" premises name
things that already ship.

Everything below is measured at `869e9dee`. Every `file:line` was opened.

---

## Summary table

| | Finding | Kind | Severity |
|---|---|---|---|
| B1 | `publish_image_record` merges `stages` **only** — the whole-record merge claim is false | plan wrong | **BLOCKING** |
| B2 | Two repoint-table rows are already shipped; one would revert P3 | plan wrong | **BLOCKING** |
| B3 | `_cli_recompile_slurm_scripts.py:557/:569` — the site the 28 xfails name — is absent | plan incomplete | **BLOCKING** |
| B4 | Task 2 cannot be implemented in the files named; the `.part` writer is in `_core` | plan incomplete | **BLOCKING** |
| B5 | Task 3 Step 6's prescribed grep reaches **zero** master readers | plan wrong | **BLOCKING** |
| B6 | D8 deletion list names 4 symbols; the real set is ~10 modules incl. the GUI | plan incomplete | **BLOCKING** |
| B7 | INV-INPUTS' own test cannot reach the branch that violates INV-INPUTS | plan wrong | **BLOCKING** |
| B8 | `source_set_digest` already ships; the real work + a P1 compat arm are unspecified | plan wrong | **BLOCKING** |
| H1 | The plan's header names a root key Task 2 explicitly rejects | self-contradiction | High |
| H2 | The no-metadata case makes P1's divergence advisory fire on every image | plan incomplete | High |
| H3 | "Write both; never derive one" is what EXECUTION.md's HARD STOP rule 3 rejects | rule conflict — **needs a ruling** | High |
| H4 | `finalize_run`'s signature drops `include_dataset_column` | plan incomplete | High |
| H5 | The inventory of `finalize_post_master_outputs` is itself incomplete (5+1, really 5+3) | plan incomplete | High |
| H6 | Retiring `_consistent_embedded_join_keys` also deletes the mixed-authority refusal | plan incomplete | High |
| H7 | The measurements Parquet's post-inversion `join_status`/keys/digest is never stated | plan incomplete | High |
| M1–M6 | write order, missing group `zarr.json`, a never-executed assertion, gate scope, an unfalsifiable count, an unwatched `sdk_` module | mixed | Medium |
| L | 18 stale citations, individually listed | citation drift | Low |
| S1–S2 | Two spec sections D-A superseded and nobody amended | **spec wrong — user ruling** | — |

---

## BLOCKING

### B1. `publish_image_record` merges `stages` and nothing else. The plan's simplification destroys record identity.

**Plan location:** "What recompile actually needs from a marker, and why the collapsed form is
*simpler*", and the repoint-table row for `_republish_table_marker`.

> **P3's merge rule removes the need for the round trip entirely** — `publish_image_record`
> merges rather than replaces (CAN-6), so recompile publishes the **new `artifacts` only** and
> the merge preserves identity and `stages` untouched.
>
> `_republish_table_marker` (`:58-82`) | **Delete the seven-field read-back.** … ~40 lines → one call.

**What the code says.** `src/phenotypic/_cli/_cli_image_record.py:99-101`:

> **`stages` is a contribution, not a replacement (CAN-6 rule 1).** It is unioned with whatever
> is on disk.

That scoping is exact. At `:157-176` the record is rebuilt from the arguments:

```python
merged = _existing_stages(output_dir, dataset, image_stem)
merged.update({str(key): dict(value) for key, value in stages.items()})
record = {
    "version": RECORD_VERSION, "work_id": work_id, "dataset": dataset,
    "image_stem": image_stem, "relative_image_path": relative_image_path,
    "mode": mode, "provenance": provenance, "stages": merged,
    "artifacts": descriptors, "attempt_id": attempt_id,
    "lifecycle_epoch": lifecycle_epoch, "completed_at": _now(),
}
```

`work_id`, `dataset`, `image_stem`, `relative_image_path`, `mode`, `attempt_id` and
`lifecycle_epoch` are **required keyword-only parameters** (`:80-95`). Nothing about them is
merged; `_existing_stages` is called for `stages` alone.

**Consequence of following the plan literally.** Either `TypeError` at the call site, or — if
the implementer supplies placeholders to satisfy the signature — the record's `work_id` is
overwritten, `valid_image_success` rejects every recompiled image, `authorized_measurement_sources`
returns `{}`, and `finalize_run` writes an **empty master and raises nothing**. That is verbatim
the CAN-22 failure the comment at `_cli_completion.py:955-964` exists to prevent:

> `{}` is a VALID schema-3 result meaning "no successful measurements yet", so P4's
> `finalize_run` would write an empty master and raise nothing. A successful-looking run that
> discarded every measurement.

**Second half, same row.** `_marker_artifacts` (`_cli_recompile_tables.py:41-57`) re-resolves
**every** artifact role on the record, not just `measurements`; `_republish_table_marker`
(`:60-84`) hands the whole map back so all of them are re-fingerprinted. "Publish updated
`artifacts`" would shrink the certified set to `measurements`, silently dropping the `image` and
`overlay` roles from `valid_image_success`'s coverage. The plan's containment-check argument is
correct in isolation (`_cli_image_record.py:149-155` does `resolve(strict=True)` +
`relative_to(output_root)`), but that is not the whole of what `_marker_artifacts` does.

**Required before dispatch:** strike the whole-record-merge claim; keep the seven-field
read-back (now reading the *record*, see B2); delete only `_marker_artifacts`' duplicated
containment check if you want that saving, and drop the "~40 lines → one call" estimate.

---

### B2. Two of the five repoint-table rows are already shipped, and following one reverts P3.

| Plan row | Actual state at `869e9dee` |
|---|---|
| `_standalone_marker_sources` (`:135-150`) — "Glob `DIR_IMAGE_RECORDS` instead of `DIR_IMAGE_COMPLETE`" | **Already both**, each on its own predicate: `_cli_recompile_tables.py:180-183` — `(progress / DIR_IMAGE_RECORDS, record_rejection), (progress / DIR_IMAGE_COMPLETE, marker_rejection)`. Its docstring (`:137-176`) calls itself "the seventh site of the defect fixed in `authorized_measurement_sources`" and explains why the legacy arm stays. |
| `_republish_table_marker` reads the legacy marker | **Already reads the record.** `_replace_and_republish_table` passes `record_path = image_record_path(...)` at `_cli_recompile_tables.py:139-145`, with a comment naming the P3 fix and the `recompile_store_lock_path` half of it. |
| `:100` isinstance → `PreparedImageTables` | **Still live**, at `:102`. This is the real crash the plan describes. |

Executing the first row as written deletes the `DIR_IMAGE_COMPLETE` arm — which the brief
independently flags as `marker_rejection`'s only caller in `src/` and a deliberate legacy read.
That is a regression dressed as the assigned work.

---

### B3. The site the 28 strict xfails name is not in P4's repoint table.

The xfail that P4 owns names its sites itself — `tests/unit/cli/test_cli_recompile.py:75-85`:

> "--mode recompile reads the legacy image_complete/ marker at five call sites until P4
> repoints them: _cli_recompile_recovery (409, 499, 659, 731) and
> **_cli_recompile_slurm_scripts (557)**. Path and SUCCESS_MARKER_VERSION->RECORD_VERSION move
> together, because those functions return None/False on a version mismatch and a path-only
> repoint would disable overlay and table authority repair silently."

`tests/unit/cli/test_embedded_measurement_recompile.py:31-40` repeats the same pair.

**Measured decoration count:** `grep -c _RECOMPILE_READS_THE_LEGACY_MARKER_UNTIL_P4` returns
3 in `test_cli_recompile.py` and 27 in `test_cli_recompile_slurm.py` — 1 definition each, so
**28 decorations**, matching the brief.

**P4's repoint table lists only `_cli_recompile_recovery.py`.** `_cli_recompile_slurm_scripts.py`
appears in the File Structure table for one unrelated reason (the `_consistent_embedded_join_keys`
serialisation at `:186-202`). The marker site is real and live:

- `_cli_recompile_slurm_scripts.py:557` — `marker_path = image_completion_marker_path(...)`
- `:569` — `marker.get("version") != SUCCESS_MARKER_VERSION`
- `:430`, `:493` — `"Cannot restore marker authority: …"`, which is where the brief's 8
  handled-domain-error xfail instances land.

Nothing in P4 says to remove the 28 markers either. This is **drift-register Entry 32's exact
shape**: a tripwire that fires on the fix, pointed at a step that will not do the thing. Under
the plan as written those instances stay at `XFAIL` — a *passing* state — and the debt is
recorded as paid.

**Required before dispatch:** add `_cli_recompile_slurm_scripts.py:557/:569` to the repoint
table; add an explicit step "delete the 28 `_RECOMPILE_READS_THE_LEGACY_MARKER_UNTIL_P4`
decorations and both definitions"; and budget for the 8 authority-repair instances and the one
inverted instance (M3), whose post-repoint outcomes are not knowable from the current run.

Related census correction: the plan's `_cli_recompile_recovery.py:52,387,477,637,709` is **four
call sites, not five** — `:409`, `:499`, `:659`, `:731`. The fifth (`:57`) is a docstring
mention inside `recompile_store_lock_path`, not a call.

---

### B4. Task 2 cannot be implemented in the files it names. `_core` appears nowhere in P4.

Task 2 says: *"Extend the `.part` writer to emit `tables/metadata/pht-metadata.parquet` … before
`OME/zarr.json` and the root. Extend the root's `attributes.phenotypic.tables` with a `metadata`
descriptor."* The File Structure table assigns this to `src/phenotypic/sdk_/_measurement_tables.py`.

**The `.part` writer and the root-attribute assembly are in `_core`:**

- `src/phenotypic/_core/_image_parts/_image_io_handler.py:1377-1390` — writes the table into the
  part, builds the descriptor, folds it into the root that is written last.
- The parameter is typed `PreparedEmbeddedMeasurementTable` at **three** signatures:
  `:1076` (`save2zarr`), `:1139`, `:1224`; threaded at `:1126`, `:1204`.

`sdk_/_measurement_tables.py` owns only `write_embedded_measurement_table` (`:86-107`) and
`build_measurement_table_descriptor` (`:109-129`) — the pieces `_core` calls.

**Also missing from the File Structure table:** the *forward* producer,
`OutputManager.save_image_store` at `_cli_output_manager.py:1936` (`prepare_embedded_measurement_table`
→ `save_kwargs["measurement_table"]` at `:1948`). Only `replace_image_store_measurements`
(`:1970-2001`, the `--mode measure` producer) is listed. The plan's own CAN-3 argument is that
missing one producer silently un-inverts a path; the forward one is missed.

**Type-change blast radius also unlisted:** `tests/_output_layout.py:170-176`,
`tests/unit/gui/results_viewer/conftest.py:98-110`, `.../test_measurement_source.py:40-48`,
`.../test_measurement_routes.py:51-58` all construct `PreparedEmbeddedMeasurementTable` directly.

---

### B5. Task 3 Step 6's prescribed grep reaches zero master readers.

> Route every in-repo master read through it — `grep -rn 'master_measurements_parquet_path' src/`
> and convert each call site.

**Measured.** That grep returns four `src/` hits, and every one is a **writer or a path
construction**, not a read:

| Site | What it is |
|---|---|
| `_cli_output_manager.py:1484` | `master_pq_path` for the write at `:1502-1511` |
| `_cli_recompile_worker.py:780` | the recompile master write |
| `_cli_chunk_writer.py:236` | the mid-run chunk master write |
| `_cli_completion.py:1013` | `"master_parquet"` descriptor path in the aggregate proof |

**Every actual master read goes through `BundleLayout.master_parquet`** (`sdk_/_io_constants.py:2681-2683`):

- `gui/results_viewer/_output_root.py:320`
- `gui/results_viewer/_processing_inventory.py:202`, `:373`
- `gui/results_viewer/_curation_labels.py:417` (`_read_clean_master`), `:763`
- `gui/results_viewer/_error_tab/_publication.py:125`
- `gui/results_viewer/_qc_tab/review/_data.py:84`
- `sdk_/_metadata_migration.py:1036`, `:1065`

P6 does not mention `read_master_measurements`, `_master_io` or `master_parquet` at all (only
P7 does, at `phase-7-migrate-mode.md:964, 1064, 1070, 1078, 1134`). So under the plan as
written, U-3's stamp ships with a reader that no consumer calls — precisely the
`DashboardManifestKey.VERSION` pattern Step 6's own rationale says is indefensible to ship
while P6 deletes it.

Note the collision with the *curation* test the plan adds: `test_curation_re_keying_still_works_against_the_intrinsic_master`
exercises `_curation_labels.py`, which is one of the six readers the conversion misses.

---

### B6. Task 4 Step 4's D8 deletion list is a fraction of the dependency set, and it includes the GUI.

Named for deletion: `MASTER_MEASUREMENTS_CSV`, `master_measurements_csv_path()`,
`BundleLayout.master_csv`, `load_master_measurements()`, the proof's `master_csv` entry, plus
"ten test files" (Q6 — **verified exactly 10**, listed at the end of this report).

Unnamed and broken by those deletions:

| Site | Why it breaks |
|---|---|
| `gui/_config.py:67,137` | re-exports `MASTER_MEASUREMENTS_CSV` in `__all__` — **import-time break** |
| `gui/_schema_cache.py:26,43` | `_FILES_BY_SOURCE["master_measurements"] = (MASTER_MEASUREMENTS_PARQUET, MASTER_MEASUREMENTS_CSV)` — the GUI's no-pyarrow CSV fallback for the master |
| `sdk_/_metadata_migration.py:1036,1066` | iterates `(layout.master_parquet, layout.master_csv)` |
| `_cli_chunk_writer.py:47,232` | writes the master CSV mid-run |
| `_cli_recompile_worker.py:39,771-778` | writes it, and **re-raises** on failure |
| `sdk_/_io_constants.py:353` | `_reserved_analysis_artifact_stems()` includes it |
| `sdk_/_io_constants.py:27,47,317-324,1271-1273,2290-2301,2686-2688` | definitions + module docstring |

**And one semantic change the plan never states.** In `_aggregate_measurements_unlocked` the CSV
write is aggregation's **success signal** and the Parquet write is best-effort:

```
_cli_output_manager.py:1493-1500   master_csv_saved = _guarded_terminal_best_effort(..., write_master_csv, default=False)
                                   if not master_csv_saved: return None
_cli_output_manager.py:1506-1511   _guarded_terminal_best_effort(..., write_master_parquet,
                                       warning="Failed to save master Parquet (CSV was saved)")
_cli_output_manager.py:1542        return master_csv_path
```

Deleting the CSV requires inverting that. The return value propagates:
`aggregate_measurements` (`:1545`) → `OutputManager.aggregate_master_csv` (`:2004`, docstring
*"Returns: Path to master_measurements.csv"*) → `phenotypicCLI.py:2982` (`master_path = …`), and
`tests/integration/cli/test_staged_gpu_local.py:520,529` monkeypatches the method by name. None
of this is in the plan.

Finally: the aggregate proof's `required_outputs` drops from four descriptors to three
(`_cli_completion.py:1011-1014`, assembled at `:1047`) with **no `AGGREGATE_PROOF_VERSION` bump**
specified, while `valid_aggregate_snapshot` / `_run_state.py:1100` validate whatever the proof
lists.

---

### B7. INV-INPUTS' gate test cannot reach the branch that violates INV-INPUTS.

`test_finalize_run_ignores_every_stale_intermediate` is the phase's gate. It plants a poisoned
`_dataset_aggregated.parquet` (among others) and asserts it is ignored.

**The `_dataset_aggregated.parquet` preference is on the legacy arm only**
(`_cli_output_manager.py:1420-1432`):

```python
authorized_sources = authorized_measurement_sources(output_dir)
if authorized_sources is None:
    flush_trailing_measurements_if_chunked(output_dir)          # writes _dataset_aggregated from chunks
    path_to_dataset = measurement_sources_by_path(discover_measurement_sources(...))
else:
    path_to_dataset = authorized_sources
```

`discover_measurement_sources` prefers the aggregate (`_measurement_sources.py:130-165`). On a
forward tree `authorized_sources` is not `None`, so the arm is unreachable — **and the planted
test runs on a forward tree.** It is green whether or not that arm survives into `finalize_run`.
Worse, the arm calls `flush_trailing_measurements_if_chunked`, which *manufactures*
`_dataset_aggregated.parquet` from `chunks/` — the exact input §7.5 forbids.

The plan never says what `finalize_run` does with the legacy arm. Step 5's mutation ("add a
fast path to step 1, confirm the test fails, remove it") proves only that a **newly added** fast
path is caught. Its prose is also self-contradictory: *"Add a `_dataset_aggregated.parquet` fast
path"* followed two lines later by *"**That fast path … is in the current code.**"*

**Required before dispatch:** state the legacy arm's fate explicitly (drop it, or keep it and
narrow §7.5's claim to the authorized path), and add a case whose fixture makes
`authorized_measurement_sources` return `None` so the assertion reaches the branch.

---

### B8. Task 3 Step 7's premise is false, and cutting `publication_id` breaks a shipped P1 compatibility arm.

> `source_set_digest` had no home in any phase before this step — it appeared only in the
> README's digest table and two prose mentions in P5. This is that home.

**Both fields already ship.** Written today in the aggregate proof:

```
_cli_completion.py:1045-1046   "source_set_digest": canonical_digest(sorted(source_work_ids)),
                               "source_image_count": len(source_work_ids),
```

Read at `_cli_completion.py:786-788` (`current_aggregate_is_current`) and
`sdk_/_run_state.py:1214-1215` (rule 1, clause 2).

**The real work is specified by shipped P1 code**, `sdk_/_run_state.py:1222-1240`:

> U-4 cuts `publication_id` and puts `source_set_digest` in the **run** proof … **That writer
> change lands in P4**; until it does, today's run proof carries neither field and the values
> live in the aggregate proof, **bound to the run proof by `publication_id`**.
> Both shapes are read here so that P1 lands on today's trees and keeps working across P4's
> writer bump, with no window in which the two comparisons silently stop being made.

So P4 must (a) add the two fields to `publish_run_completion_evidence`
(`_cli_completion.py:1140-1165`), and (b) decide the fate of `publication_id`, which is minted
at `:1032` and copied into the run proof at `:1151-1153`. The plan says only "`publication_id` is
cut", with no location and no statement about the fallback arm. If it is cut from the aggregate
proof while `_source_set_binding`'s fallback survives, every pre-P4 tree's binding returns `None`
→ `current_run_is_complete` false → **complete runs read incomplete**. If the fallback is
removed in the same commit, say so; it is P1's code and P4 is silently editing it.

---

## HIGH

### H1. The plan's own header names a root key Task 2 explicitly rejects.

Header, line 27:

> each store's `phenotypic.metadata.snapshot_sha256` records which one, and `resolve_run_state`
> raises an advisory when they diverge (P1 Task 5).

Task 2 (Step 3) rules the opposite, correctly:

> **The snapshot digest goes in a NEW root key, and the name matters (flow-r2 C5).** … That key
> is **already taken**: `PhenotypicAttr.METADATA` holds `{protected, public, imported}` …
> Use `attributes.phenotypic.metadata_table`.

Task 2 matches what P1 shipped — `sdk_/_run_state.py:364-378` (`_METADATA_TABLE_ATTR = "metadata_table"`,
with a comment saying exactly why) and the reader at `:653-667`. The header is the stale half.
This is the "accurate correction appended, document left self-contradictory" defect, in the
document's normative summary. *(D-A itself carries the same stale spelling at
`OPEN-QUESTIONS.md:99` — see S2.)*

### H2. The no-metadata case makes P1's shipped advisory fire on every image of every run.

`resolve_run_state` (`sdk_/_run_state.py:1294-1307`):

```python
current_metadata = config.get("metadata_sha256")
diverged = sorted(... if _stage_value(image, _SNAPSHOT_SHA256_ATTR) not in (None, current_metadata))
```

For a run with no `--metadata`, `current_metadata` is `None` and the producer records
`metadata_snapshot_sha256=""` (`_embedded_measurement_tables.py:56-63`). If Task 2 writes
`metadata_table` unconditionally, `"" not in (None, None)` is `True` and **every store on every
metadata-free run** is reported diverged. The plan says only "Extend the root's … with a
`metadata` descriptor" and "Use `attributes.phenotypic.metadata_table`" — it never states the
`not_requested` / `no_common_keys` behaviour. The code two blocks above the filter carries the
warning: *"An advisory that is always on teaches people to ignore the one that will matter"*
(`:1280-1281`).

**Required:** state that `metadata_table` is omitted entirely when no metadata table is written,
and pin it with a test on a metadata-free run.

### H3. "Write both; never derive one from the other" is what EXECUTION.md's HARD STOP rule 3 says to reject. **Needs a user ruling.**

Task 2, Step 3: *"Mirroring it into the root at promote time … keeps the Parquet copy as the
authority the Parquet itself carries. **Write both; never derive one from the other at read time.**"*

`EXECUTION.md:653-655`:

> 3. Does it add a *second home* for a value that already exists? → that is the defect this
>    change removes, not a fix. **Reject it without asking.**

The value does already exist: `METADATA_SNAPSHOT_SHA256 = "phenotypic.metadata.snapshot_sha256"`
(`sdk_/ngff_.py:95`), written at `sdk_/_measurement_tables.py:47`, and **required** by
`_valid_embedded_measurement_contract` (`:215-227`). The root mirror is a second home for it.

The performance argument is sound (a root JSON read vs a Parquet footer open per store, on the
deep path, from `sdk_`), P1 already shipped the root reader, and the plan states the trade
honestly. But the rule is written mechanically and this is exactly its third clause. It needs an
explicit ruling and a row in P7 Task 6's register classifying it, not a plan paragraph.

The same question applies to Step 6's `phenotypic.master_schema_version`: it is a key that
`read_master_measurements` **branches on**, which by `EXECUTION.md:648-650` test 1 reads as
"tracked state → stop and ask". I do not think it is tracked state in spirit — it is a schema
tag on a derived artifact — but the plan should say so and register it rather than leave the
classification to the implementer.

### H4. `finalize_run`'s signature drops `include_dataset_column`.

The Interfaces block gives `finalize_run(output_dir, *, dataset_names, pipeline, metadata_csv,
no_qc, study_config, shard_paths, commit_guard)`. Task 4 Step 3 says
`_aggregate_measurements_unlocked` "delegates its body".

`include_dataset_column` is a live parameter of that body: declared at
`_cli_output_manager.py:1354`, consumed at `:1452` (`aggregate_parquet_files`), supplied by
`aggregate_measurements` (`:1548,1566`) and `OutputManager.aggregate_master_csv` (`:2016`,
`self.include_dataset_column`), serialised into the recompile SLURM finalizer task
(`_cli_recompile_slurm_scripts.py:149,200`) and read back at `_cli_recompile_worker.py:368`.
It cannot be dropped without changing behaviour for `include_dataset_column=False` runs.

### H5. The plan's inventory of `finalize_post_master_outputs` is itself incomplete.

The plan's warning box is right about the mechanism and wrong about the count:

> Its docstring (`_cli_output_manager.py:1023-1050`) numbers **five** steps … plus the
> `order_measurement_columns` call at `:1104-1106` that is not in that numbered list at all.

Two further un-numbered side effects exist and would be dropped by the same rewrite:

- `migrate_legacy_qc(output_dir)` — `:1063-1067`
- `write_rembi_manifest(...)` → `deliverables/rembi.yaml` — `:1115-1131`, and it is explicitly
  fed from the **mirror**, so dropping it also drops a mirror-rule obligation.

So the real shape is five numbered plus **three** un-numbered, not one. *(The numbered list is at
`:990-1017`, not `:1023-1050` — see the citation table.)*

### H6. Retiring `_consistent_embedded_join_keys` also deletes the mixed-authority refusal.

The function carries **two** independent guards (`_cli_output_manager.py:914-966`):

```
:934-937   raise ValueError("Cannot aggregate mixed embedded and legacy measurement authority")
:963-966   raise ValueError("Embedded measurement tables have mixed metadata digests or join keys")
```

The plan argues only about the second (D-A manufactures mixed digests on the rolling-input path;
the join is now global; the recorded keys become provenance). That argument does not touch the
first. Deleting the two call sites (`:1435-1439`, `_cli_recompile_worker.py:814`) removes the
mixed-authority abort with nothing named as replacing it. `_cli_recompile_tables.py` raises
*"Legacy external measurement Parquets require --mode migrate"* on its own path, and
`datasets_needing_migration` advises — but the plan should say which of those covers the case, or
retain the check.

### H7. The measurements Parquet's post-inversion provenance triple is never stated.

The File Structure row says only *"`_valid_embedded_measurement_contract` **rejects
`join_status == "not_requested"` with a non-empty digest** … Revise in the same commit."* It
never says what `join_status`, `join_keys` and `metadata_snapshot_sha256` the **measurements**
table carries after the inversion. Three consumers read that triple:

- `_valid_embedded_measurement_contract` (`sdk_/_measurement_tables.py:132-231`), which gates
  `replace_embedded_measurement_table`'s in-place branch at `:285-291`;
- `embedded_measurement_table_matches` (`_cli/_embedded_measurement_tables.py:106-131`), which
  compares with `check_metadata=True` — so every migrator decision depends on the exact bytes;
- `_consistent_embedded_join_keys` (being deleted, but live in `_cli_recompile_slurm_scripts.py:186-202`
  at submission time until that field is removed).

One unspecified value, three gates. Name the triple explicitly in Task 1's `PreparedImageTables`
contract.

---

## MEDIUM

**M1 — the plan states a `.part` write order the code does not have.**
Task 2: *"**Order is load-bearing and is INV-PROVEN's first obligation:** chunks → both tables →
`OME/zarr.json` → root `zarr.json` → `promote_store`."* Actual order in
`_core/_image_parts/_image_io_handler.py`: arrays → OME group + `OME/zarr.json` (`:1368-1375`)
→ tables (`:1377-1388`) → root last (`:1390+`). Root-last is preserved either way, so this is not
a correctness defect — but a sequence the plan calls load-bearing, stated wrongly, is the
Category-E shape the brief asks for.

**M2 — Task 2 omits the `tables/metadata` group `zarr.json` and any schema-version constant.**
`write_embedded_measurement_table` writes group documents for `tables/` and `tables/measurements/`
(`sdk_/_measurement_tables.py:92-104`), and `_valid_embedded_measurement_contract` checks group
documents by exact equality (`:168-182`). A Zarr v3 hierarchy needs one for `tables/metadata/`
too. Task 2 says only "emit `tables/metadata/pht-metadata.parquet`". No analogue of
`MEASUREMENT_TABLE_SCHEMA_VERSION` is specified for the new descriptor either.

**M3 — one xfailed test carries an assertion that has never executed.**
`tests/unit/cli/test_cli_recompile_slurm.py:2905` — `assert not overlay.exists()` sits *after*
the `pytest.raises(SlurmGenerationInactiveError)` block (`:2892-2903`). While the test fails with
`DID NOT RAISE`, line 2905 never runs. When P4 repoints and the raise starts happening, that
assertion executes for the first time and its outcome is unknown. Drift Entry 34 is about this
exact mechanism. P4 budgets nothing for it.

**M4 — Task 5's test lives outside the phase gate.**
Task 4 Step 5's gate is `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit -q`. Task 5 writes to
`tests/integration/` and gives no command ("Step 2: Run it. Expected: PASS"). `testpaths` does
include `tests/integration` (`pyproject.toml:220`), but only for a bare `pytest`. Give Task 5 an
explicit command. Its helper `_run_full_pipeline` is also unnamed anywhere else in the plan.

**M5 — an unfalsifiable count.**
*"**Measured: 33 marker-format reads across the two files**, not the ~15 first estimated."* The
plan gives no criterion for "marker-format read", so the number cannot be reproduced and cannot
be used to check the sweep's completeness. This is the shape of P3's 384-vs-1152. Either define
the criterion (e.g. "textual occurrences of `image_completion_marker_path`,
`SUCCESS_MARKER_VERSION` or `DIR_IMAGE_COMPLETE`") or drop the number. Related: the file sizes
quoted alongside it are stale — see the citation table.

**M6 — the new `sdk_` module is outside INV-LAYER's watched set.**
`tests/unit/sdk_/test_run_state_layering.py:60-68` lists exactly seven modules by `__file__`.
`sdk_/_master_io.py` (Task 3 Step 6) would not be covered. The comment at `:55-59` explains why
`sdk_/__init__.py` was added — the same argument applies to any new `sdk_` module that
`__init__` re-exports. Add it to `_MODULES` in the same commit.

---

## LOW — stale citations

Each verified at `869e9dee`. None is individually blocking; collectively they are the reason
B1/B2 were possible, since a reader checking one citation lands on unrelated code and stops.

| Plan citation | Actual |
|---|---|
| `_cli_completion.py:768` (`authorized_measurement_sources`) | `:918` |
| `_cli_completion.py:888` (`master_csv` in `required_outputs`) | `:1011`; the list itself is `:1047` |
| `_cli_completion.py:868` (aggregate proof `required_outputs`) | `:1047` |
| `_cli_completion.py:41-47` ("the record's content anchor") | an import block. The anchor is `_store_artifact_matches`, `:110-123` |
| `_cli_output_manager.py:1023-1050` (the five numbered steps) | `:990-1017` |
| `_cli_output_manager.py:962-965` (`_consistent_embedded_join_keys` raise) | `:963-966` |
| `_cli_recompile_worker.py:764` (`_run_post_master_steps`) | `:805` |
| `_cli_recompile_worker.py:802` (`finalize_post_master_outputs`) | imported at `:816` |
| `_cli_recompile_worker.py:785` (`_consistent_embedded_join_keys`) | `:814` |
| `_cli_recompile_worker.py:777-787` (the two-caller branch) | `:805-830` |
| `_cli_recompile_tables.py:100` / `:58-82` / `:39-56` / `:135-150` | `:102` / `:60-84` / `:41-57` / `:137-…` |
| `_cli_recompile_recovery.py:38,782` | `:39`, `:804` |
| `_cli_recompile_recovery.py:52,387,477,637,709` | `:409`, `:499`, `:659`, `:731` — **four** call sites; `:57` is a docstring |
| `_cli_migrate.py:1331,1337` / `1329-1341` | `:1374`, `:1380` / `:1355-1385` |
| `_cli_migrate_image.py:278,314,777,796` | `:293`, `:329`, `:838`, `:857` |
| `_cli_migrate_image.py:278-288` / `:766-800` | `:290-300` / `:827-870` |
| `sdk_/CLAUDE.md:132-136` | `:134-138` (content matches verbatim) |
| `sdk_/_measurement_tables.py:284-290` | `:285-291` |
| "`_cli_recompile_tables.py` (292 lines)" | **333** |
| "`_cli_recompile_recovery.py` (838 lines)" | **860** |
| "1,130 lines that read the marker format" | **1193** |

**Verified correct** (recorded so a re-checker need not redo them):
`_embedded_measurement_tables.py:42, 55, 88-95, 106-131`;
`sdk_/_measurement_tables.py:109-129, 216-227, 242`;
`_cli_output_manager.py:132, 139-142, 143-153, 874, 884-893, 914-966, 969, 1077-1086, 1104-1106, 1351, 1435-1439, 1526, 1545, 1552, 1970-2001, 1992-1995`;
`sdk_/ngff_.py:95, 569-580, 1130-1138`;
`_cli/_metadata_join.py:166-171, 187-192`;
`sdk_/_metadata_helpers.py:111`;
`tests/unit/sdk_/test_ngff_promote.py:66` (and its docstring's three-subsystem claim, `:67-88`);
Q6's "ten test files" — exactly 10: `tests/_output_layout.py`, `tests/unit/cli/test_cli_chunk_writer.py`,
`test_cli_completion.py`, `test_cli_migrate_authority.py`, `test_cli_output_manager.py`,
`test_cli_recompile.py`, `test_cli_recompile_slurm.py`, `test_cli_v2.py`,
`test_recompile_manifest_completion.py`, `tests/unit/sdk_/test_io_constants.py`.

---

## "The plan is right and the spec is wrong" — needs a user ruling

**S1. §7.5 still says shards are namespaced by `scheduler_epoch`.**
`design.md:748-750`: *"Measurement shards are **per-invocation scratch, namespaced by
`scheduler_epoch`**, so a prior run's shards can never be merged."* The user ruled for
**clearing** instead, and P4 carries the correction inline in `finalize_run`'s docstring block.
The spec was never amended. A P5/P7 reader going to the spec gets the withdrawn design.

**S2. §7.4 still lists seven steps with backfill as step 6.**
`design.md:698-706` (the numbered block) and `:722` (*"re-joining the mirror **and
re-backfilling every store**"*) both survive D-A unamended. P4's "What D-A changes from spec §7"
section covers a plan-reader; the spec does not cover anyone else. Related: `OPEN-QUESTIONS.md:99`
still spells the root key `phenotypic.metadata.snapshot_sha256` (see H1), and `:101` says
*"See P1 Task 5 and **P4 Task 6**"* — P4 has five tasks (`:220, 329, 490, 1160, 1246`). A pointer
to a name that does not exist, of the Entry-32 shape; the work it names shipped in P1.

---

## Constraints checked and found NOT violated

- **INV-LAYER.** No P4 task introduces an `sdk_` → `phenotypic._cli` import. The new
  `sdk_/_master_io.py` reads a Parquet and needs no `_cli` symbol. (Coverage gap only — M6.)
- **INV-PROVEN, first obligation.** D-A's cut is respected; no task reintroduces a
  post-promotion write on a *new* path. The in-place-branch repair (CAN-3) makes the *existing*
  violation stop being one by forcing a re-promote, which strengthens the invariant. The plan
  states the unmeasured cost honestly and defers measurement — correct.
- **`deliverables/metadata.csv` byte-exactness.** Every P4 metadata read is a read:
  `metadata_csv_deliverable_path` in the two producers (`_cli_output_manager.py:1932-1938`,
  `:1990-1996`), `pl.read_csv` in `join_metadata` and `_append_metadata_only_rows`. Nothing in
  P4 writes the snapshot.
- **Master/mirror rules.** Task 3 preserves them and adds the tests §7.3 demands (curation
  re-keying, master-keeps/mirror-drops). The FINAL master write stays inside
  `finalize_post_master_outputs`' caller. The one risk is H5 — a rewrite that drops
  `write_rembi_manifest` would drop a mirror-fed deliverable.
- **State-artifact budget (4/3/2).** No new *tracked* state file. Two keys need classification
  and a register row rather than a stop: `metadata_table.snapshot_sha256` (H3, second home) and
  `phenotypic.master_schema_version` (branched on by `read_master_measurements`).

---

## What I would require before dispatch

1. **B1** — strike the whole-record-merge claim and re-specify `_republish_table_marker`.
2. **B2** — re-derive the repoint table from `869e9dee`, not from P3's plan.
3. **B3** — add `_cli_recompile_slurm_scripts.py:557/:569`; add an explicit step deleting the 28
   decorations; budget for the 8 authority-repair instances and the 1 inverted one.
4. **B4** — add `_core/_image_parts/_image_io_handler.py` (3 signatures) and
   `_cli_output_manager.py:1936` to the File Structure table, with the four test-fixture sites.
5. **B5** — replace the grep with one that finds `BundleLayout.master_parquet`, and enumerate the
   eight reader sites; decide whether the conversion is P4's or P6's and say so in both plans.
6. **B6** — replace the four-symbol deletion list with the measured dependency set, and specify
   the aggregation success-signal inversion and the proof-version question.
7. **B7** — state the legacy arm's fate; add a case that reaches it.
8. **B8** — correct the premise; specify the run-proof writer, `publication_id`'s fate, and
   whether `_source_set_binding`'s fallback arm survives.
9. **H1–H7** — as written above; **H3 is a rule conflict and needs the user, not the plan.**
10. **S1/S2** — spec amendments, user-gated.

*Reviewed by the P4 plan reviewer. No file outside this report was modified; no command with
side effects was run.*

---

# Round 2 — verification of the repair

**Re-reviewed at `869e9dee` + uncommitted edits to four files** (not three: `EXECUTION.md`
also changed, +60 lines, carrying the two rulings and the new retirement-condition rule).
`phase-4-finalize-run.md` 1280 → **1915** lines. No code touched.

Every citation below was re-opened. Where my Round 1 correction was itself wrong, it is
marked and corrected in §R2.4.

## R2.1 — Prior findings

### BLOCKING

| | Verdict | Evidence |
|---|---|---|
| **B1** | **RESOLVED** | The ⛔ block at plan `:227-266` quotes the withdrawn claim in full, cites `_cli_image_record.py:99-101` (the merge rule's own scoping sentence), `:80-95` (the seven required kwargs), and `:158-174` (the record rebuilt from arguments) — all verified. It states both failure directions and names the CAN-22 comment. Critically, the **instruction changed, not just the prose**: the repoint table row now reads *"**None required.** Optionally drop `_marker_artifacts`' duplicated `relative_to` check (`:41-57`), keeping its every-role resolution."* The "~40 lines → one call" estimate is explicitly withdrawn with the claim that justified it, and the every-role point is preserved. Verified `_cli_image_record.py:152-156` does the `resolve(strict=True)` + `relative_to`, so the one thing the plan still permits deleting is genuinely duplicated. **An implementer following the repaired text leaves `_republish_table_marker` alone, which is correct.** |
| **B2** | **RESOLVED** | Both rows corrected against shipped code, with ⛔ *"Do not touch it"* on `_standalone_marker_sources` and the reason (reverting P3 re-breaks legacy trees for the whole P4→P7 window, because the schema gate is disarmed until P7). Verified: def at `_cli_recompile_tables.py:150`, docstring `:151-176`, two-shape tuple `:181-183`, and `_replace_and_republish_table` passing `record_path` at `:139`. The plan adds a decision the first draft lacked — do the repointed recovery sites read one shape or two — with a default (both, following P3) and a retirement condition. |
| **B3** | **RESOLVED** | `_cli_recompile_slurm_scripts.py:557,569` now appears in the File Structure table *and* the repoint table, with the two `Cannot restore marker authority` arms (`:430`, `:493`) named. Xfail removal is its own step (Task 4 Step 4b) with measured counts (2 definitions + 28 decorations + the inline one at `test_embedded_measurement_recompile.py:31-40`), and the unknown-outcome instances are budgeted rather than discovered at the gate. |
| **B4** | **RESOLVED** | `_core/_image_parts/_image_io_handler.py` is now a File Structure row naming all five sites, and Task 2 Step 3 opens with *"The `.part` writer is in `_core`, not `sdk_`"*. Verified every line: `save2zarr` def `:1069` / param `:1076`, `_save_store` def `:1129` / param `:1139`, `_write_store_part` def `:1213` / param `:1224`, threading `:1126` and `:1204`, table write + descriptor + root fold `:1377-1388`. The forward producer is its own row (`_cli_output_manager.py:1871-1968`, call `:1936`, thread `:1948`) and is correctly called the larger hole. Fixture constructors: **five**, all verified — see §R2.4, my count was wrong. |
| **B5** | **RESOLVED, and better than I asked for** | The stamp is cut entirely (see H3 below) and `sdk_/_master_io.py` is repurposed as the single home of the v1/v2 discrimination. The reader table is measured off `BundleLayout.master_parquet` rather than the grep that found nothing, and correctly distinguishes frame reads from existence/mtime adds (`_processing_inventory.py:202,373`) and from writes (`_metadata_migration.py:1036,1065`). It also forces the unresolved question I raised — *"whether that conversion is P4's or P6's is itself unstated in both plans; state it in both"*. One count slip, see §R2.3 NEW-5b. |
| **B6** | **RESOLVED** | Ten-module dependency table; the semantic inversion is its own sub-block with the exact ranges (`:1493-1500` required, `:1508-1512` best-effort, `:1542` return) and the failure direction stated (*"a run that reports success having written no master at all"*); return-value propagation traced to `phenotypicCLI.py:2982` with the two name-binding tests (`test_staged_gpu_local.py:520,529`, `test_cli_v2.py:2069-2086` — both verified); `AGGREGATE_PROOF_VERSION` forced to an explicit decision with its consequence. Q6's ten files are enumerated and I re-verified the count is exactly ten. |
| **B7** | **RESOLVED in structure — but the replacement gate is green-by-construction as specified. See NEW-1.** | The self-contradiction is gone, the arm's fate is now an explicit (a)/(b) decision with a retirement condition on (b), Step 5 is two mutations because one test cannot reach both arms, and a dedicated legacy-arm test was added. All correct. The *fixture* for that test does not reach the arm — the one thing that would make the new gate real. |
| **B8** | **RESOLVED** | The false premise is quoted and struck with the shipped lines (`_cli_completion.py:1045-1046` writers; `:786-788` and `sdk_/_run_state.py:1213-1215` readers — verified). The work is four numbered items including `stable_keys` (`_cli_completion.py:1173-1180`, verified — I had not found it) and the both-writers-one-commit requirement with the exact failure (`"abc" != None` → binding `None` → complete runs read incomplete). The fallback arm is kept *and* given a retirement condition, and the ordering property that makes the migration windowless is stated and correct (`if "source_set_digest" in proof` is the first test, `sdk_/_run_state.py:1234`). |

### HIGH

| | Verdict | Evidence |
|---|---|---|
| **H1** | **RESOLVED** | Header now says `attributes.phenotypic.metadata_table.snapshot_sha256`, with a note naming the defect class and pointing at P1's shipped spelling (`sdk_/_run_state.py:377`, reasoning `:364-376`). `OPEN-QUESTIONS.md:95,99` amended in the same change, with its own ⚠ CORRECTED block covering both the key and the dangling "P4 Task 6". |
| **H2** | **RESOLVED, and refined past what I found** | Task 2 Step 3: *"Omit the block entirely when `join_status == "not_requested"`"*, with the advisory's filter cited (`sdk_/_run_state.py:1295-1300`) and `_store_metadata_snapshot`'s `None` return (`:664-667`) — both verified. Two tests, and the second one catches a distinction I missed: `no_common_keys` **did** have a snapshot and must still record it. Residual write-path gap: NEW-2. |
| **H3** | **RESOLVED by ruling; the plan implements the ruling, not a workaround** | *"Write both; never derive one from the other at read time"* is struck by name. The argument is the one the ruling gives — the premise changed, so this **satisfies** rule 3 rather than excepting it — and I confirmed no register row for a second home was added anywhere (the budget table's `metadata_table` row says "Relocation … One home before, one home after"). The cost-inversion measurement is a ⛔ gate with the right comparison stated ((a) N `pq.read_schema` vs (b) N `pq.read_schema` + N root reads) and the right instruction on a bad result (*"the answer is **not** to re-mirror the digest"*). Verified the free-ride claim: `_cli_output_manager.py:940-941` opens the schema and `:958` pulls the digest from the same object. |
| **H4** | **RESOLVED** | `include_dataset_column` is in the signature with a `# H4 -- NOT optional to omit` marker, a six-row site table, and a Task 4 Step 3 threading instruction naming the two serialization-boundary sites. |
| **H5** | **RESOLVED** | Five numbered plus **three** un-numbered, tabulated with sites and per-item consequences. It also correctly diagnoses my citation: `:1023-1050` lands on the `Args:` block, and the numbered list is `:990-1017`. |
| **H6** | **RESOLVED** | Two-guard table (`:933-936` authority, `:962-965` generations — verified), the observation that the retirement argument touches only the second, and *"Name the replacement or keep the check"* with three real candidates (`_cli_recompile_tables.py:303`, `_cli_recompile_recovery.py:748`, `datasets_needing_migration`'s advisory at `sdk_/_run_state.py:1287-1293` — all verified) and an explicit smaller option. |
| **H7** | **RESOLVED by the ruling** | The which-file-carries-which table is correct against the contract: `not_requested` / `[]` / `""` is exactly the shape `_valid_embedded_measurement_contract` already accepts (`:216-218`, verified), so the measurements table needs no contract change — which dissolves the conflict rather than papering over it. The honest note that `join_kind`/`join_left`/`join_right` become constants on that file is right: `:201-206` requires them unconditionally, and `:191-192` requires every key present, so the KV slot survives holding `""`. |

### MEDIUM / LOW

**M1–M6 all RESOLVED.** M1: order corrected against `:1368-1375` / `:1377-1388` / `:1390+`,
with the obligation narrowed to the one thing that is load-bearing (root last). M2: group
`zarr.json` in both the test and the implementation text, with the exact-equality reason
(`:168-181`) and a required schema-version constant. M3: budgeted twice. M4: gate is
`tests/unit tests/integration`, Task 5 has its own command, and the `testpaths` reason is
given. M5: replaced by a re-derivable criterion — **I ran it and got exactly the stated
15/5/2/8 split**. M6: `test_run_state_layering.py:60-68` is a File Structure row.

**S1 and S2 RESOLVED in `design.md`**: §7.4's step list is six with a ⚠ CORRECTED block, the
`process` row reads "skips 1–5", the re-backfill promise is narrowed, and §7.5's shard bullet
now says "emptied when fan-out begins" with the full rejected-alternatives record.

**The stale-citation table: re-verified independently.** Every correction in it holds at
`869e9dee`, and the repair adopted them. Four of my own were wrong or imprecise — §R2.4.

## R2.2 — The false-green sweep (new class)

`tests/_ngff_conformance.py` confirmed: **0 test functions**, exports `assert_store_conforms`
/ `assert_ome_xml_valid`, no `python_files` override in `pyproject.toml`, 10 test modules
import it, and `tests/unit/core/test_ngff_conformance.py` is the real suite. The repair's fix
is correct.

**But the class survives in three of the plan's own steps.** Applying the criterion — *does
the expected output distinguish "ran and passed" from "collected nothing"?*:

| Step | Command | Expected output | Distinguishes? |
|---|---|---|---|
| Task 1 Step 4 | **none given** | `PASS (6 passed)` | Count is right — I counted the six test functions in Step 1 — but with no command the count cannot be produced. |
| **Task 2 Step 4** | three paths | **none at all** | ❌ **The `_ngff_conformance` mechanism survives the fix that named it.** With three paths in one invocation, a path that collects zero is invisible: the others collect, the total is non-zero, exit code is 0. Needs a per-path count, or a `--collect-only` check of each path first. |
| **Task 3 Step 4** | **none given** | *"every test in this task passes except X"* | ❌ The reasoning for refusing a fabricated total is right, and nothing replaced it. With no command and no floor, an empty collection satisfies "every test passes" vacuously. Give the command and a floor: *the run must report at least N collected, where N is the number of `def test_` in the file.* |
| Task 4 Step 5 | given | none | ⚠ Full-suite gate; the `run-phenotypic-test` skill supplies the baseline, but say "compare to the recorded baseline" explicitly rather than leaving the comparison implicit. |
| Task 5 Step 2 | given, **single path** | `PASS` | ✅ Safe — a single-path invocation exits 5 on zero collection and prints "no tests ran", not "passed". `(1 passed)` would still be better. |
| Task 3 Step 5 | differential | red → green | ✅ **Safe by construction.** A mutation proof cannot be satisfied by an empty collection; this is the shape the other steps should borrow. |
| every "Run to verify failure" | none | none | ⚠ TDD red steps — same missing-command gap, milder, since a red step's evidence is the failure text. |

## R2.3 — New findings introduced by (or surviving) the repair

### NEW-1 — BLOCKING. The new legacy-arm test's fixture cannot reach the legacy arm.

**Plan location:** Task 3 Step 1, `test_finalize_run_ignores_a_stale_aggregate_on_the_legacy_arm`
— fixture `_publish_two_images_without_processing_state`, docstring *"This fixture makes
`authorized_measurement_sources` return None (no processing state, or
`success_markers_required` false) so the assertion reaches the branch."*

**Neither stated condition returns `None`.** `authorized_measurement_sources`
(`_cli_completion.py:918-933`) delegates *both* of them to `_sources_without_state`:

```python
if state is None or not state.config.get("success_markers_required", False):
    return _sources_without_state(output_dir)          # :928-933
```

and `_sources_without_state` (`:852-895`) globs **both** progress trees and returns `None`
**only when neither holds a single `*/*.json`**:

```python
payload_paths = [...]          # :889-893
if not payload_paths:
    return None                # :895
```

So a fixture that *publishes two images* — records, markers, either — gets a non-`None`
mapping back, `finalize_run` takes the authorized arm, and the poisoned
`_dataset_aggregated.parquet` is ignored **for the same reason the first test already
ignored it**. The replacement gate is green whether or not the legacy arm survives, which
is precisely the property B7 was filed against, reproduced inside the fix for B7.

**What the fixture actually has to be:** no per-image payload under `.phenotypic/progress/`
at all, with authority coming from legacy external Parquets at
`results/<ds>/measurements/*.parquet` — i.e. a **legacy tree**, which is also what makes
`discover_measurement_sources`' aggregate preference (`_measurement_sources.py:161-166`)
reachable. (A corrupt `processing_state.json` is a second route: `load_processing_state`
raising `KeyError`/`TypeError`/`ValueError` returns `None` directly at `:924-925`.) Say
which, and say it in the fixture name.

**This is one paragraph to fix and it is the only finding that would ship a false green.**

### NEW-2 — MEDIUM. H2's rule is a write rule; a store that *loses* its metadata keeps a stale block.

Task 2 Step 3 says to omit `metadata_table` when `join_status == "not_requested"`. It says
nothing about **removing** an existing block. A store built with `--metadata` and then
re-measured without it goes through `replace_embedded_measurement_table`'s root refresh; if
that refresh only adds and updates keys, the old `metadata_table.snapshot_sha256` survives
while the run's `metadata_sha256` is now `None`, `_store_metadata_snapshot` returns the
stale digest, and `sha not in (None, None)` fires the advisory on every such store — H2's
failure arriving through the measure path instead of the promote path. Add the removal rule
and a test (`_run_measure_mode(tmp_path, metadata=None)` on a store built with metadata,
asserting the block is gone).

### NEW-3 — MEDIUM. The budget table's `source_set_digest` row is wrong, and it is the row the HARD STOP is checked against.

Plan header, "State-artifact budget: this phase adds nothing":

> `source_set_digest` / `source_image_count` in the run proof | **Relocation** out of the
> aggregate proof, plus the **deletion** of `publication_id`. Net −1 field.

**They do not leave the aggregate proof, and cannot.** `current_aggregate_is_current` reads
both off the aggregate proof at `_cli_completion.py:786-788` (verified), and Task 3 Step 7
itself says *"publishes `source_set_digest` and `source_image_count` into **both** proofs"*.
So the change is a **duplication into the run proof**, not a relocation: +2 fields in the run
proof, −1 (`publication_id`) in each of two proofs → **net 0**, and `source_set_digest`
acquires a second on-disk home.

That is very likely fine — the aggregate↔run binding *is* a comparison of two independently
computed values, which is a different thing from mirroring one fact into two places, and it
is what U-4 bought by cutting the opaque hash. **But the table must say that**, because the
table is the artifact the HARD STOP is evaluated against, and a row asserting a −1 that does
not happen is exactly the kind of number this change has been punished for. Rewrite the row;
if the two-homes reading is contested, it is a rule-3 question for the user, not a table
entry.

### NEW-4 — MEDIUM. Three verification steps still cannot fail for the right reason.

See §R2.2. The two that matter are **Task 2 Step 4** (multi-path command, no expected output
at all — the surviving instance of the very mechanism that step was repaired for) and **Task
3 Step 4** (no command, and the deliberate refusal to quote a total left nothing in its
place). Task 3 Step 5's differential mutation is the model to copy.

### NEW-5 — LOW. Two census slips in the repair.

- **(a)** Task 4 Step 4 lists the `sdk_/__init__.py` deletions as *"`:91`, `:187`, `:472`,
  `:599`, `:602` — five lines: two imports, three `__all__` entries"*. Measured: **six
  lines, three imports** — `:91` `MASTER_MEASUREMENTS_CSV`, **`:184` `load_master_measurements`**,
  `:187` `master_measurements_csv_path`, then `:472`, `:599`, `:602`. The step's own
  prescribed `grep` would surface it, so this is a note.
- **(b)** Task 3 Step 6 says *"Nine `layout.master_parquet` references across **seven**
  modules"*. Nine references is right; they span **six** modules — `_output_root.py`,
  `_processing_inventory.py`, `_curation_labels.py`, `_error_tab/_publication.py`,
  `_qc_tab/review/_data.py`, `_metadata_migration.py`. The table itself has six rows, so the
  prose disagrees with the table beside it.

### Residual citation drift in the repair (all ≤2 lines, all landing inside the named construct)

`_cli_completion.py:1151-1153` (`publication_id` in the run proof) → `:1150-1152`;
`sdk_/_run_state.py:1236-1241` (fallback arm) → `:1235-1241`; `:1213-1216` (rule 1's
comparison) → `:1213-1215`; `sdk_/_measurement_tables.py:201-208` (the unconditional
join-kind check) → `:201-206`; `_cli_output_manager.py:1560` (the aggregate lock) →
`lock_path` is `:1559-1561`, `exclusive_path_lock` at `:1562`. None misleads.

## R2.4 — Corrections to my Round 1 report

Reported honestly, since the repair was told not to take my citations on trust:

1. **`aggregate_master_csv`'s `include_dataset_column` is at `_cli_output_manager.py:2032`, not `:2016`.** My H4 gave `:2016`; verified `:2032`. The repair is right.
2. **Five `PreparedEmbeddedMeasurementTable(...)` fixture constructors, not four.** I listed `tests/_output_layout.py:173`, `conftest.py:105`, `test_measurement_source.py:41`, `test_measurement_routes.py:54` and **missed `tests/unit/cli/test_embedded_measurement_aggregation.py:134`**. The repair found it.
3. **My B5 said "8 reader sites" and called them all reads.** Measured: nine `layout.master_parquet` references, of which two are **writes** (`_metadata_migration.py:1036,1065`) and two are existence/mtime adds, not frame reads (`_processing_inventory.py:202,373`). The genuine frame-read set is five references across four modules. The repair's characterization is right; its module count is off by one (NEW-5b).
4. **Two of my guard ranges were imprecise:** the required-CSV block is `:1493-1500` (I wrote `:1493-1499`) and the best-effort Parquet block is `:1508-1512` (I wrote `:1502-1511`, which starts at the inner `def`); `_consistent_embedded_join_keys`' authority guard is `:933-936` (I wrote `:934-937`). None changed a conclusion, but the repair's ranges are the ones to use.

## R2.5 — Verdict

## **REVISE** — narrowly. Two required edits, both a paragraph.

This is a genuine repair, not a rewrite of the prose around the findings. B1 is the test
case and it passes: the withdrawn claim is quoted and struck, the mechanism is cited from
the function's own docstring, the estimate is withdrawn, and — the thing that matters — **the
instruction changed**, from "delete the read-back" to "none required". Fourteen of fifteen
prior findings are resolved, several past what I asked for (H2's `no_common_keys`
distinction, B8's `stable_keys`, B5's read/write/mtime split). The rulings are implemented as
decisions rather than routed around: the digest has one home and the "write both" sentence is
struck by name, the stamp is cut in full, and both compatibility branches the phase touches
(`_source_set_binding`'s fallback, the migrator's schema gate) carry retirement conditions in
the form EXECUTION.md now requires.

**Required before dispatch:**

1. **NEW-1** — respecify `test_finalize_run_ignores_a_stale_aggregate_on_the_legacy_arm`'s
   fixture. As written it does not reach the arm, so B7's replacement gate is green by
   construction — the register's central mechanism, inside the fix for it.
2. **NEW-3** — correct the budget row. They are duplicated into the run proof, not relocated
   out of the aggregate proof; net 0, not −1.

**Strongly recommended in the same pass** (none blocking): NEW-2's block-removal rule, and
NEW-4's expected-output floors on Task 2 Step 4 and Task 3 Step 4. NEW-5's two census slips
are notes.

No re-review of anything else is needed. If the two required edits land as described, this
is dispatchable without another round.

---

# Round 3 — final

`phase-4-finalize-run.md` 1915 → **2144** lines. `phase-6-consumer-migration.md` also gained
76 lines this round (the lead's note said three files; the diff is `phase-4` + `phase-6`).
`design.md` and `OPEN-QUESTIONS.md` unchanged, as stated.

## R3.1 — The two blockers

**NEW-1 — RESOLVED, and in the strongest available form.** The test is renamed
`..._on_a_legacy_external_parquet_tree`, the fixture is `_build_legacy_external_parquet_tree`
with the inline comment `# no progress payloads`, and a ⛔ block states **both** preconditions
with their evidence. Verified every citation in it: `_sources_without_state` spans `:852-916`
(next def is `authorized_measurement_sources` at `:918`), its `payload_paths` guard is
`:889-895` with `return None` at `:895`, and its docstring says so at `:876-881`. The
route-choice is recorded with its rejection reason — the corrupt-state route
(`_cli_completion.py:928-931`) reaches the arm without producing the legacy tree the arm
exists to serve — and the chosen route is in the test's **name**, so a later reader cannot
mistake which.

The decisive addition is the last paragraph: *"**Assert both preconditions rather than
trusting them.** Before `finalize_run`, assert `authorized_measurement_sources(tmp_path) is
None` and `not _aggregate_needs_image_name_recovery(<aggregate path>)`."* That converts both
traps from silent-green into a named failure, which is the only repair that actually closes
this class rather than moving it.

**NEW-3 — RESOLVED, and the contrast is correct.** The row now reads *"Duplication into the
run proof, net 0"* with the arithmetic spelled out, and the ⚠ CORRECTED note carries the
ruling. I checked the part that was not specified:

> the metadata digest one row above, where the two copies would have been the *same* fact
> asserted twice with no comparison between them — and which is therefore a relocation to a
> single home.

**That contrast holds.** The struck instruction for the digest was *"Write both; never derive
one from the other at read time"* — two independent read paths, one value, **no comparison
anywhere**. Nothing in the codebase compares `phenotypic.metadata.snapshot_sha256` against
`metadata_table.snapshot_sha256`; the advisory reads one of them. That is a mirror. The
proof digests are read by two different checks against two independently derived live values
— `current_aggregate_is_current` compares the aggregate proof's copy to
`_current_success_work_ids` (`_cli_completion.py:786-788`), and `_run_proof_covers_current_inventory`
compares whichever proof `_source_set_binding` returns to `canonical_digest(verified)`
(`sdk_/_run_state.py:1213-1215`). Two checks, two moments, two derivations. Cross-check, not
authority. The classification is right and the distinction is drawn on the correct axis
(*is anything comparing them*), not on a surface property.

## R3.2 — Adjudicating the three citation corrections

**The repair is correct on all three. I was wrong on all three.** Measured at `869e9dee`:

| Claim | Mine | Repair's | Correct |
|---|---|---|---|
| `authorized_measurement_sources`' delegation to `_sources_without_state` | `:928-933` | `:932-935` | **Repair.** `if state is None or not state.config.get(` is `:932`; `return _sources_without_state(output_dir)` is `:935`. |
| the corrupt-state route to `None` | `:924-925` | `:928-931` | **Repair.** `:924-925` is the docstring's last line and its closing `"""`. The `try:` is `:928`, the call `:929`, the `except` `:930`, `return None` `:931`. |
| `publication_id` in the run proof payload | `:1150-1152` | `:1151-1153` | **Repair.** `:1150` is `"scientific_config_digest"`; the `"publication_id": (` key opens at `:1151` and closes at `:1153`. My Round 2 "correction" of the original `:1151-1153` was itself the error. |
| `_source_set_binding`'s fallback arm | `:1235-1241` | `:1236-1241` | **Repair.** `:1234` is the `if`, `:1235` is `return proof` — the **non**-fallback early return. The fallback begins at `:1236`. |

It also left `sdk_/_run_state.py:1213-1216` alone as equally defensible. That is the right
call: the `return (...)` expression opens at `:1213` and its two comparison lines are
`:1214-1215`, with `:1216` the closing paren. Either range names the construct.

**Running tally of my own citation errors across three rounds: seven.** The repair has been
the more reliable source on line numbers since Round 2, and its ranges should be preferred
wherever they disagree with this report.

## R3.3 — The third level, and the sweep

**The `_image_name_column` fix is correct and I verified it independently.**
`_aggregate_needs_image_name_recovery` (`_measurement_sources.py:51-58`) returns `True` when
`_image_name_column` (`:39-48`) finds no column whose `metadata_member_for_header(...) is
IMAGE.IMAGE_NAME`; `IMAGE.IMAGE_NAME = Entry("ImageName", …)` (`schema/_metadata.py:25`), so
the header is `Metadata_ImageName` and `Metadata_ImageFile` is not it. With individual
Parquets present the aggregate is skipped (`:161-167`). The plan's fix — poison carrying
`Metadata_ImageName`, `String`, non-null, non-empty after `strip_chars`, not matching
`_UUID_PATTERN` (`:24`) — satisfies all four conditions in `:59-79`. `"GHOST.tif"` does.
Keeping `Metadata_ImageFile` for the assertion is right.

### The sweep you asked for: the shape is present in about nine more assertions

**The plan already knows this discipline and names it.** `test_metadata_added_after_the_stores_still_joins_every_measured_row`
carries *"The `measured.height > 0` guard matters: without it the assertion is vacuously true
on an all-phantom frame"*, and `test_the_mirror_carries_both_joined_rows_and_phantoms` applies
it. **Two of roughly eleven candidates.** The rest assert a negative, or an equality between
two things that can both be empty, with nothing establishing that the fixture produced
anything. Ranked by stakes:

| Test | The vacuous pass | Why it matters |
|---|---|---|
| `test_every_mode_produces_a_byte_identical_master` | `_master_bytes(a) == _master_bytes(b)` — if the helper returns `b""` for a missing file, **two failed runs compare equal** | This is the phase's headline claim: three modes, one master. |
| `test_finalize_run_writes_no_byte_into_a_proven_store` | `before == after` where both are `{}` if `store.rglob("*")` finds no files | INV-PROVEN's only gate. |
| `test_a_v1_metadata_free_master_is_indistinguishable_from_v2_and_that_is_harmless` | `_reader_outcomes(v1) == _reader_outcomes(v2)` with both empty — an outcome collector that swallows exceptions returns `{}` for both | **Highest stakes in the plan**: this test is the designated falsifier for a user ruling, and a false green here silently confirms "no stamp needed". |
| `test_finalize_run_invalidates_the_intermediates_on_success` | `assert not chunk.exists()` — trivially true if `_plant_stale_chunk_parquet` never created it | |
| `test_stores_with_mixed_metadata_snapshots_do_not_abort_finalization` | "must not raise" — true if the fixture produced one snapshot, not two | This is CAN-2's test; the state it needs is the one D-A manufactures. |
| `test_process_mode_skips_finalization_entirely` | `assert not (…master_measurements.parquet).exists()` — true if `process` errored early and wrote nothing | |
| `test_finalize_run_ignores_every_stale_intermediate` | both `not in` and `master.equals(concat)` hold when both frames are empty | |
| `test_a_measured_row_absent_from_metadata_is_dropped_deliberately` | `orphan.height == 0` on an empty mirror | |
| `test_the_mirror_keeps_canonical_column_order_after_the_join` | `cols == order_measurement_columns(cols)` is a fixpoint check, true for `[]` | Mildest — its sibling pins `cols[:2]`. |

**One additional case of the same shape inside an assertion rather than a fixture:**
`test_measure_mode_refreshes_the_table_and_the_root_together` asserts `root_after !=
root_before`. A re-promote rewrites the journal's `applied_at_utc` / `duration_seconds`
(omitted only under `reproducible_provenance`), so **the root differs whether or not the
table or the snapshot was refreshed** — the first assertion passes for the wrong reason and
its failure message overclaims. The test is saved by its second assertion
(`_snapshot_sha256(store) == _sha256_of(edited)`), which is the load-bearing one. Say so, or
drop the first.

**The fix is one standing rule plus five named guards**, in the form the plan already uses:
*every assertion of a negative or of an equality must be preceded by an assertion that the
fixture produced the thing whose absence or equality is being claimed.*

## R3.4 — Command audit: RESOLVED, and stronger than I asked for

Spot-checked against your criterion. None of the expected outputs is satisfiable by a
zero-collection run, and none is a bare `>= 1` floor:

- **Task 1 Step 4** — `6 passed`, single path, exact. I counted six `def test_` in Step 1: correct.
- **Task 2 Step 4** — split into two invocations *"so their count is unambiguous"*, with **"neither location may collect zero"** and *"a non-zero collected count for each of its three paths"*. This is the precise fix for the multi-path invisibility I raised; the `tests/_ngff_conformance.py` mechanism can no longer hide.
- **Task 3 Step 4** — `15 collected, 14 passed, 1 failed`, the failure named, *"any collected count below 15 is this step failing"*. I re-derived 15 from the plan: 6 (Step 1) + 4 (Step 3) + 5 (Step 3b). Correct, and Step 6b's sixteenth is accounted for.
- **Task 4 Step 5** — *"compare the collected/failed counts against the recorded baseline … not 'PASS'"*, with **"a *drop* in the collected count is the signal"**. Right treatment for a suite with pre-existing failures.
- **Task 5 Step 2** — `1 passed`, single named file. A wrong path there errors (exit 4), it cannot collect an unrelated test.
- **The four red steps** now read *"every test in the file red, and the failure text must name the symbol under test as missing. A collection ERROR from a different cause … is also red, and is not evidence this step passed. Read the reason, not the colour."* That is the level-below distinction applied to red steps, which I had not asked for.

## R3.5 — New findings

### NEW-6 — Cutting `publication_id` turns two live checks into tautologies. Step 7 names neither.

Step 7 names four sites: the aggregate writer (`:1032`), the run writer (`:1151-1153`),
`stable_keys` (`:1173-1180`), and `_source_set_binding`'s fallback. **`grep -rn
publication_id src/` returns three more consumers**, and two of them *compare* on it:

| Site | After the cut | Live? |
|---|---|---|
| `_cli_completion.py:1220`, `:1228` — `valid_run_completion` builds `expected["publication_id"] = aggregate.get("publication_id")`, then `if any(marker.get(key) != value …)` (`:1231`) | `None != None` → False → **the entry contributes nothing**. The aggregate↔run binding stops being checked. | **Yes — five callers**: `gui/results_viewer/_output_consistency.py:381`, `gui/run_console/_slurm_observer.py:1319`, `_cli_migrate.py:1268`, `phenotypicCLI.py:2505`, `:2513`. |
| `sdk_/_run_state.py:1081-1085` — `run_proof_is_current` returns `proof.get("publication_id") == aggregate.get("publication_id") and …` | `None == None` → **True unconditionally**; only the `finalization_input_digest` half survives. | Exported (`:118`) but **no in-repo caller** — public API surface only. |
| `gui/results_viewer/_output_consistency.py:60`, `:442-443` — `aggregate_publication_id` field | permanently `None` | Non-branching (U-4 verified "zero for the GUI copy"), but P6 does not mention it and P7 mentions `publication_id` only in passing (`phase-7:1459`). |

**Neither comparison errors; both silently stop checking.** That is the same degeneration
Step 7 item 3 correctly identifies for `stable_keys` — *"With `publication_id` gone the entry
compares `None == None` and contributes nothing"* — so the plan found one instance of this
exact pattern and prescribed a replacement, then missed two more of it. An implementer
following Step 7's four numbered items edits exactly those four and leaves the rest as
tautologies.

**Fix:** name all three in Step 7, with a disposition each — replace with
`source_set_digest`/`source_image_count`, or delete the clause with a stated reason. Two
sentences.

### NEW-7 — Step 7 item 1 says *compute*; U-4's purpose says *copy*. They mean different things.

> 1. **Add both fields to `publish_run_completion_evidence`** … They must be computed the
>    same way the aggregate proof computes them (`_current_success_work_ids` →
>    `canonical_digest(sorted(...))`), or the two proofs disagree on identical trees.

But P1's shipped docstring, which Step 7 quotes two paragraphs earlier, says the point is
that *"the aggregate-to-run **binding** is stated directly instead of through an opaque
hash"* (`sdk_/_run_state.py:1222-1226`). A binding means the run proof carries **the
aggregate's** value — `publication_id` restated in the clear, i.e. a **copy**, which is
exactly what `:1152` does today.

The two readings are not equivalent:

- **Copy** — the run proof asserts *"I was published against that aggregate"*. Rule 1 then
  checks that assertion against live verification, so a success set that changed between the
  two publications is caught. This preserves today's guarantee.
- **Recompute** — the run proof asserts its own view at its own moment. Nothing compares the
  two proofs to each other any more, so a stale aggregate proof beside a fresh run proof
  passes both checks independently. The binding U-4 said it was stating directly is gone.

Item 1's trailing clause (*"or the two proofs disagree on identical trees"*) shows the author
expected agreement — which recomputation gives on an **unchanged** tree and loses on a
changed one, which is the only case that matters. **Choose, and say which and why.** One
sentence. (The evidence points to copy.)

### NEW-8 — the vacuity sweep of §R3.3, as a required edit

One standing rule plus guards on the five high-stakes tests named there.

## R3.6 — Verdict

## **REVISE** — three edits, all bounded, none structural.

Nothing found this round undoes the repair. NEW-1 came back better than I specified — it
asserts its preconditions instead of documenting them, which is the only form that closes the
class — and its third-level trap was found and fixed below where I stopped looking. NEW-3's
contrast is drawn on the right axis. The command audit is resolved past what I asked for, and
the red steps now carry a distinction I had not thought to require. On line numbers the
repair beat me three for three, seven for seven across the change; prefer its ranges over
this report's wherever they differ.

**Required before dispatch:**

1. **NEW-6** — name the three remaining `publication_id` consumers in Step 7 with a
   disposition each. Two of them are comparisons that degrade to `None == None`, one with
   five live callers, and Step 7 already diagnoses this exact degeneration one site over.
2. **NEW-7** — decide compute-vs-copy for the run proof's `source_set_digest` and say why.
   The evidence points to copy; item 1 currently says compute, and the choice changes what
   the proof asserts.
3. **NEW-8** — add the vacuity-guard rule and apply it to the five named tests. You asked
   whether "reaching the branch ≠ the branch doing the thing" fails elsewhere. It does, in
   about nine assertions, including the phase's headline three-modes-one-master test,
   INV-PROVEN's only gate, and the designated falsifier for a HARD-STOP ruling.

I would want to see (3) before implementation starts, which is why this is REVISE rather than
APPROVE-with-notes: it is the same defect class this review has spent three rounds on, and
the plan already contains the fix pattern — it is applied twice and needs to be applied five
more times.
