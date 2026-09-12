# P7 Task 4 review: projecting legacy embedded tables at read

Reviewer: implementation + test review. Analysis only; no source or test file was edited.
Tree: `.worktrees/migrate-maresca`, branch `fix/migrate-project-legacy-tables`, base `2266fc5f`.
The change under review is the uncommitted `git diff HEAD` (7 files, +1075/−77), read in full from
the lead's scratchpad copy `task4_full.diff`.

**Scope read.** `_cli_parquet_agg.py` and `_cli_finalize_run.py` in full, plus
`_cli_finalize_fanout.py:330-1109`, `_cli_recompile_worker.py:305-392, 770-834`,
`_cli_output_manager.py:150-507, 1040-1250, 1426-1543, 1994-2030`,
`_cli_completion.py:820-850, 992-1208`, `sdk_/_measurement_tables.py:20-130, 440-595, 813-899`,
`_embedded_measurement_tables.py` (whole), `_cli_migrate_image.py:273-339, 824-849`,
`_cli_migrate.py:1037-1075, 1351-1420`, `_cli_checkpoint_handler.py:277-299`,
`_cli_sentinel.py:155-180`, `phenotypicCLI.py:3070-3092, 3890-3915`,
`sdk_/_run_state.py:1280-1345`, `sdk_/_master_io.py`. Tests: every new or changed test in the three
test files, `tests/unit/cli/conftest.py:595-821`, `test_finalize_fanout.py:435-470`, and the two
existing tests that set `SCRATCH`.

**Evidence received from the lead (verbatim outputs summarized in the appendix).** Probes 1–5, and
the real-data check, job 28283848: all 6,529 stores projected, 0 excluded, and 113,814 measured
rows in the mirror, up from 0.

---

## Summary

The core fix is correct, and the real tree confirms it. On the migrated 6,529-store copy the
projected master carries no user metadata, no store is excluded, the mirror keeps 113,814
measured rows with 0 `*_right` columns, and no measured row has null `Metadata_Strain`,
`Metadata_pH` or `Metadata_Salinity`.

- **The collapse rule is sound.** Polars `unique()` treats NaN, null and ±0.0 as equal (probe 1).
  The `joined` gate cannot touch a post-inversion store, whose table is `not_requested` by
  construction.
- **The proof uses the post-exclusion set** on every path except the recompile KNOWN GAP.

**Three must-fix findings, all about fan-out (K > 1) versus the direct path.** The new tests
exercise almost only the direct path, while the forward CLI's default `--njobs -1` takes fan-out:

1. **MF-1:** the dtype advisory the R4 ruling relies on is silently skipped for drift that crosses
   a shard boundary.
2. **MF-2 (probe 4):** when every store is excluded, `njobs=1` publishes nothing and leaves the
   prior master, while `njobs=2` overwrites it with an empty master and publishes a proof
   certifying 0 images.
3. **MF-3 (probe 2):** the new docstring asserts the master is byte-identical at every shard
   count. That is false. The underlying widening predates this change, but the claim is new.

---

## Must fix before commit (3)

### MF-1. The dtype advisory is skipped for drift that crosses a shard boundary, on the CLI's default path

- **Where:** `_cli_parquet_agg.py:393`, the only call of `_warn_on_dtype_disagreement`, inside
  `aggregate_embedded_measurement_tables`. `_cli_finalize_run.py:180-186`, the shard merge
  `pl.concat(frames, how="diagonal_relaxed")`, has no advisory.
- **Scenario:** store `a` carries `Metadata_Well: Int64` and store `b` carries `String`, the
  `_restore_join_key_dtypes` failure branch. Under `njobs=2` they land in different shards. Each
  shard is internally uniform, so neither worker warns. The finalizer's merge widens the column to
  `String`, and no log line anywhere names the column.
- **Why it is on the default path (verified by reading):** the forward CLI calls
  `aggregate_master_csv(..., njobs=config.n_jobs)` (`phenotypicCLI.py:3078-3085`), which forwards
  `njobs` (`_cli_output_manager.py:1994, 2028`). `--njobs` defaults to `-1`, and
  `_aggregate_measurements_unlocked` fans out whenever `njobs != 1` (`:1487-1494`).
- **Not affected:** these callers pass no `njobs` and finalize directly: `--mode migrate`
  (`_cli_migrate.py:1051`), the sentinel (`_cli_sentinel.py:163`), local recompile
  (`phenotypicCLI.py:3899`), and SLURM index K for N < ~34,600 (K = 1). So the migration itself
  gets the advisory. A later forward re-run or `--mode measure` on the same tree does not.
- **Why must-fix:** CAN-10(b) was ruled R4, *"advise, don't repair"*. The plan's line was
  *"Never upcast silently"* (`phase-7-migrate-mode.md:1426-1427`), and the advisory is the whole of
  what R4 kept. The drift test calls `finalize_run` directly (`test_finalize_run.py:1708`), so it
  cannot see this.
- **Suggested fix direction:** also run `_warn_on_dtype_disagreement` over the shard frames in
  the `shard_paths` branch of `build_master_frame`.
- **Suggested test:** parametrize
  `test_a_string_drifted_join_key_widens_the_master_and_keeps_the_mirrors_metadata` over
  `aggregate_measurements(njobs=1)` and `njobs=2`. With 2 stores and K = 2 they land in different
  shards, and the `_warnings_naming(caplog, "Metadata_Well")` assertion fails today on the
  `njobs=2` arm.

### MF-2. Every store excluded: `njobs=1` and `njobs=2` publish different things, and fan-out certifies an empty master (verified by probe 4)

- **Verified outcome** (probe 4: both descriptors stripped, a 2×2 master planted):

  | | `njobs=1` | `njobs=2` |
  |---|---|---|
  | `aggregate_measurements` returns | `None` | the master path |
  | master on disk | the planted `(2, 2)`, untouched | **`(0, 0)`**, overwritten |
  | mirror | not written | written, `(0, 0)` |
  | aggregate proof | none | **valid, `source_image_count: 0`** |

- **Mechanism, read and then confirmed:**
  1. Each shard's projection excludes everything. `write_measurement_shard` writes
     `pl.DataFrame()` and returns `[]` (`_cli_finalize_fanout.py:477-480`).
  2. The merge concatenates K zero-column frames into a 0×0 frame, not `None`
     (`_cli_finalize_run.py:180-186`), and `finalize_run` checks only for `None` (`:448`).
  3. `finalize_post_master_outputs` has no empty-frame guard and seeds an empty mirror
     (`_cli_output_manager.py:1150-1182`).
  4. `planned_work_ids == []` is not `None`, so the proof is published against an empty set
     (`_cli_finalize_run.py:505-514`). `publish_aggregate_snapshot` refuses only when the **live**
     success count is 0 (`_cli_completion.py:1108-1110`), and here it is 2.
- **Which outcome is intended?** The **direct path's**, by the code's own contracts:
  - `finalize_run` documents `None` as *"when no measurement source could be read"* (`:425-427`).
  - `resolve_finalizer_shard_inputs` says *"a proof that should never have been published is worse
    than one a reader rejects"* (`_cli_finalize_fanout.py:714-715`).
  - The migrate caller already treats `None` with embedded tables present as a loud failure
    (`_cli_migrate.py:1062-1063`, `"aggregate rebuild produced no measurements"`).

  The fan-out outcome is a success-shaped publication that destroys the previous master and
  certifies nothing.
- **Pre-existing or new?** **Made reachable by this change.** Before it:
  - Locally, all-empty shards required an empty authorized set, and `run_local_aggregation_fanout`
    returns `None` for that, falling back to direct (`_cli_finalize_fanout.py:961-963`).
  - On SLURM, a zero-success run could produce all-empty shards and a 0×0 master, but the proof then
    raised, because the live count was 0.
  - A shard whose sources were all unreadable used to raise; the diff changes it to write an empty
    shard (`task4_full.diff:71-80`).

  The combination here (live successes > 0, merged set empty, proof published) did not exist.
- **Reachable how:** every authorized store lacks a descriptor (Step 0b: *"a store written before
  embedded tables has none"*) or disagrees under one label, and the tree is finalized by a forward
  CLI run.
- **Why must-fix:** the change creates the state, the consequence is destructive (prior master
  overwritten) and misleading (proof published), and `test_local_fanout_produces_a_byte_identical_master`
  exists precisely to pin K = 1 and K > 1 to the same result.
- **Suggested fix direction:** in `finalize_run`, treat a fan-out whose `planned_work_ids` is
  empty the way the direct path treats `master_df is None`. Or have the `shard_paths` branch of
  `build_master_frame` return `None` when every shard is empty.
- **Suggested test:** probe 4 as a pytest. Strip both descriptors, plant a master, run
  `aggregate_measurements` at `njobs=1` and `njobs=2`, then assert both arms leave identical master
  bytes, the same mirror existence, and the same `valid_aggregate_snapshot`. Today the `njobs=2` arm
  fails.

### MF-3. `_warn_on_dtype_disagreement`'s docstring claims K-invariance that is false (verified by probe 2)

- **Where:** `_cli_parquet_agg.py:328-330`: *"Logging changes no data, so the master is
  byte-identical whatever the shard count -- which a per-concat repair could not guarantee."*
- **Counterexample (probe 2, polars 1.41.2):** `Int64 7`, `Float64 7.5`, `String "x"`.

  ```
  K=1: Schema({'k': String}) ['7', '7.5', 'x']
  K=2: Schema({'k': String}) ['7.0', '7.5', 'x']
  equal: False
  ```

  Pairwise widening is not associative in its rendering: `Int64 → Float64 → String` renders
  `"7.0"`, while `Int64 → String` renders `"7"`. Concatenating all stores at once (direct) and
  concatenating within shards then across them (fan-out) differ whenever a column's three dtypes
  split across a shard boundary.
- **Pre-existing or new?** **The behaviour predates this change; the claim is new.** Before the
  change the direct path was one `aggregate_parquet_files` concat and the fan-out was per-shard
  `aggregate_parquet_files` followed by the same unchanged merge concat
  (`task4_full.diff:60-64, 156-161`). So the K-dependence existed for fresh and legacy stores
  alike. The projection actually **reduces** its reach, because it removes the user columns where
  the field's `7` → `7.0` widening happened.
- **Why must-fix (docstring only):** the sentence is added by this diff. It is the stated
  justification for choosing advise over repair (R4), and it asserts the K-invariance that
  `_cli/CLAUDE.md`'s `--wait` note depends on (*"they write the same bytes"*). `_cli/CLAUDE.md`
  itself warns that overstating an invariant *"costs the next reader the ability to verify it"*.
  The fix is to correct the sentence. The behavioural fix is FU-8, not this commit.
- **Consequence to state in the corrected docstring:** under a three-way dtype split, the master
  bytes (and so the `--wait` double-write) depend on K. The aggregate proof is unaffected, because
  it certifies a source set, not bytes.

---

## Rulings on the implementer's self-reported gaps

| Gap | Ruling |
|---|---|
| Recompile shard with every store excluded raises `"No valid measurements found for shard"` | **Follow-up** (FU-4). It fails loudly rather than publishing, which is the safe direction and the opposite of MF-2. Worth aligning after MF-2 settles which outcome is intended. |
| `_warn_on_dtype_disagreement` fires on any disagreement, including Int64 vs Float64 in forward trees | **Follow-up, and keep it.** Probe 2 shows Int64/Float64 is exactly the pair that makes the bytes K-dependent once a String joins it, so that warning is not noise. At most, lower Null-vs-typed (all-null column) disagreements to DEBUG. |
| A joined store whose own baseline has exact duplicate rows under one label is collapsed | **Follow-up, low.** Labels come from the objmap and are unique per object, so the forward measurer cannot emit two identical rows under one label. Record it as a limit. |
| Descriptor column missing from the table raises `ColumnNotFoundError` | **Agree, not a defect.** `_valid_embedded_measurement_contract` already rejects a table missing a descriptor column (`sdk_/_measurement_tables.py:551-552`), so an authorized store should not reach it. Loud is right. |
| Per-store cost, and S-2 not re-measured | **Follow-up** (FU-3), with the real-data number below. |

## On the 49 extra mirror rows

Verified by arithmetic from the job-28283848 log; no further probe.

- Measured rows: `128,598 − 14,784 dropped = 113,814`, which matches `mirror measured rows`.
- Matched metadata rows: `230,280 − 116,515 phantom = 113,765`.
- So 113,814 measured rows joined to 113,765 distinct metadata rows: **49 measured rows share a
  metadata key with another measured row.**

Two alternatives are ruled out by the log itself:

- **Metadata-side duplicate keys** would have logged *"Metadata CSV has duplicate keys"*
  (`_cli_output_manager.py:377-385`). It is absent.
- **The ragged path** would have suppressed the `dropped` count (`:393`), and the count was
  printed, so the join took the single non-ragged path.

Leftover fan-out copies are also ruled out: excluded = 0 and the projection leaves no duplicate
label within any store. The 49 are therefore **distinct objects in the same image sharing one
`(Grid_RowNum, Grid_ColNum)` cell**, the measurement-side multiplicity `join_metadata` documents as
normal. The lead's reading is correct. To pin it on data:
`mirror.filter(~QC_MetadataOnly).group_by(keys).len().filter(len > 1)` should sum to 49 extra rows.

---

## Follow-up (not blocking)

### FU-1. An excluded store appears only in a log line, and the run then reads `incomplete` with no stated reason, on every re-finalization

- **Where:** exclusions at `_cli_parquet_agg.py:245-253, 261-267, 289-296, 300-309`. Consequence:
  `sdk_/_run_state.py:1341-1344` compares the proof's source set against the **verified** set.
- **Scenario:** one store is excluded. The proof certifies N−1 (correct), every image is verified,
  and `resolve_run_state` reads `incomplete`. `grep -in "advis" sdk_/_run_state.py | grep -i
  "aggregate|source|master|proof"` returns nothing, so no advisory names the cause. Re-running
  excludes again.
- The plan's Step 0b says *"raise an advisory, per INV-VERDICT"*
  (`phase-7-migrate-mode.md:1441-1442`). **Needs a ruling** on whether a `logger.warning` meets it.
- **Suggested test:** extend
  `test_a_store_without_a_measurement_descriptor_is_skipped_with_an_advisory` to assert what
  `resolve_run_state(tmp_path, depth="deep")` reports, so the contract is pinned either way.

### FU-2. `$SCRATCH` staging on the embedded arm is pure overhead, untested, and leaks on the new raising paths

- **Where:** `_cli_finalize_run.py:190` (stages every table), `:201-212` (`read_paths`), and
  `:243-244` (cleanup, not in a `finally`).
- **Overhead:** each store's descriptor is still read from GPFS, and each table is read once to
  copy it and again to project it.
- **Untested:** no test sets `SCRATCH` and reaches the embedded arm. `test_cli_v2.py:2032` uses
  legacy external Parquets, and `test_embedded_measurement_aggregation.py:189` tests
  `_stage_to_scratch` alone. A mis-keyed `read_paths` silently reads the store instead, so the
  output stays correct and the bug is invisible. A swapped mapping would attribute one store's rows
  to another store's `filename`, and nothing would catch it.
- **Leak:** `project_embedded_measurement_table` now raises by design (`ValueError`,
  `ColumnNotFoundError`, `FileNotFoundError` from `require_readable_store`), which skips
  `_cleanup_scratch`.
- **Suggested tests:** first, set `SCRATCH`, wrap the projection to record
  `(table_path, read_path)`, and assert each `read_path` is byte-identical to its own `table_path`.
  Second, make one projection raise and assert the staging directory is removed.

### FU-3. Performance: serial per-store reads, and S-2 was measured on the old path

- **Per store:** a root `zarr.json` read plus a schema gate (already parsed once per store during
  authorization), a Parquet read, and for each fanned-out joined store a **second** footer open via
  `pq.read_schema` (`_cli_parquet_agg.py:284`). Legacy stores all take that branch. This is O(N)
  with a larger constant, serial on the direct path and on SLURM with K = 1. Nothing is quadratic.
- **Real data:** 304 s for 6,529 stores including the global join, on 4 CPUs, which is about
  0.047 s/image. `SECONDS_PER_IMAGE_S2 = 0.026` (`_cli_finalize_fanout.py:46-60`) sizes the SLURM K
  against `TARGET_TASK_SECONDS = 900` and was measured on the multithreaded multi-file read. If the
  shard worker's own cost is near 0.047 s, one task tops out near ~19k images rather than ~34.6k.
  Still fine for every N run so far. Re-measure S-2 on the projected path.
- **Cheap wins:** read the key/value metadata from the same `pq.ParquetFile` as the data; use a
  thread pool or `pl.collect_all` over per-store `scan_parquet(...).select(columns)`. The module
  docstring `_cli_parquet_agg.py:1-11` now justifies a read path that only the legacy arm uses.

### FU-4. Recompile: a shard with every store excluded fails the whole recompile, and no test drives a joined table through that shard

- **Where:** `_cli_recompile_worker.py:371-372`, against the empty-shard behaviour at
  `_cli_finalize_fanout.py:477-480`. Align them once MF-2 decides which outcome is intended.
- **Test gap:** the reworked `test_measurement_worker_derives_embedded_image_names_from_store`
  patches out `recompile_embedded_measurement_table` (`test_cli_recompile_slurm.py:1289-1291`), so
  its stores are `not_requested` and the projection is the identity. Yet the comment at
  `_cli_recompile_worker.py:363-364` says these tables are always joined.

### FU-5. Statements this change makes false (not edited; for the doc pass)

1. `CLAUDE.md:467-468`, `src/phenotypic/_cli/CLAUDE.md:615-616`, and
   `_cli_output_manager.py:1103-1105` all say *"exact pre-post concatenation of (marker-)authorized
   embedded tables"*. It is now the concatenation of tables projected onto their descriptors, with
   join fan-out collapsed, minus the excluded stores.
2. `docs/source/contrib_guide/tracked_state.md:181`, *"the master | the record-authorized embedded
   tables, and nothing else"*, omits the projection and the exclusion. Exclusion is what makes a
   fully verified run read `incomplete` (FU-1). The page asks to be updated in the same change
   (`_cli/CLAUDE.md:397-403`).
3. `_cli_finalize_run.py:144-147` claims the returned set excludes projection-excluded stores. That
   is false on the `shard_paths` branch, which returns the full selection (`:182`, `:186`), and it
   is why the recompile KNOWN GAP is reachable. Say so there.
4. `_cli_parquet_agg.py:1-11`: see FU-3.
5. `_cli_parquet_agg.py:328-330`: see MF-3.
6. The plan's Task 4 commit message (`phase-7-migrate-mode.md:1563-1566`), *"Migrate does not re-run
   finalization, so it leaves a v1-shaped master"*, is shown false by the new migrate pin. Do not
   use it verbatim.
7. `_cli/CLAUDE.md`'s `--wait` note, *"they write the same bytes"*, is false under a three-way dtype
   split (MF-3). That predates this change, but it should be qualified alongside MF-3.

### FU-6. Test-strength notes

- **`test_every_mode_produces_a_byte_identical_master`** (`test_finalize_run.py:292-317`) cannot
  see a regression common to every mode: all arms run the new projection on `not_requested`
  tables. The real old-versus-new guard for post-inversion trees is
  `test_finalize_run_ignores_every_stale_intermediate` (`:346`, `:363`), which compares against
  `_concat_of_embedded_tables`, the unchanged `aggregate_parquet_files` path, using frame
  `.equals`. Adequate. Name it as the guard in its docstring.
- **Field-regression test** (`:1500-1568`): it reproduces the real mechanism (an int user column,
  an image absent from metadata, a ragged join, `_right` columns). Its preconditions assert
  populated `a` and all-null `b`, but not the third ingredient, the dtype disagreement (`7` →
  `7.0`) the docstring describes. Add that precondition so the test cannot drift into "image absent
  from metadata" alone.
- **Migrate pin** (`test_embedded_measurement_migration.py:603-708`): the master's `st_ino` signal
  is sound on node-local scratch. The temp file is created while the old file still exists, so the
  rename cannot reuse the inode, and mtime resolution is irrelevant. The store inventory, though, is
  content SHA-256 only, which cannot see a byte-identical rewrite. D-A's claim is "not one byte
  written", and a re-promote renames the store, giving every file a new inode. Add `st_ino` per
  file.
- **`_write_pre_projection_v1_master`** (`:1242-1262`) is a genuine pre-inversion master. It is
  `aggregate_parquet_files(keep_filename=True)` plus `add_metadata_image_name_from_filename`,
  exactly the old `build_master_frame` minus staging (`task4_full.diff:156-161`), written with the
  same `PARQUET_WRITE_OPTIONS`. No finding.
- **Fan-out coverage:** every new behaviour test except `test_the_fanout_shards_project_legacy_tables_to_the_same_master`
  runs the direct path, which is why MF-1 and MF-2 went unseen. Cover the exclusion and drift tests
  at `njobs=2` as well.
- **Missing case:** no test has a joined fan-out store with null measurement values. Probe 1 shows
  the collapse handles it; a test would pin that for a future polars.

### FU-7. Guard gap: nothing asserts that the baseline migrate embeds carries no user metadata

- `_cli_migrate_image.py:289-290, 324-325, 827-828` and `_cli_migrate.py:1391-1393` embed
  `pd.read_parquet(source)` raw, and `measurement_columns = baseline.columns`
  (`_embedded_measurement_tables.py:70-73`). A per-image-joined legacy source would record user
  columns as measurements, and the projection would keep them.
- **Not present in real data** (probe 5): on both legacy trees the external Parquets carry no user
  columns and `join.status` is `None`. The migrated tables are `joined` with 12 user columns, and
  `user cols INSIDE measurement_columns: ()`. The lead's measurement on the 6,529-store copy agrees:
  a 136-column source, 136 `measurement_columns`, and a 148-column table.
- **Suggested test (optional):** a legacy Parquet already carrying `Metadata_Strain` goes through
  the migrate embed and a finalize, then assert `master_carries_user_metadata(master) is False`.
  That fails today, which would force a strip-or-refuse decision before such a tree appears.

### FU-8. The underlying K-dependent widening (pre-existing)

This is the behaviour behind MF-3. Options: cast every column to the supertype computed across
**all** sources before concatenating, in both paths, with the same advisory. Or have the merge
normalize shard dtypes against the union of their schemas before `diagonal_relaxed`. Neither
belongs in this commit.

### FU-9. Real data: 1,210 measured images dropped from the mirror; confirm they are absent by name, not by key rendering

`master distinct images: 6529`, but `measured distinct images: 5319` (14,784 rows dropped). Dropping
an object outside the described experiment is by design. But a key rendering mismatch
(`"1.0"` vs `"1"` on `Grid_RowNum`, exactly the probe-2 effect) drops whole images the same way,
and the log cannot tell the two apart. Check: take the 1,210 dropped `Metadata_ImageName`s,
anti-join them against `metadata.csv`'s `Metadata_ImageName` column alone, and expect all 1,210 to
be absent. Any image that *is* present by name but unmatched on the full key is a rendering defect.

---

## Checked and sound

- **Collapse cannot merge two objects, and it does handle NaN.** Probe 1: `unique()` keeps 3 of 6
  rows, collapsing `nan/nan`, `-0.0/0.0` and `None/None`, and `label dup after unique: False`.
  `pa.Table.from_pandas` maps NaN to null (`null_count: 1`), so pandas-written legacy tables hold
  nulls anyway. Rows collapse only under the same `Object_Label` with every projected column equal;
  objmap labels are unique per object; join keys are always inside `measurement_columns`, so
  fan-out copies are equal there.
- **The `joined` gate cannot fire on a post-inversion store.** P4 writes measurement tables through
  `measurements_payload()`, whose status is unconditionally `not_requested`
  (`sdk_/_measurement_tables.py:92-108`).
- **The projection keeps descriptor order.** Probe 1: `read_parquet(columns=['c','a']) -> ['c','a']`.
- **The proof uses the post-exclusion set on every path except the documented gap.** Direct:
  `finalize_run:505-509`. Local fan-out: `_cli_finalize_fanout.py:488, 1036-1043, 760-776`. SLURM
  forward index K: `_cli_checkpoint_handler.py:286-298`. Recompile SLURM
  (`_cli_recompile_worker.py:822-834`) is the KNOWN GAP. `resolve_finalizer_shard_inputs` requires
  only `planned ⊆ live`, as intended.
- **`filename` carries the store path** even when bytes come from a staged copy
  (`_cli_parquet_agg.py:381-383`).
- **The new fan-out test's arms really differ.** `njobs=1` bypasses fan-out
  (`_cli_output_manager.py:1487`), and the test asserts 2 shard files.
- **Unreachable guard:** `shard_df is None and merged` (`_cli_finalize_fanout.py:472-476`) cannot
  fire on the embedded arm. Harmless.

---

## Appendix: evidence

- **Probe 1** (polars 1.41.2, pyarrow 23.0.1): `unique rows: 3 [{'Object_Label': 1, 'x': nan},
  {'Object_Label': 2, 'x': -0.0}, {'Object_Label': 3, 'x': None}]`; `label dup after unique: False`;
  `from_pandas null_count: 1`; `read_parquet columns=[c,a] -> ['c', 'a']`.
- **Probe 2:** `K=1: ['7', '7.5', 'x']`; `K=2: ['7.0', '7.5', 'x']`; `equal: False`.
- **Probe 3:** `_publish_migration_aggregate` calls `aggregate_measurements` with no `njobs`
  (direct).
- **Probe 4:** `njobs=1`: `returned: None`, `master shape: (2, 2)`, `mirror exists: False`,
  `aggregate proof: None`. `njobs=2`: master path returned, `master shape: (0, 0) []`,
  `mirror exists: True (0, 0)`, `aggregate proof: {'source_image_count': 0}`.
- **Probe 5:** on both legacy trees the external Parquets have `user metadata cols: ()` and
  `join.status: None`. The embedded tables have 12 user columns, `join.status: b'joined'`, and
  `user cols INSIDE measurement_columns: ()`.
- **Job 28283848** (tree `2266fc5f` + src diff sha256 `038d1ce2b2f4806e`, exit 0, 00:05:57):
  aggregated 6529 of 6529 (excluded 0); master 128,598×136, user-metadata columns `[]`; mirror
  230,329 rows, 113,814 measured, 116,515 metadata-only, `_right` columns `[]`; null
  `Shape_Area`/`Metadata_Strain`/`Metadata_pH`/`Metadata_Salinity` on measured rows: 0; measured
  distinct images 5,319; elapsed 304 s; `RESULT: PASS`. Before the fix, the same tree's mirror had
  230,280 rows, 0 measured, and 13 `*_right` columns.
- **Advisory search:** no `_run_state.py` line pairs an advisory with aggregate/source/master/proof.
- **Footer inspection** (`tail -c | strings`): the legacy external Parquet names only
  `Metadata_{BitDepth,Dataset,ImageName,ImageType}`. The migrated table records `joined` and
  carries `Metadata_Strain` and `Metadata_PlateID`.
