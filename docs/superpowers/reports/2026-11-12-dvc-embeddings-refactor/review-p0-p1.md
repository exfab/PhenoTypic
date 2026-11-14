# Review: P0 (typed config suffix flip) and P1 (CSV → TSV) of #281

**Scope.** `git diff 00b945bef a3a4967e2 -- src tests` (P0: `39716aa71`,
`a1dd0b08e`, `ce28ec476`; P1: `9ff58070a`, `a1cba78b1`, `1ce26d06a`,
`b8caa1116`, `f363f1bc0`, `a3a4967e2`), checked against spec Parts A and B and
the P0/P1 plan files. Read-only review. No worktree file was edited. The
mutation runs used an in-memory pytest plugin kept in the session scratchpad
(`review-p0p1/mutants_plugin.py`), which monkeypatches module attributes and
writes nothing.

## Verdict

**Approve, once I1 and I2 are fixed.** No critical defects. The
spec's hard constraints hold:

- **No file under `src/phenotypic/_gui/` changed.** `git diff --stat … --
  src/phenotypic/_gui` is empty.
- **A rename never changes bytes.** The seed and the process copy copy the
  `--pipeline` bytes verbatim, and the snapshots are byte copies, checked by
  `test_seed_writes_new_name_byte_for_byte`,
  `test_process_copy_publishes_new_name_and_removes_old` and the snapshot tests.
- **Nothing rewrites `deliverables/metadata.<ext>`.** The stale sweep refuses
  the `metadata` stem, and the canonical view deletes only
  `metadata.canonical.csv`.
- **The sweep never removes the kept files.** `error_analysis.csv`, the
  snapshot and the `measurements.csv` mirror all survive, and `_KEPT_CSV_STEMS`
  is pinned by `test_the_sweep_refuses_the_csvs_that_stay`.
- **A4 holds.** The finalize writer, the `.phenotypic` process copy and
  `QcRecipe` retire the old-typed sibling, and plain `pipeline.json` is kept.
- **Old trees still read.** The resolvers cover the new, old-typed and plain
  names. A v1 analysis manifest and a v1 publication journal both read, and
  the snapshot resolvers find `.csv`.
- **The GUI import guard works.** Its AST walk is not vacuous (it finds
  `MEASUREMENTS_CSV`/`measurements_csv_path`), and every baseline signature
  matches.

**Tests.** The 10 directly touched files pass: 340 tests in 68 s, `-n 8`,
`not slow`. Of the four GUI files over changed read paths (37 tests), the one
red test is the listed P1 expected-red
`test_the_proof_still_records_every_published_artifact`.

**Can C6's tests fail?** Mostly, yes. I applied six in-memory mutants:

| Mutant | Killed by |
|---|---|
| `_cli_output_manager.remove_stale_csv_sibling` → no-op | `test_refinalize_old_tree_replaces_generated_csv`, `test_a_stale_pre_cutover_analysis_csv_is_swept` |
| `remove_other_suffix_snapshot` → no-op | `test_switching_suffix_replaces_the_snapshot_rather_than_adding_one` |
| CLI snapshot resolver sees `.csv` only | 3 tests (`.tsv` continuation/recompile, two-snapshot refusal, writer reuse) |
| finalize stops writing `measurements.csv` | 6 failures + 11 errors (the proof requires the file) |
| finalize stops unlinking the old-typed config | `test_finalize_publishes_new_name_and_removes_old` |
| **`_metadata_digest_for` reverted to `.csv`-only** | **survived**: these 5 files, plus `test_run_identity.py` and `test_embedded_table_inversion.py` (53 tests) |

**Counts:** Critical 0 · Important 2 · Minor 9 · Open design questions 7.

---

## Critical

None.

## Important

### I1. A suffix switch deletes the old snapshot before the new one is safely published

**Where.**
- `src/phenotypic/phenotypicCLI.py` `_snapshot_metadata_csv`: the
  `remove_other_suffix_snapshot(destination)` call before the state rewrite
  and `atomic_write_bytes`, in the `with exclusive_path_lock(...)` block,
  around `:1018-1031`.
- `src/phenotypic/_cli/_cli_reference.py` `snapshot_reference_metadata`:
  `remove_other_suffix_snapshot(destination)` before `atomic_write_bytes`. Its
  docstring states the order on purpose.

**Scenario.** A run was started with `--metadata plates.csv`, so
`deliverables/metadata.csv` exists. The user continues with `--metadata
plates.tsv`. Validation passes, then the function deletes `metadata.csv`.
After that, any of these failures aborts the run:

- `raise ValueError("Processing state config is invalid")`;
- a `json` decode error on `processing_state.json`;
- an `OSError` from `atomic_write_json` or `atomic_write_bytes` (quota, or a
  GPFS hiccup);
- the read-back verification `OSError`;
- a crash.

The tree now holds **no** snapshot.

**What goes wrong next.** A continuation without `--metadata` takes the full
mode path: `_snapshot_metadata_csv(None)` → `resolve_metadata_snapshot_path`
→ `None`. The run proceeds and finalizes a mirror with **no metadata join**,
and nothing refuses. `state.config["metadata_sha256"]` may already name the
new bytes that were never written. Before P1, a same-suffix replace was a
single `os.replace`, and the old snapshot survived every failure. That is the
property `test_invalid_metadata_never_replaces_existing_snapshot` guards for
parse failures, and this path now loses it for I/O failures. In process mode
the result is a loud `PF-REF-NO-TABLE`, which is better, but the
reference-table provenance is still lost.

**Fix.** Publish first, then retire. Write and verify `destination`, then call
`remove_other_suffix_snapshot(destination)`. A crash between the two leaves
both files, and `_resolve_snapshot` already refuses that loudly. The CLI
pre-check at `phenotypicCLI.py:~2706` turns it into a usage error, and
re-passing `--metadata` heals it through the same-bytes early-return branch,
which already calls `remove_other_suffix_snapshot`. That fails closed instead
of open. Update the `snapshot_reference_metadata` docstring to match.

**Test.** Monkeypatch `atomic_write_bytes` in `phenotypicCLI` to raise
`OSError` on a `.csv` → `.tsv` switch. Assert that `metadata.csv` still exists
with its old bytes and that `resolve_metadata_snapshot_path` returns it. Write
the same test for `snapshot_reference_metadata`.

### I2. The `.tsv` arm of the work-identity digest is untested, and a revert survives

**Where.** `src/phenotypic/_cli/_cli_identity.py:118-145`
(`_metadata_digest_for` now falls back to `resolve_metadata_snapshot_path`).
The only test is `tests/unit/cli/test_run_identity.py:564`
`test_the_two_metadata_digests_agree`, and it builds a `.csv` snapshot only.

**Scenario.** Revert the fallback to the pre-P1
`metadata_csv_deliverable_path` (`.csv` only). Every test in the five P0/P1
files, `test_run_identity.py` and `test_embedded_table_inversion.py` still
passes (mutant `identity_old_digest`, 53 + 69 tests green). That revert is
exactly the bug this function's own comment describes. On a continuation over
a `.tsv` snapshot that does not re-pass `--metadata`, the mint computes `None`
while `state.config` records the snapshot's real digest. The two then disagree
on every such continuation, and §7.4's late-metadata guarantee fires on every
run.
`test_continuation_and_recompile_leave_the_snapshot_byte_identical[.tsv]`
does not see it, because it asserts only the exit code and the snapshot bytes.

**Fix.** Parametrize `test_the_two_metadata_digests_agree` over
`(".csv", ".tsv")`, using `metadata_snapshot_path(root, ext)` in place of
`metadata_csv_deliverable_path(root)`. Arms 2 and 3 cover the `.tsv` case for
free.

---

## Minor

### M1. `test_write_tsv_failure_keeps_the_old_file` never reaches the atomic write

**Where.** `tests/unit/sdk_/test_delimited.py`. `write_tsv(object(), out)`
raises `TypeError` inside `_tsv_writer(frame)` (`_delimited.py`), which is
evaluated as an argument *before* `atomic_write_with_writer` is entered. The
test passes even if `write_tsv` wrote non-atomically.

**Fix.** Pass a frame whose writer raises after it starts writing, such as a
polars frame holding a `List` column (`ComputeError: CSV format does not
support nested data`, verified), or a pandas subclass whose `to_csv` writes
some bytes and then raises. Then assert the old bytes and that no `.tsv.tmp`
is left.

### M2. The old-tree mirror assertion cannot detect a finalize that stops rewriting `measurements.csv`

**Where.** `tests/unit/cli/test_tsv_cutover_old_tree.py`
`test_finalize_into_an_old_folder_keeps_the_csvs_that_stay`. It asserts
`read_csv(measurements.csv).equals(read_csv(measurements.tsv))`. The fixture
already wrote `measurements.csv` from the same frame, so the assertion holds
whether or not recompile rewrote the CSV.

The same test's `error_analysis.csv` and `metadata.csv` assertions guard
against code that does not exist, because no call site passes those stems to
the sweep. They can fail only if someone adds such a call. That has value as a
regression guard, but they say nothing about the refusal in `_KEPT_CSV_STEMS`.
The unit test `test_the_sweep_refuses_the_csvs_that_stay` is what covers that.

Today the gap is covered elsewhere: the aggregate proof requires
`measurements.csv`, and the `no_mirror_csv_rewrite` mutant failed 17 tests.

**Fix.** Make the fixture's `measurements.csv` differ from what finalize will
write, for example by appending a sentinel row or a curated-only column. Then
assert that it was replaced.

### M3. The sweep's `ValueError` escapes `_write_split`

**Where.** `src/phenotypic/_cli/_cli_output_manager.py` `_write_split`,
`try: remove_stale_csv_sibling(tsv_path) except OSError`. The kept-stem refusal
in `remove_stale_csv_sibling` raises `ValueError` and is stem-only, with no
check of the directory. A split key equal to `measurements`, `metadata` or
`error_analysis` would raise out of `_write_split` and abort finalization after
the TSV was published.

**Reachable?** Not today. Feature keys are measurer class names, the only
`CATEGORIES` label is `StartingMetrics`, and analysis ids exclude the reserved
stems through `validate_analysis_id`.

**Fix.** Either catch `(OSError, ValueError)` there and log, or make the kept
stems apply only to `deliverables/` itself. The guarded files live only there,
so `measurements_by_feature/metadata.tsv` is a legitimate split.

### M4. Recompile and migrate meet two snapshots with a traceback, not a usage error

**Where.** The two-snapshot pre-check (`phenotypicCLI.py:~2706`) is gated on
`cli_mode == "full"`. These paths call `resolve_metadata_snapshot_path` with
no guard:

- `_handle_recompile` (`:~4636`);
- `_handle_recompile_slurm` (`:~4219`);
- `_cli_migrate.py` `_run_migrate_owned` (`:1734`),
  `migrate_legacy_measurement_tables` (`:1370`), `_publish_migration_aggregate`
  (`:1066`) and `_ensure_migration_processing_state` (`:696`);
- the migrate workers (`_cli_migrate_worker.py:493,731`).

On a tree holding both `metadata.csv` and `metadata.tsv`, each raises a bare
`ValueError`. In the SLURM workers that is a failed image or reclaim task, not
a refusal.

**Fix.** Run the same `resolve_metadata_snapshot_path` pre-check for
`recompile` and `migrate` in the read-only half, and convert the error to a
`click.UsageError`.

### M5. `write_tsv` is not one byte format across its two libraries

**Where.** `src/phenotypic/sdk_/_delimited.py` `_tsv_writer`. Verified:

| Behaviour | polars | pandas |
|---|---|---|
| Datetime | `2026-01-01T12:00:00.000000` | `2026-01-01 12:00:00` |
| Empty string | `""` | empty field, the same as null |
| Named index | n/a | dropped silently (`index=False`) |

The writer is meant to be "one TSV writer", but no production caller passes a
pandas frame: analysis, splits, the mirror and the canonical view are all
polars.

**Fix.** Document that pandas frames are converted with
`pl.from_pandas(frame)` and written by polars, so there is one writer in fact
as well as in name. Or state the differences in the docstring.

### M6. A mis-suffixed table is refused, but the message hides the cause

**Scenario.** A comma-separated file named `meta.tsv` passes
`metadata_csv_payload` and the snapshot parse, because pandas reads it as one
column. The run preflight then refuses it with `PF-META-NO-KEYS` ("shares no
column with the images' ImageName…"). Verified by an end-to-end probe. That is
the correct outcome, but the message points the user at their column names
rather than at the delimiter.

**Fix.** In `delimiter_for`'s callers, or in the preflight check: when the
parsed header is a single column that contains the other delimiter, add
"…looks like a comma-separated table; name it `.csv`" (or the reverse).

### M7. `--metadata` with an unlisted suffix is a new refusal, not "still refused"

Spec B4 says "Any other suffix is still refused". Before P1, full mode's join
(`read_metadata_csv` → `pl.read_csv`) read any suffix, so a comma-separated
`plates.txt` worked. It now exits `2` at the Click callback. This is a
user-visible break that the release note in the README generator does not
mention.

**Fix.** Add one line to the README's TSV note and to the CLI reference.

### M8. Resolvers with no production caller, and tune outputs that keep two current files

**Defined, used only by tests or not at all:**
- `resolve_best_pipeline_path`;
- `resolve_phenotypic_cache_pipeline_path`;
- `tuning_spec_candidates`, used only through `resolve_tuning_spec_path`;
- `measurements_table_path`, both forms. No non-GUI code reads the mirror
  text file, so B6's "readers fall back to `.csv`" has nothing to attach to in
  `src/` outside the GUI.

**The tune CLI.** `tune/_tune_cli/_run.py:683,736,1244,1356` writes
`tuning_spec.pht-tune.json`, `best_pipeline.pht-pipe.json` and
`pareto/best_<obj>.pht-pipe.json`. Re-run into an older tune folder, it leaves
the `.json.pht-*` siblings beside them. A4 names only the run's pipeline file,
so this is in the spirit of A4 rather than its letter. Pareto files have no
old-name resolver.

**Fix.** Retire the old-typed sibling after each tune write, the same
two-line pattern as `_publish_process_pipeline_copy`. Either drop the unused
resolvers or wire them to a reader.

### M9. GUI-side effects missing from `issue-283-handoff.md`

All of these need #283 work, and none is in the expected-red table:

- `_gui/run_console/_request_safety.py:403`: `read_metadata_csv` now raises
  `ValueError` for a suffix other than `.csv`/`.tsv`, so the run console's
  metadata compatibility shows "blocked" where it used to parse. It now also
  accepts `.tsv`.
- `_gui/tune/_export.py:36`: `best_pipeline_path` is canonical-only, so the
  GUI's tune export finds nothing in a pre-flip tune folder.
- `_gui/analysis/_callbacks.py:95`: the `MergeMetadata` default
  `metadata_path: "metadata.csv"` names a file that a `.tsv` run does not
  have.
- `post/_join_metadata.py:183-186` (not GUI): `JoinMetadata` still compares
  `path.suffix == ".tsv"` case-sensitively. The CLI accepts `plates.TSV`; a
  pipeline-configured `JoinMetadata(metadata_path="plates.TSV")` is refused.
  Route it through `delimiter_for` (spec B4: "as `join_metadata` already
  does").

---

## Edge cases checked and found correct

- **`ensure_typed_json_suffix`.** `x` → `x.pht-pipe.json`. `x.json` and
  `x.JSON` take the typed stem with the caller's `.json` spelling kept.
  `x.pht-pipe.json`, `x.json.pht-pipe` and the all-caps forms are returned
  unchanged. All are pinned in `test_io_constants.py:284-297`.
- **Both config names present.** The resolver prefers the new name. The
  finalize writer and `QcRecipe` publish, then unlink the old name, under the
  publication lock, so a crash between the two leaves both files and readers
  stay correct. All three names share one lock identity (`test_io_constants.py:547`).
- **Both snapshots present.** The resolver raises, and full-mode continuation
  turns that into exit `2` before any write
  (`test_a_continuation_over_two_snapshots_is_refused_before_any_write`;
  for recompile and migrate see M4).
- **TSV quoting.** Tab, `\n`, `\r`, `"` and non-ASCII (`Δura3`) round-trip
  through both libraries. Fields are quoted only when needed. Line ends are
  `\n` with no `\r`. An empty frame writes the header only. Leading spaces are
  preserved.
- **Old-tree analysis.** A v1 manifest upgrades in memory and is rewritten as
  v2 on the next publish. A carried-forward v1 entry keeps `tsv: "<id>.csv"`,
  which the v2 validator accepts. A v1 publication journal recovers against
  its `.csv` staged and backup names. The sweep runs inside the analysis lock,
  after the manifest names the `.tsv`.
- **The aggregate proof on an old tree.** A pre-P1 proof without
  `measurements_tsv` is still valid, because the mirror descriptors are
  skipped (`_run_state.py:1280`).
- **Mixed-version locks.** A pre-flip writer maps `pipeline.json.pht-pipe` and
  `pipeline.json` to `.pipeline-config.migration.lock`, and so does the
  current writer.

---

## Open design questions (the spec leaves these open; not bugs)

1. **Which mirror text file is current after a curation?**
   `measurements_table_path` prefers `.tsv`, but until #283 a curation updates
   `.csv`/`.parquet` and leaves the `.tsv` stale. A future reader that follows
   the "TSV first" rule would read uncurated data. Should the fallback compare
   against the parquet (mtime or digest), prefer `.csv` while
   `measurements_tsv` sits in `_GUI_WRITTEN_PROOF_DESCRIPTORS`, or stay
   unwired until #283? Recommendation: leave it unwired, which is the current
   state, and say so in its docstring.
2. **"Keep the input's extension" versus lower-casing.** `plates.TSV`
   snapshots to `metadata.tsv`. Lower-casing keeps the resolver simple on a
   case-sensitive filesystem, and I think it is the right call, but spec B5
   reads as keeping the extension verbatim. Ratify it in the spec.
3. **Quote handling in user TSV.** B1 makes generated TSV quoted, and reading
   user TSV with the same quoting is symmetric. Many TSV producers, however,
   treat `"` as a literal (the IANA `text/tab-separated-values` convention). A
   user field that begins with `"` (`"WT" control`) is then mis-parsed or
   refused. Should user `.tsv` inputs be read with quoting off
   (`quote_char=None`), or is symmetry the rule?
4. **A `--metadata` that changes suffix.** The implementation replaces the
   snapshot and deletes the other suffix's file, exactly as a same-suffix edit
   always has. The spec says only "never rewritten". Is a suffix switch a
   replace (current behaviour) or a refusal? If it is a replace, fix the order
   (I1).
5. **Orphaned generated CSVs.** The sweep removes the `.csv` twin of a file it
   republishes. A split or analysis whose producer has left the pipeline keeps
   its pre-cutover `.csv` forever, so `CSV_THE_CLI_MAY_LEAVE` holds only for
   an unchanged pipeline. Is that the intended scope of B6?
6. **Pipeline basename in provenance.** Process stores record the pipeline's
   basename (`source_path`, e.g. `preprocess.pht-pipe.json`). The
   AutoConvertRaw-GC rename (A5) keeps `pipeline_sha256` but changes the bytes
   of every store written afterwards, compared with stores from before the
   rename. A5 promises only the sha. Is the change in byte-reproducibility
   across the rollout acceptable?
7. **Plain `pipeline.json` beside the new name.** The plan keeps a plain V1
   `pipeline.json` next to `pipeline.pht-pipe.json`. That is two "current"
   files, which spec A4 ("one current pipeline file per folder") does not
   state as an exception. Confirm that the V1-reader exception is intended.
