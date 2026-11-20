# C14d and C16 gates: P3 Task 4 (state, sdk_ readers) and Task 6 (upstream)

Classified by each implementer; written here by the orchestrator.

## C14d: run state, the GUI-facing sdk_ readers, the run README, the skill doc

Files: `/bigdata/exfab/anguy344/gate-results/c14d/` (`classified.tsv`,
`owners.tsv`, `focused/`).

- **Main gate:** `32095965`, 24 shards at `375f8ed3f`. 10,198 passed, 44 failed,
  11 errors, 47 skipped → 55 reds (C13: 105; C14c: 67).
  - C15 (migrate): 41, the same set as at C13.
  - GUI expected-red: 13, down from 18 — test_measurement_source 5,
    test_setup_view 3, test_setup_authoring 2, test_measurement_routes 1
    (`test_an_undeclared_column_never_opens_the_parquet`, whose test unlinks the
    flat path itself), test_curation_does_not_unfence_the_run 1, test_validation 1.
  - New: 1, `test_directory_scanner::test_readme_documents_the_store_and_its_interoperability`
    (pinned the old README path) — fixed in `b8ed191ab`.
  - Green: test_run_state ×7, C14c's 39, and five test_measurement_routes rows early:
    `test_a_non_numeric_column_is_400`, `test_a_path_traversal_column_is_400`,
    `test_an_undeclared_column_is_400`, `test_serves_one_column_keyed_by_object_label`,
    `test_serves_only_the_requested_column`.
- **Focused gate for `a25492e19`:** `32096099` (45 importer files, 8 shards) — every
  red a known C15 or GUI row, except
  `tests/integration/gui/test_run_console_callbacks.py::test_run_app_owns_one_startable_observer_lifecycle`,
  which is order-dependent and fails alone at `87f9a1d8d` too (an atexit spy sees
  an extra callback when `create_app` runs first). One shard collected nothing
  (it named C16's then-untracked test file); its five files re-ran as `32096160`:
  `148 passed in 97.32s`, clean tree at `a25492e19`.
- **Commits:** state/reader changes inside `98633c2b4` (index race, see
  p3-c14c-gate.md), `d271c8300` (run README names the v2 layout), `c600b8b11`
  (skill doc v2 layout), `375f8ed3f` (staged-guard test after a surviving mutant),
  `a25492e19` (read-only v1 branch), `b8ed191ab` (README test).
- **Decisions (orchestrator, within the spec):**
  - Q1: `_run_state._verify_image` reads the metadata snapshot and the
    projectable flag from the run whose `measurements` path equals the record's
    artifact (D6a); with no measurements artifact, the snapshot comes from
    `latest_measurement_run`.
  - Q2 (a): the record-less readers keep a **read-only** v1 branch, keyed
    strictly on tables schema 1, in one helper,
    `_table_runs.declared_measurement_tables(store, block)`. D5 only makes forward
    modes refuse v1; `resolve_run_state`, the GUI and migrate planning read
    unconverted trees. `unprojectable_stores` gives `()` on a v1 tree and still
    names a v1 store with its descriptor removed.
- **Mutation:** 9 of 9 killed (6 on the readers, 3 on the v1 branch), each file
  restored from bytes and hash-checked.
- **mypy:** the 5 `_cli_migrate_state.py` errors predate P3 (present at
  `87f9a1d8d`); `a25492e19` fixes them. Only `_measurement_tables.py:1486` remains.
- **Skill doc follow-up** (orchestrator, `git log -1 -- .claude/skills/working-with-ome-zarr/SKILL.md`):
  table rewrites are root-last with no in-place path; mixed metadata snapshot
  digests are not refused.

## C16: upstream reference (Task 6)

Files: `/bigdata/exfab/anguy344/gate-results/c16/`.

- **Commit:** `e971f84c5`. `upstream_entries_for_input`, `follow_upstream` and
  `UpstreamChangedError` in `sdk_/_table_runs.py` (exported from `phenotypic.sdk_`);
  full mode, Stage 1 and process mode (`--process-format zarr`) pass `upstream=`;
  the staged SLURM worker derives the input root from the manifest entry.
  No numbering change was needed: T2's `_write_table_runs` already numbers after
  the passed upstream entries, so a result over `1-<A>` gets `2-<B>`.
- **Gate:** `32096116`, 8 shards over 57 importer files at `e971f84c5`. 2,015 passed,
  9 failed: 8 known test_finalize_run C15 rows, and the README test C14d fixed in
  `b8ed191ab` (passes at HEAD). No C16 regression.
- **Local:** `test_upstream_reference.py` `16 passed`; six adjacent files `103 passed`;
  mypy clean on 6 modules.
- **Limits, recorded (not a fork):**
  - Inherited `upstream` entries are kept verbatim, so their `source` is relative to
    the `--input` of the run that first recorded them; `follow_upstream` on one
    needs that earlier root.
  - Stage 3 records no upstream of its own; it carries what Stage 1 recorded. A
    staged run resumed across `e971f84c5` has no `upstream` block.
