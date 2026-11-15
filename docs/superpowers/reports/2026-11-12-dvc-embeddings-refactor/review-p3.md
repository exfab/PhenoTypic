# P3 deep review: tables v2

Reviewer: fresh `implementation-test-reviewer` (Opus) over `git diff 4bdc49798..HEAD`.
Returned as two messages; written here by the orchestrator. Repro scripts:
`/scratch/anguy344/32050482/claude-5188/-bigdata-exfab-anguy344-PhenoTypic/755c852f-f7be-4fc5-96a7-36f51adcec11/scratchpad/p3rev/`
(`check_migrate_v2_recert.py`, `check_migrated_provenance.py`, `check_crash_window.py`).

Scope run: the 7 new P3 test files, `122 passed in 93s` (`-n 4 -o addopts= -m "not slow"`).
The P3 diff adds no xfail or skip markers.

## Critical

### C1. Re-running `--mode migrate` on a v2 tree re-certifies every record to the latest run folder

- **Where:** `_cli_migrate_image.py:186-198` (`_embedded_table_path` → the
  record-less default `declared_measurement_table_path`), feeding
  `_migration_marker_artifacts` (:202-214) and `_valid_migration_marker` (:279-334).
- **What goes wrong:** a record that certifies a non-latest folder reads invalid;
  the marker stage republishes it at the latest folder with
  `_existing_marker_identity` defaults.
- **Repro** (forward A, `--mode measure` B, `--mode measure` A; folder 1 = A, certified;
  folder 2 = B, latest; then `--mode migrate`): exit 0, all passes report 0 work;
  certifies `1-…` → `2-…`; completion stays `complete`; `attempt_id` → `migration`;
  `relative_image_path` `ds/plate_1.tif` → `ds/plate_1`.
- **Impact:** the next finalize/recompile aggregates pipeline B while the user last
  measured with A — a silent D6a violation, on the documented, re-runnable recovery
  step.
- **Fix:** where the image has a record, prefer the table it certifies
  (`certified_measurement_table` + `measurement_table_run`); fall back to
  `declared_measurement_table_path` only with no record or a v1 one. Test: migrate
  over `certified_older_than_latest` leaves records and tree byte-identical.

## Important

### I1. The tables_v2 republish drops `provenance: "migrated"` and `source_provenance`

- **Where:** `_cli_migrate_image.py:263-275` → `_cli_recompile_tables._republish_table_marker`
  (:51-93) → `publish_image_success`; neither forwards `provenance` /
  `source_provenance`, so `publish_image_record` writes `PROVENANCE_FORWARD`
  (`_cli_image_record.py:92`).
- **Repro:** `provenance before=migrated after=forward`; `source_provenance` → `None`.
  Variant (U-10, migrated work_id ≠ configured): the republished record turns
  invalid and is rebuilt from defaults (`attempt_id=migration`,
  `relative_image_path=ds/plate`, work_id replaced).
- **Reach:** every tree migrated before P3 (the maresca and UCR beds). Measure mode
  uses the same function and has the same pre-existing loss.
- **Fix:** carry the record's `provenance` and `source_provenance` through; test
  parametrized over forward and migrated.

### I2. `_existing_marker_identity` reads only the legacy marker

- **Where:** `_cli_migrate_image.py:147-183` reads `progress/image_complete/`, never
  `image_record_path`.
- **Crash path:** `convert_tables_v1_to_v2` promotes, the process dies before
  `_republish_table_marker`; next pass: no conversion, the record names the missing
  flat path, the marker stage republishes with defaults.
- **Repro:** `attempt_id e28847ce… → migration`, `relative_image_path ds/plate.tif → ds/plate`,
  `lifecycle_epoch` replaced.
- **Fix:** read the record first, legacy marker only when no record exists; keep the
  work_id and dataset/stem checks and `provenance` / `source_provenance`. Add a
  crash-window test.

## Minor

- **M1. D2 "never share a number" is not enforced against `upstream`**
  (`_parse_runs`, `_table_runs.py:252-292`; `_write_table_runs`,
  `_image_io_handler.py:1586-1609`). A re-saved result keeps its own `2-A` while a
  grown input adds upstream `2-Y`; or the same id lands in both `runs` and
  `upstream`. Upstream entries are merge-only, so stale ones never drop.
- **M2. Two "which run" rules inside migrate's import.**
  `replace_embedded_measurement_table` (`_measurement_tables.py:1745-1750`) writes the
  run `v1_table_pipeline_sha` picks; `_valid_embedded_measurement_contract` and
  `embedded_measurement_table_matches` judge the latest. On a multi-run v2 store with
  a legacy external Parquet the import never converges.
- **M3. Seams and dead code.** `certified_table_run` (`_cli_recompile_recovery.py:339`),
  `table_run_by_id` have no caller; `latest_table_run`, `table_run_for_pipeline` are
  test-only, and the latter restates `owned_table_run_id`. The record-less default is
  answered four ways (`latest_measurement_run`, `declared_measurement_tables()[-1]`,
  `declared_measurement_table_path`, `_comparison_table`) — they agree today; C1 is
  what a new caller reaching for one of them does.
- **M4. An unparseable root drops every other pipeline's folders.**
  `_replaced_store_block` (`_measurement_tables.py:807-825`) maps ValueError to None,
  so a save carries nothing — against `carry_table_runs`' own rule.
- **M5. `first_v1_store` cost.** One root read per store on every forward invocation,
  `--dry-run` and the GUI Validate (~34k GPFS reads on the large runs); an unknown
  tables `schema_version` is swallowed and fails per image later. Acceptable per the
  spec.
- **M6. Docs.** `docs/source/contrib_guide/tracked_state.md:292` still says "no
  `tables.measurements` descriptor". `_cli/CLAUDE.md`'s "identity verbatim" claim is
  false while I1/I2 stand. The run README, skill, `_cli/CLAUDE.md`, root `CLAUDE.md`
  and `migrate_ome_zarr.md` otherwise read correctly.
- **M7. Test gaps.** No test that `--restart` keeps another pipeline's folder; none for
  the M1 collision; none that a same-pipeline `replace_image_tables` drops that run's
  embedding entry (intended until P4).

## Open questions raised

1. The D4 ngio discovery check never ran (permission). The region path
   `../../rgb/labels/objmap` is right under URL-style resolution, but no ngio reader
   has opened it.
2. D10 can mislabel: a pre-P3 `--mode measure` with a figure-less pipeline B
   rewrote a v1 table; figures and journal both name A, so it converts as `1-A` with
   B's content.
3. D2 vs D10 on `legacy` applications: `journal_pipeline_sha256` treats `legacy` as
   unknown (D2), `v1_table_pipeline_sha` accepts it (D10). After migrate,
   `journal_table_run` returns None though `1-<legacy sha>` exists, so staged and full
   resume probes read the image as having no table of its own.
4. Mixed certified folders: after `--mode measure` B, a plain continuation with A that
   adds images certifies 2-B for old images and 1-A for new ones; the master mixes
   columns and `metrics.json`'s `pipeline_sha256` is null. Follows D6a.
5. P4: a same-pipeline `--mode measure` deletes that run's embeddings until P4 adds
   `embeddings=`.
6. #283: `_measurement_source.py:107-108` still stats the flat path (already in the
   handoff).

## Orchestrator triage

Fixed in the review-fix cluster (C16r): C1, I1, I2, M1, M2, M4, M6, M7, and open
questions 3 and 4, decided within the spec as follows.

- **OQ3:** readers resolve a `legacy` latest application to the folder D10 labelled
  (`v1_table_pipeline_sha`), so migrated trees still resume; D2's writer rule is
  unchanged. Without this, a continuation over a migrated tree would re-derive every
  image P3 previously reused.
- **OQ4:** warn, not refuse. Mixed-pipeline certified folders follow D6a; an advisory
  names the pipelines, and `metrics.json`'s null `pipeline_sha256` already signals it.
- **M1:** a writer that knows its input replaces `upstream` with the fresh set rather
  than merging; a run/upstream order or id collision is logged.
- **M4:** refuse when the root exists but cannot be parsed, except on migrate's
  explicit replace-from-HDF path, which keeps "carry nothing".
- **OQ2:** recorded as a limit (undetectable from the store).
- **OQ5:** P4's brief carries it. **OQ6:** already in the #283 handoff.
- **M3:** the simplify pass. **M5:** recorded; no change.
- **OQ1** (ngio) remains with the user.
