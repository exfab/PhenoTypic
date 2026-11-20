# C19 gate: P4 Task 4 (CLI wiring and deliverables) and the P4 phase gate

Reported by the C19 implementer and written here by the orchestrator.
Per-test list: `/bigdata/exfab/anguy344/gate-results/p4/classified.tsv`.

## Commits

- `d60b47ed1`: `DIR_EMBEDDINGS`, `EMBEDDINGS_MANIFEST_JSON`, `embeddings_dir()`,
  `embeddings_manifest_path()`.
- `6b059559d`: `build_output_schema(tables, embedding_tables=)` and
  `embedding_vector_member`. The `embedding` column is resolved through its
  producer, as `FixedSizeList<float16>[D]` plus `model`.
- `d595abe95`: the single-pass worker, Stage 3 and measure mode call
  `pipeline.embed(image)` beside `measure()` and pass `embeddings=` to the one
  store write. `_collect_feature_headers` skips embedders.
- `f435f9845`: new `_cli/_cli_embedding_deliverables.py`
  (`aggregate_embedding_tables`, `embedding_table_schemas`). `finalize_run`
  calls it after `finalize_post_master_outputs` and before
  `publish_run_summaries`, then passes the counts to
  `build_run_metrics(embeddings=)` and the tables to `schema.json`. The README
  skips embedders and gains an "Embeddings" section. `_cli/CLAUDE.md` is
  updated.
- `9c3848ae0`: fixtures and tests.
  - `tests/integration/cli/test_embedding_deliverables.py`: 10 tests, all
    needing torch. Among them is `test_same_pipeline_measure_keeps_the_runs_crop_embeddings`,
    the P3 review OQ5 test.
  - `tests/unit/cli/test_embedding_aggregation.py`: 6 torch-free tests.

## Decisions

- **D9 and "finalization fails" do not conflict.** A record certifies only the
  store root and the measurement table, so a corrupt embedding leaves every
  image `verified` (`test_corrupt_embedding_does_not_revoke_measurements`).
  - Aggregation skips a `stale` entry, counts it and names the store in a
    warning (D7).
  - A non-stale entry it cannot read raises, naming the store and the remedy (a
    same-pipeline `--mode measure`), and finalization fails like a master that
    cannot be built.
  - **Orchestrator: keep the failure.**
- **Aggregation runs in `finalize_run` itself** (the `TASK_FINALIZE` entry on
  SLURM), not in the `0-K` shards. It streams one store at a time and has
  nothing to merge, so it adds no sidecar.
- **The certified run comes from the aggregated sources.**
  `aggregated_table_runs` calls `measurement_table_run(source)` on each
  measurements artifact the records named. That is the lookup
  `certified_table_run` makes, without re-reading every record.
- **Identity columns:** `Metadata_Dataset`, `Metadata_ImageName`, then the
  table's key columns, then `embedding`.
- **Manifest:** `{schema_version: 1, entries: {key: {path, rows, sha256,
  stores, stale_skipped, descriptors}}}`. It is canonical JSON with no
  timestamps, and its byte stability is pinned.
- **Stale deliverables are removed**, and `deliverables/embeddings/` goes away
  when no certified folder embeds. `metrics.json` then omits `embeddings`.
- **Measure mode re-embeds every image**, with one backbone load per process,
  as OQ5 requires.
- **Plan mismatches:**
  - The `schema.json` member keeps C12's `"<ENUM>.<MEMBER>"` form
    (`"DINO_EMBEDDING.EMBEDDING"`).
  - The plan's by-feature skip site (`_collect_feature_headers`) is test-only.
    The real split is over master columns, which carry no embedding, but the
    skip was added as the plan asks.
  - The README categories section needed the skip too.

## Queued for the P4 review-fix round (orchestrator)

1. **Declare the embedding deliverables in the aggregate proof's
   `required_outputs`:** `manifest.json` and each published parquet, and
   nothing when no run embeds. The proof already declares and enforces
   `metrics.json` and `schema.json`, and an undeclared deliverable could change
   or vanish while the run reads as published.
2. **`shard_paths` caveat:** intersect the embedding runs with the sources the
   projection kept, so a store excluded from the master contributes no
   embeddings.

## Verification

- **Seen red first:**
  - In the shared env, the 6 unit tests failed with `ModuleNotFoundError`.
  - In the foundation venv, the integration file gave `8 failed, 2 passed`.
    Those 2 passed vacuously because no deliverable existed, and were
    tightened to assert that the deliverable exists first.
- **Green:**
  - Shared env, the unit file: `6 passed`.
  - Foundation venv, the integration file: `10 passed`.
  - Focused job `32097635` over the 31 directly touched files:
    - shared env: `1064 passed, 10 skipped`;
    - doctests: `3 passed`;
    - foundation venv: `16 passed`.
- ruff: clean.
- mypy: one error, `_cli_finalize_run.py:22` import-untyped, on an import line
  this change did not touch.

## P4 phase gate

- **Array `32097662`:** 24 shards on `short` over 598 files: p3simp's 567, the
  P4 test files, the importers of each changed module, and every file under
  `tests/integration/cli`, `tests/unit/gui` and `tests/gui`.
  - Every shard ran at `9c3848ae0` with `dirty=0` at start and end, and
    imported `phenotypic` from the gate tree.
  - **10,904 tests: 10,810 passed, 13 failed, 0 errors, 81 skipped.** The
    orchestrator's junit sum agrees.
  - **The 13 reds are the 13 GUI expected-red rows** (README P0/P1/P3, #283),
    the same names as p3simp. None is new.
- **Foundation job `32097663`:** the P4 files plus tests/unit/measure, with
  torch 2.11.0+cu130 from the gate tree, at `9c3848ae0`, `dirty=0`.
  - `482 passed, 3 skipped`. The 3 skips are two golden-capture tests and one
    kaleido test, none of them torch.
- **The 81 shard skips:**
  - **34 are torch skips in P4 files**, and every one runs and passes in the
    foundation job: 21 in test_measure_dino_embedding, 3 in test_dino_support,
    and the 10 deliverables tests.
  - **47 predate P4:** P7 U-10 markers, absent migration fixtures, kaleido,
    exiftool, Windows-only tests and the disarmed schema gate.
- **Cleanup `32097664`:** COMPLETED 0:0, and the gate worktree is removed.
