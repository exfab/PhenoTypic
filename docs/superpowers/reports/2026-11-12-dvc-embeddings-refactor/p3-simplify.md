# P3 simplify pass and phase regression gate

Simplify agent: `code-simplifier` (Opus). Its report is written here by the
orchestrator. Per-test list: `/bigdata/exfab/anguy344/gate-results/p3simp/classified.tsv`.

## Commits (quality only; no behaviour change intended)

- **`0542525fc` (review M3)** deletes `latest_table_run`,
  `table_run_for_pipeline` and `table_run_by_id`, with their sdk_ exports.
  - Tests use `list_table_runs` and `tests/_tiny_run.owned_table_run`, which is
    built on `owned_table_run_id`. Every assertion is kept.
  - The record-less default now has one owner,
    `_table_runs.record_less_measurement_table`. `latest_measurement_run`,
    `declared_measurement_table_path`, `_latest_measurement_descriptor` and
    `_run_state`'s no-record branch delegate to it.
  - A private `_run_of_block` replaces two copies of the id lookup.
  - SKILL.md now names `owned_table_run_id`.
- **`d552926fd`**: `_cli_parquet_agg._declared_measurement_descriptor` looks the
  path up through `declared_measurement_tables` instead of spelling out both
  layouts by hand.
- **`d0431d373`**: `_cli_migrate_image._certified_run_table` calls
  `certified_table_run`.
- **`e5ca5393b`** (orchestrator): the P4/P5 plan snippets use the surviving
  helpers, and `plan.html` is re-rendered.

## Decisions

- **`certified_table_run` is kept.** The plans name it as P3's output
  (p3-tables-v2.md:404) and as P4 Task 4's input (p4:352). Since `d0431d373` it
  also has a src caller.
- **Not consolidated:** `store_holds_measurement_table` and the v1 branch of
  `_comparison_table`. Both decide v1 by whether the file exists, not by the
  tables schema, so routing them through the shared rule would change answers
  for a flat file that has no descriptor.

## Verification

- Focused runs (`-n 4`/`-n 6`):
  - the 10 P3 files plus 6 directly affected files: `327 passed, 1 xfailed`;
  - the `_table_runs.py` doctests: `3 passed`;
  - the startup and deferred import guards: `249 passed`;
  - the parquet-agg consumers: `85 passed`;
  - the migrate files: `181 passed, 2 xfailed`, both pre-existing P7 U-10.
- ruff on every changed path: clean.
- mypy: the only error is `_measurement_tables.py:1510 union-attr`. That is
  pre-existing code (blame `d399f85aba`, 2026-09-23), the known error at a
  shifted line.
- **Gate:**
  - Job `32096929`, 24 shards on `short`, 567 files (C16r's list), detached
    worktree at `d0431d373`.
  - Every shard printed `dirty=0` at start and end, and all 24 COMPLETED.
  - Totals: 10,374 tests: 10,314 passed, 13 failed, 0 errors, 47 skipped. The
    orchestrator's junit sum agrees.
  - The 13 reds are the same 13 GUI expected-red rows as C16r (README
    P0/P1/P3, #283). None is new.
