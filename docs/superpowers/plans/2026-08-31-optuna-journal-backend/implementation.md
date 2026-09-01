# Standalone Optuna Journal Backend Move

**Goal:** Move only the CLI/tuning journal work from `feat/mcp-server` onto
`ome-zarr-merged`, make distributed Slurm tuning standalone, and preserve the
target branch's OME-Zarr and lifecycle behavior as canonical.

**Base:** `ome-zarr-merged` at `3e8b527930ecee8f9088512d3bbe130360050d30`.

**Spec:**
`docs/superpowers/specs/2026-08-31-optuna-journal-backend/design.md`.

## Global Constraints

- Use `uv` exclusively. Write tests first and observe the expected failure before
  every production behavior change.
- Keep the target `_load_images` OME-Zarr implementation and shared Slurm
  lifecycle authoritative. Manually layer the tune changes around them.
- Do not port `_services`, MCP/FastMCP code, service tests, MCP design documents,
  or the source branch's Optuna logic-validation script.
- A terminal `afterany` finalizer is allowed; no scheduler sidecar may run in
  parallel beside an active ordinary array.
- Any actual Git conflict or semantic divergence stops integration for user
  discussion. Do not silently resolve it.
- Full verification runs through Slurm and groups tests into massively parallel
  jobs. Use the existing `.verification/final-head` harness as the starting point.
- The final branch consists of three curated logical commits matching Tasks 1–3.

## Task 1: CLI, Slurm Arguments, and Journal Storage

Add the standalone command and storage protocol without changing distributed
publication ownership yet.

- Add `phenotypic-tune = "phenotypic.tune.__main__:main"`; retain module and
  bare-spec invocation compatibility.
- Make tune `--slurm` valid alone and repeatable as `--slurm KEY=VALUE`; retain
  legacy tune Slurm flags. Explicit key/value occurrences win. Deduplicate by
  rendered SBATCH directive so aliases such as `mem_gb` and `slurm_mem` emit one
  `--mem`.
- Resolve storage with CLI > spec > environment > mode default. Default local
  Optuna to SQLite and Slurm Optuna to an absolute run-local `journal://` URL.
  Reject explicit SQLite with Slurm, password-bearing URLs, and screen plus
  Slurm before artifacts are created.
- Port the JournalStorage store/protocol, retry, torn-tail recovery, terminal
  trial filtering, and `_base_scheme` URL helpers. Open existing journal stores
  without manufacturing a missing store.
- Add red/green tests for console/module equivalence, bare-spec compatibility,
  semantic Slurm parsing, alias deduplication, storage dispatch/precedence,
  torn-tail recovery, terminal winners, and pre-artifact failures.

## Task 2: Lifecycle-Owned Distributed Finalization and Monitor

Make Slurm tuning independently publish its result set.

- Initialize a tune lifecycle generation and submit the worker array through the
  existing drip-feed dispatcher with one terminal `afterany` finalizer. The
  finalizer uses the worker compute profile/interpreter with array settings
  removed.
- Automatic finalization uses the exact-generation publication guard, opens the
  existing backend read-only, requires the expected terminal-trial budget and a
  valid winner, writes all tuning/generalization outputs, and closes the owned
  generation while still guarded. Owned failures become failed/inactive and
  exit nonzero; stale finalizers never mutate a newer generation.
- Add `phenotypic-tune finalize OUTPUT [--force]` and normalize it as a real
  subcommand. Normal recovery holds the lifecycle lock from ownership check
  through publication and refuses an active generation. Forced recovery cancels
  the recorded generation, requires `CancellationResult.quiescent` and no
  unresolved tokens, reacquires the lock, verifies no new owner, and holds it
  through byte-idempotent publication.
- Bind finalization to the marker's exact supported study identity before any
  storage open. Bind native multi-objective vectors to the scorer's ordered,
  unique, safe axis names with exact result keys and at least two axes.
- Bound Monitor storage reads and keep fANOVA outside the polling timeout.
- Preserve target OME-Zarr loading and add a real minimal `.ome.zarr` regression
  exercised by submitter and worker paths.
- Cover stale/superseded finalizers, incomplete budgets, missing backing stores,
  submission failure, no winner, manual refusal, non-quiescent forced
  cancellation, automatic/manual exclusion, lifecycle closure, byte-identical
  refinalization, Monitor timeout behavior, and finalizer resource inheritance.

## Task 3: Standalone Documentation and Acceptance Harness

- Finalize the standalone design spec under the matching dated specs directory.
- Update tune how-to, tune explainer/data-flow documentation, root quick-start,
  GUI portable/copy commands, and help text to lead with
  `uv run phenotypic-tune`; internal GUI subprocess argv may remain
  `sys.executable -m phenotypic.tune`.
- Add or extend Slurm verification assets under the branch's ignored
  `.verification` directory: dedicated tune and tune-GUI/packaging groups in the
  broad array, a four-node GPFS journal test with synchronized start, hostname
  proof, symlink locking, and exactly 60 terminal trials, and a native
  `phenotypic-tune run ... --slurm ...` OME-Zarr acceptance run proving terminal
  publication and byte-idempotent recovery.
- Keep the Optuna/PhenoTypic cross-node harness out of
  `logic_validation_scripts/`; there is no new numeric invariant requiring such
  a script.
- Add a negative diff gate for `_services`, MCP/FastMCP production paths and
  imports, service tests, and MCP design documents.

## Final Verification and Integration

- Run explicit-path Ruff, affected-module mypy plus baseline comparison, and
  `git diff --check`.
- Submit the broad exact-head unit/integration array through Slurm, using
  `uv run --frozen --no-sync` and xdist while keeping aggregate resources below
  account limits.
- Run the four-node GPFS journal gate and native tune Slurm acceptance gate as
  separate jobs and verify their durable evidence.
- Dispatch an independent exact-head reviewer for standalone completeness,
  lifecycle safety, OME-Zarr preservation, test sufficiency, and MCP leakage.
- Merge into `ome-zarr-merged` only if its tip still equals the recorded base and
  the review and all gates are clear. Otherwise stop before modifying the target.
