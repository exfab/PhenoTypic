# P4 simplify pass and phase regression gate

The simplify agent was `code-simplifier` (Opus). Its report is written here by
the orchestrator. The per-test list is at
`/bigdata/exfab/anguy344/gate-results/p4simp/classified.tsv`.

## Commits

- **`426463e8c`** spells every SLURM log folder through `slurm_logs_dir`.
  - There were 10 sites: the sentinel, the recompile scripts, submission, the
    array scripts (2), staged, `_execution/_slurm.py` (3), the recompile
    `tail -f` hint in `phenotypicCLI.py`, and the output manager's mkdir.
  - Every path is byte-identical (`DIR_SLURM_LOGS == "slurm"`). Recompile
    keeps its `recompile/` subfolder.
- **`fd79924e3`** dedupes the embedding deliverables.
  - `_file_sha256` now imports the identical copy in `sdk_._embedding_tables`.
  - The manifest path and sha256 are set inside the staging branch.
  - The doubled `isinstance` check in `_plan_keys` is gone.
  - No output byte, message, order or exception changes.
- **`0210a0ea3`** (a behaviour change, kept outside the simplify commits): an
  embedding entry key that is not `<model>/<crops|grid>` is refused by one
  shared rule, `split_embedding_key`.
  - `EmbeddingPayload` uses the same rule on write.
  - Staging names such an entry unreadable, so nothing is published.
  - Red first: `test_a_hand_edited_key_that_is_not_model_and_region_fails_and_changes_nothing`
    gave `1 failed` (`DID NOT RAISE`).
  - After the fix:
    - shared env: `56 passed`;
    - torch: `17 passed`;
    - ruff and mypy: clean.
  - The test drives the aggregator directly. A CLI run cannot reach this case,
    because a hand-edited root voids the image's record first. It is a
    defence, like the stale branch.
  - It is not in the gate below. The end-of-implementation full regression
    covers it.

## Not simplified, and why

- `_canonical` versus `canonical_digest`, the `params_sha256` and the
  grid-edges hash: swapping them changes digest bytes (`ensure_ascii`,
  `sort_keys`).
- The measurer's `load_dino_backbone` delegate keeps `phenotypic.detect` out
  of the measure import, and tests patch that name. `_cached_backbone` is the
  per-process cache.
- The staleness refresh runs up to three times per save. Each pass is
  idempotent, and each function's docstring makes it part of its own
  contract.
- Two manifest readers, one strict and one tolerant. A shared loader would
  have to pick one rule.
- Four sites filter `MeasureFeatures` minus `MeasureEmbedding`. Each reads
  naturally where it is.
- `_prune_empty_directories` runs twice. The inner run is under the commit
  guard.

## Verification

- **Repointed sites:** `719 passed`.
- **The P4 files plus the import guards:**
  - shared env: `495 passed, 47 skipped`;
  - torch: `291 passed`.
- **ruff:** clean. **mypy:** no issues in the touched files.

**Gate.** Array `32168566` and foundation `32168567`, at `fd79924e3`.

- **Provenance:** 24/24 shards and the foundation job ran at `dirty=0` from
  start to end and imported `phenotypic` from the gate tree. Torch was
  2.11.0+cu130.
- **Files:** 609, being p4r's 604 plus:
  - `test_preflight_embedding_revision.py`;
  - three importers of the repointed modules;
  - `test_pickleable.py`.
- **Shards:** 11,106 tests. 10,992 passed, 14 failed, 0 errors, 100 skipped.
  The orchestrator's junit sum agrees.
- **The 14 reds:**
  - 13 are the GUI expected-red rows, the same names as p4r;
  - 1 is `test_run_console_callbacks::test_run_app_owns_one_startable_observer_lifecycle`;
  - none is new.
- **The observer-lifecycle red** is a test-order dependency, not a regression.
  - Run alone, it fails 3 of 3, at both `fd79924e3` and the pre-simplify HEAD
    `3c71dc3aa` (job `32170690`, same node).
  - The cause: `LocalRunner` registers its atexit hook inside the test's spy
    unless an earlier test registered it first.
  - The whole file passes 3 of 3 at `3c71dc3aa`. At `fd79924e3`, its first,
    cold run had one failure in 13.5 s, against 7.6–8.6 s warm, and the next
    two passed. Its name wasn't captured, but the pattern is the known
    cold-start flake.
  - The pass touches nothing under `_gui/`.
- **The cold-start flake** `test_immediate_local_exit_terminalizes_allocated_generation`
  is green in this gate.
- **Foundation:** 713 tests: `706 passed, 7 skipped`, 0 failed.
- **Cleanup:** `32168568` and the isolation cleanups completed 0:0, and no gate
  worktree remains.
