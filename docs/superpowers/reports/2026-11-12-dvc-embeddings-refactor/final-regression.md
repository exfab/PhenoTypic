# Final simplify pass and full regression (#281)

The `code-simplifier` agent (Opus) reported these results; the orchestrator
wrote them here. Logs and per-test tags are in
`/bigdata/exfab/anguy344/gate-results/final/` (`classified.tsv`).

## Simplify commits (quality only)

- **`73f4f10b3`:** 14 bare `sys.exit(1)` in `phenotypicCLI.py` become
  `RunExitCode.ERROR` (value 1).
- **`85fdca0f1`:** both process refusals call `writes_process_store` instead
  of restating the layer set.
- **`6c7fcb712`:** `config_in_place(config)` replaces the same `getattr` at
  the three work-id sites.
- **`cf3796a93`:** the `clear_tables` branch of `_rewrite_store_tables` is one
  nested `if`.
- **`d15870a0d`:** the `MeasureFeatures.requires_objects` docstring now says
  "skips", matching the E4 ruling.
- **`82601e09d`:** `BundleLayout`'s fields are documented once.
  - The docs build went from 734 to 730 warnings, and the four removed lines
    are the only difference.
  - The `store_publication_token` / `store_revision_identity` duplicate
    appears or not by `-j` merge order. It is recorded as a limit, and no
    `:no-index:` was added.
- **`d482863aa`:** a test-gap fix, found by this regression (see below).

Not simplified, with reasons:
- `_object_detector_path` and `_image_corrector_path`: a shared helper came
  out longer.
- Duplicate in-place dispatch: the call sites take different arguments.
- Three pipeline loads in the read-only half: each refusal tolerates its own
  load failure, and their order is pinned.
- The `level0_digest` alias: it is documented.

## Focused re-run at `82601e09d`

The run covered 189 files in both environments. The only reds were the GUI
expected-red `test_the_proof_still_records_every_published_artifact` and the
known load flake.

mypy reported 469 errors before and after the pass. ruff was clean.

## Full regression

**Array `32214187` and foundation job `32214188`, at `82601e09d`.**

- **Scope:** every `test_*.py` under testpaths, 923 files (unit 777, smoke 3,
  integration 84, gui 59). Earlier gates ran a selected 609.
- **Shards:** 24 on `short`. Every shard ran `dirty=0` at start and end and
  imported `phenotypic` from the gate tree.
- **Foundation job:** 50 torch files with torch 2.11.0+cu130.

| Lane | Tests | Passed | Failed | Errors | Skipped |
|---|---|---|---|---|---|
| Shards | 17,008 | 16,841 | 14 | 0 | 153 |
| Foundation | 1,066 | 1,052 | 3 | 0 | 11 |

### The reds

- **13 are the GUI expected-red rows** (README P0/P1/P3, #283), matching the
  same names as every earlier gate. None of the known flakes went red.
- **`tests.smoke.test_measurement::test_measurement[...MeasureDinoEmbedding...]`
  was a #281 P4 test gap.**
  - The smoke test default-builds every measurer. `MeasureDinoEmbedding`
    entered that list at `4d16b84d8`.
  - The default build needs torch and the DINOv2-base weights, which are
    about 350 MB and come from the network.
  - `tests/smoke` had never been in a gate before.
  - In isolation, interleaved with `3790f9448` (job `32214360`), the test
    failed 3 of 3 times at both SHAs.
  - **Fixed in `d482863aa`.** Embedding producers are excluded at the
    smoke parametrize site, the same way C18 did it in
    `test_output_headers`. The stub-backbone tests cover the measurer.
    `test_serialization` still round-trips embedders.
  - **Before the fix** (`82601e09d`, job `32214616`):
    - shared env: `1 failed, 381 passed`;
    - torch env: `4 failed, 378 passed`.
  - **After the fix** (`d482863aa`, job `32215338`):
    - shared env: `381 passed`;
    - torch env: `3 failed, 378 passed`.
  - The 3 torch reds are `FilFinderDetector` smoke cases. The foundation venv
    lacks the `topology` extra (astropy), which is CLAUDE.md's
    false-failure case, and they pass in the shards' env.
- **3 foundation reds were environment, not code:**
  `test_sam2_detector.TestSam2DetectorFunctional::{test_apply_produces_objects,
  test_objmask_objmap_consistency, test_pipeline_apply}`.
  - On node r42 the checkpoint download hit `CERTIFICATE_VERIFY_FAILED`.
  - In isolation all three skip at both SHAs, because the HEAD probe failed.
  - #281 doesn't touch the SAM2 code or this test.
  - The probe can pass while the real download fails. This is pre-existing
    network flakiness.

### Skips

- **Shard skips:** 100 of the 153 need torch or weights.
  - 89 of those ran and passed in the foundation job.
  - The other 11 are the 3 SAM2 env reds, and 8 that need weights or
    micro_sam there too. None of the 11 is a P4 or P5 test.
- **The remaining 53 shard skips** are pre-existing: migrate tripwires,
  absent fixtures, Chrome/kaleido, exiftool, Windows, Postgres and the like.

### Cleanup

Every gate, isolation, smoke and docs worktree was removed by its `afterany`
cleanup. One docs cleanup timed out and was finished by a follow-up job.

## Leftovers for the PR

- **#283 GUI breakage:** the 13 expected-red GUI tests. **#281 does not merge to
  `main` before #283 fixes them.**
- **Copy-mode skip warning:** it is logged once per image. In place it is
  logged once per process.
- **`--durable-writes` is inert in process mode,** copy and in place. This is
  pre-existing (P5 review M4).
- **Local and SLURM measure** use different work-id path bases.
- **The order-dependent docs duplicate** (`store_publication_token` /
  `store_revision_identity`).
- **mypy:** the touched files carry 469 errors, the same before and after.
  `_measurement_tables.py` has one pre-existing `union-attr`.
- **ngio** opens `tables/<run>/measurements` directly, but doesn't discover
  nested tables (the D4 limit).
- **The user's Hugging Face token has expired.** Refresh it with
  `hf auth login` before using DINOv3.
- **An expired token surfaces as a misleading "repository not found"** from the
  shared DINO loader. A clearer message is a follow-up.
- **Known flakes**, all in `test_run_console_callbacks`, plus one in
  `test_cli_terminal_failures`:
  - `test_immediate_local_exit_terminalizes_allocated_generation` (a cold first
    run);
  - `test_run_app_owns_one_startable_observer_lifecycle` (test order);
  - `test_concurrent_process_appends_do_not_lose_records` (load).
- **The AutoConvertRaw-GC branch `feat/pht-suffix-flip`** is local and not
  pushed.
- **C22, the AutoConvertRaw-GC rollout,** needs the user's rig1 grid edges.
  **C23, the pilot and backfill,** is the user's call.
