# #282 simplify pass (P7 Task 2, Step 1)

**Scope:** `git diff --name-only 99bbc39cd..HEAD -- src/` (38 files) at
`1d5b28476`, plus `sdk_/_table_runs.py` and `sdk_/_embedding_tables.py`. Those
two are outside the diff, but C18 attributed their autodoc warnings to #282.
The pass made quality changes only. Each fix is its own commit, and none
touches `_gui/`, the P0 goldens, the logic-validation scripts or vendored refs.

## What changed

| # | Commit | Change | Files |
|---|---|---|---|
| D1 | `cb75ee865` | Dataclass field docs moved from an `Attributes:` section to a `#:` comment on each field, so autodoc renders each field once instead of twice (napoleon plus `:members:`). Same text. Fixes the 15 duplicate-object warnings. | `abc_/_measure_embedding.py` (`PreparedRegions`), `sdk_/_embedding_tables.py` (`EmbeddingPayload`), `sdk_/_table_runs.py` (`TableRun`) |
| D2 | `f2ca763f4` | The `Returns:` section of `entry_descriptor_for` opened with ``` ``{model: ``` and napoleon split it on that colon, leaving an inline literal unclosed. The section is reworded so its first line has no colon. Same content. Fixes 2 warnings (the method is documented on two pages). | `measure/_measure_dino_embedding.py` |
| S1 | `97ce82ae2` | In `_publish`, the forward and republish branches ended with the same check, signal delete, temp delete and `completed` event. Now only the record step is a branch (`_publish_record` holds the forward path). `_consume_signals` (signals, then temps) replaces the same pair in `_publish`, `sweep_orphan_stage4_signals` and `sweep_republished_stage4_signals`. The calls and their order are unchanged, and the calls still go through the module globals that the `crash_after` and spy harnesses wrap. | `_cli/_cli_stage4_worker.py` |
| S2 | `4cee04994` | Four `except` blocks re-raised a fenced generation's cause and otherwise journalled the image. They now call one `_record_unless_fenced`. It is called inside the `except`, so `__context__` and `traceback.format_exc()` see the same exception as before. | `_cli/_cli_staged_slurm_worker.py` |
| S3 | `15b9b24b9` | `LocalParallelStrategy.run` had a forward and a republish Stage-4 branch. They are exclusive (`measure_only`), and both ended with the same eight-line count write-back. The second branch is now `elif` and the write-back runs once. | `_cli/_cli_execution_strategies.py` |
| S4 | `f4b301e75` | `_cpu_pinned(embedder)`, beside `_has_stage4_encoder`, replaces the `getattr(embedder, "device", None) == "cpu"` test in `_cpu_pinned_key` and R2. | `_cli/_cli_embed_stage.py` |
| S5 | `4bf81119d` | `_classify_image` restated `stage4_signals_complete` as `state.valid == set(keys)`. It now calls it, which also gives that tested function a production caller. | `_cli/_cli_stage4_worker.py` |
| S6 | `371b24492` | One `forward_published` predicate and one `measure_table_run` lookup, both in `_cli_staged_resume`. Before, three copies of "the record has non-empty `artifacts`" existed (the worker's `_forward_published`, the controller's `_has_published_record`, and inline in `_republish_stage4_status`), and two copies of the owned-run lookup (the worker's `_republish_run`, and inline in `_republish_stage4_status`). The `record is None` narrowing also clears 2 mypy `union-attr` errors (below). | `_cli/_cli_stage4_worker.py`, `_cli/_cli_staged_controller.py`, `_cli/_cli_staged_resume.py` |
| S7 | `cf6c88d2c` | `run_local_stage4` and `run_stage4_shard` each listed `(Stage4DeviceError, Stage4OutOfMemory, Stage4ShardError)` plus `is_out_of_memory`. The shard also listed `MemoryError`, which `is_out_of_memory` already covers. Both now call `is_stage4_node_state`. | `_cli/_cli_stage4_worker.py`, `_cli/_cli_staged_slurm_worker.py` |
| S8 | `df89b427d` | Adds `Dinov3CheckpointManager.require_license(interactive=)`. The embedder's `_backbone`, Stage 4's `_require_licences` and the manager's own `download` each wrote out `require_license_acceptance(license_key, license_name, license_url, ...)`. The message and the `interactive` flag are unchanged. | `detect/nn/_helper/_checkpoint_manager.py`, `measure/_measure_dino_embedding.py`, `_cli/_cli_stage4_worker.py` |
| S9 | `420c30d23` | Tidy only, no code path changes: <ul><li>imports: removed the controller's local imports of `stage4_status`, which it already imports at the top, and moved `staged_store_matches_work_id` and `stages_1_to_3_done` into that import; made `_cli_stage4_signal` import `dataclasses` once, and renamed a loop variable that shadowed `field`; sorted `run_local_stage4`'s imports;</li><li>blank lines: fixed the PEP 8 blank lines in the worker, the SLURM worker and `_batch_encoder`;</li><li>`stage4_signal_mtimes` builds each path once;</li><li>reflowed the overlong docstring and comment lines among #282's additions (code lines left as they were).</li></ul> | `_cli_embed_stage.py`, `_cli_stage4_signal.py`, `_cli_stage4_worker.py`, `_cli_staged_controller.py`, `_cli_staged_slurm_worker.py`, `abc_/_measure_embedding.py`, `_batch_encoder.py`, `phenotypicCLI.py` |

### The 21 sphinx warnings (C18 log `sphinx-fac392a2d…err`)

| Warnings | Source | Outcome |
|---|---|---|
| 15 duplicate object descriptions (`PreparedRegions` 7, `TableRun` 5, `EmbeddingPayload` 3) | `Attributes:` sections on dataclasses | fixed, D1 |
| 2 "Inline literal start-string without end-string" (`entry_descriptor_for:26`, through `measure/__init__` and the module) | the colon inside the literal on `Returns:`'s first line | fixed, D2 |
| 2 "Bullet list ends without a blank line" (`pydantic.main.BaseModel.model_json_schema:8`, attributed to `_measure_embedding.py` and `_measure_dino_embedding.py`) | pydantic's own inherited docstring | **not fixed** (R1) |
| 2 "Inline emphasis/strong start-string without end-string" (`phenotypic.schema.DINO_EMBEDDING.rst:84`) | the inherited `str.format` docstring, `S.format(*args, **kwargs) -> str`, in the enum's member table | **not fixed** (R2) |

I checked D1 and D2 with a minimal Sphinx project, built in the session
scratchpad and not kept. It used the repo's `_templates`, the same
`autodoc_default_options`, napoleon settings and autosummary of the three
classes, plus `automethod` of `entry_descriptor_for`.

- At the pre-change source it gives **16** warnings, ignoring the pydantic and
  `str.format` ones: the 15 duplicates, and the inline literal once, because
  the mini project documents the method on one page.
- After the change it gives **0**.
- The rendered field text is present on each class page, with one
  `sig-object` per field (checked for `TableRun.order`).

The full docs build has not been re-run. That belongs to the lead's
regression.

## What was rejected, and why

- **R1, pydantic's `model_json_schema` warning (2 of the 21).** The text is
  pydantic's own docstring, inherited by every operation because
  `autodoc_default_options` has `inherited-members: True`. The same log has 394
  warnings of this kind, across every operation module, so #282 did not
  introduce it. The only fix in our source would be overriding
  `model_json_schema` on the embedders, which changes a public method. The real
  fix is in `docs/source/conf.py`: skip inherited `pydantic.BaseModel` members,
  or clean that docstring in an `autodoc-process-docstring` hook, for all 394.
  That is a docs-config follow-up.
- **R2, `DINO_EMBEDDING.rst:84` (2 of the 21).** The table row is the inherited
  `str.format` (`S.format(*args, **kwargs)`), not the enum docstring or `desc`.
  Every `str` enum page has it (`EDGE_CORRECTION`, `EXPERIMENT`,
  `ErrorCategory`, `GENETIC`, …), so editing `desc` fixes nothing. Like R1,
  the fix belongs to the docs config.
- **Deleting `stage4_signals_complete`**, which had no `src/` caller. It is
  the plan's named API (P3), and `_classify_image` was restating it, so S5
  gives it a caller instead.
- **Deleting `MeasureDinoEmbedding.crop_geometry()`**, which has no `src/`
  caller (P1's named API, used in a test). It is a public method of a public
  class, and public signatures are a hard limit.
- **`STAGE4_PRELUDE_BASH` spells `:4096:8` rather than using
  `CUBLAS_WORKSPACE_DEFAULT`.** Using the constant would make SLURM script
  generation (`_cli_staged_slurm`) import `detect.nn._helper._batch_encoder`
  at module scope, a heavier import on a path the startup-import guards
  watch. The value is pinned by the script tests on one side and by
  `DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS` on the other.
- **`Sam3CheckpointManager.download`'s identical licence call.** That is
  #281-era code in another manager. S8 is scoped to DINOv3, the model #282
  uses.
- **A third "record has `artifacts`" copy in
  `_cli_process_single.py` (~line 823)**, and the duplicated
  `parquet_metadata` blocks in `sdk_/_measurement_tables.py`. Both predate
  #282 (`git blame`: `3743eede0`, #281), and neither is #282 code.
- **The repeated lazy `from phenotypic._cli._cli_embed_stage import …`
  inside `phenotypicCLI.py`.** The CLI keeps imports in functions by design
  (lazy entry points, `test_startup_imports.py`). Hoisting them is a startup
  decision, not a simplification.
- **The `except Exception: if is_out_of_memory(exc): raise; self._fail(...)`
  shape** in `_BackboneStream._drain`, `_signal` and `_loader_items`. Each one
  does something different after the re-raise (fail the image, fail the
  image, or raise `Stage4ShardError`), so a helper would add indirection
  without removing logic.
- **`apply_stage4_counts`' dict-of-dicts round trip** in
  `LocalParallelStrategy`. The function is shared with `StagedGpuStrategy`,
  which keeps its counts in that shape, and is tested on it. S3 removes the
  duplication without changing that signature.
- **`recorded_stage = DEFAULT_EMBED_STAGE if recorded is None else …`** in
  both `refuse_embed_stage_switch` and `adopt_recorded_embed_stage`. It is one
  expression in each, and a helper would only rename it.
- **The three `"Stage 4 left %s/%s pending: %s"` log calls.** They sit in
  different branches with different reasons; a helper would rename a logger
  call.

## Verification

Each change was tested before its commit, on a **detached worktree at a
snapshot of the tree with that change applied**. `git stash create` made the
snapshots, so the run measured exactly the tree being committed while editing
continued in the checkout. Each snapshot includes every earlier change, so the
last one (`711e6b66b`) is `HEAD`'s `src/` byte for byte (`git diff --quiet`).
Each commit's diff was then applied from its own snapshot pair
(`git diff <prev> <snap> | git apply --cached`).

- The script is `gate-results/c19-final/simplify/run_snapshot.sbatch`, on
  `intel` with 4 CPU, 32 GB and 2 h, never `-n auto`.
- The shared lane runs `.venv/bin/python -m pytest`; the torch lane runs the
  foundation venv's. Both set `PYTHONPATH=src:.`, and each job asserts that
  `phenotypic` imports from its own snapshot.
- Both lanes run `-q --no-header -p no:randomly -o addopts= -m "not slow"`.
- Logs are under `gate-results/c19-final/simplify/`.

| Change | Snapshot | Lane | Files | Summary line |
|---|---|---|---|---|
| D1, D2 | in place, at `1d5b28476` + the edits | shared | `sdk_/test_embedding_tables.py`, `test_table_runs.py`, `test_replace_run_embeddings.py`, `abc_/test_measure_embedding.py`, `test_measure_embedding_steps.py` | `157 passed, 1 warning in 148.34s` |
| D1, D2 | in place | torch | `measure/test_measure_dino_embedding.py`, `_sizes.py`, `_steps.py`, `test_embedding_inline_golden.py` | `138 passed, 6 skipped, 3 deselected in 337.82s` |
| S1 | in place, `f2ca763f4` + S1 | shared | `cli/test_cli_preflight_ordering.py` | `44 passed in 23.35s` |
| S1 | `a4edf4079` | torch | `cli/test_cli_embed_stage.py` | `72 passed in 17.38s` |
| S1 | `a4edf4079` | torch | `integration/cli/test_local_stage4.py` (crash injection at every durable boundary) | `30 passed in 530.77s` |
| S1 | `a4edf4079` | torch | `integration/cli/test_measure_stage4.py` | `14 passed in 266.21s` |
| S1 | `a4edf4079` | torch | `cli/test_stage4_slurm_worker.py` | `30 passed in 615.41s` |
| S1 | `a4edf4079` | torch | `cli/test_stage4_worker.py` | `50 passed, 1 warning in 519.12s` |
| S1 | `a4edf4079` | torch | `cli/test_staged_measure_stage4.py` | `19 passed in 174.66s` |
| S2 | `a7d162820` | shared | `integration/cli/test_lifecycle_publication_races.py`, `test_staged_gpu_local.py`, `abc_/test_infer_batch_loads_the_model.py`, `ci/test_startup_imports.py`, `cli/test_cli_durable_writes_transport.py`, `test_cli_figures_run_date.py`, `test_cli_preload.py` | `236 passed in 281.14s` |
| S2 | `a7d162820` | shared | `cli/test_cli_reference_e2e.py`, `test_cli_runtime_preload.py`, `test_cli_staged_original_transport.py`, `test_measure_pass_completion.py`, `test_staged_measure_stage4.py`, `test_staged_slurm_scripts.py`, `test_staged_slurm_scripts_stage4.py`, `test_staged_stage1_reads_like_single_pass.py`, `test_staged_stage2_prefix.py` | `142 passed, 4 warnings in 366.02s` |
| S2 | `a7d162820` | torch | `integration/cli/test_embed_stage_refusals.py`, `cli/test_stage4_deferral.py`, `test_stage4_slurm_worker.py` | `53 passed, 1 warning in 783.68s` |
| S3, S4 | `a7d162820` | shared | `integration/cli/test_figures_run_date_cli.py`, `cli/test_array_auxiliary_routing.py`, `test_cli_provenance_original.py`, `test_cli_slurm_array.py`, `test_cli_store_options.py`, `test_gpu_cpu_fallback.py`, `test_gpu_detection_tree_wide.py`, `test_local_strategy_process_only.py`, `test_process_format_cli.py`, `test_staged_routing.py`, `test_cli_preflight_ordering.py` | `195 passed in 145.39s` |
| S3 | `a7d162820` | shared | `cli/test_cli_v2.py` | `105 passed in 119.41s` |
| S3 | `a7d162820` | torch | `integration/cli/test_local_stage4.py` | `30 passed in 659.84s` |
| S3, S4 | `a7d162820` | torch | `integration/cli/test_measure_stage4.py`, `cli/test_cli_embed_stage.py` | `86 passed in 255.97s` |
| S5 | `2c9b887ce` | shared | `cli/test_cli_stage4_signal.py` | `30 passed in 9.22s` |
| S5 | `2c9b887ce` | torch | `cli/test_stage4_worker.py` | `50 passed, 1 warning in 393.88s` |
| S6 | `17e7f7fa1` | shared | `cli/test_staged_resume.py`, `test_staged_controller_stage4.py`, `test_staged_controller_retries.py`, `test_measure_pass_completion.py`, `test_staged_measure_stage4.py`, `test_staged_resume_equivalence.py`, `test_staged_resume_parity.py` | `1656 passed, 1 skipped in 329.84s` |
| S6 | `17e7f7fa1` | torch | `cli/test_stage4_worker.py`, `integration/cli/test_measure_stage4.py`, `cli/test_stage4_slurm_worker.py` | `94 passed, 1 warning in 983.69s` |
| S7 | `ec4e1fbfb` | torch | `cli/test_stage4_slurm_worker.py`, `integration/cli/test_local_stage4.py`, `cli/test_stage4_worker.py` | `110 passed, 1 warning in 1314.16s` |
| S8 | `82b5de23f` | shared | `detect/nn/test_checkpoint_manager.py`, `test_gated_checkpoint_managers.py`, `test_nn_preflight_requirements.py`, `test_helper_imports.py` | `55 passed, 6 skipped in 13.41s` |
| S8 | `82b5de23f` | torch | `measure/test_measure_dino_embedding.py`, `_sizes.py`, `cli/test_stage4_worker.py` | `159 passed, 1 deselected, 1 warning in 553.87s` |
| S9 | `711e6b66b` | shared | `cli/test_cli_stage4_signal.py`, `test_staged_controller.py`, `_retries.py`, `_stage4.py`, `test_measure_pass_completion.py`, `test_staged_slurm_scripts_stage4.py`, `test_cli_embed_stage.py`, `abc_/test_measure_embedding.py`, `_steps.py`, `ci/test_startup_imports.py`, `ci/test_deferred_imports.py` | `549 passed, 1 skipped, 1 warning in 404.13s` |
| S9 | `711e6b66b` | torch | `detect/nn/test_batch_encoder.py`, `test_batch_size.py`, `cli/test_stage4_slurm_worker.py`, `test_stage4_worker.py` | `147 passed, 1 warning in 896.12s` |

No test failed in any of these runs.

**ruff:** `uv run ruff check` over every `src/` file changed since
`1d5b28476` reported `All checks passed!`.

**mypy:** `python -m mypy src/phenotypic` ran in a detached worktree at each
end (`mypy_snapshot.sbatch`):

| Tree | Result |
|---|---|
| `1d5b28476` (before) | `Found 494 errors in 162 files` |
| `711e6b66b` (after, `HEAD`'s `src/`) | `Found 492 errors in 161 files` |

The only per-file difference is in `_cli/_cli_staged_resume.py`, which goes
from 2 errors to 0. These are the two `union-attr` errors on `record.get` that
S6's narrowing removes. No file gained an error.

## Changed `src/` files (for the focused re-run)

- `src/phenotypic/_cli/_cli_embed_stage.py`
- `src/phenotypic/_cli/_cli_execution_strategies.py`
- `src/phenotypic/_cli/_cli_stage4_signal.py`
- `src/phenotypic/_cli/_cli_stage4_worker.py`
- `src/phenotypic/_cli/_cli_staged_controller.py`
- `src/phenotypic/_cli/_cli_staged_resume.py`
- `src/phenotypic/_cli/_cli_staged_slurm_worker.py`
- `src/phenotypic/abc_/_measure_embedding.py`
- `src/phenotypic/detect/nn/_helper/_batch_encoder.py`
- `src/phenotypic/detect/nn/_helper/_checkpoint_manager.py`
- `src/phenotypic/measure/_measure_dino_embedding.py`
- `src/phenotypic/phenotypicCLI.py`
- `src/phenotypic/sdk_/_embedding_tables.py`
- `src/phenotypic/sdk_/_table_runs.py`
