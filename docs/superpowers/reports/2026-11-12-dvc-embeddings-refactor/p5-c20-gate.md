# C20 gate: P5 Tasks 1 and 2, process-mode measurers and process-store tables

Reported by the C20 implementer; written here by the orchestrator. Logs are in
`/bigdata/exfab/anguy344/gate-results/c20/`.

## Design fork: run what fits, skip the rest

Spec E4 as written refused every measurer that needs objects. That would
have broken existing process runs over full pipelines, and measurer-backed
per-image figures in process stores. The implementer measured the
spec-literal version over a 72-file focused set (job `32171020`): **49
existing tests failed**.
- About 40 were process runs whose pipeline carries ordinary measurers.
- About 9 were scoping tests that pin "process mode never runs `meas`".

**User decision (2026-11-18): option B.** Each measurer declares
`requires_objects`. Process mode then:
- runs the embedders with `requires_objects` false (grid embedders);
- skips every other entry with one warning naming each `meas:<key>`;
- refuses only a runnable entry whose output is a flat file (tiff, objmap or
  detect_mat).

The skipped entries are left out of `operations_run_in_mode`, so the
preflight and reference planning walk exactly what runs. Spec E4 and the P5
plan are amended (`2744eab1f`, `f419f26a8`).

## Commits

- **`358d5a4bf`** (T1):
  - `MeasureFeatures.requires_objects` (True by default).
  - `_cli/_cli_process_meas.py`, with `object_measurers_in_process_mode`,
    `skipped_measurers_warning`, `process_meas_refusal`,
    `ProcessMeasRefusedError` and `embed_in_process_mode`.
  - `MODE_SLOTS["process"] = {"ops","meas"}`.
  - The CLI's `_refuse_process_meas_it_would_lose`, in the read-only half.
  - The core refuses the flat-output case before reading the image. Its
    refusal bypasses the per-image scientific wrapper.
  - The embedding write path, which `write_process_only_layer` refuses for a
    flat format.
- **`38cf88330`** (T2): `PROCESS_LAYER_SEMANTICS_REVISION = 5`,
  `test_process_store_tables.py`, and the figures test's revision pin.
- **`b0ce264ed`**: docs. `_cli/CLAUDE.md` gains a section "Process mode runs
  `meas`", and the revision text is updated in both CLAUDE.md files.

## Calls

- **No `measure()` in process mode**, and no `measurement_table=`. Every entry
  `measure()` would run needs objects, and on a preset `GridImage` it would
  inject a grid finder. Process stores therefore carry embedding tables only.
  - **Orchestrator follow-up, `7aaab170d`:** `runs_in_process_mode` is now
    "a `MeasureEmbedding` that needs no objects".
    - Every other entry is named in the one-line warning, grouped by reason
      ("stores no objects" / "stores only embedding tables"), and is out of
      the process walk.
    - Two torch-free tests were red first (job `32171508`: `2 failed`).
    - Shared env over the 72-file set (job `32171507`): `2917 passed,
      38 skipped, 7 xfailed`.
- **`embed_in_process_mode` embeds only the runnable entries.** Otherwise a
  crop embedder would store an empty crops table.
- **Test helpers take a directory.** Their pipeline files are named
  `<stem>-<sha8>.pht-pipe.json`.
- **The torch-free `registered_toy_embedder` fixture** gives the behaviour
  coverage in the shared env.

## Verification

- **Seen red first** (job `32171438`, against HEAD `2744eab1f`):
  - shared env: `25 failed, 33 passed, 5 skipped`;
  - torch env: `28 failed, 35 passed`.
  - The failures were every new test in `test_process_mode_measurers.py` and
    `test_process_store_tables.py`, plus the new preflight-ordering and
    embedding-revision cases.
- **Green** (job `32171313`, over the 72-file focused set):
  - shared env: `2915 passed, 38 skipped, 7 xfailed`;
  - torch venv: `2952 passed, 1 skipped, 7 xfailed`.
  - The tree had been `cmp`-checked against the commits.
  - The set: the new files, every preflight-scope and process-mode test file,
    test_process_objmap_semantics, the importers of the changed modules, the
    DINO tests, the docs-preflight-codes test and the import guards.
- **These skip without torch:**
  - `test_process_accepts_grid_embedding`
  - `test_a_size_measurer_beside_a_dino_grid_embedder_stores_only_the_grid`
  - `test_process_store_has_grid_table_and_stays_consolidated`
  - `test_process_store_stays_byte_reproducible`
  - `test_the_grid_entry_records_the_cpu_device`

  Each has a torch-free toy twin, except the device test.
- **ruff:** clean.
- **mypy:** 59 errors before and after, identical, in the 7 touched files.
  None are in the new module.

## Plan and code mismatches

- `upstream=` and `MeasureEmbedding.requires_objects` already existed at HEAD.
- Line numbers have moved:
  - `_refuse_grid_preset_unlike_cli_grid` is at :751 and called at :3004;
  - `MODE_SLOTS` is at `_cli_preflight.py:278`;
  - `reference_operations_in_scope` is at `_cli_reference.py:254`.
- Not in the plan: the figures test's revision pin, and four docstrings that
  said process mode never runs `meas`.
