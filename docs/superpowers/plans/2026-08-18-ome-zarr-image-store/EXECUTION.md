# Execution plan — cluster-and-isolate

> Derived from each task's `Files:` / `Interfaces:` blocks by
> `refinery/build_dag.py`. This is a **version-controlled view**, not a separate
> tool: regenerate it when a task's `Files:` block changes.

**Baseline commit:** `5221aa1e` (plan + spec + refinery, nothing implemented).
**Starting state, verified:** `requires-python = ">=3.10, <3.13"`, `zarr` not
installed, `import phenotypic` clean. Phase 0 Task 0.1 is what moves the floor.

### Known-failing baseline test — NOT caused by this work

`tests/unit/cli/test_cli_terminal_failures.py::test_concurrent_process_appends_do_not_lose_records`
fails at the baseline. Recorded here so no later phase misattributes it.

```
946 passed, 1 failed, 32 skipped   (tests/unit, at 5221aa1e)
```

It is **environment-induced, not a defect**: the test spawns 8 processes and
joins each with a 20 s timeout, and `assert process.exitcode == 0` fails with
`None` — i.e. the child had not exited yet, not that it crashed. This
allocation has **4 cores** (`len(os.sched_getaffinity(0))`), so eight spawned
CPython interpreters starve. It is also pre-existing by construction: the only
commit at this point is documentation.

**Two rules follow for every implementation cluster:**

1. Report the suite as "green except this one", never as fully green — and
   re-check that it is still *this* test failing for *this* reason.
2. **Never use `pytest -n auto`.** `nproc` reports the node's cores, not the
   allocation's, so `-n auto` oversubscribes 4 real cores many times over and
   manufactures exactly this class of timeout failure. Pass an explicit `-n 4`
   or omit `-n`.

---

## Phase 8 — Review-fix cluster

**Status:** Complete 2026-08-24. The review fixes preserve full-pyramid
`full_layers=True` snapshots, make store promotion/cleanup attempt-owned and retry-safe,
encode repeated-2x NGFF sampling transforms, and preserve explicit loader overrides.
Every task and the final Windows-contention fix passed independent review.

Final verification used Slurm array `27732638`, with corrected array-environment reruns
`27732832_1` (CLI/analysis/schema) and `27732847_0` (core/SDK). Across the six disjoint
groups: **9,439 passed, 4 failed, 190 skipped, 579 deselected**. All four failures are the
recorded pre-existing baseline, not Phase 8 regressions:

- `test_inspect_remeasures_when_explicit_image_changes`;
- the three `FilFinderDetector` smoke tests that require the absent `topology` extra.

The geometry logic-validation script and explicit changed-path Ruff gate passed. Final
mypy was **412 errors in 121 files**, improving on the measured merge-base baseline of
419 errors in 123 files. Native Windows remains a CI surface: the sharing-violation
regression is deterministic on POSIX, while the real extended-path/UNC lane was not run
locally.

For future wide runs, `run_unit_suite.sbatch` now discovers the exact supported test-file
scope and round-robins 658 files across a 24-task array (27–28 files per task), with
array variables removed before pytest so scheduler identity cannot contaminate controller
tests. The committed runner passed `bash -n`, `git diff --check`, and Slurm `--test-only`.

---

## Shared files — the parallelism constraint

Two tasks that **write** one file cannot run concurrently. Derived, not assumed:

| File | Tasks that WRITE it |
|---|---|
| `src/phenotypic/sdk_/ngff_.py` | 1.1(create), 1.2(modify), 1.3(modify), 1.4(modify), 1.5(modify), 1.6(modify) |
| `src/phenotypic/phenotypicCLI.py` | 3.7(modify), 3.8(modify), 5.3(modify), 5.4(modify), 5.7(modify) |
| `src/phenotypic/_cli/_cli_process_single.py` | 3.6(modify), 3.7(modify), 3.8(modify), 6.2(modify) |
| `src/phenotypic/_cli/_cli_staged_workers.py` | 3.3(test), 3.3(modify), 3.7(modify), 6.2(modify) |
| `src/phenotypic/sdk_/_hdf_to_zarr.py` | 5.1(create), 5.2(modify), 5.3(modify), 5.6(modify) |
| `pyproject.toml` | 0.1(modify), 0.2(modify), 2.4(test) |
| `src/phenotypic/_cli/_cli_output_manager.py` | 3.1(modify), 3.7(modify), 6.2(modify) |
| `src/phenotypic/_cli/_cli_staged_slurm_worker.py` | 3.5(modify), 3.7(modify), 3.8(modify) |
| `src/phenotypic/_core/_image_parts/_image_io_handler.py` | 2.2(modify), 2.4(modify), 6.2(modify) |
| `src/phenotypic/sdk_/_io_constants.py` | 2.1(modify), 5.7(modify), 6.3(modify) |
| `tests/integration/cli/test_staged_gpu_local.py` | 3.1(test), 3.3(test), 3.5(test) |
| `.github/workflows/run-pytest-full.yml` | 0.1(modify), 7.2(modify) |
| `.github/workflows/run-pytest.yml` | 0.1(modify), 7.2(modify) |
| `src/phenotypic/_cli/_cli_completion.py` | 3.8(modify), 5.6(modify) |
| `src/phenotypic/_cli/_cli_execution_strategies.py` | 3.6(modify), 3.8(modify) |
| `src/phenotypic/_cli/_cli_sidecar.py` | 3.2(test), 3.5(delete) |
| `src/phenotypic/_core/_image_parts/_grid_image_handler.py` | 2.3(modify), 6.2(modify) |
| `src/phenotypic/gui/_shared/tiles.py` | 4.2(modify), 4.3(modify) |
| `src/phenotypic/gui/builder/_preview_cache.py` | 2.4(modify), 4.4(modify) |
| `src/phenotypic/gui/builder/_preview_tiles.py` | 4.3(modify), 4.4(modify) |
| `src/phenotypic/sdk_/_metadata_migration.py` | 5.3(modify), 6.4(modify) |
| `tests/_ngff_conformance.py` | 1.4(create), 2.5(create) |
| `tests/gui/builder/test_preview_cache.py` | 2.4(test), 4.4(test) |
| `tests/gui/builder/test_preview_compute_scope.py` | 2.4(test), 4.4(test) |
| `tests/gui/builder/test_preview_tile_blueprint.py` | 2.4(test), 4.4(test) |
| `tests/unit/cli/test_cli_sidecar.py` | 3.2(test), 3.5(delete) |

**Read-only sharing does not conflict** and is excluded above — only writers collide.

> **This table replaces an earlier one that was wrong in a way that mattered.** The first
> generator resolved its repo root to `docs/` rather than the worktree, so its filename index
> was empty and every bare filename fell back to itself. The plan mixes
> `` `src/phenotypic/_cli/_cli_process_single.py` `` with a bare
> `` `_cli_execution_strategies.py` `` **on the same line**, and writes line references as
> `` `phenotypicCLI.py:400` `` — so one file appeared under two keys and its conflict
> vanished, while `:400`-suffixed paths were not matched at all. `_cli_process_single.py`
> showed 2 writers; it has **4**. `phenotypicCLI.py` showed 4; it has **5**. Blockquote lines
> are also excluded now, because correction notes name files precisely to say a task must
> *not* touch them — Task 3.7's note names `_cli_staged_strategy.py` for exactly that reason.

## Cross-phase write overlap

| Phase pair | Shared writes |
|---|---|
| 0 ↔ 2 | `pyproject.toml` |
| 0 ↔ 7 | `.github/workflows/run-pytest-full.yml`<br>`.github/workflows/run-pytest.yml` |
| 1 ↔ 2 | `tests/_ngff_conformance.py` |
| 2 ↔ 4 | `src/phenotypic/gui/builder/_preview_cache.py`<br>`tests/gui/builder/test_preview_cache.py`<br>`tests/gui/builder/test_preview_compute_scope.py`<br>`tests/gui/builder/test_preview_tile_blueprint.py` |
| 2 ↔ 5 | `src/phenotypic/sdk_/_io_constants.py` |
| 2 ↔ 6 | `src/phenotypic/_core/_image_parts/_grid_image_handler.py`<br>`src/phenotypic/_core/_image_parts/_image_io_handler.py`<br>`src/phenotypic/sdk_/_io_constants.py` |
| 3 ↔ 5 | `src/phenotypic/_cli/_cli_completion.py`<br>`src/phenotypic/phenotypicCLI.py` |
| 3 ↔ 6 | `src/phenotypic/_cli/_cli_output_manager.py`<br>`src/phenotypic/_cli/_cli_process_single.py`<br>`src/phenotypic/_cli/_cli_staged_workers.py` |
| 5 ↔ 6 | `src/phenotypic/sdk_/_io_constants.py`<br>`src/phenotypic/sdk_/_metadata_migration.py` |

**Phase pairs with NO shared writes:** 0/1, 0/3, 0/4, 0/5, 0/6, 1/3, 1/4, 1/5, 1/6, 1/7, 2/3, 2/7, 3/4, 3/7, 4/5, 4/6, 4/7, 5/7, 6/7

The one that matters: **Phase 4 ↔ Phase 5 share nothing**, so the GUI and migration clusters
may run in parallel worktrees once Phase 3 completes. Verified rather than assumed — Task 5.7
does write a GUI file (`gui/results_viewer/_output_consistency.py`), but not one Phase 4 opens.

Phase 3 ↔ Phase 5 share `phenotypicCLI.py` and `_cli_completion.py`, and Phase 3 ↔ Phase 6
share three `_cli` modules — both already forbidden by the declared phase order, but worth
seeing rather than trusting.

---

## Clusters

Shape tags: **K**eystone (novel interdependent core), **S**weep (broad shallow
mechanical), **Se**am (one risky wiring point), **L**eaf (folded in).

| # | Tasks | Shape | Cluster |
|---|---|---|---|
| C1 | 0.1, 0.2 | K | Dependency floor + vendored schemas |
| C2 | 1.1, 1.2, 1.3 | K | Geometry, chunk/shard policy, `attributes.phenotypic` |
| C3 | 1.4 | K+Se | OME projection — **isolated** |
| C4 | 1.5, 1.6 | K | Promote primitive, durability, sweep, `valid_staged_store` |
| C5 | 2.1, 2.2, 2.3 | K+L | Store locator, `save2zarr`/`load_zarr`, grid state |
| C6 | 2.4 | Se | `save_intermediate_zarr` — GUI preview boundary |
| C7 | 2.5 | K | NGFF conformance harness |
| C8 | 3.1, 3.2 | K | `save_image_store`, Stage-2 token |
| C9 | 3.3, 3.4 | K | Three staged workers + resume classifier |
| C10 | 3.5 | S+Se | Staged wiring; delete `_cli_sidecar` — 8 files |
| C11 | 3.6 | S | Peripheral call sites — 7 files, mechanical |
| C12 | 3.7 | Se | `--durable-writes` |
| C13 | 3.8 | Se | Completion markers describe a store |
| C14 | 4.1, 4.2 | K | Store discovery + pyramid-level tile reads |
| C15 | 4.3, 4.4 | Se | Staleness traps + builder preview tiles |
| C16 | 5.1, 5.2 | K | Conversion core + canonical metadata view |
| C17 | 5.3, 5.4 | Se | `--mode migrate` wiring; drop the recompile fan-out |
| C18 | 5.6, 5.7 | Se | Run-state republication + the migration predicate |
| C19 | 6.1, 6.2, 6.3 | S | HDF removal sweep |
| C20 | 6.4 | S | Docs, CLAUDE.md, supersessions |
| C21 | 7.1, 7.2 | K | Commit-protocol e2e + Windows lane |
| C22 | 7.3, 7.4 | K | Invariant gates + full-suite sign-off |

### Why C3 is alone
Task 1.4 is the most-edited block in the plan — it absorbed ALGO-1's fatal-XML
ruling, the `_xml_text` control-character sanitizer, the `str(enum)` namespace
bug, and it bootstraps `tests/_ngff_conformance.py`. Eight ledger entries touch
it. It gets its own diff and its own gate.

### Why C10 and C11 are separate
Both are Phase 3 breadth, but C10 deletes a module and rewires six call sites
(risk), while C11 is a flat rename sweep across seven peripheral files
(mechanical). Merging them would hide the risky half inside a large diff.

### Why C19 is one cluster
Deletion sweeps are only safe when the whole sweep lands together — a partial
one leaves dangling imports. Its gate is a frontier verify pass, not a
mid-tier self-check.

---

## Parallelism

Everything runs **sequentially** except one window: after C13 (Phase 3
complete), **C14–C15 (GUI) and C16–C18 (migration) share no files** and may run
in parallel worktrees. Note 5.7 does touch
`gui/results_viewer/_output_consistency.py`, which Phase 4 does not — verified,
not assumed.

Nothing else parallelizes: within every other phase the clusters share a file.

## Gates

- **Pre-dispatch:** `plan-reviewer` over the plan (running).
- **Per cluster:** read the diff, run the cluster's tests + `ruff`/`mypy` on
  changed paths, before the next cluster starts.
- **Per phase:** `implementation-test-reviewer` over the phase's combined diff —
  it checks the tests can actually *fail*, which matters here because two
  rounds of review found mutation proofs that passed with the defect in place.
- **End:** one simplify pass, then the regression suite for affected areas.

Model: frontier/high for every K and Se cluster and every gate; mid-tier/medium
for C11 and C20 only. **Never review with a weaker model than implemented.**
