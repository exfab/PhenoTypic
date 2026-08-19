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

## Shared files — the parallelism constraint

Two tasks that touch one file cannot run concurrently. These are every such file:

| File | Tasks |
|---|---|
| `sdk_/ngff_.py` | 1.1(create), 1.2, 1.3, 1.4, 1.5, 1.6 |
| `phenotypicCLI.py` | 3.7, 5.3, 5.4, 5.7 |
| `sdk_/_hdf_to_zarr.py` | 5.1(create), 5.2, 5.3, 5.6 |
| `sdk_/_io_constants.py` | 2.1, 5.7, 6.3 |
| `_core/_image_parts/_image_io_handler.py` | 2.2, 2.4, 6.2 |
| `tests/_ngff_conformance.py` | 1.4(create), 2.5(extend) |
| `_core/_image_parts/_grid_image_handler.py` | 2.3, 6.2 |
| `gui/builder/_preview_cache.py` | 2.4, 4.4 |
| `_cli/_cli_sidecar.py`, `tests/unit/cli/test_cli_sidecar.py` | 3.2, 3.5(delete) |
| `_cli/_cli_staged_strategy.py` | 3.5, 3.7 |
| `_cli/_cli_process_single.py` | 3.6, 3.7 |
| `_cli/_cli_completion.py` | 3.8, 5.6 |
| `gui/_shared/tiles.py` | 4.2, 4.3 |
| `gui/builder/_preview_tiles.py` | 4.3, 4.4 |
| `sdk_/_metadata_migration.py` | 5.3, 6.4 |
| `pyproject.toml` | 0.1, 0.2 |
| `.github/workflows/run-pytest{,-full}.yml` | 0.1, 7.2 |

**Consequence: Phase 1 is strictly sequential** — all six tasks append to one
file. So is Phase 5's 5.1 → 5.2/5.3/5.6 chain.

## Cross-phase edges beyond the phase order

| Edge | Why |
|---|---|
| Task 5.1 → Phase 3 Task 3.4 | `staged_store_matches_work_id` |
| Tasks 5.2, 5.6, 5.7 → Phase 3 Task 3.8 | `kind`-tagged marker descriptors |
| Phase 2 Task 2.5 → Phase 1 Task 1.4 | extends `tests/_ngff_conformance.py` |
| Phase 6 → Phase 5 (all) | migration reads the legacy HDF path Phase 6 removes |

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
