# P4 review fixes: C19r and C19q

Findings are those of `review-p4.md`, written here by the orchestrator.

The C19r implementer's final report was lost when the session restarted. This
section is therefore rebuilt from its commit messages, its tests and its gate
artifacts, and states nothing those don't. The C19q report arrived and is
summarised in its own section.

## C19r: the findings that needed no user decision

| Commit | Findings | Change |
|---|---|---|
| `46e30e266` | M2, I3 | One stale predicate, `stale is not False`. Embedding files are contained lexically, so a symlinked table works. |
| `c68c70090` | I1, I3, M1, M3, M8 | See below. |
| `f6939d849` | M7 | An embedder's `MemoryError` or CUDA OOM propagates unwrapped. |
| `bd1e09990` | M4, M5, M9 | `inference_mode` for dense features. Determinism is documented per device; per-process backbone memory is documented. |
| `67bc6169d` | T1–T8 | Test fixes, plus pins for I1, I3, M1, M2, M7 and M8. |

What `c68c70090` changed:

- **I1.** `stage_embedding_deliverables` reads every non-stale entry before any
  deliverable is written (present, sha256, type, one schema per key) and
  streams each key to a hidden temporary file. `promote` runs after the
  splits, with the manifest last, so a failure changes no file under
  `deliverables/`.
- **I3.** `_store_identity` uses unresolved paths and checks with `samefile`.
- **M1a.** The manifest and each published table are declared in the aggregate
  proof's `required_outputs`.
- **M1b.** On the `shard_paths` branch, only sources the shards merged
  contribute embeddings.
- **M3.** The README no longer promises that stale stores are listed.
- **M8.** Cleanup deletes only the files finalization wrote.

Test changes in `67bc6169d`:

- **T2, T3.** The stub's CLS is the maximum of the patch tokens, not their
  mean, and a 4-register variant was added.
- **T1.** The three `rev-p4` mutants each fail one test, per the commit message.
  That was not re-run by the orchestrator.
- **T4.** Plate 2 is mirrored, and each store's rows equal its own table.
- **T6.** A two-key corrupt recompile changes no deliverable byte or mtime.
- **T7.** Byte stability is checked across two subprocesses.
- **T8.** New tests cover a grid embedder, `--restart`, and continuation
  followed by re-finalization.
- **The I1 re-run.** `test_a_rerun_after_a_failed_finalization_does_not_hide_it`
  pins it:
  - a failed finalization changes nothing;
  - a plain re-run then finds nothing to do and leaves the last good
    publication standing, which still verifies;
  - the remedy, `--mode measure`, repairs the store.

### Gate `p4r`: array `32097860` and foundation job `32097861`, at `67bc6169d`

Both started and ended at `dirty=0` and imported `phenotypic` from the gate
tree.

- **Shards:** 11,077 tests, of which 10,964 passed, 15 failed, 0 errors and 98
  were skipped. The orchestrator ran `classify.py`.
- **13 of the reds** are the GUI expected-red rows, the same names as `p4`.
- **2 reds are new**, both in `tests/integration/gui/test_run_console_callbacks.py`:
  - `test_run_app_owns_one_startable_observer_lifecycle` passed 3 of 3 in
    isolation (job `32166897`). It is order-dependent and already known.
  - `test_immediate_local_exit_terminalizes_allocated_generation` failed 2 of
    3 in isolation. A same-node comparison (job `32167190`), 5 runs on each
    tree, alternating:
    - the last pushed commit `206ba673b` gave 1 fail, then 4 passes;
    - HEAD gave the same: 1 fail, then 4 passes.
    - Each failure was the first, cold run (14–18 s against a 10 s deadline).
      Warm runs took 2.5 s.
    - `python -m phenotypic --help` took 0.24 s on both trees.
    - **Verdict:** a cold-start timing flake in the test, not a P4
      regression.
- **Foundation job:** 582 tests, 0 failed, 7 skipped.

## C19q: the user's answers to Q1 and Q2

- **`377165eae`** (Q1, and the Q2 error):
  - Every row gets an int16, non-null `descriptor_index` column, placed before
    `embedding`.
  - The manifest's `descriptors` are listed in first-appearance order over the
    sorted stores.
  - `_warn_on_mixed_descriptors` logs one warning per key, naming each
    descriptor (revision, pooling, params digest) with its stores. It refuses
    past the int16 limit.
  - `schema.json` and the README cover the column.
  - Preflight adds `PF-EMBED-REVISION-UNPINNED` (warning only, mode-scoped),
    with its HINTS entry and docs row.
  - Staging collects every unreadable non-stale store, up to 10 named. The
    error ends with the whole-tree `--mode measure` remedy and says it
    re-embeds every image.
- **`7ea89384a`** (Q2, the waiter):
  - When every image is resolved, exit 5 says the finalizer failed or never
    ran, and points at `<out>/.phenotypic/logs/slurm`, through a new
    `slurm_logs_dir` helper.
  - A cancelled submission gets its own line.
  - This is a message change only.
- **`d86054639`** (the follow-up, `model.device`):
  - With `device="auto"`, a GPU node and a CPU node had given the same
    descriptor although their vectors differ.
  - `entry_descriptor()` now records the device that ran (`"cuda"` or
    `"cpu"`, from the same `resolve_backbone_device` call that loads the
    backbone).
  - The device flows into the manifest descriptors and the warning summary,
    and the advice says to set the device explicitly.
  - Spec D3's model row now lists it.
  - Seen red first: `test_an_auto_embedder_records_the_device_that_ran[False/True]`,
    `test_stores_differing_only_in_device_get_two_descriptors`,
    `test_the_readme_names_the_device_as_a_cause`, and the updated payload
    test.
  - At `d86054639`, the P4 files plus the preflight, README and schema tests:
    - torch, job `32167544`: `219 passed`;
    - shared, job `32167543`: `172 passed, 47 skipped`.
  - ruff clean; mypy reports no issues in the 5 touched files.
- **Seen red first:**
  - in `test_embedding_aggregation.py`, the two-descriptor, single-descriptor,
    schema, README and every-store-named tests;
  - the new `test_preflight_embedding_revision.py`, with an ImportError;
  - the two waiter tests.

  `test_a_mixed_descriptor_deliverable_is_byte_stable` passed before the fix.
  It is a regression guard.
- **Focused runs at `7ea89384a`:**
  - torch, job `32167345`: `323 passed`;
  - shared, job `32167344`: `1 failed, 1634 passed, 48 skipped, 7 xfailed`.
    The one failure is `test_validation::test_tuning_spec_path_is_deployable`,
    a GUI expected-red row: the P0 suffix flip, #283.
- **ruff:** clean.
- **mypy:** no new errors. Six pre-existing errors remain in
  `_cli_preflight.py:982-993`.
- **Calls:**
  - the summary format;
  - the warning is logged before staging;
  - every unreadable store is named;
  - every unpinned DINO pipeline now gets the preflight warning, test stubs
    included.
- **Left to the simplify pass:** eight hand-joined `logs_dir(x) / "slurm"`
  sites.
