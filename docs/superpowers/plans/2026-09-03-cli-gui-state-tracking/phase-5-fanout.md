# Phase 5 — Fan-out: SLURM array and local `--njobs`

**Depends on:** P4, P0 (S-2, S-3). **Blocks:** P6, P7.

**Spec:** §8 (fan-out) — as amended by
[D-A](OPEN-QUESTIONS.md#d-a-per-store-metadata-is-written-at-promote-time-not-backfilled).

**Goal:** `finalize_run`'s aggregation fans out over SLURM array tasks and local `--njobs`,
with `TASK_FINALIZE` as a **reserved trigger entry inside the array task list** — never a
parallel sidecar job.

### What D-A changes from spec §8

§8's array task did two things per image: append to a measurement shard **and** project
metadata into a certified re-promote. The second half is gone — per-store metadata is
written at promote time (P4 Task 2). Shard workers **aggregate only**:

```
array task i ∈ [0, K):                          # aggregate
    for image in shard_i:
        read tables/measurements/table.parquet
        └─ append → measurement_shards/<scheduler_epoch>/shard_i.parquet

array task K (TASK_FINALIZE, dependent):        # reduce
    merge shard_*.parquet → master_measurements.parquet
    join + phantoms + post ops → measurements.{parquet,csv}
    pipeline.json, analysis outputs, per-feature splits
    publish aggregate proof → run proof
```

§8's "ordering and partial failure" narrows to **two** phases, not three: a run that
finishes aggregation has a valid aggregate proof and, once the finalizer publishes, a run
proof. There is no aggregated-not-backfilled state, because there is no backfill.

**The shape already exists.** Recompile has `TASK_MEASUREMENTS` (sharded by `shard_id`,
`_cli_recompile_slurm_scripts.py:146`), `TASK_OVERLAY` (`:339`) and `TASK_FINALIZE`
(`:198`). This phase promotes it to be universal rather than inventing it.

---

## File Structure

| File | Responsibility |
|---|---|
| **Create** `src/phenotypic/_cli/_cli_finalize_fanout.py` | Shard-count sizing, shard worker body, local process-pool driver. ~200 lines. |
| **Modify** `src/phenotypic/_cli/_cli_finalize_run.py` | Accept `shard_paths`; merge instead of concat when supplied. |
| **Modify** `src/phenotypic/_cli/_cli_slurm_array_scripts.py:30` | Add the finalize trigger beside `_CHECKPOINT_SENTINEL` and `_MANIFEST_SENTINEL`. |
| **Modify** `src/phenotypic/sdk_/_io_constants.py` | `measurement_shard_dir(output_dir, scheduler_epoch)`. |
| **Test** `tests/unit/cli/test_finalize_fanout.py` *(new)* | Sizing, epoch namespacing, partial-failure matrix. |
| **Test** `tests/unit/cli/test_array_auxiliary_routing.py` *(new)* | **The rule from `_cli/CLAUDE.md`: no standalone parallel job.** |

---

## Task 1: Shard sizing, counted against `MaxArraySize`

**Files:**
- Create: `src/phenotypic/_cli/_cli_finalize_fanout.py`
- Test: `tests/unit/cli/test_finalize_fanout.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_the_finalize_trigger_is_counted_against_the_array_bound():
    """Project CLAUDE.md: 'Count every trigger entry when sizing chunks against
    MaxArraySize.' A formula that sizes K to the bound and THEN appends the
    finalizer produces an array one index too long, which sbatch rejects at
    submission with a message that names neither the trigger nor the formula."""
    from phenotypic._cli._cli_finalize_fanout import shard_count

    k = shard_count(n_images=1_000_000, seconds_per_image=1.0, max_array_size=2500)
    assert k <= 2499, "K + TASK_FINALIZE must fit inside MaxArraySize"


def test_max_array_size_caps_the_index_not_the_task_count():
    """User's global CLAUDE.md: MaxArraySize (2500 here) caps the INDEX. The highest
    legal index is 2499 -- --array=0-2499 works, --array=1-2500 is rejected."""
    from phenotypic._cli._cli_finalize_fanout import array_spec, shard_count

    k = shard_count(n_images=1_000_000, seconds_per_image=1.0, max_array_size=2500)
    assert array_spec(k) == f"0-{k}"      # K shards + one finalizer index
    assert k < 2500


def test_shard_count_is_one_for_a_small_run():
    from phenotypic._cli._cli_finalize_fanout import shard_count

    assert shard_count(n_images=5, seconds_per_image=0.1, max_array_size=2500) == 1


def test_shards_are_namespaced_by_scheduler_epoch(tmp_path):
    """§7.5: measurement shards are per-invocation scratch, so a prior run's shards
    can never be merged. Recompile already does this
    (recompile/attempts/<attempt_id>/...); the pattern generalises."""
    from phenotypic.sdk_ import measurement_shard_dir

    a = measurement_shard_dir(tmp_path, "epoch-a")
    b = measurement_shard_dir(tmp_path, "epoch-b")
    assert a != b
    assert a.parent == b.parent
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

```python
#: Target wall-clock for one shard task. Chosen so a shard fits comfortably inside
#: `short`'s 2-hour cap with room for scheduler latency and a slow GPFS read, and so
#: a preempted `preempt` task loses at most this much work.
TARGET_TASK_SECONDS = 900


def shard_count(*, n_images: int, seconds_per_image: float, max_array_size: int) -> int:
    """Return K, the number of aggregation shard tasks.

    The ``- 1`` reserves the ``TASK_FINALIZE`` trigger entry's index. Project
    CLAUDE.md requires every trigger entry to be counted when sizing chunks against
    ``MaxArraySize``; the failure mode of not doing so is an sbatch rejection whose
    message names neither the trigger nor the formula.

    ``max_array_size`` caps the *index*, not the task count -- with the cluster's
    2500, the highest legal index is 2499.

    Args:
        n_images: Images to aggregate.
        seconds_per_image: Measured by spike S-2 against a real tree on GPFS.
        max_array_size: ``scontrol show config``'s ``MaxArraySize``, or
            ``MaxSubmitJobs`` when it is lower.

    Returns:
        K in ``[1, max_array_size - 1]``.
    """
    import math

    target = math.ceil(n_images * seconds_per_image / TARGET_TASK_SECONDS)
    return max(1, min(target, max_array_size - 1))
```

Use the `seconds_per_image` **S-2 measured**, recorded in `spikes/RESULTS.md`. Cite the
number and the fixture in the constant's docstring — a magic number nobody can trace back
to a measurement is a guess with better formatting.

- [ ] **Step 4: Run the tests.** Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_finalize_fanout.py \
        src/phenotypic/sdk_/_io_constants.py tests/unit/cli/test_finalize_fanout.py
git commit -m "feat(cli): shard sizing for aggregation fan-out

Spec §8. K reserves the TASK_FINALIZE index; seconds_per_image comes from S-2 on a
real GPFS tree, cited in the constant's docstring."
```

---

## Task 2: The shard worker and the reserved finalize trigger

**Files:**
- Modify: `src/phenotypic/_cli/_cli_finalize_fanout.py`
- Modify: `src/phenotypic/_cli/_cli_slurm_array_scripts.py:30`
- Test: `tests/unit/cli/test_array_auxiliary_routing.py`

**Read `src/phenotypic/_cli/CLAUDE.md`'s array-auxiliary-work contract before starting.**
It is the rule this task is most likely to break, and root `AGENTS.md` is a symlink to the
project `CLAUDE.md`, so it carries the same rule.

- [ ] **Step 1: Write the routing test first — it is the contract**

```python
"""No standalone parallel job beside an active ordinary array.

Project CLAUDE.md and _cli/CLAUDE.md: allocation and submission bounds are already
consumed by the array cohort. Ancillary work routes through reserved trigger entries
INSIDE the array task list, following the existing __PHENOTYPIC_CHECKPOINT__ and
__PHENOTYPIC_MANIFEST__ dispatch pattern (_cli_slurm_array_scripts.py:30-32).

A terminal `afterany` finalizer is NOT a parallel sidecar and is allowed.
"""


def test_finalization_submits_no_job_beside_the_array(tmp_path, fake_sbatch):
    from phenotypic._cli._cli_finalize_fanout import submit_aggregation

    submit_aggregation(tmp_path, dataset_names=["plate"], n_images=500)

    submissions = fake_sbatch.calls
    arrays = [c for c in submissions if "--array" in c.argv]
    assert len(arrays) == 1, f"expected exactly one array submission, got {submissions}"
    siblings = [c for c in submissions if c is not arrays[0] and not c.is_afterany]
    assert not siblings, f"a parallel sidecar job was submitted: {siblings}"


def test_the_finalize_entry_lives_inside_the_array_task_list(tmp_path):
    from phenotypic._cli._cli_finalize_fanout import build_task_list

    tasks = build_task_list(tmp_path, dataset_names=["plate"], n_images=500)
    assert tasks[-1]["task_type"] == "finalize"
    assert all(t["task_type"] == "measurements" for t in tasks[:-1])
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

Reuse recompile's task-type vocabulary verbatim — `TASK_MEASUREMENTS`, `TASK_FINALIZE`
(`_cli_recompile_slurm_scripts.py:51-53`) — rather than minting new names. Two vocabularies
for one dispatch pattern is the cardinality problem this whole change is about.

The shard worker body is one pass over its images: read
`tables/measurements/table.parquet`, append to
`measurement_shards/<scheduler_epoch>/shard_i.parquet`. Nothing else — no store write, no
metadata projection (D-A), no global frame.

`TASK_FINALIZE` calls `finalize_run(..., shard_paths=[...])`.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli
git commit -m "feat(cli): shard worker plus the reserved TASK_FINALIZE trigger entry

Spec §8 and _cli/CLAUDE.md's array-auxiliary contract. Reuses recompile's task-type
vocabulary rather than minting a second one."
```

---

## Task 3: Local `--njobs` uses the same decomposition

**Files:**
- Modify: `src/phenotypic/_cli/_cli_finalize_fanout.py`
- Test: `tests/unit/cli/test_finalize_fanout.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("njobs", [1, 2, 8])
def test_local_fanout_produces_a_byte_identical_master(tmp_path, njobs):
    """§8: 'Local --njobs uses the same decomposition with a process pool.' Same
    decomposition means same answer -- if the merge order can change the master's
    bytes, two runs of the same data disagree and the aggregate proof's
    source_set_digest becomes meaningless."""
    output = _run_and_finalize(tmp_path / str(njobs), njobs=njobs)
    assert _master_bytes(output) == _master_bytes(_run_and_finalize(tmp_path / "ref", njobs=1))
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL on shard-order nondeterminism.

- [ ] **Step 3: Implement**

Same `shard_count`, same shard worker, `concurrent.futures.ProcessPoolExecutor` instead of
an array. **Merge shards in sorted `shard_id` order**, and assign images to shards
deterministically by sorted `work_id` — otherwise the master's row order depends on
scheduling, and a re-run of identical inputs produces different bytes.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli
git commit -m "feat(cli): local --njobs aggregation via the same decomposition

Spec §8. Deterministic shard assignment and merge order, so the master is
byte-identical across njobs -- otherwise source_set_digest certifies nothing."
```

---

## Task 4: The partial-failure matrix — the phase gate

**Files:**
- Test: `tests/unit/cli/test_finalize_fanout.py`

Spec §14's second named test. §8's partial-failure story narrows to two phases under D-A,
so the matrix is smaller than the spec's — and that is the point.

- [ ] **Step 1: Write the matrix**

```python
@pytest.mark.parametrize(
    "kill_after,expected_completion,expected_remaining",
    [
        ("nothing",            "complete",   set()),
        ("some_images",        "incomplete", {"measure"}),
        ("all_images",         "incomplete", {"aggregate"}),
        ("some_shards",        "incomplete", {"aggregate"}),
        ("all_shards",         "incomplete", {"finalize"}),
        ("master_written",     "incomplete", {"finalize"}),
    ],
)
def test_a_run_killed_mid_finalization_resumes_only_the_missing_phase(
    tmp_path, kill_after, expected_completion, expected_remaining
):
    """Spec §14's partial-failure matrix, narrowed by D-A to two phases.

    The aggregate proof asserts master + mirror; the run proof asserts everything.
    A run that finishes aggregation and dies before the run proof has a valid
    aggregate proof and no run proof -- a resumable state.
    """
    from phenotypic.sdk_ import resolve_run_state

    output = _run_until(tmp_path, kill_after)
    state = resolve_run_state(output, depth="deep")
    assert state.completion == expected_completion
    assert _remaining_phases(state) == expected_remaining


def test_a_prior_epochs_shards_are_never_merged(tmp_path):
    """§7.5: shards are per-invocation scratch, namespaced by scheduler_epoch. A
    prior run's shards being merged is the stale-cache hazard §7.5 exists to
    forbid, arriving through the fan-out instead of through the aggregator."""
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import master_measurements_parquet_path, measurement_shard_dir

    _publish_two_successful_images(tmp_path)
    stale = measurement_shard_dir(tmp_path, "old-epoch")
    stale.mkdir(parents=True)
    pl.DataFrame({"Metadata_ImageFile": ["GHOST.tif"]}).write_parquet(
        stale / "shard_0000.parquet"
    )

    finalize_run(tmp_path, dataset_names=["plate"])
    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    assert "GHOST.tif" not in master["Metadata_ImageFile"].to_list()
```

- [ ] **Step 2: Run to verify failure, then implement until green.**

- [ ] **Step 3: Apply S-3's merge verdict**

If `spikes/RESULTS.md` says `S-3 STREAMING`, `TASK_FINALIZE` uses
`pl.scan_parquet([...]).sink_parquet(...)`. If it says `S-3 IN-MEMORY`, it uses
`pl.concat` and the finalizer's `--mem` is set to 2 × the projected peak RSS S-3 measured.
**Cite the measured number** wherever `--mem` is set.

- [ ] **Step 4: Phase gate — a real SLURM run**

The unit tests use a fake scheduler. That is necessary and not sufficient: the failure
modes this phase can produce are all scheduler-shaped. Submit one real fan-out on the
fixture tree via the **`slurm-job`** skill, and verify:

```bash
# submission ≠ start
scontrol show job <id> | grep -E 'StartTime|Reason'
# exactly one array, one dependent finalizer, no sidecar
sacct -j <id> --format=JobID,JobName,State,ExitCode
```

Confirm: exactly one array cohort plus one `afterany` finalizer; no third job; the master
matches a local `--njobs 1` run byte for byte.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli
git commit -m "test(cli): partial-failure matrix for fan-out aggregation

Spec §14, narrowed to two phases by D-A. Verified against a real SLURM submission:
one array cohort, one dependent finalizer, no sidecar, master byte-identical to the
local path."
```

---

## Task 5: The rolling-input matrix

**Files:**
- Test: `tests/integration/test_rolling_input.py` *(new)*

Spec §14's third named test, and the scenario the whole design is shaped around — the
audit's running example is a 6,000-image run that grows. It spans identity (P2),
per-image proofs (P3), `finalize_run` (P4) and fan-out (P5), which is why it lands here
rather than earlier.

**The property under test:** per-image proofs survive an arrival; only aggregate-level
proofs invalidate.

- [ ] **Step 1: Write the matrix**

```python
@pytest.mark.parametrize(
    "scenario",
    [
        "batch_added_mid_run",
        "batch_added_between_runs",
        "metadata_arrives_later",
        "unready_file_present",
    ],
)
def test_only_aggregate_proofs_invalidate_when_the_input_grows(tmp_path, scenario):
    """Spec §14's rolling-input matrix.

    §9.2: adding 10 images to a 6,000-image run today re-derives the worklist by
    validating 6,000 markers, each re-hashing its measurements parquet and overlay
    PNG. After this change the 6,000 unchanged images cost one stat() each and the
    10 arrivals are deep-verified.

    D7 is the identity half of the same property: a new image changes
    inventory_digest but NOT processing_generation, so live progress is not reset
    and in-flight workers are not fenced.
    """
    from phenotypic.sdk_ import resolve_run_state

    output = _run_to_completion(tmp_path, n_images=6)
    before = {w: s.stages for w, s in resolve_run_state(output, depth="deep").images.items()}

    _apply(scenario, output)
    _run_again(output)

    after = resolve_run_state(output, depth="deep")
    for work_id, stages in before.items():
        assert after.images[work_id].stages == stages, (
            f"{scenario} invalidated an existing image's proof; only aggregate-level "
            "proofs may invalidate when scope changes"
        )
    assert after.completion == "complete"


def test_an_unready_file_is_not_accepted_into_the_inventory(tmp_path):
    """A file still being written must not enter work_ids -- once accepted, its
    absence of a proof reads as `incomplete` forever."""
    from phenotypic.sdk_ import resolve_run_state

    output = _run_to_completion(tmp_path, n_images=6)
    _write_partial_image(output_input_dir(output) / "still-copying.tif")
    _run_again(output)
    assert resolve_run_state(output, depth="deep").completion == "complete"


def test_metadata_arriving_later_re_runs_finalize_and_nothing_else(tmp_path):
    """§7.4, as narrowed by D-A. A metadata edit changes finalization_input_digest,
    so the next invocation re-joins the mirror. Stores keep the snapshot they were
    built against and report the divergence as an advisory (P1 Task 5)."""
    import polars as pl

    from phenotypic.sdk_ import measurements_parquet_path, resolve_run_state

    output = _run_to_completion(tmp_path, n_images=6, metadata=False)
    store_mtimes = _store_mtimes(output)

    _add_metadata_csv(output)
    _run_again(output)

    mirror = pl.read_parquet(measurements_parquet_path(output))
    assert "Metadata_Strain" in mirror.columns
    assert _store_mtimes(output) == store_mtimes, "a metadata edit rewrote a store"
    assert any("metadata" in a for a in resolve_run_state(output, depth="deep").advisories)
```

- [ ] **Step 2: Run it against a real local run.** Expected: PASS.

- [ ] **Step 3: Prove the key assertion can fail**

Fold `inventory_digest` into the generation digest (undoing D7); confirm
`test_only_aggregate_proofs_invalidate_when_the_input_grows[batch_added_between_runs]`
fails because every existing proof was invalidated. Restore. **D7 is the decision this
test defends, and an undefended decision drifts.**

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_rolling_input.py
git commit -m "test: rolling-input matrix -- arrivals invalidate scope, not proofs

Spec §14, §9.2, D7. Confirmed to fail when inventory_digest is folded into the
generation digest, which is the mistake D7 exists to prevent."
```
