# Phase 2 — Identity schema: fourteen tokens to six

**Depends on:** P1. **Blocks:** P3–P7.

**Spec:** §5 (identity schema), D3, D4, D5, D6, D7 — as amended by
[D-C](OPEN-QUESTIONS.md#d-c-scientific_config_digest-is-the-existing-digest-verbatim-was-q1)
and [O-1](OPEN-QUESTIONS.md#o-1-scheduler_epoch-may-be-five-names-collapsing-to-one-owner-not-five-tokens-to-one).

**Goal:** `processing_generation` stops being a `uuid4().hex` and becomes
`sha256(pipeline_sha256 ‖ scientific_config_digest ‖ restart_epoch)`; `restart_epoch`
becomes the one tracked counter the design admits to adding; `slurm_generation` and
`lifecycle_epoch` collapse into `scheduler_epoch` where a single writer already owns the
lifetime.

**Why content-derived matters (D3):** same inputs → same token, so resume and fencing
become **emergent** rather than bookkeeping. Two invocations with the same configuration
mint the same generation without either having read the other's state — which is what lets
a SLURM worker starting cold fence itself correctly against a run it has never seen.

---

## File Structure

| File | Responsibility |
|---|---|
| **Modify** `src/phenotypic/_cli/_cli_identity.py` *(new)* | `mint_run_identity(config, *, restart)`, `read_restart_epoch`, `bump_restart_epoch`. **CLI-side, because they write.** ~110 lines. |
| **Modify** `src/phenotypic/_cli/_cli_state_management.py:237` | `processing_generation` becomes content-derived; `restart_epoch` enters `config`. |
| **Modify** `src/phenotypic/_cli/_cli_migrate.py:660-705` | **The third minting site (CAN-7).** See Task 3 Step 4 — this one is already wrong today. |
| **Modify** `src/phenotypic/phenotypicCLI.py:2640,2716` | **The fourth (CAN-20):** measure mode skips state creation and sets `processing_generation = uuid4().hex` on its own branch. |
| **Modify** `src/phenotypic/sdk_/_io_constants.py:1081` | `clear_machine_state` **preserves** `restart_epoch.json`. |
| **Modify** `src/phenotypic/_cli/_cli_slurm_lifecycle.py:78` | `slurm_generation` → `scheduler_epoch`, one writer. |
| **Modify** `src/phenotypic/_cli/_cli_completion.py:163` | `publish_image_success` takes `scheduler_epoch`, not `lifecycle_epoch`. |
| **Modify** `docs/superpowers/specs/2026-09-03-cli-gui-state-tracking/design.md` §5.3–§5.4 | Correct the field list per D-C; add §5.3's redundancy footnote; **record U-4 cutting `publication_id`**. |
| **Test** `tests/unit/cli/test_run_identity.py` *(new)* | Determinism, restart fencing, mint-once, stale-worker. |

> **"Every resume path uses the same mint" was written as one line and turned out to name
> four writers.** `create_initial_state` (`_cli_state_management.py:237`), the resume path,
> the HDF migrator (`_cli_migrate.py:686`), and measure mode (`phenotypicCLI.py:2716`).
> Round 1 found the last two; both are already minting generations this design forbids.
> Enumerate them here so the fifth is a diff a reviewer can see.

---

## Interfaces

**Produces:**

```python
# phenotypic._cli._cli_identity

def mint_run_identity(config: "ExecutionConfig", *, restart: bool) -> RunIdentity:
    """Mint the identity for a new or resumed invocation. **Writer.**"""

def read_restart_epoch(output_dir: Path) -> int:
    """Return the run's restart epoch, or 0 when absent. Never raises."""

def bump_restart_epoch(output_dir: Path) -> int:
    """Increment and persist the restart epoch. Returns the new value."""

def scientific_config_digest(config: "ExecutionConfig") -> str:
    """Return the per-image scientific configuration digest.

    D-C: this IS ``processing_configuration_digest`` -- the same function object,
    re-exported under the spec's name so §5.4's "one definition, two uses" is
    literally true rather than aspirationally true.
    """
```

**Consumes:** `phenotypic.sdk_.RunIdentity` (P1),
`phenotypic._cli._cli_failure_tracker.processing_configuration_digest`.

---

## Task 1: `restart_epoch` — the one tracked counter

**Files:**
- Create: `src/phenotypic/_cli/_cli_identity.py`
- Modify: `src/phenotypic/sdk_/_io_constants.py`
- Test: `tests/unit/cli/test_run_identity.py`

`clear_machine_state` (`_io_constants.py:1081`) currently deletes every child of
`.phenotypic/` except `terminal_failures.jsonl`. D4 requires `restart_epoch` to survive it
— otherwise a restart resets the counter and the stale-worker fence it exists for is gone.

- [ ] **Step 1: Write the failing test**

```python
def test_restart_epoch_survives_clear_machine_state(tmp_path):
    """D4: restart_epoch is THE one added tracked value, and it is worthless if a
    restart resets it -- the whole point is to distinguish 'deliberately fresh
    attempt' from 'same config again', which is exactly what a restart is."""
    from phenotypic._cli._cli_identity import bump_restart_epoch, read_restart_epoch
    from phenotypic.sdk_ import clear_machine_state

    (tmp_path / ".phenotypic").mkdir()
    assert read_restart_epoch(tmp_path) == 0
    assert bump_restart_epoch(tmp_path) == 1
    assert bump_restart_epoch(tmp_path) == 2

    clear_machine_state(tmp_path)
    assert read_restart_epoch(tmp_path) == 2, (
        "clear_machine_state destroyed the restart epoch; the fence it exists for "
        "cannot survive the operation it exists to fence"
    )


def test_reading_a_corrupt_restart_epoch_is_zero_not_an_error(tmp_path):
    """INV-DEGRADE. A restart must not be blocked by an unparseable counter."""
    from phenotypic._cli._cli_identity import read_restart_epoch

    cache = tmp_path / ".phenotypic"
    cache.mkdir()
    (cache / "restart_epoch.json").write_text("{not json", encoding="utf-8")
    assert read_restart_epoch(tmp_path) == 0
```

- [ ] **Step 2: Run to verify failure.** Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Add `RESTART_EPOCH_JSON: Final[str] = "restart_epoch.json"` and
`restart_epoch_path(output_dir)` to `_io_constants.py`, then extend
`clear_machine_state`'s preserve set:

```python
    _PRESERVED_ON_RESTART = frozenset({TERMINAL_FAILURES_JSONL, RESTART_EPOCH_JSON})
    ...
        for child in cache.iterdir():
            if child.name in _PRESERVED_ON_RESTART:
                continue
```

Update `clear_machine_state`'s docstring: it currently says it preserves "the append-only
`terminal_failures.jsonl` journal"; it now preserves that **and** the restart epoch, and
the docstring must say why — a counter that resets on the operation it fences is not a
fence.

`bump_restart_epoch` writes through `atomic_write_json`.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_identity.py src/phenotypic/sdk_/_io_constants.py \
        tests/unit/cli/test_run_identity.py
git commit -m "feat(cli): add restart_epoch, preserved across clear_machine_state

Spec §5.1 D4. One tracked integer, and the only one this design adds. It is
preserved by --restart on purpose: content-derived generations cannot tell a
deliberately fresh attempt from the same config again."
```

---

## Task 2: `scientific_config_digest` is the existing digest, and the spec is corrected

**Files:**
- Modify: `src/phenotypic/_cli/_cli_identity.py`
- Modify: `docs/superpowers/specs/2026-09-03-cli-gui-state-tracking/design.md`
- Test: `tests/unit/cli/test_run_identity.py`

Implements [D-C](OPEN-QUESTIONS.md#d-c-scientific_config_digest-is-the-existing-digest-verbatim-was-q1).

- [ ] **Step 1: Write the failing test**

```python
def test_scientific_config_digest_is_the_work_id_digest_itself(tmp_path):
    """D-C / spec §5.4: 'not a new digest ... reused verbatim'.

    §5.4's argument is that if the generation and work_id could disagree about what
    counts as scientific configuration, a change could invalidate per-image proofs
    without minting a new generation, or vice versa. Identity is the strongest form
    of agreement available, so this is an `is` check, not an equality check -- an
    equal-but-separate function would drift.
    """
    from phenotypic._cli._cli_failure_tracker import processing_configuration_digest
    from phenotypic._cli._cli_identity import scientific_config_digest

    assert scientific_config_digest is processing_configuration_digest
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement — one line, plus the docstring that explains it**

```python
# D-C: the spec calls this `scientific_config_digest`; the code has called it
# `processing_configuration_digest` since work_id was introduced. Re-export rather
# than wrap: §5.4's whole argument is that the generation and work_id must never
# disagree about what counts as scientific configuration, and identity is the only
# form of agreement that cannot drift.
#
# §5.4's prose ALSO claims include_dataset_column, overlay_alpha and save_overlays
# are excluded. They are not -- see _cli_failure_tracker.py:238. The prose is the
# wrong half (OPEN-QUESTIONS D-C); removing them from the per-image digest is a
# work_id change and belongs to its own spec with its own migration.
scientific_config_digest = processing_configuration_digest
```

- [ ] **Step 4: Correct the spec**

In `design.md` §5.4, replace the field list with the actual contents of
`processing_configuration_digest_from_values`, and add to §5.3, under the digest table:

> **Footnote (D-C).** `include_dataset_column` appears in both
> `scientific_config_digest` (via `work_id`) and `finalization_input_digest`. The two
> answer different questions and a field may be relevant to both, so "none is redundant"
> refers to the digests, not to their fields. Flipping `include_dataset_column` therefore
> still reprocesses every image; narrowing the per-image digest is a `work_id` change and
> deserves its own spec.

Mark the edit in the spec's own change log if it has one; otherwise state it in the commit
body. **Do not silently rewrite a spec section** — a reader comparing the plan to the spec
needs to see that the plan won an argument, not that the spec was always right.

- [ ] **Step 5: Run the test.** Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_cli/_cli_identity.py tests/unit/cli/test_run_identity.py \
        docs/superpowers/specs/2026-09-03-cli-gui-state-tracking/design.md
git commit -m "feat(cli): scientific_config_digest IS processing_configuration_digest

D-C. Also corrects design.md §5.4's field list, which claimed
include_dataset_column, overlay_alpha and save_overlays are excluded from work_id.
They are in it (_cli_failure_tracker.py:238), and §5.1 says work_id is unchanged --
the two could not both hold."
```

---

## Task 3: The content-derived `processing_generation`

**Files:**
- Modify: `src/phenotypic/_cli/_cli_identity.py`
- Modify: `src/phenotypic/_cli/_cli_state_management.py:237`
- Test: `tests/unit/cli/test_run_identity.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_the_same_config_mints_the_same_generation(tmp_path, execution_config):
    """D3: same inputs -> same token, so resume and fencing are emergent rather
    than bookkeeping. A SLURM worker starting cold can fence itself correctly
    against a run it has never read."""
    from phenotypic._cli._cli_identity import mint_run_identity

    a = mint_run_identity(execution_config, restart=False)
    b = mint_run_identity(execution_config, restart=False)
    assert a.processing_generation == b.processing_generation


def test_a_pipeline_edit_mints_a_new_generation(tmp_path, execution_config):
    from phenotypic._cli._cli_identity import mint_run_identity

    before = mint_run_identity(execution_config, restart=False)
    execution_config.pipeline_json.write_text(
        execution_config.pipeline_json.read_text() + "\n", encoding="utf-8"
    )
    after = mint_run_identity(execution_config, restart=False)
    assert after.processing_generation != before.processing_generation


def test_a_restart_mints_a_new_generation_for_identical_config(tmp_path, execution_config):
    """D4's reason for existing. Without restart_epoch the generation is a pure
    function of configuration, so a deliberately fresh attempt against unchanged
    config is indistinguishable from the run it replaces -- and a worker still
    holding the pre-restart generation would pass the fence."""
    from phenotypic._cli._cli_identity import mint_run_identity

    before = mint_run_identity(execution_config, restart=False)
    after = mint_run_identity(execution_config, restart=True)
    assert after.restart_epoch == before.restart_epoch + 1
    assert after.processing_generation != before.processing_generation


def test_a_new_image_does_NOT_mint_a_new_generation(tmp_path, execution_config):
    """D7: inventory_digest is deliberately OUT of the generation digest.

    Generation fences configuration; inventory_digest fences scope. Conflating them
    would make every new image under a rolling input look like a configuration
    change -- resetting live progress and fencing in-flight workers, which is the
    exact failure mode a 6,000-image rolling dataset produces daily."""
    from phenotypic._cli._cli_identity import mint_run_identity

    before = mint_run_identity(execution_config, restart=False)
    _add_image_to_input(execution_config)
    after = mint_run_identity(execution_config, restart=False)
    assert after.processing_generation == before.processing_generation
    assert after.inventory_digest != before.inventory_digest


def test_a_metadata_edit_does_NOT_mint_a_new_generation(tmp_path, execution_config):
    """§5.4/§7.4: a metadata edit changes finalization_input_digest only, so the
    next invocation re-runs finalize_run without touching a single image's
    measurement."""
    from phenotypic._cli._cli_identity import mint_run_identity

    before = mint_run_identity(execution_config, restart=False)
    execution_config.metadata_csv.write_text(
        "Metadata_Well,Metadata_Strain\nA1,new\n", encoding="utf-8"
    )
    after = mint_run_identity(execution_config, restart=False)
    assert after.processing_generation == before.processing_generation
    assert after.finalization_input_digest != before.finalization_input_digest
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

```python
def mint_run_identity(config: "ExecutionConfig", *, restart: bool) -> RunIdentity:
    """Mint the identity of a new or resumed invocation (spec §5.1, §5.4).

    ``processing_generation`` is ``sha256(pipeline_sha256 || scientific_config_digest
    || restart_epoch)``. **Writer** -- it can bump the restart epoch, which is why it
    lives in ``phenotypic._cli`` and not beside the readers in ``sdk_/_run_state.py``.

    ``inventory_digest`` is deliberately absent from the generation (D7): generation
    fences *configuration*, ``inventory_digest`` fences *scope*, and they change on
    different schedules. Folding them together makes every arrival under a rolling
    input look like a configuration change.

    Args:
        config: The invocation's execution configuration.
        restart: ``True`` for ``--restart``, which bumps and persists the epoch.

    Returns:
        A :class:`~phenotypic.sdk_.RunIdentity`.
    """
```

Then change `_cli_state_management.py:237` from `"processing_generation": uuid4().hex` to
the minted value, and add `"restart_epoch": identity.restart_epoch` to the same config
block.

**Mint exactly once per invocation, then thread the value (CAN-21).** `mint_run_identity`
bumps and persists `restart_epoch`, so it is a *writer*, and calling it twice in one run
gives that run two generations and burns an epoch. `ExecutionConfig.output_dir` is
`Optional[Path]` (`_cli_types.py:99`), so `RunIdentity` cannot re-read the epoch to make
the call idempotent. The rule is therefore structural: **the CLI entry point mints once and
passes the `RunIdentity` down.** Nothing below the entry point calls `mint_run_identity`.

```python
def test_minting_twice_in_one_invocation_is_a_programming_error(tmp_path, execution_config):
    """CAN-21. Two mints = two generations for one run, and a burned epoch. There
    is no output root on the config to make this idempotent, so the guard is a
    loud failure rather than a silent second bump."""
    import pytest

    from phenotypic._cli._cli_identity import mint_run_identity

    mint_run_identity(execution_config, restart=True)
    with pytest.raises(RuntimeError, match="already minted"):
        mint_run_identity(execution_config, restart=True)
```

- [ ] **Step 4: Fix the third and fourth minting sites — both already wrong**

**`_cli_migrate.py:684-705` (CAN-7)** sets

```python
processing_generation = hashlib.sha256(f"migration\n{inventory_payload}".encode()).hexdigest()
```

where `inventory_payload` (`:678-682`) is the full `dataset/stem:work_id` listing. **That is
the inventory folded into the generation — exactly what D7 forbids — and it is in the tree
today.** Every migrated tree therefore behaves the way D7 exists to prevent: each new image
under a rolling input changes the generation, resetting live progress and fencing in-flight
workers. Left unfixed, P5's rolling-input matrix fails on any migrated tree and the failure
looks like a bug in P5 rather than an unrevised writer.

It also writes `work_ids` (`:695`) with **no `restart_epoch`** — which is precisely P1's
`requires_conversion` signal 4 (CAN-11 moved that gate to P1), so a freshly HDF-migrated
tree would be refused by the very next `--mode full`.

Bring it to the v3 schema: content-derived generation, `restart_epoch: 0`, no
`datasets.{completed,failed,started}`.

**`phenotypicCLI.py:2640,2716` (CAN-20)** skips state creation in measure mode — *"skipped
in measure mode, which never mutates processing state"* — then sets
`config.processing_generation = uuid4().hex` at `:2716`. But §7.4 routes measure through
`finalize_run` and P4 Task 4 parametrizes a byte-identical master over
`["full","measure","recompile"]`. Decide and state: **what identity does a measure
invocation run under, and may it publish proofs?** The answer that keeps §7.4 coherent is
that measure mints the same content-derived identity as `full` — the pipeline and
per-image config are unchanged, so the generation is *the same value*, which is exactly
what D3's "same inputs → same token" is for. `--mode process` needs the same statement
(DF-16).

- [ ] **Step 5: Write the tests for both**

```python
def test_a_migrated_trees_generation_is_unchanged_by_an_image_arrival(tmp_path):
    """CAN-7 / D7. _cli_migrate.py:686 folds the inventory into the generation."""
    output = _run_hdf_migration(tmp_path)
    before = _state_config(output)["processing_generation"]
    _add_image_and_remigrate(output)
    assert _state_config(output)["processing_generation"] == before


def test_a_freshly_migrated_tree_is_not_refused_by_the_next_full_run(tmp_path):
    """CAN-7 + CAN-11. The migrator wrote work_ids with no restart_epoch, which is
    the gate's own signal 4. Build the fixture with the REAL migrator -- a
    hand-planted tree cannot catch this class of drift."""
    from phenotypic._cli._cli_schema_gate import requires_conversion

    output = _run_hdf_migration(tmp_path)
    assert requires_conversion(output) is None


@pytest.mark.parametrize("mode", ["full", "measure", "process"])
def test_every_mode_mints_the_same_identity_for_the_same_config(tmp_path, mode):
    """CAN-20 / DF-16. D3: same inputs -> same token."""
    assert _minted_generation(tmp_path, mode) == _minted_generation(tmp_path, "full")
```

- [ ] **Step 4: Run the tests.** Expected: PASS (5 passed).

- [ ] **Step 5: Regression — `--restart` still reuses surviving stores (D5)**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli -k 'restart or resume' -q
```

D5 is explicit: `--restart` keeps reusing surviving `results/` stores. The epoch fixes the
stale-worker hazard **without** turning `--restart` into `--overwrite`. If any of these
tests now show a restart reprocessing images it previously reused, the epoch has leaked
into `work_id` and the change is wrong.

- [ ] **Step 6: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli/test_run_identity.py
git commit -m "feat(cli): processing_generation becomes content-derived

Spec §5.1, §5.4, D3, D4, D7. sha256(pipeline || scientific_config || restart_epoch).
inventory_digest stays out (D7) so a rolling input's arrivals do not read as a
config change. --restart still reuses surviving stores (D5)."
```

---

## Task 4: The `scheduler_epoch` collapse — only where one writer owns the lifetime

**Files:**
- Modify: `src/phenotypic/_cli/_cli_slurm_lifecycle.py:78`
- Modify: `src/phenotypic/_cli/_cli_completion.py:163` (`publish_image_success`)
- Test: `tests/unit/cli/test_run_identity.py`

Implements [O-1](OPEN-QUESTIONS.md#o-1-scheduler_epoch-may-be-five-names-collapsing-to-one-owner-not-five-tokens-to-one).
**Read it before starting — this task is deliberately narrower than §5.1 asks for.**

§5.1 has `scheduler_epoch` absorb five tokens. Four subsystems write those five
(`_cli_slurm_lifecycle`, `_cli_staged_orchestration`, the recompile worker, the local
strategy) at four different times with four different lifetimes. Collapsing the *names*
without collapsing the *writers* gives one value with four owners — a coupling increase
dressed as a cardinality reduction.

**This task collapses only the pair that is already one value.** The audit found it
(§11.1): `_assert_worker_generation`'s `slurm_generation != attempt_id` check is "one value
passed twice, then asserted equal".

- [ ] **Step 1: Confirm the audit's claim against the code before acting on it**

```bash
grep -n '_assert_worker_generation' -A 25 src/phenotypic/_cli/*.py
```

Expected: the two compared values originate from the same source. **If they do not, stop
and report** — the collapse's justification is that finding, and a wrong finding makes this
task a behaviour change rather than a rename.

- [ ] **Step 2: Write the failing stale-worker test — against the fence that exists**

**CAN-15: the first draft of this test targeted a mechanism that is not there.**
`publish_image_success` raises "stale SLURM lifecycle" only when `SLURM_JOB_ID` is set
**and** `slurm_lifecycle.json` exists, and it compares the **lifecycle** generation
(`_cli_completion.py:181-188`) — not `processing_generation`. The draft passed neither
precondition (bare `tmp_path`, no env, `artifacts={}`), so it would have written a marker
and not raised, while also passing a `processing_generation` in as `scheduler_epoch` and
conflating two of the five tokens.

Spec §14's actual requirement is *"a worker holding the pre-restart generation must not
have its **events counted**"* — and that fence is the `generation=` argument to
`aggregate_state_from_events` (`_cli_update_state.py:266`, reached from
`_cli_state_management.py:112-125`). **No phase currently touches it.** Test both halves:

```python
def test_a_stale_workers_events_are_not_counted(tmp_path, execution_config):
    """Spec §14's stale-worker test, against the real fence.

    The event log is generation-fenced at aggregate_state_from_events. A worker
    that started before a --restart appends events tagged with the pre-restart
    generation; the post-restart state must not count them.
    """
    from phenotypic._cli._cli_identity import mint_run_identity
    from phenotypic._cli._cli_state_management import load_processing_state

    stale = mint_run_identity(execution_config, restart=False)
    _append_event(tmp_path, generation=stale.processing_generation, image="a.tif",
                  state="completed")
    fresh = mint_run_identity(execution_config, restart=True)
    assert fresh.processing_generation != stale.processing_generation

    state = load_processing_state(tmp_path)
    assert "a.tif" not in state.datasets["plate"].completed, (
        "an event tagged with the pre-restart generation was counted; the "
        "restart_epoch fence D4 exists for is not reaching the event log"
    )


def test_a_stale_slurm_worker_cannot_publish(tmp_path, execution_config, monkeypatch):
    """The OTHER half, and the one the first draft meant to test. Both
    preconditions must be set up or the guard is simply not reached."""
    import pytest

    from phenotypic._cli._cli_completion import publish_image_success

    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    _write_slurm_lifecycle(tmp_path, generation="current-epoch", active=True)

    with pytest.raises(RuntimeError, match="stale SLURM lifecycle"):
        publish_image_success(
            tmp_path,
            work_id="w", dataset="plate", relative_image_path="a.tif",
            image_stem="a", mode="full", attempt_id="attempt",
            scheduler_epoch="a-superseded-epoch",     # != the lifecycle generation
            artifacts={"store": _a_promoted_store(tmp_path)},
        )
```

Note the second test passes a real artifact: `publish_image_success` resolves every
artifact `strict=True`, so `artifacts={}` exercises a different path than the one under
test.

- [ ] **Step 3: Run to verify failure.**

- [ ] **Step 4: Implement**

Rename `publish_image_success`'s `lifecycle_epoch` parameter to `scheduler_epoch` and
update all call sites. Keep the staged `epoch` and recompile `attempt_id` as **diagnostic**
fields written under the collapsed name but never compared — spec §5.1 already classifies
per-image `attempt_id` as "written, never branched on", so this is that rule applied
consistently rather than a new exception.

**On `_assert_worker_generation`: Step 1's confirmation actually decides this, and the
first draft pre-committed to the deletion before asking (CAN-33 m8).** The audit reads it
as "one value passed twice, then asserted equal", but the function's own docstring
(`_cli_recompile_worker.py:104-108`) says both values are *"supplied **independently** by
the scheduler script"* — they are equal only because
`_cli_recompile_slurm_scripts.py:292-293` passes `attempt_id` twice. So the check is a live
guard against a hand-edited script or a directly invoked worker, and deleting it removes
that guard.

Resolve it on the evidence from Step 1:

- If the two values provably originate from one source **and nothing can invoke the worker
  with them differing**, delete the check and say so in the commit.
- Otherwise **keep it**, and record in `_cli/CLAUDE.md` (P7 Task 6) that it guards
  direct/hand-edited invocation rather than a scheduler-supplied disagreement. Spec §11.1
  lists it under deletions; a kept guard with a documented reason is a spec deviation worth
  one line, not a silent retention.

- [ ] **Step 5: Run the test and the SLURM lifecycle regression**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli -k 'lifecycle or slurm or staged' -q
```

- [ ] **Step 6: Phase gate**

```bash
uv run mypy src/phenotypic
uv run ruff check --fix src/phenotypic/_cli/_cli_identity.py \
  src/phenotypic/_cli/_cli_state_management.py src/phenotypic/_cli/_cli_slurm_lifecycle.py \
  src/phenotypic/_cli/_cli_completion.py src/phenotypic/sdk_/_io_constants.py \
  tests/unit/cli/test_run_identity.py
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli tests/unit/sdk_ -q
```

- [ ] **Step 7: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli
git commit -m "refactor(cli): collapse slurm_generation and lifecycle_epoch into scheduler_epoch

Spec §5.1, narrowed by OPEN-QUESTIONS O-1: only the pair the audit found to be one
value passed twice (§11.1) is collapsed. Staged epoch and recompile attempt_id are
written under the name as diagnostics and never compared -- four writers behind one
compared value would be a coupling increase, not a cardinality reduction."
```
