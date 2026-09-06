# P3 implementation & test review — the per-image record collapse

**Subject:** commit `1cc6740c` on branch `cli-gui-state-tracking`, reviewed against its
parent `9480dd5b`.
**Scope:** implementation correctness of the §6.1 collapse, and whether the tests
shipped with it can actually fail.
**Method:** every `src/` hunk read line by line against its call sites; the 1152-cell
table analysed mechanically rather than by sampling; each `xfail(strict=True)` traced to
the code path its `reason` names. Every finding below says whether it is **CONFIRMED**
(established by reading the code cited) or **PLAUSIBLE** (a reachability claim I could
not close without running).

**Runs.** I do not run commands in this role; the three below were executed by the lead
and returned verbatim.

- `tests/unit/cli/{test_image_record,test_staged_resume_parity,test_schema_gate,test_cli_completion_store,test_embedded_measurement_checkpoint,test_embedded_measurement_recompile}.py`
  + `tests/unit/sdk_/{test_run_state_layering,test_migration_republishes_state}.py`
  → **489 passed, 10 skipped, 13 xfailed, 0 xpassed** in 17.70 s.
- `tests/unit/cli/test_staged_resume_equivalence.py` → **1158 passed** in 12.90 s.
- The `record_rejection` clause-order probe (F4) → output inline below.
- The parent-tree reconstruction probe (F2/F3) → output inline below.
- `tests/unit/cli/test_cli_recompile{,_slurm}.py` under `--runxfail --tb=line`
  → **45 passed, 35 xfailed, 0 xpassed**; per-traceback breakdown below.

**Every line and line number in this report resolves against `1cc6740c`**, quoted via
`git show` rather than from the working tree. The working tree has since moved — F2's fix
was dispatched while the review was in progress — so a reader checking these citations
against the checkout may find them shifted or already repaired.

---

## Summary

The collapse itself is right. `publish_image_record` merges rather than replaces,
`record_rejection` is a clause-for-clause port of `marker_rejection` with a strictly
*stronger* provenance read, the reader/writer split genuinely satisfies INV-LAYER, and
the equivalence gate is **not** vacuous — I measured it (see *Was the gate real?* below).
The four regressions the commit message names are correctly diagnosed and correctly
fixed.

What it did not finish is the **repoint sweep**. Three consumers of the deleted
`image_complete/` tree were left pointing at it in ways the phase's own tripwires do not
cover, and one of them does not merely *read* the dead tree — it **creates** it, on a
forward tree, which falsifies the standing evidence P7 is meant to arm the schema gate
on. One shipped test passes for a reason other than the one it is named for, and one
strict deferral marker attributes an in-phase regression to the previous build.

**The recurring shape is documentation written as an append.** F3 and F9 are the same
error twice: text that was true when written, made false by this commit, and left
standing while a correction was added beside or below it. In F9's case the two halves are
forty lines apart in one docstring, and the stale half is the one a reader meets first.

None of the fixes below adds a tracked state, a content proof, or a file. Every one is
either a repointed read, a strengthened predicate, a rewritten docstring, or a test.
**The 4/3/2 budget is not at risk from anything in this report.**

| # | Severity | Finding | Status |
|---|---|---|---|
| F1 | **HIGH** | The recompile store lock still lives under `image_complete/` and **creates** that directory on a forward tree | CONFIRMED |
| F2 | **HIGH** | `authorized_measurement_sources` has two arms; only one was repointed. Plan Step 3b named both | CONFIRMED |
| F3 | **HIGH** | The new strict xfail on `test_embedded_measurement_checkpoint.py` misattributes F2 as pre-existing, and assigns it to the wrong owner | CONFIRMED |
| F4 | MEDIUM | `test_a_stage2_only_record_is_not_a_success_proof` never reaches the CAN-23 clause it is named for | CONFIRMED |
| F5 | MEDIUM | `_cli_process_single`'s new `record_path.is_file()` guard is not equivalent to the marker guard it replaced | CONFIRMED |
| F6 | MEDIUM | The bulk Stage-3 sweeps went from *n* `stat()`s to *n* JSON reads-and-parses | CONFIRMED |
| F7 | LOW | The INV-LAYER AST walk covers function bodies, but not attribute access through an imported parent package | CONFIRMED |
| F8 | LOW | Residual doc/comment drift left by the sweep | CONFIRMED |
| F9 | MEDIUM | `SCHEMA_GATE_ARMED`'s docstring opens with three false statements and is contradicted 40 lines later by text this commit added | CONFIRMED |

---

## Was the gate real? — the question you asked first

> *"Ask what a green run would look like if the collapse HAD broken a decision."*

**It would be red, and I can say by how much.** I parsed the committed `_EXPECTED`
block mechanically (1152 rows, matching `_COMBOS` exactly) and computed, for each axis,
how many otherwise-identical groups disagree when only that axis moves:

| axis | live? | groups whose verdict changes on this axis alone |
|---|---|---|
| `store` | yes | 384 |
| `table` | yes | 14 |
| `s2_token` | yes | 233 |
| `s2_raw` | yes | 140 |
| **`s3_done`** | **yes** | **52** |
| `layer` | yes | 226 |
| `markers_required` | yes | 6 |
| `expect_work_id` | yes | 312 |

Outcome distribution: `stage1` 528, `stage2` 274, `complete` 198, `stage3` 152 — all
four reachable, so `test_the_axes_reach_every_outcome` is not asserting a tautology.

The suite runs green — **1158 passed in 12.90 s**, with the module-scoped
`store_templates` doing its job (0.17 s of setup on the first cell, ~0.02 s per cell
after). The cost objection to a 1152-cell table does not arise.

**`s3_done` is the axis this task rewrote, and it moves 52 groups.** A collapse that
broke the stage-3 decision — `stage3_completion_exists` failing to see what
`write_stage3_completion_marker` wrote, `record_stage` clobbering a sibling entry,
`consume_stage` removing the wrong key — flips those 52 cells and the parametrized test
fails 52 times with the axis tuple in the message. That is the answer to your question:
a green run is not consistent with a broken stage-3 decision.

Three further things the harness gets right, which are worth recording because each is a
way this class of gate usually goes hollow:

- **`_plant` writes the stage-3 fact through the writer, not a hand-joined path**
  (`test_staged_resume_equivalence.py:249`). Hand-joining
  `progress/stage3_complete/<ds>/<stem>.json` would have made every `s3_done=True` cell
  fail the moment the writer moved — a tautological failure detector, not an equivalence
  gate. The module docstring says exactly this, and the code does it.
- **`test_every_axis_changes_at_least_one_outcome` is the check the two-capture protocol
  structurally cannot make.** Both captures run the same `_plant`; an axis `_plant`
  silently fails to plant produces the identical wrong table twice and the comparison
  reports "identical". My independent computation above confirms all eight are live in
  the *committed* table, so that guard is currently satisfied on real data rather than in
  principle.
- **`test_the_store_templates_have_the_predicates_they_claim` executes the `(V,S,W)`
  table that `_STORE_STATES` only comments.** Without it, `_mark_journal_in_progress`
  silently not taking would collapse two of the five store states onto their valid
  counterparts, and the store axis would stay "live" on the strength of `absent` alone.

**The hybrid risk was measured, and I re-derived it.** The commit claims
`valid_image_success` was CALLED in 576/1152 cells and TRUE in 0. 576 is exactly the
`expect_work_id=True` half, which matches `classify_staged_image`'s guard at `:235`. And
post-collapse it stays 0 for a reason I checked rather than took on faith: an `s3_done`
cell's record is written by `record_stage`, which writes no `work_id`, so
`record_rejection` refuses it. The frozen table is honest in both worlds.

**The one thing the gate cannot see**, stated so nobody assumes otherwise: it freezes
the *pair* (writer, reader). A change that moved both consistently — a renamed stage key,
say — leaves every cell green while any third-party consumer of that tree breaks.
`test_the_stage_names_come_from_one_shared_constant` covers the narrow version of that
(one shared object, `is` not `==`), and every production consumer routes through
`stage3_completion_exists`, so the residual exposure is small. It is not zero.

---

## F1 — HIGH — the recompile store lock still lives under `image_complete/`, and creates it

**CONFIRMED.**

`src/phenotypic/_cli/_cli_recompile_recovery.py:48-55`:

```python
def recompile_store_lock_path(output_dir, dataset_name, stem) -> Path:
    """Return the lock shared by canonical recompile mutations for one store."""
    return image_completion_marker_path(
        output_dir, dataset_name, stem
    ).with_suffix(".recompile-store.lock")
```

That resolves to `.phenotypic/progress/image_complete/<ds>/<stem>.recompile-store.lock`.

`src/phenotypic/_cli/_cli_recompile_tables.py:105-108` — inside the function P3
*did* repoint — takes that lock as its very first act:

```python
with exclusive_path_lock(
    recompile_store_lock_path(output_dir, dataset, stem),
    timeout=60.0,
):
```

and `sdk_/_file_locking.py:39-41` is unambiguous about what acquiring it does:

```python
path = Path(lock_path)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a+b") as handle:
```

**So `--mode recompile` on a tree this build wrote materialises
`.phenotypic/progress/image_complete/<ds>/`.** Nothing removes it — `grep -rn
'recompile-store.lock'` finds one producer and no cleanup.

### Why that matters more than a stray file

`sdk_/_schema_shape.py:278-284`, signal 1:

```python
for segment in (DIR_IMAGE_COMPLETE, _DIR_STAGE3_COMPLETE):
    if (progress / segment).is_dir():
        return (ConversionVerdict.CONVERT, ...)
```

It is a **directory** probe. It does not care that the directory holds a lock file rather
than markers. So after one recompile, `requires_conversion(tree)` returns `CONVERT` for a
tree the current publisher wrote.

That is precisely the proposition `test_image_record.py:688`
(`test_a_tree_this_build_wrote_needs_no_conversion`) exists to hold up — its own docstring
calls itself *"the standing evidence that arming is safe for forward-written trees, which
is what P7 Task 5 Step 1b needs before it flips the flag."* The test builds
`build_complete_run(tmp_path)` and never runs recompile, so it does not see this. **The
evidence P7 is meant to arm on is false for any tree that has ever been recompiled, and
the test that certifies it cannot tell.**

Blast radius once P7 arms: every writing mode refuses the tree
(`_refuse_unmigrated_output`), and the remedy it names is `--mode migrate` — which per this
phase's own `_schema_shape.py:126-133` note does not remove `image_complete/`. P7 Task 5
Step 1b's rename into `.phenotypic/legacy-v2/` would sweep it up incidentally, so this may
not survive to production; but relying on that is relying on a coincidence, and in the
P3→P7 interval the invariant is simply false.

Today the failure is also reachable *and pointless*: `_replace_and_republish_table` takes
the lock, then calls `begin_recompile_table_transition`, which at
`_cli_recompile_recovery.py:387` does `image_completion_marker_path(...).read_text()` and
raises `FileNotFoundError` on a forward tree. So the current net effect of a forward-tree
recompile is: create the legacy directory, then fail.

### Fix

Move the lock off the marker path. It is a lock, not a marker — it has no business being
derived from `image_completion_marker_path` at all, and `image_record_path(...)
.with_suffix(".recompile-store.lock")` would put it under `progress/images/`, which signal
1 does not probe. That is a one-line change with no new artifact (the lock already exists;
only its parent directory moves). If it is judged P4's to make — `_cli_recompile_recovery`
is already assigned there — then it needs a test **now**, because the hazard is live in the
interval:

```python
def test_a_forward_tree_still_needs_no_conversion_after_a_recompile(tmp_path):
    """F1: the recompile store lock must not resurrect image_complete/."""
    root = build_complete_run(tmp_path)
    assert requires_conversion(root) is None
    with contextlib.suppress(Exception):          # P4 owns making this succeed
        recompile_embedded_measurement_tables(root, None)
    assert requires_conversion(root) is None, (
        "recompile created progress/image_complete/; signal 1 now fires on a "
        "tree this build wrote, and P7 cannot arm the gate"
    )
```

**Budget: adds nothing.** The lock file is not new; only its directory changes.

---

## F2 — HIGH — `authorized_measurement_sources` has two arms; only one was repointed

**CONFIRMED.**

The plan named both. `phase-3-per-image-record.md:886`:

> `_cli_completion.py:786` globs `progress_dir/DIR_IMAGE_COMPLETE/*/*.json` **and** `:838`
> reads `image_completion_marker_path`.

P3 repointed `:838` (now `_cli_completion.py:876`, with a good CAN-22 comment). `:786`
— now `_cli_completion.py:824` — is untouched:

```python
if state is None or not state.config.get("success_markers_required", False):
    marker_root = progress_dir(output_dir) / DIR_IMAGE_COMPLETE
    marker_paths = sorted(marker_root.glob("*/*.json"))
    if not marker_paths:
        return None
    ...
    for marker_path in marker_paths:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        ...
        if not valid_image_success(output_dir, dataset=..., image_stem=..., work_id=...):
            continue
```

This arm serves the population that has **no** `success_markers_required` in its state —
which after `_cli_state_management.py:275` (`"success_markers_required": True`) means
legacy trees and stateless worker invocations. Two distinct breakages, both silent:

**(a) A legacy tree now yields `{}` where it used to yield real sources.** `marker_paths`
is non-empty (the tree has `image_complete/`), so the `return None` escape is not taken;
but `valid_image_success` now reads `progress/images/`, which a legacy tree does not have,
so every image `continue`s and the function returns an empty dict.

`{}` is not `None`, and the difference is load-bearing at both call sites:

- `_cli_chunk_writer.py:144-155` — `embedded_authority = authorized_sources is not None`
  is now **True** with an empty source list, so the `_scan_unchunked_parquets` fallback
  is skipped and the writer logs *"No new measurement files to chunk"* and returns. It
  used to aggregate the legacy parquets.
- `_cli_recompile_tables.py:238` — `if authorized is None:` is False, so the branch that
  raises `"Legacy external measurement Parquets require --mode migrate"` is never reached.
  `recompile` returns `0 changed` instead. **A loud, actionable migration error became a
  silent no-op.**

**(b) A forward tree with no processing state now yields `None`.** `image_complete/` is
absent, so `marker_paths` is empty and the arm returns `None` — see F3, which is this
case with a test attached.

### The attribution, settled

Two independent halves, both closed:

**Structural.** The legacy arm is byte-identical at `9480dd5b` and `1cc6740c` — it does
not appear in this commit's diff. Only its callee moved.

**Empirical.** Reconstructing the tree the parent's publisher wrote — publish through
`publish_image_success`, then move the payload to `image_complete/` with `version =
SUCCESS_MARKER_VERSION` and delete the record — and evaluating the parent's
`valid_image_success` by hand over it, clause for clause:

```
processing_state.json present : False      # so the legacy arm is the one taken
marker_rejection              : None       # the parent's first check passes
artifact fences               : True       # the parent's second check passes
```

`git show 9480dd5b:src/phenotypic/_cli/_cli_completion.py:282` is that function in full,
and it is exactly those two checks over `image_completion_marker_path`. So **at the
parent, on this tree, `valid_image_success` returned `True` and the legacy arm returned
`{<parquet>: 'ds'}`.** Today the same tree yields nothing, because the predicate reads
`images/`, which the parent never wrote.

The fourth line of that probe — the arm's own return value — is not obtainable from this
tree and is now moot: F2's fix landed during the review, so what the arm returns is a
property of the fix rather than evidence about the defect. The first three lines plus the
byte-identity settle it without needing the fourth.

The gating predicate is the real error, not just the path. Arm 1 opens a *legacy* marker
and then asks a *record* predicate whether it is valid. The correct predicate for a legacy
marker is `marker_rejection`, which this very commit retained in `sdk_/_run_state.py:512`
with a prominent docstring asserting it has no caller in `src/`. **That docstring is wrong
by one**: arm 1 is a third would-be caller, alongside the two it names
(`refresh_success_markers_after_metadata_migration` and P7's migrator).

### Fix

Arm 1 must handle whichever shape it finds, and gate each on its own predicate:
`image_complete/` markers via `marker_rejection` + `fenced_artifact_path`, `images/`
records via `record_rejection`. Or — if the ruling is that arm 1 is legacy-only and a
legacy tree should be refused rather than served — then say so by making it `return None`
unconditionally once the gate arms, and record the decision. Either way,
`test_authorized_sources_reads_records_not_the_deleted_tree`
(`test_image_record.py:414`) needs a sibling that does **not** call
`write_processing_state`, because that call is exactly what routes the existing test down
arm 2 and leaves arm 1 unexercised.

**Budget: adds nothing.** A predicate swap and a test.

---

## F3 — HIGH — the checkpoint tripwire misattributes F2 and assigns it to the wrong owner

**CONFIRMED.**

`tests/unit/cli/test_embedded_measurement_checkpoint.py:24-38` adds
`xfail(strict=True)` with:

> "**PRE-EXISTING, and not P3's** … Both preconditions predate this change. Owned by
> whoever next revisits the chunk writer."

The failure mechanism the reason describes is accurate. The attribution is not, and one
sentence of it is false outright.

**"never reaching the record reader P3 changed" is wrong on its face.** The legacy arm's
loop *calls* `valid_image_success` — `_cli_completion.py:838` today, `:800` at the parent
— and `valid_image_success` is precisely the record reader P3 changed. The arm reaches it
on every marker it globs.

**The arm itself is byte-identical across the two commits.** It does not appear in this
commit's diff at all; only its callee moved. `git show
9480dd5b:src/phenotypic/_cli/_cli_completion.py:282` is the parent's
`valid_image_success` in full — `marker_rejection` plus a `fenced_artifact_path` walk,
over `image_completion_marker_path`. **An unchanged function whose behaviour moved
because its callee moved is this change's, not a precondition of it.** The two facts the
reason cites as pre-existing (no state file, no external parquets) are genuinely
pre-existing; they are the preconditions on which the new breakage sits.

The test drives `_cli_process_single.main`, which calls `publish_image_success`
(`_cli_process_single.py:802` and `:956`). **At `9480dd5b` that wrote
`progress/image_complete/in/plate.json`** — the diff of this very commit is the evidence,
since it is the `atomic_write_json(marker_path, marker, ...)` block that was removed. So
before P3, arm 1's glob found a marker, `valid_image_success` read that same file and
returned True, and the function returned real sources. The chunk writer then chunked them
and the test's three assertions held.

After P3 the publisher writes `images/` instead, arm 1's glob is empty, and it returns
`None` — and only then does the `_scan_unchunked_parquets` fallback (correctly described
as long-empty) come into play. **Two long-standing preconditions plus one new one is a
regression caused by the new one.** The reason inverts that.

Consequences, in order of cost:

1. **The fix is not recorded anywhere.** F2's real repoint is not in the P4 plan, not in
   the deferred-consumer table, and not in this reason — which points a future reader at
   the chunk writer, where there is nothing to fix.
2. **Ownership is wrong.** "Whoever next revisits the chunk writer" is not P4, so the
   strict marker will sit through P4 and P5 with no owner.
3. **A strict xfail with a wrong cause is worse than none**, because it is now the
   record. Someone who later repoints arm 1 will get an XPASS here and have to
   re-derive why.

### Fix

Rewrite the reason to name `_cli_completion.py:824` (arm 1 of
`authorized_measurement_sources`) as the cause and P4 as the owner, or fix F2 and drop the
marker. **Do not** leave the current text — it is the one artefact that would stop the
real defect being found.

---

## F4 — MEDIUM — the CAN-23 test never reaches the CAN-23 clause

**CONFIRMED.** This is the "vacuous green" class you asked me to hunt, and it is the one
instance I found that the author did not already catch.

`tests/unit/cli/test_image_record.py:394-412`:

```python
def test_a_stage2_only_record_is_not_a_success_proof(tmp_path):
    """CAN-23, and the collapse is what makes it possible to get wrong. ..."""
    record_stage(tmp_path, "plate", "a", "stage2", {"at": "t"})
    assert not valid_image_success(
        tmp_path, dataset="plate", image_stem="a", work_id="w"
    ), "a record with no artifacts certified an image"
```

`record_stage` (`_cli_image_record.py:184-228`) writes `version`, `dataset`,
`image_stem` and `stages` — **and no `work_id`**. So in `record_rejection`
(`sdk_/_image_record.py:176-193`) the clauses evaluate in order:

- `version` → 1, passes
- `dataset` → `"plate"`, passes
- `image_stem` → `"a"`, passes
- `record_provenance(...) != "migrated"` → True, so `record.get("work_id")` is `None`,
  `!= "w"` → **returns `"record was written for a different work_id"`**

Execution never reaches `if not isinstance(artifacts, dict) or not artifacts`. **Delete
the CAN-23 clause entirely and this test still passes.**

Confirmed by execution as well as by reading — `record_rejection` called directly on the
two records:

```
stage-only  -> record was written for a different work_id
empty-arts  -> record declares no artifacts
```

The first line is the record `record_stage` writes, i.e. the one this test plants: it is
refused four clauses early. The second is the same record with a matching `work_id` and
`artifacts: {}` — proof that the CAN-23 clause does fire, and that nothing currently
makes it.

The clause is genuinely load-bearing, which is what makes the gap worth fixing rather than
the clause worth removing. With `artifacts: {}` and a *matching* `work_id`:

- `valid_image_success` (`_cli_completion.py:319-340`) — `record_rejection` returns
  `None`, the `isinstance(artifacts, dict)` guard passes for `{}`, the descriptor loop
  iterates zero times, and the function **returns `True`**.
- `_run_state._verify_image:725-780` — same, and the image is recorded `verdict="verified"`.

And `publish_image_record(..., artifacts={})` is reachable: three tests in this same file
call it that way.

### Fix

Keep the test (the stage-only case is worth pinning) and add the one that actually
isolates the clause:

```python
def test_a_record_with_a_matching_identity_and_no_artifacts_certifies_nothing(tmp_path):
    """CAN-23 in isolation. The sibling stage-2 test is rejected by the work_id
    clause and would survive this clause being deleted."""
    publish_image_record(
        tmp_path, work_id="w", dataset="plate", image_stem="a",
        relative_image_path="a.tif", mode="full", stages={"measured": {"at": "t"}},
        artifacts={}, attempt_id="x", lifecycle_epoch="e",
    )
    assert record_rejection(
        read_image_record(tmp_path, "plate", "a"),
        work_id="w", dataset="plate", image_stem="a",
    ) == "record declares no artifacts"
    assert not valid_image_success(
        tmp_path, dataset="plate", image_stem="a", work_id="w"
    )
```

**Related, and worth stating since you asked about clause completeness:**
`record_rejection` has **no direct unit test at all** — every assertion on it today is
indirect, through `valid_image_success` or `resolve_run_state`. `dataset` and
`image_stem` have no coverage in either direction. Against the writers it must judge, the
clause set is otherwise **complete**: `record_stage` can produce a record that passes
every identity clause, and `work_id` + `artifacts` are the only two things standing
between a stage-only file and certification. Both are present. `record_provenance`'s
`"forward"` default is the strict reading and is right — it is the value that *keeps* the
fence, and routing the comparison through the function (rather than a bare
`record.get("provenance")`) means an unrecognised value also keeps it, which is strictly
stronger than the `marker_rejection` it replaces.

**Budget: adds nothing.** One test.

---

## F5 — MEDIUM — the measure-path guard is not equivalent to the one it replaced

**CONFIRMED.** This is a defect *introduced by* one of the four fixes, not one the fix
missed.

`_cli_process_single.py:472-480`:

```python
record_path = image_record_path(output_dir, dataset_name, stem)
if record_path.is_file():
    from ._cli_recompile_tables import _republish_table_marker
    _republish_table_marker(output_dir, record_path, commit_guard=commit_guard)
```

The repoint is right; the *predicate* no longer means what it did. Before the collapse,
`image_completion_marker_path(...).is_file()` was true only after a full
`publish_image_success`, so the payload was guaranteed complete. After the collapse,
`images/<ds>/<stem>.json` also exists for a **partial** record written by `record_stage`
— and `_republish_table_marker` (`_cli_recompile_tables.py:66-73`) indexes it
unconditionally:

```python
marker = json.loads(marker_path.read_text(encoding="utf-8"))
publish_image_success(
    output_dir,
    work_id=str(marker["work_id"]),          # KeyError on a partial record
    dataset=str(marker["dataset"]),
    relative_image_path=str(marker["relative_image_path"]),
    ...
```

Partial records are a normal product of the staged engine, not a corruption case:

- `_cli_staged_slurm_worker.py:480-505` — `publish_image_success` is guarded by
  `if item.work_id:` while `write_stage3_completion_marker` at `:503` is
  **unconditional**. Empty `work_id` ⇒ record with `stages.stage3` and nothing else.
- `_cli_staged_workers.py:548-556` — `if work_id is None: write_stage3_completion_marker(...)`,
  again with no publish in that branch.
- `_cli_staged_resume.migrate_legacy_stage3_markers:391` — writes stage-3 entries for
  every already-complete item with no publish at all, and is called from the ordinary
  staged resume path (`phenotypicCLI.py:2552`), not only from `--mode migrate`.

So `--mode measure` over such a tree raises `KeyError: 'work_id'` out of
`process_single_store_measure_core` where it previously skipped cleanly.

**PLAUSIBLE, not confirmed:** I did not establish how a real `--mode measure` invocation
selects stores, so I cannot say a partial-record image is *certain* to be enumerated. The
guard is wrong regardless — it no longer encodes "this image was published".

### Fix

Ask the question the guard means:

```python
record = read_image_record(output_dir, dataset_name, stem)
if record is not None and isinstance(record.get("artifacts"), dict) and record["artifacts"]:
    _republish_table_marker(output_dir, record_path, commit_guard=commit_guard)
```

or simply `if valid_image_success(output_dir, dataset=..., image_stem=..., work_id=...)`
where a `work_id` is in hand. The same reasoning applies to
`_replace_and_republish_table` (`_cli_recompile_tables.py:129`), which is unconditional —
lower risk, because its inputs come from `authorized_measurement_sources`, which requires
a full record — but it will now raise `KeyError` where it used to raise
`FileNotFoundError` if a recovery source ever reaches it without one.

`test_the_republish_probe_names_the_record_not_the_legacy_marker`
(`test_image_record.py:718`) is an AST test asserting the *path constant*; it is honest
about being structural, and it cannot see this, because the wrong thing is the predicate
rather than the path.

**Budget: adds nothing.**

---

## F6 — MEDIUM — the bulk Stage-3 sweeps traded *n* stats for *n* reads-and-parses

**CONFIRMED.** Not a correctness bug; a cost the spec's framing hides.

The §6.1 justification is *"one JSON read replaces one read plus three `is_file()`
probes"* — true at a per-image decision point that was already reading a marker. But
`stage3_completion_exists` used to be a bare `is_file()` and is now
`read_image_record` + `json.loads` + two dict lookups
(`_cli_staged_resume.py:135-160`), and it has three **whole-inventory** callers that
previously did zero reads:

- `gui/run_console/_slurm_observer.py:1338-1354` — `_all_stage3_markers_exist` walks
  every image of every dataset in `job_metadata.json`. This is on the observer's polling
  path.
- `_cli_staged_orchestration.py:264-289` — `completed_inventory_images`, per dataset per
  poll.
- `_cli_staged_controller.py:76-98` — the retryable/terminal split, per controller round.

On a 6,000-image run on GPFS, a poll loop that was 6,000 `stat()`s becomes 6,000
`open`/`read`/`close`/`json.loads`. `_all_stage3_markers_exist` short-circuits on the
first miss, which caps the common case; the other two do not.

**PLAUSIBLE:** I have not measured it. Flagging it because the collapse's stated
efficiency argument is about the single-image path and is silent about these, and because
the observer is user-facing.

No fix proposed — the collapse is the design. Worth a sentence in the drift register so
P6, which owns the observer, does not rediscover it as a mystery slowdown.

---

## F7 — LOW — the INV-LAYER walk covers function bodies; one shape gets through

**CONFIRMED, and the part you asked about is correct.**
`tests/unit/sdk_/test_run_state_layering.py:118` uses `ast.walk(tree)`, which visits the
full tree including function bodies, so a lazy `from phenotypic._cli import x` inside a
function *is* caught. The four shapes the docstring enumerates (relative `from .._cli`,
`from phenotypic import _cli`, `from .. import _cli`, literal `import_module(...)`) are
each genuinely handled, and `_absolute_module` resolves `level` correctly. `_image_record`
is in `_MODULES`, so the new module is watched.

The one shape not covered: `import phenotypic` (legal — not the `_cli` package) followed
by `phenotypic._cli.something` as an **attribute access**. `ast.Import` sees only
`phenotypic`; there is no `ImportFrom` and no `Call` to `import_module`. It is an unlikely
way to write the violation, which is why this is LOW — but it is the only remaining hole
and it costs three lines to close (walk `ast.Attribute` chains for `phenotypic._cli`).

Also worth noting: the walk covers five named modules. `sdk_/__init__.py` and
`sdk_/_io_constants.py` are not among them. INV-LAYER as specified binds the run-state
readers, so this is a scope statement rather than a gap — but `sdk_/__init__.py` is what
the GUI actually imports, and a `_cli` import there would be reachable from every module
the walk protects.

---

## F8 — LOW — residual drift the sweep left behind

All CONFIRMED, none behavioural:

1. **`marker_rejection`'s "no caller" docstring is wrong by one.**
   `sdk_/_run_state.py:516-540` states *"THERE ARE NONE"* and names two future callers.
   `authorized_measurement_sources` arm 1 (F2) is a third, and is the one that should be
   calling it **today**.
2. **Three test docstrings still name `marker_rejection`** where the code under test now
   calls `record_rejection`: `tests/unit/sdk_/test_run_state.py:1123`, `:1135`, `:1158`.
3. **`_verify_image`'s `if provenance is not None` guard is now dead**
   (`sdk_/_run_state.py:769`). `record_provenance` never returns `None`, so
   `stage["provenance"]` is now written for every record — where the old
   `_optional_str(marker.get("provenance"))` omitted it. Harmless: the only consumer,
   `:1304`, compares `== PROVENANCE_MIGRATED`.
4. **`PROVENANCE_MIGRATED` has zero production writers.** `publish_image_success` has no
   `provenance` parameter, so it always takes `publish_image_record`'s `"forward"`
   default — including on the `--mode migrate` path. U-10's relaxation is therefore
   unreachable in production until P7 wires it. The code comments say so; recording it
   here because `record_rejection`'s migrated branch is consequently untested against any
   real writer.
5. **`test_a_legacy_tree_is_refused_now_that_the_gate_is_armed`
   (`test_schema_gate.py:878`) re-imports `pytest` inside the function body** while it is
   already imported at module scope. Cosmetic.
6. **The recompile tripwire's `reason` under-names its own modules.** It cites
   `_cli_recompile_recovery.py` (5 sites) and `_cli_recompile_slurm_scripts.py:557` — six
   — against a stated population of seven. The seventh is
   `_cli_recompile_tables._standalone_marker_sources:142`, which globs
   `progress/image_complete/*/*.json` and gates on `valid_image_success`, so like F2(a) it
   now returns `{}` on every tree.
7. **`_replace_and_republish_table`'s new comment says the legacy marker "is absent on a
   forward tree, so `_republish_table_marker` would read a file that is not there."**
   True — but the same function's *lock* still resolves through
   `image_completion_marker_path`, four lines earlier in the call (F1). The comment
   documents half of what that function does with the legacy path.

---

## F9 — MEDIUM — `SCHEMA_GATE_ARMED`'s docstring says the opposite of its own value, and of the paragraph below it

**CONFIRMED.** Raised by the lead while fixing F2; I verified it against `1cc6740c` and
found the defect is larger than the pointer that surfaced it.

`sdk_/_schema_shape.py:67-76` — the *opening* of the constant's docstring, which is where
any reader's explanation of it begins:

```
#: Whether a ``CONVERT`` verdict may be **surfaced**. **Flipped to ``True`` by
#: P3 Task 2**, in the same commit that makes ``publish_image_success`` write
#: the consolidated record -- see
#: ``test_the_gate_is_armed_exactly_when_the_forward_path_stops_writing_the``
#: ``_legacy_marker``, which fails the moment those two disagree.
#:
#: It is ``False`` today because the legacy shape and the **current** shape
#: still overlap: the forward path writes ``image_complete/`` and writes
#: ``datasets.<ds>.completed``, so **two** of the five signals below fire on a
#: tree the running build has just written.
```

**Three false statements, and they compound:**

1. **"Flipped to `True` by P3 Task 2"** — it was not. `SCHEMA_GATE_ARMED: bool = False`
   at `:144` of the same file, and the paragraph at `:114` — added by *this commit* —
   opens "**STILL DISARMED after P3, by user ruling**". The docstring contradicts itself
   across forty lines, and the stale half comes first.
2. **The test pointer does not resolve.** Assembled across the wrap it reads
   `test_the_gate_is_armed_exactly_when_the_forward_path_stops_writing_the_legacy_marker`;
   the test is `..._stops_writing_markers` (`test_schema_gate.py:941`, the only match in
   `tests/`). A reader who greps the cited name finds nothing.
3. **"the forward path writes `image_complete/` and writes `datasets.<ds>.completed`, so
   two of the five signals fire"** — both halves were made false *by this commit*. D1
   stopped the first; §4.2's demotion stopped the second.
   `test_a_tree_this_build_wrote_needs_no_conversion` asserts the exact opposite
   (`requires_conversion(root) is None` — **zero** signals), and it passes in the census.

Item 3 is the one with teeth, because it is the paragraph a P7 engineer will read when
deciding whether arming is safe. It tells them two signals still fire on current output —
which, if believed, says arming is unsafe and the P7 task should be deferred again. The
correct answer is in the tree and in a passing test; it is just not in the docstring that
purports to explain the flag.

**Why it survived:** the commit *appended* an accurate, careful correction at `:114-140`
rather than editing the now-false text above it. That is the same shape as the drift the
register tracks — a true-when-written claim that nobody re-read after the thing it
described moved — occurring inside the very paragraph that documents two earlier
instances of it.

### Fix

Rewrite `:67-76` rather than appending to it: the flag is `False`, P3 did not arm it,
zero signals fire on a tree this build writes, and the test is
`..._stops_writing_markers`. Deleting the stale sentences is most of the work — the
correction below them is already right.

**Budget: adds nothing.** Documentation only.

---

## Things I checked and found correct

Recorded so a later phase does not re-audit them.

- **`publish_image_record` merges `stages` rather than replacing** (`:175-176`), and
  `test_publishing_merges_stages_rather_than_replacing_them` would fail if it did not.
  The ordering hazard it guards is real: both worker paths call `publish_image_success`
  *before* `write_stage3_completion_marker`.
- **`record_stage` and `consume_stage` are read-merge-write and preserve siblings.**
  `consume_stage` is idempotent and its bool return is honest, including for an absent or
  unreadable record.
- **`remove_stage3_completion_marker` preserves the semantics of the `unlink` it
  replaced** — the stage-3 marker and the completion marker were separate files and
  removing one never removed the other; consuming only the `stage3` entry reproduces that
  exactly. One behavioural difference, negligible: the old code entered
  `publication_commit(commit_guard)` unconditionally, so a revoked guard raised even on a
  no-op; `consume_stage` returns `False` before consulting the guard when there is nothing
  to remove. `clear_downstream_artifacts_for_stage1` calls `delete_stage2_token` first,
  which still enters `publication_commit` unconditionally
  (`_cli_stage2_token.py:134`), so the fence still bites first at every real call site.
- **`ArtifactWorld._write_success_marker` now merges `stages` and asserts
  `record_rejection(...) is None`** (`tests/unit/cli/conftest.py:373-416`). The
  lost-update it used to have was real and is fixed, and the `is None` assertion is the
  right kind of guard — it keeps the branch failing loudly if the *shape* ever becomes
  the reason for rejection instead of the missing artifact.
- **I looked for other hand-built fixtures that copied the contract without its
  guarantees.** Two exist and neither is dangerous today:
  `test_cli_completion_store.py:119` (`legacy_file_marker`) omits `stages` and
  `provenance` but is bracketed by a positive control (`assert valid_image_success(...)
  is True`) before the mutation, so it cannot go vacuous silently; and
  `test_schema_gate.py:300,369` plant records with no `version`/`dataset`/`image_stem`,
  which every reader would reject — harmless only because signal 1 is a directory probe
  and never reads record content. If a future signal validates record *contents*,
  `_build_converted` ("what migrate leaves behind") becomes a lie. Worth one line in the
  drift register.
- **The bridge trio's vacuous green was found and fixed by the author.**
  `test_the_refresh_bridge_tolerates_a_store_descriptor`
  (`test_cli_completion_store.py:480`) asserted `refreshed == 0`, which held trivially
  once the bridge skipped every image; the spy positive control (`assert reached`) now
  makes it fail for the real reason. That is exactly the class you asked me to hunt, and
  the author got there first.
- **Every strict tripwire I traced fails for the reason its `reason` states**, verified
  against source: the two GUI ones against
  `results_viewer/_output_consistency.py:522-528` (`_string_list` returns `None` on the
  absent key ⇒ `(None,)*4` ⇒ `total == 2` fails) and
  `gui/shell/_runs_registry.py:1124-1129,1163` (`_string_set` raises `TypeError` on the
  absent key, caught and reported as "unreadable processing state"); the recompile ones
  against `_cli_recompile_recovery.py:387` etc.; the migrator ones against
  `_hdf_to_zarr._republish_image_marker`. Both GUI tripwires `pytest.skip` rather than
  `xfail` when the extra is missing, so an `ImportError` cannot be absorbed by the marker
  — which is the right shape. **The exception is F3.**
- **The census confirms all of that at runtime, for the eight files it covered: 13 XFAIL,
  0 XPASS, 489 passed.** Each XFAIL line reports its own `reason`, and every one matches
  the mechanism I traced. No marker is stale, and none of the eight files has a tripwire
  that has already started passing. `test_staged_resume_parity.py` passes, so the
  stale-record positive control the author added is live rather than merely written.
- **`test_the_gate_is_armed_exactly_when_the_forward_path_stops_writing_markers` fires in
  both directions.** `writes_legacy_marker is not SCHEMA_GATE_ARMED` is `False is not
  False` today ⇒ xfail; it XPASSes both when P7 arms the gate *and* if someone
  reintroduces the legacy write while it stays disarmed. Strictness makes both a failure.

---

## What I could not close

- **F5's reachability.** Whether `--mode measure` enumerates an image whose only record
  is a stage entry. The guard is wrong either way.
- **F6's magnitude.** Unmeasured.
- **Nine of the 35 recompile marks — root cause not established.** Eight fail through a
  handled error; the ninth is a `DID NOT RAISE` whose success criterion is inverted, and
  which is currently leaving the property it is named for unverified. See the section
  below — this is the one item P4 has to close rather than inherit.
- **`tests/migration`'s 57 goldens.** Excluded as instructed; not examined.

---

## The recompile deferral: 26 of 35 confirmed, 9 not

Run under `--runxfail --tb=line` over `test_cli_recompile.py` and
`test_cli_recompile_slurm.py`: **45 passed, 35 xfailed, 0 xpassed.**

**Why not `-rxXf`, which is what I first asked for and was wrong to.** That flag prints
the `reason` string the author wrote, not what the test did — so all 35 lines recite
*"--mode recompile reads the legacy `image_complete/` marker until P4…"*, which is the
claim under test. Reading it as confirmation is circular. The reason is a claim; the
traceback is the evidence, and `--runxfail --tb=line` is what pairs a location with an
exception.

| count | exception |
|---|---|
| **26** | `FileNotFoundError: [Errno 2] .../.phenotypic/progress/image_complete/ds/img.json` |
| 3 | `RuntimeError: Cannot safely restore measurement authority for ds/img` |
| 3 | `RuntimeError: Cannot restore marker authority: a non-overlay artifact changed` |
| 1 | `RuntimeError: Cannot safely restore marker authority for ds/img` |
| 1 | `Failed: Regex pattern did not match.` |
| 1 | `AssertionError: {'error': 'Ru…', 'status': 'failed'} == {…'status': 'completed'}` |

**26 confirm the stated mechanism exactly** — a raw `FileNotFoundError` with the
`image_complete/` path in the message.

**Nine do not.** They fail one layer up, through *handled* domain errors. Those messages
are **consistent** with the missing marker being the root cause — a caller that catches
the absence and re-raises as "cannot restore authority" produces exactly this — but
consistent is not confirmed, and asserting the cause from the message text is the same
inference that produced F3's inverted attribution. **Recorded as: fails, mark is not
stale, root cause not established.**

Those nine are where the concern behind this whole section bites hardest. A strict xfail
that fails *incidentally* stays XFAIL after P4 repoints the reader, so the tripwire never
fires and nobody learns the deferral was discharged. The 26 are safe — repoint the reader
and the `FileNotFoundError` cannot survive. **P4 must confirm the raise disappears for
the nine rather than assuming it**, and should treat a still-XFAIL among them after the
repoint as an unexplained failure, not as a marker to leave in place.

### The nine, and the one that is not like the others

Paired to their exceptions from the junit XML rather than by eye.

**Eight fail through a handled error — "too much failure":**

| test | exception |
|---|---|
| `test_finalizer_overlay_refresh_locks_store_before_lifecycle` | `RuntimeError: Cannot restore marker authority: a non-overlay artifact changed` |
| `test_finalizer_refreshes_nested_overlay_repair_authority` | same |
| `test_overlay_refresh_holds_generation_guard_only_for_marker_commit` | same |
| `test_local_recompile_restores_deleted_overlay_marker_and_master` | `RuntimeError: Cannot safely restore marker authority for ds/img` |
| `test_measurement_worker_refreshes_marker_with_active_slurm_generation` | `RuntimeError: Cannot safely restore measurement authority for ds/img` |
| `test_recoverable_overlay_and_table_share_one_slurm_task` | same |
| `test_slurm_recompile_schedules_table_bound_to_missing_overlay` | same |
| `test_slurm_overlay_worker_restores_marker_authority` | `AssertionError: {…'status': 'failed'} == {…'status': 'completed'}` |

For these eight the success criterion is the same as for the 26: **repoint the reader and
the raise must disappear.**

**The ninth is the opposite shape and needs its own line in the P4 brief:**

```
test_stale_slurm_overlay_worker_does_not_publish_rendered_bytes
    Failed: DID NOT RAISE SlurmGenerationInactiveError
```

Every other failure in this population is *too much* failure. This one is *too little* —
a guard that was expected to fire and did not. Its success criterion is therefore
**inverted: repointing must make `SlurmGenerationInactiveError` appear.** A repoint that
leaves it still not raising is a live defect, not a stale marker. Folded into "the nine",
an implementer who fixed the other eight would reasonably call the batch done.

(The class exists only as `SlurmGenerationInactiveError`,
`_cli_slurm_lifecycle.py:47` — there is no `SlurmGenerationError` to grep for.)

**And there is a second thing to check on this one, which the marker currently hides.**
Reading the test (`test_cli_recompile_slurm.py:2862-2905`): it deletes the overlay,
initializes a SLURM lifecycle, deactivates the generation, and then asserts two separate
properties —

```python
with pytest.raises(SlurmGenerationInactiveError):
    _run_overlay_task(..., {"restore_marker_authority": True, ...},
                      slurm_generation=generation)

assert not overlay.exists()
```

When the block does not raise, `pytest.raises` fails at the `with` exit and **the second
assertion never executes.** So the property this test is *named for* — that a stale
worker does not publish rendered bytes — is currently not being checked at all, by this
test or any other, and the strict marker means nobody sees that it isn't.

The likely mechanism is that the missing marker short-circuits `_run_overlay_task` before
it reaches the lifecycle fence, in which case the fence is not *silenced* but never
reached — and the no-publish outcome may well still hold, by a different route. That is
**consistent, not confirmed**, and the same discipline applies as for the other eight. But
the direction matters: a code path that returns early past a generation fence is the one
failure mode that is silent in production, so P4 should assert **both** halves when it
repoints — the error appears, *and* the overlay is still absent — rather than treating the
restored raise as sufficient.

### On the count

The population is **28 decorations across the two files** (26 + 2, measured) and **35
test instances at runtime** — five of the decorated tests are parametrized, which
accounts for the gap. My earlier "30" was neither: I read this repo's own marker comment
("the 28 marks in `test_cli_recompile{,_slurm}.py`") as a slurm-only figure and added the
two from the sibling file, when the brace expansion already covered both. Recording it
because the same shape — a stated total nobody re-measured — is what the drift register
exists for, and because a decoration count and an instance count are different questions
that the word "marks" hides.

### F2's fix discharged none of them

**0 XPASS.** The possibility was raised that F2 might route some of these into
`_cli_recompile_tables.py:238`'s *"Legacy external measurement Parquets require --mode
migrate"* branch and change their outcome. It did not — all 35 still fail. The deferral
is intact and still entirely P4's.

---

## Disposition

Recorded because it changes what a P4 reader should expect from the tree, and because two
of these outcomes are themselves evidence about the findings.

Five fixes landed while this review was in progress, verified at **152 passed, 10
skipped, 4 xfailed**. I have not confirmed which five — that is the lead's tally, not a
measurement of mine, so a P4 reader should check the tree rather than assume this
report's numbering maps onto it. Two were named explicitly, and both are evidence:

- **F3's marker was deleted, not reworded**, and the checkpoint test now passes. That is
  the outcome the attribution predicted: if the failure had been pre-existing, repointing
  the legacy arm would have left it red. It went green, which confirms the finding from
  the other direction.
- **`test_the_republish_probe_names_the_record_not_the_legacy_marker` was rewritten** from
  a substring check into an `ImportFrom`/`Name`/`Attribute` walk. The substring form
  asserted `"image_completion_marker_path" not in <source>`, which cannot distinguish a
  *use* from a *mention* — so it forbade the module from explaining, in a comment, the
  probe it had just replaced. F5's fix needed exactly that comment. Worth noting as a
  small instance of the report's recurring theme: the test was structural on purpose and
  said so, but the structure it checked was the file's text rather than its syntax.

**Which of F5–F8 remain open is not established here.** F5 is the one that still changes
behaviour if it does; F6 and F7 are notes rather than defects.
