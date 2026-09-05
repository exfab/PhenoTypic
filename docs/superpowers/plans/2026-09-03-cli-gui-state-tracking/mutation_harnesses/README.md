# Mutation harnesses

Each phase doc that says *"prove each test can fail"* has its harness here, one
file per suite. A harness reintroduces one specific bug at a time and asserts
that a named test — and, where the mutation is surgical, **only** that test —
goes red.

**Why here and not `docs/superpowers/logic_validation_scripts/`.** That
directory's contract is that nothing in it imports `phenotypic`, which is what
makes anything in it an independent witness of a numeric claim. A mutation
harness is the opposite kind of artifact: it must edit the shipped source and
run the real suite against it. The plan
[README](../README.md#spikes) draws the same line for the spikes, for the same
reason.

## Files

| File | What it does |
|---|---|
| `p1_task3_verification_cache.py` | P1 Task 3 Step 6 — thirteen mutations over `sdk_/_verification_cache.py`, covering all twenty in-process INV-VERDICT tests. |
| `p2_task0_disk_verification_cache.py` | P2 Task 0 (U-11) — twenty mutations over three targets, covering all twenty-eight on-disk tier tests, including §9.1's six corruption cases. |
| `p2_task1_restart_epoch.py` | P2 Task 1 — ten mutations over **four** targets, covering all eleven `restart_epoch` and rule-2 fence tests. |
| `check_mutation_coverage.py` | Read-only, no pytest. Name integrity, coverage, and anchor drift for every harness here. |

## Run `check_mutation_coverage.py` after touching either side

It exits non-zero, so it works as a phase-gate step rather than a habit. Three
things it catches, each of which is invisible from a green suite:

**A test no mutation claims.** An unproved test looks exactly like a proved one
from the outside: it passes, it reads as a guard, and it is discovered to have
been guarding nothing only when someone finally breaks the code it named and
the suite stays green. A mutation suite decays one well-intentioned addition at
a time.

**A mutation naming a test that does not exist.** A typo reports `NOT PROVED`,
which reads as a weak test — so the investigation starts at the test rather
than at the harness. Same shape as `F822` in `__all__`: a name asserted against
nothing.

**A drifted anchor.** Refactor the target and a mutation's `old` text stops
matching, so the harness prints `SKIPPED` for it. That reads as *not run*
rather than *not proved*, and it is easy to skim past in a twelve-row report —
at exactly the moment nobody is thinking about the harness.

## One source term, two suites: record it or lose it

A harness binds **one suite** — `check_mutation_coverage.py` would report a test
name from another file as a typo. But a single source term can be the subject of
tests in two suites, and then the proof for one of them lives in the *other*
harness, where nothing connects the two.

There is exactly one such term today, `_run_state.py:1103`:

```python
    performed: Depth = (
        "shallow"
        if requested_depth == "shallow" and warm is not None and not escalated
        else "deep"
    )
```

Delete `and not escalated` and **three** tests fail, across both suites:

| Suite | Test |
|---|---|
| `test_verification_cache.py` (tier 1) | `test_a_tampered_artifact_falls_through_even_with_a_warm_cache` (`:580`) |
| `test_verification_cache_disk.py` (tier 2) | `test_a_moved_stat_tuple_falls_through_to_deep` |
| `test_verification_cache_disk.py` (tier 2) | `test_an_absent_entry_falls_through_to_deep` |

Only the second and third are claimed by a mutation — `p2_task0`'s *"a warm cache
reports `depth=shallow` even when part of the pass was deep"*. **The first is
proved by that same mutation and by nothing in its own harness.**

**Why the P1 harness was not converted to multi-target to cover it.** It could be:
`TARGETS` exists and `p2_task0` uses it. But a second mutation of the same line
from a second harness is duplicate evidence that *reads* as independent — the
coverage output would show one source term under two harnesses with nothing saying
they are the same term. That is the failure this whole directory exists to prevent,
imported into the gate itself.

So: one mutation, and this table. The obligation the table creates is that
**deleting or weakening `p2_task0`'s escalation mutation silently unproves a tier-1
test too.** Check here before touching it.

*(Measured, not assumed: the mutation was applied to `_run_state.py` and both
suites run against it — 1 failed / 19 passed and 2 failed / 26 passed — then the
file restored and its hash confirmed.)*

## Predict the outcomes BEFORE running the harness

Say what each mutation should take down, in writing, before the run. It is not
ceremony -- it is a cheaper check than the run itself, and it catches a class the
run cannot.

**Task 1 is the worked example.** Writing the expected table found three defects
with nothing executed, against a suite where all 323 tests passed:

- **A mutation claimed a test that would not have failed.** Deleting the writer's
  epoch stamp degrades the record to ``0``, and ``0 >= 1`` is still False, so the
  run stays fenced -- the right answer for the wrong reason. It would have come
  back ``NOT PROVED`` and sent the investigation to the fence, which was correct,
  rather than to the expectation, which was not.
- **A second mutation claimed the same test**, whose record *has* the field, so
  the fallback branch it mutates never runs.
- **Chasing what would actually catch that one found a real hole.** The
  missing-epoch default was tested in one direction only:
  ``test_a_record_without_an_epoch_still_counts_on_an_unrestarted_run`` is
  satisfied by ``sys.maxsize`` exactly as well as by ``0``. A default degrading
  *upward* passes it while believing every stale authority on every restarted
  run -- the precise failure rule 2 exists to prevent. The docstring claimed both
  directions; one was untested.

**Be precise about what could and could not have found the third.** The run
*would* have found it: the mis-expecting mutation was in the harness, would have
reported ``NOT PROVED``, and asking why no test catches an upward-degrading
default leads to the same hole — one freeze cycle later, with the mutation as
prime suspect rather than the missing test. Prediction found it **cheaper and
more legibly**, not exclusively.

**``COVERAGE_OK`` could not have found it at all**, and that is the structural
claim. The gate maps tests to mutations; an untested *direction* of a
two-directional claim is invisible to it, because the test exists, it is claimed,
and the row is green. ``test_a_record_without_an_epoch_still_counts_on_an_unrestarted_run``
is satisfied by ``sys.maxsize`` exactly as well as by ``0``, and no amount of
coverage checking sees that. Prediction is where it surfaces, because writing
*"this mutation fails that test"* forces you to trace the branch, and a branch no
test enters is obvious the moment you try.

*(An earlier draft of this paragraph said the run "structurally could not have
found" it. That was wrong in the direction this file exists to prevent —
overstating what a check does — and was corrected by the author of the harness
before it shipped.)*

Same round: two absent-field tests used ``del record["restart_epoch"]``, which
**raises** when the writer stops stamping -- coupling them to an unrelated
mutation and making them fail for a reason other than what they assert.
``record.pop(..., None)`` is the shape that matches the subject, *a record with
no epoch*.

## Controls: declared, never inferred

A **control** fails when the implementation becomes *too eager* —
`test_a_clean_tree_carries_no_advisories` exists to catch an advisory firing
spuriously. It is proved by the **absence** of a mutation making it fire, so no
mutation will ever claim it, and demanding one forces a false choice: a
permanently red gate, or contrived mutations written to satisfy this script
rather than to catch a bug.

The second is much worse. It degrades the signal for everyone afterwards, and
the next reader inherits mutations whose only purpose was a green number.

So a harness declares them:

```python
CONTROLS = (
    "test_a_clean_tree_carries_no_advisories",
    "test_a_matching_metadata_snapshot_raises_no_advisory",
)
```

Declared controls are excluded from the coverage requirement and **printed on
their own line**, not silently exempted — an undeclared exemption is exactly how
a real gap hides behind a green gate, which is the thing this file exists to
catch. And a name in `CONTROLS` that is not in the suite is an error, just as a
typo in an expected-test list is, so the escape hatch cannot rot either.

## Scope: the checker sees committed harnesses only

It globs `mutation_harnesses/*.py`. A harness living in a scratchpad is invisible
to it, so **"the checker is green" does not mean "every anchor in this change is
validated"** — it means every anchor in every *committed* harness is. If you are
mutating a target whose harness is not in this directory, that target is
unwatched. Commit the harness.

### What is unwatched right now, named

`COVERAGE_OK=True` answers *"is every test claimed by a committed harness
proved?"* — not *"is every test in this change proved?"* The second question has
a different answer, and leaving it implicit is the exact failure this file's own
Scope rule warns about, applied to itself.

| Suite | Watched by |
|---|---|
| `tests/unit/sdk_/test_verification_cache.py` (20) | `p1_task3_verification_cache.py` |
| `tests/unit/sdk_/test_verification_cache_disk.py` (28) | `p2_task0_disk_verification_cache.py` |
| `tests/unit/cli/test_run_identity.py` (11) | `p2_task1_restart_epoch.py` |
| **`tests/unit/sdk_/test_run_state.py` (62)** | **nothing** |
| **`tests/unit/sdk_/test_run_state_layering.py` (2)** | **nothing** |
| **`tests/unit/cli/test_schema_gate.py`** | **nothing** |

`test_run_state.py` is **P1's largest suite and the one that pins INV-VERDICT's
run-level half** — rule 1's six comparisons, the live-authority ladder, the four
advisories. It is unwatched, so a mutation in `resolve_run_state` that no test
catches would be reported by nothing here.

That is a deliberate scope choice, not an oversight: the two committed harnesses
cover the module whose *bug class* is silent (a cache that hands back a stale
`complete`), and a 62-test harness is a phase of work in itself. But it must be
written down, because the alternative is a future reader taking `COVERAGE_OK=True`
for a claim it does not make.

**Evidence it is a real gap, not a theoretical one.** The `publication_id` param
of `test_each_of_rule_ones_comparisons_is_load_bearing` did not exist until
`b54a9613`: rule 1 binds six fields and the suite enumerated five, so deleting
`_run_state.py`'s `publication_id` comparison left the whole suite green.

A harness over this suite would not have found that either — nothing claimed the
missing param — **and neither would the checker, which is blind to params.**
`defined` comes from AST `FunctionDef` names (`:185`) and `named` strips the
bracket (`:214`), so a five-case and a six-case parametrization are byte-identical
to it. A missing `pytest.param` is precisely the gap it cannot see.

What would have found it is **writing the mutations**: enumerating rule 1's
comparisons one at a time against the source is what makes a six-field rule with
five cases visible. That is the argument for eventually declaring this suite — and
it belongs to the human step, not the tool.

*(An earlier draft of this paragraph claimed the checker would have caught it. It
would not, and the claim was corrected before it shipped — in the file whose job is
to be trustworthy about exactly what each check verifies.)*

## Two rules the P1 run produced

**Do not edit a target while its harness runs.** A harness holds the pristine
source in memory for its whole run and writes it back after every mutation, so
an edit arriving mid-run is silently reverted at the end. The instinct is to
guard against a harness that *fails* to restore; the live hazard is one that
restores **too well**, over work that arrived after it started — and a hash
check catches that only afterwards, as an unexplained mismatch. Announce start
and finish when anyone else is working in the tree.

**A suite run by someone else mid-mutation is not a suite result.** Most
mutations here are surgical by design, so the tree under one of them fails
exactly one test — indistinguishable, from outside, from a genuine one-test
regression. Both of those cost a round trip during P1 before the rule existed.
