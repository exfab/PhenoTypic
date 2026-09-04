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
| `p1_task3_verification_cache.py` | P1 Task 3 Step 6 — twelve mutations over `sdk_/_verification_cache.py`, covering all fifteen INV-VERDICT tests. |
| `check_mutation_coverage.py` | Read-only. Every harness's name-integrity and coverage gate, without the pytest runs behind them. |

## Adding a test to a covered suite

Run `check_mutation_coverage.py`. If it names your test under
**NOT covered by any mut**, add a mutation for it before you commit.

This is not bookkeeping. An unproved test looks exactly like a proved one from
the outside: it passes, it reads as a guard, and it is discovered to have been
guarding nothing only when someone finally breaks the code it named and the
suite stays green. A mutation suite decays one well-intentioned addition at a
time, and this is the check that stops it.

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
