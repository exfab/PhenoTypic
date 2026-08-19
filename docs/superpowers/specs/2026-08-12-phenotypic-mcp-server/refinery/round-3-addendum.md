# Round 3 — reviewer addendum

Read `brief.md` (round-0 context) and `ledger.md` (**now carries USER-1..24**;
all permanent) before anything else. This file carries only what changed.

## Scope: `snapshots/round-2-spec.diff` (1367 lines) + `round-2-plan.diff` (103)

Unchanged sections were reviewed in rounds 1–2 and are out of scope.

## THE SEVERITY RATCHET IS NOW BINDING

**After round 2, no new sub-Critical concerns.** Raise **Critical and Major
only**. Anything smaller you notice: list it in one line under an `Advisory`
heading, without argument. Depth over breadth — I would rather have three
airtight Majors than fifteen items.

## Your first job is verification, not discovery

Round 2 applied 24 user rulings and ~40 concerns. **Check that the resolutions
are real**, and specifically check the defect the whole panel found last round:

> Four reviewers independently found that an applied fix had left the section
> arguing the *old* position untouched. That happened four separate times in
> round 1 (PROP-1..5 in the ledger).

So for every change in the diff, ask: **what else asserted the thing this
replaced, and was it updated?** That is the highest-yield question available to
you this round. If round 2 repeated round 1's mistake, I want to know.

## What round 2 decided (do not re-litigate; verify instead)

- **USER-24 — grouping offloaded to the agent.** `group_by`, per-group trait
  overrides, and the per-group cost breakdown were all **deleted**. One campaign
  carries one `subset_id` and `user_named` is first-class, so one-subset-per-group
  makes each campaign's aggregate cost *be* the group's cost. The server keeps
  exactly one primitive: `group_filter` on the `SubsetSelector` **ABC**. This
  **supersedes USER-19** — the scorer-persistence work is gone.
- **USER-18 / USER-21 / USER-22 / USER-23** — human gate moved to `deploy_start`;
  full scope on a group-filtered subset is `parent ∩ group_filter`;
  `human_response` unconditionally required with `ack_source` in the response;
  local `W2`/`W3` children detached.
- **USER-20 + the concurrency block** — async handlers, two executors
  (`blocking`=4, `compute`=1), slot as `asyncio.Semaphore(1)` with a wall-clock
  lease, orphans refused rather than watched, CAS on `(status, artifact_digest)`,
  fan-out as a background task, seven new error codes.
- **Measured, not assumed:** §10.6.1's header sweep is **0.081 s for 460 real
  images on GPFS** (0.18 ms/image) — see
  `logic_validation_scripts/2026-08-12-phenotypic-mcp-server/header_sweep_cost.py`.
  It runs. The `W1` class on `deploy_plan {scope:"full"}` is justified by the
  **re-probe**, not the sweep. Cold and warm came out identical, so the "cold"
  number is not truly cold — that caveat is in the spec and should stay.

## Where I am least confident

1. **The spec GREW by 672 lines this round** (5081 → 5753), in a round whose
   loudest finding was that round 1 grew while claiming to cut. Most of it is the
   concurrency block. **Is it specification, or is it narration?** SIMP-30 (open,
   advisory) argues ~35 lines of "an earlier draft…" memorials should move to the
   ledger. Does that argument now apply to more than 35 lines?
2. **The concurrency block was written by a subagent from a decision list.** I
   verified its two code citations and they are accurate. I have *not*
   independently verified that its ~430 lines are internally consistent with the
   rest of the spec.
3. **USER-24's offload.** It deleted a lot. Did it delete something load-bearing —
   the way round 1's `catalog_measurements` cut took the `header_scheme()`
   derivation with it?
4. **`deploy.approve` lineage + the token's grown binding set** (`run_name`,
   resolved `array`, `estimate.node_hours`, `argv_digest` now defined). New, and
   on the safety-critical path.

## Still open from earlier rounds, untouched by any diff

GEN-4, 5, 6, 8, 9, 10, 11, 12; FLOW-1, 2, 5; CONC-18. **CONC-8 was confirmed as a
real defect in shipped code** (`RunRegistry.allocate` holds `self._lock` across a
30 s `exclusive_path_lock`) — that one is a Phase 1b code fix, not a spec fix.

## Output

Write to `refinery/reports/round-3-<your-role>.md` **as you go** — a restart ate
an entire round's reports once already. End with severity-tagged concerns bearing
stable IDs continuing your own numbering, and `VERDICT: APPROVE` or
`VERDICT: REVISE`. **APPROVE is a real option this round** — say so if the diff
is sound. Tag spec-touching items `spec-change` and human-ruling items
`needs-user-input`.
