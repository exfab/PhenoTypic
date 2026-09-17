# Nested `GpuDetector` staging — two plans, in sequence

**Plan A lands first. Plan B follows it.** They are not alternatives to choose
between: A is the shortest path to a working `F1gfd5` run, and part of it is
foundation that B builds on. Which parts, per task, is in the table under *Why A
is not wasted work*.

> **Status (2026-09-17).** Plan A is implemented (PR #224) and passed a real-GPU
> smoke run (`../../reports/2026-09-15-nested-gpu-staging/gpu-smoke.md`). Plan B
> was re-checked against the built code in
> `../../reports/2026-09-15-nested-gpu-staging/plan-b-staleness-review.md`, and
> it is **blocked on design decisions** (§4 of that report). Most important: for
> `F1gfd5` the built Plan A already runs every operation once, so Plan B's
> headline saving does not apply to the driver pipeline.

| | **Plan A** — composite path | **Plan B** — phase protocol |
|---|---|---|
| File | `plan-a-composite-path.md` | `plan-b-phase-protocol.md` |
| Spec | `../../specs/2026-09-15-nested-gpu-staging/design.md` | **none yet** — must be written |
| Independent review | ✅ two plan reviews + an implementation review | ⚠️ staleness review only; design blocked |
| Operation interface | unchanged | `_phases`, `_child_order`, `_phase_state` on every op |
| Tasks | 16 | 7, but each is larger, and a spec + review come first |
| Supports | `ImagePipeline`, `CompositeDetector`, `CompositeEnhance` | any op that declares phases |
| `FilamentousFungiDetector` | refused | stageable (Task B6) |
| `TwoKFilamentousDetector` | refused | still refused — mixed child order |
| GPU detectors per pipeline | 1 | 1, with `N > 1` additive |
| Prefix CPU work | a non-empty in-branch prefix runs twice (empty for `F1gfd5`) | runs **once** |
| Scheduling | static, from the split | static, from the flattened phase chain |

## Why A first

**It is the only one that fixes something today.** `pipeline_requires_gpu`
returns `False` for the driver pipeline, so a 33,923-image run is routed to CPU
partitions and reports success with different numbers. That is live now.

**It is reviewed.** Plan A has been through an independent review that found five
blockers — two of them defects in the *spec*, including a provenance mitigation
that reproduced the bug it was written to prevent. Plan B has had none of that
scrutiny, and it is the larger change.

**B needs a spec first.** The committed spec describes A. B's design currently
lives only in its own plan preamble, which is not the same thing.

## Why A is not wasted work

| Plan A task | Under B |
|---|---|
| 1 shared traversal · 2 tune consolidation · 15 regression | **reused** |
| 3 tree-wide detection + refusal | **partly reused**: detection, the slot refusal and the refusal surfaces stay; `_CHILD_CONTRACT` / `validate_ancestor_contracts` are replaced |
| 4 per-branch step path | segment scheme reused; the mechanism only if phases run inside `apply()` |
| 6a slot-keyed signal | reused; `staged_detector_slot` and every `plan.gpu_path` call site are repointed |
| 9 path invariant | the resolvability test is reused; the gpu-path test is rewritten |
| 10 equivalence gate | patterns reused, shapes rewritten, prefix tests retired |
| 11 process mode · 14 docs | **rewritten** (process export uses `build_replay_pipeline`; docs and their pin tests describe replay) |
| 12 continuation digest | reused for process mode; full-run invalidation is a new decision |
| 5 path-shaped `StagePlan` · 7 branch prefix · 13 forward prefix | superseded |
| 6 `ReplayDetector` · 8 stub substitution | superseded only if parents stop being replayed (staleness review F-2) |

The discarded part is the splitter and replay machinery — the pieces that exist
*because* Plan A has to infer an operation's internal structure from outside.
Plan B removes the need to infer, so those go; everything else stands.

## What B buys, when it comes

- **No inference.** Declaring `_phases` + `_child_order` *is* the opt-in, so the
  `_CHILD_CONTRACT` table, the "composition primitives only" rule and the guard
  test over class kinds all disappear.
- **No re-execution.** Each phase runs exactly once. Plan A runs a *non-empty*
  in-branch prefix twice. Measured, `FocusEdgePhase` at production size is
  **71 s and 7.4 GB peak per image**, so that doubling *would* cost 669 CPU-hours
  on a 33,923-image run — but in `F1gfd5` it is a Stage-1 op, the prefix is
  empty, and nothing runs twice. The saving is hypothetical for today's
  pipelines.
- **A declaration instead of a class-kind rule.** Plan A as built is also
  tree-wide (slots included), so this changes where the answer comes from, not
  what it covers.
- **Arbitrary containers become stageable** by splitting their `_operate`, rather
  than being admitted or refused by a rule about what kind of class they are.

## Sequencing

1. Plan A, tasks 1 → 15.
2. Write Plan B's spec; run it through an independent review.
3. Plan B, reusing A's foundation.

Nothing in A needs to be undone to start B.
