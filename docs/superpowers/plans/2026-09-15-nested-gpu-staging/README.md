# Nested `GpuDetector` staging — two plans, in sequence

**Plan A lands first. Plan B follows it.** They are not alternatives to choose
between: A is the shortest path to a working `F1gfd5` run, and roughly 70% of it
is foundation that B builds on.

| | **Plan A** — composite path | **Plan B** — phase protocol |
|---|---|---|
| File | `plan-a-composite-path.md` | `plan-b-phase-protocol.md` |
| Spec | `../../specs/2026-09-15-nested-gpu-staging/design.md` | **none yet** — must be written |
| Independent review | ✅ 20 findings, all applied | ❌ not reviewed |
| Operation interface | unchanged | `_phases`, `_child_order`, `_phase_state` on every op |
| Tasks | 16 | 7, but each is larger, and a spec + review come first |
| Supports | `ImagePipeline`, `CompositeDetector`, `CompositeEnhance` | any op that declares phases |
| `FilamentousFungiDetector` | refused | stageable (Task B6) |
| `TwoKFilamentousDetector` | refused | still refused — mixed child order |
| GPU detectors per pipeline | 1 | 1, with `N > 1` additive |
| Prefix CPU work | runs twice (Stage 2 + Stage 3) | runs **once** |
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
| 1 shared traversal · 3 tree-wide detection · 4 step-path descent · 6a slot-keyed signal · 9 path invariant · 10 equivalence gate · 11 process mode · 12 continuation digest · 14 docs · 15 regression | **reused** |
| 5 path-shaped `StagePlan` · 6 `ReplayDetector` · 7 branch prefix · 8 stub substitution · 13 forward prefix | superseded |

The discarded part is the splitter and replay machinery — the pieces that exist
*because* Plan A has to infer an operation's internal structure from outside.
Plan B removes the need to infer, so those go; everything else stands.

## What B buys, when it comes

- **No inference.** Declaring `_phases` + `_child_order` *is* the opt-in, so the
  `_CHILD_CONTRACT` table, the "composition primitives only" rule and the guard
  test over class kinds all disappear.
- **No re-execution.** Each phase runs exactly once. Plan A runs a branch prefix
  twice; measured, `FocusEdgePhase` at production size is **71 s and 7.4 GB peak
  per image**, so that doubling is 669 CPU-hours on a 33,923-image run.
- **No nesting blind spot.** `needs_gpu` becomes a scan of the flattened phase
  list, so the bug this whole effort exists to fix stops being *possible* rather
  than being fixed.
- **Arbitrary containers become stageable** by splitting their `_operate`, rather
  than being admitted or refused by a rule about what kind of class they are.

## Sequencing

1. Plan A, tasks 1 → 15.
2. Write Plan B's spec; run it through an independent review.
3. Plan B, reusing A's foundation.

Nothing in A needs to be undone to start B.
