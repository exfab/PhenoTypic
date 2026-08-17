# Phase 1 — Execution: dependency DAG, clusters, gates

**Method:** `execute-plan-orchestration` — cluster cohesive interdependent work,
isolate broad sweeps and risky seams, one agent per cluster, gate between.

**Model policy:** every cluster and every gate runs on **Opus, high effort**. The
clusters below are deliberately sized to *use* that — each is a coherent refactor
one agent holds entirely in context, rather than a checkbox handed to a fresh
agent that must re-derive the same background. The skill's rule that a reviewer
is never weaker than the implementer is satisfied trivially as a result.

---

## Dependency DAG (derived from the plan's `Files`/`Interfaces` blocks)

```
Phase 1a
  T1 (_services pkg + purity gate)
   ├─> T3 (registry)  ─────────────────────────────> T10, T11  [phase 1b]
   ├─> T4 (sandbox)
   ├─> T5 (runs)          ── requires ── T2 (IMAGE_EXTS -> sdk_)
   ├─> T6 (_space split) ─> T7 (tune_spec consolidation)   [same target file]
   └─> T8 (argv)
  T9 (build_array_script_spec)   — independent of T1–T8 entirely (_cli, not gui)

Phase 1b
  T10 (shared module list) ─┬─> T11 (describe_operation) ─> T12 (derive_columns)
                            └─> T14 (subset/)  ── + T13 (directory_digest) ──> T17 (staging)
  T15 (screen guard) ── T16 (--slurm k=v) ── T18 (finalize)   [ALL share _run.py]
```

### Two corrections to `phase-1b-engine-prerequisites.md`

That document's header claims Tasks 10–18 are "mutually independent and may be
executed in parallel by separate agents… only Task 14 and Task 17 touch each
other." **Both halves are wrong**, verified against the code:

| Conflict | Evidence |
|---|---|
| **T15, T16, T18 all edit `tune/_tune_cli/_run.py`** — one 1051-line file | T15's guard goes before `if slurm:` (`:593`) and `if screen:` (`:623`); T16 widens the `slurm_args` chain (`:798-804`); T18 must call `_finalize_best_params` (`:705`), `_finalize_generalization` (`:907`), `_finalize_outputs` (`:945`), `_finalize_pareto_outputs` (`:982`). Parallel agents would collide on every one. |
| **T14 edits the same literal T10 lifts** | T10 turns `submodules = [` (`_serializable_pipeline.py:645`) into `PHENOTYPIC_CLASS_MODULES`; T14 must add `"phenotypic.subset"` to it. Running them in parallel means one rewrites what the other is mid-edit. |
| **T11 and T12 share `_services/catalog.py`** | T11 creates it, T12 extends it. |

The clustering below is built from the corrected DAG. **Fix the header claim in
`phase-1b` as part of C4's commit** so the two documents stop disagreeing.

---

## Clusters

| # | Tasks | Shape | Why this grouping | Parallel with |
|---|---|---|---|---|
| **C1** | T1, T2, **T2.5**, T3, T4, T5, T8 | Keystone + Leaves | Six tasks, **one idiom**: move a module into `_services`, leave a re-export shim, assert the shim is the *same object*. One agent writing all five shims produces one consistent seam; five agents produce five dialects — the exact failure the skill names. T1 opens it (the gate everything is verified by) and T2 is a 3-file prerequisite of T5. Per-task commits keep it bisectable. | — |
| **C2** | T6, T7 | Keystone | The one genuine refactor in Phase 1a: `_space.py` must split because `_setup_authoring.py:28` imports its pure symbols while the module imports Dash at `:33-34`. T7 then folds four more modules into the same destination file. Same file, same judgment call — inseparable. | — |
| **C3** | T9 | **Seam** | Isolated despite being small: it is the only `_cli` change in the phase, its whole contract is *absence of I/O*, and Phase 2C's `deploy_plan` depends on it. Risk ≠ size. | C1, C2 |
| **C4** | T10, T11, T12 | Keystone | One subject — the catalog the agent browses. T11/T12 share `catalog.py`; T10 is the enumeration both read. | C6 |
| **C5** | T13, T14, T17 | Keystone | One subject — the subset boundary: digest → selectors → staging. T17 consumes both predecessors. **Must follow C4** (T14 edits T10's constant). | — |
| **C6** | T15, T16, T18 | Keystone/Seam | Forced: all three edit `_run.py`. Grouping them is not a preference, it is the only correct answer. | C4 |

**Sequence:** `C1 → C2 → C3 → [1a gates] → C4 → C6 → C5 → [1b gates]`

### Post-review amendments

The plan review ([review-findings.md](review-findings.md)) landed nine blockers.
Three bear on sequencing, and **two of them were already satisfied by this
clustering** — recorded so nobody "fixes" them twice:

| Finding | Status against this clustering |
|---|---|
| **B1** — Task 5 fails the purity gate because `gui/shell/__init__.py` is eager | **NOT covered. Fixed by adding Task 2.5 to C1**, ordered before Task 5. This was a real defect in the plan, not in the clustering. |
| **B2** — Task 7 depends on Task 8, but is numbered first | **Already satisfied.** Task 8 sits in C1 and Task 7 in C2, and C1 precedes C2 — so Task 8 already runs first. The numeric order in `phase-1a` is misleading; the execution order is correct. |
| **B6** — `10 → 11 → 12` is a chain, not parallel work | **Already satisfied.** All three are inside C4, which is one agent working sequentially. The reviewer's warning was against staffing them as parallel agents, which this clustering never did. |

**B5 grows C4:** Task 10 splits into 10a (lift the constant), 10b (categories and
base classes for prefabs/scorers/strategies), 10c (`__all__` walk for lazy
modules). C4 becomes 10a, 10b, 10c, 11, 12 — still one cluster, still one agent,
but a materially bigger one. Re-evaluate whether it should split at the C4 gate
rather than now.

**C4 ∥ C6 parallelism withdrawn** pending B3/B4 (Task 16's CLI framework and merge
point) and B7 (Task 18's finalize signature). Those are open decisions, and
dispatching C6 against an undecided contract wastes the agent.

C3 has zero file overlap with C1/C2 and C4/C6 have none with each other, so those
are worktree-parallel candidates (`isolation: "worktree"`). Everything else is
sequential because it shares files.

---

## Gates

**After every cluster — independent reviewer.** A fresh
`execute-plan-orchestration:implementation-test-reviewer` (Opus, high effort) over
that cluster's diff only. It checks the three things the plan's per-task review
step names: no false greens (each new test must fail when its behaviour is
mutated — the "prove it can fail" steps are implementer *claims* until verified),
no scope leak, and `Interfaces` blocks matching what was actually produced. The
cluster's own tests plus `uv run ruff check <changed paths>` and
`uv run mypy src/phenotypic` run before the reviewer is dispatched, not after.

**A cluster does not hand off with an unaddressed correctness finding.** Findings
are fixed in a follow-up commit or recorded with a reason. Any finding that
conflicts with a *design* decision stops the line and comes back to the user
rather than being resolved by the executing agent.

**End of each phase — simplify.** After 1a (C1–C3) and again after 1b (C4–C6),
dispatch `code-simplifier:code-simplifier` (Opus) over the phase's combined diff:
dedupe, reduce, clarify — **quality only, no behaviour change**. Apply its fixes,
then re-run the affected suites plus `tests/unit/gui` and `tests/integration/gui`
to prove the simplification changed nothing observable.

**Phase exit gates** (in `phase-1a` / `phase-1b`) remain in force on top of all of
the above, including the CI ledger gates and the requirement that every "prove it
can fail" step was actually run with the failure observed.

---

## Dispatch record

| Cluster | Agent | Status | Reviewer verdict |
|---|---|---|---|
| plan review | `plan-reviewer` | **ABANDONED — unresponsive.** Dispatched 2026-08-14; two status pings accepted into its inbox, no reply, no completion notification across ~2 days. Its seven-area brief was **folded into C1's post-cluster reviewer** rather than dropped. | — |
| C1 | `C1-promotion` (Opus) | dispatched 2026-08-16 | — |
| C2 | — | pending | — |
| C3 | — | pending | — |
| C4 | — | pending | — |
| C5 | — | pending | — |
| C6 | — | pending | — |

Update this table as clusters complete — it is the phase's execution state, and
the next dispatcher reads it rather than reconstructing intent from git log.
