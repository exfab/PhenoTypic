# Execution — cluster DAG, gates, and dispatch order

> Derived from each task's `Files:` block by
> `scratchpad/dag.py`. **Regenerate rather than trust it** — the overlap table is what
> decides parallelism, and a stale one silently authorises two agents onto one file.

**Protocol:** every cluster is dispatched under the **`orchestrate-subagent`** contract.
Subagents read and edit files directly; **every command with a side effect — pytest, ruff,
mypy, git, sbatch, spikes — is sent to the orchestrator, run there, and returned verbatim.**
Read-only auto-approving commands (`grep`, `ls`, `sed -n`, `find`) stay with the subagent.

**Why that matters here and not only as ceremony:** P0's spikes are SLURM submissions and
P5's gate is a real array run. A command from a subagent that auto-mode will not
auto-approve never reaches the approval UI at all — the subagent blocks with no prompt
anywhere and the run stalls silently. Routing them is what puts them in front of the user.

---

## The plan-reviewer gate is already discharged

The skill opens with *"run `plan-reviewer` over the plan itself"*. **Skipped deliberately,
not overlooked.** Four rounds of `plan-refinery` just ran over this plan with a panel whose
scope is a superset: `general-reviewer` (traceability, feasibility, failure modes),
`data-flow-reviewer`, `simplicity-reviewer`, and a rotating migration specialist — plus a
resolution verifier each round. It produced 40+ findings, two user-gated Criticals, and a
provenance-locked ledger at
[`../../specs/2026-09-03-cli-gui-state-tracking/refinery/ledger.md`](../../specs/2026-09-03-cli-gui-state-tracking/refinery/ledger.md).

Re-running a single plan reviewer over the same artifact would cost a full context load for
near-zero marginal signal.

---

## Phase order

Linear, by the README's dependency column, with one exception:

```
P0 ────────────────────────────┐  (concurrent; gates P5 only)
                               ▼
P1 ──► P2 ──► P3 ──► P4 ──► P5 ──► P6 ──► P7
```

**No cross-phase parallelism.** P1's `_run_state.py` is touched by five tasks across two
phases, `phenotypicCLI.py` by three across three, `_cli_completion.py` by three across
three. The overlap table below is why this runs sequentially rather than fanned out.

---

## Clusters

Shape tags: **K**eystone (novel interdependent logic) · **S**eam (one risky wiring point)
· **W**eep — sweep (broad, shallow) · **L**eaf.

| # | Cluster | Tasks | Shape | Model | Why grouped / isolated |
|---|---|---|---|---|---|
| **0.1** | Spike gate | P0 S-2, S-3 | S | Opus/high | SLURM submissions; their verdicts parameterise P5. Isolated so the user sees each `sbatch`. |
| **1.1** | Module skeleton + types | P1 T1, T2 | K | Opus/high | `_state_types` is the leaf both others import; INV-LAYER's AST test lands with the stubs it binds. |
| **1.2** | Verification cache | P1 T3 | K | Opus/high | Carries INV-VERDICT's mutation suite **and the S-5 on-disk-tier decision**. A decision inside a cluster gets its own gate. |
| **1.3** | The readers | P1 T4, T5, T6 | K | Opus/high | All three write `_run_state.py` + `test_run_state.py`. Cannot be split without two agents on one file. |
| **1.4** | `requires_conversion` | P1 T3b | S | Opus/high | Disjoint files (`_cli_schema_gate.py`). **This is the guard that stops P3's clean break turning a legacy tree into an empty master** — isolated so its gate is its own. |
| **2.1** | Identity minting | P2 T1, T2, T3 | K | Opus/high | Share `_cli_identity.py` + `test_run_identity.py`. |
| **2.2** | `scheduler_epoch` collapse | P2 T4 | S | Opus/high | Collapses a token "only where one writer owns the lifetime" — the risk is deciding *where*. |
| **3.1** | Record writer/reader | P3 T1 | K | Opus/high | The record schema, `provenance` (U-10), and the shared vocabulary constants. |
| **3.2** | Publishers onto the record | P3 T2 | K | Opus/high | Four files including `_run_state.py` and `_cli_completion.py`; carries U-10's `valid_image_success` split. |
| **3.3** | Stage 2/3 collapse | P3 T3 | S | Opus/high | The plan's own "risky task": rewrites `classify_staged_image` behind a 384-combination equivalence gate, with FLOW-40 surviving verbatim. |
| **4.1** | Table inversion | P4 T1, T2 | K | Opus/high | Share `test_embedded_table_inversion.py`; the split and its promote-time write are one contract. |
| **4.2** | `finalize_run` + entry points | P4 T3, T4, T5 | K | Opus/high | Share `test_finalize_run.py`. T5 is the end-to-end verification of T2 — it is this cluster's gate, not a task. |
| **5.1** | Fan-out engine | P5 T1, T2, T3 | K+S | Opus/high | Share `_cli_finalize_fanout.py`. **Contains the array-auxiliary Seam:** the finalize trigger must be a reserved entry in the task list, never a parallel sidecar job. |
| **5.2** | Failure + rolling matrices | P5 T4, T5 | S | Opus/high | Both are phase gates needing **real** runs — one SLURM, one local. |
| **6.1** | CLI completion split | P6 T0 | S | Opus/high | Ten files; every later P6 task assumes one completion predicate. Goes first. |
| **6.2** | Results-viewer consumers | P6 T1, T2, T3 | W | Sonnet/med | Mechanical call-site migration + a 617-line deletion. Frontier verify at the phase gate. |
| **6.3** | Registry + observer | P6 T4, T5, T6 | W | Sonnet/med | T4/T5 share `_runs_registry.py`. |
| **6.4** | Deletions + GUI register | P6 T7, T8 | W+L | Sonnet/med | Each deletion is re-grepped before it happens; T8 is docs. |
| **7.1** | Detection + refusal | P7 T1 | L | Sonnet/med | **Mostly already built** — P1 T3b builds this task in full (CAN-11). Verify and wire; do not rebuild. |
| **7.2** | Marker + state conversion | P7 T2, T2b, T3 | K | Opus/high | Share `_cli_migrate_state.py`. Carries U-10's marked records and the ported promoter. |
| **7.3** | Master stamp | P7 T4 | K | Opus/high | The embedded-table question and the schema stamp, whose ordering was MIG-2. |
| **7.4** | Dry-run, rollback, register | P7 T5, T6 | S+L | Opus/high | The rename/revert protocol is the Seam; the `_cli/CLAUDE.md` register is the phase's deliverable. |

---

## Files touched by more than one task

The parallelism veto. Regenerate with `dag.py` before trusting it.

| Touchers | File |
|---|---|
| 5 | `sdk_/_run_state.py` — P1 T1/T4/T5/T6, **P3 T2** |
| 5 | `tests/unit/sdk_/test_run_state.py` — P1 T2/T4/T5/T6, **P3 T2** |
| 4 | `tests/unit/cli/test_run_identity.py` — P2 T1/T2/T3/T4 |
| 4 | `_cli/_cli_migrate_state.py` + its test — P7 T2/T3/T4/T5 |
| 3 | `phenotypicCLI.py` — **P1 T3b, P6 T0, P7 T1** (three phases) |
| 3 | `_cli/_cli_identity.py` — P2 T1/T2/T3 |
| 3 | `_cli/_cli_completion.py` — **P2 T4, P3 T2, P6 T0** (three phases) |
| 3 | `_cli/_cli_finalize_fanout.py` — P5 T1/T2/T3 |
| 2 | `sdk_/_io_constants.py` — P1 T1, P2 T1 |
| 2 | `_cli/_cli_schema_gate.py` + test — **P1 T3b, P7 T1** |
| 2 | `_cli/_cli_staged_resume.py` — P3 T3, P6 T0 |
| 2 | `_cli/_cli_recompile_worker.py` — P4 T4, P6 T0 |
| 2 | `_cli/_cli_migrate.py` — P6 T0, P7 T2 |
| 2 | `gui/shell/_runs_registry.py` — P6 T4, P6 T5 |

---

## Gates

### Per cluster — light, run by the orchestrator
Read the changed files, run the cluster's own test selection plus `ruff check --fix` on
**explicit changed paths only**, review the diff. **Pause and surface to the user** any
design question the review raises before the next cluster.

### Per phase — three agents, all frontier
Never review with a weaker model than implemented.

1. **`implementation-test-reviewer`** over the phase's combined diff — every phase adds
   tests, and the question is whether they *can fail*, not whether they pass.
2. **Spec-adherence reviewer** *(added at user request)* — see the brief below.
3. **Orchestrator triage** — fix high-signal findings before the next phase starts.

Then: `uv run mypy src/phenotypic`, `ruff` on changed paths, the phase's test selection, and
a commit that passes its own gate — so a bisect lands on a phase boundary, not mid-rewrite
(README, "Why this is one change and not seven PRs").

### Spec-adherence reviewer — the brief

A **different question** from code review, and it must not be folded into it. Code review
asks *"is this correct?"*. This asks *"is this what we said we would build, all of it?"* —
the failure it catches is a phase that is green, correct, and **missing a third of its
scope**.

Given: the spec, the phase's plan doc, the phase's combined diff, and the ledger. Report
four categories, each finding citing `file:line` in **both** the plan/spec and the diff:

| | Category | The question |
|---|---|---|
| **A** | Specified, not implemented | Every spec requirement this phase claims — is there code for it? |
| **B** | Planned, not done | Every task step and every named test — did it land? An unchecked box with no code is this. |
| **C** | Implemented, but differs | Does the code do what the spec says, or something adjacent? Names, signatures, and **ordering** count. |
| **D** | Implemented, never specified | Scope creep, and the mechanism by which a plan grows a fourth authority nobody agreed to. |

Constraints that make it worth running:

- **Verify by reading the diff, never by reading the checkboxes.** A checked box is a claim.
- **Cite both sides.** "P4 §7.3 requires X; `_cli_finalize_run.py:212` does Y" is a finding;
  "spec adherence looks good" is not.
- **Category A over an unconverted call site is the phase's own failure mode.** This plan's
  recurring defect across four review rounds was *a reader in a file nobody named* — six
  separate instances. Regenerate the consumer greps; do not trust the plan's file lists.
- Analysis only. Never edits.

### Every test run past a single file goes through the sharded array

**Standing instruction (user, 2026-09-04): cluster any large suite and run it massively in
parallel on SLURM.** Not only the final regression — every phase gate, and any run whose
numbers get quoted as a baseline.

Harness: [`run_suite.sbatch`](run_suite.sbatch) + [`collect_results.py`](collect_results.py).

```bash
# whole suite
sbatch run_suite.sbatch
# one phase's scope, same harness
SCOPE="tests/unit/sdk_ tests/unit/cli" sbatch run_suite.sbatch
# then, always:
uv run python collect_results.py <results_dir> --baseline <baseline_dir>
```

48 shards, at most 32 resident. **Sizing is against the account cap, not the node** —
`iwheeldonlab` is 384 CPU / 1 TB shared across every running job, and exceeding an *account*
cap does not fail at submit: it queues with `Reason=AssocGrpCpuLimit`, which `--test-only`
does not catch. 32 × 8 CPU = 256 leaves deliberate headroom so a concurrent spike does not
queue behind the gate.

Measured limits, not assumed (2026-09-04): `MaxArraySize = 2500`, `MaxJobCount = 50000`,
`MaxSubmitJobs = 5000`. **`MaxSubmitJobs` is an association limit and does not appear in
`scontrol show config` at all** — it comes from `sacctmgr show assoc`. The phase docs' formula
`min(MaxArraySize, MaxSubmitJobs)` therefore reduces to `MaxArraySize` here, and a script that
reads it from `scontrol` gets an empty string, which in arithmetic becomes zero.

**Do not buy shards by starving them.** CPUs-per-task stays at 8. This suite has a documented
population of load-sensitive flakes — a 1 s subprocess import, a 20 s multiprocessing join, a
0.5 s patched read deadline — that pass alone and fail contended. Narrowing each shard
manufactures more of them, and they cost more to triage than the wall-clock saved.

**Compare names, never counts.** That is what `collect_results.py` is for: the aggregate count
moves with node load and with how shards happened to pack, while the failing *set* does not.
Four failures are known pre-existing, three of which fail only on compute nodes; the standing
list is in the `phenotypic-regression-baseline` memory. A regression is a name that is failing
here and passing at baseline — and even then, **run it alone before believing it.**

Traps the harness already closes, each of which produces a *wrong answer* rather than a slow
one: missing `QT_QPA_PLATFORM=offscreen` aborts the interpreter partway with no summary;
`-n auto` reads the node's cores rather than the allocation's; the repo's default `addopts`
streams uncaptured output and can triple runtime when stdout is on shared storage; `-x`
truncates a run that then gets recorded as a baseline; and leaking `SLURM_ARRAY_TASK_ID` into
tests that mock scheduler state makes them read the *gate's* identity instead of their own.

### End of run
One `code-simplifier` pass (quality only, no behaviour change), apply fixes, then the full
suite through the same harness and `collect_results.py --baseline`.
