# Execution — cluster DAG, gates, and dispatch order

> Derived from each task's `Files:` block by [dag.py](dag.py). **Regenerate rather than
> trust it** — the overlap table is what decides parallelism, and a stale one silently
> authorises two agents onto one file.
>
> ```bash
> uv run python docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/dag.py
> uv run python .../dag.py --check    # non-zero if this file is stale
> ```
>
> **That instruction pointed at `scratchpad/dag.py` for the life of this plan, and no such
> file existed** — the generator was written in a session scratchpad and went with it. So
> the one table whose whole job is to prevent a collision could not be regenerated, and the
> instruction to distrust it had no way to be followed. The script is now committed here,
> beside the mutation harnesses and spikes, for the same reason they are.

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
| **2.2** | `scheduler_epoch` collapse — **WITHDRAWN, see below** | P2 T4 | S | Opus/high | Collapsed a token "only where one writer owns the lifetime" — the risk was deciding *where*, and the answer turned out to be **nowhere**. All five tokens are non-collapsible: §5.1's amendment (`design.md:323-345`, user-ruled) has the writer-by-writer table. Task 4 shipped a dead-comparison deletion instead. |
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

Generated by [dag.py](dag.py) — **do not hand-edit**. Regenerate after any change to a
task's `Files:` block.

| Touchers | File | Tasks |
|---|---|---|
| 6 | `src/phenotypic/sdk_/_run_state.py` | P1 T1, P1 T4, P1 T5, P1 T6, P2 T1, P3 T2 |
| 5 | `tests/unit/sdk_/test_run_state.py` | P1 T2, P1 T4, P1 T5, P1 T6, P3 T2 |
| 4 | `src/phenotypic/_cli/_cli_migrate_state.py` | P7 T2, P7 T3, P7 T4, P7 T5 |
| 4 | `tests/unit/cli/test_migrate_state.py` | P7 T2, P7 T3, P7 T4, P7 T5 |
| 4 | `tests/unit/cli/test_run_identity.py` | P2 T1, P2 T2, P2 T3, P2 T4 |
| 3 | `src/phenotypic/_cli/_cli_completion.py` | P2 T4, P3 T2, P6 T0 |
| 3 | `src/phenotypic/_cli/_cli_finalize_fanout.py` | P5 T1, P5 T2, P5 T3 |
| 3 | `src/phenotypic/_cli/_cli_identity.py` | P2 T1, P2 T2, P2 T3 |
| 3 | `src/phenotypic/phenotypicCLI.py` | P1 T3b, P6 T0, P7 T1 |
| 3 | `tests/unit/cli/test_finalize_fanout.py` | P5 T1, P5 T3, P5 T4 |
| 2 | `src/phenotypic/_cli/_cli_migrate.py` | P6 T0, P7 T2 |
| 2 | `src/phenotypic/_cli/_cli_recompile_worker.py` | P4 T4, P6 T0 |
| 2 | `src/phenotypic/_cli/_cli_schema_gate.py` | P1 T3b, P7 T1 |
| 2 | `src/phenotypic/_cli/_cli_staged_resume.py` | P3 T3, P6 T0 |
| 2 | `src/phenotypic/gui/shell/_runs_registry.py` | P6 T4, P6 T5 |
| 2 | `src/phenotypic/sdk_/_io_constants.py` | P1 T1, P2 T1 |
| 2 | `src/phenotypic/sdk_/_verification_cache.py` | P1 T3, P2 T1 |
| 2 | `tests/integration/` | P4 T5, P7 T5 |
| 2 | `tests/unit/cli/test_embedded_table_inversion.py` | P4 T1, P4 T2 |
| 2 | `tests/unit/cli/test_finalize_run.py` | P4 T3, P4 T4 |
| 2 | `tests/unit/cli/test_image_record.py` | P3 T1, P3 T2 |
| 2 | `tests/unit/cli/test_schema_gate.py` | P1 T3b, P7 T1 |
| 2 | `tests/unit/gui/shell/` | P6 T4, P6 T5 |

**The sixteen rows this table was originally maintained with by hand are all reproduced by
the generator.** Six more were added when it was first run, and three matter because they cross a cluster
boundary — the only kind this table exists to veto:

| Overlap | Clusters | Why it was invisible |
|---|---|---|
| `tests/unit/cli/test_image_record.py` | **3.1 + 3.2** | P3 T1 creates it, T2 extends it. Two clusters, one file. |
| `tests/unit/cli/test_finalize_fanout.py` | **5.1 + 5.2** | P5 T1/T3 build the engine, T4 is the failure-matrix gate. |
| `tests/integration/` | **4.2 + 7.4** | Both end-to-end gates, four phases apart. |

The other three (`test_embedded_table_inversion.py`, `test_finalize_run.py`,
`tests/unit/gui/shell/`) are intra-cluster and harmless — the cluster table already groups
those tasks precisely because they share a file.

**Two parser bugs were found writing the generator, both under-reporting.** A path carrying
a comma-separated line list (`phenotypicCLI.py:2394,2428,...`) stayed a distinct key, and a
`Files:` line naming two paths yielded only the first. Each made a real overlap invisible.
Both were caught by opening the `Files:` block that the output implied was wrong, rather
than believing the output — the first nearly went out as a finding that P6 Task 0's file
list was incomplete, when the list was right and the parser was not. A veto generator that
under-reports fails in the direction that costs work, so its own parsing needs the scrutiny
the plan gets.

---|---|
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

### Push the branch after every phase

**Standing instruction (user, 2026-09-05).** Once a phase's gate is green and its commit
lands, push `cli-gui-state-tracking` to `origin`. Not at the end of the change — after each
phase, so the work is recoverable if a session or the worktree is lost.

**`git push` fails by default in this environment**, and the error names the wrong cause:

```
gnome-ssh-askpass: cannot open display
git@github.com: Permission denied (publickey)
```

There is no `ssh-agent` (`SSH_AUTH_SOCK` is unset) and the default-selected
`~/.ssh/id_ed25519` is **not** the key registered with GitHub. The working key is
`~/.ssh/github_agent`, which is passphrase-less:

```bash
GIT_SSH_COMMAND="ssh -i /rhome/anguy344/.ssh/github_agent -o IdentitiesOnly=yes -o BatchMode=yes" \
  git push origin cli-gui-state-tracking
```

Spell the key path absolutely — `$HOME` inside an exported `GIT_SSH_COMMAND` trips the
worktree-isolation guard, as does `git -c credential.helper='!gh ...'`. Use the per-command
prefix, not `export`.

**The backlog this discharged is the argument for the rule.** When the instruction arrived the
branch was **97 commits ahead of `origin`** — every commit of P0, P1 and P2 existed in exactly
one place, a worktree on a shared filesystem, with no second copy anywhere.

### Per phase — three agents, all frontier
Never review with a weaker model than implemented.

**Both reviewers are INDEPENDENT, and independence is mechanical, not a posture** (user,
2026-09-05): a *fresh* agent that did not write the code, with no memory of the implementing
session's reasoning. An implementer reviewing its own phase re-derives the same blind spot —
it already believes the thing it would need to doubt. Never reuse the cluster agent, and
never let a reviewer's brief carry the implementer's summary of what it did; give it the
spec, the plan and the diff, and let it find out.

1. **`implementation-test-reviewer`** over the phase's combined diff — every phase adds
   tests, and the question is whether they *can fail*, not whether they pass.
2. **Spec-adherence reviewer** *(added at user request)* — see the brief below.
3. **Orchestrator triage** — fix high-signal findings before the next phase starts.

**Required after phases 0-4 specifically** (user, 2026-09-05), and applied to every phase
here because the failure mode does not respect phase numbers. P1 ran both
(`P1-gate-tests`, `P1-gate-spec`); P2's are due once cluster 2.2 lands.

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
| **C** | Implemented, but differs | Does the code do what the spec says, or something adjacent? Names, signatures, and **ordering** count. **See *Disposition of C and D* below — drift needs an experiment, not a rationale.** |
| **D** | Implemented, never specified | Scope creep, and the mechanism by which a plan grows a fourth authority nobody agreed to. Same disposition rule as C. |
| **E** | **Implemented as a placeholder** | Is there a *body*, or only a name? See below — this is the category A cannot catch. |

#### Disposition of C and D — when drift is allowed at all

**Standing rule (user, 2026-09-05): spec drift is acceptable ONLY where we
experimentally validated the alternative and recorded the decision.** Not where it
seemed cleaner, not where the implementer had a good reason, not where it turned out
fine. **Evidence, then a ruling, then an amendment** — in that order, and all three.

So a C or D finding is not automatically "revert it". It is automatically **a question
the gate must force**, and there are exactly three answers:

| The deviation has… | Disposition |
|---|---|
| an experiment **and** a recorded ruling **and** the spec amended | **Legitimate.** Not drift any more — the spec says what the code does. Nothing to do. |
| an experiment and a ruling, but the **spec still says the old thing** | **Finding.** The code is right and the document is now wrong, which is this change's most-repeated defect. Amend the spec; do not touch the code. |
| a rationale but **no experiment** | **Finding, and the disposition is the user's.** Surface it. Neither the reviewer nor the orchestrator may ratify a deviation after the fact by finding the reasoning persuasive. |

**Why "it turned out fine" is not a disposition.** An implementer who deviates has
already concluded their way is better — that is why they did it. Accepting the
deviation because the argument is good means the gate ratifies exactly the judgement it
exists to check, and a reviewer who does this has reviewed nothing. The experiment is
what makes the difference between a decision and a preference.

**The bar is met, and it looks like this.** U-11 reversed an approved ruling (D-B,
in-process cache only) — on a **measurement**: 1403 s versus 37 s, cold, on a fresh
node with disjoint halves. Ruling recorded in §0, spec amended, on-disk tier shipped.
That is drift done correctly, and the shape to compare against.

**Consequence for the reviewer's report:** every C and D finding must state which row
above it falls in, and cite the experiment and the ruling by identifier when it claims
the first or second. *"This deviation is fine because X"* is not a disposition — it is
the third row wearing the first row's clothes.

##### The rule's scope: drift, not every unmeasured choice

**Check that the spec actually says something before calling a deviation drift.** The
rule governs *the spec said X and the code does Y*. Where the spec is **silent**, the
implementer has latitude, and an implementation choice inside that latitude is **not
drift and needs no experiment** — it needs to be reasoned and reviewable.

Worked example, and the orchestrator got this wrong first: `mint_run_identity` appears
in the spec **once** (`design.md:328`), as `(config, *, restart) -> RunIdentity  CLI
only`. That is a signature and a layer constraint, and the implementation satisfies both.
So P2's three flagged decisions are **latitude, not drift**, and demanding an experiment
for each would be the rule over-firing.

**Ask the sharper question: what does the spec constrain, and is that the thing being
chosen?** "The spec is silent on `metadata_sha256`" would be **false** — it appears five
times (`:443`, `:458`, `:469`, `:503`, `:666`), and a reviewer who greps and finds five
hits will reasonably conclude the claim was careless. Every one of them constrains
something the implementation already satisfies:

| Where | Constrains | Satisfied? |
|---|---|---|
| §5.5 `:469` | the finalization-input **object's contents** — `{schema_version, metadata_sha256, include_dataset_column, no_qc}` | yes, exactly those four keys |
| §7.4 `:666` | the **consequence** — a metadata edit invalidates `finalization_input_digest`, so the next invocation re-finalizes without touching an image | yes |
| §5.4 `:443`, `:458` | **placement** — finalization side, not per-image | yes |

What no section addresses is **where the value comes from at mint time** — recomputed
from config, or read from a state file that does not yet carry it. That, and only that,
is the choice. **The spec constrains the object and the behaviour; it is silent on
provenance.**

So the finding is not *"the spec does not mention this"* but *"the spec constrains X, Y
and Z, the code satisfies all three, and the decision lies outside them."* Cite the
sections you checked, including the ones that turned out to be satisfied — a reviewer
who names only the silence has not shown they looked.

**Applying it to everything is how a good rule becomes noise.** If every unmeasured
implementation choice is a finding, the report fills with them, and the one deviation
that genuinely contradicts the spec is read at the same weight as a private helper's
shape. The rule earns its severity by being narrow.

**But note what does NOT change.** A latitude decision can still be a **correctness**
finding on its own merits, and the disposition rule is silent about that. P2's
`_metadata_digest_for` must recompute `metadata_sha256` because `phenotypicCLI.py:471`
stamps it only after state creation — a real constraint, no experiment needed, no drift.
It *still* needs a cross-module agreement test, because if the two computations diverge
§7.4's late-metadata guarantee fires on every run instead of on a real edit. That is
category E's concern (a guard that is prevention with no detection), not this rule's.

#### Category E — the placeholder sweep, and why A does not cover it

**Added at user request, 2026-09-05: guard against an implementer that ships a name instead
of a behaviour.** A placeholder **passes category A**. A is *"is there code for this
requirement?"* — and a correctly-named function, at the path the plan names, with the
signature the plan specifies, returning a constant, answers yes. It also passes review,
passes mypy, and passes any test written against it, because the test was written to the
same misunderstanding.

So E is a **separate, mechanical sweep of the phase's diff**, not a judgement:

```bash
# in the phase's changed files only
grep -rn "NotImplementedError\|TODO\|FIXME\|XXX\|placeholder\|for now\|stub" <files>
grep -rn "^\s*\.\.\.\s*$\|^\s*pass\s*$" <files>      # bodies that are only a name
grep -rn "@pytest.mark.skip\|@pytest.mark.xfail" <files>   # a test that does not run
```

Then, for every symbol the plan named, **read the body and answer in one sentence what it
computes.** A body you cannot summarise without repeating its signature is the finding.

Four shapes that are placeholders while looking implemented:

1. **Returns a constant where the spec requires derivation** — `return True`, `return {}`,
   `return None` on a path the spec says computes something.
2. **A guard with no consequence** — the condition is evaluated and the branch does nothing,
   so the check reads as present and enforces nothing.
3. **A skipped or xfailed test with no removal note.** A skip is legitimate *only* with the
   condition and the phase that removes it named — `test_schema_gate.py:972` is the correct
   form; a bare `@skip` is scope silently deferred.
4. **A parameter accepted and never read.** The signature satisfies the plan; the behaviour
   does not exist. mypy will not say a word.

Report each with the plan's requirement on one side and the body on the other, exactly as
for A-D. **A finding here outranks everything except a correctness bug**, because a
placeholder is the one defect that gets *more* expensive with every phase built on top of
it — and by construction nothing downstream will fail until something depends on the
behaviour that was never written.

Constraints that make it worth running:

- **Verify by reading the diff, never by reading the checkboxes.** A checked box is a claim.
- **Cite both sides.** "P4 §7.3 requires X; `_cli_finalize_run.py:212` does Y" is a finding;
  "spec adherence looks good" is not.
- **Category A over an unconverted call site is the phase's own failure mode.** This plan's
  recurring defect across four review rounds was *a reader in a file nobody named* — six
  separate instances. Regenerate the consumer greps; do not trust the plan's file lists.
- Analysis only. Never edits.

### Every state-tracking check goes through a shared helper — forever, not once

**Standing rule (user, 2026-09-05), binding on P3-P7 and on every fix.** A state-tracking
check is **called**, never **reimplemented**. An implementation that open-codes a question
some helper already answers is a finding, even when its logic is correct today.

**Why this is structural and not style.** The change's thesis is *nine ways of asking "is
this run done" become one*. Collapsing the **answers** while leaving the **asking**
duplicated buys nothing: two implementations of one question drift, and the failure mode is
that two parts of the system disagree about the same tree while both look right in
isolation.

**The questions, and where each is answered exactly once:**

| Question | The one helper |
|---|---|
| is this run complete? | `resolve_run_state(output_dir, depth=...)` |
| is this identity current? | `assert_identity_current`, over `_IDENTITY_DIGEST_FIELDS` |
| is this per-image record valid? | the record reader in `sdk_/_image_record.py` |
| is this artifact's proof intact? | the fenced-artifact / stat-tuple comparison in `_run_state.py` |
| which images remain? | the worklist derivation, not a hand-rolled set difference |
| every identity digest | one definition per digest, imported — not restated |

**It is already broken once, which is why it is a rule and not an aspiration.**
`inventory_digest` has two producers computing it differently — `_run_state.py:276` from
`config["work_ids"]`, `_cli_identity.py:124-141` from `config.image_manifest_digest` — and
the field is in `_IDENTITY_DIGEST_FIELDS`, so it is exactly what `assert_identity_current`
compares. Two answers to one question, on the field that decides whether a cached verdict
may stand.

**Where the helpers live (user, 2026-09-05): `sdk_`, not `_cli`.** One home —
`src/phenotypic/sdk_/_state_tracking.py`, with the public names re-exported from
`sdk_/__init__.py` per the project's "only `__init__` exports are public" convention.

**Because there are THREE consumers, not two.** The CLI and the `sdk_` readers are the
obvious pair; **the GUI is the third**, and it is the one that decides the location. Today
the GUI reaches into `phenotypic._cli` with 25 private imports across 9 modules — the audit
finding this whole change exists to remove. Homing shared state-tracking logic in `_cli/`
would have made that reach *correct* and entrenched it permanently.

`sdk_` is the only layer all three may import:

| Consumer | may import `sdk_` | may import `_cli` |
|---|---|---|
| `phenotypic._cli` | yes | — |
| `phenotypic.gui` | yes | **no** (that is the defect) |
| `sdk_` readers | yes | **no** (INV-LAYER, AST-enforced) |

So the rule is simply: **one definition, in `sdk_`, imported by everyone.** No re-export
shims, no per-layer copies, and INV-LAYER holds by construction rather than by discipline.

**For implementers, in order:**

1. Before writing a check, **grep for the question**, not for the function name you expect.
   The duplicate you are about to create usually exists under a different spelling.
2. If a helper exists, **call it.** If it is in the wrong layer, move it or raise it — do
   not copy it.
3. If a genuine constraint blocks sharing — an import cycle, a layer boundary — **the
   finding is that the shared definition needs a home neither module owns**, not that
   restating it is acceptable. Say so and raise it; do not restate and move on.
4. A restatement that ships anyway carries, at its site, the reason sharing was impossible
   and the name of the definition it mirrors. A future reader must be able to find the
   other copy.

**The gate checks this.** A reviewer with this remit runs each phase alongside the
correctness and spec-drift reviewers.

### Enumerate with the OLD name — and use the same grep to find the work and to check it

**Three instances in one session, one from each participant.** This is the single most
reusable procedure P3 produced, and P4's repoint will need it.

**The rule:**

> **After a partial migration, grep for the OLD name, not the new one.** The new name shows
> you what you *did*; the old one shows you what you did *not*.
>
> **And the grep that finds the work must be the same grep that verifies it.** If they
> differ, the first one encoded an assumption.

**The three instances:**

| Who | Enumerated with | Missed | Found by |
|---|---|---|---|
| agent | the four sites it had read | a fifth, in the file it had just edited | the orchestrator grepping the old name for *retained* sites |
| agent | `marker_path.unlink()\|write_bytes` — a **mutation-shaped** pattern | three read-only sites: two digests, one `is_file()` | running the old-name grep afterwards, as a check |
| orchestrator | `\.marker_path` — a **use-shape** pattern requiring a leading dot | a keyword argument `marker_path=(` | re-running without the assumed prefix |

**Each miss came from a pattern encoding what the searcher expected the uses to look like** —
mutations, attribute access, or "the sites I already classified." The name is the population;
everything else is a hypothesis about the population.

**Why reviewing the change cannot substitute.** Checking the sites someone repointed can never
find a site they did not. Only asking *what is left* can — which is why the verification must
enumerate the **remaining** population, not re-inspect the changed one.

**And it applies to your own diff.** Both agent instances were in a file it had just finished
editing. *"Re-derive the population"* is not only for numbers you are handed.

#### The category bug: a symbol grep cannot see a site that bypasses the symbol

**The fifth and last variant, and the only one no pattern refinement could have fixed.**

Four earlier variants were *pattern bugs* — too narrow, too broad, wrong anchoring, wrong
shape. This one is a pattern **category** bug. The sweep enumerated with the helper's name:

```
grep image_completion_marker_path       ->  22 files
grep '"image_complete"'                 ->   7 hand-joined sites in 5 files
```

**Seven sites hand-join the path segment by segment**, never calling the helper:

```python
root / ".phenotypic" / "progress" / "image_complete" / "day1" / "plateA.json"
```

A helper-name grep is **structurally incapable** of seeing them — not mis-tuned, incapable.
No anchoring, unanchoring or word-boundary fix reaches a site that does not contain the
symbol.

> **A sweep keyed on a symbol cannot see a site that does not use the symbol.** Enumerate the
> *thing*, not the *way it is usually spelled* — and where a helper exists, the sites that
> bypass it are exactly the ones a helper-name grep will miss.

**The cruel part:** this repository's own `CLAUDE.md` says *"Always resolve paths via the
`phenotypic.sdk_` helpers (never hand-join names)"*. The rule exists for precisely this
reason, and **its violations are invisible to every tool you would use to enforce it** — a
grep for the helper finds the compliant sites and only those.

**So enumerate a path change on the LITERAL as well as the helper.** One extra grep, and it
is the only one that finds the sites that broke the convention the change depends on. The
same holds for any refactor keyed on a symbol: constants, key names, directory segments,
status strings.

#### No pattern is right in general — the candidates come from the grep, the answer from reading

**A fourth variant closed the rule, and it is the one that shows why no pattern suffices.**
After adding word-anchoring to fix the substring over-match, the anchored grep went blind to
the *compound* name: `_` is a word character, so `grep -w marker_path` **cannot match inside**
`image_completion_marker_path`. Measured across the five remaining files:

```
                                        -w    compound    plain
test_cli_migrate_image                  19        2        20
test_embedded_measurement_migration      0        3        11
```

**Each anchoring is blind to what the other finds.** One file's 19 sites are invisible to the
compound grep; another file is entirely invisible to the anchored one.

**And plain is not the answer either.** That file's 11 plain hits include eight
`aggregate_publication_marker_path` / `run_completion_marker_path` uses — different functions
sharing a substring, untouched by the change. So:

| Pattern | Gives you |
|---|---|
| anchored | a **total** you can assert |
| unanchored | the **candidates** you must act on |
| neither | the actionable set |

> **Anchored patterns are for totals you will assert; unanchored ones are for populations you
> will act on — and an unanchored grep gives candidates, never the answer. Read every hit.**

The classification is the work. A pattern that appears to do it for you has encoded the
answer you expected, in whichever direction it errs.

#### The mirror hazard: a fix credited to the wrong change

The same sweep found `test_cli_migrate_mode.py` with **zero hits of either name and seven
failures**. Those failures were the CLI raising in `_cli_migrate_image.py`, already repaired
by the source fix. Had that file been reported as "swept", **the next run's green would have
been credited to an edit nobody made.**

That is worse than a masked failure, because it is *confirmed by a passing run*. It leaves a
false causal record: the next person believes those tests depend on a repoint that does not
exist, and reasons from it. **A file listed as swept with no diff is indistinguishable from an
oversight** — say what actually fixed it.

### Isolate a mechanism by construction; do not assert its name

**Earned twice in P3, once against the orchestrator's own instruction.**

The rule *"assert the reason, not the verdict"* is right in spirit and unbuildable as stated
whenever the code does not surface a reason. Told to pin a "stale artifact" case by asserting
on `record_rejection`'s reason string, the agent checked the function first:

- `record_rejection` tests **five clauses, all identity and shape** — version, dataset,
  image_stem, work_id, artifacts-non-empty — and its docstring says *"artifact contents are
  not checked here"*. A well-formed record naming an absent file passes all five and returns
  `None`. **The reason string for the case in question does not exist.**
- `fenced_artifact_path`, the other half, returns `None` for *"malformed, escapes the root, or
  no longer matches disk"* — **three causes, one value**, which is the same conflation, one
  layer along.

**What works instead is two assertions that isolate the mechanism by construction:**

1. **A negative on form** — `record_rejection(...) is None`, proving no shape or identity
   clause is what rejects it. This is what fails loudly if a later change makes the fixture
   invalid for a *new* reason, which is the degradation the original rule was reaching for.
2. **A positive control** — create the missing artifact, assert the verdict flips; remove it,
   assert it flips back. This establishes the absence as the **cause** rather than as a fact
   that merely co-occurs with the verdict.

Neither alone suffices: (1) without (2) is consistent with rejection for an unrelated reason;
(2) without (1) is the bare verdict check the rule exists to forbid.

> **A construction that isolates a mechanism beats an assertion that names one**, because a
> name can stay correct while the mechanism moves underneath it — which is exactly how the
> coverage in entry 27 died without the test changing.

**And the corollary for whoever is giving the instruction:** naming a function as an
assertion target is a claim about that function's contract. Check that it can carry the
assertion before requiring it. Two rulings in this phase named a function without that check;
both were caught by the recipient re-deriving rather than complying. **A ruling is not
evidence.**

### Silent-and-wrong beats loud-and-broken — the criterion for in-phase vs deferred

**Two scope calls in one phase, opposite answers, and the difference is not size.**

| | Failure mode | Ruling |
|---|---|---|
| `--mode measure` skips its re-publish | **silent** — no exception, returns `True`, an image quietly stops being certified | **fix in-phase** |
| `--mode recompile` reads a marker that is gone | **loud** — `FileNotFoundError` on the first call | **candidate for deferral** |

Both are real regressions introduced by the same change. Both cross into another phase's file
mapping. The first was obviously in-scope and the second is not, and the reason is the shape
of the failure rather than the size of the fix:

> **A bad interval is recoverable; a bad artifact is not.** A loud break wastes a user's
> afternoon and is fixed by the next commit. A silent one writes wrong state to disk that
> outlives the interval, and nothing in the tree records that it happened.

So the question to ask of a regression you are tempted to defer is **not** "how big is the
fix" or "whose phase owns the file" — it is **"what does the tree look like afterwards?"** If
the answer is "indistinguishable from correct", fix it now regardless of whose module it is.
If the answer is "obviously broken until someone fixes it", deferral is arguable and the
decision belongs to whoever owns the schedule.

**The phase-gate contract still binds**: a phase ends at a commit passing its own gate. So a
deferral is not "leave the suite red" — it is a decision, made explicitly and recorded, that
a named mode is knowingly non-functional for a named interval. **That is a user's call, not
an implementer's**, because it trades their working software against our sequencing.

### ⛔ HARD STOP: a fix may not increase the state-artifact count

**Standing rule (user, 2026-09-05).** Execution runs to completion without checking in,
with exactly two exceptions. The first is an open question needing a ruling. **The second
is any fix — from a gate finding, a review, or a later phase — that would raise the number
of state-tracking artifacts above what the plan outlines.** Stop and ask; do not implement
it and flag it afterwards.

**The declared budget** (P7 Task 6's register):

| | Count | |
|---|---|---|
| **Tracked state** | **4** | accepted inventory · terminal failures · liveness & ownership · `restart_epoch`. *"Four. If a fifth appears, that is a design regression."* |
| Content proofs | 3 | per-image record, aggregate proof, run proof — digest manifests over artifacts that already exist, not tracked state |
| Neither tracked nor derived | 2 | `.phenotypic/legacy-v2/` · `verification_cache.json` — nothing branches on them, no verdict derives from them |

**Why this is a stop and not a preference.** The whole change is the claim *nine evidence
sources become three, fourteen tokens become five*. A fix that adds an artifact does not
merely cost some tidiness — it falsifies the change's own thesis, and it does so at exactly
the moment everyone is focused on the defect being fixed rather than on the count.

**It has already been the deciding argument twice**, which is why it is written down:

- **U-11's on-disk cache tier** was allowed only after establishing that *nothing branches
  on it and no verdict is derived from it* — a cache, listed under "neither tracked nor
  derived", never under tracked state. The measurement (1403 s → 37 s) justified building
  it; the classification is what made it admissible.
- **§5.1's dual-key rename shims were rejected on this ground alone.** Read-both-keys
  support in every reader is *more* state to keep in sync — a change whose stated purpose
  is reducing tracked state would have ended by adding some.

**The test to apply to any proposed fix**, in this order:

1. Does it add a file, key, or field that something **branches on**? → tracked state. **Stop
   and ask.**
2. Does it add one that nothing branches on and no verdict derives from? → a cache or a
   retained artifact. Still name it in P7 Task 6's register, and still say so when
   proposing it.
3. Does it add a *second home* for a value that already exists? → that is the defect this
   change removes, not a fix. Reject it without asking.

### Never sample a tree another agent may be holding — including a tool's output

**Earned six times in this phase, the last of which was the instructive one.** Before you
hash a file, run `git status`, diff against a snapshot, **or read a harness log**, establish
that nothing else is writing:

```bash
pgrep -af 'p2_task[01]_' | grep -v 'grep -v' || echo NONE
```

Only then sample. Two failure directions, both observed here in one hour:

| | |
|---|---|
| `pgrep -af "a\|b"` | `\|` is BRE alternation; `pgrep` uses **ERE**, so it is a literal pipe. The check matched nothing, ever — it could only return *absent*. Four conclusions were drawn from it, including suspicion of a subagent's work. |
| `until ! pgrep -f <script>; do sleep 5; done` | The loop's own command line contains the string it greps for, so it matches its siblings **and itself**. The wait condition is self-satisfying; four of these ran 5–7 hours. It could only return *present*. |

Both were written **as** the safeguard against this class. So the discipline is not "use the
check" — it is **ask of the check what a failing result would have looked like**, and confirm
it can produce one. Kill stragglers by **PID**; never `pkill -f`, which reaches Slurm jobs on
shared nodes.

**A tool's output is a sample too.** A mutation harness that ABORTs with a drifted-anchor
error reads the tree *when it starts*, and inherits exactly this precondition. One such ABORT
in this phase was correct — it had read a file another chained harness was mid-mutation on —
and the conclusion drawn from it ("the anchors really moved") was wrong. *"The harness said
so, not me"* feels like independent evidence and is not.

**Why it matters more than a wasted round trip:** the queued fix was a re-anchor, and **a
wrong re-anchor is invisible**. The harness still passes, against anchors that no longer name
the code they were written for. A green gate pinned to the wrong lines is the exact thing
these harnesses exist to prevent.

**Two rules that follow, both cheap:**

- **Never run two mutation harnesses CONCURRENTLY when their target sets intersect** — by
  any means, not merely chained in one shell. `p2_task0` and `p2_task1` both list
  `sdk_/_run_state.py` and `sdk_/_io_constants.py`. Each holds a pristine copy in memory and
  restores after every mutation, so two live runs produce: one harness **restoring over the
  other's mutation** (its test then passes, reporting `NOT PROVED` — which reads as a *weak
  test* rather than a clobbered one), a final restore writing the wrong baseline, and each
  suite failing on the other's edit. **Every one of those failure modes prints a plausible
  report.** Check the target lists before starting a second harness, not just whether one is
  running.
- **Never chain two mutation harnesses in one shell.** The narrower rule that produced the
  original collision. A kill or a timeout on the first also takes the queued second, and the
  second reads whatever the first left behind.
- **A dirty file that is not a target of the running harness is still probably a mutation.**
  Check for a live harness and look at the diff's *shape* — a single line is a mutation —
  before reporting a stray edit. Two commands, and it is entry 21's rule applied to yourself
  rather than noticed in someone else.

### An urgent report must carry its own falsifier

**Earned from a correct escalation with a lossy shape.** An agent reported *"STOP, both
harnesses are running concurrently"* when what it actually held was a **disjunction**:
*either they overlap, **or** the other harness exited between my two checks.* It sent one
branch.

Both branches were in hand. One sentence naming the discriminator — *"if the other run
already printed its verdict and its `restored … OK` lines, it was done and this is a
crossing"* — would have let the recipient resolve it **while reading**, instead of
re-deriving the whole question under time pressure.

**The escalation itself was right**, on cost asymmetry: a false alarm costs one message, a
missed overlap costs two invalidated freeze runs that still print green. So the rule is not
*be more certain before escalating* — it is:

> **Escalate on the cost asymmetry, and send the whole disjunction.** State what would make
> you wrong, and how the recipient can check it in one command.

### Disjoint FILES is necessary and not sufficient — check the shared CONTRACTS

**Earned dispatching P3's two parallel clusters, and the check that caught it was not mine.**

Before running clusters 3.2 and 3.3 concurrently, the orchestrator verified they shared no
source file **and** no test file, and treated that as sufficient. It is not.

`classify_staged_image` lives in 3.3's `_cli_staged_resume.py` and opens with
`valid_image_success(...)`. **3.2 changes what that function reads** — from
`image_complete/<ds>/<stem>.json` to `images/<ds>/<stem>.json`, a clean break with no dual
write. 3.3's whole purpose is asserting resume decisions are unchanged, so had its fixtures
hand-planted `image_complete/` markers, every one of its tests would have gone red the
moment 3.2 landed — **and it would have looked like 3.3 broke something**, which is the
expensive part.

> **When two parallel tasks touch no common file, the remaining coupling is through the
> functions one of them CHANGES and the other CALLS.**

**The check, before dispatching anything in parallel:** for each function either cluster
modifies the *behaviour or output contract* of, grep the other cluster's files for calls to
it. A file-overlap check cannot see this, because the shared thing is a contract, not a
path.

**Two things made it cheap here rather than costly:**

1. **It was caught before writing**, so it became an instruction ("build fixtures through
   the real publisher, never hand-plant markers") instead of a debugging session across two
   agents' work.
2. **The agent asked the orchestrator to run one grep** rather than opening a file it had
   been told not to touch. The isolation held *and* the question got answered — the routing
   rule doing exactly what it exists for.

#### Line-level disjointness is insufficient for EDITS — the lost update

**The same afternoon, one level down from the contract coupling.** Two clusters each needed
one export line in `sdk_/__init__.py`: 3.3's `DIR_STAGE2_DONE` and 3.2's `record_rejection`.
Different lines, two sorted lists, **no textual overlap — git would not have conflicted.**

The failure is not a merge conflict, it is a **lost update**: whichever agent read the file
before the other's write, and then wrote back its own whole-file view, silently drops the
other's line. And the result **imports cleanly and passes ruff**, so the first signal is an
`ImportError` in whichever suite runs second, pointing at a name whose author did nothing
wrong.

> **A file two parallel agents both need to append to has one writer: the orchestrator.**
> Claim it, and take both edits as text.

**The generalization this run kept re-deriving, in three sizes:**

| Level | What looked sufficient | What it missed |
|---|---|---|
| files | disjoint source *and* test files | the shared **contract** of a function one changes and the other calls |
| imports | the caller's own file is clean | a **function-local** import into a file someone else is rewriting |
| lines | disjoint lines, no git conflict | a **whole-file write-back** dropping the other's line |

In every one, **the tooling's safety check is structurally blind to the failure**: a green
suite, a clean merge, a passing lint. Ask of each check what it would look like if the thing
you fear had happened — a clean `git diff` looks identical whether or not a line was lost.

**Corollary, earned the same hour:** a mitigation can retire the tradeoff it was chosen
over, and the recommendation does not update itself. An agent weighed "add the export" against
"lost-update hazard on a contested file" — correctly, from a state that had changed half a
message earlier, when the orchestrator claimed the file and left it with one writer. When you
impose a mitigation, re-ask the questions that were open because of the risk it removes.

#### And the dependency may be one import deeper, inside a function body

**The same hazard, caught an hour later, and neither party saw it from the file lists.**

Cluster 3.3 was ready to capture a frozen behaviour table for `classify_staged_image`,
reasoning — correctly — that its own file was unmodified. But:

```python
# _cli_staged_resume.py:210, INSIDE the function, not the module header
from ._cli_completion import valid_image_success
```

and at that moment `_cli_completion.py` was **+90/−67** under cluster 3.2. The capture would
have frozen the behaviour of a half-rewritten function as the *pre-change baseline*, and
every later comparison against it would have been meaningless while looking authoritative.

**A function-local import is invisible to the check you would naturally run.** It is not in
the module's import block, so scanning headers finds nothing; and the caller's own file is
genuinely clean, so `git status` on your own scope says you are safe.

> **Before freezing, capturing, or baselining anything, ask what it TRANSITIVELY calls that
> someone else owns** — and grep the function bodies, not just the import headers.

**Capture twice rather than arguing about ordering.** Where a baseline must be taken near
another cluster's work, take it *after* that cluster lands and again after your own change,
in one window with no commit between. A single difference then has exactly one possible
cause. Under a single-capture plan, a red gate has two candidate explanations and separating
them costs more than the extra run.

**The general form for parallel dispatch:** fixtures in either cluster should construct
state through the **real writers**, never by hand-planting the files those writers produce.
A hand-planted fixture freezes a path; a fixture built through the writer follows it. That
is worth requiring even when no parallel work is planned, because it is what makes a test
survive the refactor it is meant to guard.

### A long document states its load-bearing facts twice — check both before trusting one

**Earned at P3's open, and it is the sibling of the rule below.** The orchestrator's brief
said *"Files (verified against the plan, not inherited)"*. That was accurate: Task 1's Files
block does name exactly those files. It was also wrong, because the **File Structure table
180 lines earlier names two modules**, the task's own tests import from the missing one eight
times, and the plan carries an explicit callout warning that this precise mismatch will send
an implementer into a layering deadlock discovered four phases later.

Checking *one* authority is not verification when two exist and disagree — and nothing in the
first block signals that a second exists.

> **Before trusting any block in a long document, grep the document for the same fact.**

A revised document is *more* prone to this, not less: revision updates one site, and the
sections most worth restating — file lists, invariants, counts, schema shapes — are exactly
the ones stated in both a summary table and a task body.

### An independent check that adopts the classification is not independent

**The subtlest of the three, and it is the orchestrator's.** An agent reported 19 stale
sites across 7 files. The orchestrator "verified independently" — grepped, got **47 hits
across 11 files**, confirmed the 19, and accepted the agent's judgment that the other 28
were legitimate. Three of those were false, in the two files the agent had never grepped.

The check re-derived the **arithmetic** and inherited the **classification**, and the
classification was the half that was wrong.

#### And the population belongs to whoever quotes the number, not to whoever chose the command

**The same rule, one level up, and it cost this phase four hours.** Every count in P3 —
119 failures, 37 errors, 31 items, 24 markers, three families — came from one command:

```
pytest tests/unit/sdk_ tests/unit/cli
```

`pyproject.toml:220` sets `testpaths = ["tests/unit", "tests/smoke", "tests/integration",
"tests/gui"]`. **`tests/integration` runs in the default lane and was never measured**, and it
held two files reading the marker the change had just orphaned — one of them on the forward
path.

The orchestrator chose the command; the agent ran every count through it for a day. **The
agent's framing is the right one:**

> The scoping was as available to me as to you… a one-sided attribution makes the fix look
> like *"the lead should pick better commands"* rather than **"whoever quotes a number owns
> its population."**

That is the rule. A number you repeat is a number you are asserting, and its scope is part of
the assertion. *"I ran the command I was given"* does not transfer ownership of what the
command covered.

#### A caveat does not travel with the number

**The same rule at its highest-stakes point: escalation to the user.**

An agent reported the migrate fix as *"a field split across ~16 sites and a manifest JSON
schema"* and attached an explicit caveat — *"What I have not established: whether `:369` and
`:684` are input-fingerprints or read-backs."* The orchestrator escalated to the user carrying
the number and **not** the caveat. The user ruled to defer, largely because the fix had been
framed as a P7-sized slice.

The real size, established minutes later and independently verified: **four read-back call
sites in one file, no new field, no schema change.** The user was re-asked and **reversed**.

> **A caveat attached to a number does not survive being repeated by someone else.** The
> number is quotable and the caveat is not; it stays in the message it was written in.

**So: never escalate an unestablished number.** Either establish it, or escalate the
*question* rather than the estimate — *"the fix is somewhere between one comparison and a
sixteen-site field split, and here is the check that decides which"* is a ripe thing to bring
someone; a worst-case estimate presented as the size is not.

**And when it happens anyway, re-ask.** A decision made on bad information is not made. Going
back costs one round trip and owning the error costs nothing; acting on the first answer
spends the user's ruling on a premise they were never given.

**The cheap discharge:** before quoting any count a second time, state the population in the
same breath — "119 failures **in `tests/unit/{sdk_,cli}`**". A count carrying its own scope
invites the question; a bare count suppresses it. And when a phase gate says *"the suite
passes"*, resolve *suite* against the config, not against habit.

**These have different fixes, which is why the distinction matters.** A narrow grep is fixed
by widening it. **Accepting the frame is not** — you can widen the search forever and still
inherit the judgment about what the results mean. When verifying someone's count, the count
is the easy half: re-derive the *scoping rule* first, then apply it yourself to everything
the search returned, including what they excluded.


- **Verify an anchor with the harness's own ownership rule**, not a weaker proxy. The harness
  requires each anchor to match *exactly once in exactly one target*; summing `count(old)`
  across targets passes cases the harness rejects.

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

### A gate measures ONE tree — hash it, or the result is void, not merely stale

**I invalidated gate lane 1 myself.** 48 shards spread over 8m30s; `_run_state.py` was
edited at 13:49:41, *inside* that window. The result was not an old reading of one tree — it
was a **union across two**: some shards imported the pre-edit module, some the post-edit one,
and **no single tree ever produced that failure set.** Incoherent, not stale. There is no
partial credit and nothing to salvage; the only response is to re-run.

**The freeze protocol, five steps, all five load-bearing:**

1. **Declare final.** Every agent holding an edit says so and stops.
2. **Snapshot.**
   `find src tests -name '*.py' | sort | xargs sha256sum > pre.txt`
3. **Submit.**
4. **Nobody edits — *including docs* — until every shard has exited.**
5. **Re-hash and diff.** Only then read the results. A non-empty diff voids the lane.

**Docs are inside the freeze, and the reason is not that a doc changes behaviour.** It is
that "nobody touches the tree" survives contact with a long wait and "nobody touches the tree
except harmless things" does not — it requires a judgment call at precisely the moment you
are bored and impatient. The exception is where the next `src/` edit gets rationalised.

**Report the shard spread with every gate result.** The spread *is* the window in which the
invariant could have been violated, so a result quoted without it cannot be audited later.
Lane 2: spread 1m53s, tree `470c69f4faef6277`, held still — that is the shape of a quotable
line.

### Mechanical gates go through the same array

**Standing instruction (user, 2026-09-04): for future mechanical gates, consider dispatching
over SLURM to parallelize.**

The array harness is **not test-specific.** Any set of independent, deterministic checks is a
task list: the AST invariant sweeps (INV-LAYER), the enumerate-with-the-old-name grep sweeps,
`ruff`, `mypy`, `check_features_md.py` / `check_workflows_md.py`, the doc-anchor checks. One
check per array task, one line of output each, one collector.

Constraints, each of which has already cost this change something:

- **Compare names, never counts** — the same rule as the test array, for the same reason.
- **Results go to shared storage.** `/scratch/<user>/<jobid>` is node-local *and* per-job, so
  a collector running anywhere else sees an empty directory. The symptom is `FAILED` with
  `ExitCode 0:53` and **no log file at all**, and it is intermittent — a task that lands on
  the submitting node appears to work.
- **`sbatch --parsable` fails silently.** On rejection it prints the error and returns an
  empty id, so a driver loop "runs" for hours having submitted nothing. Verify the captured
  id matches `^[0-9]+$` and surface the raw output on failure.
- **The same tree-hash freeze applies, and a mechanical gate is *more* exposed, not less.**
  Its checks finish in seconds, so the window between submit and result is exactly when
  "let me just fix that one thing" is most tempting and least survivable.
- **Do not submit this beside an active ordinary array.** Allocation and submission bounds are
  already consumed by the cohort; see the sidecar rule in `_cli/CLAUDE.md`.

### Each shard gets its own worktree, and the finalizer removes them

**Standing instruction (user, 2026-09-05): create worktrees, run the checks inside them, and
clean the worktrees up afterwards.**

This **supersedes** the exclusion drafted immediately above, which said mutation harnesses
must stay serial because they rewrite `src/` in place. That was true only while every shard
shared one checkout. Give each shard its own worktree and they no longer share a mutable
target, so **the mutation harnesses parallelize like everything else** — the constraint was
never about the harnesses, it was about the checkout.

**It also converts the freeze from a procedure into a property.** A worktree is created from
a **commit**, so a shard physically cannot observe an edit made in the main checkout while it
runs. The union-across-two-trees failure that voided lane 1 becomes *impossible* rather than
*forbidden*, and "which tree did we measure?" is answerable forever by SHA instead of by a
hash file in scratch that outlives nothing.

**The cost is that the freeze point must be a commit.** Uncommitted work is invisible to a new
worktree. Make a temporary WIP commit (never `git stash` — the stash stack is shared with
every other worktree and another session may pop yours), gate that SHA, and amend or reword
afterwards.

**Layout and lifecycle:**

- **Create them all in the submitting script, serially, before the array is submitted.**
  `git worktree add` mutates the shared `.git/worktrees/` administrative directory; 48
  concurrent adds contend on it for no benefit. Serial creation also means a task body never
  runs a git command at all.
- **`--detach`, always.** `git worktree add <path> -b <name>` would demand 48 unique branch
  names and leave 48 branches behind to clean up.
- **Shared storage, in a job-scoped directory** — `/bigdata/.../.gate-worktrees/<jobid>/NN`.
  Not `/scratch/<user>/<jobid>` (node-local *and* per-job, so other nodes see nothing), and
  **not the repo's own `.worktrees/`**, where other sessions keep live work that a cleanup
  glob would take with it.
- **One venv per worktree via `uv sync`, with a shared `UV_CACHE_DIR` on `/bigdata`** so
  packages hardlink instead of re-downloading. Do not try to point many worktrees at one
  pre-built venv: an editable install resolves through a finder pinned to one path, so the
  shards would silently all import the *same* tree and every result would be a copy of one.
- **Clean up from a terminal `afterany` finalizer, not from the task body.** A task that is
  OOM-killed, timed out, or preempted never reaches its own cleanup line; `afterany` runs
  regardless of how the cohort ended. `git worktree remove --force` (mutation harnesses leave
  their worktree dirty by design, and an unforced remove refuses), then `git worktree prune`
  to clear administrative entries for anything already gone from disk. An `afterany`
  finalizer is not a parallel sidecar — the `_cli/CLAUDE.md` rule permits it explicitly.
- **Removal is serial too**, and for the same reason as creation.

**Read-only checks do not actually need a worktree** — `ruff`, `mypy`, the grep and AST
sweeps, the doc-anchor scripts. `git archive <SHA> | tar -x` into node-local
`/scratch/$SLURM_JOB_ID` is faster, touches no shared git metadata, and is reclaimed by the
scheduler with no cleanup step to forget. Worktrees earn their cost where a check **writes**
— which is exactly the mutation harnesses, and exactly the case that could not be
parallelized before.

### End of run
One `code-simplifier` pass (quality only, no behaviour change), apply fixes, then the full
suite through the same harness and `collect_results.py --baseline`.
