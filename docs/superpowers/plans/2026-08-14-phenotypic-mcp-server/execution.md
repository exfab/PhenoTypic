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

### C8 and C9 — the two tasks added after the refinery closed

*(Naming note: `C8`/`C9` in `MAIN-MERGE.md` are **L1 SLURM validation job**
labels, a different namespace. These are clusters.)*

| # | Tasks | Shape | Why | Files |
|---|---|---|---|---|
| **C8** | T19 (P8 manifest) **+ GEN-18's four flags** | **Seam** | One risky wiring point: what the server can express as argv. Isolated despite being small — it sits on the irreversible full-deploy path, and `argv_digest` is now a *bound row of a consent-carrying token* | `phenotypicCLI.py`, `_services/argv.py` |
| **C9** | T20 (`RunRegistry` lock order) | **Seam** | Concurrency correctness in shipped code. Tiny, and isolated precisely because risk ≠ size | `_services/runs.py` |

**Fold GEN-18 into C8 rather than tracking it separately.** T19 must already add
a manifest field to `RunConsoleState` and a branch to `to_argv`; `--restart`,
`--slurm k=v`, `--gpu-slurm` and `--gpu-shards` are the same file, the same
function, the same shape of change. Opening `to_argv` twice for one class of
defect is the waste the cluster rule exists to prevent.

**C8 and C9 are parallel-worktree candidates — but NOT against C6.**

*Corrected 2026-08-19, before dispatch.* An earlier version of this line claimed
C8 had zero overlap with every other cluster. **It does not.** Task 16, inside
C6, carries the instruction *"Edit `src/phenotypic/_services/argv.py`, not
`gui/tune/_run_argv.py`"* (`phase-1b:103`) — because Task 8 already promoted the
tune argv builder there. Task 19 edits the same file for the manifest field
(`phase-1b:1236`). **Same file, and the collision rule that forced C6 to exist
applies here too.**

| Pair | Safe to parallel? | Shared file |
|---|---|---|
| C8 ∥ C4 | **yes** | C4 owns `catalog.py` only |
| C9 ∥ anything | **yes** | C9 owns `_services/runs.py` alone |
| **C8 ∥ C6** | **NO** | `_services/argv.py` |
| C8 ∥ C5, C8 ∥ C7 | yes | no shared file |

**So: `C8` and `C9` run in worktrees alongside `C4`, and `C8` must be gated and
merged before `C6` is dispatched.** That sequences naturally — C6 already follows
C4 — but it is a constraint, not a coincidence, and dropping it would put two
agents in one function.

This is the anti-pattern the skill names outright ("overlap kills parallelism —
check shared files before fanning out"), and the check that caught it was
mechanical: parse every task's `Files` block, intersect per cluster. Worth
re-running whenever a task is added, since Tasks 19 and 20 were added after the
original clustering and that is exactly how the overlap got in.

**Sequence:** `C1 → C2 → C3 → [1a gates] → C4 → C6 → C7 → C5 → [1b gates]`
with **C8 ∥ C9 ∥ C4** — the two seams run in their own worktrees alongside the
first keystone, and each takes its own gate before merge.

**C8 was blocked pending the OME-Zarr impact review; it is now UNBLOCKED**
(2026-08-19). The store is one-per-input-image but lives under `results/`, and
ingesting third-party OME-Zarr as pipeline *input* is an explicit non-goal — the
projection is write-only. Input images stay ordinary files, so a line in the
manifest is unchanged and T19's two-file scope stands.

### C7 — P1 (JournalStorage), moved into Phase 1b (user ruling, 2026-08-19)

§7 sequenced P1 after v1 ("MCP v1 ships without P1"). **That is wrong for this
workload.** Filamentous-fungi pipelines cost ~30 min per evaluation, so a
200-trial arm is ~100 h serial and a 3-arm campaign ~300 h. Worse than slow:
§1.5 has a `W2` routed **local** hold the single `LocalComputeSlot` for its
entire subprocess lifetime, so one local study would block every
`pipeline_probe` from every subagent for days. Without distributed tune, v1 is
not a reduced capability — it is a deadlock generator.

P1 is empirically unblocked: **C9 (job 27555152) persisted 400 trials across 8
distinct nodes**, with ~4,500× append-rate headroom at 30-min evaluations
(see `MAIN-MERGE.md`).

**C7 = P1**: the five storage construction sites (`_run.py:475`, `_run.py:785`,
`_worker.py:50`, `_optuna_store.py:106-109` + `gui/tune/_callbacks.py:871`,
`strategy/_optuna.py:239`) plus **B1–B4** — the transient-retry predicate must
become backend-aware, the "read-only" Monitor open must stop creating the file,
the Monitor's timeout bounding must not assume file storage cannot network-hang,
and `journal.log` never compacts (state expected sizes).

**Must follow C6, cannot parallel it:** two of the five sites are in
`tune/_tune_cli/_run.py`, which C6's T15/T16/T18 already own. Same file, same
collision rule that forced C6 to exist.

**Scope note:** §7 calls P1 "an engine change needing its own spec". It is the
largest single item in Phase 1b — consider whether it splits (storage dispatch |
B1 retry predicate | B2/B3 Monitor safety) at the C7 gate rather than up front.

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

| Cluster | Agent | Status | Gate verdict |
|---|---|---|---|
| plan review | `plan-reviewer` | **DONE** — silent for ~3 days, delivered only when asked directly | 9 blockers, 8 improvements; all folded in |
| **C1** (T1,2,2.5,3,4,5,8) | `C1-promotion` (Opus) | **COMPLETE & MERGED** — 7 commits, `af0c8596e`..`1292a946b` | `C1-gate-review`: 3 blockers, all fixed in `3d7a4f16a` |
| **C2** (T6,7) | `C2-space-split-v2` (Opus) | dispatched 2026-08-18 (v1 produced nothing and never replied) | — |
| C3 (T9) | — | pending; B8 fixed so it is dispatchable | — |
| C4 (T10a,10b,10c,11,12) | — | pending; B5 splits T10 into three | — |
| C5 (T13,14,17) | — | pending; must follow C4 | — |
| C6 (T15,16,18) | — | pending; B3/B4/B7/B9 resolved in phase-1b corrections | — |

## Agent-reliability notes (earned the hard way)

Three of five agents this session failed to deliver through the message channel:
two completed real work and went silent until asked directly; one produced
nothing at all. Consequences adopted:

- **Agents write progress to a committed file**, not only to messages. A file in
  the repo cannot be stranded; `C2-PROGRESS.md` is the first use.

  **Correction (2026-08-18):** the orchestrator concluded mid-cluster that its
  replies to C2 were not arriving, and said so. That was **wrong**. C2 received
  all four. What actually happened is a timing artifact: two approvals were
  delivered together, immediately *after* C2 had written its "still blocked"
  report — so every report was composed before the corresponding reply landed,
  which from the sender's side is indistinguishable from replies vanishing.
  Diagnose a delivery failure from the *receiver's* account, not from the
  pattern of your own outbox. The file channel is still worth keeping, but for
  a different reason than the one claimed: it let C2 confirm the decision had
  not changed between reading and acting.
- **Require an acknowledgment as the agent's first action**, so "working" is
  distinguishable from "never started" within minutes rather than hours.
- **Idle ≠ finished.** Poll the tree (`git log`, target files, recent mtimes)
  rather than trusting an idle notification.
- **Never run two implementation agents in one working tree.** They share a git
  index; incident X1 was exactly this, and a second occurrence would not
  necessarily be cosmetic.

---

## PHASE 1a — CLOSED 2026-08-19

Ten tasks (1, 2, 2.5, 3, 4, 5, 6, 7, 8, 9), three clusters, three gates, one
merge of `origin/main`, one simplify pass. Exit gate green on every item:

| Check | Result |
|---|---|
| `tests/unit/services` | 61 passed |
| `tests/unit/cli` | 552 passed |
| `tests/unit/gui` + `tests/integration/gui` | 1746 passed, 3 skipped |
| `tests/gui` | 662 passed, 1 deselected |
| `check_features_md.py` | OK — 444 rows, 370 shipping |
| `check_workflows_md.py` | OK — 20 workflows, 20 capture fns, 20 dispatched |
| mypy (cold cache) | 417 errors / 124 files — **empty diff** vs the pre-phase tree |
| ruff | clean on every changed path |

**What the gates actually bought.** Ten blocker-class findings that a green suite
would not have surfaced:

- **C1 gate** — the purity gate missed nested subpackages; two lint sinks named
  `_` collided into new mypy errors when two modules merged; the tier claimed to
  be GUI-free in two docstrings while `runs.py` imported `gui.shell._classifier`.
- **C2 gate** — the allowlist exactness pin covered one key of two, so a one-line
  edit dissolved the boundary with 59 tests still passing; seven identity
  assertions could not fail because `typing.Literal` is `_tp_cache`d; the optuna
  guards were inert locally and tautological in CI.
- **C3+merge gate** — main's identity arrays (`EXPECTED_WORK_IDS`,
  `EXPECTED_INPUT_SHA256S`) have **no test coverage at all**: empty or corrupt
  them and all 552 CLI tests stay green. Pre-existing on main, reported upstream.
- **Plan review** — the phase's central architectural claim was wrong: the eager
  `gui/shell/__init__.py` was the Dash leak, not the modules, and Task 5 would
  have failed a gate it was forbidden to weaken. Fixed by adding Task 2.5.

**Three claims the orchestrator wrote and agents disproved:** that the eager
`__init__` files were out of scope; that mypy's error count was unstable (it was
cache warmth); that `deploy_plan` is a `W0` call (post-merge it reads every input
image twice).

Two incidents, both from sharing one working tree: `git add -A` swallowed an
agent's staged rename, and an agent kept working in the pre-move directory,
duplicating three tasks onto a detached head. Both recorded; practices adopted.

---

## Pre-dispatch gate outcome (2026-08-19) — what changed before anything ran

`plan-reviewer` returned **not safe to dispatch as written**: nine blockers, four
Majors. Applied:

| Item | Change |
|---|---|
| **B-2/B-3/B-4 → USER-31** | **C9 (Task 20) is WITHDRAWN from Phase 1b.** Four methods nest the locks, not one; inverting only `allocate` produces an **ABBA deadlock** against `compare_and_set` (the status-poll hot path) that resolves via the 30 s timeout and then raises out of a method documented never to raise. Its acceptance test passed against the broken fix. The complete fix changes `publish_if_current_generation`'s **documented** contract, which holds an external callback inside the critical section. That is a design pass, not a task |
| **B-1a** | Task 16's `Files` block now names `_services/argv.py`. **This is why the overlap check missed the C6/C8 collision** — it parses `Files` blocks, and the file was only mentioned in prose. The symptom was fixed earlier; this fixes the input |
| **B-5/B-7/B-8/B-9 → USER-30** | Task 19 rewritten: **five files, not two.** Manifest passed *alongside* `--input <parent>` (pre-decided — pointing `--input` at the manifest makes work IDs basename-only, which **diverges** from the parent run's IDs for the same images. *Corrected post-implementation: the original 'collides across datasets' rationale was false — `compute_work_id` hashes `dataset` separately, proven by mutation. The decision stands on divergence*); `load_staged_manifest` reuse claim dropped; `gui/` emitters **promoted** rather than inlined; resume participation added; plus an argv **coverage test** |
| **C7 split** | **C7a** = the five storage construction sites + backend dispatch (mechanical, one idiom). **C7b** = B1–B4 (retry predicate, Monitor safety, compaction — behavioural judgment, each needing its own test). Interleaving a mechanical rename with concurrency semantics is where reviewer attention degrades |
| **M-1** | Task 10a = lift the literal, **one** reader, assert the constant equals the old literal *in order* (resolution is first-match). "Both consumers" moves to 10b |
| **M-3** | `README.md:253` uses **C7** for the cross-node JournalStorage validation *job*; execution.md uses C7 for the P1 *cluster*. Third namespace collision in this document — cluster labels are `C4–C9`, SLURM job labels live in `MAIN-MERGE.md` and `README.md` |
| **M-2** | Citation drift in five tasks. The dangerous one was Task 20's: `FileLockTimeout` at `_cli/_cli_file_locking.py:50` — but `runs.py:72` imports `sdk_._file_locking`, where it is **`ArtifactLockTimeout`**. A class of the cited name exists at the cited path, so the citation looks right, is wrong, and a test written against it never fires |

**Corrected scope claim.** An earlier line here read *"T19's two-file scope
stands"*. It does not — see B-9. The manifest reaches `ExecutionConfig` and
`ProcessingState` because `validate_resume_compatibility` compares `input_path`
and nothing about the image set, so two different manifests under one parent are
resume-compatible. Since the server now calls that function directly (spec §5.4),
the guard and the gap are the same code path.

**Revised sequence:** `C4 → C6 → C7a → C7b → C5`, with `C8 ∥ C4`, and **C8 gated
and merged before C6** (shared `_services/argv.py`). C9 no longer participates.

---

## PENDING: relocate the authoritative checkout back to `exfab`

**User's decision (2026-08-19): `/bigdata/exfab/anguy344/PhenoTypic` is the tree
they want to work in.** Work moved to `iwheeldonlab` earlier only because `exfab`
was at 100% of its 36T quota. **That is resolved — `exfab` is now at 50% (18T
used, 19T free)** — so the reason for the split no longer exists.

**Do this only when C4, C8, C6, C7a, C7b and C5 have all settled.** Relocating
mid-cluster means an agent's worktree and the authoritative branch disagree about
where HEAD is, which is the whole class of failure that cost four agent-cycles
today.

### Procedure

1. **Gates clear**, and `c8-manifest` is merged into `feat/mcp-server` in
   `iwheeldonlab`. Do not relocate an unmerged cluster.
2. **Fetch, do not copy** — a local fetch, no network, no remote:

       git -C /bigdata/exfab/anguy344/PhenoTypic \
           fetch /bigdata/iwheeldonlab/anguy344/PhenoTypic \
           'refs/heads/feat/mcp-server:refs/heads/feat/mcp-server'

   `rsync`/`mv` of the directory would break the ten worktrees registered under
   `exfab`, including the live `worktree-ome-zarr-image-store`.
3. **Check it out** in `exfab` and verify HEAD matches `iwheeldonlab`'s tip.
4. **Retire the `iwheeldonlab` copy immediately** — rename its branch or remove
   the clone. Leaving two checkouts both holding a live `feat/mcp-server` simply
   re-creates the collision in the opposite direction, and the next agent to land
   in the *wrong* one would be landing in what is now the stale copy.
5. **Remove the `c8-manifest` worktree** at
   `/bigdata/iwheeldonlab/anguy344/PhenoTypic-worktrees/c8-manifest` once merged.
6. **Delete `STALE-DO-NOT-USE-see-iwheeldonlab`** in `exfab` — it is `b4401dee`,
   a strict ancestor, holding nothing unique.

### Do not lose in the move

- `rescued/ome-zarr-image-store` and `rescued/streamlit-run-monitor` in
  `iwheeldonlab` are **local mirrors** of branches whose originals live in
  `exfab`. They can be dropped after the move *only* because the originals are
  there — verify that before deleting anything.
- **origin was missing 9 commits** of `worktree-ome-zarr-image-store` as of
  2026-08-19. That branch is still single-copy-ish and worth pushing, which is
  the `ome-zarr-cli` session's call, not ours.

**Net effect:** one authoritative checkout, at the path the user actually works
in, and the `bigdata_exfab` symlink — the session's default cwd — finally points
at the right tree instead of a stale one.

---

## Pre-existing defects found by the Phase 1b gates — report against `main`

Both are live on `main` today, independent of the MCP work, and both were found
only because a gate mutated code rather than reading it.

| Defect | Site on `main` | Effect |
|---|---|---|
| **Zero-field pydantic models fall through to `inspect.signature`** | `gui/_operation_registry.py:353` — `if model_fields:` is falsy for a model with no fields | The GUI renders a **phantom required parameter `data`** for 14 of 143 operations, including 6 of 11 measurers. Passing it raises `ValidationError: Extra inputs are not permitted`, so the field the GUI shows cannot be filled. Fix is one character class: `if model_fields is not None:` |
| **`_schema_description` looks up the Python name, not the alias** | `tune/_search_space/_infer.py:174` | `_schema_description(RemoveGridOutliers(), "cutoff_multiplier")` returns `""`. Cosmetic — descriptions silently missing for aliased params |
| **`RunRegistry` nests an in-process lock inside a 30 s file lock** | `_services/runs.py:316/330`, and three sibling methods | One thread on a contended mount blocks every other thread from any registry method. Tracked as CONC-8; **withdrawn from Phase 1b (USER-31)** because the complete fix touches four sites and a documented public contract |

**Worth stating plainly:** the first two were found by a reviewer *mutating* the
implementation and watching which tests survived, not by reading the diff. The
third was found by a reviewer checking a citation. None would have surfaced from
a passing suite — which is the argument for keeping the mutation requirement in
every cluster brief for the rest of this phase.

### AUTHORIZED: execute the relocation automatically at the next quiet tree

**User authorization, 2026-08-20: do it without checking back.**

**Trigger — all four must hold:**
1. C6 has landed and passed its cluster gate.
2. No agent is mid-cluster (`git status --porcelain` shows nothing but untracked reports).
3. C7a has **not** been dispatched.
4. `c8-manifest` is merged (done: `b753df3c6`).

Move *before* C7a/C7b/C5, not after the phase. Every cluster dispatched against
`iwheeldonlab` is another brief carrying a path about to become wrong, and the
remaining three can start against `exfab` cleanly.

**Procedure — in this order, aborting on any mismatch:**

```
LIVE=/bigdata/iwheeldonlab/anguy344/PhenoTypic
EXFAB=/bigdata/exfab/anguy344/PhenoTypic

# 1. capture the tip we are moving
TIP=$(git -C $LIVE rev-parse feat/mcp-server)

# 2. local fetch -- NEVER rsync/mv: ten worktrees are registered under EXFAB,
#    including the LIVE worktree-ome-zarr-image-store another session uses
git -C $EXFAB fetch $LIVE 'refs/heads/feat/mcp-server:refs/heads/feat/mcp-server'

# 3. verify byte-identical before trusting it
[ "$(git -C $EXFAB rev-parse feat/mcp-server)" = "$TIP" ] || abort

# 4. check it out there
git -C $EXFAB checkout feat/mcp-server

# 5. retire the LIVE copy IN THE SAME SITTING -- two live feat/mcp-server
#    checkouts re-create the collision with the roles reversed
git -C $LIVE branch -m feat/mcp-server MOVED-TO-exfab-do-not-use

# 6. drop the merged worktree
git -C $LIVE worktree remove $LIVE-worktrees/c8-manifest

# 7. only now, delete the stale marker branch (b4401dee, a strict ancestor)
git -C $EXFAB branch -D STALE-DO-NOT-USE-see-iwheeldonlab
```

**Before deleting anything else, verify — do not assume:**
- `rescued/ome-zarr-image-store` and `rescued/streamlit-run-monitor` in `$LIVE`
  are mirrors of branches whose originals live in `$EXFAB`. Confirm the originals
  are still there (`git -C $EXFAB rev-parse <branch>`) before dropping the
  mirrors. They were rescued because the originals were single-copy.
- Do **not** touch `$EXFAB/.worktrees/worktree-ome-zarr-image-store`. Another
  session commits there.

**Then** dispatch C7a against `$EXFAB`, and update every remaining brief's path.

---

## C6 gate: Task 18 blocker — a killed worker publishes a phantom winner

**Verified against source. Two defects compose into a default-path failure, no
`force` flag needed.**

- **B1** — `_optuna_store.py`: `_to_trial` maps `frozen.value is None` to
  `score = 0.0`, and under the minimize-cost convention **0.0 is the best
  possible cost**. `best()` (`:304-309`) filters only `t.failed`, so a trial left
  `RUNNING` by a Slurm-killed worker is "valid" and wins `min()` over every real
  trial.
- **B2** — `_finalize.py`: `n_seen = len(store.trials)` counts
  `COMPLETE|PRUNED|FAIL|RUNNING`, while the budget it compares against is what
  `OptunaStrategy.is_exhausted` measures as `COMPLETE + PRUNED` only. The gate
  opens early by `(#failed + #in-flight)`.

B2 opens the gate; B1 then picks the orphan. Reproduced end-to-end through the
real entry point: 5 real trials (best cost 0.30) plus one orphaned `RUNNING`,
budget 6 →

    best_params.json = {"trial_number": 5, "score": 0.0, "params": {},
                        "selection": "single_best"}

`params={}` means `prepare_best_from_run` exports the **untuned base pipeline as
the tuned optimum, with a perfect reported cost**. No error, no warning,
`best_params_written=True`.

**Why no test caught it:** no fixture can produce a `RUNNING` trial. The suite is
structurally incapable of reaching the state.

**Fix:** filter to terminal trials before both the gate and the winner selection;
add an orphaned `RUNNING` trial to `_build_study`. One change closes both.

**Not shipping** — nothing calls `finalize_distributed_study` today. **It must not
be wired in Phase 2 as-is**, and C7a is editing the same code path.

### F1 — marker v2's read side is untested (false green)

`_build_study` hardcodes `"nrows": None, "ncols": None`, and no test writes a
non-null `nrows` into a marker. Replacing the `_load_images(images_dir,
nrows=…, ncols=…)` call with a bare one: **18 passed, exit 0.** The write side is
pinned; the half that prevents the wrong generalization verdict is not.

### Real coverage gap the gate surfaced

C8's argv coverage gate is **structurally blind to `phenotypic.tune`'s argparse
flags** — `_cli_option_flags()` reads only `phenotypic_cli.params` (Click) and
`_EMITTING_FUNCTIONS` excludes `tune_run_tail` by design. Consequence:
`--nrows`, `--ncols`, `--slurm-constraint` are **not emittable from the service
tier**, so every MCP- or GUI-launched tune run records `nrows: null` — which
defeats marker v2 on exactly the path the MCP server will use.

### Non-blocking

- **N1** — the GUI marks a `--screen --slurm` plan **valid** and emits the argv,
  which Task 15 now hard-refuses. A silent drop became a post-launch crash with
  no preflight guard. One line in `build_tune_command`.
- **N2** — `--slurm-mem 8G --slurm mem_gb=16` still emits `#SBATCH --mem` twice.
- Tasks **15 and 16 are clean**; the gate replicated all of C6's mutation claims
  and refuted its own deletable-wiring hypothesis (1158 passed with only the
  injected mutation failing).

---

## Process lesson: mutation-test in an isolated copy, never the shared tree

The C5 gate reported an edit to `subset/_selector.py` being reverted under it
within minutes. **Not a hook** — the two active git hooks are graphify
graph-rebuilders that write only to `graphify-out/`, and the `SessionStart` hook
runs `uv sync`. The cause was a concurrent agent committing while the gate had an
unstaged mutation in the same tree.

The gate handled it correctly: it moved to a `git archive` copy under scratch with
a `PYTHONPATH` override and re-derived everything there. Its results stand.

**Standing rule for the rest of this project:** any agent doing mutation testing
works in an isolated copy. In-tree mutation while another agent is live produces
results that cannot be trusted — and the failure is silent, because a reverted
mutation simply looks like a surviving one, which reads as a false green that
is not real.

This is the second coordination hazard from running several writers in one tree;
the first was an agent `--amend`-ing a commit that was not its own. File-ownership
boundaries have held throughout — no two agents have edited the same file — but
staging and working-tree state are shared whether the files are or not.
**Prefer a worktree per writer**, as C8 had; its merge was the cleanest of the
phase.

---

## The defect that recurred after being diagnosed and written down

**FLOW-40, found in the pre-dispatch gate:** USER-26 reached the spec's defining
sections and **no plan task owned it**. The plan's own coverage claim — *"every
P2–P7 item has an owning task"* — was the sentence concealing it, because it
enumerated a fixed range and silently stopped checking once §7 gained P8.

I recorded the lesson at the time: *a coverage claim must name the range it
checked, or it stops checking.*

**Then the phase review found X1: USER-33 reached §5.4 and §6.2 — error code
`sample_excludes_manifest` and all — and no task owns it either.** Written into
the spec by me roughly an hour after recording the lesson.

**Why the lesson did not take.** I applied it to the *spec* and never to the
*task list*. The defining-sections map (`refinery/defining-sections-map.md`)
lists "plan decision records and phase task docs" among defining sections, but
every ruling I propagated after building it, I propagated into spec files only,
then checked the spec and called it applied.

**The check that would have caught both**, and which belongs in any future
phase's exit gate:

> For every USER ruling, grep the plan's task documents for an owning task —
> not the spec. A ruling with a defining-section home and no task is a design
> decision nobody will build.

The failure mode is quiet in a specific way: the spec looks complete, the tests
pass, and the missing behaviour is *absent* rather than wrong — so nothing fails
until an agent relies on a guarantee that was only ever written down.

**Both instances sat on the consent path** — the manifest a human approves, and
the refusal that stops a sampled subset silently shrinking it. That is not
coincidence: the rulings that arrive late are the ones responding to a reviewer's
finding, and reviewers find the most on the paths that matter most.
