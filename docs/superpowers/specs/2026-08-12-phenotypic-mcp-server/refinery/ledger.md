# Concern ledger — PhenoTypic MCP server

Append-only. Statuses: `open` · `resolved (round N: what changed)` ·
`settled-by-user (round N: ruling)` · `conflict (vs ID)` · `advisory`.

## Pre-loop rulings (permanent — no reviewer may re-raise absent new evidence)

### USER-1 [settled-by-user (round 0)]
Performance/NFR anchor. **Ruling:** interactive responsiveness for `W0`/`W1`;
`W0` < 1 s and never blocking the event loop; `W1` bounded by
`probe_max_images` + `probe_timeout_s`; `W2`/`W3` submit-and-poll with no
latency requirement. Written into the spec as §1.6.1.

### USER-2 [settled-by-user (round 0)]
Rotating specialist = **concurrency**. Parallelism is the spec's central
mechanism and three open hazards are concurrency-shaped.

### USER-3 [settled-by-user (earlier)]
MCPB packaging **rejected**; local stdio confirmed. Bundling would produce two
divergent copies of `phenotypic`.

### USER-4 [settled-by-user (earlier)]
SDK = PyPI **`fastmcp` 3.x** (D1a), superseding the official SDK's bundled
FastMCP 1.0.

### USER-5 [settled-by-user (earlier)]
**Elicitation adopted** (D6) for `campaign_approve` and the §10.5 promotion
gate — shaped now, implemented in Phase 2C, gated on a live test of whether it
surfaces from a subagent's call.

### USER-6 [settled-by-user (earlier)]
No server-side `plate.nrows`/`ncols` backstop (D3) — grid sections are not
always filled, so the product is a poor proxy for expected count.

### USER-7 [settled-by-user (earlier)]
Reviewers run at **cluster boundaries**, not per task (I8).

## Round 1 concerns

### simplicity-reviewer — round 1, VERDICT REVISE (17 concerns)

Full report retained in the round-1 transcript. **All spec-change flagged — the
loop is paused on these; none applied.** Batched for a single user decision
alongside the other three reviewers rather than presented piecemeal.

**Its framing, which is the finding underneath the findings:** four prior passes
each ADDED a mechanism and carried its justification forward, while the sections
that later RETIRED those justifications were written afterward and never
propagated back. **The spec contains the argument for its own reduction in four
places** — §9.3.5, §10.5, §10.3, §7 P5 — without following through.

| ID | Sev | Concern | Status |
|---|---|---|---|
| SIMP-1 | Major | Promotion is a second token + 2 tools + an artifact for a decision `deploy_plan {scope:"full"}` already gates. §10.5's own text: a full run "has no other way to obtain" a plan token. Promotion's real contribution is *content*, not a second lock. Elicitation (USER-5) moves to `deploy_start {full}` — better placement, at the point of spend | open · spec-change · needs-user-input |
| SIMP-2 | Major | `assay_put`/`assay_get` validate a file §9.3.5 says "the server never acts on" — the one place deliberately not using `extra="forbid"`, serving a JSON file whose only consumers are a human and a skill | open · spec-change |
| SIMP-3 | Major | `mode`/`layer`/`sample` serve no workflow section, force §5.4's most intricate conditional, and are **the largest OME-Zarr collision** | open · spec-change |
| SIMP-4 | Major | `phenotypic-mcp setup` is multi-harness distribution machinery for one cluster; §9.7 concedes the per-harness shapes "must be verified at implementation time" | open · spec-change |
| SIMP-5 | Major | Task 15/P4 exists only because §4.2 exposes `screen` — opt-in, default off, no stated v1 payoff, and it arrives with four mandatory corrections | open · spec-change |
| SIMP-6 | Major | P5 and §5.2.1's expressibility check are both being built; §7 says the second is vestigial once the first lands | open · spec-change · needs-user-input |
| SIMP-7 | Minor | `pipeline_patch.exploration` reconstructs a step counter from the journal each call to emit an advisory string the agent already knows | open · spec-change |
| SIMP-8 | Minor | `workspace_lineage`'s load-bearing hop was retired when §10.3.1 made `subset_id` mandatory ("comparable by construction") | open · spec-change |
| SIMP-9 | Minor | `EmbeddingSubsetSelector` ships as a class built to raise, plus `W2` routing for a selector that does not exist | open · spec-change |
| SIMP-10 | Minor | `_services/catalog.py` and `staging.py` have one consumer each; §1.4 defines the tier as *shared by two surfaces* | open |
| SIMP-11 | Minor | `next_recommended`/`blocked` recompute per call what `instructions` + tool descriptions do once per session | open · spec-change |
| SIMP-12 | Minor | `campaign_status {since}`'s stat cursor optimizes a call §4.4 says to poll on a human timescale | open · spec-change |
| SIMP-13 | Minor | Four trims the spec already argues for: `pipeline_diff`, `get{raw}`, `save_overlay`, `campaign_get` | open · spec-change |
| SIMP-14 | Minor | `catalog_measurements` duplicates `produces_columns`; Task 10c force-imports the NN stack on the first catalog call (compounds F3) | open · spec-change · needs-user-input |
| SIMP-15 | Advisory | `traits.yaml` is a rules format nothing parses; §9.3.4 says "the server never reads this file" | open · spec-change |
| SIMP-16 | Advisory | §1.5's local queue/reconciliation machinery is sized for a workstation, not a login node | open · spec-change |
| SIMP-17 | Advisory | §7 P3.4 and §8.3 still justify the digest by a comparability path §10.3.1 made unfailable | open · spec-change |

**Net effect if SIMP-1, 2, 8, 13, 14 are accepted: 32 tools → 24**, nine groups → eight.

**Its answers to the brief's two direct questions:** the prerequisite stack is
*mostly load-bearing* — P2, P3 (10a/10b/11/12), P6, P7 and the digest all have
real v1 consumers — with **P4 the clear exception**; and the `_services` tier is
sound, but Phase 1b has begun using it as a default destination rather than the
shared-by-two-surfaces contract §1.4 defines.

**Coverage:** all of §1–§10, plan README, execution.md, phase-1b. Per charter it
did not traverse the codebase and worked from the brief's summaries of the four
prior finding registers rather than re-deriving them.

### concurrency-specialist — round 1, VERDICT REVISE (18 concerns, 2 Critical)

| ID | Sev | Concern | Status |
|---|---|---|---|
| CONC-1 | **Critical** | `LocalComputeSlot` has no stated primitive and no defined release path across the thread/loop boundary. Reap is signalled by a **daemon thread that swallows the callback's exceptions** (`_services/runs.py:1852-1875`); an `ArtifactLockTimeout` from the 30 s spin strands the slot permanently, silently. Four release paths; the cancel path (F4) is only one | open · spec-change |
| CONC-2 | **Critical** | Restart reconciliation lets a live orphan **claim** the slot and never says who releases it. `rehydrate_from_sandbox` cannot rebuild `LocalRunHandle`s — the `Popen` belonged to the dead server — so no exit observer exists and the slot is held until the new server dies. Also `RunRecord` stores `pid` with no create-time, so PID reuse gives a false orphan or kills a live run | open · spec-change |
| CONC-3 | Major | `LocalRunner`'s process-wide `atexit` hook SIGTERMs every child, so **normal session end kills every local W2/W3 the agent launched**. §1.5 designs recovery for the SIGKILL case and is silent on the common one | open · spec-change |
| CONC-4 | Major | Artifact mutation is unguarded read-modify-write; two concurrent `pipeline_patch` calls lose one edit and **both return `ok:true`** | open · spec-change |
| CONC-5 | Major | §8.3's snapshot covers the read side only — two partial `campaign_start` calls both pass the status CAS and the second clobbers the first's `study_id`, orphaning a running arm | open · spec-change |
| CONC-6 | Major | `campaign_start` fan-out has no execution model and **cannot complete as specified under local routing**; nothing launches arm 4 when arm 1 finishes; partial failure undefined | open · spec-change |
| CONC-7 | Major | Store-open subprocess fan-out is unbounded — one per arm per poll, N subagents, no cap anywhere | open · spec-change |
| CONC-8 | Major | `RunRegistry` holds a `threading.Lock` **across** a 30 s interprocess spin; no §6.2 code for `ArtifactLockTimeout` or `allocate`'s bare `RuntimeError` | open · spec-change |
| CONC-9 | Major | `get_registry()` publishes `_REGISTRY` **before** `discover()` runs, unlocked — a loser thread can get a partially populated registry, silently short | open · spec-change |
| CONC-10 | Major | Subset staging is idempotent by key but not by completion — a reader can launch a fleet against a half-built symlink tree; `_load_images` takes whatever it finds | open · spec-change |
| CONC-11 | Major | The post-start CAS races the exit observer; an immediately-failing run can be **overwritten back to `running`**, permanently blocking `allocate` on that output dir | open · spec-change |
| CONC-12 | Major | One slot makes `W1` unavailable for the whole of any local W2/W3, and `probe_timeout_s` turns the wait into a guaranteed error — off-cluster, the entire §8.7 loop dies whenever deploy runs | open · spec-change · needs-user-input |
| CONC-13 | Minor | Probe-worker exclusivity is derived from the slot, never stated as an invariant; no liveness check before use | open |
| CONC-14 | Minor | Probe output path collides across probes (keyed by pipeline alone) | open |
| CONC-15 | Minor | Three blocking `W0` calls missing from F3's table: `workspace_lineage` reads, `workspace_cancel`, `workspace_info{refresh}` | open |
| CONC-16 | Advisory | `campaign_status{since}` cursor: absent-artifact is not a defined cursor state, and `study.db` never exists under journal storage | open |
| CONC-17 | Advisory | The handler concurrency model is never chosen, and CONC-1/7/8 all depend on it | open · spec-change |
| CONC-18 | Advisory | OME-Zarr's real coupling is **concurrency**: the sidecar works around a read-only-while-open constraint, so removing it changes the resume contract's shape | open |

**F4 answered:** partly settled and worse than stated — the cancel path is one of
four release paths and the *exit-thread* path has no design at all. The shared
probe worker is **refuted** as a hazard, but only because the slot admits one
probe at a time — a safety derived, never stated.

**Charter areas NOT reached:** `subset_generate` with a `W2` selector;
**elicitation under a shared connection** (whether A's `campaign_approve` prompt
can be answered by B — nobody has drawn the request-vs-connection scope line for
elicitation that F8/F10 draw for progress vs logging); P1/C7's B1 retry predicate
as code; `SandboxRoot` thread-safety; staged-GPU controller internals.

### data-flow-reviewer — round 1, VERDICT REVISE (16 concerns, 5 Critical)

| ID | Sev | Concern | Status |
|---|---|---|---|
| FLOW-1 | **Critical** | `pipeline_probe`'s measurement frame matches **neither engine**: `measure` defaults `apply_post=True`, every per-image engine passes `False`. A probe can condemn a working pipeline, and §8.7's post-slot loop optimizes a transform nothing scores | open · spec-change |
| FLOW-2 | **Critical** | `produces_columns` describes a **third frame no consumer sees** — not the scorer's input, not the mirror | open · spec-change |
| FLOW-3 | **Critical** | **The dataset has no home in the workspace.** §2.3's tree has no place for input images, every worked example uses `data/plates` as if inside, and `SandboxRoot.resolve()` rejects escapes. Either the root contains the data — colliding with §2.2's immutability rule, the `.git` warning and the CWD default — or the flagship flow cannot start | open · spec-change · **needs-user-input** |
| FLOW-4 | **Critical** | `flat/` staging keys symlinks by **bare filename** — the exact collision §10.2 introduced parent-relative paths to prevent. An image is lost silently while `n_images` still stands | open · spec-change |
| FLOW-5 | **Critical** | The deploy argument contract is stale against merged main and **the guard is inverted**: `--resume` no longer exists (continuation is derived), `mode` has four values not three, and `output_not_empty` refuses the case the CLI now handles while the real hazard — silently continuing a run — has no code | open · spec-change |
| FLOW-6 | Major | **Token binding answered:** `argv_digest` is undefined in the spec. If it covers §5.3's `argv` it is sound but narrow — it binds *what* runs, not the SBATCH directives or array width, which §5.2 derives live from `scontrol`/`sacctmgr`. So the cluster can silently re-chunk between plan and start with every check passing | open · spec-change |
| FLOW-7 | Major | Probe timing reaches the estimate with **no digest binding** — `pipeline_patch` mutates in place up to 12×, so a "measured" estimate may come from a different pipeline | open · spec-change |
| FLOW-8 | Major | `campaign_start` is not idempotent; a kill mid-fan-out is unrecoverable — re-calling hits `allocate`'s nonterminal rejection, and no code covers it | open · spec-change |
| FLOW-9 | Major | Assay and subset are **joined by nothing** — a campaign can be planned from an assay characterized on a different dataset with nothing failing | open · spec-change |
| FLOW-10 | Major | `deploy.start` lineage carries no subset or scope, so promotion's "measured" node-hours has no identifiable source | open · spec-change |
| FLOW-11 | Major | **`comparable` answered:** it is typed a bare boolean, so "unknown" is **unrepresentable** — a truncated journal forces asserting a comparison the artifacts cannot support, or suppressing a valid one. Needs tri-state + `basis` | open · spec-change |
| FLOW-12 | Minor | Subset digest recorded once, never re-verified before staging; no GC for staging, tokens or probes | open · spec-change |
| **FLOW-13** | Minor | **REFUTES the brief's OME-Zarr exposure.** Checked by grep: the MCP spec **never names the `.npy` sidecar** (one hit, unrelated); `deploy_status` reads `manifest.json`, which contains no HDF reference; P6 staging symlinks *inputs*, untouched by an output-store change. Real hits are four cosmetic prose lines | open |
| FLOW-14 | Major | The **real** OME-Zarr coupling is the mode list plus a migration guard: any dataset with `.h5` results and no store becomes a hard failure on every result-consuming mode, and the server has no equivalent of the GUI's report. Also flags `--durable-writes` reaching the already-promoted `argv.py`, and `requires-python <3.13` unchecked against `fastmcp` 3.x | open · spec-change |
| FLOW-15 | Minor | `deploy_plan`'s W0 cost confirmed; its preferred remedy (precomputed identity rows) is **exactly what OME-Zarr redesigns**, so defer with §5 | open · spec-change |
| FLOW-16 | Advisory | Not an MCP defect: the OME-Zarr design contradicts itself on whether migrate rewrites `deliverables/metadata.csv` — relay upstream | open |

**Flow coverage:** 4 of 5 traced to completion. **Not reached:** the probe-worker
subprocess lifecycle across a restart, and slot reconciliation against a live
orphan — which CONC-2 independently found to be Critical.

### general-reviewer — round 1, VERDICT REVISE (12 concerns, 1 Critical)

| ID | Sev | Concern | Status |
|---|---|---|---|
| GEN-1 | **Critical** | **The spec specifies the pre-ruling model for both human gates.** `grep -rni "elicit\|fastmcp\|annotation"` over all 11 docs returns **nothing**, while §8.2 and §10.5 still assert the server "cannot verify that a human approved anything". D1a, D5 and D6 live only in the plan — so Phase 2C, which is written from §8/§10, would be built to the superseded contract | open · spec-change |
| GEN-2 | Major | **`to_argv` cannot emit `deploy_*`'s own arguments** — no `--layer`, `--restart`, `--overwrite`, `--gpu-slurm`; `RunConsoleState` has no `mode`/`layer`/`restart` field; `--slurm k=v` lives in `gui/run_console/_slurm.py`, which P2 never promoted. This is the coverage gap *behind* FLOW-5: even with the values corrected, **no `_services` symbol can emit them** | open · spec-change · alias FLOW-5 (same site, disjoint content) |
| GEN-3 | Major | Three mechanisms the spec names live under `phenotypic.gui` (`from_pipeline_dag`+`validate` for §3.2/§6.6, `deploy_tune_run` for §4.3, `_slurm_argv_extension` for §5.4), none in P2's list — and **the "never import gui" rule is already false**: the purity allowlist sanctions two upward edges, so anything importing `RunRegistry` transitively imports `phenotypic.gui`. All three probe **clean of Dash**, so this is an architecture-rule problem, not a dependency one | open · spec-change · needs-user-input |
| GEN-4 | Major | OME-Zarr adds a **new mid-run refusal** — any dataset with `.h5` results and no store fails on every result-consuming mode — i.e. a fresh `sys.exit(1)` inside the subprocess `deploy_start` launches, the exact opaque-exit failure §5.4 pre-validates to prevent | open · spec-change |
| GEN-5 | Major | `campaign_status{since}` **under-reports local SQLite studies**: `_optuna_store.py:88-89` enables WAL, so trials land in `study.db-wal` while `study.db` sits stat-unchanged. The default local path is reported frozen while progressing | open · spec-change |
| GEN-6 | Major | `campaign_start`'s write-back can **silently revert a §10.4 amendment** — §2.6 CASes on `status`, which cannot detect an arm-set change, and amendments leave `status` unchanged on both sides | open · spec-change · aliases CONC-5 (same file, different mechanism) |
| GEN-7 | Major | `promotion_request` has **no work class at all** and escalates `W0`→`W1` at runtime: §10.6.1 always sweeps headers over every parent image and, on mismatch, re-probes 2 — acquiring the slot implicitly | open · spec-change |
| GEN-8 | Major | The killable store-open subprocess (§1.6's ninth new piece) has **no owning task**, and T18's private `_finalize.py` forces `python -c "from phenotypic.tune._tune_cli._finalize import …"` as a process boundary | open · spec-change |
| GEN-9 | Minor | `subset_generate` is specified synchronous but may report `cost_class()=="W2"` — a synchronous `W2` has no contract in this design | open · spec-change |
| GEN-10 | Minor | `subset_required`'s raw-path fallback is a concurrency-sensitive predicate evaluated per call, and carries no `hint` — contradicting §6.2's own rule | open · spec-change |
| GEN-11 | Minor | §1.4 still says the lazy `__init__`s are "deferred cleanup, not a prerequisite" — **Task 2.5 landed them**, and there is no DR row | open · spec-change |
| GEN-12 | Advisory | **Independently confirms FLOW-13** by grepping all 11 docs: three of the brief's five OME-Zarr exposures are false. Adds a real one — **`--mode migrate` inverts what `--input` means** (an output dir, no `--pipeline`), which would invert `SandboxRoot` resolution, subset scoping and the plan-token digest at once. **Narrows the sequencing call:** "2C waits for OME-Zarr" over-blocks — three of 2C's five tools are format-independent | open · spec-change |

**Charter areas NOT reached:** `phase-1a` task documents; the four finding
registers in full (worked from README summaries); **§6.5's test plan vs the tests
that exist**; §9's skills and installer; §9.3/§9.4 domain content.

---

## Round 1 merge — cross-reviewer dedup and convergence

**63 concerns, four `REVISE`.** Collapsing:

| Convergence | Reviewers | Weight |
|---|---|---|
| **The OME-Zarr exposure in the brief is wrong** | FLOW-13 + GEN-12, by independent methods (grep of all 11 docs, plus reading the OME-Zarr design) | **The orchestrator's brief was wrong.** The spec never names the `.npy` sidecar; `deploy_status` reads a format-agnostic manifest; P6 stages *inputs*. Real coupling = the mode list, a migration guard, and `--input` inversion |
| **The deploy surface is stale and unbuildable** | FLOW-5 (values wrong) + GEN-2 (no symbol can emit them) + SIMP-3 (drop `mode`/`layer` entirely) | Three lenses, one site. SIMP-3's cut would dissolve FLOW-5 *and* GEN-2 |
| **`campaign_start` is not safe to run** | CONC-5, CONC-6, GEN-6, FLOW-8 | Four independent findings on one tool: races the write-back, has no execution model, reverts amendments, is not idempotent |
| **The slot can be stranded** | CONC-1, CONC-2, CONC-11, CONC-12 | Two Criticals |
| **`W0` blocks** | audit F3 + CONC-15 + GEN-7 + FLOW-15 | F3's table is incomplete; GEN-7 adds a tool with *no class at all* |

**No true conflicts requiring the precedence table.** GEN-2 flags `CONFLICT with`
data-flow's stale-contract finding, but inspection shows the same site with
disjoint content — recorded as an alias, not a conflict.

---

## Round 1 rulings — permanent, no reviewer may re-raise absent new evidence

### USER-8 [settled-by-user (round 1)] — scope cut, 32 → 26 tools
Resolves SIMP-1, 2, 3, 13, 14 (partial), and dissolves FLOW-5 + GEN-2 by deletion.

**Cut (6):** `promotion_request`, `promotion_approve` (fold the decision content —
winner provenance, subset score, gap, coverage warnings, §10.6.1's header sweep —
into `deploy_plan {scope:"full"}`, which already carries `pending_human_ack`/
`ack_prompt`); `experiment_profile_put` (the triage skill writes the file);
`pipeline_diff`; `campaign_get` (→ `campaign_status {detail:"artifact"}`);
`catalog_measurements` (`produces_columns` answers the workflow question).
**Also cut:** `mode`, `layer`, `sample` from the deploy tools — v1 deploy is
always the full pipeline.

**Explicitly KEPT against the reviewer's proposal:** `experiment_profile_get` and
`workspace_lineage`. Rationale the reviewer did not weigh — dropping them assumes
the agent reads workspace files directly, which holds in Claude Code but breaks on
any host giving the agent MCP and nothing else, turning §1.7's addable HTTP
transport into a breaking change. **`workspace_lineage` is additionally the only
read path to the anti-repetition evidence** (USER-9).

### USER-9 [settled-by-user (round 1)] — anti-repetition, a gap none of the 63 concerns found
Nothing in the spec requires an agent to consult §8.7's `pipeline.step` trail
before proposing an edit, so a compacted agent re-tries what it already rejected
and sibling subagents each burn probe budget on the same dead end.
**Ruling:** when an edit matches one already recorded for that pipeline,
`pipeline_patch` returns the prior attempt's evidence and `decision` as an
**advisory issue** — never a refusal, so a deliberate retry stays possible. Uses
the journal scan the tool already performs for its step counter.

### USER-10 [settled-by-user (round 1)] — local slot policy (resolves CONC-12)
Keep one slot. §1.5 states plainly that a locally-routed `W2`/`W3` **suspends
interactive probing** for its duration. Nearly unreachable on the cluster, where
`W2`/`W3` route to SLURM; made explicit rather than left as a surprise off-cluster.

### USER-11 [settled-by-user (round 1)] — dataset location (resolves FLOW-3)
**The workspace root must contain the image data.** §2.2/§2.3 reconcile to it:
drop the CWD default, keep the `.git` warning. Rejected the `data_roots`
allowlist as a second containment concept.

### USER-12 [settled-by-user (round 1)] — rulings into the spec (resolves GEN-1)
Write D1a (`fastmcp` 3.x), D5 (tool annotations), D6 (elicitation) and the
cluster-boundary reviewer cadence into the spec **now**, so the panel and Phase 2C
stop reviewing a superseded contract.

### USER-13 [settled-by-user (round 1)] — the §9.3 artifact is renamed
`assay` → **`experiment_profile`**. `profiles/<dataset>.experiment.json`;
`experiment_profile_get`. "Assay" names a measurement *procedure* in biology; this
artifact describes what is imaged and how it was captured.

### USER-14 [settled-by-user (round 1)] — local concurrency cap
**Locally, run 1–2 arms at a time to avoid OOM.** `budget.max_concurrent_arms`
takes a lower effective cap when routed local. Bears on CONC-6, which found
`campaign_start`'s fan-out has no execution model under local routing.

### USER-15 [settled-by-user (round 1)] — multi-group experiments

The spec assumed homogeneity in three places: one profile per dataset (§9.3.7),
one `pipeline_id` per deploy (§5.4), one scorer per campaign (§8.2). A real
experiment may hold several **species × media groups** needing different
pipelines, different parameters, and **different expected counts** — which is the
scorer, not merely the pipeline.

**Rulings:**

1. **Grouping is by one or more metadata columns, supplied by the agent as a
   parameter** — not by directory layout. So `scan_directory_structure`'s
   one-subdirectory-level dataset rule does **not** cover this case, and the
   grouping key must be explicit rather than implicit in the tree.
2. **Strategy: try a single pipeline across the whole experiment first, and
   descend to per-group only where evidence requires it.** This mirrors §9.4's
   prefab-first discipline — start general, specialize on evidence — and it is
   *judgment*, so per §9.1 it belongs in a skill. The *mechanism* it needs must
   exist in the server.
3. **`experiment_profile` keeps its name.** The experiment is the container; the
   profile therefore describes the experiment **and must carry per-group trait
   overrides**, since morphology, contrast and medium opacity can differ by group
   while `plate.format` and `imaging.modality` usually do not.

**Design implications — to be worked in round 2, not settled here:**

- **§9.3 profile gains `group_by: list[str]` and a `groups` map** of per-group
  trait overrides over the experiment-wide `traits`. The existing envelope
  already round-trips unknown keys and treats every trait as individually
  optional, so this extends rather than breaks it.
- **Subset selectors gain a group filter.** `MetadataGroupSubsetSelector` already
  *stratifies* across metadata groups; selecting *only* a group is the same join
  with a predicate. A per-group campaign then falls out of a per-group subset
  with no campaign change — §8.2's one-scorer invariant holds *within* a group.
- **Per-group deploy comes free from subset staging.** §10.3.1 already
  materializes a subset as a directory tree, so a group-scoped subset stages only
  that group's images and the existing single-`--pipeline` CLI needs no change.
  What needs deciding is what `scope:"full"` means for a group-scoped subset —
  the group's images, or the whole parent.
- **The escalation signal is missing.** "Descend to per-group if needed" requires
  the agent to *detect* need. A winner that scores well overall while failing one
  group is invisible on a single aggregate cost. **`campaign_status` should report
  a per-group cost breakdown when the subset is group-aware** — that is the
  evidence the escalation decision rests on, and without it the strategy is
  unactionable.

**Open for round 2:** whether `group_by` lives on the profile, the subset, or
both; the `scope:"full"` semantics above; and whether the per-group breakdown is
a `campaign_status` field or a scorer responsibility.

---

## Round 1 → 2 status: what the rulings actually resolved

**Applied** (verify these, do not re-raise): SIMP-1 (promotion folded into
`deploy_plan {scope:"full"}`), SIMP-3 (`mode`/`layer`/`sample` gone), CONC-12
(§1.5 states local batch suspends probing), FLOW-3 (§2.3 root mandatory and must
contain the data), GEN-1 (D1a/D5/D6 written into §1.4/§3.0/§8.2/§10.5), USER-9
(`edit_previously_tried` advisory), USER-13 (rename), USER-15 (§9.3.0.2
multi-group).

**Partially applied:** SIMP-2 (`experiment_profile_put` cut, `_get` **kept** —
see USER-8's rationale), SIMP-13 (`pipeline_diff` and `campaign_get` cut;
`get{raw}` and `save_overlay` **not** cut), SIMP-14 (`catalog_measurements` cut;
Task 10c **undecided**), FLOW-5 (`mode` corrected; the continuation semantics and
`output_not_empty` inversion **not** addressed).

**Still open — the bulk.** Both concurrency Criticals (CONC-1 slot release across
the thread boundary, CONC-2 orphan claims the slot with nothing to release it);
FLOW-1/2 (probe measures a frame no engine uses); FLOW-4 (`flat/` staging
collision); FLOW-6..16; GEN-2..12; SIMP-4..12, 15..17.

**Three questions deliberately left for round 2** (from USER-15): whether
`group_by` lives on the profile, the subset, or both; what `scope:"full"` means
for a group-scoped subset; and whether the per-group cost breakdown belongs to
`campaign_status` or to the scorer.

### USER-16 [settled-by-user (round 2)] — deferral criterion

A concern may be dispositioned **`deferred-to-2A`** — carried as a named Phase 2A
acceptance test rather than resolved in the spec — **only when its resolution
depends on observing behaviour that does not exist yet.** The test must be
written down with its pass condition; a deferral without one is an omission
wearing a schedule.

**Qualifies** (no server, no `fastmcp` dependency, nothing to observe):

- Does `fastmcp` 3.x deliver `CancelledError` into a handler on host
  cancellation? (CONC-1's fourth release path, audit F4)
- Is the host's tool-call timeout above or below `probe_timeout_s = 300`?
- Does an elicitation raised from a **subagent's** call surface to the human
  under §1.3's shared connection — and to whom is it attributed when two are in
  flight? (D6/USER-5, CONC round 1)
- Can `fastmcp` set `isError` without discarding the response body? Does its
  `Context` expose `elicit` and `report_progress`? (D1a)
- What does `tools/list` actually cost per turn against §1.6.1's budget? (F5)

**Does NOT qualify — these are design decisions, specifiable today:**

- **CONC-1's release symmetry.** Naming the slot's primitive, requiring release
  in `finally` at the innermost layer, and making the exit callback
  release-first/record-second are decisions, not measurements. *Which* of the
  four paths fires is observable later; *that all four must release* is not.
- **CONC-2's orphan watcher.** Whether reconciliation installs a watcher or
  refuses to claim the slot is a design choice. Only PID-reuse frequency is
  empirical, and the fix (record `(pid, create_time)`) does not depend on it.
- **CONC-17's handler concurrency model.** Choosing `async def` + one named
  executor is a decision three other findings depend on; deferring it leaves
  them unresolvable rather than pending.

**The test:** if the concern would still need a decision *after* the experiment
returned either result, it is design work and does not qualify.

---

## Round 2 — concurrency specialist, VERDICT REVISE (10 concerns, 1 Critical)

**Three are defects in the orchestrator's own round-1 edits.** Marked ✱.

| ID | Sev | Concern | Status |
|---|---|---|---|
| ✱ **CONC-19** | Major | **The "1–2 local arms" cap is unachievable.** Two arms cannot be concurrent under a semaphore of capacity 1 — the routing table two rows above gives a local `W2` the slot for its entire subprocess lifetime. So the effective cap is 1, always, and the OOM rationale describes a parallelism the same section forbids: *two uncoordinated mechanisms for one invariant*, the exact defect §1.5 opens by criticising. Worse, arm 2's wait is unbounded — if `campaign_start` awaits the slot the call blocks for hours against a host timeout, and an abandoned coroutine still holds a reservation, creating an orphan with no crash involved. Contradicts USER-1's "submit-and-poll" | open · spec-change · **needs-user-input** |
| ✱ **CONC-22** | **Critical** | **`deploy_plan {scope:"full"}` is declared `W0` while doing four things in one handler:** the whole-parent header sweep, a possible slot-acquiring `W1` re-probe, the token mint, and an unbounded human wait. §1.6.1 defines `W0` as under one second and never blocking. **And the ack/token state is self-contradictory** — the example returns `plan_token` *and* `pending_human_ack:true` together, while the scope table requires a token "minted with the human ack recorded". The agent holds the token before the human answers, so the ack is a second mutable field racing `deploy_start`, and §2.6 has no row for it (its token row still says "promotion tokens", a concept this round deleted) | open · spec-change · **needs-user-input** |
| ✱ **CONC-25** | Major | **`edit_previously_tried` is not free, and is racy in the case it was added for.** §2.5 offloads lineage *writes* because the lock spins to 30 s; reads are not offloaded (CONC-15). USER-9 makes that read unconditional on the most frequently called mutating tool. And §8.7 records a step with its keep/revert decision **only after the probe completes** — so a sibling mid-probe on the identical edit has written nothing, and both burn the budget. The compacted-single-agent case works; **the sibling case, which USER-9 cited as its rationale, does not** | open · spec-change |
| CONC-20 | Major | The suspension bargain is voided by CONC-3: probing is suspended for hours by design, then the local run is SIGTERMed at session end by `LocalRunner`'s `atexit` hook | open · spec-change |
| CONC-21 | Major | "Unverified" is right for *whether elicitation delivers* and wrong for **attribution, single-flight, and non-answer states** — no live test fixes a prompt that names no artifact, or a timeout that is not an approval. Three rules the spec owns regardless of the test | open · spec-change · needs-user-input |
| CONC-23 | Major | Multi-group multiplies two uncapped fan-outs: `max_concurrent_arms` is **per-campaign**, so N groups × M arms has no aggregate ceiling. `species × medium` is a cross-product — the spec's own example implies six campaigns | open · spec-change |
| CONC-24 | Major | The per-group breakdown **defeats the polling economy built for the same tool** — `since`'s value is skipping the store open, and a breakdown cannot come from a stat. Also has no stated data source: `QCScorer` returns one scalar per trial. **Answers USER-15's deferred question: the breakdown belongs to the scorer**, written into trial user attrs at scoring time | open · spec-change · needs-user-input |
| CONC-26 | Minor | `campaign_status {detail:"artifact"}` is the session-recovery entry point and reads an artifact `campaign_start` may be mid-clobber of; nothing distinguishes "not started" from "started by a server that died" | open · spec-change |
| CONC-27 | Major | The new annotation rule is a **list, not a rule**, and `deploy_plan` breaks it: named like a read, it now mints a token, sweeps every parent image, can acquire the slot, and prompts a human — yet a host told `readOnlyHint` may auto-approve it, putting the sole full-dataset human gate behind a tool declared safe to call unasked. Same for `pipeline_probe`, which mutates nothing but holds the exclusive slot | open · spec-change |
| CONC-28 | Major | Elicitation widens `campaign_approve`'s CAS window from milliseconds to **minutes**, and GEN-6 already showed the `status` CAS cannot detect an arm-set change — so the human approves a summary that can go stale while they read it | open · spec-change · depends GEN-6 |

**Its sequencing advice:** CONC-19, 20, 22 and 25 all resolve differently
depending on how **CONC-1**'s primitive and release path are settled, so CONC-1
and CONC-2 should be taken first. Both remain open from round 1.

---

## User rulings, round 2 (permanent — no reviewer may re-litigate absent new evidence)

### USER-17 — the local arm cap is the slot's capacity (settles CONC-19)
**There is one mechanism, not two.** The "1–2 local arms" prose in §1.5 is
deleted. The `LocalComputeSlot` is the sole owner of the local-OOM invariant,
and its **capacity is a configuration knob defaulting to 1**. A large node may
set it to 2; the user's "1–2 arms" is therefore a *supported configuration
range*, never a concurrency promise the spec makes on its own.

**Corollary — no unbounded wait.** A second local arm arriving at a full slot is
**told the slot is busy and returns**; it does not block awaiting it. This is
what keeps the local path consistent with USER-1's submit-and-poll and removes
the abandoned-coroutine orphan CONC-19 describes.

### USER-18 — the human gate lives in `deploy_start` (settles CONC-22, Critical)
`deploy_plan` returns to being a **genuine read-only preview**: fast, mints
nothing, waits on no human, and is honestly `W0`. The elicitation fires in
`deploy_start`, **which is where §10.5's own words already put "the point of
spend"** — this makes the spec consistent with the rationale already written for
it. The `plan_token` / `pending_human_ack` contradiction dissolves because there
is only one state to carry, not two. §2.6 needs no ack row.

The round-1 fold itself **stands** — no 27th tool. Only the gate moves.

### USER-19 — the per-group cost breakdown is the scorer's output (settles CONC-24, closes USER-15's deferral)
The scorer records per-group figures as **Optuna trial user attributes at
scoring time**. `campaign_status` reads them; it never recomputes. This is what
preserves the `since` polling economy — a recomputed breakdown would defeat the
one thing `since` exists to buy.

### USER-20 — handlers are async; blocking work is offloaded (settles CONC-17; unblocks CONC-1, 2, 19, 20, 25)
Tool handlers are `async def`. **Anything that blocks goes to a worker thread** —
subprocess waits, the lineage reads whose lock spins to 30 s, and SLURM polling.
This is what `fastmcp` expects, it is what keeps the server answering an
interactive probe during a long run, and **it is the only model under which
§1.6.1's NFR table is satisfiable at all.**

This was the decision the concurrency specialist identified as the one four of
its findings hang from; it is now made, and those findings resolve against it
rather than around it.

---

## Round 2 dispositions (after USER-17..20)

| ID | Status | What changed |
|---|---|---|
| CONC-17 | **settled-by-user** (USER-20) | §1.5 now states the handler model once — async handlers, all blocking work offloaded — and the spec relies on it rather than re-deriving per tool |
| CONC-19 | **settled-by-user** (USER-17) | §1.5's "1–2 arms" prose deleted. The slot is the sole owner; capacity is `local_slot_capacity` (default 1). A second arm is refused, not parked — no unbounded wait, no orphan reservation |
| CONC-20 | **open** | USER-20 removes the event-loop half. The `atexit`-kills-the-local-run half is CONC-3's and is still open |
| CONC-21 | **open · needs-user-input** | Attribution / single-flight / non-answer are design decisions, not observations — they do **not** qualify for `deferred-to-2A` under USER-16. To be put to the user with the round-2 panel |
| CONC-22 | **settled-by-user** (USER-18) — *was Critical* | Elicitation moved to `deploy_start`; `ack_prompt` is text to show, `pending_human_ack` deleted from that response, §2.6 needs no row. Fold stands. Additionally §5.3's flat `W0` corrected to `W0` at subset / `W1` at full |
| CONC-23 | **open** | Aggregate ceiling across per-group campaigns still unspecified. Candidate fix drafted; not yet applied pending the panel |
| CONC-24 | **settled-by-user** (USER-19) | §9.3 now states the scorer writes per-group figures to trial user attrs at scoring time; `campaign_status` reads and never recomputes. Closes USER-15's deferral |
| CONC-25 | **resolved** (round 2) | Read offloaded by USER-20. The race fixed at its source: §8.7 journals a step as `in_flight` on acceptance rather than writing the whole row after the probe. Residual window stated, not hidden |
| CONC-26 | **open** | Artifact-read-vs-clobber and "died" vs "never started" still unaddressed |
| CONC-27 | **resolved** (round 2) | §3.0's enumeration replaced with a derivation keyed on cost as well as mutation; `pipeline_probe` and `deploy_plan` correctly fall outside `readOnlyHint` |
| CONC-28 | **open** · depends GEN-6 | Elicitation widens `campaign_approve`'s CAS window. Held until the general reviewer confirms or closes GEN-6 this round |

**Deferred-to-2A under USER-16** (resolution genuinely depends on unobservable
behaviour; each carries its pass condition): does `fastmcp` deliver
`CancelledError` on client disconnect; host timeout vs `probe_timeout_s`; whether
subagent elicitation surfaces and how it is attributed; whether `Context` exposes
`elicit`/`report_progress`; `tools/list` cost at 26 tools.
