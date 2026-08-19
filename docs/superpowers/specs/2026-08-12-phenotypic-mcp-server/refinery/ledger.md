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
