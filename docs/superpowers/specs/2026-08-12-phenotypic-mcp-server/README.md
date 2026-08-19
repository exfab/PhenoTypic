# PhenoTypic MCP Server — Design Spec

**Status:** draft, **refinery rounds 1–2 applied, round 3 propagation applied**.
Ten sections, plus an MCPB/deployment evaluation, an interface audit against
official MCP guidance, and a rotating four-reviewer refinery panel (general,
data-flow, simplicity, concurrency). **Twenty-seven user rulings are recorded in
`refinery/ledger.md` with their rationale, and they are permanent** — a reviewer
may not re-raise one absent new evidence.

**What round 1 changed:** the catalog went **32 tools → 26** — promotion folded
into `deploy_plan {scope:"full"}`, the `assay` artifact became
`experiment_profile`, and `mode`/`layer`/`sample` left the deploy tools. §8.2's
concession that the server "cannot verify that a human approved anything" was
retired in favour of **elicitation**. And §3.2 gained the advisory that stops an
agent repeating an edit it already rejected — a gap none of the 63 concerns
found.

**What round 2 changed:** the **human gate moved to `deploy_start`**, the point
of spend (USER-18), with `human_response` required for a `plan` token — the
unattended campaign arm carries consent forward instead (USER-22, §5.4) — and
`ack_source` carrying the elicited-vs-asserted distinction in the response
(USER-22). Handlers became `async` with two named executors (USER-20); the local
slot became a configurable capacity that is the *sole* owner of the local-OOM
invariant (USER-17); campaign fan-out became a background task with a launcher
lease. And **multi-group experiments were offloaded to the agent** (USER-24):
`group_by` on the profile, per-group trait overrides, and the per-group cost
breakdown were all **deleted**, because one subset per group already gives one
campaign per group whose aggregate cost *is* the group's cost. The server keeps
exactly two things from that design — `group_filter` on the `SubsetSelector` ABC
(§10.3), and `derived_from` on the campaign artifact (§8.2).

**What round 3 changed:** no new decisions — a propagation sweep that carried
eleven rulings from the sections that argue them into the argument tables,
artifact schemas, error rows and plan decision records they had never reached.
`refinery/defining-sections-map.md` is the standing check that stops that
recurring.
**Date:** 2026-08-12 (last propagated 2026-08-19)

## What this is

A design for an MCP server that lets an LLM agent build `ImagePipeline`
configurations, tune them with `phenotypic.tune`, and deploy them over datasets
— locally or on SLURM.

The intended UX is **collaborative planning, then delegated execution, on a
subset**: you and the agent characterize the assay and pick a development
subset, decide what is worth trying, write that agreement down as a *campaign*,
and the agent executes it across parallel subagents without you in the loop —
**bounded to the subset**. The full dataset is touched once, after a separate
human promotion. §8 describes the flow, §9 the division of labour, §10 the
subset and the promotion gate; read those three first if you want the shape
before the mechanics.

Two gates, asking different questions: **campaign approval** ("is this a
sensible experiment?") before subset compute, and **promotion** ("is this winner
worth the full dataset?") before the expensive irreversible step.

**Mechanism and judgment are deliberately separated.** §1–§8 specify what the
server does and refuses. §9 specifies what the *agent* should know — how to
triage an organism's traits, why prefab pipelines come before custom ones, how to
read a leaderboard — and ships that as bundled skills. The rule dividing them:
the server makes wrong things impossible; the skills make right things likely.

## Sections

| § | File | Covers |
|---|---|---|
| 1 | [01-architecture.md](01-architecture.md) | Process model, layering, `_services` promotion, work-class routing, `LocalComputeSlot` |
| 2 | [02-state-and-identity.md](02-state-and-identity.md) | Disk-as-authority, path identity and its limits, workspace tree, `RunRegistry` reuse, lineage |
| 3 | [03-tool-catalog.md](03-tool-catalog.md) | Catalog, pipeline, and workspace tools; the probe worker |
| 4 | [04-tune-integration.md](04-tune-integration.md) | Structured knob targets, spec authoring, launch, polling, best-pipeline export |
| 5 | [05-deploy-and-slurm.md](05-deploy-and-slurm.md) | SLURM profiles and caps, plan-then-submit, deploy, status, cancellation |
| 6 | [06-errors-limits-testing.md](06-errors-limits-testing.md) | Error taxonomy, limits, safety boundary, test plan |
| 7 | [07-prerequisites.md](07-prerequisites.md) | P1 JournalStorage backend, P2 promotion, P3 catalog+descriptor, P4 `--screen` guard, rollout |
| 8 | [08-workflow-and-campaigns.md](08-workflow-and-campaigns.md) | The phased UX and the campaign artifact |
| 9 | [09-responsibilities-and-skills.md](09-responsibilities-and-skills.md) | Server-vs-skill boundary, experiment-profile triage, prefab-first construction, the four bundled skills |
| 10 | [10-subsets-and-promotion.md](10-subsets-and-promotion.md) | The development subset as the unit of work, the `SubsetSelector` hierarchy, and the promotion gate before full-dataset compute |

## Executable evidence

`docs/superpowers/logic_validation_scripts/2026-08-12-phenotypic-mcp-server/optuna_journal_storage.py`

Two scripts, both re-deriving their claims from the dependency or the data
directly rather than from `phenotypic`.

**`optuna_journal_storage.py`** — the claims behind §7 P1. **Read its `DISCRIMINATION` verdict, not just the ok lines** — on a
local filesystem the negative control also passes, meaning C2a there measures OS
`O_APPEND` atomicity rather than the lock. `--require-discrimination` is the gate
that must pass on the target cluster mount before P1 is implemented.

It also measures throughput headroom (C6), which is what retired the claim that
Postgres remains right for large fleets.

**`contrast_trait_measure.py`** — the choice of contrast measure for §9.3.2.
Establishes that Otsu's η and Cohen's d are scale-invariant (unchanged across a
20× contrast reduction), that Michelson tracks contrast linearly, that η's
per-cell span on real plates is 1.8% of its nominal range, and that whole-frame
Otsu splits plate-from-surround rather than colony-from-agar. It killed a
measure that would otherwise have shipped looking principled.

## Design commitments worth knowing before reading

- **The codebase already anticipates this server** in four places
  (`abc_/_base_operation.py:192`, `sdk_/_docstring_params.py:7`,
  `tune/_search_space/_discovery.py:4`, `tune/_spec.py:293`). Those sites fix
  parts of the contract — notably that the agent **selects a structured tuning
  target, never authors a string key**.
- **One shared stdio server per session.** Subagents inherit the parent's MCP
  connection; they do not get their own process. Hence one `LocalComputeSlot`.
- **Disk is the authority.** The server holds no state whose loss matters.
- **Roughly 80% of the substrate exists**, mostly as a Dash-free tier under
  `gui/`. The server is a thin adapter plus the genuinely new pieces enumerated
  in **§1.6's table** — deliberately not re-counted here, because the number
  drifted every time it was restated (§1.6). They include
  descriptor projection + column derivation, profile governance,
  routing + the compute slot, the `_space.py` pure/view split, a pure sbatch-spec
  extraction, subset staging, the token store, the probe worker, and the killable
  store-open subprocess. All but the first three surfaced only under review; the
  count rose repeatedly; read it off §1.6's table rather than from any sentence.
- **Two hard refusals:** no `--overwrite` (it is `shutil.rmtree`), and no raw
  sbatch passthrough (`parse_slurm_args` constrains neither keys nor values).
- **Development happens on a subset.** The full dataset is touched once, behind
  a promotion gate separate from campaign approval (§10). Subset-scoped tools
  take a `subset_id`, not a path, so the boundary is enforced rather than
  merely asserted.

## Open questions

**None.** The last one closed on 2026-08-19 by **USER-26**. For the record, it
asked how `parent ∩ group_filter` — USER-21's resolution of `scope:"full"` on a
group-filtered subset — reaches an engine that accepts no file list: through
§10.3.1's staging, or restricted to filters expressible as whole `--input`
subtrees? Neither. The intersection is resolved to a **manifest** at plan time
and bound by the token's digest, so nothing is copied and the approved image set
cannot drift. It carries a new §7 prerequisite: the public `--input` is a single
`click.Path`, so a top-level manifest flag is new work.

Every other question raised during design or by the independent review
passes has been resolved and recorded in the relevant section's "Resolved since
first draft" block.

The last two were closed by measurement rather than by decision:

- **OQ-9.4 (contrast bands)** did not need calibrating — it needed *refuting*.
  The proposed measure, Otsu's η, turns out to be **invariant to contrast**: a
  20× reduction left it numerically unchanged. Replaced with per-cell Michelson,
  which tracks contrast linearly; the categorical band stays human-sourced until
  a dataset spanning low contrast exists. Evidence:
  `logic_validation_scripts/.../contrast_trait_measure.py`.
- **OQ-10.1 (promotion re-probe)** resolved to a header sweep always, re-probe
  only on mismatch — what breaks the timing extrapolation is readable from image
  headers without decoding.

**Resolved:** topology (stdio on the login node) · parallelism (agent-side
fan-out) · state (on-disk workspace, `RunRegistry` reused rather than a new
index) · SLURM authority (named profiles + capped overrides) · coupling
(`_services` promotion) · catalog breadth (reconcile both enumeration lists) ·
workspace root (`--workspace`, **mandatory, and must contain the image data**) ·
defaulting (explicit always)
· deploy gate (plan-then-submit mandatory) · distributed storage (JournalStorage
backend, gated on L1) · skill packaging (in-repo + `phenotypic-mcp setup`) ·
profile scope (per-dataset) · profile validation (structure and provenance only —
never biology).
