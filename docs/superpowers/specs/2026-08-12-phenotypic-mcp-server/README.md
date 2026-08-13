# PhenoTypic MCP Server — Design Spec

**Status:** draft. §1–§3 and §7 reviewed once by an independent reviewer and
revised; §4–§6 and §8 pending review.
**Date:** 2026-08-12

## What this is

A design for an MCP server that lets an LLM agent build `ImagePipeline`
configurations, tune them with `phenotypic.tune`, and deploy them over datasets
— locally or on SLURM.

The intended UX is **collaborative planning, then delegated execution**: you and
the agent characterize the assay, decide upfront what is worth trying, write that
agreement down as a *campaign*, and the agent then executes it across parallel
subagents without you in the loop. §8 describes the flow and §9 the division of
labour; read those two first if you want the shape before the mechanics.

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
| 9 | [09-responsibilities-and-skills.md](09-responsibilities-and-skills.md) | Server-vs-skill boundary, assay triage, prefab-first construction, the four bundled skills |

## Executable evidence

`docs/superpowers/logic_validation_scripts/2026-08-12-phenotypic-mcp-server/optuna_journal_storage.py`

Re-derives the claims behind §7 P1 from Optuna directly, never importing
`phenotypic`. **Read its `DISCRIMINATION` verdict, not just the ok lines** — on a
local filesystem the negative control also passes, meaning C2a there measures OS
`O_APPEND` atomicity rather than the lock. `--require-discrimination` is the gate
that must pass on the target cluster mount before P1 is implemented.

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
  `gui/`. The server is a thin adapter plus four genuinely new pieces.
- **Two hard refusals:** no `--overwrite` (it is `shutil.rmtree`), and no raw
  sbatch passthrough (`parse_slurm_args` constrains neither keys nor values).

## Open questions

| OQ | Section | Question |
|---|---|---|
| 2.1 | §2.7 | Should authored tune specs also be written to the GUI's preset dir so they appear in its Load dropdown? |
| 3.2 | §3.5 | Should `pipeline_probe` render overlay PNGs the agent cannot see? |
| 4.1 | §4.7 | Expose screening at all, given §7 P4 shows `--screen` + `--slurm` is a silent no-op today? |
| 4.3 | §4.7 | Should `tune_export_best` also write `trials.parquet` for distributed studies, since no `--recompile` exists? |
| 6.1 | §6.6 | Surface all GUI DAG advisory kinds on every `pipeline_put`, or a curated subset? |
| 8.1 | §8.6 | Should a campaign be able to carry a deploy arm, launching a full run past the human checkpoint? |
| 8.2 | §8.6 | Can the agent amend a campaign mid-flight, or does an amendment need re-approval? |
| 9.4 | §9.3.2 | The contrast bands (η ≥ 0.75 / 0.45–0.75 / < 0.45) are provisional cut points on a principled measure. They need calibrating against real plates before they mean anything. |

**Resolved:** topology (stdio on the login node) · parallelism (agent-side
fan-out) · state (on-disk workspace, `RunRegistry` reused rather than a new
index) · SLURM authority (named profiles + capped overrides) · coupling
(`_services` promotion) · catalog breadth (reconcile both enumeration lists) ·
workspace root (`--workspace`, defaulting to CWD) · defaulting (explicit always)
· deploy gate (plan-then-submit mandatory) · distributed storage (JournalStorage
backend, gated on L1) · skill packaging (in-repo + `phenotypic-mcp setup`) ·
assay scope (per-dataset) · assay validation (structure and provenance only —
never biology).
