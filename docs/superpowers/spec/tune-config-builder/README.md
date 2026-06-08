# Tune Config Builder — Design Spec

**Status:** Design complete and review-passed (plan-reviewer, 2026-06-08;
resolutions folded in as D13). One planning-time spike remains before/early in
implementation: confirm `_param_forms.param_form` can render non-`ImageOperation`
scorer pydantic models (else add a scorer registry). Otherwise ready for an
implementation plan.
**Date:** 2026-06-08
**Branch:** `worktree-tune-config-builder`
**Interactive mockup:** `mockups/tune-config-builder.html` (the canonical visual reference for everything below).

## Problem

The GUI mounts a read-only `/tune/` co-pilot that can *analyze* a finished
tuning run but cannot **author** a `TuningSpec` or **deploy** a run. Users must
hand-write `spec.json.pht-tune` and launch from the CLI. This project adds full
spec authoring + deploy (local + SLURM) by evolving the existing `/tune/`
surface — not a new top-level page.

## The shape in one paragraph

The `/tune/` Dash app gains a **hamburger menu** with three destinations —
**Setup / Run / Monitor** (author → deploy → inspect). **Setup** ("what to tune
& how to judge") is a progressive-disclosure form: pipeline + an inference-prefilled,
editable **search space** + **scorer**. **Run** ("how to run it") holds strategy,
budget, advanced eval, and the compute target, then deploys via the run-console's
process/SLURM engine. **Monitor** is the existing read-only co-pilot plus a run
switcher, a Local-vs-SLURM live view, run cancellation, and an **Export best
pipeline** action that closes the loop (best params → a runnable pipeline config).

## Document set

| Doc | Covers |
|-----|--------|
| [00-decisions-and-open-questions.md](00-decisions-and-open-questions.md) | The decision log (D1–D12) and the closed open-questions list. Read this first. |
| [01-placement-and-ia.md](01-placement-and-ia.md) | Placement rationale, the Setup/Run/Monitor hamburger IA, navigation, empty-state gating, ledger obligations, files to touch. |
| [02-search-space-and-scorer.md](02-search-space-and-scorer.md) | The Setup surface: inference prefill, the per-knob domain editor, the `FloatRange.step` extension, validation, scale affordances, scorer. |
| [03-run-deploy-and-monitor.md](03-run-deploy-and-monitor.md) | The Run surface (strategy/budget/compute/deploy), the shared runner extraction, SLURM divergence, cancellation, and the Monitor export-best loop. |
| [04-data-model-and-naming.md](04-data-model-and-naming.md) | `TuningSpec` serialization, the `phenotypic_version` stamp, the typed `.pht-*` suffixes, the save/load library, and the formal validation contract. |

## Required non-GUI (backend) changes

Most of this is GUI work, but two engine/model changes are prerequisites and are
specified in doc 02/04:

1. **`FloatRange.step`** — add an optional step to `FloatRange` (quniform), wire
   it into grid/random/Optuna, and add the step↔log guard. Unlocks quantized
   floats *and* makes a stepped float grid-enumerable.
2. **`TuningSpec.phenotypic_version`** — add a top-level provenance field to the
   spec so loads can warn on version mismatch.

## Non-goals (v1)

- Wizard overlay for first-timers (deferred; progressive form ships first).
- Conditional/nested knobs in the UI (engine supports `conditional_on`, but
  inference never populates it in v1).
- Optimizer-level relational constraints between knobs (no Optuna
  `constraints_func` exists; the GUI validates `min < max` client-side only).
- New config encodings, in-place file renames, or changes to JSON payload schemas.
