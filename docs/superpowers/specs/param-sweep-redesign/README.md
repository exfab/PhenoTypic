# Parameter Sweep Redesign — Spec Bundle

This folder holds the design for refactoring the `phenotypic.sweep` parameter
sweep CLI into a **robust, sample-efficient parameter tuning engine** (screen →
optimize → score), eventually exposable as an MCP for agentic parameter tuning.

The **master design** is the source of truth for architecture, decisions, and
phasing. Each **companion doc** goes deep on one aspect that the master only
summarises — so the master stays readable while the details live where they can
grow.

> **Resuming this work?** Start with [RESUME.md](RESUME.md) — status, the remaining
> stub docs to write, the review-team recipe, and global carry-forward caveats.

## Master design

- [2026-06-01 — Parameter tuning engine design](2026-06-01-parameter-tuning-engine-design.md)
  — the reframe (ask-and-tell loop), the five components + screening phase, the
  pluggable `Scorer`, the `SearchStrategy` Protocol (Optuna default), calibration-set
  robust evaluation, `infer_search_space`, the CLI/MCP/Dash drivers over one shared
  study, back-compat, dependencies, file layout, testing, and the 6-phase rollout.

## Companion docs

Deep dives on individual aspects. Each maps to a section/component of the master
design; the master links out to these rather than inlining the detail.

| Doc | Covers | Maps to | Status |
|-----|--------|---------|--------|
| [`reference-free-segmentation-metrics.md`](reference-free-segmentation-metrics.md) | Non-ground-truth ("unsupervised") segmentation-quality metrics for the `ReferenceFreeScorer` — catalogue (~36 metrics, 5 families), math, validity for arrayed-colony phenotyping, which to combine and how, **plus the mandatory meta-validation gate** (correlate the proxy against a small GT set; abstain if weak). 68 peer-reviewed/preprint sources, citation-audited. | master §4 (`Scorer`), §2 | **✅ written** |
| [`supervised-scorers.md`](supervised-scorers.md) | Ground-truth scoring (~24 metrics, 5 families): region/overlap (Dice/IoU/Tversky), boundary (HD95/NSD), instance/detection (PQ/AJI/SEG/AP), counting/localization (MAE/CCC/FROC/grid-cell F1), partition + selection (ARI/VI, Metrics Reloaded). Modality-tiered composite + the matching-free ARI/VI guard for touching colonies; doubles as the meta-validation gate's reference signal. 81 sources, citation-audited. | master §4 (`Scorer`), D1 | **✅ written** |
| [`qc-objective-mapping.md`](qc-objective-mapping.md) | The `QCScorer`: maps the existing `analysis/` `QualityCheck`s into a no-GT objective. Two-phase Scorer (per-image Count + batch reliability panel), threshold-anchored smooth normalizer + special-value policy, coverage-weighted trimmed-mean reduction, hybrid geometric fusion, four-part anti-gaming guard, graceful degradation. | master §4 (`Scorer`), D1 | **✅ written** |
| [`search-space-inference.md`](search-space-inference.md) | `infer_search_space` → a reviewable `InferredSearchSpace` proposal: two-tier resolution (`TuneSpec` marker, hint-only + the `⊆` invariant → type/constraint heuristics), the unbounded-window heuristic + `d≤0` surfacing, `_tune_optional` flat `__enabled__` presence wrapping, one-level nested-op recursion (path keys + apply-time class-validation), per-knob provenance + the autonomy gate. Agent-reviewed against source. | master §5, D7 | **✅ written** |
| [`operation-tuning-annotations.md`](operation-tuning-annotations.md) | Field-migration workstream: annotate operation fields with `Field(ge=,le=)` (validity) + `TuneSpec` (search) so inference reads real envelopes instead of guessing. Validity-vs-search distinction, back-compat-safe `TuneSpec`-first/`Field`-second staging, coverage + `⊆` guardrail tests. Decoupled from tune Phase 1; the dial toward D6 MCP-autonomy. | master §5, D6; supports `search-space-inference.md` | **✅ written** |
| `screening-importance.md` | fANOVA importance over optimizer trials (main effects + interactions), freezing thresholds, zero-dep fallback, the importance report | master §4 (`ScreeningPhase`), §3 D8 | planned |
| [`robust-evaluation.md`](robust-evaluation.md) | The `Evaluator`: uniform 3-step loop (`score_image` → robust-aggregate per term → `finalize`), per-term `median − λ·IQR` stability penalty (λ=0.5), Scorer-owned normalization (Böck-safe), group-aware CV (replicate-safe; `--cv-group` auto-infer), multi-fidelity pruning + incremental caching, adaptive held-out guard, failure/seeding/degradation policy. Agent-reviewed; carries reciprocal edits to qc §3 + master §4/D4. | master §4 (`Evaluator`), D4 | **✅ written** |
| `optuna-integration.md` | `OptunaStrategy` — sampler choice, pruning, multi-objective/NSGA-II, SQLite study persistence + concurrency | master §4 (`SearchStrategy`), §6 | planned |
| `mcp-server-design.md` | The `tune_*` tool surface, autonomous vs. steering modes, shared-study session semantics | master §6 | planned |
| `dash-copilot-design.md` | The `/tune/` Dash view — candidate review UI, write-back to study, FEATURES/WORKFLOWS gates | master §6 D5 | planned |

> Convention: companion docs use plain topic filenames (no date prefix) since they
> evolve alongside implementation. The master design keeps its dated
> `YYYY-MM-DD-…-design.md` name as the brainstorming artifact of record.
