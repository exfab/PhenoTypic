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
| `qc-objective-mapping.md` | Reusing the existing `analysis/` QC checks (expected-vs-detected count, ICC, MAD/Tukey, edge effects) as a tuning objective | master §4 (`Scorer`) | planned |
| `search-space-inference.md` | `infer_search_space` — the `TuneSpec` field marker, pydantic type/constraint heuristics, `Presence` auto-wrapping | master §5 | planned |
| `screening-importance.md` | fANOVA importance over optimizer trials (main effects + interactions), freezing thresholds, zero-dep fallback, the importance report | master §4 (`ScreeningPhase`), §3 D8 | planned |
| `robust-evaluation.md` | Calibration/held-out split, aggregation + stability-penalty math, overfitting guard | master §4 (`Evaluator`), §3 D4 | planned |
| `optuna-integration.md` | `OptunaStrategy` — sampler choice, pruning, multi-objective/NSGA-II, SQLite study persistence + concurrency | master §4 (`SearchStrategy`), §6 | planned |
| `mcp-server-design.md` | The `tune_*` tool surface, autonomous vs. steering modes, shared-study session semantics | master §6 | planned |
| `dash-copilot-design.md` | The `/tune/` Dash view — candidate review UI, write-back to study, FEATURES/WORKFLOWS gates | master §6 D5 | planned |

> Convention: companion docs use plain topic filenames (no date prefix) since they
> evolve alongside implementation. The master design keeps its dated
> `YYYY-MM-DD-…-design.md` name as the brainstorming artifact of record.
