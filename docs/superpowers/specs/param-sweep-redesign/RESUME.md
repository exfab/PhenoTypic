# RESUME — Parameter Sweep Redesign

Pick-up notes for the `redesign/param-sweep` work. Read this first when resuming
cold, then the [README](README.md) index and the
[master spec](2026-06-01-parameter-tuning-engine-design.md).

## Status snapshot

- **Branch:** `redesign/param-sweep` (pushed to `origin`, exfab/PhenoTypic). No PR opened yet.
- **Commits so far:**
  - `a5d89d16` — master design spec
  - `6515b827` — organize specs into the bundle folder
  - `e8bcc7f1` — revise spec after literature self-review (fANOVA, meta-validation gate, etc.)
  - `ef10d269` — reference-free segmentation-metrics companion doc (+ Chen & Murphy 2023 fix)
  - `51742e13` — supervised-scorers companion doc
  - `ac4c8c9b` — qc-objective-mapping companion doc
  - `cc189216` — search-space-inference + operation-tuning-annotations companion docs
  - *(this commit)* — robust-evaluation companion doc (+ reciprocal qc §3 / master §4·D4 edits)
- **Done:** master design spec; bundle README index; the **reference-free metrics**,
  **supervised-scorers**, **qc-objective-mapping**, **search-space-inference**,
  **operation-tuning-annotations**, and **robust-evaluation** companions.
- **Not started:** 4 stub companion docs (screening-importance, optuna-integration,
  mcp-server-design, dash-copilot-design) + the implementation plan.

## Remaining stub companion docs to go over

Each maps to a master-spec section. Tackle in roughly this order (objectives →
engine internals → surfaces). For each, the relevant literature/decisions already
captured upstream are noted so the doc starts from context, not zero.

- [x] **`supervised-scorers.md`** — master §4 (`Scorer`). **✅ written** (~1,585 lines,
  81 citation-audited sources; 5 families, modality-tiered composite, ARI/VI guard for
  touching colonies, gate-reference role).
  GT-based scoring: count error, IoU/Dice, F1, adjusted Rand; annotation formats
  (colony positions vs. masks), tolerance handling. *Carry-forward:* Jozdani 2020
  (F-measure/QR/SEI track the visual optimum best for over/under-seg); this is the
  trusted reference the meta-validation gate correlates against. See reference-free
  doc §E.

- [x] **`qc-objective-mapping.md`** — master §4, D1 (`QCScorer` is the Phase-1 default).
  **✅ written** (two-phase Scorer, threshold-anchored normalizer + special-value
  policy, coverage-weighted trimmed-mean reduction, hybrid geometric fusion,
  anti-gaming guard; Phase-1 ships Count-only, batch panel follows).
  Reuse the existing `analysis/` QC checks (expected-vs-detected count, ICC,
  MAD/Tukey, edge effects) as the tuning objective. *Carry-forward:* the
  reference-free doc §C.6 already catalogs the directly-portable colony QC
  (Galardini/Viéitez >5% / >90% rules, grid-regularity, border ratio) — this doc
  maps them onto PhenoTypic's existing QC module and clarifies QC-vs-reference-free
  overlap (reuse, don't duplicate).

- [x] **`search-space-inference.md`** — master §5, D7. **✅ written** (two-tier resolution:
  `TuneSpec` marker hint-only + the `⊆` invariant → type/constraint heuristics;
  unbounded-window `[d/unbounded_factor, d·unbounded_factor]` with `d≤0` surfaced;
  `_tune_optional` flat `__enabled__` presence wrapping; one-level nested-op recursion with
  path keys + apply-time class-validation; the `InferredSearchSpace` proposal + autonomy
  gate). Agent-reviewed — the load-bearing `⊆`-retrieval path and `Presence` Cartesian
  semantics were verified against source. *Open-question resolutions:* factor-4 as an
  `unbounded_factor` kwarg; recurse one level on by default; never parse docstring ranges;
  `__enabled__` dunder. *Spun off:* the "bounded is rare today" finding →
  `operation-tuning-annotations.md`.

- [x] **`operation-tuning-annotations.md`** (new — not in the original stub list).
  **✅ written.** Field-migration workstream: `Field(ge=,le=)` (validity) + `TuneSpec`
  (search) so inference reads real envelopes; back-compat-safe `TuneSpec`-first /
  `Field`-second staging; coverage + `⊆` guardrail tests. Decoupled from tune Phase 1;
  the dial toward D6 autonomy.

- [ ] **`screening-importance.md`** — master §4 (`ScreeningPhase`), D8.
  fANOVA importance over the optimizer's own trials (Optuna `get_param_importances`),
  PED-ANOVA for top-performance subspaces, freezing thresholds, zero-dep fallback,
  the importance report. *Carry-forward:* the self-review already settled fANOVA >
  Morris (interactions, reuses trials); **the Böck normalization trap means the
  scores fANOVA consumes must be grid-independent** or importances inherit the
  instability (reference-free doc §B, Recommendation 2).

- [x] **`robust-evaluation.md`** — master §4 (`Evaluator`), D4. **✅ written** (uniform
  3-step Eval↔Scorer loop; per-term `median − λ·IQR` stability penalty, λ=0.5, clamp;
  Scorer-owned normalization; group-aware CV with `--cv-group` auto-infer + label
  overloading; multi-fidelity pruning + incremental per-image-frame caching; adaptive
  held-out guard / CV-only MVP; failure taxonomy, deterministic seeding, availability
  matrix). Agent-reviewed (composition vs qc/master verified; all API claims real).
  **Carried reciprocal edits** to qc §3 (`finalize` signature + `Count_agg` source) and
  master §4/D4 (normalization ownership + per-term penalty scope). *Open:* default `λ`,
  first-rung policy, held-out gap margin, `min_stability_n` (master §14).

- [ ] **`optuna-integration.md`** — master §4 (`SearchStrategy`), §6.
  `OptunaStrategy`: sampler choice (TPE/CMA-ES/GP/NSGA-II), pruning (ASHA/Hyperband),
  multi-objective, SQLite study persistence + concurrency (WAL → RDB), ask-and-tell.
  *Carry-forward:* dependency lives in a `tune` extra (master §10); keep the Protocol
  seam open for a future `AxStrategy`.

- [ ] **`mcp-server-design.md`** — master §6.
  The `tune_*` tool surface, autonomous vs. steering modes, shared-study session
  semantics, transport/packaging. *Carry-forward:* packaging location is an open
  question (master §14, `src/phenotypic/mcp/` vs. standalone).

- [ ] **`dash-copilot-design.md`** — master §6, D5.
  The `/tune/` Dash view: candidate-review UI, write-back to the study, the Pareto /
  objective-curve / importance visuals. *Carry-forward:* trips the `FEATURES.md` /
  `WORKFLOWS.md` CI gates + tutorial-screenshot round-trip (see root CLAUDE.md);
  reuse `_design.py` tokens.

## Reusable review-team recipe (worked well for the reference-free doc)

For any literature-heavy stub, re-run the same pattern:
- **5 parallel researchers** (Opus, `general-purpose`) → each writes a structured
  mini-report to an untracked `_research/` scratch file; split the topic into
  mutually-exclusive lanes.
- **1 synthesizer** (Opus) → merges into the companion doc, cross-referencing the
  master spec; preserves caveat flags; dedups references.
- **1 citation verifier** (Opus) → audits every DOI via scite (`dois` array, no
  `term`, batched ~15/call), checks retractions, spot-checks load-bearing numbers.
- **Mandate:** scite.ai (`mcp__0b8f8a5d-…__search_literature`) + Consensus
  (`mcp__3e1026d1-…__search`) + WebSearch/WebFetch; only cite retrieved papers;
  check `editorialNotices`; flag preprints; small `limit` (8–12) to avoid the
  tool-result overflow.
- Delete `_research/` after synthesis (or keep untracked); commit only the final doc.
- **Always inform the user of the team composition (agents + model + roles) before
  deploying** — that was an explicit instruction.

## Implementation path (when docs are ready)

The master spec §12 defines a 6-phase rollout. The shippable, zero-dependency unit
is **Phase 1** (engine core: `SearchSpace`, `SearchStrategy` Protocol, Grid+Random
strategies, `Evaluator`, `TuningEngine`, `QCScorer`, CLI). Next concrete step:
invoke the **`writing-plans`** skill scoped to Phase 1.

## Global carry-forward caveats (don't re-derive these)

1. **Böck min–max normalization trap.** Never normalize a reference-free/GEOBIA
   score by min–max over the tested parameter set — the argmin then depends on the
   sweep grid, which is *fatal for a grid-sweep tuning engine* and corrupts fANOVA
   inputs. Use fixed/external normalization or rank/covariance combination.
2. **Reference-free meta-validation gate.** Gate any `ReferenceFreeScorer` on
   Spearman-ρ rank agreement **and** an argmax test vs. a small GT set before it
   drives optimization; suggested bars ρ≥~0.7 pass / ≥~0.8 unattended are engineering
   inference, not a cited cutoff (master §14 open question). Validate on a large
   synthetic set + ~10–30 real plates; re-validate per domain (yeast ≠ bacteria).
3. **Screening = fANOVA, not Morris OAT** (categorical/conditional space + interactions).
4. **Chen & Murphy is 2023** (Mol Biol Cell 34(6) ar50); 2021 is the bioRxiv preprint.

## Open questions still on the table (master §14)

- Default `λ` for the stability penalty; fANOVA freezing threshold + warm-up trials;
  pruning low-fidelity representativeness; reference-free correlation threshold + GT
  set size; `Sweep` range types in-place vs. a richer `SearchSpace`; MCP packaging
  location.
