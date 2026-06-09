# Tune Engine — Phase 3: Auto-Space + Reference-Free Scoring (Structured Outline)

> **Status: OUTLINE.** A structured task map, not a full TDD plan. Expand into bite-sized TDD
> tasks before implementing, grounding against the live operation models + `model_json_schema()`.

**Goal:** Remove the two biggest manual burdens — *authoring the search space* and *needing
ground truth*. Add `infer_search_space(pipeline) → InferredSearchSpace` (mine the pydantic op
contract instead of hand-writing ranges), the `--auto-space` CLI, the `TuneSpec` field marker,
**nested-op overlay** in the builder, and a `ReferenceFreeScorer` (no-GT segmentation-quality
proxy) gated behind a mandatory **meta-validation** step.

**Maps to:** `search-space-inference.md` (whole doc); `reference-free-segmentation-metrics.md`
(the scorer + the meta-validation gate); `operation-tuning-annotations.md` (the `TuneSpec`
marker it reads — the annotation *workstream* is parallel, see `workstream-operation-annotations.md`).

**Depends on:** Phase 1 (hand-authored `SearchSpace` + `Scorer` ABC + `build_pipeline`). Phase 2
optional (auto-space + reference-free work with Grid/Random too).

---

## Scope — what it adds / changes

| Phase-1 piece | Phase-3 change |
|---------------|----------------|
| hand-authored `SearchSpace` | add `infer_search_space` → `InferredSearchSpace` proposal → `to_search_space()` |
| `build_pipeline` (flat + presence; **raises on nested keys**) | **lift the nested-key `NotImplementedError`** — one-level nested-op overlay (path keys `1.detectors[0].x` + apply-time class validation) |
| `Knob` provenance fields (`source`/`needs_review`) defaulted | populated by inference (`tune_spec`/`bool`/`enum`/`bounded`/`unbounded_heuristic`/`presence_optin`) |
| `QCScorer` only | add `ReferenceFreeScorer` (+ the meta-validation gate) |

## Key components (interfaces — bodies TBD)

- **`TuneSpec`** field marker (`_search_space/_tune_spec.py`) — `Annotated[float, TuneSpec(0.5,
  3.0, log=True)]` / `TuneSpec(tunable=False)`; read from `model_fields` metadata (the GUI
  registry already walks this). Tier-1 (precise, opt-in).
- **`InferredSearchSpace`** / **`Excluded`** (frozen pydantic value-models, engine-arch §5) —
  `knobs: list[Knob]`, `excluded: list[Excluded]`, summary props, `to_search_space()`.
- **`infer_search_space(pipeline, *, recurse_nested=True) → InferredSearchSpace`**
  (`_search_space/_infer.py`) — two-tier resolution: Tier-1 `TuneSpec`; Tier-2 type/constraint
  heuristics (`bool`→Categorical, `Enum`/`Literal`→Categorical(members), bounded numeric→
  Int/FloatRange with auto-`log`, unbounded `d`→`[d/4, d·4]` flagged `needs_review`, `d≤0`
  surfaced, `NdArrayField`/paths/names→excluded, optional ops→`__enabled__` presence wrap).
  One-level nested recursion → path-keyed knobs; per-knob provenance + the autonomy gate.
- **builder nested overlay** — `_parse_key` learns `pos.list[i].field` (construct leaf op →
  inject into parent kwargs → reconstruct parent); apply-time class validation
  (`detectors[0]` is still an `OtsuDetector`, error loudly if not); skip `None` slots.
- **`ReferenceFreeScorer`** (`Scorer` subclass) — combines no-GT proxies (the curated subset
  from reference-free §catalogue); `availability()` runs the **meta-validation gate**:
  correlate the proxy against a small GT set, **abstain** (unavailable) if the correlation is
  weak. `score_image` returns the proxy term(s).
- **`--auto-space` CLI** — takes a `pipeline.json` positional + `-i` (context); prints the
  review table (`✓`/`⚠ needs_review`/excluded) and writes a proposed `tuning_spec.json`. No run.

## Task breakdown (high-level)

1. **`TuneSpec` marker** + a couple of annotated operation fields as smoke tests (the broad
   annotation rollout is the parallel workstream).
2. **`InferredSearchSpace`/`Excluded` value-models** + `to_search_space()`.
3. **`infer_search_space` Tier-2 heuristics** (flat) — the bulk; table-driven per-type tests.
4. **`infer_search_space` Tier-1** (`TuneSpec` override) + the `⊆` invariant test.
5. **Nested recursion** + **builder nested overlay** + apply-time class validation (the lock:
   a flat-only space still enumerates byte-identically to Phase-1's golden — recursion is
   strictly additive, search-space §6 risk 4).
6. **`--auto-space` CLI** + the review-table renderer.
7. **`ReferenceFreeScorer`** proxies + **the meta-validation gate** (the safety-critical part —
   reference-free §E; abstain-on-weak-correlation).

## Deferred / out of scope
- Supervised scoring + multi-objective → Phase 4.
- MCP `tune_infer_space` tool → deferred MCP.
- Two-level nested presence (`conditional_on` chains of depth >1) → explicitly deferred
  (search-space §6 risk 3).

## Review findings (address at full-planning)

Opus plan-review flagged these — fix when expanding to TDD:

- **Nested keys are a `_parse_key` grammar rewrite, not a branch delete.** `1.detectors[0].x` dot-splits into 3 segments with a non-`__enabled__` tail — which **collides with Phase-1c's 3-segment presence arity**. Specify a real key grammar/tokenizer that disambiguates a 3-segment presence key from a 3-segment nested key, plus a Phase-1c builder-regression gate (every flat/presence key still parses byte-identically). The golden-lock "strictly additive" claim is **contingent** on this two-sided invariant.
- **The meta-validation gate cannot live in zero-arg `availability() -> bool`** (it needs a GT set + the candidate grid). Give it a separate `meta_validate(gt_images, grid)` step that caches a pass/fail; `availability()` reads the cached flag. Keep `availability()` the cheap boolean Phase-1c promised.
- **HARD upstream dependency:** search-space-inference §6 defers the canonical knob parent-identifier form (position index vs `Class#0`) to the master `SearchSpace` design — Phase 3 can't land until that's settled. Add to "Depends on."
- **Tier-2 completeness** for the full plan: the `⊆` check is **blind to validator-enforced bounds** (the common case — see the annotations workstream); **int outward rounding**; **instance-value anchoring** (not class default); the `T | None` union rows; the shared-core/two-adapter equivalence test.
- **Reference-free gate:** lift §E.4's concrete recipe into the body (synthetic-GT Dice/Jaccard, Spearman ρ ≥ ~0.7 pass / ~0.8 unattended + an argmax test, stratified, per-domain re-validation, fail-safe-to-`QCScorer`). Name the term subset (grid/count QC + solidity/circularity/eccentricity + a contrast term + within-replicate size-CV) and require **fixed (not min–max-over-grid) normalization** (the Böck trap). Reuse `QCScorer`'s checks, don't duplicate.

## Open questions for the full plan
- The autonomy gate: when may `--auto-space` be used *non-interactively* (agent/MCP) vs.
  requiring human review of `⚠ needs_review` knobs?
- Reference-free meta-validation: where does the small GT set come from, and what correlation
  floor triggers abstain? (reference-free §E owns the threshold; confirm against real plates.)
