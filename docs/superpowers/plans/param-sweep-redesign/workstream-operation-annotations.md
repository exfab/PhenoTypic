# Tune Engine — Operation Tuning Annotations (Structured Outline, Parallel Workstream)

> **Status: OUTLINE.** A structured task map, not a full TDD plan. **Decoupled from the tune
> phases** — it can proceed in parallel with Phase 1+ and lands incrementally per operation.

**Goal:** Annotate operation fields with **validity** bounds (`Field(ge=, le=)`) and **search**
hints (`TuneSpec`) so `infer_search_space` (Phase 3) reads *real* envelopes instead of guessing.
This is the dial toward higher inference autonomy (fewer `⚠ needs_review` knobs) and the D6 MCP
end-state.

**Maps to:** `operation-tuning-annotations.md` (whole doc); supports
`search-space-inference.md` (Tier-1 `TuneSpec` + the type/constraint Tier-2 reading real
`Field` bounds). master §5, D6.

**Depends on:** the `TuneSpec` marker type (defined in Phase 3, `_search_space/_tune_spec.py`).
A handful of annotated fields can land *with* Phase 3 as smoke tests; the broad rollout is this
workstream.

---

## Scope — two orthogonal annotations

| Annotation | Purpose | Read by |
|------------|---------|---------|
| `Field(ge=, le=, gt=, lt=)` | **validity** — the values a param *may* take (already partly used; pydantic enforces) | inference Tier-2 (bounded numeric → `Int/FloatRange`) |
| `TuneSpec(lo, hi, log=, tunable=)` | **search** — the values *worth trying* (a subset of validity) | inference Tier-1 (precise, opt-in) |

**The `⊆` invariant:** the search envelope must be a subset of the validity envelope. This is
the workstream's central guardrail test — a `TuneSpec` range outside the `Field` bounds is a
bug.

## Staging (back-compat-safe — operation-tuning-annotations §"staging")

1. **`TuneSpec`-first** — add `TuneSpec(...)` search hints to operation fields (pure metadata;
   no runtime behavior change, fully back-compat).
2. **`Field`-second** — tighten/add validity bounds where missing (this *can* change validation
   behavior → guarded rollout + tests that previously-valid configs still construct).

## Task breakdown (high-level, per operation group)

For each operation family (enhance → detect → refine → grid → correction), one task each:

1. **Audit fields** — list each tunable scalar/enum/bool field + its current `Field` bounds.
2. **Add `TuneSpec`** search hints (the values worth sweeping; `tunable=False` for seeds/paths).
3. **Add/tighten `Field`** validity bounds where the param has a real domain.
4. **Tests:** the `⊆` invariant (every `TuneSpec` ⊆ its `Field`), inference **coverage** (every
   intended knob is surfaced, none silently excluded), and back-compat (existing constructions
   + serialized pipelines still validate).

## Coverage + guardrail tests (the workstream's contract)

- **`⊆` invariant** across all annotated fields.
- **Inference coverage** — `infer_search_space` over a representative pipeline surfaces the
  annotated knobs with `source="tune_spec"` (Tier-1 wins over Tier-2 heuristics).
- **Back-compat** — no previously-valid op/pipeline fails to construct or deserialize after a
  `Field` tightening.

## Deferred / out of scope
- The inference engine itself (Phase 3) — this workstream *feeds* it.
- Nested-op annotations beyond depth 1 (search-space depth cap).

## Review findings (address at full-planning)

Opus plan-review (premise verified against live `enhance/`+`detect/`) flagged these — fix when expanding to TDD:

- **The `⊆` invariant is BLIND to validator-enforced bounds — the common case here.** Across `enhance/`+`detect/`, **0** ops use `Field(ge=/le=)` and **~15** enforce bounds in a `field_validator`. Those bounds are **not** in `model_fields[].metadata`, so a `TuneSpec` exceeding a validator bound passes the `⊆` check and only fails at apply/trial time. Add the caveat, make apply-time the real backstop, and couple it to the `Field`-second stage (express the bound as `Field(...)` when adding a `TuneSpec` to a validator-bounded field).
- **The autonomy/D6 framing has a concrete mechanism worth stating:** `proposal.needs_review` is `True` whenever *any* unannotated numeric field exists — i.e. **always, today** — so annotation coverage is literally the dial that makes the autonomy gate meaningful (search-space-inference §7).
- **The coverage test = a shrinking allowlist** of not-yet-migrated fields (CI fails if it *grows*), advisory during early rollout → hard-gated once substantially complete. Name that mechanism; lift "when to hard-gate" into the open questions.
- **Migration rule (engages a project convention):** source ranges from docstrings + `field_validator` code + domain knowledge; **convert a validator→`Field` only for a bare scalar bound — leave normalizing/conditional validators in place.** Back-compat: representative existing `pipeline.json` **fixtures** must still load without `ValidationError` after `Field` tightening (the corpus may need building — an unscoped dependency).

## Open questions for the full plan
- Per-op rollout order — annotate the highest-value tuning targets first (detectors,
  enhancers) and let the rest follow?
- Where a `Field` bound would be *too* tight for legitimate edge uses, keep validity loose and
  encode the practical range only in `TuneSpec`?
