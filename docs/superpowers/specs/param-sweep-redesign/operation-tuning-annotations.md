# Operation Tuning Annotations (field-migration workstream)

Companion to the [parameter tuning engine design](2026-06-01-parameter-tuning-engine-design.md)
and to [search-space-inference.md](search-space-inference.md). Specifies the effort to
annotate operation fields so the tuning engine reads **real** search/validity envelopes
from the operation contract instead of guessing.

- **Status:** Planned workstream (pre-implementation). **Decoupled from tune Phase 1** —
  the engine works without it; this is an *experience upgrade* that enables unattended
  (MCP-autonomous) tuning.
- **Maps to:** master §5 (`TuneSpec`), D6 (autonomous MCP driver); improves
  search-space-inference.md §3–4 (the `bounded` path + the `⊆` invariant) and §7 (the
  autonomy gate).

---

## 1. Why this exists — the problem

`infer_search_space` (search-space-inference.md) mines tunable domains from pydantic
operation fields. Its **trustworthy** path is `bounded` — a field with a declared
`Field(ge=, le=)` envelope yields a clean range, `needs_review=False`. But in the current
codebase **essentially no operation declares `Field(ge=, le=)`**: fields are written
`sigma: float = 2.0`, `min_size: int = 50`, `compactness: float = 0.001`, and where bounds
*are* enforced they live in `field_validator` code (invisible to inference). The real
ranges exist — in docstrings ("Typical range: 0.5–5.0"), in validator guards, and in the
authors' heads — just not in the machine-readable contract.

Four consequences (see search-space-inference.md §4 "Reality check"):

1. **Almost every numeric knob routes through the unbounded heuristic** → arrives
   `needs_review=True`. The clean `bounded` path is theoretically correct but practically
   dead.
2. **The autonomy gate (search-space-inference.md §7) trips for essentially every
   pipeline.** `proposal.needs_review` is `True` whenever any unannotated numeric field
   exists — i.e. always — so an MCP agent in "pause if `needs_review`" mode always pauses.
   The gradient the gate was meant to provide collapses to "always on."
3. **The `⊆` invariant has almost nothing to guard.** With no `Field` bounds in metadata,
   a `TuneSpec` is effectively unconstrained, and validator-enforced bounds are invisible
   to it.
4. **The windows are pure guesses.** `[d/4, d·4]` around `min_size=50` → `[12, 200]`;
   around `compactness=0.001` → `[0.00025, 0.004]`. Maybe right, maybe far off.

This workstream moves knowledge **that already exists** into the contract, converting the
engine from "works, but supervised" toward "runs unattended for annotated ops."

---

## 2. The conceptual key: validity bound ≠ search bound

This distinction is *why* there are two mechanisms, and it governs every per-field
decision:

- **`Field(ge=, le=)` — the hard validity envelope.** Values outside are
  invalid/nonsensical (`sigma` must be `> 0`; a fraction must be `∈ [0, 1]`). Set it
  **generously** — the true physical/algorithmic limit. Benefits beyond tuning: it
  hardens validation, and the GUI param forms could render proper slider min/max from it.
- **`TuneSpec(low, high, log=…)` — the sensible search window.** Narrower; where you would
  actually look (`sigma ∈ [0.5, 5]` even though any positive float is valid). **Hint-only
  / non-enforcing** (search-space-inference.md §3).

For an important knob, declare both, and the `⊆` invariant ties them together:

```python
class BlurGauss(ImageEnhancer):
    sigma: Annotated[float, TuneSpec(0.5, 5.0, log=True)] = Field(2.0, gt=0.0)
    #      └ search window (tight, hint)                    └ validity (wide) + default
    #      invariant: [0.5, 5.0] ⊆ (0, ∞) ✓
```

Many fields need only **one**:

- a true hard envelope, no special search preference → `Field(ge=, le=)` alone (the
  `bounded` path handles it, `needs_review=False`);
- valid on all positives/reals but a clear search window → `TuneSpec` alone (`tune_spec`
  path);
- a closed set already (`Literal`/`Enum`/`bool`) → **no work** (already clean).

---

## 3. Where the ground-truth ranges come from

A human migrator (not the engine) reads three existing sources:

- **Docstrings.** Many already state ranges — `BlurGauss.sigma`: *"Typical range:
  0.5–5.0. Keep below the smallest colony radius."* Inference deliberately does **not**
  auto-parse these (search-space-inference.md §4 "No docstring-range parsing"), but they
  are the author's stated intent and the primary input to a manual migration.
- **`field_validator` code.** Explicit guards (e.g. `InoculumDetector` keeps a bound as a
  validator, not `Field(gt=0)`).
- **Domain knowledge.** Pixel scales (`sigma`, `min_size`), fractions/probabilities in
  `[0, 1]`, connectivity in `{1, 2}`, etc.

---

## 4. Staged rollout — `TuneSpec`-first, `Field`-second

### The back-compat hazard

Adding `Field(ge=, le=)` is the one risky move: if a user's existing `pipeline.json`
carries a value outside a newly-added bound, it now raises `ValidationError` **on load**.
For a CI-gated, GUI-coupled feature with serialized user configs, that is a real
regression hazard.

### Why `TuneSpec`-first is safe

Because `TuneSpec` is **non-enforcing**, you can annotate aggressive search windows on
every tunable knob *today* with **zero** back-compat risk — and that alone flips those
knobs to `source="tune_spec"`, `needs_review=False`, with real ranges. So:

- **Phase A — `TuneSpec` annotations** on high-value tunable knobs (detector thresholds,
  enhancer strengths, refiner sizes). Safe, tuning-focused, immediately makes the autonomy
  gate meaningful. **This is the bulk of the tuning benefit at near-zero risk.**
- **Phase B — `Field` validity bounds**, selective and generous, where the true envelope
  is unambiguous and existing valid pipelines already respect it. Broader benefit
  (validation + GUI), but done with care and a back-compat load test (§6).

---

## 5. The migration procedure (per operation)

1. **Inventory** the operation's tunable fields; skip already-closed ones
   (`Literal`/`Enum`/`bool`).
2. **Decide per field** — validity, search, or both (§2) — and **source the range** (§3).
3. **Apply the annotation.** Rewrite `sigma: float = 2.0` →
   `sigma: Annotated[float, TuneSpec(...)] = Field(2.0, gt=0.0)`. Where a `field_validator`
   enforced *only* a scalar bound, convert it to `Field`; where it does normalization or
   conditional logic, **leave it** (those validators are intentional per the project
   conventions) and optionally add a parallel `Field` for the bound it also implies.
4. **Verify.** The docstring-description hook (`apply_docstring_descriptions` via
   `__pydantic_init_subclass__`) re-runs automatically, so `model_json_schema()`
   descriptions still flow. All doctests must stay runnable on `load_synth_yeast_plate()`.

---

## 6. Guardrails — turning migration into an enforceable contract

- **Coverage test.** Assert every *tunable* numeric field has either a `Field` bound, a
  `TuneSpec`, or an explicit `TuneSpec(tunable=False)` — so nothing lands in the
  needs-review bucket **by accident** rather than **by decision**. Implemented as an
  allowlist of not-yet-migrated fields that **shrinks** as the migration proceeds (CI
  fails if the allowlist grows). Advisory during Phase A, hard-gated once migration is
  substantially complete (§9 open question).
- **Invariant test.** Assert `TuneSpec[low, high] ⊆ Field[bounds]` for every annotated
  field — catching author mistakes at *test* time, not just at inference time.
- **Back-compat load test (Phase B).** A set of representative existing `pipeline.json`
  fixtures must still load without `ValidationError` after `Field` bounds are added —
  the regression lock for the validity-bound work.

---

## 7. Scope, sequencing, ownership

- **Decoupled from the tune engine.** Phases 1–6 of the master rollout do not depend on
  this; the engine functions on heuristics + review without any annotation.
- **Independently valuable.** `Field` bounds harden validation and unlock GUI slider
  min/max regardless of tuning — so the workstream earns its keep outside this feature.
- **Incremental, op-by-op.** Prioritize high-value tunable knobs first (detector
  thresholds, enhancer strengths); the long tail of rarely-tuned knobs can wait.
- **Its own PR(s),** not folded into a tuning phase — it is a broad, cross-cutting touch
  across dozens of operation classes, each change small but requiring domain judgment.

---

## 8. Relationship to the tune engine

This workstream is the concrete path from the supervised default to the **D6
MCP-autonomous** vision. The autonomy gate (search-space-inference.md §7) becomes
meaningful exactly as annotation coverage grows:

- a **fully annotated** pipeline returns `proposal.needs_review == False`, so an agent can
  proceed to tune unattended;
- a **partially annotated** pipeline surfaces precisely the unannotated knobs for human
  review — a useful, shrinking worklist rather than an all-or-nothing wall.

So coverage of this migration is, in effect, the dial between "human must review every
tuning run" and "the agent can self-drive."

---

## 9. Open questions

- **Prioritization order** — which operations / knobs migrate first (likely the detectors'
  thresholds and the enhancers' strength parameters, as the highest-leverage tunables).
- **CI gating of the coverage test** — advisory throughout, or hard-gated once a coverage
  threshold is reached? (Hard-gating too early blocks unrelated work; never gating lets
  coverage rot.)
- **GUI slider affordance** — whether to add a GUI param-form enhancement that reads
  `Field` bounds for slider min/max is a separate GUI effort (FEATURES/WORKFLOWS gates),
  noted here only as a downstream beneficiary.
