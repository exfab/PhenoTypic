# 02 — Setup: Search Space & Scorer

Setup answers two questions — **what is tunable** and **how is it judged** — as a
single progressive-disclosure page with three sections: Pipeline, Search space,
Scorer. Pipeline is the gate (doc 01). This doc specifies the Search space and
Scorer sections, plus the one model change they require.

## Layout: progressive form, not a wizard

A single scrolling page with collapsible sections; rare controls (per-knob
domain editor, evaluator internals) hide behind progressive disclosure. We chose
this over a step-by-step wizard (D6) because the audience is expert-leaning,
`infer_search_space` means the biggest section arrives pre-filled, and *editing
an existing spec* is a first-class use case a wizard makes slow. A first-timer
wizard overlay is a deferred non-goal.

## Search space section

### Prefill from inference

On pipeline selection, the section is populated by
`infer_search_space(pipeline)` (`tune/_search_space/_infer.py`). Each proposed
**knob** renders as a table row: an on/off switch, the target
(`Operation.field`), a clickable **domain summary**, and a **source** badge.

The **source** badge surfaces the knob's `KnobSource`, i.e. how inference chose
the domain — and it is load-bearing for trust:

| Badge | `KnobSource` | Meaning |
|-------|-------------|---------|
| `bounded` | bounded | Range derived from a real min/max field validator. Trustworthy. |
| `bool` / `enum` | bool / enum / literal | Domain taken directly from the type. Exact. |
| `presence` | presence_optin | An op on/off (`__enabled__`) structural knob. |
| `heuristic · review` | unbounded_heuristic | No validator/annotation — inference *guessed* a range from the default (≈`default/4 … default×4`, log when span > ~10×) and set `needs_review = True`. Verify before spending budget. |

`needs_review` knobs are visually flagged so the user sanity-checks the guessed
range. Toggling a knob **off** pins it (becomes `Fixed`, excluded from the
sweep).

### Per-knob domain editor

Clicking a knob's domain summary expands an inline editor (kept inline, not a
popover, to avoid clipping inside the section card). It models the two distinct
routes to discreteness:

- **Range mode** — `low`, `high`, an optional **`step`**, and a **"Sample across
  orders of magnitude"** toggle (the user-facing name for log scale; D9).
  - Empty step = continuous (float) or stride-1 (int).
  - A set step = uniform quantization.
  - **Step ↔ magnitude-sampling are mutually exclusive** (Optuna forbids
    `log=True` with `step≠1`): setting one disables the other, with an inline
    note explaining the constraint. This mirrors the guard in
    `tune/_strategies/_optuna.py`, which silently forces `step=1` under log
    today — the UI makes that visible instead.
- **Choices mode** — an explicit value list (chips) → `Categorical`. This is the
  route for *arbitrary* discrete sets (`{0.5, 1, 2, 5, 10}`), as opposed to
  uniform stepping.

The collapsed summary reflects state compactly, e.g. `20–400 · step 1 ·
by-magnitude` or `{0.5, 1, 2, 4}`.

Bool/enum/presence knobs open straight to a choices/structural editor (no
step/magnitude — discrete sets don't have them). The presence knob notes that
its conditional children are **v1-deferred** (see non-goals).

### Naming: "Sample across orders of magnitude"

The log-scale control is labeled **"Sample across orders of magnitude"** with a
`?` tooltip; the compact chip tag reads **`· by-magnitude`**; the term *"log
scale"* survives only as the tooltip alias (it still maps to the `log: true`
spec field and Optuna's own vocabulary). "Scale-invariant sampling" was rejected
as higher jargon than the term it replaces. See D9.

### Help tooltips from pydantic descriptions

Each target carries a `?` icon whose hover/focus tooltip shows the parameter
name, type, and **docstring** — sourced at runtime from the pydantic field
description (the same Google-style `Args:` text `model_json_schema()` exposes),
so it can never drift from the code.

### Scale affordances

Real pipelines reach 20–40 knobs, so the table has a toolbar:

- a **filter** box (live-filters rows by target text) with a "showing X of N"
  count;
- a **needs-review only** toggle (isolates the heuristic-flagged knobs);
- **Re-infer** — re-runs inference against the current pipeline, **preserving
  manually edited knobs** and resetting only untouched ones; added ops
  contribute knobs, removed ops drop theirs. The edit-preservation behavior is
  surfaced as an explicit note (it's the non-obvious part).
- **bulk actions** (disable-all-needs-review, reset-to-inferred).

### Feasibility feedback

A live line reports grid feasibility: if any active knob is a continuous float,
grid is unavailable ("give it a step, pin it, or use Optuna"); otherwise it
estimates the candidate count. This is the search-space half of the
strategy↔space coupling that Run's pre-flight (doc 03) enforces at deploy.

## The `FloatRange.step` model change (prerequisite)

`IntRange` already has `step` (fully wired into grid/random/Optuna).
`FloatRange` has **no step today** — it is continuous-only, and grid *rejects*
it outright. The domain editor's float-step affordance therefore requires a
real model extension:

1. Add `step: float | None = None` to `FloatRange`
   (`tune/_search_space/_domains.py`).
2. Add a `values()` method that quantizes when `step` is set, so grid can
   enumerate it. **Use `linspace`-style generation** —
   `num = round((high - low) / step) + 1`, then `low + i*step` — rather than
   `numpy.arange`, which accumulates float error and is unreliable at the
   endpoint (e.g. `arange(0.1, 0.4, 0.1)` may drop `0.4`). Round to a sane
   precision so the enumerated values are clean.
3. Wire `step` into: Optuna (`suggest_float(..., step=...)`), random sampling
   (choice from the quantized set), and grid enumeration.
4. Add the step↔log guard mirroring `IntRange`'s (`log=True` requires step
   absent; warn/normalize otherwise).

The payoff is twofold: quantized-float search *and* a stepped float becomes
grid-enumerable, removing the current hard "grid can't do FloatRange" error for
that knob. (Inference does not auto-set a float step; it stays a human decision.)

## Validation (search-space portion)

The section participates in the blocked-deploy contract (formalized in doc 04).
Search-space-level issues:

- **No active knobs** — nothing to tune.
- **`low ≥ high`** on any active range knob — a bounds error. This is also the
  *only* place relational validation lives, and it is **client-side only**:
  there is no optimizer-level constraint mechanism (no Optuna `constraints_func`
  is wired), so the GUI red-flags it and blocks launch but cannot make the
  sampler avoid an infeasible region. Cross-knob constraints
  (`min_area < max_area`, `a + b ≤ N`) are an explicit non-goal.

Issues raise a red section badge + inline field errors and feed the aggregated
footer (doc 04).

## Scorer section

The scorer defines the objective (higher = better; the optimizer maximizes it),
so it lives in Setup with the search space — changing it changes *what
experiment you're running*.

- Choice of scorer: **QC** (expected-vs-detected count), **Reference-free**,
  **Supervised** (GT masks), **Composite / multi-objective**.
- Params render via the shared `gui/_param_forms.py` machinery (same as the
  Builder), driven by each scorer's pydantic fields.
  - **Planning spike (unverified):** scorers (`QCScorer`, `ReferenceFreeScorer`,
    `SupervisedScorer`, `CompositeScorer`) are pydantic models but **not
    `ImageOperation` subclasses**, and `param_form` is built around the
    `OperationRegistry`'s `ParamInfo`. The plan must first confirm `param_form`
    can render a non-operation pydantic model directly; if it can't, add a small
    scorer registry (walk `scorer.model_fields`) or a thin `ParamInfo` adapter.
    Resolve this before committing the scorer-section UI design.
- **QC requires a metadata CSV** (expected counts); absence is a blocking
  validation issue scoped to the scorer section (doc 04). The requirement is
  scorer-specific — switching away from QC clears it.

## Non-goals carried by this surface

- **Conditional / nested knobs.** `Knob.conditional_on` is fully honored by all
  three strategies (`is_active()`), but inference always emits `None` in v1
  (presence-opt-in is off), so nothing populates it. v1 = independent per-knob
  domains; presence rows flag their would-be children as deferred. No engine
  work is needed to add this later — only the authoring UI.
- **Relational constraints** between knobs (see Validation above).
