# Branch-response reference-free scorers — design

Status: proposal. Turns the two response-map metrics developed while tuning
`FocusEdgePhase` for *Neurospora* branch enhancement — **gini** (concentration)
and **active** (coverage) — plus their notebook composite `gini × min(active/0.10, 1)`
into first-class `phenotypic.tune.score` scorers.

## 1 · Motivation and scoring target

The branch-enhancement sweeps (Neurospora `PhaseEnhancement_*` notebooks) ranked
enhancer configs by two scalars computed on the **enhancer response map**
(`image.detect_mat`, a `[0, 1]` per-pixel edge/ridge response), with no
ground-truth branch mask:

- **gini** — the Gini coefficient of the response values. High ⇒ a few strong
  pixels on a quiet background (thin structures); low ⇒ diffuse (noise/flat).
  Scale-invariant, natively bounded `[0, 1]`.
- **active** — fraction of pixels above `0.05 · p99(response)`; a coverage proxy.
  Natively `[0, 1]`, but **not** monotone-goodness: too low = over-suppressed
  (faint branches dropped), too high = agar noise. The useful signal is a
  one-sided floor.
- **composite** (notebook) — `score = gini × min(active / 0.10, 1.0)`: gini is
  the quality signal, the `min(active/0.10, 1)` factor is a saturating **guard**
  that kills over-suppressed maps without rewarding coverage past the floor.

**Scoring target differs from the existing scorers.** `QCScorer`,
`SupervisedScorer`, and `ReferenceFreeScorer` all score a *segmentation*
(measurements + mask) produced by a detector. These score the **enhancer output
before detection** — `image.detect_mat`. Consequences:

- The tuned pipeline ends at a `FocusEdge` enhancer (no detector/measurements
  required). `_score_terms(image, measurements)` reads `image.detect_mat[:]`
  and **ignores `measurements`** (mirroring how `QCScorer` ignores `image`).
- They belong to the reference-free family conceptually, but they are **not**
  gated behind `ReferenceFreeScorer`'s GT-correlation meta-validation — there is
  no per-colony GT here. `availability()` returns `True` unconditionally (see
  §5 for the caveat on why that is acceptable and what would change it).

## 2 · Contract mapping (the framework does the rest)

Per the `Scorer` base contract (`tune/score/_scorer.py`, `tune/CLAUDE.md`):
each scorer emits **natural goodness** terms in `[0, 1]`, declares
`_TERM_SENSE = HIGHER_BETTER`, and the base `score_image` complements each to
cost `∈ [0, 1]` via `to_cost` (`cost = 1 − goodness`). The `Evaluator`
robust-aggregates each term across the calibration set (`median + λ·IQR`) and
calls `finalize` (default = mean of terms). Authors add **no** scalarization
parameters and do **no** flipping/normalizing by hand.

Both terms are natively bounded `[0, 1]`, so **no `_term_anchor` override is
needed** (anchors are only for unbounded terms). This also keeps us clear of the
"Böck trap" (`reference-free-segmentation-metrics.md §B.3`): the unit interval is
a **fixed** external scale, never min–max over the tested grid, so the optimum
cannot migrate when the sweep endpoints change.

## 3 · The two leaf scorers

### 3.1 `GiniScorer`

Emits one term, `{"Concentration": gini(detect_mat)}`.

- `gini(m)` = Gini coefficient of the flattened, sorted, non-negative response
  values (`(2·Σ i·aᵢ)/(n·Σ aᵢ) − (n+1)/n`); `0.0` for an all-zero map.
- Natively `[0, 1]`, `HIGHER_BETTER`, no anchor. Stateless.
- No parameters. (Optionally a `layer: Literal["detect_mat"] = "detect_mat"`
  field if we ever want to score `gray`/a named channel instead, but detect_mat
  is the only sane default and additional fields are discouraged unless used.)

### 3.2 `ActiveScorer`

Emits one term, `{"Coverage": min(active / active_target, 1.0)}` — the
**saturating guard**, not raw `active`.

- `active(m)` = `mean(m > peak_fraction · percentile(m, 99))`.
- The emitted goodness is `min(active / active_target, 1.0)`: `0` when empty,
  rising linearly, saturating at `1.0` once `active ≥ active_target`. This is the
  correct fixed-normalization of a **non-monotone** raw quantity into a
  monotone-goodness `[0, 1]` — raw `active` must *not* be emitted, because higher
  raw active is not better (noise inflates it).
- Fields (fixed domain anchors, **not** grid-derived, so §B.3-safe):
  - `active_target: float = 0.10` — coverage floor (the notebook `ACTIVE_TARGET`).
  - `peak_fraction: float = 0.05` — response fraction of `p99` counted as "active".
- `HIGHER_BETTER`, no `_term_anchor` (the emitted term is already `[0, 1]`).
  Stateless.

## 4 · The composite: three ways to "combine like we did", and the recommendation

The notebook combiner is a **product of goodnesses**, `gini × retention`. Under
the framework's cost convention (`cost = 1 − goodness`) that is
`cost = 1 − gini·retention`, which **no built-in `CompositeBlend` reproduces
exactly**:

| Combiner | Cost form | Semantics |
|---|---|---|
| Notebook product | `1 − gini·retention` | a strong axis *partially* compensates a weak one (multiplicative) |
| `blend="tchebycheff"` (default) | `max(w_g·(1−gini), w_a·(1−retention))` | **conjunctive** — worst axis dominates, no compensation |
| `blend="weighted_mean"` | `mean of the two costs` | **compensatory** — arithmetic trade-off |

Three implementation options:

- **(A, recommended) Idiomatic composite.**
  `CompositeScorer(scorers=[GiniScorer(), ActiveScorer()], blend="tchebycheff")`.
  Conjunctive: a config must be good on **both** concentration and coverage; a
  weak axis cannot be masked. This is arguably *better* than the notebook product
  for this problem — it makes concentration a hard requirement rather than a
  multiplier, and it comes free with the framework's per-axis diagnostics
  (importance, Pareto via `multi_objective=True`, robust aggregation). It diverges
  from the notebook only in that a very high gini can no longer buy back a
  mediocre coverage (and vice-versa).

- **(B, faithful) Single fused scorer `BranchResponseScorer`.** Emits one term
  `{"BranchResponse": gini · min(active/active_target, 1.0)}`. Reproduces the
  notebook product **exactly** (a single term passes through `finalize`
  unchanged). Cost: forfeits per-axis decomposition — the optimizer sees one
  opaque objective, so no Pareto front and no per-term importance. Choose this
  only if exact parity with the published sweep numbers is required.

- **(C, not recommended) New blend mode.** A `"product"`/`"geometric"`
  `CompositeBlend`. Explicitly rejected in `_composite.py`: a geometric/product
  cost blend inverts the conjunctive property (one perfect axis annihilates the
  product), which is why it was removed. Do not re-add it for this.

**Recommendation:** ship **A** as the default (`GiniScorer` + `ActiveScorer` +
`CompositeScorer(tchebycheff)`), and add **B** as a thin convenience scorer for
teams that want the exact notebook product. A and B agree on the extremes
(both-good → best, either-empty/noisy → penalized) and differ only in the
compensation regime in between; document that one line so users pick knowingly.

## 5 · Known blind spot and validity

Both terms are **statistics of the value distribution, not of spatial
structure**. Dense agar noise (e.g. the `hsv`/`l2` colour-phase maps in the
sweep) produces high `active` + moderate `gini` and lands mid-pack under either
composite — a pile of scattered noise pixels and a connected hypha can share the
same gini/active. This is a real limitation, not a bug:

- It is why `availability()` returning `True` unconditionally is acceptable *for
  ranking within a sensible enhancer family* but should not be sold as a general
  segmentation-quality oracle. The spec deliberately does **not** add a
  meta-validation gate (there is no per-pixel GT), so the honest framing is
  "unsupervised response-map shape proxy," carried in the docstring.
- **Future term (out of scope):** a spatial-structure term (fraction of
  thresholded response in large connected components, or a ridge-continuity
  measure) would separate connected branches from speckle. It slots in as a third
  leaf scorer under the same `CompositeScorer` with no change to A/B. Flagged
  here so the composite is designed to grow, not be rebuilt.

## 6 · Files, registration, tests

Per `tune/CLAUDE.md §"Adding a Scorer"` — *"register or it's invisible."*

New modules in `src/phenotypic/tune/score/`:

- `_gini_scorer.py` → `GiniScorer`
- `_active_scorer.py` → `ActiveScorer`
- `_branch_response_scorer.py` → `BranchResponseScorer` (option B)

Register each:

- Re-export from `tune/score/__init__.py` (add to imports and `__all__`).
- Ensure importable so `_find_class_in_phenotypic` / the class registry resolves
  them for the GUI dropdown and `TuningSpec.from_json` round-tripping.
- Shared helpers: put `gini(m)` and `active(m)` in a small private
  `_response_map_metrics.py` so both leaf scorers and the fused scorer call one
  implementation (single source of truth for the formulas).

Tests (`tests/unit/tune/score/`), matching the module's test discipline:

- **Bounds & orientation:** each term ∈ `[0, 1]`; a hand-built maximally-sparse
  map scores gini→1, a uniform map gini→0; an empty map → `ActiveScorer` term 0,
  a map with `active ≥ target` → term 1.0 (saturation pinned).
- **Fixed-normalization (§B.3):** scoring the *same* map is invariant to which
  other configs are in the sweep — assert the score does not move when the
  calibration set changes (the anti-Böck guarantee).
- **Composite equivalence & divergence:** on a both-good and an
  either-degenerate map, A and B agree on the ordering of extremes; construct one
  intermediate map where the multiplicative product (B) and the conjunctive max
  (A) rank two configs differently, and pin that divergence so a future refactor
  cannot silently collapse them.
- **Mutation check:** reintroduce the raw-`active` bug (emit `active` instead of
  `min(active/target, 1)`) and confirm a noise-heavy map's score jumps — proving
  the guard term is load-bearing.
- **Determinism/statelessness:** the same image scored twice yields identical
  terms (engine reuses one instance across trials).

## 7 · Open questions

1. **Score source coupling.** These read `detect_mat`. Confirm the tuning engine
   populates `detect_mat` for an enhancer-only pipeline (no detector) — if the
   `Evaluator` currently assumes a measurements frame, an enhancer-terminated
   study path may need a small allowance. Verify before implementing.
2. **`active_target` default.** `0.10` is from a single-plate Neurospora crop.
   It is a fixed anchor, not grid-derived, so it is §B.3-safe, but the *value*
   should be sanity-checked on 2–3 more plates before it hardens into a default
   (same caveat the enhancer sweep carries).
3. **Ship B at all?** If nobody needs bit-exact parity with the notebook numbers,
   drop `BranchResponseScorer` and keep only A — two leaf scorers plus the
   existing `CompositeScorer` is the smaller surface.
