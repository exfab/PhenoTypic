# Design Spec — Tier A & B fungi-detection method ports

- **Date:** 2026-07-13
- **Status:** Draft (design, pre-plan) · rev 2 (algorithm cores relocated to a new
  `sdk_.reconnect` helper subpackage; O1 resolved — Tier-B deps behind a `topology` extra)
- **Topic:** Cheap-to-implement reconnection, ridge-enhancement, and false-alarm
  methods that attack hyphal-detection fragmentation, implemented as pure helper
  functions in a new `sdk_.reconnect` subpackage with thin operation wrappers.
- **Related:**
  - Artifact `docs/superpowers/artifacts/2026-06-26-fungi-detection-method-tree/index-fieldnotebook.html` (method tree)
  - Artifact `.../cross-field-reconnection.html` (cross-field reconnection + availability + inputs)
  - Precedent (pure algorithm library consumed by the detector): `sdk_/branch_pathfinding/`
  - Precedent (port discipline): `docs/superpowers/specs/2026-07-08-alt-phase-detection/` (drift register + `verify_claims.py`)

---

## 1. Motivation & goal

The filamentous-fungi detector produces **fragmented** hyphal masks: faint ridges
are missed, junctions break, and the baseline reconnector (multi-source Dijkstra
over a phase-congruency cost surface) bridges gaps but does not explicitly
reinforce bifurcations or reject agar-grain false alarms. The cross-field survey
identified a set of methods that attack exactly these failure modes and are
**cheap here specifically because the repo already ships the substrate they need**:

- `phenotypic.enhance` already provides the phase-congruency front end, Hessian
  ridge filters (Frangi/Sato/Meijering), coherence-enhancing diffusion, and the
  structure-tensor orientation/coherence field.
- `phenotypic.sdk_.branch_pathfinding` already provides a numba multi-source
  Dijkstra plus a composite **cost surface** built from phase-congruency energy,
  fractional anisotropy, orientation θ, orientation coherence, and local MAD.
- `scipy.ndimage.distance_transform_edt`, `skimage` ridge/transform primitives,
  and `numba` are already dependencies.

This spec designs the **Tier A** (do-first) and **Tier B** (cheap experiments)
methods from the implementation-effort map. SOAX/SIFNE/CT-FIRE and the learned-net
riders are out of scope (§2).

**Success criterion:** each algorithm lands as a **pure helper function** in
`sdk_.reconnect` (numpy/numba, no pydantic, no `Image` coupling), reused by a thin
operation wrapper and/or the fungi detector, and pinned by a golden fixture + a
logic-validation script for the numeric claim it rests on; ports also carry a
drift register per the `porting-a-reference-algorithm` skill.

---

## 2. Scope

**In scope (11 methods):**

| Tier | Method | Consumed as | New dep |
|---|---|---|---|
| A | GWDT cost term (APP2) | reconnection cost (detector) | no |
| A | Tensor voting | ridge/junction enhancer + reconnection | no |
| A | Jerman vesselness | ridge enhancer | no |
| A | Bowler-hat transform | ridge enhancer | no |
| A | Kalman predict-and-coast | reconnection strategy (detector) | no (filterpy optional) |
| A | Cellular-automaton track finder | reconnection strategy (detector) | no |
| B | A-contrario / NFA gate | validation refiner | no |
| B | RORPO / oriented path openings | ridge enhancer | no |
| B | Rolling Hough Transform (RHT) | orientation map (enhancer) | no (fil-finder test-only) |
| B | FilFinder skeleton/graph | diagnostic detector (external wrap) | `topology` extra |
| B | Persistence denoise (GUDHI) | ridge denoise (enhancer) | `topology` extra |

**Out of scope (tracked, not designed here):** SOAX/TSOAX (C++/ITK, see the SOAX
decision memo), SIFNE and CT-FIRE/FIRE (MATLAB ports), GNN track building, learned
topology-loss riders on SAM2/SAM3 (training-time), and the generic oriented
filters that merely duplicate the PC/Frangi front end (Gabor, matched-filter bank,
Radon/Hough, template/NCC).

---

## 3. Architecture: `sdk_.reconnect` helper subpackage + thin operation wrappers

**Principle (this rev's central decision — resolves the old open question O2):**
every algorithm **core** is a **pure helper function** in a new subpackage
`phenotypic.sdk_.reconnect`, mirroring `sdk_.branch_pathfinding`: arrays in →
arrays/graphs out, numba where hot, **no pydantic and no `Image`/accessor
coupling**. Domain knowledge (thresholds, seeds, which layer, how to wire) stays
in the *callers* — the thin operation wrappers and the fungi detector. This keeps
the numeric cores unit-testable in isolation, makes the logic-validation scripts
target the helpers directly, and lets the same core serve both a user-facing
operation and the detector's reconnection stage without duplication.

### 3.1 Subpackage layout

```
src/phenotypic/sdk_/reconnect/
  __init__.py              # exports the public helper functions
  CLAUDE.md                # conventions (mirrors branch_pathfinding/CLAUDE.md)
  _gwdt.py                 # grey_weighted_distance(img, seeds, weight_exp, eps)
  _tensor_voting.py        # tensor_vote(response, theta, sigma) -> (stick, ball)
  _jerman.py               # jerman_vesselness(img, sigmas, tau, black_ridges)
  _bowler_hat.py           # bowler_hat(img, scales, n_orientations)
  _kalman.py               # kalman_coast(endpoints, theta, gate_chi2) -> tracks
  _cellular_automaton.py   # ca_track(points, theta, tol) -> tracks
  _nfa.py                  # nfa_meaningful(segments, eps, n_tests_model) -> mask/scores
  _rorpo.py                # rorpo(img, path_length, n_orientations)
  _rolling_hough.py        # rolling_hough(img, dw, dk, z) -> orientation
  _persistence.py          # persistence_denoise(img, threshold)  [lazy gudhi]
```

- Numba kernels use `@numba.njit(cache=True)`; the `branch_pathfinding/CLAUDE.md`
  stale-cache caveat applies — after editing a kernel file, delete
  `sdk_/reconnect/__pycache__` or numba may reuse a stale signature and raise a
  `SystemError` at import. Note this in the subpackage `CLAUDE.md`.
- Helpers stay **import-cheap**: any optional heavy dependency (gudhi for
  `_persistence`, filterpy for `_kalman`) is imported lazily inside the function
  body, so `import phenotypic.sdk_.reconnect` never pulls them.
- Each helper is a plain function with a Google-style docstring, typed
  `np.ndarray` in/out; **no** pydantic, **no** `image.detect_mat`. Conversion
  to/from the `Image` accessor is the wrapper's job.

### 3.2 Thin operation wrappers (delegate to `sdk_.reconnect`)

The wrappers are where the operation ABC + pydantic conventions live; they read a
layer, call one helper, write `detect_mat`. Each is a few lines of glue.

- **Ridge enhancers** → `FocusEdge` marker subclasses in `phenotypic.enhance`
  (mirroring `FocusEdgeSato`), delegating to the helper:
  `FocusEdgeJerman → reconnect.jerman_vesselness`,
  `FocusEdgeBowlerHat → reconnect.bowler_hat` (`FootprintMixin` for the disk SE),
  `FocusEdgeRORPO → reconnect.rorpo`,
  `FocusEdgeRollingHough → reconnect.rolling_hough`
  (`output: Literal["response","orientation"]`; orientation is a **diagnostic**
  map, not detector-safe — the `FocusEdgeMonogenicPhase` precedent),
  `FocusEdgeTensorVoting → reconnect.tensor_vote`
  (`output: Literal["response","junction"]` = stick vs ball saliency),
  `FocusEdgePersistenceDenoise → reconnect.persistence_denoise` (optional dep).
- **Validation refiner** → `NFAValidation(ObjectRefiner)` in `phenotypic.refine`
  delegating to `reconnect.nfa_meaningful`.
- **Reconnection strategies** → *no standalone operation*. The fungi detector's
  reconnection stage gains a
  `reconnect_strategy: Literal["dijkstra","kalman","cellular_automaton"]` field
  and calls `reconnect.kalman_coast` / `reconnect.ca_track`; the GWDT term is
  folded into the cost surface it hands to `run_multisource_dijkstra`.
- **FilFinder** → stays a **wrapper detector** `FilFinderDetector(ObjectDetector)`
  because it wraps an *external* maintained package (not our algorithm to
  helper-ize); it degrades gracefully when fil-finder is absent (the
  `MicroSamDetector` precedent). Its RHT is used only to golden-test our
  `reconnect.rolling_hough`.

`branch_pathfinding` stays untouched and Dijkstra-only; `sdk_.reconnect` is its
sibling for the non-Dijkstra reconnection algorithms and the ridge/gap cores.

### 3.3 Conventions every item follows

- Wrappers follow the `adding-an-operation` skill: keyword-only pydantic fields
  with Google-style `Args:` docstrings; closed value sets as `Literal` aliases in
  `sdk_/typing_.py`; every numeric field tunable or `TuneSpec(tunable=False)` for
  the tune annotation-coverage gate; new enhancers registered (markers unexported)
  and rostered in `tests/unit/abc_/test_enhancer_taxonomy.py`.
- **Ports** (helpers transcribing an external implementation: `_tensor_voting`,
  `_jerman`, `_bowler_hat`, `_rorpo`, `_rolling_hough`, `_cellular_automaton`,
  `_nfa`, and the GWDT geodesic) follow `porting-a-reference-algorithm`: assemble
  the reference locally, cite `file:line` per transcribed line, diff line-by-line,
  golden fixture (all outputs) + behavioural controls, mutation-test, prove the
  fixture fails when the guarded bug is reintroduced, one drift-register row per
  deviation. Because the helpers are pure functions, the golden fixture and the
  logic-validation script both target the helper directly.

---

## 4. Method specifications

Each entry: **helper** (`sdk_.reconnect` function) · **wrapper** · input · key
params · algorithm + reference to transcribe · repo reuse · wrap-vs-port ·
load-bearing numeric invariant (the logic-validation script) · effort.

### Tier A

#### 4.1 GWDT cost term (APP2)
- **Helper:** `reconnect.grey_weighted_distance(img, seeds, weight_exp, eps)` in `_gwdt.py`.
- **Wrapper:** none; the fungi detector composes it into the cost surface it hands
  to `branch_pathfinding.run_multisource_dijkstra`.
- **Input:** `detect_mat` (ridge/intensity map) + seeds. **Params:** `weight_exp`,
  `eps` (tunable via the detector).
- **Algorithm:** grey-weighted distance transform — geodesic distance where step
  cost is a function of inverse local intensity, so the least-cost path follows dim
  signal across a gap. Reference: Xiao & Peng 2013 (APP2), GWDT stage.
- **Reuse:** `scipy.ndimage.distance_transform_edt`; the existing
  `assemble_composite_cost` + `run_multisource_dijkstra`.
- **Wrap/port:** bounded port (geodesic weighting), then compose.
- **Invariant:** on a synthetic 1-D intensity valley, GWDT geodesic cost equals the
  analytic ∑(inverse-intensity·step) along the monotone path; `weight_exp=0`
  reduces to the plain EDT.
- **Effort:** ~1 day.

#### 4.2 Tensor voting
- **Helper:** `reconnect.tensor_vote(response, theta, sigma) -> (stick, ball)` in `_tensor_voting.py` (numba vote kernel).
- **Wrapper:** `FocusEdgeTensorVoting(FocusEdge)` (`output` = response/junction);
  also callable by the detector's reconnection stage.
- **Input:** `detect_mat` ridge map + orientation θ. **Params:** `sigma_vote`,
  `output: Literal["response","junction"]`, tunable.
- **Algorithm:** token tensor `T = (λ₁−λ₂)ê₁ê₁ᵀ + λ₂(ê₁ê₁ᵀ+ê₂ê₂ᵀ)`; cast
  stick/ball votes through a distance-and-curvature-decaying field; tensor-sum;
  re-decompose. Stick saliency `λ₁−λ₂` fills gaps; ball saliency `λ₂` flags
  junctions. Reference: Medioni/Tang tensor-voting formalism; Risser et al. 2008
  for the curvilinear gap-filling variant (C++/MATLAB reference — the one genuine
  port in Tier A). Input is a token field from the ridge map + orientation, not raw
  fragments.
- **Reuse:** orientation θ already computed for the cost surface; numba.
- **Wrap/port:** port (numba vote-field kernel).
- **Invariant:** for two collinear tokens across a gap, stick saliency is maximal
  along the connecting line and decays with the published curvature-penalty kernel;
  for two orthogonal tokens, ball saliency dominates at the crossing. Re-derive the
  saliency decomposition from the tensor sum.
- **Effort:** ~2–3 days (Tier-A capstone).

#### 4.3 Jerman vesselness
- **Helper:** `reconnect.jerman_vesselness(img, sigmas, tau, black_ridges)` in `_jerman.py`.
- **Wrapper:** `FocusEdgeJerman(FocusEdge)` (mirrors `FocusEdgeSato`).
- **Input:** `detect_mat`. **Params:** `sigmas`, `tau`, `black_ridges=False`,
  `mode`, `cval`.
- **Algorithm:** multiscale Hessian eigenvalues; regularize λ₂ against `τ·max|λ₂|`
  → `λ_ρ`; `V = λ₂²·(λ_ρ−λ₂)·[3/(λ₂+λ_ρ)]³`, normalized to [0,1], max over σ.
  Reference: Jerman et al. 2016 (IEEE TMI); MATLAB `timjerman/JermanEnhancementFilter`.
- **Reuse:** `skimage.feature.hessian_matrix` / `hessian_matrix_eigvals` (the
  `FocusEdgeSato` per-σ memory-deletion loop).
- **Wrap/port:** short port (~30 transcribed lines).
- **Invariant:** at a modeled bifurcation Frangi → 0 while Jerman `V > 0`; `V` is
  monotone in the eigenvalue ratio over the vessel regime and normalizes to exactly
  1 at the strongest response.
- **Effort:** hours–1 day.

#### 4.4 Bowler-hat transform
- **Helper:** `reconnect.bowler_hat(img, scales, n_orientations)` in `_bowler_hat.py`.
- **Wrapper:** `FocusEdgeBowlerHat(FootprintMixin, FocusEdge)`.
- **Input:** `detect_mat`. **Params:** `scales`, `n_orientations`, tunable.
- **Algorithm:** `B = max_d[max_θ open(I, line_{d,θ}) − open(I, disk_d)]` — an
  oriented line SE survives across a junction while a disk SE does not. Reference:
  Sazak, Nelson & Obara 2019 (Pattern Recognition); `CigdemSazak/bowler-hat-2d`.
- **Reuse:** `FootprintMixin` disk SE; oriented line SEs by rotating a rectangle
  (`scipy.ndimage.rotate`); `skimage.morphology.opening`.
- **Wrap/port:** port (morphology composition).
- **Invariant:** on a synthetic ×-junction, bowler-hat response at the crossing ≥
  response along a straight bar of equal width (defining property), whereas a plain
  white-top-hat dips at the crossing.
- **Effort:** ~1 day.

#### 4.5 Kalman predict-and-coast
- **Helper:** `reconnect.kalman_coast(endpoints, theta, gate_chi2) -> tracks` in `_kalman.py`.
- **Wrapper:** none; detector `reconnect_strategy="kalman"` + `gate_chi2`.
- **Input:** fragment endpoints + orientation θ.
- **Algorithm:** predict `x̂⁻=Fx̂, P⁻=FPFᵀ+Q`; associate ridge hits inside the
  Mahalanobis gate `d²=νᵀS⁻¹ν ≤ χ²`; update if a hit exists, else **coast**
  (predict-only) through the gap; link tracks. Reference: combinatorial Kalman
  track-finding.
- **Reuse:** orientation θ for the motion model; optional `filterpy` (lazy) or a
  ~40-line numpy KF (no new dep).
- **Wrap/port:** port of the predict-and-coast loop.
- **Invariant:** straight-line model, no measurements → coasted state advances
  exactly along θ; the χ² gate admits/rejects at the analytic Mahalanobis boundary.
- **Effort:** ~2 days.

#### 4.6 Cellular-automaton track finder
- **Helper:** `reconnect.ca_track(points, theta, tol) -> tracks` in `_cellular_automaton.py` (numba parallel cell update).
- **Wrapper:** none; detector `reconnect_strategy="cellular_automaton"` + `tol`.
- **Input:** neighbouring ridge points + orientation θ.
- **Algorithm:** segment graph; CA rule (a cell increments when it has a collinear
  neighbour one state lower, updated in parallel); walk longest descending-state
  chains; emit tracks (forks enumerated). Reference: HEP CA track finder.
- **Reuse:** orientation θ for collinearity; numba.
- **Wrap/port:** port of the cell rule.
- **Invariant:** straight chain → state counter equals chain depth after
  convergence; a fork yields two equal-state terminal cells.
- **Effort:** ~2 days.

### Tier B

#### 4.7 A-contrario / NFA gate
- **Helper:** `reconnect.nfa_meaningful(segments, eps, n_tests_model) -> mask/scores` in `_nfa.py`.
- **Wrapper:** `NFAValidation(ObjectRefiner)` in `phenotypic.refine`.
- **Input:** a labeled/segment map. **Params:** `eps`, `n_tests` model params.
- **Algorithm:** keep only structures with `NFA = N_tests · P_H0(≥k of n aligned) ≤ ε`.
  Reference: Desolneux–Moisan–Morel; NFA as used in LSD (Grompone von Gioi).
- **Reuse:** `scipy.stats` binomial tail; existing refiner scaffolding.
- **Wrap/port:** port of the NFA test.
- **Invariant:** under a uniform-random-orientation H0, the expected count of
  ε-meaningful detections ≤ ε (Monte Carlo vs the binomial-tail formula).
- **Effort:** ~1–2 days.

#### 4.8 RORPO / oriented path openings
- **Helper:** `reconnect.rorpo(img, path_length, n_orientations)` in `_rorpo.py` (numba path-opening DAG).
- **Wrapper:** `FocusEdgeRORPO(FocusEdge)`.
- **Input:** `detect_mat`. **Params:** `path_length` (L), `n_orientations`.
- **Algorithm:** path-open along ~4 orientations, sort, `RORPO = PO^{θ1} − PO^{θk}`.
  Reference: Merveille et al. 2018 (IEEE TPAMI); `path-openings/RORPO`.
- **Reuse:** numba (no skimage path-opening exists → genuine bounded port).
- **Wrap/port:** port.
- **Invariant:** single straight line → large rank-gap; isotropic blob → ~0;
  incomplete-path variant tolerates a parametrized k-pixel gap.
- **Effort:** ~2 days.

#### 4.9 Rolling Hough Transform
- **Helper:** `reconnect.rolling_hough(img, dw, dk, z) -> orientation` in `_rolling_hough.py`.
- **Wrapper:** `FocusEdgeRollingHough(FocusEdge)` (`output: Literal["response","orientation"]`; orientation is diagnostic).
- **Input:** `detect_mat`. **Params:** `window_diameter` (D_W),
  `smoothing_diameter` (D_K), `coherence_fraction` (Z).
- **Algorithm:** unsharp-mask → bitmask; roll a circular window, 1-D Hough at ρ=0,
  record θ iff ≥ Z·D_W on-bits align. Reference: Clark, Peek & Putman 2014;
  optional `fil-finder` RHT as a golden-test reference.
- **Reuse:** the `output`-Literal diagnostic-map pattern from `FocusEdgeMonogenicPhase`.
- **Wrap/port:** port (or wrap fil-finder's RHT).
- **Invariant:** on a faint sub-threshold diagonal, RHT recovers the orientation a
  per-pixel gradient does not; Z gates noise as specified.
- **Effort:** ~1–2 days.

#### 4.10 FilFinder skeleton/graph (external wrap)
- **Helper:** none — it wraps an external maintained package, not our algorithm.
- **Wrapper:** `FilFinderDetector(ObjectDetector)` in `phenotypic.detect`, optional
  dep, graceful degradation (`MicroSamDetector` precedent).
- **Input:** a ridge mask (`detect_mat` thresholded). **Params:** pruning
  length/intensity thresholds (pass-through).
- **Algorithm:** adaptive mask → medial-axis skeleton → prune → longest-path spine
  + width. Reference: Koch & Rosolowsky 2015; `pip install fil-finder` (MIT).
- **Wrap/port:** wrap (+ optional dep).
- **Invariant:** none to re-derive (wraps a maintained package); golden fixture on
  `load_synth_yeast_plate` + a smoke test that skips cleanly when fil-finder is absent.
- **Effort:** hours + dependency decision.

#### 4.11 Persistence denoise (GUDHI)
- **Helper:** `reconnect.persistence_denoise(img, threshold)` in `_persistence.py` (lazy gudhi import).
- **Wrapper:** `FocusEdgePersistenceDenoise(FocusEdge)`, optional dep.
- **Input:** `detect_mat` scalar ridge map. **Params:** `persistence_threshold`
  (Nσ / lifetime cut).
- **Algorithm:** sublevel-set filtration → persistence pairs (β₀/β₁); cancel pairs
  below the threshold. Reference: GUDHI cubical complex; DisPerSE-style cancellation.
- **Reuse:** `gudhi` for persistence pairs (lazy); the cancellation/Morse step is a
  small addition.
- **Wrap/port:** wrap (persistence) + small port.
- **Invariant:** one strong ridge + N low-amplitude bumps → N sub-cut persistence
  pairs; cancelling leaves exactly the strong ridge (β₀ count matches).
- **Effort:** ~1–2 days + dependency decision.

---

## 5. Dependencies

- **Tier A: no new dependencies.** numpy/scipy/skimage/numba only. `filterpy` is an
  optional lazy import for `_kalman` (a ~40-line numpy KF avoids it).
- **Tier B deps (O1 RESOLVED):** `_rolling_hough` is ported **dependency-free**
  (numpy/scipy/numba); `fil-finder` is only a **test oracle** for its golden
  fixture, so it goes in the **dev dependency group**, not the runtime deps. The two
  runtime-optional packages are gated behind one new extra **`phenotypic[topology]`**:
  - `gudhi` (MIT) — required by `_persistence`, lazy-imported inside the helper.
  - `fil-finder` (MIT) — required by `FilFinderDetector` at runtime.
  Both degrade gracefully when absent (`ImportError` handled at call time, the
  `MicroSamDetector` precedent). Verify manylinux / macOS-arm64 / Windows wheels for
  `gudhi` before wiring the extra; where a platform lacks a wheel, `_persistence`
  stays unavailable there rather than blocking install. Nothing in Tier A or the
  base package gains a runtime dependency.

---

## 6. Testing & validation strategy

Because the cores are pure helpers, tests target `sdk_.reconnect` directly, with a
thin wrapper test on top:

1. **Helper unit test** — `sdk_.reconnect.<fn>` on synthetic arrays (the invariants
   in §4), plus a golden fixture of all helper outputs.
2. **Wrapper doctest on `load_synth_yeast_plate()`** — every operation ships a
   runnable example (repo rule); a spy confirms the wrapper forwards params to the
   helper (the phase-congruency forwarding lesson: test the call, not just a number).
3. **Logic-validation script** —
   `docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/<method>.py`
   re-derives the §4 invariant from scratch (stdlib + numpy/scipy, no `phenotypic`
   import, non-zero exit on failure). Committed alongside the implementation.
4. **Behavioural controls + mutation test** (ports) — prove the fixture *fails* when
   the guarded bug is reintroduced (e.g. Jerman's τ-regularization dropped, tensor
   voting's curvature penalty removed). A silent skip is a failure, not a pass.
5. **Drift register** (`drift-register.md` in this spec folder) — one row per
   deviation per port (the `2026-07-08-alt-phase-detection` precedent).
6. **Taxonomy + tune gates** — new enhancers rostered in
   `tests/unit/abc_/test_enhancer_taxonomy.py`; every numeric field tunable or
   `TuneSpec(tunable=False)`.

Numeric tolerances derive from a mechanism (e.g. 1 ulp × N ops), not a guess.

---

## 7. Phasing & sequencing

- **Phase 0 (scaffold):** create `sdk_/reconnect/` (`__init__.py`, `CLAUDE.md` with
  the numba-cache caveat) — an empty, import-cheap subpackage.
- **Phase 1 (cost + cheap enhancers, no new dep):** `_gwdt` (+ detector wiring) →
  `_jerman` + `FocusEdgeJerman` → `_bowler_hat` + `FocusEdgeBowlerHat`. Establishes
  the helper→wrapper→fixture→logic-validation→drift-register harness on small cases.
- **Phase 2 (capstone):** `_tensor_voting` + `FocusEdgeTensorVoting` (the one real
  Tier-A port; covers gaps and junctions).
- **Phase 3 (reconnection strategies):** `_kalman`, `_cellular_automaton`, and the
  detector `reconnect_strategy` selector.
- **Phase 4 (Tier B):** `_nfa` + `NFAValidation`, `_rorpo` + `FocusEdgeRORPO`, and
  `_rolling_hough` + `FocusEdgeRollingHough` — all **dependency-free**; then add the
  `phenotypic[topology]` extra (`gudhi`, `fil-finder`) with `_persistence` +
  `FocusEdgePersistenceDenoise` and the `FilFinderDetector` wrapper.

Each phase is independently shippable and reviewable.

---

## 8. Risks & open questions

- **~~O1 — Tier B dependencies~~ (RESOLVED):** `_rolling_hough` ported
  dependency-free (`fil-finder` demoted to a dev/test-only oracle); `gudhi`
  (`_persistence`) and `fil-finder` (`FilFinderDetector`) gated behind one optional
  `phenotypic[topology]` extra, lazy-imported and gracefully degrading. Tier A and
  the base package take no new runtime dependency. See §5.
- **~~O2 — reconnection home~~ (RESOLVED):** all algorithm cores live in the new
  `sdk_.reconnect` helper subpackage; `branch_pathfinding` stays Dijkstra-only.
- **R1 — Tensor voting is the only genuine port in Tier A** (C++/MATLAB reference).
  Budget ~3 days with the full port discipline; do not under-scope the vote kernel.
- **R2 — Domain fit.** Defaults must be tuned on `load_synth_yeast_plate` /
  filamentous exemplars, not inherited from the microscopy/vascular literature
  (cf. the colour-PC finding that colour buys nothing on round plates).
- **R3 — Diagnostic vs detector-safe outputs.** `rolling_hough` orientation and
  `tensor_vote` ball/junction maps are diagnostic (like the monogenic angle
  outputs) and must not be fed straight into a detector; enforce via the wrapper's
  `output` Literal + a docstring warning.
- **R4 — Helper purity boundary.** Keep `sdk_.reconnect` free of `Image`/pydantic
  imports so it stays a testable algorithm library; all accessor conversion and
  parameter validation lives in the wrappers/detector (the `branch_pathfinding`
  discipline).

---

## 9. Deliverables checklist

- [ ] Phase 0: `sdk_/reconnect/` subpackage (`__init__.py`, `CLAUDE.md`).
- [ ] Phase 1: `_gwdt` (+ detector cost wiring) · `_jerman` + `FocusEdgeJerman` ·
      `_bowler_hat` + `FocusEdgeBowlerHat` — each with helper unit test, wrapper
      doctest, golden fixture, logic-validation script, drift register, taxonomy +
      tune-gate entries.
- [ ] Phase 2: `_tensor_voting` + `FocusEdgeTensorVoting` (`output` = response/junction).
- [ ] Phase 3: `_kalman`, `_cellular_automaton` + detector `reconnect_strategy` selector.
- [ ] Phase 4 (dependency-free): `_nfa` + `NFAValidation` · `_rorpo` +
      `FocusEdgeRORPO` · `_rolling_hough` + `FocusEdgeRollingHough`.
- [ ] Phase 4 (`topology` extra): `pyproject.toml`
      `[project.optional-dependencies] topology = ["gudhi", "fil-finder"]`,
      `fil-finder` added to the dev group as the RHT test oracle · `_persistence` +
      `FocusEdgePersistenceDenoise` · `FilFinderDetector`.
- [ ] `drift-register.md` in this folder (rows per port).
- [ ] Logic-validation scripts under
      `docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/`.
