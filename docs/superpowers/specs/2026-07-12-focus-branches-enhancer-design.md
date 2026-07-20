# `FocusBranches` enhancer — design

Status: proposal (not committed). Packages the branch-enhancement recipe developed
while tuning `FocusEdgePhase` on *Neurospora* plates into a single named
`phenotypic.enhance` operation:

```
FlattenIllumination(sigma=200)  →  ContrastStretching(70, 99)  →  FocusEdgePhase(n_orient=8, k=2, min_wavelength=5)
```

All three steps act on `detect_mat`; the output is a phase-congruency edge response in
`[0, 1]`. This is the encapsulation of the empirical result that (a) `min_wavelength≈5`
oriented phase congruency isolates hyphae from agar texture, (b) a narrow-window contrast
stretch lifts faint branches into the response band, and (c) illumination flattening makes
the stretch behave consistently once the plate background is non-uniform (full-scale crops).

## 1 · What it is and where it sits in the taxonomy

`FocusBranches` produces an **edge/ridge response map**, so its purpose-group marker is
`FocusEdge` (`phenotypic.abc_.FocusEdge`), exactly like `FocusEdgePhase`.

**It subclasses the marker `FocusEdge`, not the concrete `FocusEdgePhase`.** Two reasons,
both verified against the codebase:

- **No precedent for concrete-subclasses-concrete.** Every enhancer in `enhance/`
  subclasses a marker ABC directly (`grep '^class .*(FocusEdgePhase' enhance/*.py` is empty).
  The module CLAUDE.md states the rule explicitly: concrete enhancers subclass a purpose-group
  marker, and the only meta-enhancer, `CompositeEnhance`, subclasses `ImageEnhancer` directly.
- **Curated surface.** Subclassing `FocusEdgePhase` would inherit its full parameter set
  (`n_scale`, `mult`, `sigma_onf`, `cutoff`, `g`, `noise_method`, …). `FocusBranches` is a
  *recipe*; it should expose only the knobs that matter for branches and forward the rest at
  `FocusEdgePhase`'s own defaults.

So `FocusBranches` **delegates** to `FocusEdgePhase` (and to `FlattenIllumination` /
`ContrastStretching`) inside `_operate`, the same way `CompositeEnhance._operate` invokes its
child enhancers via `.apply(...)`.

## 2 · Composition and `_operate`

The three steps run sequentially on the **same** `detect_mat`. A single enhancer's
`apply(image, inplace=True)` does **not** reset `detect_mat` from the source channel (reset is
a *pipeline*-level flag, `_image_pipeline_core.py:191`), so chaining accumulates correctly —
each step reads the previous step's output. `FlattenIllumination` clips its result to `[0, 1]`
(`_flatten_illumination.py:161`) and `ContrastStretching` rescales to `[0, 1]`, so the map
handed to phase congruency is always in range.

```python
def _operate(self, image: Image) -> Image:
    if self.flatten:
        FlattenIllumination(sigma=self.flatten_sigma).apply(image, inplace=True)
    ContrastStretching(
        lower_percentile=self.lower_percentile,
        upper_percentile=self.upper_percentile,
    ).apply(image, inplace=True)
    FocusEdgePhase(
        n_orient=self.n_orient, k=self.k,
        min_wavelength=self.min_wavelength, output=self.output,
    ).apply(image, inplace=True)
    return image
```

Order is load-bearing: **flatten first** so the percentile stretch keys off an
illumination-corrected histogram (on a plate with a background gradient, a fixed
`lower_percentile` on the *raw* map would cut unevenly across the field); **stretch second** to
push faint branch pixels into the response band; **phase congruency last** as the detector of
thin oriented structure.

Sub-enhancers are constructed fresh from scalar fields each call (stateless, cheap). They are
**not** stored as nested-enhancer fields — scalar fields keep `to_json()` /
`TuningSpec.from_json` round-tripping and the GUI flat, matching every other enhancer.

## 3 · Fields, defaults, tunability

| Field | Type / default | Tune window | Role |
|---|---|---|---|
| `flatten` | `bool = True` | (not tuned) | toggle the illumination-flatten step |
| `flatten_sigma` | `float = 200.0` | `TuneSpec(100, 400)` | homomorphic background scale |
| `lower_percentile` | `int = 70` | `TuneSpec(2, 90)` | **coverage knob** — higher = fuller/thicker branches |
| `upper_percentile` | `int = 99` | `TuneSpec(95, 100)` | bright clip point |
| `min_wavelength` | `float = 5.0` | `TuneSpec(2, 10)` | **dominant lever** — skips agar-texture band |
| `k` | `float = 2.0` | `TuneSpec(0.5, 20)` | phase noise threshold |
| `n_orient` | `int = 8` | `TuneSpec(4, 8)` | angular resolution for arbitrary hyphal angles |
| `output` | `Literal["pc_sum","M","m"] = "pc_sum"` | (not tuned) | phase quantity passthrough |

Defaults `min_wavelength=5, k=2, n_orient=8` are the branch-tuned values from the sweeps and
**differ from `FocusEdgePhase`'s own defaults** (`3 / 2 / 6`). `lower_percentile=70` is the
requested contrast default (fuller than 50, before the branch-thickening/merging of 80–90).

**Validation** (mirror the sub-enhancers so bad configs fail at construction, not mid-apply):
`Field(ge=2.0)` on `min_wavelength`, `Field(ge=0.0)` on `k`, percentiles constrained to
`[0, 100]` as `int`, plus a `model_validator` asserting `lower_percentile < upper_percentile`.
`ContrastStretching` requires **integer** percentiles (`lower_percentile: Annotated[int, ...]`),
so keep these `int`, not `float`.

## 4 · Registration (three edits — "register or it's invisible")

1. `enhance/_focus_branches.py` → `class FocusBranches(FocusEdge)`.
2. `enhance/__init__.py` — import and add `"FocusBranches"` to `__all__` (the builder registry
   walks `phenotypic.enhance`, so this is what surfaces it in the GUI enhancer dropdown).
3. `tests/unit/abc_/test_enhancer_taxonomy.py` — add `"FocusBranches"` to the `FocusEdge`
   entry of the `TAXONOMY` dict. `TestConcreteEnhancerReparenting.test_concrete_inherits_marker`
   asserts `issubclass(FocusBranches, FocusEdge)`, which holds by construction.

## 5 · Docstring (follow the module's established shape)

Match the `FocusEdgePhase` docstring skeleton: one-line summary, **Best For** (filamentous
plates with faint branches, plate-scale illumination gradients), **Consider Also**
(`FocusEdgePhase` directly for full phase-parameter control; `CompositeEnhance` to fuse it with
another response), **Args** (each field), **Returns** (`Image` with `detect_mat` = branch
response in `[0, 1]`; `rgb`/`gray` unchanged), **Raises** (the validation errors above),
**References** (the phase-congruency citations inherited from `FocusEdgePhase`).

## 6 · Known limitations (carry in the docstring, do not silently absorb)

- **Plate rim, full-scale.** On a whole-plate crop the bright dish wall is a strong real edge;
  phase congruency fires on it, producing a non-branch border response. `FocusBranches` does
  **not** mask the plate — document that a rim-excluding crop or plate mask is the caller's
  responsibility. (A future `FocusBlobLoG`/grid-aware mask could be composed upstream.)
- **`flatten` is a no-op on already-uniform fields** (measured: identical response on the
  uniform central crop) and only earns its keep on non-uniform backgrounds. It is `True` by
  default because it is *harmless when unneeded and load-bearing when needed*; expose the toggle
  so a caller certain of flat illumination can skip the ~0.3s cost.
- **Contrast stretch is image-statistics-dependent.** `lower_percentile` removes background only
  because the branch pixels sit above that percentile; on a plate whose colonies are much fainter
  or whose background is much brighter, `70` may cut real signal or leave agar. This is why the
  step follows `flatten` (which normalizes the background first) and why `lower_percentile` is
  the primary tunable.

## 7 · Tests (`tests/unit/enhance/`)

- **Contract:** `FocusBranches().apply(plate)` returns `detect_mat ∈ [0, 1]`, `rgb`/`gray`
  untouched (the `ImageEnhancer` invariant).
- **Composition equivalence:** `FocusBranches(flatten=False, lower_percentile=70, ...)` produces
  a `detect_mat` **bit-identical** to manually chaining `ContrastStretching(70,99)` then
  `FocusEdgePhase(n_orient=8,k=2,min_wavelength=5)` on the same image — pins that the class is a
  faithful encapsulation, not a reimplementation that can drift.
- **Order is load-bearing:** on a synthetic plate with an injected illumination gradient,
  `flatten=True` yields a more spatially-uniform response (lower across-field variance of the
  branch response) than `flatten=False`; a mutation swapping stretch-before-flatten changes the
  output — proving the ordering is not incidental.
- **Defaults pinned:** `min_wavelength==5.0`, `k==2.0`, `n_orient==8`, `lower_percentile==70`,
  `flatten is True` (guards against a silent default drift).
- **Serialization round-trip:** `FocusBranches(...).to_json()` → `from_json` reconstructs an
  equal instance (flat scalar fields, GUI/tune compatible).
- **Taxonomy:** covered by the roster edit in §4.

## 8 · Open questions

1. **Nested `apply` inside `_operate`.** `CompositeEnhance` establishes that calling a child
   enhancer's `.apply()` from within `_operate` is supported, but confirm there is no
   integrity-validation or progress-tracking guard that rejects re-entrant `apply` when the
   outer call itself came through a pipeline. If there is, call the children's `_operate`
   directly instead (same effect, no re-entrancy).
2. **`output != "pc_sum"` range.** `FocusEdgePhase` clips `pc_sum` to `[0, 1]`; confirm the `M`
   and `m` moment outputs are also `[0, 1]` after its internal clip, or restrict `output` to
   `"pc_sum"` for `FocusBranches` (the only value the sweeps used).
3. **Default hardening.** `lower_percentile=70`, `min_wavelength=5`, `flatten_sigma=200` come
   from a single plate. They are fixed defaults (not grid-derived), but should be sanity-checked
   on 2–3 more plates spanning the illumination/strain range before they harden — the same
   caveat the enhancer sweeps carry. The `PhaseEnhancement_ContrastFullScale` notebook is the
   vehicle for that check.
