# Design: `ContrastGamma` / `ContrastLog` / `ContrastSigmoid` + the `input_layer` mixin

- **Date:** 2026-07-09
- **Status:** Design — awaiting approval for planning
- **Branch / worktree:** `contrast-enh`
- **Scope:** three new `ContrastAdjustment` enhancers wrapping skimage's
  `adjust_gamma` / `adjust_log` / `adjust_sigmoid`; a reusable `InputLayerMixin`
  that **appends** an `input_layer` field to a pydantic operation; retrofitting
  `ContrastStretching` with the same capability; a `compute_from_rgb` path on
  `DetectionMode`; and closing the one remaining `detect_mat ∈ [0,1]` violation.

---

## 1. Motivation & summary

`adjust_gamma`, `adjust_log`, and `adjust_sigmoid` are **pointwise** intensity
curves. Applying such a curve to a 3-channel RGB image and *then* projecting to a
1-channel detection matrix is **not** the same as projecting first and applying the
curve to the result — precisely because the curve is non-linear and projection is
(for most detect modes) linear. Gamma-correcting the red channel before taking
`MinRGB` gives a different, and often better, colony/background separation than
gamma-correcting `MinRGB` itself.

Today every `ImageEnhancer` reads `detect_mat` and only `detect_mat`. This spec adds
an opt-in second input domain — the pristine `rgb` layer — while leaving the output
contract untouched: **the only layer an enhancer ever writes is `detect_mat`.**

The capability is delivered as a mixin (`InputLayerMixin`) rather than baked into the
four ops, so future enhancers with a meaningful RGB-domain interpretation can adopt it
by inheritance.

### Deliverables

| Kind | Item |
|---|---|
| New op | `ContrastGamma` (`gamma`, `gain`) |
| New op | `ContrastLog` (`gain`, `inv`) |
| New op | `ContrastSigmoid` (`cutoff`, `gain`, `inv`) |
| New mixin | `InputLayerMixin` → appends `input_layer: InputLayer` |
| Retrofit | `ContrastStretching` gains `input_layer` + `per_channel` |
| New core API | `DetectionMode.compute_from_rgb(rgb, *, image)` on all 11 modes |
| Extraction | `rgb_to_xyz(...)` free function, lifted out of `XYZAccessor` |
| Invariant | `FocusEdgeLaplace` normalized; CI gate pins `detect_mat ∈ [0,1]` |

---

## 2. Evidence gathered during design

Every load-bearing claim below was measured against this repo at `92f15359a`, not
recalled. The numbers drive the design decisions in §4–§6.

### 2.1 `gain` is annihilated by a full-range rescale

`adjust_gamma` computes `((I/scale)**gamma) * scale * gain` and `adjust_log` computes
`scale * gain * log2(1 + I/scale)`. In both, `gain` is a **uniform multiplicative
factor applied after the curve**, so `rescale_intensity(out, out_range=(0,1))` (which
maps `min→0`, `max→1`) divides it back out exactly.

Measured `max|Δ|` when `gain` moves `1.0 → 1.9`:

| Op | under `rescale_intensity` | under `np.clip` |
|---|---|---|
| `adjust_gamma` | `1.192e-07` (float32 rounding) | `4.733e-01` |
| `adjust_log` | `2.384e-07` (float32 rounding) | `4.736e-01` |
| `adjust_sigmoid` | `1.370e-01` | — |

`adjust_sigmoid` is exempt: its `gain` sits **inside the exponent**
(`scale/(1+exp(gain*(cutoff - I/scale)))`), reshaping the curve rather than scaling it.

**Consequence:** a `rescale`-only output policy would ship a tunable `gain` on
`ContrastGamma`/`ContrastLog` that provably cannot change the result, and the tune
search would burn trials on it. See §5.

### 2.2 `clip` is the house style; `rescale` is an algorithm, not a guard

Across `src/phenotypic/enhance/`: **9 modules call `np.clip`, 1 calls
`rescale_intensity`** — and that one is `ContrastStretching`, where rescaling between
percentiles *is* the algorithm. Five enhancers already expose the guard as a field
(`clip: bool = True`): `BayesShrinkEnhancer`, `EnhanceBlockMatch`, `LocalEdgeDenoise`,
`VisuShrinkEnhancer`, and `CompositeEnhance` (which defaults it to `False`).

Semantically the two differ in kind. Clip is the identity on in-range pixels and only
touches the tails, preserving absolute intensity. Rescale preserves ordering but
discards absolute scale and makes the output a function of the input's **extremes** —
one specular highlight sets the max, and two plates in one batch receive two different
mappings. That is exactly why `ContrastStretching` rescales between *percentiles*
rather than between min and max.

### 2.3 `clip=False` already means "leave the values alone" — and it is load-bearing

`_GATSupportMixin._GAT_DEFER_ATTRS` is documented as *"Boolean attributes that must be
`False` inside the GAT region … any knob that would corrupt the stabilized
round-trip."* Seven classes list `"clip"` there: `VisuShrinkEnhancer`,
`BayesShrinkEnhancer`, `LocalEdgeDenoise`, `EnhanceBlockMatch`, plus
`_visushrink_corrector` and `_bayesshrink_corrector`. `ClipControlMixin._disable_clipping`
sets the same flag, and **duck-types on any op exposing a `.clip` attribute.**

Measured on the synth plate: `detect_mat ∈ [0.5451, 0.9548]` maps into the GAT-stabilized
domain at `[1.9185, 2.3065]`. Rescaling that stabilized signal to `[0,1]` before the
inverse GAT collapses the reconstruction to **all zeros** (`max|err|` `1.87e-01` →
`9.55e-01`).

**Consequence:** `clip=False ⇒ rescale` cannot be a global contract, and a field literally
named `clip` on the new ops would be found and set by `_disable_clipping`, silently
rescaling inside a composite that required pass-through. See §5.

### 2.4 The `detect_mat ∈ [0,1]` invariant is one op away from true

All 30 zero-arg enhancers in `enhance.__all__`, applied to `load_synth_yeast_plate()`:

- **29 respect `[0,1]`.**
- **1 violates it:** `FocusEdgeLaplace` at `[-1.5157, +1.4787]` (114,268 negative
  pixels — a Laplacian is signed by construction).

Separately, all three new skimage functions **raise `ValueError`** on negative input
(*"Image Correction methods work correctly only on images with non-negative values"*).

I initially assumed background subtraction was the negative-value source. It is not:
`rolling_ball` and `white_tophat` subtract a background bounded above by the image, and
`SubtractGaussian` already clips. Measured minima — `SubtractRollingBall` `+0.0000`,
`SubtractWhiteTophat` `+0.5451`, `SubtractGaussian` `+0.0000`, `FlattenIllumination`
`+0.6399`. The hazard is edge filters, not subtraction.

### 2.5 Joint vs per-channel stretching are *both* established practice

- **GIMP** "Stretch Contrast" ships a **"Keep colors"** checkbox, documented as
  *"Impact each color channel with the same amount"* (joint). Unchecked ⇒ per-channel.
- **Photoshop's** Auto Color Correction Options offers **"Enhance Monochromatic
  Contrast"** (joint; the `Auto Contrast` command) versus **"Enhance Per Channel
  Contrast"** (per-channel; `Auto Tone`), noting the latter *"may remove or introduce
  color casts."*
- **scikit-image** `rescale_intensity(in_range='image')` is **joint** — its
  `intensity_range` branch is `i_min = np.min(image); i_max = np.max(image)`, a whole-array
  reduction. Confirmed empirically: on a red-dominant array only the red channel reached
  `1.0` (`[1.0, 0.336, 0.206]`).
- **earthpy's** `_stretch_im` (used by `plot_rgb(stretch=True)`) loops
  `for ii, band in enumerate(arr)` and takes `np.nanpercentile(band, …)` **per band** —
  the remote-sensing convention.
- **ImageJ** is an outlier: *"normalization of RGB images is not supported."*

**Consequence:** per the "if both methods apply then use the flag" rule,
`ContrastStretching` gets `per_channel: bool = False`. See §6.

### 2.6 The field-append mixin works (verified against the real ABC)

A throwaway `ContrastGamma(InputLayerMixin, ContrastAdjustment)` was built against
pydantic 2.12.5 and probed:

- MRO resolves: `ContrastGamma → InputLayerMixin → ContrastAdjustment → ImageEnhancer →
  FootprintMixin → ImageOperation → BaseOperation → BaseModel → LazyWidgetMixin → ABC`.
- `input_layer` lands **last** in `model_fields`, `model_json_schema()["properties"]`,
  `model_dump()`, and the `to_json()` `params` envelope.
- `BaseOperation.__pydantic_init_subclass__`'s `apply_docstring_descriptions` still
  populates `input_layer`'s description from the `Args:` block.
- **`TuneSpec` metadata on `gamma` survives `model_rebuild(force=True)`** — the primary
  risk, since a forced rebuild regenerates the core schema.
- `ContrastGamma(input_layer="gray")` raises `ValidationError`.
- `ContrastStretching.model_fields` is unchanged — the mixin does not leak.

---

## 3. `InputLayerMixin`

**Location:** `src/phenotypic/sdk_/mixin/_input_layer_mixin.py`, exported from
`phenotypic.sdk_` alongside `FootprintMixin` / `ClipControlMixin`. Reusable by design.

**Type alias** (`sdk_/typing_.py`, per the closed-value-set convention — `Literal`-only,
no `Enum` partner, since there is no separate documentation surface, mirroring
`DetectMode` and `ExecutionMode`):

```python
InputLayer = Literal["detect_mat", "rgb"]
```

```python
class InputLayerMixin(BaseModel):
    input_layer: InputLayer = "detect_mat"

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)   # cooperative: BaseOperation's runs too
        fields = cls.__pydantic_fields__
        if "input_layer" in fields and list(fields)[-1] != "input_layer":
            fields["input_layer"] = fields.pop("input_layer")
            cls.model_rebuild(force=True)
```

The reorder is the whole point: pydantic collects fields in reverse-MRO order, so a
mixin's field would normally be **frontloaded** ahead of the operation's own parameters.
Popping and reinserting moves it to the end of the insertion-ordered dict, and the forced
rebuild regenerates the validator and JSON schema from the new order.

### Helpers (deferred to the class, never applied upstream of `_operate`)

Per the design constraint, `apply()` is untouched — nothing reads `input_layer` before
`_operate` runs. The mixin exposes two helpers the op body calls itself:

```python
def _read_input_layer(self, image: Image) -> np.ndarray:
    """Return the source array: 2-D detect_mat, or 3-D float RGB in [0,1]."""

def _project_to_detect_mat(self, image: Image, arr: np.ndarray) -> np.ndarray:
    """Collapse a 3-D array to 2-D via the image's own detect_mode. 2-D passes through."""
```

`_read_input_layer` returns `image.rgb.normed()` cast to **float32** (the accessor's
`normalize_rgb_bitdepth` returns float64; on a 5000×5000 plate the cast halves a 600 MB
intermediate to 300 MB — the "be mindful of memory" rule).

An `input_layer="rgb"` read on a grayscale-only image raises `EmptyImageError` from the
`rgb` accessor. No extra guard is added; the accessor's error is already correct and
specific.

---

## 4. The RGB → `detect_mat` projection

`input_layer="rgb"` adjusts all three channels, then must collapse to 2-D. **The collapse
reuses the image's own `detect_mode`**, so `SetDetectMode(mode="MinRGB")` upstream means
the adjusted RGB is projected through `MinRGB`, not through a hardcoded luminance.

### 4.1 Why this needs a core change

Eight of the eleven registered modes are pure functions of an RGB array
(`gray`, `red`, `green`, `blue`, `MinRGB`, `HsvS`, `HsvV`, `InvS` — indexing, `np.min`,
`rgb2gray`, `rgb2hsv`). **Three are not.** `LabL`/`LabA`/`LabB` route through
`image.color.Lab` → `image.color.XYZ`, whose `_subject_arr` reads `image.gamma`,
`image.illuminant`, and `image._observer` and dispatches a four-arm `match` into
`colour.RGB_to_XYZ` with CCTF decoding on or off. There is no way to project an *adjusted*
RGB array through `LabA` without carrying that per-image color configuration.

### 4.2 The change

**(a)** Extract the `match` block out of
`_core/_image_parts/color_space_accessors/_xyz_accessor.py` into a new sibling module
`_xyz_conversion.py`:

```python
def rgb_to_xyz(rgb_normed: np.ndarray, *, gamma, illuminant, observer) -> np.ndarray:
```

`XYZAccessor._subject_arr` becomes a thin caller. Behavior is byte-identical; this is a
pure lift.

**(b)** Add to `DetectionMode`:

```python
@abstractmethod
def compute_from_rgb(self, rgb: np.ndarray, *, image: Image) -> np.ndarray:
    """Project a substitute RGB array (float, [0,1]) to a 2-D detect_mat.

    ``image`` supplies color configuration only (gamma / illuminant / observer);
    ``image._data.rgb`` is never read.
    """
```

Each mode implements it. The RGB-requiring modes then have `compute(image)` delegate:
`return self.compute_from_rgb(normalize_rgb_bitdepth(image._data.rgb), image=image)`.

**`GrayDetectionMode` is deliberately exempt from delegation.** Its `requires_rgb` is
`False` and `compute()` must keep returning `image._data.gray.copy()` so grayscale-only
images still work. Its `compute_from_rgb` is `rgb2gray(rgb)`. The two agree because
`_set_from_rgb` derives gray as exactly `rgb2gray(rgb_array)`.

### 4.3 The golden test that pins it

For every one of the 11 registered modes, on an RGB image:

```python
assert_allclose(
    mode.compute(image),
    mode.compute_from_rgb(normalize_rgb_bitdepth(image.rgb[:]), image=image),
    atol=1e-6,
)
```

**Tolerance derivation (mechanism, not guess):** the two paths differ only by float
accumulation order in a 3-term dot product (`rgb2gray`) or a 3×3 matmul (`RGB_to_XYZ`),
computed in float64 and cast to float32. float32 eps is `1.19e-7`; three fused operations
bound the discrepancy at `~3 × 1.19e-7 ≈ 3.6e-7`. `atol=1e-6` sits ~3× above that — loose
enough to survive reassociation, tight enough that a genuine channel swap or a dropped
CCTF decode (which move values by `>1e-2`) fails the test.

**The test must be able to fail.** Before trusting it, reintroduce the bug it guards:
make `LabAMode.compute_from_rgb` skip `apply_cctf_decoding` and confirm the assertion
trips.

---

## 5. Output range policy: `out_range`

Because §2.3 shows `clip` is a poisoned identifier (`_disable_clipping` duck-types on it),
the new ops use a **closed value set** instead of a bool:

```python
OutRange = Literal["clip", "rescale", "passthrough"]   # sdk_/typing_.py
```

| Value | Behavior | When |
|---|---|---|
| `"clip"` *(default)* | `np.clip(out, 0.0, 1.0)` | Normal use. `gain` is meaningful (§2.1). Preserves absolute intensity, so `detect_mat` stays comparable across a batch. |
| `"rescale"` | `rescale_intensity(out, out_range=(0,1))` | Full-histogram normalization. Absorbs `gain` on gamma/log — documented, not hidden. |
| `"passthrough"` | `out` unchanged | GAT regions and `CompositeEnhance` on non-normalized maps. |

Both `"clip"` and `"rescale"` uphold the `detect_mat ∈ [0,1]` contract; they differ only in
saturate-versus-compress. `"passthrough"` is the explicit, named escape hatch.

Being a `Literal`, `out_range` is excluded from the tune coverage gate
(`is_numeric_tunable` skips `Literal`, `Enum`, and `bool`).

**Required follow-through:** `ClipControlMixin._disable_clipping` must learn to map
`out_range → "passthrough"` on ops that expose it. Without this, nesting a `ContrastGamma`
inside `CompositeEnhance` silently keeps clipping. This is a real behavior change to a
shared mixin and needs its own test.

### 5.1 Defensive input normalization

The three skimage functions raise `ValueError` on negative input (§2.4), which
`FocusEdgeLaplace` can produce. When the input array falls outside `[0,1]`, the op
**rescales it to `[0,1]` before applying the curve**.

Two constraints on this rule:

- It is **skipped when `out_range="passthrough"`.** Otherwise a GAT-stabilized input at
  `[1.92, 2.31]` would be silently normalized on the way *in*, reintroducing the exact
  corruption §2.3 exists to prevent.
- It is **conditional**, triggering only on an actual out-of-range input, so the common
  path costs one `min()`/`max()` and nothing else.

The docstrings must state plainly that the curve is then applied to a *shifted* signal, so
results downstream of a signed filter depend on that filter's output range.

---

## 6. The four operations

All subclass the **`ContrastAdjustment`** purpose-group marker (not `ImageEnhancer`
directly), per `enhance/CLAUDE.md`. Mixin goes first in the bases, matching the
`class MaskDilation(FootprintMixin, ObjectRefiner)` house style.

```python
class ContrastGamma(InputLayerMixin, ContrastAdjustment):
    gamma:  Annotated[float, TuneSpec(0.1, 5.0, log=True)] = 1.0
    gain:   Annotated[float, TuneSpec(0.5, 2.0)]           = 1.0
    out_range: OutRange = "clip"
    # input_layer: InputLayer = "detect_mat"   <- appended by the mixin

class ContrastLog(InputLayerMixin, ContrastAdjustment):
    gain:   Annotated[float, TuneSpec(0.5, 2.0)] = 1.0
    inv:    bool                                 = False
    out_range: OutRange = "clip"

class ContrastSigmoid(InputLayerMixin, ContrastAdjustment):
    cutoff: Annotated[float, TuneSpec(0.0, 1.0)]  = 0.5
    gain:   Annotated[float, TuneSpec(1.0, 20.0)] = 10.0
    inv:    bool                                  = False
    out_range: OutRange = "clip"
```

`cutoff=0.5` / `gain=10.0` are skimage's own defaults. `inv` and `out_range` are `bool` /
`Literal` and therefore outside the tune coverage gate; `gamma`, `gain`, and `cutoff` carry
`TuneSpec` windows and satisfy it.

Canonical `_operate`, identical in shape across the three:

```python
def _operate(self, image: Image) -> Image:
    src = self._read_input_layer(image)             # 2-D detect_mat or 3-D rgb
    src = self._guard_input_range(src)              # §5.1
    adj = adjust_gamma(src, gamma=self.gamma, gain=self.gain)
    out = self._project_to_detect_mat(image, adj)   # 3-D -> 2-D via detect_mode
    image.detect_mat[:] = self._apply_out_range(out).astype(np.float32)
    return image
```

### 6.1 `ContrastStretching` retrofit

```python
class ContrastStretching(InputLayerMixin, ContrastAdjustment):
    lower_percentile: Annotated[int, TuneSpec(1, 5)]   = 2
    upper_percentile: Annotated[int, TuneSpec(95, 99)] = 98
    per_channel: bool = False
    # input_layer appended by the mixin
```

`ContrastStretching` keeps `rescale_intensity` and gains **no** `out_range` — rescaling
between percentiles *is* its algorithm (§2.2), and its output is `[0,1]` by construction.

`per_channel` (§2.5) is meaningful only when `input_layer="rgb"`; it is documented as
ignored for 2-D input.

- `per_channel=False` *(default)*: one `(p_lo, p_hi)` from the flattened `H×W×3` array,
  applied to all three channels. Preserves channel balance, so a hue-sensitive downstream
  `detect_mode` (`MinRGB`, `HsvS`, `InvS`, `LabA`) sees the plate's true color. Matches
  skimage and GIMP's "Keep colors".
- `per_channel=True`: independent `(p_lo, p_hi)` per channel. Effectively a per-channel
  white balance; removes color casts, shifts hue. Matches Photoshop's "Enhance Per Channel
  Contrast" and earthpy's per-band convention.

Its parameters are statistics *over the input*, which is why it alone needs this flag —
the other three ops are pointwise and have nothing to compute a statistic over.

---

## 7. The `[0,1]` invariant

Per §2.4 the contract already holds for 29/30 enhancers. This spec closes the gap:

1. **Normalize `FocusEdgeLaplace`'s output** into `[0,1]`. Its Laplacian response is signed
   by construction, so this changes its output for existing users and pipelines — a
   deliberate, breaking-ish correction, called out in the changelog.
2. **Add a CI gate**: every enhancer in `enhance.__all__` that constructs with no required
   args must leave `detect_mat` within `[0, 1]` after `apply()` on the synth plate. This
   would have caught the violation, and stops the contract from rotting.

The gate must be **able to fail**: verified by reverting the `FocusEdgeLaplace` fix and
confirming the new test goes red before the fix is reapplied.

---

## 8. Semantics documented, not enforced

`input_layer="rgb"` reads the **pristine** `rgb` layer, so it **discards any enhancement a
prior op wrote to `detect_mat`** — the same footgun `SetDetectMode` already carries
(*"Prior enhancements to `detect_mat` are discarded."*). This is documented in each
docstring's `Returns:` block. No runtime check: detecting it means comparing `detect_mat`
against a fresh reset, a full-array comparison per `apply()` on large plate images, for a
lint-grade message.

---

## 9. Files

**New**

- `src/phenotypic/sdk_/mixin/_input_layer_mixin.py`
- `src/phenotypic/enhance/_contrast_gamma.py`
- `src/phenotypic/enhance/_contrast_log.py`
- `src/phenotypic/enhance/_contrast_sigmoid.py`
- `src/phenotypic/_core/_image_parts/color_space_accessors/_xyz_conversion.py`

**Modified**

- `sdk_/typing_.py` — `InputLayer`, `OutRange` aliases
- `sdk_/mixin/__init__.py`, `sdk_/__init__.py` — export `InputLayerMixin`
- `sdk_/mixin/_clip_control_mixin.py` — honor `out_range` (§5)
- `_core/_image_parts/detection_modes/_detection_mode.py` — `+compute_from_rgb`
- `detection_modes/{_gray,_color_channel,_min_rgb,_lab_channel,_hsv_channel,_inv_saturation}*.py`
- `color_space_accessors/_xyz_accessor.py` — thin caller
- `enhance/_contrast_streching.py` — `input_layer` + `per_channel`
- `enhance/_focus_edge_laplace.py` — invariant fix
- `enhance/__init__.py` — three new exports
- `enhance/CLAUDE.md`, `sdk_/CLAUDE.md` — mixin + `out_range` conventions

**Tests**

- `tests/unit/abc_/test_enhancer_taxonomy.py` — roster `+3` under `ContrastAdjustment`
- `tests/unit/tune/test_enhance_annotations.py` — new `TuneSpec` windows
- `tests/unit/sdk_/test_input_layer_mixin.py` — field appended last across
  `model_fields` / schema / `to_json`; `TuneSpec` survives rebuild; no leak
- `tests/unit/core/test_detection_modes_from_rgb.py` — the §4.3 golden equivalence, ×11
- `tests/unit/enhance/test_contrast_ops.py` — curve correctness, both input layers,
  all three `out_range` values, negative-input guard, `per_channel` joint-vs-split
- `tests/unit/enhance/test_detect_mat_invariant.py` — the §7 gate

GUI builder registry and Sphinx `autosummary` both walk `phenotypic.enhance`
automatically; no manual registration or API-doc edit is required.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| The `rgb_to_xyz` lift silently changes color output | Pure extraction, no logic edit; pinned by the §4.3 golden test across all 11 modes, mutation-checked by reintroducing a dropped CCTF decode. |
| `model_rebuild(force=True)` drops `TuneSpec` metadata | **Already disproved** (§2.6) — metadata survives. Re-asserted in `test_input_layer_mixin.py`. |
| `_disable_clipping` silently no-ops on the new ops | §5 makes updating it a required deliverable with its own test. |
| `FocusEdgeLaplace` output change breaks a downstream pipeline | Called out in the changelog; the §7 gate makes the new contract explicit. |
| `rgb` path doubles peak memory on large plates | float32 cast in `_read_input_layer` (§3); the 3-D intermediate is transient and freed before the `detect_mat` write. |

---

## 11. Open items for review

1. `per_channel` on `ContrastStretching` was **derived** from the "if both methods apply
   then use the flag" rule plus the §2.5 research, not chosen explicitly. Confirm the name
   (`per_channel=False`) over GIMP's inverted `keep_colors=True`.
2. `out_range` is a third field on each op. If `"passthrough"` is judged unnecessary — the
   `Contrast*` ops are not GAT participants — it can collapse to `Literal["clip","rescale"]`
   and `_disable_clipping` needs no change.
3. §7's `FocusEdgeLaplace` normalization changes existing behavior. Confirm that is
   acceptable on this branch rather than a separate one.
