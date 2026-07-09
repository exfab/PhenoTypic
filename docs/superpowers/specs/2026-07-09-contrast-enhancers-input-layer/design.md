# Design: `ContrastGamma` / `ContrastLog` / `ContrastSigmoid` + the `input_layer` mixin

- **Date:** 2026-07-09
- **Status:** Design — all open questions resolved (§12); ready for planning
- **Target release:** `0.18.0` (three breaking changes — see §11)
- **Branch / worktree:** `contrast-enh`
- **Scope:** three new `ContrastAdjustment` enhancers wrapping skimage's
  `adjust_gamma` / `adjust_log` / `adjust_sigmoid`; a reusable `InputLayerMixin`
  that **appends** an `input_layer` field to a pydantic operation; retrofitting
  `ContrastStretching` with the same capability; a `compute_from_rgb` path on
  `DetectionMode`; retiring `clip: bool` in favour of `norm: NormOut` repo-wide;
  and closing the one remaining `detect_mat ∈ [0,1]` violation.
- **Breaking:** `clip` → `norm` (serialization + kwargs), `ClipControlMixin` →
  `NormControlMixin`, `FocusEdgeLaplace` output range. See §5.2, §10.

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
| New mixin | `NormalizedOutputMixin` → appends `norm: NormOut` |
| Retrofit | `ContrastStretching` gains `input_layer` + `keep_colors` |
| Refactor | `clip: bool` → `norm: NormOut` across 8 classes; `ClipControlMixin` → `NormControlMixin`; `_GAT_DEFER_ATTRS` → `_GAT_DEFER_VALUES` |
| New core API | `DetectionMode.compute_from_rgb(rgb, *, image)` on all 11 modes |
| Extraction | `rgb_to_xyz(...)` free function, lifted out of `XYZAccessor` |
| Invariant | `FocusEdgeLaplace` normalized; CI gate pins `detect_mat ∈ [0,1]` |
| Convention | `adding-an-operation` skill documents the `norm` + append-mixin rules |

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
percentiles *is* the algorithm.

**Eight classes expose the guard as a `clip: bool` field** — five enhancers
(`BayesShrinkEnhancer`, `EnhanceBlockMatch`, `LocalEdgeDenoise`, `VisuShrinkEnhancer`,
and `CompositeEnhance`, which alone defaults it to `False`) and three correctors
(`_color_denoise`, `_visushrink_corrector`, `_bayesshrink_corrector`). In **every** case
the flag is consumed by us as `if self.clip: np.clip(...)` and is **never forwarded to
skimage** — which is what makes folding them into a single `norm` field (§5)
behavior-preserving.

`rescale_sigma: bool` is **not** one of these. It is forwarded to skimage's
`denoise_wavelet(rescale_sigma=)` and governs whether skimage rescales the *noise sigma*
when it internally converts integer dtypes to float. Its own docstring notes it *"has no
observable effect on the float32 `detect_mat` used in this project."* It is unrelated to
output range and is left alone.

Semantically the two differ in kind. Clip is the identity on in-range pixels and only
touches the tails, preserving absolute intensity. Rescale preserves ordering but
discards absolute scale and makes the output a function of the input's **extremes** —
one specular highlight sets the max, and two plates in one batch receive two different
mappings. That is exactly why `ContrastStretching` rescales between *percentiles*
rather than between min and max.

### 2.3 `clip=False` already means "leave the values alone" — and it is load-bearing

`_GATSupportMixin._GAT_DEFER_ATTRS` is documented as *"Boolean attributes that must be
`False` inside the GAT region … any knob that would corrupt the stabilized
round-trip."* **Six classes list `"clip"` there:** `VisuShrinkEnhancer`,
`BayesShrinkEnhancer`, `LocalEdgeDenoise`, `EnhanceBlockMatch`, `_visushrink_corrector`,
and `_bayesshrink_corrector`. `ClipControlMixin._disable_clipping` sets the same flag, and
**duck-types on any op exposing a `.clip` attribute.**

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
`ContrastStretching` gets `keep_colors: bool = True` (GIMP's name). See §6.1.

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
`phenotypic.sdk_` alongside `FootprintMixin` / `NormControlMixin`. Reusable by design.

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

## 5. Output normalization: the `norm` field (repo-wide)

`clip: bool` conflates two questions — *should the output be normalized?* and *how?* —
and it cannot express "rescale". §2.3 also shows the identifier is spoken for:
`_disable_clipping` duck-types on `.clip`. Both problems are solved by one closed value
set, and the `clip: bool` flag is **retired repo-wide**.

```python
# sdk_/typing_.py — Literal alias, no Enum partner (no separate documentation
# surface), mirroring DetectMode / ExecutionMode.
NormOut: TypeAlias = Optional[Literal["clip", "rescale"]]
```

| `norm` | Behavior | When |
|---|---|---|
| `"clip"` *(default)* | `np.clip(out, 0.0, 1.0)` | Normal use. `gain` stays meaningful (§2.1). Preserves absolute intensity, so `detect_mat` is comparable across a batch. |
| `"rescale"` | `rescale_intensity(out, out_range=(0,1))` | Full-histogram normalization. Absorbs `gain` on gamma/log — documented, not hidden. |
| `None` | `out` unchanged | GAT regions and `CompositeEnhance` on non-normalized maps. |

Both `"clip"` and `"rescale"` uphold `detect_mat ∈ [0,1]`; they differ only in
saturate-versus-compress. `None` is the explicit pass-through.

`NormOut` is `Optional[Literal[...]]`, so it is excluded from the tune coverage gate
(`is_numeric_tunable` returns `False` for `MULTI_UNION`, `Literal`, `Enum`, and `bool`).

### 5.1 `NormalizedOutputMixin`

The eight `if self.clip: np.clip(...)` sites collapse into one helper. A second field-append
mixin, alongside `InputLayerMixin` in `sdk_/mixin/`:

```python
class NormalizedOutputMixin(BaseModel):
    norm: NormOut = "clip"

    def _apply_norm(self, arr: np.ndarray) -> np.ndarray: ...
```

`CompositeEnhance` overrides the default to `None`, preserving today's `clip: bool = False`.

**Field order with two append-mixins is deterministic.** Each mixin's
`__pydantic_init_subclass__` calls `super()` *first*, then pops its own field to the end.
With MRO `ContrastGamma → InputLayerMixin → NormalizedOutputMixin → ContrastAdjustment`,
`NormalizedOutputMixin`'s pop runs inside the `super()` call and therefore **first**,
leaving the final order `[…op params…, norm, input_layer]`. This ordering is pinned by a
test, not left to inference.

### 5.2 Migration (8 classes) — hard break with an explicit error

`enhance/`: `LocalEdgeDenoise`, `BayesShrinkEnhancer`, `EnhanceBlockMatch`,
`VisuShrinkEnhancer`, `CompositeEnhance`.
`correction/`: `_color_denoise`, `_visushrink_corrector`, `_bayesshrink_corrector`.

Mapping: `clip=True → norm="clip"`, `clip=False → norm=None`. `rescale_sigma` untouched.

Because `BaseOperation.model_config` sets `extra="forbid"`, a bare rename makes
`from_json` on any saved pipeline fail with an opaque *"Extra inputs are not permitted"*
naming `clip`. Verified against `tests/fixtures/tune/back_compat_pipelines/`, which pins
deserialization of old configs and contains `"clip": true`.

So the mixin carries a before-validator that turns the schema complaint into a migration
message:

```python
@model_validator(mode="before")
@classmethod
def _reject_legacy_clip(cls, data):
    if isinstance(data, dict) and "clip" in data:
        raise ValueError(
            f"{cls.__name__}: `clip` was replaced by `norm` in <version>. "
            f"Use norm='clip' (was clip=True) or norm=None (was clip=False)."
        )
    return data
```

Both `back_compat_pipelines` fixtures are regenerated with `norm`, and a test asserts the
legacy key raises the migration message rather than a schema error.

### 5.3 Two dependent mechanisms

**`_GATSupportMixin`.** `_GAT_DEFER_ATTRS: tuple[str, ...]` becomes
`_GAT_DEFER_VALUES: ClassVar[dict[str, Any]]` mapping each attribute to its inert value —
`{"norm": None, "rescale_sigma": False}`. The tuple form cannot express this, since `norm`'s
inert value is `None` while `rescale_sigma`'s is `False`. The docstring stops saying
"Boolean attributes". Six classes update; `NonLocalMeansDenoiser`'s empty tuple becomes `{}`.

**`ClipControlMixin` → `NormControlMixin`.** `_disable_clipping` → `_disable_normalization`,
setting `norm=None`. This is a **public API rename** (both are exported from
`phenotypic.sdk_`), requiring updates to `sdk_/__init__.py`, `sdk_/mixin/__init__.py`, and
`sdk_/CLAUDE.md`. Without it, nesting a `ContrastGamma` inside `CompositeEnhance` would
silently keep normalizing.

### 5.4 Defensive input normalization

The three skimage functions raise `ValueError` on negative input (§2.4), which
`FocusEdgeLaplace` can produce. When the input array falls outside `[0,1]`, the op
**rescales it to `[0,1]` before applying the curve**.

Two constraints:

- **Skipped when `norm is None`.** Otherwise a GAT-stabilized input at `[1.92, 2.31]` would
  be silently normalized on the way *in*, reintroducing the exact corruption §2.3 exists to
  prevent.
- **Conditional**, triggering only on genuinely out-of-range input, so the common path costs
  one `min()`/`max()` and nothing else.

The docstrings must state plainly that the curve is then applied to a *shifted* signal, so
results downstream of a signed filter depend on that filter's output range.

---

## 6. The four operations

All subclass the **`ContrastAdjustment`** purpose-group marker (not `ImageEnhancer`
directly), per `enhance/CLAUDE.md`. Mixin goes first in the bases, matching the
`class MaskDilation(FootprintMixin, ObjectRefiner)` house style.

```python
class ContrastGamma(InputLayerMixin, NormalizedOutputMixin, ContrastAdjustment):
    gamma:  Annotated[float, TuneSpec(0.1, 5.0, log=True)] = 1.0
    gain:   Annotated[float, TuneSpec(0.5, 2.0)]           = 1.0
    # norm: NormOut = "clip"                   <- appended by NormalizedOutputMixin
    # input_layer: InputLayer = "detect_mat"   <- appended by InputLayerMixin (last)

class ContrastLog(InputLayerMixin, NormalizedOutputMixin, ContrastAdjustment):
    gain:   Annotated[float, TuneSpec(0.5, 2.0)] = 1.0
    inv:    bool                                 = False

class ContrastSigmoid(InputLayerMixin, NormalizedOutputMixin, ContrastAdjustment):
    cutoff: Annotated[float, TuneSpec(0.0, 1.0)]  = 0.5
    gain:   Annotated[float, TuneSpec(1.0, 20.0)] = 10.0
    inv:    bool                                  = False
```

`cutoff=0.5` / `gain=10.0` are skimage's own defaults. `inv` and `norm` are `bool` /
`Optional[Literal]` and therefore outside the tune coverage gate; `gamma`, `gain`, and
`cutoff` carry `TuneSpec` windows and satisfy it.

Canonical `_operate`, identical in shape across the three:

```python
def _operate(self, image: Image) -> Image:
    src = self._read_input_layer(image)             # 2-D detect_mat or 3-D rgb
    src = self._guard_input_range(src)              # §5.4
    adj = adjust_gamma(src, gamma=self.gamma, gain=self.gain)
    out = self._project_to_detect_mat(image, adj)   # 3-D -> 2-D via detect_mode
    image.detect_mat[:] = self._apply_norm(out).astype(np.float32)
    return image
```

### 6.1 `ContrastStretching` retrofit

```python
class ContrastStretching(InputLayerMixin, ContrastAdjustment):
    lower_percentile: Annotated[int, TuneSpec(1, 5)]   = 2
    upper_percentile: Annotated[int, TuneSpec(95, 99)] = 98
    keep_colors: bool = True
    # input_layer appended by the mixin
```

`ContrastStretching` keeps `rescale_intensity` and gains **no** `norm` field — rescaling
between percentiles *is* its algorithm (§2.2), and its output is `[0,1]` by construction.
A `norm` here could only mean "do it twice" (`"clip"`, a no-op) or "undo the op" (`None`),
so it is deliberately absent rather than accepted for family symmetry. The docstring states
plainly that the output is always `[0,1]` and no `norm` field is offered.

`keep_colors` (§2.5) is meaningful only when `input_layer="rgb"`; it is documented as
ignored for 2-D input. The name is taken verbatim from GIMP's "Stretch Contrast" checkbox,
and names the **invariant it protects** (channel balance / hue) rather than the mechanism.

- `keep_colors=True` *(default)*: one `(p_lo, p_hi)` from the flattened `H×W×3` array,
  applied to all three channels — GIMP's *"Impact each color channel with the same amount."*
  Preserves channel balance, so a hue-sensitive downstream `detect_mode` (`MinRGB`, `HsvS`,
  `InvS`, `LabA`) sees the plate's true color. Matches skimage's `rescale_intensity`.
- `keep_colors=False`: independent `(p_lo, p_hi)` per channel. Effectively a per-channel
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
- `src/phenotypic/sdk_/mixin/_normalized_output_mixin.py`
- `src/phenotypic/enhance/_contrast_gamma.py`
- `src/phenotypic/enhance/_contrast_log.py`
- `src/phenotypic/enhance/_contrast_sigmoid.py`
- `src/phenotypic/_core/_image_parts/color_space_accessors/_xyz_conversion.py`

**Modified — new capability**

- `sdk_/typing_.py` — `InputLayer`, `NormOut` aliases
- `sdk_/mixin/__init__.py`, `sdk_/__init__.py` — export both mixins
- `_core/_image_parts/detection_modes/_detection_mode.py` — `+compute_from_rgb`
- `detection_modes/{_gray,_color_channel,_min_rgb,_lab_channel,_hsv_channel,_inv_saturation}*.py`
- `color_space_accessors/_xyz_accessor.py` — thin caller
- `enhance/_contrast_streching.py` — `input_layer` + `keep_colors`
- `enhance/_focus_edge_laplace.py` — invariant fix (§7)
- `enhance/__init__.py` — three new exports

**Modified — `clip` → `norm` migration (§5.2)**

- `sdk_/mixin/_clip_control_mixin.py` → `_norm_control_mixin.py` (public rename)
- `sdk_/mixin/_gat_support_mixin.py` — `_GAT_DEFER_ATTRS` → `_GAT_DEFER_VALUES`
- `enhance/{_local_edge_denoise,_bayesshrink_enhancer,_enhance_block_match,`
  `_visushrink_enhancer,_composite_enhance}.py`
- `correction/{_color_denoise,_visushrink_corrector,_bayesshrink_corrector}.py`
- `prefab/_heavy_round_peaks_pipeline.py` — passes `clip=bm3d_clip`
- `tests/fixtures/tune/back_compat_pipelines/*.json` — regenerate with `norm`
- `src/phenotypic/__init__.py` — `__version__` → `"0.18.0"` (§11)

**Modified — conventions**

- `enhance/CLAUDE.md`, `sdk_/CLAUDE.md` — mixins + `norm` convention
- `.claude/skills/adding-an-operation/SKILL.md` — `norm: NormOut` for any op with an
  output-range guard; the append-mixin pattern for cross-cutting fields

**Tests**

- `tests/unit/abc_/test_enhancer_taxonomy.py` — roster `+3` under `ContrastAdjustment`
- `tests/unit/tune/test_enhance_annotations.py` — new `TuneSpec` windows
- `tests/unit/sdk_/test_input_layer_mixin.py` — field appended last across
  `model_fields` / schema / `to_json`; `TuneSpec` survives rebuild; no leak; **and the
  two-mixin order `[…, norm, input_layer]`** (§5.1)
- `tests/unit/sdk_/test_norm_migration.py` — legacy `clip` key raises the migration
  message; `norm=None` round-trips; `_disable_normalization` sets `norm=None`
- `tests/unit/sdk_/mixin/test_gat_support_mixin.py` — update for `_GAT_DEFER_VALUES`;
  assert the GAT round-trip still holds with `norm=None` deferred
- `tests/unit/core/test_detection_modes_from_rgb.py` — the §4.3 golden equivalence, ×11
- `tests/unit/enhance/test_contrast_ops.py` — curve correctness, both input layers,
  all three `norm` values, negative-input guard, `keep_colors` joint-vs-split
- `tests/unit/enhance/test_detect_mat_invariant.py` — the §7 gate

Existing call sites passing `clip=` (`tests/unit/correction/test_color_denoise.py`,
`tests/unit/enhance/test_composite_enhance.py`, `tests/unit/sdk_/mixin/test_gat_support_mixin.py`)
must be updated to `norm=`.

GUI builder registry and Sphinx `autosummary` both walk `phenotypic.enhance`
automatically; no manual registration or API-doc edit is required.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| The `rgb_to_xyz` lift silently changes color output | Pure extraction, no logic edit; pinned by the §4.3 golden test across all 11 modes, mutation-checked by reintroducing a dropped CCTF decode. |
| `model_rebuild(force=True)` drops `TuneSpec` metadata | **Already disproved** (§2.6) — metadata survives. Re-asserted in `test_input_layer_mixin.py`. |
| **`clip` → `norm` breaks users' saved pipelines** | Unavoidable by choice (hard break). The §5.2 before-validator converts the opaque `extra="forbid"` schema error into a message naming the rename and the replacement value. Needs a changelog entry and a version bump note. |
| `_GAT_DEFER_VALUES` migration corrupts a GAT round-trip | The stabilized-domain round-trip is asserted directly in `test_gat_support_mixin.py`; §2.3 shows the failure mode is loud (output collapses to zeros), not subtle. |
| `NormControlMixin` rename breaks a downstream import | Public API rename, exported from `phenotypic.sdk_`. Same changelog entry; no deprecation shim, per the hard-break decision. |
| `_disable_normalization` silently no-ops on the new ops | §5.3 makes retargeting it to `.norm` a required deliverable with its own test. |
| `FocusEdgeLaplace` output change breaks a downstream pipeline | Called out in the changelog; the §7 gate makes the new contract explicit. |
| `rgb` path doubles peak memory on large plates | float32 cast in `_read_input_layer` (§3); the 3-D intermediate is transient and freed before the `detect_mat` write. |

---

## 11. Release: `0.18.0`

The branch carries **three** breaking changes, all landing together:

| Break | Surface |
|---|---|
| `clip: bool` → `norm: NormOut` | Pipeline JSON/YAML; `op(clip=...)` kwargs |
| `ClipControlMixin` → `NormControlMixin` | `phenotypic.sdk_` public export |
| `FocusEdgeLaplace` output normalized | `detect_mat` values for existing pipelines |

Pre-1.0 SemVer permits these in a minor bump, so `src/phenotypic/__init__.py` moves
`__version__` to `"0.18.0"`.

**The repo has no CHANGELOG file** (verified: no `CHANGELOG*` at root, nothing matching
`*changelog*` / `*release*` / `*whatsnew*` under `docs/`). So the user-facing signal is
exactly two things, and both must work:

1. `SerializablePipeline.from_json` **already** warns on any version mismatch
   (`_serializable_pipeline.py:359`) — *"Pipeline was saved with phenotypic version
   0.17.3 but current version is 0.18.0."* This fires first, for free.
2. The §5.2 before-validator then raises an actionable error naming the rename, rather
   than pydantic's opaque *"Extra inputs are not permitted."*

The migration message must name `0.18.0` explicitly. Starting a `CHANGELOG.md` is worth
doing but is **out of scope** for this branch — flagged, not silently adopted.

---

## 12. Resolved design questions

Recorded so the plan does not relitigate them.

| Question | Decision | Rationale |
|---|---|---|
| RGB → 2-D collapse | Project through the image's own `detect_mode` | Respects `SetDetectMode`; forces the §4.2 `rgb_to_xyz` extraction |
| Output range | `norm: NormOut` (`"clip"` default) | `rescale` provably annihilates `gain` (§2.1); `clip` is house style (§2.2) |
| `clip=False` reuse | Rejected — `norm=None` instead | `clip=False` means "leave alone" and is load-bearing for GAT (§2.3) |
| Migration scope | All 8 `clip: bool` classes | One spelling repo-wide; `rescale_sigma` untouched |
| Back-compat | Hard break + explicit migration error | `extra="forbid"` makes a bare rename opaque (§5.2) |
| GAT defer | `_GAT_DEFER_VALUES` dict | Tuple cannot express `norm→None` alongside `rescale_sigma→False` |
| Mixin home | `sdk_/mixin/`, reusable | Mirrors `FootprintMixin`; future ops inherit `input_layer` free |
| Negative input | Defensive pre-rescale, skipped when `norm is None` | Curves raise `ValueError`; `FocusEdgeLaplace` emits negatives (§2.4) |
| Stretch flag | `keep_colors: bool = True` | Both methods are established practice (§2.5); GIMP's name, names the invariant |
| `norm` on `ContrastStretching` | Absent, deliberately | Percentile rescale *is* the algorithm; `norm` could only no-op or undo it |
| `FocusEdgeLaplace` fix | This branch, with the CI gate | Keeps `detect_mat ∈ [0,1]` true and enforced in one change |
| Release | `0.18.0` minor bump | Pre-1.0; existing version-mismatch warning already primes users |
| `discard` semantics | Documented, not enforced | Matches `SetDetectMode`; a runtime check costs a full-array compare per `apply()` |

### Still deferred

- `.claude/skills/adding-an-operation/SKILL.md` gains the `norm` + append-mixin convention
  **in the implementation commit**, not before — documenting `NormOut` while it does not
  exist would send agents after an unimportable symbol.
- `CHANGELOG.md` does not exist; creating one is a separate change.
