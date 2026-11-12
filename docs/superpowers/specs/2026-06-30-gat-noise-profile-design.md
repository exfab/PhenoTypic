# GATNoiseProfile — Poisson–Gaussian noise parameter estimator

**Date:** 2026-06-30
**Status:** Design (approved for planning)
**Branch:** `worktree-estimate-gat-params`

## 1. Problem & motivation

The repo already *applies* the Generalized Anscombe Transform (GAT) for
Poisson–Gaussian noise: `gat_forward`/`gat_inverse` live in
`src/phenotypic/sdk_/_anscombe.py` and are wired into GAT-aware denoisers via
`_GATSupportMixin` (`VisuShrinkEnhancer`, `BayesShrinkEnhancer`,
`EnhanceBlockMatch`) and directly in the correctors `DenoiseBlockMatch`
(grayscale BM3D) and `ColorDenoise` (CBM3D on RGB).

What is missing is an **estimator** for the three noise-model parameters. Today
`gat_gain`, `gat_mu`, `gat_read_sigma` default to neutral values (1.0 / 0.0 /
0.0), i.e. the plain Anscombe transform — the "blank estimator". Users must
hand-set them. This spec adds a fitted **profile** that estimates them from one
or many images, so the GAT can be configured from data instead of guessed, and
so the Optuna tune module starts from calibrated values rather than blanks.

## 2. Scope

**In scope**

- `GATNoiseProfile` — a fitted profile class that estimates
  `(gain, mu, read_sigma)` from a single image or a pooled set of images, for
  three channel modes: `gray`, `detect_mat` (both scalar / single-channel), and
  `rgb` (per-channel, three parameter sets).
- Extending `ColorDenoise` (CBM3D) to accept **per-channel** GAT parameters so
  the `rgb` profile is consumable end-to-end.

**Out of scope (future work)**

- Cumulant / PoGaIN-style blind μ estimation (μ without a dark region).
- Per-image `None`-sentinel auto-estimation inside `_GATSupportMixin`.
- Automatic dark-region discovery.
- Moving `ColorCheckerProfile` into `util/` — deferred to a **subsequent PR**
  due to large blast radius. `GATNoiseProfile` establishes the util-profile
  precedent this move will follow.

## 3. Background & method (literature)

The canonical single-image noise estimator (Foi et al., 2008, *IEEE TIP*
17(10); DOI 10.1109/TIP.2008.2001399) fits a **two-parameter** noise-level
function on (local mean, local variance) pairs taken from weakly-textured
regions:

```
Var(x) = a·E[x] + b
```

For the repo's noise model `x = gain·p + n`, `p ~ Poisson`, `n ~ N(μ, σ²)`:

```
Var(x) = gain·(E[x] − μ) + σ²  =  gain·E[x] + (σ² − gain·μ)
```

So **slope `a` = gain** (μ-independent) and **intercept `b` = σ² − gain·μ**.

**Identifiability limit.** μ and σ both live in the single intercept — one
equation, two unknowns. Pooling more images adds points along the *same line*
and does not separate them. The literature treats μ (the pedestal / black
level) as a **known** hardware constant (dark-frame / optical-black), not a
free single-image parameter; Foi explicitly notes `b` can go negative *because
of* the pedestal. There is no canonical blind single-image method that isolates
μ from σ.

**μ policy (this design).** μ is recovered only from an explicit dark anchor a
caller supplies (`dark_roi` or `dark_mask`) — a region known to be genuinely
unilluminated (photons ≈ 0), where `E[x] → μ` and `Var → σ²`. When no anchor is
supplied, μ falls back to the `mu_prior` hyperparameter (default 0.0, the
dark-subtracted assumption) and is flagged low-confidence. Auto-discovery is
deliberately excluded: these plates have no genuine photon≈0 zone, so the
darkest *illuminated* agar would only bound μ from above and mislead.

**Per-channel RGB / CBM3D.** The Poisson–Gaussian model is valid in
**linear light**, per channel; white balance gives R/G/B different gains, so a
single scalar GAT across RGB is suboptimal. CBM3D (Dabov et al., 2007 ICIP;
DOI 10.1109/ICIP.2007.4378954) denoises in an orthonormal opponent
(luminance–chrominance) space, block-matching once in luminance and reusing the
grouping for chrominance; its transform-domain noise level is valid only when
the input noise is white and equal-variance across channels. Per-channel VST
maps each channel to ≈ N(0,1); an orthonormal opponent transform of three
unit-variance white channels stays unit-variance and white, so CBM3D with
`sigma_psd=1.0` is theoretically correct (Mäkitalo & Foi, 2013 *IEEE TIP*
22(1); DOI 10.1109/TIP.2012.2202675, generalized per-channel). The fixed order:

```
decode_srgb → per-channel forward GAT → CBM3D(opp, σ=1) → per-channel inverse GAT → encode_srgb
```

This is exactly `ColorDenoise._denoise_gat` today, except it applies scalar
params; this design makes those params per-channel.

## 4. Public API

Location: `src/phenotypic/util/_gat_noise_profile.py`, exported from
`phenotypic.util`. Pydantic `BaseModel`, **outside** the ImageOperation /
pipeline framework, mirroring `ColorCheckerProfile`'s fit/serialize/dashboard
shape.

```python
class GATNoiseProfile(BaseModel):
    # --- hyperparameters (constructor args, sklearn: no trailing underscore) ---
    channel: Literal["gray", "detect_mat", "rgb"] = "gray"
    patch_size: int = 8                 # local window side for mean/variance
    texture_percentile: float = 0.25    # keep flattest fraction of windows
    saturation_frac: float = 0.98       # drop windows with pixels >= frac*max
    min_patches: int = 50               # fewer surviving windows -> fit fails
    fit_method: Literal["theilsen", "ransac"] = "theilsen"
    mu_prior: float = 0.0               # per-channel-broadcast when channel="rgb"
    scale_factor: float | None = None   # counts conversion; None -> bit-depth

    # --- fitted results (sklearn: trailing underscore; None until fit()) ---
    gat_gain_: float | tuple[float, ...] | None = None
    gat_mu_: float | tuple[float, ...] | None = None
    gat_read_sigma_: float | tuple[float, ...] | None = None

    # --- fitted diagnostics (data-derived, trailing underscore) ---
    n_channels_: int | None = None            # 1 for gray/detect_mat, 3 for rgb
    n_patches_used_: int | tuple[int, ...] | None = None
    intensity_range_: tuple[float, float] | tuple[tuple[float, float], ...] | None = None
    mu_source_: Literal["dark_anchor", "prior"] | None = None
    fit_residual_: float | tuple[float, ...] | None = None

    def fit(self, images, *, dark_roi=None, dark_mask=None) -> "GATNoiseProfile": ...
    def gat_params(self) -> dict: ...     # op-facing keys (no underscore)
    def configure(self, op): ...          # returns a copy of op with GAT params set
    def to_json(self, filepath=None): ...
    @classmethod
    def from_json(cls, json_data): ...
    def dashboard(self, show=True): ...
```

- `fit(images, ...)` accepts a **single `Image` or an iterable of `Image`s**.
- `gat_params()` maps underscored fitted fields to the op-facing names
  (`gat_gain_` → `"gat_gain"`, etc.) and includes `gat_scale_factor`. Scalar
  modes emit floats; `rgb` emits length-3 tuples.
- `configure(op)` returns a copy of a GAT-aware op with `use_gat=True` and the
  fitted params set; it validates op/mode compatibility (a scalar op cannot take
  `rgb` per-channel params; `rgb` params target `ColorDenoise`).
- An unfitted profile (`*_` fields `None`) raises a clear "profile not fitted"
  error from `gat_params()` / `configure()` (sklearn-style).
- `to_json` / `from_json` mirror `ColorCheckerProfile`, using
  `phenotypic.sdk_._json_io` and a new `CONFIG_SUFFIX_GAT_NOISE` in
  `sdk_/_io_constants.py`. Trailing-underscore field names serialize fine.

Consumption examples:

```python
prof = GATNoiseProfile(channel="detect_mat").fit(images)
enh  = prof.configure(VisuShrinkEnhancer())          # scalar

cprof = GATNoiseProfile(channel="rgb").fit(images, dark_roi=(slice(0,60), slice(0,60)))
cden  = cprof.configure(ColorDenoise())              # per-channel
# or: ColorDenoise(use_gat=True, **cprof.gat_params())
```

## 5. Algorithm (fit)

Per image (per channel when `channel="rgb"`):

1. Read the channel array. `gray`/`detect_mat` are used directly; `rgb` is
   converted to linear light via `decode_srgb` first (matches
   `ColorDenoise`). Scale to counts via
   `resolve_scale_factor(image, scale_factor)`.
2. Compute local mean and local variance maps. **Reuse**
   `util.image_metrics.ImageMetricsCalculator.compute_local_variance` /
   `compute_local_contrast` (sliding `uniform_filter`) rather than
   re-implementing tiling; `patch_size` maps to `window_size`.
3. Reject windows that are textured (local contrast above the
   `texture_percentile` cut — discards colonies and edges, keeps flat agar and
   flat colony interiors) or saturated (any pixel ≥ `saturation_frac · max`).
   Record dropped counts.
4. Collect surviving (mean, variance) points.

Then, once across everything:

5. **Pool** all surviving points (per channel) into one cloud and run one
   robust line fit (`theilsen` / `ransac`): `Var = gain·mean + b`. `gain` =
   slope.
6. μ: if `dark_roi`/`dark_mask` given, `mu = mean(counts in region)` (per
   channel), `mu_source_ = "dark_anchor"`; else `mu = mu_prior`,
   `mu_source_ = "prior"`.
7. `read_sigma = sqrt(max(b + gain·mu, 0))`.
8. Store `gat_gain_`, `gat_mu_`, `gat_read_sigma_` and diagnostics
   (`n_channels_`, `n_patches_used_`, `intensity_range_`, `fit_residual_`).

Pooling assumes the image set shares one camera/acquisition noise model
(explicit design choice: "pool then fit once").

## 6. Integration

- **`GATNoiseProfile`**: new file, exported from `phenotypic.util.__init__`.
- **`ColorDenoise`** (`correction/_color_denoise.py`): accept per-channel
  (length-3) `gat_gain` / `gat_mu` / `gat_read_sigma`; scalars still accepted
  and broadcast. `_denoise_gat` reshapes per-channel params to `(1, 1, 3)`
  before `gat_forward` / `gat_inverse`. Field validators updated to accept a
  scalar or a length-3 sequence. **No change to `_anscombe.py`** — `gat_forward`
  / `gat_inverse` are pure elementwise and already broadcast `(1,1,3)` params.
- **`sdk_/_io_constants.py`**: add `CONFIG_SUFFIX_GAT_NOISE`.
- **Unchanged**: `_GATSupportMixin`, `DenoiseBlockMatch`, the wavelet
  denoisers, the pipeline, and the Optuna tune module. The profile's output
  simply flows into the existing `gat_*` fields; the tune module now has a
  calibrated starting point instead of the 1/0/0 blank.

## 7. Error handling & edge cases

- Surviving windows `< min_patches` (per channel) → raise `ValueError`
  (no silent garbage).
- Narrow intensity range (ill-conditioned slope, e.g. an all-flat-background
  set) → set a low-confidence signal in `fit_residual_` and warn: gain is
  unstable without dynamic range; flat colony interiors extend the range and
  help.
- Negative intercept term (`b + gain·μ < 0`) → clamp `read_sigma` to 0 and warn
  (matches Foi's "b can go negative from the pedestal").
- `dark_roi` out of bounds or empty `dark_mask` → `ValueError`.
- `configure()` with a mode/op mismatch (e.g. `rgb` params into a scalar
  grayscale op) → `ValueError`.
- `gat_params()` / `configure()` on an unfitted profile → `ValueError`.
- Field validators: `patch_size ≥ 2`, `texture_percentile`/`saturation_frac` in
  `(0, 1]`, `min_patches ≥ 1`, `scale_factor > 0` when set.

## 8. Testing

All synthetic tests build on `load_synth_yeast_plate()` and add known
Poisson–Gaussian noise so ground-truth `(gain, μ, σ)` is available.

- Scalar recovery: fit `gray`/`detect_mat`, assert `(gain, σ)` within tolerance
  of ground truth.
- Per-channel recovery: fit `rgb` with distinct per-channel gains, assert each
  channel recovered; assert fit runs in linear light (decoded) not sRGB.
- Pooling: pooled fit over N noisy images beats a single-image fit (tighter to
  ground truth).
- μ paths: `dark_roi` / `dark_mask` recovers μ and sets
  `mu_source_="dark_anchor"`; no anchor → `mu_source_="prior"`, μ = `mu_prior`.
- Saturation: injected clipped highlights are rejected and do not bias the
  slope.
- Serialization: `to_json` / `from_json` round-trip (scalar and per-channel).
- Consumption: `configure(ColorDenoise())` and
  `configure(VisuShrinkEnhancer())` produce runnable ops; the CBM3D path runs
  end-to-end on an RGB image.
- Failures: `< min_patches`, out-of-bounds `dark_roi`, unfitted-profile access,
  and mode/op mismatch all raise.
- `ColorDenoise` per-channel: scalar vs length-3 params both valid; length-3
  broadcasts correctly; a wrong-length sequence raises.

## 9. References

- Foi, A., Trimeche, M., Katkovnik, V., & Egiazarian, K. (2008). Practical
  Poissonian-Gaussian noise modeling and fitting for single-image raw-data.
  *IEEE TIP*, 17(10), 1737–1754. https://doi.org/10.1109/TIP.2008.2001399
- Mäkitalo, M., & Foi, A. (2013). Optimal inversion of the generalized Anscombe
  transformation for Poisson-Gaussian noise. *IEEE TIP*, 22(1), 91–103.
  https://doi.org/10.1109/TIP.2012.2202675
- Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K. (2007). Color image
  denoising via sparse 3D collaborative filtering with grouping constraint in
  luminance-chrominance space. *IEEE ICIP*.
  https://doi.org/10.1109/ICIP.2007.4378954
- Liu, W. et al. (2021). Parameter estimation of Poisson–Gaussian
  signal-dependent noise from a single image of CMOS/CCD image sensor using
  local binary cyclic jumping. *Sensors*, 21(24), 8330.
  https://doi.org/10.3390/s21248330
- PoGaIN: Poisson-Gaussian image noise modeling from paired samples (2022).
  arXiv:2210.04866. https://arxiv.org/abs/2210.04866
