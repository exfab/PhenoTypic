# GATNoiseProfile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `GATNoiseProfile`, a fitted Poisson–Gaussian noise estimator in `phenotypic.util` that produces GAT parameters `(gain, mu, read_sigma)` for the existing GAT-aware denoisers, plus per-channel support in `ColorDenoise` for CBM3D.

**Architecture:** A pydantic `BaseModel` (mirroring `ColorCheckerProfile`) fits a robust noise-level-function `Var = gain·mean + b` over weakly-textured local-window statistics, pooled across one or many images. Scalar modes (`gray`, `detect_mat`) yield one parameter set; `rgb` mode fits per-channel in linear light for CBM3D. The class produces params (`gat_params()` / `configure(op)`) that flow into existing `gat_*` op fields — it does not transform images itself.

**Tech Stack:** Python, pydantic v2, NumPy, `scipy.stats.theilslopes`, `sklearn.linear_model.RANSACRegressor`, `scipy.ndimage.uniform_filter` (via `util.image_metrics.ImageMetricsCalculator`), existing `sdk_/_anscombe.py` GAT functions, `bm3d`.

## Global Constraints

- **Package manager/runner:** `uv` only. Run tests with `uv run pytest ...`, type-check with `uv run mypy src/phenotypic`, lint with `uv run ruff check --fix`.
- **Pydantic models are keyword-only, class-level annotated fields.** No hand-written `__init__`. Validation via `field_validator`. `model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, extra="forbid")` (mirror `ColorCheckerProfile`).
- **sklearn convention:** hyperparameters have no trailing underscore; every attribute learned from data has a trailing underscore (`gat_gain_`, etc.).
- **Google-style docstrings**; all doctest examples runnable via `load_synth_yeast_plate()`.
- **Public API only via `__init__.py`**; implementation files are `_`-prefixed and private.
- **Immutability:** operations return copies; never mutate `image.rgb`/`image.gray` in place. The profile reads image data, never writes it.
- **No new dependencies** — scipy, scikit-learn, numpy, bm3d, pydantic are already declared.

---

### Task 1: Package skeleton — constant, empty class, exports

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py:100-108`
- Create: `src/phenotypic/util/_gat_noise_profile.py`
- Modify: `src/phenotypic/util/__init__.py`
- Test: `tests/unit/util/test_gat_noise_profile.py`

**Interfaces:**
- Consumes: `phenotypic.sdk_._io_constants` suffix pattern.
- Produces:
  - `CONFIG_SUFFIX_GAT_NOISE: Final[str] = ".json.pht-gat"` added to `CONFIG_SUFFIXES`.
  - `class GATNoiseProfile(BaseModel)` with hyperparameter fields
    (`channel: Literal["gray","detect_mat","rgb"]="gray"`, `patch_size: int=8`,
    `texture_percentile: float=0.25`, `saturation_frac: float=0.98`,
    `min_patches: int=50`, `fit_method: Literal["theilsen","ransac"]="theilsen"`,
    `mu_prior: float=0.0`, `scale_factor: float|None=None`) and fitted fields
    (`gat_gain_`, `gat_mu_`, `gat_read_sigma_` typed
    `float | tuple[float, ...] | None = None`; `n_channels_: int|None=None`;
    `n_patches_used_: int | tuple[int, ...] | None=None`;
    `intensity_range_: tuple[float,float] | tuple[tuple[float,float], ...] | None=None`;
    `mu_source_: Literal["dark_anchor","prior"] | None=None`;
    `fit_residual_: float | tuple[float, ...] | None=None`).
  - Exported as `phenotypic.util.GATNoiseProfile`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/util/test_gat_noise_profile.py
import pytest
from phenotypic.util import GATNoiseProfile


def test_unfitted_profile_constructs_with_defaults():
    prof = GATNoiseProfile()
    assert prof.channel == "gray"
    assert prof.patch_size == 8
    assert prof.texture_percentile == 0.25
    assert prof.saturation_frac == 0.98
    assert prof.min_patches == 50
    assert prof.fit_method == "theilsen"
    assert prof.mu_prior == 0.0
    assert prof.scale_factor is None
    # fitted fields are None until fit()
    assert prof.gat_gain_ is None
    assert prof.gat_mu_ is None
    assert prof.gat_read_sigma_ is None
    assert prof.mu_source_ is None


def test_extra_field_is_rejected():
    with pytest.raises(Exception):  # pydantic ValidationError
        GATNoiseProfile(typo_field=1)


def test_config_suffix_registered():
    from phenotypic.sdk_._io_constants import (
        CONFIG_SUFFIX_GAT_NOISE,
        CONFIG_SUFFIXES,
    )

    assert CONFIG_SUFFIX_GAT_NOISE == ".json.pht-gat"
    assert CONFIG_SUFFIX_GAT_NOISE in CONFIG_SUFFIXES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py -v`
Expected: FAIL — `ImportError: cannot import name 'GATNoiseProfile'`.

- [ ] **Step 3: Add the io constant**

In `src/phenotypic/sdk_/_io_constants.py`, add after `CONFIG_SUFFIX_TUNING` (line ~101):

```python
CONFIG_SUFFIX_GAT_NOISE: Final[str] = ".json.pht-gat"
```

Then add `CONFIG_SUFFIX_GAT_NOISE,` inside the `CONFIG_SUFFIXES` frozenset literal (the block at lines ~102-108).

- [ ] **Step 4: Create the skeleton module**

```python
# src/phenotypic/util/_gat_noise_profile.py
"""Fitted Poisson-Gaussian noise profile producing GAT parameters."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class GATNoiseProfile(BaseModel):
    """Fit Poisson-Gaussian noise parameters (gain, mu, read_sigma) from images.

    See ``docs/superpowers/specs/2026-06-30-gat-noise-profile-design.md``.
    """

    model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
            extra="forbid",
    )

    # -- hyperparameters (sklearn: no trailing underscore) --
    channel: Literal["gray", "detect_mat", "rgb"] = "gray"
    patch_size: int = 8
    texture_percentile: float = 0.25
    saturation_frac: float = 0.98
    min_patches: int = 50
    fit_method: Literal["theilsen", "ransac"] = "theilsen"
    mu_prior: float = 0.0
    scale_factor: float | None = None

    # -- fitted results (sklearn: trailing underscore; None until fit()) --
    gat_gain_: float | tuple[float, ...] | None = None
    gat_mu_: float | tuple[float, ...] | None = None
    gat_read_sigma_: float | tuple[float, ...] | None = None

    # -- fitted diagnostics --
    n_channels_: int | None = None
    n_patches_used_: int | tuple[int, ...] | None = None
    intensity_range_: (
        tuple[float, float] | tuple[tuple[float, float], ...] | None
    ) = None
    mu_source_: Literal["dark_anchor", "prior"] | None = None
    fit_residual_: float | tuple[float, ...] | None = None
```

- [ ] **Step 5: Wire the export**

In `src/phenotypic/util/__init__.py`, add the import (after the existing imports) and the `__all__` entry:

```python
from ._gat_noise_profile import GATNoiseProfile
```

Add `"GATNoiseProfile",` to the `__all__` list.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/sdk_/_io_constants.py src/phenotypic/util/_gat_noise_profile.py src/phenotypic/util/__init__.py tests/unit/util/test_gat_noise_profile.py
git commit -m "feat(util): GATNoiseProfile skeleton + CONFIG_SUFFIX_GAT_NOISE"
```

---

### Task 2: NLF fit math helpers

**Files:**
- Modify: `src/phenotypic/util/_gat_noise_profile.py`
- Test: `tests/unit/util/test_gat_noise_profile.py`

**Interfaces:**
- Produces (module-level functions):
  - `_fit_nlf(means: np.ndarray, variances: np.ndarray, method: str) -> tuple[float, float, float]`
    returns `(slope, intercept, residual)` where `residual` is the median
    absolute residual of the robust fit (lower is better).
  - `_resolve_read_sigma(intercept: float, gain: float, mu: float) -> float`
    returns `sqrt(max(intercept + gain*mu, 0.0))`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/util/test_gat_noise_profile.py
import numpy as np
from phenotypic.util._gat_noise_profile import _fit_nlf, _resolve_read_sigma


def test_fit_nlf_recovers_slope_and_intercept_theilsen():
    rng = np.random.default_rng(0)
    means = np.linspace(10.0, 1000.0, 400)
    true_slope, true_intercept = 2.0, 25.0
    variances = true_slope * means + true_intercept + rng.normal(0, 5, means.size)
    slope, intercept, residual = _fit_nlf(means, variances, "theilsen")
    assert slope == pytest.approx(true_slope, rel=0.1)
    assert intercept == pytest.approx(true_intercept, abs=15.0)
    assert residual >= 0.0


def test_fit_nlf_robust_to_outliers_ransac():
    rng = np.random.default_rng(1)
    means = np.linspace(10.0, 1000.0, 400)
    variances = 2.0 * means + 25.0 + rng.normal(0, 5, means.size)
    variances[:40] += 5000.0  # gross outliers (e.g. textured patches)
    slope, intercept, residual = _fit_nlf(means, variances, "ransac")
    assert slope == pytest.approx(2.0, rel=0.15)


def test_resolve_read_sigma_clamps_negative_to_zero():
    # intercept + gain*mu < 0 -> clamp
    assert _resolve_read_sigma(intercept=-100.0, gain=1.0, mu=0.0) == 0.0
    # normal case: sqrt(b + gain*mu)
    assert _resolve_read_sigma(intercept=16.0, gain=2.0, mu=0.0) == pytest.approx(4.0)
    assert _resolve_read_sigma(intercept=0.0, gain=2.0, mu=8.0) == pytest.approx(4.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py -k "fit_nlf or read_sigma" -v`
Expected: FAIL — `ImportError: cannot import name '_fit_nlf'`.

- [ ] **Step 3: Implement the helpers**

Add to `src/phenotypic/util/_gat_noise_profile.py` (imports at top, functions above the class):

```python
import numpy as np
from scipy.stats import theilslopes
from sklearn.linear_model import RANSACRegressor


def _fit_nlf(
        means: np.ndarray, variances: np.ndarray, method: str
) -> tuple[float, float, float]:
    """Robustly fit ``variance = slope * mean + intercept``.

    Args:
        means: Per-window local mean intensities (counts).
        variances: Per-window local variances (counts^2).
        method: ``"theilsen"`` (scipy) or ``"ransac"`` (sklearn).

    Returns:
        ``(slope, intercept, residual)``; ``residual`` is the median absolute
        residual of the fit.
    """
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    if method == "theilsen":
        slope, intercept, _, _ = theilslopes(variances, means)
    elif method == "ransac":
        model = RANSACRegressor(random_state=0)
        model.fit(means.reshape(-1, 1), variances)
        slope = float(model.estimator_.coef_[0])
        intercept = float(model.estimator_.intercept_)
    else:  # pragma: no cover - guarded by Literal typing
        raise ValueError(f"unknown fit_method: {method!r}")
    predicted = slope * means + intercept
    residual = float(np.median(np.abs(variances - predicted)))
    return float(slope), float(intercept), residual


def _resolve_read_sigma(intercept: float, gain: float, mu: float) -> float:
    """Recover read-noise sigma from the NLF intercept given gain and mu.

    ``intercept = sigma^2 - gain*mu`` so ``sigma = sqrt(intercept + gain*mu)``,
    clamped at zero (Foi 2008: the intercept can go negative from the pedestal).
    """
    return float(np.sqrt(max(intercept + gain * mu, 0.0)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py -k "fit_nlf or read_sigma" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/util/_gat_noise_profile.py tests/unit/util/test_gat_noise_profile.py
git commit -m "feat(util): robust NLF fit + read-sigma resolver for GAT profile"
```

---

### Task 3: Patch-statistics extraction (reusing ImageMetricsCalculator)

**Files:**
- Modify: `src/phenotypic/util/_gat_noise_profile.py`
- Test: `tests/unit/util/test_gat_noise_profile.py`

**Interfaces:**
- Consumes: `phenotypic.util.image_metrics.ImageMetricsCalculator.compute_local_variance`.
- Produces:
  - `_extract_patch_stats(counts: np.ndarray, patch_size: int, texture_percentile: float, saturation_frac: float, max_count: float) -> tuple[np.ndarray, np.ndarray]`
    returns `(means, variances)` for retained quasi-independent windows
    (subsampled on a `patch_size` grid; textured and saturated windows dropped).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/util/test_gat_noise_profile.py
from phenotypic.util._gat_noise_profile import _extract_patch_stats


def _synthetic_counts(rng, gain, mu, sigma, shape=(256, 256)):
    """Smooth intensity ramp corrupted by Poisson-Gaussian noise (counts)."""
    lam = np.linspace(20.0, 900.0, shape[1])[None, :] * np.ones(shape)
    poisson = rng.poisson(lam).astype(np.float64)
    read = rng.normal(mu, sigma, size=shape)
    return gain * poisson + read


def test_extract_patch_stats_returns_mean_variance_points():
    rng = np.random.default_rng(3)
    counts = _synthetic_counts(rng, gain=2.0, mu=0.0, sigma=5.0)
    means, variances = _extract_patch_stats(
        counts, patch_size=8, texture_percentile=0.5,
        saturation_frac=0.98, max_count=4095.0,
    )
    assert means.shape == variances.shape
    assert means.size > 50
    # variance should trend up with mean (signal-dependent noise)
    order = np.argsort(means)
    assert np.corrcoef(means[order], variances[order])[0, 1] > 0.5


def test_extract_patch_stats_drops_saturated_windows():
    rng = np.random.default_rng(4)
    counts = _synthetic_counts(rng, gain=2.0, mu=0.0, sigma=5.0)
    counts[:, -32:] = 4095.0  # saturate the brightest strip
    means, _ = _extract_patch_stats(
        counts, patch_size=8, texture_percentile=1.0,
        saturation_frac=0.98, max_count=4095.0,
    )
    # no retained window should sit at the saturation ceiling
    assert means.max() < 4095.0 * 0.98
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py -k extract_patch_stats -v`
Expected: FAIL — `ImportError: cannot import name '_extract_patch_stats'`.

- [ ] **Step 3: Implement the extractor**

Add to `src/phenotypic/util/_gat_noise_profile.py`:

```python
from scipy.ndimage import uniform_filter

from .image_metrics import ImageMetricsCalculator


def _extract_patch_stats(
        counts: np.ndarray,
        patch_size: int,
        texture_percentile: float,
        saturation_frac: float,
        max_count: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (local mean, local variance) points from weakly-textured windows.

    Reuses ``ImageMetricsCalculator.compute_local_variance`` for the variance
    map; local mean and a Weber-contrast texture measure use ``uniform_filter``
    directly. Windows are subsampled on a ``patch_size`` grid so retained points
    are quasi-independent, then textured and saturated windows are dropped.

    Args:
        counts: 2-D image in count units.
        patch_size: Local window side (maps to ``window_size``).
        texture_percentile: Keep the flattest fraction (by local contrast).
        saturation_frac: Drop windows whose local max reaches
            ``saturation_frac * max_count``.
        max_count: Saturation ceiling in count units.

    Returns:
        ``(means, variances)`` 1-D arrays for retained windows.
    """
    counts = np.asarray(counts, dtype=np.float64)
    calc = ImageMetricsCalculator(counts)
    variance_map = calc.compute_local_variance(counts, window_size=patch_size)
    mean_map = uniform_filter(counts, size=patch_size)
    contrast_map = calc.compute_local_contrast(counts, window_size=patch_size)
    local_max = uniform_filter(counts, size=patch_size, mode="nearest")
    # a windowed maximum: dilate via max filter for a strict saturation test
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(counts, size=patch_size, mode="nearest")

    # subsample on a patch_size grid for quasi-independent points
    s = slice(patch_size // 2, None, patch_size)
    means = mean_map[s, s].ravel()
    variances = variance_map[s, s].ravel()
    contrast = contrast_map[s, s].ravel()
    win_max = local_max[s, s].ravel()

    keep = win_max < saturation_frac * max_count
    means, variances, contrast = means[keep], variances[keep], contrast[keep]
    if means.size == 0:
        return means, variances
    # keep the flattest fraction by local contrast
    thresh = np.quantile(contrast, texture_percentile)
    flat = contrast <= thresh
    return means[flat], variances[flat]
```

Note: remove the first (overwritten) `local_max = uniform_filter(...)` line when
implementing — it is shown only to flag that a *maximum* filter, not a mean, is
required for the saturation test. Final code uses `maximum_filter` once, with
its import hoisted to the top of the module.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py -k extract_patch_stats -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/util/_gat_noise_profile.py tests/unit/util/test_gat_noise_profile.py
git commit -m "feat(util): weakly-textured patch-stat extraction for GAT profile"
```

---

### Task 4: Scalar `fit()` for gray/detect_mat

**Files:**
- Modify: `src/phenotypic/util/_gat_noise_profile.py`
- Test: `tests/unit/util/test_gat_noise_profile.py`

**Interfaces:**
- Consumes: `_extract_patch_stats`, `_fit_nlf`, `_resolve_read_sigma`,
  `phenotypic.sdk_._anscombe.resolve_scale_factor`, `phenotypic._core._image.Image`.
- Produces:
  - `GATNoiseProfile.fit(self, images, *, dark_roi=None, dark_mask=None) -> GATNoiseProfile`
    (this task: `channel` in `{"gray","detect_mat"}`; sets scalar `gat_gain_`,
    `gat_mu_`, `gat_read_sigma_`, `n_channels_=1`, `n_patches_used_`,
    `intensity_range_`, `mu_source_`, `fit_residual_`).
  - Private field validators: `patch_size >= 2`,
    `0 < texture_percentile <= 1`, `0 < saturation_frac <= 1`,
    `min_patches >= 1`, `scale_factor > 0 when set`.
  - Helper `_channel_counts(self, image: Image) -> np.ndarray` for scalar modes.
  - Helper `_iter_images(images) -> list[Image]` (accept single or iterable).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/util/test_gat_noise_profile.py
from phenotypic._core._image import Image


def _counts_to_uint8_image(counts, max_count):
    """Wrap a counts array as an 8-bit Image (single-channel replicated to RGB)."""
    norm = np.clip(counts / max_count, 0, 1)
    rgb = np.repeat((norm * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    return Image(rgb)


def test_fit_scalar_recovers_gain_from_pooled_images():
    rng = np.random.default_rng(5)
    # detect_mat is [0,1]; use scale 255 so counts are recoverable
    imgs = []
    for _ in range(4):
        counts = _synthetic_counts(rng, gain=3.0, mu=0.0, sigma=4.0)
        imgs.append(_counts_to_uint8_image(counts, max_count=255.0))
    prof = GATNoiseProfile(channel="detect_mat", patch_size=8,
                           texture_percentile=0.5, min_patches=10).fit(imgs)
    assert prof.n_channels_ == 1
    assert isinstance(prof.gat_gain_, float)
    assert prof.gat_gain_ > 0
    assert prof.mu_source_ == "prior"
    assert prof.gat_mu_ == 0.0


def test_fit_too_few_patches_raises():
    rng = np.random.default_rng(6)
    img = _counts_to_uint8_image(_synthetic_counts(rng, 2.0, 0.0, 4.0), 255.0)
    with pytest.raises(ValueError, match="patches"):
        GATNoiseProfile(channel="gray", patch_size=8,
                        texture_percentile=0.01, min_patches=100000).fit(img)


def test_fit_dark_roi_sets_mu_source_dark_anchor():
    rng = np.random.default_rng(7)
    counts = _synthetic_counts(rng, gain=2.0, mu=10.0, sigma=4.0)
    counts[:40, :40] = rng.normal(10.0, 4.0, size=(40, 40))  # dark anchor ~ mu
    img = _counts_to_uint8_image(counts, 255.0)
    prof = GATNoiseProfile(channel="gray", patch_size=8, texture_percentile=0.5,
                           min_patches=10).fit(img, dark_roi=(slice(0, 40), slice(0, 40)))
    assert prof.mu_source_ == "dark_anchor"
    assert isinstance(prof.gat_mu_, float)


def test_field_validators_reject_bad_hyperparameters():
    with pytest.raises(Exception):
        GATNoiseProfile(patch_size=1)
    with pytest.raises(Exception):
        GATNoiseProfile(texture_percentile=0.0)
    with pytest.raises(Exception):
        GATNoiseProfile(saturation_frac=1.5)
    with pytest.raises(Exception):
        GATNoiseProfile(scale_factor=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py -k "fit_scalar or too_few or dark_roi or field_validators" -v`
Expected: FAIL — `AttributeError: 'GATNoiseProfile' object has no attribute 'fit'`.

- [ ] **Step 3: Implement validators + scalar fit**

Add imports and members to `GATNoiseProfile`:

```python
from typing import TYPE_CHECKING, Iterable
from pydantic import field_validator

from ..sdk_._anscombe import resolve_scale_factor

if TYPE_CHECKING:
    from phenotypic._core._image import Image
```

Field validators (inside the class):

```python
    @field_validator("patch_size")
    @classmethod
    def _check_patch_size(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"patch_size must be >= 2, got {v}")
        return v

    @field_validator("texture_percentile", "saturation_frac")
    @classmethod
    def _check_unit_fraction(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {v}")
        return v

    @field_validator("min_patches")
    @classmethod
    def _check_min_patches(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"min_patches must be >= 1, got {v}")
        return v

    @field_validator("scale_factor")
    @classmethod
    def _check_scale_factor(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"scale_factor must be > 0, got {v}")
        return v
```

Helpers + `fit` (inside the class):

```python
    @staticmethod
    def _iter_images(images) -> list:
        """Normalize a single Image or an iterable of Images to a list."""
        from phenotypic._core._image import Image as _Image
        if isinstance(images, _Image):
            return [images]
        return list(images)

    def _channel_counts(self, image: "Image") -> np.ndarray:
        """Return the scalar channel in count units."""
        scale = resolve_scale_factor(image, self.scale_factor)
        if self.channel == "gray":
            arr = np.asarray(image.gray[:], dtype=np.float64)
        else:  # detect_mat
            arr = np.asarray(image.detect_mat[:], dtype=np.float64)
        return arr * scale

    def _dark_mu(self, counts_list, dark_roi, dark_mask) -> float | None:
        """Estimate mu from an explicit dark region, else None."""
        if dark_roi is None and dark_mask is None:
            return None
        vals = []
        for counts in counts_list:
            if dark_roi is not None:
                vals.append(counts[dark_roi].ravel())
            else:
                if dark_mask.shape != counts.shape:
                    raise ValueError("dark_mask shape does not match image")
                vals.append(counts[dark_mask].ravel())
        pooled = np.concatenate(vals)
        if pooled.size == 0:
            raise ValueError("dark region selected no pixels")
        return float(np.mean(pooled))

    def fit(self, images, *, dark_roi=None, dark_mask=None) -> "GATNoiseProfile":
        """Fit noise parameters, pooling patch statistics across all images."""
        image_list = self._iter_images(images)
        if self.channel == "rgb":
            return self._fit_rgb(image_list, dark_roi, dark_mask)

        counts_list = [self._channel_counts(im) for im in image_list]
        max_count = resolve_scale_factor(image_list[0], self.scale_factor)
        means, variances = [], []
        for counts in counts_list:
            m, v = _extract_patch_stats(
                counts, self.patch_size, self.texture_percentile,
                self.saturation_frac, max_count,
            )
            means.append(m)
            variances.append(v)
        means = np.concatenate(means) if means else np.array([])
        variances = np.concatenate(variances) if variances else np.array([])
        if means.size < self.min_patches:
            raise ValueError(
                f"only {means.size} weakly-textured patches survived "
                f"(min_patches={self.min_patches}); relax texture_percentile "
                f"or lower min_patches"
            )
        slope, intercept, residual = _fit_nlf(means, variances, self.fit_method)
        dark = self._dark_mu(counts_list, dark_roi, dark_mask)
        mu = dark if dark is not None else self.mu_prior
        gain = max(slope, 1e-8)
        self.gat_gain_ = gain
        self.gat_mu_ = float(mu)
        self.gat_read_sigma_ = _resolve_read_sigma(intercept, gain, mu)
        self.n_channels_ = 1
        self.n_patches_used_ = int(means.size)
        self.intensity_range_ = (float(means.min()), float(means.max()))
        self.mu_source_ = "dark_anchor" if dark is not None else "prior"
        self.fit_residual_ = residual
        return self
```

Add a temporary stub so the class imports before Task 5:

```python
    def _fit_rgb(self, image_list, dark_roi, dark_mask) -> "GATNoiseProfile":
        raise NotImplementedError("rgb mode lands in Task 5")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py -k "fit_scalar or too_few or dark_roi or field_validators" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/util/_gat_noise_profile.py tests/unit/util/test_gat_noise_profile.py
git commit -m "feat(util): scalar gray/detect_mat fit for GATNoiseProfile"
```

---

### Task 5: Per-channel `rgb` fit (linear light)

**Files:**
- Modify: `src/phenotypic/util/_gat_noise_profile.py`
- Test: `tests/unit/util/test_gat_noise_profile.py`

**Interfaces:**
- Consumes: `phenotypic.sdk_.colourspace.decode_srgb`, `_extract_patch_stats`,
  `_fit_nlf`, `_resolve_read_sigma`, `resolve_scale_factor`.
- Produces: `GATNoiseProfile._fit_rgb(...)` populating length-3 tuple fitted
  fields and `n_channels_=3`; `mu_prior` and dark anchor apply per channel.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/util/test_gat_noise_profile.py
from phenotypic.data import load_synth_yeast_plate


def test_fit_rgb_produces_per_channel_tuples():
    image = load_synth_yeast_plate()
    prof = GATNoiseProfile(channel="rgb", patch_size=8, texture_percentile=0.5,
                           min_patches=10).fit(image)
    assert prof.n_channels_ == 3
    assert isinstance(prof.gat_gain_, tuple) and len(prof.gat_gain_) == 3
    assert isinstance(prof.gat_read_sigma_, tuple) and len(prof.gat_read_sigma_) == 3
    assert isinstance(prof.gat_mu_, tuple) and len(prof.gat_mu_) == 3
    assert all(g > 0 for g in prof.gat_gain_)
    assert prof.mu_source_ == "prior"


def test_fit_rgb_requires_rgb_data():
    # a single-channel gray Image has empty rgb -> rgb fit should error clearly
    rng = np.random.default_rng(8)
    gray = (np.clip(_synthetic_counts(rng, 2.0, 0.0, 4.0) / 255.0, 0, 1))
    img = Image(gray.astype(np.float64))
    with pytest.raises(ValueError, match="RGB"):
        GATNoiseProfile(channel="rgb", min_patches=10).fit(img)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py -k "fit_rgb" -v`
Expected: FAIL — `NotImplementedError: rgb mode lands in Task 5`.

- [ ] **Step 3: Implement `_fit_rgb`**

Add import and replace the stub:

```python
from ..sdk_.colourspace import decode_srgb
```

```python
    def _rgb_counts_per_channel(self, image: "Image") -> list[np.ndarray]:
        """Return the three linear-light channels in count units."""
        if image.rgb.isempty():
            raise ValueError(
                "channel='rgb' requires a 3-channel RGB image; this image has "
                "no RGB data"
            )
        raw = np.asarray(image._data.rgb)
        vmax = np.iinfo(raw.dtype).max
        rgb01 = raw.astype(np.float64) / vmax
        rgb_lin = decode_srgb(rgb01)
        scale = resolve_scale_factor(image, self.scale_factor)
        counts = rgb_lin * scale
        return [counts[:, :, c] for c in range(3)]

    def _fit_rgb(self, image_list, dark_roi, dark_mask) -> "GATNoiseProfile":
        per_image = [self._rgb_counts_per_channel(im) for im in image_list]
        max_count = resolve_scale_factor(image_list[0], self.scale_factor)
        gains, mus, sigmas, ns, ranges, residuals = [], [], [], [], [], []
        mu_source = "prior"
        for c in range(3):
            channel_counts = [pi[c] for pi in per_image]
            means, variances = [], []
            for counts in channel_counts:
                m, v = _extract_patch_stats(
                    counts, self.patch_size, self.texture_percentile,
                    self.saturation_frac, max_count,
                )
                means.append(m)
                variances.append(v)
            means = np.concatenate(means) if means else np.array([])
            variances = np.concatenate(variances) if variances else np.array([])
            if means.size < self.min_patches:
                raise ValueError(
                    f"channel {c}: only {means.size} weakly-textured patches "
                    f"survived (min_patches={self.min_patches})"
                )
            slope, intercept, residual = _fit_nlf(means, variances, self.fit_method)
            dark = self._dark_mu(channel_counts, dark_roi, dark_mask)
            mu = dark if dark is not None else self.mu_prior
            gain = max(slope, 1e-8)
            gains.append(gain)
            mus.append(float(mu))
            sigmas.append(_resolve_read_sigma(intercept, gain, mu))
            ns.append(int(means.size))
            ranges.append((float(means.min()), float(means.max())))
            residuals.append(residual)
            if dark is not None:
                mu_source = "dark_anchor"
        self.gat_gain_ = tuple(gains)
        self.gat_mu_ = tuple(mus)
        self.gat_read_sigma_ = tuple(sigmas)
        self.n_channels_ = 3
        self.n_patches_used_ = tuple(ns)
        self.intensity_range_ = tuple(ranges)
        self.mu_source_ = mu_source
        self.fit_residual_ = tuple(residuals)
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py -k "fit_rgb" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/util/_gat_noise_profile.py tests/unit/util/test_gat_noise_profile.py
git commit -m "feat(util): per-channel linear-light rgb fit for GATNoiseProfile"
```

---

### Task 6: `gat_params()` and `configure(op)`

**Files:**
- Modify: `src/phenotypic/util/_gat_noise_profile.py`
- Test: `tests/unit/util/test_gat_noise_profile.py`

**Interfaces:**
- Consumes: fitted fields; `phenotypic.correction.ColorDenoise`,
  `phenotypic.enhance.VisuShrinkEnhancer`.
- Produces:
  - `GATNoiseProfile.gat_params(self) -> dict` with keys
    `"gat_gain"`, `"gat_mu"`, `"gat_read_sigma"`, `"gat_scale_factor"`
    (scalar floats for scalar modes; length-3 tuples for `rgb`).
  - `GATNoiseProfile.configure(self, op)` returns `op.model_copy(update=...)`
    with `use_gat=True` and the params set; validates mode/op compatibility;
    raises on unfitted profile.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/util/test_gat_noise_profile.py
from phenotypic.correction import ColorDenoise
from phenotypic.enhance import VisuShrinkEnhancer


def test_gat_params_unfitted_raises():
    with pytest.raises(ValueError, match="not fitted"):
        GATNoiseProfile().gat_params()


def test_configure_scalar_op_sets_params():
    rng = np.random.default_rng(9)
    img = _counts_to_uint8_image(_synthetic_counts(rng, 3.0, 0.0, 4.0), 255.0)
    prof = GATNoiseProfile(channel="detect_mat", patch_size=8,
                           texture_percentile=0.5, min_patches=10).fit(img)
    enh = prof.configure(VisuShrinkEnhancer())
    assert enh.use_gat is True
    assert enh.gat_gain == prof.gat_gain_
    # original op is unchanged (model_copy)
    assert VisuShrinkEnhancer().use_gat is False


def test_configure_rejects_rgb_params_into_scalar_op():
    image = load_synth_yeast_plate()
    prof = GATNoiseProfile(channel="rgb", patch_size=8, texture_percentile=0.5,
                           min_patches=10).fit(image)
    with pytest.raises(ValueError, match="rgb"):
        prof.configure(VisuShrinkEnhancer())


def test_configure_rgb_into_color_denoise():
    image = load_synth_yeast_plate()
    prof = GATNoiseProfile(channel="rgb", patch_size=8, texture_percentile=0.5,
                           min_patches=10).fit(image)
    cden = prof.configure(ColorDenoise())
    assert cden.use_gat is True
    assert tuple(cden.gat_gain) == prof.gat_gain_
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py -k "gat_params or configure" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'gat_params'`.

- [ ] **Step 3: Implement `gat_params` and `configure`**

```python
    def _require_fitted(self) -> None:
        if self.gat_gain_ is None:
            raise ValueError("profile is not fitted; call fit() first")

    def gat_params(self) -> dict:
        """Return GAT parameters keyed for the GAT-aware op fields."""
        self._require_fitted()
        return {
            "gat_gain": self.gat_gain_,
            "gat_mu": self.gat_mu_,
            "gat_read_sigma": self.gat_read_sigma_,
            "gat_scale_factor": self.scale_factor,
        }

    def configure(self, op):
        """Return a copy of ``op`` configured with the fitted GAT params.

        The scalar modes target any GAT-aware op with a ``use_gat`` field
        (``VisuShrinkEnhancer``, ``BayesShrinkEnhancer``, ``EnhanceBlockMatch``,
        ``DenoiseBlockMatch``). The ``rgb`` mode targets ``ColorDenoise``.
        """
        self._require_fitted()
        from phenotypic.correction import ColorDenoise as _ColorDenoise

        is_color = isinstance(op, _ColorDenoise)
        if self.channel == "rgb" and not is_color:
            raise ValueError(
                "rgb profile produces per-channel params; configure a "
                "ColorDenoise op, not "
                f"{type(op).__name__}"
            )
        if self.channel != "rgb" and is_color:
            raise ValueError(
                f"{self.channel!r} profile produces scalar params; use a "
                "scalar GAT-aware op, not ColorDenoise"
            )
        if not hasattr(op, "use_gat"):
            raise ValueError(f"{type(op).__name__} is not GAT-aware")
        return op.model_copy(update={"use_gat": True, **self.gat_params()})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py -k "gat_params or configure" -v`
Expected: PASS (4 tests). NOTE: `test_configure_rgb_into_color_denoise` depends
on Task 9 (per-channel `ColorDenoise`); if run before Task 9 it fails validation
on the tuple `gat_gain`. Mark it `@pytest.mark.xfail(reason="needs Task 9")`
until Task 9, then remove the marker.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/util/_gat_noise_profile.py tests/unit/util/test_gat_noise_profile.py
git commit -m "feat(util): gat_params() and configure() for GATNoiseProfile"
```

---

### Task 7: `to_json` / `from_json` round-trip

**Files:**
- Modify: `src/phenotypic/util/_gat_noise_profile.py`
- Test: `tests/unit/util/test_gat_noise_profile.py`

**Interfaces:**
- Consumes: `phenotypic.sdk_._json_io.read_json_source`,
  `phenotypic.sdk_._io_constants.ensure_typed_json_suffix`,
  `CONFIG_SUFFIX_GAT_NOISE`.
- Produces: `to_json(self, filepath=None) -> str | None`,
  `from_json(cls, json_data) -> GATNoiseProfile` (mirror `ColorCheckerProfile`).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/util/test_gat_noise_profile.py
import tempfile
from pathlib import Path


def test_to_from_json_roundtrip_scalar():
    rng = np.random.default_rng(10)
    img = _counts_to_uint8_image(_synthetic_counts(rng, 3.0, 0.0, 4.0), 255.0)
    prof = GATNoiseProfile(channel="detect_mat", patch_size=8,
                           texture_percentile=0.5, min_patches=10).fit(img)
    loaded = GATNoiseProfile.from_json(prof.to_json())
    assert loaded.gat_gain_ == prof.gat_gain_
    assert loaded.channel == "detect_mat"
    assert loaded.mu_source_ == prof.mu_source_


def test_to_from_json_roundtrip_rgb_file():
    image = load_synth_yeast_plate()
    prof = GATNoiseProfile(channel="rgb", patch_size=8, texture_percentile=0.5,
                           min_patches=10).fit(image)
    from phenotypic.sdk_._io_constants import (
        CONFIG_SUFFIX_GAT_NOISE, ensure_typed_json_suffix,
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "noise.json"
        prof.to_json(p)
        saved = ensure_typed_json_suffix(p, CONFIG_SUFFIX_GAT_NOISE)
        loaded = GATNoiseProfile.from_json(saved)
    assert isinstance(loaded.gat_gain_, tuple) and len(loaded.gat_gain_) == 3
    assert tuple(loaded.gat_gain_) == tuple(prof.gat_gain_)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py -k "roundtrip" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'to_json'`.

- [ ] **Step 3: Implement serialization**

```python
from pathlib import Path

from ..sdk_._io_constants import CONFIG_SUFFIX_GAT_NOISE, ensure_typed_json_suffix
from ..sdk_._json_io import read_json_source
```

```python
    def to_json(self, filepath: str | Path | None = None) -> str | None:
        """Serialize this profile to JSON (mirrors ColorCheckerProfile.to_json)."""
        json_str = self.model_dump_json(indent=2)
        if filepath is not None:
            ensure_typed_json_suffix(
                filepath, CONFIG_SUFFIX_GAT_NOISE
            ).write_text(json_str)
            return None
        return json_str

    @classmethod
    def from_json(cls, json_data: str | Path | dict) -> "GATNoiseProfile":
        """Reconstruct a profile from JSON written by :meth:`to_json`."""
        return cls.model_validate(read_json_source(json_data))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py -k "roundtrip" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/util/_gat_noise_profile.py tests/unit/util/test_gat_noise_profile.py
git commit -m "feat(util): JSON serialization for GATNoiseProfile"
```

---

### Task 8: `dashboard()` diagnostic figure

**Files:**
- Modify: `src/phenotypic/util/_gat_noise_profile.py`
- Test: `tests/unit/util/test_gat_noise_profile.py`

**Interfaces:**
- Produces: `dashboard(self, show: bool = True)` returns a `matplotlib.figure.Figure`
  with the pooled (mean, variance) scatter and the fitted line annotated with
  `(gain, mu, read_sigma)`. Explicit matplotlib (no pyplot state).
- Note: this task re-derives the pooled scatter by re-extracting patches from a
  stored reference image. To keep the profile serializable and lightweight, the
  scatter is recomputed from an image passed to `dashboard(image=...)` rather
  than cached. Update signature to `dashboard(self, image, *, show=True)`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/util/test_gat_noise_profile.py
def test_dashboard_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    rng = np.random.default_rng(11)
    img = _counts_to_uint8_image(_synthetic_counts(rng, 3.0, 0.0, 4.0), 255.0)
    prof = GATNoiseProfile(channel="detect_mat", patch_size=8,
                           texture_percentile=0.5, min_patches=10).fit(img)
    fig = prof.dashboard(img, show=False)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py -k dashboard -v`
Expected: FAIL — `AttributeError: ... has no attribute 'dashboard'`.

- [ ] **Step 3: Implement `dashboard`**

```python
    def dashboard(self, image, *, show: bool = True):
        """Plot the (mean, variance) cloud and fitted NLF line for one image."""
        self._require_fitted()
        from matplotlib.figure import Figure

        max_count = resolve_scale_factor(image, self.scale_factor)
        if self.channel == "rgb":
            channels = self._rgb_counts_per_channel(image)
            gains = self.gat_gain_
        else:
            channels = [self._channel_counts(image)]
            gains = (self.gat_gain_,)

        fig = Figure(figsize=(6, 4 * len(channels)))
        for idx, counts in enumerate(channels):
            m, v = _extract_patch_stats(
                counts, self.patch_size, self.texture_percentile,
                self.saturation_frac, max_count,
            )
            ax = fig.add_subplot(len(channels), 1, idx + 1)
            ax.scatter(m, v, s=4, alpha=0.3, label="patches")
            line_x = np.array([m.min(), m.max()]) if m.size else np.array([0, 1])
            ax.plot(line_x, gains[idx] * line_x, "r-", label="fitted gain")
            ax.set_xlabel("local mean (counts)")
            ax.set_ylabel("local variance")
            ax.set_title(f"channel {idx}: gain={gains[idx]:.3g}")
            ax.legend()
        fig.tight_layout()
        if show:
            fig.show()
        return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py -k dashboard -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/util/_gat_noise_profile.py tests/unit/util/test_gat_noise_profile.py
git commit -m "feat(util): dashboard() diagnostic for GATNoiseProfile"
```

---

### Task 9: Per-channel params in `ColorDenoise`

**Files:**
- Modify: `src/phenotypic/correction/_color_denoise.py:125-183` (fields + validators) and `:243-263` (`_denoise_gat`)
- Test: `tests/unit/correction/test_color_denoise.py` (add cases; create if absent)

**Interfaces:**
- Consumes: `phenotypic.sdk_._anscombe.gat_forward`, `gat_inverse`.
- Produces: `ColorDenoise.gat_gain`, `gat_mu`, `gat_read_sigma` accept
  `float | tuple[float, float, float]` (scalars still valid, broadcast);
  `_denoise_gat` reshapes per-channel params to `(1, 1, 3)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/correction/test_color_denoise.py
import numpy as np
import pytest
from phenotypic._core._image import Image
from phenotypic.correction import ColorDenoise
from phenotypic.data import load_synth_yeast_plate


def test_color_denoise_accepts_per_channel_gat_params():
    op = ColorDenoise(use_gat=True, gat_gain=(2.0, 3.0, 1.5),
                      gat_read_sigma=(4.0, 5.0, 3.0), gat_mu=(0.0, 0.0, 0.0))
    assert tuple(op.gat_gain) == (2.0, 3.0, 1.5)


def test_color_denoise_per_channel_runs_end_to_end():
    image = load_synth_yeast_plate()
    op = ColorDenoise(use_gat=True, block_size=8,
                      gat_gain=(2.0, 3.0, 1.5), gat_read_sigma=(4.0, 5.0, 3.0))
    out = op.apply(image)
    assert out.rgb[:].shape == image.rgb[:].shape


def test_color_denoise_rejects_wrong_length_sequence():
    with pytest.raises(Exception):
        ColorDenoise(gat_gain=(2.0, 3.0))  # length 2, not scalar or 3


def test_color_denoise_scalar_gat_still_valid():
    op = ColorDenoise(use_gat=True, gat_gain=1.5, gat_read_sigma=2.0)
    assert op.gat_gain == 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/correction/test_color_denoise.py -v`
Expected: FAIL — pydantic rejects the tuple (`gat_gain` is typed `float`).

- [ ] **Step 3: Widen the field types and validators**

In `src/phenotypic/correction/_color_denoise.py`, change the three field
declarations (lines ~131-133):

```python
    gat_gain: Annotated[
        float | tuple[float, float, float], TuneSpec(tunable=False)
    ] = 1.0
    gat_mu: Annotated[
        float | tuple[float, float, float], TuneSpec(tunable=False)
    ] = 0.0
    gat_read_sigma: Annotated[
        float | tuple[float, float, float], TuneSpec(tunable=False)
    ] = 0.0
```

Replace the `_check_gat_gain` and `_check_gat_read_sigma` validators with
sequence-aware versions:

```python
    @field_validator("gat_gain")
    @classmethod
    def _check_gat_gain(cls, v):
        """Require positive camera gain(s); scalar or length-3 per channel."""
        values = v if isinstance(v, tuple) else (v,)
        if any(g <= 0 for g in values):
            raise ValueError(f"gat_gain must be > 0, got {v}")
        return v

    @field_validator("gat_read_sigma")
    @classmethod
    def _check_gat_read_sigma(cls, v):
        """Require non-negative read-noise std(s); scalar or length-3."""
        values = v if isinstance(v, tuple) else (v,)
        if any(s < 0 for s in values):
            raise ValueError(f"gat_read_sigma must be >= 0, got {v}")
        return v
```

(Length is enforced by the `tuple[float, float, float]` type: a length-2 tuple
fails validation.)

- [ ] **Step 4: Broadcast per-channel params in `_denoise_gat`**

Replace the body of `_denoise_gat` (lines ~243-263) so params reshape to
`(1, 1, 3)` when per-channel:

```python
    @staticmethod
    def _as_channel_param(value):
        """Reshape a scalar or length-3 param to broadcast over RGB (1,1,C)."""
        if isinstance(value, tuple):
            return np.asarray(value, dtype=np.float64).reshape(1, 1, 3)
        return value

    def _denoise_gat(self, rgb_lin: np.ndarray, image: Image) -> np.ndarray:
        """Run CBM3D in the GAT-stabilized domain (scalar or per-channel)."""
        scale = resolve_scale_factor(image, self.gat_scale_factor)
        counts = rgb_lin * scale
        mu = self._as_channel_param(self.gat_mu)
        sigma = self._as_channel_param(self.gat_read_sigma)
        gain = self._as_channel_param(self.gat_gain)
        stabilized = gat_forward(counts, mu, sigma, gain)
        denoised = bm3d.bm3d_rgb(stabilized, 1.0, self._build_profile(), "opp")
        recovered = gat_inverse(denoised, mu, sigma, gain)
        return np.asarray(recovered / scale, dtype=np.float64)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/correction/test_color_denoise.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Un-xfail the Task 6 test and run the profile suite**

Remove the `@pytest.mark.xfail` marker on `test_configure_rgb_into_color_denoise`
(added in Task 6), then run:

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py tests/unit/correction/test_color_denoise.py -v`
Expected: PASS (all).

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/correction/_color_denoise.py tests/unit/util/test_gat_noise_profile.py tests/unit/correction/test_color_denoise.py
git commit -m "feat(correction): per-channel GAT params in ColorDenoise for CBM3D"
```

---

### Task 10: Type-check, lint, docs

**Files:**
- Modify: `src/phenotypic/util/CLAUDE.md` (if present) or add a note to
  `src/phenotypic/correction/CLAUDE.md`
- Verify: whole suite

**Interfaces:** none (finalization).

- [ ] **Step 1: Type-check and lint**

Run: `uv run mypy src/phenotypic/util/_gat_noise_profile.py src/phenotypic/correction/_color_denoise.py`
Expected: no new errors. Fix any that appear (e.g. add `# type: ignore[...]`
only where a third-party stub is missing).

Run: `uv run ruff check --fix src/phenotypic/util/_gat_noise_profile.py src/phenotypic/correction/_color_denoise.py tests/unit/util/test_gat_noise_profile.py tests/unit/correction/test_color_denoise.py`
Expected: clean.

- [ ] **Step 2: Document the new consumer relationship**

Add a short paragraph to `src/phenotypic/correction/CLAUDE.md` (or create
`src/phenotypic/util/CLAUDE.md` if the util package should have one) noting:
`GATNoiseProfile` (in `phenotypic.util`) fits Poisson–Gaussian noise params and
feeds them to GAT-aware ops via `gat_params()` / `configure(op)`; `rgb` mode is
per-channel for `ColorDenoise` (CBM3D); `ColorCheckerProfile` will move to
`util/` in a later PR.

- [ ] **Step 3: Run the full affected suite**

Run: `uv run pytest tests/unit/util/test_gat_noise_profile.py tests/unit/correction/ -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs(util): document GATNoiseProfile consumer relationship; lint/type pass"
```

---

## Self-Review

**Spec coverage:**
- §4 API surface (fields, `fit`, `gat_params`, `configure`, `to_json`/`from_json`, `dashboard`) → Tasks 1, 4, 5, 6, 7, 8.
- §5 algorithm (patch extraction, pooling, robust fit, μ policy, σ resolve) → Tasks 2, 3, 4, 5.
- §6 integration (`ColorDenoise` per-channel, `CONFIG_SUFFIX_GAT_NOISE`, exports; no `_anscombe` change) → Tasks 1, 9.
- §7 error handling (min_patches, negative intercept clamp, bad ROI, unfitted, mode/op mismatch, validators) → Tasks 2, 4, 6, 9.
- §8 testing (scalar + per-channel recovery, pooling, μ paths, saturation, serialization, consumption end-to-end, failures) → Tasks 2–9.
- §2 out-of-scope (cumulant μ, None-sentinel, auto dark discovery, ColorCheckerProfile move) → intentionally not implemented; ColorCheckerProfile move noted in Task 10 docs.

**Placeholder scan:** No TBD/TODO. Task 3 flags the intentional two-line `local_max` illustration and instructs removing the overwritten line — implement with a single `maximum_filter`. Every code step shows real code.

**Type consistency:** fitted fields `gat_gain_`/`gat_mu_`/`gat_read_sigma_` are `float | tuple[float, ...] | None` throughout; `gat_params()` maps to op field names `gat_gain`/`gat_mu`/`gat_read_sigma`/`gat_scale_factor`; `ColorDenoise` fields widened to `float | tuple[float, float, float]` to consume them; `_fit_nlf` returns `(slope, intercept, residual)` consistently; `_extract_patch_stats` returns `(means, variances)` consistently.

**Known cross-task dependency:** `test_configure_rgb_into_color_denoise` (Task 6) requires Task 9; xfail marker instructed in Task 6, removed in Task 9 Step 6.
