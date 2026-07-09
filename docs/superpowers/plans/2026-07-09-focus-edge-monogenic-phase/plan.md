# FocusEdgeMonogenicPhase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `FocusEdgeMonogenicPhase`, a contrast-invariant edge enhancer that computes phase congruency from an isotropic log-Gabor bandpass plus the Riesz transform, eliminating `FocusEdgePhase`'s orientation sweep.

**Architecture:** Extract the four frequency-domain helpers that `FocusEdgePhase` already owns into a new shared, pure-function module `enhance/_monogenic_kernels.py`; add the Riesz multiplier, the periodic FFT and the monogenic congruency routine to it; then build a thin pydantic operation on top. `FocusEdgePhase` is refactored onto the same helpers and must stay bit-identical.

**Tech Stack:** Python 3.12, numpy, pydantic v2, pytest, `uv` (sole package manager/runner).

**Source spec:** [`docs/superpowers/specs/2026-07-08-alt-phase-detection/monogenic-phase-congruency.md`](../../specs/2026-07-08-alt-phase-detection/monogenic-phase-congruency.md)
**Deviations register:** [`drift-register.md`](../../specs/2026-07-08-alt-phase-detection/drift-register.md)
**Executable evidence for every number below:** [`verify_claims.py`](../../specs/2026-07-08-alt-phase-detection/verify_claims.py) (21/21 passing)

---

## Global Constraints

- **`uv` is the sole runner.** Never bare `python`/`pip`. Tests: `uv run pytest`. Lint: `uv run ruff check --fix`. Types: `uv run mypy src/phenotypic`.
- **This is a port, not a derivation.** Where this plan and Kovesi's `phasecongmono` disagree, the reference wins. Every deviation is already enumerated in `drift-register.md` rows M1–M5; do not add new ones without adding a row.
- **`EPSILON_MONOGENIC = 1e-4`**, a module constant. It is **not** `FocusEdgePhase`'s `1e-5`. `1e-5` belongs to *Julia's* `phasecong3`; MATLAB's `phasecong3.m` uses `1e-4`; all three `phasecongmono` references use `1e-4`.
- **`fft2`, not `perfft2`** (drift M4). Kovesi's MATLAB `phasecongmono.m:156` uses `perfft2`; his Julia `phasecongruency.jl:446` explicitly does not (`IMG = fft(img)   # Use fft rather than perfft2`). We ship the Julia branch. The kernel takes `periodic: bool = False`; **the operation does not expose it**. The golden fixture pins the `periodic=True` branch because that is what `phasepack` computes. This is the **only** genuine fork between Kovesi's two implementations.
- **Every frequency axis is divided by `N`.** Both `frequencyfilt.jl:73` and `filtergrid.m:49` do this. `phasepack` divides an odd axis by `N−1` (in `filtergrid` *and* `lowpassfilter`) — a `phasepack` bug, not a Kovesi divergence. All three agree at even sizes; the fixture is 64×64 so it cannot inherit it.
- **Cite a file and a line, never "the reference".** Three separate claims in this spec were generalised from `phasepack` — the only runnable implementation — and all three were wrong (`drift-register.md` S7). Runnability and authority are unrelated.
- **`T` is floored at `EPSILON_MONOGENIC`** (drift M5). That floor is `phasepack`'s, not Kovesi's. It is inactive on every non-constant image (smallest measured `T` across the five fixture images is `3.7e-3`, 37× the floor). Keep it; the fixture encodes it.
- **The `acos` argument is clipped to `[-1, 1]`** (drift M1). No reference clips. Count the clamps and assert the count is zero on all three shipped plates.
- **Operations are keyword-only pydantic models.** `FocusEdgeMonogenicPhase(n_scale=4)`, never positional. No hand-written `__init__`. Bounds go in `Field(...)`, search hints in `TuneSpec(...)`, and every `TuneSpec` window must be a subset of its `Field` bounds (`tests/unit/tune/test_annotation_subset_invariant.py` enforces this automatically).
- **`detect_mat ∈ [0, 1]`, float32 on assignment.** Angles must be mapped before writing (drift M2).
- **Never mutate `image.rgb` or `image.gray`.** The integrity validator on `ImageEnhancer.apply` enforces this.
- **Docstrings are Google-style, doctests must run** against `load_synth_yeast_plate()`.
- **Attribution:** the class docstring credits **Kovesi's `phasecongmono`**. It must *not* cite Wang Lijuan et al., CCDC 2014 — that paper was never read.

---

## File Structure

| File | Responsibility |
|---|---|
| **Create** `src/phenotypic/enhance/_monogenic_kernels.py` | Pure frequency-domain functions, no `Image` dependency. Owns `EPSILON_MONOGENIC`, the filter grids, log-Gabor radial, Riesz multiplier, periodic FFT, Rayleigh mode, spread weight, and `monogenic_phase_congruency()`. Unit-testable without fixtures. Shared by `FocusEdgePhase`, `FocusEdgeMonogenicPhase`, and later `FocusEdgeColorPhase`. |
| **Modify** `src/phenotypic/enhance/_focus_edge_phase.py` | Delete the four private helper methods; import them from `_monogenic_kernels`. Tighten `n_scale` to `ge=2`. `_phasecong3` **stays a method** — `detect/_filamentous_fungi_detector.py:424` calls it directly. `_compute_angular_spread` stays put. |
| **Create** `src/phenotypic/enhance/_focus_edge_monogenic_phase.py` | The `FocusEdgeMonogenicPhase(FocusEdge)` operation. Thin: field declarations, `_operate`, the angle→`[0,1]` map. |
| **Modify** `src/phenotypic/sdk_/typing_.py` | Add `MonogenicOutput` `TypeAlias` next to `FootprintShape`/`DetectMode`. |
| **Modify** `src/phenotypic/enhance/__init__.py` | Export `FocusEdgeMonogenicPhase` (import + `__all__`). This is what puts it in the GUI builder dropdown — `gui/_operation_registry.py` discovers by scanning the `phenotypic.enhance` module. |
| **Move** `docs/superpowers/specs/2026-07-08-alt-phase-detection/golden_phasecongmono.npz` → `tests/fixtures/phasecongmono_golden.npz` | One canonical copy. `verify_claims.py::check_19` resolves it by walking up from `__file__`. |
| **Modify** `docs/superpowers/specs/2026-07-08-alt-phase-detection/verify_claims.py` | Fixture path resolution only. |
| **Create** `tests/unit/enhance/test_monogenic_kernels.py` | Unit tests for every pure helper. |
| **Create** `tests/unit/enhance/_kovesi_synthetic.py` | Kovesi's four MIT test-image generators, lifted from `verify_claims.py` with the notice. |
| **Create** `tests/unit/enhance/test_focus_edge_monogenic_phase.py` | Golden fixture, operation contract, behavioural controls, axis convention. |
| **Modify** `tests/unit/abc_/test_enhancer_taxonomy.py` | Add `"FocusEdgeMonogenicPhase"` to the `FocusEdge` tuple. |
| **Modify** `tests/unit/tune/test_enhance_annotations.py` | Add annotation-resolution cases. |
| **Modify** `src/phenotypic/enhance/CLAUDE.md` | One line documenting `_monogenic_kernels.py`. |

---

### Task 1: `_monogenic_kernels.py` — pure helpers

Extract the four helpers `FocusEdgePhase` already owns, and add the three new pure functions the monogenic path needs. `_focus_edge_phase.py` is **not** touched in this task; it keeps its private copies. That keeps this task independently revertible.

**Files:**
- Create: `src/phenotypic/enhance/_monogenic_kernels.py`
- Test: `tests/unit/enhance/test_monogenic_kernels.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `EPSILON_MONOGENIC: float = 1e-4`
  - `construct_filter_grids(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` → `(radius, sintheta, costheta, freq)`
  - `lowpass_filter(radius: np.ndarray, cutoff: float = 0.45, order: int = 15) -> np.ndarray`
  - `log_gabor_scale(radius: np.ndarray, lowpass: np.ndarray, wavelength: float, sigma_onf: float) -> np.ndarray`
  - `log_gabor_radial(radius: np.ndarray, n_scale: int, min_wavelength: float, mult: float, sigma_onf: float) -> list[np.ndarray]`
  - `riesz_multiplier(sintheta: np.ndarray, costheta: np.ndarray) -> np.ndarray`
  - `periodic_fft2(img: np.ndarray) -> np.ndarray`
  - `rayleigh_mode(amplitude: np.ndarray) -> float`
  - `spread_weight(sum_amplitude, max_amplitude, n_scale: int, cutoff: float, g: float, epsilon: float) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/enhance/test_monogenic_kernels.py`:

```python
"""Unit tests for the shared frequency-domain kernels.

These functions are pure: no ``Image``, no fixtures, no I/O. Every assertion here
is a property of the maths, checkable by hand.
"""

import numpy as np
import pytest
from numpy.fft import fft2, ifft2

from phenotypic.enhance._monogenic_kernels import (
    EPSILON_MONOGENIC,
    construct_filter_grids,
    log_gabor_radial,
    log_gabor_scale,
    lowpass_filter,
    periodic_fft2,
    rayleigh_mode,
    riesz_multiplier,
    spread_weight,
)


def test_epsilon_is_kovesis_phasecongmono_value_not_phasecong3s():
    """1e-4 in all three phasecongmono references. 1e-5 is Julia's phasecong3.

    Guards against someone "unifying" this with ``FocusEdgePhase``'s epsilon.
    Julia phasecongruency.jl:441, MATLAB phasecongmono.m:153, phasepack:129.
    """
    assert EPSILON_MONOGENIC == 1e-4


class TestConstructFilterGrids:
    def test_dc_is_at_the_corner_and_radius_is_fudged_to_one(self):
        radius, sintheta, costheta, freq = construct_filter_grids(8, 8)
        assert freq[0, 0] == 0.0
        assert radius[0, 0] == 1.0  # so log(radius) never sees zero
        assert sintheta[0, 0] == 0.0
        assert costheta[0, 0] == 0.0

    def test_even_axis_spans_minus_half_to_just_under_half(self):
        _, _, _, freq = construct_filter_grids(1, 8)
        # ifftshift puts DC first; the raw axis was -4/8 .. 3/8
        assert freq.max() == pytest.approx(0.5)

    def test_odd_axis_divides_by_n_not_n_minus_one(self):
        """Both of Kovesi's implementations divide an odd axis by N (frequencyfilt.jl:73,
        filtergrid.m:49). phasepack divides by N-1, in filtergrid AND lowpassfilter -- a
        phasepack bug. k/N is the true DFT bin frequency. All three agree at even sizes."""
        _, _, _, freq = construct_filter_grids(1, 5)
        assert freq.max() == pytest.approx(2.0 / 5.0)  # not 0.5, which is k/(N-1)

    def test_sintheta_and_costheta_are_a_unit_vector_off_dc(self):
        _, sintheta, costheta, _ = construct_filter_grids(16, 16)
        norm = np.hypot(sintheta, costheta)
        assert np.allclose(norm[1:, 1:], 1.0)


class TestLogGabor:
    def test_dc_is_zeroed_at_every_scale(self):
        radius, _, _, _ = construct_filter_grids(32, 32)
        for filt in log_gabor_radial(radius, 4, 3.0, 2.1, 0.55):
            assert filt[0, 0] == 0.0

    def test_returns_one_filter_per_scale(self):
        radius, _, _, _ = construct_filter_grids(32, 32)
        assert len(log_gabor_radial(radius, 5, 3.0, 2.1, 0.55)) == 5

    def test_each_scale_peaks_at_its_own_centre_frequency(self):
        radius, _, _, _ = construct_filter_grids(256, 256)
        filters = log_gabor_radial(radius, 4, 3.0, 2.1, 0.55)
        for s, filt in enumerate(filters):
            f0 = 1.0 / (3.0 * 2.1**s)
            peak_radius = radius[np.unravel_index(np.argmax(filt), filt.shape)]
            assert peak_radius == pytest.approx(f0, rel=0.05)

    def test_log_gabor_radial_is_log_gabor_scale_per_scale(self):
        radius, _, _, _ = construct_filter_grids(32, 32)
        lp = lowpass_filter(radius)
        expected = [log_gabor_scale(radius, lp, 3.0 * 2.1**s, 0.55) for s in range(3)]
        actual = log_gabor_radial(radius, 3, 3.0, 2.1, 0.55)
        for e, a in zip(expected, actual):
            assert np.array_equal(e, a)


class TestRieszMultiplier:
    def test_packs_the_two_odd_channels_into_one_complex_array(self):
        _, sintheta, costheta, _ = construct_filter_grids(16, 16)
        riesz = riesz_multiplier(sintheta, costheta)
        assert np.array_equal(riesz.real, -costheta)
        assert np.array_equal(riesz.imag, sintheta)

    def test_dc_bin_is_zero(self):
        _, sintheta, costheta, _ = construct_filter_grids(16, 16)
        assert riesz_multiplier(sintheta, costheta)[0, 0] == 0

    def test_magnitude_is_one_off_dc(self):
        _, sintheta, costheta, _ = construct_filter_grids(16, 16)
        riesz = riesz_multiplier(sintheta, costheta)
        assert np.allclose(np.abs(riesz)[1:, 1:], 1.0)


class TestPeriodicFft2:
    def test_dc_bin_is_preserved_exactly(self):
        """The mean belongs to the periodic component: S[0,0] is forced to 0."""
        rng = np.random.default_rng(0)
        img = rng.normal(size=(16, 16)) + 5.0
        assert periodic_fft2(img)[0, 0] == pytest.approx(fft2(img)[0, 0])

    def test_a_constant_image_has_no_smooth_component(self):
        img = np.full((16, 16), 0.7)
        assert np.allclose(periodic_fft2(img), fft2(img))

    def test_it_removes_the_border_discontinuity_of_a_step_edge(self):
        """A step edge tiles badly: fft2 sees a second, spurious edge at the wrap."""
        img = np.zeros((32, 32))
        img[:, 16:] = 1.0
        periodic = np.real(ifft2(periodic_fft2(img)))
        smooth = img - periodic
        assert np.abs(smooth).max() > 0.1  # the wrap jump lives in `smooth`
        assert np.abs(periodic + smooth - img).max() < 1e-12


class TestRayleighMode:
    def test_recovers_the_scale_of_a_rayleigh_sample(self):
        """A 50-bin histogram mode is biased low against Rayleigh's long tail. Measured
        relative error across seeds 0-5: 0.015 to 0.092. The tolerance has headroom."""
        rng = np.random.default_rng(3)
        sigma = 2.0
        sample = sigma * np.sqrt(rng.normal(size=200_000) ** 2 + rng.normal(size=200_000) ** 2)
        assert rayleigh_mode(sample) == pytest.approx(sigma, rel=0.15)

    def test_all_zero_input_returns_zero(self):
        assert rayleigh_mode(np.zeros((4, 4))) == 0.0


class TestSpreadWeight:
    def test_a_single_active_scale_gives_width_zero(self):
        """One non-zero component => width 0 => the narrow-band penalty applies."""
        sum_amp = np.array([[1.0]])
        max_amp = np.array([[1.0]])
        weight = spread_weight(sum_amp, max_amp, 4, 0.5, 10.0, EPSILON_MONOGENIC)
        assert weight[0, 0] == pytest.approx(1.0 / (1.0 + np.exp(10.0 * 0.5)), rel=1e-3)

    def test_all_scales_equal_gives_width_one(self):
        sum_amp = np.array([[4.0]])
        max_amp = np.array([[1.0]])
        weight = spread_weight(sum_amp, max_amp, 4, 0.5, 10.0, EPSILON_MONOGENIC)
        assert weight[0, 0] == pytest.approx(1.0 / (1.0 + np.exp(10.0 * -0.5)), rel=1e-3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/enhance/test_monogenic_kernels.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'phenotypic.enhance._monogenic_kernels'`

- [ ] **Step 3: Write the module**

Create `src/phenotypic/enhance/_monogenic_kernels.py`:

```python
"""Shared frequency-domain kernels for the phase-congruency operations.

Pure functions with no :class:`~phenotypic._core._image.Image` dependency, so they
are unit-testable without fixtures. Used by :class:`FocusEdgePhase` (Kovesi's
``phasecong3``) and :class:`FocusEdgeMonogenicPhase` (Kovesi's ``phasecongmono``).

References:
    Peter Kovesi, ``ImagePhaseCongruency.jl`` (Julia) and ``MatlabFns/PhaseCongruency``
    (MATLAB). The MIT-licensed ``phasepack`` is a third, independent transcription.
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.fft import fft2, ifftshift

#: Division guard for ``phasecongmono``. All three references agree on ``1e-4``:
#: Julia ``phasecongruency.jl`` line 441, MATLAB ``phasecongmono.m`` line 153,
#: ``phasepack/phasecongmono.py`` line 129.
#:
#: **Do not unify this with** :class:`FocusEdgePhase`'s ``1e-5``. That value belongs
#: to *Julia's* ``phasecong3`` (MATLAB's ``phasecong3.m`` uses ``1e-4``), and
#: ``FocusEdgePhase`` is a port of the Julia.
EPSILON_MONOGENIC = 1e-4


def construct_filter_grids(
        rows: int, cols: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct frequency domain grids for filter construction.

    Grids are quadrant-shifted so the DC component is at ``[0, 0]``. Every axis is
    divided by ``N``, matching **both** of Kovesi's implementations
    (``frequencyfilt.jl`` l.73, ``filtergrid.m`` l.49). ``k/N`` is the true DFT bin
    frequency.

    ``phasepack`` differs at odd sizes -- ``linspace(-0.5, 0.5, N, endpoint=True)`` in
    its ``filtergrid``, and ``/(N - 1)`` in its ``lowpassfilter`` -- which is a
    ``phasepack`` bug, not a Kovesi divergence. All three agree at even sizes, which is
    why the golden fixture is 64x64.

    Args:
        rows: Number of rows in image.
        cols: Number of columns in image.

    Returns:
        Tuple of ``(radius, sintheta, costheta, freq)`` where:

        - ``radius``: radial frequency with ``DC = 1`` so ``log(radius)`` is safe
        - ``sintheta``: ``fx / freq``, the angular filter's sine grid
        - ``costheta``: ``fy / freq``, the angular filter's cosine grid
        - ``freq``: radial frequency with ``DC = 0``
    """
    if cols % 2 == 1:  # odd
        fx_range = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / cols
    else:  # even
        fx_range = np.arange(-cols / 2, cols / 2) / cols

    if rows % 2 == 1:  # odd
        fy_range = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / rows
    else:  # even
        fy_range = np.arange(-rows / 2, rows / 2) / rows

    fx_range = ifftshift(fx_range)
    fy_range = ifftshift(fy_range)

    fx, fy = np.meshgrid(fx_range, fy_range)

    freq = np.sqrt(fx ** 2 + fy ** 2)

    radius = freq.copy()
    radius[0, 0] = 1.0

    freq_safe = freq.copy()
    freq_safe[0, 0] = 1.0
    sintheta = fx / freq_safe
    costheta = fy / freq_safe

    sintheta[0, 0] = 0.0
    costheta[0, 0] = 0.0

    return radius, sintheta, costheta, freq


def lowpass_filter(radius: np.ndarray, cutoff: float = 0.45, order: int = 15) -> np.ndarray:
    """Butterworth lowpass, Kovesi's ``lowpassfilter(size, 0.45, 15)``.

    Args:
        radius: Radial frequency grid.
        cutoff: Normalized cutoff frequency.
        order: Butterworth order; the exponent is ``2 * order``.

    Returns:
        Lowpass transfer function, same shape as ``radius``.
    """
    return 1.0 / (1.0 + (radius / cutoff) ** (2 * order))


def log_gabor_scale(
        radius: np.ndarray, lowpass: np.ndarray, wavelength: float, sigma_onf: float
) -> np.ndarray:
    """One log-Gabor radial bandpass, lowpassed, with the DC bin zeroed.

    Args:
        radius: Radial frequency grid with ``DC = 1``.
        lowpass: Precomputed :func:`lowpass_filter` output.
        wavelength: Centre wavelength in pixels; the centre frequency is its reciprocal.
        sigma_onf: Ratio of the filter's Gaussian sigma to its centre frequency.

    Returns:
        The transfer function, same shape as ``radius``.
    """
    f0 = 1.0 / wavelength

    with np.errstate(divide="ignore", invalid="ignore"):
        log_rad_over_f0 = np.log(radius / f0)

    log_gabor = np.exp(-(log_rad_over_f0 ** 2) / (2 * np.log(sigma_onf) ** 2))
    log_gabor[0, 0] = 0.0

    return log_gabor * lowpass


def log_gabor_radial(
        radius: np.ndarray,
        n_scale: int,
        min_wavelength: float,
        mult: float,
        sigma_onf: float,
) -> List[np.ndarray]:
    """Construct log-Gabor filters for each scale.

    Log-Gabor filters have Gaussian transfer functions on a logarithmic frequency
    scale, providing a constant shape ratio across scales.

    Args:
        radius: Radial frequency grid with ``DC = 1``.
        n_scale: Number of scales.
        min_wavelength: Wavelength of the finest scale, in pixels.
        mult: Wavelength multiplier between successive scales.
        sigma_onf: Ratio of the filter's Gaussian sigma to its centre frequency.

    Returns:
        List of ``n_scale`` filter arrays.
    """
    lowpass = lowpass_filter(radius)
    return [
        log_gabor_scale(radius, lowpass, min_wavelength * (mult ** s), sigma_onf)
        for s in range(n_scale)
    ]


def riesz_multiplier(sintheta: np.ndarray, costheta: np.ndarray) -> np.ndarray:
    """Kovesi's ``packedmonogenicfilters``: ``H = (i*fx - fy)/freq``.

    Packs both odd (Riesz) channels into one complex array, so a single ``ifft2``
    yields ``h1`` in the real part and ``h2`` in the imaginary part.

    **Axis convention.** Swapping ``sintheta`` and ``costheta`` rotates every
    orientation by 90 degrees while leaving ``pc`` unchanged to ``1.5e-17``. The
    sign on ``costheta`` encodes a y-up convention; flipping it mirrors every
    orientation about the x-axis, which axis-aligned test edges cannot see. Both
    bugs are covered by the ``starsine`` test.

    Args:
        sintheta: ``fx / freq`` grid from :func:`construct_filter_grids`.
        costheta: ``fy / freq`` grid from :func:`construct_filter_grids`.

    Returns:
        Complex transfer function with a zero DC bin.
    """
    return 1j * sintheta - costheta


def periodic_fft2(img: np.ndarray) -> np.ndarray:
    """Moisan's periodic component of the FFT — Kovesi's ``perfft2``.

    ``fft2`` treats the image as tiled, so the intensity jump between opposite
    borders leaks a cross-shaped artifact into every frequency band. The
    periodic/smooth decomposition splits ``img = p + s`` with ``s`` carrying that
    jump, and this returns ``F(p)``.

    Kovesi's MATLAB ``phasecongmono.m`` line 156 uses it; his Julia line 446 does
    not (``IMG = fft(img)   # Use fft rather than perfft2``). The shipped operation
    follows the Julia. This exists so the golden fixture, generated from
    ``phasepack`` (a MATLAB transcription), remains reproducible.

    Args:
        img: Real 2-D array.

    Returns:
        Complex spectrum of the periodic component.
    """
    rows, cols = img.shape
    smooth = np.zeros_like(img, dtype=np.float64)
    smooth[0, :] = img[0, :] - img[-1, :]
    smooth[-1, :] = -smooth[0, :]
    smooth[:, 0] += img[:, 0] - img[:, -1]
    smooth[:, -1] -= img[:, 0] - img[:, -1]

    cx, cy = np.meshgrid(
            2 * np.pi * np.arange(cols) / cols,
            2 * np.pi * np.arange(rows) / rows,
    )
    denominator = 2.0 * (2.0 - np.cos(cx) - np.cos(cy))
    denominator[0, 0] = 1.0  # avoid /0

    smooth_fft = fft2(smooth) / denominator
    smooth_fft[0, 0] = 0.0  # the mean belongs to the periodic component

    return fft2(img) - smooth_fft


def rayleigh_mode(amplitude: np.ndarray) -> float:
    """Estimate the Rayleigh distribution parameter from amplitude data.

    For filter responses to Gaussian noise, amplitudes follow a Rayleigh
    distribution whose mode equals sigma.

    Args:
        amplitude: Array of amplitude values.

    Returns:
        Estimated Rayleigh sigma, or ``0.0`` if every value is non-positive.
    """
    amp_flat = amplitude.flatten()
    amp_flat = amp_flat[amp_flat > 0]

    if len(amp_flat) == 0:
        return 0.0

    n_bins = 50  # matches Julia
    hist, bin_edges = np.histogram(amp_flat, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mode_idx = np.argmax(hist)
    return float(bin_centers[mode_idx])


def spread_weight(
        sum_amplitude: np.ndarray,
        max_amplitude: np.ndarray,
        n_scale: int,
        cutoff: float,
        g: float,
        epsilon: float,
) -> np.ndarray:
    """Kovesi's sigmoidal frequency-spread weighting ``W``.

    Penalizes narrow frequency distributions. ``width`` is 0 when a single scale
    responds and 1 when all scales respond equally.

    Args:
        sum_amplitude: Sum of per-scale amplitudes.
        max_amplitude: Elementwise max of the per-scale amplitudes.
        n_scale: Number of scales; must be at least 2 (the divisor is ``n_scale - 1``).
        cutoff: Fractional width below which the weight is penalized.
        g: Sharpness of the sigmoid.
        epsilon: Division guard. ``1e-5`` for ``phasecong3``, ``1e-4`` for
            ``phasecongmono`` — the callers differ, so it is a parameter.

    Returns:
        Weight array in ``(0, 1)``.
    """
    width = (sum_amplitude / (max_amplitude + epsilon) - 1) / (n_scale - 1)
    return 1.0 / (1.0 + np.exp((cutoff - width) * g))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/enhance/test_monogenic_kernels.py -q`
Expected: all pass (26 tests).

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check --fix src/phenotypic/enhance/_monogenic_kernels.py tests/unit/enhance/test_monogenic_kernels.py && uv run mypy src/phenotypic/enhance/_monogenic_kernels.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/enhance/_monogenic_kernels.py tests/unit/enhance/test_monogenic_kernels.py
git commit -m "feat(enhance): add shared frequency-domain kernels for phase congruency"
```

---

### Task 2: Refactor `FocusEdgePhase` onto the kernels

Bit-identical behaviour is the whole requirement. `_phasecong3` and `_compute_angular_spread` stay methods — `detect/_filamentous_fungi_detector.py:424` calls `FocusEdgePhase(...)._phasecong3(arr)` directly.

Also fixes the latent `n_scale=1` bug: `_phasecong3` divides by `n_scale - 1` (line 320) and silently returns an all-zero `detect_mat`. Tighten to `ge=2`. No test or caller uses `n_scale=1`.

**Files:**
- Modify: `src/phenotypic/enhance/_focus_edge_phase.py`
- Test: `tests/unit/enhance/test_phase_congruency.py` (must pass **unchanged**)

**Interfaces:**
- Consumes: Task 1's `construct_filter_grids`, `log_gabor_radial`, `rayleigh_mode`, `spread_weight`.
- Produces: nothing new. `FocusEdgePhase._phasecong3` keeps its signature and returns `_PhaseCong3Result`.

- [ ] **Step 1: Capture the pre-refactor output as a bit-identity baseline**

The existing test file asserts properties, not exact arrays. Capture the actual numbers first, or the refactor is unverified.

```bash
uv run python - <<'PY'
import numpy as np
from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import FocusEdgePhase
img = np.asarray(load_synth_yeast_plate().detect_mat[:], dtype=np.float64)
r = FocusEdgePhase()._phasecong3(img)
np.savez("/tmp/phasecong3_baseline.npz", M=r.M, m=r.m, orientation=r.orientation,
         feature_type=r.feature_type, T=np.array(r.T), pc_sum=r.pc_sum)
print("baseline saved; pc_sum.max =", r.pc_sum.max())
PY
```
Expected: `baseline saved; pc_sum.max = ...` (a float around 0.33).

- [ ] **Step 2: Run the existing test file to confirm it is green before you touch anything**

Run: `uv run pytest tests/unit/enhance/test_phase_congruency.py -q`
Expected: all pass.

- [ ] **Step 3: Replace the four helper methods with imports**

In `src/phenotypic/enhance/_focus_edge_phase.py`:

Change the imports at the top:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
from numpy.fft import fft2, ifft2
from pydantic import Field

from ._monogenic_kernels import (
    construct_filter_grids,
    log_gabor_radial,
    rayleigh_mode,
    spread_weight,
)
from ..abc_ import FocusEdge
from ..sdk_.typing_ import TuneSpec
```

(`ifftshift` and `List` are no longer used here — remove them.)

Tighten the field:

```python
    # ge=2: _phasecong3 divides by (n_scale - 1). At n_scale=1 that is a
    # divide-by-zero which silently yields an all-zero detect_mat.
    n_scale: Annotated[int, TuneSpec(3, 6)] = Field(4, ge=2)
```

**Delete** the methods `_construct_filter_grids`, `_construct_log_gabor_filters`, `_rayleigh_mode` (they run from roughly line 379 to the end of the class), and replace their three call sites inside `_phasecong3`:

```python
        radius, sintheta, costheta, freq = construct_filter_grids(rows, cols)

        log_gabor_list = log_gabor_radial(
                radius, self.n_scale, self.min_wavelength, self.mult, self.sigma_onf
        )
```

```python
                    elif abs(self.noise_method + 2) < epsilon:
                        # Mode-based Rayleigh estimation
                        tau = rayleigh_mode(amplitude)
```

And replace the inline `width`/`weight` block:

```python
            weight = spread_weight(
                    sum_amplitude, max_amplitude, self.n_scale, self.cutoff, self.g, epsilon
            )
```

`_compute_angular_spread` stays exactly where it is.

- [ ] **Step 4: Verify bit-identity against the baseline**

```bash
uv run python - <<'PY'
import numpy as np
from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import FocusEdgePhase
base = np.load("/tmp/phasecong3_baseline.npz")
img = np.asarray(load_synth_yeast_plate().detect_mat[:], dtype=np.float64)
r = FocusEdgePhase()._phasecong3(img)
now = {"M": r.M, "m": r.m, "orientation": r.orientation,
       "feature_type": r.feature_type, "T": np.array(r.T), "pc_sum": r.pc_sum}
for key, want in base.items():
    assert np.array_equal(want, now[key]), f"{key} changed: max|d| = {np.abs(want-now[key]).max():.3e}"
print("BIT-IDENTICAL across all six outputs")
PY
```
Expected: `BIT-IDENTICAL across all six outputs`. Anything else means the refactor changed the algorithm — fix it, do not adjust the check.

- [ ] **Step 5: Run the existing test file, the taxonomy test, and the filamentous detector's tests**

Run: `uv run pytest tests/unit/enhance/test_phase_congruency.py tests/unit/abc_/test_enhancer_taxonomy.py tests/unit/tune/test_enhance_annotations.py -q`
Expected: all pass, unchanged.

Run: `uv run pytest tests/unit/detect -q -k filamentous`
Expected: all pass (this exercises the `._phasecong3` caller).

- [ ] **Step 6: Add the `n_scale >= 2` regression test**

Append to `tests/unit/enhance/test_phase_congruency.py`, inside `TestPhaseCongruencyEnhancerParameterValidation`:

```python
    def test_n_scale_one_is_rejected(self):
        """n_scale=1 divides by (n_scale - 1) and silently returns an all-zero
        detect_mat. Rejected at construction rather than producing garbage."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(n_scale=1)

    def test_n_scale_two_is_accepted(self):
        assert FocusEdgePhase(n_scale=2).n_scale == 2
```

- [ ] **Step 7: Run the tests, lint, type-check**

Run: `uv run pytest tests/unit/enhance/test_phase_congruency.py -q && uv run ruff check --fix src/phenotypic/enhance/_focus_edge_phase.py && uv run mypy src/phenotypic`
Expected: pass, clean.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/enhance/_focus_edge_phase.py tests/unit/enhance/test_phase_congruency.py
git commit -m "refactor(enhance): move FocusEdgePhase's frequency helpers into _monogenic_kernels

Bit-identical on load_synth_yeast_plate across all six _phasecong3 outputs.
Also tightens n_scale to ge=2: _phasecong3 divides by (n_scale - 1), so
n_scale=1 silently returned an all-zero detect_mat."
```

---

### Task 3: `monogenic_phase_congruency()` and the golden fixture

The algorithm itself, driven by the fixture. `phasepack` is **not** reinstalled — the fixture is the reference now.

**Files:**
- Modify: `src/phenotypic/enhance/_monogenic_kernels.py`
- Move: `docs/superpowers/specs/2026-07-08-alt-phase-detection/golden_phasecongmono.npz` → `tests/fixtures/phasecongmono_golden.npz`
- Modify: `docs/superpowers/specs/2026-07-08-alt-phase-detection/verify_claims.py`
- Create: `tests/unit/enhance/_kovesi_synthetic.py`
- Test: `tests/unit/enhance/test_monogenic_kernels.py` (append)

**Interfaces:**
- Consumes: everything from Task 1.
- Produces:
  - `MonogenicResult` frozen dataclass with fields `pc: np.ndarray`, `orientation: np.ndarray` (radians, `(-pi/2, pi/2]`), `feature_type: np.ndarray` (radians, `[-pi/2, pi/2]`), `threshold: float`, `n_clamped: int`
  - `monogenic_phase_congruency(img, *, n_scale=4, min_wavelength=3.0, mult=2.1, sigma_onf=0.55, k=3.0, cutoff=0.5, g=10.0, deviation_gain=1.5, noise_method=-1.0, periodic=False) -> MonogenicResult`
  - `tests/unit/enhance/_kovesi_synthetic.py` exporting `step2line`, `circsine`, `starsine`, `noiseonf`, `unit_variance`, `centred_axis`

- [ ] **Step 1: Move the fixture to its canonical home**

```bash
mkdir -p tests/fixtures
git mv docs/superpowers/specs/2026-07-08-alt-phase-detection/golden_phasecongmono.npz tests/fixtures/phasecongmono_golden.npz
```

In `docs/superpowers/specs/2026-07-08-alt-phase-detection/verify_claims.py`, inside `check_19_golden_fixture_agrees_with_phasepack`, replace:

```python
    path = Path(__file__).with_name("golden_phasecongmono.npz")
```

with:

```python
    path = None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "tests" / "fixtures" / "phasecongmono_golden.npz"
        if candidate.exists():
            path = candidate
            break
    if path is None:
        return Result("19 golden fixture agrees with phasepack (rtol=1e-6)", True,
                      "SKIPPED: tests/fixtures/phasecongmono_golden.npz absent")
```

and delete the now-dead `if not path.exists():` block below it.

Update the README table entry's path:

```bash
sed -i '' 's#\[`golden_phasecongmono.npz`\](./golden_phasecongmono.npz)#[`tests/fixtures/phasecongmono_golden.npz`](../../../../tests/fixtures/phasecongmono_golden.npz)#' docs/superpowers/specs/2026-07-08-alt-phase-detection/README.md
```

Run: `uv run python docs/superpowers/specs/2026-07-08-alt-phase-detection/verify_claims.py | tail -1`
Expected: `21/21 checks passed.`

- [ ] **Step 2: Port Kovesi's generators into the test helper**

Create `tests/unit/enhance/_kovesi_synthetic.py`. Copy the four generators, the MIT notice, and `unit_variance` verbatim from `docs/superpowers/specs/2026-07-08-alt-phase-detection/verify_claims.py` (the section headed "Kovesi's synthetic test images"). Rename its private `_centred_axis` to a public `centred_axis`, and import `filter_grid` from `phenotypic.enhance._monogenic_kernels.construct_filter_grids` instead of re-porting it:

```python
"""Peter Kovesi's synthetic test images, for testing phase-congruency operators.

Ported from ``src/syntheticimages.jl`` of ImagePhaseCongruency.jl. The originals
describe themselves as images that "cause considerable grief for gradient based
operators", which is exactly why they are the right controls for a congruency
measure: an independent, adversarial ground truth authored by the algorithm's own
author rather than by us.

  Copyright (c) 2015-2017 Peter Kovesi -- peterkovesi.com

  MIT License:

  Permission is hereby granted, free of charge, to any person obtaining a copy of
  this software and associated documentation files (the "Software"), to deal in the
  Software without restriction, including without limitation the rights to use, copy,
  modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
  and to permit persons to whom the Software is furnished to do so, subject to the
  following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
  PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
  CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
  OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Faithfulness notes, in the order they bite:

  * Only ODD harmonics are summed (``1:2:(2*nscales-1)``). Even harmonics would break
    the half-wave symmetry that makes the feature type well defined.
  * ``ampexponent = -1`` sums ``1/k`` -> a square wave (step features). ``-2`` with
    ``offset = pi/2`` sums ``cos(k x)/k^2`` -> a triangle wave (line features).
  * Julia's ``[f(x, y) for x = l:u, y = l:u]`` puts ``x`` on the FIRST axis, so these
    use ``indexing="ij"`` and ``theta = arctan2(Y, X)`` with ``X`` the row coordinate.
    Getting this backwards transposes every image.
  * ``circsine``'s ``trim`` option is not ported: in the original it multiplies by
    ``(r < c) + (r >= c)``, which is identically 1.
"""

from __future__ import annotations

import numpy as np

from phenotypic.enhance._monogenic_kernels import construct_filter_grids


def centred_axis(sze: int) -> np.ndarray:
    """Kovesi's ``l:u``: ``-sze/2 : sze/2-1`` when even, ``-(sze-1)/2 : (sze-1)/2`` when odd."""
    if sze % 2 == 0:
        return np.arange(-sze // 2, sze // 2, dtype=float)
    return np.arange(-(sze - 1) // 2, (sze - 1) // 2 + 1, dtype=float)


def step2line(sze: int = 512, *, nscales: int = 50, ampexponent: float = -1.0,
              ncycles: float = 1.5, phasecycles: float = 0.25) -> np.ndarray:
    """A phase-congruent image whose FEATURE TYPE sweeps step -> line down the rows.

    Every row is the same odd-harmonic series with a growing phase offset. The
    congruency points sit at fixed columns ``x = m*pi`` for every row: there each odd
    harmonic ``sin(k x + phi)`` has phase ``m*pi + phi``, so they align whatever ``phi``
    is. Congruency is therefore constant down the image while the feature morphs from a
    step into a line. Gradient magnitude is not.
    """
    x = np.arange(sze) / (sze - 1) * ncycles * 2 * np.pi
    offsets = phasecycles * 2 * np.pi * np.arange(sze) / sze
    img = np.zeros((sze, sze))
    for scale in range(1, 2 * nscales, 2):  # ODD harmonics only
        img += float(scale) ** ampexponent * np.sin(scale * x[None, :] + offsets[:, None])
    return img


def circsine(sze: int = 512, *, wavelength: float = 40.0, nscales: int = 50,
             ampexponent: float = -1.0, offset: float = 0.0, p: int = 2) -> np.ndarray:
    """Concentric circular waveform. Isophotes are exact circles: curvature ``1/r``, always."""
    if p % 2:
        raise ValueError("p should be an even number")
    ax = centred_axis(sze)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    r = (X ** p + Y ** p) ** (1.0 / p)
    img = np.zeros_like(r)
    for scale in range(1, 2 * nscales, 2):
        img += float(scale) ** ampexponent * np.sin(scale * r * 2 * np.pi / wavelength + offset)
    return img


def starsine(sze: int = 512, *, ncycles: float = 10.0, nscales: int = 50,
             ampexponent: float = -1.0, offset: float = 0.0) -> np.ndarray:
    """An angular waveform: radial rays at every orientation at once."""
    ax = centred_axis(sze)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    theta = np.arctan2(Y, X)  # Julia: atan(y, x), y on the SECOND axis
    img = np.zeros_like(theta)
    for scale in range(1, 2 * nscales, 2):
        img += float(scale) ** ampexponent * np.sin(scale * ncycles * theta + offset)
    return img


def noiseonf(sze: int, p: float, *, seed: int = 0) -> np.ndarray:
    """``1/f^p`` noise: random phase, amplitude spectrum replaced by ``1/radius^p``.

    The phase spectrum is pure noise, so there is no congruency anywhere -- the negative
    control for the Rayleigh threshold ``T``. ``p = 1.5`` is roughly the amplitude
    falloff of natural images.
    """
    rng = np.random.default_rng(seed)
    spectrum = np.fft.fft2(rng.normal(size=(sze, sze)))
    magnitude = np.abs(spectrum)
    magnitude[magnitude == 0.0] = 1.0
    radius = construct_filter_grids(sze, sze)[3] * sze + 1.0
    return np.real(np.fft.ifft2((spectrum / magnitude) / radius ** p))


def unit_variance(img: np.ndarray) -> np.ndarray:
    """Zero mean, unit standard deviation. ``epsilon`` is absolute, so scale matters."""
    return (img - img.mean()) / img.std()
```

- [ ] **Step 3: Write the failing tests for `monogenic_phase_congruency`**

First **extend the import block at the top** of `tests/unit/enhance/test_monogenic_kernels.py` — do not add imports mid-file, `ruff` flags that as `E402`:

```python
from pathlib import Path

from phenotypic.enhance._monogenic_kernels import (
    EPSILON_MONOGENIC,
    construct_filter_grids,
    log_gabor_radial,
    log_gabor_scale,
    lowpass_filter,
    monogenic_phase_congruency,
    periodic_fft2,
    rayleigh_mode,
    riesz_multiplier,
    spread_weight,
)

from ._kovesi_synthetic import circsine, noiseonf, starsine, step2line, unit_variance

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "phasecongmono_golden.npz"

_GOLDEN_PARAMS = dict(
    n_scale=4, min_wavelength=3.0, mult=2.1, sigma_onf=0.55,
    k=3.0, cutoff=0.5, g=10.0, deviation_gain=1.5,
)


def _golden_cases() -> dict[str, np.ndarray]:
    n = 64
    step = np.zeros((n, n))
    step[:, n // 2:] = 1.0
    return {
        "step": step,
        "step2line": unit_variance(step2line(n)),
        "starsine": unit_variance(starsine(n, ncycles=8)),
        "circsine": unit_variance(circsine(n, wavelength=16.0)),
        "noiseonf": unit_variance(noiseonf(n, 1.5, seed=1)),
    }


class TestGoldenFixture:
    """Numeric agreement with phasepack 1.5, an independent transcription of Kovesi's MATLAB.

    Generated once under an approved install, then the dependency was dropped. The fixture
    pins the ``periodic=True`` branch, because that is what phasepack computes; the shipped
    operation defaults to ``periodic=False``, following Kovesi's Julia. The branches differ
    only in which spectrum enters an otherwise identical chain, so this validates the chain.

    Note: phasepack ships no tests of its own, so this pins *transcription*, not
    *correctness*. The behavioural controls below are what speak to correctness.
    """

    @pytest.mark.parametrize("name", ["step", "step2line", "starsine", "circsine", "noiseonf"])
    def test_pc_matches_phasepack(self, name):
        golden = np.load(_FIXTURE, allow_pickle=False)
        img = _golden_cases()[name]
        result = monogenic_phase_congruency(img, periodic=True, **_GOLDEN_PARAMS)
        np.testing.assert_allclose(result.pc, golden[f"{name}__pc"], rtol=1e-6, atol=1e-9)

    @pytest.mark.parametrize("name", ["step", "step2line", "starsine", "circsine", "noiseonf"])
    def test_feature_type_matches_phasepack(self, name):
        golden = np.load(_FIXTURE, allow_pickle=False)
        img = _golden_cases()[name]
        result = monogenic_phase_congruency(img, periodic=True, **_GOLDEN_PARAMS)
        np.testing.assert_allclose(
                result.feature_type, golden[f"{name}__ft"], rtol=1e-6, atol=1e-9
        )

    @pytest.mark.parametrize("name", ["step", "step2line", "starsine", "circsine", "noiseonf"])
    def test_threshold_matches_phasepack(self, name):
        golden = np.load(_FIXTURE, allow_pickle=False)
        img = _golden_cases()[name]
        result = monogenic_phase_congruency(img, periodic=True, **_GOLDEN_PARAMS)
        assert result.threshold == pytest.approx(float(golden[f"{name}__T"]), rel=1e-6)

    def test_the_fixture_is_load_bearing(self):
        """A fixture that cannot fail proves nothing.

        Regressing to the ``fft2`` branch — which no behavioural test in this repo
        detects — must break this fixture. Measured drift: 0.67 absolute.
        """
        golden = np.load(_FIXTURE, allow_pickle=False)
        img = _golden_cases()["step2line"]
        wrong_branch = monogenic_phase_congruency(img, periodic=False, **_GOLDEN_PARAMS)
        assert np.abs(wrong_branch.pc - golden["step2line__pc"]).max() > 0.1


class TestNoiseThreshold:
    def test_noise_method_zero_disables_the_threshold(self):
        img = unit_variance(step2line(64))
        assert monogenic_phase_congruency(img, noise_method=0.0).threshold == 0.0

    def test_noise_method_positive_is_used_verbatim(self):
        img = unit_variance(step2line(64))
        assert monogenic_phase_congruency(img, noise_method=0.25).threshold == 0.25

    def test_threshold_is_floored_at_epsilon_on_a_constant_image(self):
        """phasepack's floor, not Kovesi's. Only a constant image ever reaches it."""
        result = monogenic_phase_congruency(np.full((64, 64), 0.5))
        assert result.threshold == EPSILON_MONOGENIC

    def test_rayleigh_mode_path_is_reachable(self):
        img = unit_variance(noiseonf(64, 1.5, seed=1))
        assert monogenic_phase_congruency(img, noise_method=-2.0).threshold > 0.0


class TestAcosClampIsInert:
    """Drift M1. No reference clamps; roundoff can push the ratio above 1 and yield NaN.

    We clamp, and assert the clamp never fires on any real plate.
    """

    @pytest.mark.parametrize("loader_name", ["load_synth_yeast_plate", "load_yeast_plate", "load_fungi_plate"])
    def test_clamp_never_fires_on_a_shipped_plate(self, loader_name):
        import phenotypic.data as data
        img = getattr(data, loader_name)()
        mat = np.asarray(img.detect_mat[:], dtype=np.float64)
        assert monogenic_phase_congruency(mat).n_clamped == 0

    def test_no_nans_or_infs_in_the_output(self):
        from phenotypic.data import load_synth_yeast_plate
        mat = np.asarray(load_synth_yeast_plate().detect_mat[:], dtype=np.float64)
        result = monogenic_phase_congruency(mat)
        assert np.isfinite(result.pc).all()
        assert np.isfinite(result.orientation).all()
        assert np.isfinite(result.feature_type).all()


class TestContrastAndIlluminationInvariance:
    """The defining property of phase congruency: pc(a*f + b) == pc(f).

    Exact in exact arithmetic. The residual is entirely ``EPSILON_MONOGENIC``, which is
    absolute and therefore not 1-homogeneous. Measured worst case across the three shipped
    plates and (a, b) in {(3, 7), (0.5, -0.2), (10, 0)}: max |dpc| = 3.8e-2, mean = 1.6e-4.
    """

    @pytest.mark.parametrize("a, b", [(3.0, 7.0), (0.5, -0.2), (10.0, 0.0)])
    def test_pc_is_invariant_to_an_affine_intensity_change(self, a, b):
        from phenotypic.data import load_synth_yeast_plate
        mat = np.asarray(load_synth_yeast_plate().detect_mat[:], dtype=np.float64)
        base = monogenic_phase_congruency(mat).pc
        shifted = monogenic_phase_congruency(a * mat + b).pc
        drift = np.abs(shifted - base)
        assert drift.max() < 0.05
        assert drift.mean() < 1e-3

    def test_a_pure_dc_offset_is_removed_exactly(self):
        """The log-Gabor DC bin is zeroed, so ``+b`` cannot survive the bandpass."""
        from phenotypic.data import load_synth_yeast_plate
        mat = np.asarray(load_synth_yeast_plate().detect_mat[:], dtype=np.float64)
        base = monogenic_phase_congruency(mat).pc
        offset = monogenic_phase_congruency(mat + 7.0).pc
        assert np.abs(offset - base).max() < 1e-12
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/enhance/test_monogenic_kernels.py -q -k "Golden or Noise or Acos or Contrast"`
Expected: collection error — `ImportError: cannot import name 'monogenic_phase_congruency'`

- [ ] **Step 5: Implement `monogenic_phase_congruency`**

Append to `src/phenotypic/enhance/_monogenic_kernels.py` (and add `from dataclasses import dataclass` plus `ifft2` to the imports):

```python
@dataclass(frozen=True)
class MonogenicResult:
    """Output of :func:`monogenic_phase_congruency`.

    Attributes:
        pc: Phase congruency in ``[0, 1]``. High where the log-Gabor components are
            maximally in phase, independent of their amplitude.
        orientation: Feature orientation in radians, ``(-pi/2, pi/2]``. ``0`` is a
            vertical edge (intensity varying across columns), ``pi/2`` a horizontal one.
            Measured with y increasing upward — that is what the sign on ``sum_h2``
            encodes.
        feature_type: Local weighted mean phase angle in radians, ``[-pi/2, pi/2]``.
            ``0`` is a step edge, ``+pi/2`` a bright line, ``-pi/2`` a dark line.
        threshold: The Rayleigh noise threshold ``T`` actually applied.
        n_clamped: How many pixels needed the ``acos`` argument clipped into
            ``[-1, 1]``. Must be ``0``; a non-zero value means roundoff escaped the
            ``epsilon`` guard.
    """

    pc: np.ndarray
    orientation: np.ndarray
    feature_type: np.ndarray
    threshold: float
    n_clamped: int


def monogenic_phase_congruency(
        img: np.ndarray,
        *,
        n_scale: int = 4,
        min_wavelength: float = 3.0,
        mult: float = 2.1,
        sigma_onf: float = 0.55,
        k: float = 3.0,
        cutoff: float = 0.5,
        g: float = 10.0,
        deviation_gain: float = 1.5,
        noise_method: float = -1.0,
        periodic: bool = False,
) -> MonogenicResult:
    """Kovesi's ``phasecongmono``: phase congruency from the monogenic signal.

    An isotropic log-Gabor bandpass supplies the even channel; the Riesz transform
    supplies the two odd channels. There is no orientation sweep -- orientation falls
    out of the odd pair::

        PC = W * max(1 - deviation_gain*acos(E/(sumAn + eps)), 0) * max(E - T, 0)/(E + eps)

    This is **not** ``phasecong3``'s formula. The noise threshold is applied as a
    multiplicative fraction rather than subtracted from the numerator (Kovesi: subtracting
    it early "would interfere with the phase deviation computation"), and the phase
    deviation term is ``acos(E/sumAn)`` scaled by ``deviation_gain``.

    Args:
        img: Real 2-D array. Not required to lie in any particular range.
        n_scale: Number of log-Gabor scales. Must be at least 2.
        min_wavelength: Wavelength of the finest scale, in pixels.
        mult: Wavelength multiplier between successive scales.
        sigma_onf: Ratio of each filter's Gaussian sigma to its centre frequency.
        k: Number of noise standard deviations above the mean at which ``T`` is set.
            ``phasecongmono``'s default is ``3.0``, not ``phasecong3``'s ``2.0``.
        cutoff: Fractional frequency-spread below which ``W`` penalizes the response.
        g: Sharpness of ``W``'s sigmoid.
        deviation_gain: Scales the phase-deviation term. Kovesi: "sensible values are
            from 1 to about 2."
        noise_method: ``-1`` estimates the Rayleigh parameter from the median of the
            finest scale's amplitude; ``-2`` from its histogram mode; any value ``>= 0``
            is used verbatim as ``T`` (so ``0.0`` disables thresholding).
        periodic: Bandpass the image's periodic component (Moisan's decomposition,
            Kovesi's ``perfft2``) rather than the raw FFT. Kovesi's MATLAB does this;
            his Julia explicitly does not, and we follow the Julia. **Leave this
            ``False``** except when reproducing the golden fixture, which was generated
            from ``phasepack`` (a MATLAB transcription).

    Returns:
        A :class:`MonogenicResult`.

    References:
        Kovesi, P. "Image features from phase congruency." *Videre* 1(3), 1--26 (1999).
    """
    img = np.asarray(img, dtype=np.float64)
    rows, cols = img.shape
    epsilon = EPSILON_MONOGENIC

    radius, sintheta, costheta, _ = construct_filter_grids(rows, cols)
    riesz = riesz_multiplier(sintheta, costheta)
    lowpass = lowpass_filter(radius)

    spectrum = periodic_fft2(img) if periodic else fft2(img)

    sum_amplitude = np.zeros((rows, cols), dtype=np.float64)
    max_amplitude = np.zeros((rows, cols), dtype=np.float64)
    sum_even = np.zeros((rows, cols), dtype=np.float64)
    sum_h1 = np.zeros((rows, cols), dtype=np.float64)
    sum_h2 = np.zeros((rows, cols), dtype=np.float64)
    tau: float = 0.0

    for s in range(n_scale):
        log_gabor = log_gabor_scale(radius, lowpass, min_wavelength * (mult ** s), sigma_onf)
        band = spectrum * log_gabor

        even = np.real(ifft2(band))
        odd = ifft2(band * riesz)
        h1, h2 = odd.real, odd.imag
        amplitude = np.sqrt(even * even + h1 * h1 + h2 * h2)

        sum_amplitude += amplitude
        sum_even += even
        sum_h1 += h1
        sum_h2 += h2

        if s == 0:
            # sum_amplitude == amplitude here; Kovesi reads the accumulator.
            if abs(noise_method + 1.0) < 1e-9:
                tau = float(np.median(sum_amplitude)) / np.sqrt(np.log(4.0))
            elif abs(noise_method + 2.0) < 1e-9:
                tau = rayleigh_mode(sum_amplitude)
            max_amplitude = amplitude.copy()
        else:
            max_amplitude = np.maximum(max_amplitude, amplitude)

    weight = spread_weight(sum_amplitude, max_amplitude, n_scale, cutoff, g, epsilon)

    if noise_method >= 0:
        threshold = float(noise_method)
    else:
        # Filter bandwidths are scaled inversely, so the noise sums geometrically.
        total_tau = tau * (1.0 - (1.0 / mult) ** n_scale) / (1.0 - 1.0 / mult)
        noise_mean = total_tau * np.sqrt(np.pi / 2.0)
        noise_sigma = total_tau * np.sqrt((4.0 - np.pi) / 2.0)
        # The epsilon floor is phasepack's, not Kovesi's. Inactive unless img is constant.
        threshold = float(max(noise_mean + k * noise_sigma, epsilon))

    energy = np.sqrt(sum_even ** 2 + sum_h1 ** 2 + sum_h2 ** 2)

    ratio = energy / (sum_amplitude + epsilon)
    n_clamped = int(np.count_nonzero((ratio > 1.0) | (ratio < -1.0)))
    phase_deviation = np.maximum(1.0 - deviation_gain * np.arccos(np.clip(ratio, -1.0, 1.0)), 0.0)
    pc = weight * phase_deviation * np.maximum(energy - threshold, 0.0) / (energy + epsilon)

    # Kovesi writes atan(-sumh2/sumh1). arctan2 is equal mod pi and never divides by
    # zero; fold it back into (-pi/2, pi/2] so the [0,1] map is a straight affine one.
    orientation = np.arctan2(-sum_h2, sum_h1)
    orientation = np.where(orientation > np.pi / 2, orientation - np.pi, orientation)
    orientation = np.where(orientation <= -np.pi / 2, orientation + np.pi, orientation)

    feature_type = np.arctan2(sum_even, np.hypot(sum_h1, sum_h2))

    return MonogenicResult(pc, orientation, feature_type, threshold, n_clamped)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/enhance/test_monogenic_kernels.py -q`
Expected: all pass. The golden-fixture assertions should clear `rtol=1e-6` by eight orders (actual `max|Δpc| = 3.5e-14`).

- [ ] **Step 7: Confirm the fixture is load-bearing, not decorative**

`tests/` is not an importable package (there is no `tests/__init__.py`), so this must run
through pytest rather than as a standalone script.

Run: `uv run pytest tests/unit/enhance/test_monogenic_kernels.py -q -k "load_bearing" -v`
Expected: `test_the_fixture_is_load_bearing PASSED` — reintroducing the `fft2` branch shifts
`pc` by `0.67`, far past the `0.1` guard, so a silent regression cannot slip through.

- [ ] **Step 8: Lint, type-check, and re-run the spec's own suite**

Run: `uv run ruff check --fix src/phenotypic/enhance/_monogenic_kernels.py tests/unit/enhance/ && uv run mypy src/phenotypic && uv run python docs/superpowers/specs/2026-07-08-alt-phase-detection/verify_claims.py | tail -1`
Expected: clean, and `21/21 checks passed.`

- [ ] **Step 9: Commit**

```bash
git add src/phenotypic/enhance/_monogenic_kernels.py tests/unit/enhance/ tests/fixtures/phasecongmono_golden.npz docs/superpowers/specs/2026-07-08-alt-phase-detection/
git commit -m "feat(enhance): implement monogenic_phase_congruency, pinned by the golden fixture

Agrees with phasepack 1.5 to max|dpc| = 3.5e-14 across five 64x64 images,
eight orders inside the rtol=1e-6 target. The fixture moves to
tests/fixtures/; verify_claims.py resolves it by walking up from __file__."
```

---

### Task 4: The `FocusEdgeMonogenicPhase` operation

**Files:**
- Modify: `src/phenotypic/sdk_/typing_.py`
- Create: `src/phenotypic/enhance/_focus_edge_monogenic_phase.py`
- Modify: `src/phenotypic/enhance/__init__.py`
- Modify: `tests/unit/abc_/test_enhancer_taxonomy.py`
- Modify: `tests/unit/tune/test_enhance_annotations.py`
- Test: `tests/unit/enhance/test_focus_edge_monogenic_phase.py`

**Interfaces:**
- Consumes: Task 3's `monogenic_phase_congruency`, `MonogenicResult`.
- Produces:
  - `MonogenicOutput = Literal["pc", "orientation", "feature_type"]` in `phenotypic.sdk_.typing_`
  - `FocusEdgeMonogenicPhase` exported from `phenotypic.enhance`, with fields
    `n_scale: int`, `min_wavelength: float`, `mult: float`, `sigma_onf: float`, `k: float`,
    `deviation_gain: float`, `cutoff: float`, `g: float`, `noise_method: float`,
    `output: MonogenicOutput`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/enhance/test_focus_edge_monogenic_phase.py`:

```python
"""Tests for FocusEdgeMonogenicPhase.

Kovesi's phasecongmono, ported. The numerical agreement with the reference is pinned in
``test_monogenic_kernels.py``; this file covers the operation contract and the
behavioural properties that a transcription check cannot see.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from phenotypic import Image, ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import FocusEdgeMonogenicPhase, FocusEdgePhase


class TestParameterValidation:
    def test_constructible_with_no_arguments(self):
        op = FocusEdgeMonogenicPhase()
        assert op.n_scale == 4
        assert op.k == 3.0  # phasecongmono's default, not phasecong3's 2.0
        assert op.deviation_gain == 1.5
        assert op.output == "pc"

    def test_positional_arguments_are_rejected(self):
        with pytest.raises(TypeError):
            FocusEdgeMonogenicPhase(4)

    def test_n_scale_one_is_rejected(self):
        """The spread weight divides by (n_scale - 1)."""
        with pytest.raises(ValidationError):
            FocusEdgeMonogenicPhase(n_scale=1)

    def test_n_scale_two_is_accepted(self):
        assert FocusEdgeMonogenicPhase(n_scale=2).n_scale == 2

    @pytest.mark.parametrize("kwargs", [
        {"min_wavelength": 1.0},
        {"mult": 1.0},
        {"sigma_onf": 0.0},
        {"sigma_onf": 1.5},
        {"k": -1.0},
        {"deviation_gain": 0.0},
        {"cutoff": 0.0},
        {"cutoff": 1.0},
        {"g": 0.0},
    ])
    def test_out_of_bounds_fields_raise(self, kwargs):
        with pytest.raises(ValidationError):
            FocusEdgeMonogenicPhase(**kwargs)

    def test_unknown_output_is_rejected(self):
        with pytest.raises(ValidationError):
            FocusEdgeMonogenicPhase(output="pc_sum")


class TestOperationContract:
    def test_detect_mat_stays_in_unit_range(self):
        image = load_synth_yeast_plate()
        result = FocusEdgeMonogenicPhase().apply(image)
        mat = result.detect_mat[:]
        assert mat.min() >= 0.0
        assert mat.max() <= 1.0

    @pytest.mark.parametrize("output", ["pc", "orientation", "feature_type"])
    def test_every_output_mode_stays_in_unit_range(self, output):
        image = load_synth_yeast_plate()
        result = FocusEdgeMonogenicPhase(output=output).apply(image)
        mat = result.detect_mat[:]
        assert mat.min() >= 0.0
        assert mat.max() <= 1.0

    def test_rgb_and_gray_are_not_mutated(self):
        image = load_synth_yeast_plate()
        rgb_before = image.rgb[:].copy()
        gray_before = image.gray[:].copy()
        FocusEdgeMonogenicPhase().apply(image)
        assert np.array_equal(image.rgb[:], rgb_before)
        assert np.array_equal(image.gray[:], gray_before)

    def test_json_round_trip(self):
        op = FocusEdgeMonogenicPhase(n_scale=5, k=4.0, output="orientation")
        restored = FocusEdgeMonogenicPhase.from_json(op.to_json())
        assert restored.n_scale == 5
        assert restored.k == 4.0
        assert restored.output == "orientation"

    def test_pipeline_round_trip(self):
        pipeline = ImagePipeline(ops=[FocusEdgeMonogenicPhase()])
        restored = ImagePipeline.from_json(pipeline.to_json())
        assert isinstance(restored.ops[0], FocusEdgeMonogenicPhase)

    def test_pc_is_equivariant_under_90_degree_rotation(self):
        """Measured: 6.7e-16. The filter bank is isotropic; nothing prefers an axis."""
        arr = np.zeros((96, 96), dtype=np.float32)
        arr[:, 48:] = 1.0
        straight = FocusEdgeMonogenicPhase().apply(_image_from(arr)).detect_mat[:]
        rotated = FocusEdgeMonogenicPhase().apply(_image_from(np.rot90(arr))).detect_mat[:]
        assert np.abs(np.rot90(straight) - rotated).max() < 1e-6


def _image_from(arr: np.ndarray) -> Image:
    """Wrap a float array in an Image by broadcasting it to RGB."""
    rgb = np.repeat((arr[..., None] * 255).astype(np.uint8), 3, axis=2)
    return Image(rgb)


class TestAxisConvention:
    """Spec test 7. Two axis bugs, and axis-aligned edges catch only one.

    An fx/fy swap rotates every orientation by 90 degrees while leaving pc unchanged to
    1.5e-17, so only an orientation assertion sees it. An ``atan2(+sum_h2, ...)`` sign flip
    mirrors every orientation about the x-axis, and 0 and pi/2 are their own mirror images
    mod pi -- so the axis-aligned pair is blind to *that* one in pc and orientation alike.
    ``starsine``'s rays span every orientation at once and catch both.
    """

    @staticmethod
    def _orientation_at_peak(arr):
        from phenotypic.enhance._monogenic_kernels import monogenic_phase_congruency
        result = monogenic_phase_congruency(arr)
        peak = np.unravel_index(np.argmax(result.pc), result.pc.shape)
        return float(np.degrees(result.orientation[peak]))

    def test_vertical_edge_reads_zero_degrees(self):
        arr = np.zeros((128, 128))
        arr[:, 64:] = 1.0  # intensity varies across COLUMNS
        assert self._orientation_at_peak(arr) == pytest.approx(0.0, abs=0.01)

    def test_horizontal_edge_reads_ninety_degrees(self):
        arr = np.zeros((128, 128))
        arr[64:, :] = 1.0
        assert self._orientation_at_peak(arr) == pytest.approx(90.0, abs=0.01)

    def test_diagonal_edge_reads_forty_five_degrees(self):
        arr = np.fromfunction(lambda i, j: (j - i > 0).astype(float), (128, 128))
        assert self._orientation_at_peak(arr) == pytest.approx(45.18, abs=0.5)

    def test_starsine_orientation_matches_the_generators_own_theta(self):
        """The only test that catches the -sum_h2 sign flip. Measured: 0.98 deg median."""
        from phenotypic.enhance._monogenic_kernels import monogenic_phase_congruency
        from ._kovesi_synthetic import centred_axis, starsine, unit_variance

        n = 256
        img = unit_variance(starsine(n, ncycles=8))
        ax = centred_axis(n)
        X, Y = np.meshgrid(ax, ax, indexing="ij")
        theta = np.arctan2(Y, X)
        radius = np.hypot(X, Y)

        result = monogenic_phase_congruency(img)
        mask = (result.pc > 0.4) & (radius > 30) & (radius < n // 2 - 10)
        assert mask.sum() > 1000

        diff = result.orientation[mask] - theta[mask]
        err = np.abs((diff + np.pi / 2) % np.pi - np.pi / 2)  # undirected, mod pi
        assert np.degrees(np.median(err)) < 2.0
        assert np.degrees(np.quantile(err, 0.9)) < 8.0


class TestBehaviouralControls:
    """Spec tests 8 and 9, on Kovesi's own adversarial images.

    These answer "does it behave like a phase-congruency operator?", which the golden
    fixture cannot. The fixture answers "is it *this* operator?". Both are necessary.
    """

    def test_congruency_survives_the_step_to_line_sweep(self):
        """step2line: pc is constant while feature_type sweeps; |grad| collapses 18x.

        That gap is the operator's entire reason to exist.
        """
        from phenotypic.enhance._monogenic_kernels import monogenic_phase_congruency
        from ._kovesi_synthetic import step2line, unit_variance

        n, col = 256, 170  # the congruency column x = 2*pi
        img = unit_variance(step2line(n))
        result = monogenic_phase_congruency(img)
        grad_y, grad_x = np.gradient(img)
        grad = np.hypot(grad_x, grad_y)

        rows = np.arange(8, n - 8)  # trim the FFT wrap-around
        pc_col = result.pc[rows, col]
        grad_col = grad[rows, col]
        ft_deg = np.degrees(result.feature_type[rows, col])

        # feature type sweeps step (0) -> line (+-90); phasecycles=0.25
        assert 75.0 < (ft_deg[-1] - ft_deg[0]) < 95.0
        # ...while congruency does not
        assert pc_col.max() / pc_col.min() < 1.6
        assert 0.9 < pc_col[-1] / pc_col[0] < 1.15
        # ...and gradient magnitude does
        assert grad_col.max() / grad_col.min() > 8.0
        assert grad_col[-1] / grad_col[0] < 0.15
        assert result.n_clamped == 0

    def test_the_rayleigh_threshold_rejects_one_over_f_noise(self):
        """noiseonf has a pure-noise phase spectrum, so it contains no congruent features.

        Congruency alone does not reject it -- with T off, its 99.9th percentile reaches
        0.72 against 0.95 for a genuinely congruent image. T is what separates them.
        """
        from phenotypic.enhance._monogenic_kernels import monogenic_phase_congruency
        from ._kovesi_synthetic import noiseonf, step2line, unit_variance

        noise = unit_variance(noiseonf(256, 1.5, seed=1))
        with_threshold = monogenic_phase_congruency(noise)
        without = monogenic_phase_congruency(noise, noise_method=0.0)

        assert np.quantile(without.pc, 0.999) > 0.5  # congruency alone cannot reject it
        assert np.quantile(with_threshold.pc, 0.999) < 0.5  # the threshold can

        # ...and the signal is essentially untouched: T adapts to the image's noise floor.
        signal = monogenic_phase_congruency(unit_variance(step2line(256)))
        assert np.quantile(signal.pc, 0.999) > 0.9
        assert with_threshold.threshold / signal.threshold > 3.0


class TestAgreementWithFocusEdgePhase:
    def test_both_localize_a_step_edge_to_the_same_column(self):
        """Spec test 10. Search a window around the true edge: on a bare step both
        operators peak just as hard on the FFT wrap-around edge at column 0, so a
        global argmax is genuinely ambiguous for *both* of them."""
        from phenotypic.enhance._monogenic_kernels import monogenic_phase_congruency

        n, edge = 128, 64
        arr = np.zeros((n, n))
        arr[:, edge:] = 1.0
        mono = monogenic_phase_congruency(arr).pc
        pc3 = FocusEdgePhase()._phasecong3(arr).pc_sum

        lo, hi = edge - 10, edge + 11
        for row in range(10, n - 10):
            mono_col = lo + int(np.argmax(mono[row, lo:hi]))
            pc3_col = lo + int(np.argmax(pc3[row, lo:hi]))
            assert abs(mono_col - pc3_col) <= 1
            assert abs(mono_col - edge) <= 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/enhance/test_focus_edge_monogenic_phase.py -q`
Expected: collection error — `ImportError: cannot import name 'FocusEdgeMonogenicPhase' from 'phenotypic.enhance'`

- [ ] **Step 3: Add the `MonogenicOutput` alias**

In `src/phenotypic/sdk_/typing_.py`, immediately after the `DetectMode` line:

```python
#: Which of ``phasecongmono``'s three maps :class:`FocusEdgeMonogenicPhase` writes to
#: ``detect_mat``. ``orientation`` and ``feature_type`` are angles and are mapped from
#: radians into ``[0, 1]`` by ``(theta + pi/2)/pi`` before being written.
MonogenicOutput = Literal["pc", "orientation", "feature_type"]
```

- [ ] **Step 4: Write the operation**

Create `src/phenotypic/enhance/_focus_edge_monogenic_phase.py`:

```python
"""Monogenic phase congruency: contrast-invariant edges without an orientation sweep.

Implements Peter Kovesi's ``phasecongmono``, cross-checked against his Julia
(``ImagePhaseCongruency.jl``), his MATLAB (``phasecongmono.m``), and the MIT-licensed
``phasepack``.

References:
    Kovesi, P. "Image features from phase congruency." *Videre* 1(3), 1--26 (1999).
    https://github.com/peterkovesi/ImagePhaseCongruency.jl
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import numpy as np
from pydantic import Field

from ._monogenic_kernels import monogenic_phase_congruency
from ..abc_ import FocusEdge
from ..sdk_.typing_ import MonogenicOutput, TuneSpec

if TYPE_CHECKING:
    from phenotypic._core._image import Image


class FocusEdgeMonogenicPhase(FocusEdge):
    """Enhance colony edges in ``detect_mat`` using monogenic phase congruency.

    Detects features where the log-Gabor Fourier components are maximally in phase,
    producing an edge response that depends on phase agreement rather than amplitude.
    The result is invariant to local illumination level and scanner vignetting, so
    faint or translucent colony boundaries stay visible where intensity-gradient
    methods fail.

    Unlike :class:`FocusEdgePhase`, which sweeps a bank of oriented filters, this uses
    the **Riesz transform** to obtain the two odd (quadrature) channels isotropically.
    Orientation falls out of that pair instead of being searched for, so there is no
    ``n_orient`` parameter and the filter bank is ``n_orient`` times smaller.

    Best For:
        - Colony boundaries that vary in opacity or contrast across the plate
        - Filamentous edges where an oriented bank's angular quantization blurs the
          response between two adjacent orientations
        - Plates where you want a cheaper, isotropic alternative to
          :class:`FocusEdgePhase`

    Args:
        n_scale: Number of log-Gabor scales. Must be at least 2 — the frequency-spread
            weight divides by ``n_scale - 1``. More scales widen the frequency coverage
            at linear cost.
        min_wavelength: Wavelength of the finest scale, in pixels. Raise it to ignore
            fine texture such as agar speckle.
        mult: Wavelength multiplier between successive scales. ``2.1`` with
            ``sigma_onf=0.55`` gives roughly two-octave filter bandwidths.
        sigma_onf: Ratio of each filter's Gaussian sigma to its centre frequency.
            Smaller means narrower bandwidth, more scales needed for coverage.
        k: Number of noise standard deviations above the mean at which the noise
            threshold sits. **``phasecongmono``'s default is 3.0**, not
            :class:`FocusEdgePhase`'s 2.0. Raise it on noisy scans.
        deviation_gain: Scales the phase-deviation term, sharpening edge localization.
            Kovesi: "sensible values are from 1 to about 2." Above ~2 the response
            becomes very sparse.
        cutoff: Fractional frequency-spread below which the response is penalized, so
            that a feature excited at a single scale scores lower than a broadband one.
        g: Sharpness of the frequency-spread sigmoid.
        noise_method: ``-1`` estimates the Rayleigh noise parameter from the median of
            the finest scale's amplitude; ``-2`` uses its histogram mode. Any value
            ``>= 0`` is used verbatim as the threshold, so ``0.0`` disables it.
        output: Which map to write to ``detect_mat``. ``"pc"`` is the congruency in
            ``[0, 1]``. ``"orientation"`` and ``"feature_type"`` are angles in
            ``[-pi/2, pi/2]``, mapped to ``[0, 1]`` by ``(theta + pi/2)/pi``, since
            ``detect_mat`` must lie in the unit interval; invert the map to recover
            radians. For ``"orientation"``, ``0.5`` is a vertical edge and ``1.0`` a
            horizontal one. For ``"feature_type"``, ``0.5`` is a step edge, ``1.0`` a
            bright line and ``0.0`` a dark line.

    Examples:
        Enhance colony boundaries on a synthetic yeast plate. Phase congruency responds
        at colony rims regardless of how opaque each colony is:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import FocusEdgeMonogenicPhase
        >>> image = load_synth_yeast_plate()
        >>> enhanced = FocusEdgeMonogenicPhase().apply(image)
        >>> bool(enhanced.detect_mat[:].max() > 0.5)
        True

        Ask instead whether each feature is a step (a colony rim) or a line (a hypha or
        a scratch). ``0.5`` is a step edge:

        >>> feature_type = FocusEdgeMonogenicPhase(output="feature_type")
        >>> classified = feature_type.apply(load_synth_yeast_plate())
        >>> bool(0.0 <= classified.detect_mat[:].min() <= classified.detect_mat[:].max() <= 1.0)
        True

    Note:
        This is a port of Kovesi's ``phasecongmono``. The field notebook attributes
        monogenic phase congruency to Wang Lijuan et al., CCDC 2014; that paper was not
        consulted and this operation does not claim to reproduce its formulation.

    See Also:
        :class:`FocusEdgePhase` for the oriented log-Gabor bank, which additionally
        yields corner strength via the moment tensor.
    """

    n_scale: Annotated[int, TuneSpec(3, 6)] = Field(4, ge=2)
    min_wavelength: Annotated[float, TuneSpec(2.0, 10.0)] = Field(3.0, ge=2.0)
    mult: Annotated[float, TuneSpec(1.5, 3.0)] = Field(2.1, gt=1.0)
    sigma_onf: Annotated[float, TuneSpec(0.1, 1.0)] = Field(0.55, ge=0.1, le=1.0)
    # Lower search bound 0.5 (not 0.0): k=0 disables noise thresholding, a degenerate
    # anchor the optimizer should never spend trials on.
    k: Annotated[float, TuneSpec(0.5, 20.0)] = Field(3.0, ge=0.0)
    deviation_gain: Annotated[float, TuneSpec(1.0, 2.0)] = Field(1.5, gt=0.0)
    cutoff: Annotated[float, TuneSpec(0.3, 0.7)] = Field(0.5, gt=0.0, lt=1.0)
    g: Annotated[float, TuneSpec(2.0, 20.0)] = Field(10.0, gt=0.0)
    noise_method: Annotated[float, TuneSpec(tunable=False)] = -1.0
    output: MonogenicOutput = "pc"

    def _operate(self, image: Image) -> Image:
        """Replace the detection matrix with the selected monogenic map."""
        result = monogenic_phase_congruency(
                image.detect_mat[:],
                n_scale=self.n_scale,
                min_wavelength=self.min_wavelength,
                mult=self.mult,
                sigma_onf=self.sigma_onf,
                k=self.k,
                cutoff=self.cutoff,
                g=self.g,
                deviation_gain=self.deviation_gain,
                noise_method=self.noise_method,
        )

        if self.output == "pc":
            selected = result.pc
        elif self.output == "orientation":
            selected = (result.orientation + np.pi / 2) / np.pi
        else:
            selected = (result.feature_type + np.pi / 2) / np.pi

        # detect_mat enforces float32 on assignment, so no explicit cast is needed.
        image.detect_mat[:] = np.clip(selected, 0.0, 1.0)
        return image
```

- [ ] **Step 5: Export it**

In `src/phenotypic/enhance/__init__.py`, add the import next to the other `FocusEdge*` lines:

```python
from ._focus_edge_monogenic_phase import FocusEdgeMonogenicPhase
```

and add `"FocusEdgeMonogenicPhase",` to `__all__`, immediately after `"FocusEdgePhase",`.

That is all the GUI needs: `gui/_operation_registry.py::discover` scans the `phenotypic.enhance` module, so acceptance criterion 6 (dropdown listing) follows from the export.

- [ ] **Step 6: Register it in the taxonomy and annotation tests**

In `tests/unit/abc_/test_enhancer_taxonomy.py`, add to the `FocusEdge` tuple:

```python
    FocusEdge             : (
        "FocusEdgePhase",
        "FocusEdgeMonogenicPhase",
        "FocusEdgeHessian",
        "FocusEdgeMeijering",
        "FocusEdgeFrangi",
        "FocusEdgeSato",
        "FocusEdgeLaplace",
        "FocusEdgeSobel",
    ),
```

In `tests/unit/tune/test_enhance_annotations.py`, add `FocusEdgeMonogenicPhase` to the `phenotypic.enhance` import block and add these cases to the parametrize list, after the `FocusEdgePhase` ones:

```python
            (FocusEdgeMonogenicPhase(), "n_scale", IntRange, (3, 6)),
            (FocusEdgeMonogenicPhase(), "min_wavelength", FloatRange, (2.0, 10.0, False)),
            (FocusEdgeMonogenicPhase(), "sigma_onf", FloatRange, (0.1, 1.0, False)),
            (FocusEdgeMonogenicPhase(), "k", FloatRange, (0.5, 20.0, False)),
            (FocusEdgeMonogenicPhase(), "deviation_gain", FloatRange, (1.0, 2.0, False)),
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/enhance/test_focus_edge_monogenic_phase.py tests/unit/abc_/test_enhancer_taxonomy.py tests/unit/tune -q`
Expected: all pass. `tests/unit/tune/test_annotation_subset_invariant.py` and `test_annotation_coverage.py` pick the new op up automatically — every `TuneSpec` window above is a strict subset of its `Field` bounds.

- [ ] **Step 8: Verify the GUI dropdown lists it**

```bash
uv run python -c "
from phenotypic.gui import OperationRegistry
reg = OperationRegistry(); reg.discover()
names = [n for n in dir(reg)]
import phenotypic.enhance as e
assert 'FocusEdgeMonogenicPhase' in e.__all__
print('exported:', 'FocusEdgeMonogenicPhase' in e.__all__)
"
```
Expected: `exported: True`

- [ ] **Step 9: Run the doctests**

Run: `uv run pytest --doctest-modules src/phenotypic/enhance/_focus_edge_monogenic_phase.py -q`
Expected: 1 passed (the `Examples:` block).

- [ ] **Step 10: Lint and type-check**

Run: `uv run ruff check --fix src/phenotypic tests/unit/enhance && uv run mypy src/phenotypic`
Expected: clean.

- [ ] **Step 11: Commit**

```bash
git add src/phenotypic/enhance/_focus_edge_monogenic_phase.py src/phenotypic/enhance/__init__.py \
        src/phenotypic/sdk_/typing_.py tests/unit/enhance/test_focus_edge_monogenic_phase.py \
        tests/unit/abc_/test_enhancer_taxonomy.py tests/unit/tune/test_enhance_annotations.py
git commit -m "feat(enhance): add FocusEdgeMonogenicPhase

Kovesi's phasecongmono as a FocusEdge enhancer. Isotropic log-Gabor plus the
Riesz transform, so orientation falls out of the odd channel pair instead of
being swept. Pinned by the golden fixture, plus behavioural controls on
Kovesi's own adversarial test images (step2line, noiseonf, starsine)."
```

---

### Task 5: Documentation and final gates

**Files:**
- Modify: `src/phenotypic/enhance/CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-07-08-alt-phase-detection/README.md`
- Modify: `docs/superpowers/specs/2026-07-08-alt-phase-detection/monogenic-phase-congruency.md`

- [ ] **Step 1: Document the shared kernels module**

Append to `src/phenotypic/enhance/CLAUDE.md`:

```markdown
## `_monogenic_kernels.py`

Pure frequency-domain functions shared by `FocusEdgePhase` (`phasecong3`) and
`FocusEdgeMonogenicPhase` (`phasecongmono`). No `Image` dependency, so they are
unit-testable without fixtures. `FocusEdgeColorPhase` will reuse them wholesale.

Two traps, both covered by tests:

- **`EPSILON_MONOGENIC = 1e-4`** is `phasecongmono`'s. `FocusEdgePhase` passes its own
  `1e-5` (Julia's `phasecong3`) into `spread_weight`. Do not unify them.
- **`riesz_multiplier`'s axes.** Swapping `sintheta`/`costheta` rotates every orientation
  by 90° while leaving `pc` identical to `1.5e-17`. Flipping the sign on `costheta`
  mirrors orientation about the x-axis, and axis-aligned test edges are blind to that in
  `pc` *and* `orientation`. Both are caught only by the `starsine` test.

`periodic_fft2` exists for the golden fixture, which was generated from `phasepack` (a
transcription of Kovesi's MATLAB, which uses `perfft2`). The shipped operation follows
Kovesi's Julia and does **not** use it. See `drift-register.md` M4.
```

- [ ] **Step 2: Mark the spec's status and next step**

In `docs/superpowers/specs/2026-07-08-alt-phase-detection/README.md`, change the `Next steps` list so item 1 reads:

```markdown
1. ~~Plan and implement `FocusEdgeMonogenicPhase`.~~ **Done.** Plan:
   [`plans/2026-07-09-focus-edge-monogenic-phase/plan.md`](../../plans/2026-07-09-focus-edge-monogenic-phase/plan.md).
```

In `monogenic-phase-congruency.md`, change the status line at the top:

```markdown
**Status: IMPLEMENTED.** See [the plan](../../plans/2026-07-09-focus-edge-monogenic-phase/plan.md).
```

- [ ] **Step 3: Run the full gate**

```bash
uv run pytest tests/unit/enhance tests/unit/abc_ tests/unit/tune -q
uv run pytest --doctest-modules src/phenotypic/enhance/_focus_edge_monogenic_phase.py -q
uv run python docs/superpowers/specs/2026-07-08-alt-phase-detection/verify_claims.py | tail -1
uv run mypy src/phenotypic
uv run ruff check
```
Expected: all green; `21/21 checks passed.`; mypy and ruff clean.

- [ ] **Step 4: Confirm every acceptance criterion**

| # | Criterion | Where |
|---|---|---|
| 1 | `test_phase_congruency.py` passes unchanged | Task 2 Step 5, plus the bit-identity check at Step 4 |
| 2 | Golden-fixture agreement at `rtol=1e-6` | `TestGoldenFixture` (Task 3) |
| 3 | The `acos` clamp is provably inert on all three plates | `TestAcosClampIsInert` (Task 3) |
| 4 | The axis-convention test passes | `TestAxisConvention` (Task 4) |
| 5 | `mypy` and `ruff` clean | Step 3 above |
| 6 | The operation appears in the GUI builder dropdown | Task 4 Step 8 (export drives discovery) |
| 7 | Docstring credits Kovesi's `phasecongmono`, not CCDC 2014 | The `Note:` block in Task 4 Step 4 |

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/enhance/CLAUDE.md docs/superpowers/
git commit -m "docs(enhance): document _monogenic_kernels; mark the monogenic spec implemented"
```

---

## Self-Review

**Spec coverage.** Every numbered spec section maps to a task:

| Spec | Task |
|---|---|
| §2.0 `fft2` vs `perfft2` | Task 3 (`periodic` kwarg, default `False`); fixture uses `True` |
| §2.1 congruency formula | Task 3 |
| §2.2 noise threshold + `ε` floor | Task 3, `TestNoiseThreshold` |
| §2.3 `ε = 1e-4`, clamp the `acos` | Task 1 (`EPSILON_MONOGENIC`), Task 3 (`n_clamped`) |
| §3.1 `_monogenic_kernels.py` | Task 1 |
| §3.2 bounded refactor | Task 2 |
| §3.3 latent `n_scale=1` bug | Task 2 Steps 3 and 6 |
| §4 fields | Task 4 |
| §5 error handling | Task 4, `TestParameterValidation` |
| §6 tests 1–12 | 1→T2S5, 2→T3, 3→T1, 4→T3, 5→T3, 6→T4, 7→T4, 8→T4, 9→T4, 10→T4, 11→T4S9, 12→T4 |
| §7 golden fixture | Task 3 |
| §8 acceptance | Task 5 Step 4 |

**Two spec statements this plan deliberately corrects, having measured them:**

1. **§6 test 10** says the two operators must "localize a synthetic step edge to within 1 px of each other." Measured with a *global* argmax, `phasecong3` peaks at column 0, not column 64 — because a bare step edge has a second, equally strong edge at the FFT wrap-around, and `pc_sum` there is `0.3578`, exactly its value at the real edge. The monogenic operator has the identical ambiguity (`0.5890` at both). The test therefore searches a ±10 px window around the true edge, where both land on column 63 with zero disagreement. This is a defect in the test as specified, not in either operator.

2. **§6 test 5** says affine invariance holds "within tolerance" without naming one. Measured: `max|Δpc| = 3.8e-2` on `load_synth_yeast_plate`, `mean = 1.6e-4`, across `(a,b) ∈ {(3,7), (0.5,−0.2), (10,0)}`. The residual is entirely `EPSILON_MONOGENIC`, which is absolute and hence not 1-homogeneous. A *pure* DC offset is removed exactly (`<1e-12`) because the log-Gabor DC bin is zeroed — so the test asserts both, separately.

**Placeholder scan.** No `TBD`, no "add error handling", no "similar to Task N". Every code step carries its code; every command carries its expected output.

**Type consistency.** `monogenic_phase_congruency` returns `MonogenicResult` with `.pc`, `.orientation`, `.feature_type`, `.threshold`, `.n_clamped` — used under exactly those names in Tasks 3, 4 and 5. `spread_weight` takes `epsilon` as its sixth positional parameter in both call sites (`_phasecong3` passes `1e-5`, `monogenic_phase_congruency` passes `EPSILON_MONOGENIC`). `construct_filter_grids` returns a 4-tuple `(radius, sintheta, costheta, freq)` and both callers unpack four. `_kovesi_synthetic.centred_axis` is public (the spec's `verify_claims.py` keeps it private as `_centred_axis`); Task 4's `starsine` test imports the public name.

**One risk the plan cannot retire.** `phasepack` ships no tests of its own, so `TestGoldenFixture` pins *transcription*, not *correctness*. The behavioural controls (`TestBehaviouralControls`, `TestAxisConvention`) are what speak to correctness, and they are blind to the things the fixture catches. Keep both; neither substitutes for the other.

---

# Execution

Derived from the per-task `Files`/`Interfaces` blocks above via the `orchestration-clustering`
procedure. This section is the DAG of record.

## Dependency DAG

```
  S0  fixture move + verify_claims path resolution        [Seam, orchestrator, inline]
      │
      ▼
  A   _monogenic_kernels.py (helpers + algorithm)          [Keystone]
      _kovesi_synthetic.py, test_monogenic_kernels.py
      │
      ├───────────────┬──────────  zero write-overlap → parallel worktrees
      ▼               ▼
  B  FocusEdgePhase  C  FocusEdgeMonogenicPhase           [Seam]  [Keystone + Leaves]
     refactor           operation + exports + registration
      └───────────────┴──────────┐
                                 ▼
                            D  docs                        [Sweep]
                                 ▼
                            E  domain-stripped copy        [Sweep, consistency-critical]
                                 ▼
                            F  scoped Fable review         [Judgment, sandboxed]
```

**Shared-file matrix** (writes only; reads don't constrain ordering):

| Cluster | Writes |
|---|---|
| S0 | `tests/fixtures/phasecongmono_golden.npz`, `verify_claims.py`, spec `README.md` |
| A | `enhance/_monogenic_kernels.py`, `tests/unit/enhance/_kovesi_synthetic.py`, `tests/unit/enhance/test_monogenic_kernels.py` |
| B | `enhance/_focus_edge_phase.py`, `tests/unit/enhance/test_phase_congruency.py` |
| C | `enhance/_focus_edge_monogenic_phase.py`, `enhance/__init__.py`, `sdk_/typing_.py`, `tests/unit/enhance/test_focus_edge_monogenic_phase.py`, `tests/unit/abc_/test_enhancer_taxonomy.py`, `tests/unit/tune/test_enhance_annotations.py` |
| D | `enhance/CLAUDE.md`, spec `README.md` + `monogenic-phase-congruency.md` |

**B and C write disjoint sets** — they are the only parallel-worktree pair. B *reads* `_monogenic_kernels.py`; C *reads* it too. Neither writes it after A. C's `TestAgreementWithFocusEdgePhase` calls `FocusEdgePhase()._phasecong3`, which B leaves bit-identical, so C is correct against either side of B.

## Clusters

### S0 — Seam (orchestrator, inline)

Hoisted out of Task 3 Step 1, because it is a **risky tiny wiring point** that the rest of the work depends on and that fails *silently*.

`verify_claims.py::check_19` returns `Result(..., passed=True, "SKIPPED")` when the fixture is absent. A botched `git mv` therefore leaves the suite reporting **21/21 passed** while the single check that pins the transcription is not running at all. Isolate it, and gate on the check *executing*, not on the suite being green.

- `git mv` the npz to `tests/fixtures/phasecongmono_golden.npz`
- Rewrite `check_19`'s path resolution to walk up from `__file__`
- Fix the spec `README.md` link

**Gate:** the suite prints `21/21`, **and** `check_19`'s detail line contains `max|dpc|`, not `SKIPPED`. Assert both, not just the first.

Under 30 lines across two files — the orchestrator does this directly, no dispatch.

### A — Keystone: the kernels module

Tasks 1 + 3 merged. They write the *same two files*; splitting them means a second agent re-reads and appends to the first agent's module and test file, paying a full context load to produce a half-module in between.

Merged, one agent holds the whole picture: extract the four helpers, add `riesz_multiplier` / `periodic_fft2` / `lowpass_filter` / `log_gabor_scale`, then `monogenic_phase_congruency`. `_kovesi_synthetic.py` is a **Leaf** — folded in, since A's tests are its only consumer.

*Cluster rule check:* shares intent (one module), one reviewable diff (~350 src + ~400 test lines), and **self-verifies in a single pass** — the golden fixture is a hard numeric oracle, so "did I finish correctly" is answerable by the agent itself. Passes.

**Model:** Opus, high effort. This is the mathematical core.
**Gate:** `pytest tests/unit/enhance/test_monogenic_kernels.py` green; the `load_bearing` test green; `verify_claims.py` still 21/21; then a **deep code-review agent** (Opus, high) over A's diff before anything is built on it.

### B — Seam: refactor `FocusEdgePhase`

Small, and the riskiest change in the plan. It touches **shipped, exercised code**, and `detect/_filamentous_fungi_detector.py:424` reaches into `_phasecong3` directly. It also changes a public bound (`n_scale` `ge=1` → `ge=2`) on an operation users already construct.

Isolated for its own gate even though it is ~40 lines. Risk ≠ size.

**Model:** Opus, high effort.
**Gate:** the bit-identity check against `/tmp/phasecong3_baseline.npz` must report **BIT-IDENTICAL across all six outputs** — captured *before* the edit, in the same tree. Then `test_phase_congruency.py`, `test_enhancer_taxonomy.py`, `test_enhance_annotations.py`, and `pytest tests/unit/detect -k filamentous`. If bit-identity fails, fix the refactor; **never relax the check**.

### C — Keystone + folded Leaves: the operation

The operation itself is Keystone. `MonogenicOutput`, the `__init__.py` export, the taxonomy tuple and the two tune-annotation cases are **Leaves** — each is 1–5 lines, none is independently reviewable, and all four exist only to make the operation reachable. Folded in.

The `__init__.py` export is quietly load-bearing: `gui/_operation_registry.py::discover` scans the `phenotypic.enhance` module, so acceptance criterion 6 (GUI dropdown) is satisfied by the export and by nothing else. Gate on it explicitly.

**Model:** Opus, high effort.
**Gate:** `pytest tests/unit/enhance tests/unit/abc_ tests/unit/tune`; doctests; `FocusEdgeMonogenicPhase in phenotypic.enhance.__all__`.

### B ∥ C

Run in **separate worktrees**, merge after both. Zero write overlap (see matrix). After the merge, a single **deep code-review agent** (Opus, high) over the combined B+C diff, then `mypy` + `ruff` + the full affected suite.

### D — Sweep: documentation

`enhance/CLAUDE.md`, the spec's status line and `README.md` next-steps. Mechanical, no judgment, no test surface.

**Model:** Sonnet, medium effort.

### End gate — simplify

One **simplify pass** (Opus) over everything A–D produced: dedupe, reduce, clarify. Quality only, no behaviour change. Apply, then re-run `pytest tests/unit/enhance tests/unit/abc_ tests/unit/tune` plus `verify_claims.py` as the regression check.

---

## E — The domain-stripped copy

**Goal:** a standalone, self-contained corpus a reviewer can judge on the mathematics alone, with **the code logic bit-for-bit intact** and every trace of the application domain removed.

**Location:** the session scratchpad, at `<scratchpad>/math-review/`. Deliberately **not** in the repo:

- it would be a second, drifting copy of shipped code;
- it must never be importable, or someone will `from math_review import ...`;
- the reviewer is told to read nothing outside its own directory, which a repo path makes impossible to enforce.

The reviewer's *report* is what comes back into the repo, under `docs/superpowers/plans/2026-07-09-focus-edge-monogenic-phase/reviews/`.

**Transform rules — what must survive verbatim:**

- Every numeric constant, formula, tolerance, and threshold.
- Every reference citation (Kovesi, Felsberg & Sommer, Fleischmann/Wietzke/Sommer, Shi et al.), every line-number reference into the reference implementations, the `perfft2` fork record, and the `phasepack` odd-grid bug record. **The `refs/` directory is already assembled** at `<scratchpad>/math-review/refs/` — Kovesi's `phasecongruency.jl`, `frequencyfilt.jl`, `syntheticimages.jl`, `phasecongmono.m`, `phasecong3.m`, `perfft2.m`, `filtergrid.m`, `lowpassfilter.m`, and `phasepack`'s three modules. Every claim about "the references" must cite one of these by file and line.
- The drift register rows M1–M5 and the C-series, with their evidence.
- `verify_claims.py` in full — it is already domain-free; confirm, don't rewrite.
- The MIT notice on Kovesi's generators.

**What is removed:**

- All domain nouns and framing. Ban-list, checked mechanically: `colony`, `colonies`, `agar`, `plate`, `yeast`, `fungi`, `fungal`, `hypha`, `hyphal`, `mycel`, `septa`, `microbe`, `microbio`, `phenotyp`, `petri`, `culture`, `biolog`, `organism`.
- `import phenotypic` anywhere. The stripped kernels stand alone on numpy/scipy.
- `load_synth_yeast_plate()` and friends → a deterministic synthetic-array generator, so the doctests and tests still run.
- The spec's `references.md` §1 (provenance) and §3 (what the field notebook gets wrong) — pure domain narrative.

**What is renamed, not deleted:** `FocusEdgeMonogenicPhase` → the operator's mathematical description; "colony boundary" → "step edge"; "hyphal ridge" → "line feature". The *structure* of every docstring stays, so the reviewer sees the same claims.

**Also assembled into the sandbox** (the reviewer cannot fetch these itself, and two of them are gone from disk):

| Artifact | Provenance | Handling |
|---|---|---|
| `refs/phasecongruency.jl` | fetch from `peterkovesi/ImagePhaseCongruency.jl` | MIT, read-only |
| `refs/frequencyfilt.jl`, `refs/syntheticimages.jl` | same | MIT, read-only |
| `refs/phasecongmono.m`, `refs/phasecong3.m` | fetch from `peterkovesi.com/matlabfns` | read-only |
| `refs/phasepack_phasecongmono.py` | fetch from `alimuldal/phasepack` | MIT, read-only. **Do not reinstall the package.** |
| `refs/cmpcm_matlab/` | `Vivianyuwei/…-Conformal-Phase` | **No licence — all rights reserved.** Read-only cross-check. **Never copy its code.** |
| `papers/*.txt` | extracted text of the JMIV / DAGM / CMPCM PDFs | **Copyrighted. Must never be committed.** |

**Model:** Opus, high effort. It is a Sweep, but consistency-critical: a wrong call about what is "biological" versus what is load-bearing mathematics silently corrupts the thing being reviewed.

**Gate (mechanical, then human):**
1. `grep -rniE '<ban-list>' <scratchpad>/math-review/` returns nothing.
2. `grep -rn 'import phenotypic' <scratchpad>/math-review/` returns nothing.
3. The stripped `verify_claims.py` runs standalone and prints `21/21 checks passed.`
4. The stripped kernels reproduce the golden fixture to `rtol=1e-6` — proving the transform preserved the logic and not merely the prose.
5. A diff of the stripped kernels against `src/phenotypic/enhance/_monogenic_kernels.py` shows **changes confined to docstrings, comments and identifiers** — no expression, constant or control-flow edit.

Gate 4 is the one that matters. Without it, "keeping the actual code logic intact" is an assertion; with it, it is a measurement.

## F — Scoped Fable review

A fresh reviewer that has never seen this repository, judging the design on mathematics and faithfulness alone.

**Model:** Fable 5 (`claude-fable-5`), high effort. Frontier tier, so it does not review work produced by a stronger model — the rule holds.

**Context discipline:**
- Working directory is `<scratchpad>/math-review/`. The agent is instructed to read **only** files beneath it. No repo paths appear anywhere in its brief.
- No biological framing in the prompt. The task is described purely as a signal-processing port.
- Tools: read/grep/glob within the sandbox, `WebSearch` and `WebFetch`, and `Bash` restricted to running the stripped tests (with the same memory ceiling that OOM-killed an earlier reviewer: no array over 5e6 elements, no meshgrid over 2000², total live under 500 MB).
- It may read `refs/` and `papers/`, and must treat the unlicensed MATLAB as read-only.

**Charge:** the governing principle, stated to it directly — *faithfulness to validated reference logic beats a convenient shortcut.* Specifically: find shortcuts, unstated assumptions, places where the implementation silently picks a side in a reference disagreement, tolerances that are too loose to catch a plausible bug, and tests that could pass while the code is wrong. It should consult the literature and the reference implementations rather than taking the spec's word for what they say — this spec has already been wrong about that twice.

**It does not fix anything.** Findings come back as a report; the orchestrator triages, surfaces design-level conflicts to the user, and only then applies changes to the *real* spec and code.

**Gate:** every finding is either (a) reproduced against the real code and fixed, (b) reproduced and consciously accepted with a new `drift-register.md` row, or (c) refuted with evidence. No finding is closed by assertion.

---

## Notes on ordering

- **Nothing dispatches until the background plan review is triaged.** It may invalidate A's code before it is written.
- The plan's Task 3 Step 1 (the npz move) executes as **S0**, ahead of Task 1, not inside Task 3. Task 3's remaining steps are absorbed into cluster A.
- The plan's Task 2 and Task 4 map to B and C; Task 5 maps to D.
- Per-cluster gates are light (diff read + tests). The two **deep** review gates are after A, and after the B∥C merge. Any design-level question a review raises goes to the user before the next cluster starts, not after.
