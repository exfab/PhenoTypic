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
- **Invoke the `adding-an-operation` skill before touching any operation parameter.** This is mandatory for clusters **B** and **C** — B changes `FocusEdgePhase.n_scale`'s bound, C declares ten new fields and a closed value set. Cluster A is exempt: `_monogenic_kernels.py` holds pure functions, not operations. Read the skill first, then write. Its two rules that bite here:
  - **Annotation-coverage gate.** Every new numeric (`int`/`float`) field on an `enhance/` operation is pulled into `tests/unit/tune/test_annotation_coverage.py` and must be covered by a `TuneSpec` **or** a `Field` bound, or CI fails. Pick by *intent*, not to silence the gate: a fixed sensible window → `TuneSpec(low, high)`; structural/scene-derived → `TuneSpec(tunable=False)`; tunable but range depends on runtime context → bare `TuneSpec()`. Here: `noise_method` is `TuneSpec(tunable=False)` because it selects an estimator, not a magnitude; every other numeric field has a real window.
  - **Closed value sets.** `MonogenicOutput` is defined **once** as a `TypeAlias` in `sdk_/typing_.py` and reused. Never accept a bare `str`, never derive the `Literal` from a runtime expression. No `Enum` shadow exists, so the enum/literal parity test does not apply.
- **Operations are keyword-only pydantic models.** `FocusEdgeMonogenicPhase(n_scale=4)`, never positional. No hand-written `__init__`. Put normalization and guards in a `field_validator`, never an `__init__`. Bounds go in `Field(...)`, search hints in `TuneSpec(...)`, and every `TuneSpec` window must be a subset of its `Field` bounds (`tests/unit/tune/test_annotation_subset_invariant.py` enforces this automatically).
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
| **Move** `docs/superpowers/specs/2026-07-08-alt-phase-detection/golden_phasecongmono.npz` → `tests/fixtures/phasecongmono_golden.npz` | One canonical copy, shared by `verify_claims.py::check_19` and `test_monogenic_kernels.py`. `check_19` resolves it by walking up from `__file__`, stopping at the `.git` checkout root so a worktree cannot validate against the parent repo's copy. **Done as S0.** |
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
  - `construct_filter_grids(rows: int, cols: int) -> tuple[np.ndarray, ...]` → `(radius, sintheta, costheta, freq, fx, fy)` — **six** values
  - `lowpass_filter(radius: np.ndarray, cutoff: float = 0.45, order: int = 15) -> np.ndarray`
  - `log_gabor_scale(radius: np.ndarray, lowpass: np.ndarray, wavelength: float, sigma_onf: float) -> np.ndarray`
  - `log_gabor_radial(radius: np.ndarray, n_scale: int, min_wavelength: float, mult: float, sigma_onf: float) -> list[np.ndarray]`
  - `riesz_multiplier(fx: np.ndarray, fy: np.ndarray, radius: np.ndarray) -> np.ndarray`
  - `periodic_fft2(img: np.ndarray) -> np.ndarray`
  - `rayleigh_mode(amplitude: np.ndarray, n_bins: int = 50) -> float`
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
        radius, sintheta, costheta, freq, fx, fy = construct_filter_grids(8, 8)
        assert freq[0, 0] == 0.0
        assert radius[0, 0] == 1.0  # so log(radius) never sees zero
        assert sintheta[0, 0] == 0.0
        assert costheta[0, 0] == 0.0

    def test_even_axis_spans_minus_half_to_just_under_half(self):
        freq = construct_filter_grids(1, 8)[3]
        # ifftshift puts DC first; the raw axis was -4/8 .. 3/8
        assert freq.max() == pytest.approx(0.5)

    def test_odd_axis_divides_by_n_not_n_minus_one(self):
        """Both of Kovesi's implementations divide an odd axis by N (frequencyfilt.jl:73,
        filtergrid.m:49). phasepack divides by N-1, in filtergrid AND lowpassfilter -- a
        phasepack bug. k/N is the true DFT bin frequency. All three agree at even sizes."""
        freq = construct_filter_grids(1, 5)[3]
        assert freq.max() == pytest.approx(2.0 / 5.0)  # not 0.5, which is k/(N-1)

    def test_sintheta_and_costheta_are_a_unit_vector_off_dc(self):
        _, sintheta, costheta, _, _, _ = construct_filter_grids(16, 16)
        norm = np.hypot(sintheta, costheta)
        assert np.allclose(norm[1:, 1:], 1.0)


class TestLogGabor:
    def test_dc_is_zeroed_at_every_scale(self):
        radius = construct_filter_grids(32, 32)[0]
        for filt in log_gabor_radial(radius, 4, 3.0, 2.1, 0.55):
            assert filt[0, 0] == 0.0

    def test_returns_one_filter_per_scale(self):
        radius = construct_filter_grids(32, 32)[0]
        assert len(log_gabor_radial(radius, 5, 3.0, 2.1, 0.55)) == 5

    def test_each_scale_peaks_at_its_own_centre_frequency(self):
        radius = construct_filter_grids(256, 256)[0]
        filters = log_gabor_radial(radius, 4, 3.0, 2.1, 0.55)
        for s, filt in enumerate(filters):
            f0 = 1.0 / (3.0 * 2.1**s)
            peak_radius = radius[np.unravel_index(np.argmax(filt), filt.shape)]
            assert peak_radius == pytest.approx(f0, rel=0.05)

    def test_log_gabor_radial_is_log_gabor_scale_per_scale(self):
        radius = construct_filter_grids(32, 32)[0]
        lp = lowpass_filter(radius)
        expected = [log_gabor_scale(radius, lp, 3.0 * 2.1**s, 0.55) for s in range(3)]
        actual = log_gabor_radial(radius, 3, 3.0, 2.1, 0.55)
        for e, a in zip(expected, actual):
            assert np.array_equal(e, a)


class TestRieszMultiplier:
    def test_packs_the_two_odd_channels_into_one_complex_array(self):
        radius, _, _, _, fx, fy = construct_filter_grids(16, 16)
        riesz = riesz_multiplier(fx, fy, radius)
        assert np.array_equal(riesz.real, -fy / radius)
        assert np.array_equal(riesz.imag, fx / radius)

    def test_dc_bin_is_zero_without_being_forced(self):
        """fx[0,0] = fy[0,0] = 0 and radius[0,0] = 1, so the DC bin falls out as 0."""
        radius, _, _, _, fx, fy = construct_filter_grids(16, 16)
        assert riesz_multiplier(fx, fy, radius)[0, 0] == 0

    def test_magnitude_is_one_off_dc(self):
        radius, _, _, _, fx, fy = construct_filter_grids(16, 16)
        riesz = riesz_multiplier(fx, fy, radius)
        assert np.allclose(np.abs(riesz)[1:, 1:], 1.0)

    def test_it_divides_componentwise_as_kovesi_does_not_as_numpy_would(self):
        """`(fx/radius)*1j - (fy/radius)`, NOT numpy's `(1j*fx - fy)/radius`.

        The three references print the same glyphs; the languages differ beneath them.
        MATLAB's `./` and Julia's `/(z::Complex, x::Real)` (base/complex.jl:348 ->
        `Complex(real(z)/x, imag(z)/x)`) divide each component -- a true division.
        numpy promotes the real denominator and runs `nc_quot`, which with a zero
        imaginary part reduces to `scl = 1/r` then a *multiply*. Not the same rounding:
        42.8% of elements differ, by up to 1.41 ulp.

        So the numpy expression is bit-faithful to `phasepack:156` -- an untested
        transcription with a known odd-grid bug -- and not to Kovesi. Settled by running
        frequencyfilt.jl:238 in Julia and diffing IEEE-754 bit patterns. Drift M8.

        Cost: the fixture is phasepack's, so agreement loosens 3.52e-14 -> 5.32e-14,
        still 7.27 orders inside rtol=1e-6.
        """
        radius, sintheta, costheta, _, fx, fy = construct_filter_grids(64, 64)
        shipped = riesz_multiplier(fx, fy, radius)

        assert np.array_equal(shipped, 1j * sintheta - costheta)  # Kovesi: bit-exact
        numpy_form = (1j * fx - fy) / radius
        assert not np.array_equal(shipped, numpy_form)            # phasepack: rejected
        assert np.abs(shipped - numpy_form).max() < 2 * np.spacing(1.0)


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
        """A step edge tiles badly: fft2 sees a second, spurious edge at the wrap.

        Assert the thing perfft2 exists for -- that the PERIODIC component's
        opposite-border jump is far smaller than the input's, while still reconstructing
        the image.

        An earlier version asserted ``|periodic + smooth - img| < 1e-12`` after defining
        ``smooth = img - periodic``, which is ``|img - img|``. It passed for
        ``return np.zeros_like(img)``. The three assertions below were each checked to
        kill a distinct stub: a zeros stub (fails on the mean and the reconstruction),
        plain ``fft2`` (fails on the jump), and ``0.5 * perfft2`` (fails on the
        reconstruction).
        """
        img = np.zeros((32, 32))
        img[:, 16:] = 1.0
        periodic = np.real(ifft2(periodic_fft2(img)))

        input_jump = np.abs(img[:, 0] - img[:, -1]).max()
        periodic_jump = np.abs(periodic[:, 0] - periodic[:, -1]).max()

        assert periodic_jump < 0.25 * input_jump  # measured 0.031x; fft2 gives 1.0x
        assert periodic.mean() == pytest.approx(img.mean())  # S[0,0] = 0 keeps the DC
        assert np.abs(periodic - img).max() < 0.6  # measured 0.4844; it IS still the image


class TestRayleighMode:
    def test_recovers_the_scale_of_a_rayleigh_sample(self):
        """A 50-bin histogram mode is biased low against Rayleigh's long tail."""
        rng = np.random.default_rng(3)
        sigma = 2.0
        sample = sigma * np.sqrt(rng.normal(size=200_000) ** 2 + rng.normal(size=200_000) ** 2)
        assert rayleigh_mode(sample) == pytest.approx(sigma, rel=0.15)

    def test_all_zero_input_returns_zero(self):
        assert rayleigh_mode(np.zeros((4, 4))) == 0.0

    def test_bins_are_anchored_at_zero_and_zeros_are_retained(self):
        """Kovesi: `edges = 0:mx/nbins:mx; n = histc(data, edges)` (phasecongmono.m:465).

        A port that drops zeros and lets numpy place edges at `data.min()` gives a
        different answer. This test pins the VALUE, not merely that it is positive --
        the mutation audit showed a doubled `rayleigh_mode` is otherwise invisible to
        every test in this suite, because the golden fixture is noiseMethod=-1 only.
        """
        # Zeros must count, and the bins must start at 0. mx = 10, n_bins = 10, so the
        # first bin is [0, 1) and holds all 1000 zeros -> centre 0.5.
        data = np.concatenate([np.zeros(1000), np.full(10, 10.0)])
        assert rayleigh_mode(data, n_bins=10) == pytest.approx(0.5)

        # The rejected port -- drop zeros, let numpy place edges at data.min() -- returns
        # 10.05 on the same input. This assertion is what distinguishes the two.
        assert rayleigh_mode(data, n_bins=10) < 1.0

    def test_a_doubled_mode_is_detectable(self):
        """Guard the guard: this suite must be able to see a wrong rayleigh_mode."""
        rng = np.random.default_rng(3)
        sample = 2.0 * np.sqrt(rng.normal(size=50_000) ** 2 + rng.normal(size=50_000) ** 2)
        assert rayleigh_mode(sample) != pytest.approx(2 * rayleigh_mode(sample), rel=0.15)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        Tuple of ``(radius, sintheta, costheta, freq, fx, fy)`` where:

        - ``radius``: radial frequency with ``DC = 1`` so ``log(radius)`` is safe
        - ``sintheta``: ``fx / freq``, the angular filter's sine grid
        - ``costheta``: ``fy / freq``, the angular filter's cosine grid
        - ``freq``: radial frequency with ``DC = 0``
        - ``fx``, ``fy``: the raw signed frequency grids, for :func:`riesz_multiplier`

    Note:
        ``radius`` is bit-equal to the ``freq_safe`` used to build ``sintheta``/``costheta``
        -- both are ``freq`` with ``[0, 0]`` set to 1. So ``sintheta == fx / radius``
        exactly, and :func:`riesz_multiplier` can take ``(fx, fy, radius)`` and divide once.
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

    return radius, sintheta, costheta, freq, fx, fy


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


def riesz_multiplier(fx: np.ndarray, fy: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Kovesi's ``packedmonogenicfilters``: ``H = (i*fx - fy)/radius``.

    Packs both odd (Riesz) channels into one complex array, so a single ``ifft2``
    yields ``h1`` in the real part and ``h2`` in the imaginary part.

    **Divide each component, because that is what the references compute.** The three
    source texts print the same glyphs -- ``phasecongmono.m`` l.183 ``H = (1i*u1 - u2)./radius``,
    ``frequencyfilt.jl`` l.238 ``H = (im.*fx .- fy)./f``, ``phasepack`` l.156
    ``H = (1j*u1 - u2)/radius`` -- but the languages disagree beneath them. MATLAB's ``./``
    and Julia's ``/(z::Complex, x::Real)`` (``base/complex.jl`` l.348,
    ``Complex(real(z)/x, imag(z)/x)``) do a true division per component. numpy promotes the
    real denominator and runs ``nc_quot``, which with a zero imaginary part reduces to
    ``scl = 1/r`` then a **multiply** -- 42.8% of elements differ, by up to 1.41 ulp.

    So ``(1j*fx - fy)/radius`` is bit-faithful to ``phasepack``, not to Kovesi. Settled by
    executing ``frequencyfilt.jl`` l.238 in Julia and diffing IEEE-754 bit patterns.
    Golden agreement loosens ``3.52e-14 -> 5.32e-14``, still 7.27 orders inside
    ``rtol = 1e-6``. Drift ``M8``.

    ``radius`` carries the ``[0, 0] = 1`` fudge, so the DC bin comes out ``0`` on its own.

    **Axis convention.** Swapping ``fx`` and ``fy`` rotates every orientation by 90
    degrees while leaving ``pc`` unchanged to ``1.5e-17``. The sign on ``fy`` encodes a
    y-up convention; flipping it mirrors every orientation about the x-axis, which
    axis-aligned test edges cannot see. Both bugs are caught by ``starsine`` and, since
    the fixture now stores ``orientation``, by the golden fixture.

    Args:
        fx: Signed horizontal frequency grid from :func:`construct_filter_grids`.
        fy: Signed vertical frequency grid from :func:`construct_filter_grids`.
        radius: Radial frequency with ``DC = 1``.

    Returns:
        Complex transfer function with a zero DC bin.
    """
    # Componentwise, NOT `(1j * fx - fy) / radius`. numpy would turn that into a
    # reciprocal-multiply and drift up to 1.41 ulp from Kovesi. See above.
    return (fx / radius) * 1j - (fy / radius)


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


def rayleigh_mode(amplitude: np.ndarray, n_bins: int = 50) -> float:
    """Estimate the Rayleigh distribution parameter from amplitude data.

    For filter responses to Gaussian noise, amplitudes follow a Rayleigh distribution
    whose mode equals sigma.

    **Histogram bins are anchored at zero and zeros are retained**, matching Kovesi::

        % phasecongmono.m:465-471
        mx = max(data(:));
        edges = 0:mx/nbins:mx;
        n = histc(data(:),edges);

    An earlier version of this port dropped zeros and let ``np.histogram`` place its
    edges at ``data.min()``. That is an undeclared deviation from all three references
    (``phasepack``'s ``tools.py`` l.86 does not drop zeros either), and it shifts ``T`` by
    ``0.0130%`` on ``load_synth_yeast_plate``. Neither form is measurably more accurate on
    synthetic Rayleigh samples -- the reason to prefer this one is faithfulness. See
    ``drift-register.md`` M6; this changes shipped ``FocusEdgePhase`` output at
    ``noise_method = -2``.

    Args:
        amplitude: Array of amplitude values.
        n_bins: Number of histogram bins. Kovesi's default is 50.

    Returns:
        Estimated Rayleigh sigma, or ``0.0`` if the maximum is non-positive.
    """
    data = amplitude.flatten()
    maximum = float(data.max()) if data.size else 0.0

    if maximum <= 0.0:
        return 0.0

    edges = np.arange(n_bins + 1) * (maximum / n_bins)
    hist, _ = np.histogram(data, bins=edges)

    mode_idx = int(np.argmax(hist))
    return float((edges[mode_idx] + edges[mode_idx + 1]) / 2)


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
Expected: all pass (24 tests).

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

Also fixes the latent `n_scale=1` bug: `_phasecong3` divides by `n_scale - 1` (line 320) and returns an all-zero `detect_mat` (measured on `load_synth_yeast_plate()`: `max=0` at `n_scale=1` versus `max=0.971004` at `n_scale=4`, with a `RuntimeWarning` and no NaN). Tighten to `ge=2`.

> **This is a deliberate validity narrowing, and one committed fixture depends on the old bound.**
> No *caller* uses `n_scale=1` — `detect/_filamentous_fungi_detector.py:424` passes no `n_scale` and
> takes the default `4`. But `tests/fixtures/tune/back_compat_pipelines/enhance_features_edges.json`
> serializes `"n_scale": 1`, and `test_annotation_back_compat.py::test_legacy_pipeline_json_still_loads`
> deserializes it through `ImagePipeline.from_json`. Applying the `ge=2` edit alone yields
> `1 failed, 14 passed` on that file (verified 2026-07-09).
>
> That corpus is *designed* to fire here: its docstring says each fixture "sits at the tightest
> legal-today edge ... so a too-tight bound that excluded a previously-valid config would trip this
> lock immediately." We are consciously accepting the trip, because the config it locks in is one
> that already produces nothing. The fixture moves to the new tightest legal edge, `n_scale=2`
> (Step 6), and the old bound is pinned shut by `test_n_scale_one_is_rejected` (Step 7) so a future
> revert to `ge=1` cannot silently restore the all-zero path. Recorded by **updating drift `M3`**,
> which already covers this change — do **not** open a new row. (`M8` is the Riesz division.)
>
> There is no `CHANGELOG` in this repository; the breaking change is recorded in
> `drift-register.md` and pinned by the test, not by prose.

**Files:**
- Modify: `src/phenotypic/enhance/_focus_edge_phase.py`
- Modify: `tests/fixtures/tune/back_compat_pipelines/enhance_features_edges.json` (`"n_scale": 1` → `2`)
- Test: `tests/unit/enhance/test_phase_congruency.py` — its existing 25 tests must pass **unchanged**; Step 7 *appends* two.

**Interfaces:**
- Consumes: Task 1's `construct_filter_grids`, `log_gabor_radial`, `rayleigh_mode`, `spread_weight`.
- Produces: nothing new. `FocusEdgePhase._phasecong3` keeps its signature and returns `_PhaseCong3Result`.

- [ ] **Step 1: Capture the pre-refactor output as a bit-identity baseline**

The existing test file asserts properties, not exact arrays. Capture the actual numbers first, or the refactor is unverified.

Capture **both** noise methods. `-1` (median) must stay bit-identical; `-2` is the only caller of `rayleigh_mode` and must change by a bounded amount. Write to the session scratchpad, not `/tmp`.

```bash
uv run python - <<'PY'
import numpy as np
from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import FocusEdgePhase

img = np.asarray(load_synth_yeast_plate().detect_mat[:], dtype=np.float64)
out = {}
for noise_method, tag in ((-1.0, "m1"), (-2.0, "m2")):
    r = FocusEdgePhase(noise_method=noise_method)._phasecong3(img)
    out |= {f"{tag}_M": r.M, f"{tag}_m": r.m, f"{tag}_orientation": r.orientation,
            f"{tag}_feature_type": r.feature_type, f"{tag}_T": np.array(r.T),
            f"{tag}_pc_sum": r.pc_sum}
np.savez(f"{SCRATCH}/phasecong3_baseline.npz", **out)
print("baseline saved; pc_sum.max =", out["m1_pc_sum"].max())
PY
```
Expected: `baseline saved; pc_sum.max = 0.805410725745`. Substitute your session scratchpad path for `{SCRATCH}`. Anything materially different means the tree is not clean.

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

**Keep `List`.** Only `ifftshift` becomes dead. `_compute_angular_spread` survives this refactor and its return annotation at line 473 is `List[np.ndarray]`. `from __future__ import annotations` defers the name, so pytest passes — but Step 7's `mypy` reports `Name "List" is not defined [name-defined]` and `ruff` reports `F821`, which is **not auto-fixable**. Removing `List` makes Step 7 fail.

Tighten the field:

```python
    # ge=2: _phasecong3 divides by (n_scale - 1). At n_scale=1 that is a
    # divide-by-zero which silently yields an all-zero detect_mat.
    n_scale: Annotated[int, TuneSpec(3, 6)] = Field(4, ge=2)
```

**Delete exactly three methods**, at these spans. Do **not** use a "to the end of the class" range — `_compute_angular_spread` sits between them and must survive:

| Method | Lines | Fate |
|---|---|---|
| `_construct_filter_grids` | 379–433 | delete |
| `_construct_log_gabor_filters` | 435–469 | delete |
| **`_compute_angular_spread`** | **471–510** | **KEEP — do not touch** |
| `_rayleigh_mode` | 512–542 | delete |

Deleting `_compute_angular_spread` breaks `_phasecong3:218` and `detect/_filamentous_fungi_detector.py:424`.

Then replace their three call sites inside `_phasecong3`. Note `construct_filter_grids` now returns **six** values — the last two are `fx`/`fy`, which `_phasecong3` does not need:

```python
        radius, sintheta, costheta, freq, _, _ = construct_filter_grids(rows, cols)

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

> **Pre-validated.** Each helper was transcribed from Task 1 and diffed against the method it replaces, before this plan was dispatched:
>
> | Helper | vs | Result |
> |---|---|---|
> | `construct_filter_grids` | `_construct_filter_grids` | **bit-identical** on the first four outputs, at 64², 255², 600×800, 63×64 (it now returns two more, `fx`/`fy`) |
> | `log_gabor_radial` | `_construct_log_gabor_filters` | **bit-identical**, 4 scales @ 600×800; DC zeroed at every scale |
> | `spread_weight(…, 1e-5)` | the inline block at `:320-321` | **bit-identical** |
> | `rayleigh_mode` | `_rayleigh_mode` | **DELIBERATELY DIFFERENT** — drift `M6`. Kovesi anchors bins at 0 and retains zeros; the shipped port does neither |
>
> **The epsilon is the trap.** Passing `1e-4` instead of `1e-5` into `spread_weight` shifts the weight by `max|Δ| = 0.094` — a 9.4% error, silent, and nothing else in `_phasecong3` would notice.
>
> **`rayleigh_mode` changes shipped behaviour, but only on one path.** `_phasecong3` calls it *only* when `noise_method = -2`; at the default `-1` it uses the median. So the bit-identity baseline is untouched. Measured on `load_synth_yeast_plate`:
>
> | `noise_method` | `T` before | `T` after | max abs Δ`pc_sum` | pixels changed |
> |---|---|---|---|---|
> | `-1` (default) | `0.00085782` | `0.00085782` | `0.0` | 0 |
> | `-2` | `0.00392685` | `0.00392681` | `5.578e-06` | 414 182 / 480 000 |
>
> The gate must therefore capture **both** settings: bit-identity at `-1`, and a *bounded, deliberate* change at `-2`. A gate that checks only `-1` would let an accidental `rayleigh_mode` regression through. A gate that demands bit-identity at `-2` would reject the correction we intend.

- [ ] **Step 4: Verify bit-identity against the baseline**

```bash
uv run python - <<'PY'
import numpy as np
from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import FocusEdgePhase

base = np.load(f"{SCRATCH}/phasecong3_baseline.npz")
img = np.asarray(load_synth_yeast_plate().detect_mat[:], dtype=np.float64)
KEYS = ("M", "m", "orientation", "feature_type", "T", "pc_sum")

# noise_method = -1: the median path. rayleigh_mode is never called. MUST be bit-identical.
r1 = FocusEdgePhase(noise_method=-1.0)._phasecong3(img)
now1 = dict(zip(KEYS, (r1.M, r1.m, r1.orientation, r1.feature_type, np.array(r1.T), r1.pc_sum)))
for key in KEYS:
    want = base[f"m1_{key}"]
    assert np.array_equal(want, now1[key]), f"-1 {key} changed: max|d| = {np.abs(want - now1[key]).max():.3e}"
print("noise_method=-1: BIT-IDENTICAL across all six outputs")

# noise_method = -2: the rayleigh_mode path. MUST change, by a bounded amount (drift M6).
r2 = FocusEdgePhase(noise_method=-2.0)._phasecong3(img)
delta_t = abs(float(r2.T) - float(base["m2_T"])) / float(base["m2_T"])
delta_pc = float(np.abs(r2.pc_sum - base["m2_pc_sum"]).max())
assert 0.0 < delta_t < 1e-4, f"-2 T moved {delta_t:.2e}; expected ~9.2e-06 (drift M6)"
assert 0.0 < delta_pc < 1e-4, f"-2 pc_sum moved {delta_pc:.2e}; expected ~5.6e-06"
print(f"noise_method=-2: changed as intended -- dT = {delta_t:.2e}, max|d pc_sum| = {delta_pc:.2e}")
PY
```
Expected:

```
noise_method=-1: BIT-IDENTICAL across all six outputs
noise_method=-2: changed as intended -- dT = 8.64e-06, max|d pc_sum| = 5.58e-06
```

The `-1` assertion is a hard gate: anything else means the refactor changed the algorithm — fix it, do not adjust the check. The `-2` assertion is its mirror image — it *must* change, because drift `M6` corrects `rayleigh_mode` to Kovesi's zero-anchored bins. If `-2` comes out bit-identical, `rayleigh_mode` was not actually corrected.

- [ ] **Step 5: Run the existing test file, the taxonomy test, and the filamentous detector's tests**

Run: `uv run pytest tests/unit/enhance/test_phase_congruency.py tests/unit/abc_/test_enhancer_taxonomy.py tests/unit/tune/test_enhance_annotations.py -q`
Expected: all pass, unchanged. (`test_phase_congruency.py` = **25 passed**; verified under `ge=2` before this plan was written, so the bound change does not disturb it.)

Run: `uv run pytest tests/unit/detect -q -k filamentous`
Expected: all pass (this exercises the `._phasecong3` caller).

- [ ] **Step 6: Watch the back-compat lock fire, then move it to the new edge**

Do **not** skip straight to editing the fixture. Run the lock first and confirm it fails — a guard you never saw fail is a guard you cannot trust.

Run: `uv run pytest tests/unit/tune/test_annotation_back_compat.py -q`
Expected: exactly

```
FAILED tests/unit/tune/test_annotation_back_compat.py::test_legacy_pipeline_json_still_loads[enhance_features_edges]
1 failed, 14 passed
```

If it passes, the `ge=2` edit did not land — go back to Step 3. If a *different* fixture fails, stop: another op's bound was disturbed and this plan does not cover it.

Now move that fixture to the new tightest legal edge. Edit `tests/fixtures/tune/back_compat_pipelines/enhance_features_edges.json`, changing **only** the one key (the file's other values — `n_orient: 1`, `min_wavelength: 2.0`, `k: 0.0`, `sigma_onf: 0.1`, `cutoff: 0.5` — are all still legal and must stay at their edges):

```json
        "n_scale": 2,
```

Re-run: `uv run pytest tests/unit/tune/test_annotation_back_compat.py -q`
Expected: `15 passed`.

- [ ] **Step 7: Pin the old bound shut**

The corpus no longer guards `n_scale=1`, so a plain revert of `ge=2` → `ge=1` would silently restore the all-zero path. Replace that guard with a direct one.

Append to `tests/unit/enhance/test_phase_congruency.py`, inside `TestPhaseCongruencyEnhancerParameterValidation`:

```python
    def test_n_scale_one_is_rejected(self):
        """n_scale=1 divides by (n_scale - 1) and returns an all-zero detect_mat
        (max=0 versus 0.971004 at n_scale=4). Rejected at construction rather
        than producing garbage. See drift-register M3."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(n_scale=1)

    def test_n_scale_two_is_accepted(self):
        assert FocusEdgePhase(n_scale=2).n_scale == 2
```

The existing `test_n_scale_less_than_one_raises_error` (line 24) only asserts `n_scale=0` raises, which held under `ge=1` too. Rename it to `test_n_scale_zero_is_rejected` so the file does not carry a name that implies coverage it never had.

**Prove the new test can fail:** temporarily restore `ge=1`, run `pytest tests/unit/enhance/test_phase_congruency.py -k n_scale_one -q`, confirm it FAILS, then restore `ge=2`. A `pytest.raises` that never saw the non-raising branch is not evidence.

- [ ] **Step 8: Run the tests, lint, type-check**

Run: `uv run pytest tests/unit/enhance/test_phase_congruency.py tests/unit/tune/test_annotation_back_compat.py -q && uv run ruff check --fix src/phenotypic/enhance/_focus_edge_phase.py && uv run mypy src/phenotypic`
Expected: pass, clean.

- [ ] **Step 9: Update drift `M3` (do not open a new row)**

`drift-register.md` **already has** `M3` for `n_scale ≥ 2`. Update it in place with the evidence: deviates from `phasecong3.m`, which also divides by `nscale-1` and is equally undefined at `nscale=1` — we raise where Kovesi would emit a warning and zeros. Evidence: `max=0` vs `max=0.971004`; the back-compat corpus fired (`1 failed, 14 passed`) and its fixture moved from `1` to `2`; `test_n_scale_one_is_rejected` replaces the guardrail. There is no `CHANGELOG` in this repository — the test is the record.

(`M8` is taken: it is the Riesz componentwise-division correction. Numbering a second `M8` would break every `file:line` reference into the register.)

- [ ] **Step 10: Commit**

```bash
git add src/phenotypic/enhance/_focus_edge_phase.py \
        tests/unit/enhance/test_phase_congruency.py \
        tests/fixtures/tune/back_compat_pipelines/enhance_features_edges.json \
        docs/superpowers/specs/2026-07-08-alt-phase-detection/drift-register.md
git commit -m "refactor(enhance)!: move FocusEdgePhase's frequency helpers into _monogenic_kernels

Bit-identical on load_synth_yeast_plate across all six _phasecong3 outputs.

BREAKING: tightens n_scale to ge=2. _phasecong3 divides by (n_scale - 1), so
n_scale=1 returned an all-zero detect_mat (max=0 vs 0.971004 at n_scale=4).
The back-compat fixture enhance_features_edges.json moves to the new tightest
legal edge; test_n_scale_one_is_rejected pins the old bound shut. Drift M3."
```

---

### Task 3: `monogenic_phase_congruency()` and the golden fixture

The algorithm itself, driven by the fixture. `phasepack` is **not** reinstalled — the fixture is the reference now.

**Files:**
- Modify: `src/phenotypic/enhance/_monogenic_kernels.py`
- Create: `tests/unit/enhance/_kovesi_synthetic.py`
- Test: `tests/unit/enhance/test_monogenic_kernels.py` (append)

(The fixture move and `verify_claims.py` rewiring are **no longer part of this task** — see Step 1.)

**Interfaces:**
- Consumes: everything from Task 1.
- Produces:
  - `MonogenicResult` frozen dataclass with fields `pc: np.ndarray`, `orientation: np.ndarray` (radians, `(-pi/2, pi/2]`), `feature_type: np.ndarray` (radians, `[-pi/2, pi/2]`), `threshold: float`, `n_clamped: int`
  - `monogenic_phase_congruency(img, *, n_scale=4, min_wavelength=3.0, mult=2.1, sigma_onf=0.55, k=3.0, cutoff=0.5, g=10.0, deviation_gain=1.5, noise_method=-1.0, periodic=False) -> MonogenicResult`
  - `tests/unit/enhance/_kovesi_synthetic.py` exporting `step2line`, `circsine`, `starsine`, `noiseonf`, `unit_variance`, `centred_axis`

- [x] **Step 1: Move the fixture to its canonical home — ALREADY DONE, as cluster S0**

> **Do not re-run this step.** It was hoisted out of Task 3 and executed inline by the orchestrator
> as **S0** (see `# Execution`), because it fails *silently* and everything downstream depends on it.
> The fixture is already at `tests/fixtures/phasecongmono_golden.npz`; `git mv` will error, and the
> `sed` below will no-op. The content is kept here as the record of what S0 did and why.
>
> **Verified at S0 close:** `21/21 checks passed`, `check_19` reports
> `max|dpc| 3.52e-14 ... max|dorientation| 0 deg (exact)`, exit `0`. With the fixture moved aside:
> `20/21`, `FIXTURE MISSING`, exit `1`. With a decoy `.npz` planted in the *parent* checkout and the
> worktree's copy removed: still `FIXTURE MISSING`, exit `1` — the `.git` bound held.

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
        if (parent / ".git").exists():
            break  # checkout root reached; do not escape into an enclosing repository
    if path is None:
        # FAIL, do not skip. A skip here reports "21/21 passed" while the only check that
        # pins the transcription never runs -- and this file is what caught the perfft2 fork.
        return Result("19 golden fixture agrees with phasepack (rtol=1e-6)", False,
                      "FIXTURE MISSING: tests/fixtures/phasecongmono_golden.npz not found in any "
                      "parent of this file. The check did not run; do not read the suite as green.")
```

Two things here, both load-bearing:

1. `passed=False`, not `True`. Commit `18d856b58` removed a fail-open skip from exactly this check
   after it let the suite print `21/21 checks passed` and exit `0` while `check_19` never ran. Do not
   reintroduce it under any refactor.
2. The ascent **stops at the checkout root** (`.git` — a directory in a clone, a *file* in a worktree,
   so `.exists()` and not `.is_dir()`). This work happens in a git worktree nested under the main
   checkout. Without the bound, a worktree whose own fixture went missing climbs into the parent
   repository and validates against *its* copy — a different tree at a different commit — and
   `check_19` reports PASS for code it never tested. Verified by planting a decoy `.npz` in the
   parent checkout, deleting the worktree's copy, and confirming the suite still reports
   `FIXTURE MISSING` and exits `1`.

Then delete the now-dead `if not path.exists():` block below it, **and** delete the stale docstring
line `"Skips with a PASS if the fixture is absent, so the file stays runnable from a bare checkout
of the spec text alone."` — the code has not done that since `18d856b58`.

Update the README table entry's path:

```bash
sed -i '' 's#\[`golden_phasecongmono.npz`\](./golden_phasecongmono.npz)#[`tests/fixtures/phasecongmono_golden.npz`](../../../../tests/fixtures/phasecongmono_golden.npz)#' docs/superpowers/specs/2026-07-08-alt-phase-detection/README.md
```

Run: `uv run python docs/superpowers/specs/2026-07-08-alt-phase-detection/verify_claims.py`
Expected: the last line is `21/21 checks passed.` **and** `check_19`'s line contains `max|dpc|` — not
`SKIPPED`, not `FIXTURE MISSING`. Assert both. A green suite alone does not prove the check ran.

Then prove the guard is live: `mv` the fixture aside, re-run, confirm the suite exits **non-zero**
and prints `FIXTURE MISSING`, and `mv` it back.

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

    @pytest.mark.parametrize("name", ["step", "step2line", "starsine", "circsine", "noiseonf"])
    def test_orientation_matches_phasepack(self, name):
        """Without this the fixture is blind to BOTH axis bugs.

        `flip_h2_sign` leaves pc, ft and T untouched and passes the vertical/horizontal
        pair (0 and 90 deg are self-mirrored mod pi). Measured with orientation checked:
        flip_h2_sign is off by 5 deg, swap_axes by 90 deg, and both now die here.

        phasepack stores `ori` as np.fix((atan(-h2/h1) % pi)/pi*180) -- integer degrees
        in [0, 180). Ours is already mod pi, so the same quantisation applies. Agreement
        is EXACT: 100% of 4096 px on all five images, max|d| = 0 deg.
        """
        golden = np.load(_FIXTURE, allow_pickle=False)
        img = _golden_cases()[name]
        result = monogenic_phase_congruency(img, periodic=True, **_GOLDEN_PARAMS)

        expected = golden[f"{name}__or"]
        finite = np.isfinite(expected)  # phasepack yields NaN where sumh1 == 0

        # `% np.pi` FIRST. Our orientation is folded to (-pi/2, pi/2], so it can be
        # negative, and np.fix truncates toward zero: -0.5 deg would become -0, not 179.
        # Omitting the modulo yields a spurious 1-degree disagreement everywhere.
        ours = np.fix((result.orientation % np.pi) / np.pi * 180.0)
        delta = np.abs(ours[finite] - expected[finite])
        delta = np.minimum(delta, 180.0 - delta)  # circular, mod 180
        assert delta.max() == 0.0

    def test_the_fixture_is_load_bearing(self):
        """A fixture that cannot fail proves nothing.

        Regressing to the ``fft2`` branch — which no behavioural test in this repo
        detects — must break this fixture. Measured drift at n=64: 0.5342 absolute.
        (check_18 reports 0.67; that is measured at n=256.)
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

    def test_rayleigh_mode_path_produces_the_rayleigh_mode_threshold(self):
        """`threshold > 0` would pass for any transcription error. Pin the value.

        The -2 branch is NOT covered by the golden fixture (its `_params` record
        noiseMethod = -1), and the mutation audit showed a doubled `rayleigh_mode` is
        otherwise invisible to every test here.
        """
        img = unit_variance(noiseonf(64, 1.5, seed=1))
        result = monogenic_phase_congruency(img, noise_method=-2.0)

        # Recompute T from rayleigh_mode by hand, exactly as the kernel must.
        radius, _, _, _, fx, fy = construct_filter_grids(64, 64)
        riesz = riesz_multiplier(fx, fy, radius)
        band = np.fft.fft2(img) * log_gabor_scale(radius, lowpass_filter(radius), 3.0, 0.55)
        even = np.real(np.fft.ifft2(band))
        odd = np.fft.ifft2(band * riesz)
        amplitude = np.sqrt(even**2 + odd.real**2 + odd.imag**2)

        tau = rayleigh_mode(amplitude)
        total_tau = tau * (1 - (1 / 2.1) ** 4) / (1 - 1 / 2.1)
        expected = max(total_tau * np.sqrt(np.pi / 2) + 3.0 * total_tau * np.sqrt((4 - np.pi) / 2),
                       EPSILON_MONOGENIC)
        assert result.threshold == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("bad", [-1.5, -3.0, -0.5, -2.5])
    def test_out_of_range_noise_method_raises(self, bad):
        """-1.5 matches neither branch, leaves tau = 0, and silently reduces T to epsilon.

        Kovesi's MATLAB errors on the undefined tau. Fail loudly rather than degrade.
        """
        with pytest.raises(ValueError, match="noise_method"):
            monogenic_phase_congruency(np.zeros((32, 32)), noise_method=bad)


class TestThePeriodicDefaultIsPinned:
    """Nothing else pins it, and flipping it changes what the operation computes.

    TestGoldenFixture passes `periodic=True` explicitly; `test_the_fixture_is_load_bearing`
    passes `False` explicitly; every behavioural control uses the default. Measured: flip
    the kernel default to True and step2line, noiseonf, axis_pair, starsine and affine ALL
    still pass. Without this test, a one-character change to the signature silently moves
    the shipped operation onto the MATLAB branch -- 0.67 absolute in pc -- with a green
    suite. That is drift-register S6, in the feature that taught us S6.
    """

    def test_the_kernel_defaults_to_fft2_not_perfft2(self):
        img = unit_variance(step2line(64))
        default = monogenic_phase_congruency(img)
        explicit_fft2 = monogenic_phase_congruency(img, periodic=False)
        explicit_perfft2 = monogenic_phase_congruency(img, periodic=True)

        assert np.array_equal(default.pc, explicit_fft2.pc)
        assert np.abs(default.pc - explicit_perfft2.pc).max() > 0.1

    # The operation-level half of this test does NOT belong here. This file is Task A's
    # and must stay green on its own; FocusEdgeMonogenicPhase does not exist until Task C.
    # See test_focus_edge_monogenic_phase.py::test_the_operation_uses_the_fft2_branch.


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

    if noise_method < 0 and not (
            abs(noise_method + 1.0) < epsilon or abs(noise_method + 2.0) < epsilon
    ):
        raise ValueError(
                f"noise_method must be -1 (median), -2 (Rayleigh mode), or >= 0 "
                f"(a literal threshold); got {noise_method!r}. A value like -1.5 would "
                f"silently leave tau = 0 and reduce T to epsilon."
        )

    radius, _, _, _, fx, fy = construct_filter_grids(rows, cols)
    riesz = riesz_multiplier(fx, fy, radius)
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
            # The dispatch compares against `epsilon`, as phasecongruency.jl:512 and
            # phasecongmono.m:224 do. Out-of-range values already raised above.
            if abs(noise_method + 1.0) < epsilon:
                tau = float(np.median(sum_amplitude)) / np.sqrt(np.log(4.0))
            elif abs(noise_method + 2.0) < epsilon:
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
Expected: all pass.

**This code was executed against the fixture before the plan was dispatched.** Transcribed verbatim from the steps above and run standalone, it gives `max|Δpc|` of `1.8e-15` (step), `5.4e-15` (step2line), **`3.5e-14`** (starsine), `2.1e-15` (circsine), `2.0e-15` (noiseonf), and `max|Δorientation| = 0` degrees on all five. Worst case `3.525e-14`, which is `log10(1e-6 / 3.525e-14) = 7.45` orders inside `rtol=1e-6`. If your run differs materially from those numbers, you have transcribed something wrong; do not adjust the tolerance.

This matches `verify_claims.py::check_19` exactly, because `riesz_multiplier` now divides once, as the references do. An earlier revision of this plan specified `1j*sintheta - costheta`, which reassociates into two divisions and gave `5.324e-14` — harmless, but a shortcut. See `drift-register.md` and `TestRieszMultiplier::test_it_divides_once_as_the_references_do`.

- [ ] **Step 7: Confirm the fixture is load-bearing, not decorative**

`tests/` is not an importable package (there is no `tests/__init__.py`), so this must run
through pytest rather than as a standalone script.

Run: `uv run pytest tests/unit/enhance/test_monogenic_kernels.py -q -k "load_bearing" -v`
Expected: `test_the_fixture_is_load_bearing PASSED` — reintroducing the `fft2` branch shifts
`pc` by `0.5342` at `n=64`, far past the `0.1` guard, so a silent regression cannot slip through.

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

    def test_the_operation_uses_the_fft2_branch(self):
        """The operation must not quietly pass periodic=True.

        `_operate` never passes `periodic`, so the shipped branch is enforced solely by
        the kernel's signature default. Flipping that default changes what this operation
        computes by up to 0.67 absolute, and every behavioural test in this suite is blind
        to it. This is the only assertion standing between us and that regression.
        """
        from phenotypic.enhance._monogenic_kernels import monogenic_phase_congruency

        arr = np.zeros((64, 64), dtype=np.float32)
        arr[:, 32:] = 1.0
        image = _image_from(arr)

        produced = np.asarray(FocusEdgeMonogenicPhase().apply(image).detect_mat[:], dtype=np.float64)
        mat = np.asarray(_image_from(arr).detect_mat[:], dtype=np.float64)
        expected = np.clip(monogenic_phase_congruency(mat, periodic=False).pc, 0.0, 1.0)
        wrong = np.clip(monogenic_phase_congruency(mat, periodic=True).pc, 0.0, 1.0)

        np.testing.assert_allclose(produced, expected, rtol=1e-5, atol=1e-6)
        assert np.abs(produced - wrong).max() > 0.05  # and the two branches really differ

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
        """Measured 45.0075 deg. An earlier revision anchored on 45.18 with abs=0.5,
        so the tolerance was carrying the assertion rather than the anchor."""
        arr = np.fromfunction(lambda i, j: (j - i > 0).astype(float), (128, 128))
        assert self._orientation_at_peak(arr) == pytest.approx(45.01, abs=0.1)

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

Query the registry. Asserting only that the name is in `enhance.__all__` tests the export, not the discovery — acceptance criterion 6 is about the dropdown.

`get_by_category` returns `OperationInfo` objects, not strings, so extract `.name`. There are **29** enhancers before this task, across every enhancer family (not just `FocusEdge*`).

```bash
uv run python -c "
from phenotypic.gui import OperationRegistry
reg = OperationRegistry(); reg.discover()
names = [op.name for op in reg.get_by_category('Enhancer')]
assert 'FocusEdgeMonogenicPhase' in names, f'not in the dropdown: {sorted(names)}'
print('dropdown lists it; Enhancer count =', len(names))
"
```
Expected: `dropdown lists it; Enhancer count = 30`

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
                            simplify pass                  [quality only, no behaviour change]
                                 │
                    ═════════════╪═════════════  THE FINAL GATE  ═════════════
                                 ▼
                            E  domain-stripped copy        [Sweep, consistency-critical]
                               code + tests + fixture +
                               spec + plan + references
                                 ▼
                            F  scoped Fable review         [Judgment, sandboxed]
                                 ▼
                            triage: fix / accept+drift-row / refute
                                 ▼
                              merge
```

**The final gate is E → F.** A–D and the simplify pass only produce something worth reviewing; they do not close the work. Nothing merges until F's findings are each fixed, consciously accepted with a `drift-register.md` row, or refuted with evidence.

**Shared-file matrix** (writes only; reads don't constrain ordering). Re-derived from each task's
`Files` block, not copied forward:

| Cluster | Writes |
|---|---|
| S0 | `tests/fixtures/phasecongmono_golden.npz`, `verify_claims.py`, spec `README.md`, this plan's Task 3 Step 1 |
| A | `enhance/_monogenic_kernels.py`, `tests/unit/enhance/_kovesi_synthetic.py`, `tests/unit/enhance/test_monogenic_kernels.py` |
| B | `enhance/_focus_edge_phase.py`, `tests/unit/enhance/test_phase_congruency.py`, `tests/fixtures/tune/back_compat_pipelines/enhance_features_edges.json`, spec `drift-register.md` |
| C | `enhance/_focus_edge_monogenic_phase.py`, `enhance/__init__.py`, `sdk_/typing_.py`, `tests/unit/enhance/test_focus_edge_monogenic_phase.py`, `tests/unit/abc_/test_enhancer_taxonomy.py`, `tests/unit/tune/test_enhance_annotations.py` |
| D | `enhance/CLAUDE.md`, spec `README.md` + `monogenic-phase-congruency.md` |

**B ∩ C = ∅** — they are the only parallel-worktree pair. Verified, not assumed:

- B's only source write is `_focus_edge_phase.py`. Its three other writes are a test file, a JSON
  fixture, and a spec doc, none of which C touches.
- B *reads* `_monogenic_kernels.py`; C *reads* it too. Neither writes it after A.
- C's `TestAgreementWithFocusEdgePhase` calls `FocusEdgePhase()._phasecong3`, which B leaves
  bit-identical (that is B's hard gate), so C is correct against either side of B.

**S0 ∩ D = {spec `README.md`}** — sequential and far apart, so harmless; recorded so a future
parallelisation attempt does not miss it.

**Task 2 (B) consumes only Task 1's helpers, not Task 3's.** In principle B could start as soon as
Task 1 lands, i.e. against A's first half. We deliberately do **not** do that: Tasks 1 and 3 write the
*same two files*, so splitting A to unblock B early would force a second agent to re-read and append
to a half-finished module — paying a full context load to produce an unreviewable intermediate. B
waits. The cost is bounded because B and C then run concurrently anyway.

## Clusters

### S0 — Seam (orchestrator, inline)

Hoisted out of Task 3 Step 1, because it is a **risky tiny wiring point** that the rest of the work depends on and that fails *silently*.

`verify_claims.py::check_19` returns `Result(..., passed=True, "SKIPPED")` when the fixture is absent. A botched `git mv` therefore leaves the suite reporting **21/21 passed** while the single check that pins the transcription is not running at all. Isolate it, and gate on the check *executing*, not on the suite being green.

- `git mv` the npz to `tests/fixtures/phasecongmono_golden.npz`
- Rewrite `check_19`'s path resolution to walk up from `__file__`, keeping the **fail-loud** `passed=False` branch
- Delete `check_19`'s stale docstring line claiming it "Skips with a PASS if the fixture is absent" — untrue since `18d856b58`
- Fix the spec `README.md` link
- **Fix this plan's own Task 3 Step 1**, whose code block reintroduced the `passed=True, "SKIPPED"` branch that `18d856b58` removed. Left alone, a future executor of Task 3 would silently undo S0.

**Gate:** the suite prints `21/21`, **and** `check_19`'s detail line contains `max|dpc|`, not `SKIPPED`. Assert both, not just the first. Then move the fixture aside and confirm the suite exits **non-zero** — a fail-loud branch nobody watched fail is not fail-loud.

Under 30 lines across four files — the orchestrator does this directly, no dispatch.

### A — Keystone: the kernels module

Tasks 1 + 3 merged. They write the *same two files*; splitting them means a second agent re-reads and appends to the first agent's module and test file, paying a full context load to produce a half-module in between.

Merged, one agent holds the whole picture: extract the four helpers, add `riesz_multiplier` / `periodic_fft2` / `lowpass_filter` / `log_gabor_scale`, then `monogenic_phase_congruency`. `_kovesi_synthetic.py` is a **Leaf** — folded in, since A's tests are its only consumer.

*Cluster rule check:* shares intent (one module), one reviewable diff (~350 src + ~400 test lines), and **self-verifies in a single pass** — the golden fixture is a hard numeric oracle, so "did I finish correctly" is answerable by the agent itself. Passes.

**Model:** Opus, high effort. This is the mathematical core.
**Gate:** `pytest tests/unit/enhance/test_monogenic_kernels.py` green; the `load_bearing` test green; `verify_claims.py` still 21/21; then a **deep code-review agent** (Opus, high) over A's diff before anything is built on it.

### B — Seam: refactor `FocusEdgePhase`

Small, and the riskiest change in the plan. It touches **shipped, exercised code**, and `detect/_filamentous_fungi_detector.py:424` reaches into `_phasecong3` directly. It also **breaks a public bound** (`n_scale` `ge=1` → `ge=2`) on an operation users already construct and have serialized.

Isolated for its own gate even though it is ~40 lines. Risk ≠ size.

**Required skill:** `adding-an-operation`, invoked *before* editing. The `n_scale` bound change is an operation-parameter edit: `Field(4, ge=1)` → `Field(4, ge=2)`. Confirm `TuneSpec(3, 6)` stays a subset of the new bound (it does), and that no other field's coverage changes.

**The back-compat lock will fire, and that is expected.** `tests/fixtures/tune/back_compat_pipelines/enhance_features_edges.json` serializes `"n_scale": 1`; `test_annotation_back_compat.py` loads it via `ImagePipeline.from_json`. Measured: the bound edit alone gives `1 failed, 14 passed`. Task 2 Step 6 requires the agent to *observe the failure first*, then move the fixture to the new tightest legal edge (`2`); Step 7 pins the old bound shut with `test_n_scale_one_is_rejected` and proves that test can fail. Do not let an agent "fix" the lock by editing the fixture before it has seen it fail — a guardrail nobody watched fire is not a guardrail.

**Model:** Opus, high effort.
**Gate:** the bit-identity check against the scratchpad `phasecong3_baseline.npz` must report **BIT-IDENTICAL across all six outputs** at `noise_method=-1`, and a *bounded, non-zero* change at `-2` (`0 < dT < 1e-4`, `0 < dpc < 1e-4`; expect `dT = 8.64e-06`, `dpc = 5.58e-06`) — baseline captured *before* the edit, in the same tree. Then `test_phase_congruency.py`, **`test_annotation_back_compat.py` (15 passed)**, `test_enhancer_taxonomy.py`, `test_enhance_annotations.py`, `tests/unit/tune/test_annotation_coverage.py`, `test_annotation_subset_invariant.py`, and `pytest tests/unit/detect -k filamentous`. If bit-identity fails, fix the refactor; **never relax the check**.

### C — Keystone + folded Leaves: the operation

The operation itself is Keystone. `MonogenicOutput`, the `__init__.py` export, the taxonomy tuple and the two tune-annotation cases are **Leaves** — each is 1–5 lines, none is independently reviewable, and all four exist only to make the operation reachable. Folded in.

The `__init__.py` export is quietly load-bearing: `gui/_operation_registry.py::discover` scans the `phenotypic.enhance` module, so acceptance criterion 6 (GUI dropdown) is satisfied by the export and by nothing else. Gate on it explicitly.

**Required skill:** `adding-an-operation`, invoked *before* writing the operation. This cluster is the skill's central case — ten new pydantic fields, one closed value set, and a new `enhance/` operation that the annotation-coverage gate will discover automatically. In particular:
- `deviation_gain` is a **new** numeric field with no counterpart on `FocusEdgePhase`. It needs a `TuneSpec` chosen by intent (`TuneSpec(1.0, 2.0)` — Kovesi: "sensible values are from 1 to about 2"), not a `tunable=False` to quiet the gate.
- `noise_method` selects an estimator (`-1` median, `-2` Rayleigh mode, `>= 0` literal threshold), not a magnitude. It is `TuneSpec(tunable=False)` for that reason, matching `FocusEdgePhase`.
- `MonogenicOutput` is a `TypeAlias` declared once in `sdk_/typing_.py` beside `FootprintShape` and `DetectMode`, and reused. No `Enum` shadow, so no parity test.

**Model:** Opus, high effort.
**Gate:** `pytest tests/unit/enhance tests/unit/abc_ tests/unit/tune` (which includes `test_annotation_coverage.py` and `test_annotation_subset_invariant.py`, both of which discover the new operation on their own); doctests; `FocusEdgeMonogenicPhase in phenotypic.enhance.__all__`.

### B ∥ C

Run in **separate worktrees**, merge after both. Zero write overlap (see matrix). After the merge, a single **deep code-review agent** (Opus, high) over the combined B+C diff, then `mypy` + `ruff` + the full affected suite.

### D — Sweep: documentation

`enhance/CLAUDE.md`, the spec's status line and `README.md` next-steps. Mechanical, no judgment, no test surface.

**Model:** Sonnet, medium effort.

### Simplify pass (precedes the final gate)

One **simplify pass** (Opus) over everything A–D produced: dedupe, reduce, clarify. Quality only, no behaviour change. If it touches any operation field it must invoke `adding-an-operation` first — "simplifying" a `TuneSpec` away is exactly the failure the annotation-coverage gate exists to catch. Apply, then re-run `pytest tests/unit/enhance tests/unit/abc_ tests/unit/tune` plus `verify_claims.py` as the regression check.

This is **not** the last step. The work is not done until **E** and **F** below have run and their findings are triaged — the simplify pass only makes the corpus worth reviewing.

---

# The final gate: E → F

**The final gate is the domain-stripped corpus (E) handed to a scoped Fable reviewer (F).** Nothing merges until F's findings are triaged. A–D produce code that passes *our* tests; E and F are what test whether the mathematics is right and the port is faithful, judged by a reader with no stake in our framing and no way to be primed by it.

## E — The domain-stripped copy

**Goal:** a standalone, self-contained corpus a reviewer can judge on the mathematics alone, with **the code logic bit-for-bit intact** and every trace of the application domain removed.

**Scope — everything F needs, and nothing that leaks the domain.** This is the full manifest; ban-list hit counts measured on the pre-strip files, 2026-07-09:

| Sandbox path | Source | Ban-list hits to fix |
|---|---|---|
| `kernels/_monogenic_kernels.py` | `src/phenotypic/enhance/_monogenic_kernels.py` (from A) | — (new file; keep it clean at birth) |
| `kernels/_focus_edge_phase.py` | `src/phenotypic/enhance/_focus_edge_phase.py` (from B) | **18** |
| `kernels/_focus_edge_monogenic_phase.py` | `src/phenotypic/enhance/_focus_edge_monogenic_phase.py` (from C) | — (new file) |
| `kernels/typing_.py` | the `MonogenicOutput` alias from `sdk_/typing_.py` (excerpt, not the module) | — |
| `tests/test_monogenic_kernels.py`, `tests/_kovesi_synthetic.py`, `tests/test_focus_edge_monogenic_phase.py`, `tests/test_phase_congruency.py` | from A, B, C | check each |
| `tests/fixtures/phasecongmono_golden.npz` | `tests/fixtures/phasecongmono_golden.npz` | binary; **required by gate 4** |
| `verify_claims.py` | spec's copy | **1** (see below) |
| `spec/monogenic-phase-congruency.md` | spec | **6** |
| `spec/color-phase-congruency.md` | spec | **14** |
| `spec/conformal-lift.md` | spec | **6** |
| `spec/references.md` | spec | **21** |
| `spec/drift-register.md` | spec | **5** |
| `spec/README.md` | spec | **0** |
| `plan/plan.md` | this plan | **110** — the largest strip job; do not skip it |
| `plan/reviews/2026-07-09-plan-review.md` | this plan's review | check |
| `refs/`, `papers/` | already assembled (see table below) | verbatim, read-only |

The plan and the spec go in because F's charge is to find *shortcuts and unstated assumptions*, and both live in the prose, not only the code. Handing over the kernels alone would ask F to re-derive the reasoning it is meant to audit.

> **Correction to a claim this plan used to make.** `verify_claims.py` is described below as "already
> domain-free; confirm, don't rewrite." It is not, quite: line 8 reads *"…**not** import
> ``phenotypic``…"*, and `phenotyp` is on the ban-list, so gate 1 fires on the one file the rule says
> to copy verbatim. Rewrite that single phrase to "does not import the host package" and leave every
> other byte alone. Measured: exactly **1** hit, in prose, in a docstring. No code changes.

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

**Also assembled into the sandbox** (the reviewer cannot fetch these itself).

> **Durable master:** `~/.claude/refs/phenotypic-alt-phase-detection/{refs,papers,refimpl}/`.
> Verified 2026-07-09 to lie outside every git work tree, so the copyrighted PDFs **cannot** be
> committed even by accident. `~/.claude` is not a repository; the repo's own `.claude/` is only
> `.gitignore`d, which is one `git add -f` away from a licence problem. E **copies** from the master
> into the sandbox; it does not symlink, so the sandbox stays self-contained for a reviewer told to
> read nothing above its own root. The previous home was the session scratchpad under `/private/tmp`,
> which is reaped — that is why two paper texts went missing once already.

Status audited 2026-07-09. **Present** means verified on disk by extracting text and matching the
title/authors/pages against `references.md` — not by trusting the filename.

| Artifact | Provenance | Status | Handling |
|---|---|---|---|
| `refs/phasecongruency.jl`, `frequencyfilt.jl`, `syntheticimages.jl` | `peterkovesi/ImagePhaseCongruency.jl` | **present** | MIT, read-only |
| `refs/phasecongmono.m`, `phasecong3.m`, `perfft2.m`, `filtergrid.m`, `lowpassfilter.m` | `peterkovesi.com/matlabfns` | **present** | read-only |
| `refs/phasepack_phasecongmono.py`, `phasepack_tools.py`, `phasepack_filtergrid.py` | `alimuldal/phasepack` | **present** | MIT, read-only. **Do not reinstall the package.** |
| `refimpl/test_phasecongruency.jl`, `test_syntheticimages.jl`, `runtests.jl` | Kovesi's own Julia test suite | **present** | MIT, read-only. Note `references.md:775`: **0 assertions** — it only checks the functions run. |
| `papers/felsberg2001_monogenic_signal.pdf` + `.txt` | IEEE TSP 49(12) 3136–3144 (2001), author-hosted copy | **present** | **Copyrighted. Never commit.** |
| `papers/felsberg2004_monogenic_scale_space.pdf` + `.txt` | JMIV 21(1–2) 5–26 (2004) | **present** | **Copyrighted. Never commit.** |
| `papers/wietzke2008_conformal_monogenic_signal_DAGM.pdf` + `.txt` | DAGM 2008, LNCS 5096, 527–536 | **present** | **Copyrighted. Never commit.** |
| `papers/fleischmann2011_conformal_embedding.pdf` + `.txt` | JMIV, "Image Analysis by Conformal Embedding" | **present** | **Copyrighted. Never commit.** |
| `papers/shi2019_cmpcm_color_edge_detection.pdf` + `.txt` | MTAP 78, 10701–10716 (2019) — **the CMPCM paper** | **present** | **Copyrighted. Never commit.** |
| `papers/wang2014_monogenic_phase_congruency_CCDC.pdf` | 26th CCDC (2014) 2033–2038, IEEE | **ABSENT** — IEEE returns `502` to every non-browser client | See below. |
| `refs/cmpcm_matlab/` | `Vivianyuwei/…-Conformal-Phase` | **ABSENT** — never fetched | **No licence — all rights reserved.** Read-only cross-check if obtained. **Never copy its code.** |

**Two artifacts are absent, and F must be told so explicitly rather than left to infer it:**

1. **Wang Lijuan et al. (2014), the monogenic-PC paper.** `references.md:46` already flags it *"Not read directly"* — the author names were reconstructed from CMPCM's reference [13]. Those names are now **confirmed correct** against IEEE Xplore's own author list (Wang Lijuan; Zhang Changsheng; Liu Ziyu; Sun Bin; Tian Haiyong), but the text has still never been read. Nothing in `_monogenic_kernels.py` depends on it — the port's oracle is Kovesi's code plus the golden fixture — so this does **not** block A/B/C. It does mean no one has checked our monogenic-PC prose against its nominal source.
2. **The unlicensed CMPCM MATLAB.** `references.md:709` records that it implements a *precursor grayscale paper*, not CMPCM, and contradicts our reading on the one structural question that matters. Its absence costs a cross-check, not a reference.

If either is later obtained, add it to the master and re-run E. Do **not** let E's gate 4 (manifest completeness) pass by deleting these rows — mark them absent and carry the mark into F's brief.

**What the CMPCM PDF unblocked.** `color-phase-congruency.md` and `references.md` both record that CMPCM's "Tables 1–3 render as images; we have the PFOM *ordering* only, not the numbers." That was true of the Springer **HTML**. The **PDF** carries Table 1 as text:

> `Table 1 PFOM performance evaluation of different edge detection algorithms`
> `Methods  Canny   Log     VPMM    PC      MPC     CMPCM`
> `FOM      0.8888  0.9008  0.9934  0.9099  0.9321  0.9989`

These reproduce the paper's own stated ordering (CMPCM > VPMM > MPC > PC > Log > Canny) exactly, which is the check that they were read off the right row.

**They are not reproduction targets.** The table's PFOM is measured on the paper's Fig. 4a "geometry image" (`176 × 298` — the PDF states that size exactly once, confirming `color-phase-congruency.md` §7.1), and that image's pixels are published nowhere. Knowing the answer does not give us the input. §7.1's design — synthesise a geometric colour image with known ideal edges, regress the *ranking* — stands unchanged. What the numbers add is a tolerance: the `CMPCM 0.9989` / `VPMM 0.9934` gap is `0.0055`, so any ranking regression that must separate them needs better than ~0.5% PFOM resolution. Recorded in `references.md`; out of scope for this plan.

**Model:** Opus, high effort. It is a Sweep, but consistency-critical: a wrong call about what is "biological" versus what is load-bearing mathematics silently corrupts the thing being reviewed.

**Gate (mechanical, then human).** Every check runs over the **whole** sandbox — code, tests, spec, plan, references — not just the kernels:

1. `grep -rniE '<ban-list>' <scratchpad>/math-review/ --exclude-dir=refs --exclude-dir=papers` returns nothing. `refs/` and `papers/` are excluded because they are third-party and read-only; nothing else is.
2. `grep -rn 'import phenotypic' <scratchpad>/math-review/` returns nothing.
3. `grep -rn 'phenotypic\|PhenoTypic' <scratchpad>/math-review/ --exclude-dir=refs --exclude-dir=papers` returns nothing — no repo name, no import path, no URL. Gate 1's `phenotyp` pattern already covers this; run it separately anyway, because a reviewer who learns the project name can search for it.
4. **Manifest completeness:** every row of the table above exists in the sandbox. A silently-missing `plan/plan.md` is the most likely failure, and it is the file F most needs.
5. The stripped `verify_claims.py` runs standalone and prints `21/21 checks passed.` **and** `check_19`'s line contains `max|dpc|` — the same two-part assertion S0 established. It must also exit `0`.
6. The stripped kernels reproduce `tests/fixtures/phasecongmono_golden.npz` to `rtol=1e-6` — proving the transform preserved the logic and not merely the prose.
7. `diff` of each stripped source against its repo original shows **changes confined to docstrings, comments and identifiers** — no expression, constant, or control-flow edit. Mechanise it: strip comments/docstrings from both sides with `ast` and compare `ast.dump()` of the results. Identical dumps, or the transform touched logic.

Gates 6 and 7 are the ones that matter. Without them, "keeping the actual code logic intact" is an assertion; with them, it is a measurement. Gate 7 catches what gate 6 cannot — a logic edit on a path the fixture never exercises.

**Do not let E strip by regex alone.** The ban-list finds the nouns; it does not find *framing* ("a run of a 96-well array", "the operator was tuned on our imaging rig"). A pass that scores 0 on gate 1 can still hand F an obviously applied paper. Read the prose.

## F — Scoped Fable review

A fresh reviewer that has never seen this repository, judging the design on mathematics and faithfulness alone. **This is the last gate; the branch does not merge until F's findings are triaged.**

**Model:** Fable 5 (`claude-fable-5`), high effort. Frontier tier, so it does not review work produced by a stronger model — the rule holds.

**What it receives** — the whole of E's manifest, not just the code:

| Given | Why F needs it |
|---|---|
| `kernels/` + `tests/` + `tests/fixtures/phasecongmono_golden.npz` | the implementation, and the oracle that pins it |
| `verify_claims.py` | 21 executable re-derivations it can run and try to break |
| `spec/` — all six documents | the *claims*. F's job is to check them against the sources, not to trust them |
| `plan/plan.md` + `plan/reviews/` | the reasoning and the prior review. Shortcuts hide in the justification, not the diff |
| `refs/` + `refimpl/` | Kovesi's Julia + MATLAB, `phasepack`, and Kovesi's own (assertion-free) test suite, so every "the reference says X" is checkable at `file:line` |
| `papers/` | Felsberg & Sommer 2001 + 2004, Wietzke & Sommer 2008 (DAGM), Fleischmann 2011 (JMIV), **Shi et al. 2019 (CMPCM)** — five full texts |
| `WebSearch` / `WebFetch` | to consult the literature independently of what our prose asserts |

**Tell F what it does NOT have.** Two artifacts are absent (see E's table): Wang Lijuan et al. (2014), the nominal monogenic-PC source, which *nobody has ever read* — and the unlicensed CMPCM MATLAB. F must not treat their absence as their non-existence, and must not "verify" a claim whose source it cannot open. Any finding that turns on Wang 2014 should be reported as **unverifiable**, not as confirmed or refuted.

Handing over the kernels alone would ask F to re-derive the reasoning it is meant to audit. Handing over the spec without `refs/` would make it take our word for what the references say — the exact failure that produced drift lessons S7 and three misattributions.

**Context discipline:**
- Working directory is `<scratchpad>/math-review/`. The agent is instructed to read **only** files beneath it. No repo paths appear anywhere in its brief.
- No biological framing in the prompt. The task is described purely as a signal-processing port. If F names the application domain unprompted in its report, **E leaked** — treat that as an E failure and re-strip before believing the review.
- Tools: read/grep/glob within the sandbox, `WebSearch` and `WebFetch`, and `Bash` restricted to running the stripped tests (with the same memory ceiling that OOM-killed an earlier reviewer: no array over 5e6 elements, no meshgrid over 2000², total live under 500 MB, scratch runs wrapped in `ulimit -v 4000000`).
- It may read `refs/` and `papers/`. The unlicensed MATLAB under `refs/cmpcm_matlab/` is **read-only cross-check; never copy its code**. `papers/` is **copyrighted and must never be committed**.

**Charge:** the governing principle, stated to it directly — *faithfulness to validated reference logic beats a convenient shortcut.* Specifically: find shortcuts, unstated assumptions, places where the implementation silently picks a side in a reference disagreement, tolerances that are too loose to catch a plausible bug, and tests that could pass while the code is wrong. It should consult the literature and the reference implementations rather than taking the spec's word for what they say — **this spec has already been wrong about that three times**, each time by reading `phasepack` as though it were Kovesi (`perfft2`, the `T` floor, the odd-grid divisor; drift lesson S7).

Give F the mutation-audit table too, and ask the question that table exists to answer: **which of these tests would still pass if the code were wrong?** The audit already found one mutant (`rayleigh_broken`) that survived every behavioural control, and one shipped default (`periodic=False`) that no test pinned at all.

**It does not fix anything.** Findings come back as a report; the orchestrator triages, surfaces design-level conflicts to the user, and only then applies changes to the *real* spec and code.

**Gate:** every finding is either (a) reproduced against the real code and fixed, (b) reproduced and consciously accepted with a new `drift-register.md` row, or (c) refuted with evidence. **No finding is closed by assertion.** The report lands in `docs/superpowers/plans/2026-07-09-focus-edge-monogenic-phase/reviews/`; the sandbox itself never enters the repo.

---

## Notes on ordering

- The background plan review was triaged and applied in `3cd894943` (2 blockers, 4 majors, 3 approved design changes). Dispatch is unblocked.
- The plan's Task 3 Step 1 (the npz move) executes as **S0**, ahead of Task 1, not inside Task 3. Task 3's remaining steps are absorbed into cluster A.
- The plan's Task 2 and Task 4 map to B and C; Task 5 maps to D.
- Per-cluster gates are light (diff read + tests). The two **deep** review gates are after A, and after the B∥C merge. Any design-level question a review raises goes to the user before the next cluster starts, not after.
- **Decided 2026-07-09 (user):** the `n_scale` `ge=1` → `ge=2` narrowing ships *inside* B rather than being split into its own commit, and the back-compat fixture moves to the new edge. The two alternatives considered and rejected were (a) keeping `ge=1` and deferring, and (b) keeping `ge=1` while defining `width = 0.0` at `n_scale == 1`. Recorded because spec §3.3 explicitly left this open ("may be split into its own commit").
