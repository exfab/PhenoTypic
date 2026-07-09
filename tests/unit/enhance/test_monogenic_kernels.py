"""Unit tests for the shared frequency-domain kernels.

These functions are pure: no ``Image``, no fixtures, no I/O. Every assertion here
is a property of the maths, checkable by hand.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.fft import fft2, ifft2

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


def _sum_riesz_channels(
        img: np.ndarray,
        n_scale: int = 4,
        min_wavelength: float = 3.0,
        mult: float = 2.1,
        sigma_onf: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """``(sum_h1, sum_h2)`` rebuilt from the primitives, independent of the final block.

    Lets a test compare the kernel's folded ``orientation`` against the raw
    ``arctan2(-sum_h2, sum_h1)`` without trusting the kernel to report it.
    """
    rows, cols = img.shape
    radius, _, _, _, fx, fy = construct_filter_grids(rows, cols)
    riesz = riesz_multiplier(fx, fy, radius)
    lowpass = lowpass_filter(radius)
    spectrum = fft2(img)  # periodic=False, the shipped default

    sum_h1 = np.zeros((rows, cols))
    sum_h2 = np.zeros((rows, cols))
    for s in range(n_scale):
        band = spectrum * log_gabor_scale(
                radius, lowpass, min_wavelength * (mult ** s), sigma_onf
        )
        odd = ifft2(band * riesz)
        sum_h1 += odd.real
        sum_h2 += odd.imag
    return sum_h1, sum_h2


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
        """Real part carries the ``-fy`` channel, imaginary part the ``fx`` channel.

        Bit-exact, because we divide componentwise (Kovesi's arithmetic). Had we written
        the naive numpy ``(1j*fx - fy) / radius``, this would only hold to ~0.707 ulp:
        numpy promotes the real denominator and runs ``nc_quot``, which reduces to
        ``scl = 1/radius`` and a multiply. Exactness here is therefore a *consequence* of
        the faithfulness fix, and it guards against silently reverting to it.
        """
        radius, _, _, _, fx, fy = construct_filter_grids(16, 16)
        riesz = riesz_multiplier(fx, fy, radius)
        assert np.array_equal(riesz.real, -(fy / radius))
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
        """Pins the componentwise division, and pins that it is NOT the numpy expression.

        All three references print the same glyphs -- `phasecongmono.m:183`
        `H = (1i*u1 - u2)./radius`, `frequencyfilt.jl:238` `H = (im.*fx .- fy)./f`,
        `phasepack/phasecongmono.py:156` `H = (1j*u1 - u2)/radius` -- but the languages
        disagree beneath them:

        * MATLAB `./` and Julia `/(z::Complex, x::Real)` (`base/complex.jl:348`,
          `Complex(real(z)/x, imag(z)/x)`) divide each component. True division.
        * numpy promotes the real denominator and runs `nc_quot`; with a zero imaginary
          part that branch is `scl = 1/r` then a *multiply*. Different rounding.

        So the naive numpy port is bit-faithful to `phasepack` -- an untested third-party
        transcription with a known odd-grid bug -- and not to Kovesi. Confirmed by running
        `frequencyfilt.jl:238` in Julia and comparing IEEE-754 bit patterns: componentwise
        matches, numpy does not. Drift `M8`.

        `sintheta` and `costheta` are exactly `fx/radius` and `fy/radius`, so
        `1j*sintheta - costheta` is the componentwise form and must be bit-equal.
        """
        radius, sintheta, costheta, _, fx, fy = construct_filter_grids(64, 64)
        shipped = riesz_multiplier(fx, fy, radius)

        # What Kovesi computes: bit-exact.
        assert np.array_equal(shipped, 1j * sintheta - costheta)

        # What phasepack computes: reachable, and NOT what we ship.
        numpy_form = (1j * fx - fy) / radius
        assert not np.array_equal(shipped, numpy_form)

        # Bounded by the rounding fork alone. |riesz| == 1 off DC, so each component is
        # <= 1; the two forms differ only in the rounding of a division whose result has
        # magnitude at most 1. Two ulps at 1.0 bounds it (measured 0.707).
        assert np.abs(shipped - numpy_form).max() < 2 * np.spacing(1.0)


class TestPeriodicFft2:
    def test_dc_bin_is_preserved_exactly(self):
        """The mean belongs to the periodic component: ``smooth_fft[0,0]`` is forced to 0.

        Bit-exact, not ``approx``, because ``x - 0.0 == x`` is an algebraic identity, not
        reassociated float maths. And the image matters. ``smooth``'s entries cancel by
        construction, so on a small, modest-amplitude image the unforced
        ``smooth_fft[0,0]`` rounds to exactly ``0j`` and deleting the forcing is
        invisible. Scale a zero-mean image up and the cancellation leaves a residual:
        ``1e3 * step2line(64)`` gives ``3.6e-12``.

        Mutating away ``smooth_fft[0, 0] = 0.0`` survived the entire suite -- including
        this test's earlier ``pytest.approx`` form on ``normal(16,16) + 5.0``, whose
        residual is exactly ``0j`` -- until both halves below were fixed.
        """
        rng = np.random.default_rng(0)
        modest = rng.normal(size=(16, 16)) + 5.0
        exposing = 1e3 * unit_variance(step2line(64))  # unforced residual 3.6e-12

        assert periodic_fft2(modest)[0, 0] == fft2(modest)[0, 0]
        assert periodic_fft2(exposing)[0, 0] == fft2(exposing)[0, 0]

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
        """Kovesi: `edges = 0:mx/nbins:mx; n = histc(data, edges)` (phasecongmono.m:467).

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

    def test_the_rel_015_window_is_tight_enough_to_see_a_doubled_mode(self):
        """Guard the guard: prove `rel=0.15` above is narrow enough to catch a 2x error.

        A wide tolerance makes `test_recovers_the_scale_of_a_rayleigh_sample` decorative.
        This asserts the window both admits the true estimate and rejects twice it.

        The earlier form of this test read
        `assert rayleigh_mode(s) != pytest.approx(2 * rayleigh_mode(s), rel=0.15)`,
        which compares the function against twice *itself* and therefore holds for any
        non-zero return. Verified: under a doubled-`rayleigh_mode` mutant it PASSED,
        while its name and docstring claimed to catch exactly that. This form fails.
        """
        rng = np.random.default_rng(3)
        sigma = 2.0
        sample = sigma * np.sqrt(rng.normal(size=200_000) ** 2 + rng.normal(size=200_000) ** 2)
        estimate = rayleigh_mode(sample)  # measured 1.8124, i.e. 9.4% low

        assert estimate == pytest.approx(sigma, rel=0.15)  # the guard admits the truth...
        assert 2 * estimate != pytest.approx(sigma, rel=0.15)  # ...and rejects a doubling


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

        The `-2` branch is NOT covered by the golden fixture, whose `_params` record
        `noiseMethod = -1`. This is the only test that exercises it, and it pins the
        plumbing around `rayleigh_mode`: the geometric bandwidth sum, the Rayleigh
        mean/sigma constants, `k`, and the epsilon floor. Verified by mutation: it kills
        `k = 3.0 -> 2.0` and a flipped `periodic` default.

        It does NOT catch a wrong `rayleigh_mode`, because it recomputes `tau` by calling
        `rayleigh_mode` itself -- double the estimator and both sides double. Verified:
        this test PASSES under a doubled-mode mutant. That mutant is killed by
        `TestRayleighMode::test_recovers_the_scale_of_a_rayleigh_sample` and
        `::test_bins_are_anchored_at_zero_and_zeros_are_retained`, which compare against
        independently known values.
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


class TestOrientationIsFoldedIntoTheHalfOpenHalfPlane:
    """`MonogenicResult.orientation` is documented as `(-pi/2, pi/2]`. Pin it.

    `arctan2` alone yields `(-pi, pi]`. The golden orientation check compares
    `np.fix((orientation % np.pi)/np.pi*180)`, and `% np.pi` is invariant under a +/-pi
    shift -- so deleting the fold leaves all five fixture images agreeing to 0 degrees.
    It survived the whole suite. What it breaks is downstream: Task C maps this angle to
    `detect_mat`'s `[0, 1]` with a straight affine on `(-pi/2, pi/2]`, and a `(-pi, pi]`
    input silently doubles the range.

    Kovesi's own two implementations differ here and neither returns this interval:
    `phasecongmono.m:292-295` folds to `[0, pi)` and quantises to integer degrees;
    `phasecongruency.jl:580` leaves single-arg `atan`'s `(-pi/2, pi/2)`. We return the
    Julia interval, closed at `+pi/2` because `arctan2` reaches it when `sum_h1 == 0`.
    """

    def test_orientation_lies_in_the_half_open_half_plane(self):
        img = unit_variance(starsine(64, ncycles=8))
        orientation = monogenic_phase_congruency(img).orientation
        assert (orientation > -np.pi / 2).all()
        assert (orientation <= np.pi / 2).all()

    def test_the_fold_is_a_fold_not_a_clamp_and_it_is_not_vacuous(self):
        img = unit_variance(starsine(64, ncycles=8))
        orientation = monogenic_phase_congruency(img).orientation

        sum_h1, sum_h2 = _sum_riesz_channels(img)
        raw = np.arctan2(-sum_h2, sum_h1)  # (-pi, pi]

        # A fold shifts each pixel by exactly 0 or exactly +/-pi. A clamp would park
        # out-of-range pixels at the boundary, leaving some other residual. Bound from
        # mechanism: pi has ulp 4.4e-16, and at most two roundings separate the forms.
        shift = np.abs(orientation - raw)
        assert np.minimum(shift, np.abs(shift - np.pi)).max() < 1e-12

        # ...and the fold is not a no-op: starsine drives sum_h1 negative on half the
        # image. Measured 50.0% of 4096 pixels move.
        assert (shift > 1e-12).mean() > 0.25


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
