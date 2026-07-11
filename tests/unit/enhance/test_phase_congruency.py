"""
Tests for FocusEdgePhase.

Tests parameter validation, output properties, and basic functionality.
"""

import pytest
import numpy as np
from pydantic import ValidationError

from pathlib import Path

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import FocusEdgePhase
from phenotypic.enhance._monogenic_kernels import monogenic_phase_congruency

_CHARACTERIZATION = (
    Path(__file__).resolve().parents[2] / "fixtures" / "phasecong3_characterization.npz"
)


class TestPhaseCongruencyEnhancerParameterValidation:
    """Test FocusEdgePhase parameter validation.

    The bare-scalar bounds migrated from ``field_validator``s to
    ``Field(ge=, le=, gt=, lt=)`` (the annotations workstream), so these assert
    on the ``ValidationError`` *type* (a ``ValueError`` subclass) rather than the
    old hand-rolled messages — the rejection contract is unchanged.
    """

    def test_n_scale_zero_is_rejected(self):
        """Test that n_scale=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(n_scale=0)

    def test_n_scale_one_is_rejected(self):
        """n_scale=1 divides by (n_scale - 1) and returns an all-zero detect_mat
        (max=0 versus 0.971004 at n_scale=4). Rejected at construction rather
        than producing garbage. See drift-register M3."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(n_scale=1)

    def test_n_scale_two_is_accepted(self):
        assert FocusEdgePhase(n_scale=2).n_scale == 2

    def test_n_orient_less_than_one_raises_error(self):
        """Test that n_orient < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(n_orient=0)

    def test_min_wavelength_less_than_two_raises_error(self):
        """Test that min_wavelength < 2 raises ValidationError."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(min_wavelength=1.5)

    def test_mult_less_than_or_equal_one_raises_error(self):
        """Test that mult <= 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(mult=1.0)
        with pytest.raises(ValidationError):
            FocusEdgePhase(mult=0.5)

    def test_sigma_onf_out_of_range_raises_error(self):
        """Test that sigma_onf outside [0.1, 1.0] raises ValidationError."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(sigma_onf=0.05)
        with pytest.raises(ValidationError):
            FocusEdgePhase(sigma_onf=1.5)

    def test_negative_k_raises_error(self):
        """Test that k < 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(k=-1.0)

    def test_cutoff_out_of_range_raises_error(self):
        """Test that cutoff outside (0, 1) raises ValidationError."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(cutoff=0.0)
        with pytest.raises(ValidationError):
            FocusEdgePhase(cutoff=1.0)

    def test_g_non_positive_raises_error(self):
        """Test that g <= 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            FocusEdgePhase(g=0.0)
        with pytest.raises(ValidationError):
            FocusEdgePhase(g=-5.0)

    def test_invalid_output_raises_error(self):
        """Test that invalid output mode raises ValueError.

        ``output`` is a ``Literal`` field, so an out-of-set value is
        rejected by pydantic with a ``literal_error`` (a subclass of
        ``ValueError``) rather than the legacy hand-rolled message.
        """
        with pytest.raises(ValueError, match="Input should be"):
            FocusEdgePhase(output="invalid")

    def test_valid_parameters_accepted(self):
        """Test that valid parameters are accepted."""
        enhancer = FocusEdgePhase(
            n_scale=4,
            n_orient=6,
            min_wavelength=3.0,
            mult=2.1,
            sigma_onf=0.55,
            k=2.0,
            cutoff=0.5,
            g=10.0,
            noise_method=-1,
            output="pc_sum",
        )
        assert enhancer.n_scale == 4
        assert enhancer.n_orient == 6
        assert enhancer.min_wavelength == 3.0
        assert enhancer.mult == 2.1
        assert enhancer.sigma_onf == 0.55
        assert enhancer.k == 2.0
        assert enhancer.cutoff == 0.5
        assert enhancer.g == 10.0
        assert enhancer.noise_method == -1
        assert enhancer.output == "pc_sum"


def _characterization_scene() -> np.ndarray:
    """The exact image the fixture was generated from. No data loaders, no I/O."""
    rng = np.random.default_rng(20260709)
    n = 64
    img = np.zeros((n, n), dtype=np.float64)
    img[:, n // 2:] = 1.0                       # vertical step edge
    img[14:26, 14:26] = 0.5                     # square -> corners, so `m` is non-trivial
    yy, xx = np.mgrid[0:n, 0:n]
    img[np.abs(yy - xx) < 2] = 0.75             # diagonal -> an off-axis orientation
    img += 0.02 * rng.standard_normal((n, n))   # noise floor -> exercises the T path
    return img


def _edge_at(theta_deg: float, n: int = 128) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    t = np.deg2rad(theta_deg)
    return ((xx - c) * np.sin(t) - (yy - c) * np.cos(t) > 0).astype(np.float64)


def _median_angle_deg(angles_rad: np.ndarray, mask: np.ndarray) -> float:
    """Circular median on a 180-degree period, so wrap-around cannot skew it."""
    v = np.rad2deg(angles_rad[mask]) % 180.0
    best, spread = np.nan, np.inf
    for shift in range(0, 180, 3):
        r = (v + shift) % 180.0
        if r.std() < spread:
            spread, best = r.std(), (np.median(r) - shift) % 180.0
    return float(best)


class TestPhaseCong3Characterization:
    """A numeric oracle for `_phasecong3`. Before this existed, there was none.

    A mutation audit against the whole 585-test suite found that **five of seven**
    single-line mutations to this function survived, including one that moves `pc_sum` by
    `0.81` (most of its range) and one that shifts `M` by `0.025`. Nothing in the repository
    could see them: `test_phase_congruency.py` asserted properties (ranges, shapes,
    monotonicity), never values. The bit-identity check that protected the kernels refactor
    was run by hand and never committed.

    This is a **characterization** fixture, not a golden one. It pins what this code does
    today so a refactor cannot change it silently. It does *not* claim agreement with
    Kovesi -- `phasecong3` has no reference fixture, and drift `M13` records where our
    `orientation` deliberately departs from his Julia. Regenerate only with a conscious
    decision, and say why in the drift register.

    `n_orient=5` is included because it is **odd**: the oriented bank at odd `n_orient` is
    not the even one permuted, and `TuneSpec(4, 8)` admits it.
    """

    # Mechanism, not taste: a 64x64 FFT round-trip accumulates O(N log N) ~ 2.5e4 roundings
    # at ~1.1e-16 each, so ~1e-12 relative. 1e-9 leaves three orders of margin for a
    # different BLAS or FFT backend, and is still eight orders tighter than the smallest
    # surviving mutant (`pc3_covxy_2_over_n`, which moves `M` by 2.5e-02).
    RTOL = 1e-9
    ATOL = 1e-12

    @pytest.mark.parametrize("n_orient", [6, 5])
    @pytest.mark.parametrize("field", ["M", "m", "orientation", "feature_type", "pc_sum"])
    def test_output_matches_the_characterization_fixture(self, n_orient, field):
        golden = np.load(_CHARACTERIZATION, allow_pickle=False)
        result = FocusEdgePhase(n_orient=n_orient)._phasecong3(_characterization_scene())
        np.testing.assert_allclose(
                getattr(result, field), golden[f"o{n_orient}_{field}"],
                rtol=self.RTOL, atol=self.ATOL,
        )

    @pytest.mark.parametrize("n_orient", [6, 5])
    def test_threshold_matches_the_characterization_fixture(self, n_orient):
        golden = np.load(_CHARACTERIZATION, allow_pickle=False)
        result = FocusEdgePhase(n_orient=n_orient)._phasecong3(_characterization_scene())
        assert result.T == pytest.approx(float(golden[f"o{n_orient}_T"]), rel=self.RTOL)


class TestOrientationConvention:
    """Drift `M13`. `0` is a vertical edge, `pi/2` horizontal, positive anticlockwise.

    Kovesi's Julia `phasecong3` reports `pi/2 - phi`, contradicting its own docstring and
    his `phasecongmono` in the same package. Verified by running it: a vertical edge reads
    `-86.35` degrees from `phasecong3` and `-0.03` from `phasecongmono`. His MATLAB
    `phasecong3` satisfies "0 = vertical" but reports `-phi`, violating "anticlockwise".

    We reflect the reported angle. `orientation` is a pure output, so `M`, `m`, `pc_sum` and
    `feature_type` are bit-identical to the unreflected code.
    """

    def test_a_vertical_edge_reads_zero_and_a_horizontal_edge_reads_ninety(self):
        vertical = np.zeros((96, 96))
        vertical[:, 48:] = 1.0
        horizontal = np.zeros((96, 96))
        horizontal[48:, :] = 1.0

        rv = FocusEdgePhase(n_orient=8)._phasecong3(vertical)
        rh = FocusEdgePhase(n_orient=8)._phasecong3(horizontal)

        ov = _median_angle_deg(rv.orientation, rv.pc_sum > 0.4 * rv.pc_sum.max())
        oh = _median_angle_deg(rh.orientation, rh.pc_sum > 0.4 * rh.pc_sum.max())

        assert ov == pytest.approx(0.0, abs=1.0), f"vertical edge read {ov} deg, expected 0"
        assert oh == pytest.approx(90.0, abs=1.0), f"horizontal edge read {oh} deg, expected 90"

    def test_it_agrees_with_the_monogenic_operator_across_angles(self):
        """Both report `phi`. Before the M13 reflection they disagreed by 35.89 deg mean.

        The residual is this operator's angular quantisation over `n_orient=8` filters; the
        monogenic estimator is continuous. Tolerance from that mechanism: half the angular
        spacing is 11.25 deg, and the observed mean is 1.64 -- so 5 deg is loose enough to
        survive a backend change and tight enough that the 35.89 deg pre-fix state fails.
        """
        errors = []
        for theta in range(0, 180, 30):
            img = _edge_at(theta)
            r3 = FocusEdgePhase(n_orient=8)._phasecong3(img)
            o3 = _median_angle_deg(r3.orientation, r3.pc_sum > 0.4 * r3.pc_sum.max())

            rm = monogenic_phase_congruency(img)
            om = _median_angle_deg(rm.orientation, rm.pc > 0.4 * rm.pc.max())

            errors.append(min(abs(o3 - om), 180.0 - abs(o3 - om)))

        assert np.mean(errors) < 5.0, f"mean disagreement {np.mean(errors):.2f} deg"

    def test_the_range_is_the_half_open_half_plane(self):
        r = FocusEdgePhase()._phasecong3(_edge_at(37))
        assert r.orientation.min() > -np.pi / 2 - 1e-9
        assert r.orientation.max() <= np.pi / 2 + 1e-9


class TestSigmaOnfOneIsRejectedAtApplyTime:
    """`FocusEdgePhase(sigma_onf=1.0)` used to return an all-zero / all-NaN detect_mat.

    It reaches `log_gabor_radial` directly and never passes through
    `monogenic_phase_congruency`, so the guard that drift M10 added there did not cover it.
    Measured before the guard moved to `log_gabor_scale`: on a 64x64 step edge every
    log-Gabor filter was identically zero and `.apply()` returned an all-zero map that
    passes a naive `0 <= x <= 1` check; on a real plate it returned all-NaN. Neither raised.

    `FloatRange` appends `high` exactly (`tune/_search_space/_domains.py:86`), so a grid tune
    over the old `TuneSpec(0.1, 1.0)` evaluated exactly 1.0 and scored a dead enhancer.

    The `Field` bound stays `le=1.0` on purpose: `ImagePipeline.from_json` must keep loading
    `enhance_features_sigma_onf_high.json`, which pins `sigma_onf: 1.0`. So construction
    succeeds and `.apply()` raises. Drift M10.
    """

    def test_construction_still_succeeds(self):
        """Legacy pipelines must keep deserialising."""
        assert FocusEdgePhase(sigma_onf=1.0).sigma_onf == 1.0

    def test_apply_raises_rather_than_returning_a_dead_map(self):
        """`.apply()` surfaces a bare `Exception`, not the `ValueError` the kernel raises.

        `ImageOperation` wraps **twice** -- `_apply_to_single_image` at
        `abc_/_image_operation.py:422` and `apply` again at `:469` -- each doing
        `raise Exception(f"{cls_name} failed on image {name}: {e}") from e`. The exception
        *type* is destroyed at both levels; only the message and the `__cause__` chain
        survive. So walk the chain to the original.

        Matching a bare `Exception` alone would pass against anything, which is why this
        asserts on the root cause's type and message too.
        """
        image = load_synth_yeast_plate()
        with pytest.raises(Exception, match="sigma_onf") as excinfo:
            FocusEdgePhase(sigma_onf=1.0).apply(image)

        root = excinfo.value
        while root.__cause__ is not None:
            root = root.__cause__
        assert isinstance(root, ValueError), f"expected a ValueError root cause, got {root!r}"
        assert "sigma_onf must lie strictly in (0, 1)" in str(root)

    def test_the_tune_window_stops_short_of_the_degenerate_point(self):
        """Guard the guard: a grid run must never reach the value that raises."""
        from phenotypic.sdk_.typing_ import TuneSpec

        spec = next(
                a for a in FocusEdgePhase.model_fields["sigma_onf"].metadata
                if isinstance(a, TuneSpec)
        )
        assert spec.high == 0.99
        assert spec.high < 1.0  # the raising value


class TestTheEpsilonSeamIsLocked:
    """`_phasecong3` must hand `spread_weight` phasecong3's 1e-5, not the module's 1e-4.

    Before the kernels refactor, `epsilon = 1e-5` was a local literal inside `_phasecong3`.
    It is now an argument to a shared function whose module constant `EPSILON_MONOGENIC` is
    `1e-4` -- `phasecongmono`'s value, not `phasecong3`'s. That is a seam, and nothing in
    the repository locked it: substituting `1e-4` leaves `tests/unit/enhance` and the
    filamentous detector's suite entirely green while shifting `pc_sum` by 7.48%
    (`max|d| = 6.026e-02`, 469165 / 480000 pixels changed on `load_synth_yeast_plate`).
    """

    def test_phasecong3_passes_phasecong3s_epsilon_to_spread_weight(self, monkeypatch):
        """Capture the value at the call boundary rather than inferring it from output."""
        import phenotypic.enhance._focus_edge_phase as fep

        seen: list[float] = []
        real = fep.spread_weight

        def spy(sum_amplitude, max_amplitude, n_scale, cutoff, g, epsilon):
            seen.append(epsilon)
            return real(sum_amplitude, max_amplitude, n_scale, cutoff, g, epsilon)

        monkeypatch.setattr(fep, "spread_weight", spy)
        FocusEdgePhase(n_scale=2, n_orient=2)._phasecong3(np.zeros((32, 32)))

        assert seen, "spread_weight was never called"
        assert set(seen) == {1e-5}, f"expected phasecong3's 1e-5, got {sorted(set(seen))}"

    def test_the_two_epsilons_are_actually_different(self):
        """Guard the guard: if these ever unify, the test above becomes vacuous."""
        from phenotypic.enhance._monogenic_kernels import EPSILON_MONOGENIC

        assert EPSILON_MONOGENIC == 1e-4
        assert EPSILON_MONOGENIC != 1e-5


class TestPhaseCongruencyEnhancerOutputProperties:
    """Test FocusEdgePhase output properties."""

    @pytest.fixture
    def synthetic_image(self):
        """Create a synthetic test image with edges."""
        # Create 128x128 image with a vertical edge in the middle
        arr = np.zeros((128, 128), dtype=np.float64)
        arr[:, 64:] = 1.0  # Right half is bright
        return Image(arr=arr)

    @pytest.fixture
    def uniform_image(self):
        """Create a uniform (featureless) test image."""
        arr = np.ones((64, 64), dtype=np.float64) * 0.5
        return Image(arr=arr)

    def test_output_shape_preserved(self, synthetic_image):
        """Test that output has same shape as input."""
        enhancer = FocusEdgePhase(n_scale=3, n_orient=4)
        result = enhancer.apply(synthetic_image)
        assert result.detect_mat[:].shape == synthetic_image.detect_mat[:].shape

    def test_output_range_clipped_to_unit_interval(self, synthetic_image):
        """Test that output is in [0, 1] range."""
        enhancer = FocusEdgePhase()
        result = enhancer.apply(synthetic_image)
        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0

    def test_pc_sum_output_mode(self, synthetic_image):
        """Test pc_sum output mode works."""
        enhancer = FocusEdgePhase(output="pc_sum")
        result = enhancer.apply(synthetic_image)
        assert result.detect_mat[:].shape == synthetic_image.detect_mat[:].shape

    def test_M_output_mode(self, synthetic_image):
        """Test M (edge strength) output mode works."""
        enhancer = FocusEdgePhase(output="M")
        result = enhancer.apply(synthetic_image)
        assert result.detect_mat[:].shape == synthetic_image.detect_mat[:].shape

    def test_m_output_mode(self, synthetic_image):
        """Test m (corner strength) output mode works."""
        enhancer = FocusEdgePhase(output="m")
        result = enhancer.apply(synthetic_image)
        assert result.detect_mat[:].shape == synthetic_image.detect_mat[:].shape

    def test_uniform_image_low_response(self, uniform_image):
        """Test that uniform image produces low phase congruency."""
        enhancer = FocusEdgePhase(n_scale=3, n_orient=4)
        result = enhancer.apply(uniform_image)
        # Uniform regions should have low PC values
        assert result.detect_mat[:].mean() < 0.3


class TestPhaseCongruencyEnhancerEdgeDetection:
    """Test FocusEdgePhase edge detection capabilities."""

    def test_vertical_edge_detected(self):
        """Test that vertical edges are detected with high M values."""
        # Create image with sharp vertical edge
        arr = np.zeros((128, 128), dtype=np.float64)
        arr[:, 64:] = 1.0
        image = Image(arr=arr)

        enhancer = FocusEdgePhase(output="M", n_scale=3, n_orient=4)
        result = enhancer.apply(image)

        # Edge region (columns ~60-68) should have higher values than uniform regions
        edge_region = result.detect_mat[:, 60:68]
        left_uniform = result.detect_mat[:, 10:30]
        right_uniform = result.detect_mat[:, 90:110]

        assert edge_region.mean() > left_uniform.mean()
        assert edge_region.mean() > right_uniform.mean()

    def test_horizontal_edge_detected(self):
        """Test that horizontal edges are detected with high M values."""
        # Create image with sharp horizontal edge
        arr = np.zeros((128, 128), dtype=np.float64)
        arr[64:, :] = 1.0
        image = Image(arr=arr)

        enhancer = FocusEdgePhase(output="M", n_scale=3, n_orient=4)
        result = enhancer.apply(image)

        # Edge region (rows ~60-68) should have higher values than uniform regions
        edge_region = result.detect_mat[60:68, :]
        top_uniform = result.detect_mat[10:30, :]
        bottom_uniform = result.detect_mat[90:110, :]

        assert edge_region.mean() > top_uniform.mean()
        assert edge_region.mean() > bottom_uniform.mean()


class TestPhaseCongruencyEnhancerNoiseHandling:
    """Test noise estimation methods."""

    @pytest.fixture
    def noisy_image(self):
        """Create image with step edge and noise."""
        arr = np.zeros((128, 128), dtype=np.float64)
        arr[:, 64:] = 0.8
        # Add some noise
        np.random.seed(42)
        arr += np.random.normal(0, 0.05, arr.shape)
        arr = np.clip(arr, 0, 1)
        return Image(arr=arr)

    def test_noise_method_median(self, noisy_image):
        """Test median noise estimation method (-1)."""
        enhancer = FocusEdgePhase(noise_method=-1, n_scale=3, n_orient=4)
        result = enhancer.apply(noisy_image)
        assert result.detect_mat[:].shape == noisy_image.detect_mat[:].shape

    def test_noise_method_mode(self, noisy_image):
        """Test mode noise estimation method (-2)."""
        enhancer = FocusEdgePhase(noise_method=-2, n_scale=3, n_orient=4)
        result = enhancer.apply(noisy_image)
        assert result.detect_mat[:].shape == noisy_image.detect_mat[:].shape

    def test_noise_method_fixed(self, noisy_image):
        """Test fixed noise threshold (>= 0)."""
        enhancer = FocusEdgePhase(noise_method=0.1, n_scale=3, n_orient=4)
        result = enhancer.apply(noisy_image)
        assert result.detect_mat[:].shape == noisy_image.detect_mat[:].shape

    def test_higher_k_reduces_response(self, noisy_image):
        """Test that higher k (more noise rejection) reduces overall response."""
        enhancer_low_k = FocusEdgePhase(k=2.0, n_scale=3, n_orient=4)
        enhancer_high_k = FocusEdgePhase(k=10.0, n_scale=3, n_orient=4)

        result_low_k = enhancer_low_k.apply(noisy_image)
        result_high_k = enhancer_high_k.apply(noisy_image)

        # Higher k should produce lower overall response (more aggressive thresholding)
        assert result_high_k.detect_mat[:].mean() <= result_low_k.detect_mat[:].mean()


class TestPhaseCongruencyEnhancerIntegration:
    """Integration tests with phenotypic data and Image class."""

    def test_apply_preserves_image_rgb(self):
        """Test that apply() does not modify image.rgb (immutability)."""
        arr = np.random.rand(64, 64, 3).astype(np.float64)
        image = Image(arr=arr)
        original_rgb = image.rgb[:].copy()

        enhancer = FocusEdgePhase(n_scale=3, n_orient=4)
        enhancer.apply(image)

        # Original image rgb should be unchanged
        assert np.array_equal(image.rgb[:], original_rgb)

    def test_apply_preserves_image_gray(self):
        """Test that apply() does not modify image.gray (immutability)."""
        arr = np.random.rand(64, 64).astype(np.float64)
        image = Image(arr=arr)
        original_gray = image.gray[:].copy()

        enhancer = FocusEdgePhase(n_scale=3, n_orient=4)
        enhancer.apply(image)

        # Original image gray should be unchanged
        assert np.array_equal(image.gray[:], original_gray)

    def test_inplace_modifies_original(self):
        """Test that inplace=True modifies the original image."""
        arr = np.random.rand(64, 64).astype(np.float64)
        image = Image(arr=arr)
        original_detect_mat = image.detect_mat[:].copy()

        enhancer = FocusEdgePhase(n_scale=3, n_orient=4)
        result = enhancer.apply(image, inplace=True)

        # Result should be the same object
        assert result is image
        # detect_mat should be modified (not equal to original)
        assert not np.array_equal(image.detect_mat[:], original_detect_mat)


# Run all tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
