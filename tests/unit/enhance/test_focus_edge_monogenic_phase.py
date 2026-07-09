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


def _image_from(arr: np.ndarray) -> Image:
    """Wrap a float array in an Image by broadcasting it to RGB."""
    rgb = np.repeat((arr[..., None] * 255).astype(np.uint8), 3, axis=2)
    return Image(rgb)


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
        # ImagePipeline.ops is a dict keyed by class name (not a list).
        restored_ops = list(restored.ops.values())
        assert isinstance(restored_ops[0], FocusEdgeMonogenicPhase)

    def test_orientation_output_differs_from_pc(self):
        """The three output modes are genuinely different maps, not the same array."""
        pc = FocusEdgeMonogenicPhase(output="pc").apply(load_synth_yeast_plate()).detect_mat[:]
        orient = FocusEdgeMonogenicPhase(output="orientation").apply(
            load_synth_yeast_plate()
        ).detect_mat[:]
        assert not np.allclose(pc, orient)

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


class TestAngleToUnitMap:
    """The affine map ``(theta + pi/2)/pi`` that carries the angle outputs into [0, 1].

    This map is a bijection onto ``(0, 1]`` **only** because the kernel folds
    orientation into ``(-pi/2, pi/2]`` (``_monogenic_kernels.py`` lines 488-489). A
    ``(-pi, pi]`` input would need ``(theta + pi)/(2*pi)`` instead; feeding it to this
    map would send half the range past 1.0, where the clip flattens it. These
    assertions pin the interval/map pairing, and the operation-level test below ties the
    map to the code that actually runs.
    """

    @staticmethod
    def _map(theta):
        return (theta + np.pi / 2) / np.pi

    def test_endpoints_hit_zero_and_one_exactly(self):
        # (-pi/2 + pi/2)/pi = 0/pi = 0 ; (pi/2 + pi/2)/pi = pi/pi = 1
        assert self._map(-np.pi / 2) == 0.0
        assert self._map(np.pi / 2) == pytest.approx(1.0)

    def test_is_a_strictly_increasing_bijection_on_the_fold_interval(self):
        thetas = np.linspace(-np.pi / 2, np.pi / 2, 1001)
        mapped = self._map(thetas)
        assert np.all(np.diff(mapped) > 0)  # injective: strictly increasing
        assert mapped.min() == pytest.approx(0.0)  # onto: covers [0, 1]
        assert mapped.max() == pytest.approx(1.0)
        # invertible: theta = pi*u - pi/2 recovers the input exactly
        np.testing.assert_allclose(np.pi * mapped - np.pi / 2, thetas, atol=1e-12)

    def test_a_half_open_pi_input_would_overflow_the_unit_interval(self):
        """Why the kernel's fold is load-bearing: an angle in (pi/2, pi] -- which the
        fold removes -- maps above 1.0, so the clip would silently collapse it."""
        assert self._map(3 * np.pi / 4) > 1.0

    def test_operation_orientation_is_the_affine_map_of_kernel_orientation(self):
        """The operation output equals ``clip((kernel_orientation + pi/2)/pi, 0, 1)``.

        This ties the endpoints/bijection above to the code that runs: mutating the
        constant in ``_operate`` (e.g. ``pi/2`` -> ``pi``) breaks this assertion.
        """
        from phenotypic.enhance._monogenic_kernels import monogenic_phase_congruency

        mat = np.asarray(load_synth_yeast_plate().detect_mat[:], dtype=np.float64)
        kernel_orientation = monogenic_phase_congruency(mat).orientation
        expected = np.clip((kernel_orientation + np.pi / 2) / np.pi, 0.0, 1.0)

        produced = np.asarray(
            FocusEdgeMonogenicPhase(output="orientation").apply(load_synth_yeast_plate()).detect_mat[:],
            dtype=np.float64,
        )
        np.testing.assert_allclose(produced, expected, rtol=1e-5, atol=1e-6)


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
