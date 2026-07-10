"""Cross-channel fusion of monogenic phase congruency.

Every assertion here is tied to a claim in
``docs/superpowers/specs/2026-07-08-alt-phase-detection/color-phase-congruency.md``, and
the numeric ones are re-derived independently by that spec's
``logic_validation_scripts/2026-07-09-focus-edge-color-phase/fusion_algebra.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.enhance._color_phase_kernels import (
    FUSIONS,
    ColorPhaseResult,
    color_phase_congruency,
    fuse_coherent,
    fuse_joint,
    fuse_l2,
)
from phenotypic.enhance._monogenic_kernels import (
    EPSILON_MONOGENIC,
    monogenic_channel_response,
    monogenic_phase_congruency,
)

_KWARGS = dict(cutoff=0.5, g=10.0, deviation_gain=1.5, epsilon=EPSILON_MONOGENIC)


def _step(rows: int = 48, cols: int = 48) -> np.ndarray:
    """A vertical step edge -- the canonical positive control."""
    return np.add.outer(np.zeros(rows), np.arange(cols) > cols // 2).astype(float)


def _three_channels(seed: int = 11, n_scale: int = 4):
    """Three channels from three genuinely different images."""
    rng = np.random.default_rng(seed)
    base = _step()
    return [
        monogenic_channel_response(base + 0.1 * rng.normal(size=base.shape), n_scale=n_scale)
        for _ in range(3)
    ]


class TestZeroChromaReducesToTheMonogenicPort:
    """Spec §7 test 1. With both chroma weights at 0, every mode returns the port.

    Bit-identical, not ``rtol``. ``sqrt((1.0 * x) ** 2) == x`` holds exactly in IEEE-754
    round-to-nearest for all finite non-negative ``x`` (verified over 2.4M samples), so even
    ``l2`` -- which squares and re-roots -- reduces exactly.

    This also pins the *unexposed* ``orientation`` and ``feature_type`` (drift ``C15``) for
    free: at zero chroma the fused vector collapses to ``v_L``.
    """

    @pytest.mark.parametrize("fusion", FUSIONS)
    def test_pc_and_both_angles_are_bit_identical(self, fusion):
        rng = np.random.default_rng(7)
        img = _step() + 0.05 * rng.normal(size=(48, 48))
        distractors = [rng.normal(size=(48, 48)) for _ in range(2)]

        channels = [monogenic_channel_response(img)] + [
            monogenic_channel_response(d) for d in distractors
        ]
        fused = color_phase_congruency(
                channels, np.array([1.0, 0.0, 0.0]), fusion=fusion, n_scale=4
        )
        reference = monogenic_phase_congruency(img)

        assert np.array_equal(fused.pc, reference.pc), f"{fusion}: pc moved"
        assert np.array_equal(fused.orientation, reference.orientation)
        assert np.array_equal(fused.feature_type, reference.feature_type)
        assert fused.threshold == reference.threshold
        assert fused.n_clamped == reference.n_clamped == 0

    def test_a_nonzero_chroma_weight_really_does_change_the_answer(self):
        """Guard the guard. If the distractors could not perturb the result, the reduction
        above would be vacuous -- it would hold for *any* weights, and prove nothing.

        Note what is and is not true of the distractors. Their ``pc`` is small (max ``0.04``
        on white noise, because the Rayleigh threshold ``T`` is estimated from that very
        noise and kills it). But their **amplitude** is large -- ``sum_amplitude.max``
        ``4.67`` -- and amplitude is what enters ``joint``'s shared denominator. So a
        non-zero chroma weight moves ``pc`` by ``0.53`` absolute. That asymmetry *is*
        §3.3's claim: a channel with amplitude but no phase agreement vetoes rather than
        asserts.
        """
        rng = np.random.default_rng(7)
        img = _step() + 0.05 * rng.normal(size=(48, 48))
        channels = [monogenic_channel_response(img)] + [
            monogenic_channel_response(rng.normal(size=(48, 48))) for _ in range(2)
        ]
        reference = monogenic_phase_congruency(img).pc

        for distractor in channels[1:]:
            assert distractor.sum_amplitude.max() > 1.0, (
                "a flat distractor would make the zero-weight reduction vacuous"
            )

        perturbed = color_phase_congruency(
                channels, np.array([1.0, 0.5, 0.5]), fusion="joint", n_scale=4
        ).pc
        assert not np.array_equal(perturbed, reference)
        assert np.abs(perturbed - reference).max() > 0.1, (
            "chroma at weight 0.5 must visibly move the response, or the zero-weight "
            "reduction is testing nothing"
        )


class TestTheAcosArgumentNeverLeavesTheUnitInterval:
    """Spec §6. ``E_total <= A_total`` analytically, so ``n_clamped`` must be 0.

    Two triangle inequalities: ``||sum_s v_is|| <= sum_s ||v_is||`` per channel gives
    ``E_joint <= A_total``; ``||sum_i w_i v_i|| <= sum_i w_i ||v_i||`` gives
    ``E_coherent <= E_joint``. ``fusion_algebra.py`` check 03 confirms it over 200k draws.
    """

    @pytest.mark.parametrize("fusion", FUSIONS)
    @pytest.mark.parametrize("seed", range(4))
    def test_n_clamped_is_zero_under_random_weights(self, fusion, seed):
        rng = np.random.default_rng(seed)
        channels = _three_channels(seed=seed)
        weights = np.array([1.0, rng.uniform(0, 8), rng.uniform(0, 8)])
        result = color_phase_congruency(channels, weights, fusion=fusion, n_scale=4)
        assert result.n_clamped == 0
        assert np.isfinite(result.pc).all()

    def test_a_negative_weight_is_rejected_because_the_bound_needs_it(self):
        """Not cosmetic validation: a negative weight can break ``E_total <= A_total``."""
        channels = _three_channels()
        with pytest.raises(ValueError, match="non-negative"):
            color_phase_congruency(
                    channels, np.array([1.0, -0.5, 1.0]), fusion="joint", n_scale=4
            )


class TestFusionSanity:
    """Spec §7 test 2. An edge in all three channels must outscore the same edge in one.

    The L2-numerator-over-L1-denominator form fails this outright: at the shipped
    ``deviation_gain=1.5`` its three-channel response is **exactly 0** (spec §3.1,
    ``fusion_algebra.py`` check 01), so a coherent colour edge would be annihilated while a
    single-channel fringe passed at full strength -- inverting §7.2's acceptance criterion.
    """

    @staticmethod
    def _edge_and_flat():
        edge = _step()
        flat = np.zeros((48, 48))
        all_three = [monogenic_channel_response(edge) for _ in range(3)]
        one_only = [monogenic_channel_response(edge)] + [
            monogenic_channel_response(flat) for _ in range(2)
        ]
        return all_three, one_only

    @pytest.mark.parametrize("fusion", ["joint", "coherent"])
    def test_three_coherent_channels_outscore_one(self, fusion):
        all_three, one_only = self._edge_and_flat()
        w = np.ones(3)
        a = color_phase_congruency(all_three, w, fusion=fusion, n_scale=4).pc.max()
        b = color_phase_congruency(one_only, w, fusion=fusion, n_scale=4).pc.max()
        assert a > b, f"{fusion}: three coherent channels scored {a} <= one channel's {b}"

    def test_the_l2_over_l1_form_would_invert_it(self):
        """The mutation this regression exists to catch, executed rather than described."""
        all_three, one_only = self._edge_and_flat()
        w = np.ones(3)

        def rogue(channels):
            """`fuse_joint` with an L2 numerator over its L1 denominator."""
            from phenotypic.enhance._color_phase_kernels import _weighted_scalars
            from phenotypic.enhance._monogenic_kernels import congruency_from_accumulators
            energy = np.sqrt(sum((wi * c.energy) ** 2 for wi, c in zip(w, channels)))
            a_total, t_total, a_max = _weighted_scalars(channels, w)
            pc, _ = congruency_from_accumulators(
                    energy, a_total, a_max, t_total, n_scale=4, **_KWARGS
            )
            return pc.max()

        assert rogue(all_three) < rogue(one_only), (
            "the L2-over-L1 form must invert the ranking -- if it no longer does, this "
            "regression has stopped guarding anything"
        )


class TestL2IsNotDividedByTheWeightNorm:
    """Drift ``C3``. The paper does not normalise; §7.1 must see the paper's quantity.

    Three **identical** channels make the arithmetic exact: ``l2 = sqrt(sum_i (w_i F)**2)``
    ``= ||w|| * F``. Dividing by ``||w||`` would instead return ``F``. Those two are far
    apart, so the test discriminates rather than merely permits.

    Note that ``l2`` does **not** exceed ``1`` at ``w = (1,1,1)``: a real step edge has
    ``F.max = 0.5138``, and ``sqrt(3) * 0.5138 = 0.8899``. It needs ``||w|| > 1/0.5138``
    ``= 1.946``. An earlier draft of this test asserted ``> 1.0`` at unit weights and failed
    -- correctly. ``[0, ||w||]`` is a *bound*, not a value that gets attained.
    """

    def test_l2_is_the_weight_norm_times_the_single_channel_response(self):
        step = _step()
        identical = [monogenic_channel_response(step) for _ in range(3)]
        single = monogenic_phase_congruency(step).pc

        for weights in (np.ones(3), np.array([1.0, 2.0, 2.0])):
            out = color_phase_congruency(identical, weights, fusion="l2", n_scale=4).pc
            np.testing.assert_allclose(out, np.linalg.norm(weights) * single, rtol=1e-12)
            assert not np.allclose(out, single), (
                "l2 must NOT be divided by ||w||; dividing would collapse it onto the "
                "single-channel response"
            )

    def test_l2_exceeds_one_once_the_weight_norm_does(self):
        step = _step()
        identical = [monogenic_channel_response(step) for _ in range(3)]
        out = color_phase_congruency(
                identical, np.array([1.0, 2.0, 2.0]), fusion="l2", n_scale=4
        ).pc
        assert out.max() > 1.0, "l2 is returned un-clipped; the caller clips (drift C3)"

    def test_joint_and_coherent_are_already_bounded(self):
        channels = _three_channels()
        for fusion in ("joint", "coherent"):
            out = color_phase_congruency(channels, np.ones(3), fusion=fusion, n_scale=4).pc
            assert 0.0 <= out.min() and out.max() <= 1.0


class TestJointVetoesWhereL2CannotSpec32:
    """Spec §3.2: ``l2`` has **no cross-channel interaction at all**.

    A loud, incoherent chroma channel must drag ``joint``'s response down (its amplitude
    enters the shared denominator) and must leave ``l2``'s luminance response untouched (it
    combines three finished maps). That contrast is the entire argument for the default.
    """

    def test_incoherent_chroma_vetoes_joint_but_not_l2(self):
        rng = np.random.default_rng(3)
        edge = _step()
        noise = 3.0 * rng.normal(size=edge.shape)  # amplitude, no phase agreement

        quiet = [monogenic_channel_response(edge)] + [
            monogenic_channel_response(np.zeros_like(edge)) for _ in range(2)
        ]
        loud = [monogenic_channel_response(edge)] + [
            monogenic_channel_response(noise) for _ in range(2)
        ]
        w = np.ones(3)

        joint_quiet = color_phase_congruency(quiet, w, fusion="joint", n_scale=4).pc.max()
        joint_loud = color_phase_congruency(loud, w, fusion="joint", n_scale=4).pc.max()
        assert joint_loud < joint_quiet, "joint must let incoherent chroma veto the edge"

        # l2's luminance term is computed in isolation, so the edge survives untouched.
        l2_quiet, _, _ = fuse_l2(quiet, w, n_scale=4, **_KWARGS)
        l2_loud, _, _ = fuse_l2(loud, w, n_scale=4, **_KWARGS)
        luminance_only = monogenic_phase_congruency(edge).pc

        assert l2_quiet.max() == pytest.approx(luminance_only.max()), (
            "with flat chroma, l2 must reduce to the luminance congruency"
        )
        assert l2_loud.max() >= luminance_only.max(), (
            "l2 combines finished maps; a noisy chroma channel can only ADD to the "
            "root-sum-of-squares, never veto"
        )


class TestOnlyCoherentBuildsTheVectorThatProducedItsPc:
    """Drift ``C15``, stated as an executable fact rather than a docstring claim.

    ``coherent``'s energy IS the norm of the fused vector. ``joint``'s is the weighted sum
    of the per-channel norms. They coincide only when the channels' vectors are parallel.
    """

    def test_coherent_energy_is_the_fused_vector_norm_and_joint_is_not(self):
        from phenotypic.enhance._color_phase_kernels import _fused_vector

        channels = _three_channels(seed=5)
        w = np.array([1.0, 2.0, 3.0])
        v_even, v_h1, v_h2 = _fused_vector(channels, w)
        coherent_energy = np.sqrt(v_even ** 2 + v_h1 ** 2 + v_h2 ** 2)
        joint_energy = sum(wi * c.energy for wi, c in zip(w, channels))

        assert (coherent_energy <= joint_energy + 1e-12).all(), "triangle inequality"
        assert not np.allclose(coherent_energy, joint_energy), (
            "on non-parallel channels the two energies must differ -- if they do not, "
            "the test image no longer distinguishes the two fusion modes"
        )


class TestTheC10InventionsHaveTheRightShape:
    """Drift ``C10``: ``T_total = sum_i w_i T_i`` and ``A_max = sum_i w_i max_s A_is`` have
    **no reference**. Nothing external validates them, so pin the properties they must have
    for the rest of the formula to mean anything.

    Found by mutation testing: replacing ``A_max``'s weighted **sum** with an elementwise
    **max across channels** left all 35 other tests green. Both are 1-homogeneous in ``w``,
    so §4.2's argument cannot tell them apart. What separates them is the *meaning* of
    ``width``.
    """

    def test_width_stays_in_the_unit_interval(self):
        """``width = (A_total/(A_max + eps) - 1)/(n_scale - 1)`` must land in ``[0, 1]``.

        Because ``sum_s A_is <= n_scale * max_s A_is`` per channel, summing over channels
        with non-negative weights gives ``A_total <= n_scale * A_max``. An elementwise max
        across channels breaks this: three equal channels at unit weight make ``A_total``
        three times larger while leaving ``A_max`` unchanged, so ``width`` reaches ``3``
        and the spread sigmoid saturates.
        """
        from phenotypic.enhance._color_phase_kernels import _weighted_scalars

        n_scale = 4
        channels = _three_channels(seed=9, n_scale=n_scale)
        for weights in (np.ones(3), np.array([1.0, 7.0, 0.25])):
            a_total, _, a_max = _weighted_scalars(channels, weights)
            width = (a_total / (a_max + EPSILON_MONOGENIC) - 1.0) / (n_scale - 1)
            assert width.min() >= -1e-9, f"width below 0: {width.min()}"
            assert width.max() <= 1.0 + 1e-9, (
                f"width above 1 ({width.max():.4f}): A_max is no longer a weighted SUM "
                f"over channels, so A_total/A_max escaped [1, n_scale]"
            )

    def test_a_max_is_a_weighted_sum_over_channels_not_an_elementwise_max(self):
        """Three copies of one channel must treble ``A_max``, not leave it alone.

        Asserted on ``A_max`` itself rather than on ``width``, deliberately. Routing this
        through ``width`` would compare ``3A/(3A_max + eps)`` against ``A/(A_max + eps)``,
        and those are **not** equal -- ``eps`` is not 1-homogeneous, which is drift ``C17``,
        the very thing that made §4.2's old "~1%" claim false. A first draft of this test
        asserted their equality at ``rtol=1e-12`` and failed on correct code.
        """
        from phenotypic.enhance._color_phase_kernels import _weighted_scalars

        single = monogenic_channel_response(_step(), n_scale=4)
        a_total, _, a_max = _weighted_scalars([single] * 3, np.ones(3))

        np.testing.assert_array_equal(a_max, 3.0 * single.max_amplitude)
        np.testing.assert_array_equal(a_total, 3.0 * single.sum_amplitude)
        assert not np.allclose(a_max, single.max_amplitude), (
            "an elementwise max across channels would leave A_max unchanged, treble "
            "A_total/A_max, and push `width` outside [0, 1]"
        )

    def test_epsilon_is_why_that_is_asserted_on_a_max_and_not_on_width(self):
        """Pin the reason, so nobody 'simplifies' the test above back into a width check.

        ``width`` is invariant to a global weight rescale only in the limit ``eps -> 0``.
        Measured here on a real step edge; the residual is ``O(eps/A_max)`` and is exactly
        drift ``C17``'s mechanism, in miniature.
        """
        from phenotypic.enhance._color_phase_kernels import _weighted_scalars

        n_scale = 4
        single = monogenic_channel_response(_step(), n_scale=n_scale)
        a_total, _, a_max = _weighted_scalars([single] * 3, np.ones(3))

        def width(numer, denom):
            return (numer / (denom + EPSILON_MONOGENIC) - 1.0) / (n_scale - 1)

        trebled = width(a_total, a_max)
        alone = width(single.sum_amplitude, single.max_amplitude)
        residual = np.abs(trebled - alone).max()

        assert residual > 0.0, "if eps were 1-homogeneous these would coincide exactly"
        assert residual < 1e-3, f"residual {residual:.3e} is larger than O(eps/A_max)"

    def test_the_three_denominators_are_one_homogeneous_in_the_weights(self):
        """§4.2's two-degrees-of-freedom argument rests on exactly this."""
        from phenotypic.enhance._color_phase_kernels import _weighted_scalars

        channels = _three_channels(seed=2)
        weights = np.array([1.0, 3.0, 0.5])
        base = _weighted_scalars(channels, weights)
        scaled = _weighted_scalars(channels, 4.0 * weights)

        np.testing.assert_allclose(scaled[0], 4.0 * base[0], rtol=1e-12)
        assert scaled[1] == pytest.approx(4.0 * base[1], rel=1e-12)
        np.testing.assert_allclose(scaled[2], 4.0 * base[2], rtol=1e-12)


class TestL2AppliesEachChannelsOwnThresholdAndSumsTheClampCounts:
    """``l2`` runs three independent congruencies. Two facts nothing else could observe.

    ``n_clamped`` is ``0`` on every real input (drift ``M1``), so an aggregation bug in it
    is invisible to every other test here -- mutation testing confirmed that keeping only
    the **last** channel's count leaves all 35 green. A spy is the only way to see it.
    """

    def test_l2_calls_the_congruency_once_per_channel_with_that_channels_threshold(
            self, monkeypatch
    ):
        import phenotypic.enhance._color_phase_kernels as cpk

        channels = _three_channels(seed=4)
        seen: list[float] = []
        counts = iter([2, 3, 5])

        def spy(energy, sum_amplitude, max_amplitude, threshold, **kwargs):
            seen.append(threshold)
            return np.zeros_like(sum_amplitude), next(counts)

        monkeypatch.setattr(cpk, "congruency_from_accumulators", spy)
        _, threshold, n_clamped = cpk.fuse_l2(channels, np.ones(3), n_scale=4, **_KWARGS)

        assert seen == [c.threshold for c in channels], (
            "each channel must be thresholded with its own T_i, never with T_total"
        )
        assert n_clamped == 10, "n_clamped must SUM across the three channels"
        assert threshold == pytest.approx(sum(c.threshold for c in channels)), (
            "the reported threshold is sum_i w_i T_i, for reporting only (drift C10)"
        )

    def test_joint_calls_the_congruency_once_with_the_fused_threshold(self, monkeypatch):
        import phenotypic.enhance._color_phase_kernels as cpk

        channels = _three_channels(seed=4)
        weights = np.array([1.0, 2.0, 3.0])
        seen: list[float] = []

        def spy(energy, sum_amplitude, max_amplitude, threshold, **kwargs):
            seen.append(threshold)
            return np.zeros_like(sum_amplitude), 0

        monkeypatch.setattr(cpk, "congruency_from_accumulators", spy)
        cpk.fuse_joint(channels, weights, n_scale=4, **_KWARGS)

        expected = float(sum(w * c.threshold for w, c in zip(weights, channels)))
        assert len(seen) == 1, "joint evaluates the congruency exactly once"
        assert seen[0] == pytest.approx(expected)


class TestWeightVectorScaleInvariance:
    """Spec §7 test 4, and it **cannot live on the operation**.

    ``FocusEdgeColorPhase`` pins ``weights[0] = 1.0``, so a *global* rescale ``w -> c*w`` is
    not expressible through its fields at all -- that is the whole point of §4.2's
    two-degrees-of-freedom argument. The invariance is a property of
    :func:`color_phase_congruency`, where ``weights`` is a free vector, and it is tested
    here. The spec said to test it on ``load_synth_yeast_plate()`` through the operation;
    that instruction is not implementable.

    The invariance is **approximate, direction-dependent, and only holds on a masked set**.
    ``eps`` sits inside ``A_max + eps``, which is fed to a sigmoid of sharpness ``g``. Under
    ``w -> c*w`` the relative perturbation of ``width`` grows like ``eps/(c * A_max)``, so
    **shrinking** the weights (``c < 1``) hurts and growing them helps. Measured on
    ``load_synth_yeast_plate`` at ``w = (1, 2, 3)``:

    | mask | ``c = 0.01`` | ``c = 100`` |
    |---|---|---|
    | none | ``1.0`` | ``22.3`` |
    | ``pc > 0.05`` | ``6.88e-02`` | ``7.13e-04`` |
    | ``pc > 0.2`` | ``4.16e-02`` | ``4.25e-04`` |

    The retracted ``rtol=2e-2`` would fail at ``c = 0.01`` even masked. Drift ``C17``.
    """

    WEIGHTS = np.array([1.0, 2.0, 3.0])
    MASK_FLOOR = 0.05

    @staticmethod
    def _plate_channels():
        from phenotypic.data import load_synth_yeast_plate
        from phenotypic.enhance import FocusEdgeColorPhase

        image = load_synth_yeast_plate()
        return [
            monogenic_channel_response(channel)
            for channel in FocusEdgeColorPhase()._extract_channels(image)
        ]

    def _deviation(self, channels, c, floor):
        base = color_phase_congruency(channels, self.WEIGHTS, fusion="joint", n_scale=4).pc
        scaled = color_phase_congruency(
            channels, c * self.WEIGHTS, fusion="joint", n_scale=4
        ).pc
        mask = base > floor
        return float((np.abs(scaled[mask] - base[mask]) / base[mask]).max())

    def test_unmasked_the_invariance_fails_outright(self):
        """If it held unmasked, the mask would be doing no work and the test no thinking."""
        channels = self._plate_channels()
        assert self._deviation(channels, 0.01, 0.0) > 0.5
        assert self._deviation(channels, 100.0, 0.0) > 0.5

    def test_masked_the_invariance_holds_within_the_measured_bound(self):
        channels = self._plate_channels()
        assert self._deviation(channels, 0.01, self.MASK_FLOOR) < 0.10
        assert self._deviation(channels, 100.0, self.MASK_FLOOR) < 2e-3

    def test_shrinking_the_weights_hurts_far_more_than_growing_them(self):
        """The mechanism, not just the magnitude. ``eps/(c*A_max)`` grows as ``c -> 0``.

        A test that only asserted "both are small" would pass if ``eps`` moved into
        ``E + eps`` instead -- which is where drift ``C17`` records the spec wrongly put it.
        """
        channels = self._plate_channels()
        shrunk = self._deviation(channels, 0.01, self.MASK_FLOOR)
        grown = self._deviation(channels, 100.0, self.MASK_FLOOR)
        assert shrunk > 10.0 * grown, f"shrunk {shrunk:.3e} vs grown {grown:.3e}"

    def test_the_ratio_itself_is_exactly_one_homogeneous(self):
        """What §4.2 actually needs: ``E_total/A_total`` is scale-free. No ``eps``."""
        from phenotypic.enhance._color_phase_kernels import _weighted_scalars

        channels = self._plate_channels()
        for c in (0.01, 100.0):
            a_base, t_base, m_base = _weighted_scalars(channels, self.WEIGHTS)
            a_sc, t_sc, m_sc = _weighted_scalars(channels, c * self.WEIGHTS)
            np.testing.assert_allclose(a_sc, c * a_base, rtol=1e-12)
            np.testing.assert_allclose(m_sc, c * m_base, rtol=1e-12)
            assert t_sc == pytest.approx(c * t_base, rel=1e-12)


class TestArgumentValidation:
    def test_unknown_fusion_raises(self):
        with pytest.raises(ValueError, match="fusion must be one of"):
            color_phase_congruency(_three_channels(), np.ones(3), fusion="mean", n_scale=4)

    @pytest.mark.parametrize("n_channels", [2, 4])
    def test_wrong_channel_count_raises(self, n_channels):
        channels = _three_channels()[:1] * n_channels
        with pytest.raises(ValueError, match="expected 3 channels"):
            color_phase_congruency(channels, np.ones(3), fusion="joint", n_scale=4)

    def test_wrong_weight_shape_raises(self):
        with pytest.raises(ValueError, match="expected 3 channels"):
            color_phase_congruency(_three_channels(), np.ones(2), fusion="joint", n_scale=4)

    def test_mismatched_channel_shapes_raise(self):
        channels = _three_channels()[:2] + [monogenic_channel_response(_step(32, 32))]
        with pytest.raises(ValueError, match="share a shape"):
            color_phase_congruency(channels, np.ones(3), fusion="joint", n_scale=4)

    def test_nan_weight_raises(self):
        with pytest.raises(ValueError, match="finite"):
            color_phase_congruency(
                    _three_channels(), np.array([1.0, np.nan, 1.0]), fusion="joint", n_scale=4
            )

    @pytest.mark.parametrize("bad", [1, 0])
    def test_n_scale_below_two_propagates_from_the_kernel(self, bad):
        """Drift ``M9``: the guard lives in ``congruency_from_accumulators``, not here."""
        with pytest.raises(ValueError, match="n_scale"):
            color_phase_congruency(_three_channels(), np.ones(3), fusion="joint", n_scale=bad)


class TestTheResultIsAFrozenDataclass:
    def test_fields_and_immutability(self):
        result = color_phase_congruency(
                _three_channels(), np.ones(3), fusion="joint", n_scale=4
        )
        assert isinstance(result, ColorPhaseResult)
        with pytest.raises(Exception):
            result.pc = np.zeros((1, 1))  # type: ignore[misc]


class TestTheDispatchIsExhaustive:
    def test_every_fusion_name_has_a_function(self):
        assert set(FUSIONS) == {"joint", "coherent", "l2"}
        for name, fn in (("joint", fuse_joint), ("coherent", fuse_coherent), ("l2", fuse_l2)):
            pc, threshold, n_clamped = fn(_three_channels(), np.ones(3), n_scale=4, **_KWARGS)
            assert pc.shape == (48, 48)
            assert isinstance(threshold, float)
            assert n_clamped == 0
