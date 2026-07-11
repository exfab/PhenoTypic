"""``FocusEdgeColorPhase``: construction, channel order, guards, and field parity.

Numeric fidelity, the ``[0,1]`` bound, scale invariance, the hue-wrap artefact and the
PFOM ranking regression live in the sibling files added alongside this one.
"""

from __future__ import annotations

import json
from typing import get_args

import numpy as np
import pytest
from pydantic import ValidationError
from skimage.exposure import rescale_intensity

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import FocusEdgeColorPhase, FocusEdgeMonogenicPhase
from phenotypic.enhance._focus_edge_color_phase import _LUMINANCE_FIRST
from phenotypic.sdk_.typing_ import (
    ColorPhaseOutput,
    ColorSpaceName,
    PhaseFusion,
    PhaseLift,
)


def _causes(exc: BaseException) -> list[BaseException]:
    """Walk the whole ``__cause__`` chain.

    ``ImageOperation.apply`` wraps a raised ``ValueError`` **twice** -- into a
    ``RuntimeError`` at ``abc_/_image_operation.py:423`` and then a bare ``Exception`` at
    ``:470`` -- so the original type survives only on the chain. Measured chain:
    ``RuntimeError -> Exception -> ValueError``.
    """
    chain, current = [], exc
    while current is not None:
        chain.append(current)
        current = current.__cause__
    return chain


class TestConstruction:
    def test_constructible_with_no_arguments(self):
        op = FocusEdgeColorPhase()
        assert (op.color_space, op.fusion, op.lift, op.output) == (
            "lab", "joint", "monogenic", "pc",
        )
        assert op.chroma_weight_1 == op.chroma_weight_2 == 1.0
        assert op.norm == "clip"

    @pytest.mark.parametrize("operation", [FocusEdgeColorPhase, FocusEdgeMonogenicPhase])
    def test_norm_is_appended_in_schema_and_serialization(self, operation):
        assert list(operation.model_fields)[-1] == "norm"
        assert list(operation.model_json_schema()["properties"])[-1] == "norm"
        assert list(json.loads(operation().to_json())["params"])[-1] == "norm"

    def test_construction_is_keyword_only(self):
        with pytest.raises(TypeError):
            FocusEdgeColorPhase("lab")  # type: ignore[misc]

    def test_unknown_kwarg_is_rejected(self):
        with pytest.raises(ValidationError):
            FocusEdgeColorPhase(colour_space="lab")  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "field,bad",
        [
            ("chroma_weight_1", -0.1),
            ("chroma_weight_2", -1.0),
            ("n_scale", 1),
            ("mult", 1.0),
            ("sigma_onf", 1.0),
            ("cutoff", 0.0),
            ("g", 0.0),
            ("deviation_gain", 0.0),
            ("min_wavelength", 1.5),
            ("color_space", "rgb"),
            ("fusion", "mean"),
            ("output", "pc_sum"),
            ("norm", "normalize"),
        ],
    )
    def test_out_of_range_fields_are_rejected(self, field, bad):
        with pytest.raises(ValidationError):
            FocusEdgeColorPhase(**{field: bad})

    def test_sigma_onf_just_below_one_is_accepted(self):
        assert FocusEdgeColorPhase(sigma_onf=0.999).sigma_onf == 0.999


class TestTheGatedConformalLift:
    """Drift ``C5``/``C15``'s neighbour: the field exists, the path does not."""

    def test_conformal_lift_raises_not_implemented_at_construction(self):
        with pytest.raises(NotImplementedError, match="conformal"):
            FocusEdgeColorPhase(lift="conformal")

    def test_pydantic_does_not_swallow_the_type(self):
        """pydantic v2 traps ValueError/AssertionError only; everything else propagates.

        If this ever regresses to ``ValidationError``, the raise must move into
        ``_operate`` and the test must walk the ``__cause__`` chain instead. Do not leave a
        test asserting a type the code cannot raise.
        """
        with pytest.raises(NotImplementedError):
            FocusEdgeColorPhase(lift="conformal")
        assert not issubclass(NotImplementedError, ValueError)

    def test_monogenic_lift_is_the_default_and_constructs(self):
        assert FocusEdgeColorPhase().lift == "monogenic"


class TestChannelOrderIsLuminanceFirst:
    """Spec §4.2. ``lab -> (L*, a*, b*)``, ``hsv -> (V, H, S)``.

    Read literally, an earlier §3 (*"in color_space channel order"*) pinned **``H``** at
    ``1.0`` under ``hsv`` -- the one channel that is circular, ill-conditioned at low
    saturation, and carries no luminance. §4.2's table pins ``V``, and wins.
    """

    def test_the_permutation_table_is_what_the_spec_says(self):
        assert _LUMINANCE_FIRST == {"lab": (0, 1, 2), "hsv": (2, 0, 1)}
        assert set(_LUMINANCE_FIRST) == set(get_args(ColorSpaceName))

    def test_lab_channels_are_L_then_a_then_b(self):
        image = load_synth_yeast_plate()
        channels = FocusEdgeColorPhase(color_space="lab")._extract_channels(image)
        lab = image.color.Lab[:]
        for i, channel in enumerate(channels):
            np.testing.assert_array_equal(channel, lab[..., i].astype(np.float64))

    def test_hsv_channels_are_V_then_H_then_S_not_H_then_S_then_V(self):
        image = load_synth_yeast_plate()
        channels = FocusEdgeColorPhase(color_space="hsv")._extract_channels(image)
        hsv = image.color.hsv[:]

        np.testing.assert_array_equal(channels[0], hsv[..., 2].astype(np.float64))  # V
        np.testing.assert_array_equal(channels[1], hsv[..., 0].astype(np.float64))  # H
        np.testing.assert_array_equal(channels[2], hsv[..., 1].astype(np.float64))  # S

        assert not np.array_equal(channels[0], hsv[..., 0]), (
            "channel 0 must be V, not H -- the pinned weight belongs to luminance"
        )

    def test_the_pinned_channel_is_the_one_with_the_largest_dynamic_range(self):
        """A sanity net: on a real plate, luminance dominates. §4.1."""
        image = load_synth_yeast_plate()
        for space in get_args(ColorSpaceName):
            channels = FocusEdgeColorPhase(color_space=space)._extract_channels(image)
            spreads = [float(c.std()) for c in channels]
            assert spreads[0] == max(spreads), (
                f"{space}: channel 0 (std {spreads[0]:.4f}) is not the widest of {spreads}"
            )


class TestAchromaticInputIsRejected:
    """Drift ``C11``. ``a*``/``b*`` vanish and ``joint`` divides a congruency by itself.

    The achromatic image is **constructed**, not made by writing greyscale into a plate's
    ``rgb``. That assignment raises ``assignment destination is read-only``: the accessor
    hands back an immutable view, which is the ``@validate_operation_integrity`` contract
    working as designed. An earlier draft of this test mutated ``rgb`` and failed for that
    reason rather than for the reason it was testing.
    """

    @staticmethod
    def _achromatic_image():
        import phenotypic

        grey = np.zeros((64, 64), dtype=np.uint8)
        grey[:, 32:] = 200  # a step edge, so the image is not degenerate in any other way
        return phenotypic.Image(np.stack([grey] * 3, axis=-1))

    def test_apply_raises_value_error_through_the_wrapper(self):
        with pytest.raises(Exception) as excinfo:
            FocusEdgeColorPhase().apply(self._achromatic_image())

        chain = _causes(excinfo.value)
        assert any(
            isinstance(err, ValueError) and "chromatic" in str(err) for err in chain
        ), f"no ValueError about chromatic input on the chain: {[type(e).__name__ for e in chain]}"

    def test_the_helper_raises_it_directly_unwrapped(self):
        with pytest.raises(ValueError, match="chromatic"):
            FocusEdgeColorPhase()._extract_channels(self._achromatic_image())

    @pytest.mark.parametrize("space", ["lab", "hsv"])
    def test_both_colour_spaces_reject_it(self, space):
        """Under ``hsv`` an achromatic image has ``S == 0`` and undefined ``H``."""
        with pytest.raises(ValueError, match="chromatic"):
            FocusEdgeColorPhase(color_space=space)._extract_channels(self._achromatic_image())

    def test_a_genuinely_chromatic_plate_is_accepted(self):
        """Guard the guard: the shipped plate must not trip the achromatic check."""
        FocusEdgeColorPhase()._extract_channels(load_synth_yeast_plate())

    def test_a_single_least_significant_bit_of_chroma_is_enough(self):
        """One pixel, one LSB. The guard is exact equality, not approximate.

        This also records why substituting ``np.allclose`` for ``np.array_equal`` is an
        **equivalent mutant** rather than a defect: ``rgb`` is always an integer array
        (``uint8`` on the shipped plates, ``uint16`` when built from floats), so two
        channels that differ do so by at least one LSB. ``allclose``'s tolerance at the top
        of the ``uint8`` range is ``atol + rtol*255 = 2.6e-3``, far below ``1``. The two
        predicates therefore agree on every reachable input. A mutation sweep confirmed the
        substitution survives the whole suite, and it should -- an equivalent mutant cannot
        be killed, and a test written to kill it would be testing nothing.

        ``array_equal`` is still the right call: it says what is meant, costs less, and
        stays correct if ``rgb`` ever becomes floating point.
        """
        import phenotypic

        grey = np.zeros((64, 64), dtype=np.uint8)
        grey[:, 32:] = 200
        rgb = np.stack([grey] * 3, axis=-1)
        rgb[0, 0, 1] += 1  # exactly one LSB of chroma, in exactly one pixel

        image = phenotypic.Image(rgb)
        assert image.rgb[:].dtype.kind in "ui", "the equivalence argument needs integer rgb"
        FocusEdgeColorPhase()._extract_channels(image)  # must not raise


class TestFieldParityWithTheMonogenicPort:
    """The nine shared fields are duplicated, not inherited. They must not drift.

    A mixin would have put two ``BaseModel`` bases in the MRO of a class the operation
    registry and ``from_json`` both walk; subclassing would have made
    ``isinstance(colour, FocusEdgeMonogenicPhase)`` true, which it is not. Duplication plus
    this test is the cheaper, *checkable* option -- a mixin would only have been an
    assertion.
    """

    SHARED = [
        "n_scale", "min_wavelength", "mult", "sigma_onf", "k",
        "deviation_gain", "cutoff", "g", "noise_method", "norm",
    ]

    @pytest.mark.parametrize("name", SHARED)
    def test_default_matches(self, name):
        colour = FocusEdgeColorPhase.model_fields[name]
        mono = FocusEdgeMonogenicPhase.model_fields[name]
        assert colour.default == mono.default, f"{name}: default drifted"

    @pytest.mark.parametrize("name", SHARED)
    def test_bounds_and_tune_spec_match(self, name):
        colour = FocusEdgeColorPhase.model_fields[name]
        mono = FocusEdgeMonogenicPhase.model_fields[name]
        assert repr(colour.metadata) == repr(mono.metadata), f"{name}: bounds drifted"

    def test_no_shared_field_was_forgotten(self):
        mono = set(FocusEdgeMonogenicPhase.model_fields) - {"output"}
        assert mono == set(self.SHARED), (
            "FocusEdgeMonogenicPhase gained or lost a field; mirror it on the colour "
            f"operation. Difference: {mono.symmetric_difference(set(self.SHARED))}"
        )

    def test_the_colour_operation_adds_exactly_the_colour_fields(self):
        extra = set(FocusEdgeColorPhase.model_fields) - set(FocusEdgeMonogenicPhase.model_fields)
        assert extra == {"color_space", "fusion", "chroma_weight_1", "chroma_weight_2", "lift"}


class TestTheClosedValueSets:
    def test_literals_match_the_shipped_members(self):
        assert set(get_args(ColorSpaceName)) == {"lab", "hsv"}
        assert set(get_args(PhaseFusion)) == {"joint", "coherent", "l2"}
        assert set(get_args(PhaseLift)) == {"monogenic", "conformal"}

    def test_color_output_matches_the_monogenic_response_set(self):
        assert get_args(ColorPhaseOutput) == ("pc", "orientation", "feature_type")

    def test_the_fusion_literal_matches_the_kernel_dispatch(self):
        from phenotypic.enhance._color_phase_kernels import FUSIONS

        assert set(get_args(PhaseFusion)) == set(FUSIONS)


class TestTheOperationIsExactlyTheKernelWithItsOwnFields:
    """Every field must reach the kernel, with the right value, in the right slot.

    Written after a mutation sweep found that **seven** distinct source mutations survived
    the suite -- all of them invisible at default parameters. ``deviation_gain`` is the
    sharpest example: the kernel's own default *is* ``1.5``, so deleting
    ``deviation_gain=self.deviation_gain`` from the call site changes nothing until someone
    constructs the operation with a different gain. Likewise ``n_scale=4`` hardcoded at the
    fusion call, and ``weights = [cw1, 1.0, cw2]`` with all weights at ``1.0``.

    So: construct with **every** field off its default, and compare against the kernel
    invoked by hand.
    """

    #: Every field off its default. ``cutoff``/``g``/``deviation_gain`` are chosen so the
    #: frequency-spread sigmoid is **not saturated** -- see
    #: ``test_the_chosen_configuration_can_actually_see_n_scale``.
    NON_DEFAULT = dict(
        color_space="hsv", fusion="coherent",
        chroma_weight_1=2.5, chroma_weight_2=0.25,
        n_scale=5, min_wavelength=3.5, mult=2.2, sigma_onf=0.5,
        k=6.0, deviation_gain=1.0, cutoff=0.65, g=5.0, noise_method=-2.0,
    )

    def _hand_rolled(self, image, *, n_scale=None):
        from phenotypic.enhance._color_phase_kernels import color_phase_congruency
        from phenotypic.enhance._monogenic_kernels import monogenic_channel_response

        cfg = dict(self.NON_DEFAULT)
        fusion_scale = cfg["n_scale"] if n_scale is None else n_scale
        hsv = image.color.hsv[:]
        channels = [
            monogenic_channel_response(
                np.asarray(hsv[..., index], dtype=np.float64),
                n_scale=cfg["n_scale"], min_wavelength=cfg["min_wavelength"],
                mult=cfg["mult"], sigma_onf=cfg["sigma_onf"], k=cfg["k"],
                noise_method=cfg["noise_method"],
            )
            for index in (2, 0, 1)  # V, H, S
        ]
        return color_phase_congruency(
            channels,
            np.array([1.0, cfg["chroma_weight_1"], cfg["chroma_weight_2"]]),
            fusion=cfg["fusion"], n_scale=fusion_scale,
            cutoff=cfg["cutoff"], g=cfg["g"], deviation_gain=cfg["deviation_gain"],
        )

    def test_bit_identical_to_a_hand_rolled_kernel_call(self):
        image = load_synth_yeast_plate()
        actual = FocusEdgeColorPhase(**self.NON_DEFAULT)._color_phase_congruency(image)
        expected = self._hand_rolled(image)

        assert np.array_equal(actual.pc, expected.pc)
        assert np.array_equal(actual.orientation, expected.orientation)
        assert np.array_equal(actual.feature_type, expected.feature_type)
        assert actual.threshold == expected.threshold
        assert actual.n_clamped == expected.n_clamped == 0

    def test_the_chosen_configuration_can_actually_see_n_scale(self):
        """Guard the guard, and it caught a real hole.

        The first draft of ``NON_DEFAULT`` used ``cutoff=0.35, g=14.0, deviation_gain=1.2``.
        There, ``pc`` is non-zero only where the phase-deviation term survives, and *there*
        the frequency-spread sigmoid is saturated -- so passing ``n_scale=4`` to the fusion
        while building the channels at ``n_scale=5`` changed the output by **exactly
        `0.0`**. The bit-identity test above sailed through a genuine call-site desync.

        A bit-identity test is only as good as its configuration's sensitivity. Assert it.
        """
        image = load_synth_yeast_plate()
        matched = self._hand_rolled(image).pc
        desynced = self._hand_rolled(image, n_scale=4).pc
        assert np.abs(matched - desynced).max() > 0.01, (
            "this configuration cannot distinguish the fusion's n_scale from the channels' "
            "-- pick cutoff/g/deviation_gain that leave the spread sigmoid unsaturated"
        )


class TestTheOperationForwardsItsFieldsVerbatim:
    """Spy on the kernel call. **This, not a numeric comparison, is what catches desync.**

    A mutation sweep found that hardcoding ``n_scale=4`` at the fusion call site survived
    every numeric test in this file, because whether it is visible depends on
    ``cutoff``/``g``/``deviation_gain``. The forwarded keyword arguments are checkable
    directly, and the check does not depend on the image or the parameter regime at all.
    """

    NON_DEFAULT = TestTheOperationIsExactlyTheKernelWithItsOwnFields.NON_DEFAULT

    def test_the_fusion_receives_exactly_the_operations_fields(self, monkeypatch):
        import phenotypic.enhance._focus_edge_color_phase as module

        real = module.color_phase_congruency
        seen: dict = {}

        def spy(channels, weights, **kwargs):
            seen["weights"] = np.asarray(weights).copy()
            seen["kwargs"] = dict(kwargs)
            seen["n_channels"] = len(channels)
            return real(channels, weights, **kwargs)

        monkeypatch.setattr(module, "color_phase_congruency", spy)
        op = FocusEdgeColorPhase(**self.NON_DEFAULT)
        op._color_phase_congruency(load_synth_yeast_plate())

        assert seen["n_channels"] == 3
        assert seen["kwargs"] == {
            "fusion": op.fusion,
            "n_scale": op.n_scale,
            "cutoff": op.cutoff,
            "g": op.g,
            "deviation_gain": op.deviation_gain,
        }
        np.testing.assert_array_equal(
            seen["weights"], np.array([1.0, op.chroma_weight_1, op.chroma_weight_2])
        )

    def test_each_channel_is_built_with_the_operations_scale_parameters(self, monkeypatch):
        import phenotypic.enhance._focus_edge_color_phase as module

        real = module.monogenic_channel_response
        calls: list[dict] = []

        def spy(image, **kwargs):
            calls.append(dict(kwargs))
            return real(image, **kwargs)

        monkeypatch.setattr(module, "monogenic_channel_response", spy)
        op = FocusEdgeColorPhase(**self.NON_DEFAULT)
        op._color_phase_congruency(load_synth_yeast_plate())

        assert len(calls) == 3, "one monogenic chain per colour channel"
        expected = {
            "n_scale": op.n_scale,
            "min_wavelength": op.min_wavelength,
            "mult": op.mult,
            "sigma_onf": op.sigma_onf,
            "k": op.k,
            "noise_method": op.noise_method,
        }
        for call in calls:
            assert call == expected

    def test_the_fusion_scale_equals_the_channel_scale(self, monkeypatch):
        """The desync, stated as its own invariant rather than inferred from a number."""
        import phenotypic.enhance._focus_edge_color_phase as module

        real_fuse, real_channel = module.color_phase_congruency, module.monogenic_channel_response
        scales: dict = {"channels": set(), "fusion": None}

        def channel_spy(image, **kwargs):
            scales["channels"].add(kwargs["n_scale"])
            return real_channel(image, **kwargs)

        def fuse_spy(channels, weights, **kwargs):
            scales["fusion"] = kwargs["n_scale"]
            return real_fuse(channels, weights, **kwargs)

        monkeypatch.setattr(module, "monogenic_channel_response", channel_spy)
        monkeypatch.setattr(module, "color_phase_congruency", fuse_spy)

        for n_scale in (2, 4, 5, 6):
            scales["channels"].clear()
            FocusEdgeColorPhase(n_scale=n_scale)._color_phase_congruency(
                load_synth_yeast_plate()
            )
            assert scales["channels"] == {n_scale}
            assert scales["fusion"] == n_scale, (
                f"channels built at n_scale={n_scale} but fused at {scales['fusion']}; "
                f"spread_weight would divide by the wrong (n_scale - 1)"
            )

    @pytest.mark.parametrize(
        "field,alternative,least_change",
        [
            ("n_scale", 5, 0.3),
            ("deviation_gain", 1.0, 0.3),
            ("cutoff", 0.35, 0.01),
            ("g", 14.0, 0.01),
            ("k", 6.0, 0.01),
            ("mult", 2.5, 0.01),
            ("sigma_onf", 0.4, 0.01),
            ("min_wavelength", 4.0, 0.01),
            ("noise_method", -2.0, 1e-6),
            ("chroma_weight_1", 4.0, 0.01),
            ("chroma_weight_2", 4.0, 0.01),
            ("fusion", "l2", 0.3),
            ("color_space", "hsv", 0.1),
        ],
    )
    def test_every_field_actually_moves_the_output(self, field, alternative, least_change):
        """A field that changes nothing is a field that is not being passed through."""
        image = load_synth_yeast_plate()
        baseline = FocusEdgeColorPhase()._color_phase_congruency(image).pc
        altered = FocusEdgeColorPhase(**{field: alternative})._color_phase_congruency(image).pc
        moved = float(np.abs(altered - baseline).max())
        assert moved >= least_change, (
            f"{field}={alternative!r} moved the output by only {moved:.3e}; it is probably "
            f"not reaching the kernel"
        )

    def test_the_two_chroma_weights_are_not_interchangeable(self):
        """``chroma_weight_1`` weights ``a*``; ``chroma_weight_2`` weights ``b*``.

        Swapping them at the call site survived every other test in this file.
        """
        image = load_synth_yeast_plate()
        a = FocusEdgeColorPhase(chroma_weight_1=4.0, chroma_weight_2=0.0)
        b = FocusEdgeColorPhase(chroma_weight_1=0.0, chroma_weight_2=4.0)
        difference = np.abs(
            a._color_phase_congruency(image).pc - b._color_phase_congruency(image).pc
        ).max()
        assert difference > 0.1, f"(4,0) and (0,4) differ by only {difference:.3e}"

    def test_luminance_is_pinned_at_one_not_permuted_into_a_chroma_slot(self):
        """``weights[0]`` is luminance's, always. §4.2.

        With both chroma weights at ``0`` the result must be the *luminance* congruency. If
        the weight vector were built as ``[cw1, 1.0, cw2]`` this would silently become the
        ``a*`` congruency instead -- and, with the defaults all equal to ``1.0``, nothing
        else in the suite would notice.
        """
        from phenotypic.enhance._monogenic_kernels import monogenic_phase_congruency

        image = load_synth_yeast_plate()
        op = FocusEdgeColorPhase(chroma_weight_1=0.0, chroma_weight_2=0.0)
        luminance = np.asarray(image.color.Lab[..., 0], dtype=np.float64)

        np.testing.assert_array_equal(
            op._color_phase_congruency(image).pc, monogenic_phase_congruency(luminance).pc
        )


class TestZeroChromaReducesToTheMonogenicPort:
    """Spec §7 test 1. The fusion must not perturb the port -- bit-identically.

    Also pins the diagnostic ``orientation`` and ``feature_type`` outputs (drift ``C15``):
    at zero chroma the fused vector collapses to ``v_L``.
    """

    @pytest.mark.parametrize("fusion", ["joint", "coherent", "l2"])
    def test_pc_and_both_angles_match_focus_edge_monogenic_phase(self, fusion):
        from phenotypic.enhance._monogenic_kernels import monogenic_phase_congruency

        image = load_synth_yeast_plate()
        result = FocusEdgeColorPhase(
            fusion=fusion, chroma_weight_1=0.0, chroma_weight_2=0.0
        )._color_phase_congruency(image)
        reference = monogenic_phase_congruency(
            np.asarray(image.color.Lab[..., 0], dtype=np.float64)
        )

        np.testing.assert_allclose(result.pc, reference.pc, rtol=1e-10)
        np.testing.assert_allclose(result.orientation, reference.orientation, rtol=1e-10)
        np.testing.assert_allclose(result.feature_type, reference.feature_type, rtol=1e-10)


class TestTheThreeFusionModesAreDistinct:
    """If two modes agreed, one of them would not be implemented.

    Measured on ``load_synth_yeast_plate`` at unit weights:
    ``|joint - coherent| = 0.406``, ``|joint - l2| = 0.658``, ``|coherent - l2| = 0.735``.
    """

    @pytest.mark.parametrize(
        "left,right,least", [("joint", "coherent", 0.2), ("joint", "l2", 0.3), ("coherent", "l2", 0.3)]
    )
    def test_modes_differ_on_a_real_plate(self, left, right, least):
        image = load_synth_yeast_plate()
        a = FocusEdgeColorPhase(fusion=left)._color_phase_congruency(image).pc
        b = FocusEdgeColorPhase(fusion=right)._color_phase_congruency(image).pc
        assert np.abs(a - b).max() > least


class TestOutputMapsAndNormalization:
    @pytest.mark.parametrize("fusion", ["joint", "coherent", "l2"])
    @pytest.mark.parametrize("output", ["orientation", "feature_type"])
    def test_angle_outputs_are_the_normalized_fused_vector_maps(self, fusion, output):
        op = FocusEdgeColorPhase(fusion=fusion, output=output)
        result = op._color_phase_congruency(load_synth_yeast_plate())
        source = getattr(result, output)
        expected = np.clip((source + np.pi / 2) / np.pi, 0.0, 1.0)

        actual = op.apply(load_synth_yeast_plate()).detect_mat[:]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("output", ["orientation", "feature_type"])
    @pytest.mark.parametrize("norm", ["rescale", None])
    def test_norm_does_not_change_angle_maps(self, output, norm):
        baseline = FocusEdgeColorPhase(output=output, norm="clip").apply(
            load_synth_yeast_plate()
        ).detect_mat[:]
        altered = FocusEdgeColorPhase(output=output, norm=norm).apply(
            load_synth_yeast_plate()
        ).detect_mat[:]
        np.testing.assert_array_equal(altered, baseline)

    def test_l2_pc_obeys_all_norm_policies(self):
        raw = FocusEdgeColorPhase(fusion="l2")._color_phase_congruency(
            load_synth_yeast_plate()
        ).pc
        assert raw.max() > 1.0

        clipped = FocusEdgeColorPhase(fusion="l2", norm="clip").apply(
            load_synth_yeast_plate()
        ).detect_mat[:]
        rescaled = FocusEdgeColorPhase(fusion="l2", norm="rescale").apply(
            load_synth_yeast_plate()
        ).detect_mat[:]
        unbounded = FocusEdgeColorPhase(fusion="l2", norm=None).apply(
            load_synth_yeast_plate()
        ).detect_mat[:]

        np.testing.assert_allclose(clipped, np.clip(raw, 0.0, 1.0), atol=1e-6)
        np.testing.assert_allclose(
            rescaled, rescale_intensity(raw, out_range=(0.0, 1.0)), atol=1e-6
        )
        np.testing.assert_allclose(unbounded, raw, atol=1e-6)
        assert unbounded.max() > 1.0


class TestTheUnitIntervalBoundAndTheClip:
    """Spec §7 test 3, and the default clip that makes it true."""

    def test_l2_exceeds_one_before_the_clip_even_at_default_weights(self):
        """The clip is load-bearing with **no tuning at all**.

        Unclipped ``l2`` on ``load_synth_yeast_plate`` reaches ``1.0989`` at
        ``w = (1,1,1)``. ``detect_mat.__setitem__`` enforces ``float32`` but does **not**
        clamp -- writing ``2.5`` reads back ``2.5``.
        """
        image = load_synth_yeast_plate()
        unclipped = FocusEdgeColorPhase(fusion="l2")._color_phase_congruency(image).pc
        assert unclipped.max() > 1.0, f"unclipped l2 max is {unclipped.max():.6f}"

        written = FocusEdgeColorPhase(fusion="l2").apply(load_synth_yeast_plate()).detect_mat[:]
        assert written.max() <= 1.0

    @pytest.mark.parametrize("fusion", ["joint", "coherent", "l2"])
    @pytest.mark.parametrize("seed", range(4))
    def test_detect_mat_is_finite_and_in_the_unit_interval(self, fusion, seed):
        rng = np.random.default_rng(seed)
        op = FocusEdgeColorPhase(
            fusion=fusion,
            chroma_weight_1=float(rng.uniform(0, 8)),
            chroma_weight_2=float(rng.uniform(0, 8)),
        )
        out = op.apply(load_synth_yeast_plate()).detect_mat[:]
        assert np.isfinite(out).all(), "NaN passes a naive 0 <= x <= 1 check (drift M10)"
        assert 0.0 <= out.min() and out.max() <= 1.0

    @pytest.mark.parametrize("fusion", ["joint", "coherent", "l2"])
    def test_the_acos_clamp_stays_inert(self, fusion):
        """Drift ``M1``: ``n_clamped`` must be ``0`` on every shipped image."""
        for weights in ((1.0, 1.0), (8.0, 8.0), (0.0, 0.0)):
            op = FocusEdgeColorPhase(
                fusion=fusion, chroma_weight_1=weights[0], chroma_weight_2=weights[1]
            )
            assert op._color_phase_congruency(load_synth_yeast_plate()).n_clamped == 0

    @pytest.mark.parametrize("bound", [0.0, 8.0])
    def test_both_ends_of_the_tune_spec_window_are_legal_at_apply_time(self, bound):
        """Drift ``M10``'s lesson. ``FloatRange`` appends ``high`` exactly, so a grid run
        evaluates ``chroma_weight = 8.0``. It must not raise, and must not return NaN.
        """
        op = FocusEdgeColorPhase(chroma_weight_1=bound, chroma_weight_2=bound)
        out = op.apply(load_synth_yeast_plate()).detect_mat[:]
        assert np.isfinite(out).all()
        assert 0.0 <= out.min() and out.max() <= 1.0


class TestHueWrapArtifactIsReal:
    """Drift ``C16``, **demonstrated** rather than asserted.

    A flat image of constant ``S`` and ``V`` whose hue ramps smoothly *through* red contains
    no edge. ``color_space="hsv"`` band-passes raw ``H`` across its ``0.996 -> 0.002`` seam
    and manufactures one; ``color_space="lab"`` does not.

    The border ring must be excluded. The FFT treats the image as tiled, so wrap-around at
    the frame edge dominates ``pc.max`` and swamps the effect -- a first draft of this test
    compared global maxima, measured ``hsv/lab = 0.99x``, and concluded the artefact was not
    real. Interior only, it is ``115.7x`` over background.

    If this test ever goes green on ``hsv``, someone has started unwrapping hue and has
    silently diverged from CMPCM.
    """

    MARGIN = 24

    @staticmethod
    def _hue_ramp_through_red(size: int = 128):
        import colorsys

        import phenotypic

        hue = (np.linspace(-0.12, 0.12, size) % 1.0)
        rgb = np.zeros((size, size, 3))
        for column, h in enumerate(hue):
            rgb[:, column] = colorsys.hsv_to_rgb(h, 0.6, 0.6)
        return phenotypic.Image((rgb * 255).round().astype(np.uint8)), size // 2

    def _seam_and_background(self, color_space):
        image, seam = self._hue_ramp_through_red()
        pc = FocusEdgeColorPhase(color_space=color_space)._color_phase_congruency(image).pc
        interior = pc[self.MARGIN:-self.MARGIN, self.MARGIN:-self.MARGIN]
        columns = np.arange(self.MARGIN, pc.shape[1] - self.MARGIN)
        band = np.abs(columns - seam) <= 2
        return float(interior[:, band].max()), float(interior[:, ~band].max())

    def test_the_input_really_has_no_luminance_edge(self):
        """Guard the guard: if V or S varied, both spaces would respond and prove nothing."""
        image, _ = self._hue_ramp_through_red()
        hsv = image.color.hsv[:]
        assert np.ptp(hsv[..., 1]) < 0.02, "saturation must be constant"
        assert np.ptp(hsv[..., 2]) < 0.02, "value must be constant"

    def test_hsv_manufactures_an_edge_at_the_seam(self):
        seam, background = self._seam_and_background("hsv")
        assert seam / background > 20.0, f"hsv seam {seam:.4f} vs background {background:.4f}"

    def test_lab_sees_no_seam(self):
        seam, background = self._seam_and_background("lab")
        assert seam / background < 3.0, f"lab seam {seam:.4f} vs background {background:.4f}"

    def test_hsv_responds_an_order_of_magnitude_more_than_lab(self):
        hsv_seam, _ = self._seam_and_background("hsv")
        lab_seam, _ = self._seam_and_background("lab")
        assert hsv_seam / lab_seam > 10.0


class TestRotationEquivariance:
    """The monogenic filters are isotropic, so a 90-degree rotation should commute.

    **It does not commute exactly, and the reason is structural.** At even sizes the
    frequency grid is ``arange(-N/2, N/2)/N`` -- it contains ``-N/2`` and omits ``+N/2``, so
    it is not symmetric under a quarter turn, and a rotation maps a sampled frequency onto
    one that is not sampled. Measured on a 256x256 crop: ``max|d| = 1.0e-03`` at ``k=1`` and
    ``1.4e-03`` at ``k=2``.

    The plan's claimed ``rtol=1e-8`` ("the FFT's own reproducibility") was a guess and is
    wrong by five orders. The tolerance below is the measurement, with a 3x margin.
    """

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_a_quarter_turn_commutes_to_1e_minus_3(self, k):
        import phenotypic

        square = load_synth_yeast_plate().rgb[:][100:356, 100:356]

        def congruency(array):
            return FocusEdgeColorPhase()._color_phase_congruency(
                phenotypic.Image(array.copy())
            ).pc

        straight = congruency(square)
        turned = np.rot90(congruency(np.rot90(square, k, axes=(0, 1))), -k, axes=(0, 1))
        assert np.abs(straight - turned).max() < 3e-3


class TestOperationContract:
    def test_it_is_registered_as_an_enhancer(self):
        import phenotypic.enhance as enhance

        assert "FocusEdgeColorPhase" in enhance.__all__

    def test_rgb_and_gray_are_not_mutated(self):
        image = load_synth_yeast_plate()
        rgb_before, gray_before = image.rgb[:].copy(), image.gray[:].copy()
        FocusEdgeColorPhase().apply(image)
        np.testing.assert_array_equal(image.rgb[:], rgb_before)
        np.testing.assert_array_equal(image.gray[:], gray_before)

    def test_json_round_trip(self):
        op = FocusEdgeColorPhase(
            color_space="hsv", fusion="l2", chroma_weight_1=2.5, chroma_weight_2=0.25,
            n_scale=5, k=7.0,
        )
        restored = ImagePipeline(pipe_cfgs=[op]).from_json(ImagePipeline(pipe_cfgs=[op]).to_json())
        rebuilt = list(restored._ops.values())[0]
        assert rebuilt.model_dump() == op.model_dump()

    def test_operations_use_apply_not_call(self):
        with pytest.raises(TypeError):
            FocusEdgeColorPhase()(load_synth_yeast_plate())  # type: ignore[operator]
