"""``FocusEdgeColorPhase``: construction, channel order, guards, and field parity.

Numeric fidelity, the ``[0,1]`` bound, scale invariance, the hue-wrap artefact and the
PFOM ranking regression live in the sibling files added alongside this one.
"""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest
from pydantic import ValidationError

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
            ("output", "orientation"),
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
        "deviation_gain", "cutoff", "g", "noise_method",
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

    def test_only_pc_is_exposed(self):
        """Drift ``C15``. Only ``coherent`` builds a fused monogenic vector."""
        assert get_args(ColorPhaseOutput) == ("pc",)

    def test_the_fusion_literal_matches_the_kernel_dispatch(self):
        from phenotypic.enhance._color_phase_kernels import FUSIONS

        assert set(get_args(PhaseFusion)) == set(FUSIONS)


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
