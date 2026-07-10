"""Colour phase congruency: per-channel monogenic PC, then a cross-channel fusion.

Reuses :mod:`_monogenic_kernels` verbatim and adds no new signal theory. The ``l2`` fusion
is the CMPCM paper's rule; ``joint`` and ``coherent`` are ours.

References:
    Shi, Y., Zhang, X., Liu, X., et al. "Color edge detection based on the fusion of
    monogenic phase congruency and color morphology." *Multimed. Tools Appl.* 78,
    10701--10716 (2019).

    Kovesi, P. "Image features from phase congruency." *Videre* 1(3), 1--26 (1999).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import numpy as np
from pydantic import Field, model_validator

from ._color_phase_kernels import ColorPhaseResult, color_phase_congruency
from ._monogenic_kernels import monogenic_channel_response
from ..abc_ import FocusEdge
from ..sdk_.typing_ import (
    ColorPhaseOutput,
    ColorSpaceName,
    PhaseFusion,
    PhaseLift,
    TuneSpec,
)

if TYPE_CHECKING:
    from phenotypic._core._image import Image

#: Accessor channel indices in **luminance-first** order (spec §4.2). ``Lab`` is natively
#: ``(L*, a*, b*)``; ``hsv`` is natively ``(H, S, V)`` and must be permuted so index ``0``
#: is always the axis whose weight is pinned at ``1.0``. Read literally, an earlier spec
#: (*"in color_space channel order"*) pinned ``H`` under ``hsv`` -- the one channel that is
#: circular, ill-conditioned at low saturation, and carries no luminance at all.
_LUMINANCE_FIRST: dict[str, tuple[int, int, int]] = {
    "lab": (0, 1, 2),
    "hsv": (2, 0, 1),
}


class FocusEdgeColorPhase(FocusEdge):
    """Enhance colony edges using colour phase congruency across three channels.

    Runs :class:`FocusEdgeMonogenicPhase`'s monogenic chain independently on each of three
    colour channels, then fuses the results. Phase congruency is already invariant to
    illumination level; fusing across colour additionally lets a channel with amplitude but
    no phase agreement -- pigment speckle, agar grain, Bayer demosaic noise -- **veto** an
    edge the luminance channel would otherwise assert.

    Best For:
        - Pigmented colonies whose rims are stronger in chroma than in lightness
        - Plates where agar texture produces spurious luminance edges that carry no
          matching chromatic structure
        - Any case where :class:`FocusEdgeMonogenicPhase` on luminance alone over-responds
          to grain

    Consider Also:
        - :class:`FocusEdgeMonogenicPhase` when the plate is near-achromatic, which this
          operation rejects outright.
        - :class:`FocusEdgePhase` when you also want corner strength from the moment tensor.

    Args:
        color_space: ``"lab"`` (default) or ``"hsv"``. Channels are taken in
            **luminance-first** order: ``lab`` gives ``(L*, a*, b*)``, ``hsv`` gives
            ``(V, H, S)``. Raw CIELAB is already the perceptual common scale -- CIE76's
            ``dE*ab`` is the Euclidean norm over raw ``L*a*b*`` -- so no per-axis rescaling
            is applied; dividing by nominal axis ranges would corrupt an already-normalized
            space and bias against chroma by ``128/100``.
        fusion: ``"joint"`` (default) shares one denominator across channels, so incoherent
            chroma amplitude vetoes a spurious luminance edge. ``"l2"`` is CMPCM's rule --
            three independent congruencies combined by root-sum-of-squares -- and has **no**
            cross-channel interaction whatsoever. ``"coherent"`` sums the monogenic vectors
            before taking their norm; it cancels opposite-phase responses, **including a
            genuine anti-correlated chromatic edge** where lightness falls as yellowness
            rises. Opt-in, never default.
        chroma_weight_1: Weight on the first chromatic axis (``a*`` under ``lab``, ``H``
            under ``hsv``). Luminance is pinned at ``1.0``, so there are two degrees of
            freedom, not three. At ``0.0`` the axis is switched off entirely and, with
            ``chroma_weight_2`` also ``0.0``, the operation reduces **bit-for-bit** to
            :class:`FocusEdgeMonogenicPhase` on the luminance channel.
        chroma_weight_2: Weight on the second chromatic axis (``b*`` / ``S``). The search
            bound of ``8.0`` brackets "chroma off" through "chroma dominates" for the axis
            that carries signal on real plates -- ``b*`` reaches parity with ``L*`` at
            ``2.6`` (Rhodotorula) and ``4.4`` (Neurospora) -- and deliberately refuses to
            let ``a*`` reach parity, which needs ``19`` to ``61``.
        lift: ``"monogenic"`` (default). ``"conformal"`` raises
            :exc:`NotImplementedError` **at construction** -- the field exists so the
            surface is stable, but the path is gated on an experiment it may well fail.
        n_scale: Number of log-Gabor scales. Must be at least 2; the frequency-spread weight
            divides by ``n_scale - 1``.
        min_wavelength: Wavelength of the finest scale, in pixels.
        mult: Wavelength multiplier between successive scales.
        sigma_onf: Ratio of each filter's Gaussian sigma to its centre frequency. Strictly
            below ``1.0``: at exactly ``1.0`` the log-Gabor's Gaussian width is
            ``log(1.0) = 0`` and the filter bank divides by zero.
        k: Noise standard deviations above the mean at which the threshold sits.
            ``phasecongmono``'s default is ``3.0``, not :class:`FocusEdgePhase`'s ``2.0``.
        deviation_gain: Scales the phase-deviation term. Kovesi: "sensible values are from
            1 to about 2."
        cutoff: Fractional frequency-spread below which the response is penalized.
        g: Sharpness of the frequency-spread sigmoid.
        noise_method: ``-1`` estimates the Rayleigh parameter from the median of the finest
            scale's amplitude; ``-2`` uses its histogram mode; any value ``>= 0`` is the
            threshold verbatim, so ``0.0`` disables it.
        output: Only ``"pc"``, the fused congruency in ``[0, 1]``. The fused ``orientation``
            and ``feature_type`` are computed and returned by
            :meth:`_color_phase_congruency`, but **not exposed**: of the three fusion modes
            only ``"coherent"`` builds a fused monogenic vector, so under the default they
            would describe a quantity the response never touched.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the fused congruency map, clipped
        to ``[0, 1]``. ``rgb`` and ``gray`` are unchanged.

    Raises:
        NotImplementedError: If ``lift="conformal"``, at construction time.
        ValueError: If the image is achromatic -- all three RGB channels identical -- since
            ``a*`` and ``b*`` are then identically zero and ``joint`` degenerates to a
            luminance congruency divided by itself. Raised inside ``_operate``, so
            :meth:`ImageOperation.apply` wraps it; walk the ``__cause__`` chain to catch it.
        ValidationError: On any out-of-range field.

    Note:
        **This operation reads ``image.rgb``, not ``detect_mat``.** It is a pipeline
        *source*, like :class:`SetDetectMode`: **any enhancer placed before it in an**
        :class:`ImagePipeline` **has no effect on its output.** Colour phase congruency is
        defined on colour, and ``rgb`` is not a supported ``detect_mat`` layer. This is
        legal under ``@validate_operation_integrity``, which forbids *mutating* ``rgb`` and
        ``gray`` and says nothing about reading them.

    Warning:
        ``color_space="hsv"`` band-passes **raw hue across its wrap discontinuity**. Hue is
        circular on ``[0, 1)``, so a log-Gabor filter sees a unit step at the ``0.99 ->
        0.01`` seam where the colour is in fact continuous, and a near-red boundary
        manufactures a phantom edge. We do not unwrap, because the CMPCM paper does not and
        ``("hsv", "l2")`` is the configuration its ranking regression reproduces. Prefer the
        ``"lab"`` default, which has no seam.

    Examples:
        Fuse three channels of a synthetic yeast plate. The output is a congruency map in
        ``[0, 1]``, like every other :class:`FocusEdge`:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import FocusEdgeColorPhase
        >>> enhanced = FocusEdgeColorPhase().apply(load_synth_yeast_plate())
        >>> bool(0.0 <= enhanced.detect_mat[:].min() <= enhanced.detect_mat[:].max() <= 1.0)
        True

        Switch chroma off and recover the luminance-only monogenic port exactly:

        >>> luminance_only = FocusEdgeColorPhase(chroma_weight_1=0.0, chroma_weight_2=0.0)
        >>> bool(luminance_only.apply(load_synth_yeast_plate()).detect_mat[:].max() > 0.5)
        True

    See Also:
        :class:`FocusEdgeMonogenicPhase`, which this reduces to when both chroma weights are
        ``0.0``.
    """

    color_space: ColorSpaceName = "lab"
    fusion: PhaseFusion = "joint"
    chroma_weight_1: Annotated[float, TuneSpec(0.0, 8.0)] = Field(1.0, ge=0.0)
    chroma_weight_2: Annotated[float, TuneSpec(0.0, 8.0)] = Field(1.0, ge=0.0)
    lift: PhaseLift = "monogenic"

    # Ported verbatim from FocusEdgeMonogenicPhase. Duplicated rather than inherited or
    # mixed in: a shared BaseModel mixin would put two pydantic bases in the MRO of a class
    # that the operation registry and `from_json` both walk, and subclassing the monogenic
    # operation would make `isinstance(colour_op, FocusEdgeMonogenicPhase)` true, which it
    # is not. `TestFieldParityWithTheMonogenicPort` pins them against drift -- a checkable
    # oracle, where a mixin would only have been an assertion.
    n_scale: Annotated[int, TuneSpec(3, 6)] = Field(4, ge=2)
    min_wavelength: Annotated[float, TuneSpec(2.0, 10.0)] = Field(3.0, ge=2.0)
    mult: Annotated[float, TuneSpec(1.5, 3.0)] = Field(2.1, gt=1.0)
    sigma_onf: Annotated[float, TuneSpec(0.1, 0.99)] = Field(0.55, ge=0.1, lt=1.0)
    k: Annotated[float, TuneSpec(0.5, 20.0)] = Field(3.0, ge=0.0)
    deviation_gain: Annotated[float, TuneSpec(1.0, 2.0)] = Field(1.5, gt=0.0)
    cutoff: Annotated[float, TuneSpec(0.3, 0.7)] = Field(0.5, gt=0.0, lt=1.0)
    g: Annotated[float, TuneSpec(2.0, 20.0)] = Field(10.0, gt=0.0)
    noise_method: Annotated[float, TuneSpec(tunable=False)] = -1.0

    output: ColorPhaseOutput = "pc"

    @model_validator(mode="after")
    def _reject_the_gated_conformal_lift(self) -> "FocusEdgeColorPhase":
        """``lift="conformal"`` raises at construction, not at apply.

        pydantic v2 traps ``ValueError`` and ``AssertionError`` and re-raises them as a
        ``ValidationError``; it lets everything else propagate. ``NotImplementedError``
        therefore reaches the caller with its type intact -- unlike a ``ValueError`` raised
        inside ``_operate``, which :meth:`ImageOperation.apply` wraps **twice**, into a
        ``RuntimeError`` (``abc_/_image_operation.py:423``) and then a bare ``Exception``
        (``:470``). Verified, not assumed.
        """
        if self.lift == "conformal":
            raise NotImplementedError(
                    "lift='conformal' is gated on conformal-lift.md §4's three-arm junction "
                    "experiment and is not implemented. The field exists so the operation's "
                    "surface is stable. Use lift='monogenic'."
            )
        return self

    def _extract_channels(self, image: Image) -> list[np.ndarray]:
        """Three scalar channels from ``rgb``, in luminance-first order.

        Raises:
            ValueError: If the image is not 3-channel RGB, or is achromatic. Drift ``C11``.
        """
        rgb = image.rgb[:]
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                    f"FocusEdgeColorPhase needs a 3-channel RGB image; got shape {rgb.shape}."
            )
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
            raise ValueError(
                    "FocusEdgeColorPhase requires a chromatic image: all three RGB channels "
                    "are identical, so a* and b* are identically zero and fusion='joint' "
                    "degenerates to a luminance congruency divided by itself. Use "
                    "FocusEdgeMonogenicPhase on a greyscale image."
            )

        stack = image.color.Lab[:] if self.color_space == "lab" else image.color.hsv[:]
        return [
            np.asarray(stack[..., index], dtype=np.float64)
            for index in _LUMINANCE_FIRST[self.color_space]
        ]

    def _color_phase_congruency(self, image: Image) -> ColorPhaseResult:
        """Fuse three channels' monogenic accumulators. **``pc`` is un-clipped.**

        Protected, and returns more than :meth:`_operate` exposes: ``orientation`` and
        ``feature_type`` ride along on the result so a future consumer can reach them
        without a breaking change -- mirroring :meth:`FocusEdgePhase._phasecong3`, whose
        result carries both angles while ``output`` exposes only ``M``/``m``/``pc_sum``.
        Drift ``C15``.

        The PFOM ranking regression consumes this rather than :meth:`apply`, because
        ``fusion="l2"`` ranges over ``[0, ||w||]`` and clipping would truncate the paper's
        actual quantity (drift ``C3``).
        """
        channels = [
            monogenic_channel_response(
                    channel,
                    n_scale=self.n_scale,
                    min_wavelength=self.min_wavelength,
                    mult=self.mult,
                    sigma_onf=self.sigma_onf,
                    k=self.k,
                    noise_method=self.noise_method,
            )
            for channel in self._extract_channels(image)
        ]
        weights = np.array([1.0, self.chroma_weight_1, self.chroma_weight_2])
        return color_phase_congruency(
                channels,
                weights,
                fusion=self.fusion,
                n_scale=self.n_scale,
                cutoff=self.cutoff,
                g=self.g,
                deviation_gain=self.deviation_gain,
        )

    def _operate(self, image: Image) -> Image:
        """Replace the detection matrix with the fused congruency map."""
        result = self._color_phase_congruency(image)
        # The clip is load-bearing for `l2`, whose range is [0, ||w||], and redundant for
        # the other two. Keep it in all three cases: `detect_mat.__setitem__` enforces
        # float32 but does **not** clamp -- verified by writing 2.5 and reading 2.5 back.
        #
        # Note for a future reader: the `adding-an-operation` skill asks output-clamping
        # operations to declare `norm: NormOut` via `NormalizedOutputMixin`. That machinery
        # does not exist on this branch -- no `NormOut`, no `NormalizedOutputMixin`, no
        # `_apply_norm`, and `sdk_/mixin/` still ships `_clip_control_mixin.py`. Every
        # `FocusEdge` sibling clips inline. Follow the code that exists; adopt `norm` when
        # the branch that introduces it lands, alongside the other thirty enhancers.
        image.detect_mat[:] = np.clip(result.pc, 0.0, 1.0)
        return image
