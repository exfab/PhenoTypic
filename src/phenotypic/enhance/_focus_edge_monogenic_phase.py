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
from ..sdk_.mixin import NormalizedOutputMixin
from ..sdk_.typing_ import MonogenicOutput, TuneSpec

if TYPE_CHECKING:
    from phenotypic._core._image import Image


class FocusEdgeMonogenicPhase(NormalizedOutputMixin, FocusEdge):
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
        n_scale: Number of log-Gabor scales. Must be at least 2 -- the frequency-spread
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

            **The two angle maps are diagnostic, not detectable.** An angle is defined
            everywhere, including where there is no feature, so the output is a noise field
            wherever ``pc`` is small. On ``load_synth_yeast_plate`` 89.6% of pixels have
            ``pc < 0.02``; over those, ``"orientation"`` spans the full ``[0, 1]`` with
            ``std = 0.307`` and only 3.3% lie near the ``0.5`` that means "vertical edge".
            Kovesi consumes his ``or`` masked by ``pc`` (his comment: *"Quantize to 0 - 180
            degrees (for NONMAXSUP)"*). Feed ``"pc"`` to a detector; read the angles for
            inspection, or mask them yourself.

            ``"orientation"``'s true image is ``(0, 1]``, not ``[0, 1]``: the fold is
            half-open, so ``-pi/2`` is unattainable. ``"feature_type"`` attains both ends.
        norm: Output-range policy for ``output="pc"``. ``"clip"`` (default) saturates to
            [0, 1], ``"rescale"`` remaps the observed PC range to [0, 1], and ``None``
            preserves it. It does not affect the two angle maps, whose fixed [0, 1]
            encoding would otherwise lose its physical meaning.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the selected monogenic map. PC
        output follows ``norm``; angle outputs retain their normalized [0, 1] encoding.
        ``rgb`` and ``gray`` are unchanged.

    Raises:
        ValidationError: If ``n_scale`` < 2, ``min_wavelength`` < 2, ``mult`` <= 1,
            ``sigma_onf`` outside ``[0.1, 1.0]``, ``k`` < 0, ``deviation_gain`` <= 0,
            ``cutoff`` outside ``(0, 1)``, ``g`` <= 0, or ``output`` is not one of
            ``"pc"``, ``"orientation"``, ``"feature_type"``.

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
    # Strictly below 1.0: log_gabor_scale's Gaussian width is log(sigma_onf), so
    # sigma_onf == 1.0 divides by zero and returns an all-NaN detect_mat -- which is worse
    # than an all-zero one, because NaN silently passes a `0 <= x <= 1` range check. The
    # kernel raises (drift M10); this bound moves the failure to construction time.
    # TuneSpec stops at 0.99 because FloatRange appends `high` exactly
    # (tune/_search_space/_domains.py:86), so a grid run would otherwise evaluate 1.0.
    sigma_onf: Annotated[float, TuneSpec(0.1, 0.99)] = Field(0.55, ge=0.1, lt=1.0)
    # Lower search bound 0.5 (not 0.0). k=0 does NOT disable noise thresholding -- it drops
    # the standard-deviation term, leaving T = total_tau*sqrt(pi/2), the noise MEAN.
    # `noise_method=0.0` is what sets T = 0.0 exactly. k=0 remains a degenerate anchor, but
    # not for the reason this comment used to give.
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
            selected = self._apply_norm(result.pc)
        elif self.output == "orientation":
            selected = np.clip((result.orientation + np.pi / 2) / np.pi, 0.0, 1.0)
        else:
            selected = np.clip((result.feature_type + np.pi / 2) / np.pi, 0.0, 1.0)

        # detect_mat enforces float32 on assignment, so no explicit cast is needed.
        image.detect_mat[:] = selected
        return image
