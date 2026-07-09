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

    Returns:
        Image: Input image with ``detect_mat`` replaced by the selected monogenic map,
        clipped to ``[0, 1]``. ``rgb`` and ``gray`` are unchanged.

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
