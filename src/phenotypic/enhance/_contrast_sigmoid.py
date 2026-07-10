from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.exposure import adjust_sigmoid

from ..abc_ import ContrastAdjustment
from ..sdk_.mixin import InputLayerMixin, NormalizedOutputMixin
from ..sdk_.typing_ import TuneSpec


class ContrastSigmoid(InputLayerMixin, NormalizedOutputMixin, ContrastAdjustment):
    """Apply a sigmoid intensity curve that steepens contrast about a chosen cutoff.

    Computes ``1 / (1 + exp(gain * (cutoff - I)))``, an S-shaped curve centred on
    ``cutoff``. Pixels below the cutoff are pushed toward 0 and pixels above it
    toward 1, while the transition stays continuous. Raising ``gain`` steepens the
    S until it approaches a hard threshold.

    Best For:
        - Pushing a soft, gradual colony/agar boundary toward a binary decision
          before a global threshold such as Otsu or Triangle, without committing
          to a hard cut.
        - Suppressing low-amplitude agar texture while leaving colony interiors
          saturated.
        - Plates where the colony/background split sits at a known intensity that
          ``cutoff`` can be set to directly.

    Consider Also:
        - :class:`ContrastGamma` when the whole tonal range should be reweighted
          rather than steepened about one intensity.
        - :class:`ContrastStretching` when the histogram is merely narrow and a
          linear remap suffices.

    Args:
        cutoff: Intensity about which the sigmoid is centred, in [0, 1]. Pixels
            below it are pushed toward 0, above it toward 1. Set near the
            agar/colony boundary intensity. Default: 0.5.
        gain: Steepness of the sigmoid. Larger values approach a hard threshold;
            smaller values blend gradually. Typical range: 5--15. Default: 10.0.
            Unlike :class:`ContrastGamma`, this ``gain`` survives ``norm="rescale"``
            because it reshapes the curve rather than scaling its output.
        inv: When ``True``, invert the sigmoid so bright regions are suppressed.
            Default: ``False``.
        norm: Output range policy. ``"clip"`` (default) saturates values outside
            [0, 1]; ``"rescale"`` remaps the full observed range onto [0, 1];
            ``None`` passes values through untouched.
        input_layer: Source layer. ``"detect_mat"`` (default) applies the curve to
            the 2-D detection matrix. ``"rgb"`` applies it to all three colour
            channels, then collapses the result to 2-D through the image's own
            ``detect_mode``. Because the curve is non-linear, the two routes
            generally differ -- except under a ``detect_mode`` that is a per-pixel
            selection (``"red"``/``"green"``/``"blue"``/``"MinRGB"``/``"HsvV"``),
            which commutes with any monotonically increasing curve and so yields
            an identical result. Default: ``"detect_mat"``.

    Returns:
        Image: Input image with ``detect_mat`` sigmoid-corrected. ``rgb`` and
        ``gray`` are unchanged. With ``input_layer="rgb"``, any enhancement a prior
        operation wrote to ``detect_mat`` is discarded, as with
        :class:`SetDetectMode`.

    Examples:
        Steepen the colony/agar transition about the plate's mean intensity:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import ContrastSigmoid
        >>> plate = load_synth_yeast_plate()
        >>> cutoff = float(plate.detect_mat[:].mean())
        >>> enhanced = ContrastSigmoid(cutoff=cutoff, gain=10.0).apply(plate)
        >>> float(enhanced.detect_mat[:].std()) > float(plate.detect_mat[:].std())
        True
    """

    cutoff: Annotated[float, TuneSpec(0.0, 1.0)] = 0.5
    gain: Annotated[float, TuneSpec(1.0, 20.0)] = 10.0
    inv: bool = False

    def _operate(self, image: Image) -> Image:
        src = self._guard_input_range(self._read_input_layer(image))
        adjusted = adjust_sigmoid(src, cutoff=self.cutoff, gain=self.gain, inv=self.inv)
        collapsed = self._project_to_detect_mat(image, adjusted)
        image.detect_mat[:] = self._apply_norm(collapsed).astype(np.float32)
        return image
