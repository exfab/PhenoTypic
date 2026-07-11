from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.exposure import adjust_log

from ..abc_ import ContrastAdjustment
from ..sdk_.mixin import InputLayerMixin, NormalizedOutputMixin
from ..sdk_.typing_ import TuneSpec


class ContrastLog(InputLayerMixin, NormalizedOutputMixin, ContrastAdjustment):
    """Apply a logarithmic intensity curve to lift faint colonies out of dark agar.

    Computes ``gain * log2(1 + I)``, which expands the dark end of the histogram
    while compressing highlights. Faint colonies sitting just above the agar
    background gain contrast; already-bright colonies compress toward saturation.
    Setting ``inv=True`` applies the inverse exponential curve, which expands the
    bright end instead.

    Best For:
        - Dark-field or transmitted-light plates where colonies are dim and the
          background is near black.
        - Recovering small colonies whose intensity sits within a few percent of
          the agar background.

    Consider Also:
        - :class:`ContrastGamma` for a tunable power-law curve rather than a fixed
          logarithmic shape.
        - :class:`ContrastSigmoid` when contrast should steepen around one
          intensity rather than across the shadows.

    Args:
        gain: Constant multiplier applied after the curve. Default: 1.0.
            Has no effect when ``norm="rescale"``, which divides it back out.
        inv: When ``True``, apply the inverse (exponential) curve, expanding the
            bright end rather than the dark end. Default: ``False``.
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
        Image: Input image with ``detect_mat`` log-corrected. ``rgb`` and ``gray``
        are unchanged. With ``input_layer="rgb"``, any enhancement a prior operation
        wrote to ``detect_mat`` is discarded, as with :class:`SetDetectMode`.

    Examples:
        Lift dim colonies on a dark plate:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import ContrastLog
        >>> plate = load_synth_yeast_plate()
        >>> enhanced = ContrastLog().apply(plate)
        >>> float(enhanced.detect_mat[:].mean()) > float(plate.detect_mat[:].mean())
        True
    """

    gain: Annotated[float, TuneSpec(0.5, 2.0)] = 1.0
    inv: bool = False

    def _operate(self, image: Image) -> Image:
        src = self._guard_input_range(self._read_input_layer(image))
        adjusted = adjust_log(src, gain=self.gain, inv=self.inv)
        collapsed = self._project_to_detect_mat(image, adjusted)
        image.detect_mat[:] = self._apply_norm(collapsed).astype(np.float32)
        return image
