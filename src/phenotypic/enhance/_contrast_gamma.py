from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.exposure import adjust_gamma

from ..abc_ import ContrastAdjustment
from ..sdk_.mixin import InputLayerMixin, NormalizedOutputMixin
from ..sdk_.typing_ import TuneSpec


class ContrastGamma(InputLayerMixin, NormalizedOutputMixin, ContrastAdjustment):
    """Apply a power-law (gamma) intensity curve to boost faint or washed-out colonies.

    Raises each pixel to the power ``gamma`` after normalizing to [0, 1], then scales
    by ``gain``. Values above 1 darken midtones and deepen the agar background, making
    bright colonies stand out. Values below 1 brighten midtones, rescuing faint or
    translucent colonies that a global threshold would otherwise miss.

    Unlike :class:`ContrastStretching`, the mapping is non-linear, so it redistributes
    tonal weight rather than merely rescaling the range.

    Best For:
        - Faint or translucent colonies lost against a bright agar background
          (``gamma`` below 1).
        - Over-exposed plates where colony interiors saturate (``gamma`` above 1).
        - Pigmented colonies whose colour separation is stronger in a single channel:
          set ``input_layer="rgb"`` so the curve applies per-channel before the
          detection matrix is derived.

    Consider Also:
        - :class:`ContrastStretching` when the histogram is merely narrow and a
          linear remap suffices.
        - :class:`ContrastSigmoid` when you want to steepen contrast around a
          specific intensity rather than across the whole range.

    Args:
        gamma: Power-law exponent. Below 1 brightens midtones; above 1 darkens them.
            ``1.0`` is the identity. Typical range: 0.5--2.5. Default: 1.0.
        gain: Constant multiplier applied after the curve. Default: 1.0.
            Has no effect when ``norm="rescale"``, which divides it back out.
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
        Image: Input image with ``detect_mat`` gamma-corrected. ``rgb`` and ``gray``
        are unchanged. With ``input_layer="rgb"``, any enhancement a prior operation
        wrote to ``detect_mat`` is discarded, as with :class:`SetDetectMode`.

    Examples:
        Darken the background to sharpen bright yeast colonies:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import ContrastGamma
        >>> plate = load_synth_yeast_plate()
        >>> enhanced = ContrastGamma(gamma=2.0).apply(plate)
        >>> float(enhanced.detect_mat[:].max()) <= 1.0
        True

        Apply the curve in colour space before deriving the detection matrix.
        Use a channel-*mixing* ``detect_mode`` such as ``'LabA'`` — under a
        selection mode (``'MinRGB'``, ``'red'``, ...) the curve commutes with the
        projection and ``input_layer='rgb'`` changes nothing:

        >>> import numpy as np
        >>> plate = load_synth_yeast_plate()
        >>> plate.set_detect_mode('LabA')
        >>> via_rgb = ContrastGamma(gamma=2.0, input_layer='rgb').apply(plate)
        >>> via_dm = ContrastGamma(gamma=2.0, input_layer='detect_mat').apply(plate)
        >>> bool(np.abs(via_rgb.detect_mat[:] - via_dm.detect_mat[:]).max() > 1e-3)
        True
    """

    gamma: Annotated[float, TuneSpec(0.1, 5.0, log=True)] = 1.0
    gain: Annotated[float, TuneSpec(0.5, 2.0)] = 1.0

    def _operate(self, image: Image) -> Image:
        src = self._guard_input_range(self._read_input_layer(image))
        adjusted = adjust_gamma(src, gamma=self.gamma, gain=self.gain)
        collapsed = self._project_to_detect_mat(image, adjusted)
        image.detect_mat[:] = self._apply_norm(collapsed).astype(np.float32)
        return image
