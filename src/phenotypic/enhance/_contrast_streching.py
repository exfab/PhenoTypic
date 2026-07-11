from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.exposure import rescale_intensity

from ..abc_ import ContrastAdjustment
from ..sdk_.mixin import InputLayerMixin
from ..sdk_.typing_ import TuneSpec


class ContrastStretching(InputLayerMixin, ContrastAdjustment):
    """Stretch the intensity range of ``detect_mat`` to fill the full dynamic range.

    Rescales pixel values by clipping at lower and upper percentiles, then
    linearly remapping the retained range to [0, 1]. Outliers such as specular
    highlights and deep shadows are clamped, expanding the range where colony
    intensities reside. Simpler and faster than :class:`EnhanceLocalContrast`,
    with no local tile artefacts.

    For how contrast adjustment fits into the pipeline, see
    :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Plates with narrow intensity histograms from under-exposure or low
          scanner gain.
        - Normalizing exposure variation across imaging sessions or plate batches.
        - Quick preprocessing before global thresholding (Otsu, Triangle).
        - Images with bright specular highlights or very dark border regions
          that compress the useful intensity range.

    Consider Also:
        - :class:`EnhanceLocalContrast` when illumination varies spatially
          across the plate and per-tile equalization is needed.
        - :class:`FlattenIllumination` when the primary issue is a large-scale
          brightness gradient rather than a narrow dynamic range.

    Args:
        lower_percentile: Dark clipping point. Pixels below this percentile
            are mapped to 0. Typical range: 1--5. Default: 2.
        upper_percentile: Bright clipping point. Pixels above this percentile
            are mapped to 1. Typical range: 95--99. Default: 98.
        keep_colors: When ``input_layer="rgb"``, take a single pair of percentiles
            jointly across all three colour channels and rescale them together,
            preserving channel balance and hue. When ``False``, compute percentiles
            per channel and rescale each independently -- effectively a white
            balance, which removes colour casts but shifts hue. Ignored for 2-D
            input. Default: ``True``.
        input_layer: Source layer. ``"detect_mat"`` (default) stretches the 2-D
            detection matrix. ``"rgb"`` stretches all three colour channels, then
            collapses the result to 2-D through the image's own ``detect_mode``.

    Returns:
        Image: Input image with ``detect_mat`` rescaled to the full dynamic
        range. ``rgb`` and ``gray`` are unchanged. The output always fills
        [0, 1] by construction, so no ``norm`` field is offered. With
        ``input_layer="rgb"``, any enhancement a prior operation wrote to
        ``detect_mat`` is discarded, as with :class:`SetDetectMode`.

    Examples:
        Stretch the detection matrix across the full dynamic range:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import ContrastStretching
        >>> plate = load_synth_yeast_plate()
        >>> enhanced = ContrastStretching().apply(plate)
        >>> float(enhanced.detect_mat[:].min()), float(enhanced.detect_mat[:].max())
        (0.0, 1.0)

        Stretch in colour space, holding channel balance fixed, before the
        detection matrix is derived:

        >>> plate = load_synth_yeast_plate()
        >>> plate.set_detect_mode('MinRGB')
        >>> enhanced = ContrastStretching(input_layer='rgb').apply(plate)
        >>> enhanced.detect_mat[:].ndim
        2

    See Also:
        :doc:`/how_to/notebooks/enhance_low_contrast` for a comparison of
        contrast enhancement methods on real plate images.
        :doc:`/explanation/what_enhancement_does` for how enhancement fits
        into the pipeline model.
    """

    lower_percentile: Annotated[int, TuneSpec(1, 5)] = 2
    upper_percentile: Annotated[int, TuneSpec(95, 99)] = 98
    keep_colors: bool = True

    def _operate(self, image: Image) -> Image:
        src = self._read_input_layer(image)
        if src.ndim == 3 and not self.keep_colors:
            adjusted = np.empty_like(src)
            for channel in range(src.shape[2]):
                p_lower, p_upper = np.percentile(
                        src[..., channel],
                        (self.lower_percentile, self.upper_percentile),
                )
                adjusted[..., channel] = rescale_intensity(
                        image=src[..., channel],
                        in_range=(p_lower, p_upper),
                        out_range=(0, 1),
                )
        else:
            p_lower, p_upper = np.percentile(
                    src, (self.lower_percentile, self.upper_percentile)
            )
            adjusted = rescale_intensity(
                    image=src, in_range=(p_lower, p_upper), out_range=(0, 1)
            )
        collapsed = self._project_to_detect_mat(image, adjusted)
        image.detect_mat[:] = collapsed.astype(np.float32)
        return image
