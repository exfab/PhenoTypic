from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from pydantic import Field
from skimage.filters import unsharp_mask

from ..abc_ import ContrastAdjustment
from ..sdk_.typing_ import TuneSpec


class SharpenEdgeGauss(ContrastAdjustment):
    """Sharpen colony edges in ``detect_mat`` with unsharp masking.

    Subtracts a Gaussian-blurred copy from the original and scales the
    difference to emphasize high-contrast boundaries. Makes soft or
    indistinct colony edges more pronounced, improving thresholding and
    edge-detection accuracy downstream.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Colonies with soft, gradual edges caused by translucent growth
          or slight scanner blur.
        - Dense plates where colony boundaries blend gradually into the
          agar background.
        - Pre-threshold sharpening to improve segmentation accuracy on
          mildly blurred images.
        - Plates with slight lens or scanner defocus that softens colony
          boundaries.

    Consider Also:
        - :class:`LocalEdgeDenoise` for denoising before sharpening on
          grainy images, to avoid amplifying noise alongside edges.
        - :class:`FocusEdgeLaplace` for second-derivative edge detection
          that replaces rather than enhances the intensity profile.
        - :class:`FocusEdgePhase` for contrast-invariant edge detection
          under uneven illumination.

    Args:
        radius: Standard deviation of the Gaussian blur kernel in pixels.
            Controls the spatial scale of features enhanced. Typical
            range: 0.5--5.0 for fine colony edges; up to 15 for broader
            low-frequency enhancement. Default: 2.0.
        amount: Strength multiplier for the sharpening effect. Values
            below 1.0 produce subtle enhancement; 1.0--1.5 gives
            moderate sharpening; values above 2.0 risk halo artifacts
            along high-contrast edges. Default: 1.0.
        preserve_range: Preserve the input pixel value range during
            filtering. Set ``True`` when the downstream operation
            requires values within the original bounds. Default:
            ``False``.
        n_iter: Number of successive sharpening passes. Each pass
            compounds the effect. Typical range: 1--3. Default: 1.

    Returns:
        Image: Input image with ``detect_mat`` sharpened via unsharp
        masking. ``rgb`` and ``gray`` are unchanged.

    Raises:
        ValueError: If ``radius`` is not positive.
        ValueError: If ``n_iter`` is less than 1.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of edge sharpening on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        unsharp masking and sharpening strategies.
    """

    radius: Annotated[float, TuneSpec(0.5, 15.0, log=True)] = Field(2.0, gt=0.0)
    amount: Annotated[float, TuneSpec(0.3, 2.0)] = 1.0
    preserve_range: bool = False
    n_iter: Annotated[int, TuneSpec(1, 3)] = Field(1, ge=1)

    def _operate(self, image: Image) -> Image:
        """Apply unsharp masking to enhance colony edges in the detection matrix channel."""
        for _ in range(self.n_iter):
            image.detect_mat[:] = unsharp_mask(
                    image=image.detect_mat[:],
                    radius=self.radius,
                    amount=self.amount,
                    preserve_range=self.preserve_range,
                    channel_axis=None,
            )
        return image
