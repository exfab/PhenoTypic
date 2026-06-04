from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from pydantic import Field
from skimage.filters import unsharp_mask

from ..abc_ import ImageEnhancer
from ..tune._search_space._tune_spec import TuneSpec


class SharpenEdgeGauss(ImageEnhancer):
    """Sharpen colony edges in ``detect_mat`` with unsharp masking.

    Subtracts a Gaussian-blurred copy from the original and scales the
    difference to emphasize high-contrast boundaries. Makes soft or
    indistinct colony edges more pronounced, improving thresholding and
    edge-detection accuracy.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        radius: Standard deviation of the Gaussian blur in pixels.
            Controls the scale of features enhanced. Small values
            (0.5--2.0) sharpen fine details; larger values (5--15)
            enhance broader features. Default: 2.0.
        amount: Multiplier for the sharpening effect. Low values
            (0.3--0.7) produce subtle enhancement; standard values
            (1.0--1.5) give moderate sharpening; high values (2.0+)
            create aggressive enhancement with risk of halo artifacts.
            Default: 1.0.
        preserve_range: Preserve the original pixel value range.
            Default: ``False``.
        n_iter: Number of successive sharpening passes. Multiple passes
            compound the effect. Typical range: 1--3. Default: 1.

    Returns:
        Image: Input image with ``detect_mat`` sharpened via unsharp
        masking. ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Low-contrast colonies with soft, gradual edges (translucent
          growth).
        - Dense plates where colonies blend into background.
        - Pre-threshold sharpening to improve segmentation accuracy.
        - Slight scanner or lens blur that softens colony boundaries.

    Consider Also:
        - :class:`LocalEdgeDenoise` for denoising before sharpening on
          grainy images to avoid amplifying noise.
        - :class:`SharpenEdgeLaplace` for second-derivative edge detection
          that replaces rather than enhances the intensity profile.
        - :class:`EnhanceFeatures` for contrast-invariant edge
          detection under uneven illumination.

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
