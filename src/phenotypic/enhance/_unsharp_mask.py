from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.filters import unsharp_mask

from ..abc_ import ImageEnhancer


class UnsharpMask(ImageEnhancer):
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
        - :class:`BilateralDenoise` for denoising before sharpening on
          grainy images to avoid amplifying noise.
        - :class:`LaplaceEnhancer` for second-derivative edge detection
          that replaces rather than enhances the intensity profile.
        - :class:`PhaseCongruencyEnhancer` for contrast-invariant edge
          detection under uneven illumination.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of edge sharpening on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        unsharp masking and sharpening strategies.
    """

    def __init__(
            self,
            radius: float = 2.0,
            amount: float = 1.0,
            preserve_range: bool = False,
            n_iter: int = 1,
    ):
        """
        Parameters:
            radius (float): Standard deviation (sigma) of the Gaussian blur in pixels.
                Defines the scale of features to enhance. Small values (0.5–2) sharpen
                fine details (thin colony edges, small morphologies); larger values
                (5–15) enhance broad features (large colonies, colony-background
                separation). Must be > 0. For fungal colonies, keep below the typical
                colony width to avoid merging adjacent colonies. Recommended: 2.0–3.0
                for general-purpose use, 1.0 for high-density plates, 5.0+ for
                emphasizing large-scale features on low-resolution images.
            amount (float): Amplification factor for the sharpening effect. Controls
                how much the edge enhancement contributes to the output. Typical range:
                0.3–2.5. Low values (0.3–0.7) produce subtle enhancement suitable for
                noisy images; standard values (1.0–1.5) give balanced sharpening;
                high values (2.0+) create aggressive enhancement for very low-contrast
                colonies. Can be negative to produce blurring instead. Excessive amounts
                risk visible artifacts and noise amplification.
            preserve_range (bool): If False (default), output may be rescaled if
                necessary. If True, the original range of input values is preserved.
                Keep as False for consistency with other enhancers.
            n_iter (int): Number of successive unsharp mask passes to apply. Must be >= 1.
                One pass (default) applies the filter once. Multiple passes (2+) compound
                the sharpening effect for progressively more aggressive enhancement, but
                at increased risk of noise amplification and halo artifacts.
        """
        if radius <= 0:
            raise ValueError("width must be > 0")
        if n_iter < 1:
            raise ValueError("n_iter must be >= 1")

        self.radius = float(radius)
        self.amount = float(amount)
        self.preserve_range = bool(preserve_range)
        self.n_iter = int(n_iter)

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
