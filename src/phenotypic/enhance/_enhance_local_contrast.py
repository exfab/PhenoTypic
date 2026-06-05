from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Optional

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.exposure import equalize_adapthist

from phenotypic.abc_ import ContrastAdjustment
from phenotypic.tools_.typing_ import TuneSpec


class EnhanceLocalContrast(ContrastAdjustment):
    """Boost local contrast in detect_mat using adaptive histogram equalization.

    Divides detect_mat into tiles and equalizes the histogram within each tile,
    with a clip limit that prevents excessive noise amplification. Faint colonies
    become more visible and easier to threshold, even when illumination varies
    across the plate.

    For a discussion of contrast enhancement strategies, see
    :doc:`/explanation/what_enhancement_does`.

    Args:
        kernel_size: Tile size for local equalization. Smaller tiles reveal
            tiny colonies but amplify agar texture; larger tiles produce
            smoother results. ``None`` auto-selects based on image size
            (typically min(height, width) / 15). Default: ``None``.
        clip_limit: Maximum local contrast amplification. Typical range:
            0.005--0.05. Lower values suppress noise; higher values make
            faint colonies stand out more. Default: 0.01.

    Returns:
        Image: Input image with ``detect_mat`` contrast-enhanced.
        ``rgb`` and ``gray`` are unchanged.

    Raises:
        ValueError: If the detect_mat value range is invalid.

    Best For:
        - Plates with faint or translucent colonies that blend into agar.
        - Uneven illumination (vignetting, shadows from plate lids).
        - Pre-conditioning before global thresholding (Otsu, Triangle).
        - Early time-point plates where colonies are barely visible.

    Consider Also:
        - :class:`ContrastStretching` for a simpler global contrast adjustment
          when illumination is already uniform.
        - :class:`FlattenIllumination` when the primary problem is a large-scale
          illumination gradient rather than local contrast.
        - :class:`SharpenEdgeGauss` when edges need sharpening rather than contrast
          boosting.

    References:
        [1] S. M. Pizer et al., "Adaptive histogram equalization and its
        variations," *Computer Vision, Graphics, and Image Processing*,
        vol. 39, no. 3, pp. 355--368, Sep. 1987.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of EnhanceLocalContrast before detection.
        :doc:`/how_to/notebooks/enhance_low_contrast` for a comparison of
        contrast enhancement methods.
    """

    kernel_size: Annotated[Optional[int], TuneSpec(tunable=False)] = None
    clip_limit: Annotated[float, TuneSpec(0.005, 0.05, log=True)] = 0.01

    def _operate(self, image: Image) -> Image:
        try:
            image.detect_mat[:] = equalize_adapthist(
                    image=image.detect_mat[:],
                    kernel_size=self.kernel_size
                    if self.kernel_size
                    else self._auto_kernel_size(image),
                    clip_limit=self.clip_limit,
                    nbins=2 ** int(image.bit_depth),
            )
            return image
        except RuntimeError as e:
            raise ValueError(f"Value Range: {image.detect_mat.val_range()}") from e

    @staticmethod
    def _auto_kernel_size(image: Image) -> int:
        return int(min(image.gray.shape[:1]) * (1.0 / 15.0))
