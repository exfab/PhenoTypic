from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from skimage.filters import laplace
from typing import Optional

from ..abc_ import ImageEnhancer
from phenotypic.tools_.typing_ import NdArrayField


class LaplaceEnhancer(ImageEnhancer):
    """Enhance colony edges in ``detect_mat`` with a Laplacian operator.

    Applies a discrete Laplacian that responds to rapid intensity changes,
    highlighting colony margins and ring-like features such as swarming
    fronts. Useful as a preprocessing step for contour detection, watershed
    seeding, or separating touching colonies.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        kernel_size: Size of the Laplacian kernel. Smaller values (3)
            capture fine edges but amplify noise; larger values (5--7)
            smooth noise and emphasize broader boundaries. Default: 3.
        mask: Boolean or 0/1 mask to restrict processing to regions of
            interest (e.g., the circular plate area). ``None`` (default)
            processes the full image.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the Laplacian
        edge response. ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Emphasizing colony edges before edge-based segmentation.
        - Detecting ring patterns around colonies for swarming phenotyping.
        - Generating watershed seeds by highlighting boundary regions.

    Consider Also:
        - :class:`HessianFilter` for multi-scale edge and ridge detection
          with more tuning control.
        - :class:`UnsharpMask` for edge enhancement that retains the
          original intensity profile.
        - :class:`PhaseCongruencyEnhancer` for contrast-invariant edge
          detection under uneven illumination.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of edge enhancement on plate images.
    """

    kernel_size: Optional[int] = 3
    mask: NdArrayField | None = None

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = laplace(
            image=image.detect_mat[:],
            ksize=self.kernel_size,
            mask=self.mask,
        )
        return image
