from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image
from skimage.filters import laplace
from typing import Optional
import numpy as np

from ..abc_ import ImageEnhancer


class LaplaceEnhancer(ImageEnhancer):
    """
    Laplacian edge enhancement for colony boundaries.

    Applies a Laplacian operator that responds to rapid intensity changes and
    highlights edges. In agar plate images, this can delineate colony margins
    and ring-like features, improving contour detection or watershed seeds.

    Use cases (agar plates):
    - Emphasize colony edges prior to edge-based segmentation or as a cue for
      separating touching colonies.
    - Detect ring patterns around colonies (e.g., swarming fronts) for phenotyping.

    Tuning and effects:
    - kernel_size: Larger kernels produce a smoother, more global edge response
      and can suppress small noise; smaller kernels capture fine edges but may
      amplify noise and agar texture.
    - mask: Restrict processing to the plate region to avoid dish edge glare or
      labels. A binary mask focusing on the circular plate is often useful.

    Caveats:
    - Laplacian is sensitive to noise; consider a light `GaussianBlur` first.
    - May enhance non-biological artifacts (scratches, dust). Combine with masking
      or artifact removal if necessary.

    Parameters:
        kernel_size (Optional[int]): Size of the Laplacian kernel controlling
            edge scale; smaller captures fine edges, larger smooths noise.
        mask (Optional[numpy.ndarray]): Optional boolean/0-1 mask to limit the
            operation to specific regions (e.g., the plate area).
    """

    def __init__(self, kernel_size: Optional[int] = 3, mask: Optional[np.ndarray] = None):
        """
        Parameters:
            kernel_size (Optional[int]): Controls the edge scale. Smaller values
                pick up fine edges but increase noise sensitivity; larger values
                smooth noise and emphasize broader boundaries.
            mask (Optional[np.ndarray]): Boolean/0-1 mask to limit processing to
                regions of interest (e.g., the circular plate), reducing artifacts
                from dish rims or labels.
        """
        self.kernel_size: Optional[np.ndarray] = kernel_size
        self.mask: Optional[np.ndarray] = mask

    def _operate(self, image: Image) -> Image:
        image.enh_gray[:] = laplace(
                image=image.enh_gray[:],
                ksize=self.kernel_size,
                mask=self.mask,
        )
        return image
