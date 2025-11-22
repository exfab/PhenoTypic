from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from skimage.exposure import equalize_adapthist

from phenotypic.abc_ import ImageEnhancer


class CLAHE(ImageEnhancer):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)

    Applies adaptive histogram equalization while limiting local contrast
    amplification to control noise. For colony images on solid-media agar
    plates, this operation is helpful when illumination is uneven (vignetting,
    shadows from lids) or when colonies are low-contrast/translucent. By
    boosting local contrast tile-by-tile, faint colonies become easier to
    separate from agar background in later thresholding or edge-based steps.

    Use cases (agar plates):
    - Improve visibility of small, faint, or translucent colonies that blend
      into agar.
    - Compensate for gradual illumination roll-off across the plate or
      scanner/lens vignetting.
    - Preconditioning before global/otsu thresholding to reduce sensitivity to
      global intensity shifts.

    Tuning and effects:
    - kernel_size (tile size): Smaller tiles emphasize very local contrast and
      can reveal tiny colonies, but may also accentuate agar texture and noise.
      Larger tiles produce smoother results and are safer for noisy images. If
      left as None, a size proportional to the image dimensions is chosen.
    - clip_limit: Controls how much local contrast is allowed. Lower values
      suppress noise amplification and halos but may leave faint colonies under-
      enhanced. Higher values make colonies pop more strongly but can create
      ringing around dust, condensation droplets, or plate edges.

    Caveats:
    - Can amplify non-biological artifacts (scratches, dust, glare); consider a
      mild blur or artifact suppression before CLAHE if this occurs.
    - Different tiles adjust differently; ensure consistent parameters across a
      batch to avoid biasing downstream measurements.
    - Excessive enhancement may distort intensity-based phenotypes (e.g., pigment
      quantification). Prefer using it only in the `enh_gray` pipeline channel.

    Attributes:
        kernel_size (int | None): Tile size for local equalization. None selects
            an automatic size based on image dimensions.
        clip_limit (float): Normalized contrast limit per tile; smaller values
            reduce amplification, larger values increase it. Default 0.01.
    """

    def __init__(self, kernel_size: int | None = None, clip_limit: float = 0.01, ):
        """
        Parameters:
            kernel_size (int | None): Tile size for adaptive equalization. Smaller
                tiles enhance very local contrast (revealing tiny colonies) but can
                amplify agar texture; larger tiles produce smoother, gentler effects.
                None selects an automatic size based on image dimensions.
            clip_limit (float): Maximum local contrast amplification. Lower values
                reduce noise/halo amplification; higher values make faint colonies
                stand out more but can emphasize dust or condensation.
        """
        self.kernel_size: int = kernel_size
        self.clip_limit: float = clip_limit

    def _operate(self, image: Image) -> Image:
        image.enh_gray[:] = equalize_adapthist(
                image=image.enh_gray[:],
                kernel_size=self.kernel_size if self.kernel_size else self._auto_kernel_size(image),
                clip_limit=self.clip_limit,
                nbins=2 ** int(image.bit_depth)
        )
        return image

    @staticmethod
    def _auto_kernel_size(image: Image) -> int:
        return int(min(image.gray.shape[:1])*(1.0/15.0))
