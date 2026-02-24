from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class ManualDetector(ThresholdDetector):
    """Manual threshold detector for colony segmentation with user-defined threshold.

    ManualDetector applies a user-specified intensity cutoff to convert the enhanced
    grayscale image into a binary colony mask. Unlike automatic methods (Otsu, Li, etc.),
    this provides explicit control over detection sensitivity, enabling tuning for
    specific imaging conditions or experimental requirements.

    Args:
        threshold: Intensity value for binary thresholding (0-255 for 8-bit, 0-65535
            for 16-bit). Pixels >= threshold become colonies (True), others background
            (False). Higher values detect fewer/larger colonies (conservative), lower
            values detect more/smaller colonies (sensitive).

        ignore_zeros: If True (default), exclude zero-intensity pixels from threshold
            computation. Essential for images with black borders or masks.

        ignore_borders: If True (default), remove colonies touching image edges via
            clear_border(). Eliminates partial colonies at plate boundaries.

    Attributes:
        threshold: The user-specified threshold intensity value.
        ignore_zeros: Whether to exclude zero-intensity pixels.
        ignore_borders: Whether to remove edge-touching colonies.

    Returns:
        Image: Input image with objmask set to binary mask (True = colony, False = background).

    Raises:
        ValueError: If threshold is negative or exceeds image intensity range.

    **Use cases**

    - **Known-good threshold:** Empirical testing has identified an optimal threshold
      for your imaging setup.
    - **Standardized imaging:** Plates are imaged under fixed conditions (lighting,
      exposure, camera settings) where a single threshold works reliably.
    - **Override automatic methods:** Automatic methods over/under-segment; manual
      threshold provides precise control.
    - **High-contrast imaging:** Colonies are bright/dark relative to background with
      minimal variation. A fixed cutoff cleanly separates foreground from background.

    **Limitations**

    - No spatial adaptation. A single global threshold doesn't handle uneven illumination
      or intensity gradients. Consider local/adaptive methods for severe vignetting.
    - Tuning required. You must determine optimal threshold through trial and error or
      prior histogram analysis. Poor choices lead to over/under-segmentation.
    - Not portable. A threshold tuned for one imaging setup may not generalize to
      different cameras, lighting, or agar types.
    - Bit-depth dependent. Threshold values differ for 8-bit vs 16-bit images.

    **Parameter effects on colony detection**

    - **threshold:** Higher values → fewer colonies, lower false positives (stricter).
      Lower values → more colonies, higher noise inclusion (more sensitive). Start with
      histogram inspection to find intensity valley between background and colony peaks.
    - **ignore_zeros/ignore_borders:** Remove preprocessing artifacts (black borders,
      partial edge colonies).

    Examples:
        Basic detection with manual threshold::

            from phenotypic import Image
            from phenotypic.detect import ManualDetector

            plate = Image.imread("agar_plate.jpg")
            detector = ManualDetector(threshold=120, ignore_zeros=True)
            detected = detector.apply(plate)
            mask = detected.objmask[:]
            print(f"Foreground pixels: {mask.sum()}")

        Pipeline with manual threshold for standardized imaging::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur, CLAHE
            from phenotypic.detect import ManualDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                CLAHE(clip_limit=2.0),
                ManualDetector(threshold=130, ignore_borders=True)
            ])

            image = Image.imread("plate.jpg")
            result = pipeline.apply(image)
    """

    def __init__(
            self,
            threshold: float = 0.5,
            ignore_zeros: bool = True,
            ignore_borders: bool = True
    ):
        self.threshold = threshold
        self.ignore_zeros = ignore_zeros
        self.ignore_borders = ignore_borders

    def _operate(self, image: Image) -> Image:
        """Apply manual binary thresholding to the detection matrix image.

        This function modifies the input image by applying a user-specified threshold
        to its enhanced matrix (``detect_mat``). Pixels with intensity >= threshold
        become foreground (True in the binary mask), pixels < threshold become
        background (False). The resulting binary mask is stored in the image's
        ``objmask`` attribute.

        Args:
            image: The input image object. Must have an ``detect_mat`` attribute
                (detection matrix for processing). Optionally uses
                ``bit_depth`` to validate threshold range.

        Returns:
            Image: The input image with its ``objmask`` attribute updated to the
                computed binary mask.

        Raises:
            ValueError: If threshold is negative or exceeds the image's intensity
                range (inferred from bit depth if available).
        """
        enh_matrix = image.detect_mat[:]

        # Validate threshold range
        if self.threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {self.threshold}")

        # Apply threshold
        mask = enh_matrix >= self.threshold

        # Optionally clear borders
        mask = clear_border(mask) if self.ignore_borders else mask

        # Set objmask
        image.objmask = mask
        return image


# Set the docstring so that it appears in the sphinx documentation
ManualDetector.apply.__doc__ = ManualDetector._operate.__doc__
