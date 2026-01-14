from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from skimage.filters import threshold_otsu
from phenotypic.abc_ import ThresholdDetector


class SecondaryOtsuDetector(ThresholdDetector):
    """Two-stage Otsu refinement detector for improved boundary accuracy.

    SecondaryOtsuDetector applies Otsu's threshold in two stages: (1) initial Otsu
    on full image (or use existing objmask), (2) re-apply Otsu to intensities within
    detected objects only. This refines boundaries by focusing the second threshold
    on the colony intensity distribution, improving edge sharpness when the initial
    detection is coarse or blurred.

    Args:
        None

    Attributes:
        None

    Returns:
        Image: Input image with objmask refined via two-stage Otsu thresholding.

    Raises:
        ValueError: If threshold computation fails (e.g., all pixels same value).

    **Use cases**

    - **Boundary refinement:** Initial detection is correct but blurry at edges.
      Secondary Otsu sharpens colony boundaries within detected regions.
    - **Two-peak distributions:** Colony intensities form distinct peak; applying
      Otsu twice focuses on colony pixels, ignoring background tail.
    - **Halo suppression:** Soft halos around colonies from preprocessing can blur
      edges. Secondary thresholding removes halo while keeping colony centers.

    **Limitations**

    - Two-stage approach loses small colonies at edges. Initial Otsu may miss faint
      or small objects; secondary stage can't recover them.
    - Requires prior detection. If no objects initially detected (objmask all zeros),
      falls back to applying Otsu twice on full image, which may not improve results.
    - Assumes two-phase intensity distribution. If colony pixels span wide intensity
      range or background overlaps with colonies, secondary Otsu may over-segment.
    - Not suitable for isolated detections. If only scattered pixels detected,
      secondary Otsu may aggressively threshold remnants.

    **Parameter effects on colony detection**

    - No user-tunable parameters. Behavior is deterministic: apply Otsu twice.
      Results depend entirely on input image intensity distribution and initial
      objmask quality.

    Examples:
        Refine initial detection with secondary Otsu::

            from phenotypic import Image
            from phenotypic.detect import OtsuDetector, SecondaryOtsuDetector

            plate = Image.imread("plate.jpg")

            # Initial Otsu detection
            detector1 = OtsuDetector()
            intermediate = detector1.apply(plate)

            # Refine boundaries with secondary Otsu
            detector2 = SecondaryOtsuDetector()
            refined = detector2.apply(intermediate)
            mask = refined.objmask[:]
            print(f"Refined mask: {mask.sum()} colony pixels")

        Pipeline combining threshold refinement::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur
            from phenotypic.detect import OtsuDetector, SecondaryOtsuDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                OtsuDetector(),
                SecondaryOtsuDetector()
            ])

            image = Image.imread("plate.jpg")
            result = pipeline.apply(image)
    """

    def _operate(self, image: Image) -> Image:
        """Applies otsu thresholding again to an image that already has an object map. If no,
        object map is found, it will apply the otsu threshold twice"""

        # If there are no objects in the image already perform an initial otsu
        enh_gray = image.enh_gray[:]
        if image.num_objects == 0:
            objmask = enh_gray >= threshold_otsu(enh_gray)
        else:
            objmask = image.objmask[:]

        objvals = enh_gray * objmask
        image.objmask = objvals >= threshold_otsu(objvals[objvals.nonzero()])
        return image
