from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class ManualDetector(ThresholdDetector):
    """Manual threshold detector for colony segmentation with user-defined threshold.

    ManualDetector applies a user-specified threshold value to convert the enhanced
    grayscale image into a binary colony mask. Unlike automatic methods (Otsu, Li, etc.),
    this detector gives explicit control over the threshold cutoff, enabling tuning for
    specific imaging conditions or experimental requirements.

    **When to use ManualDetector**

    Manual thresholding is ideal when:

    - **Known threshold value:** Empirical testing or domain knowledge has identified
      an optimal threshold for your imaging setup.
    - **Consistent imaging:** Plates are imaged under standardized conditions (fixed
      lighting, exposure, background) where a single threshold works reliably.
    - **Fine-tuning control:** Automatic methods over/under-segment and you need
      precise adjustment of the intensity cutoff.
    - **Baseline comparison:** Testing automatic methods against a known-good manual
      threshold for validation.

    **Use cases and imaging artifacts**

    - **High-contrast plates:** Colonies are bright/dark relative to background with
      minimal variation. A fixed threshold cleanly separates foreground from background.
    - **Calibrated imaging pipeline:** Camera settings and illumination are locked,
      producing reproducible intensity distributions across plates.
    - **Iterative refinement:** Start with automatic method (e.g., Otsu), inspect
      results, then switch to manual threshold to correct systematic errors.
    - **Edge case handling:** Automatic methods fail on atypical histograms (heavily
      skewed, multimodal). Manual threshold provides direct control.

    **Caveats and limitations**

    - **No adaptivity:** A single global threshold doesn't handle uneven illumination
      or intensity gradients across the plate. Consider local/adaptive methods if
      lighting varies spatially.
    - **Tuning required:** You must determine the optimal threshold value through
      trial and error or prior analysis. Poor threshold choice leads to over/under-
      segmentation.
    - **Bit depth dependency:** Threshold values are sensitive to image bit depth
      (8-bit: 0-255, 16-bit: 0-65535). Ensure threshold matches your image's
      intensity range.
    - **Non-portable:** A threshold tuned for one imaging setup may not generalize to
      different cameras, lighting, or agar types.

    **Parameter effects on colony detection**

    - **threshold (float):**

      - **Higher values:** Fewer, larger colonies detected. Reduces false positives
        (noise) but may miss faint or small colonies. Use for bright colonies on
        dark background or to filter weak signals.
      - **Lower values:** More, smaller colonies detected. Increases sensitivity to
        faint growth but prone to noise and false positives. Use for dark colonies
        on bright background or low-contrast plates.
      - **Tuning strategy:** Start with image histogram (``np.histogram(image.enh_gray[:])``).
        Identify the intensity valley between background and colony peaks. Set
        threshold in that valley. Adjust up/down based on mask quality.

    - **ignore_zeros (bool, default=True):**

      - **True:** Exclude pure black pixels (intensity=0) from threshold computation.
        Essential for images with black borders, masks, or padding. Prevents
        artificial skewing of the threshold toward zero.
      - **False:** Include all pixels. Use only if the image has no border artifacts
        and zero is a meaningful intensity (rare).

    - **ignore_borders (bool, default=True):**

      - **True:** Remove colonies touching image edges via ``clear_border()``.
        Eliminates partial colonies at plate boundaries, improves measurement
        accuracy. Recommended for grid-based analysis where edge wells are excluded.
      - **False:** Keep edge-touching colonies. Use if edges contain valid data or
        for non-grid single-colony images.

    **Workflow: Determine threshold empirically**

    1. Load a representative test image::

        from phenotypic import Image
        import numpy as np
        import matplotlib.pyplot as plt

        image = Image("test_plate.jpg")
        enh = image.enh_gray[:]

    2. Inspect the intensity histogram::

        plt.hist(enh.ravel(), bins=256, range=(0, 255))
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.show()

    3. Identify the threshold value at the valley between background and colony peaks.

    4. Test the threshold::

        detector = ManualDetector(threshold=120, ignore_zeros=True)
        result = detector.apply(image)
        result.show_overlay()  # Visual QC

    5. Adjust threshold up/down based on over/under-segmentation.

    **Comparison with automatic methods**

    - **Otsu:** Minimizes within-class variance. Good for balanced histograms but
      may fail on skewed distributions. ManualDetector allows override when Otsu's
      assumption breaks.
    - **Li/Yen/Triangle:** Alternative automatic strategies with different histogram
      assumptions. ManualDetector provides deterministic control when automatic
      methods disagree.
    - **Local/Adaptive:** Spatially varying thresholds for uneven illumination.
      ManualDetector is global (single threshold). Use adaptive if lighting varies.

    **Integration with ImagePipeline**

    Manual thresholding typically follows enhancement operations::

        from phenotypic import ImagePipeline
        from phenotypic.enhance import GaussianBlur, CLAHE
        from phenotypic.detect import ManualDetector

        pipeline = ImagePipeline([
            GaussianBlur(sigma=1.5),         # Denoise
            CLAHE(clip_limit=2.0),           # Enhance local contrast
            ManualDetector(threshold=130)    # Apply tuned threshold
        ])

    Args:
        threshold: Intensity value for binary thresholding. Pixels with intensity
            >= threshold become foreground (colonies), pixels < threshold become
            background. Must be within the image's intensity range (e.g., 0-255 for
            8-bit, 0-65535 for 16-bit). Higher values = stricter detection (fewer
            colonies), lower values = more sensitive (more colonies).
        ignore_zeros: If True, exclude pixels with intensity=0 from threshold
            computation. Prevents black borders or masks from skewing the threshold.
            Default is True (recommended for most plate images).
        ignore_borders: If True, remove colonies touching image edges via
            ``skimage.segmentation.clear_border()``. Eliminates partial colonies at
            plate boundaries. Default is True (recommended for grid analysis).

    Attributes:
        threshold: The user-specified threshold value for segmentation.
        ignore_zeros: Whether to exclude zero-intensity pixels.
        ignore_borders: Whether to remove edge-touching objects.

    Returns:
        Image: The input image with ``objmask`` attribute set to the binary mask
        (True = colony, False = background). The ``objmap`` attribute is not set by
        this detector; use a refiner or labeling operation to generate labeled
        object maps.

    Raises:
        ValueError: If threshold is negative or exceeds the image's intensity range.

    Examples:
        .. dropdown:: Detect colonies with a manually tuned threshold

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.detect import ManualDetector

                # Load a plate image
                plate = Image("agar_plate.jpg")

                # Apply manual threshold (tuned via histogram inspection)
                detector = ManualDetector(threshold=120, ignore_zeros=True)
                detected = detector.apply(plate)

                # Access binary mask
                mask = detected.objmask[:]
                num_foreground_pixels = mask.sum()
                print(f"Foreground pixels: {num_foreground_pixels}")

        .. dropdown:: Compare manual vs automatic thresholding

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.detect import OtsuDetector, ManualDetector
                import matplotlib.pyplot as plt

                plate = Image("agar_plate.jpg")

                # Automatic Otsu threshold
                otsu_result = OtsuDetector().apply(plate)

                # Manual threshold (empirically tuned)
                manual_result = ManualDetector(threshold=130).apply(plate)

                # Compare masks
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(otsu_result.objmask[:], cmap='gray')
                axes[0].set_title("Otsu (automatic)")
                axes[1].imshow(manual_result.objmask[:], cmap='gray')
                axes[1].set_title(f"Manual (threshold={130})")
                plt.show()

        .. dropdown:: Tune threshold based on histogram analysis

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.detect import ManualDetector
                import numpy as np
                import matplotlib.pyplot as plt

                plate = Image("agar_plate.jpg")
                enh = plate.enh_gray[:]

                # Plot histogram to identify threshold
                plt.hist(enh.ravel(), bins=256, range=(0, 255))
                plt.axvline(x=120, color='r', linestyle='--', label='threshold=120')
                plt.xlabel("Pixel Intensity")
                plt.ylabel("Frequency")
                plt.legend()
                plt.show()

                # Apply threshold
                detector = ManualDetector(threshold=120)
                result = detector.apply(plate)

        .. dropdown:: Pipeline with manual threshold for standardized imaging

            .. code-block:: python

                from phenotypic import ImagePipeline
                from phenotypic.enhance import GaussianBlur, ContrastStretching
                from phenotypic.detect import ManualDetector
                from phenotypic.refine import RemoveSmallObjectsRefiner

                # Build pipeline for consistent imaging setup
                pipeline = ImagePipeline([
                    GaussianBlur(sigma=1.0),              # Denoise
                    ContrastStretching(),                 # Normalize intensity
                    ManualDetector(threshold=115),        # Known-good threshold
                    RemoveSmallObjectsRefiner(min_size=20)  # Cleanup noise
                ])

                # Process standardized plate images
                from phenotypic import Image
                plates = [Image(f"plate_{i}.jpg") for i in range(10)]
                results = pipeline.operate(plates)
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
        """Apply manual binary thresholding to the enhanced grayscale image.

        This function modifies the input image by applying a user-specified threshold
        to its enhanced matrix (``enh_gray``). Pixels with intensity >= threshold
        become foreground (True in the binary mask), pixels < threshold become
        background (False). The resulting binary mask is stored in the image's
        ``objmask`` attribute.

        Args:
            image: The input image object. Must have an ``enh_gray`` attribute
                (enhanced grayscale matrix for processing). Optionally uses
                ``bit_depth`` to validate threshold range.

        Returns:
            Image: The input image with its ``objmask`` attribute updated to the
                computed binary mask.

        Raises:
            ValueError: If threshold is negative or exceeds the image's intensity
                range (inferred from bit depth if available).
        """
        enh_matrix = image.enh_gray[:]

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
