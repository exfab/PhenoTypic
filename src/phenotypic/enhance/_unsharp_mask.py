from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from skimage.filters import unsharp_mask

from ..abc_ import ImageEnhancer


class UnsharpMask(ImageEnhancer):
    """
    Unsharp masking for enhanced colony edge definition on agar plates.

    Unsharp masking is a classical sharpening technique that enhances edges by
    subtracting a blurred copy of the image from the original, then scaling the
    difference to emphasize high-contrast boundaries. On fungal colony plates,
    this makes soft or indistinct colony edges more pronounced, improving the
    ability of thresholding and edge-detection algorithms to identify colony
    boundaries precisely.

    Use cases (agar plates):
    - Low-contrast colonies with soft, gradual edges (translucent growth)
    - Dense plates where colonies blend into background
    - Pre-threshold sharpening to improve segmentation accuracy
    - Enhancing subtle colony morphologies before downstream measurement
    - Improving edge definition when scanner or lens causes slight blurring

    Tuning and effects:
    - radius: Controls the scale of features enhanced. Small values (0.5–2) sharpen
      fine details like small colony boundaries and surface texture; larger values
      (5–15+) enhance broader features and colony-background contrast. For fungal
      colonies, keep radius smaller than the minimum colony radius to avoid creating
      visible halos or merging adjacent colonies. Start at 2.0 for general-purpose
      enhancement.
    - amount: Determines the magnitude of edge enhancement (how much darker/brighter
      the edges become). Low values (0.3–0.7) produce subtle improvements safe for
      noisy images; standard values (1.0–1.5) give moderate sharpening suitable for
      most colony plates; high values (2.0+) create aggressive enhancement but risk
      amplifying noise and creating visible bright/dark halos around colonies.
      Negative amounts produce blur (opposite effect).
    - preserve_range: Leave as False for consistency with other enhancers in the
      framework.

    Caveats:
    - Amplifies noise: In noisy images, unsharp masking sharpens both signal
      (colony edges) and noise artifacts. Consider denoising first (e.g., with
      GaussianBlur, BilateralDenoise, or MedianFilter) on very grainy agar scans.
    - Halo artifacts: Excessive radius or amount creates bright/dark rims around
      colonies, which can be mistaken for separate objects or cause thresholding
      to fail.
    - Already-sharp images: Applying unsharp mask to crisp, high-contrast colonies
      may be redundant and introduce artifacts. Reserve for low-contrast scenarios.

    Attributes:
        radius (float): Standard deviation of Gaussian blur used to compute edges,
            in pixels. Controls the scale of features enhanced.
        amount (float): Multiplier for the sharpening effect. Controls intensity
            of edge enhancement.
        preserve_range (bool): Whether to keep the original range of pixel values
            (False by default).

    Examples:
        .. dropdown:: Sharpening low-contrast fungal colonies before detection

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.enhance import UnsharpMask
                from phenotypic.detect import OtsuDetector

                # Load image of low-contrast plate (e.g., translucent yeasts)
                image = Image.from_image_path("yeast_plate.jpg")

                # Apply unsharp masking with moderate settings
                sharpener = UnsharpMask(radius=2.0, amount=1.2)
                sharpened = sharpener.apply(image)

                # Detect colonies in sharpened enhanced grayscale
                detector = OtsuDetector()
                detected = detector.apply(sharpened)

                # Original image untouched, detection on enhanced data
                colonies = detected.objects
                print(f"Detected {len(colonies)} colonies")

        .. dropdown:: Tuning radius and amount for dense high-throughput plates

            .. code-block:: python

                from phenotypic import Image, ImagePipeline
                from phenotypic.enhance import UnsharpMask, GaussianBlur
                from phenotypic.detect import OtsuDetector

                # For high-resolution 384-well plate scans with tiny colonies,
                # use small radius to avoid merging adjacent growth

                pipeline = ImagePipeline()

                # Step 1: Light blur to reduce scanner noise
                pipeline.add(GaussianBlur(sigma=1))

                # Step 2: Enhance edges with small radius for dense plates
                # radius=1.0 emphasizes only fine features (individual colonies)
                pipeline.add(UnsharpMask(radius=1.0, amount=1.5))

                # Step 3: Detect in enhanced grayscale
                pipeline.add(OtsuDetector())

                # Process a batch of images
                images = [Image.from_image_path(f) for f in image_paths]
                results = pipeline.operate(images)

                for i, result in enumerate(results):
                    print(f"Plate {i}: {len(result.objects)} colonies")

        .. dropdown:: Aggressive sharpening for very translucent colonies

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.enhance import UnsharpMask

                # For extremely low-contrast colonies (e.g., slow-growing mutants,
                # low-turbidity liquid culture plates), use higher amount

                image = Image.from_image_path("faint_colonies.jpg")

                # Aggressive parameters: larger radius for broader features,
                # higher amount for stronger enhancement
                aggressive_sharpener = UnsharpMask(radius=5.0, amount=2.5)
                enhanced = aggressive_sharpener.apply(image)

                # Inspect result for artifacts (halos); adjust if needed
                # If halos appear, reduce amount to 1.5–2.0
                print("Sharpening applied. Check for halo artifacts around large colonies.")
    """

    def __init__(
        self,
        radius: float = 2.0,
        amount: float = 1.0,
        preserve_range: bool = False,
    ):
        """
        Parameters:
            radius (float): Standard deviation (sigma) of the Gaussian blur in pixels.
                Defines the scale of features to enhance. Small values (0.5–2) sharpen
                fine details (thin colony edges, small morphologies); larger values
                (5–15) enhance broad features (large colonies, colony-background
                separation). Must be > 0. For fungal colonies, keep below the typical
                colony radius to avoid merging adjacent colonies. Recommended: 2.0–3.0
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
        """
        if radius <= 0:
            raise ValueError("radius must be > 0")

        self.radius = float(radius)
        self.amount = float(amount)
        self.preserve_range = bool(preserve_range)

    def _operate(self, image: Image) -> Image:
        """Apply unsharp masking to enhance colony edges in the enhanced grayscale channel."""
        image.enh_gray[:] = unsharp_mask(
            image=image.enh_gray[:],
            radius=self.radius,
            amount=self.amount,
            preserve_range=self.preserve_range,
            channel_axis=None,
        )
        return image
