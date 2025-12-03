from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from skimage.restoration import denoise_bilateral

from ..abc_ import ImageEnhancer


class BilateralDenoise(ImageEnhancer):
    """
    Bilateral filtering for edge-preserving noise reduction on fungal colony plates.

    Bilateral filtering is a non-linear denoising technique that averages pixel values
    based on two criteria: spatial proximity (nearby pixels are weighted higher) and
    radiometric similarity (pixels with similar intensities are weighted higher). This
    dual constraint preserves sharp intensity discontinuities (colony edges) while
    smoothing uniform regions (agar background, colony interiors). On fungal colony
    plates, bilateral denoising effectively removes scanner noise, agar grain, dust
    speckles, and condensation artifacts without blurring colony boundaries—ideal
    preprocessing before segmentation algorithms.

    Use cases (agar plates):
    - Noisy or grainy agar scans (high ISO photography, old scanners)
    - Uneven agar texture, surface condensation, dust speckles
    - Background variations that confuse thresholding
    - Preprocessing before ObjectDetector when colony edges must remain sharp
    - Handling low-quality captures while preserving colony morphology

    Tuning and effects:
    - sigma_color: Controls how strictly pixel intensities must match to influence
      each other. Small values (0.02–0.05) only average pixels with very similar
      brightness, preserving subtle colony boundaries but leaving more noise. Medium
      values (0.05–0.15) balance denoising and edge preservation, suitable for most
      fungal colony plates. Large values (0.2–0.5) aggressively average pixels across
      a wider brightness range, producing heavy smoothing but risking loss of faint
      colony features or boundary blurring. If None (default), automatically estimated
      from image statistics. For float images in [0,1], these are reasonable defaults;
      for uint8 images, scale values proportionally (e.g., 0.05 float ≈ 13 for uint8).
    - sigma_spatial: Controls the spatial neighborhood size; larger values smooth over
      wider areas. Small values (1–5) apply local denoising that preserves fine colony
      texture but removes only local noise. Medium values (10–20) provide balanced
      regional smoothing, recommended for general-purpose use. Large values (30–50)
      smooth over wide regions, helpful for correcting illumination gradients but
      risky for small colonies or dense plates. Keep below the minimum expected colony
      diameter to avoid over-smoothing or merging adjacent colonies.
    - win_size: Window size for filter computations. If None (default), automatically
      calculated from sigma_spatial; generally safe to leave unset.
    - mode: Border handling strategy. 'constant' (default) pads with cval; 'reflect'
      mirrors edges. 'constant' with cval=0 works well for agar plates.
    - cval: Fill value at image boundaries when mode='constant'. 0 (black) is typical
      for agar backgrounds.

    Caveats:
    - Computational cost: Bilateral filtering is slower than simple Gaussian blur,
      especially with large sigma_spatial. For large images, keep sigma_spatial ≤ 15
      to maintain reasonable speed.
    - Data type sensitivity: The function internally converts images to float [0,1].
      Parameter interpretation (especially sigma_color) assumes this range. Very bright
      or very dark images may require parameter adjustment.
    - Over-smoothing: If sigma_color is too high, the filter may blur colony boundaries
      or merge nearby colonies into connected regions, breaking segmentation.
    - Not a substitute for proper illumination correction: Bilateral denoising smooths
      background variations but does not remove large-scale illumination gradients
      (vignetting, shadows). Use RollingBallRemoveBG or GaussianSubtract for that.

    Attributes:
        sigma_color (float | None): Standard deviation of intensity/color difference for
            similarity weighting. Controls edge preservation vs smoothing trade-off. None
            means auto-estimate from image.
        sigma_spatial (float): Standard deviation of spatial distance for weighting.
            Controls neighborhood size.
        win_size (int | None): Window size for bilateral filtering. None means
            auto-calculate.
        mode (str): Boundary handling mode ('constant', 'edge', 'symmetric', 'reflect',
            'wrap').
        cval (float): Constant fill value when mode='constant'.

    Examples:
        .. dropdown:: Denoising a grainy agar plate scan before colony detection

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.enhance import BilateralDenoise
                from phenotypic.detect import OtsuDetector

                # Load a noisy scan (e.g., high-ISO smartphone image or old scanner)
                image = Image.from_image_path("noisy_plate.jpg")

                # Apply bilateral denoising with moderate settings
                denoiser = BilateralDenoise(sigma_color=0.1, sigma_spatial=15)
                denoised = denoiser.apply(image)

                # Detect colonies in cleaned enhanced grayscale
                detector = OtsuDetector()
                detected = detector.apply(denoised)

                colonies = detected.objects
                print(f"Detected {len(colonies)} colonies in denoised image")

        .. dropdown:: Chaining denoising and sharpening for challenging images

            .. code-block:: python

                from phenotypic import Image, ImagePipeline
                from phenotypic.enhance import BilateralDenoise, UnsharpMask
                from phenotypic.detect import OtsuDetector

                # Scenario: Noisy image with low-contrast colonies
                # Solution: Denoise first (remove artifacts), then sharpen (enhance edges)

                pipeline = ImagePipeline()

                # Step 1: Remove noise while preserving colony edges
                # sigma_color=0.08 balances denoising and edge sharpness
                pipeline.add(BilateralDenoise(sigma_color=0.08, sigma_spatial=15))

                # Step 2: Sharpen remaining edges for better segmentation
                pipeline.add(UnsharpMask(radius=2.0, amount=1.5))

                # Step 3: Detect
                pipeline.add(OtsuDetector())

                images = [Image.from_image_path(f) for f in image_paths]
                results = pipeline.operate(images)

        .. dropdown:: Heavy denoising for very grainy plates with large colonies

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.enhance import BilateralDenoise

                # For large-colony plates (e.g., petri dishes, sparse growth) with heavy
                # scanner noise or texture, use larger sigma_spatial to smooth broader regions

                image = Image.from_image_path("sparse_grainy_plate.jpg")

                # Heavy denoising: large spatial neighborhood, moderate color tolerance
                heavy_denoiser = BilateralDenoise(
                    sigma_color=0.15,      # Blend pixels across wider brightness range
                    sigma_spatial=30,      # Smooth over large neighborhoods
                )
                denoised = heavy_denoiser.apply(image)

                # Result: Agar grain and dust removed, but large colony edges preserved
                print("Heavy denoising applied.")

        .. dropdown:: Selective denoising for high-resolution dense plates

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.enhance import BilateralDenoise

                # For high-resolution 384-well plates with tiny colonies, small sigma_spatial
                # preserves fine structure while removing only local speckles

                image = Image.from_image_path("dense_hires_plate.jpg")

                # Conservative denoising: small spatial neighborhood, strict color matching
                conservative_denoiser = BilateralDenoise(
                    sigma_color=0.04,      # Only average similar pixels
                    sigma_spatial=8,       # Small neighborhood, preserves fine details
                )
                denoised = conservative_denoiser.apply(image)

                # Result: Local speckles removed, but colony boundaries and microstructure intact
                print("Light denoising applied; fine morphology preserved.")
    """

    def __init__(
        self,
        sigma_color: float | None = None,
        sigma_spatial: float = 15,
        win_size: int | None = None,
        mode: str = "constant",
        cval: float = 0,
    ):
        """
        Parameters:
            sigma_color (float | None): Standard deviation for grayvalue/color similarity.
                Controls how permissive the filter is when averaging nearby pixels. Small
                values (0.02–0.05 for float images) enforce strict color matching,
                preserving edges but leaving more noise. Medium values (0.05–0.15)
                provide balanced denoising and edge preservation—recommended for most
                fungal colony imaging. Large values (0.2–0.5) aggressively average
                across brightness ranges, risking boundary blur. If None (default),
                automatically estimated from the standard deviation of the image.
                For uint8 images (0–255), scale values proportionally: 0.05 float
                corresponds roughly to 13 in uint8 scale. Recommended: leave as None
                for automatic estimation, or set to 0.08–0.12 for typical colony plates.
            sigma_spatial (float): Standard deviation for spatial distance in pixels.
                Controls the extent of the neighborhood influencing each pixel. Small
                values (1–5) apply highly local denoising, preserving fine texture.
                Medium values (10–20) smooth regionally without over-smoothing—suitable
                for general use. Large values (30–50) smooth broad areas, helpful for
                correcting illumination variations but risking loss of small colonies
                or merging of adjacent growth. Recommended: 15 for balanced results;
                adjust based on colony size (keep smaller than minimum colony diameter).
            win_size (int | None): Window size for bilateral filter computation. If None
                (default), automatically calculated as max(5, 2 * ceil(3 * sigma_spatial) + 1).
                Generally safe to leave as None; adjust only if you have specific
                performance or memory constraints.
            mode (str): How to handle image boundaries. Options: 'constant' (default,
                pad with cval), 'edge' (replicate edge), 'symmetric', 'reflect', 'wrap'.
                'constant' with cval=0 works well for agar plate backgrounds (black edges).
                'reflect' mirrors edges, useful for non-border regions.
            cval (float): Constant fill value for boundaries when mode='constant'. Default
                is 0 (black), appropriate for agar backgrounds.
        """
        if sigma_spatial <= 0:
            raise ValueError("sigma_spatial must be > 0")

        if sigma_color is not None and sigma_color <= 0:
            raise ValueError("sigma_color must be > 0 or None")

        if mode not in ["constant", "edge", "symmetric", "reflect", "wrap"]:
            raise ValueError(
                f'mode must be one of "constant", "edge", "symmetric", "reflect", '
                f'"wrap"; got {mode!r}'
            )

        self.sigma_color = sigma_color
        self.sigma_spatial = float(sigma_spatial)
        self.win_size = win_size
        self.mode = mode
        self.cval = cval

    def _operate(self, image: Image) -> Image:
        """Apply bilateral denoising to reduce noise while preserving colony edges in the enhanced grayscale channel."""
        # denoise_bilateral may require a writable array, so create a copy
        image.enh_gray[:] = denoise_bilateral(
            image=image.enh_gray[:].copy(),
            sigma_color=self.sigma_color,
            sigma_spatial=self.sigma_spatial,
            win_size=self.win_size,
            mode=self.mode,
            cval=self.cval,
            channel_axis=None,
        )
        return image
