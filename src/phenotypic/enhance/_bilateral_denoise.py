from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.restoration import denoise_bilateral

from ..abc_ import ImageEnhancer


class BilateralDenoise(ImageEnhancer):
    """Denoise ``detect_mat`` with edge-preserving bilateral filtering.

    Averages pixel values based on both spatial proximity and intensity
    similarity, preserving sharp colony boundaries while smoothing uniform
    regions such as agar background. Effectively removes scanner noise,
    agar grain, dust speckles, and condensation artifacts without blurring
    colony edges.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Noisy or grainy agar scans from high-ISO photography or old scanners.
        - Plates with surface condensation, dust speckles, or uneven agar
          texture.
        - Preprocessing before thresholding when colony edges must remain
          sharp.
        - Low-quality captures where colony morphology must be preserved.

    Consider Also:
        - :class:`NonLocalMeansDenoiser` for stronger denoising of repetitive
          textures at higher computational cost.
        - :class:`BM3DDenoiser` for state-of-the-art structured noise removal.
        - :class:`SubtractGaussian` when the primary problem is illumination
          gradients rather than pixel-level noise.

    Args:
        sigma_color: Intensity similarity weighting. Small values
            (0.02--0.05) preserve subtle boundaries; medium values
            (0.05--0.15) balance denoising and edge preservation; large
            values (0.2--0.5) smooth aggressively. ``None`` (default)
            auto-estimates from image statistics.
        sigma_spatial: Spatial distance weighting in pixels. Small values
            (1--5) apply local denoising; medium values (10--20) smooth
            regionally; large values (30--50) smooth wide areas. Keep
            below the minimum colony diameter. Default: 15.
        win_size: Window size for filter computation. ``None`` (default)
            auto-calculates from ``sigma_spatial``.
        mode: Boundary handling. Accepted values: ``'constant'``,
            ``'edge'``, ``'symmetric'``, ``'reflect'``, ``'wrap'``.
            Default: ``'constant'``.
        cval: Fill value when ``mode='constant'``. Default: 0.
        clip: Clip output to [0, 1]. Default: ``True``. Set to ``False``
            when using with variance-stabilizing transforms (e.g., GAT).

    Returns:
        Image: Input image with ``detect_mat`` smoothed by bilateral
        filtering. ``rgb`` and ``gray`` are unchanged.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of denoising pipelines on plate images.
        :doc:`/how_to/notebooks/denoise_low_light` for edge-preserving
        denoising strategies on low-light plate images.
    """

    def __init__(
            self,
            sigma_color: float | None = None,
            sigma_spatial: float = 15,
            *,
            win_size: int | None = None,
            mode: str = "constant",
            cval: float = 0,
            clip: bool = True,
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
            clip (bool): Whether to clip output to [0, 1] range. Default True. Set to
                False when using with variance-stabilizing transforms (e.g., GAT) that
                require preserving the original scale of transformed data.
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
        self.clip = clip

    def _operate(self, image: Image) -> Image:
        """Apply bilateral denoising to reduce noise while preserving colony edges in the detection matrix channel."""
        # denoise_bilateral may require a writable array, so create a copy
        result = denoise_bilateral(
                image=image.detect_mat[:].copy(),
                sigma_color=self.sigma_color,
                sigma_spatial=self.sigma_spatial,
                win_size=self.win_size,
                mode=self.mode,
                cval=self.cval,
                channel_axis=None,
        )
        if self.clip:
            result = result.clip(0.0, 1.0)
        image.detect_mat[:] = result
        return image
