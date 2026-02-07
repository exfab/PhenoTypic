from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import colour
import numpy as np

from phenotypic.tools_.constants_ import GAMMA_ENCODINGS
from ._lensfun_base import _LensfunCorrectorBase


class LensVignettingCorrector(_LensfunCorrectorBase):
    """Correct lens vignetting (light falloff toward image edges) using lensfunpy.

    Extracts a multiplicative correction factor from the lensfunpy vignetting
    model and applies it to color data (RGB, gray, detect_mat). Handles sRGB
    gamma by linearizing before multiplication and re-encoding after, ensuring
    physically correct brightness correction.

    Does NOT modify objmap or objmask — vignetting correction is purely
    radiometric and does not affect spatial geometry or object labels.

    Attributes:
        cam_maker (str | None): Camera manufacturer. Auto from EXIF 'Image Make'.
        cam_model (str | None): Camera model. Auto from EXIF 'Image Model'.
        lens_maker (str | None): Lens manufacturer. Defaults to cam_maker.
        lens_model (str | None): Lens model. Auto from EXIF 'EXIF LensModel'.
        focal_length (float | None): Focal length in mm. Auto from EXIF.
        aperture (float | None): Aperture f-number. Auto from EXIF.
        distance (float): Subject distance in meters. Default 0.5.

    Returns:
        Image with corrected brightness uniformity across the field.

    Raises:
        ImportError: If lensfunpy is not installed.
        ValueError: If camera/lens parameters cannot be resolved.

    **Use cases for colony phenotyping:**

    - **Uniform brightness** for quantitative colony measurements — vignetting
      makes edge colonies appear darker, skewing size and intensity features.
    - **Improve background subtraction** — uneven illumination from vignetting
      confounds flat-field and rolling-ball background corrections.
    - **DSLR macro photography** — wide-aperture macro shots of plates have
      pronounced vignetting that this corrector removes.

    **Limitations:**

    - Requires the lens to have vignetting calibration data in lensfunpy's DB.
    - Does not correct illumination non-uniformity from external sources (lamps,
      reflections). Use background subtraction for that.
    - At very wide apertures, correction factors can be large (>2x at corners),
      amplifying noise in those regions.
    - Only modifies color data — detection masks are unaffected.

    **Parameter effects:**

    - ``aperture`` is the most important parameter for vignetting — wider
      apertures produce stronger vignetting. The correction model is aperture-
      dependent.
    - ``focal_length`` affects vignetting for zoom lenses where falloff
      varies with focal length.
    - ``distance`` has minimal effect on vignetting for most lenses.

    Examples:
        Correct vignetting with explicit parameters:

        >>> from phenotypic.correction import LensVignettingCorrector
        >>> corrector = LensVignettingCorrector(
        ...     cam_maker="Nikon", cam_model="D3S",
        ...     lens_model="Nikkor 28mm f/2.8D",
        ...     focal_length=28.0, aperture=2.8
        ... )  # doctest: +SKIP
        >>> corrected = corrector.apply(image)  # doctest: +SKIP

        Pipeline with vignetting after distortion correction:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.correction import (
        ...     LensDistortionCorrector, LensVignettingCorrector
        ... )
        >>> pipeline = ImagePipeline([
        ...     LensDistortionCorrector(),
        ...     LensVignettingCorrector(),
        ... ])  # doctest: +SKIP
        >>> result = pipeline.operate(image)  # doctest: +SKIP
    """

    def _operate(self, image: Image) -> Image:
        """Apply vignetting correction to color data only.

        Args:
            image: Input image.

        Returns:
            Image with corrected brightness uniformity.
        """
        import lensfunpy

        params = self._resolve_params(image)
        h, w = image.gray[:].shape[:2]
        mod = self._build_modifier(
            params, h, w, flags=lensfunpy.ModifyFlags.VIGNETTING
        )

        # Extract correction factor by applying to an all-ones array
        ones = np.ones((h, w, 3), dtype=np.float64)
        if not mod.apply_color_modification(ones):
            warnings.warn(
                "No vignetting calibration data available for this lens. "
                "Image returned unchanged.",
                stacklevel=2,
            )
            return image

        # ones is now the multiplicative correction factor per pixel per channel
        factor_3ch = ones
        # Mean across channels for single-channel data
        factor_1ch = factor_3ch.mean(axis=2)

        is_srgb = getattr(image, "gamma", None) == GAMMA_ENCODINGS.SRGB

        # Apply to RGB if present
        if not image.rgb.isempty():
            rgb = image._data.rgb.astype(np.float64) / 255.0
            if is_srgb:
                rgb = colour.CCTF_DECODINGS["sRGB"](rgb)
            rgb *= factor_3ch
            if is_srgb:
                rgb = colour.CCTF_ENCODINGS["sRGB"](rgb)
            image._data.rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

        # Apply to gray (float [0, 1])
        gray = image._data.gray.astype(np.float64)
        if is_srgb:
            gray = colour.CCTF_DECODINGS["sRGB"](gray)
        gray *= factor_1ch
        if is_srgb:
            gray = colour.CCTF_ENCODINGS["sRGB"](gray)
        image._data.gray = np.clip(gray, 0.0, 1.0).astype(
            image._data.gray.dtype
        )

        # Apply to detect_mat (float [0, 1])
        detect = image._data.detect_mat.astype(np.float64)
        if is_srgb:
            detect = colour.CCTF_DECODINGS["sRGB"](detect)
        detect *= factor_1ch
        if is_srgb:
            detect = colour.CCTF_ENCODINGS["sRGB"](detect)
        image._data.detect_mat = np.clip(detect, 0.0, 1.0).astype(
            image._data.detect_mat.dtype
        )

        return image
