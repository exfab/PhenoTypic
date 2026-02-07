from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import cv2
import numpy as np

from ._lensfun_base import _LensfunCorrectorBase


class LensTCACorrector(_LensfunCorrectorBase):
    """Correct transverse chromatic aberration (color fringing) using lensfunpy.

    Remaps each RGB channel independently using per-channel coordinate maps
    from lensfunpy's TCA model. This removes color fringing at high-contrast
    edges caused by wavelength-dependent refraction in the lens.

    Only modifies RGB data. Gray, detect_mat, and objmap are unaffected since
    TCA is a color-channel phenomenon with negligible spatial impact on
    single-channel data.

    Attributes:
        cam_maker (str | None): Camera manufacturer. Auto from EXIF 'Image Make'.
        cam_model (str | None): Camera model. Auto from EXIF 'Image Model'.
        lens_maker (str | None): Lens manufacturer. Defaults to cam_maker.
        lens_model (str | None): Lens model. Auto from EXIF 'EXIF LensModel'.
        focal_length (float | None): Focal length in mm. Auto from EXIF.
        aperture (float | None): Aperture f-number. Auto from EXIF.
        distance (float): Subject distance in meters. Default 0.5.

    Returns:
        Image with corrected RGB alignment. Non-RGB components unchanged.

    Raises:
        ImportError: If lensfunpy is not installed.
        ValueError: If camera/lens parameters cannot be resolved.

    **Use cases for colony phenotyping:**

    - **Remove color fringing** at colony boundaries in DSLR images, which
      can confuse color-based phenotyping (e.g., chromogenic assays).
    - **Improve color accuracy** at image edges for experiments measuring
      colony pigmentation or fluorescence bleed-through.
    - **Publication-quality images** — TCA correction eliminates visible
      purple/green fringing at high-contrast colony-to-agar boundaries.

    **Limitations:**

    - Only affects RGB images. Skipped entirely if the image has no RGB data.
    - Requires TCA calibration data in lensfunpy's database (less common
      than distortion calibration for some lenses).
    - Minimal visual effect for lenses with low TCA or at small apertures.
    - Does not correct longitudinal CA (axial color, focus shift by wavelength).

    **Parameter effects:**

    - ``focal_length`` is critical for zoom lenses where TCA varies with zoom.
    - ``aperture`` has less effect on TCA than on vignetting or distortion.
    - TCA correction is typically sub-pixel and most visible at image corners.

    Examples:
        Correct TCA with explicit parameters:

        >>> from phenotypic.correction import LensTCACorrector
        >>> corrector = LensTCACorrector(
        ...     cam_maker="Nikon", cam_model="D3S",
        ...     lens_model="Nikkor 28mm f/2.8D",
        ...     focal_length=28.0, aperture=2.8
        ... )  # doctest: +SKIP
        >>> corrected = corrector.apply(image)  # doctest: +SKIP

        Full lens correction pipeline:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.correction import (
        ...     LensDistortionCorrector, LensVignettingCorrector, LensTCACorrector
        ... )
        >>> pipeline = ImagePipeline([
        ...     LensDistortionCorrector(),
        ...     LensVignettingCorrector(),
        ...     LensTCACorrector(),
        ... ])  # doctest: +SKIP
        >>> result = pipeline.operate(image)  # doctest: +SKIP
    """

    def _operate(self, image: Image) -> Image:
        """Remap RGB channels independently to correct chromatic aberration.

        Args:
            image: Input image.

        Returns:
            Image with corrected RGB channel alignment.
        """
        # TCA only affects RGB; skip if no RGB data
        if image.rgb.isempty():
            return image

        import lensfunpy

        params = self._resolve_params(image)
        h, w = image.gray[:].shape[:2]
        mod = self._build_modifier(
            params, h, w, flags=lensfunpy.ModifyFlags.TCA
        )

        coords = mod.apply_subpixel_distortion()
        if coords is None:
            warnings.warn(
                "No TCA calibration data available for this lens. "
                "Image returned unchanged.",
                stacklevel=2,
            )
            return image

        # coords shape: (h, w, 2, 3) — (x, y) maps for each of 3 channels
        rgb = image._data.rgb.copy()
        corrected = np.empty_like(rgb)

        for ch in range(3):
            map_x = coords[:, :, 0, ch].astype(np.float32)
            map_y = coords[:, :, 1, ch].astype(np.float32)
            corrected[:, :, ch] = cv2.remap(
                rgb[:, :, ch], map_x, map_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

        image._data.rgb = corrected

        return image
