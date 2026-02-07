from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import cv2
import numpy as np
from scipy.sparse import csc_matrix

from ._lensfun_base import _LensfunCorrectorBase


class LensDistortionCorrector(_LensfunCorrectorBase):
    """Correct barrel/pincushion lens distortion using lensfunpy calibration data.

    Remaps all image components (RGB, gray, detect_mat, objmap) to remove
    geometric distortion introduced by the camera lens. Uses lensfunpy's
    calibration database to compute per-pixel coordinate corrections.

    Attributes:
        cam_maker (str | None): Camera manufacturer. Auto from EXIF 'Image Make'.
        cam_model (str | None): Camera model. Auto from EXIF 'Image Model'.
        lens_maker (str | None): Lens manufacturer. Defaults to cam_maker.
        lens_model (str | None): Lens model. Auto from EXIF 'EXIF LensModel'.
        focal_length (float | None): Focal length in mm. Auto from EXIF.
        aperture (float | None): Aperture f-number. Auto from EXIF.
        distance (float): Subject distance in meters. Default 0.5.

    Returns:
        Image with corrected geometry. All components are remapped together.

    Raises:
        ImportError: If lensfunpy is not installed.
        ValueError: If camera/lens parameters cannot be resolved.

    **Use cases for colony phenotyping:**

    - **Correct barrel distortion** from wide-angle or macro lenses used to
      photograph agar plates, ensuring colonies near edges are not spatially
      compressed or stretched.
    - **Improve grid alignment** by removing geometric distortion before
      grid detection, so detected grid lines are straight.
    - **Quantitative morphometry** — distortion changes apparent colony size
      and shape near image edges; correction ensures measurements are uniform
      across the field of view.

    **Limitations:**

    - Requires the camera and lens to be in lensfunpy's calibration database.
    - Remapping may introduce minor interpolation artifacts at sub-pixel level.
    - Does not correct radial motion blur or focus fall-off (use vignetting corrector).
    - GridImage positions must be re-detected after geometric correction.

    **Parameter effects:**

    - All parameters (cam_maker, cam_model, etc.) select the distortion model
      from lensfunpy's database. No user-tunable correction strength — the model
      is derived from calibration data for the specific lens.
    - ``distance`` affects the correction model for lenses with focus-dependent
      distortion. Default 0.5m suits typical plate photography.

    Examples:
        Correct distortion with explicit parameters:

        >>> from phenotypic.correction import LensDistortionCorrector
        >>> corrector = LensDistortionCorrector(
        ...     cam_maker="Nikon", cam_model="D3S",
        ...     lens_model="Nikkor 28mm f/2.8D",
        ...     focal_length=28.0, aperture=2.8
        ... )  # doctest: +SKIP
        >>> corrected = corrector.apply(image)  # doctest: +SKIP

        Auto-detect from EXIF in a pipeline:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.correction import LensDistortionCorrector
        >>> pipeline = ImagePipeline([
        ...     LensDistortionCorrector(),
        ... ])  # doctest: +SKIP
        >>> result = pipeline.operate(image)  # doctest: +SKIP
    """

    def _operate(self, image: Image) -> Image:
        """Remap all image components to correct geometric lens distortion.

        Args:
            image: Input image. May be Image or GridImage.

        Returns:
            Image with all components remapped to correct distortion.
        """
        import lensfunpy

        params = self._resolve_params(image)
        h, w = image.gray[:].shape[:2]
        mod = self._build_modifier(
            params, h, w, flags=lensfunpy.ModifyFlags.DISTORTION
        )

        undist_coords = mod.apply_geometry_distortion()
        if undist_coords is None:
            warnings.warn(
                "No distortion calibration data available for this lens. "
                "Image returned unchanged.",
                stacklevel=2,
            )
            return image

        # undist_coords shape: (h, w, 2) — (x, y) coordinate maps
        map_x = undist_coords[:, :, 0].astype(np.float32)
        map_y = undist_coords[:, :, 1].astype(np.float32)

        # Remap RGB if present
        if not image.rgb.isempty():
            image._data.rgb = cv2.remap(
                image._data.rgb, map_x, map_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

        # Remap gray
        image._data.gray = cv2.remap(
            image._data.gray.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(image._data.gray.dtype)

        # Remap detect_mat
        image._data.detect_mat = cv2.remap(
            image._data.detect_mat.astype(np.float32), map_x, map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(image._data.detect_mat.dtype)

        # Remap objmap: sparse → dense → remap (nearest) → sparse
        dense_objmap = image._data.sparse_object_map.toarray()
        if dense_objmap.any():
            remapped_objmap = cv2.remap(
                dense_objmap.astype(np.float32), map_x, map_y,
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ).astype(dense_objmap.dtype)
            image._data.sparse_object_map = csc_matrix(remapped_objmap)

        # GridImage preservation
        from phenotypic import GridImage

        if isinstance(image, GridImage):
            image = GridImage(
                arr=image,
                name=image.name,
                grid_finder=image.grid_finder,
                nrows=image.nrows,
                ncols=image.ncols,
                bit_depth=image.bit_depth,
                illuminant=image.illuminant,
                gamma=image.gamma,
            )

        return image
