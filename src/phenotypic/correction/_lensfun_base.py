from __future__ import annotations

import importlib.util
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np

from ..abc_ import ImageCorrector

# Lazy import of lensfunpy
_LENSFUNPY_AVAILABLE = importlib.util.find_spec("lensfunpy") is not None


def _require_lensfunpy():
    """Raise ImportError with helpful message if lensfunpy is not installed."""
    if not _LENSFUNPY_AVAILABLE:
        raise ImportError(
            "lensfunpy is required for lens correction operations. "
            "Install it with: pip install phenotypic[lens]"
        )


class _LensfunCorrectorBase(ImageCorrector):
    """Private base class for lensfunpy-based lens correction operations.

    Provides shared camera/lens lookup logic and EXIF parameter resolution
    for LensDistortionCorrector, LensVignettingCorrector, and LensTCACorrector.

    Not exported publicly. Subclasses implement _operate() for specific
    correction types.

    Args:
        cam_maker: Camera manufacturer. Auto-detected from EXIF 'Image Make'.
        cam_model: Camera model name. Auto-detected from EXIF 'Image Model'.
        lens_maker: Lens manufacturer. Defaults to cam_maker if not specified.
        lens_model: Lens model name. Auto-detected from EXIF 'EXIF LensModel'.
        focal_length: Focal length in mm. Auto-detected from EXIF 'EXIF FocalLength'.
        aperture: Aperture f-number. Auto-detected from EXIF 'EXIF FNumber'.
        distance: Subject distance in meters. Defaults to 0.5 (typical plate distance).
    """

    def __init__(
        self,
        cam_maker: str | None = None,
        cam_model: str | None = None,
        lens_maker: str | None = None,
        lens_model: str | None = None,
        focal_length: float | None = None,
        aperture: float | None = None,
        distance: float = 0.5,
    ):
        _require_lensfunpy()
        self.cam_maker = cam_maker
        self.cam_model = cam_model
        self.lens_maker = lens_maker
        self.lens_model = lens_model
        self.focal_length = focal_length
        self.aperture = aperture
        self.distance = distance

    @staticmethod
    def _parse_exif_value(value: Any) -> float | None:
        """Parse EXIF rational strings or numeric types to float.

        Args:
            value: EXIF value — may be a string like ``"28"``, ``"14/5"``,
                a numeric type, or None.

        Returns:
            Parsed float value, or None if the value cannot be parsed.

        Examples:
            >>> _LensfunCorrectorBase._parse_exif_value("28")
            28.0
            >>> _LensfunCorrectorBase._parse_exif_value("14/5")
            2.8
            >>> _LensfunCorrectorBase._parse_exif_value(2.8)
            2.8
            >>> _LensfunCorrectorBase._parse_exif_value(None) is None
            True
        """
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.strip()
            if "/" in value:
                parts = value.split("/")
                try:
                    return float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    return None
            try:
                return float(value)
            except ValueError:
                return None
        # numpy numeric types
        if isinstance(value, np.generic):
            return float(value)
        return None

    def _resolve_params(self, image: Image) -> dict[str, Any]:
        """Merge user-provided params with EXIF metadata from image.

        User-provided values (set in ``__init__``) take priority. Missing values
        are filled from ``image.metadata`` using standard EXIF keys.

        Args:
            image: Image with metadata accessor containing EXIF tags.

        Returns:
            Dict with keys: cam_maker, cam_model, lens_maker, lens_model,
            focal_length, aperture, distance.

        Raises:
            ValueError: If any required parameter (cam_maker, cam_model,
                lens_model, focal_length, aperture) cannot be resolved from
                either user params or EXIF metadata.
        """
        meta = image.metadata

        cam_maker = self.cam_maker or meta.get("Image Make")
        cam_model = self.cam_model or meta.get("Image Model")
        lens_model = self.lens_model or meta.get("EXIF LensModel")
        lens_maker = self.lens_maker or cam_maker
        focal_length = self.focal_length or self._parse_exif_value(
            meta.get("EXIF FocalLength")
        )
        aperture = self.aperture or self._parse_exif_value(
            meta.get("EXIF FNumber")
        )

        # Validate required params
        missing = []
        if not cam_maker:
            missing.append("cam_maker (EXIF 'Image Make')")
        if not cam_model:
            missing.append("cam_model (EXIF 'Image Model')")
        if not lens_model:
            missing.append("lens_model (EXIF 'EXIF LensModel')")
        if focal_length is None:
            missing.append("focal_length (EXIF 'EXIF FocalLength')")
        if aperture is None:
            missing.append("aperture (EXIF 'EXIF FNumber')")

        if missing:
            raise ValueError(
                f"Cannot resolve lens parameters: {', '.join(missing)}. "
                "Provide them explicitly or use an image with EXIF metadata."
            )

        return {
            "cam_maker": str(cam_maker),
            "cam_model": str(cam_model),
            "lens_maker": str(lens_maker),
            "lens_model": str(lens_model),
            "focal_length": float(focal_length),
            "aperture": float(aperture),
            "distance": float(self.distance),
        }

    def _build_modifier(
        self, params: dict[str, Any], height: int, width: int, flags: int
    ) -> Any:
        """Look up camera/lens in lensfunpy database and create a Modifier.

        Args:
            params: Resolved parameter dict from ``_resolve_params``.
            height: Image height in pixels.
            width: Image width in pixels.
            flags: lensfunpy.ModifyFlags value for the correction type.

        Returns:
            Initialized ``lensfunpy.Modifier`` instance.

        Raises:
            ValueError: If camera or lens is not found in the lensfunpy database.
        """
        import lensfunpy

        db = lensfunpy.Database()

        # Look up camera
        cam = db.find_cameras(params["cam_maker"], params["cam_model"])
        if not cam:
            raise ValueError(
                f"Camera not found in lensfunpy database: "
                f"'{params['cam_maker']}' '{params['cam_model']}'. "
                f"Check spelling or use lensfunpy.Database().find_cameras() "
                f"to list available cameras."
            )
        cam = cam[0]

        # Look up lens
        lens = db.find_lenses(cam, params["lens_maker"], params["lens_model"])
        if not lens:
            raise ValueError(
                f"Lens not found in lensfunpy database: "
                f"'{params['lens_maker']}' '{params['lens_model']}'. "
                f"Check spelling or use lensfunpy.Database().find_lenses() "
                f"to list available lenses."
            )
        lens = lens[0]

        mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
        mod.initialize(
            params["focal_length"],
            params["aperture"],
            params["distance"],
            pixel_format=np.float64,
            flags=flags,
        )
        return mod
