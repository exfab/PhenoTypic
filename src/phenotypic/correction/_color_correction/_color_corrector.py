"""Color corrector applying a fitted ColorCheckerProfile to images.

Implements the root-polynomial colour correction (Finlayson 2015) as an
:class:`~phenotypic.abc_.ImageCorrector` operation, transforming RGB, gray,
and detect_mat in a single pass.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, overload

import colour
import numpy as np
from pydantic import PrivateAttr, field_validator
from skimage.color import rgb2gray

from ...abc_ import ImageCorrector
from ._capture_metadata import CaptureMetadata
from ._color_checker_profile import ColorCheckerProfile

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage
    from phenotypic._core._image import Image


class ColorCorrector(ImageCorrector):
    """Apply root-polynomial color correction using a fitted ColorCheckerProfile.

    Decodes sRGB gamma to linear light, applies the root-polynomial expansion
    and the profile's correction matrix, re-encodes to sRGB, and recomputes
    grayscale and detect_mat from the corrected RGB. The correction matrix is
    derived from a color checker captured under the same imaging conditions, so
    colony color becomes comparable across batches and acquisition setups.

    For algorithm details, see :doc:`/explanation/color_spaces_for_phenotyping`.

    Best For:
        - Multi-batch studies imaged across sessions or scanners where colony
          color must be cross-batch comparable.
        - Plates with pigmented yeast or bacterial colonies where chromatic
          phenotype is the primary readout.
        - Publication workflows needing perceptually accurate color
          reproduction.
        - Removing color casts from DSLR or flatbed-scanner illuminants that
          shift colony hue relative to the agar background.

    Consider Also:
        - :class:`FlattenIllumination` when the artifact is a spatial
          brightness gradient rather than a chromatic cast.
        - :class:`ColorDenoise` when high-ISO color noise obscures colony
          boundaries without a systematic color-cast problem.
        - :class:`BayesShrinkCorrector` or :class:`VisuShrinkCorrector` when
          the goal is noise reduction rather than color accuracy.

    Args:
        profile: A fitted :class:`ColorCheckerProfile` supplying the
            root-polynomial correction matrix and expansion degree. An
            unfitted profile is rejected with a ``ValueError``.
        output_illuminant: Target illuminant label used when reporting the
            corrected color space (informational only; does not alter the
            matrix computation). Default: ``"D65"``.

    Returns:
        Image: ``rgb`` mapped through the correction matrix; ``gray`` and
        ``detect_mat`` recomputed from the corrected RGB. All three components
        are updated in a single pass.

    Raises:
        ValueError: If ``profile`` is not yet fitted.
        UserWarning: If the image's camera EXIF indicates a different camera
            body or lens than the one used when the profile was fitted.

    See Also:
        :doc:`/tutorials/notebooks/01_your_first_plate_image` for an
        introduction to loading and correcting full-plate images.
        :doc:`/explanation/color_spaces_for_phenotyping` for a discussion of
        root-polynomial color correction in phenotyping.
    """

    profile: ColorCheckerProfile
    output_illuminant: str = "D65"

    #: Float64 view of the profile's correction matrix, materialised once
    #: in :meth:`model_post_init` and read by :meth:`_operate`.
    _ccm: np.ndarray = PrivateAttr()

    @field_validator("profile")
    @classmethod
    def _require_fitted_profile(
        cls, profile: ColorCheckerProfile
    ) -> ColorCheckerProfile:
        """Reject an unfitted profile (pre-migration ``__init__`` guard)."""
        if not profile.is_fitted:
            raise ValueError("ColorCheckerProfile must be fitted before use.")
        return profile

    def model_post_init(self, __context: Any) -> None:
        """Materialise the float64 correction matrix after construction.

        Extends :meth:`BaseOperation.model_post_init` (which sets up the
        logger): caches a ``float64`` copy of the fitted profile's
        correction matrix so :meth:`_operate` does not rebuild it per call.

        Args:
            __context: Pydantic post-init context (unused).
        """
        super().model_post_init(__context)
        self._ccm = np.asarray(self.profile.correction_matrix, dtype=np.float64)

    @property
    def correction_matrix(self) -> list:
        """The root-polynomial correction matrix as a nested list."""
        return self._ccm.tolist()

    @property
    def degree(self) -> int:
        """Polynomial expansion degree, matching the profile."""
        return self.profile.degree

    @overload
    def apply(self, image: GridImage, inplace: bool = False) -> GridImage: ...

    @overload
    def apply(self, image: Image, inplace: bool = False) -> Image: ...

    def apply(self, image: Image, inplace: bool = False) -> Image:
        """Apply the correction, first checking capture-metadata compatibility.

        The compatibility check runs against the *input* image (before the
        defensive copy ``apply`` makes for ``inplace=False``), because
        :meth:`Image.copy` does not carry imported EXIF forward.

        Args:
            image: Image to correct.
            inplace: When ``False`` (default) operate on a copy.

        Returns:
            The corrected image.
        """
        self._warn_on_metadata_mismatch(image)
        return super().apply(image, inplace=inplace)

    def _warn_on_metadata_mismatch(self, image: Image) -> None:
        """Warn when *image* was captured on different optics than the profile.

        Compares the image's camera EXIF against the
        :attr:`~ColorCheckerProfile.capture_metadata` recorded when the profile
        was fitted.  Camera-body or lens differences raise a
        :class:`UserWarning` (the correction may be invalid); exposure-setting
        differences (ISO, exposure time, F-number, focal length) are only
        logged at info level.  Does nothing when the profile carries no capture
        metadata (e.g. an old serialised profile or one fitted from
        pre-measured patch colours).

        Args:
            image: The image about to be corrected.
        """
        expected = self.profile.capture_metadata
        if expected is None:
            return
        actual = CaptureMetadata.from_image(image)
        critical, informational = expected.compare(actual)
        if critical:
            warnings.warn(
                f"Image '{image.name}' was captured on a different "
                f"camera/lens than the colour-correction profile "
                f"({'; '.join(critical)}). The correction may be invalid.",
                UserWarning,
                stacklevel=3,  # _warn_on_metadata_mismatch -> apply -> caller
            )
        if informational:
            self._logger.info(
                "Capture-setting differences for '%s': %s",
                image.name,
                "; ".join(informational),
            )

    def _operate(self, image: Image) -> Image:
        """Apply root-polynomial colour correction to the image.

        Args:
            image: Input image to correct.

        Returns:
            Image with corrected RGB, gray, and detect_mat.
        """
        ccm = self._ccm
        rgb_raw = image.rgb[:]
        original_dtype = rgb_raw.dtype

        # 1. Normalise to [0, 1] float.
        if np.issubdtype(original_dtype, np.integer):
            max_val = float(np.iinfo(original_dtype).max)
        else:
            max_val = 1.0
        rgb_normed = rgb_raw.astype(np.float64) / max_val

        # 2. Decode sRGB gamma to linear light.
        H, W, C = rgb_normed.shape
        rgb_linear = colour.cctf_decoding(
            rgb_normed.reshape(-1, 3), function="sRGB"
        )

        # 3-4. Build root-polynomial features and apply correction matrix.
        corrected_linear = (
            colour.characterisation.apply_matrix_colour_correction_Finlayson2015(
                rgb_linear,
                ccm,
                degree=self.degree,  # type: ignore[arg-type]
                root_polynomial_expansion=True,
            )
        )
        corrected_linear = np.clip(corrected_linear, 0.0, None)

        # 5. Re-encode to sRGB gamma.
        corrected_srgb = colour.cctf_encoding(corrected_linear, function="sRGB")
        corrected_srgb = np.clip(  # type: ignore[assignment]
            corrected_srgb, 0.0, 1.0
        ).reshape(H, W, C)

        # 6. Scale back to original dtype.
        if np.issubdtype(original_dtype, np.integer):
            corrected_int = (corrected_srgb * max_val + 0.5).astype(original_dtype)
        else:
            corrected_int = corrected_srgb.astype(original_dtype)

        # 7. Update image data.
        image._data.rgb = corrected_int

        # 8. Recompute grayscale from corrected sRGB.
        corrected_gray = rgb2gray(corrected_srgb).astype(np.float64)
        image._data.gray = corrected_gray

        # 9. Update detect_mat to match corrected grayscale.
        image._data.detect_mat = corrected_gray.copy()

        return image

    def dashboard(self, show: bool = True) -> Any:
        """Build the Plotly color-correction diagnostic report.

        Delegates to the underlying profile's :meth:`ColorCheckerProfile.dashboard`.

        Args:
            show: When ``True`` and running inside a Jupyter notebook, display
                the figure inline before returning it.

        Returns:
            A ``plotly.graph_objects.Figure`` (always returned, even after an
            inline display).
        """
        return self.profile.dashboard(show=show)
