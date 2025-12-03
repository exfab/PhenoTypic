import colour
import numpy as np

from ..accessor_abstracts import ColorSpaceAccessor


class CieLabAccessor(ColorSpaceAccessor):
    """Provides access to CIE L*a*b* color space representation.

    Converts XYZ color space data to perceptually uniform CIE L*a*b* coordinates
    using the Colour library's `XYZ_to_Lab` function. The CIE L*a*b* color space,
    defined by the International Commission on Illumination (CIE), is designed to
    be perceptually uniform, meaning that equal distances in the color space
    approximately correspond to equal perceptual color differences.

    The color space is represented as three channels:
    - L* (lightness): ranges from 0 (black) to 100 (white)
    - a* (green-red): negative values represent green, positive values represent red
    - b* (blue-yellow): negative values represent blue, positive values represent yellow

    The conversion is performed on-demand from the parent image's XYZ color space
    using the specified illuminant and observer angle. The accessor computes the
    Lab coordinates when accessed, rather than storing them permanently.

    Attributes:
        _accessor_property_name (str): Identifier for this accessor as "color.Lab",
            used for metadata tracking when saving color space arrays.

    Examples:
        .. dropdown:: Access Lab color space data from an image

            .. code-block:: python

                lab_array = image.color.Lab
                lightness = lab_array[:, :, 0]  # Extract L* channel
                green_red = lab_array[:, :, 1]  # Extract a* channel
                blue_yellow = lab_array[:, :, 2]  # Extract b* channel

        .. dropdown:: Save Lab data to a TIFF file

            .. code-block:: python

                image.color.Lab.imsave("output_lab.tif")

        .. dropdown:: Create a histogram of the L* channel

            .. code-block:: python

                fig, axes = image.color.Lab.histogram(channel_names=["L*", "a*", "b*"])
    """

    _accessor_property_name: str = "color.Lab"

    @property
    def _subject_arr(self) -> np.ndarray:
        """Compute CIE L*a*b* color space array from XYZ data.

        Converts the parent image's XYZ color space data to perceptually uniform
        CIE L*a*b* coordinates. The conversion uses the parent image's illuminant
        and standard observer settings for accurate color representation.

        Returns:
            np.ndarray: A 3D array with shape (height, width, 3) containing the
                L*a*b* color values. The three channels represent:
                - Channel 0: L* (lightness) in range [0, 100]
                - Channel 1: a* (green-red) typically in range [-128, 127]
                - Channel 2: b* (blue-yellow) typically in range [-128, 127]
                The dtype is typically float64, suitable for scientific computation.

        Notes:
            The conversion is computed on each access, not cached. For repeated
            access patterns, consider storing the result in a local variable.
            This design ensures consistency with the parent image's XYZ data.
        """
        return colour.XYZ_to_Lab(
            XYZ=self._root_image.color.XYZ[:],
            illuminant=colour.CCS_ILLUMINANTS[self._root_image._observer][
                self._root_image.illuminant
            ],
        )
