from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from phenotypic import GridImage

from phenotypic.abc_ import ImageCorrector
from phenotypic.abc_ import GridOperation
from phenotypic.tools.exceptions_ import GridImageInputError, OutputValueError
from abc import ABC


class GridCorrector(ImageCorrector, GridOperation, ABC):
    """Apply whole-image transformations (rotation, alignment, perspective) to GridImage objects.

    GridCorrector is a type-safe wrapper around ImageCorrector that enforces GridImage input
    and output types. It is specialized for grid-aware image corrections on arrayed plate images.

    **Purpose**

    Use GridCorrector when implementing transformations that modify entire GridImage objects
    while respecting their grid structure. Like ImageCorrector, it updates all image components
    (rgb, gray, enh_gray, objmask, objmap) together to maintain synchronization. The difference
    is that it requires GridImage input and output, making explicit that your transformation
    works in the context of grid-structured plate images.

    **What GridCorrector modifies**

    GridCorrector operations modify ALL image components simultaneously:

    - **Color data:** rgb, gray (pixel coordinates change due to rotation/perspective)
    - **Preprocessed data:** enh_gray (enhanced grayscale also rotates/transforms)
    - **Detection results:** objmask, objmap (colony masks and labels transform identically)
    - **Grid structure:** Grid rotation angle and alignment state (optional, depends on operation)

    This ensures that a rotated colony mask aligns perfectly with the rotated rgb and gray data.

    **GridImage vs Image**

    - **Image:** Generic image with optional, unvalidated grid information.
    - **GridImage:** Specialized Image subclass with validated grid structure (row/column
      layout, well positions, grid alignment angle). Typically used after GridFinder detects
      the grid structure.

    **When to use GridCorrector vs ImageCorrector**

    - **ImageCorrector:** Transformation works on any Image. Examples: rotation, perspective
      correction for individual (non-gridded) images. Use when grid structure is irrelevant.

    - **GridCorrector:** Transformation assumes or modifies grid structure. Examples: aligning
      colonies to grid rows/columns, rotating to match grid axes, per-well perspective correction.
      Use when the transformation is grid-aware or affects well-level alignment.

    **Typical Use Cases**

    - **Grid alignment:** Rotate the entire image so detected colonies align with grid rows
      and columns. Improves downstream grid-based analysis. Example: GridAligner rotates
      to make colony rows parallel to image axes.
    - **Perspective correction:** Correct camera tilt or lens distortion that skews the grid.
    - **Plate reorientation:** Rotate plate image to canonical orientation for consistent analysis.
    - **Color calibration per well:** Apply per-well color correction that respects grid boundaries.

    **Implementation Pattern**

    Inherit from GridCorrector and implement ``_operate()`` as normal:

    .. code-block:: python

        from phenotypic.abc_ import GridCorrector
        from phenotypic import GridImage

        class GridAligner(GridCorrector):
            '''Rotate GridImage to align colonies with grid rows/columns.'''

            def __init__(self, axis: int = 0):
                super().__init__()
                self.axis = axis

            @staticmethod
            def _operate(image: GridImage, axis: int = 0) -> GridImage:
                # image is guaranteed to be GridImage
                # Rotate all components together
                rotation_angle = calculate_grid_rotation(image, axis)
                image.rotate(angle_of_rotation=rotation_angle, mode='edge')
                return image

    **Critical Implementation Detail**

    Ensure ALL image components are transformed identically:

    .. code-block:: python

        @staticmethod
        def _operate(image: GridImage, **kwargs) -> GridImage:
            # Apply transformation to rgb/gray
            angle = kwargs.get('angle', 0)
            image.rotate(angle_of_rotation=angle, mode='edge')

            # The image.rotate() method automatically handles:
            # - Rotating enh_gray identically
            # - Rotating objmask and objmap with same angle
            # - Updating grid rotation state if applicable

            return image

    **Interpolation Considerations**

    When rotating or warping:

    - **Color data (rgb, gray):** Use smooth interpolation (order=1+) to preserve colony edges
    - **Detection data (objmask, objmap):** Use nearest-neighbor interpolation (order=0) to
      preserve discrete object labels (must remain integers)
    - **Enhanced grayscale:** Use same interpolation as color data for consistency

    **Notes**

    - GridCorrector has no integrity checks (@validate_operation_integrity), by design.
      All components are intentionally modified together; there is nothing to validate.
    - Grid rotation angle and alignment state may be updated after the transformation.
      Downstream grid-aware operations will work with the updated grid structure.
    - GridImage must have valid grid structure before correction. Use GridFinder or specify
      grid manually before applying GridCorrector.
    - Output is always GridImage (type-safe). Attempting to apply to plain Image raises error.

    Examples:
        .. dropdown:: GridAligner: rotate to align colonies with grid axes

            .. code-block:: python

                from phenotypic import GridImage, Image
                from phenotypic.detect import RoundPeaksDetector
                from phenotypic.correction import GridAligner

                # Load and detect colonies
                image = Image.from_image_path('plate.jpg')
                image = RoundPeaksDetector().operate(image)

                # Create GridImage with grid structure
                grid_image = GridImage(image)
                grid_image.detect_grid()

                # Align entire image to grid rows/columns
                aligner = GridAligner(axis=0)  # Align rows horizontally
                aligned = aligner.apply(grid_image)

                # All components (rgb, gray, masks, map) rotated together
                # Grid structure updated to reflect rotation
                print(f\"Rotation angle: {aligned.grid.rotation_angle}\")

        .. dropdown:: Custom perspective correction (conceptual)

            .. code-block:: python

                from phenotypic.abc_ import GridCorrector
                from phenotypic import GridImage

                class GridPerspectiveCorrector(GridCorrector):
                    \"\"\"Correct camera tilt or lens distortion on grid plate.\"\"\"

                    def __init__(self, tilt_angle: float):
                        super().__init__()
                        self.tilt_angle = tilt_angle

                    @staticmethod
                    def _operate(image: GridImage, tilt_angle: float) -> GridImage:
                        # Apply perspective transform to all components
                        # (Implementation depends on specific correction needed)
                        # image.apply_perspective(...) or similar
                        return image

                # Usage: correct skewed plate image
                corrector = GridPerspectiveCorrector(tilt_angle=10.0)
                corrected = corrector.apply(grid_image)
    """

    # Do not modify this method in inherited functions
    def apply(self, image: GridImage, inplace=False) -> GridImage:
        from phenotypic import GridImage

        if not isinstance(image, GridImage):
            raise GridImageInputError
        output = super().apply(image, inplace=inplace)
        if not isinstance(output, GridImage):
            raise OutputValueError("GridImage")
        return output
