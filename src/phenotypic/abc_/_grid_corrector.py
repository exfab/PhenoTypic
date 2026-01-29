from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import GridImage

from phenotypic.abc_ import ImageCorrector
from phenotypic.abc_ import GridOperation
from phenotypic.tools_.exceptions_ import GridImageInputError, OutputValueError
from abc import ABC


class GridCorrector(ImageCorrector, GridOperation, ABC):
    """Apply whole-image transformations (rotation, alignment, perspective) to GridImage objects.

    GridCorrector is a type-safe wrapper around ImageCorrector that enforces GridImage input
    and output types. It is specialized for grid-aware image corrections on arrayed plate images.

    **Quick Decision Guide**

    Choose GridCorrector vs ImageCorrector:

    - **GridCorrector:** Transformation modifies well structure or assumes grid layout. Examples: align
      colonies to grid axes, per-well perspective correction, rotation with grid-aware interpolation.
    - **ImageCorrector:** Transformation works on any Image. Examples: general rotation, perspective
      correction on ungridded images, generic image transformations without grid awareness.
    - **Grid-aware operations:** Use GridCorrector when grid rows/columns matter for the transformation
      or when the transformation affects well-level alignment and downstream analysis.
    - **Type safety:** GridCorrector enforces GridImage input/output; ImageCorrector accepts plain Image.
    - **Implementation complexity:** GridCorrector typically requires understanding grid.rotation_angle,
      grid.alignment, and per-well coordinate mapping via image.grid.info().
    - **Output consistency:** GridCorrector guarantees all components rotate identically. ImageCorrector
      gives you flexibility but requires manual synchronization of all components.
    - **Grid preservation:** GridCorrector may update grid state after transformation. ImageCorrector
      leaves grid structure unchanged (if present).

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

    **Typical Use Cases**

    - **Grid alignment:** Rotate the entire image so detected colonies align with grid rows and columns.
      Improves downstream grid-based analysis via [GridAligner](src/phenotypic/correction/_grid_aligner.py).
    - **Perspective correction:** Correct camera tilt or lens distortion that skews the grid layout.
    - **Plate reorientation:** Rotate plate image to canonical orientation for consistent well assignment.
    - **Color calibration per well:** Apply per-well color correction that respects grid well boundaries.

    **Implementation Pattern**

    Inherit from GridCorrector and implement ``_operate()`` as an instance method:

    .. code-block:: python

        from phenotypic.abc_ import GridCorrector
        from phenotypic import GridImage

        class GridAligner(GridCorrector):
            '''Rotate GridImage to align colonies with grid rows/columns.'''

            def __init__(self, axis: int = 0, max_rotation: float = 45.0):
                super().__init__()
                self.axis = axis
                self.max_rotation = max_rotation

            def _operate(self, image: GridImage) -> GridImage:
                # image is guaranteed to be GridImage with valid grid structure
                # Access grid structure and compute needed transformation
                grid_info = image.grid.info()
                nrows, ncols = grid_info['grid_shape']

                # Calculate rotation needed to align colonies with grid axes
                rotation_angle = self._calculate_grid_rotation(image, self.axis)

                # Clamp rotation to reasonable range to prevent over-correction
                if abs(rotation_angle) > self.max_rotation:
                    rotation_angle = self.max_rotation * (1 if rotation_angle > 0 else -1)

                # Apply rotation to all image components automatically
                image.rotate(angle_of_rotation=rotation_angle, mode='edge')

                # Grid structure (rows/cols) unchanged; rotation_angle updated automatically
                return image

    **Critical Implementation Details**

    Ensure ALL image components are transformed identically:

    - **Transformation synchronization:** When you rotate/warp rgb, also rotate gray, enh_gray, objmask, objmap.
      Use image.rotate() or similar methods that handle this automatically. Failure to synchronize causes
      misalignment between visual and label data.
    - **Coordinate system consistency:** Grid coordinates (well centers, row/column positions) must match
      the transformed pixel coordinates. Update grid.rotation_angle and grid.alignment after transformation
      so downstream operations use correct well boundaries.
    - **Grid state preservation:** Maintain grid.rows, grid.cols, and grid.well_size unchanged unless
      explicitly needed (e.g., perspective correction may change well size). Update only rotation_angle
      and alignment for simple rotations.
    - **Interpolation order:** Use order=1+ for color data (smooth), order=0 for labels (preserve integers).
      Mixed interpolation on different components is OK and expected.
    - **Edge handling:** Define how image boundaries are handled during transformation (reflect, wrap, constant).
      Choose mode='edge' or mode='constant' depending on plate structure and whether border wells matter.
    - **In-place vs return:** GridCorrector operations typically modify image in-place AND return it. Follow
      this pattern for consistency with parent ImageCorrector class.

    **Interpolation Considerations**

    When rotating or warping:

    - **Color data (rgb, gray):** Use smooth interpolation (order=1+) to preserve colony edges
    - **Detection data (objmask, objmap):** Use nearest-neighbor interpolation (order=0) to
      preserve discrete object labels (must remain integers)
    - **Enhanced grayscale:** Use same interpolation as color data for consistency

    **Common Transformations**

    - **Rotation:** Most common GridCorrector operation. Rotate entire plate to align colonies with grid axes.
      Use image.rotate(angle_of_rotation, mode='edge') to handle all components automatically.
    - **Perspective correction:** Correct camera tilt or lens distortion. More complex; requires affine or
      homography transformation on all components with careful interpolation.
    - **Scale and crop:** Resize plate image while maintaining grid structure. Update grid.well_size accordingly.
    - **Flip/transpose:** Flip or transpose plate (e.g., for reorientation). Update grid rows/cols and rotation_angle.

    **Best Practices**

    - Always verify grid structure is valid before applying GridCorrector (check grid.rows, grid.cols > 0).
    - Test transformation on synthetic data first; grid misalignment can cascade through entire pipeline.
    - Update grid metadata (rotation_angle, alignment) immediately after transformation for consistency.
    - Log the transformation (angle, parameters) for reproducibility and debugging.
    - Consider edge well effects (evaporation, contamination); some corrections may need well-specific logic.

    **Notes**

    - GridCorrector has no integrity checks (@validate_operation_integrity), by design.
      All components are intentionally modified together; there is nothing to validate.
    - Grid rotation angle and alignment state may be updated after the transformation.
      Downstream grid-aware operations will work with the updated grid structure.
    - GridImage must have valid grid structure before correction. Use GridFinder or specify
      grid manually before applying GridCorrector.
    - Output is always GridImage (type-safe). Attempting to apply to plain Image raises error.
    - Coordinate system: Grid rows/cols are logical indices; transformation affects pixel coordinates only.
      Update rotation_angle to track cumulative transformations.

    **Relationship to GridFinder and GridOperation**

    - **GridFinder:** Detects grid structure automatically. Use BEFORE GridCorrector to establish rows/cols.
    - **GridCorrector:** Adjusts/aligns detected grid. Use AFTER GridFinder to optimize grid alignment.
    - **GridOperation:** Base class for all grid-aware operations. GridCorrector is a specialized subclass
      that combines ImageCorrector functionality with grid awareness.
    - **Pipeline integration:** Typical order: detect_grid (GridFinder) → correct_grid (GridCorrector) →
      measure/analyze (GridMeasureFeatures).

    **Image Synchronization Details**

    When implementing ``_operate()``, ensure these components stay in sync:

    - **rgb & gray:** Always rotate together. Gray is derived from rgb, so maintain pixel correspondence.
    - **enh_gray:** Enhanced grayscale processed from gray. Must rotate with same angle/transformation.
    - **objmask & objmap:** Detection results. Must use SAME interpolation as rgb to maintain object alignment.
    - **Grid metadata:** Update grid.rotation_angle if rotation applied. Keep grid.rows/cols unchanged unless
      grid structure itself changes.

    **Example Coordinate Transformation**

    For a 96-well plate rotated by θ degrees:

    - Original well positions: grid.info()['well_centers'] in absolute pixel coordinates
    - After rotation by θ: well_centers shift to new pixel positions
    - Update grid.rotation_angle += θ
    - Grid rows/cols (logical structure) remain unchanged (8 rows × 12 cols)

    **Known Implementations**

    Reference implementations in the PhenoTypic framework:

    - **[GridAligner](src/phenotypic/correction/_grid_aligner.py):** Rotates entire image to align detected
      colonies with grid rows and columns. Uses Hough transform to detect dominant angles in colony positions.
    - **ImageRotation (ImageCorrector):** Simple rotation without grid awareness. Baseline for understanding
      how GridCorrector extends basic functionality with grid metadata updates.
    - Custom implementations: Users can subclass GridCorrector for domain-specific plate corrections
      (e.g., multi-stage perspective correction, plate-specific calibrations).

    **Testing GridCorrector Implementations**

    Best practices for testing new GridCorrector subclasses:

    - Use ``load_synth_yeast_plate()`` from phenotypic.data (creates GridImage with synthetic colonies).
    - Verify all components (rgb, gray, enh_gray, objmask, objmap) rotate identically by computing pixel
      differences before/after transformation.
    - Check that grid.rotation_angle is updated correctly and accumulated rotations are tracked.
    - Validate on multiple plate formats (96-well, 384-well) to ensure well positions are handled correctly.

    Examples:
        GridAligner: rotate to align colonies with grid axes:

        >>> from phenotypic import GridImage, Image
        >>> from phenotypic.detect import RoundPeaksDetector
        >>> from phenotypic.correction import GridAligner
        >>> # Load and detect colonies
        >>> image = Image('plate.jpg')
        >>> image = RoundPeaksDetector().operate(image)
        >>> # Create GridImage with grid structure
        >>> grid_image = GridImage(image)
        >>> grid_image.detect_grid()
        >>> # Align entire image to grid rows/columns
        >>> aligner = GridAligner(axis=0)  # Align rows horizontally
        >>> aligned = aligner.apply(grid_image)
        >>> # All components (rgb, gray, masks, map) rotated together
        >>> # Grid structure updated to reflect rotation
        >>> print(f"Rotation angle: {aligned.grid.rotation_angle}")

        Custom perspective correction (conceptual):

        >>> from phenotypic.abc_ import GridCorrector
        >>> from phenotypic import GridImage
        >>> class GridPerspectiveCorrector(GridCorrector):
        ...     '''Correct camera tilt or lens distortion on grid plate.'''
        ...
        ...     def __init__(self, tilt_angle: float):
        ...         super().__init__()
        ...         self.tilt_angle = tilt_angle
        ...
        ...     def _operate(self, image: GridImage) -> GridImage:
        ...         # Apply perspective transform to all components
        ...         # Update grid coordinates accordingly
        ...         grid_info = image.grid.info()
        ...         image.apply_perspective(self.tilt_angle)
        ...         # Re-validate grid after transformation
        ...         return image
        >>> # Usage: correct skewed plate image
        >>> corrector = GridPerspectiveCorrector(tilt_angle=10.0)
        >>> corrected = corrector.apply(grid_image)
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
