from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import GridImage
from phenotypic.abc_ import ImageOperation
from abc import ABC


class GridOperation(ImageOperation, ABC):
    """Abstract base class for operations on grid-aligned plate images.

    GridOperation is a marker abstract base class that enforces type safety for operations
    designed to work exclusively with GridImage objects. It's a lightweight subclass of
    ImageOperation that overrides the apply() method to require a GridImage input instead
    of a generic Image.

    **What is GridOperation?**

    GridOperation exists to distinguish between two categories of image operations:

    - **ImageOperation:** Works on single, unaligned Image objects. The image may or may not
      have grid information. Used for general-purpose preprocessing, detection, and measurement.
      Examples: GaussianBlur, OtsuDetector, MeasureColorComposition.

    - **GridOperation:** Works only on GridImage objects that have grid structure information
      (row/column layout of wells on an agar plate). The operation assumes grid information
      is present and available. Used for grid-aware operations where well-level analysis or
      grid alignment is required. Examples: GridObjectDetector, GridCorrector, GridRefiner.

    **Why GridOperation exists**

    GridOperation provides three benefits:

    1. **Type Safety:** The apply() method signature requires a GridImage argument, catching
       misuse at runtime if someone tries to apply a grid operation to a plain Image.

    2. **Intent Clarity:** Developers can immediately see which operations require grid information,
       making the design space clear: "Use ImageOperation for general image ops, GridOperation
       for plate-specific grid-aware ops."

    3. **Documentation:** Allows documentation and tutorials to clearly distinguish operations
       by their input type requirements.

    **What is GridImage?**

    GridImage is a specialized Image subclass that adds grid structure information:

    - **Inherits from Image:** All standard image capabilities (RGB, grayscale, color spaces,
      object detection results, etc.) are available.

    - **Adds grid field:** Contains a ``grid`` attribute (GridInfo object) storing the detected
      or specified grid layout (row/column positions, cell dimensions, rotation angle).

    - **Arrayed plate context:** Represents images of agar plates with samples arranged in
      regular grids (96-well, 384-well, 1536-well formats). Typical nrows=8, ncols=12 for
      96-well plates.

    - **Grid accessors:** Via ``image.grid``, provides row/column counts, well positions, and
      grid-related metadata.

    **GridOperation subclasses**

    Most concrete grid operations inherit from BOTH a specific operation ABC (like ObjectDetector)
    AND GridOperation to create specialized grid-aware variants:

    .. code-block:: text

        GridOperation (marker ABC)
        ├── GridObjectDetector (inherits ObjectDetector + GridOperation)
        │   ├── GridInstanceDetector, GridThresholdDetector, GridCannyDetector, ...
        │   └── Use for: well-level colony detection on gridded plates
        │
        ├── GridCorrector (inherits ImageCorrector + GridOperation)
        │   ├── GridAligner, ...
        │   └── Use for: grid alignment, rotation, color correction per-well
        │
        └── GridObjectRefiner (inherits ObjectRefiner + GridOperation)
            ├── GridSizeRefiner, ...
            └── Use for: per-well mask refinement, filtering by well location

    **When to use GridOperation vs ImageOperation**

    - **ImageOperation:** Input is a plain Image with unknown grid state.
      Typical use: preprocessing (blur, contrast), general-purpose detection,
      color measurements that don't depend on grid layout.

    - **GridOperation:** Input is a GridImage with detected/specified grid structure.
      Typical use: well-level analysis, grid-based refinement, operations that
      reference well positions or grid-aligned regions.

    - **Overlap:** Some operations work on both. E.g., a ColorComposition measurement
      can apply to an Image, but a GridColorComposition can specialize to per-well
      measurements on a GridImage.

    **When to subclass GridOperation**

    Subclass GridOperation when your operation:

    1. **Requires grid information:** Needs to access ``image.grid`` to get well positions,
       row/column structure, or grid-aligned regions.

    2. **Operates on well-level data:** Processes colonies at the well level rather than
       globally on the image (e.g., per-well filtering, well-based alignment).

    3. **Makes assumptions about grid structure:** Your algorithm assumes a regular grid layout
       and would fail or produce nonsensical results on an image without grid info.

    Otherwise, subclass ImageOperation instead. GridOperation operations are more specialized
    and less broadly applicable.

    **Multiple inheritance pattern**

    Most GridOperation subclasses use multiple inheritance:

    .. code-block:: python

        class GridObjectDetector(ObjectDetector, GridOperation, ABC):
            '''Detects objects using grid structure.'''
            def apply(self, image: GridImage, inplace=False) -> GridImage:
                if not isinstance(image, GridImage):
                    raise GridImageInputError
                return super().apply(image=image, inplace=inplace)

    This combines:

    - **ObjectDetector behavior:** Sets image.objmask and image.objmap, with integrity checks.
    - **GridOperation type safety:** Requires GridImage input, enforced at runtime.
    - **ABC pattern:** Subclasses implement _operate() with grid-aware logic.

    The key insight: GridOperation is just a type annotation layer over ImageOperation that
    makes the grid requirement explicit in the method signature.

    Notes:
        - GridOperation is a marker class with no implementation. It only overrides apply()
          to specify the GridImage type and enforce input validation.

        - GridImage inherits all Image functionality. Grid information is accessed via
          the ``grid`` accessor: ``image.grid.nrows``, ``image.grid.ncols``, etc.

        - If you're unsure whether your operation needs GridOperation, ask: "Does this
          algorithm fundamentally depend on grid structure?" If yes, use GridOperation.
          If it works equally well on plain Images, use ImageOperation.

        - GridImage is typically created with ImageGridHandler or GridFinder operations
          that detect grid structure. GridFinder is an ImageOperation, but the result
          is a GridImage suitable for downstream GridOperation subclasses.

    Examples:
        .. dropdown:: Using a GridOperation subclass

            .. code-block:: python

                from phenotypic import GridImage
                from phenotypic.detect import GridObjectDetector

                # Load plate image (96-well)
                grid_image = GridImage('plate_scan.jpg', nrows=8, ncols=12)

                # Apply a grid-aware detector (subclass of GridObjectDetector)
                # This operation requires GridImage and uses well structure
                detector = GridObjectDetector()  # Concrete subclass
                detected = detector.apply(grid_image)  # Type-safe: GridImage -> GridImage

                # Access detected colonies per well
                for well_row in range(grid_image.nrows):
                    for well_col in range(grid_image.ncols):
                        # Per-well analysis available because operation is grid-aware
                        pass

        .. dropdown:: Understanding the type safety benefit

            .. code-block:: python

                from phenotypic import Image, GridImage
                from phenotypic.enhance import GaussianBlur
                from phenotypic.detect import GridObjectDetector

                image = Image.from_image_path('generic.jpg')  # Plain Image
                grid_image = GridImage('plate.jpg')           # GridImage

                # ImageOperation (GaussianBlur) accepts both
                enhancer = GaussianBlur(sigma=2)
                result1 = enhancer.apply(image)       # OK: Image -> Image
                result2 = enhancer.apply(grid_image)  # OK: GridImage -> GridImage

                # GridOperation requires GridImage
                detector = GridObjectDetector()  # Subclass of GridOperation
                result3 = detector.apply(grid_image)  # OK: GridImage -> GridImage
                # result4 = detector.apply(image)  # ERROR: raises GridImageInputError
    """

    def apply(self, image: GridImage, inplace: bool = False) -> GridImage:
        return super().apply(image=image, inplace=inplace)
