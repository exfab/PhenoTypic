from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage
from phenotypic.abc_ import ImageOperation
from abc import ABC


class GridOperation(ImageOperation, ABC):
    """Abstract base class for operations on grid-aligned plate images.

    GridOperation is a marker abstract base class that enforces type safety for operations
    designed to work exclusively with GridImage objects. It's a lightweight subclass of
    ImageOperation that overrides the apply() method to require a GridImage input instead
    of a generic Image.

    **Quick Decision Guide**

    Use GridOperation when:
    - Your operation requires grid structure information (well positions, row/column layout)
    - You're processing arrayed plate images with regular grid layouts (96-well, 384-well)
    - Your algorithm needs per-well analysis or grid-aligned regions
    - You want to enforce that input must be GridImage (type safety)

    Use ImageOperation when:
    - Your operation works on general Image objects regardless of grid state
    - You're doing global preprocessing, detection, or measurement
    - Your algorithm doesn't depend on well structure or grid alignment
    - Your operation should accept both Image and GridImage inputs

    Combining GridOperation with ImageOperation:
    - GridOperation is typically paired with other ABCs (ObjectDetector, ImageCorrector, etc.)
    - Use multiple inheritance: class GridObjectDetector(ObjectDetector, GridOperation, ABC)
    - GridOperation adds type safety without changing algorithm implementation
    - Most grid operations inherit from both a specific ABC and GridOperation

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

    GridOperation provides three key benefits:

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

    **GridOperation Subclasses**

    Most concrete grid operations inherit from BOTH a specific operation ABC (like ObjectDetector)
    AND GridOperation to create specialized grid-aware variants:

    - ``GridObjectDetector``: Detects objects using
      grid structure. Subclasses implement well-level colony detection on gridded plates.
    - ``GridCorrector``: Corrects grid alignment, rotation,
      and per-well color correction. Improves grid positioning and well-level alignment.
    - ``GridObjectRefiner``: Refines detection masks at
      the well level. Filters and adjusts masks based on well location and size constraints.
    - ``GridMeasureFeatures``: Extracts per-well measurements.
      Computes features organized by grid coordinates rather than globally.
    - ``GridFinder``: Detects grid structure from object
      positions. Assigns detected objects to grid cells and determines well locations.

    **Multiple Inheritance Pattern**

    Most GridOperation subclasses use multiple inheritance to combine operation behavior with
    grid type safety:

    - Combine with ObjectDetector: class GridObjectDetector(ObjectDetector, GridOperation, ABC)
    - Combine with ImageCorrector: class GridCorrector(ImageCorrector, GridOperation, ABC)
    - Combine with any operation ABC: class CustomGridOp(SomeABC, GridOperation, ABC)

    The inheritance order matters: specific ABC first, then GridOperation.

    Example of multiple inheritance pattern:

        >>> from phenotypic.abc_ import ImageOperation, GridOperation
        >>> from phenotypic import GridImage, Image
        >>> # Concrete implementation combining ObjectDetector + GridOperation
        >>> # class GridObjectDetector(ObjectDetector, GridOperation, ABC):
        >>> #     def _operate(self, image: GridImage) -> GridImage:
        >>> #         # Implementation uses grid structure from image.grid
        >>> #         return image

    This combines:

    - **Operation behavior:** Sets image.objmask and image.objmap, with integrity checks.
    - **GridOperation type safety:** Requires GridImage input, enforced at runtime.
    - **ABC pattern:** Subclasses implement _operate() with grid-aware logic.

    The key insight: GridOperation is just a type annotation layer over ImageOperation that
    makes the grid requirement explicit in the method signature.

    **Type Safety Example**

    GridOperation enforces type checking at apply() time to catch errors early:

        >>> from phenotypic import Image, GridImage
        >>> from phenotypic.abc_ import GridOperation
        >>> # When a GridOperation is called with wrong type:
        >>> # detector = SomeGridOperation()  # subclass of GridOperation
        >>> # result = detector.apply(Image('plain.jpg'))  # Raises GridImageInputError
        >>> # result = detector.apply(GridImage('plate.jpg', nrows=8, ncols=12))  # OK

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

    Notes:
        - GridOperation is a marker class with no implementation. It only overrides apply()
          to specify the GridImage type and enforce input validation.

        - GridImage inherits all Image functionality. Grid information is accessed via
          the ``grid`` accessor: ``image.grid.nrows``, ``image.grid.ncols``, etc.

        - If you're unsure whether your operation needs GridOperation, ask: "Does this
          algorithm fundamentally depend on grid structure?" If yes, use GridOperation.
          If it works equally well on plain Images, use ImageOperation.

        - GridImage is typically created with GridFinder operations that detect grid structure.
          GridFinder detects grid positions and creates a GridImage suitable for downstream
          GridOperation subclasses.

    Examples:
        Using a GridOperation subclass with GridImage:

        >>> from phenotypic import GridImage
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.detect import GridObjectDetector
        >>> # Load plate image with grid info
        >>> image = load_synth_yeast_plate()  # GridImage with detected colonies
        >>> grid_image = image
        >>> # Apply a grid-aware detector (subclass of GridObjectDetector)
        >>> # GridImage is required - type-safe operation
        >>> # detector = GridObjectDetector()  # Concrete subclass in practice
        >>> # detected = detector.apply(grid_image)

        Type safety: GridOperation prevents misuse:

        >>> from phenotypic import Image, GridImage
        >>> from phenotypic.enhance import GaussianBlur
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = Image('generic.jpg')  # Plain Image
        >>> grid_image = load_synth_yeast_plate()  # GridImage
        >>> # ImageOperation (GaussianBlur) accepts both
        >>> enhancer = GaussianBlur(sigma=2)
        >>> result1 = enhancer.apply(image)       # OK: Image -> Image
        >>> result2 = enhancer.apply(grid_image)  # OK: GridImage -> GridImage
        >>> # GridOperation requires GridImage only
        >>> # detector = SomeGridOperation()  # subclass of GridOperation
        >>> # result3 = detector.apply(grid_image)  # OK: GridImage -> GridImage
        >>> # result4 = detector.apply(image)  # ERROR: raises GridImageInputError
    """

    def apply(self, image: GridImage, inplace: bool = False) -> GridImage:
        return super().apply(image=image, inplace=inplace)
