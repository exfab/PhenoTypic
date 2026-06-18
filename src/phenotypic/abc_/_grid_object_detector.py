from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

from phenotypic.abc_ import ObjectDetector, GridOperation
from phenotypic.sdk_.funcs_ import validate_operation_integrity
from phenotypic.sdk_.exceptions_ import GridImageInputError


class GridObjectDetector(ObjectDetector, GridOperation, ABC):
    """Detect and label colonies in GridImage objects using grid structure.

    GridObjectDetector is a type-safe wrapper around ObjectDetector that enforces GridImage
    input type. It is specialized for colony detection on arrayed plate images with grid structure.

    **Purpose**

    Use GridObjectDetector when implementing detection algorithms that find and label colonies
    in grid-structured agar plate images. Like ObjectDetector, it sets image.objmask and
    image.objmap. The difference is that it requires GridImage input, making explicit that
    your detection may leverage or assumes grid structure (well boundaries, grid alignment).

    **What GridObjectDetector produces**

    GridObjectDetector sets two outputs:

    - **image.objmask:** Binary mask (True=colony pixel, False=background)
    - **image.objmap:** Labeled integer map (0=background, 1..N=colony labels)

    Both are set synchronously to ensure consistency. The labels in objmap match the row/column
    structure of the grid (useful for tracking which colonies are in which wells).

    **GridImage vs Image**

    - **Image:** Generic image with optional, unvalidated grid information.
    - **GridImage:** Specialized Image subclass with validated grid structure (row/column
      layout, well positions, grid alignment). Suitable for 96-well, 384-well, or other
      arrayed plate formats. Created by GridFinder or manually specified.

    **When to use GridObjectDetector vs ObjectDetector**

    - **ObjectDetector:** Detection works equally well on any Image (with or without grid).
      Examples: Otsu thresholding, Canny edges, round peak detection on single images.
      Use when detection is global and grid-independent.

    - **GridObjectDetector:** Detection assumes or leverages grid structure. Examples:
      per-well detection (find colonies only within well boundaries), grid-aware peak detection
      (use well centers as hints), adaptive detection per well (tuning per grid region).
      Use when grid structure is essential to the detection algorithm.

    **Typical Use Cases**

    - **Per-well detection:** Find colonies only within well boundaries; one mask/label per well.
    - **Grid-hinted detection:** Use well center positions or grid-aligned regions as hints
      to improve detection accuracy.
    - **Adaptive detection:** Adjust detection parameters (threshold, sensitivity) per well
      to handle uneven plate illumination.
    - **Well isolation:** Ensure detected colonies don't bleed across well boundaries.

    **Implementation Pattern**

    Inherit from GridObjectDetector and implement ``_operate()`` as normal:

    .. code-block:: python

        from phenotypic.abc_ import GridObjectDetector
        from phenotypic import GridImage

        class GridAdaptiveDetector(GridObjectDetector):
            '''Detect colonies using per-well adaptive thresholding.'''

            neighborhood_size: int = 15  # Annotated class-level field

            def _operate(self, image: GridImage) -> GridImage:
                # image is guaranteed to be GridImage with grid structure
                # Use well positions to apply per-well detection
                from scipy.ndimage import label
                from skimage.filters import threshold_local

                enh = image.detect_mat[:]
                grid = image.grid  # Access grid structure

                # Apply adaptive threshold per well
                mask = threshold_local(enh, self.neighborhood_size) > enh

                # Label connected components
                labeled, _ = label(mask)

                image.objmask[:] = mask
                image.objmap[:] = labeled
                return image

    **Critical Implementation Detail**

    GridObjectDetector includes input validation (GridImage required) but NO output integrity
    checks. Like ObjectDetector, it is READ-ONLY for rgb, gray, detect_mat. You may only write
    to objmask and objmap.

    .. code-block:: python

        @staticmethod
        def _operate(image: GridImage, **kwargs) -> GridImage:
            # Read (protected by @validate_operation_integrity):
            enh = image.detect_mat[:]
            gray = image.gray[:]
            rgb = image.rgb[:]

            # Write (allowed):
            image.objmask[:] = binary_mask
            image.objmap[:] = labeled_map

            # GridImage structure (optional modification):
            # image.grid can be read, but typically not written

            return image

    **Grid-Aware Detection Patterns**

    1. **Per-well detection:** Create a mask/label per well independently
    2. **Well-boundary enforcement:** Mask pixels outside well boundaries after detection
    3. **Well-center hinting:** Use well positions as priors for peak detection
    4. **Adaptive parameters:** Vary detection thresholds based on well position or intensity

    **Notes**

    - GridObjectDetector enforces GridImage input type at runtime. Passing plain Image raises error.
    - Input validation uses @validate_operation_integrity('image.rgb', 'image.gray', 'image.detect_mat')
      to ensure image color data is not modified.
    - GridImage must have valid grid structure before detection. Typically set by GridFinder
      or manually specified grid before applying GridObjectDetector.
    - All ObjectDetector helper methods and patterns apply identically.
    - Output is always GridImage (input type is preserved).

    Examples:
        Per-well Otsu detection with grid structure:

        >>> from phenotypic import GridImage, Image
        >>> from phenotypic.abc_ import GridObjectDetector
        >>> from scipy.ndimage import label
        >>> from skimage.filters import threshold_otsu
        >>> import numpy as np
        >>> class GridOtsuDetector(GridObjectDetector):
        ...     '''Detect colonies using global Otsu threshold on grid plate.'''
        ...
        ...     def _operate(self, image: GridImage) -> GridImage:
        ...         enh = image.detect_mat[:]
        ...         # Apply global Otsu threshold
        ...         threshold = threshold_otsu(enh)
        ...         binary_mask = enh > threshold
        ...         # Label connected components
        ...         labeled_map, _ = label(binary_mask)
        ...         # Set detection results
        ...         image.objmask[:] = binary_mask
        ...         image.objmap[:] = labeled_map
        ...         return image
        >>> # Usage
        >>> image = Image.imread('plate.jpg')
        >>> grid_image = GridImage(image)
        >>> grid_image.detect_grid()
        >>> detector = GridOtsuDetector()
        >>> detected = detector.operate(grid_image)
        >>> # Grid structure preserved; can access wells
        >>> for well_row in range(grid_image.nrows):
        ...     for well_col in range(grid_image.ncols):
        ...         # Colonies in this well available via grid accessor
        ...         pass

        Per-well adaptive detection using well centers as hints:

        >>> from phenotypic.abc_ import GridObjectDetector
        >>> from phenotypic import GridImage
        >>> from scipy.ndimage import label
        >>> from skimage.filters import threshold_local
        >>> class GridAdaptiveDetector(GridObjectDetector):
        ...     '''Adaptive per-well detection using well center positions.'''
        ...
        ...     neighborhood_size: int = 31
        ...
        ...     def _operate(self, image: GridImage) -> GridImage:
        ...         enh = image.detect_mat[:]
        ...         grid = image.grid
        ...         # Apply local adaptive threshold (per-well region)
        ...         binary_mask = threshold_local(
        ...             enh, self.neighborhood_size
        ...         ) > enh
        ...         # Label and store
        ...         labeled_map, _ = label(binary_mask)
        ...         image.objmask[:] = binary_mask
        ...         image.objmap[:] = labeled_map
        ...         return image
        >>> # Usage: handle uneven illumination on large plates
        >>> detector = GridAdaptiveDetector(neighborhood_size=31)
        >>> detected = detector.operate(grid_image)
    """

    @validate_operation_integrity("image.rgb", "image.gray", "image.detect_mat")
    def apply(self, image: GridImage, inplace=False) -> GridImage:
        from phenotypic import GridImage

        if not isinstance(image, GridImage):
            raise GridImageInputError()
        return super().apply(image=image, inplace=inplace)

    @abstractmethod
    def _operate(self, image: GridImage) -> GridImage:  # type: ignore[override]
        return image
