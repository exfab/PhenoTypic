from __future__ import annotations
import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import GridImage

from phenotypic.abc_ import ObjectRefiner
from phenotypic.abc_ import GridOperation
from phenotypic.tools.exceptions_ import GridImageInputError
from phenotypic.tools.funcs_ import validate_operation_integrity
from abc import ABC


class GridObjectRefiner(ObjectRefiner, GridOperation, ABC):
    """Abstract base class for post-detection refinement operations on grid-aligned plate images.

    GridObjectRefiner is the grid-aware variant of ObjectRefiner, combining object mask refinement
    with grid structure awareness. It refines detected objects (colony masks and labeled maps) while
    respecting well boundaries and grid-aligned regions in arrayed plate images (96-well, 384-well,
    etc.). Like ObjectRefiner, it protects original image data (RGB, grayscale, enhanced grayscale)
    and modifies only detection results.

    **What is GridObjectRefiner?**

    GridObjectRefiner is the specialized version of ObjectRefiner for GridImage objects:

    - **GridImage requirement:** Accepts only GridImage input (with detected grid structure),
      enforced at runtime via ``GridImageInputError``.

    - **Grid-aware refinement:** Can access well positions, grid cell boundaries, and row/column
      structure via ``image.grid`` to make refinement decisions (e.g., remove colonies that exceed
      well boundaries, filter by grid position).

    - **Detection-only modification:** Like ObjectRefiner, modifies only ``image.objmask[:]`` and
      ``image.objmap[:]``. Original image components are protected via ``@validate_operation_integrity``.

    **When to use GridObjectRefiner vs ObjectRefiner**

    - **ObjectRefiner:** Use when refining detections on a plain Image without grid structure.
      Examples: general-purpose size filtering, morphological cleanup, shape filtering (applies
      globally regardless of position).

    - **GridObjectRefiner:** Use when refining detections on a GridImage where well structure matters.
      Examples: removing objects larger than their grid cell (``GridOversizedObjectRemover``),
      per-well filtering, grid-aligned edge removal. The grid structure enables position-aware
      refinement that improves array phenotyping accuracy.

    **Typical Use Cases**

    GridObjectRefiner is useful for addressing grid-specific artifacts:

    - **Oversized colonies:** Objects spanning nearly an entire well (merged colonies, agar edges,
      segmentation spillover). Filtering improves per-well consistency.

    - **Inter-well artifacts:** Detections touching or bridging grid cell boundaries from uneven
      lighting or thresholding errors.

    - **Boundary contamination:** Colonies near plate edges that are incomplete or distorted.
      Grid structure allows identifying and filtering boundary-adjacent objects.

    - **Grid registration errors:** When grid detection is imperfect, some objects may be mis-assigned
      to wells; grid-aware refinement can filter or relocate based on position.

    **Implementing a Custom GridObjectRefiner**

    Subclass GridObjectRefiner and implement ``_operate()``:

    .. code-block:: python

        from phenotypic.abc_ import GridObjectRefiner
        from phenotypic import GridImage
        import numpy as np

        class MyGridRefiner(GridObjectRefiner):
            def __init__(self, max_width_fraction: float = 0.9):
                super().__init__()
                self.max_width_fraction = max_width_fraction

            @staticmethod
            def _operate(image: GridImage, max_width_fraction: float = 0.9) -> GridImage:
                # Get grid info
                col_edges = image.grid.get_col_edges()
                max_cell_width = (col_edges[1:] - col_edges[:-1]).max()

                # Measure object widths
                objmap = image.objmap[:]
                from skimage.measure import regionprops_table

                props = regionprops_table(objmap, properties=['label', 'bbox'])
                # ... compute widths and filter ...

                return image

    **Key Rules**

    1. ``_operate()`` must be static (for parallel execution).
    2. All parameters except ``image`` must exist as instance attributes.
    3. Only modify ``image.objmask[:]`` and ``image.objmap[:]``.
    4. Access grid via ``image.grid`` (row/column edges, well positions, metadata).
    5. Return the modified GridImage.

    **Grid Access Patterns**

    Within ``_operate()``, access grid information via the GridImage accessor:

    .. code-block:: python

        # Grid structure
        nrows, ncols = image.grid.nrows, image.grid.ncols
        row_edges = image.grid.get_row_edges()          # Row boundary positions (y-coordinates)
        col_edges = image.grid.get_col_edges()          # Col boundary positions (x-coordinates)
        cell_info = image.grid.info()                   # DataFrame with per-object grid info

        # Per-object grid metadata (label, row, col, boundary flags)
        grid_data = image.grid.info()  # pd.DataFrame with object properties

    Notes:
        - **GridImage input required:** ``apply()`` enforces GridImage type at runtime.
          Passing a plain Image raises ``GridImageInputError``.

        - **Protected components:** The ``@validate_operation_integrity`` decorator ensures
          ``image.rgb``, ``image.gray``, ``image.enh_gray`` cannot be modified.
          Only ``image.objmask`` and ``image.objmap`` can be refined.

        - **Immutability by default:** ``apply(image)`` returns a modified copy. Set
          ``inplace=True`` for memory-efficient in-place modification.

        - **Grid structure assumption:** Your algorithm should assume a valid, registered grid.
          If grid metadata is unreliable, refinement may fail or produce wrong results.

        - **Static _operate() requirement:** Must be static for parallel execution in pipelines.

        - **Parameter matching:** All ``_operate()`` parameters except ``image`` must exist
          as instance attributes for automatic parameter matching.

    Examples:
        .. dropdown:: Remove objects larger than their grid cell width

            .. code-block:: python

                from phenotypic.abc_ import GridObjectRefiner
                from phenotypic import GridImage
                import numpy as np

                class OversizedObjectRemover(GridObjectRefiner):
                    '''Remove objects exceeding cell dimensions.'''

                    def __init__(self):
                        super().__init__()

                    @staticmethod
                    def _operate(image: GridImage) -> GridImage:
                        # Get grid boundaries
                        col_edges = image.grid.get_col_edges()
                        row_edges = image.grid.get_row_edges()
                        max_width = (col_edges[1:] - col_edges[:-1]).max()
                        max_height = (row_edges[1:] - row_edges[:-1]).max()

                        # Measure objects
                        objmap = image.objmap[:]
                        from skimage.measure import regionprops_table
                        props = regionprops_table(objmap, properties=['label', 'bbox'])

                        # Filter oversized
                        import pandas as pd
                        df = pd.DataFrame(props)
                        df['width'] = df['bbox-2'] - df['bbox-0']
                        df['height'] = df['bbox-3'] - df['bbox-1']
                        keep = df[(df['width'] < max_width) &
                                  (df['height'] < max_height)]['label'].values

                        # Refine map
                        refined = np.where(np.isin(objmap, keep), objmap, 0)
                        image.objmap[:] = refined
                        return image

                # Usage on gridded plate image
                from phenotypic.detect import OtsuDetector
                image = GridImage.from_image_path('plate.jpg', nrows=8, ncols=12)
                detected = OtsuDetector().apply(image)
                cleaned = OversizedObjectRemover().apply(detected)

        .. dropdown:: Chaining grid and non-grid refinements

            .. code-block:: python

                from phenotypic import GridImage, ImagePipeline
                from phenotypic.detect import OtsuDetector
                from phenotypic.refine import SmallObjectRemover, GridOversizedObjectRemover

                # Create detection pipeline with mixed refinements
                pipeline = ImagePipeline()
                pipeline.add(OtsuDetector())                        # Detect colonies
                pipeline.add(SmallObjectRemover(min_size=100))      # Global size filter
                pipeline.add(GridOversizedObjectRemover())          # Grid-aware filter

                # Apply to gridded plate
                image = GridImage.from_image_path('plate.jpg', nrows=8, ncols=12)
                results = pipeline.operate([image])
                refined_image = results[0]

                print(f"Refined: {refined_image.objmap[:].max()} colonies")
    """

    @validate_operation_integrity("image.rgb", "image.gray", "image.enh_gray")
    def apply(self, image: GridImage, inplace: bool = False) -> GridImage:
        from phenotypic import GridImage

        if not isinstance(image, GridImage):
            raise GridImageInputError()
        output = super().apply(image=image, inplace=inplace)
        return output

    @abc.abstractmethod
    def _operate(self, image: GridImage) -> GridImage:
        return image
