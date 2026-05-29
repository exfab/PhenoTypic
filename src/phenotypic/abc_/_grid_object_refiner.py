from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

from phenotypic.abc_ import ObjectRefiner
from phenotypic.abc_ import GridOperation
from phenotypic.tools_.exceptions_ import GridImageInputError
from phenotypic.tools_.funcs_ import validate_operation_integrity


class GridObjectRefiner(ObjectRefiner, GridOperation, ABC):
    """Abstract base class for post-detection refinement operations on grid-aligned plate images.

    GridObjectRefiner is the grid-aware variant of ObjectRefiner, combining object mask refinement
    with grid structure awareness. It refines detected objects (colony masks and labeled maps) while
    respecting well boundaries and grid-aligned regions in arrayed plate images (96-well, 384-well,
    etc.). Like ObjectRefiner, it protects original image data (RGB, grayscale, detection matrix)
    and modifies only detection results.

    **Quick Decision Guide: GridObjectRefiner vs ObjectRefiner**

    - **Use GridObjectRefiner if:** Refining detections on GridImage where well structure or grid
      position affects refinement logic (per-well filtering, boundary enforcement, position-aware cleanup).
    - **Use ObjectRefiner if:** Refining on plain Image without grid, or using global algorithms that
      don't need grid awareness (size filtering, shape filtering, general morphology).
    - **GridImage requirement:** GridObjectRefiner only accepts GridImage input; plain Image raises
      GridImageInputError at runtime.
    - **Grid-aware refinement:** Access well positions, row/column boundaries, and grid metadata via
      image.grid to make position-aware decisions (e.g., filter colonies oversized for their well).
    - **Oversized colonies:** [GridOversizedObjectRemover](src/phenotypic/refine/_grid_oversized_object_remover.py)
      removes objects spanning nearly entire well (merged colonies, segmentation spillover).
    - **Per-well largest:** [KeepSectionLargest](src/phenotypic/refine/_keep_section_largest.py) keeps
      only the largest object per grid cell (one colony per well).
    - **Multi-well reducer:** [ReduceSectionsByLine](src/phenotypic/refine/_reduce_multiple_grid_objects.py)
      merges multiple objects per well into single representative region.
    - **Border handling:** Grid structure enables identifying and filtering objects near plate/well
      edges that may be incomplete or distorted.
    - **Grid registration:** When grid detection is imperfect, grid-aware refinement helps filter
      mis-assigned or boundary-adjacent objects by position.
    - **When to chain:** Combine grid refiners in ImagePipeline (e.g., global size filter, then
      per-well filtering) for comprehensive cleanup respecting array structure.

    **What is GridObjectRefiner?**

    GridObjectRefiner is the specialized version of ObjectRefiner for GridImage objects, adding grid-aware
    refinement to the core post-detection cleanup workflow:

    - **Grid-structure awareness:** Unlike ObjectRefiner (which operates globally), GridObjectRefiner can
      access grid boundaries, well positions, row/column indices, and per-cell metadata via ``image.grid``.
      This enables refinement logic that respects array structure.

    - **GridImage requirement:** Accepts only GridImage input (with detected grid structure), enforced
      at runtime via ``GridImageInputError``. Passing a plain Image raises an error.

    - **Grid access interface:** Within ``_operate()``, call ``image.grid.get_row_edges()``,
      ``image.grid.get_col_edges()``, and ``image.grid.info()`` to retrieve grid metadata and make
      position-aware refinement decisions.

    - **Detection-only modification:** Like ObjectRefiner, modifies only ``image.objmask[:]`` and
      ``image.objmap[:]``. Original image components (RGB, grayscale, detection matrix) are protected
      via ``@validate_operation_integrity`` decorator.

    - **Array phenotyping:** Well-suited for high-throughput plate analysis where grid structure matters
      for biology (one colony per well expected, oversized objects indicate merging, boundary objects
      may be incomplete or distorted).

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

    Subclass GridObjectRefiner and implement ``_operate()``. Use this template:

    .. code-block:: python

        from phenotypic.abc_ import GridObjectRefiner
        from phenotypic import GridImage
        import numpy as np

        class MyGridRefiner(GridObjectRefiner):
            max_width_fraction: float = 0.9  # Annotated class-level field

            def _operate(self, image: GridImage) -> GridImage:
                # Get grid structure information
                col_edges = image.grid.get_col_edges()      # x-coordinates of column boundaries
                row_edges = image.grid.get_row_edges()      # y-coordinates of row boundaries
                nrows, ncols = image.grid.nrows, image.grid.ncols

                # Get per-object grid metadata (label, row, column, position)
                grid_info = image.grid.info()  # pd.DataFrame

                # Measure object properties
                objmap = image.objmap[:]
                from skimage.measure import regionprops_table
                props = regionprops_table(objmap, properties=['label', 'bbox', 'area'])

                # Make position-aware refinement decisions
                # Example: filter objects by grid position or size relative to well
                max_width = (col_edges[1:] - col_edges[:-1]).max()
                # ... implement your grid-aware logic ...

                return image

    **Key Rules**

    1. ``_operate()`` must be an instance method (access parameters via ``self``).
    2. All parameters except ``image`` must be declared as annotated class-level fields.
    3. Only modify ``image.objmask[:]`` and ``image.objmap[:]``.
    4. Access grid via ``image.grid`` methods: get_row_edges(), get_col_edges(), info().
    5. Return the modified GridImage object.

    **Per-Well Filtering Patterns**

    Common grid-aware refinement strategies:

    - **Remove oversized objects:** Filter objects larger than well dimensions (merged colonies, segmentation
      spillover). Compare object bounding box to max_width and max_height of grid cells.

    - **Keep largest per well:** Select only the largest object per grid cell (assumes one colony per well).
      Use grid_info to group objects by row and column, then keep max-area object per group.

    - **Remove boundary objects:** Filter objects touching or near well edges (incomplete detections).
      Use grid_info's boundary flags or compute distance to nearest grid boundary.

    - **Per-row/column filtering:** Apply different thresholds by row or column position (accounts for
      uneven illumination across plate). Use grid_info to stratify objects by position.

    **Grid Access Interface**

    Within ``_operate()``, use the GridImage accessor to retrieve grid metadata:

    .. code-block:: python

        # Grid structure information
        nrows, ncols = image.grid.nrows, image.grid.ncols
        row_edges = image.grid.get_row_edges()          # Row boundary positions (y-coordinates)
        col_edges = image.grid.get_col_edges()          # Col boundary positions (x-coordinates)
        grid_info = image.grid.info()                   # DataFrame with per-object grid metadata

        # grid_info columns (example):
        # ['label', 'row', 'col', 'row_edge_min', 'row_edge_max', 'col_edge_min', 'col_edge_max', ...]

        # Per-well cell dimensions
        cell_heights = row_edges[1:] - row_edges[:-1]
        cell_widths = col_edges[1:] - col_edges[:-1]

    Notes:
        - **GridImage input required:** ``apply()`` enforces GridImage type at runtime.
          Passing a plain Image raises ``GridImageInputError``.

        - **Protected components:** The ``@validate_operation_integrity`` decorator ensures
          ``image.rgb``, ``image.gray``, ``image.detect_mat`` cannot be modified.
          Only ``image.objmask`` and ``image.objmap`` can be refined.

        - **Immutability by default:** ``apply(image)`` returns a modified copy. Set
          ``inplace=True`` for memory-efficient in-place modification.

        - **Grid structure assumption:** Your algorithm should assume a valid, registered grid.
          If grid metadata is unreliable, refinement may fail or produce wrong results.

        - **Instance _operate() method:** ``_operate()`` is an instance method; access parameters via ``self``.

        - **Field-based parameters:** All ``_operate()`` parameters except ``image``
          are declared as annotated class-level fields.

    Examples:
        Remove objects larger than their grid cell width:

        >>> from phenotypic.abc_ import GridObjectRefiner
        >>> from phenotypic import GridImage
        >>> import numpy as np
        >>> class OversizedObjectRemover(GridObjectRefiner):
        ...     '''Remove objects exceeding cell dimensions.'''
        ...
        ...     def _operate(self, image: GridImage) -> GridImage:
        ...         # Get grid boundaries
        ...         col_edges = image.grid.get_col_edges()
        ...         row_edges = image.grid.get_row_edges()
        ...         max_width = (col_edges[1:] - col_edges[:-1]).max()
        ...         max_height = (row_edges[1:] - row_edges[:-1]).max()
        ...         # Measure objects
        ...         objmap = image.objmap[:]
        ...         from skimage.measure import regionprops_table
        ...         props = regionprops_table(objmap, properties=['label', 'bbox'])
        ...         # Filter oversized
        ...         import pandas as pd
        ...         df = pd.DataFrame(props)
        ...         df['width'] = df['bbox-2'] - df['bbox-0']
        ...         df['height'] = df['bbox-3'] - df['bbox-1']
        ...         keep = df[(df['width'] < max_width) &
        ...                   (df['height'] < max_height)]['label'].values
        ...         # Refine map
        ...         refined = np.where(np.isin(objmap, keep), objmap, 0)
        ...         image.objmap[:] = refined
        ...         return image
        >>> # Usage on gridded plate image
        >>> from phenotypic.detect import OtsuDetector
        >>> image = GridImage.imread('plate.jpg', nrows=8, ncols=12)
        >>> detected = OtsuDetector().apply(image)
        >>> cleaned = OversizedObjectRemover().apply(detected)

        Per-well filtering: remove colonies oversized for their well:

        >>> from phenotypic.abc_ import GridObjectRefiner
        >>> from phenotypic import GridImage
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> import numpy as np
        >>> import pandas as pd
        >>> class PerWellOversizedRemover(GridObjectRefiner):
        ...     '''Remove objects that exceed size threshold relative to their well.'''
        ...
        ...     max_area_fraction: float = 0.8
        ...
        ...     def _operate(self, image: GridImage) -> GridImage:
        ...         # Get grid structure
        ...         col_edges = image.grid.get_col_edges()
        ...         row_edges = image.grid.get_row_edges()
        ...         # Compute well area (assume uniform grid cells)
        ...         cell_width = (col_edges[1:] - col_edges[:-1]).mean()
        ...         cell_height = (row_edges[1:] - row_edges[:-1]).mean()
        ...         max_cell_area = cell_width * cell_height
        ...         # Get grid info and measure object areas
        ...         objmap = image.objmap[:]
        ...         grid_info = image.grid.info()
        ...         from skimage.measure import regionprops_table
        ...         props = regionprops_table(objmap, properties=['label', 'area'])
        ...         props_df = pd.DataFrame(props)
        ...         # Filter: keep objects smaller than max_area_fraction of their well
        ...         max_allowed_area = max_cell_area * max_area_fraction
        ...         keep = props_df[props_df['area'] < max_allowed_area]['label'].values
        ...         # Refine map
        ...         refined = np.where(np.isin(objmap, keep), objmap, 0)
        ...         image.objmap[:] = refined
        ...         return image
        >>> # Usage: remove merged colonies and artifacts spanning most of well
        >>> from phenotypic.detect import OtsuDetector
        >>> image = load_synth_yeast_plate()  # Returns GridImage
        >>> detected = OtsuDetector().apply(image)
        >>> # Remove colonies > 80% of well area (likely merged or segmentation error)
        >>> cleaner = PerWellOversizedRemover(max_area_fraction=0.8)
        >>> refined = cleaner.apply(detected)
        >>> print(f"Removed oversized: {detected.objmap[:].max()} -> {refined.objmap[:].max()}")

        Chaining grid and non-grid refinements:

        >>> from phenotypic import GridImage, ImagePipeline
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.refine import SmallObjectRemover, GridOversizedObjectRemover
        >>> # Create detection pipeline with mixed refinements
        >>> pipeline = ImagePipeline()
        >>> pipeline.add(OtsuDetector())                        # Detect colonies
        >>> pipeline.add(SmallObjectRemover(min_size=100))      # Global size filter
        >>> pipeline.add(GridOversizedObjectRemover())          # Grid-aware filter
        >>> # Apply to gridded plate
        >>> image = GridImage.imread('plate.jpg', nrows=8, ncols=12)
        >>> results = pipeline.operate([image])
        >>> refined_image = results[0]
        >>> print(f"Refined: {refined_image.objmap[:].max()} colonies")
    """

    @validate_operation_integrity("image.rgb", "image.gray", "image.detect_mat")
    def apply(self, image: GridImage, inplace: bool = False) -> GridImage:
        from phenotypic import GridImage

        if not isinstance(image, GridImage):
            raise GridImageInputError()
        output = super().apply(image=image, inplace=inplace)
        return output

    @abstractmethod
    def _operate(self, image: GridImage) -> GridImage:  # type: ignore[override]
        return image
