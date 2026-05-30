from __future__ import annotations

import json
import warnings
from typing import Union, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import napari

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phenotypic.abc_ import GridFinder
from phenotypic._core._image_parts.accessors import GridAccessor
from phenotypic.grid import AutoGridFinder
from phenotypic.measure import MeasureBounds
from phenotypic.schema import METADATA
from phenotypic.tools_.constants_ import IMAGE_TYPES
from phenotypic.schema import BBOX
from phenotypic.tools_.exceptions_ import IllegalAssignmentError
from .._image import Image


class ImageGridHandler(Image):
    """
    A specialized Image object that supports grid-based processing and overlay visualization.

    This class extends the base `Image` class functionality to include grid handling,
    grid-based slicing, and advanced visualization capabilities such as displaying overlay information
    with gridlines and annotations. It interacts with the provided grid handling utilities
    to determine grid structure and assign/overlay it effectively on the image.

    Attributes:
        _GRIDLINE_WIDTH_FACTOR: Tunable factor for gridline width calculation. The line
            width is calculated as max(1, int(min(height, width) * factor)). Increase
            this value for thicker gridlines on large images. Default is 0.002.

    Args:
            arr (Optional[Union[np.ndarray, Type[Image]]]): The im
                image, which can be a NumPy array or an image-like object. If
                this parameter is not provided, it defaults to None.
            grid_finder (Optional[GridFinder]): An optional GridFinder instance
                for defining grids on the image. If not provided, it defaults to
                a center grid setter.
            nrows (int): An integer passed to the grid setter to specify the number of nrows in the grid
                (Defaults to 8).
            ncols (int): An integer passed to the grid setter to specify the number of columns in the grid
                (Defaults to 12).

    Attributes:
        grid_finder (Optional[GridFinder]): An object responsible for defining and optimizing the grid
            layout over the image, defaulting to an `OptimalCenterGridSetter` instance if none is provided.
        _accessors.grid (GridAccessor): An internal utility for managing grid-based operations such as
            accessing row and column edges and generating section maps for the image's grid system.
    """

    # Tunable factor for gridline width. Line width = max(1, int(min(h, w) * factor))
    # Increase for thicker gridlines on large/high-resolution images.
    _GRIDLINE_WIDTH_FACTOR: float = 0.002

    def __init__(
            self,
            arr: Optional[Union[np.ndarray, Image]] = None,
            name: str = None,
            grid_finder: Optional[GridFinder] = None,
            nrows: int = 8,
            ncols: int = 12,
            **kwargs,
    ):
        """
        Initializes the instance with the given image, format, grid finding
        mechanism, and dimensions of the grid.

        Args:
            arr (Optional[Union[np.ndarray, Image]]): The input image provided
                as a NumPy array or an image object. Can be None if the image is
                optional for initialization.
            name (str): The name identifier for the image.
            grid_finder (Optional[GridFinder]): Mechanism responsible for finding a grid
                within the image. If None, an optimal center grid finder is instantiated.
            nrows (int): Number of nrows in the grid. Defaults to 8.
            ncols (int): Number of columns in the grid. Defaults to 12.

        Attributes:
            _grid_setter (Optional[GridFinder]): Private attribute storing the grid finding
                mechanism, which is either passed as input or is generated internally.
            _accessors.grid (GridAccessor): The grid accessor object for managing and
                accessing grid-related functionalities.
        """
        super().__init__(arr=arr, name=name, **kwargs)

        if hasattr(arr, "grid_finder"):
            grid_finder = arr.grid_finder
        elif grid_finder is None:
            grid_finder = AutoGridFinder(nrows=nrows, ncols=ncols)

        self._grid_finder: Optional[GridFinder] = grid_finder
        self._accessors.grid = GridAccessor(self)
        self.metadata[METADATA.IMAGE_TYPE] = IMAGE_TYPES.GRID.value

    @property
    def grid(self) -> GridAccessor:
        """Returns the GridAccessor object for grid-related operations.

        Returns:
            GridAccessor: Provides access to Grid-related operations.

        See Also :class:`GridAccessor`
        """
        return self._accessors.grid

    @grid.setter
    def grid(self, grid):
        raise IllegalAssignmentError("grid")

    def info(self, include_metadata: bool = True) -> pd.DataFrame:
        return self.grid.info(include_metadata=include_metadata)

    @property
    def grid_finder(self) -> GridFinder:
        """Get the GridFinder object responsible for detecting and aligning the grid.

        The GridFinder determines the positions of grid lines (rows and columns) in the
        image, enabling well-level detection and measurement. The finder is initialized
        during construction and can be customized via the grid_finder setter.

        Returns:
            GridFinder: The grid finding/alignment algorithm currently in use for this image.

        Examples:
            Access and inspect grid finder:

            >>> from phenotypic import GridImage
            >>> from phenotypic.grid import AutoGridFinder
            >>> grid_img = GridImage('plate.jpg')
            >>> finder = grid_img.grid_finder
            >>> print(type(finder))  # GridFinder instance

        See Also:
            GridFinder: Base class for grid finding algorithms.
        """
        return self._grid_finder

    @grid_finder.setter
    def grid_finder(self, value: GridFinder):
        """Set a new GridFinder for grid detection and alignment.

        Replaces the current grid finding algorithm with a new one. This is useful for
        switching between different grid detection strategies or fine-tuning grid
        parameters after image loading.

        Args:
            value (GridFinder): A GridFinder instance to use for grid operations.

        Raises:
            TypeError: If value is not a GridFinder instance.

        Examples:
            Replace grid finder with custom implementation:

            >>> from phenotypic import GridImage
            >>> from phenotypic.grid import AutoGridFinder
            >>> grid_img = GridImage('plate.jpg')
            >>> new_finder = AutoGridFinder(nrows=16, ncols=24)
            >>> grid_img.grid_finder = new_finder
        """
        if isinstance(value, GridFinder):
            self._grid_finder = value
        else:
            raise TypeError(f"Expected GridFinder, got {type(value)}")

    @property
    def nrows(self) -> int:
        """
        Retrieves the number of nrows in the grid.

        This property is used to access the number of nrows present in the grid
        object. It encapsulates the `nrows` attribute of the `grid` and returns
        it as an integer.

        Returns:
            int: The number of nrows in the grid.
        """
        return self.grid_finder.nrows

    @nrows.setter
    def nrows(self, nrows: int):
        """Set the number of rows in the grid structure.

        Updates the grid finder with a new row count. This is useful for adjusting
        the grid layout after the image has been loaded, which affects well-level
        detection and measurement operations.

        Args:
            nrows (int): The number of rows to set in the grid. Must be a positive integer.
                Common values: 8 (96-well), 16 (384-well), 32 (1536-well).

        Raises:
            TypeError: If nrows is not of type int.

        Examples:
            Adjust grid rows for different plate formats:

            >>> from phenotypic import GridImage
            >>> grid_img = GridImage('plate.jpg')
            >>> grid_img.nrows = 16  # Switch to 384-well format
            >>> print(grid_img.nrows)  # Output: 16
        """
        if not isinstance(nrows, int):
            raise TypeError(f"Expected int, got {type(nrows)}")
        self.grid_finder.nrows = nrows

    @property
    def ncols(self) -> int:
        """
        Gets the number of columns in the grid.

        This property retrieves the total number of columns in the grid
        by accessing the corresponding attribute of the underlying grid
        instance. It provides a read-only interface to the `ncols` value.

        Returns:
            int: The number of columns in the grid.
        """
        return self.grid_finder.ncols

    @ncols.setter
    def ncols(self, ncols: int):
        """Set the number of columns in the grid structure.

        Updates the grid finder with a new column count. This is useful for adjusting
        the grid layout after the image has been loaded, which affects well-level
        detection and measurement operations.

        Args:
            ncols (int): The number of columns to set in the grid. Must be a positive integer.
                Common values: 12 (96-well), 24 (384-well), 48 (1536-well).

        Raises:
            TypeError: If ncols is not of type int.

        Examples:
            Adjust grid columns for different plate formats:

            >>> from phenotypic import GridImage
            >>> grid_img = GridImage('plate.jpg')
            >>> grid_img.ncols = 24  # Switch to 384-well format
            >>> print(grid_img.ncols)  # Output: 24
        """
        if not isinstance(ncols, int):
            raise TypeError(f"Expected int, got {type(ncols)}")
        self.grid_finder.ncols = ncols

    def __getitem__(self, key) -> Image:
        """Returns a copy of the image at the slices specified as a regular Image object.

        Returns:
            Image: A copy of the image at the slices indicated
        """
        if not self.rgb.isempty():
            subimage = Image(arr=self.rgb[key])
        else:
            subimage = Image(arr=self.gray[key])

        # Propagate detect_mode before setting detect_mat data
        if self._data.detect_mode != "gray":
            subimage._data.detect_mode = self._data.detect_mode
        subimage.detect_mat[:] = self.detect_mat[key]
        subimage.objmap[:] = self.objmap[key]
        return subimage

    def napari(
            self, name: str | None = None, reset: bool = False,
            *, viewer: napari.Viewer | None = None,
    ) -> napari.Viewer:
        """Add all image layers and grid overlay to a persistent napari viewer.

        Extends the base :meth:`Image.napari` by automatically adding
        gridline and section-box overlay layers via
        :meth:`GridAccessor.napari`.

        Args:
            name: Optional custom name for layers. Defaults to image name.
            reset: If True, close and recreate the viewer. Defaults to False.
            viewer: Optional external napari viewer instance to use instead of
                the global viewer. When provided, global viewer management is
                bypassed. Defaults to None.

        Returns:
            The global napari viewer with image and grid layers.

        Raises:
            ImportError: If napari is not installed.

        Examples:
            >>> from phenotypic.data import load_synth_yeast_plate
            >>> gi = load_synth_yeast_plate()
            >>> viewer = gi.napari()  # doctest: +SKIP
        """
        result = super().napari(name=name, reset=reset, viewer=viewer)
        if self.num_objects > 0:
            result = self.grid.napari(name=name, viewer=viewer)
        return result

    def _draw_gridlines_on_overlay(
            self,
            overlay_arr: np.ndarray,
            gridline_color: Tuple[int, int, int] = (0, 255, 255),
    ) -> np.ndarray:
        """Draw gridlines on an overlay array for GridImage visualization.

        This method adds grid structure visualization to an overlay array by
        drawing lines at row and column boundaries. Line width scales dynamically
        with image size based on _GRIDLINE_WIDTH_FACTOR.

        Args:
            overlay_arr: 8-bit RGB array (H x W x 3) to draw gridlines on.
            gridline_color: RGB tuple for gridline color. Defaults to cyan
                (0, 255, 255).

        Returns:
            np.ndarray: Copy of overlay_arr with gridlines drawn.
        """
        from skimage.draw import line

        arr = overlay_arr.copy()
        h, w = arr.shape[:2]

        # Calculate dynamic line width based on image size
        line_width = max(1, int(min(h, w) * self._GRIDLINE_WIDTH_FACTOR))
        dash_len = max(4, line_width * 6)

        # Get grid edges
        col_edges = self.grid.get_col_edges()
        row_edges = self.grid.get_row_edges()

        if len(col_edges) == 0 or len(row_edges) == 0:
            return arr

        row_min = int(np.clip(row_edges.min(), 0, h - 1))
        row_max = int(np.clip(row_edges.max(), 0, h - 1))
        col_min = int(np.clip(col_edges.min(), 0, w - 1))
        col_max = int(np.clip(col_edges.max(), 0, w - 1))

        # Draw vertical lines at column edges (dashed)
        for x in col_edges:
            x = int(np.clip(x, 0, w - 1))
            for offset in range(-line_width // 2, line_width // 2 + 1):
                x_off = int(np.clip(x + offset, 0, w - 1))
                rr, cc = line(row_min, x_off, row_max - 1, x_off)
                valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                rr, cc = rr[valid], cc[valid]
                dash_mask = (rr % (2 * dash_len)) < dash_len
                arr[rr[dash_mask], cc[dash_mask]] = gridline_color

        # Draw horizontal lines at row edges (dashed)
        for y in row_edges:
            y = int(np.clip(y, 0, h - 1))
            for offset in range(-line_width // 2, line_width // 2 + 1):
                y_off = int(np.clip(y + offset, 0, h - 1))
                rr, cc = line(y_off, col_min, y_off, col_max - 1)
                valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                rr, cc = rr[valid], cc[valid]
                dash_mask = (cc % (2 * dash_len)) < dash_len
                arr[rr[dash_mask], cc[dash_mask]] = gridline_color

        return arr

    def _draw_section_boxes_on_overlay(
            self,
            overlay_arr: np.ndarray,
            box_colors: list[Tuple[int, int, int]] | None = None,
    ) -> np.ndarray:
        """Draw colored bounding boxes around each grid section's detected objects.

        This method adds colored rectangular boxes around each grid section that
        contains detected colonies, similar to the Rectangle patches in
        show(overlay=True). Uses skimage.draw for pixel-level drawing. Line width
        scales dynamically with image size based on _GRIDLINE_WIDTH_FACTOR.

        Args:
            overlay_arr: 8-bit RGB array (H x W x 3) to draw boxes on.
            box_colors: Optional list of RGB tuples for cycling through section
                colors. Defaults to tab20 colormap colors (avoiding cyan to
                differentiate from gridlines).

        Returns:
            np.ndarray: Copy of overlay_arr with section boxes drawn.
        """
        from skimage.draw import rectangle_perimeter

        arr = overlay_arr.copy()
        h, w = arr.shape[:2]

        # Calculate dynamic line width based on image size
        line_width = max(1, int(min(h, w) * self._GRIDLINE_WIDTH_FACTOR))

        # Default colors from tab20 (0-255 RGB)
        if box_colors is None:
            cmap = plt.get_cmap("tab20")
            box_colors = [
                tuple(int(c * 255) for c in cmap(i)[:3])
                for i in range(cmap.N)
            ]

        # Get section map and measure bounds per section
        img_copy = self.copy()
        img_copy.objmap = self.grid.get_section_map()

        if img_copy.num_objects == 0:
            return arr

        gs_table = MeasureBounds().measure(img_copy)

        # Draw bounding box for each section
        color_idx = 0
        for section_label in gs_table.index.unique():
            subtable = gs_table.loc[section_label, :]

            # Handle both single-row and multi-row subtables
            if isinstance(subtable, pd.Series):
                min_rr = int(subtable.loc[str(BBOX.MIN_RR)])
                max_rr = int(subtable.loc[str(BBOX.MAX_RR)])
                min_cc = int(subtable.loc[str(BBOX.MIN_CC)])
                max_cc = int(subtable.loc[str(BBOX.MAX_CC)])
            else:
                min_rr = int(subtable[str(BBOX.MIN_RR)].min())
                max_rr = int(subtable[str(BBOX.MAX_RR)].max())
                min_cc = int(subtable[str(BBOX.MIN_CC)].min())
                max_cc = int(subtable[str(BBOX.MAX_CC)].max())

            # Clip to valid bounds
            min_rr = max(0, min_rr)
            max_rr = min(h - 1, max_rr)
            min_cc = max(0, min_cc)
            max_cc = min(w - 1, max_cc)

            # Get color for this section
            color = box_colors[color_idx % len(box_colors)]
            color_idx += 1

            # Draw rectangle perimeter with line width
            for offset in range(-line_width // 2, line_width // 2 + 1):
                try:
                    rr, cc = rectangle_perimeter(
                            start=(min_rr + offset, min_cc + offset),
                            end=(max_rr - offset, max_cc - offset),
                            shape=(h, w),
                    )
                    valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                    arr[rr[valid], cc[valid]] = color
                except (ValueError, IndexError):
                    continue

        return arr

    # ------------------------------------------------------------------
    # HDF5 round-trip — schema_version=2 /grid/ subgroup
    # ------------------------------------------------------------------
    def _save_image2hdfgroup(
            self,
            grp,
            compression="gzip",
            compression_opts=4,
    ):
        """Save GridImage data + grid state into an HDF5 group.

        Defers base layers/metadata/root-attr writing to
        :meth:`Image._save_image2hdfgroup`, then persists grid-specific state
        under ``/grid/``:
          - attrs: ``nrows``, ``ncols``
          - ``grid_finder_json`` dataset: JSON blob of the serialised
            ``grid_finder`` (class + params).
        """
        super()._save_image2hdfgroup(
                grp,
                compression=compression,
                compression_opts=compression_opts,
        )

        grid = grp.require_group("grid")
        grid.attrs["nrows"] = int(self.nrows)
        grid.attrs["ncols"] = int(self.ncols)

        if self.grid_finder is not None:
            # Lazy import to avoid import cycles.
            from phenotypic._core._pipeline_parts._serializable_pipeline import (
                SerializablePipeline,
            )

            payload = {
                "class" : type(self.grid_finder).__name__,
                "params": SerializablePipeline._serialize_single_operation(
                        self.grid_finder
                ),
            }
            # Overwrite any pre-existing payload for idempotent re-saves.
            if "grid_finder_json" in grid:
                del grid["grid_finder_json"]
            grid.create_dataset(
                    "grid_finder_json",
                    data=json.dumps(payload),
                    dtype=h5py.string_dtype(encoding="utf-8"),
            )

    @classmethod
    def _load_from_hdf5_group(cls, group, **kwargs):
        """Load a GridImage from an HDF5 group.

        Reads ``/grid/`` (when present) into ``kwargs`` via ``setdefault``
        so explicit caller kwargs take priority, then delegates to the base
        Image loader.
        """
        if "grid" in group:
            grid = group["grid"]
            if "nrows" in grid.attrs:
                try:
                    kwargs.setdefault("nrows", int(grid.attrs["nrows"]))
                except (TypeError, ValueError):
                    pass
            if "ncols" in grid.attrs:
                try:
                    kwargs.setdefault("ncols", int(grid.attrs["ncols"]))
                except (TypeError, ValueError):
                    pass
            if "grid_finder_json" in grid:
                from phenotypic._core._pipeline_parts._serializable_pipeline import (
                    SerializablePipeline,
                )

                raw = grid["grid_finder_json"][()]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                try:
                    payload = json.loads(raw)
                    grid_finder = SerializablePipeline._deserialize_operations(
                            {"__gf__": payload}
                    )["__gf__"]
                    kwargs.setdefault("grid_finder", grid_finder)
                except (json.JSONDecodeError, KeyError, AttributeError) as e:
                    warnings.warn(
                        f"GridFinder deserialization failed ({type(e).__name__}: {e}); "
                        f"falling back to default AutoGridFinder. Grid configuration may be incorrect.",
                        UserWarning,
                        stacklevel=2,
                    )

        return super()._load_from_hdf5_group(group, **kwargs)
