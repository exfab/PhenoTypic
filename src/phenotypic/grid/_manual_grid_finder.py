from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import pandas as pd
import numpy as np
from pydantic import field_validator, model_validator

from phenotypic.abc_ import GridFinder
from phenotypic.tools_.typing_ import TuneSpec


class ManualGridFinder(GridFinder):
    """
    A GridFinder implementation where users directly specify grid row and column coordinates.

    This class allows for complete manual control over grid placement by accepting
    explicit row and column edge coordinates. No optimization or automatic calculation
    is performed - the grid is defined exactly as specified by the user.

    Attributes:
        nrows (int): Number of rows in the grid (derived from row_edges).
        ncols (int): Number of columns in the grid (derived from col_edges).
        row_edges (list[int]): Sorted row edge coordinates defining grid rows.
        col_edges (list[int]): Sorted column edge coordinates defining grid columns.

    Args:
        row_edges: Row edge coordinates. Length should be nrows + 1.
            Example: ``[0, 100, 200, 300]`` defines 3 rows. A list,
            tuple, or ``np.ndarray`` is accepted; values are cast to
            ``int`` and sorted ascending.
        col_edges: Column edge coordinates. Length should be ncols + 1.
            Example: ``[0, 80, 160, 240, 320]`` defines 4 columns. A
            list, tuple, or ``np.ndarray`` is accepted; values are cast
            to ``int`` and sorted ascending.

    Raises:
        ValueError: If row_edges or col_edges have fewer than 2 elements.

    Example:
        Create a 3x4 grid with specific coordinates:

        >>> import numpy as np
        >>> from phenotypic.grid import ManualGridFinder
        >>> # Create a 3x4 grid with specific coordinates
        >>> row_edges = np.array([0, 100, 200, 300])  # 3 rows
        >>> col_edges = np.array([0, 80, 160, 240, 320])  # 4 columns
        >>> finder = ManualGridFinder(row_edges=row_edges, col_edges=col_edges)
        >>> grid_info = finder.measure(image)  # doctest: +SKIP
    """

    # ``nrows`` / ``ncols`` are required fields on the ``GridFinder`` ABC,
    # but ``ManualGridFinder`` derives them from the edge arrays instead of
    # taking them as constructor parameters. Declaring defaults here lets
    # pydantic build the model; ``_derive_grid_shape`` (a post-validator)
    # overwrites them from the validated edges, reproducing the legacy
    # ``__init__`` behaviour.
    # ``nrows`` / ``ncols`` are structural plate-geometry parameters derived
    # from the edge arrays, not accuracy knobs, so they are excluded from
    # the tuning search.
    nrows: Annotated[int, TuneSpec(tunable=False)] = 0
    ncols: Annotated[int, TuneSpec(tunable=False)] = 0
    row_edges: list[int]
    col_edges: list[int]

    @field_validator("row_edges", mode="before")
    @classmethod
    def _normalize_row_edges(cls, value: Any) -> list[int]:
        """Coerce row edges to a sorted ``list[int]``.

        Accepts a list, tuple, or ``np.ndarray`` (the legacy constructor
        took an ``np.ndarray``); reproduces the pre-migration
        ``np.asarray(row_edges, dtype=int)`` cast followed by ``.sort()``.
        """
        edges = np.asarray(value, dtype=int)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError(
                "row_edges must have at least 2 elements to define at least 1 row"
            )
        return sorted(int(x) for x in edges)

    @field_validator("col_edges", mode="before")
    @classmethod
    def _normalize_col_edges(cls, value: Any) -> list[int]:
        """Coerce column edges to a sorted ``list[int]``.

        Accepts a list, tuple, or ``np.ndarray`` (the legacy constructor
        took an ``np.ndarray``); reproduces the pre-migration
        ``np.asarray(col_edges, dtype=int)`` cast followed by ``.sort()``.
        """
        edges = np.asarray(value, dtype=int)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError(
                "col_edges must have at least 2 elements to define at least 1 column"
            )
        return sorted(int(x) for x in edges)

    @model_validator(mode="after")
    def _derive_grid_shape(self) -> ManualGridFinder:
        """Derive ``nrows`` / ``ncols`` from the validated edge arrays.

        Reproduces the legacy ``__init__`` lines
        ``self.nrows = len(self._row_edges) - 1`` and
        ``self.ncols = len(self._col_edges) - 1``.
        """
        object.__setattr__(self, "nrows", len(self.row_edges) - 1)
        object.__setattr__(self, "ncols", len(self.col_edges) - 1)
        return self

    def _operate(self, image: Image) -> pd.DataFrame:
        """
        Processes an image to assign objects to grid cells based on manually specified edges.

        Args:
            image (Image): The image containing objects to be gridded.

        Returns:
            pd.DataFrame: A DataFrame containing the grid results including boundary intervals,
                grid indices, and section numbers corresponding to the manually defined grid.
        """
        # Use base class method to assemble grid info with our predefined edges
        return self._get_grid_info(
            image=image,
            row_edges=np.asarray(self.row_edges, dtype=int),
            col_edges=np.asarray(self.col_edges, dtype=int),
        )

    def get_row_edges(self, image: Image) -> np.ndarray:
        """
        Returns the manually specified row edges.

        Args:
            image (Image): The image (not used, but required by interface).

        Returns:
            np.ndarray: Array of row edge coordinates.
        """
        return np.asarray(self.row_edges, dtype=int)

    def get_col_edges(self, image: Image) -> np.ndarray:
        """
        Returns the manually specified column edges.

        Args:
            image (Image): The image (not used, but required by interface).

        Returns:
            np.ndarray: Array of column edge coordinates.
        """
        return np.asarray(self.col_edges, dtype=int)
