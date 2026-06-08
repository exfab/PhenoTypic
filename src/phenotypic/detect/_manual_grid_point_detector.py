from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

from phenotypic.abc_ import GridObjectDetector
from phenotypic.tools_.mixin._footprint_mixin import FootprintMixin


class ManualGridPointDetector(GridObjectDetector, FootprintMixin):
    """Detect colonies by stamping footprint masks at evenly-spaced grid positions derived from reference coordinates.

    Compute a regular grid of colony positions from one or two user-supplied
    pixel coordinates, then stamp a morphological footprint at each position
    to produce ``objmask`` and ``objmap``. This purely geometric approach
    bypasses intensity-based detection entirely, making it ideal when colony
    positions follow a known pattern but automatic grid detection is
    unreliable. For a full comparison see
    :doc:`/explanation/detection_strategies_compared`.

    In **one-coordinate mode**, *coord1* defines the top-left cell centre and
    symmetric margins are assumed: row spacing =
    ``(H - 2*y) / (nrows - 1)``, column spacing =
    ``(W - 2*x) / (ncols - 1)``. In **two-coordinate mode**, *coord1* and
    *coord2* define cells (0, 0) and (1, 1); row and column spacing are
    derived from their difference and extrapolated across all grid cells.

    Best For:
        * Plates where automatic grid finders fail due to low contrast,
          missing wells, or non-standard plate formats.
        * Template-based detection when colony positions are known a priori
          from plate layout metadata or robotic spotting coordinates.
        * Generating ground-truth masks for testing or validating other
          detection pipelines.
        * Quick prototyping when full detection is unnecessary and grid
          geometry is well-characterised.

    Consider Also:
        * :class:`RoundPeaksDetector` when grid positions can be inferred
          automatically from intensity profiles.
        * :class:`WatershedDetector` when colonies are not on a regular
          grid and must be separated by region growing.
        * :class:`InoculumDetector` when inoculation sites must be detected
          from image content rather than geometric templates.

    Args:
        coord1: ``(y, x)`` pixel position of the top-left grid cell centre
            (row 0, column 0), picked by the user (no universal value).
            This is the anchor from which all other positions are
            calculated. Default ``(0, 0)``. Place it on the visible centre
            of the first colony; adjust by observing whether the stamped
            grid drifts off the colonies toward the far rows/columns.

        coord2: Optional ``(y, x)`` pixel position of the diagonally
            adjacent cell (row 1, column 1). When provided, row and column
            spacing are derived from the difference between *coord2* and
            *coord1* -- the most reliable mode, since it pins the true pitch
            from two real colonies rather than assuming symmetric margins.
            When omitted, spacing is computed from image dimensions assuming
            symmetric margins. Default: None.

        shape: Morphological footprint shape stamped at each grid position.
            ``"disk"`` (default) preserves round colony geometry.
            ``"square"`` covers rectangular well regions. ``"diamond"``
            offers a compromise between the two.

        width: Diameter of the footprint in pixels. Larger values cover
            more area per grid cell (risking overlap of adjacent cells);
            smaller values produce tighter, more precise masks. A
            reasonable starting point is roughly the visible colony
            diameter, then shrink it if neighbouring stamps merge or grow
            it if colonies overflow their stamp. Typical range: 5--50,
            depending on image resolution and colony size. Default: 15.

    Returns:
        GridImage: Input image with ``objmask`` set to the union of all
        stamped footprints and ``objmap`` set to uniquely labelled regions
        (1-indexed, row-major order).

    Raises:
        GridImageInputError: If a plain Image is passed instead of a
            GridImage.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies`
            Step-by-step tutorial for basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared`
            In-depth comparison of all detection strategies.
    """

    coord1: tuple[int, int] = (0, 0)
    coord2: tuple[int, int] | None = None
    shape: Literal["square", "diamond", "disk"] = "disk"
    width: int = 15

    def napari(self, image: GridImage) -> ManualGridPointDetector:
        """Interactively pick 1--2 anchor coordinates using a napari viewer.

        Opens a blocking napari viewer displaying the plate image layers
        (RGB, grayscale, detection matrix). Click up to two points to
        define the grid anchor positions, then click **Confirm** in the
        dock widget. The picked coordinates update *coord1* and
        *coord2* on this detector instance.

        Args:
            image: The GridImage to display for coordinate selection.

        Returns:
            ManualGridPointDetector: Self, for method chaining.
        """
        from phenotypic.tools_.napari_ import PointPickerWidget

        points = PointPickerWidget(max_points=2).run(image)

        if len(points) >= 1:
            self.coord1 = (int(round(points[0][0])), int(round(points[0][1])))
        if len(points) >= 2:
            self.coord2 = (int(round(points[1][0])), int(round(points[1][1])))
        elif len(points) == 1:
            self.coord2 = None

        return self

    def _operate(self, image: GridImage) -> GridImage:  # type: ignore[override]
        h, w = image.shape[:2]
        nrows, ncols = image.nrows, image.ncols
        y1, x1 = self.coord1

        if self.coord2 is not None:
            y2, x2 = self.coord2
            row_spacing = y2 - y1
            col_spacing = x2 - x1
        else:
            row_spacing = (h - 2 * y1) / max(1, nrows - 1)
            col_spacing = (w - 2 * x1) / max(1, ncols - 1)

        fp_mask = self._make_footprint(shape=self.shape, width=self.width).astype(bool)
        mask = np.zeros((h, w), dtype=bool)
        labeled = np.zeros((h, w), dtype=np.int32)

        for i in range(nrows):
            for j in range(ncols):
                cy = int(round(y1 + i * row_spacing))
                cx = int(round(x1 + j * col_spacing))
                self._stamp_footprint(
                        mask, labeled, fp_mask, cy, cx, i * ncols + j + 1,
                )

        image.objmask[:] = mask
        image.objmap[:] = labeled
        return image


ManualGridPointDetector.apply.__doc__ = ManualGridPointDetector._operate.__doc__
