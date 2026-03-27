from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

from phenotypic.abc_ import GridObjectDetector
from phenotypic.tools_.mixin._footprint_mixin import FootprintMixin


class ManualGridDetector(GridObjectDetector, FootprintMixin):
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
            (row 0, column 0). This is the anchor point from which all
            other positions are calculated. Default ``(0, 0)``.

        coord2: Optional ``(y, x)`` pixel position of the diagonally
            adjacent cell (row 1, column 1). When provided, row and column
            spacing are derived from the difference between *coord2* and
            *coord1*. When omitted, spacing is computed from image
            dimensions assuming symmetric margins.

        shape: Morphological footprint shape stamped at each grid position.
            ``"disk"`` (default) preserves round colony geometry.
            ``"square"`` covers rectangular well regions. ``"diamond"``
            offers a compromise between the two.

        width: Diameter of the footprint in pixels (default 15). Larger
            values cover more area per grid cell; smaller values produce
            tighter, more precise masks. Typical range: 5--50, depending
            on image resolution and colony size.

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

    def __init__(
        self,
        coord1: tuple[int, int] = (0, 0),
        coord2: tuple[int, int] | None = None,
        shape: Literal["square", "diamond", "disk"] = "disk",
        width: int = 15,
    ):
        super().__init__()
        self.coord1 = coord1
        self.coord2 = coord2
        self.shape = shape
        self.width = width

    def __setattr__(self, name: str, value: object) -> None:
        if name in ("coord1", "coord2") and value is not None:
            value = tuple(value)  # type: ignore[arg-type]
        super().__setattr__(name, value)

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

        footprint = self._make_footprint(shape=self.shape, width=self.width)
        fp_h, fp_w = footprint.shape
        fp_mask = footprint.astype(bool)

        mask = np.zeros((h, w), dtype=bool)
        labeled = np.zeros((h, w), dtype=np.int32)

        for i in range(nrows):
            for j in range(ncols):
                cy = int(round(y1 + i * row_spacing))
                cx = int(round(x1 + j * col_spacing))

                # Image region bounds (clipped to image)
                img_y0 = max(0, cy - fp_h // 2)
                img_y1 = min(h, cy - fp_h // 2 + fp_h)
                img_x0 = max(0, cx - fp_w // 2)
                img_x1 = min(w, cx - fp_w // 2 + fp_w)

                # Corresponding footprint region
                fp_y0 = img_y0 - (cy - fp_h // 2)
                fp_x0 = img_x0 - (cx - fp_w // 2)
                fp_slice = fp_mask[
                    fp_y0 : fp_y0 + (img_y1 - img_y0),
                    fp_x0 : fp_x0 + (img_x1 - img_x0),
                ]

                mask[img_y0:img_y1, img_x0:img_x1] |= fp_slice

                label_id = i * ncols + j + 1
                region = labeled[img_y0:img_y1, img_x0:img_x1]
                labeled[img_y0:img_y1, img_x0:img_x1] = np.where(
                    fp_slice, label_id, region
                )

        image.objmask[:] = mask
        image.objmap[:] = labeled
        return image


ManualGridDetector.apply.__doc__ = ManualGridDetector._operate.__doc__
