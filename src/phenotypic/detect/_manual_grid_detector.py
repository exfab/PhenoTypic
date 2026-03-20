from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

from phenotypic.abc_ import GridObjectDetector
from phenotypic.tools_.mixin._footprint_mixin import FootprintMixin


class ManualGridDetector(GridObjectDetector, FootprintMixin):
    """Place footprint masks at evenly-spaced grid positions from reference coordinates.

    Args:
        coord1: (y, x) pixel position of the top-left grid cell center (row 0,
            column 0). This is the anchor point from which all other grid positions
            are calculated.

        coord2: Optional (y, x) pixel position of the diagonally adjacent cell
            (row 1, column 1). When provided, row and column spacing are derived
            from the difference between coord2 and coord1. When omitted, spacing
            is computed from image dimensions assuming symmetric margins.

        shape: Morphological footprint shape stamped at each grid position.
            ``"disk"`` preserves round colony geometry, ``"square"`` covers
            rectangular well regions, ``"diamond"`` offers a compromise.

        width: Diameter of the footprint in pixels. Controls the size of each
            stamped mask region. Larger values cover more area per grid cell;
            smaller values produce tighter, more precise masks.

    Returns:
        GridImage: Input image with objmask set to the union of all stamped
            footprints and objmap set to uniquely labeled regions (1-indexed,
            row-major order).

    Raises:
        GridImageInputError: If a plain Image is passed instead of GridImage.

    ManualGridDetector computes a regular grid of positions from one or two
    user-supplied reference coordinates, then stamps a morphological footprint
    at each position to produce objmask and objmap. This is useful when colony
    positions follow a known regular pattern but automatic grid detection is
    unreliable or unnecessary.

    **One-coordinate mode:** coord1 defines the top-left cell center. The grid
    is assumed to have symmetric margins — the spacing is calculated so that
    the last row/column center mirrors coord1's distance from the opposite
    image edge. Row spacing = ``(H - 2*y) / (nrows - 1)``, column spacing =
    ``(W - 2*x) / (ncols - 1)``.

    **Two-coordinate mode:** coord1 and coord2 define adjacent cells (0,0) and
    (1,1). Row spacing = ``coord2[0] - coord1[0]``, column spacing =
    ``coord2[1] - coord1[1]``. The grid is extrapolated from coord1 using
    these spacings for all nrows x ncols positions.

    **Use cases**

    - **Manual grid specification:** When automatic grid finders fail due to
      low contrast, missing wells, or non-standard plate formats.
    - **Synthetic mask generation:** Creating ground-truth masks for testing
      or validation of detection pipelines.
    - **Template-based detection:** When colony positions are known a priori
      from plate layout metadata or robotic spotting coordinates.
    - **Quick prototyping:** Rapidly generating grid masks without running
      full detection algorithms.

    **Limitations**

    - Assumes a perfectly regular grid. Cannot handle irregular spacing,
      rotated grids, or missing wells.
    - Does not use image content (detect_mat) — positions are purely geometric.
      Colonies that deviate from expected positions will be missed.
    - Footprints may extend beyond image bounds at edges; these are clipped
      silently.

    Examples:
        Single-coordinate mode using image dimensions:

        >>> import numpy as np
        >>> from phenotypic import GridImage
        >>> from phenotypic.detect import ManualGridDetector
        >>> arr = np.zeros((200, 300, 3), dtype=np.uint8)
        >>> grid_img = GridImage(arr=arr, nrows=4, ncols=6)
        >>> detector = ManualGridDetector(coord1=(25, 25), shape="disk", width=11)
        >>> result = detector.apply(grid_img)
        >>> result.objmask[:].any()
        True

        Two-coordinate mode with explicit spacing:

        >>> detector2 = ManualGridDetector(
        ...     coord1=(20, 30), coord2=(70, 80), shape="square", width=9
        ... )
        >>> result2 = detector2.apply(grid_img)
        >>> # Labels are 1-indexed, row-major: cell (0,0)=1, (0,1)=2, ...
        >>> int(result2.objmap[:].max()) == 4 * 6
        True
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
