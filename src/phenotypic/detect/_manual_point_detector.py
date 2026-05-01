from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectDetector
from phenotypic.tools_.mixin._footprint_mixin import FootprintMixin
from phenotypic.tools_.mixin._point_picker_mixin import PointPickerMixin


class ManualPointDetector(ObjectDetector, PointPickerMixin, FootprintMixin):
    """Detect objects by stamping footprint masks at user-specified coordinates.

    Place a morphological footprint at each explicitly provided ``(y, x)``
    centre coordinate to produce ``objmask`` and ``objmap``. Unlike
    :class:`ManualGridDetector`, which extrapolates a regular grid from one
    or two anchor points, this class requires an explicit list of *every*
    colony centre. This makes it suitable for plates without regular
    geometry, sparse or irregular layouts, and manual annotation workflows
    where each object position is known individually.

    Args:
        centers: An N x 2 array-like of ``(y, x)`` pixel coordinates
            specifying each colony centre. Accepts any sequence that
            ``np.asarray`` can convert (list of tuples, nested list, or
            NumPy array). When *None* or empty, :meth:`apply` zeros out
            ``objmask`` and ``objmap`` and returns immediately.

        shape: Morphological footprint shape stamped at each coordinate.
            ``"disk"`` (default) preserves round colony geometry.
            ``"square"`` covers rectangular regions. ``"diamond"`` offers
            a compromise between the two.

        width: Diameter of the footprint in pixels (default 15). Larger
            values cover more area per colony; smaller values produce
            tighter, more precise masks. Typical range: 5--50, depending
            on image resolution and colony size.

    Returns:
        Image: Input image with ``objmask`` set to the union of all
        stamped footprints and ``objmap`` set to uniquely labelled regions
        (1-indexed, in the order centres were supplied).

    Best For:
        * Manual annotation and ground-truth mask generation for
          benchmarking detection algorithms.
        * Non-grid plates (e.g., streak plates, random inoculations,
          environmental samples) where colony positions are irregular.
        * Validating other detection algorithms by comparing their output
          against user-curated centre coordinates.
        * Quick prototyping on small numbers of colonies without needing
          automatic detection.

    Consider Also:
        * :class:`ManualGridDetector` when colonies lie on a regular grid
          and only one or two anchor coordinates are needed.
        * :class:`RoundPeaksDetector` when colony centres can be inferred
          automatically from intensity profiles.

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
        centers: np.ndarray | list | None = None,
        shape: Literal["square", "diamond", "disk"] = "disk",
        width: int = 15,
    ):
        super().__init__()
        self.centers = centers
        self.shape = shape
        self.width = width

    def _operate(self, image: Image) -> Image:  # type: ignore[override]
        h, w = image.shape[:2]

        if self.centers is None or len(self.centers) == 0:
            image.objmask[:] = np.zeros((h, w), dtype=bool)
            image.objmap[:] = np.zeros((h, w), dtype=np.int32)
            return image

        fp_mask = self._make_footprint(shape=self.shape, width=self.width).astype(bool)
        mask = np.zeros((h, w), dtype=bool)
        labeled = np.zeros((h, w), dtype=np.int32)

        for idx, (cy, cx) in enumerate(self.centers, start=1):
            self._stamp_footprint(
                mask, labeled, fp_mask,
                int(round(cy)), int(round(cx)), idx,
            )

        image.objmask[:] = mask
        image.objmap[:] = labeled
        return image


ManualPointDetector.apply.__doc__ = ManualPointDetector._operate.__doc__
