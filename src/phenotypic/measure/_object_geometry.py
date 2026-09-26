"""Per-object geometry shared by the measurers.

Each helper is the single home of a correctness rule that has been broken
before, so no measurer re-derives it inline:

* :func:`convex_hull_area` -- in 2-D, ``scipy.spatial.ConvexHull.volume`` is
  the hull's area and ``ConvexHull.area`` is its perimeter.
* :func:`object_edt` -- ``distance_transform_edt`` binarizes its input, so it
  must see one object's crop, never the labelled objmap; otherwise touching
  colonies see no background between them.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull, QhullError


def convex_hull_area(coords: np.ndarray) -> tuple[ConvexHull | None, float]:
    """Build an object's convex hull and return it with its area.

    Args:
        coords: ``(N, 2)`` pixel coordinates of one object
            (``regionprops.coords``).

    Returns:
        ``(hull, hull.volume)``, or ``(None, nan)`` when Qhull cannot build a
        hull (a single pixel or collinear pixels). ``volume`` is the area in
        2-D. The hull passes through pixel centres, so it is slightly smaller
        than the pixel count of the same shape.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Qhull")
            hull = ConvexHull(coords)
    except QhullError:
        return None, float("nan")
    return hull, float(hull.volume)


def object_edt(obj_mask: np.ndarray) -> np.ndarray:
    """Euclidean distance transform of one object's crop, background-padded.

    Args:
        obj_mask: Boolean mask of a single object within its bounding box
            (``regionprops.image``), already isolated from neighbouring labels.

    Returns:
        The distance from each pixel to the nearest background pixel, at the
        shape of ``obj_mask``. The crop is padded by one background pixel on
        every side first, so the bounding box and the image border both
        count as an edge.
    """
    # asarray narrows the scipy stub's tuple-or-ndarray return union; with
    # return_indices=False the call yields an ndarray and copies nothing.
    return np.asarray(distance_transform_edt(np.pad(obj_mask, 1)))[1:-1, 1:-1]
