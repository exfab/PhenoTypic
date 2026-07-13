"""Grey-weighted distance helpers derived from APP2."""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray


Connectivity = Literal[4, 8]

_NEIGHBORS_4: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, 0, 1.0),
)
_NEIGHBORS_8: tuple[tuple[int, int, float], ...] = (
    (-1, -1, math.sqrt(2.0)),
    (-1, 0, 1.0),
    (-1, 1, math.sqrt(2.0)),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, math.sqrt(2.0)),
    (1, 0, 1.0),
    (1, 1, math.sqrt(2.0)),
)
_SOURCE_INFINITY = np.float32(1e20)
_GIVALS = np.array(
    [
        22026.5, 20368, 18840.3, 17432.5, 16134.8, 14938.4, 13834.9, 12816.8,
        11877.4, 11010.2, 10209.4, 9469.8, 8786.47, 8154.96, 7571.17, 7031.33,
        6531.99, 6069.98, 5642.39, 5246.52, 4879.94, 4540.36, 4225.71, 3934.08,
        3663.7, 3412.95, 3180.34, 2964.5, 2764.16, 2578.14, 2405.39, 2244.9,
        2095.77, 1957.14, 1828.24, 1708.36, 1596.83, 1493.05, 1396.43, 1306.47,
        1222.68, 1144.62, 1071.87, 1004.06, 940.819, 881.837, 826.806, 775.448,
        727.504, 682.734, 640.916, 601.845, 565.329, 531.193, 499.271, 469.412,
        441.474, 415.327, 390.848, 367.926, 346.454, 326.336, 307.481, 289.804,
        273.227, 257.678, 243.089, 229.396, 216.541, 204.469, 193.129, 182.475,
        172.461, 163.047, 154.195, 145.868, 138.033, 130.659, 123.717, 117.179,
        111.022, 105.22, 99.7524, 94.5979, 89.7372, 85.1526, 80.827, 76.7447,
        72.891, 69.2522, 65.8152, 62.5681, 59.4994, 56.5987, 53.856, 51.2619,
        48.8078, 46.4854, 44.2872, 42.2059, 40.2348, 38.3676, 36.5982, 34.9212,
        33.3313, 31.8236, 30.3934, 29.0364, 27.7485, 26.526, 25.365, 24.2624,
        23.2148, 22.2193, 21.273, 20.3733, 19.5176, 18.7037, 17.9292, 17.192,
        16.4902, 15.822, 15.1855, 14.579, 14.0011, 13.4503, 12.9251, 12.4242,
        11.9464, 11.4905, 11.0554, 10.6401, 10.2435, 9.86473, 9.50289, 9.15713,
        8.82667, 8.51075, 8.20867, 7.91974, 7.64333, 7.37884, 7.12569, 6.88334,
        6.65128, 6.42902, 6.2161, 6.01209, 5.81655, 5.62911, 5.44938, 5.27701,
        5.11167, 4.95303, 4.80079, 4.65467, 4.51437, 4.37966, 4.25027, 4.12597,
        4.00654, 3.89176, 3.78144, 3.67537, 3.57337, 3.47528, 3.38092, 3.29013,
        3.20276, 3.11868, 3.03773, 2.9598, 2.88475, 2.81247, 2.74285, 2.67577,
        2.61113, 2.54884, 2.48881, 2.43093, 2.37513, 2.32132, 2.26944, 2.21939,
        2.17111, 2.12454, 2.07961, 2.03625, 1.99441, 1.95403, 1.91506, 1.87744,
        1.84113, 1.80608, 1.77223, 1.73956, 1.70802, 1.67756, 1.64815, 1.61976,
        1.59234, 1.56587, 1.54032, 1.51564, 1.49182, 1.46883, 1.44664, 1.42522,
        1.40455, 1.3846, 1.36536, 1.3468, 1.3289, 1.31164, 1.29501, 1.27898,
        1.26353, 1.24866, 1.23434, 1.22056, 1.2073, 1.19456, 1.18231, 1.17055,
        1.15927, 1.14844, 1.13807, 1.12814, 1.11864, 1.10956, 1.10089, 1.09262,
        1.08475, 1.07727, 1.07017, 1.06345, 1.05709, 1.05109, 1.04545, 1.04015,
        1.03521, 1.0306, 1.02633, 1.02239, 1.01878, 1.0155, 1.01253, 1.00989,
        1.00756, 1.00555, 1.00385, 1.00246, 1.00139, 1.00062, 1.00015, 1,
    ],
    dtype=np.float64,
)


def _validated_real_image(array: np.ndarray, *, name: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must have a real numeric dtype")

    values = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError(f"{name} must contain only nonnegative values")
    return values


def _validated_background(background: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(background, np.ndarray):
        raise TypeError("background must be a numpy.ndarray")
    if background.dtype != np.bool_:
        raise TypeError("background must have boolean dtype")
    if background.ndim != 2 or background.shape != shape:
        raise ValueError("background must have the same two-dimensional shape as image")
    return background


def grey_weighted_distance(
    image: np.ndarray,
    background: np.ndarray,
    *,
    connectivity: Connectivity = 8,
) -> np.ndarray:
    r"""Compute Vaa3D APP2's grey-weighted distance from background pixels.

    Background pixels start at their input intensity. APP2 initializes their immediate
    frontier with destination intensity but no geometric factor. Subsequent moves into
    pixel :math:`q` cost :math:`I(q)\lVert q-p\rVert_2`. The initialization asymmetry
    is intentional source behavior, not a conventional shortest-path simplification.
    Coordinates are ``(row, column)`` on a unit-pixel grid; neighbors outside the array
    are clipped. Units are intensity-pixels after the frontier. Higher input intensity
    and longer in-image paths increase distance. Strict ``<`` updates retain an existing
    equal trial distance; no parent or path is exposed, so equal-path order is not public.

    Args:
        image: Finite, nonnegative, real-valued 2-D intensity array.
        background: Boolean seed mask with the same shape as ``image``.
        connectivity: Four-neighbor or eight-neighbor movement.

    Returns:
        A same-shape float32 array containing cumulative weighted distance. With no
        background seed, every value is the Vaa3D float32 sentinel ``1e20``.

    Raises:
        TypeError: An input has the wrong container or dtype.
        ValueError: An input has an invalid shape or value, or connectivity is invalid.
    """
    values = _validated_real_image(image, name="image")
    seeds = _validated_background(background, values.shape)
    if (
        isinstance(connectivity, (bool, np.bool_))
        or not isinstance(connectivity, (int, np.integer))
        or connectivity not in (4, 8)
    ):
        raise ValueError("connectivity must be 4 or 8")
    source_values: NDArray[np.float32] = values.astype(np.float32)
    distances = np.full(values.shape, _SOURCE_INFINITY, dtype=np.float32)
    distances[seeds] = source_values[seeds]
    alive = seeds.copy()
    trial = np.zeros(values.shape, dtype=bool)
    counter = itertools.count()
    heap: list[tuple[float, int, int, int]] = []
    neighbors = _NEIGHBORS_4 if connectivity == 4 else _NEIGHBORS_8
    rows, columns = values.shape

    for seed_row, seed_column in np.argwhere(seeds):
        for row_offset, column_offset, step_length in neighbors:
            row = int(seed_row) + row_offset
            column = int(seed_column) + column_offset
            if row < 0 or row >= rows or column < 0 or column >= columns:
                continue
            if alive[row, column] or trial[row, column]:
                continue
            minimum_row = int(seed_row)
            minimum_column = int(seed_column)
            if distances[minimum_row, minimum_column] > 0.0:
                for alive_row_offset, alive_column_offset, _ in neighbors:
                    alive_row = row + alive_row_offset
                    alive_column = column + alive_column_offset
                    if (
                        0 <= alive_row < rows
                        and 0 <= alive_column < columns
                        and alive[alive_row, alive_column]
                        and distances[alive_row, alive_column]
                        < distances[minimum_row, minimum_column]
                    ):
                        minimum_row = alive_row
                        minimum_column = alive_column
            candidate = np.float32(
                float(distances[minimum_row, minimum_column])
                + float(source_values[row, column])
            )
            distances[row, column] = candidate
            trial[row, column] = True
            heapq.heappush(heap, (float(candidate), next(counter), row, column))

    while heap:
        distance, _, row, column = heapq.heappop(heap)
        if alive[row, column] or distance != float(distances[row, column]):
            continue
        alive[row, column] = True
        trial[row, column] = False

        for row_offset, column_offset, step_length in neighbors:
            neighbor_row = row + row_offset
            neighbor_column = column + column_offset
            if (
                neighbor_row < 0
                or neighbor_row >= rows
                or neighbor_column < 0
                or neighbor_column >= columns
                or alive[neighbor_row, neighbor_column]
            ):
                continue
            candidate = np.float32(
                distance
                + float(source_values[neighbor_row, neighbor_column]) * step_length
            )
            if candidate < distances[neighbor_row, neighbor_column]:
                distances[neighbor_row, neighbor_column] = candidate
                trial[neighbor_row, neighbor_column] = True
                heapq.heappush(
                    heap,
                    (float(candidate), next(counter), neighbor_row, neighbor_column),
                )

    return distances


def app2_gwdt_cost(
    distance: np.ndarray,
) -> np.ndarray:
    """Apply Vaa3D APP2's fixed, quantized inverse-intensity lookup.

    Vaa3D normalizes the GWDT range, truncates it into 256 bins, and indexes the fixed
    ``givals`` table corresponding to an exponential strength of ten. Constant maps
    have no defined source lookup; this adapter returns ones deterministically. The
    transform is pointwise on the same ``(row, column)`` grid and has no boundary rule.
    Costs are unitless, largest at the minimum distance, and smallest at the maximum.
    Equal normalized values use the same truncating lookup bin.

    Args:
        distance: Finite, nonnegative, real-valued 2-D GWDT map.
    Returns:
        A same-shape float64 cost array with one of the 256 source lookup values.

    Raises:
        TypeError: An input has the wrong container or dtype.
        ValueError: An input has an invalid shape or value.
    """
    values = _validated_real_image(distance, name="distance")
    minimum = float(np.min(values))
    span = float(np.max(values)) - minimum
    if span == 0.0:
        return np.ones(values.shape, dtype=np.float64)
    lookup_indices = ((values - minimum) / span * 255.0).astype(np.intp)
    return _GIVALS[lookup_indices]


__all__ = ["app2_gwdt_cost", "grey_weighted_distance"]
