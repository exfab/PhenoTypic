"""Radial geometry around a colony centre, using only NumPy."""

from __future__ import annotations

import numpy as np

#: Vertex count of a closed overlay circle (5-degree steps).
CIRCLE_VERTICES: int = 72


def circle_xy(
    cx: float,
    cy: float,
    radius: float,
    n_vertices: int = CIRCLE_VERTICES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xs, ys)`` for a closed circle in plot coordinates.

    The first vertex is repeated at the end so plotly draws a closed outline.

    Args:
        cx: Circle centre x (image column).
        cy: Circle centre y (image row).
        radius: Circle radius in pixels.
        n_vertices: Number of vertices including the repeated closing vertex.

    Returns:
        Two float64 arrays of length ``n_vertices``.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=True)
    return cx + radius * np.cos(theta), cy + radius * np.sin(theta)


def distance_from_point(
    shape: tuple[int, int], center_rc: tuple[float, float]
) -> np.ndarray:
    """Euclidean distance from each pixel to a point.

    Args:
        shape: ``(height, width)`` of the array.
        center_rc: ``(row, col)`` centre coordinates.

    Returns:
        Float64 array of distances with the given shape.
    """
    rows, cols = np.indices(shape)
    return np.sqrt((rows - center_rc[0]) ** 2 + (cols - center_rc[1]) ** 2)
