"""Tests for literal crossing diagnostic plot helpers."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver

from phenotypic.sdk_.orientation_fields import (
    LiteralSkeletonRingCrossing,
    LiteralSkeletonRingCrossingTransform,
    literal_crossing_ring_profile,
    plot_literal_crossing_map,
    plot_literal_crossing_outward_profile,
    plot_literal_crossing_population,
)

matplotlib.use("Agg")


def _transform() -> LiteralSkeletonRingCrossingTransform:
    """Return a small plotted transform with one crossing per quadrant."""
    center = (10.0, 10.0)
    specifications = (
        (10.0, 15.0, 0.0, 30.0),
        (15.0, 10.0, 0.5 * np.pi, -30.0),
        (10.0, 5.0, 0.0, 89.0),
        (5.0, 10.0, 0.5 * np.pi, -89.0),
    )
    crossings = tuple(
        LiteralSkeletonRingCrossing(
            point_id=index,
            ring_index=0,
            radius=5.0,
            row=row,
            col=col,
            anchor_row=int(row),
            anchor_col=int(col),
            fiber_axis=float(axis),
            radial_tilt=float(np.radians(tilt)),
            coherence=1.0,
            resultant=1.0,
            pixel_count=1,
        )
        for index, (row, col, axis, tilt) in enumerate(specifications)
    )
    skeleton = np.zeros((21, 21), dtype=bool)
    for point in crossings:
        skeleton[point.anchor_row, point.anchor_col] = True
    return LiteralSkeletonRingCrossingTransform(
        crossings=crossings,
        reliable_skeleton=skeleton,
        radii=np.asarray([5.0]),
        center=center,
        crossing_half_width=1.5,
    )


def test_crossing_map_returns_outward_normalized_quiver() -> None:
    """Arrowheads should use the documented increasing-radius convention."""
    transform = _transform()
    figure = Figure()
    axis = figure.subplots()

    arrows = plot_literal_crossing_map(
        axis,
        np.arange(21 * 21, dtype=float).reshape(21, 21),
        transform,
        boundary_radii=(3.0, 7.0),
    )

    assert isinstance(arrows, Quiver)
    radial_x = arrows.X - transform.center[1]
    radial_y = arrows.Y - transform.center[0]
    assert np.all(arrows.U * radial_x + arrows.V * radial_y >= 0.0)
    assert np.allclose(arrows.get_array(), [30.0, -30.0, 89.0, -89.0])
    assert len(axis.patches) == 3
    assert not axis.axison


def test_population_plot_has_crossings_consensus_and_legend() -> None:
    """The population diagnostic should expose raw and summarized evidence."""
    transform = _transform()
    figure = Figure()
    axis = figure.subplots()

    scatter = plot_literal_crossing_population(axis, transform)

    assert scatter.get_offsets().shape == (4, 2)
    assert len(axis.lines) == 2
    assert axis.get_legend() is not None
    assert axis.get_ylim() == pytest.approx((-95.0, 95.0))


def test_outward_profile_uses_separate_resultant_axis() -> None:
    """Change and ring reliability should be readable on separate scales."""
    transform = _transform()
    profile = literal_crossing_ring_profile(transform)
    figure = Figure()
    axis = figure.subplots()

    scatter, resultant_axis = plot_literal_crossing_outward_profile(
        axis, profile
    )

    assert scatter.get_offsets().shape == (1, 2)
    assert resultant_axis is not axis
    assert resultant_axis.get_ylim() == pytest.approx((0.0, 1.0))
    assert resultant_axis.get_ylabel() == "Axial resultant"


def test_population_plot_derives_profile_with_requested_guard() -> None:
    """The diagnostic must derive its consensus from the plotted transform."""
    transform = _transform()
    figure = Figure()
    axis = figure.subplots()

    plot_literal_crossing_population(axis, transform, minimum_points=5)

    consensus_line = axis.lines[0]
    assert np.isnan(consensus_line.get_ydata()[0])
