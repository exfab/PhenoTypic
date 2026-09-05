"""Matplotlib diagnostics for literal skeleton-ring orientation evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from collections.abc import Sequence

import numpy as np
# matplotlib is imported inside the methods that draw. A module-scope import
# put pyplot (541 ms, the single largest remaining cost) on the
# `import phenotypic` path for every run, most of which draw nothing.
from numpy.typing import NDArray

from ._literal_crossings import (
    LiteralCrossingRingProfile,
    LiteralSkeletonRingCrossingTransform,
    literal_crossing_ring_profile,
)

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.colors import Normalize
    from matplotlib.collections import PathCollection
    from matplotlib.quiver import Quiver


def _validated_image(
    image: NDArray[np.floating],
    transform: LiteralSkeletonRingCrossingTransform,
) -> NDArray[np.floating]:
    """Return a two-dimensional image matching the transform skeleton."""
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("image must be a 2-D array")
    if array.shape != transform.reliable_skeleton.shape:
        raise ValueError(
            "image and transform skeleton must have the same shape, got "
            f"{array.shape} and {transform.reliable_skeleton.shape}"
        )
    if not np.isfinite(array).any():
        raise ValueError("image must contain at least one finite value")
    return array


def plot_literal_crossing_map(
    axis: Axes,
    image: NDArray[np.floating],
    transform: LiteralSkeletonRingCrossingTransform,
    *,
    cmap: str | Colormap = "twilight_shifted",
    norm: Normalize | None = None,
    arrow_length: float = 7.0,
    show_skeleton: bool = True,
    show_rings: bool = True,
    boundary_radii: Sequence[float] = (),
    image_percentiles: tuple[float, float] = (1.0, 99.8),
    title: str | None = None,
) -> Quiver:
    """Plot outward-normalized local orientation arrows on the source image.

    The measured fiber orientation is axial and has no intrinsic head or tail.
    Each plotted vector is flipped, when necessary, so that its radial dot
    product is nonnegative. Arrowheads therefore indicate increasing radius by
    construction, not measured growth polarity.

    Args:
        axis: Matplotlib axis to draw on.
        image: Two-dimensional source image used for orientation estimation.
        transform: Literal crossing transform to visualize.
        cmap: Cyclic Matplotlib colormap for signed radial-relative tilt. The
            default gives the equivalent ``-90`` and ``+90`` axial seam matching
            endpoint colors. Non-cyclic maps such as ``Spectral`` remain
            available but introduce a display discontinuity at that seam.
        norm: Color normalization in degrees. Defaults to ``[-90, 90]``.
        arrow_length: Arrow length in image pixels.
        show_skeleton: Overlay the reliable measurement skeleton.
        show_rings: Draw every sampled ring center.
        boundary_radii: Additional emphasized radial boundaries in pixels.
        image_percentiles: Percentiles used for grayscale display limits.
        title: Optional axis title.

    Returns:
        Quiver artist whose scalar array is radial-relative tilt in degrees.

    Raises:
        ValueError: If image or plotting parameters are invalid.
    """
    from matplotlib.patches import Circle
    from matplotlib.colors import Normalize

    source = _validated_image(image, transform)
    if not np.isfinite(arrow_length) or arrow_length <= 0.0:
        raise ValueError("arrow_length must be finite and > 0")
    percentiles = np.asarray(image_percentiles, dtype=float)
    if (
        percentiles.shape != (2,)
        or not np.isfinite(percentiles).all()
        or not 0.0 <= percentiles[0] < percentiles[1] <= 100.0
    ):
        raise ValueError(
            "image_percentiles must satisfy 0 <= low < high <= 100"
        )
    validated_boundaries = np.asarray(boundary_radii, dtype=float)
    if validated_boundaries.ndim != 1 or (
        validated_boundaries.size
        and (
            not np.isfinite(validated_boundaries).all()
            or np.any(validated_boundaries < 0.0)
        )
    ):
        raise ValueError(
            "boundary_radii must contain finite nonnegative values"
        )
    resolved_norm = Normalize(vmin=-90.0, vmax=90.0) if norm is None else norm
    finite = source[np.isfinite(source)]
    display_limits = np.asarray(
        np.percentile(finite, percentiles), dtype=float
    )
    low = float(display_limits[0])
    high = float(display_limits[1])
    axis.imshow(source, cmap="gray", vmin=low, vmax=high)
    if show_skeleton:
        skeleton = transform.reliable_skeleton
        axis.imshow(
            np.ma.masked_where(~skeleton, skeleton),
            cmap="gray",
            vmin=0,
            vmax=1,
            alpha=0.22,
        )
    center_row, center_col = transform.center
    if show_rings:
        for radius in transform.radii:
            axis.add_patch(
                Circle(
                    (center_col, center_row),
                    float(radius),
                    fill=False,
                    color="white",
                    linewidth=0.3,
                    alpha=0.16,
                )
            )
    for radius in validated_boundaries:
        axis.add_patch(
            Circle(
                (center_col, center_row),
                float(radius),
                fill=False,
                color="#F0F0F0",
                linewidth=1.0,
                alpha=0.72,
            )
        )

    rows = np.asarray(
        [point.row for point in transform.crossings], dtype=float
    )
    cols = np.asarray(
        [point.col for point in transform.crossings], dtype=float
    )
    values = np.degrees(
        np.asarray(
            [point.radial_tilt for point in transform.crossings], dtype=float
        )
    )
    horizontal = np.cos(
        np.asarray(
            [point.fiber_axis for point in transform.crossings], dtype=float
        )
    )
    vertical = np.sin(
        np.asarray(
            [point.fiber_axis for point in transform.crossings], dtype=float
        )
    )
    outward_dot = horizontal * (cols - center_col) + vertical * (
        rows - center_row
    )
    reverse = outward_dot < 0.0
    horizontal[reverse] *= -1.0
    vertical[reverse] *= -1.0
    arrows = axis.quiver(
        cols,
        rows,
        arrow_length * horizontal,
        arrow_length * vertical,
        values,
        cmap=cmap,
        norm=resolved_norm,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        units="xy",
        width=0.65,
        headwidth=3.3,
        headlength=4.2,
        headaxislength=3.7,
        pivot="middle",
        alpha=0.95,
    )
    axis.set_xlim(-0.5, source.shape[1] - 0.5)
    axis.set_ylim(source.shape[0] - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.text(
        0.01,
        0.01,
        "Arrowheads point outward by display convention; tangential polarity is ambiguous\n"
        "+/-90 degrees are one axial seam; non-cyclic colors split that seam",
        transform=axis.transAxes,
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
        verticalalignment="bottom",
    )
    if title is not None:
        axis.set_title(title)
    return arrows


def plot_literal_crossing_population(
    axis: Axes,
    transform: LiteralSkeletonRingCrossingTransform,
    *,
    minimum_points: int = 3,
    minimum_resultant: float = 0.15,
    cmap: str | Colormap = "twilight_shifted",
    norm: Normalize | None = None,
    title: str | None = None,
) -> PathCollection:
    """Plot every crossing tilt and the equal-crossing ring consensus.

    Args:
        axis: Matplotlib axis to draw on.
        transform: Literal crossing evidence to scatter.
        minimum_points: Smallest crossing population accepted at one ring.
        minimum_resultant: Smallest accepted ring-level axial resultant.
        cmap: Cyclic Matplotlib colormap for signed local tilt. The default
            matches colors at the equivalent ``-90`` and ``+90`` axial seam.
        norm: Color normalization in degrees. Defaults to ``[-90, 90]``.
        title: Optional axis title.

    Returns:
        Crossing scatter artist suitable for a colorbar.

    The ring profile is derived internally so raw crossings cannot be paired
    with a profile calculated from another transform.
    """
    from matplotlib.colors import Normalize

    profile = literal_crossing_ring_profile(
        transform,
        minimum_points=minimum_points,
        minimum_resultant=minimum_resultant,
    )
    resolved_norm = Normalize(vmin=-90.0, vmax=90.0) if norm is None else norm
    radii = np.asarray(
        [point.radius for point in transform.crossings], dtype=float
    )
    tilts = np.degrees(
        np.asarray(
            [point.radial_tilt for point in transform.crossings], dtype=float
        )
    )
    scatter = axis.scatter(
        radii,
        tilts,
        c=tilts,
        cmap=cmap,
        norm=resolved_norm,
        s=12,
        alpha=0.42,
        linewidths=0,
    )
    axis.plot(
        profile.radii,
        np.degrees(profile.consensus_tilt),
        color="black",
        marker="o",
        markersize=4.5,
        linestyle="none",
        label="equal-crossing ring consensus",
    )
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_ylim(-95.0, 95.0)
    axis.set_xlabel("Radius from inoculum center (pixels)")
    axis.set_ylabel("Tilt (degrees; 0 radial, +/-90 tangential)")
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(alpha=0.2)
    axis.text(
        0.01,
        0.02,
        "+ clockwise / - counterclockwise in the image view",
        transform=axis.transAxes,
        fontsize=8,
        color="#444444",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        verticalalignment="bottom",
    )
    axis.text(
        0.01,
        0.07,
        "+/-90 degrees meet at the same axial seam",
        transform=axis.transAxes,
        fontsize=8,
        color="#444444",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        verticalalignment="bottom",
    )
    if title is not None:
        axis.set_title(title)
    return scatter


def plot_literal_crossing_outward_profile(
    axis: Axes,
    profile: LiteralCrossingRingProfile,
    *,
    resultant_axis: Axes | None = None,
    cmap: str | Colormap = "Spectral",
    norm: Normalize | None = None,
    title: str | None = None,
) -> tuple[PathCollection, Axes]:
    """Plot contiguous-run outward change and ring-level reliability.

    Args:
        axis: Matplotlib axis for accumulated orientation change.
        profile: Equal-crossing outward orientation profile.
        resultant_axis: Optional secondary axis for the ring resultant. A twin
            axis is created when omitted.
        cmap: Matplotlib colormap for signed accumulated change.
        norm: Color normalization in degrees. Defaults to ``[-180, 180]``.
        title: Optional axis title.

    Returns:
        Tuple of the change scatter artist and resultant axis.
    """
    from matplotlib.colors import Normalize

    resolved_norm = (
        Normalize(vmin=-180.0, vmax=180.0) if norm is None else norm
    )
    supported = profile.supported
    change_degrees = np.degrees(profile.contiguous_change)
    supported_run_ids = np.unique(profile.run_id[profile.run_id >= 0])
    for current_run_id in supported_run_ids:
        in_run = profile.run_id == current_run_id
        axis.plot(
            profile.radii[in_run],
            change_degrees[in_run],
            color="#202020",
            marker="o",
            markersize=4.5,
            linewidth=1.6,
        )
        run_start = int(np.flatnonzero(in_run)[0])
        if current_run_id > 0:
            axis.scatter(
                [profile.radii[run_start]],
                [change_degrees[run_start]],
                marker="s",
                s=72,
                facecolors="none",
                edgecolors="#202020",
                linewidths=1.0,
                zorder=4,
            )
    scatter = axis.scatter(
        profile.radii[supported],
        change_degrees[supported],
        c=change_degrees[supported],
        cmap=cmap,
        norm=resolved_norm,
        s=46,
        edgecolors="black",
        linewidths=0.35,
        zorder=3,
    )
    reliability_axis = (
        axis.twinx() if resultant_axis is None else resultant_axis
    )
    reliability_axis.plot(
        profile.radii,
        profile.resultant,
        color="#777777",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
    )
    reliability_axis.set_ylim(0.0, 1.0)
    reliability_axis.set_ylabel("Axial resultant", color="#666666")
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_xlabel("Radius from inoculum center (pixels)")
    axis.set_ylabel("Consensus change within supported run (degrees)")
    axis.grid(alpha=0.2)
    axis.text(
        0.01,
        0.02,
        "Each run is zero-relative; open squares mark restarts after gaps or 90-degree ambiguity",
        transform=axis.transAxes,
        fontsize=8,
        color="#444444",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        verticalalignment="bottom",
    )
    if title is not None:
        axis.set_title(title)
    return scatter, reliability_axis
