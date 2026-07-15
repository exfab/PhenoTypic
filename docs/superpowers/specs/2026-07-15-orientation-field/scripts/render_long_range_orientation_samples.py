from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import numpy as np

from neurospora_orientation_samples import reproduce_notebook_segmentation
from phenotypic.measure import MeasureOrientationZones
from phenotypic.measure._measure_orientation_zones import (
    long_range_ring_rotation_profile,
    radial_ring_orientation_profile,
    signed_radial_relative_field,
)
from phenotypic.measure._zone_segmentation import (
    compute_zone_segmentation,
    distance_from_point,
)
from phenotypic.util._orientation_field import orientation_field


OUTPUT_DIR = Path(
    "/Users/alex/.codex/visualizations/2026/07/15/"
    "019f6340-b68c-7a81-b738-983ed6ea1a27/orientation-real-image"
)
COLONIES = (
    ("A", 1116, 35),
    ("B", 626, 18),
)
RING_WIDTH = 8.0
RADIAL_LAG = 16.0
N_SECTORS = 36


def _cell_summaries(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    absolute = np.full(cells.shape[0], np.nan)
    signed = np.full(cells.shape[0], np.nan)
    support = np.zeros(cells.shape[0])
    for index, row in enumerate(cells):
        finite = np.isfinite(row)
        support[index] = float(finite.sum()) / float(row.size)
        if finite.any():
            absolute[index] = float(np.mean(np.abs(row[finite])))
            signed[index] = float(np.mean(row[finite]))
    return absolute, signed, support


def _matrix_panel(
    figure,
    axis,
    x: np.ndarray,
    matrix_degrees: np.ndarray,
    title: str,
    colorbar_label: str,
    color_limit: float,
) -> None:
    if x.size == 0 or matrix_degrees.size == 0:
        axis.text(0.5, 0.5, "No reliable ring pairs", ha="center", va="center")
        axis.set_title(title)
        axis.set_axis_off()
        return
    x_step = float(np.median(np.diff(x))) if x.size > 1 else RING_WIDTH
    x_edges = np.concatenate(
        ([x[0] - 0.5 * x_step], 0.5 * (x[:-1] + x[1:]), [x[-1] + 0.5 * x_step])
    )
    sector_edges = np.linspace(0.0, 360.0, N_SECTORS + 1)
    colormap = matplotlib.colormaps["Spectral"].copy()
    colormap.set_bad((0.93, 0.93, 0.93, 1.0))
    image = axis.pcolormesh(
        x_edges,
        sector_edges,
        np.ma.masked_invalid(matrix_degrees.T),
        cmap=colormap,
        norm=Normalize(-color_limit, color_limit),
        shading="flat",
    )
    axis.set_title(title, loc="left", fontsize=11)
    axis.set_xlabel("Radius from inoculum center (px)")
    axis.set_ylabel("Polar sector (degrees)")
    axis.set_yticks([0, 90, 180, 270, 360])
    bar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.02)
    bar.set_label(colorbar_label)


def render_colony(
    segmented,
    colony_name: str,
    label: int,
    section_index: int,
) -> Path:
    section = segmented.grid[section_index]
    objmap = section.objmap[:]
    section.objmap[:] = np.where(objmap == label, label, 0)
    operation = MeasureOrientationZones(
        radial_ring_width=RING_WIDTH,
        long_range_lag=RADIAL_LAG,
        quiver_block=24,
    )
    measurements = operation.measure(section)
    inspect_figure = operation.inspect(section)
    inspect_figure.write_image(
        str(
            OUTPUT_DIR
            / f"long_range_inspect_spectral_colony_{colony_name.lower()}.png"
        ),
        width=1400,
        height=1100,
        scale=1,
    )
    props, label_to_section = operation._prep(section)
    prop = max(props, key=lambda candidate: candidate.area)
    segmentation = compute_zone_segmentation(
        section,
        prop,
        params=operation._zone_params(),
    )
    tile, object_mask, centre = operation._resolve_tile(
        section,
        segmentation,
        prop,
        label_to_section,
    )
    phi, coherence, _gradient = orientation_field(
        tile,
        operation.sigma_d,
        operation.sigma_i,
    )
    distance = distance_from_point(tile.shape, centre)
    signed_tilt, signed_turning, _magnitude, polar = signed_radial_relative_field(
        phi,
        centre,
        distance,
    )
    core_radius = float(segmentation.core_end_radius)
    outer_radius = min(
        float(segmentation.sparse_end_radius),
        float(segmentation.symmetric_radius),
    )
    structure = (
        object_mask
        & (distance >= core_radius)
        & (distance < outer_radius)
    )
    reliable = structure & (coherence >= 0.15)
    ring_radii, sector_tilt, _sector_resultant = radial_ring_orientation_profile(
        signed_tilt,
        polar,
        coherence,
        distance,
        structure,
        core_radius,
        outer_radius,
        RING_WIDTH,
        N_SECTORS,
    )
    pair_midpoints, sector_rotation = long_range_ring_rotation_profile(
        ring_radii,
        sector_tilt,
        RADIAL_LAG,
    )

    figure = plt.figure(figsize=(18, 13), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.05, 1.0))
    image_axis = figure.add_subplot(grid[:, 0])
    lower, upper = np.percentile(tile, (1.0, 99.8))
    image_axis.imshow(tile, cmap="gray", vmin=float(lower), vmax=float(upper))
    turning_degrees = np.degrees(signed_turning)
    raw_limit = float(np.nanmax(np.abs(turning_degrees[reliable])))
    turning_layer = np.where(reliable, turning_degrees, np.nan)
    spectral = matplotlib.colormaps["Spectral"].copy()
    spectral.set_bad((0.0, 0.0, 0.0, 0.0))
    overlay = image_axis.imshow(
        np.ma.masked_invalid(turning_layer),
        cmap=spectral,
        vmin=-raw_limit,
        vmax=raw_limit,
        alpha=0.73,
    )
    centre_row, centre_col = centre
    for radius in ring_radii:
        image_axis.add_patch(
            Circle(
                (centre_col, centre_row),
                float(radius),
                fill=False,
                edgecolor="white",
                linewidth=0.65,
                alpha=0.78,
            )
        )
    for radius, color, style in (
        (core_radius, "#DC267F", ":"),
        (float(segmentation.dense_end_radius), "#56B4E9", "--"),
        (outer_radius, "#003660", "--"),
    ):
        image_axis.add_patch(
            Circle(
                (centre_col, centre_row),
                radius,
                fill=False,
                edgecolor=color,
                linewidth=1.8,
                linestyle=style,
            )
        )
    image_axis.set_title(
        f"A. Colony {colony_name}: local signed outward turning + orientation rings\n"
        f"Spectral range uses raw peak ±{raw_limit:.2f} deg/px; inoculum core is uncolored",
        loc="left",
        fontsize=12,
    )
    image_axis.set_axis_off()
    turning_bar = figure.colorbar(overlay, ax=image_axis, fraction=0.035, pad=0.015)
    turning_bar.set_label("Signed outward turning (deg/px)")

    tilt_axis = figure.add_subplot(grid[0, 1])
    _matrix_panel(
        figure,
        tilt_axis,
        ring_radii,
        np.degrees(sector_tilt),
        "B. Signed radial-relative tilt by ring and sector",
        "Signed radial tilt (degrees)",
        90.0,
    )

    rotation_axis = figure.add_subplot(grid[1, 1])
    rotation_degrees = np.degrees(sector_rotation)
    rotation_limit = (
        float(np.nanmax(np.abs(rotation_degrees)))
        if np.isfinite(rotation_degrees).any()
        else 1.0
    )
    _matrix_panel(
        figure,
        rotation_axis,
        pair_midpoints,
        rotation_degrees,
        f"C. Accumulated rotation across a {RADIAL_LAG:g} px radial lag",
        "Signed long-range rotation (degrees)",
        rotation_limit,
    )

    ring_abs, ring_signed, ring_support = _cell_summaries(
        np.degrees(sector_tilt)
    )
    pair_abs, pair_signed, pair_support = _cell_summaries(rotation_degrees)
    profile_axis = rotation_axis.inset_axes([0.08, -0.54, 0.84, 0.33])
    profile_axis.plot(
        ring_radii,
        ring_abs,
        color="#003660",
        linewidth=2.0,
        label="Mean |radial tilt|",
    )
    profile_axis.plot(
        ring_radii,
        ring_signed,
        color="#56B4E9",
        linewidth=1.6,
        linestyle="--",
        label="Mean signed radial tilt",
    )
    profile_axis.plot(
        pair_midpoints,
        pair_abs,
        color="#D55E00",
        linewidth=2.0,
        label=f"Mean |{RADIAL_LAG:g} px rotation|",
    )
    profile_axis.plot(
        pair_midpoints,
        pair_signed,
        color="#E69F00",
        linewidth=1.6,
        linestyle=":",
        label=f"Mean signed {RADIAL_LAG:g} px rotation",
    )
    profile_axis.axhline(0.0, color="0.35", linewidth=0.8)
    profile_axis.set_xlabel("Radius or pair midpoint (px)")
    profile_axis.set_ylabel("Degrees")
    profile_axis.grid(axis="y", color="0.88", linewidth=0.7)
    profile_axis.legend(ncol=2, fontsize=8, loc="upper left")
    support_axis = profile_axis.twinx()
    support_axis.plot(ring_radii, ring_support, color="0.45", alpha=0.45)
    support_axis.plot(pair_midpoints, pair_support, color="0.1", alpha=0.45)
    support_axis.set_ylim(0.0, 1.0)
    support_axis.set_ylabel("Sector support", color="0.3")

    row = measurements.loc[measurements["Object_Label"] == label].iloc[0]
    summary = (
        f"Overall {RADIAL_LAG:g}px: |Δ|="
        f"{row['OrientZones_LongRangeRotation-Mask-Overall']:.2f}°, signed="
        f"{row['OrientZones_SignedLongRangeRotation-Mask-Overall']:.2f}°, "
        f"support={row['OrientZones_LongRangeRotationSupport-Mask-Overall']:.2f}\n"
        "Dense→Sparse: |Δ|="
        f"{row['OrientZones_LongRangeRotation-Mask-DenseToSparse']:.2f}°, signed="
        f"{row['OrientZones_SignedLongRangeRotation-Mask-DenseToSparse']:.2f}°, "
        f"support={row['OrientZones_LongRangeRotationSupport-Mask-DenseToSparse']:.2f}"
    )
    figure.suptitle(
        "Sectorized Sholl-style radial orientation diagnostic\n" + summary,
        fontsize=14,
    )
    output = OUTPUT_DIR / f"long_range_sholl_spectral_colony_{colony_name.lower()}.png"
    figure.savefig(output, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output


def render_long_range_samples() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    segmented, _workflow_stats = reproduce_notebook_segmentation()
    for colony_name, label, section_index in COLONIES:
        output = render_colony(segmented, colony_name, label, section_index)
        print(output, flush=True)


if __name__ == "__main__":
    render_long_range_samples()
