from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from neurospora_orientation_samples import reproduce_notebook_segmentation
from phenotypic.measure import MeasureOrientationZones
from phenotypic.measure._measure_orientation_zones import (
    cumulative_ring_rotation_profile,
    radial_ring_orientation_profile,
    radial_ring_sector_field,
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
COLONIES = (("A", 1116, 35), ("B", 626, 18))
RING_WIDTH = 8.0
N_SECTORS = 36


def _add_overlay(
    figure,
    axis,
    tile: np.ndarray,
    values: np.ndarray,
    ring_radii: np.ndarray,
    centre: tuple[float, float],
    title: str,
    units: str,
) -> float:
    lower, upper = np.percentile(tile, (1.0, 99.8))
    axis.imshow(tile, cmap="gray", vmin=float(lower), vmax=float(upper))
    finite = np.isfinite(values)
    raw_peak = float(np.nanmax(np.abs(values))) if finite.any() else 1.0
    if raw_peak <= np.finfo(float).eps:
        raw_peak = float(np.finfo(np.float32).eps)
    spectral = matplotlib.colormaps["Spectral"].copy()
    spectral.set_bad((0.0, 0.0, 0.0, 0.0))
    image = axis.imshow(
        np.ma.masked_invalid(values),
        cmap=spectral,
        vmin=-raw_peak,
        vmax=raw_peak,
        alpha=0.76,
    )
    centre_row, centre_col = centre
    for radius in ring_radii:
        axis.add_patch(
            Circle(
                (centre_col, centre_row),
                float(radius),
                fill=False,
                edgecolor=(1.0, 1.0, 1.0, 0.48),
                linewidth=0.55,
            )
        )
    axis.set_title(
        f"{title}\nraw peak: ±{raw_peak:.2f} {units}",
        loc="left",
        fontsize=11,
    )
    axis.set_axis_off()
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.015)
    colorbar.set_label(units)
    return raw_peak


def render_colony(segmented, name: str, label: int, section_index: int) -> Path:
    section = segmented.grid[section_index]
    objmap = section.objmap[:]
    section.objmap[:] = np.where(objmap == label, label, 0)
    operation = MeasureOrientationZones(
        radial_ring_width=RING_WIDTH,
        long_range_lag=16.0,
        quiver_block=24,
    )
    operation.measure(section)
    exact_figure = operation.cumulative_rotation_overlay(section)
    exact_figure.write_image(
        str(
            OUTPUT_DIR
            / f"cumulative_rotation_overlay_colony_{name.lower()}.png"
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
    signed_tilt, signed_turning, _magnitude, polar = (
        signed_radial_relative_field(phi, centre, distance)
    )
    inner_radius = float(segmentation.core_end_radius)
    outer_radius = min(
        float(segmentation.sparse_end_radius),
        float(segmentation.symmetric_radius),
    )
    structure = (
        object_mask
        & (distance >= inner_radius)
        & (distance < outer_radius)
    )
    reliable = structure & (coherence >= 0.15)
    ring_radii, sector_tilt, _resultants = radial_ring_orientation_profile(
        signed_tilt,
        polar,
        coherence,
        distance,
        structure,
        inner_radius,
        outer_radius,
        RING_WIDTH,
        N_SECTORS,
    )
    cumulative = cumulative_ring_rotation_profile(sector_tilt)
    cumulative_field = radial_ring_sector_field(
        cumulative,
        polar,
        distance,
        reliable,
        inner_radius,
        RING_WIDTH,
    )
    local_degrees = np.where(reliable, np.degrees(signed_turning), np.nan)
    cumulative_degrees = np.degrees(cumulative_field)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8.5),
        constrained_layout=True,
    )
    local_peak = _add_overlay(
        figure,
        axes[0],
        tile,
        local_degrees,
        ring_radii,
        centre,
        "A. Local outward turning rate",
        "deg/px",
    )
    cumulative_peak = _add_overlay(
        figure,
        axes[1],
        tile,
        cumulative_degrees,
        ring_radii,
        centre,
        "B. Cumulative ring-to-ring rotation",
        "degrees",
    )
    finite_cumulative = np.isfinite(cumulative)
    supported_cells = int(finite_cumulative.sum())
    total_cells = int(cumulative.size)
    figure.suptitle(
        f"Colony {name}: local change versus accumulated outward rotation\n"
        f"8 px rings, 36 equal angular sectors; inoculum and unsupported "
        f"cells are blank; cumulative support {supported_cells}/{total_cells}\n"
        f"Raw peaks retained: local ±{local_peak:.2f} deg/px, cumulative "
        f"±{cumulative_peak:.2f}°",
        fontsize=14,
    )
    output = OUTPUT_DIR / f"local_vs_cumulative_colony_{name.lower()}.png"
    figure.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output


def render_cumulative_samples() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    segmented, _stats = reproduce_notebook_segmentation()
    for name, label, section_index in COLONIES:
        print(render_colony(segmented, name, label, section_index), flush=True)


if __name__ == "__main__":
    render_cumulative_samples()
