from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from render_matched_ring_comparison import (  # noqa: E402
    extract_profiles,
    label_centroid,
    profile_metrics,
)
from render_ring_compounded_rotation import (  # noqa: E402
    compound_ring_tilt,
    extract_full_length_ring_fields,
    ring_consensus_profiles,
)
from render_tangential_method_comparison import (  # noqa: E402
    draw_rings,
)
from render_twok_reconnected_orientation import (  # noqa: E402
    isolated_global_crop,
    load_twok_detection,
    report,
)
from phenotypic.measure import MeasureOrientationZones  # noqa: E402
from phenotypic.measure._measure_orientation_zones import (  # noqa: E402
    _RADIAL_RELATIVE_N_SECTORS,
    radial_ring_orientation_profile,
    radial_ring_sector_field,
    signed_radial_relative_field,
)


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ANALYSIS_DIR / "artifacts"
COLONIES = (("R3C4", 24), ("R4C6", 36))
FULL_ANGLE_NORM = Normalize(vmin=-180.0, vmax=180.0)
COLORBAR_TICKS = (-180.0, -120.0, -60.0, -40.0, 0.0, 40.0, 60.0, 120.0, 180.0)


def median_compounded_field(
    profiles: dict,
    coherence: np.ndarray,
    selector: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the equal-sector axial-median compounded rotation field.

    Args:
        profiles: Per-colony radial profile inputs.
        coherence: Axial orientation-field coherence at each pixel.
        selector: Reliable non-inoculum structure mask.
        phi: Local axial orientation angle in radians.

    Returns:
        A tuple containing the pixel field in radians, ring-center radii in
        pixels, and cumulative median rotation in radians.
    """
    signed_tilt, _signed_turn, _turn_magnitude, polar_angle = (
        signed_radial_relative_field(
            phi,
            profiles["centre"],
            profiles["dist_map"],
        )
    )
    radii, sector_tilt, sector_resultant = radial_ring_orientation_profile(
        signed_tilt,
        polar_angle,
        coherence,
        profiles["dist_map"],
        selector,
        profiles["inner_radius"],
        profiles["outer_radius"],
        profiles["ring_width"],
        _RADIAL_RELATIVE_N_SECTORS,
    )
    _mean_tilt, median_tilt, _ring_resultant, _sector_support = (
        ring_consensus_profiles(sector_tilt, sector_resultant)
    )
    median_cumulative = compound_ring_tilt(radii, median_tilt)
    median_lattice = np.repeat(
        median_cumulative[:, np.newaxis],
        sector_tilt.shape[1],
        axis=1,
    )
    field = radial_ring_sector_field(
        median_lattice,
        polar_angle,
        profiles["dist_map"],
        profiles["reliable_structure"],
        profiles["inner_radius"],
        profiles["ring_width"],
    )
    return field, radii, median_cumulative


def render_colormap_comparison(
    colony: str,
    profiles: dict,
    field: np.ndarray,
    radii: np.ndarray,
    median_cumulative: np.ndarray,
) -> Path:
    """Render HSV and RdBu views of the same median-compounded field.

    Args:
        colony: Colony grid identifier.
        profiles: Per-colony radial profile inputs.
        field: Pixelwise cumulative rotation field in radians.
        radii: Ring-center radii in pixels.
        median_cumulative: Cumulative median rotation at each ring in radians.

    Returns:
        Path to the rendered PNG comparison.
    """
    support, p95, raw_peak = profile_metrics(median_cumulative)
    base = profiles["base"]
    low, high = np.percentile(base[np.isfinite(base)], (1.0, 99.8))
    field_degrees = np.ma.masked_invalid(np.degrees(field))
    figure, axes = plt.subplots(1, 2, figsize=(15, 7.5), constrained_layout=True)

    for axis, colormap_name, display_name in (
        (axes[0], "hsv", "HSV cyclic"),
        (axes[1], "RdBu", "RdBu diverging"),
    ):
        colormap = plt.get_cmap(colormap_name).copy()
        colormap.set_bad((0.0, 0.0, 0.0, 0.0))
        axis.imshow(base, cmap="gray", vmin=low, vmax=high)
        overlay = axis.imshow(
            field_degrees,
            cmap=colormap,
            norm=FULL_ANGLE_NORM,
            alpha=0.82,
        )
        draw_rings(axis, profiles)
        axis.set_xlim(-0.5, base.shape[1] - 0.5)
        axis.set_ylim(base.shape[0] - 0.5, -0.5)
        axis.set_aspect("equal")
        axis.set_axis_off()
        axis.set_title(
            f"{display_name}\n"
            f"median |rotation| p95 {p95:.1f} degrees; "
            f"raw peak {raw_peak:.1f} degrees",
            fontsize=11,
        )
        colorbar = figure.colorbar(
            overlay,
            ax=axis,
            orientation="vertical",
            fraction=0.047,
            pad=0.02,
            ticks=COLORBAR_TICKS,
        )
        colorbar.set_label(
            "Cumulative signed rotation (degrees), fixed -180 to +180"
        )
        for tick, label in zip(COLORBAR_TICKS, colorbar.ax.get_yticklabels()):
            if abs(tick) == 40.0:
                label.set_fontweight("bold")

    continuous_count = int(round(support * radii.size))
    figure.suptitle(
        f"{colony}: equal-sector axial-median ring compounding\n"
        f"full detected length {profiles['object_extent_radius']:.1f} px; "
        f"outer ring boundary {profiles['outer_radius']:.1f} px; continuous rings "
        f"{continuous_count}/{radii.size} ({support:.1%})",
        fontsize=14,
    )
    output = (
        OUTPUT_DIR
        / f"twok_{colony}_equal_sector_axial_median_hsv_vs_rdbu.png"
    )
    figure.savefig(
        output,
        dpi=190,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(figure)
    return output


def render_all_median_colormap_comparisons() -> None:
    """Render equal-sector median HSV/RdBu comparisons for both colonies."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detected, _old = load_twok_detection()
    for colony, label in COLONIES:
        section = isolated_global_crop(
            detected,
            label,
            label_centroid(detected, label),
        )
        operation = MeasureOrientationZones(
            radial_ring_width=8.0,
            long_range_lag=32.0,
            quiver_block=24,
        )
        profiles = extract_profiles(section, operation)
        profiles, phi, coherence, selector = extract_full_length_ring_fields(
            section,
            operation,
            profiles,
        )
        field, radii, median_cumulative = median_compounded_field(
            profiles,
            coherence,
            selector,
            phi,
        )
        profiles = {**profiles, "radii": radii}
        output = render_colormap_comparison(
            colony,
            profiles,
            field,
            radii,
            median_cumulative,
        )
        report(str(output))


if __name__ == "__main__":
    render_all_median_colormap_comparisons()
