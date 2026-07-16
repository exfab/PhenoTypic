from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from render_matched_ring_comparison import (  # noqa: E402
    extract_profiles,
    label_centroid,
    profile_metrics,
)
from render_ring_compounded_median_trimmed_colormaps import (  # noqa: E402
    orange_to_magenta_hsv,
)
from render_ring_compounded_rotation import (  # noqa: E402
    axial_difference,
    compound_ring_tilt,
    extract_full_length_ring_fields,
    ring_consensus_profiles,
)
from render_tangential_method_comparison import draw_rings  # noqa: E402
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


ANALYSIS_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = ANALYSIS_DIR / "artifacts"
COLONIES = (("R3C4", 24), ("R4C6", 36))
FULL_ANGLE_NORM = Normalize(vmin=-180.0, vmax=180.0)
BLUE = "#0072B2"
ORANGE = "#D55E00"
PURPLE = "#7B3294"
GRAY = "#666666"


def cumulative_axial_median_change_profile(
    ring_median_tilt: np.ndarray,
    *,
    ambiguity_atol: float = 1e-12,
) -> np.ndarray:
    """Accumulate seam-safe changes in adjacent ring axial medians.

    This is an orientation-state calculation, not the tangent-based radial
    path predictor. It therefore remains finite as a ring median approaches
    tangency. The first supported ring is zero. Missing rings and exactly
    90-degree axial changes terminate the continuous profile.

    Args:
        ring_median_tilt: Ring-level axial medians in radians.
        ambiguity_atol: Absolute tolerance for treating a 90-degree axial
            difference as directionally ambiguous.

    Returns:
        Cumulative unwrapped axial change in radians with the input shape.

    Raises:
        ValueError: If the input is not one-dimensional or the tolerance is
            non-finite or negative.
    """
    tilt = np.asarray(ring_median_tilt, dtype=float)
    if tilt.ndim != 1:
        raise ValueError("ring_median_tilt must be one-dimensional")
    if not np.isfinite(ambiguity_atol) or ambiguity_atol < 0.0:
        raise ValueError("ambiguity_atol must be finite and >= 0")

    cumulative = np.full_like(tilt, np.nan)
    starts = np.flatnonzero(np.isfinite(tilt))
    if starts.size == 0:
        return cumulative
    start = int(starts[0])
    cumulative[start] = 0.0
    for ring in range(start, tilt.size - 1):
        if not (
            np.isfinite(cumulative[ring])
            and np.isfinite(tilt[ring])
            and np.isfinite(tilt[ring + 1])
        ):
            break
        change = float(axial_difference(tilt[ring + 1], tilt[ring]))
        if np.isclose(
            abs(change),
            0.5 * np.pi,
            rtol=0.0,
            atol=ambiguity_atol,
        ):
            break
        cumulative[ring + 1] = cumulative[ring] + change
    return cumulative


def calculate_colony_axial_change(
    section,
    operation: MeasureOrientationZones,
) -> dict:
    """Calculate full-length ring medians and both cumulative definitions.

    Args:
        section: Image crop containing one isolated colony.
        operation: Configured orientation-zone measurer.

    Returns:
        Dictionary containing profiles, ring evidence, cumulative profiles,
        and pixel fields for rendering and export.
    """
    profiles = extract_profiles(section, operation)
    profiles, phi, coherence, selector = extract_full_length_ring_fields(
        section,
        operation,
        profiles,
    )
    return calculate_axial_change_from_fields(
        profiles,
        phi,
        coherence,
        selector,
    )


def calculate_axial_change_from_fields(
    profiles: dict,
    phi: np.ndarray,
    coherence: np.ndarray,
    selector: np.ndarray,
) -> dict:
    """Calculate ring evidence and cumulative profiles from one selector.

    Args:
        profiles: Full-length colony profile metadata.
        phi: Image-derived local axial orientation field in radians.
        coherence: Image-derived local orientation coherence.
        selector: Structure pixels allowed to contribute to ring-sector means.

    Returns:
        Dictionary containing ring evidence, cumulative profiles, and pixel
        fields painted on ``profiles["reliable_structure"]``.

    Raises:
        ValueError: If the field and selector arrays do not share a shape.
    """
    arrays = (coherence, selector, profiles["dist_map"])
    if any(array.shape != phi.shape for array in arrays):
        raise ValueError("orientation fields and selector must share one shape")
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
    _mean_tilt, median_tilt, ring_resultant, sector_support = (
        ring_consensus_profiles(sector_tilt, sector_resultant)
    )
    axial_change = cumulative_axial_median_change_profile(median_tilt)
    radial_path = compound_ring_tilt(radii, median_tilt)
    n_sectors = sector_tilt.shape[1]
    axial_lattice = np.repeat(axial_change[:, np.newaxis], n_sectors, axis=1)
    radial_lattice = np.repeat(radial_path[:, np.newaxis], n_sectors, axis=1)
    axial_field = radial_ring_sector_field(
        axial_lattice,
        polar_angle,
        profiles["dist_map"],
        profiles["reliable_structure"],
        profiles["inner_radius"],
        profiles["ring_width"],
    )
    radial_field = radial_ring_sector_field(
        radial_lattice,
        polar_angle,
        profiles["dist_map"],
        profiles["reliable_structure"],
        profiles["inner_radius"],
        profiles["ring_width"],
    )
    return {
        "profiles": {**profiles, "radii": radii},
        "radii": radii,
        "median_tilt": median_tilt,
        "ring_resultant": ring_resultant,
        "sector_support": sector_support,
        "axial_change": axial_change,
        "radial_path": radial_path,
        "axial_field": axial_field,
        "radial_field": radial_field,
        "polar_angle": polar_angle,
    }


def render_axial_change_comparison(colony: str, result: dict) -> Path:
    """Render the new full-length profile against the radial-path reference.

    Args:
        colony: Colony grid identifier.
        result: Output from :func:`calculate_colony_axial_change`.

    Returns:
        Path to the rendered PNG figure.
    """
    profiles = result["profiles"]
    radii = result["radii"]
    median_tilt = result["median_tilt"]
    axial_change = result["axial_change"]
    radial_path = result["radial_path"]
    axial_support, axial_p95, axial_raw = profile_metrics(axial_change)
    radial_support, radial_p95, radial_raw = profile_metrics(radial_path)
    base = profiles["base"]
    low, high = np.percentile(base[np.isfinite(base)], (1.0, 99.8))
    colormap = orange_to_magenta_hsv()
    colormap.set_bad((0.0, 0.0, 0.0, 0.0))

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 12),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.35, 0.8)},
    )
    overlay_specs = (
        (
            axes[0, 0],
            result["axial_field"],
            "Full-length cumulative axial-median change",
            axial_support,
            axial_p95,
            axial_raw,
        ),
        (
            axes[0, 1],
            result["radial_field"],
            "Tangent-based radial-path reference",
            radial_support,
            radial_p95,
            radial_raw,
        ),
    )
    for axis, field, name, support, p95, raw_peak in overlay_specs:
        axis.imshow(base, cmap="gray", vmin=low, vmax=high)
        overlay = axis.imshow(
            np.ma.masked_invalid(np.degrees(field)),
            cmap=colormap,
            norm=FULL_ANGLE_NORM,
            alpha=0.82,
        )
        draw_rings(axis, profiles)
        axis.set_xlim(-0.5, base.shape[1] - 0.5)
        axis.set_ylim(base.shape[0] - 0.5, -0.5)
        axis.set_aspect("equal")
        axis.set_axis_off()
        supported_count = int(round(support * radii.size))
        axis.set_title(
            f"{name}\n"
            f"supported rings {supported_count}/{radii.size}; "
            f"|change| p95 {p95:.1f} degrees; raw peak {raw_peak:.1f} degrees",
            fontsize=10,
        )

    colorbar = figure.colorbar(
        overlay,
        ax=axes[0, :],
        orientation="horizontal",
        fraction=0.046,
        pad=0.02,
        ticks=(-180.0, -120.0, -60.0, -40.0, 0.0, 40.0, 60.0, 120.0, 180.0),
    )
    colorbar.set_label(
        "Signed cumulative angle (degrees), fixed -180 to +180"
    )

    axes[1, 0].plot(
        radii,
        np.degrees(median_tilt),
        color=BLUE,
        marker="o",
        linewidth=1.7,
        label="Equal-sector axial median",
    )
    axes[1, 0].axhline(0.0, color=GRAY, linewidth=0.8)
    axes[1, 0].set_title("Ring-level radial-relative orientation state")
    axes[1, 0].set_xlabel("Radius from inoculum center (px)")
    axes[1, 0].set_ylabel("Axial-median tilt (degrees)")
    axes[1, 0].set_ylim(-100.0, 100.0)
    axes[1, 0].grid(alpha=0.20)
    evidence_axis = axes[1, 0].twinx()
    evidence_axis.plot(
        radii,
        result["sector_support"],
        color=GRAY,
        linestyle=":",
        linewidth=1.2,
        label="Sector support",
    )
    evidence_axis.plot(
        radii,
        result["ring_resultant"],
        color=PURPLE,
        linestyle="--",
        linewidth=1.2,
        label="Ring resultant",
    )
    evidence_axis.set_ylabel("Sector support / ring resultant")
    evidence_axis.set_ylim(0.0, 1.02)
    main_handles, main_labels = axes[1, 0].get_legend_handles_labels()
    evidence_handles, evidence_labels = evidence_axis.get_legend_handles_labels()
    axes[1, 0].legend(
        [*main_handles, *evidence_handles],
        [*main_labels, *evidence_labels],
        frameon=False,
        loc="lower left",
    )

    axes[1, 1].plot(
        radii,
        np.degrees(axial_change),
        color=BLUE,
        marker="o",
        linewidth=1.8,
        label="Cumulative axial-median change",
    )
    axes[1, 1].plot(
        radii,
        np.degrees(radial_path),
        color=ORANGE,
        marker="s",
        markerfacecolor="none",
        linestyle="--",
        linewidth=1.5,
        label="Tangent-based radial path",
    )
    axes[1, 1].axhline(0.0, color=GRAY, linewidth=0.8)
    axes[1, 1].set_title("Cumulative profiles")
    axes[1, 1].set_xlabel("Radius from inoculum center (px)")
    axes[1, 1].set_ylabel("Cumulative signed change (degrees)")
    axes[1, 1].grid(alpha=0.20)
    axes[1, 1].legend(frameon=False, loc="best")

    figure.suptitle(
        f"{colony}: full-length equal-sector axial-median change\n"
        f"detected extent {profiles['object_extent_radius']:.1f} px; "
        f"outer ring boundary {profiles['outer_radius']:.1f} px",
        fontsize=14,
    )
    figure.text(
        0.5,
        -0.01,
        (
            "New profile: C[k] = C[k-1] + seam-safe axial difference between "
            "adjacent ring medians. It has no tan(tilt) singularity. Missing "
            "ring consensus still terminates continuous accumulation."
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )
    output = OUTPUT_DIR / f"twok_{colony}_ring_median_axial_change_2x2.png"
    figure.savefig(
        output,
        dpi=190,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(figure)
    return output


def render_all_ring_median_axial_change() -> None:
    """Calculate, export, and render the new profile for both colonies."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detected, _old = load_twok_detection()
    summary_rows: list[dict[str, float | int | str]] = []
    profile_rows: list[dict[str, float | int | str]] = []
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
        result = calculate_colony_axial_change(section, operation)
        output = render_axial_change_comparison(colony, result)
        report(str(output))

        radii = result["radii"]
        axial_change = result["axial_change"]
        radial_path = result["radial_path"]
        axial_support, axial_p95, axial_raw = profile_metrics(axial_change)
        radial_support, radial_p95, radial_raw = profile_metrics(radial_path)
        profiles = result["profiles"]
        summary_rows.append(
            {
                "Colony": colony,
                "TotalRings": int(radii.size),
                "AxialChangeSupportedRings": int(np.isfinite(axial_change).sum()),
                "AxialChangeSupportedFraction": axial_support,
                "AxialChangeAbsP95Deg": axial_p95,
                "AxialChangeRawPeakDeg": axial_raw,
                "RadialPathSupportedRings": int(np.isfinite(radial_path).sum()),
                "RadialPathSupportedFraction": radial_support,
                "RadialPathAbsP95Deg": radial_p95,
                "RadialPathRawPeakDeg": radial_raw,
                "ObjectExtentRadiusPx": profiles["object_extent_radius"],
                "OuterRingRadiusPx": profiles["outer_radius"],
            }
        )
        for ring, radius in enumerate(radii):
            profile_rows.append(
                {
                    "Colony": colony,
                    "Ring": ring,
                    "RadiusPx": float(radius),
                    "MedianTiltDeg": float(
                        np.degrees(result["median_tilt"][ring])
                    ),
                    "RingResultant": float(result["ring_resultant"][ring]),
                    "SectorSupport": float(result["sector_support"][ring]),
                    "AxialMedianCumulativeChangeDeg": float(
                        np.degrees(axial_change[ring])
                    ),
                    "RadialPathCumulativeDeg": float(
                        np.degrees(radial_path[ring])
                    ),
                }
            )

    summary_path = OUTPUT_DIR / "twok_ring_median_axial_change_summary.csv"
    profiles_path = OUTPUT_DIR / "twok_ring_median_axial_change_profiles.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(profile_rows).to_csv(profiles_path, index=False)
    report(str(summary_path))
    report(str(profiles_path))
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    render_all_ring_median_axial_change()
