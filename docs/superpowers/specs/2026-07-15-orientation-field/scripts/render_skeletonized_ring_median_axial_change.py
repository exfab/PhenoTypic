from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

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
    extract_full_length_ring_fields,
)
from render_ring_median_axial_change import (  # noqa: E402
    calculate_axial_change_from_fields,
)
from render_tangential_method_comparison import draw_rings  # noqa: E402
from render_twok_reconnected_orientation import (  # noqa: E402
    isolated_global_crop,
    load_twok_detection,
    report,
)
from phenotypic.measure import MeasureOrientationZones  # noqa: E402
from phenotypic.measure._measure_orientation_zones import (  # noqa: E402
    _RADIAL_RELATIVE_MIN_COHERENCE,
    _RADIAL_RELATIVE_N_SECTORS,
    radial_ring_sector_field,
)


ANALYSIS_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = ANALYSIS_DIR / "artifacts"
COLONIES = (("R3C4", 24), ("R4C6", 36))
FULL_ANGLE_NORM = Normalize(vmin=-180.0, vmax=180.0)
BLUE = "#0072B2"
ORANGE = "#D55E00"
PURPLE = "#7B3294"
GRAY = "#666666"


def calculate_mask_and_skeleton_variants(
    section,
    operation: MeasureOrientationZones,
) -> tuple[dict, dict, np.ndarray]:
    """Calculate identical orientation profiles with two sampling masks.

    The baseline uses every detected object pixel outside the inoculum. The
    skeleton variant uses a one-pixel morphological skeleton of the full object
    mask, then applies the same inoculum exclusion, coherence threshold, ring
    geometry, sector rules, and cumulative axial-change calculation. Local
    orientation and coherence still come from the original image array.

    Args:
        section: Image crop containing one isolated colony.
        operation: Configured orientation-zone measurer.

    Returns:
        Baseline result, skeleton result, and the full one-pixel skeleton mask.
    """
    initial_profiles = extract_profiles(section, operation)
    profiles, phi, coherence, full_selector = extract_full_length_ring_fields(
        section,
        operation,
        initial_profiles,
    )
    baseline = calculate_axial_change_from_fields(
        profiles,
        phi,
        coherence,
        full_selector,
    )

    skeleton = np.asarray(skeletonize(profiles["obj_mask"]), dtype=bool)
    skeleton_selector = skeleton & full_selector
    skeleton_reliable = (
        skeleton_selector
        & np.isfinite(coherence)
        & (coherence >= _RADIAL_RELATIVE_MIN_COHERENCE)
    )
    skeleton_profiles = {
        **profiles,
        "reliable_structure": skeleton_reliable,
    }
    skeleton_result = calculate_axial_change_from_fields(
        skeleton_profiles,
        phi,
        coherence,
        skeleton_selector,
    )
    return baseline, skeleton_result, skeleton


def paint_profile_on_detected_mask(result: dict, detected_profiles: dict) -> np.ndarray:
    """Paint ring values on the full mask for a legible comparison overlay.

    Args:
        result: Axial-change result estimated from either sampling mask.
        detected_profiles: Baseline full-mask profile metadata.

    Returns:
        Pixel field in radians painted only on reliable detected structure.
    """
    values = result["axial_change"]
    lattice = np.repeat(
        values[:, np.newaxis],
        _RADIAL_RELATIVE_N_SECTORS,
        axis=1,
    )
    return radial_ring_sector_field(
        lattice,
        result["polar_angle"],
        detected_profiles["dist_map"],
        detected_profiles["reliable_structure"],
        detected_profiles["inner_radius"],
        detected_profiles["ring_width"],
    )


def render_skeleton_comparison(
    colony: str,
    baseline: dict,
    skeleton_result: dict,
    skeleton: np.ndarray,
) -> Path:
    """Render mask-versus-skeleton evidence and cumulative profiles.

    Args:
        colony: Colony grid identifier.
        baseline: Full detected-mask calculation.
        skeleton_result: Skeleton-mask calculation.
        skeleton: One-pixel morphological skeleton of the object mask.

    Returns:
        Path to the rendered PNG comparison.
    """
    profiles = baseline["profiles"]
    base = profiles["base"]
    radii = baseline["radii"]
    low, high = np.percentile(base[np.isfinite(base)], (1.0, 99.8))
    baseline_field = baseline["axial_field"]
    skeleton_field = paint_profile_on_detected_mask(skeleton_result, profiles)
    colormap = orange_to_magenta_hsv()
    colormap.set_bad((0.0, 0.0, 0.0, 0.0))
    baseline_support, baseline_p95, baseline_raw = profile_metrics(
        baseline["axial_change"]
    )
    skeleton_support, skeleton_p95, skeleton_raw = profile_metrics(
        skeleton_result["axial_change"]
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(19, 11),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.25, 0.8)},
    )
    overlay_specs = (
        (
            axes[0, 0],
            baseline_field,
            "Detected-mask sampling",
            baseline_support,
            baseline_p95,
            baseline_raw,
        ),
        (
            axes[0, 1],
            skeleton_field,
            "Skeleton sampling, values painted on detected mask",
            skeleton_support,
            skeleton_p95,
            skeleton_raw,
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

    skeleton_axis = axes[0, 2]
    skeleton_axis.imshow(base, cmap="gray", vmin=low, vmax=high)
    skeleton_outside_core = (
        skeleton
        & (profiles["dist_map"] >= profiles["inner_radius"])
        & (profiles["dist_map"] < profiles["outer_radius"])
    )
    measurement_skeleton = skeleton_result["profiles"]["reliable_structure"]
    widened_skeleton = binary_dilation(measurement_skeleton, iterations=1)
    skeleton_overlay = np.ma.masked_where(
        ~widened_skeleton,
        np.ones(widened_skeleton.shape, dtype=float),
    )
    skeleton_axis.imshow(
        skeleton_overlay,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        alpha=0.92,
    )
    draw_rings(skeleton_axis, profiles)
    skeleton_axis.set_xlim(-0.5, base.shape[1] - 0.5)
    skeleton_axis.set_ylim(base.shape[0] - 0.5, -0.5)
    skeleton_axis.set_aspect("equal")
    skeleton_axis.set_axis_off()
    original_count = int(np.count_nonzero(profiles["obj_mask"]))
    skeleton_count = int(np.count_nonzero(skeleton))
    skeleton_axis.set_title(
        "Measurement skeleton\n"
        f"{np.count_nonzero(measurement_skeleton):,} reliable centerline pixels; "
        f"{np.count_nonzero(skeleton_outside_core):,} outside core; "
        f"{skeleton_count:,} total\n"
        f"from {original_count:,} mask pixels; display widened 1 px",
        fontsize=10,
    )

    colorbar = figure.colorbar(
        overlay,
        ax=axes[0, :2],
        orientation="horizontal",
        fraction=0.046,
        pad=0.02,
        ticks=(-180.0, -120.0, -60.0, -40.0, 0.0, 40.0, 60.0, 120.0, 180.0),
    )
    colorbar.set_label(
        "Cumulative axial-median change (degrees), fixed -180 to +180"
    )

    axes[1, 0].plot(
        radii,
        np.degrees(baseline["median_tilt"]),
        color=BLUE,
        marker="o",
        linewidth=1.7,
        label="Detected mask",
    )
    axes[1, 0].plot(
        radii,
        np.degrees(skeleton_result["median_tilt"]),
        color=ORANGE,
        marker="s",
        markerfacecolor="none",
        linewidth=1.5,
        label="Skeleton",
    )
    axes[1, 0].axhline(0.0, color=GRAY, linewidth=0.8)
    axes[1, 0].set_title("Equal-sector axial-median tilt")
    axes[1, 0].set_xlabel("Radius from inoculum center (px)")
    axes[1, 0].set_ylabel("Ring median tilt (degrees)")
    axes[1, 0].set_ylim(-100.0, 100.0)
    axes[1, 0].grid(alpha=0.20)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(
        radii,
        np.degrees(baseline["axial_change"]),
        color=BLUE,
        marker="o",
        linewidth=1.8,
        label="Detected mask",
    )
    axes[1, 1].plot(
        radii,
        np.degrees(skeleton_result["axial_change"]),
        color=ORANGE,
        marker="s",
        markerfacecolor="none",
        linewidth=1.6,
        label="Skeleton",
    )
    axes[1, 1].axhline(0.0, color=GRAY, linewidth=0.8)
    axes[1, 1].set_title("Cumulative axial-median change")
    axes[1, 1].set_xlabel("Radius from inoculum center (px)")
    axes[1, 1].set_ylabel("Cumulative signed change (degrees)")
    axes[1, 1].grid(alpha=0.20)
    axes[1, 1].legend(frameon=False)

    evidence_axis = axes[1, 2]
    evidence_axis.plot(
        radii,
        baseline["sector_support"],
        color=BLUE,
        linewidth=1.6,
        label="Mask sector support",
    )
    evidence_axis.plot(
        radii,
        skeleton_result["sector_support"],
        color=ORANGE,
        linewidth=1.6,
        label="Skeleton sector support",
    )
    evidence_axis.plot(
        radii,
        baseline["ring_resultant"],
        color=BLUE,
        linestyle="--",
        linewidth=1.2,
        label="Mask ring resultant",
    )
    evidence_axis.plot(
        radii,
        skeleton_result["ring_resultant"],
        color=ORANGE,
        linestyle="--",
        linewidth=1.2,
        label="Skeleton ring resultant",
    )
    evidence_axis.axhline(
        0.15,
        color=PURPLE,
        linestyle=":",
        linewidth=1.0,
        label="Resultant threshold 0.15",
    )
    evidence_axis.axhline(
        3.0 / _RADIAL_RELATIVE_N_SECTORS,
        color=GRAY,
        linestyle=":",
        linewidth=1.0,
        label="Three-sector minimum",
    )
    evidence_axis.set_title("Ring evidence")
    evidence_axis.set_xlabel("Radius from inoculum center (px)")
    evidence_axis.set_ylabel("Sector support / ring resultant")
    evidence_axis.set_ylim(0.0, 1.02)
    evidence_axis.grid(alpha=0.20)
    evidence_axis.legend(frameon=False, fontsize=8, ncol=2, loc="best")

    figure.suptitle(
        f"{colony}: full-length axial-median change with skeleton sampling\n"
        "Only the contributing branch mask changes; orientation field and "
        "calculation are identical",
        fontsize=14,
    )
    output = OUTPUT_DIR / f"twok_{colony}_skeletonized_axial_change_2x3.png"
    figure.savefig(
        output,
        dpi=190,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(figure)
    return output


def render_all_skeletonized_axial_change() -> None:
    """Render and export mask-versus-skeleton comparisons."""
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
        baseline, skeleton_result, skeleton = (
            calculate_mask_and_skeleton_variants(section, operation)
        )
        output = render_skeleton_comparison(
            colony,
            baseline,
            skeleton_result,
            skeleton,
        )
        report(str(output))

        summary: dict[str, float | int | str] = {
            "Colony": colony,
            "TotalRings": int(baseline["radii"].size),
            "ObjectMaskPixels": int(
                np.count_nonzero(baseline["profiles"]["obj_mask"])
            ),
            "SkeletonPixels": int(np.count_nonzero(skeleton)),
            "MaskReliableMeasurementPixels": int(
                np.count_nonzero(baseline["profiles"]["reliable_structure"])
            ),
            "SkeletonReliableMeasurementPixels": int(
                np.count_nonzero(
                    skeleton_result["profiles"]["reliable_structure"]
                )
            ),
        }
        for prefix, result in (
            ("Mask", baseline),
            ("Skeleton", skeleton_result),
        ):
            support, p95, raw_peak = profile_metrics(result["axial_change"])
            summary.update(
                {
                    f"{prefix}SupportedRings": int(
                        np.isfinite(result["axial_change"]).sum()
                    ),
                    f"{prefix}SupportedFraction": support,
                    f"{prefix}AbsP95Deg": p95,
                    f"{prefix}RawPeakDeg": raw_peak,
                    f"{prefix}MedianSectorSupport": float(
                        np.nanmedian(result["sector_support"])
                    ),
                    f"{prefix}MedianRingResultant": float(
                        np.nanmedian(result["ring_resultant"])
                    ),
                }
            )
        common = np.isfinite(baseline["axial_change"]) & np.isfinite(
            skeleton_result["axial_change"]
        )
        summary["CommonRingProfileMAEDeg"] = (
            float(
                np.mean(
                    np.abs(
                        np.degrees(
                            baseline["axial_change"][common]
                            - skeleton_result["axial_change"][common]
                        )
                    )
                )
            )
            if common.any()
            else np.nan
        )
        summary_rows.append(summary)

        for ring, radius in enumerate(baseline["radii"]):
            profile_rows.append(
                {
                    "Colony": colony,
                    "Ring": ring,
                    "RadiusPx": float(radius),
                    "MaskMedianTiltDeg": float(
                        np.degrees(baseline["median_tilt"][ring])
                    ),
                    "SkeletonMedianTiltDeg": float(
                        np.degrees(skeleton_result["median_tilt"][ring])
                    ),
                    "MaskCumulativeChangeDeg": float(
                        np.degrees(baseline["axial_change"][ring])
                    ),
                    "SkeletonCumulativeChangeDeg": float(
                        np.degrees(skeleton_result["axial_change"][ring])
                    ),
                    "MaskSectorSupport": float(baseline["sector_support"][ring]),
                    "SkeletonSectorSupport": float(
                        skeleton_result["sector_support"][ring]
                    ),
                    "MaskRingResultant": float(
                        baseline["ring_resultant"][ring]
                    ),
                    "SkeletonRingResultant": float(
                        skeleton_result["ring_resultant"][ring]
                    ),
                }
            )

    summary_path = OUTPUT_DIR / "twok_skeletonized_axial_change_summary.csv"
    profiles_path = OUTPUT_DIR / "twok_skeletonized_axial_change_profiles.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(profile_rows).to_csv(profiles_path, index=False)
    report(str(summary_path))
    report(str(profiles_path))
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    render_all_skeletonized_axial_change()
