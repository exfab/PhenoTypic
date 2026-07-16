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
    _RADIAL_RELATIVE_MIN_COHERENCE,
    _RADIAL_RELATIVE_N_SECTORS,
    radial_ring_orientation_profile,
    radial_ring_sector_field,
    signed_radial_relative_field,
    zone_selector,
)


OUTPUT_DIR = SCRIPT_DIR.parent / "artifacts"
COLONIES = (("R3C4", 24), ("R4C6", 36))
SPECTRAL_NORM = Normalize(vmin=-180.0, vmax=180.0)


def extract_full_length_ring_fields(
    section,
    operation: MeasureOrientationZones,
    profiles: dict,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Extend ring compounding to the full detected-object radius.

    Unlike the production Sholl-style diagnostics, this prototype must not
    stop at ``min(sparse_end_radius, symmetric_radius)``. Its exclusive outer
    bound is therefore the first complete ring boundary beyond the farthest
    detected object pixel. The inferred inoculum core remains excluded.

    Args:
        section: Image crop containing one isolated detected colony.
        operation: Configured orientation-zone measurer.
        profiles: Profiles initially extracted by ``extract_profiles``.

    Returns:
        A tuple containing updated profiles, local orientation angles,
        coherence, and the full-length non-inoculum object selector.

    Raises:
        RuntimeError: If the crop does not yield exactly one object field or
            contains no detected structure outside the inoculum.
    """
    props, label2section = operation._prep(section)
    records = list(operation._iter_object_fields(section, props, label2section))
    if len(records) != 1:
        raise RuntimeError("Expected exactly one object field")
    (
        _prop,
        seg,
        obj_mask,
        phi,
        coherence,
        _gradient,
        dist_map,
        centre,
    ) = records[0]
    if phi.shape != profiles["base"].shape:
        raise RuntimeError("Pixel field and display tile shapes differ")
    if not np.allclose(centre, profiles["centre"]):
        raise RuntimeError("Full-length field changed the inoculum center")

    inner_radius = float(seg.core_end_radius)
    outside_core = (
        obj_mask
        & np.isfinite(dist_map)
        & (dist_map >= inner_radius)
    )
    if not outside_core.any():
        raise RuntimeError("No detected structure exists outside the inoculum")
    object_extent_radius = float(np.max(dist_map[outside_core]))
    radial_span = np.nextafter(object_extent_radius, np.inf) - inner_radius
    n_rings = max(1, int(np.ceil(radial_span / operation.radial_ring_width)))
    outer_radius = inner_radius + n_rings * operation.radial_ring_width
    selector = zone_selector(
        dist_map,
        inner_radius,
        outer_radius,
        obj_mask,
        "Mask",
    )
    updated_profiles = dict(profiles)
    updated_profiles.update(
        {
            "inner_radius": inner_radius,
            "outer_radius": outer_radius,
            "object_extent_radius": object_extent_radius,
            "obj_mask": obj_mask,
            "dist_map": dist_map,
            "reliable_structure": (
                selector
                & np.isfinite(coherence)
                & (coherence >= _RADIAL_RELATIVE_MIN_COHERENCE)
            ),
        }
    )
    return updated_profiles, phi, coherence, selector


def axial_difference(outer: np.ndarray | float, inner: float) -> np.ndarray:
    """Return seam-safe axial differences in [-pi/2, pi/2]."""
    difference = np.asarray(outer, dtype=float) - inner
    return 0.5 * np.arctan2(
        np.sin(2.0 * difference),
        np.cos(2.0 * difference),
    )


def axial_mean(values: np.ndarray) -> tuple[float, float]:
    """Return the equal-cell axial mean and resultant."""
    cosine = float(np.mean(np.cos(2.0 * values)))
    sine = float(np.mean(np.sin(2.0 * values)))
    return 0.5 * float(np.arctan2(sine, cosine)), float(np.hypot(cosine, sine))


def axial_median(values: np.ndarray, mean_angle: float) -> float:
    """Return a deterministic sample axial median.

    The selected observed angle minimizes total absolute axial distance. Ties
    are resolved by proximity to the equal-cell axial mean.
    """
    distances = np.abs(
        axial_difference(values[:, np.newaxis], values[np.newaxis, :])
    )
    costs = distances.sum(axis=1)
    best = np.flatnonzero(np.isclose(costs, costs.min(), rtol=0.0, atol=1e-12))
    if best.size == 1:
        return float(values[best[0]])
    tie_distance = np.abs(axial_difference(values[best], mean_angle))
    return float(values[best[int(np.argmin(tie_distance))]])


def ring_consensus_profiles(
    sector_tilt: np.ndarray,
    sector_resultant: np.ndarray,
    *,
    minimum_sectors: int = 3,
    minimum_ring_resultant: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Summarize each ring once using equal reliable-sector weight."""
    n_rings = sector_tilt.shape[0]
    mean_tilt = np.full(n_rings, np.nan)
    median_tilt = np.full(n_rings, np.nan)
    ring_resultant = np.full(n_rings, np.nan)
    sector_support = np.zeros(n_rings, dtype=float)
    reliable = np.isfinite(sector_tilt) & np.isfinite(sector_resultant)
    for ring in range(n_rings):
        values = sector_tilt[ring, reliable[ring]]
        sector_support[ring] = values.size / sector_tilt.shape[1]
        if values.size < minimum_sectors:
            continue
        mean_angle, resultant = axial_mean(values)
        ring_resultant[ring] = resultant
        if resultant < minimum_ring_resultant:
            continue
        mean_tilt[ring] = mean_angle
        median_tilt[ring] = axial_median(values, mean_angle)
    return mean_tilt, median_tilt, ring_resultant, sector_support


def compound_ring_tilt(
    radii: np.ndarray,
    ring_tilt: np.ndarray,
    *,
    maximum_abs_tilt: float = np.deg2rad(75.0),
) -> np.ndarray:
    """Integrate ring-level radial-relative tilt into angular path rotation.

    For adjacent ring centers, the constant-tilt polar predictor is
    ``d_alpha = tan(delta) * log(r_next / r_current)``. The returned value is
    the cumulative sum of those increments in radians. A missing ring or a
    near-tangent ring terminates continuous accumulation.
    """
    radii = np.asarray(radii, dtype=float)
    tilt = np.asarray(ring_tilt, dtype=float)
    output = np.full_like(tilt, np.nan)
    starts = np.flatnonzero(np.isfinite(tilt))
    if starts.size == 0:
        return output
    start = int(starts[0])
    output[start] = 0.0
    for ring in range(start, tilt.size - 1):
        if not (
            np.isfinite(output[ring])
            and np.isfinite(tilt[ring])
            and np.isfinite(tilt[ring + 1])
        ):
            break
        if abs(tilt[ring]) > maximum_abs_tilt:
            break
        step = np.tan(tilt[ring]) * np.log(radii[ring + 1] / radii[ring])
        output[ring + 1] = output[ring] + step
    return output


def render_compounded_rotation(
    colony: str,
    profiles: dict,
    coherence: np.ndarray,
    selector: np.ndarray,
    phi: np.ndarray,
) -> tuple[Path, list[dict[str, float | str]], list[dict[str, float | str]]]:
    """Render equal-sector mean and median compounded Sholl profiles."""
    signed_tilt, _signed_turn, _turn_mag, polar_angle = (
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
    mean_tilt, median_tilt, ring_resultant, sector_support = (
        ring_consensus_profiles(sector_tilt, sector_resultant)
    )
    mean_cumulative = compound_ring_tilt(radii, mean_tilt)
    median_cumulative = compound_ring_tilt(radii, median_tilt)
    profiles = {**profiles, "radii": radii}
    n_sectors = sector_tilt.shape[1]
    mean_lattice = np.repeat(mean_cumulative[:, np.newaxis], n_sectors, axis=1)
    median_lattice = np.repeat(
        median_cumulative[:, np.newaxis], n_sectors, axis=1
    )
    mean_field = radial_ring_sector_field(
        mean_lattice,
        polar_angle,
        profiles["dist_map"],
        profiles["reliable_structure"],
        profiles["inner_radius"],
        profiles["ring_width"],
    )
    median_field = radial_ring_sector_field(
        median_lattice,
        polar_angle,
        profiles["dist_map"],
        profiles["reliable_structure"],
        profiles["inner_radius"],
        profiles["ring_width"],
    )

    mean_support, mean_p95, mean_raw = profile_metrics(mean_cumulative)
    median_support, median_p95, median_raw = profile_metrics(median_cumulative)
    base = profiles["base"]
    low, high = np.percentile(base[np.isfinite(base)], (1.0, 99.8))
    cmap = plt.get_cmap("Spectral").copy()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    figure, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

    for axis, field, name, support, p95, raw in (
        (
            axes[0, 0],
            mean_field,
            "Equal-sector axial mean",
            mean_support,
            mean_p95,
            mean_raw,
        ),
        (
            axes[0, 1],
            median_field,
            "Equal-sector axial median",
            median_support,
            median_p95,
            median_raw,
        ),
    ):
        axis.imshow(base, cmap="gray", vmin=low, vmax=high)
        axis.imshow(
            np.ma.masked_invalid(np.degrees(field)),
            cmap=cmap,
            norm=SPECTRAL_NORM,
            alpha=0.82,
        )
        draw_rings(axis, profiles)
        axis.set_xlim(-0.5, base.shape[1] - 0.5)
        axis.set_ylim(base.shape[0] - 0.5, -0.5)
        axis.set_aspect("equal")
        axis.set_axis_off()
        continuous_count = int(round(support * radii.size))
        axis.set_title(
            f"{name} ring compounding\n"
            f"continuous cumulative rings {continuous_count}/{radii.size} "
            f"({support:.1%}); |rotation| p95 {p95:.1f}°; "
            f"raw peak {raw:.1f}°",
            fontsize=10,
        )

    axes[1, 0].plot(
        radii,
        np.degrees(mean_tilt),
        marker="o",
        linewidth=1.6,
        label="Equal-sector mean",
        color="#0072B2",
    )
    axes[1, 0].plot(
        radii,
        np.degrees(median_tilt),
        marker="s",
        linewidth=1.6,
        label="Equal-sector median",
        color="#D55E00",
    )
    axes[1, 0].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1, 0].set_title("Ring consensus radial-relative tilt")
    axes[1, 0].set_xlabel("Radius from inoculum center (px)")
    axes[1, 0].set_ylabel("Signed radial-relative tilt (degrees)")
    axes[1, 0].grid(alpha=0.22)
    support_axis = axes[1, 0].twinx()
    support_line = support_axis.plot(
        radii,
        sector_support,
        color="#666666",
        linestyle=":",
        linewidth=1.2,
        label="Sector support",
    )
    resultant_line = support_axis.plot(
        radii,
        ring_resultant,
        color="#7B3294",
        linestyle="--",
        linewidth=1.2,
        label="Ring resultant",
    )
    threshold_line = support_axis.axhline(
        0.15,
        color="#7B3294",
        linestyle=":",
        linewidth=0.8,
        alpha=0.7,
        label="Resultant threshold 0.15",
    )
    support_axis.set_ylabel("Sector support / ring resultant")
    support_axis.set_ylim(0.0, 1.02)
    main_handles, main_labels = axes[1, 0].get_legend_handles_labels()
    secondary_handles = [*support_line, *resultant_line, threshold_line]
    secondary_labels = [handle.get_label() for handle in secondary_handles]
    axes[1, 0].legend(
        [*main_handles, *secondary_handles],
        [*main_labels, *secondary_labels],
        frameon=False,
        loc="best",
    )

    axes[1, 1].plot(
        radii,
        np.degrees(mean_cumulative),
        marker="o",
        linewidth=1.8,
        label="Mean compounded",
        color="#0072B2",
    )
    axes[1, 1].plot(
        radii,
        np.degrees(median_cumulative),
        marker="s",
        linewidth=1.8,
        label="Median compounded",
        color="#D55E00",
    )
    axes[1, 1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1, 1].set_title(
        "Compounded outward path rotation\n"
        "Δα = tan(ring tilt) × log(r_next/r_current)"
    )
    axes[1, 1].set_xlabel("Radius from inoculum center (px)")
    axes[1, 1].set_ylabel("Cumulative signed rotation (degrees)")
    axes[1, 1].grid(alpha=0.22)
    axes[1, 1].legend(frameon=False)

    colorbar_axis = axes[0, 1].inset_axes([1.035, 0.08, 0.035, 0.84])
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=SPECTRAL_NORM, cmap=cmap),
        cax=colorbar_axis,
        orientation="vertical",
    )
    colorbar.set_label(
        "Compounded signed rotation (degrees), fixed −180° to +180°"
    )
    figure.suptitle(
        f"{colony}: colony-wide Sholl ring tilt compounded outward\n"
        f"full detected length {profiles['object_extent_radius']:.1f} px; "
        f"outer ring boundary {profiles['outer_radius']:.1f} px",
        fontsize=14,
    )
    figure.text(
        0.5,
        -0.01,
        (
            "Each reliable 10° sector contributes once to its ring. Mean and median "
            "therefore do not reward more pixels in an already supported sector. "
            "The inoculum is excluded. A missing ring, ring resultant <0.15, or "
            "|tilt| >75° stops the profile."
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )
    output = OUTPUT_DIR / f"twok_{colony}_ring_compounded_rotation_2x2.png"
    figure.savefig(
        output,
        dpi=190,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(figure)

    summary_rows = [
        {
            "Colony": colony,
            "Aggregator": "Equal-sector axial mean",
            "ContinuousRingFraction": mean_support,
            "ContinuousRingCount": int(round(mean_support * radii.size)),
            "TotalRings": int(radii.size),
            "AbsP95Deg": mean_p95,
            "RawPeakDeg": mean_raw,
            "ObjectExtentRadiusPx": profiles["object_extent_radius"],
            "OuterRingRadiusPx": profiles["outer_radius"],
        },
        {
            "Colony": colony,
            "Aggregator": "Equal-sector axial median",
            "ContinuousRingFraction": median_support,
            "ContinuousRingCount": int(round(median_support * radii.size)),
            "TotalRings": int(radii.size),
            "AbsP95Deg": median_p95,
            "RawPeakDeg": median_raw,
            "ObjectExtentRadiusPx": profiles["object_extent_radius"],
            "OuterRingRadiusPx": profiles["outer_radius"],
        },
    ]
    profile_rows: list[dict[str, float | str]] = []
    for ring, radius in enumerate(radii):
        profile_rows.append(
            {
                "Colony": colony,
                "Ring": float(ring),
                "RadiusPx": float(radius),
                "MeanTiltDeg": float(np.degrees(mean_tilt[ring])),
                "MedianTiltDeg": float(np.degrees(median_tilt[ring])),
                "RingResultant": float(ring_resultant[ring]),
                "SectorSupport": float(sector_support[ring]),
                "MeanCumulativeDeg": float(
                    np.degrees(mean_cumulative[ring])
                ),
                "MedianCumulativeDeg": float(
                    np.degrees(median_cumulative[ring])
                ),
            }
        )
    return output, summary_rows, profile_rows


def render_all_ring_compounded_rotation() -> None:
    """Render mean/median ring compounding for both measured colonies."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detected, _old = load_twok_detection()
    summary_rows: list[dict[str, float | str]] = []
    profile_rows: list[dict[str, float | str]] = []
    for colony, label in COLONIES:
        section = isolated_global_crop(
            detected, label, label_centroid(detected, label)
        )
        operation = MeasureOrientationZones(
            radial_ring_width=8.0,
            long_range_lag=32.0,
            quiver_block=24,
        )
        profiles = extract_profiles(section, operation)
        profiles, phi, coherence, selector = extract_full_length_ring_fields(
            section, operation, profiles
        )
        output, colony_summary, colony_profiles = render_compounded_rotation(
            colony,
            profiles,
            coherence,
            selector,
            phi,
        )
        summary_rows.extend(colony_summary)
        profile_rows.extend(colony_profiles)
        report(str(output))
    summary = pd.DataFrame(summary_rows)
    profiles_table = pd.DataFrame(profile_rows)
    summary_path = OUTPUT_DIR / "twok_ring_compounded_rotation_summary.csv"
    profiles_path = OUTPUT_DIR / "twok_ring_compounded_rotation_profiles.csv"
    summary.to_csv(summary_path, index=False)
    profiles_table.to_csv(profiles_path, index=False)
    report(str(summary_path))
    report(str(profiles_path))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    render_all_ring_compounded_rotation()
