from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from render_matched_ring_comparison import (  # noqa: E402
    extract_profiles,
    label_centroid,
)
from render_point_matched_ring_orientation import (  # noqa: E402
    RingCrossing,
    PointState,
    annular_corridor_labels,
    collect_policy_states,
    load_point_matching_detection,
)
from render_ring_compounded_rotation import (  # noqa: E402
    extract_full_length_ring_fields,
)
from render_tangential_method_comparison import draw_rings  # noqa: E402
from render_twok_reconnected_orientation import (  # noqa: E402
    isolated_global_crop,
    report,
)
from phenotypic.enhance import StructureSmoothing  # noqa: E402
from phenotypic.measure import MeasureOrientationZones  # noqa: E402
from phenotypic.measure._measure_orientation_zones import (  # noqa: E402
    _FIBER_AXIS_OFFSET,
    zone_selector,
)
from phenotypic.sdk_.orientation_fields import (  # noqa: E402
    LiteralSkeletonRingCrossingTransform,
    literal_crossing_ring_profile,
    literal_skeleton_ring_crossings,
    plot_literal_crossing_map,
    plot_literal_crossing_outward_profile,
    plot_literal_crossing_population,
)


ANALYSIS_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = ANALYSIS_DIR / "artifacts"
COLONIES = (("R3C4", 24), ("R4C6", 36))
POLICY = "independent_many_to_one"
CED_PARAMETERS = {
    "num_iter": 30,
    "sigma": 1.5,
    "rho": 3.0,
    "dt": 0.1,
    "alpha": 0.001,
    "C": 90.0,
}
LOCAL_TILT_NORM = Normalize(vmin=-90.0, vmax=90.0)
CUMULATIVE_NORM = Normalize(vmin=-180.0, vmax=180.0)


@dataclass(frozen=True)
class CrossingAnalysis:
    """Orientation evidence collected for one preprocessing condition."""

    condition: str
    profiles: dict
    points: list[RingCrossing]
    reliable_skeleton: np.ndarray
    states: dict[int, PointState]
    radii: np.ndarray
    fiber_axis: np.ndarray
    coherence: np.ndarray
    transform: LiteralSkeletonRingCrossingTransform


def apply_ced_preserving_object_map(section, parameters: dict | None = None):
    """Apply branch-scale CED while preserving detection and inoculum geometry.

    Args:
        section: Isolated colony crop with a fixed TwoK object map.
        parameters: Optional `StructureSmoothing` keyword arguments. The
            comparison parameters are used when omitted.

    Returns:
        Copy whose ``detect_mat`` has been diffusion-smoothed and whose object
        map is identical to the input.
    """
    objmap = np.asarray(section.objmap[:]).copy()
    resolved_parameters = CED_PARAMETERS if parameters is None else parameters
    smoothed = StructureSmoothing(**resolved_parameters).apply(
        section,
        inplace=False,
    )
    smoothed.objmap[:] = objmap
    return smoothed


def collect_crossing_analysis(
    section,
    condition: str,
    geometry_reference: CrossingAnalysis | None = None,
) -> CrossingAnalysis:
    """Collect literal crossings using optional fixed reference geometry.

    Args:
        section: Isolated colony crop containing the orientation source.
        condition: Display and export label for this preprocessing condition.
        geometry_reference: Optional control analysis whose object mask,
            inoculum center, distance map, exclusion radius, and outer radius
            define the radial sampling geometry.

    Returns:
        Full-length crossing evidence and many-to-one states.

    Raises:
        RuntimeError: If a condition changes fixed object geometry.
    """
    operation = MeasureOrientationZones(
        radial_ring_width=8.0,
        long_range_lag=32.0,
        quiver_block=24,
    )
    initial_profiles = extract_profiles(section, operation)
    profiles, phi, coherence, selector = extract_full_length_ring_fields(
        section,
        operation,
        initial_profiles,
    )
    if geometry_reference is not None:
        reference = geometry_reference.profiles
        if not np.array_equal(profiles["obj_mask"], reference["obj_mask"]):
            raise RuntimeError("CED condition changed the fixed object mask")
        if not np.allclose(profiles["centre"], reference["centre"]):
            raise RuntimeError(
                "CED condition changed the fixed inoculum center"
            )
        if not np.allclose(profiles["dist_map"], reference["dist_map"]):
            raise RuntimeError(
                "CED condition changed the fixed radial distance map"
            )
        profiles = dict(profiles)
        for key in (
            "inner_radius",
            "outer_radius",
            "object_extent_radius",
            "obj_mask",
            "dist_map",
            "centre",
        ):
            profiles[key] = reference[key]
        selector = zone_selector(
            profiles["dist_map"],
            profiles["inner_radius"],
            profiles["outer_radius"],
            profiles["obj_mask"],
            "Mask",
        )
    n_rings = int(
        np.floor(
            (profiles["outer_radius"] - profiles["inner_radius"])
            / profiles["ring_width"]
            + 1e-12
        )
    )
    radii = (
        profiles["inner_radius"]
        + (np.arange(n_rings, dtype=float) + 0.5) * profiles["ring_width"]
    )
    transform = literal_skeleton_ring_crossings(
        profiles["obj_mask"],
        phi + _FIBER_AXIS_OFFSET,
        coherence,
        profiles["dist_map"],
        profiles["centre"],
        radii,
        selector=selector,
    )
    points = list(transform.crossings)
    reliable_skeleton = transform.reliable_skeleton
    corridor_labels = annular_corridor_labels(
        points,
        reliable_skeleton,
        profiles["dist_map"],
        radii,
    )
    states = collect_policy_states(
        points,
        POLICY,
        corridor_labels,
        ring_width=profiles["ring_width"],
    )
    return CrossingAnalysis(
        condition=condition,
        profiles=profiles,
        points=points,
        reliable_skeleton=reliable_skeleton,
        states=states,
        radii=radii,
        fiber_axis=phi + _FIBER_AXIS_OFFSET,
        coherence=coherence,
        transform=transform,
    )


def axial_ring_consensus(
    analysis: CrossingAnalysis,
    *,
    minimum_points: int = 3,
    minimum_resultant: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return equal-crossing axial consensus and outward unwrapped change.

    Every literal crossing receives one vote. This population summary does not
    use parent-child links and therefore does not directly follow branches.

    Args:
        analysis: Collected literal-crossing evidence.
        minimum_points: Smallest ring sample accepted as a consensus.
        minimum_resultant: Smallest doubled-angle resultant accepted.

    Returns:
        Per-ring consensus, resultant, point count, and seam-safe consensus
        change within each contiguous supported run, all aligned to
        ``analysis.radii``. Every run restarts at zero after a gap or exact
        90-degree ambiguity.
    """
    profile = literal_crossing_ring_profile(
        analysis.transform,
        minimum_points=minimum_points,
        minimum_resultant=minimum_resultant,
    )
    return (
        profile.consensus_tilt,
        profile.resultant,
        profile.crossing_count,
        profile.contiguous_change,
    )


def _prepare_image_axis(axis, analysis: CrossingAnalysis) -> None:
    """Draw the condition-specific source, skeleton, and radial grid."""
    base = np.asarray(analysis.profiles["base"], dtype=float)
    finite = base[np.isfinite(base)]
    low, high = np.percentile(finite, (1.0, 99.8))
    axis.imshow(base, cmap="gray", vmin=low, vmax=high)
    axis.imshow(
        np.ma.masked_where(
            ~analysis.reliable_skeleton,
            analysis.reliable_skeleton,
        ),
        cmap="gray",
        vmin=0,
        vmax=1,
        alpha=0.22,
    )
    draw_rings(axis, analysis.profiles)
    axis.set_xlim(-0.5, base.shape[1] - 0.5)
    axis.set_ylim(base.shape[0] - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.set_axis_off()


def _draw_literal_crossings(axis, analysis: CrossingAnalysis) -> None:
    """Draw outward-normalized axial arrows at literal ring crossings.

    The structure-tensor orientation is axial and therefore has no measured
    polarity. Each equivalent axis is flipped, when needed, only so its
    arrowhead points toward increasing radius from the inoculum.
    """
    plot_literal_crossing_map(
        axis,
        np.asarray(analysis.profiles["base"], dtype=float),
        analysis.transform,
        cmap="Spectral",
        norm=LOCAL_TILT_NORM,
        boundary_radii=(
            analysis.profiles["inner_radius"],
            float(analysis.profiles["seg"].dense_end_radius),
            analysis.profiles["outer_radius"],
        ),
    )
    median_coherence = float(
        np.median([point.coherence for point in analysis.points])
    )
    median_resultant = float(
        np.median([point.resultant for point in analysis.points])
    )
    axis.set_title(
        f"{analysis.condition}: literal crossings\n"
        f"n={len(analysis.points)}; median coherence={median_coherence:.3f}; "
        f"median crossing resultant={median_resultant:.3f}\n"
        "arrowhead normalized toward increasing radius (not measured polarity)",
        fontsize=10,
    )


def render_literal_crossing_outward_metric(
    colony: str,
    analysis: CrossingAnalysis,
    *,
    output_tag: str = "",
) -> Path:
    """Render the literal-crossing outward-orientation metric without tracks."""
    figure = plt.figure(figsize=(14, 8.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.12, 1.0))
    overlay_axis = figure.add_subplot(grid[:, 0])
    local_axis = figure.add_subplot(grid[0, 1])
    cumulative_axis = figure.add_subplot(grid[1, 1])
    _draw_literal_crossings(overlay_axis, analysis)

    profile = literal_crossing_ring_profile(analysis.transform)
    local_scatter = plot_literal_crossing_population(
        local_axis,
        analysis.transform,
        cmap="Spectral",
        norm=LOCAL_TILT_NORM,
        title="Literal crossings and ring population consensus",
    )
    local_colorbar = figure.colorbar(
        local_scatter,
        ax=local_axis,
        fraction=0.05,
        pad=0.03,
        ticks=(-90.0, -45.0, 0.0, 45.0, 90.0),
    )
    local_colorbar.set_label("Local tilt (degrees)")

    cumulative_scatter, _resultant_axis = (
        plot_literal_crossing_outward_profile(
            cumulative_axis,
            profile,
            norm=CUMULATIVE_NORM,
        )
    )
    supported = profile.supported
    peak = np.degrees(profile.raw_peak)
    cumulative_axis.set_title(
        f"Outward population profile; raw peak={peak:.1f} degrees; "
        f"supported rings={int(supported.sum())}/{len(profile.radii)}"
    )
    cumulative_colorbar = figure.colorbar(
        cumulative_scatter,
        ax=cumulative_axis,
        fraction=0.05,
        pad=0.09,
        ticks=(-180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0),
    )
    cumulative_colorbar.set_label("Contiguous-run change (degrees)")
    figure.suptitle(
        f"{colony}: literal skeleton-ring crossing outward-orientation metric\n"
        "Each crossing has one vote; no branch correspondence is inferred",
        fontsize=14,
    )
    tag = f"_{output_tag}" if output_tag else ""
    output = (
        OUTPUT_DIR / f"twok_{colony}_literal_crossing_outward_metric{tag}.png"
    )
    figure.savefig(output, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output


def render_literal_crossing_before_after(
    colony: str,
    control: CrossingAnalysis,
    ced: CrossingAnalysis,
    *,
    output_tag: str = "",
) -> Path:
    """Render matched-scale literal-crossing arrow maps before and after CED."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    _draw_literal_crossings(axes[0], control)
    _draw_literal_crossings(axes[1], ced)
    scalar_mappable = plt.cm.ScalarMappable(
        norm=LOCAL_TILT_NORM,
        cmap="Spectral",
    )
    colorbar = figure.colorbar(
        scalar_mappable,
        ax=axes,
        orientation="horizontal",
        fraction=0.045,
        pad=0.025,
        ticks=(-90.0, -45.0, 0.0, 45.0, 90.0),
    )
    colorbar.set_label("Local radial-relative axial tilt (degrees)")
    figure.suptitle(
        f"{colony}: literal skeleton-ring crossings before and after "
        f"{ced.condition}\n"
        "Fixed mask, inoculum center, radial grid, and outward arrow normalization",
        fontsize=14,
    )
    tag = f"_{output_tag}" if output_tag else ""
    output = (
        OUTPUT_DIR / f"twok_{colony}_literal_crossing_before_after{tag}.png"
    )
    figure.savefig(output, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output


def _draw_many_to_one(axis, analysis: CrossingAnalysis) -> None:
    """Draw accepted independent many-to-one inheritance links."""
    _prepare_image_axis(axis, analysis)
    point_lookup = {point.point_id: point for point in analysis.points}
    segments: list[list[tuple[float, float]]] = []
    values: list[float] = []
    supported_rows: list[float] = []
    supported_cols: list[float] = []
    for point in analysis.points:
        state = analysis.states[point.point_id]
        if not np.isfinite(state.cumulative_signed):
            continue
        supported_rows.append(point.row)
        supported_cols.append(point.col)
        if state.parent_id is None:
            continue
        parent = point_lookup[state.parent_id]
        segments.append([(parent.col, parent.row), (point.col, point.row)])
        values.append(np.degrees(state.cumulative_signed))
    if segments:
        collection = LineCollection(
            segments,
            cmap="Spectral",
            norm=CUMULATIVE_NORM,
            linewidths=2.0,
            alpha=0.95,
        )
        collection.set_array(np.asarray(values))
        axis.add_collection(collection)
    axis.scatter(
        supported_cols,
        supported_rows,
        s=8,
        facecolors="white",
        edgecolors="black",
        linewidths=0.25,
        alpha=0.9,
    )
    peak = float(np.max(np.abs(values))) if values else np.nan
    axis.set_title(
        f"{analysis.condition}: independent many-to-one\n"
        f"accepted edges={len(segments)}; supported={len(supported_rows)}/"
        f"{len(analysis.points)}; raw peak={peak:.1f} degrees",
        fontsize=10,
    )


def render_ced_overlay_comparison(
    colony: str,
    control: CrossingAnalysis,
    ced: CrossingAnalysis,
    *,
    output_tag: str = "",
) -> Path:
    """Render literal crossings and inherited links before and after CED."""
    figure, axes = plt.subplots(
        2, 2, figsize=(14, 13), constrained_layout=True
    )
    _draw_literal_crossings(axes[0, 0], control)
    _draw_literal_crossings(axes[0, 1], ced)
    _draw_many_to_one(axes[1, 0], control)
    _draw_many_to_one(axes[1, 1], ced)
    local_mappable = plt.cm.ScalarMappable(
        norm=LOCAL_TILT_NORM, cmap="Spectral"
    )
    cumulative_mappable = plt.cm.ScalarMappable(
        norm=CUMULATIVE_NORM,
        cmap="Spectral",
    )
    top_colorbar = figure.colorbar(
        local_mappable,
        ax=axes[0, :],
        orientation="horizontal",
        fraction=0.035,
        pad=0.02,
        ticks=(-90.0, -45.0, 0.0, 45.0, 90.0),
    )
    top_colorbar.set_label("Local radial-relative axial tilt (degrees)")
    bottom_colorbar = figure.colorbar(
        cumulative_mappable,
        ax=axes[1, :],
        orientation="horizontal",
        fraction=0.035,
        pad=0.02,
        ticks=(-180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0),
    )
    bottom_colorbar.set_label(
        "Inherited cumulative fiber-axis change (degrees)"
    )
    figure.suptitle(
        f"{colony}: CED effect on literal skeleton-ring orientation evidence\n"
        "Object mask, inoculum center, 8 px rings, and matching guards held fixed",
        fontsize=14,
    )
    tag = f"_{output_tag}" if output_tag else ""
    output = OUTPUT_DIR / (
        f"twok_{colony}_ced_point_crossing_comparison{tag}_2x2.png"
    )
    figure.savefig(output, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output


def render_population_trend_comparison(
    colony: str,
    control: CrossingAnalysis,
    ced: CrossingAnalysis,
    *,
    output_tag: str = "",
) -> Path:
    """Render branch-tracking-free ring population consensus before/after CED."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for column, analysis in enumerate((control, ced)):
        profile = literal_crossing_ring_profile(analysis.transform)
        top = axes[0, column]
        plot_literal_crossing_population(
            top,
            analysis.transform,
            cmap="Spectral",
            norm=LOCAL_TILT_NORM,
            title=f"{analysis.condition}: literal crossing population",
        )

        bottom = axes[1, column]
        scatter, _reliability_axis = plot_literal_crossing_outward_profile(
            bottom,
            profile,
            norm=CUMULATIVE_NORM,
        )
        supported = profile.supported
        maximum = np.degrees(profile.raw_peak)
        bottom.set_title(
            f"Population trend; raw peak={maximum:.1f} degrees; "
            f"supported rings={int(supported.sum())}/{len(profile.radii)}"
        )
    colorbar = figure.colorbar(
        scatter,
        ax=axes[1, :],
        orientation="horizontal",
        fraction=0.045,
        pad=0.12,
        ticks=(-180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0),
    )
    colorbar.set_label(
        "Contiguous-run ring-population orientation change (degrees)"
    )
    figure.suptitle(
        f"{colony}: branch-tracking-free orientation trend from literal crossings\n"
        "Each crossing receives one vote within its ring",
        fontsize=14,
    )
    tag = f"_{output_tag}" if output_tag else ""
    output = OUTPUT_DIR / (
        f"twok_{colony}_ced_literal_crossing_trend{tag}_2x2.png"
    )
    figure.savefig(output, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output


def analysis_rows(
    colony: str,
    analysis: CrossingAnalysis,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    """Return per-ring and summary records for one condition."""
    mean, resultant, counts, cumulative = axial_ring_consensus(analysis)
    ring_rows = [
        {
            "Colony": colony,
            "Condition": analysis.condition,
            "Ring": ring,
            "RadiusPx": analysis.radii[ring],
            "Crossings": counts[ring],
            "ConsensusTiltDeg": np.degrees(mean[ring]),
            "RingResultant": resultant[ring],
            "ConsensusChangeDeg": np.degrees(cumulative[ring]),
        }
        for ring in range(analysis.radii.size)
    ]
    supported_states = [
        state
        for state in analysis.states.values()
        if np.isfinite(state.cumulative_signed)
    ]
    matched_states = [
        state for state in supported_states if state.parent_id is not None
    ]
    cumulative_values = np.asarray(
        [state.cumulative_signed for state in matched_states],
        dtype=float,
    )
    supported_rings = np.isfinite(cumulative)
    summary = {
        "Colony": colony,
        "Condition": analysis.condition,
        "Crossings": len(analysis.points),
        "ReliableSkeletonPixels": int(analysis.reliable_skeleton.sum()),
        "MedianCrossingCoherence": float(
            np.median([point.coherence for point in analysis.points])
        ),
        "MedianCrossingResultant": float(
            np.median([point.resultant for point in analysis.points])
        ),
        "ManyToOneSupportedPoints": len(supported_states),
        "ManyToOneAcceptedEdges": len(matched_states),
        "ManyToOneRawPeakDeg": (
            float(np.max(np.abs(np.degrees(cumulative_values))))
            if cumulative_values.size
            else np.nan
        ),
        "PopulationSupportedRings": int(supported_rings.sum()),
        "PopulationRawPeakDeg": (
            float(np.max(np.abs(np.degrees(cumulative[supported_rings]))))
            if supported_rings.any()
            else np.nan
        ),
    }
    return ring_rows, summary


def render_all_ced_point_crossing_comparisons() -> None:
    """Run CED comparisons on both selected real colonies."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detected = load_point_matching_detection()
    ring_rows: list[dict[str, float | int | str]] = []
    summaries: list[dict[str, float | int | str]] = []
    for colony, label in COLONIES:
        section = isolated_global_crop(
            detected,
            label,
            label_centroid(detected, label),
        )
        control = collect_crossing_analysis(section, "Original")
        report(f"Applying CED to {colony}: {CED_PARAMETERS}")
        ced_section = apply_ced_preserving_object_map(section)
        ced = collect_crossing_analysis(
            ced_section,
            "CED",
            geometry_reference=control,
        )
        overlay = render_ced_overlay_comparison(colony, control, ced)
        trend = render_population_trend_comparison(colony, control, ced)
        report(str(overlay))
        report(str(trend))
        for analysis in (control, ced):
            condition_rows, summary = analysis_rows(colony, analysis)
            ring_rows.extend(condition_rows)
            summaries.append(summary)

    ring_output = OUTPUT_DIR / "twok_ced_literal_crossing_ring_profiles.csv"
    summary_output = OUTPUT_DIR / "twok_ced_literal_crossing_summary.csv"
    pd.DataFrame(ring_rows).to_csv(ring_output, index=False)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(summary_output, index=False)
    report(str(ring_output))
    report(str(summary_output))
    print(summary_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    render_all_ced_point_crossing_comparisons()
