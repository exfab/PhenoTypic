from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

sys.path.insert(0, "/private/tmp")

from render_twok_reconnected_orientation import (  # noqa: E402
    isolated_global_crop,
    load_twok_detection,
    report,
)

from phenotypic.measure import MeasureOrientationZones  # noqa: E402
from phenotypic.measure._measure_orientation_zones import (  # noqa: E402
    _FIBER_AXIS_OFFSET,
    _RADIAL_RELATIVE_MIN_COHERENCE,
    _RADIAL_RELATIVE_N_SECTORS,
    cumulative_ring_rotation_profile,
    radial_ring_orientation_profile,
    radial_ring_sector_field,
    signed_radial_relative_field,
    zone_selector,
)
from phenotypic.util._matched_ring_rotation import (  # noqa: E402
    matched_ring_cumulative_rotation_profile,
    matched_tracks_to_ring_sector_values,
)


OUTPUT_DIR = Path(
    "/Users/alex/.codex/visualizations/2026/07/15/"
    "019f6340-b68c-7a81-b738-983ed6ea1a27/orientation-real-image"
)
COLONIES = (
    ("R3C4", 24),
    ("R4C6", 36),
)


def label_centroid(image, label: int) -> tuple[float, float]:
    """Return the full-image centroid of one detector label."""
    rows, cols = np.nonzero(np.asarray(image.objmap[:]) == label)
    if rows.size == 0:
        raise RuntimeError(f"Detector label {label} is absent")
    return float(rows.mean()), float(cols.mean())


def extract_profiles(section, operation: MeasureOrientationZones) -> dict:
    """Calculate fixed and matched cumulative profiles for one colony crop."""
    operation.measure(section)
    props, label2section = operation._prep(section)
    records = list(
        operation._iter_object_fields(section, props, label2section)
    )
    if len(records) != 1:
        raise RuntimeError(f"Expected one isolated object, got {len(records)}")
    (
        prop,
        seg,
        obj_mask,
        phi,
        coherence,
        _gradient,
        dist_map,
        centre,
    ) = records[0]
    tile, resolved_mask, resolved_centre = operation._resolve_tile(
        section,
        seg,
        prop,
        label2section,
    )
    if tile.shape != phi.shape or resolved_mask.shape != obj_mask.shape:
        raise RuntimeError("Resolved measurement tile shape changed")
    if not np.allclose(resolved_centre, centre):
        raise RuntimeError("Resolved inoculum center changed")
    signed_tilt, _turning, _magnitude, polar_angle = (
        signed_radial_relative_field(
            phi,
            centre,
            dist_map,
        )
    )
    inner_radius = float(seg.core_end_radius)
    outer_radius = min(
        float(seg.sparse_end_radius),
        float(seg.symmetric_radius),
    )
    selector = zone_selector(
        dist_map,
        inner_radius,
        outer_radius,
        obj_mask,
        "Mask",
    )
    radii, relative_orientation, relative_resultant = (
        radial_ring_orientation_profile(
            signed_tilt,
            polar_angle,
            coherence,
            dist_map,
            selector,
            inner_radius,
            outer_radius,
            operation.radial_ring_width,
            _RADIAL_RELATIVE_N_SECTORS,
        )
    )
    fiber_axis = phi + _FIBER_AXIS_OFFSET
    fiber_radii, fiber_orientation, fiber_resultant = (
        radial_ring_orientation_profile(
            fiber_axis,
            polar_angle,
            coherence,
            dist_map,
            selector,
            inner_radius,
            outer_radius,
            operation.radial_ring_width,
            _RADIAL_RELATIVE_N_SECTORS,
        )
    )
    if not np.array_equal(radii, fiber_radii):
        raise RuntimeError("Relative and fiber-axis ring grids differ")
    fixed_relative = cumulative_ring_rotation_profile(relative_orientation)
    fixed_fiber = cumulative_ring_rotation_profile(fiber_orientation)
    matched_seed, paths = matched_ring_cumulative_rotation_profile(
        radii,
        fiber_orientation,
        fiber_resultant,
        max_sector_shift=2,
    )
    matched_fiber = matched_tracks_to_ring_sector_values(matched_seed, paths)
    reliable_structure = (
        selector
        & np.isfinite(coherence)
        & (coherence >= _RADIAL_RELATIVE_MIN_COHERENCE)
    )
    fields = [
        radial_ring_sector_field(
            values,
            polar_angle,
            dist_map,
            reliable_structure,
            inner_radius,
            operation.radial_ring_width,
        )
        for values in (fixed_relative, fixed_fiber, matched_fiber)
    ]
    return {
        "base": np.asarray(tile, dtype=float),
        "seg": seg,
        "centre": centre,
        "radii": radii,
        "inner_radius": inner_radius,
        "outer_radius": outer_radius,
        "fixed_relative": fixed_relative,
        "fixed_fiber": fixed_fiber,
        "matched_fiber": matched_fiber,
        "matched_seed": matched_seed,
        "paths": paths,
        "fields": fields,
        "fiber_resultant": fiber_resultant,
        "fiber_orientation": fiber_orientation,
        "polar_angle": polar_angle,
        "dist_map": dist_map,
        "reliable_structure": reliable_structure,
        "ring_width": operation.radial_ring_width,
        "relative_resultant": relative_resultant,
    }


def profile_metrics(values: np.ndarray) -> tuple[float, float, float]:
    """Return support, absolute 95th percentile, and raw absolute maximum."""
    finite = np.isfinite(values)
    support = float(finite.sum()) / float(values.size) if values.size else 0.0
    if not finite.any():
        return support, np.nan, np.nan
    degrees = np.abs(np.degrees(values[finite]))
    return support, float(np.percentile(degrees, 95.0)), float(np.max(degrees))


def render_colony(
    section,
    colony: str,
    profiles: dict,
) -> tuple[Path, dict[str, float | str]]:
    """Render three fixed/matched calculations and their polar lattices."""
    base = profiles["base"]
    finite_base = base[np.isfinite(base)]
    low, high = np.percentile(finite_base, (1.0, 99.8))
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.45, 0.75)},
    )
    cmap = plt.get_cmap("Spectral").copy()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    norm = Normalize(vmin=-180.0, vmax=180.0)
    titles = (
        "Existing fixed sector\nradial-relative tilt accumulation",
        "Fixed-sector control\nfiber-axis rotation accumulation",
        "Matched nearby sectors\nfiber-axis rotation accumulation",
    )
    matrices = (
        profiles["fixed_relative"],
        profiles["fixed_fiber"],
        profiles["matched_fiber"],
    )
    centre = profiles["centre"]
    radii = profiles["radii"]
    fields = profiles["fields"]
    for column, (axis, title, field) in enumerate(
        zip(axes[0], titles, fields)
    ):
        axis.imshow(base, cmap="gray", vmin=low, vmax=high)
        masked_degrees = np.ma.masked_invalid(np.degrees(field))
        axis.imshow(masked_degrees, cmap=cmap, norm=norm, alpha=0.82)
        for radius in radii:
            axis.add_patch(
                plt.Circle(
                    (centre[1], centre[0]),
                    radius,
                    fill=False,
                    color="white",
                    linewidth=0.35,
                    alpha=0.22,
                )
            )
        for radius, style, width in (
            (profiles["inner_radius"], "-", 1.1),
            (float(profiles["seg"].dense_end_radius), "--", 0.9),
            (profiles["outer_radius"], "-", 1.1),
        ):
            axis.add_patch(
                plt.Circle(
                    (centre[1], centre[0]),
                    radius,
                    fill=False,
                    color="#E6E6E6",
                    linestyle=style,
                    linewidth=width,
                    alpha=0.8,
                )
            )
        support, p95, raw = profile_metrics(matrices[column])
        axis.set_title(
            f"{title}\nsupport {support:.1%} | |rotation| p95 {p95:.1f}° | raw {raw:.1f}°",
            fontsize=11,
        )
        axis.set_axis_off()

    n_sectors = profiles["paths"].shape[1]
    sector_angles = (np.arange(n_sectors, dtype=float) + 0.5) * (
        2.0 * np.pi / n_sectors
    )
    for seed in range(n_sectors):
        supported = np.flatnonzero(
            (profiles["paths"][:, seed] >= 0)
            & np.isfinite(profiles["matched_seed"][:, seed])
        )
        if supported.size < 2:
            continue
        sectors = profiles["paths"][supported, seed]
        angles = sector_angles[sectors]
        xs = centre[1] + radii[supported] * np.cos(angles)
        ys = centre[0] + radii[supported] * np.sin(angles)
        axes[0, 2].plot(xs, ys, color="white", linewidth=0.75, alpha=0.58)

    for axis in axes[0]:
        axis.set_xlim(-0.5, base.shape[1] - 0.5)
        axis.set_ylim(base.shape[0] - 0.5, -0.5)
        axis.set_aspect("equal")

    angle_edges = np.linspace(0.0, 360.0, n_sectors + 1)
    radial_width = float(radii[1] - radii[0]) if radii.size > 1 else 8.0
    radius_edges = np.r_[
        radii - radial_width / 2.0, radii[-1] + radial_width / 2.0
    ]
    for axis, title, matrix in zip(axes[1], titles, matrices):
        axis.pcolormesh(
            radius_edges,
            angle_edges,
            np.degrees(matrix).T,
            cmap=cmap,
            norm=norm,
            shading="flat",
        )
        axis.axvline(
            float(profiles["seg"].dense_end_radius),
            color="#333333",
            linestyle="--",
            linewidth=1.0,
        )
        axis.set_title(title.split("\n")[0] + " polar lattice")
        axis.set_xlabel("Radius from inoculum center (px)")
        axis.set_ylabel("Spatial sector angle (deg)")
        axis.set_ylim(0.0, 360.0)
        axis.set_yticks([0, 90, 180, 270, 360])
        axis.grid(color="#333333", alpha=0.12, linewidth=0.5)

    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        orientation="horizontal",
        fraction=0.035,
        pad=0.04,
    )
    colorbar.set_label("Cumulative signed orientation rotation (degrees)")
    figure.suptitle(
        f"{colony}: fixed-sector and matched-ring cumulative rotation",
        fontsize=15,
        y=1.02,
    )
    output = (
        OUTPUT_DIR / f"twok_{colony}_fixed_vs_matched_cumulative_rotation.png"
    )
    figure.savefig(
        output,
        dpi=180,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(figure)

    metric_row: dict[str, float | str] = {"Colony": colony}
    for prefix, matrix in zip(
        ("FixedRelative", "FixedFiber", "MatchedFiber"),
        matrices,
    ):
        support, p95, raw = profile_metrics(matrix)
        metric_row[f"{prefix}Support"] = support
        metric_row[f"{prefix}AbsP95Deg"] = p95
        metric_row[f"{prefix}RawMaxDeg"] = raw
    paths = profiles["paths"]
    transitions = (paths[1:] >= 0) & (paths[:-1] >= 0)
    changed = transitions & (paths[1:] != paths[:-1])
    metric_row["MatchedSectorChangeFraction"] = (
        float(changed.sum()) / float(transitions.sum())
        if transitions.any()
        else np.nan
    )
    seed_persistence = np.sum(paths >= 0, axis=0) / max(1, paths.shape[0])
    active_seeds = seed_persistence > 0.0
    metric_row["MatchedMedianSeedPersistence"] = (
        float(np.median(seed_persistence[active_seeds]))
        if active_seeds.any()
        else 0.0
    )
    return output, metric_row


def render_matched_ring_comparison() -> None:
    """Render fixed-sector and matched-ring comparisons for both colonies."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detected, _old = load_twok_detection()
    metrics: list[dict[str, float | str]] = []
    for colony, label in COLONIES:
        centroid = label_centroid(detected, label)
        section = isolated_global_crop(detected, label, centroid)
        operation = MeasureOrientationZones(
            radial_ring_width=8.0,
            long_range_lag=32.0,
            quiver_block=24,
        )
        profiles = extract_profiles(section, operation)
        output, row = render_colony(section, colony, profiles)
        row["DetectorLabel"] = label
        metrics.append(row)
        report(str(output))
    metrics_output = (
        OUTPUT_DIR / "twok_fixed_vs_matched_cumulative_rotation.csv"
    )
    pd.DataFrame(metrics).to_csv(metrics_output, index=False)
    report(str(metrics_output))
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == "__main__":
    render_matched_ring_comparison()
