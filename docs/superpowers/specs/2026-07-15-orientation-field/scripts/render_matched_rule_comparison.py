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

from render_matched_ring_comparison import (  # noqa: E402
    extract_profiles,
    label_centroid,
    profile_metrics,
)
from render_twok_reconnected_orientation import (  # noqa: E402
    isolated_global_crop,
    load_twok_detection,
    report,
)
from phenotypic.measure import MeasureOrientationZones  # noqa: E402
from phenotypic.measure._measure_orientation_zones import (  # noqa: E402
    radial_ring_sector_field,
)
from phenotypic.util._matched_ring_rotation import (  # noqa: E402
    matched_ring_cumulative_rotation_profile,
    matched_tracks_to_ring_sector_values,
)


OUTPUT_DIR = Path(
    "/Users/alex/.codex/visualizations/2026/07/15/"
    "019f6340-b68c-7a81-b738-983ed6ea1a27/orientation-real-image"
)
COLONIES = (("R3C4", 24), ("R4C6", 36))
VARIANTS = (
    ("Strict baseline", False, False),
    ("Gap bridging only", True, False),
    ("Segment restarts only", False, True),
    ("Gap bridging + restarts", True, True),
)


def variant_fields(profiles: dict) -> list[dict]:
    """Calculate tracks and projected lattices for four continuity policies."""
    outputs: list[dict] = []
    for title, allow_gaps, allow_restarts in VARIANTS:
        cumulative, paths = matched_ring_cumulative_rotation_profile(
            profiles["radii"],
            profiles["fiber_orientation"],
            profiles["fiber_resultant"],
            max_sector_shift=2,
            allow_gap_bridging=allow_gaps,
            allow_restarts=allow_restarts,
        )
        field = matched_tracks_to_ring_sector_values(cumulative, paths)
        outputs.append(
            {
                "title": title,
                "allow_gaps": allow_gaps,
                "allow_restarts": allow_restarts,
                "cumulative": cumulative,
                "paths": paths,
                "field": field,
            }
        )
    return outputs


def render_rule_comparison(
    colony: str,
    label: int,
    profiles: dict,
) -> tuple[Path, list[dict[str, float | str | int | bool]]]:
    """Render one 2x2 polar-lattice comparison for a measured colony."""
    variants = variant_fields(profiles)
    radii = profiles["radii"]
    n_sectors = variants[0]["field"].shape[1]
    angle_edges = np.linspace(0.0, 360.0, n_sectors + 1)
    radial_width = float(radii[1] - radii[0]) if radii.size > 1 else 8.0
    radius_edges = np.r_[
        radii - radial_width / 2.0,
        radii[-1] + radial_width / 2.0,
    ]
    cmap = plt.get_cmap("Spectral").copy()
    cmap.set_bad("white")
    norm = Normalize(vmin=-180.0, vmax=180.0)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    baseline_finite = np.isfinite(variants[0]["field"])
    rows: list[dict[str, float | str | int | bool]] = []
    for axis, result in zip(axes.flat, variants):
        variant = result["title"]
        field = result["field"]
        allow_gaps = result["allow_gaps"]
        allow_restarts = result["allow_restarts"]
        axis.pcolormesh(
            radius_edges,
            angle_edges,
            np.degrees(field).T,
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
        support, p95, raw = profile_metrics(field)
        finite = np.isfinite(field)
        added = int(np.count_nonzero(finite & ~baseline_finite))
        value_basis = "segment-relative" if allow_restarts else "inoculum-path"
        axis.set_title(
            f"{variant}\n"
            f"{value_basis} | support {support:.1%} | "
            f"|rotation| p95 {p95:.1f}° | raw {raw:.1f}°\n"
            f"+{added} cells versus strict",
            fontsize=10.5,
        )
        axis.set_xlabel("Radius from inoculum center (px)")
        axis.set_ylabel("Spatial sector angle (deg)")
        axis.tick_params(labelbottom=True)
        axis.set_ylim(0.0, 360.0)
        axis.set_yticks([0, 90, 180, 270, 360])
        axis.grid(color="#333333", alpha=0.12, linewidth=0.5)
        rows.append(
            {
                "Colony": colony,
                "DetectorLabel": label,
                "Variant": variant,
                "AllowGapBridging": allow_gaps,
                "AllowRestarts": allow_restarts,
                "ValueBasis": value_basis,
                "Support": support,
                "AbsP95Deg": p95,
                "RawMaxDeg": raw,
                "AdditionalCellsVsStrict": added,
            }
        )
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        orientation="horizontal",
        fraction=0.045,
        pad=0.055,
    )
    colorbar.set_label("Cumulative signed fiber-axis rotation (degrees)")
    figure.suptitle(
        f"{colony}: matched-ring gap and restart rule comparison",
        fontsize=15,
        y=1.03,
    )
    figure.text(
        0.5,
        -0.035,
        (
            "White cells have no assigned value. Gap bridging leaves skipped "
            "rings white; restarted segments reset to 0°. Dashed line: dense-zone boundary."
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )
    output = OUTPUT_DIR / f"twok_{colony}_matched_gap_restart_2x2.png"
    figure.savefig(
        output,
        dpi=190,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(figure)
    return output, rows


def render_overlay_comparison(
    colony: str,
    profiles: dict,
) -> Path:
    """Render the four continuity policies on the measured array tile."""
    variants = variant_fields(profiles)
    base = profiles["base"]
    finite_base = base[np.isfinite(base)]
    low, high = np.percentile(finite_base, (1.0, 99.8))
    centre = profiles["centre"]
    radii = profiles["radii"]
    n_sectors = variants[0]["paths"].shape[1]
    sector_angles = (np.arange(n_sectors, dtype=float) + 0.5) * (
        2.0 * np.pi / n_sectors
    )
    cmap = plt.get_cmap("Spectral").copy()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    norm = Normalize(vmin=-180.0, vmax=180.0)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 12),
        constrained_layout=True,
    )
    for axis, result in zip(axes.flat, variants):
        field = result["field"]
        local_field = radial_ring_sector_field(
            field,
            profiles["polar_angle"],
            profiles["dist_map"],
            profiles["reliable_structure"],
            profiles["inner_radius"],
            profiles["ring_width"],
        )
        axis.imshow(base, cmap="gray", vmin=low, vmax=high)
        axis.imshow(
            np.ma.masked_invalid(np.degrees(local_field)),
            cmap=cmap,
            norm=norm,
            alpha=0.82,
        )
        for radius in radii:
            axis.add_patch(
                plt.Circle(
                    (centre[1], centre[0]),
                    radius,
                    fill=False,
                    color="white",
                    linewidth=0.35,
                    alpha=0.20,
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
        if not result["allow_restarts"]:
            paths = result["paths"]
            cumulative = result["cumulative"]
            for seed in range(n_sectors):
                supported = np.flatnonzero(
                    (paths[:, seed] >= 0) & np.isfinite(cumulative[:, seed])
                )
                if supported.size < 2:
                    continue
                sectors = paths[supported, seed]
                angles = sector_angles[sectors]
                xs = centre[1] + radii[supported] * np.cos(angles)
                ys = centre[0] + radii[supported] * np.sin(angles)
                for point_index in range(1, supported.size):
                    is_bridge = (
                        supported[point_index] - supported[point_index - 1] > 1
                    )
                    axis.plot(
                        xs[point_index - 1 : point_index + 1],
                        ys[point_index - 1 : point_index + 1],
                        color="white",
                        linestyle="--" if is_bridge else "-",
                        linewidth=0.75,
                        alpha=0.50,
                    )
        support, p95, raw = profile_metrics(field)
        value_basis = (
            "segment-relative" if result["allow_restarts"] else "inoculum-path"
        )
        axis.set_title(
            f"{result['title']}\n{value_basis} | support {support:.1%} | "
            f"|rotation| p95 {p95:.1f}° | raw {raw:.1f}°",
            fontsize=10,
        )
        axis.set_xlim(-0.5, base.shape[1] - 0.5)
        axis.set_ylim(base.shape[0] - 0.5, -0.5)
        axis.set_aspect("equal")
        axis.set_axis_off()
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        orientation="horizontal",
        fraction=0.035,
        pad=0.04,
    )
    colorbar.set_label("Cumulative signed fiber-axis rotation (degrees)")
    figure.suptitle(
        f"{colony}: matched-ring continuity rules on measured array",
        fontsize=15,
        y=1.02,
    )
    figure.text(
        0.5,
        -0.025,
        (
            "Solid white: adjacent observed match. Dashed white: bridged gap. "
            "Paths are hidden for restart panels because segments are discontinuous."
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )
    output = OUTPUT_DIR / f"twok_{colony}_matched_gap_restart_overlay_2x2.png"
    figure.savefig(
        output,
        dpi=190,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(figure)
    return output


def render_all_rule_comparisons() -> None:
    """Render both colonies and export their four-condition summary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detected, _old = load_twok_detection()
    summary_rows: list[dict[str, float | str | int | bool]] = []
    for colony, label in COLONIES:
        section = isolated_global_crop(
            detected,
            label,
            label_centroid(detected, label),
        )
        profiles = extract_profiles(
            section,
            MeasureOrientationZones(
                radial_ring_width=8.0,
                long_range_lag=32.0,
                quiver_block=24,
            ),
        )
        output, rows = render_rule_comparison(
            colony,
            label,
            profiles,
        )
        overlay_output = render_overlay_comparison(colony, profiles)
        summary_rows.extend(rows)
        report(str(output))
        report(str(overlay_output))
    summary = pd.DataFrame(summary_rows)
    csv_output = OUTPUT_DIR / "twok_matched_gap_restart_comparison.csv"
    summary.to_csv(csv_output, index=False)
    report(str(csv_output))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    render_all_rule_comparisons()
