from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from skimage.measure import regionprops

from render_twok_reconnected_orientation import (
    OUTPUT_DIR,
    isolated_global_crop,
    load_twok_detection,
)
from phenotypic.measure import MeasureOrientationZones


CANDIDATES = ((24, 23), (36, 35))
RING_WIDTH = 8.0
LONG_LAG = 32.0


def render_sholl_profile(
    section,
    operation: MeasureOrientationZones,
    label: int,
    grid_section: int,
) -> Path:
    """Render actual Sholl rings beside fixed-lag rotation profiles."""
    record = operation._cache[label]
    profile = record["ring_profile"]
    radii = np.asarray(profile["radii"], dtype=float)
    mean_tilt = np.asarray(profile["mean_absolute_tilt"], dtype=float)
    ring_support = np.asarray(profile["support"], dtype=float)
    midpoints = np.asarray(profile["pair_midpoints"], dtype=float)
    absolute_rotation = np.asarray(
        profile["mean_absolute_rotation"], dtype=float
    )
    signed_rotation = np.asarray(
        profile["mean_signed_rotation"], dtype=float
    )
    pair_support = np.asarray(profile["pair_support"], dtype=float)
    centre_row, centre_col = record["centroid_global"]
    radii_record = record["radii"]

    base = np.asarray(section.detect_mat[:], dtype=float)
    finite = base[np.isfinite(base)]
    lower, upper = np.percentile(finite, (1.0, 99.8))
    figure, (image_axis, profile_axis) = plt.subplots(
        1,
        2,
        figsize=(15, 7),
        gridspec_kw={"width_ratios": (1.05, 1.0)},
        constrained_layout=True,
    )
    image_axis.imshow(base, cmap="gray", vmin=lower, vmax=upper)
    for radius in radii:
        image_axis.add_patch(
            Circle(
                (centre_col, centre_row),
                radius,
                fill=False,
                edgecolor="white",
                linewidth=0.65,
                alpha=0.55,
            )
        )
    for radius in midpoints:
        image_axis.add_patch(
            Circle(
                (centre_col, centre_row),
                radius,
                fill=False,
                edgecolor="#E69F00",
                linewidth=0.85,
                linestyle=":",
                alpha=0.8,
            )
        )
    core_end = float(radii_record["core_end"])
    image_axis.add_patch(
        Circle(
            (centre_col, centre_row),
            core_end,
            fill=False,
            edgecolor="#DC267F",
            linewidth=2.0,
        )
    )
    image_axis.set_title(
        "Sholl-style orientation bands\n"
        f"white: {RING_WIDTH:g}px bands · orange: {LONG_LAG:g}px-pair midpoints"
    )
    image_axis.set_axis_off()

    dense_end = float(radii_record["dense_end"])
    sparse_end = float(radii_record["sparse_end"])
    profile_axis.axvspan(
        core_end,
        dense_end,
        color="#CC79A7",
        alpha=0.12,
        label="Dense zone",
    )
    profile_axis.axvspan(
        dense_end,
        sparse_end,
        color="#E69F00",
        alpha=0.10,
        label="Sparse zone",
    )
    profile_axis.plot(
        midpoints,
        absolute_rotation,
        color="#E69F00",
        marker="o",
        linewidth=2.0,
        label=f"Mean |rotation| over {LONG_LAG:g}px lag",
    )
    finite_signed = signed_rotation[np.isfinite(signed_rotation)]
    signed_limit = (
        float(np.max(np.abs(finite_signed))) if finite_signed.size else 1.0
    )
    signed_limit = max(signed_limit, 1e-6)
    scatter = profile_axis.scatter(
        midpoints,
        signed_rotation,
        c=signed_rotation,
        cmap="Spectral",
        norm=Normalize(vmin=-signed_limit, vmax=signed_limit),
        marker="D",
        s=50,
        edgecolor="#17324D",
        linewidth=0.6,
        label="Signed rotation",
        zorder=4,
    )
    profile_axis.axhline(0.0, color="#17324D", linewidth=0.8)
    profile_axis.set_xlabel("Pair midpoint radius from inoculum (px)")
    profile_axis.set_ylabel("Rotation (deg)")
    profile_axis.grid(alpha=0.25)
    support_axis = profile_axis.twinx()
    support_axis.plot(
        midpoints,
        pair_support,
        color="#777777",
        marker=".",
        linestyle="--",
        linewidth=1.3,
        label="Paired-sector support",
    )
    support_axis.set_ylabel("Reliable paired-sector fraction")
    support_axis.set_ylim(0.0, 1.0)
    handles, labels = profile_axis.get_legend_handles_labels()
    support_handles, support_labels = support_axis.get_legend_handles_labels()
    profile_axis.legend(
        handles + support_handles,
        labels + support_labels,
        loc="upper right",
        frameon=False,
    )
    figure.colorbar(
        scatter,
        ax=profile_axis,
        orientation="horizontal",
        location="bottom",
        pad=0.12,
        fraction=0.05,
        label="Signed rotation (deg): counterclockwise ← 0 → clockwise",
    )
    long_range = record["long_range"]
    summary_lines = []
    for region in ("Overall", "Dense", "Sparse", "DenseToSparse"):
        magnitude, signed, support = long_range[region]
        summary_lines.append(
            f"{region}: |Δ|={magnitude:.2f}°, signed={signed:.2f}°, support={support:.2f}"
        )
    profile_axis.text(
        0.01,
        0.01,
        "\n".join(summary_lines),
        transform=profile_axis.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    figure.suptitle(
        f"Label {label}, grid {grid_section}: Sholl rotation at {LONG_LAG:g}px lag",
        fontsize=15,
    )
    output = OUTPUT_DIR / f"twok_label_{label}_sholl_lag_{int(LONG_LAG)}.png"
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return output


def render_higher_scale_and_longer_sholl() -> None:
    """Render σ=32 bend and 32px-lag Sholl profiles for two candidates."""
    detected, _old_segmented = load_twok_detection()
    props = {
        int(prop.label): prop
        for prop in regionprops(np.asarray(detected.objmap[:]))
    }
    measurement_frames: list[pd.DataFrame] = []
    for label, grid_section in CANDIDATES:
        centroid = tuple(float(value) for value in props[label].centroid)
        section = isolated_global_crop(detected, label, centroid)
        operation = MeasureOrientationZones(
            quiver_block=24,
            radial_ring_width=RING_WIDTH,
            long_range_lag=LONG_LAG,
        )
        measurements = operation.measure(section)
        measurements.insert(0, "GridSection", grid_section)
        measurements.insert(0, "DetectorLabel", label)
        measurement_frames.append(measurements)

        bend_figure = operation.fiber_bend_overlay(
            section,
            base_layer="detect_mat",
            scale_set="broad",
        )
        bend_figure.update_layout(
            title=f"Label {label}, grid {grid_section}: higher-scale fiber bend"
        )
        bend_output = OUTPUT_DIR / f"twok_label_{label}_bend_sigma_8_16_32.png"
        bend_figure.write_image(str(bend_output), width=1800, height=900, scale=1)
        sholl_output = render_sholl_profile(
            section,
            operation,
            label,
            grid_section,
        )
        print(bend_output)
        print(sholl_output)

    output = OUTPUT_DIR / "twok_candidates_long_range_lag32_measurements.csv"
    pd.concat(measurement_frames, ignore_index=True).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    render_higher_scale_and_longer_sholl()
