from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ANALYSIS_DIR / "artifacts/twok_ring_compounded_rotation_profiles.csv"
OUTPUT_PATH = ANALYSIS_DIR / "artifacts/twok_R3C4_ring_gate_diagnostic.png"
MAXIMUM_ABS_TILT_DEG = 75.0
MINIMUM_RING_RESULTANT = 0.15
MINIMUM_SECTOR_SUPPORT = 3.0 / 36.0
BLUE = "#0072B2"
ORANGE = "#D55E00"
PURPLE = "#7B3294"
GRAY = "#666666"


def render_r3c4_ring_gate_diagnostic() -> Path:
    """Show why R3C4 compounding stops despite available orientation.

    Returns:
        Path to the rendered diagnostic PNG.
    """
    profiles = pd.read_csv(INPUT_PATH)
    colony = profiles.loc[profiles["Colony"].eq("R3C4")].copy()
    if colony.empty:
        raise RuntimeError("R3C4 is absent from the ring profile table")

    radii = colony["RadiusPx"].to_numpy(dtype=float)
    tilt = colony["MedianTiltDeg"].to_numpy(dtype=float)
    resultant = colony["RingResultant"].to_numpy(dtype=float)
    support = colony["SectorSupport"].to_numpy(dtype=float)
    cumulative = colony["MedianCumulativeDeg"].to_numpy(dtype=float)
    stop_candidates = np.flatnonzero(
        np.isfinite(tilt) & (np.abs(tilt) > MAXIMUM_ABS_TILT_DEG)
    )
    if stop_candidates.size == 0:
        raise RuntimeError("R3C4 has no near-tangential stopping ring")
    stop_ring = int(stop_candidates[0])

    predicted_step = np.full_like(tilt, np.nan)
    valid_pairs = np.isfinite(tilt[:-1]) & np.isfinite(tilt[1:])
    predicted_step[:-1][valid_pairs] = np.degrees(
        np.tan(np.radians(tilt[:-1][valid_pairs]))
        * np.log(radii[1:][valid_pairs] / radii[:-1][valid_pairs])
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.2, 1.0, 0.9)},
    )

    before_stop = np.arange(tilt.size) <= stop_ring
    after_stop = np.arange(tilt.size) > stop_ring
    axes[0].plot(
        radii[before_stop],
        tilt[before_stop],
        color=BLUE,
        marker="o",
        linewidth=1.8,
        label="Compounded segment",
    )
    axes[0].plot(
        radii[after_stop],
        tilt[after_stop],
        color=ORANGE,
        marker="o",
        markerfacecolor="none",
        linestyle=":",
        linewidth=1.5,
        label="Orientation still measured after stop",
    )
    for threshold in (-MAXIMUM_ABS_TILT_DEG, MAXIMUM_ABS_TILT_DEG):
        axes[0].axhline(
            threshold,
            color=ORANGE,
            linestyle="--",
            linewidth=1.0,
        )
    axes[0].axhline(0.0, color=GRAY, linewidth=0.8)
    axes[0].scatter(
        [radii[stop_ring]],
        [tilt[stop_ring]],
        color=ORANGE,
        edgecolor="white",
        linewidth=0.8,
        s=70,
        zorder=4,
    )
    axes[0].annotate(
        f"stop: median tilt {tilt[stop_ring]:.1f} degrees exceeds 75 degrees",
        xy=(radii[stop_ring], tilt[stop_ring]),
        xytext=(radii[stop_ring] + 8.0, 52.0),
        arrowprops={"arrowstyle": "->", "color": ORANGE},
        color=ORANGE,
    )
    axes[0].set_ylabel("Axial-median radial tilt (degrees)")
    axes[0].set_ylim(-100.0, 100.0)
    axes[0].set_title("Ring orientation remains available beyond the stopping ring")
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].grid(alpha=0.20)

    axes[1].plot(
        radii,
        resultant,
        color=PURPLE,
        marker="s",
        linewidth=1.6,
        label="Axial ring resultant",
    )
    axes[1].plot(
        radii,
        support,
        color=GRAY,
        marker="o",
        markerfacecolor="none",
        linewidth=1.4,
        label="Reliable-sector fraction",
    )
    axes[1].axhline(
        MINIMUM_RING_RESULTANT,
        color=PURPLE,
        linestyle="--",
        linewidth=1.0,
        label="Resultant threshold 0.15",
    )
    axes[1].axhline(
        MINIMUM_SECTOR_SUPPORT,
        color=GRAY,
        linestyle=":",
        linewidth=1.0,
        label="Three-sector minimum",
    )
    axes[1].set_ylabel("Fraction / resultant")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Consensus remains admissible through ring 16")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")
    axes[1].grid(alpha=0.20)

    axes[2].plot(
        radii,
        predicted_step,
        color=BLUE,
        marker="o",
        linewidth=1.5,
    )
    axes[2].axhline(0.0, color=GRAY, linewidth=0.8)
    axes[2].scatter(
        [radii[stop_ring]],
        [predicted_step[stop_ring]],
        color=ORANGE,
        edgecolor="white",
        linewidth=0.8,
        s=70,
        zorder=4,
    )
    axes[2].annotate(
        f"next-ring predictor: {predicted_step[stop_ring]:.1f} degrees",
        xy=(radii[stop_ring], predicted_step[stop_ring]),
        xytext=(radii[stop_ring] + 12.0, predicted_step[stop_ring] + 80.0),
        arrowprops={"arrowstyle": "->", "color": ORANGE},
        color=ORANGE,
    )
    axes[2].set_ylabel("Predicted path step (degrees)")
    axes[2].set_xlabel("Radius from inoculum center (px)")
    axes[2].set_title(
        "The tan(tilt) radial predictor becomes singular near 90 degrees"
    )
    axes[2].grid(alpha=0.20)

    stop_radius = radii[stop_ring]
    for axis in axes:
        axis.axvline(
            stop_radius,
            color=ORANGE,
            linestyle=":",
            linewidth=1.0,
        )
    finite_cumulative = np.flatnonzero(np.isfinite(cumulative))
    figure.suptitle(
        "R3C4 ring-level evidence and radial-compounding gate\n"
        f"8 px rings; cumulative output remains finite through ring "
        f"{finite_cumulative[-1]} of {tilt.size - 1}",
        fontsize=14,
    )
    figure.savefig(
        OUTPUT_PATH,
        dpi=190,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(figure)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(render_r3c4_ring_gate_diagnostic())
