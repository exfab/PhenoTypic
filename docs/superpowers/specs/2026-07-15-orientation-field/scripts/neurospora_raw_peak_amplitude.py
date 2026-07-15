from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.measure import regionprops

import neurospora_orientation_samples as base
import neurospora_radial_relative_samples as radial
from phenotypic.measure import MeasureOrientationZones


SELECTED = radial.DISPLAY_PAIR


def raw_and_normalized_profile(field, zone: str) -> tuple[np.ndarray, np.ndarray]:
    """Return density-sensitive evidence and its count-normalized distribution."""
    selector = field.valid_selectors[zone]
    low, high = np.percentile(field.tile[selector], (5.0, 99.5))
    intensity = np.clip((field.tile - low) / max(high - low, 1e-12), 0.0, 1.0)
    alignment = np.clip(np.cos(2.0 * field.delta), 0.0, 1.0)
    evidence = intensity * field.coherence * alignment * selector
    raw, _ = np.histogram(
        np.mod(field.alpha, 2.0 * np.pi),
        bins=np.linspace(0.0, 2.0 * np.pi, radial.N_ANGLE_BINS + 1),
        weights=evidence,
    )
    raw = gaussian_filter1d(raw, 1.4, mode="wrap")
    normalized = raw / max(float(raw.sum()), 1e-12)
    return raw, normalized


def render_raw_peak_diagnostic(fields, output_path: Path) -> None:
    """Contrast density-sensitive raw amplitude with normalized angle shape."""
    figure = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1.15, 1.0, 1.0))
    degrees = np.linspace(2.5, 357.5, radial.N_ANGLE_BINS)
    radians = np.radians(degrees)
    colors = {"Dense": "#CC79A7", "Sparse": "#D55E00"}
    for row, field in enumerate(fields):
        image_axis = figure.add_subplot(grid[row, 0])
        base.show_actual_layer(image_axis, field.tile, f"Colony {row + 1}: actual detect_mat")
        radial.add_local_axes(image_axis, field, "Overall")
        base.draw_zone_boundaries(image_axis, field.centre, field.radii)

        raw_axis = figure.add_subplot(grid[row, 1], projection="polar")
        normalized_axis = figure.add_subplot(grid[row, 2], projection="polar")
        diagnostics = []
        for zone in ("Dense", "Sparse"):
            raw, normalized = raw_and_normalized_profile(field, zone)
            raw_axis.plot(
                np.r_[radians, radians[0]],
                np.r_[raw, raw[0]],
                color=colors[zone],
                lw=2,
                label=zone,
            )
            normalized_axis.plot(
                np.r_[radians, radians[0]],
                np.r_[normalized, normalized[0]],
                color=colors[zone],
                lw=2,
                label=zone,
            )
            diagnostics.append(
                f"{zone}: raw peak={raw.max():.2f}, total={raw.sum():.1f}, "
                f"normalized peak={normalized.max():.3f}"
            )
        for axis in (raw_axis, normalized_axis):
            axis.set_theta_zero_location("E")
            axis.set_theta_direction(-1)
            axis.set_yticklabels([])
            axis.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.2, -0.05))
        raw_axis.set_title(
            "RAW peak amplitude\nDENSITY-SENSITIVE diagnostic\n" + diagnostics[0] + "\n" + diagnostics[1],
            fontsize=9,
            color="#A33A00",
        )
        normalized_axis.set_title(
            "Normalized angular distribution\nCOUNT-SCALE-INVARIANT shape\n"
            "multiplying all branch evidence leaves this unchanged",
            fontsize=10,
            color="#003660",
        )
    figure.suptitle(
        "Formation-angle evidence: retain raw amplitude, but keep it outside the orientation phenotype",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def create_raw_peak_figure() -> None:
    """Build the requested raw-versus-normalized diagnostic on the chosen pair."""
    segmented, _ = base.reproduce_notebook_segmentation()
    operation = MeasureOrientationZones(quiver_block=base.BLOCK)
    props = {int(prop.label): prop for prop in regionprops(
        segmented.objmap[:],
        intensity_image=segmented.gray[:].astype(np.float64, copy=False),
    )}
    fields = []
    for label, section in SELECTED:
        field = radial.build_colony_field(segmented, props[label], section, operation)
        if field is None:
            raise RuntimeError(f"Could not rebuild selected label {label}")
        fields.append(field)
    render_raw_peak_diagnostic(
        fields,
        base.OUTPUT_DIR / "09_raw_vs_normalized_formation_peaks.png",
    )


if __name__ == "__main__":
    create_raw_peak_figure()
