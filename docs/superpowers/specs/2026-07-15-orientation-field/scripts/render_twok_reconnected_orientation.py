from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from skimage.measure import regionprops

from neurospora_orientation_samples import (
    CACHE_DIR,
    COORD1,
    COORD2,
    OUTPUT_DIR,
    WIDTH,
    load_notebook_base,
    reproduce_notebook_segmentation,
)
from phenotypic.detect import ManualGridPointDetector, TwoKFilamentousDetector
from phenotypic.measure import MeasureOrientationZones


COLONIES = (
    ("A", 35, 1116),
    ("B", 18, 626),
)
TWOK_OBJMAP = CACHE_DIR / "twok_branch_reconnection_objmap.npy"


def report(message: str) -> None:
    """Print a timestamped progress message."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_twok_detection():
    """Run or load the updated TwoK detector and restore the notebook layer."""
    base = load_notebook_base()
    old_segmented, _stats = reproduce_notebook_segmentation()
    composite = np.asarray(old_segmented.detect_mat[:], dtype=np.float32)
    if TWOK_OBJMAP.exists():
        report("Loading cached branch-reconnection TwoK objmap")
        detected = base.copy()
        objmap = np.asarray(np.load(TWOK_OBJMAP, mmap_mode="r"))
    else:
        report("Running TwoKFilamentousDetector(reconnect_scope='branches')")
        detector = TwoKFilamentousDetector(
            center_detector=ManualGridPointDetector(
                coord1=COORD1,
                coord2=COORD2,
                shape="disk",
                width=WIDTH,
            ),
            reconnect_scope="branches",
        )
        started = time.perf_counter()
        detected = detector.apply(base, inplace=False)
        report(f"TwoK detection finished in {time.perf_counter() - started:.1f}s")
        objmap = np.asarray(detected.objmap[:]).copy()
        np.save(TWOK_OBJMAP, objmap)
    # Hold the orientation-field source fixed at the notebook composite. This
    # isolates the effect of the improved branch mask and reconnection.
    detected.detect_mat[:] = composite
    # Setting an upstream image layer correctly invalidates downstream cached
    # object data, so restore the detector result after fixing the source layer.
    detected.objmap[:] = objmap
    return detected, old_segmented


def match_new_label(old_segmented, detected, old_label: int) -> tuple[int, int]:
    """Match a reference colony to the new label with maximum pixel overlap."""
    old_mask = np.asarray(old_segmented.objmap[:]) == old_label
    labels, counts = np.unique(
        np.asarray(detected.objmap[:])[old_mask],
        return_counts=True,
    )
    foreground = labels > 0
    if not foreground.any():
        raise RuntimeError(f"No TwoK label overlaps old label {old_label}")
    labels = labels[foreground]
    counts = counts[foreground]
    best = int(np.argmax(counts))
    return int(labels[best]), int(counts[best])


def old_label_centroid(old_segmented, old_label: int) -> tuple[float, float]:
    """Return the global centroid for one reference label."""
    props = {
        int(prop.label): prop
        for prop in regionprops(np.asarray(old_segmented.objmap[:]))
    }
    return tuple(float(value) for value in props[old_label].centroid)


def isolated_global_crop(
    image,
    label: int,
    centroid: tuple[float, float],
    half_width: int = 350,
):
    """Crop fixed global coordinates and retain only one physical colony."""
    center_row, center_col = (int(round(value)) for value in centroid)
    row_start = max(0, center_row - half_width)
    col_start = max(0, center_col - half_width)
    row_stop = min(image.shape[0], center_row + half_width)
    col_stop = min(image.shape[1], center_col + half_width)
    section = image[row_start:row_stop, col_start:col_stop]
    objmap = np.asarray(section.objmap[:])
    section.objmap[:] = np.where(objmap == label, label, 0)
    return section


def render_mask_comparison(
    old_segmented,
    detected,
    centroid: tuple[float, float],
    old_label: int,
    new_label: int,
    name: str,
) -> Path:
    """Show the old fragmented and updated reconnected masks side by side."""
    center_row, center_col = (int(round(value)) for value in centroid)
    half_width = 350
    row_slice = slice(max(0, center_row - half_width), center_row + half_width)
    col_slice = slice(max(0, center_col - half_width), center_col + half_width)
    base = np.asarray(
        detected.detect_mat[row_slice, col_slice],
        dtype=float,
    )
    low, high = np.percentile(base[np.isfinite(base)], (1.0, 99.8))
    overlays = (
        (
            np.asarray(old_segmented.objmap[row_slice, col_slice]) == old_label,
            "Notebook hysteresis",
        ),
        (
            np.asarray(detected.objmap[row_slice, col_slice]) == new_label,
            "TwoK + branch reconnection",
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for axis, (mask, title) in zip(axes, overlays):
        axis.imshow(base, cmap="gray", vmin=low, vmax=high)
        overlay = np.ma.masked_where(~mask, mask)
        axis.imshow(
            overlay,
            cmap=ListedColormap(["#00E5FF"]),
            interpolation="nearest",
            alpha=0.55,
        )
        axis.set_title(f"Colony {name}: {title}")
        axis.set_axis_off()
    output = OUTPUT_DIR / f"detector_mask_comparison_colony_{name.lower()}.png"
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return output


def render_measurement_summary(measurements: pd.DataFrame) -> Path:
    """Plot the primary TwoK-based zone measurements for both colonies."""
    authoritative = measurements.loc[
        measurements["Detector"] == "TwoKBranches"
    ].set_index("Colony")
    zones = ("Overall", "Dense", "Sparse")
    panels = (
        ("RadialTilt", "Radial-relative tilt (deg)", (0.0, 90.0)),
        ("OutwardTurning", "Outward turning (deg/px)", None),
        ("LongRangeRotation", "Long-range rotation (deg)", (0.0, 90.0)),
        (
            "LongRangeRotationSupport",
            "Long-range support (fraction)",
            (0.0, 1.0),
        ),
    )
    colors = {"A": "#0072B2", "B": "#D55E00"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    positions = np.arange(len(zones), dtype=float)
    width = 0.34
    for axis, (metric, title, limits) in zip(axes.flat, panels):
        for offset, colony in zip((-width / 2, width / 2), ("A", "B")):
            values = np.asarray(
                [
                    authoritative.loc[
                        colony,
                        f"OrientZones_{metric}-Mask-{zone}",
                    ]
                    for zone in zones
                ],
                dtype=float,
            )
            bars = axis.bar(
                positions + offset,
                np.nan_to_num(values, nan=0.0),
                width,
                label=f"Colony {colony}",
                color=colors[colony],
                alpha=0.88,
            )
            for bar, value in zip(bars, values):
                label = "n/a" if not np.isfinite(value) else f"{value:.2f}"
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        axis.set_title(title)
        axis.set_xticks(positions, zones)
        axis.grid(axis="y", alpha=0.25)
        if limits is not None:
            axis.set_ylim(*limits)
        elif axis.get_ylim()[1] > 0:
            axis.set_ylim(0.0, axis.get_ylim()[1] * 1.15)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        "Orientation-zone measurements from TwoK branch-reconnection masks",
        fontsize=15,
    )
    output = OUTPUT_DIR / "twok_orientation_measurement_summary.png"
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return output


def rerun_orientation_measurements() -> None:
    """Measure and render both colonies using the updated detector mask."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detected, old_segmented = load_twok_detection()
    measurement_frames: list[pd.DataFrame] = []
    for name, section_index, old_label in COLONIES:
        new_label, overlap_pixels = match_new_label(
            old_segmented,
            detected,
            old_label,
        )
        centroid = old_label_centroid(old_segmented, old_label)
        new_section = isolated_global_crop(detected, new_label, centroid)
        old_section = isolated_global_crop(old_segmented, old_label, centroid)
        operation = MeasureOrientationZones(quiver_block=24)
        measurements = operation.measure(new_section)
        measurements.insert(0, "Detector", "TwoKBranches")
        measurements.insert(0, "DetectorLabel", new_label)
        measurements.insert(0, "ReferenceGridSection", section_index)
        measurements.insert(0, "Colony", name)
        measurements.insert(
            4,
            "DetectedPixels",
            int(np.count_nonzero(new_section.objmask[:])),
        )
        measurements.insert(5, "OverlapWithReference", overlap_pixels)
        measurement_frames.append(measurements)

        old_measurements = MeasureOrientationZones(quiver_block=24).measure(
            old_section
        )
        old_measurements.insert(0, "Detector", "NotebookHysteresis")
        old_measurements.insert(0, "DetectorLabel", old_label)
        old_measurements.insert(0, "ReferenceGridSection", section_index)
        old_measurements.insert(0, "Colony", name)
        old_measurements.insert(
            4,
            "DetectedPixels",
            int(np.count_nonzero(old_section.objmask[:])),
        )
        old_measurements.insert(
            5,
            "OverlapWithReference",
            int(np.count_nonzero(old_section.objmask[:])),
        )
        measurement_frames.append(old_measurements)

        figure = operation.fiber_bend_overlay(
            new_section,
            base_layer="detect_mat",
            scale_set="balanced",
        )
        bend_output = (
            OUTPUT_DIR
            / f"twok_branch_reconnection_bend_colony_{name.lower()}.png"
        )
        figure.write_image(str(bend_output), width=1800, height=900, scale=1)
        comparison_output = render_mask_comparison(
            old_segmented,
            detected,
            centroid,
            old_label,
            new_label,
            name,
        )
        report(str(bend_output))
        report(str(comparison_output))

    measurements = pd.concat(measurement_frames, ignore_index=True)
    csv_output = OUTPUT_DIR / "twok_branch_reconnection_orientation_measurements.csv"
    measurements.to_csv(csv_output, index=False)
    summary_output = render_measurement_summary(measurements)
    report(str(csv_output))
    report(str(summary_output))
    report(
        measurements[
            [
                "Colony",
                "ReferenceGridSection",
                "DetectorLabel",
                "Detector",
                "DetectedPixels",
                "OverlapWithReference",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    rerun_orientation_measurements()
