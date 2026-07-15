from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import regionprops

from render_twok_reconnected_orientation import (
    OUTPUT_DIR,
    isolated_global_crop,
    load_twok_detection,
)
from phenotypic.measure import MeasureOrientationZones
from phenotypic.measure import _measure_orientation_zones as orientation_module
from phenotypic.measure._measure_orientation_zones import zone_selector
from phenotypic.util._nematic_bend import fiber_bend_field


CANDIDATES = ((24, 23), (36, 35))
SCALES = (16.0, 32.0, 48.0)
MIN_COHERENCE = 0.15
MIN_RESULTANT = 0.15


def scale_coverage(section, operation: MeasureOrientationZones, label: int):
    """Return bend magnitude and reliability coverage at each requested scale."""
    props, label_to_section = operation._prep(section)
    fields = list(
        operation._iter_object_fields(section, props, label_to_section)
    )
    if len(fields) != 1:
        raise RuntimeError(f"Expected one field for label {label}; got {len(fields)}")
    (
        _prop,
        seg,
        obj_mask,
        phi,
        coherence,
        _gradient,
        dist_map,
        _centre,
    ) = fields[0]
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
    eligible = (
        selector
        & np.isfinite(coherence)
        & (coherence >= MIN_COHERENCE)
    )
    denominator = int(np.count_nonzero(eligible))
    records: list[dict[str, float | int]] = []
    for sigma in SCALES:
        bend, resultant = fiber_bend_field(
            phi,
            coherence,
            selector,
            sigma,
        )
        finite_resultant = np.isfinite(resultant)
        cancelled = eligible & finite_resultant & (resultant < MIN_RESULTANT)
        unsupported = eligible & ~finite_resultant
        retained = (
            eligible
            & np.isfinite(bend)
            & finite_resultant
            & (resultant >= MIN_RESULTANT)
        )
        values = np.degrees(bend[retained])
        records.append(
            {
                "DetectorLabel": label,
                "Sigma": sigma,
                "EligiblePixels": denominator,
                "RetainedPixels": int(np.count_nonzero(retained)),
                "RetainedFraction": (
                    float(np.mean(retained[eligible])) if denominator else np.nan
                ),
                "CancelledPixels": int(np.count_nonzero(cancelled)),
                "CancelledFraction": (
                    float(np.mean(cancelled[eligible])) if denominator else np.nan
                ),
                "UnsupportedPixels": int(np.count_nonzero(unsupported)),
                "BendP95": (
                    float(np.percentile(values, 95.0)) if values.size else np.nan
                ),
                "RawPeak": float(np.max(values)) if values.size else np.nan,
                "MedianScaleResultant": (
                    float(np.median(resultant[eligible & finite_resultant]))
                    if np.any(eligible & finite_resultant)
                    else np.nan
                ),
            }
        )
    return records


def render_coverage_summary(coverage: pd.DataFrame):
    """Plot retained and Q-cancelled area fractions versus sigma."""
    figure, (coverage_axis, bend_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True,
    )
    colors = {24: "#0072B2", 36: "#D55E00"}
    for label in (24, 36):
        rows = coverage.loc[coverage["DetectorLabel"] == label]
        coverage_axis.plot(
            rows["Sigma"],
            100.0 * rows["RetainedFraction"],
            marker="o",
            linewidth=2.0,
            color=colors[label],
            label=f"Label {label}: retained",
        )
        coverage_axis.plot(
            rows["Sigma"],
            100.0 * rows["CancelledFraction"],
            marker="x",
            linestyle="--",
            linewidth=1.5,
            color=colors[label],
            label=f"Label {label}: Q cancellation",
        )
        bend_axis.plot(
            rows["Sigma"],
            rows["BendP95"],
            marker="o",
            linewidth=2.0,
            color=colors[label],
            label=f"Label {label}",
        )
        for row in rows.itertuples():
            bend_axis.annotate(
                f"{row.BendP95:.2f}",
                (row.Sigma, row.BendP95),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )
    coverage_axis.set_xlabel("Q-field sigma (px)")
    coverage_axis.set_ylabel("Fraction of eligible detected structure (%)")
    coverage_axis.set_ylim(0.0, 105.0)
    coverage_axis.set_title("Why colored sections disappear")
    coverage_axis.grid(alpha=0.25)
    coverage_axis.legend(frameon=False, fontsize=8)
    bend_axis.set_xlabel("Q-field sigma (px)")
    bend_axis.set_ylabel("Bend 95th percentile (deg/px)")
    bend_axis.set_title("Robust bend remaining at each scale")
    bend_axis.grid(alpha=0.25)
    bend_axis.legend(frameon=False)
    output = OUTPUT_DIR / "twok_sigma_16_32_48_coverage.png"
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return output


def render_extended_scale_overlays() -> None:
    """Render σ=16/32/48 overlays and quantify disappearing support."""
    detected, _old_segmented = load_twok_detection()
    props = {
        int(prop.label): prop
        for prop in regionprops(np.asarray(detected.objmap[:]))
    }
    all_coverage: list[dict[str, float | int]] = []
    orientation_module._BEND_SCALE_PRESETS["extended"] = SCALES
    for label, grid_section in CANDIDATES:
        centroid = tuple(float(value) for value in props[label].centroid)
        section = isolated_global_crop(detected, label, centroid)
        operation = MeasureOrientationZones(quiver_block=24)
        operation.measure(section)
        records = scale_coverage(section, operation, label)
        all_coverage.extend(records)
        figure = operation.fiber_bend_overlay(
            section,
            base_layer="detect_mat",
            scale_set="extended",
        )
        by_scale = {float(row["Sigma"]): row for row in records}
        for annotation, sigma in zip(figure.layout.annotations[:3], SCALES):
            row = by_scale[sigma]
            annotation.text += (
                f"<br><sup>retained {100.0 * float(row['RetainedFraction']):.1f}% · "
                f"Q-cancelled {100.0 * float(row['CancelledFraction']):.1f}%</sup>"
            )
        figure.update_layout(
            title=f"Label {label}, grid {grid_section}: fiber bend at σ=16/32/48 px"
        )
        output = OUTPUT_DIR / f"twok_label_{label}_bend_sigma_16_32_48.png"
        figure.write_image(str(output), width=1800, height=900, scale=1)
        print(output)
    coverage = pd.DataFrame.from_records(all_coverage)
    csv_output = OUTPUT_DIR / "twok_sigma_16_32_48_coverage.csv"
    coverage.to_csv(csv_output, index=False)
    summary_output = render_coverage_summary(coverage)
    print(coverage.to_string(index=False))
    print(csv_output)
    print(summary_output)


if __name__ == "__main__":
    render_extended_scale_overlays()
