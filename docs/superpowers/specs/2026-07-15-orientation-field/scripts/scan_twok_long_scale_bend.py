from __future__ import annotations

from pathlib import Path

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
from phenotypic.measure._measure_orientation_zones import zone_selector
from phenotypic.util._nematic_bend import fiber_bend_field


SIGMAS = (8.0, 16.0)
MIN_VALID_PIXELS = 100
MIN_COHERENCE = 0.15
MIN_RESULTANT = 0.15


def bend_values_at_scale(
    phi: np.ndarray,
    coherence: np.ndarray,
    selector: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Return reliable fiber-bend values in degrees per pixel."""
    bend, resultant = fiber_bend_field(
        phi,
        coherence,
        selector,
        sigma,
    )
    valid = (
        selector
        & np.isfinite(bend)
        & np.isfinite(coherence)
        & np.isfinite(resultant)
        & (coherence >= MIN_COHERENCE)
        & (resultant >= MIN_RESULTANT)
    )
    return np.degrees(bend[valid])


def scan_broad_scale_bend(detected) -> pd.DataFrame:
    """Calculate density-neutral upper-tail bend summaries per colony."""
    operation = MeasureOrientationZones(quiver_block=24)
    props, label_to_section = operation._prep(detected)
    records: list[dict[str, float | int | bool]] = []
    fields = operation._iter_object_fields(detected, props, label_to_section)
    for prop, seg, obj_mask, phi, coherence, _gradient, dist_map, _centre in fields:
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
        row: dict[str, float | int | bool] = {
            "DetectorLabel": int(prop.label),
            "GridSection": int(label_to_section.get(prop.label, prop.label - 1)),
            "ZonesComputed": bool(seg.zones_computed),
            "CoreEndRadius": inner_radius,
            "OuterRadius": outer_radius,
        }
        valid_zone_geometry = (
            bool(seg.zones_computed)
            and inner_radius > 0.0
            and outer_radius > inner_radius
        )
        for sigma in SIGMAS:
            values = (
                bend_values_at_scale(phi, coherence, selector, sigma)
                if valid_zone_geometry
                else np.empty(0, dtype=np.float64)
            )
            key = f"Sigma{int(sigma)}"
            row[f"{key}_ValidPixels"] = int(values.size)
            if values.size:
                row[f"{key}_Median"] = float(np.median(values))
                row[f"{key}_P90"] = float(np.percentile(values, 90.0))
                row[f"{key}_P95"] = float(np.percentile(values, 95.0))
                row[f"{key}_RawPeak"] = float(np.max(values))
                row[f"{key}_FractionAbove3"] = float(np.mean(values >= 3.0))
            else:
                row[f"{key}_Median"] = np.nan
                row[f"{key}_P90"] = np.nan
                row[f"{key}_P95"] = np.nan
                row[f"{key}_RawPeak"] = np.nan
                row[f"{key}_FractionAbove3"] = np.nan
        p95_mid = float(row["Sigma8_P95"])
        p95_broad = float(row["Sigma16_P95"])
        row["P95PersistenceRatio16to8"] = (
            p95_broad / p95_mid
            if np.isfinite(p95_mid) and p95_mid > 0.0
            else np.nan
        )
        records.append(row)
    return pd.DataFrame.from_records(records)


def render_ranking(scan: pd.DataFrame) -> Path:
    """Render the ten strongest robust broad-scale bend candidates."""
    eligible = scan.loc[
        scan["Sigma16_ValidPixels"] >= MIN_VALID_PIXELS
    ].nlargest(10, "Sigma16_P95")
    labels = [
        f"label {int(row.DetectorLabel)} · grid {int(row.GridSection)}"
        for row in eligible.itertuples()
    ]
    positions = np.arange(len(eligible))
    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    bars = axis.barh(
        positions,
        eligible["Sigma16_P95"],
        color="#0072B2",
        alpha=0.88,
    )
    axis.invert_yaxis()
    axis.set_yticks(positions, labels)
    axis.set_xlabel("σ=16 bend, 95th percentile (deg/px)")
    axis.set_title("Strongest robust long-scale bend candidates")
    axis.grid(axis="x", alpha=0.25)
    for bar, row in zip(bars, eligible.itertuples()):
        axis.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            (
                f"  p95={row.Sigma16_P95:.2f}, "
                f"peak={row.Sigma16_RawPeak:.2f}, "
                f"n={int(row.Sigma16_ValidPixels)}"
            ),
            va="center",
            fontsize=9,
        )
    output = OUTPUT_DIR / "twok_sigma16_bend_ranking.png"
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return output


def render_top_candidates(detected, scan: pd.DataFrame) -> list[Path]:
    """Render balanced multiscale overlays for the top three candidates."""
    eligible = scan.loc[
        scan["Sigma16_ValidPixels"] >= MIN_VALID_PIXELS
    ].nlargest(3, "Sigma16_P95")
    props = {
        int(prop.label): prop
        for prop in regionprops(np.asarray(detected.objmap[:]))
    }
    outputs: list[Path] = []
    for rank, row in enumerate(eligible.itertuples(), start=1):
        label = int(row.DetectorLabel)
        centroid = tuple(float(value) for value in props[label].centroid)
        section = isolated_global_crop(detected, label, centroid)
        operation = MeasureOrientationZones(quiver_block=24)
        operation.measure(section)
        figure = operation.fiber_bend_overlay(
            section,
            base_layer="detect_mat",
            scale_set="balanced",
        )
        figure.update_layout(
            title=(
                f"Rank {rank}: label {label}, grid {int(row.GridSection)} · "
                f"σ=16 p95 {row.Sigma16_P95:.3f} deg/px"
            )
        )
        output = OUTPUT_DIR / (
            f"twok_long_bend_rank_{rank}_label_{label}.png"
        )
        figure.write_image(str(output), width=1800, height=900, scale=1)
        outputs.append(output)
    return outputs


def scan_and_render_long_bend_candidates() -> None:
    """Scan all colonies, save rankings, and render the top candidates."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detected, _old_segmented = load_twok_detection()
    scan = scan_broad_scale_bend(detected)
    csv_output = OUTPUT_DIR / "twok_sigma16_bend_scan.csv"
    scan.to_csv(csv_output, index=False)
    ranking_output = render_ranking(scan)
    candidate_outputs = render_top_candidates(detected, scan)
    columns = [
        "DetectorLabel",
        "GridSection",
        "Sigma16_ValidPixels",
        "Sigma16_P95",
        "Sigma16_RawPeak",
        "Sigma16_FractionAbove3",
        "P95PersistenceRatio16to8",
    ]
    print(scan.nlargest(10, "Sigma16_P95")[columns].to_string(index=False))
    print(csv_output)
    print(ranking_output)
    for output in candidate_outputs:
        print(output)


if __name__ == "__main__":
    scan_and_render_long_bend_candidates()
