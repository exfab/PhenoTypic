from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import convolve

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from render_ced_point_crossing_comparison import (  # noqa: E402
    analysis_rows,
    apply_ced_preserving_object_map,
    collect_crossing_analysis,
    render_ced_overlay_comparison,
    render_literal_crossing_before_after,
    render_literal_crossing_outward_metric,
    render_population_trend_comparison,
)
from render_matched_ring_comparison import label_centroid  # noqa: E402
from render_point_matched_ring_orientation import (  # noqa: E402
    load_point_matching_detection,
)
from render_twok_reconnected_orientation import (  # noqa: E402
    isolated_global_crop,
    report,
)


ANALYSIS_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = ANALYSIS_DIR / "artifacts"
COLONIES = (("R3C4", 24), ("R4C6", 36))


def ced_parameter_grid() -> list[dict[str, float | int | str]]:
    """Return a coarse CED sweep plus default and first-run controls."""
    configurations: dict[tuple[float, float, int, float], dict] = {}
    for sigma, rho_ratio, num_iter, threshold in itertools.product(
        (0.75, 1.5, 2.5),
        (1.0, 2.0),
        (15, 30),
        (80.0, 95.0),
    ):
        rho = sigma * rho_ratio
        key = (sigma, rho, num_iter, threshold)
        configurations[key] = {
            "sigma": sigma,
            "rho": rho,
            "num_iter": num_iter,
            "C": threshold,
        }
    for sigma, rho, num_iter, threshold in (
        (1.5, 1.5, 20, 99.0),
        (1.5, 3.0, 30, 90.0),
    ):
        configurations[(sigma, rho, num_iter, threshold)] = {
            "sigma": sigma,
            "rho": rho,
            "num_iter": num_iter,
            "C": threshold,
        }
    rows: list[dict[str, float | int | str]] = []
    for index, parameters in enumerate(configurations.values(), start=1):
        row = dict(parameters)
        row.update(
            {
                "ConfigID": f"CED{index:02d}",
                "dt": 0.1,
                "alpha": 0.001,
            }
        )
        rows.append(row)
    return rows


def skeleton_angle_roughness(analysis) -> tuple[float, float, int]:
    """Return axial roughness along non-junction skeleton interiors.

    Branch endpoints and junction pixels are excluded because their large
    orientation changes are geometric, not the dotted-intensity failure mode
    targeted by CED.
    """
    skeleton = analysis.reliable_skeleton
    neighbor_count = convolve(
        skeleton.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    ) - skeleton.astype(np.uint8)
    branch_interior = skeleton & (neighbor_count == 2)
    axis = analysis.fiber_axis
    differences: list[np.ndarray] = []
    for row_offset, col_offset in ((0, 1), (1, -1), (1, 0), (1, 1)):
        if col_offset < 0:
            left_cols = slice(1, None)
            right_cols = slice(None, -1)
        elif col_offset > 0:
            left_cols = slice(None, -1)
            right_cols = slice(1, None)
        else:
            left_cols = slice(None)
            right_cols = slice(None)
        if row_offset:
            upper_rows = slice(None, -1)
            lower_rows = slice(1, None)
        else:
            upper_rows = slice(None)
            lower_rows = slice(None)
        first_selector = branch_interior[upper_rows, left_cols]
        second_selector = branch_interior[lower_rows, right_cols]
        valid = first_selector & second_selector
        if not valid.any():
            continue
        difference = (
            axis[lower_rows, right_cols][valid]
            - axis[upper_rows, left_cols][valid]
        )
        axial = 0.5 * np.arctan2(
            np.sin(2.0 * difference),
            np.cos(2.0 * difference),
        )
        differences.append(np.abs(np.degrees(axial)))
    if not differences:
        return np.nan, np.nan, 0
    combined = np.concatenate(differences)
    return (
        float(np.median(combined)),
        float(np.percentile(combined, 90.0)),
        int(combined.size),
    )


def normalized_source_rmse(control, candidate) -> float:
    """Return object-region RMSE normalized by the control robust range."""
    control_base = np.asarray(control.profiles["base"], dtype=float)
    candidate_base = np.asarray(candidate.profiles["base"], dtype=float)
    selector = (
        control.profiles["obj_mask"]
        & np.isfinite(control.profiles["dist_map"])
        & (control.profiles["dist_map"] >= control.profiles["inner_radius"])
    )
    values = control_base[selector]
    low, high = np.percentile(values, (1.0, 99.0))
    scale = max(float(high - low), np.finfo(float).eps)
    rmse = float(
        np.sqrt(np.mean(np.square(candidate_base[selector] - values)))
    )
    return rmse / scale


def metric_row(colony: str, config: dict, control, candidate) -> dict:
    """Return one colony/configuration sweep record."""
    _ring_rows, summary = analysis_rows(colony, candidate)
    median_roughness, p90_roughness, neighbor_pairs = (
        skeleton_angle_roughness(candidate)
    )
    return {
        "Colony": colony,
        **config,
        "Crossings": summary["Crossings"],
        "ReliableSkeletonPixels": summary["ReliableSkeletonPixels"],
        "MedianCrossingCoherence": summary["MedianCrossingCoherence"],
        "MedianCrossingResultant": summary["MedianCrossingResultant"],
        "NeighborAngleMedianDeg": median_roughness,
        "NeighborAngleP90Deg": p90_roughness,
        "NeighborPairs": neighbor_pairs,
        "NormalizedSourceRMSE": normalized_source_rmse(control, candidate),
        "ManyToOneSupportedPoints": summary["ManyToOneSupportedPoints"],
        "ManyToOneRawPeakDeg": summary["ManyToOneRawPeakDeg"],
        "PopulationSupportedRings": summary["PopulationSupportedRings"],
        "PopulationRawPeakDeg": summary["PopulationRawPeakDeg"],
    }


def add_relative_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add control-relative metrics and aggregate configuration diagnostics."""
    output = frame.copy()
    controls = output[output["ConfigID"] == "CONTROL"].set_index("Colony")
    for index, row in output.iterrows():
        control = controls.loc[row["Colony"]]
        output.loc[index, "CrossingRatio"] = (
            row["Crossings"] / control["Crossings"]
        )
        output.loc[index, "ReliablePixelRatio"] = (
            row["ReliableSkeletonPixels"] / control["ReliableSkeletonPixels"]
        )
        output.loc[index, "CoherenceGainFraction"] = (
            row["MedianCrossingCoherence"]
            / control["MedianCrossingCoherence"]
            - 1.0
        )
        output.loc[index, "P90RoughnessReductionFraction"] = 1.0 - (
            row["NeighborAngleP90Deg"] / control["NeighborAngleP90Deg"]
        )
    return output


def aggregate_configurations(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate both colonies and mark the multi-objective Pareto frontier."""
    ced = frame[frame["ConfigID"] != "CONTROL"].copy()
    group_columns = [
        "ConfigID",
        "sigma",
        "rho",
        "num_iter",
        "C",
        "dt",
        "alpha",
    ]
    aggregate = (
        ced.groupby(group_columns, as_index=False)
        .agg(
            MeanCoherenceGain=("CoherenceGainFraction", "mean"),
            MeanRoughnessReduction=("P90RoughnessReductionFraction", "mean"),
            WorstCrossingDeviation=(
                "CrossingRatio",
                lambda values: float(np.max(np.abs(values - 1.0))),
            ),
            WorstReliablePixelDeviation=(
                "ReliablePixelRatio",
                lambda values: float(np.max(np.abs(values - 1.0))),
            ),
            MeanNormalizedRMSE=("NormalizedSourceRMSE", "mean"),
        )
        .reset_index(drop=True)
    )
    aggregate["Pareto"] = True
    objectives = aggregate[
        [
            "MeanCoherenceGain",
            "MeanRoughnessReduction",
            "WorstCrossingDeviation",
            "MeanNormalizedRMSE",
        ]
    ].to_numpy()
    for index, candidate in enumerate(objectives):
        maximized = objectives[:, :2] >= candidate[:2]
        minimized = objectives[:, 2:] <= candidate[2:]
        at_least_as_good = np.all(maximized, axis=1) & np.all(minimized, axis=1)
        strictly_better = np.any(objectives[:, :2] > candidate[:2], axis=1) | np.any(
            objectives[:, 2:] < candidate[2:],
            axis=1,
        )
        if np.any(at_least_as_good & strictly_better):
            aggregate.loc[index, "Pareto"] = False
    aggregate["DiagnosticScore"] = (
        aggregate["MeanCoherenceGain"]
        + aggregate["MeanRoughnessReduction"]
        - 2.0 * aggregate["WorstCrossingDeviation"]
        - aggregate["MeanNormalizedRMSE"]
    )
    return aggregate.sort_values("DiagnosticScore", ascending=False)


def render_sweep(frame: pd.DataFrame, aggregate: pd.DataFrame) -> Path:
    """Render colony-specific and aggregate CED trade-offs."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ced = frame[frame["ConfigID"] != "CONTROL"]
    for axis, colony in zip(axes[0], ("R3C4", "R4C6")):
        subset = ced[ced["Colony"] == colony]
        scatter = axis.scatter(
            subset["NeighborAngleP90Deg"],
            subset["MedianCrossingCoherence"],
            c=subset["sigma"],
            s=20.0 + subset["num_iter"],
            cmap="viridis",
            edgecolors="black",
            linewidths=0.35,
            alpha=0.85,
        )
        control = frame[
            (frame["Colony"] == colony) & (frame["ConfigID"] == "CONTROL")
        ].iloc[0]
        axis.scatter(
            [control["NeighborAngleP90Deg"]],
            [control["MedianCrossingCoherence"]],
            marker="*",
            s=180,
            color="#d73027",
            edgecolors="black",
            label="no CED",
        )
        axis.set_title(f"{colony}: local reliability trade-off")
        axis.set_xlabel(
            "P90 non-junction skeleton axial difference (degrees; lower better)"
        )
        axis.set_ylabel("Median crossing coherence (higher better)")
        axis.grid(alpha=0.2)
        axis.legend()
    colorbar = figure.colorbar(scatter, ax=axes[0, :], fraction=0.035, pad=0.02)
    colorbar.set_label("CED sigma (pixels); marker size increases with iterations")

    aggregate_axis = axes[1, 0]
    aggregate_scatter = aggregate_axis.scatter(
        100.0 * aggregate["MeanRoughnessReduction"],
        100.0 * aggregate["MeanCoherenceGain"],
        c=aggregate["MeanNormalizedRMSE"],
        s=55,
        cmap="magma_r",
        edgecolors=np.where(aggregate["Pareto"], "#00a6d6", "black"),
        linewidths=np.where(aggregate["Pareto"], 2.0, 0.4),
    )
    for _, row in aggregate[aggregate["Pareto"]].iterrows():
        aggregate_axis.annotate(
            row["ConfigID"],
            (100.0 * row["MeanRoughnessReduction"], 100.0 * row["MeanCoherenceGain"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    aggregate_axis.axvline(0.0, color="#777777", linewidth=0.8)
    aggregate_axis.axhline(0.0, color="#777777", linewidth=0.8)
    aggregate_axis.set_xlabel("Mean P90 roughness reduction across colonies (%)")
    aggregate_axis.set_ylabel("Mean coherence gain across colonies (%)")
    aggregate_axis.set_title("Aggregate gains; cyan outline = Pareto frontier")
    aggregate_axis.grid(alpha=0.2)
    aggregate_colorbar = figure.colorbar(
        aggregate_scatter,
        ax=aggregate_axis,
        fraction=0.05,
        pad=0.03,
    )
    aggregate_colorbar.set_label("Mean normalized source RMSE")

    fidelity_axis = axes[1, 1]
    fidelity_scatter = fidelity_axis.scatter(
        100.0 * aggregate["WorstCrossingDeviation"],
        100.0 * aggregate["MeanNormalizedRMSE"],
        c=aggregate["DiagnosticScore"],
        s=55,
        cmap="coolwarm",
        edgecolors=np.where(aggregate["Pareto"], "#00a6d6", "black"),
        linewidths=np.where(aggregate["Pareto"], 2.0, 0.4),
    )
    for _, row in aggregate.head(5).iterrows():
        fidelity_axis.annotate(
            row["ConfigID"],
            (
                100.0 * row["WorstCrossingDeviation"],
                100.0 * row["MeanNormalizedRMSE"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    fidelity_axis.set_xlabel("Worst crossing-count deviation (%)")
    fidelity_axis.set_ylabel("Mean normalized source RMSE (%)")
    fidelity_axis.set_title("Geometry and intensity fidelity")
    fidelity_axis.grid(alpha=0.2)
    fidelity_colorbar = figure.colorbar(
        fidelity_scatter,
        ax=fidelity_axis,
        fraction=0.05,
        pad=0.03,
    )
    fidelity_colorbar.set_label("Diagnostic score (not a biological objective)")

    figure.suptitle(
        "CED parameter sweep for literal skeleton-ring orientation collection\n"
        "Selection proxies reward coherence and local smoothness while auditing geometry",
        fontsize=14,
    )
    output = OUTPUT_DIR / "twok_ced_literal_crossing_parameter_sweep.png"
    figure.savefig(output, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output


def sweep_ced_literal_crossings() -> None:
    """Evaluate a shared CED grid on both real colonies."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detected = load_point_matching_detection()
    configurations = ced_parameter_grid()
    rows: list[dict] = []
    source_sections: dict[str, object] = {}
    controls: dict[str, object] = {}
    for colony, label in COLONIES:
        section = isolated_global_crop(
            detected,
            label,
            label_centroid(detected, label),
        )
        control = collect_crossing_analysis(section, "Original")
        source_sections[colony] = section
        controls[colony] = control
        control_config = {
            "ConfigID": "CONTROL",
            "sigma": np.nan,
            "rho": np.nan,
            "num_iter": 0,
            "C": np.nan,
            "dt": np.nan,
            "alpha": np.nan,
        }
        rows.append(metric_row(colony, control_config, control, control))
        for index, config in enumerate(configurations, start=1):
            parameters = {
                key: config[key]
                for key in ("sigma", "rho", "num_iter", "C", "dt", "alpha")
            }
            ced_section = apply_ced_preserving_object_map(section, parameters)
            candidate = collect_crossing_analysis(
                ced_section,
                str(config["ConfigID"]),
                geometry_reference=control,
            )
            rows.append(metric_row(colony, config, control, candidate))
            if index % 5 == 0 or index == len(configurations):
                report(f"{colony}: completed {index}/{len(configurations)} CED settings")

    raw_frame = pd.DataFrame(rows)
    frame = add_relative_metrics(raw_frame)
    aggregate = aggregate_configurations(frame)
    conservative = aggregate[
        (aggregate["MeanRoughnessReduction"] >= -0.01)
        & (aggregate["WorstCrossingDeviation"] <= 0.02)
    ].sort_values("MeanCoherenceGain", ascending=False)
    if conservative.empty:
        raise RuntimeError("No CED configuration passed conservative sweep guards")
    selected = conservative.iloc[0]
    detail_output = OUTPUT_DIR / "twok_ced_literal_crossing_parameter_sweep.csv"
    aggregate_output = (
        OUTPUT_DIR / "twok_ced_literal_crossing_parameter_sweep_aggregate.csv"
    )
    frame.to_csv(detail_output, index=False)
    aggregate.to_csv(aggregate_output, index=False)
    figure_output = render_sweep(frame, aggregate)
    selected_parameters = {
        "sigma": float(selected["sigma"]),
        "rho": float(selected["rho"]),
        "num_iter": int(selected["num_iter"]),
        "C": float(selected["C"]),
        "dt": float(selected["dt"]),
        "alpha": float(selected["alpha"]),
    }
    for colony, _label in COLONIES:
        selected_section = apply_ced_preserving_object_map(
            source_sections[colony],
            selected_parameters,
        )
        selected_analysis = collect_crossing_analysis(
            selected_section,
            f"Sweep-selected {selected['ConfigID']}",
            geometry_reference=controls[colony],
        )
        overlay = render_ced_overlay_comparison(
            colony,
            controls[colony],
            selected_analysis,
            output_tag=str(selected["ConfigID"]),
        )
        trend = render_population_trend_comparison(
            colony,
            controls[colony],
            selected_analysis,
            output_tag=str(selected["ConfigID"]),
        )
        literal_metric = render_literal_crossing_outward_metric(
            colony,
            selected_analysis,
            output_tag=str(selected["ConfigID"]),
        )
        before_after = render_literal_crossing_before_after(
            colony,
            controls[colony],
            selected_analysis,
            output_tag=str(selected["ConfigID"]),
        )
        report(str(overlay))
        report(str(trend))
        report(str(literal_metric))
        report(str(before_after))
    report(str(detail_output))
    report(str(aggregate_output))
    report(str(figure_output))
    print("Conservative sweep selection:", flush=True)
    print(selected.to_string(), flush=True)
    print(aggregate.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    sweep_ced_literal_crossings()
