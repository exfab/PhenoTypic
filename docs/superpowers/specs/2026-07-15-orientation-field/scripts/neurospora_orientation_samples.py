from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
from skimage.exposure import adjust_gamma, rescale_intensity

import phenotypic as pht
from phenotypic.correction import ImageCropper
from phenotypic.detect import HysteresisDetector, ManualGridPointDetector
from phenotypic.enhance import (
    ContrastStretching,
    FlattenIllumination,
    FocusEdgePhase,
    SubtractGaussian,
)
from phenotypic.measure import MeasureOrientationZones
from phenotypic.measure._measure_orientation_zones import (
    _resultant_direction,
    aggregate_orientation,
    zone_selector,
)
from phenotypic.measure._zone_segmentation import (
    compute_zone_segmentation,
    distance_from_point,
)
from phenotypic.util._orientation_field import orientation_field


IMAGE_PATH = Path(
    "/Volumes/T9/exfab/UCR-010-I-D_Neurospora/data/"
    "denoised_media_subsets_FrameIdx10-12/xylan/"
    "d000273_300_001_2025-12-12_02-00-49_rgb.tiff"
)
OUTPUT_DIR = Path(
    "/Users/alex/.codex/visualizations/2026/07/15/"
    "019f6340-b68c-7a81-b738-983ed6ea1a27/orientation-real-image"
)
CACHE_DIR = Path("/private/tmp/neurospora_orientation_cache")

K = 5
COORD1 = (400, 550)
COORD2 = (800, 950)
WIDTH = 100
FLATTEN_SIGMA = 300.0
GAUSS_SIGMA = 300.0
TARGET_RC = np.asarray(COORD1, dtype=float)
BLOCK = 24

ZONE_COLORS = {
    "Dense": "#CC79A7",
    "Sparse": "#D55E00",
}


def report(message: str) -> None:
    """Print a timestamped progress message."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_response(response: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Rescale a response map to [0, 1] using the notebook's robust range."""
    upper = float(np.percentile(response, percentile))
    return rescale_intensity(
        np.asarray(response, dtype=float),
        in_range=(0.0, upper),
        out_range=(0.0, 1.0),
    )


def load_notebook_base() -> pht.GridImage:
    """Load and preprocess the notebook's source image."""
    image = pht.GridImage.imread(IMAGE_PATH, nrows=6, ncols=10)
    ImageCropper(left=650, right=650, top=600, bottom=600).apply(
        image,
        inplace=True,
    )
    image.set_image(adjust_gamma(image.rgb[:], gamma=2.0, gain=3.0))
    return image


def reproduce_notebook_segmentation() -> tuple[pht.GridImage, dict[str, float]]:
    """Reproduce the notebook pipeline, caching its expensive array outputs."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    composite_path = CACHE_DIR / "composite.npy"
    objmap_path = CACHE_DIR / "objmap.npy"
    base = load_notebook_base()

    if composite_path.exists() and objmap_path.exists():
        report("Loading cached notebook composite and objmap")
        composite = np.load(composite_path, mmap_mode="r")
        objmap = np.load(objmap_path, mmap_mode="r")
        segmented = base.copy()
        segmented.detect_mat[:] = np.asarray(composite, dtype=np.float32)
        segmented.objmap[:] = np.asarray(objmap, dtype=segmented.objmap[:].dtype)
        return segmented, {
            "branch_p99": float("nan"),
            "center_p99": float("nan"),
        }

    report("Flattening illumination")
    flat = base.copy()
    FlattenIllumination(sigma=FLATTEN_SIGMA).apply(flat, inplace=True)

    report("Computing branch phase-congruency map")
    branch_image = flat.copy()
    ContrastStretching(lower_percentile=70, upper_percentile=99).apply(
        branch_image,
        inplace=True,
    )
    start = time.perf_counter()
    FocusEdgePhase(n_orient=8, k=K, min_wavelength=5.0).apply(
        branch_image,
        inplace=True,
    )
    branch = normalize_response(branch_image.detect_mat[:])
    report(f"Phase-congruency map finished in {time.perf_counter() - start:.1f}s")

    report("Computing grid-gated center-fill map")
    center_image = flat.copy()
    SubtractGaussian(sigma=GAUSS_SIGMA, n_iter=2).apply(
        center_image,
        inplace=True,
    )
    grid = base.copy()
    ManualGridPointDetector(
        coord1=COORD1,
        coord2=COORD2,
        shape="disk",
        width=WIDTH,
    ).apply(grid, inplace=True)
    grid_mask = grid.objmask[:] > 0
    raw_center = np.asarray(center_image.detect_mat[:], dtype=float)
    center_scale = float(np.percentile(raw_center[grid_mask], 99))
    center = np.clip(raw_center / center_scale, 0.0, 1.0) * grid_mask
    composite = np.maximum(branch, center).astype(np.float32)

    report("Running notebook hysteresis segmentation")
    segmented = base.copy()
    segmented.detect_mat[:] = composite
    HysteresisDetector(
        low="triangle",
        high="otsu",
        ignore_zeros=False,
        ignore_borders=False,
    ).apply(segmented, inplace=True)

    np.save(composite_path, composite)
    np.save(objmap_path, segmented.objmap[:])
    return segmented, {
        "branch_p99": float(np.percentile(branch, 99)),
        "center_p99": float(np.percentile(center[grid_mask], 99)),
    }


def select_real_colony_section(
    segmented: pht.GridImage,
) -> tuple[object, int, int]:
    """Select the mapped colony nearest the notebook's one-colony coordinate."""
    operation = MeasureOrientationZones()
    props, label_to_section = operation._prep(segmented)
    mapped = [prop for prop in props if prop.label in label_to_section]
    if not mapped:
        raise RuntimeError("No grid-mapped segmented objects were found")
    target_prop = min(
        mapped,
        key=lambda prop: float(
            np.linalg.norm(np.asarray(prop.centroid) - TARGET_RC)
        ),
    )
    section_index = int(label_to_section[target_prop.label])
    section = segmented.grid[section_index]
    # ``GridImage.grid[idx]`` crops the section but retains incidental fragment
    # labels. Isolate the mapped colony so this figure tests the intended grid
    # object rather than unrelated one-pixel-wide components in the same tile.
    section_objmap = section.objmap[:]
    section.objmap[:] = np.where(
        section_objmap == target_prop.label,
        target_prop.label,
        0,
    )
    return section, int(target_prop.label), section_index


def block_summaries(
    phi: np.ndarray,
    coherence: np.ndarray,
    selector: np.ndarray,
    block: int,
) -> list[dict[str, float]]:
    """Aggregate an axial orientation and confidence within image blocks."""
    records: list[dict[str, float]] = []
    height, width = phi.shape
    for row0 in range(0, height, block):
        for col0 in range(0, width, block):
            row1 = min(row0 + block, height)
            col1 = min(col0 + block, width)
            local_selector = selector[row0:row1, col0:col1]
            if int(local_selector.sum()) < 8:
                continue
            local_coherence = coherence[row0:row1, col0:col1][local_selector]
            weight_sum = float(local_coherence.sum())
            if weight_sum <= 1e-9:
                continue
            local_phi = phi[row0:row1, col0:col1][local_selector]
            mean_cos = float((local_coherence * np.cos(2.0 * local_phi)).sum()) / weight_sum
            mean_sin = float((local_coherence * np.sin(2.0 * local_phi)).sum()) / weight_sum
            concentration = float(np.hypot(mean_cos, mean_sin))
            gradient_normal = 0.5 * np.arctan2(mean_sin, mean_cos)
            fiber_axis = gradient_normal + np.pi / 2.0
            records.append(
                {
                    "row": 0.5 * (row0 + row1 - 1),
                    "col": 0.5 * (col0 + col1 - 1),
                    "theta": float(fiber_axis),
                    "R": concentration,
                    "C": float(local_coherence.mean()),
                }
            )
    return records


def draw_zone_boundaries(axis, centre: tuple[float, float], radii: dict[str, float]) -> None:
    """Draw the exact radial boundaries used by the operation."""
    row, col = centre
    styles = (
        ("symmetric", "#785EF0", "-", 2.0),
        ("core_end", "#DC267F", ":", 1.5),
        ("dense_end", "#56B4E9", "--", 1.7),
        ("sparse_end", "#E69F00", "--", 1.7),
    )
    for name, color, linestyle, linewidth in styles:
        radius = radii.get(name, np.nan)
        if np.isfinite(radius) and radius > 0:
            axis.add_patch(
                Circle(
                    (col, row),
                    float(radius),
                    fill=False,
                    edgecolor=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                )
            )


def add_headless_axes(
    axis,
    records: list[dict[str, float]],
    block: int,
    *,
    fixed_color: str | None = None,
) -> None:
    """Draw centered axial segments with block R as length and C as opacity."""
    if not records:
        return
    colormap = plt.get_cmap("viridis")
    normalizer = Normalize(vmin=0.0, vmax=1.0)
    segments = []
    colors = []
    for record in records:
        half_length = 0.46 * block * max(record["R"], 0.08)
        dx = half_length * np.cos(record["theta"])
        dy = half_length * np.sin(record["theta"])
        segments.append(
            [
                (record["col"] - dx, record["row"] - dy),
                (record["col"] + dx, record["row"] + dy),
            ]
        )
        base_color = (
            matplotlib.colors.to_rgba(fixed_color)
            if fixed_color is not None
            else colormap(normalizer(record["C"]))
        )
        colors.append((*base_color[:3], 0.18 + 0.82 * record["C"]))
    axis.add_collection(LineCollection(segments, colors=colors, linewidths=2.2))


def add_double_headed_axes(
    axis,
    records: list[dict[str, float]],
    block: int,
    color: str,
) -> None:
    """Draw centered double-headed arrows for an explicitly axial glyph."""
    for record in records:
        half_length = 0.43 * block * max(record["R"], 0.08)
        dx = half_length * np.cos(record["theta"])
        dy = half_length * np.sin(record["theta"])
        axis.add_patch(
            FancyArrowPatch(
                (record["col"] - dx, record["row"] - dy),
                (record["col"] + dx, record["row"] + dy),
                arrowstyle="<->",
                mutation_scale=5.0,
                linewidth=1.4,
                color=color,
                alpha=0.15 + 0.85 * record["C"],
            )
        )


def show_actual_layer(axis, tile: np.ndarray, title: str) -> None:
    """Render the exact array passed to ``orientation_field``."""
    lower, upper = np.percentile(tile, (1.0, 99.8))
    axis.imshow(tile, cmap="magma", vmin=float(lower), vmax=float(upper))
    axis.set_title(title, fontsize=12)
    axis.set_axis_off()


def render_block_overlay(
    tile: np.ndarray,
    overall_records: list[dict[str, float]],
    centre: tuple[float, float],
    radii: dict[str, float],
    output_path: Path,
) -> None:
    """Render the recommended blockwise regional-mean orientation overlay."""
    figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    show_actual_layer(
        axis,
        tile,
        "Candidate 1: blockwise mean fiber axes on the actual detect_mat",
    )
    add_headless_axes(axis, overall_records, BLOCK)
    draw_zone_boundaries(axis, centre, radii)
    scalar = matplotlib.cm.ScalarMappable(
        norm=Normalize(0.0, 1.0),
        cmap="viridis",
    )
    colorbar = figure.colorbar(scalar, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Mean block coherence C")
    axis.text(
        0.01,
        0.01,
        "Centered needles are axial (no forward direction). Length = block R; "
        "color and opacity = block C.",
        transform=axis.transAxes,
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 5},
    )
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def render_zone_arrows(
    tile: np.ndarray,
    zone_records: dict[str, list[dict[str, float]]],
    centre: tuple[float, float],
    radii: dict[str, float],
    output_path: Path,
) -> None:
    """Render double-headed block arrows colored by measured zone."""
    figure, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    show_actual_layer(
        axis,
        tile,
        "Candidate 2: double-headed regional axes, colored by measurement zone",
    )
    for zone in ("Dense", "Sparse"):
        add_double_headed_axes(
            axis,
            zone_records[zone],
            BLOCK,
            ZONE_COLORS[zone],
        )
    draw_zone_boundaries(axis, centre, radii)
    handles = [
        plt.Line2D([0], [0], color=ZONE_COLORS[zone], lw=2, label=f"{zone} blocks")
        for zone in ("Dense", "Sparse")
    ]
    axis.legend(handles=handles, loc="lower right", framealpha=0.85)
    axis.text(
        0.01,
        0.01,
        "Two arrowheads preserve the 180° ambiguity of orientation. "
        "Opacity = block C; length = block R.",
        transform=axis.transAxes,
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 5},
    )
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def render_calculation_triptych(
    tile: np.ndarray,
    overall_records: list[dict[str, float]],
    coherence: np.ndarray,
    grad_phi: np.ndarray,
    selectors: dict[str, np.ndarray],
    centre: tuple[float, float],
    radii: dict[str, float],
    metrics: dict[str, tuple[float, float, float]],
    output_path: Path,
) -> None:
    """Render source field, local coherence, and local turning side by side."""
    figure, axes = plt.subplots(1, 3, figsize=(19, 6.5), constrained_layout=True)
    show_actual_layer(axes[0], tile, "A. Actual detect_mat + local fiber axes")
    add_headless_axes(axes[0], overall_records, BLOCK, fixed_color="#56B4E9")
    draw_zone_boundaries(axes[0], centre, radii)

    coherence_map = np.where(selectors["Overall"], coherence, np.nan)
    coherence_image = axes[1].imshow(
        coherence_map,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("B. Local coherence C (orientation confidence)")
    axes[1].set_axis_off()
    figure.colorbar(coherence_image, ax=axes[1], fraction=0.045, pad=0.02)

    reliable = selectors["Overall"] & (coherence >= 0.15)
    turning_map = np.where(reliable, grad_phi, np.nan)
    finite_turning = turning_map[np.isfinite(turning_map)]
    turning_upper = (
        float(np.percentile(finite_turning, 99))
        if finite_turning.size
        else 1.0
    )
    turning_image = axes[2].imshow(
        turning_map,
        cmap="inferno",
        vmin=0.0,
        vmax=turning_upper,
    )
    axes[2].set_title("C. Local |grad phi|, faded where C < 0.15")
    axes[2].set_axis_off()
    turning_bar = figure.colorbar(
        turning_image,
        ax=axes[2],
        fraction=0.045,
        pad=0.02,
    )
    turning_bar.set_label("rad/px")

    summary = "   ".join(
        f"{zone}: R={values[0]:.3f}, turning={values[1]:.4f} rad/px, C={values[2]:.3f}"
        for zone, values in metrics.items()
    )
    figure.suptitle(
        "Candidate 3: calculation explainer\n"
        "R = |sum C exp(i2phi)| / sum C; turning = sum C|grad phi| / sum C\n"
        + summary,
        fontsize=12,
    )
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def render_axial_rose(
    tile: np.ndarray,
    overall_records: list[dict[str, float]],
    phi: np.ndarray,
    coherence: np.ndarray,
    selectors: dict[str, np.ndarray],
    grad_phi: np.ndarray,
    centre: tuple[float, float],
    radii: dict[str, float],
    output_path: Path,
) -> None:
    """Render a source overlay beside axial orientation roses by zone."""
    figure = plt.figure(figsize=(17, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1.45, 1.0, 1.0))
    image_axis = figure.add_subplot(grid[:, 0])
    show_actual_layer(image_axis, tile, "Candidate 4: field plus coherence-weighted axial roses")
    add_headless_axes(image_axis, overall_records, BLOCK, fixed_color="#56B4E9")
    draw_zone_boundaries(image_axis, centre, radii)

    zone_positions = {
        "Overall": grid[0, 1],
        "Dense": grid[0, 2],
        "Sparse": grid[1, 1],
    }
    colors = {"Overall": "#E69F00", **ZONE_COLORS}
    for zone, position in zone_positions.items():
        axis = figure.add_subplot(position, projection="polar")
        selector = selectors[zone]
        theta = np.mod(phi[selector] + np.pi / 2.0, np.pi)
        weights = coherence[selector]
        edges = np.linspace(0.0, np.pi, 19)
        counts, _ = np.histogram(theta, bins=edges, weights=weights)
        probability = counts / counts.sum() if counts.sum() > 0 else counts
        centers = 0.5 * (edges[:-1] + edges[1:])
        axial_centers = np.concatenate([centers, centers + np.pi])
        axial_probability = np.concatenate([probability, probability])
        axis.bar(
            axial_centers,
            axial_probability,
            width=(np.pi / 18.0) * 0.92,
            color=colors[zone],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.4,
        )
        concentration, turning, mean_coherence = aggregate_orientation(
            phi,
            coherence,
            grad_phi,
            selector,
        )
        direction = _resultant_direction(phi, coherence, selector) + np.pi / 2.0
        peak = float(axial_probability.max()) if axial_probability.size else 1.0
        resultant_radius = peak * concentration
        axis.plot(
            [direction, direction],
            [0.0, resultant_radius],
            color="#003660",
            lw=3.0,
        )
        axis.plot(
            [direction + np.pi, direction + np.pi],
            [0.0, resultant_radius],
            color="#003660",
            lw=3.0,
        )
        axis.set_theta_zero_location("E")
        axis.set_theta_direction(-1)
        axis.set_yticklabels([])
        axis.set_title(
            f"{zone}\nR={concentration:.3f}, T={turning:.4f}, C={mean_coherence:.3f}",
            fontsize=11,
        )

    explanation_axis = figure.add_subplot(grid[1, 2])
    explanation_axis.axis("off")
    explanation_axis.text(
        0.0,
        0.95,
        "How the rose explains R\n\n"
        "Each bar is coherence-weighted orientation frequency.\n"
        "The distribution is duplicated 180° because a fiber has no forward end.\n\n"
        "The navy diameter is the mean axis. Its relative length is R.\n"
        "A narrow rose gives R near 1; a broad or radial mix gives lower R.",
        va="top",
        fontsize=12,
        linespacing=1.35,
    )
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def render_interactive_block_overlay(
    tile: np.ndarray,
    records: list[dict[str, float]],
    centre: tuple[float, float],
    radii: dict[str, float],
    output_path: Path,
) -> None:
    """Render a zoomable Plotly block-axis overlay with per-block hover."""
    import plotly.graph_objects as go

    lower, upper = np.percentile(tile, (1.0, 99.8))
    figure = go.Figure(
        go.Heatmap(
            z=tile,
            colorscale="Magma",
            zmin=float(lower),
            zmax=float(upper),
            showscale=False,
            hoverinfo="skip",
            name="Actual detect_mat",
        )
    )
    coherence_bins = (
        ("Low C", 0.0, 0.4, "rgba(86,180,233,0.30)"),
        ("Medium C", 0.4, 0.7, "rgba(86,180,233,0.60)"),
        ("High C", 0.7, np.inf, "rgba(86,180,233,0.95)"),
    )
    for name, minimum, maximum, color in coherence_bins:
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        for record in records:
            if not minimum <= record["C"] < maximum:
                continue
            half_length = 0.46 * BLOCK * max(record["R"], 0.08)
            dx = half_length * np.cos(record["theta"])
            dy = half_length * np.sin(record["theta"])
            x_values.extend([record["col"] - dx, record["col"] + dx, None])
            y_values.extend([record["row"] - dy, record["row"] + dy, None])
        if x_values:
            figure.add_trace(
                go.Scattergl(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    line={"color": color, "width": 3},
                    name=f"Block axis: {name}",
                    hoverinfo="skip",
                )
            )

    figure.add_trace(
        go.Scattergl(
            x=[record["col"] for record in records],
            y=[record["row"] for record in records],
            mode="markers",
            marker={"size": 18, "color": "rgba(0,0,0,0.01)"},
            customdata=[
                [record["R"], record["C"], np.degrees(record["theta"]) % 180.0]
                for record in records
            ],
            hovertemplate=(
                "Block R=%{customdata[0]:.3f}<br>"
                "Block C=%{customdata[1]:.3f}<br>"
                "Fiber axis=%{customdata[2]:.1f}°<extra></extra>"
            ),
            name="Block values (hover)",
            showlegend=False,
        )
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 181)
    row, col = centre
    boundary_styles = (
        ("symmetric", "Overall selector limit", "#785EF0", "solid"),
        ("core_end", "Dense zone inner boundary", "#DC267F", "dot"),
        ("dense_end", "Dense / sparse boundary", "#56B4E9", "dash"),
        ("sparse_end", "Sparse zone outer boundary", "#E69F00", "dash"),
    )
    for key, name, color, dash in boundary_styles:
        radius = radii.get(key, np.nan)
        if not np.isfinite(radius) or radius <= 0:
            continue
        figure.add_trace(
            go.Scatter(
                x=col + radius * np.cos(theta),
                y=row + radius * np.sin(theta),
                mode="lines",
                line={"color": color, "width": 2, "dash": dash},
                name=name,
                hoverinfo="skip",
            )
        )
    figure.update_layout(
        title=(
            "Candidate 5: interactive blockwise fiber axes on the actual detect_mat"
            "<br><sup>Needle length = block R; visibility = block C; hover blocks for values</sup>"
        ),
        template="plotly_white",
        width=1000,
        height=900,
        legend={"groupclick": "toggleitem"},
    )
    figure.update_xaxes(showgrid=False, zeroline=False, constrain="domain")
    figure.update_yaxes(
        showgrid=False,
        zeroline=False,
        autorange="reversed",
        scaleanchor="x",
        scaleratio=1,
    )
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)


def render_samples() -> None:
    """Run the real notebook workflow and create all candidate figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report("Reproducing LightDetectFungi_Workflow.ipynb")
    segmented, workflow_stats = reproduce_notebook_segmentation()
    report(
        f"Segmentation contains {int(segmented.objmap[:].max())} labels; "
        f"foreground fraction={(segmented.objmask[:] > 0).mean():.4f}"
    )

    section, source_label, section_index = select_real_colony_section(segmented)
    report(
        f"Selected grid section {section_index} for mapped source label {source_label}; "
        f"section shape={section.gray[:].shape}"
    )

    operation = MeasureOrientationZones(quiver_block=BLOCK)
    report("Running MeasureOrientationZones on the real grid section")
    measurement_frame = operation.measure(section)
    current_inspect = operation.inspect(section, for_save=True)
    current_path = OUTPUT_DIR / "00_current_inspect_real_section.png"
    current_inspect.write_image(
        str(current_path),
        width=1500,
        height=1050,
        scale=1,
    )
    report(f"Wrote {current_path.name}")

    props, label_to_section = operation._prep(section)
    if not props:
        raise RuntimeError("The selected grid section has no measurable object")
    prop = max(props, key=lambda candidate: candidate.area)
    segmentation = compute_zone_segmentation(
        section,
        prop,
        params=operation._zone_params(),
    )
    tile, object_mask, centre = operation._resolve_tile(
        section,
        segmentation,
        prop,
        label_to_section,
    )
    phi, coherence, grad_phi = orientation_field(
        tile,
        operation.sigma_d,
        operation.sigma_i,
    )
    distance_map = distance_from_point(tile.shape, centre)
    bounds = operation._zone_bounds(segmentation)
    selectors = {
        zone: zone_selector(
            distance_map,
            lower,
            upper,
            object_mask,
            "Radial",
        )
        for zone, (lower, upper) in bounds.items()
    }
    records = {
        zone: block_summaries(phi, coherence, selector, BLOCK)
        for zone, selector in selectors.items()
    }
    metrics = {
        zone: aggregate_orientation(
            phi,
            coherence,
            grad_phi,
            selector,
        )
        for zone, selector in selectors.items()
    }
    radii = {
        "symmetric": float(segmentation.symmetric_radius),
        "core_end": float(segmentation.core_end_radius),
        "dense_end": float(segmentation.dense_end_radius),
        "sparse_end": float(segmentation.sparse_end_radius),
    }

    render_block_overlay(
        tile,
        records["Overall"],
        centre,
        radii,
        OUTPUT_DIR / "01_blockwise_mean_axes.png",
    )
    render_zone_arrows(
        tile,
        records,
        centre,
        radii,
        OUTPUT_DIR / "02_zone_double_headed_axes.png",
    )
    render_calculation_triptych(
        tile,
        records["Overall"],
        coherence,
        grad_phi,
        selectors,
        centre,
        radii,
        metrics,
        OUTPUT_DIR / "03_calculation_triptych.png",
    )
    render_axial_rose(
        tile,
        records["Overall"],
        phi,
        coherence,
        selectors,
        grad_phi,
        centre,
        radii,
        OUTPUT_DIR / "04_axial_rose_by_zone.png",
    )
    render_interactive_block_overlay(
        tile,
        records["Overall"],
        centre,
        radii,
        OUTPUT_DIR / "05_interactive_block_axes.html",
    )

    summary = {
        "source_notebook": str(
            Path("/Users/alex/Projects/Neurospora/notebooks/LightDetectFungi_Workflow.ipynb")
        ),
        "source_image": str(IMAGE_PATH),
        "processed_shape": list(segmented.detect_mat[:].shape),
        "segmentation_labels": int(segmented.objmap[:].max()),
        "foreground_fraction": float((segmented.objmask[:] > 0).mean()),
        "workflow_stats": workflow_stats,
        "selected_source_label": source_label,
        "selected_section_index": section_index,
        "section_shape": list(section.detect_mat[:].shape),
        "analysis_tile_shape": list(tile.shape),
        "analysis_label": int(prop.label),
        "analysis_centre_rc": [float(value) for value in centre],
        "radii": radii,
        "metrics_radial": {
            zone: {
                "R": float(values[0]),
                "turning_rad_per_px": float(values[1]),
                "coherence": float(values[2]),
            }
            for zone, values in metrics.items()
        },
        "measurement_columns": measurement_frame.to_dict(orient="records"),
        "block_size_px": BLOCK,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    report("All real-image samples completed")


if __name__ == "__main__":
    render_samples()
