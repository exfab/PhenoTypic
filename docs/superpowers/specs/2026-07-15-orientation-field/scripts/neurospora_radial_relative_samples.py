from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.measure import regionprops

import neurospora_orientation_samples as base
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


OUTPUT_DIR = base.OUTPUT_DIR
N_ANGLE_BINS = 72
DISPLAY_PAIR = ((1116, 35), (626, 18))


@dataclass
class ColonyField:
    """All arrays and summaries needed for one real colony example."""

    label: int
    section_index: int
    area: float
    tile: np.ndarray
    object_mask: np.ndarray
    centre: tuple[float, float]
    radii: dict[str, float]
    phi: np.ndarray
    theta: np.ndarray
    coherence: np.ndarray
    grad_phi: np.ndarray
    alpha: np.ndarray
    delta: np.ndarray
    selectors: dict[str, np.ndarray]
    valid_selectors: dict[str, np.ndarray]
    metrics: dict[str, tuple[float, float, float]]
    direction_dense: float
    radial_order: dict[str, float]
    radial_concentration: dict[str, float]
    mean_abs_delta_deg: dict[str, float]
    radial_turning: dict[str, float]
    effective_support: dict[str, float]
    occupied_angle_fraction: dict[str, float]
    formation_profiles: dict[str, np.ndarray]
    formation_peaks_deg: dict[str, list[float]]
    sector_delta_deg: dict[str, np.ndarray]
    sector_reliability: dict[str, np.ndarray]


def axial_wrap(angle: np.ndarray | float) -> np.ndarray:
    """Wrap an axial angular difference to [-pi/2, pi/2)."""
    value = np.asarray(angle)
    return 0.5 * np.arctan2(np.sin(2.0 * value), np.cos(2.0 * value))


def weighted_axial_stats(
    delta: np.ndarray,
    coherence: np.ndarray,
    selector: np.ndarray,
) -> tuple[float, float, float]:
    """Return radial order, delta concentration, and mean absolute delta."""
    weights = coherence[selector]
    if weights.size == 0 or float(weights.sum()) <= 1e-12:
        return np.nan, np.nan, np.nan
    values = delta[selector]
    mean_cos = float(np.sum(weights * np.cos(2.0 * values)) / weights.sum())
    mean_sin = float(np.sum(weights * np.sin(2.0 * values)) / weights.sum())
    order = mean_cos
    concentration = float(np.hypot(mean_cos, mean_sin))
    mean_abs = float(np.degrees(np.sum(weights * np.abs(values)) / weights.sum()))
    return order, concentration, mean_abs


def radial_derivative(
    delta: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Return pi-safe orientation change projected along radial spokes."""
    cosine = np.cos(2.0 * delta)
    sine = np.sin(2.0 * delta)
    cosine_y, cosine_x = np.gradient(cosine)
    sine_y, sine_x = np.gradient(sine)
    radial_x = np.cos(alpha)
    radial_y = np.sin(alpha)
    cosine_r = cosine_x * radial_x + cosine_y * radial_y
    sine_r = sine_x * radial_x + sine_y * radial_y
    return 0.5 * np.sqrt(cosine_r**2 + sine_r**2)


def angular_profile(
    tile: np.ndarray,
    alpha: np.ndarray,
    delta: np.ndarray,
    coherence: np.ndarray,
    selector: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    """Compute exploratory evidence for radially aligned formation angles."""
    low, high = np.percentile(tile[selector], (5.0, 99.5))
    intensity = np.clip((tile - low) / max(high - low, 1e-12), 0.0, 1.0)
    radial_alignment = np.clip(np.cos(2.0 * delta), 0.0, 1.0)
    weights = intensity * coherence * radial_alignment * selector
    angle = np.mod(alpha, 2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, N_ANGLE_BINS + 1)
    weighted, _ = np.histogram(angle, bins=edges, weights=weights)
    support, _ = np.histogram(angle, bins=edges, weights=selector.astype(float))
    profile = weighted / np.maximum(support, 1.0)
    profile = gaussian_filter1d(profile, 1.4, mode="wrap")
    if float(profile.max()) > 0:
        peaks, _ = find_peaks(
            np.r_[profile, profile, profile],
            prominence=0.10 * float(profile.max()),
            distance=3,
        )
        peaks = np.unique(peaks[(peaks >= N_ANGLE_BINS) & (peaks < 2 * N_ANGLE_BINS)] % N_ANGLE_BINS)
    else:
        peaks = np.asarray([], dtype=int)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_degrees = [float(np.degrees(centers[index]) % 360.0) for index in peaks]
    return profile, peak_degrees


def sector_delta(
    alpha: np.ndarray,
    delta: np.ndarray,
    coherence: np.ndarray,
    selector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize radial-relative tilt in matched polar-angle sectors."""
    edges = np.linspace(0.0, 2.0 * np.pi, N_ANGLE_BINS + 1)
    angle = np.mod(alpha, 2.0 * np.pi)
    result = np.full(N_ANGLE_BINS, np.nan)
    reliability = np.zeros(N_ANGLE_BINS)
    for index in range(N_ANGLE_BINS):
        selected = selector & (angle >= edges[index]) & (angle < edges[index + 1])
        weights = coherence[selected]
        if weights.size < 8 or float(weights.sum()) <= 1e-12:
            continue
        values = delta[selected]
        mean_cos = float(np.sum(weights * np.cos(2.0 * values)))
        mean_sin = float(np.sum(weights * np.sin(2.0 * values)))
        result[index] = np.degrees(0.5 * np.arctan2(mean_sin, mean_cos))
        reliability[index] = np.hypot(mean_cos, mean_sin) / float(weights.sum())
    return result, reliability


def build_colony_field(segmented, prop, section_index: int, operation) -> ColonyField | None:
    """Compute current and proposed fields for one mapped colony."""
    segmentation = compute_zone_segmentation(segmented, prop, params=operation._zone_params())
    if not segmentation.zones_computed:
        return None
    tile, object_mask, centre = operation._resolve_tile(
        segmented,
        segmentation,
        prop,
        {int(prop.label): int(section_index)},
    )
    if min(tile.shape) < 3:
        return None
    phi, coherence, grad_phi = orientation_field(tile, operation.sigma_d, operation.sigma_i)
    theta = phi + np.pi / 2.0
    distance = distance_from_point(tile.shape, centre)
    row, col = np.indices(tile.shape)
    alpha = np.arctan2(row - centre[0], col - centre[1])
    delta = axial_wrap(theta - alpha)
    bounds = operation._zone_bounds(segmentation)
    selectors = {
        zone: zone_selector(distance, lower, upper, object_mask, "Radial")
        for zone, (lower, upper) in bounds.items()
    }
    valid_selectors = {
        zone: zone_selector(distance, lower, upper, object_mask, "Mask")
        for zone, (lower, upper) in bounds.items()
    }
    if any(int(selectors[zone].sum()) < 20 for zone in ("Dense", "Sparse")):
        return None
    metrics = {
        zone: aggregate_orientation(phi, coherence, grad_phi, selector)
        for zone, selector in selectors.items()
    }
    direction_dense = float(_resultant_direction(phi, coherence, selectors["Dense"]) + np.pi / 2.0)
    dr = radial_derivative(delta, alpha)
    radial_order: dict[str, float] = {}
    radial_concentration: dict[str, float] = {}
    mean_abs_delta_deg: dict[str, float] = {}
    radial_turning: dict[str, float] = {}
    effective_support: dict[str, float] = {}
    occupied_angle_fraction: dict[str, float] = {}
    formation_profiles: dict[str, np.ndarray] = {}
    formation_peaks_deg: dict[str, list[float]] = {}
    sector_delta_deg: dict[str, np.ndarray] = {}
    sector_reliability: dict[str, np.ndarray] = {}
    for zone in ("Dense", "Sparse"):
        # The phenotype is normalized over detected local structure. More pixels
        # or branches with the same delta therefore change support, not the
        # orientation estimate. Support is retained as a separate QC value.
        selector = valid_selectors[zone]
        if int(selector.sum()) < 20:
            return None
        order, concentration, mean_abs = weighted_axial_stats(delta, coherence, selector)
        radial_order[zone] = order
        radial_concentration[zone] = concentration
        mean_abs_delta_deg[zone] = mean_abs
        weights = coherence[selector]
        radial_turning[zone] = float(np.sum(weights * dr[selector]) / weights.sum())
        effective_support[zone] = float(weights.sum() ** 2 / max(float(np.sum(weights**2)), 1e-12))
        occupied_bins, _ = np.histogram(
            np.mod(alpha[selector], 2.0 * np.pi),
            bins=np.linspace(0.0, 2.0 * np.pi, N_ANGLE_BINS + 1),
        )
        occupied_angle_fraction[zone] = float(np.mean(occupied_bins > 0))
        profile, peaks = angular_profile(tile, alpha, delta, coherence, selector)
        formation_profiles[zone] = profile
        formation_peaks_deg[zone] = peaks
        sector, reliability = sector_delta(alpha, delta, coherence, selector)
        sector_delta_deg[zone] = sector
        sector_reliability[zone] = reliability
    return ColonyField(
        label=int(prop.label),
        section_index=int(section_index),
        area=float(prop.area),
        tile=tile,
        object_mask=object_mask,
        centre=(float(centre[0]), float(centre[1])),
        radii={
            "symmetric": float(segmentation.symmetric_radius),
            "core_end": float(segmentation.core_end_radius),
            "dense_end": float(segmentation.dense_end_radius),
            "sparse_end": float(segmentation.sparse_end_radius),
        },
        phi=phi,
        theta=theta,
        coherence=coherence,
        grad_phi=grad_phi,
        alpha=alpha,
        delta=delta,
        selectors=selectors,
        valid_selectors=valid_selectors,
        metrics=metrics,
        direction_dense=direction_dense,
        radial_order=radial_order,
        radial_concentration=radial_concentration,
        mean_abs_delta_deg=mean_abs_delta_deg,
        radial_turning=radial_turning,
        effective_support=effective_support,
        occupied_angle_fraction=occupied_angle_fraction,
        formation_profiles=formation_profiles,
        formation_peaks_deg=formation_peaks_deg,
        sector_delta_deg=sector_delta_deg,
        sector_reliability=sector_reliability,
    )


def select_two_colonies(segmented) -> tuple[list[ColonyField], list[dict[str, float]]]:
    """Select a reliable pair whose current dense mean axes differ strongly."""
    operation = MeasureOrientationZones(quiver_block=base.BLOCK)
    props, label_to_section = operation._prep(segmented)
    mapped = [prop for prop in props if int(prop.label) in label_to_section and prop.area >= 1000]
    fields: list[ColonyField] = []
    for prop in mapped:
        try:
            field = build_colony_field(segmented, prop, label_to_section[int(prop.label)], operation)
        except (ValueError, IndexError, FloatingPointError):
            continue
        if field is not None and np.isfinite(field.metrics["Dense"][0]):
            fields.append(field)
    if len(fields) < 2:
        raise RuntimeError("Fewer than two mapped colonies had usable dense and sparse zones")
    best: tuple[float, ColonyField, ColonyField] | None = None
    for first_index, first in enumerate(fields):
        for second in fields[first_index + 1 :]:
            angle_difference = float(abs(axial_wrap(first.direction_dense - second.direction_dense)))
            reliability = min(first.metrics["Dense"][0], second.metrics["Dense"][0])
            contrast = angle_difference / (np.pi / 2.0)
            score = reliability * contrast
            if best is None or score > best[0]:
                best = (score, first, second)
    assert best is not None
    ranking = [
        {
            "label": float(field.label),
            "section": float(field.section_index),
            "area": field.area,
            "dense_axis_deg": float(np.degrees(field.direction_dense) % 180.0),
            "dense_R": float(field.metrics["Dense"][0]),
            "dense_current_turning": float(field.metrics["Dense"][1]),
            "dense_radial_order": float(field.radial_order["Dense"]),
            "dense_mean_abs_delta_deg": float(field.mean_abs_delta_deg["Dense"]),
        }
        for field in sorted(fields, key=lambda item: item.metrics["Dense"][0], reverse=True)
    ]
    return [best[1], best[2]], ranking


def add_local_axes(axis, field: ColonyField, zone: str = "Dense", *, valid_only: bool = False) -> None:
    """Overlay local blockwise fiber axes for one zone."""
    records = base.block_summaries(
        field.phi,
        field.coherence,
        field.valid_selectors[zone] if valid_only else field.selectors[zone],
        base.BLOCK,
    )
    base.add_headless_axes(axis, records, base.BLOCK, fixed_color="#56B4E9")


def render_current_pair(fields: list[ColonyField], output_path: Path) -> None:
    """Show the current absolute-orientation calculation for two colonies."""
    figure = plt.figure(figsize=(16, 13), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
    for row, field in enumerate(fields):
        image_axis = figure.add_subplot(grid[row, 0])
        base.show_actual_layer(
            image_axis,
            field.tile,
            f"Colony {row + 1}: actual detect_mat + Dense local axes",
        )
        add_local_axes(image_axis, field)
        base.draw_zone_boundaries(image_axis, field.centre, field.radii)
        rose_axis = figure.add_subplot(grid[row, 1], projection="polar")
        selector = field.selectors["Dense"]
        theta = np.mod(field.theta[selector], np.pi)
        weights = field.coherence[selector]
        edges = np.linspace(0.0, np.pi, 19)
        counts, _ = np.histogram(theta, bins=edges, weights=weights)
        probability = counts / counts.sum()
        centers = 0.5 * (edges[:-1] + edges[1:])
        rose_axis.bar(
            np.r_[centers, centers + np.pi],
            np.r_[probability, probability],
            width=(np.pi / 18.0) * 0.92,
            color="#CC79A7",
            alpha=0.75,
        )
        direction = field.direction_dense
        rose_axis.plot([direction, direction], [0, probability.max()], color="#003660", lw=3)
        rose_axis.plot([direction + np.pi, direction + np.pi], [0, probability.max()], color="#003660", lw=3)
        rose_axis.set_theta_zero_location("E")
        rose_axis.set_theta_direction(-1)
        rose_axis.set_yticklabels([])
        current = field.metrics["Dense"]
        rose_axis.set_title(
            f"Dense absolute fiber-axis distribution\n"
            f"mean axis={np.degrees(direction) % 180:.1f}°, "
            f"R={current[0]:.3f}, current T={current[1]:.4f} rad/px",
            fontsize=11,
        )
    difference = float(np.degrees(abs(axial_wrap(fields[0].direction_dense - fields[1].direction_dense))))
    figure.suptitle(
        "Existing calculation on two real colonies\n"
        f"Their Dense mean axes differ by {difference:.1f}°, but that global difference is not turning. "
        "R measures common absolute alignment within each ring.",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def render_radial_pair(fields: list[ColonyField], output_path: Path) -> None:
    """Show radial-relative tilt and formation-angle evidence for two colonies."""
    figure = plt.figure(figsize=(22, 13), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, width_ratios=(1.2, 1.05, 1.0, 1.15))
    angle_centers = np.linspace(2.5, 357.5, N_ANGLE_BINS)
    for row, field in enumerate(fields):
        image_axis = figure.add_subplot(grid[row, 0])
        base.show_actual_layer(image_axis, field.tile, f"Colony {row + 1}: source + local fiber axes")
        add_local_axes(image_axis, field, "Overall")
        base.draw_zone_boundaries(image_axis, field.centre, field.radii)
        for degree in np.arange(0.0, 360.0, 45.0):
            angle = np.radians(degree)
            radius = field.radii["sparse_end"]
            image_axis.plot(
                [field.centre[1], field.centre[1] + radius * np.cos(angle)],
                [field.centre[0], field.centre[0] + radius * np.sin(angle)],
                color="white",
                lw=0.55,
                alpha=0.35,
            )

        delta_axis = figure.add_subplot(grid[row, 1])
        shown = np.where(field.selectors["Overall"], np.degrees(field.delta), np.nan)
        delta_image = delta_axis.imshow(
            shown,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-90.0, vcenter=0.0, vmax=90.0),
        )
        base.draw_zone_boundaries(delta_axis, field.centre, field.radii)
        delta_axis.set_title(
            "Radial-relative tilt δ = fiber axis − local spoke\n"
            "0° = straight radial; ±90° = tangential",
            fontsize=10,
        )
        delta_axis.set_axis_off()
        figure.colorbar(delta_image, ax=delta_axis, fraction=0.045, pad=0.02, label="δ (degrees)")

        polar_axis = figure.add_subplot(grid[row, 2], projection="polar")
        for zone, color in (("Dense", "#CC79A7"), ("Sparse", "#D55E00")):
            profile = field.formation_profiles[zone]
            normalized = profile / max(float(profile.max()), 1e-12)
            radians = np.radians(angle_centers)
            polar_axis.plot(np.r_[radians, radians[0]], np.r_[normalized, normalized[0]], color=color, lw=2, label=zone)
            for degree in field.formation_peaks_deg[zone]:
                polar_axis.plot(np.radians(degree), 1.05, marker="o", color=color, ms=5)
        polar_axis.set_theta_zero_location("E")
        polar_axis.set_theta_direction(-1)
        polar_axis.set_yticklabels([])
        polar_axis.set_title("Formation-angle evidence\npeaks = strong radial structure", fontsize=10)
        polar_axis.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=8)

        sector_axis = figure.add_subplot(grid[row, 3])
        for zone, color in (("Dense", "#CC79A7"), ("Sparse", "#D55E00")):
            values = field.sector_delta_deg[zone].copy()
            values[field.sector_reliability[zone] < 0.15] = np.nan
            sector_axis.plot(angle_centers, values, color=color, lw=1.8, label=f"{zone} mean δ")
        dense = field.sector_delta_deg["Dense"]
        sparse = field.sector_delta_deg["Sparse"]
        reliable = (field.sector_reliability["Dense"] >= 0.15) & (field.sector_reliability["Sparse"] >= 0.15)
        change = np.degrees(axial_wrap(np.radians(sparse - dense)))
        change[~reliable] = np.nan
        sector_axis.plot(angle_centers, change, color="#009E73", lw=1.2, alpha=0.9, label="Sparse − Dense δ")
        sector_axis.axhline(0.0, color="black", lw=0.7)
        sector_axis.set_xlim(0, 360)
        sector_axis.set_ylim(-90, 90)
        sector_axis.set_xticks(np.arange(0, 361, 90))
        sector_axis.set_xlabel("Position around colony (degrees; 0° = right, clockwise)")
        sector_axis.set_ylabel("radial-relative tilt δ (degrees)")
        sector_axis.grid(alpha=0.2)
        sector_axis.legend(fontsize=8, loc="upper right")
        sector_axis.set_title(
            f"Matched angular sectors\n"
            f"Dense: Srad={field.radial_order['Dense']:.2f}, "
            f"mean|δ|={field.mean_abs_delta_deg['Dense']:.1f}°, "
            f"radial T={field.radial_turning['Dense']:.4f} rad/px\n"
            f"QC only: effective support={field.effective_support['Dense']:.0f}, "
            f"angular coverage={field.occupied_angle_fraction['Dense']:.0%}",
            fontsize=10,
        )
    figure.suptitle(
        "Proposed field-based calculation: compare each local axis with the radial spoke at the same angle\n"
        "This makes straight branches at 0°, 90°, or 180° equivalent (δ≈0), while preserving where around the colony radial structures emerge.",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def render_concept_control(output_path: Path) -> None:
    """Demonstrate current versus radial-relative behavior on ideal controls."""
    size = 181
    row, col = np.indices((size, size))
    center = (size - 1) / 2.0
    alpha = np.arctan2(row - center, col - center)
    annulus = (np.hypot(row-center, col-center) > 15) & (np.hypot(row-center, col-center) < 80)
    angular_width = np.radians(5.0)
    horizontal = annulus & (np.abs(axial_wrap(alpha)) < angular_width)
    vertical = annulus & (np.abs(axial_wrap(alpha - np.pi / 2.0)) < angular_width)
    single_right = annulus & (np.abs(axial_wrap(alpha)) < angular_width) & (np.cos(alpha) > 0)
    controls = (
        ("One radial branch at 0°", alpha, single_right),
        ("Opposing radial branches 0°/180°", alpha, horizontal),
        ("Opposing radial branches 90°/270°", alpha, vertical),
        ("Dense straight radial fan", alpha, annulus),
        (
            "Dense fan + outward 25° bend",
            alpha + np.radians(25.0) * np.clip(np.hypot(row-center, col-center) / center, 0, 1),
            annulus,
        ),
    )
    figure, axes = plt.subplots(1, 5, figsize=(23, 5), constrained_layout=True)
    for axis, (name, theta, mask) in zip(axes, controls):
        delta = axial_wrap(theta - alpha)
        cosine = np.cos(2.0 * theta)
        sine = np.sin(2.0 * theta)
        cy, cx = np.gradient(cosine)
        sy, sx = np.gradient(sine)
        current_turning = 0.5 * np.mean(np.sqrt(cx[mask]**2 + cy[mask]**2 + sx[mask]**2 + sy[mask]**2))
        proposed_turning_map = radial_derivative(delta, alpha)
        proposed_turning = float(np.mean(proposed_turning_map[mask]))
        current_r = float(np.hypot(np.mean(np.cos(2.0 * theta[mask])), np.mean(np.sin(2.0 * theta[mask]))))
        radial_order = float(np.mean(np.cos(2.0 * delta[mask])))
        axis.imshow(np.where(mask, np.degrees(delta), np.nan), cmap="RdBu_r", vmin=-90, vmax=90)
        step = 18
        rows = np.arange(18, size-18, step)
        cols = np.arange(18, size-18, step)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        inside = mask[rr, cc]
        u = np.cos(theta[rr, cc])[inside]
        v = np.sin(theta[rr, cc])[inside]
        axis.quiver(cc[inside], rr[inside], u, v, color="black", pivot="middle", headlength=0, headaxislength=0, scale=18)
        axis.set_title(
            f"{name}\nCurrent R={current_r:.2f}, T={current_turning:.4f}\n"
            f"Radial S={radial_order:.2f}, radial T={proposed_turning:.4f}",
            fontsize=10,
        )
        axis.set_axis_off()
    figure.suptitle(
        "Branch angle and branch number controls: every straight radial case has radial S=1 and radial T=0",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def create_samples() -> None:
    """Create two-colony and ideal-control calculation figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    segmented, _ = base.reproduce_notebook_segmentation()
    operation = MeasureOrientationZones(quiver_block=base.BLOCK)
    props = {
        int(prop.label): prop
        for prop in regionprops(
            segmented.objmap[:],
            intensity_image=segmented.gray[:].astype(np.float64, copy=False),
        )
    }
    fields = []
    for label, section in DISPLAY_PAIR:
        field = build_colony_field(segmented, props[label], section, operation)
        if field is None:
            raise RuntimeError(f"Could not rebuild display colony label {label}")
        fields.append(field)
    ranking: list[dict[str, float]] = []
    for field in fields:
        print(
            f"selected label={field.label} section={field.section_index} "
            f"axis={np.degrees(field.direction_dense) % 180:.1f} "
            f"R={field.metrics['Dense'][0]:.3f} "
            f"currentT={field.metrics['Dense'][1]:.4f} "
            f"Srad={field.radial_order['Dense']:.3f}",
            flush=True,
        )
    render_current_pair(fields, OUTPUT_DIR / "06_two_colonies_current_calculation.png")
    render_radial_pair(fields, OUTPUT_DIR / "07_two_colonies_radial_relative.png")
    render_concept_control(OUTPUT_DIR / "08_orientation_metric_controls.png")
    summary = {
        "selected": [
            {
                "label": field.label,
                "section_index": field.section_index,
                "area": field.area,
                "dense_absolute_axis_deg": float(np.degrees(field.direction_dense) % 180.0),
                "current_metrics": {
                    zone: {"R": values[0], "turning_rad_per_px": values[1], "coherence": values[2]}
                    for zone, values in field.metrics.items()
                },
                "proposed_metrics": {
                    zone: {
                        "radial_order": field.radial_order[zone],
                        "radial_delta_concentration": field.radial_concentration[zone],
                        "mean_abs_delta_deg": field.mean_abs_delta_deg[zone],
                        "radial_turning_rad_per_px": field.radial_turning[zone],
                        "effective_support_qc": field.effective_support[zone],
                        "occupied_angle_fraction_qc": field.occupied_angle_fraction[zone],
                        "formation_peaks_deg": field.formation_peaks_deg[zone],
                    }
                    for zone in ("Dense", "Sparse")
                },
            }
            for field in fields
        ],
        "top_candidates": ranking[:15],
    }
    (OUTPUT_DIR / "two_colony_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    create_samples()
