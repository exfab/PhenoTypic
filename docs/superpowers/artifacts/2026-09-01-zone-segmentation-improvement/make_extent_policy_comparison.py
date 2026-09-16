"""Render a corrected, independently refit P95 versus P100 comparison.

The input crop is an illustrative extreme case chosen to make the extent-policy
difference visible. It is not presented as representative.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Patch
from numpy.typing import NDArray
from PIL import Image as PILImage
from scipy.ndimage import distance_transform_edt

from phenotypic.sdk_.orientation_fields import (
    literal_crossing_ring_profile,
    literal_skeleton_ring_crossings,
)
from phenotypic.util._orientation_field import orientation_field


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

ROOT = Path(
    "/Users/alex/Library/CloudStorage/GoogleDrive-anguy344@ucr.edu/"
    "My Drive/Active Projects/rbeck/BranchZoneMask/Images/HyphaeAnalysis/"
    "NeurosporaXylan/crops"
)
STEM = "Image29_obj1_d000273_300_001_2025-12-12_03-00-49_rgb"
CACHE = Path(
    "/Users/alex/Projects/PhenoTypic/scratch/"
    "orientation_zone_final_detector_feature_cache/"
    "NeurosporaXylan__Image29_obj1_d000273_300_001_2025-12-12_03-00-49_rgb"
    "__ac9d8f0780575bce.npz"
)
OUTPUT = Path(__file__).with_name("extent-policy-comparison.png")
EXPECTED_HASHES = {
    ROOT / f"{STEM}.tiff": (
        "fae2af401d518b1d2a321089e2bacb205f8363d44e525464eb9a4a5d1853520d"
    ),
    ROOT / f"{STEM}.labels.png": (
        "fbacf8b5cabdc63baa853462775d2d78a69af7ef54f2864ae71491bba5f72d83"
    ),
    CACHE: "d360542a39f82c967413861ec0d371e9fc62a514e26aec43bacbd476f47b534b",
}

RING_WIDTH = 8.0
MINIMUM_SEGMENT = 4
MINIMUM_CROSSINGS = 3
MINIMUM_RESULTANT = 0.30
MINIMUM_RING_COHERENCE = 0.15
SUPPORT_WEIGHT = 4.0
OUTER_SUPPORT_MARGIN = 0.0

UNRESOLVED_COLOR = "#D55E00"
DENSE_COLOR = "#009E73"
SPARSE_COLOR = "#0072B2"
TAIL_COLOR = "#CC79A7"


@dataclass(frozen=True)
class FittedZones:
    """One independently fitted Method B extent policy."""

    percentile: float
    unresolved: float
    dense: float
    outer: float
    full: float
    retained_fraction: float
    ring_count: int
    crossing_count: int
    method: str


def verify_input_hashes() -> None:
    """Reject changed external evidence before rendering."""
    for path, expected in EXPECTED_HASHES.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(
                f"Input hash mismatch for {path}: expected {expected}, got {digest}"
            )


def distance_map(
    shape: tuple[int, int], center: tuple[float, float]
) -> FloatArray:
    """Return Euclidean distance from one row-column center."""
    rows, cols = np.indices(shape, dtype=float)
    return np.hypot(rows - center[0], cols - center[1])


def distance_transform_center(mask: BoolArray) -> tuple[float, float]:
    """Return the row-major first maximum of the mask distance transform."""
    transform = distance_transform_edt(mask)
    row, col = np.unravel_index(int(np.argmax(transform)), transform.shape)
    return float(row), float(col)


def scaled_object_signal(
    signal: FloatArray,
    mask: BoolArray,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Return statistical scaling, derivative fill, and source validity."""
    source = np.asarray(signal, dtype=float)
    validity = mask & np.isfinite(source)
    population = source[validity]
    if population.size == 0:
        zeros = np.zeros_like(source)
        return zeros, zeros, validity
    low, high = np.percentile(population, [2.0, 98.0], method="linear")
    if not np.isfinite(high - low) or high <= low:
        zeros = np.zeros_like(source)
        return zeros, zeros, validity
    scaled = np.clip((source - low) / (high - low), 0.0, 1.0)
    derivative_fill = float(np.median(scaled[validity]))
    derivative_scaled = np.where(np.isfinite(scaled), scaled, derivative_fill)
    return scaled, derivative_scaled, validity


def robust_standardize(matrix: FloatArray) -> FloatArray:
    """Median-impute and robust-standardize feature columns."""
    result = np.asarray(matrix, dtype=float).copy()
    for column in range(result.shape[1]):
        values = result[:, column]
        finite = np.isfinite(values)
        fill = float(np.median(values[finite])) if finite.any() else 0.0
        values[~finite] = fill
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = max(1.4826 * mad, float(np.std(values)), np.finfo(float).eps)
        result[:, column] = (values - median) / scale
    return result


def segment_sse(matrix: FloatArray, start: int, stop: int) -> float:
    """Return direct within-segment sum of squared error."""
    segment = matrix[start:stop]
    if segment.size == 0:
        return float("inf")
    return float(np.square(segment - segment.mean(axis=0)).sum())


def bridge_short_gaps(support: BoolArray, maximum_gap: int = 0) -> BoolArray:
    """Bridge bounded internal unsupported runs."""
    result = np.asarray(support, dtype=bool).copy()
    if maximum_gap <= 0:
        return result
    padded = np.concatenate(([True], result, [True]))
    changes = np.diff(padded.astype(np.int8))
    for start, stop in zip(
        np.flatnonzero(changes == -1),
        np.flatnonzero(changes == 1),
        strict=True,
    ):
        if start > 0 and stop < result.size and stop - start <= maximum_gap:
            result[start:stop] = True
    return result


def fit_method_b(
    mask: BoolArray,
    signal: FloatArray,
    center: tuple[float, float],
    percentile: float,
) -> FittedZones:
    """Build and fit the complete Method B profile for one extent policy."""
    distances = distance_map(mask.shape, center)
    mask_distances = distances[mask]
    full = float(np.max(mask_distances))
    outer = (
        full
        if percentile == 100.0
        else float(np.percentile(mask_distances, percentile, method="linear"))
    )
    inclusive_outer = np.nextafter(outer, np.inf)
    selected_mask = mask & (distances < inclusive_outer)
    retained = float(selected_mask.sum() / mask.sum())
    ring_count = max(1, int(np.ceil(outer / RING_WIDTH)))
    radii = (np.arange(ring_count, dtype=float) + 0.5) * RING_WIDTH

    scaled, derivative_scaled, signal_validity = scaled_object_signal(
        signal, mask
    )
    gradient_y, gradient_x = np.gradient(derivative_scaled)
    edge_energy = np.hypot(gradient_x, gradient_y)
    phi, coherence, _ = orientation_field(derivative_scaled)
    fiber_axis = (phi + np.pi / 2.0 + np.pi / 2.0) % np.pi - np.pi / 2.0
    rows, cols = np.indices(mask.shape, dtype=float)
    azimuth = np.arctan2(rows - center[0], cols - center[1])
    radial_tilt = 0.5 * np.arctan2(
        np.sin(2.0 * (fiber_axis - azimuth)),
        np.cos(2.0 * (fiber_axis - azimuth)),
    )

    transform = literal_skeleton_ring_crossings(
        mask,
        fiber_axis,
        coherence,
        distances,
        center,
        radii,
        selector=selected_mask,
        minimum_coherence=MINIMUM_RING_COHERENCE,
        minimum_crossing_resultant=0.15,
    )
    raw_profile = literal_crossing_ring_profile(
        transform,
        minimum_points=1,
        minimum_resultant=0.0,
    )

    features: list[list[float]] = []
    for index, radius in enumerate(radii):
        geometric_ring = (np.abs(distances - radius) <= RING_WIDTH / 2.0) & (
            distances < inclusive_outer
        )
        selected = geometric_ring & mask
        valid_selected = selected & signal_validity
        reliable = (
            valid_selected
            & np.isfinite(coherence)
            & (coherence >= MINIMUM_RING_COHERENCE)
        )
        if valid_selected.any():
            intensity_mean = float(np.mean(scaled[valid_selected]))
            intensity_variance = float(np.var(scaled[valid_selected]))
            mean_edge = float(np.mean(edge_energy[valid_selected]))
        else:
            intensity_mean = intensity_variance = mean_edge = float("nan")
        if reliable.any():
            mean_coherence = float(np.mean(coherence[reliable]))
            tilt_resultant = float(
                abs(np.mean(np.exp(2j * radial_tilt[reliable])))
            )
        else:
            mean_coherence = tilt_resultant = float("nan")
        occupancy = (
            float(selected.sum() / geometric_ring.sum())
            if geometric_ring.any()
            else float("nan")
        )
        features.append(
            [
                intensity_mean,
                intensity_variance,
                occupancy,
                mean_coherence,
                tilt_resultant,
                mean_edge,
                float(raw_profile.resultant[index]),
            ]
        )

    crossing_resultant = np.nan_to_num(
        raw_profile.resultant,
        nan=-np.inf,
    )
    ring_coherence = np.asarray(features, dtype=float)[:, 3]
    support = bridge_short_gaps(
        (raw_profile.crossing_count >= MINIMUM_CROSSINGS)
        & (crossing_resultant >= MINIMUM_RESULTANT)
        & (
            np.nan_to_num(ring_coherence, nan=-np.inf)
            >= MINIMUM_RING_COHERENCE
        )
    )
    matrix = np.column_stack(
        (robust_standardize(np.asarray(features)), support * SUPPORT_WEIGHT)
    )

    best: tuple[float, int, int] | None = None
    for first in range(
        MINIMUM_SEGMENT,
        ring_count - 2 * MINIMUM_SEGMENT + 1,
    ):
        support_gain = float(support[first:].mean()) - float(
            support[:first].mean()
        )
        if support_gain < OUTER_SUPPORT_MARGIN:
            continue
        for second in range(
            first + MINIMUM_SEGMENT,
            ring_count - MINIMUM_SEGMENT + 1,
        ):
            if not support[first:second].any() or not support[second:].any():
                continue
            cost = (
                segment_sse(matrix, 0, first)
                + segment_sse(matrix, first, second)
                + segment_sse(matrix, second, ring_count)
            )
            candidate = (cost, first, second)
            if best is None or candidate < best:
                best = candidate

    method = "exact"
    if best is not None:
        _, first, second = best
        unresolved = float(radii[first] - RING_WIDTH / 2.0)
        dense = float(radii[second] - RING_WIDTH / 2.0)
    else:
        candidates: list[tuple[float, int]] = []
        unresolved_evidence = 1.0 - support.astype(float)
        for boundary in range(
            MINIMUM_SEGMENT,
            ring_count - MINIMUM_SEGMENT + 1,
        ):
            if not support[boundary:].any():
                continue
            cost = segment_sse(unresolved_evidence[:, None], 0, boundary)
            cost += segment_sse(
                unresolved_evidence[:, None],
                boundary,
                ring_count,
            )
            candidates.append((cost, boundary))
        if not candidates:
            raise RuntimeError(f"No supported Method B fit at P{percentile:g}")
        _, boundary = min(candidates)
        unresolved = dense = float(radii[boundary] - RING_WIDTH / 2.0)
        method = "collapsed"

    return FittedZones(
        percentile=percentile,
        unresolved=float(np.clip(unresolved, 0.0, outer)),
        dense=float(np.clip(dense, unresolved, outer)),
        outer=outer,
        full=full,
        retained_fraction=retained,
        ring_count=ring_count,
        crossing_count=len(transform.crossings),
        method=method,
    )


def hand_reference_radii(
    labels: NDArray[np.uint8],
    center: tuple[float, float],
) -> tuple[float, float, float]:
    """Return cumulative Neurospora P95 hand-reference radii."""
    normalized = np.asarray(labels, dtype=np.uint8).copy()
    normalized[normalized == 3] = 4
    distances = distance_map(normalized.shape, center)
    masks = (
        normalized == 1,
        (normalized == 1) | (normalized == 2),
        (normalized == 1) | (normalized == 2) | (normalized == 4),
    )
    return tuple(
        float(np.percentile(distances[selected], 95.0, method="linear"))
        for selected in masks
    )


def display_rgb(values: NDArray[np.generic]) -> FloatArray:
    """Scale a high-bit-depth RGB crop to a visible floating-point image."""
    rgb = np.asarray(values, dtype=float)
    finite = rgb[np.isfinite(rgb)]
    if finite.size == 0:
        return np.zeros_like(rgb)
    low, high = np.percentile(finite, [0.5, 99.5], method="linear")
    if not np.isfinite(high - low) or high <= low:
        return np.zeros_like(rgb)
    return np.clip((rgb - low) / (high - low), 0.0, 1.0)


def add_circle(
    axis: object,
    center: tuple[float, float],
    radius: float,
    color: str,
    *,
    linestyle: str,
    linewidth: float,
) -> None:
    """Add one row-column radial boundary to an image axis."""
    axis.add_patch(
        Circle(
            (center[1], center[0]),
            radius,
            fill=False,
            edgecolor=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    )


def render_extent_policy_comparison() -> None:
    """Render the three-panel extent-policy audit figure."""
    verify_input_hashes()
    cache = np.load(CACHE, allow_pickle=False)
    mask = np.asarray(cache["object_mask"], dtype=bool)
    signal = np.asarray(cache["signal"], dtype=float)
    rgb = display_rgb(np.asarray(tifffile.imread(ROOT / f"{STEM}.tiff")))
    labels = np.asarray(PILImage.open(ROOT / f"{STEM}.labels.png"))
    center = distance_transform_center(mask)
    p95 = fit_method_b(mask, signal, center, 95.0)
    p100 = fit_method_b(mask, signal, center, 100.0)
    hand = hand_reference_radii(labels, center)
    distances = distance_map(mask.shape, center)
    p95_selected = mask & (distances < np.nextafter(p95.outer, np.inf))
    excluded_tail = mask & ~p95_selected

    figure = Figure(figsize=(15.8, 6.4), dpi=180, facecolor="#F7F6F2")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    for axis in axes:
        axis.imshow(rgb)
        axis.set_axis_off()
        axis.set_aspect("equal")

    tail_overlay = np.zeros((*mask.shape, 4), dtype=float)
    tail_overlay[excluded_tail] = (0.80, 0.47, 0.65, 0.78)
    axes[0].imshow(tail_overlay)
    axes[0].plot(center[1], center[0], "+", color="white", ms=10, mew=2)
    for radius, color in zip(
        hand,
        (UNRESOLVED_COLOR, DENSE_COLOR, SPARSE_COLOR),
        strict=True,
    ):
        add_circle(
            axes[0],
            center,
            radius,
            color,
            linestyle=(0, (2, 2)),
            linewidth=2.0,
        )
    axes[0].set_title(
        "A  Hand grades and P95-excluded tail\n"
        f"Final-mask center = ({center[0]:.0f}, {center[1]:.0f})",
        loc="left",
        fontsize=11,
        weight="bold",
    )

    for axis, fit, panel in ((axes[1], p95, "B"), (axes[2], p100, "C")):
        for radius, color in (
            (fit.unresolved, UNRESOLVED_COLOR),
            (fit.dense, DENSE_COLOR),
            (fit.outer, SPARSE_COLOR),
        ):
            add_circle(
                axis,
                center,
                radius,
                color,
                linestyle="solid",
                linewidth=2.4,
            )
        if fit.percentile < 100.0:
            add_circle(
                axis,
                center,
                fit.full,
                TAIL_COLOR,
                linestyle=(0, (1, 2)),
                linewidth=1.7,
            )
        axis.plot(center[1], center[0], "+", color="white", ms=10, mew=2)
        axis.set_title(
            f"{panel}  Method B, P{fit.percentile:g} (independent refit)\n"
            f"U={fit.unresolved:.1f}, D={fit.dense:.1f}, O={fit.outer:.1f} px\n"
            f"{fit.retained_fraction:.1%} mask | {fit.crossing_count} crossings | "
            f"{fit.method}",
            loc="left",
            fontsize=9.5,
            weight="bold",
        )

    figure.suptitle(
        "outer_zone_percentile changes the fitted profile window and measurement limit",
        x=0.04,
        y=0.985,
        ha="left",
        fontsize=15,
        weight="bold",
    )
    legend = [
        Patch(facecolor=UNRESOLVED_COLOR, label="Unresolved boundary"),
        Patch(facecolor=DENSE_COLOR, label="Dense boundary"),
        Patch(facecolor=SPARSE_COLOR, label="Sparse measurement limit"),
        Patch(facecolor=TAIL_COLOR, label="P95-excluded selected-mask tail"),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.045),
    )
    figure.text(
        0.5,
        0.012,
        "Illustrative extreme case chosen to make the extent difference visible. "
        "Solid = algorithm; dotted in panel A = human P95 reference.\n"
        "Dotted magenta in panel B = detected P100 extent excluded at P95. "
        "P95 and P100 use the same final-mask center and are independently refit.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#333333",
    )
    figure.subplots_adjust(
        left=0.02, right=0.99, top=0.84, bottom=0.17, wspace=0.05
    )
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())


if __name__ == "__main__":
    render_extent_policy_comparison()
