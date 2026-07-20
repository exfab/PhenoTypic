from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phenotypic as pht
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.ndimage import label as connected_components
from scipy.optimize import linear_sum_assignment

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from render_matched_ring_comparison import (  # noqa: E402
    extract_profiles,
    label_centroid,
)
from render_ring_compounded_rotation import (  # noqa: E402
    extract_full_length_ring_fields,
)
from render_tangential_method_comparison import draw_rings  # noqa: E402
from render_twok_reconnected_orientation import (  # noqa: E402
    isolated_global_crop,
    load_twok_detection,
    report,
)
from phenotypic.measure import MeasureOrientationZones  # noqa: E402
from phenotypic.measure._measure_orientation_zones import (  # noqa: E402
    _FIBER_AXIS_OFFSET,
    _RADIAL_RELATIVE_MIN_COHERENCE,
)
from phenotypic.sdk_.orientation_fields import (  # noqa: E402
    LiteralSkeletonRingCrossing as RingCrossing,
)
from phenotypic.sdk_.orientation_fields import (  # noqa: E402
    literal_skeleton_ring_crossings,
)


ANALYSIS_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = ANALYSIS_DIR / "artifacts"
CACHE_DIR = ANALYSIS_DIR / "cache"
COMPOSITE_CACHE = CACHE_DIR / "composite.npy"
TWOK_OBJMAP_CACHE = CACHE_DIR / "twok_branch_reconnection_objmap.npy"
COLONIES = (("R3C4", 24), ("R4C6", 36))
POLICIES = (
    "reciprocal_one_to_one",
    "independent_many_to_one",
    "global_one_to_one",
)
SPECTRAL_NORM = Normalize(vmin=-180.0, vmax=180.0)


def load_point_matching_detection():
    """Load the detect-matrix and object-map layers used by this diagnostic.

    The local cache reconstructs only the two layers consumed by this script.
    It does not restore the notebook RGB or gray layers, so this helper must not
    be reused by diagnostics whose intensity source is anything other than
    ``detect_mat``. If either cache is absent, the full notebook loader is used.

    Returns:
        Grid image containing the cached orientation source and TwoK object map.
    """
    if COMPOSITE_CACHE.exists() and TWOK_OBJMAP_CACHE.exists():
        report("Loading cached detect_mat and branch-reconnection TwoK objmap")
        composite = np.asarray(
            np.load(COMPOSITE_CACHE, mmap_mode="r"), dtype=np.float32
        )
        detected = pht.GridImage(composite, nrows=6, ncols=10)
        detected.detect_mat[:] = composite
        detected.objmap[:] = np.asarray(
            np.load(TWOK_OBJMAP_CACHE, mmap_mode="r")
        )
        return detected

    detected, _old_segmented = load_twok_detection()
    return detected


@dataclass(frozen=True)
class CandidateEdge:
    """One eligible previous-ring correspondence candidate."""

    inner_id: int
    outer_id: int
    distance: float
    axial_delta: float
    unwrapped_outer: float
    signed_step: float
    prediction_residual: float
    cost: float


@dataclass(frozen=True)
class PointState:
    """Accumulation state assigned to one crossing under one policy."""

    status: str
    parent_id: int | None = None
    unwrapped_axis: float = np.nan
    signed_step: float = np.nan
    cumulative_signed: float = np.nan
    cumulative_absolute: float = np.nan
    distance: float = np.nan
    axial_delta: float = np.nan
    prediction_residual: float = np.nan
    cost: float = np.nan


def axial_difference(outer: float, inner: float) -> float:
    """Return the seam-safe signed axial difference in radians."""
    difference = outer - inner
    return float(
        0.5
        * np.arctan2(
            np.sin(2.0 * difference),
            np.cos(2.0 * difference),
        )
    )


def extract_skeleton_ring_crossings(
    profiles: dict,
    phi: np.ndarray,
    coherence: np.ndarray,
    selector: np.ndarray,
    radii: np.ndarray,
    *,
    crossing_half_width: float = 1.5,
) -> tuple[list[RingCrossing], np.ndarray]:
    """Collect connected skeleton intersections with each Sholl circle.

    The detected object is skeletonized before applying the inoculum exclusion
    and coherence gate. Pixels within ``crossing_half_width`` of a ring centre
    are grouped by 8-connectivity. Every connected group contributes one point
    whose orientation is a coherence-weighted doubled-angle mean.

    Args:
        profiles: Full-length object metadata and distance map.
        phi: Image-derived local gradient-normal orientation field.
        coherence: Image-derived local orientation coherence.
        selector: Full-length detected-object selector outside the inoculum.
        radii: Sholl-circle radii in pixels.
        crossing_half_width: Half-width of the rasterized ring intersection.

    Returns:
        Crossing records and the reliable one-pixel measurement skeleton.

    Raises:
        ValueError: If shapes differ or the crossing half-width is invalid.
    """
    transform = literal_skeleton_ring_crossings(
        profiles["obj_mask"],
        phi + _FIBER_AXIS_OFFSET,
        coherence,
        profiles["dist_map"],
        profiles["centre"],
        radii,
        selector=selector,
        minimum_coherence=_RADIAL_RELATIVE_MIN_COHERENCE,
        crossing_half_width=crossing_half_width,
        minimum_crossing_resultant=0.15,
    )
    return list(transform.crossings), transform.reliable_skeleton


def annular_corridor_labels(
    points: list[RingCrossing],
    reliable_skeleton: np.ndarray,
    dist_map: np.ndarray,
    radii: np.ndarray,
    *,
    crossing_half_width: float = 1.5,
) -> dict[tuple[int, int], int]:
    """Label skeleton connectivity between every adjacent ring pair.

    Args:
        points: Extracted skeleton-ring crossings.
        reliable_skeleton: Coherence-eligible one-pixel skeleton.
        dist_map: Pixel distance from the inoculum centre.
        radii: Sholl-circle radii in pixels.
        crossing_half_width: Radial margin retained around each circle.

    Returns:
        Mapping ``(outer_ring, point_id) -> corridor component label`` for
        crossings on the inner and outer boundary of that ring pair.
    """
    labels_by_pair: dict[tuple[int, int], int] = {}
    by_ring: dict[int, list[RingCrossing]] = {}
    for point in points:
        by_ring.setdefault(point.ring_index, []).append(point)
    structure = np.ones((3, 3), dtype=np.uint8)
    for outer_ring in range(1, len(radii)):
        lower = radii[outer_ring - 1] - crossing_half_width
        upper = radii[outer_ring] + crossing_half_width
        corridor = (
            reliable_skeleton & (dist_map >= lower) & (dist_map <= upper)
        )
        components, _count = connected_components(
            corridor, structure=structure
        )
        for point in [
            *by_ring.get(outer_ring - 1, []),
            *by_ring.get(outer_ring, []),
        ]:
            labels_by_pair[(outer_ring, point.point_id)] = int(
                components[point.anchor_row, point.anchor_col]
            )
    return labels_by_pair


def _lift_against_prediction(
    wrapped_outer: float,
    parent: PointState,
) -> tuple[float, float, float, bool]:
    """Lift one axial angle by 180 degrees against the parent's trend."""
    previous_step = (
        parent.signed_step if np.isfinite(parent.signed_step) else 0.0
    )
    prediction = parent.unwrapped_axis + previous_step
    centre_lift = int(round((prediction - wrapped_outer) / np.pi))
    candidates = wrapped_outer + np.pi * (
        centre_lift + np.arange(-1, 2, dtype=float)
    )
    errors = np.abs(candidates - prediction)
    order = np.argsort(errors, kind="stable")
    ambiguous = bool(
        errors.size > 1
        and np.isclose(
            errors[order[0]], errors[order[1]], atol=1e-12, rtol=0.0
        )
    )
    chosen = float(candidates[order[0]])
    return (
        chosen,
        chosen - parent.unwrapped_axis,
        chosen - prediction,
        ambiguous,
    )


def build_candidate_edges(
    previous_points: list[RingCrossing],
    current_points: list[RingCrossing],
    states: dict[int, PointState],
    corridor_labels: dict[tuple[int, int], int],
    *,
    max_distance: float,
    max_axial_delta: float,
    max_step: float,
    max_prediction_residual: float,
) -> list[CandidateEdge]:
    """Build the common hard-gated evidence graph for one ring pair."""
    edges: list[CandidateEdge] = []
    for outer in current_points:
        for inner in previous_points:
            parent = states.get(inner.point_id)
            if parent is None or not np.isfinite(parent.cumulative_signed):
                continue
            corridor_key = (outer.ring_index, outer.point_id)
            outer_component = corridor_labels.get(corridor_key, 0)
            inner_component = corridor_labels.get(
                (outer.ring_index, inner.point_id),
                0,
            )
            if outer_component == 0 or outer_component != inner_component:
                continue
            distance = float(
                np.hypot(outer.row - inner.row, outer.col - inner.col)
            )
            if distance > max_distance:
                continue
            axial_delta = axial_difference(outer.fiber_axis, inner.fiber_axis)
            if abs(axial_delta) > max_axial_delta:
                continue
            lifted, step, prediction_residual, lift_ambiguous = (
                _lift_against_prediction(outer.fiber_axis, parent)
            )
            if (
                lift_ambiguous
                or abs(step) > max_step
                or abs(prediction_residual) > max_prediction_residual
            ):
                continue
            cost = float(
                np.square(distance / max_distance)
                + np.square(abs(axial_delta) / max_axial_delta)
            )
            edges.append(
                CandidateEdge(
                    inner_id=inner.point_id,
                    outer_id=outer.point_id,
                    distance=distance,
                    axial_delta=axial_delta,
                    unwrapped_outer=lifted,
                    signed_step=step,
                    prediction_residual=prediction_residual,
                    cost=cost,
                )
            )
    return edges


def _decisive_local_best(
    edges: list[CandidateEdge],
    *,
    key: Literal["inner", "outer"],
    ambiguity_margin: float,
) -> tuple[dict[int, CandidateEdge], set[int]]:
    """Return decisive minimum-cost edges grouped by one endpoint."""
    grouped: dict[int, list[CandidateEdge]] = {}
    for edge in edges:
        endpoint = edge.inner_id if key == "inner" else edge.outer_id
        grouped.setdefault(endpoint, []).append(edge)
    best: dict[int, CandidateEdge] = {}
    ambiguous: set[int] = set()
    for endpoint, candidates in grouped.items():
        ordered = sorted(
            candidates,
            key=lambda edge: (edge.cost, edge.inner_id, edge.outer_id),
        )
        if (
            len(ordered) > 1
            and ordered[1].cost - ordered[0].cost <= ambiguity_margin
        ):
            ambiguous.add(endpoint)
            continue
        best[endpoint] = ordered[0]
    return best, ambiguous


def _select_policy_edges(
    edges: list[CandidateEdge],
    previous_points: list[RingCrossing],
    current_points: list[RingCrossing],
    policy: str,
    *,
    ambiguity_margin: float,
    unmatched_cost: float,
) -> tuple[dict[int, CandidateEdge], dict[int, str]]:
    """Select one parent per current point under one matching policy."""
    statuses = {point.point_id: "no_candidate" for point in current_points}
    outer_best, outer_ambiguous = _decisive_local_best(
        edges,
        key="outer",
        ambiguity_margin=ambiguity_margin,
    )
    for point_id in outer_ambiguous:
        statuses[point_id] = "local_ambiguity"

    if policy == "independent_many_to_one":
        selected = dict(outer_best)
    elif policy == "reciprocal_one_to_one":
        inner_best, inner_ambiguous = _decisive_local_best(
            edges,
            key="inner",
            ambiguity_margin=ambiguity_margin,
        )
        selected = {}
        for outer_id, edge in outer_best.items():
            if edge.inner_id in inner_ambiguous:
                statuses[outer_id] = "inner_ambiguity"
            elif inner_best.get(edge.inner_id) == edge:
                selected[outer_id] = edge
            else:
                statuses[outer_id] = "claimed_by_other"
    elif policy == "global_one_to_one":
        selected, globally_ambiguous = _select_global_edges(
            edges,
            previous_points,
            current_points,
            ambiguity_margin=ambiguity_margin,
            unmatched_cost=unmatched_cost,
        )
        edge_outer_ids = {edge.outer_id for edge in edges}
        for point in current_points:
            if point.point_id in globally_ambiguous:
                statuses[point.point_id] = "global_ambiguity"
            elif (
                point.point_id in edge_outer_ids
                and point.point_id not in selected
            ):
                statuses[point.point_id] = "global_unmatched"
    else:
        raise ValueError(f"Unknown matching policy: {policy}")

    for outer_id in selected:
        statuses[outer_id] = "matched"
    return selected, statuses


def _select_global_edges(
    edges: list[CandidateEdge],
    previous_points: list[RingCrossing],
    current_points: list[RingCrossing],
    *,
    ambiguity_margin: float,
    unmatched_cost: float,
) -> tuple[dict[int, CandidateEdge], set[int]]:
    """Return global one-to-one edges and ambiguous outer point IDs."""
    if not previous_points or not current_points:
        return {}, set()
    inner_ids = [point.point_id for point in previous_points]
    outer_ids = [point.point_id for point in current_points]
    inner_index = {point_id: index for index, point_id in enumerate(inner_ids)}
    outer_index = {point_id: index for index, point_id in enumerate(outer_ids)}
    n_outer = len(outer_ids)
    n_inner = len(inner_ids)
    forbidden = 1e6
    costs = np.full((n_outer, n_inner + n_outer), forbidden, dtype=float)
    edge_lookup: dict[tuple[int, int], CandidateEdge] = {}
    for edge in edges:
        row = outer_index[edge.outer_id]
        col = inner_index[edge.inner_id]
        if edge.cost < costs[row, col]:
            costs[row, col] = edge.cost
            edge_lookup[(row, col)] = edge
    for row in range(n_outer):
        costs[row, n_inner + row] = unmatched_cost
    rows, cols = linear_sum_assignment(costs)
    base_total = float(costs[rows, cols].sum())
    tentative: list[tuple[int, int, CandidateEdge]] = []
    for row, col in zip(rows, cols):
        edge = edge_lookup.get((int(row), int(col)))
        if edge is not None and edge.cost < unmatched_cost:
            tentative.append((int(row), int(col), edge))

    selected: dict[int, CandidateEdge] = {}
    ambiguous: set[int] = set()
    for row, col, edge in tentative:
        alternative_costs = costs.copy()
        alternative_costs[row, col] = forbidden
        alt_rows, alt_cols = linear_sum_assignment(alternative_costs)
        alternative_total = float(alternative_costs[alt_rows, alt_cols].sum())
        if alternative_total - base_total <= ambiguity_margin:
            ambiguous.add(edge.outer_id)
            continue
        selected[edge.outer_id] = edge
    return selected, ambiguous


def collect_policy_states(
    points: list[RingCrossing],
    policy: str,
    corridor_labels: dict[tuple[int, int], int],
    *,
    ring_width: float,
    max_axial_delta: float = np.deg2rad(20.0),
    max_radial_tilt: float = np.deg2rad(75.0),
    max_step: float = np.deg2rad(60.0),
    max_prediction_residual: float = np.deg2rad(30.0),
    ambiguity_margin: float = 0.05,
    unmatched_cost: float = 1.0,
) -> dict[int, PointState]:
    """Collect strict previous-ring inheritance states for one policy.

    Args:
        points: Literal skeleton-ring crossings.
        policy: One of the three configured point-matching policies.
        corridor_labels: Adjacent-ring skeleton connectivity labels.
        ring_width: Sholl ring separation in pixels.
        max_axial_delta: Hard axial-orientation similarity gate.
        max_radial_tilt: Radial tilt used to derive the spatial reach gate.
        max_step: Maximum accepted unwrapped adjacent orientation change.
        max_prediction_residual: Maximum deviation from the previous step.
        ambiguity_margin: Minimum normalized-cost separation for decisiveness.
        unmatched_cost: Global-assignment cost of leaving a point unmatched.

    Returns:
        Mapping from point ID to strict accumulation state.

    Raises:
        ValueError: If policy parameters are invalid.
    """
    if ring_width <= 0.0:
        raise ValueError("ring_width must be > 0")
    if not (0.0 < max_radial_tilt < 0.5 * np.pi):
        raise ValueError("max_radial_tilt must be in (0, pi/2)")
    max_distance = ring_width / np.cos(max_radial_tilt)
    by_ring: dict[int, list[RingCrossing]] = {}
    for point in points:
        by_ring.setdefault(point.ring_index, []).append(point)
    if not by_ring:
        return {}
    max_ring = max(by_ring)
    states: dict[int, PointState] = {}
    for point in by_ring.get(0, []):
        states[point.point_id] = PointState(
            status="seed",
            unwrapped_axis=point.fiber_axis,
            signed_step=0.0,
            cumulative_signed=0.0,
            cumulative_absolute=0.0,
        )
    for ring in range(1, max_ring + 1):
        previous_points = by_ring.get(ring - 1, [])
        current_points = by_ring.get(ring, [])
        edges = build_candidate_edges(
            previous_points,
            current_points,
            states,
            corridor_labels,
            max_distance=max_distance,
            max_axial_delta=max_axial_delta,
            max_step=max_step,
            max_prediction_residual=max_prediction_residual,
        )
        selected, statuses = _select_policy_edges(
            edges,
            previous_points,
            current_points,
            policy,
            ambiguity_margin=ambiguity_margin,
            unmatched_cost=unmatched_cost,
        )
        for point in current_points:
            edge = selected.get(point.point_id)
            if edge is None:
                states[point.point_id] = PointState(
                    status=statuses[point.point_id]
                )
                continue
            parent = states[edge.inner_id]
            states[point.point_id] = PointState(
                status="matched",
                parent_id=edge.inner_id,
                unwrapped_axis=edge.unwrapped_outer,
                signed_step=edge.signed_step,
                cumulative_signed=(
                    parent.cumulative_signed + edge.signed_step
                ),
                cumulative_absolute=(
                    parent.cumulative_absolute + abs(edge.signed_step)
                ),
                distance=edge.distance,
                axial_delta=edge.axial_delta,
                prediction_residual=edge.prediction_residual,
                cost=edge.cost,
            )
    return states


def render_point_policy_comparison(
    colony: str,
    profiles: dict,
    points: list[RingCrossing],
    reliable_skeleton: np.ndarray,
    states_by_policy: dict[str, dict[int, PointState]],
) -> Path:
    """Render raw crossing orientations and three inheritance policies."""
    base = profiles["base"]
    finite = base[np.isfinite(base)]
    low, high = np.percentile(finite, (1.0, 99.8))
    point_lookup = {point.point_id: point for point in points}
    figure, axes = plt.subplots(
        2, 2, figsize=(14, 13), constrained_layout=True
    )
    cmap = plt.get_cmap("Spectral").copy()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))

    for axis in axes.flat:
        axis.imshow(base, cmap="gray", vmin=low, vmax=high)
        axis.imshow(
            np.ma.masked_where(~reliable_skeleton, reliable_skeleton),
            cmap="gray",
            vmin=0,
            vmax=1,
            alpha=0.22,
        )
        draw_rings(axis, profiles)
        axis.set_xlim(-0.5, base.shape[1] - 0.5)
        axis.set_ylim(base.shape[0] - 0.5, -0.5)
        axis.set_aspect("equal")
        axis.set_axis_off()

    extraction_axis = axes[0, 0]
    glyphs: list[list[tuple[float, float]]] = []
    glyph_values: list[float] = []
    half_length = 3.0
    for point in points:
        dx = half_length * np.cos(point.fiber_axis)
        dy = half_length * np.sin(point.fiber_axis)
        glyphs.append(
            [
                (point.col - dx, point.row - dy),
                (point.col + dx, point.row + dy),
            ]
        )
        glyph_values.append(np.degrees(point.radial_tilt))
    extraction_collection = LineCollection(
        glyphs,
        cmap=cmap,
        norm=SPECTRAL_NORM,
        linewidths=1.5,
        alpha=0.95,
    )
    extraction_collection.set_array(np.asarray(glyph_values))
    extraction_axis.add_collection(extraction_collection)
    extraction_axis.set_title(
        "Literal skeleton-ring crossings\n"
        f"{len(points)} crossings; glyph direction = absolute fiber axis; "
        "color = radial-relative tilt",
        fontsize=10,
    )

    policy_axes = {
        "reciprocal_one_to_one": axes[0, 1],
        "independent_many_to_one": axes[1, 0],
        "global_one_to_one": axes[1, 1],
    }
    policy_titles = {
        "reciprocal_one_to_one": "Reciprocal one-to-one (recommended)",
        "independent_many_to_one": "Independent many-to-one",
        "global_one_to_one": "Global one-to-one with unmatched option",
    }
    scalar_mappable = plt.cm.ScalarMappable(norm=SPECTRAL_NORM, cmap=cmap)
    for policy, axis in policy_axes.items():
        states = states_by_policy[policy]
        segments: list[list[tuple[float, float]]] = []
        values: list[float] = []
        supported_rows: list[float] = []
        supported_cols: list[float] = []
        unsupported_rows: list[float] = []
        unsupported_cols: list[float] = []
        for point in points:
            state = states.get(point.point_id)
            if state is None or not np.isfinite(state.cumulative_signed):
                unsupported_rows.append(point.row)
                unsupported_cols.append(point.col)
                continue
            supported_rows.append(point.row)
            supported_cols.append(point.col)
            if state.parent_id is None:
                continue
            parent = point_lookup[state.parent_id]
            segments.append([(parent.col, parent.row), (point.col, point.row)])
            values.append(np.degrees(state.cumulative_signed))
        if segments:
            collection = LineCollection(
                segments,
                cmap=cmap,
                norm=SPECTRAL_NORM,
                linewidths=2.0,
                alpha=0.95,
            )
            collection.set_array(np.asarray(values))
            axis.add_collection(collection)
        axis.scatter(
            supported_cols,
            supported_rows,
            s=8,
            facecolors="white",
            edgecolors="black",
            linewidths=0.25,
            alpha=0.9,
        )
        axis.scatter(
            unsupported_cols,
            unsupported_rows,
            s=7,
            marker="x",
            color="#888888",
            linewidths=0.45,
            alpha=0.55,
        )
        finite_values = np.asarray(values, dtype=float)
        raw_peak = (
            float(np.max(np.abs(finite_values)))
            if finite_values.size
            else np.nan
        )
        axis.set_title(
            f"{policy_titles[policy]}\n"
            f"accepted edges {len(segments)}; supported points "
            f"{len(supported_rows)}/{len(points)}; raw peak {raw_peak:.1f} degrees",
            fontsize=10,
        )

    colorbar = figure.colorbar(
        scalar_mappable,
        ax=axes,
        orientation="horizontal",
        fraction=0.035,
        pad=0.025,
        ticks=(-180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0),
    )
    colorbar.set_label(
        "Signed degrees, fixed -180 to +180; extraction panel shows local "
        "radial-relative tilt, policy panels show cumulative fiber-axis change"
    )
    figure.suptitle(
        f"{colony}: previous-ring orientation inheritance on skeleton crossings\n"
        "8 px Sholl spacing; 20 degree hard axial gate; strict no-gap and no-restart",
        fontsize=14,
    )
    output = OUTPUT_DIR / f"twok_{colony}_point_matched_orientation_2x2.png"
    figure.savefig(
        output,
        dpi=190,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(figure)
    return output


def _point_rows(
    colony: str,
    points: list[RingCrossing],
) -> list[dict[str, float | int | str]]:
    """Return raw crossing rows for CSV export."""
    return [
        {
            "Colony": colony,
            "PointID": point.point_id,
            "Ring": point.ring_index,
            "RadiusPx": point.radius,
            "Row": point.row,
            "Col": point.col,
            "AnchorRow": point.anchor_row,
            "AnchorCol": point.anchor_col,
            "FiberAxisDeg": np.degrees(point.fiber_axis),
            "RadialRelativeTiltDeg": np.degrees(point.radial_tilt),
            "MeanCoherence": point.coherence,
            "AxialResultant": point.resultant,
            "SkeletonPixelCount": point.pixel_count,
        }
        for point in points
    ]


def _state_rows(
    colony: str,
    policy: str,
    points: list[RingCrossing],
    states: dict[int, PointState],
) -> list[dict[str, float | int | str]]:
    """Return auditable point-state rows for one matching policy."""
    rows: list[dict[str, float | int | str]] = []
    for point in points:
        state = states.get(point.point_id, PointState(status="unprocessed"))
        rows.append(
            {
                "Colony": colony,
                "Policy": policy,
                "PointID": point.point_id,
                "Ring": point.ring_index,
                "Status": state.status,
                "ParentID": state.parent_id,
                "DistancePx": state.distance,
                "AxialDeltaDeg": np.degrees(state.axial_delta),
                "PredictionResidualDeg": np.degrees(state.prediction_residual),
                "SignedStepDeg": np.degrees(state.signed_step),
                "UnwrappedFiberAxisDeg": np.degrees(state.unwrapped_axis),
                "CumulativeSignedDeg": np.degrees(state.cumulative_signed),
                "CumulativeAbsoluteDeg": np.degrees(state.cumulative_absolute),
                "NormalizedCost": state.cost,
            }
        )
    return rows


def render_all_point_matched_orientation() -> None:
    """Render and export three point-level inheritance policies."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detected = load_point_matching_detection()
    crossing_rows: list[dict[str, float | int | str]] = []
    state_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for colony, label in COLONIES:
        section = isolated_global_crop(
            detected,
            label,
            label_centroid(detected, label),
        )
        operation = MeasureOrientationZones(
            radial_ring_width=8.0,
            long_range_lag=32.0,
            quiver_block=24,
        )
        initial_profiles = extract_profiles(section, operation)
        profiles, phi, coherence, selector = extract_full_length_ring_fields(
            section,
            operation,
            initial_profiles,
        )
        n_rings = int(
            np.floor(
                (profiles["outer_radius"] - profiles["inner_radius"])
                / profiles["ring_width"]
                + 1e-12
            )
        )
        radii = (
            profiles["inner_radius"]
            + (np.arange(n_rings, dtype=float) + 0.5) * profiles["ring_width"]
        )
        points, reliable_skeleton = extract_skeleton_ring_crossings(
            profiles,
            phi,
            coherence,
            selector,
            radii,
        )
        corridor_labels = annular_corridor_labels(
            points,
            reliable_skeleton,
            profiles["dist_map"],
            radii,
        )
        states_by_policy = {
            policy: collect_policy_states(
                points,
                policy,
                corridor_labels,
                ring_width=profiles["ring_width"],
            )
            for policy in POLICIES
        }
        output = render_point_policy_comparison(
            colony,
            profiles,
            points,
            reliable_skeleton,
            states_by_policy,
        )
        report(str(output))
        crossing_rows.extend(_point_rows(colony, points))
        for policy, states in states_by_policy.items():
            state_rows.extend(_state_rows(colony, policy, points, states))
            supported = [
                state
                for state in states.values()
                if np.isfinite(state.cumulative_signed)
            ]
            matched = [
                state for state in supported if state.parent_id is not None
            ]
            summary_rows.append(
                {
                    "Colony": colony,
                    "Policy": policy,
                    "Crossings": len(points),
                    "SupportedPoints": len(supported),
                    "AcceptedEdges": len(matched),
                    "SupportFraction": (
                        len(supported) / len(points) if points else 0.0
                    ),
                    "RawCumulativePeakDeg": (
                        float(
                            np.max(
                                np.abs(
                                    np.degrees(
                                        [
                                            state.cumulative_signed
                                            for state in matched
                                        ]
                                    )
                                )
                            )
                        )
                        if matched
                        else np.nan
                    ),
                }
            )

    crossings_path = OUTPUT_DIR / "twok_point_ring_crossings.csv"
    states_path = OUTPUT_DIR / "twok_point_matched_orientation_states.csv"
    summary_path = OUTPUT_DIR / "twok_point_matched_orientation_summary.csv"
    pd.DataFrame(crossing_rows).to_csv(crossings_path, index=False)
    pd.DataFrame(state_rows).to_csv(states_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    report(str(crossings_path))
    report(str(states_path))
    report(str(summary_path))
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    render_all_point_matched_orientation()
