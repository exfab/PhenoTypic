from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from render_matched_ring_comparison import (  # noqa: E402
    extract_profiles,
    label_centroid,
    profile_metrics,
)
from render_twok_reconnected_orientation import (  # noqa: E402
    isolated_global_crop,
    load_twok_detection,
    report,
)
from phenotypic.measure import MeasureOrientationZones  # noqa: E402
from phenotypic.measure._measure_orientation_zones import (  # noqa: E402
    _FIBER_AXIS_OFFSET,
    _RADIAL_RELATIVE_MIN_COHERENCE,
    radial_ring_sector_field,
    signed_radial_relative_field,
    zone_selector,
)
from phenotypic.util._matched_ring_rotation import (  # noqa: E402
    matched_ring_cumulative_rotation_profile,
    matched_tracks_to_ring_sector_values,
)
from phenotypic.util._nematic_bend import fiber_bend_field  # noqa: E402


OUTPUT_DIR = SCRIPT_DIR.parent / "artifacts"
COLONIES = (("R3C4", 24), ("R4C6", 36))
SPECTRAL_NORM = Normalize(vmin=-180.0, vmax=180.0)


def axial_difference(outer: np.ndarray | float, inner: float) -> np.ndarray:
    """Return seam-safe axial differences in [-pi/2, pi/2]."""
    difference = np.asarray(outer, dtype=float) - inner
    return 0.5 * np.arctan2(
        np.sin(2.0 * difference),
        np.cos(2.0 * difference),
    )


def circular_difference(outer: np.ndarray, inner: float) -> np.ndarray:
    """Return signed circular differences in [-pi, pi]."""
    return np.arctan2(np.sin(outer - inner), np.cos(outer - inner))


@dataclass(frozen=True)
class TangentialGraphResult:
    """Output from the bounded same-ring detour prototype."""

    cumulative: np.ndarray
    paths: np.ndarray
    routes: dict[tuple[int, int], tuple[tuple[int, int], ...]]


def bounded_tangential_ring_profile(
    radii: np.ndarray,
    orientation: np.ndarray,
    resultant: np.ndarray,
    *,
    max_outward_shift: int = 2,
    max_tangential_steps: int = 2,
    max_abs_radial_tilt: float = np.deg2rad(75.0),
    max_axis_mismatch: float = np.deg2rad(35.0),
    reliability_weight: float = 0.25,
    tangential_step_penalty: float = 0.20,
) -> TangentialGraphResult:
    """Match outward rings after an optional bounded same-ring walk.

    The same-ring walk is only a rescue route. It is considered together with
    the direct route, cannot reverse direction, cannot revisit a sector, and is
    bounded to ``max_tangential_steps``. Rotation along every observed lateral
    and outward edge is accumulated.
    """
    radii = np.asarray(radii, dtype=float)
    orientation = np.asarray(orientation, dtype=float)
    resultant = np.asarray(resultant, dtype=float)
    n_rings, n_sectors = orientation.shape
    cumulative = np.full_like(orientation, np.nan)
    paths = np.full((n_rings, n_sectors), -1, dtype=int)
    routes: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    reliable = np.isfinite(orientation) & np.isfinite(resultant)
    sector_width = 2.0 * np.pi / n_sectors
    sector_angles = (np.arange(n_sectors) + 0.5) * sector_width
    outward_offsets = np.arange(-max_outward_shift, max_outward_shift + 1)

    for seed in range(n_sectors):
        starts = np.flatnonzero(reliable[:, seed])
        if starts.size == 0:
            continue
        ring = int(starts[0])
        sector = seed
        cumulative[ring, seed] = 0.0
        paths[ring, seed] = sector
        while ring + 1 < n_rings:
            direct_candidates: list[
                tuple[float, int, float, tuple[tuple[int, int], ...]]
            ] = []
            tangential_candidates: list[
                tuple[float, int, float, tuple[tuple[int, int], ...]]
            ] = []
            for direction in (0, -1, 1):
                lateral_options = (0,) if direction == 0 else range(
                    1, max_tangential_steps + 1
                )
                lateral_rotation = 0.0
                previous_sector = sector
                route_nodes: list[tuple[int, int]] = [(ring, sector)]
                for lateral_count in lateral_options:
                    if lateral_count:
                        lateral_sector = (
                            sector + direction * lateral_count
                        ) % n_sectors
                        if not reliable[ring, lateral_sector]:
                            break
                        step = float(
                            axial_difference(
                                orientation[ring, lateral_sector],
                                orientation[ring, previous_sector],
                            )
                        )
                        if np.isclose(
                            abs(step),
                            np.pi / 2.0,
                            atol=1e-12,
                            rtol=0.0,
                        ):
                            break
                        edge_bearing = (
                            sector_angles[previous_sector]
                            + direction * (np.pi / 2.0 + sector_width / 2.0)
                        )
                        previous_mismatch = abs(
                            float(
                                axial_difference(
                                    orientation[ring, previous_sector],
                                    edge_bearing,
                                )
                            )
                        )
                        next_mismatch = abs(
                            float(
                                axial_difference(
                                    orientation[ring, lateral_sector],
                                    edge_bearing,
                                )
                            )
                        )
                        if max(previous_mismatch, next_mismatch) > max_axis_mismatch:
                            break
                        lateral_rotation += step
                        previous_sector = lateral_sector
                        route_nodes.append((ring, lateral_sector))
                    else:
                        lateral_sector = sector

                    current_orientation = orientation[ring, lateral_sector]
                    current_alpha = sector_angles[lateral_sector]
                    radial_tilt = float(
                        axial_difference(current_orientation, current_alpha)
                    )
                    if abs(radial_tilt) > max_abs_radial_tilt:
                        continue
                    radial_ratio = radii[ring + 1] / radii[ring]
                    predicted_step = np.tan(radial_tilt) * np.log(radial_ratio)
                    maximum_step = (
                        max_outward_shift + 0.5
                    ) * sector_width
                    if abs(predicted_step) > maximum_step:
                        continue
                    outward_sectors = np.unique(
                        np.mod(lateral_sector + outward_offsets, n_sectors)
                    )
                    outward_sectors = outward_sectors[
                        reliable[ring + 1, outward_sectors]
                    ]
                    if outward_sectors.size == 0:
                        continue
                    outward_rotation = axial_difference(
                        orientation[ring + 1, outward_sectors],
                        current_orientation,
                    )
                    unambiguous = ~np.isclose(
                        np.abs(outward_rotation),
                        np.pi / 2.0,
                        atol=1e-12,
                        rtol=0.0,
                    )
                    if not unambiguous.any():
                        continue
                    outward_sectors = outward_sectors[unambiguous]
                    outward_rotation = outward_rotation[unambiguous]
                    predicted_alpha = current_alpha + predicted_step
                    residual = circular_difference(
                        sector_angles[outward_sectors], predicted_alpha
                    )
                    cost = np.square(residual / sector_width)
                    cost += np.square(outward_rotation / sector_width)
                    cost += reliability_weight * (
                        1.0 - resultant[ring + 1, outward_sectors]
                    )
                    cost += tangential_step_penalty * lateral_count
                    choice = int(np.argmin(cost))
                    next_sector = int(outward_sectors[choice])
                    total_rotation = lateral_rotation + float(
                        outward_rotation[choice]
                    )
                    route = tuple(
                        [*route_nodes, (ring + 1, next_sector)]
                    )
                    target = (
                        direct_candidates
                        if lateral_count == 0
                        else tangential_candidates
                    )
                    target.append(
                        (float(cost[choice]), next_sector, total_rotation, route)
                    )
            # A same-ring walk is a rescue for a failed radial continuation.
            # It may not replace a defensible direct edge merely because its
            # local score is smaller.
            candidates = direct_candidates or tangential_candidates
            if not candidates:
                break
            _cost, next_sector, step_rotation, route = min(
                candidates, key=lambda item: item[0]
            )
            cumulative[ring + 1, seed] = cumulative[ring, seed] + step_rotation
            paths[ring + 1, seed] = next_sector
            routes[(seed, ring + 1)] = route
            ring += 1
            sector = next_sector
    return TangentialGraphResult(cumulative, paths, routes)


@dataclass(frozen=True)
class Streamline:
    """One axial-director integral curve and its accumulated rotation."""

    seed_sector: int
    points: np.ndarray
    cumulative_degrees: np.ndarray
    ambiguous_launch: bool


def smoothed_director_q(
    fiber_axis: np.ndarray,
    coherence: np.ndarray,
    selector: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mask-normalized doubled-angle components and reliability."""
    valid = selector & np.isfinite(fiber_axis) & np.isfinite(coherence)
    weight = np.where(valid, coherence, 0.0)
    smooth_weight = gaussian_filter(weight, sigma, mode="constant", cval=0.0)
    q_cos = gaussian_filter(
        weight * np.cos(2.0 * np.where(valid, fiber_axis, 0.0)),
        sigma,
        mode="constant",
        cval=0.0,
    )
    q_sin = gaussian_filter(
        weight * np.sin(2.0 * np.where(valid, fiber_axis, 0.0)),
        sigma,
        mode="constant",
        cval=0.0,
    )
    resultant = np.divide(
        np.hypot(q_cos, q_sin),
        smooth_weight,
        out=np.zeros_like(q_cos),
        where=smooth_weight > 1e-12,
    )
    return q_cos, q_sin, resultant


def sample_scalar(field: np.ndarray, point: np.ndarray) -> float:
    """Bilinearly sample one image field at (row, column)."""
    return float(
        map_coordinates(
            field,
            np.asarray([[point[0]], [point[1]]]),
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )[0]
    )


def sample_director(
    q_cos: np.ndarray,
    q_sin: np.ndarray,
    resultant: np.ndarray,
    point: np.ndarray,
    min_resultant: float,
) -> float | None:
    """Bilinearly sample an axial angle through doubled-angle components."""
    c = sample_scalar(q_cos, point)
    s = sample_scalar(q_sin, point)
    r = sample_scalar(resultant, point)
    if not np.isfinite(c + s + r) or r < min_resultant or np.hypot(c, s) <= 1e-12:
        return None
    return 0.5 * float(np.arctan2(s, c))


def integrate_one_direction(
    seed_sector: int,
    seed: np.ndarray,
    sign: float,
    q_cos: np.ndarray,
    q_sin: np.ndarray,
    resultant: np.ndarray,
    active_distance: np.ndarray,
    centre: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    *,
    step_size: float = 2.0,
    max_steps: int = 900,
    min_resultant: float = 0.15,
    max_active_distance: float = 4.0,
) -> Streamline:
    """Integrate one signed hypothesis of a nematic director curve."""
    points = [np.asarray(seed, dtype=float)]
    rotations = [0.0]
    previous_vector: np.ndarray | None = None
    previous_axis: float | None = None
    centre_array = np.asarray(centre, dtype=float)
    initial_axis = sample_director(
        q_cos, q_sin, resultant, points[0], min_resultant
    )
    if initial_axis is None:
        return Streamline(
            seed_sector,
            np.asarray(points),
            np.asarray(rotations),
            False,
        )
    radial = points[0] - centre_array
    radial /= max(np.linalg.norm(radial), 1e-12)
    initial_vector = np.asarray(
        [np.sin(initial_axis), np.cos(initial_axis)], dtype=float
    )
    ambiguous = abs(float(np.dot(initial_vector, radial))) < 0.10

    for _step in range(max_steps):
        current = points[-1]
        axis = sample_director(
            q_cos, q_sin, resultant, current, min_resultant
        )
        if axis is None:
            break
        vector = np.asarray([np.sin(axis), np.cos(axis)], dtype=float)
        if previous_vector is None:
            vector *= sign
        elif float(np.dot(vector, previous_vector)) < 0.0:
            vector *= -1.0
        midpoint = current + 0.5 * step_size * vector
        midpoint_axis = sample_director(
            q_cos, q_sin, resultant, midpoint, min_resultant
        )
        if midpoint_axis is None:
            break
        midpoint_vector = np.asarray(
            [np.sin(midpoint_axis), np.cos(midpoint_axis)], dtype=float
        )
        if float(np.dot(midpoint_vector, vector)) < 0.0:
            midpoint_vector *= -1.0
        next_point = current + step_size * midpoint_vector
        radius = float(np.linalg.norm(next_point - centre_array))
        if radius < inner_radius or radius > outer_radius:
            break
        distance = sample_scalar(active_distance, next_point)
        if not np.isfinite(distance) or distance > max_active_distance:
            break
        if (
            next_point[0] < 0
            or next_point[0] >= q_cos.shape[0] - 1
            or next_point[1] < 0
            or next_point[1] >= q_cos.shape[1] - 1
        ):
            break
        if len(points) > 8:
            old_points = np.asarray(points[:-8])
            if np.min(np.linalg.norm(old_points - next_point, axis=1)) < 1.5:
                break
        next_axis = sample_director(
            q_cos, q_sin, resultant, next_point, min_resultant
        )
        if next_axis is None:
            break
        if previous_axis is None:
            step_rotation = float(axial_difference(next_axis, axis))
        else:
            step_rotation = float(axial_difference(next_axis, previous_axis))
        points.append(next_point)
        rotations.append(rotations[-1] + np.degrees(step_rotation))
        previous_axis = next_axis
        previous_vector = midpoint_vector
    return Streamline(
        seed_sector,
        np.asarray(points),
        np.asarray(rotations),
        ambiguous,
    )


def streamline_hypotheses(
    profiles: dict,
    fiber_axis: np.ndarray,
    coherence: np.ndarray,
    selector: np.ndarray,
    *,
    sigma: float = 8.0,
) -> list[Streamline]:
    """Launch outward-resolved or bidirectional tangent hypotheses."""
    q_cos, q_sin, resultant = smoothed_director_q(
        fiber_axis, coherence, selector, sigma
    )
    active_distance = distance_transform_edt(~selector)
    n_sectors = profiles["fiber_orientation"].shape[1]
    sector_width = 2.0 * np.pi / n_sectors
    streams: list[Streamline] = []
    for sector in range(n_sectors):
        ring_ids = np.flatnonzero(
            np.isfinite(profiles["fiber_orientation"][:, sector])
        )
        if ring_ids.size == 0:
            continue
        ring = int(ring_ids[0])
        angle = (sector + 0.5) * sector_width
        radius = float(profiles["radii"][ring])
        nominal_seed = np.asarray(
            [
                profiles["centre"][0] + radius * np.sin(angle),
                profiles["centre"][1] + radius * np.cos(angle),
            ]
        )
        radial_half_width = 0.5 * float(profiles["ring_width"])
        cell = (
            selector
            & (np.abs(profiles["dist_map"] - radius) <= radial_half_width)
            & (
                np.abs(circular_difference(profiles["polar_angle"], angle))
                <= 0.5 * sector_width
            )
        )
        cell_rows, cell_cols = np.nonzero(cell)
        if cell_rows.size == 0:
            continue
        distances = np.square(cell_rows - nominal_seed[0]) + np.square(
            cell_cols - nominal_seed[1]
        )
        closest = int(np.argmin(distances))
        seed = np.asarray(
            [float(cell_rows[closest]), float(cell_cols[closest])]
        )
        axis = sample_director(q_cos, q_sin, resultant, seed, 0.15)
        if axis is None:
            continue
        director = np.asarray([np.sin(axis), np.cos(axis)])
        radial = seed - np.asarray(profiles["centre"])
        radial /= max(np.linalg.norm(radial), 1e-12)
        radial_projection = float(np.dot(director, radial))
        if abs(radial_projection) < 0.10:
            signs = (-1.0, 1.0)
        else:
            signs = (1.0 if radial_projection > 0.0 else -1.0,)
        for sign in signs:
            stream = integrate_one_direction(
                sector,
                seed,
                sign,
                q_cos,
                q_sin,
                resultant,
                active_distance,
                profiles["centre"],
                profiles["inner_radius"],
                profiles["outer_radius"],
            )
            if stream.points.shape[0] >= 3:
                streams.append(stream)
    return streams


def extract_pixel_fields(
    section,
    operation: MeasureOrientationZones,
    profiles: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover the object-local field used by the measured profiles."""
    props, label2section = operation._prep(section)
    records = list(operation._iter_object_fields(section, props, label2section))
    if len(records) != 1:
        raise RuntimeError("Expected exactly one object field")
    _prop, seg, obj_mask, phi, coherence, _gradient, dist_map, _centre = records[0]
    selector = zone_selector(
        dist_map,
        float(seg.core_end_radius),
        min(float(seg.sparse_end_radius), float(seg.symmetric_radius)),
        obj_mask,
        "Mask",
    )
    fiber_axis = phi + _FIBER_AXIS_OFFSET
    if phi.shape != profiles["base"].shape:
        raise RuntimeError("Pixel field and display tile shapes differ")
    return phi, fiber_axis, coherence, selector


def draw_rings(axis, profiles: dict) -> None:
    """Draw unobtrusive sampled and zone-boundary rings."""
    centre = profiles["centre"]
    for radius in profiles["radii"]:
        axis.add_patch(
            plt.Circle(
                (centre[1], centre[0]),
                radius,
                fill=False,
                color="white",
                linewidth=0.3,
                alpha=0.16,
            )
        )
    for radius, style, width in (
        (profiles["inner_radius"], "-", 1.0),
        (float(profiles["seg"].dense_end_radius), "--", 0.8),
        (profiles["outer_radius"], "-", 1.0),
    ):
        axis.add_patch(
            plt.Circle(
                (centre[1], centre[0]),
                radius,
                fill=False,
                color="#F0F0F0",
                linestyle=style,
                linewidth=width,
                alpha=0.72,
            )
        )


def draw_ring_paths(
    axis,
    profiles: dict,
    cumulative: np.ndarray,
    paths: np.ndarray,
    routes: dict[tuple[int, int], tuple[tuple[int, int], ...]] | None = None,
) -> None:
    """Draw strict or tangential ring-graph routes."""
    n_sectors = paths.shape[1]
    angles = (np.arange(n_sectors) + 0.5) * (2.0 * np.pi / n_sectors)
    for seed in range(n_sectors):
        supported = np.flatnonzero(
            (paths[:, seed] >= 0) & np.isfinite(cumulative[:, seed])
        )
        if supported.size < 2:
            continue
        if routes is None:
            route_nodes = [
                (int(ring), int(paths[ring, seed])) for ring in supported
            ]
            route_groups = [tuple(route_nodes)]
        else:
            route_groups = [
                routes[(seed, int(ring))]
                for ring in supported[1:]
                if (seed, int(ring)) in routes
            ]
        for group in route_groups:
            if len(group) < 2:
                continue
            xs = [
                profiles["centre"][1]
                + profiles["radii"][ring] * np.cos(angles[sector])
                for ring, sector in group
            ]
            ys = [
                profiles["centre"][0]
                + profiles["radii"][ring] * np.sin(angles[sector])
                for ring, sector in group
            ]
            axis.plot(xs, ys, color="white", linewidth=0.72, alpha=0.52)


def draw_streamlines(
    axis,
    streams: list[Streamline],
) -> tuple[int, int, int, float, float]:
    """Draw streams and summarize one endpoint magnitude per seed location."""
    raw_sample_values: list[float] = []
    endpoint_by_seed: dict[int, list[float]] = {}
    for stream in streams:
        points = stream.points
        segments = np.stack([points[:-1, ::-1], points[1:, ::-1]], axis=1)
        values = stream.cumulative_degrees[1:]
        raw_sample_values.extend(np.abs(values).tolist())
        endpoint_by_seed.setdefault(stream.seed_sector, []).append(
            abs(float(stream.cumulative_degrees[-1]))
        )
        collection = LineCollection(
            segments,
            cmap="Spectral",
            norm=SPECTRAL_NORM,
            linewidths=1.7 if not stream.ambiguous_launch else 1.1,
            alpha=0.78 if not stream.ambiguous_launch else 0.52,
            linestyles="solid" if not stream.ambiguous_launch else "dashed",
        )
        collection.set_array(values)
        axis.add_collection(collection)
    if not endpoint_by_seed:
        return 0, len(streams), 0, np.nan, np.nan
    endpoint_values = np.asarray(
        [np.median(values) for values in endpoint_by_seed.values()],
        dtype=float,
    )
    ambiguous_seeds = sum(
        len(values) > 1 for values in endpoint_by_seed.values()
    )
    return (
        len(endpoint_by_seed),
        len(streams),
        ambiguous_seeds,
        float(np.percentile(endpoint_values, 95.0)),
        float(np.max(raw_sample_values)),
    )


def render_method_comparison(
    colony: str,
    profiles: dict,
    phi: np.ndarray,
    fiber_axis: np.ndarray,
    coherence: np.ndarray,
    selector: np.ndarray,
) -> tuple[Path, list[dict[str, float | str]]]:
    """Render the four tangential-handling prototypes on the measured tile."""
    strict_cumulative, strict_paths = matched_ring_cumulative_rotation_profile(
        profiles["radii"],
        profiles["fiber_orientation"],
        profiles["fiber_resultant"],
        max_sector_shift=2,
    )
    strict_lattice = matched_tracks_to_ring_sector_values(
        strict_cumulative, strict_paths
    )
    tangential = bounded_tangential_ring_profile(
        profiles["radii"],
        profiles["fiber_orientation"],
        profiles["fiber_resultant"],
    )
    tangential_lattice = matched_tracks_to_ring_sector_values(
        tangential.cumulative, tangential.paths
    )
    strict_field = radial_ring_sector_field(
        strict_lattice,
        profiles["polar_angle"],
        profiles["dist_map"],
        profiles["reliable_structure"],
        profiles["inner_radius"],
        profiles["ring_width"],
    )
    tangential_field = radial_ring_sector_field(
        tangential_lattice,
        profiles["polar_angle"],
        profiles["dist_map"],
        profiles["reliable_structure"],
        profiles["inner_radius"],
        profiles["ring_width"],
    )
    streams = streamline_hypotheses(
        profiles, fiber_axis, coherence, selector, sigma=8.0
    )
    bend, bend_resultant = fiber_bend_field(
        phi, coherence, selector, sigma_q=32.0
    )
    signed_tilt, _signed_turn, _turn_mag, _polar = signed_radial_relative_field(
        phi, profiles["centre"], profiles["dist_map"]
    )
    bend_valid = (
        selector
        & np.isfinite(bend)
        & np.isfinite(bend_resultant)
        & (bend_resultant >= 0.15)
        & np.isfinite(coherence)
        & (coherence >= _RADIAL_RELATIVE_MIN_COHERENCE)
    )
    bend_degrees = np.where(bend_valid, np.degrees(bend), np.nan)
    tangential_pixels = bend_valid & (np.abs(np.degrees(signed_tilt)) >= 75.0)

    base = profiles["base"]
    finite_base = base[np.isfinite(base)]
    low, high = np.percentile(finite_base, (1.0, 99.8))
    cmap = plt.get_cmap("Spectral").copy()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    figure, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)
    for axis in axes.flat:
        axis.imshow(base, cmap="gray", vmin=low, vmax=high)
        draw_rings(axis, profiles)
        axis.set_xlim(-0.5, base.shape[1] - 0.5)
        axis.set_ylim(base.shape[0] - 0.5, -0.5)
        axis.set_aspect("equal")
        axis.set_axis_off()

    strict_support, strict_p95, strict_raw = profile_metrics(strict_lattice)
    axes[0, 0].imshow(
        np.ma.masked_invalid(np.degrees(strict_field)),
        cmap=cmap,
        norm=SPECTRAL_NORM,
        alpha=0.82,
    )
    draw_ring_paths(axes[0, 0], profiles, strict_cumulative, strict_paths)
    axes[0, 0].set_title(
        "A. Current radial matched rings\n"
        f"support {strict_support:.1%}; |rotation| p95 {strict_p95:.1f}°\n"
        f"raw peak {strict_raw:.1f}°",
        fontsize=9.5,
    )

    tangent_support, tangent_p95, tangent_raw = profile_metrics(
        tangential_lattice
    )
    axes[0, 1].imshow(
        np.ma.masked_invalid(np.degrees(tangential_field)),
        cmap=cmap,
        norm=SPECTRAL_NORM,
        alpha=0.82,
    )
    draw_ring_paths(
        axes[0, 1],
        profiles,
        tangential.cumulative,
        tangential.paths,
        tangential.routes,
    )
    axes[0, 1].set_title(
        "B. Tangential rescue graph (≤2 aligned lateral cells)\n"
        f"support {tangent_support:.1%}; |rotation| p95 {tangent_p95:.1f}°\n"
        f"raw peak {tangent_raw:.1f}°",
        fontsize=9.5,
    )

    (
        supported_seeds,
        stream_hypotheses,
        ambiguous_seeds,
        streamline_p95,
        streamline_raw,
    ) = draw_streamlines(axes[1, 0], streams)
    axes[1, 0].set_title(
        "C. Cartesian streamlines\n"
        f"Q σ=8 px; ≤4 px from mask; {supported_seeds}/36 seeds; "
        f"{stream_hypotheses} hyp.; {ambiguous_seeds} ambiguous\n"
        f"endpoint |rotation| p95 {streamline_p95:.1f}°; "
        f"raw sample peak {streamline_raw:.1f}°",
        fontsize=9.5,
    )

    finite_bend = bend_degrees[np.isfinite(bend_degrees)]
    bend_p95 = (
        float(np.percentile(finite_bend, 95.0)) if finite_bend.size else np.nan
    )
    bend_raw = float(np.max(finite_bend)) if finite_bend.size else np.nan
    bend_limit = max(bend_p95, np.finfo(float).eps)
    axes[1, 1].imshow(
        np.ma.masked_invalid(bend_degrees),
        cmap="magma",
        vmin=0.0,
        vmax=bend_limit,
        alpha=0.82,
    )
    tangent_rows, tangent_cols = np.nonzero(tangential_pixels)
    if tangent_rows.size:
        display = np.arange(0, tangent_rows.size, max(1, tangent_rows.size // 1500))
        axes[1, 1].scatter(
            tangent_cols[display],
            tangent_rows[display],
            s=3.0,
            facecolors="none",
            edgecolors="#00E5FF",
            linewidths=0.45,
            alpha=0.72,
        )
    tangential_fraction = (
        float(tangential_pixels.sum()) / float(bend_valid.sum())
        if bend_valid.any()
        else np.nan
    )
    tangential_count = int(tangential_pixels.sum())
    bend_valid_count = int(bend_valid.sum())
    axes[1, 1].set_title(
        "D. Local bend σ=32 px + near-tangent occupancy (cyan)\n"
        f"near-tangent/bend-valid {tangential_count}/{bend_valid_count} "
        f"= {tangential_fraction:.1%}\n"
        f"bend p95 {bend_p95:.3f}°/px; raw peak {bend_raw:.3f}°/px",
        fontsize=9.5,
    )

    rotation_cax = axes[0, 1].inset_axes([1.035, 0.08, 0.035, 0.84])
    shared_colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=SPECTRAL_NORM, cmap=cmap),
        cax=rotation_cax,
        orientation="vertical",
    )
    shared_colorbar.set_label(
        "A-C: accumulated signed rotation (degrees), fixed −180° to +180°"
    )
    bend_cax = axes[1, 1].inset_axes([1.035, 0.08, 0.035, 0.84])
    bend_colorbar = figure.colorbar(
        plt.cm.ScalarMappable(
            norm=Normalize(vmin=0.0, vmax=bend_limit), cmap="magma"
        ),
        cax=bend_cax,
        orientation="vertical",
    )
    bend_colorbar.set_label("Local bend (degrees/pixel), clipped at p95")
    figure.suptitle(
        f"{colony}: four ways to handle radial and tangential branch rotation",
        fontsize=14,
    )
    figure.text(
        0.5,
        -0.01,
        (
            "The inoculum is excluded in every panel. A/B retain ring-based outward "
            "correspondence. C follows the continuous axial field; dashed pairs mark "
            "exact/near-tangent launch ambiguity. D is local and unsigned, not a "
            "cumulative outward-turn metric."
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )
    output = OUTPUT_DIR / f"twok_{colony}_tangential_methods_overlay_2x2.png"
    figure.savefig(
        output,
        dpi=190,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(figure)
    rows = [
        {
            "Colony": colony,
            "Method": "Current radial matched rings",
            "Support": strict_support,
            "AbsP95": strict_p95,
            "RawPeak": strict_raw,
            "Units": "degrees",
        },
        {
            "Colony": colony,
            "Method": "Bounded tangential ring graph",
            "Support": tangent_support,
            "AbsP95": tangent_p95,
            "RawPeak": tangent_raw,
            "Units": "degrees",
        },
        {
            "Colony": colony,
            "Method": "Cartesian director streamlines",
            "Support": float(supported_seeds) / 36.0,
            "SupportDenominator": 36,
            "Hypotheses": stream_hypotheses,
            "AbsP95": streamline_p95,
            "RawPeak": streamline_raw,
            "Units": "degrees; Support=supported seed fraction",
        },
        {
            "Colony": colony,
            "Method": "Local bend + tangent occupancy",
            "Support": tangential_fraction,
            "SupportDenominator": bend_valid_count,
            "Hypotheses": tangential_count,
            "AbsP95": bend_p95,
            "RawPeak": bend_raw,
            "Units": "degrees/pixel; Support=tangent pixel fraction",
        },
    ]
    return output, rows


def render_all_tangential_method_comparisons() -> None:
    """Render both measured colonies and export a compact comparison table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detected, _old = load_twok_detection()
    rows: list[dict[str, float | str]] = []
    for colony, label in COLONIES:
        section = isolated_global_crop(
            detected, label, label_centroid(detected, label)
        )
        operation = MeasureOrientationZones(
            radial_ring_width=8.0,
            long_range_lag=32.0,
            quiver_block=24,
        )
        profiles = extract_profiles(section, operation)
        phi, fiber_axis, coherence, selector = extract_pixel_fields(
            section, operation, profiles
        )
        output, colony_rows = render_method_comparison(
            colony,
            profiles,
            phi,
            fiber_axis,
            coherence,
            selector,
        )
        rows.extend(colony_rows)
        report(str(output))
    table = pd.DataFrame(rows)
    csv_output = OUTPUT_DIR / "twok_tangential_methods_comparison.csv"
    table.to_csv(csv_output, index=False)
    report(str(csv_output))
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    render_all_tangential_method_comparisons()
