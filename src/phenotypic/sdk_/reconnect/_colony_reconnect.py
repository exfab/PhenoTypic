"""Composite-cost + tiled Dijkstra reconnection for filamentous colonies.

Pure array functions extracted from FilamentousFungiDetector. Callers pass
already-computed phase-congruency arrays (pc_sum, M, m, orientation) and a
ReconnectConfig of scalar parameters — never an Image, operation, or the
_PhaseCong3Result dataclass (see package CLAUDE.md import contract).
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as ndi_label
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import dilation, disk, remove_small_objects
from skimage.segmentation import find_boundaries

from ..branch_pathfinding import (
    DijkstraResult,
    _apply_border_penalty_inplace,
    _apply_distance_gap_penalty_inplace,
    _apply_structure_mask_inplace,
    _compute_screening_envelope,
    apply_filter_cascade,
    assemble_composite_cost,
    assign_fragments_to_colonies,
    calibrate_screening_threshold,
    calibrate_thresholds,
    compute_anisotropy,
    compute_local_mad_map,
    compute_orientation_coherence,
    extract_calibration_branches,
    extract_fragment_paths,
    prescreen_fragments,
    run_multisource_dijkstra,
)
from ._gwdt import app2_gwdt_cost, grey_weighted_distance


@dataclass(frozen=True)
class ReconnectConfig:
    """Scalar parameters for cost-surface assembly + tiled Dijkstra reconnection.

    All values are pre-derived by the caller (e.g. a detector's scene-derivation
    validator). Field meanings mirror the identically named FilamentousFungiDetector
    fields / ClassVars.
    """
    beta: float                     # anisotropy exponent in composite cost
    gamma: float                    # MAD penalty weight in composite cost numerator
    delta: float                    # Dijkstra radial retreat penalty
    coherence_window_radius: int
    mad_window: int
    gap_crossing_penalty: float
    border_margin_px: int
    frag_reach_px: int
    tile_size: int
    tile_overlap: int
    max_gap_length: int
    path_dilation_radius: int
    snr_margin: int
    reconnection_tolerance: float


def identify_pseudo_fragments(
    colony_labels: np.ndarray,
    center_objmask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify pseudo-fragments: per-label CCs that don't overlap the inoculum.

    Returns (central_mask, fragment_labels) where central_mask is the main
    colony mass (CCs overlapping ``center_objmask``) and fragment_labels is a
    labeled map of the disconnected blobs.
    """
    foreground = colony_labels > 0
    cc_map, n_cc = ndi_label(foreground)
    if n_cc == 0:
        return (np.zeros_like(foreground),
                np.zeros(foreground.shape, dtype=np.int32))
    seeded_ccs = np.unique(cc_map[center_objmask & foreground])
    is_central = np.zeros(n_cc + 1, dtype=bool)
    is_central[seeded_ccs] = True
    central_mask = is_central[cc_map]
    fragment_mask = foreground & ~central_mask
    if fragment_mask.any():
        fragment_labels = label(fragment_mask).astype(np.int32)
    else:
        fragment_labels = np.zeros(foreground.shape, dtype=np.int32)
    return central_mask, fragment_labels


def select_reconnect_fragments(
    colony_labels: np.ndarray,
    center_mask: np.ndarray,
    colony_mask: np.ndarray,
    structure_mask: np.ndarray,
    *,
    scope: str = "branches",
    min_fragment_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the fragment set the Dijkstra reconnection is allowed to bridge.

    ``scope="pseudo"`` reproduces :func:`identify_pseudo_fragments` exactly: only the
    Voronoi-cut pieces of ``colony_labels`` that miss the inoculum. ``scope="branches"``
    additionally admits the disconnected branch components the overlap filter drops
    (``colony_mask & ~structure_mask``), so genuinely-severed hyphae reach reconnection
    instead of being deleted before it runs. Fragments the reconnection cannot bridge are
    still dropped downstream by the caller's ``final_mask`` step (they are never painted
    into ``colony_labels``), so no extra deletion is needed here.

    Args:
        colony_labels: Voronoi colony labels built from ``structure_mask`` (the Dijkstra
            targets). Zero is background.
        center_mask: Boolean inoculum-center mask.
        colony_mask: The pre-filter union ``branch_mask | center_mask``.
        structure_mask: ``filter_mask_by_overlap(colony_mask, center_mask)`` — the
            center-connected bodies kept today.
        scope: ``"branches"`` (default) admits disconnected branch fragments;
            ``"pseudo"`` restricts to Voronoi-cut pseudo-fragments (legacy behavior).
        min_fragment_size: Drop connected fragments smaller than this many pixels
            (``scope="branches"`` only). ``1`` keeps all.

    Returns:
        ``(central_mask, fragment_labels)``. ``central_mask`` is the trusted colony
        bodies (unchanged vs the pseudo path); ``fragment_labels`` is an int32 relabeled
        map of the fragments to reconnect.

    Raises:
        ValueError: If ``scope`` is not ``"branches"`` or ``"pseudo"``.
    """
    central_mask, fragment_labels = identify_pseudo_fragments(colony_labels, center_mask)
    if scope == "pseudo":
        return central_mask, fragment_labels
    if scope != "branches":
        raise ValueError(f"scope must be 'branches' or 'pseudo', got {scope!r}")

    media_frag_mask = (
        np.asarray(colony_mask, dtype=bool) & ~np.asarray(structure_mask, dtype=bool)
    )
    fragment_mask = (fragment_labels > 0) | media_frag_mask
    if min_fragment_size > 1:
        fragment_mask = remove_small_objects(fragment_mask, min_size=min_fragment_size)
    return central_mask, label(fragment_mask).astype(np.int32)


def _apply_penalties_inplace(
    cost: np.ndarray,
    pct_energy: np.ndarray,
    colony_labels: np.ndarray,
    cfg: ReconnectConfig,
) -> None:
    """Apply distance-gap and border penalties in place."""
    _apply_distance_gap_penalty_inplace(
        cost, pct_energy, colony_labels, cfg.gap_crossing_penalty,
    )
    _apply_border_penalty_inplace(cost, cfg.border_margin_px)


def build_reconnect_cost(
    pc_sum: np.ndarray,
    M: np.ndarray,
    m: np.ndarray,
    orientation: np.ndarray,
    enhanced_arr: np.ndarray,
    colony_labels: np.ndarray,
    central_mask: np.ndarray,
    cfg: ReconnectConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the composite cost surface from phase-congruency feature arrays.

    Args:
        pc_sum, M, m, orientation: Phase-congruency arrays (the fields of a
            _PhaseCong3Result, passed separately to keep this package pure).
        enhanced_arr: 2D contrast-stretched detection matrix for local MAD.
        colony_labels: Labeled colony assignment.
        central_mask: Boolean mask of branch pixels overlapping colonies.
        cfg: Scalar reconnection parameters.

    Returns:
        (unmasked_cost, cost_surface). ``cost_surface`` has colony/central
        pixels set to near-zero traversal cost.
    """
    anisotropy = compute_anisotropy(M, m)
    coherence = compute_orientation_coherence(orientation, cfg.coherence_window_radius)
    mad = compute_local_mad_map(enhanced_arr, cfg.mad_window)
    base_cost = assemble_composite_cost(pc_sum, anisotropy, coherence, mad, cfg.beta, cfg.gamma)

    unmasked_cost = base_cost.copy()
    _apply_penalties_inplace(unmasked_cost, pc_sum, colony_labels, cfg)

    colony_mask = (colony_labels > 0) | central_mask
    _apply_structure_mask_inplace(base_cost, colony_mask.astype(np.int32))
    _apply_penalties_inplace(base_cost, pc_sum, colony_labels, cfg)
    return unmasked_cost, base_cost


_APP2_NEIGHBORS = (
    (0, 1, 1.0),
    (-1, 1, 1.414214),
    (-1, 0, 1.0),
    (-1, -1, 1.414214),
    (0, -1, 1.0),
    (1, -1, 1.414214),
    (1, 0, 1.0),
    (1, 1, 1.414214),
)


def _run_app2_gwdt_dijkstra(
    gi_cost: np.ndarray,
    colony_labels: np.ndarray,
) -> DijkstraResult:
    """Propagate colony ownership with APP2 endpoint-averaged GI edge costs.

    This is deliberately separate from the existing destination-only Dijkstra
    kernel. APP2 charges an axial edge ``(GI(p) + GI(q)) / 2`` and a diagonal
    edge the same average multiplied by its source constant ``1.414214``.
    Unlike APP2's image-threshold gate, this detector seam deliberately permits
    traversal through every finite GI pixel; downstream path-quality filters own
    the gap-acceptance decision. Equal-cost multi-colony ties retain the first
    boundary seed in row-major order, then the first path reached through the
    fixed ``_APP2_NEIGHBORS`` order.

    Args:
        gi_cost: Finite, nonnegative, two-dimensional APP2 lookup map.
        colony_labels: Same-shaped integer colony labels with zero background.

    Returns:
        Dijkstra propagation maps compatible with the shared path extractor.

    Raises:
        ValueError: If the two arrays do not satisfy the internal seam contract.
    """
    costs = np.asarray(gi_cost)
    labels = np.asarray(colony_labels)
    if costs.ndim != 2 or labels.ndim != 2 or costs.shape != labels.shape:
        raise ValueError("gi_cost and colony_labels must be same-shaped 2-D arrays")
    if not np.issubdtype(costs.dtype, np.number) or not np.all(np.isfinite(costs)):
        raise ValueError("gi_cost must contain finite numeric values")
    if np.any(costs < 0.0):
        raise ValueError("gi_cost must be nonnegative")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("colony_labels must contain nonnegative integers")

    rows, columns = costs.shape
    cost_distance = np.full(costs.shape, np.inf, dtype=np.float64)
    colony_id = np.full(costs.shape, -1, dtype=np.int32)
    predecessor = np.full(costs.shape, -1, dtype=np.int32)
    visited = np.zeros(costs.shape, dtype=np.bool_)

    colony_mask = labels > 0
    boundary_mask = find_boundaries(colony_mask, mode="inner", connectivity=2)
    cost_distance[colony_mask] = 0.0
    colony_id[colony_mask] = labels[colony_mask].astype(np.int32, copy=False)
    visited[colony_mask & ~boundary_mask] = True

    centroids: dict[int, tuple[float, float]] = {}
    for colony_label in np.unique(labels[colony_mask]):
        coordinates = np.argwhere(labels == colony_label)
        centroid = coordinates.mean(axis=0)
        centroids[int(colony_label)] = (float(centroid[0]), float(centroid[1]))

    counter = itertools.count()
    heap: list[tuple[float, int, int, int]] = []
    for row, column in np.argwhere(boundary_mask):
        heapq.heappush(heap, (0.0, next(counter), int(row), int(column)))

    while heap:
        distance, _, row, column = heapq.heappop(heap)
        if visited[row, column] or distance != cost_distance[row, column]:
            continue
        visited[row, column] = True
        source_label = colony_id[row, column]

        for row_offset, column_offset, factor in _APP2_NEIGHBORS:
            neighbor_row = row + row_offset
            neighbor_column = column + column_offset
            if (
                neighbor_row < 0
                or neighbor_row >= rows
                or neighbor_column < 0
                or neighbor_column >= columns
                or visited[neighbor_row, neighbor_column]
            ):
                continue
            edge_cost = (
                (float(costs[row, column]) + float(costs[neighbor_row, neighbor_column]))
                * factor
                / 2.0
            )
            candidate = distance + edge_cost
            if candidate < cost_distance[neighbor_row, neighbor_column]:
                cost_distance[neighbor_row, neighbor_column] = candidate
                colony_id[neighbor_row, neighbor_column] = source_label
                predecessor[neighbor_row, neighbor_column] = row * columns + column
                heapq.heappush(
                    heap,
                    (candidate, next(counter), neighbor_row, neighbor_column),
                )

    return DijkstraResult(
        cost_distance=cost_distance,
        colony_id=colony_id,
        predecessor=predecessor,
        colony_centroids=centroids,
    )


def _generate_tiles(
    image_shape: tuple[int, int],
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    """Generate overlapping tile coordinates covering the full image.

    Args:
        image_shape: (height, width) of the image.
        tile_size: Side length of square tiles.
        overlap: Overlap in pixels between adjacent tiles.

    Returns:
        List of (row_start, row_end, col_start, col_end) tuples.
    """
    H, W = image_shape
    step = tile_size - overlap
    tiles: list[tuple[int, int, int, int]] = []

    row = 0
    while row < H:
        row_end = min(row + tile_size, H)
        col = 0
        while col < W:
            col_end = min(col + tile_size, W)
            tiles.append((row, row_end, col, col_end))
            if col_end == W:
                break
            col += step
        if row_end == H:
            break
        row += step

    return tiles


def _merge_tile_into_output(
    output: np.ndarray,
    tile_labels: np.ndarray,
    row_start: int,
    col_start: int,
) -> None:
    """Write tile results into global output array.

    Only overwrites pixels that are currently unlabeled (0) in the output,
    preserving existing colony labels from earlier tiles or the watershed.

    Args:
        output: Global output label array (modified in place).
        tile_labels: Processed tile label array.
        row_start: Row offset of this tile in the global image.
        col_start: Column offset of this tile in the global image.
    """
    tile_h, tile_w = tile_labels.shape
    out_slice = output[row_start:row_start + tile_h, col_start:col_start + tile_w]
    new_pixels = (tile_labels > 0) & (out_slice == 0)
    out_slice[new_pixels] = tile_labels[new_pixels]


def compute_full_image_app2_gi_cost(
    image: np.ndarray,
    *,
    background: np.ndarray,
) -> np.ndarray:
    """Compute the APP2 GI lookup once before any tile is extracted.

    The detector defines background as pixels excluded by its full-image
    dual-mask branch detector. Both background and foreground must be
    present because the source lookup is not a meaningful detector seam
    for either constant class mask.

    Args:
        image: Full contrast-stretched detection image.
        background: Full-image boolean background mask.

    Returns:
        Full-image float64 APP2 GI lookup values.

    Raises:
        ValueError: If either background or foreground is absent.
    """
    if not np.any(background) or np.all(background):
        raise ValueError(
            "app2_gwdt reconnection requires both background and foreground"
        )
    distance = grey_weighted_distance(image, background, connectivity=8)
    return app2_gwdt_cost(distance)


def _process_tile(
    tile_cost: np.ndarray,
    tile_raw: np.ndarray,
    tile_colony: np.ndarray,
    tile_frags: np.ndarray,
    tile_pct: np.ndarray,
    tile_gray: np.ndarray,
    pct_noise_ceil: float,
    cfg: ReconnectConfig,
    tile_app2_gi: np.ndarray | None = None,
) -> np.ndarray:
    """Process a single tile: Dijkstra, assign, paths, quality filter, assemble.

    Args:
        tile_cost: Masked cost surface for this tile.
        tile_raw: Unmasked cost surface for quality calibration.
        tile_colony: Colony labels for this tile.
        tile_frags: Fragment labels for this tile.
        tile_pct: PCT energy map for this tile.
        tile_gray: Grayscale image for this tile.
        pct_noise_ceil: PCT energy threshold for F5 background masking.
        cfg: Scalar reconnection parameters.
        tile_app2_gi: Optional slice of the full-image APP2 GI map.

    Returns:
        Updated tile colony labels with reconnected fragments.
    """
    if tile_frags.max() == 0:
        return tile_colony

    if tile_colony.max() == 0:
        return tile_colony

    # Run Dijkstra from colony boundaries
    if tile_app2_gi is None:
        dijkstra = run_multisource_dijkstra(
            tile_cost, tile_colony, cfg.delta
        )
    else:
        dijkstra = _run_app2_gwdt_dijkstra(tile_app2_gi, tile_colony)

    # Assign fragments to colonies by majority vote
    assignments = assign_fragments_to_colonies(
        tile_frags, dijkstra.colony_id, dijkstra.cost_distance
    )

    # Extract minimum-cost paths from fragments to colonies
    paths, _unconnected = extract_fragment_paths(
        tile_frags, assignments, dijkstra, tile_cost
    )

    if not paths:
        return tile_colony

    # Quality filter: calibrate from colony skeleton branches
    calibration = extract_calibration_branches(
        tile_colony, tile_raw,
        window_cost=cfg.max_gap_length,
        dilation_radius=cfg.path_dilation_radius,
        pct_energy=tile_pct,
        grayscale=tile_gray,
        snr_margin=cfg.snr_margin,
        pct_noise_ceil=pct_noise_ceil,
    )

    # Only apply quality filters if we have calibration data
    if calibration.median_cost_values.size > 0:
        thresholds = calibrate_thresholds(
            calibration, k=cfg.reconnection_tolerance
        )
        filter_result = apply_filter_cascade(
            paths, tile_raw, thresholds,
            window_cost=cfg.max_gap_length,
            dilation_radius=cfg.path_dilation_radius,
            pct_energy=tile_pct,
            grayscale=tile_gray,
            snr_margin=cfg.snr_margin,
            pct_noise_ceil=pct_noise_ceil,
        )
        passed_ids = filter_result.passed_ids
    else:
        # No calibration data: accept all paths
        passed_ids = set(paths.keys())

    # Build result: paint fragment + dilated path with colony ID
    result = tile_colony.copy()
    selem = disk(cfg.path_dilation_radius)

    # Group path coords by colony for batched dilation
    colony_coords: dict[int, list[np.ndarray]] = {}

    for fid in passed_ids:
        if fid not in paths or fid not in assignments:
            continue
        path = paths[fid]
        cid = assignments[fid].colony_id
        if cid < 0:
            continue

        # Paint fragment pixels
        frag_mask = tile_frags == fid
        result[frag_mask] = cid

        # Collect path coords for batched dilation
        rows = path.coords[:, 0]
        cols = path.coords[:, 1]
        valid = (
            (rows >= 0) & (rows < result.shape[0])
            & (cols >= 0) & (cols < result.shape[1])
        )
        colony_coords.setdefault(cid, []).append(
            path.coords[valid]
        )

    # Single dilation per colony
    for cid, coord_list in colony_coords.items():
        all_coords = np.vstack(coord_list)
        path_mask = np.zeros(result.shape, dtype=np.bool_)
        path_mask[all_coords[:, 0], all_coords[:, 1]] = True
        dilated = dilation(path_mask, selem)
        result[dilated] = cid

    return result


def reconnect_fragments_tiled(
    colony_labels: np.ndarray,
    fragment_labels: np.ndarray,
    cost_surface: np.ndarray,
    unmasked_cost: np.ndarray,
    pct_energy: np.ndarray,
    grayscale: np.ndarray,
    cfg: ReconnectConfig,
    app2_gi_cost: np.ndarray | None = None,
) -> np.ndarray:
    """Generate tiles, process each, merge results into output mask.

    The configured overlap is the processing halo: there is no separate
    core/halo crop. Tiles run in row-major order, edge tiles clip to the
    image, every auxiliary array including the nonlocal APP2 map uses the
    same bounds, and the first processed tile owns an overlap pixel.

    Args:
        colony_labels: Labeled colony assignment from watershed.
        fragment_labels: Labeled array of disconnected branch fragments.
        cost_surface: Masked composite cost surface for Dijkstra.
        unmasked_cost: Unmasked composite cost for quality calibration.
        pct_energy: Float32 (H, W) PCT energy map for quality filtering.
        grayscale: Float32 (H, W) enhanced grayscale for SNR filtering.
        cfg: Scalar reconnection parameters.
        app2_gi_cost: Optional full-image APP2 GI map. When present, each
            tile receives a view of this already-computed nonlocal map.

    Returns:
        Updated colony labels with reconnected fragments painted in.
    """
    if fragment_labels.max() == 0:
        return colony_labels

    # Prescreen fragments: compute envelope once, share across calibration + screening
    colony_branch_mask = (colony_labels > 0).astype(np.int32)
    min_cost_envelope, _ = _compute_screening_envelope(
        cost_surface, colony_branch_mask, cfg.frag_reach_px
    )
    tau_screen, _ = calibrate_screening_threshold(
        cost_surface, colony_branch_mask, r_screen=cfg.frag_reach_px,
        min_cost_envelope=min_cost_envelope,
    )

    screen_result = prescreen_fragments(
        cost_surface, fragment_labels,
        r_screen=cfg.frag_reach_px,
        tau_screen=tau_screen,
        colony_branch_mask=colony_branch_mask,
        min_cost_envelope=min_cost_envelope,
    )
    screened_frags = screen_result.screened_fragment_labels

    if screened_frags.max() == 0:
        return colony_labels

    # Compute PCT noise ceiling for F5 background masking
    pct_noise_ceil = float(threshold_otsu(pct_energy))

    # Generate tiles
    tiles = _generate_tiles(
        colony_labels.shape, cfg.tile_size, cfg.tile_overlap
    )

    output = colony_labels.copy()

    for row_start, row_end, col_start, col_end in tiles:
        tile_cost = cost_surface[row_start:row_end, col_start:col_end]
        tile_raw = unmasked_cost[row_start:row_end, col_start:col_end]
        tile_colony = output[row_start:row_end, col_start:col_end]
        tile_frags = screened_frags[row_start:row_end, col_start:col_end]
        tile_pct = pct_energy[row_start:row_end, col_start:col_end]
        tile_gray = grayscale[row_start:row_end, col_start:col_end]
        tile_app2_gi = (
            None
            if app2_gi_cost is None
            else app2_gi_cost[row_start:row_end, col_start:col_end]
        )

        tile_result = _process_tile(
            tile_cost, tile_raw, tile_colony, tile_frags,
            tile_pct, tile_gray, pct_noise_ceil, cfg, tile_app2_gi,
        )
        _merge_tile_into_output(
            output, tile_result, row_start, col_start
        )

    return output
