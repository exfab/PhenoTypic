"""Multi-source Dijkstra propagation and fragment-to-colony connection.

Implements seeded multi-source Dijkstra from all colony boundaries
simultaneously, with a radial progress penalty that discourages
wavefronts from retreating toward their colony centroid. After
propagation, fragments are assigned to their nearest colony by
majority vote of the colony-ID map, and minimum-cost paths are
backtracked from each fragment to its assigned colony.

Stage 2: Seeded multi-source Dijkstra with radial penalty
Stage 3: Fragment-to-colony path extraction and assembly
"""

from __future__ import annotations

import numba
import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from ._dataclasses import DijkstraResult, FragmentAssignment, FragmentPath

# ── 8-connectivity direction tables ──────────────────────────────────

# (dr, dc) for 8-connected neighbors: E, NE, N, NW, W, SW, S, SE
DR = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int32)
DC = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
DIST = np.array(
    [
        1.0,
        1.4142135623730951,
        1.0,
        1.4142135623730951,
        1.0,
        1.4142135623730951,
        1.0,
        1.4142135623730951,
    ],
    dtype=np.float64,
)


# ── Helper functions ─────────────────────────────────────────────────


def compute_colony_centroids(
    colony_labels: np.ndarray,
) -> dict[int, tuple[float, float]]:
    """Extract centroid (row, col) for each labeled colony.

    Args:
        colony_labels: Integer label array (H, W). 0 is background,
            positive integers are colony IDs.

    Returns:
        Dict mapping colony_id to (centroid_row, centroid_col).
    """
    centroids: dict[int, tuple[float, float]] = {}
    for prop in regionprops(colony_labels):
        centroids[prop.label] = prop.centroid  # (row, col)
    return centroids


def extract_colony_boundaries(
    colony_labels: np.ndarray,
) -> np.ndarray:
    """Extract inner boundary pixels of all colonies combined.

    Uses inner boundaries with 8-connectivity on the combined colony mask
    to ensure boundary pixels are inside colonies. These pixels serve as
    Dijkstra source seeds.

    Args:
        colony_labels: Integer label array (H, W). 0 is background.

    Returns:
        Boolean mask (H, W) where True indicates a boundary pixel of
        any colony.
    """
    return find_boundaries(colony_labels > 0, mode="inner", connectivity=2)


# ── Numba JIT kernels ────────────────────────────────────────────────


@numba.njit(inline="always")
def _heap_push(heap_cost, heap_row, heap_col, heap_size, cost, row, col):
    """Push (cost, row, col) onto a min-heap of parallel arrays.

    Args:
        heap_cost: Float64 array backing the heap costs.
        heap_row: Int32 array backing the heap row indices.
        heap_col: Int32 array backing the heap column indices.
        heap_size: Current number of elements in the heap.
        cost: Cost value to push.
        row: Row index to push.
        col: Column index to push.

    Returns:
        New heap size after insertion.
    """
    if heap_size >= heap_cost.shape[0]:
        return heap_size  # overflow guard: silently drop
    idx = heap_size
    heap_cost[idx] = cost
    heap_row[idx] = row
    heap_col[idx] = col
    heap_size += 1
    # Sift up
    while idx > 0:
        parent = (idx - 1) >> 1
        if heap_cost[parent] <= heap_cost[idx]:
            break
        # Swap
        heap_cost[idx], heap_cost[parent] = heap_cost[parent], heap_cost[idx]
        heap_row[idx], heap_row[parent] = heap_row[parent], heap_row[idx]
        heap_col[idx], heap_col[parent] = heap_col[parent], heap_col[idx]
        idx = parent
    return heap_size


@numba.njit(inline="always")
def _heap_pop(heap_cost, heap_row, heap_col, heap_size):
    """Pop minimum from min-heap.

    Args:
        heap_cost: Float64 array backing the heap costs.
        heap_row: Int32 array backing the heap row indices.
        heap_col: Int32 array backing the heap column indices.
        heap_size: Current number of elements in the heap.

    Returns:
        Tuple of (cost, row, col, new_size).
    """
    cost = heap_cost[0]
    row = heap_row[0]
    col = heap_col[0]
    heap_size -= 1
    # Move last element to root
    heap_cost[0] = heap_cost[heap_size]
    heap_row[0] = heap_row[heap_size]
    heap_col[0] = heap_col[heap_size]
    # Sift down
    idx = 0
    while True:
        left = 2 * idx + 1
        right = 2 * idx + 2
        smallest = idx
        if left < heap_size and heap_cost[left] < heap_cost[smallest]:
            smallest = left
        if right < heap_size and heap_cost[right] < heap_cost[smallest]:
            smallest = right
        if smallest == idx:
            break
        heap_cost[idx], heap_cost[smallest] = heap_cost[smallest], heap_cost[idx]
        heap_row[idx], heap_row[smallest] = heap_row[smallest], heap_row[idx]
        heap_col[idx], heap_col[smallest] = heap_col[smallest], heap_col[idx]
        idx = smallest
    return cost, row, col, heap_size


@numba.njit(cache=True)
def _dijkstra_kernel(
    cost_surface,
    cost_distance,
    colony_id,
    predecessor,
    visited,
    centroid_row,
    centroid_col,
    seed_rows,
    seed_cols,
    delta,
    DR,
    DC,
    DIST,
):
    """Numba-accelerated Dijkstra inner loop with manual binary heap.

    Expands from all seed pixels simultaneously. Each step applies
    a radial penalty when the wavefront retreats toward its colony
    centroid (squared-distance comparison avoids sqrt).

    Args:
        cost_surface: Float32/64 (H, W) composite cost map.
        cost_distance: Float64 (H, W) output cost-distance, initialized
            to inf everywhere except colony pixels (0).
        colony_id: Int32 (H, W) colony ownership map, initialized to
            colony labels inside colonies and -1 elsewhere.
        predecessor: Int32 (H, W) flat-index predecessor map, initialized
            to -1.
        visited: Bool (H, W) visited flags, initialized to True for
            colony interior pixels.
        centroid_row: Float64 (max_cid+1,) row centroids indexed by
            colony ID.
        centroid_col: Float64 (max_cid+1,) col centroids indexed by
            colony ID.
        seed_rows: Int32 (N,) row indices of boundary seed pixels.
        seed_cols: Int32 (N,) col indices of boundary seed pixels.
        delta: Radial penalty factor applied to retreating steps.
        DR: Int32 (8,) row offsets for 8-connectivity.
        DC: Int32 (8,) col offsets for 8-connectivity.
        DIST: Float64 (8,) step distances for 8-connectivity.
    """
    H = cost_surface.shape[0]
    W = cost_surface.shape[1]

    # Allocate heap arrays — 2x pixel count to accommodate lazy-deletion
    # duplicates (a pixel can be pushed multiple times before being popped)
    capacity = 2 * H * W
    heap_cost = np.empty(capacity, dtype=np.float64)
    heap_row = np.empty(capacity, dtype=np.int32)
    heap_col = np.empty(capacity, dtype=np.int32)
    heap_size = 0

    # Seed boundary pixels
    for i in range(seed_rows.shape[0]):
        r = seed_rows[i]
        c = seed_cols[i]
        heap_size = _heap_push(
            heap_cost, heap_row, heap_col, heap_size, 0.0, r, c
        )

    # Main Dijkstra loop
    while heap_size > 0:
        cost_u, r, c, heap_size = _heap_pop(
            heap_cost, heap_row, heap_col, heap_size
        )

        if visited[r, c]:
            continue
        visited[r, c] = True

        k = colony_id[r, c]
        cr = centroid_row[k]
        cc = centroid_col[k]

        # Squared distance from current pixel to its colony centroid
        dr_c = r - cr
        dc_c = c - cc
        r_sq_current = dr_c * dr_c + dc_c * dc_c

        for d in range(8):
            nr = r + DR[d]
            nc = c + DC[d]

            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            if visited[nr, nc]:
                continue

            # Base step cost: cost_surface at destination * step distance
            step_cost = cost_surface[nr, nc] * DIST[d]

            # Radial penalty: penalize steps that retreat toward centroid
            dr_n = nr - cr
            dc_n = nc - cc
            r_sq_next = dr_n * dr_n + dc_n * dc_n

            if r_sq_next < r_sq_current:
                step_cost *= 1.0 + delta

            new_cost = cost_u + step_cost

            if new_cost < cost_distance[nr, nc]:
                cost_distance[nr, nc] = new_cost
                colony_id[nr, nc] = k
                predecessor[nr, nc] = r * W + c  # flat index
                heap_size = _heap_push(
                    heap_cost, heap_row, heap_col, heap_size, new_cost, nr, nc
                )


# ── Stage 2: Multi-source Dijkstra ───────────────────────────────────


def run_multisource_dijkstra(
    cost_surface: np.ndarray,
    colony_labels: np.ndarray,
    delta: float = 1.0,
) -> DijkstraResult:
    """Seeded multi-source Dijkstra with radial progress penalty.

    All colony boundary pixels are seeded simultaneously at cost 0.
    Wavefronts expand outward; when a step moves closer to its own
    colony centroid (retreating), the step cost is penalized by
    factor (1 + delta). This encourages outward radial expansion
    consistent with filamentous fungal growth.

    Squared distances are compared to avoid sqrt in the inner loop.

    Args:
        cost_surface: Float32 (H, W) composite cost map. Colony pixels
            should already be set to epsilon (~1e-6).
        colony_labels: Int32 (H, W) labeled colony mask. 0 is background.
        delta: Radial penalty factor. When a step retreats toward the
            colony centroid, step_cost is multiplied by (1 + delta).
            Default 1.0 doubles the cost of retreating steps.

    Returns:
        DijkstraResult with cost_distance, colony_id, predecessor maps
        and colony centroid lookup.
    """
    H, W = cost_surface.shape

    # Output arrays
    cost_distance = np.full((H, W), np.inf, dtype=np.float64)
    colony_id = np.full((H, W), -1, dtype=np.int32)
    predecessor = np.full((H, W), -1, dtype=np.int32)
    visited = np.zeros((H, W), dtype=np.bool_)

    # Centroid lookup: flat arrays indexed by colony ID for fast access
    centroids = compute_colony_centroids(colony_labels)
    max_cid = max(centroids.keys()) if centroids else 0
    centroid_row = np.zeros(max_cid + 1, dtype=np.float64)
    centroid_col = np.zeros(max_cid + 1, dtype=np.float64)
    for cid, (cr, cc) in centroids.items():
        centroid_row[cid] = cr
        centroid_col[cid] = cc

    # Identify colony interior and boundary pixels
    colony_mask = colony_labels > 0
    boundary_mask = extract_colony_boundaries(colony_labels)

    # Initialize all colony pixels: cost=0, colony_id from labels
    cost_distance[colony_mask] = 0.0
    colony_id[colony_mask] = colony_labels[colony_mask]

    # Mark interior (non-boundary) colony pixels as visited
    interior = colony_mask & ~boundary_mask
    visited[interior] = True

    # Run Numba-accelerated Dijkstra kernel
    bnd_rows, bnd_cols = np.where(boundary_mask)
    seed_rows = bnd_rows.astype(np.int32)
    seed_cols = bnd_cols.astype(np.int32)
    _dijkstra_kernel(
        cost_surface,
        cost_distance,
        colony_id,
        predecessor,
        visited,
        centroid_row,
        centroid_col,
        seed_rows,
        seed_cols,
        delta,
        DR,
        DC,
        DIST,
    )

    return DijkstraResult(
        cost_distance=cost_distance,
        colony_id=colony_id,
        predecessor=predecessor,
        colony_centroids=centroids,
    )


# ── Stage 3: Fragment assignment and path extraction ─────────────────


def assign_fragments_to_colonies(
    fragment_labels: np.ndarray,
    colony_id_map: np.ndarray,
    cost_distance: np.ndarray,
    bridge_threshold: float = 0.20,
) -> dict[int, FragmentAssignment]:
    """Assign each fragment to a colony by majority vote of colony_id.

    For each fragment, samples the colony_id_map at all fragment pixels.
    The majority colony wins. If the second-most-common colony exceeds
    bridge_threshold fraction, the fragment is flagged as a bridge.

    Args:
        fragment_labels: Int32 (H, W) labeled fragment mask.
        colony_id_map: Int32 (H, W) colony ownership from Dijkstra.
        cost_distance: Float64 (H, W) cost-distance from Dijkstra.
        bridge_threshold: Minority fraction above which a fragment is
            flagged as a bridge. Default 0.20 (20%).

    Returns:
        Dict mapping fragment_id to FragmentAssignment.
    """
    frag_ids = np.unique(fragment_labels)
    frag_ids = frag_ids[frag_ids > 0]

    assignments: dict[int, FragmentAssignment] = {}
    for fid in frag_ids:
        fid = int(fid)
        mask = fragment_labels == fid
        pixel_colony_ids = colony_id_map[mask]
        pixel_costs = cost_distance[mask]

        # Filter out unreached pixels (colony_id == -1)
        valid = pixel_colony_ids >= 0
        if not np.any(valid):
            # Fragment entirely unreached
            assignments[fid] = FragmentAssignment(
                fragment_id=fid,
                colony_id=-1,
                is_bridge=False,
                majority_fraction=0.0,
                mean_cost=float("inf"),
            )
            continue

        valid_ids = pixel_colony_ids[valid]
        unique_ids, counts = np.unique(valid_ids, return_counts=True)
        total_valid = int(counts.sum())

        # Majority vote
        best_idx = np.argmax(counts)
        best_colony = int(unique_ids[best_idx])
        majority_frac = float(counts[best_idx]) / total_valid

        # Bridge detection: check if any minority exceeds threshold
        is_bridge = False
        if len(unique_ids) > 1:
            minority_frac = 1.0 - majority_frac
            is_bridge = minority_frac > bridge_threshold

        mean_cost = float(np.mean(pixel_costs[valid]))

        assignments[fid] = FragmentAssignment(
            fragment_id=fid,
            colony_id=best_colony,
            is_bridge=is_bridge,
            majority_fraction=majority_frac,
            mean_cost=mean_cost,
        )

    return assignments


def backtrack_path(
    seed_row: int,
    seed_col: int,
    predecessor: np.ndarray,
    cost_distance: np.ndarray,
    cost_surface: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Follow predecessor pointers from seed to colony (cost_distance==0).

    Args:
        seed_row: Starting row (typically the min-cost pixel within a
            fragment).
        seed_col: Starting column.
        predecessor: Int32 (H, W) flat-index predecessor map.
        cost_distance: Float64 (H, W) cost-distance map.
        cost_surface: Float32 (H, W) for recording cost profile.

    Returns:
        Tuple of (coords, cost_profile) where coords is (N, 2) int32
        array of (row, col) and cost_profile is (N,) float64 array,
        or None if backtracking fails (e.g., seed is unreached).
    """
    W = predecessor.shape[1]
    max_steps = predecessor.shape[0] * predecessor.shape[1]  # safety limit

    if cost_distance[seed_row, seed_col] == np.inf:
        return None

    path_r = [seed_row]
    path_c = [seed_col]
    costs = [cost_distance[seed_row, seed_col]]

    r, c = seed_row, seed_col
    steps = 0
    while cost_distance[r, c] > 0.0:
        pred_flat = predecessor[r, c]
        if pred_flat < 0:
            break
        r = pred_flat // W
        c = pred_flat % W
        path_r.append(r)
        path_c.append(c)
        costs.append(cost_distance[r, c])
        steps += 1
        if steps > max_steps:
            return None  # cycle guard

    coords = np.column_stack([path_r, path_c]).astype(np.int32)
    cost_profile = np.array(costs, dtype=np.float64)
    return coords, cost_profile


def extract_fragment_paths(
    fragment_labels: np.ndarray,
    assignments: dict[int, FragmentAssignment],
    dijkstra: DijkstraResult,
    cost_surface: np.ndarray,
) -> tuple[dict[int, FragmentPath], list[int]]:
    """Extract minimum-cost paths from each fragment back to its colony.

    For each assigned fragment, finds the pixel with minimum cost-distance
    and backtracks to the colony boundary.

    Args:
        fragment_labels: Int32 (H, W) labeled fragment mask.
        assignments: Fragment-to-colony assignments.
        dijkstra: Dijkstra propagation result.
        cost_surface: Float32 (H, W) cost map.

    Returns:
        Tuple of (paths_dict, unconnected_ids) where paths_dict maps
        fragment_id to FragmentPath and unconnected_ids lists fragments
        that failed path extraction.
    """
    paths: dict[int, FragmentPath] = {}
    unconnected: list[int] = []

    for fid, assign in assignments.items():
        if assign.colony_id < 0:
            unconnected.append(fid)
            continue

        mask = fragment_labels == fid
        rows, cols = np.where(mask)
        frag_costs = dijkstra.cost_distance[rows, cols]

        # Find minimum-cost pixel in this fragment
        local_min = np.argmin(frag_costs)
        seed_r, seed_c = int(rows[local_min]), int(cols[local_min])

        if frag_costs[local_min] == np.inf:
            unconnected.append(fid)
            continue

        result = backtrack_path(
            seed_r,
            seed_c,
            dijkstra.predecessor,
            dijkstra.cost_distance,
            cost_surface,
        )

        if result is None:
            unconnected.append(fid)
            continue

        coords, cost_profile = result
        paths[fid] = FragmentPath(
            fragment_id=fid,
            colony_id=assign.colony_id,
            coords=coords,
            cost_profile=cost_profile,
            total_cost=float(cost_profile[0]),
            path_length=len(coords),
        )

    return paths, unconnected


def assemble_connected_mask(
    colony_labels: np.ndarray,
    fragment_labels: np.ndarray,
    assignments: dict[int, FragmentAssignment],
    paths: dict[int, FragmentPath],
) -> np.ndarray:
    """Paint connected fragments and their paths into the colony mask.

    Each connected fragment and its path pixels receive the colony label
    of the assigned colony, extending the colony territory.

    Args:
        colony_labels: Int32 (H, W) original colony label mask.
        fragment_labels: Int32 (H, W) fragment label mask.
        assignments: Fragment-to-colony assignments.
        paths: Fragment-to-colony paths.

    Returns:
        Int32 (H, W) extended label mask with colonies, connected
        fragments, and path pixels labeled by colony ID.
    """
    connected = colony_labels.copy()

    for fid, path in paths.items():
        assign = assignments[fid]
        cid = assign.colony_id

        # Paint fragment pixels
        frag_mask = fragment_labels == fid
        connected[frag_mask] = cid

        # Paint path pixels
        rows = path.coords[:, 0]
        cols = path.coords[:, 1]
        connected[rows, cols] = cid

    return connected
