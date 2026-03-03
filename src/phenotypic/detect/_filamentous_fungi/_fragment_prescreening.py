"""Fragment pre-screening against the composite cost surface.

Rapidly identifies fragments surrounded by high-cost regions that have
no viable path corridor to any colony.  These fragments are rejected
before Dijkstra propagation, saving significant computation.

The approach uses ``scipy.ndimage.minimum_filter`` to precompute the
local minimum cost within *r_screen* pixels of every pixel in a single
O(M) pass.  Each fragment's boundary pixels are then checked against
this pre-filtered map.  If no boundary pixel sees a low-cost region
within *r_screen*, no Dijkstra path through that fragment can succeed.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter
from skimage.segmentation import find_boundaries

from ._dataclasses import PrescreenResult


def compute_min_cost_envelope(
    cost_surface: np.ndarray,
    r_screen: int,
) -> np.ndarray:
    """Precompute local minimum cost within *r_screen* of every pixel.

    Applies a minimum filter with kernel size ``2 * r_screen + 1`` to the
    cost surface.  The output value at each pixel is the minimum cost
    found within an *r_screen*-pixel square neighbourhood.

    Args:
        cost_surface: Composite cost array, shape ``(H, W)``, values > 0.
        r_screen: Screening radius in pixels.  Should match or slightly
            exceed the maximum biologically plausible gap distance.
            Typical range: 25--40 px.

    Returns:
        Minimum-filtered cost array, same shape as input.  Value at
        ``(y, x)`` is the minimum cost within *r_screen* pixels.
    """
    kernel_size = 2 * r_screen + 1
    return minimum_filter(cost_surface, size=kernel_size)


def calibrate_screening_threshold(
    cost_surface: np.ndarray,
    colony_branch_mask: np.ndarray,
    r_screen: int = 30,
    percentile: float = 99.0,
) -> tuple[float, np.ndarray]:
    """Derive the screening threshold from known-good branch endpoints.

    Identifies pixels at the outer boundary of connected colonies in the
    inoculum+branch mask and computes their local minimum cost
    environment.  The threshold is set at the specified percentile of
    this distribution, ensuring that fragments in environments
    comparable to known-good branches are retained.

    This should be run once per image (or per imaging session if
    conditions are stable) before calling :func:`prescreen_fragments`.

    Args:
        cost_surface: Composite cost array, shape ``(H, W)``.
        colony_branch_mask: Inoculum+branch mask, shape ``(H, W)``.
            Nonzero values indicate colony pixels.
        r_screen: Screening radius, same value as will be used in
            :func:`prescreen_fragments`.
        percentile: Threshold percentile.  Default 99.0 (very
            permissive: only the worst 1% of known-good environments
            would be rejected).

    Returns:
        Tuple of ``(tau_screen, calibration_values)`` where
        *tau_screen* is the derived threshold and
        *calibration_values* is the array of per-boundary-pixel
        minimum costs used to compute it.  The *calibration_values*
        can be passed to :func:`prescreen_fragments` or used for
        diagnostic plotting.

    Raises:
        ValueError: If no boundary pixels are found in
            *colony_branch_mask*.
    """
    cost_for_filter = cost_surface.copy()
    colony_pixels = colony_branch_mask > 0
    non_colony_mask = ~colony_pixels

    if np.any(non_colony_mask):
        fill_value = float(np.percentile(cost_surface[non_colony_mask], 50))
    else:
        fill_value = 1.0

    cost_for_filter[colony_pixels] = fill_value
    min_cost_envelope = compute_min_cost_envelope(cost_for_filter, r_screen)

    # Outer boundary of the colony mask: pixels at tips/edges of
    # known-good branches representing worst-case connected environment.
    outer_boundary = find_boundaries(colony_pixels, mode="outer")

    boundary_coords = np.argwhere(outer_boundary)
    if boundary_coords.size == 0:
        raise ValueError(
            "No boundary pixels found in colony_branch_mask. "
            "Is the mask empty?"
        )

    rows, cols = boundary_coords[:, 0], boundary_coords[:, 1]
    calibration_values = min_cost_envelope[rows, cols].astype(np.float64)

    tau_screen = float(np.percentile(calibration_values, percentile))

    return tau_screen, calibration_values


def _extract_fragment_boundaries(
    fragment_labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Extract boundary pixel coordinates for each labelled fragment.

    Boundary pixels are those within the fragment mask that have at
    least one 8-connected neighbour outside the fragment (including
    neighbours belonging to other fragments).

    For very small fragments (< 4 pixels), all pixels are treated as
    boundary pixels since they are entirely "surface."

    Args:
        fragment_labels: Labelled fragment array, shape ``(H, W)``.
            0 = background, positive integers = fragment IDs.

    Returns:
        Dict mapping ``fragment_id`` to an ``(N, 2)`` array of
        ``(row, col)`` boundary pixel coordinates.
    """
    fragment_ids = np.unique(fragment_labels)
    fragment_ids = fragment_ids[fragment_ids > 0]

    boundaries: dict[int, np.ndarray] = {}
    for fid in fragment_ids:
        mask = fragment_labels == fid
        pixel_count = np.count_nonzero(mask)

        if pixel_count < 4:
            # Very small fragments: all pixels are boundary.
            coords = np.argwhere(mask)
        else:
            boundary_mask = find_boundaries(mask, mode="inner")
            coords = np.argwhere(boundary_mask)

            # Fallback: if find_boundaries returns nothing (single-pixel
            # fragments that inner mode misses), use all pixels.
            if coords.size == 0:
                coords = np.argwhere(mask)

        boundaries[int(fid)] = coords

    return boundaries


def prescreen_fragments(
    cost_surface: np.ndarray,
    fragment_labels: np.ndarray,
    r_screen: int = 30,
    tau_screen: float | None = None,
    calibration_cost_values: np.ndarray | None = None,
    calibration_percentile: float = 99.0,
    colony_branch_mask: np.ndarray | None = None,
) -> PrescreenResult:
    """Pre-screen fragments against the local cost environment.

    Rejects fragments surrounded entirely by high-cost regions before
    Dijkstra propagation.  Uses a minimum filter over the cost surface
    to efficiently check whether any low-cost corridor exists within
    *r_screen* pixels of each fragment boundary.

    The threshold *tau_screen* can be set explicitly or derived from
    calibration data (known-good fragment boundary cost values).  If
    calibration data is provided, the threshold is set at the specified
    percentile, making screening very permissive (only reject clearly
    hopeless cases).

    Args:
        cost_surface: Composite cost array, shape ``(H, W)``, values > 0.
            Output of cost surface construction, with known objects
            already masked to epsilon_free.
        fragment_labels: Labelled fragment array, shape ``(H, W)``.
            0 = background, positive integers = fragment IDs.  Fragments
            already connected to colonies (in the inoculum+branch mask)
            should *not* be included here.
        r_screen: Screening radius in pixels.  Maximum gap distance
            considered biologically plausible.  Typical: 25--40 px.
        tau_screen: Cost threshold for rejection.  Fragments whose
            minimum environmental cost exceeds this are rejected.  If
            ``None``, *calibration_cost_values* must be provided.
        calibration_cost_values: Array of minimum environmental cost
            values from known-good fragments (those already in the
            inoculum+branch mask).  Used to derive *tau_screen* at the
            specified percentile.  Ignored if *tau_screen* is set.
        calibration_percentile: Percentile of calibration distribution
            to use as threshold.  Default 99.0 (very permissive).
        colony_branch_mask: If provided, the inoculum+branch mask (any
            nonzero value = colony).  Used to exclude colony pixels from
            the minimum-filtered cost map so that fragment screening
            reflects the gap environment, not the free-traversal zone
            inside colonies.  Shape ``(H, W)``.

    Returns:
        :class:`PrescreenResult` containing the filtered fragment
        labels, sets of passed/rejected IDs, and the threshold used.

    Raises:
        ValueError: If neither *tau_screen* nor
            *calibration_cost_values* is provided, if
            *fragment_labels* contains no fragments, or if the shapes
            of *cost_surface* and *fragment_labels* do not match.
    """
    # --- Input validation ---
    if cost_surface.shape != fragment_labels.shape:
        raise ValueError(
            f"Shape mismatch: cost_surface {cost_surface.shape} vs "
            f"fragment_labels {fragment_labels.shape}"
        )

    fragment_ids = np.unique(fragment_labels)
    fragment_ids = fragment_ids[fragment_ids > 0]
    if fragment_ids.size == 0:
        raise ValueError("fragment_labels contains no labeled fragments")

    if tau_screen is None and calibration_cost_values is None:
        raise ValueError(
            "Must provide either tau_screen or calibration_cost_values"
        )

    # --- Threshold determination ---
    if tau_screen is None:
        assert calibration_cost_values is not None  # guarded above
        tau_screen = float(
            np.percentile(calibration_cost_values, calibration_percentile)
        )

    # --- Precompute minimum cost envelope ---
    # If colony mask provided, temporarily set colony pixels to median
    # background cost so the minimum filter reflects gap environment,
    # not free-traversal zones.
    if colony_branch_mask is not None:
        cost_for_filter = cost_surface.copy()
        colony_pixels = colony_branch_mask > 0
        fill_value = np.percentile(cost_surface[~colony_pixels], 50)
        cost_for_filter[colony_pixels] = fill_value
    else:
        cost_for_filter = cost_surface

    min_cost_envelope = compute_min_cost_envelope(cost_for_filter, r_screen)

    # --- Extract fragment boundaries ---
    boundaries = _extract_fragment_boundaries(fragment_labels)

    # --- Screen each fragment ---
    passed: list[int] = []
    rejected: list[int] = []

    for fid in fragment_ids:
        fid_int = int(fid)
        if fid_int not in boundaries:
            # Fragment had no extractable boundary (should not happen).
            rejected.append(fid_int)
            continue

        coords = boundaries[fid_int]
        rows, cols = coords[:, 0], coords[:, 1]

        # Sample the minimum-filtered cost at boundary pixels.
        # The value at each boundary pixel already represents the
        # minimum cost within r_screen pixels of that point.
        boundary_min_costs = min_cost_envelope[rows, cols]

        # The fragment's environmental cost is the best (lowest)
        # value across all its boundary pixels.  This asks: "is there
        # ANY low-cost region within r_screen of ANY part of this
        # fragment's boundary?"
        frag_min_env = float(np.min(boundary_min_costs))

        if frag_min_env > tau_screen:
            rejected.append(fid_int)
        else:
            passed.append(fid_int)

    # --- Build output ---
    screened_labels = fragment_labels.copy()
    if rejected:
        rejection_mask = np.isin(fragment_labels, rejected)
        screened_labels[rejection_mask] = 0

    return PrescreenResult(
        screened_fragment_labels=screened_labels,
        passed_ids=set(passed),
        rejected_ids=set(rejected),
        threshold_used=tau_screen,
    )
