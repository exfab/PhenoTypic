"""Cost surface construction for Dijkstra-based hyphal branch reconnection.

Builds a composite cost surface from phase congruency features and local
texture statistics. Low cost indicates likely hyphal structure; high cost
indicates bare agar or noise. The cost surface drives shortest-path routing
that reconnects fragmented branches to their parent colonies.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter, uniform_filter


def compute_anisotropy(M: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Compute fractional anisotropy from phase congruency moment maps.

    Args:
        M: Maximum moment of phase congruency covariance (edge strength).
        m: Minimum moment of phase congruency covariance (corner strength).

    Returns:
        Fractional anisotropy in [0, 1] as float32. Values near 1 indicate
        strongly oriented structure (e.g. a hypha), values near 0 indicate
        isotropic texture (e.g. uniform agar background).
    """
    return ((M - m) / (M + m + 1e-6)).astype(np.float32)


def compute_orientation_coherence(
    theta: np.ndarray, r_coh: int = 12
) -> np.ndarray:
    """Compute local orientation coherence from phase congruency orientations.

    Measures the circular mean resultant length of doubled orientations over
    a square neighbourhood. High coherence indicates consistently oriented
    structure such as a hypha traversing the window; low coherence indicates
    disorganised texture or bare agar.

    Args:
        theta: Orientation map in radians from phase congruency.
        r_coh: Radius of the square averaging kernel. The kernel side length
            is ``2 * r_coh + 1``.

    Returns:
        Orientation coherence in [0, 1] as float32. Values near 1 indicate
        strong local alignment of features.
    """
    kernel_size = 2 * r_coh + 1
    cos2 = np.cos(2.0 * theta).astype(np.float32)
    sin2 = np.sin(2.0 * theta).astype(np.float32)
    c_bar = uniform_filter(cos2, size=kernel_size, mode="reflect")
    s_bar = uniform_filter(sin2, size=kernel_size, mode="reflect")
    return np.sqrt(c_bar**2 + s_bar**2).astype(np.float32)


def compute_local_mad_map(
    image_ced: np.ndarray, window_size: int = 7
) -> np.ndarray:
    """Compute a local median absolute deviation (MAD) map.

    Two-pass median filter implementation: the first pass computes the local
    median, then absolute deviations from that median are themselves
    median-filtered to yield a robust local spread estimate. High MAD
    indicates textured regions (e.g. hyphal networks); low MAD indicates
    smooth agar background.

    Args:
        image_ced: 2-D image array, typically coherence-enhanced diffusion
            output.
        window_size: Side length of the square median filter kernel. Must be
            odd.

    Returns:
        Local MAD map as float32.

    Raises:
        ValueError: If *image_ced* is not 2-D or *window_size* is even.
    """
    if image_ced.ndim != 2:
        raise ValueError(
            f"image_ced must be 2-D, got {image_ced.ndim}-D array."
        )
    if window_size % 2 == 0:
        raise ValueError(
            f"window_size must be odd, got {window_size}."
        )
    image = image_ced.astype(np.float32, copy=False)
    local_median = median_filter(image, size=window_size)
    abs_deviation = np.abs(image - local_median)
    return median_filter(abs_deviation, size=window_size).astype(np.float32)


def assemble_composite_cost(
    pct_energy: np.ndarray,
    anisotropy: np.ndarray,
    orientation_coherence: np.ndarray,
    mad_map: np.ndarray,
    beta: float = 2.0,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Assemble the composite cost surface from feature maps.

    Combines phase congruency energy, anisotropy, orientation coherence, and
    local MAD into a single scalar cost field. Pixels with strong, coherent,
    anisotropic structure (likely hyphae) receive low cost; featureless agar
    receives high cost. The MAD term in the numerator penalises noisy regions
    that might otherwise appear low-cost.

    The formula is::

        C = (1 + gamma * MAD) / (P * A^beta * O_coh + eps)

    Args:
        pct_energy: Normalised phase congruency energy (pc_sum from
            phasecong3).
        anisotropy: Fractional anisotropy from :func:`compute_anisotropy`.
        orientation_coherence: Coherence from
            :func:`compute_orientation_coherence`.
        mad_map: Local MAD from :func:`compute_local_mad_map`.
        beta: Exponent on the anisotropy term. Higher values sharpen the
            preference for strongly oriented features.
        gamma: Weight of the MAD penalty in the numerator.
        eps: Small constant to avoid division by zero.

    Returns:
        Composite cost surface as float32. Lower values indicate likely
        hyphal structure.
    """
    denominator = pct_energy * (anisotropy**beta) * orientation_coherence + eps
    numerator = 1.0 + gamma * mad_map
    return (numerator / denominator).astype(np.float32)


def apply_structure_mask(
    cost: np.ndarray,
    colony_mask: np.ndarray,
    eps_free: float = 1e-6,
) -> np.ndarray:
    """Set cost to near-zero for pixels belonging to known colony structure.

    Pixels already identified as part of a colony (e.g. by a preceding
    detection step) are given minimal traversal cost so that Dijkstra paths
    pass through established structure for free.

    Args:
        cost: Composite cost surface from :func:`assemble_composite_cost`.
        colony_mask: Binary or labeled mask where positive values indicate
            known colony pixels.
        eps_free: Cost assigned to known-structure pixels. Should be small
            but positive to keep the cost surface strictly positive.

    Returns:
        Modified cost surface as float32. A copy is returned; *cost* is not
        modified in place.
    """
    c_final = cost.astype(np.float32, copy=True)
    c_final[colony_mask > 0] = eps_free
    return c_final


def apply_border_penalty(
    cost: np.ndarray,
    edge_margin: int = 50,
) -> np.ndarray:
    """Apply a linear ramp penalty near image borders to prevent edge-routing.

    Args:
        cost: Float32 (H, W) cost surface.
        edge_margin: Width in pixels of the penalized border zone.

    Returns:
        Cost surface with border penalty applied. A copy is returned.
    """
    H, W = cost.shape
    max_cost = float(cost.max())

    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    dist_to_edge = np.minimum(
        np.minimum(rows, H - 1 - rows),
        np.minimum(cols, W - 1 - cols),
    ).astype(np.float32)

    in_margin = dist_to_edge < edge_margin
    ramp = max_cost * (1.0 - dist_to_edge / edge_margin)
    result = cost.copy()
    result[in_margin] = np.maximum(cost[in_margin], ramp[in_margin])
    return result


def apply_distance_gap_penalty(
    cost: np.ndarray,
    pct_energy: np.ndarray,
    colony_labels: np.ndarray,
    alpha: float = 5.0,
) -> np.ndarray:
    """Penalize gap-crossing proportionally to distance from colonies.

    Near colonies, gaps in PCT energy are cheap (branch crossover is common).
    Far from colonies, gaps are expensive (growth is radial, gaps indicate noise).

    Args:
        cost: Float32 (H, W) cost surface.
        pct_energy: Float (H, W) PCT energy map, range [0, 1].
        colony_labels: Int32 (H, W) labeled colony mask. 0 is background.
        alpha: Penalty strength. Higher values impose stronger distance gating.

    Returns:
        Cost surface with distance-weighted gap penalty applied. A copy is
        returned.
    """
    colony_mask = colony_labels > 0
    dist = np.asarray(distance_transform_edt(~colony_mask), dtype=np.float32)
    d_norm = dist / (dist.max() + 1e-6)

    gap = (1.0 - np.clip(pct_energy, 0, 1)).astype(np.float32)
    penalty = 1.0 + alpha * d_norm * gap

    return (cost * penalty).astype(np.float32)


def _apply_distance_gap_penalty_inplace(
    cost: np.ndarray,
    pct_energy: np.ndarray,
    colony_labels: np.ndarray,
    alpha: float = 5.0,
) -> None:
    """In-place distance-gap penalty. Mutates *cost*.

    Equivalent to :func:`apply_distance_gap_penalty` but avoids
    allocating a new array.

    Args:
        cost: Float32 (H, W) cost surface, modified in place.
        pct_energy: Float (H, W) PCT energy map, range [0, 1].
        colony_labels: Int32 (H, W) labeled colony mask. 0 is background.
        alpha: Penalty strength.
    """
    colony_mask = colony_labels > 0
    dist = np.asarray(distance_transform_edt(~colony_mask), dtype=np.float32)
    d_norm = dist / (dist.max() + 1e-6)
    gap = (1.0 - np.clip(pct_energy, 0, 1)).astype(np.float32)
    penalty = 1.0 + alpha * d_norm * gap
    cost *= penalty


def _apply_border_penalty_inplace(
    cost: np.ndarray,
    edge_margin: int = 50,
) -> None:
    """In-place border penalty. Mutates *cost*.

    Equivalent to :func:`apply_border_penalty` but avoids allocating
    a new array.

    Args:
        cost: Float32 (H, W) cost surface, modified in place.
        edge_margin: Width in pixels of the penalized border zone.
    """
    H, W = cost.shape
    max_cost = float(cost.max())
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    dist_to_edge = np.minimum(
        np.minimum(rows, H - 1 - rows),
        np.minimum(cols, W - 1 - cols),
    ).astype(np.float32)
    in_margin = dist_to_edge < edge_margin
    ramp = max_cost * (1.0 - dist_to_edge / edge_margin)
    np.maximum(cost, ramp, out=cost, where=in_margin)


def _apply_structure_mask_inplace(
    cost: np.ndarray,
    colony_mask: np.ndarray,
    eps_free: float = 1e-6,
) -> None:
    """In-place structure mask. Mutates *cost*.

    Equivalent to :func:`apply_structure_mask` but avoids allocating
    a new array.

    Args:
        cost: Float32 (H, W) cost surface, modified in place.
        colony_mask: Binary or labeled mask where positive values indicate
            known colony pixels.
        eps_free: Cost assigned to known-structure pixels.
    """
    cost[colony_mask > 0] = eps_free
