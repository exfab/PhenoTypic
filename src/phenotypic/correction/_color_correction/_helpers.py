"""Private helper functions for color correction pipeline.

Provides image filtering, border detection, run-length analysis, checker card
centering/padding, swatch mask generation, Hungarian matching, geometric
median computation, background trimming, core mask extraction, and patch
shape validation.
"""

from __future__ import annotations

from typing import Literal

import colour
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment

_SRGB_CS = colour.RGB_COLOURSPACES["sRGB"]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize_to_unit_float(image: np.ndarray) -> np.ndarray:
    """Normalize an RGB image to ``[0, 1]`` float64.

    Handles uint8 (÷255), uint16 (÷65535), and arbitrary integer ranges.
    Float images already in ``[0, 1]`` are returned as-is (cast to float64).

    Args:
        image: RGB array with shape ``(H, W, 3)`` or similar.

    Returns:
        Float64 array in ``[0, 1]``.
    """
    img_float = image.astype(np.float64)
    if img_float.max() <= 1.0:
        return img_float
    if image.dtype == np.uint8:
        return img_float / 255.0
    if image.dtype == np.uint16:
        return img_float / 65535.0
    return img_float / img_float.max()


def _rgb_to_lab(rgb_float: np.ndarray, illuminant_xy: np.ndarray | None = None) -> np.ndarray:
    """Convert a ``[0, 1]`` sRGB float array to CIE Lab.

    Args:
        rgb_float: sRGB array in ``[0, 1]``.
        illuminant_xy: CIE xy chromaticity of the white point.  Defaults to
            the sRGB (D65) whitepoint.

    Returns:
        Lab array with the same spatial dimensions.
    """
    if illuminant_xy is None:
        illuminant_xy = _SRGB_CS.whitepoint
    XYZ = colour.RGB_to_XYZ(rgb_float, colourspace=_SRGB_CS, apply_cctf_decoding=True)
    return colour.XYZ_to_Lab(XYZ, illuminant=illuminant_xy)


# ---------------------------------------------------------------------------
# Image filtering
# ---------------------------------------------------------------------------


def median_filter_rgb(image: np.ndarray, size: int | tuple[int, int] = 3) -> np.ndarray:
    """Apply a median filter independently to each RGB channel.

    Args:
        image: RGB image as ``(H, W, 3)`` NumPy array.
        size: Median filter kernel size. Integer or 2-D tuple.

    Returns:
        Filtered RGB image with the same shape and dtype as *image*.

    Raises:
        ValueError: If *image* is not a 3-channel array.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be an RGB image with shape (H, W, 3).")

    return np.stack(
        [median_filter(image[..., c], size=size) for c in range(3)],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Cross-channel standard deviation for border detection
# ---------------------------------------------------------------------------


def find_cross_channel_stddev_magnitude(
    lab_image: np.ndarray,
    axis: Literal[0, 1],
    filter_size: int = 10,
    return_stddev: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute per-row or per-column stddev magnitude across L, a, b channels.

    A median filter is applied before computing standard deviations to reduce
    noise.  The magnitude is the Euclidean norm of the per-channel stddev
    vectors, useful for detecting uniform border regions (low magnitude) vs.
    swatch interiors (high magnitude).

    Args:
        lab_image: CIE Lab image array with shape ``(H, W, 3)``.
        axis: Reduction axis.  ``0`` computes stddev across rows (one value
            per column), ``1`` computes stddev across columns (one value per
            row).
        filter_size: Kernel size for the per-channel median pre-filter.
        return_stddev: If ``True``, also return the ``(3, N)`` array of
            per-channel standard deviations.

    Returns:
        If *return_stddev* is ``False``, a 2-D array whose non-singleton
        dimension matches the axis **not** reduced (shape ``(1, W)`` when
        ``axis=0``, ``(H, 1)`` when ``axis=1``).  If ``True``, a tuple
        ``(magnitude, stddev_per_channel)`` where *stddev_per_channel* has
        shape ``(3, N)`` (``axis=0``) or ``(N, 3)`` (``axis=1``).

    Raises:
        ValueError: If *axis* is not 0 or 1.
    """
    if axis not in {0, 1}:
        raise ValueError("axis must be 0 or 1")

    lab_filtered = median_filter_rgb(lab_image, size=filter_size)

    stddev_per_channel: list[np.ndarray] = []
    for channel in range(lab_filtered.shape[-1]):
        stddev_per_channel.append(np.std(lab_filtered[..., channel], axis=axis))

    stddev_arr = np.vstack(stddev_per_channel)
    if axis:
        stddev_arr = stddev_arr.T

    # Shape the magnitude so it broadcasts with the original image along the
    # surviving spatial axis.
    mag_shape = (-1, 1) if axis else (1, -1)
    magnitude = np.sqrt(np.sum(stddev_arr ** 2, axis=axis)).reshape(*mag_shape)

    if return_stddev:
        return magnitude, stddev_arr
    return magnitude


# ---------------------------------------------------------------------------
# Run-length analysis on boolean / binary arrays
# ---------------------------------------------------------------------------


def get_run_of_ones_positions(
    binary_array: np.ndarray,
    return_lengths: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Find start/end positions of every contiguous run of True values.

    Args:
        binary_array: 1-D boolean or 0/1 array (squeezable arrays with a
            single non-singleton dimension are also accepted).
        return_lengths: If ``True``, also return the length of each run.

    Returns:
        A ``(N, 2)`` integer array of ``[start, end]`` positions (inclusive
        indices).  If *return_lengths* is ``True``, returns
        ``(positions, lengths)`` where *lengths* is a 1-D integer array of
        length N.  When no runs exist, *positions* is an empty array (and
        *lengths* an empty array when requested).
    """
    arr = np.asarray(binary_array)
    if arr.size == 0:
        empty = np.array([], dtype=int).reshape(0, 2)
        return (empty, np.array([], dtype=int)) if return_lengths else empty

    if arr.ndim != 1:
        if sum(dim != 1 for dim in arr.shape) == 1:
            arr = arr.ravel()
        else:
            raise ValueError("Input must be 1-D (or squeezable to 1-D).")

    # Pad with 0 at both ends to detect edges of runs.
    padded = np.r_[0, arr.astype(int), 0]
    diff = np.diff(padded)

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1  # inclusive end indices

    if starts.size == 0:
        empty = np.array([], dtype=int).reshape(0, 2)
        return (empty, np.array([], dtype=int)) if return_lengths else empty

    positions = np.column_stack([starts, ends])
    if return_lengths:
        lengths = ends - starts + 1
        return positions, lengths
    return positions


def longest_run_of_ones(
    binary_array: np.ndarray,
) -> tuple[int | None, int | None, int]:
    """Find the longest contiguous run of True / 1 values.

    Args:
        binary_array: 1-D boolean or 0/1 array.

    Returns:
        ``(start, end, length)`` of the longest run.  *start* and *end* are
        inclusive indices.  If no run exists, returns ``(None, None, 0)``.

    Raises:
        ValueError: If input is not 1-D.
    """
    result = get_run_of_ones_positions(binary_array, return_lengths=True)
    positions, lengths = result  # type: ignore[misc]

    if positions.size == 0:
        return None, None, 0

    best = int(np.argmax(lengths))
    return int(positions[best, 0]), int(positions[best, 1]), int(lengths[best])


# ---------------------------------------------------------------------------
# Checker card centering and padding
# ---------------------------------------------------------------------------


def center_and_pad_checker(
    checker_image: np.ndarray,
    filter_size: int = 20,
    stddev_mag_threshold: float = 15.0,
) -> np.ndarray:
    """Detect borders, center an asymmetric checker card, and reflect-pad.

    When a color checker card is only partially captured (e.g. the camera
    FOV clips one side), this function locates the central uniform border
    column, centres it within the image, and uses reflection padding to
    reconstruct the missing edge patches.

    The algorithm:
      1. Convert to Lab and compute cross-column stddev magnitude.
      2. Threshold to identify uniform (border) columns.
      3. Find the largest contiguous run of border columns — the centre
         divider.
      4. Compute the midpoint of that divider.
      5. Edge-pad the short side so the divider is centred.
      6. Reflect-pad both sides to rebuild missing swatch area.
      7. Edge-pad to restore the border width on each side.

    Args:
        checker_image: RGB image array with shape ``(H, W, 3)``.  Any
            numeric dtype is accepted.
        filter_size: Kernel size for the internal median filter passes.
        stddev_mag_threshold: Columns with cross-channel stddev magnitude
            at or below this value are classified as border.

    Returns:
        Padded and centred RGB image as a NumPy array.
    """
    if checker_image.ndim != 3 or checker_image.shape[2] != 3:
        raise ValueError("checker_image must have shape (H, W, 3).")

    # --- Convert to Lab for border analysis --------------------------------
    img_float = _normalize_to_unit_float(checker_image)
    lab_arr = _rgb_to_lab(img_float)

    # --- Detect border columns (axis=0 reduces rows) ----------------------
    stddev_mag = find_cross_channel_stddev_magnitude(
        lab_arr, axis=0, filter_size=filter_size
    )

    border_mask = np.asarray(stddev_mag).ravel() <= stddev_mag_threshold

    positions, lengths = get_run_of_ones_positions(border_mask, return_lengths=True)
    if positions.size == 0:
        # No border detected — return a median-filtered copy unchanged.
        return median_filter_rgb(checker_image.copy(), size=filter_size)

    # Assume the largest run is the centre divider.
    center_idx = int(np.argmax(lengths))
    center_start, center_end = int(positions[center_idx, 0]), int(positions[center_idx, 1])
    column_midpoint = (center_start + center_end) / 2.0

    H, W = checker_image.shape[:2]
    left_half = column_midpoint
    right_half = W - column_midpoint

    centering_pad = int(abs(left_half - right_half))

    # Work on the original-dtype image (preserve precision).
    rgb = median_filter_rgb(checker_image.copy(), size=filter_size)

    # Edge-pad the short side so the centre divider sits at the image midpoint.
    if centering_pad != 0:
        image_center = rgb.shape[1] // 2
        if column_midpoint > image_center:
            # More pixels on the left — pad right.
            pad_width = ((0, 0), (0, centering_pad), (0, 0))
        else:
            # More pixels on the right — pad left.
            pad_width = ((0, 0), (centering_pad, 0), (0, 0))

        rgb = np.pad(rgb, pad_width=pad_width, mode="edge")
        rgb = median_filter_rgb(rgb, size=filter_size)

    # Reflect-pad to rebuild the missing swatch area.
    reflect_pad = int(max(left_half, right_half))
    rgb = np.pad(
        rgb,
        pad_width=((0, 0), (reflect_pad, reflect_pad), (0, 0)),
        mode="reflect",
    )

    # Edge-pad to restore the border on each outer edge.
    border_width = int(column_midpoint - center_start)
    if border_width > 0:
        rgb = np.pad(
            rgb,
            pad_width=((0, 0), (border_width, border_width), (0, 0)),
            mode="edge",
        )
        rgb = median_filter_rgb(rgb, size=filter_size)

    return rgb


# ---------------------------------------------------------------------------
# Geometric median (Weiszfeld algorithm)
# ---------------------------------------------------------------------------


def geometric_median(
    points: np.ndarray,
    eps: float = 1e-3,
    max_iter: int = 20,
) -> np.ndarray:
    """Compute the geometric median of a point set via Weiszfeld's algorithm.

    The geometric median minimises the sum of Euclidean distances to all
    points.  This is a simple iterative re-weighting implementation suitable
    for small-to-moderate point sets such as the pixels within a single
    colour-checker swatch.

    Args:
        points: Array of shape ``(N, D)`` with *N* points in *D* dimensions.
        eps: Convergence tolerance.  Iteration stops when the update norm
            drops below *eps*, or when the current estimate coincides with a
            data point.
        max_iter: Maximum number of Weiszfeld iterations.

    Returns:
        1-D array of shape ``(D,)`` — the geometric median.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        return points.copy()
    if points.shape[0] == 1:
        return points[0].copy()

    guess = points.mean(axis=0)

    for _ in range(max_iter):
        distances = np.linalg.norm(points - guess, axis=1)

        # If the guess coincides with a data point, return it.
        if np.any(distances < eps):
            return points[distances.argmin()].copy()

        weights = 1.0 / np.clip(distances, eps, None)
        new_guess = np.average(points, axis=0, weights=weights)

        if np.linalg.norm(new_guess - guess) < eps:
            return new_guess
        guess = new_guess

    return guess


# ---------------------------------------------------------------------------
# Swatch mask generation via nearest-neighbour Lab assignment
# ---------------------------------------------------------------------------


def lab_checker_cluster_masks(
    lab_image: np.ndarray,
    ref_Lab: dict[str, tuple[float, float, float]],
    border_distance_threshold: float = 12.0,
    border_label: str = "border",
    roi_mask: np.ndarray | None = None,
    include_labels: bool = False,
) -> (
    tuple[list[np.ndarray], list[np.ndarray]]
    | tuple[list[np.ndarray], list[np.ndarray], list[str]]
):
    """Assign each pixel in a Lab checker image to the nearest reference swatch.

    For every pixel the Euclidean distance in Lab space to each reference
    colour is computed.  The pixel is assigned to the nearest reference if
    that distance is below *border_distance_threshold*; otherwise it is
    classified as border/background.

    Args:
        lab_image: Image in CIE Lab space with shape ``(H, W, 3)``.
        ref_Lab: Mapping of swatch identifiers to target Lab triplets.
        border_distance_threshold: Minimum Euclidean distance (Delta-E*ab)
            required to mark a pixel as border rather than assigning it to the
            closest swatch.
        border_label: Key used for the border mask in the output lists.  Must
            not collide with any key in *ref_Lab*.
        roi_mask: Optional boolean mask selecting pixels to consider.  Pixels
            outside receive ``False`` in all output masks.
        include_labels: If ``True``, a third list of label strings is
            returned alongside the masks and bounding boxes.

    Returns:
        A tuple ``(masks, bboxes)`` where *masks* is a list of boolean arrays
        with shape ``(H, W)`` and *bboxes* stores
        ``[min_row, min_col, max_row, max_col]`` for each mask.  The list
        order follows *ref_Lab* insertion order, with a final entry for
        *border_label*.  When *include_labels* is ``True``, returns
        ``(masks, bboxes, labels)``.

    Raises:
        ValueError: If input shapes are inconsistent or *border_label*
            clashes with a reference key.
    """
    if lab_image.ndim != 3 or lab_image.shape[-1] != 3:
        raise ValueError("lab_image must have shape (H, W, 3).")

    if border_label in ref_Lab:
        raise ValueError("border_label must not clash with reference swatch keys.")

    H, W, _ = lab_image.shape
    ref_names = list(ref_Lab.keys())
    ref_array = np.asarray(
        [ref_Lab[name] for name in ref_names], dtype=np.float64
    )
    if ref_array.ndim != 2 or ref_array.shape[1] != 3:
        raise ValueError("ref_Lab values must be Lab triplets of length 3.")

    if roi_mask is not None:
        if roi_mask.shape != (H, W):
            raise ValueError("roi_mask must match the spatial dimensions of lab_image.")
        roi_mask = roi_mask.astype(bool)
    else:
        roi_mask = np.ones((H, W), dtype=bool)

    lab_flat = lab_image.reshape(-1, 3).astype(np.float64)
    roi_flat = roi_mask.ravel()

    active_pixels = lab_flat[roi_flat]
    if active_pixels.size == 0:
        raise ValueError("roi_mask selects zero pixels; nothing to cluster.")

    # Compute distances in chunks to avoid O(N * R * 3) memory spike.
    from scipy.spatial.distance import cdist

    n_active = active_pixels.shape[0]
    chunk_size = max(1, min(n_active, 100_000))
    closest_idx = np.empty(n_active, dtype=int)
    min_dist = np.empty(n_active, dtype=np.float64)

    for start in range(0, n_active, chunk_size):
        end = min(start + chunk_size, n_active)
        dists = cdist(active_pixels[start:end], ref_array, metric="euclidean")
        closest_idx[start:end] = np.argmin(dists, axis=1)
        min_dist[start:end] = dists[np.arange(end - start), closest_idx[start:end]]

    border_active = min_dist > border_distance_threshold

    # Map back to full image raster.
    labels_flat = np.full(lab_flat.shape[0], fill_value=-1, dtype=int)
    labels_flat[roi_flat] = closest_idx

    borders_flat = np.zeros(lab_flat.shape[0], dtype=bool)
    borders_flat[roi_flat] = border_active

    # Border pixels must not be assigned to any swatch.
    labels_flat[borders_flat] = -1

    label_image_2d = labels_flat.reshape(H, W)
    border_mask_2d = borders_flat.reshape(H, W)

    def _mask_bbox(mask: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return np.array([0, 0, 0, 0], dtype=int)
        return np.array([ys.min(), xs.min(), ys.max() + 1, xs.max() + 1], dtype=int)

    mask_list: list[np.ndarray] = []
    bbox_list: list[np.ndarray] = []
    label_list: list[str] = []

    for idx, name in enumerate(ref_names):
        swatch_mask = label_image_2d == idx
        mask_list.append(swatch_mask)
        bbox_list.append(_mask_bbox(swatch_mask))
        label_list.append(name)

    mask_list.append(border_mask_2d)
    bbox_list.append(_mask_bbox(border_mask_2d))
    label_list.append(border_label)

    if include_labels:
        return mask_list, bbox_list, label_list
    return mask_list, bbox_list


# ---------------------------------------------------------------------------
# Hungarian matching on Delta-E 2000 cost matrix
# ---------------------------------------------------------------------------


def hungarian_match_swatches(
    observed_Lab: np.ndarray,
    ref_Lab: dict[str, tuple[float, float, float]],
) -> dict[str, int]:
    """Match observed swatch colours to reference colours via Hungarian algorithm.

    Builds a cost matrix of CIE Delta-E 2000 values between every observed
    swatch and every reference colour, then solves the linear-sum assignment
    problem to find the minimum-cost one-to-one mapping.

    Args:
        observed_Lab: Array of shape ``(K, 3)`` with *K* observed swatch Lab
            values (e.g. geometric medians of detected patches).
        ref_Lab: Mapping of reference swatch names to Lab triplets.

    Returns:
        Dictionary mapping reference swatch name to the row index in
        *observed_Lab* that was assigned to it.

    Raises:
        ValueError: If *observed_Lab* has wrong shape or is empty.
    """
    observed_Lab = np.asarray(observed_Lab, dtype=np.float64)
    if observed_Lab.ndim != 2 or observed_Lab.shape[1] != 3:
        raise ValueError("observed_Lab must have shape (K, 3).")

    ref_names = list(ref_Lab.keys())
    ref_array = np.asarray([ref_Lab[n] for n in ref_names], dtype=np.float64)

    K = observed_Lab.shape[0]
    R = ref_array.shape[0]
    cost = np.zeros((K, R), dtype=np.float64)

    for i in range(K):
        cost[i] = colour.difference.delta_E_CIE2000(
            observed_Lab[i], ref_array
        )

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping: dict[str, int] = {}
    for ri, ci in zip(row_ind, col_ind):
        mapping[ref_names[ci]] = int(ri)
    return mapping


# ---------------------------------------------------------------------------
# Background edge trimming
# ---------------------------------------------------------------------------


def trim_background_edges(
    image: np.ndarray,
    n_edge_pixels: int = 10,
    variance_threshold: float = 5.0,
) -> np.ndarray:
    """Trim uniform background strips from the outer edges of an image.

    Examines the outermost *n_edge_pixels* rows and columns. For each edge
    (top, bottom, left, right), per-row or per-column variance is computed
    in Lab space.  If the mean variance of the strip is below
    *variance_threshold*, the strip is trimmed.  Trimming proceeds
    iteratively from each edge inward one strip at a time.

    Args:
        image: RGB image array with shape ``(H, W, 3)``.
        n_edge_pixels: Width of the strip examined on each edge per
            iteration.
        variance_threshold: Maximum mean Lab variance for a strip to be
            considered uniform background.

    Returns:
        Trimmed image (may be the same array if no trimming was performed).

    Raises:
        ValueError: If *image* is not a 3-channel array.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3).")

    # Convert once to Lab.
    img_float = _normalize_to_unit_float(image)
    lab = _rgb_to_lab(img_float)

    H, W, _ = lab.shape
    top = 0
    bottom = H
    left = 0
    right = W

    def _strip_variance(strip: np.ndarray) -> float:
        """Mean per-pixel variance across Lab channels."""
        # strip shape: (rows, cols, 3) — variance per channel, averaged.
        return float(np.mean(np.var(strip.reshape(-1, 3), axis=0)))

    # Trim top
    while (top + n_edge_pixels) < bottom:
        strip = lab[top : top + n_edge_pixels, left:right, :]
        if _strip_variance(strip) < variance_threshold:
            top += n_edge_pixels
        else:
            break

    # Trim bottom
    while (bottom - n_edge_pixels) > top:
        strip = lab[bottom - n_edge_pixels : bottom, left:right, :]
        if _strip_variance(strip) < variance_threshold:
            bottom -= n_edge_pixels
        else:
            break

    # Trim left
    while (left + n_edge_pixels) < right:
        strip = lab[top:bottom, left : left + n_edge_pixels, :]
        if _strip_variance(strip) < variance_threshold:
            left += n_edge_pixels
        else:
            break

    # Trim right
    while (right - n_edge_pixels) > left:
        strip = lab[top:bottom, right - n_edge_pixels : right, :]
        if _strip_variance(strip) < variance_threshold:
            right -= n_edge_pixels
        else:
            break

    return image[top:bottom, left:right, :]


# ---------------------------------------------------------------------------
# Core mask extraction
# ---------------------------------------------------------------------------


def compute_core_mask(
    patch_mask: np.ndarray,
    core_fraction: float = 0.5,
) -> np.ndarray:
    """Keep only the inner portion of a boolean patch mask.

    Computes the centroid of the ``True`` pixels, then determines the maximum
    distance from centroid to any boundary pixel.  Pixels whose distance from
    the centroid exceeds ``core_fraction * max_distance`` are excluded.

    This is useful for discarding noisy edge pixels in a detected swatch
    region, retaining only the perceptually stable core.

    Args:
        patch_mask: 2-D boolean array where ``True`` marks the patch.
        core_fraction: Fraction of the centroid-to-boundary distance to
            keep.  ``0.5`` retains the inner half.  Values should be in
            ``(0, 1]``.

    Returns:
        Boolean array of the same shape as *patch_mask* with only the core
        pixels set to ``True``.

    Raises:
        ValueError: If *patch_mask* is not 2-D or *core_fraction* is out of
            range.
    """
    if patch_mask.ndim != 2:
        raise ValueError("patch_mask must be a 2-D array.")
    if not (0.0 < core_fraction <= 1.0):
        raise ValueError("core_fraction must be in (0, 1].")

    ys, xs = np.nonzero(patch_mask)
    if ys.size == 0:
        return np.zeros_like(patch_mask, dtype=bool)

    centroid_y = float(np.mean(ys))
    centroid_x = float(np.mean(xs))

    distances = np.sqrt((ys - centroid_y) ** 2 + (xs - centroid_x) ** 2)
    max_distance = distances.max()

    if max_distance == 0:
        # Single-pixel mask — the core is the pixel itself.
        return patch_mask.copy()

    threshold = core_fraction * max_distance

    core = np.zeros_like(patch_mask, dtype=bool)
    within = distances <= threshold
    core[ys[within], xs[within]] = True
    return core


# ---------------------------------------------------------------------------
# Patch shape validation
# ---------------------------------------------------------------------------


def validate_patch_shape(
    patch_mask: np.ndarray,
    expected_area_range: tuple[int, int] = (100, 50_000),
    aspect_ratio_range: tuple[float, float] = (0.3, 3.0),
) -> tuple[bool, list[str]]:
    """Check that a detected patch mask has a plausible shape.

    Validates the mask based on area, bounding-box aspect ratio,
    compactness (area / convex-hull area), and whether the mask touches the
    image boundary.

    Args:
        patch_mask: 2-D boolean array marking the patch region.
        expected_area_range: ``(min_area, max_area)`` in pixels.  A mask
            outside this range is flagged.
        aspect_ratio_range: ``(min_ratio, max_ratio)`` for the bounding-box
            width / height ratio.

    Returns:
        ``(is_valid, warnings)`` where *is_valid* is ``True`` when all
        checks pass and *warnings* is a list of human-readable strings
        describing each failed check.

    Raises:
        ValueError: If *patch_mask* is not 2-D.
    """
    if patch_mask.ndim != 2:
        raise ValueError("patch_mask must be a 2-D array.")

    warnings_list: list[str] = []
    H, W = patch_mask.shape

    ys, xs = np.nonzero(patch_mask)
    area = ys.size

    # --- Area check -------------------------------------------------------
    min_area, max_area = expected_area_range
    if area < min_area:
        warnings_list.append(
            f"Area {area} is below minimum {min_area}."
        )
    elif area > max_area:
        warnings_list.append(
            f"Area {area} exceeds maximum {max_area}."
        )

    if area == 0:
        warnings_list.append("Patch mask is empty.")
        return False, warnings_list

    # --- Bounding-box aspect ratio ----------------------------------------
    min_row, max_row = int(ys.min()), int(ys.max())
    min_col, max_col = int(xs.min()), int(xs.max())
    bb_height = max_row - min_row + 1
    bb_width = max_col - min_col + 1

    if bb_height == 0:
        bb_height = 1
    aspect_ratio = bb_width / bb_height

    lo, hi = aspect_ratio_range
    if aspect_ratio < lo or aspect_ratio > hi:
        warnings_list.append(
            f"Aspect ratio {aspect_ratio:.2f} outside expected range "
            f"[{lo:.2f}, {hi:.2f}]."
        )

    # --- Compactness (fill ratio of bounding box) -------------------------
    bb_area = bb_height * bb_width
    compactness = area / bb_area
    if compactness < 0.5:
        warnings_list.append(
            f"Compactness {compactness:.2f} is low (< 0.50); patch may be "
            "fragmented or non-convex."
        )

    # --- Boundary contact -------------------------------------------------
    touches_top = bool(np.any(patch_mask[0, :]))
    touches_bottom = bool(np.any(patch_mask[-1, :]))
    touches_left = bool(np.any(patch_mask[:, 0]))
    touches_right = bool(np.any(patch_mask[:, -1]))

    touched_edges: list[str] = []
    if touches_top:
        touched_edges.append("top")
    if touches_bottom:
        touched_edges.append("bottom")
    if touches_left:
        touched_edges.append("left")
    if touches_right:
        touched_edges.append("right")

    if touched_edges:
        warnings_list.append(
            f"Patch touches image boundary on: {', '.join(touched_edges)}."
        )

    is_valid = len(warnings_list) == 0
    return is_valid, warnings_list
