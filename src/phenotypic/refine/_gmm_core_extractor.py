from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
import cv2
from sklearn.mixture import GaussianMixture

from ..abc_ import ObjectRefiner


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def _build_ellipse_kernel(radius: int) -> np.ndarray | None:
    """Create an elliptical morphological structuring element.

    Args:
        radius: Radius of the kernel.  When *radius* <= 0 the function
            returns ``None`` (meaning "no morphological operation").

    Returns:
        np.ndarray | None: A ``(2*radius+1, 2*radius+1)`` uint8 kernel
            produced by ``cv2.getStructuringElement(MORPH_ELLIPSE, ...)``,
            or ``None`` when *radius* is non-positive.
    """
    if radius <= 0:
        return None
    k = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _normalized_separation(gmm: GaussianMixture) -> float:
    """Compute normalised mean separation of a fitted two-component GMM.

    Args:
        gmm: A fitted ``GaussianMixture`` (typically *n_components=2*).

    Returns:
        float: ``|mu_0 - mu_1| / (sigma_0 + sigma_1)``.  Returns 0.0 when
            the summed standard deviations are negligible (< 1e-10).

    Raises:
        ValueError: If *gmm* has fewer than 2 components.
    """
    mu = gmm.means_.ravel()
    cov = gmm.covariances_
    if cov.ndim == 3:
        var = cov[:, 0, 0]
    elif cov.ndim == 2:
        var = cov[:, 0]
    else:
        var = cov.ravel()
    sigma_sum = np.sqrt(var[0]) + np.sqrt(var[1])
    if sigma_sum < 1e-10:
        return 0.0
    return float(np.abs(mu[1] - mu[0]) / sigma_sum)


def _extract_single_core(
    intensity: np.ndarray,
    label_map: np.ndarray,
    label: int,
    n_components: int,
    separation_threshold: float,
    min_core_area: int,
    open_kernel: np.ndarray | None,
    close_kernel: np.ndarray | None,
) -> np.ndarray:
    """Run GMM core extraction on a single labelled region.

    Args:
        intensity: 2-D float intensity image.
        label_map: Integer label map (same shape as *intensity*).
        label: The label value to process.
        n_components: Number of Gaussian components.
        separation_threshold: Normalised separation below which the
            original mask is returned unchanged.
        min_core_area: Minimum acceptable core area in pixels.
        open_kernel: Structuring element for morphological opening
            (``None`` to skip).
        close_kernel: Structuring element for morphological closing
            (``None`` to skip).

    Returns:
        np.ndarray: Boolean mask (same shape as *intensity*) for the
            refined region.
    """
    mask = label_map == label
    area = int(mask.sum())

    # Too small — keep as-is
    if area < min_core_area:
        return mask

    pixels = intensity[mask].reshape(-1, 1).astype(np.float64)

    # Uniform region — keep as-is
    if pixels.std() < 1e-6:
        return mask

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=3,
        random_state=42,
    )
    gmm.fit(pixels)
    sep = _normalized_separation(gmm)

    if sep < separation_threshold:
        return mask

    # Determine bounding box of the region
    rows, cols = np.where(mask)
    r_min, r_max = rows.min(), rows.max() + 1
    c_min, c_max = cols.min(), cols.max() + 1

    roi = intensity[r_min:r_max, c_min:c_max]
    mask_roi = mask[r_min:r_max, c_min:c_max]

    bright_comp = int(np.argmax(gmm.means_.ravel()))
    labels_flat = gmm.predict(roi.reshape(-1, 1).astype(np.float64))
    core_roi = (labels_flat.reshape(roi.shape) == bright_comp) & mask_roi

    core_u8 = core_roi.astype(np.uint8) * 255
    if open_kernel is not None:
        core_u8 = cv2.morphologyEx(core_u8, cv2.MORPH_OPEN, open_kernel)
    if close_kernel is not None:
        core_u8 = cv2.morphologyEx(core_u8, cv2.MORPH_CLOSE, close_kernel)

    n_labels, cc_map, stats, _ = cv2.connectedComponentsWithStats(
        core_u8, connectivity=8
    )

    best_cc = -1
    best_area = 0
    for lbl in range(1, n_labels):
        a = stats[lbl, cv2.CC_STAT_AREA]
        if a >= min_core_area and a > best_area:
            best_cc = lbl
            best_area = a

    if best_cc < 0:
        # No valid connected component — keep original mask
        return mask

    H, W = intensity.shape
    core_mask = np.zeros((H, W), dtype=bool)
    core_mask[r_min:r_max, c_min:c_max] = cc_map == best_cc
    return core_mask


# ---------------------------------------------------------------------------
# Public module-level function
# ---------------------------------------------------------------------------


def extract_gmm_cores(
    intensity_array: np.ndarray,
    label_map: np.ndarray,
    n_components: int = 2,
    separation_threshold: float = 0.8,
    min_core_area: int = 30,
    morph_open_radius: int = 1,
    morph_close_radius: int = 2,
) -> np.ndarray:
    """Extract compact bright cores from labelled regions using a GMM.

    Args:
        intensity_array: 2-D float grayscale intensity image.
        label_map: Integer-labelled object map (0 = background, >0 = object).
        n_components: Number of Gaussian components to fit per region
            (K=2 separates core from surround).
        separation_threshold: Normalised mean separation below which a
            region is left unchanged (i.e. the two components are not
            well separated).
        min_core_area: Minimum core area in pixels.  Regions smaller than
            this are kept as-is; connected components smaller than this
            are discarded during cleanup.
        morph_open_radius: Radius of the elliptical kernel used for
            morphological opening (0 to disable).
        morph_close_radius: Radius of the elliptical kernel used for
            morphological closing (0 to disable).

    Returns:
        np.ndarray: Refined integer label map with the same dtype and
            shape as *label_map*.

    Examples:
        >>> import numpy as np
        >>> from phenotypic.refine._gmm_core_extractor import extract_gmm_cores
        >>> intensity = np.zeros((60, 60), dtype=np.float64)
        >>> label_map = np.zeros((60, 60), dtype=np.int32)
        >>> label_map[10:50, 10:50] = 1
        >>> intensity[10:50, 10:50] = 0.3
        >>> intensity[20:40, 20:40] = 0.9
        >>> result = extract_gmm_cores(intensity, label_map)
        >>> result.shape == label_map.shape
        True
    """
    open_kernel = _build_ellipse_kernel(morph_open_radius)
    close_kernel = _build_ellipse_kernel(morph_close_radius)

    labels = np.unique(label_map)
    labels = labels[labels != 0]

    output = np.zeros_like(label_map)

    for label in labels:
        core_mask = _extract_single_core(
            intensity_array,
            label_map,
            label,
            n_components,
            separation_threshold,
            min_core_area,
            open_kernel,
            close_kernel,
        )
        output[core_mask] = label

    return output


# ---------------------------------------------------------------------------
# ObjectRefiner wrapper class
# ---------------------------------------------------------------------------


class GMMCoreExtractor(ObjectRefiner):
    """Extract compact bright cores from labelled colonies using a GMM.

    Args:
        n_components: Number of Gaussian mixture components per region.
        separation_threshold: Normalised separation below which the
            region is left unchanged.
        min_core_area: Minimum core area (pixels).
        morph_open_radius: Morphological opening radius (0 to disable).
        morph_close_radius: Morphological closing radius (0 to disable).

    Returns:
        Image: Image with ``objmap`` refined to bright-core masks.

    Raises:
        RuntimeError: If the underlying GMM fitting fails for all regions.

    Intuition:
        Colonies on agar plates often consist of a compact bright
        inoculum core surrounded by a dimmer halo of diffuse growth.
        Fitting a two-component Gaussian mixture to each labelled
        region's intensity histogram allows the bright core to be
        separated from the surround, yielding tighter masks that better
        represent the actively growing centre of each colony.

    Use cases (agar plates):
        - Tighten colony masks after initial thresholding that includes
          halo or edge diffusion.
        - Extract inoculum spots in pinned-array experiments where a
          dense bright centre is surrounded by thin outgrowth.
        - Pre-measurement cleanup so that intensity and shape features
          reflect the colony core rather than the full footprint.

    Tuning and effects:
        - **n_components**: Keep at 2 for the canonical core-vs-surround
          split.  Higher values may capture additional structure but
          increase fitting cost and risk over-segmentation.
        - **separation_threshold**: Higher values require stronger
          intensity contrast before extraction is attempted.  Lowering
          the threshold extracts cores even when the contrast is subtle,
          at the risk of false positives.
        - **min_core_area**: Raise to avoid extracting tiny noise
          fragments; lower to retain very small inoculum spots.
        - **morph_open_radius / morph_close_radius**: Control
          post-extraction morphological cleanup.  Opening removes thin
          protrusions; closing fills small gaps.  Set to 0 to disable.

    Caveats:
        - Regions with nearly uniform intensity (std < 1e-6) are left
          unchanged because the GMM cannot separate them.
        - Regions smaller than *min_core_area* are kept as-is.
        - The GMM is fitted independently per region, so runtime scales
          linearly with the number of labelled objects.

    Attributes:
        (No public attributes)

    Examples:
        .. dropdown:: Extract bright colony cores after detection

            >>> from phenotypic.refine._gmm_core_extractor import GMMCoreExtractor
            >>> op = GMMCoreExtractor(separation_threshold=0.6)
            >>> image = op.apply(image, inplace=True)  # doctest: +SKIP
    """

    def __init__(
        self,
        n_components: int = 2,
        separation_threshold: float = 0.8,
        min_core_area: int = 30,
        morph_open_radius: int = 1,
        morph_close_radius: int = 2,
    ):
        """Initialise the GMM core extractor.

        Args:
            n_components (int): Number of Gaussian components to fit per
                labelled region (default 2 — core vs. surround).
            separation_threshold (float): Normalised mean separation
                below which the region is left unchanged (0.0–1.0+).
            min_core_area (int): Minimum core area in pixels.  Regions
                or connected components below this size are kept as-is
                or discarded.
            morph_open_radius (int): Radius for morphological opening
                (0 disables).
            morph_close_radius (int): Radius for morphological closing
                (0 disables).
        """
        self.__n_components = n_components
        self.__separation_threshold = separation_threshold
        self.__min_core_area = min_core_area
        self.__morph_open_radius = morph_open_radius
        self.__morph_close_radius = morph_close_radius

    def _operate(self, image: Image) -> Image:
        intensity = image.detect_mat[:].astype(np.float64)
        label_map = image.objmap[:]

        refined = extract_gmm_cores(
            intensity_array=intensity,
            label_map=label_map,
            n_components=self.__n_components,
            separation_threshold=self.__separation_threshold,
            min_core_area=self.__min_core_area,
            morph_open_radius=self.__morph_open_radius,
            morph_close_radius=self.__morph_close_radius,
        )

        image.objmap[:] = refined
        return image
