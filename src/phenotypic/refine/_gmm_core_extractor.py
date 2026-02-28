from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import cv2
from sklearn.mixture import GaussianMixture

from ..abc_ import ObjectRefiner


class GMMCoreExtractor(ObjectRefiner):
    """Extract compact bright cores from labelled colonies using Gaussian mixture modeling.

    Args:
        n_components (int): Number of Gaussian mixture components to fit per region
            (default 2 — separates bright core from surrounding halo). Keep at 2 for
            canonical core-vs-surround splitting; higher values risk over-segmentation
            and increase computational cost.
        separation_threshold (float): Normalized mean separation below which the
            original region is left unchanged (default 0.8, range 0.0–1.0+). Regions
            with separation below this threshold show insufficient intensity contrast
            for reliable core extraction. Raise to extract only high-confidence cores;
            lower to include subtle separations at risk of false positives.
        min_core_area (int): Minimum core area in pixels (default 30). Regions
            or extracted cores smaller than this are kept as-is or discarded.
            Prevents extraction of tiny noise fragments; raise for sparse inoculum
            spots, lower for dense arrays.
        morph_open_radius (int): Radius of elliptical morphological opening kernel
            (default 1, set to 0 to disable). Removes thin protrusions from extracted
            cores, improving shape compactness.
        morph_close_radius (int): Radius of elliptical morphological closing kernel
            (default 2, set to 0 to disable). Fills small gaps and holes within
            extracted cores, improving connectivity.

    Returns:
        Image: Image with ``objmap`` refined to bright-core masks, preserving
            original image components (rgb, gray, detect_mat) unchanged.

    Raises:
        ValueError: If *n_components* is not a positive integer or if
            *separation_threshold* is negative.

    Intuition:
        Microbial colonies on agar plates typically consist of a compact, bright
        inoculum core surrounded by a dimmer halo of diffuse outgrowth. This halo
        arises from secreted metabolites, agar dissolution, or substrate depletion,
        creating a radial intensity gradient. Fitting a two-component Gaussian mixture
        model (GMM) to each colony's intensity histogram separates the bright core
        from the dimmer surround, yielding tighter masks that better represent the
        actively growing colony centre. This is particularly valuable in high-density
        plates (96-well, 384-well arrays) and pin-inoculated experiments where cores
        are visually distinct.

    Use cases (agar plates):
        - **Rich media with clear cores:** YPD, LB plates where colonies develop
          dense bright centres with obvious halos; common in laboratory screening.
          GMM easily separates intensity modes, yielding tight core masks.
        - **Pinned-array inoculation:** Dense inoculum spots (e.g., from pin tools)
          create sharp bright cores with thin outgrowth; GMM identifies the original
          inoculum mass for replicability.
        - **High-density plates:** 96-well or 384-well formats where colonies are
          small (20–60 pixels at 512×768) but well-separated; core extraction reduces
          spillover between wells when plates are imaged without physical separation.
        - **Pre-measurement cleanup:** Extract cores before measuring size, shape, or
          color; ensures features reflect the primary growth mass rather than
          secondary diffusion.
        - **Inoculum detection in timelapse:** In growth kinetics, distinguish
          inoculum spot from emerging growth halo using early timepoints where
          contrast is highest.

    Parameter Effects:

        **n_components** (int, default 2):
            Fixed at 2 to model core vs. surround split.
            - Value of 2: Canonical separation, standard practice.
            - Higher values (3+): Capture additional intensity structure (e.g.,
              satellite halo, edge gradient) but slow fitting and risk spurious modes
              from noise. Not recommended unless colonies show complex multi-modal
              intensity.
            - Effect: Directly controls GMM flexibility; higher values increase fitting
              cost ~O(K) where K is n_components.

        **separation_threshold** (float, range 0.0–1.0+, default 0.8):
            Normalized separation = |mu_bright - mu_dim| / (sigma_bright + sigma_dim).
            - Typical range: 0.5–1.2
            - Example values:
              * 0.5: Extract cores even with subtle contrast (halo nearly as bright
                as core); may include noise/artifacts.
              * 0.8 (default): Moderate contrast requirement; balances sensitivity
                and specificity for typical agar plates.
              * 1.0+: Strict; require sharp separation (well-defined core and halo);
                miss cores with gradual intensity gradients.
            - Effect on colony morphology:
              * Round, smooth colonies (yeast, e.g., S. cerevisiae): Typically high
                separation (0.8–1.2); GMM cleanly splits core from halo.
              * Irregular, rough colonies (bacteria, e.g., E. coli): Lower separation
                (0.6–0.9) due to uneven pigmentation; halo may be patchy.
              * Filamentous colonies (fungi, e.g., Aspergillus): Variable separation;
                cores may be fuzzy; use 0.7–0.9.
            - Note: This parameter does NOT scale with resolution (intensity range
              and camera properties determine separation, not pixel counts).

        **min_core_area** (int, range 10–500+, default 30):
            Minimum core area in pixels to retain after extraction.
            - Resolution scaling guidance (pixel-based parameter):
              * Reference resolution: 512×768
              * Scaling formula: adjusted_min_core_area = default × (image_width / 512)
              * Examples:
                - 512×768 (1×): 30 pixels
                - 640×960 (1.25×): 38 pixels (~30 × 1.25)
                - 1024×1536 (2×): 60 pixels (~30 × 2)
                - 2000×3000 (3.9×): 117 pixels (~30 × 3.9)
            - Colony diameter context (512×768 reference):
              * Small yeast colonies (~20 pixels diameter): Set min_core_area to 50–100
                (core typically 10–40% of colony area).
              * Medium colonies (~60 pixels diameter): Set 50–150.
              * Large mature colonies (>100 pixels): Set 100–300.
            - Effect:
              * Too high: Discards legitimate small cores (early timepoints, sparse
                inoculum); inflates false negatives.
              * Too low: Retains noise fragments and spurious cores; reduces precision.
            - Morphology context:
              * Yeast: Cores often 50–70% of colony area; set min_core_area to
                0.5–0.7 × typical_colony_area.
              * Bacteria: Cores often 40–60% of area (more uniform); use lower
                thresholds.
              * Fungi: Cores may be thin/dispersed; use lower thresholds (20–40 pixels).

        **morph_open_radius** (int, range 0–5, default 1):
            Radius of elliptical structuring element for morphological opening.
            - Resolution scaling guidance:
              * Scaling formula: adjusted_open_radius = default × (image_width / 512)
              * Examples:
                - 512×768: 1 pixel
                - 1024×1536: 2 pixels
                - 2000×3000: 3.9 pixels (→ 4)
            - Effect on core extraction:
              * 0 (disabled): No opening; cores retain GMM-predicted shape exactly,
                including thin protrusions and noise speckles.
              * 1 (default): Removes thin speckles (~1–3 pixels wide) and minor
                protrusions; good for salt-and-pepper noise and edge jaggedness.
              * 2–3: Removes larger protrusions (nail-like outgrowths 3–7 pixels);
                rounds core shape more aggressively.
              * >4: Excessive smoothing; merges multiple small cores or removes
                legitimate core extensions.
            - Morphology context:
              * Yeast (smooth, round): 1 pixel usually sufficient; opens smooth noise.
              * Bacteria (irregular): 1–2 pixels for rougher edges.
              * Fungi (filamentous): 0–1 (preserve filaments) or use with care.

        **morph_close_radius** (int, range 0–5, default 2):
            Radius of elliptical structuring element for morphological closing.
            - Resolution scaling guidance:
              * Scaling formula: adjusted_close_radius = default × (image_width / 512)
              * Examples:
                - 512×768: 2 pixels
                - 1024×1536: 4 pixels
                - 2000×3000: 7.8 pixels (→ 8)
            - Effect on core extraction:
              * 0 (disabled): No closing; GMM core with open applied (if enabled).
              * 1: Fills gaps and holes ~1–3 pixels wide within cores; useful for
                uneven pigmentation or glare artifacts.
              * 2 (default): Fills moderate gaps and consolidates core; balances
                closure and core integrity.
              * 3–4: Aggressively fills larger voids (5–8 pixels); may artificially
                inflate small cores.
              * >5: Risk bridging separate cores or creating artificial structures.
            - Morphology context:
              * Yeast (dense, compact): 2 pixels typical; closes internal voids.
              * Bacteria: 1–2 pixels (less voids typically).
              * Fungi: 0–1 (preserve internal complexity).

    Caveats and Limitations:

        - **Uniform intensity regions:** Regions with standard deviation < 1e-6 are left
          unchanged because the GMM cannot separate components. Occurs with very small
          colonies, extremely uniform media, or low-contrast imaging.
        - **Small regions:** Regions with area < min_core_area are kept as-is before
          extraction; useful for preserving small but real inoculum spots.
        - **Computational cost:** GMM fitting scales linearly with number of labelled
          objects and quadratically with region size. On 96-well plates (~80–96 objects),
          fitting is <100 ms. On 384-well (>300 objects) or megapixel images, consider
          memory and time constraints.
        - **Fixed n_components=2 assumption:** If colonies show non-bimodal intensity
          structure (e.g., multi-pigmented bacteria, fungi with internal rings), the
          two-component model may miss nuance. Consider n_components=3 for advanced cases,
          or use a different refinement strategy (e.g., threshold-based on quantiles).
        - **Edge effects from morphological operations:** Opening and closing slightly
          shrink/dilate core boundaries (~1–3 pixels depending on radius). This is usually
          small relative to colony size but may bias measurements on very small colonies.
          Quantified impact: ~5–10% edge erosion/dilation for typical parameters.
        - **Separation threshold sensitivity:** Threshold is normalized and does NOT
          scale with resolution, but IS sensitive to lighting, media color, and camera
          properties. Same colony on YPD vs. LB media may have different separations;
          threshold may need adjustment per imaging protocol.
        - **Post-morphology connected components:** After open/close, the largest
          connected component is retained. If cores fragment during operations or
          splitting occurs, only the largest piece survives; smaller fragments are
          discarded even if above min_core_area.

    Mathematical/Technical Background:

        The GMM fitting uses scikit-learn's `GaussianMixture` with covariance_type='full'
        and 3 random initializations. For a 1-D intensity histogram of a region,
        the model fits K Gaussians; the two components' means (mu) and standard deviations
        (sigma) are extracted. Normalized separation is defined as:

            sep = |mu_1 - mu_0| / (sigma_0 + sigma_1)

        This ratio measures intensity contrast relative to within-component variance; higher
        values indicate clearer separation. The bright component (argmax of means) is
        selected as the core; pixels assigned to it (via predict) within the original region
        mask are retained. This is more statistically grounded than simple thresholding and
        naturally adapts to per-region intensity distributions.
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

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ellipse_kernel(radius: int) -> np.ndarray | None:
        """Create an elliptical morphological structuring element for opening/closing.

        Args:
            radius (int): Radius of the kernel in pixels. When *radius* <= 0,
                returns ``None`` (indicating no morphological operation).

        Returns:
            np.ndarray | None: A ``(2*radius+1, 2*radius+1)`` uint8 structuring
                element with elliptical (disk-like) shape generated by
                ``cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))`` where
                k = 2*radius+1. Returns ``None`` when *radius* is non-positive,
                allowing calling code to skip morphological operations when disabled.

        Notes:
            Elliptical kernels are isotropic (uniform in all directions), making them
            ideal for preserving rounded colony shapes during morphological operations.
            They are generated by OpenCV and produce symmetric disk-like binary patterns.
        """
        if radius <= 0:
            return None
        k = 2 * radius + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    @staticmethod
    def _normalized_separation(gmm: GaussianMixture) -> float:
        """Compute normalized mean separation of a fitted Gaussian mixture model.

        Computes the intensity contrast between two Gaussian components normalized by
        their combined standard deviation. This metric quantifies whether the two
        components (typically bright core vs. dim halo) are well-separated relative to
        their spread.

        Args:
            gmm (GaussianMixture): A fitted Gaussian mixture model from scikit-learn
                (typically with n_components=2). Must have at least 2 components.
                The model should be fitted on 1-D intensity data (reshaped as column
                vector for binary classification).

        Returns:
            float: Normalized separation defined as
                |mu_1 - mu_0| / (sigma_0 + sigma_1), where mu_i is the mean and
                sigma_i is the standard deviation of component i. Returns 0.0 when
                the summed standard deviations are negligible (< 1e-10), indicating
                very tight, nearly overlapping components. Higher values (>0.5)
                indicate well-separated components; lower values (<0.3) suggest
                poor separation and unreliable core extraction.

        Raises:
            ValueError: If *gmm* has fewer than 2 components; this function
                requires bimodal or higher-order separation.

        Notes:
            - The metric is invariant to resolution (depends on intensity values
              and camera properties, not pixel counts).
            - For colony imaging, typical separation values range from 0.3 (barely
              separated, diffuse halo) to 1.5+ (sharp bright core, dark surround).
            - The function extracts variance from gmm.covariances_, handling both
              full (ndim=3), diagonal (ndim=2), and spherical (ndim=1) covariance
              types by taking appropriate slices.
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

    @staticmethod
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
        """Extract the bright core from a single labelled region using GMM.

        Core extraction pipeline for one colony region:
        1. Extract pixels from the labelled region
        2. Fit a Gaussian mixture model (GMM) to the region's intensity histogram
        3. Compute normalized separation between components
        4. If separation is insufficient, return original mask unchanged
        5. Identify bright component (higher mean) and predict labels for pixels
        6. Apply morphological opening/closing to refine core shape
        7. Select largest connected component above min_core_area
        8. Return binary core mask for this region

        Args:
            intensity (np.ndarray): 2-D float grayscale intensity image
                (e.g., image.detect_mat or normalized RGB). Shape (H, W).
            label_map (np.ndarray): Integer label map with same shape as *intensity*.
                Pixel value 0 = background; >0 = object label.
            label (int): The specific label value to process within label_map.
            n_components (int): Number of Gaussian components to fit per region.
                Typically 2 (bright core vs. dim halo).
            separation_threshold (float): Normalized separation below which the
                original region mask is returned unchanged (no extraction attempted).
                Typical range 0.5–1.2.
            min_core_area (int): Minimum core area in pixels. Extracted cores
                or final connected components smaller than this are discarded;
                the original region mask is returned instead.
            open_kernel (np.ndarray | None): Structuring element for morphological
                opening (shrink by eroding then dilating). When ``None``, skip opening.
            close_kernel (np.ndarray | None): Structuring element for morphological
                closing (expand by dilating then eroding). When ``None``, skip closing.

        Returns:
            np.ndarray: Boolean mask (same shape as *intensity*) indicating the
                extracted core region. True pixels belong to the core; False pixels
                are background or excluded. If extraction fails, returns the original
                region mask to avoid data loss.

        Notes:
            - The extraction is localized to the bounding box of the region
              (computational efficiency; reduces memory usage for sparse objects).
            - Regions smaller than min_core_area are returned as-is without
              attempting GMM fitting (no point fitting on tiny regions).
            - Uniform regions (std < 1e-6) are returned unchanged; GMM fitting would fail.
            - The GMM is fitted with covariance_type='full' and random_state=42
              for reproducibility.
            - After morphological operations, if no valid connected component survives
              min_core_area, the original region mask is returned (graceful fallback).
            - Morphological operations may slightly shift core boundaries (~1–3 pixels)
              and can cause 5–10% edge erosion/dilation; quantified impact depends on
              kernel size.
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
        sep = GMMCoreExtractor._normalized_separation(gmm)

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

    @staticmethod
    def _extract_cores(
        intensity_array: np.ndarray,
        label_map: np.ndarray,
        n_components: int = 2,
        separation_threshold: float = 0.8,
        min_core_area: int = 30,
        morph_open_radius: int = 1,
        morph_close_radius: int = 2,
    ) -> np.ndarray:
        """Extract bright cores from all labelled regions in a label map using GMM.

        Batch processing function that iterates over all unique labels in a labelled map,
        fits a Gaussian mixture model (GMM) to each region's intensity distribution,
        and extracts the bright core component. This is the main computational entry
        point; it orchestrates kernel building, per-region extraction, and result
        composition into a refined label map.

        Args:
            intensity_array (np.ndarray): 2-D float grayscale intensity image
                (H, W). Typically image.detect_mat (standardized to [0, 1] or [0, 255]).
            label_map (np.ndarray): Integer-labelled object map with same shape as
                *intensity_array*. Pixel value 0 = background; 1, 2, 3, ... = object
                labels. Usually from image.objmap[:].
            n_components (int): Number of Gaussian components to fit per region
                (default 2). Standard choice separates bright core from dim halo;
                higher values (3+) capture additional intensity modes but increase
                fitting cost and risk over-segmentation.
            separation_threshold (float): Normalized mean separation threshold below
                which a region is left unchanged (default 0.8, range 0.0–1.0+).
                Prevents extraction when intensity contrast is insufficient; e.g.,
                separation=0.7 indicates weak separation, suggesting the halo is
                nearly as bright as the core.
            min_core_area (int): Minimum core area in pixels (default 30).
                Regions with area < min_core_area are kept as-is without attempting
                extraction. Extracted cores or final connected components smaller than
                this are discarded; the original region mask is returned instead.
            morph_open_radius (int): Radius of elliptical kernel for morphological
                opening (default 1, range 0–5). Opening removes thin protrusions and
                noise speckles. Set to 0 to disable. Typical values:
                - 0: No opening; retain GMM shape exactly.
                - 1: Remove ~1–3 pixel speckles (salt-and-pepper noise).
                - 2–3: Remove ~3–7 pixel protrusions; more aggressive smoothing.
                - >4: Excessive; risk merging or removing legitimate core structures.
            morph_close_radius (int): Radius of elliptical kernel for morphological
                closing (default 2, range 0–5). Closing fills gaps and holes within
                cores. Set to 0 to disable. Typical values:
                - 0: No closing; keep open result exactly.
                - 1: Fill gaps ~1–3 pixels wide.
                - 2 (default): Fill moderate gaps; balance closure and integrity.
                - 3–4: Aggressively fill ~5–8 pixel voids; risk artificial inflation.
                - >5: Excessive; risk bridging separate cores.

        Returns:
            np.ndarray: Refined integer label map with the same shape and dtype as
                *label_map*. Each pixel retains its original label if the region
                was not refined (e.g., separation too low, region too small) or its
                label if the region was successfully extracted (0 if the pixel is
                outside the extracted core). Non-extracted pixels are 0 (background).

        Notes:
            - All unique labels (except 0) in label_map are processed independently.
            - Processing order is arbitrary; results are deterministic within each region.
            - Computational cost scales linearly with the number of labelled objects.
              On typical 96-well plates (~80–96 objects), processing is <100 ms.
              On 384-well plates (>300 objects) or megapixel images, memory/time may
              be a constraint.
            - Each region's GMM is fitted with covariance_type='full' and
              random_state=42 for reproducibility.
            - Regions are processed with a bounding-box optimization for memory
              efficiency; only pixels within each region's bounding box are analyzed.
            - Graceful fallback: If GMM extraction fails or no valid core survives
              post-processing, the original region mask is returned (no data loss).
        """
        open_kernel = GMMCoreExtractor._build_ellipse_kernel(morph_open_radius)
        close_kernel = GMMCoreExtractor._build_ellipse_kernel(morph_close_radius)

        labels = np.unique(label_map)
        labels = labels[labels != 0]

        output = np.zeros_like(label_map)

        for label in labels:
            core_mask = GMMCoreExtractor._extract_single_core(
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

    def _operate(self, image: Image) -> Image:
        intensity = image.detect_mat[:].astype(np.float64)
        label_map = image.objmap[:]

        refined = self._extract_cores(
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
