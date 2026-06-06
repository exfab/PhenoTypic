from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from skimage.measure._regionprops import RegionProperties

    from phenotypic._core._image import Image

import logging

import numpy as np
from pydantic import field_validator
from scipy.ndimage import distance_transform_edt, label as ndi_label
from skimage.measure import euler_number, regionprops

from ..abc_ import ObjectRefiner
from ..measure._measure_symmetric_zones import MeasureSymmetricZones

_log = logging.getLogger(__name__)


class TrimAsymmetry(ObjectRefiner):
    """Trim asymmetric spurs and reticulated noise beyond each colony's symmetric radius.

    Computes a per-colony symmetric radius (``R_sym``) using equal-area radial
    annuli and angular-sector coverage profiling, then removes mask pixels
    beyond that radius that belong to topologically noisy or geometrically
    spurious connected components. When ``beehive_threshold`` is set, linearly
    branched structures (genuine hyphae, zero topological holes) are preserved
    while web-like reticulated components are discarded based on their
    holes-per-pixel density.

    For the underlying symmetric-radius algorithm, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Filamentous fungi detections where Dijkstra-reconnected bridges
          produce thin asymmetric spurs extending past the colony body.
        - Plates where reticulated or web-like mask artefacts surround
          otherwise symmetric colony cores.
        - Compact yeast or bacterial colonies that acquired noise appendages
          from high-contrast agar texture during detection.
        - Pipeline stages after :class:`FilamentousFungiDetector` where
          reconnection loops must be removed while linear hyphae are kept.

    Consider Also:
        - :class:`SmallObjectRemover` when spurious fragments are
          distinguishable by size alone rather than by radial position
          relative to the colony centre.
        - :class:`RemoveNonCircular` when the entire colony shape should be
          judged against a circularity threshold rather than a radial
          symmetric envelope.
        - :class:`MaskErosion` for a simpler boundary retraction when the
          asymmetric extent is shallow and uniform around the colony perimeter.
        - :class:`MergeFragmentChains` when asymmetric components are genuine
          but fragmented, and should be reconnected rather than removed.

    Args:
        symmetry_threshold: Minimum fraction of ``n_angular_bins`` sectors
            that must contain mask pixels for an annulus to count as symmetric.
            Lower values push ``R_sym`` outward (less trimming); higher values
            pull it inward (more trimming). Practical window: 0.33--0.83.
            Default: 0.5 (3/6), intentionally looser than
            :class:`MeasureSymmetricZones` (4/6) so ``R_sym`` stays near the
            true colony edge rather than the inoculum core. On small colonies
            (radius < 30 px) with a coarse 6-sector grid, consider reducing
            ``n_angular_bins`` to 4 rather than lowering this threshold.
        n_angular_bins: Number of equal-width angular sectors (360°/n each)
            for the coverage diagnostic. Fewer bins (default 6, 60° each) give
            stable coverage estimates on small colonies; more bins resolve
            subtle sector-level asymmetry on large colonies. Typical range:
            4--12. Default: 6.
        n_annuli: Target number of equal-area radial annuli for profiling.
            Auto-clamped to ``max(6, min(n_annuli, max_pixel_radius))``,
            so increasing beyond ``max_pixel_radius`` has no effect. More
            annuli give finer PELT changepoint and ``R_sym`` resolution at
            proportionally higher runtime. Typical range: 10--200. Default:
            100.
        pelt_penalty: L2 changepoint penalty for PELT inoculum-core detection.
            Lower values detect more changepoints and place the core boundary
            closer to the inoculum; higher values suppress changepoints (no
            detection sets ``core_radius=0``). The penalty is on the BIC scale,
            of order ``log(n_annuli)`` for the mean-change (``l2``) model [1];
            raise it on noisy or fragmented masks to suppress spurious cores
            and lower it when a diffuse inoculum-to-colony transition needs an
            earlier boundary. Typical range: 1.0--20.0. Default: 5.0.
        smoothing_window: Moving-average window (annuli) applied to the
            angular-coverage profile before the ``R_sym`` threshold test.
            Larger windows prevent ``R_sym`` from collapsing due to isolated
            empty annuli from thin or fragmented masks; 1 disables smoothing.
            Typical range: 1--10. Default: 3.
        method: Inoculum-centre estimator for the distance map. ``"distance"``
            uses the Euclidean distance-transform peak (most inscribed point),
            robust on any input including all-zero grayscale. ``"intensity"``
            uses the intensity-weighted centroid of the grayscale crop, falling
            back to the distance peak when grayscale is flat or zero. Accepted
            values: ``"distance"``, ``"intensity"``. Default: ``"distance"``.
        beehive_threshold: Minimum holes-per-pixel density
            ``max(0, 1 - euler_number) / area`` required to remove a
            candidate connected component. ``None`` removes every component
            past ``R_sym`` regardless of topology. Positive values preserve
            linear hyphae (zero topological holes) while removing reticulated
            web-like components whose hole density exceeds the threshold.
            Practical range: 0.0--0.05; a reasonable starting point is
            ``0.002``, which removes components with roughly one hole per 500
            pixels while protecting genuine hyphae. Default: ``None``.
        min_cc_area: Minimum area (pixels) of an asymmetric connected
            component for a topology-based removal decision; smaller components
            are kept unchanged because Euler-number statistics are unreliable
            on tiny regions. Scale to 100--200 px on high-resolution scans.
            Typical range: 1--500. Default: 50.
        min_object_area: Minimum colony area (pixels) before any trimming is
            attempted; smaller colonies are skipped entirely to avoid
            degenerate geometry in the radial pipeline. Typical range:
            10--10 000. Default: 100.

    Returns:
        Image: Input image with ``objmap`` updated to remove asymmetric mask
        pixels beyond ``R_sym``; ``objmask`` refreshes automatically via the
        accessor. All other image components are unchanged.

    Raises:
        ValueError: If ``symmetry_threshold`` is outside ``[0, 1]``.
        ValueError: If ``beehive_threshold`` is negative.

    References:
        [1] R. Killick, P. Fearnhead, and I. A. Eckley, "Optimal detection
        of changepoints with a linear computational cost," *J. Amer. Statist.
        Assoc.*, vol. 107, no. 500, pp. 1590--1598, Dec. 2012.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for a visual
        walkthrough of spur and reticulated noise removal on real plate images.
        :doc:`/explanation/refinement_strategies` for the symmetric-radius
        algorithm and per-colony topology classification.
    """

    symmetry_threshold: float = 3 / 6
    n_angular_bins: int = 6
    n_annuli: int = 100
    pelt_penalty: float = 5.0
    smoothing_window: int = 3
    method: Literal["distance", "intensity"] = "distance"
    beehive_threshold: float | None = None
    min_cc_area: int = 50
    min_object_area: int = 100

    @field_validator("symmetry_threshold")
    @classmethod
    def _validate_symmetry_threshold(cls, symmetry_threshold: float) -> float:
        """Reject a ``symmetry_threshold`` outside ``[0, 1]``.

        Reproduces the pre-migration ``__init__`` guard verbatim.
        """
        if not 0.0 <= symmetry_threshold <= 1.0:
            raise ValueError(
                    "symmetry_threshold must be in [0, 1]; "
                    f"got {symmetry_threshold}."
            )
        return symmetry_threshold

    @field_validator("beehive_threshold")
    @classmethod
    def _validate_beehive_threshold(
            cls, beehive_threshold: float | None
    ) -> float | None:
        """Reject a negative ``beehive_threshold``.

        Reproduces the pre-migration ``__init__`` guard verbatim.
        """
        if beehive_threshold is not None and beehive_threshold < 0.0:
            raise ValueError(
                    "beehive_threshold must be non-negative or None; "
                    f"got {beehive_threshold}."
            )
        return beehive_threshold

    def _operate(self, image: Image) -> Image:
        """Trim asymmetric spurs / beehive noise from each colony in the objmap.

        Args:
            image: Detected image with a populated ``objmap``.

        Returns:
            Image: Same image with ``objmap`` updated in place.
        """
        objmap = image.objmap[:]
        if objmap.max() == 0:
            return image

        # Work on a copy we can paint zeros into; write back once at the end.
        modified = objmap.copy()

        # Intensity-weighted centroid needs a gray source.
        gray_full = image.gray[:] if self.method == "intensity" else None

        for prop in regionprops(objmap):
            if prop.area < self.min_object_area:
                continue

            slc = prop.slice
            objmap_crop = modified[slc]
            obj_mask = objmap_crop == prop.label

            to_remove = self._trim_mask_for_object(
                    obj_mask=obj_mask,
                    prop=prop,
                    gray_full=gray_full,
            )
            if to_remove is None or not to_remove.any():
                continue

            objmap_crop[to_remove] = 0

        image.objmap[:] = modified
        return image

    def _trim_mask_for_object(
            self,
            obj_mask: np.ndarray,
            prop: "RegionProperties",
            gray_full: np.ndarray | None,
    ) -> np.ndarray | None:
        """Compute the boolean "pixels to remove" mask for one colony crop.

        Args:
            obj_mask: Boolean crop of the object in its bounding box.
            prop: ``skimage.measure._regionprops.RegionProperties`` instance
                from iterating ``regionprops(objmap)``; used for ``slice``,
                ``label``, and ``centroid_weighted``.
            gray_full: Full-image grayscale used for intensity-weighted
                centroid (``method="intensity"``). ``None`` in distance mode.

        Returns:
            A boolean array the same shape as ``obj_mask`` with ``True``
            at pixels to remove, or ``None`` when the colony is ineligible
            (no asymmetric region or degenerate geometry).
        """
        # 1. Centroid (local bbox coords). Distance-transform peak is a safe
        # fallback; intensity mode uses it when the intensity is all zero and
        # regionprops produces a NaN weighted centroid.
        dt = distance_transform_edt(obj_mask)
        peak_idx = np.unravel_index(int(np.argmax(dt)), obj_mask.shape)
        distance_centroid = (float(peak_idx[0]), float(peak_idx[1]))

        if self.method == "distance":
            centroid_rc = distance_centroid
        else:
            assert gray_full is not None
            slc = prop.slice
            gray_crop = gray_full[slc]
            intensity_crop = gray_crop
            if gray_crop.ndim == 3:
                intensity_crop = gray_crop[..., 0]
            local_props = regionprops(
                    obj_mask.astype(np.int32),
                    intensity_image=intensity_crop,
            )
            if not local_props:
                return None
            cw = local_props[0].centroid_weighted
            if np.isnan(cw[0]) or np.isnan(cw[1]):
                # Intensity is flat/zero on this crop; fall back to distance peak.
                centroid_rc = distance_centroid
            else:
                centroid_rc = (float(cw[0]), float(cw[1]))

        # 2. Distance map from centroid.
        dist_map = MeasureSymmetricZones._distance_from_point(
                obj_mask.shape, centroid_rc
        )

        mask_distances = dist_map[obj_mask]
        if mask_distances.size == 0:
            return None
        max_pixel_radius = int(mask_distances.max())
        if max_pixel_radius <= 0:
            return None

        effective_annuli = max(6, min(self.n_annuli, max_pixel_radius))

        # 3. R_sym pipeline.
        density_profile, annulus_radii = (
            MeasureSymmetricZones._compute_radial_density_profile(
                    obj_mask, dist_map, effective_annuli,
            )
        )
        if annulus_radii.size == 0:
            return None

        try:
            core_radius = MeasureSymmetricZones._find_core_radius(
                    density_profile, annulus_radii, self.pelt_penalty,
            )
        except Exception:  # pragma: no cover — ruptures failure is rare
            _log.debug(
                    "PELT changepoint detection failed for label %d; "
                    "treating core as zero.",
                    prop.label, exc_info=True,
            )
            core_radius = 0.0

        _, _, angular_coverage = (
            MeasureSymmetricZones._compute_sholl_angular_profile(
                    obj_mask, dist_map, centroid_rc, annulus_radii,
                    self.n_angular_bins,
            )
        )
        r_sym = MeasureSymmetricZones._find_symmetric_radius(
                annulus_radii,
                angular_coverage,
                core_radius,
                self.symmetry_threshold,
                self.smoothing_window,
        )

        # 4. Asymmetric region.
        asymmetric_region = obj_mask & (dist_map > r_sym)
        if not asymmetric_region.any():
            return None

        # 5. Per-CC decision.
        cc_map, n_cc = ndi_label(asymmetric_region)
        if n_cc == 0:
            return None

        to_remove = np.zeros_like(obj_mask, dtype=bool)
        for cc_id in range(1, n_cc + 1):
            cc_mask = cc_map == cc_id
            cc_area = int(cc_mask.sum())
            if cc_area < self.min_cc_area:
                continue

            if self.beehive_threshold is None:
                to_remove |= cc_mask
                continue

            holes = max(0, 1 - int(euler_number(cc_mask, connectivity=2)))
            density = holes / float(cc_area)
            if density >= self.beehive_threshold:
                to_remove |= cc_mask

        if not to_remove.any():
            return None

        return to_remove
