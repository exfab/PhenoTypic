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
    """Trim spurs and web-like noise beyond each colony's symmetric envelope.

    Reuses the symmetric-radius machinery from :class:`MeasureSymmetricZones`
    to locate, per colony, the radius past which growth stops being angularly
    symmetric (``R_sym``). Every mask pixel beyond ``R_sym`` is a candidate for
    removal. Candidates are segmented into connected components and, when
    ``beehive_threshold`` is provided, classified by their topology: CCs with
    many enclosed holes per pixel (reticulated "beehive" noise) are removed,
    while nearly linear branches (holes per pixel ≈ 0) are preserved.

    Compared to the measurement-time version in :class:`MeasureSymmetricZones`,
    this refiner is deliberately less harsh:

    * The default ``symmetry_threshold`` is ``3/6`` rather than ``4/6`` so
      ``R_sym`` lands further from the inoculum core.
    * Per-CC segmentation localizes the decision — trimming one noisy spur
      never touches a legitimate branch on the other side of the colony.

    Args:
        symmetry_threshold: Minimum angular coverage (fraction of
            ``n_angular_bins`` populated) required for growth to count as
            symmetric. Lower values push ``R_sym`` outward, shrinking the
            candidate region. Defaults to ``3/6``.
        n_angular_bins: Number of uniform angular bins used for the coverage
            diagnostic that feeds ``R_sym``. Defaults to 6.
        n_annuli: Target number of equal-area annuli for the radial density
            profile. Auto-scaled down for small colonies. Defaults to 100.
        pelt_penalty: Penalty controlling PELT changepoint sensitivity for
            the inoculum core detection. Defaults to 5.0.
        smoothing_window: Moving-average window (in annuli) applied to the
            angular coverage profile before the ``R_sym`` threshold test.
            Defaults to 3.
        method: Inoculum-centre estimator. ``"distance"`` uses the peak of
            the Euclidean distance transform; ``"intensity"`` uses the
            intensity-weighted centroid. Defaults to ``"distance"``.
        beehive_threshold: Minimum holes-per-pixel density required to trim
            a candidate connected component. When ``None`` (default), every
            CC past ``R_sym`` is trimmed — the "pure R_sym" mode. When a
            float, CCs with ``(1 - euler_number) / area`` below the threshold
            are kept as legitimate linear branches.
        min_cc_area: Minimum candidate-CC area (pixels) required to make a
            topological decision. Smaller CCs are kept unchanged because
            Euler statistics are unreliable on tiny regions. Defaults to 50.
        min_object_area: Minimum colony area (pixels) below which the
            refiner skips the colony entirely. Defaults to 100.

    Returns:
        Image: Input image with ``objmap`` updated; ``objmask`` refreshes
        automatically via the accessor.

    Best For:
        * Filamentous-fungi detections where Dijkstra-reconnected bridges
          occasionally produce thin asymmetric spurs off a symmetric body.
        * Plates where noise manifests as beehive / reticulated web
          structures while legitimate hyphae remain linear.

    Consider Also:
        * :class:`SmallObjectRemover` for spurious noise distinguished by
          size alone rather than spatial relationship to the colony body.
        * :class:`RemoveLowCircularity` when the whole colony should be
          judged by shape, not just its outer extent.

    Examples:
        Pure ``R_sym`` trimming (default — every CC past the symmetric
        envelope is removed). On the synthetic yeast plate the colonies
        are already compact/symmetric so the op is a no-op; this example
        simply shows that wiring the refiner into a detection pipeline
        does not break it:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.refine import TrimAsymmetry
        >>> plate = load_synth_yeast_plate()
        >>> detected = OtsuDetector().apply(plate)
        >>> trimmer = TrimAsymmetry()
        >>> refined = trimmer.apply(detected)
        >>> bool(refined.objmap[:].max() >= 0)
        True

        Two-stage beehive-gated trim — legitimate linear branches past the
        envelope are preserved; only topologically web-like regions are
        removed:

        >>> trimmer = TrimAsymmetry(beehive_threshold=0.002)
        >>> refined = trimmer.apply(detected)
        >>> bool(refined.objmap[:].max() >= 0)
        True
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
