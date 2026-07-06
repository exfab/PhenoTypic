"""MeasureOrientationZones: per-zone hyphal orientation concentration/turning."""
from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import PrivateAttr, field_validator

# Control/FigureProvider/figure are re-exported from phenotypic.abc_ (this is
# exactly what _measure_symmetric_zones.py imports).
from phenotypic.abc_ import FigureProvider, MeasureFeatures
from phenotypic.schema import OBJECT, ORIENTATION_ZONES
from phenotypic.util._orientation_field import orientation_field
from phenotypic.measure._zone_segmentation import (
    ZoneSegmentation,
    ZoneSegmentationParams,
    compute_zone_segmentation,
    distance_from_point,
    expand_slice_around_center,
)

_VARIANTS = ("Radial", "Mask")
_ZONES = ("Overall", "Dense", "Sparse")
_METRICS = ("Concentration", "Turning", "Coherence")
_EPS = 1e-9


def zone_selector(dist_map, r_lo, r_hi, obj_mask, variant):
    """Boolean selector for a radial zone on a tile; ``Mask`` also ∩ obj_mask.

    Args:
        dist_map: Per-pixel distance-from-centre map (tile shape).
        r_lo: Inner radius (inclusive) of the zone in pixels.
        r_hi: Outer radius (exclusive) of the zone in pixels.
        obj_mask: Boolean object mask (tile shape) used by the ``Mask`` variant.
        variant: ``"Radial"`` (all tile pixels in the ring) or ``"Mask"``
            (the ring intersected with ``obj_mask``).

    Returns:
        Boolean array (tile shape). All-False when the radius range is invalid
        (non-finite or ``r_hi <= r_lo``).
    """
    if not np.isfinite(r_lo) or not np.isfinite(r_hi) or r_hi <= r_lo:
        return np.zeros(dist_map.shape, dtype=bool)
    radial = (dist_map >= r_lo) & (dist_map < r_hi)
    if variant == "Mask":
        return radial & obj_mask
    return radial


def aggregate_orientation(phi, coherence, grad_phi, selector, eps=_EPS):
    """Coherence-weighted (R, turning, mean-coherence) over a selector.

    Args:
        phi: Orientation field in radians (tile shape).
        coherence: Structure-tensor coherence in [0, 1] (tile shape).
        grad_phi: Orientation-gradient magnitude in rad/px (tile shape).
        selector: Boolean pixel selector (tile shape).
        eps: Numerical floor for the summed-coherence denominator.

    Returns:
        ``(R, turning, mean_coherence)`` scalars. Returns ``(nan, nan, nan)``
        when the selector is empty or ``sum(coherence) ~ 0``.
    """
    if not selector.any():
        return (np.nan, np.nan, np.nan)
    C = coherence[selector]
    sumC = float(C.sum())
    if sumC < eps:
        return (np.nan, np.nan, np.nan)
    c2 = np.cos(2.0 * phi[selector])
    s2 = np.sin(2.0 * phi[selector])
    Rx = float((C * c2).sum()) / sumC
    Ry = float((C * s2).sum()) / sumC
    R = float(np.hypot(Rx, Ry))
    turning = float((C * grad_phi[selector]).sum()) / sumC
    return (R, turning, float(C.mean()))


def _downsample_quiver(phi, coherence, block):
    """Block-mean the doubled-angle field → (rows, cols, phi_block, coh_block).

    Circular-averages cos2φ/sin2φ (coherence-weighted) and means coherence over
    block×block cells. Returns block-centre coords in the TILE frame plus per-block
    orientation and coherence — a few KB, the only array kept in the lean cache.

    Args:
        phi: Orientation field in radians (tile shape).
        coherence: Structure-tensor coherence in [0, 1] (tile shape).
        block: Block edge length in pixels.

    Returns:
        Tuple ``(rows, cols, phi_block, coh_block)`` of ``(nr, nc)`` arrays:
        block-centre row/col in tile coordinates, per-block orientation (NaN
        where the block coherence is ~0), and per-block mean coherence.
    """
    h, w = phi.shape
    block = max(1, int(block))
    nr, nc = max(h // block, 1), max(w // block, 1)
    rows = np.empty((nr, nc))
    cols = np.empty((nr, nc))
    pb = np.empty((nr, nc))
    cb = np.empty((nr, nc))
    c2, s2 = np.cos(2.0 * phi), np.sin(2.0 * phi)
    for i in range(nr):
        for j in range(nc):
            rsl, csl = slice(i * block, (i + 1) * block), slice(j * block, (j + 1) * block)
            cc = coherence[rsl, csl]
            rows[i, j], cols[i, j] = i * block + block / 2, j * block + block / 2
            cb[i, j] = float(cc.mean())
            wsum = float(cc.sum())
            pb[i, j] = (0.5 * np.arctan2((cc * s2[rsl, csl]).sum(), (cc * c2[rsl, csl]).sum())
                        if wsum > 1e-12 else np.nan)
    return rows, cols, pb, cb


def _resultant_direction(phi, coherence, selector):
    """Coherence-weighted mean orientation over a selector (for the inspect glyph).

    Args:
        phi: Orientation field in radians (tile shape).
        coherence: Structure-tensor coherence in [0, 1] (tile shape).
        selector: Boolean pixel selector (tile shape).

    Returns:
        Mean orientation in radians, or NaN when the selector is empty or the
        summed coherence is ~0.
    """
    if not selector.any():
        return np.nan
    C = coherence[selector]
    if float(C.sum()) < _EPS:
        return np.nan
    return 0.5 * np.arctan2(float((C * np.sin(2.0 * phi[selector])).sum()),
                            float((C * np.cos(2.0 * phi[selector])).sum()))


class MeasureOrientationZones(MeasureFeatures, FigureProvider):
    """Measure per-zone hyphal orientation concentration, turning, and coherence.

    Computes the structure-tensor orientation field over a mask-free tile (grid
    section when the image is a GridImage, else an expanded crop) and aggregates
    coherence-weighted metrics over radially-defined zones bounded by the
    symmetric radius, in both a ``Radial`` and a raw ``Mask`` variant. Emits the
    :class:`~phenotypic.schema.ORIENTATION_ZONES` columns.

    Args:
        intensity_source: Image array for the structure tensor and zone
            segmentation (``"detect_mat"`` default, ``"gray"`` alternative).
        sigma_d: Gaussian-derivative (gradient) scale in pixels, ~ hypha width.
        sigma_i: Structure-tensor integration scale in pixels.
        quiver_block: inspect() quiver downsample block size in pixels.
        n_annuli: Number of equal-area annuli in the shared zone segmentation.
        pelt_penalty: PELT penalty controlling core-changepoint sensitivity.
        symmetry_threshold: Minimum angular coverage for symmetric growth.
        n_angular_bins: Number of angular bins for the coverage diagnostic.
        smoothing_window: Moving-average window (annuli) for the coverage test.
        method: Inoculum-centre estimator (``"distance"`` or ``"intensity"``).
        extent_margin: Fractional expansion of the analysis tile past the mask.
        min_samples_per_ring: Minimum pixel count per ring before interpolation.
        tau_core: Colony-ness threshold for the core/dense boundary.
        tau_dense: Colony-ness threshold for the dense/sparse boundary.
        tau_sparse: Colony-ness threshold for the sparse/outside boundary.

    Examples:
        >>> from phenotypic.data import load_synth_filamentous_plate
        >>> from phenotypic.measure import MeasureOrientationZones
        >>> image = load_synth_filamentous_plate()
        >>> df = MeasureOrientationZones().measure(image)
        >>> 'OrientZones_Concentration-Radial-Overall' in df.columns
        True
    """

    _measurement_infoclass: ClassVar[type] = ORIENTATION_ZONES

    intensity_source: Literal["gray", "detect_mat"] = "detect_mat"
    sigma_d: float = 1.5
    sigma_i: float = 4.0
    quiver_block: int = 12
    # --- zone passthrough (defaults identical to MeasureSymmetricZones) ---
    n_annuli: int = 100
    pelt_penalty: float = 5.0
    symmetry_threshold: float = 4 / 6
    n_angular_bins: int = 6
    smoothing_window: int = 3
    method: Literal["distance", "intensity"] = "distance"
    extent_margin: float = 0.05
    min_samples_per_ring: int = 5
    tau_core: float = 0.9
    tau_dense: float = 0.5
    tau_sparse: float = 0.1
    # Per-object figure intermediates, populated by _operate. PrivateAttr keeps
    # it out of model_dump()/JSON (mirrors MeasureSymmetricZones' cache pattern).
    _cache: dict = PrivateAttr(default_factory=dict)
    _cache_image: "object | None" = PrivateAttr(default=None)

    @field_validator("sigma_d", "sigma_i")
    @classmethod
    def _positive_sigma(cls, v):
        if v <= 0:
            raise ValueError("sigma_d and sigma_i must be > 0")
        return v

    def _zone_params(self) -> ZoneSegmentationParams:
        return ZoneSegmentationParams(
            n_annuli=self.n_annuli, pelt_penalty=self.pelt_penalty,
            symmetry_threshold=self.symmetry_threshold, n_angular_bins=self.n_angular_bins,
            smoothing_window=self.smoothing_window, method=self.method,
            extent_margin=self.extent_margin, min_samples_per_ring=self.min_samples_per_ring,
            tau_core=self.tau_core, tau_dense=self.tau_dense, tau_sparse=self.tau_sparse,
            intensity_source=self.intensity_source,
        )

    def _resolve_tile(self, image, seg: ZoneSegmentation, prop, label2section):
        """Return (tile_intensity, obj_mask_tile, centre_rc) for one object.

        Preferred: the object's **grid section** via ``image.grid[idx]`` — an
        object-aware cropped Image (only this object's label survives; the crop
        preserves the complete object, so it is a superset of the object's
        pixels). Verified API: ``image.grid[section_idx]`` returns a cropped
        ``Image``; the crop origin is recovered by the public exact identity
        ``origin = prop.centroid(full) - regionprops(section)[label].centroid``.
        Falls back to the mask-free expanded crop when the image is not a
        GridImage, the section lookup fails, or the section does not cover the
        r_max disk around the centre (crowded/overgrown plate).
        """
        from skimage.measure import regionprops
        r_max = max(seg.sparse_end_radius, seg.symmetric_radius) * (1 + self.extent_margin)
        if hasattr(image, "grid") and seg.label in label2section:
            try:
                section = image.grid[label2section[seg.label]]
                sec_props = {p.label: p for p in regionprops(section.objmap[:])}
                sp = sec_props.get(seg.label)
                if sp is not None:
                    origin = (prop.centroid[0] - sp.centroid[0],
                              prop.centroid[1] - sp.centroid[1])
                    centre = (seg.centroid_global[0] - origin[0],
                              seg.centroid_global[1] - origin[1])
                    H, W = section.objmap[:].shape[:2]
                    if (centre[0] - r_max >= 0 and centre[0] + r_max <= H
                            and centre[1] - r_max >= 0 and centre[1] + r_max <= W):
                        tile = np.asarray(getattr(section, self.intensity_source)[:], dtype=np.float64)
                        return tile, (section.objmap[:] == seg.label), centre
            except (KeyError, IndexError, ValueError, AttributeError):
                pass
        # Fallback: expanded crop on the full plate (non-grid / clipped section).
        hw = image.gray[:].shape[:2]            # 2-tuple; image.shape is (H,W,3) for RGB
        sl = expand_slice_around_center(seg.centroid_global, r_max, hw)
        tile = np.asarray(getattr(image, self.intensity_source)[sl], dtype=np.float64)
        obj_mask = (image.objmap[:][sl] == seg.label)
        centre = (seg.centroid_global[0] - sl[0].start, seg.centroid_global[1] - sl[1].start)
        return tile, obj_mask, centre

    def _zone_bounds(self, seg: ZoneSegmentation):
        return {
            "Overall": (0.0, seg.symmetric_radius),
            "Dense": (seg.core_end_radius, seg.dense_end_radius),
            "Sparse": (seg.dense_end_radius, seg.sparse_end_radius),
        }

    def _prep(self, image):
        """Regionprops + label→grid-section map, computed ONCE per image.

        grid.info() is slow on filamentous plates, so never call it per object.
        intensity_image is required so compute_zone_segmentation can read
        prop.centroid_weighted when method="intensity" (else AttributeError).
        """
        from skimage.measure import regionprops
        from phenotypic.schema import GRID
        props = regionprops(image.objmap[:],
                            intensity_image=image.gray[:].astype(np.float64, copy=False))
        label2section = {}
        if hasattr(image, "grid"):
            info = image.grid.info()
            lab, rmi = str(OBJECT.LABEL), str(GRID.ROW_MAJOR_IDX)
            label2section = dict(zip(info[lab].astype(int), info[rmi].astype(int)))
        return props, label2section

    def _iter_object_fields(self, image, props, label2section):
        """Yield (prop, seg, obj_mask, phi, coh, grad, dist_map, centre) per object.

        SINGLE source of truth for the heavy orientation compute — reused by
        _operate() (which keeps only compact summaries) and by dashboard()'s
        coherence panel (which recomputes on demand). The full-resolution arrays
        yielded here are consumed and discarded by each caller; nothing full-res
        is retained on the instance. Tiny objects (area<10) are skipped.
        """
        for prop in props:
            if prop.area < 10:
                continue
            seg = compute_zone_segmentation(image, prop, params=self._zone_params())
            tile, obj_mask, centre = self._resolve_tile(image, seg, prop, label2section)
            phi, coh, grad = orientation_field(tile, self.sigma_d, self.sigma_i)
            dist_map = distance_from_point(tile.shape, centre)
            yield prop, seg, obj_mask, phi, coh, grad, dist_map, centre

    def _operate(self, image) -> pd.DataFrame:  # type: ignore[override]
        props, label2section = self._prep(image)
        headers = ORIENTATION_ZONES.get_headers()
        # pre-seed every object's row with NaN so skipped/failed objects still appear
        base: dict[int, dict] = {}
        for prop in props:
            r: dict = {OBJECT.LABEL: prop.label}
            r.update({h: np.nan for h in headers})
            base[prop.label] = r
        self._cache.clear()          # compact per-object figure records only
        self._cache_image = image    # single reference (not a copy) for no-arg figures
        for prop, seg, obj_mask, phi, coh, grad, dist_map, centre in \
                self._iter_object_fields(image, props, label2section):
            per_zone = self._fill_metrics(base[prop.label], seg, obj_mask, phi, coh, grad, dist_map)
            # LEAN CACHE: store compact summaries only — NO full-res tile/phi/coh/
            # grad/dist_map and NO seg dataclass. Bounds memory to O(objects*blocks).
            self._cache[prop.label] = {
                "centroid_global": tuple(seg.centroid_global),
                "centre": centre,
                "radii": {"core": seg.core_radius, "symmetric": seg.symmetric_radius,
                          "core_end": seg.core_end_radius, "dense_end": seg.dense_end_radius,
                          "sparse_end": seg.sparse_end_radius},
                "zones_computed": seg.zones_computed,
                "quiver": _downsample_quiver(phi, coh, self.quiver_block),  # block-res
                "per_zone": per_zone,
            }
        return pd.DataFrame([base[p.label] for p in props], columns=[OBJECT.LABEL, *headers])

    def _fill_metrics(self, row, seg, obj_mask, phi, coh, grad, dist_map):
        """Write the 18 columns for one object; return the compact per_zone dict."""
        per_zone = {}
        for zone, (r_lo, r_hi) in self._zone_bounds(seg).items():
            zone_ok = seg.zones_computed or zone == "Overall"
            for variant in _VARIANTS:
                if not zone_ok:
                    R = t = cm = direction = np.nan
                else:
                    sel = zone_selector(dist_map, r_lo, r_hi, obj_mask, variant)
                    R, t, cm = aggregate_orientation(phi, coh, grad, sel)
                    direction = _resultant_direction(phi, coh, sel)
                per_zone[(variant, zone)] = (R, t, cm, direction)   # scalars only
                row[f"OrientZones_Concentration-{variant}-{zone}"] = R
                row[f"OrientZones_Turning-{variant}-{zone}"] = t
                row[f"OrientZones_Coherence-{variant}-{zone}"] = cm
        return per_zone

    def _coherence_canvas(self, image, downsample: int = 4):
        """Recompute per-object coherence and composite onto a plate canvas.

        Used only by dashboard()'s heatmap. Full-res fields are recomputed via
        _iter_object_fields and discarded here — the heatmap costs compute, not
        persistent memory. Returned canvas is downsampled for a light figure.
        """
        props, label2section = self._prep(image)
        canvas = np.full(image.gray[:].shape[:2], np.nan)
        for _prop, seg, _mask, _phi, coh, _grad, _dist, centre in \
                self._iter_object_fields(image, props, label2section):
            r0 = int(round(seg.centroid_global[0] - centre[0]))
            c0 = int(round(seg.centroid_global[1] - centre[1]))
            h, w = coh.shape
            r1, c1 = min(r0 + h, canvas.shape[0]), min(c0 + w, canvas.shape[1])
            canvas[max(r0, 0):r1, max(c0, 0):c1] = coh[: r1 - max(r0, 0), : c1 - max(c0, 0)]
        return canvas[::downsample, ::downsample]
