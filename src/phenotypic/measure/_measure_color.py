from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import pandas as pd
import logging

from scipy import ndimage

from phenotypic.abc_ import MeasureFeatures
from phenotypic.schema import OBJECT
from phenotypic.schema import ColorXYZ, Colorxy, ColorLab, ColorHSV
from phenotypic.util import (
    robust_color_center,
    medoid_ciede2000,
    delta_e2000_spread,
    hsv_to_cone,
    cone_to_hsv,
    lab_to_srgb_hex,
)

logger = logging.getLogger(__name__)


class MeasureColor(MeasureFeatures):
    """Measure robust colorimetric statistics for each colony.

    Default output (always on):

    - **CIE L*a*b*** -- ΔE76 geometric-median center, ΔE2000 medoid center,
      ΔE2000 within-colony consistency (median/mean/P95 from the medoid),
      ``LabTotalVariance``, and an sRGB hex swatch (plot-only).
    - **HSV** -- a cone-embedded robust center (circular-correct) and
      ``HSVConeVariance``.

    Opt-in, hidden from the reference doc:

    - **CIE XYZ** (``include_XYZ=True``) and **xy chromaticity**
      (``include_xy=True``) -- legacy per-channel min/Q1/mean/median/Q3/max/
      stddev/CoeffVar suites.

    Args:
        include_XYZ: Emit the legacy CIE XYZ per-channel suite. Default ``False``.
        include_xy: Emit the legacy xy chromaticity per-channel suite. Default
            ``False``.
        geomedian_max_iter: Weiszfeld iteration cap for the L*a*b* geometric
            median. Default ``50``.
        geomedian_tol: Weiszfeld convergence tolerance. Default ``1e-4``.
        medoid_max_pixels: Subsample cap for the O(N^2) ΔE2000 medoid selection;
            consistency scalars still use all pixels. Default ``1000``.
        random_seed: Seed for reproducible medoid subsampling. Default ``0``.

    Examples:
        Measure robust colorimetric statistics for a detected plate:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.measure import MeasureColor
        >>> from phenotypic.schema import ColorLab
        >>> image = OtsuDetector().apply(load_synth_yeast_plate())
        >>> df = MeasureColor().measure(image)
        >>> str(ColorLab.MEDOID_COLOR_HEX) in df.columns
        True
    """

    _measurement_infoclasses: ClassVar[list[type]] = [
        ColorXYZ, Colorxy, ColorLab, ColorHSV]

    include_XYZ: bool = False
    include_xy: bool = False
    geomedian_max_iter: int = 50
    geomedian_tol: float = 1e-4
    medoid_max_pixels: int = 1000
    random_seed: int = 0

    def _operate(self, image: Image) -> pd.DataFrame:
        objmap = image.objmap[:]
        data = {OBJECT.LABEL: image.objects.labels2series()}

        if self.include_XYZ:
            data.update(self._legacy_xyz_metrics(image, objmap))
        if self.include_xy:
            data.update(self._legacy_xy_metrics(image, objmap))

        data.update(self._robust_lab_hsv_metrics(image, objmap))
        return pd.DataFrame(data=data)

    # ------------------------------------------------------------------
    # Robust default block
    # ------------------------------------------------------------------
    def _robust_lab_hsv_metrics(self, image: Image, objmap: np.ndarray) -> dict:
        lab = image.color.Lab[:]
        hsv = image.color.hsv[:]
        labels = np.unique(objmap)
        labels = labels[labels != 0]

        # One linear pass yields each label's bounding-box slices, so per-object
        # work touches only that colony's window instead of rescanning the whole
        # image (O(N) total vs O(K*N) full-image masks). bbox_slices[L - 1] is the
        # slice tuple for label L (None for absent labels).
        bbox_slices = ndimage.find_objects(objmap)

        rows: list[dict] = []
        for label in labels:
            sl = bbox_slices[label - 1]
            submask = objmap[sl] == label  # exclude neighbours sharing the bbox
            rows.append({**self._robust_lab_row(lab[sl][submask]),
                         **self._robust_hsv_row(hsv[sl][submask])})

        # Assemble column-major dict; preserve header order.
        columns = ColorLab.robust_headers() + ColorHSV.robust_headers()
        return {col: [row[col] for row in rows] for col in columns}

    def _robust_lab_row(self, lab_px: np.ndarray) -> dict:
        gm = robust_color_center(
            lab_px, max_iter=self.geomedian_max_iter, tol=self.geomedian_tol
        )
        medoid, deltas = medoid_ciede2000(
            lab_px, max_pixels=self.medoid_max_pixels, seed=self.random_seed
        )
        de_median, de_mean, de_p95 = delta_e2000_spread(deltas)
        total_var = (
            float(lab_px.var(axis=0, ddof=0).sum()) if lab_px.shape[0] else float("nan")
        )
        return {
            str(ColorLab.L_STAR_GEOMEDIAN): float(gm[0]),
            str(ColorLab.A_STAR_GEOMEDIAN): float(gm[1]),
            str(ColorLab.B_STAR_GEOMEDIAN): float(gm[2]),
            str(ColorLab.L_STAR_MEDOID): float(medoid[0]),
            str(ColorLab.A_STAR_MEDOID): float(medoid[1]),
            str(ColorLab.B_STAR_MEDOID): float(medoid[2]),
            str(ColorLab.DELTA_E2000_MEDIAN): de_median,
            str(ColorLab.DELTA_E2000_MEAN): de_mean,
            str(ColorLab.DELTA_E2000_P95): de_p95,
            str(ColorLab.LAB_TOTAL_VARIANCE): total_var,
            str(ColorLab.MEDOID_COLOR_HEX): lab_to_srgb_hex(medoid),
        }

    def _robust_hsv_row(self, hsv_px: np.ndarray) -> dict:
        cone = hsv_to_cone(hsv_px)
        center_cone = robust_color_center(
            cone, max_iter=self.geomedian_max_iter, tol=self.geomedian_tol
        )
        center_hsv = cone_to_hsv(center_cone)
        cone_var = (
            float(cone.var(axis=0, ddof=0).sum()) if cone.shape[0] else float("nan")
        )
        return {
            str(ColorHSV.HUE_ROBUST_MEAN): float(center_hsv[0]),
            str(ColorHSV.SATURATION_ROBUST_MEAN): float(center_hsv[1]),
            str(ColorHSV.VALUE_ROBUST_MEAN): float(center_hsv[2]),
            str(ColorHSV.HSV_CONE_VARIANCE): cone_var,
        }

    # ------------------------------------------------------------------
    # Legacy opt-in blocks (8-stat suites)
    # ------------------------------------------------------------------
    def _legacy_xyz_metrics(self, image: Image, objmap: np.ndarray) -> dict:
        fg = image.color.XYZ.foreground()
        out = {}
        for ch, headers in (
            (0, ColorXYZ.cieX_headers()),
            (1, ColorXYZ.cieY_headers()),
            (2, ColorXYZ.cieZ_headers()),
        ):
            metrics = MeasureColor._compute_color_metrics(foreground=fg[..., ch], objmap=objmap)
            out.update({k: v for k, v in zip(headers, metrics)})
        return out

    def _legacy_xy_metrics(self, image: Image, objmap: np.ndarray) -> dict:
        fg = image.color.xy.foreground()
        out = {}
        for ch, headers in ((0, Colorxy.x_headers()), (1, Colorxy.y_headers())):
            metrics = MeasureColor._compute_color_metrics(foreground=fg[..., ch], objmap=objmap)
            out.update({k: v for k, v in zip(headers, metrics)})
        return out

    @staticmethod
    def _compute_color_metrics(foreground: np.ndarray, objmap: np.ndarray):
        """Per-object 8-stat suite for the legacy opt-in XYZ/xy paths."""
        return [
            MeasureFeatures._calculate_minimum(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_q1(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_mean(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_median(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_q3(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_maximum(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_stddev(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_coeff_variation(array=foreground, objmap=objmap),
        ]


# Reference-doc RST: only the default colorimetric spaces (Lab, HSV).
MeasureColor.__doc__ = ColorHSV.append_rst_to_doc(
    ColorLab.append_rst_to_doc(MeasureColor)
)
