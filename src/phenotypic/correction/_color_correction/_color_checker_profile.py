"""Color checker profile for root-polynomial color correction.

Encapsulates the full workflow of extracting patch colors from a color checker
card image, fitting a root-polynomial correction matrix (Finlayson 2015), and
providing serializable diagnostics for quality assessment.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import colour
import numpy as np

from ._helpers import (
    _normalize_to_unit_float,
    _rgb_to_lab,
    center_and_pad_checker,
    compute_core_mask,
    geometric_median,
    lab_checker_cluster_masks,
    median_filter_rgb,
    trim_background_edges,
    validate_patch_shape,
)

if TYPE_CHECKING:
    from phenotypic._core._image import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Illuminant constants
# ---------------------------------------------------------------------------

_ILLUMINANT_XY: dict[str, np.ndarray] = {
    "D50": colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"],
    "D65": colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"],
}


def _illuminant_xy(name: str) -> np.ndarray:
    """Return CIE xy chromaticity for a named illuminant.

    Args:
        name: Illuminant identifier (e.g. ``'D50'``, ``'D65'``).

    Returns:
        2-element array of CIE xy chromaticity coordinates.

    Raises:
        ValueError: If *name* is not a recognised illuminant.
    """
    if name in _ILLUMINANT_XY:
        return _ILLUMINANT_XY[name]
    try:
        return colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][name]
    except KeyError:
        raise ValueError(
            f"Unknown illuminant '{name}'. Expected one of: "
            f"{list(colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'].keys())}"
        ) from None


def _illuminant_XYZ(name: str) -> np.ndarray:
    """Return normalised XYZ (Y=1) for a named illuminant."""
    return colour.xy_to_XYZ(_illuminant_xy(name))


# ---------------------------------------------------------------------------
# Reference data loading
# ---------------------------------------------------------------------------


def _load_reference_data(
    checker_type: str,
    target_illuminant: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Load reference Lab values and linear RGB for a colour checker.

    The ``colour`` library stores checker data as xyY under the checker's
    native illuminant (typically D50).  This function converts xyY to XYZ,
    applies Bradford chromatic adaptation to *target_illuminant* when needed,
    then converts to both CIE Lab and linear sRGB.

    Args:
        checker_type: Key into ``colour.CCS_COLOURCHECKERS``.
        target_illuminant: Target illuminant name (e.g. ``'D65'``).

    Returns:
        Tuple of ``(ref_Lab, ref_linear_rgb, target_wp_xy)`` where
        *ref_Lab* maps patch names to ``(L, a, b)`` arrays, *ref_linear_rgb*
        maps patch names to ``(R, G, B)`` linear arrays, and *target_wp_xy*
        is the CIE xy whitepoint of the target illuminant.
    """
    cc = colour.CCS_COLOURCHECKERS[checker_type]
    checker_wp_xy = cc.illuminant
    target_wp_xy = _illuminant_xy(target_illuminant)

    # Determine whether chromatic adaptation is necessary.
    need_adapt = not np.allclose(checker_wp_xy, target_wp_xy, atol=1e-4)

    checker_wp_XYZ = colour.xy_to_XYZ(checker_wp_xy)
    target_wp_XYZ = colour.xy_to_XYZ(target_wp_xy)

    ref_Lab: dict[str, np.ndarray] = {}
    ref_linear: dict[str, np.ndarray] = {}

    cs = colour.RGB_COLOURSPACES["sRGB"]

    for name, xyY in cc.data.items():
        XYZ = colour.xyY_to_XYZ(xyY)

        if need_adapt:
            XYZ = colour.chromatic_adaptation(
                XYZ,
                XYZ_w=checker_wp_XYZ,
                XYZ_wr=target_wp_XYZ,
                transform="Bradford",
            )

        Lab = colour.XYZ_to_Lab(XYZ, illuminant=target_wp_xy)
        ref_Lab[name] = Lab

        # XYZ -> linear sRGB (under target illuminant / D65 for sRGB).
        linear_rgb = colour.XYZ_to_RGB(
            XYZ,
            colourspace=cs,
            illuminant=target_wp_xy,
            apply_cctf_encoding=False,
        )
        ref_linear[name] = np.clip(linear_rgb, 0.0, None)

    return ref_Lab, ref_linear, target_wp_xy


# ---------------------------------------------------------------------------
# ColorCheckerProfile
# ---------------------------------------------------------------------------


class ColorCheckerProfile:
    """Profile fitted from a colour checker card for root-polynomial correction.

    Measures patch colours from one or more ROIs in an image, matches them
    against published reference values for the chosen checker type, rejects
    outlier patches, and solves for a root-polynomial colour correction
    matrix (Finlayson 2015).  The resulting matrix can be applied to entire
    images via :class:`ColorCorrector`.

    Args:
        checker_type: Key in ``colour.CCS_COLOURCHECKERS``.  Defaults to the
            post-2014 X-Rite ColorChecker 24.
        degree: Root-polynomial degree (1--4).  ``2`` is recommended.
        target_illuminant: Target whitepoint for the correction (e.g.
            ``'D65'``).
        median_filter_size: Kernel size for per-ROI median filtering.
        stddev_mag_threshold: Column-stddev threshold for border detection
            during checker centering.
        border_distance_threshold: Lab Euclidean distance below which a
            pixel is classified as border rather than swatch.
        core_fraction: Fraction of centroid-to-boundary distance used to
            define the reliable core of each patch.
        ridge_lambda: Ridge-regression regularisation parameter for the
            root-polynomial fit.
        outlier_sigma: Patches whose Delta-E 2000 exceeds
            ``mean + outlier_sigma * stddev`` are rejected as outliers.

    Attributes:
        correction_matrix: Fitted correction matrix once :meth:`fit_from_rois`
            or :meth:`fit_from_patch_colors` has been called.
        diagnostics: Per-patch and aggregate quality metrics.
        is_fitted: ``True`` after a successful fit.
    """

    def __init__(
        self,
        checker_type: str = "ColorChecker24 - After November 2014",
        degree: int = 2,
        target_illuminant: str = "D65",
        median_filter_size: int = 10,
        stddev_mag_threshold: float = 15.0,
        border_distance_threshold: float = 12.0,
        core_fraction: float = 0.5,
        ridge_lambda: float = 1e-3,
        outlier_sigma: float = 2.0,
    ) -> None:
        if degree not in {1, 2, 3, 4}:
            raise ValueError(f"degree must be 1, 2, 3, or 4, got {degree}")
        if not (0.0 < core_fraction <= 1.0):
            raise ValueError(f"core_fraction must be in (0, 1], got {core_fraction}")
        if ridge_lambda < 0:
            raise ValueError(f"ridge_lambda must be >= 0, got {ridge_lambda}")
        if outlier_sigma <= 0:
            raise ValueError(f"outlier_sigma must be > 0, got {outlier_sigma}")

        self.checker_type = checker_type
        self.degree = degree
        self.target_illuminant = target_illuminant
        self.median_filter_size = median_filter_size
        self.stddev_mag_threshold = stddev_mag_threshold
        self.border_distance_threshold = border_distance_threshold
        self.core_fraction = core_fraction
        self.ridge_lambda = ridge_lambda
        self.outlier_sigma = outlier_sigma

        self._correction_matrix: np.ndarray | None = None
        self._diagnostics: dict[str, Any] = {}
        self._is_fitted: bool = False

    # -- properties ---------------------------------------------------------

    @property
    def correction_matrix(self) -> np.ndarray:
        """Root-polynomial correction matrix (shape ``(3, F)``)."""
        if self._correction_matrix is None:
            raise RuntimeError(
                "Profile is not fitted. Call fit_from_rois() or "
                "fit_from_patch_colors() first."
            )
        return self._correction_matrix

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Diagnostic metrics from the most recent fit."""
        return self._diagnostics

    @property
    def is_fitted(self) -> bool:
        """Whether a correction matrix has been fitted."""
        return self._is_fitted

    # -- high-level fitting -------------------------------------------------

    def fit_from_rois(
        self,
        image: Image,
        rois: list[tuple[slice, slice]],
    ) -> ColorCheckerProfile:
        """Fit the profile from one or more checker-card ROIs in an image.

        For each ROI the function extracts the sub-image, pre-processes it
        (background trimming, median filtering, centering/padding), segments
        patches via nearest-neighbour Lab assignment, measures the core color
        of each patch with geometric median, and pools the results.  Outlier
        patches are rejected before fitting the root-polynomial matrix.

        Args:
            image: Source image containing visible checker cards.
            rois: List of ``(row_slice, col_slice)`` tuples delimiting the
                checker card regions.

        Returns:
            ``self`` for method chaining.
        """
        ref_Lab, ref_linear, target_wp_xy = _load_reference_data(
            self.checker_type, self.target_illuminant
        )
        patch_names = list(ref_Lab.keys())
        n_expected = len(patch_names)

        # Accumulate per-patch measurements across all ROIs.
        measured_srgb: dict[str, list[tuple[np.ndarray, int, float, list[str]]]] = {
            name: [] for name in patch_names
        }
        per_card_summary: list[dict[str, Any]] = []

        for roi_idx, (row_sl, col_sl) in enumerate(rois):
            # 1. Extract sub-image and preprocess.
            sub_rgb = image.rgb[row_sl, col_sl].copy()
            sub_rgb = trim_background_edges(sub_rgb)
            sub_rgb = median_filter_rgb(sub_rgb, size=self.median_filter_size)

            # 2. Center and pad the checker.
            sub_rgb_padded = center_and_pad_checker(
                sub_rgb,
                filter_size=self.median_filter_size,
                stddev_mag_threshold=self.stddev_mag_threshold,
            )

            # Re-convert padded image to Lab.
            padded_float = _normalize_to_unit_float(sub_rgb_padded)
            lab_padded = _rgb_to_lab(padded_float, illuminant_xy=target_wp_xy)

            # 4. Cluster masks.
            ref_Lab_tuples = {
                name: tuple(ref_Lab[name].tolist()) for name in patch_names
            }
            cluster_result = lab_checker_cluster_masks(
                lab_padded,
                ref_Lab_tuples,
                border_distance_threshold=self.border_distance_threshold,
                include_labels=True,
            )
            masks, bboxes, labels = cluster_result  # type: ignore[misc]

            card_detected = 0

            for mask, bbox, label in zip(masks, bboxes, labels):
                if label == "border":
                    continue
                if not mask.any():
                    continue

                # 5. Compute core mask and validate.
                core = compute_core_mask(mask, core_fraction=self.core_fraction)
                is_valid, warnings = validate_patch_shape(core)
                core_pixels_fraction = float(core.sum()) / max(float(mask.sum()), 1.0)

                # 6. Measure color from core pixels in the padded sRGB image.
                core_pixels_srgb = padded_float[core]
                if core_pixels_srgb.size == 0:
                    continue

                # Geometric median in sRGB space.
                patch_srgb = geometric_median(core_pixels_srgb)
                measured_srgb[label].append(
                    (patch_srgb, roi_idx, core_pixels_fraction, warnings)
                )
                card_detected += 1

            per_card_summary.append(
                {
                    "roi_index": roi_idx,
                    "patches_detected": card_detected,
                    "patches_expected": n_expected,
                }
            )

        # 7. Pool measurements: pick the best observation per patch (highest
        #    core fraction) when multiple ROIs contribute.
        measured_rgb_final: dict[str, np.ndarray] = {}
        patch_meta: dict[str, dict[str, Any]] = {}

        for name in patch_names:
            observations = measured_srgb[name]
            if not observations:
                continue

            # Select the observation with the largest core fraction.
            best = max(observations, key=lambda o: o[2])
            measured_rgb_final[name] = best[0]
            patch_meta[name] = {
                "roi_index": best[1],
                "core_fraction_used": best[2],
                "validation_warnings": best[3],
            }

        self._fit_from_measured(
            measured_rgb_final,
            ref_Lab,
            ref_linear,
            target_wp_xy,
            patch_meta,
            per_card_summary,
        )
        return self

    def fit_from_patch_colors(
        self,
        measured_rgb: np.ndarray,
        patch_names: list[str] | None = None,
    ) -> ColorCheckerProfile:
        """Fit from pre-measured patch colors.

        Lower-level entry point for when patch RGB values have already been
        extracted externally (e.g. from an automatic checker detector).

        Args:
            measured_rgb: ``(N, 3)`` float array of measured sRGB values in
                ``[0, 1]``.
            patch_names: Optional list of patch identifiers matching the rows
                of *measured_rgb*.  If ``None``, names are taken from the
                reference checker in order.

        Returns:
            ``self`` for method chaining.
        """
        ref_Lab, ref_linear, target_wp_xy = _load_reference_data(
            self.checker_type, self.target_illuminant
        )
        all_names = list(ref_Lab.keys())

        if patch_names is None:
            if measured_rgb.shape[0] != len(all_names):
                raise ValueError(
                    f"measured_rgb has {measured_rgb.shape[0]} rows but the "
                    f"checker expects {len(all_names)} patches. Supply "
                    f"patch_names explicitly if providing a subset."
                )
            patch_names = all_names

        measured_dict: dict[str, np.ndarray] = {}
        for i, name in enumerate(patch_names):
            measured_dict[name] = measured_rgb[i]

        patch_meta: dict[str, dict[str, Any]] = {
            name: {
                "roi_index": 0,
                "core_fraction_used": 1.0,
                "validation_warnings": [],
            }
            for name in patch_names
        }

        self._fit_from_measured(
            measured_dict,
            ref_Lab,
            ref_linear,
            target_wp_xy,
            patch_meta,
            per_card_summary=[],
        )
        return self

    # -- internal fitting logic ---------------------------------------------

    def _fit_from_measured(
        self,
        measured_srgb: dict[str, np.ndarray],
        ref_Lab: dict[str, np.ndarray],
        ref_linear: dict[str, np.ndarray],
        target_wp_xy: np.ndarray,
        patch_meta: dict[str, dict[str, Any]],
        per_card_summary: list[dict[str, Any]],
    ) -> None:
        """Core fitting routine shared by both public entry points.

        Takes measured sRGB (normalised [0,1]), reference Lab and linear RGB
        dicts, computes Delta-E 2000 for outlier rejection, fits the
        root-polynomial matrix, and populates diagnostics.
        """
        cs = colour.RGB_COLOURSPACES["sRGB"]
        all_patch_names = list(ref_Lab.keys())
        warnings_list: list[str] = []

        if not measured_srgb:
            raise ValueError("No patches were successfully measured.")

        # --- Convert measured sRGB -> Lab for Delta-E computation ----------
        measured_Lab: dict[str, np.ndarray] = {}
        measured_linear: dict[str, np.ndarray] = {}

        for name, srgb in measured_srgb.items():
            srgb_clipped = np.clip(srgb, 0.0, 1.0)
            XYZ = colour.RGB_to_XYZ(
                srgb_clipped, colourspace=cs, apply_cctf_decoding=True
            )
            measured_Lab[name] = colour.XYZ_to_Lab(XYZ, illuminant=target_wp_xy)
            measured_linear[name] = colour.cctf_decoding(srgb_clipped, function="sRGB")

        # --- Compute before-correction Delta-E 2000 -----------------------
        deltaE_before: dict[str, float] = {}
        for name in measured_Lab:
            dE = float(
                colour.difference.delta_E_CIE2000(
                    measured_Lab[name], ref_Lab[name]
                )
            )
            deltaE_before[name] = dE

        # --- Outlier rejection --------------------------------------------
        dE_values = np.array(list(deltaE_before.values()))
        dE_mean = float(np.mean(dE_values))
        dE_std = float(np.std(dE_values))
        threshold = dE_mean + self.outlier_sigma * dE_std

        rejected: list[str] = []
        kept_names: list[str] = []
        for name in list(measured_Lab.keys()):
            if deltaE_before[name] > threshold:
                rejected.append(name)
                logger.info(
                    "Rejecting patch '%s': Delta-E 2000 = %.2f > threshold %.2f",
                    name,
                    deltaE_before[name],
                    threshold,
                )
            else:
                kept_names.append(name)

        if len(kept_names) < 4:
            warnings_list.append(
                f"Only {len(kept_names)} patches remaining after outlier "
                f"rejection (minimum recommended: 4)."
            )

        if not kept_names:
            raise ValueError(
                "All patches were rejected as outliers. Cannot fit correction matrix."
            )

        # --- Build arrays for fitting -------------------------------------
        n_kept = len(kept_names)
        measured_arr = np.zeros((n_kept, 3), dtype=np.float64)
        reference_arr = np.zeros((n_kept, 3), dtype=np.float64)

        for i, name in enumerate(kept_names):
            measured_arr[i] = measured_linear[name]
            reference_arr[i] = ref_linear[name]

        # --- Fit root-polynomial correction matrix -------------------------
        # colour.characterisation.matrix_colour_correction_Finlayson2015
        # M_T = measured (test), M_R = reference
        # Returns CCM of shape (3, F) where F = polynomial feature count
        try:
            CCM = colour.characterisation.matrix_colour_correction_Finlayson2015(
                measured_arr,
                reference_arr,
                degree=self.degree,  # type: ignore[arg-type]
                root_polynomial_expansion=True,
            )
        except Exception:
            # Fallback: manual ridge regression.
            CCM = self._fit_ridge(measured_arr, reference_arr)

        self._correction_matrix = CCM

        # --- Compute after-correction Delta-E 2000 -------------------------
        corrected_Lab: dict[str, np.ndarray] = {}
        deltaE_after: dict[str, float] = {}

        for name in measured_Lab:
            linear_in = measured_linear[name].reshape(1, 3)
            corrected_linear = (
                colour.characterisation.apply_matrix_colour_correction_Finlayson2015(
                    linear_in,
                    CCM,
                    degree=self.degree,  # type: ignore[arg-type]
                    root_polynomial_expansion=True,
                )
            )
            corrected_linear = np.clip(corrected_linear.ravel(), 0.0, None)
            corrected_srgb = colour.cctf_encoding(corrected_linear, function="sRGB")
            corrected_srgb = np.clip(corrected_srgb, 0.0, 1.0)  # type: ignore[assignment]

            XYZ_corr = colour.RGB_to_XYZ(
                corrected_srgb, colourspace=cs, apply_cctf_decoding=True
            )
            Lab_corr = colour.XYZ_to_Lab(XYZ_corr, illuminant=target_wp_xy)
            corrected_Lab[name] = Lab_corr

            dE = float(
                colour.difference.delta_E_CIE2000(Lab_corr, ref_Lab[name])
            )
            deltaE_after[name] = dE

        # --- Compute condition number --------------------------------------
        cond = float(np.linalg.cond(CCM))

        # --- Assemble diagnostics -----------------------------------------
        patches_diag: dict[str, dict[str, Any]] = {}
        for name in all_patch_names:
            entry: dict[str, Any] = {
                "reference_lab": ref_Lab[name].tolist(),
            }
            if name in measured_Lab:
                entry["measured_lab"] = measured_Lab[name].tolist()
                entry["corrected_lab"] = corrected_Lab[name].tolist()
                entry["deltaE00_before"] = deltaE_before[name]
                entry["deltaE00_after"] = deltaE_after[name]
            else:
                entry["measured_lab"] = None
                entry["corrected_lab"] = None
                entry["deltaE00_before"] = None
                entry["deltaE00_after"] = None

            meta = patch_meta.get(name, {})
            entry["core_fraction_used"] = meta.get("core_fraction_used", None)
            entry["roi_index"] = meta.get("roi_index", None)
            entry["validation_warnings"] = meta.get("validation_warnings", [])
            patches_diag[name] = entry

        after_values = [
            v for v in deltaE_after.values() if v is not None
        ]
        after_arr = np.array(after_values) if after_values else np.array([0.0])

        self._diagnostics = {
            "checker_type": self.checker_type,
            "degree": self.degree,
            "target_illuminant": self.target_illuminant,
            "n_patches_detected": len(measured_Lab),
            "n_patches_expected": len(all_patch_names),
            "n_patches_rejected": len(rejected),
            "rejected_patches": rejected,
            "per_card_summary": per_card_summary,
            "patches": patches_diag,
            "mean_deltaE00_before": float(np.mean(dE_values)),
            "mean_deltaE00_after": float(np.mean(after_arr)),
            "max_deltaE00_after": float(np.max(after_arr)),
            "median_deltaE00_after": float(np.median(after_arr)),
            "correction_matrix_condition_number": cond,
            "warnings": warnings_list,
        }

        self._is_fitted = True
        logger.info(
            "ColorCheckerProfile fitted: %d/%d patches, "
            "mean dE00 %.2f -> %.2f",
            len(measured_Lab),
            len(all_patch_names),
            self._diagnostics["mean_deltaE00_before"],
            self._diagnostics["mean_deltaE00_after"],
        )

    def _fit_ridge(
        self,
        measured_linear: np.ndarray,
        reference_linear: np.ndarray,
    ) -> np.ndarray:
        """Fallback ridge-regression fit when colour library call fails.

        Builds root-polynomial features manually and solves the regularised
        least-squares problem.

        Args:
            measured_linear: ``(N, 3)`` linear RGB of measured patches.
            reference_linear: ``(N, 3)`` linear RGB of reference patches.

        Returns:
            Correction matrix of shape ``(3, F)``.
        """
        Phi = colour.characterisation.polynomial_expansion_Finlayson2015(
            measured_linear,
            degree=self.degree,  # type: ignore[arg-type]
            root_polynomial_expansion=True,
        )
        # Solve: reference = Phi @ W.T  =>  W.T = (Phi^T Phi + lam I)^-1 Phi^T reference
        AtA = Phi.T @ Phi
        F = AtA.shape[0]
        W_T = np.linalg.solve(
            AtA + self.ridge_lambda * np.eye(F),
            Phi.T @ reference_linear,
        )
        # W_T has shape (F, 3); CCM convention in colour is (3, F)
        return W_T.T

    # -- reference data accessor -------------------------------------------

    def _load_refs(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
        """Load reference Lab and linear RGB for the configured checker.

        Returns:
            Tuple of ``(ref_Lab, ref_linear_rgb, target_wp_xy)``.
        """
        return _load_reference_data(self.checker_type, self.target_illuminant)

    # -- diagnostic --------------------------------------------------------

    def diagnostic(
        self,
        image: Image | None = None,
        rois: list[tuple[slice, slice]] | None = None,
    ) -> Any:
        """Return an interactive Panel dashboard for inspection.

        Args:
            image: Source image containing checker cards (optional).
            rois: ROI slices for pipeline step visualisation (optional).

        Returns:
            Interactive dashboard instance.  Call ``.panel()`` to display.

        Raises:
            RuntimeError: If the profile has not been fitted.
            ImportError: If Panel is not installed.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Cannot create diagnostic for an unfitted profile."
            )
        from ._diagnostic_dashboard import PANEL_AVAILABLE

        if not PANEL_AVAILABLE:
            raise ImportError(
                "Panel is required for interactive diagnostics. "
                "Install with: pip install 'phenotypic[gui]'"
            )
        from ._diagnostic_dashboard import ColorCorrectionDashboard

        return ColorCorrectionDashboard(
            profile=self, image=image, rois=rois,
        )

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the profile to a JSON-compatible dictionary.

        Returns:
            Dictionary containing all parameters and the fitted correction
            matrix (as a nested list).
        """
        data: dict[str, Any] = {
            "checker_type": self.checker_type,
            "degree": self.degree,
            "target_illuminant": self.target_illuminant,
            "median_filter_size": self.median_filter_size,
            "stddev_mag_threshold": self.stddev_mag_threshold,
            "border_distance_threshold": self.border_distance_threshold,
            "core_fraction": self.core_fraction,
            "ridge_lambda": self.ridge_lambda,
            "outlier_sigma": self.outlier_sigma,
            "is_fitted": self._is_fitted,
        }
        if self._correction_matrix is not None:
            data["correction_matrix"] = self._correction_matrix.tolist()
        if self._diagnostics:
            data["diagnostics"] = self._diagnostics
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColorCheckerProfile:
        """Reconstruct a profile from a serialised dictionary.

        Args:
            data: Dictionary previously produced by :meth:`to_dict`.

        Returns:
            Reconstructed ``ColorCheckerProfile`` instance.
        """
        profile = cls(
            checker_type=data.get("checker_type", "ColorChecker24 - After November 2014"),
            degree=data.get("degree", 2),
            target_illuminant=data.get("target_illuminant", "D65"),
            median_filter_size=data.get("median_filter_size", 10),
            stddev_mag_threshold=data.get("stddev_mag_threshold", 15.0),
            border_distance_threshold=data.get("border_distance_threshold", 12.0),
            core_fraction=data.get("core_fraction", 0.5),
            ridge_lambda=data.get("ridge_lambda", 1e-3),
            outlier_sigma=data.get("outlier_sigma", 2.0),
        )
        if data.get("correction_matrix") is not None:
            profile._correction_matrix = np.asarray(
                data["correction_matrix"], dtype=np.float64
            )
        if data.get("diagnostics"):
            profile._diagnostics = data["diagnostics"]
        profile._is_fitted = data.get("is_fitted", False)
        return profile

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"ColorCheckerProfile(checker_type={self.checker_type!r}, "
            f"degree={self.degree}, target_illuminant={self.target_illuminant!r}, "
            f"status={status})"
        )
