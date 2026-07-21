"""Color checker profile for root-polynomial color correction.

Encapsulates the full workflow of extracting patch colors from a color checker
card image, fitting a root-polynomial correction matrix (Finlayson 2015), and
providing serializable diagnostics for quality assessment.
"""

from __future__ import annotations

import logging
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple

import colour
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from phenotypic.sdk_._json_io import read_json_source
from phenotypic.sdk_._io_constants import (
    CONFIG_SUFFIX_COLOR_CHECKER,
    ensure_typed_json_suffix,
)
from phenotypic.sdk_.typing_ import NdArrayField, TuneSpec

from ._capture_metadata import CaptureMetadata
from ._helpers import (
    center_and_pad_checker,
    compute_core_mask,
    compute_swatch_roi_mask,
    geometric_median,
    median_filter_rgb,
    segment_chips_by_border_fill,
    trim_background_edges,
    validate_patch_shape,
)

if TYPE_CHECKING:
    from phenotypic._core._image import Image

logger = logging.getLogger(__name__)


def _in_jupyter_notebook() -> bool:
    """Detect whether code is running inside a Jupyter notebook kernel.

    Returns ``True`` only for notebook kernels (``ZMQInteractiveShell``),
    not plain IPython terminals — used to decide whether to inline-display
    a figure via ``Figure.show()``.

    Returns:
        ``True`` when running in a Jupyter notebook kernel.
    """
    try:
        from IPython import get_ipython as _get_ipython
    except ImportError:
        return False
    shell = _get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


class _RoiPreprocessing(NamedTuple):
    """Stages of per-ROI preprocessing shared by fit and the dashboard.

    Attributes:
        original: Raw RGB crop from the source image.
        trimmed: Result of :func:`trim_background_edges`.  Equals
            *original* when ``pad_checker`` is ``False``.
        filtered: Median-filtered RGB.
        padded: RGB after the optional centring + reflect-pad.  Equals
            *filtered* when ``pad_checker`` is ``False``.
        padded_normed: ``[0, 1]`` float view of *padded* via the canonical
            :pyattr:`Image.rgb.normed`.
        lab: CIE Lab of *padded* via the canonical
            :pyattr:`Image.color.Lab` accessor.
        swatch_roi_mask: 2-D boolean mask, ``True`` where *padded* is a
            swatch-interior pixel and ``False`` on uniform border rows or
            columns (outer frame, central divider, inter-swatch gutters).
            Consumed by :func:`segment_chips_by_border_fill` as the chip
            source so dark-frame pixels cannot be misassigned to a chip.
    """

    original: np.ndarray
    trimmed: np.ndarray
    filtered: np.ndarray
    padded: np.ndarray
    padded_normed: np.ndarray
    lab: np.ndarray
    swatch_roi_mask: np.ndarray

# ---------------------------------------------------------------------------
# ROI-slice field annotation
# ---------------------------------------------------------------------------
#: A single ``slice`` ROI bound. ``slice`` is a Python-native object that
#: pydantic accepts under ``arbitrary_types_allowed`` but cannot describe in
#: JSON Schema, so ``WithJsonSchema`` supplies a descriptive entry. The
#: ``rois`` field is transient (set at construction, excluded from
#: :meth:`~pydantic.BaseModel.model_dump`); the schema entry exists only so
#: ``model_json_schema()`` succeeds for downstream tooling.
_RoiSlice = Annotated[
    slice,
    WithJsonSchema(
            {
                "type"       : "object",
                "description": "A Python slice (start, stop, step) bounding a "
                               "color-checker ROI along one image axis.",
            }
    ),
]

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


class ColorCheckerProfile(BaseModel):
    """Profile fitted from a colour checker card for root-polynomial correction.

    Measures patch colours from one or more ROIs in an image, matches them
    against published reference values for the chosen checker type, rejects
    outlier patches, and solves for a root-polynomial colour correction
    matrix (Finlayson 2015).  The resulting matrix can be applied to entire
    images via :class:`ColorCorrector`.

    This is a pydantic v2 ``BaseModel``: the constructor parameters and the
    post-fit state are declared as annotated class-level fields, so an
    unfitted profile still constructs (``correction_matrix`` is ``None``,
    ``is_fitted`` is ``False`` until :meth:`fit` runs).

    Args:
        checker_type: Key in ``colour.CCS_COLOURCHECKERS``.  Defaults to the
            post-2014 X-Rite ColorChecker 24.
        degree: Root-polynomial degree (1--4).  ``2`` is recommended.
        target_illuminant: Target whitepoint for the correction (e.g.
            ``'D65'``).
        median_filter_size: Kernel size for per-ROI median filtering.
        stddev_mag_threshold: Column-stddev threshold for border detection
            during checker centering.  Ignored when ``pad_checker`` is
            ``False``.
        pad_checker: When ``True``, each ROI is centred and reflect-padded
            via :func:`center_and_pad_checker` to recover partially-clipped
            checker cards.  Defaults to ``False`` because border-fill
            segmentation requires a fully-visible grid with intact gutters:
            reflect-padding fabricates extra mirrored swatch cells, which
            makes the connected-component count exceed the expected number
            of patches and trips the strict count gate.  Only enable it for
            the clustering-era clipped-card workflow.
        min_swatch_area_frac: During border-fill segmentation, connected
            components smaller than this fraction of the median component
            area are discarded as noise before the strict chip-count gate.
        core_fraction: Fraction of centroid-to-boundary distance used to
            define the reliable core of each patch.
        ridge_lambda: Ridge-regression regularisation parameter for the
            root-polynomial fit.
        outlier_sigma: Patches whose Delta-E 2000 exceeds
            ``mean + outlier_sigma * stddev`` are rejected as outliers.
        rois: List of ``(row_slice, col_slice)`` tuples delimiting checker
            card regions in the source image.  Stored for use by
            :meth:`fit` and :meth:`report`.

    Attributes:
        correction_matrix: Fitted correction matrix once :meth:`fit`
            or :meth:`_fit_from_patch_colors` has been called; ``None``
            on an unfitted profile.
        diagnostics: Per-patch and aggregate quality metrics.
        is_fitted: ``True`` after a successful fit.
        capture_metadata: Camera EXIF (make/model, lens, ISO, exposure,
            F-number, focal length) read from the calibration image during
            :meth:`fit`.  ``None`` for an unfitted profile or one fitted via
            :meth:`_fit_from_patch_colors` (no source image).  Serialises as a
            nested object under :meth:`~pydantic.BaseModel.model_dump`; consumed
            by :class:`ColorCorrector` to warn when a corrected image was shot
            on different optics.  See :class:`CaptureMetadata`.
    """

    model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
            extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_fields(cls, data: Any) -> Any:
        """Drop fields removed since older serialised profiles were written.

        Runs before validation, so the obsolete keys are stripped while
        ``extra="forbid"`` still rejects genuine typo'd kwargs (the project's
        strict-construction contract). ``border_distance_threshold`` was
        removed when chip segmentation switched from Lab clustering to
        border-fill.
        """
        if isinstance(data, dict):
            data = {
                k: v for k, v in data.items()
                if k not in {"border_distance_threshold"}
            }
        return data

    # -- constructor parameters --------------------------------------------
    checker_type: str = "ColorChecker24 - After November 2014"
    degree: Annotated[int, TuneSpec(categories=(1, 2, 3, 4))] = 2
    target_illuminant: str = "D65"
    median_filter_size: int = 10
    stddev_mag_threshold: float = 15.0
    pad_checker: bool = False
    min_swatch_area_frac: Annotated[float, TuneSpec(0.1, 0.6)] = 0.3
    core_fraction: Annotated[float, TuneSpec(0.3, 0.8)] = 0.5
    ridge_lambda: Annotated[float, TuneSpec(1e-4, 1e-1, log=True)] = 1e-3
    outlier_sigma: Annotated[float, TuneSpec(1.5, 4.0)] = 2.0
    rois: list[tuple[_RoiSlice, _RoiSlice]] | None = Field(default=None, exclude=True)

    # -- post-fit state (defaults so an unfitted profile still constructs) --
    correction_matrix: NdArrayField | None = None
    diagnostics: dict[str, Any] = {}
    is_fitted: bool = False
    capture_metadata: CaptureMetadata | None = None

    # -- transient (set weakly by fit(), never serialized) -----------------
    _image_ref: "weakref.ReferenceType[Image] | None" = PrivateAttr(default=None)

    @property
    def _image(self) -> Image | None:
        """Return the live calibration image without retaining it strongly."""
        return self._image_ref() if self._image_ref is not None else None

    @field_validator("degree")
    @classmethod
    def _validate_degree(cls, degree: int) -> int:
        """Require ``degree`` in ``{1, 2, 3, 4}`` (pre-migration guard)."""
        if degree not in {1, 2, 3, 4}:
            raise ValueError(f"degree must be 1, 2, 3, or 4, got {degree}")
        return degree

    @field_validator("core_fraction")
    @classmethod
    def _validate_core_fraction(cls, core_fraction: float) -> float:
        """Require ``core_fraction`` in ``(0, 1]`` (pre-migration guard)."""
        if not (0.0 < core_fraction <= 1.0):
            raise ValueError(
                    f"core_fraction must be in (0, 1], got {core_fraction}"
            )
        return core_fraction

    @field_validator("min_swatch_area_frac")
    @classmethod
    def _validate_min_swatch_area_frac(cls, frac: float) -> float:
        """Require ``min_swatch_area_frac`` in ``(0, 1]``."""
        if not (0.0 < frac <= 1.0):
            raise ValueError(
                    f"min_swatch_area_frac must be in (0, 1], got {frac}"
            )
        return frac

    @field_validator("ridge_lambda")
    @classmethod
    def _validate_ridge_lambda(cls, ridge_lambda: float) -> float:
        """Require ``ridge_lambda`` to be non-negative (pre-migration guard)."""
        if ridge_lambda < 0:
            raise ValueError(f"ridge_lambda must be >= 0, got {ridge_lambda}")
        return ridge_lambda

    @field_validator("outlier_sigma")
    @classmethod
    def _validate_outlier_sigma(cls, outlier_sigma: float) -> float:
        """Require ``outlier_sigma`` to be positive (pre-migration guard)."""
        if outlier_sigma <= 0:
            raise ValueError(f"outlier_sigma must be > 0, got {outlier_sigma}")
        return outlier_sigma

    # -- serialisation ------------------------------------------------------

    def to_json(self, filepath: str | Path | None = None) -> str | None:
        """Serialize this profile to JSON.

        Serializes via :meth:`~pydantic.BaseModel.model_dump_json`: the fitted
        ``correction_matrix`` is emitted as a nested list, ``capture_metadata``
        as a nested object, and the transient ``rois`` / ``_image`` fields are
        excluded. Mirrors :meth:`ImagePipeline.to_json`.

        Args:
            filepath: Optional path to write the JSON to. When None, the JSON
                string is returned instead. Accepts a ``str`` or ``Path``.

        Returns:
            The JSON string when ``filepath`` is None, otherwise None.

        Example:
            >>> import tempfile
            >>> from pathlib import Path
            >>> from phenotypic.correction import ColorCheckerProfile
            >>> from phenotypic.sdk_ import CONFIG_SUFFIX_COLOR_CHECKER, ensure_typed_json_suffix
            >>> with tempfile.TemporaryDirectory() as d:
            ...     p = Path(d) / "profile.json"
            ...     saved = ensure_typed_json_suffix(p, CONFIG_SUFFIX_COLOR_CHECKER)
            ...     ColorCheckerProfile(degree=3).to_json(p)
            ...     loaded = ColorCheckerProfile.from_json(saved)
            >>> loaded.degree
            3
        """
        json_str = self.model_dump_json(indent=2)
        if filepath is not None:
            ensure_typed_json_suffix(
                filepath, CONFIG_SUFFIX_COLOR_CHECKER
            ).write_text(json_str)
            return None
        return json_str

    @classmethod
    def from_json(cls, json_data: str | Path | dict) -> ColorCheckerProfile:
        """Reconstruct a profile from JSON written by :meth:`to_json`.

        Accepts a JSON string, a path to a JSON file, or a pre-parsed dict (same
        input handling as :meth:`ImagePipeline.from_json`). Validation runs the
        :meth:`_drop_legacy_fields` hook, so profiles saved before a field was
        removed still load.

        Args:
            json_data: A JSON string, path to a JSON file, or a parsed dict.

        Returns:
            The reconstructed profile instance.
        """
        return cls.model_validate(read_json_source(json_data))

    # -- high-level fitting -------------------------------------------------

    def fit(self, image: Image) -> ColorCheckerProfile:
        """Fit the profile from checker-card ROIs stored at initialisation.

        When ``rois`` was not provided at initialisation, the entire image is
        treated as a single ROI.

        Args:
            image: Source image containing visible checker cards.

        Returns:
            ``self`` for method chaining.
        """
        if self.rois is None:
            self.rois = [(slice(None), slice(None))]
        self._image_ref = weakref.ref(image)
        self.capture_metadata = CaptureMetadata.from_image(image)
        return self._fit_from_rois(image, self.rois)

    def _preprocess_roi(
            self,
            image: Image,
            row_sl: slice,
            col_sl: slice,
    ) -> _RoiPreprocessing:
        """Run per-ROI preprocessing and canonical Lab conversion.

        Shared by :meth:`_fit_from_rois` and the diagnostic report so
        both observe the same pixels: ``pad_checker`` controls whether
        ``trim_background_edges`` and ``center_and_pad_checker`` run, and
        Lab is always produced via :pyattr:`Image.color.Lab` (the same
        XYZ pipeline used everywhere else in the package).

        Args:
            image: Source image (provides ``gamma`` for the wrapped sub-image).
            row_sl: Row slice into ``image.rgb``.
            col_sl: Column slice into ``image.rgb``.

        Returns:
            A :class:`_RoiPreprocessing` with every preprocessing stage.
        """
        from phenotypic._core._image import Image as _Image

        original = image.rgb[row_sl, col_sl].copy()
        trimmed = trim_background_edges(original) if self.pad_checker else original
        filtered = median_filter_rgb(trimmed, size=self.median_filter_size)
        if self.pad_checker:
            padded = center_and_pad_checker(
                    filtered,
                    filter_size=self.median_filter_size,
                    stddev_mag_threshold=self.stddev_mag_threshold,
            )
        else:
            padded = filtered

        sub_image = _Image(
                arr=padded,
                gamma=image.gamma,
                illuminant=self.target_illuminant,  # type: ignore[arg-type]
        )
        lab = sub_image.color.Lab[:]
        swatch_roi_mask = compute_swatch_roi_mask(
                lab,
                stddev_mag_threshold=self.stddev_mag_threshold,
                filter_size=self.median_filter_size,
        )
        if not swatch_roi_mask.any():
            logger.warning(
                    "Cross-channel stddev ROI mask is empty for ROI; falling "
                    "back to a full-ROI mask. No gutters were detected, so "
                    "segmentation will fail the chip-count gate. Consider "
                    "lowering stddev_mag_threshold (currently %.2f).",
                    self.stddev_mag_threshold,
            )
            swatch_roi_mask = np.ones(padded.shape[:2], dtype=bool)
        return _RoiPreprocessing(
                original=original,
                trimmed=trimmed,
                filtered=filtered,
                padded=padded,
                padded_normed=sub_image.rgb.normed(),
                lab=lab,
                swatch_roi_mask=swatch_roi_mask,
        )

    def _fit_from_rois(
            self,
            image: Image,
            rois: list[tuple[slice, slice]],
    ) -> ColorCheckerProfile:
        """Fit the profile from one or more checker-card ROIs in an image.

        For each ROI the function extracts the sub-image, pre-processes it
        (background trimming, median filtering, centering/padding), segments
        chips geometrically as filled connected components of the swatch ROI
        mask and labels them by Hungarian colour match
        (:func:`segment_chips_by_border_fill`), measures the core color of
        each chip with geometric median, and pools the results.  Outlier
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

        ref_Lab_tuples = {
            name: tuple(ref_Lab[name].tolist()) for name in patch_names
        }

        for roi_idx, (row_sl, col_sl) in enumerate(rois):
            prep = self._preprocess_roi(image, row_sl, col_sl)
            padded_float = prep.padded_normed

            # 4. Segment chips geometrically and label them by colour.
            blob_masks, blob_names = segment_chips_by_border_fill(
                    prep.swatch_roi_mask,
                    prep.lab,
                    ref_Lab_tuples,
                    min_swatch_area_frac=self.min_swatch_area_frac,
                    strict=True,
            )

            card_detected = 0

            for blob_mask, name in zip(blob_masks, blob_names):
                # 5. Compute core mask and validate.
                core = compute_core_mask(blob_mask, core_fraction=self.core_fraction)
                _, warnings = validate_patch_shape(core)
                core_pixels_fraction = float(core.sum()) / max(
                        float(blob_mask.sum()), 1.0
                )

                # 6. Measure color from core pixels in the padded sRGB image.
                core_pixels_srgb = padded_float[core]
                if core_pixels_srgb.size == 0:
                    continue

                # Geometric median in sRGB space.
                patch_srgb = geometric_median(core_pixels_srgb)
                measured_srgb[name].append(
                        (patch_srgb, roi_idx, core_pixels_fraction, warnings)
                )
                card_detected += 1

            per_card_summary.append(
                    {
                        "roi_index"       : roi_idx,
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
                "roi_index"          : best[1],
                "core_fraction_used" : best[2],
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

    def _fit_from_patch_colors(
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
                "roi_index"          : 0,
                "core_fraction_used" : 1.0,
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
        measured_arr: np.ndarray = np.zeros((n_kept, 3), dtype=np.float64)
        reference_arr: np.ndarray = np.zeros((n_kept, 3), dtype=np.float64)

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

        self.correction_matrix = CCM

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
            corrected_srgb = np.clip(corrected_srgb, 0.0,
                                     1.0)  # type: ignore[assignment]

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

        self.diagnostics = {
            "checker_type"                      : self.checker_type,
            "degree"                            : self.degree,
            "target_illuminant"                 : self.target_illuminant,
            "n_patches_detected"                : len(measured_Lab),
            "n_patches_expected"                : len(all_patch_names),
            "n_patches_rejected"                : len(rejected),
            "rejected_patches"                  : rejected,
            "per_card_summary"                  : per_card_summary,
            "patches"                           : patches_diag,
            "mean_deltaE00_before"              : float(np.mean(dE_values)),
            "mean_deltaE00_after"               : float(np.mean(after_arr)),
            "max_deltaE00_after"                : float(np.max(after_arr)),
            "median_deltaE00_after"             : float(np.median(after_arr)),
            "correction_matrix_condition_number": cond,
            "warnings"                          : warnings_list,
        }

        self.is_fitted = True
        logger.info(
                "ColorCheckerProfile fitted: %d/%d patches, "
                "mean dE00 %.2f -> %.2f",
                len(measured_Lab),
                len(all_patch_names),
                self.diagnostics["mean_deltaE00_before"],
                self.diagnostics["mean_deltaE00_after"],
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

    # -- report ------------------------------------------------------------

    def report(self, show: bool = True) -> Any:
        """Build the Plotly color-correction diagnostic report.

        Constructs a :class:`ColorCorrectionReport` from this fitted profile and
        returns its :meth:`~phenotypic.abc_.plotting.PhtPlot.report` rendering.
        The report declares no interactive controls, so ``report()`` composes the
        per-section figures into a single ``plotly.graph_objects.Figure``.

        Uses the weak image reference and ROIs recorded during :meth:`fit`.
        Callers must keep the source image alive while rendering. When the
        image has been released, or the profile was fitted from patch colors,
        the pipeline-step and segmentation figures are omitted.

        Args:
            show: When ``True`` and running inside a Jupyter notebook, display
                the figure inline before returning it. Set ``False`` in tests or
                for programmatic use.

        Returns:
            A ``plotly.graph_objects.Figure`` (always returned, even after an
            inline display).

        Raises:
            RuntimeError: If the profile has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                    "Cannot create report for an unfitted profile."
            )
        from ._color_correction_report import ColorCorrectionReport

        image = self._image_ref() if self._image_ref is not None else None
        report = ColorCorrectionReport(profile=self, image=image, rois=self.rois)
        figure = report.report()

        if show and _in_jupyter_notebook():
            figure.show()

        return figure

    # -- serialisation ------------------------------------------------------
    #
    # ``ColorCheckerProfile`` is a pydantic ``BaseModel``: serialise via
    # :meth:`~pydantic.BaseModel.model_dump` (use ``mode="json"`` for a
    # JSON-native dict — ``NdArrayField`` rewrites ``correction_matrix`` to a
    # nested list, and ``rois`` is field-level ``exclude=True``) and round-trip
    # via :meth:`~pydantic.BaseModel.model_validate`.

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        n_rois = len(self.rois) if self.rois else 0
        return (
            f"ColorCheckerProfile(checker_type={self.checker_type!r}, "
            f"degree={self.degree}, target_illuminant={self.target_illuminant!r}, "
            f"rois={n_rois}, status={status})"
        )
