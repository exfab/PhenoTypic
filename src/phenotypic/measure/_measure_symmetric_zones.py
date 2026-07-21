"""Mask-based radial expansion and symmetry measurement operator.

Implements :class:`MeasureSymmetricZones`, a branch-free operator that answers
two colony-level questions directly from the binary object mask: how far has
growth progressed past the inoculum, and out to what radius does that growth
remain angularly uniform?
"""

from __future__ import annotations

import weakref
from dataclasses import fields, replace
from typing import ClassVar, Literal, TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import pandas as pd
from pydantic import PrivateAttr
from skimage.measure import approximate_polygon, find_contours, regionprops

from phenotypic.abc_ import MeasureFeatures
from phenotypic.abc_.plotting import Control, PlotImage, figure
from phenotypic.measure._zone_segmentation import (
    ZoneSegmentation,
    ZoneSegmentationParams,
    compute_zone_segmentation,
)
from phenotypic.schema import OBJECT
from phenotypic.schema import SYMMETRIC_ZONES

# Back-compat alias so the rest of this module (inspect/_operate/overlay
# helpers) keeps referring to the relocated dataclass by its old name.
_SymmetryIntermediates: TypeAlias = ZoneSegmentation


def _own_intermediate_arrays(
    intermediates: _SymmetryIntermediates,
) -> _SymmetryIntermediates:
    """Return compact intermediates whose arrays own their memory.

    A cropped NumPy view can retain the complete plate-sized backing array.
    Plot caches therefore copy every derived array before retaining the
    per-object record. Scalar measurement results are unchanged.
    """
    owned = {
        item.name: value.copy()
        for item in fields(intermediates)
        if isinstance((value := getattr(intermediates, item.name)), np.ndarray)
    }
    return replace(intermediates, **owned)

_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)

# Zone-segmentation constants (retained for the overlay helpers below).
_N_ANGULAR_SECTORS = 360

# Diagnostic-overlay constants
_ZONE_POLYGON_STRIDE = 5  # 360 / 5 = 72 vertices per zone polygon
_OBJMAP_POLYGON_TOLERANCE = 0.5  # Douglas-Peucker pixel tolerance

# Okabe-Ito palette constants for diagnostic plots
_OI_NAVY = "#003660"
_OI_ORANGE = "#E69F00"
_OI_SKY = "#56B4E9"
_OI_GREEN = "#009E73"
_OI_BLUE = "#0072B2"
_OI_PURPLE = "#CC79A7"
_OI_VERMILION = "#D55E00"
_OI_GREY = "#BBBBBB"

# Which image array backs the plotly overlay; a select Control bound (by identity)
# to the inspect() figure's ``base_layer`` kwarg for the interactive dashboard.
BASE_LAYER = Control(
        label="Base layer",
        kind="select",
        default="gray",
        options=("rgb", "gray", "detect_mat"),
        help="Image array rendered behind the symmetric-radius overlay.",
)


class MeasureSymmetricZones(MeasureFeatures, PlotImage):
    """Measure colony radial expansion and angular symmetry from the object mask alone.

    Quantifies each colony by four scalars derived directly from its binary
    mask and distance-from-inoculum map — no skeletonization, no branch
    tracing, no runner outlier flagging. The headline output is
    ``SymmetricRadius``, the first radius past the inoculum core at which the
    per-annulus circular mean resultant length of mask-boundary pixels drops
    below a tunable symmetry threshold. ``CoreRadius`` (PELT changepoint on
    the radial density profile) anchors the measurement; ``MeanExpansion``
    and ``MaxExpansion`` summarise how far growth reached past that core.

    Zone segmentation (core / dense / sparse) uses a 1D per-annulus
    **normalised colony-ness** signal ``c(r)`` computed from the ring-wide
    mean intensity. After calibrating ``I_core`` and ``I_agar`` from
    percentiles of the expanded crop, each ring's mean intensity is mapped
    into ``[0, 1]`` where 1 = pure colony and 0 = pure agar. For a
    roughly circular colony, ``c(r)`` decreases monotonically outward
    (uniform dense core ≈ 1, mixed dense branching ≈ 0.5–0.8, sparse
    branching ≈ 0.1–0.4, agar = 0), so zone boundaries follow directly
    from threshold crossings — no peak-finding. Zones are emitted as
    concentric circles at the three scalar radii. Variance and raw mean
    per ring are retained in the intermediates for diagnostic access but
    do not drive segmentation.

    The mask's only role in this signal is to (1) isolate which colony is
    being analysed, (2) seed the inoculum centre as the peak of the
    in-mask Euclidean distance transform, and (3) detect the
    colony-vs-agar intensity direction so the normalisation handles dark
    colonies (gray convention) and bright colonies (detect_mat
    convention) uniformly. The ring accumulation extends out to
    ``r_max = max_mask_radius × (1 + extent_margin)`` regardless of mask
    boundaries, so background pixels contribute truthfully to the signal.

    Args:
        n_annuli: Number of equal-area annuli used for the radial density
            profile and angular analysis. Defaults to 100.
        pelt_penalty: PELT penalty controlling changepoint sensitivity for
            core detection. Defaults to 5.0.
        symmetry_threshold: Minimum angular coverage (fraction of angular
            bins occupied) for growth to be considered symmetric. Defaults
            to 4/6 (~0.667); with 6 angular bins this means at least 4
            of 6 60-degree sectors must contain mask pixels.
        n_angular_bins: Number of angular bins used to compute the
            per-annulus angular coverage diagnostic. Defaults to 36
            (10 degree resolution).
        smoothing_window: Moving-average window (in annuli) applied to the
            angular R̄ profile before the threshold test. Defaults to 3.
        method: Inoculum centre estimator --- ``"distance"`` uses the peak
            of the Euclidean distance transform, ``"intensity"`` uses the
            intensity-weighted centroid. Defaults to ``"distance"``.
        extent_margin: Fractional expansion of the ring accumulator past
            the farthest mask pixel, so the outer annuli sample a small
            agar tail for the ``I_agar`` reference. Defaults to 0.05
            (5%) — deliberately small because tight plates risk touching
            neighbouring colonies.
        min_samples_per_ring: Minimum pixel count required to compute
            a mean for a given ring; below this the ring's mean is
            filled by linear interpolation from its neighbours before
            normalisation. Defaults to 5.
        tau_core: Colony-ness threshold that marks the core/dense
            boundary. The core zone extends out to the last ring where
            ``c(r) ≥ tau_core``. Defaults to 0.9 (last "≥90% colony"
            ring).
        tau_dense: Colony-ness threshold that marks the dense/sparse
            boundary. The dense zone extends to the last ring where
            ``c(r) ≥ tau_dense``. Defaults to 0.5 (last "majority
            colony" ring).
        tau_sparse: Colony-ness threshold that marks the sparse/outside
            boundary. The sparse zone extends to the last ring where
            ``c(r) ≥ tau_sparse``, capped at the mask envelope. Defaults
            to 0.1.
        intensity_source: Image array used for the mean-intensity
            calculation -- ``"gray"`` uses the grayscale (dark = colony),
            ``"detect_mat"`` uses the detection matrix (bright = colony).
            Direction is auto-detected. Defaults to ``"gray"``.

    Returns:
        pd.DataFrame: Object-level radial symmetry measurements with
        columns:

            - Object_Label: unique object identifier.
            - SymZones_CoreRadius: inoculum core radius (pixels).
            - SymZones_SymmetricRadius: first radius past the core
              where R̄ exceeds the symmetry threshold (pixels).
            - SymZones_MeanExpansion: mean boundary-pixel distance
              beyond the core (pixels, clamped at 0).
            - SymZones_MaxExpansion: maximum mask-pixel distance
              beyond the core (pixels, clamped at 0).
            - SymZones_CoreEndRadius: mean per-angle core boundary
              radius from the bright-fraction outward walk (pixels).
            - SymZones_DenseEndRadius: mean per-angle outer radius
              of the dense branching zone (pixels).
            - SymZones_SparseEndRadius: mean per-angle outer radius
              of the sparse branching zone, capped at the symmetric
              envelope (pixels).
            - SymZones_CoreArea: pixel^2 area of the inoculum core
              zone integrated across the 360-sector polar polygon.
            - SymZones_DenseArea: pixel^2 area of the dense
              branching zone (annular region between core and dense
              boundaries).
            - SymZones_SparseArea: pixel^2 area of the sparse
              branching zone (annular region between dense and outer
              boundaries).

    Best For:
        - Summarising colony-level radial growth with a single symmetry
          figure of merit.
        - Distinguishing uniformly-expanding colonies from those with
          sectors, lopsided growth, or directional bias.
        - Comparing wild-type versus mutant expansion phenotypes when the
          biological question is about the colony envelope, not individual
          hyphae.
        - Time-course assays where runner counts are noisy but colony
          extent is informative.

    Consider Also:
        - :class:`MeasureShape` for general morphological descriptors
          (circularity, eccentricity) that do not require radial analysis.
        - :class:`MeasureBounds` for lightweight bounding-box data
          without any radial pipeline.

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of measuring and exporting colony data.
    """

    _measurement_infoclass: ClassVar[type] = SYMMETRIC_ZONES

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
    intensity_source: Literal["gray", "detect_mat"] = "gray"

    # Diagnostic cache populated by ``measure()`` so ``inspect()`` /
    # ``napari()`` can reuse the per-object intermediates. Pure runtime
    # state — never a constructor parameter — so modeled as private attrs.
    __cache_image_ref: "weakref.ReferenceType[Image] | None" = PrivateAttr(
        default=None
    )
    __cache_intermediates: "dict[int, _SymmetryIntermediates]" = PrivateAttr(
            default_factory=dict)
    __cache_signature: str | None = PrivateAttr(default=None)

    # ── shared pipeline for one object ───────────────────────────────

    def _zone_params(self) -> ZoneSegmentationParams:
        """Snapshot this operator's zone parameters for the shared pipeline."""
        return ZoneSegmentationParams(
                n_annuli=self.n_annuli,
                pelt_penalty=self.pelt_penalty,
                symmetry_threshold=self.symmetry_threshold,
                n_angular_bins=self.n_angular_bins,
                smoothing_window=self.smoothing_window,
                method=self.method,
                extent_margin=self.extent_margin,
                min_samples_per_ring=self.min_samples_per_ring,
                tau_core=self.tau_core,
                tau_dense=self.tau_dense,
                tau_sparse=self.tau_sparse,
                intensity_source=self.intensity_source,
        )

    def _compute_intermediates(
            self,
            image: Image,
            object_label: int | None = None,
            prop=None,
    ) -> ZoneSegmentation:
        """Run the shared symmetric-radius pipeline for a single object.

        Thin delegator to
        :func:`phenotypic.measure._zone_segmentation.compute_zone_segmentation`.
        Every caller in this module supplies ``prop`` explicitly; the shared
        helper resolves the largest object by area when ``prop`` is *None*.

        Args:
            image: Detected Image with objmap/objmask.
            object_label: Retained for signature back-compat; unused because
                callers always pass ``prop``.
            prop: Pre-computed RegionProperties object for the target object.

        Returns:
            ZoneSegmentation with all computed fields populated.
        """
        return compute_zone_segmentation(image, prop, params=self._zone_params())

    # ── MeasureFeatures interface ────────────────────────────────────

    def _operate(self, image: Image) -> pd.DataFrame:
        """Populate the symmetric-radius measurement DataFrame.

        Args:
            image: Detected Image with objmap/objmask.

        Returns:
            pd.DataFrame with one row per detected object and a leading
            ``OBJECT.LABEL`` column followed by the four SYMMETRIC_ZONES
            columns.
        """
        measurements = {
            str(feature): np.full(image.num_objects, np.nan)
            for feature in SYMMETRIC_ZONES
            if feature != SYMMETRIC_ZONES.CATEGORY
        }

        props = regionprops(
                image.objmap[:],
                intensity_image=image.gray[:].astype(np.float64, copy=False),
        )

        # Refresh cache so inspect() can reuse these results
        self.__cache_image_ref = weakref.ref(image)
        self.__cache_intermediates = {}

        # Zone columns get 0.0 for tiny objects (no zones to resolve);
        # the original four columns stay NaN to disambiguate "not measurable"
        # from legitimately-zero values.
        _zero_for_tiny = (
            SYMMETRIC_ZONES.CORE_END_RADIUS,
            SYMMETRIC_ZONES.DENSE_END_RADIUS,
            SYMMETRIC_ZONES.SPARSE_END_RADIUS,
            SYMMETRIC_ZONES.CORE_AREA,
            SYMMETRIC_ZONES.DENSE_AREA,
            SYMMETRIC_ZONES.SPARSE_AREA,
        )

        for idx, prop in enumerate(props):
            if prop.area < 10:
                for feat in _zero_for_tiny:
                    measurements[str(feat)][idx] = 0.0
                continue

            try:
                inter = self._compute_intermediates(image, prop.label, prop=prop)
            except Exception:
                import logging

                logging.getLogger(__name__).debug(
                        "Skipping object label %d", prop.label, exc_info=True,
                )
                continue  # leave NaN

            self.__cache_intermediates[prop.label] = _own_intermediate_arrays(inter)

            measurements[str(SYMMETRIC_ZONES.CORE_RADIUS)][idx] = inter.core_radius
            measurements[str(SYMMETRIC_ZONES.SYMMETRIC_RADIUS)][
                idx] = inter.symmetric_radius
            measurements[str(SYMMETRIC_ZONES.MEAN_EXPANSION)][
                idx] = inter.mean_expansion
            measurements[str(SYMMETRIC_ZONES.MAX_EXPANSION)][idx] = inter.max_expansion
            measurements[str(SYMMETRIC_ZONES.CORE_AREA)][idx] = inter.core_area
            measurements[str(SYMMETRIC_ZONES.DENSE_AREA)][idx] = inter.dense_area
            measurements[str(SYMMETRIC_ZONES.SPARSE_AREA)][idx] = inter.sparse_area
            measurements[str(SYMMETRIC_ZONES.CORE_END_RADIUS)][
                idx] = inter.core_end_radius
            measurements[str(SYMMETRIC_ZONES.DENSE_END_RADIUS)][
                idx] = inter.dense_end_radius
            measurements[str(SYMMETRIC_ZONES.SPARSE_END_RADIUS)][
                idx] = inter.sparse_end_radius

        df = pd.DataFrame(measurements)
        df.insert(0, OBJECT.LABEL, image.objects.labels2series())
        self.__cache_signature = self.model_dump_json()
        return df

    # ── cache helpers ────────────────────────────────────────────────

    def _require_cache_image(self) -> Image:
        """Return the cached image or raise if :meth:`measure` has not run."""
        image = self.__cached_image()
        if image is None:
            raise RuntimeError(
                    "MeasureSymmetricZones: no live image subject is available. "
                    "Pass an image to .inspect(), or keep the measured image "
                    "alive for no-argument rendering."
            )
        return image

    def __cached_image(self) -> Image | None:
        """Return the weakly held image without extending its lifetime."""
        return (
            self.__cache_image_ref()
            if self.__cache_image_ref is not None
            else None
        )

    def _get_local_obj_mask(self, label: int) -> np.ndarray:
        """Local (bbox-cropped) boolean mask for the object with ``label``."""
        inter = self.__cache_intermediates[label]
        return inter.obj_mask

    def _get_local_dist_map(self, label: int) -> np.ndarray:
        """Local distance-from-centroid map for the object with ``label``."""
        inter = self.__cache_intermediates[label]
        return inter.dist_map

    def _get_global_offset(self, label: int) -> tuple[int, int]:
        """Top-left (row, col) offset from the local bbox to plate coordinates."""
        inter = self.__cache_intermediates[label]
        slc = inter.bbox_slice
        return int(slc[0].start), int(slc[1].start)

    # ── overlay polygon helpers ──────────────────────────────────────

    @staticmethod
    def _object_polygon_xy(
            local_mask: np.ndarray,
            slc: tuple[slice, slice],
            tolerance: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Largest simplified mask contour as plate-coordinate (x, y) arrays.

        The mask is padded by one zero pixel before contour extraction so
        objects that touch the local bbox edge still produce closed
        contours. Without padding, ``find_contours`` returns an open
        curve for edge-touching masks and plotly's ``fill="toself"``
        closes it with a straight last→first segment that slices across
        the object.

        Args:
            local_mask: Boolean mask in local bbox coordinates.
            slc: ``(row_slice, col_slice)`` mapping the local bbox to
                plate coordinates.
            tolerance: Douglas-Peucker tolerance in pixels.

        Returns:
            ``(xs, ys)`` float ndarrays for the simplified polygon in
            plate coordinates, or ``(None, None)`` if no usable contour
            exists (mask empty or contour collapses below 3 vertices).
        """
        if not local_mask.any():
            return None, None

        padded = np.pad(local_mask, 1, mode="constant", constant_values=False)
        contours = find_contours(padded.astype(np.float64), 0.5)
        if not contours:
            return None, None

        contour = max(contours, key=len)
        if contour.shape[0] < 3:
            return None, None

        simplified = approximate_polygon(contour, tolerance=tolerance)
        if simplified.shape[0] < 3:
            return None, None

        # Subtract 1 to undo the padding offset before mapping to plate coords.
        r0 = float(slc[0].start)
        c0 = float(slc[1].start)
        ys = simplified[:, 0] - 1.0 + r0
        xs = simplified[:, 1] - 1.0 + c0
        return xs, ys

    @staticmethod
    def _polar_polygon_xy(
            centroid_xy: tuple[float, float],
            r_per_angle: np.ndarray,
            stride: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Closed polar polygon as plate-coordinate (x, y) arrays.

        ``r_per_angle`` is the per-degree radius array (length 360);
        ``stride`` downsamples it (``stride=5`` → 72 vertices). The
        polygon is closed by repeating the first vertex.

        Args:
            centroid_xy: ``(cx, cy)`` centroid in plate coordinates.
            r_per_angle: (360,) per-degree radii.
            stride: Subsampling stride applied to the angle axis.

        Returns:
            Tuple ``(xs, ys)`` of float64 arrays, both length
            ``ceil(360 / stride) + 1`` (closed).
        """
        cx, cy = centroid_xy
        angles_deg: np.ndarray = np.arange(
            0, _N_ANGULAR_SECTORS, stride, dtype=np.int32
        )
        theta = np.deg2rad(angles_deg)
        radii = r_per_angle[angles_deg].astype(np.float64)
        xs = cx + radii * np.cos(theta)
        ys = cy + radii * np.sin(theta)
        xs = np.concatenate([xs, xs[:1]])
        ys = np.concatenate([ys, ys[:1]])
        return xs, ys

    @staticmethod
    def _polar_annulus_xy(
            centroid_xy: tuple[float, float],
            r_inner_per_angle: np.ndarray,
            r_outer_per_angle: np.ndarray,
            stride: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Closed polar annulus as plate-coordinate (x, y) arrays.

        Traces the outer ring forward then the inner ring in reverse,
        producing a ring polygon that plotly's ``fill="toself"`` renders
        correctly (the reversed inner winding subtracts the inner disk
        from the outer disk without self-intersection).

        Args:
            centroid_xy: ``(cx, cy)`` centroid in plate coordinates.
            r_inner_per_angle: (360,) per-degree inner radii.
            r_outer_per_angle: (360,) per-degree outer radii.
            stride: Subsampling stride applied to the angle axis.

        Returns:
            Tuple ``(xs, ys)`` of float64 arrays, closed.
        """
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        cx, cy = centroid_xy
        angles_deg: np.ndarray = np.arange(
            0, _N_ANGULAR_SECTORS, stride, dtype=np.int32
        )
        theta = np.deg2rad(angles_deg)
        r_out = r_outer_per_angle[angles_deg].astype(np.float64)
        r_in = r_inner_per_angle[angles_deg].astype(np.float64)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        outer_x = cx + r_out * cos_t
        outer_y = cy + r_out * sin_t
        inner_x = cx + r_in * cos_t
        inner_y = cy + r_in * sin_t
        # Outer ring forward, inner ring reversed, close back to the first
        # outer vertex. The opposite windings let plotly's non-zero-fill
        # "toself" render the ring (the inner traversal subtracts the hole).
        xs = np.concatenate([outer_x, inner_x[::-1], outer_x[:1]])
        ys = np.concatenate([outer_y, inner_y[::-1], outer_y[:1]])
        return xs, ys

    # ── diagnostics ──────────────────────────────────────────────────

    @figure(
            title="Symmetric-radius overlay",
            primary=True,
            controls={"base_layer": BASE_LAYER},
    )
    def inspect(
            self,
            image: Image | None = None,
            base_layer: Literal["rgb", "gray", "detect_mat"] = "gray",
            *,
            for_save: bool = False,
    ):
        """Plate-level diagnostic overlay for symmetric-radius measurement.

        Args:
            image: Detected Image with objmap/objmask. If *None*, the
                image cached by the most recent :meth:`measure` call is
                reused.
            base_layer: Which image array to use as the plotly background.
            for_save: When *True*, every overlay trace is force-shown
                (no ``visible="legendonly"``) so the figure renders
                meaningfully as a static raster. The CLI's
                PlotImage publication passes this. Defaults to *False*
                (interactive Jupyter use, with overlay layers toggleable
                from the legend).

        Returns:
            plotly.graph_objects.Figure with toggleable overlay layers.
            Renders natively in Jupyter via the plotly mime bundle. For
            scroll-to-zoom, call
            ``fig.show(config={"scrollZoom": True})``.
        """
        from phenotypic.sdk_._plotly_helpers import _require_plotly

        _require_plotly()

        import plotly.graph_objects as go

        valid_base_layers = BASE_LAYER.options or ()
        if base_layer not in valid_base_layers:
            allowed = ", ".join(repr(value) for value in valid_base_layers)
            raise ValueError(
                f"base_layer must be one of {allowed}; got {base_layer!r}"
            )

        if image is None:
            image = self._require_cache_image()

        props = regionprops(
                image.objmap[:],
                intensity_image=image.gray[:].astype(np.float64, copy=False),
        )
        intermediates_cache: dict[int, _SymmetryIntermediates] = {}
        cache_is_current = (
            self.__cached_image() is image
            and self.__cache_signature == self.model_dump_json()
        )
        for prop in props:
            if (
                    cache_is_current
                    and prop.label in self.__cache_intermediates
            ):
                intermediates_cache[prop.label] = self.__cache_intermediates[prop.label]
                continue
            try:
                inter = self._compute_intermediates(image, prop.label, prop=prop)
                intermediates_cache[prop.label] = _own_intermediate_arrays(inter)
            except Exception:
                continue

        self.__cache_image_ref = weakref.ref(image)
        self.__cache_intermediates = intermediates_cache
        self.__cache_signature = self.model_dump_json()

        if not intermediates_cache:
            fig = go.Figure()
            fig.add_annotation(
                    text="No objects found for symmetric-radius analysis.",
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(color=_OI_NAVY),  # family from the phenotypic template
            )
            return fig

        fig = self._build_plate_overview(
                image, intermediates_cache,
                base_layer=base_layer,
        )
        h, w = image.gray[:].shape[:2]
        overview_h = int(900 * h / w)

        fig.update_layout(
                title=dict(
                        text=f"Symmetric Radius -- {len(intermediates_cache)} objects",
                        font=dict(color=_OI_NAVY),  # family from the phenotypic template
                ),
                height=overview_h,
        )

        if for_save:
            for trace in fig.data:
                if getattr(trace, "visible", True) == "legendonly":
                    trace.visible = True
        return fig

    @staticmethod
    def _build_plate_overview(
            image: Image,
            intermediates_cache: dict[int, _SymmetryIntermediates],
            base_layer: str = "gray",
    ):
        """Build a zoomable plotly plate overview with toggleable overlays.

        Args:
            image: Detected Image with objmap/objmask.
            intermediates_cache: Pre-computed intermediates keyed by label.
            base_layer: Which image array to use as background.

        Returns:
            plotly.graph_objects.Figure.
        """
        from phenotypic.sdk_._plotly_helpers import (
            plotly_imshow,
        )

        if base_layer == "rgb":
            arr = image.rgb[:]
        elif base_layer == "detect_mat":
            arr = image.detect_mat[:]
        else:
            arr = image.gray[:]

        h, w = arr.shape[:2]
        display_w = 900
        display_h = int(display_w * h / w)
        fig = plotly_imshow(
                arr, title="Plate Overview",
                figsize=(display_w // 100, display_h // 100),
        )
        fig.update_coloraxes(showscale=False)

        MeasureSymmetricZones._add_overlay_traces(
                fig, image, intermediates_cache,
        )
        return fig

    @staticmethod
    def _add_overlay_traces(
            fig,
            image: Image,
            intermediates_cache: dict[int, _SymmetryIntermediates],
    ) -> None:
        """Add toggleable trace layers to the plate overview.

        Layers (all legend-toggleable):
        - **Objmap** — simplified mask contour per object, filled with a
          label-keyed overlay palette.
        - **Zones** — three legend entries grouped under the "Zones"
          heading: **Sparse zone** (sky, annulus r_dense_end..r_outer),
          **Dense zone** (navy, annulus r_core..r_dense_end), and
          **Core zone** (vermilion, disk 0..r_core). ``r_outer`` is capped
          at the symmetric envelope. The legend group is set to toggle
          as a group, so clicking any entry hides/shows all three; the
          entries can also be toggled individually.
        - **Centroids** — inoculum centre markers for every object.
        - **Core radius** — dashed vermilion circles.
        - **Symmetric radius** — purple dashed circles (all objects).
        - **Outer envelope** — solid green per-angle polygon tracing the
          uncapped mask reach per object. Drawn for every object with any
          mask pixels, including those whose symmetry thresholding failed
          (``zones_computed=False``); those objects intentionally have no
          Zones traces.

        Args:
            fig: Plotly figure to modify in-place.
            image: Detected Image (used for objmap overlay).
            intermediates_cache: Per-object intermediates keyed by label.
        """
        import plotly.graph_objects as go

        _N_CIRCLE_PTS = 72
        _theta = np.linspace(0, 2 * np.pi, _N_CIRCLE_PTS, endpoint=True)

        def _circle_xy(cx: float, cy: float, r: float):
            return cx + r * np.cos(_theta), cy + r * np.sin(_theta)

        def _hex_to_rgba(hex_str: str, alpha: float) -> str:
            h = hex_str.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r}, {g}, {b}, {alpha:.3f})"

        # Make grouped legend entries toggle every trace in the group.
        fig.update_layout(legend=dict(groupclick="togglegroup"))

        _FILL_ALPHA = 0.30

        # ── Objmap polygons (label-keyed palette, NaN-separated buckets) ──
        # Use the same overlay palette as skimage-based objmap overlays so
        # the Objmap layer here matches colors elsewhere in phenotypic, and
        # key it on ``label`` (not enumeration index) so neighbouring objects
        # cycle through the whole palette rather than colliding.
        from phenotypic._core._image_parts.accessor_abstracts._image_accessor_base_parents._accessor_mpl_handler import (  # noqa: E501
            AccessorMplHandler,
        )

        _palette_rgb = (AccessorMplHandler._OVERLAY_COLORS * 255).astype(np.int32)
        n_palette = len(_palette_rgb)

        bucket_xs: list[list[float]] = [[] for _ in range(n_palette)]
        bucket_ys: list[list[float]] = [[] for _ in range(n_palette)]
        sorted_inters = sorted(
                intermediates_cache.values(), key=lambda x: x.label,
        )
        for inter in sorted_inters:
            xs, ys = MeasureSymmetricZones._object_polygon_xy(
                    inter.obj_mask, inter.bbox_slice, _OBJMAP_POLYGON_TOLERANCE,
            )
            if xs is None or ys is None:
                continue
            # Labels start at 1; subtract before mod so label 1 → palette 0.
            bi = (int(inter.label) - 1) % n_palette
            bucket_xs[bi].extend(xs.tolist())
            bucket_xs[bi].append(float("nan"))
            bucket_ys[bi].extend(ys.tolist())
            bucket_ys[bi].append(float("nan"))

        first_objmap_drawn = False
        for bi, (bxs, bys) in enumerate(zip(bucket_xs, bucket_ys)):
            if not bxs:
                continue
            r, g, b = _palette_rgb[bi]
            fillcolor = f"rgba({int(r)}, {int(g)}, {int(b)}, {_FILL_ALPHA:.3f})"
            fig.add_trace(go.Scatter(
                    x=bxs, y=bys, mode="lines",
                    line=dict(width=0),
                    fill="toself",
                    fillcolor=fillcolor,
                    legendgroup="objmap",
                    name="Objmap",
                    showlegend=not first_objmap_drawn,
                    visible="legendonly",
                    hoverinfo="skip",
            ))
            first_objmap_drawn = True

        # ── Zone polygons — nested annuli + core disk ───────────────
        # Core: solid disk 0..r_core. Dense: annulus r_core..r_dense_end.
        # Sparse: annulus r_dense_end..r_outer (capped at symmetric radius).
        # Drawn sparse → dense → core so narrower zones render on top of
        # wider ones and the legend-group toggle handles all three at once.
        sparse_xs: list[float] = []
        sparse_ys: list[float] = []
        dense_xs: list[float] = []
        dense_ys: list[float] = []
        zcore_xs: list[float] = []
        zcore_ys: list[float] = []

        for inter in intermediates_cache.values():
            if not inter.zones_computed:
                continue
            slc = inter.bbox_slice
            r0, c0 = int(slc[0].start), int(slc[1].start)
            cxy = (inter.centroid_rc[1] + c0, inter.centroid_rc[0] + r0)

            # Zone radii are scalars; broadcast to per-angle arrays so the
            # existing polar-polygon helpers draw concentric circles.
            r_core_arr: np.ndarray = np.full(
                    _N_ANGULAR_SECTORS, inter.core_end_radius, dtype=np.float64,
            )
            r_dense_arr: np.ndarray = np.full(
                    _N_ANGULAR_SECTORS, inter.dense_end_radius, dtype=np.float64,
            )
            r_outer_arr: np.ndarray = np.full(
                    _N_ANGULAR_SECTORS, inter.sparse_end_radius, dtype=np.float64,
            )

            sxs, sys = MeasureSymmetricZones._polar_annulus_xy(
                    cxy, r_dense_arr, r_outer_arr, _ZONE_POLYGON_STRIDE,
            )
            sparse_xs.extend(sxs.tolist())
            sparse_xs.append(float("nan"))
            sparse_ys.extend(sys.tolist())
            sparse_ys.append(float("nan"))

            dxs, dys = MeasureSymmetricZones._polar_annulus_xy(
                    cxy, r_core_arr, r_dense_arr, _ZONE_POLYGON_STRIDE,
            )
            dense_xs.extend(dxs.tolist())
            dense_xs.append(float("nan"))
            dense_ys.extend(dys.tolist())
            dense_ys.append(float("nan"))

            cxs, cys = MeasureSymmetricZones._polar_polygon_xy(
                    cxy, r_core_arr, _ZONE_POLYGON_STRIDE,
            )
            zcore_xs.extend(cxs.tolist())
            zcore_xs.append(float("nan"))
            zcore_ys.extend(cys.tolist())
            zcore_ys.append(float("nan"))

        # Three legend entries share legendgroup="zones"; groupclick is set
        # to "togglegroup" above so clicking any of them toggles all three at
        # once, while individual toggles are still available.
        zone_layers = [
            (sparse_xs, sparse_ys, _OI_SKY, "Sparse zone"),
            (dense_xs, dense_ys, _OI_NAVY, "Dense zone"),
            (zcore_xs, zcore_ys, _OI_VERMILION, "Core zone"),
        ]
        for zxs, zys, zcolor, zname in zone_layers:
            if not zxs:
                continue
            fig.add_trace(go.Scatter(
                    x=zxs, y=zys, mode="lines",
                    line=dict(width=0),
                    fill="toself",
                    fillcolor=_hex_to_rgba(zcolor, _FILL_ALPHA),
                    legendgroup="zones",
                    legendgrouptitle_text="Zones",
                    name=zname,
                    visible="legendonly",
                    hoverinfo="skip",
            ))

        # ── Centroids ───────────────────────────────────────────────
        cent_x, cent_y = [], []
        for inter in intermediates_cache.values():
            slc = inter.bbox_slice
            r0, c0 = int(slc[0].start), int(slc[1].start)
            cent_x.append(inter.centroid_rc[1] + c0)
            cent_y.append(inter.centroid_rc[0] + r0)
        if cent_x:
            fig.add_trace(go.Scatter(
                    x=cent_x, y=cent_y, mode="markers",
                    marker=dict(
                            color=_OI_ORANGE, size=8, symbol="circle",
                            line=dict(color=_OI_NAVY, width=1),
                    ),
                    name="Centroids",
                    hoverinfo="skip",
            ))

        # ── Core radius circles (all objects) ───────────────────────
        core_x: list[float] = []
        core_y: list[float] = []
        for inter in intermediates_cache.values():
            if inter.core_radius <= 0:
                continue
            slc = inter.bbox_slice
            r0, c0 = int(slc[0].start), int(slc[1].start)
            cx = inter.centroid_rc[1] + c0
            cy = inter.centroid_rc[0] + r0
            xs, ys = _circle_xy(cx, cy, inter.core_radius)
            core_x.extend(xs.tolist())
            core_y.extend(ys.tolist())
            core_x.append(float("nan"))
            core_y.append(float("nan"))
        if core_x:
            fig.add_trace(go.Scattergl(
                    x=core_x, y=core_y, mode="lines",
                    line=dict(color=_OI_VERMILION, width=1.5, dash="dash"),
                    name="Core radius",
                    hoverinfo="skip",
            ))

        # ── Symmetric radius circles (all objects) ──────────────────
        sym_x: list[float] = []
        sym_y: list[float] = []
        for inter in intermediates_cache.values():
            sr = inter.symmetric_radius
            if sr <= 0 or not np.isfinite(sr):
                continue
            slc = inter.bbox_slice
            r0, c0 = int(slc[0].start), int(slc[1].start)
            cx = inter.centroid_rc[1] + c0
            cy = inter.centroid_rc[0] + r0
            xs, ys = _circle_xy(cx, cy, sr)
            sym_x.extend(xs.tolist())
            sym_y.extend(ys.tolist())
            sym_x.append(float("nan"))
            sym_y.append(float("nan"))
        if sym_x:
            fig.add_trace(go.Scattergl(
                    x=sym_x, y=sym_y, mode="lines",
                    line=dict(color=_OI_PURPLE, width=2.5, dash="dash"),
                    name="Symmetric radius",
                    hoverinfo="skip",
            ))

        # ── Outer envelope polygons (per-angle, uncapped mask reach) ──
        env_x: list[float] = []
        env_y: list[float] = []
        for inter in intermediates_cache.values():
            r_env = inter.r_outer_full_per_angle
            if float(r_env.max()) <= 0:
                continue
            slc = inter.bbox_slice
            r0, c0 = int(slc[0].start), int(slc[1].start)
            cxy = (inter.centroid_rc[1] + c0, inter.centroid_rc[0] + r0)
            xs, ys = MeasureSymmetricZones._polar_polygon_xy(
                    cxy, r_env, _ZONE_POLYGON_STRIDE,
            )
            env_x.extend(xs.tolist())
            env_y.extend(ys.tolist())
            env_x.append(float("nan"))
            env_y.append(float("nan"))
        if env_x:
            fig.add_trace(go.Scattergl(
                    x=env_x, y=env_y, mode="lines",
                    line=dict(color=_OI_GREEN, width=1.25),
                    name="Outer envelope",
                    hoverinfo="skip",
            ))


MeasureSymmetricZones.__doc__ = SYMMETRIC_ZONES.append_rst_to_doc(
        MeasureSymmetricZones
)
