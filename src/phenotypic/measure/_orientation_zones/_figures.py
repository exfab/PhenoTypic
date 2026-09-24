"""Diagnostic figures for :class:`MeasureOrientationZones`.

The figure methods live on a private mixin so the operation module reads as a
measurement. Every method here is reached through ``self``; the attributes it
needs from the operation are declared in the ``TYPE_CHECKING`` block below,
which is the complete list of what the figures depend on.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Iterator, Literal, cast

import numpy as np

from phenotypic.abc_.plotting import Control, PlotImage, figure
from phenotypic.sdk_._palette import (
    OKABE_ITO_GREEN,
    OKABE_ITO_NAVY,
    OKABE_ITO_ORANGE,
    OKABE_ITO_SKY,
    OKABE_ITO_VERMILION,
)
from phenotypic.sdk_._radial_geometry import circle_xy
from phenotypic.sdk_.orientation_fields import (
    FIBER_AXIS_OFFSET,
    MIN_AXIAL_RESULTANT,
    N_SECTORS,
    RELIABLE_PIXEL_COHERENCE,
    cumulative_ring_rotation_profile,
    fiber_bend_field,
    matched_ring_cumulative_rotation_profile,
    matched_tracks_to_ring_sector_values,
    radial_ring_orientation_profile,
    radial_ring_sector_field,
    signed_radial_relative_field,
)

from ._common import _EPS, _VARIANTS, _ZONES

if TYPE_CHECKING:
    from phenotypic.measure._zone_segmentation import ZoneSegmentation

    from ._common import _ObjectZoneAnalysis
    from ._operation import MeasureOrientationZones

# Three opacity levels make local structure-tensor confidence visible without
# creating one plotly trace per line segment. Blocks below the low cutoff are
# omitted because their orientation is not reliably defined.
_QUIVER_COHERENCE_BINS = (
    ("Low C", 0.15, 0.40, 0.30),
    ("Medium C", 0.40, 0.70, 0.55),
    ("High C", 0.70, np.inf, 0.90),
)

# Base-layer selector for inspect(); mirrors MeasureSymZones.BASE_LAYER
# but defaults to "detect_mat" (the tensor/segmentation source for this op).
BASE_LAYER = Control(
        label="Base layer",
        kind="select",
        default="detect_mat",
        options=("rgb", "gray", "detect_mat"),
        help="Image array rendered behind the orientation-field overlay.",
)

_BEND_SCALE_PRESETS = {
    "fine"    : (2.0, 4.0, 8.0),
    "balanced": (4.0, 8.0, 16.0),
    "broad"   : (8.0, 16.0, 32.0),
}


class _OrientationZonesFigures(PlotImage):
    """Figure methods of :class:`MeasureOrientationZones` (private mixin)."""

    if TYPE_CHECKING:
        # Operation state and measurement helpers the figures read.
        legacy_mode: bool
        include_diagnostics: bool
        long_range_lag: float
        outer_zone_percentile: float
        quiver_block: int
        radial_ring_width: float
        _cache: dict
        _cache_image_ref: weakref.ReferenceType[object] | None
        _cache_signature: str | None

        def measure(self, image: Any, *args: Any, **kwargs: Any) -> Any: ...
        def model_dump_json(self, *args: Any, **kwargs: Any) -> str: ...
        def _prep(self, image: Any) -> tuple[list, dict]: ...
        def _analyze_objects(
            self, image: Any, props: Any, label2section: dict
        ) -> Iterator[_ObjectZoneAnalysis]: ...
        def _lacks_zone_evidence(self, seg: ZoneSegmentation) -> bool: ...
        def _orientation_outer_radius(self, seg: ZoneSegmentation) -> float: ...
        @staticmethod
        def _zone_selector(
            dist_map: np.ndarray,
            r_lo: float,
            r_hi: float,
            obj_mask: np.ndarray,
            variant: str,
            *,
            include_upper: bool = False,
        ) -> np.ndarray: ...

    def _coherence_canvas(self, image, downsample: int = 4):
        """Recompute per-object coherence and composite onto a plate canvas.

        Used only by report()'s heatmap. Full-res fields are recomputed via
        _analyze_objects and discarded here — the heatmap costs compute, not
        persistent memory. Returned canvas is downsampled for a light figure.
        """
        props, label2section = self._prep(image)
        canvas = np.full(image.gray[:].shape[:2], np.nan)
        for analysis in self._analyze_objects(image, props, label2section):
            seg = analysis.segmentation
            r0 = int(round(seg.centroid_global[0] - analysis.center[0]))
            c0 = int(round(seg.centroid_global[1] - analysis.center[1]))
            h, w = analysis.coherence.shape
            r1, c1 = min(r0 + h, canvas.shape[0]), min(c0 + w, canvas.shape[1])
            canvas[max(r0, 0): r1, max(c0, 0): c1] = analysis.coherence[
                : r1 - max(r0, 0), : c1 - max(c0, 0)
            ]
        return canvas[::downsample, ::downsample]

    def _require_cache_image(self):
        """Return the cached image or raise if :meth:`measure` has not run."""
        image = self._cached_image()
        if image is None:
            raise RuntimeError(
                    "MeasureOrientationZones: no live image subject is available. "
                    "Pass an image to .inspect()/.report(), or keep the measured "
                    "image alive for no-argument rendering."
            )
        return image

    def _cached_image(self):
        """Return the weakly held image subject without extending its lifetime."""
        return (
            self._cache_image_ref()
            if self._cache_image_ref is not None
            else None
        )

    def _ensure_diagnostic_cache(self, image) -> None:
        """Measure ``image`` when its compact diagnostic cache is stale."""
        if (
                not self._cache
                or self._cached_image() is not image
                or self._cache_signature != self.model_dump_json()
        ):
            self.measure(image)

    @figure(
            title="Orientation-field overlay",
            backend="plotly",
            primary=True,
            controls={"base_layer": BASE_LAYER},
    )
    def inspect(
            self,
            image=None,
            base_layer: Literal["rgb", "gray", "detect_mat"] = "detect_mat",
            *,
            for_save: bool = False,
    ):
        """Plate overview of the pixels and local fiber axes being aggregated.

        Uses the compact per-object cache populated by the most recent
        :meth:`measure` call for axes, rings, means, and hover values. The signed
        outward-turning overlay is recomputed on demand and discarded after the
        figure is assembled. Local axes are clipped to the overall radial
        selector, rotated from the structure-tensor gradient normal to the fiber
        axis, and confidence-coded by coherence. Zone metrics are available by
        hovering the colony centre.

        Args:
            image: Detected Image with objmap. If *None*, the image cached by the
                most recent :meth:`measure` call is reused.
            base_layer: Which image array to render behind the overlay
                (``"rgb"``, ``"gray"`` or ``"detect_mat"``).
            for_save: When *True*, every legend-only overlay trace is force-shown
                so the figure renders meaningfully as a static raster (the CLI's
                PlotImage publication passes this). Defaults to *False*.

        Returns:
            A ``plotly.graph_objects.Figure`` with toggleable overlay layers.

        Examples:
            >>> from phenotypic.data import load_synth_filamentous_plate
            >>> from phenotypic.measure import MeasureOrientationZones
            >>> op = MeasureOrientationZones()
            >>> _ = op.measure(load_synth_filamentous_plate())
            >>> fig = op.inspect()
            >>> len(fig.data) > 0
            True
        """
        from phenotypic.sdk_._plotly_helpers import (
            _require_plotly,
            plotly_imshow,
        )

        _require_plotly()

        valid_base_layers = BASE_LAYER.options or ()
        if base_layer not in valid_base_layers:
            allowed = ", ".join(repr(value) for value in valid_base_layers)
            raise ValueError(
                    f"base_layer must be one of {allowed}; got {base_layer!r}"
            )

        if image is None:
            image = self._require_cache_image()
        self._ensure_diagnostic_cache(image)

        base = getattr(image, base_layer)[:]
        h, w = base.shape[:2]
        display_w = 900
        display_h = int(display_w * h / w)
        fig = plotly_imshow(
                base,
                title="Orientation zones: local fiber axes and measurement regions",
                figsize=(display_w // 100, display_h // 100),
        )
        fig.update_coloraxes(showscale=False)
        fig.update_layout(legend=dict(groupclick="togglegroup"))

        self._add_signed_outward_turning_trace(fig, image, (h, w))
        self._add_long_range_ring_traces(fig)
        self._add_mask_selector_trace(fig, image)
        self._add_quiver_trace(fig)
        self._add_zone_ring_traces(fig)
        self._add_mean_axis_traces(fig)
        self._add_metric_hover_trace(fig)
        fig.update_xaxes(range=[-0.5, w - 0.5], constrain="domain")
        fig.update_yaxes(
                range=[h - 0.5, -0.5],
                scaleanchor="x",
                scaleratio=1,
        )
        if for_save:
            for trace in fig.data:
                if getattr(trace, "visible", True) == "legendonly":
                    trace.visible = True
        return fig

    @figure(
            title="Cumulative radial rotation overlay",
            backend="plotly",
            controls={"base_layer": BASE_LAYER},
    )
    def cumulative_rotation_overlay(
            self,
            image=None,
            base_layer: Literal["rgb", "gray", "detect_mat"] = "detect_mat",
    ):
        """Show accumulated ring-to-ring orientation change on the source layer.

        Each angular sector's first supported ring outside the inferred
        inoculum is its zero reference. Adjacent seam-safe axial changes are
        unwrapped and summed while radial support remains continuous. The
        result is painted only onto detected structure that contributed to the
        calculation. This is a visualization-only view; it does not add a
        branch-density measurement or alter exported metrics.

        Args:
            image: Detected Image with objmap. If *None*, reuse the image cached
                by the most recent :meth:`measure` call.
            base_layer: Image array shown beneath the Spectral overlay
                (``"rgb"``, ``"gray"`` or ``"detect_mat"``).

        Returns:
            A ``plotly.graph_objects.Figure`` with cumulative signed rotation in
            degrees, sampled ring boundaries, and optional local fiber axes.

        Examples:
            >>> from phenotypic.data import load_synth_filamentous_plate
            >>> from phenotypic.measure import MeasureOrientationZones
            >>> op = MeasureOrientationZones()
            >>> _ = op.measure(load_synth_filamentous_plate())
            >>> fig = op.cumulative_rotation_overlay()
            >>> len(fig.data) > 0
            True
        """
        from phenotypic.sdk_._plotly_helpers import (
            _require_plotly,
            plotly_imshow,
        )

        _require_plotly()
        valid_base_layers = BASE_LAYER.options or ()
        if base_layer not in valid_base_layers:
            allowed = ", ".join(repr(value) for value in valid_base_layers)
            raise ValueError(
                    f"base_layer must be one of {allowed}; got {base_layer!r}"
            )
        if image is None:
            image = self._require_cache_image()
        self._ensure_diagnostic_cache(image)

        base = getattr(image, base_layer)[:]
        height, width = base.shape[:2]
        display_width = 900
        display_height = int(display_width * height / width)
        fig = plotly_imshow(
                base,
                title=(
                    "Cumulative radial rotation: accumulated change from the "
                    "first supported ring in each sector"
                ),
                figsize=(display_width // 100, display_height // 100),
        )
        fig.update_coloraxes(showscale=False)
        self._add_cumulative_rotation_trace(fig, image, (height, width))
        self._add_long_range_ring_traces(fig)
        self._add_quiver_trace(fig)
        fig.add_annotation(
                text=(
                    "<b>Cumulative rotation:</b> zero at each sector's first "
                    f"supported {self.radial_ring_width:g} px ring outside the "
                    "inoculum; "
                    "then the signed, seam-safe change between adjacent rings is "
                    "summed outward within each angular sector.<br>"
                    "Positive and negative values are opposite turning senses. "
                    "Signed unwrapping assumes adjacent-ring changes are less "
                    "than 90°. "
                    "Blank regions are the excluded inoculum, background, or a "
                    "sector after continuous radial support was lost. Short blue "
                    "bars show the local fiber axes and can be toggled in the "
                    "legend."
                ),
                xref="paper",
                yref="paper",
                x=0.0,
                y=-0.19,
                xanchor="left",
                yanchor="top",
                align="left",
                showarrow=False,
                font=dict(color=OKABE_ITO_NAVY, size=11),
        )
        fig.update_layout(margin=dict(b=300))
        fig.update_xaxes(range=[-0.5, width - 0.5], constrain="domain")
        fig.update_yaxes(
                range=[height - 0.5, -0.5],
                scaleanchor="x",
                scaleratio=1,
        )
        return fig

    @figure(
            title="Matched-ring cumulative fiber rotation overlay",
            backend="plotly",
            controls={"base_layer": BASE_LAYER},
    )
    def matched_cumulative_rotation_overlay(
            self,
            image=None,
            base_layer: Literal["rgb", "gray", "detect_mat"] = "detect_mat",
            *,
            max_sector_shift: int = 2,
            allow_gap_bridging: bool = False,
            allow_restarts: bool = False,
    ):
        """Show fiber-axis rotation accumulated along nearby matched ring cells.

        Unlike :meth:`cumulative_rotation_overlay`, which remains in each fixed
        angular sector and accumulates radial-relative tilt changes, this
        diagnostic follows reliable fiber-axis means into nearby sectors on the
        next annular band. It accumulates seam-safe fiber-axis changes along the
        matched path. The inferred inoculum core and unsupported path segments
        remain blank. This view does not alter exported measurements.

        Args:
            image: Detected Image with objmap. If *None*, reuse the image cached
                by the most recent :meth:`measure` call.
            base_layer: Image array shown beneath the Spectral overlay
                (``"rgb"``, ``"gray"`` or ``"detect_mat"``).
            max_sector_shift: Maximum nearby 10-degree sector displacement
                allowed between adjacent rings.
            allow_gap_bridging: Whether a path may scan past rings with no
                reliable nearby candidate. Skipped rings remain blank.
            allow_restarts: Whether a terminated seed may start a new segment
                at its next reliable cell. Restarted segments reset cumulative
                rotation to zero and are not inoculum-path measurements.

        Returns:
            A ``plotly.graph_objects.Figure`` with cumulative signed fiber-axis
            rotation in degrees and the matched annular paths.

        Raises:
            ValueError: If ``base_layer`` is unsupported or
                ``max_sector_shift`` is not an integer greater than or equal to
                zero, or either continuity flag is not boolean.

        Examples:
            >>> from phenotypic.data import load_synth_filamentous_plate
            >>> from phenotypic.measure import MeasureOrientationZones
            >>> op = MeasureOrientationZones()
            >>> _ = op.measure(load_synth_filamentous_plate())
            >>> fig = op.matched_cumulative_rotation_overlay()
            >>> len(fig.data) > 0
            True
        """
        from phenotypic.sdk_._plotly_helpers import (
            _require_plotly,
            plotly_imshow,
        )

        _require_plotly()
        valid_base_layers = BASE_LAYER.options or ()
        if base_layer not in valid_base_layers:
            allowed = ", ".join(repr(value) for value in valid_base_layers)
            raise ValueError(
                    f"base_layer must be one of {allowed}; got {base_layer!r}"
            )
        if isinstance(max_sector_shift, bool) or not isinstance(
                max_sector_shift, (int, np.integer)
        ):
            raise ValueError("max_sector_shift must be an integer >= 0")
        if max_sector_shift < 0:
            raise ValueError("max_sector_shift must be an integer >= 0")
        if not isinstance(allow_gap_bridging, (bool, np.bool_)):
            raise ValueError("allow_gap_bridging must be a boolean")
        if not isinstance(allow_restarts, (bool, np.bool_)):
            raise ValueError("allow_restarts must be a boolean")
        if image is None:
            image = self._require_cache_image()
        self._ensure_diagnostic_cache(image)

        base = getattr(image, base_layer)[:]
        height, width = base.shape[:2]
        display_width = 900
        display_height = int(display_width * height / width)
        fig = plotly_imshow(
                base,
                title=(
                    "Matched-ring cumulative fiber rotation: nearby annular "
                    "orientation cells are connected outward"
                ),
                figsize=(display_width // 100, display_height // 100),
        )
        fig.update_coloraxes(showscale=False)
        self._add_matched_cumulative_rotation_trace(
                fig,
                image,
                (height, width),
                max_sector_shift=max_sector_shift,
                allow_gap_bridging=allow_gap_bridging,
                allow_restarts=allow_restarts,
        )
        self._add_long_range_ring_traces(fig)
        path_note = (
            "The signed, seam-safe fiber-axis changes are summed along the "
            "white path. "
            if not allow_restarts
            else "White path lines are hidden because restarted segment "
                 "boundaries are not continuous trajectories. "
        )
        fig.add_annotation(
                text=(
                    "<b>Matched-ring accumulation:</b> each geometric seed starts "
                    f"at its first supported {self.radial_ring_width:g} px ring. "
                    f"The next ring may move up to {max_sector_shift} nearby "
                    "10° sectors, guided by outward radial-relative tilt, fiber-axis "
                    "continuity, and orientation reliability.<br>"
                    f"Gap bridging is {'enabled' if allow_gap_bridging else 'disabled'}; "
                    f"segment restarts are {'enabled' if allow_restarts else 'disabled'}. "
                    "Skipped rings remain blank. Restarted segments reset to 0° and "
                    "are segment-relative rather than cumulative from the inoculum.<br>"
                    f"{path_note}Spectral colors use the fixed full range from "
                    "−180° to +180°. Blank regions are inoculum, background, or "
                    "terminated/unsupported paths."
                ),
                xref="paper",
                yref="paper",
                x=0.0,
                y=-0.19,
                xanchor="left",
                yanchor="top",
                align="left",
                showarrow=False,
                font=dict(color=OKABE_ITO_NAVY, size=11),
        )
        fig.update_layout(margin=dict(b=300))
        fig.update_xaxes(range=[-0.5, width - 0.5], constrain="domain")
        fig.update_yaxes(
                range=[height - 0.5, -0.5],
                scaleanchor="x",
                scaleratio=1,
        )
        return fig

    def fiber_bend_overlay(
            self,
            image=None,
            base_layer: Literal["rgb", "gray", "detect_mat"] = "detect_mat",
            scale_set: Literal["fine", "balanced", "broad"] = "balanced",
    ):
        """Compare director-line curvature at three Q-averaging scales.

        This prototype is diagnostic-only. It recomputes mask-aware fiber bend
        from the same structure-tensor field used by the existing orientation
        metrics, but it does not change measurements, schemas, caches, the
        primary :meth:`inspect` figure, or the cumulative radial reference.
        Bend is the nonnegative curvature magnitude of the local fiber-director
        integral curves, reported in degrees per pixel.

        Args:
            image: Detected Image with objmap. If *None*, reuse the image cached
                by the most recent :meth:`measure` call.
            base_layer: Image array shown under each bend panel (``"rgb"``,
                ``"gray"`` or ``"detect_mat"``).
            scale_set: Three Q-field Gaussian standard deviations in pixels:
                ``"fine"`` is 2/4/8, ``"balanced"`` is 4/8/16, and
                ``"broad"`` is 8/16/32.

        Returns:
            A three-panel ``plotly.graph_objects.Figure`` with each scale's
            complete observed bend-magnitude range.

        Examples:
            >>> from phenotypic.data import load_synth_filamentous_plate
            >>> from phenotypic.measure import MeasureOrientationZones
            >>> op = MeasureOrientationZones()
            >>> _ = op.measure(load_synth_filamentous_plate())
            >>> fig = op.fiber_bend_overlay(scale_set="balanced")
            >>> len(fig.data) >= 3
            True
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        from phenotypic.sdk_._plotly_helpers import _require_plotly

        _require_plotly()
        valid_base_layers = BASE_LAYER.options or ()
        if base_layer not in valid_base_layers:
            allowed = ", ".join(repr(value) for value in valid_base_layers)
            raise ValueError(
                    f"base_layer must be one of {allowed}; got {base_layer!r}"
            )
        if scale_set not in _BEND_SCALE_PRESETS:
            allowed = ", ".join(repr(value) for value in _BEND_SCALE_PRESETS)
            raise ValueError(
                    f"scale_set must be one of {allowed}; got {scale_set!r}"
            )
        if image is None:
            image = self._require_cache_image()
        self._ensure_diagnostic_cache(image)

        base = np.asarray(getattr(image, base_layer)[:])
        height, width = base.shape[:2]
        scales = _BEND_SCALE_PRESETS[scale_set]
        rasters, raw_peaks, stride = self._bend_scale_rasters(
                image,
                (height, width),
                scales,
        )
        subplot_titles = [
            (
                f"Q scale σ={scale:g} px"
                f"<br><sup>raw peak {peak:.3f} deg/px</sup>"
            )
            for scale, peak in zip(scales, raw_peaks)
        ]
        fig = make_subplots(
                rows=1,
                cols=3,
                shared_yaxes=True,
                horizontal_spacing=0.025,
                subplot_titles=subplot_titles,
        )
        raster_height, raster_width = rasters[0].shape
        raster_x = (np.arange(raster_width) + 0.5) * stride - 0.5
        raster_y = (np.arange(raster_height) + 0.5) * stride - 0.5
        display_base = base[::stride, ::stride]
        base_is_rgb = base.ndim == 3
        finite_base = display_base[np.isfinite(display_base)]
        if finite_base.size:
            base_min, base_max = np.percentile(finite_base, (1.0, 99.8))
            if base_max <= base_min:
                base_max = base_min + _EPS
        else:
            base_min, base_max = 0.0, 1.0
        if base_is_rgb:
            rgb = np.asarray(display_base, dtype=np.float64)
            rgb = (rgb - base_min) / (base_max - base_min)
            display_base = np.clip(
                    np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0),
                    0.0,
                    1.0,
            )
            display_base = np.round(display_base * 255.0).astype(np.uint8)
        for column, (scale, raster) in enumerate(
                zip(scales, rasters),
                start=1,
        ):
            coloraxis_name = (
                "coloraxis" if column == 1 else f"coloraxis{column}"
            )
            if base_is_rgb:
                fig.add_trace(
                        go.Image(
                                z=display_base,
                                x0=float(raster_x[0]),
                                y0=float(raster_y[0]),
                                dx=stride,
                                dy=stride,
                                hoverinfo="skip",
                                name=base_layer,
                        ),
                        row=1,
                        col=column,
                )
            else:
                fig.add_trace(
                        go.Heatmap(
                                x=raster_x,
                                y=raster_y,
                                z=display_base,
                                colorscale=((0.0, "black"), (1.0, "white")),
                                zmin=float(base_min),
                                zmax=float(base_max),
                                showscale=False,
                                hoverinfo="skip",
                                name=base_layer,
                                showlegend=False,
                        ),
                        row=1,
                        col=column,
                )
            fig.add_trace(
                    go.Heatmap(
                            x=raster_x,
                            y=raster_y,
                            z=raster,
                            coloraxis=coloraxis_name,
                            opacity=0.72,
                            name=f"Fiber bend σ={scale:g} px",
                            hovertemplate=(
                                f"Q scale={scale:g} px<br>"
                                "Fiber bend=%{z:.3f} deg/px<extra></extra>"
                            ),
                            connectgaps=False,
                    ),
                    row=1,
                    col=column,
            )
        coloraxis_layout: dict[str, dict] = {}
        for column, peak in enumerate(raw_peaks, start=1):
            coloraxis_name = (
                "coloraxis" if column == 1 else f"coloraxis{column}"
            )
            full_range = peak
            if not np.isfinite(full_range) or full_range <= _EPS:
                full_range = float(np.finfo(np.float32).eps)
            coloraxis_layout[coloraxis_name] = dict(
                    colorscale="Viridis",
                    cmin=0.0,
                    cmax=full_range,
                    colorbar=dict(
                            title=dict(text="Bend (deg/px)", side="top"),
                            orientation="h",
                            x=(column - 0.5) / 3.0,
                            xanchor="center",
                            y=-0.08,
                            yanchor="top",
                            len=0.27,
                            thickness=12,
                    ),
            )
        fig.update_layout(
                title=(
                    "Multiscale fiber bend: curvature along the local director "
                    "field"
                ),
                width=1500,
                height=max(520, min(1050, int(500 * height / width + 280))),
                margin=dict(b=260),
                showlegend=False,
                **coloraxis_layout,
        )
        for index in range(1, 4):
            xaxis_name = "xaxis" if index == 1 else f"xaxis{index}"
            yaxis_name = "yaxis" if index == 1 else f"yaxis{index}"
            xref = "x" if index == 1 else f"x{index}"
            fig.layout[xaxis_name].update(
                    range=[-0.5, width - 0.5],
                    constrain="domain",
                    showticklabels=False,
            )
            fig.layout[yaxis_name].update(
                    range=[height - 0.5, -0.5],
                    scaleanchor=xref,
                    scaleratio=1,
                    showticklabels=False,
            )
        fig.add_annotation(
                text=(
                    "<b>Interpretation:</b> brighter values are stronger local "
                    "curvature along the fiber-orientation field. Scale controls "
                    "Q-field averaging, not the structure-tensor scales. The "
                    "inoculum, background, low-coherence pixels, and mixed "
                    "scale-local orientations are blank.<br>"
                    "A feature that persists across scales is less likely to be a "
                    "single-scale texture artifact. Each panel uses its own full "
                    "raw range, so compare spatial persistence across panels, not "
                    "color brightness.<br>Bend is unsigned because a fiber director "
                    "has no intrinsic arrowhead. Existing radial and cumulative "
                    "figures remain the signed references."
                ),
                xref="paper",
                yref="paper",
                x=0.0,
                y=-0.18,
                xanchor="left",
                yanchor="top",
                align="left",
                showarrow=False,
                font=dict(color=OKABE_ITO_NAVY, size=11),
        )
        return fig

    def _bend_scale_rasters(
            self,
            image,
            image_shape: tuple[int, int],
            scales: tuple[float, ...],
    ) -> tuple[list[np.ndarray], list[float], int]:
        """Composite scale-local fiber bend into bounded plate rasters."""
        props, label2section = self._prep(image)
        height, width = image_shape
        stride = max(1, int(np.ceil(max(height, width) / 900.0)))
        raster_height = int(np.ceil(height / stride))
        raster_width = int(np.ceil(width / stride))
        maxima = [
            np.full(raster_height * raster_width, -np.inf, dtype=np.float32)
            for _scale in scales
        ]
        raw_peaks = [0.0 for _scale in scales]
        for analysis in self._analyze_objects(image, props, label2section):
            seg = analysis.segmentation
            if self._lacks_zone_evidence(seg):
                continue
            inner_radius = float(seg.core_end_radius)
            outer_radius = self._orientation_outer_radius(seg)
            if inner_radius <= _EPS or outer_radius <= inner_radius:
                continue
            selector = self._zone_selector(
                    analysis.distance_map,
                    inner_radius,
                    outer_radius,
                    analysis.object_mask,
                    "Mask",
                    include_upper=not self.legacy_mode,
            )
            origin_row = int(
                round(seg.centroid_global[0] - analysis.center[0])
            )
            origin_col = int(
                round(seg.centroid_global[1] - analysis.center[1])
            )
            for scale_index, scale in enumerate(scales):
                bend, scale_resultant = fiber_bend_field(
                        analysis.orientation,
                        analysis.coherence,
                        selector,
                        scale,
                )
                valid = (
                        selector
                        & np.isfinite(bend)
                        & np.isfinite(analysis.coherence)
                        & np.isfinite(scale_resultant)
                        & (
                            analysis.coherence
                            >= RELIABLE_PIXEL_COHERENCE
                        )
                        & (scale_resultant >= MIN_AXIAL_RESULTANT)
                )
                if not valid.any():
                    continue
                rows, cols = np.nonzero(valid)
                global_rows = rows + origin_row
                global_cols = cols + origin_col
                inside = (
                        (global_rows >= 0)
                        & (global_rows < height)
                        & (global_cols >= 0)
                        & (global_cols < width)
                )
                if not inside.any():
                    continue
                values = np.degrees(bend[valid]).astype(np.float32)[inside]
                raw_peaks[scale_index] = max(
                        raw_peaks[scale_index],
                        float(np.max(values)),
                )
                flat_ids = (
                                   global_rows[inside] // stride
                           ) * raster_width + global_cols[inside] // stride
                np.maximum.at(maxima[scale_index], flat_ids, values)
        rasters: list[np.ndarray] = []
        for scale_maxima in maxima:
            raster = scale_maxima.reshape(raster_height, raster_width)
            raster = np.where(np.isfinite(raster), raster, np.nan).astype(
                    np.float32,
                    copy=False,
            )
            rasters.append(raster)
        return rasters, raw_peaks, stride

    def _add_long_range_ring_traces(self, fig) -> None:
        """Overlay sampled orientation bands and fixed-lag pair midpoints.

        At most eight rings per object are drawn to keep plate-scale figures
        responsive. The complete fixed-width profile still drives the metrics;
        display subsampling does not change any calculation. Line traces skip
        hover; one compact marker per displayed circle carries the values.
        """
        import plotly.graph_objects as go

        ring_xs: list[float | None] = []
        ring_ys: list[float | None] = []
        ring_marker_x: list[float] = []
        ring_marker_y: list[float] = []
        ring_hover: list[str] = []
        pair_xs: list[float | None] = []
        pair_ys: list[float | None] = []
        pair_marker_x: list[float] = []
        pair_marker_y: list[float] = []
        pair_hover: list[str] = []
        for record in self._cache.values():
            profile = record.get("ring_profile", {})
            radii = np.asarray(profile.get("radii", []), dtype=float)
            absolute = np.asarray(
                    profile.get("mean_absolute_tilt", []), dtype=float
            )
            signed = np.asarray(
                    profile.get("mean_signed_tilt", []), dtype=float
            )
            support = np.asarray(profile.get("support", []), dtype=float)
            if radii.size == 0:
                continue
            step = max(1, int(np.ceil(radii.size / 8.0)))
            indices = list(range(0, radii.size, step))
            if indices[-1] != radii.size - 1:
                indices.append(radii.size - 1)
            cy, cx = record["centroid_global"]
            for index in indices:
                radius = float(radii[index])
                circle_x, circle_y = circle_xy(cx, cy, radius)
                hover = (
                    "<b>Sholl-style orientation band</b><br>"
                    f"Radius={radius:.1f} px<br>"
                    f"Mean |radial tilt|={absolute[index]:.2f}°<br>"
                    f"Mean signed radial tilt={signed[index]:.2f}°<br>"
                    f"Sector support={support[index]:.3f}"
                )
                ring_xs.extend([*circle_x.tolist(), None])
                ring_ys.extend([*circle_y.tolist(), None])
                ring_marker_x.append(float(cx + radius))
                ring_marker_y.append(float(cy))
                ring_hover.append(hover)

            pair_midpoints = np.asarray(
                    profile.get("pair_midpoints", []), dtype=float
            )
            pair_absolute = np.asarray(
                    profile.get("mean_absolute_rotation", []), dtype=float
            )
            pair_signed = np.asarray(
                    profile.get("mean_signed_rotation", []), dtype=float
            )
            pair_support = np.asarray(
                    profile.get("pair_support", []), dtype=float
            )
            if pair_midpoints.size == 0:
                continue
            pair_step = max(1, int(np.ceil(pair_midpoints.size / 8.0)))
            pair_indices = list(range(0, pair_midpoints.size, pair_step))
            if pair_indices[-1] != pair_midpoints.size - 1:
                pair_indices.append(pair_midpoints.size - 1)
            for index in pair_indices:
                radius = float(pair_midpoints[index])
                circle_x, circle_y = circle_xy(cx, cy, radius)
                hover = (
                    "<b>Fixed-lag ring comparison</b><br>"
                    f"Pair midpoint={radius:.1f} px<br>"
                    f"Radial lag={self.long_range_lag:g} px<br>"
                    f"Mean |rotation|={pair_absolute[index]:.2f}°<br>"
                    f"Mean signed rotation={pair_signed[index]:.2f}°<br>"
                    f"Paired-sector support={pair_support[index]:.3f}"
                )
                pair_xs.extend([*circle_x.tolist(), None])
                pair_ys.extend([*circle_y.tolist(), None])
                pair_marker_x.append(float(cx + radius))
                pair_marker_y.append(float(cy))
                pair_hover.append(hover)

        if ring_xs:
            fig.add_trace(
                    go.Scattergl(
                            x=ring_xs,
                            y=ring_ys,
                            mode="lines",
                            line=dict(color="rgba(255,255,255,0.72)", width=0.8),
                            name=(
                                f"Orientation rings ({self.radial_ring_width:g} px bands)"
                            ),
                            legendgroup="long-range-rings",
                            legendgrouptitle_text="Long-range radial rotation",
                            hoverinfo="skip",
                    )
            )
            fig.add_trace(
                    go.Scatter(
                            x=ring_marker_x,
                            y=ring_marker_y,
                            text=ring_hover,
                            mode="markers",
                            marker=dict(
                                    size=6,
                                    color="white",
                                    line=dict(color=OKABE_ITO_NAVY, width=0.8),
                            ),
                            name="Ring orientation (hover)",
                            legendgroup="long-range-rings",
                            showlegend=False,
                            hovertemplate="%{text}<extra></extra>",
                    )
            )
        if pair_xs:
            fig.add_trace(
                    go.Scattergl(
                            x=pair_xs,
                            y=pair_ys,
                            mode="lines",
                            line=dict(color=OKABE_ITO_ORANGE, width=1.2, dash="dot"),
                            name=f"{self.long_range_lag:g} px pair midpoints",
                            legendgroup="long-range-rings",
                            hoverinfo="skip",
                    )
            )
            fig.add_trace(
                    go.Scatter(
                            x=pair_marker_x,
                            y=pair_marker_y,
                            text=pair_hover,
                            mode="markers",
                            marker=dict(
                                    size=7,
                                    symbol="diamond",
                                    color=OKABE_ITO_ORANGE,
                                    line=dict(color=OKABE_ITO_NAVY, width=0.8),
                            ),
                            name="Long-range rotation (hover)",
                            legendgroup="long-range-rings",
                            showlegend=False,
                            hovertemplate="%{text}<extra></extra>",
                    )
            )

    def _add_signed_outward_turning_trace(
            self,
            fig,
            image,
            image_shape: tuple[int, int],
    ) -> None:
        """Overlay reliable signed outward turning as a bounded raster.

        The field is recomputed from the same ``intensity_source`` and tensor
        scales used for measurement. Values are not averaged, clipped, or
        percentile-normalized. The Spectral colorscale spans the complete
        observed range symmetrically around zero so equal clockwise and
        counterclockwise rates receive equal color distance from the midpoint.
        The raster is capped at the inspect display resolution; each display
        cell retains the signed source pixel with the greatest absolute rate,
        preventing opposite directions from cancelling while avoiding one
        Plotly marker per source pixel.
        """
        import plotly.graph_objects as go

        props, label2section = self._prep(image)
        height, width = image_shape
        stride = max(1, int(np.ceil(max(height, width) / 900.0)))
        raster_height = int(np.ceil(height / stride))
        raster_width = int(np.ceil(width / stride))
        raster = np.full(
                (raster_height, raster_width),
                np.nan,
                dtype=np.float32,
        )
        positive_max = np.full(raster.size, -np.inf, dtype=np.float32)
        negative_abs_max = np.full(raster.size, -np.inf, dtype=np.float32)
        full_range = 0.0
        for analysis in self._analyze_objects(image, props, label2section):
            seg = analysis.segmentation
            if self._lacks_zone_evidence(seg):
                continue
            _tilt, signed_turning, _magnitude, _polar = (
                signed_radial_relative_field(
                    analysis.orientation,
                    analysis.center,
                    analysis.distance_map,
                )
            )
            # Match the Dense zone's actual inner boundary. The separate PELT
            # ``core_radius`` can equal the full symmetric radius when no early
            # density changepoint is found, which would erase the entire view.
            core_exclusion_radius = float(seg.core_end_radius)
            outer_radius = self._orientation_outer_radius(seg)
            selector = self._zone_selector(
                    analysis.distance_map,
                    core_exclusion_radius,
                    outer_radius,
                    analysis.object_mask,
                    "Mask",
                    include_upper=not self.legacy_mode,
            )
            valid = (
                    selector
                    & (analysis.distance_map > _EPS)
                    & np.isfinite(signed_turning)
                    & np.isfinite(analysis.coherence)
                    & (
                        analysis.coherence
                        >= RELIABLE_PIXEL_COHERENCE
                    )
            )
            if not valid.any():
                continue
            rows, cols = np.nonzero(valid)
            origin_row = int(
                round(seg.centroid_global[0] - analysis.center[0])
            )
            origin_col = int(
                round(seg.centroid_global[1] - analysis.center[1])
            )
            global_rows = rows + origin_row
            global_cols = cols + origin_col
            inside = (
                    (global_rows >= 0)
                    & (global_rows < height)
                    & (global_cols >= 0)
                    & (global_cols < width)
            )
            if not inside.any():
                continue
            values = np.degrees(signed_turning[valid]).astype(np.float32)
            values = values[inside]
            full_range = max(full_range, float(np.max(np.abs(values))))
            flat_ids = (
                               global_rows[inside] // stride
                       ) * raster_width + global_cols[inside] // stride
            positive = values >= 0.0
            np.maximum.at(
                    positive_max,
                    flat_ids[positive],
                    values[positive],
            )
            negative = ~positive
            np.maximum.at(
                    negative_abs_max,
                    flat_ids[negative],
                    -values[negative],
            )
        has_positive = np.isfinite(positive_max)
        has_negative = np.isfinite(negative_abs_max)
        choose_positive = has_positive & (
                ~has_negative | (positive_max >= negative_abs_max)
        )
        raster_flat = raster.ravel()
        raster_flat[choose_positive] = positive_max[choose_positive]
        choose_negative = has_negative & ~choose_positive
        raster_flat[choose_negative] = -negative_abs_max[choose_negative]
        if not np.isfinite(raster).any():
            return

        if not np.isfinite(full_range) or full_range <= _EPS:
            full_range = float(np.finfo(np.float32).eps)
        fig.add_trace(
                go.Heatmap(
                        x=(np.arange(raster_width) + 0.5) * stride - 0.5,
                        y=(np.arange(raster_height) + 0.5) * stride - 0.5,
                        z=raster,
                        colorscale="Spectral",
                        zmin=-full_range,
                        zmax=full_range,
                        zmid=0.0,
                        colorbar=dict(
                                title=dict(
                                        text="Signed outward turning (deg/px)",
                                        side="top",
                                ),
                                orientation="h",
                                x=0.45,
                                xanchor="center",
                                y=-0.075,
                                yanchor="top",
                                len=0.55,
                                thickness=14,
                        ),
                        opacity=0.55,
                        name="Signed outward turning",
                        legendgroup="directional-turning",
                        legendgrouptitle_text="Directional diagnostic",
                        hovertemplate=(
                            "Signed outward turning=%{z:.3f} deg/px<extra></extra>"
                        ),
                        connectgaps=False,
                )
        )

    def _add_cumulative_rotation_trace(
            self,
            fig,
            image,
            image_shape: tuple[int, int],
    ) -> None:
        """Overlay cumulative ring-to-ring axial rotation in degrees."""
        import plotly.graph_objects as go

        props, label2section = self._prep(image)
        height, width = image_shape
        stride = max(1, int(np.ceil(max(height, width) / 900.0)))
        raster_height = int(np.ceil(height / stride))
        raster_width = int(np.ceil(width / stride))
        raster = np.full(
                (raster_height, raster_width),
                np.nan,
                dtype=np.float32,
        )
        strongest = np.full(raster.size, -np.inf, dtype=np.float32)
        signed_value = np.full(raster.size, np.nan, dtype=np.float32)
        full_range = 0.0
        for analysis in self._analyze_objects(image, props, label2section):
            seg = analysis.segmentation
            if self._lacks_zone_evidence(seg):
                continue
            signed_tilt, _turning, _magnitude, polar_angle = (
                signed_radial_relative_field(
                    analysis.orientation,
                    analysis.center,
                    analysis.distance_map,
                )
            )
            inner_radius = float(seg.core_end_radius)
            outer_radius = self._orientation_outer_radius(seg)
            structure_selector = self._zone_selector(
                    analysis.distance_map,
                    inner_radius,
                    outer_radius,
                    analysis.object_mask,
                    "Mask",
                    include_upper=not self.legacy_mode,
            )
            _radii, sector_tilt, _resultant = radial_ring_orientation_profile(
                    signed_tilt,
                    polar_angle,
                    analysis.coherence,
                    analysis.distance_map,
                    structure_selector,
                    inner_radius,
                    outer_radius,
                    self.radial_ring_width,
                    N_SECTORS,
                    include_outer=not self.legacy_mode,
            )
            cumulative = cumulative_ring_rotation_profile(sector_tilt)
            reliable_structure = (
                    structure_selector
                    & np.isfinite(analysis.coherence)
                    & (
                        analysis.coherence
                        >= RELIABLE_PIXEL_COHERENCE
                    )
            )
            local_field = radial_ring_sector_field(
                    cumulative,
                    polar_angle,
                    analysis.distance_map,
                    reliable_structure,
                    inner_radius,
                    self.radial_ring_width,
            )
            valid = np.isfinite(local_field)
            if not valid.any():
                continue
            rows, cols = np.nonzero(valid)
            origin_row = int(
                round(seg.centroid_global[0] - analysis.center[0])
            )
            origin_col = int(
                round(seg.centroid_global[1] - analysis.center[1])
            )
            global_rows = rows + origin_row
            global_cols = cols + origin_col
            inside = (
                    (global_rows >= 0)
                    & (global_rows < height)
                    & (global_cols >= 0)
                    & (global_cols < width)
            )
            if not inside.any():
                continue
            values = np.degrees(local_field[valid]).astype(np.float32)[inside]
            full_range = max(full_range, float(np.max(np.abs(values))))
            flat_ids = (
                               global_rows[inside] // stride
                       ) * raster_width + global_cols[inside] // stride
            magnitudes = np.abs(values)
            order = np.lexsort((magnitudes, flat_ids))
            ordered_ids = flat_ids[order]
            last_for_id = np.r_[
                ordered_ids[1:] != ordered_ids[:-1],
                True,
            ]
            chosen_ids = ordered_ids[last_for_id]
            chosen_values = values[order][last_for_id]
            chosen_magnitudes = magnitudes[order][last_for_id]
            stronger = chosen_magnitudes >= strongest[chosen_ids]
            strongest[chosen_ids[stronger]] = chosen_magnitudes[stronger]
            signed_value[chosen_ids[stronger]] = chosen_values[stronger]

        available = np.isfinite(strongest)
        raster.ravel()[available] = signed_value[available]
        if not np.isfinite(raster).any():
            return
        if not np.isfinite(full_range) or full_range <= _EPS:
            full_range = float(np.finfo(np.float32).eps)
        fig.add_trace(
                go.Heatmap(
                        x=(np.arange(raster_width) + 0.5) * stride - 0.5,
                        y=(np.arange(raster_height) + 0.5) * stride - 0.5,
                        z=raster,
                        colorscale="Spectral",
                        zmin=-full_range,
                        zmax=full_range,
                        zmid=0.0,
                        colorbar=dict(
                                title=dict(
                                        text="Cumulative signed radial rotation (deg)",
                                        side="top",
                                ),
                                orientation="h",
                                x=0.45,
                                xanchor="center",
                                y=-0.075,
                                yanchor="top",
                                len=0.55,
                                thickness=14,
                        ),
                        opacity=0.58,
                        name="Cumulative radial rotation",
                        legendgroup="cumulative-radial-rotation",
                        legendgrouptitle_text="Cumulative directional diagnostic",
                        hovertemplate=(
                            "Cumulative signed rotation=%{z:.2f}°<extra></extra>"
                        ),
                        connectgaps=False,
                )
        )

    def _add_matched_cumulative_rotation_trace(
            self,
            fig,
            image,
            image_shape: tuple[int, int],
            *,
            max_sector_shift: int,
            allow_gap_bridging: bool = False,
            allow_restarts: bool = False,
    ) -> None:
        """Overlay matched-ring cumulative fiber-axis rotation in degrees."""
        import plotly.graph_objects as go

        props, label2section = self._prep(image)
        height, width = image_shape
        stride = max(1, int(np.ceil(max(height, width) / 900.0)))
        raster_height = int(np.ceil(height / stride))
        raster_width = int(np.ceil(width / stride))
        raster = np.full(
                (raster_height, raster_width),
                np.nan,
                dtype=np.float32,
        )
        strongest = np.full(raster.size, -np.inf, dtype=np.float32)
        signed_value = np.full(raster.size, np.nan, dtype=np.float32)
        path_x: list[float | None] = []
        path_y: list[float | None] = []
        bridge_x: list[float | None] = []
        bridge_y: list[float | None] = []
        for analysis in self._analyze_objects(image, props, label2section):
            seg = analysis.segmentation
            if self._lacks_zone_evidence(seg):
                continue
            _signed_tilt, _turning, _magnitude, polar_angle = (
                signed_radial_relative_field(
                    analysis.orientation,
                    analysis.center,
                    analysis.distance_map,
                )
            )
            fiber_axis = analysis.orientation + FIBER_AXIS_OFFSET
            inner_radius = float(seg.core_end_radius)
            outer_radius = self._orientation_outer_radius(seg)
            structure_selector = self._zone_selector(
                    analysis.distance_map,
                    inner_radius,
                    outer_radius,
                    analysis.object_mask,
                    "Mask",
                    include_upper=not self.legacy_mode,
            )
            radii, sector_orientation, sector_resultant = (
                radial_ring_orientation_profile(
                        fiber_axis,
                        polar_angle,
                        analysis.coherence,
                        analysis.distance_map,
                        structure_selector,
                        inner_radius,
                        outer_radius,
                        self.radial_ring_width,
                        N_SECTORS,
                        include_outer=not self.legacy_mode,
                )
            )
            cumulative, path_sectors = (
                matched_ring_cumulative_rotation_profile(
                        radii,
                        sector_orientation,
                        sector_resultant,
                        max_sector_shift=max_sector_shift,
                        allow_gap_bridging=allow_gap_bridging,
                        allow_restarts=allow_restarts,
                )
            )
            ring_sector_values = matched_tracks_to_ring_sector_values(
                    cumulative,
                    path_sectors,
            )
            reliable_structure = (
                    structure_selector
                    & np.isfinite(analysis.coherence)
                    & (
                        analysis.coherence
                        >= RELIABLE_PIXEL_COHERENCE
                    )
            )
            local_field = radial_ring_sector_field(
                    ring_sector_values,
                    polar_angle,
                    analysis.distance_map,
                    reliable_structure,
                    inner_radius,
                    self.radial_ring_width,
            )
            valid = np.isfinite(local_field)
            origin_row = int(
                round(seg.centroid_global[0] - analysis.center[0])
            )
            origin_col = int(
                round(seg.centroid_global[1] - analysis.center[1])
            )
            if valid.any():
                rows, cols = np.nonzero(valid)
                global_rows = rows + origin_row
                global_cols = cols + origin_col
                inside = (
                        (global_rows >= 0)
                        & (global_rows < height)
                        & (global_cols >= 0)
                        & (global_cols < width)
                )
                if inside.any():
                    values = np.degrees(local_field[valid]).astype(np.float32)
                    values = values[inside]
                    flat_ids = (
                                       global_rows[inside] // stride
                               ) * raster_width + global_cols[inside] // stride
                    magnitudes = np.abs(values)
                    order = np.lexsort((magnitudes, flat_ids))
                    ordered_ids = flat_ids[order]
                    last_for_id = np.r_[
                        ordered_ids[1:] != ordered_ids[:-1],
                        True,
                    ]
                    chosen_ids = ordered_ids[last_for_id]
                    chosen_values = values[order][last_for_id]
                    chosen_magnitudes = magnitudes[order][last_for_id]
                    stronger = chosen_magnitudes >= strongest[chosen_ids]
                    strongest[chosen_ids[stronger]] = chosen_magnitudes[
                        stronger
                    ]
                    signed_value[chosen_ids[stronger]] = chosen_values[
                        stronger
                    ]

            n_sectors = sector_orientation.shape[1]
            sector_angles = (np.arange(n_sectors, dtype=np.float64) + 0.5) * (
                    2.0 * np.pi / float(n_sectors)
            )
            for seed_sector in range(n_sectors):
                supported = np.flatnonzero(
                        (path_sectors[:, seed_sector] >= 0)
                        & np.isfinite(cumulative[:, seed_sector])
                )
                if supported.size < 2:
                    continue
                matched_sectors = path_sectors[supported, seed_sector]
                angles = sector_angles[matched_sectors]
                local_rows = (
                    analysis.center[0] + radii[supported] * np.sin(angles)
                )
                local_cols = (
                    analysis.center[1] + radii[supported] * np.cos(angles)
                )
                global_rows = local_rows + origin_row
                global_cols = local_cols + origin_col
                inside = (
                        (global_rows >= 0.0)
                        & (global_rows < height)
                        & (global_cols >= 0.0)
                        & (global_cols < width)
                )
                if np.count_nonzero(inside) < 2:
                    continue
                inside_rings = supported[inside]
                inside_cols = global_cols[inside]
                inside_rows = global_rows[inside]
                for point_index in range(1, inside_rings.size):
                    is_bridge = (
                            inside_rings[point_index]
                            - inside_rings[point_index - 1]
                            > 1
                    )
                    target_x = bridge_x if is_bridge else path_x
                    target_y = bridge_y if is_bridge else path_y
                    target_x.extend(
                            [
                                float(inside_cols[point_index - 1]),
                                float(inside_cols[point_index]),
                                None,
                            ]
                    )
                    target_y.extend(
                            [
                                float(inside_rows[point_index - 1]),
                                float(inside_rows[point_index]),
                                None,
                            ]
                    )

        available = np.isfinite(strongest)
        raster.ravel()[available] = signed_value[available]
        if np.isfinite(raster).any():
            fig.add_trace(
                    go.Heatmap(
                            x=(np.arange(raster_width) + 0.5) * stride - 0.5,
                            y=(np.arange(raster_height) + 0.5) * stride - 0.5,
                            z=raster,
                            colorscale="Spectral",
                            zmin=-180.0,
                            zmax=180.0,
                            zmid=0.0,
                            colorbar=dict(
                                    title=dict(
                                            text="Matched cumulative fiber rotation (deg)",
                                            side="top",
                                    ),
                                    orientation="h",
                                    x=0.45,
                                    xanchor="center",
                                    y=-0.075,
                                    yanchor="top",
                                    len=0.55,
                                    thickness=14,
                            ),
                            opacity=0.68,
                            name="Matched cumulative fiber rotation",
                            legendgroup="matched-cumulative-rotation",
                            legendgrouptitle_text="Matched-ring directional diagnostic",
                            hovertemplate=(
                                "Matched cumulative rotation=%{z:.2f}°<extra></extra>"
                            ),
                            connectgaps=False,
                    )
            )
        if path_x and not allow_restarts:
            fig.add_trace(
                    go.Scattergl(
                            x=path_x,
                            y=path_y,
                            mode="lines",
                            line=dict(color="rgba(255,255,255,0.58)", width=1.0),
                            name="Matched outward ring paths",
                            legendgroup="matched-cumulative-rotation",
                            hoverinfo="skip",
                    )
            )
        if bridge_x and not allow_restarts:
            fig.add_trace(
                    go.Scattergl(
                            x=bridge_x,
                            y=bridge_y,
                            mode="lines",
                            line=dict(
                                    color="rgba(255,255,255,0.48)",
                                    width=1.0,
                                    dash="dash",
                            ),
                            name="Bridged unsupported rings",
                            legendgroup="matched-cumulative-rotation",
                            hoverinfo="skip",
                    )
            )

    @staticmethod
    def _add_mask_selector_trace(fig, image) -> None:
        """Add the detected-mask boundary used by the ``Mask`` variant.

        The contour is legend-only during interactive use. Static inspect
        export shows it so the mask-intersected selector can be compared with
        the concentric ``Radial`` selector without displaying object numbers.
        """
        import plotly.graph_objects as go

        mask = (image.objmap[:] > 0).astype(np.uint8)
        if not mask.any():
            return
        fig.add_trace(
                go.Contour(
                        z=mask,
                        autocontour=False,
                        contours=dict(
                                start=0.5,
                                end=0.5,
                                size=1.0,
                                coloring="lines",
                                showlabels=False,
                        ),
                        line=dict(color=OKABE_ITO_GREEN, width=1.5),
                        showscale=False,
                        showlegend=True,
                        name="Detected-mask selector",
                        legendgroup="selectors",
                        legendgrouptitle_text="Selectors",
                        visible="legendonly",
                        hoverinfo="skip",
                )
        )

    @staticmethod
    def _tile_origin(record) -> tuple[float, float]:
        """Plate-frame (row, col) origin of a cached object's tile.

        The tile pixel ``(r_tile, c_tile)`` sits at plate coordinates
        ``(r_tile + origin_row, c_tile + origin_col)``; the inoculum centre lands
        on ``centroid_global`` by construction (``origin = centroid_global -
        centre``).
        """
        cg = record["centroid_global"]
        ctr = record["centre"]
        return (cg[0] - ctr[0], cg[1] - ctr[1])

    def _add_quiver_trace(self, fig) -> None:
        """Draw confidence-coded local fiber axes inside the overall selector.

        Reads only the pre-downsampled block quiver ``(rows, cols, phi_block,
        coh_block)`` from each cached record. Canonical blocks are clipped to
        ``[CoreEndRadius, SparseEndRadius]``, including the exact global outer
        boundary. Legacy blocks retain their historical symmetric-radius domain.
        The stored gradient-normal axis is rotated 90° to show the local fiber
        axis. Segment half-length scales with coherence; three traces provide
        low/medium/high confidence opacity levels.
        """
        import plotly.graph_objects as go

        binned_xy: list[tuple[list[float | None], list[float | None]]] = [
            ([], []) for _ in _QUIVER_COHERENCE_BINS
        ]
        half = 0.5 * max(1, int(self.quiver_block))
        for record in self._cache.values():
            rows, cols, phi_block, coh_block = record["quiver"]
            origin_r, origin_c = self._tile_origin(record)
            centre_r, centre_c = record["centre"]
            inner_radius = (
                0.0
                if self.legacy_mode
                else record["radii"].get("core_end", np.nan)
            )
            outer_radius = record["radii"].get(
                "symmetric" if self.legacy_mode else "sparse_end", np.nan
            )
            if (
                not np.isfinite(inner_radius)
                or not np.isfinite(outer_radius)
                or outer_radius <= inner_radius
            ):
                continue
            for i in range(phi_block.shape[0]):
                for j in range(phi_block.shape[1]):
                    phi = phi_block[i, j]
                    coh = coh_block[i, j]
                    if not np.isfinite(phi) or not np.isfinite(coh):
                        continue
                    radius = np.hypot(
                        rows[i, j] - centre_r,
                        cols[i, j] - centre_c,
                    )
                    selector_outer = (
                        np.nextafter(outer_radius, np.inf)
                        if not self.legacy_mode
                        else outer_radius
                    )
                    if radius < inner_radius or radius >= selector_outer:
                        continue
                    bin_idx = next(
                            (
                                idx
                                for idx, (_, lower, upper, _) in enumerate(
                                    _QUIVER_COHERENCE_BINS
                            )
                                if lower <= coh < upper
                            ),
                            None,
                    )
                    if bin_idx is None:
                        continue
                    # Block centre in plate coords (x=col, y=row).
                    cx = cols[i, j] + origin_c
                    cy = rows[i, j] + origin_r
                    length = half * float(coh)
                    fiber_phi = phi + FIBER_AXIS_OFFSET
                    dx = length * np.cos(fiber_phi)
                    dy = length * np.sin(fiber_phi)
                    xs, ys = binned_xy[bin_idx]
                    xs.extend([cx - dx, cx + dx, None])
                    ys.extend([cy - dy, cy + dy, None])
        for (name, _lower, _upper, opacity), (xs, ys) in zip(
                _QUIVER_COHERENCE_BINS,
                binned_xy,
        ):
            if not xs:
                continue
            fig.add_trace(
                    go.Scattergl(
                            x=xs,
                            y=ys,
                            mode="lines",
                            line=dict(color=OKABE_ITO_SKY, width=1.6),
                            opacity=opacity,
                            name=f"Local fiber axis · {name}",
                            legendgroup="fiber-axes",
                            legendgrouptitle_text="Local fiber axes",
                            hoverinfo="skip",
                    )
            )

    def _add_zone_ring_traces(self, fig) -> None:
        """Concentric zone-boundary circles centred at each object's inoculum.

        Draws the symmetric, core-end, dense-end and sparse-end radii (skipping
        non-finite radii) as legend-toggleable circle polygons read from the
        cached ``radii`` + ``centroid_global`` scalars.
        """
        import plotly.graph_objects as go

        ring_styles = (
            ("symmetric", "Independent symmetry radius", "#785EF0", "solid"),
            ("core_end", "CoreZone / resolved boundary", "#DC267F", "dot"),
            ("dense_end", "Dense / sparse boundary", OKABE_ITO_NAVY, "dash"),
            (
                "sparse_end",
                f"Sparse outer boundary (P{self.outer_zone_percentile:g})",
                OKABE_ITO_SKY,
                "dash",
            ),
        )
        for key, name, color, dash in ring_styles:
            xs: list[float | None] = []
            ys: list[float | None] = []
            for record in self._cache.values():
                r = record["radii"].get(key, np.nan)
                if r is None or not np.isfinite(r) or r <= 0:
                    continue
                cy, cx = record["centroid_global"]
                cxs, cys = circle_xy(cx, cy, float(r))
                xs.extend([*cxs.tolist(), None])
                ys.extend([*cys.tolist(), None])
            if not xs:
                continue
            fig.add_trace(
                    go.Scatter(
                            x=xs,
                            y=ys,
                            mode="lines",
                            line=dict(color=color, width=1.5, dash=dash),
                            name=name,
                            legendgroup="rings",
                            legendgrouptitle_text="Radial selectors",
                            hoverinfo="skip",
                    )
            )

    def _add_mean_axis_traces(self, fig) -> None:
        """Add one centered, undirected mean fiber axis per Radial zone.

        Orientation is axial modulo 180°, so centered line segments are used
        instead of arrows. Each zone has its own color and trace. Segment length
        is proportional to concentration ``R``; Mask-variant values remain in
        the hover summary rather than being overplotted at the same centre.
        """
        import plotly.graph_objects as go

        zone_styles = (
            ("Overall", OKABE_ITO_ORANGE),
            ("Dense", "#CC79A7"),
            ("Sparse", OKABE_ITO_VERMILION),
        )
        for zone, color in zone_styles:
            axis_x: list[float | None] = []
            axis_y: list[float | None] = []
            for record in self._cache.values():
                cy, cx = record["centroid_global"]
                scale_radius = record["radii"].get(
                    "symmetric" if self.legacy_mode else "sparse_end", np.nan
                )
                scale = (
                    float(scale_radius)
                    if scale_radius is not None
                    and np.isfinite(scale_radius)
                    and scale_radius > 0
                    else 20.0
                )
                R, _turning, _coh, direction = record["per_zone"][
                    ("Radial", zone)
                ]
                if not np.isfinite(R) or not np.isfinite(direction):
                    continue
                half_length = 0.5 * scale * float(R)
                fiber_direction = direction + FIBER_AXIS_OFFSET
                dx = half_length * np.cos(fiber_direction)
                dy = half_length * np.sin(fiber_direction)
                axis_x.extend([cx - dx, cx + dx, None])
                axis_y.extend([cy - dy, cy + dy, None])
            if not axis_x:
                continue
            fig.add_trace(
                    go.Scatter(
                            x=axis_x,
                            y=axis_y,
                            mode="lines",
                            line=dict(color=color, width=3.0),
                            name=f"Mean fiber axis · {zone}",
                            legendgroup="mean-axes",
                            legendgrouptitle_text="Radial mean axes (length = R)",
                            visible="legendonly",
                            hoverinfo="skip",
                    )
            )

    def _add_metric_hover_trace(self, fig) -> None:
        """Add invisible centroid hit targets containing all zone metrics."""
        import plotly.graph_objects as go

        def _value(value: float, digits: int) -> str:
            return f"{value:.{digits}f}" if np.isfinite(value) else "NaN"

        xs: list[float] = []
        ys: list[float] = []
        hover_text: list[str] = []
        for record in self._cache.values():
            cy, cx = record["centroid_global"]
            lines = [
                "<b>Primary outward-rotation metrics</b>",
                "Peak and net are degrees; rate is deg/px; consistency is "
                "dimensionless",
            ]
            for zone in _ZONES:
                values = record["outward_rotation"][zone]
                lines.append(
                        f"{zone}: Sustained peak="
                        f"{_value(values['OutwardRotationSustainedPeak'], 2)}°, "
                        f"Net={_value(values['OutwardRotationNet'], 2)}°, "
                        f"Rate={_value(values['OutwardRotationRate'], 4)} deg/px, "
                        f"Consistency="
                        f"{_value(values['OutwardRotationConsistency'], 3)}"
                )
            if self.include_diagnostics:
                lines.extend(
                        [
                            "<b>Diagnostic orientation metrics</b>",
                            "R = parallel concentration; T = turning (deg/px); "
                            "C = coherence",
                            "RTilt = absolute tilt from radial (deg); "
                            "OutT = outward radial turning (deg/px); "
                            "Support = reliable sector fraction (QC)",
                        ]
                )
                for variant in _VARIANTS:
                    lines.append(f"<b>{variant} selector</b>")
                    for zone in _ZONES:
                        R, turning, coherence, _direction = record["per_zone"][
                            (variant, zone)
                        ]
                        lines.append(
                                f"{zone}: R={_value(R, 3)}, "
                                f"T={_value(turning, 4)}, C={_value(coherence, 3)}"
                        )
                lines.append("<b>Detected structure · radial-relative</b>")
                for zone in _ZONES:
                    radial_tilt, radial_turning, radial_support = record[
                        "radial_relative"
                    ][zone]
                    lines.append(
                            f"{zone}: RTilt={_value(radial_tilt, 3)}, "
                            f"OutT={_value(radial_turning, 4)}, "
                            f"Support={_value(radial_support, 3)}"
                    )
                lines.append(
                        f"<b>Long range · {self.long_range_lag:g} px ring lag</b>"
                )
                for region in (*_ZONES, "DenseToSparse"):
                    magnitude, signed, support = record["long_range"][region]
                    lines.append(
                            f"{region}: |Δ|={_value(magnitude, 2)}°, "
                            f"signed Δ={_value(signed, 2)}°, "
                            f"Support={_value(support, 3)}"
                    )
            xs.append(float(cx))
            ys.append(float(cy))
            hover_text.append("<br>".join(lines))
        if not xs:
            return
        fig.add_trace(
                go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers",
                        marker=dict(size=16, color="rgba(0, 0, 0, 0.01)"),
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                        name="Zone metrics (hover centres)",
                        showlegend=False,
                )
        )

    def report(self, subject=None, *, show: bool = True, **overrides):
        """Composed notebook diagnostic (returns a single ``go.Figure``).

        Stacks three vertically-arranged panels: the :meth:`inspect` overview,
        a recomputed coherence heatmap, and a per-zone primary outward-rotation
        summary table. Calls :meth:`measure` first when the compact cache is
        empty or was built for a different image.

        Args:
            subject: Detected Image to render. If *None*, the image cached by the
                most recent :meth:`measure` call is reused.
            show: When *True*, call ``fig.show()`` before returning (best-effort;
                swallowed outside a display context). Defaults to *True*.
            **overrides: Report-specific overrides are unsupported.

        Returns:
            A single composed ``plotly.graph_objects.Figure`` stacking the three
            panels vertically.

        Examples:
            >>> from phenotypic.data import load_synth_filamentous_plate
            >>> from phenotypic.measure import MeasureOrientationZones
            >>> op = MeasureOrientationZones()
            >>> fig = op.report(load_synth_filamentous_plate(), show=False)
            >>> any(getattr(tr, "type", None) == "table" for tr in fig.data)
            True
        """
        if overrides:
            raise ValueError(
                    f"report(): unsupported override(s) {sorted(overrides)}"
            )
        image = subject if subject is not None else self._require_cache_image()
        self._ensure_diagnostic_cache(image)
        from ._report import _OrientationZonesReport

        report = _OrientationZonesReport()
        producer = cast("MeasureOrientationZones", self)
        fig = report.report(image, producer=producer)
        if show:
            try:
                fig.show()
            except Exception:  # pragma: no cover - display-context dependent
                pass
        return fig
