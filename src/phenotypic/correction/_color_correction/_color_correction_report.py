"""Notebook-only Plotly report for color-correction diagnostics.

Ports the legacy Panel ``ColorCorrectionDashboard`` to the renderer-neutral
:class:`~phenotypic.abc_.FigureProvider` protocol. The report is a *helper*
provider: it holds the data it needs from a fitted
:class:`~phenotypic.correction.ColorCheckerProfile` (and the optional source
image / ROIs), so every ``@figure`` method reads ``self`` and takes no subject
parameter.

The four legacy ``show_*`` toggles become ``section`` tags (collapsible cards),
per migration design decision D12 — they are *not* :class:`Control`\\ s. With no
controls declared, :meth:`~phenotypic.abc_.FigureProvider.dash` composes the
figures into a single stacked ``go.Figure`` rather than an ipywidgets shell.
The interactive ROI-selector control specified in design.md §9 is deferred
(see ``DEFERRED.md`` → "Scope reductions recorded post-review"); this report
ships control-free for now.

Sections
--------
* ``delta_e`` — Delta-E 2000 before/after per patch (``go.Bar``).
* ``patches`` — matched reference/measured/corrected swatch strip
  (``go.Image``).
* ``pipeline`` — per-ROI preprocessing stages (``go.Image``); only when a
  source image + ROIs are available.
* ``segmentation`` — preprocessed image + chip-mask overlay (``go.Image``);
  only when a source image + ROIs are available.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import colour
import numpy as np
import plotly.graph_objects as go

from phenotypic.abc_ import FigureProvider, figure
from phenotypic.viz.figures._theme import GOLD, NAVY, OKABE_ITO

if TYPE_CHECKING:
    from phenotypic._core._image import Image

    from ._color_checker_profile import ColorCheckerProfile

__all__ = ["ColorCorrectionReport"]

# ---------------------------------------------------------------------------
# Section tags (the legacy ``show_*`` toggles → collapsible cards, D12).
# ---------------------------------------------------------------------------
_SECTION_DELTA_E = "delta_e"
_SECTION_PATCHES = "patches"
_SECTION_PIPELINE = "pipeline"
_SECTION_SEGMENTATION = "segmentation"

# ---------------------------------------------------------------------------
# Quality-threshold colors for Delta-E coding. These are explicit per-datum
# colors (not theme styling), carried over from the Okabe-Ito theme palette.
# ---------------------------------------------------------------------------
_GREEN = OKABE_ITO[3]  # ≤ 1.0 — just-noticeable
_ORANGE = OKABE_ITO[1]  # ≤ 5.0 — significant
_VERMILION = OKABE_ITO[6]  # > 5.0 / rejected
_SKY = OKABE_ITO[2]  # "after"-correction bars / condition number
_GREY = "#BBBBBB"  # perceptibility reference lines

#: Delta-E 2000 perceptibility reference levels (value, label).
_REFERENCE_LINES: tuple[tuple[float, str], ...] = (
    (1.0, "Just noticeable"),
    (2.3, "Perceptible"),
    (5.0, "Significant"),
)
_DOMAIN_TRACE_TYPES = {
    "funnelarea",
    "icicle",
    "indicator",
    "parcats",
    "parcoords",
    "pie",
    "sankey",
    "sunburst",
    "table",
    "treemap",
}


def _lab_to_srgb(
    lab: np.ndarray,
    illuminant: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a CIE Lab array to clipped sRGB in ``[0, 1]``.

    Args:
        lab: Array of shape ``(..., 3)`` in CIE Lab.
        illuminant: CIE xy chromaticity of the whitepoint the Lab values are
            relative to. Defaults to the ``colour`` library default (D65).

    Returns:
        sRGB array clipped to ``[0, 1]``.
    """
    kwargs: dict[str, Any] = {}
    if illuminant is not None:
        kwargs["illuminant"] = illuminant
    XYZ = colour.Lab_to_XYZ(lab, **kwargs)
    rgb = colour.XYZ_to_sRGB(XYZ)
    return np.clip(rgb, 0.0, 1.0)


def _delta_e_color(value: float) -> str:
    """Map a Delta-E 2000 value to a perceptual quality color.

    Args:
        value: A Delta-E 2000 magnitude.

    Returns:
        Green (``≤ 1``), navy (``≤ 2.3``), orange (``≤ 5``), else vermilion.
    """
    if value <= 1.0:
        return _GREEN
    if value <= 2.3:
        return NAVY
    if value <= 5.0:
        return _ORANGE
    return _VERMILION


def _normalize_display(arr: np.ndarray) -> np.ndarray:
    """Return a ``[0, 1]`` float RGB view suitable for ``go.Image``.

    Args:
        arr: An RGB array (integer or float, any positive scale).

    Returns:
        A float64 RGB array clipped to ``[0, 1]``.
    """
    display = arr.astype(np.float64)
    peak = float(display.max()) if display.size else 0.0
    if peak > 1.0:
        display = display / peak
    return np.clip(display, 0.0, 1.0)


def _axis_layout_key(axis_ref: str, axis: str) -> str:
    """Return the layout key for a trace axis reference."""
    return f"{axis}axis{axis_ref[len(axis):]}"


def _axis_ref_from_layout_key(layout_key: str, axis: str) -> str:
    """Return the trace reference for a layout axis key."""
    return f"{axis}{layout_key[len(f'{axis}axis'):]}"


def _axis_ref_order(axis_ref: str, axis: str) -> int:
    """Sort Plotly refs as x, x2, x3... or y, y2, y3."""
    suffix = axis_ref[len(axis):]
    return 1 if suffix == "" else int(suffix)


def _next_axis_ref(counts: dict[str, int], axis: str) -> str:
    """Allocate the next axis reference for a composed figure."""
    counts[axis] += 1
    return axis if counts[axis] == 1 else f"{axis}{counts[axis]}"


def _remap_domain_value(value: float, y0: float, y1: float) -> float:
    """Scale a child-domain coordinate into a parent domain, clamped for Plotly."""
    remapped = y0 + value * (y1 - y0)
    return min(1.0, max(0.0, remapped))


class ColorCorrectionReport(FigureProvider):
    """Plotly diagnostic report for a fitted color-correction profile.

    A notebook-only :class:`~phenotypic.abc_.FigureProvider` replacing the
    legacy Panel ``ColorCorrectionDashboard``. It holds the diagnostics (and
    optional source image / ROIs) extracted from a fitted
    :class:`~phenotypic.correction.ColorCheckerProfile` and exposes one
    ``@figure`` method per legacy panel.

    The figures declare no :class:`~phenotypic.abc_.Control`, so
    :meth:`~phenotypic.abc_.FigureProvider.dash` returns a composed
    ``go.Figure`` (not an ipywidgets shell). The legacy ``show_*`` visibility
    toggles map to per-figure ``section`` tags (collapsible cards) per design
    decision D12.

    Args:
        profile: A fitted :class:`ColorCheckerProfile`.
        image: Optional source image used for the pipeline-step and
            segmentation figures. When ``None`` those figures are skipped.
        rois: Optional list of ``(row_slice, col_slice)`` ROI bounds matching
            *image*. When ``None`` the pipeline/segmentation figures are
            skipped.

    Attributes:
        profile: The fitted profile whose diagnostics drive every figure.

    Example:
        >>> from phenotypic.correction import ColorCheckerProfile
        >>> from phenotypic.correction._color_correction._color_correction_report import (
        ...     ColorCorrectionReport,
        ... )
        >>> import colour, numpy as np
        >>> checker = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
        >>> rng = np.random.default_rng(0)
        >>> names, srgb = [], []
        >>> for name, xyY in checker.data.items():
        ...     names.append(name)
        ...     srgb.append(np.clip(colour.XYZ_to_sRGB(colour.xyY_to_XYZ(xyY)), 0, 1))
        >>> measured = np.clip(np.array(srgb) + rng.normal(0, 0.01, (24, 3)), 0, 1)
        >>> profile = ColorCheckerProfile(degree=2)
        >>> _ = profile._fit_from_patch_colors(measured, patch_names=names)
        >>> report = ColorCorrectionReport(profile)
        >>> fig = report.fig_delta_e()
        >>> fig.data[0].type
        'bar'
    """

    def __init__(
        self,
        profile: ColorCheckerProfile,
        image: Image | None = None,
        rois: list[tuple[slice, slice]] | None = None,
    ) -> None:
        self.profile = profile
        self._image = image
        self._rois = rois
        # bool(rois) guards the empty-list case: rois=[] would otherwise enable
        # the image sections and crash make_subplots(rows=0, ...).
        self._has_image = image is not None and bool(rois)

    # -- subject resolution -------------------------------------------------

    def _figure_subject(self) -> Any:
        """Return the held subject for the ``FigureProvider`` mixin.

        This report is a helper provider: every ``@figure`` method reads
        ``self`` directly, so the returned subject is informational only (the
        fitted :class:`ColorCheckerProfile`).
        """
        return self.profile

    # -- introspection ------------------------------------------------------

    def iter_figures(self) -> list[Any]:
        """Return the figure specs, dropping image-dependent ones when no image.

        The pipeline-step and segmentation figures require a source image and
        ROIs (the legacy "hide when fitted from patch colors" behaviour); when
        either is missing those specs are omitted so :meth:`dash` and
        :meth:`inspect` only consider renderable figures.

        Returns:
            The list of :class:`~phenotypic.abc_.FigureSpec` to render.
        """
        specs = super().iter_figures()
        if self._has_image:
            return specs
        image_only = {_SECTION_PIPELINE, _SECTION_SEGMENTATION}
        return [spec for spec in specs if spec.section not in image_only]

    # -- composed dashboard -------------------------------------------------

    def dash(self, subject: Any = None) -> go.Figure:
        """Compose report figures while preserving child subplot layouts.

        The base ``FigureProvider`` composer is intentionally trace-only. This
        report renders nested subplot grids for the pipeline and segmentation
        sections, so it needs to remap each child figure into its own vertical
        domain instead of reassigning every trace to one subplot row.

        Args:
            subject: Unused; accepted for ``FigureProvider`` signature parity.

        Returns:
            A single themed ``plotly.graph_objects.Figure``.
        """
        from phenotypic.viz.figures._theme import apply_theme

        specs = self.iter_figures()
        if len(specs) == 1:
            return self._render_spec(specs[0], subject)

        rendered = [self._render_spec(spec, subject) for spec in specs]
        weights = [self._figure_vertical_weight(fig) for fig in rendered]
        domains = self._vertical_domains(weights)

        composed = go.Figure()
        axis_counts = {"x": 0, "y": 0}
        for spec, subfig, (y0, y1) in zip(specs, rendered, domains):
            self._append_figure_to_domain(
                composed,
                subfig,
                y0=y0,
                y1=y1,
                axis_counts=axis_counts,
            )
            composed.add_annotation(
                xref="paper",
                yref="paper",
                x=0.0,
                y=min(1.0, y1 + 0.012),
                text=f"<b>{spec.title}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(color=NAVY, size=14),
            )

        composed.update_layout(
            title_text="Color Correction Diagnostics",
            height=max(650, int(280 * sum(weights))),
            barmode="group",
            showlegend=True,
        )
        return apply_theme(composed)

    @staticmethod
    def _vertical_domains(
        weights: list[float], *, gap: float = 0.055
    ) -> list[tuple[float, float]]:
        """Return top-to-bottom paper domains for stacked child figures."""
        if not weights:
            return []
        usable_height = 1.0 - gap * (len(weights) - 1)
        total_weight = float(sum(weights))
        domains: list[tuple[float, float]] = []
        top = 1.0
        for weight in weights:
            height = usable_height * weight / total_weight
            bottom = top - height
            domains.append((bottom, top))
            top = bottom - gap
        return domains

    @staticmethod
    def _figure_vertical_weight(fig: go.Figure) -> float:
        """Estimate vertical space from the number of child y-axis domains."""
        layout = fig.layout.to_plotly_json()
        domains = {
            tuple(axis.get("domain", [0.0, 1.0]))
            for key, axis in layout.items()
            if key.startswith("yaxis") and isinstance(axis, dict)
        }
        return max(1.0, float(len(domains)))

    @classmethod
    def _append_figure_to_domain(
        cls,
        composed: go.Figure,
        source: go.Figure,
        *,
        y0: float,
        y1: float,
        axis_counts: dict[str, int],
    ) -> None:
        """Append ``source`` into a vertical paper-domain slice."""
        layout = source.layout.to_plotly_json()
        x_mapping = cls._axis_ref_mapping(source, layout, "x", axis_counts)
        y_mapping = cls._axis_ref_mapping(source, layout, "y", axis_counts)

        for axis_ref, target_ref in x_mapping.items():
            axis_payload = copy.deepcopy(
                layout.get(_axis_layout_key(axis_ref, "x"), {})
            )
            axis_payload.setdefault("domain", [0.0, 1.0])
            cls._remap_axis_payload(axis_payload, x_mapping, y_mapping)
            composed.update_layout(
                {_axis_layout_key(target_ref, "x"): axis_payload}
            )

        for axis_ref, target_ref in y_mapping.items():
            axis_payload = copy.deepcopy(
                layout.get(_axis_layout_key(axis_ref, "y"), {})
            )
            domain = axis_payload.get("domain", [0.0, 1.0])
            axis_payload["domain"] = [
                _remap_domain_value(float(domain[0]), y0, y1),
                _remap_domain_value(float(domain[1]), y0, y1),
            ]
            cls._remap_axis_payload(axis_payload, x_mapping, y_mapping)
            composed.update_layout(
                {_axis_layout_key(target_ref, "y"): axis_payload}
            )

        for trace in source.data:
            payload = trace.to_plotly_json()
            if payload.get("type") in _DOMAIN_TRACE_TYPES or "domain" in payload:
                cls._remap_trace_domain(payload, y0=y0, y1=y1)
            else:
                payload["xaxis"] = x_mapping.get(payload.get("xaxis", "x"), "x")
                payload["yaxis"] = y_mapping.get(payload.get("yaxis", "y"), "y")
            composed.add_trace(payload)

        for shape in source.layout.shapes:
            payload = copy.deepcopy(shape.to_plotly_json())
            cls._remap_layout_ref_payload(payload, "xref", "x", x_mapping)
            cls._remap_layout_ref_payload(
                payload, "yref", "y", y_mapping, y0=y0, y1=y1
            )
            composed.add_shape(payload)

        for annotation in source.layout.annotations:
            payload = copy.deepcopy(annotation.to_plotly_json())
            cls._remap_layout_ref_payload(payload, "xref", "x", x_mapping)
            cls._remap_layout_ref_payload(
                payload, "yref", "y", y_mapping, y0=y0, y1=y1
            )
            composed.add_annotation(payload)

    @classmethod
    def _axis_ref_mapping(
        cls,
        source: go.Figure,
        layout: dict[str, Any],
        axis: str,
        axis_counts: dict[str, int],
    ) -> dict[str, str]:
        """Map source axis refs to newly allocated composed axis refs."""
        refs = {
            _axis_ref_from_layout_key(key, axis)
            for key in layout
            if key.startswith(f"{axis}axis")
        }
        for trace in source.data:
            refs.add(trace.to_plotly_json().get(f"{axis}axis", axis))
        if not refs:
            refs.add(axis)
        ordered_refs = sorted(refs, key=lambda ref: _axis_ref_order(ref, axis))
        return {ref: _next_axis_ref(axis_counts, axis) for ref in ordered_refs}

    @staticmethod
    def _remap_axis_payload(
        payload: dict[str, Any],
        x_mapping: dict[str, str],
        y_mapping: dict[str, str],
    ) -> None:
        """Rewrite axis-linking fields such as anchor and scaleanchor."""
        for key, value in list(payload.items()):
            if not isinstance(value, str):
                continue
            if value in x_mapping:
                payload[key] = x_mapping[value]
            elif value in y_mapping:
                payload[key] = y_mapping[value]

    @staticmethod
    def _remap_trace_domain(
        payload: dict[str, Any], *, y0: float, y1: float
    ) -> None:
        """Scale a domain-based trace into the requested vertical slice."""
        domain = copy.deepcopy(payload.get("domain", {}))
        ydomain = domain.get("y", [0.0, 1.0])
        domain["y"] = [
            _remap_domain_value(float(ydomain[0]), y0, y1),
            _remap_domain_value(float(ydomain[1]), y0, y1),
        ]
        payload["domain"] = domain

    @staticmethod
    def _remap_layout_ref_payload(
        payload: dict[str, Any],
        ref_key: str,
        axis: str,
        mapping: dict[str, str],
        *,
        y0: float | None = None,
        y1: float | None = None,
    ) -> None:
        """Rewrite an annotation/shape xref or yref into the composed figure."""
        ref = payload.get(ref_key, "paper")
        coord_key = axis
        if ref == "paper":
            if axis == "y" and y0 is not None and y1 is not None:
                for key in (coord_key, f"{coord_key}0", f"{coord_key}1"):
                    if key in payload and isinstance(payload[key], (int, float)):
                        payload[key] = _remap_domain_value(
                            float(payload[key]), y0, y1
                        )
            return

        suffix = " domain" if ref.endswith(" domain") else ""
        base_ref = ref[: -len(" domain")] if suffix else ref
        if base_ref in mapping:
            payload[ref_key] = f"{mapping[base_ref]}{suffix}"

    # -- helpers ------------------------------------------------------------

    def _preprocess_roi(self, roi_idx: int) -> Any:
        """Run the profile's preprocessing on a single ROI.

        Delegates to :meth:`ColorCheckerProfile._preprocess_roi` so the report
        observes the same pixels (and respects ``pad_checker``) as the fit path.

        Args:
            roi_idx: Index into the held ROI list.

        Returns:
            A ``_RoiPreprocessing`` named tuple of every preprocessing stage.
        """
        row_sl, col_sl = self._rois[roi_idx]  # type: ignore[index]
        return self.profile._preprocess_roi(
            self._image,  # type: ignore[arg-type]
            row_sl,
            col_sl,
        )

    def _sorted_patches(self, *, key: str) -> list[tuple[str, dict[str, Any]]]:
        """Return per-patch diagnostics sorted worst-first by *key*.

        Args:
            key: The patch metric to sort by (e.g. ``"deltaE00_before"``).

        Returns:
            ``(name, patch_dict)`` pairs, descending by *key* (``None`` → 0).
        """
        patches = self.profile.diagnostics.get("patches", {})
        return sorted(
            patches.items(),
            key=lambda kv: kv[1].get(key) or 0.0,
            reverse=True,
        )

    # -- Section A: Delta-E 2000 -------------------------------------------

    @figure(
        title="Delta E 2000 Before & After Correction",
        section=_SECTION_DELTA_E,
        primary=True,
    )
    def fig_delta_e(self) -> go.Figure:
        """Per-patch Delta-E 2000 before/after correction (legacy section C).

        Mirrors ``_delta_e_section``: a grouped bar chart of each patch's
        Delta-E 2000 before and after correction (sorted worst-first by the
        before value), overlaid with the just-noticeable / perceptible /
        significant reference levels. The aggregate mean/max/median dE00 and
        the correction-matrix condition number are annotated.

        Returns:
            A ``go.Figure`` whose primary traces are two ``go.Bar`` series.
        """
        diag = self.profile.diagnostics
        sorted_items = self._sorted_patches(key="deltaE00_before")

        names = [name for name, _ in sorted_items]
        de_before = [p.get("deltaE00_before") or 0.0 for _, p in sorted_items]
        de_after = [p.get("deltaE00_after") or 0.0 for _, p in sorted_items]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=names,
                y=de_before,
                name="Before",
                marker=dict(color=NAVY),
            )
        )
        fig.add_trace(
            go.Bar(
                x=names,
                y=de_after,
                name="After",
                marker=dict(color=_SKY),
            )
        )
        for ref_val, ref_label in _REFERENCE_LINES:
            fig.add_hline(
                y=ref_val,
                line=dict(color=_GREY, width=1, dash="dash"),
                annotation_text=ref_label,
                annotation_position="top right",
                annotation_font=dict(size=10, color=_GREY),
            )

        mean_after = diag.get("mean_deltaE00_after", 0.0)
        max_after = diag.get("max_deltaE00_after", 0.0)
        median_after = diag.get("median_deltaE00_after", 0.0)
        cond = diag.get("correction_matrix_condition_number", 0.0)
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            showarrow=False,
            align="right",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=GOLD,
            borderwidth=1,
            text=(
                f"<b>Mean dE00:</b> {mean_after:.2f}<br>"
                f"<b>Max dE00:</b> {max_after:.2f}<br>"
                f"<b>Median dE00:</b> {median_after:.2f}<br>"
                f"<b>Condition #:</b> {cond:.1f}"
            ),
        )
        fig.update_layout(
            title="Delta E 2000 Before & After Correction",
            barmode="group",
            xaxis_title="Patch",
            yaxis_title="ΔE₀₀",
            xaxis=dict(tickangle=-45),
        )
        return fig

    # -- Section B: Matched patches ---------------------------------------

    @figure(
        title="Matched Patches: Reference | Measured | Corrected",
        section=_SECTION_PATCHES,
    )
    def fig_patch_swatches(self) -> go.Figure:
        """Matched reference/measured/corrected swatch strip (legacy section D).

        Mirrors ``_patches_section``: one row per patch (sorted worst-first by
        the after-correction dE00) with three columns — reference, measured,
        and corrected sRGB swatches — rendered as a ``go.Image``. Rejected
        patches and high-dE patches are flagged in the row labels.

        Returns:
            A ``go.Figure`` whose primary trace is a ``go.Image`` swatch grid.
        """
        from ._color_checker_profile import _illuminant_xy

        target_xy = _illuminant_xy(self.profile.target_illuminant)
        rejected = set(self.profile.diagnostics.get("rejected_patches", []))
        sorted_items = self._sorted_patches(key="deltaE00_after")

        n = len(sorted_items)
        # Rows top-to-bottom = worst patch first; go.Image draws row 0 at the
        # top, which already matches the worst-first ordering.
        swatches = np.zeros((max(n, 1), 3, 3), dtype=np.float64)
        row_labels: list[str] = []
        for idx, (name, patch) in enumerate(sorted_items):
            for col, key in enumerate(
                ("reference_lab", "measured_lab", "corrected_lab")
            ):
                lab = patch.get(key)
                if lab is not None:
                    swatches[idx, col] = _lab_to_srgb(
                        np.asarray(lab, dtype=np.float64), illuminant=target_xy
                    )
            de_after = patch.get("deltaE00_after")
            de_str = f"{de_after:.1f}" if de_after is not None else "N/A"
            flag = " [REJ]" if name in rejected else ""
            row_labels.append(f"{name}{flag} (dE={de_str})")

        # one patch label per cell so hover shows the patch name (not the pixel
        # row index that ``%{y}`` would give); shape (max(n,1), 3) matches z.
        hover_labels = row_labels if n > 0 else [""]
        customdata = np.array(
            [[lbl] * 3 for lbl in hover_labels], dtype=object
        )
        fig = go.Figure(
            go.Image(
                z=np.round(swatches * 255).astype(np.uint8),
                customdata=customdata,
                hovertemplate="%{customdata}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Matched Patches: Reference | Measured | Corrected",
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=["Reference", "Measured", "Corrected"],
            showgrid=False,
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(n)),
            ticktext=row_labels,
            showgrid=False,
            autorange="reversed",
        )
        return fig

    # -- Section C: Pipeline steps ----------------------------------------

    @figure(title="Pipeline Preprocessing Steps", section=_SECTION_PIPELINE)
    def fig_pipeline_steps(self) -> go.Figure:
        """Per-ROI preprocessing stages (legacy section A).

        Mirrors ``_pipeline_section``: for each ROI, the original crop, the
        background-trimmed, median-filtered, centred/padded, and border-masked
        stages laid out as a faceted grid of ``go.Image`` panels. The
        border-masked stage darkens excluded pixels to a third intensity so the
        swatch interior the fit actually samples stands out.

        Returns:
            A faceted ``go.Figure`` (``plotly.subplots``) of ``go.Image``
            panels, one row per ROI and one column per stage.
        """
        from plotly.subplots import make_subplots

        stage_labels = [
            "1. Original Crop",
            "2. Background Trimmed",
            "3. Median Filtered",
            "4. Centered & Padded",
            "5. Border Mask",
        ]
        n_rois = len(self._rois)  # type: ignore[arg-type]
        fig = make_subplots(
            rows=n_rois,
            cols=len(stage_labels),
            subplot_titles=stage_labels * n_rois,
            horizontal_spacing=0.02,
            vertical_spacing=0.05,
        )
        for roi_idx in range(n_rois):
            prep = self._preprocess_roi(roi_idx)
            mask_overlay = prep.padded.copy()
            mask_overlay[~prep.swatch_roi_mask] = (
                mask_overlay[~prep.swatch_roi_mask] // 3
            )
            stages = [
                prep.original,
                prep.trimmed,
                prep.filtered,
                prep.padded,
                mask_overlay,
            ]
            for col, arr in enumerate(stages, start=1):
                display = _normalize_display(arr)
                fig.add_trace(
                    go.Image(z=np.round(display * 255).astype(np.uint8)),
                    row=roi_idx + 1,
                    col=col,
                )
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        fig.update_layout(title="Pipeline Preprocessing Steps")
        return fig

    # -- Section D: Segmentation ------------------------------------------

    @figure(title="Patch Segmentation", section=_SECTION_SEGMENTATION)
    def fig_segmentation(self) -> go.Figure:
        """Preprocessed image with chip-mask overlay (legacy section B).

        Mirrors ``_segmentation_section``: for each ROI, the preprocessed image
        beside the same image with each detected chip mask tinted a distinct
        color (brighter on the reliable core). Chips are segmented
        non-strictly so partial detections still render. Both panels are
        ``go.Image`` traces in a faceted grid.

        Returns:
            A faceted ``go.Figure`` (``plotly.subplots``) of ``go.Image``
            panels: two columns (preprocessed, overlay) per ROI row.
        """
        from plotly.subplots import make_subplots

        from ._helpers import compute_core_mask, segment_chips_by_border_fill

        ref_Lab_dict, _, _ = self.profile._load_refs()
        ref_Lab_tuples = {
            name: tuple(lab.tolist()) for name, lab in ref_Lab_dict.items()
        }
        n_rois = len(self._rois)  # type: ignore[arg-type]
        fig = make_subplots(
            rows=n_rois,
            cols=2,
            subplot_titles=["Preprocessed Image", "Chip Masks (bright = core)"]
            * n_rois,
            horizontal_spacing=0.04,
            vertical_spacing=0.08,
        )
        rng = np.random.default_rng(42)
        for roi_idx in range(n_rois):
            prep = self._preprocess_roi(roi_idx)
            display = _normalize_display(prep.padded_normed)

            blob_masks, _blob_names = segment_chips_by_border_fill(
                prep.swatch_roi_mask,
                prep.lab,
                ref_Lab_tuples,
                min_swatch_area_frac=self.profile.min_swatch_area_frac,
                strict=False,
            )
            overlay = display.copy()
            for mask in blob_masks:
                if not mask.any():
                    continue
                color = rng.uniform(0.3, 0.9, size=3)
                overlay[mask] = 0.6 * overlay[mask] + 0.4 * color
                core = compute_core_mask(
                    mask, core_fraction=self.profile.core_fraction
                )
                overlay[core] = 0.3 * overlay[core] + 0.7 * color

            fig.add_trace(
                go.Image(z=np.round(display * 255).astype(np.uint8)),
                row=roi_idx + 1,
                col=1,
            )
            fig.add_trace(
                go.Image(
                    z=np.round(np.clip(overlay, 0, 1) * 255).astype(np.uint8)
                ),
                row=roi_idx + 1,
                col=2,
            )
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        fig.update_layout(title="Patch Segmentation")
        return fig
