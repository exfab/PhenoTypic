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
        customdata = np.array([[lbl] * 3 for lbl in hover_labels], dtype=object)
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
                go.Image(z=np.round(np.clip(overlay, 0, 1) * 255).astype(np.uint8)),
                row=roi_idx + 1,
                col=2,
            )
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        fig.update_layout(title="Patch Segmentation")
        return fig
