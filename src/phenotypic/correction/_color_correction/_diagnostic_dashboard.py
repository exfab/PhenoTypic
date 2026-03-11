"""Interactive Panel dashboard for color correction diagnostics.

Visualises each key step of the color correction pipeline: cropping,
segmentation, Delta E 2000 scores per patch, and matched patches
(reference vs measured vs corrected).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import colour
import matplotlib.pyplot as plt
import numpy as np

try:
    import param
    import panel as pn

    PANEL_AVAILABLE = True
except ImportError:
    PANEL_AVAILABLE = False
    param = None  # type: ignore[assignment]
    pn = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from phenotypic._core._image import Image

    from ._color_checker_profile import ColorCheckerProfile

# ---------------------------------------------------------------------------
# Style constants (Okabe-Ito / dashboard style guide)
# ---------------------------------------------------------------------------

_COLOR_NAVY = "#003660"
_COLOR_SKY = "#56B4E9"
_COLOR_GREEN = "#009E73"
_COLOR_ORANGE = "#E69F00"
_COLOR_VERMILION = "#D55E00"
_COLOR_GREY = "#BBBBBB"
_COLOR_MUTED = "#8892a4"

_REFERENCE_LINES = [
    (1.0, "Just noticeable"),
    (2.3, "Perceptible"),
    (5.0, "Significant"),
]


_DASHBOARD_STYLE: dict[str, Any] = {
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#f5f7fa",
    "axes.edgecolor": "#dde3ed",
    "axes.grid": True,
    "grid.color": "#e8ecf2",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": "#2e3a4e",
    "xtick.color": _COLOR_MUTED,
    "ytick.color": _COLOR_MUTED,
    "axes.titlecolor": _COLOR_NAVY,
    "axes.titleweight": "600",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


# ---------------------------------------------------------------------------
# Lab -> sRGB conversion helper
# ---------------------------------------------------------------------------


def _lab_to_srgb(
    lab: np.ndarray,
    illuminant: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a Lab array to clipped sRGB [0, 1].

    Args:
        lab: Array of shape ``(..., 3)`` in CIE Lab.
        illuminant: CIE xy chromaticity of the whitepoint the Lab values
            are relative to.  Defaults to D65.

    Returns:
        sRGB array clipped to ``[0, 1]``.
    """
    kwargs: dict[str, Any] = {}
    if illuminant is not None:
        kwargs["illuminant"] = illuminant
    XYZ = colour.Lab_to_XYZ(lab, **kwargs)
    rgb = colour.XYZ_to_sRGB(XYZ)
    return np.clip(rgb, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Dashboard class
# ---------------------------------------------------------------------------

if PANEL_AVAILABLE:

    class ColorCorrectionDashboard(param.Parameterized):
        """Interactive Panel dashboard for color correction quality inspection.

        Provides live-updating views of pipeline preprocessing steps,
        patch segmentation, Delta E 2000 before/after scores, and
        matched color swatches.

        Not registered globally (not accessed via ``image.panel.*``).
        Accessed directly via ``profile.diagnostic(image, rois)``.
        """

        selected_roi = param.Integer(
            default=0, bounds=(0, None),
            doc="Index of the ROI to inspect for pipeline/segmentation views.",
        )
        show_pipeline = param.Boolean(
            default=True, doc="Show pipeline preprocessing steps.",
        )
        show_segmentation = param.Boolean(
            default=True, doc="Show patch segmentation masks.",
        )
        show_delta_e = param.Boolean(
            default=True, doc="Show Delta E 2000 bar chart.",
        )
        show_patches = param.Boolean(
            default=True, doc="Show matched patch swatches.",
        )

        def __init__(
            self,
            profile: ColorCheckerProfile,
            image: Image | None = None,
            rois: list[tuple[slice, slice]] | None = None,
            **params: Any,
        ) -> None:
            """Initialise the dashboard.

            Args:
                profile: A fitted ColorCheckerProfile.
                image: Source image for pipeline step visualisation.
                rois: ROI slices for pipeline step visualisation.
                **params: Additional param.Parameterized parameters.
            """
            super().__init__(**params)
            self._profile = profile
            self._image = image
            self._rois = rois
            self._has_image = image is not None and rois is not None

            if self._has_image and rois:
                self.param.selected_roi.bounds = (0, len(rois) - 1)

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def _make_panel_figure(
            self,
            plot_func: Any,
            *args: Any,
            figsize: tuple[float, float] = (5, 4),
        ) -> pn.pane.Matplotlib:
            """Create a Panel-wrapped matplotlib figure.

            Args:
                plot_func: Callable(ax, *args) that draws on an Axes.
                *args: Extra args forwarded to *plot_func* after the Axes.
                figsize: Figure size in inches.

            Returns:
                A Panel Matplotlib pane.
            """
            with plt.rc_context(_DASHBOARD_STYLE):
                fig, ax = plt.subplots(figsize=figsize)
                plot_func(ax, *args)
                fig.tight_layout()
                pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
                plt.close(fig)
            return pane

        def _preprocess_roi(
            self, roi_idx: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """Run preprocessing pipeline on a single ROI.

            Args:
                roi_idx: Index into ``self._rois``.

            Returns:
                Tuple of ``(original_crop, trimmed, filtered, padded)``
                arrays.
            """
            from ._helpers import (
                center_and_pad_checker,
                median_filter_rgb,
                trim_background_edges,
            )

            row_sl, col_sl = self._rois[roi_idx]  # type: ignore[index]
            original = self._image.rgb[row_sl, col_sl].copy()  # type: ignore[union-attr]
            trimmed = trim_background_edges(original)
            filtered = median_filter_rgb(
                trimmed, size=self._profile.median_filter_size,
            )
            padded = center_and_pad_checker(
                filtered,
                filter_size=self._profile.median_filter_size,
                stddev_mag_threshold=self._profile.stddev_mag_threshold,
            )
            return original, trimmed, filtered, padded

        # ------------------------------------------------------------------
        # Section A: Pipeline Steps
        # ------------------------------------------------------------------

        @param.depends("selected_roi", "show_pipeline")
        def _pipeline_section(self) -> pn.Column:
            if not self.show_pipeline or not self._has_image:
                return pn.Column()

            original, trimmed, filtered, padded = self._preprocess_roi(
                self.selected_roi,
            )

            stages = [
                ("1. Original Crop", original),
                ("2. Background Trimmed", trimmed),
                ("3. Median Filtered", filtered),
                ("4. Centered & Padded", padded),
            ]

            with plt.rc_context(_DASHBOARD_STYLE):
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                for ax, (title, arr) in zip(axes, stages):
                    display = arr.astype(np.float64)
                    if display.max() > 1.0:
                        display = display / display.max()
                    ax.imshow(np.clip(display, 0, 1))
                    ax.set_title(title, fontsize=9, color=_COLOR_NAVY)
                    ax.axis("off")
                fig.suptitle(
                    f"Pipeline Steps -- ROI {self.selected_roi}",
                    fontsize=12, color=_COLOR_NAVY, fontweight="600",
                )
                fig.tight_layout()
                pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
                plt.close(fig)

            return pn.Card(
                pane,
                title="Pipeline Preprocessing Steps",
                collapsed=False,
                sizing_mode="stretch_width",
            )

        # ------------------------------------------------------------------
        # Section B: Segmentation
        # ------------------------------------------------------------------

        @param.depends("selected_roi", "show_segmentation")
        def _segmentation_section(self) -> pn.Column:
            if not self.show_segmentation or not self._has_image:
                return pn.Column()

            from ._helpers import (
                _normalize_to_unit_float,
                _rgb_to_lab,
                compute_core_mask,
                lab_checker_cluster_masks,
                validate_patch_shape,
            )

            _, _, _, padded = self._preprocess_roi(self.selected_roi)

            # Load reference Lab for clustering.
            ref_Lab_dict, _, target_wp_xy = self._profile._load_refs()
            padded_float = _normalize_to_unit_float(padded)
            lab_padded = _rgb_to_lab(padded_float, illuminant_xy=target_wp_xy)

            ref_Lab_tuples = {
                name: tuple(lab.tolist())
                for name, lab in ref_Lab_dict.items()
            }
            masks, bboxes, labels = lab_checker_cluster_masks(
                lab_padded,
                ref_Lab_tuples,
                border_distance_threshold=self._profile.border_distance_threshold,
                include_labels=True,
            )  # type: ignore[misc]

            # Build overlay image.
            display = padded_float.copy()
            if display.max() > 1.0:
                display = display / display.max()

            rng = np.random.default_rng(42)
            overlay = np.zeros((*display.shape[:2], 4), dtype=np.float64)

            warnings_text: list[str] = []
            for mask, label in zip(masks, labels):
                if label == "border":
                    continue
                if not mask.any():
                    continue
                color = rng.uniform(0.3, 0.9, size=3)
                overlay[mask, :3] = color
                overlay[mask, 3] = 0.4

                core = compute_core_mask(
                    mask, core_fraction=self._profile.core_fraction,
                )
                overlay[core, 3] = 0.7  # brighter for core

                _, patch_warnings = validate_patch_shape(core)
                if patch_warnings:
                    warnings_text.append(
                        f"**{label}**: {'; '.join(patch_warnings)}"
                    )

            with plt.rc_context(_DASHBOARD_STYLE):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].imshow(np.clip(display, 0, 1))
                axes[0].set_title(
                    "Preprocessed Image", fontsize=9, color=_COLOR_NAVY,
                )
                axes[0].axis("off")

                axes[1].imshow(np.clip(display, 0, 1))
                axes[1].imshow(overlay)
                axes[1].set_title(
                    "Cluster Masks (bright = core)",
                    fontsize=9, color=_COLOR_NAVY,
                )
                axes[1].axis("off")

                fig.suptitle(
                    f"Segmentation -- ROI {self.selected_roi}",
                    fontsize=12, color=_COLOR_NAVY, fontweight="600",
                )
                fig.tight_layout()
                pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
                plt.close(fig)

            items = [pane]
            if warnings_text:
                items.append(
                    pn.pane.Markdown(
                        "**Patch validation warnings:**\n\n"
                        + "\n\n".join(warnings_text),
                    )
                )

            return pn.Card(
                *items,
                title="Patch Segmentation",
                collapsed=False,
                sizing_mode="stretch_width",
            )

        # ------------------------------------------------------------------
        # Section C: Delta E 2000 Scores
        # ------------------------------------------------------------------

        @param.depends("show_delta_e")
        def _delta_e_section(self) -> pn.Column:
            if not self.show_delta_e:
                return pn.Column()

            diag = self._profile.diagnostics
            patches = diag.get("patches", {})
            if not patches:
                return pn.Column(
                    pn.pane.Markdown("*No patch data available.*"),
                )

            # Sort by dE before (worst first).
            sorted_items = sorted(
                patches.items(),
                key=lambda kv: kv[1].get("deltaE00_before") or 0.0,
                reverse=True,
            )

            names = [name for name, _ in sorted_items]
            de_before = [
                p.get("deltaE00_before") or 0.0 for _, p in sorted_items
            ]
            de_after = [
                p.get("deltaE00_after") or 0.0 for _, p in sorted_items
            ]

            with plt.rc_context(_DASHBOARD_STYLE):
                fig, ax = plt.subplots(
                    figsize=(max(10, len(names) * 0.5), 5),
                )
                x = np.arange(len(names))
                width = 0.35

                ax.bar(
                    x - width / 2, de_before, width,
                    label="Before", color=_COLOR_NAVY, zorder=3,
                )
                ax.bar(
                    x + width / 2, de_after, width,
                    label="After", color=_COLOR_SKY, zorder=3,
                )

                # Reference lines.
                for ref_val, ref_label in _REFERENCE_LINES:
                    ax.axhline(
                        ref_val, color=_COLOR_GREY, linestyle="--",
                        linewidth=1, zorder=2,
                    )
                    ax.text(
                        len(names) - 0.5, ref_val + 0.15, ref_label,
                        fontsize=7, color=_COLOR_MUTED, ha="right",
                    )

                ax.set_xticks(x)
                ax.set_xticklabels(
                    [n[:16] for n in names],
                    rotation=45, ha="right", fontsize=7,
                )
                ax.set_ylabel(r"$\Delta E_{00}$", fontsize=9)
                ax.set_title(
                    "Delta E 2000 Before & After Correction",
                    fontsize=11, color=_COLOR_NAVY, fontweight="600",
                )
                ax.legend(fontsize=8)
                fig.tight_layout()
                chart_pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
                plt.close(fig)

            # Summary stat cards.
            mean_after = diag.get("mean_deltaE00_after", 0.0)
            max_after = diag.get("max_deltaE00_after", 0.0)
            median_after = diag.get("median_deltaE00_after", 0.0)
            cond = diag.get("correction_matrix_condition_number", 0.0)

            def _de_color(val: float) -> str:
                if val <= 1.0:
                    return _COLOR_GREEN
                if val <= 2.3:
                    return _COLOR_NAVY
                if val <= 5.0:
                    return _COLOR_ORANGE
                return _COLOR_VERMILION

            stats_html = (
                "<div style='display:flex; gap:16px; flex-wrap:wrap; "
                "margin-bottom:8px;'>"
                f"<div style='text-align:center; padding:8px 16px; "
                f"border:1px solid #dde3ed; border-radius:10px;'>"
                f"<div style='font-size:24px; font-weight:600; "
                f"color:{_de_color(mean_after)};'>{mean_after:.2f}</div>"
                f"<div style='font-size:11px; color:{_COLOR_MUTED}; "
                f"text-transform:uppercase;'>Mean dE00</div></div>"
                f"<div style='text-align:center; padding:8px 16px; "
                f"border:1px solid #dde3ed; border-radius:10px;'>"
                f"<div style='font-size:24px; font-weight:600; "
                f"color:{_de_color(max_after)};'>{max_after:.2f}</div>"
                f"<div style='font-size:11px; color:{_COLOR_MUTED}; "
                f"text-transform:uppercase;'>Max dE00</div></div>"
                f"<div style='text-align:center; padding:8px 16px; "
                f"border:1px solid #dde3ed; border-radius:10px;'>"
                f"<div style='font-size:24px; font-weight:600; "
                f"color:{_de_color(median_after)};'>{median_after:.2f}</div>"
                f"<div style='font-size:11px; color:{_COLOR_MUTED}; "
                f"text-transform:uppercase;'>Median dE00</div></div>"
                f"<div style='text-align:center; padding:8px 16px; "
                f"border:1px solid #dde3ed; border-radius:10px;'>"
                f"<div style='font-size:24px; font-weight:600; "
                f"color:{_COLOR_SKY};'>{cond:.1f}</div>"
                f"<div style='font-size:11px; color:{_COLOR_MUTED}; "
                f"text-transform:uppercase;'>Condition #</div></div>"
                "</div>"
            )

            return pn.Card(
                pn.pane.HTML(stats_html),
                chart_pane,
                title="Delta E 2000 Scores",
                collapsed=False,
                sizing_mode="stretch_width",
            )

        # ------------------------------------------------------------------
        # Section D: Matched Patches
        # ------------------------------------------------------------------

        @param.depends("show_patches")
        def _patches_section(self) -> pn.Column:
            if not self.show_patches:
                return pn.Column()

            diag = self._profile.diagnostics
            patches = diag.get("patches", {})
            rejected = set(diag.get("rejected_patches", []))
            if not patches:
                return pn.Column(
                    pn.pane.Markdown("*No patch data available.*"),
                )

            # Resolve illuminant for Lab -> sRGB conversion.
            from ._color_checker_profile import _illuminant_xy

            target_xy = _illuminant_xy(
                self._profile.target_illuminant,
            )

            # Sort by deltaE after (worst first).
            sorted_items = sorted(
                patches.items(),
                key=lambda kv: kv[1].get("deltaE00_after") or 0.0,
                reverse=True,
            )

            n = len(sorted_items)
            cols = min(6, n)
            rows = (n + cols - 1) // cols

            with plt.rc_context(_DASHBOARD_STYLE):
                fig, axes = plt.subplots(
                    rows, cols, figsize=(cols * 2.5, rows * 2.8),
                )
                if n == 1:
                    axes = np.array([[axes]])
                elif rows == 1:
                    axes = axes.reshape(1, -1)
                elif cols == 1:
                    axes = axes.reshape(-1, 1)

                for idx, (name, p) in enumerate(sorted_items):
                    r, c = divmod(idx, cols)
                    ax = axes[r, c]

                    ref_lab = p.get("reference_lab")
                    meas_lab = p.get("measured_lab")
                    corr_lab = p.get("corrected_lab")
                    de_after = p.get("deltaE00_after")

                    # Build 3-swatch strip: ref | measured | corrected.
                    swatches = np.zeros((1, 3, 3))
                    if ref_lab is not None:
                        swatches[0, 0] = _lab_to_srgb(
                            np.array(ref_lab), illuminant=target_xy,
                        )
                    if meas_lab is not None:
                        swatches[0, 1] = _lab_to_srgb(
                            np.array(meas_lab), illuminant=target_xy,
                        )
                    if corr_lab is not None:
                        swatches[0, 2] = _lab_to_srgb(
                            np.array(corr_lab), illuminant=target_xy,
                        )

                    ax.imshow(
                        np.clip(swatches, 0, 1),
                        aspect="auto",
                        interpolation="nearest",
                    )
                    ax.set_xticks([0, 1, 2])
                    ax.set_xticklabels(
                        ["Ref", "Meas", "Corr"], fontsize=6,
                    )
                    ax.set_yticks([])

                    # Title with dE and rejection status.
                    short_name = (
                        name[:14] + "..." if len(name) > 16 else name
                    )
                    if name in rejected:
                        title_color = _COLOR_VERMILION
                        short_name += " [REJ]"
                    elif de_after is not None and de_after > 5.0:
                        title_color = _COLOR_VERMILION
                    elif de_after is not None and de_after > 2.3:
                        title_color = _COLOR_ORANGE
                    else:
                        title_color = _COLOR_NAVY

                    de_str = (
                        f"{de_after:.1f}"
                        if de_after is not None
                        else "N/A"
                    )
                    ax.set_title(
                        f"{short_name}\ndE={de_str}",
                        fontsize=7,
                        color=title_color,
                        fontweight="500",
                    )

                # Hide unused axes.
                for idx in range(n, rows * cols):
                    r, c = divmod(idx, cols)
                    axes[r, c].set_visible(False)

                fig.suptitle(
                    "Matched Patches: Reference | Measured | Corrected",
                    fontsize=11, color=_COLOR_NAVY, fontweight="600",
                )
                fig.tight_layout()
                pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
                plt.close(fig)

            return pn.Card(
                pane,
                title="Matched Patch Comparison",
                collapsed=False,
                sizing_mode="stretch_width",
            )

        # ------------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------------

        def panel(self) -> pn.Column:
            """Return the interactive Panel layout.

            Returns:
                A Panel Column containing the full interactive dashboard.
            """
            # Build sidebar controls.
            controls_items: list[Any] = []

            if self._has_image:
                controls_items.append(
                    pn.Param(
                        self.param.selected_roi,
                        widgets={"selected_roi": pn.widgets.IntSlider},
                    )
                )

            controls_sections = pn.Card(
                pn.Param(self.param.show_delta_e),
                pn.Param(self.param.show_patches),
                *(
                    [
                        pn.Param(self.param.show_pipeline),
                        pn.Param(self.param.show_segmentation),
                    ]
                    if self._has_image
                    else []
                ),
                title="Sections",
                collapsed=False,
                width=300,
            )

            if controls_items:
                controls_params = pn.Card(
                    *controls_items,
                    title="Parameters",
                    collapsed=False,
                    width=300,
                )
                sidebar = pn.Column(controls_params, controls_sections, width=320)
            else:
                sidebar = pn.Column(controls_sections, width=320)

            # Diagnostics summary.
            diag = self._profile.diagnostics
            n_det = diag.get("n_patches_detected", 0)
            n_exp = diag.get("n_patches_expected", 0)
            n_rej = diag.get("n_patches_rejected", 0)
            summary = (
                f"**{diag.get('checker_type', 'Unknown')}** | "
                f"Illuminant: {diag.get('target_illuminant', 'D65')} | "
                f"Degree: {diag.get('degree', '?')} | "
                f"Patches: {n_det}/{n_exp} (rejected: {n_rej})"
            )

            sections: list[Any] = [
                pn.pane.Markdown("# Color Correction Diagnostics"),
                pn.pane.Markdown(summary),
                pn.Row(
                    sidebar,
                    pn.Column(
                        self._delta_e_section,
                        sizing_mode="stretch_width",
                    ),
                    sizing_mode="stretch_width",
                ),
                self._patches_section,
            ]

            if self._has_image:
                sections.extend([
                    self._pipeline_section,
                    self._segmentation_section,
                ])

            return pn.Column(*sections, sizing_mode="stretch_width")


__all__ = ["PANEL_AVAILABLE"]
if PANEL_AVAILABLE:
    __all__.append("ColorCorrectionDashboard")
