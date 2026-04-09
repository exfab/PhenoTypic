"""Detect mode comparison dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from phenotypic.tools_.register import register_dashboard

from ._base_dashboard import BaseDashboard

if TYPE_CHECKING:
    from phenotypic._core._image import Image


@register_dashboard
class DetectModeDashboard(BaseDashboard):
    """Dashboard for comparing detection mode matrices side-by-side.

    Provides an interactive Panel dashboard that displays all registered
    detection modes as thumbnails and allows selecting one for a larger
    preview, making it easy to choose the best source channel for colony
    detection on a given plate image.
    """

    call_name = "detect_modes"

    def __init__(self, root_image: Image) -> None:
        super().__init__(root_image)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_modes(self):
        """Return an interactive Panel dashboard comparing detection modes.

        The dashboard contains:
        - A header showing the image name and current detection mode.
        - A dropdown selector to pick any registered mode (plus the current
          ``detect_mat``).
        - A large central view that updates when the selector changes.
        - A row of thumbnails for every available mode for quick comparison.

        Returns:
            A ``pn.Column`` Panel layout that can be displayed with
            ``.show()`` in a script or ``.servable()`` in a notebook.

        Raises:
            ImportError: If the ``panel`` package is not installed.

        Examples:
            >>> from phenotypic.data import load_synth_yeast_plate
            >>> image = load_synth_yeast_plate()
            >>> dashboard = image.panel.detect_modes()
            >>> # dashboard.show()  # opens in browser
        """
        from phenotypic.tools_.panel_ import require_panel

        require_panel()

        import panel as pn

        # ---- Gather data ------------------------------------------------
        mode_matrices = self._compute_all_mode_matrices()

        # Add the current (possibly enhanced) detect_mat as a special entry
        detect_mat_label = f"detect_mat (current: {self._root_image.detect_mode})"
        mode_matrices[detect_mat_label] = self._root_image.detect_mat[:]

        mode_names = list(mode_matrices.keys())

        # ---- Header -----------------------------------------------------
        image_name = getattr(self._root_image, "name", None) or "Unnamed Image"
        header = pn.pane.Markdown(
            f"## Detect Mode Explorer — {image_name}\n"
            f"**Current mode:** `{self._root_image.detect_mode}`",
            sizing_mode="stretch_width",
        )

        # ---- Selector + main view --------------------------------------
        selector = pn.widgets.Select(
            name="Detection Mode",
            options=mode_names,
            value=detect_mat_label,
            width=200,
        )

        main_pane = pn.pane.Matplotlib(
            self._make_figure(mode_matrices[detect_mat_label], detect_mat_label),
            tight=True,
            dpi=100,
        )

        def _update_main(event):
            fig = self._make_figure(
                mode_matrices[event.new], event.new, figsize=(7, 7)
            )
            main_pane.object = fig

        selector.param.watch(_update_main, "value")

        selector_row = pn.Row(selector, main_pane)

        # ---- Thumbnails -------------------------------------------------
        thumbnails = []
        for name, arr in mode_matrices.items():
            thumb = self._render_array_as_pane(
                arr, title=name, figsize=(2.5, 2.5)
            )
            thumbnails.append(thumb)

        thumbnail_row = pn.Row(*thumbnails, scroll=True)

        # ---- Assemble ---------------------------------------------------
        dashboard = pn.Column(
            header,
            pn.layout.Divider(),
            selector_row,
            pn.layout.Divider(),
            pn.pane.Markdown("### All Modes"),
            thumbnail_row,
        )

        return dashboard

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_figure(
        arr: np.ndarray,
        title: str,
        figsize: tuple[float, float] = (7, 7),
    ) -> plt.Figure:
        """Create a matplotlib figure for the main preview pane."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(arr, cmap="gray", aspect="equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12)
        fig.tight_layout(pad=0.5)
        return fig
