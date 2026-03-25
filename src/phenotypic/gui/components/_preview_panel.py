"""Preview panel component for PhenoTypic GUI.

Provides image preview with view selection, image loading options,
GridImage support, and loading indicators.
"""

from __future__ import annotations

from typing import Callable, List, Optional


class PreviewPanel:
    """Image preview panel with view selector and image loading.

    Features:
    - Multiple image loading options: file picker, drag-drop, phenotypic.data samples
    - Support for Image and GridImage (with nrows/ncols inputs)
    - Loading spinner during pipeline execution
    - Manual "Update Preview" button (no auto-update)
    """

    def __init__(
        self,
        image=None,  # Image or GridImage
        get_pipeline: Optional[Callable[[], Any]] = None,
        **params,
    ):
        """Initialize PreviewPanel.

        Args:
            image: Initial Image or GridImage to display
            get_pipeline: Callback to get current ImagePipeline for preview
            **params: Additional parameters
        """
        self._image = image
        self._get_pipeline = get_pipeline
        self._preview_image = None

        # State
        self.selected_view = "overlay"
        self.image_type = "GridImage"
        self.nrows = 8
        self.ncols = 12
        self.is_loading = False

    def panel(self):
        """Build the preview panel with image loading options.

        Returns:
            Panel Column widget
        """
        import panel as pn

        # === Image Loading Section ===
        # File picker
        file_input = pn.widgets.FileInput(accept=".tiff,.tif,.png,.jpg,.jpeg")
        file_input.param.watch(self._load_from_file, "value")

        # Sample data selector
        sample_options = self._get_sample_options()
        sample_select = pn.widgets.Select(
            name="Load Sample",
            options=[""] + sample_options,
            value="",
        )
        sample_select.param.watch(self._load_sample, "value")

        # GridImage options (shown when image_type == 'GridImage')
        nrows_input = pn.widgets.IntInput(
            name="Rows", value=self.nrows, start=1, end=64
        )
        ncols_input = pn.widgets.IntInput(
            name="Cols", value=self.ncols, start=1, end=96
        )

        def update_nrows(event):
            self.nrows = event.new

        def update_ncols(event):
            self.ncols = event.new

        nrows_input.param.watch(update_nrows, "value")
        ncols_input.param.watch(update_ncols, "value")

        grid_opts = pn.Column(
            nrows_input,
            ncols_input,
            visible=self.image_type == "GridImage",
        )

        image_type_select = pn.widgets.Select(
            name="Image Type", options=["Image", "GridImage"], value=self.image_type
        )

        def update_image_type(event):
            self.image_type = event.new
            grid_opts.visible = event.new == "GridImage"

        image_type_select.param.watch(update_image_type, "value")

        load_section = pn.Card(
            pn.Column(
                pn.Row(file_input, sample_select),
                pn.Row(image_type_select, grid_opts),
            ),
            header="Load Image",
            collapsed=self._image is not None,
        )

        # === Preview Section ===
        view_select = pn.widgets.Select(
            name="View",
            options=["rgb", "gray", "detect_mat", "objmask", "objmap", "overlay"],
            value=self.selected_view,
        )

        def update_view(event):
            self.selected_view = event.new
            self._update_display()

        view_select.param.watch(update_view, "value")

        # Update button
        update_btn = pn.widgets.Button(name="Update Preview", button_type="primary")
        update_btn.on_click(self._update_preview)

        # Loading spinner (hidden by default)
        self._loading_indicator = pn.indicators.LoadingSpinner(
            value=False, size=25, name="Processing...", visible=False
        )

        # Image display (left-aligned)
        self._image_pane = pn.pane.Matplotlib(
            sizing_mode="stretch_width",
            height=420,
            align="start",  # Left-align the image
        )

        # Create controls row with loading indicator (initially hidden)
        self._controls_row = pn.Row(
            view_select, update_btn, self._loading_indicator, sizing_mode="stretch_width"
        )

        # Build layout
        layout = pn.Column(
            load_section,
            self._controls_row,
            self._image_pane,
            sizing_mode="stretch_width",
            align="start",  # Left-align content
        )

        # Update display after layout is created
        if self._image:
            self._update_display()

        return layout

    def _load_from_file(self, event):
        """Load image from uploaded file."""
        if event.new is None:
            return
        import tempfile
        from pathlib import Path

        # Save to temp file and load
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tiff") as f:
            f.write(event.new)
            temp_path = Path(f.name)

        self._load_image(temp_path)

    def _load_sample(self, event):
        """Load sample image from phenotypic.data."""
        if not event.new:
            return
        from phenotypic import data

        loader = getattr(data, event.new, None)
        if loader:
            self._image = loader()
            self._preview_image = None
            self._update_display()

    def _load_image(self, path):
        """Load image from path with current settings."""
        from phenotypic import Image, GridImage

        if self.image_type == "GridImage":
            self._image = GridImage.imread(path, nrows=self.nrows, ncols=self.ncols)
        else:
            self._image = Image.imread(path)
        self._preview_image = None
        self._update_display()

    def _get_sample_options(self) -> List[str]:
        """Get available sample data loaders."""
        from phenotypic import data

        return [name for name in dir(data) if name.startswith("load_")]

    def _update_preview(self, event=None):
        """Apply pipeline and update preview with dynamic loading indicator."""
        if self._get_pipeline is None or self._image is None:
            return

        import copy

        # Show spinner by making it visible
        self.is_loading = True
        self._loading_indicator.value = True
        self._loading_indicator.visible = True

        try:
            pipeline = self._get_pipeline()
            self._preview_image = copy.deepcopy(self._image)
            pipeline.apply(self._preview_image, inplace=True)
            self._update_display()
        finally:
            # Hide spinner by making it invisible
            self.is_loading = False
            self._loading_indicator.value = False
            self._loading_indicator.visible = False

    def _update_display(self):
        """Update displayed image based on selected view."""
        import matplotlib.pyplot as plt

        image = self._preview_image or self._image
        if image is None:
            return

        try:
            view = self.selected_view

            # Handle overlay separately since it creates its own figure
            if view == "overlay":
                fig, ax = image.show(overlay=True)
            else:
                # Create figure for other views
                fig, ax = plt.subplots(figsize=(8, 6))

                if view == "rgb" and not image.rgb.isempty():
                    ax.imshow(image.rgb[:])
                elif view == "gray":
                    ax.imshow(image.gray[:], cmap="gray")
                elif view == "detect_mat":
                    ax.imshow(image.detect_mat[:], cmap="gray")
                elif view == "objmask":
                    ax.imshow(image.objmask[:], cmap="gray")
                elif view == "objmap":
                    ax.imshow(image.objmap[:], cmap="nipy_spectral")

                ax.axis("off")

            self._image_pane.object = fig
            plt.close(fig)
        except Exception as e:
            print(f"Error updating display: {e}")
            import traceback
            traceback.print_exc()
