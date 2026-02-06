"""Orchestrator: napari viewer with docked sweep-browsing widgets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from skimage import io as skio

from ._sweep_data_model import SweepOutputData, SweepOutputScanner

logger = logging.getLogger(__name__)


class NapariSweepViewer:
    """Napari-based viewer for browsing sweep output directories.

    Args:
        sweep_dir: Root of the sweep output (contains ``sweep_manifest.json``
            and ``results/``).
    """

    def __init__(self, sweep_dir: Path):
        self._sweep_dir = Path(sweep_dir).resolve()
        self._data: Optional[SweepOutputData] = None
        self._viewer = None
        self._current_layer_name: Optional[str] = None

    def launch(self):
        """Scan the sweep directory, create the napari viewer, and dock widgets.

        Returns:
            The :class:`napari.Viewer` instance.
        """
        import napari

        self._data = SweepOutputScanner.scan(self._sweep_dir)

        self._viewer = napari.Viewer(
            title=f"Sweep Viewer — {self._sweep_dir.name}",
        )

        # Lazy-import widgets (they need qtpy which is available once napari
        # has been imported).
        from ._file_tree_widget import SweepFileTreeWidget
        from ._measurements_table_widget import MeasurementsTableWidget
        from ._pipeline_info_widget import PipelineInfoWidget

        # Build widgets
        self._tree = SweepFileTreeWidget(self._data)
        self._info = PipelineInfoWidget(self._data.pipeline_configs)
        self._meas = MeasurementsTableWidget(self._data)

        # Dock them
        self._viewer.window.add_dock_widget(
            self._tree, name="File Browser", area="left",
        )
        self._viewer.window.add_dock_widget(
            self._info, name="Pipeline Info", area="right",
        )
        self._viewer.window.add_dock_widget(
            self._meas, name="Measurements", area="bottom",
        )

        # Wire signals
        self._tree.image_selected.connect(self._on_image_selected)
        self._tree.pipeline_selected.connect(self._info.set_pipeline)
        self._tree.compare_requested.connect(self._on_compare_requested)

        return self._viewer

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_image_selected(self, file_path_str: str) -> None:
        """Load a single image into the viewer, replacing the previous one."""
        arr = self._load_image_array(file_path_str)
        if arr is None:
            return

        layer_name = self._make_layer_name(file_path_str)

        # Remove old primary layer
        if self._current_layer_name and self._current_layer_name in self._viewer.layers:
            self._viewer.layers.remove(self._current_layer_name)

        # Determine layer type
        path = Path(file_path_str)
        info = self._path_info(path)
        if info["component"] == "objmap":
            self._viewer.add_labels(arr, name=layer_name)
        else:
            self._viewer.add_image(arr, name=layer_name)

        self._current_layer_name = layer_name

        # Disable grid (single image)
        self._viewer.grid.enabled = False

        # Update info + measurements pane
        self._info.set_pipeline(info["pipeline"])
        self._meas.set_selection(info["pipeline"], info["image_stem"])

    def _on_compare_requested(
        self, entries: List[Tuple[str, str]],
    ) -> None:
        """Load the same component from all pipelines for side-by-side view.

        Args:
            entries: List of ``(pipeline_name, file_path)`` tuples.
        """
        # Clear existing layers
        self._viewer.layers.clear()
        self._current_layer_name = None

        for pipe_name, fpath in entries:
            arr = self._load_image_array(fpath)
            if arr is None:
                continue
            name = self._make_layer_name(fpath)
            info = self._path_info(Path(fpath))
            if info["component"] == "objmap":
                self._viewer.add_labels(arr, name=name)
            else:
                self._viewer.add_image(arr, name=name)

        n = len(self._viewer.layers)
        if n > 1:
            self._viewer.grid.enabled = True
            self._viewer.grid.shape = (1, n)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image_array(file_path: str) -> Optional[np.ndarray]:
        """Load an image file into a NumPy array.

        Args:
            file_path: Absolute path string.

        Returns:
            NumPy array or ``None`` on failure.
        """
        try:
            p = Path(file_path)
            if p.exists():
                return skio.imread(str(p))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", file_path, exc)
        return None

    @staticmethod
    def _make_layer_name(file_path: str) -> str:
        """Derive ``pipeline/component/stem`` from file path."""
        p = Path(file_path)
        # Expected: …/results/<pipeline>/<component>/<file>
        try:
            component = p.parent.name
            pipeline = p.parent.parent.name
            return f"{pipeline}/{component}/{p.stem}"
        except Exception:
            return p.stem

    @staticmethod
    def _path_info(path: Path) -> dict:
        """Extract pipeline, component, and image_stem from a result path."""
        return {
            "component": path.parent.name,
            "pipeline": path.parent.parent.name,
            "image_stem": path.stem,
        }


def launch_sweep_viewer(sweep_dir: Path) -> None:
    """Convenience entry point: create viewer and start the napari event loop.

    Args:
        sweep_dir: Root of the sweep output directory.
    """
    import napari

    viewer = NapariSweepViewer(sweep_dir)
    viewer.launch()
    napari.run()
