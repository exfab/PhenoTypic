"""Orchestrator: napari viewer with docked sweep-browsing widgets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from skimage import io as skio

from ._sweep_data_model import SweepOutputData, SweepOutputScanner

logger = logging.getLogger(__name__)

# Components whose pixel values are integer object IDs (render as Labels).
_LABEL_COMPONENTS = frozenset({"objmap", "objmask"})


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
        self._current_layer_names: List[str] = []

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
            self._tree, name="File Browser", area="right",
        )
        self._viewer.window.add_dock_widget(
            self._info, name="Pipeline Info", area="right",
        )
        self._viewer.window.add_dock_widget(
            self._meas, name="Measurements", area="bottom",
        )

        # Wire signals
        self._tree.pipeline_selected.connect(self._info.set_pipeline)
        self._tree.stem_selected.connect(self._on_stem_selected)
        self._tree.stem_compare_requested.connect(
            self._on_stem_compare,
        )

        return self._viewer

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_stem_selected(self, entries: List[dict]) -> None:
        """Load all component layers for an image stem as a stack.

        Args:
            entries: List of dicts with ``path``, ``pipeline``,
                ``component``, and ``image_stem`` keys.
        """
        logger.debug("_on_stem_selected: %d entries", len(entries))
        self._clear_current_layers()

        loaded = 0
        for entry in entries:
            arr = self._load_image_array(entry["path"])
            if arr is None:
                continue
            name = self._make_layer_name(entry["path"])
            self._add_layer(arr, name, entry["component"])
            self._current_layer_names.append(name)
            loaded += 1

        failed = len(entries) - loaded
        if failed:
            logger.warning(
                "_on_stem_selected: %d/%d layers failed to load",
                failed, len(entries),
            )
        else:
            logger.debug(
                "_on_stem_selected: loaded %d layers", loaded,
            )

        self._viewer.grid.enabled = False

        # Update info + measurements only if layers were loaded
        if self._current_layer_names and entries:
            self._info.set_pipeline(entries[0]["pipeline"])
            self._meas.set_selection(
                entries[0]["pipeline"],
                entries[0]["image_stem"],
            )

    def _on_stem_compare(self, entries: List[dict]) -> None:
        """Accumulate layers for compare mode (no clearing).

        Adds all component layers for the given pipeline+stem on top
        of existing layers, then enables grid mode so each pipeline
        group occupies its own column.

        Args:
            entries: List of dicts with ``path``, ``pipeline``,
                ``component``, and ``image_stem`` keys.
        """
        logger.debug("_on_stem_compare: %d entries", len(entries))
        loaded = 0
        for entry in entries:
            arr = self._load_image_array(entry["path"])
            if arr is None:
                continue
            name = self._make_layer_name(entry["path"])
            self._add_layer(arr, name, entry["component"])
            self._current_layer_names.append(name)
            loaded += 1

        failed = len(entries) - loaded
        if failed:
            logger.warning(
                "_on_stem_compare: %d/%d layers failed to load",
                failed, len(entries),
            )
        else:
            logger.debug(
                "_on_stem_compare: loaded %d layers", loaded,
            )

        # Enable grid: one column per component in this group
        n = len(self._viewer.layers)
        if n > 1 and loaded > 0:
            self._viewer.grid.enabled = True
            self._viewer.grid.shape = (-1, loaded)

        if entries:
            self._meas.set_selection(
                entries[0]["pipeline"],
                entries[0]["image_stem"],
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_label_component(component: str) -> bool:
        """Return ``True`` if *component* should render as a Labels layer."""
        return component in _LABEL_COMPONENTS

    def _add_layer(
        self,
        arr: np.ndarray,
        name: str,
        component: str,
    ) -> None:
        """Add *arr* to the viewer as a Labels or Image layer.

        Labels layers require 2-D integer arrays.  If the component is
        nominally a label type but the array is not 2-D (e.g. an RGB
        overlay), it falls back to an Image layer.
        """
        if self._is_label_component(component) and arr.ndim == 2:
            self._viewer.add_labels(
                arr.astype(np.intp), name=name,
            )
        else:
            self._viewer.add_image(arr, name=name)

    def _clear_current_layers(self) -> None:
        """Remove all layers tracked in ``_current_layer_names``."""
        for name in self._current_layer_names:
            if name in self._viewer.layers:
                self._viewer.layers.remove(name)
        self._current_layer_names = []

    @staticmethod
    def _load_image_array(file_path: str) -> Optional[np.ndarray]:
        """Load an image file into a NumPy array.

        Args:
            file_path: Absolute path string.

        Returns:
            NumPy array or ``None`` on failure.
        """
        p = Path(file_path)
        if not p.exists():
            logger.warning("File not found: %s", file_path)
            return None
        try:
            arr = skio.imread(str(p))
            logger.debug(
                "Loaded %s — shape=%s dtype=%s",
                p.name, arr.shape, arr.dtype,
            )
            return arr
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


def launch_sweep_viewer(sweep_dir: Path) -> None:
    """Convenience entry point: create viewer and start the napari event loop.

    Args:
        sweep_dir: Root of the sweep output directory.
    """
    import napari

    viewer_obj = NapariSweepViewer(sweep_dir)
    viewer_obj.launch()
    logger.info(
        "Viewer launched — %d pipelines, %d images, %d components",
        len(viewer_obj._data.pipeline_names),
        len(viewer_obj._data.image_stems),
        len(viewer_obj._data.components),
    )
    napari.run()
