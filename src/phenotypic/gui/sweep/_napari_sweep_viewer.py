"""Orchestrator: napari viewer with docked sweep-browsing widgets."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from ._sweep_data_model import SweepOutputData, SweepOutputScanner

logger = logging.getLogger(__name__)

_LAYER_ORDER = ("rgb", "gray", "detect_mat", "objmap")


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
        from ._grouped_layer_widget import GroupedLayerWidget
        from ._pipeline_info_widget import PipelineInfoWidget

        # Build widgets
        self._tree = SweepFileTreeWidget(self._data)
        self._info = PipelineInfoWidget(self._data.pipeline_configs)
        self._layer_tree = GroupedLayerWidget(self._viewer)

        # Dock them
        self._viewer.window.add_dock_widget(
            self._tree, name="File Browser", area="right",
        )
        self._viewer.window.add_dock_widget(
            self._info, name="Pipeline Info", area="right",
        )
        self._viewer.window.add_dock_widget(
            self._layer_tree, name="Layers", area="left",
        )

        # Wire signals
        self._tree.pipeline_selected.connect(self._info.set_pipeline)
        self._tree.stem_selected.connect(self._on_stem_selected)
        self._tree.stem_compare_requested.connect(
            self._on_stem_compare,
        )

        self._tabify_native_layer_docks()

        return self._viewer

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_stem_selected(self, entries: List[dict]) -> None:
        """Load HDF5 layers for a single image stem.

        Args:
            entries: List of dicts with ``h5_path``, ``pipeline``,
                and ``image_stem`` keys.
        """
        logger.debug(
            "_on_stem_selected: %d entries", len(entries),
        )
        self._clear_current_layers()
        loaded_entries: list[dict] = []

        for entry in entries:
            layers = self._load_hdf5_layers(
                entry["h5_path"],
                entry["pipeline"],
                entry["image_stem"],
            )
            for layer in layers:
                if layer["is_labels"]:
                    self._viewer.add_labels(
                        layer["data"].astype(np.intp),
                        name=layer["name"],
                    )
                else:
                    self._viewer.add_image(
                        layer["data"], name=layer["name"],
                    )
                self._current_layer_names.append(layer["name"])
                loaded_entries.append(
                    {
                        "pipeline": layer["pipeline"],
                        "component": layer["component"],
                        "image_stem": layer["image_stem"],
                    }
                )

        self._viewer.grid.enabled = False

        if loaded_entries and entries:
            self._info.set_pipeline(entries[0]["pipeline"])

        self._layer_tree.set_layers(loaded_entries)

    def _on_stem_compare(self, entries: List[dict]) -> None:
        """Accumulate HDF5 layers for compare mode (no clearing).

        Args:
            entries: List of dicts with ``h5_path``, ``pipeline``,
                and ``image_stem`` keys.
        """
        logger.debug(
            "_on_stem_compare: %d entries", len(entries),
        )
        loaded_entries: list[dict] = []

        for entry in entries:
            layers = self._load_hdf5_layers(
                entry["h5_path"],
                entry["pipeline"],
                entry["image_stem"],
            )
            for layer in layers:
                if layer["is_labels"]:
                    self._viewer.add_labels(
                        layer["data"].astype(np.intp),
                        name=layer["name"],
                    )
                else:
                    self._viewer.add_image(
                        layer["data"], name=layer["name"],
                    )
                self._current_layer_names.append(layer["name"])
                loaded_entries.append(
                    {
                        "pipeline": layer["pipeline"],
                        "component": layer["component"],
                        "image_stem": layer["image_stem"],
                    }
                )

        n = len(self._viewer.layers)
        if n > 1:
            self._viewer.grid.enabled = True
            self._viewer.grid.stride = 4
            self._viewer.grid.shape = (-1, -1)

        self._layer_tree.add_layers(loaded_entries)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_hdf5_layers(
        self,
        h5_path: str,
        pipeline: str,
        image_stem: str,
    ) -> list[dict]:
        """Load HDF5 via Image.load_hdf5(), extract layers, free Image.

        Args:
            h5_path: Absolute path to the ``.h5`` file.
            pipeline: Pipeline name for layer naming.
            image_stem: Image stem for layer naming.

        Returns:
            List of dicts with keys: ``name``, ``data``, ``component``,
            ``pipeline``, ``image_stem``, ``is_labels``.
        """
        from phenotypic import Image

        p = Path(h5_path)
        if not p.exists():
            logger.warning("HDF5 file not found: %s", h5_path)
            return []

        try:
            image = Image.load_hdf5(h5_path)
        except Exception as exc:
            logger.warning(
                "Failed to load HDF5 %s: %s", h5_path, exc,
            )
            return []

        layers: list[dict] = []

        # RGB (check availability)
        if not image.rgb.isempty():
            layers.append(
                {
                    "name": f"{pipeline}/rgb/{image_stem}",
                    "data": image.rgb[:].copy(),
                    "component": "rgb",
                    "pipeline": pipeline,
                    "image_stem": image_stem,
                    "is_labels": False,
                }
            )

        # Gray (always available)
        layers.append(
            {
                "name": f"{pipeline}/gray/{image_stem}",
                "data": image.gray[:].copy(),
                "component": "gray",
                "pipeline": pipeline,
                "image_stem": image_stem,
                "is_labels": False,
            }
        )

        # Detection matrix
        layers.append(
            {
                "name": f"{pipeline}/detect_mat/{image_stem}",
                "data": image.detect_mat[:].copy(),
                "component": "detect_mat",
                "pipeline": pipeline,
                "image_stem": image_stem,
                "is_labels": False,
            }
        )

        # Object map (labels layer)
        layers.append(
            {
                "name": f"{pipeline}/objmap/{image_stem}",
                "data": image.objmap[:].copy(),
                "component": "objmap",
                "pipeline": pipeline,
                "image_stem": image_stem,
                "is_labels": True,
            }
        )

        del image
        gc.collect()

        return layers

    def _clear_current_layers(self) -> None:
        """Remove all layers tracked in ``_current_layer_names``."""
        for name in self._current_layer_names:
            if name in self._viewer.layers:
                self._viewer.layers.remove(name)
        self._current_layer_names = []
        if hasattr(self, "_layer_tree"):
            self._layer_tree.clear()

    def _tabify_native_layer_docks(self) -> None:
        """Tab napari's native layer-list and layer-controls docks."""
        try:
            qt_window = self._viewer.window._qt_window
            layer_list = (
                self._viewer.window._qt_viewer.dockLayerList
            )
            layer_controls = (
                self._viewer.window._qt_viewer.dockLayerControls
            )

            qt_window.tabifyDockWidget(
                layer_list, layer_controls,
            )
            layer_list.show()
            layer_list.raise_()
        except Exception as exc:
            logger.debug(
                "Could not tabify native layer docks: %s", exc,
            )


def launch_sweep_viewer(sweep_dir: Path) -> None:
    """Convenience entry point: create viewer and start the napari event loop.

    Args:
        sweep_dir: Root of the sweep output directory.
    """
    import napari

    viewer_obj = NapariSweepViewer(sweep_dir)
    viewer_obj.launch()
    logger.info(
        "Viewer launched — %d pipelines, %d images",
        len(viewer_obj._data.pipeline_names),
        len(viewer_obj._data.image_stems),
    )
    napari.run()
