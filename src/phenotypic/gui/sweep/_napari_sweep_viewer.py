"""Orchestrator: napari viewer with docked sweep-browsing widgets."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from ._sweep_data_model import (
    IntermediateStep,
    ResolvedLayerSources,
    SweepOutputData,
    SweepOutputScanner,
    build_layer_resolution_index,
)
from ._swept_param_analysis import get_swept_param_names

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

        # Split view state
        self._main_pipeline: Optional[str] = None
        self._split_pipeline: Optional[str] = None
        self._main_stem: Optional[str] = None
        self._split_stem: Optional[str] = None
        self._main_steps: List[IntermediateStep] = []
        self._split_steps: List[IntermediateStep] = []
        self._main_resolution_index: dict = {}
        self._split_resolution_index: dict = {}

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
        from ._parameter_explorer_widget import ParameterExplorerWidget
        from ._pipeline_config_bar import PipelineConfigBar
        from ._split_step_slider_widget import SplitStepSliderWidget

        # Build widgets
        self._tree = SweepFileTreeWidget(self._data)
        self._param_explorer = ParameterExplorerWidget()
        self._layer_toggle = GroupedLayerWidget(self._viewer)
        self._config_bar = PipelineConfigBar()
        self._step_slider = SplitStepSliderWidget()

        # Configure parameter explorer with sweep configs
        self._param_explorer.set_configs(self._data.pipeline_configs)

        # Dock widgets
        dock_fb = self._viewer.window.add_dock_widget(
            self._tree, name="File Browser", area="right",
        )
        dock_pe = self._viewer.window.add_dock_widget(
            self._param_explorer, name="Parameter Explorer", area="right",
        )
        dock_layers = self._viewer.window.add_dock_widget(
            self._layer_toggle, name="Layers", area="right",
        )
        self._viewer.window.add_dock_widget(
            self._config_bar, name="Pipeline Config", area="bottom",
        )
        self._viewer.window.add_dock_widget(
            self._step_slider, name="Step Slider", area="bottom",
        )

        # Wire file tree signals
        self._tree.pipeline_selected.connect(
            self._param_explorer.set_pipeline,
        )
        self._tree.stem_selected.connect(self._on_stem_selected)
        self._tree.stem_compare_requested.connect(
            self._on_stem_compare,
        )

        # Wire parameter explorer signals
        self._param_explorer.view_requested.connect(
            self._on_param_view_requested,
        )
        self._param_explorer.view_split_requested.connect(
            self._on_param_split_requested,
        )

        # Wire step slider signals
        self._step_slider.main_step_changed.connect(
            self._on_main_step_changed,
        )
        self._step_slider.split_step_changed.connect(
            self._on_split_step_changed,
        )

        self._organize_dock_layout(dock_fb, dock_pe, dock_layers)

        return self._viewer

    # ------------------------------------------------------------------
    # Signal handlers — file tree
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
        if not entries:
            return

        pipeline = entries[0]["pipeline"]
        stem = entries[0]["image_stem"]

        # Update parameter explorer to show current pipeline
        self._param_explorer.set_pipeline(pipeline)

        # Load as main view
        self._load_main_view(pipeline, stem)

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

        self._layer_toggle.add_layers(loaded_entries)

        # Compare mode accumulates layers without main/split prefixes,
        # so step scrubbing is not meaningful — clear both sliders.
        self._step_slider.clear_main()
        self._step_slider.clear_split()

    # ------------------------------------------------------------------
    # Signal handlers — parameter explorer
    # ------------------------------------------------------------------

    def _on_param_view_requested(self, pipeline_name: str) -> None:
        """Handle "View" button: load pipeline as main view, clear split.

        Args:
            pipeline_name: Pipeline to load.
        """
        logger.debug(
            "_on_param_view_requested: %s", pipeline_name,
        )
        # Use most recently active stem, or first available
        stem = self._main_stem or self._split_stem
        if stem is None:
            # Pick the first available stem for this pipeline
            by_pipe = self._data.by_pipeline.get(pipeline_name, {})
            if by_pipe:
                stem = next(iter(sorted(by_pipe.keys())))
            else:
                logger.warning(
                    "No images found for pipeline %r", pipeline_name,
                )
                return

        self._load_main_view(pipeline_name, stem)

    def _on_param_split_requested(self, pipeline_name: str) -> None:
        """Handle "View Split" button: load pipeline as right-side split.

        Args:
            pipeline_name: Pipeline to load as split view.
        """
        logger.debug(
            "_on_param_split_requested: %s", pipeline_name,
        )
        # If no main view, treat as a normal "View"
        if self._main_pipeline is None:
            self._on_param_view_requested(pipeline_name)
            return

        # Use the same stem as the main view
        stem = self._main_stem
        if stem is None:
            logger.warning("No main stem for split view")
            return

        self._load_split_view(pipeline_name, stem)

    # ------------------------------------------------------------------
    # Signal handlers — step sliders
    # ------------------------------------------------------------------

    def _on_main_step_changed(self, step_index: int) -> None:
        """Reload main layers from selected intermediate or final HDF5.

        Args:
            step_index: Index into ``_main_steps`` for intermediates,
                or ``len(_main_steps)`` for the final output.
        """
        if not self._main_pipeline or not self._main_stem:
            return

        result = self._resolve_step_sources(
            self._main_pipeline,
            self._main_stem,
            self._main_steps,
            step_index,
            self._main_resolution_index,
        )
        if result is None:
            return

        if isinstance(result, str):
            self._replace_main_layers(result, self._main_pipeline, self._main_stem)
        else:
            self._replace_main_layers_composited(
                result, self._main_pipeline, self._main_stem,
            )

    def _on_split_step_changed(self, step_index: int) -> None:
        """Reload split layers from selected intermediate or final HDF5.

        Args:
            step_index: Index into ``_split_steps`` for intermediates,
                or ``len(_split_steps)`` for the final output.
        """
        if not self._split_pipeline or not self._split_stem:
            return

        result = self._resolve_step_sources(
            self._split_pipeline,
            self._split_stem,
            self._split_steps,
            step_index,
            self._split_resolution_index,
        )
        if result is None:
            return

        if isinstance(result, str):
            self._replace_split_layers(result, self._split_pipeline, self._split_stem)
        else:
            self._replace_split_layers_composited(
                result, self._split_pipeline, self._split_stem,
            )

    # ------------------------------------------------------------------
    # View loading
    # ------------------------------------------------------------------

    def _load_main_view(self, pipeline_name: str, stem: str) -> None:
        """Load a pipeline as the main (left) view, clearing any split.

        Args:
            pipeline_name: Pipeline name to load.
            stem: Image stem to load.
        """
        # Clear everything
        self._clear_all_layers()

        # Resolve HDF5 path
        hdf5_entry = (
            self._data.by_image.get(stem, {}).get(pipeline_name)
        )
        if hdf5_entry is None:
            logger.warning(
                "No HDF5 for stem=%r pipeline=%r", stem, pipeline_name,
            )
            return

        # Load main layers
        layers = self._load_hdf5_layers(
            str(hdf5_entry.path), pipeline_name, stem, prefix="main",
        )
        loaded_entries = self._add_layers_to_viewer(layers)
        self._layer_toggle.set_layers(loaded_entries)

        # Update state
        self._main_pipeline = pipeline_name
        self._main_stem = stem
        self._split_pipeline = None
        self._split_stem = None
        self._split_steps = []

        # Disable grid for single view
        self._viewer.grid.enabled = False

        # Update config bar
        swept_names = get_swept_param_names(
            self._param_explorer.swept_params,
        )
        config = self._data.pipeline_configs.get(pipeline_name)
        if config:
            self._config_bar.set_main_pipeline(config, swept_names)
        self._config_bar.clear_split()

        # Configure step sliders
        self._main_steps = self._get_steps(stem, pipeline_name)
        self._main_resolution_index = build_layer_resolution_index(self._main_steps)
        if self._main_steps:
            self._step_slider.set_main_steps(self._main_steps)
        else:
            self._step_slider.clear_main()
        self._step_slider.clear_split()

    def _load_split_view(self, pipeline_name: str, stem: str) -> None:
        """Load a pipeline as the split (right) view.

        Args:
            pipeline_name: Pipeline name to load.
            stem: Image stem to load.
        """
        # Clear only split layers
        self._clear_split_layers()

        # Resolve HDF5 path
        hdf5_entry = (
            self._data.by_image.get(stem, {}).get(pipeline_name)
        )
        if hdf5_entry is None:
            logger.warning(
                "No HDF5 for stem=%r pipeline=%r", stem, pipeline_name,
            )
            return

        # Load split layers
        layers = self._load_hdf5_layers(
            str(hdf5_entry.path), pipeline_name, stem, prefix="split",
        )
        loaded_entries = self._add_layers_to_viewer(layers)
        self._layer_toggle.add_layers(loaded_entries)

        # Update state
        self._split_pipeline = pipeline_name
        self._split_stem = stem

        # Enable grid for split view (4 components per column)
        self._viewer.grid.enabled = True
        self._viewer.grid.stride = 4
        self._viewer.grid.shape = (-1, -1)

        # Update config bar
        swept_names = get_swept_param_names(
            self._param_explorer.swept_params,
        )
        config = self._data.pipeline_configs.get(pipeline_name)
        if config:
            self._config_bar.set_split_pipeline(config, swept_names)

        # Configure split step slider
        self._split_steps = self._get_steps(stem, pipeline_name)
        self._split_resolution_index = build_layer_resolution_index(self._split_steps)
        if self._split_steps:
            self._step_slider.set_split_steps(self._split_steps)
        else:
            self._step_slider.clear_split()

    # ------------------------------------------------------------------
    # Layer management helpers
    # ------------------------------------------------------------------

    def _add_layers_to_viewer(self, layers: list[dict]) -> list[dict]:
        """Add loaded layer dicts to the napari viewer.

        Args:
            layers: List of layer dicts from :meth:`_load_hdf5_layers`.

        Returns:
            List of entry dicts for the layer toggle widget.
        """
        loaded_entries: list[dict] = []
        for layer in layers:
            if layer["is_labels"]:
                self._viewer.add_labels(
                    layer["data"].astype(np.intp), name=layer["name"],
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
        return loaded_entries

    def _replace_main_layers(
        self, h5_path: str, pipeline: str, stem: str,
    ) -> None:
        """Clear main layers and reload from an HDF5 path.

        Args:
            h5_path: Path to the HDF5 file.
            pipeline: Pipeline name.
            stem: Image stem.
        """
        self._clear_main_layers()
        layers = self._load_hdf5_layers(h5_path, pipeline, stem, prefix="main")
        loaded = self._add_layers_to_viewer(layers)
        self._layer_toggle.set_layers(
            loaded + self._get_split_layer_entries(),
        )

    def _replace_split_layers(
        self, h5_path: str, pipeline: str, stem: str,
    ) -> None:
        """Clear split layers and reload from an HDF5 path.

        Args:
            h5_path: Path to the HDF5 file.
            pipeline: Pipeline name.
            stem: Image stem.
        """
        self._clear_split_layers()
        layers = self._load_hdf5_layers(h5_path, pipeline, stem, prefix="split")
        loaded = self._add_layers_to_viewer(layers)
        self._layer_toggle.add_layers(loaded)

    def _replace_main_layers_composited(
        self, resolved: ResolvedLayerSources, pipeline: str, stem: str,
    ) -> None:
        """Clear main layers and reload from composited sources.

        Args:
            resolved: Resolved layer sources.
            pipeline: Pipeline name.
            stem: Image stem.
        """
        self._clear_main_layers()
        layers = self._load_composited_layers(resolved, pipeline, stem, prefix="main")
        loaded = self._add_layers_to_viewer(layers)
        self._layer_toggle.set_layers(
            loaded + self._get_split_layer_entries(),
        )

    def _replace_split_layers_composited(
        self, resolved: ResolvedLayerSources, pipeline: str, stem: str,
    ) -> None:
        """Clear split layers and reload from composited sources.

        Args:
            resolved: Resolved layer sources.
            pipeline: Pipeline name.
            stem: Image stem.
        """
        self._clear_split_layers()
        layers = self._load_composited_layers(resolved, pipeline, stem, prefix="split")
        loaded = self._add_layers_to_viewer(layers)
        self._layer_toggle.add_layers(loaded)

    def _get_split_layer_entries(self) -> list[dict]:
        """Return entry dicts for currently loaded split layers."""
        entries: list[dict] = []
        for name in self._current_layer_names:
            if name.startswith("split/"):
                parts = name.split("/")
                if len(parts) >= 4:
                    entries.append(
                        {
                            "pipeline": parts[1],
                            "component": parts[2],
                            "image_stem": parts[3],
                        }
                    )
        return entries

    def _load_hdf5_layers(
        self,
        h5_path: str,
        pipeline: str,
        image_stem: str,
        prefix: str = "",
    ) -> list[dict]:
        """Load HDF5 via Image.load_hdf5(), extract layers, free Image.

        Args:
            h5_path: Absolute path to the ``.h5`` file.
            pipeline: Pipeline name for layer naming.
            image_stem: Image stem for layer naming.
            prefix: Optional prefix for layer names (``"main"`` or
                ``"split"``).  When non-empty, layer names become
                ``prefix/pipeline/component/stem``.

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

        name_prefix = f"{prefix}/{pipeline}" if prefix else pipeline

        layers: list[dict] = []

        # RGB (check availability)
        if not image.rgb.isempty():
            layers.append(
                {
                    "name": f"{name_prefix}/rgb/{image_stem}",
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
                "name": f"{name_prefix}/gray/{image_stem}",
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
                "name": f"{name_prefix}/detect_mat/{image_stem}",
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
                "name": f"{name_prefix}/objmap/{image_stem}",
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

    def _load_composited_layers(
        self,
        resolved: ResolvedLayerSources,
        pipeline: str,
        image_stem: str,
        prefix: str = "",
    ) -> list[dict]:
        """Load layers from potentially multiple HDF5 source files.

        Args:
            resolved: Maps each layer name to its source HDF5 file.
            pipeline: Pipeline name for layer naming.
            image_stem: Image stem for layer naming.
            prefix: Optional prefix (``"main"`` or ``"split"``).

        Returns:
            List of layer dicts, same format as :meth:`_load_hdf5_layers`.
        """
        import h5py

        name_prefix = f"{prefix}/{pipeline}" if prefix else pipeline

        # Collect unique file paths and which layers to read from each
        file_layers: dict[Path, list[str]] = {}
        for layer_name in _LAYER_ORDER:
            source = getattr(resolved, layer_name, None)
            if source is not None:
                file_layers.setdefault(source, []).append(layer_name)

        # Read datasets from each unique file
        loaded: dict[str, np.ndarray] = {}
        for path, layer_names in file_layers.items():
            try:
                with h5py.File(path, "r") as f:
                    for ln in layer_names:
                        if ln in f:
                            loaded[ln] = f[ln][()]
            except Exception as exc:
                logger.warning(
                    "Failed to read %s from %s: %s",
                    layer_names, path, exc,
                )

        # Assemble layer dicts in standard order
        layers: list[dict] = []
        for component in _LAYER_ORDER:
            if component not in loaded:
                continue
            layers.append(
                {
                    "name": f"{name_prefix}/{component}/{image_stem}",
                    "data": loaded[component].copy(),
                    "component": component,
                    "pipeline": pipeline,
                    "image_stem": image_stem,
                    "is_labels": component == "objmap",
                }
            )

        return layers

    def _clear_all_layers(self) -> None:
        """Remove all tracked layers (main + split)."""
        for name in self._current_layer_names:
            if name in self._viewer.layers:
                self._viewer.layers.remove(name)
        self._current_layer_names = []
        if hasattr(self, "_layer_toggle"):
            self._layer_toggle.clear()

    def _clear_main_layers(self) -> None:
        """Remove only layers with ``main/`` prefix."""
        remaining: List[str] = []
        for name in self._current_layer_names:
            if name.startswith("main/"):
                if name in self._viewer.layers:
                    self._viewer.layers.remove(name)
            else:
                remaining.append(name)
        self._current_layer_names = remaining

    def _clear_split_layers(self) -> None:
        """Remove only layers with ``split/`` prefix."""
        remaining: List[str] = []
        for name in self._current_layer_names:
            if name.startswith("split/"):
                if name in self._viewer.layers:
                    self._viewer.layers.remove(name)
            else:
                remaining.append(name)
        self._current_layer_names = remaining

    # ------------------------------------------------------------------
    # Step resolution helpers
    # ------------------------------------------------------------------

    def _resolve_step_sources(
        self,
        pipeline: str,
        stem: str,
        steps: List[IntermediateStep],
        step_index: int,
        resolution_index: Optional[dict] = None,
    ) -> Optional[ResolvedLayerSources | str]:
        """Resolve a step index to layer sources.

        Args:
            pipeline: Pipeline name.
            stem: Image stem.
            steps: List of intermediate steps.
            step_index: Index into steps, or ``len(steps)`` for final.
            resolution_index: Precomputed resolution index.

        Returns:
            :class:`ResolvedLayerSources` for composited loading,
            a path string for single-file loading, or ``None``.
        """
        if step_index >= len(steps):
            # Final output
            hdf5_entry = (
                self._data.by_image.get(stem, {}).get(pipeline)
            )
            if hdf5_entry is None:
                return None
            return str(hdf5_entry.path)

        # Use resolution index if available
        if resolution_index and step_index in resolution_index:
            resolved = resolution_index[step_index]
            # If all layers are resolved and point to the same file,
            # use simple single-file path (avoids composited loading)
            sources = [
                getattr(resolved, ln) for ln in _LAYER_ORDER
            ]
            non_none = {s for s in sources if s is not None}
            if len(non_none) == 1 and all(s is not None for s in sources):
                return str(non_none.pop())
            return resolved

        # Fallback: direct file path (backward compat with old snapshots)
        return str(steps[step_index].h5_path)

    def _get_steps(
        self, stem: str, pipeline: str,
    ) -> List[IntermediateStep]:
        """Look up intermediate steps for a stem/pipeline pair.

        Args:
            stem: Image stem.
            pipeline: Pipeline name.

        Returns:
            List of intermediate steps, or empty list.
        """
        return list(
            self._data.intermediates
            .get(stem, {})
            .get(pipeline, [])
        )

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _organize_dock_layout(self, dock_fb, dock_pe, dock_layers) -> None:
        """Arrange dock widgets into two right-side tab groups.

        Top-right tabs: File Browser + Parameter Explorer.
        Bottom-right tabs: Layers + native layer list + native layer controls.
        """
        try:
            qt_window = self._viewer.window._qt_window
            layer_list = (
                self._viewer.window._qt_viewer.dockLayerList
            )
            layer_controls = (
                self._viewer.window._qt_viewer.dockLayerControls
            )

            # Top-right tabs: File Browser + Parameter Explorer
            qt_window.tabifyDockWidget(dock_fb, dock_pe)
            dock_fb.show()
            dock_fb.raise_()

            # Bottom-right tabs: Layers + native layer list + controls
            qt_window.tabifyDockWidget(dock_layers, layer_list)
            qt_window.tabifyDockWidget(dock_layers, layer_controls)
            dock_layers.show()
            dock_layers.raise_()
        except Exception as exc:
            logger.debug(
                "Could not organize dock layout: %s", exc,
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
