"""File tree widget with dual-mode toggle for sweep output browsing."""

from __future__ import annotations

from typing import Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ._sweep_data_model import SweepOutputData


class SweepFileTreeWidget(QWidget):
    """Tree browser for sweep output with Pipeline-first / Image-first modes.

    Signals:
        image_selected: Emitted with the absolute file path string on leaf click.
        pipeline_selected: Emitted with the pipeline name on pipeline node click.
        compare_requested: Emitted with a list of ``(pipeline_name, file_path)``
            tuples when compare mode is active and a leaf is clicked.

    Args:
        data: Indexed sweep output data.
        parent: Optional parent widget.
    """

    image_selected = Signal(str)
    pipeline_selected = Signal(str)
    compare_requested = Signal(list)

    _PIPELINE_FIRST = 0
    _IMAGE_FIRST = 1

    def __init__(
        self,
        data: SweepOutputData,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._data = data

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Controls row
        controls = QHBoxLayout()
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Pipeline first", "Image first"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        controls.addWidget(self._mode_combo)

        self._compare_cb = QCheckBox("Compare")
        self._compare_cb.setToolTip(
            "Enable same-image comparison across all pipelines"
        )
        controls.addWidget(self._compare_cb)
        controls.addStretch()
        layout.addLayout(controls)

        # Tree
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree)

        self._build_tree()

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build_tree(self) -> None:
        self._tree.clear()
        if self._mode_combo.currentIndex() == self._PIPELINE_FIRST:
            self._build_pipeline_first()
        else:
            self._build_image_first()

    def _build_pipeline_first(self) -> None:
        """Pipeline -> Image -> Component."""
        for pipe_name in self._data.pipeline_names:
            pipe_item = QTreeWidgetItem([pipe_name])
            pipe_item.setData(0, Qt.UserRole, {"pipeline": pipe_name})
            self._tree.addTopLevelItem(pipe_item)

            stems = self._data.by_pipeline.get(pipe_name, {})
            for stem in sorted(stems):
                stem_item = QTreeWidgetItem([stem])
                stem_item.setData(
                    0, Qt.UserRole, {"pipeline": pipe_name, "image_stem": stem}
                )
                pipe_item.addChild(stem_item)

                for comp in sorted(stems[stem]):
                    sf = stems[stem][comp]
                    leaf = QTreeWidgetItem([comp])
                    leaf.setData(
                        0,
                        Qt.UserRole,
                        {
                            "path": str(sf.path),
                            "pipeline": pipe_name,
                            "component": comp,
                            "image_stem": stem,
                        },
                    )
                    stem_item.addChild(leaf)

    def _build_image_first(self) -> None:
        """Image -> Component -> Pipeline."""
        for stem in self._data.image_stems:
            stem_item = QTreeWidgetItem([stem])
            stem_item.setData(0, Qt.UserRole, {"image_stem": stem})
            self._tree.addTopLevelItem(stem_item)

            comps = self._data.by_image.get(stem, {})
            for comp in sorted(comps):
                comp_item = QTreeWidgetItem([comp])
                comp_item.setData(
                    0, Qt.UserRole, {"image_stem": stem, "component": comp}
                )
                stem_item.addChild(comp_item)

                for pipe_name in sorted(comps[comp]):
                    sf = comps[comp][pipe_name]
                    leaf = QTreeWidgetItem([pipe_name])
                    leaf.setData(
                        0,
                        Qt.UserRole,
                        {
                            "path": str(sf.path),
                            "pipeline": pipe_name,
                            "component": comp,
                            "image_stem": stem,
                        },
                    )
                    comp_item.addChild(leaf)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_mode_changed(self, _index: int) -> None:
        self._build_tree()

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        info = item.data(0, Qt.UserRole)
        if info is None:
            return

        # Leaf node (has "path")
        if "path" in info:
            if self._compare_cb.isChecked():
                self._emit_compare(info)
            else:
                self.image_selected.emit(info["path"])
                self.pipeline_selected.emit(info["pipeline"])
            return

        # Pipeline-level node
        if "pipeline" in info and "image_stem" not in info:
            self.pipeline_selected.emit(info["pipeline"])

    def _emit_compare(self, info: dict) -> None:
        """Collect the same image_stem + component from ALL pipelines."""
        stem = info.get("image_stem")
        comp = info.get("component")
        if not stem or not comp:
            return

        pipes = self._data.by_image.get(stem, {}).get(comp, {})
        result = [
            (pipe_name, str(sf.path))
            for pipe_name, sf in sorted(pipes.items())
        ]
        if result:
            self.compare_requested.emit(result)
