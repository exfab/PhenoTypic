"""Parameter explorer widget for swept pipeline parameters.

Displays interactive dropdown controls for parameters that vary across
sweep pipeline configurations.  Selecting parameter values resolves to a
pipeline name, which can be loaded via the View or View Split buttons.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ._swept_param_analysis import (
    SweptParameter,
    _build_param_index,
    build_param_to_pipeline_map,
    compute_structural_signature,
    detect_swept_parameters,
    group_configs_by_structure,
    resolve_pipeline_name,
)

if TYPE_CHECKING:
    from ._sweep_data_model import PipelineConfig

logger = logging.getLogger(__name__)


class ParameterExplorerWidget(QWidget):
    """Widget for exploring swept parameters across pipeline configurations.

    Analyzes pipeline configs to detect which parameters were swept,
    then presents dropdown controls grouped by operation class.  Changing
    any control resolves the current selection to a matching pipeline
    name shown in the status label.

    Signals:
        view_requested(str): Emitted with the pipeline name when the
            "View" button is clicked.
        view_split_requested(str): Emitted with the pipeline name when
            the "View Split" button is clicked.

    Args:
        parent: Optional parent widget.
    """

    view_requested = Signal(str)
    view_split_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._configs: Dict[str, PipelineConfig] = {}
        self._structural_groups: Dict[tuple, Dict[str, PipelineConfig]] = {}
        self._active_group_key: Optional[tuple] = None
        self._swept_params: List[SweptParameter] = []
        self._lookup: Dict[tuple, str] = {}
        self._controls: Dict[Tuple[str, str], QComboBox] = {}
        self._param_map: Dict[Tuple[str, str], SweptParameter] = {}

        # --- Main layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Parameter Explorer")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        # --- Scroll area for parameter controls ---
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_content = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setContentsMargins(2, 2, 2, 2)
        self._scroll_layout.addStretch()
        self._scroll_area.setWidget(self._scroll_content)
        layout.addWidget(self._scroll_area, stretch=1)

        # --- Button bar ---
        button_bar = QHBoxLayout()

        self._view_btn = QPushButton("View")
        self._view_btn.setEnabled(False)
        self._view_btn.clicked.connect(self._on_view_clicked)
        button_bar.addWidget(self._view_btn)

        self._view_split_btn = QPushButton("View Split")
        self._view_split_btn.setEnabled(False)
        self._view_split_btn.clicked.connect(self._on_view_split_clicked)
        button_bar.addWidget(self._view_split_btn)

        self._status_label = QLabel("")
        button_bar.addWidget(self._status_label)

        layout.addLayout(button_bar)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def swept_params(self) -> List[SweptParameter]:
        """Currently detected swept parameters."""
        return list(self._swept_params)

    def set_configs(self, configs: Dict[str, PipelineConfig]) -> None:
        """Analyze configs and rebuild controls for swept parameters.

        Args:
            configs: Mapping of pipeline name to :class:`PipelineConfig`.
        """
        self._configs = configs

        if not configs:
            self._structural_groups = {}
            self._active_group_key = None
            self._swept_params = []
            self._lookup = {}
            self._rebuild_controls()
            self._update_status_label()
            return

        self._structural_groups = group_configs_by_structure(configs)
        self._active_group_key = None  # Force rebuild on activate

        # Activate the first group alphabetically by first pipeline name
        first_key = min(
            self._structural_groups,
            key=lambda k: min(self._structural_groups[k]),
        )
        self._activate_group(first_key)

        logger.debug(
            "ParameterExplorer: %d structural groups, %d swept params, "
            "%d lookup entries",
            len(self._structural_groups),
            len(self._swept_params),
            len(self._lookup),
        )

    def set_pipeline(self, name: str) -> None:
        """Sync all controls to match a given pipeline's parameter values.

        Args:
            name: Pipeline name whose parameter values should be
                reflected in the controls.
        """
        if name not in self._configs:
            logger.warning(
                "set_pipeline: unknown pipeline %r", name,
            )
            return

        cfg = self._configs[name]

        # Switch structural group if needed (rebuilds controls)
        sig = compute_structural_signature(cfg)
        self._activate_group(sig)

        param_index = _build_param_index(cfg)

        # Block signals while syncing controls
        for key, widget in self._controls.items():
            widget.blockSignals(True)

        try:
            for sp in self._swept_params:
                sp_key = (sp.operation_name, sp.param_name)
                value = param_index.get(sp_key)
                ctrl = self._controls.get(sp_key)
                if ctrl is None:
                    continue

                canon = json.dumps(value, sort_keys=True, default=str)
                for i in range(ctrl.count()):
                    item_data = ctrl.itemData(
                        i, Qt.ItemDataRole.UserRole,
                    )
                    if item_data == canon:
                        ctrl.setCurrentIndex(i)
                        break
        finally:
            for key, widget in self._controls.items():
                widget.blockSignals(False)

        self._update_status_label()

    def set_view_enabled(self, enabled: bool) -> None:
        """Enable or disable both View and View Split buttons.

        Args:
            enabled: Whether the buttons should be enabled.
        """
        self._view_btn.setEnabled(enabled)
        self._view_split_btn.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Private: structural group management
    # ------------------------------------------------------------------

    def _activate_group(self, group_key: tuple) -> None:
        """Activate a structural group, rebuilding controls if needed.

        Args:
            group_key: Structural signature key from
                :func:`group_configs_by_structure`.
        """
        if group_key == self._active_group_key:
            return

        self._active_group_key = group_key
        group_configs = self._structural_groups.get(group_key, {})

        self._swept_params = detect_swept_parameters(group_configs)
        self._lookup = build_param_to_pipeline_map(
            group_configs, self._swept_params,
        )

        self._rebuild_controls()
        self._update_status_label()

    # ------------------------------------------------------------------
    # Private: control building
    # ------------------------------------------------------------------

    def _rebuild_controls(self) -> None:
        """Clear and rebuild the scroll area with controls for swept params."""
        self._controls.clear()
        self._param_map.clear()

        # Remove old widgets from scroll layout
        while self._scroll_layout.count() > 0:
            item = self._scroll_layout.takeAt(0)
            if item is not None:
                child = item.widget()
                if child is not None:
                    child.deleteLater()

        if not self._swept_params:
            self._scroll_layout.addStretch()
            return

        # Group swept params by operation_class
        groups: Dict[str, List[SweptParameter]] = defaultdict(list)
        for sp in self._swept_params:
            groups[sp.operation_class].append(sp)

        for op_class in sorted(groups.keys()):
            group_box = QGroupBox(op_class)
            form = QFormLayout(group_box)
            form.setContentsMargins(6, 6, 6, 6)

            for sp in groups[op_class]:
                key = (sp.operation_name, sp.param_name)
                self._param_map[key] = sp

                if len(groups[op_class]) > 1:
                    label_text = f"{sp.param_name} ({sp.operation_name}):"
                else:
                    label_text = f"{sp.param_name}:"

                widget = self._create_combo(sp, key)

                form.addRow(label_text, widget)

            self._scroll_layout.addWidget(group_box)

        self._scroll_layout.addStretch()

    def _create_combo(
        self,
        sp: SweptParameter,
        key: Tuple[str, str],
    ) -> QComboBox:
        """Create a combo box for a categorical parameter.

        Args:
            sp: Swept parameter metadata.
            key: ``(operation_name, param_name)`` tuple.

        Returns:
            Configured QComboBox.
        """
        combo = QComboBox()
        for val in sp.values:
            canon = json.dumps(val, sort_keys=True, default=str)
            combo.addItem(str(val), userData=canon)

        self._controls[key] = combo
        combo.currentIndexChanged.connect(self._on_param_changed)

        return combo

    # ------------------------------------------------------------------
    # Private: slots
    # ------------------------------------------------------------------

    def _on_param_changed(self, *_args: object) -> None:
        """Handle any parameter control value change."""
        self._update_status_label()

    def _update_status_label(self) -> None:
        """Resolve current selections to a pipeline name and update label."""
        if not self._swept_params:
            self._status_label.setText("")
            self._view_btn.setEnabled(False)
            self._view_split_btn.setEnabled(False)
            return

        selections = self._gather_selections()
        name = resolve_pipeline_name(
            selections, self._lookup, self._swept_params,
        )

        has_match = name is not None
        self._view_btn.setEnabled(has_match)
        self._view_split_btn.setEnabled(has_match)

        if has_match:
            self._status_label.setText(name)
        else:
            self._status_label.setText("No match")

    def _gather_selections(self) -> Dict[Tuple[str, str], object]:
        """Gather current values from all controls.

        Returns:
            Dict mapping ``(operation_name, param_name)`` to the
            currently selected raw value.
        """
        selections: Dict[Tuple[str, str], object] = {}

        for sp in self._swept_params:
            key = (sp.operation_name, sp.param_name)
            widget = self._controls.get(key)
            if widget is None:
                continue

            canon_str = widget.currentData(Qt.ItemDataRole.UserRole)
            if canon_str is not None:
                selections[key] = json.loads(canon_str)
            else:
                selections[key] = None

        return selections

    def _on_view_clicked(self) -> None:
        """Handle View button click."""
        name = self._resolve_current_pipeline()
        if name is not None:
            self.view_requested.emit(name)

    def _on_view_split_clicked(self) -> None:
        """Handle View Split button click."""
        name = self._resolve_current_pipeline()
        if name is not None:
            self.view_split_requested.emit(name)

    def _resolve_current_pipeline(self) -> Optional[str]:
        """Resolve the current control selections to a pipeline name.

        Returns:
            Pipeline name, or ``None`` if no match.
        """
        if not self._swept_params:
            return None
        selections = self._gather_selections()
        return resolve_pipeline_name(
            selections, self._lookup, self._swept_params,
        )
