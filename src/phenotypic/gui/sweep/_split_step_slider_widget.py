"""Composite step slider widget for independent main/split step scrubbing."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ._step_slider_widget import StepSliderWidget

if TYPE_CHECKING:
    from ._sweep_data_model import IntermediateStep


class SplitStepSliderWidget(QWidget):
    """Composes two :class:`StepSliderWidget` instances for independent
    main and split step scrubbing.

    The widget starts hidden and becomes visible when either main or split
    steps are set via :meth:`set_main_steps` or :meth:`set_split_steps`.
    It hides again only when both are cleared.

    Signals:
        main_step_changed(int): Forwarded from the main
            :class:`StepSliderWidget`.
        split_step_changed(int): Forwarded from the split
            :class:`StepSliderWidget`.
    """

    main_step_changed = Signal(int)
    split_step_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Main row ---
        self._main_row = QWidget()
        main_row_layout = QHBoxLayout(self._main_row)
        main_row_layout.setContentsMargins(0, 0, 0, 0)

        self._main_label = QLabel("Main:")
        main_row_layout.addWidget(self._main_label)

        self._main_slider = StepSliderWidget()
        self._main_slider.step_changed.connect(self.main_step_changed)
        main_row_layout.addWidget(self._main_slider, stretch=1)

        self._main_row.setVisible(False)
        layout.addWidget(self._main_row)

        # --- Split row ---
        self._split_row = QWidget()
        split_row_layout = QHBoxLayout(self._split_row)
        split_row_layout.setContentsMargins(0, 0, 0, 0)

        self._split_label = QLabel("Split:")
        split_row_layout.addWidget(self._split_label)

        self._split_slider = StepSliderWidget()
        self._split_slider.step_changed.connect(self.split_step_changed)
        split_row_layout.addWidget(self._split_slider, stretch=1)

        self._split_row.setVisible(False)
        layout.addWidget(self._split_row)

        self.setVisible(False)

    def set_main_steps(self, steps: List[IntermediateStep]) -> None:
        """Configure the main slider for a list of intermediate steps.

        Args:
            steps: Sorted list of intermediate steps to display.
        """
        self._main_slider.set_steps(steps)
        self._main_row.setVisible(True)
        self._update_visibility()

    def clear_main(self) -> None:
        """Clear the main slider and hide its row."""
        self._main_slider.clear()
        self._main_row.setVisible(False)
        self._update_visibility()

    def set_split_steps(self, steps: List[IntermediateStep]) -> None:
        """Configure the split slider for a list of intermediate steps.

        Args:
            steps: Sorted list of intermediate steps to display.
        """
        self._split_slider.set_steps(steps)
        self._split_row.setVisible(True)
        self._update_visibility()

    def clear_split(self) -> None:
        """Clear the split slider and hide its row."""
        self._split_slider.clear()
        self._split_row.setVisible(False)
        self._update_visibility()

    def _update_visibility(self) -> None:
        """Show the widget when either row is active, hide when both are cleared."""
        self.setVisible(
            self._main_row.isVisible() or self._split_row.isVisible()
        )
