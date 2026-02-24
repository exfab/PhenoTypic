"""Step slider widget for scrubbing through intermediate pipeline states."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget

if TYPE_CHECKING:
    from ._sweep_data_model import IntermediateStep


class StepSliderWidget(QWidget):
    """Horizontal slider for selecting intermediate pipeline steps.

    The slider range covers ``0..len(steps)`` inclusive.  Positions
    ``0`` through ``len(steps)-1`` correspond to intermediate snapshots;
    the last position (``len(steps)``) represents the final pipeline
    output.

    Signals:
        step_changed(int): Emitted when the slider moves.  The value is
            the step index (``0`` … ``len-1`` for intermediates, or
            ``len`` for "final").
    """

    step_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._steps: List[IntermediateStep] = []

        # --- Layout ---
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self._prefix_label = QLabel("Step:")
        layout.addWidget(self._prefix_label)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(1)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self._slider, stretch=1)

        self._step_label = QLabel("(final)")
        self._step_label.setMinimumWidth(160)
        layout.addWidget(self._step_label)

        self.setVisible(False)

    def set_steps(self, steps: List[IntermediateStep]) -> None:
        """Configure the slider for a list of intermediate steps.

        Args:
            steps: Sorted list of intermediate steps.  The slider adds
                one extra position at the end for "final".
        """
        self._steps = list(steps)
        n = len(self._steps)
        self._slider.setMaximum(n)  # 0..n-1 = intermediates, n = final
        self._slider.setValue(n)     # Start at final
        self._update_label(n)
        self.setVisible(True)

    def clear(self) -> None:
        """Reset and hide the widget."""
        self._steps = []
        self._slider.setMaximum(0)
        self._slider.setValue(0)
        self._step_label.setText("(final)")
        self.setVisible(False)

    def _on_value_changed(self, value: int) -> None:
        """Handle slider movement."""
        self._update_label(value)
        self.step_changed.emit(value)

    def _update_label(self, value: int) -> None:
        """Update the step name label based on slider position."""
        if not self._steps or value >= len(self._steps):
            self._step_label.setText("(final)")
        else:
            step = self._steps[value]
            self._step_label.setText(f"[{step.index:02d}] {step.operation_name}")
