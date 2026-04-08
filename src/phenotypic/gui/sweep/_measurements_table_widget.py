"""Docked table widget showing measurement CSV data for the selected image.

Not currently used by the sweep viewer — loading and displaying
measurement CSVs on every stem selection introduced noticeable GUI
latency.  Kept here so it can be re-enabled once the performance issue
is resolved.
"""

from __future__ import annotations

import logging
from typing import Optional

from qtpy.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ._sweep_data_model import SweepOutputData

logger = logging.getLogger(__name__)


class MeasurementsTableWidget(QWidget):
    """Table showing per-image measurement CSV for the current selection.

    Args:
        data: The indexed sweep output data.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        data: SweepOutputData,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._data = data

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._header = QLabel("Measurements")
        self._header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self._header)

        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents,
        )
        layout.addWidget(self._table)

        self._placeholder = QLabel("No measurements available")
        self._placeholder.setStyleSheet("color: gray; padding: 8px;")
        layout.addWidget(self._placeholder)
        self._placeholder.hide()

    def set_selection(self, pipeline_name: str, image_stem: str) -> None:
        """Load the measurements CSV for *pipeline_name* / *image_stem*.

        Args:
            pipeline_name: Pipeline directory name.
            image_stem: Image filename stem.
        """
        self._header.setText(f"{pipeline_name} / {image_stem}")

        csv_path = (
            self._data.root_dir
            / "results"
            / pipeline_name
            / "measurements"
            / f"{image_stem}.csv"
        )
        logger.debug("Looking for measurements CSV: %s", csv_path)

        if not csv_path.exists():
            logger.debug(
                "Measurements CSV not found: %s", csv_path,
            )
            self._show_placeholder()
            return

        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", csv_path, exc)
            self._show_placeholder()
            return

        self._populate_table(df)
        logger.debug(
            "Loaded measurements: %d rows, %d columns",
            len(df), len(df.columns),
        )

    def clear(self) -> None:
        """Reset to empty state."""
        self._header.setText("Measurements")
        self._table.clear()
        self._table.setRowCount(0)
        self._table.setColumnCount(0)
        self._show_placeholder()

    # ------------------------------------------------------------------

    def _populate_table(self, df) -> None:
        """Fill the QTableWidget from a pandas DataFrame."""
        self._placeholder.hide()
        self._table.show()

        cols = list(df.columns)
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setRowCount(len(df))

        for row_idx in range(len(df)):
            for col_idx, col_name in enumerate(cols):
                value = df.iloc[row_idx, col_idx]
                item = QTableWidgetItem(str(value))
                self._table.setItem(row_idx, col_idx, item)

    def _show_placeholder(self) -> None:
        self._table.hide()
        self._table.clear()
        self._table.setRowCount(0)
        self._table.setColumnCount(0)
        self._placeholder.show()
