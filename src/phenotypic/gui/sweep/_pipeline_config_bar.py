"""Bottom bar widget displaying pipeline configurations for main and split views.

Shows the full pipeline configuration for each active view with all
operations listed as a numbered list.  Swept parameters are **bolded**
while fixed parameters appear in normal weight.  Each operation's
parameters are rendered as a bulleted sub-list.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Set, Tuple

from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ._sweep_data_model import PipelineConfig


class PipelineConfigBar(QWidget):
    """Horizontal bar showing pipeline configs for the main and split views.

    The bar contains two side-by-side group boxes — one for the main
    view and one for the split view.  The split panel is hidden until
    :meth:`set_split_pipeline` is called.

    Args:
        parent: Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        # --- Main view panel ---
        self._main_group = QGroupBox("Main View")
        main_group_layout = QVBoxLayout(self._main_group)
        main_group_layout.setContentsMargins(4, 4, 4, 4)
        self._main_browser = QTextBrowser()
        self._main_browser.setOpenExternalLinks(False)
        main_group_layout.addWidget(self._main_browser)
        layout.addWidget(self._main_group)

        # --- Split view panel ---
        self._split_group = QGroupBox("Split View")
        split_group_layout = QVBoxLayout(self._split_group)
        split_group_layout.setContentsMargins(4, 4, 4, 4)
        self._split_browser = QTextBrowser()
        self._split_browser.setOpenExternalLinks(False)
        split_group_layout.addWidget(self._split_browser)
        layout.addWidget(self._split_group)

        # Start with both panels hidden
        self._main_group.hide()
        self._split_group.hide()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_main_pipeline(
        self,
        config: PipelineConfig,
        swept_param_names: Set[Tuple[str, str]],
    ) -> None:
        """Update the main view panel with pipeline configuration.

        Args:
            config: Pipeline configuration to display.
            swept_param_names: Set of ``(operation_name, param_name)``
                tuples identifying parameters that were swept.  Matching
                parameters are rendered in bold.
        """
        self._main_browser.setHtml(
            self._format_config(config, swept_param_names),
        )
        self._main_group.show()

    def set_split_pipeline(
        self,
        config: PipelineConfig,
        swept_param_names: Set[Tuple[str, str]],
    ) -> None:
        """Update and show the split view panel with pipeline configuration.

        Args:
            config: Pipeline configuration to display.
            swept_param_names: Set of ``(operation_name, param_name)``
                tuples identifying parameters that were swept.  Matching
                parameters are rendered in bold.
        """
        self._split_browser.setHtml(
            self._format_config(config, swept_param_names),
        )
        self._split_group.show()

    def clear_split(self) -> None:
        """Hide the split view panel and clear its content."""
        self._split_browser.clear()
        self._split_group.hide()

    def clear(self) -> None:
        """Hide both panels and clear all content."""
        self._main_browser.clear()
        self._main_group.hide()
        self._split_browser.clear()
        self._split_group.hide()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_config(
        config: PipelineConfig,
        swept_param_names: Set[Tuple[str, str]],
    ) -> str:
        """Render a pipeline configuration as HTML.

        Args:
            config: Pipeline configuration to format.
            swept_param_names: Set of ``(operation_name, param_name)``
                tuples.  Parameters matching an entry in this set are
                wrapped in ``<b>`` tags.

        Returns:
            HTML string with a numbered operation list and bulleted
            parameter sub-lists.
        """
        parts: list[str] = ["<ol>"]

        for op in config.operations:
            op_name = op.get("name", "")
            op_class = html.escape(str(op.get("class", "Unknown")))
            params = op.get("params", {})

            parts.append(f"<li><code>{op_class}</code>")

            if params:
                parts.append("<ul>")
                for param_key, param_value in params.items():
                    escaped_key = html.escape(str(param_key))
                    escaped_value = html.escape(str(param_value))
                    line = f"{escaped_key} = {escaped_value}"

                    if (op_name, param_key) in swept_param_names:
                        parts.append(f"<li><b>{line}</b></li>")
                    else:
                        parts.append(f"<li>{line}</li>")
                parts.append("</ul>")

            parts.append("</li>")

        parts.append("</ol>")

        if config.measurements:
            parts.append("<b>Measurements:</b><ol>")
            for meas in config.measurements:
                meas_name = meas.get("name", "")
                meas_class = html.escape(
                    str(meas.get("class", "Unknown")),
                )
                params = meas.get("params", {})

                parts.append(f"<li><code>{meas_class}</code>")

                if params:
                    parts.append("<ul>")
                    for param_key, param_value in params.items():
                        escaped_key = html.escape(str(param_key))
                        escaped_value = html.escape(str(param_value))
                        line = f"{escaped_key} = {escaped_value}"

                        if (meas_name, param_key) in swept_param_names:
                            parts.append(f"<li><b>{line}</b></li>")
                        else:
                            parts.append(f"<li>{line}</li>")
                    parts.append("</ul>")

                parts.append("</li>")
            parts.append("</ol>")

        return "".join(parts)
