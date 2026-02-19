"""Read-only widget displaying the selected pipeline's configuration."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel, QTextBrowser

from ._sweep_data_model import PipelineConfig

logger = logging.getLogger(__name__)


class PipelineInfoWidget(QWidget):
    """Displays pipeline operations and parameters from the manifest.

    Args:
        configs: Mapping of pipeline name to :class:`PipelineConfig`.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        configs: Dict[str, PipelineConfig],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._configs = configs

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._header = QLabel("Pipeline Info")
        self._header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(self._header)

        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(False)
        layout.addWidget(self._browser)

    def set_pipeline(self, pipeline_name: str) -> None:
        """Update the display for *pipeline_name*.

        Args:
            pipeline_name: Key into the configs dict.
        """
        config = self._configs.get(pipeline_name)
        if config is None:
            logger.warning(
                "No config found for pipeline %r",
                pipeline_name,
            )
            self._browser.setHtml(
                f"<i>No config found for '{pipeline_name}'</i>"
            )
            return
        logger.debug(
            "Displaying config for pipeline %r", pipeline_name,
        )
        self._header.setText(f"Pipeline: {pipeline_name}")
        self._browser.setHtml(self._format_config(config))

    def clear(self) -> None:
        """Reset to empty state."""
        self._header.setText("Pipeline Info")
        self._browser.clear()

    # ------------------------------------------------------------------

    @staticmethod
    def _format_config(config: PipelineConfig) -> str:
        """Render a :class:`PipelineConfig` as HTML for the text browser."""
        parts = [
            f"<b>Config group:</b> {config.config_group}<br>",
            "<b>Operations:</b><ol>",
        ]
        for op in config.operations:
            params_str = ", ".join(
                f"{k}={v}" for k, v in op.get("params", {}).items()
            )
            parts.append(
                f"<li><code>{op['class']}</code>"
                f"{' — ' + params_str if params_str else ''}</li>"
            )
        parts.append("</ol>")

        if config.measurements:
            parts.append("<b>Measurements:</b><ol>")
            for meas in config.measurements:
                params_str = ", ".join(
                    f"{k}={v}" for k, v in meas.get("params", {}).items()
                )
                parts.append(
                    f"<li><code>{meas['class']}</code>"
                    f"{' — ' + params_str if params_str else ''}</li>"
                )
            parts.append("</ol>")

        return "".join(parts)
