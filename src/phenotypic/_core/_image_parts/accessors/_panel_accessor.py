"""Panel accessor composing all dashboard classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from phenotypic._core._image_parts.panel_accessor import DetectModeDashboard

if TYPE_CHECKING:
    from phenotypic import Image


class PanelAccessor(DetectModeDashboard):
    """Provides interactive Panel dashboards for image exploration.

    This accessor composes all dashboard classes (currently
    ``DetectModeDashboard``) into a single entry point accessible via
    ``image.panel``.  New dashboards are added by extending the
    inheritance chain.

    Examples:
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> dashboard = image.panel.detect_modes()
    """

    def __init__(self, root_image: Image) -> None:
        DetectModeDashboard.__init__(self, root_image)


__all__ = ("PanelAccessor",)
