"""Plotly helper functions for interactive image visualization.

Backward-compatible shim — real implementations now live on
:class:`~phenotypic._core._image_parts.accessor_abstracts._image_accessor_base_parents._accessor_dash_handler.AccessorDashHandler`.
"""

from phenotypic._core._image_parts.accessor_abstracts._image_accessor_base_parents._accessor_dash_handler import (  # noqa: E501
    AccessorDashHandler as _Dash,
    PLOTLY_AVAILABLE,
    _MPL_TO_PLOTLY,
)

PLOTLY_CONFIG = _Dash._PLOTLY_CONFIG

_require_plotly = _Dash._require_plotly
mpl_cmap_to_plotly = _Dash._mpl_cmap_to_plotly
plotly_imshow = _Dash._plotly_imshow
add_plotly_gridlines = _Dash._add_plotly_gridlines
add_plotly_section_boxes = _Dash._add_plotly_section_boxes
add_plotly_obj_labels = _Dash._add_plotly_obj_labels

__all__ = [
    "PLOTLY_AVAILABLE",
    "PLOTLY_CONFIG",
    "_MPL_TO_PLOTLY",
    "_require_plotly",
    "mpl_cmap_to_plotly",
    "plotly_imshow",
    "add_plotly_gridlines",
    "add_plotly_section_boxes",
    "add_plotly_obj_labels",
]
