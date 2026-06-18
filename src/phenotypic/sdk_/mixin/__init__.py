"""Mixin classes for PhenoTypic operations.

This subpackage provides reusable mixin classes that add specific capabilities
to ImageOperation subclasses and other PhenoTypic components.

Available mixins:
- FootprintMixin: Generate morphological footprints (structuring elements)
- GridInferenceMixin: Infer grid structure from binary masks using peak detection
- LazyWidgetMixin: Generate interactive Jupyter widgets for parameter tuning
- ClipControlMixin: Control output clipping behavior in composite operations
- PointPickerMixin: Marker + shared plumbing for operations that take user-picked
  ``(y, x)`` coordinates as a primary parameter
- _GATSupportMixin: Optional Generalized Anscombe Transform variance
  stabilization for noise-driven enhancers and correctors
"""

from ._footprint_mixin import FootprintMixin
from ._gat_support_mixin import _GATSupportMixin
from ._grid_inference_mixin import GridInferenceMixin
from ._lazy_widget_mixin import LazyWidgetMixin
from ._clip_control_mixin import ClipControlMixin
from ._point_picker_mixin import PointPickerMixin

__all__ = [
    "FootprintMixin",
    "GridInferenceMixin",
    "LazyWidgetMixin",
    "ClipControlMixin",
    "PointPickerMixin",
    "_GATSupportMixin",
]
