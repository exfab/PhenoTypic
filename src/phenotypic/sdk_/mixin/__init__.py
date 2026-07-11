"""Mixin classes for PhenoTypic operations.

This subpackage provides reusable mixin classes that add specific capabilities
to ImageOperation subclasses and other PhenoTypic components.

Available mixins:
- FootprintMixin: Generate morphological footprints (structuring elements)
- GridInferenceMixin: Infer grid structure from binary masks using peak detection
- LazyWidgetMixin: Generate interactive Jupyter widgets for parameter tuning
- NormControlMixin: Disable output normalization on inner operations of a composite
- InputLayerMixin: Append an ``input_layer`` field selecting ``detect_mat`` or ``rgb``
- NormalizedOutputMixin: Append a ``norm`` output-range policy field to an operation
- PointPickerMixin: Marker + shared plumbing for operations that take user-picked
  ``(y, x)`` coordinates as a primary parameter
- _GATSupportMixin: Optional Generalized Anscombe Transform variance
  stabilization for noise-driven enhancers and correctors
"""

from ._footprint_mixin import FootprintMixin
from ._gat_support_mixin import _GATSupportMixin
from ._grid_inference_mixin import GridInferenceMixin
from ._input_layer_mixin import InputLayerMixin
from ._lazy_widget_mixin import LazyWidgetMixin
from ._norm_control_mixin import NormControlMixin
from ._normalized_output_mixin import NormalizedOutputMixin
from ._point_picker_mixin import PointPickerMixin

__all__ = [
    "FootprintMixin",
    "GridInferenceMixin",
    "InputLayerMixin",
    "LazyWidgetMixin",
    "NormControlMixin",
    "NormalizedOutputMixin",
    "PointPickerMixin",
    "_GATSupportMixin",
]
