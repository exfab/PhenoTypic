"""Public measurement schema for the PhenoTypic library.

This subpackage is the canonical, public home for PhenoTypic's measurement
naming conventions: the :class:`MeasurementInfo` base class plus every
``MeasurementInfo`` subclass that names a column in an output DataFrame. Each
enum lives in its own module (``_<name>.py``) and is re-exported here, so the
public import surface stays stable:

    from phenotypic.schema import MeasurementInfo, BBOX, GRID, SHAPE

It also hosts the metadata vocabulary: ``IMAGE`` (framework-populated image
bookkeeping) and eight semantic owners (``GENETIC``, ``SAMPLE``, ``PLATE``,
``CONDITION``, ``CULTURE``, ``ACQUISITION``, ``EXPERIMENT``, and ``STUDY``)
that standardize ``Metadata_*`` columns for the ``--metadata`` join and
``post/`` operations.
"""

import sys
import warnings

from ._base._measurement_info import (
    Entry,
    MeasurementInfo,
    parse_qualified_header,
    qualified_header,
)
from ._base._rembi import (
    REMBI_MODULE as REMBI_MODULE,
    header_to_module as header_to_module,
)
from ._base._categories import CATEGORIES, CategoryEntry
from ._base._tiers import (
    DerivedMeasure as DerivedMeasure,
    DescriptiveTrait as DescriptiveTrait,
    DirectPhenotype as DirectPhenotype,
    DiscriminativeFeature as DiscriminativeFeature,
    IdentityInfo as IdentityInfo,
    MetadataInfo as MetadataInfo,
    PrimaryMeasure as PrimaryMeasure,
    QualityInfo as QualityInfo,
)
from ._metadata._image import IMAGE
from ._metadata._experimental_tags import (
    ACQUISITION,
    CONDITION,
    CULTURE,
    EXPERIMENT,
    GENETIC,
    PLATE,
    SAMPLE,
    STUDY,
)

from ._measure._bbox import BBOX
from ._measure._color_composition import ColorComposition
from ._measure._color_hsv import ColorHSV
from ._measure._color_lab import ColorLab
from ._measure._color_xy import Colorxy
from ._measure._color_xyz import ColorXYZ
from ._analysis._edge_correction import EDGE_CORRECTION
from ._shared._grid import GRID
from ._measure._grid_linreg_stats import GRID_LINREG_STATS
from ._measure._neighbor_dist import NEIGHBOR_DIST
from ._measure._grid_spread import GRID_SPREAD
from ._analysis._models._linear_cap_and_lag_model import LINEAR_CAP_AND_LAG_MODEL
from ._measure._intensity import INTENSITY
from ._analysis._models._linear_lag_model import LINEAR_LAG_MODEL
from ._analysis._models._log_growth_model import LOG_GROWTH_MODEL
from ._analysis._models._model_metrics import MODEL_METRICS
from ._shared._object import OBJECT
from ._shared._curation import CURATION
from ._shared._error_category import ErrorCategory
from ._shared._metadata_match import METADATA_MATCH
from ._analysis._qc._quality_check import QUALITY_CHECK
from ._analysis._qc._quality_count import QUALITY_COUNT
from ._analysis._qc._quality_icc import QUALITY_ICC
from ._analysis._qc._quality_mad import QUALITY_MAD
from ._analysis._qc._quality_occupancy import QUALITY_OCCUPANCY
from ._analysis._qc._quality_se import QUALITY_SE
from ._analysis._qc._quality_tukey import QUALITY_TUKEY
from ._analysis._qc._quality_zmax import QUALITY_ZMAX
from ._shared._radial_expansion import RADIAL_EXPANSION
from ._measure._orientation_zones import (
    ORIENTATION_ZONE_DIAGNOSTIC,
    ORIENTATION_ZONE_PRIMARY,
    ORIENTATION_ZONES,
)
from ._measure._shape import SHAPE
from ._measure._size import SIZE
from ._measure._symmetric_zones import SYMMETRIC_ZONES
from ._measure._texture import TEXTURE

__all__ = [
    "Entry",
    "CATEGORIES",
    "CategoryEntry",
    "MeasurementInfo",
    "MetadataInfo",
    "parse_qualified_header",
    "qualified_header",
    "REMBI_MODULE",
    "header_to_module",
    "METADATA_MATCH",
    "IMAGE",
    "GENETIC",
    "SAMPLE",
    "PLATE",
    "CONDITION",
    "CULTURE",
    "EXPERIMENT",
    "STUDY",
    "ACQUISITION",
    "BBOX",
    "ColorComposition",
    "ColorHSV",
    "ColorLab",
    "Colorxy",
    "ColorXYZ",
    "CURATION",
    "LINEAR_CAP_AND_LAG_MODEL",
    "EDGE_CORRECTION",
    "ErrorCategory",
    "GRID",
    "GRID_LINREG_STATS",
    "NEIGHBOR_DIST",
    "GRID_SPREAD",
    "INTENSITY",
    "LINEAR_LAG_MODEL",
    "LOG_GROWTH_MODEL",
    "MODEL_METRICS",
    "OBJECT",
    "QUALITY_CHECK",
    "QUALITY_COUNT",
    "QUALITY_ICC",
    "QUALITY_MAD",
    "QUALITY_OCCUPANCY",
    "QUALITY_SE",
    "QUALITY_TUKEY",
    "QUALITY_ZMAX",
    "RADIAL_EXPANSION",
    "SHAPE",
    "SIZE",
    "ORIENTATION_ZONES",
    "ORIENTATION_ZONE_DIAGNOSTIC",
    "ORIENTATION_ZONE_PRIMARY",
    "SYMMETRIC_ZONES",
    "TEXTURE",
]

_LEGACY_METADATA_NAMES = {
    "METADATA": IMAGE,
    "GENETIC_METADATA": GENETIC,
    "SAMPLE_METADATA": SAMPLE,
    "PLATE_METADATA": PLATE,
    "CONDITION_METADATA": CONDITION,
    "CULTURE_METADATA": CULTURE,
    "ACQUISITION_METADATA": ACQUISITION,
    "EXPERIMENT_METADATA": EXPERIMENT,
    "STUDY_METADATA": STUDY,
}


def __getattr__(name: str):
    """Resolve one-release compatibility names for metadata enum owners."""
    value = _LEGACY_METADATA_NAMES.get(name)
    if value is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    caller = sys._getframe(1)
    is_fromlist_probe = (
        caller.f_code.co_name == "_handle_fromlist"
        and caller.f_globals.get("__name__") == "importlib._bootstrap"
    )
    if not is_fromlist_probe:
        warnings.warn(
            f"phenotypic.schema.{name} is deprecated; use {value.__name__} instead",
            DeprecationWarning,
            stacklevel=2,
        )
    return value
