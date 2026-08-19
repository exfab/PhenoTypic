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

from ._measurement_info import (
    Entry,
    MeasurementInfo,
    parse_qualified_header,
    qualified_header,
)
from ._rembi import (
    REMBI_MODULE as REMBI_MODULE,
    header_to_module as header_to_module,
)
from ._tiers import (
    DerivedMeasure as DerivedMeasure,
    DescriptiveTrait as DescriptiveTrait,
    DirectPhenotype as DirectPhenotype,
    DiscriminativeFeature as DiscriminativeFeature,
    IdentityInfo as IdentityInfo,
    MetadataInfo as MetadataInfo,
    PrimaryMeasure as PrimaryMeasure,
    QualityInfo as QualityInfo,
)
from ._metadata import IMAGE
from ._experimental_tags import (
    ACQUISITION,
    CONDITION,
    CULTURE,
    EXPERIMENT,
    GENETIC,
    PLATE,
    SAMPLE,
    STUDY,
)

from ._bbox import BBOX
from ._color_composition import ColorComposition
from ._color_hsv import ColorHSV
from ._color_lab import ColorLab
from ._color_xy import Colorxy
from ._color_xyz import ColorXYZ
from ._edge_correction import EDGE_CORRECTION
from ._grid import GRID
from ._grid_linreg_stats import GRID_LINREG_STATS
from ._neighbor_dist import NEIGHBOR_DIST
from ._grid_spread import GRID_SPREAD
from ._linear_cap_and_lag_model import LINEAR_CAP_AND_LAG_MODEL
from ._intensity import INTENSITY
from ._linear_lag_model import LINEAR_LAG_MODEL
from ._log_growth_model import LOG_GROWTH_MODEL
from ._model_metrics import MODEL_METRICS
from ._object import OBJECT
from ._curation import CURATION
from ._error_category import ErrorCategory
from ._metadata_match import METADATA_MATCH
from ._quality_check import QUALITY_CHECK
from ._quality_count import QUALITY_COUNT
from ._quality_icc import QUALITY_ICC
from ._quality_mad import QUALITY_MAD
from ._quality_occupancy import QUALITY_OCCUPANCY
from ._quality_se import QUALITY_SE
from ._quality_tukey import QUALITY_TUKEY
from ._quality_zmax import QUALITY_ZMAX
from ._radial_expansion import RADIAL_EXPANSION
from ._orientation_zones import (
    ORIENTATION_ZONE_DIAGNOSTIC,
    ORIENTATION_ZONE_PRIMARY,
    ORIENTATION_ZONES,
)
from ._shape import SHAPE
from ._size import SIZE
from ._symmetric_zones import SYMMETRIC_ZONES
from ._texture import TEXTURE

__all__ = [
    "Entry",
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
