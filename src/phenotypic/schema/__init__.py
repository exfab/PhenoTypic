"""Public measurement schema for the PhenoTypic library.

This subpackage is the canonical, public home for PhenoTypic's measurement
naming conventions: the :class:`MeasurementInfo` base class plus every
``MeasurementInfo`` subclass that names a column in an output DataFrame. Each
enum lives in its own module (``_<name>.py``) and is re-exported here, so the
public import surface stays stable:

    from phenotypic.schema import MeasurementInfo, BBOX, GRID, SHAPE

It also hosts the metadata vocabulary: ``METADATA`` (framework-populated image
bookkeeping) and the seven experimental-tag enums (``GENETIC_METADATA``,
``SAMPLE_METADATA``, ``PLATE_METADATA``, ``CONDITION_METADATA``,
``CULTURE_METADATA``, ``ACQUISITION_METADATA``, ``EXPERIMENT_METADATA``) that
standardize ``Metadata_*`` columns for the ``--metadata`` join and ``post/`` ops.
"""

from ._measurement_info import Entry, MeasurementInfo
from ._rembi import REMBI_MODULE as REMBI_MODULE
from ._tiers import (
    DerivedMeasure as DerivedMeasure,
    DescriptiveTrait as DescriptiveTrait,
    DirectPhenotype as DirectPhenotype,
    DiscriminativeFeature as DiscriminativeFeature,
    IdentityInfo as IdentityInfo,
    PrimaryMeasure as PrimaryMeasure,
    QualityInfo as QualityInfo,
)
from ._metadata import METADATA
from ._experimental_tags import (
    ACQUISITION_METADATA,
    CONDITION_METADATA,
    CULTURE_METADATA,
    EXPERIMENT_METADATA,
    GENETIC_METADATA,
    PLATE_METADATA,
    SAMPLE_METADATA,
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
from ._quality_check import QUALITY_CHECK
from ._quality_count import QUALITY_COUNT
from ._quality_icc import QUALITY_ICC
from ._quality_mad import QUALITY_MAD
from ._quality_occupancy import QUALITY_OCCUPANCY
from ._quality_se import QUALITY_SE
from ._quality_tukey import QUALITY_TUKEY
from ._quality_zmax import QUALITY_ZMAX
from ._radial_expansion import RADIAL_EXPANSION
from ._shape import SHAPE
from ._size import SIZE
from ._symmetric_zones import SYMMETRIC_ZONES
from ._texture import TEXTURE

__all__ = [
    "Entry",
    "MeasurementInfo",
    "REMBI_MODULE",
    "METADATA",
    "ACQUISITION_METADATA",
    "CONDITION_METADATA",
    "CULTURE_METADATA",
    "EXPERIMENT_METADATA",
    "GENETIC_METADATA",
    "PLATE_METADATA",
    "SAMPLE_METADATA",
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
    "SYMMETRIC_ZONES",
    "TEXTURE",
]
