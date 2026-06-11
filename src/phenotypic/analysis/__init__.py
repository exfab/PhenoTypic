"""Analytics for quantified fungal colony plates.

Provides post-measurement tools that adjust colony statistics for plate layout artifacts,
fit growth curves, and prune outliers so downstream comparisons reflect biology rather
than imaging geometry. Includes edge correction for grid layouts, log-phase growth
modeling across time courses, and Tukey-style outlier removal for colony metrics.
"""

from ._double_softplus import DoubleSoftplus
from ._edge_correction import EdgeCorrector
from ._error_cutoffs import ErrorCutoffFinder
from ._linear_softplus import LinearSoftplus
from ._log_growth_model import LogGrowthModel
from ._mad_outlier import MADOutlierRemover
from .qc import (
    ExpectedVsDetectedCount,
    ICC,
    MaxModifiedZScore,
    RelativeMAD,
    ReplicateAgreement,
    TukeyOutlierFraction,
)
from ._tukey_outlier import TukeyOutlierRemover

__all__ = [
    "DoubleSoftplus",
    "EdgeCorrector",
    "ErrorCutoffFinder",
    "ExpectedVsDetectedCount",
    "ICC",
    "LinearSoftplus",
    "LogGrowthModel",
    "MADOutlierRemover",
    "MaxModifiedZScore",
    "RelativeMAD",
    "ReplicateAgreement",
    "TukeyOutlierFraction",
    "TukeyOutlierRemover",
]
