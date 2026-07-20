"""Analytics for quantified fungal colony plates.

Provides post-measurement tools that adjust colony statistics for plate layout artifacts,
fit growth curves, and prune outliers so downstream comparisons reflect biology rather
than imaging geometry. Includes edge correction for grid layouts, log-phase growth
modeling across time courses, and Tukey-style outlier removal for colony metrics.
"""

from ._linear_cap_and_lag_model import LinearCapAndLagModel
from ._cubical_persistence import PersistencePairsResult, cubical_persistence
from .edge import EdgeCorrector
from ._error_cutoffs import ErrorCutoffFinder
from ._helper import (
    filter_spec_json,
    filter_spec_query,
    render_error_analysis_html,
    render_error_analysis_report,
)
from ._linear_lag_model import LinearLagModel
from ._log_growth_model import LogGrowthModel
from .filter import MADOutlierRemover, TukeyOutlierRemover
from .qc import (
    ExpectedVsDetectedCount,
    GridOccupancy,
    ICC,
    MaxModifiedZScore,
    RelativeMAD,
    ReplicateAgreement,
    TukeyOutlierFraction,
)

__all__ = [
    "LinearCapAndLagModel",
    "EdgeCorrector",
    "ErrorCutoffFinder",
    "ExpectedVsDetectedCount",
    "GridOccupancy",
    "ICC",
    "LinearLagModel",
    "LogGrowthModel",
    "MADOutlierRemover",
    "MaxModifiedZScore",
    "PersistencePairsResult",
    "RelativeMAD",
    "ReplicateAgreement",
    "TukeyOutlierFraction",
    "TukeyOutlierRemover",
    "cubical_persistence",
    "filter_spec_json",
    "filter_spec_query",
    "render_error_analysis_html",
    "render_error_analysis_report",
]
