"""Analytics for quantified fungal colony plates.

Provides post-measurement tools that adjust colony statistics for plate layout artifacts,
fit growth curves, and prune outliers so downstream comparisons reflect biology rather
than imaging geometry. Includes edge correction for grid layouts, log-phase growth
modeling across time courses, and Tukey-style outlier removal for colony metrics.
"""

from ._double_softplus import DoubleSoftplus
from ._edge_correction import EdgeCorrector
from ._expected_vs_detected import ExpectedVsDetectedCount
from ._linear_softplus import LinearSoftplus
from ._log_growth_model import LogGrowthModel
from ._replicate_agreement import ReplicateAgreement
from ._tukey_outlier import TukeyOutlierRemover

__all__ = [
    "DoubleSoftplus",
    "EdgeCorrector",
    "ExpectedVsDetectedCount",
    "LinearSoftplus",
    "LogGrowthModel",
    "ReplicateAgreement",
    "TukeyOutlierRemover",
]
