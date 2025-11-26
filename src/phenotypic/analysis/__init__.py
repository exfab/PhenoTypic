"""Analytics for quantified fungal colony plates.

Provides post-measurement tools that adjust colony statistics for plate layout artifacts,
fit growth curves, and prune outliers so downstream comparisons reflect biology rather
than imaging geometry. Includes edge correction for grid layouts, log-phase growth
modeling across time courses, and Tukey-style outlier removal for colony metrics.
"""
from ._edge_correction import EdgeCorrector
from ._log_growth_model import LogGrowthModel
from ._tukey_outlier import TukeyOutlierRemover

__all__ = [
    "EdgeCorrector",
    "LogGrowthModel",
    "TukeyOutlierRemover",
]
