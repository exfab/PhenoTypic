"""
The analysis module contains classes for performing data
analytics and cleaning on ImageSets.
"""
from ._edge_correction import EdgeCorrector
from ._log_growth_model import LogGrowthModel
from ._tukey_outlier import TukeyOutlierDetector

__all__ = [
    "EdgeCorrector",
    "LogGrowthModel",
    "TukeyOutlierDetector",
]
