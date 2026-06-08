"""QualityCheck implementations for smart-QC pipeline analysis.

Each class is a :class:`~phenotypic.analysis.abc_.QualityCheck` subclass
that flags groups of colony measurements whose statistical properties
indicate data quality problems (outliers, replicate disagreement, count
mismatches, etc.).
"""

from ._expected_vs_detected import ExpectedVsDetectedCount
from ._icc import ICC
from ._max_modz import MaxModifiedZScore
from ._relative_mad import RelativeMAD
from ._replicate_agreement import ReplicateAgreement
from ._tukey_fraction import TukeyOutlierFraction

__all__ = [
    "ExpectedVsDetectedCount",
    "ICC",
    "MaxModifiedZScore",
    "RelativeMAD",
    "ReplicateAgreement",
    "TukeyOutlierFraction",
]
