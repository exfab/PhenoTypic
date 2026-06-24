"""Set-analyzer filters that prune outlier rows from measurement frames.

Each class is a :class:`~phenotypic.analysis.abc_.SetAnalyzer` subclass that
removes colony measurements whose statistics mark them as outliers
(MAD-based or Tukey-fence-based), so downstream comparisons are not skewed
by a handful of extreme values.
"""

from ._mad_outlier import MADOutlierRemover
from ._tukey_outlier import TukeyOutlierRemover

__all__ = [
    "MADOutlierRemover",
    "TukeyOutlierRemover",
]
