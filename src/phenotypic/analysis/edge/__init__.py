"""Grid-aware edge-effect correction analyzers.

:class:`EdgeCorrector` caps edge-colony measurements that are inflated by
missing orthogonal neighbors, using the grid topology and abstract template
provided by :class:`~phenotypic.analysis.abc_.EdgeCorrection`.
"""

from ._edge_correction import EdgeCorrector

__all__ = ["EdgeCorrector"]
