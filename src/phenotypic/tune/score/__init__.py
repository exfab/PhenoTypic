"""Tuning objectives — the public ``phenotypic.tune.score`` namespace.

The pluggable scoring objectives a :class:`~phenotypic.tune.TuningSpec` drives:
the abstract :class:`Scorer` base plus the four concrete objectives (no-GT
:class:`QCScorer`, supervised :class:`SupervisedScorer`, reference-free
:class:`ReferenceFreeScorer`, and the :class:`CompositeScorer` panel). The
inner ``_*`` modules stay private; import the classes from this package.
"""
from __future__ import annotations

from phenotypic.sdk_.typing_ import CompositeBlend

from ._gt_loader import GroundTruthMasks
from ._qc_scorer import QCScorer, _threshold_anchored
from ._reference_free_scorer import ReferenceFreeScorer
from ._scorer import Scorer, ScorerField
from ._supervised import SupervisedScorer

# ``_composite`` is imported last: ``CompositeScorer`` types its children with a
# polymorphic ``ScorerField`` built from the local ``Scorer`` base (defined in
# ``_composite`` itself, not imported from ``.._spec``, to keep ``_composite``
# below ``_spec`` in the import graph). Importing it after the leaf scorers are
# bound keeps this leaf package self-contained.
from ._composite import CompositeScorer

__all__ = [
    "Scorer",
    "ScorerField",
    "QCScorer",
    "ReferenceFreeScorer",
    "GroundTruthMasks",
    "SupervisedScorer",
    "CompositeScorer",
    "CompositeBlend",
    "_threshold_anchored",
]
