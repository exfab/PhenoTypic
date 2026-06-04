"""Internal scoring value types (private)."""
from __future__ import annotations

from ._gt_loader import GroundTruthMasks
from ._qc_scorer import QCScorer, _threshold_anchored
from ._reference_free_scorer import ReferenceFreeScorer
from ._scorer import Scorer
from ._supervised import SupervisedScorer

# ``_composite`` is imported last: ``CompositeScorer`` types its children with a
# polymorphic ``ScorerField`` built from the local ``Scorer`` base (defined in
# ``_composite`` itself, not imported from ``.._spec``, to keep ``_composite``
# below ``_spec`` in the import graph). Importing it after the leaf scorers are
# bound keeps this leaf package self-contained.
from ._composite import CompositeScorer

__all__ = [
    "Scorer",
    "QCScorer",
    "ReferenceFreeScorer",
    "GroundTruthMasks",
    "SupervisedScorer",
    "CompositeScorer",
    "_threshold_anchored",
]
