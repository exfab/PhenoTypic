"""The ``TuneSpec`` per-field tuning-metadata marker (re-export).

The marker's canonical definition lives in :mod:`phenotypic.sdk_.typing_`
alongside the sibling field markers (``OperationField``, ``NdArrayField``,
``ColumnRef``). It is re-exported here so the historical import path
``phenotypic.tune._search_space._tune_spec.TuneSpec`` — and the public
``from phenotypic.tune import TuneSpec`` — keep working, while operation modules
can import the marker from ``sdk_`` without dragging in the tune engine
(which would create an import cycle for foundational modules loaded before
``ImagePipeline`` exists, e.g. the GAT mixin).
"""
from __future__ import annotations

from phenotypic.sdk_.typing_ import TuneSpec

__all__ = ["TuneSpec"]
