"""Internal evaluation machinery (private)."""
from __future__ import annotations

from ._builder import build_pipeline
from ._evaluator import EvaluationResult, Evaluator, _robust_aggregate
from ._generalization import (
    GeneralizationReport,
    compute_generalization_gap,
    run_held_out,
)
from ._held_out import HeldOutConfig, infer_group_key
from ._split import (
    Split,
    _dataset_identity,
    derive_split,
    read_split,
    resolve_split,
    write_split,
)

__all__ = [
    "build_pipeline",
    "Evaluator",
    "EvaluationResult",
    "_robust_aggregate",
    "GeneralizationReport",
    "compute_generalization_gap",
    "run_held_out",
    "HeldOutConfig",
    "infer_group_key",
    "Split",
    "_dataset_identity",
    "derive_split",
    "read_split",
    "resolve_split",
    "write_split",
]
