"""Internal evaluation machinery (private)."""
from __future__ import annotations

from ._builder import build_pipeline
from ._evaluator import EvaluationResult, Evaluator, _robust_aggregate
from ._held_out import HeldOutConfig, infer_group_key
from ._split import Split, derive_split, read_split, resolve_split, write_split

__all__ = [
    "build_pipeline",
    "Evaluator",
    "EvaluationResult",
    "_robust_aggregate",
    "HeldOutConfig",
    "infer_group_key",
    "Split",
    "derive_split",
    "read_split",
    "resolve_split",
    "write_split",
]
