"""Internal evaluation machinery (private)."""
from __future__ import annotations

from ._builder import build_pipeline
from ._evaluator import EvaluationResult, Evaluator, _robust_aggregate

__all__ = ["build_pipeline", "Evaluator", "EvaluationResult", "_robust_aggregate"]
