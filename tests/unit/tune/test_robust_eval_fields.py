"""4.5p1 A1 — ``EvaluationResult`` gains the robust-eval ``gap`` + ``suspicious``.

Per-trial ``gap`` is a cheap relative across-plate dispersion signal (an
instability / overfit-risk flag, NOT a held-out gap); ``suspicious`` flags the
qc §5 "great score on under-detection" gaming signature. Both default to the
neutral "no signal" value (``gap is None``, ``suspicious is False``) so every
existing construction site keeps working unchanged.
"""
from __future__ import annotations

from phenotypic.tune import EvaluationResult


def test_evaluation_result_gap_suspicious_default_none():
    result = EvaluationResult(score=0.5, terms={}, n_images=1)
    assert result.gap is None
    assert result.suspicious is False
