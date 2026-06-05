"""4.5p1 C1 — the per-trial ``gap`` (relative across-plate dispersion).

``gap`` is the relative IQR ``(q75 - q25) / max(|median|, eps)`` of the **primary
(first) term**'s per-image scores — a cheap instability / overfit-risk flag, NOT
a held-out gap. A flat term → ``gap ≈ 0``; a single-image trial → ``0.0`` (no
dispersion); below ``min_stability_n`` images → ``None`` (dispersion unreliable,
mirroring the stability small-n guard).
"""
from __future__ import annotations

from pydantic import PrivateAttr

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.tune import EvaluationResult, Evaluator, Scorer
from phenotypic.tune._evaluation._evaluator import _per_trial_dispersion


class _SequenceCountScorer(Scorer):
    """Returns preset per-call ``Count`` values, ignoring its inputs.

    The relative-IQR ``gap`` is order-independent, so the value→image mapping the
    rung ladder's id-sort imposes does not matter — only the multiset of scores.
    """

    values: list[float]
    _cursor: int = PrivateAttr(default=0)

    def score_image(self, image, measurements) -> dict[str, float]:
        value = self.values[self._cursor % len(self.values)]
        self._cursor += 1
        return {"Count": float(value)}


def _plates(n):
    return [load_synth_yeast_plate() for _ in range(n)]


# -- _per_trial_dispersion (unit) ---------------------------------------------


def test_dispersion_single_image_is_zero():
    assert _per_trial_dispersion({"Count": [0.5]}, min_n=4) == 0.0


def test_dispersion_below_min_n_is_none():
    # 2 or 3 images (below min_n=4) → unreliable → None.
    assert _per_trial_dispersion({"Count": [0.5, 0.9]}, min_n=4) is None
    assert _per_trial_dispersion({"Count": [0.5, 0.9, 0.7]}, min_n=4) is None


def test_dispersion_flat_term_is_zero():
    flat = {"Count": [0.7, 0.7, 0.7, 0.7, 0.7]}
    assert _per_trial_dispersion(flat, min_n=4) == 0.0


def test_dispersion_relative_iqr_of_primary_term():
    import numpy as np

    values = [0.2, 0.4, 0.6, 0.8, 1.0]
    q75, q25 = np.percentile(values, [75, 25])
    median = float(np.median(values))
    expected = (q75 - q25) / max(abs(median), 1e-12)
    got = _per_trial_dispersion({"Count": values}, min_n=4)
    assert got is not None
    assert abs(got - expected) < 1e-9


def test_dispersion_uses_first_term_only():
    # A second term with wild spread must not affect the primary-term gap.
    per_term = {"Count": [0.5, 0.5, 0.5, 0.5], "Other": [0.0, 1.0, 0.0, 1.0]}
    assert _per_trial_dispersion(per_term, min_n=4) == 0.0


def test_dispersion_empty_is_none():
    assert _per_trial_dispersion({}, min_n=4) is None


# -- end-to-end through Evaluator.evaluate ------------------------------------


def test_gap_is_relative_dispersion_of_primary_term():
    import numpy as np

    spread = [0.2, 0.4, 0.6, 0.8, 1.0]
    base = ImagePipeline(ops=[OtsuDetector()])
    result = Evaluator(min_stability_n=4).evaluate(
        base, _SequenceCountScorer(values=spread), {}, _plates(5),
    )
    assert isinstance(result, EvaluationResult)
    q75, q25 = np.percentile(spread, [75, 25])
    median = float(np.median(spread))
    expected = (q75 - q25) / max(abs(median), 1e-12)
    assert result.gap is not None
    assert abs(result.gap - expected) < 1e-9


def test_gap_none_below_min_stability_n():
    # 2 images, below default min_stability_n=4.
    base = ImagePipeline(ops=[OtsuDetector()])
    result = Evaluator().evaluate(
        base, _SequenceCountScorer(values=[0.5, 0.9]), {}, _plates(2),
    )
    assert result.gap is None


def test_gap_zero_for_single_image():
    base = ImagePipeline(ops=[OtsuDetector()])
    result = Evaluator().evaluate(
        base, _SequenceCountScorer(values=[0.5]), {}, _plates(1),
    )
    assert result.gap == 0.0


def test_evaluator_has_robust_eval_config_defaults():
    ev = Evaluator()
    assert ev.min_stability_n == 4
    assert ev.suspicious_score_floor == 0.7
    assert ev.suspicious_count_floor == 0.3


# -- C2: the suspicious gaming-signature flag ---------------------------------


def test_is_suspicious_high_score_low_count():
    from phenotypic.tune._evaluation._evaluator import _is_suspicious

    # High score paired with a low Count → the qc §5 under-detection signature.
    assert _is_suspicious(
        0.95, {"Count": 0.2}, 5, score_floor=0.7, count_floor=0.3
    ) is True


def test_is_suspicious_faithful_high_count():
    from phenotypic.tune._evaluation._evaluator import _is_suspicious

    # High score with a faithful (high) Count is NOT suspicious.
    assert _is_suspicious(
        0.95, {"Count": 0.9}, 5, score_floor=0.7, count_floor=0.3
    ) is False


def test_is_suspicious_respects_thresholds():
    from phenotypic.tune._evaluation._evaluator import _is_suspicious

    # Score below the floor → not flagged even with a low Count.
    assert _is_suspicious(
        0.5, {"Count": 0.1}, 5, score_floor=0.7, count_floor=0.3
    ) is False
    # Count above the floor → not flagged even with a high score.
    assert _is_suspicious(
        0.99, {"Count": 0.31}, 5, score_floor=0.7, count_floor=0.3
    ) is False
    # Exactly on both boundaries (>= score_floor AND <= count_floor) → flagged.
    assert _is_suspicious(
        0.7, {"Count": 0.3}, 5, score_floor=0.7, count_floor=0.3
    ) is True


def test_is_suspicious_missing_count_defaults_faithful():
    from phenotypic.tune._evaluation._evaluator import _is_suspicious

    # No Count term → default 1.0 (faithful) → never suspicious.
    assert _is_suspicious(0.99, {}, 5, score_floor=0.7, count_floor=0.3) is False


class _GamingScorer(Scorer):
    """A gamer: a high finalized ``score`` masking a low aggregated ``Count``.

    ``score_image`` reports a low Count term per image; ``finalize`` overrides to
    return an inflated scalar regardless of the term — reproducing the qc §5
    "great score on under-detection" signature the suspicious flag catches.
    """

    def score_image(self, image, measurements) -> dict[str, float]:
        return {"Count": 0.1}

    def finalize(self, terms):
        return 0.95  # inflated, ignores the low Count term


def test_suspicious_high_score_low_count_end_to_end():
    base = ImagePipeline(ops=[OtsuDetector()])
    result = Evaluator(
        suspicious_score_floor=0.7, suspicious_count_floor=0.3
    ).evaluate(base, _GamingScorer(), {}, _plates(4))
    # score 0.95 (>= 0.7) AND aggregated Count ~0.1 (<= 0.3) → suspicious.
    assert result.score == 0.95
    assert result.terms["Count"] <= 0.3
    assert result.suspicious is True


def test_not_suspicious_faithful():
    base = ImagePipeline(ops=[OtsuDetector()])
    # High Count term + the default mean finalize → faithful high score.
    result = Evaluator().evaluate(
        base, _SequenceCountScorer(values=[0.9, 0.9, 0.9, 0.9]), {}, _plates(4),
    )
    assert result.suspicious is False
