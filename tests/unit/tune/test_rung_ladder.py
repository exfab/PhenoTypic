"""The ASHA-style rung ladder inside ``Evaluator.evaluate`` (robust-eval §7).

The Evaluator scores the calibration set in growing rung blocks, reports the
running robust aggregate to the pruning channel after each rung, and checks
``should_prune()`` between rungs. The no-op channel reproduces the old
single-pass score exactly; a pruning channel can short-circuit to a partial
``EvaluationResult(pruned=True)``.
"""
from __future__ import annotations

import pytest
from pydantic import PrivateAttr

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.tune._evaluation._evaluator import EvaluationResult, Evaluator
from phenotypic.tune._scoring._scorer import Scorer


class _PerImageScorer(Scorer):
    """Scores term ``X`` as the image's id mod-mapped value (id → value)."""

    mapping: dict[int, float]

    def score_image(self, image, measurements) -> dict[str, float]:
        return {"X": float(self.mapping[id(image)])}


class _CountingScorer(Scorer):
    """Records how many distinct images it scored (by id), returns constant."""

    _seen: set[int] = PrivateAttr(default_factory=set)

    def score_image(self, image, measurements) -> dict[str, float]:
        self._seen.add(id(image))
        return {"X": 1.0}


class _RecordingChannel:
    def __init__(self) -> None:
        self.reports: list[tuple[float, int]] = []

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return False


class _PruneAfterFirstRungChannel:
    """Prunes on the first ``should_prune`` check (i.e. after rung 1)."""

    def __init__(self) -> None:
        self.reports: list[tuple[float, int]] = []

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return True


def _imgs(n: int) -> list:
    # Distinct image objects so id()-keyed memoization and per-image scoring work.
    return [load_synth_yeast_plate() for _ in range(n)]


# --- _rung_sizes geometry ------------------------------------------------------

def test_rung_sizes_geometric_growth_factor_3():
    ev = Evaluator()  # floor 6, factor 3, min_rungs 2
    assert ev._rung_sizes(18) == [6, 18]
    assert ev._rung_sizes(54) == [18, 54]
    # ceil(n/3) dominates the floor once n is large.
    assert ev._rung_sizes(90) == [30, 90]


def test_rung_sizes_first_rung_respects_floor():
    ev = Evaluator()
    # ceil(9/3)=3 < floor 6 → first rung is the floor, last is all.
    assert ev._rung_sizes(9) == [6, 9]


def test_rung_sizes_self_disables_when_too_small():
    ev = Evaluator()
    # n below/at the floor cannot split into >=2 rungs → a single all-images rung.
    assert ev._rung_sizes(3) == [3]
    assert ev._rung_sizes(6) == [6]


# --- ladder behavior -----------------------------------------------------------

def test_report_called_once_per_rung():
    ev = Evaluator()
    channel = _RecordingChannel()
    imgs = _imgs(18)
    ev.evaluate(
        ImagePipeline(ops=[OtsuDetector()]),
        _CountingScorer(),
        {},
        imgs,
        channel=channel,
    )
    # Two rungs ([6, 18]) → two reports, with monotonically growing plate counts.
    assert [step for _, step in channel.reports] == [6, 18]


def test_should_prune_short_circuits_to_partial_pruned_result():
    ev = Evaluator()
    channel = _PruneAfterFirstRungChannel()
    scorer = _CountingScorer()
    imgs = _imgs(18)
    result = ev.evaluate(
        ImagePipeline(ops=[OtsuDetector()]), scorer, {}, imgs, channel=channel
    )
    assert isinstance(result, EvaluationResult)
    assert result.pruned is True
    assert result.failed is False
    # Only the first rung's images were scored (partial), not the full set.
    assert result.n_images == 6
    assert len(scorer._seen) == 6


def test_unpruned_full_pass_equals_single_pass_score():
    base = ImagePipeline(ops=[OtsuDetector()])
    imgs = _imgs(18)
    mapping = {id(im): float(i) for i, im in enumerate(imgs)}
    # Ladder pass (no-op channel never prunes → scores all 18).
    laddered = Evaluator().evaluate(base, _PerImageScorer(mapping=mapping), {}, imgs)
    # Reference single pass over the SAME images in id-sorted order.
    ordered = sorted(imgs, key=id)
    ref = Evaluator().evaluate(
        base, _PerImageScorer(mapping=mapping), {}, ordered
    )
    assert laddered.n_images == 18
    assert laddered.pruned is False
    assert laddered.score == pytest.approx(ref.score)
    assert laddered.terms == pytest.approx(ref.terms)


def test_memoization_scores_each_image_once_across_rungs():
    ev = Evaluator()
    scorer = _CountingScorer()
    imgs = _imgs(18)
    ev.evaluate(
        ImagePipeline(ops=[OtsuDetector()]), scorer, {}, imgs, channel=_RecordingChannel()
    )
    # Every image scored exactly once even though rung 2 ⊃ rung 1.
    assert len(scorer._seen) == 18


def test_per_image_exception_drags_score_not_whole_fail():
    # robust-eval §10: one image raising mid-scoring contributes the worst term
    # and the loop continues — NOT a whole-candidate FAIL.
    class _OneRaisesScorer(Scorer):
        _calls: int = PrivateAttr(default=0)

        def score_image(self, image, measurements) -> dict[str, float]:
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("one bad plate")
            return {"X": 1.0}

    base = ImagePipeline(ops=[OtsuDetector()])
    imgs = _imgs(3)  # single-rung ladder
    result = Evaluator().evaluate(base, _OneRaisesScorer(), {}, imgs)
    assert result.failed is False
    assert result.n_images == 3
    # The worst term (0.0) drags the aggregate below the clean 1.0.
    assert result.terms["X"] < 1.0


def test_all_images_error_is_a_fail():
    class _AllRaiseScorer(Scorer):
        def score_image(self, image, measurements) -> dict[str, float]:
            raise RuntimeError("every plate errors")

    base = ImagePipeline(ops=[OtsuDetector()])
    result = Evaluator().evaluate(base, _AllRaiseScorer(), {}, _imgs(3))
    assert result.failed is True


def test_build_failure_is_a_fail():
    # A candidate whose pipeline cannot be built is a true FAIL (no scoring).
    base = ImagePipeline(ops=[OtsuDetector()])
    bad_params = {"0.this_param_does_not_exist": 123}
    result = Evaluator().evaluate(base, _CountingScorer(), {}, _imgs(3), )
    # sanity: clean params succeed
    assert result.failed is False
    result_bad = Evaluator().evaluate(base, _CountingScorer(), bad_params, _imgs(3))
    assert result_bad.failed is True
