from __future__ import annotations

import pytest

from phenotypic.tune._evaluation._generalization import (
    compute_generalization_gap,
    run_held_out,
)


def test_compute_gap_doctest_inputs_are_goodness_equivalents():
    # The formula is unchanged (accuracy-space train - test). Read its inputs as
    # goodness-equivalents (1 - cost): cal_g 0.9, heldout_g 0.5 -> drop 0.4, flagged.
    rel, absolute, flagged = compute_generalization_gap(
        0.9, 0.5, rel_margin=0.15, abs_margin=0.05
    )
    assert (round(rel, 3), round(absolute, 3), flagged) == (0.444, 0.4, True)


class _FakeSplit:
    kind = "group"
    held_out = ["h1"]
    group_key = None
    within_group_caveat = False
    dataset_identity = "id-1"


class _FakeResult:
    def __init__(self, score: float) -> None:
        self.score = score


class _FakeEvaluator:
    def __init__(self, heldout_cost: float) -> None:
        self._heldout_cost = heldout_cost

    def evaluate(self, pipeline, scorer, params, images):
        return _FakeResult(self._heldout_cost)


class _FakeHeldOutCfg:
    gap_margin_relative = 0.15
    gap_margin_absolute = 0.05


class _FakeSpec:
    def __init__(self, heldout_cost: float) -> None:
        self.evaluator = _FakeEvaluator(heldout_cost)
        self.pipeline = object()
        self.scorer = object()
        self.held_out = _FakeHeldOutCfg()


class _FakeWinner:
    def __init__(self, cal_cost: float) -> None:
        self.params = {}
        self.score = cal_cost
        self.gap = 0.0


def test_overfit_winner_is_flagged_under_cost():
    # Overfit: held-out cost (0.6) >> calibration cost (0.1). The standard
    # loss-space gap heldout_cost - cal_cost = 0.5 > 0 must FLAG.
    report = run_held_out(
        _FakeSpec(heldout_cost=0.6),
        _FakeWinner(cal_cost=0.1),
        _FakeSplit(),
        {"h1": object()},
    )
    assert report.gap == pytest.approx(0.5)
    assert report.flagged is True


def test_good_generaliser_is_not_flagged_under_cost():
    # Good generaliser: held-out cost (0.12) ≈ calibration cost (0.10). Gap 0.02
    # is below the absolute margin -> NOT flagged (and not mis-flagged as overfit).
    report = run_held_out(
        _FakeSpec(heldout_cost=0.12),
        _FakeWinner(cal_cost=0.10),
        _FakeSplit(),
        {"h1": object()},
    )
    assert report.gap == pytest.approx(0.02)
    assert report.flagged is False
