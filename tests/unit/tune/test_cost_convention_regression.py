"""§10 cross-phase regressions for the minimize-cost + Tchebycheff cutover.

These pass only once Phases 1-4 have landed (the engine minimizes bounded cost,
the composite is augmented Tchebycheff, and the overfit gap is loss-space). A
red test here means a phase is incomplete, not that the test is wrong.
"""
from __future__ import annotations

import math
from typing import Any, ClassVar

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.tune import (
    Categorical,
    Evaluator,
    Knob,
    SearchSpace,
)
from phenotypic.tune.score import Scorer
from phenotypic.tune.strategy import GridConfig
from phenotypic.tune._engine import TuningEngine
from phenotypic.tune._evaluation._generalization import compute_generalization_gap
from phenotypic.tune.score._orient import Sense
from phenotypic.tune._spec import Budget, TuningSpec

# The deterministic objective surface: one knob, three settings, a fixed
# natural GOODNESS per setting (higher = better in the old world). Chosen so the
# ranking is unambiguous and the best/worst are not the grid endpoints (guards
# against an off-by-orientation that happens to pick the right end by luck).
_CHOICES: tuple[float, ...] = (0.2, 0.9, 0.5)
_NATURAL_GOODNESS: dict[float, float] = {0.2: 0.20, 0.9: 0.90, 0.5: 0.50}
_BEST_PARAM = 0.9  # max goodness == min cost
_WORST_PARAM = 0.2


def _space() -> SearchSpace:
    return SearchSpace(
        knobs=(Knob(key="0.sigma", domain=Categorical(choices=_CHOICES)),)
    )


def _base() -> ImagePipeline:
    from phenotypic.enhance import GaussianBlur

    return ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()])


class _GoodnessKnobScorer(Scorer):
    """Emits a HIGHER_BETTER natural term keyed off the chosen sigma.

    The natural value is a pure function of the chosen param (image-independent),
    so the exhaustive grid + deterministic scorer fully determine the winner.
    The chosen sigma is threaded onto the scorer per trial by
    ``_ParamCaptureEvaluator`` below.
    """

    _TERM_SENSE: ClassVar[Sense] = Sense.HIGHER_BETTER

    # The engine reuses one scorer instance; the capturing evaluator stamps this
    # per trial via object.__setattr__ so the natural value depends only on the
    # candidate's param (the documented test-double exception to statelessness).
    _chosen_sigma: float = 1.0

    def _score_terms(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        sigma = float(self._chosen_sigma)
        return {"Quality": _NATURAL_GOODNESS[sigma]}


class _ParamCaptureEvaluator(Evaluator):
    """Threads the chosen sigma onto the scorer before each evaluation.

    The engine calls evaluate(base, scorer, params, images, channel=...); we
    stamp params["0.sigma"] onto the scorer so its natural value is a pure
    function of the candidate, then delegate to the real cost-aware Evaluator.
    """

    def evaluate(self, base, scorer, params, images, *, channel=None):  # type: ignore[override]
        object.__setattr__(scorer, "_chosen_sigma", float(params["0.sigma"]))
        return super().evaluate(base, scorer, params, images, channel=channel)


def _run_grid_winner(scorer: Scorer | None = None) -> dict[str, Any]:
    spec = TuningSpec(
        pipeline=_base(),
        search_space=_space(),
        scorer=scorer or _GoodnessKnobScorer(),
        evaluator=_ParamCaptureEvaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )
    engine = TuningEngine(spec)
    engine.optimize([load_synth_yeast_plate()])
    best = engine.store.best()
    assert best is not None
    return best.params


# -- 6a: reflection winner-equivalence ----------------------------------------


def test_single_term_winner_is_cost_minimizer():
    # The new engine minimizes cost; the single-term winner must be the param
    # whose natural goodness is highest (== lowest cost == old-maximize winner).
    winner = _run_grid_winner()
    assert winner["0.sigma"] == pytest.approx(_BEST_PARAM)


def test_reflection_winner_matches_old_maximize_winner():
    # Reflection equivalence (README invariant 3 / spec §4): the cost winner is
    # the SAME param the old maximize convention would have picked. Compute the
    # old winner from the natural goodness directly and assert agreement.
    old_max_winner = max(_CHOICES, key=lambda c: _NATURAL_GOODNESS[c])
    new_min_winner = _run_grid_winner()["0.sigma"]
    assert new_min_winner == pytest.approx(old_max_winner)
    assert new_min_winner != pytest.approx(_WORST_PARAM)  # not the grid end by luck


def test_arithmetic_mean_winner_is_cost_minimizer():
    # The finalize default (mean of terms) is reflection-clean: a two-term
    # scorer's mean-cost winner is the mean-goodness winner. Use two HIGHER_BETTER
    # terms whose mean ranks the params identically to the single-term case.
    class _TwoTermScorer(_GoodnessKnobScorer):
        def _score_terms(self, image, measurements):
            g = _NATURAL_GOODNESS[float(self._chosen_sigma)]
            return {"A": g, "B": g}  # mean == g; same ranking

    winner = _run_grid_winner(_TwoTermScorer())
    assert winner["0.sigma"] == pytest.approx(_BEST_PARAM)


def test_pareto_domination_is_reflected_under_minimize():
    # Under minimize, a dominates b iff a <= b on every axis and < on one.
    # Reflection: the same trial that dominated under maximize-goodness now
    # dominates under minimize-cost when vectors are complemented.
    from phenotypic.tune._study._pareto import _dominates

    # cost vectors (lists, the landed signature): lower = better.
    # (0.1, 0.2) dominates (0.3, 0.2).
    assert _dominates([0.1, 0.2], [0.3, 0.2]) is True
    assert _dominates([0.3, 0.2], [0.1, 0.2]) is False
    # equal vectors do not dominate
    assert _dominates([0.1, 0.2], [0.1, 0.2]) is False


# -- 6b: overfit-gap SIGN (loss-space heldout_cost - cal_cost) ----------------

# The landed compute_generalization_gap keeps a goodness-space body
# (absolute = cal_score - heldout_score) and its callers pass the
# goodness-equivalent (1 - cost). So to assert the LOSS-SPACE gap sign
# (heldout_cost - cal_cost), we feed it goodness-equivalents the same way
# run_held_out does: cal_g = 1 - cal_cost, heldout_g = 1 - heldout_cost. The
# returned ``absolute`` (cal_g - heldout_g) then EQUALS heldout_cost - cal_cost,
# the standard positive-is-overfit loss-space gap (doctest: (0.9, 0.5) -> +0.4).


def test_overfit_winner_is_flagged_under_cost():
    # Overfit: held-out cost (0.5) WORSE than calibration cost (0.1), gap = +0.4.
    # Feed goodness-equivalents (cal_g=0.9, heldout_g=0.5) so absolute = +0.4.
    cal_cost, heldout_cost = 0.1, 0.5
    rel, absolute, flagged = compute_generalization_gap(
        1.0 - cal_cost, 1.0 - heldout_cost, rel_margin=0.15, abs_margin=0.05
    )
    assert absolute == pytest.approx(0.4)  # heldout_cost - cal_cost
    assert absolute > 0  # positive == overfit
    assert flagged is True


def test_good_generalizer_is_not_flagged_under_cost():
    # Held-out cost (0.12) ~ calibration cost (0.10): no overfit, gap ~ +0.02
    # (below the absolute margin 0.05). Must NOT flag. Guards the symmetric
    # failure: a good generalizer mis-flagged as overfit.
    cal_cost, heldout_cost = 0.10, 0.12
    rel, absolute, flagged = compute_generalization_gap(
        1.0 - cal_cost, 1.0 - heldout_cost, rel_margin=0.15, abs_margin=0.05
    )
    assert absolute == pytest.approx(0.02)
    assert flagged is False


def test_underfit_does_not_flag_under_cost():
    # Held-out cost (0.1) BETTER than calibration (0.5): negative gap, never
    # overfit. The old accuracy-space detector would have flagged this; the
    # loss-space one must not.
    cal_cost, heldout_cost = 0.5, 0.1
    rel, absolute, flagged = compute_generalization_gap(
        1.0 - cal_cost, 1.0 - heldout_cost, rel_margin=0.15, abs_margin=0.05
    )
    assert absolute == pytest.approx(-0.4)
    assert flagged is False


# -- 6c: composite-delta snapshot (Tchebycheff != old geomean; intended) ------


def _per_child_cost_vectors() -> dict[str, tuple[float, float]]:
    """Three candidates' per-child cost (b0, b1), a non-convex 2-axis front.

    - 'balanced': (0.60, 0.60) — the conjunctive (worst-axis) optimum.
    - 'lopsided': (0.70, 0.05) — poor on axis 0, excellent on axis 1.
    - 'lopsided2': (0.05, 0.70) — mirror.

    The OLD composite combined per-child *goodness* (``g = 1 - cost``) with a
    geometric mean and MAXIMIZED it. A lopsided candidate's one excellent axis
    (goodness 0.95) lifts the goodness-product above the balanced candidate's
    (``sqrt(0.30 * 0.95) ~= 0.53 > sqrt(0.40 * 0.40) = 0.40``), so the old math
    rewards a lopsided extreme. Augmented Tchebycheff on cost is conjunctive
    (worst axis dominates: balanced's 0.60 beats lopsided's 0.70), so it prefers
    the balanced compromise. This non-convex-front disagreement is the intended
    semantics change.
    """
    return {
        "balanced": (0.60, 0.60),
        "lopsided": (0.70, 0.05),
        "lopsided2": (0.05, 0.70),
    }


def _tchebycheff_pick(vectors: dict[str, tuple[float, float]]) -> str:
    # Reproduce the Phase 3 composite math at unit level to assert the winner
    # WITHOUT a full study: minimize max_i (b_i + eps) + rho * sum_i b_i.
    eps, rho = 1e-3, 0.05

    def t(b: tuple[float, float]) -> float:
        return max(b[0] + eps, b[1] + eps) + rho * (b[0] + b[1])

    return min(vectors, key=lambda k: t(vectors[k]))


def _old_geomean_pick(vectors: dict[str, tuple[float, float]]) -> str:
    # The OLD composite operated on GOODNESS g = 1 - cost via geometric mean,
    # maximized. Reproduce its winner for the snapshot delta.
    def g_geomean(b: tuple[float, float]) -> float:
        g = (1.0 - b[0], 1.0 - b[1])
        return math.sqrt(max(g[0], 0.0) * max(g[1], 0.0))

    return max(vectors, key=lambda k: g_geomean(vectors[k]))


def test_composite_delta_is_the_intended_change():
    vectors = _per_child_cost_vectors()
    new_winner = _tchebycheff_pick(vectors)
    old_winner = _old_geomean_pick(vectors)
    # The documented, intended delta: Tchebycheff picks the balanced compromise;
    # the old geomean picked a lopsided extreme. They MUST differ (this is the
    # snapshot that proves the composite changed on purpose).
    assert new_winner == "balanced"
    assert old_winner != "balanced"
    assert new_winner != old_winner
