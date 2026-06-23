# tests/unit/tune/test_optuna_direction.py
"""Phase 2: the tuner minimizes cost (lower-is-better), one ``_MINIMIZE`` literal.

The canonical direction literal and the ``study_objective_kwargs`` mapping it
feeds ``create_study`` are the single source of the optimizer's direction. After
the cost cutover every study (and every axis of a multi-objective one) minimizes.
"""
from __future__ import annotations

from phenotypic.tune.strategy._optuna_support import (
    _MINIMIZE,
    study_objective_kwargs,
)


def test_minimize_is_the_canonical_literal():
    assert _MINIMIZE == "minimize"


def test_single_objective_kwargs_minimize():
    # None or a single-axis directions list → the scalar minimize study.
    assert study_objective_kwargs(None) == {"direction": "minimize"}
    assert study_objective_kwargs(["minimize"]) == {"direction": "minimize"}


def test_multi_objective_kwargs_all_minimize():
    kwargs = study_objective_kwargs(["minimize", "minimize"])
    assert kwargs == {"directions": ["minimize", "minimize"]}


def test_objective_directions_all_minimize():
    from phenotypic.tune._multi_objective import objective_directions

    class _MultiScorer:
        multi_objective = True

        def objective_names(self):
            return ["s0", "s1", "s2"]

    assert objective_directions(_MultiScorer()) == ["minimize", "minimize", "minimize"]


def test_objective_directions_single_objective_is_none():
    from phenotypic.tune._multi_objective import objective_directions

    class _ScalarScorer:
        multi_objective = False

    assert objective_directions(_ScalarScorer()) is None
