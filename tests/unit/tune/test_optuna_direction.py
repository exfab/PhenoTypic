# tests/unit/tune/test_optuna_direction.py
"""Phase 2: the tuner minimizes cost (lower-is-better), one ``_MINIMIZE`` literal.

The canonical direction literal and the ``study_objective_kwargs`` mapping it
feeds ``create_study`` are the single source of the optimizer's direction. After
the cost cutover every study (and every axis of a multi-objective one) minimizes.
"""
from __future__ import annotations

import pytest

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


def test_duplicate_objective_names_are_rejected() -> None:
    """A repeated scorer axis cannot define a valid objective vector."""
    from phenotypic.tune._multi_objective import objective_names

    class _DuplicateAxisScorer:
        multi_objective = True

        def objective_names(self):
            return ["s0", "s0"]

    with pytest.raises(ValueError, match="unique"):
        objective_names(_DuplicateAxisScorer())


@pytest.mark.parametrize(
    "names",
    [
        ["Dice", "dice"],
        ["Straße", "STRASSE"],
    ],
)
def test_objective_names_reject_casefold_filename_collisions(names) -> None:
    """Distinct spellings cannot alias the same artifact on Windows."""
    from phenotypic.tune._multi_objective import objective_names

    scorer = type(
        "CasefoldCollisionScorer",
        (),
        {"multi_objective": True, "objective_names": lambda self: names},
    )()

    with pytest.raises(ValueError, match="case-insensitive|casefold|unique"):
        objective_names(scorer)


def test_ordered_objective_values_rejects_duplicate_raw_axes() -> None:
    """A duplicate raw axis cannot pass exact-key validation through a set."""
    from phenotypic.tune._multi_objective import ordered_objective_values

    with pytest.raises(ValueError, match="unique"):
        ordered_objective_values({"s0": 0.2}, ("s0", "s0"))



@pytest.mark.parametrize("names", [[], ["only"]])
def test_multi_objective_requires_at_least_two_axes(names) -> None:
    """A multi-objective flag cannot silently create a scalar study."""
    from phenotypic.tune._multi_objective import objective_directions

    scorer = type(
        "DegenerateMultiScorer",
        (),
        {"multi_objective": True, "objective_names": lambda self: names},
    )()

    with pytest.raises(ValueError, match="at least two"):
        objective_directions(scorer)


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "../escape",
        r"..\escape",
        "/absolute",
        r"C:\escape",
        "bad\x00axis",
        "bad\naxis",
        r"\\server\share",
        "bad<axis",
        "bad>axis",
        'bad"axis',
        "bad:axis",
        "bad|axis",
        "bad?axis",
        "bad*axis",
        "bad\x01axis",
        "bad\x1faxis",
        "bad\x7faxis",
        "bad\x80axis",
        "bad\x9faxis",
        "trailing.",
        "trailing ",
        "CON",
        "con.txt",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "com9.csv",
        "LPT1",
        "lpt9.json",
        "CONIN$",
        "CONOUT$",
        "COM¹",
        "LPT³.txt",
    ],
)
def test_objective_names_reject_unsafe_filename_components(name) -> None:
    """Scorer axes must be safe human-readable Pareto filename components."""
    from phenotypic.tune._multi_objective import objective_names

    scorer = type(
        "UnsafeAxisScorer",
        (),
        {"multi_objective": True, "objective_names": lambda self: ["safe", name]},
    )()

    with pytest.raises(ValueError, match="safe filename"):
        objective_names(scorer)


def test_objective_directions_single_objective_is_none():
    from phenotypic.tune._multi_objective import objective_directions

    class _ScalarScorer:
        multi_objective = False

    assert objective_directions(_ScalarScorer()) is None
