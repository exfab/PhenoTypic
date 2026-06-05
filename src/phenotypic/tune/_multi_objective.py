"""Multi-objective inference from the scorer (plan §0b: "inferred from the scorer").

The single place that decides whether a tuning run is multi-objective and what
its objective axes are — read off the **scorer**, never a separate flag. A
``CompositeScorer(multi_objective=True)`` is the documented multi-objective
objective (its ``finalize`` returns a per-child dict); its
:meth:`CompositeScorer.objective_names` give the stable axis order that the
NSGA-II ``directions``, the ``objectives_json`` keys, and the ``pareto/`` axis
labels all share.

Used by the engine (to build a multi-objective Optuna study), the
``TuningSpec``/run validation guard (grid/random + multi-objective → reject,
4.8), and ``run_tuning``'s ``pareto/`` writer (4.7).
"""
from __future__ import annotations

from typing import Any

#: Every objective in a tuning study is normalized higher-is-better
#: (robust-eval §5), so a multi-objective study maximizes every axis.
_MAXIMIZE = "maximize"


def is_multi_objective(scorer: Any) -> bool:
    """Whether ``scorer`` drives a multi-objective (Pareto) run.

    Inferred from the scorer (plan §0b), not a separate spec flag: a scorer is
    multi-objective when it carries a truthy ``multi_objective`` attribute — i.e.
    a ``CompositeScorer(multi_objective=True)``, whose ``finalize`` returns a
    per-objective ``dict``. Any other scorer (``QCScorer``, a scalar
    ``CompositeScorer``, …) is single-objective.

    Args:
        scorer: The tuning spec's scorer.

    Returns:
        ``True`` for a multi-objective scorer; ``False`` otherwise.
    """
    return bool(getattr(scorer, "multi_objective", False))


def objective_names(scorer: Any) -> list[str]:
    """The ordered objective-axis names of a multi-objective ``scorer``.

    Delegates to the scorer's own ``objective_names`` (a
    ``CompositeScorer`` returns its child handles ``["s0", "s1", …]``) so the
    axis order is authoritative and matches the emitted ``objectives_json`` keys.

    Args:
        scorer: A multi-objective scorer.

    Returns:
        The objective-axis names in order; ``[]`` when the scorer exposes none.
    """
    names = getattr(scorer, "objective_names", None)
    if callable(names):
        return list(names())
    return []


def reject_grid_random_multi_objective(scorer: Any, strategy: Any) -> None:
    """Reject a multi-objective scorer paired with a non-Optuna strategy.

    The single guard behind both the ``TuningSpec`` construction-time validator
    and the ``run_tuning`` run-validation backstop (a ``--strategy`` override
    bypasses the model validator). Multi-objective (Pareto) search needs an
    Optuna NSGA-II study; the exhaustive grid and seeded-random strategies are
    single-objective only, so the pairing is a configuration error.

    Args:
        scorer: The tuning spec's scorer.
        strategy: The (possibly ``--strategy``-overridden) strategy config.

    Raises:
        ValueError: When ``scorer`` is multi-objective but ``strategy`` is not an
            Optuna strategy. The message is actionable (points at
            ``--strategy nsga2`` / an Optuna strategy).
    """
    if not is_multi_objective(scorer):
        return
    # Duck-typed Optuna check: an Optuna strategy carries the ``sampler`` field
    # (avoids importing the concrete config here and keeps the guard reusable).
    if getattr(strategy, "kind", None) == "optuna":
        return
    raise ValueError(
        "multi-objective scoring (a CompositeScorer with multi_objective=True) "
        "requires an Optuna strategy — the grid and random strategies are "
        f"single-objective. Got {type(strategy).__name__}; use --strategy nsga2 "
        "(or another Optuna sampler) / an OptunaConfig strategy."
    )


def objective_directions(scorer: Any) -> list[str] | None:
    """The per-objective Optuna ``directions`` for a multi-objective ``scorer``.

    ``["maximize"] * n`` over the scorer's objective axes — every tuning
    objective is higher-is-better (robust-eval §5). ``None`` for a
    single-objective scorer (a scalar study), and also ``None`` when a
    multi-objective scorer resolves to fewer than two named axes (a single axis
    is not a Pareto problem — fall back to the scalar path rather than build a
    degenerate one-objective "multi-objective" study).

    Args:
        scorer: The tuning spec's scorer.

    Returns:
        The directions list (length ≥ 2), or ``None`` for the single-objective
        path.
    """
    if not is_multi_objective(scorer):
        return None
    names = objective_names(scorer)
    if len(names) < 2:
        return None
    return [_MAXIMIZE] * len(names)
