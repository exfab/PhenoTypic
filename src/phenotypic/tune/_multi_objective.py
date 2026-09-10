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

from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any
import unicodedata

from .strategy._optuna_support import _MINIMIZE


_WINDOWS_INVALID_FILENAME_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "CONIN$",
        "CONOUT$",
        *(f"COM{suffix}" for suffix in (*range(1, 10), "¹", "²", "³")),
        *(f"LPT{suffix}" for suffix in (*range(1, 10), "¹", "²", "³")),
    }
)


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


def _safe_objective_axis_name(name: Any) -> str:
    """Return a scorer axis safe for human-readable Pareto filenames."""
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise ValueError(
            "multi-objective scorer axes must be nonempty safe filename components"
        )
    if any(char in _WINDOWS_INVALID_FILENAME_CHARS for char in name):
        raise ValueError(
            f"multi-objective scorer axis {name!r} is not a safe filename component"
        )
    if (
        PurePosixPath(name).is_absolute()
        or PureWindowsPath(name).is_absolute()
        or bool(PureWindowsPath(name).drive)
        or any(unicodedata.category(char) == "Cc" for char in name)
        or name.endswith((".", " "))
        or name.split(".", 1)[0].upper() in _WINDOWS_RESERVED_DEVICE_NAMES
    ):
        raise ValueError(
            f"multi-objective scorer axis {name!r} is not a safe filename component"
        )
    return name


def validate_objective_axes(
    objective_axes: Sequence[Any], *, multi_objective: bool = True
) -> tuple[str, ...]:
    """Validate and preserve one authoritative ordered objective-axis tuple."""
    axes = tuple(_safe_objective_axis_name(name) for name in objective_axes)
    seen_exact: set[str] = set()
    seen_casefold: dict[str, str] = {}
    duplicates: list[str] = []
    for name in axes:
        folded = name.casefold()
        if name in seen_exact and name not in duplicates:
            duplicates.append(name)
        elif folded in seen_casefold and name not in duplicates:
            duplicates.append(name)
        seen_exact.add(name)
        seen_casefold.setdefault(folded, name)
    if duplicates:
        rendered = ", ".join(repr(name) for name in duplicates)
        raise ValueError(
            "multi-objective objective axes must be unique under Unicode "
            "case-insensitive comparison; "
            f"duplicate name(s): {rendered}"
        )
    if multi_objective and len(axes) < 2:
        raise ValueError("multi-objective objective axes must declare at least two axes")
    return axes


def ordered_objective_values(
    objectives: Mapping[str, Any], objective_axes: Sequence[str]
) -> list[float]:
    """Return an exact objective vector in scorer-authoritative axis order."""
    axes = validate_objective_axes(objective_axes)
    actual = set(objectives)
    required = set(axes)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise ValueError(
            "multi-objective results must contain exactly the scorer-declared axes; "
            f"missing={missing!r}, extra={extra!r}"
        )
    return [float(objectives[name]) for name in axes]


def objective_names(scorer: Any) -> list[str]:
    """The ordered objective-axis names of a multi-objective ``scorer``.

    Delegates to the scorer's own ``objective_names`` (a
    ``CompositeScorer`` returns its child handles ``["s0", "s1", …]``) so the
    axis order is authoritative and matches the emitted ``objectives_json`` keys.

    Args:
        scorer: A multi-objective scorer.

    Returns:
        The objective-axis names in order; ``[]`` when the scorer exposes none.

    Raises:
        ValueError: When a scorer repeats an objective name. Silently
            deduplicating would change the scorer-declared vector dimension.
    """
    names = getattr(scorer, "objective_names", None)
    raw = list(names()) if callable(names) else []
    return list(
        validate_objective_axes(raw, multi_objective=is_multi_objective(scorer))
    )


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
    objective_names(scorer)
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

    ``["minimize"] * n`` over the scorer's objective axes — every tuning
    objective is bounded cost, lower-is-better (cost convention §4). ``None`` for
    a single-objective scorer (a scalar study). A scorer marked multi-objective
    must declare at least two safe, ordered, unique axes or validation fails.

    Args:
        scorer: The tuning spec's scorer.

    Returns:
        The directions list (length ≥ 2), or ``None`` for the single-objective
        path.
    """
    if not is_multi_objective(scorer):
        return None
    names = objective_names(scorer)
    return [_MINIMIZE] * len(names)
