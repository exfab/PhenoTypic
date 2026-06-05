"""The Optuna dependency boundary — lazy import behind the ``tune`` extra.

``import optuna`` happens **lazily inside this module's functions**, never at
package import (optuna-integration.md §10). The umbrella ``import phenotypic``
and every Grid/Random tuning path must stay Optuna-free; only an
explicitly-requested Optuna strategy (a later chunk) calls
:func:`_require_optuna`. Requesting it without the extra raises a clear,
actionable :class:`ImportError` pointing at ``uv sync --extras tune``.
"""
from __future__ import annotations

from types import ModuleType
from typing import Any, Final, Optional, Sequence

#: The actionable message shown when the ``tune`` extra is not installed.
_MISSING_OPTUNA_MSG = (
    "Optuna is required for this strategy. Install the 'tune' extra: "
    "uv sync --extras tune"
)

#: Every objective in a tuning study is normalized higher-is-better
#: (robust-eval §5), so a single-objective study (and every axis of a
#: multi-objective one) maximizes. The one canonical ``"maximize"`` literal the
#: strategy, the study store, and the multi-objective inference all share.
_MAXIMIZE: Final[str] = "maximize"


def is_multi_objective_directions(directions: Optional[Sequence[str]]) -> bool:
    """Whether ``directions`` describes a multi-objective (≥2 axes) study.

    Args:
        directions: Per-objective Optuna directions, or ``None`` for the
            single-objective path.

    Returns:
        ``True`` when ``directions`` carries two or more axes; ``False`` for
        ``None`` or a single axis (a degenerate one-axis "multi-objective"
        study is treated as scalar).
    """
    return directions is not None and len(directions) > 1


def study_objective_kwargs(
    directions: Optional[Sequence[str]],
) -> dict[str, Any]:
    """The ``optuna.create_study`` objective kwargs for ``directions``.

    Maps the per-objective directions onto the mutually-exclusive ``create_study``
    objective argument: ``{"directions": [...]}`` for a multi-objective study,
    else ``{"direction": "maximize"}`` for the single-objective scalar path. The
    one place that decides the create-study objective shape, shared by
    ``OptunaStrategy`` and ``OptunaStudyStore``.

    Args:
        directions: Per-objective directions (≥2 → multi-objective), or ``None``
            / a single axis for the scalar maximize study.

    Returns:
        ``{"directions": list(directions)}`` when multi-objective, else
        ``{"direction": _MAXIMIZE}``.
    """
    if is_multi_objective_directions(directions):
        assert directions is not None  # narrowed by is_multi_objective_directions
        return {"directions": list(directions)}
    return {"direction": _MAXIMIZE}


def _require_optuna() -> ModuleType:
    """Import and return the ``optuna`` module, or raise an actionable error.

    The import is deliberately inside the function body so importing this
    module (and therefore ``phenotypic.tune``) never pulls in Optuna; only an
    actual call resolves the dependency.

    Returns:
        The imported ``optuna`` module.

    Raises:
        ImportError: If Optuna is not installed, with a message pointing at
            ``uv sync --extras tune``.
    """
    try:
        import optuna  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(_MISSING_OPTUNA_MSG) from exc
    return optuna
