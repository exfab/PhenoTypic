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
from typing import TYPE_CHECKING, Any, Final, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only; never imports optuna at runtime
    import optuna

#: The actionable message shown when the ``tune`` extra is not installed.
_MISSING_OPTUNA_MSG = (
    "Optuna is required for this strategy. Install the 'tune' extra: "
    "uv sync --extras tune"
)

#: ``user_attrs`` keys carrying the non-native :class:`Trial` fields on each
#: Optuna trial, namespaced so they never collide with a user's own attrs. The
#: ONE canonical contract shared by the writer (:func:`set_trial_user_attrs`, on
#: the strategy's ``ask``/``tell`` path) and the reader
#: (``OptunaStudyStore._to_trial``), so a strategy-written trial reconstructs the
#: exact ``Trial`` record. ``PHENO_NUMBER`` is read-only legacy (the strategy
#: lets Optuna's native per-study ``trial.number`` stand in); the store still
#: honors it when an older ``add_trial`` mirror set it.
PHENO_NUMBER: Final[str] = "pheno_number"
PHENO_PARAMS: Final[str] = "pheno_params"
PHENO_TERMS: Final[str] = "pheno_terms"
PHENO_N_IMAGES: Final[str] = "pheno_n_images"
#: The multi-objective sidecar (plan §0a): the ``{objective: value}`` dict stored
#: verbatim so a reopened study restores the original objective names.
PHENO_OBJECTIVES: Final[str] = "pheno_objectives"
#: The 4.5p1 robust-eval signals: per-trial relative dispersion ``gap`` (stored
#: only when not ``None``) + the under-detection ``suspicious`` flag (stored only
#: when ``True``); an absent attr restores the neutral default.
PHENO_GAP: Final[str] = "pheno_gap"
PHENO_SUSPICIOUS: Final[str] = "pheno_suspicious"


def set_trial_user_attrs(
    trial: "optuna.trial.Trial", *, params: Any, result: Any
) -> None:
    """Stamp our non-native :class:`Trial` fields onto an in-flight Optuna trial.

    Called by :meth:`OptunaStrategy.register_result` just before ``study.tell``,
    so the strategy's native ``ask``/``tell`` trial — which already carries the
    sampler distributions — *also* carries the off-model fields
    ``OptunaStudyStore._to_trial`` reads back: the full materialized ``params``
    (including ``Fixed`` knobs the sampler never suggested), the per-image
    ``terms``, the ``n_images`` count, the multi-objective ``objectives`` sidecar,
    and the robust-eval ``gap`` / ``suspicious`` signals. This makes the strategy
    the *sole* writer of one shared study (no ``add_trial`` mirror, no phantom).

    Args:
        trial: The in-flight Optuna trial (from ``study.ask``) about to be told.
        params: The full materialized combo for this trial (the strategy's
            ``suggest`` return, ``Fixed`` constants included).
        result: The :class:`EvaluationResult` — its ``terms`` / ``n_images`` /
            ``objectives`` / ``gap`` / ``suspicious`` are read defensively
            (``getattr`` with neutral defaults) so a minimal fake still works.
    """
    trial.set_user_attr(PHENO_PARAMS, dict(params))
    trial.set_user_attr(PHENO_TERMS, dict(getattr(result, "terms", {}) or {}))
    trial.set_user_attr(PHENO_N_IMAGES, int(getattr(result, "n_images", 0) or 0))
    objectives = getattr(result, "objectives", None)
    if objectives is not None:
        trial.set_user_attr(PHENO_OBJECTIVES, dict(objectives))
    gap = getattr(result, "gap", None)
    if gap is not None:
        trial.set_user_attr(PHENO_GAP, float(gap))
    if getattr(result, "suspicious", False):
        trial.set_user_attr(PHENO_SUSPICIOUS, True)

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
