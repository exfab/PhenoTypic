"""The Optuna dependency boundary — lazy import behind the ``tune`` extra.

``import optuna`` happens **lazily inside this module's functions**, never at
package import (optuna-integration.md §10). The umbrella ``import phenotypic``
and every Grid/Random tuning path must stay Optuna-free; only an
explicitly-requested Optuna strategy (a later chunk) calls
:func:`_require_optuna`. Requesting it without the extra raises a clear,
actionable :class:`ImportError` pointing at ``uv sync --extras tune``.
"""
from __future__ import annotations

import logging
import time
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Final, Optional, Sequence, TypeVar

if TYPE_CHECKING:  # pragma: no cover - typing only; never imports optuna at runtime
    import optuna  # type: ignore[import-not-found]

_logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: Bounded transient-DB retry policy (Change 5). A shared SQLite-WAL / Postgres
#: study occasionally raises a transient ``OperationalError`` (a lock timeout, a
#: dropped connection) under concurrent ask/tell; a few short backoffs clear it.
#: A non-transient error is never retried.
_RETRY_ATTEMPTS: Final[int] = 3
_RETRY_BASE_DELAY_S: Final[float] = 0.1

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

#: Every objective in a tuning study is normalized to a bounded ``[0,1]`` **cost**
#: (lower-is-better, ``0`` perfect, ``1`` worst — cost convention §4), so a
#: single-objective study (and every axis of a multi-objective one) **minimizes**.
#: The one canonical ``"minimize"`` literal the strategy, the study store, and the
#: multi-objective inference all share.
_MINIMIZE: Final[str] = "minimize"


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
    else ``{"direction": "minimize"}`` for the single-objective scalar path. The
    one place that decides the create-study objective shape, shared by
    ``OptunaStrategy`` and ``OptunaStudyStore``.

    Args:
        directions: Per-objective directions (≥2 → multi-objective), or ``None``
            / a single axis for the scalar maximize study.

    Returns:
        ``{"directions": list(directions)}`` when multi-objective, else
        ``{"direction": _MINIMIZE}``.
    """
    if is_multi_objective_directions(directions):
        assert directions is not None  # narrowed by is_multi_objective_directions
        return {"directions": list(directions)}
    return {"direction": _MINIMIZE}


#: The pre-cutover study name. Correctness is the ``_STUDY_NAME`` bump (a legacy
#: study is never reopened); these helpers are UX (friendly message) only.
_LEGACY_STUDY_NAME: Final[str] = "tune"


def is_legacy_study_name(study_name: str) -> bool:
    """True for a pre-cutover study name (name-only — no storage probe).

    The read-only GUI Monitor uses this to classify a run from its recorded
    ``study_name`` without connecting to a (legacy) study.

    Args:
        study_name: The recorded study name to classify.

    Returns:
        ``True`` when ``study_name`` is the pre-cutover ``"tune"`` name.
    """
    return study_name == _LEGACY_STUDY_NAME


def is_legacy_study_present(
    storage: Any, *, study_name: str = _LEGACY_STUDY_NAME
) -> bool:
    """True iff a pre-cutover study exists in ``storage`` (storage-probing).

    ``storage`` is a storage URL or an Optuna storage object. ``optuna`` is
    imported function-local (the lazy boundary). ``load_study`` raises
    ``KeyError`` when the study is absent (common case → ``False``); any other
    error → ``False`` (best-effort detection must never abort study startup).

    Args:
        storage: An Optuna storage URL or storage object to probe.
        study_name: The pre-cutover study name to look for.

    Returns:
        ``True`` when a study with ``study_name`` exists in ``storage``.
    """
    import optuna

    try:
        optuna.load_study(storage=storage, study_name=study_name)
    except KeyError:
        return False
    except Exception:  # noqa: BLE001 - detection is best-effort UX
        return False
    return True


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


def fail_stale_running_trials(study: "optuna.Study") -> int:
    """Mark every still-``RUNNING`` trial in ``study`` as ``FAIL`` (Change 3).

    A worker killed mid-trial (node failure, SLURM timeout, OOM) leaves its
    in-flight trial in :obj:`~optuna.trial.TrialState.RUNNING` forever. Such a
    zombie never reaches ``COMPLETE``/``PRUNED``, so it does not consume the
    shared budget — but it lingers in the study and, on some pruners/samplers,
    skews cross-trial statistics. Reconciling it to ``FAIL`` before a fresh
    worker enters the ask/tell loop keeps the budget accounting honest and the
    study clean.

    This legacy reconciliation helper predates the ask/tell heartbeat wrapper in
    :mod:`phenotypic.tune._strategies._optuna`. It enumerates the RUNNING trials
    and tells each ``FAIL`` by trial number itself (``skip_if_finished=True``
    tolerates a concurrent worker that finalized the same trial first).
    ``import optuna`` stays lazy in the body.

    Args:
        study: The shared Optuna study to reconcile.

    Returns:
        The number of trials transitioned to ``FAIL``.
    """
    import optuna

    running = study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
    )
    failed = 0
    for trial in running:
        try:
            study.tell(
                trial.number,
                state=optuna.trial.TrialState.FAIL,
                skip_if_finished=True,
            )
            failed += 1
        except Exception:  # pragma: no cover - defensive; a racing finalize
            # A concurrent worker may have just finalized this trial; don't let
            # one reconciliation failure abort the worker's startup.
            _logger.warning(
                "could not fail stale RUNNING trial %d during reconciliation",
                trial.number,
                exc_info=True,
            )
    if failed:
        _logger.info("reconciled %d stale RUNNING trial(s) to FAIL", failed)
    return failed


def retry_on_transient_db_error(
    func: Callable[[], _T],
    *,
    trial_number: Optional[int] = None,
    attempts: int = _RETRY_ATTEMPTS,
) -> _T:
    """Call ``func`` with a bounded exponential backoff on transient DB errors.

    Wraps a single ``study.ask`` / ``study.tell`` / user-attr / append call
    (Change 5) so a transient :class:`sqlalchemy.exc.OperationalError` — a lock
    timeout or a momentarily-dropped connection against the shared SQLite-WAL /
    Postgres study — is retried up to ``attempts`` times with a doubling delay,
    rather than crashing the whole worker. Any **non**-``OperationalError`` (a
    programming bug, a constraint violation, a study-state error) propagates
    immediately — only the transient class is retried. Each retry is logged with
    the trial number when known.

    ``sqlalchemy`` is imported **function-local** (the lazy-import boundary):
    this module must stay importable without the ``tune`` extra. If SQLAlchemy
    is somehow unavailable, no error class can match, so ``func`` is called
    once and any error propagates.

    Args:
        func: The zero-argument DB operation to run (e.g. ``lambda: study.ask()``).
        trial_number: The trial number for log context, or ``None``.
        attempts: The maximum number of tries (default :data:`_RETRY_ATTEMPTS`).

    Returns:
        Whatever ``func`` returns on the first successful call.

    Raises:
        Exception: The last transient error after ``attempts`` exhausted, or any
            non-transient error on the first occurrence.
    """
    try:
        from sqlalchemy.exc import OperationalError
    except ImportError:  # pragma: no cover - sqlalchemy ships with the tune extra
        return func()

    last_exc: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            return func()
        except OperationalError as exc:
            last_exc = exc
            if attempt + 1 >= attempts:
                break
            delay = _RETRY_BASE_DELAY_S * (2**attempt)
            _logger.warning(
                "transient DB error on trial %s (attempt %d/%d); retrying in "
                "%.2fs: %s",
                trial_number if trial_number is not None else "?",
                attempt + 1,
                attempts,
                delay,
                exc,
            )
            time.sleep(delay)
    assert last_exc is not None  # only reached after an OperationalError
    raise last_exc
