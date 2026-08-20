"""The Optuna dependency boundary — lazy import behind the ``tune`` extra.

``import optuna`` happens **lazily inside this module's functions**, never at
package import (optuna-integration.md §10). The umbrella ``import phenotypic``
and every Grid/Random tuning path must stay Optuna-free; only an
explicitly-requested Optuna strategy (a later chunk) calls
:func:`_require_optuna`. Requesting it without the extra raises a clear,
actionable :class:`ImportError` pointing at ``uv sync --extras tune``.
"""
from __future__ import annotations

import errno
import json
import logging
import time
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Final, Optional, Sequence, TypeVar

if TYPE_CHECKING:  # pragma: no cover - typing only; never imports optuna at runtime
    import optuna  # type: ignore[import-not-found]

_logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: Bounded transient-storage retry policy (Change 5). A shared study —
#: SQLite-WAL / Postgres under ``RDBStorage``, or the ``journal://`` file
#: backend a ``--slurm`` fleet defaults to — occasionally fails a single
#: ask/tell for a reason that clears on its own; a few short backoffs absorb it.
#: A non-transient error is never retried. What counts as transient is
#: :func:`is_transient_storage_error`.
_RETRY_ATTEMPTS: Final[int] = 3
_RETRY_BASE_DELAY_S: Final[float] = 0.1

#: ``errno`` names a shared-filesystem call can fail with **transiently** — the
#: journal backend's half of :func:`is_transient_storage_error`. Resolved
#: through :mod:`errno` by name because not every platform defines every one
#: (the cross-platform rule); a name this interpreter lacks is simply dropped.
#:
#: Every member means "the filesystem could not serve this call *right now*":
#: a GPFS/NFS server hiccup (``EIO``, ``EREMOTEIO``, ``ETIMEDOUT``,
#: ``ECONNRESET``), a mount re-exported under us (``ESTALE``), a momentarily
#: unavailable resource or lock (``EAGAIN``/``EWOULDBLOCK``, ``EBUSY``,
#: ``ENOLCK``), or a signal landing mid-syscall (``EINTR``).
#:
#: Deliberately **absent**, because retrying them is a busy-wait on a condition
#: only an operator can clear: ``ENOSPC`` / ``EDQUOT`` (the filesystem or quota
#: is full), ``EACCES`` / ``EPERM`` / ``EROFS`` (the run cannot write there),
#: and ``ENOENT`` (the output directory was removed out from under the run).
_TRANSIENT_ERRNO_NAMES: Final[tuple[str, ...]] = (
    "EIO",
    "ESTALE",
    "EAGAIN",
    "EWOULDBLOCK",
    "EBUSY",
    "EINTR",
    "ENOLCK",
    "ETIMEDOUT",
    "ECONNRESET",
    "EREMOTEIO",
)

#: The resolved ``errno`` values of :data:`_TRANSIENT_ERRNO_NAMES`.
_TRANSIENT_ERRNOS: Final[frozenset[int]] = frozenset(
    value
    for value in (getattr(errno, name, None) for name in _TRANSIENT_ERRNO_NAMES)
    if isinstance(value, int)
)

#: The **exact** message optuna's journal reader raises for a line carrying no
#: trailing newline — a record another worker was still appending when the
#: reader reached it (``optuna/storages/journal/_file.py``,
#: ``JournalFileBackend.read_logs``, optuna 4.9.0).
#:
#: A newline-less line is by definition the file's last, so the reader only
#: *stores* the error there; it escapes when the writer's remaining bytes land
#: before the reader's next ``readline``, turning the tail into another line and
#: re-raising what was stored. Its sibling outcome — the torn record joined to
#: the next one, yielding invalid JSON — is a :class:`json.JSONDecodeError`, and
#: is the arm reachable without racing the writer.
#:
#: Optuna raises a bare :class:`ValueError` with no distinguishing type, so the
#: message is the only signal; matching a bare ``ValueError`` instead would
#: swallow every programming error on the same path.
#: ``test_journal_torn_line_message_is_still_optunas`` pins this string against
#: the installed optuna's source, so an upstream rewording fails a test rather
#: than silently switching the retry off.
_JOURNAL_TORN_LINE_MSG: Final[str] = "Invalid log format."

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
#: The evaluated scalar cost, stamped **in addition to** the native
#: ``trial.value``. Optuna carries no value for a trial told as ``PRUNED`` or
#: ``FAIL`` (``study.tell(trial, state=...)`` takes no ``values``), so without
#: this sidecar the store reads such a trial's cost back as "missing" and has to
#: substitute one — the phantom-winner hazard the ``0.0`` substitution created
#: (a pruned trial's cost is real, partial-fidelity data, not an absence).
PHENO_SCORE: Final[str] = "pheno_score"


def set_trial_user_attrs(
    trial: "optuna.trial.Trial", *, params: Any, result: Any
) -> None:
    """Stamp our non-native :class:`Trial` fields onto an in-flight Optuna trial.

    Called by :meth:`OptunaStrategy.register_result` just before ``study.tell``,
    so the strategy's native ``ask``/``tell`` trial — which already carries the
    sampler distributions — *also* carries the off-model fields
    ``OptunaStudyStore._to_trial`` reads back: the full materialized ``params``
    (including ``Fixed`` knobs the sampler never suggested), the per-image
    ``terms``, the ``n_images`` count, the scalar ``score``, the multi-objective
    ``objectives`` sidecar, and the robust-eval ``gap`` / ``suspicious`` signals.
    This makes the strategy the *sole* writer of one shared study (no
    ``add_trial`` mirror, no phantom).

    The ``score`` sidecar exists because the next call is often
    ``study.tell(trial, state=PRUNED|FAIL)``, which stores **no** ``value``: an
    early-stopped trial's real partial-fidelity cost would otherwise be
    unrecoverable on read-back (see :data:`PHENO_SCORE`).

    Args:
        trial: The in-flight Optuna trial (from ``study.ask``) about to be told.
        params: The full materialized combo for this trial (the strategy's
            ``suggest`` return, ``Fixed`` constants included).
        result: The :class:`EvaluationResult` — its ``score`` / ``terms`` /
            ``n_images`` / ``objectives`` / ``gap`` / ``suspicious`` are read
            defensively (``getattr`` with neutral defaults) so a minimal fake
            still works.
    """
    trial.set_user_attr(PHENO_PARAMS, dict(params))
    trial.set_user_attr(PHENO_TERMS, dict(getattr(result, "terms", {}) or {}))
    score = getattr(result, "score", None)
    if score is not None:
        # Stamped for the PRUNED/FAIL trials Optuna stores with no ``value`` at
        # all; a COMPLETE trial's native ``value`` still wins on read-back.
        trial.set_user_attr(PHENO_SCORE, float(score))
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
            / a single axis for the scalar minimize study.

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
    :mod:`phenotypic.tune.strategy._optuna`. It enumerates the RUNNING trials
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


def is_transient_storage_error(exc: BaseException) -> bool:
    """Whether ``exc`` is a study-storage failure worth retrying.

    Classifies by the **exception**, not by the configured backend, because the
    two backends' transient classes are disjoint and a caller cannot then wire
    the wrong predicate to the wrong storage:

    * **RDB** (SQLite-WAL / Postgres) — :class:`sqlalchemy.exc.OperationalError`:
      a lock timeout, a momentarily-dropped connection. SQLAlchemy is imported
      function-local (the lazy-import boundary); when it is absent no error can
      match that arm and only the journal arms apply.
    * **journal** — an :class:`OSError` whose ``errno`` is in
      :data:`_TRANSIENT_ERRNOS` (a GPFS/NFS hiccup on the ``symlink`` /
      ``open`` / ``write`` / ``fsync`` the append path makes), or a torn read of
      a record another worker was mid-append on: a
      :class:`json.JSONDecodeError`, or optuna's bare
      :data:`_JOURNAL_TORN_LINE_MSG` ``ValueError``. Re-reading is the one
      unambiguously safe retry here — the log is append-only, so the second read
      sees the completed record.

    Lock *contention* never reaches this predicate at all:
    ``JournalFileSymlinkLock.acquire`` blocks with its own doubling backoff and,
    after a 30 s grace period, forcibly steals a lock nobody is refreshing. So
    the journal arms above are about the filesystem failing a call, not about
    workers queueing behind one another.

    Two journal failures are deliberately **not** transient:

    * ``RuntimeError("Error: did not possess lock")`` from
      ``JournalFileSymlinkLock.release``. It is raised from the lock context
      manager's ``finally``, so it is **ambiguous**: it means another worker
      forcibly stole the lock after the 30 s grace period, and the append it
      guarded may have already landed (the release runs after a *successful*
      write just as it does after a failed one). Retrying the ask/tell would
      then duplicate the record. A worker that dies here is recoverable; a
      study that silently doubled a record is not.
    * A non-transient ``errno`` — full disk, permissions, a deleted output
      directory. See :data:`_TRANSIENT_ERRNO_NAMES` for the full reasoning.

    Args:
        exc: The exception raised by a single storage operation.

    Returns:
        ``True`` when a bounded retry is worth attempting.

    Examples:
        >>> import errno, json
        >>> is_transient_storage_error(OSError(errno.EIO, "Input/output error"))
        True
        >>> is_transient_storage_error(OSError(errno.ENOSPC, "No space left"))
        False
        >>> is_transient_storage_error(ValueError("Invalid log format."))
        True
        >>> is_transient_storage_error(RuntimeError("Error: did not possess lock"))
        False
    """
    try:
        from sqlalchemy.exc import OperationalError
    except ImportError:  # pragma: no cover - sqlalchemy ships with the tune extra
        pass
    else:
        if isinstance(exc, OperationalError):
            return True

    if isinstance(exc, OSError):
        return exc.errno in _TRANSIENT_ERRNOS
    if isinstance(exc, json.JSONDecodeError):
        return True
    # `JSONDecodeError` is itself a `ValueError`, so this arm is reached only by
    # a bare one — match optuna's exact torn-line message, never the type.
    return type(exc) is ValueError and str(exc) == _JOURNAL_TORN_LINE_MSG


def retry_on_transient_db_error(
    func: Callable[[], _T],
    *,
    trial_number: Optional[int] = None,
    attempts: int = _RETRY_ATTEMPTS,
) -> _T:
    """Call ``func`` with a bounded exponential backoff on transient failures.

    Wraps a single ``study.ask`` / ``study.tell`` / user-attr / append call
    (Change 5) so a failure :func:`is_transient_storage_error` recognizes is
    retried up to ``attempts`` times with a doubling delay, rather than crashing
    the whole worker. Anything else (a programming bug, a constraint violation,
    a study-state error, a full disk) propagates immediately. Each retry is
    logged with the trial number when known.

    **Backend-aware by exception class.** Until P1 this matched
    ``sqlalchemy.exc.OperationalError`` alone, which made it inert the moment
    ``journal://`` became the ``--slurm`` default — a journal backend raises no
    such class, so every filesystem hiccup killed a worker on first occurrence.
    That resilience matters more on the journal path than it ever did on an RDB:
    the journal backend supports no heartbeat, so ``fail_stale_trials`` cannot
    reclaim what a dead worker left ``RUNNING`` (see ``build_optuna_storage``).

    **Retrying is not idempotent, and that is a deliberate trade.** A retried
    ``ask`` can leave an extra ``RUNNING`` trial behind, and a retry after a
    partial append can leave a torn record in the log. Both are survivable —
    a non-terminal trial is excluded from winner selection and from the budget
    gate, and optuna's reader tolerates a torn *trailing* line — whereas losing
    the worker costs the rest of its Slurm walltime. The one case where the
    duplicate would be worse than the crash (the ambiguous stolen-lock
    ``RuntimeError``) is excluded by the predicate, not by this loop.

    Args:
        func: The zero-argument storage operation to run (e.g.
            ``lambda: study.ask()``).
        trial_number: The trial number for log context, or ``None``.
        attempts: The maximum number of tries (default :data:`_RETRY_ATTEMPTS`).

    Returns:
        Whatever ``func`` returns on the first successful call.

    Raises:
        Exception: The last transient error after ``attempts`` exhausted, or any
            non-transient error on the first occurrence.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            return func()
        except Exception as exc:
            if not is_transient_storage_error(exc):
                raise
            last_exc = exc
            if attempt + 1 >= attempts:
                break
            delay = _RETRY_BASE_DELAY_S * (2**attempt)
            _logger.warning(
                "transient study-storage error on trial %s (attempt %d/%d); "
                "retrying in %.2fs: %r",
                trial_number if trial_number is not None else "?",
                attempt + 1,
                attempts,
                delay,
                exc,
            )
            time.sleep(delay)
    assert last_exc is not None  # only reached after a transient error
    raise last_exc
