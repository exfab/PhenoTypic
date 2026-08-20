"""``OptunaStudyStore`` — the StudyStore Protocol backed by a live Optuna study.

The Phase-2 alternative to the Phase-1 ``JournalStudyStore``: it persists trials
in an Optuna study that **resumes in place** — re-opening the study by name +
storage URL restores every trial *and* the sampler state, so the engine skips
the deterministic fast-forward replay (``is_resumable_in_place() → True``).

The storage object behind that study is chosen by scheme, not hardcoded: see
:func:`~phenotypic.tune._study._storage.build_optuna_storage`. A local run
defaults to SQLite-WAL under ``RDBStorage``; a ``--slurm`` fleet defaults to the
``journal://`` file backend, which is safe on the shared filesystems SQLite-WAL
is not.

``import optuna`` stays lazy inside the bodies here, preserving the package-wide
lazy-import boundary. Our :class:`~phenotypic.tune._study_store.Trial` carries
fields Optuna does not model natively (``terms`` / ``n_images`` / ``failed`` /
``pruned`` / our ``number``); these ride along in the trial's ``user_attrs`` so a
reopened study reconstructs the exact ``Trial`` records. The Optuna trial
``state`` mirrors our taxonomy: ``COMPLETE`` for a clean trial, ``PRUNED`` for an
early-stopped one, ``FAIL`` for a failed candidate.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Optional

from ..strategy._optuna_support import (
    PHENO_GAP as _ATTR_GAP,
    PHENO_N_IMAGES as _ATTR_N_IMAGES,
    PHENO_NUMBER as _ATTR_NUMBER,
    PHENO_OBJECTIVES as _ATTR_OBJECTIVES,
    PHENO_PARAMS as _ATTR_PARAMS,
    PHENO_SCORE as _ATTR_SCORE,
    PHENO_SUSPICIOUS as _ATTR_SUSPICIOUS,
    PHENO_TERMS as _ATTR_TERMS,
    is_multi_objective_directions,
    study_objective_kwargs,
)
from .._study_store import Trial
from ._storage import (
    build_optuna_storage,
    is_journal_url,
    is_sqlite_url,
    journal_path_from_url,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; never imports optuna at runtime
    from pathlib import Path

    import optuna  # type: ignore[import-not-found]

_HEARTBEAT_INTERVAL_S = 60
_GRACE_PERIOD_S = 180

#: Size at which :func:`warn_if_journal_oversized` speaks up (B4).
#:
#: **Measured growth** (optuna 4.9.0, the real ``ask`` → ``set_trial_user_attrs``
#: → ``tell`` path, 12 knobs, 6 score terms, a quarter of trials pruned after
#: three ASHA rungs — ``test_journal_growth_per_trial_stays_near_the_measured_rate``
#: pins it): **~5.9 KB and ~20 log records per trial**, linear — 200 trials
#: produced 1,168,306 B and 2,000 produced 11,707,588 B (5,841 vs 5,854 B/trial).
#:
#: So the real case — a 200-trial × 30-min campaign — is a **1.2 MB** log, and a
#: 10× overrun is 12 MB. Replay is what scales, not disk: every worker start and
#: every Monitor open re-reads the whole log (0.07 s at 200 trials, 1.0 s at
#: 2,000, linear), and per-trial write cost grows from 28 ms to 90 ms over that
#: range — 0.005% of a 30-minute evaluation. Nothing here needs doing.
#:
#: 64 MiB is therefore ~11,000 trials, ~55× the real campaign and ~9 s of replay
#: per open: not a campaign, but an output directory that has been reused across
#: many, which is the only way this log gets large. **Compaction is not offered,
#: and could not be:** ``JournalFileBackend`` addresses records by byte offset in
#: a per-process ``_log_number_offset`` map, so rewriting the file shorter would
#: leave every live worker and Monitor seeking into the middle of a record. The
#: supported remedy is a fresh output directory (or Postgres).
#:
#: The single carve-out is the *trailing* record: an unterminated stump left by
#: a failed append is truncated before the next one
#: (:func:`~phenotypic.tune._study._storage.truncate_torn_journal_tail`). That is
#: not compaction and does not weaken the argument above — ``read_logs`` deletes
#: the offset entry for a bad line, so the stump is the one record in the file
#: nobody addresses by offset.
_JOURNAL_SIZE_WARN_BYTES: int = 64 * 1024 * 1024

_logger = logging.getLogger(__name__)

#: Stamped on every freshly-created/loaded cost-convention study for
#: observability and future cutovers (spec §7 Phase 2 #2).
_CONVENTION_ATTR: str = "tune_convention"
_CONVENTION_VALUE: str = "minimize-cost-v1"
# `_LEGACY_STUDY_NAME` ("tune") and the two legacy-study helpers live in the
# shared `strategy/_optuna_support.py` so the CLI store guard and the GUI
# monitor (Phase 4) agree — import them, do not re-spell "tune" here.


def backing_file_for_url(storage_url: str) -> Optional["Path"]:
    """The local file a storage URL is backed by, or ``None`` when it has none.

    Both file-backed schemes are covered — ``journal:///…/journal.log`` and
    ``sqlite:///…/study.db`` — because both are *created on open* by the library
    that serves them, and neither Postgres nor SQLite's ``:memory:`` has a file
    to speak of.

    **The SQLite path is SQLAlchemy's, not ``urlsplit``'s.** ``urlsplit`` reads
    ``sqlite:///out/study.db`` as the absolute path ``/out/study.db``;
    SQLAlchemy reads the same URL as the **relative** path ``out/study.db``,
    because the third slash is the (empty) authority separator and everything
    after it is the database name. Taking ``urlsplit``'s answer therefore
    resolved a perfectly ordinary ``-o out`` run — ``_default_study_db_url``
    does not absolutize — to a filesystem-root path that does not exist, so the
    read-only guard refused a study that was right there. Dropping the leading
    separator is SQLAlchemy's actual rule and also round-trips the four-slash
    absolute form and a Windows drive letter.

    A relative result stays relative: it resolves against the *reader's* cwd,
    which for the GUI is not necessarily the cwd the run was launched from.
    That ambiguity is the URL's, not this function's — ``journal://`` refuses
    relative paths outright for the same reason (:func:`journal_url_for_path`).

    Args:
        storage_url: A resolved tune storage URL.

    Returns:
        The backing path, or ``None`` for a server-backed or in-memory URL.

    Examples:
        >>> backing_file_for_url("journal:///runs/out/journal.log").as_posix()
        '/runs/out/journal.log'
        >>> backing_file_for_url("sqlite:///runs/out/study.db").as_posix()
        'runs/out/study.db'
        >>> backing_file_for_url("sqlite:////runs/out/study.db").as_posix()
        '/runs/out/study.db'
        >>> backing_file_for_url("postgresql+psycopg://host/db") is None
        True
        >>> backing_file_for_url("sqlite:///:memory:") is None
        True
    """
    from pathlib import Path
    from urllib.parse import urlsplit

    if is_journal_url(storage_url):
        return journal_path_from_url(storage_url)
    if is_sqlite_url(storage_url):
        # ``[1:]`` drops the authority separator, not a path component: see the
        # docstring — this is SQLAlchemy's own reading of the database name.
        database = urlsplit(storage_url).path[1:]
        if not database or database.endswith(":memory:"):
            return None
        return Path(database)
    return None


def require_existing_backing_store(storage_url: str) -> None:
    """Raise unless a file-backed ``storage_url`` already exists (B2).

    The ``create=False`` open is documented as read-only, but nothing about it
    was: ``JournalFileBackend.__init__`` ``open(path, "ab")``-s its log into
    existence, and SQLAlchemy creates a missing SQLite file on connect — so
    pointing the GUI Monitor at a run that has not started yet **manufactured**
    ``.pht-tune-cache/journal.log`` (and its parent tree) in the user's output
    directory, then failed to load the study inside it. An absent study then
    read as a present-but-empty one to anything checking the file, which is why
    a file-existence assertion is not evidence a study exists.

    Server-backed and in-memory URLs are untouched: there is no file to
    accidentally create, and probing a Postgres host for existence is the
    connect this guard runs *before*.

    Args:
        storage_url: The resolved tune storage URL being opened read-only.

    Raises:
        FileNotFoundError: When the URL is file-backed and the file is absent.
            The caller degrades (the Monitor falls back to the parquet journal);
            it is raised rather than returned so the read-only open has exactly
            one failure channel.
    """
    backing = backing_file_for_url(storage_url)
    if backing is not None and not backing.exists():
        raise FileNotFoundError(str(backing))


def warn_if_journal_oversized(storage_url: str) -> None:
    """Log once per open when ``journal.log`` has grown past the sane bound (B4).

    ``JournalStorage`` is append-only and **never compacts** — see
    :data:`_JOURNAL_SIZE_WARN_BYTES` for the measured growth model and for why
    compaction is not offered rather than merely unimplemented. Any failure to
    stat is swallowed: this is observability on the open path, never a reason an
    open fails.

    Args:
        storage_url: The resolved tune storage URL that was just opened.
    """
    if not is_journal_url(storage_url):
        return
    try:
        size = journal_path_from_url(storage_url).stat().st_size
    except OSError:  # pragma: no cover - the open that just succeeded stat-ed it
        return
    if size < _JOURNAL_SIZE_WARN_BYTES:
        return
    _logger.warning(
        "the journal study log is %.1f MiB (%s); it is append-only and never "
        "compacts, so every worker and Monitor poll replays all of it on open. "
        "Start the next campaign in a fresh output directory (or use Postgres) "
        "rather than reusing this one.",
        size / (1024 * 1024),
        journal_path_from_url(storage_url),
    )


class OptunaStudyStore:
    """A :class:`StudyStore` over a persistent, resumable Optuna study.

    Args:
        storage_url: The Optuna storage URL — ``sqlite:///path/study.db``, a
            ``postgresql+psycopg://...`` URL, or ``journal:///path/journal.log``
            for the NFS-safe file backend a distributed run defaults to. The
            scheme is dispatched by :func:`build_optuna_storage`; SQLite URLs
            are additionally switched to WAL journal mode for concurrent
            readers/writers (a no-op concept for the other two backends).
        study_name: The study name; re-opening with the same name + URL restores
            the persisted trials and sampler state.
        directions: Per-objective directions for a multi-objective study; ``None``
            → a single-objective ``minimize`` (cost) study.
        create: When ``True`` (the engine path), create the study if it is
            missing. When ``False`` (read-only monitor path), load only an
            existing study.
    """

    def __init__(
        self,
        *,
        storage_url: str,
        study_name: str,
        directions: Optional[list[str]] = None,
        create: bool = True,
    ) -> None:
        import optuna

        self._storage_url = storage_url
        self._study_name = study_name
        self._multi_objective = is_multi_objective_directions(directions)

        if create:
            storage = build_optuna_storage(
                storage_url,
                heartbeat_interval=_HEARTBEAT_INTERVAL_S,
                grace_period=_GRACE_PERIOD_S,
            )
            if is_sqlite_url(storage_url):
                self._enable_sqlite_wal(storage)

            # UX-only (correctness is the _STUDY_NAME bump): if a pre-cutover
            # "tune" study still sits in this storage, it cannot be resumed under
            # the cost convention — say so with an actionable message instead of
            # silently starting fresh beside it.
            self._warn_if_legacy_study_present(storage)

            create_kwargs: dict[str, Any] = {
                "storage": storage,
                "study_name": study_name,
                "load_if_exists": True,
                **study_objective_kwargs(directions),
            }
            self._study = optuna.create_study(**create_kwargs)
            self._study.set_user_attr(_CONVENTION_ATTR, _CONVENTION_VALUE)
        else:
            # A read-only open must not materialize what it claims only to
            # observe — the guard runs BEFORE any storage is built.
            require_existing_backing_store(storage_url)
            # Dispatch here too: `load_study` resolves a storage STRING through
            # the same RDB-only resolver, so a `journal://` URL would die on
            # `NoSuchModuleError` instead of loading. Hand it the built object.
            self._study = optuna.load_study(
                storage=build_optuna_storage(storage_url),
                study_name=study_name,
            )
        warn_if_journal_oversized(storage_url)

    @property
    def study(self) -> "optuna.Study":
        """The live Optuna study object (the strategy reuses this ONE handle).

        Exposed so :meth:`OptunaConfig.build` can hand the strategy the study the
        store already created (and whose RDB schema it materialized), instead of
        the strategy opening a **second** ``create_study(load_if_exists=True)`` on
        the same URL + name. The strategy re-attaches its own sampler/pruner to
        this object (both live on the in-memory :class:`optuna.Study`, not in
        storage), so there is one handle, one schema-create, and the sampler the
        run asked for.
        """
        return self._study

    @property
    def study_name(self) -> str:
        """The shared study name (so the strategy can bind to this same study)."""
        return self._study_name

    @property
    def storage_url(self) -> str:
        """The storage URL backing this study (the strategy binds to the same one)."""
        return self._storage_url

    @staticmethod
    def _enable_sqlite_wal(storage: Any) -> None:
        """Switch the SQLite database to WAL journal mode (persistent property).

        WAL lets the distributed ask-and-tell workers read and write the shared
        study concurrently (optuna-integration §7). Run once via the engine — WAL
        is a persistent database property, so subsequent opens inherit it.
        """
        from sqlalchemy import text

        with storage.engine.begin() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))

    def _warn_if_legacy_study_present(self, storage: Any) -> None:
        """Log an actionable note when a pre-cutover ``"tune"`` study exists.

        UX only — the ``_STUDY_NAME`` bump already makes the silent direction
        mismatch impossible (optuna ``load_if_exists`` would otherwise keep the
        legacy ``maximize`` direction, verified 4.9.0). A pre-cutover study cannot
        be resumed under the cost convention; this points the user at a fresh run.
        Delegates the probe to the shared :func:`is_legacy_study_present`.
        """
        from phenotypic.tune.strategy._optuna_support import (
            _LEGACY_STUDY_NAME,
            is_legacy_study_present,
        )

        if self._study_name == _LEGACY_STUDY_NAME:
            return  # never warn about ourselves
        if is_legacy_study_present(storage):
            _logger.warning(
                "a pre-cutover %r study is present in this storage; it cannot be "
                "resumed under the minimize-cost convention. Starting a fresh %r "
                "study beside it (or use a new output dir).",
                _LEGACY_STUDY_NAME,
                self._study_name,
            )

    # -- writes ---------------------------------------------------------------

    def append(self, trial: Trial) -> None:
        """Record one completed :class:`Trial` into the study.

        The trial is added as a frozen Optuna trial whose ``state`` mirrors the
        ``failed`` / ``pruned`` taxonomy and whose ``user_attrs`` carry the
        non-native fields (our ``number``, ``params``, ``terms``, ``n_images``,
        the multi-objective ``objectives`` sidecar, and the 4.5p1 robust-eval
        ``gap`` / ``suspicious`` signals — each stored only when non-neutral).
        ``params`` /
        ``distributions`` are left empty — the search dimensions live in
        ``user_attrs`` so ``add_trial`` needs no distribution metadata. On a
        multi-objective study the trial's ``values`` are the objective vector (in
        the study's ``directions`` order) so ``study.best_trials`` computes the
        Pareto front natively; a single-objective study tells the scalar ``value``.
        """
        import optuna

        if trial.failed:
            state = optuna.trial.TrialState.FAIL
        elif trial.pruned:
            state = optuna.trial.TrialState.PRUNED
        else:
            state = optuna.trial.TrialState.COMPLETE

        user_attrs: dict[str, Any] = {
            _ATTR_NUMBER: trial.number,
            _ATTR_PARAMS: trial.params,
            _ATTR_TERMS: trial.terms,
            _ATTR_N_IMAGES: trial.n_images,
        }
        if trial.objectives is not None:
            user_attrs[_ATTR_OBJECTIVES] = dict(trial.objectives)
        if trial.gap is not None:
            user_attrs[_ATTR_GAP] = float(trial.gap)
        if trial.suspicious:
            user_attrs[_ATTR_SUSPICIOUS] = True

        # A FAIL/PRUNED trial may legitimately carry no value/values; a COMPLETE
        # multi-objective trial carries the per-objective vector Optuna's Pareto
        # ranking reads, a single-objective one the scalar score ``best`` ranks by.
        create_kwargs: dict[str, Any] = {
            "state": state,
            "params": {},
            "distributions": {},
            "user_attrs": user_attrs,
        }
        if trial.failed:
            create_kwargs["values" if self._multi_objective else "value"] = None
        elif self._multi_objective and trial.objectives is not None:
            create_kwargs["values"] = [
                float(v) for v in trial.objectives.values()
            ]
        else:
            create_kwargs["value"] = float(trial.score)
        frozen = optuna.trial.create_trial(**create_kwargs)
        self._study.add_trial(frozen)

    # -- reads ----------------------------------------------------------------

    def _to_trial(self, frozen: "optuna.trial.FrozenTrial") -> Trial:
        """Reconstruct our :class:`Trial` from an Optuna frozen trial.

        Restores the multi-objective ``objectives`` sidecar **and** the 4.5p1
        robust-eval ``gap`` / ``suspicious`` signals from ``user_attrs`` (an
        absent attr restores the neutral default); a multi-objective trial's
        scalar ``score`` is the mean of its objectives (the same projection the
        Evaluator applies), since a multi-objective Optuna trial carries no
        scalar ``value``.

        The cost is resolved in three steps, and the order matters:

        1. the native ``value`` / ``values`` — authoritative for a ``COMPLETE``
           trial, and the only thing the ``append`` mirror writes;
        2. the ``PHENO_SCORE`` sidecar — a ``PRUNED``/``FAIL`` trial is told with
           a state and **no** value, so its real cost lives only here;
        3. :data:`math.inf` — a trial with no cost anywhere (a ``RUNNING``
           orphan, or a pre-sidecar ``PRUNED`` row) is *unknown*, not perfect.
           Under the minimize-cost convention ``0.0`` is the **best possible**
           score, so substituting it made a never-evaluated trial outrank every
           real one; ``inf`` is the only substitution that can never win.
        """
        import optuna

        attrs = frozen.user_attrs
        failed = frozen.state == optuna.trial.TrialState.FAIL
        pruned = frozen.state == optuna.trial.TrialState.PRUNED
        raw_objectives = attrs.get(_ATTR_OBJECTIVES)
        objectives = (
            {str(k): float(v) for k, v in raw_objectives.items()}
            if isinstance(raw_objectives, dict)
            else None
        )
        # ``frozen.value`` raises on a multi-objective study (read ``values``);
        # the scalar ``score`` is then the mean of the objectives (the same
        # projection the Evaluator applies).
        raw_score = attrs.get(_ATTR_SCORE)
        if self._multi_objective and objectives:
            score = float(sum(objectives.values()) / len(objectives))
        elif not self._multi_objective and frozen.value is not None:
            score = float(frozen.value)
        elif raw_score is not None:
            score = float(raw_score)
        else:
            score = math.inf
        raw_gap = attrs.get(_ATTR_GAP)
        gap = float(raw_gap) if raw_gap is not None else None
        return Trial(
            number=int(attrs.get(_ATTR_NUMBER, frozen.number)),
            params=dict(attrs.get(_ATTR_PARAMS, {})),
            score=score,
            terms=dict(attrs.get(_ATTR_TERMS, {})),
            n_images=int(attrs.get(_ATTR_N_IMAGES, 0)),
            objectives=objectives,
            failed=failed,
            pruned=pruned,
            gap=gap,
            suspicious=bool(attrs.get(_ATTR_SUSPICIOUS, False)),
        )

    @property
    def trials(self) -> list[Trial]:
        """Every trial in the study, reconstructed as :class:`Trial` (ordered).

        Includes the **non-terminal** ones — a ``RUNNING`` trial belonging to a
        live worker, or the orphan a Slurm-killed worker left behind. That is
        deliberate: this is the honest "what the study holds" view, used for
        reporting how much work is in flight. Anything that ranks or counts
        *finished* work must use :meth:`terminal_trials` instead.
        """
        frozen = self._study.get_trials(deepcopy=False)
        return [self._to_trial(f) for f in frozen]

    def terminal_trials(self) -> list[Trial]:
        """The trials that will never change again: ``COMPLETE|PRUNED|FAIL``.

        Excludes ``RUNNING`` / ``WAITING``. A trial in either of those states has
        no result yet, so it can be neither a winner nor evidence of progress —
        and on the distributed path it may never acquire one, because a worker
        killed at its Slurm walltime leaves its trial ``RUNNING`` forever.
        """
        import optuna

        frozen = self._study.get_trials(
            deepcopy=False,
            states=(
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
                optuna.trial.TrialState.FAIL,
            ),
        )
        return [self._to_trial(f) for f in frozen]

    def __len__(self) -> int:
        return len(self._study.get_trials(deepcopy=False))

    def best(self) -> Optional[Trial]:
        """The finished, non-failed trial with the lowest cost, or ``None``.

        Ranks :meth:`terminal_trials`, never the raw :attr:`trials`: an
        un-told ``RUNNING`` trial is not ``failed``, so ranking the raw list let
        an orphaned worker's trial win the study outright.

        A trial whose cost is not finite is skipped for the same reason. That is
        every trial ``_to_trial`` had to fall back to ``inf`` for — chiefly a
        ``PRUNED`` row written before the ``PHENO_SCORE`` sidecar existed, whose
        real partial-fidelity cost is simply not recoverable. It is unrankable,
        not perfect.
        """
        valid = [
            t
            for t in self.terminal_trials()
            if not t.failed and math.isfinite(t.score)
        ]
        if not valid:
            return None
        return min(valid, key=lambda t: t.score)

    def is_resumable_in_place(self) -> bool:
        """Always ``True``: the Optuna storage restores trials + sampler state."""
        return True

    def completed_count(self) -> int:
        """The number of budget-consuming trials: ``COMPLETE + PRUNED``.

        Exactly the quantity :meth:`OptunaStrategy.is_exhausted` compares against
        ``n_trials`` — failed trials do not consume the budget, and neither do
        in-flight ones. Counting the raw :attr:`trials` instead over-reported
        progress by ``#failed + #in-flight``, which is how a budget gate could
        open on a fleet that still had a dozen trials to run.
        """
        return sum(1 for t in self.terminal_trials() if not t.failed)

    def param_importances(self) -> Optional[dict[str, float]]:
        """The study's fANOVA importances, or ``None`` when unavailable.

        Delegates to ``optuna.importance.get_param_importances`` (default fANOVA
        evaluator), whose variance decomposition attributes each parameter's main
        **and** interaction contribution. Returns ``None`` — so the screening
        layer falls back to its RandomForest + permutation estimate — when the
        study has no native sampler dimensions (the :meth:`append` path stores
        params in ``user_attrs``, leaving native ``params`` empty) or fewer than
        two completed trials, both of which make fANOVA degenerate.

        ``import optuna`` stays lazy in the body (the lazy-import boundary).
        """
        import optuna

        try:
            importances = optuna.importance.get_param_importances(self._study)
        except (ValueError, RuntimeError):
            # Too few trials / single-objective requirement unmet → no model.
            return None
        if not importances:
            # No native dimensions (append-path trials carry params off-band).
            return None
        return {key: float(value) for key, value in importances.items()}

    def pareto_front(self) -> list[Trial]:
        """The non-dominated trials by their ``objectives`` sidecar (plan §0a).

        A multi-objective study exposes its native Pareto front via
        ``study.best_trials`` (Optuna's own non-domination over the per-objective
        ``values``); each is reconstructed into our :class:`Trial` (objectives
        restored from ``user_attrs``). A single-objective study has no
        multi-objective front, so this returns ``[]`` (the back-compat lock).
        """
        if not self._multi_objective:
            return []
        return [self._to_trial(f) for f in self._study.best_trials]

    def knee_point(self, front: list[Trial]) -> Optional[Trial]:
        """The ``front`` trial at max perpendicular distance to the chord.

        Delegates to the store-agnostic :func:`knee_point_of` so the Optuna and
        journal backends pick the same knee from the same front; ``None`` for an
        empty front.
        """
        from ._pareto import knee_point_of

        return knee_point_of(front)
