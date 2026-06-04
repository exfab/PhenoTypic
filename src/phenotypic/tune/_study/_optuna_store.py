"""``OptunaStudyStore`` — the StudyStore Protocol backed by a live Optuna study.

The Phase-2 alternative to the Phase-1 ``JournalStudyStore``: it persists trials
in an Optuna study (``RDBStorage``, SQLite-WAL by default) that **resumes in
place** — re-opening the study by name + storage URL restores every trial *and*
the sampler state, so the engine skips the deterministic fast-forward replay
(``is_resumable_in_place() → True``).

``import optuna`` stays lazy inside the bodies here, preserving the package-wide
lazy-import boundary. Our :class:`~phenotypic.tune._study_store.Trial` carries
fields Optuna does not model natively (``terms`` / ``n_images`` / ``failed`` /
``pruned`` / our ``number``); these ride along in the trial's ``user_attrs`` so a
reopened study reconstructs the exact ``Trial`` records. The Optuna trial
``state`` mirrors our taxonomy: ``COMPLETE`` for a clean trial, ``PRUNED`` for an
early-stopped one, ``FAIL`` for a failed candidate.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .._study_store import Trial

if TYPE_CHECKING:  # pragma: no cover - typing only; never imports optuna at runtime
    import optuna

#: ``user_attrs`` keys carrying the non-native ``Trial`` fields on each Optuna
#: trial, namespaced so they never collide with a user's own attrs.
_ATTR_NUMBER = "pheno_number"
_ATTR_PARAMS = "pheno_params"
_ATTR_TERMS = "pheno_terms"
_ATTR_N_IMAGES = "pheno_n_images"


class OptunaStudyStore:
    """A :class:`StudyStore` over a persistent, resumable Optuna study.

    Args:
        storage_url: The Optuna storage URL (``sqlite:///path/study.db`` or a
            ``postgresql+psycopg://...`` URL). SQLite URLs are switched to WAL
            journal mode for concurrent readers/writers.
        study_name: The study name; re-opening with the same name + URL restores
            the persisted trials and sampler state.
        directions: Per-objective directions for a multi-objective study; ``None``
            → a single-objective ``maximize`` study.
    """

    def __init__(
        self,
        *,
        storage_url: str,
        study_name: str,
        directions: Optional[list[str]] = None,
    ) -> None:
        import optuna

        self._storage_url = storage_url
        self._study_name = study_name
        self._multi_objective = directions is not None and len(directions) > 1

        if storage_url.startswith("sqlite"):
            self._enable_sqlite_wal(optuna, storage_url)

        create_kwargs: dict[str, Any] = {
            "storage": storage_url,
            "study_name": study_name,
            "load_if_exists": True,
        }
        if self._multi_objective:
            create_kwargs["directions"] = directions
        else:
            create_kwargs["direction"] = "maximize"
        self._study = optuna.create_study(**create_kwargs)

    @staticmethod
    def _enable_sqlite_wal(optuna: Any, storage_url: str) -> None:
        """Switch the SQLite database to WAL journal mode (persistent property).

        WAL lets the distributed ask-and-tell workers read and write the shared
        study concurrently (optuna-integration §7). Run once via the engine — WAL
        is a persistent database property, so subsequent opens inherit it.
        """
        from sqlalchemy import text

        storage = optuna.storages.RDBStorage(url=storage_url)
        with storage.engine.begin() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))

    # -- writes ---------------------------------------------------------------

    def append(self, trial: Trial) -> None:
        """Record one completed :class:`Trial` into the study.

        The trial is added as a frozen Optuna trial whose ``state`` mirrors the
        ``failed`` / ``pruned`` taxonomy and whose ``user_attrs`` carry the
        non-native fields (our ``number``, ``params``, ``terms``, ``n_images``).
        ``params``/``distributions`` are left empty — the search dimensions live
        in ``user_attrs`` so ``add_trial`` needs no distribution metadata.
        """
        import optuna

        if trial.failed:
            state = optuna.trial.TrialState.FAIL
        elif trial.pruned:
            state = optuna.trial.TrialState.PRUNED
        else:
            state = optuna.trial.TrialState.COMPLETE

        # A FAIL/PRUNED trial may legitimately carry no value; COMPLETE carries
        # the score Optuna ranks ``best`` by.
        value = None if trial.failed else float(trial.score)
        frozen = optuna.trial.create_trial(
            state=state,
            value=value,
            params={},
            distributions={},
            user_attrs={
                _ATTR_NUMBER: trial.number,
                _ATTR_PARAMS: trial.params,
                _ATTR_TERMS: trial.terms,
                _ATTR_N_IMAGES: trial.n_images,
            },
        )
        self._study.add_trial(frozen)

    # -- reads ----------------------------------------------------------------

    def _to_trial(self, frozen: "optuna.trial.FrozenTrial") -> Trial:
        """Reconstruct our :class:`Trial` from an Optuna frozen trial."""
        import optuna

        attrs = frozen.user_attrs
        failed = frozen.state == optuna.trial.TrialState.FAIL
        pruned = frozen.state == optuna.trial.TrialState.PRUNED
        score = (
            float(frozen.value)
            if frozen.value is not None
            else 0.0
        )
        return Trial(
            number=int(attrs.get(_ATTR_NUMBER, frozen.number)),
            params=dict(attrs.get(_ATTR_PARAMS, {})),
            score=score,
            terms=dict(attrs.get(_ATTR_TERMS, {})),
            n_images=int(attrs.get(_ATTR_N_IMAGES, 0)),
            failed=failed,
            pruned=pruned,
        )

    @property
    def trials(self) -> list[Trial]:
        """The study's trials reconstructed as :class:`Trial` records (ordered)."""
        frozen = self._study.get_trials(deepcopy=False)
        return [self._to_trial(f) for f in frozen]

    def __len__(self) -> int:
        return len(self._study.get_trials(deepcopy=False))

    def best(self) -> Optional[Trial]:
        """The non-failed trial with the highest score, or ``None``."""
        valid = [t for t in self.trials if not t.failed]
        if not valid:
            return None
        return max(valid, key=lambda t: t.score)

    def is_resumable_in_place(self) -> bool:
        """Always ``True``: the Optuna storage restores trials + sampler state."""
        return True

    def completed_count(self) -> int:
        """The number of completed (non-failed) trials; pruned counts as done."""
        return sum(1 for t in self.trials if not t.failed)
