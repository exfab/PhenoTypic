"""``OptunaStudyStore`` — the StudyStore Protocol backed by a live Optuna study.

The Phase-2 alternative to the Phase-1 ``JournalStudyStore`` persists trials in
an Optuna Journal, SQLite-WAL, or external RDB study that **resumes in place** —
re-opening by study name + storage URL restores every trial *and* the sampler
state, so the engine skips the deterministic fast-forward replay
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

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

from .._multi_objective import ordered_objective_values, validate_objective_axes
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
    is_sqlite_url,
    require_existing_optuna_storage,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; never imports optuna at runtime
    import optuna  # type: ignore[import-not-found]

_HEARTBEAT_INTERVAL_S = 60
_GRACE_PERIOD_S = 180

_logger = logging.getLogger(__name__)

#: Stamped on every freshly-created/loaded cost-convention study for
#: observability and future cutovers (spec §7 Phase 2 #2).
_CONVENTION_ATTR: str = "tune_convention"
_CONVENTION_VALUE: str = "minimize-cost-v1"
# `_LEGACY_STUDY_NAME` ("tune") and the two legacy-study helpers live in the
# shared `strategy/_optuna_support.py` so the CLI store guard and the GUI
# monitor (Phase 4) agree — import them, do not re-spell "tune" here.


def _require_existing_backing_store(storage_url: str) -> None:
    """Refuse a non-creating open unless its backing Optuna storage exists."""
    require_existing_optuna_storage(storage_url)


class OptunaStudyStore:
    """A :class:`StudyStore` over a persistent, resumable Optuna study.

    Args:
        storage_url: The Optuna storage URL (``journal://``, SQLite, or an
            external RDB URL). SQLite URLs are switched to WAL for local
            concurrency; the distributed CLI rejects SQLite.
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
        objective_axes: Optional[Sequence[str]] = None,
        create: bool = True,
    ) -> None:
        import optuna

        self._storage_url = storage_url
        self._study_name = study_name
        self._multi_objective = is_multi_objective_directions(directions)
        self._directions = tuple(directions or ())
        self._objective_axes = (
            validate_objective_axes(objective_axes) if objective_axes is not None else None
        )
        if self._objective_axes is not None and (
            not self._multi_objective
            or len(self._objective_axes) != len(directions or ())
        ):
            raise ValueError(
                "objective_axes must match the multi-objective study directions"
            )

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
            self._study = optuna.load_study(
                storage=build_optuna_storage(storage_url, create=False),
                study_name=study_name,
            )

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
        multi-objective study the trial's ``values`` are the objective vector in
        the study's ``directions`` order; publication applies the shared
        COMPLETE-only selector to the persisted sidecar. A single-objective study
        stores the scalar ``value``.
        """
        import optuna

        values: Optional[list[float]] = None
        if self._multi_objective:
            if trial.objectives is None:
                if not trial.failed and not trial.pruned:
                    raise ValueError(
                        "complete multi-objective trials must contain exactly "
                        "the declared objective axes"
                    )
            elif self._objective_axes is None:
                if len(trial.objectives) != len(self._directions):
                    raise ValueError(
                        "multi-objective trials must contain exactly one value "
                        "per study direction"
                    )
                values = [float(value) for value in trial.objectives.values()]
            else:
                values = ordered_objective_values(
                    trial.objectives, self._objective_axes
                )

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
        elif self._multi_objective:
            create_kwargs["values"] = values
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
            score = (
                float(sum(objectives.values()) / len(objectives))
                if objectives
                else math.inf
            )
        elif frozen.value is not None:
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
        """The study's trials reconstructed as :class:`Trial` records (ordered)."""
        frozen = self._study.get_trials(deepcopy=False)
        return [self._to_trial(f) for f in frozen]

    def terminal_trials(self) -> list[Trial]:
        """Return completed, pruned, and failed trials, excluding in-flight work."""
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
        """The finite COMPLETE trial with the lowest cost, or ``None``."""
        valid = [
            trial
            for trial in self.terminal_trials()
            if not trial.failed
            and not trial.pruned
            and math.isfinite(trial.score)
        ]
        if not valid:
            return None
        return min(valid, key=lambda t: t.score)

    def is_resumable_in_place(self) -> bool:
        """Always ``True``: the Optuna storage restores trials + sampler state."""
        return True

    def completed_count(self) -> int:
        """The number of completed (non-failed) trials; pruned counts as done."""
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

    def pareto_front(
        self, objective_axes: Sequence[str] | None = None
    ) -> list[Trial]:
        """Return finite non-dominated COMPLETE trials on optional fixed axes."""
        validated_axes = (
            validate_objective_axes(objective_axes)
            if objective_axes is not None
            else None
        )
        if not self._multi_objective:
            return []
        from ._pareto import pareto_front_of

        return pareto_front_of(
            self.terminal_trials(), objective_axes=validated_axes
        )

    def knee_point(
        self,
        front: list[Trial],
        objective_axes: Sequence[str] | None = None,
    ) -> Optional[Trial]:
        """Return the ``front`` knee using optional fixed scorer axes."""
        from ._pareto import knee_point_of

        return knee_point_of(front, objective_axes=objective_axes)
