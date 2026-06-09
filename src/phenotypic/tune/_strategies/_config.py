"""Serializable strategy configs; each builds its live SearchStrategy.

These are a closed set in Phase 1 (grid/random) → a discriminated union.
Phase 2 adds ``OptunaConfig``; the polymorphic-field path (engine-architecture
§6) lets the open Scorer/Strategy sets extend, but the in-spec config field uses
this union for the built-in kinds.
"""
from __future__ import annotations

import os
from abc import abstractmethod
from typing import Annotated, Any, Literal, Optional, TypeAlias, Union, get_args

from pydantic import BaseModel, ConfigDict, Field

from .._search_space import SearchSpace
from ._grid import GridStrategy
from ._protocol import SearchStrategy
from ._random import RandomStrategy

#: Closed set of strategy-config discriminator tags (reused by the union).
StrategyKind = Literal["grid", "random"]

#: The Optuna sampler roster — a closed set (never a bare ``str``). ``"tpe"`` is
#: the default (handles our mixed categorical/conditional space); ``"cmaes"`` is
#: the continuous-dominant / post-screening focused-round sampler; ``"gp"`` the
#: low-dim expensive-eval Gaussian-process sampler; ``"nsga2"`` the
#: multi-objective genetic sampler (auto-selected for a multi-objective study).
SamplerKind: TypeAlias = Literal["tpe", "cmaes", "gp", "nsga2"]

#: The :data:`SamplerKind` roster as a runtime ``frozenset`` — the single source
#: the CLI's "is this strategy name an Optuna sampler?" test reads (so the roster
#: lives once, beside the ``Literal`` it mirrors).
OPTUNA_SAMPLERS: frozenset[str] = frozenset(get_args(SamplerKind))

#: The full ``--strategy`` choice tuple: the homegrown ``grid``/``random`` plus
#: every Optuna sampler, derived from :data:`SamplerKind` so the CLI choices and
#: the sampler roster cannot drift apart.
STRATEGY_CHOICES: tuple[str, ...] = ("grid", "random", *get_args(SamplerKind))

#: Environment variable naming the Optuna storage URL when ``OptunaConfig`` is
#: built with ``storage_url=None`` (e.g. a SLURM array exporting a shared
#: Postgres/SQLite URL to every worker). Resolved inside ``build`` only — never
#: at construction — so the lazy-import boundary holds.
PHENOTYPIC_TUNE_STORAGE_URL_ENV: str = "PHENOTYPIC_TUNE_STORAGE_URL"


class StrategyConfig(BaseModel):
    """Abstract, serializable config that builds its live ``SearchStrategy``.

    A frozen value-model; concrete subclasses (``GridConfig`` / ``RandomConfig``)
    carry a ``kind`` discriminator and implement ``build``. ``StrategyConfig``
    itself cannot be instantiated (it has an abstract method).

    Args:
        seed: The RNG seed forwarded to seeded strategies; defaults to ``0``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    seed: int = 0

    @abstractmethod
    def build(
        self,
        space: SearchSpace,
        store: Optional[Any],
        *,
        directions: Optional[list[str]] = None,
    ) -> SearchStrategy:
        """Construct the live strategy for ``space``.

        Args:
            space: The search space the built strategy operates over.
            store: The study store (Phase 1d's ``StudyStore``); accepted for a
                uniform factory signature but unused by the zero-dependency
                grid/random strategies.
            directions: Per-objective Optuna ``directions`` for a multi-objective
                run (``["maximize"] * n``), inferred from the scorer by the
                engine. ``None`` → single-objective. Only the Optuna backend
                honors it; grid/random ignore it (4.8 rejects pairing them with a
                multi-objective scorer at validation, so they never receive it).
        """
        ...


class GridConfig(StrategyConfig):
    """Builds a ``GridStrategy`` (exhaustive conditional-Cartesian enumeration).

    Args:
        kind: The discriminator tag; always ``"grid"``.
    """

    kind: Literal["grid"] = "grid"

    def build(
        self,
        space: SearchSpace,
        store: Optional[Any],
        *,
        directions: Optional[list[str]] = None,
    ) -> SearchStrategy:
        # Grid enumeration is single-objective; ``directions`` is ignored (a
        # multi-objective scorer with grid is rejected at validation, 4.8).
        return GridStrategy(space)


class RandomConfig(StrategyConfig):
    """Builds a ``RandomStrategy`` (seeded random sampling).

    Args:
        kind: The discriminator tag; always ``"random"``.
        n_trials: The number of configurations to sample before exhaustion.
    """

    kind: Literal["random"] = "random"
    n_trials: int

    def build(
        self,
        space: SearchSpace,
        store: Optional[Any],
        *,
        directions: Optional[list[str]] = None,
    ) -> SearchStrategy:
        # Random sampling is single-objective; ``directions`` is ignored (a
        # multi-objective scorer with random is rejected at validation, 4.8).
        return RandomStrategy(space, n_trials=self.n_trials, seed=self.seed)


class OptunaConfig(StrategyConfig):
    """Builds an ``OptunaStrategy`` (TPE/CMA-ES/GP/NSGA-II + ASHA pruning).

    The Phase-2 backend config. Construction and serialization stay
    Optuna-free; only :meth:`build` resolves the optional ``tune`` extra (via
    ``_require_optuna``). It rides the ``TuningSpec.strategy`` polymorphic field
    (not the closed ``StrategyConfigUnion``), so it round-trips through the
    class registry like any ``StrategyConfig`` subclass.

    Args:
        kind: The discriminator tag; always ``"optuna"``.
        sampler: The Optuna sampler (a closed :data:`SamplerKind` set); defaults
            to ``"tpe"``. A multi-objective study auto-selects NSGA-II at build
            time regardless of this value (Phase 4 wires the scorer).
        n_trials: The completed+pruned trial budget before exhaustion.
        prune: Whether to enable ASHA pruning (opt-in; the explore round runs
            unpruned). Defaults to ``False``.
        seed: The sampler seed for reproducibility; defaults to ``0``.
        storage_url: The Optuna storage URL (SQLite/Postgres). ``None`` resolves
            from ``$PHENOTYPIC_TUNE_STORAGE_URL`` at build time; if that is also
            unset the strategy uses an in-memory study.
    """

    kind: Literal["optuna"] = "optuna"
    sampler: SamplerKind = "tpe"
    n_trials: int
    prune: bool = False
    storage_url: Optional[str] = None

    def build(
        self,
        space: SearchSpace,
        store: Optional[Any],
        *,
        directions: Optional[list[str]] = None,
    ) -> SearchStrategy:
        """Construct the live ``OptunaStrategy`` (resolves the ``tune`` extra).

        ``import optuna`` is deferred to ``OptunaStrategy``'s body; this method
        calls ``_require_optuna`` first to raise an actionable error when the
        extra is missing. The storage URL falls back to
        ``$PHENOTYPIC_TUNE_STORAGE_URL`` when ``storage_url is None``. When
        ``directions`` is multi-objective the strategy auto-selects **NSGA-II**
        (the multi-objective sampler) regardless of :attr:`sampler` and disables
        pruning (Optuna pruners are single-objective; optuna-integration §9).

        Args:
            space: The search space to materialize each trial from.
            store: The study store (the Optuna study owns persistence; passed for
                the uniform factory signature).
            directions: Per-objective ``["maximize"] * n`` for a multi-objective
                run, inferred from the scorer; ``None`` → single-objective.
        """
        from ._optuna import OptunaStrategy
        from ._optuna_support import _require_optuna

        _require_optuna()
        # Bind the strategy to the STORE's shared study so its native ask/tell is
        # the single persisted record (sampler resumes in place, the SLURM fleet
        # drains one budget, no phantom auto-named study). An Optuna store exposes
        # ``study_name`` + ``storage_url``; a non-Optuna store (the screening
        # rounds' journal) exposes neither, so the strategy falls back to its own
        # study from the explicit URL / the ``$PHENOTYPIC_TUNE_STORAGE_URL`` env.
        study_name = getattr(store, "study_name", None)
        storage_url = getattr(store, "storage_url", None) or self.storage_url
        if storage_url is None:
            storage_url = os.environ.get(PHENOTYPIC_TUNE_STORAGE_URL_ENV)
        return OptunaStrategy(
            space,
            sampler=self.sampler,
            n_trials=self.n_trials,
            prune=self.prune,
            seed=self.seed,
            storage_url=storage_url,
            store=store,
            study_name=study_name,
            directions=directions,
        )


#: Discriminated union of the built-in (Phase 1) strategy configs. ``OptunaConfig``
#: is intentionally **not** a member: it rides the ``TuningSpec.strategy``
#: polymorphic field (registry round-trip), not this closed union.
StrategyConfigUnion = Annotated[
    Union[GridConfig, RandomConfig], Field(discriminator="kind")
]
