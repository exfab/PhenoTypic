"""``OptunaStrategy`` — ask-and-tell over an Optuna study (optuna-integration §3-§9).

``import optuna`` happens **lazily inside the function/method bodies** here, never
at module import: importing this module (e.g. via ``OptunaConfig.build``) must not
pull Optuna in until a call actually needs it, preserving the package-wide
lazy-import boundary (§10). The strategy drives Optuna through ``study.ask`` /
``study.tell`` so the engine owns the loop; ``suggest`` materializes the
``SearchSpace`` into trial suggestions (define-by-run conditionals, ``Fixed``
injected as constants), and ``register_result`` tells the study the outcome with
the right ``TrialState``. A per-trial :class:`OptunaPruningChannel` backs the
generic pruning channel; the study's ASHA pruner is derived from the Evaluator's
rung ladder so the two cannot disagree (§6).
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional

from .._search_space import (
    Categorical,
    Domain,
    Fixed,
    FloatRange,
    IntRange,
    SearchSpace,
)
from ._config import SamplerKind
from ._pruning import NoOpChannel, PruningChannel

if TYPE_CHECKING:  # pragma: no cover - typing only, never imports optuna at runtime
    import optuna

_logger = logging.getLogger(__name__)


class OptunaPruningChannel:
    """Backs the generic :class:`PruningChannel` with a live Optuna trial.

    A pure passthrough: ``report`` forwards to ``trial.report`` and
    ``should_prune`` forwards to ``trial.should_prune``, so ASHA's cross-trial
    comparisons at each ``step`` are exactly Optuna's (identical pruning
    accuracy) while the Evaluator depends only on the two-method Protocol.

    Args:
        trial: The in-flight Optuna trial this channel reports into.
    """

    def __init__(self, trial: "optuna.trial.Trial") -> None:
        self._trial = trial

    def report(self, value: float, step: int) -> None:
        """Forward an interim ``value`` at ``step`` to the trial."""
        self._trial.report(value, step)

    def should_prune(self) -> bool:
        """Whether the trial should early-stop, per the study's pruner."""
        return bool(self._trial.should_prune())


class OptunaStrategy:
    """A :class:`SearchStrategy` over an Optuna study (TPE/CMA-ES/GP/NSGA-II + ASHA).

    Holds **one in-flight trial** between :meth:`suggest` and
    :meth:`register_result` (per-worker, serial ``ask→evaluate→tell``); concurrency
    comes from running one instance per worker against a shared study (§7). The
    study is created with a sampler chosen from ``sampler`` (or NSGA-II when
    ``directions`` makes it multi-objective) and an ASHA
    ``SuccessiveHalvingPruner`` derived from ``rung_floor``/``rung_factor`` so the
    Evaluator's ladder and the pruner agree (§6).

    Args:
        space: The search space to materialize each trial from.
        sampler: The sampler kind (a closed :data:`SamplerKind` set). Ignored
            when ``directions`` is multi-objective (NSGA-II is forced).
        n_trials: The completed+pruned budget :meth:`is_exhausted` counts against.
        prune: Whether the pruning channel is Optuna-backed (opt-in). The explore
            round and multi-objective studies always get a no-op channel.
        seed: The sampler seed for reproducibility.
        storage_url: The Optuna storage URL (SQLite/Postgres); ``None`` → an
            in-memory study.
        store: Accepted for the uniform factory signature; the Optuna study owns
            persistence so this is unused (the engine reads trials from the
            ``OptunaStudyStore`` wrapping the same study).
        study_name: Optional study name (for resume / shared studies).
        directions: Per-objective directions for a multi-objective study (e.g.
            ``["maximize", "maximize"]``); ``None`` → single-objective maximize.
        rung_floor: The ASHA ``min_resource`` (first-rung plates); mirrors the
            Evaluator's ``rung_floor``.
        rung_factor: The ASHA ``reduction_factor``; mirrors the Evaluator's
            ``rung_factor``.
    """

    def __init__(
        self,
        space: SearchSpace,
        *,
        sampler: SamplerKind = "tpe",
        n_trials: int,
        prune: bool = False,
        seed: int = 0,
        storage_url: Optional[str] = None,
        store: Optional[Any] = None,
        study_name: Optional[str] = None,
        directions: Optional[Sequence[str]] = None,
        rung_floor: int = 6,
        rung_factor: int = 3,
    ) -> None:
        import optuna

        self._space = space
        self._sampler_kind: SamplerKind = sampler
        self._n_trials = n_trials
        self._prune = prune
        self._seed = seed
        self._storage_url = storage_url
        self._directions = list(directions) if directions is not None else None
        self._multi_objective = self._directions is not None and len(
            self._directions
        ) > 1
        self._rung_floor = rung_floor
        self._rung_factor = rung_factor
        self._stashed: Optional["optuna.trial.Trial"] = None

        sampler_obj = self._make_sampler(optuna)
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=rung_floor, reduction_factor=rung_factor
        )
        create_kwargs: dict[str, Any] = {
            "sampler": sampler_obj,
            "pruner": pruner,
            "storage": storage_url,
            "study_name": study_name,
            "load_if_exists": study_name is not None,
        }
        if self._multi_objective:
            create_kwargs["directions"] = self._directions
        else:
            create_kwargs["direction"] = "maximize"
        self._study = optuna.create_study(**create_kwargs)

    # -- sampler selection ----------------------------------------------------

    def _make_sampler(self, optuna: Any) -> Any:
        """Pick the Optuna sampler: NSGA-II if multi-objective, else by kind."""
        if self._multi_objective:
            # Optuna pruners are single-objective; NSGA-II is the multi-objective
            # default (§9). Phase 4 wires the dict-returning scorer.
            return optuna.samplers.NSGAIISampler(seed=self._seed)
        if self._sampler_kind == "tpe":
            return optuna.samplers.TPESampler(seed=self._seed)
        if self._sampler_kind == "cmaes":
            # CMA-ES falls back to independent sampling for categoricals/
            # conditionals natively; the CLI warns about categorical-heavy spaces.
            return optuna.samplers.CmaEsSampler(seed=self._seed)
        if self._sampler_kind == "gp":
            return optuna.samplers.GPSampler(seed=self._seed)
        if self._sampler_kind == "nsga2":
            return optuna.samplers.NSGAIISampler(seed=self._seed)
        raise ValueError(f"unsupported sampler kind {self._sampler_kind!r}")

    # -- ask ------------------------------------------------------------------

    def suggest(
        self, *, explore: bool = False
    ) -> tuple[Mapping[str, Any], PruningChannel]:
        """Ask the study for a trial, materialize params, return ``(params, channel)``.

        Args:
            explore: When ``True`` (the screening explore round) the trial runs
                **unpruned** regardless of ``prune`` — a :class:`NoOpChannel` is
                returned so fANOVA's importance sample stays unbiased (§6).

        Returns:
            A ``(params, channel)`` pair: ``params`` is the materialized combo
            (conditional children absent when their parent is inactive; ``Fixed``
            knobs injected as constants), and ``channel`` is an
            :class:`OptunaPruningChannel` when pruning is active for this trial,
            else a :class:`NoOpChannel`.
        """
        trial = self._study.ask()
        self._stashed = trial
        params = self._materialize(trial)
        prune_active = self._prune and not explore and not self._multi_objective
        channel: PruningChannel = (
            OptunaPruningChannel(trial) if prune_active else NoOpChannel()
        )
        return params, channel

    def _materialize(self, trial: "optuna.trial.Trial") -> dict[str, Any]:
        """Walk knobs in order; suggest active ones; inject ``Fixed`` constants.

        A conditional child is suggested only when each ``(parent_key, value)``
        in its ``conditional_on`` matches what was already chosen this trial
        (define-by-run); an inactive child is simply absent from the dict, which
        Optuna's samplers handle natively.
        """
        chosen: dict[str, Any] = {}
        for knob in self._space.knobs:
            if knob.conditional_on is not None and not all(
                chosen.get(pk) == pv for pk, pv in knob.conditional_on
            ):
                continue
            chosen[knob.key] = self._suggest_domain(trial, knob.key, knob.domain)
        return chosen

    def _suggest_domain(
        self, trial: "optuna.trial.Trial", key: str, domain: Domain
    ) -> Any:
        """Map one domain to the right ``trial.suggest_*`` call (or a constant)."""
        if isinstance(domain, Fixed):
            # Never a trial dimension — injected as a constant (§4).
            return domain.value
        if isinstance(domain, Categorical):
            return trial.suggest_categorical(key, list(domain.choices))
        if isinstance(domain, IntRange):
            step, log = domain.step, domain.log
            if step != 1 and log:
                # Optuna forbids suggest_int(step≠1, log=True): normalize to
                # step=1 (log) with a logged note (§4).
                _logger.warning(
                    "IntRange %r: step=%d with log=True is unsupported by Optuna; "
                    "normalizing to step=1 (log scale).",
                    key,
                    step,
                )
                step = 1
            return trial.suggest_int(key, domain.low, domain.high, step=step, log=log)
        if isinstance(domain, FloatRange):
            return trial.suggest_float(key, domain.low, domain.high, log=domain.log)
        raise TypeError(f"unsupported domain {type(domain).__name__}")

    # -- tell -----------------------------------------------------------------

    def register_result(
        self, params: Mapping[str, Any], result: Any, *, pruned: bool = False
    ) -> None:
        """Tell the study how the stashed trial ended (COMPLETE / PRUNED / FAIL).

        Args:
            params: The combo this trial evaluated (unused — the stashed trial
                carries its own suggestions; accepted for Protocol parity).
            result: The :class:`EvaluationResult`; its ``score`` is told on a
                clean completion.
            pruned: Whether the rung ladder early-stopped the trial → told as
                ``TrialState.PRUNED``.
        """
        import optuna

        trial = self._stashed
        if trial is None:  # pragma: no cover - register without a prior suggest
            raise RuntimeError("register_result called before suggest")
        self._stashed = None

        if getattr(result, "failed", False):
            self._study.tell(trial, state=optuna.trial.TrialState.FAIL)
            return
        if pruned or getattr(result, "pruned", False):
            self._study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            return
        self._study.tell(trial, float(result.score))

    # -- budget ---------------------------------------------------------------

    def is_exhausted(self) -> bool:
        """Whether completed + pruned trials have reached ``n_trials`` (§8).

        Failed trials do not consume the budget; minor parallel overshoot is
        tolerated by the engine, not here.
        """
        import optuna

        done = self._study.get_trials(
            deepcopy=False,
            states=(
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
            ),
        )
        return len(done) >= self._n_trials
