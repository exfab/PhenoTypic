"""Two-round screening freeze — explore → freeze → focused (screening-importance.md).

The :class:`ScreeningController` turns the screening lifecycle (§3) into a
runnable two-round flow:

1. **Explore round** — all params free, **pruning off** (the survivorship-bias
   guard, §4): a full-fidelity importance sample.
2. **Freeze gate** — compute importance (fANOVA / RF dispatch, G1), and freeze
   the least-important params whose **cumulative total** importance is ``< ε``
   (§6). A param with a small main effect but large interaction share has a large
   *total* importance and is never frozen. Each frozen param is pinned to the
   **central tendency** (median numeric / mode categorical) of its value across
   the top-k explore trials, as a :class:`~phenotypic.tune.Fixed` knob in a
   reduced :class:`~phenotypic.tune.SearchSpace`.
3. **Focused round** — a **fresh study** on the reduced space, **warm-started**
   with the top-k explore configs (§3); pruning is whatever the spec configures.

Guards: screening is **off below** ``free_param_floor`` free params; a **warm-up
floor** ``W = max(warmup_floor, warmup_c · n_params)`` blocks a premature freeze;
the RF-permutation path freezes **fewer** params and flags interactions
unverified (§6). The **winner is the best held-out trial across both rounds**, so
freezing can never make the result worse than the explore best; if the focused
round underperforms the explore best on held-out, the freeze is **flagged** and
the explore best is returned with a re-run recommendation (G3, §6).
"""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ._screening import ImportanceMethod, ImportanceReport, compute_param_importance_report
from ._search_space import Fixed, Knob, SearchSpace
from ._spec import TuningSpec
from ._study_store import JournalStudyStore, Trial


@dataclass(frozen=True)
class ScreeningConfig:
    """Conservative, config-exposed freeze thresholds (screening-importance §14).

    Args:
        free_param_floor: Screening is off until the space has **more than** this
            many free (non-``Fixed``) params (default ``6``).
        freeze_epsilon: The cumulative-tail budget ``ε``: freeze the least-important
            params whose total importance collectively stays ``< ε`` (default
            ``0.10`` → keep the params covering ~90% of explained variance).
        warmup_floor: The absolute warm-up trial floor (default ``20``).
        warmup_c: The per-param warm-up multiplier; the freeze-grade floor is
            ``max(warmup_floor, warmup_c · n_params)`` (default ``3``).
        top_k: How many best explore trials feed the central-tendency freeze value
            and the focused round's warm-start (default ``10``).
    """

    free_param_floor: int = 6
    freeze_epsilon: float = 0.10
    warmup_floor: int = 20
    warmup_c: int = 3
    top_k: int = 10


@dataclass(frozen=True)
class ScreeningResult:
    """The outcome of a (possibly two-round) screening run.

    Args:
        winner: The best held-out :class:`Trial` across both rounds (``None`` only
            if nothing succeeded).
        frozen: ``{frozen_key: pinned_value}`` — empty when no freeze occurred.
        method: The importance method used (fANOVA vs RF-permutation).
        interactions_estimated: Whether ``method`` accounts for interactions.
        screened: Whether the focused (freeze) round actually ran.
        freeze_flagged: ``True`` when the focused round underperformed explore on
            held-out and the freeze was flagged as likely-bad (G3).
        reduced_space: The reduced (frozen) space the focused round searched, or
            ``None`` when no freeze occurred.
        recommendation: A plain-English next step for the operator.
    """

    winner: Optional[Trial]
    frozen: dict[str, Any] = field(default_factory=dict)
    method: ImportanceMethod = "rf-permutation"
    interactions_estimated: bool = False
    screened: bool = False
    freeze_flagged: bool = False
    reduced_space: Optional[SearchSpace] = None
    recommendation: str = ""


# --- pure helpers -------------------------------------------------------------


def count_free_params(space: SearchSpace) -> int:
    """Count the free (non-``Fixed``) knobs in ``space``.

    Args:
        space: The search space.

    Returns:
        The number of knobs whose domain is not :class:`Fixed`.
    """
    return sum(1 for knob in space.knobs if not isinstance(knob.domain, Fixed))


def screening_warmup_floor(n_params: int, *, floor: int = 20, c: int = 3) -> int:
    """The freeze-grade warm-up floor ``W = max(floor, c · n_params)`` (§4).

    Args:
        n_params: The number of free params.
        floor: The absolute trial floor.
        c: The per-param multiplier.

    Returns:
        The minimum explore-trial count before a freeze is trustworthy.
    """
    return max(floor, c * n_params)


def select_params_to_freeze(
    report: ImportanceReport, *, epsilon: float = 0.10
) -> list[str]:
    """The cumulative-tail freeze set over **total** importance (§6).

    Walks params from least to most important, accumulating their importance
    shares; freezes each one while the running cumulative stays **strictly below**
    ``epsilon``. Because the importances are *total* (main + interaction on the
    fANOVA path), a low-main/high-interaction param carries a large share and is
    never reached by the tail. On the RF-permutation path (interactions
    unverified) the threshold is **halved** so freezing is strictly more
    conservative — it freezes a subset of what fANOVA would freeze (§6).

    Args:
        report: The importance estimate + honesty flags.
        epsilon: The cumulative-tail budget over total importance.

    Returns:
        The keys to freeze (least-important first); empty if nothing fits.
    """
    if not report.importances:
        return []
    # RF-permutation can't see interactions → be conservative (freeze fewer).
    budget = epsilon if report.interactions_estimated else epsilon / 2.0

    ascending = sorted(report.importances.items(), key=lambda kv: kv[1])
    frozen: list[str] = []
    cumulative = 0.0
    for key, share in ascending:
        if cumulative + share < budget:
            cumulative += share
            frozen.append(key)
        else:
            break
    return frozen


def freeze_value(key: str, trials: list[Trial], *, top_k: int = 10) -> Any:
    """The central-tendency value of ``key`` across the top-k explore trials (§6).

    Numeric values → the **median**; non-numeric (categorical/bool) → the **mode**
    ("the value good configs tend to use", robust to single-trial noise). Only the
    ``top_k`` highest-scoring non-failed trials that actually carry ``key`` count.

    Args:
        key: The frozen param's combo key.
        trials: The explore-round trials.
        top_k: How many best trials to take the central tendency over.

    Returns:
        The pinned value for the frozen knob.

    Raises:
        ValueError: If no top-k trial carries ``key`` (nothing to freeze at).
    """
    ranked = sorted(
        (t for t in trials if not t.failed and key in t.params),
        key=lambda t: t.score,
        reverse=True,
    )[:top_k]
    values = [t.params[key] for t in ranked]
    if not values:
        raise ValueError(f"no top-k trial carries {key!r}; cannot pick a freeze value")
    if all(isinstance(v, bool) for v in values) or not all(
        isinstance(v, (int, float)) for v in values
    ):
        # Booleans are ints in Python but are categorical here → mode.
        return Counter(values).most_common(1)[0][0]
    return statistics.median(values)


def build_reduced_space(
    space: SearchSpace, frozen_values: dict[str, Any]
) -> SearchSpace:
    """Pin every ``frozen_values`` key to a :class:`Fixed` domain (§6).

    Kept knobs retain their original domain; a frozen knob's domain is replaced by
    ``Fixed(value=...)`` so the focused round treats it as a constant, never a
    trial dimension. Provenance and conditioning are preserved.

    Args:
        space: The explore-round (full) space.
        frozen_values: ``{key: pinned_value}`` for the params to freeze.

    Returns:
        A new :class:`SearchSpace` with the frozen knobs pinned.
    """
    new_knobs: list[Knob] = []
    for knob in space.knobs:
        if knob.key in frozen_values:
            new_knobs.append(
                knob.model_copy(update={"domain": Fixed(value=frozen_values[knob.key])})
            )
        else:
            new_knobs.append(knob)
    return SearchSpace(knobs=tuple(new_knobs))


# --- orchestration ------------------------------------------------------------


class ScreeningController:
    """Orchestrate the explore → freeze → focused two-round screening flow.

    The controller is engine-opt-in: a caller hands it a :class:`TuningSpec` and a
    :class:`ScreeningConfig`, and :meth:`run` returns a :class:`ScreeningResult`
    with the winner, the freeze decision, and the honesty flags. Both rounds'
    stores are exposed (:attr:`explore_store`, :attr:`focused_store`) for the CLI
    to journal and for the winner-across-both-rounds rule.

    Args:
        spec: The tuning recipe (its ``search_space`` is the explore space; the
            focused round runs the reduced space).
        config: The freeze thresholds.
        importance_report_fn: Override for the importance computation (defaults to
            :func:`compute_param_importance_report`); injected in tests to force a
            deterministic freeze decision.
        engine_factory: Builds a tuning engine from ``(spec, store)``; defaults to
            the real :class:`~phenotypic.tune.TuningEngine`. Injectable for tests.
        _focused_score_penalty: Test seam — subtract this from every fresh
            focused-round score to exercise the wrong-freeze recovery path. Never
            set in production (default ``0.0``).
    """

    def __init__(
        self,
        spec: TuningSpec,
        *,
        config: ScreeningConfig = ScreeningConfig(),
        importance_report_fn: Optional[
            Callable[[Any], ImportanceReport]
        ] = None,
        engine_factory: Optional[Callable[[TuningSpec, Any], Any]] = None,
        _focused_score_penalty: float = 0.0,
    ) -> None:
        self._spec = spec
        self._config = config
        self._importance_report_fn = (
            importance_report_fn or compute_param_importance_report
        )
        self._engine_factory = engine_factory or self._default_engine
        self._focused_penalty = _focused_score_penalty

        self.explore_store: JournalStudyStore = JournalStudyStore()
        self.focused_store: Optional[JournalStudyStore] = None
        #: How many leading focused-store trials are warm-start seeds (re-journaled
        #: explore configs), not genuinely-focused trials. Set in :meth:`run`.
        self._warm_count: int = 0

    @staticmethod
    def _default_engine(spec: TuningSpec, store: Any) -> Any:
        from ._engine import TuningEngine

        return TuningEngine(spec, store=store)

    def run(self, images: list) -> ScreeningResult:
        """Run the (possibly two-round) screening flow over ``images``.

        Args:
            images: The calibration images (non-empty).

        Returns:
            The :class:`ScreeningResult` (winner across both rounds, the freeze
            decision, and the honesty flags).
        """
        # -- explore round (all params free; the spec's own channel) ----------
        self.explore_store = JournalStudyStore()
        explore_engine = self._engine_factory(self._spec, self.explore_store)
        explore_best = explore_engine.optimize(images)

        report = self._importance_report_fn(self.explore_store)

        # -- freeze gate ------------------------------------------------------
        n_free = count_free_params(self._spec.search_space)
        warmup = screening_warmup_floor(
            n_free, floor=self._config.warmup_floor, c=self._config.warmup_c
        )
        gate_open = (
            n_free > self._config.free_param_floor
            and self.explore_store.completed_count() >= warmup
        )

        frozen_values: dict[str, Any] = {}
        if gate_open:
            frozen_values = self._compute_freeze(report)

        if not frozen_values:
            return ScreeningResult(
                winner=explore_best,
                method=report.method,
                interactions_estimated=report.interactions_estimated,
                screened=False,
                reduced_space=None,
                recommendation=self._no_freeze_recommendation(n_free),
            )

        # -- focused round (fresh study on the reduced space, warm-started) ---
        reduced = build_reduced_space(self._spec.search_space, frozen_values)
        self.focused_store = self._warm_started_store(reduced)
        self._warm_count = len(self.focused_store)
        focused_spec = self._spec.model_copy(update={"search_space": reduced})
        focused_engine = self._engine_factory(focused_spec, self.focused_store)
        focused_engine.optimize(images)
        self._apply_focused_penalty()

        return self._resolve_winner(explore_best, frozen_values, report, reduced)

    # -- freeze gate internals ------------------------------------------------

    def _compute_freeze(self, report: ImportanceReport) -> dict[str, Any]:
        """Pick the freeze set and its central-tendency pinned values."""
        keys = select_params_to_freeze(
            report, epsilon=self._config.freeze_epsilon
        )
        frozen: dict[str, Any] = {}
        for key in keys:
            try:
                frozen[key] = freeze_value(
                    key, self.explore_store.trials, top_k=self._config.top_k
                )
            except ValueError:
                # No top-k trial carried the key (e.g. a never-active conditional)
                # → nothing to freeze it at; leave it free.
                continue
        return frozen

    def _warm_started_store(self, reduced: SearchSpace) -> JournalStudyStore:
        """Seed a fresh focused store with the top-k explore configs (§3 warm-start).

        Each warm-start trial echoes a top explore config projected onto the
        reduced space (frozen knobs pinned to their ``Fixed`` value, kept knobs
        retained), so the focused round reuses what explore learned about the kept
        params and the winner-across-both-rounds union already includes them.
        """
        frozen_keys = {
            knob.key
            for knob in reduced.knobs
            if isinstance(knob.domain, Fixed)
        }
        fixed_values = {
            knob.key: knob.domain.value
            for knob in reduced.knobs
            if isinstance(knob.domain, Fixed)
        }
        top = sorted(
            (t for t in self.explore_store.trials if not t.failed),
            key=lambda t: t.score,
            reverse=True,
        )[: self._config.top_k]

        store = JournalStudyStore()
        for offset, trial in enumerate(top):
            projected = {
                key: (fixed_values[key] if key in frozen_keys else value)
                for key, value in trial.params.items()
            }
            store.append(
                Trial(
                    number=offset,
                    params=projected,
                    score=trial.score,
                    terms=trial.terms,
                    n_images=trial.n_images,
                )
            )
        return store

    def _apply_focused_penalty(self) -> None:
        """Depress genuinely-focused scores (test-only seam for G3 recovery)."""
        if self._focused_penalty == 0.0 or self.focused_store is None:
            return
        penalized: list[Trial] = []
        for index, trial in enumerate(self.focused_store.trials):
            if index >= self._warm_count and not trial.failed:
                penalized.append(
                    trial.model_copy(
                        update={"score": trial.score - self._focused_penalty}
                    )
                )
            else:
                penalized.append(trial)
        self.focused_store = JournalStudyStore(penalized)

    def _genuinely_focused_best(self) -> Optional[Trial]:
        """The best trial the focused round *produced* (excluding warm-start seeds).

        The recovery decision compares this against the explore best: warm-start
        seeds are re-journaled explore configs, so counting them as the focused
        result would mask a freeze that actually hurt the genuinely-focused
        search.
        """
        assert self.focused_store is not None
        fresh = [
            t
            for t in self.focused_store.trials[self._warm_count :]
            if not t.failed
        ]
        if not fresh:
            return None
        return max(fresh, key=lambda t: t.score)

    def _resolve_winner(
        self,
        explore_best: Optional[Trial],
        frozen_values: dict[str, Any],
        report: ImportanceReport,
        reduced: SearchSpace,
    ) -> ScreeningResult:
        """Pick the winner across both rounds + the wrong-freeze recovery (G3)."""
        assert self.focused_store is not None

        explore_score = (
            explore_best.score if explore_best is not None else float("-inf")
        )
        focused_best = self._genuinely_focused_best()
        focused_score = (
            focused_best.score if focused_best is not None else float("-inf")
        )

        # Wrong-freeze recovery: the genuinely-focused round underperformed
        # explore on held-out → flag the freeze, return the explore best,
        # recommend re-run (no mid-study unfreeze).
        if focused_score < explore_score:
            return ScreeningResult(
                winner=explore_best,
                frozen=frozen_values,
                method=report.method,
                interactions_estimated=report.interactions_estimated,
                screened=True,
                freeze_flagged=True,
                reduced_space=reduced,
                recommendation=(
                    "Focused round underperformed explore on held-out — the "
                    "freeze was likely wrong. Returning the explore best; "
                    "re-run without freezing (--no-screen) to confirm."
                ),
            )

        # Winner = best held-out across both rounds (the full union, including
        # warm-start seeds), so freezing never beats the explore best by accident.
        union = self.explore_store.trials + self.focused_store.trials
        winner = max(
            (t for t in union if not t.failed),
            key=lambda t: t.score,
            default=explore_best,
        )
        return ScreeningResult(
            winner=winner,
            frozen=frozen_values,
            method=report.method,
            interactions_estimated=report.interactions_estimated,
            screened=True,
            freeze_flagged=False,
            reduced_space=reduced,
            recommendation=(
                f"Froze {len(frozen_values)} low-importance param(s); "
                "focused round held its own on held-out."
            ),
        )

    def _no_freeze_recommendation(self, n_free: int) -> str:
        """The recommendation string when no freeze occurred."""
        if n_free <= self._config.free_param_floor:
            return (
                f"{n_free} free param(s) ≤ floor "
                f"{self._config.free_param_floor}: screening reports importance "
                "only (no freeze)."
            )
        return (
            "Importance is still warming up (not freeze-grade); ran a single "
            "round without freezing."
        )
