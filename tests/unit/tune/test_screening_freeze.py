"""Two-round screening freeze — the ``ScreeningController`` (G2 + G3).

Covers the pure freeze helpers (free-param count gate, warm-up floor,
cumulative-tail selection over total importance, central-tendency freeze value,
reduced-space ``Fixed`` domains, conservative RF fallback) and the orchestrating
``run`` (screening off below trigger; explore round unpruned; focused warm-start;
winner across both rounds; warm-up guard blocks premature freeze; wrong-freeze
recovery falls back to the explore best). Hermetic — orchestration drives a tiny
grid space over a single synthetic plate with a constant scorer.
"""
from __future__ import annotations

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tune import (
    Budget,
    Categorical,
    Evaluator,
    EvaluationResult,
    Fixed,
    FloatRange,
    GridConfig,
    ImportanceReport,
    IntRange,
    Knob,
    RandomConfig,
    Scorer,
    SearchSpace,
    TuningSpec,
)
from phenotypic.tune._screening_freeze import (
    ScreeningConfig,
    ScreeningController,
    ScreeningResult,
    build_reduced_space,
    count_free_params,
    freeze_value,
    screening_warmup_floor,
    select_params_to_freeze,
)
from phenotypic.tune._study_store import Trial


# --- pure helpers -------------------------------------------------------------


def test_count_free_params_excludes_fixed():
    space = SearchSpace(knobs=(
        Knob(key="0.a", domain=FloatRange(low=0.0, high=1.0)),
        Knob(key="0.b", domain=Categorical(choices=(True, False))),
        Knob(key="0.c", domain=Fixed(value=3)),
    ))
    assert count_free_params(space) == 2


def test_warmup_floor_is_max_of_floor_and_c_times_n():
    # W = max(20, 3 * n_params): floor dominates for small spaces.
    assert screening_warmup_floor(3, floor=20, c=3) == 20
    # c * n dominates once the space is large enough.
    assert screening_warmup_floor(10, floor=20, c=3) == 30


def test_select_freezes_cumulative_tail_under_epsilon():
    # Importances summing to 1.0; the tail {d:0.02, c:0.03} sums to 0.05 < 0.10,
    # so both freeze; adding b (0.15) would exceed epsilon, so b stays free.
    report = ImportanceReport(
        importances={"a": 0.80, "b": 0.15, "c": 0.03, "d": 0.02},
        method="fanova",
        interactions_estimated=True,
    )
    frozen = select_params_to_freeze(report, epsilon=0.10)
    assert set(frozen) == {"c", "d"}


def test_high_interaction_param_is_not_frozen_under_total_importance():
    # fANOVA importances are TOTAL (main + interaction): a param with a large
    # interaction share has a large TOTAL importance and must NOT be frozen,
    # even though its main effect alone would look tiny.
    report = ImportanceReport(
        importances={"a": 0.55, "interacts": 0.40, "noise": 0.05},
        method="fanova",
        interactions_estimated=True,
    )
    frozen = select_params_to_freeze(report, epsilon=0.10)
    assert "interacts" not in frozen
    assert set(frozen) == {"noise"}


def test_conservative_rf_fallback_freezes_fewer():
    # Same shares, but the RF-permutation method (interactions unverified) is
    # conservative: it freezes a strict subset of what fANOVA would freeze.
    # fANOVA (ε=0.10): tail {d:0.03, c:0.04} sums to 0.07 < 0.10 → both freeze.
    # RF (budget ε/2=0.05): only d:0.03 < 0.05 fits → {d}, a strict subset.
    importances = {"a": 0.83, "b": 0.10, "c": 0.04, "d": 0.03}
    fanova = select_params_to_freeze(
        ImportanceReport(
            importances=importances, method="fanova", interactions_estimated=True
        ),
        epsilon=0.10,
    )
    rf = select_params_to_freeze(
        ImportanceReport(
            importances=importances,
            method="rf-permutation",
            interactions_estimated=False,
        ),
        epsilon=0.10,
    )
    assert set(rf) < set(fanova)


def test_freeze_value_numeric_is_top_k_median():
    # Best-scoring trials carry x in {2,4,6}; median = 4 over the top 3.
    trials = [
        Trial(number=0, params={"x": 2.0}, score=0.9, terms={}, n_images=1),
        Trial(number=1, params={"x": 4.0}, score=0.8, terms={}, n_images=1),
        Trial(number=2, params={"x": 6.0}, score=0.7, terms={}, n_images=1),
        Trial(number=3, params={"x": 99.0}, score=0.1, terms={}, n_images=1),
    ]
    assert freeze_value("x", trials, top_k=3) == 4.0


def test_freeze_value_categorical_is_top_k_mode():
    trials = [
        Trial(number=0, params={"m": "ridge"}, score=0.9, terms={}, n_images=1),
        Trial(number=1, params={"m": "ridge"}, score=0.85, terms={}, n_images=1),
        Trial(number=2, params={"m": "flat"}, score=0.8, terms={}, n_images=1),
        Trial(number=3, params={"m": "flat"}, score=0.1, terms={}, n_images=1),
    ]
    assert freeze_value("m", trials, top_k=3) == "ridge"


def test_build_reduced_space_pins_frozen_to_fixed():
    space = SearchSpace(knobs=(
        Knob(key="0.a", domain=FloatRange(low=0.0, high=8.0)),
        Knob(key="0.b", domain=Categorical(choices=("x", "y"))),
    ))
    reduced = build_reduced_space(space, {"0.b": "x"})
    # The kept knob keeps its domain; the frozen knob becomes Fixed(value).
    assert isinstance(reduced.domain("0.a"), FloatRange)
    frozen_domain = reduced.domain("0.b")
    assert isinstance(frozen_domain, Fixed)
    assert frozen_domain.value == "x"


# --- orchestration ------------------------------------------------------------


class _SigmaScorer(Scorer):
    """Score rewards a sigma near 3.0; ignore_zeros is pure noise.

    Lets a freeze decision sensibly pin the noise param while keeping sigma.
    """

    def score_image(self, image, measurements) -> dict[str, float]:
        return {"Count": 1.0}


def _grid_space() -> SearchSpace:
    # Grid-enumerable (IntRange + Categorical) so GridConfig can build it.
    return SearchSpace(knobs=(
        Knob(key="0.sigma", domain=IntRange(low=1, high=4)),
        Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))


def _random_space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="0.sigma", domain=FloatRange(low=0.5, high=6.0)),
        Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))


def _spec(strategy, space=None, budget=None) -> TuningSpec:
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()]),
        search_space=space if space is not None else _grid_space(),
        scorer=_SigmaScorer(),
        evaluator=Evaluator(),
        strategy=strategy,
        budget=budget if budget is not None else Budget(),
    )


def test_screening_off_below_free_param_floor():
    # Only 2 free params < floor 6: screening is OFF, no freeze, single round.
    spec = _spec(GridConfig())
    controller = ScreeningController(
        spec, config=ScreeningConfig(free_param_floor=6)
    )
    result = controller.run([load_synth_yeast_plate()])
    assert isinstance(result, ScreeningResult)
    assert result.screened is False
    assert result.frozen == {}
    assert result.winner is not None


def test_explore_round_runs_unpruned():
    # The explore round must request unpruned evaluation (survivorship guard):
    # capture whether any pruning channel was active in the explore round.
    spec = _spec(GridConfig(), budget=Budget(n_trials=2))
    controller = ScreeningController(
        spec, config=ScreeningConfig(free_param_floor=0)
    )
    controller.run([load_synth_yeast_plate()])
    # No trial in the explore store was pruned (unpruned full-fidelity pass).
    assert all(not t.pruned for t in controller.explore_store.trials)


def test_warmup_guard_blocks_premature_freeze():
    # With a warm-up floor far above the trial budget, importance is not
    # freeze-grade: no params freeze even though the gate is enabled.
    spec = _spec(RandomConfig(n_trials=3), space=_random_space(), budget=Budget(n_trials=3))
    controller = ScreeningController(
        spec,
        config=ScreeningConfig(free_param_floor=0, warmup_floor=1000, warmup_c=3),
    )
    result = controller.run([load_synth_yeast_plate()])
    assert result.frozen == {}
    assert result.screened is False
    assert result.winner is not None


def test_screened_run_freezes_and_warm_starts_focused_round():
    # Force the freeze path with a deterministic importance + a generous budget.
    # The focused round must be warm-started: its store carries the top-k explore
    # configs as its first trials, and the reduced space pins the frozen knob.
    spec = _spec(RandomConfig(n_trials=6), space=_random_space(), budget=Budget(n_trials=6))

    forced = ImportanceReport(
        importances={"0.sigma": 0.97, "1.ignore_zeros": 0.03},
        method="fanova",
        interactions_estimated=True,
    )
    controller = ScreeningController(
        spec,
        config=ScreeningConfig(free_param_floor=0, warmup_floor=1, top_k=3),
        importance_report_fn=lambda store: forced,
    )
    result = controller.run([load_synth_yeast_plate()])
    assert result.screened is True
    assert "1.ignore_zeros" in result.frozen
    # The reduced space pins the frozen knob to a Fixed domain.
    assert isinstance(result.reduced_space.domain("1.ignore_zeros"), Fixed)
    # Warm-start: the focused store's first trials echo the top-k explore configs.
    assert controller.focused_store is not None
    assert len(controller.focused_store) >= 1


def test_winner_is_best_held_out_across_both_rounds():
    # The winner must be the best trial across the union of explore + focused
    # trials, never just the focused round.
    spec = _spec(RandomConfig(n_trials=6), space=_random_space(), budget=Budget(n_trials=6))
    forced = ImportanceReport(
        importances={"0.sigma": 0.97, "1.ignore_zeros": 0.03},
        method="fanova",
        interactions_estimated=True,
    )
    controller = ScreeningController(
        spec,
        config=ScreeningConfig(free_param_floor=0, warmup_floor=1, top_k=3),
        importance_report_fn=lambda store: forced,
    )
    result = controller.run([load_synth_yeast_plate()])
    combined = controller.explore_store.trials + (
        controller.focused_store.trials if controller.focused_store else []
    )
    best = max((t for t in combined if not t.failed), key=lambda t: t.score)
    assert result.winner.score == best.score


# --- G3: wrong-freeze recovery ------------------------------------------------


def test_bad_freeze_falls_back_to_explore_best():
    # When the focused round underperforms the explore round on held-out, the
    # controller returns the best EXPLORE config, flags the freeze, and
    # recommends re-running without it (no mid-study unfreeze).
    spec = _spec(RandomConfig(n_trials=6), space=_random_space(), budget=Budget(n_trials=6))
    forced = ImportanceReport(
        importances={"0.sigma": 0.97, "1.ignore_zeros": 0.03},
        method="fanova",
        interactions_estimated=True,
    )

    # A scorer whose focused-round trials always score worse than explore: the
    # controller injects scores so the explore best strictly beats the focused
    # best, triggering the recovery path.
    controller = ScreeningController(
        spec,
        config=ScreeningConfig(free_param_floor=0, warmup_floor=1, top_k=3),
        importance_report_fn=lambda store: forced,
        _focused_score_penalty=10.0,  # test seam: depress focused scores
    )
    result = controller.run([load_synth_yeast_plate()])
    assert result.freeze_flagged is True
    assert "re-run" in result.recommendation.lower()
    # The winner is the explore best, not a depressed focused trial.
    explore_best = max(
        (t for t in controller.explore_store.trials if not t.failed),
        key=lambda t: t.score,
    )
    assert result.winner.score == explore_best.score


class _WarmStartEvaluator(Evaluator):
    def evaluate(self, pipeline, scorer, params, images, channel=None):
        return EvaluationResult(
            score=42.0,
            terms={"rechecked": 42.0},
            n_images=len(images),
        )


def test_changed_warm_start_projection_is_re_evaluated():
    """Projected frozen params must not keep the original explore score."""

    def _engine_factory(spec, store):
        class _Engine:
            def optimize(self, images):
                if store is controller.explore_store:
                    store.append(
                        Trial(
                            number=0,
                            params={"0.sigma": 1.0, "1.ignore_zeros": True},
                            score=10.0,
                            terms={"old": 10.0},
                            n_images=1,
                        )
                    )
                    store.append(
                        Trial(
                            number=1,
                            params={"0.sigma": 2.0, "1.ignore_zeros": False},
                            score=9.0,
                            terms={"old": 9.0},
                            n_images=1,
                        )
                    )
                    return store.trials[0]
                return None

        return _Engine()

    spec = _spec(
        RandomConfig(n_trials=2),
        space=_random_space(),
        budget=Budget(n_trials=2),
    ).model_copy(update={"evaluator": _WarmStartEvaluator()})
    forced = ImportanceReport(
        importances={"0.sigma": 0.97, "1.ignore_zeros": 0.03},
        method="fanova",
        interactions_estimated=True,
    )
    controller = ScreeningController(
        spec,
        config=ScreeningConfig(
            free_param_floor=0, warmup_floor=1, warmup_c=0, top_k=2
        ),
        importance_report_fn=lambda store: forced,
        engine_factory=_engine_factory,
    )

    controller.run([load_synth_yeast_plate()])

    assert controller.focused_store is not None
    changed = controller.focused_store.trials[1]
    assert changed.params == {"0.sigma": 2.0, "1.ignore_zeros": True}
    assert changed.score == 42.0
    assert changed.terms == {"rechecked": 42.0}
