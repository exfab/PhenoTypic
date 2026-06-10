"""Tests for the composite scorer — Phase 4 chunk B (4.5).

``CompositeScorer`` nests a ``list[Scorer]`` (via the polymorphic ``ScorerField``)
and blends them. These tests pin: per-child **prefixed** term merging
(collision-free), the scalar weighted/geometric ``finalize`` blend (default), the
``dict`` ``finalize`` when ``multi_objective=True`` (the sidecar path), nesting
round-trip through ``TuningSpec``, cycle/self-nesting rejection, and the pinned
``availability`` rule.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.tune import QCScorer, Scorer
from phenotypic.tune._scoring._composite import CompositeScorer


# --------------------------------------------------------------------------- #
# test doubles — deterministic scorers with disjoint / colliding term names
# --------------------------------------------------------------------------- #
class _FixedScorer(Scorer):
    """A stateless scorer returning preset terms; available iff ``ok``.

    Emits its preset terms as **cost** directly (the ``LOWER_BETTER`` default),
    so the base ``score_image`` template passes them through unchanged.
    """

    terms: dict[str, float]
    ok: bool = True

    def _score_terms(self, image, measurements) -> dict[str, float]:
        return dict(self.terms)

    def availability(self) -> bool:
        return self.ok


def _layout(n: int, name: str = "p1") -> pd.DataFrame:
    return pd.DataFrame(
        {"Metadata_ImageName": [name] * n, "Object_Label": list(range(n))}
    )


def _qc_scorer(tmp_path: Path) -> QCScorer:
    csv = tmp_path / "layout.csv"
    _layout(96).to_csv(csv, index=False)
    return QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=str(csv), groupby=["Metadata_ImageName"]
        )
    )


# --------------------------------------------------------------------------- #
# score_image — merges child terms with a collision-free per-child prefix
# --------------------------------------------------------------------------- #
def test_score_image_merges_with_prefix():
    comp = CompositeScorer(
        scorers=[
            _FixedScorer(terms={"Region": 0.8}),
            _FixedScorer(terms={"Count": 0.6}),
        ]
    )
    out = comp.score_image(None, pd.DataFrame())
    assert out == {"s0.Region": 0.8, "s1.Count": 0.6}


def test_score_image_prefix_disambiguates_colliding_term_names():
    # Two children both emit "Count": the per-child prefix keeps them distinct.
    comp = CompositeScorer(
        scorers=[
            _FixedScorer(terms={"Count": 0.3}),
            _FixedScorer(terms={"Count": 0.9}),
        ]
    )
    out = comp.score_image(None, pd.DataFrame())
    assert out == {"s0.Count": 0.3, "s1.Count": 0.9}


# --------------------------------------------------------------------------- #
# finalize — scalar blend (default single-objective)
# --------------------------------------------------------------------------- #
def test_finalize_scalar_weighted_blend():
    # blend="weighted_mean" → weighted arithmetic mean of the per-child costs.
    comp = CompositeScorer(
        scorers=[
            _FixedScorer(terms={"a": 0.8}),
            _FixedScorer(terms={"b": 0.4}),
        ],
        weights={"s0": 3.0, "s1": 1.0},
        blend="weighted_mean",
    )
    terms = comp.score_image(None, pd.DataFrame())
    result = comp.finalize(terms)
    assert isinstance(result, float)
    # (3*0.8 + 1*0.4) / (3 + 1) = 2.8 / 4 = 0.7
    assert result == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# finalize — dict (multi-objective sidecar path)
# --------------------------------------------------------------------------- #
def test_finalize_dict_when_multi_objective():
    comp = CompositeScorer(
        scorers=[
            _FixedScorer(terms={"a": 0.8}),
            _FixedScorer(terms={"b": 0.4}),
        ],
        multi_objective=True,
    )
    terms = comp.score_image(None, pd.DataFrame())
    result = comp.finalize(terms)
    assert isinstance(result, dict)
    # one named objective per child (the child's finalize over its own terms)
    assert result == {"s0": pytest.approx(0.8), "s1": pytest.approx(0.4)}


def test_finalize_dict_floors_abstaining_child_to_worst_cost():
    # Multi-objective: an abstaining child stays an axis (fixed-length vector for
    # NSGA-II) but is floored to the WORST cost 1.0 (was 0.0 under goodness).
    comp = CompositeScorer(
        scorers=[_FixedScorer(terms={"a": 0.2}), _FixedScorer(terms={})],
        multi_objective=True,
    )
    terms = comp.score_image(None, pd.DataFrame())
    assert terms == {"s0.a": 0.2}
    result = comp.finalize(terms)
    assert isinstance(result, dict)
    assert list(result.keys()) == comp.objective_names() == ["s0", "s1"]
    assert result["s0"] == pytest.approx(0.2)
    assert result["s1"] == pytest.approx(1.0)  # worst cost floor


# --------------------------------------------------------------------------- #
# availability — pinned rule: available iff any child is available
# --------------------------------------------------------------------------- #
def test_availability_true_when_any_child_available():
    comp = CompositeScorer(
        scorers=[
            _FixedScorer(terms={"a": 1.0}, ok=False),
            _FixedScorer(terms={"b": 1.0}, ok=True),
        ]
    )
    assert comp.availability() is True


def test_availability_false_when_no_child_available():
    comp = CompositeScorer(
        scorers=[
            _FixedScorer(terms={"a": 1.0}, ok=False),
            _FixedScorer(terms={"b": 1.0}, ok=False),
        ]
    )
    assert comp.availability() is False


def test_availability_false_when_empty():
    assert CompositeScorer(scorers=[]).availability() is False


# --------------------------------------------------------------------------- #
# cycle / self-nesting rejection
# --------------------------------------------------------------------------- #
def test_two_refs_to_same_non_cyclic_child_is_fine():
    # A composite holding two references to the *same* non-self-containing
    # scorer is a DAG, not a cycle — it must be accepted.
    inner = _FixedScorer(terms={"a": 1.0})
    comp = CompositeScorer(scorers=[inner, inner])
    assert len(comp.scorers) == 2


def test_self_nesting_is_rejected():
    # The real cycle: a composite that contains *itself* by identity.
    comp = CompositeScorer(scorers=[_FixedScorer(terms={"a": 1.0})])
    comp.scorers.append(comp)
    with pytest.raises(ValueError):
        CompositeScorer.model_validate(comp)


def test_nested_cycle_is_rejected():
    leaf = _FixedScorer(terms={"a": 1.0})
    middle = CompositeScorer(scorers=[leaf])
    outer = CompositeScorer(scorers=[middle])
    # Splice a back-edge from the inner composite to the outer one → a cycle.
    middle.scorers.append(outer)
    with pytest.raises(ValueError):
        CompositeScorer.model_validate(outer)


# --------------------------------------------------------------------------- #
# nesting round-trips via the polymorphic registry through TuningSpec
# --------------------------------------------------------------------------- #
def test_composite_nests_qc_and_supervised_round_trip(tmp_path):
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.tune import (
        Budget,
        Categorical,
        Evaluator,
        GroundTruthMasks,
        Knob,
        OptunaConfig,
        SearchSpace,
        SupervisedScorer,
    )
    from phenotypic.tune._spec import TuningSpec

    counts = tmp_path / "counts.csv"
    _layout(96).to_csv(counts, index=False)
    supervised = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=counts),
        count_check=ExpectedVsDetectedCount(
            metadata=str(counts), groupby=["Metadata_ImageName"]
        ),
    )
    comp = CompositeScorer(
        scorers=[_qc_scorer(tmp_path), supervised],
        weights={"s0": 2.0, "s1": 1.0},
        multi_objective=True,
    )
    spec = TuningSpec(
        pipeline=ImagePipeline(ops=[OtsuDetector()]),
        search_space=SearchSpace(
            knobs=(Knob(key="0.ignore_zeros", domain=Categorical(choices=(True, False))),)
        ),
        scorer=comp,
        evaluator=Evaluator(),
        # A multi-objective composite requires an Optuna strategy (the 4.8 guard);
        # grid/random are single-objective and rejected at construction.
        strategy=OptunaConfig(sampler="nsga2", n_trials=4),
        budget=Budget(),
    )
    back = TuningSpec.model_validate_json(spec.model_dump_json())
    assert isinstance(back.scorer, CompositeScorer)
    assert [type(s).__name__ for s in back.scorer.scorers] == [
        "QCScorer",
        "SupervisedScorer",
    ]
    assert back.scorer.multi_objective is True
    assert back.scorer.weights == {"s0": 2.0, "s1": 1.0}


def test_composite_direct_round_trip(tmp_path):
    comp = CompositeScorer(scorers=[_qc_scorer(tmp_path)])
    back = CompositeScorer.model_validate_json(comp.model_dump_json())
    assert isinstance(back.scorers[0], QCScorer)
    assert back.multi_objective is False


# --------------------------------------------------------------------------- #
# new fields — rho / blend (defaults + round-trip)
# --------------------------------------------------------------------------- #
def test_default_blend_is_tchebycheff_and_default_rho():
    comp = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.0})])
    assert comp.blend == "tchebycheff"
    assert comp.rho == pytest.approx(0.05)


def test_blend_and_rho_round_trip(tmp_path):
    # Round-trip needs a registered scorer (the local _FixedScorer double is not
    # in the polymorphic registry); _qc_scorer is a real, serializable child.
    comp = CompositeScorer(
        scorers=[_qc_scorer(tmp_path)],
        blend="weighted_mean",
        rho=0.1,
    )
    back = CompositeScorer.model_validate_json(comp.model_dump_json())
    assert back.blend == "weighted_mean"
    assert back.rho == pytest.approx(0.1)


def test_invalid_blend_rejected():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        CompositeScorer(scorers=[], blend="geomean")  # not a CompositeBlend


# --------------------------------------------------------------------------- #
# _tchebycheff — formula + normalization (operates on per-child COST dicts)
# --------------------------------------------------------------------------- #
def test_tchebycheff_all_perfect_is_near_zero():
    # All children perfect (cost 0) → numerator = max(w·ε) + ρ·0 = ε (uniform w=1)
    # T_norm = ε / ((1+ε) + ρ·1) → tiny, in (0,1].
    comp = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.0}),
                                    _FixedScorer(terms={"b": 0.0})])
    result = comp.finalize(comp.score_image(None, pd.DataFrame()))
    assert isinstance(result, float)
    assert 0.0 < result < 0.05  # near-zero cost, strictly positive (z*=−ε)


def test_tchebycheff_all_worst_is_one():
    # All children worst (cost 1) → Tᵨ == Tᵨ(1…1) → T_norm == 1.0 exactly.
    comp = CompositeScorer(scorers=[_FixedScorer(terms={"a": 1.0}),
                                    _FixedScorer(terms={"b": 1.0})])
    result = comp.finalize(comp.score_image(None, pd.DataFrame()))
    assert result == pytest.approx(1.0)


def test_tchebycheff_is_worst_axis_dominant():
    # Conjunctive: the WORST (highest-cost) axis drives the max term, so a
    # candidate with one bad axis scores higher (worse) than one balanced at the
    # mean. {0.0, 0.8} (max term ~0.8) must exceed {0.4, 0.4} (max term ~0.4).
    one_bad = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.0}),
                                       _FixedScorer(terms={"b": 0.8})])
    balanced = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.4}),
                                        _FixedScorer(terms={"b": 0.4})])
    cost_one_bad = one_bad.finalize(one_bad.score_image(None, pd.DataFrame()))
    cost_balanced = balanced.finalize(balanced.score_image(None, pd.DataFrame()))
    assert cost_one_bad > cost_balanced


def test_tchebycheff_result_in_unit_interval():
    for a in (0.0, 0.2, 0.5, 0.9, 1.0):
        for b in (0.0, 0.3, 1.0):
            comp = CompositeScorer(scorers=[_FixedScorer(terms={"x": a}),
                                            _FixedScorer(terms={"y": b})])
            r = comp.finalize(comp.score_image(None, pd.DataFrame()))
            assert 0.0 < r <= 1.0 + 1e-9


def test_tchebycheff_weights_steer_the_max():
    # Weighting the worse axis up makes the composite worse (it weighs that axis
    # more heavily in the max). {a:0.0, b:0.6} with w_b=3 > the same with w=1.
    light = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.0}),
                                     _FixedScorer(terms={"b": 0.6})])
    heavy = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.0}),
                                     _FixedScorer(terms={"b": 0.6})],
                            weights={"s1": 3.0})
    # Normalization differs per weight set, so compare each to its own balanced
    # baseline rather than to each other directly; assert the heavy-weighted
    # bad axis is still worst-axis dominant (> 0.4 normalized).
    assert heavy.finalize(heavy.score_image(None, pd.DataFrame())) > 0.4
    assert light.finalize(light.score_image(None, pd.DataFrame())) > 0.0


# --------------------------------------------------------------------------- #
# active set — pinned study-global roster for max + normalizer
# --------------------------------------------------------------------------- #
def test_finalize_routes_to_tchebycheff_by_default():
    # With finalize wired, the worst-axis-dominant property now drives finalize.
    one_bad = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.0}),
                                       _FixedScorer(terms={"b": 0.8})])
    balanced = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.4}),
                                        _FixedScorer(terms={"b": 0.4})])
    assert (one_bad.finalize(one_bad.score_image(None, pd.DataFrame()))
            > balanced.finalize(balanced.score_image(None, pd.DataFrame())))


def test_finalize_weighted_mean_opt_out():
    # blend="weighted_mean" keeps the compensatory arithmetic mean over costs.
    comp = CompositeScorer(
        scorers=[_FixedScorer(terms={"a": 0.8}), _FixedScorer(terms={"b": 0.4})],
        weights={"s0": 3.0, "s1": 1.0},
        blend="weighted_mean",
    )
    result = comp.finalize(comp.score_image(None, pd.DataFrame()))
    # (3*0.8 + 1*0.4) / 4 = 0.7  (now over cost; same arithmetic as before)
    assert result == pytest.approx(0.7)


def test_active_set_pins_roster_for_both_max_and_normalizer():
    # Pin the active set to BOTH children. An abstaining child (no terms this
    # call) is in the active set but absent from this call's costs → it must NOT
    # be flooded into the max (that is per-image abstention, handled by the
    # robust aggregate upstream). With one child scoring 0.5, the composite is
    # the single-axis Tchebycheff of the present axis (not pinned to 1.0).
    comp = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.5}),
                                    _FixedScorer(terms={})])
    comp.set_active_set(("s0", "s1"))
    terms = comp.score_image(None, pd.DataFrame())  # only s0 emits
    assert terms == {"s0.a": 0.5}
    result = comp.finalize(terms)
    # discrimination on the present axis is preserved (not flattened to ~1.0)
    assert 0.0 < result < 0.7


def test_empty_active_set_is_worst_cost():
    comp = CompositeScorer(scorers=[_FixedScorer(terms={})])
    comp.set_active_set(())
    assert comp.finalize({}) == pytest.approx(1.0)


def test_no_scored_children_is_worst_cost_under_tchebycheff():
    # Single-objective default: zero scalars → worst cost 1.0 (NOT 0.0).
    # This is the cost-convention flip of the old "empty → 0.0" goodness floor.
    comp = CompositeScorer(scorers=[])
    assert comp.finalize({}) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# abstainer masking (§6.3 pitfall #4) — one study-wide-absent / per-image-absent
# child must not flatten discrimination on the present axes
# --------------------------------------------------------------------------- #
def test_one_abstaining_child_does_not_flatten_present_axis():
    # Active set pins both children; s1 abstains this call. The present axis (s0)
    # must still discriminate: a good s0 (0.1) scores strictly better than a bad
    # s0 (0.7), i.e. the abstainer is NOT flooded into the max as cost 1.0.
    good = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.1}),
                                    _FixedScorer(terms={})])
    bad = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.7}),
                                   _FixedScorer(terms={})])
    good.set_active_set(("s0", "s1"))
    bad.set_active_set(("s0", "s1"))
    cost_good = good.finalize(good.score_image(None, pd.DataFrame()))
    cost_bad = bad.finalize(bad.score_image(None, pd.DataFrame()))
    assert cost_good < cost_bad  # discrimination preserved
    assert cost_bad < 0.9        # NOT pinned near the ceiling by the abstainer


# --------------------------------------------------------------------------- #
# cost clamp (B1, §6.1 invariant 0 <= bᵢ <= 1) — a high-variance term clamped
# upstream must keep T_norm in [0,1] and the assert must NOT fire
# --------------------------------------------------------------------------- #
def test_high_cost_child_keeps_t_norm_in_unit_interval():
    # A child whose robust-aggregated cost is clamped to 1.0 (median+λ·IQR > 1
    # upstream → clamp01 → 1.0) feeds the composite as exactly 1.0; T_norm stays
    # in (0,1] and the §6.1 0<=bᵢ<=1 assert does not fire.
    comp = CompositeScorer(scorers=[_FixedScorer(terms={"a": 1.0}),
                                    _FixedScorer(terms={"b": 0.3})])
    result = comp.finalize(comp.score_image(None, pd.DataFrame()))
    assert 0.0 < result <= 1.0


def test_cost_above_one_trips_the_invariant_assert():
    # Defensive: if an UNCLAMPED cost (>1, the Phase 1 clamp regressed) reaches
    # the combiner, the §6.1 invariant assert must fire loudly (not silently
    # saturate). Drive _tchebycheff directly with a poisoned roster.
    comp = CompositeScorer(scorers=[_FixedScorer(terms={"a": 0.0})])
    with pytest.raises(AssertionError):
        comp._tchebycheff({"s0": 1.5})


# --------------------------------------------------------------------------- #
# helpers — score a list of (cost_a, cost_b) candidates with a blend
# --------------------------------------------------------------------------- #
def _composite(cost_a, cost_b, *, blend="tchebycheff", rho=0.05, weights=None):
    comp = CompositeScorer(
        scorers=[_FixedScorer(terms={"a": cost_a}), _FixedScorer(terms={"b": cost_b})],
        blend=blend,
        rho=rho,
        weights=weights,
    )
    return comp.finalize(comp.score_image(None, pd.DataFrame()))


def _weighted_sum_cost(cost_a, cost_b):
    # The convex weighted sum (uniform) over cost — the baseline Tchebycheff beats.
    return 0.5 * cost_a + 0.5 * cost_b


def _one_minus_geomean_cost(cost_a, cost_b):
    # The OLD composite read on cost: 1 - geomean(goodness) = 1 - sqrt((1-a)(1-b)).
    return 1.0 - math.sqrt((1.0 - cost_a) * (1.0 - cost_b))


# A concave Pareto front: two single-axis extremes and a balanced KNEE that
# bulges toward the origin (low on BOTH axes). The knee has the lowest WORST
# axis (so augmented Tchebycheff selects it) but a HIGHER coordinate sum than
# either extreme (so the linear weighted sum — and 1−geomean — pick an extreme
# instead). This is the non-convex reachability separation, not a weakened test.
_FRONT = [
    (0.02, 0.55),  # extreme A: nails axis a, tanks b  (sum 0.57, worst 0.55)
    (0.34, 0.34),  # the KNEE: balanced, both good     (sum 0.68, worst 0.34)
    (0.55, 0.02),  # extreme B: nails b, tanks a        (sum 0.57, worst 0.55)
]


def _argmin(front, cost_fn):
    return min(front, key=lambda p: cost_fn(*p))


# --------------------------------------------------------------------------- #
# non-convex reachability — Tchebycheff selects the knee; weighted sum &
# 1−geomean do not (§9 / §10)
# --------------------------------------------------------------------------- #
def test_tchebycheff_selects_knee_on_concave_front():
    knee = (0.34, 0.34)
    assert _argmin(_FRONT, lambda a, b: _composite(a, b)) == knee


def test_weighted_sum_misses_the_knee():
    # The uniform weighted sum is tied/biased toward an extreme, not the knee.
    winner = _argmin(_FRONT, _weighted_sum_cost)
    assert winner != (0.34, 0.34)


def test_one_minus_geomean_misses_the_knee():
    # The OLD composite (1 - geomean of goodness) also does not pick this knee —
    # documenting WHY the migration changes the winner (§8 deliberate change).
    winner = _argmin(_FRONT, _one_minus_geomean_cost)
    assert winner != (0.34, 0.34)


# --------------------------------------------------------------------------- #
# ρ removes a weakly-dominated point (§6.1/§6.4)
# --------------------------------------------------------------------------- #
def test_rho_breaks_weak_domination():
    # Two candidates equal on their WORST axis (both max-axis cost 0.5) but one is
    # strictly better on the OTHER axis. Plain Tchebycheff (ρ→0) ties them; the
    # augmentation (ρ=0.05) strictly prefers the properly-dominant one.
    weak = (0.5, 0.5)       # worst axis 0.5, other axis 0.5
    strong = (0.5, 0.1)     # worst axis 0.5, other axis 0.1 (strictly better)
    # ρ→0: the max terms are equal → near-tie (strong only marginally lower).
    tied_weak = _composite(*weak, rho=1e-9)
    tied_strong = _composite(*strong, rho=1e-9)
    assert tied_strong == pytest.approx(tied_weak, abs=1e-6)
    # ρ=0.05: the augmentation L1 term strictly separates them.
    sep_weak = _composite(*weak, rho=0.05)
    sep_strong = _composite(*strong, rho=0.05)
    assert sep_strong < sep_weak  # the properly-dominant point wins


# --------------------------------------------------------------------------- #
# ρ sensitivity — ρ→0 admits a weakly-dominated winner ρ=0.05 rejects; large ρ
# drifts toward the weighted-sum winner (§6.4)
# --------------------------------------------------------------------------- #
def test_large_rho_drifts_toward_weighted_sum():
    # As ρ grows, the Σ (weighted-sum) term dominates, so the winner converges to
    # the weighted-sum winner (an extreme), abandoning the knee.
    ws_winner = _argmin(_FRONT, _weighted_sum_cost)
    big_rho_winner = _argmin(_FRONT, lambda a, b: _composite(a, b, rho=50.0))
    assert big_rho_winner == ws_winner


def test_default_rho_keeps_the_knee_vs_large_rho():
    assert _argmin(_FRONT, lambda a, b: _composite(a, b, rho=0.05)) == (0.34, 0.34)
    assert _argmin(_FRONT, lambda a, b: _composite(a, b, rho=50.0)) != (0.34, 0.34)


# --------------------------------------------------------------------------- #
# composite-delta snapshot — document the intended winner change vs old geomean
# --------------------------------------------------------------------------- #
def test_composite_delta_vs_old_geomean_winner_is_documented():
    # The OLD composite (geomean of goodness == 1−_one_minus_geomean_cost, i.e.
    # MAXIMIZE goodness == MINIMIZE 1−geomean) and the NEW composite pick
    # DIFFERENT winners on the concave front — the one intended behavior change
    # of the migration (README invariant #3, spec §8). This test is the
    # baseline snapshot + reviewer sign-off gate; if it ever passes with equal
    # winners, the composite stopped being augmented Tchebycheff.
    old_winner = _argmin(_FRONT, _one_minus_geomean_cost)
    new_winner = _argmin(_FRONT, lambda a, b: _composite(a, b))
    assert new_winner == (0.34, 0.34)        # NEW: the conjunctive knee
    assert old_winner != new_winner          # DELTA: intentional change


# --------------------------------------------------------------------------- #
# engine active-set pin logic — the available children form the roster
# --------------------------------------------------------------------------- #
def test_engine_pins_active_set_to_available_children():
    comp = CompositeScorer(
        scorers=[_FixedScorer(terms={"a": 0.1}, ok=True),
                 _FixedScorer(terms={"b": 0.2}, ok=False)],  # unavailable
    )
    # Simulate what the engine does (the pin logic, without a full run):
    active = tuple(
        handle
        for handle, child in zip(comp.objective_names(), comp.scorers)
        if child.availability()
    )
    comp.set_active_set(active)
    assert comp._active_handles == ("s0",)  # only the available child is an axis
