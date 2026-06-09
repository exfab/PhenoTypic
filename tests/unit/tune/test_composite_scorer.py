"""Tests for the composite scorer — Phase 4 chunk B (4.5).

``CompositeScorer`` nests a ``list[Scorer]`` (via the polymorphic ``ScorerField``)
and blends them. These tests pin: per-child **prefixed** term merging
(collision-free), the scalar weighted/geometric ``finalize`` blend (default), the
``dict`` ``finalize`` when ``multi_objective=True`` (the sidecar path), nesting
round-trip through ``TuningSpec``, cycle/self-nesting rejection, and the pinned
``availability`` rule.
"""
from __future__ import annotations

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
    """A stateless scorer returning preset terms; available iff ``ok``."""

    terms: dict[str, float]
    ok: bool = True

    def score_image(self, image, measurements) -> dict[str, float]:
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
def test_finalize_scalar_geometric_blend_default():
    # Default (no weights) → geometric mean of the per-child finalized scalars.
    # child0 mean = (0.81)=0.81 ; child1 mean = 0.49 ; geo = sqrt(0.81*0.49)=0.63
    comp = CompositeScorer(
        scorers=[
            _FixedScorer(terms={"a": 0.81}),
            _FixedScorer(terms={"b": 0.49}),
        ]
    )
    terms = comp.score_image(None, pd.DataFrame())
    result = comp.finalize(terms)
    assert isinstance(result, float)
    assert result == pytest.approx((0.81 * 0.49) ** 0.5)


def test_finalize_scalar_weighted_blend():
    # Weights keyed by child prefix → weighted arithmetic mean of child scalars.
    comp = CompositeScorer(
        scorers=[
            _FixedScorer(terms={"a": 0.8}),
            _FixedScorer(terms={"b": 0.4}),
        ],
        weights={"s0": 3.0, "s1": 1.0},
    )
    terms = comp.score_image(None, pd.DataFrame())
    result = comp.finalize(terms)
    assert isinstance(result, float)
    # (3*0.8 + 1*0.4) / (3 + 1) = 2.8 / 4 = 0.7
    assert result == pytest.approx(0.7)


def test_finalize_scalar_empty_is_zero():
    comp = CompositeScorer(scorers=[])
    assert comp.finalize({}) == 0.0


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


def test_finalize_dict_floors_abstaining_child_to_zero():
    # A child that emits no terms (abstains this run — e.g. a SupervisedScorer in
    # mask tier whose GT masks are all missing) must still appear in the
    # multi-objective sidecar, floored to 0.0 (the higher-is-better worst score)
    # rather than dropped. The dict keys + order must stay invariant and exactly
    # match objective_names() / the Optuna study's fixed `directions` — a dropped
    # axis makes the NSGA-II value vector the wrong length and crashes `tell`.
    comp = CompositeScorer(
        scorers=[
            _FixedScorer(terms={"a": 0.8}),
            _FixedScorer(terms={}),  # abstains: contributes no terms
        ],
        multi_objective=True,
    )
    terms = comp.score_image(None, pd.DataFrame())
    assert terms == {"s0.a": 0.8}  # only the scoring child emits a prefixed term
    result = comp.finalize(terms)
    assert isinstance(result, dict)
    assert list(result.keys()) == comp.objective_names() == ["s0", "s1"]
    assert result["s0"] == pytest.approx(0.8)
    assert result["s1"] == 0.0


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
