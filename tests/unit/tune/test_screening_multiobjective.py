"""4.0d — per-objective parameter importance (multi-objective, plan §0a).

``compute_param_importance`` / ``compute_param_importance_report`` gain
``objective: str | None``. ``None`` ranks against ``Trial.score`` (the unchanged
single-objective path); a name ranks against ``Trial.objectives[name]``, skipping
trials that lack the objective. The single-objective path must stay byte-identical.
"""
from __future__ import annotations

from phenotypic.tune._screening import (
    ImportanceReport,
    compute_param_importance,
    compute_param_importance_report,
)
from phenotypic.tune._study_store import StudyStore, Trial


def _single_objective_store() -> StudyStore:
    # score depends entirely on `a` (True→1.0, False→0.0); `b` is noise.
    store = StudyStore()
    for i in range(24):
        a = i % 2 == 0
        store.append(Trial(
            number=i, params={"a": a, "b": (i // 2) % 3},
            score=1.0 if a else 0.0, terms={"Count": 1.0 if a else 0.0},
            n_images=2,
        ))
    return store


def _multi_objective_store() -> StudyStore:
    # Dice is driven by `a`; IoU is driven (oppositely) by `b==0`. The two
    # objectives have different driving params so per-objective importance differs.
    store = StudyStore()
    for i in range(24):
        a = i % 2 == 0
        b = (i // 2) % 3
        dice = 1.0 if a else 0.0
        iou = 1.0 if b == 0 else 0.0
        store.append(Trial(
            number=i, params={"a": a, "b": b},
            score=(dice + iou) / 2.0,
            terms={"Dice": dice, "IoU": iou},
            n_images=2,
            objectives={"Dice": dice, "IoU": iou},
        ))
    return store


def test_default_objective_none_ranks_against_score_unchanged():
    # objective=None is the legacy single-objective path: identical result.
    store = _single_objective_store()
    baseline = compute_param_importance(store)
    explicit_none = compute_param_importance(store, objective=None)
    assert explicit_none == baseline
    assert explicit_none["a"] > explicit_none["b"]


def test_per_objective_importance_targets_named_objective():
    store = _multi_objective_store()
    dice_imp = compute_param_importance(store, objective="Dice")
    iou_imp = compute_param_importance(store, objective="IoU")
    # Dice is driven by `a`; IoU is driven by `b`.
    assert dice_imp["a"] > dice_imp["b"]
    assert iou_imp["b"] > iou_imp["a"]


def test_per_objective_report_carries_method():
    store = _multi_objective_store()
    report = compute_param_importance_report(store, objective="Dice")
    assert isinstance(report, ImportanceReport)
    assert report.method == "rf-permutation"
    assert set(report.importances) == {"a", "b"}


def test_objective_missing_on_some_trials_is_skipped():
    # Trials lacking the requested objective are dropped, not crashed on.
    store = StudyStore()
    for i in range(24):
        a = i % 2 == 0
        obj = {"Dice": 1.0 if a else 0.0}
        # Every 5th trial has no objectives at all (e.g. a legacy/failed mix).
        store.append(Trial(
            number=i, params={"a": a, "b": (i // 2) % 3},
            score=1.0 if a else 0.0, terms={"Dice": 1.0 if a else 0.0},
            n_images=2,
            objectives=None if i % 5 == 0 else obj,
        ))
    imp = compute_param_importance(store, objective="Dice")
    assert set(imp) == {"a", "b"}
    assert imp["a"] > imp["b"]


def test_objective_absent_everywhere_yields_empty():
    # No trial carries the objective → nothing to fit → empty importances.
    store = _single_objective_store()  # objectives all None
    assert compute_param_importance(store, objective="Dice") == {}
