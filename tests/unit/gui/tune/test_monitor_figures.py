"""Unit tests for the Monitor figure builders + Pareto-visibility helper.

The Monitor figures are pure builders over a trial list / importance map, so
they unit-test headless (no Dash, no live study). We assert the structural
contract the poll callback relies on: the objective figure's best-trace is
monotone non-decreasing, the importance figure has one bar per param, and the
Pareto card is hidden for a single-objective run.
"""
from __future__ import annotations

from pathlib import Path

from phenotypic.gui.tune._run_root import TuneRunRoot
from phenotypic.tune._study_store import Trial


def _trial(number: int, score: float) -> Trial:
    return Trial(number=number, params={"thresh": float(number)}, score=score, terms={}, n_images=3)


def _single_objective_root(path: Path) -> TuneRunRoot:
    return TuneRunRoot(
        path=path,
        trials_path=None,
        storage_url=None,
        study_name="tune",
        directions=None,
        images_dir=None,
        best_pipeline_path=path / "best_pipeline.json",
    )


def test_build_objective_figure_best_trace_is_monotone_non_increasing() -> None:
    from phenotypic.gui.tune._study_read import build_objective_figure

    trials = [_trial(0, 0.7), _trial(1, 0.5), _trial(2, 0.6), _trial(3, 0.3)]
    fig = build_objective_figure(trials)
    best_traces = [
        tr for tr in fig.data if getattr(tr, "mode", None) and "lines" in tr.mode
    ]
    assert best_traces, "expected a running-best line trace"
    ys = list(best_traces[0].y)
    assert ys == [0.7, 0.5, 0.5, 0.3]
    assert ys == sorted(ys, reverse=True)  # cost: non-increasing
    # y-axis is relabeled to cost (lower is better).
    assert "cost" in fig.layout.yaxis.title.text.lower()


def test_build_objective_figure_empty_is_safe() -> None:
    from phenotypic.gui.tune._study_read import build_objective_figure

    fig = build_objective_figure([])
    assert fig is not None  # no raise on an empty journal


def test_build_importance_figure_one_bar_per_param() -> None:
    from phenotypic.gui.tune._study_read import build_importance_figure

    importances = {"thresh": 0.6, "min_size": 0.3, "sigma": 0.1}
    fig = build_importance_figure(importances)

    bar_traces = [tr for tr in fig.data if tr.type == "bar"]
    assert len(bar_traces) == 1
    # One bar per param: the single bar trace carries one x/y entry per param.
    assert len(bar_traces[0].x) == len(importances)
    assert set(bar_traces[0].x) == set(importances.keys())


def test_monitor_pareto_visible_false_for_single_objective(tmp_path: Path) -> None:
    from phenotypic.gui.tune._study_read import monitor_pareto_visible

    root = _single_objective_root(tmp_path)
    assert monitor_pareto_visible(root) is False


def test_monitor_pareto_visible_true_for_multi_objective(tmp_path: Path) -> None:
    from phenotypic.gui.tune._study_read import monitor_pareto_visible

    root = TuneRunRoot(
        path=tmp_path,
        trials_path=None,
        storage_url=None,
        study_name="tune",
        directions=["maximize", "maximize"],
        images_dir=None,
        best_pipeline_path=tmp_path / "best_pipeline.json",
    )
    assert monitor_pareto_visible(root) is True
