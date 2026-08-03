"""GUI lifecycle refresh delegates to the shared plotting coordinator."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from phenotypic.gui import _plot_refresh
from phenotypic.plotting._pipeline import AnalysisResult


class _CoordinatorSpy:
    def __init__(self) -> None:
        self.measurements = None
        self.analysis_call = None
        self.analysis_calls = []
        self.refreshed_analysis_ids = ()
        self.qc_call = None
        self.dependent_qc_calls = []

    def emit_measurements(self, measurements):
        self.measurements = measurements

    def emit_analyses(self, measurements, registry, **kwargs):
        self.analysis_call = (measurements, registry, kwargs)
        self.analysis_calls.append(self.analysis_call)
        return self.refreshed_analysis_ids

    def emit_qc(self, measurements, registry, **kwargs):
        self.qc_call = (measurements, registry, kwargs)

    def emit_dependent_qc(self, measurements, registry, **kwargs):
        self.dependent_qc_calls.append((measurements, registry, kwargs))


def _layout(tmp_path: Path) -> SimpleNamespace:
    qc_dir = tmp_path / "qc"
    return SimpleNamespace(
        deliverables_base=tmp_path,
        output_root=None,
        plots_dir=tmp_path / "plots",
        qc_duckdb=qc_dir / "qc.duckdb",
        qc_review_state_path=qc_dir / "review_state.json",
    )


def test_measurement_refresh_passes_exact_current_frame(monkeypatch, tmp_path) -> None:
    spy = _CoordinatorSpy()
    monkeypatch.setattr(
        _plot_refresh,
        "_coordinator",
        lambda *_args, **_kwargs: spy,
    )
    frame = pd.DataFrame({"Size_Area": [1.0]})

    _plot_refresh.refresh_measurement_plots(object(), _layout(tmp_path), frame)

    assert spy.measurements is frame
    seen_measurements, _registry, kwargs = spy.analysis_call
    assert seen_measurements is frame
    assert kwargs["updated_input"].kind == "measurements"
    assert kwargs["refresh_producers"] is True
    assert len(spy.dependent_qc_calls) == 1
    assert spy.dependent_qc_calls[0][2]["updated_input"].kind == "measurements"


def test_measurement_refresh_fans_out_refreshed_analysis_dependencies(
    monkeypatch, tmp_path
) -> None:
    spy = _CoordinatorSpy()
    spy.refreshed_analysis_ids = ("LinearLagModel",)
    monkeypatch.setattr(
        _plot_refresh,
        "_coordinator",
        lambda *_args, **_kwargs: spy,
    )
    frame = pd.DataFrame({"Size_Area": [1.0]})

    _plot_refresh.refresh_measurement_plots(object(), _layout(tmp_path), frame)

    assert len(spy.analysis_calls) == 2
    assert spy.analysis_calls[1][2]["updated_input"].analysis_id == "LinearLagModel"
    assert len(spy.dependent_qc_calls) == 2
    assert (
        spy.dependent_qc_calls[1][2]["updated_input"].analysis_id
        == "LinearLagModel"
    )


def test_analysis_refresh_registers_exact_gui_result(monkeypatch, tmp_path) -> None:
    spy = _CoordinatorSpy()
    monkeypatch.setattr(
        _plot_refresh,
        "_coordinator",
        lambda *_args, **_kwargs: spy,
    )
    measurements = pd.DataFrame({"Size_Area": [1.0]})
    analysis = pd.DataFrame({"lag": [2.0]})
    producer = object()
    result = AnalysisResult(
        analysis_id="LinearLagModel",
        table=analysis,
        producer=producer,
    )

    _plot_refresh.refresh_analysis_plots(
        object(), _layout(tmp_path), measurements, result
    )

    seen_measurements, registry, kwargs = spy.analysis_call
    assert seen_measurements is measurements
    registered = registry.get("LinearLagModel")
    assert registered is not None
    assert registered.table is analysis
    assert registered.producer is producer
    assert kwargs["updated_input"].analysis_id == "LinearLagModel"
    assert len(spy.dependent_qc_calls) == 1
    assert (
        spy.dependent_qc_calls[0][2]["updated_input"].analysis_id
        == "LinearLagModel"
    )


def test_qc_refresh_threads_modules_database_and_review_snapshot(
    monkeypatch, tmp_path
) -> None:
    spy = _CoordinatorSpy()
    monkeypatch.setattr(
        _plot_refresh,
        "_coordinator",
        lambda *_args, **_kwargs: spy,
    )
    layout = _layout(tmp_path)
    layout.qc_review_state_path.parent.mkdir(parents=True)
    layout.qc_review_state_path.write_text(
        json.dumps({"qc-1": {"reviewed": ["A"]}}),
        encoding="utf-8",
    )
    module = SimpleNamespace(instance_id="qc-1", check=object())
    measurements = pd.DataFrame({"Size_Area": [1.0]})

    _plot_refresh.refresh_qc_plots(
        object(), layout, measurements, [module]
    )

    seen_measurements, _registry, kwargs = spy.qc_call
    assert seen_measurements is measurements
    assert kwargs["successful_modules"] == {"qc-1": module}
    assert kwargs["qc_database"] == layout.qc_duckdb
    assert kwargs["review_state"] == {"qc-1": {"reviewed": ["A"]}}
