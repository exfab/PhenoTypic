"""Public import surface for concrete plotting models."""

from __future__ import annotations


def test_plotting_root_actively_exports_only_concrete_models() -> None:
    import phenotypic.plotting as plotting

    expected = {
        "PlotColonyMetricOverTime",
        "PlotDetectModes",
        "PlotDiagnostics",
        "PlotMeasTimeSeries",
    }

    assert set(plotting.__all__) == expected
    assert not hasattr(plotting, "__getattr__")
    for name in expected:
        assert name in vars(plotting)


def test_plotting_root_does_not_export_runtime_infrastructure() -> None:
    import phenotypic.plotting as plotting

    for name in (
        "AnalysisRegistry",
        "FigureAdapter",
        "PlotBinding",
        "PlotCoordinator",
        "PlotOutput",
        "PlotPage",
        "publish_plot_output",
    ):
        assert not hasattr(plotting, name)
