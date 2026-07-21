"""ModelFitter emits qualified headers and implements PlotAnalysis."""

import inspect
from unittest.mock import patch

import matplotlib
import pandas as pd
import pytest

from phenotypic.analysis import LinearLagModel
from phenotypic.analysis.abc_ import ModelFitter
from phenotypic.abc_.plotting import PlotAnalysis
from phenotypic.schema import LINEAR_LAG_MODEL, MODEL_METRICS, qualified_header

matplotlib.use("Agg")


def _toy_df() -> pd.DataFrame:
    rows = []
    for strain in ("A", "B"):
        for t in range(8):
            rows.append(
                {
                    "MetadataGenetic_Strain": strain,
                    "MetadataCulture_Time": float(t),
                    "Shape_Area": 1.0 + 2.0 * t,
                }
            )
    return pd.DataFrame(rows)


def test_analyze_returns_metric_qualified_columns():
    model = LinearLagModel(on="Shape_Area", groupby=["MetadataGenetic_Strain"])
    res = model.analyze(_toy_df())
    assert qualified_header(LINEAR_LAG_MODEL.v, "Area") in res.columns
    assert qualified_header(MODEL_METRICS.RMSE, "Area") in res.columns
    assert "LinearLagModel_v" not in res.columns  # hard cutover, no legacy header


def test_results_returns_the_qualified_frame():
    model = LinearLagModel(on="Shape_Area", groupby=["MetadataGenetic_Strain"])
    model.analyze(_toy_df())
    assert qualified_header(LINEAR_LAG_MODEL.s0, "Area") in model.results().columns


def test_show_works_after_qualified_analyze():
    model = LinearLagModel(on="Shape_Area", groupby=["MetadataGenetic_Strain"])
    model.analyze(_toy_df())
    fig, ax = model.show()
    assert ax is not None


def test_model_fitter_is_plot_analysis_with_keyword_only_inspect():
    model = LinearLagModel(on="Shape_Area", groupby=["MetadataGenetic_Strain"])

    assert isinstance(model, PlotAnalysis)
    assert "dash" not in ModelFitter.__dict__
    params = inspect.signature(model.inspect).parameters
    assert params["for_save"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["tmax"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["criteria"].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        model.inspect(5)  # type: ignore[misc]


def test_inspect_and_report_delegate_to_one_plotly_builder():
    model = LinearLagModel(on="Shape_Area", groupby=["MetadataGenetic_Strain"])
    sentinel = object()

    with patch.object(
        ModelFitter, "_build_plotly_figure", return_value=sentinel
    ) as builder:
        assert model.inspect(for_save=True, tmax=5) is sentinel
        builder.assert_called_once_with(
            tmax=5,
            criteria=None,
            figsize=(6, 4),
            cmap="tab20",
            legend=True,
        )

    with patch.object(
        ModelFitter, "_build_plotly_figure", return_value=sentinel
    ) as builder:
        assert model.report(tmax=3, legend=False) is sentinel
        builder.assert_called_once_with(tmax=3, legend=False)


def test_inspect_and_report_reuse_analyzed_state_without_refitting():
    model = LinearLagModel(on="Shape_Area", groupby=["MetadataGenetic_Strain"])
    expected = model.analyze(_toy_df()).copy(deep=True)

    with patch.object(
        LinearLagModel,
        "analyze",
        side_effect=AssertionError("plotting must not refit"),
    ):
        inspected = model.inspect()
        reported = model.report()

    assert len(inspected.data) == len(reported.data)
    pd.testing.assert_frame_equal(model.results(), expected)
