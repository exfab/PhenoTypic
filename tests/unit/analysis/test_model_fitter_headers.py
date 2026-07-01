"""ModelFitter emits metric-qualified headers and still plots."""

import matplotlib
import pandas as pd

from phenotypic.analysis import LinearLagModel
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
