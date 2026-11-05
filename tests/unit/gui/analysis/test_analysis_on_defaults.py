"""The ``on`` column an analysis card pre-selects when added from its dropdown.

Filter and edge cards target the colony area; every growth-model card targets
the colony's robust mean radius.
"""

import pytest

from phenotypic._gui.analysis import _callbacks
from phenotypic.schema import SIZE


def test_filter_and_edge_defaults_target_size_area():
    tables = (_callbacks._FILTER_DEFAULTS, _callbacks._EDGE_DEFAULTS)
    ons = [params["on"] for table in tables for params in table.values()]
    assert ons and set(ons) == {str(SIZE.AREA)}


@pytest.mark.parametrize(
    "model_name", ["LogGrowthModel", "LinearLagModel", "LinearCapAndLagModel"]
)
def test_growth_model_defaults_target_robust_mean_radius(model_name):
    """Each growth model builds from its defaults and fits the radius."""
    assert set(_callbacks._MODEL_DEFAULTS) == {
        "LogGrowthModel", "LinearLagModel", "LinearCapAndLagModel",
    }
    model = _callbacks._instantiate("model", model_name)
    assert model is not None, f"{model_name} failed to build from its defaults"
    assert model.on == str(SIZE.ROBUST_MEAN_RADIUS)
