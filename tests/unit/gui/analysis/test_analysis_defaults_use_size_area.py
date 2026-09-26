from phenotypic._gui.analysis import _callbacks
from phenotypic.schema import SIZE


def test_analysis_defaults_target_size_area():
    tables = (_callbacks._FILTER_DEFAULTS, _callbacks._EDGE_DEFAULTS, _callbacks._MODEL_DEFAULTS)
    ons = [params["on"] for table in tables for params in table.values() if "on" in params]
    assert ons and set(ons) == {str(SIZE.AREA)}
