"""Contract tests for the EdgeCorrection abstract base."""
import abc

import pytest

from phenotypic.analysis.abc_ import EdgeCorrection, SetAnalyzer


def test_edge_correction_is_abstract_setanalyzer():
    assert issubclass(EdgeCorrection, SetAnalyzer)
    assert issubclass(EdgeCorrection, abc.ABC)
    with pytest.raises(TypeError):
        EdgeCorrection(on="Shape_Area", groupby=["Metadata_Strain"])


def test_edge_correction_validates_grid():
    class _Concrete(EdgeCorrection):
        def _group_config(self):
            return {}

        @staticmethod
        def _apply2group_func(group, **config):
            return group

    with pytest.raises(ValueError):
        _Concrete(on="Shape_Area", groupby=["g"], connectivity=5)
    with pytest.raises(ValueError):
        _Concrete(on="Shape_Area", groupby=["g"], nrows=0)
    ok = _Concrete(on="Shape_Area", groupby=["g"], nrows=8, ncols=12, connectivity=8)
    assert ok.ncols == 12
