"""Every producer resolves every column's unit, including fitted rates (plan 02, Task 5).

``output_unit(member, on)`` sits next to ``output_header``. A static unit
resolves to itself; an input-dependent one (a growth rate, a model metric, a
QC statistic, an edge-corrected value) resolves against ``unit_of_column(on)``.
It never makes a unit up: an unknown or unit-less ``on`` is refused by name.
"""

from __future__ import annotations

import pytest

from phenotypic.abc_ import MeasureFeatures
from phenotypic.analysis import EdgeCorrector, LinearCapAndLagModel, LinearLagModel
from phenotypic.analysis.abc_ import SetAnalyzer
from phenotypic.measure import MeasureSize
from phenotypic.schema import (
    EDGE_CORRECTION,
    GENETIC,
    LINEAR_CAP_AND_LAG_MODEL,
    LINEAR_LAG_MODEL,
    MODEL_METRICS,
    NEIGHBOR_DIST,
    SIZE,
    ResolvedUnit,
)
from phenotypic.util import measurement_producers
from phenotypic.util._measurement_outputs import unit_of_column


def test_static_member_unit():
    assert MeasureSize.output_unit(SIZE.AREA) == ResolvedUnit("px²", 2)


def test_fitted_rate_takes_the_unit_of_its_column():
    assert LinearLagModel.output_unit(LINEAR_LAG_MODEL.v, on="Size_RobustMeanRadius") == ResolvedUnit("px/time", 1)
    assert LinearLagModel.output_unit(LINEAR_LAG_MODEL.v, on="Size_Area") == ResolvedUnit("px²/time", 2)
    assert LinearLagModel.output_unit(MODEL_METRICS.MSE, on="Size_RobustMeanRadius") == ResolvedUnit("px²", 2)


def test_per_input_is_the_reciprocal_of_its_column():
    # beta multiplies (smax - s), so it is 1/(unit of s).
    beta = LINEAR_CAP_AND_LAG_MODEL.beta
    assert LinearCapAndLagModel.output_unit(beta, on="Size_RobustMeanRadius") == ResolvedUnit("1/px", -1)
    assert LinearCapAndLagModel.output_unit(beta, on="Size_Area") == ResolvedUnit("1/px²", -2)


def test_static_units_of_an_analyzer_ignore_on():
    assert LinearLagModel.output_unit(LINEAR_LAG_MODEL.lam, on="Size_Area") == ResolvedUnit("time", 0)
    assert LinearLagModel.output_unit(MODEL_METRICS.R2) == ResolvedUnit("", 0)
    assert MeasureSize.output_unit(SIZE.AREA, on="my_custom_col") == ResolvedUnit("px²", 2)


def test_edge_corrected_value_takes_the_unit_of_its_column():
    assert EdgeCorrector.output_unit(EDGE_CORRECTION.NEW_VAL, on="Size_Area") == ResolvedUnit("px²", 2)


def test_unknown_column_is_refused_by_name():
    with pytest.raises(ValueError, match="my_custom_col"):
        LinearLagModel.output_unit(LINEAR_LAG_MODEL.v, on="my_custom_col")


@pytest.mark.parametrize(
    "on",
    [
        pytest.param("NeighborDist_NearestObjLabel", id="not-a-quantity"),
        pytest.param("Metadata_Strain", id="metadata"),
        pytest.param("LinearLagModel_RobustMeanRadius_v", id="itself-input-dependent"),
        # K0-17: edge-corrected headers are not resolved recursively.
        pytest.param("EdgeCorrection_NewVal-Size_Area", id="edge-corrected"),
    ],
)
def test_a_column_without_a_unit_of_its_own_is_refused_by_name(on):
    with pytest.raises(ValueError, match=on):
        LinearLagModel.output_unit(LINEAR_LAG_MODEL.v, on=on)


def test_input_dependent_unit_needs_on():
    with pytest.raises(ValueError, match="on="):
        LinearLagModel.output_unit(LINEAR_LAG_MODEL.v)


def test_a_member_without_a_unit_is_refused_by_name():
    with pytest.raises(ValueError, match="GENETIC.STRAIN"):
        MeasureSize.output_unit(GENETIC.STRAIN)


def test_unit_of_column():
    assert unit_of_column("Size_Area") == ResolvedUnit("px²", 2)
    assert unit_of_column(str(NEIGHBOR_DIST.NEAREST_OBJ_LABEL)) == ResolvedUnit("", None)
    assert unit_of_column("Texture_Contrast-deg000-scale05") == ResolvedUnit("", 0)
    assert unit_of_column("my_custom_col") is None
    assert unit_of_column("Metadata_Strain") is None
    assert unit_of_column("LinearLagModel_Area_v") is None
    # A statically tagged fit parameter keeps its unit, whatever was fitted.
    assert unit_of_column("LinearLagModel_Area_lambda") == ResolvedUnit("time", 0)
    assert unit_of_column("LinearLagModel_Area_alpha") == ResolvedUnit("1/time", 0)
    assert unit_of_column("EdgeCorrection_NewVal-Size_Area") is None


def test_every_producer_resolves_every_member():
    for prod in measurement_producers():
        for info in prod.primary_infos + prod.shared_infos:
            for member in info:
                on = "Size_RobustMeanRadius"
                assert isinstance(prod.producer.output_unit(member, on), ResolvedUnit), (prod.output_key, str(member))


def test_the_registry_delegates_to_the_producer():
    size = next(p for p in measurement_producers() if p.output_key == "MeasureSize")
    assert size.output_unit(SIZE.AREA) == MeasureSize.output_unit(SIZE.AREA)
    fit = next(p for p in measurement_producers() if p.output_key == "LinearLagModel")
    assert fit.output_unit(LINEAR_LAG_MODEL.v, "Size_Area") == ResolvedUnit("px²/time", 2)


def test_no_producer_overrides_output_unit():
    # One default covers every producer (K0-04); an override would be a second
    # resolver that the docs and the tests above would not see.
    overriding = [
        f"{klass.__module__}.{klass.__qualname__}"
        for prod in measurement_producers()
        for klass in prod.producer.__mro__
        if "output_unit" in vars(klass) and klass not in (MeasureFeatures, SetAnalyzer)
    ]
    assert overriding == []
