"""README Models & Analysis section renders metric-qualified headers."""

from types import SimpleNamespace

from phenotypic import ImagePipeline
from phenotypic.analysis import LinearLagModel
from phenotypic._cli._cli_readme_generator import READMEGenerator
from phenotypic.measure import MeasureOrientationZones
from phenotypic.schema import (
    LINEAR_LAG_MODEL,
    MODEL_METRICS,
    ORIENTATION_ZONE_DIAGNOSTIC,
    ORIENTATION_ZONE_PRIMARY,
    qualified_header,
)


def _generator_with_model() -> READMEGenerator:
    pipe = ImagePipeline()
    pipe.set_model(
        LinearLagModel(on="Shape_Area", groupby=["MetadataGenetic_Strain"])
    )
    return READMEGenerator(config=SimpleNamespace(), pipeline=pipe)


def test_model_section_documents_qualified_headers():
    section = _generator_with_model()._generate_model_section()
    assert "## Models & Analysis" in section
    assert "LinearLagModel" in section
    assert "Shape_Area" in section
    assert f"`{qualified_header(LINEAR_LAG_MODEL.v, 'Area')}`" in section
    assert f"`{qualified_header(MODEL_METRICS.RMSE, 'Area')}`" in section


def test_model_section_is_empty_without_a_model():
    pipe = ImagePipeline()
    gen = READMEGenerator(config=SimpleNamespace(), pipeline=pipe)
    assert gen._generate_model_section() == ""


def test_orientation_zone_readme_schema_follows_diagnostic_flag():
    """README column docs should match the measurer's emitted schema."""
    pipe = ImagePipeline()
    generator = READMEGenerator(config=SimpleNamespace(), pipeline=pipe)

    assert generator._get_measurement_infoclasses(
        MeasureOrientationZones()
    ) == [ORIENTATION_ZONE_PRIMARY]
    assert generator._get_measurement_infoclasses(
        MeasureOrientationZones(include_diagnostics=True)
    ) == [ORIENTATION_ZONE_PRIMARY, ORIENTATION_ZONE_DIAGNOSTIC]
