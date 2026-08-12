"""README Models & Analysis section renders metric-qualified headers."""

from types import SimpleNamespace
from typing import ClassVar

from phenotypic import ImagePipeline
from phenotypic.analysis import LinearLagModel
from phenotypic.abc_ import MeasureFeatures
from phenotypic._cli._cli_readme_generator import READMEGenerator
from phenotypic.measure import (
    MeasureColor,
    MeasureOrientationZones,
    MeasureSymZones,
)
from phenotypic.schema import (
    LINEAR_LAG_MODEL,
    MODEL_METRICS,
    ORIENTATION_ZONE_DIAGNOSTIC,
    ORIENTATION_ZONE_PRIMARY,
    ColorHSV,
    ColorLab,
    ColorXYZ,
    Colorxy,
    Entry,
    MeasurementInfo,
    SYMMETRIC_ZONES,
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


def test_color_readme_schema_follows_optional_output_flags():
    """README column docs should match the enabled color output schemas."""
    generator = READMEGenerator(
        config=SimpleNamespace(), pipeline=ImagePipeline()
    )

    assert generator._get_measurement_infoclasses(MeasureColor()) == [
        ColorLab,
        ColorHSV,
    ]
    assert generator._get_measurement_infoclasses(
        MeasureColor(include_XYZ=True, include_xy=True)
    ) == [ColorXYZ, Colorxy, ColorLab, ColorHSV]


def test_dynamic_readme_schema_supports_pydantic_private_declarations():
    """Legacy custom measurers need not annotate schema declarations."""

    class MeasureLegacySingle(MeasureFeatures):
        _measurement_infoclass = SYMMETRIC_ZONES

        def _operate(self, image):
            raise NotImplementedError

    class MeasureLegacyPlural(MeasureFeatures):
        _measurement_infoclasses = [ColorLab, ColorHSV]

        def _operate(self, image):
            raise NotImplementedError

    generator = READMEGenerator(
        config=SimpleNamespace(), pipeline=ImagePipeline()
    )

    assert generator._get_measurement_infoclasses(MeasureLegacySingle()) == [
        SYMMETRIC_ZONES
    ]
    assert generator._get_measurement_infoclasses(MeasureLegacyPlural()) == [
        ColorLab,
        ColorHSV,
    ]


def test_dynamic_readme_schema_preserves_default_named_member():
    """A schema member named ``default`` must not look like a private attr."""

    class CUSTOM_DEFAULT(MeasurementInfo):
        @classmethod
        def category(cls) -> str:
            return "CustomDefault"

        default = Entry("Default", "A valid lowercase schema member.")

    class MeasureCustomDefault(MeasureFeatures):
        _measurement_infoclass: ClassVar[type] = CUSTOM_DEFAULT

        def _operate(self, image):
            raise NotImplementedError

    generator = READMEGenerator(
        config=SimpleNamespace(), pipeline=ImagePipeline()
    )

    assert generator._get_measurement_infoclasses(MeasureCustomDefault()) == [
        CUSTOM_DEFAULT
    ]


def test_color_readme_schema_preserves_subclass_extensions():
    """Color flag filtering should not discard third-party schemas."""

    class MeasureExtendedColor(MeasureColor):
        _measurement_infoclasses: ClassVar[list[type]] = [
            ColorXYZ,
            Colorxy,
            ColorLab,
            ColorHSV,
            SYMMETRIC_ZONES,
        ]

    generator = READMEGenerator(
        config=SimpleNamespace(), pipeline=ImagePipeline()
    )

    assert generator._get_measurement_infoclasses(MeasureExtendedColor()) == [
        ColorLab,
        ColorHSV,
        SYMMETRIC_ZONES,
    ]


def test_symmetric_zones_readme_uses_measurement_schema():
    """README column docs should include the symmetric-zone output schema."""
    pipe = ImagePipeline()
    pipe.set_meas([MeasureSymZones()])
    generator = READMEGenerator(
        config=SimpleNamespace(image_type="Image"), pipeline=pipe
    )

    assert generator._get_measurement_infoclasses(MeasureSymZones()) == [
        SYMMETRIC_ZONES
    ]

    section = generator._generate_measurements_section()

    assert "### SymZones" in section
    assert f"`{SYMMETRIC_ZONES.CORE_RADIUS}`" in section
    assert f"`{SYMMETRIC_ZONES.SPARSE_AREA}`" in section
    assert "*No measurement documentation available.*" not in section
