"""Public-API contract for the phenotypic.schema subpackage."""

import importlib

import pytest


def test_enums_and_base_importable_from_schema():
    from phenotypic.schema import MeasurementInfo, SHAPE

    assert issubclass(SHAPE, MeasurementInfo)
    assert SHAPE.AREA.value == "Shape_Area"
    assert "Shape_Area" in SHAPE.get_headers()
    assert "Area" in SHAPE.get_labels()


def test_schema_exposed_on_top_level_package():
    import phenotypic

    assert phenotypic.schema.SIZE.get_headers()  # non-empty


def test_base_class_reexported_identically_from_abc():
    import phenotypic.abc_ as abc_
    import phenotypic.schema as schema

    assert abc_.MeasurementInfo is schema.MeasurementInfo


def test_old_measurement_info_path_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("phenotypic.sdk_.measurement_info")
