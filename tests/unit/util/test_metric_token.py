"""Metric-token derivation from a fitter's `on` column."""

import phenotypic.schema as schema
from phenotypic.schema import MeasurementInfo
from phenotypic.util._measurement_outputs import metric_token


def test_strips_known_category():
    assert metric_token("Shape_Area") == "Area"
    assert metric_token("Size_IntegratedIntensity") == "IntegratedIntensity"


def test_endorsed_examples():
    # category-strip works even though "Radius" is not a SIZE member
    assert metric_token("Size_Radius") == "Radius"
    assert metric_token("x") == "x"            # unknown token → verbatim
    assert metric_token("Area") == "Area"      # bare label → verbatim


def test_longest_prefix_wins_for_qc_family():
    # "QC" and "QC_Tukey" are both real categories; the longest must win
    assert metric_token("QC_Tukey_NumOutliers") == "NumOutliers"


def test_every_category_strips_to_the_remainder():
    for name in schema.__all__:
        obj = getattr(schema, name, None)
        if not (
            isinstance(obj, type)
            and issubclass(obj, MeasurementInfo)
            and obj is not MeasurementInfo
            and list(obj)
        ):
            continue
        cat = obj.category()
        assert metric_token(f"{cat}_Foo") == "Foo", cat


def test_sanitizes_whitespace():
    assert metric_token("  Shape_Area  ") == "Area"
