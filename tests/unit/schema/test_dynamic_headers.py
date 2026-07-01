"""Dynamic output-header emission and scheme-aware recognition."""

import phenotypic.schema as schema
from phenotypic.schema import (
    LINEAR_CAP_AND_LAG_MODEL,
    LINEAR_LAG_MODEL,
    LOG_GROWTH_MODEL,
    MODEL_METRICS,
    MeasurementInfo,
    SHAPE,
    TEXTURE,
    parse_qualified_header,
    qualified_header,
)


def test_qualified_header_format():
    assert qualified_header(LINEAR_LAG_MODEL.v, "Area") == "LinearLagModel_Area_v"
    assert qualified_header(MODEL_METRICS.RMSE, "Radius") == "ModelMetrics_Radius_RMSE"


def test_qualified_roundtrip_including_underscored_metric():
    for token in ("Area", "Radius", "x", "my_custom"):
        for member in list(LINEAR_LAG_MODEL) + list(MODEL_METRICS):
            header = qualified_header(member, token)
            assert parse_qualified_header(type(member), header) == (token, member)


def test_metric_qualified_scheme_recognition():
    header = qualified_header(LINEAR_LAG_MODEL.s0, "Area")  # LinearLagModel_Area_s0
    assert LINEAR_LAG_MODEL.header_scheme() == "metric_qualified"
    assert LINEAR_LAG_MODEL.owns_header(header)
    assert LINEAR_LAG_MODEL.member_for_header(header) is LINEAR_LAG_MODEL.s0
    # legacy unqualified is NOT recognized (graceful degrade, hard cutover)
    assert not LINEAR_LAG_MODEL.owns_header("LinearLagModel_s0")
    assert LINEAR_LAG_MODEL.member_for_header("LinearLagModel_s0") is None


def test_static_scheme_is_default():
    assert SHAPE.header_scheme() == "static"
    assert SHAPE.owns_header("Shape_Area")
    assert SHAPE.member_for_header("Shape_Area") is SHAPE.AREA
    assert not SHAPE.owns_header("Shape_Area_extra")


def test_texture_scheme_recognition():
    headers = TEXTURE.get_headers(scale=5, matrix_name="Gray")
    directional = headers[0]  # e.g. Texture_AngularSecondMoment-deg000-scale05
    assert TEXTURE.header_scheme() == "texture"
    assert TEXTURE.owns_header(directional)
    member = TEXTURE.member_for_header(directional)
    assert member is not None and member.label in directional
    avg = next(h for h in headers if "-avg-scale" in h)
    assert TEXTURE.owns_header(avg)
    # a bare base label is not an emitted texture header
    assert not TEXTURE.owns_header("Texture_AngularSecondMoment")


def test_no_label_is_underscore_suffix_of_another_label():
    """Guardrail: protects parse_qualified_header's suffix anchoring."""
    for name in schema.__all__:
        obj = getattr(schema, name, None)
        if not (
            isinstance(obj, type)
            and issubclass(obj, MeasurementInfo)
            and obj is not MeasurementInfo
            and list(obj)
        ):
            continue
        labels = [m.label for m in obj]
        for a in labels:
            for b in labels:
                if a is not b:
                    assert not a.endswith("_" + b), (obj.__name__, a, b)


def test_double_softplus_and_log_growth_own_qualified_headers():
    for member in list(LINEAR_CAP_AND_LAG_MODEL):
        header = qualified_header(member, "Area")
        assert LINEAR_CAP_AND_LAG_MODEL.owns_header(header)
    for member in list(LOG_GROWTH_MODEL):
        header = qualified_header(member, "Area")
        assert LOG_GROWTH_MODEL.owns_header(header)
