"""Dynamic output-header emission and scheme-aware recognition."""

import pytest

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


def test_texture_get_headers_emits_scale_direction_feature_order():
    """Exact spelling and order: feature-outer x angle-inner (52), then 13 averages.

    ``MeasureTexture._compute_haralick`` fills values by *position*, so this
    order is a contract, not a presentation detail.
    """
    headers = TEXTURE.get_headers(scale=5, matrix_name="Gray")
    labels = TEXTURE.get_labels()
    assert TEXTURE.header_scheme() == "texture"
    assert len(headers) == 65
    assert headers[0] == "Texture_05px-deg000-AngularSecondMoment"
    assert headers[1] == "Texture_05px-deg045-AngularSecondMoment"
    assert headers[3] == "Texture_05px-deg135-AngularSecondMoment"
    assert headers[4] == f"Texture_05px-deg000-{labels[1]}"
    assert headers[52] == "Texture_05px-avg-AngularSecondMoment"
    expected = [
        f"Texture_05px-deg{angle:03d}-{label}"
        for label in labels
        for angle in (0, 45, 90, 135)
    ] + [f"Texture_05px-avg-{label}" for label in labels]
    assert headers == expected


def test_texture_scale_zero_padding_is_a_minimum_width():
    assert TEXTURE.get_headers(scale=1)[0] == "Texture_01px-deg000-AngularSecondMoment"
    assert TEXTURE.get_headers(scale=10)[-1] == "Texture_10px-avg-InfoCorrelation2"
    assert TEXTURE.get_headers(scale=100)[0] == "Texture_100px-deg000-AngularSecondMoment"


def test_texture_headers_round_trip_to_their_member():
    """Every emitted header resolves to the member whose label ends it.

    Scales >= 100 emit three digits (``{scale:02d}`` is a minimum width) and
    must still be recognized.
    """
    for scale in (1, 5, 10, 100, 250):
        for header in TEXTURE.get_headers(scale=scale, matrix_name="Gray"):
            member = TEXTURE.member_for_header(header)
            assert member is not None, header
            assert header.endswith("-" + member.label), (header, member)
            assert TEXTURE.owns_header(header), header


def test_texture_legacy_spelling_is_still_recognized():
    """Stored tables written before the rename keep their texture ownership.

    Without this, legacy columns fall through to the metadata classifiers and
    get renamed ``Metadata_Texture_...``.
    """
    for header in (
        "Texture_Contrast-deg000-scale05",
        "Texture_Contrast-deg135-scale05",
        "Texture_Contrast-avg-scale05",
        "Texture_Contrast-avg-scale100",
    ):
        assert TEXTURE.member_for_header(header) is TEXTURE.CONTRAST, header
        assert TEXTURE.owns_header(header), header


@pytest.mark.parametrize(
    "header",
    [
        "Texture_Contrast",  # bare base label
        "Texture_05px-Contrast",  # no direction token
        "Texture_05px-deg000",  # no feature label
        "Texture_05px-avg-Nope",  # unknown feature
        "Shape_05px-avg-Contrast",  # wrong category
        "TextureGray_Contrast-deg000-scale05",  # older legacy prefix
        "Texture_05-deg000-Contrast",  # missing px
        "Texture_5px-deg0-Contrast",  # unpadded scale and angle
        "Texture_05px-deg45-Contrast",  # unpadded angle
        "Texture_005px-deg000-Contrast",  # over-padded scale
        "Texture_00px-avg-Contrast",  # scale 0
        "Texture_05px-deg030-Contrast",  # non-emitted angle
        "Texture_05px-deg0000-Contrast",  # over-padded angle
        # `$` matches before a trailing newline; `fullmatch` must not.
        "Texture_05px-avg-Contrast\n",  # current spelling + newline
        "Texture_05px-deg045-Contrast\n",
        "Texture_Contrast-avg-scale05\n",  # legacy spelling + newline
        "Texture_Contrast-deg000-scale05\n",
    ],
)
def test_texture_rejects_non_canonical_headers(header):
    assert TEXTURE.member_for_header(header) is None
    assert not TEXTURE.owns_header(header)


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
