"""Which columns a pipeline will produce, without running it.

A blanket ``get_headers()`` is wrong for two of the three header schemes,
and for one of them it raises. The dispatch exists because ``TEXTURE``
emits one column per (member x angle x scale) — 130 for two scales, not
the 13 base labels — and because a metric-qualified enum has no headers at
all until a runtime metric token is known.
"""

from __future__ import annotations

import pytest

from phenotypic._services.catalog import derive_columns, measurement_headers


def test_texture_headers_expand_per_scale():
    from phenotypic import ImagePipeline
    from phenotypic.measure import MeasureTexture

    pipe = ImagePipeline(meas=[MeasureTexture(scale=[5, 10])])
    cols = derive_columns(pipe)

    assert len(cols) == 130, f"expected 130 expanded texture columns, got {len(cols)}"
    assert "Texture_AngularSecondMoment-deg000-scale05" in cols
    assert any("scale10" in c for c in cols)


def test_texture_columns_track_the_instance_scale_list():
    """One scale is 65 columns; the count must follow the live instance."""
    from phenotypic import ImagePipeline
    from phenotypic.measure import MeasureTexture

    one = derive_columns(ImagePipeline(meas=[MeasureTexture(scale=[5])]))
    assert len(one) == 65
    assert not any("scale10" in c for c in one)


def test_static_scheme_still_works():
    from phenotypic import ImagePipeline
    from phenotypic.measure import MeasureSize

    assert "Size_Area" in derive_columns(ImagePipeline(meas=[MeasureSize()]))


def test_blanket_get_headers_would_have_raised():
    """Pin the reason this dispatch exists, so nobody 'simplifies' it back."""
    from phenotypic.schema import TEXTURE

    with pytest.raises(TypeError, match="scale"):
        TEXTURE.get_headers()


def test_color_columns_follow_the_instance_not_the_class():
    from phenotypic import ImagePipeline
    from phenotypic.measure import MeasureColor

    with_xyz = derive_columns(ImagePipeline(meas=[MeasureColor(include_XYZ=True)]))
    without = derive_columns(ImagePipeline(meas=[MeasureColor(include_XYZ=False)]))
    assert len(with_xyz) > len(without)
    assert set(without) < set(with_xyz)


def test_columns_are_ordered_and_deduplicated():
    from phenotypic import ImagePipeline
    from phenotypic.measure import MeasureShape, MeasureSize

    cols = derive_columns(ImagePipeline(meas=[MeasureSize(), MeasureShape(), MeasureSize()]))
    assert len(cols) == len(set(cols)), "a repeated measurer must not duplicate columns"
    assert cols.index("Size_Area") < cols.index("Shape_Circularity")


def test_an_empty_pipeline_produces_nothing():
    from phenotypic import ImagePipeline

    assert derive_columns(ImagePipeline()) == []


def test_metric_qualified_scheme_uses_the_runtime_metric_token():
    """``LOG_GROWTH_MODEL`` has no headers until ``on`` names a metric."""
    from phenotypic.analysis import LogGrowthModel
    from phenotypic.schema import MODEL_METRICS

    model = LogGrowthModel(on="Size_Area", groupby=["Metadata_Well"])
    headers = measurement_headers(MODEL_METRICS, model)

    assert "ModelMetrics_Area_MAE" in headers
    assert "ModelMetrics_MAE" not in headers, "that is the static spelling"


def test_metric_qualified_without_a_token_yields_nothing():
    """No runtime metric means no derivable header — not a fabricated one."""
    from phenotypic.schema import MODEL_METRICS
    from phenotypic.measure import MeasureSize

    assert measurement_headers(MODEL_METRICS, MeasureSize()) == []


def test_texture_scheme_reaches_get_headers_with_the_scale():
    """Direct check that the texture branch passes ``scale`` through."""
    from phenotypic.measure import MeasureTexture
    from phenotypic.schema import TEXTURE

    headers = measurement_headers(TEXTURE, MeasureTexture(scale=[5, 10]))
    assert headers[:len(TEXTURE.get_headers(5))] == TEXTURE.get_headers(5)
    assert TEXTURE.get_headers(10)[0] in headers


def test_derive_columns_matches_a_real_measure_run():
    """The end-to-end anchor: the derived list is what ``measure()`` emits."""
    from phenotypic import ImagePipeline
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureShape, MeasureSize, MeasureTexture

    pipe = ImagePipeline(
        pipe_cfgs=[OtsuDetector()],
        meas=[MeasureSize(), MeasureShape(), MeasureTexture(scale=[5])],
    )
    measured = pipe.measure(load_synth_yeast_plate())
    derived = derive_columns(pipe)

    assert len(derived) == 65 + len(derive_columns(ImagePipeline(
        meas=[MeasureSize(), MeasureShape()]
    ))), "the anchor must not pass vacuously on an empty derivation"
    missing = [c for c in derived if c not in measured.columns]
    assert not missing, f"derived columns absent from the real output: {missing[:10]}"
    assert "Texture_AngularSecondMoment-deg000-scale05" in measured.columns
