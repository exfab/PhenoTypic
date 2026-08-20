"""Which columns a pipeline will produce, without running it.

A blanket ``get_headers()`` is wrong for two of the three header schemes,
and for one of them it raises. The dispatch exists because ``TEXTURE``
emits one column per (member x angle x scale) — 130 for two scales, not
the 13 base labels — and because a metric-qualified enum has no headers at
all until a runtime metric token is known.
"""

from __future__ import annotations

import pytest

from phenotypic._services.catalog import (
    _info_block_columns,
    derive_columns,
    measurement_headers,
)


def _measurement_columns(pipe, *, image_type: str = "GridImage") -> list[str]:
    """*pipe*'s derived columns minus the info block every run carries.

    The info block is what an empty pipeline derives, so subtracting it
    leaves exactly what the ``meas`` slot contributed — which is what the
    header-scheme tests below are about.
    """
    from phenotypic import ImagePipeline

    info = set(derive_columns(ImagePipeline(), image_type=image_type))
    return [c for c in derive_columns(pipe, image_type=image_type) if c not in info]


def test_texture_headers_expand_per_scale():
    from phenotypic import ImagePipeline
    from phenotypic.measure import MeasureTexture

    pipe = ImagePipeline(meas=[MeasureTexture(scale=[5, 10])])
    cols = _measurement_columns(pipe)

    assert len(cols) == 130, f"expected 130 expanded texture columns, got {len(cols)}"
    assert "Texture_AngularSecondMoment-deg000-scale05" in cols
    assert any("scale10" in c for c in cols)


def test_texture_columns_track_the_instance_scale_list():
    """One scale is 65 columns; the count must follow the live instance."""
    from phenotypic import ImagePipeline
    from phenotypic.measure import MeasureTexture

    one = _measurement_columns(ImagePipeline(meas=[MeasureTexture(scale=[5])]))
    assert len(one) == 65
    assert not any("scale10" in c for c in one)


def test_static_scheme_still_works():
    from phenotypic import ImagePipeline
    from phenotypic.measure import MeasureSize

    cols = derive_columns(ImagePipeline(meas=[MeasureSize()]), image_type="Image")
    assert "Size_Area" in cols


def test_blanket_get_headers_would_have_raised():
    """Pin the reason this dispatch exists, so nobody 'simplifies' it back."""
    from phenotypic.schema import TEXTURE

    with pytest.raises(TypeError, match="scale"):
        TEXTURE.get_headers()


def test_color_columns_follow_the_instance_not_the_class():
    from phenotypic import ImagePipeline
    from phenotypic.measure import MeasureColor

    with_xyz = _measurement_columns(ImagePipeline(meas=[MeasureColor(include_XYZ=True)]))
    without = _measurement_columns(ImagePipeline(meas=[MeasureColor(include_XYZ=False)]))
    assert len(with_xyz) > len(without)
    assert set(without) < set(with_xyz)


def test_columns_are_ordered_and_deduplicated():
    from phenotypic import ImagePipeline
    from phenotypic.measure import MeasureShape, MeasureSize

    cols = derive_columns(
        ImagePipeline(meas=[MeasureSize(), MeasureShape(), MeasureSize()]),
        image_type="GridImage",
    )
    assert len(cols) == len(set(cols)), "a repeated measurer must not duplicate columns"
    assert cols.index("Size_Area") < cols.index("Shape_Circularity")


def test_an_empty_pipeline_still_produces_the_info_block():
    """``measure()`` appends the info block whether or not a measurer ran."""
    from phenotypic import ImagePipeline

    assert derive_columns(ImagePipeline(), image_type="Image") == _info_block_columns(
        "Image"
    )


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
    """The end-to-end anchor, asserted in **both** directions.

    ``derived <= measured`` alone cannot see under-reporting, which is how
    the info block went missing from the derivation for so long: the
    original assertion passed while 15 of the 38 real columns were absent
    from the answer an agent gets. The converse direction is the half that
    makes this an anchor, so both are asserted here and the anti-vacuity
    guard is computed from the schema enums rather than from
    ``derive_columns`` itself.
    """
    from phenotypic import ImagePipeline
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureShape, MeasureSize, MeasureTexture
    from phenotypic.schema import BBOX, OBJECT, SHAPE, SIZE, TEXTURE
    from phenotypic.sdk_ import metadata_member_for_header

    pipe = ImagePipeline(
        pipe_cfgs=[OtsuDetector()],
        meas=[MeasureSize(), MeasureShape(), MeasureTexture(scale=[5])],
    )
    measured = pipe.measure(load_synth_yeast_plate())
    derived = derive_columns(pipe, image_type="GridImage")

    # Anti-vacuity, computed independently of the function under test: the
    # earlier guard called ``derive_columns`` again and so was blind to the
    # same under-reporting the assertions below now catch.
    independently_expected = (
        set(SIZE.get_headers())
        | set(SHAPE.get_headers())
        | set(TEXTURE.get_headers(5))
        | set(OBJECT.get_headers())
        | set(BBOX.get_headers())
        | {"Grid_RowNum", "Grid_ColNum"}
    )
    assert len(independently_expected) > 80, "guard fixture drifted"
    assert independently_expected <= set(derived)

    real = {str(c) for c in measured.columns}
    # ``Metadata_*`` is the one documented exclusion: the framework block
    # comes off the image and experimental metadata off the run's CSV, so
    # neither is derivable from a pipeline. Ownership is asked of the
    # schema, never of the string prefix.
    derivable = {c for c in real if metadata_member_for_header(c) is None}

    assert set(derived) == derivable, (
        f"derived-not-measured: {sorted(set(derived) - derivable)}; "
        f"measured-not-derived: {sorted(derivable - set(derived))}"
    )
    assert "Texture_AngularSecondMoment-deg000-scale05" in real


def test_the_anchor_also_holds_for_a_plain_image():
    """The ``Grid_*`` half is the only thing the image class changes."""
    from phenotypic import Image, ImagePipeline
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureShape, MeasureSize
    from phenotypic.sdk_ import metadata_member_for_header

    plain = Image(load_synth_yeast_plate().rgb[:])
    pipe = ImagePipeline(pipe_cfgs=[OtsuDetector()], meas=[MeasureSize(), MeasureShape()])
    measured = pipe.apply_and_measure(plain)
    derived = derive_columns(pipe, image_type="Image")

    real = {str(c) for c in measured.columns}
    derivable = {c for c in real if metadata_member_for_header(c) is None}

    assert "Size_Area" in derivable, "anti-vacuity: the run measured nothing"
    assert set(derived) == derivable
    assert not [c for c in derived if c.startswith("Grid_")]


def test_the_info_block_is_appended_after_the_measurements():
    """Mirrors ``measure()``'s own ``[measurements] -> [info block]`` order."""
    from phenotypic import ImagePipeline
    from phenotypic.measure import MeasureSize

    cols = derive_columns(ImagePipeline(meas=[MeasureSize()]), image_type="GridImage")
    assert cols.index("Size_Area") < cols.index("Object_Label")
    assert cols.index("Object_Label") < cols.index("Bbox_CenterRR")
    assert cols.index("Bbox_CenterRR") < cols.index("Grid_RowNum")


def test_grid_columns_appear_only_for_a_gridimage():
    """A ``GridFinder`` refuses a plain ``Image``, so the two must differ."""
    from phenotypic import ImagePipeline
    from phenotypic.measure import MeasureSize

    pipe = ImagePipeline(meas=[MeasureSize()])
    grid = set(derive_columns(pipe, image_type="GridImage"))
    plain = set(derive_columns(pipe, image_type="Image"))

    assert grid - plain == {
        "Grid_RowNum",
        "Grid_ColNum",
        "Grid_RowMajorIdx",
        "Grid_ColMajorIdx",
    }
    assert plain < grid


def test_the_grid_interval_columns_are_not_claimed():
    """``GRID`` declares four interval bounds the info block never emits."""
    from phenotypic.schema import GRID

    derived = set(_info_block_columns("GridImage"))
    declared = set(GRID.get_headers())

    assert declared - derived == {
        "Grid_RowIntervalStart",
        "Grid_RowIntervalEnd",
        "Grid_ColIntervalStart",
        "Grid_ColIntervalEnd",
    }


def test_an_unknown_image_type_is_rejected():
    """A typo must not be read as 'not a grid' and silently drop four columns."""
    from phenotypic import ImagePipeline

    with pytest.raises(ValueError, match="image_type"):
        derive_columns(ImagePipeline(), image_type="gridimage")  # type: ignore[arg-type]
