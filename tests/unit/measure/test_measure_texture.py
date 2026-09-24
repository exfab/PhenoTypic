"""MeasureTexture: one GLCM scale per measurer, emitted as ``Texture_{scale:02d}px-...``."""

from __future__ import annotations

import json

import numpy as np
import pydantic
import pytest

from phenotypic import ImagePipeline
from phenotypic.measure import MeasureTexture
from phenotypic.schema import OBJECT, TEXTURE

_ONE_SCALE_MESSAGE = "MeasureTexture measures one scale; add one MeasureTexture per scale"


# ---------------------------------------------------------------------------
# scale field: a single int
# ---------------------------------------------------------------------------


def test_scale_is_a_single_int():
    measurer = MeasureTexture(scale=5)
    assert measurer.scale == 5
    assert type(measurer.scale) is int


def test_default_scale_is_five():
    assert MeasureTexture().scale == 5


@pytest.mark.parametrize("legacy", [[5], (5,)])
def test_one_element_sequence_is_coerced_to_its_element(legacy):
    """Every pre-rename ``pipeline.json`` stores ``"scale": [5]``."""
    measurer = MeasureTexture(scale=legacy)
    assert measurer.scale == 5
    assert type(measurer.scale) is int


def test_multi_element_scale_list_is_refused_with_guidance():
    with pytest.raises(pydantic.ValidationError, match=_ONE_SCALE_MESSAGE):
        MeasureTexture(scale=[3, 4])


def test_empty_scale_list_is_refused_with_guidance():
    with pytest.raises(pydantic.ValidationError, match=_ONE_SCALE_MESSAGE):
        MeasureTexture(scale=[])


@pytest.mark.parametrize("bad", [0, -1, [0]])
def test_non_positive_scale_is_refused(bad):
    with pytest.raises(pydantic.ValidationError):
        MeasureTexture(scale=bad)


def test_assignment_is_validated_too():
    measurer = MeasureTexture()
    measurer.scale = [7]
    assert measurer.scale == 7
    with pytest.raises(pydantic.ValidationError, match=_ONE_SCALE_MESSAGE):
        measurer.scale = [3, 4]


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def test_pipeline_json_writes_scale_as_an_int():
    config = json.loads(ImagePipeline(meas=[MeasureTexture(scale=5)]).to_json())
    assert config["meas"]["MeasureTexture"]["params"]["scale"] == 5


def _legacy_pipeline_json(scale) -> str:
    """A pipeline JSON as written before ``scale`` became an int."""
    return json.dumps(
        {
            "pipe_cfgs": {},
            "meas": {
                "MeasureTexture": {
                    "class": "MeasureTexture",
                    "params": {
                        "scale": scale,
                        "quant_lvl": 32,
                        "enhance": False,
                        "warn": False,
                    },
                }
            },
        }
    )


def test_legacy_pipeline_json_with_one_scale_loads():
    loaded = ImagePipeline.from_json(_legacy_pipeline_json([5]))
    assert loaded._meas["MeasureTexture"].scale == 5


def test_legacy_pipeline_json_with_several_scales_is_refused():
    """The documented D7 cost: the error users see in recompile / QC rebuild."""
    with pytest.raises(pydantic.ValidationError, match=_ONE_SCALE_MESSAGE):
        ImagePipeline.from_json(_legacy_pipeline_json([3, 4]))


# ---------------------------------------------------------------------------
# measure() output
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def texture_table(synth_plate_detected):
    return MeasureTexture(scale=5).measure(synth_plate_detected.copy())


def test_measure_emits_exactly_one_scale_of_columns(texture_table):
    expected = [str(OBJECT.LABEL), *TEXTURE.get_headers(5)]
    assert len(expected) == 66
    assert list(texture_table.columns) == expected
    assert len(texture_table) > 0


def test_every_emitted_texture_column_is_owned_by_texture(texture_table):
    for column in texture_table.columns:
        if column == str(OBJECT.LABEL):
            continue
        assert TEXTURE.owns_header(column), column


def test_average_column_is_the_mean_of_its_four_directions(texture_table):
    """Order guard: names and values line up.

    ``_compute_haralick`` fills values by position (feature-major ravel, then
    4-wide averaging strides), so if ``get_headers`` ever emits a different
    order the values land under the wrong names with no error. Averages are
    compared by *name* here, so a transposed header order goes red.
    Objects whose Haralick computation fails are all-NaN, hence ``equal_nan``.
    """
    finite_rows = 0
    for label in TEXTURE.get_labels():
        directional = texture_table[
            [f"Texture_05px-deg{angle:03d}-{label}" for angle in (0, 45, 90, 135)]
        ].to_numpy()
        average = texture_table[f"Texture_05px-avg-{label}"].to_numpy()
        np.testing.assert_allclose(
            average,
            directional.mean(axis=1),
            rtol=1e-12,
            atol=0.0,
            equal_nan=True,
            err_msg=label,
        )
        finite_rows = max(finite_rows, int(np.isfinite(average).sum()))
    # The guard does no work if every object failed Haralick.
    assert finite_rows > 0
