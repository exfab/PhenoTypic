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


@pytest.mark.parametrize("bad", [True, False, [True], (False,)])
def test_bool_scale_is_refused(bad):
    """Pydantic's lax mode would read ``True`` as distance 1 (review m1)."""
    with pytest.raises(pydantic.ValidationError, match="scale must be an integer"):
        MeasureTexture(scale=bad)


def test_bool_scale_is_refused_on_assignment_too():
    measurer = MeasureTexture()
    with pytest.raises(pydantic.ValidationError, match="scale must be an integer"):
        measurer.scale = True
    assert measurer.scale == 5


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


# ---------------------------------------------------------------------------
# scale reaches the GLCM distance (review I2)
# ---------------------------------------------------------------------------


def _haralick_by_hand(image, label_index: int, distance: int, quant_lvl: int):
    """Recompute one object's (4, 13) Haralick matrix straight from mahotas.

    Mirrors ``_compute_haralick``'s documented preprocessing independently:
    the object's bounding-box crop of ``gray.foreground()``, other labels
    zeroed, ``floor(x * quant_lvl)`` clipped to ``quant_lvl - 1`` as ``uint8``,
    no ``enhance`` rescale, then ``haralick(ignore_zeros=True)`` at *distance*.
    """
    import mahotas

    label = image.objects.labels[label_index]
    crop = image.objects.props[label_index].slice
    obj = image.gray.foreground()[crop].copy()
    obj[image.objmap[:][crop] != label] = 0
    quantized = np.clip(np.floor(obj * quant_lvl), 0, quant_lvl - 1).astype(np.uint8)
    return mahotas.features.haralick(
        quantized, distance=distance, ignore_zeros=True, return_mean=False
    )


@pytest.mark.parametrize("scale", [1, 2, 5])
def test_scale_is_the_glcm_distance(synth_plate_detected, scale):
    """Each ``Texture_{NN}px-*`` column holds the Haralick value AT distance NN.

    Without this, hard-coding ``distance=5`` in ``_compute_haralick`` left
    every test green (review I2) while mislabelling the ``02px`` columns with
    distance-5 values -- the exact error the scale-in-the-name scheme exists to
    prevent. Tolerance: the reference performs the same float operations on
    the same input in the same library, so the values are bit-identical; the
    ``rtol=1e-12`` only admits reassociation noise, orders of magnitude below
    the distance-1-vs-5 gap the control below asserts.
    """
    image = synth_plate_detected.copy()
    table = MeasureTexture(scale=scale).measure(image)
    directional = TEXTURE.get_headers(scale)[:-13]
    finite = np.flatnonzero(np.isfinite(table[directional].to_numpy()).all(axis=1))
    # A handful of real colonies is enough to pin the mapping; the check does
    # no work if every object failed Haralick.
    assert finite.size >= 3
    for idx in finite[:5]:
        expected = _haralick_by_hand(image, int(idx), scale, quant_lvl=32).T.ravel()
        np.testing.assert_allclose(
            table.iloc[idx][directional].to_numpy(dtype=np.float64),
            expected,
            rtol=1e-12,
            atol=0.0,
            err_msg=f"object index {idx} at scale {scale}",
        )


def test_different_scales_give_different_values(synth_plate_detected):
    """Control: the per-scale reference above only proves something if the
    distance actually moves the features on this plate."""
    fine = MeasureTexture(scale=1).measure(synth_plate_detected.copy())
    coarse = MeasureTexture(scale=5).measure(synth_plate_detected.copy())
    fine_contrast = fine["Texture_01px-avg-Contrast"].to_numpy()
    coarse_contrast = coarse["Texture_05px-avg-Contrast"].to_numpy()
    both = np.isfinite(fine_contrast) & np.isfinite(coarse_contrast)
    assert both.sum() >= 3
    assert not np.allclose(fine_contrast[both], coarse_contrast[both])
