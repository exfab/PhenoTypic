"""Unit tests for MeasureTexture's multi-scale output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from phenotypic import Image
from phenotypic.measure import MeasureTexture
from phenotypic.schema import OBJECT, TEXTURE


@pytest.fixture
def textured_pair_image() -> Image:
    """Two 30x30 colonies with seeded random surface texture, large enough that
    a scale-10 co-occurrence offset stays inside each object."""
    rng = np.random.default_rng(0)
    objmap = np.zeros((70, 70), dtype=int)
    objmap[5:35, 5:35] = 1
    objmap[35:65, 35:65] = 2
    rgb = np.zeros((*objmap.shape, 3), dtype=np.uint8)
    rgb[objmap > 0] = rng.integers(40, 255, size=(int((objmap > 0).sum()), 1))
    image = Image(rgb)
    image.objmap[:] = objmap
    return image


def test_every_scale_writes_its_own_column_set(textured_pair_image):
    """Mutation: drop the assignment of the per-scale merge -> the scale10 set is
    missing."""
    frame = MeasureTexture(scale=[5, 10]).measure(textured_pair_image)

    scale05 = TEXTURE.get_headers(5, "Gray")
    scale10 = TEXTURE.get_headers(10, "Gray")
    assert len(scale05) == len(scale10) == 65
    assert list(frame.columns) == [str(OBJECT.LABEL), *scale05, *scale10]
    assert len(frame) == 2


def test_multi_scale_values_match_single_scale_runs(textured_pair_image):
    """Each scale's block, rows matched by Object_Label, equals that scale
    measured on its own."""
    frame = MeasureTexture(scale=[5, 10]).measure(textured_pair_image)

    for scale in (5, 10):
        single = MeasureTexture(scale=scale).measure(textured_pair_image)
        headers = [str(OBJECT.LABEL), *TEXTURE.get_headers(scale, "Gray")]
        assert not single[headers[1:]].isna().all().any()
        pd.testing.assert_frame_equal(
            frame[headers].reset_index(drop=True),
            single[headers].reset_index(drop=True),
        )


def test_scale10_columns_are_distance_10_haralick(textured_pair_image):
    """Mutation: hard-code ``distance=1`` in ``_compute_haralick`` -> the scale10
    block no longer matches a direct distance-10 mahotas call."""
    import mahotas

    frame = MeasureTexture(scale=[5, 10]).measure(textured_pair_image)
    # Object 1's crop is exactly [5:35, 5:35] with no neighbour pixels, so the
    # oracle needs no masking. quant_lvl=32 is the MeasureTexture default.
    fg = textured_pair_image.gray.foreground()[5:35, 5:35]
    quantized = np.clip(np.floor(fg * 32), 0, 31).astype(np.uint8)
    expected = mahotas.features.haralick(
        quantized, distance=10, ignore_zeros=True, return_mean=False
    ).T.ravel()
    got = frame.loc[
        frame[OBJECT.LABEL] == 1, TEXTURE.get_headers(10, "Gray")[:52]
    ].to_numpy().ravel()
    # Same numpy/mahotas operations in the same order on the same input.
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("scale", [[5, 5], [5, 10, 5], []])
def test_repeated_or_empty_scales_are_rejected(scale):
    """A repeated scale would write suffixed ``_x``/``_y`` columns no schema
    header recognizes; an empty list has no scale to measure."""
    with pytest.raises(ValidationError, match="scale"):
        MeasureTexture(scale=scale)
