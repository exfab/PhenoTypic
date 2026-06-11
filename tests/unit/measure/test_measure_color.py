import numpy as np
import pytest

from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureColor
from phenotypic.schema import OBJECT, ColorLab, ColorHSV


@pytest.fixture(scope="module")
def detected_image():
    img = load_synth_yeast_plate()
    return OtsuDetector().apply(img)


def test_default_output_is_robust_only(detected_image):
    df = MeasureColor().measure(detected_image)
    cols = set(df.columns)
    # robust Lab + HSV present
    assert set(ColorLab.robust_headers()).issubset(cols)
    assert set(ColorHSV.robust_headers()).issubset(cols)
    # XYZ/xy absent by default
    assert not any(c.startswith("ColorXYZ_") for c in cols)
    assert not any(c.startswith("Colorxy_") for c in cols)
    # one row per object
    assert len(df) == detected_image.num_objects


def test_hex_column_is_string(detected_image):
    df = MeasureColor().measure(detected_image)
    hexcol = df[str(ColorLab.MEDOID_COLOR_HEX)]
    assert hexcol.dtype == object
    assert hexcol.iloc[0].startswith("#") and len(hexcol.iloc[0]) == 7


def test_deltae_scalars_nonnegative(detected_image):
    df = MeasureColor().measure(detected_image)
    for col in [ColorLab.DELTA_E2000_MEDIAN, ColorLab.DELTA_E2000_MEAN, ColorLab.DELTA_E2000_P95]:
        vals = df[str(col)].to_numpy()
        assert np.all(vals[~np.isnan(vals)] >= 0)


def test_opt_in_xyz_and_xy(detected_image):
    df = MeasureColor(include_XYZ=True, include_xy=True).measure(detected_image)
    assert any(c.startswith("ColorXYZ_") for c in df.columns)
    assert any(c.startswith("Colorxy_") for c in df.columns)


def test_serialization_roundtrip(detected_image):
    op = MeasureColor(medoid_max_pixels=300, random_seed=3)
    restored = MeasureColor.from_json(op.to_json())
    assert restored.medoid_max_pixels == 300
    assert restored.random_seed == 3


def test_hex_column_survives_numeric_aggregation(detected_image):
    df = MeasureColor().measure(detected_image)
    # Simulate the master-aggregation numeric reduction: must not raise on the
    # string hex column and must skip it.
    numeric_means = df.drop(columns=[OBJECT.LABEL]).mean(numeric_only=True)
    assert str(ColorLab.MEDOID_COLOR_HEX) not in numeric_means.index
    # group-mean (replicate aggregation shape) also tolerates the string column
    grouped = df.groupby(OBJECT.LABEL).mean(numeric_only=True)
    assert str(ColorLab.MEDOID_COLOR_HEX) not in grouped.columns
