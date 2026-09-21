import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

from phenotypic import Image
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
    op = MeasureColor(medoid_candidates=64)
    restored = MeasureColor.from_json(op.to_json())
    assert restored.medoid_candidates == 64


def test_hex_column_survives_numeric_aggregation(detected_image):
    df = MeasureColor().measure(detected_image)
    # Simulate the master-aggregation numeric reduction: must not raise on the
    # string hex column and must skip it.
    numeric_means = df.drop(columns=[OBJECT.LABEL]).mean(numeric_only=True)
    assert str(ColorLab.MEDOID_COLOR_HEX) not in numeric_means.index
    # group-mean (replicate aggregation shape) also tolerates the string column
    grouped = df.groupby(OBJECT.LABEL).mean(numeric_only=True)
    assert str(ColorLab.MEDOID_COLOR_HEX) not in grouped.columns


_MEDOID_COLUMNS = (ColorLab.L_STAR_MEDOID, ColorLab.A_STAR_MEDOID, ColorLab.B_STAR_MEDOID)


def _exhaustive_de2000_medoid(lab: np.ndarray, chunk: int = 128) -> np.ndarray:
    """Independent reference: the pixel minimising total ΔE2000 to *every* pixel.

    Written out here, O(N^2), on purpose -- calling ``candidate_medoid`` would
    make the test a tautology.
    """
    import colour

    totals = np.empty(lab.shape[0])
    for start in range(0, lab.shape[0], chunk):
        block = lab[start : start + chunk]
        totals[start : start + chunk] = np.asarray(
            colour.difference.delta_E_CIE2000(block[:, None, :], lab[None, :, :])
        ).sum(axis=1)
    return lab[int(totals.argmin())]


@pytest.fixture(scope="module")
def ring_around_foreign_colony():
    """A tan ring (label 1) whose hole holds a blue colony (label 2).

    Label 2 lies entirely inside label 1's bounding box, so a measurement that
    read the bounding box instead of the label mask would mix blue into the
    ring's colour. The ring has ~4400 pixels, over four times the old
    1000-pixel subsample cap, so an estimator that picks its medoid from a
    random subsample does not reproduce the exhaustive medoid.
    """
    rng = np.random.default_rng(20260921)
    size = 96
    rr, cc = np.mgrid[:size, :size]
    radius = np.hypot(rr - size / 2, cc - size / 2)
    ring = (radius >= 14) & (radius < 40)
    hole_colony = radius < 10

    rgb = np.full((size, size, 3), 0.05)
    rgb[ring] = np.clip([0.62, 0.50, 0.30] + rng.normal(0.0, 0.03, (ring.sum(), 3)), 0.0, 1.0)
    rgb[hole_colony] = np.clip(
        [0.10, 0.20, 0.85] + rng.normal(0.0, 0.03, (hole_colony.sum(), 3)), 0.0, 1.0
    )
    objmap = np.zeros((size, size), dtype=np.uint16)
    objmap[ring] = 1
    objmap[hole_colony] = 2

    image = Image(rgb)
    image.objmap[:] = objmap
    return image


def test_medoid_uses_only_the_objects_own_pixels(ring_around_foreign_colony):
    image = ring_around_foreign_colony
    objmap = image.objmap[:]
    ring_rows, ring_cols = np.nonzero(objmap == 1)
    foreign_rows, foreign_cols = np.nonzero(objmap == 2)
    # Precondition: label 2 really sits inside label 1's bounding box.
    assert ring_rows.min() < foreign_rows.min() and foreign_rows.max() < ring_rows.max()
    assert ring_cols.min() < foreign_cols.min() and foreign_cols.max() < ring_cols.max()
    assert (objmap == 1).sum() > 4000

    lab = image.color.Lab[:]
    ring_lab = lab[objmap == 1].astype(np.float64)
    foreign_lab = lab[objmap == 2].astype(np.float64)
    expected = _exhaustive_de2000_medoid(ring_lab)

    df = MeasureColor().measure(image)
    row = df.loc[df[OBJECT.LABEL] == 1].iloc[0]
    measured = np.array([row[str(col)] for col in _MEDOID_COLUMNS], dtype=np.float64)

    np.testing.assert_allclose(measured, expected, rtol=0, atol=1e-9)
    assert not np.any(np.all(np.isclose(foreign_lab, measured, rtol=0, atol=1e-6), axis=1))


def test_medoid_is_deterministic(ring_around_foreign_colony, detected_image):
    """Same pixels, same medoid -- across fresh runs and across pixel order.

    Repeat runs agreeing is necessary but weak: a seeded subsample is repeatable
    too. The stronger property is that the medoid is a function of the object's
    pixel *set*: shuffling where the ring's pixels sit (same colours, same mask)
    must not move it. A subsample drawn by pixel index fails that.
    """
    for image in (ring_around_foreign_colony, detected_image):
        first = MeasureColor().measure(image)
        second = MeasureColor().measure(image)
        for col in (*_MEDOID_COLUMNS, ColorLab.MEDOID_COLOR_HEX):
            assert first[str(col)].tolist() == second[str(col)].tolist()

    original = ring_around_foreign_colony
    ring = original.objmap[:] == 1
    rgb = original.rgb[:].copy()
    ring_rgb = rgb[ring]
    rgb[ring] = ring_rgb[np.random.default_rng(7).permutation(ring_rgb.shape[0])]
    shuffled = Image(rgb)
    shuffled.objmap[:] = original.objmap[:]

    def ring_medoid(image):
        df = MeasureColor().measure(image)
        row = df.loc[df[OBJECT.LABEL] == 1].iloc[0]
        return np.array([row[str(col)] for col in _MEDOID_COLUMNS], dtype=np.float64)

    np.testing.assert_allclose(ring_medoid(shuffled), ring_medoid(original), rtol=0, atol=1e-9)


_LEGACY_PARAMS = {"medoid_max_pixels": 300, "random_seed": 3}


def _legacy_params_json() -> str:
    """A saved ``MeasureColor`` from before the deterministic medoid."""
    params = json.loads(MeasureColor().to_json())
    params["params"].pop("medoid_candidates")
    params["params"].update(_LEGACY_PARAMS)
    return json.dumps(params)


def _load_legacy_via_model_validate() -> MeasureColor:
    return MeasureColor.model_validate(dict(_LEGACY_PARAMS))


def _load_legacy_via_from_json() -> MeasureColor:
    return MeasureColor.from_json(_legacy_params_json())


def _load_legacy_via_pipeline_from_json() -> MeasureColor:
    """The path a saved pipeline takes: a legacy ``MeasureColor`` in ``meas``."""
    from phenotypic import ImagePipeline

    payload = json.loads(ImagePipeline(ops=[OtsuDetector()], meas=[MeasureColor()]).to_json())
    entry = payload["meas"]["MeasureColor"]["params"]
    entry.pop("medoid_candidates")
    entry.update(_LEGACY_PARAMS)
    pipeline = ImagePipeline.from_json(json.dumps(payload))
    (op,) = pipeline.meas.values()
    return op


@pytest.mark.parametrize(
    "load_legacy",
    [_load_legacy_via_model_validate, _load_legacy_via_from_json, _load_legacy_via_pipeline_from_json],
    ids=["model_validate", "from_json", "ImagePipeline.from_json"],
)
def test_legacy_medoid_fields_still_load(load_legacy):
    with pytest.warns(FutureWarning, match="medoid_max_pixels, random_seed"):
        op = load_legacy()
    assert isinstance(op, MeasureColor)
    assert op.medoid_candidates == 256
    assert not hasattr(op, "medoid_max_pixels")
    assert not hasattr(op, "random_seed")


def test_legacy_medoid_warning_is_visible_under_default_filters():
    """A user running a saved pipeline must see the warning, not just pytest.

    pytest enables DeprecationWarning display and ``pytest.warns`` captures any
    category, so the in-process test cannot tell a warning Python's default
    filters would hide. A fresh interpreter with no ``-W`` and no
    ``PYTHONWARNINGS`` can.
    """
    script = (
        "from phenotypic.measure import MeasureColor\n"
        f"MeasureColor.from_json({_legacy_params_json()!r})\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    completed = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=300
    )
    assert completed.returncode == 0, completed.stderr
    assert "MeasureColor ignores medoid_max_pixels, random_seed" in completed.stderr, completed.stderr


def test_current_fields_load_without_a_legacy_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        MeasureColor.model_validate({"medoid_candidates": 64})
