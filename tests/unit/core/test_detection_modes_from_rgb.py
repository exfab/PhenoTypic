"""Golden gate: compute() and compute_from_rgb() must agree, for every mode.

Tolerance derivation (mechanism, not guess): the two paths differ only by float
accumulation order in a 3-term dot product (rgb2gray) or a 3x3 matmul
(RGB_to_XYZ), computed in float64 then cast to float32. float32 eps is 1.19e-7;
three fused operations bound the discrepancy at ~3 x 1.19e-7 = 3.6e-7. atol=1e-6
sits ~3x above that -- loose enough to survive reassociation, tight enough that a
channel swap or a dropped CCTF decode (which move values by >1e-2) fails.

Why the golden literals exist
-----------------------------
``test_compute_from_rgb_matches_compute`` is a **wiring test, not a correctness
guard**. Nine of the eleven modes had ``compute()`` rewritten to delegate to
``compute_from_rgb()``, so both sides of that assertion now route through the same
code: mutating ``compute_from_rgb`` mutates both sides equally and the assertion
can never catch it. Verified empirically -- passing a wrong ``observer`` to
``rgb_to_xyz`` inside ``_LabChannelMode.compute_from_rgb`` still passed.

``GOLDEN`` and ``GOLDEN_MEAN`` are therefore the independent oracle. Their literals
were captured from the **pre-refactor** ``compute()`` implementations (the inline
``image.color.Lab`` / ``image.color.hsv`` / channel-slice bodies at ``92f15359a``),
so they move independently of the code under test.
"""

import numpy as np
import pytest

from phenotypic._core._image_parts.detection_modes import (
    available_modes,
    get_detection_mode,
)
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_.funcs_ import normalize_rgb_bitdepth

ATOL = 1e-6

#: Pixel coordinates the golden literals were sampled at.
COORDS = ((0, 0), (37, 91), (128, 256), (300, 300), (511, 511))

#: ``compute()`` output at ``COORDS``, captured from the pre-refactor modes.
GOLDEN: dict[str, list[float]] = {
    "HsvS": [
        0.035211268812417984,
        0.03546099364757538,
        0.04583333432674408,
        0.03496503457427025,
        0.03496503457427025,
    ],
    "HsvV": [
        0.5568627715110779,
        0.5529412031173706,
        0.9411764740943909,
        0.5607843399047852,
        0.5607843399047852,
    ],
    "InvS": [
        0.9647887349128723,
        0.9645389914512634,
        0.9541666507720947,
        0.9650349617004395,
        0.9650349617004395,
    ],
    "LabA": [
        0.495299369096756,
        0.4952908456325531,
        0.5105344653129578,
        0.49385741353034973,
        0.49385741353034973,
    ],
    "LabB": [
        0.5113918781280518,
        0.5114040970802307,
        0.5114434361457825,
        0.5108820199966431,
        0.5108820199966431,
    ],
    "LabL": [
        0.5872133374214172,
        0.5833606719970703,
        0.9252877831459045,
        0.5902552604675293,
        0.5902552604675293,
    ],
    "MinRGB": [
        0.5372549295425415,
        0.5333333611488342,
        0.8980392217636108,
        0.5411764979362488,
        0.5411764979362488,
    ],
    "blue": [
        0.5372549295425415,
        0.5333333611488342,
        0.8980392217636108,
        0.5411764979362488,
        0.5411764979362488,
    ],
    "gray": [
        0.5537823438644409,
        0.5498607754707336,
        0.915622353553772,
        0.5568705797195435,
        0.5568705797195435,
    ],
    "green": [
        0.5568627715110779,
        0.5529412031173706,
        0.9098039269447327,
        0.5607843399047852,
        0.5607843399047852,
    ],
    "red": [
        0.5490196347236633,
        0.545098066329956,
        0.9411764740943909,
        0.5490196347236633,
        0.5490196347236633,
    ],
}

#: Mean of the full ``compute()`` output, captured from the pre-refactor modes.
#: Five sampled pixels cannot see a spatial permutation; the mean can.
GOLDEN_MEAN: dict[str, float] = {
    "HsvS": 0.037739434118301142,
    "HsvV": 0.64807180202230807,
    "InvS": 0.962260566258058,
    "LabA": 0.49807997000440957,
    "LabB": 0.5113914558349798,
    "LabL": 0.66776284580764667,
    "MinRGB": 0.62297056985832755,
    "blue": 0.62297056985832755,
    "gray": 0.6397646796940516,
    "green": 0.64121086288876827,
    "red": 0.64059426161833111,
}

#: A per-channel tint. Uniform scaling is invisible to the saturation-based modes
#: (``HsvS``/``InvS`` are scale-invariant) and a single-channel scale is invisible
#: to the other two channel modes. Only a non-uniform, all-channel perturbation
#: moves every one of the eleven.
TINT = np.array([0.4, 0.7, 0.9], dtype=np.float64)


@pytest.fixture(scope="module")
def image():
    return load_synth_yeast_plate()


@pytest.fixture(scope="module")
def rgb(image):
    return normalize_rgb_bitdepth(image.rgb[:])


def test_all_eleven_modes_are_registered():
    assert len(available_modes()) == 11


def test_golden_covers_every_registered_mode():
    """A mode added without a golden literal must fail here, not silently skip."""
    assert set(GOLDEN) == set(available_modes())
    assert set(GOLDEN_MEAN) == set(available_modes())


def test_golden_arms_are_mutually_distinct():
    """Guard against a vacuous fixture: near-identical arms would gate nothing.

    ``blue`` and ``MinRGB`` are the sole legitimate duplicate -- on the synthetic
    yeast plate the blue channel *is* the per-pixel minimum at every pixel, so the
    two modes agree by construction, not by accident.
    """
    duplicates = {
        frozenset((a, b))
        for a in GOLDEN
        for b in GOLDEN
        if a != b and np.allclose(GOLDEN[a], GOLDEN[b], atol=ATOL)
    }
    assert duplicates == {frozenset(("blue", "MinRGB"))}


@pytest.mark.parametrize("name", available_modes())
def test_compute_from_rgb_matches_golden(name, image, rgb):
    """The independent oracle. Literals predate the delegation refactor."""
    actual = get_detection_mode(name).compute_from_rgb(rgb, image=image)
    sampled = [float(actual[r, c]) for r, c in COORDS]
    np.testing.assert_allclose(sampled, GOLDEN[name], atol=ATOL)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64).mean(), GOLDEN_MEAN[name], atol=ATOL
    )


@pytest.mark.parametrize("name", available_modes())
def test_compute_from_rgb_matches_compute(name, image, rgb):
    """Wiring check: ``compute()`` forwards the right array. See module docstring --
    this cannot catch a defect inside ``compute_from_rgb`` for the delegating modes."""
    mode = get_detection_mode(name)
    expected = mode.compute(image)
    actual = mode.compute_from_rgb(rgb, image=image)
    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=ATOL)


@pytest.mark.parametrize("name", available_modes())
def test_compute_from_rgb_output_is_unit_range(name, image, rgb):
    out = get_detection_mode(name).compute_from_rgb(rgb, image=image)
    assert out.min() >= -ATOL and out.max() <= 1.0 + ATOL


@pytest.mark.parametrize("name", available_modes())
def test_compute_from_rgb_ignores_the_images_own_rgb(name, image, rgb):
    """Feeding a different array must produce a different result -- proving the
    method reads `rgb`, not `image._data.rgb`."""
    mode = get_detection_mode(name)
    baseline = mode.compute_from_rgb(rgb, image=image)
    tinted = mode.compute_from_rgb(rgb*TINT, image=image)
    assert not np.allclose(baseline, tinted, atol=ATOL)


def test_min_rgb_from_rgb_is_linear_in_its_argument(image, rgb):
    """A stronger statement than "differs": MinRGB is positively homogeneous, so
    halving the input must halve the output exactly."""
    mode = get_detection_mode("MinRGB")
    baseline = mode.compute_from_rgb(rgb, image=image)
    darkened = mode.compute_from_rgb(rgb*0.5, image=image)
    np.testing.assert_allclose(darkened, baseline*0.5, atol=ATOL)
