"""rgb_to_xyz is a pure lift out of XyzAccessor: byte-identical output."""
import numpy as np
import pytest

from phenotypic._core._image_parts.color_space_accessors._xyz_conversion import rgb_to_xyz
from phenotypic.data import load_synth_yeast_plate


def test_matches_accessor_exactly():
    """The extraction must not perturb a single value. Same code, same inputs."""
    image = load_synth_yeast_plate()
    via_accessor = image.color.XYZ[:]
    via_function = rgb_to_xyz(
        image.rgb.normed(),
        gamma=image.gamma,
        illuminant=image.illuminant,
        observer=image._observer,
    )
    np.testing.assert_array_equal(via_function, via_accessor)


def test_unknown_illuminant_raises():
    image = load_synth_yeast_plate()
    with pytest.raises(ValueError, match="Unknown color_profile|illuminant"):
        rgb_to_xyz(image.rgb.normed(), gamma=image.gamma, illuminant="D99",
                   observer=image._observer)


# Golden XYZ captured from the PRE-REFACTOR XyzAccessor (commit 92f15359a), before
# rgb_to_xyz existed. These literals are an INDEPENDENT oracle: unlike
# test_matches_accessor_exactly -- where the accessor now delegates to rgb_to_xyz, so
# both sides of the assertion move together -- a wrong `match` arm, a dropped CCTF
# decode, or a swapped illuminant cannot satisfy these values.
#
# Tolerance: colour-science computes a 3x3 matmul plus (for sRGB) a per-channel CCTF
# power law, in float64. float64 eps is 2.2e-16; ~10 chained operations bound the
# relative error near 1e-15. rtol=1e-12 sits three orders above that -- immune to
# libm/BLAS reassociation across platforms, yet the smallest real defect it must catch
# (a D65->D50 whitepoint swap) moves values by ~5e-2, and a dropped CCTF decode by ~1e-1.
GOLDEN_RGB = np.array(
    [[[10, 20, 30], [200, 150, 100]], [[255, 0, 128], [64, 192, 32]]], dtype=np.uint8
)

GOLDEN_XYZ = {
    "SRGB_D65": np.array(
        [[[0.006096741300883735, 0.006585790668061925, 0.01323280584338241], [0.3702601384632135, 0.35012152918558015, 0.16863130559150558]], [[0.4513628202705587, 0.22818512810822353, 0.22447540535826127], [0.21224700735494878, 0.388935470262757, 0.07755049686883791]]]
    ),
    "SRGB_D50": np.array(
        [[[0.005874908767879019, 0.0064772122317652385, 0.009993708846747733], [0.3875419611716216, 0.35487800296695915, 0.12867519563232502]], [[0.46696010670049665, 0.23558929434935422, 0.1680940057059938], [0.22739747776728603, 0.39016077957847456, 0.06221495451776425]]]
    ),
    "LINEAR_D65": np.array(
        [[[0.06545490196078428, 0.07292549019607841, 0.12192941176470588], [0.6045882352941174, 0.6157647058823529, 0.458]], [[0.5030039215686273, 0.248841568627451, 0.4964137254901961], [0.39540705882352917, 0.6009223529411764, 0.2138729411764706]]]
    ),
    "LINEAR_D50": np.array(
        [[[0.0640549136230721, 0.07194404637254326, 0.0925588031143878], [0.6251613884539237, 0.6206740771136094, 0.34608888371510343]], [[0.5058888574261406, 0.25000567061150686, 0.3767254443595579], [0.4195090352299197, 0.6063063875821004, 0.16040999691954388]]]
    ),
}

_OBSERVER = "CIE 1931 2 Degree Standard Observer"


@pytest.mark.parametrize("key", sorted(GOLDEN_XYZ))
def test_rgb_to_xyz_matches_pre_refactor_golden(key):
    """Guard every `match` arm against an independent oracle."""
    from phenotypic.sdk_.constants_ import GAMMA_ENCODINGS
    from phenotypic.sdk_.funcs_ import normalize_rgb_bitdepth

    gamma_name, illuminant = key.split("_")
    gamma = getattr(GAMMA_ENCODINGS, gamma_name)
    actual = rgb_to_xyz(
        normalize_rgb_bitdepth(GOLDEN_RGB),
        gamma=gamma,
        illuminant=illuminant,
        observer=_OBSERVER,
    )
    np.testing.assert_allclose(actual, GOLDEN_XYZ[key], rtol=1e-12)


def test_golden_arms_are_mutually_distinct():
    """If two arms produced identical values the golden test would be vacuous."""
    keys = sorted(GOLDEN_XYZ)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            assert not np.allclose(GOLDEN_XYZ[a], GOLDEN_XYZ[b], rtol=1e-6), (
                f"{a} and {b} are indistinguishable -- the golden fixture cannot "
                f"detect an arm swap between them"
            )
