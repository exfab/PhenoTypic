"""rgb_to_xyz is a pure lift out of XyzAccessor: byte-identical output."""
import numpy as np

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
    import pytest

    image = load_synth_yeast_plate()
    with pytest.raises(ValueError, match="Unknown color_profile|illuminant"):
        rgb_to_xyz(image.rgb.normed(), gamma=image.gamma, illuminant="D99",
                   observer=image._observer)
