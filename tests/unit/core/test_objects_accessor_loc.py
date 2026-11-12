"""``objects.loc(label)`` returns the colony with that label.

``get_label_idx`` used to compare the ``labels`` list to the integer as a
whole, so every ``loc`` call raised ``ValueError``. Two colonies are removed
from the synthetic plate first, so its labels are not contiguous and a label
no longer equals its position plus one.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage.measure import regionprops

from phenotypic.data import load_synth_yeast_plate

_REMOVED = (3, 4)


@pytest.fixture
def plate():
    """The synthetic plate's 96 labelled colonies, without labels 3 and 4."""
    image = load_synth_yeast_plate()
    image.objmap[np.isin(image.objmap[:], _REMOVED)] = 0
    return image


def test_labels_skip_the_removed_colonies(plate):
    assert plate.objects.labels[:4] == [1, 2, 5, 6]


@pytest.mark.parametrize(("label", "position"), [(1, 0), (5, 2), (96, 93)])
def test_get_label_idx_maps_a_label_to_its_position(plate, label, position):
    assert plate.objects.get_label_idx(label) == position


@pytest.mark.parametrize("label", [1, 5, 96])
def test_loc_returns_the_colony_with_that_label(plate, label):
    props = {region.label: region for region in regionprops(plate.objmap[:])}
    min_row, min_col, max_row, max_col = props[label].bbox

    colony = plate.objects.loc(label)

    assert colony.shape[:2] == (max_row - min_row, max_col - min_col)
    assert (colony.objmap[:] == label).sum() == props[label].area
    np.testing.assert_array_equal(
        colony.rgb[:], plate.rgb[min_row:max_row, min_col:max_col]
    )
    # The same crop as by position.
    by_position = plate.objects.iloc(plate.objects.get_label_idx(label))
    np.testing.assert_array_equal(colony.objmap[:], by_position.objmap[:])


@pytest.mark.parametrize("label", [*_REMOVED, 97])
def test_loc_refuses_a_label_that_is_not_on_the_plate(plate, label):
    with pytest.raises(IndexError, match=f"No object with label {label}"):
        plate.objects.loc(label)
