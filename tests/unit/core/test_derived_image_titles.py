"""Default matplotlib titles tell a crop from its plate.

A slice keeps its source's name (0.21.0), so its default title adds its type
and the first 8 hex digits of its UUID, as its napari layer names do
(``test_derived_image_display_names.py``). A plate's titles are unchanged.
Needs matplotlib only, not napari.
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt

from phenotypic.data import load_synth_yeast_plate

_PLATE_NAME = "Synthetic96PlateWithObjects"


def test_default_titles_tell_a_crop_from_its_plate():
    plate = load_synth_yeast_plate()
    crop = plate[100:300, 100:300]
    figures = []
    try:
        titles = {}
        for label, image in (("plate", plate), ("crop", crop)):
            fig, _ = image.gray.histogram()
            channel_fig, channel_ax = image.rgb.show(channel=0)
            figures += [fig, channel_fig]
            titles[label] = (fig.axes[0].get_title(), channel_ax.get_title())
    finally:
        for fig in figures:
            plt.close(fig)

    assert titles["plate"] == (_PLATE_NAME, f"{_PLATE_NAME} - Channel 0")
    crop_title, crop_channel_title = titles["crop"]
    assert re.fullmatch(rf"{_PLATE_NAME}_crop_[0-9a-f]{{8}}", crop_title)
    assert crop_channel_title == f"{crop_title} - Channel 0"
