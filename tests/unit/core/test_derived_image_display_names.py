"""A crop, colony or grid section never takes over its plate's napari layers.

A slice keeps its source's metadata, name included (0.21.0), and ``napari()``
names a layer ``{accessor}_{image name}`` and replaces a layer that already has
that name. So a derived image's default name adds its type and the first 8 hex
digits of its UUID, and so do the default matplotlib titles. A plate's own
layer names are unchanged. CropImage returns the image itself, cropped,
so it keeps the plate's layers. The matching matplotlib titles are tested in
``test_derived_image_titles.py``, which does not need napari.
"""

from __future__ import annotations

import re

import pytest

from phenotypic import GridImage, Image
from phenotypic.correction import CropImage
from phenotypic.data import load_synth_yeast_plate

pytest.importorskip("napari")

_PLATE_NAME = "Synthetic96PlateWithObjects"
_ACCESSORS = ("rgb", "gray", "detect_mat", "objmask", "objmap")


class _FakeLayer:
    def __init__(self, data):
        self.data = data


class _FakeLayers(dict):
    """``viewer.layers``: a ``KeyError`` for a missing name, like napari's."""


class _FakeViewer:
    """Records the layers ``napari()`` adds or replaces, keyed by name."""

    def __init__(self):
        self.layers = _FakeLayers()

    def add_image(self, data, *, name, **kwargs):
        self.layers[name] = _FakeLayer(data)

    def add_labels(self, data, *, name, **kwargs):
        self.layers[name] = _FakeLayer(data)


@pytest.fixture(autouse=True)
def _napari_available(monkeypatch):
    monkeypatch.setattr(
        "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
        True,
    )


def _plate(image_class):
    """The synthetic plate, with its 96 colonies, as an Image or a GridImage."""
    plate = load_synth_yeast_plate()
    return plate if image_class is GridImage else Image(plate)


def _derived_name(prefix: str, kind: str) -> re.Pattern[str]:
    return re.compile(rf"{prefix}_{_PLATE_NAME}_{kind}_[0-9a-f]{{8}}")


@pytest.mark.parametrize("image_class", [Image, GridImage])
def test_root_image_layer_names_are_unchanged(image_class):
    plate = _plate(image_class)
    viewer = _FakeViewer()

    for accessor in _ACCESSORS:
        getattr(plate, accessor).napari(viewer=viewer)

    assert set(viewer.layers) == {f"{accessor}_{_PLATE_NAME}" for accessor in _ACCESSORS}


@pytest.mark.parametrize("image_class", [Image, GridImage])
def test_plate_slice_and_colony_get_distinct_layers(image_class):
    plate = _plate(image_class)
    viewer = _FakeViewer()
    for accessor in ("rgb", "gray", "objmap"):
        getattr(plate, accessor).napari(viewer=viewer)
    plate_shapes = {name: layer.data.shape for name, layer in viewer.layers.items()}

    crop = plate[100:300, 100:300]
    crop.rgb.napari(viewer=viewer)
    crop.gray.napari(viewer=viewer)
    colony = plate.objects[0]
    colony.gray.napari(viewer=viewer)
    colony.objmap.napari(viewer=viewer)
    plate.objects.iloc(0).rgb.napari(viewer=viewer)
    derived = [name for name in viewer.layers if name not in plate_shapes]

    assert len(viewer.layers) == len(plate_shapes) + 5
    assert {name: viewer.layers[name].data.shape for name in plate_shapes} == plate_shapes
    assert plate_shapes[f"rgb_{_PLATE_NAME}"] == (600, 800, 3)
    expected = [
        _derived_name("rgb", "crop"),
        _derived_name("gray", "crop"),
        _derived_name("gray", "object"),
        _derived_name("objmap", "object"),
        _derived_name("rgb", "crop"),
    ]
    assert len(derived) == len(expected)
    for pattern, name in zip(expected, derived):
        assert pattern.fullmatch(name), name


def test_grid_section_gets_its_own_layer():
    plate = load_synth_yeast_plate()
    viewer = _FakeViewer()
    plate.objmap.napari(viewer=viewer)

    plate.grid[5].objmap.napari(viewer=viewer)

    assert viewer.layers[f"objmap_{_PLATE_NAME}"].data.shape == (600, 800)
    (section_layer,) = [name for name in viewer.layers if "gridsection" in name]
    assert _derived_name("objmap", "gridsection").fullmatch(section_layer)


def test_two_crops_of_one_plate_do_not_share_a_layer():
    plate = load_synth_yeast_plate()
    viewer = _FakeViewer()

    plate[0:100, 0:100].gray.napari(viewer=viewer)
    plate[100:300, 100:300].gray.napari(viewer=viewer)

    assert sorted(layer.data.shape for layer in viewer.layers.values()) == [
        (100, 100),
        (200, 200),
    ]


def test_explicit_name_is_used_as_given():
    crop = load_synth_yeast_plate()[100:300, 100:300]
    viewer = _FakeViewer()

    crop.gray.napari("colony_a1", viewer=viewer)

    assert set(viewer.layers) == {"gray_colony_a1"}


@pytest.mark.parametrize("image_class", [Image, GridImage])
def test_two_cropimage_runs_give_the_same_default_layer_names(image_class):
    """Re-running a crop updates the plate's layers instead of adding new ones."""
    plate = _plate(image_class)
    crop = CropImage(left=20, right=20, top=20, bottom=20)
    viewer = _FakeViewer()

    for _ in range(2):
        cropped = crop.apply(plate)
        cropped.rgb.napari(viewer=viewer)
        cropped.objmap.napari(viewer=viewer)

    assert set(viewer.layers) == {f"rgb_{_PLATE_NAME}", f"objmap_{_PLATE_NAME}"}
    assert viewer.layers[f"rgb_{_PLATE_NAME}"].data.shape == (560, 760, 3)
