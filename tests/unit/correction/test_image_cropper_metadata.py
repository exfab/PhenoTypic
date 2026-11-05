"""A slice or a crop keeps the metadata of the image it came from.

A slice is a new image (its own UUID) of the same plate, so every other
metadata key keeps its value: the file name and suffix, the bit depth, the
EXIF block, and keys a user set by hand. Only two keys change on purpose:
``Metadata_UUID``, and ``Metadata_ImageType``, which a slice sets to ``Crop``.
``CropImage`` on a ``GridImage`` rebuilds a ``GridImage``, which stamps its
own ``GridImage`` type, as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import phenotypic
from phenotypic import GridImage, Image
from phenotypic.correction import CropImage
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.schema import GENETIC, GRID, IMAGE
from phenotypic.sdk_.constants_ import IMAGE_TYPES

_PLATE_FILE = Path(phenotypic.data.__file__).parent / "StandardDay6.jpg"
_CHANGED_ON_PURPOSE = {str(IMAGE.UUID), str(IMAGE.IMAGE_TYPE)}
# A key outside the schema, kept as a literal key by the metadata accessor.
_FREE_FORM_KEY = "Incubator"


def _read_plate(image_class):
    """Read the sample plate from its file and set two keys by hand."""
    image = image_class.imread(_PLATE_FILE)
    image.metadata[GENETIC.STRAIN] = "strain_0042"
    image.metadata[_FREE_FORM_KEY] = "shelf-3"
    return image


def _keys(image) -> dict[str, object]:
    """Every metadata key the accessor exposes, by its header spelling."""
    return {str(key): value for key, value in image.metadata.items()}


def _slice(image):
    return image[20:-20, 20:-20]


def _crop(image):
    return CropImage(left=20, right=20, top=20, bottom=20).apply(image)


@pytest.mark.parametrize(
    ("image_class", "derive", "expected_type"),
    [
        pytest.param(Image, _slice, IMAGE_TYPES.CROP, id="Image-slice"),
        pytest.param(GridImage, _slice, IMAGE_TYPES.CROP, id="GridImage-slice"),
        pytest.param(Image, _crop, IMAGE_TYPES.CROP, id="Image-CropImage"),
        pytest.param(GridImage, _crop, IMAGE_TYPES.GRID, id="GridImage-CropImage"),
    ],
)
def test_derived_image_keeps_every_metadata_key(image_class, derive, expected_type):
    source = _read_plate(image_class)
    before = _keys(source)
    source_imported = dict(source._metadata.imported)

    derived = derive(source)
    after = _keys(derived)

    assert source_imported, "the sample JPEG should carry EXIF to compare"
    assert set(after) == set(before)
    assert {k: v for k, v in after.items() if k not in _CHANGED_ON_PURPOSE} == {
        k: v for k, v in before.items() if k not in _CHANGED_ON_PURPOSE
    }
    assert derived._metadata.imported == source_imported
    assert derived.metadata[IMAGE.UUID] != source.metadata[IMAGE.UUID]
    assert derived.metadata[IMAGE.IMAGE_TYPE] == expected_type.value
    # The source is untouched.
    assert _keys(source) == before


def test_user_keys_spelled_out():
    """The hand-set keys and the file suffix, named, on a sliced GridImage."""
    sliced = _slice(_read_plate(GridImage))

    assert sliced.metadata[GENETIC.STRAIN] == "strain_0042"
    assert sliced.metadata[_FREE_FORM_KEY] == "shelf-3"
    assert sliced.metadata[IMAGE.SUFFIX] == ".jpg"
    assert sliced.metadata[IMAGE.IMAGE_NAME] == "StandardDay6"


@pytest.mark.parametrize("image_class", [Image, GridImage])
def test_unnamed_array_image_crop_keeps_its_unset_name(image_class):
    """An image built from an array has no name, and its crop has none either.

    ``name`` falls back to the image's own UUID in both cases, so a crop's
    ``name`` is its own UUID, as before; only the stored key no longer turns
    into a UUID string.
    """
    source = image_class(load_synth_yeast_plate().rgb[:].copy())
    cropped = _crop(source)

    assert source.metadata[IMAGE.IMAGE_NAME] is None
    assert cropped.metadata[IMAGE.IMAGE_NAME] is None
    assert cropped.name == str(cropped.uuid)


def test_slice_metadata_is_a_copy():
    """Changing a slice's metadata leaves the source's alone."""
    source = _read_plate(Image)
    sliced = _slice(source)

    sliced.metadata[GENETIC.STRAIN] = "strain_0099"
    sliced._metadata.imported.clear()

    assert source.metadata[GENETIC.STRAIN] == "strain_0042"
    assert source._metadata.imported


@pytest.fixture(scope="module")
def detected_plates():
    """The sample plate read from its file, keys set, Otsu-detected, per class."""
    return {cls: OtsuDetector().apply(_read_plate(cls)) for cls in (Image, GridImage)}


def _first_label(image):
    return image.objects.labels[0]


def _first_occupied_section(image):
    info = image.grid.info(include_metadata=False)
    return int(info[str(GRID.ROW_MAJOR_IDX)].iloc[0])


@pytest.mark.parametrize(
    ("image_class", "derive", "expected_type"),
    [
        pytest.param(Image, lambda im: im.objects[0], IMAGE_TYPES.OBJECT, id="Image-objects[0]"),
        pytest.param(GridImage, lambda im: im.objects[0], IMAGE_TYPES.OBJECT, id="GridImage-objects[0]"),
        pytest.param(Image, lambda im: im.objects.iloc(0), IMAGE_TYPES.CROP, id="Image-iloc"),
        pytest.param(GridImage, lambda im: im.objects.iloc(0), IMAGE_TYPES.CROP, id="GridImage-iloc"),
        pytest.param(Image, lambda im: im.objects.loc(_first_label(im)), IMAGE_TYPES.CROP, id="Image-loc"),
        pytest.param(
            GridImage, lambda im: im.objects.loc(_first_label(im)), IMAGE_TYPES.CROP,
            id="GridImage-loc",
        ),
        pytest.param(
            GridImage, lambda im: im.grid[_first_occupied_section(im)],
            IMAGE_TYPES.GRID_SECTION, id="GridImage-grid-section",
        ),
    ],
)
def test_colony_and_section_images_keep_metadata(
    detected_plates, image_class, derive, expected_type
):
    """Colony crops and grid sections are built by slicing, so they keep it too."""
    source = detected_plates[image_class]
    before = _keys(source)

    derived = derive(source)
    after = _keys(derived)

    assert derived.metadata[IMAGE.IMAGE_TYPE] == expected_type.value
    assert {k: v for k, v in after.items() if k not in _CHANGED_ON_PURPOSE} == {
        k: v for k, v in before.items() if k not in _CHANGED_ON_PURPOSE
    }
    assert derived._metadata.imported == source._metadata.imported


def _operation_names(image):
    return [record["operation_name"] for record in image.provenance]


def test_slice_starts_a_fresh_journal_and_crop_records_one_operation():
    """A slice does not copy the provenance journal; CropImage.apply extends it."""
    source = OtsuDetector().apply(_read_plate(Image))
    assert _operation_names(source) == ["OtsuDetector"]

    sliced = _slice(source)
    cropped = _crop(source)

    assert sliced._metadata.provenance_journal["applications"] == []
    assert _operation_names(cropped) == ["OtsuDetector", "CropImage"]
