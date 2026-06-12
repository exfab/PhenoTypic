"""Entry — the declarative value type for MeasurementInfo members."""

import pytest

from phenotypic.schema import Entry


def test_entry_minimal_defaults():
    e = Entry("Area")
    assert e.label == "Area"
    assert e.desc == ""
    assert e.bio_desc == ""
    assert e.image is None


def test_entry_positional_label_and_desc():
    e = Entry("Area", "Pixel count of the mask.")
    assert e.desc == "Pixel count of the mask."


def test_entry_optional_fields_are_keyword_only():
    with pytest.raises(TypeError):
        Entry("Area", "tech", "bio")  # type: ignore[misc]


def test_entry_rich():
    e = Entry("Area", "tech", bio_desc="biology", image="shape/area.png")
    assert (e.bio_desc, e.image) == ("biology", "shape/area.png")


def test_entry_is_frozen():
    e = Entry("Area")
    with pytest.raises(Exception):  # FrozenInstanceError
        e.label = "Other"  # type: ignore[misc]


def test_entry_rejects_empty_label():
    with pytest.raises(ValueError):
        Entry("")


def test_entry_rejects_non_string_desc():
    with pytest.raises(TypeError):
        Entry("Area", 123)  # type: ignore[arg-type]
