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


def test_member_declared_with_raw_tuple_is_rejected():
    import pytest
    from phenotypic.schema import MeasurementInfo

    with pytest.raises(TypeError):
        class BAD(MeasurementInfo):  # noqa: N801
            @classmethod
            def category(cls):
                return "Bad"

            X = ("X", "raw tuple no longer allowed")


def test_member_declared_with_bare_string_is_rejected():
    import pytest
    from phenotypic.schema import MeasurementInfo

    with pytest.raises(TypeError):
        class BAD2(MeasurementInfo):  # noqa: N801
            @classmethod
            def category(cls):
                return "Bad2"

            X = "X"
