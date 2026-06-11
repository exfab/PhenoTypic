"""Tests for the ErrorCategory + CURATION schema enums."""

from phenotypic.schema import CURATION, ErrorCategory


CORE_LABELS = [
    "oversegmented",
    "undersegmented",
    "merged",
    "background_noise",
    "debris",
    "other",
]


def test_error_category_labels_are_the_core_taxonomy():
    assert ErrorCategory.labels() == CORE_LABELS


def test_error_category_label_is_bare_and_filename_safe():
    # The persisted token is the bare label, not the prefixed value.
    assert ErrorCategory.OVERSEGMENTED.label == "oversegmented"
    assert str(ErrorCategory.OVERSEGMENTED) == "ErrorCategory_oversegmented"
    for member in ErrorCategory:
        assert member.label.replace("_", "").isalnum(), member.label


def test_error_category_descriptions_present():
    for member in ErrorCategory:
        assert member.desc, f"{member.label} missing a description"


def test_other_is_the_reserved_reasonless_bucket():
    assert ErrorCategory.OTHER.label == "other"


def test_from_label_round_trips_and_rejects_unknown():
    assert ErrorCategory.from_label("debris") is ErrorCategory.DEBRIS
    assert ErrorCategory.from_label("not_a_category") is None


def test_curation_category_column_name():
    assert str(CURATION.ERROR_CATEGORY) == "Curation_Category"
    assert CURATION.ERROR_CATEGORY.label == "Category"
