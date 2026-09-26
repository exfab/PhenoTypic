"""split_measurements_by_category: one frame per category, feature-split context."""

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

from phenotypic.schema import CATEGORIES, IMAGE, Entry, MeasurementInfo
from phenotypic.util import split_measurements, split_measurements_by_category
from phenotypic.util import _measurement_outputs as mo

_CONTEXT = ["Metadata_Dataset", "Object_Label"]


def _frame() -> dict[str, list[object]]:
    return {
        "Metadata_Dataset": ["ds1", "ds1"],
        "Object_Label": [1, 2],
        "Size_Area": [10.0, 12.0],
        "Shape_Circularity": [0.9, 0.8],
        "ColorLab_L*Medoid": [50.0, 51.0],
        "Intensity_IntegratedIntensity": [100.0, 110.0],
        "Size_IntegratedIntensity": [100.0, 110.0],
    }


@pytest.mark.parametrize("ctor", [pd.DataFrame, pl.DataFrame], ids=["pandas", "polars"])
def test_starting_metrics_split_holds_context_then_categorized_columns(ctor) -> None:
    df = ctor(_frame())
    splits = split_measurements_by_category(df)
    assert set(splits) == {"StartingMetrics"}
    out = splits["StartingMetrics"]
    assert type(out) is type(df)
    assert list(out.columns) == [
        *_CONTEXT,
        "Size_Area",
        "ColorLab_L*Medoid",
        "Intensity_IntegratedIntensity",
        "Size_IntegratedIntensity",
    ]


def test_uncategorized_measurements_are_not_context() -> None:
    out = split_measurements_by_category(pd.DataFrame(_frame()))["StartingMetrics"]
    assert "Shape_Circularity" not in out.columns


def test_both_integrated_intensities_appear_once_each() -> None:
    out = split_measurements_by_category(pd.DataFrame(_frame()))["StartingMetrics"]
    cols = list(out.columns)
    assert cols.count("Size_IntegratedIntensity") == 1
    assert cols.count("Intensity_IntegratedIntensity") == 1


def test_context_matches_the_feature_split() -> None:
    # The real finalize frame puts IMAGE metadata after the measurements, so
    # build it measurements-first and check both splits hoist the same context.
    trailing_context = ["Metadata_Dataset", str(IMAGE.IMAGE_NAME), "Object_Label"]
    df = pd.DataFrame(
        {
            "Size_Area": [10.0],
            "Shape_Circularity": [0.9],
            "Metadata_Dataset": ["ds1"],
            str(IMAGE.IMAGE_NAME): ["img1"],
            "Object_Label": [1],
        }
    )
    feature = list(split_measurements(df)["MeasureShape"].columns)
    category = list(split_measurements_by_category(df)["StartingMetrics"].columns)
    assert feature == [*trailing_context, "Shape_Circularity"]
    assert category == [*trailing_context, "Size_Area"]


def test_member_lookup_agrees_with_every_category_member() -> None:
    # The split resolves a header to the first public class claiming it; that
    # must be the very member the category lists, or its tags are read from
    # the wrong class.
    for category in CATEGORIES:
        for member in category.members():
            assert mo._member_for_column(member.value) is member, (category, member)


def test_category_with_no_present_columns_has_no_key() -> None:
    df = pd.DataFrame(
        {"Metadata_Dataset": ["ds1"], "Object_Label": [1], "Shape_Circularity": [0.9]}
    )
    assert split_measurements_by_category(df) == {}


def test_frame_with_no_measurements_splits_to_nothing() -> None:
    assert split_measurements_by_category(pd.DataFrame({"Metadata_Dataset": ["ds1"]})) == {}


def test_non_frame_input_names_the_category_split() -> None:
    with pytest.raises(TypeError, match=r"split_measurements_by_category\(\)"):
        split_measurements_by_category({"Size_Area": [1.0]})  # type: ignore[arg-type]


def test_every_categorized_header_is_owned_by_a_producer() -> None:
    headers = [m.value for c in CATEGORIES for m in c.members()]
    groups = mo._producer_column_groups(headers)
    owned = {col for cols in groups.values() for col in cols}
    assert set(headers) <= owned, sorted(set(headers) - owned)


def test_column_in_two_categories_appears_in_both(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only one real category exists today, so substitute a two-member stand-in
    # for the module's CATEGORIES and a resolver returning a member tagged with
    # both. Declaration order (FIRST, then SECOND) must drive key order.
    class FakeCategories(str, Enum):
        FIRST = "First"
        SECOND = "Second"

        @property
        def label(self) -> str:
            return self.value

        @classmethod
        def in_order(cls, cats):
            order = list(cls)
            return tuple(sorted(set(cats), key=order.index))

    member = SimpleNamespace(categories=frozenset({FakeCategories.SECOND, FakeCategories.FIRST}))
    monkeypatch.setattr(mo, "CATEGORIES", FakeCategories)
    monkeypatch.setattr(
        mo, "_member_for_column", lambda column: member if column == "Size_Area" else None
    )
    groups = mo._category_column_groups(["Metadata_Dataset", "Size_Area"])
    assert list(groups) == ["First", "Second"]
    assert groups == {"First": ["Size_Area"], "Second": ["Size_Area"]}


def test_dynamic_header_resolves_through_member_for_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DYNAMIC(MeasurementInfo):
        @classmethod
        def metric_family(cls) -> str:
            return "Dynamic"

        @classmethod
        def member_for_header(cls, column: str):
            return cls.VALUE if column.startswith("Dynamic_Value-scale") else None

        VALUE = Entry("Value", "A value.", categories=CATEGORIES.STARTING_METRICS)

    monkeypatch.setattr(mo, "_public_info_classes", lambda: (DYNAMIC,))
    assert mo._category_column_groups(["Dynamic_Value-scale05"]) == {
        "StartingMetrics": ["Dynamic_Value-scale05"]
    }
