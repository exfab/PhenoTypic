"""selectable_axis_columns: the uncapped (max_cardinality=None) path (spec §16.5)."""
from __future__ import annotations

import polars as pl

from phenotypic.gui.results_viewer.colony_view._grid import selectable_axis_columns


def _frame_with_high_cardinality_metadata() -> pl.DataFrame:
    # 74 distinct plate numbers — above the 50 default cap; 3 datasets.
    return pl.DataFrame(
        {
            "Metadata_PlateNum": [str(i % 74) for i in range(148)],
            "Metadata_Dataset": [f"ds{i % 3}" for i in range(148)],
            "Object_Label": list(range(148)),
            "Size_Area": [1.0] * 148,  # measurement-prefixed → always excluded
        }
    )


def _value_sets(df: pl.DataFrame) -> dict[str, list[str]]:
    return {
        col: df.get_column(col).cast(pl.String).drop_nulls().unique().sort().to_list()
        for col in df.columns
    }


def test_default_cap_excludes_high_cardinality_metadata() -> None:
    df = _frame_with_high_cardinality_metadata()
    cols = selectable_axis_columns(df, _value_sets(df))  # default 50
    assert "Metadata_PlateNum" not in cols  # 74 > 50
    assert "Metadata_Dataset" in cols       # 3 in [2, 50]


def test_none_cap_is_uncapped_and_admits_high_cardinality() -> None:
    df = _frame_with_high_cardinality_metadata()
    cols = selectable_axis_columns(df, _value_sets(df), max_cardinality=None)
    assert "Metadata_PlateNum" in cols      # 74 now allowed
    assert "Metadata_Dataset" in cols
    # Exclusions still hold: measurement-prefixed + per-object id are dropped.
    assert "Size_Area" not in cols
    assert "Object_Label" not in cols


def test_none_cap_still_excludes_singleton_columns() -> None:
    # cardinality < 2 is excluded regardless of cap (a constant axis is useless).
    df = pl.DataFrame(
        {"Metadata_Const": ["x"] * 10, "Metadata_Dataset": ["a", "b"] * 5}
    )
    cols = selectable_axis_columns(df, _value_sets(df), max_cardinality=None)
    assert "Metadata_Const" not in cols
    assert "Metadata_Dataset" in cols
