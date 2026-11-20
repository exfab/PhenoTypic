"""``schema.json``'s builder (#281 spec C12): one row per column, from the schema enums."""

from __future__ import annotations

import polars as pl

from phenotypic.schema import OBJECT, SIZE, UNITS
from phenotypic.util._measurement_outputs import build_output_schema


def _column(schema, table, name):
    return next(c for c in schema["tables"][table] if c["name"] == name)


def test_schema_names_producer_and_unit(tiny_mirror):
    schema = build_output_schema({"measurements": tiny_mirror})
    area = next(c for c in schema["tables"]["measurements"] if c["name"] == "Size_Area")
    assert area["producer"] == "MeasureSize" and area["unit"] == str(UNITS.PX2)


def test_every_mirror_column_is_listed_in_order(tiny_mirror):
    schema = build_output_schema({"measurements": tiny_mirror})
    rows = schema["tables"]["measurements"]
    assert [row["name"] for row in rows] == tiny_mirror.columns
    assert [row["dtype"] for row in rows] == [str(t) for t in tiny_mirror.dtypes]
    for row in rows:
        assert set(row) == {"name", "dtype", "unit", "producer", "member", "desc"}


def test_a_member_column_carries_its_enum_desc():
    frame = pl.DataFrame({str(SIZE.AREA): [1.0]})
    area = _column(build_output_schema({"t": frame}), "t", str(SIZE.AREA))
    assert area["member"] == "SIZE.AREA"
    assert area["desc"] == SIZE.AREA.desc
    assert area["dtype"] == "Float64"


def test_an_owned_column_no_measurer_writes_has_no_producer():
    frame = pl.DataFrame({str(OBJECT.LABEL): [1]})
    label = _column(build_output_schema({"t": frame}), "t", str(OBJECT.LABEL))
    assert label["member"] == "OBJECT.LABEL" and label["producer"] is None


def test_an_unowned_column_is_all_null_but_its_name_and_dtype():
    frame = pl.DataFrame({"Strain": ["BY4741"]})
    assert build_output_schema({"t": frame})["tables"]["t"] == [
        {"name": "Strain", "dtype": "String", "unit": None,
         "producer": None, "member": None, "desc": None}
    ]


def test_a_parquet_footer_reads_the_same_as_its_frame(tmp_path):
    frame = pl.DataFrame({str(SIZE.AREA): [1.0], "Strain": ["x"], str(OBJECT.LABEL): [3]})
    frame.write_parquet(tmp_path / "t.parquet")
    footer = pl.read_parquet_schema(tmp_path / "t.parquet")
    assert build_output_schema({"t": footer}) == build_output_schema({"t": frame})


def test_a_pandas_frame_is_accepted():
    frame = pl.DataFrame({str(SIZE.AREA): [1.0]})
    rows = build_output_schema({"t": frame.to_pandas()})["tables"]["t"]
    assert rows[0]["member"] == "SIZE.AREA" and rows[0]["dtype"] == "float64"
