"""Time-axis predicate + (dataset, stem) record adapter for the Timeline tab."""
from __future__ import annotations

from pathlib import Path

import polars as pl
from PIL import Image as PILImage

from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.timeline_view._grid import (
    build_timeline_records,
    has_eligible_time_axis,
    is_large_time_axis,
    selectable_time_columns,
)
from tests._output_layout import write_master, write_measurements_mirror
from phenotypic.schema import CULTURE, IMAGE


def _value_sets(df: pl.DataFrame) -> dict[str, list[str]]:
    return {
        col: df.get_column(col).cast(pl.String).drop_nulls().unique().sort().to_list()
        for col in df.columns
    }


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Metadata_Dataset": ["ds"] * 4,
            str(IMAGE.IMAGE_NAME): ["a", "a", "b", "b"],
            "Metadata_ImageNumber": pl.Series([1, 2, 1, 2], dtype=pl.Int64),
            "Metadata_Time": ["09:00", "10:00", "09:00", "10:00"],
            "Metadata_PlateNum": ["1", "1", "2", "2"],
            "Object_Label": [10, 11, 12, 13],
            "Size_Area": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_numeric_column_is_an_eligible_time_axis() -> None:
    # M3: dtype is the AUTHORITATIVE eligibility path — Metadata_ImageNumber
    # (Int64, the UCR_029 reference column) is eligible by dtype alone.
    df = _df()
    cols = selectable_time_columns(df, _value_sets(df))
    assert "Metadata_ImageNumber" in cols   # Int64 dtype → eligible


def test_numeric_dtype_eligible_even_without_time_like_name() -> None:
    # M3: the dtype path admits a numeric column whose NAME does not match the
    # Metadata_Time-like regex — proving dtype is authoritative, name is a
    # string-typed FALLBACK (for String-stored time columns).
    df = pl.DataFrame({"Metadata_Generation": pl.Series([0, 1, 2], dtype=pl.Int64)})
    cols = selectable_time_columns(df, _value_sets(df))
    assert str(CULTURE.GENERATION) in cols


def test_metadata_time_name_match_is_eligible_even_if_string() -> None:
    # M3: the name regex is the FALLBACK path — Metadata_Time stored as pl.String
    # (join_metadata casts join keys to String) has no numeric/temporal dtype, but
    # its name matches the Metadata_Time-like pattern → still offered.
    df = _df()
    cols = selectable_time_columns(df, _value_sets(df))
    assert str(CULTURE.TIME) in cols


def test_legacy_time_header_is_returned_in_canonical_form() -> None:
    source = pl.DataFrame(
        {
            "Metadata_Time": ["09:00", "10:00"],
            "Object_Label": [1, 2],
        }
    )
    original = source.clone()

    cols = selectable_time_columns(source, _value_sets(source))

    assert str(CULTURE.TIME) in cols
    assert source.equals(original)


def test_measurement_and_object_label_columns_are_excluded() -> None:
    df = _df()
    cols = selectable_time_columns(df, _value_sets(df))
    assert "Size_Area" not in cols       # measurement-prefixed
    assert "Object_Label" not in cols    # per-object id


def test_no_cardinality_cap_on_time_axis() -> None:
    # 200 distinct numeric timepoints must remain eligible (the 50-cap on the
    # row axis would have hidden a long course — spec §15.2).
    df = pl.DataFrame({"Metadata_ImageNumber": pl.Series(range(200), dtype=pl.Int64)})
    cols = selectable_time_columns(df, _value_sets(df))
    assert "Metadata_ImageNumber" in cols


def test_is_large_time_axis() -> None:
    assert is_large_time_axis(150) is True
    assert is_large_time_axis(50) is False
    assert is_large_time_axis(100) is False   # threshold is "> threshold"


def test_empty_state_predicate_false_without_any_time_column() -> None:
    df = pl.DataFrame(
        {
            "Metadata_Dataset": ["ds", "ds"],
            str(IMAGE.IMAGE_NAME): ["a", "b"],
            "Metadata_PlateNum": ["1", "2"],  # categorical, no name/dtype match
            "Object_Label": [1, 2],
            "Size_Area": [1.0, 2.0],
        }
    )
    assert has_eligible_time_axis(df, _value_sets(df)) is False


def test_empty_state_predicate_true_with_a_time_column() -> None:
    df = _df()
    assert has_eligible_time_axis(df, _value_sets(df)) is True


def _make_output_root(tmp_path: Path) -> OutputRoot:
    cli_out = tmp_path / "out"
    df = _df()
    write_master(cli_out, df)
    write_measurements_mirror(cli_out, df)
    (cli_out / "results" / "ds" / "measurements").mkdir(parents=True, exist_ok=True)
    overlays = cli_out / "deliverables" / "overlays" / "ds"
    overlays.mkdir(parents=True, exist_ok=True)
    for stem in ("a", "b"):
        PILImage.new("RGB", (40, 30), (10, 20, 30)).save(overlays / f"{stem}.png")
    return OutputRoot.discover(
        cli_out,
        cache_root=tmp_path / ".test-phenotypic-viewer-cache",
    )


def test_build_timeline_records_emits_one_per_overlay_pair(tmp_path: Path) -> None:
    root = _make_output_root(tmp_path)
    records = build_timeline_records(
        root, root.master_df, row_col="Metadata_PlateNum", time_col="Metadata_ImageNumber"
    )
    refs = {r["cell_ref"] for r in records}
    assert ("ds", "a") in refs and ("ds", "b") in refs
    rows = {r["row_value"] for r in records}
    assert rows == {"1", "2"}        # plate numbers
    times = {r["time_value"] for r in records}
    assert times == {"1", "2"}       # image numbers (stringified)
