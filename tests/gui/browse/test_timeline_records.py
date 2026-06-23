"""Per-axis source resolution for the Browse Timeline (pure)."""
from __future__ import annotations

from phenotypic.gui.browse._timeline_records import (
    BrowseAxisConfig,
    build_browse_records,
)


def _datasets() -> dict[str, list[str]]:
    # Two timepoint folders, each holding the same two plate filenames.
    return {
        "2024-01-01": ["plateA.tif", "plateB.tif"],
        "2024-01-02": ["plateA.tif", "plateB.tif"],
    }


def test_default_folder_rows_exif_time() -> None:
    # row=folder, time=exif (fallback to filename when EXIF missing).
    config = BrowseAxisConfig(row_source="folder", time_source="exif")
    records, warnings = build_browse_records(
        _datasets(), "src", config, capture_time_of=lambda rel: None
    )
    rows = {r["row_value"] for r in records}
    assert rows == {"2024-01-01", "2024-01-02"}
    assert warnings == []
    # cell_ref is the sandbox-relative POSIX path.
    refs = {r["cell_ref"] for r in records}
    assert "src/2024-01-01/plateA.tif" in refs


def test_folder_time_source_orders_by_folder_not_filename() -> None:
    # Folder-per-timepoint layout: each folder is a timepoint, the SAME plate
    # filename repeats across folders. With time_source="folder" the time axis
    # must be the FOLDER names (not the repeated filenames), so a plate's
    # time-course spans the folders. Rows stay the (repeated) plate filenames'
    # folder grouping is irrelevant here — we drive rows off a pattern so each
    # plate is one row across all timepoint folders.
    datasets = {
        "t0": ["plateA.tif", "plateB.tif"],
        "t1": ["plateA.tif", "plateB.tif"],
        "t2": ["plateA.tif", "plateB.tif"],
    }
    config = BrowseAxisConfig(
        row_source="pattern", time_source="folder", pattern="{plate}"
    )
    records, warnings = build_browse_records(
        datasets, "src", config, capture_time_of=lambda rel: None
    )
    assert warnings == []
    # Time axis == the folder names (one per timepoint), NOT the filenames.
    times = {r["time_value"] for r in records}
    assert times == {"t0", "t1", "t2"}
    # Each plate (pattern {plate}) is one row spanning all three folders.
    rows = {r["row_value"] for r in records}
    assert rows == {"t0/plateA", "t0/plateB", "t1/plateA", "t1/plateB",
                    "t2/plateA", "t2/plateB"}
    # And crucially the time value is never a filename stem.
    assert "plateA" not in times and "plateB" not in times


def test_pattern_rows_are_folder_scoped() -> None:
    # Flat-style names inside each folder; row=pattern {plate}, time=pattern {time}.
    datasets = {
        "runX": ["plateA_t01.tif", "plateA_t02.tif", "plateB_t01.tif"],
    }
    config = BrowseAxisConfig(
        row_source="pattern", time_source="pattern", pattern="{plate}_t{time}"
    )
    records, _ = build_browse_records(
        datasets, "src", config, capture_time_of=lambda rel: None
    )
    rows = {r["row_value"] for r in records}
    # Folder-scoped: row key is "<folder>/<plate>".
    assert rows == {"runX/plateA", "runX/plateB"}
    times = {r["time_value"] for r in records}
    assert times == {"01", "02"}


def test_csv_source_joins_by_stem_and_warns_on_cross_folder_collision() -> None:
    csv_rows = [
        {"image": "plateA", "media": "YPD", "tp": "0h"},
        {"image": "plateB", "media": "SD", "tp": "0h"},
    ]
    config = BrowseAxisConfig(
        row_source="csv",
        time_source="csv",
        csv_image_col="image",
        row_csv_col="media",
        time_csv_col="tp",
    )
    records, warnings = build_browse_records(
        _datasets(), "src", config, csv_rows=csv_rows, capture_time_of=lambda rel: None
    )
    rows = {r["row_value"] for r in records}
    assert rows == {"YPD", "SD"}
    # plateA/plateB stems each appear in TWO folders while a CSV axis is active
    # → collision warning (same stem can't disambiguate per-folder rows).
    assert any("plateA" in w or "stem" in w.lower() for w in warnings)
