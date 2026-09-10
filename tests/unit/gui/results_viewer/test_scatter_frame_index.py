"""Frame index ranks images chronologically within a plate."""

from __future__ import annotations

import polars as pl

from phenotypic.gui.results_viewer._scatter_tab._facets import derive_frame_index


def test_index_ranks_within_each_plate_independently() -> None:
    df = pl.DataFrame(
        {
            "Metadata_PlateID": ["A", "A", "A", "B", "B"],
            "Metadata_ImageDatetime": [
                "2026-07-26T06:00:00",
                "2026-07-26T18:00:00",
                "2026-07-27T06:00:00",
                "2026-07-26T09:00:00",
                "2026-07-26T21:00:00",
            ],
        }
    )
    out = derive_frame_index(df)
    assert out["Computed_FrameIndex"].to_list() == [0, 1, 2, 0, 1]


def test_repeated_timestamps_share_one_index() -> None:
    """Two colonies in the same image are the same frame."""
    df = pl.DataFrame(
        {
            "Metadata_PlateID": ["A", "A", "A"],
            "Metadata_ImageDatetime": [
                "2026-07-26T06:00:00",
                "2026-07-26T06:00:00",
                "2026-07-26T18:00:00",
            ],
        }
    )
    assert derive_frame_index(df)["Computed_FrameIndex"].to_list() == [0, 0, 1]


def test_null_datetimes_get_a_null_index_not_zero() -> None:
    """The fixture has 81 such rows; they must be excluded, not ranked 0."""
    df = pl.DataFrame(
        {
            "Metadata_PlateID": ["A", "A"],
            "Metadata_ImageDatetime": ["2026-07-26T06:00:00", None],
        }
    )
    assert derive_frame_index(df)["Computed_FrameIndex"].to_list() == [0, None]
