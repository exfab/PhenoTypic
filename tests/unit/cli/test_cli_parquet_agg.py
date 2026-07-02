from pathlib import Path

import polars as pl

from phenotypic._cli._cli_parquet_agg import (
    SOURCE_PATH_COLUMN,
    aggregate_parquet_files,
)


def test_dataset_mapping_tolerates_source_path_separator_mismatch(monkeypatch):
    """Polars source paths may not preserve the OS separator spelling."""
    windows_style_path = Path(
        r"C:\run\results\plate_A\measurements\img_001.parquet"
    )
    reported_source_path = "C:/run/results/plate_A/measurements/img_001.parquet"

    def fake_read_parquet(paths, *, include_file_paths):
        assert paths == [str(windows_style_path)]
        assert include_file_paths == SOURCE_PATH_COLUMN
        return pl.DataFrame(
            {
                "area": [10],
                SOURCE_PATH_COLUMN: [reported_source_path],
            }
        )

    monkeypatch.setattr(pl, "read_parquet", fake_read_parquet)

    frame = aggregate_parquet_files(
        file_paths=[windows_style_path],
        path_to_dataset={windows_style_path: "plate_A"},
        include_dataset_column=True,
    )

    assert frame is not None
    assert frame["MetadataExperiment_Dataset"].to_list() == ["plate_A"]
