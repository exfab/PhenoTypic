"""Unit tests for CLI measurement source discovery helpers."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from phenotypic._cli._measurement_sources import (
    add_metadata_image_name_from_filename,
    discover_measurement_sources,
    measurement_sources_by_path,
)
from phenotypic.schema import METADATA
from phenotypic.sdk_ import DATASET_AGGREGATED_PARQUET


def _touch(path: Path) -> None:
    """Create an empty test file and parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_discover_measurement_sources_prefers_dataset_aggregated_file(
    tmp_path: Path,
) -> None:
    """A dataset aggregate suppresses individual per-image parquet sources."""
    agg = (
        tmp_path
        / "results"
        / "plate_a"
        / "measurements"
        / DATASET_AGGREGATED_PARQUET
    )
    raw = tmp_path / "results" / "plate_a" / "measurements" / "img.parquet"
    _touch(agg)
    _touch(raw)

    sources = discover_measurement_sources(tmp_path, ["plate_a"])

    assert [(src.path, src.dataset) for src in sources] == [(agg, "plate_a")]


def test_discover_measurement_sources_skips_internal_files_and_sorts(
    tmp_path: Path,
) -> None:
    """Without an aggregate, non-internal parquet files are sorted."""
    meas_dir = tmp_path / "results" / "plate_a" / "measurements"
    b_path = meas_dir / "b.parquet"
    a_path = meas_dir / "a.parquet"
    _touch(b_path)
    _touch(a_path)
    _touch(meas_dir / "_internal.parquet")

    sources = discover_measurement_sources(tmp_path, ["plate_a"])

    assert [src.path for src in sources] == [a_path, b_path]
    assert [src.dataset for src in sources] == ["plate_a", "plate_a"]


def test_discover_measurement_sources_can_discover_dataset_dirs(
    tmp_path: Path,
) -> None:
    """When dataset names are omitted, subdirectories under results are used."""
    a_path = tmp_path / "results" / "a" / "measurements" / "img.parquet"
    b_path = tmp_path / "results" / "b" / "measurements" / "img.parquet"
    _touch(b_path)
    _touch(a_path)

    sources = discover_measurement_sources(tmp_path)

    assert [(src.path, src.dataset) for src in sources] == [
        (a_path, "a"),
        (b_path, "b"),
    ]


def test_measurement_sources_by_path_preserves_source_mapping(
    tmp_path: Path,
) -> None:
    """The path-to-dataset mapping matches aggregate_parquet_files input."""
    path = tmp_path / "results" / "a" / "measurements" / "img.parquet"
    _touch(path)
    sources = discover_measurement_sources(tmp_path, ["a"])

    assert measurement_sources_by_path(sources) == {path: "a"}


def test_add_metadata_image_name_from_filename_derives_and_drops_filename() -> None:
    """Filename is converted to Metadata_ImageName only when needed."""
    frame = pl.DataFrame({"filename": ["/tmp/plate/img001.parquet"], "value": [7]})

    out = add_metadata_image_name_from_filename(frame)

    assert "filename" not in out.columns
    assert out[str(METADATA.IMAGE_NAME)].to_list() == ["img001"]
    assert out["value"].to_list() == [7]


def test_add_metadata_image_name_from_filename_preserves_existing_column() -> None:
    """Existing Metadata_ImageName wins over filename-derived values."""
    frame = pl.DataFrame(
        {
            "filename": ["/tmp/plate/wrong.parquet"],
            str(METADATA.IMAGE_NAME): ["kept"],
        }
    )

    out = add_metadata_image_name_from_filename(frame)

    assert "filename" not in out.columns
    assert out[str(METADATA.IMAGE_NAME)].to_list() == ["kept"]
