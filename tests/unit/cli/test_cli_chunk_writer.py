"""Regression tests for rolling SLURM measurement aggregation."""

from __future__ import annotations

import polars as pl

from phenotypic._cli._cli_chunk_writer import _aggregate_chunks_locked
from phenotypic.schema import IMAGE
from phenotypic.sdk_ import (
    analysis_full_parquet_path,
    master_measurements_csv_path,
    master_measurements_parquet_path,
    progress_dir,
)


def test_chunk_aggregation_preserves_rolling_and_master_outputs(tmp_path) -> None:
    """Retiring static analysis leaves rolling measurement publication intact."""
    output_dir = tmp_path / "out"
    measurements_dir = output_dir / "results" / "plate_a" / "measurements"
    measurements_dir.mkdir(parents=True)
    prog_dir = progress_dir(output_dir)
    prog_dir.mkdir(parents=True)

    pl.DataFrame({"Size_Area": [10.0]}).write_parquet(
        measurements_dir / "image_1.parquet"
    )
    _aggregate_chunks_locked(output_dir, prog_dir)

    pl.DataFrame({"Size_Area": [20.0]}).write_parquet(
        measurements_dir / "image_2.parquet"
    )
    _aggregate_chunks_locked(output_dir, prog_dir)

    image_name = str(IMAGE.IMAGE_NAME)
    rolling = pl.read_parquet(analysis_full_parquet_path(prog_dir)).sort(image_name)
    master_parquet = pl.read_parquet(
        master_measurements_parquet_path(output_dir)
    ).sort(image_name)
    master_csv = pl.read_csv(master_measurements_csv_path(output_dir)).sort(
        image_name
    )

    assert rolling["Size_Area"].to_list() == [10.0, 20.0]
    assert master_parquet.equals(rolling)
    assert master_csv["Size_Area"].to_list() == [10.0, 20.0]
    assert master_csv[image_name].to_list() == ["image_1", "image_2"]

    assert not (prog_dir / "analysis_scatter.json").exists()
    assert not (prog_dir / "analysis_stats.json").exists()
    assert not (
        output_dir / "deliverables" / "overlays" / "overlay_manifest.json"
    ).exists()
