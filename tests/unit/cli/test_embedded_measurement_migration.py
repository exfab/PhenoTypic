"""Legacy external Parquets migrate into OME-Zarr table authority."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
import polars as pl

from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH, zarr_store_path


def test_migration_embeds_parquet_preserves_then_safely_deletes_source(
    legacy_run: Path,
) -> None:
    """Default preservation, idempotence, and explicit source deletion."""
    source = legacy_run / "results" / "ds" / "measurements" / "img.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "Object_Label": [1],
            "Size_Area": [25.0],
            "Metadata_ImageName": ["img"],
        }
    ).write_parquet(source)

    first = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_run)],
    )
    assert first.exit_code == 0, first.output
    table = (
        zarr_store_path(legacy_run, "ds", "img")
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    assert table.is_file(), "migration left measurement authority external"
    assert source.is_file(), "sources must be preserved by default"
    first_bytes = table.read_bytes()

    second = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "migrate", "--output", str(legacy_run)],
    )
    assert second.exit_code == 0, second.output
    assert table.read_bytes() == first_bytes

    deleting = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(legacy_run),
            "--delete-sources",
        ],
    )
    assert deleting.exit_code == 0, deleting.output
    assert not source.exists()
    assert table.is_file()
