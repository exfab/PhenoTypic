"""``recompile`` stops REWRITING legacy headers but keeps READING them.

This supersedes flat-metadata decision #1 ("every recompile migrates
automatically... not restricted to a special command"). Decision #3 --
permanent stored-data compatibility -- is untouched, so no existing output
directory breaks; recompile simply no longer mutates one as a side effect.

Two distinct fixtures, and the distinction is the point (OPEN-QUESTIONS D16):

``legacy_format_run``
    ``.h5`` results, no stores. ``recompile`` **fails** with a pointer to
    ``--mode migrate`` -- the forward path cannot read those images at all.

``legacy_headers_run``
    already converted to stores, metadata headers still legacy per-topic.
    ``recompile`` **succeeds**, reads them, and leaves them alone.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import dataset_measurements_dir


def _read_headers(output_dir: Path) -> set[str]:
    """Return every column name across the per-dataset measurement parquets."""
    headers: set[str] = set()
    for dataset_dir in sorted((output_dir / "results").iterdir()):
        if not dataset_dir.is_dir():
            continue
        measurements = dataset_measurements_dir(output_dir, dataset_dir.name)
        if not measurements.is_dir():
            continue
        for parquet in sorted(measurements.glob("*.parquet")):
            if parquet.name.startswith(("_", ".")):
                continue
            headers.update(pl.read_parquet(parquet).columns)
    return headers


def test_the_fixture_really_carries_legacy_headers(legacy_headers_run) -> None:
    """Guard the two tests below from passing on an already-canonical tree."""
    assert "MetadataGenetic_Strain" in _read_headers(legacy_headers_run)


def test_recompile_still_reads_legacy_headers(legacy_headers_run) -> None:
    """Decision #3 is untouched: no existing output directory breaks."""
    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "recompile", "--output", str(legacy_headers_run)]
    )
    assert result.exit_code == 0, result.output


def test_recompile_does_not_rewrite_headers(legacy_headers_run) -> None:
    before = _read_headers(legacy_headers_run)
    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "recompile", "--output", str(legacy_headers_run)]
    )
    assert result.exit_code == 0, result.output
    assert _read_headers(legacy_headers_run) == before


def test_recompile_writes_no_migration_receipt(legacy_headers_run) -> None:
    """A receipt is the durable trace of a rewrite; recompile must leave none."""
    CliRunner().invoke(
        phenotypic_cli, ["--mode", "recompile", "--output", str(legacy_headers_run)]
    )
    receipts = legacy_headers_run / ".phenotypic" / "metadata_migration"
    assert not receipts.exists() or not list(receipts.glob("*.json"))


def test_the_slurm_fanout_modules_are_gone() -> None:
    import importlib

    for name in (
        "phenotypic._cli._cli_recompile_metadata_migration_slurm",
        "phenotypic._cli._cli_recompile_metadata_migration_worker",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_migrate_performs_the_header_migration(legacy_headers_run) -> None:
    before = _read_headers(legacy_headers_run)
    result = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(legacy_headers_run)]
    )
    assert result.exit_code == 0, result.output
    after = _read_headers(legacy_headers_run)
    assert after != before
    assert "MetadataGenetic_Strain" not in after
    assert "Metadata_Strain" in after
