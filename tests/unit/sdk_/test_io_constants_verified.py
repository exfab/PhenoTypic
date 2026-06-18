"""Path-helper tests for the GUI-written ``deliverables/verified.parquet``."""

from pathlib import Path

from phenotypic.sdk_ import VERIFIED_PARQUET, deliverables_dir, verified_parquet_path


def test_verified_parquet_filename():
    assert VERIFIED_PARQUET == "verified.parquet"


def test_verified_parquet_path_under_deliverables(tmp_path: Path):
    out = tmp_path / "run"
    assert verified_parquet_path(out) == deliverables_dir(out) / "verified.parquet"
