"""Startup metadata snapshot tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.phenotypicCLI import _snapshot_metadata_csv
from phenotypic.sdk_ import metadata_csv_deliverable_path


def test_metadata_snapshot_is_byte_exact_and_reused(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(b"Metadata_Strain,condition\na,control\n")

    snapshot = _snapshot_metadata_csv(tmp_path / "output", source)
    assert snapshot == metadata_csv_deliverable_path(tmp_path / "output")
    assert snapshot.read_bytes() == source.read_bytes()
    original_mtime = snapshot.stat().st_mtime_ns

    assert _snapshot_metadata_csv(tmp_path / "output", source) == snapshot
    assert snapshot.stat().st_mtime_ns == original_mtime

    source.unlink()
    assert _snapshot_metadata_csv(tmp_path / "output", None) == snapshot


def test_invalid_metadata_never_replaces_existing_snapshot(tmp_path: Path) -> None:
    output = tmp_path / "output"
    valid = tmp_path / "valid.csv"
    valid.write_text("Metadata_Strain\na\n", encoding="utf-8")
    snapshot = _snapshot_metadata_csv(output, valid)
    before = snapshot.read_bytes()

    invalid = tmp_path / "invalid.csv"
    invalid.write_bytes(b'"unterminated')
    with pytest.raises(Exception):
        _snapshot_metadata_csv(output, invalid)

    assert snapshot.read_bytes() == before
