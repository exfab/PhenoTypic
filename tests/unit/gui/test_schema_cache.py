"""Unit tests for ``phenotypic.gui._schema_cache.MeasurementSchema``."""
from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import pytest

from phenotypic.gui._schema_cache import MeasurementSchema
from phenotypic.sdk_ import deliverables_dir


@pytest.fixture()
def output_root(tmp_path: Path) -> Path:
    return tmp_path


def _deliv(output_root: Path) -> Path:
    """Return ``<output_root>/deliverables/``, creating it on first use.

    ``MeasurementSchema`` reads ``measurements.{parquet,csv}`` and
    ``master_measurements.parquet`` from the deliverables subdir, so tests
    must seed those files there (not at the output root).
    """
    deliverables = deliverables_dir(output_root)
    deliverables.mkdir(parents=True, exist_ok=True)
    return deliverables


class TestColumnsForParquet:
    def test_reads_parquet_when_present(self, output_root):
        df = pl.DataFrame(
            {
                "Metadata_Strain": ["a", "b"],
                "Shape_Area": [1.0, 2.0],
                "Metadata_Time": [0, 1],
            }
        )
        df.write_parquet(_deliv(output_root) / "measurements.parquet")

        schema = MeasurementSchema(output_root=output_root)
        assert schema.columns_for("measurements") == [
            "Metadata_Strain",
            "Shape_Area",
            "Metadata_Time",
        ]

    def test_caches_repeated_calls(self, output_root):
        pl.DataFrame({"a": [1], "b": [2]}).write_parquet(
            _deliv(output_root) / "measurements.parquet"
        )
        schema = MeasurementSchema(output_root=output_root)
        first = schema.columns_for("measurements")
        second = schema.columns_for("measurements")
        # Same identity — cache reused.
        assert first is second


class TestCsvFallback:
    def test_falls_back_to_csv_when_no_parquet(self, output_root):
        pl.DataFrame({"x": [1], "y": [2]}).write_csv(
            _deliv(output_root) / "measurements.csv"
        )
        schema = MeasurementSchema(output_root=output_root)
        assert schema.columns_for("measurements") == ["x", "y"]


class TestMissingFile:
    def test_returns_empty_list_when_neither_present(self, output_root):
        schema = MeasurementSchema(output_root=output_root)
        assert schema.columns_for("measurements") == []
        assert schema.columns_for("master_measurements") == []


class TestMtimeInvalidation:
    def test_invalidates_when_mtime_advances(self, output_root):
        path = _deliv(output_root) / "measurements.parquet"
        pl.DataFrame({"a": [1]}).write_parquet(path)
        schema = MeasurementSchema(output_root=output_root)
        assert schema.columns_for("measurements") == ["a"]

        # Sleep enough to guarantee a distinct mtime_ns then rewrite with a
        # different schema.
        time.sleep(0.01)
        pl.DataFrame({"b": [2], "c": [3]}).write_parquet(path)
        assert schema.columns_for("measurements") == ["b", "c"]


class TestMasterMeasurements:
    def test_master_measurements_uses_master_filename(self, output_root):
        # measurements.parquet must NOT shadow master.
        pl.DataFrame({"only_in_master": [1]}).write_parquet(
            _deliv(output_root) / "master_measurements.parquet"
        )
        pl.DataFrame({"only_in_curated": [1]}).write_parquet(
            _deliv(output_root) / "measurements.parquet"
        )
        schema = MeasurementSchema(output_root=output_root)
        assert schema.columns_for("master_measurements") == ["only_in_master"]
        assert schema.columns_for("measurements") == ["only_in_curated"]


class TestUnknownSource:
    def test_unknown_source_returns_empty(self, output_root):
        schema = MeasurementSchema(output_root=output_root)
        assert schema.columns_for("not_a_real_source") == []  # type: ignore[arg-type]


class TestRaceConditionRecovery:
    """File deleted between mtime probe and read still resolves cleanly."""

    def test_parquet_deleted_after_mtime_probe(
        self, output_root, monkeypatch
    ):
        path = _deliv(output_root) / "measurements.parquet"
        pl.DataFrame({"a": [1]}).write_parquet(path)

        schema = MeasurementSchema(output_root=output_root)

        # Inject a fault: between the mtime probe (in columns_for) and
        # _read_columns running, simulate the file being deleted by
        # patching scan_parquet to raise FileNotFoundError exactly once.
        original_scan = pl.scan_parquet
        calls = {"n": 0}

        def fake_scan(p, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise FileNotFoundError(p)
            return original_scan(p, *args, **kwargs)

        monkeypatch.setattr(pl, "scan_parquet", fake_scan)

        # No CSV mirror; falls through to []. Importantly, the call
        # does NOT raise.
        assert schema.columns_for("measurements") == []

    def test_csv_fallback_when_parquet_scan_raises(
        self, output_root, monkeypatch
    ):
        # parquet exists but raises on scan; CSV exists and succeeds.
        pl.DataFrame({"a": [1]}).write_parquet(
            _deliv(output_root) / "measurements.parquet"
        )
        pl.DataFrame({"x": [1], "y": [2]}).write_csv(
            _deliv(output_root) / "measurements.csv"
        )

        original_scan = pl.scan_parquet

        def fake_scan(*args, **kwargs):
            raise RuntimeError("simulated parquet failure")

        monkeypatch.setattr(pl, "scan_parquet", fake_scan)
        schema = MeasurementSchema(output_root=output_root)
        assert schema.columns_for("measurements") == ["x", "y"]
        # Restore so other tests aren't affected by the fixture's monkeypatch
        # going out of scope without explicit reset.
        monkeypatch.setattr(pl, "scan_parquet", original_scan)
