"""Unit tests for the post-aggregation per-feature split in
:mod:`phenotypic._cli._cli_output_manager`.

Covers the three entry points added alongside the splitter:

- :func:`_collect_feature_headers` — driven by pipeline ``_meas``.
- :func:`_load_pipeline_from_output_dir` — fallback used by SLURM sentinel
  and ``--recompile`` paths.
- :func:`split_master_by_feature` — writes per-feature CSV + Parquet.
- :func:`aggregate_measurements` — end-to-end glue that auto-loads the
  pipeline from the output directory when none is passed in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

from phenotypic import ImagePipeline
from phenotypic.measure import MeasureColor, MeasureShape, MeasureSize
from phenotypic.tools_ import (
    master_measurements_csv_path,
    master_measurements_parquet_path,
    measurements_by_feature_dir,
    measurements_csv_path,
    measurements_parquet_path,
)
from phenotypic._cli._cli_output_manager import (
    _collect_feature_headers,
    _load_pipeline_from_output_dir,
    aggregate_measurements,
    split_master_by_feature,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="CLI output manager uses POSIX atomic writes",
)


def _make_master_df(pipeline: ImagePipeline) -> pl.DataFrame:
    """Build a synthetic master DataFrame matching *pipeline*'s features."""
    headers = _collect_feature_headers(pipeline)
    cols: dict[str, list] = {
        "Metadata_Dataset": ["ds1", "ds1", "ds1"],
        "Metadata_ImageFile": ["img1", "img1", "img2"],
        "Object_Label": [1, 2, 1],
        "RowNum": [0, 0, 1],
        "ColNum": [0, 1, 0],
    }
    for feature_cols in headers.values():
        for i, hdr in enumerate(feature_cols):
            cols[hdr] = [float(i + 1), float(i + 2), float(i + 3)]
    return pl.DataFrame(cols)


class TestCollectFeatureHeaders:
    """``_collect_feature_headers`` maps ``_meas`` keys to expected columns."""

    def test_singular_infoclass(self) -> None:
        pipeline = ImagePipeline(meas=[MeasureSize()])
        headers = _collect_feature_headers(pipeline)
        assert set(headers) == {"MeasureSize"}
        assert "Size_Area" in headers["MeasureSize"]
        assert "Size_IntegratedIntensity" in headers["MeasureSize"]

    def test_plural_infoclasses_are_merged(self) -> None:
        """``MeasureColor`` exposes ``_measurement_infoclasses`` (a list)."""
        pipeline = ImagePipeline(meas=[MeasureColor(include_XYZ=True)])
        headers = _collect_feature_headers(pipeline)
        assert set(headers) == {"MeasureColor"}
        color_headers = headers["MeasureColor"]
        assert any(h.startswith("ColorLab_") for h in color_headers)
        assert any(h.startswith("ColorHSV_") for h in color_headers)

    def test_measurer_without_info_attributes_is_skipped(self) -> None:
        """Custom measurers missing both attributes do not appear."""
        pipeline = ImagePipeline(meas=[MeasureSize()])

        class _Stub:
            pass

        pipeline._meas["Custom"] = _Stub()  # type: ignore[assignment]
        headers = _collect_feature_headers(pipeline)
        assert "Custom" not in headers
        assert "MeasureSize" in headers


class TestLoadPipelineFromOutputDir:
    """``_load_pipeline_from_output_dir`` is the sentinel/recompile fallback."""

    def _stage(self, output_dir: Path, pipeline: ImagePipeline, *,
               name: str = "my_pipeline.json",
               state_override: dict | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline_json = output_dir / name
        pipeline_json.write_text(pipeline.to_json(), encoding="utf-8")
        state = {
            "pipeline_path": str(pipeline_json),
            "version": "2.0.0",
            "input_path": str(output_dir),
            "output_dir": str(output_dir),
            "timestamp": "2026-01-01T00:00:00",
            "execution_mode": "local",
            "last_updated": "2026-01-01T00:00:00",
            "datasets": {},
            "config": {},
        }
        if state_override is not None:
            state.update(state_override)
        (output_dir / "processing_state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
        return pipeline_json

    def test_returns_none_without_state_file(self, tmp_path: Path) -> None:
        assert _load_pipeline_from_output_dir(tmp_path) is None

    def test_loads_via_state_file(self, tmp_path: Path) -> None:
        pipeline = ImagePipeline(meas=[MeasureSize(), MeasureShape()])
        self._stage(tmp_path, pipeline)
        loaded = _load_pipeline_from_output_dir(tmp_path)
        assert loaded is not None
        assert set(loaded._meas.keys()) == {"MeasureSize", "MeasureShape"}

    def test_missing_pipeline_copy_returns_none(self, tmp_path: Path) -> None:
        pipeline = ImagePipeline(meas=[MeasureSize()])
        self._stage(tmp_path, pipeline)
        # Remove the copy but keep the state file referencing it.
        (tmp_path / "my_pipeline.json").unlink()
        assert _load_pipeline_from_output_dir(tmp_path) is None

    def test_missing_pipeline_path_field_returns_none(self, tmp_path: Path) -> None:
        pipeline = ImagePipeline(meas=[MeasureSize()])
        self._stage(
            tmp_path, pipeline, state_override={"pipeline_path": ""}
        )
        assert _load_pipeline_from_output_dir(tmp_path) is None


class TestSplitMasterByFeature:
    """``split_master_by_feature`` writes per-feature spreadsheets."""

    def test_writes_csv_and_parquet_per_feature(self, tmp_path: Path) -> None:
        pipeline = ImagePipeline(meas=[MeasureSize(), MeasureShape()])
        master = _make_master_df(pipeline)

        written = split_master_by_feature(master, tmp_path, pipeline)

        assert set(written) == {"MeasureSize", "MeasureShape"}
        for key in written:
            assert (measurements_by_feature_dir(tmp_path) / f"{key}.csv").exists()
            assert (measurements_by_feature_dir(tmp_path) / f"{key}.parquet").exists()

    def test_metadata_columns_preserved_in_every_split(self, tmp_path: Path) -> None:
        pipeline = ImagePipeline(meas=[MeasureSize(), MeasureShape()])
        master = _make_master_df(pipeline)
        split_master_by_feature(master, tmp_path, pipeline)

        for key in ("MeasureSize", "MeasureShape"):
            df = pl.read_csv(measurements_by_feature_dir(tmp_path) / f"{key}.csv")
            for meta_col in ("Metadata_Dataset", "Metadata_ImageFile",
                             "Object_Label", "RowNum", "ColNum"):
                assert meta_col in df.columns, (
                    f"{meta_col} missing from {key} split"
                )
            assert df.height == master.height

    def test_feature_columns_do_not_leak_between_splits(self, tmp_path: Path) -> None:
        pipeline = ImagePipeline(meas=[MeasureSize(), MeasureShape()])
        master = _make_master_df(pipeline)
        split_master_by_feature(master, tmp_path, pipeline)

        headers = _collect_feature_headers(pipeline)
        size_df = pl.read_csv(
            measurements_by_feature_dir(tmp_path) / "MeasureSize.csv"
        )
        shape_df = pl.read_csv(
            measurements_by_feature_dir(tmp_path) / "MeasureShape.csv"
        )

        for hdr in headers["MeasureSize"]:
            assert hdr in size_df.columns
            assert hdr not in shape_df.columns
        for hdr in headers["MeasureShape"]:
            assert hdr in shape_df.columns
            assert hdr not in size_df.columns

    def test_skips_features_whose_columns_are_absent(self, tmp_path: Path) -> None:
        """A measurer whose columns never made it to the master is skipped."""
        pipeline = ImagePipeline(meas=[MeasureSize(), MeasureShape()])
        master = _make_master_df(pipeline)
        # Drop every shape column to simulate a failed measurer.
        shape_cols = _collect_feature_headers(pipeline)["MeasureShape"]
        master = master.drop(shape_cols)

        written = split_master_by_feature(master, tmp_path, pipeline)
        assert written == {
            "MeasureSize": measurements_by_feature_dir(tmp_path) / "MeasureSize.csv"
        }
        assert not (measurements_by_feature_dir(tmp_path) / "MeasureShape.csv").exists()

    def test_empty_meas_returns_empty(self, tmp_path: Path) -> None:
        pipeline = ImagePipeline()  # no measurers
        master = pl.DataFrame({"Metadata_Dataset": ["ds"], "Object_Label": [1]})
        assert split_master_by_feature(master, tmp_path, pipeline) == {}
        assert not measurements_by_feature_dir(tmp_path).exists()


class TestAggregateMeasurementsAutoResolve:
    """End-to-end: ``aggregate_measurements`` picks up the pipeline from
    ``output_dir`` when called without an explicit ``pipeline`` kwarg."""

    def test_split_runs_via_state_file_fallback(self, tmp_path: Path) -> None:
        pipeline = ImagePipeline(meas=[MeasureSize(), MeasureShape()])
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        # Stage a copied pipeline JSON + processing_state.json (as the main
        # CLI would have done).
        pipeline_json = output_dir / "my_pipeline.json"
        pipeline_json.write_text(pipeline.to_json(), encoding="utf-8")
        (output_dir / "processing_state.json").write_text(
            json.dumps({
                "pipeline_path": str(pipeline_json),
                "version": "2.0.0",
                "input_path": str(output_dir),
                "output_dir": str(output_dir),
                "timestamp": "2026-01-01T00:00:00",
                "execution_mode": "local",
                "last_updated": "2026-01-01T00:00:00",
                "datasets": {},
                "config": {},
            }),
            encoding="utf-8",
        )

        # Stage a fake per-image parquet with the columns aggregate_measurements
        # expects (dataset + image + a couple of feature columns).
        ds_dir = output_dir / "results" / "ds1" / "measurements"
        ds_dir.mkdir(parents=True)
        row = pl.DataFrame({
            "Metadata_Dataset": ["ds1"],
            "Metadata_ImageFile": ["img1"],
            "Object_Label": [1],
            "RowNum": [0],
            "ColNum": [0],
            "Size_Area": [10.0],
            "Size_IntegratedIntensity": [100.0],
            "Shape_Area": [10.0],
            "Shape_Perimeter": [12.0],
        })
        row.write_parquet(ds_dir / "img1.parquet")

        master_path = aggregate_measurements(
            output_dir=output_dir,
            dataset_names=["ds1"],
            include_dataset_column=True,
        )

        assert master_path == master_measurements_csv_path(output_dir)
        assert master_path.exists()

        # The CLI also seeds an editable measurements.{csv,parquet}
        # copy that the GUI results viewer mutates in place.
        seed_csv = measurements_csv_path(output_dir)
        seed_parquet = measurements_parquet_path(output_dir)
        assert seed_csv.exists()
        assert seed_parquet.exists()

        master_df = pl.read_csv(master_path)
        master_pq = pl.read_parquet(master_measurements_parquet_path(output_dir))
        seed_df = pl.read_csv(seed_csv)
        seed_pq_df = pl.read_parquet(seed_parquet)
        # CSV round-trip: shapes + columns match (CSV always re-encodes
        # numerics; full equals on the polars CSV reader can be lossy).
        assert seed_df.shape == master_df.shape
        assert seed_df.columns == master_df.columns
        # Parquet round-trip: full row-by-row equality with the master.
        assert seed_pq_df.equals(master_pq)

        split_dir = measurements_by_feature_dir(output_dir)
        assert split_dir.is_dir()
        size_csv = split_dir / "MeasureSize.csv"
        shape_csv = split_dir / "MeasureShape.csv"
        assert size_csv.exists()
        assert shape_csv.exists()

        size_df = pl.read_csv(size_csv)
        assert "Size_Area" in size_df.columns
        assert "Shape_Area" not in size_df.columns
        assert "Metadata_ImageFile" in size_df.columns

    def test_no_state_file_keeps_master_but_skips_splits(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        ds_dir = output_dir / "results" / "ds1" / "measurements"
        ds_dir.mkdir(parents=True)
        pl.DataFrame({
            "Metadata_Dataset": ["ds1"],
            "Metadata_ImageFile": ["img1"],
            "Object_Label": [1],
            "Size_Area": [10.0],
        }).write_parquet(ds_dir / "img1.parquet")

        master_path = aggregate_measurements(
            output_dir=output_dir,
            dataset_names=["ds1"],
            include_dataset_column=True,
        )

        assert master_path is not None
        assert master_path.exists()
        assert not measurements_by_feature_dir(output_dir).exists()
