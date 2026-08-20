"""Unit tests for the post-aggregation per-feature split in
:mod:`phenotypic._cli._cli_output_manager`.

Covers the three entry points added alongside the splitter:

- :func:`_collect_feature_headers` — driven by pipeline ``_meas``.
- :func:`_load_pipeline_from_output_dir` — fallback used by SLURM sentinel
  and recompile-mode paths.
- :func:`split_master_by_feature` — writes per-feature CSV + Parquet.
- :func:`aggregate_measurements` — end-to-end glue that auto-loads the
  pipeline from the output directory when none is passed in.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from phenotypic import ImagePipeline
from phenotypic.measure import MeasureColor, MeasureShape, MeasureSize
from phenotypic.sdk_ import (
    master_measurements_csv_path,
    master_measurements_parquet_path,
    measurements_by_feature_dir,
    measurements_csv_path,
    measurements_parquet_path,
    qc_review_state_path,
)
from phenotypic.sdk_._file_locking import exclusive_path_lock
from phenotypic._cli._cli_output_manager import (
    OutputManager,
    _image_metadata_from_mirror,
    _collect_feature_headers,
    _load_pipeline_from_output_dir,
    _reset_qc_review_state,
    aggregate_measurements,
    finalize_post_master_outputs,
    split_master_by_feature,
)
from phenotypic.schema import EXPERIMENT, IMAGE

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="CLI output manager uses POSIX atomic writes",
)


def test_review_state_reset_serializes_with_gui_writer(tmp_path: Path) -> None:
    """CLI reset waits for the same interprocess boundary as GUI saves."""
    output_dir = tmp_path / "out"
    state_path = qc_review_state_path(output_dir)
    state_path.parent.mkdir(parents=True)
    state_path.write_text('{"reviewed": true}', encoding="utf-8")
    lock_path = state_path.with_name(f".{state_path.name}.lock")
    lock_held = threading.Event()
    release_lock = threading.Event()

    def _hold_gui_writer_lock() -> None:
        with exclusive_path_lock(lock_path):
            lock_held.set()
            assert release_lock.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(_hold_gui_writer_lock)
        assert lock_held.wait(timeout=5)
        reset = pool.submit(_reset_qc_review_state, output_dir)
        with pytest.raises(TimeoutError):
            reset.result(timeout=0.1)
        assert state_path.exists()
        release_lock.set()
        writer.result(timeout=5)
        reset.result(timeout=5)

    assert not state_path.exists()


def _make_master_df(pipeline: ImagePipeline) -> pl.DataFrame:
    """Build a synthetic master DataFrame matching *pipeline*'s features."""
    headers = _collect_feature_headers(pipeline)
    cols: dict[str, list] = {
        "Metadata_Dataset": ["ds1", "ds1", "ds1"],
        str(IMAGE.IMAGE_NAME): ["img1", "img1", "img2"],
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

    def _stage(
        self,
        output_dir: Path,
        pipeline: ImagePipeline,
        *,
        name: str = "my_pipeline.json",
        state_override: dict | None = None,
    ) -> Path:
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

    def test_missing_pipeline_path_field_returns_none(
        self, tmp_path: Path
    ) -> None:
        pipeline = ImagePipeline(meas=[MeasureSize()])
        self._stage(tmp_path, pipeline, state_override={"pipeline_path": ""})
        assert _load_pipeline_from_output_dir(tmp_path) is None


class TestSplitMasterByFeature:
    """``split_master_by_feature`` writes per-feature spreadsheets."""

    def test_writes_csv_and_parquet_per_feature(self, tmp_path: Path) -> None:
        pipeline = ImagePipeline(meas=[MeasureSize(), MeasureShape()])
        master = _make_master_df(pipeline)

        written = split_master_by_feature(master, tmp_path, pipeline)

        assert set(written) == {"MeasureSize", "MeasureShape"}
        for key in written:
            assert (
                measurements_by_feature_dir(tmp_path) / f"{key}.csv"
            ).exists()
            assert (
                measurements_by_feature_dir(tmp_path) / f"{key}.parquet"
            ).exists()

    def test_metadata_columns_preserved_in_every_split(
        self, tmp_path: Path
    ) -> None:
        pipeline = ImagePipeline(meas=[MeasureSize(), MeasureShape()])
        master = _make_master_df(pipeline)
        split_master_by_feature(master, tmp_path, pipeline)

        for key in ("MeasureSize", "MeasureShape"):
            df = pl.read_csv(
                measurements_by_feature_dir(tmp_path) / f"{key}.csv"
            )
            for meta_col in (
                str(EXPERIMENT.DATASET),
                str(IMAGE.IMAGE_NAME),
                "Object_Label",
                "RowNum",
                "ColNum",
            ):
                assert meta_col in df.columns, (
                    f"{meta_col} missing from {key} split"
                )
            assert df.height == master.height

    def test_feature_columns_do_not_leak_between_splits(
        self, tmp_path: Path
    ) -> None:
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

    def test_skips_features_whose_columns_are_absent(
        self, tmp_path: Path
    ) -> None:
        """A measurer whose columns never made it to the master is skipped."""
        pipeline = ImagePipeline(meas=[MeasureSize(), MeasureShape()])
        master = _make_master_df(pipeline)
        # Drop every shape column to simulate a failed measurer.
        shape_cols = _collect_feature_headers(pipeline)["MeasureShape"]
        master = master.drop(shape_cols)

        written = split_master_by_feature(master, tmp_path, pipeline)
        assert written == {
            "MeasureSize": measurements_by_feature_dir(tmp_path)
            / "MeasureSize.csv"
        }
        assert not (
            measurements_by_feature_dir(tmp_path) / "MeasureShape.csv"
        ).exists()

    def test_empty_meas_returns_empty(self, tmp_path: Path) -> None:
        pipeline = ImagePipeline()  # no measurers
        master = pl.DataFrame(
            {"Metadata_Dataset": ["ds"], "Object_Label": [1]}
        )
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
            json.dumps(
                {
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
            ),
            encoding="utf-8",
        )

        # Stage a fake per-image parquet with the columns aggregate_measurements
        # expects (dataset + image + a couple of feature columns).
        ds_dir = output_dir / "results" / "ds1" / "measurements"
        ds_dir.mkdir(parents=True)
        row = pl.DataFrame(
            {
                "Metadata_Dataset": ["ds1"],
                str(IMAGE.IMAGE_NAME): ["img1"],
                "Object_Label": [1],
                "RowNum": [0],
                "ColNum": [0],
                "Size_Area": [10.0],
                "Size_IntegratedIntensity": [100.0],
                "Shape_Area": [10.0],
                "Shape_Perimeter": [12.0],
            }
        )
        row.write_parquet(ds_dir / "img1.parquet")

        # Staged finalization calls aggregate_measurements without a pipeline.
        # Recovering the copied config must still reach the configured plotting
        # lifecycle rather than silently stopping after the mirror/splits.
        with patch(
            "phenotypic.plotting._pipeline.PlotCoordinator.emit_measurements"
        ) as mock_emit_measurements:
            master_path = aggregate_measurements(
                output_dir=output_dir,
                dataset_names=["ds1"],
                include_dataset_column=True,
            )

        mock_emit_measurements.assert_called_once()

        assert master_path == master_measurements_csv_path(output_dir)
        assert master_path.exists()

        # The CLI also seeds an editable measurements.{csv,parquet}
        # copy that the GUI results viewer mutates in place.
        seed_csv = measurements_csv_path(output_dir)
        seed_parquet = measurements_parquet_path(output_dir)
        assert seed_csv.exists()
        assert seed_parquet.exists()

        master_df = pl.read_csv(master_path)
        master_pq = pl.read_parquet(
            master_measurements_parquet_path(output_dir)
        )
        seed_df = pl.read_csv(seed_csv)
        seed_pq_df = pl.read_parquet(seed_parquet)
        # CSV round-trip: shapes + columns match (CSV always re-encodes
        # numerics; full equals on the polars CSV reader can be lossy).
        assert seed_df.shape == master_df.shape
        # The mirror is cluster-ordered (framework MetadataImage_ trails the
        # measurements); the master archive keeps its raw order. Same columns,
        # different order.
        assert set(seed_df.columns) == set(master_df.columns)
        assert seed_df.columns.index(
            str(IMAGE.IMAGE_NAME)
        ) > seed_df.columns.index("Shape_Area")
        # Parquet round-trip: full row-by-row equality with the master after
        # aligning to the master's column order (polars .equals() is
        # column-order sensitive).
        assert seed_pq_df.select(master_pq.columns).equals(master_pq)

        split_dir = measurements_by_feature_dir(output_dir)
        assert split_dir.is_dir()
        size_csv = split_dir / "MeasureSize.csv"
        shape_csv = split_dir / "MeasureShape.csv"
        assert size_csv.exists()
        assert shape_csv.exists()

        size_df = pl.read_csv(size_csv)
        assert "Size_Area" in size_df.columns
        assert "Shape_Area" not in size_df.columns
        assert str(IMAGE.IMAGE_NAME) in size_df.columns

    def test_no_state_file_keeps_master_and_splits_known_columns(
        self, tmp_path: Path
    ) -> None:
        """Dynamic splits no longer require a recovered pipeline."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        ds_dir = output_dir / "results" / "ds1" / "measurements"
        ds_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "Metadata_Dataset": ["ds1"],
                str(IMAGE.IMAGE_NAME): ["img1"],
                "Object_Label": [1],
                "Size_Area": [10.0],
            }
        ).write_parquet(ds_dir / "img1.parquet")

        master_path = aggregate_measurements(
            output_dir=output_dir,
            dataset_names=["ds1"],
            include_dataset_column=True,
        )

        assert master_path is not None
        assert master_path.exists()
        split_dir = measurements_by_feature_dir(output_dir)
        assert split_dir.is_dir()
        size_csv = split_dir / "MeasureSize.csv"
        assert size_csv.exists()
        size_df = pl.read_csv(size_csv)
        assert size_df.columns == [
            str(EXPERIMENT.DATASET),
            str(IMAGE.IMAGE_NAME),
            "Object_Label",
            "Size_Area",
        ]


class TestFinalizeReemitsErrorDeliverables:
    """``finalize_post_master_outputs`` re-emits errors/* + error_analysis.* from
    a durable labels store, keyed off the clean master, while still resetting
    ``review_state.json`` and never wiping the labels parquet (Phase 5)."""

    @staticmethod
    def _master_df() -> pl.DataFrame:
        """Clean master: 30 small + 10 clearly-larger ``Size_Area`` objects."""
        import numpy as np

        rng = np.random.default_rng(0)
        n_good, n_err = 30, 10
        n = n_good + n_err
        labels = list(range(1, n + 1))
        area = np.concatenate(
            [rng.normal(100.0, 5.0, n_good), rng.normal(500.0, 5.0, n_err)]
        )
        return pl.DataFrame(
            {
                "Metadata_Dataset": ["ds1"] * n,
                str(IMAGE.IMAGE_NAME): ["plateA"] * n,
                "Object_Label": labels,
                "Bbox_CenterRR": [10.0 * i for i in labels],
                "Bbox_CenterCC": [20.0 * i for i in labels],
                "Size_Area": area.tolist(),
                "Shape_Circularity": rng.normal(0.8, 0.05, n).tolist(),
            }
        )

    @staticmethod
    def _stage_labels(
        output_dir: Path,
        master: pl.DataFrame,
        err_labels: list[int],
        category: str,
    ) -> None:
        import phenotypic.sdk_ as tools_

        rows = master.filter(pl.col("Object_Label").is_in(err_labels))
        labels = pl.DataFrame(
            {
                str(IMAGE.IMAGE_NAME): rows.get_column(
                    str(IMAGE.IMAGE_NAME)
                ).to_list(),
                "Object_Label": rows.get_column("Object_Label").to_list(),
                "Curation_Category": [category] * rows.height,
                "Bbox_CenterRR": rows.get_column("Bbox_CenterRR").to_list(),
                "Bbox_CenterCC": rows.get_column("Bbox_CenterCC").to_list(),
            },
            schema={
                str(IMAGE.IMAGE_NAME): pl.String,
                "Object_Label": pl.Int64,
                "Curation_Category": pl.String,
                "Bbox_CenterRR": pl.Float64,
                "Bbox_CenterCC": pl.Float64,
            },
        )
        path = tools_.curation_labels_parquet_path(output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        labels.write_parquet(path)

    def test_finalize_emits_errors_and_preserves_labels(
        self, tmp_path: Path
    ) -> None:
        import phenotypic.sdk_ as tools_
        from phenotypic import ImagePipeline

        output_dir = tmp_path / "out"
        output_dir.mkdir()
        master = self._master_df()
        self._stage_labels(output_dir, master, list(range(31, 41)), "debris")

        # Seed a pre-existing review_state.json so we can assert finalize resets it.
        review_state = tools_.qc_review_state_path(output_dir)
        review_state.parent.mkdir(parents=True, exist_ok=True)
        review_state.write_text("{}", encoding="utf-8")

        finalize_post_master_outputs(
            output_dir, master, ImagePipeline(), no_qc=True
        )

        # Error deliverables emitted from the labels store.
        debris = pl.read_parquet(
            tools_.error_category_parquet_path(output_dir, "debris")
        )
        assert debris.height == 10
        ea = pl.read_parquet(tools_.error_analysis_parquet_path(output_dir))
        assert ea.columns[0] == "category"
        assert "debris" in set(ea.get_column("category").to_list())

        # review_state.json was reset (existing finalize behavior).
        assert not review_state.exists()
        # Durable labels store survived (no wipe).
        assert tools_.curation_labels_parquet_path(output_dir).exists()
        # verified.parquet never written headlessly.
        assert not tools_.verified_parquet_path(output_dir).exists()

    def test_finalize_no_labels_store_is_a_clean_no_op(
        self, tmp_path: Path
    ) -> None:
        import phenotypic.sdk_ as tools_
        from phenotypic import ImagePipeline

        output_dir = tmp_path / "out"
        output_dir.mkdir()
        master = self._master_df()

        finalize_post_master_outputs(
            output_dir, master, ImagePipeline(), no_qc=True
        )

        # No labels store → no error deliverables, but the mirror seed is written.
        assert not tools_.errors_dir(output_dir).exists()
        assert not tools_.error_analysis_parquet_path(output_dir).exists()
        assert tools_.measurements_parquet_path(output_dir).exists()


class TestFinalizeDoesNotCopyMetadataCsv:
    """Finalization consumes only the already-snapshotted metadata path."""

    @staticmethod
    def _master_df() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Metadata_Dataset": ["ds1", "ds1"],
                str(IMAGE.IMAGE_NAME): ["plateA", "plateA"],
                "Object_Label": [1, 2],
                "Size_Area": [100.0, 110.0],
            }
        )

    def test_does_not_copy_metadata_during_finalization(
        self, tmp_path: Path
    ) -> None:
        import phenotypic.sdk_ as tools_

        output_dir = tmp_path / "out"
        output_dir.mkdir()
        # A real metadata CSV whose key column matches the master so the inner
        # join succeeds (the copy must not depend on join success, but this
        # mirrors a normal run).
        source = tmp_path / "meta.csv"
        # Non-ASCII strain name (accented) so the byte-level read_bytes()
        # comparison also catches any accidental text-mode re-encode in a
        # future refactor (e.g. a copy that round-trips through str / a
        # platform-default codec). A byte-for-byte copy preserves the UTF-8 cell.
        source.write_text(
            str(IMAGE.IMAGE_NAME) + ",Metadata_Strain\nplateA,Säccharomyces\n",
            encoding="utf-8",
        )

        finalize_post_master_outputs(
            output_dir,
            self._master_df(),
            ImagePipeline(),
            metadata_csv=source,
            no_qc=True,
        )

        assert not tools_.metadata_csv_deliverable_path(output_dir).exists()

    def test_legacy_snapshot_stays_byte_exact_while_outputs_are_canonical(
        self, tmp_path: Path
    ) -> None:
        import phenotypic.sdk_ as tools_

        output_dir = tmp_path / "out"
        source = tools_.metadata_csv_deliverable_path(output_dir)
        source.parent.mkdir(parents=True)
        original = (
            b"MetadataImage_ImageName,MetadataGenetic_Strain\nplateA,S288C\n"
        )
        source.write_bytes(original)

        finalized = finalize_post_master_outputs(
            output_dir,
            self._master_df(),
            ImagePipeline(),
            metadata_csv=source,
            no_qc=True,
        )

        assert source.read_bytes() == original
        assert "Metadata_Strain" in finalized.columns
        assert "MetadataGenetic_Strain" not in finalized.columns

    def test_no_metadata_csv_means_no_copy(self, tmp_path: Path) -> None:
        import phenotypic.sdk_ as tools_

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        finalize_post_master_outputs(
            output_dir,
            self._master_df(),
            ImagePipeline(),
            metadata_csv=None,
            no_qc=True,
        )

        assert not tools_.metadata_csv_deliverable_path(output_dir).exists()
        # finalize still produced the mirror.
        assert tools_.measurements_parquet_path(output_dir).exists()

    def test_missing_source_does_not_publish_metadata(
        self, tmp_path: Path
    ) -> None:
        import phenotypic.sdk_ as tools_

        output_dir = tmp_path / "out"
        output_dir.mkdir()
        missing = tmp_path / "does_not_exist.csv"

        # The metadata snapshot is a startup responsibility. Finalization may
        # tolerate a missing legacy path, but it must never manufacture a copy.
        finalize_post_master_outputs(
            output_dir,
            self._master_df(),
            ImagePipeline(),
            metadata_csv=missing,
            no_qc=True,
        )

        assert not tools_.metadata_csv_deliverable_path(output_dir).exists()
        assert tools_.measurements_parquet_path(output_dir).exists()


class TestRembiImageMetadataExcludesPhantoms:
    """REMBI must not invent an image for a strain that was never captured.

    ``--metadata`` is a left join, so the mirror carries a row for every CSV key
    including strains that matched no measured object. REMBI is a publication
    manifest: folding a phantom into ``image_data`` fabricates a record for an
    image that does not exist.
    """

    @staticmethod
    def _mirror(image_names, phantom_flags):
        return pl.DataFrame(
            {
                str(IMAGE.IMAGE_NAME): image_names,
                "QC_MetadataOnly": phantom_flags,
            }
        )

    def test_phantom_with_a_real_image_name_is_excluded(self):
        """The documented per-image join: the phantom KEEPS the image name.

        This is the case a null-name filter cannot catch — ``plateZ`` is the
        join key, so it is emphatically not null. Without the flag filter REMBI
        reported ``n_images: 3`` for two captured plates.
        """
        mirror = self._mirror(
            ["plateA", "plateB", "plateZ"], [False, False, True]
        )

        rows = _image_metadata_from_mirror(mirror)

        assert sorted(r["ImageName"] for r in rows) == ["plateA", "plateB"]

    def test_phantom_with_a_null_image_name_is_excluded(self):
        """The per-colony join (on Grid_RowNum/ColNum): the name IS null."""
        mirror = self._mirror(["plateA", None], [False, True])

        rows = _image_metadata_from_mirror(mirror)

        assert [r["ImageName"] for r in rows] == ["plateA"]

    def test_no_op_without_a_flag_column(self):
        """A run without --metadata has no flag; every image must survive."""
        mirror = pl.DataFrame({str(IMAGE.IMAGE_NAME): ["plateA", "plateB"]})

        rows = _image_metadata_from_mirror(mirror)

        assert sorted(r["ImageName"] for r in rows) == ["plateA", "plateB"]


# ---------------------------------------------------------------------------
# save_image_store — the OME-Zarr replacement for save_image_hdf
# ---------------------------------------------------------------------------


def _make_manager(tmp_path: Path) -> OutputManager:
    """Same construction the real save_image_hdf tests use.

    See tests/integration/cli/test_staged_gpu_local.py:114.
    """
    return OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)


def _synth_image():
    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate

    return Image(load_synth_yeast_plate())


def test_save_image_store_writes_under_results_dataset_zarr(tmp_path) -> None:
    from phenotypic.sdk_ import zarr_store_path

    manager = _make_manager(tmp_path)
    saved = manager.save_image_store(_synth_image(), "ds", "img")
    assert saved == zarr_store_path(tmp_path, "ds", "img")
    assert saved.is_dir()


def test_save_image_store_round_trips_the_pixels_it_was_given(tmp_path) -> None:
    """A store at the right path holding the wrong image is still a bug."""
    import numpy as np

    from phenotypic import Image

    source = _synth_image()
    manager = _make_manager(tmp_path)
    saved = manager.save_image_store(source, "ds", "img")

    reloaded = Image.load_zarr(saved)
    np.testing.assert_array_equal(reloaded.rgb[:], source.rgb[:])
    np.testing.assert_allclose(reloaded.gray[:], source.gray[:], rtol=0, atol=0)


def test_save_image_store_writes_work_id_at_write_time(tmp_path) -> None:
    """The root zarr.json is written last, so a post-hoc patch is impossible."""
    from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes

    manager = _make_manager(tmp_path)
    saved = manager.save_image_store(_synth_image(), "ds", "img", work_id="w-7")
    assert read_phenotypic_attributes(saved)[PhenotypicAttr.WORK_ID] == "w-7"


def test_save_image_store_forwards_durable_to_the_writer(tmp_path, monkeypatch) -> None:
    """`durable` is the SLURM fsync lever; dropping it silently loses durability."""
    from phenotypic import Image

    seen: dict = {}

    def _fake_save2zarr(self, path, *, work_id=None, durable=None):
        seen["path"] = Path(path)
        seen["work_id"] = work_id
        seen["durable"] = durable
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)

    monkeypatch.setattr(Image, "save2zarr", _fake_save2zarr)
    manager = _make_manager(tmp_path)
    manager.save_image_store(
        _synth_image(), "ds", "img", work_id="w-9", durable=True
    )
    assert seen["durable"] is True
    assert seen["work_id"] == "w-9"

    seen.clear()
    manager.save_image_store(_synth_image(), "ds", "img", durable=False)
    assert seen["durable"] is False
    assert seen["work_id"] is None


def test_save_image_store_returns_none_and_logs_on_failure(
    tmp_path, monkeypatch, caplog
) -> None:
    """Preserves save_image_hdf's contract: the workers raise, not the manager."""
    from phenotypic import Image

    manager = _make_manager(tmp_path)
    monkeypatch.setattr(
        Image, "save2zarr", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    with caplog.at_level(logging.WARNING):
        assert manager.save_image_store(_synth_image(), "ds", "img") is None
    messages = [record.getMessage() for record in caplog.records]
    assert any("Failed to save" in message for message in messages)
    # The identifying detail, not just the prefix: which image, and why.
    assert any(
        "ds" in message and "img" in message and "disk full" in message
        for message in messages
    )


def test_save_image_store_cleans_up_the_part_directory_on_failure(
    tmp_path, monkeypatch
) -> None:
    """The failing writer leaves a .part behind; the manager must remove it.

    The writer stub deliberately CREATES a .part before raising. A stub that
    only raises leaves nothing to clean up, so the assertion would hold with
    the cleanup call deleted -- a test that can never fail.
    """
    from phenotypic import Image
    from phenotypic.sdk_ import dataset_zarr_dir
    from phenotypic.sdk_.ngff_ import new_part_path

    def _fail_after_making_a_part(self, path, *, work_id=None, durable=None):
        part = new_part_path(Path(path))
        (part / "OME").mkdir(parents=True)
        raise OSError("boom")

    monkeypatch.setattr(Image, "save2zarr", _fail_after_making_a_part)
    manager = _make_manager(tmp_path)
    manager.save_image_store(_synth_image(), "ds", "img")
    leftovers = list(dataset_zarr_dir(tmp_path, "ds").glob("*.part"))
    assert leftovers == []


def test_save_image_store_leaves_a_sibling_stores_part_alone(
    tmp_path, monkeypatch
) -> None:
    """Cleanup is scoped to one store; another writer's .part is not ours."""
    from phenotypic import Image
    from phenotypic.sdk_ import dataset_zarr_dir, zarr_store_path
    from phenotypic.sdk_.ngff_ import new_part_path

    other = new_part_path(zarr_store_path(tmp_path, "ds", "other"))
    other.mkdir(parents=True)

    def _fail_after_making_a_part(self, path, *, work_id=None, durable=None):
        new_part_path(Path(path)).mkdir(parents=True)
        raise OSError("boom")

    monkeypatch.setattr(Image, "save2zarr", _fail_after_making_a_part)
    manager = _make_manager(tmp_path)
    manager.save_image_store(_synth_image(), "ds", "img")
    leftovers = sorted(p.name for p in dataset_zarr_dir(tmp_path, "ds").glob("*.part"))
    assert leftovers == [other.name]


def test_save_image_store_result_passes_valid_staged_store(tmp_path) -> None:
    from phenotypic.sdk_.ngff_ import valid_staged_store

    manager = _make_manager(tmp_path)
    saved = manager.save_image_store(_synth_image(), "ds", "img")
    assert valid_staged_store(saved) is True
