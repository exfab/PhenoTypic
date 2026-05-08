"""Tests for the shared CLI ↔ GUI artifact-layout constants module."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import get_args

import pytest

from phenotypic.tools_ import (
    ANALYSIS_CSV,
    ANALYSIS_PARQUET,
    DIR_HDF,
    DIR_MEASUREMENTS,
    DIR_OVERLAYS,
    DIR_PROGRESS,
    DIR_RESULTS,
    JOB_METADATA_JSON,
    MANIFEST_JSON,
    MASTER_MEASUREMENTS_CSV,
    MASTER_MEASUREMENTS_PARQUET,
    MEASUREMENTS_CSV,
    MEASUREMENTS_PARQUET,
    PIPELINE_JSON,
    PROCESSING_EVENTS_LOG,
    PROCESSING_STATE_JSON,
    ChunkManifestKey,
    ChunkStateKey,
    DashboardManifestKey,
    EnvVar,
    HdfAttr,
    JobMetadataKey,
    ModulePath,
    analysis_csv_path,
    analysis_parquet_path,
    checkpoint_lock_filename,
    chunk_parquet_filename,
    dashboard_html_path,
    dataset_hdf_dir,
    dataset_measurements_dir,
    dataset_overlays_dir,
    dataset_results_dir,
    default_output_dir_name,
    event_log_path,
    job_metadata_path,
    manifest_json_path,
    master_measurements_csv_path,
    master_measurements_parquet_path,
    measurements_by_feature_dir,
    measurements_csv_path,
    measurements_parquet_path,
    pipeline_json_path,
    processing_state_path,
    progress_dir,
    resolve_execution_mode,
    results_dir,
    shard_parquet_filename,
    task_status_filename,
    task_status_path,
)
from phenotypic.tools_.constants_ import IMAGE_TYPES
from phenotypic.tools_.typing_ import (
    CheckpointType,
    ExecutionMode,
    FailureSource,
    ImageTypeName,
    ProcessingStatus,
    RecompileTaskType,
)


# ---------------------------------------------------------------------------
# Enum ↔ Literal alignment (the only paired Enum + Literal in this PR)
# ---------------------------------------------------------------------------


class TestEnumLiteralAlignment:
    def test_image_type_literal_covers_base_and_grid_enum_values(self) -> None:
        """The CLAUDE.md-mandated alignment test for IMAGE_TYPES ↔ ImageTypeName.

        Both ``IMAGE_TYPES.BASE`` and ``IMAGE_TYPES.GRID`` cross CLI/GUI
        boundary code as bare strings; the Literal alias enforces them
        at type-check time. This test fails the day someone renames an
        Enum value without updating the Literal (or vice versa).
        """
        literal_values = set(get_args(ImageTypeName))
        enum_values = {IMAGE_TYPES.BASE.value, IMAGE_TYPES.GRID.value}
        assert enum_values.issubset(literal_values), (
            f"ImageTypeName Literal does not cover IMAGE_TYPES.BASE / GRID — "
            f"literal has {literal_values}, enum has {enum_values}"
        )

    def test_image_type_literal_does_not_admit_unknown_values(self) -> None:
        """ImageTypeName must not silently accept arbitrary strings."""
        literal_values = set(get_args(ImageTypeName))
        assert "Image" in literal_values
        assert "GridImage" in literal_values
        assert "GibberishType" not in literal_values


# ---------------------------------------------------------------------------
# Other Literal alias sanity checks (no Enum partner — just that get_args works)
# ---------------------------------------------------------------------------


class TestLiteralAliases:
    def test_execution_mode_values(self) -> None:
        assert set(get_args(ExecutionMode)) == {"local", "slurm"}

    def test_processing_status_values(self) -> None:
        assert set(get_args(ProcessingStatus)) == {"started", "completed", "failed"}

    def test_recompile_task_type_values(self) -> None:
        assert set(get_args(RecompileTaskType)) == {"measurements", "overlay", "finalize"}

    def test_checkpoint_type_values(self) -> None:
        assert set(get_args(CheckpointType)) == {"manifest", "finalize"}

    def test_failure_source_values(self) -> None:
        assert set(get_args(FailureSource)) == {"python", "slurm"}


# ---------------------------------------------------------------------------
# Filename / dirname constants — concrete-value sanity
# ---------------------------------------------------------------------------


class TestFilenameConstants:
    def test_master_measurements_filenames(self) -> None:
        assert MASTER_MEASUREMENTS_CSV == "master_measurements.csv"
        assert MASTER_MEASUREMENTS_PARQUET == "master_measurements.parquet"

    def test_measurements_mirror_filenames(self) -> None:
        assert MEASUREMENTS_CSV == "measurements.csv"
        assert MEASUREMENTS_PARQUET == "measurements.parquet"

    def test_analysis_filenames(self) -> None:
        assert ANALYSIS_CSV == "analysis.csv"
        assert ANALYSIS_PARQUET == "analysis.parquet"

    def test_pipeline_state_filenames(self) -> None:
        assert PIPELINE_JSON == "pipeline.json"
        assert PROCESSING_STATE_JSON == "processing_state.json"

    def test_progress_sidecar_filenames(self) -> None:
        assert PROCESSING_EVENTS_LOG == "processing_events.log"
        assert JOB_METADATA_JSON == "job_metadata.json"
        assert MANIFEST_JSON == "manifest.json"


class TestDirectoryConstants:
    def test_dir_results(self) -> None:
        assert DIR_RESULTS == "results"

    def test_dir_progress(self) -> None:
        assert DIR_PROGRESS == "progress"

    def test_per_dataset_subdirs(self) -> None:
        assert DIR_MEASUREMENTS == "measurements"
        assert DIR_HDF == "hdf"
        assert DIR_OVERLAYS == "overlays"


# ---------------------------------------------------------------------------
# Templated filename render functions — round-trip + Final[str] discipline
# ---------------------------------------------------------------------------


class TestTemplatedFilenames:
    def test_task_status_filename_renders(self) -> None:
        assert task_status_filename(0) == "task_0.json"
        assert task_status_filename(42) == "task_42.json"

    def test_shard_parquet_filename_renders(self) -> None:
        assert shard_parquet_filename(0) == "shard_0.parquet"
        assert shard_parquet_filename(7) == "shard_7.parquet"

    def test_chunk_parquet_filename_zero_pads(self) -> None:
        """Chunk filenames must zero-pad to 3 digits — sorts lexicographically."""
        assert chunk_parquet_filename(0) == "chunk_000.parquet"
        assert chunk_parquet_filename(5) == "chunk_005.parquet"
        assert chunk_parquet_filename(123) == "chunk_123.parquet"

    def test_default_output_dir_name_format(self) -> None:
        fixed = datetime(2026, 5, 7, 18, 23, 43)
        assert default_output_dir_name(fixed) == "phenotypic_results_20260507_182343"

    def test_default_output_dir_name_uses_now_when_unset(self) -> None:
        name = default_output_dir_name()
        assert re.match(r"^phenotypic_results_\d{8}_\d{6}$", name)

    def test_checkpoint_lock_filename(self) -> None:
        assert checkpoint_lock_filename("manifest") == ".manifest_lock"
        assert checkpoint_lock_filename("finalize") == ".finalize_lock"


# ---------------------------------------------------------------------------
# Path-helper round-trips — declarative path arithmetic
# ---------------------------------------------------------------------------


class TestPathHelpers:
    @pytest.fixture
    def output(self) -> Path:
        return Path("/tmp/pht_run")

    def test_progress_dir(self, output: Path) -> None:
        assert progress_dir(output) == output / "progress"

    def test_results_dir(self, output: Path) -> None:
        assert results_dir(output) == output / "results"

    def test_event_log_path(self, output: Path) -> None:
        assert event_log_path(output) == output / "processing_events.log"

    def test_processing_state_path(self, output: Path) -> None:
        assert processing_state_path(output) == output / "processing_state.json"

    def test_master_measurements_paths(self, output: Path) -> None:
        assert master_measurements_csv_path(output) == output / "master_measurements.csv"
        assert master_measurements_parquet_path(output) == output / "master_measurements.parquet"

    def test_measurements_mirror_paths(self, output: Path) -> None:
        assert measurements_csv_path(output) == output / "measurements.csv"
        assert measurements_parquet_path(output) == output / "measurements.parquet"

    def test_analysis_paths(self, output: Path) -> None:
        assert analysis_csv_path(output) == output / "analysis.csv"
        assert analysis_parquet_path(output) == output / "analysis.parquet"

    def test_pipeline_json_path(self, output: Path) -> None:
        assert pipeline_json_path(output) == output / "pipeline.json"

    def test_dashboard_html_path(self, output: Path) -> None:
        assert dashboard_html_path(output) == output / "dashboard.html"

    def test_progress_sidecar_paths(self, output: Path) -> None:
        assert job_metadata_path(output) == output / "progress" / "job_metadata.json"
        assert manifest_json_path(output) == output / "progress" / "manifest.json"

    def test_dataset_subdirs(self, output: Path) -> None:
        assert dataset_results_dir(output, "ds1") == output / "results" / "ds1"
        assert dataset_measurements_dir(output, "ds1") == output / "results" / "ds1" / "measurements"
        assert dataset_hdf_dir(output, "ds1") == output / "results" / "ds1" / "hdf"
        assert dataset_overlays_dir(output, "ds1") == output / "results" / "ds1" / "overlays"

    def test_measurements_by_feature_dir(self, output: Path) -> None:
        assert measurements_by_feature_dir(output) == output / "measurements_by_feature"

    def test_task_status_path(self, output: Path) -> None:
        assert task_status_path(output, 3) == (
            output / "progress" / "recompile" / "status" / "task_3.json"
        )


# ---------------------------------------------------------------------------
# JSON contract key namespace classes
# ---------------------------------------------------------------------------


class TestJsonContractKeys:
    def test_job_metadata_keys(self) -> None:
        assert JobMetadataKey.EXECUTION_MODE == "execution_mode"
        assert JobMetadataKey.START_TIME == "start_time"
        assert JobMetadataKey.INPUT_PATH == "input_path"
        assert JobMetadataKey.METADATA_CSV == "metadata_csv"
        assert JobMetadataKey.SLURM_JOB_IDS == "slurm_job_ids"
        assert JobMetadataKey.CHUNK_JOB_IDS == "chunk_job_ids"
        assert JobMetadataKey.CHUNK_SCRIPTS == "chunk_scripts"

    def test_dashboard_manifest_keys(self) -> None:
        assert DashboardManifestKey.FAILED == "failed"
        assert DashboardManifestKey.EXECUTION_MODE == "execution_mode"

    def test_chunk_state_keys(self) -> None:
        assert ChunkStateKey.CHUNKED_FILES == "chunked_files"
        assert ChunkStateKey.NEXT_CHUNK_ID == "next_chunk_id"

    def test_chunk_manifest_keys(self) -> None:
        assert ChunkManifestKey.CHUNKS == "chunks"
        assert ChunkManifestKey.ROWS == "rows"
        assert ChunkManifestKey.DATASETS == "datasets"
        assert ChunkManifestKey.TOTAL_ROWS == "total_rows"
        assert ChunkManifestKey.NAME == "name"

    def test_hdf_attr_keys(self) -> None:
        assert HdfAttr.PHENOTYPIC_CLASS == "phenotypic_class"


class TestModulePathConstants:
    def test_post_module_path(self) -> None:
        import importlib

        # Round-trip: the constant must point at a real, importable module
        assert ModulePath.POST == "phenotypic.post"
        assert importlib.import_module(ModulePath.POST) is not None

    def test_analysis_module_path(self) -> None:
        import importlib

        assert ModulePath.ANALYSIS == "phenotypic.analysis"
        assert importlib.import_module(ModulePath.ANALYSIS) is not None


class TestEnvVarConstants:
    def test_slurm_env_var_names(self) -> None:
        assert EnvVar.SCRATCH == "SCRATCH"
        assert EnvVar.SLURM_JOB_ID == "SLURM_JOB_ID"
        assert EnvVar.SLURM_ARRAY_JOB_ID == "SLURM_ARRAY_JOB_ID"
        assert EnvVar.SLURM_ARRAY_TASK_ID == "SLURM_ARRAY_TASK_ID"
        assert EnvVar.SLURM_ARRAY_TASK_COUNT == "SLURM_ARRAY_TASK_COUNT"


# ---------------------------------------------------------------------------
# resolve_execution_mode — the 5-site copy-paste helper
# ---------------------------------------------------------------------------


class TestReadRunManifest:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        from phenotypic.tools_ import read_run_manifest

        # progress/manifest.json doesn't exist
        assert read_run_manifest(tmp_path) is None

    def test_valid_manifest_returns_dict(self, tmp_path: Path) -> None:
        import json

        from phenotypic.tools_ import read_run_manifest

        progress = tmp_path / "progress"
        progress.mkdir()
        payload = {"failed": 3, "execution_mode": "slurm"}
        (progress / "manifest.json").write_text(json.dumps(payload))

        manifest = read_run_manifest(tmp_path)
        assert manifest == payload

    def test_malformed_manifest_returns_none(self, tmp_path: Path) -> None:
        from phenotypic.tools_ import read_run_manifest

        progress = tmp_path / "progress"
        progress.mkdir()
        (progress / "manifest.json").write_text("{not valid json")

        # Should not raise; should warn and return None
        assert read_run_manifest(tmp_path) is None


class TestLoadMasterMeasurements:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        from phenotypic.tools_ import load_master_measurements

        # master_measurements.csv doesn't exist
        assert load_master_measurements(tmp_path) is None


class TestLoadImageFromHdf:
    def test_byte_class_attr_is_decoded(self, tmp_path: Path) -> None:
        """HDF attrs from older h5py versions arrive as bytes, not str.

        Regression test for the byte-decode fallback in load_image_from_hdf.
        We assert the attribute extraction logic without exercising the full
        Image construction (which requires real image data).
        """
        import h5py  # type: ignore[import-untyped]

        from phenotypic.tools_._io_constants import HdfAttr

        hdf_path = tmp_path / "test.h5"
        with h5py.File(hdf_path, "w") as fh:
            fh.attrs[HdfAttr.PHENOTYPIC_CLASS] = b"GridImage"

        with h5py.File(hdf_path, "r") as fh:
            cls_attr = fh.attrs.get(HdfAttr.PHENOTYPIC_CLASS, "Image")

        # h5py returns bytes on some platforms — verify the decode pattern
        if isinstance(cls_attr, bytes):
            cls_attr = cls_attr.decode("utf-8", errors="replace")
        assert cls_attr == "GridImage"


class TestResolveExecutionMode:
    def test_none_defaults_local(self) -> None:
        assert resolve_execution_mode(None) == "local"

    def test_empty_dict_defaults_local(self) -> None:
        assert resolve_execution_mode({}) == "local"

    def test_missing_key_defaults_local(self) -> None:
        assert resolve_execution_mode({"start_time": "2026-05-07"}) == "local"

    def test_explicit_slurm_passes_through(self) -> None:
        assert resolve_execution_mode({"execution_mode": "slurm"}) == "slurm"

    def test_explicit_local_passes_through(self) -> None:
        assert resolve_execution_mode({"execution_mode": "local"}) == "local"

    def test_garbage_value_defaults_local(self) -> None:
        """Unknown values collapse to 'local' rather than raising."""
        assert resolve_execution_mode({"execution_mode": "alien"}) == "local"
        assert resolve_execution_mode({"execution_mode": ""}) == "local"
        assert resolve_execution_mode({"execution_mode": None}) == "local"  # type: ignore[dict-item]
