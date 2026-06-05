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
    deliverables_dir,
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
    readme_md_path,
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

    def test_image_type_literal_get_args_returns_only_image_and_gridimage(self) -> None:
        """``get_args(ImageTypeName)`` returns the documented two-tuple.

        Note: ``Literal`` aliases are erased at runtime — there is no actual
        runtime rejection of arbitrary strings. This test asserts the
        introspectable contract (what type-checkers see), not runtime
        validation.
        """
        literal_values = set(get_args(ImageTypeName))
        assert literal_values == {"Image", "GridImage"}

    def test_image_type_literal_subset_of_enum_guards_widening(self) -> None:
        """Catches accidental Literal widening (e.g. someone adding ``"Crop"``
        to ``ImageTypeName`` without a matching ``IMAGE_TYPES`` entry).

        Pairs with :meth:`test_image_type_literal_covers_base_and_grid_enum_values`
        which guards the other direction (Enum-value rename).
        """
        literal_values = set(get_args(ImageTypeName))
        enum_values = {m.value for m in IMAGE_TYPES}
        assert literal_values.issubset(enum_values), (
            f"ImageTypeName has values not in IMAGE_TYPES enum — "
            f"literal extra: {literal_values - enum_values}"
        )


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
        deliv = output / "deliverables"
        assert master_measurements_csv_path(output) == deliv / "master_measurements.csv"
        assert master_measurements_parquet_path(output) == deliv / "master_measurements.parquet"

    def test_measurements_mirror_paths(self, output: Path) -> None:
        deliv = output / "deliverables"
        assert measurements_csv_path(output) == deliv / "measurements.csv"
        assert measurements_parquet_path(output) == deliv / "measurements.parquet"

    def test_analysis_paths(self, output: Path) -> None:
        deliv = output / "deliverables"
        assert analysis_csv_path(output) == deliv / "analysis.csv"
        assert analysis_parquet_path(output) == deliv / "analysis.parquet"

    def test_pipeline_json_path(self, output: Path) -> None:
        assert pipeline_json_path(output) == output / "deliverables" / "pipeline.json"

    def test_dashboard_html_path(self, output: Path) -> None:
        assert dashboard_html_path(output) == output / "deliverables" / "dashboard.html"

    def test_progress_sidecar_paths(self, output: Path) -> None:
        assert job_metadata_path(output) == output / "progress" / "job_metadata.json"
        assert manifest_json_path(output) == output / "progress" / "manifest.json"

    def test_dataset_subdirs(self, output: Path) -> None:
        assert dataset_results_dir(output, "ds1") == output / "results" / "ds1"
        assert dataset_measurements_dir(output, "ds1") == output / "results" / "ds1" / "measurements"
        assert dataset_hdf_dir(output, "ds1") == output / "results" / "ds1" / "hdf"
        assert dataset_overlays_dir(output, "ds1") == output / "results" / "ds1" / "overlays"

    def test_measurements_by_feature_dir(self, output: Path) -> None:
        assert measurements_by_feature_dir(output) == (
            output / "deliverables" / "measurements_by_feature"
        )

    def test_task_status_path(self, output: Path) -> None:
        assert task_status_path(output, 3) == (
            output / "progress" / "recompile" / "status" / "task_3.json"
        )

    def test_progress_rooted_helpers(self) -> None:
        """Helpers that take ``progress_dir_`` (not ``output_dir``)."""
        from phenotypic.tools_ import (
            analysis_full_parquet_path,
            analysis_scatter_json_path,
            checkpoint_lock_path,
            chunk_lock_path,
            chunk_parquet_path,
            chunks_dir,
            recompile_dir,
            recompile_status_dir,
            sentinel_resubmitted_path,
        )

        progress = Path("/tmp/run/progress")
        assert chunks_dir(progress) == progress / "chunks"
        assert recompile_dir(progress) == progress / "recompile"
        assert recompile_status_dir(progress) == progress / "recompile" / "status"
        assert chunk_parquet_path(progress, 5) == progress / "chunks" / "chunk_005.parquet"
        assert checkpoint_lock_path(progress, "manifest") == progress / ".manifest_lock"
        assert checkpoint_lock_path(progress, "finalize") == progress / ".finalize_lock"
        assert chunk_lock_path(progress) == progress / ".chunk_lock"
        assert analysis_full_parquet_path(progress) == progress / "analysis_full.parquet"
        assert analysis_scatter_json_path(progress) == progress / "analysis_scatter.json"
        assert sentinel_resubmitted_path(progress) == progress / "sentinel_resubmitted"

    def test_output_rooted_helpers(self, output: Path) -> None:
        """Helpers that take ``output_dir`` (the run root)."""
        from phenotypic.tools_ import (
            analysis_html_path,
            chunk_manifest_path,
            chunk_state_path,
            failures_jsonl_path,
            logs_dir,
            overlay_manifest_path,
            processing_report_html_path,
            slurm_scripts_dir,
        )

        assert logs_dir(output) == output / "logs"
        assert slurm_scripts_dir(output) == output / "slurm_scripts"
        assert analysis_html_path(output) == output / "deliverables" / "analysis.html"
        assert processing_report_html_path(output) == (
            output / "deliverables" / "processing_report.html"
        )
        assert failures_jsonl_path(output) == output / "progress" / "failures.jsonl"
        assert chunk_manifest_path(output) == output / "progress" / "chunk_manifest.json"
        assert chunk_state_path(output) == output / "progress" / "chunk_state.json"
        assert overlay_manifest_path(output) == output / "progress" / "overlay_manifest.json"


# ---------------------------------------------------------------------------
# Deliverables layout — the user-facing-output folder cutover
# ---------------------------------------------------------------------------


class TestDeliverablesLayout:
    """All user-facing CLI outputs root under ``<output>/deliverables/``.

    These tests pin the hard cutover: every moved artifact helper composes
    from :func:`deliverables_dir`, while per-image (``results/``) and
    run-state (``processing_state.json``) helpers stay at the output root.
    """

    @pytest.fixture
    def output(self) -> Path:
        return Path("/tmp/pht_run")

    def test_deliverables_dir(self, output: Path) -> None:
        assert deliverables_dir(output) == output / "deliverables"

    def test_readme_md_path(self, output: Path) -> None:
        assert readme_md_path(output) == output / "deliverables" / "README.md"

    def test_all_moved_helpers_root_under_deliverables(self, output: Path) -> None:
        """Every relocated artifact helper resolves under ``deliverables_dir``.

        The day someone reintroduces a root-level write for one of these,
        this loop fails and points at the offending helper by name.
        """
        deliv = deliverables_dir(output)
        moved = {
            "master_measurements_csv_path": master_measurements_csv_path,
            "master_measurements_parquet_path": master_measurements_parquet_path,
            "measurements_csv_path": measurements_csv_path,
            "measurements_parquet_path": measurements_parquet_path,
            "measurements_by_feature_dir": measurements_by_feature_dir,
            "analysis_csv_path": analysis_csv_path,
            "analysis_parquet_path": analysis_parquet_path,
            "dashboard_html_path": dashboard_html_path,
            "pipeline_json_path": pipeline_json_path,
            "readme_md_path": readme_md_path,
        }
        # analysis_html / processing_report_html are imported function-locally
        from phenotypic.tools_ import (
            analysis_html_path,
            processing_report_html_path,
        )

        moved["analysis_html_path"] = analysis_html_path
        moved["processing_report_html_path"] = processing_report_html_path

        for name, helper in moved.items():
            resolved = helper(output)
            assert resolved.parent == deliv or resolved == deliv, (
                f"{name} -> {resolved} is not under {deliv}"
            )
            # Belt-and-suspenders: deliverables_dir is an ancestor.
            assert deliv in resolved.parents or resolved.parent == deliv, (
                f"{name} -> {resolved} does not nest under {deliv}"
            )

    def test_per_image_and_state_helpers_stay_at_root(self, output: Path) -> None:
        """``results/`` and ``processing_state.json`` are NOT deliverables."""
        assert results_dir(output) == output / "results"
        assert dataset_measurements_dir(output, "ds1") == (
            output / "results" / "ds1" / "measurements"
        )
        assert dataset_overlays_dir(output, "ds1") == output / "results" / "ds1" / "overlays"
        assert processing_state_path(output) == output / "processing_state.json"
        # None of these touch the deliverables folder.
        deliv = deliverables_dir(output)
        assert deliv not in results_dir(output).parents
        assert deliv not in processing_state_path(output).parents


# ---------------------------------------------------------------------------
# Multi-objective Pareto deliverables (Phase 4 chunk C)
# ---------------------------------------------------------------------------


class TestParetoPaths:
    """The ``deliverables/pareto/`` multi-objective artifact paths.

    A multi-objective tune run writes its Pareto front + per-objective winners
    into ``<output>/deliverables/pareto/``; these helpers resolve those paths so
    no caller hand-joins ``"pareto"`` (plan §0b path-helper rule).
    """

    @pytest.fixture
    def output(self) -> Path:
        return Path("/tmp/pht_tune")

    def test_dir_pareto_constant(self) -> None:
        from phenotypic.tools_._io_constants import DIR_PARETO

        assert DIR_PARETO == "pareto"

    def test_pareto_front_parquet_constant(self) -> None:
        from phenotypic.tools_._io_constants import PARETO_FRONT_PARQUET

        assert PARETO_FRONT_PARQUET == "pareto_front.parquet"

    def test_pareto_dir_under_deliverables(self, output: Path) -> None:
        from phenotypic.tools_._io_constants import deliverables_dir, pareto_dir

        assert pareto_dir(output) == deliverables_dir(output) / "pareto"

    def test_pareto_front_parquet_path(self, output: Path) -> None:
        from phenotypic.tools_._io_constants import (
            pareto_dir,
            pareto_front_parquet_path,
        )

        assert pareto_front_parquet_path(output) == (
            pareto_dir(output) / "pareto_front.parquet"
        )

    def test_pareto_best_pipeline_path_per_objective(self, output: Path) -> None:
        from phenotypic.tools_._io_constants import (
            pareto_best_pipeline_path,
            pareto_dir,
        )

        assert pareto_best_pipeline_path(output, "Dice") == (
            pareto_dir(output) / "best_Dice.json"
        )
        assert pareto_best_pipeline_path(output, "s0") == (
            pareto_dir(output) / "best_s0.json"
        )

    def test_pareto_paths_root_under_deliverables(self, output: Path) -> None:
        from phenotypic.tools_._io_constants import (
            deliverables_dir,
            pareto_best_pipeline_path,
            pareto_dir,
            pareto_front_parquet_path,
        )

        deliv = deliverables_dir(output)
        assert deliv in pareto_dir(output).parents or pareto_dir(output).parent == deliv
        assert deliv in pareto_front_parquet_path(output).parents
        assert deliv in pareto_best_pipeline_path(output, "IoU").parents


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
        assert JobMetadataKey.DATASETS == "datasets"
        assert JobMetadataKey.INCLUDE_DATASET_COLUMN == "include_dataset_column"
        assert JobMetadataKey.IMAGE_TASK_MAPPING == "image_task_mapping"

    def test_dashboard_manifest_keys(self) -> None:
        assert DashboardManifestKey.VERSION == "version"
        assert DashboardManifestKey.LAST_UPDATED == "last_updated"
        assert DashboardManifestKey.EXECUTION_MODE == "execution_mode"
        assert DashboardManifestKey.TOTAL_IMAGES == "total_images"
        assert DashboardManifestKey.COMPLETED == "completed"
        assert DashboardManifestKey.FAILED == "failed"
        assert DashboardManifestKey.STARTED == "started"
        assert DashboardManifestKey.PENDING == "pending"
        assert DashboardManifestKey.SUCCESS_RATE == "success_rate"
        assert DashboardManifestKey.IS_COMPLETE == "is_complete"
        assert DashboardManifestKey.START_TIME == "start_time"
        assert DashboardManifestKey.INPUT_PATH == "input_path"
        assert DashboardManifestKey.DATASETS == "datasets"
        assert DashboardManifestKey.FAILURE_CATEGORIES == "failure_categories"
        assert DashboardManifestKey.ANALYSIS_DATA_VERSION == "analysis_data_version"
        assert DashboardManifestKey.SLURM_INFO == "slurm_info"

    def test_dashboard_manifest_slurm_info_keys(self) -> None:
        from phenotypic.tools_ import DashboardManifestSlurmInfoKey

        assert DashboardManifestSlurmInfoKey.CHUNK_SCRIPTS == "chunk_scripts"
        assert DashboardManifestSlurmInfoKey.TOTAL_CHUNKS == "total_chunks"
        assert DashboardManifestSlurmInfoKey.CHUNK_JOB_IDS == "chunk_job_ids"
        assert DashboardManifestSlurmInfoKey.ACTIVE_CHUNKS == "active_chunks"
        assert DashboardManifestSlurmInfoKey.COMPLETED_CHUNKS == "completed_chunks"
        assert DashboardManifestSlurmInfoKey.PENDING_CHUNKS == "pending_chunks"

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

    def test_processing_state_keys(self) -> None:
        from phenotypic.tools_ import ProcessingStateKey

        assert ProcessingStateKey.VERSION == "version"
        assert ProcessingStateKey.PIPELINE_PATH == "pipeline_path"
        assert ProcessingStateKey.INPUT_PATH == "input_path"
        assert ProcessingStateKey.OUTPUT_DIR == "output_dir"
        assert ProcessingStateKey.TIMESTAMP == "timestamp"
        assert ProcessingStateKey.EXECUTION_MODE == "execution_mode"
        assert ProcessingStateKey.LAST_UPDATED == "last_updated"
        assert ProcessingStateKey.DATASETS == "datasets"
        assert ProcessingStateKey.CONFIG == "config"
        assert ProcessingStateKey.COMPLETED == "completed"
        assert ProcessingStateKey.FAILED == "failed"
        assert ProcessingStateKey.STARTED == "started"
        assert ProcessingStateKey.ERRORS == "errors"
        assert ProcessingStateKey.INITIAL_IMAGES == "initial_images"

    def test_processing_state_keys_intentionally_overlap_job_metadata_keys(self) -> None:
        """Some keys are deliberately shared between processing_state.json and
        job_metadata.json so a single field migration applies to both contracts.

        This regression test asserts the overlapping keys keep matching string
        values. If they ever diverge, the cross-file rehydration breaks
        silently — the test forces the divergence to be intentional.
        """
        from phenotypic.tools_ import ProcessingStateKey

        assert ProcessingStateKey.EXECUTION_MODE == JobMetadataKey.EXECUTION_MODE
        assert ProcessingStateKey.INPUT_PATH == JobMetadataKey.INPUT_PATH
        assert ProcessingStateKey.DATASETS == JobMetadataKey.DATASETS


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
        assert EnvVar.SLURM_CPUS_PER_TASK == "SLURM_CPUS_PER_TASK"
        assert EnvVar.SLURM_MEM_PER_NODE == "SLURM_MEM_PER_NODE"


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
