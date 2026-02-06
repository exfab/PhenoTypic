"""Tests for the sweep CLI: output manager, flat-dir scanning, config validation."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from phenotypic.sweep import Sweep, generate_sweep_manifest
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureShape

from phenotypic.sweep._sweep_output import SweepOutputManager
from phenotypic.sweep._sweep_cli import _scan_flat_image_dir, _flatten_pipelines


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sweep_manifest(tmp_path):
    """Create a sweep manifest JSON on disk and return its path."""
    config = [
        Sweep(GaussianBlur, sigma=(1.0, 2.0)),
        Sweep(OtsuDetector),
    ]
    manifest_path = tmp_path / "manifest.json"
    generate_sweep_manifest(config, meas=[MeasureShape()], filepath=manifest_path)
    return manifest_path


@pytest.fixture
def pipeline_names():
    """Pipeline names matching simple_config fixture."""
    return ["Pipeline_0", "Pipeline_1"]


@pytest.fixture
def output_manager(tmp_path, pipeline_names):
    """SweepOutputManager with default settings."""
    mgr = SweepOutputManager(
        base_dir=tmp_path / "output",
        save_layers={"rgb": False, "gray": False, "detect_mat": False,
                     "objmask": False, "objmap": False, "objmap_overlay": False,
                     "detect_mat_overlay": False, "objmask_overlay": False},
        extensions={"rgb": ".tiff", "gray": ".tiff", "detect_mat": ".tiff",
                    "objmask": ".png", "objmap": ".png", "objmap_overlay": ".png"},
    )
    mgr.create_structure(pipeline_names)
    return mgr


@pytest.fixture
def flat_image_dir(tmp_path):
    """Create a flat directory with dummy image files."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(3):
        (img_dir / f"plate_{i}.tiff").touch()
    return img_dir


# ---------------------------------------------------------------------------
# SweepOutputManager tests
# ---------------------------------------------------------------------------


class TestSweepOutputManager:

    def test_create_structure_creates_directories(self, output_manager, pipeline_names):
        """Verify that output structure has results/<pipeline>/measurements/ and overlays/."""
        for name in pipeline_names:
            pipe_dir = output_manager.results_dir / name
            assert pipe_dir.is_dir()
            assert (pipe_dir / "measurements").is_dir()
            assert (pipe_dir / "overlays").is_dir()

    def test_create_structure_creates_logs_dir(self, output_manager):
        assert output_manager.logs_dir.is_dir()
        assert (output_manager.logs_dir / "slurm").is_dir()

    def test_create_structure_with_layers(self, tmp_path):
        """Verify optional layer directories are created when enabled."""
        mgr = SweepOutputManager(
            base_dir=tmp_path / "out",
            save_layers={"rgb": True, "gray": False, "objmask": True,
                         "detect_mat": False, "objmap": False,
                         "objmap_overlay": False, "detect_mat_overlay": False,
                         "objmask_overlay": False},
            extensions={},
        )
        mgr.create_structure(["P_0"])
        assert (mgr.results_dir / "P_0" / "rgb").is_dir()
        assert (mgr.results_dir / "P_0" / "objmask").is_dir()
        assert not (mgr.results_dir / "P_0" / "gray").exists()

    def test_get_output_path_measurements(self, output_manager):
        path = output_manager.get_output_path("Pipeline_0", "measurements", "img1")
        assert path == output_manager.results_dir / "Pipeline_0" / "measurements" / "img1.csv"

    def test_get_output_path_overlays(self, output_manager):
        path = output_manager.get_output_path("Pipeline_0", "overlays", "img1")
        assert path == output_manager.results_dir / "Pipeline_0" / "overlays" / "img1.png"

    def test_get_output_path_disabled_layer_raises(self, output_manager):
        with pytest.raises(ValueError, match="not enabled"):
            output_manager.get_output_path("Pipeline_0", "rgb", "img1")

    def test_save_measurements_adds_pipeline_column(self, output_manager):
        df = pd.DataFrame({"Area": [100, 200], "Perimeter": [40, 60]})
        path = output_manager.save_measurements(df, "Pipeline_0", "plate_1")

        assert path.exists()
        loaded = pd.read_csv(path)
        assert "Metadata_Pipeline" in loaded.columns
        assert (loaded["Metadata_Pipeline"] == "Pipeline_0").all()

    def test_save_measurements_preserves_existing_pipeline_col(self, output_manager):
        """If Metadata_Pipeline already exists, don't duplicate it."""
        df = pd.DataFrame({
            "Metadata_Pipeline": ["P_custom"] * 3,
            "Area": [1, 2, 3],
        })
        path = output_manager.save_measurements(df, "Pipeline_0", "plate_1")
        loaded = pd.read_csv(path)
        assert loaded.columns.tolist().count("Metadata_Pipeline") == 1
        assert (loaded["Metadata_Pipeline"] == "P_custom").all()

    def test_aggregate_pipeline_csv(self, output_manager):
        """Per-pipeline CSV aggregation combines all image CSVs."""
        # Write two per-image CSVs
        for stem in ["img1", "img2"]:
            df = pd.DataFrame({
                "Metadata_Pipeline": ["Pipeline_0"],
                "Area": [100 if stem == "img1" else 200],
            })
            output_manager.save_measurements(df, "Pipeline_0", stem)

        agg_path = output_manager.aggregate_pipeline_csv("Pipeline_0")
        assert agg_path is not None
        assert agg_path.exists()

        combined = pd.read_csv(agg_path)
        assert len(combined) == 2
        assert (combined["Metadata_Pipeline"] == "Pipeline_0").all()

    def test_aggregate_pipeline_csv_empty_returns_none(self, output_manager):
        assert output_manager.aggregate_pipeline_csv("Pipeline_0") is None

    def test_aggregate_master_csv(self, output_manager, pipeline_names):
        """Master CSV combines across all pipelines."""
        for pipe_name in pipeline_names:
            df = pd.DataFrame({
                "Area": [100],
            })
            output_manager.save_measurements(df, pipe_name, "img1")

        master = output_manager.aggregate_master_csv(pipeline_names)
        assert master is not None
        assert master.exists()

        master_df = pd.read_csv(master)
        assert len(master_df) == 2  # one row per pipeline
        assert "Metadata_Pipeline" in master_df.columns
        assert set(master_df["Metadata_Pipeline"]) == set(pipeline_names)

    def test_aggregate_master_csv_no_data_returns_none(self, output_manager, pipeline_names):
        assert output_manager.aggregate_master_csv(pipeline_names) is None


# ---------------------------------------------------------------------------
# Flat directory scanning tests
# ---------------------------------------------------------------------------


class TestScanFlatImageDir:

    def test_scans_images(self, flat_image_dir):
        images = _scan_flat_image_dir(flat_image_dir)
        assert len(images) == 3
        assert all(p.suffix == ".tiff" for p in images)

    def test_sorted_output(self, flat_image_dir):
        images = _scan_flat_image_dir(flat_image_dir)
        assert images == sorted(images)

    def test_nonexistent_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _scan_flat_image_dir(tmp_path / "nope")

    def test_not_a_directory_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.touch()
        with pytest.raises(ValueError, match="not a directory"):
            _scan_flat_image_dir(f)

    def test_empty_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No valid images"):
            _scan_flat_image_dir(empty)

    def test_subdirs_with_images_rejected(self, flat_image_dir):
        """Subdirectories containing images should cause an error."""
        sub = flat_image_dir / "subdir"
        sub.mkdir()
        (sub / "nested.tiff").touch()
        with pytest.raises(ValueError, match="flat image directory"):
            _scan_flat_image_dir(flat_image_dir)

    def test_subdirs_without_images_ok(self, flat_image_dir):
        """Non-image subdirectories are ignored."""
        sub = flat_image_dir / "logs"
        sub.mkdir()
        (sub / "log.txt").touch()
        images = _scan_flat_image_dir(flat_image_dir)
        assert len(images) == 3

    def test_ignores_non_image_files(self, flat_image_dir):
        """Non-image files in the flat directory are ignored."""
        (flat_image_dir / "readme.txt").touch()
        (flat_image_dir / "data.csv").touch()
        images = _scan_flat_image_dir(flat_image_dir)
        assert len(images) == 3


# ---------------------------------------------------------------------------
# Flatten pipelines tests
# ---------------------------------------------------------------------------


class TestFlattenPipelines:

    def test_flatten_returns_dict(self, sweep_manifest):
        result = _flatten_pipelines(sweep_manifest)
        assert isinstance(result, dict)
        assert len(result) == 2  # 2 sigma values × 1 OtsuDetector = 2

    def test_flatten_keys_are_pipeline_names(self, sweep_manifest):
        result = _flatten_pipelines(sweep_manifest)
        assert "Pipeline_0" in result
        assert "Pipeline_1" in result

    def test_flatten_values_are_json_strings(self, sweep_manifest):
        result = _flatten_pipelines(sweep_manifest)
        for json_str in result.values():
            assert isinstance(json_str, str)
            parsed = json.loads(json_str)
            assert "pipe_cfgs" in parsed

    def test_nonexistent_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _flatten_pipelines(tmp_path / "no.json")
