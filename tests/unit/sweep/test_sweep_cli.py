"""Tests for the sweep CLI: output manager, flat-dir scanning, config validation."""

import json

import pandas as pd
import pytest

from phenotypic.sweep import Sweep, generate_sweep_manifest
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureShape

from phenotypic.sweep._sweep_cli._sweep_output import SweepOutputManager, clear_previous_run
from phenotypic.sweep._sweep_cli._sweep_cli import _scan_flat_image_dir, _flatten_pipelines


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
def output_manager(tmp_path):
    """SweepOutputManager with default settings."""
    mgr = SweepOutputManager(base_dir=tmp_path / "output")
    mgr.create_structure()
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

    def test_create_structure_creates_base_directories(self, output_manager):
        """Verify that output structure has results/, logs/, and logs/slurm/."""
        assert output_manager.results_dir.is_dir()
        assert output_manager.logs_dir.is_dir()
        assert (output_manager.logs_dir / "slurm").is_dir()
        assert output_manager.failures_dir.is_dir()

    def test_save_measurements_creates_image_first_path(self, output_manager):
        """Measurements saved at results/<image_stem>/<pipeline>/<stem>.csv."""
        df = pd.DataFrame({"Area": [100, 200], "Perimeter": [40, 60]})
        path = output_manager.save_measurements(df, "Pipeline_0", "plate_1")

        assert path is not None
        assert path.exists()
        expected = (
            output_manager.results_dir
            / "plate_1" / "Pipeline_0" / "plate_1.csv"
        )
        assert path == expected

    def test_save_measurements_adds_pipeline_column(self, output_manager):
        df = pd.DataFrame({"Area": [100, 200], "Perimeter": [40, 60]})
        path = output_manager.save_measurements(df, "Pipeline_0", "plate_1")

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

    def test_save_image_hdf5_creates_image_first_path(self, output_manager):
        """HDF5 saved at results/<image_stem>/<pipeline>/<stem>.h5."""
        from unittest.mock import MagicMock

        mock_image = MagicMock()
        path = output_manager.save_image_hdf5(mock_image, "Pipeline_0", "plate_1")

        assert path is not None
        expected = (
            output_manager.results_dir
            / "plate_1" / "Pipeline_0" / "plate_1.h5"
        )
        assert path == expected
        mock_image.save2hdf5.assert_called_once_with(expected)

    def test_save_image_hdf5_returns_none_on_error(self, output_manager):
        """If save2hdf5 raises, returns None without propagating."""
        from unittest.mock import MagicMock

        mock_image = MagicMock()
        mock_image.save2hdf5.side_effect = OSError("disk full")
        path = output_manager.save_image_hdf5(mock_image, "Pipeline_0", "plate_1")
        assert path is None


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


# ---------------------------------------------------------------------------
# Clear previous run tests
# ---------------------------------------------------------------------------


class TestClearPreviousRun:

    def test_clear_deletes_previous_results(self, tmp_path):
        """Previous results/, logs/, and manifest are deleted."""
        out = tmp_path / "output"
        results = out / "results" / "Pipeline_0" / "measurements"
        results.mkdir(parents=True)
        (results / "img1.csv").write_text("a,b\n1,2")
        (out / "logs").mkdir()
        (out / "sweep_manifest.json").write_text("{}")

        assert clear_previous_run(out) is True

        # Original locations are gone
        assert not (out / "results").exists()
        assert not (out / "logs").exists()
        assert not (out / "sweep_manifest.json").exists()
        # Output dir itself still exists but is empty
        assert out.is_dir()
        assert not any(out.iterdir())

    def test_noop_when_no_results(self, tmp_path):
        """Empty output dir (no results/) returns False."""
        out = tmp_path / "output"
        out.mkdir()

        assert clear_previous_run(out) is False

    def test_noop_when_results_empty(self, tmp_path):
        """Empty results/ directory returns False."""
        out = tmp_path / "output"
        (out / "results").mkdir(parents=True)

        assert clear_previous_run(out) is False

    def test_noop_when_dir_missing(self, tmp_path):
        """Nonexistent output dir returns False."""
        assert clear_previous_run(tmp_path / "nonexistent") is False
