"""Tests for sweep failure logging.

Verifies that SweepOutputManager writes per-failure log files with
timestamp, image path, pipeline name, traceback, and pipeline JSON.
"""

from pathlib import Path

import pytest

from phenotypic.sweep._sweep_cli._sweep_output import SweepOutputManager


@pytest.fixture
def output_manager(tmp_path):
    """Create a SweepOutputManager with a temporary base directory."""
    return SweepOutputManager(base_dir=tmp_path)


class TestCreateStructure:
    """Tests for failures directory creation during create_structure."""

    def test_creates_failures_dir(self, output_manager, tmp_path):
        output_manager.create_structure()
        assert (tmp_path / "logs" / "failures").is_dir()

    def test_failures_dir_attribute(self, output_manager, tmp_path):
        assert output_manager.failures_dir == tmp_path / "logs" / "failures"


class TestWriteFailureLog:
    """Tests for write_failure_log method."""

    def test_writes_log_file(self, output_manager, tmp_path):
        log_path = output_manager.write_failure_log(
            image_path=Path("/data/images/plate001.tiff"),
            pipeline_name="Config_0",
            traceback_str="Traceback (most recent call last):\n  ...\nValueError: boom",
            pipeline_json_str='{"operations": []}',
        )
        assert log_path is not None
        assert log_path.exists()
        assert log_path.name == "plate001__Config_0.log"

    def test_log_file_content(self, output_manager):
        log_path = output_manager.write_failure_log(
            image_path=Path("/data/images/plate001.tiff"),
            pipeline_name="Config_0",
            traceback_str="ValueError: boom",
            pipeline_json_str='{"ops": []}',
        )
        content = log_path.read_text()
        assert "Timestamp:" in content
        assert str(Path("/data/images/plate001.tiff")) in content
        assert "Pipeline:  Config_0" in content
        assert "ValueError: boom" in content
        assert '{"ops": []}' in content

    def test_creates_failures_dir_lazily(self, output_manager, tmp_path):
        """SLURM workers skip create_structure; write_failure_log creates dir."""
        assert not (tmp_path / "logs" / "failures").exists()
        log_path = output_manager.write_failure_log(
            image_path=Path("img.tiff"),
            pipeline_name="P0",
            traceback_str="err",
            pipeline_json_str="{}",
        )
        assert log_path is not None
        assert (tmp_path / "logs" / "failures").is_dir()

    def test_sanitizes_pipeline_name(self, output_manager):
        log_path = output_manager.write_failure_log(
            image_path=Path("img.tiff"),
            pipeline_name="A/B\\C",
            traceback_str="err",
            pipeline_json_str="{}",
        )
        assert log_path is not None
        assert "/" not in log_path.name
        assert "\\" not in log_path.name
        assert log_path.name == "img__A_B_C.log"

    def test_returns_none_on_write_error(self, output_manager, tmp_path, monkeypatch):
        """If disk write fails, returns None without raising."""
        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", _boom)
        result = output_manager.write_failure_log(
            image_path=Path("img.tiff"),
            pipeline_name="P0",
            traceback_str="err",
            pipeline_json_str="{}",
        )
        assert result is None
