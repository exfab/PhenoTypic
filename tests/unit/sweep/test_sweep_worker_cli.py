"""Tests for the sweep worker CLI (--pipeline-name filtering)."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from phenotypic.sweep import Sweep, generate_sweep_manifest
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureShape
from phenotypic.sweep._sweep_process_image import sweep_worker_cli


@pytest.fixture
def manifest_path(tmp_path):
    """Create a sweep manifest with 2 pipelines."""
    config = [
        Sweep(GaussianBlur, sigma=(1.0, 2.0)),
        Sweep(OtsuDetector),
    ]
    path = tmp_path / "manifest.json"
    generate_sweep_manifest(config, meas=[MeasureShape()], filepath=path)
    return path


@pytest.fixture
def dummy_image(tmp_path):
    """Create a dummy image file."""
    img = tmp_path / "plate_0.tiff"
    img.touch()
    return img


class TestPipelineNameOption:

    def test_unknown_pipeline_name_exits_with_error(
        self, manifest_path, dummy_image, tmp_path
    ):
        """--pipeline-name with nonexistent name should exit with error."""
        runner = CliRunner()
        result = runner.invoke(
            sweep_worker_cli,
            [
                "--manifest", str(manifest_path),
                "--image", str(dummy_image),
                "--output-dir", str(tmp_path / "output"),
                "--pipeline-name", "NonexistentPipeline",
            ],
        )
        assert result.exit_code != 0
        assert "NonexistentPipeline" in result.output or "NonexistentPipeline" in (result.stderr_bytes or b"").decode()

    @patch("phenotypic.sweep._sweep_process_image.process_image_all_pipelines_sequential")
    def test_pipeline_name_filters_to_single_pipeline(
        self, mock_process, manifest_path, dummy_image, tmp_path
    ):
        """--pipeline-name should filter pipeline_json_strs to just that pipeline."""
        mock_process.return_value = [("Pipeline_0", True, "")]

        runner = CliRunner()
        result = runner.invoke(
            sweep_worker_cli,
            [
                "--manifest", str(manifest_path),
                "--image", str(dummy_image),
                "--output-dir", str(tmp_path / "output"),
                "--pipeline-name", "Pipeline_0",
            ],
        )

        # Verify process was called with only Pipeline_0
        assert mock_process.called
        call_kwargs = mock_process.call_args[1]
        assert list(call_kwargs["pipeline_json_strs"].keys()) == ["Pipeline_0"]

    @patch("phenotypic.sweep._sweep_process_image.process_image_all_pipelines_sequential")
    def test_no_pipeline_name_passes_all_pipelines(
        self, mock_process, manifest_path, dummy_image, tmp_path
    ):
        """Without --pipeline-name, all pipelines should be passed."""
        mock_process.return_value = [
            ("Pipeline_0", True, ""),
            ("Pipeline_1", True, ""),
        ]

        runner = CliRunner()
        result = runner.invoke(
            sweep_worker_cli,
            [
                "--manifest", str(manifest_path),
                "--image", str(dummy_image),
                "--output-dir", str(tmp_path / "output"),
            ],
        )

        assert mock_process.called
        call_kwargs = mock_process.call_args[1]
        assert len(call_kwargs["pipeline_json_strs"]) == 2

    @patch("phenotypic.sweep._sweep_process_image.process_image_all_pipelines_sequential")
    @patch("phenotypic._cli._cli_update_state.append_completion_event")
    def test_pipeline_name_uses_composite_event_id(
        self, mock_event, mock_process, manifest_path, dummy_image, tmp_path
    ):
        """With --pipeline-name, event log should use composite image::pipeline ID."""
        mock_process.return_value = [("Pipeline_0", True, "")]

        event_log = tmp_path / "events.log"

        runner = CliRunner()
        result = runner.invoke(
            sweep_worker_cli,
            [
                "--manifest", str(manifest_path),
                "--image", str(dummy_image),
                "--output-dir", str(tmp_path / "output"),
                "--pipeline-name", "Pipeline_0",
                "--event-log", str(event_log),
            ],
        )

        assert mock_event.called
        call_kwargs = mock_event.call_args[1]
        assert "::" in call_kwargs["image"]
        assert "Pipeline_0" in call_kwargs["image"]
