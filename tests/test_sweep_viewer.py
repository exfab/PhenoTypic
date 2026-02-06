"""Tests for the napari sweep viewer data model and utilities.

All tests in this module run without a display server or Qt—only the data
model and remote-display helpers are exercised.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_MANIFEST = {
    "version": "0.13.0",
    "description": "test sweep",
    "total_pipelines": 2,
    "configs": {
        "Pipeline": {
            "n_combinations": 2,
            "pipelines": {
                "Pipeline_0": {
                    "pipe_cfgs": {
                        "GaussianBlur_0": {
                            "class": "GaussianBlur",
                            "params": {"sigma": 1.0},
                        },
                        "OtsuDetector_0": {
                            "class": "OtsuDetector",
                            "params": {"ignore_zeros": True},
                        },
                    },
                    "meas_cfgs": {
                        "MeasureColor_0": {
                            "class": "MeasureColor",
                            "params": {},
                        },
                    },
                },
                "Pipeline_1": {
                    "pipe_cfgs": {
                        "GaussianBlur_0": {
                            "class": "GaussianBlur",
                            "params": {"sigma": 2.0},
                        },
                        "OtsuDetector_0": {
                            "class": "OtsuDetector",
                            "params": {"ignore_zeros": False},
                        },
                    },
                    "meas_cfgs": {},
                },
            },
        }
    },
}


@pytest.fixture()
def mock_sweep_dir(tmp_path: Path) -> Path:
    """Create a minimal mock sweep output directory."""
    # Write manifest
    (tmp_path / "sweep_manifest.json").write_text(
        json.dumps(MOCK_MANIFEST, indent=2)
    )

    # Create result images (1x1 white PNG via raw bytes is fragile,
    # so just create empty files with image extensions).
    results = tmp_path / "results"
    for pipe in ("Pipeline_0", "Pipeline_1"):
        for comp in ("overlays", "rgb", "objmask"):
            d = results / pipe / comp
            d.mkdir(parents=True, exist_ok=True)
            for stem in ("plate_001", "plate_002"):
                (d / f"{stem}.png").write_bytes(b"")

        # measurements CSVs
        meas_dir = results / pipe / "measurements"
        meas_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {"col_area": [100, 200], "col_roundness": [0.9, 0.8]}
        )
        df.to_csv(meas_dir / "plate_001.csv", index=False)

    return tmp_path


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestSweepOutputScanner:
    """Tests for SweepOutputScanner."""

    def test_detect_sweep_dir_valid(self, mock_sweep_dir: Path):
        from phenotypic.gui.sweep import SweepOutputScanner

        result = SweepOutputScanner.detect_sweep_dir(mock_sweep_dir)
        assert result == mock_sweep_dir.resolve()

    def test_detect_sweep_dir_missing(self, tmp_path: Path):
        from phenotypic.gui.sweep import SweepOutputScanner

        with pytest.raises(FileNotFoundError, match="sweep_manifest.json"):
            SweepOutputScanner.detect_sweep_dir(tmp_path)

    def test_parse_manifest(self, mock_sweep_dir: Path):
        from phenotypic.gui.sweep._sweep_data_model import SweepOutputScanner

        manifest_path = mock_sweep_dir / "sweep_manifest.json"
        raw, configs = SweepOutputScanner._parse_manifest(manifest_path)

        assert raw["total_pipelines"] == 2
        assert "Pipeline_0" in configs
        assert "Pipeline_1" in configs

        cfg0 = configs["Pipeline_0"]
        assert cfg0.config_group == "Pipeline"
        assert len(cfg0.operations) == 2
        assert cfg0.operations[0]["class"] == "GaussianBlur"
        assert cfg0.operations[0]["params"]["sigma"] == 1.0
        assert len(cfg0.measurements) == 1
        assert cfg0.measurements[0]["class"] == "MeasureColor"

        cfg1 = configs["Pipeline_1"]
        assert len(cfg1.measurements) == 0

    def test_scan_results(self, mock_sweep_dir: Path):
        from phenotypic.gui.sweep._sweep_data_model import SweepOutputScanner

        results_dir = mock_sweep_dir / "results"
        files = SweepOutputScanner._scan_results(results_dir)

        # 2 pipelines x 3 components x 2 images = 12
        assert len(files) == 12

        # Verify structure
        components = {f.component for f in files}
        assert components == {"overlays", "rgb", "objmask"}

        pipelines = {f.pipeline_name for f in files}
        assert pipelines == {"Pipeline_0", "Pipeline_1"}

        stems = {f.image_stem for f in files}
        assert stems == {"plate_001", "plate_002"}

    def test_full_scan(self, mock_sweep_dir: Path):
        from phenotypic.gui.sweep import SweepOutputScanner

        data = SweepOutputScanner.scan(mock_sweep_dir)

        assert data.root_dir == mock_sweep_dir.resolve()
        assert data.pipeline_names == ["Pipeline_0", "Pipeline_1"]
        assert data.image_stems == ["plate_001", "plate_002"]
        assert sorted(data.components) == ["objmask", "overlays", "rgb"]

    def test_by_pipeline_index(self, mock_sweep_dir: Path):
        from phenotypic.gui.sweep import SweepOutputScanner

        data = SweepOutputScanner.scan(mock_sweep_dir)

        # Lookup: Pipeline_0 -> plate_001 -> rgb
        sf = data.by_pipeline["Pipeline_0"]["plate_001"]["rgb"]
        assert sf.pipeline_name == "Pipeline_0"
        assert sf.image_stem == "plate_001"
        assert sf.component == "rgb"
        assert sf.path.name == "plate_001.png"

    def test_by_image_index(self, mock_sweep_dir: Path):
        from phenotypic.gui.sweep import SweepOutputScanner

        data = SweepOutputScanner.scan(mock_sweep_dir)

        # Lookup: plate_002 -> objmask -> Pipeline_1
        sf = data.by_image["plate_002"]["objmask"]["Pipeline_1"]
        assert sf.pipeline_name == "Pipeline_1"
        assert sf.image_stem == "plate_002"
        assert sf.component == "objmask"


# ---------------------------------------------------------------------------
# Remote display tests
# ---------------------------------------------------------------------------


class TestRemoteDisplay:
    """Tests for remote display detection and configuration."""

    def test_detect_remote_session_false(self, monkeypatch: pytest.MonkeyPatch):
        from phenotypic.gui.sweep._remote_display import detect_remote_session

        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        assert detect_remote_session() is False

    def test_detect_remote_session_true_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from phenotypic.gui.sweep._remote_display import detect_remote_session

        monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 56789 10.0.0.1 22")
        assert detect_remote_session() is True

    def test_detect_remote_session_true_client(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from phenotypic.gui.sweep._remote_display import detect_remote_session

        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 56789 22")
        assert detect_remote_session() is True

    def test_configure_remote_display(self, monkeypatch: pytest.MonkeyPatch):
        from phenotypic.gui.sweep._remote_display import configure_remote_display

        monkeypatch.delenv("QT_OPENGL", raising=False)
        monkeypatch.delenv("LIBGL_ALWAYS_SOFTWARE", raising=False)
        monkeypatch.setenv("LIBGL_ALWAYS_INDIRECT", "1")

        configure_remote_display()

        assert os.environ["QT_OPENGL"] == "software"
        assert os.environ["LIBGL_ALWAYS_SOFTWARE"] == "1"
        assert "LIBGL_ALWAYS_INDIRECT" not in os.environ


# ---------------------------------------------------------------------------
# Measurements CSV loading test
# ---------------------------------------------------------------------------


class TestMeasurementsCSV:
    """Test that measurement CSVs are found and parsed correctly."""

    def test_measurements_csv_loading(self, mock_sweep_dir: Path):
        csv_path = (
            mock_sweep_dir
            / "results"
            / "Pipeline_0"
            / "measurements"
            / "plate_001.csv"
        )
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["col_area", "col_roundness"]
        assert len(df) == 2
        assert df["col_area"].iloc[0] == 100
