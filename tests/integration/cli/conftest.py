"""Shared fixtures for CLI integration tests.

Mirrors the synth-plate + pipeline-JSON setup used by
``test_cli_hdf_output.py`` so multiple integration modules can share one
deterministic input dir and serialized pipeline.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from PIL import Image as PILImage

from phenotypic.data import load_synth_yeast_plate
from phenotypic.prefab import RoundPeaksPipeline


def _write_synth_image(target_path: Path) -> None:
    """Render a synthetic yeast plate as RGB and save to ``target_path``."""
    grid_image = load_synth_yeast_plate()
    pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
    pil_img.save(target_path)


@pytest.fixture
def synth_plate_dir(tmp_path: Path) -> Path:
    """Input directory named ``plates`` carrying one synth plate image.

    Named ``plates`` so the CLI's dataset discovery (basename of the input
    dir) yields a predictable ``results/plates/...`` layout.
    """
    input_dir = tmp_path / "plates"
    input_dir.mkdir()
    _write_synth_image(input_dir / "plate_001.png")
    return input_dir


@pytest.fixture
def synth_one_level_input(tmp_path: Path) -> Path:
    """One-level input tree: ``<tmp>/in/day1/plateA.tif`` (one synth plate).

    Returns the input root (``<tmp>/in``) so callers can pass it as
    ``--input`` and assert on the mirrored ``--process-only`` output tree.
    """
    root = tmp_path / "in"
    day = root / "day1"
    day.mkdir(parents=True)
    _write_synth_image(day / "plateA.tif")
    return root


@pytest.fixture
def simple_pipeline_json():
    """Write a minimal RoundPeaksPipeline JSON to a temp file."""
    pipeline = RoundPeaksPipeline(
        blur_sigma=3,
        detector_thresh_method="otsu",
        detector_subtract_background=True,
        detector_remove_noise=True,
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as handle:
        handle.write(pipeline.to_json())
        pipeline_path = Path(handle.name)
    try:
        yield pipeline_path
    finally:
        if pipeline_path.exists():
            pipeline_path.unlink()
