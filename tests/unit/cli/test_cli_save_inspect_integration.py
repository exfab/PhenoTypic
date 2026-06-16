"""Integration tests for ``--save-inspect`` end-to-end through the per-image
worker (`process_single_image_core` and `process_single_hdf_measure_core`).

These bypass the ``click`` command surface (covered by the unit tests in
:mod:`test_cli_output_manager_inspect`) and exercise the dispatch loop
that fires after measurement: every measurer with an ``.inspect()``
method is rendered through :meth:`OutputManager.save_inspect` and the
resulting PNG lands at the expected path.

The covered measurer is :class:`MeasureSymmetricZones` — currently the
only opt-in implementation. Adding a second opt-in measurer in a future
PR will exercise the per-key subdirectory split automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.unit.cli._kaleido_utils import requires_kaleido_chrome

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="OutputManager uses POSIX atomic writes",
)


from PIL import Image as PILImage

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSymmetricZones
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_process_single import (
    process_single_hdf_measure_core,
    process_single_image_core,
)
from phenotypic._cli._cli_types import Dataset


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create input/, output/ subdirs and return the parent."""
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    return tmp_path


@pytest.fixture
def saved_image_path(temp_workspace: Path) -> Path:
    """Save the synth yeast plate as a PNG ready for CLI ingestion."""
    grid_image = load_synth_yeast_plate()
    img_path = temp_workspace / "input" / "plate.png"
    PILImage.fromarray(grid_image.rgb[:].astype("uint8")).save(img_path)
    return img_path


@pytest.fixture
def inspect_pipeline_path(temp_workspace: Path) -> Path:
    """Serialize a minimal pipeline with detection + MeasureSymmetricZones."""
    pipeline = ImagePipeline(
        ops=[OtsuDetector()],
        meas=[MeasureSymmetricZones()],
    )
    pipeline_path = temp_workspace / "pipeline.json"
    pipeline_path.write_text(pipeline.to_json())
    return pipeline_path


def _make_output_manager(
    base_dir: Path, *, save_inspects: bool,
) -> OutputManager:
    """Construct an OutputManager wired for the forward-run shape."""
    om = OutputManager.from_config(
        base_dir=base_dir,
        ext=".png",
        include_dataset_column=False,
        overlay_alpha=0.3,
        save_overlays=False,
        save_inspects=save_inspects,
    )
    om.create_structure(
        [Dataset(name="plate", images=[], input_dir=base_dir, output_dir=base_dir)],
    )
    return om


class TestSaveInspectForwardRun:
    """``--save-inspect`` writes a PNG per measurer per image on a forward run."""

    @requires_kaleido_chrome
    def test_save_inspect_writes_png_for_symmetric_zones(
        self, temp_workspace: Path, saved_image_path: Path,
        inspect_pipeline_path: Path,
    ) -> None:
        output_dir = temp_workspace / "output"
        om = _make_output_manager(output_dir, save_inspects=True)

        process_single_image_core(
            pipeline_path=inspect_pipeline_path,
            image_path=saved_image_path,
            output_dir=output_dir,
            dataset_name="plate",
            image_type="Image",
            read_kwargs={},
            output_manager=om,
        )

        expected = (
            output_dir / "results" / "plate" / "inspect"
            / "MeasureSymmetricZones" / "plate.png"
        )
        assert expected.exists(), f"missing {expected}"
        assert expected.stat().st_size > 0
        assert expected.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_inspect_dir_absent_when_flag_disabled(
        self, temp_workspace: Path, saved_image_path: Path,
        inspect_pipeline_path: Path,
    ) -> None:
        output_dir = temp_workspace / "output"
        om = _make_output_manager(output_dir, save_inspects=False)

        process_single_image_core(
            pipeline_path=inspect_pipeline_path,
            image_path=saved_image_path,
            output_dir=output_dir,
            dataset_name="plate",
            image_type="Image",
            read_kwargs={},
            output_manager=om,
        )

        # The inspect/ tree is provisioned in create_structure only when
        # save_inspects=True. Confirm the absence so a regression that
        # accidentally always-creates the directory is caught.
        assert not (output_dir / "results" / "plate" / "inspect").exists()


class TestSaveInspectMeasureRerun:
    """``--save-inspect`` also fires on the measure-mode HDF rerun path."""

    @requires_kaleido_chrome
    def test_save_inspect_regenerates_on_hdf_rerun(
        self, temp_workspace: Path, saved_image_path: Path,
        inspect_pipeline_path: Path,
    ) -> None:
        output_dir = temp_workspace / "output"

        # First: forward run WITHOUT save_inspects, so the HDF is written
        # but the inspect PNG is not.
        om_forward = _make_output_manager(output_dir, save_inspects=False)
        process_single_image_core(
            pipeline_path=inspect_pipeline_path,
            image_path=saved_image_path,
            output_dir=output_dir,
            dataset_name="plate",
            image_type="Image",
            read_kwargs={},
            output_manager=om_forward,
        )
        hdf_path = output_dir / "results" / "plate" / "hdf" / "plate.h5"
        assert hdf_path.exists()
        assert not (output_dir / "results" / "plate" / "inspect").exists()

        # Now: measure-mode rerun WITH save_inspects=True. Reuses the saved
        # HDF, re-runs pipeline.measure(image), repopulates the per-measurer
        # diagnostic cache, and renders the inspect PNG.
        om_remeasure = _make_output_manager(output_dir, save_inspects=True)
        process_single_hdf_measure_core(
            pipeline_path=inspect_pipeline_path,
            hdf_path=hdf_path,
            output_dir=output_dir,
            dataset_name="plate",
            image_type="Image",
            output_manager=om_remeasure,
        )

        expected = (
            output_dir / "results" / "plate" / "inspect"
            / "MeasureSymmetricZones" / "plate.png"
        )
        assert expected.exists(), f"missing {expected}"
        assert expected.stat().st_size > 0
