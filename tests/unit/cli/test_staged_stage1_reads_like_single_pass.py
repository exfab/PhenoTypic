"""Staged Stage 1 reads each input exactly as a single-pass run does.

Before PR H the staged engine's Stage 1 never applied ``--nrows``/``--ncols``
or the pipeline's grid preset: the local path built ``read_kwargs`` from
``bit_depth``/``detect_mode`` only, and the SLURM worker called
``stage1_preprocess_core`` with no ``read_kwargs`` at all, so on SLURM
``--bit-depth`` and ``--detect-mode`` were dropped too. Every staged
``GridImage`` was 8 x 12, whatever the flags said. Both paths now build their
kwargs with ``input_read_settings`` + ``image_read_kwargs``, the helpers the
single-pass cores use, and the SLURM worker reads the flags from
``job_metadata.json``, which the submitter writes before any worker runs.

The reference here is built independently of those helpers: ``imread`` with
the kwargs spelled out, then ``apply_and_measure`` in-process.

Fixtures are a 4 x 6 synthetic plate (so the default 8 x 12 grid is wrong for
it) and, for ``--bit-depth``, a dim frame stored as ``int32`` whose values fit
8 bits: without the flag ``Image`` narrows it to ``uint8`` and divides by 255;
``--bit-depth 16`` keeps it on the 16-bit scale, 257 x darker.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from click.testing import CliRunner

import phenotypic
from phenotypic import GridImage, ImagePipeline
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_staged_slurm import _write_staged_job_metadata
from phenotypic._cli._cli_types import Dataset, ExecutionConfig
from phenotypic._cli._stages import STAGE_GPU_DETECT
from phenotypic.data._synthetic_data import make_synthetic_plate
from phenotypic.measure import MeasureSize
from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    event_log_path,
    zarr_store_path,
)
from tests._fakes.fake_gpu_detector import FakeGpuDetector

DATASET = "plates"
STEM = "plate"
#: The columns compared between the staged table and the reference.
COMPARED = ["Size_Area", "Grid_RowNum", "Grid_ColNum", "Bbox_CenterRR", "Bbox_CenterCC"]


@pytest.fixture(autouse=True)
def _register_fake_gpu_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    """``from_json`` resolves classes by bare name in the phenotypic namespace."""
    monkeypatch.setattr(phenotypic, "FakeGpuDetector", FakeGpuDetector, raising=False)


def _write_plate(root: Path, pixels: np.ndarray) -> Path:
    path = root / "images" / DATASET / f"{STEM}.tiff"
    path.parent.mkdir(parents=True)
    tifffile.imwrite(path, pixels)
    return path


@pytest.fixture
def plate_4x6(tmp_path: Path) -> Path:
    """A 4 x 6 synthetic plate, 8-bit RGB."""
    pixels = make_synthetic_plate(nrows=4, ncols=6, plate_h=300, plate_w=450)
    return _write_plate(tmp_path, pixels)


@pytest.fixture
def dim_int32_plate(tmp_path: Path) -> Path:
    """A dim frame of a 12-bit camera, saved as int32: every value fits 8 bits."""
    pixels = make_synthetic_plate(nrows=4, ncols=6, plate_h=300, plate_w=450, bit_depth=16)
    dim = (pixels >> 4).astype(np.int32) // 20  # 12-bit, then dim
    assert dim.max() <= 255
    return _write_plate(tmp_path, dim)


@pytest.fixture
def pipeline_file(tmp_path: Path) -> Path:
    path = tmp_path / "pipeline.json"
    pipeline = ImagePipeline(
        ops={"FakeGpuDetector": FakeGpuDetector(threshold=0.75)},
        meas={"MeasureSize": MeasureSize()},
    )
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return path


def _reference(pipeline_file: Path, image_path: Path, **read) -> tuple[GridImage, pd.DataFrame]:
    """The single-pass result, with its reader kwargs spelled out by hand."""
    detect_mode = read.pop("detect_mode", "gray")
    image = GridImage.imread(image_path, **read)
    if detect_mode != "gray":
        image.set_detect_mode(detect_mode)
    table = ImagePipeline.from_json(pipeline_file).apply_and_measure(
        image, inplace=True, apply_post=False
    )
    return image, table


def _store_table(out: Path) -> pd.DataFrame:
    path = zarr_store_path(out, DATASET, STEM) / MEASUREMENT_TABLE_RELATIVE_PATH
    return pd.read_parquet(path)


def _comparable(table: pd.DataFrame) -> pd.DataFrame:
    frame = table.copy()
    frame.columns = [str(c) for c in frame.columns]
    frame = frame.set_index("Object_Label").sort_index()[COMPARED]
    return frame.astype({"Grid_RowNum": int, "Grid_ColNum": int})


def _config(out: Path, pipeline_file: Path, image_path: Path, **flags) -> ExecutionConfig:
    return ExecutionConfig(
        pipeline_json=pipeline_file,
        input_path=image_path.parents[1],
        output_dir=out,
        image_type="GridImage",
        nrows=flags.get("nrows"),
        ncols=flags.get("ncols"),
        bit_depth=flags.get("bit_depth"),
        detect_mode=flags.get("detect_mode", "gray"),
        n_jobs=1,
        slurm_args={},
        force_local=True,
        wait=False,
        ext=".tiff",
        overlay_alpha=0.3,
        include_dataset_column=True,
        dry_run=False,
        sample=None,
        resume=False,
        retry_failures=False,
        skip_validation=True,
        save_overlays=False,
    )


def _run_local_staged(out: Path, pipeline_file: Path, image_path: Path, *flags: str) -> None:
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline", str(pipeline_file),
            "--input", str(image_path.parents[1]),
            "--output", str(out),
            "--force-local",
            "--skip-validation",
            "--njobs", "1",
            *flags,
        ],
    )
    assert result.exit_code == 0, result.output
    events = event_log_path(out).read_text(encoding="utf-8")
    assert STAGE_GPU_DETECT in events, "the run did not take the staged path"


def _run_slurm_workers(out: Path, pipeline_file: Path, image_path: Path, **flags) -> None:
    """The three staged SLURM worker steps, after the submitter's metadata write."""
    import phenotypic._cli._cli_staged_slurm_worker as worker

    config = _config(out, pipeline_file, image_path, **flags)
    dataset = Dataset(DATASET, [image_path], image_path.parent, out)
    OutputManager.from_config(out, ".tiff", save_overlays=False).create_structure([dataset])
    placeholder = out / "unused.sh"
    _write_staged_job_metadata(
        datasets=[dataset],
        output_dir=out,
        config=config,
        scripts={
            "stage1": [], "controller": placeholder, "stage2": placeholder,
            "stage3": [], "finalizer": placeholder,
        },
        job_ids=[],
        start_time=datetime.now(),
        epoch="test-epoch",
    )
    manifest = [(DATASET, STEM, str(image_path))]
    worker.run_stage1_step(
        pipeline_path=pipeline_file, output_dir=out, image_type="GridImage",
        manifest=manifest, index=0,
    )
    worker.run_stage2_shard(
        pipeline_path=pipeline_file, output_dir=out, image_type="GridImage",
        manifest=manifest, shard_index=0, n_shards=1,
    )
    worker.run_stage3_step(
        pipeline_path=pipeline_file, output_dir=out, image_type="GridImage",
        manifest=manifest, index=0,
    )


def _assert_store_matches(out: Path, reference: GridImage, table: pd.DataFrame) -> None:
    stored = GridImage.load_zarr(zarr_store_path(out, DATASET, STEM))
    assert (stored.nrows, stored.ncols) == (reference.nrows, reference.ncols)
    assert stored.bit_depth == reference.bit_depth
    np.testing.assert_array_equal(stored.gray[:], reference.gray[:])
    pd.testing.assert_frame_equal(_comparable(_store_table(out)), _comparable(table))


def test_the_reference_is_a_4x6_grid_unlike_the_default(pipeline_file, plate_4x6) -> None:
    """The control: at 8 x 12 the same plate's grid columns differ."""
    _, at_4x6 = _reference(pipeline_file, plate_4x6, nrows=4, ncols=6)
    _, at_8x12 = _reference(pipeline_file, plate_4x6, nrows=8, ncols=12)

    assert len(at_4x6) == len(at_8x12) > 0
    assert int(at_4x6["Grid_RowNum"].astype(int).max()) == 3
    assert not _comparable(at_4x6).equals(_comparable(at_8x12))


def test_local_staged_run_uses_the_grid_flags(tmp_path, pipeline_file, plate_4x6) -> None:
    out = tmp_path / "out"
    reference, table = _reference(pipeline_file, plate_4x6, nrows=4, ncols=6)

    _run_local_staged(out, pipeline_file, plate_4x6, "--nrows", "4", "--ncols", "6")

    _assert_store_matches(out, reference, table)


def test_slurm_staged_workers_use_the_grid_flags(tmp_path, pipeline_file, plate_4x6) -> None:
    out = tmp_path / "out"
    reference, table = _reference(pipeline_file, plate_4x6, nrows=4, ncols=6)

    _run_slurm_workers(out, pipeline_file, plate_4x6, nrows=4, ncols=6)

    _assert_store_matches(out, reference, table)


def test_slurm_staged_workers_use_the_pipeline_preset(tmp_path, plate_4x6) -> None:
    """No flags: the preset is the grid, as ``resolve_grid_shape`` says."""
    pipeline_file = tmp_path / "preset.json"
    pipeline_file.write_text(
        ImagePipeline(
            ops={"FakeGpuDetector": FakeGpuDetector(threshold=0.75)},
            meas={"MeasureSize": MeasureSize()},
            nrows=4,
            ncols=6,
        ).to_json(),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    reference, table = _reference(pipeline_file, plate_4x6, nrows=4, ncols=6)

    _run_slurm_workers(out, pipeline_file, plate_4x6)

    _assert_store_matches(out, reference, table)


def test_the_bit_depth_flag_changes_this_frames_intensities(pipeline_file, dim_int32_plate) -> None:
    """The control: the flag is what keeps the frame on the 16-bit scale."""
    pinned, _ = _reference(pipeline_file, dim_int32_plate, nrows=4, ncols=6, bit_depth=16)
    inferred, _ = _reference(pipeline_file, dim_int32_plate, nrows=4, ncols=6)

    assert (pinned.bit_depth, inferred.bit_depth) == (16, 8)
    np.testing.assert_allclose(inferred.gray[:], pinned.gray[:] * 65535 / 255, rtol=1e-5)


@pytest.mark.parametrize("path", ["local", "slurm"])
def test_staged_runs_read_with_bit_depth_and_detect_mode(
    tmp_path, pipeline_file, dim_int32_plate, path
) -> None:
    """At the default grid, so only the two reader settings are under test.

    The local path already carried them before PR H; the SLURM worker did not.
    """
    out = tmp_path / "out"
    reference, table = _reference(
        pipeline_file, dim_int32_plate, nrows=8, ncols=12, bit_depth=16, detect_mode="LabL"
    )

    if path == "local":
        _run_local_staged(
            out, pipeline_file, dim_int32_plate, "--bit-depth", "16", "--detect-mode", "LabL"
        )
    else:
        _run_slurm_workers(
            out, pipeline_file, dim_int32_plate, bit_depth=16, detect_mode="LabL"
        )

    _assert_store_matches(out, reference, table)
    stored = GridImage.load_zarr(zarr_store_path(out, DATASET, STEM))
    assert stored.detect_mode == "LabL"
