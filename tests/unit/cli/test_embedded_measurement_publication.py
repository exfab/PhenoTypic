"""Forward-path authority for embedded measurement tables."""

from __future__ import annotations

import json
from pathlib import Path

from phenotypic._cli._cli_completion import valid_image_success
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_types import Dataset
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    image_completion_marker_path,
    zarr_store_path,
)


def test_new_run_structure_does_not_create_external_measurements_dir(
    tmp_path: Path,
) -> None:
    """Provisioning the legacy directory would preserve a split authority."""
    manager = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    manager.create_structure([Dataset("ds", [], tmp_path, tmp_path)])

    assert (tmp_path / "results" / "ds" / "zarr").is_dir()
    assert not (tmp_path / "results" / "ds" / "measurements").exists()


def test_single_pass_writes_only_the_embedded_measurement_table(
    tmp_path: Path,
) -> None:
    """Writing the old per-image Parquet would reintroduce dual authority."""
    from skimage.io import imsave

    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_process_single import process_single_image_core
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize

    image_path = tmp_path / "plate.tiff"
    imsave(
        str(image_path), load_synth_yeast_plate().rgb[:], check_contrast=False
    )
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        ImagePipeline(ops=[OtsuDetector()], meas=[MeasureSize()]).to_json(),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    process_single_image_core(
        pipeline_path,
        image_path,
        output,
        "ds",
        "Image",
        {},
        OutputManager.from_config(output, ".tiff", save_overlays=False),
    )

    store = zarr_store_path(output, "ds", "plate")
    assert (store / MEASUREMENT_TABLE_RELATIVE_PATH).is_file()
    assert not (output / "results" / "ds" / "measurements").exists()
    assert not list(output.rglob("plate.parquet"))


def test_marker_binds_the_embedded_table_hash(tmp_path: Path) -> None:
    """A marker that hashes only root zarr.json accepts a corrupted table."""
    from click.testing import CliRunner
    from skimage.io import imsave

    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_process_single import main
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize

    input_root = tmp_path / "in"
    input_root.mkdir()
    image_path = input_root / "plate.tiff"
    imsave(
        str(image_path), load_synth_yeast_plate().rgb[:], check_contrast=False
    )
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        ImagePipeline(ops=[OtsuDetector()], meas=[MeasureSize()]).to_json(),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        [
            "--pipeline",
            str(pipeline_path),
            "--image",
            str(image_path),
            "--output-dir",
            str(output),
            "--dataset-name",
            "in",
            "--input-root",
            str(input_root),
            "--no-save-overlays",
        ],
    )
    assert result.exit_code == 0, result.output

    marker = json.loads(
        image_completion_marker_path(output, "in", "plate").read_text(
            encoding="utf-8"
        )
    )
    table_descriptor = marker["artifacts"]["measurements"]
    assert table_descriptor["path"].endswith(
        "plate.ome.zarr/tables/measurements/table.parquet"
    )
    assert table_descriptor["kind"] == "file"
    work_id = marker["work_id"]
    assert valid_image_success(
        output, dataset="in", image_stem="plate", work_id=work_id
    )

    table_path = output / table_descriptor["path"]
    table_path.write_bytes(b"corrupt")
    assert not valid_image_success(
        output, dataset="in", image_stem="plate", work_id=work_id
    )
