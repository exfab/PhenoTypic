"""Forward-path authority for embedded measurement tables."""

from __future__ import annotations

import json
from pathlib import Path

from phenotypic._cli._cli_completion import valid_image_success
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_types import Dataset
from phenotypic.sdk_ import (
    image_record_path,
    latest_measurement_run,
    zarr_store_path,
)


def _store_table(store: Path) -> Path:
    """The store's measurements table, through the v2 resolver (#281 D6a)."""
    run = latest_measurement_run(store)
    assert run is not None, f"{store} has no measurement run"
    return run.measurements_path()


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
    assert _store_table(store).is_file()
    assert not (output / "results" / "ds" / "measurements").exists()
    assert not list(output.rglob("plate.parquet"))


def test_direct_store_source_uses_canonical_store_and_overlay_keys(
    tmp_path: Path,
) -> None:
    """Source ``p01.ome.zarr`` must not publish ``p01.ome`` artifacts."""
    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_process_single import process_single_image_core
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize

    source = tmp_path / "p01.ome.zarr"
    load_synth_yeast_plate().save2zarr(source)
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        ImagePipeline(ops=[OtsuDetector()], meas=[MeasureSize()]).to_json(),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    process_single_image_core(
        pipeline_path,
        source,
        output,
        "ds",
        "Image",
        {},
        OutputManager.from_config(output, ".tiff", save_overlays=True),
    )

    canonical_store = zarr_store_path(output, "ds", "p01")
    assert _store_table(canonical_store).is_file()
    assert (output / "deliverables" / "overlays" / "ds" / "p01.png").is_file()
    assert not zarr_store_path(output, "ds", "p01.ome").exists()
    assert not (
        output / "deliverables" / "overlays" / "ds" / "p01.ome.png"
    ).exists()


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
        image_record_path(output, "in", "plate").read_text(
            encoding="utf-8"
        )
    )
    table_descriptor = marker["artifacts"]["measurements"]
    # The run folder the save wrote, tables/<n>-<sha12>/ (#281 D6a).
    assert (output / table_descriptor["path"]) == _store_table(
        zarr_store_path(output, "in", "plate")
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
