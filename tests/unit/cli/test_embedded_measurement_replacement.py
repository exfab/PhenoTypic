"""Measure-only replacement of authoritative embedded tables."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    PhenotypicAttr,
    read_phenotypic_attributes,
    zarr_store_path,
)


def _tree_hashes(root: Path) -> dict[str, str]:
    """Hash every file below a pixel-array directory."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_test_image_and_pipelines(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write one synthetic plate and size/shape measurement pipelines."""
    from skimage.io import imsave

    from phenotypic import ImagePipeline
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureShape, MeasureSize

    image_path = tmp_path / "plate.tiff"
    imsave(
        str(image_path), load_synth_yeast_plate().rgb[:], check_contrast=False
    )
    initial_pipeline = tmp_path / "initial.json"
    initial_pipeline.write_text(
        ImagePipeline(ops=[OtsuDetector()], meas=[MeasureSize()]).to_json(),
        encoding="utf-8",
    )
    replacement_pipeline = tmp_path / "replacement.json"
    replacement_pipeline.write_text(
        ImagePipeline(meas=[MeasureShape()]).to_json(),
        encoding="utf-8",
    )
    return image_path, initial_pipeline, replacement_pipeline


def test_measure_only_atomically_replaces_table_without_rewriting_pixels(
    tmp_path: Path,
) -> None:
    """Remeasurement changes table authority but leaves pixel bytes untouched."""
    from phenotypic._cli._cli_process_single import (
        process_single_image_core,
        process_single_store_measure_core,
    )

    image_path, initial_pipeline, replacement_pipeline = (
        _write_test_image_and_pipelines(tmp_path)
    )
    output = tmp_path / "out"
    manager = OutputManager.from_config(output, ".tiff", save_overlays=False)
    process_single_image_core(
        initial_pipeline,
        image_path,
        output,
        "ds",
        "Image",
        {},
        manager,
    )
    store = zarr_store_path(output, "ds", "plate")
    before_pixels = _tree_hashes(store / "rgb")
    before_table = pq.read_table(store / MEASUREMENT_TABLE_RELATIVE_PATH)
    assert "Size_Area" in before_table.column_names

    process_single_store_measure_core(
        replacement_pipeline,
        store,
        output,
        "ds",
        "Image",
        manager,
    )

    after_table = pq.read_table(store / MEASUREMENT_TABLE_RELATIVE_PATH)
    assert "Shape_Circularity" in after_table.column_names
    assert "Size_Area" not in after_table.column_names
    assert _tree_hashes(store / "rgb") == before_pixels
    attrs = read_phenotypic_attributes(store)
    descriptor = attrs[PhenotypicAttr.TABLES]["measurements"]
    assert descriptor["measurement_columns"] == after_table.column_names
    assert not (output / "results" / "ds" / "measurements").exists()


def test_measure_failure_before_final_marker_leaves_new_table_unauthorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refreshed marker is the final successful per-image publication."""
    from phenotypic._cli._cli_completion import (
        publish_image_success,
        valid_image_success,
    )
    from phenotypic._cli._cli_process_single import (
        process_single_image_core,
        process_single_store_measure_core,
    )
    from phenotypic.plotting._pipeline import PlotCoordinator

    image_path, initial_pipeline, replacement_pipeline = (
        _write_test_image_and_pipelines(tmp_path)
    )
    output = tmp_path / "out"
    manager = OutputManager.from_config(output, ".tiff", save_overlays=False)
    process_single_image_core(
        initial_pipeline,
        image_path,
        output,
        "ds",
        "Image",
        {},
        manager,
    )
    store = zarr_store_path(output, "ds", "plate")
    table = store / MEASUREMENT_TABLE_RELATIVE_PATH
    publish_image_success(
        output,
        work_id="measure-order-work",
        dataset="ds",
        relative_image_path="ds/plate.tiff",
        image_stem="plate",
        mode="full",
        attempt_id="forward",
        lifecycle_epoch="epoch-1",
        artifacts={"measurements": table, "store": store},
    )
    assert valid_image_success(
        output,
        dataset="ds",
        image_stem="plate",
        work_id="measure-order-work",
    )

    def fail_plot(*args, **kwargs) -> None:
        raise RuntimeError("simulated post-table publication failure")

    monkeypatch.setattr(PlotCoordinator, "emit_image", fail_plot)
    with pytest.raises(
        RuntimeError, match="simulated post-table publication failure"
    ):
        process_single_store_measure_core(
            replacement_pipeline,
            store,
            output,
            "ds",
            "Image",
            manager,
        )

    assert not valid_image_success(
        output,
        dataset="ds",
        image_stem="plate",
        work_id="measure-order-work",
    ), "marker was refreshed before all per-image publication work completed"
