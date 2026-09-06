"""Recompile rewrites embedded tables rather than rejoining an aggregate."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path

import polars as pl
import pytest

from phenotypic._cli._cli_completion import valid_image_success
from phenotypic.schema import IMAGE
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    image_completion_marker_path,
    zarr_store_path,
)


def _pixel_digest(store: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((store / "rgb").rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(store).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "--mode recompile still reads image_completion_marker_path "
        "(_cli_recompile_recovery.py:52,387,477,637,709 and "
        "_cli_recompile_slurm_scripts.py:557), which D1's clean break stopped "
        "writing. Deferred to P4 by user ruling; this is the same deferral as "
        "the 28 marks in test_cli_recompile{,_slurm}.py, carried here because "
        "the file has no shared marker of its own."
    ),
)
def test_recompile_replaces_each_embedded_table_and_refreshes_marker(
    tmp_path: Path,
) -> None:
    """New metadata reaches stores first and marker authority is republished last."""
    module_name = "phenotypic._cli._cli_recompile_tables"
    assert importlib.util.find_spec(module_name) is not None, (
        "embedded-table recompile phase is missing"
    )
    recompile_tables = importlib.import_module(module_name)

    from click.testing import CliRunner
    from skimage.io import imsave

    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_process_single import main
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize

    input_root = tmp_path / "input"
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
            "input",
            "--input-root",
            str(input_root),
            "--no-save-overlays",
        ],
    )
    assert result.exit_code == 0, result.output
    store = zarr_store_path(output, "input", "plate")
    pixels_before = _pixel_digest(store)
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): ["plate"],
            "Strain": ["mutant"],
        }
    ).write_csv(metadata)

    changed = recompile_tables.recompile_embedded_measurement_tables(
        output, metadata
    )

    assert changed == 1
    table = pl.read_parquet(store / MEASUREMENT_TABLE_RELATIVE_PATH)
    assert table["Metadata_Strain"].to_list() == ["mutant"] * table.height
    assert _pixel_digest(store) == pixels_before
    marker = __import__("json").loads(
        image_completion_marker_path(output, "input", "plate").read_text()
    )
    assert valid_image_success(
        output,
        dataset="input",
        image_stem="plate",
        work_id=marker["work_id"],
    )
