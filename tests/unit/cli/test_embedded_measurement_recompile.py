"""Recompile rewrites embedded tables rather than rejoining an aggregate."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import shutil
from pathlib import Path

import polars as pl

from phenotypic._cli._cli_completion import valid_image_success
from phenotypic.schema import IMAGE
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    image_record_path,
    zarr_store_path,
)


def _pixel_digest(store: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((store / "rgb").rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(store).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


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
    # The RECORD. `_cli_process_single.main` above is the forward publisher,
    # and D1's clean break moved what it writes out of `image_complete/`, so
    # the legacy path is simply absent on the tree this test just built.
    record = __import__("json").loads(
        image_record_path(output, "input", "plate").read_text(
            encoding="utf-8"
        )
    )
    assert valid_image_success(
        output,
        dataset="input",
        image_stem="plate",
        work_id=record["work_id"],
    )


def _build_store(
    *,
    pipeline_path: Path,
    input_root: Path,
    image_path: Path,
    output: Path,
) -> None:
    """Drive the real forward worker for one image, asserting it succeeded."""
    from click.testing import CliRunner

    from phenotypic._cli._cli_process_single import main

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


def _table_sha256(output: Path, stem: str) -> str:
    path = (
        zarr_store_path(output, "input", stem) / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_mixed_tree_is_refused_before_any_store_is_rewritten(
    tmp_path: Path,
) -> None:
    """The pre-flight scan, and the case the per-store guard cannot cover.

    ``_refuse_inverted_store`` is correctly ordered *within* a store, so a
    uniform ``--metadata`` tree fails on the first store having destroyed
    nothing. A **mixed** tree is different: absent a whole-tree scan,
    recompile rewrites the un-inverted stores it reaches first and only then
    hits an inverted one, leaving exactly the mixed Parquet generations
    ``recompile_embedded_measurement_tables`` documents as an interruption
    hazard -- except caused by the guard rather than by an interruption.

    **The tree is built by the real vector, not by hand-editing a root.**
    ``prepare_image_tables`` emits a metadata table only when
    ``deliverables/metadata.csv`` exists as the image is written, and
    ``metadata_csv`` is in neither ``processing_configuration_digest`` nor
    ``compute_work_id`` -- so a run begun without ``--metadata`` and resumed
    with it keeps every finished image's un-inverted store (their work ids did
    not change, so continuation does not reprocess them) and inverts only the
    new ones. That is what the two invocations below reproduce.

    **The control is what stops a do-nothing scan from passing.** Phase 3
    re-runs the identical call on the identical tree with only the inverted
    store removed, and requires ``aaa``'s bytes to change. So "unchanged" in
    phase 2 is a fact about the scan, not about a store recompile would have
    left alone anyway:

    * a scan that does nothing fails phase 2's ``raises``;
    * a scan ordered after the first write fails phase 2's byte equality;
    * a scan that refuses every tree fails phase 3.
    """
    import json

    import pytest
    from skimage.io import imsave

    from phenotypic import ImagePipeline
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize
    from phenotypic.sdk_ import metadata_csv_deliverable_path
    from phenotypic.sdk_.ngff_ import METADATA_TABLE_GROUP, PhenotypicAttr
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_tables,
    )

    input_root = tmp_path / "input"
    input_root.mkdir()
    pixels = load_synth_yeast_plate().rgb[:]
    # "aaa" sorts before "zzz", and the rewrite loop iterates sorted by table
    # path -- so absent the scan, aaa is rewritten BEFORE zzz is refused.
    for stem in ("aaa", "zzz"):
        imsave(
            str(input_root / f"{stem}.tiff"), pixels, check_contrast=False
        )
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        ImagePipeline(ops=[OtsuDetector()], meas=[MeasureSize()]).to_json(),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    # Phase 1a: the run BEFORE --metadata -- aaa's store is un-inverted.
    _build_store(
        pipeline_path=pipeline_path,
        input_root=input_root,
        image_path=input_root / "aaa.tiff",
        output=output,
    )

    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): ["aaa", "zzz"],
            "Strain": ["mutant", "wildtype"],
        }
    ).write_csv(metadata)

    # Phase 1b: the resume WITH --metadata -- zzz's store is inverted.
    snapshot = metadata_csv_deliverable_path(output)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(metadata.read_bytes())
    _build_store(
        pipeline_path=pipeline_path,
        input_root=input_root,
        image_path=input_root / "zzz.tiff",
        output=output,
    )

    def _tables(stem: str) -> dict:
        root = json.loads(
            (zarr_store_path(output, "input", stem) / "zarr.json").read_text(
                encoding="utf-8"
            )
        )
        return root["attributes"][PhenotypicAttr.ROOT][PhenotypicAttr.TABLES]

    # The tree really is MIXED. Without this, everything below could hold on a
    # tree that was uniform in either direction.
    assert METADATA_TABLE_GROUP not in _tables("aaa")
    assert METADATA_TABLE_GROUP in _tables("zzz")

    before = _table_sha256(output, "aaa")

    # Phase 2: the whole run is refused, and aaa is untouched.
    with pytest.raises(RuntimeError, match="inverted"):
        recompile_embedded_measurement_tables(output, metadata)

    assert _table_sha256(output, "aaa") == before, (
        "the scan let recompile rewrite a store before refusing the tree"
    )

    # Phase 3, the control: same call, same tree, inverted store removed.
    # If this does not rewrite aaa, phase 2 proved nothing.
    shutil.rmtree(zarr_store_path(output, "input", "zzz"))
    image_record_path(output, "input", "zzz").unlink()

    assert recompile_embedded_measurement_tables(output, metadata) == 1
    assert _table_sha256(output, "aaa") != before, (
        "recompile would not have rewritten aaa anyway -- phase 2 is vacuous"
    )
    table = pl.read_parquet(
        zarr_store_path(output, "input", "aaa")
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    assert table["Metadata_Strain"].to_list() == ["mutant"] * table.height
