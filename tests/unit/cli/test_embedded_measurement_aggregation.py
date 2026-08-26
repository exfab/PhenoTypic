"""Aggregation semantics for metadata already joined into embedded tables."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from phenotypic._cli import _cli_recompile_worker
from phenotypic._cli._cli_output_manager import (
    _consistent_embedded_join_keys,
    _remap_to_scratch,
    _stage_to_scratch,
    finalize_post_master_outputs,
)
from phenotypic.schema import IMAGE, METADATA_MATCH, OBJECT, SIZE
from phenotypic.sdk_ import (
    PreparedEmbeddedMeasurementTable,
    measurements_parquet_path,
    write_embedded_measurement_table,
)


def test_finalize_appends_metadata_only_rows_once_without_rejoining_master(
    tmp_path: Path,
) -> None:
    """The master is already joined; only the metadata anti-join belongs here."""
    assert (
        "metadata_join_keys"
        in inspect.signature(finalize_post_master_outputs).parameters
    )
    image_name = str(IMAGE.IMAGE_NAME)
    metadata_only = str(METADATA_MATCH.METADATA_ONLY)
    object_label = str(OBJECT.LABEL)
    area = str(SIZE.AREA)
    master = pl.DataFrame(
        {
            image_name: ["plate-1.tiff"],
            "Metadata_Strain": ["WT"],
            object_label: [1],
            area: [12.5],
        }
    )
    metadata_csv = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            image_name: ["plate-1.tiff", "plate-2.tiff"],
            "Metadata_Strain": ["WT", "mutant"],
        }
    ).write_csv(metadata_csv)

    mirror = finalize_post_master_outputs(
        tmp_path,
        master,
        pipeline=None,
        metadata_csv=metadata_csv,
        metadata_join_keys=(image_name,),
        no_qc=True,
    )

    assert master.height == 1
    assert mirror.height == 2
    measured = mirror.filter(~pl.col(metadata_only))
    phantom = mirror.filter(pl.col(metadata_only))
    assert measured.select(image_name, "Metadata_Strain").row(0) == (
        "plate-1.tiff",
        "WT",
    )
    assert phantom.select(image_name, "Metadata_Strain").row(0) == (
        "plate-2.tiff",
        "mutant",
    )
    assert phantom[object_label].null_count() == 1
    assert phantom[area].null_count() == 1
    assert (
        pl.read_parquet(measurements_parquet_path(tmp_path))[
            metadata_only
        ].sum()
        == 1
    )


def test_recompile_finalizer_reads_provenance_after_shard_rewrite(
    tmp_path: Path,
) -> None:
    """Finalization must derive keys from rewritten stores, not submit-time state."""
    table_path = (
        tmp_path / "store.ome.zarr" / "tables/measurements/table.parquet"
    )
    observed: dict[str, object] = {}

    def capture_finalize(*args: object, **kwargs: object) -> pl.DataFrame:
        observed.update(kwargs)
        return args[1]  # type: ignore[return-value]

    with (
        patch(
            "phenotypic._cli._cli_output_manager._consistent_embedded_join_keys",
            return_value=("Metadata_ImageName",),
        ) as consistent,
        patch(
            "phenotypic._cli._cli_output_manager._load_pipeline_from_output_dir",
            return_value=None,
        ),
        patch(
            "phenotypic._cli._cli_output_manager.finalize_post_master_outputs",
            side_effect=capture_finalize,
        ),
    ):
        _cli_recompile_worker._run_post_master_steps(
            tmp_path,
            {
                "measurement_sources": [str(table_path)],
                "metadata_join_keys": [],
            },
            pl.DataFrame({"Metadata_ImageName": ["plate.tiff"]}),
        )

    consistent.assert_called_once_with([table_path])
    assert observed["metadata_join_keys"] == ("Metadata_ImageName",)


def test_aggregation_rejects_mixed_embedded_metadata_generations(
    tmp_path: Path,
) -> None:
    """A finalizer must not publish stores with different metadata digests."""
    table_paths: list[Path] = []
    for index, digest in enumerate(("a" * 64, "b" * 64)):
        store = tmp_path / f"image-{index}.ome.zarr"
        prepared = PreparedEmbeddedMeasurementTable(
            frame=pd.DataFrame(
                {"Metadata_ImageName": [f"plate-{index}.tiff"]}
            ),
            measurement_columns=("Metadata_ImageName",),
            join_status="joined",
            join_keys=("Metadata_ImageName",),
            metadata_snapshot_sha256=digest,
        )
        table_paths.append(write_embedded_measurement_table(store, prepared))

    with pytest.raises(
        ValueError,
        match="mixed metadata digests or join keys",
    ):
        _consistent_embedded_join_keys(table_paths)


def test_scratch_staging_keeps_embedded_table_sources_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedded tables with fixed relative names must not overwrite in scratch."""
    source_paths: list[Path] = []
    for index in range(2):
        table = (
            tmp_path
            / "results"
            / "dataset"
            / "zarr"
            / f"image-{index}.ome.zarr"
            / "tables"
            / "measurements"
            / "table.parquet"
        )
        table.parent.mkdir(parents=True)
        pl.DataFrame({"source": [index]}).write_parquet(table)
        source_paths.append(table)

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("SCRATCH", str(scratch))

    staged = _stage_to_scratch(source_paths)

    assert staged is not None
    remapped = _remap_to_scratch(
        {source: "dataset" for source in source_paths}, staged
    )
    assert len(remapped) == 2
    assert sorted(
        pl.read_parquet(path)["source"].item() for path in remapped
    ) == [0, 1]
