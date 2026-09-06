"""Aggregation semantics after §7.3 moved the metadata join to finalization."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from phenotypic._cli import _cli_recompile_worker
from phenotypic._cli._cli_output_manager import (
    _remap_to_scratch,
    _stage_to_scratch,
    finalize_post_master_outputs,
)
from phenotypic.schema import IMAGE, METADATA_MATCH, OBJECT, SIZE
from phenotypic.sdk_ import measurements_parquet_path


def test_finalize_joins_measured_rows_and_appends_phantoms_in_one_call(
    tmp_path: Path,
) -> None:
    """P4 §7.3: ONE ``join_metadata`` call, both halves.

    Rewritten from ``test_finalize_appends_metadata_only_rows_once_without_
    rejoining_master``, which asserted ``metadata_join_keys`` was in the
    signature and exercised the append-phantoms branch. That branch's premise
    -- "measured rows already carry their publication-time metadata from the
    embedded tables" -- is exactly what the inversion falsified, so the
    parameter and the branch are gone. The frame handed in here is now
    UN-joined, which is what a post-inversion master is.
    """
    assert (
        "metadata_join_keys"
        not in inspect.signature(finalize_post_master_outputs).parameters
    ), "the retired parameter is still on the signature"

    image_name = str(IMAGE.IMAGE_NAME)
    metadata_only = str(METADATA_MATCH.METADATA_ONLY)
    object_label = str(OBJECT.LABEL)
    area = str(SIZE.AREA)
    master = pl.DataFrame(
        {
            image_name: ["plate-1.tiff"],
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
        no_qc=True,
    )

    assert master.height == 1
    assert "Metadata_Strain" not in master.columns, (
        "the master under test is not un-joined; this is the pre-inversion "
        "shape and the join below proves nothing"
    )
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


def test_recompile_finalizer_no_longer_carries_submit_time_join_keys(
    tmp_path: Path,
) -> None:
    """CAN-2: the recompile finalizer derives nothing from recorded keys.

    Rewritten from ``test_recompile_finalizer_reads_provenance_after_shard_
    rewrite``. Re-reading the stores' provenance after the shards were
    rewritten was the right fix for a design where the finalizer needed those
    keys; after P4 it needs none, and the ``measurement_sources`` /
    ``metadata_join_keys`` split in ``_run_post_master_steps`` goes with them.
    What survives is the mixed-AUTHORITY refusal (H6).
    """
    table_path = (
        tmp_path / "store.ome.zarr" / "tables/measurements/table.parquet"
    )
    observed: dict[str, object] = {}

    def capture_finalize(*args: object, **kwargs: object) -> pl.DataFrame:
        observed.update(kwargs)
        return args[1]  # type: ignore[return-value]

    with (
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
            {"measurement_sources": [str(table_path)]},
            pl.DataFrame({"Metadata_ImageName": ["plate.tiff"]}),
        )

    assert observed, "the finalizer was never called; the negative is vacuous"
    assert "metadata_join_keys" not in observed


def test_the_recompile_finalizer_still_refuses_mixed_authority(
    tmp_path: Path,
) -> None:
    """H6: retiring the mixed-GENERATION guard must not retire this one.

    Replaces ``test_aggregation_rejects_mixed_embedded_metadata_generations``.
    That test pinned the refusal D-A deliberately manufactures -- stores keep
    the snapshot they were built against, so mixed digests are the NORMAL
    rolling-input state and a finalizer that aborts on them is the defect.
    Its tolerance is pinned in ``test_finalize_run.py``
    (``test_stores_with_mixed_metadata_snapshots_do_not_abort_finalization``);
    what must still raise is a mixture of embedded and legacy AUTHORITY.
    """
    embedded = tmp_path / "store.ome.zarr" / "tables/measurements/table.parquet"
    legacy = tmp_path / "results" / "plate" / "measurements" / "b.parquet"

    with pytest.raises(
        ValueError, match="mixed embedded and legacy measurement authority"
    ):
        _cli_recompile_worker._run_post_master_steps(
            tmp_path,
            {"measurement_sources": [str(embedded), str(legacy)]},
            pl.DataFrame({"Metadata_ImageName": ["plate.tiff"]}),
        )


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


def test_embedded_filename_recovers_distinct_missing_image_names() -> None:
    """Fixed table filenames must derive identity from their owning stores."""
    from phenotypic._cli._measurement_sources import (
        add_metadata_image_name_from_filename,
    )

    frame = pl.DataFrame(
        {
            "filename": [
                "/run/results/ds/zarr/alpha.ome.zarr/tables/measurements/table.parquet",
                "/run/results/ds/zarr/beta.ome.zarr/tables/measurements/table.parquet",
            ],
            str(SIZE.AREA): [1.0, 2.0],
        }
    )

    recovered = add_metadata_image_name_from_filename(frame)

    assert recovered[str(IMAGE.IMAGE_NAME)].to_list() == ["alpha", "beta"]
    assert "filename" not in recovered.columns


def test_embedded_filename_repairs_uuid_image_name_from_store() -> None:
    """A UUID fallback must be replaced by the owning embedded store stem."""
    from phenotypic._cli._measurement_sources import (
        add_metadata_image_name_from_filename,
    )

    frame = pl.DataFrame(
        {
            "filename": [
                "/run/results/ds/zarr/alpha.ome.zarr/tables/measurements/table.parquet"
            ],
            str(IMAGE.IMAGE_NAME): ["123e4567-e89b-12d3-a456-426614174000"],
        }
    )

    recovered = add_metadata_image_name_from_filename(frame)

    assert recovered[str(IMAGE.IMAGE_NAME)].to_list() == ["alpha"]
