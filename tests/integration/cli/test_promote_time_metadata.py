"""End-to-end: a promoted store is joinable by a third party (D-A, P4 Task 5).

D-A cut ``finalize_run``'s seventh step -- *"backfill
``pht-metadata.parquet`` per store, certified re-promote"* -- on the grounds
that the metadata table can be written at **promote time**, in the same
``.part`` as the measurements and before the root ``zarr.json``. That is the
whole justification, and it is only worth anything if the store that comes out
of a **real run** is self-describing without any post-hoc rewrite.

So the assertion path here imports no ``phenotypic`` code. It opens the two
Parquets with plain pyarrow, reads the join keys out of the metadata table's
own Arrow schema metadata, and joins on them -- which is what a napari, QuPath
or ``bioformats2raw`` user has available. A test that asked PhenoTypic to read
its own store back would prove the round trip and nothing about
"self-describing".
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.schema import IMAGE

#: The one image ``synth_plate_dir`` writes, and the strain it is given.
_IMAGE_STEM = "plate_001"
_STRAIN = "Säccharomyces"


def _run_full_pipeline(
    tmp_path: Path,
    synth_plate_dir: Path,
    simple_pipeline_json: Path,
    *,
    metadata: bool,
) -> Path:
    """Drive the real CLI over one synth plate and return the output root.

    Deliberately the shipped entry point rather than ``finalize_run`` or
    ``prepare_image_tables`` directly: the property under test is that the
    **shipped** forward path leaves a joinable store, and every layer between
    ``phenotypic_cli`` and ``write_image_tables`` is a place that could drop
    the metadata table without any unit test noticing.
    """
    out = tmp_path / "out"
    args = [
        "--pipeline",
        str(simple_pipeline_json),
        "--input",
        str(synth_plate_dir),
        "--output",
        str(out),
        "--force-local",
        "--skip-validation",
        "--njobs",
        "1",
    ]
    if metadata:
        source = tmp_path / "meta.csv"
        source.write_text(
            f"{IMAGE.IMAGE_NAME},Metadata_Strain\n{_IMAGE_STEM},{_STRAIN}\n",
            encoding="utf-8",
        )
        args += ["--metadata", str(source)]

    result = CliRunner().invoke(phenotypic_cli, args)
    assert result.exit_code == 0, result.output
    return out


def test_a_real_run_leaves_stores_a_third_party_can_join(
    tmp_path: Path, synth_plate_dir: Path, simple_pipeline_json: Path
) -> None:
    """D-A's whole justification: the store is self-describing WITHOUT any
    post-hoc rewrite. Read it back with plain pyarrow -- no phenotypic import
    in the assertion path -- and join it, the way a napari or QuPath user would.
    """
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    output = _run_full_pipeline(
        tmp_path, synth_plate_dir, simple_pipeline_json, metadata=True
    )
    stores = sorted(output.rglob("*.ome.zarr"))
    assert len(stores) == 1, f"expected exactly one store, got {stores}"
    store = stores[0]

    measurements_path = store / "tables" / "measurements" / "table.parquet"
    metadata_path = store / "tables" / "metadata" / "pht-metadata.parquet"
    assert measurements_path.is_file(), (
        "the promoted store carries no measurement table"
    )
    assert metadata_path.is_file(), (
        "the promoted store carries no metadata table -- promote-time writing "
        "is exactly what D-A traded the backfill step for"
    )

    measurements = pq.read_table(measurements_path)
    metadata = pq.read_table(metadata_path)

    # The join keys come off the metadata table's OWN Arrow metadata. A reader
    # with only these two files and no PhenoTypic must be able to find them.
    schema_metadata = metadata.schema.metadata or {}
    assert b"phenotypic.join.keys" in schema_metadata, (
        "pht-metadata.parquet does not name its join keys, so a third party "
        "has no way to join it"
    )
    keys = json.loads(schema_metadata[b"phenotypic.join.keys"])
    assert keys, (
        "the recorded join key list is empty; the merge below would be a "
        "cross product, not a join"
    )

    # STANDING RULE. `assert "Metadata_Strain" in joined.columns` is satisfied
    # by a join over zero rows, and by a metadata table that carried the column
    # and matched nothing. Establish that both sides have rows first, then that
    # the VALUE arrived on the measured rows.
    assert measurements.num_rows > 0, "the store measured no objects"
    assert metadata.num_rows > 0, "the store's metadata table is empty"

    joined = measurements.to_pandas().merge(
        metadata.to_pandas(), on=keys, how="left"
    )
    assert len(joined) == measurements.num_rows, (
        "the join changed the row count, so it fanned out or dropped rows"
    )
    assert "Metadata_Strain" in joined.columns
    assert joined["Metadata_Strain"].notna().all(), (
        "the join produced nulls; the recorded keys do not actually match"
    )
    assert set(joined["Metadata_Strain"]) == {_STRAIN}


def test_the_measurement_table_carries_no_user_metadata(
    tmp_path: Path, synth_plate_dir: Path, simple_pipeline_json: Path
) -> None:
    """The other half of the inversion, on a real run.

    ``pht-metadata.parquet`` existing is not by itself evidence the tables are
    inverted -- a producer that wrote both the metadata table *and* the joined
    measurements would pass the test above. What makes the store inverted is
    that ``Metadata_Strain`` appears on exactly one of the two files.
    """
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    output = _run_full_pipeline(
        tmp_path, synth_plate_dir, simple_pipeline_json, metadata=True
    )
    store = sorted(output.rglob("*.ome.zarr"))[0]
    measurements = pq.read_table(
        store / "tables" / "measurements" / "table.parquet"
    )
    metadata = pq.read_table(
        store / "tables" / "metadata" / "pht-metadata.parquet"
    )

    # STANDING RULE: the negative below is satisfied by a store whose metadata
    # table was never written with the column either. Establish where it IS.
    assert "Metadata_Strain" in metadata.schema.names, (
        "the metadata table does not carry the user column; the assertion "
        "below would pass on a run that never joined anything"
    )
    assert "Metadata_Strain" not in measurements.schema.names


def test_process_mode_skips_finalization_entirely(
    tmp_path: Path, synth_one_level_input: Path, simple_pipeline_json: Path
) -> None:
    """§7.4's table: ``process`` writes one layer, measures nothing.

    ``process_only_layer`` short-circuits both the aggregate proof and the
    ``aggregate_master_csv`` call, so no master is written and no finalization
    happens at all.
    """
    out = tmp_path / "out"
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_one_level_input),
            "--output",
            str(out),
            "--mode",
            "process",
            "--layer",
            "gray",
            "--force-local",
            "--skip-validation",
            "--njobs",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output

    # STANDING RULE. The negative below is satisfied by a `process` run that
    # ERRORED early and wrote nothing at all -- which is a different fact from
    # "process ran and correctly skipped finalization". Establish that it did
    # its own work first.
    assert list(out.rglob("*.ome.zarr")), (
        "process produced no store; the absence of a master proves nothing"
    )

    assert not (out / "deliverables" / "master_measurements.parquet").exists()
    assert not (out / "deliverables" / "master_measurements.csv").exists()
    assert not (out / "deliverables" / "measurements.parquet").exists()
