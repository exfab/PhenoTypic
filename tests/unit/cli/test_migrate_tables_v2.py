"""#281 P3 Task 5, cluster C15: v1 stores are refused forward and converted by migrate.

Spec D5 and D10. A store in the v1 flat layout (``tables/measurements/``,
``tables/metadata/``) is refused by every forward mode, before anything is
written, with a pointer at ``--mode migrate``; migrate moves its tables into
``tables/1-<sha12>/`` (or ``1-unknown``) and republishes the image's record and
then the run proof, so a tree that read ``complete`` still does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import zarr

from phenotypic._cli._cli_tables_v1 import first_v1_store
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    METADATA_TABLE_RELATIVE_PATH,
    PreparedEmbeddedMeasurementTable,
    convert_tables_v1_to_v2,
    list_table_runs,
    replace_embedded_measurement_table,
    resolve_run_state,
    store_tables_schema,
    v1_table_pipeline_sha,
)
from tests._tiny_run import one_store, run_cli, tree_digest

SHA_A = "a" * 64
SHA_B = "b" * 64

#: The digest the committed v1 fixture's journal records: the bytes of the
#: `tiny_pipeline` file the generator ran (`tests/fixtures/tables_v1`).
_V1_FIXTURE_SHA = "ae2757956880293d6534d21cebb9358ca7a309b95e92e704ac174e619750747f"


def journal_with_full_application(sha: str) -> dict:
    """A schema-2 provenance block with one ``full`` application of pipeline *sha*."""
    return {
        "schema_version": 2,
        "status": "complete",
        "applications": [
            {
                "sequence": 1,
                "kind": "full",
                "status": "complete",
                "pipeline": {"source_path": "p.pht-pipe.json", "sha256": sha},
                "operations": [],
            }
        ],
    }


def prepared_from(store: Path) -> PreparedEmbeddedMeasurementTable:
    """A prepared payload rebuilt from *store*'s own declared measurement table."""
    from phenotypic._cli._embedded_measurement_tables import (
        prepare_embedded_measurement_table,
    )
    from phenotypic.sdk_ import declared_measurement_table_path

    payload = declared_measurement_table_path(store)
    assert payload is not None and payload.is_file()
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    tables = root["attributes"]["phenotypic"]["tables"]
    descriptor = (
        tables["measurements"]
        if "schema_version" not in tables
        else list(tables["runs"].values())[-1]["measurements"]
    )
    frame = pq.read_table(payload).to_pandas()
    return prepare_embedded_measurement_table(
        frame.loc[:, descriptor["measurement_columns"]], None
    )


# ---------------------------------------------------------------------------
# D5: the forward refusal
# ---------------------------------------------------------------------------


def test_forward_run_refuses_a_v1_store(v1_tree, tiny_inputs):
    # Review Focus 1
    before = tree_digest(v1_tree)
    result = run_cli(tiny_inputs, v1_tree)
    assert result.exit_code == 2 and "--mode migrate" in result.output
    assert str(one_store(v1_tree)) in result.output
    assert tree_digest(v1_tree) == before  # refused before anything is written


@pytest.mark.parametrize("mode", ["measure", "recompile"])
def test_every_forward_mode_refuses_a_v1_store(v1_tree, mode):
    before = tree_digest(v1_tree)
    result = run_cli(None, v1_tree, mode=mode)
    assert result.exit_code == 2 and "--mode migrate" in result.output
    assert tree_digest(v1_tree) == before


def test_first_v1_store_finds_the_v1_store_and_nothing_in_a_v2_tree(
    v1_tree, finished_tiny_run
):
    assert first_v1_store(v1_tree) == one_store(v1_tree)
    assert first_v1_store(finished_tiny_run) is None
    assert first_v1_store(v1_tree / "absent") is None


def test_the_run_state_reader_does_not_report_a_v1_tree_as_needing_migration(v1_tree):
    """D5: the check is CLI-only; `datasets_needing_migration` is untouched."""
    from phenotypic.sdk_ import datasets_needing_migration

    assert datasets_needing_migration(v1_tree) == []


# ---------------------------------------------------------------------------
# D10: the conversion
# ---------------------------------------------------------------------------


def test_migrate_converts_and_is_idempotent(v1_store):
    run = convert_tables_v1_to_v2(v1_store, durable=False)
    assert run.run_id.startswith("1-") and run.measurements_path().is_file()
    assert store_tables_schema(v1_store) == 2
    assert convert_tables_v1_to_v2(v1_store, durable=False) is None


def test_conversion_names_the_journal_pipeline_and_moves_every_table(v1_store):
    payload = (v1_store / MEASUREMENT_TABLE_RELATIVE_PATH).read_bytes()
    inode = (v1_store / MEASUREMENT_TABLE_RELATIVE_PATH).stat().st_ino
    run = convert_tables_v1_to_v2(v1_store, durable=False)
    assert run.run_id == f"1-{_V1_FIXTURE_SHA[:12]}"
    assert run.pipeline_sha256 == _V1_FIXTURE_SHA
    # Bytes linked, never rewritten; the flat folders are gone.
    assert run.measurements_path().read_bytes() == payload
    assert run.measurements_path().stat().st_ino == inode
    assert not (v1_store / MEASUREMENT_TABLE_RELATIVE_PATH).exists()
    assert not (v1_store / METADATA_TABLE_RELATIVE_PATH).exists()
    root = json.loads((v1_store / "zarr.json").read_text(encoding="utf-8"))
    tables = root["attributes"]["phenotypic"]["tables"]
    assert list(tables["runs"]) == [run.run_id]
    assert tables["runs"][run.run_id]["measurements"]["path"] == (
        f"tables/{run.run_id}/measurements/table.parquet"
    )
    assert "metadata_table" not in root["attributes"]["phenotypic"]


def test_the_converted_store_reads_with_independent_readers(v1_store):
    run = convert_tables_v1_to_v2(v1_store, durable=False)
    group = zarr.open_group(v1_store, mode="r")
    attrs = group["tables"][run.run_id]["measurements"].attrs
    assert attrs["type"] == "feature_table"
    assert attrs["region"] == {"path": "../../rgb/labels/objmap"}
    assert pq.read_table(run.measurements_path()).num_rows > 0


def test_the_converted_store_passes_the_table_contract(v1_store):
    from phenotypic.sdk_._measurement_tables import (
        _valid_embedded_measurement_contract,
    )

    assert _valid_embedded_measurement_contract(v1_store)
    convert_tables_v1_to_v2(v1_store, durable=False)
    assert _valid_embedded_measurement_contract(v1_store)


def test_unknown_digest_is_never_fabricated(v1_store_legacy_null_pipeline):
    run = convert_tables_v1_to_v2(v1_store_legacy_null_pipeline, durable=False)
    assert run.run_id == "1-unknown" and run.pipeline_sha256 is None


def test_the_figure_run_sha_wins_over_the_journal():
    block = {
        "figures": {"runs": {"2026-11-01-" + SHA_B[:12]: {"pipeline_sha256": SHA_B}}},
        "provenance": journal_with_full_application(SHA_A),
    }
    assert v1_table_pipeline_sha(block) == SHA_B


def test_the_most_recent_figure_run_wins():
    block = {
        "figures": {
            "runs": {
                "2026-11-02-" + SHA_A[:12]: {"date": "2026-11-02", "pipeline_sha256": SHA_A},
                "2026-10-30-" + SHA_B[:12]: {"date": "2026-10-30", "pipeline_sha256": SHA_B},
            }
        }
    }
    assert v1_table_pipeline_sha(block) == SHA_A


def test_without_figures_the_latest_measuring_application_names_the_table():
    journal = journal_with_full_application(SHA_A)
    journal["applications"].append(
        {"sequence": 2, "kind": "programmatic", "pipeline": None, "operations": []}
    )
    assert v1_table_pipeline_sha({"provenance": journal}) == SHA_A
    legacy = {"applications": [{"kind": "legacy", "pipeline": None}]}
    assert v1_table_pipeline_sha({"provenance": legacy}) is None


# ---------------------------------------------------------------------------
# D10: the migrate import writes v2
# ---------------------------------------------------------------------------


def test_the_migrate_import_writes_v2(v1_store):
    convert_tables_v1_to_v2(v1_store, durable=False)
    replace_embedded_measurement_table(v1_store, prepared_from(v1_store))
    assert not (v1_store / MEASUREMENT_TABLE_RELATIVE_PATH).exists()
    assert len(list_table_runs(v1_store)) == 1


def test_the_import_into_a_v1_store_converts_it_in_the_same_transaction(v1_store):
    written = replace_embedded_measurement_table(v1_store, prepared_from(v1_store))
    assert store_tables_schema(v1_store) == 2
    (run,) = list_table_runs(v1_store)
    assert written == run.measurements_path() and written.is_file()
    assert not (v1_store / MEASUREMENT_TABLE_RELATIVE_PATH).exists()


def test_the_import_keeps_every_other_run_folder_byte_identical(store_with_two_runs):
    first, second = list_table_runs(store_with_two_runs)
    before = tree_digest(store_with_two_runs / "tables" / first.run_id)
    written = replace_embedded_measurement_table(
        store_with_two_runs, prepared_from(store_with_two_runs)
    )
    # The journal names pipeline B, so the import owns B's folder.
    assert written == second.measurements_path()
    assert [r.run_id for r in list_table_runs(store_with_two_runs)] == [
        first.run_id,
        second.run_id,
    ]
    assert tree_digest(store_with_two_runs / "tables" / first.run_id) == before


# ---------------------------------------------------------------------------
# D10: the whole tree
# ---------------------------------------------------------------------------


def test_a_completed_v1_tree_reads_complete_after_migrate(v1_tree):
    assert resolve_run_state(v1_tree, depth="deep").completion == "complete"
    assert run_cli(None, v1_tree, mode="migrate").exit_code == 0
    assert store_tables_schema(one_store(v1_tree)) == 2
    assert resolve_run_state(v1_tree, depth="deep").completion == "complete"


def test_migrate_republishes_the_record_at_the_run_folder(v1_tree):
    record_path = v1_tree / ".phenotypic" / "progress" / "images" / "ds" / "plate.json"
    before = json.loads(record_path.read_text(encoding="utf-8"))
    assert run_cli(None, v1_tree, mode="migrate").exit_code == 0
    after = json.loads(record_path.read_text(encoding="utf-8"))
    (run,) = list_table_runs(one_store(v1_tree))
    assert after["artifacts"]["measurements"]["path"] == (
        f"results/ds/zarr/plate.ome.zarr/tables/{run.run_id}/measurements/table.parquet"
    )
    for field in ("work_id", "relative_image_path", "attempt_id", "mode"):
        assert after[field] == before[field]


def test_a_migrated_v1_tree_is_accepted_by_the_forward_check(v1_tree):
    assert run_cli(None, v1_tree, mode="migrate").exit_code == 0
    assert first_v1_store(v1_tree) is None


def test_a_direct_v1_store_target_is_converted(v1_store_legacy_null_pipeline):
    """The provenance-only store array carries the ``tables_v2`` step too."""
    result = run_cli(None, v1_store_legacy_null_pipeline, mode="migrate")
    assert result.exit_code == 0, result.output
    (run,) = list_table_runs(v1_store_legacy_null_pipeline)
    assert run.run_id == "1-unknown" and run.measurements_path().is_file()


def test_a_migrate_dry_run_converts_nothing(v1_tree):
    before = tree_digest(one_store(v1_tree))
    result = run_cli(None, v1_tree, mode="migrate", extra=["--dry-run"])
    assert result.exit_code == 0, result.output
    assert tree_digest(one_store(v1_tree)) == before
    assert store_tables_schema(one_store(v1_tree)) == 1
