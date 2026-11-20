"""Run identity and the table-run resolver (#281 P3 Task 1, spec D2/D6/D6a)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic import Image
from phenotypic._cli._cli_failure_tracker import file_sha256
from phenotypic._core._provenance import (
    current_pipeline_sha256,
    initialize_cli_provenance,
)
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import (
    TABLE_RUN_ID_RE,
    choose_table_run_id,
    journal_measurement_table,
    journal_table_run,
    latest_measurement_run,
    list_table_runs,
    next_table_run_order,
    store_of_table_path,
    store_tables_schema,
    table_run_id,
)
from tests._tiny_run import owned_table_run, v2_block

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_P = "0123456789ab" + "f" * 52


def write_root_block(directory: Path, block: dict) -> Path:
    """Write a store root at ``<directory>/s.ome.zarr`` whose ``phenotypic`` block is *block*."""
    store = directory / "s.ome.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"phenotypic": block},
            }
        ),
        encoding="utf-8",
    )
    return store


def test_run_id_format():
    assert table_run_id(2, "a71e09c2b4f5" + "0" * 52) == "2-a71e09c2b4f5"
    assert table_run_id(3, None) == "3-unknown"
    assert TABLE_RUN_ID_RE.match("0-abc") is None


def test_run_id_refuses_a_fabricated_digest():
    with pytest.raises(ValueError):
        table_run_id(1, "not-a-digest")
    with pytest.raises(ValueError):
        table_run_id(0, SHA_A)


def test_choose_reuses_the_same_pipeline_folder():
    block = v2_block(runs={"2-" + SHA_A[:12]: {"order": 2, "pipeline_sha256": SHA_A}})
    assert choose_table_run_id(block, pipeline_sha256=SHA_A) == "2-" + SHA_A[:12]


def test_a_new_pipeline_gets_the_next_number():
    block = v2_block(
        runs={
            "1-" + SHA_A[:12]: {"order": 1, "pipeline_sha256": SHA_A},
            "2-" + SHA_B[:12]: {"order": 2, "pipeline_sha256": SHA_B},
        }
    )
    assert choose_table_run_id(block, pipeline_sha256=SHA_C) == "3-" + SHA_C[:12]


def test_upstream_entries_count_toward_the_next_number():
    block = v2_block(upstream={"1-" + SHA_P[:12]: {"pipeline_sha256": SHA_P}})
    assert choose_table_run_id(block, pipeline_sha256=SHA_A) == "2-" + SHA_A[:12]
    assert next_table_run_order(block) == 2


def test_an_unknown_sha_replaces_the_highest_unknown_folder():
    block = v2_block(
        runs={
            "1-unknown": {"order": 1, "pipeline_sha256": None},
            "2-" + SHA_A[:12]: {"order": 2, "pipeline_sha256": SHA_A},
            "3-unknown": {"order": 3, "pipeline_sha256": None},
        }
    )
    assert choose_table_run_id(block, pipeline_sha256=None) == "3-unknown"
    assert choose_table_run_id(v2_block(), pipeline_sha256=None) == "1-unknown"


def test_numbers_are_never_reused_after_a_gap():
    # Folder 2 is gone (only --overwrite deletes folders, but the counter must
    # not depend on that): a new pipeline still takes 4, never 2.
    block = v2_block(
        runs={
            "1-" + SHA_A[:12]: {"order": 1, "pipeline_sha256": SHA_A},
            "3-" + SHA_B[:12]: {"order": 3, "pipeline_sha256": SHA_B},
        }
    )
    assert choose_table_run_id(block, pipeline_sha256=SHA_C) == "4-" + SHA_C[:12]


def test_two_runs_with_one_number_are_refused(tmp_path):
    store = write_root_block(
        tmp_path,
        v2_block(
            runs={"1-" + SHA_A[:12]: {"order": 1}, "1-" + SHA_B[:12]: {"order": 1}}
        ),
    )
    with pytest.raises(ValueError, match="order"):
        list_table_runs(store)


def test_a_run_whose_order_disagrees_with_its_folder_is_refused(tmp_path):
    store = write_root_block(
        tmp_path, v2_block(runs={"2-" + SHA_A[:12]: {"order": 1, "pipeline_sha256": SHA_A}})
    )
    with pytest.raises(ValueError, match="order"):
        list_table_runs(store)


def test_an_unknown_tables_schema_version_is_refused(tmp_path):
    block = v2_block()
    block["tables"]["schema_version"] = 3
    with pytest.raises(ValueError, match="schema_version"):
        store_tables_schema(write_root_block(tmp_path, block))


def test_the_resolver_orders_runs_and_picks_defaults(tmp_path):
    measurements = {"measurements": {"path": "x"}}
    store = write_root_block(
        tmp_path,
        v2_block(
            runs={
                "3-" + SHA_C[:12]: {"order": 3, "pipeline_sha256": SHA_C},  # grid-only
                "1-" + SHA_A[:12]: {"order": 1, "pipeline_sha256": SHA_A, **measurements},
                "2-" + SHA_B[:12]: {"order": 2, "pipeline_sha256": SHA_B, **measurements},
            }
        ),
    )
    assert [run.order for run in list_table_runs(store)] == [1, 2, 3]
    assert list_table_runs(store)[-1].run_id == "3-" + SHA_C[:12]
    latest = latest_measurement_run(store)
    assert latest.run_id == "2-" + SHA_B[:12] and latest.has_measurements
    assert owned_table_run(store, SHA_A).order == 1
    assert owned_table_run(store, "d" * 64) is None
    run = owned_table_run(store, SHA_B)
    assert run.measurements_path() == store / "tables" / run.run_id / "measurements" / "table.parquet"
    assert run.metadata_path() == store / "tables" / run.run_id / "metadata" / "pht-metadata.parquet"
    assert run.embedding_path("dinov2-vitb14", "objects") == (
        store / "tables" / run.run_id / "embeddings" / "dinov2-vitb14" / "objects" / "table.parquet"
    )


def test_an_empty_journal_has_no_pipeline_sha(tiny_pipeline_file):
    image = Image(arr=load_synth_yeast_plate().rgb[:])
    assert current_pipeline_sha256(image) is None
    initialize_cli_provenance(image, tiny_pipeline_file, kind="full", input_filename="p.tif")
    assert current_pipeline_sha256(image) == file_sha256(tiny_pipeline_file)


def test_a_programmatic_application_after_the_cli_has_no_pipeline_sha(tiny_pipeline_file):
    from phenotypic._core._provenance import set_provenance_status
    from phenotypic.detect import OtsuDetector

    image = Image(arr=load_synth_yeast_plate().rgb[:])
    initialize_cli_provenance(image, tiny_pipeline_file, kind="full", input_filename="p.tif")
    set_provenance_status(image, "complete")
    assert current_pipeline_sha256(OtsuDetector().apply(image)) is None


def test_store_of_table_path_handles_both_depths(tmp_path):
    s = tmp_path / "x.ome.zarr"
    assert store_of_table_path(s / "tables/measurements/table.parquet") == s
    assert store_of_table_path(s / "tables/2-a71e09c2b4f5/measurements/table.parquet") == s
    with pytest.raises(ValueError):
        store_of_table_path(tmp_path / "not-a-store" / "table.parquet")


def test_the_v1_fixtures_are_v1(v1_store, v1_store_legacy_null_pipeline):
    assert store_tables_schema(v1_store) == 1 == store_tables_schema(v1_store_legacy_null_pipeline)
    assert list_table_runs(v1_store) == ()
    assert latest_measurement_run(v1_store) is None


def test_a_store_without_tables_is_schema_0(tmp_path):
    assert store_tables_schema(write_root_block(tmp_path, {"store_schema_version": 3})) == 0


def _journal(sha: str | None, *, kind: str = "full") -> dict:
    pipeline = None if sha is None else {"source_path": "p.json", "sha256": sha}
    return {"schema_version": 2, "applications": [{"kind": kind, "pipeline": pipeline}]}


def _measured(order: int, sha: str | None) -> tuple[str, dict]:
    run_id = table_run_id(order, sha)
    return run_id, {"order": order, "pipeline_sha256": sha, "measurements": {}}


def _store_with_journal(tmp_path: Path, journal: dict, *runs: tuple[str, dict]) -> Path:
    block = v2_block(runs=dict(runs))
    block["provenance"] = journal
    return write_root_block(tmp_path, block)


def test_the_journal_run_is_the_journals_pipeline_folder_not_the_latest(tmp_path):
    # A re-run of A after B: folder 1 is A's, the latest folder is B's.
    store = _store_with_journal(tmp_path, _journal(SHA_A), _measured(1, SHA_A), _measured(2, SHA_B))
    assert journal_table_run(store).run_id == table_run_id(1, SHA_A)
    assert latest_measurement_run(store).run_id == table_run_id(2, SHA_B)


@pytest.mark.parametrize("journal", [_journal(None), _journal(SHA_A, kind="programmatic"), {}])
def test_a_journal_without_a_digest_owns_the_highest_unknown_folder(tmp_path, journal):
    store = _store_with_journal(
        tmp_path, journal, _measured(1, None), _measured(2, SHA_A), _measured(3, None)
    )
    assert journal_table_run(store).run_id == "3-unknown"


def test_a_journal_whose_pipeline_owns_no_folder_has_no_run(tmp_path):
    store = _store_with_journal(tmp_path, _journal(SHA_C), _measured(1, SHA_A))
    assert journal_table_run(store) is None
    assert journal_measurement_table(store) is None


def test_the_journal_probe_degrades_to_none(tmp_path):
    assert journal_measurement_table(tmp_path / "absent.ome.zarr") is None
    store = _store_with_journal(tmp_path, _journal(SHA_A), _measured(1, SHA_A))
    assert journal_table_run(store) is not None
    assert journal_measurement_table(store) is None  # declared, but no file on disk
    table = journal_table_run(store).measurements_path()
    table.parent.mkdir(parents=True)
    table.write_bytes(b"")
    assert journal_measurement_table(store) == table
