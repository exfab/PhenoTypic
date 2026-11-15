"""The ``--mode measure`` table rewrite over per-pipeline run folders (#281 P3 Task 3).

``replace_image_tables`` picks its run folder by the measuring pipeline's
digest (spec D6): the same pipeline replaces its folder, keeping its number,
as new inodes; another pipeline adds a folder with the next number; every
other folder is carried byte for byte (D5). The rewrite writes no objmap, so
it leaves every embedding's ``stale`` flag as it found it (D7).
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from phenotypic._cli._cli_failure_tracker import file_sha256
from phenotypic.sdk_ import latest_table_run, list_table_runs, replace_image_tables
from phenotypic.sdk_._table_runs import table_run_by_id
from tests._tiny_run import (
    crop_entry,
    other_tables,
    resave_with_objmap,
    tables_for_store,
    tree_digest,
)


@pytest.fixture
def SHA_B(pipeline_file_b) -> str:  # noqa: N802 - the plan's constant name
    """The digest of pipeline B, a pipeline other than the store's A."""
    return file_sha256(pipeline_file_b)


def _markers(run) -> set[str]:
    return set(pq.read_table(run.measurements_path())["Pipeline_Marker"].to_pylist())


def test_measure_with_a_new_pipeline_adds_a_folder(full_run_store, SHA_B):
    # Review Focus 3
    first = latest_table_run(full_run_store)
    before = tree_digest(full_run_store / "tables" / first.run_id)
    run = replace_image_tables(full_run_store, other_tables(full_run_store), pipeline_sha256=SHA_B)
    assert run.order == 2 and len(list_table_runs(full_run_store)) == 2
    assert tree_digest(full_run_store / "tables" / first.run_id) == before
    assert run.run_id == f"2-{SHA_B[:12]}" and run.pipeline_sha256 == SHA_B
    assert _markers(run) == {"other"} and _markers(first) == {"A"}


def test_measure_with_the_same_pipeline_twice_keeps_one_folder_and_its_rows(full_run_store):
    first = latest_table_run(full_run_store)
    rows = pq.read_table(first.measurements_path()).num_rows
    inode = first.measurements_path().stat().st_ino
    for _ in range(2):
        run = replace_image_tables(full_run_store, tables_for_store(full_run_store),
                                   pipeline_sha256=first.pipeline_sha256)
    assert run.run_id == first.run_id and len(list_table_runs(full_run_store)) == 1
    assert pq.read_table(run.measurements_path()).num_rows == rows
    # D6: the folder's files are rewritten as new inodes, never written through.
    assert run.measurements_path().stat().st_ino != inode


def test_the_measure_rewrite_leaves_stale_unchanged(store_with_crop_entry, SHA_B):
    # Built before the all-zeros resave: a store with no objects cannot be measured.
    tables = other_tables(store_with_crop_entry)
    resave_with_objmap(store_with_crop_entry, zeros=True)
    assert crop_entry(store_with_crop_entry)["stale"] is True
    carried = list_table_runs(store_with_crop_entry)[0]
    before = tree_digest(carried.folder)
    replace_image_tables(store_with_crop_entry, tables, pipeline_sha256=SHA_B)
    assert crop_entry(store_with_crop_entry)["stale"] is True     # D7: no objmap written
    assert tree_digest(carried.folder) == before


def test_the_returned_run_is_the_one_the_promoted_root_declares(full_run_store, SHA_B):
    # The per-image record declares this run's table (D6a), so it must be the
    # root's entry, not a description of an intended write.
    run = replace_image_tables(full_run_store, other_tables(full_run_store), pipeline_sha256=SHA_B)
    declared = table_run_by_id(full_run_store, run.run_id)
    assert declared is not None and dict(declared.descriptor) == dict(run.descriptor)
    assert run.measurements_path().is_file()


def test_an_unknown_digest_replaces_the_highest_unknown_folder(full_run_store):
    first = latest_table_run(full_run_store)
    for _ in range(2):
        run = replace_image_tables(full_run_store, other_tables(full_run_store), pipeline_sha256=None)
    assert [r.run_id for r in list_table_runs(full_run_store)] == [first.run_id, "2-unknown"]
    assert run.run_id == "2-unknown" and run.pipeline_sha256 is None
