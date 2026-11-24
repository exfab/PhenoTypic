"""``replace_image_tables(carry_embeddings=...)``: measure mode keeps comparable embeddings (#282 U8).

A deferring ``--mode measure`` core writes no embeddings of its own: Stage 4
computes them afterwards. Until it has, the run folder carries the image's
existing entries forward, files as hard links and fields unchanged, so a
Stage-4 failure leaves the previous embeddings and a record that verifies.
"""

from __future__ import annotations

import json

import pytest

from phenotypic.sdk_ import list_table_runs, ngff_, replace_image_tables
from phenotypic.sdk_._embedding_tables import read_embedding_table
from phenotypic.sdk_._table_runs import (
    TableRun,
    embedding_table_relative_path,
    latest_measurement_run,
)
from tests._tiny_run import tables_for_store


def embedding_file(store, run: TableRun):
    """The one embedding table file *run* declares."""
    (key,) = run.descriptor["embeddings"]
    return store / run.descriptor["embeddings"][key]["files"][0]["path"]


def relocated(source: TableRun, run: TableRun, key: str) -> list[dict]:
    """*source*'s entry ``files`` with each path moved to *run*'s folder, sha256 unchanged."""
    return [
        {**file, "path": embedding_table_relative_path(run.run_id, key).as_posix()}
        for file in source.descriptor["embeddings"][key]["files"]
    ]


def _root(store) -> dict:
    return json.loads((store / ngff_.STORE_ROOT_JSON).read_text(encoding="utf-8"))


def test_carry_keeps_the_runs_embedding_files_and_block(store_with_crop_entry):
    store = store_with_crop_entry
    run = latest_measurement_run(store)
    old = dict(run.descriptor["embeddings"])
    inode = embedding_file(store, run).stat().st_ino
    replace_image_tables(store, tables_for_store(store), pipeline_sha256=run.pipeline_sha256,
                         carry_embeddings=[(run.run_id, key) for key in old])
    run = latest_measurement_run(store)
    assert run.descriptor["embeddings"] == old and embedding_file(store, run).stat().st_ino == inode


def test_carry_from_another_run_into_a_new_folder(store_with_crop_entry):
    store = store_with_crop_entry
    source = latest_measurement_run(store)
    (key,) = source.descriptor["embeddings"]
    run = replace_image_tables(store, tables_for_store(store), pipeline_sha256="b" * 64,
                               carry_embeddings=[(source.run_id, key)])
    assert run.run_id != source.run_id
    assert run.descriptor["embeddings"][key] == {**source.descriptor["embeddings"][key],
                                                 "files": relocated(source, run, key)}
    # The carried folder is a complete group tree a reader verifies against
    # its sha256, and the source run keeps its own entry and file.
    assert read_embedding_table(store, key, run_id=run.run_id).num_rows > 0
    assert read_embedding_table(store, key, run_id=source.run_id).num_rows > 0
    assert embedding_file(store, run).stat().st_ino == embedding_file(store, source).stat().st_ino
    for group in embedding_table_relative_path(run.run_id, key).parents[1:4]:
        assert (store / group / ngff_.STORE_ROOT_JSON).is_file()


def test_carry_refuses_an_undeclared_pair(store_with_crop_entry):
    before = _root(store_with_crop_entry)
    with pytest.raises(ValueError, match="9-unknown"):
        replace_image_tables(store_with_crop_entry, tables_for_store(store_with_crop_entry),
                             pipeline_sha256=None, carry_embeddings=[("9-unknown", "x/rgb/objects")])
    assert _root(store_with_crop_entry) == before
    assert not list(store_with_crop_entry.parent.glob("*.part"))


def test_carry_refuses_a_key_the_run_does_not_declare(store_with_crop_entry):
    run = latest_measurement_run(store_with_crop_entry)
    with pytest.raises(ValueError, match="dinov2-vitb14/rgb/grid"):
        replace_image_tables(store_with_crop_entry, tables_for_store(store_with_crop_entry),
                             pipeline_sha256=run.pipeline_sha256,
                             carry_embeddings=[(run.run_id, "dinov2-vitb14/rgb/grid")])


def test_carry_refuses_a_key_written_twice(store_with_crop_entry):
    run = latest_measurement_run(store_with_crop_entry)
    (key,) = run.descriptor["embeddings"]
    with pytest.raises(ValueError, match="written twice"):
        replace_image_tables(store_with_crop_entry, tables_for_store(store_with_crop_entry),
                             pipeline_sha256=run.pipeline_sha256,
                             carry_embeddings=[(run.run_id, key), (run.run_id, key)])


def test_without_carry_the_embeddings_go(store_with_crop_entry):
    store = store_with_crop_entry
    run = latest_measurement_run(store)
    replace_image_tables(store, tables_for_store(store), pipeline_sha256=run.pipeline_sha256)
    assert "embeddings" not in latest_measurement_run(store).descriptor          # #281 behaviour, unchanged


def test_a_carried_stale_flag_is_kept_as_recorded(store_with_crop_entry):
    """The measure write writes no objmap, so ``stale`` is never recomputed for a carried entry."""
    store = store_with_crop_entry
    run = latest_measurement_run(store)
    (key,) = run.descriptor["embeddings"]
    entries = [r for r in list_table_runs(store) if r.run_id == run.run_id]
    assert entries[0].descriptor["embeddings"][key]["stale"] is False
    replace_image_tables(store, tables_for_store(store), pipeline_sha256="c" * 64,
                         carry_embeddings=[(run.run_id, key)])
    assert latest_measurement_run(store).descriptor["embeddings"][key]["stale"] is False
