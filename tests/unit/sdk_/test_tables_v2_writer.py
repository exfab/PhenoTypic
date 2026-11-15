"""The v2 store writer, the carry step and stale binding (#281 P3 Task 2, spec D1-D7)."""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import zarr

from phenotypic import Image
from phenotypic._cli._cli_failure_tracker import file_sha256
from phenotypic.sdk_ import (
    latest_table_run,
    list_table_runs,
    table_run_for_pipeline,
)
from phenotypic.sdk_._measurement_tables import (
    _valid_embedded_measurement_contract,
    ngff_feature_table_attrs,
    objmap_digest,
    refresh_embedding_staleness,
)
from tests._tiny_run import (
    crop_entry,
    resave_with_objmap,
    resave_with_pipeline,
    tables_for,
    tree_digest,
)


def _root_tables(store):
    return json.loads((store / "zarr.json").read_text(encoding="utf-8"))["attributes"][
        "phenotypic"
    ]["tables"]


def test_save_writes_run_folder_and_v2_descriptor(tmp_path, measured_image, pipeline_file_a):
    store = measured_image.save2zarr(tmp_path / "p.ome.zarr", measurement_table=tables_for(measured_image))
    run = latest_table_run(store)
    assert run.run_id == f"1-{file_sha256(pipeline_file_a)[:12]}" and run.measurements_path().is_file()
    root = _root_tables(store)
    assert root["schema_version"] == 2 and list(root["runs"]) == [run.run_id]
    assert root["runs"][run.run_id]["order"] == 1
    assert root["runs"][run.run_id]["pipeline_sha256"] == file_sha256(pipeline_file_a)
    assert root["runs"][run.run_id]["measurements"]["path"] == (
        f"tables/{run.run_id}/measurements/table.parquet"
    )
    assert not (store / "tables" / "measurements").exists()   # nothing at the v1 path


def test_the_root_metadata_table_block_moves_into_its_run(tmp_path, measured_image):
    from dataclasses import replace

    tables = replace(
        tables_for(measured_image),
        join_status="no_common_keys",
        metadata_snapshot_sha256="e" * 64,
    )
    store = measured_image.save2zarr(tmp_path / "p.ome.zarr", measurement_table=tables)
    phenotypic = json.loads((store / "zarr.json").read_text(encoding="utf-8"))["attributes"]["phenotypic"]
    assert "metadata_table" not in phenotypic
    run = latest_table_run(store)
    assert run.descriptor["metadata_table"] == {"snapshot_sha256": "e" * 64}


def test_a_programmatic_save_is_unknown_and_resaves_into_one_folder(tmp_path, measured_image):
    image = Image(arr=measured_image.rgb[:])                  # empty journal
    store = image.save2zarr(tmp_path / "q.ome.zarr", measurement_table=tables_for(measured_image))
    image.save2zarr(store, measurement_table=tables_for(measured_image))
    assert [r.run_id for r in list_table_runs(store)] == ["1-unknown"]
    assert list_table_runs(store)[0].pipeline_sha256 is None


def test_a_second_pipeline_gets_folder_2(tmp_path, measured_image, pipeline_file_b):
    store = measured_image.save2zarr(tmp_path / "p.ome.zarr", measurement_table=tables_for(measured_image))
    resave_with_pipeline(store, pipeline_file_b)
    assert [r.order for r in list_table_runs(store)] == [1, 2]
    assert list_table_runs(store)[1].run_id == f"2-{file_sha256(pipeline_file_b)[:12]}"


def test_resave_carries_other_runs_byte_identical(store_with_two_runs, pipeline_file_b):
    first = list_table_runs(store_with_two_runs)[0]
    before = tree_digest(store_with_two_runs / "tables" / first.run_id)
    resave_with_pipeline(store_with_two_runs, pipeline_file_b)
    assert tree_digest(store_with_two_runs / "tables" / first.run_id) == before
    assert list_table_runs(store_with_two_runs)[0].descriptor == first.descriptor


def test_table_less_save_keeps_its_own_folder(store_with_two_runs, pipeline_file_b):
    # Review Focus 2: the initial-checkpoint and Stage-1 saves write no tables
    own = table_run_for_pipeline(store_with_two_runs, file_sha256(pipeline_file_b))
    before = tree_digest(store_with_two_runs / "tables" / own.run_id)
    resave_with_pipeline(store_with_two_runs, pipeline_file_b, tables=False)
    assert tree_digest(store_with_two_runs / "tables" / own.run_id) == before
    assert [r.order for r in list_table_runs(store_with_two_runs)] == [1, 2]


def test_the_same_pipeline_replaces_its_folder_with_new_inodes(store_with_two_runs, pipeline_file_b):
    own = table_run_for_pipeline(store_with_two_runs, file_sha256(pipeline_file_b))
    inode = own.measurements_path().stat().st_ino
    resave_with_pipeline(store_with_two_runs, pipeline_file_b)
    again = table_run_for_pipeline(store_with_two_runs, file_sha256(pipeline_file_b))
    assert again.run_id == own.run_id and again.measurements_path().stat().st_ino != inode


def test_the_carried_folder_is_hard_linked_not_rewritten(store_with_two_runs, pipeline_file_b):
    first = list_table_runs(store_with_two_runs)[0]
    inode = first.measurements_path().stat().st_ino
    resave_with_pipeline(store_with_two_runs, pipeline_file_b)
    assert list_table_runs(store_with_two_runs)[0].measurements_path().stat().st_ino == inode


def test_stale_is_recomputed_not_latched(store_with_crop_entry):
    # Review Focus 2: Stage 1 writes zeros, Stage 3 restores the same objmap
    assert crop_entry(store_with_crop_entry)["stale"] is False
    resave_with_objmap(store_with_crop_entry, zeros=True)
    assert crop_entry(store_with_crop_entry)["stale"] is True
    resave_with_objmap(store_with_crop_entry, zeros=False)
    assert crop_entry(store_with_crop_entry)["stale"] is False


def test_the_crop_payload_is_carried_by_a_table_less_save(store_with_crop_entry):
    path = store_with_crop_entry / crop_entry(store_with_crop_entry)["files"][0]["path"]
    before = path.read_bytes()
    resave_with_objmap(store_with_crop_entry, zeros=True)
    assert path.read_bytes() == before


def test_grid_staleness_follows_the_bound_array():
    runs = {
        "1-unknown": {
            "order": 1,
            "pipeline_sha256": None,
            "embeddings": {
                "m/grid": {"input": {"series": "rgb", "level": 0},
                           "binds": {"array_sha256": "x", "grid_sha256": "g"}, "stale": False},
                "m/crops": {"binds": {"objmap_path": "rgb/labels/objmap", "objmap_sha256": "o"},
                            "stale": True},
            },
        }
    }
    refresh_embedding_staleness(runs, objmap_sha256="o", array_sha256={"rgb": "y"})
    embeddings = runs["1-unknown"]["embeddings"]
    assert embeddings["m/grid"]["stale"] is True and embeddings["m/crops"]["stale"] is False
    refresh_embedding_staleness(runs, objmap_sha256=None, array_sha256={"rgb": "x"})
    assert embeddings["m/grid"]["stale"] is False and embeddings["m/crops"]["stale"] is True


def test_objmap_digest_is_of_the_labels_as_stored():
    labels = np.arange(12, dtype=np.uint16).reshape(3, 4)
    assert objmap_digest(labels) == objmap_digest(labels.copy())
    assert objmap_digest(labels) != objmap_digest(labels.astype(np.uint32))
    assert objmap_digest(labels) != objmap_digest(labels.reshape(4, 3))
    assert objmap_digest(np.asfortranarray(labels)) == objmap_digest(labels)


def test_a_save_replaces_a_store_with_no_phenotypic_block(tmp_path, measured_image):
    # C12 gate: `--mode migrate` rebuilds an invalid store from its HDF source
    # with save2zarr; a root with no `phenotypic` block carries nothing.
    target = tmp_path / "p.ome.zarr"
    target.mkdir()
    (target / "zarr.json").write_text(
        '{"zarr_format": 3, "node_type": "group", "attributes": {}}', encoding="utf-8"
    )
    store = measured_image.save2zarr(target, measurement_table=tables_for(measured_image))
    assert [r.order for r in list_table_runs(store)] == [1]


def test_an_ordinary_save_refuses_a_store_whose_root_cannot_be_parsed(
    tmp_path, measured_image
):
    """Review M4: the root may declare other pipelines' folders; never drop them."""
    target = tmp_path / "p.ome.zarr"
    (target / "tables" / "1-unknown").mkdir(parents=True)
    (target / "zarr.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be read"):
        measured_image.save2zarr(target, measurement_table=tables_for(measured_image))
    with pytest.raises(ValueError, match="cannot be read"):
        measured_image.save2zarr(target)  # a table-less save reads the root too
    assert (target / "zarr.json").read_text(encoding="utf-8") == "{not json"
    assert (target / "tables" / "1-unknown").is_dir()


def test_the_hdf_rebuild_replaces_an_unparseable_store_whole(tmp_path, measured_image):
    """The one explicit exception: migrate's replace-from-HDF carries nothing."""
    target = tmp_path / "p.ome.zarr"
    target.mkdir()
    (target / "zarr.json").write_text("{not json", encoding="utf-8")
    store = measured_image.save2zarr(
        target, measurement_table=tables_for(measured_image), replace_unreadable=True
    )
    assert [r.order for r in list_table_runs(store)] == [1]


def test_migrate_hdf_to_zarr_passes_the_explicit_replace(monkeypatch, tmp_path):
    """The HDF rebuild opts in by keyword, never by sniffing the root."""
    from phenotypic.sdk_ import _hdf_to_zarr

    seen: dict = {}

    class _FakeImage:
        def save2zarr(self, target, **kwargs):
            seen.update(kwargs)
            return target

    monkeypatch.setattr(_hdf_to_zarr, "_load_for_migration", lambda _src: _FakeImage())
    monkeypatch.setattr(_hdf_to_zarr, "_stored_work_id", lambda _src: None)
    _hdf_to_zarr.migrate_hdf_to_zarr(tmp_path / "x.h5", tmp_path / "x.ome.zarr")
    assert seen["replace_unreadable"] is True


def test_the_contract_validator_reads_v2(tmp_path, measured_image):
    store = measured_image.save2zarr(tmp_path / "p.ome.zarr", measurement_table=tables_for(measured_image))
    assert _valid_embedded_measurement_contract(store)
    run = latest_table_run(store)
    run.measurements_path().write_bytes(b"corrupt")
    assert not _valid_embedded_measurement_contract(store)


def test_the_contract_validator_still_reads_v1(v1_store):
    assert _valid_embedded_measurement_contract(v1_store)


def test_independent_reader(tmp_path, measured_image):
    store = measured_image.save2zarr(tmp_path / "p.ome.zarr", measurement_table=tables_for(measured_image))
    run = latest_table_run(store)
    g = zarr.open_group(store, mode="r")
    attrs = dict(g["tables"][run.run_id]["measurements"].attrs)
    assert attrs["type"] == "feature_table"
    assert attrs == ngff_feature_table_attrs("../../rgb/labels/objmap")
    assert pq.read_table(run.measurements_path()).num_rows == measured_image.num_objects
    # The region path resolves, from the table's parent group, to the objmap label.
    region = (store / "tables" / run.run_id / attrs["region"]["path"]).resolve()
    assert region == (store / "rgb" / "labels" / "objmap").resolve()
    assert zarr.open_group(region, mode="r").attrs["ome"]["image-label"] is not None
