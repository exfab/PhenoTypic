"""Embedding tables in a store's run folder (#281 P4 Task 2, spec D1, D3, D4, D7, D9).

``save2zarr(embeddings=)`` writes each payload as
``tables/<run>/embeddings/<model>/<crops|grid>/table.parquet``, with an
``embedding`` column of Arrow ``FixedSizeList<float16>[D]``, and describes it
in the run entry's ``embeddings`` block: the producer's descriptor plus
``binds``, ``stale`` and the file's sha256. ``read_embedding_table`` checks
that hash and refuses a stale entry.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from phenotypic._cli._cli_failure_tracker import file_sha256
from phenotypic.sdk_ import list_table_runs
from phenotypic.sdk_._embedding_tables import (
    EmbeddingTableUnreadableError,
    read_embedding_table,
)
from phenotypic.sdk_._measurement_tables import objmap_digest
from tests._tiny_run import (
    TOY_DIM,
    resave_with_objmap,
    run_entry,
    save_with_payload,
    tables_for,
)


def test_vectors_are_fixed_size_float16(tmp_path, image_with_crop_payload):
    store = save_with_payload(tmp_path, image_with_crop_payload)
    t = pq.read_table(list_table_runs(store)[-1].embedding_path("toy-v1", "crops"))
    assert t.schema.field("embedding").type == pa.list_(pa.float16(), 4)


def test_entry_binds_objmap_and_lists_file_hash(tmp_path, image_with_crop_payload):
    store = save_with_payload(tmp_path, image_with_crop_payload)
    entry = run_entry(store)["embeddings"]["toy-v1/crops"]
    assert entry["binds"]["objmap_sha256"] == objmap_digest(image_with_crop_payload.objmap[:])
    assert entry["files"][0]["sha256"] == file_sha256(store / entry["files"][0]["path"])


def test_two_models_coexist(tmp_path, image_with_two_models):
    store = save_with_payload(tmp_path, image_with_two_models)
    assert set(run_entry(store)["embeddings"]) == {"toy-v1/crops", "toy-v2/crops"}


def test_a_corrupt_file_is_unreadable(tmp_path, image_with_crop_payload):
    store = save_with_payload(tmp_path, image_with_crop_payload)
    list_table_runs(store)[-1].embedding_path("toy-v1", "crops").write_bytes(b"corrupt")
    with pytest.raises(EmbeddingTableUnreadableError):
        read_embedding_table(store, "toy-v1/crops")


# ---------------------------------------------------------------------------
# The entry, the table and the groups (D1, D3, D4)
# ---------------------------------------------------------------------------


def test_the_entry_carries_the_descriptor_and_is_fresh(tmp_path, image_with_crop_payload):
    store = save_with_payload(tmp_path, image_with_crop_payload)
    run = list_table_runs(store)[-1]
    entry = run_entry(store)["embeddings"]["toy-v1/crops"]
    assert entry["stale"] is False
    assert entry["dim"] == TOY_DIM
    assert entry["dtype"] == "float16"
    assert entry["key"] == "Object_Label"
    assert entry["input"] == {"series": "rgb", "level": 0}
    assert entry["binds"]["objmap_path"] == "rgb/labels/objmap"
    assert entry["files"] == [
        {
            "path": run.embedding_path("toy-v1", "crops").relative_to(store).as_posix(),
            "sha256": entry["files"][0]["sha256"],
            "media_type": "application/vnd.apache.parquet",
        }
    ]
    # The run's measurements are written beside the embeddings, in one folder.
    assert run.has_measurements


def test_an_independent_reader_finds_the_groups_and_the_rows(tmp_path, image_with_crop_payload):
    """zarr-python and pyarrow alone, never PhenoTypic's reopen (README conventions)."""
    import zarr

    store = save_with_payload(tmp_path, image_with_crop_payload)
    run = list_table_runs(store)[-1]
    root = zarr.open_group(str(store), mode="r")
    group = root[f"tables/{run.run_id}/embeddings/toy-v1/crops"]
    assert dict(group.attrs) == {
        "type": "phenotypic:embedding_table",
        "backend": "parquet",
        "index_key": "Object_Label",
    }
    table = pq.read_table(run.embedding_path("toy-v1", "crops"))
    assert table.column_names == ["Object_Label", "embedding"]
    assert table.schema.field("Object_Label").type == pa.int64()
    assert table.num_rows == image_with_crop_payload.num_objects
    assert sorted(table.column("Object_Label").to_pylist()) == image_with_crop_payload.objects.labels


def test_read_embedding_table_returns_the_vectors(tmp_path, image_with_crop_payload):
    store = save_with_payload(tmp_path, image_with_crop_payload)
    table = read_embedding_table(store, "toy-v1/crops")
    assert table.num_rows == image_with_crop_payload.num_objects
    first = table.column("embedding")[0].as_py()
    assert len(first) == TOY_DIM


def test_identical_saves_write_identical_table_bytes(tmp_path, image_with_crop_payload):
    first = save_with_payload(tmp_path / "a", image_with_crop_payload)
    second = save_with_payload(tmp_path / "b", image_with_crop_payload)
    path_a = list_table_runs(first)[-1].embedding_path("toy-v1", "crops")
    path_b = list_table_runs(second)[-1].embedding_path("toy-v1", "crops")
    assert path_a.read_bytes() == path_b.read_bytes()


def test_a_grid_payload_is_written_with_its_keys_and_binds_the_array(
    tmp_path, image_with_grid_payload
):
    store = save_with_payload(tmp_path, image_with_grid_payload)
    entry = run_entry(store)["embeddings"]["toy-v1/grid"]
    assert entry["stale"] is False
    assert set(entry["binds"]) == {"array_sha256", "grid_sha256"}
    table = read_embedding_table(store, "toy-v1/grid")
    assert table.column_names[0] == "Grid_RowMajorIdx"
    assert table.column_names[-1] == "embedding"
    assert table.num_rows == 4


def test_an_embedding_only_save_writes_a_run_without_measurements(
    tmp_path, image_with_grid_payload
):
    """A grid-only folder (AutoConvertRaw-GC's) has no ``measurements`` (spec D1, D6a)."""
    from tests._tiny_run import payloads_of

    store = image_with_grid_payload.save2zarr(
        tmp_path / "grid_only.ome.zarr", embeddings=payloads_of(image_with_grid_payload)
    )
    (run,) = list_table_runs(store)
    assert not run.has_measurements
    assert set(run.descriptor["embeddings"]) == {"toy-v1/grid"}


def test_a_payload_key_must_match_its_mapping_key(tmp_path, image_with_crop_payload):
    from tests._tiny_run import payloads_of

    payload = payloads_of(image_with_crop_payload)["toy-v1/crops"]
    with pytest.raises(ValueError, match="toy-v9/crops"):
        image_with_crop_payload.save2zarr(
            tmp_path / "x.ome.zarr", embeddings={"toy-v9/crops": payload}
        )


@pytest.mark.parametrize(
    "key",
    [
        "a/b/c", "toy-v1", "toy-v1/", "/crops", "../crops", "a\\b/grid", "toy-v1/tiles",
        # #281 named grid regions: ``grid-<slug>`` and nothing looser.
        "toy-v1/grid-", "toy-v1/grid-Wells", "toy-v1/grid-wells-a", "toy-v1/grid-_wells",
        "toy-v1/grid-wells/x", "toy-v1/grid-wells\n", "toy-v1/gridwells", "toy-v1/grid_wells",
        "toy-v1/crops-wells", "toy-v1/grid-" + "a" * 33,
    ],
)
def test_a_payload_and_a_read_entry_refuse_a_key_by_one_rule(key):
    """#281 P4 simplify, key validation: ``split_embedding_key`` is the rule
    ``EmbeddingPayload`` refuses a key on, word for word, so the deliverables
    aggregation (which calls it on a store's entry keys) cannot drift from it."""
    import pandas as pd

    from phenotypic.sdk_._embedding_tables import EmbeddingPayload
    from phenotypic.sdk_._table_runs import split_embedding_key

    with pytest.raises(ValueError) as split:
        split_embedding_key(key)
    with pytest.raises(ValueError) as built:
        EmbeddingPayload(key=key, frame=pd.DataFrame(), descriptor={})
    assert str(built.value) == str(split.value)


@pytest.mark.parametrize(
    ("key", "parts"),
    [
        ("toy-v1/crops", ("toy-v1", "crops")),
        ("dinov2-vitb14/grid", ("dinov2-vitb14", "grid")),
        ("dinov2-vitb14/grid-wells", ("dinov2-vitb14", "grid-wells")),
        ("dinov2-vitb14/grid-agar", ("dinov2-vitb14", "grid-agar")),
        ("toy-v1/grid-0", ("toy-v1", "grid-0")),
        ("toy-v1/grid-rig1_wells", ("toy-v1", "grid-rig1_wells")),
        ("toy-v1/grid-" + "a" * 32, ("toy-v1", "grid-" + "a" * 32)),
    ],
)
def test_a_model_and_region_key_splits_into_its_folders(key, parts):
    from phenotypic.sdk_._table_runs import split_embedding_key

    assert split_embedding_key(key) == parts


# ---------------------------------------------------------------------------
# Staleness is recomputed on every write (D7), and readers refuse it
# ---------------------------------------------------------------------------


def test_a_resave_with_another_objmap_marks_the_entry_stale_and_unreadable(
    tmp_path, image_with_crop_payload
):
    store = save_with_payload(tmp_path, image_with_crop_payload)
    resave_with_objmap(store, zeros=True)
    assert run_entry(store)["embeddings"]["toy-v1/crops"]["stale"] is True
    with pytest.raises(EmbeddingTableUnreadableError, match="stale"):
        read_embedding_table(store, "toy-v1/crops")
    resave_with_objmap(store, zeros=False)
    assert run_entry(store)["embeddings"]["toy-v1/crops"]["stale"] is False
    assert read_embedding_table(store, "toy-v1/crops").num_rows > 0


def test_a_payload_computed_from_another_objmap_is_written_stale(
    tmp_path, image_with_crop_payload
):
    """The binds are the embedded image's; the writer compares them with what it writes."""
    from tests._tiny_run import payloads_of

    payloads = payloads_of(image_with_crop_payload)
    image_with_crop_payload.objmap[:] = 0
    store = image_with_crop_payload.save2zarr(
        tmp_path / "moved.ome.zarr", embeddings=payloads
    )
    assert run_entry(store)["embeddings"]["toy-v1/crops"]["stale"] is True


def test_an_unknown_key_is_a_key_error(tmp_path, image_with_crop_payload):
    store = save_with_payload(tmp_path, image_with_crop_payload)
    with pytest.raises(KeyError, match="toy-v7/crops"):
        read_embedding_table(store, "toy-v7/crops")


def test_a_missing_file_is_unreadable(tmp_path, image_with_crop_payload):
    store = save_with_payload(tmp_path, image_with_crop_payload)
    list_table_runs(store)[-1].embedding_path("toy-v1", "crops").unlink()
    with pytest.raises(EmbeddingTableUnreadableError, match="missing"):
        read_embedding_table(store, "toy-v1/crops")


def test_the_root_certifies_the_embedding_file(tmp_path, image_with_crop_payload):
    """D9: the file's sha256 lives in the root, which the completion marker hashes."""
    store = save_with_payload(tmp_path, image_with_crop_payload)
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    runs = root["attributes"]["phenotypic"]["tables"]["runs"]
    (entry,) = runs.values()
    path = store / entry["embeddings"]["toy-v1/crops"]["files"][0]["path"]
    assert entry["embeddings"]["toy-v1/crops"]["files"][0]["sha256"] == file_sha256(path)


# ---------------------------------------------------------------------------
# The measure rewrite: replace_image_tables(embeddings=)
# ---------------------------------------------------------------------------


def test_replace_image_tables_writes_the_given_embeddings(tmp_path, image_with_crop_payload):
    from phenotypic.sdk_ import replace_image_tables
    from tests._tiny_run import payloads_of

    store = save_with_payload(tmp_path, image_with_crop_payload)
    (run,) = list_table_runs(store)
    before = run_entry(store)["embeddings"]["toy-v1/crops"]
    written = replace_image_tables(
        store,
        tables_for(image_with_crop_payload),
        pipeline_sha256=run.pipeline_sha256,
        embeddings=payloads_of(image_with_crop_payload),
        durable=False,
    )
    assert written.run_id == run.run_id
    after = run_entry(store)["embeddings"]["toy-v1/crops"]
    assert after == before
    assert read_embedding_table(store, "toy-v1/crops").num_rows == (
        image_with_crop_payload.num_objects
    )


def test_two_named_grids_write_two_tables_in_one_run(tmp_path, pipeline_file_a, plate):
    """#281 named grid regions: ``grid-wells`` and ``grid-agar`` of one model are
    two tables in one run folder, each holding its own grid's cells."""
    import pyarrow.parquet as pq

    from phenotypic.schema import GRID
    from phenotypic.sdk_ import list_table_runs, read_embedding_table
    from tests._tiny_run import (
        _IMAGE_PAYLOADS,
        _with_cli_payloads,
        named_grid_embedders,
        save_with_payload,
        toy_grid_embedder_of,
    )

    image = _with_cli_payloads(plate, pipeline_file_a, named_grid_embedders(toy_grid_embedder_of))
    try:
        store = save_with_payload(tmp_path, image)
    finally:
        _IMAGE_PAYLOADS.pop(id(image), None)
    (run,) = list_table_runs(store)
    assert list(run.descriptor["embeddings"]) == ["toy-v1/grid-wells", "toy-v1/grid-agar"]
    for region, cells in (("grid-wells", 4), ("grid-agar", 1)):
        path = run.embedding_path("toy-v1", region)
        assert path == store / "tables" / run.run_id / "embeddings/toy-v1" / region / "table.parquet"
        assert pq.read_metadata(path).num_rows == cells
        entry = run.descriptor["embeddings"][f"toy-v1/{region}"]
        assert entry["files"][0]["path"].endswith(f"embeddings/toy-v1/{region}/table.parquet")
        assert entry["key"] == str(GRID.ROW_MAJOR_IDX)
    wells = read_embedding_table(store, "toy-v1/grid-wells")
    agar = read_embedding_table(store, "toy-v1/grid-agar")
    entries = run.descriptor["embeddings"]
    assert entries["toy-v1/grid-wells"]["binds"]["grid_sha256"] != (
        entries["toy-v1/grid-agar"]["binds"]["grid_sha256"]
    )
    assert entries["toy-v1/grid-wells"]["producer"]["params_sha256"] != (
        entries["toy-v1/grid-agar"]["producer"]["params_sha256"]
    )
    assert agar.column(str(GRID.ROW_INTERVAL_END)).to_pylist() == [200]
    assert wells.column(str(GRID.ROW_INTERVAL_END)).to_pylist() == [100, 100, 200, 200]
