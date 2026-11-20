"""Embedding tables in a store's run folder (#281 P4 Task 2, spec D1, D3, D4, D7, D9).

``save2zarr(embeddings=)`` writes each payload as
``tables/<run>/embeddings/<model>/<objects|grid>/table.parquet``, with an
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


def test_vectors_are_fixed_size_float16(tmp_path, image_with_objects_payload):
    store = save_with_payload(tmp_path, image_with_objects_payload)
    t = pq.read_table(list_table_runs(store)[-1].embedding_path("toy-v1/rgb/objects"))
    assert t.schema.field("embedding").type == pa.list_(pa.float16(), 4)


def test_entry_binds_objmap_and_lists_file_hash(tmp_path, image_with_objects_payload):
    store = save_with_payload(tmp_path, image_with_objects_payload)
    entry = run_entry(store)["embeddings"]["toy-v1/rgb/objects"]
    assert entry["binds"]["objmap_sha256"] == objmap_digest(image_with_objects_payload.objmap[:])
    assert entry["files"][0]["sha256"] == file_sha256(store / entry["files"][0]["path"])


def test_two_models_coexist(tmp_path, image_with_two_models):
    store = save_with_payload(tmp_path, image_with_two_models)
    assert set(run_entry(store)["embeddings"]) == {"toy-v1/rgb/objects", "toy-v2/rgb/objects"}


def test_a_corrupt_file_is_unreadable(tmp_path, image_with_objects_payload):
    store = save_with_payload(tmp_path, image_with_objects_payload)
    list_table_runs(store)[-1].embedding_path("toy-v1/rgb/objects").write_bytes(b"corrupt")
    with pytest.raises(EmbeddingTableUnreadableError):
        read_embedding_table(store, "toy-v1/rgb/objects")


# ---------------------------------------------------------------------------
# The entry, the table and the groups (D1, D3, D4)
# ---------------------------------------------------------------------------


def test_the_entry_carries_the_descriptor_and_is_fresh(tmp_path, image_with_objects_payload):
    store = save_with_payload(tmp_path, image_with_objects_payload)
    run = list_table_runs(store)[-1]
    entry = run_entry(store)["embeddings"]["toy-v1/rgb/objects"]
    assert entry["stale"] is False
    assert entry["dim"] == TOY_DIM
    assert entry["dtype"] == "float16"
    assert entry["key"] == "Object_Label"
    assert entry["input"] == {"series": "rgb", "level": 0, "layer": "rgb"}
    assert entry["binds"]["objmap_path"] == "rgb/labels/objmap"
    assert entry["files"] == [
        {
            "path": run.embedding_path("toy-v1/rgb/objects").relative_to(store).as_posix(),
            "sha256": entry["files"][0]["sha256"],
            "media_type": "application/vnd.apache.parquet",
        }
    ]
    # The run's measurements are written beside the embeddings, in one folder.
    assert run.has_measurements


def test_an_independent_reader_finds_the_groups_and_the_rows(tmp_path, image_with_objects_payload):
    """zarr-python and pyarrow alone, never PhenoTypic's reopen (README conventions)."""
    import zarr

    store = save_with_payload(tmp_path, image_with_objects_payload)
    run = list_table_runs(store)[-1]
    root = zarr.open_group(str(store), mode="r")
    group = root[f"tables/{run.run_id}/embeddings/toy-v1/rgb/objects"]
    assert dict(group.attrs) == {
        "type": "phenotypic:embedding_table",
        "backend": "parquet",
        "index_key": "Object_Label",
    }
    table = pq.read_table(run.embedding_path("toy-v1/rgb/objects"))
    assert table.column_names == ["Object_Label", "embedding"]
    assert table.schema.field("Object_Label").type == pa.int64()
    assert table.num_rows == image_with_objects_payload.num_objects
    assert sorted(table.column("Object_Label").to_pylist()) == image_with_objects_payload.objects.labels


def test_read_embedding_table_returns_the_vectors(tmp_path, image_with_objects_payload):
    store = save_with_payload(tmp_path, image_with_objects_payload)
    table = read_embedding_table(store, "toy-v1/rgb/objects")
    assert table.num_rows == image_with_objects_payload.num_objects
    first = table.column("embedding")[0].as_py()
    assert len(first) == TOY_DIM


def test_identical_saves_write_identical_table_bytes(tmp_path, image_with_objects_payload):
    first = save_with_payload(tmp_path / "a", image_with_objects_payload)
    second = save_with_payload(tmp_path / "b", image_with_objects_payload)
    path_a = list_table_runs(first)[-1].embedding_path("toy-v1/rgb/objects")
    path_b = list_table_runs(second)[-1].embedding_path("toy-v1/rgb/objects")
    assert path_a.read_bytes() == path_b.read_bytes()


def test_a_grid_payload_is_written_with_its_keys_and_binds_the_array(
    tmp_path, image_with_grid_payload
):
    store = save_with_payload(tmp_path, image_with_grid_payload)
    entry = run_entry(store)["embeddings"]["toy-v1/rgb/grid"]
    assert entry["stale"] is False
    assert set(entry["binds"]) == {"array_sha256", "grid_sha256"}
    table = read_embedding_table(store, "toy-v1/rgb/grid")
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
    assert set(run.descriptor["embeddings"]) == {"toy-v1/rgb/grid"}


def test_a_payload_key_must_match_its_mapping_key(tmp_path, image_with_objects_payload):
    from tests._tiny_run import payloads_of

    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    with pytest.raises(ValueError, match="toy-v9/rgb/objects"):
        image_with_objects_payload.save2zarr(
            tmp_path / "x.ome.zarr", embeddings={"toy-v9/rgb/objects": payload}
        )


@pytest.mark.parametrize(
    "key",
    [
        "a/b/c", "toy-v1", "toy-v1/", "/rgb/objects", "../rgb/objects", "a\\b/rgb/grid",
        "toy-v1/rgb/tiles",
        # #281 layer folder: exactly <model>/<layer>/<region>, layer rgb or gray.
        "toy-v1/objects", "toy-v1/grid-wells", "toy-v1/rgb/grid/x", "toy-v1/hsv/grid",
        "toy-v1/RGB/grid", "toy-v1//grid", "/toy-v1/rgb/grid",
        # #281 named regions: ``<objects|grid>-<slug>`` and nothing looser.
        "toy-v1/rgb/grid-", "toy-v1/rgb/grid-Wells", "toy-v1/rgb/grid-wells-a", "toy-v1/rgb/grid-_wells",
        "toy-v1/rgb/grid-wells/x", "toy-v1/rgb/grid-wells\n", "toy-v1/rgb/gridwells", "toy-v1/rgb/grid_wells",
        "toy-v1/rgb/objects-", "toy-v1/rgb/objects-Cls", "toy-v1/rgb/tiles-wells",
        "toy-v1/rgb/crops", "toy-v1/rgb/grid-" + "a" * 33,
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
        ("toy-v1/rgb/objects", ("toy-v1", "rgb", "objects")),
        ("toy-v1/gray/objects", ("toy-v1", "gray", "objects")),
        ("dinov2-vitb14/rgb/grid", ("dinov2-vitb14", "rgb", "grid")),
        ("dinov2-vitb14/rgb/grid-wells", ("dinov2-vitb14", "rgb", "grid-wells")),
        ("dinov2-vitb14/gray/grid-wells", ("dinov2-vitb14", "gray", "grid-wells")),
        ("dinov2-vitb14/rgb/grid-agar", ("dinov2-vitb14", "rgb", "grid-agar")),
        ("toy-v1/rgb/grid-0", ("toy-v1", "rgb", "grid-0")),
        ("toy-v1/rgb/grid-rig1_wells", ("toy-v1", "rgb", "grid-rig1_wells")),
        ("dinov2-vitb14/rgb/objects-cls", ("dinov2-vitb14", "rgb", "objects-cls")),
        ("toy-v1/rgb/grid-" + "a" * 32, ("toy-v1", "rgb", "grid-" + "a" * 32)),
    ],
)
def test_a_model_layer_and_region_key_splits_into_its_folders(key, parts):
    from phenotypic.sdk_._table_runs import split_embedding_key

    assert split_embedding_key(key) == parts


# ---------------------------------------------------------------------------
# Staleness is recomputed on every write (D7), and readers refuse it
# ---------------------------------------------------------------------------


def test_a_resave_with_another_objmap_marks_the_entry_stale_and_unreadable(
    tmp_path, image_with_objects_payload
):
    store = save_with_payload(tmp_path, image_with_objects_payload)
    resave_with_objmap(store, zeros=True)
    assert run_entry(store)["embeddings"]["toy-v1/rgb/objects"]["stale"] is True
    with pytest.raises(EmbeddingTableUnreadableError, match="stale"):
        read_embedding_table(store, "toy-v1/rgb/objects")
    resave_with_objmap(store, zeros=False)
    assert run_entry(store)["embeddings"]["toy-v1/rgb/objects"]["stale"] is False
    assert read_embedding_table(store, "toy-v1/rgb/objects").num_rows > 0


def test_a_payload_computed_from_another_objmap_is_written_stale(
    tmp_path, image_with_objects_payload
):
    """The binds are the embedded image's; the writer compares them with what it writes."""
    from tests._tiny_run import payloads_of

    payloads = payloads_of(image_with_objects_payload)
    image_with_objects_payload.objmap[:] = 0
    store = image_with_objects_payload.save2zarr(
        tmp_path / "moved.ome.zarr", embeddings=payloads
    )
    assert run_entry(store)["embeddings"]["toy-v1/rgb/objects"]["stale"] is True


def test_an_unknown_key_is_a_key_error(tmp_path, image_with_objects_payload):
    store = save_with_payload(tmp_path, image_with_objects_payload)
    with pytest.raises(KeyError, match="toy-v7/rgb/objects"):
        read_embedding_table(store, "toy-v7/rgb/objects")


def test_a_missing_file_is_unreadable(tmp_path, image_with_objects_payload):
    store = save_with_payload(tmp_path, image_with_objects_payload)
    list_table_runs(store)[-1].embedding_path("toy-v1/rgb/objects").unlink()
    with pytest.raises(EmbeddingTableUnreadableError, match="missing"):
        read_embedding_table(store, "toy-v1/rgb/objects")


def test_the_root_certifies_the_embedding_file(tmp_path, image_with_objects_payload):
    """D9: the file's sha256 lives in the root, which the completion marker hashes."""
    store = save_with_payload(tmp_path, image_with_objects_payload)
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    runs = root["attributes"]["phenotypic"]["tables"]["runs"]
    (entry,) = runs.values()
    path = store / entry["embeddings"]["toy-v1/rgb/objects"]["files"][0]["path"]
    assert entry["embeddings"]["toy-v1/rgb/objects"]["files"][0]["sha256"] == file_sha256(path)


# ---------------------------------------------------------------------------
# The measure rewrite: replace_image_tables(embeddings=)
# ---------------------------------------------------------------------------


def test_replace_image_tables_writes_the_given_embeddings(tmp_path, image_with_objects_payload):
    from phenotypic.sdk_ import replace_image_tables
    from tests._tiny_run import payloads_of

    store = save_with_payload(tmp_path, image_with_objects_payload)
    (run,) = list_table_runs(store)
    before = run_entry(store)["embeddings"]["toy-v1/rgb/objects"]
    written = replace_image_tables(
        store,
        tables_for(image_with_objects_payload),
        pipeline_sha256=run.pipeline_sha256,
        embeddings=payloads_of(image_with_objects_payload),
        durable=False,
    )
    assert written.run_id == run.run_id
    after = run_entry(store)["embeddings"]["toy-v1/rgb/objects"]
    assert after == before
    assert read_embedding_table(store, "toy-v1/rgb/objects").num_rows == (
        image_with_objects_payload.num_objects
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
    assert list(run.descriptor["embeddings"]) == ["toy-v1/rgb/grid-wells", "toy-v1/rgb/grid-agar"]
    for region, cells in (("grid-wells", 4), ("grid-agar", 1)):
        path = run.embedding_path(f"toy-v1/rgb/{region}")
        assert path == store / "tables" / run.run_id / "embeddings/toy-v1/rgb" / region / "table.parquet"
        assert pq.read_metadata(path).num_rows == cells
        entry = run.descriptor["embeddings"][f"toy-v1/rgb/{region}"]
        assert entry["files"][0]["path"].endswith(f"embeddings/toy-v1/rgb/{region}/table.parquet")
        assert entry["key"] == str(GRID.ROW_MAJOR_IDX)
    wells = read_embedding_table(store, "toy-v1/rgb/grid-wells")
    agar = read_embedding_table(store, "toy-v1/rgb/grid-agar")
    entries = run.descriptor["embeddings"]
    assert entries["toy-v1/rgb/grid-wells"]["binds"]["grid_sha256"] != (
        entries["toy-v1/rgb/grid-agar"]["binds"]["grid_sha256"]
    )
    assert entries["toy-v1/rgb/grid-wells"]["producer"]["params_sha256"] != (
        entries["toy-v1/rgb/grid-agar"]["producer"]["params_sha256"]
    )
    assert agar.column(str(GRID.ROW_INTERVAL_END)).to_pylist() == [200]
    assert wells.column(str(GRID.ROW_INTERVAL_END)).to_pylist() == [100, 100, 200, 200]


def test_an_rgb_and_a_gray_copy_write_both_tables(tmp_path, pipeline_file_a, plate):
    """#281 layer folder: ``wells`` on rgb and on gray, one name, two tables
    under ``<model>/rgb/`` and ``<model>/gray/``, each entry naming its layer."""
    import pyarrow.parquet as pq

    from phenotypic.sdk_ import list_table_runs, read_embedding_table
    from tests._tiny_run import (
        _IMAGE_PAYLOADS,
        _with_cli_payloads,
        save_with_payload,
        toy_grid_embedder_of,
    )

    embedders = {
        "wells_rgb": toy_grid_embedder_of(name="wells"),
        "wells_gray": toy_grid_embedder_of(name="wells", layer="gray"),
    }
    image = _with_cli_payloads(plate, pipeline_file_a, embedders)
    try:
        store = save_with_payload(tmp_path, image)
    finally:
        _IMAGE_PAYLOADS.pop(id(image), None)
    (run,) = list_table_runs(store)
    entries = run.descriptor["embeddings"]
    assert set(entries) == {"toy-v1/rgb/grid-wells", "toy-v1/gray/grid-wells"}
    for layer in ("rgb", "gray"):
        key = f"toy-v1/{layer}/grid-wells"
        assert entries[key]["input"]["layer"] == layer
        assert run.embedding_path(key) == (
            store / "tables" / run.run_id / "embeddings/toy-v1" / layer / "grid-wells/table.parquet"
        )
        assert pq.read_metadata(run.embedding_path(key)).num_rows == 4
    layer_group = store / "tables" / run.run_id / "embeddings/toy-v1/gray/zarr.json"
    assert layer_group.is_file()
    rgb = read_embedding_table(store, "toy-v1/rgb/grid-wells").column("embedding")
    gray = read_embedding_table(store, "toy-v1/gray/grid-wells").column("embedding")
    assert not rgb.equals(gray)


def test_the_writer_refuses_an_input_layer_that_is_not_the_keys(tmp_path, plate):
    from dataclasses import replace

    from phenotypic.sdk_._embedding_tables import write_embedding_tables
    from tests._tiny_run import toy_grid_embedder_of

    payload = toy_grid_embedder_of(layer="gray").payload(plate)
    wrong = replace(
        payload,
        descriptor={**payload.descriptor, "input": {**payload.descriptor["input"], "layer": "rgb"}},
    )
    part = tmp_path / "x.ome.zarr.part"
    part.mkdir()
    with pytest.raises(ValueError, match="input.layer 'rgb', not its key's layer 'gray'"):
        write_embedding_tables(
            part, "1-unknown", {wrong.key: wrong}, objmap_sha256=None, array_sha256=None
        )
    assert not (part / "tables").exists()
