"""``aggregate_embedding_tables`` without torch (#281 P4 T4, spec E5, D6a, D7).

The CLI end-to-end tests in ``tests/integration/cli/test_embedding_deliverables.py``
need torch for the stub DINO backbone and skip without it. These build the
stores directly from the toy embedder's payloads, so the aggregation's
streaming, stale-skip, run selection and failure rules are pinned in every
environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._tiny_run import (
    TOY_DIM,
    payloads_of,
    resave_with_objmap,
    tables_for,
)

KEY = "toy-v1/crops"


def _store(out: Path, stem: str, image) -> Path:
    from phenotypic.sdk_ import zarr_store_path

    return image.save2zarr(
        zarr_store_path(out, "ds", stem),
        measurement_table=tables_for(image),
        embeddings=payloads_of(image),
    )


def _latest(store: Path):
    from phenotypic.sdk_ import list_table_runs

    return list_table_runs(store)[-1]


@pytest.fixture
def three_stores(tmp_path: Path, image_with_crop_payload):
    """Three stores of one image under ``<tmp>/out``; ``p3``'s crop entry is stale."""
    out = tmp_path / "out"
    stores = [_store(out, stem, image_with_crop_payload) for stem in ("p1", "p2", "p3")]
    resave_with_objmap(stores[2], zeros=True)
    return out, [_latest(store) for store in stores]


def test_rows_stream_one_row_group_per_store_and_stale_is_skipped(three_stores):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables

    out, runs = three_stores
    counts = aggregate_embedding_tables(out, runs)

    per_store = pq.read_metadata(runs[0].embedding_path("toy-v1", "crops")).num_rows
    assert counts == {KEY: {"rows": 2 * per_store, "stale_skipped": 1}}
    path = out / "deliverables" / "embeddings" / "toy-v1" / "crops.parquet"
    metadata = pq.read_metadata(path)
    assert metadata.num_row_groups == 2
    table = pq.read_table(path)
    assert table.column_names == [
        "Metadata_Dataset", "Metadata_ImageName", "Object_Label", "embedding",
    ]
    assert table.schema.field("embedding").type == pa.list_(pa.float16(), TOY_DIM)
    assert set(table.column("Metadata_ImageName").to_pylist()) == {"p1", "p2"}
    assert set(table.column("Metadata_Dataset").to_pylist()) == {"ds"}
    manifest = json.loads((path.parent.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["entries"][KEY]["stale_skipped"] == ["results/ds/zarr/p3.ome.zarr"]
    assert manifest["entries"][KEY]["rows"] == 2 * per_store


def test_a_run_without_embeddings_contributes_nothing(tmp_path, image_with_crop_payload):
    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables
    from phenotypic.sdk_ import zarr_store_path

    out = tmp_path / "out"
    with_embeddings = _store(out, "p1", image_with_crop_payload)
    without = image_with_crop_payload.save2zarr(
        zarr_store_path(out, "ds", "p2"),
        measurement_table=tables_for(image_with_crop_payload),
    )
    counts = aggregate_embedding_tables(out, [_latest(with_embeddings), _latest(without)])
    assert set(counts) == {KEY} and counts[KEY]["stale_skipped"] == 0


def test_the_certified_run_is_read_not_the_latest(
    tmp_path, image_with_crop_payload, pipeline_file_b
):
    """D6a: the run handed in is aggregated, even when the store has a later one."""
    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables
    from phenotypic.sdk_ import list_table_runs

    from tests._tiny_run import resave_with_pipeline

    out = tmp_path / "out"
    store = _store(out, "p1", image_with_crop_payload)
    certified = _latest(store)
    resave_with_pipeline(store, pipeline_file_b)  # adds B's folder, without embeddings
    latest = list_table_runs(store)[-1]
    assert latest.run_id != certified.run_id
    assert aggregate_embedding_tables(out, [latest]) == {}
    (current,) = [r for r in list_table_runs(store) if r.run_id == certified.run_id]
    counts = aggregate_embedding_tables(out, [current])
    assert counts[KEY]["rows"] > 0


def test_an_unreadable_entry_fails_with_the_store_and_the_remedy(three_stores):
    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables

    out, runs = three_stores
    runs[0].embedding_path("toy-v1", "crops").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match=r"p1\.ome\.zarr.*--mode measure"):
        aggregate_embedding_tables(out, runs)


def test_no_embeddings_removes_a_previous_deliverable(three_stores, measured_image):
    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables
    from phenotypic.sdk_ import zarr_store_path

    out, runs = three_stores
    aggregate_embedding_tables(out, runs)
    assert (out / "deliverables" / "embeddings").is_dir()
    plain = measured_image.save2zarr(
        zarr_store_path(out, "ds", "plain"), measurement_table=tables_for(measured_image)
    )
    assert aggregate_embedding_tables(out, [_latest(plain)]) == {}
    assert not (out / "deliverables" / "embeddings").exists()


def test_a_key_dropped_since_the_last_finalization_is_removed(
    tmp_path, image_with_two_models
):
    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables
    from phenotypic.sdk_ import zarr_store_path

    out = tmp_path / "out"
    store = image_with_two_models.save2zarr(
        zarr_store_path(out, "ds", "p1"),
        measurement_table=tables_for(image_with_two_models),
        embeddings=payloads_of(image_with_two_models),
    )
    assert set(aggregate_embedding_tables(out, [_latest(store)])) == {
        "toy-v1/crops", "toy-v2/crops",
    }
    one_model = {KEY: payloads_of(image_with_two_models)[KEY]}
    store = image_with_two_models.save2zarr(
        store, measurement_table=tables_for(image_with_two_models), embeddings=one_model
    )
    assert set(aggregate_embedding_tables(out, [_latest(store)])) == {KEY}
    assert not (out / "deliverables" / "embeddings" / "toy-v2").exists()
