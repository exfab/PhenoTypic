"""The embedding deliverables without torch (#281 P4 T4, P4 review).

The CLI end-to-end tests in ``tests/integration/cli/test_embedding_deliverables.py``
need torch for the stub DINO backbone and skip without it. These build the
stores directly from the toy embedder's payloads, so the aggregation's
streaming, stale-skip, run selection, all-or-nothing publication (review
I1), unresolved paths (I3), shard-path filter (M1b), one stale predicate
(M2), cleanup (M8) and out-of-memory rule (M7) are pinned in every
environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests._tiny_run import (
    TOY_DIM,
    payloads_of,
    resave_with_objmap,
    tables_for,
    toy_embedder_class,
)

KEY = "toy-v1/crops"


def _store(out: Path, stem: str, image, payloads=None) -> Path:
    from phenotypic.sdk_ import zarr_store_path

    return image.save2zarr(
        zarr_store_path(out, "ds", stem),
        measurement_table=tables_for(image),
        embeddings=payloads_of(image) if payloads is None else payloads,
    )


def _latest(store: Path):
    from phenotypic.sdk_ import list_table_runs

    return list_table_runs(store)[-1]


def _mirrored_plate(pipeline_file: Path):
    """The synth plate mirrored left to right, detected, as a ``full`` application."""
    from phenotypic import Image
    from phenotypic._core._provenance import initialize_cli_provenance
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector

    rgb = np.ascontiguousarray(load_synth_yeast_plate().rgb[:][:, ::-1])
    image = OtsuDetector().apply(Image(rgb))
    initialize_cli_provenance(image, pipeline_file, kind="full", input_filename="m.tif")
    return image


def _embed(image, embedder=None) -> dict:
    from phenotypic import ImagePipeline

    return ImagePipeline(meas={"emb": embedder or toy_embedder_class()()}).embed(image)


def _snapshot(out: Path) -> dict[str, tuple[bytes, int]]:
    root = out / "deliverables"
    if not root.exists():
        return {}
    return {
        p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def three_stores(tmp_path: Path, image_with_crop_payload, pipeline_file_a: Path):
    """Three stores under ``<tmp>/out``: ``p2`` is a mirrored plate, ``p3``'s entry is stale."""
    out = tmp_path / "out"
    mirrored = _mirrored_plate(pipeline_file_a)
    stores = [
        _store(out, "p1", image_with_crop_payload),
        _store(out, "p2", mirrored, _embed(mirrored)),
        _store(out, "p3", image_with_crop_payload),
    ]
    resave_with_objmap(stores[2], zeros=True)
    return out, [_latest(store) for store in stores]


def test_rows_stream_one_row_group_per_store_and_stale_is_skipped(three_stores):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables

    out, runs = three_stores
    counts = aggregate_embedding_tables(out, runs)

    rows = [pq.read_metadata(run.embedding_path("toy-v1", "crops")).num_rows for run in runs[:2]]
    assert counts == {KEY: {"rows": sum(rows), "stale_skipped": 1}}
    path = out / "deliverables" / "embeddings" / "toy-v1" / "crops.parquet"
    metadata = pq.read_metadata(path)
    assert metadata.num_row_groups == 2
    table = pq.read_table(path)
    assert table.column_names == [
        "Metadata_Dataset", "Metadata_ImageName", "Object_Label", "descriptor_index", "embedding",
    ]
    assert table.schema.field("embedding").type == pa.list_(pa.float16(), TOY_DIM)
    assert set(table.column("Metadata_ImageName").to_pylist()) == {"p1", "p2"}
    assert set(table.column("Metadata_Dataset").to_pylist()) == {"ds"}
    manifest = json.loads((path.parent.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["entries"][KEY]["stale_skipped"] == ["results/ds/zarr/p3.ome.zarr"]
    assert manifest["entries"][KEY]["rows"] == sum(rows)


def test_each_stores_rows_are_its_own_table(three_stores):
    """Review T4: ``p2`` is a different plate, and each store's rows in the
    deliverable are exactly its own embedding table, in order."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables
    from phenotypic.sdk_ import read_embedding_table

    out, runs = three_stores
    aggregate_embedding_tables(out, runs)
    published = pq.read_table(out / "deliverables/embeddings/toy-v1/crops.parquet")
    own = {}
    for stem, run in (("p1", runs[0]), ("p2", runs[1])):
        own[stem] = read_embedding_table(run.store, KEY)
        rows = published.filter(pc.equal(published["Metadata_ImageName"], stem))
        assert rows.select(own[stem].column_names).equals(own[stem]), stem
    assert not own["p1"].column("embedding").equals(own["p2"].column("embedding"))


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


# ---------------------------------------------------------------------------
# Review I1: a failure changes no deliverable
# ---------------------------------------------------------------------------


@pytest.fixture
def two_key_stores(tmp_path: Path, image_with_two_models, pipeline_file_a: Path):
    """Two stores holding ``toy-v1/crops`` and ``toy-v2/crops``, aggregated once."""
    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables

    out = tmp_path / "out"
    mirrored = _mirrored_plate(pipeline_file_a)
    stores = [
        _store(out, "p1", image_with_two_models),
        _store(
            out, "p2", mirrored,
            {**_embed(mirrored), **_embed(mirrored, toy_embedder_class()(token="toy-v2"))},
        ),
    ]
    runs = [_latest(store) for store in stores]
    aggregate_embedding_tables(out, runs)
    return out, runs


def test_a_failure_in_a_later_key_leaves_every_deliverable_as_it_was(two_key_stores):
    """``toy-v1`` sorts first, so it is the table a non-atomic writer would have
    replaced before ``toy-v2`` of ``p2`` failed: no file changes, in bytes or
    mtime, and no temporary is left behind."""
    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables

    out, runs = two_key_stores
    runs[1].embedding_path("toy-v2", "crops").write_bytes(b"corrupt")
    before = _snapshot(out)
    assert {"embeddings/toy-v1/crops.parquet", "embeddings/toy-v2/crops.parquet"} <= set(before)
    with pytest.raises(RuntimeError, match=r"p2\.ome\.zarr"):
        aggregate_embedding_tables(out, runs)
    assert _snapshot(out) == before
    assert not list((out / "deliverables").rglob("*.tmp"))


def test_staging_writes_no_deliverable_until_promoted(three_stores):
    from phenotypic._cli._cli_embedding_deliverables import stage_embedding_deliverables

    out, runs = three_stores
    staged = stage_embedding_deliverables(out, runs)
    assert not (out / "deliverables/embeddings/toy-v1/crops.parquet").exists()
    assert not (out / "deliverables/embeddings/manifest.json").exists()
    staged.discard()
    assert not (out / "deliverables" / "embeddings").exists()


def test_a_schema_mismatch_between_stores_fails_and_changes_nothing(
    tmp_path, image_with_crop_payload, pipeline_file_a
):
    """One key, two stores of different vector length: one deliverable cannot
    hold both, so finalization fails before any write."""
    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables

    out = tmp_path / "out"
    mirrored = _mirrored_plate(pipeline_file_a)

    class Wider(toy_embedder_class()):
        def entry_descriptor(self) -> dict:
            return {**super().entry_descriptor(), "dim": TOY_DIM + 1}

        def _embed_boxes(self, image, boxes):
            return np.repeat(boxes.sum(axis=1)[:, None], TOY_DIM + 1, axis=1).astype(np.float32)

    stores = [
        _store(out, "p1", image_with_crop_payload),
        _store(out, "p2", mirrored, _embed(mirrored, Wider())),
    ]
    with pytest.raises(RuntimeError, match="schema"):
        aggregate_embedding_tables(out, [_latest(s) for s in stores])
    assert _snapshot(out) == {}


# ---------------------------------------------------------------------------
# Review I3: unresolved paths
# ---------------------------------------------------------------------------


def test_a_symlinked_results_folder_is_the_runs_own(tmp_path, image_with_crop_payload):
    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables
    from phenotypic.sdk_ import zarr_store_path

    out = tmp_path / "out"
    out.mkdir()
    elsewhere = tmp_path / "bigdisk" / "results"
    elsewhere.mkdir(parents=True)
    (out / "results").symlink_to(elsewhere)
    store = _store(out, "p1", image_with_crop_payload)
    assert store == zarr_store_path(out, "ds", "p1")
    counts = aggregate_embedding_tables(out, [_latest(store)])
    assert counts[KEY]["rows"] > 0
    manifest = json.loads((out / "deliverables/embeddings/manifest.json").read_text(encoding="utf-8"))
    assert manifest["entries"][KEY]["stores"] == 1


def test_an_embedding_file_symlinked_out_of_the_store_is_read(tmp_path, image_with_crop_payload):
    """DVC's ``cache.type=symlink``: the table is a symlink into a cache
    outside the store, and it is still that store's file."""
    from phenotypic.sdk_ import read_embedding_table

    out = tmp_path / "out"
    store = _store(out, "p1", image_with_crop_payload)
    table = _latest(store).embedding_path("toy-v1", "crops")
    cache = tmp_path / "dvc-cache" / "ab" / "cdef"
    cache.parent.mkdir(parents=True)
    os.replace(table, cache)
    table.symlink_to(cache)
    assert read_embedding_table(store, KEY).num_rows > 0


def test_an_entry_path_that_climbs_out_of_the_store_is_refused(tmp_path, image_with_crop_payload):
    from phenotypic.sdk_ import EmbeddingTableUnreadableError
    from phenotypic.sdk_._embedding_tables import read_embedding_entry

    store = _store(tmp_path / "out", "p1", image_with_crop_payload)
    entry = dict(_latest(store).descriptor["embeddings"][KEY])
    entry["files"] = [{**entry["files"][0], "path": "../../elsewhere.parquet"}]
    with pytest.raises(EmbeddingTableUnreadableError, match="outside the store"):
        read_embedding_entry(store, KEY, entry)


# ---------------------------------------------------------------------------
# Review M1b: the shard path embeds only what the projection kept
# ---------------------------------------------------------------------------


def test_the_shard_path_embeds_only_the_sources_the_projection_kept(monkeypatch, tmp_path):
    from phenotypic._cli import _cli_finalize_fanout
    from phenotypic._cli._cli_finalize_run import _embedding_runs

    sources = {tmp_path / f"s{i}": "ds" for i in range(3)}
    runs = ["run0", "run1", None]
    monkeypatch.setattr(
        _cli_finalize_fanout, "_work_ids_for_sources", lambda out, src: ["w0", "w1", "w2"]
    )
    kept = _embedding_runs(tmp_path, sources, runs, kept_work_ids=["w1", "w2"])
    assert kept == ["run1"]                       # w0 excluded; w2 has no run
    assert _embedding_runs(tmp_path, sources, runs, kept_work_ids=None) == ["run0", "run1"]


def test_a_run_that_embeds_nothing_declares_no_embedding_output(finished_tiny_run):
    """Review M1a, its empty half: no embedder, nothing declared."""
    from phenotypic.sdk_ import aggregate_publication_marker_path

    proof = json.loads(aggregate_publication_marker_path(finished_tiny_run).read_text(encoding="utf-8"))
    assert not any(name.startswith("embeddings") for name in proof["required_outputs"])
    assert not (finished_tiny_run / "deliverables" / "embeddings").exists()


def test_the_proof_names_each_published_embedding_file(three_stores):
    from phenotypic._cli._cli_embedding_deliverables import (
        aggregate_embedding_tables,
        published_embedding_files,
    )

    out, runs = three_stores
    assert published_embedding_files(out) == {}
    aggregate_embedding_tables(out, runs)
    assert published_embedding_files(out) == {
        "embeddings_manifest": out / "deliverables/embeddings/manifest.json",
        "embeddings/toy-v1/crops": out / "deliverables/embeddings/toy-v1/crops.parquet",
    }


# ---------------------------------------------------------------------------
# Review M2 and M8
# ---------------------------------------------------------------------------


def test_an_entry_without_a_stale_flag_reads_as_stale_everywhere(tmp_path, image_with_crop_payload):
    """M2: one predicate. An entry whose ``stale`` is missing is skipped and
    counted by the aggregation, and refused by the reader, never read."""
    import dataclasses

    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables
    from phenotypic.sdk_ import EmbeddingTableUnreadableError
    from phenotypic.sdk_._embedding_tables import (
        embedding_entry_is_stale,
        read_embedding_entry,
    )

    out = tmp_path / "out"
    run = _latest(_store(out, "p1", image_with_crop_payload))
    entry = {k: v for k, v in run.descriptor["embeddings"][KEY].items() if k != "stale"}
    assert embedding_entry_is_stale(entry)
    assert embedding_entry_is_stale({**entry, "stale": None})
    assert not embedding_entry_is_stale({**entry, "stale": False})
    edited = dataclasses.replace(
        run, descriptor={**run.descriptor, "embeddings": {KEY: entry}}
    )
    assert aggregate_embedding_tables(out, [edited]) == {KEY: {"rows": 0, "stale_skipped": 1}}
    with pytest.raises(EmbeddingTableUnreadableError, match="stale"):
        read_embedding_entry(run.store, KEY, entry)


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


def test_cleanup_removes_only_what_finalization_wrote(three_stores, measured_image):
    """M8: a user's file under ``deliverables/embeddings/`` survives."""
    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables
    from phenotypic.sdk_ import zarr_store_path

    out, runs = three_stores
    aggregate_embedding_tables(out, runs)
    notes = out / "deliverables" / "embeddings" / "NOTES.txt"
    notes.write_text("mine", encoding="utf-8")
    plain = measured_image.save2zarr(
        zarr_store_path(out, "ds", "plain"), measurement_table=tables_for(measured_image)
    )
    aggregate_embedding_tables(out, [_latest(plain)])
    assert notes.read_text(encoding="utf-8") == "mine"
    assert sorted(p.name for p in notes.parent.rglob("*")) == ["NOTES.txt"]


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


# ---------------------------------------------------------------------------
# Review M7: out of memory is never the image's terminal failure
# ---------------------------------------------------------------------------


class _OutOfMemory(Exception):
    """Stands in for ``torch.OutOfMemoryError`` where torch is absent."""


@pytest.mark.parametrize("error", [MemoryError("host"), "torch"])
def test_out_of_memory_leaves_embed_and_measure_unwrapped(monkeypatch, detected_plate, error):
    import sys
    import types

    from phenotypic import ImagePipeline

    if error == "torch":
        torch = sys.modules.get("torch")
        if torch is None:
            # torch.cuda.OutOfMemoryError only: the older-torch spelling.
            fake = types.ModuleType("torch")
            fake.cuda = types.SimpleNamespace(OutOfMemoryError=_OutOfMemory)
            monkeypatch.setitem(sys.modules, "torch", fake)
            error = _OutOfMemory("cuda")
        else:
            error = torch.OutOfMemoryError("cuda")
    embedder = toy_embedder_class()()

    def _boom(self, image, boxes):
        raise error

    monkeypatch.setattr(type(embedder), "_embed_boxes", _boom)
    with pytest.raises(type(error)) as raised:
        embedder.measure(detected_plate)
    assert raised.value is error
    with pytest.raises(type(error)) as raised:
        ImagePipeline(meas={"emb": embedder}).embed(detected_plate)
    assert raised.value is error


def test_an_embedder_that_runs_out_of_memory_records_no_terminal_failure(
    tmp_path, tiny_inputs, monkeypatch
):
    """The image is left pending, not journalled terminal, so the next run
    retries it (review M7). torch-free: the embedder fails before its
    backbone loads, and ``--skip-validation`` passes the torch preflight."""
    from phenotypic._cli._cli_failure_tracker import terminal_failure_index
    from phenotypic.measure import MeasureDinoEmbedding

    from tests._tiny_run import run_cli, with_dino_crops

    def _oom(self, image):
        raise MemoryError("the node ran out")

    monkeypatch.setattr(MeasureDinoEmbedding, "_operate", _oom)
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, pipeline=with_dino_crops(tmp_path), extra=["--skip-validation"])
    assert result.exit_code != 0
    assert terminal_failure_index(out) == {}


# ---------------------------------------------------------------------------
# P4 review Q1: a key with several descriptors publishes, warns, tags rows
# ---------------------------------------------------------------------------

#: A pinned revision whose canonical JSON sorts BEFORE the toy's ``null``, so
#: an order by canonical JSON and the first-appearance order disagree.
_PINNED = "0123456789abcdef0123456789abcdef01234567"


def _repinned_embedder():
    """The toy embedder with another revision and pooling: a second descriptor."""

    class Repinned(toy_embedder_class()):
        def entry_descriptor(self) -> dict:
            base = super().entry_descriptor()
            return {**base, "model": {**base["model"], "revision": _PINNED}, "pooling": "box_mean"}

    return Repinned()


@pytest.fixture
def mixed_descriptor_stores(tmp_path, image_with_crop_payload, pipeline_file_a):
    """``p1`` and ``p3`` carry the toy descriptor; ``p2`` (a mirrored plate) another."""
    out = tmp_path / "out"
    mirrored = _mirrored_plate(pipeline_file_a)
    stores = [
        _store(out, "p1", image_with_crop_payload),
        _store(out, "p2", mirrored, _embed(mirrored, _repinned_embedder())),
        _store(out, "p3", image_with_crop_payload),
    ]
    return out, [_latest(store) for store in stores]


def test_a_two_descriptor_key_publishes_warns_and_tags_each_row(mixed_descriptor_stores, caplog):
    import logging

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables

    out, runs = mixed_descriptor_stores
    with caplog.at_level(logging.WARNING, logger="phenotypic._cli._cli_embedding_deliverables"):
        counts = aggregate_embedding_tables(out, runs)
    assert counts[KEY]["stale_skipped"] == 0

    (warning,) = [r.getMessage() for r in caplog.records if "descriptor" in r.getMessage()]
    assert KEY in warning
    assert "[0]" in warning and "[1]" in warning
    assert "pooling box_sum" in warning and "pooling box_mean" in warning
    assert f"revision {_PINNED[:8]}" in warning
    assert "2 stores" in warning and "1 store" in warning
    assert "results/ds/zarr/p2.ome.zarr" in warning

    manifest = json.loads((out / "deliverables/embeddings/manifest.json").read_text(encoding="utf-8"))
    descriptors = manifest["entries"][KEY]["descriptors"]
    # First appearance in sorted store order: p1's descriptor, then p2's.
    assert [d["pooling"] for d in descriptors] == ["box_sum", "box_mean"]
    assert descriptors[1]["model"]["revision"] == _PINNED

    table = pq.read_table(out / "deliverables/embeddings/toy-v1/crops.parquet")
    assert table.schema.field("descriptor_index").type == pa.int16()
    assert not table.schema.field("descriptor_index").nullable
    for stem, index in (("p1", 0), ("p2", 1), ("p3", 0)):
        rows = table.filter(pc.equal(table["Metadata_ImageName"], stem))
        assert rows.num_rows > 0
        assert set(rows.column("descriptor_index").to_pylist()) == {index}, stem
        assert descriptors[index]["pooling"] == runs[int(stem[1]) - 1].descriptor[
            "embeddings"][KEY]["pooling"]


def test_a_mixed_descriptor_deliverable_is_byte_stable(mixed_descriptor_stores):
    """The descriptor order and the indices come from the sorted store order,
    never from the order the runs are handed in."""
    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables

    out, runs = mixed_descriptor_stores
    root = out / "deliverables/embeddings"
    aggregate_embedding_tables(out, runs)
    first = {p.name: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    aggregate_embedding_tables(out, list(reversed(runs)))
    second = {p.name: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert first == second and set(first) == {"crops.parquet", "manifest.json"}


def test_a_single_descriptor_key_tags_every_row_zero(three_stores, caplog):
    import logging

    import pyarrow as pa
    import pyarrow.parquet as pq

    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables

    out, runs = three_stores
    with caplog.at_level(logging.WARNING, logger="phenotypic._cli._cli_embedding_deliverables"):
        aggregate_embedding_tables(out, runs)
    assert not [r for r in caplog.records if "descriptors" in r.getMessage()]
    table = pq.read_table(out / "deliverables/embeddings/toy-v1/crops.parquet")
    assert table.schema.field("descriptor_index").type == pa.int16()
    assert set(table.column("descriptor_index").to_pylist()) == {0}
    manifest = json.loads((out / "deliverables/embeddings/manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["entries"][KEY]["descriptors"]) == 1


def test_schema_json_describes_the_descriptor_index(three_stores):
    from phenotypic._cli._cli_embedding_deliverables import (
        aggregate_embedding_tables,
        embedding_table_schemas,
    )
    from phenotypic.util._measurement_outputs import build_output_schema

    out, runs = three_stores
    aggregate_embedding_tables(out, runs)
    schema = build_output_schema({}, embedding_tables=embedding_table_schemas(out))
    rows = {row["name"]: row for row in schema["tables"][f"embeddings/{KEY}"]}
    assert list(rows) == [
        "Metadata_Dataset", "Metadata_ImageName", "Object_Label", "descriptor_index", "embedding",
    ]
    assert rows["descriptor_index"]["dtype"] == "Int16"
    assert "manifest.json" in rows["descriptor_index"]["desc"]
    assert "descriptors" in rows["descriptor_index"]["desc"]


def test_the_readme_explains_the_descriptor_index():
    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_readme_generator import READMEGenerator

    from tests.unit.cli._preflight_support import make_config

    pipeline = ImagePipeline(meas={"emb": toy_embedder_class()()})
    section = READMEGenerator(make_config(), pipeline)._generate_embeddings_section()
    assert "`descriptor_index`" in section
    assert "`descriptors`" in section


def test_stores_differing_only_in_device_get_two_descriptors(
    tmp_path, image_with_crop_payload, pipeline_file_a, caplog
):
    """Q1 follow-up: ``device="auto"`` picks CUDA on a GPU node and the CPU
    elsewhere with the same pipeline, so the entry's ``model.device`` is the
    only difference, and it must still split the key's descriptors."""
    import logging

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables

    # One class and one parameter set for both stores: only the device the
    # (pretend) backbone resolved differs, as with device="auto".
    ran_on = {"device": "cpu"}

    class Placed(toy_embedder_class()):
        def entry_descriptor(self) -> dict:
            base = super().entry_descriptor()
            return {**base, "model": {**base["model"], "device": ran_on["device"]}}

    out = tmp_path / "out"
    mirrored = _mirrored_plate(pipeline_file_a)
    on_cpu = _embed(image_with_crop_payload, Placed())
    ran_on["device"] = "cuda"
    on_gpu = _embed(mirrored, Placed())
    cpu_entry, gpu_entry = (dict(p[KEY].descriptor) for p in (on_cpu, on_gpu))
    differing = {n for n in cpu_entry if cpu_entry[n] != gpu_entry[n]} - {"binds"}
    assert differing == {"model"}
    assert {k for k in cpu_entry["model"] if cpu_entry["model"][k] != gpu_entry["model"][k]} == {
        "device"
    }

    runs = [
        _latest(_store(out, "p1", image_with_crop_payload, on_cpu)),
        _latest(_store(out, "p2", mirrored, on_gpu)),
    ]
    with caplog.at_level(logging.WARNING, logger="phenotypic._cli._cli_embedding_deliverables"):
        aggregate_embedding_tables(out, runs)
    (warning,) = [r.getMessage() for r in caplog.records if "descriptor" in r.getMessage()]
    assert "[0] revision none, device cpu" in warning
    assert "[1] revision none, device cuda" in warning
    manifest = json.loads((out / "deliverables/embeddings/manifest.json").read_text(encoding="utf-8"))
    assert [d["model"]["device"] for d in manifest["entries"][KEY]["descriptors"]] == ["cpu", "cuda"]
    table = pq.read_table(out / "deliverables/embeddings/toy-v1/crops.parquet")
    for stem, index in (("p1", 0), ("p2", 1)):
        rows = table.filter(pc.equal(table["Metadata_ImageName"], stem))
        assert set(rows.column("descriptor_index").to_pylist()) == {index}, stem


def test_the_readme_names_the_device_as_a_cause():
    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_readme_generator import READMEGenerator

    from tests.unit.cli._preflight_support import make_config

    pipeline = ImagePipeline(meas={"emb": toy_embedder_class()()})
    section = READMEGenerator(make_config(), pipeline)._generate_embeddings_section()
    assert 'device="auto"' in section


# ---------------------------------------------------------------------------
# P4 review Q2: the remedy is a whole-tree --mode measure
# ---------------------------------------------------------------------------


def test_every_unreadable_store_is_named_with_the_whole_tree_remedy(three_stores):
    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables

    out, runs = three_stores
    for run in runs[:2]:
        run.embedding_path("toy-v1", "crops").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError) as raised:
        aggregate_embedding_tables(out, runs)
    message = str(raised.value)
    assert "results/ds/zarr/p1.ome.zarr" in message
    assert "results/ds/zarr/p2.ome.zarr" in message
    assert "--mode measure" in message and f"--output {out}" in message
    assert "every image" in message
    assert not (out / "deliverables" / "embeddings").exists()
