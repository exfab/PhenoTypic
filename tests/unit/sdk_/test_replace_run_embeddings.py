"""``replace_run_embeddings``: one run's embeddings, root last (#282 spec §4.2, §5).

Stage 4's store transaction. It rewrites the ``embeddings`` block of one run
and nothing else: measurements, metadata, the other runs, ``upstream``,
figures and the journal are carried, and the root is written last.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from phenotypic.sdk_ import (
    TableRun,
    list_table_runs,
    ngff_,
    replace_run_embeddings,
)
from phenotypic.sdk_._measurement_tables import (
    StoreRootChangedError,
    _rewrite_store_tables,
    array_digest,
    objmap_digest,
)
from tests._tiny_run import (
    objects_payload_for,
    root_sha256,
    root_tables,
    touch_root,
    tree_digest,
)

KEY = "toy-v1/rgb/objects"


# ---------------------------------------------------------------------------
# Module-local helpers
# ---------------------------------------------------------------------------


def _level0(store: Path, path: str):
    """The level-0 array at store-relative *path*, read through zarr-python."""
    return zarr.open_group(str(store), mode="r")[f"{path}/0"][...]


def digests(store: Path, *, root: bool = True) -> dict:
    """The digest keywords of a commit, read from the live store (plan review I11).

    ``objmap_sha256`` from the stored level-0 objmap and ``array_sha256`` of
    every level-0 series the store holds; ``expected_root_sha256`` too, when
    *root*.
    """
    labels = ngff_.read_phenotypic_attributes(store)[ngff_.PhenotypicAttr.LABELS]
    out: dict = {
        "objmap_sha256": objmap_digest(_level0(store, labels[ngff_.OBJMAP_LABEL])),
        "array_sha256": {
            series: array_digest(_level0(store, series))
            for series in ("rgb", "gray")
            if (store / series).is_dir()
        },
    }
    if root:
        out["expected_root_sha256"] = root_sha256(store)
    return out


def owned_run(store: Path, run_id: str) -> TableRun:
    """The :class:`TableRun` *run_id* of *store*, as its root declares it."""
    return next(run for run in list_table_runs(store) if run.run_id == run_id)


def other_run_dir(store: Path, run_id: str) -> Path:
    """The folder of the one run of *store* that is not *run_id*."""
    (other,) = [run for run in list_table_runs(store) if run.run_id != run_id]
    return other.folder


def consolidate_store(store: Path) -> None:
    """Publish *store* consolidated: a part, ``_consolidate_store_part``, then the promote.

    The way a consolidated store is published today; a promoted store's root
    is never consolidated in place.
    """
    _rewrite_store_tables(
        store,
        plan=lambda _phenotypic: lambda _part: None,
        durable=False,
        commit_guard=None,
        clear_tables=False,
        consolidate=True,
    )
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    assert "consolidated_metadata" in root, "the fixture must really be consolidated"


def leftover_parts(store: Path) -> list[Path]:
    """Every ``.part`` sibling of *store* (``ngff_.new_part_path`` names them ``.<name>.<uuid>.part``)."""
    return list(store.parent.glob(f".{store.name}.*{ngff_.PART_SUFFIX}"))


def _files(store: Path) -> dict[str, tuple[int, bytes]]:
    """``{relative path: (inode, bytes)}`` of every file under *store*."""
    return {
        p.relative_to(store).as_posix(): (p.stat().st_ino, p.read_bytes())
        for p in store.rglob("*")
        if p.is_file()
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_the_part_glob_sees_a_part(staged_store_with_tables):
    """The no-part assertions below can fail: the glob matches a real part name."""
    store, _ = staged_store_with_tables
    part = ngff_.new_part_path(store)
    part.mkdir()
    try:
        assert leftover_parts(store) == [part]
    finally:
        part.rmdir()


def test_only_the_runs_embeddings_change(staged_store_with_tables):
    store, run_id = staged_store_with_tables
    before = root_tables(store)
    assert before["upstream"], "the fixture must carry an upstream entry"
    assert len(before["runs"]) == 2, "the fixture must carry a second run"
    replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))
    after = root_tables(store)
    assert {k: v for k, v in after["runs"].items() if k != run_id} == \
           {k: v for k, v in before["runs"].items() if k != run_id}
    assert after["upstream"] == before["upstream"]
    assert set(after["runs"][run_id]["embeddings"]) == {KEY}
    assert {k: v for k, v in after["runs"][run_id].items() if k != "embeddings"} == \
           {k: v for k, v in before["runs"][run_id].items() if k != "embeddings"}


def test_everything_but_the_runs_embeddings_and_the_root_is_carried(
    staged_store_with_tables,
):
    """Pixels, labels, the other run and this run's tables are the same inodes.

    Group documents (``zarr.json``) are compared by bytes: the table writer
    re-ensures ``tables/`` and ``tables/<run>/`` by atomic rename, which
    gives them new inodes without writing through the live store's links.
    """
    store, run_id = staged_store_with_tables
    before = _files(store)
    replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))
    after = _files(store)
    embeddings = f"tables/{run_id}/embeddings/"
    new = {p for p in after if p.startswith(embeddings)}
    assert new, "the write must add the run's embedding files"
    assert set(after) - new == set(before)
    carried = sorted(p for p in before if p != "zarr.json")
    data = [p for p in carried if not p.endswith("zarr.json")]
    assert data, "the fixture must hold data files to carry"
    assert {p: after[p][0] for p in data} == {p: before[p][0] for p in data}
    assert {p: after[p][1] for p in carried} == {p: before[p][1] for p in carried}


def test_measurement_and_metadata_inodes_are_carried(staged_store_with_tables):
    store, run_id = staged_store_with_tables
    run = owned_run(store, run_id)
    inodes = {p: p.stat().st_ino for p in (run.measurements_path(), run.metadata_path())}
    replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))
    assert {p: p.stat().st_ino for p in inodes} == inodes


def test_other_runs_files_are_byte_equal(staged_store_with_tables):
    store, run_id = staged_store_with_tables
    other = other_run_dir(store, run_id)
    digest = tree_digest(other)
    replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))
    assert tree_digest(other) == digest


def test_the_store_own_digests_write_the_entry_fresh(staged_store_with_tables):
    """The control for the stale test below: the same payload, live digests, ``stale: false``."""
    store, run_id = staged_store_with_tables
    replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))
    assert root_tables(store)["runs"][run_id]["embeddings"][KEY]["stale"] is False


def test_stale_is_recomputed_against_the_passed_digests(staged_store_with_tables):
    store, run_id = staged_store_with_tables
    replace_run_embeddings(store, run_id, objects_payload_for(store),
                           expected_root_sha256=root_sha256(store),
                           objmap_sha256="0" * 64, array_sha256={})
    assert root_tables(store)["runs"][run_id]["embeddings"][KEY]["stale"] is True


def _two_model_payloads(store: Path) -> dict:
    """:func:`objects_payload_for` plus a ``toy-v2/rgb/objects`` payload of the same image."""
    from phenotypic import ImagePipeline
    from tests._tiny_run import _close_open_application, _load_store, toy_embedder_class

    image = _load_store(store)
    _close_open_application(image)
    payloads = ImagePipeline(meas={"v2": toy_embedder_class()(token="toy-v2")}).embed(image)
    return {**objects_payload_for(store), **payloads}


def test_a_second_write_replaces_the_block(staged_store_with_tables):
    """The block is a total description of the write: an entry it lacks goes, files and all."""
    store, run_id = staged_store_with_tables
    two = _two_model_payloads(store)
    assert set(two) == {KEY, "toy-v2/rgb/objects"}
    replace_run_embeddings(store, run_id, two, **digests(store))
    run = owned_run(store, run_id)
    dropped = run.embedding_path("toy-v2/rgb/objects")
    assert dropped.is_file()
    inode = run.embedding_path(KEY).stat().st_ino
    replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))
    assert set(root_tables(store)["runs"][run_id]["embeddings"]) == {KEY}
    assert not dropped.exists()
    assert run.embedding_path(KEY).stat().st_ino != inode


def test_empty_payloads_leave_the_run_with_no_embeddings(staged_store_with_tables):
    store, run_id = staged_store_with_tables
    replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))
    replace_run_embeddings(store, run_id, {}, **digests(store))
    assert "embeddings" not in root_tables(store)["runs"][run_id]
    assert not (owned_run(store, run_id).folder / "embeddings").exists()


def test_a_changed_root_is_refused_and_leaves_no_part(staged_store_with_tables):
    store, run_id = staged_store_with_tables
    planned = root_sha256(store)
    touch_root(store)
    with pytest.raises(StoreRootChangedError):
        replace_run_embeddings(store, run_id, objects_payload_for(store),
                               expected_root_sha256=planned, **digests(store, root=False))
    assert not leftover_parts(store)
    assert "embeddings" not in root_tables(store)["runs"][run_id]
    replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))  # the retry
    assert set(root_tables(store)["runs"][run_id]["embeddings"]) == {KEY}


def test_a_root_changed_during_the_build_is_refused_and_leaves_no_part(
    staged_store_with_tables, monkeypatch
):
    """The second check, immediately before the promote: a root replaced mid-write."""
    import phenotypic.sdk_._embedding_tables as embedding_tables

    store, run_id = staged_store_with_tables
    real = embedding_tables.write_embedding_tables

    def _write_then_touch(*args, **kwargs):
        written = real(*args, **kwargs)
        touch_root(store)
        return written

    monkeypatch.setattr(embedding_tables, "write_embedding_tables", _write_then_touch)
    with pytest.raises(StoreRootChangedError):
        replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))
    assert not leftover_parts(store)
    assert "embeddings" not in root_tables(store)["runs"][run_id]


def test_a_failed_write_leaves_the_store_unchanged(staged_store_with_tables):
    """Root last: a payload refused inside the part leaves the live root's bytes and no part."""
    store, run_id = staged_store_with_tables
    (payload,) = objects_payload_for(store).values()
    planned = root_sha256(store)
    with pytest.raises(ValueError, match="given under the key"):
        replace_run_embeddings(store, run_id, {"toy-v9/rgb/objects": payload},
                               **digests(store))
    assert root_sha256(store) == planned
    assert not leftover_parts(store)


def test_a_consolidated_root_stays_consolidated(staged_store_with_tables):
    store, run_id = staged_store_with_tables
    consolidate_store(store)
    replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))
    meta = json.loads((store / "zarr.json").read_text(encoding="utf-8"))["consolidated_metadata"]
    assert any(k.endswith(f"{run_id}/embeddings/toy-v1/rgb/objects") for k in meta["metadata"])


def test_an_independent_reader_sees_the_table(staged_store_with_tables):
    store, run_id = staged_store_with_tables
    replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))
    group = zarr.open_group(str(store), mode="r")
    assert f"tables/{run_id}/embeddings/toy-v1/rgb/objects" in group
    t = pq.read_table(store / f"tables/{run_id}/embeddings/toy-v1/rgb/objects/table.parquet")
    assert t.schema.field("embedding").type == pa.list_(pa.float16(), 4)
    assert t.num_rows > 0, "the synth plate has colonies, so the table has rows"


def test_the_returned_run_is_the_promoted_roots(staged_store_with_tables):
    store, run_id = staged_store_with_tables
    run = replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))
    declared = owned_run(store, run_id)
    assert (run.run_id, run.order, run.pipeline_sha256) == \
           (declared.run_id, declared.order, declared.pipeline_sha256)
    assert dict(run.descriptor) == dict(declared.descriptor)
    assert run.embedding_path(KEY).is_file()


def test_an_unknown_run_is_refused_before_any_part(staged_store_with_tables):
    store, _ = staged_store_with_tables
    with pytest.raises(ValueError, match="no run '9-unknown'"):
        replace_run_embeddings(store, "9-unknown", objects_payload_for(store), **digests(store))
    assert not leftover_parts(store)


def test_a_store_without_v2_tables_is_refused_before_any_part(tmp_path, measured_image):
    """A table-less store (schema 0) has no run to embed into."""
    store = measured_image.save2zarr(tmp_path / "bare.ome.zarr")
    assert "tables" not in ngff_.read_phenotypic_attributes(store)
    planned = root_sha256(store)
    with pytest.raises(ValueError, match=r"no run '1-unknown'.*schema 0"):
        replace_run_embeddings(store, "1-unknown", objects_payload_for(store), **digests(store))
    assert root_sha256(store) == planned
    assert not leftover_parts(store)
