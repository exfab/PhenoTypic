"""Builder DAG manifest: the ``store`` key and the schema-version gate.

The per-node artifact moved from a single ``.h5`` file to an ``.ome.zarr``
store, which renames a GUI-visible manifest key. ``MANIFEST_VERSION`` is
introduced here so a manifest written by an older session is rebuilt rather
than read back through a key that no longer exists.
"""
from __future__ import annotations

import pytest

from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._state import (
    BlockNode, Edge, _DagBuilderScope, _DagBuilderState, _new_block_id,
)


def _linear_root_state(op_blocks):
    scope = _DagBuilderScope()  # __post_init__ seeds InputImage at index 0
    scope.blocks.extend(op_blocks)
    prev = scope.blocks[0].block_id
    for block in op_blocks:
        scope.edges.append(
                Edge(
                        edge_id=_new_block_id(), source_block_id=prev,
                        source_port="out", target_block_id=block.block_id,
                        target_port="in", kind="image",
                )
        )
        prev = block.block_id
    return _DagBuilderState(root=scope)


@pytest.fixture
def cached_scope(tmp_path, monkeypatch):
    """Compute one real root scope and hand back its manifest."""
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    blur = BlockNode(
            block_id=_new_block_id(), class_name="BlurGauss", params={"sigma": 1},
    )
    state = _linear_root_state([blur])
    manifest = pc.compute_scope("s", state, [], None, None, None)
    assert manifest["error"] is None, manifest["error"]
    return state, manifest


def test_manifest_node_key_is_store_not_hdf(cached_scope) -> None:
    """GUI-visible contract change; a stale 'hdf' key must not be read."""
    _state, manifest = cached_scope
    assert manifest["nodes"], "expected at least the input node"
    for node in manifest["nodes"].values():
        assert "hdf" not in node
        assert node["store"].endswith(".ome.zarr")


def test_manifest_nodes_describe_the_store_they_point_at(cached_scope) -> None:
    """``layers``/``shape``/``num_objects`` are read off the store, not guessed."""
    _state, manifest = cached_scope
    sdir = pc.scope_dir("s", [])
    by_store = {node["store"]: node for node in manifest["nodes"].values()}
    assert set(by_store) == {"base_00.ome.zarr", "00_BlurGauss.ome.zarr"}
    for name, node in by_store.items():
        assert (sdir / name).is_dir()
        assert node["shape"] == [600, 800]
        # ``full_layers=True`` writes a complete snapshot: every layer present.
        assert node["layers"] == ["rgb", "gray", "detect_mat", "objmap"]
    # A VALUE read out of the store, not a constant: the synthetic plate ships
    # 96 labelled colonies, and enhancing invalidates the detection.
    assert by_store["base_00.ome.zarr"]["num_objects"] == 96
    assert by_store["00_BlurGauss.ome.zarr"]["num_objects"] == 0


def test_manifest_carries_a_schema_version(cached_scope) -> None:
    _state, manifest = cached_scope
    assert pc.MANIFEST_VERSION >= 2
    assert manifest["version"] == pc.MANIFEST_VERSION


def test_a_manifest_without_a_version_is_treated_as_stale(cached_scope) -> None:
    """Every cache written before this change lacks the field entirely.

    Without the reader half, a pre-existing manifest passes the fingerprint
    check and is returned as a cache hit, and the very next read hits
    ``node["store"]`` on a dict that only has ``"hdf"``.
    """
    state, manifest = cached_scope
    stale = dict(manifest)
    stale.pop("version", None)
    stale["nodes"] = {
        block_id: {**node, "hdf": node.pop("store")}
        for block_id, node in ((k, dict(v)) for k, v in manifest["nodes"].items())
    }
    pc.write_manifest("s", [], stale)
    assert pc.read_manifest("s", []).get("version") is None

    rebuilt = pc.compute_scope("s", state, [], None, None, None)
    assert rebuilt["version"] == pc.MANIFEST_VERSION
    assert all("store" in node for node in rebuilt["nodes"].values())


def test_a_matching_version_and_fingerprint_is_still_a_cache_hit(
        cached_scope, monkeypatch,
) -> None:
    """The version gate must not defeat caching outright."""
    state, _manifest = cached_scope

    def _fail(*_args, **_kwargs):
        raise AssertionError("cache hit should not recompute")

    monkeypatch.setattr(
            "phenotypic._core._image_pipeline.ImagePipeline.apply_with_intermediates",
            _fail,
    )
    assert pc.compute_scope("s", state, [], None, None, None)["version"] == (
        pc.MANIFEST_VERSION
    )


def test_the_error_path_manifest_carries_the_version(tmp_path, monkeypatch) -> None:
    """Otherwise a failed scope is permanently re-read as stale."""
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    bad = BlockNode(
            block_id=_new_block_id(), class_name="BlurGauss", params={"sigma": -5},
    )
    manifest = pc.compute_scope("s", _linear_root_state([bad]), [], None, None, None)
    assert manifest["error"] is not None
    assert manifest["version"] == pc.MANIFEST_VERSION
