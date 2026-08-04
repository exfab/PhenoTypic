"""Nested previews: faithful threaded input + scope coexistence + route serves."""
from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._app import create_app
from phenotypic.gui.builder._state import (
    BlockNode,
    Edge,
    _DagBuilderScope,
    _DagBuilderState,
    _new_block_id,
)


def _img_edge(src, tgt):
    return Edge(edge_id=_new_block_id(), source_block_id=src, source_port="out",
                target_block_id=tgt, target_port="in", kind="image")


def _nested_state():
    inner = _DagBuilderScope()
    inner_in = inner.blocks[0]
    inner_op = BlockNode(block_id=_new_block_id(), class_name="OtsuDetector", params={})
    inner.blocks.append(inner_op)
    inner.edges.append(_img_edge(inner_in.block_id, inner_op.block_id))
    container = BlockNode(block_id=_new_block_id(), class_name="ImagePipeline",
                          params={}, nested=inner)
    parent_blur = BlockNode(block_id=_new_block_id(), class_name="BlurGauss",
                            params={"sigma": 5})
    scope = _DagBuilderScope()
    inp = scope.blocks[0]
    scope.blocks.extend([parent_blur, container])
    scope.edges.append(_img_edge(inp.block_id, parent_blur.block_id))
    scope.edges.append(_img_edge(parent_blur.block_id, container.block_id))
    return _DagBuilderState(root=scope), container, inner_op


def test_nested_scopes_coexist_and_serve(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    # create_app runs init_preview_cache() which wipes the cache root, so build
    # the app FIRST, then compute_scope writes into the (surviving) cache.
    app = create_app(image_root=tmp_path)
    state, container, inner_op = _nested_state()
    sid = "nestedsess0001"
    scope_path = [container.block_id]

    manifest = pc.compute_scope(sid, state, scope_path, None, None, None)
    assert manifest["error"] is None
    assert inner_op.block_id in manifest["nodes"]
    # parent + inner dirs coexist
    assert pc.read_manifest(sid, []) is not None
    assert pc.read_manifest(sid, scope_path) is not None

    # the inner detector's objmap tile serves through the blueprint
    client = app.server.test_client()
    shash = pc.scope_hash(scope_path)
    resp = client.get(
        f"/preview-tiles/{sid}/{shash}/{inner_op.block_id}/detect_mat.dzi"
    )
    assert resp.status_code == 200
    assert "deepzoom" in resp.get_data(as_text=True).lower()


def test_parent_edit_invalidates_inner(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    state, container, inner_op = _nested_state()
    sid = "nestedsess0002"
    scope_path = [container.block_id]
    fp1 = pc.compute_scope(sid, state, scope_path, None, None, None)["fingerprint"]

    # edit the PARENT enhancer; inner fingerprint must change (chaining)
    for b in state.root.blocks:
        if b.class_name == "BlurGauss":
            b.params["sigma"] = 1
    fp2 = pc.compute_scope(sid, state, scope_path, None, None, None)["fingerprint"]
    assert fp1 != fp2
