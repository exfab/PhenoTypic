"""Nested previews: faithful threaded input + scope coexistence + route serves."""
import numpy as np
import pandas as pd
import pytest
import tifffile

from phenotypic._gui.builder import _preview_cache as pc
from phenotypic.sdk_ import load_image_from_store
from phenotypic._gui.builder._app import create_app
from phenotypic._gui.builder._preview_zarr_routes import preview_zarr_url
from phenotypic._gui.results_viewer._zarr_routes import store_generation_token
from phenotypic.sdk_.ngff_ import STORE_ROOT_JSON
from phenotypic._gui.builder._state import (
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

    # the inner detector's node store serves through the byte blueprint
    client = app.server.test_client()
    shash = pc.scope_hash(scope_path)
    store = (
        pc._scope_path_by_hash(sid, shash)
        / manifest["nodes"][inner_op.block_id]["store"]
    )
    base = preview_zarr_url(
        "/", sid, shash, inner_op.block_id, store_generation_token(store)
    )
    assert client.get(f"{base}/{STORE_ROOT_JSON}").status_code == 200
    # A real level-0 chunk, not only the metadata: 2-D ``gray`` uses the
    # two-index v3 key ``c.0.0``, where 3-D ``rgb`` uses ``c.0.0.0``.
    assert client.get(f"{base}/gray/0/c.0.0").status_code == 200


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


def test_nested_subtract_blank_previews_against_the_picked_table(tmp_path, monkeypatch):
    """A nested scope reloads its input from a store; the name and root must survive.

    The lookup keys on the image's name, so the store round-trip must keep
    ``t01``; blanks must resolve beside the source image, not the scope store.
    """
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    plates = tmp_path / "plates"
    plates.mkdir()
    target = np.full((64, 64), 120, dtype=np.uint8)
    target[24:40, 24:40] = 220
    tifffile.imwrite(plates / "t01.tif", target)
    tifffile.imwrite(plates / "t00.tif", np.full((64, 64), 120, dtype=np.uint8))
    table = tmp_path / "blank_map.csv"
    pd.DataFrame({"ImageName": ["t01"], "BlankImage": ["t00"]}).to_csv(table, index=False)

    inner = _DagBuilderScope()
    subtract = BlockNode(block_id=_new_block_id(), class_name="SubtractBlank", params={})
    inner.blocks.append(subtract)
    inner.edges.append(_img_edge(inner.blocks[0].block_id, subtract.block_id))
    container = BlockNode(block_id=_new_block_id(), class_name="ImagePipeline",
                          params={}, nested=inner)
    scope = _DagBuilderScope()
    scope.blocks.append(container)
    scope.edges.append(_img_edge(scope.blocks[0].block_id, container.block_id))
    state = _DagBuilderState(root=scope)
    state.reference_metadata_path = str(table)
    sid = "nestedsess0003"
    image = str(plates / "t01.tif")

    assert pc.compute_scope(sid, state, [], image, None, None)["error"] is None
    manifest = pc.compute_scope(sid, state, [container.block_id], image, None, None)
    assert manifest["error"] is None, manifest["error"]

    store = pc.scope_dir(sid, [container.block_id]) / manifest["nodes"][subtract.block_id]["store"]
    result = load_image_from_store(store)
    assert result.name == "t01"
    detect_mat = result.detect_mat[:]
    assert float(detect_mat.max()) == pytest.approx(100 / 255, abs=1e-6)
    assert float(detect_mat.min()) == pytest.approx(0.0, abs=1e-6)
