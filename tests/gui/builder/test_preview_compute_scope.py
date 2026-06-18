"""compute_scope: full-res cache, threaded nested input, chained staleness."""
import h5py
from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._state import (
    BlockNode, Edge, _DagBuilderState, _DagBuilderScope, _new_block_id,
)


def _image_edge(src, tgt):
    return Edge(edge_id=_new_block_id(), source_block_id=src,
                source_port="out", target_block_id=tgt, target_port="in",
                kind="image")


def _linear_root_state(op_blocks):
    """Build a root scope: InputImage -> op_blocks[0] -> ... chained."""
    scope = _DagBuilderScope()  # __post_init__ seeds InputImage at index 0
    input_block = scope.blocks[0]
    scope.blocks.extend(op_blocks)
    prev = input_block.block_id
    for b in op_blocks:
        scope.edges.append(_image_edge(prev, b.block_id))
        prev = b.block_id
    return _DagBuilderState(root=scope)


def test_root_scope_caches_all_nodes(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    blur = BlockNode(block_id=_new_block_id(), class_name="GaussianBlur",
                     params={"sigma": 1})
    state = _linear_root_state([blur])

    manifest = pc.compute_scope("sess1", state, [], image_path=None,
                                nrows=None, ncols=None)

    assert manifest["error"] is None
    # input node + the blur op both have entries
    assert blur.block_id in manifest["nodes"]
    blur_hdf = pc.scope_dir("sess1", [])/ manifest["nodes"][blur.block_id]["hdf"]
    assert blur_hdf.exists()
    with h5py.File(blur_hdf, "r") as f:
        assert "layers" in f


def test_fingerprint_stable_then_invalidates_on_edit(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    blur = BlockNode(block_id=_new_block_id(), class_name="GaussianBlur",
                     params={"sigma": 1})
    state = _linear_root_state([blur])

    fp1 = pc.compute_scope("s", state, [], None, None, None)["fingerprint"]
    fp2 = pc.compute_scope("s", state, [], None, None, None)["fingerprint"]
    assert fp1 == fp2  # no change -> same fingerprint

    blur.params["sigma"] = 5  # edit a param
    state2 = _linear_root_state([blur])
    fp3 = pc.compute_scope("s", state2, [], None, None, None)["fingerprint"]
    assert fp3 != fp1


def test_nested_scope_threads_parent_output(tmp_path, monkeypatch):
    """An inner node sees the parent enhancer's detect_mat, not the raw sample."""
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")

    # Parent scope: InputImage -> GaussianBlur(parent) -> sub-pipeline container
    inner_scope = _DagBuilderScope()
    inner_input = inner_scope.blocks[0]
    inner_op = BlockNode(block_id=_new_block_id(), class_name="GaussianBlur",
                         params={"sigma": 1})
    inner_scope.blocks.append(inner_op)
    inner_scope.edges.append(_image_edge(inner_input.block_id, inner_op.block_id))

    container = BlockNode(block_id=_new_block_id(), class_name="ImagePipeline",
                          params={}, nested=inner_scope)
    parent_blur = BlockNode(block_id=_new_block_id(), class_name="GaussianBlur",
                            params={"sigma": 7})
    state = _linear_root_state([parent_blur, container])

    scope_path = [container.block_id]
    manifest = pc.compute_scope("s", state, scope_path, None, None, None)
    assert manifest["error"] is None
    assert inner_op.block_id in manifest["nodes"]

    # parent + inner scope dirs both exist (no clobbering)
    parent_manifest = pc.read_manifest("s", [])
    assert parent_manifest is not None
    assert pc.read_manifest("s", scope_path) is not None
    # chained fingerprint: inner fp folds in parent fp
    parent_fp = parent_manifest["fingerprint"]
    assert parent_fp in manifest["fingerprint_inputs"]

    # Threaded input (not just chained fingerprint): the inner scope's
    # base snapshot IS the parent enhancer's output, not the raw sample.
    import numpy as np

    parent_dir = pc.scope_dir("s", [])
    parent_blur_hdf = (
        parent_dir / parent_manifest["nodes"][parent_blur.block_id]["hdf"]
    )
    inner_base = pc.scope_dir("s", scope_path) / "base_00.h5"

    def _detect_mat(path):
        with h5py.File(path, "r") as f:
            grp = f["layers"] if "layers" in f else f
            return grp["detect_mat"][()]

    from phenotypic.data._synthetic_data import load_synth_yeast_plate

    raw_dm = load_synth_yeast_plate().detect_mat[:]
    assert np.array_equal(_detect_mat(inner_base), _detect_mat(parent_blur_hdf))
    assert not np.array_equal(_detect_mat(inner_base), raw_dm)
