"""Disk cache primitives: scope dirs, manifest round-trip, lifecycle."""
from phenotypic.gui.builder import _preview_cache as pc


def test_scope_hash_root_vs_nested():
    assert pc.scope_hash([]) == pc.scope_hash([])
    assert pc.scope_hash(["a" * 32]) != pc.scope_hash([])


def test_scope_dir_isolated_per_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    root_dir = pc.scope_dir("sess1", [])
    nested_dir = pc.scope_dir("sess1", ["c" * 32])
    assert root_dir.is_dir()
    assert nested_dir.is_dir()
    assert root_dir != nested_dir


def test_manifest_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    manifest = {
        "fingerprint": "abc",
        "scope_key": "",
        "nodes": {"blk": {"hdf": "base_00.h5", "layers": ["rgb"],
                          "shape": [10, 10], "num_objects": 0}},
        "error": None,
    }
    pc.write_manifest("sess1", [], manifest)
    assert pc.read_manifest("sess1", []) == manifest
    assert pc.read_manifest("sess1", ["zzz"]) is None  # missing scope


def test_wipe_scope_removes_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "root")
    d = pc.scope_dir("sess1", [])
    (d / "base_00.h5").write_bytes(b"x")
    pc.wipe_scope("sess1", [])
    assert not d.exists()
