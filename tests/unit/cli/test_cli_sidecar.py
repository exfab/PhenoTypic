import numpy as np

from phenotypic._cli._cli_sidecar import (
    sidecar_path,
    write_sidecar,
    load_sidecar,
    sidecar_exists,
    delete_sidecar,
)


def test_path_layout(tmp_path):
    p = sidecar_path(tmp_path, "ds1", "img42")
    assert p == tmp_path / "results" / "ds1" / "objmap" / "img42.npy"


def test_write_load_exists_delete(tmp_path):
    arr = np.arange(12, dtype=np.uint16).reshape(3, 4)
    assert not sidecar_exists(tmp_path, "ds1", "img42")
    write_sidecar(tmp_path, "ds1", "img42", arr)
    assert sidecar_exists(tmp_path, "ds1", "img42")
    np.testing.assert_array_equal(load_sidecar(tmp_path, "ds1", "img42"), arr)
    delete_sidecar(tmp_path, "ds1", "img42")
    assert not sidecar_exists(tmp_path, "ds1", "img42")


def test_write_is_atomic_no_partial_file(tmp_path):
    # the temp file must not remain after a successful write
    write_sidecar(tmp_path, "ds1", "img42", np.zeros((2, 2), np.uint16))
    objmap_dir = tmp_path / "results" / "ds1" / "objmap"
    assert [p.name for p in objmap_dir.iterdir()] == ["img42.npy"]


def test_delete_missing_is_noop(tmp_path):
    # deleting a non-existent sidecar must not raise
    delete_sidecar(tmp_path, "ds1", "nope")
