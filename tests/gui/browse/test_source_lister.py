import pytest

from phenotypic.gui.browse._source_item import (
    resolve_source_item,
    source_item_relative_path,
)
from phenotypic.gui.browse._source_lister import list_datasets


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_flat_source_uses_dot_key(tmp_path):
    _touch(tmp_path / "a.png")
    _touch(tmp_path / "b.jpg")
    assert list_datasets(tmp_path) == {".": ["a.png", "b.jpg"]}


def test_nested_groups_by_relative_parent(tmp_path):
    _touch(tmp_path / "plates" / "batch7" / "A1.png")
    _touch(tmp_path / "plates" / "batch7" / "A2.png")
    _touch(tmp_path / "plates" / "batch8" / "B1.tif")
    result = list_datasets(tmp_path)
    assert result == {
        "plates/batch7": ["A1.png", "A2.png"],
        "plates/batch8": ["B1.tif"],
    }


def test_non_image_and_hidden_skipped(tmp_path):
    _touch(tmp_path / "keep.png")
    _touch(tmp_path / "notes.txt")
    _touch(tmp_path / ".phenotypic" / "view" / "cached.png")
    assert list_datasets(tmp_path) == {".": ["keep.png"]}


def test_empty_dir(tmp_path):
    assert list_datasets(tmp_path) == {}


def test_symlink_escaping_source_root_is_excluded(tmp_path):
    # An image whose real target lives OUTSIDE the source root must not be
    # listed, even though the symlink itself sits under the root.
    source_root = tmp_path / "source"
    outside = tmp_path / "outside"
    _touch(source_root / "real.png")  # genuine in-root image (must survive)
    _touch(outside / "secret.png")  # lives outside the source root

    link = source_root / "escapes.png"
    try:
        link.symlink_to(outside / "secret.png")
    except (OSError, NotImplementedError):  # pragma: no cover - platform guard
        pytest.skip("symlinks not supported on this platform/filesystem")

    assert list_datasets(source_root) == {".": ["real.png"]}


def test_store_children_are_atomic_images_and_are_not_descended(tmp_path, monkeypatch):
    store = tmp_path / "nested" / "p01.ome.zarr"
    chunk = store / "rgb" / "c" / "0" / "0"
    _touch(chunk)
    _touch(tmp_path / "nested" / "ordinary.png")

    original_stat = type(chunk).stat

    def guarded_stat(path, *args, **kwargs):
        if path == chunk:
            raise AssertionError("Browse descended into an atomic store")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(type(chunk), "stat", guarded_stat)

    assert list_datasets(tmp_path) == {
        "nested": ["ordinary.png", "p01.ome.zarr"]
    }


def test_direct_store_root_is_one_root_dataset_item(tmp_path):
    store = tmp_path / "p01.ome.zarr"
    _touch(store / "zarr.json")

    assert list_datasets(store) == {".": ["p01.ome.zarr"]}
    assert resolve_source_item(store, ".", "p01.ome.zarr") == store
    assert (
        source_item_relative_path(
            "inputs/p01.ome.zarr", ".", "p01.ome.zarr"
        )
        == "inputs/p01.ome.zarr"
    )


def test_plain_zarr_directory_is_listed_and_resolved_atomically(tmp_path):
    store = tmp_path / "generic.zarr"
    _touch(store / "zarr.json")

    assert list_datasets(tmp_path) == {".": ["generic.zarr"]}
    assert resolve_source_item(tmp_path, ".", "generic.zarr") == store


def test_container_source_item_resolution_does_not_change(tmp_path):
    store = tmp_path / "batch" / "p01.ome.zarr"
    _touch(store / "zarr.json")

    assert resolve_source_item(tmp_path, "batch", "p01.ome.zarr") == store


def test_store_symlink_is_not_listed(tmp_path):
    outside = tmp_path / "outside.ome.zarr"
    _touch(outside / "zarr.json")
    source = tmp_path / "source"
    source.mkdir()
    link = source / "p01.ome.zarr"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - platform guard
        pytest.skip("symlinks not supported on this platform/filesystem")

    assert list_datasets(source) == {}
