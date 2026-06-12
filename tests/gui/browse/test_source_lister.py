import pytest

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
