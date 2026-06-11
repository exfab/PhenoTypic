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
