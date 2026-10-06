"""ReferenceContext: table validation, lookup grain, resolution, activation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import tifffile

from phenotypic import Image, ReferenceContext
from phenotypic._core import _reference_context as rc
from phenotypic._core._reference_context import (
    ReferenceImageError,
    ReferenceLookupError,
    ReferenceTableError,
)


@pytest.fixture(autouse=True)
def _empty_image_cache():
    rc._clear_image_cache()
    yield
    rc._clear_image_cache()


def _table(tmp_path: Path, rows: dict, name: str = "layout.csv") -> Path:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_missing_file_fails_at_construction(tmp_path):
    with pytest.raises(ReferenceTableError, match="not found"):
        ReferenceContext(tmp_path / "absent.csv")


def test_table_must_carry_image_name(tmp_path):
    path = _table(tmp_path, {"Metadata_BlankImage": ["t00"]})
    with pytest.raises(ReferenceTableError, match="Metadata_ImageName"):
        ReferenceContext(path)


def test_per_colony_rows_collapse_to_one_value(tmp_path):
    path = _table(tmp_path, {
        "Metadata_ImageName": ["t04", "t04", "t04"],
        "Grid_RowNum": [1, 2, 3],
        "Metadata_BlankImage": ["t00", "t00", "t00"],
    })
    ctx = ReferenceContext(path)
    assert ctx.lookup("t04", ["Metadata_BlankImage"]) == {"Metadata_BlankImage": "t00"}


def test_disagreeing_rows_raise_ambiguous_and_list_values(tmp_path):
    path = _table(tmp_path, {
        "Metadata_ImageName": ["t04", "t04"],
        "Metadata_BlankImage": ["t00", "t01"],
    })
    with pytest.raises(ReferenceLookupError) as info:
        ReferenceContext(path).lookup("t04", ["Metadata_BlankImage"])
    assert info.value.reason == "ambiguous"
    assert "'t00'" in str(info.value) and "'t01'" in str(info.value)


def test_missing_row_raises_unmatched(tmp_path):
    path = _table(tmp_path, {"Metadata_ImageName": ["t04"], "Metadata_BlankImage": ["t00"]})
    with pytest.raises(ReferenceLookupError) as info:
        ReferenceContext(path).lookup("t05", ["Metadata_BlankImage"])
    assert info.value.reason == "unmatched"
    assert info.value.image_name == "t05"


def test_empty_cell_raises_null(tmp_path):
    path = _table(tmp_path, {"Metadata_ImageName": ["t04"], "Metadata_BlankImage": [None]})
    with pytest.raises(ReferenceLookupError) as info:
        ReferenceContext(path).lookup("t04", ["Metadata_BlankImage"])
    assert info.value.reason == "null"


def test_unknown_column_is_a_table_error(tmp_path):
    path = _table(tmp_path, {"Metadata_ImageName": ["t04"], "Metadata_BlankImage": ["t00"]})
    ctx = ReferenceContext(path)
    assert not ctx.has_column("Metadata_FlatField")
    with pytest.raises(ReferenceTableError, match="Metadata_FlatField"):
        ctx.lookup("t04", ["Metadata_FlatField"])


# Review Focus 1
def test_digit_only_stems_keep_leading_zeros(tmp_path):
    path = tmp_path / "layout.csv"
    path.write_text("Metadata_ImageName,Metadata_BlankImage\n000123,000100\n", encoding="utf-8")
    assert ReferenceContext(path).lookup("000123", ["Metadata_BlankImage"]) == {
        "Metadata_BlankImage": "000100"
    }


# Review Focus 3
def test_bare_headers_resolve_like_prefixed_ones(tmp_path):
    path = _table(tmp_path, {"ImageName": ["t04"], "BlankImage": ["t00"]})
    ctx = ReferenceContext(path)
    assert ctx.lookup("t04", ["BlankImage"]) == {"BlankImage": "t00"}
    assert ctx.lookup("t04", ["Metadata_BlankImage"]) == {"Metadata_BlankImage": "t00"}


def test_dataset_narrows_lookup_when_table_has_dataset(tmp_path):
    path = _table(tmp_path, {
        "Metadata_Dataset": ["A", "B"],
        "Metadata_ImageName": ["t04", "t04"],
        "Metadata_BlankImage": ["a0", "b0"],
    })
    ctx = ReferenceContext(path)
    assert ctx.narrow(dataset="B").lookup("t04", ["Metadata_BlankImage"]) == {
        "Metadata_BlankImage": "b0"
    }


# Review Focus 4
def test_same_stem_two_datasets_without_dataset_column_is_ambiguous(tmp_path):
    path = _table(tmp_path, {
        "Metadata_ImageName": ["t04", "t04"],
        "Metadata_BlankImage": ["a0", "b0"],
    })
    with pytest.raises(ReferenceLookupError) as info:
        ReferenceContext(path).narrow(dataset="A").lookup("t04", ["Metadata_BlankImage"])
    assert info.value.reason == "ambiguous"


def test_accepts_pandas_and_polars_frames():
    rows = {"Metadata_ImageName": ["t04"], "Metadata_BlankImage": ["t00"]}
    for frame in (pd.DataFrame(rows), pl.DataFrame(rows)):
        ctx = ReferenceContext(frame)
        assert ctx.table_sha256 is None
        assert ctx.lookup("t04", ["Metadata_BlankImage"]) == {"Metadata_BlankImage": "t00"}


def test_table_sha256_is_file_digest(tmp_path):
    import hashlib

    path = _table(tmp_path, {"Metadata_ImageName": ["t04"], "Metadata_BlankImage": ["t00"]})
    assert ReferenceContext(path).table_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


# Review Focus 2
def test_resolve_by_stem_and_by_filename(tmp_path):
    root = tmp_path / "imgs"
    root.mkdir()
    (root / "t00.tif").write_bytes(b"x")
    path = _table(tmp_path, {"Metadata_ImageName": ["t04"]})
    ctx = ReferenceContext(path, image_root=root)
    assert ctx.resolve_image("t00") == root / "t00.tif"
    assert ctx.resolve_image("t00.tif") == root / "t00.tif"


def test_resolve_refuses_two_files_with_one_stem(tmp_path):
    root = tmp_path / "imgs"
    root.mkdir()
    (root / "t00.tif").write_bytes(b"x")
    (root / "t00.png").write_bytes(b"x")
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}), image_root=root)
    with pytest.raises(ReferenceImageError, match="2 files"):
        ctx.resolve_image("t00")


def test_resolve_without_root_or_mapping_raises(tmp_path):
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}))
    with pytest.raises(ReferenceImageError, match="image_root"):
        ctx.resolve_image("t00")


def test_images_mapping_takes_precedence(tmp_path):
    blank = Image(arr=np.zeros((4, 4), dtype=np.float32), name="t00")
    ctx = ReferenceContext(
        _table(tmp_path, {"Metadata_ImageName": ["t04"]}),
        image_root=tmp_path,
        images={"t00": blank},
    )
    assert ctx.resolve_image("t00") is blank
    assert ctx.load_image("t00") is blank
    assert ctx.reference_image_digest("t00") is None


def test_load_image_reads_each_file_once_and_rereads_on_change(tmp_path, monkeypatch):
    root = tmp_path / "imgs"
    root.mkdir()
    blank_path = root / "t00.tif"
    tifffile.imwrite(blank_path, np.full((8, 8), 40, dtype=np.uint8))
    calls: list[Path] = []
    real = rc._read_image

    def counting(path, read_kwargs):
        calls.append(path)
        return real(path, read_kwargs)

    monkeypatch.setattr(rc, "_read_image", counting)
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}), image_root=root)
    first = ctx.load_image("t00")
    assert ctx.load_image("t00") is first
    assert len(calls) == 1
    tifffile.imwrite(blank_path, np.full((8, 8), 41, dtype=np.uint8))
    stat = blank_path.stat()
    os.utime(blank_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    ctx.load_image("t00")
    assert len(calls) == 2
    assert ctx.reference_image_digest("t00") is not None


def test_current_is_none_outside_any_context():
    assert ReferenceContext.current() is None


def test_nesting_replaces_then_restores(tmp_path):
    outer = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}))
    inner = outer.narrow(dataset="B")
    with outer:
        assert ReferenceContext.current() is outer
        with inner:
            assert ReferenceContext.current() is inner
            assert inner.table is outer.table
        assert ReferenceContext.current() is outer
    assert ReferenceContext.current() is None


def test_context_restored_after_exception(tmp_path):
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}))
    with pytest.raises(RuntimeError):
        with ctx:
            raise RuntimeError("boom")
    assert ReferenceContext.current() is None


def test_resolve_ome_zarr_store_by_its_source_stem(tmp_path):
    root = tmp_path / "imgs"
    (root / "t00.ome.zarr").mkdir(parents=True)
    (root / "t00.ome.zarr" / "zarr.json").write_text("{}", encoding="utf-8")
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}), image_root=root)
    assert ctx.resolve_image("t00") == root / "t00.ome.zarr"
    assert rc.reference_file_digest(root / "t00.ome.zarr") is not None
