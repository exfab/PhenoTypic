"""ReferenceContext: table validation, lookup grain, resolution, activation."""

from __future__ import annotations

import hashlib
import os
import threading
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
    assert ctx.reference_image_digest("t00") == hashlib.sha256(blank_path.read_bytes()).hexdigest()


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


# ------------------------------------------------------------ table reading
def test_parquet_tables_are_read_as_strings(tmp_path):
    path = tmp_path / "layout.parquet"
    pl.DataFrame({"Metadata_ImageName": ["t04"], "Metadata_BlankImage": ["t00"]}).write_parquet(path)
    ctx = ReferenceContext(path)
    assert ctx.lookup("t04", ["Metadata_BlankImage"]) == {"Metadata_BlankImage": "t00"}
    assert ctx.table_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_unsupported_suffix_is_a_table_error(tmp_path):
    path = tmp_path / "layout.xlsx"
    path.write_bytes(b"not a table")
    with pytest.raises(ReferenceTableError, match=r"\.csv or \.parquet"):
        ReferenceContext(path)


@pytest.mark.parametrize("value", ["", " ", " \t "])
def test_blank_string_values_are_null(value):
    frame = pd.DataFrame({"Metadata_ImageName": ["t04"], "Metadata_BlankImage": [value]})
    with pytest.raises(ReferenceLookupError) as info:
        ReferenceContext(frame).lookup("t04", ["Metadata_BlankImage"])
    assert info.value.reason == "null"


def test_whitespace_only_csv_cell_is_null(tmp_path):
    path = tmp_path / "layout.csv"
    path.write_text("Metadata_ImageName,Metadata_BlankImage\nt04, \n", encoding="utf-8")
    with pytest.raises(ReferenceLookupError) as info:
        ReferenceContext(path).lookup("t04", ["Metadata_BlankImage"])
    assert info.value.reason == "null"


def test_values_are_stripped_at_load(tmp_path):
    path = tmp_path / "layout.csv"
    path.write_text("Metadata_ImageName,Metadata_BlankImage\n t04 , t00 \n", encoding="utf-8")
    assert ReferenceContext(path).lookup("t04", ["Metadata_BlankImage"]) == {
        "Metadata_BlankImage": "t00"
    }


def test_two_spellings_of_one_column_are_both_served(tmp_path):
    path = _table(tmp_path, {"Metadata_ImageName": ["t04"], "Metadata_BlankImage": ["t00"]})
    assert ReferenceContext(path).lookup("t04", ["BlankImage", "Metadata_BlankImage"]) == {
        "BlankImage": "t00",
        "Metadata_BlankImage": "t00",
    }


def test_key_columns_are_served_from_the_key(tmp_path):
    path = _table(tmp_path, {
        "Metadata_Dataset": ["A", "B"],
        "Metadata_ImageName": ["t04", "t04"],
        "Metadata_BlankImage": ["a0", "b0"],
    })
    ctx = ReferenceContext(path)
    assert ctx.lookup("t04", ["Metadata_ImageName"]) == {"Metadata_ImageName": "t04"}
    assert ctx.narrow(dataset="B").lookup(
        "t04", ["Metadata_Dataset", "ImageName", "Metadata_BlankImage"]
    ) == {"Metadata_Dataset": "B", "ImageName": "t04", "Metadata_BlankImage": "b0"}


# ---------------------------------------------------------- image resolution
def _counting_directory_reads(monkeypatch, root: Path) -> list:
    """Count every scandir/listdir of *root* (Path.iterdir uses listdir)."""
    reads: list = []
    real_scandir, real_listdir = os.scandir, os.listdir

    def is_root(path) -> bool:
        return isinstance(path, (str, os.PathLike)) and Path(path) == root

    def scandir(path=".", *args, **kwargs):
        if is_root(path):
            reads.append(("scandir", path))
        return real_scandir(path, *args, **kwargs)

    def listdir(path=".", *args, **kwargs):
        if is_root(path):
            reads.append(("listdir", path))
        return real_listdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(os, "listdir", listdir)
    return reads


def test_resolving_many_names_reads_the_root_once(tmp_path, monkeypatch):
    """The planner resolves once per input image; a rescan per name is O(N^2)."""
    root = tmp_path / "imgs"
    root.mkdir()
    names = [f"t{i:02d}" for i in range(20)]
    for name in names:
        (root / f"{name}.tif").write_bytes(b"x")
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}), image_root=root)
    reads = _counting_directory_reads(monkeypatch, root)
    for name in names:
        assert ctx.resolve_image(name) == root / f"{name}.tif"
        assert ctx.resolve_image(f"{name}.tif") == root / f"{name}.tif"
    assert ctx.narrow(dataset="B").resolve_image("t03") == root / "t03.tif"
    assert len(reads) == 1


def test_root_index_sees_a_file_added_later(tmp_path):
    root = tmp_path / "imgs"
    root.mkdir()
    (root / "t00.tif").write_bytes(b"x")
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}), image_root=root)
    assert ctx.resolve_image("t00") == root / "t00.tif"
    (root / "t01.tif").write_bytes(b"x")
    stat = root.stat()
    os.utime(root, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    assert ctx.resolve_image("t01") == root / "t01.tif"


def test_sidecar_files_do_not_count_as_candidates(tmp_path):
    root = tmp_path / "imgs"
    root.mkdir()
    for name in ("t00.tif", "t00.json", "t00.xmp", "t00.txt"):
        (root / name).write_bytes(b"x")
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}), image_root=root)
    assert ctx.resolve_image("t00") == root / "t00.tif"


def test_uppercase_image_suffixes_are_candidates(tmp_path):
    root = tmp_path / "imgs"
    root.mkdir()
    (root / "t00.TIF").write_bytes(b"x")
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}), image_root=root)
    assert ctx.resolve_image("t00") == root / "t00.TIF"


@pytest.mark.parametrize("kind", ["parent", "absolute", "subdirectory"])
def test_names_outside_the_root_are_refused(tmp_path, kind):
    root = tmp_path / "imgs"
    (root / "sub").mkdir(parents=True)
    (tmp_path / "x.tif").write_bytes(b"x")
    (root / "sub" / "x.tif").write_bytes(b"x")
    name = {
        "parent": "../x.tif",
        "absolute": str(tmp_path / "x.tif"),
        "subdirectory": "sub/x.tif",
    }[kind]
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}), image_root=root)
    with pytest.raises(ReferenceImageError, match="image_root"):
        ctx.resolve_image(name)


# ------------------------------------------------------------- image cache
def test_read_kwargs_are_part_of_the_cache_key(tmp_path, monkeypatch):
    root = tmp_path / "imgs"
    root.mkdir()
    tifffile.imwrite(root / "t00.tif", np.full((8, 8), 40, dtype=np.uint8))
    calls: list = []
    real = rc._read_image

    def counting(path, read_kwargs):
        calls.append(dict(read_kwargs))
        return real(path, read_kwargs)

    monkeypatch.setattr(rc, "_read_image", counting)
    table = _table(tmp_path, {"Metadata_ImageName": ["t04"]})
    plain = ReferenceContext(table, image_root=root)
    eight_bit = ReferenceContext(table, image_root=root, read_kwargs={"bit_depth": 8})
    plain.load_image("t00")
    eight_bit.load_image("t00")
    plain.load_image("t00")
    assert calls == [{}, {"bit_depth": 8}]


def test_cache_holds_at_most_its_bound_and_evicts_least_recent(tmp_path, monkeypatch):
    root = tmp_path / "imgs"
    root.mkdir()
    names = [f"t{i:02d}" for i in range(rc._IMAGE_CACHE_SIZE + 1)]
    for i, name in enumerate(names):
        tifffile.imwrite(root / f"{name}.tif", np.full((4, 4), i, dtype=np.uint8))
    calls: list = []
    real = rc._read_image

    def counting(path, read_kwargs):
        calls.append(Path(path).name)
        return real(path, read_kwargs)

    monkeypatch.setattr(rc, "_read_image", counting)
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}), image_root=root)
    for name in names:
        ctx.load_image(name)
    assert len(rc._IMAGE_CACHE) == rc._IMAGE_CACHE_SIZE
    ctx.load_image(names[-1])          # still cached
    ctx.load_image(names[0])           # evicted first, so read again
    assert calls == [f"{n}.tif" for n in names] + [f"{names[0]}.tif"]


def test_store_cache_key_follows_its_root_zarr_json(tmp_path, monkeypatch):
    """Rewriting a store leaves its directory's mtime/size alone; the root
    zarr.json is what changes, so it must be in the key."""
    root = tmp_path / "imgs"
    store = root / "t00.ome.zarr"
    store.mkdir(parents=True)
    zarr_json = store / "zarr.json"
    zarr_json.write_text("{}", encoding="utf-8")
    calls: list = []

    def fake_read(path, read_kwargs):
        calls.append(path)
        return Image(arr=np.zeros((4, 4), dtype=np.float32), name="t00")

    monkeypatch.setattr(rc, "_read_image", fake_read)
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}), image_root=root)
    ctx.load_image("t00")
    store_stat = store.stat()
    json_stat = zarr_json.stat()
    zarr_json.write_text('{"rewritten": true}', encoding="utf-8")
    os.utime(zarr_json, ns=(json_stat.st_atime_ns, json_stat.st_mtime_ns + 10_000_000))
    os.utime(store, ns=(store_stat.st_atime_ns, store_stat.st_mtime_ns))
    ctx.load_image("t00")
    assert len(calls) == 2
    assert ctx.reference_image_digest("t00") == hashlib.sha256(zarr_json.read_bytes()).hexdigest()


# ----------------------------------------------------------- planned digests
def _planned(tmp_path: Path) -> tuple[ReferenceContext, ReferenceContext, Path]:
    root = tmp_path / "imgs"
    root.mkdir()
    blank = root / "t00.tif"
    tifffile.imwrite(blank, np.full((8, 8), 40, dtype=np.uint8))
    base = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}), image_root=root)
    planned = base.narrow(
        planned_digests={str(blank.resolve()): rc.reference_file_digest(blank)}
    )
    return base, planned, blank


def test_a_planned_file_with_its_planned_bytes_loads(tmp_path):
    _, planned, _ = _planned(tmp_path)
    assert planned.load_image("t00").gray.shape == (8, 8)
    # A narrowed clone of a planned context keeps the plan.
    assert planned.narrow(dataset="d").planned_digests == planned.planned_digests


def test_a_file_rewritten_since_planning_is_refused_and_never_served_from_cache(tmp_path):
    """The digest is cached with the pixels of one load, under a key the
    rewrite changes, so the check always judges the pixels it returns."""
    base, planned, blank = _planned(tmp_path)
    planned.load_image("t00")  # cached with the planned digest
    # A different shape changes the size, so the key moves even inside one
    # mtime tick.
    tifffile.imwrite(blank, np.full((9, 9), 40, dtype=np.uint8))
    with pytest.raises(rc.ReferenceImageChangedError, match="t00"):
        planned.load_image("t00")
    # Not a ReferenceContextError: the image is not at fault (the CLI makes it
    # its non-terminal ReferencePlanStaleError).
    assert not issubclass(rc.ReferenceImageChangedError, rc.ReferenceContextError)
    # A context without a plan reads the new bytes, not the cached old ones.
    assert base.load_image("t00").gray.shape == (9, 9)


def test_a_file_absent_from_the_plan_is_refused(tmp_path):
    base, _, _ = _planned(tmp_path)
    with pytest.raises(rc.ReferenceImageChangedError):
        base.narrow(planned_digests={}).load_image("t00")


def test_in_memory_images_are_never_checked(tmp_path):
    image = Image(arr=np.zeros((4, 4), dtype=np.float32), name="mem")
    ctx = ReferenceContext(
        _table(tmp_path, {"Metadata_ImageName": ["t04"]}), images={"mem": image}
    ).narrow(planned_digests={})
    assert ctx.load_image("mem") is image


# --------------------------------------------------------------- activation
def test_one_instance_entered_from_two_threads_restores_cleanly(tmp_path):
    """Each thread's exit must reset its own activation. The order is forced:
    a enters, then b; a leaves while b is still inside, so a shared stack
    would hand a the token b pushed."""
    ctx = ReferenceContext(_table(tmp_path, {"Metadata_ImageName": ["t04"]}))
    a_inside, b_inside, a_left = threading.Event(), threading.Event(), threading.Event()
    errors: list[BaseException] = []
    seen: list[tuple[str, bool]] = []

    def thread_a() -> None:
        try:
            with ctx:
                a_inside.set()
                b_inside.wait(timeout=10)
                seen.append(("a", ReferenceContext.current() is ctx))
            seen.append(("a", ReferenceContext.current() is None))
        except BaseException as exc:  # noqa: BLE001 -- surfaced by the assert
            errors.append(exc)
        finally:
            a_inside.set()
            a_left.set()

    def thread_b() -> None:
        try:
            a_inside.wait(timeout=10)
            with ctx:
                b_inside.set()
                a_left.wait(timeout=10)
                seen.append(("b", ReferenceContext.current() is ctx))
            seen.append(("b", ReferenceContext.current() is None))
        except BaseException as exc:  # noqa: BLE001 -- surfaced by the assert
            errors.append(exc)
        finally:
            b_inside.set()

    threads = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert errors == []
    assert sorted(seen) == [("a", True), ("a", True), ("b", True), ("b", True)]
