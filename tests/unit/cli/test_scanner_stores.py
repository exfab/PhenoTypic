"""A tree of .ome.zarr stores is valid input, and scanning it stays cheap."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic._cli._cli_directory_scanner import (
    collect_image_paths,
    get_input_structure_summary,
    scan_directory_structure,
)


def _store(parent: Path, stem: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    img = Image(load_synth_yeast_plate())
    return img._save_store(
        parent / f"{stem}{ngff_.STORE_SUFFIX}",
        series=("gray",),
        write_objmap=False,
        levels=ngff_.pyramid_level_count(*img.gray[:].shape[:2]),
        work_id=None,
        durable=False,
        write_image_class=False,
    )


def test_a_flat_tree_of_stores_scans_as_one_dataset(tmp_path: Path) -> None:
    root = tmp_path / "corrected"
    for stem in ("p01", "p02", "p03"):
        _store(root, stem)
    datasets = scan_directory_structure(root)
    assert list(datasets) == ["corrected"]
    assert [p.name for p in datasets["corrected"]] == [
        f"p0{i}{ngff_.STORE_SUFFIX}" for i in (1, 2, 3)
    ]


def test_nested_datasets_of_stores_scan_per_subdirectory(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _store(root / "plateA", "p01")
    _store(root / "plateB", "p02")
    datasets = scan_directory_structure(root)
    assert sorted(datasets) == ["plateA", "plateB"]


def test_a_mixed_tree_of_files_and_stores_is_the_union(tmp_path: Path) -> None:
    """Spec 7.4: no ordering or precedence rule, just the union."""
    root = tmp_path / "mixed"
    _store(root, "p01")
    Image(load_synth_yeast_plate()).rgb.imsave(filepath=root / "p02.tiff")
    names = {p.name for p in collect_image_paths(root)}
    assert names == {f"p01{ngff_.STORE_SUFFIX}", "p02.tiff"}


def test_a_single_store_path_scans_as_a_single_image(tmp_path: Path) -> None:
    store = _store(tmp_path / "corrected", "p01")
    assert scan_directory_structure(store) == {"single_image": [store]}


def test_the_dry_run_summary_agrees_with_the_real_scan(tmp_path: Path) -> None:
    """`get_input_structure_summary` keeps its own copy of every predicate.

    It is NOT the --dry-run path, despite what an earlier version of this
    docstring said: `grep -rn get_input_structure_summary src/` returns only
    the definition, and --dry-run routes through `scan_directory_structure`
    (`phenotypicCLI.py:1803`). So a divergence here cannot produce the
    user-visible "no valid images found" failure that was claimed for it.

    It is pinned anyway, because the duplicate predicates exist and the next
    reader -- or the next caller -- will assume the two agree.
    """
    root = tmp_path / "corrected"
    for stem in ("p01", "p02"):
        _store(root, stem)
    summary = get_input_structure_summary(root)
    assert summary["total_images"] == 2
    assert summary["datasets"] == {"corrected": 2}
    assert f"p01{ngff_.STORE_SUFFIX}" not in summary["datasets"]


def test_the_dry_run_summary_accepts_a_single_store(tmp_path: Path) -> None:
    """Its single-path guard is a bare suffix check that rejects a store."""
    store = _store(tmp_path / "corrected", "p01")
    assert get_input_structure_summary(store)["total_images"] == 1


def _record_directory_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record every directory `os` is asked to read, and return the list.

    Both `os.listdir` and `os.scandir` are wrapped, and both are required.
    Measured on CPython 3.12.10: `Path.iterdir` calls `os.listdir` and never
    `os.scandir`, while `Path.glob`/`Path.rglob` call `os.scandir` and never
    `os.listdir`. Wrapping only one leaves the test blind to half the
    implementations it exists to judge.
    """
    visited: list[Path] = []

    def _wrap(real: Callable) -> Callable:
        def _counting(path=".", *args, **kwargs):
            try:
                visited.append(Path(path))
            except TypeError:
                # A file descriptor or bytes path -- not something this test
                # reasons about, and not something the scanner passes.
                pass
            return real(path, *args, **kwargs)

        return _counting

    monkeypatch.setattr(os, "listdir", _wrap(os.listdir))
    monkeypatch.setattr(os, "scandir", _wrap(os.scandir))
    return visited


def _assert_no_store_was_read(visited: list[Path], stores: list[Path]) -> None:
    assert visited, "no directory read was recorded; the wrappers missed the walk"
    for store in stores:
        assert not any(
            store == seen or store in seen.parents for seen in visited
        ), f"scanner descended into {store}"


def test_scanning_does_not_descend_into_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trap B. An rglob port yields the SAME list and only differs in cost.

    A store holds ~8 files; touching them turns a 3-image scan into ~24 extra
    stats, and 10k images into 400k. The output is identical either way, so
    the only observable difference is which directories got read.

    Counting `Path.iterdir` calls -- as an earlier version did -- cannot see
    that difference: `rglob` and `glob` never call `Path.iterdir`, so a port
    to `rglob` with output-preserving parent filters recorded ZERO calls and
    passed vacuously. Measured: `rglob` over this tree yielded 3 entries with
    0 `Path.iterdir` calls. Record the `os` syscalls instead
    (`_record_directory_reads`), which both spellings reach.
    """
    root = tmp_path / "corrected"
    stores = [_store(root, f"p0{i}") for i in (1, 2, 3)]

    visited = _record_directory_reads(monkeypatch)
    scan_directory_structure(root)
    monkeypatch.undo()

    _assert_no_store_was_read(visited, stores)


def test_the_summary_does_not_descend_into_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The summary's store skip is a cost guard, and this is what it buys.

    Its own copy of the skip is likewise not a correctness guard: a store's
    top level holds no image input, so `sub_count` is 0 and it never becomes
    a dataset with or without it. Delete the skip and the dataset counts are
    unchanged -- only the directory reads move. So this is asserted as cost,
    not dressed up as correctness.
    """
    root = tmp_path / "corrected"
    stores = [_store(root, f"p0{i}") for i in (1, 2, 3)]

    visited = _record_directory_reads(monkeypatch)
    summary = get_input_structure_summary(root)
    monkeypatch.undo()

    assert summary["total_images"] == 3
    _assert_no_store_was_read(visited, stores)


def test_an_in_flight_part_store_is_not_an_input(tmp_path: Path) -> None:
    """`_is_image_input`'s dot test is what excludes a half-written store.

    `ngff_.new_part_path` builds `.{name}.{uuid}.part`, so an in-flight
    promote is a DIRECTORY whose name ends in `.part` and begins with a dot.
    Nothing else excludes it: it is not a `*.ome.zarr` name, so `_is_store_dir`
    already rejects it -- but a `.hidden.ome.zarr` directory would pass the
    suffix test and is caught only by the leading dot. Both are pinned here,
    because the docstring claims both and neither had a test.
    """
    root = tmp_path / "corrected"
    real = _store(root, "p01")
    part = ngff_.new_part_path(real)
    part.mkdir()
    (part / "zarr.json").write_text("{}", encoding="utf-8")
    hidden = root / f".sneaky{ngff_.STORE_SUFFIX}"
    hidden.mkdir()

    assert part.name.startswith(".") and part.name.endswith(".part")
    assert {p.name for p in collect_image_paths(root)} == {real.name}
    assert scan_directory_structure(root) == {"corrected": [real]}
