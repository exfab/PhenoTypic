"""A tree of .ome.zarr stores is valid input, and scanning it stays cheap."""

from __future__ import annotations

from pathlib import Path

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


def test_a_store_is_never_mistaken_for_a_dataset(tmp_path: Path) -> None:
    """Trap A. A store is a directory; it must not become a dataset name."""
    root = tmp_path / "corrected"
    _store(root, "p01")
    datasets = scan_directory_structure(root)
    assert f"p01{ngff_.STORE_SUFFIX}" not in datasets


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
    """`get_input_structure_summary` is the --dry-run path a user runs FIRST.

    It has its own copy of every predicate. Leaving those unpatched gives a dry
    run that reports "no valid images found" for a tree the real run processes
    -- which reads as a broken input, not as a broken summary.
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


def test_scanning_does_not_descend_into_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trap B. An rglob port yields the SAME list and only differs in cost.

    A store holds ~8 files; touching them turns a 3-image scan into ~24
    extra stats, and 10k images into 400k. Count iterdir calls instead of
    comparing output.
    """
    root = tmp_path / "corrected"
    stores = [_store(root, f"p0{i}") for i in (1, 2, 3)]

    visited: list[Path] = []
    real_iterdir = Path.iterdir

    def _counting_iterdir(self: Path):
        visited.append(self)
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _counting_iterdir)
    scan_directory_structure(root)

    for store in stores:
        assert not any(
            store == seen or store in seen.parents for seen in visited
        ), f"scanner descended into {store}"
