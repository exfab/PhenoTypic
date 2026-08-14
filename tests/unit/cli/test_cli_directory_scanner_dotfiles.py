"""AppleDouble sidecars must never be counted as input images.

macOS writes a ``._<name>`` sidecar beside every file on exFAT/FAT volumes —
the normal way an external drive is formatted. ``Path("._x.tif").suffix`` is
``".tif"``, so a filter that tests the extension alone admits every sidecar and
each image is counted twice.

Observed on a real run: ``.phenotypic/progress/manifest.json`` reported
``total_images: 60`` for 30 images, with ``is_complete: false`` on a run that
had finished. Anything gating on completion — the GUI runs registry, the SLURM
observer — reads such a run as unfinished forever.
"""

from pathlib import Path

import polars as pl

from phenotypic._cli._cli_chunk_writer import _scan_unchunked_parquets
from phenotypic._cli._cli_directory_scanner import (
    get_input_structure_summary,
    scan_directory_structure,
)
from phenotypic.sdk_ import dataset_measurements_dir

APPLEDOUBLE = b"\x00\x05\x16\x07\x00\x02\x00\x00"


def _make_tree(root: Path) -> None:
    """Two real images per dataset, each shadowed by an AppleDouble sidecar."""
    for ds in ("plate_a", "plate_b"):
        d = root / ds
        d.mkdir(parents=True)
        for stem in ("img_001", "img_002"):
            (d / f"{stem}.tif").write_bytes(b"real")
            (d / f"._{stem}.tif").write_bytes(APPLEDOUBLE)


def test_scan_excludes_appledouble_sidecars(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    datasets = scan_directory_structure(tmp_path)
    assert datasets, "no datasets discovered"
    for name, images in datasets.items():
        names = [p.name for p in images]
        assert not any(n.startswith("._") for n in names), (
            f"dataset {name} admitted AppleDouble sidecars: {names}"
        )
        assert len(images) == 2, f"dataset {name} has {len(images)}, want 2"


def test_count_excludes_appledouble_sidecars(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    summary = get_input_structure_summary(tmp_path)
    assert summary["total_images"] == 4, (
        f"summary counted {summary}, want 2 per dataset"
    )
    assert summary["datasets"] == {"plate_a": 2, "plate_b": 2}, summary


def test_root_level_images_exclude_sidecars(tmp_path: Path) -> None:
    """The flat-directory case takes a different code path than subdirs."""
    (tmp_path / "img_001.tif").write_bytes(b"real")
    (tmp_path / "._img_001.tif").write_bytes(APPLEDOUBLE)
    datasets = scan_directory_structure(tmp_path)
    all_names = [p.name for images in datasets.values() for p in images]
    assert all_names == ["img_001.tif"], all_names


def test_dotfiles_generally_excluded(tmp_path: Path) -> None:
    """Any dotfile with an image extension is not an input, not just ._ ones."""
    d = tmp_path / "plate_a"
    d.mkdir(parents=True)
    (d / "img_001.tif").write_bytes(b"real")
    (d / ".hidden.tif").write_bytes(b"nope")
    datasets = scan_directory_structure(tmp_path)
    images = next(iter(datasets.values()))
    assert [p.name for p in images] == ["img_001.tif"]


def test_chunk_scan_excludes_appledouble_parquets(tmp_path: Path) -> None:
    """The chunk writer has the same filter, and the same gap.

    ``_scan_unchunked_parquets`` skips ``_``-prefixed names (to avoid its own
    ``_dataset_aggregated.parquet``) but not ``.``-prefixed ones, so on an
    exFAT volume it would hand a binary AppleDouble file to the parquet reader.
    """
    meas = dataset_measurements_dir(tmp_path, "plate_a")
    meas.mkdir(parents=True)
    pl.DataFrame({"a": [1]}).write_parquet(meas / "img_001.parquet")
    (meas / "._img_001.parquet").write_bytes(APPLEDOUBLE)

    found = _scan_unchunked_parquets(tmp_path / "results", set())
    names = [p.name for p in found]
    assert names == ["img_001.parquet"], names
