"""A store is a directory. Work-ID derivation has to survive that."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic._cli._cli_failure_tracker import file_sha256


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


def test_a_store_digests_without_raising(tmp_path: Path) -> None:
    """Today: IsADirectoryError, from `with path.open("rb")`."""
    store = _store(tmp_path / "in", "p01")
    assert len(file_sha256(store)) == 64


def test_the_digest_is_the_root_zarr_json(tmp_path: Path) -> None:
    """Named explicitly, so a future change to the tree walk is deliberate."""
    store = _store(tmp_path / "in", "p01")
    expected = hashlib.sha256((store / "zarr.json").read_bytes()).hexdigest()
    assert file_sha256(store) == expected


def test_the_digest_changes_when_the_store_content_does(tmp_path: Path) -> None:
    """The root records the series map, pyramid, metadata, and provenance."""
    a = _store(tmp_path / "a", "p01")
    root = a / "zarr.json"
    before = file_sha256(a)
    root.write_text(
        root.read_text(encoding="utf-8").replace('"gray"', '"grey"'),
        encoding="utf-8",
    )
    assert file_sha256(a) != before


def test_a_flat_file_digest_is_untouched(tmp_path: Path) -> None:
    """The whole-file streaming read is the path 99% of inputs still take."""
    target = tmp_path / "p01.tiff"
    target.write_bytes(b"not really a tiff, but it is a file")
    assert file_sha256(target) == hashlib.sha256(target.read_bytes()).hexdigest()


def test_a_plain_directory_is_still_refused(tmp_path: Path) -> None:
    """A directory that is not a store has no fingerprint. Say so."""
    plain = tmp_path / "just_a_folder"
    plain.mkdir()
    with pytest.raises(IsADirectoryError):
        file_sha256(plain)
