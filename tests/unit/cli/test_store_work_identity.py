"""A store is a directory. Work-ID derivation has to survive that."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic._cli._cli_failure_tracker import file_sha256, work_id_for_image


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


def _config(input_path: Path, pipeline: Path) -> SimpleNamespace:
    """The ExecutionConfig surface `work_id_for_image` actually reads."""
    return SimpleNamespace(
        input_path=input_path,
        pipeline_json=pipeline,
        image_type="Image",
        nrows=None,
        ncols=None,
        bit_depth=None,
        detect_mode="gray",
        process_only_layer="rgb",
        ext=".tiff",
        include_dataset_column=True,
        overlay_alpha=0.3,
        save_overlays=True,
        drop_originals=False,
        measure_only=False,
    )


def test_a_store_named_as_input_keeps_its_name_as_the_relative_path(
    tmp_path: Path,
) -> None:
    """`--input <one store>` must not derive a relative path of ".".

    A store is a directory, so it never takes the `is_file` branch; it falls
    through to `relative_to`, which yields `Path(".")` when the two paths are
    the same.
    """
    store = _store(tmp_path / "in", "p01")
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text("{}", encoding="utf-8")

    _, relative_path = work_id_for_image(
        _config(store, pipeline), "single_image", store
    )
    assert relative_path == f"p01{ngff_.STORE_SUFFIX}"


def test_two_stores_named_as_input_do_not_share_one_work_id(
    tmp_path: Path,
) -> None:
    """The half that matters: "." collapses every store onto one identity.

    The two stores are byte-identical copies, so `input_sha256` cannot tell
    them apart and the relative path is the only discriminator left. With the
    degenerate `Path(".")`, both work IDs are the same string -- a real
    collision, not a theoretical one.
    """
    a = _store(tmp_path / "in", "p01")
    b = tmp_path / "in2" / "p02.ome.zarr"
    b.parent.mkdir(parents=True)
    shutil.copytree(a, b)
    assert file_sha256(a) == file_sha256(b), "the copies must be indistinguishable"

    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text("{}", encoding="utf-8")

    work_id_a, rel_a = work_id_for_image(_config(a, pipeline), "single_image", a)
    work_id_b, rel_b = work_id_for_image(_config(b, pipeline), "single_image", b)

    assert rel_a != rel_b
    assert work_id_a != work_id_b


def test_a_store_under_a_parent_input_is_unaffected(tmp_path: Path) -> None:
    """The ordinary path: `--input` is the tree, so `relative_to` is real."""
    root = tmp_path / "corrected"
    store = _store(root, "p01")
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text("{}", encoding="utf-8")

    _, relative_path = work_id_for_image(
        _config(root, pipeline), "corrected", store
    )
    assert relative_path == f"p01{ngff_.STORE_SUFFIX}"
