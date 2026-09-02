"""A store is a directory. Work-ID derivation has to survive that."""

from __future__ import annotations

import hashlib

import numpy as np
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic._cli._cli_failure_tracker import file_sha256, work_id_for_image
from tests._process_stores import write_process_store


def _store(parent: Path, stem: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return write_process_store(
        parent / f"{stem}{ngff_.STORE_SUFFIX}", Image(load_synth_yeast_plate())
    )


def test_a_store_digests_without_raising(tmp_path: Path) -> None:
    """Today: IsADirectoryError, from `with path.open("rb")`."""
    store = _store(tmp_path / "in", "p01")
    assert len(file_sha256(store)) == 64


def test_the_digest_covers_every_member_of_the_tree(tmp_path: Path) -> None:
    """Named explicitly, so a future change to the tree walk is deliberate.

    Supersedes an earlier version that pinned the digest to the root
    ``zarr.json`` alone. That contract was wrong -- see
    ``test_two_stores_with_different_pixels_get_different_digests`` -- and
    pinning it made the defect look intentional.
    """
    store = _store(tmp_path / "in", "p01")

    members = sorted(
        (p for p in store.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(store).as_posix(),
    )
    assert len(members) > 1, "a one-file store would not exercise the walk"
    before = file_sha256(store)
    member = members[-1]
    moved = member.with_name(f"renamed-{member.name}")
    member.rename(moved)
    assert file_sha256(store) != before


def test_store_digest_frames_member_boundaries_unambiguously(
    tmp_path: Path,
) -> None:
    """A path/content boundary must not be forgeable with member bytes.

    The old ``path + NUL + content`` encoding maps these two distinct trees
    to the same byte stream: ``a -> b"xb\\0y"`` versus
    ``a -> b"x", b -> b"y"``.
    """
    one = tmp_path / "one.ome.zarr"
    two = tmp_path / "two.ome.zarr"
    one.mkdir()
    two.mkdir()
    (one / "a").write_bytes(b"xb\0y")
    (two / "a").write_bytes(b"x")
    (two / "b").write_bytes(b"y")

    assert file_sha256(one) != file_sha256(two)


def test_the_digest_notices_a_changed_shard(tmp_path: Path) -> None:
    """The pixel bytes are inside a shard, not the root -- the whole point."""
    store = _store(tmp_path / "in", "p01")
    shard = next(p for p in store.rglob("c.*") if p.is_file())
    before = file_sha256(store)
    shard.write_bytes(shard.read_bytes() + b"\0")
    assert file_sha256(store) != before


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


def test_a_store_shaped_directory_digests_and_fails_in_imread(
    tmp_path: Path,
) -> None:
    """Where a ``*.ome.zarr`` directory with no root document actually fails.

    `_is_store_dir` matches by NAME and promises such a directory "fails
    later, loudly, in imread". This pins that the promise is kept, and that
    `file_sha256` is not the place it breaks: the digest is over whatever
    members exist, so an empty one yields the domain-separated empty-tree
    digest rather than raising. Two empty store-shaped directories therefore
    collide -- an honest consequence of a name-matched predicate, and harmless
    because
    `imread` refuses before any of it matters.
    """
    empty = tmp_path / f"empty{ngff_.STORE_SUFFIX}"
    empty.mkdir()
    assert len(file_sha256(empty)) == 64

    other = tmp_path / f"other{ngff_.STORE_SUFFIX}"
    other.mkdir()
    assert file_sha256(other) == file_sha256(empty)

    with pytest.raises(FileNotFoundError, match="zarr.json"):
        Image.imread(empty)


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
        process_format="tiff",
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


def test_a_store_named_input_mirrors_to_a_named_output(tmp_path: Path) -> None:
    """``--input <one store>`` must not write ``<out>/.ome.zarr``.

    The output-path half of the degenerate relative path this module's
    work-ID tests cover: ``image_path.relative_to(input_root)`` is
    ``Path(".")`` when the two are the same path, and ``Path(".").stem`` is
    ``""``. Pre-existing on the flat-file side -- ``--input <one tiff>``
    wrote ``<out>/.tiff`` -- so this is not a regression the store
    introduces, but the store case is the one this design makes routine.

    Deferred out of Task 10a, which landed in an isolated worktree in
    parallel with the task that gave ``process_only_output_path`` its
    ``fmt`` parameter; re-added here once both had merged.
    """
    from phenotypic._cli._cli_process_only import process_only_output_path

    store = _store(tmp_path / "in", "p01")
    assert (
        process_only_output_path(
            tmp_path / "out", store, store, "rgb", fmt="zarr"
        ).name
        == f"p01{ngff_.STORE_SUFFIX}"
    )

    single_file = tmp_path / "in" / "p02.tiff"
    Image(load_synth_yeast_plate()).rgb.imsave(filepath=single_file)
    assert (
        process_only_output_path(
            tmp_path / "out", single_file, single_file, "rgb", fmt="tiff"
        ).name
        == "p02.tiff"
    )


def test_two_stores_with_different_pixels_get_different_digests(
    tmp_path: Path,
) -> None:
    """The store digest must move when the pixels do.

    Regression for a real defect: ``file_sha256`` digested only the root
    ``zarr.json``, on the spec's claim that it "changes whenever any published
    content does". It does not -- it carries the schema version, series map,
    pyramid geometry, metadata sections and provenance journal, none of which
    move when pixels change. Two entirely different images hashed identically,
    which silently breaks content-change detection: the digest feeds
    ``work_id_for_image`` and the SLURM identity ledger, so an edited store
    would keep its work ID and continuation would reuse stale output.

    The flat-file path digests every pixel byte. The store path must not be
    weaker.
    """
    import numpy as np

    base = np.asarray(Image(load_synth_yeast_plate()).rgb[:])
    noisy = (
        np.random.default_rng(0)
        .integers(0, 256, size=base.shape)
        .astype(base.dtype)
    )

    def _write(name: str, arr: "np.ndarray") -> Path:
        return write_process_store(
            tmp_path / f"{name}{ngff_.STORE_SUFFIX}", Image(arr)
        )

    plain, altered = _write("plain", base), _write("altered", noisy)

    # Precondition: the roots really are identical, so this test would be
    # vacuous against the old implementation for the wrong reason otherwise.
    assert (plain / "zarr.json").read_bytes() == (
        altered / "zarr.json"
    ).read_bytes()
    assert file_sha256(plain) != file_sha256(altered)


def test_a_store_digest_is_stable_across_an_unchanged_rewrite(
    tmp_path: Path,
) -> None:
    """Same content written twice must hash the same, or nothing continues."""
    arr = np.asarray(Image(load_synth_yeast_plate()).rgb[:])

    def _write(name: str) -> Path:
        return write_process_store(
            tmp_path / f"{name}{ngff_.STORE_SUFFIX}", Image(arr)
        )

    assert file_sha256(_write("one")) == file_sha256(_write("two"))
