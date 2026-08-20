"""Materializing a subset as two directory layouts — §10.3.1.

Neither engine accepts a file list. ``tune``'s ``-i`` is a **non-recursive**
``iterdir``; the forward CLI's ``-i`` is a single path whose one level of
subdirectories becomes ``Metadata_Dataset``. So a subset has to be
materialized, and a single layout cannot serve both: a nested directory
matches zero images for ``tune``, and a flat one silently relabels every row's
dataset to the staging folder name for deploy.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest


@pytest.fixture
def parent(tmp_path):
    """A parent with ``plateA/`` (3 images) and ``plateB/`` (2)."""
    root = tmp_path / "data" / "plates"
    for dataset, count in (("plateA", 3), ("plateB", 2)):
        (root / dataset).mkdir(parents=True)
        for index in range(1, count + 1):
            (root / dataset / f"{dataset}_{index:02d}.tif").write_bytes(
                f"{dataset}{index}".encode()
            )
    return root


@pytest.fixture
def subset(parent):
    """Three of the parent's five images: two from ``plateA``, one from ``plateB``."""
    from phenotypic._services.staging import SubsetToStage
    from phenotypic.subset import ImageRef

    chosen = [
        "plateA/plateA_01.tif",
        "plateA/plateA_03.tif",
        "plateB/plateB_02.tif",
    ]
    return SubsetToStage(
        parent=parent,
        images=tuple(
            ImageRef(path=parent / relative, relative_path=relative)
            for relative in chosen
        ),
    )


@pytest.fixture
def staged(subset, tmp_path):
    from phenotypic._services.staging import stage_subset

    return stage_subset(subset, cache_root=tmp_path / "workspace")


# --------------------------------------------------------------------------
# The two layouts
# --------------------------------------------------------------------------


def test_flat_layout_is_a_non_recursive_iterdir_match(staged):
    """What tune's ``_load_images`` actually does."""
    files = [p for p in staged.flat.iterdir() if p.is_file()]
    assert len(files) == 3


def test_flat_layout_has_no_subdirectories(staged):
    """A nested tree here matches **zero** images and the run dies on SystemExit."""
    assert not [p for p in staged.flat.iterdir() if p.is_dir()]


def test_nested_layout_preserves_dataset_names(staged):
    from phenotypic._cli._cli_directory_scanner import scan_directory_structure

    datasets = scan_directory_structure(staged.nested)
    assert set(datasets) == {"plateA", "plateB"}


def test_nested_layout_keeps_each_image_in_its_own_dataset(staged):
    from phenotypic._cli._cli_directory_scanner import scan_directory_structure

    datasets = scan_directory_structure(staged.nested)
    assert sorted(p.name for p in datasets["plateA"]) == [
        "plateA_01.tif",
        "plateA_03.tif",
    ]
    assert sorted(p.name for p in datasets["plateB"]) == ["plateB_02.tif"]


def test_both_layouts_resolve_to_the_parent_images(staged, parent):
    """The staged entries are the parent's bytes, not empty placeholders."""
    for layout in (staged.flat, staged.nested):
        staged_bytes = sorted(
            path.read_bytes() for path in layout.rglob("*") if path.is_file()
        )
        assert staged_bytes == [b"plateA1", b"plateA3", b"plateB2"]


def test_staging_never_writes_into_the_parent(staged, parent):
    """``--restart``/``--overwrite`` must not be able to reach the parent."""
    assert sorted(p.name for p in parent.rglob("*") if p.is_file()) == [
        "plateA_01.tif",
        "plateA_02.tif",
        "plateA_03.tif",
        "plateB_01.tif",
        "plateB_02.tif",
    ]
    assert parent not in staged.root.parents


def test_staging_lives_under_the_server_scratch_namespace(staged, tmp_path):
    """Server scratch, not ``runs/`` and not the parent."""
    relative = staged.root.relative_to(tmp_path / "workspace")
    assert relative.parts[0] == ".phenotypic-mcp"
    assert relative.parts[1] == "subset-staging"


# --------------------------------------------------------------------------
# Idempotency, completeness, and the reported link mode
# --------------------------------------------------------------------------


def test_restaging_the_same_digest_is_a_noop(subset, tmp_path, monkeypatch):
    """Observably a no-op: no tree is built, and the published one is untouched.

    ``first.flat == second.flat`` is a pure function of the digest and holds
    whether or not the second call rebuilt the entire tree, so it proves
    nothing. Two things are checked instead. The builder is **not invoked** --
    the only observable difference the completion-marker early return makes,
    since a rebuild that loses the race is adopted at the mid-build check and
    so leaves the same tree behind. And the published tree is untouched: a
    file a fleet worker left inside it survives, and the staged entries are
    the *same inodes* rather than fresh links wearing the same names.
    """
    from phenotypic._services import staging

    first = staging.stage_subset(subset, cache_root=tmp_path)
    sentinel = first.root / "opened-by-a-worker"
    sentinel.touch()
    before = {
        path.name: path.lstat().st_ino for path in sorted(first.flat.iterdir())
    }

    builds = []
    real_build = staging._build_layouts
    monkeypatch.setattr(
        staging,
        "_build_layouts",
        lambda subset_, building: builds.append(building) or real_build(
            subset_, building
        ),
    )
    second = staging.stage_subset(subset, cache_root=tmp_path)

    assert builds == []
    assert second.root == first.root
    assert sentinel.exists()
    assert {
        path.name: path.lstat().st_ino for path in sorted(second.flat.iterdir())
    } == before


def test_a_different_image_set_stages_somewhere_else(subset, tmp_path):
    """Keyed by content, so two arms cannot collide on one directory."""
    import dataclasses

    from phenotypic._services.staging import stage_subset

    first = stage_subset(subset, cache_root=tmp_path)
    smaller = dataclasses.replace(subset, images=subset.images[:2])
    assert stage_subset(smaller, cache_root=tmp_path).flat != first.flat


def test_a_half_built_tree_is_rebuilt_not_consumed(subset, tmp_path):
    """Idempotent by **completion**, not by key.

    A key-only contract says only "this directory exists", and
    ``_load_images`` takes whatever it finds there — so a fleet could launch
    against a partial symlink tree and every run would silently train on a
    subset of the subset.
    """
    from phenotypic._services.staging import COMPLETION_MARKER, stage_subset

    staged = stage_subset(subset, cache_root=tmp_path)
    (staged.root / COMPLETION_MARKER).unlink()
    for orphan in list(staged.flat.iterdir()):
        orphan.unlink()

    restaged = stage_subset(subset, cache_root=tmp_path)
    assert restaged.root == staged.root
    assert len([p for p in restaged.flat.iterdir() if p.is_file()]) == 3
    assert (restaged.root / COMPLETION_MARKER).exists()


def test_a_refused_tree_leaves_nothing_marked_usable(subset, tmp_path):
    """A tree that failed its fidelity check must not be readable as usable.

    The build stages into a temp directory and only ``os.replace``s it onto the
    digest-keyed name once the check has passed, so a refusal leaves no marked
    directory anywhere under the cache root — and a later reader rebuilds
    rather than consuming a tree nothing vouched for.
    """
    from phenotypic._services import staging

    def _wrong_dataset(ref):
        return "wrong"

    original = staging._dataset_of
    staging._dataset_of = _wrong_dataset
    try:
        with pytest.raises(ValueError):
            staging.stage_subset(subset, cache_root=tmp_path)
    finally:
        staging._dataset_of = original

    scratch = tmp_path / staging.SERVER_SCRATCH_DIR
    assert list(tmp_path.rglob(staging.COMPLETION_MARKER)) == []
    assert list(scratch.rglob("*.tif")) == []

    # And the digest-keyed name is still free, so the retry is a fresh build.
    staged = staging.stage_subset(subset, cache_root=tmp_path)
    assert (staged.root / staging.COMPLETION_MARKER).is_file()
    assert len([p for p in staged.flat.iterdir() if p.is_file()]) == 3


def test_a_builder_that_loses_the_race_adopts_the_winners_tree(
    subset, tmp_path, monkeypatch
):
    """Sharing a directory per digest stops two arms racing to build it.

    It must not then let the loser delete the winner's published tree — a
    fleet worker may already be reading through it. Both trees are the same
    content by construction, since the digest is what named them.
    """
    from phenotypic._services import staging

    winner = staging.stage_subset(subset, cache_root=tmp_path)
    sentinel = winner.root / "opened-by-a-worker"
    sentinel.touch()

    # Rewind to the moment before the winner published, then let it publish
    # midway through the loser's build.
    (winner.root / staging.COMPLETION_MARKER).unlink()
    real_build = staging._build_layouts

    def _build_then_lose(subset_, building):
        mode = real_build(subset_, building)
        (winner.root / staging.COMPLETION_MARKER).touch()
        return mode

    monkeypatch.setattr(staging, "_build_layouts", _build_then_lose)
    loser = staging.stage_subset(subset, cache_root=tmp_path)

    assert loser.root == winner.root
    assert sentinel.exists()


def test_the_reported_link_mode_matches_the_tree(staged):
    """Reported, not guessed — and checked against what is on disk.

    ``link_mode in {"symlink", "copy"}`` is true of any hardcoded value. The
    field is only worth reading if it describes the tree, so both branches are
    asserted against the entries themselves.
    """
    entries = [path for path in staged.flat.iterdir() if path.is_file()]
    assert entries
    if staged.link_mode == "symlink":
        assert all(path.is_symlink() for path in entries)
    else:
        assert not any(path.is_symlink() for path in entries)


def test_symlinks_are_the_default_where_they_work(staged):
    """Cheap enough to restage repeatedly across an unattended campaign."""
    if staged.link_mode != "symlink":
        pytest.skip("this filesystem does not support symlink creation")
    assert all(
        path.is_symlink()
        for path in staged.flat.iterdir()
        if path.is_file()
    )


def test_copy_fallback_is_used_and_reported(subset, tmp_path, monkeypatch):
    """Windows symlink creation needs elevated privileges or Developer Mode.

    A silent copy of a large subset is a surprise worth surfacing, so the mode
    is reported rather than inferred.
    """
    from phenotypic._services import staging

    def _refuse(*args, **kwargs):
        # Spelled with the errno Windows and a FAT/network mount actually
        # report: the fallback is keyed on the reason, not on "any OSError".
        raise OSError(errno.EPERM, "symlink creation is not permitted")

    monkeypatch.setattr(staging.os, "symlink", _refuse)
    staged = staging.stage_subset(subset, cache_root=tmp_path)

    assert staged.link_mode == "copy"
    assert [p for p in staged.flat.iterdir() if p.is_file()]
    assert not any(p.is_symlink() for p in staged.flat.iterdir())
    assert sorted(p.read_bytes() for p in staged.flat.iterdir()) == [
        b"plateA1", b"plateA3", b"plateB2",
    ]


# --------------------------------------------------------------------------
# Fidelity — the check that lives in the builder or nowhere
# --------------------------------------------------------------------------


def test_fidelity_check_rejects_a_mismatched_layout(subset, tmp_path, monkeypatch):
    """The builder must verify its own output round-trips.

    ``scan_directory_structure`` only rejects *internally* inconsistent
    directories; it has no way to know what the parent looked like.
    """
    from phenotypic._services import staging

    monkeypatch.setattr(staging, "_dataset_of", lambda ref: "wrong")
    with pytest.raises(ValueError, match="fidelity|dataset"):
        staging.stage_subset(subset, cache_root=tmp_path)


def test_fidelity_check_rejects_a_collapsed_multi_dataset_parent(
    subset, tmp_path, monkeypatch
):
    """One dataset name for every image is the exact corruption nesting prevents."""
    from phenotypic._services import staging

    monkeypatch.setattr(staging, "_dataset_of", lambda ref: "plateA")
    with pytest.raises(ValueError, match="fidelity|dataset"):
        staging.stage_subset(subset, cache_root=tmp_path)


def test_fidelity_check_rejects_images_filed_under_the_wrong_dataset(
    subset, tmp_path, monkeypatch
):
    """The check compares the *contents* of each dataset, not just its name.

    Swapping the two plates leaves the set of dataset names identical, so a
    name-only comparison passes — while every row's ``Metadata_Dataset`` would
    come out wrong, which is the corruption the whole check exists to catch.
    """
    from phenotypic._services import staging

    swap = {"plateA": "plateB", "plateB": "plateA"}
    monkeypatch.setattr(
        staging,
        "_dataset_of",
        lambda ref: swap[ref.relative_path.split("/", 1)[0]],
    )
    with pytest.raises(ValueError, match="fidelity|dataset"):
        staging.stage_subset(subset, cache_root=tmp_path)


def test_a_missing_source_image_is_refused(subset, tmp_path):
    """Staging fewer images than the subset names is the silent-thinning failure."""
    from phenotypic._services.staging import stage_subset

    (subset.parent / "plateB" / "plateB_02.tif").unlink()
    with pytest.raises(FileNotFoundError):
        stage_subset(subset, cache_root=tmp_path)


# --------------------------------------------------------------------------
# Flat-layout name collisions
# --------------------------------------------------------------------------


def test_flat_layout_disambiguates_a_repeated_filename(tmp_path):
    """Two datasets can both contain ``plate_001.tif`` (§10.2).

    Flattening them onto one name would drop an image and the run would train
    on a subset of the subset with nothing to report it.
    """
    from phenotypic._services.staging import SubsetToStage, stage_subset
    from phenotypic.subset import ImageRef

    root = tmp_path / "plates"
    for dataset in ("plateA", "plateB"):
        (root / dataset).mkdir(parents=True)
        (root / dataset / "plate_001.tif").write_bytes(dataset.encode())

    subset = SubsetToStage(
        parent=root,
        images=tuple(
            ImageRef(path=root / relative, relative_path=relative)
            for relative in ("plateA/plate_001.tif", "plateB/plate_001.tif")
        ),
    )
    staged = stage_subset(subset, cache_root=tmp_path / "workspace")

    files = [p for p in staged.flat.iterdir() if p.is_file()]
    assert len(files) == 2
    assert sorted(p.read_bytes() for p in files) == [b"plateA", b"plateB"]


def test_flat_layout_keeps_the_image_suffix_when_disambiguating(tmp_path):
    """``_load_images`` filters on the suffix; a mangled one matches nothing."""
    from phenotypic._services.staging import SubsetToStage, stage_subset
    from phenotypic.subset import ImageRef

    root = tmp_path / "plates"
    for dataset in ("plateA", "plateB"):
        (root / dataset).mkdir(parents=True)
        (root / dataset / "plate_001.tif").write_bytes(dataset.encode())

    subset = SubsetToStage(
        parent=root,
        images=tuple(
            ImageRef(path=root / relative, relative_path=relative)
            for relative in ("plateA/plate_001.tif", "plateB/plate_001.tif")
        ),
    )
    staged = stage_subset(subset, cache_root=tmp_path / "workspace")

    assert all(p.suffix == ".tif" for p in staged.flat.iterdir() if p.is_file())


# --------------------------------------------------------------------------
# A flat parent — no subdirectories at all
# --------------------------------------------------------------------------


def test_a_flat_parent_stages_both_layouts(tmp_path):
    """``scan_directory_structure`` names a flat parent's dataset after the
    directory, so the nested layout has to reproduce that name too."""
    from phenotypic._cli._cli_directory_scanner import scan_directory_structure
    from phenotypic._services.staging import SubsetToStage, stage_subset
    from phenotypic.subset import ImageRef

    root = tmp_path / "myplates"
    root.mkdir()
    for index in (1, 2, 3):
        (root / f"p{index:02d}.tif").write_bytes(f"p{index}".encode())

    subset = SubsetToStage(
        parent=root,
        images=tuple(
            ImageRef(path=root / relative, relative_path=relative)
            for relative in ("p01.tif", "p03.tif")
        ),
    )
    staged = stage_subset(subset, cache_root=tmp_path / "workspace")

    assert len([p for p in staged.flat.iterdir() if p.is_file()]) == 2
    assert set(scan_directory_structure(staged.nested)) == {"myplates"}


def test_an_empty_subset_is_refused(parent, tmp_path):
    """A study of nothing passes every downstream shape check."""
    from phenotypic._services.staging import SubsetToStage, stage_subset

    # Matched on the deliberate message: without the guard an empty tree still
    # raises, but from ``scan_directory_structure`` finding no images -- an
    # incidental refusal that would disappear the moment the check moved.
    with pytest.raises(ValueError, match="empty subset"):
        stage_subset(
            SubsetToStage(parent=parent, images=()), cache_root=tmp_path
        )


def test_the_digest_is_prefixed_and_content_defined(subset, tmp_path):
    """Shares the ``sha256:`` spelling every other fingerprint here uses."""
    from phenotypic._services.staging import stage_subset, subset_digest

    staged = stage_subset(subset, cache_root=tmp_path)
    assert staged.digest.startswith("sha256:")
    assert staged.digest == subset_digest(subset)
    assert staged.root.name == staged.digest.split(":", 1)[1]


def test_the_digest_ignores_the_order_the_images_were_listed_in(subset):
    """A subset is a *set*; a reshuffled list must not stage twice."""
    import dataclasses

    from phenotypic._services.staging import subset_digest

    reversed_ = dataclasses.replace(subset, images=tuple(reversed(subset.images)))
    assert subset_digest(reversed_) == subset_digest(subset)


def test_the_digest_ignores_where_the_parent_lives(subset, tmp_path):
    """The same images under a different mount are the same subset."""
    import dataclasses
    import shutil

    from phenotypic._services.staging import subset_digest
    from phenotypic.subset import ImageRef

    elsewhere = tmp_path / "mirror" / "plates"
    shutil.copytree(subset.parent, elsewhere)
    moved = dataclasses.replace(
        subset,
        parent=elsewhere,
        images=tuple(
            ImageRef(path=elsewhere / ref.relative_path,
                     relative_path=ref.relative_path)
            for ref in subset.images
        ),
    )
    assert subset_digest(moved) == subset_digest(subset)


def test_the_staging_tree_is_created_with_the_process_umask(subset, tmp_path):
    """Staging neither tightens nor loosens the caller's umask.

    ``os.access(..., R_OK | X_OK)`` was asked as the **owner** of a directory
    this process had just created, who always has access — so it could not
    fail, and in particular could not see the case it named. The mode bits are
    asserted directly against a umask fixed for the duration instead, which
    does discriminate: a builder that created the tree ``0o700``, or that
    chmod'd it, fails here.

    Inheriting the umask is the right policy and not an oversight. Every
    reader of a staging tree on this cluster is a SLURM job running as the
    submitting user, so widening it would hand out access nobody asked for,
    and narrowing it would be staging making a site policy decision it has no
    standing to make.
    """
    import stat as stat_module

    from phenotypic._services.staging import stage_subset

    original = os.umask(0o022)
    try:
        staged = stage_subset(subset, cache_root=tmp_path / "workspace")
    finally:
        os.umask(original)

    for directory in (staged.root, staged.flat, staged.nested):
        assert stat_module.S_IMODE(directory.stat().st_mode) == 0o755


def test_a_failure_that_is_not_a_symlink_refusal_is_not_absorbed(
    subset, tmp_path, monkeypatch
):
    """A blanket ``except OSError`` answered every failure by copying.

    A full disk or a read-only cache root is not a reason to duplicate tens of
    gigabytes, and ``FileExistsError`` — a builder bug — became a ``copy2``
    **over** the destination, with the whole subset then reported as
    ``"copy"`` on the strength of it.
    """
    from phenotypic._services import staging

    def _no_space(*args, **kwargs):
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr(staging.os, "symlink", _no_space)
    with pytest.raises(OSError) as raised:
        staging.stage_subset(subset, cache_root=tmp_path)
    assert raised.value.errno == errno.ENOSPC


def test_a_destination_that_already_exists_is_not_silently_overwritten(tmp_path):
    """``FileExistsError`` is a builder bug, not a reason to copy over it."""
    from phenotypic._services.staging import _link_or_copy

    source = tmp_path / "a.tif"
    source.write_bytes(b"source")
    destination = tmp_path / "b.tif"
    destination.write_bytes(b"occupied")

    with pytest.raises(FileExistsError):
        _link_or_copy(source, destination)
    assert destination.read_bytes() == b"occupied"


# --------------------------------------------------------------------------
# Duplicate refs
# --------------------------------------------------------------------------


def test_a_repeated_image_is_deduplicated_on_construction(parent):
    """Neither layout can represent a duplicate.

    ``_flat_names`` counts the repeat as a *cross-dataset* collision and
    mangles the flat filename, and the second link surfaces as a bare
    ``shutil.SameFileError`` from deep inside the builder.
    """
    from phenotypic._services.staging import SubsetToStage
    from phenotypic.subset import ImageRef

    relative = "plateA/plateA_01.tif"
    ref = ImageRef(path=parent / relative, relative_path=relative)
    subset = SubsetToStage(parent=parent, images=(ref, ref))

    assert subset.images == (ref,)


def test_a_repeated_image_still_stages_under_its_own_name(parent, tmp_path):
    """The dedupe has to happen before ``_flat_names`` counts basenames."""
    from phenotypic._services.staging import SubsetToStage, stage_subset
    from phenotypic.subset import ImageRef

    relative = "plateA/plateA_01.tif"
    ref = ImageRef(path=parent / relative, relative_path=relative)
    staged = stage_subset(
        SubsetToStage(parent=parent, images=(ref, ref)), cache_root=tmp_path
    )

    assert [path.name for path in staged.flat.iterdir() if path.is_file()] == [
        "plateA_01.tif"
    ]


def test_two_different_files_claiming_one_relative_path_are_refused(parent, tmp_path):
    """The relative path is the identity; it cannot name two files."""
    from phenotypic._services.staging import SubsetToStage
    from phenotypic.subset import ImageRef

    relative = "plateA/plateA_01.tif"
    with pytest.raises(ValueError, match="twice"):
        SubsetToStage(
            parent=parent,
            images=(
                ImageRef(path=parent / relative, relative_path=relative),
                ImageRef(path=tmp_path / "elsewhere.tif", relative_path=relative),
            ),
        )


# --------------------------------------------------------------------------
# The digest keys a *subset*, not a layout
# --------------------------------------------------------------------------


def _one_plate_parent(root, marker):
    """A parent with the layout every plate-imaging experiment here shares."""
    from phenotypic._services.staging import SubsetToStage
    from phenotypic.subset import ImageRef

    (root / "plateA").mkdir(parents=True)
    chosen = ("plateA/plateA_01.tif", "plateA/plateA_02.tif")
    for relative in chosen:
        (root / relative).write_bytes(marker + relative.encode())
    return SubsetToStage(
        parent=root,
        images=tuple(
            ImageRef(path=root / relative, relative_path=relative)
            for relative in chosen
        ),
    )


def test_two_parents_with_the_same_layout_do_not_share_a_staging_tree(tmp_path):
    """``plateA/plateA_01.tif`` is a naming convention, not an identity.

    Keyed on relative paths alone, two unrelated experiments produced one
    digest, and the second caller took the completion-marker early return and
    received a tree of symlinks into the **first** experiment's images. The
    fidelity check never fired, because nothing was built. One workspace
    ``cache_root`` shared across a campaign is the normal configuration, so
    this was the expected setup rather than a corner.
    """
    from phenotypic._services.staging import stage_subset

    cache_root = tmp_path / "workspace"
    first = _one_plate_parent(tmp_path / "experiment1" / "plates", b"EXPERIMENT-ONE-")
    second = _one_plate_parent(tmp_path / "experiment2" / "plates", b"EXPERIMENT-TWO-")

    staged_first = stage_subset(first, cache_root=cache_root)
    staged_second = stage_subset(second, cache_root=cache_root)

    assert staged_first.root != staged_second.root
    assert all(
        path.read_bytes().startswith(b"EXPERIMENT-TWO-")
        for path in staged_second.flat.iterdir()
        if path.is_file()
    )
    assert all(
        Path(second.parent) in path.resolve().parents
        for path in staged_second.flat.iterdir()
        if path.is_file()
    )


def test_the_digest_changes_when_an_image_changes_under_a_stable_name(
    subset, tmp_path
):
    """Names are not identities: the bytes behind them are in the key too."""
    from phenotypic._services.staging import subset_digest

    before = subset_digest(subset)
    (subset.parent / "plateA" / "plateA_01.tif").write_bytes(b"rescanned")
    assert subset_digest(subset) != before


def test_the_digest_survives_a_preserving_copy_at_another_mount(subset, tmp_path):
    """Content identity, so the timestamp skew a real copy introduces is moot.

    Measured on this cluster's ``gpfs``, ``shutil.copytree`` of a freshly
    written file moves the destination's ``mtime_ns`` forward by 3-32 ms every
    time, so the timestamp is deliberately *not* a term. The copy's mtimes are
    left exactly as the filesystem wrote them here.
    """
    import dataclasses
    import shutil

    from phenotypic._services.staging import subset_digest
    from phenotypic.subset import ImageRef

    elsewhere = tmp_path / "mirror" / "plates"
    shutil.copytree(subset.parent, elsewhere)
    moved = dataclasses.replace(
        subset,
        parent=elsewhere,
        images=tuple(
            ImageRef(path=elsewhere / ref.relative_path,
                     relative_path=ref.relative_path)
            for ref in subset.images
        ),
    )
    assert subset_digest(moved) == subset_digest(subset)
