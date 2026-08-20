"""Materialize a subset as the two directory layouts the engines need.

**Neither engine accepts a file list.** ``python -m phenotypic.tune run``
takes ``-i/--input`` as an image *directory* and loads it with a
**non-recursive** ``Path(input_dir).iterdir()``; ``python -m phenotypic`` takes
``-i/--input`` as a single path with no ``multiple=True``, and
``scan_directory_structure`` reads one level of subdirectories as separate
datasets. There is no manifest flag and no repeated ``-i`` on either, and
``--sample N`` only randomly *thins* datasets already discovered. So §10's
whole subset boundary depends on staging a directory and passing that.

**Two layouts, because the two engines want opposite things.** At the root of
a *nested* directory ``_load_images`` sees only subdirectories, matches zero
images, and the run dies on ``SystemExit("no images found under …")``.
Conversely ``scan_directory_structure`` derives ``Metadata_Dataset`` from
subdirectory names, so handing *deploy* a flat directory silently relabels
every row's dataset to the staging folder name — the exact corruption nesting
prevents. The split is cheap because both layouts are link trees under one
digest: the marginal cost is inodes, not bytes.

::

    <cache_root>/.phenotypic-mcp/subset-staging/<subset-digest>/
    ├── .complete             # written last; absent means "not usable"
    ├── flat/                 # tune — a NON-RECURSIVE iterdir
    │   ├── plateA_01.tif -> …/data/plates/plateA/plateA_01.tif
    │   └── plateB_03.tif -> …
    └── nested/               # deploy — Metadata_Dataset comes from subdir names
        ├── plateA/plateA_01.tif -> …
        └── plateB/plateB_03.tif -> …
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from phenotypic.sdk_._io_constants import file_fingerprint, update_framed_term
from phenotypic.subset import ImageRef

#: Server scratch namespace. Staging lives here and not under ``runs/`` or the
#: parent, so ``--restart``/``--overwrite`` semantics can never reach the
#: parent images through a staging directory.
SERVER_SCRATCH_DIR: Final[str] = ".phenotypic-mcp"

#: Subdirectory of :data:`SERVER_SCRATCH_DIR` holding one directory per subset.
SUBSET_STAGING_DIR: Final[str] = "subset-staging"

#: Marker written **last**, after the fidelity check passes. A reader that does
#: not find it must treat the directory as absent — see :func:`stage_subset`.
COMPLETION_MARKER: Final[str] = ".complete"

#: Layout subdirectory names, one per engine.
FLAT_LAYOUT_DIR: Final[str] = "flat"
NESTED_LAYOUT_DIR: Final[str] = "nested"

#: Separator joining a dataset name to a filename when two datasets contribute
#: the same basename to the flat layout. Double so it cannot be confused with
#: an underscore already in a plate name.
_FLAT_COLLISION_SEPARATOR: Final[str] = "__"

LinkMode = Literal["symlink", "copy"]


@dataclass(frozen=True)
class SubsetToStage:
    """The subset fields staging reads (§10.2).

    Attributes:
        parent: The parent image directory the subset was drawn from. Needed
            in its own right, not just as a prefix: the fidelity check rescans
            it to learn what dataset names the engine *would* derive.
        images: The chosen images. ``relative_path`` is the identity — bare
            filenames cannot disambiguate two datasets that both contain
            ``plate_001.tif``.
    """

    parent: Path
    images: tuple[ImageRef, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Drop repeated refs, and refuse two refs claiming one relative path.

        Neither layout can represent a duplicate: ``_flat_names`` counts the
        repeat as a *cross-dataset* collision and mangles the flat filename,
        and the second link surfaces as a bare ``shutil.SameFileError`` from
        deep inside the builder. ``SubsetSelection.select`` deduplicates, but
        this dataclass accepts any tuple and is constructed directly by the
        hand-picked ``user_named`` path.

        Two *different* refs sharing a relative path are not deduplicable --
        the relative path is the identity, so the pair is a contradiction
        about which file that identity names -- and are refused.
        """
        unique: dict[str, ImageRef] = {}
        for ref in self.images:
            existing = unique.setdefault(ref.relative_path, ref)
            if existing.path != ref.path:
                raise ValueError(
                    f"subset names {ref.relative_path!r} twice, at "
                    f"{existing.path} and {ref.path}; the relative path is "
                    "the identity, so it cannot name two different files"
                )
        object.__setattr__(self, "images", tuple(unique.values()))


@dataclass(frozen=True)
class StagedSubset:
    """Where a subset was materialized, and how.

    Attributes:
        root: The digest-keyed staging directory holding both layouts.
        flat: Pass this to ``phenotypic.tune``.
        nested: Pass this to ``python -m phenotypic``.
        link_mode: ``"symlink"`` normally; ``"copy"`` where symlink creation
            is refused. Reported rather than inferred, because a silent copy
            of a large subset is a surprise worth surfacing.
        digest: The ``sha256:``-prefixed content digest keying :attr:`root`.
    """

    root: Path
    flat: Path
    nested: Path
    link_mode: LinkMode
    digest: str


def _dataset_of(ref: ImageRef) -> str:
    """The dataset name the engine would derive for one image.

    The first segment of the parent-relative path for a nested parent. For a
    flat parent there is no segment, and the engine names the dataset after the
    directory itself — which the ref does not know, so the caller substitutes
    the parent's name. Module-level rather than a method so the fidelity check
    has something independent to disagree with.
    """
    head, separator, _ = ref.relative_path.partition("/")
    return head if separator else ""


def subset_digest(subset: SubsetToStage) -> str:
    """Content digest of a subset's image set.

    Each chosen image contributes its sorted parent-relative **path** and its
    **content fingerprint**. Sorting is what makes a reshuffled list the same
    subset; the pair of terms is what makes the key an identity rather than a
    layout.

    **Both halves are load-bearing, and the second is the one that keeps two
    experiments apart.** Paths alone are parent-*blind*:
    ``plateA/plateA_01.tif`` is a naming convention, not an identity, so two
    unrelated experiments that share a layout produce one digest, and
    :func:`stage_subset` hands the second caller a tree of links into the
    **first** experiment's images -- silently, since a tree it adopts is never
    built and so never fidelity-checked. One workspace ``cache_root`` shared
    across a campaign is the normal configuration, so that is the expected
    setup rather than a corner.

    **Why contents and not ``(size, mtime_ns)``**, which is what
    :func:`~phenotypic.sdk_.directory_digest` uses one layer up: the digest
    must stay stable across the ``/rhome`` <-> ``/bigdata`` mount aliases and
    across a preserving copy, and ``mtime_ns`` does not deliver that. Measured
    on this cluster's ``gpfs``, ``shutil.copytree`` of a freshly written file
    moves the destination's ``mtime_ns`` forward by 3-32 ms every time, so a
    timestamp-keyed subset would restage a copy as a different subset.
    Contents give the property by construction. The cost that forbids it for
    ``directory_digest`` does not apply here: a subset is tens of images, not
    the 480-image parent -- that bound is the whole reason the subset boundary
    exists.

    The parent's own location is deliberately **absent**: the same images
    reached through a different mount are the same subset, and folding in
    ``parent.resolve()`` would key the staging tree on which alias the caller
    happened to use.

    Args:
        subset: The subset to key. Every named image must exist.

    Returns:
        A ``"sha256:<hex>"`` digest, matching the spelling
        :func:`~phenotypic.sdk_.file_fingerprint` uses.

    Raises:
        FileNotFoundError: If a named image is not present.
    """
    digest = hashlib.sha256()
    for ref in sorted(subset.images, key=lambda ref: ref.relative_path):
        for term in (ref.relative_path, file_fingerprint(Path(ref.path))):
            update_framed_term(digest, term)
    return f"sha256:{digest.hexdigest()}"


def subset_staging_dir(cache_root: Path, digest: str) -> Path:
    """The digest-keyed staging directory for ``digest`` under ``cache_root``."""
    return (
        Path(cache_root)
        / SERVER_SCRATCH_DIR
        / SUBSET_STAGING_DIR
        / digest.split(":", 1)[-1]
    )


def stage_subset(subset: SubsetToStage, *, cache_root: Path) -> StagedSubset:
    """Materialize ``subset`` as a flat and a nested link tree.

    **Idempotent by completion, not by key.** Sharing one directory per digest
    is what stops two arms racing to build it; it is not what stops the second
    arm *reading* it half-built, and ``_load_images`` takes whatever it finds
    there — so a fleet could launch against a partial tree and every run would
    silently train on a subset of the subset. The tree is therefore built in a
    temp directory, ``os.replace``d onto the digest-keyed name, and marked
    :data:`COMPLETION_MARKER` **last**, after the fidelity check. A directory
    without the marker is treated as absent and rebuilt.

    **Fidelity is a check, not just a property.** Nothing in the engines can
    catch a mismatch: ``scan_directory_structure`` rejects only *internally*
    inconsistent directories and has no way to know what the parent looked
    like. So the builder rescans the parent, rescans its own ``nested/`` tree,
    and refuses if the dataset names disagree.

    Args:
        subset: The subset to materialize.
        cache_root: Workspace root. Staging is placed under its
            :data:`SERVER_SCRATCH_DIR` subdirectory.

    Returns:
        A :class:`StagedSubset` naming both layouts and the link mode used.

    Raises:
        ValueError: If the subset is empty, or the staged ``nested/`` tree does
            not reproduce the parent's dataset names for these images.
        FileNotFoundError: If a named image is not present under the parent.
            Staging fewer images than the subset names is the silent-thinning
            failure the whole boundary exists to prevent.
    """
    if not subset.images:
        raise ValueError(
            "refusing to stage an empty subset: an empty image set passes "
            "every downstream shape check and produces a study of nothing"
        )
    for ref in subset.images:
        if not Path(ref.path).is_file():
            raise FileNotFoundError(
                f"subset image {ref.relative_path!r} is not present at {ref.path}"
            )

    digest = subset_digest(subset)
    root = subset_staging_dir(cache_root, digest)
    if (root / COMPLETION_MARKER).is_file():
        return _adopt(root, digest)

    root.parent.mkdir(parents=True, exist_ok=True)
    building = root.parent / f".{root.name}.building.{os.getpid()}"
    if building.exists():
        shutil.rmtree(building)

    try:
        link_mode = _build_layouts(subset, building)
        _assert_layout_fidelity(subset, building / NESTED_LAYOUT_DIR)
        if (root / COMPLETION_MARKER).is_file():
            # Another builder finished while this one worked. Adopt its tree:
            # replacing a *marked* directory would delete files a fleet worker
            # may already be reading through, and both trees are the same
            # content by construction — the digest is what named them.
            return _adopt(root, digest)
        if root.exists():
            shutil.rmtree(root)
        # The rename is atomic within a filesystem; the marker covers the
        # window after it in which the tree exists but nothing has vouched
        # for it.
        os.replace(building, root)
        (root / COMPLETION_MARKER).touch()
    finally:
        if building.exists():
            shutil.rmtree(building, ignore_errors=True)

    return _describe(root, digest, link_mode)


def _describe(root: Path, digest: str, link_mode: LinkMode) -> StagedSubset:
    """Name both layout subdirectories of one staging root.

    The two layout names are derived here and nowhere else, so a builder's
    result and an adopter's cannot come to disagree about where ``flat/`` and
    ``nested/`` are.
    """
    return StagedSubset(
        root=root,
        flat=root / FLAT_LAYOUT_DIR,
        nested=root / NESTED_LAYOUT_DIR,
        link_mode=link_mode,
        digest=digest,
    )


def _adopt(root: Path, digest: str) -> StagedSubset:
    """Describe an already-complete staging directory without rebuilding it.

    The link mode is *observed* from the tree rather than assumed: whoever
    built it may have fallen back to copying on a filesystem this caller has
    no reason to know about.
    """
    return _describe(root, digest, _observed_link_mode(root / FLAT_LAYOUT_DIR))


def _build_layouts(subset: SubsetToStage, staging: Path) -> LinkMode:
    """Write both layouts under ``staging`` and report the link mode used."""
    flat = staging / FLAT_LAYOUT_DIR
    nested = staging / NESTED_LAYOUT_DIR
    flat.mkdir(parents=True)
    nested.mkdir(parents=True)

    fallback_dataset = Path(subset.parent).name
    flat_names = _flat_names(subset)
    mode: LinkMode = "symlink"

    for ref in subset.images:
        source = Path(ref.path).resolve()
        dataset = _dataset_of(ref) or fallback_dataset
        nested_dir = nested / dataset
        nested_dir.mkdir(parents=True, exist_ok=True)
        for destination in (
            flat / flat_names[ref.relative_path],
            nested_dir / Path(ref.relative_path).name,
        ):
            if _link_or_copy(source, destination) == "copy":
                mode = "copy"
    return mode


def _flat_names(subset: SubsetToStage) -> dict[str, str]:
    """Map each relative path to a unique name for the flat layout.

    Two datasets can both contain ``plate_001.tif``; flattening them onto one
    name would drop an image and the run would train on a subset of the subset
    with nothing to report it. Colliding names are prefixed with their dataset,
    and only the colliding ones — an uncollided plate keeps the name a human
    recognizes. The suffix is preserved, because ``_load_images`` filters on it
    and a mangled suffix matches nothing.
    """
    counts: dict[str, int] = {}
    for ref in subset.images:
        name = Path(ref.relative_path).name
        counts[name] = counts.get(name, 0) + 1

    names: dict[str, str] = {}
    for ref in subset.images:
        path = Path(ref.relative_path)
        if counts[path.name] == 1:
            names[ref.relative_path] = path.name
            continue
        dataset = _dataset_of(ref) or "root"
        names[ref.relative_path] = (
            f"{dataset}{_FLAT_COLLISION_SEPARATOR}{path.stem}{path.suffix}"
        )
    return names


#: ``errno`` values meaning "this platform or filesystem will not make a
#: symlink here". Windows without Developer Mode reports ``EPERM``/``EINVAL``
#: (``ERROR_PRIVILEGE_NOT_HELD``); a FAT/exFAT or restrictive network mount
#: reports ``EPERM``, ``EACCES``, ``ENOSYS`` or ``EOPNOTSUPP``.
_SYMLINK_UNSUPPORTED: Final[frozenset[int]] = frozenset({
    errno.EACCES,
    errno.EINVAL,
    errno.ENOSYS,
    errno.EOPNOTSUPP,
    errno.EPERM,
})


def _link_or_copy(source: Path, destination: Path) -> LinkMode:
    """Symlink ``source`` to ``destination``, copying if the platform refuses.

    Windows symlink creation needs elevated privileges or Developer Mode and
    this project supports Windows, so the failure is expected rather than
    exceptional — but it must be *reported*, not absorbed.

    Only the errors that actually mean "no symlinks here" fall back. A blanket
    ``except OSError`` also caught ``FileExistsError`` and answered it by
    ``copy2``-ing **over** the destination — turning a builder bug into a
    silent overwrite, and reporting the whole subset as ``"copy"`` on the
    strength of it. Everything else propagates: a full disk or a read-only
    cache root is not a reason to duplicate tens of gigabytes.

    Raises:
        OSError: For any failure that is not a refusal to create symlinks.
    """
    try:
        os.symlink(source, destination)
    except NotImplementedError:
        pass
    except OSError as error:
        if error.errno not in _SYMLINK_UNSUPPORTED:
            raise
    else:
        return "symlink"
    shutil.copy2(source, destination)
    return "copy"


def _observed_link_mode(flat: Path) -> LinkMode:
    """Report the mode an already-complete staging directory was built with."""
    for path in flat.iterdir():
        return "symlink" if path.is_symlink() else "copy"
    return "symlink"


def _assert_layout_fidelity(subset: SubsetToStage, nested: Path) -> None:
    """Refuse a nested tree whose dataset names differ from the parent's.

    Both sides are derived independently: the expectation comes from scanning
    the **parent**, the observation from scanning the tree just built. Deriving
    both from :func:`_dataset_of` would make the check agree with itself no
    matter what that function returned.
    """
    from phenotypic._cli._cli_directory_scanner import scan_directory_structure

    chosen = {ref.relative_path for ref in subset.images}
    expected: dict[str, set[str]] = {}
    for dataset, paths in scan_directory_structure(Path(subset.parent)).items():
        for path in paths:
            relative = Path(path).relative_to(subset.parent).as_posix()
            if relative in chosen:
                expected.setdefault(dataset, set()).add(Path(relative).name)

    observed: dict[str, set[str]] = {
        dataset: {Path(path).name for path in paths}
        for dataset, paths in scan_directory_structure(nested).items()
    }

    if observed != expected:
        raise ValueError(
            "subset staging fidelity check failed: the nested layout would "
            f"give the engine datasets {_summarize(observed)} where the parent "
            f"gives {_summarize(expected)}. Passing it on would silently "
            "relabel every row's Metadata_Dataset."
        )


def _summarize(datasets: dict[str, set[str]]) -> str:
    """Render a dataset→filenames map deterministically for an error message."""
    return "{" + ", ".join(
        f"{name}: {sorted(files)}" for name, files in sorted(datasets.items())
    ) + "}"
