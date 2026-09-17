"""Crash-recovery evidence and locking for per-store measurement authority.

**The writers are gone.** ``--mode recompile`` no longer rewrites per-store
embedded measurement tables (user ruling, 2026-09-11), so
``begin_``/``promote_``/``clear_recompile_table_transition`` and their staging
helpers went with the rewrite. What remains is the **reading** half, and it is
not dead: a tree that a previous release left mid-transition -- promoted bytes,
a stale record, an uncleared receipt -- is still recoverable, and
:func:`recoverable_recompile_measurement_sources` is what keeps
:func:`assert_no_unrecoverable_measurement_authority` from aborting on it.
Retire the transition readers once no supported release can have written a
receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


from phenotypic.sdk_ import (
    DIR_RESULTS,
    DIR_ZARR,
    MEASUREMENT_TABLE_RELATIVE_PATH,
    STORE_SUFFIX,
    dataset_measurements_dir,
    image_completion_marker_path,
    image_record_path,
    progress_dir,
    zarr_store_path,
)
from phenotypic.sdk_._measurement_tables import (
    _valid_embedded_measurement_contract,
)

from phenotypic.sdk_._image_record import RECORD_VERSION

from ._cli_completion import (
    ARTIFACT_KIND_FILE,
    ARTIFACT_KIND_STORE,
    SUCCESS_MARKER_VERSION,
    _sha256,
    _store_artifact_matches,
)

_TRANSITION_VERSION = 1
_TRANSITION_DIR = "table-transitions"


def _image_authority_shapes(
    output_root: Path, dataset_name: str, stem: str
) -> tuple[tuple[Path, int], ...]:
    """Return ``(payload path, the version that shape must carry)``, best first.

    **Both shapes, each on its own predicate** -- P3's precedent, and the
    ruling that governs every repointed site in this module. The record is
    what a forward tree writes (D1's clean break); the legacy
    ``image_complete/`` marker is what a pre-record tree still has, and a
    legacy tree can still reach ``--mode recompile`` until the schema gate is
    armed.

    The version travels **with** the path rather than being checked against a
    single constant, because the two shapes disagree on it -- ``RECORD_VERSION``
    is 1 and ``SUCCESS_MARKER_VERSION`` is 2. A shape-blind ``version in {1, 2}``
    would accept a marker found at the record path, which is the confusion this
    pairing exists to make impossible.

    LEGACY MARKER ARM -- DELETE WHEN: the schema gate is armed and refuses
    legacy trees before they reach recompile (P7 Task 5 Step 1d sets
    ``_schema_shape.SCHEMA_GATE_ARMED = True``). The same trigger retires
    every other legacy arm this phase adds, so they go together rather than
    one at a time. When it holds, this returns the record shape alone and the
    pairing collapses. (``_standalone_marker_sources``, which this note used
    to name as the sibling arm, went with the recompile rewrite on
    2026-09-11.)
    """
    return (
        (image_record_path(output_root, dataset_name, stem), RECORD_VERSION),
        (
            image_completion_marker_path(output_root, dataset_name, stem),
            SUCCESS_MARKER_VERSION,
        ),
    )


def image_authority_path(
    output_root: Path, dataset_name: str, stem: str
) -> Path:
    """Return the payload path recompile should read for one image.

    The record when it exists, the legacy marker when only that does, and the
    record path when neither does -- the shape a forward tree is supposed to
    have, so a caller that only asks "does this claim authority?" gets ``False``
    rather than a path pointing at the wrong schema.
    """
    shapes = _image_authority_shapes(output_root, dataset_name, stem)
    for path, _version in shapes:
        if path.is_file():
            return path
    return shapes[0][0]


def image_authority_payload(
    output_root: Path, dataset_name: str, stem: str
) -> tuple[Path, dict[str, Any], int]:
    """Read one image's authority payload and say which shape it is.

    Args:
        output_root: Resolved run output root.
        dataset_name: Dataset name.
        stem: Image stem.

    Returns:
        ``(path, payload, expected_version)``.

    Raises:
        FileNotFoundError: Neither shape is present.
        OSError: The payload could not be read.
        json.JSONDecodeError: The payload is not JSON. **Deliberately not
            caught**: a corrupt record must not silently fall back to a legacy
            marker, which would let a tree be judged on the wrong schema.
    """
    shapes = _image_authority_shapes(output_root, dataset_name, stem)
    for path, version in shapes:
        if not path.is_file():
            continue
        return path, json.loads(path.read_text(encoding="utf-8")), version
    raise FileNotFoundError(str(shapes[0][0]))


def recompile_store_lock_path(
    output_dir: Path, dataset_name: str, stem: str
) -> Path:
    """Return the lock shared by canonical recompile mutations for one store.

    **Derived from the record path, and it must not go back to the marker
    path.** ``exclusive_path_lock`` does ``path.parent.mkdir(parents=True,
    exist_ok=True)`` before opening, so deriving this from
    ``image_completion_marker_path`` made merely *taking* the lock create
    ``.phenotypic/progress/image_complete/<ds>/`` on a tree the current build
    wrote. Nothing removes it, and schema signal 1
    (``sdk_/_schema_shape.py``) is a **directory-existence** probe -- so one
    ``--mode recompile`` was enough to make ``requires_conversion`` answer
    ``CONVERT`` for a forward tree, and every writing mode would then refuse
    it and point at ``--mode migrate``, which does not remove that directory.

    That is the exact proposition
    ``test_a_tree_this_build_wrote_needs_no_conversion`` exists to hold up --
    the standing evidence P7 Task 5 Step 1d needs before arming the gate --
    and it could not see this, because it never runs recompile.

    A lock is not a marker and has no business being keyed off one. Under
    ``progress/images/`` it sits beside the records it guards, and signal 1
    does not probe that directory.
    """
    return image_record_path(output_dir, dataset_name, stem).with_suffix(
        ".recompile-store.lock"
    )


def _transition_root(output_dir: Path, dataset_name: str) -> Path:
    """Return the durable transition directory for one dataset."""
    return (
        progress_dir(Path(output_dir))
        / "recompile"
        / _TRANSITION_DIR
        / dataset_name
    )


def recompile_table_transition_path(
    output_dir: Path, dataset_name: str, stem: str
) -> Path:
    """Return the durable transition record for one embedded table."""
    return _transition_root(output_dir, dataset_name) / f"{stem}.json"


def _marker_measurement_fingerprint(
    output_root: Path,
    marker: dict[str, Any],
    table_path: Path,
) -> tuple[int, str]:
    """Return the marker-bound prior table fingerprint or raise."""
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Marker has no artifact mapping")
    descriptor = artifacts.get("measurements")
    if not isinstance(descriptor, dict):
        raise ValueError("Marker has no measurement descriptor")
    relative = descriptor.get("path")
    size = descriptor.get("size")
    sha256 = descriptor.get("sha256")
    if (
        not isinstance(relative, str)
        or (output_root / relative).resolve()
        != table_path.resolve(strict=True)
        or descriptor.get("kind", ARTIFACT_KIND_FILE) != ARTIFACT_KIND_FILE
        or not isinstance(size, int)
        or size < 0
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
    ):
        raise ValueError("Marker measurement descriptor is invalid")
    return size, sha256


_IDENTITY_BOUND_DIRECTORY_OPERATIONS = (
    os.name == "posix"
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_NONBLOCK")
    and os.listdir in os.supports_fd
    and all(
        operation in os.supports_dir_fd
        for operation in (os.open, os.mkdir, os.stat, os.unlink, os.rename)
    )
)


def _require_identity_bound_directory_operations() -> None:
    """Fail closed unless directory-relative no-follow I/O is available."""
    if not _IDENTITY_BOUND_DIRECTORY_OPERATIONS:
        raise RuntimeError(
            "This platform cannot safely access recompile transition directories"
        )


def _validate_transition_component(component: str) -> None:
    """Reject non-canonical transition path components."""
    if component in {"", ".", ".."} or Path(component).name != component:
        raise ValueError("Transition directory is not canonical")


@contextmanager
def _open_transition_directory(
    output_root: Path,
    dataset_name: str,
) -> Iterator[tuple[Path, int]]:
    """Hold an identity-bound descriptor for the transition directory.

    **Read-only.** It used to take ``create=`` and ``mkdir`` the directory
    for ``begin_recompile_table_transition``; that writer is gone, so
    every remaining caller is reading evidence a previous release left and a
    missing directory is simply ``FileNotFoundError``.
    """
    _require_identity_bound_directory_operations()
    canonical_output = Path(output_root).resolve(strict=True)
    root = _transition_root(canonical_output, dataset_name)
    try:
        relative = root.relative_to(canonical_output)
    except ValueError as exc:
        raise ValueError("Transition directory escapes output root") from exc
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(canonical_output, flags)
    try:
        try:
            for component in relative.parts:
                _validate_transition_component(component)
                child_fd = os.open(component, flags, dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = child_fd
            identity = os.fstat(directory_fd)
            if not stat.S_ISDIR(identity.st_mode):
                raise ValueError("Transition directory is not canonical")
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise ValueError("Transition directory is not canonical") from exc
        yield root, directory_fd
    finally:
        os.close(directory_fd)


def _transition_receipt_name(stem: str) -> str:
    """Return the canonical receipt entry name for an image stem."""
    if Path(stem).name != stem or stem in {"", ".", ".."}:
        raise ValueError("Transition image stem is not canonical")
    return f"{stem}.json"


def _read_regular_file_at(directory_fd: int, name: str) -> bytes:
    """Read a single-link regular file relative to a held directory."""
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Transition file is not canonical")
    file_fd = os.open(
        name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    try:
        identity = os.fstat(file_fd)
        if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
            raise ValueError("Transition file is not canonical")
        with os.fdopen(file_fd, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(file_fd)


def _transition_staged_name(
    output_root: Path,
    root: Path,
    dataset_name: str,
    stem: str,
    transition: dict[str, Any],
) -> str:
    """Return the canonical private staged entry name in a receipt."""
    relative = transition.get("prepared_path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Transition prepared path is not relative")
    candidate = output_root / relative
    if (
        root != _transition_root(output_root, dataset_name)
        or candidate.parent != root
    ):
        raise ValueError("Transition prepared payload is not canonical")
    if (
        re.fullmatch(
            rf"{re.escape(stem)}\.[0-9a-f]{{32}}\.parquet",
            candidate.name,
        )
        is None
    ):
        raise ValueError("Transition prepared payload is not canonical")
    return candidate.name


def _fingerprint_bytes(payload: bytes) -> tuple[int, str]:
    """Return an exact size/SHA-256 fingerprint for immutable bytes."""
    return len(payload), hashlib.sha256(payload).hexdigest()


def marker_claims_measurement_authority(marker_path: Path) -> bool:
    """Return whether a marker declares an embedded measurement artifact."""
    try:
        marker = json.loads(Path(marker_path).read_text(encoding="utf-8"))
        artifacts = marker.get("artifacts")
    except (OSError, AttributeError, json.JSONDecodeError):
        return False
    return isinstance(artifacts, dict) and "measurements" in artifacts


def _marker_measurement_source(
    output_root: Path, marker_path: Path
) -> Path | None:
    """Resolve a marker's in-tree measurement source, if well formed."""
    try:
        marker = json.loads(Path(marker_path).read_text(encoding="utf-8"))
        descriptor = marker["artifacts"]["measurements"]
        relative = descriptor["path"]
        if not isinstance(relative, str):
            return None
        source = (output_root / relative).resolve()
        source.relative_to(output_root)
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None
    return source


def recoverable_recompile_table_transition(
    output_dir: Path,
    dataset_name: str,
    stem: str,
    store_path: Path,
) -> bool:
    """Return whether durable evidence exactly authorizes current table bytes."""
    output_root = Path(output_dir).resolve()
    try:
        with _open_transition_directory(
            output_root,
            dataset_name,
        ) as (root, directory_fd):
            receipt_name = _transition_receipt_name(stem)
            transition = json.loads(
                _read_regular_file_at(directory_fd, receipt_name)
            )
            store = Path(store_path)
            if store.is_symlink():
                return False
            store = store.resolve(strict=True)
            canonical_store = zarr_store_path(
                output_root, dataset_name, stem
            ).resolve(strict=True)
            table = store / MEASUREMENT_TABLE_RELATIVE_PATH
            marker_path, marker, authority_version = image_authority_payload(
                output_root, dataset_name, stem
            )
            prior_table_size, prior_table_sha256 = (
                _marker_measurement_fingerprint(
                    output_root,
                    marker,
                    table,
                )
            )
            prepared_name = _transition_staged_name(
                output_root,
                root,
                dataset_name,
                stem,
                transition,
            )
            prepared_fingerprint = _fingerprint_bytes(
                _read_regular_file_at(directory_fd, prepared_name)
            )
            table_fingerprint = (table.stat().st_size, _sha256(table))
            if (
                transition.get("version") != _TRANSITION_VERSION
                or transition.get("dataset") != dataset_name
                or transition.get("image_stem") != stem
                or transition.get("work_id") != marker.get("work_id")
                or transition.get("store_path")
                != store.relative_to(output_root).as_posix()
                or transition.get("table_path")
                != table.relative_to(output_root).as_posix()
                or transition.get("marker_sha256") != _sha256(marker_path)
                or transition.get("prior_table_size") != prior_table_size
                or transition.get("prior_table_sha256") != prior_table_sha256
                or transition.get("prepared_size") != prepared_fingerprint[0]
                or transition.get("prepared_sha256") != prepared_fingerprint[1]
                or prepared_fingerprint != table_fingerprint
                or store != canonical_store
                or not _marker_allows_table_transition(
                    output_root,
                    dataset_name,
                    stem,
                    marker,
                    table,
                    expected_version=authority_version,
                )
                or not _valid_embedded_measurement_contract(store)
            ):
                return False
            return True
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def assert_no_unrecoverable_measurement_authority(
    output_dir: Path,
    dataset_names: list[str],
    accepted_sources: set[Path],
) -> None:
    """Abort rather than omit any measured store without exact recovery proof."""
    output_root = Path(output_dir).resolve()
    accepted = {Path(path).resolve() for path in accepted_sources}
    for dataset_name in dataset_names:
        zarr_dir = output_root / DIR_RESULTS / dataset_name / DIR_ZARR
        if not zarr_dir.is_dir():
            continue
        for store in sorted(zarr_dir.glob(f"*{STORE_SUFFIX}")):
            if not store.is_dir() or store.name.startswith("."):
                continue
            stem = store.name[: -len(STORE_SUFFIX)]
            table = store / MEASUREMENT_TABLE_RELATIVE_PATH
            marker_path = image_authority_path(output_root, dataset_name, stem)
            if table.resolve() in accepted:
                continue
            marker_source = _marker_measurement_source(
                output_root, marker_path
            )
            canonical_external = (
                dataset_measurements_dir(output_root, dataset_name)
                / f"{stem}.parquet"
            ).resolve()
            if (
                marker_source == canonical_external
                and marker_source in accepted
            ):
                raise RuntimeError(
                    "Legacy external measurement Parquets require --mode "
                    "migrate before recompile"
                )
            if table.is_file() or marker_claims_measurement_authority(
                marker_path
            ):
                raise RuntimeError(
                    "Cannot safely restore measurement authority for "
                    f"{dataset_name}/{stem}"
                )


def recoverable_recompile_measurement_sources(
    output_dir: Path, dataset_names: list[str]
) -> dict[Path, str]:
    """Return only tables backed by complete exact transition evidence."""
    output_root = Path(output_dir).resolve()
    sources: dict[Path, str] = {}
    for dataset_name in dataset_names:
        try:
            with _open_transition_directory(
                output_root,
                dataset_name,
            ) as (_root, directory_fd):
                receipt_names = sorted(
                    name
                    for name in os.listdir(directory_fd)
                    if name.endswith(".json")
                )
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "Cannot safely enumerate the recompile transition directory"
            ) from exc
        for receipt_name in receipt_names:
            stem = Path(receipt_name).stem
            store = zarr_store_path(output_root, dataset_name, stem)
            if recoverable_recompile_table_transition(
                output_root, dataset_name, stem, store
            ):
                sources[store / MEASUREMENT_TABLE_RELATIVE_PATH] = dataset_name
    return sources


def _marker_allows_table_transition(
    output_root: Path,
    dataset_name: str,
    stem: str,
    marker: dict[str, Any],
    table_path: Path,
    *,
    expected_version: int,
) -> bool:
    """Validate payload identity and every artifact except replaced table bytes.

    ``expected_version`` travels with the payload from
    :func:`image_authority_payload` rather than being a module constant: a
    record carries ``RECORD_VERSION`` and a legacy marker
    ``SUCCESS_MARKER_VERSION``, and checking one shape against the other's
    number returns ``False`` silently -- which is exactly how a path-only
    repoint would have disabled table-authority repair with nothing failing.
    """
    work_id = marker.get("work_id")
    if (
        marker.get("version") != expected_version
        or marker.get("dataset") != dataset_name
        or marker.get("image_stem") != stem
        or not isinstance(work_id, str)
        or not work_id
    ):
        return False
    raw_artifacts = marker.get("artifacts")
    if not isinstance(raw_artifacts, dict):
        return False
    measurement = raw_artifacts.get("measurements")
    if not isinstance(measurement, dict):
        return False
    relative = measurement.get("path")
    if not isinstance(relative, str):
        return False
    if (output_root / relative).resolve() != table_path.resolve(
        strict=True
    ) or measurement.get("kind", ARTIFACT_KIND_FILE) != ARTIFACT_KIND_FILE:
        return False
    for name, descriptor in raw_artifacts.items():
        if name == "measurements":
            continue
        if not isinstance(descriptor, dict):
            return False
        relative = descriptor.get("path")
        if not isinstance(relative, str):
            return False
        artifact = (output_root / relative).resolve()
        artifact.relative_to(output_root)
        kind = descriptor.get("kind", ARTIFACT_KIND_FILE)
        if kind == ARTIFACT_KIND_STORE:
            if not _store_artifact_matches(artifact, descriptor):
                return False
        elif kind == ARTIFACT_KIND_FILE:
            if (
                not artifact.is_file()
                or artifact.stat().st_size != descriptor.get("size")
                or _sha256(artifact) != descriptor.get("sha256")
            ):
                return False
        else:
            return False
    return True


__all__ = [
    "assert_no_unrecoverable_measurement_authority",
    "marker_claims_measurement_authority",
    "recoverable_recompile_measurement_sources",
    "recoverable_recompile_table_transition",
    "recompile_store_lock_path",
    "recompile_table_transition_path",
]
