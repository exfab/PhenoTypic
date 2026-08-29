"""Crash-recovery evidence and locking for per-store recompile mutation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from uuid import uuid4


from phenotypic.sdk_ import (
    DIR_RESULTS,
    DIR_ZARR,
    MEASUREMENT_TABLE_RELATIVE_PATH,
    STORE_SUFFIX,
    CommitGuard,
    PreparedEmbeddedMeasurementTable,
    atomic_write_bytes,
    image_completion_marker_path,
    progress_dir,
    zarr_store_path,
)
from phenotypic.sdk_._measurement_tables import (
    _valid_embedded_measurement_contract,
    _write_validated_parquet,
)

from ._cli_completion import (
    ARTIFACT_KIND_FILE,
    ARTIFACT_KIND_STORE,
    SUCCESS_MARKER_VERSION,
    _sha256,
    _store_artifact_matches,
    valid_image_success,
)

_TRANSITION_VERSION = 1
_TRANSITION_DIR = "table-transitions"


def recompile_store_lock_path(
    output_dir: Path, dataset_name: str, stem: str
) -> Path:
    """Return the lock shared by canonical recompile mutations for one store."""
    return image_completion_marker_path(
        output_dir, dataset_name, stem
    ).with_suffix(".recompile-store.lock")


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


def _fsync_recompile_directory(path: Path) -> None:
    """Durably commit directory-entry changes without following the directory."""
    _require_identity_bound_directory_operations()
    directory_fd = os.open(
        Path(path),
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        identity = os.fstat(directory_fd)
        if not stat.S_ISDIR(identity.st_mode):
            raise ValueError("Recompile transaction directory is invalid")
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _validate_transition_component(component: str) -> None:
    """Reject non-canonical transition path components."""
    if component in {"", ".", ".."} or Path(component).name != component:
        raise ValueError("Transition directory is not canonical")


@contextmanager
def _open_transition_directory(
    output_root: Path,
    dataset_name: str,
    *,
    create: bool,
) -> Iterator[tuple[Path, int]]:
    """Hold an identity-bound descriptor for the transition directory."""
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
                try:
                    child_fd = os.open(component, flags, dir_fd=directory_fd)
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=directory_fd)
                    except FileExistsError:
                        pass
                    else:
                        os.fsync(directory_fd)
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


def _write_exclusive_file_at(
    directory_fd: int,
    name: str,
    payload: bytes,
) -> None:
    """Create one private regular file relative to a held directory."""
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Transition file is not canonical")
    file_fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        identity = os.fstat(file_fd)
        if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
            raise ValueError("Transition file is not canonical")
        remaining = memoryview(payload)
        while remaining:
            written = os.write(file_fd, remaining)
            if written <= 0:
                raise OSError("Unable to write recompile transition file")
            remaining = remaining[written:]
        os.fsync(file_fd)
    except BaseException:
        try:
            os.unlink(name, dir_fd=directory_fd)
        except OSError:
            pass
        raise
    finally:
        os.close(file_fd)


def _write_json_at(
    directory_fd: int,
    receipt_name: str,
    payload: dict[str, Any],
) -> None:
    """Atomically publish a receipt inside a held transition directory."""
    try:
        os.stat(receipt_name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        _read_regular_file_at(directory_fd, receipt_name)
    temporary_name = f".{receipt_name}.{uuid4().hex}.tmp"
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    _write_exclusive_file_at(directory_fd, temporary_name, encoded)
    try:
        os.rename(
            temporary_name,
            receipt_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except OSError:
            pass
        raise


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


def _cleanup_orphan_staging_payloads_at(
    directory_fd: int,
    stem: str,
    *,
    keep_name: str,
) -> None:
    """Remove canonical orphan entries relative to the held directory only."""
    _read_regular_file_at(directory_fd, keep_name)
    pattern = re.compile(rf"{re.escape(stem)}\.[0-9a-f]{{32}}\.parquet")
    for name in os.listdir(directory_fd):
        if name == keep_name or pattern.fullmatch(name) is None:
            continue
        try:
            _read_regular_file_at(directory_fd, name)
        except (OSError, ValueError):
            continue
        os.unlink(name, dir_fd=directory_fd)
    os.fsync(directory_fd)


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


def begin_recompile_table_transition(
    output_dir: Path,
    dataset_name: str,
    stem: str,
    store_path: Path,
    prepared: PreparedEmbeddedMeasurementTable,
) -> Path:
    """Publish exact intended-table evidence before replacing canonical bytes."""
    output_root = Path(output_dir).resolve()
    store = Path(store_path).resolve(strict=True)
    if store != zarr_store_path(output_root, dataset_name, stem).resolve(
        strict=True
    ):
        raise ValueError("Recompile transition store is not canonical")
    marker_path = image_completion_marker_path(output_root, dataset_name, stem)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    prior_table_size, prior_table_sha256 = _marker_measurement_fingerprint(
        output_root,
        marker,
        store / MEASUREMENT_TABLE_RELATIVE_PATH,
    )
    work_id = str(marker["work_id"])
    marker_authorized = valid_image_success(
        output_root,
        dataset=dataset_name,
        image_stem=stem,
        work_id=work_id,
    )
    transition_authorized = recoverable_recompile_table_transition(
        output_root, dataset_name, stem, store
    )
    if not marker_authorized and not transition_authorized:
        raise RuntimeError(
            "Cannot replace an embedded table without marker or transition authority"
        )
    receipt_name = _transition_receipt_name(stem)
    staged_name = f"{stem}.{uuid4().hex}.parquet"
    with _open_transition_directory(
        output_root,
        dataset_name,
        create=True,
    ) as (root, directory_fd):
        with TemporaryDirectory(prefix="phenotypic-transition-") as temporary:
            prepared_path = Path(temporary) / "table.parquet"
            _write_validated_parquet(prepared_path, prepared)
            prepared_bytes = prepared_path.read_bytes()
        _write_exclusive_file_at(directory_fd, staged_name, prepared_bytes)
        os.fsync(directory_fd)
        prepared_size, prepared_sha256 = _fingerprint_bytes(prepared_bytes)
        transition = {
            "version": _TRANSITION_VERSION,
            "dataset": dataset_name,
            "image_stem": stem,
            "work_id": work_id,
            "store_path": store.relative_to(output_root).as_posix(),
            "table_path": (store / MEASUREMENT_TABLE_RELATIVE_PATH)
            .relative_to(output_root)
            .as_posix(),
            "marker_sha256": _sha256(marker_path),
            "prior_table_size": prior_table_size,
            "prior_table_sha256": prior_table_sha256,
            "prepared_path": (root / staged_name)
            .relative_to(output_root)
            .as_posix(),
            "prepared_size": prepared_size,
            "prepared_sha256": prepared_sha256,
        }
        _write_json_at(directory_fd, receipt_name, transition)
        _cleanup_orphan_staging_payloads_at(
            directory_fd,
            stem,
            keep_name=staged_name,
        )
    return root / staged_name


def promote_recompile_table_transition(
    output_dir: Path,
    dataset_name: str,
    stem: str,
    store_path: Path,
    staged_path: Path,
    *,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Promote only the exact journaled staged bytes to the canonical table."""
    output_root = Path(output_dir).resolve()
    store = Path(store_path).resolve(strict=True)
    canonical_store = zarr_store_path(output_root, dataset_name, stem).resolve(
        strict=True
    )
    if store != canonical_store:
        raise RuntimeError("Transition store is not canonical")
    table = store / MEASUREMENT_TABLE_RELATIVE_PATH
    try:
        with _open_transition_directory(
            output_root,
            dataset_name,
            create=False,
        ) as (root, directory_fd):
            receipt_name = _transition_receipt_name(stem)
            transition = json.loads(
                _read_regular_file_at(directory_fd, receipt_name)
            )
            marker_path = image_completion_marker_path(
                output_root, dataset_name, stem
            )
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            prior_size, prior_sha256 = _marker_measurement_fingerprint(
                output_root,
                marker,
                table,
            )
            staged_name = _transition_staged_name(
                output_root,
                root,
                dataset_name,
                stem,
                transition,
            )
            staged_bytes = _read_regular_file_at(directory_fd, staged_name)
            intended_size = transition.get("prepared_size")
            intended_sha256 = transition.get("prepared_sha256")
            if (
                root / staged_name != Path(staged_path)
                or transition.get("version") != _TRANSITION_VERSION
                or transition.get("dataset") != dataset_name
                or transition.get("image_stem") != stem
                or transition.get("work_id") != marker.get("work_id")
                or transition.get("store_path")
                != store.relative_to(output_root).as_posix()
                or transition.get("table_path")
                != table.relative_to(output_root).as_posix()
                or transition.get("marker_sha256") != _sha256(marker_path)
                or transition.get("prior_table_size") != prior_size
                or transition.get("prior_table_sha256") != prior_sha256
                or not isinstance(intended_size, int)
                or intended_size < 0
                or not isinstance(intended_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", intended_sha256) is None
                or _fingerprint_bytes(staged_bytes)
                != (intended_size, intended_sha256)
                or not _marker_allows_table_transition(
                    output_root, dataset_name, stem, marker, table
                )
                or not _valid_embedded_measurement_contract(store)
            ):
                raise RuntimeError("Recompile transition evidence is invalid")

            def _fingerprint(path: Path) -> tuple[int, str]:
                return path.stat().st_size, _sha256(path)

            current = _fingerprint(table)
            intended = (intended_size, intended_sha256)
            prior = (prior_size, prior_sha256)
            if current == intended:
                _fsync_recompile_directory(table.parent)
                return table
            if current != prior:
                raise RuntimeError(
                    "Canonical table matches neither prior nor intended transition"
                )

            def _validate_immediately_before_replace() -> None:
                staged_fingerprint = _fingerprint_bytes(
                    _read_regular_file_at(directory_fd, staged_name)
                )
                if (
                    staged_fingerprint != intended
                    or _fingerprint(table) != prior
                ):
                    raise RuntimeError(
                        "Recompile transition changed before table promotion"
                    )

            atomic_write_bytes(
                table,
                staged_bytes,
                pre_replace=_validate_immediately_before_replace,
                commit_guard=commit_guard,
            )
            _fsync_recompile_directory(table.parent)
            if _fingerprint(
                table
            ) != intended or not _valid_embedded_measurement_contract(store):
                raise RuntimeError("Promoted embedded table failed validation")
            return table
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError("Recompile transition evidence is invalid") from exc


def clear_recompile_table_transition(
    output_dir: Path, dataset_name: str, stem: str
) -> None:
    """Remove transition entries relative to their identity-bound directory."""
    output_root = Path(output_dir).resolve()
    mutation_started = False
    try:
        with _open_transition_directory(
            output_root,
            dataset_name,
            create=False,
        ) as (root, directory_fd):
            receipt_name = _transition_receipt_name(stem)
            try:
                transition = json.loads(
                    _read_regular_file_at(directory_fd, receipt_name)
                )
            except (OSError, ValueError, json.JSONDecodeError):
                return
            try:
                staged_name = _transition_staged_name(
                    output_root,
                    root,
                    dataset_name,
                    stem,
                    transition,
                )
                _read_regular_file_at(directory_fd, staged_name)
                os.unlink(staged_name, dir_fd=directory_fd)
            except (OSError, ValueError):
                pass
            else:
                mutation_started = True
            os.unlink(receipt_name, dir_fd=directory_fd)
            mutation_started = True
            os.fsync(directory_fd)
    except (OSError, ValueError):
        if mutation_started:
            raise


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
            create=False,
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
            marker_path = image_completion_marker_path(
                output_root, dataset_name, stem
            )
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
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
                    output_root, dataset_name, stem, marker, table
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
            marker_path = image_completion_marker_path(
                output_root, dataset_name, stem
            )
            if table.resolve() in accepted:
                continue
            marker_source = _marker_measurement_source(
                output_root, marker_path
            )
            if marker_source in accepted:
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
                create=False,
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
) -> bool:
    """Validate marker identity and every artifact except replaced table bytes."""
    work_id = marker.get("work_id")
    if (
        marker.get("version") != SUCCESS_MARKER_VERSION
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
    "begin_recompile_table_transition",
    "clear_recompile_table_transition",
    "marker_claims_measurement_authority",
    "promote_recompile_table_transition",
    "recoverable_recompile_measurement_sources",
    "recoverable_recompile_table_transition",
    "recompile_store_lock_path",
    "recompile_table_transition_path",
]
