"""Indexed authority for provenance-only migration arrays."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any, Final, Literal, Mapping, Sequence

from phenotypic.sdk_ import CommitGuard, atomic_write_json
from phenotypic.sdk_.ngff_ import STORE_ROOT_JSON

from ._cli_migrate_manifest import validate_migration_generation
from ._cli_migrate_provenance import (
    ProvenanceMigrationTarget,
    ProvenanceUpgradeResult,
)


_SCHEMA_VERSION: Final = 1
_RECORDS_MAGIC: Final = b"PHTPROV1"
_RECORDS_HEADER: Final = _RECORDS_MAGIC + struct.pack(">I", _SCHEMA_VERSION)
_MANIFEST_FILENAME: Final = "provenance_manifest.json"
_RECORDS_FILENAME: Final = "provenance_manifest.records"
_OFFSETS_FILENAME: Final = "provenance_manifest.offsets"
_TASK_TYPE: Final = "provenance_store"
_TASK_FIELDS: Final = frozenset(
    {
        "schema_version",
        "task_type",
        "generation",
        "inventory_digest",
        "index",
        "store_path",
        "merkle_proof",
    }
)


@dataclass(frozen=True)
class ProvenanceMigrationTask:
    """One root-only store upgrade selected by a migration manifest."""

    index: int
    store_path: Path
    task_type: Literal["provenance_store"] = "provenance_store"


@dataclass(frozen=True)
class ProvenanceMigrationManifest:
    """Caller-bound metadata for one indexed provenance inventory."""

    schema_version: int
    generation: str
    target_kind: Literal["direct_store", "process_tree"]
    target_root: Path
    control_root: Path
    task_count: int
    inventory_digest: str
    records_path: Path
    offsets_path: Path


@dataclass(frozen=True)
class ProvenanceMigrationSeal:
    """Terminal barrier evidence for all provenance store statuses."""

    generation: str
    manifest_digest: str
    ordered_status_digest: str
    clean: bool
    upgraded: int
    failures: tuple[tuple[Path, str], ...]
    seal_path: Path


def _canonical_identity(task: ProvenanceMigrationTask) -> bytes:
    """Return stable leaf bytes for one ordered store identity."""
    return json.dumps(
        {
            "index": task.index,
            "store_path": str(task.store_path),
            "task_type": task.task_type,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _leaf_hash(task: ProvenanceMigrationTask) -> bytes:
    return hashlib.sha256(b"leaf\0" + _canonical_identity(task)).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"node\0" + left + right).digest()


def _merkle_root_and_proofs(
    tasks: Sequence[ProvenanceMigrationTask],
) -> tuple[bytes, dict[int, list[dict[str, str]]]]:
    """Return one ordered inventory root and an inclusion proof per task."""
    if not tasks:
        return hashlib.sha256(b"empty-provenance-inventory").digest(), {}
    levels: list[list[bytes]] = [[_leaf_hash(task) for task in tasks]]
    while len(levels[-1]) > 1:
        level = levels[-1]
        levels.append(
            [
                _node_hash(
                    level[index],
                    level[index + 1] if index + 1 < len(level) else level[index],
                )
                for index in range(0, len(level), 2)
            ]
        )
    proofs: dict[int, list[dict[str, str]]] = {}
    for task in tasks:
        proof: list[dict[str, str]] = []
        position = task.index
        for level in levels[:-1]:
            if position % 2:
                sibling = position - 1
                side = "left"
            else:
                sibling = position + 1 if position + 1 < len(level) else position
                side = "right"
            proof.append({"side": side, "hash": level[sibling].hex()})
            position //= 2
        proofs[task.index] = proof
    return levels[-1][0], proofs


def _normalized_task(
    task: ProvenanceMigrationTask,
    target: ProvenanceMigrationTarget,
) -> ProvenanceMigrationTask:
    """Bind one store identity to the classified target without descent."""
    if not isinstance(task.index, int) or isinstance(task.index, bool) or task.index < 0:
        raise ValueError("provenance migration task index must be non-negative")
    store = Path(task.store_path).resolve()
    root = target.root.resolve()
    if target.kind == "direct_store":
        if store != root:
            raise ValueError("direct-store provenance task does not match target")
    elif target.kind == "process_tree":
        if not store.is_relative_to(root):
            raise ValueError("process provenance task escapes target root")
    else:
        raise ValueError("full runs do not use provenance-only manifests")
    if store.is_symlink() or not (store / STORE_ROOT_JSON).is_file():
        raise ValueError(f"invalid provenance migration store: {store}")
    return ProvenanceMigrationTask(index=task.index, store_path=store)


def write_provenance_migration_manifest(
    target: ProvenanceMigrationTarget,
    *,
    generation: str,
    control_root: Path,
) -> ProvenanceMigrationManifest:
    """Write framed direct-access store tasks and their versioned header."""
    generation = validate_migration_generation(generation)
    control = Path(control_root).resolve()
    control.mkdir(parents=True, exist_ok=True)
    tasks = tuple(
        _normalized_task(ProvenanceMigrationTask(index, store), target)
        for index, store in enumerate(target.stores)
    )
    if not tasks:
        raise ValueError("provenance-only migration requires at least one store")
    if len({task.store_path for task in tasks}) != len(tasks):
        raise ValueError("provenance migration store identities must be unique")
    root, proofs = _merkle_root_and_proofs(tasks)
    records_path = control / _RECORDS_FILENAME
    offsets_path = control / _OFFSETS_FILENAME
    offsets: list[int] = []
    with records_path.open("wb") as records:
        records.write(_RECORDS_HEADER)
        for task in tasks:
            payload = json.dumps(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "task_type": task.task_type,
                    "generation": generation,
                    "inventory_digest": root.hex(),
                    "index": task.index,
                    "store_path": str(task.store_path),
                    "merkle_proof": proofs[task.index],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            offsets.append(records.tell())
            records.write(struct.pack(">Q", len(payload)))
            records.write(payload)
            records.write(hashlib.sha256(payload).digest())
    with offsets_path.open("wb") as offsets_file:
        for offset in offsets:
            offsets_file.write(struct.pack(">Q", offset))
    if target.kind == "direct_store":
        target_kind: Literal["direct_store", "process_tree"] = "direct_store"
    elif target.kind == "process_tree":
        target_kind = "process_tree"
    else:
        raise ValueError("full runs do not use provenance-only manifests")
    manifest = ProvenanceMigrationManifest(
        schema_version=_SCHEMA_VERSION,
        generation=generation,
        target_kind=target_kind,
        target_root=target.root.resolve(),
        control_root=control,
        task_count=len(tasks),
        inventory_digest=root.hex(),
        records_path=records_path,
        offsets_path=offsets_path,
    )
    atomic_write_json(
        control / _MANIFEST_FILENAME,
        {
            "schema_version": manifest.schema_version,
            "task_type": _TASK_TYPE,
            "generation": manifest.generation,
            "target_kind": manifest.target_kind,
            "target_root": str(manifest.target_root),
            "control_root": str(manifest.control_root),
            "task_count": manifest.task_count,
            "inventory_digest": manifest.inventory_digest,
            "records_path": str(manifest.records_path),
            "offsets_path": str(manifest.offsets_path),
        },
    )
    return manifest


def _read_manifest(
    manifest_path: Path,
    *,
    expected_target_root: Path,
    expected_control_root: Path,
) -> ProvenanceMigrationManifest:
    """Load an exact caller-bound provenance manifest header."""
    control = Path(expected_control_root).resolve()
    target_root = Path(expected_target_root).resolve()
    path = Path(manifest_path).resolve()
    if path != control / _MANIFEST_FILENAME or path.is_symlink():
        raise ValueError("provenance manifest has wrong control root")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid provenance migration manifest") from exc
    fields = {
        "schema_version", "task_type", "generation", "target_kind",
        "target_root", "control_root", "task_count", "inventory_digest",
        "records_path", "offsets_path",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != fields
        or raw.get("schema_version") != _SCHEMA_VERSION
        or raw.get("task_type") != _TASK_TYPE
        or raw.get("target_kind") not in {"direct_store", "process_tree"}
        or Path(str(raw.get("target_root"))).resolve() != target_root
        or Path(str(raw.get("control_root"))).resolve() != control
    ):
        raise ValueError("invalid provenance migration manifest schema")
    generation = validate_migration_generation(raw.get("generation"))
    count = raw.get("task_count")
    digest = raw.get("inventory_digest")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 1
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise ValueError("invalid provenance migration manifest identity")
    records_path = Path(str(raw.get("records_path"))).resolve()
    offsets_path = Path(str(raw.get("offsets_path"))).resolve()
    if records_path.parent != control or offsets_path.parent != control:
        raise ValueError("provenance manifest artifacts escape control root")
    return ProvenanceMigrationManifest(
        schema_version=_SCHEMA_VERSION,
        generation=generation,
        target_kind=raw["target_kind"],
        target_root=target_root,
        control_root=control,
        task_count=count,
        inventory_digest=digest,
        records_path=records_path,
        offsets_path=offsets_path,
    )


def read_provenance_migration_task(
    manifest_path: Path,
    index: int,
    *,
    expected_target_root: Path,
    expected_control_root: Path,
) -> ProvenanceMigrationTask:
    """Seek directly to one checksum- and Merkle-verified store task."""
    manifest = _read_manifest(
        manifest_path,
        expected_target_root=expected_target_root,
        expected_control_root=expected_control_root,
    )
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < manifest.task_count:
        raise IndexError(f"provenance migration task index out of range: {index}")
    try:
        with manifest.offsets_path.open("rb") as offsets:
            if os.fstat(offsets.fileno()).st_size != manifest.task_count * 8:
                raise ValueError("provenance offset file has invalid alignment")
            offsets.seek(index * 8)
            offset_bytes = offsets.read(8)
        offset = struct.unpack(">Q", offset_bytes)[0]
        with manifest.records_path.open("rb") as records:
            if records.read(len(_RECORDS_HEADER)) != _RECORDS_HEADER:
                raise ValueError("provenance records have invalid version")
            records.seek(offset)
            length = struct.unpack(">Q", records.read(8))[0]
            payload = records.read(length)
            checksum = records.read(32)
    except (OSError, struct.error) as exc:
        raise ValueError("cannot read provenance migration record") from exc
    if len(payload) != length or hashlib.sha256(payload).digest() != checksum:
        raise ValueError("provenance migration record checksum mismatch")
    try:
        raw: Any = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid provenance migration record") from exc
    if (
        not isinstance(raw, Mapping)
        or set(raw) != _TASK_FIELDS
        or raw.get("schema_version") != _SCHEMA_VERSION
        or raw.get("task_type") != _TASK_TYPE
        or raw.get("generation") != manifest.generation
        or raw.get("inventory_digest") != manifest.inventory_digest
        or raw.get("index") != index
        or not isinstance(raw.get("store_path"), str)
    ):
        raise ValueError("invalid provenance migration record schema")
    task = ProvenanceMigrationTask(index=index, store_path=Path(raw["store_path"]))
    target = ProvenanceMigrationTarget(
        manifest.target_kind, manifest.target_root, (task.store_path,)
    )
    task = _normalized_task(task, target)
    proof = raw.get("merkle_proof")
    if not isinstance(proof, list):
        raise ValueError("invalid provenance migration Merkle proof")
    computed = _leaf_hash(task)
    for step in proof:
        if (
            not isinstance(step, Mapping)
            or set(step) != {"side", "hash"}
            or step.get("side") not in {"left", "right"}
            or not isinstance(step.get("hash"), str)
            or len(step["hash"]) != 64
        ):
            raise ValueError("invalid provenance migration Merkle proof")
        try:
            sibling = bytes.fromhex(step["hash"])
        except ValueError as exc:
            raise ValueError("invalid provenance migration Merkle proof") from exc
        computed = (
            _node_hash(sibling, computed)
            if step["side"] == "left"
            else _node_hash(computed, sibling)
        )
    if computed.hex() != manifest.inventory_digest:
        raise ValueError("provenance task does not match inventory digest")
    return task


def provenance_worker_status_path(
    control_root: Path, generation: str, index: int
) -> Path:
    """Return the canonical typed status path for one store index."""
    generation = validate_migration_generation(generation)
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("provenance worker index must be non-negative")
    return Path(control_root).resolve() / "provenance_status" / generation / f"{index}.json"


def provenance_seal_path(control_root: Path, generation: str) -> Path:
    """Return the canonical provenance barrier evidence path."""
    generation = validate_migration_generation(generation)
    return Path(control_root).resolve() / "provenance_seal" / generation / "seal.json"


def publish_provenance_worker_status(
    manifest_path: Path,
    *,
    expected_target_root: Path,
    expected_control_root: Path,
    generation: str,
    index: int,
    result: ProvenanceUpgradeResult | None,
    reason: str | None = None,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Publish exact success or failure evidence for one indexed store."""
    manifest = _read_manifest(
        manifest_path,
        expected_target_root=expected_target_root,
        expected_control_root=expected_control_root,
    )
    if manifest.generation != generation:
        raise ValueError("provenance status generation does not match manifest")
    task = read_provenance_migration_task(
        manifest_path,
        index,
        expected_target_root=expected_target_root,
        expected_control_root=expected_control_root,
    )
    if result is not None and result.store_path.resolve() != task.store_path:
        raise ValueError("provenance result store does not match manifest")
    if (result is None) == (reason is None):
        raise ValueError("provenance status requires exactly one result or reason")
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "task_type": _TASK_TYPE,
        "state": "complete" if result is not None else "failed",
        "generation": generation,
        "manifest_digest": manifest.inventory_digest,
        "index": index,
        "store_path": str(task.store_path),
        "schema_before": None if result is None else result.schema_before,
        "upgraded": False if result is None else result.upgraded,
        "reason": reason,
    }
    path = provenance_worker_status_path(expected_control_root, generation, index)
    atomic_write_json(path, payload, commit_guard=commit_guard)
    return path


_STATUS_FIELDS: Final = frozenset(
    {
        "schema_version", "task_type", "state", "generation",
        "manifest_digest", "index", "store_path", "schema_before",
        "upgraded", "reason",
    }
)


def _ordered_status_digest(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def seal_provenance_migration(
    manifest_path: Path,
    *,
    expected_target_root: Path,
    expected_control_root: Path,
    generation: str,
    commit_guard: CommitGuard | None = None,
) -> ProvenanceMigrationSeal:
    """Validate every typed store result and publish one exact barrier."""
    manifest = _read_manifest(
        manifest_path,
        expected_target_root=expected_target_root,
        expected_control_root=expected_control_root,
    )
    if manifest.generation != generation:
        raise ValueError("provenance seal generation does not match manifest")
    status_dir = provenance_worker_status_path(
        expected_control_root, generation, 0
    ).parent
    paths = sorted(status_dir.glob("*.json")) if status_dir.is_dir() else []
    failures: list[tuple[Path, str]] = []
    upgraded = 0
    if len(paths) != manifest.task_count:
        failures.append(
            (manifest.target_root, "provenance status count does not match manifest")
        )
    for index in range(manifest.task_count):
        task = read_provenance_migration_task(
            manifest_path,
            index,
            expected_target_root=expected_target_root,
            expected_control_root=expected_control_root,
        )
        path = provenance_worker_status_path(
            expected_control_root, generation, index
        )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            failures.append((task.store_path, "provenance status is missing or corrupt"))
            continue
        valid_common = (
            isinstance(raw, Mapping)
            and set(raw) == _STATUS_FIELDS
            and raw.get("schema_version") == _SCHEMA_VERSION
            and raw.get("task_type") == _TASK_TYPE
            and raw.get("generation") == generation
            and raw.get("manifest_digest") == manifest.inventory_digest
            and raw.get("index") == index
            and raw.get("store_path") == str(task.store_path)
            and isinstance(raw.get("upgraded"), bool)
        )
        if not valid_common:
            failures.append((task.store_path, "provenance status has invalid schema"))
            continue
        if raw.get("state") == "complete" and raw.get("reason") is None:
            schema_before = raw.get("schema_before")
            valid_schema_before = schema_before is None or (
                isinstance(schema_before, int)
                and not isinstance(schema_before, bool)
                and schema_before in {1, 2}
            )
            if not valid_schema_before or raw["upgraded"] != (
                schema_before == 1
            ):
                failures.append((task.store_path, "provenance status has invalid prior schema"))
                continue
            upgraded += int(raw["upgraded"])
        elif raw.get("state") == "failed" and isinstance(raw.get("reason"), str) and raw["reason"]:
            failures.append((task.store_path, raw["reason"]))
        else:
            failures.append((task.store_path, "provenance status is not terminal"))
    try:
        ordered_digest = _ordered_status_digest(paths)
    except OSError as exc:
        failures.append((manifest.target_root, f"cannot digest provenance statuses: {exc}"))
        ordered_digest = hashlib.sha256(b"").hexdigest()
    seal_path = provenance_seal_path(expected_control_root, generation)
    atomic_write_json(
        seal_path,
        {
            "schema_version": _SCHEMA_VERSION,
            "task_type": "provenance_seal",
            "generation": generation,
            "manifest_digest": manifest.inventory_digest,
            "ordered_status_digest": ordered_digest,
            "clean": not failures,
            "upgraded": upgraded,
            "failures": [
                {"store_path": str(path), "reason": reason}
                for path, reason in failures
            ],
        },
        commit_guard=commit_guard,
    )
    return ProvenanceMigrationSeal(
        generation=generation,
        manifest_digest=manifest.inventory_digest,
        ordered_status_digest=ordered_digest,
        clean=not failures,
        upgraded=upgraded,
        failures=tuple(failures),
        seal_path=seal_path,
    )


def read_provenance_migration_seal(
    manifest_path: Path,
    *,
    expected_target_root: Path,
    expected_control_root: Path,
    generation: str,
) -> ProvenanceMigrationSeal:
    """Load a seal only when its current ordered statuses still match."""
    manifest = _read_manifest(
        manifest_path,
        expected_target_root=expected_target_root,
        expected_control_root=expected_control_root,
    )
    path = provenance_seal_path(expected_control_root, generation)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("provenance migration seal is missing or corrupt") from exc
    fields = {
        "schema_version", "task_type", "generation", "manifest_digest",
        "ordered_status_digest", "clean", "upgraded", "failures",
    }
    upgraded = raw.get("upgraded") if isinstance(raw, Mapping) else None
    if (
        not isinstance(raw, Mapping)
        or set(raw) != fields
        or raw.get("schema_version") != _SCHEMA_VERSION
        or raw.get("task_type") != "provenance_seal"
        or raw.get("generation") != generation
        or raw.get("manifest_digest") != manifest.inventory_digest
        or not isinstance(raw.get("clean"), bool)
        or not isinstance(upgraded, int)
        or isinstance(upgraded, bool)
        or upgraded < 0
        or upgraded > manifest.task_count
        or not isinstance(raw.get("failures"), list)
    ):
        raise ValueError("provenance migration seal has invalid schema")
    status_dir = provenance_worker_status_path(
        expected_control_root, generation, 0
    ).parent
    paths = sorted(status_dir.glob("*.json")) if status_dir.is_dir() else []
    if raw.get("ordered_status_digest") != _ordered_status_digest(paths):
        raise ValueError("provenance migration seal is stale")
    failures: list[tuple[Path, str]] = []
    for failure in raw["failures"]:
        if (
            not isinstance(failure, Mapping)
            or set(failure) != {"store_path", "reason"}
            or not isinstance(failure.get("store_path"), str)
            or not isinstance(failure.get("reason"), str)
            or not failure["reason"]
        ):
            raise ValueError("provenance migration seal has invalid failures")
        failures.append((Path(failure["store_path"]), failure["reason"]))
    if raw["clean"] != (not failures):
        raise ValueError("provenance migration seal has contradictory status")
    return ProvenanceMigrationSeal(
        generation=generation,
        manifest_digest=manifest.inventory_digest,
        ordered_status_digest=raw["ordered_status_digest"],
        clean=raw["clean"],
        upgraded=raw["upgraded"],
        failures=tuple(failures),
        seal_path=path,
    )


__all__ = [
    "ProvenanceMigrationManifest",
    "ProvenanceMigrationSeal",
    "ProvenanceMigrationTask",
    "provenance_seal_path",
    "provenance_worker_status_path",
    "publish_provenance_worker_status",
    "read_provenance_migration_seal",
    "read_provenance_migration_task",
    "seal_provenance_migration",
    "write_provenance_migration_manifest",
]
