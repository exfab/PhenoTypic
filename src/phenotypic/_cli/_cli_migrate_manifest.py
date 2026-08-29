"""Indexed, tamper-evident inventory for resumable output migration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any, Final, Mapping, Sequence

from phenotypic.sdk_ import (
    CommitGuard,
    STORE_SUFFIX,
    atomic_write_json,
    dataset_measurements_dir,
    dataset_overlays_dir,
    deliverables_dir,
    image_completion_marker_path,
    phenotypic_cache_dir,
    results_dir,
    store_stem,
    zarr_store_path,
)


_SCHEMA_VERSION: Final = 2
_LEGACY_SCHEMA_VERSION: Final = 1
_RECORDS_MAGIC: Final = b"PHTMIGR1"
_RECORDS_HEADER: Final = _RECORDS_MAGIC + struct.pack(">I", _SCHEMA_VERSION)
_MANIFEST_FILENAME: Final = "migration_manifest.json"
_RECORDS_FILENAME: Final = "migration_manifest.records"
_OFFSETS_FILENAME: Final = "migration_manifest.offsets"
_TASK_FIELDS: Final = frozenset(
    {
        "generation",
        "hdf_path",
        "index",
        "marker_path",
        "measurement_path",
        "merkle_proof",
        "overlay_path",
        "store_path",
        "stem",
        "dataset",
    }
)


@dataclass(frozen=True)
class MigrationImageTask:
    """One migration target and its canonical artifact paths."""

    index: int
    dataset: str
    stem: str
    hdf_path: Path | None
    store_path: Path
    measurement_path: Path | None
    overlay_path: Path
    marker_path: Path


@dataclass(frozen=True)
class MigrationManifest:
    """Metadata locating one immutable indexed migration inventory."""

    schema_version: int
    generation: str
    scientific_output: Path
    task_count: int
    inventory_digest: str
    records_path: Path
    offsets_path: Path
    output_root: Path
    control_root: Path


@dataclass(frozen=True)
class MigrationImageSeal:
    """Terminal evidence that every manifest image has current authority."""

    generation: str
    manifest_digest: str
    ordered_status_digest: str
    metadata_terminal_digest: str
    clean: bool
    failures: tuple[str, ...]
    seal_path: Path

    @property
    def status_digest(self) -> str:
        """Return the ordered task-status digest under its shorter name."""
        return self.ordered_status_digest


@dataclass(frozen=True)
class MigrationReclaimSeal:
    """Terminal evidence that every requested source reclaim is complete."""

    generation: str
    manifest_digest: str
    ordered_reclaim_status_digest: str
    deletion_requested: bool
    clean: bool
    failures: tuple[str, ...]
    seal_path: Path

    @property
    def status_digest(self) -> str:
        """Return the ordered reclaim-status digest under its shorter name."""
        return self.ordered_reclaim_status_digest


def _iter_hdf_candidates(dataset_dir: Path) -> tuple[Path, ...]:
    """Return direct legacy HDF candidates for one dataset."""
    directory = dataset_dir / "hdf"
    return tuple(sorted(directory.glob("*.h5"))) if directory.is_dir() else ()


def _iter_store_candidates(dataset_dir: Path) -> tuple[Path, ...]:
    """Return direct OME-Zarr store candidates for one dataset."""
    directory = dataset_dir / "zarr"
    return (
        tuple(sorted(directory.glob(f"*{STORE_SUFFIX}"))) if directory.is_dir() else ()
    )


def _iter_measurement_candidates(dataset_dir: Path) -> tuple[Path, ...]:
    """Return direct external per-image table candidates for one dataset."""
    directory = dataset_dir / "measurements"
    return (
        tuple(sorted(directory.glob("*.parquet"))) if directory.is_dir() else ()
    )


def _candidate_stem(path: Path) -> str:
    """Return the canonical identity stem for one discovered artifact."""
    if path.name.endswith(STORE_SUFFIX):
        return store_stem(path)
    return path.stem


def _checked_path(path: Path, output_root: Path) -> Path:
    """Resolve a candidate only when every component is inside the run tree."""
    root = Path(output_root).absolute()
    candidate = Path(path).absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"migration candidate escapes output directory: {candidate}") from exc
    if root.is_symlink():
        raise ValueError(f"migration candidate is a symlink: {output_root}")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"migration candidate is a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"migration candidate escapes output directory: {candidate}")
    return resolved


def _safe_identity_component(value: object, name: str) -> str:
    """Return one dataset/stem component after rejecting path syntax."""
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or (os.sep and os.sep in value)
        or (os.altsep is not None and os.altsep in value)
    ):
        raise ValueError(f"migration {name} must be a safe path component")
    return value


def _safe_generation(value: object) -> str:
    """Return one generation after applying the path-component contract."""
    return _safe_identity_component(value, "generation")


def validate_migration_generation(value: object) -> str:
    """Return a generation safe to use as one control-tree component."""
    return _safe_generation(value)


def _migration_generation_dir(control_root: Path, generation: str) -> Path:
    """Return the private authority root for one migration generation."""
    return Path(control_root).absolute() / "migration_generations" / _safe_generation(
        generation
    )


def migration_task_status_path(
    control_root: Path, generation: str, index: int
) -> Path:
    """Return the canonical image-task status path for one manifest index."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("migration status index must be a non-negative integer")
    return (
        _migration_generation_dir(control_root, generation)
        / "image_status"
        / f"{index:08d}.json"
    )


def migration_image_seal_path(control_root: Path, generation: str) -> Path:
    """Return the canonical image-stage seal path for one generation."""
    return _migration_generation_dir(control_root, generation) / "image_seal.json"


def migration_reclaim_status_path(
    control_root: Path, generation: str, index: int
) -> Path:
    """Return the canonical reclaim status path for one manifest index."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("migration reclaim index must be a non-negative integer")
    return (
        _migration_generation_dir(control_root, generation)
        / "reclaim_status"
        / f"{index:08d}.json"
    )


def migration_reclaim_seal_path(control_root: Path, generation: str) -> Path:
    """Return the canonical reclaim-stage seal path for one generation."""
    return _migration_generation_dir(control_root, generation) / "reclaim_seal.json"


def _sha256_bytes(data: bytes) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def _valid_hex_digest(value: object) -> bool:
    """Return whether *value* is one exact lowercase SHA-256 hex digest."""
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        return bytes.fromhex(value).hex() == value
    except ValueError:
        return False


def _valid_metadata_terminal_digest(value: object) -> bool:
    """Return whether *value* names a Task-2 terminal receipt digest."""
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and _valid_hex_digest(value.removeprefix("sha256:"))
    )


def _ordered_file_digest(paths: Sequence[Path]) -> str:
    """Bind ordered status filenames and exact payload bytes."""
    digest = hashlib.sha256()
    for path in paths:
        name = path.name.encode("utf-8")
        payload = path.read_bytes()
        digest.update(struct.pack(">Q", len(name)))
        digest.update(name)
        digest.update(struct.pack(">Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _checked_control_root(path: Path) -> Path:
    """Return an absolute control root after rejecting symlink traversal."""
    candidate = Path(path).absolute()
    current = candidate
    while True:
        if current.is_symlink():
            raise ValueError(f"migration control root is a symlink: {candidate}")
        if current.parent == current:
            break
        current = current.parent
    return candidate.resolve()


def _manifest_paths(
    output_root: Path, control_root: Path | None = None
) -> tuple[Path, Path, Path, Path]:
    """Validate scientific and control roots before any manifest write."""
    state_dir = _checked_control_root(
        phenotypic_cache_dir(output_root)
        if control_root is None
        else control_root
    )
    records_path = state_dir / _RECORDS_FILENAME
    offsets_path = state_dir / _OFFSETS_FILENAME
    manifest_path = state_dir / _MANIFEST_FILENAME
    if control_root is None:
        for path in (state_dir, records_path, offsets_path, manifest_path):
            _checked_path(path, output_root)
    return state_dir, records_path, offsets_path, manifest_path


def _add_candidate(
    candidates: dict[tuple[str, str], dict[str, Path]],
    *,
    dataset: str,
    stem: str,
    kind: str,
    path: Path,
) -> None:
    """Add one artifact candidate, refusing an ambiguous duplicate kind."""
    identity = (dataset, stem)
    _safe_identity_component(dataset, "dataset")
    _safe_identity_component(stem, "stem")
    artifacts = candidates.setdefault(identity, {})
    if kind in artifacts:
        raise ValueError(
            f"ambiguous migration {kind} candidates for {dataset}/{stem}: "
            f"{artifacts[kind]} and {path}"
        )
    artifacts[kind] = path


def discover_migration_tasks(output_dir: Path) -> tuple[MigrationImageTask, ...]:
    """Discover one bounded, deterministic migration inventory.

    Args:
        output_dir: Root of the output tree being migrated.

    Returns:
        Tasks ordered by dataset then canonical image stem.

    Raises:
        ValueError: If a candidate is a symlink, escapes the run, is ambiguous,
            or has only an external measurement table.
    """
    output_root = Path(output_dir).resolve()
    candidates: dict[tuple[str, str], dict[str, Path]] = {}
    root = results_dir(output_root)
    if not root.is_dir():
        return ()
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue
        checked_dataset = _checked_path(dataset_dir, output_root)
        dataset = checked_dataset.name
        for kind, paths in (
            ("hdf", _iter_hdf_candidates(checked_dataset)),
            ("store", _iter_store_candidates(checked_dataset)),
            ("measurement", _iter_measurement_candidates(checked_dataset)),
        ):
            for candidate in paths:
                if candidate.name.startswith("."):
                    continue
                checked = _checked_path(candidate, output_root)
                if kind == "store" and not checked.is_dir():
                    continue
                if kind != "store" and not checked.is_file():
                    continue
                _add_candidate(
                    candidates,
                    dataset=dataset,
                    stem=_candidate_stem(checked),
                    kind=kind,
                    path=checked,
                )

    tasks: list[MigrationImageTask] = []
    for index, ((dataset, stem), artifacts) in enumerate(sorted(candidates.items())):
        if set(artifacts) == {"measurement"}:
            raise ValueError(
                f"measurement-only migration identity is unsupported: {dataset}/{stem}"
            )
        tasks.append(
            MigrationImageTask(
                index=index,
                dataset=dataset,
                stem=stem,
                hdf_path=artifacts.get("hdf"),
                store_path=_checked_path(
                    zarr_store_path(output_root, dataset, stem), output_root
                ),
                measurement_path=artifacts.get("measurement"),
                overlay_path=_checked_path(
                    dataset_overlays_dir(output_root, dataset) / f"{stem}.png",
                    output_root,
                ),
                marker_path=_checked_path(
                    image_completion_marker_path(output_root, dataset, stem),
                    output_root,
                ),
            )
        )
    return tuple(tasks)


def _task_value(task: MigrationImageTask) -> dict[str, Any]:
    """Return one canonical task value with absolute path strings."""
    value = asdict(task)
    for name in (
        "hdf_path",
        "store_path",
        "measurement_path",
        "overlay_path",
        "marker_path",
    ):
        path = value[name]
        value[name] = None if path is None else str(path)
    return value


def _leaf_payload(task: MigrationImageTask) -> bytes:
    """Serialize the index-bound canonical value covered by the Merkle root."""
    return json.dumps(
        _task_value(task), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _leaf_hash(task: MigrationImageTask) -> bytes:
    """Return an index-bound leaf hash for one canonical migration task."""
    return hashlib.sha256(
        b"migration-leaf\0" + struct.pack(">Q", task.index) + _leaf_payload(task)
    ).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    """Return a domain-separated binary Merkle parent hash."""
    return hashlib.sha256(b"migration-node\0" + left + right).digest()


def _merkle_root_and_proofs(
    tasks: Sequence[MigrationImageTask],
) -> tuple[bytes, tuple[tuple[dict[str, str], ...], ...]]:
    """Build a deterministic binary Merkle root and proof for every task."""
    if not tasks:
        return hashlib.sha256(b"migration-empty\0").digest(), ()
    proofs: list[list[dict[str, str]]] = [[] for _ in tasks]
    nodes: list[tuple[bytes, tuple[int, ...]]] = [
        (_leaf_hash(task), (task.index,)) for task in tasks
    ]
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        parents: list[tuple[bytes, tuple[int, ...]]] = []
        for position in range(0, len(nodes), 2):
            left, left_indexes = nodes[position]
            right, right_indexes = nodes[position + 1]
            for index in left_indexes:
                proofs[index].append({"side": "right", "hash": right.hex()})
            if right_indexes != left_indexes:
                for index in right_indexes:
                    proofs[index].append({"side": "left", "hash": left.hex()})
            parent_indexes = (
                left_indexes
                if right_indexes == left_indexes
                else left_indexes + right_indexes
            )
            parents.append((_node_hash(left, right), parent_indexes))
        nodes = parents
    return nodes[0][0], tuple(tuple(proof) for proof in proofs)


def _task_payload(
    task: MigrationImageTask, generation: str, proof: Sequence[Mapping[str, str]]
) -> bytes:
    """Serialize one task, its generation binding, and Merkle inclusion proof."""
    value = _task_value(task)
    value["generation"] = generation
    value["merkle_proof"] = [dict(step) for step in proof]
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _normalized_task(task: MigrationImageTask, output_root: Path) -> MigrationImageTask:
    """Validate a task's canonical identity and return absolute artifact paths."""
    if not isinstance(task.index, int) or task.index < 0:
        raise ValueError(f"migration task index must be a non-negative integer: {task.index!r}")
    _safe_identity_component(task.dataset, "dataset")
    _safe_identity_component(task.stem, "stem")
    hdf_path = (
        None if task.hdf_path is None else _checked_path(task.hdf_path, output_root)
    )
    measurement_path = (
        None
        if task.measurement_path is None
        else _checked_path(task.measurement_path, output_root)
    )
    store_path = _checked_path(task.store_path, output_root)
    overlay_path = _checked_path(task.overlay_path, output_root)
    marker_path = _checked_path(task.marker_path, output_root)
    if store_path != _checked_path(
        zarr_store_path(output_root, task.dataset, task.stem), output_root
    ):
        raise ValueError(f"migration task has non-canonical store path: {store_path}")
    if hdf_path is not None and hdf_path != (
        output_root / "results" / task.dataset / "hdf" / f"{task.stem}.h5"
    ):
        raise ValueError(f"migration task has non-canonical HDF path: {hdf_path}")
    if measurement_path is not None and measurement_path != (
        dataset_measurements_dir(output_root, task.dataset) / f"{task.stem}.parquet"
    ):
        raise ValueError(
            f"migration task has non-canonical measurement path: {measurement_path}"
        )
    if overlay_path != (
        dataset_overlays_dir(output_root, task.dataset) / f"{task.stem}.png"
    ):
        raise ValueError(f"migration task has non-canonical overlay path: {overlay_path}")
    if marker_path != image_completion_marker_path(
        output_root, task.dataset, task.stem
    ):
        raise ValueError(f"migration task has non-canonical marker path: {marker_path}")
    return MigrationImageTask(
        index=task.index,
        dataset=task.dataset,
        stem=task.stem,
        hdf_path=hdf_path,
        store_path=store_path,
        measurement_path=measurement_path,
        overlay_path=overlay_path,
        marker_path=marker_path,
    )


def _inventory_digest(tasks: Sequence[MigrationImageTask]) -> str:
    """Return the deterministic digest of an ordered, normalized inventory."""
    root, _ = _merkle_root_and_proofs(tasks)
    return root.hex()


def write_migration_manifest(
    output_dir: Path,
    *,
    generation: str,
    scientific_output: Path,
    tasks: Sequence[MigrationImageTask],
    control_root: Path | None = None,
) -> MigrationManifest:
    """Write a header, framed records, and direct-access offset table.

    Args:
        output_dir: Root of the output tree being migrated.
        generation: Immutable migration generation identifier.
        scientific_output: Canonical public deliverables directory for this run.
        tasks: Ordered migration tasks to serialize.

    Returns:
        The manifest metadata used by subsequent migration stages.

    Raises:
        ValueError: If paths escape the run or task indexes are not contiguous.
    """
    output_root = Path(output_dir).resolve()
    if not isinstance(generation, str) or not generation:
        raise ValueError("migration generation must be a non-empty string")
    normalized = tuple(_normalized_task(task, output_root) for task in tasks)
    if [task.index for task in normalized] != list(range(len(normalized))):
        raise ValueError("migration task indexes must be contiguous from zero")
    if len({(task.dataset, task.stem) for task in normalized}) != len(normalized):
        raise ValueError("migration task identities must be unique")
    scientific_path = _checked_path(scientific_output, output_root)
    if scientific_path != deliverables_dir(output_root):
        raise ValueError("scientific output must be the canonical deliverables directory")

    state_dir, records_path, offsets_path, manifest_path = _manifest_paths(
        output_root, control_root
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    state_dir, records_path, offsets_path, manifest_path = _manifest_paths(
        output_root, control_root
    )
    offsets: list[int] = []
    root, proofs = _merkle_root_and_proofs(normalized)
    with records_path.open("wb") as records:
        records.write(_RECORDS_HEADER)
        for task in normalized:
            payload = _task_payload(task, generation, proofs[task.index])
            offsets.append(records.tell())
            records.write(struct.pack(">Q", len(payload)))
            records.write(payload)
            records.write(hashlib.sha256(payload).digest())
    with offsets_path.open("wb") as offsets_file:
        for offset in offsets:
            offsets_file.write(struct.pack(">Q", offset))

    manifest = MigrationManifest(
        schema_version=_SCHEMA_VERSION,
        generation=generation,
        scientific_output=scientific_path,
        task_count=len(normalized),
        inventory_digest=root.hex(),
        records_path=records_path.resolve(),
        offsets_path=offsets_path.resolve(),
        output_root=output_root,
        control_root=state_dir,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": manifest.schema_version,
                "generation": manifest.generation,
                "scientific_output": str(manifest.scientific_output),
                "task_count": manifest.task_count,
                "inventory_digest": manifest.inventory_digest,
                "records_path": str(manifest.records_path),
                "offsets_path": str(manifest.offsets_path),
                "output_root": str(manifest.output_root),
                "control_root": str(manifest.control_root),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return manifest


def _read_manifest(
    manifest_path: Path,
    expected_scientific_output: Path,
    *,
    expected_control_root: Path | None = None,
) -> tuple[Path, MigrationManifest]:
    """Decode and validate one manifest header and all path boundaries."""
    supplied_path = Path(manifest_path).absolute()
    scientific_output = Path(expected_scientific_output).resolve()
    output_root = scientific_output.parent.resolve()
    if scientific_output != deliverables_dir(output_root):
        raise ValueError("expected scientific output is not canonical")
    default_control = phenotypic_cache_dir(output_root).resolve()
    bound_control = _checked_control_root(
        default_control if expected_control_root is None else expected_control_root
    )
    header_path = supplied_path.resolve()
    if (
        supplied_path.is_symlink()
        or header_path.name != _MANIFEST_FILENAME
        or header_path.parent != bound_control
    ):
        raise ValueError(
            "migration manifest path does not match the expected control root: "
            f"{manifest_path}"
        )
    try:
        raw = json.loads(header_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid migration manifest header: {manifest_path}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("invalid migration manifest header schema")
    schema_version = raw.get("schema_version")
    expected = {
        "schema_version",
        "generation",
        "scientific_output",
        "task_count",
        "inventory_digest",
        "records_path",
        "offsets_path",
    }
    if schema_version == _SCHEMA_VERSION:
        expected |= {"output_root", "control_root"}
    elif schema_version != _LEGACY_SCHEMA_VERSION:
        raise ValueError("invalid migration manifest header schema")
    if set(raw) != expected:
        raise ValueError("invalid migration manifest header schema")
    generation = raw.get("generation")
    task_count = raw.get("task_count")
    digest = raw.get("inventory_digest")
    if (
        not isinstance(generation, str)
        or not generation
        or not isinstance(task_count, int)
        or task_count < 0
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise ValueError("invalid migration manifest header schema")
    fields: tuple[str, ...] = (
        "scientific_output",
        "records_path",
        "offsets_path",
    )
    if schema_version == _SCHEMA_VERSION:
        fields += ("output_root", "control_root")
    if any(not isinstance(raw[name], str) or not Path(raw[name]).is_absolute() for name in fields):
        raise ValueError("migration manifest paths must be absolute")
    if schema_version == _SCHEMA_VERSION:
        if Path(raw["output_root"]).resolve() != output_root:
            raise ValueError("migration manifest output root does not match expected output root")
        if Path(raw["control_root"]).resolve() != bound_control:
            raise ValueError("migration manifest control root does not match expected control root")
    elif bound_control != default_control:
        raise ValueError("legacy migration manifest requires its historical control root")
    records_path = Path(raw["records_path"]).resolve()
    offsets_path = Path(raw["offsets_path"]).resolve()
    if records_path.parent != bound_control or offsets_path.parent != bound_control:
        raise ValueError("migration manifest control artifacts escape the expected control root")
    manifest = MigrationManifest(
        schema_version=int(schema_version),
        generation=generation,
        scientific_output=_checked_path(Path(raw["scientific_output"]), output_root),
        task_count=task_count,
        inventory_digest=digest,
        records_path=records_path,
        offsets_path=offsets_path,
        output_root=output_root,
        control_root=bound_control,
    )
    if manifest.scientific_output != deliverables_dir(output_root):
        raise ValueError("migration manifest has non-canonical scientific output")
    if manifest.scientific_output != scientific_output:
        raise ValueError(
            "migration manifest scientific output does not match expected scientific output"
        )
    return output_root, manifest


def _read_record_payload(manifest: MigrationManifest, index: int) -> bytes:
    """Seek directly to one checked frame and return its checksum-verified payload."""
    if index < 0 or index >= manifest.task_count:
        raise IndexError(f"migration task index out of range: {index}")
    try:
        offsets = manifest.offsets_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read migration offsets: {manifest.offsets_path}") from exc
    if len(offsets) != manifest.task_count * 8:
        raise ValueError("migration offset file has invalid alignment")
    offset = struct.unpack(">Q", offsets[index * 8 : (index + 1) * 8])[0]
    try:
        with manifest.records_path.open("rb") as records:
            expected_header = _RECORDS_MAGIC + struct.pack(
                ">I", manifest.schema_version
            )
            if records.read(len(expected_header)) != expected_header:
                raise ValueError("migration records file has invalid magic/version")
            if offset < len(expected_header):
                raise ValueError("migration record offset precedes frames")
            records.seek(offset)
            length_bytes = records.read(8)
            if len(length_bytes) != 8:
                raise ValueError("migration record frame is truncated")
            length = struct.unpack(">Q", length_bytes)[0]
            payload = records.read(length)
            checksum = records.read(32)
    except OSError as exc:
        raise ValueError(f"cannot read migration records: {manifest.records_path}") from exc
    if len(payload) != length or len(checksum) != 32:
        raise ValueError("migration record frame is truncated")
    if hashlib.sha256(payload).digest() != checksum:
        raise ValueError("migration record payload checksum mismatch")
    return payload


def _task_from_payload(
    payload: bytes, *, output_root: Path, manifest: MigrationManifest, index: int
) -> MigrationImageTask:
    """Decode one validated JSON payload and enforce its manifest binding."""
    try:
        raw: Any = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("migration record payload is not valid JSON") from exc
    if not isinstance(raw, Mapping) or set(raw) != _TASK_FIELDS:
        raise ValueError("migration record has invalid schema")
    if raw.get("generation") != manifest.generation:
        raise ValueError("migration record generation does not match manifest")
    if raw.get("index") != index:
        raise ValueError("migration record index does not match requested index")
    path_names = ("hdf_path", "store_path", "measurement_path", "overlay_path", "marker_path")
    if any(
        raw[name] is not None
        and (not isinstance(raw[name], str) or not Path(raw[name]).is_absolute())
        for name in path_names
    ):
        raise ValueError("migration record paths must be absolute strings")
    if not isinstance(raw.get("dataset"), str) or not isinstance(raw.get("stem"), str):
        raise ValueError("migration record has invalid identity")
    task = MigrationImageTask(
        index=raw["index"],
        dataset=raw["dataset"],
        stem=raw["stem"],
        hdf_path=None if raw["hdf_path"] is None else Path(raw["hdf_path"]),
        store_path=Path(raw["store_path"]),
        measurement_path=(
            None if raw["measurement_path"] is None else Path(raw["measurement_path"])
        ),
        overlay_path=Path(raw["overlay_path"]),
        marker_path=Path(raw["marker_path"]),
    )
    normalized = _normalized_task(task, output_root)
    proof = raw["merkle_proof"]
    if not isinstance(proof, list):
        raise ValueError("migration record has invalid Merkle proof")
    computed = _leaf_hash(normalized)
    for step in proof:
        if (
            not isinstance(step, Mapping)
            or set(step) != {"side", "hash"}
            or step.get("side") not in {"left", "right"}
            or not isinstance(step.get("hash"), str)
            or len(step["hash"]) != 64
        ):
            raise ValueError("migration record has invalid Merkle proof")
        try:
            sibling = bytes.fromhex(step["hash"])
        except ValueError as exc:
            raise ValueError("migration record has invalid Merkle proof") from exc
        computed = (
            _node_hash(sibling, computed)
            if step["side"] == "left"
            else _node_hash(computed, sibling)
        )
    if computed.hex() != manifest.inventory_digest:
        raise ValueError("migration record Merkle proof does not match inventory digest")
    return normalized


def read_migration_task(
    manifest_path: Path,
    index: int,
    *,
    expected_scientific_output: Path,
    expected_control_root: Path | None = None,
) -> MigrationImageTask:
    """Read exactly one indexed migration task without parsing prior records.

    Args:
        manifest_path: Canonical migration manifest header path.
        index: Zero-based array index to load.
        expected_scientific_output: Caller-authorized deliverables root.

    Returns:
        The checksum-verified task at *index*.

    Raises:
        ValueError: If the header, offsets, frame, checksum, schema, or
            generation is invalid.
        IndexError: If *index* is outside the inventory.
    """
    output_root, manifest = _read_manifest(
        manifest_path,
        expected_scientific_output,
        expected_control_root=expected_control_root,
    )
    return _task_from_payload(
        _read_record_payload(manifest, index),
        output_root=output_root,
        manifest=manifest,
        index=index,
    )


def _validated_manifest_for_authority(
    control_root: Path,
    manifest_path: Path,
    expected_scientific_output: Path,
    generation: str,
) -> tuple[Path, MigrationManifest]:
    """Load a manifest and prove it belongs to this control generation."""
    output_root, manifest = _read_manifest(
        manifest_path,
        expected_scientific_output,
        expected_control_root=control_root,
    )
    if manifest.generation != generation:
        raise ValueError("migration authority generation does not match manifest")
    return output_root, manifest


def publish_migration_task_status(
    control_root: Path,
    *,
    manifest_path: Path,
    expected_scientific_output: Path,
    generation: str,
    metadata_terminal_digest: str,
    result: Any,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Atomically publish one completed image result under its generation fence."""
    output_root, manifest = _validated_manifest_for_authority(
        control_root,
        manifest_path,
        expected_scientific_output,
        generation,
    )
    if not _valid_metadata_terminal_digest(metadata_terminal_digest):
        raise ValueError("migration status has an invalid metadata digest")
    if not isinstance(getattr(result, "index", None), int):
        raise ValueError("migration result has an invalid manifest index")
    task = read_migration_task(
        manifest_path,
        result.index,
        expected_scientific_output=expected_scientific_output,
        expected_control_root=control_root,
    )
    if (
        result.dataset != task.dataset
        or result.stem != task.stem
        or result.index != task.index
    ):
        raise ValueError("migration result identity does not match manifest")

    from ._cli_migrate_image import (
        _configured_work_id,
        _valid_migration_marker,
    )

    expected_work_id = _configured_work_id(
        output_root, task.dataset, task.stem
    )
    if result.work_id != expected_work_id:
        raise ValueError("migration result work ID does not match current state")
    if not _valid_hex_digest(result.marker_digest):
        raise ValueError("migration result has an invalid marker digest")
    try:
        current_marker_digest = _sha256_bytes(task.marker_path.read_bytes())
    except OSError as exc:
        raise ValueError("migration result marker is missing") from exc
    if current_marker_digest != result.marker_digest:
        raise ValueError("migration result marker digest does not match current bytes")
    if not _valid_migration_marker(
        output_root,
        task,
        result.work_id,
    ):
        raise ValueError("migration result lacks current semantic marker authority")

    path = migration_task_status_path(control_root, generation, task.index)
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "state": "complete",
            "generation": manifest.generation,
            "manifest_digest": manifest.inventory_digest,
            "index": task.index,
            "dataset": task.dataset,
            "stem": task.stem,
            "work_id": result.work_id,
            "metadata_terminal_digest": metadata_terminal_digest,
            "marker_payload_digest": result.marker_digest,
        },
        commit_guard=commit_guard,
    )
    return path


_IMAGE_STATUS_FIELDS: Final = frozenset(
    {
        "schema_version",
        "state",
        "generation",
        "manifest_digest",
        "index",
        "dataset",
        "stem",
        "work_id",
        "metadata_terminal_digest",
        "marker_payload_digest",
    }
)


def _read_json_mapping(path: Path, role: str) -> Mapping[str, Any]:
    """Read one exact JSON mapping or raise a role-specific value error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}: {path.name}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid {role}: {path.name}")
    return value


def seal_migration_image_stage(
    control_root: Path,
    *,
    manifest_path: Path,
    expected_scientific_output: Path,
    generation: str,
    metadata_terminal_digest: str,
    commit_guard: CommitGuard | None = None,
) -> MigrationImageSeal:
    """Validate all image statuses and publish a clean or diagnostic seal."""
    output_root, manifest = _validated_manifest_for_authority(
        control_root,
        manifest_path,
        expected_scientific_output,
        generation,
    )
    failures: list[str] = []
    if not _valid_metadata_terminal_digest(metadata_terminal_digest):
        failures.append("invalid expected metadata digest")
    status_dir = migration_task_status_path(
        control_root, generation, 0
    ).parent
    status_paths = sorted(status_dir.glob("*.json")) if status_dir.is_dir() else []
    seen: dict[int, list[tuple[Path, Mapping[str, Any]]]] = {}
    for status_path in status_paths:
        try:
            status = _read_json_mapping(status_path, "migration image status")
        except ValueError as exc:
            failures.append(str(exc))
            continue
        index = status.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            failures.append(f"invalid status index in {status_path.name}")
            continue
        seen.setdefault(index, []).append((status_path, status))
        if index >= manifest.task_count:
            failures.append(f"extra status index {index}")
            continue
        if status_path != migration_task_status_path(control_root, generation, index):
            failures.append(f"non-canonical status path for index {index}")

    for index, entries in sorted(seen.items()):
        if len(entries) > 1:
            failures.append(f"duplicate status index {index}")

    from ._cli_migrate_image import (
        _configured_work_id,
        _valid_migration_marker,
    )

    for index in range(manifest.task_count):
        entries = seen.get(index, [])
        if not entries:
            failures.append(f"missing status index {index}")
            continue
        if len(entries) != 1:
            continue
        _, status = entries[0]
        if set(status) != _IMAGE_STATUS_FIELDS:
            failures.append(f"status index {index} has invalid schema")
            continue
        if status.get("schema_version") != 1 or status.get("state") != "complete":
            failures.append(f"status index {index} is not complete")
        if status.get("generation") != generation:
            failures.append(f"status index {index} has wrong generation")
        if status.get("manifest_digest") != manifest.inventory_digest:
            failures.append(f"status index {index} has wrong manifest digest")
        task = read_migration_task(
            manifest_path,
            index,
            expected_scientific_output=expected_scientific_output,
            expected_control_root=control_root,
        )
        if status.get("dataset") != task.dataset or status.get("stem") != task.stem:
            failures.append(f"status index {index} has wrong image identity")
        expected_work_id = _configured_work_id(
            output_root, task.dataset, task.stem
        )
        if status.get("work_id") != expected_work_id:
            failures.append(f"status index {index} has wrong work ID")
        if status.get("metadata_terminal_digest") != metadata_terminal_digest:
            failures.append(f"status index {index} has wrong metadata digest")
        marker_digest = status.get("marker_payload_digest")
        try:
            current_marker_digest = _sha256_bytes(task.marker_path.read_bytes())
        except OSError:
            failures.append(f"status index {index} has missing current marker")
        else:
            if not _valid_hex_digest(marker_digest):
                failures.append(f"status index {index} has invalid marker digest")
            elif marker_digest != current_marker_digest:
                failures.append(f"status index {index} has wrong marker digest")
        if not _valid_migration_marker(
            output_root,
            task,
            expected_work_id,
        ):
            failures.append(
                f"status index {index} lacks current semantic marker authority"
            )

    try:
        ordered_status_digest = _ordered_file_digest(status_paths)
    except OSError as exc:
        failures.append(f"could not digest migration image statuses: {exc}")
        ordered_status_digest = hashlib.sha256(b"").hexdigest()
    seal_path = migration_image_seal_path(control_root, generation)
    payload = {
        "schema_version": 1,
        "generation": generation,
        "manifest_digest": manifest.inventory_digest,
        "ordered_status_digest": ordered_status_digest,
        "metadata_terminal_digest": metadata_terminal_digest,
        "clean": not failures,
        "failures": failures,
    }
    atomic_write_json(seal_path, payload, commit_guard=commit_guard)
    return MigrationImageSeal(
        generation=generation,
        manifest_digest=manifest.inventory_digest,
        ordered_status_digest=ordered_status_digest,
        metadata_terminal_digest=metadata_terminal_digest,
        clean=not failures,
        failures=tuple(failures),
        seal_path=seal_path,
    )


def _source_state_value(state: Any) -> dict[str, Any]:
    """Serialize one Task-3 source-state result without losing absence."""
    return {
        "path": None if state.path is None else str(Path(state.path)),
        "exists": state.exists,
        "size": state.size,
        "sha256": state.sha256,
    }


def _current_source_state_value(path: Path | None) -> dict[str, Any]:
    """Fingerprint one live source file or record its exact absence."""
    if path is None:
        return {"path": None, "exists": False, "size": None, "sha256": None}
    source = Path(path)
    try:
        payload = source.read_bytes()
    except FileNotFoundError:
        return {
            "path": str(source),
            "exists": False,
            "size": None,
            "sha256": None,
        }
    return {
        "path": str(source),
        "exists": True,
        "size": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _valid_source_state_value(value: object, expected_path: Path | None) -> bool:
    """Validate one serialized source fingerprint and its manifest path."""
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "exists",
        "size",
        "sha256",
    }:
        return False
    expected = None if expected_path is None else str(expected_path)
    if value.get("path") != expected or not isinstance(value.get("exists"), bool):
        return False
    if value["exists"]:
        return (
            isinstance(value.get("size"), int)
            and not isinstance(value.get("size"), bool)
            and value["size"] >= 0
            and _valid_hex_digest(value.get("sha256"))
        )
    return value.get("size") is None and value.get("sha256") is None


def _expected_reclaim_paths(task: MigrationImageTask) -> tuple[Path, ...]:
    """Return the manifest-ordered source deletion intent for one image."""
    return tuple(
        path for path in (task.hdf_path, task.measurement_path) if path is not None
    )


def _validate_reclaim_result(
    output_root: Path,
    task: MigrationImageTask,
    result: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Return one canonical reclaim payload plus complete validation failures."""
    failures: list[str] = []
    if (
        result.index != task.index
        or result.dataset != task.dataset
        or result.stem != task.stem
    ):
        failures.append("reclaim result identity does not match manifest")

    from ._cli_migrate_image import _configured_work_id

    expected_work_id = _configured_work_id(
        output_root, task.dataset, task.stem
    )
    if result.work_id != expected_work_id:
        failures.append("reclaim result work ID does not match current state")
    try:
        current_marker_digest = _sha256_bytes(task.marker_path.read_bytes())
    except OSError:
        current_marker_digest = ""
    retained_after_unclean_image = (
        current_marker_digest == ""
        and result.marker_digest == ""
        and result.reason == "image seal was not clean; sources retained"
        and not result.deleted_paths
    )
    if current_marker_digest == "" and not retained_after_unclean_image:
        failures.append("reclaim result marker is missing")
    if not retained_after_unclean_image and (
        not _valid_hex_digest(result.marker_digest)
        or result.marker_digest != current_marker_digest
    ):
        failures.append("reclaim result marker digest does not match current bytes")

    expected_deletions = _expected_reclaim_paths(task)
    intended = tuple(Path(path) for path in result.intended_deletions)
    if intended != expected_deletions:
        failures.append("reclaim result intended deletion set does not match manifest")
    hdf_prestate = _source_state_value(result.hdf_prestate)
    parquet_prestate = _source_state_value(result.parquet_prestate)
    if not _valid_source_state_value(hdf_prestate, task.hdf_path):
        failures.append("reclaim result HDF source prestate is invalid")
    if not _valid_source_state_value(parquet_prestate, task.measurement_path):
        failures.append("reclaim result Parquet source prestate is invalid")
    try:
        observed_values = tuple(
            _source_state_value(state) for state in result.observed_poststate
        )
    except TypeError:
        observed_values = ()
    if len(observed_values) != 2:
        failures.append("reclaim result observed poststate is incomplete")
        observed_values = (
            _current_source_state_value(task.hdf_path),
            _current_source_state_value(task.measurement_path),
        )
    else:
        for name, value, path in zip(
            ("HDF", "Parquet"),
            observed_values,
            (task.hdf_path, task.measurement_path),
            strict=True,
        ):
            if not _valid_source_state_value(value, path):
                failures.append(f"reclaim result {name} poststate is invalid")
            if value != _current_source_state_value(path):
                failures.append(
                    f"reclaim result {name} poststate does not match current source"
                )

    deleted = tuple(Path(path) for path in result.deleted_paths)
    retained = tuple(Path(path) for path in result.retained_paths)
    if len(set(deleted)) != len(deleted) or any(
        path not in expected_deletions for path in deleted
    ):
        failures.append("reclaim result deleted paths are invalid")
    if len(set(retained)) != len(retained) or any(
        path not in expected_deletions for path in retained
    ):
        failures.append("reclaim result retained paths are invalid")
    if set(deleted) & set(retained):
        failures.append("reclaim result classifies one source twice")

    source_pairs = tuple(zip(
        (task.hdf_path, task.measurement_path),
        (hdf_prestate, parquet_prestate),
        observed_values,
        strict=True,
    ))
    for path, prestate, poststate in source_pairs:
        if path is None:
            continue
        if path in deleted and (
            prestate.get("exists") is not True or poststate.get("exists") is not False
        ):
            failures.append(f"reclaim result source transition is invalid: {path}")
        if path in retained and poststate.get("exists") is not True:
            failures.append(f"reclaim result retained source is absent: {path}")
        if path in retained and prestate != poststate:
            failures.append(
                f"reclaim result prestate does not match current source: {path}"
            )
        if prestate.get("exists") is True and path not in deleted and path not in retained:
            failures.append(f"reclaim result omits source disposition: {path}")

    expected_deleted = tuple(
        path
        for path, prestate, poststate in source_pairs
        if path is not None
        and prestate.get("exists") is True
        and poststate.get("exists") is False
    )
    expected_retained = tuple(
        path
        for path, _prestate, poststate in source_pairs
        if path is not None and poststate.get("exists") is True
    )
    if deleted != expected_deleted:
        failures.append("reclaim result deleted paths do not match exact transition")
    if retained != expected_retained:
        failures.append("reclaim result retained paths do not match exact transition")

    reason = result.reason
    if reason is not None and not isinstance(reason, str):
        failures.append("reclaim result reason is invalid")
        reason = str(reason)
    return (
        {
            "schema_version": 1,
            "state": "complete",
            "index": task.index,
            "dataset": task.dataset,
            "stem": task.stem,
            "work_id": result.work_id,
            "marker_payload_digest": result.marker_digest,
            "intended_deletions": [str(path) for path in intended],
            "hdf_prestate": hdf_prestate,
            "parquet_prestate": parquet_prestate,
            "observed_poststate": list(observed_values),
            "deleted_paths": [str(path) for path in deleted],
            "retained_paths": [str(path) for path in retained],
            "reason": reason,
        },
        failures,
    )


def publish_migration_reclaim_status(
    control_root: Path,
    *,
    manifest_path: Path,
    expected_scientific_output: Path,
    generation: str,
    result: Any,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Atomically publish one exact source-reclamation result."""
    output_root, manifest = _validated_manifest_for_authority(
        control_root,
        manifest_path,
        expected_scientific_output,
        generation,
    )
    if not isinstance(getattr(result, "index", None), int):
        raise ValueError("reclaim result has an invalid manifest index")
    task = read_migration_task(
        manifest_path,
        result.index,
        expected_scientific_output=expected_scientific_output,
        expected_control_root=control_root,
    )
    payload, failures = _validate_reclaim_result(output_root, task, result)
    if failures:
        raise ValueError("; ".join(failures))
    payload["generation"] = manifest.generation
    payload["manifest_digest"] = manifest.inventory_digest
    path = migration_reclaim_status_path(control_root, generation, task.index)
    atomic_write_json(path, payload, commit_guard=commit_guard)
    return path


_RECLAIM_STATUS_FIELDS: Final = frozenset(
    {
        "schema_version",
        "state",
        "generation",
        "manifest_digest",
        "index",
        "dataset",
        "stem",
        "work_id",
        "marker_payload_digest",
        "intended_deletions",
        "hdf_prestate",
        "parquet_prestate",
        "observed_poststate",
        "deleted_paths",
        "retained_paths",
        "reason",
    }
)


def _image_seal_matches(
    image_seal: MigrationImageSeal, manifest: MigrationManifest, generation: str
) -> bool:
    """Return whether supplied image evidence names this clean manifest."""
    return (
        image_seal.clean
        and image_seal.generation == generation
        and image_seal.manifest_digest == manifest.inventory_digest
        and image_seal.seal_path == migration_image_seal_path(
            image_seal.seal_path.parents[2], generation
        )
    )


def seal_migration_reclaim_stage(
    control_root: Path,
    *,
    manifest_path: Path,
    expected_scientific_output: Path,
    generation: str,
    deletion_requested: bool,
    image_seal: MigrationImageSeal | None = None,
    commit_guard: CommitGuard | None = None,
) -> MigrationReclaimSeal | None:
    """Validate every source transition when deletion was requested."""
    if not deletion_requested:
        return None
    output_root, manifest = _validated_manifest_for_authority(
        control_root,
        manifest_path,
        expected_scientific_output,
        generation,
    )
    failures: list[str] = []
    if image_seal is None:
        failures.append("a clean image seal is required before reclaim")
    elif not _image_seal_matches(
        image_seal, manifest, generation
    ) or not valid_migration_image_seal(
        control_root,
        image_seal,
        manifest_path=manifest_path,
        expected_scientific_output=expected_scientific_output,
    ):
        failures.append("image seal is not clean or does not match this manifest")

    status_dir = migration_reclaim_status_path(
        control_root, generation, 0
    ).parent
    status_paths = sorted(status_dir.glob("*.json")) if status_dir.is_dir() else []
    seen: dict[int, list[tuple[Path, Mapping[str, Any]]]] = {}
    for status_path in status_paths:
        try:
            status = _read_json_mapping(status_path, "migration reclaim status")
        except ValueError as exc:
            failures.append(str(exc))
            continue
        index = status.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            failures.append(f"invalid reclaim status index in {status_path.name}")
            continue
        seen.setdefault(index, []).append((status_path, status))
        if index >= manifest.task_count:
            failures.append(f"extra reclaim status index {index}")
            continue
        if status_path != migration_reclaim_status_path(
            control_root, generation, index
        ):
            failures.append(f"non-canonical reclaim status path for index {index}")
    for index, entries in sorted(seen.items()):
        if len(entries) > 1:
            failures.append(f"duplicate reclaim status index {index}")

    from ._cli_migrate_image import _configured_work_id

    for index in range(manifest.task_count):
        entries = seen.get(index, [])
        if not entries:
            failures.append(f"missing reclaim status index {index}")
            continue
        if len(entries) != 1:
            continue
        _, status = entries[0]
        if set(status) != _RECLAIM_STATUS_FIELDS:
            failures.append(f"reclaim status index {index} has invalid schema")
            continue
        if status.get("schema_version") != 1 or status.get("state") != "complete":
            failures.append(f"reclaim status index {index} is not complete")
        if status.get("generation") != generation:
            failures.append(f"reclaim status index {index} has wrong generation")
        if status.get("manifest_digest") != manifest.inventory_digest:
            failures.append(f"reclaim status index {index} has wrong manifest digest")
        task = read_migration_task(
            manifest_path,
            index,
            expected_scientific_output=expected_scientific_output,
            expected_control_root=control_root,
        )
        if status.get("dataset") != task.dataset or status.get("stem") != task.stem:
            failures.append(f"reclaim status index {index} has wrong image identity")
        expected_work_id = _configured_work_id(
            output_root, task.dataset, task.stem
        )
        if status.get("work_id") != expected_work_id:
            failures.append(f"reclaim status index {index} has wrong work ID")
        try:
            marker_digest = _sha256_bytes(task.marker_path.read_bytes())
        except OSError:
            marker_digest = ""
        if status.get("marker_payload_digest") != marker_digest:
            failures.append(f"reclaim status index {index} has wrong marker digest")

        expected_paths = _expected_reclaim_paths(task)
        if status.get("intended_deletions") != [
            str(path) for path in expected_paths
        ]:
            failures.append(
                f"reclaim status index {index} has wrong intended deletion set"
            )
        prestates = (status.get("hdf_prestate"), status.get("parquet_prestate"))
        for name, prestate, path in zip(
            ("HDF", "Parquet"),
            prestates,
            (task.hdf_path, task.measurement_path),
            strict=True,
        ):
            if not _valid_source_state_value(prestate, path):
                failures.append(
                    f"reclaim status index {index} has invalid {name} source prestate"
                )
        observed = status.get("observed_poststate")
        if not isinstance(observed, list) or len(observed) != 2:
            failures.append(f"reclaim status index {index} has incomplete poststate")
            observed = [None, None]
        for name, poststate, path in zip(
            ("HDF", "Parquet"),
            observed,
            (task.hdf_path, task.measurement_path),
            strict=True,
        ):
            if not _valid_source_state_value(poststate, path):
                failures.append(
                    f"reclaim status index {index} has invalid {name} poststate"
                )
            if poststate != _current_source_state_value(path):
                failures.append(
                    f"reclaim status index {index} {name} poststate does not match current source"
                )
        source_transitions = tuple(zip(
            (task.hdf_path, task.measurement_path),
            prestates,
            observed,
            strict=True,
        ))
        deleted = status.get("deleted_paths")
        expected_deleted = [
            str(path)
            for path, prestate, poststate in source_transitions
            if path is not None
            and isinstance(prestate, Mapping)
            and prestate.get("exists") is True
            and isinstance(poststate, Mapping)
            and poststate.get("exists") is False
        ]
        if deleted != expected_deleted:
            failures.append(
                f"reclaim status index {index} has wrong deleted paths"
            )
        retained = status.get("retained_paths")
        expected_retained = [
            str(path)
            for path, _prestate, poststate in source_transitions
            if path is not None
            and isinstance(poststate, Mapping)
            and poststate.get("exists") is True
        ]
        if retained != expected_retained:
            failures.append(f"reclaim status index {index} has invalid retained paths")
        elif expected_retained:
            failures.append(f"reclaim status index {index} retained sources: {retained}")
        for path, prestate, poststate in source_transitions:
            if (
                path is not None
                and str(path) in expected_retained
                and prestate != poststate
            ):
                failures.append(
                    f"reclaim status index {index} prestate does not match retained source: {path}"
                )
        if any(
            isinstance(poststate, Mapping) and poststate.get("exists") is True
            for poststate in observed
        ):
            failures.append(f"reclaim status index {index} has sources still present")

    try:
        ordered_digest = _ordered_file_digest(status_paths)
    except OSError as exc:
        failures.append(f"could not digest migration reclaim statuses: {exc}")
        ordered_digest = hashlib.sha256(b"").hexdigest()
    seal_path = migration_reclaim_seal_path(control_root, generation)
    payload = {
        "schema_version": 1,
        "generation": generation,
        "manifest_digest": manifest.inventory_digest,
        "ordered_reclaim_status_digest": ordered_digest,
        "deletion_requested": True,
        "clean": not failures,
        "failures": failures,
    }
    atomic_write_json(seal_path, payload, commit_guard=commit_guard)
    return MigrationReclaimSeal(
        generation=generation,
        manifest_digest=manifest.inventory_digest,
        ordered_reclaim_status_digest=ordered_digest,
        deletion_requested=True,
        clean=not failures,
        failures=tuple(failures),
        seal_path=seal_path,
    )


def valid_migration_image_seal(
    control_root: Path,
    seal: MigrationImageSeal,
    *,
    manifest_path: Path | None = None,
    expected_scientific_output: Path | None = None,
) -> bool:
    """Return whether an image seal and its ordered status bytes are current."""
    expected_path = migration_image_seal_path(control_root, seal.generation)
    if seal.seal_path != expected_path:
        return False
    try:
        payload = _read_json_mapping(expected_path, "migration image seal")
        status_dir = migration_task_status_path(
            control_root, seal.generation, 0
        ).parent
        status_paths = (
            sorted(status_dir.glob("*.json")) if status_dir.is_dir() else []
        )
        current_digest = _ordered_file_digest(status_paths)
    except (OSError, ValueError):
        return False
    seal_is_current = dict(payload) == {
        "schema_version": 1,
        "generation": seal.generation,
        "manifest_digest": seal.manifest_digest,
        "ordered_status_digest": seal.ordered_status_digest,
        "metadata_terminal_digest": seal.metadata_terminal_digest,
        "clean": seal.clean,
        "failures": list(seal.failures),
    } and current_digest == seal.ordered_status_digest
    if not seal_is_current:
        return False
    if manifest_path is None and expected_scientific_output is None:
        return True
    if manifest_path is None or expected_scientific_output is None:
        return False
    try:
        output_root, manifest = _validated_manifest_for_authority(
            control_root,
            manifest_path,
            expected_scientific_output,
            seal.generation,
        )
        if manifest.inventory_digest != seal.manifest_digest:
            return False
        from ._cli_migrate_image import (
            _configured_work_id,
            _valid_migration_marker,
        )

        for index in range(manifest.task_count):
            task = read_migration_task(
                manifest_path,
                index,
                expected_scientific_output=expected_scientific_output,
                expected_control_root=control_root,
            )
            status = _read_json_mapping(
                migration_task_status_path(control_root, seal.generation, index),
                "migration image status",
            )
            marker_digest = _sha256_bytes(task.marker_path.read_bytes())
            if status.get("marker_payload_digest") != marker_digest:
                return False
            work_id = _configured_work_id(
                output_root,
                task.dataset,
                task.stem,
            )
            if not _valid_migration_marker(output_root, task, work_id):
                return False
    except (OSError, ValueError):
        return False
    return True


def valid_migration_reclaim_seal(
    control_root: Path,
    seal: MigrationReclaimSeal,
    *,
    manifest_path: Path | None = None,
    expected_scientific_output: Path | None = None,
) -> bool:
    """Return whether a reclaim seal and ordered status bytes are current."""
    expected_path = migration_reclaim_seal_path(control_root, seal.generation)
    if seal.seal_path != expected_path:
        return False
    try:
        payload = _read_json_mapping(expected_path, "migration reclaim seal")
        status_dir = migration_reclaim_status_path(
            control_root, seal.generation, 0
        ).parent
        status_paths = (
            sorted(status_dir.glob("*.json")) if status_dir.is_dir() else []
        )
        current_digest = _ordered_file_digest(status_paths)
    except (OSError, ValueError):
        return False
    seal_is_current = dict(payload) == {
        "schema_version": 1,
        "generation": seal.generation,
        "manifest_digest": seal.manifest_digest,
        "ordered_reclaim_status_digest": seal.ordered_reclaim_status_digest,
        "deletion_requested": seal.deletion_requested,
        "clean": seal.clean,
        "failures": list(seal.failures),
    } and current_digest == seal.ordered_reclaim_status_digest
    if not seal_is_current:
        return False
    if manifest_path is None and expected_scientific_output is None:
        return True
    if manifest_path is None or expected_scientific_output is None:
        return False
    try:
        _, manifest = _validated_manifest_for_authority(
            control_root,
            manifest_path,
            expected_scientific_output,
            seal.generation,
        )
        if manifest.inventory_digest != seal.manifest_digest:
            return False
        for index in range(manifest.task_count):
            task = read_migration_task(
                manifest_path,
                index,
                expected_scientific_output=expected_scientific_output,
                expected_control_root=control_root,
            )
            status = _read_json_mapping(
                migration_reclaim_status_path(control_root, seal.generation, index),
                "migration reclaim status",
            )
            try:
                marker_digest = _sha256_bytes(task.marker_path.read_bytes())
            except OSError:
                marker_digest = ""
            if status.get("marker_payload_digest") != marker_digest:
                return False
            observed = status.get("observed_poststate")
            if observed != [
                _current_source_state_value(task.hdf_path),
                _current_source_state_value(task.measurement_path),
            ]:
                return False
            if status.get("retained_paths"):
                return False
    except (OSError, ValueError):
        return False
    return True
