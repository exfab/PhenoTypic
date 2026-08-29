"""Indexed, tamper-evident inventory for resumable output migration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Final, Mapping, Sequence

from phenotypic.sdk_ import (
    STORE_SUFFIX,
    dataset_measurements_dir,
    dataset_overlays_dir,
    deliverables_dir,
    image_completion_marker_path,
    phenotypic_cache_dir,
    results_dir,
    store_stem,
    zarr_store_path,
)


_SCHEMA_VERSION: Final = 1
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
    candidate = Path(path)
    try:
        relative = candidate.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"migration candidate escapes output directory: {candidate}") from exc
    current = output_root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"migration candidate is a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(output_root):
        raise ValueError(f"migration candidate escapes output directory: {candidate}")
    return resolved


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


def _task_payload(task: MigrationImageTask, generation: str) -> bytes:
    """Serialize one already-validated task as a canonical JSON frame payload."""
    value = asdict(task)
    value["generation"] = generation
    for name in ("hdf_path", "store_path", "measurement_path", "overlay_path", "marker_path"):
        path = value[name]
        value[name] = None if path is None else str(path)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _normalized_task(task: MigrationImageTask, output_root: Path) -> MigrationImageTask:
    """Validate a task's canonical identity and return absolute artifact paths."""
    if not isinstance(task.index, int) or task.index < 0:
        raise ValueError(f"migration task index must be a non-negative integer: {task.index!r}")
    if not isinstance(task.dataset, str) or not task.dataset:
        raise ValueError("migration task dataset must be a non-empty string")
    if not isinstance(task.stem, str) or not task.stem:
        raise ValueError("migration task stem must be a non-empty string")
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
    content = b"[" + b",".join(_task_payload(task, "") for task in tasks) + b"]"
    return hashlib.sha256(content).hexdigest()


def write_migration_manifest(
    output_dir: Path,
    *,
    generation: str,
    scientific_output: Path,
    tasks: Sequence[MigrationImageTask],
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

    state_dir = phenotypic_cache_dir(output_root)
    state_dir.mkdir(parents=True, exist_ok=True)
    records_path = state_dir / _RECORDS_FILENAME
    offsets_path = state_dir / _OFFSETS_FILENAME
    manifest_path = state_dir / _MANIFEST_FILENAME
    offsets: list[int] = []
    with records_path.open("wb") as records:
        records.write(_RECORDS_HEADER)
        for task in normalized:
            payload = _task_payload(task, generation)
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
        inventory_digest=_inventory_digest(normalized),
        records_path=records_path.resolve(),
        offsets_path=offsets_path.resolve(),
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
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return manifest


def _read_manifest(manifest_path: Path) -> tuple[Path, MigrationManifest]:
    """Decode and validate one manifest header and all path boundaries."""
    header_path = Path(manifest_path).resolve()
    output_root = header_path.parent.parent
    if header_path.name != _MANIFEST_FILENAME or header_path.parent != phenotypic_cache_dir(
        output_root
    ):
        raise ValueError(f"not a canonical migration manifest path: {manifest_path}")
    try:
        raw = json.loads(header_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid migration manifest header: {manifest_path}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("invalid migration manifest header schema")
    expected = {
        "schema_version",
        "generation",
        "scientific_output",
        "task_count",
        "inventory_digest",
        "records_path",
        "offsets_path",
    }
    if set(raw) != expected or raw.get("schema_version") != _SCHEMA_VERSION:
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
    fields = ("scientific_output", "records_path", "offsets_path")
    if any(not isinstance(raw[name], str) or not Path(raw[name]).is_absolute() for name in fields):
        raise ValueError("migration manifest paths must be absolute")
    manifest = MigrationManifest(
        schema_version=_SCHEMA_VERSION,
        generation=generation,
        scientific_output=_checked_path(Path(raw["scientific_output"]), output_root),
        task_count=task_count,
        inventory_digest=digest,
        records_path=_checked_path(Path(raw["records_path"]), output_root),
        offsets_path=_checked_path(Path(raw["offsets_path"]), output_root),
    )
    if manifest.scientific_output != deliverables_dir(output_root):
        raise ValueError("migration manifest has non-canonical scientific output")
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
            if records.read(len(_RECORDS_HEADER)) != _RECORDS_HEADER:
                raise ValueError("migration records file has invalid magic/version")
            if offset < len(_RECORDS_HEADER):
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
    return _normalized_task(task, output_root)


def read_migration_task(manifest_path: Path, index: int) -> MigrationImageTask:
    """Read exactly one indexed migration task without parsing prior records.

    Args:
        manifest_path: Canonical migration manifest header path.
        index: Zero-based array index to load.

    Returns:
        The checksum-verified task at *index*.

    Raises:
        ValueError: If the header, offsets, frame, checksum, schema, or
            generation is invalid.
        IndexError: If *index* is outside the inventory.
    """
    output_root, manifest = _read_manifest(manifest_path)
    return _task_from_payload(
        _read_record_payload(manifest, index),
        output_root=output_root,
        manifest=manifest,
        index=index,
    )
