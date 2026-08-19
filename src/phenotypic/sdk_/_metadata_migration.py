"""Durable migration of historical metadata headers to the flat namespace.

This module is intentionally isolated from ordinary readers. Readers normalize
legacy spellings in memory; the functions here are the explicit mutation API
used by standalone callers and, through a private CLI facade, recompile.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
import pickle
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pandas as pd

from ._atomic_io import atomic_write_json
from ._io_constants import (
    DATASET_AGGREGATED_PARQUET,
    BundleLayout,
    deliverables_dir,
    file_fingerprint,
    resolve_processing_state_path,
)
from ._metadata_compatibility import LEGACY_HEADER_TO_CANONICAL
from ._metadata_helpers import (
    ensure_metadata_prefix,
    metadata_member_for_label,
    normalize_metadata_columns,
)

MigrationStatus: TypeAlias = Literal[
    "compatible", "migratable", "blocked", "applied", "rolled_back", "failed"
]
TargetKind: TypeAlias = Literal["csv", "parquet", "json", "hdf", "frame"]

# Version 3 adds a dynamic HDF ``rollback_fingerprint``. Version-2 HDF
# receipts cannot safely distinguish a valid semantic rollback from external
# byte-level changes. The receipt schema is global, so all version-2 receipts
# are explicitly rejected rather than partially or heuristically upgraded.
_RECEIPT_SCHEMA_VERSION = 3
_FLAT_METADATA_SCHEMA_VERSION = 2
_METADATA_SCHEMA_ATTR = "metadata_schema_version"
_HDF_SUFFIXES = frozenset({".h5", ".hdf5", ".hdf"})


def _absolute_path(path: str | Path) -> Path:
    """Return an absolute, lexically normalized path without following links."""
    return Path(os.path.abspath(os.fspath(path)))


def _require_safe_migration_path(
    path: str | Path,
    *,
    role: str,
    root: str | Path | None = None,
) -> Path:
    """Reject symlink components and optional containment escapes.

    ``Path.resolve()`` alone is unsafe for migration journals because it hides
    the fact that a lexical component such as ``.phenotypic`` was replaced by
    a link.  Migration paths are therefore required to already be their own
    resolved spelling.  This also rejects a broken final symlink.
    """
    candidate = _absolute_path(path)
    resolved = candidate.resolve(strict=False)
    if candidate.is_symlink() or resolved != candidate:
        raise ValueError(f"{role} contains a symlink component: {candidate}")
    if root is not None:
        boundary = _absolute_path(root)
        if boundary.is_symlink() or boundary.resolve(strict=False) != boundary:
            raise ValueError(
                f"{role} authoritative root contains a symlink component: {boundary}"
            )
        try:
            candidate.relative_to(boundary)
            resolved.relative_to(boundary)
        except ValueError as exc:
            raise ValueError(
                f"{role} escapes its authoritative root: {candidate}"
            ) from exc
    return candidate


@dataclass(frozen=True)
class MetadataMigrationTarget:
    """Immutable preflight description of one migration target."""

    path: str
    kind: TargetKind
    status: MigrationStatus
    source_fingerprint: str
    proposed_header_map: tuple[tuple[str, str], ...] = ()
    needs_metadata_marker: bool = False
    hdf_snapshot_fingerprint: str | None = None
    conflicts: tuple[str, ...] = ()
    mixed_table: bool = False


@dataclass(frozen=True)
class MetadataMigrationReport:
    """Immutable preflight result for a file, frame, or bundle."""

    source: str
    status: MigrationStatus
    source_fingerprint: str
    plan_fingerprint: str
    targets: tuple[MetadataMigrationTarget, ...]
    conflicts: tuple[str, ...] = ()

    @property
    def compatible_count(self) -> int:
        """Return the number of already-canonical targets."""
        return sum(target.status == "compatible" for target in self.targets)

    @property
    def migratable_count(self) -> int:
        """Return the number of targets requiring migration."""
        return sum(target.status == "migratable" for target in self.targets)

    @property
    def blocked_count(self) -> int:
        """Return the number of targets that cannot migrate losslessly."""
        return sum(target.status == "blocked" for target in self.targets)


@dataclass(frozen=True)
class MetadataMigrationResult:
    """Immutable outcome of migration or rollback."""

    status: MigrationStatus
    source: str
    source_fingerprint: str
    resulting_fingerprint: str | None
    plan_fingerprint: str
    receipt_path: Path | None
    migrated_targets: tuple[str, ...] = ()
    skipped_targets: tuple[str, ...] = ()
    blocked_targets: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()


def _sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _frame_fingerprint(frame: Any) -> str:
    """Return a content/dtype/order fingerprint without mutating ``frame``."""
    module = type(frame).__module__.split(".", maxsplit=1)[0]
    if module == "polars":
        frame = frame.to_pandas()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas or Polars DataFrame")
    digest = hashlib.sha256()
    digest.update(
        json.dumps([str(column) for column in frame.columns]).encode()
    )
    digest.update(json.dumps([str(dtype) for dtype in frame.dtypes]).encode())
    digest.update(pickle.dumps(frame, protocol=5))
    return f"sha256:{digest.hexdigest()}"


def _header_map(columns: Iterable[object]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for raw_column in columns:
        column = str(raw_column)
        canonical = ensure_metadata_prefix(column)
        if canonical != column:
            pairs.append((column, canonical))
    return tuple(sorted(set(pairs)))


def _known_header_map(
    columns: Iterable[object],
) -> tuple[tuple[str, str], ...]:
    """Return rewrites safe for a mixed measurement/metadata table."""
    pairs: list[tuple[str, str]] = []
    for raw_column in columns:
        column = str(raw_column)
        canonical = _stored_hdf_header_target(column)
        if canonical != column:
            pairs.append((column, canonical))
    return tuple(sorted(set(pairs)))


def _normalize_mixed_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize only known metadata aliases in a mixed measurement frame.

    The public normalizer intentionally treats every bare column in an external
    metadata file as metadata. A standalone clean master is different: it also
    contains measurements and locator columns, which must retain their names.
    Temporary canonical-looking placeholders let the shared coalescer retain
    its dtype/conflict guarantees without teaching it a second policy.
    """
    source_names = [str(column) for column in frame.columns]
    temporary_names: list[str] = []
    restored_names: dict[str, str] = {}
    occupied = set(source_names)
    for position, column in enumerate(source_names):
        canonical = _stored_hdf_header_target(column)
        if canonical != column or column.startswith("Metadata_"):
            temporary_names.append(canonical)
            continue
        placeholder = f"Metadata___preserved_mixed_column_{position}"
        while placeholder in occupied:
            placeholder += "_"
        occupied.add(placeholder)
        restored_names[placeholder] = column
        temporary_names.append(placeholder)
    prepared = frame.copy(deep=True)
    prepared.columns = temporary_names
    normalized = normalize_metadata_columns(prepared)
    normalized.columns = [
        restored_names.get(str(column), str(column))
        for column in normalized.columns
    ]
    return normalized


def _target_status(
    header_map: tuple[tuple[str, str], ...],
    conflicts: tuple[str, ...],
    *,
    needs_metadata_marker: bool = False,
) -> MigrationStatus:
    if conflicts:
        return "blocked"
    return (
        "migratable" if header_map or needs_metadata_marker else "compatible"
    )


def _preflight_frame(frame: Any, *, source: str) -> MetadataMigrationTarget:
    source_fingerprint = _frame_fingerprint(frame)
    header_map = _header_map(frame.columns)
    conflicts: tuple[str, ...] = ()
    try:
        normalize_metadata_columns(frame)
    except (TypeError, ValueError) as exc:
        conflicts = (str(exc),)
    return MetadataMigrationTarget(
        path=source,
        kind="frame",
        status=_target_status(header_map, conflicts),
        source_fingerprint=source_fingerprint,
        proposed_header_map=header_map,
        conflicts=conflicts,
    )


def _load_table(path: Path, kind: TargetKind) -> pd.DataFrame:
    if kind == "csv":
        # CSV has no persisted dtype schema. Reading as text prevents header-only
        # migration from converting values such as zero-padded plate IDs.
        return pd.read_csv(path, dtype=str, keep_default_na=False).replace(
            "", pd.NA
        )
    return pd.read_parquet(path)


def _preflight_table(
    path: Path, kind: TargetKind, *, mixed_table: bool = False
) -> MetadataMigrationTarget:
    fingerprint = file_fingerprint(path)
    try:
        frame = _load_table(path, kind)
        header_map = (
            _known_header_map(frame.columns)
            if mixed_table
            else _header_map(frame.columns)
        )
        if mixed_table:
            _normalize_mixed_table(frame)
        else:
            normalize_metadata_columns(frame)
        conflicts: tuple[str, ...] = ()
    except (
        Exception
    ) as exc:  # malformed input is a blocked target, not a crash
        header_map = ()
        conflicts = (f"{path}: {exc}",)
    return MetadataMigrationTarget(
        path=str(path),
        kind=kind,
        status=_target_status(header_map, conflicts),
        source_fingerprint=fingerprint,
        proposed_header_map=header_map,
        conflicts=conflicts,
        mixed_table=mixed_table,
    )


_OUTPUT_COLUMN_FIELDS = frozenset(
    {
        ("phenotypic.post._append_string", "AppendString", "column"),
        ("phenotypic.post._expand_metadata", "ExpandMetadata", "column"),
        ("phenotypic.post._expand_metadata", "ExpandMetadata", "labels"),
        ("phenotypic.post._merge_metadata", "MergeMetadata", "columns"),
        ("phenotypic.post._merge_metadata", "MergeMetadata", "label"),
        ("phenotypic.post._prepend_string", "PrependString", "column"),
    }
)


def _normalize_json_column_reference(
    value: Any,
) -> tuple[Any, set[tuple[str, str]]]:
    """Normalize metadata spellings inside one typed column-reference field."""
    mappings: set[tuple[str, str]] = set()
    if isinstance(value, list):
        normalized_items: list[Any] = []
        for item in value:
            normalized, child_maps = _normalize_json_column_reference(item)
            normalized_items.append(normalized)
            mappings.update(child_maps)
        return normalized_items, mappings
    if isinstance(value, tuple):
        normalized_items = []
        for item in value:
            normalized, child_maps = _normalize_json_column_reference(item)
            normalized_items.append(normalized)
            mappings.update(child_maps)
        return normalized_items, mappings
    if isinstance(value, str):
        target = _stored_hdf_header_target(value)
        if target != value:
            mappings.add((value, target))
        return target, mappings
    return value, mappings


def _serialized_class(class_name: str) -> type[Any] | None:
    """Resolve a public serialized class without importing custom code."""
    from phenotypic._core._pipeline_parts._serializable_pipeline import (
        SerializablePipeline,
    )

    candidate = SerializablePipeline._find_class_in_phenotypic(class_name)
    return candidate if isinstance(candidate, type) else None


def _is_column_reference_field(
    class_name: str, class_: type[Any], field: str
) -> bool:
    """Return whether one known serialized field carries column names."""
    from ._column_ref import _ColumnRefMarker

    if (class_.__module__, class_name, field) in _OUTPUT_COLUMN_FIELDS:
        return True
    model_field = getattr(class_, "model_fields", {}).get(field)
    return model_field is not None and any(
        isinstance(metadata, _ColumnRefMarker)
        for metadata in model_field.metadata
    )


def _normalize_known_envelope(
    envelope: Mapping[Any, Any], class_name: str, class_: type[Any]
) -> tuple[dict[Any, Any], set[tuple[str, str]]]:
    """Normalize marked fields of one recognized serialized class envelope."""
    mappings: set[tuple[str, str]] = set()
    normalized = dict(envelope)
    params = envelope.get("params")
    if isinstance(params, dict):
        normalized_params: dict[Any, Any] = {}
        for field, raw_value in params.items():
            if _is_column_reference_field(class_name, class_, str(field)):
                normalized_value, child_maps = (
                    _normalize_json_column_reference(raw_value)
                )
            else:
                normalized_value, child_maps = _normalize_nested_envelopes(
                    raw_value
                )
            normalized_params[field] = normalized_value
            mappings.update(child_maps)
        normalized["params"] = normalized_params
    config = envelope.get("config")
    if isinstance(config, dict):
        normalized_config, child_maps = _normalize_pipeline_envelope(config)
        normalized["config"] = normalized_config
        mappings.update(child_maps)
    return normalized, mappings


def _known_inline_plot_class(
    module_name: Any, qualname: Any
) -> type[Any] | None:
    """Resolve an inline plot class only from the PhenoTypic package."""
    if (
        not isinstance(module_name, str)
        or not isinstance(qualname, str)
        or not (
            module_name == "phenotypic"
            or module_name.startswith("phenotypic.")
        )
        or "<locals>" in qualname
    ):
        return None
    try:
        candidate: Any = importlib.import_module(module_name)
        for component in qualname.split("."):
            candidate = getattr(candidate, component)
    except (ImportError, AttributeError):
        return None
    if (
        not isinstance(candidate, type)
        or not candidate.__module__.startswith("phenotypic.")
        or not hasattr(candidate, "model_fields")
    ):
        return None
    return candidate


def _normalize_inline_plot(
    entry: Mapping[Any, Any],
) -> tuple[dict[Any, Any], set[tuple[str, str]]]:
    """Normalize marked parameters in one recognized inline plot binding."""
    normalized = dict(entry)
    inline = entry.get("inline")
    if not isinstance(inline, dict):
        return normalized, set()
    plot_class = _known_inline_plot_class(
        inline.get("module"), inline.get("qualname")
    )
    if plot_class is None:
        return normalized, set()
    params = inline.get("params")
    if not isinstance(params, dict):
        return normalized, set()
    normalized_params: dict[Any, Any] = {}
    mappings: set[tuple[str, str]] = set()
    for field, raw_value in params.items():
        if _is_column_reference_field(
            plot_class.__name__, plot_class, str(field)
        ):
            normalized_value, child_maps = _normalize_json_column_reference(
                raw_value
            )
            normalized_params[field] = normalized_value
            mappings.update(child_maps)
        else:
            normalized_params[field] = raw_value
    normalized_inline = dict(inline)
    normalized_inline["params"] = normalized_params
    normalized["inline"] = normalized_inline
    return normalized, mappings


def _normalize_nested_envelopes(
    value: Any,
) -> tuple[Any, set[tuple[str, str]]]:
    """Recurse only into recognized operation or pipeline envelopes."""
    mappings: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        if "inline" in value:
            return _normalize_inline_plot(value)
        class_name = value.get("class")
        if isinstance(class_name, str):
            class_ = _serialized_class(class_name)
            if class_ is not None:
                return _normalize_known_envelope(value, class_name, class_)
            # Unknown/custom envelopes are opaque except for explicitly nested,
            # independently recognized operation envelopes.
        if any(key in value for key in ("pipe_cfgs", "meas", "post")):
            return _normalize_pipeline_envelope(value)
        normalized: dict[Any, Any] = {}
        for key, raw_value in value.items():
            normalized_value, child_maps = _normalize_nested_envelopes(
                raw_value
            )
            normalized[key] = normalized_value
            mappings.update(child_maps)
        return normalized, mappings
    if isinstance(value, list):
        result: list[Any] = []
        for item in value:
            normalized_item, child_maps = _normalize_nested_envelopes(item)
            result.append(normalized_item)
            mappings.update(child_maps)
        return result, mappings
    return value, mappings


def _normalize_pipeline_envelope(
    payload: Mapping[Any, Any],
) -> tuple[dict[Any, Any], set[tuple[str, str]]]:
    """Normalize known entries in a serialized ImagePipeline envelope."""
    normalized: dict[Any, Any] = {}
    mappings: set[tuple[str, str]] = set()
    for key, value in payload.items():
        if key in {"pipe_cfgs", "meas", "post", "filters"} and isinstance(
            value, dict
        ):
            normalized_slot: dict[Any, Any] = {}
            for name, entry in value.items():
                normalized_entry, child_maps = _normalize_nested_envelopes(
                    entry
                )
                normalized_slot[name] = normalized_entry
                mappings.update(child_maps)
            normalized[key] = normalized_slot
        elif key in {"model", "qc", "plots"}:
            normalized_value, child_maps = _normalize_nested_envelopes(value)
            normalized[key] = normalized_value
            mappings.update(child_maps)
        else:
            normalized[key] = value
    return normalized, mappings


def _normalize_json_value(value: Any) -> tuple[Any, set[tuple[str, str]]]:
    """Normalize typed pipeline column references with class-aware traversal."""
    if not isinstance(value, dict):
        return value, set()
    return _normalize_pipeline_envelope(value)


def _preflight_json(path: Path) -> MetadataMigrationTarget:
    fingerprint = file_fingerprint(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _, mappings = _normalize_json_value(payload)
        header_map = tuple(sorted(mappings))
        conflicts: tuple[str, ...] = ()
    except Exception as exc:
        header_map = ()
        conflicts = (f"{path}: {exc}",)
    return MetadataMigrationTarget(
        path=str(path),
        kind="json",
        status=_target_status(header_map, conflicts),
        source_fingerprint=fingerprint,
        proposed_header_map=header_map,
        conflicts=conflicts,
    )


def _is_metadata_attr_group(group: Any) -> bool:
    name = group.name.rstrip("/").rsplit("/", maxsplit=1)[-1]
    if name in {"protected_metadata", "public_metadata"}:
        return True
    return name in {"protected", "public", "imported"} and (
        group.parent.name.rstrip("/").rsplit("/", maxsplit=1)[-1] == "metadata"
    )


def _metadata_root_for_attr_group(group: Any) -> Any:
    parent_name = group.parent.name.rstrip("/").rsplit("/", maxsplit=1)[-1]
    return group.parent.parent if parent_name == "metadata" else group.parent


def _hdf_values_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if (
        left_array.dtype != right_array.dtype
        or left_array.shape != right_array.shape
    ):
        return False
    try:
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left_array, right_array))


def _stored_hdf_header_target(header: str) -> str:
    """Return canonical spelling for exact legacy or bare known headers."""
    if header in LEGACY_HEADER_TO_CANONICAL:
        return LEGACY_HEADER_TO_CANONICAL[header]
    member = metadata_member_for_label(header)
    return member.value if member is not None else header


def _inspect_hdf(
    path: Path,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...], bool, str | None]:
    import h5py  # type: ignore[import-untyped]

    mappings: set[tuple[str, str]] = set()
    conflicts: list[str] = []
    needs_metadata_marker = False
    with h5py.File(path, "r") as handle:
        groups: list[Any] = []

        def collect(_name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Group) and _is_metadata_attr_group(obj):
                groups.append(obj)

        handle.visititems(collect)
        for group in groups:
            root = _metadata_root_for_attr_group(group)
            try:
                marker = int(root.attrs.get(_METADATA_SCHEMA_ATTR, 0))
            except (TypeError, ValueError):
                marker = 0
            needs_metadata_marker |= marker != _FLAT_METADATA_SCHEMA_VERSION
            aliases_by_target: dict[str, list[str]] = {}
            for raw_header in group.attrs:
                source_header = str(raw_header)
                canonical = _stored_hdf_header_target(source_header)
                aliases_by_target.setdefault(canonical, []).append(
                    source_header
                )
                if canonical != source_header:
                    mappings.add((source_header, canonical))
            for canonical, aliases in aliases_by_target.items():
                if len(aliases) < 2:
                    continue
                reference = aliases[0]
                for alias in aliases[1:]:
                    if not _hdf_values_equal(
                        group.attrs[reference], group.attrs[alias]
                    ):
                        conflicts.append(
                            f"{path}:{group.name} has conflicting attributes "
                            f"{reference!r} and {alias!r} converging on "
                            f"{canonical!r}"
                        )
    header_map = tuple(sorted(mappings))
    snapshot_fingerprint = (
        _hdf_snapshot_fingerprint(_read_hdf_rollback_snapshot(path))
        if header_map or needs_metadata_marker
        else None
    )
    return (
        header_map,
        tuple(conflicts),
        needs_metadata_marker,
        snapshot_fingerprint,
    )


def _preflight_hdf(path: Path) -> MetadataMigrationTarget:
    fingerprint = file_fingerprint(path)
    needs_metadata_marker = False
    snapshot_fingerprint: str | None = None
    try:
        (
            header_map,
            conflicts,
            needs_metadata_marker,
            snapshot_fingerprint,
        ) = _inspect_hdf(path)
    except Exception as exc:
        header_map = ()
        conflicts = (f"{path}: {exc}",)
    return MetadataMigrationTarget(
        path=str(path),
        kind="hdf",
        status=_target_status(
            header_map,
            conflicts,
            needs_metadata_marker=needs_metadata_marker,
        ),
        source_fingerprint=fingerprint,
        proposed_header_map=header_map,
        needs_metadata_marker=needs_metadata_marker,
        hdf_snapshot_fingerprint=snapshot_fingerprint,
        conflicts=conflicts,
    )


def _kind_for_file(path: Path) -> TargetKind:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    if suffix in _HDF_SUFFIXES:
        return "hdf"
    if ".json" in path.name.lower():
        return "json"
    raise ValueError(f"Unsupported metadata migration target: {path}")


def _preflight_file(
    path: Path, *, mixed_table: bool = False
) -> MetadataMigrationTarget:
    path = _require_safe_migration_path(path, role="Migration source")
    if not path.is_file():
        raise FileNotFoundError(path)
    kind = _kind_for_file(path)
    if kind in {"csv", "parquet"}:
        return _preflight_table(path, kind, mixed_table=mixed_table)
    if kind == "json":
        return _preflight_json(path)
    return _preflight_hdf(path)


def _discover_bundle_targets(layout: BundleLayout) -> tuple[Path, ...]:
    """Return authoritative sources only, never external metadata copies."""
    if layout.output_root is not None:
        bundle_root = _require_safe_migration_path(
            layout.output_root, role="Bundle root"
        )
        deliverables_root = deliverables_dir(bundle_root)
        results_root = bundle_root / "results"
        _require_safe_migration_path(
            deliverables_root, role="Bundle deliverables", root=bundle_root
        )
        _require_safe_migration_path(
            results_root, role="Bundle results", root=bundle_root
        )
    else:
        # The standalone bundle root itself may have been supplied through a
        # symlink. Resolve it once, then reject symlinks below that boundary.
        deliverables_root = _require_safe_migration_path(
            layout.deliverables_base, role="Standalone bundle root"
        )
        bundle_root = deliverables_root
        results_root = None

    def validated_candidate(candidate: Path, root: Path, role: str) -> Path:
        if candidate.is_symlink():
            raise ValueError(
                f"Bundle-owned {role} cannot be a symlink: {candidate}"
            )
        safe = _require_safe_migration_path(
            candidate, role=f"Bundle-owned {role}", root=root
        )
        if not safe.is_file():
            raise FileNotFoundError(safe)
        return safe

    def validated_directory(candidate: Path, root: Path, role: str) -> Path:
        if candidate.is_symlink():
            raise ValueError(
                f"Bundle-owned {role} cannot be a symlink: {candidate}"
            )
        safe = _require_safe_migration_path(
            candidate, role=f"Bundle-owned {role}", root=root
        )
        if not safe.is_dir():
            raise ValueError(
                f"Bundle-owned {role} is not a directory: {candidate}"
            )
        return safe

    def legacy_root_pipeline() -> Path | None:
        """Return the contained processing-state pipeline fallback, if any."""
        if layout.output_root is None:
            return None
        state_path = resolve_processing_state_path(bundle_root)
        if not state_path.is_file():
            return None
        safe_state = validated_candidate(
            state_path, bundle_root, "processing state"
        )
        try:
            payload = json.loads(safe_state.read_text(encoding="utf-8"))
            original = payload.get("pipeline_path")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(original, str) or not original:
            return None
        candidate = bundle_root / Path(original).name
        if not (candidate.exists() or candidate.is_symlink()):
            return None
        return validated_candidate(
            candidate, bundle_root, "legacy root pipeline"
        )

    targets: list[Path] = []
    pipeline = layout.resolved_pipeline_config_path
    if pipeline.exists() or pipeline.is_symlink():
        targets.append(
            validated_candidate(pipeline, deliverables_root, "pipeline")
        )
    root_pipeline = legacy_root_pipeline()
    if root_pipeline is not None:
        targets.append(root_pipeline)
    if layout.output_root is not None:
        if results_root is not None and results_root.is_dir():
            for dataset_candidate in sorted(results_root.iterdir()):
                if dataset_candidate.is_symlink():
                    raise ValueError(
                        "Bundle-owned dataset cannot be a symlink: "
                        f"{dataset_candidate}"
                    )
                if not dataset_candidate.is_dir():
                    continue
                dataset_root = validated_directory(
                    dataset_candidate,
                    root=results_root,
                    role="dataset",
                )
                hdf_candidate = dataset_root / "hdf"
                if hdf_candidate.exists() or hdf_candidate.is_symlink():
                    hdf_root = validated_directory(
                        hdf_candidate, dataset_root, "HDF directory"
                    )
                    for path in sorted(hdf_root.rglob("*")):
                        if path.is_symlink():
                            raise ValueError(
                                f"Bundle-owned HDF cannot be a symlink: {path}"
                            )
                        if (
                            path.is_file()
                            and path.suffix.lower() in _HDF_SUFFIXES
                        ):
                            targets.append(
                                validated_candidate(path, hdf_root, "HDF")
                            )
                measurements_candidate = dataset_root / "measurements"
                if not (
                    measurements_candidate.exists()
                    or measurements_candidate.is_symlink()
                ):
                    continue
                validated_measurements = validated_directory(
                    measurements_candidate,
                    dataset_root,
                    "measurements directory",
                )
                individual = [
                    validated_candidate(
                        path, validated_measurements, "measurement"
                    )
                    for path in sorted(
                        validated_measurements.glob("*.parquet")
                    )
                    if not path.name.startswith(("_", "."))
                ]
                targets.extend(individual)
                aggregate = validated_measurements / DATASET_AGGREGATED_PARQUET
                if not individual and (
                    aggregate.exists() or aggregate.is_symlink()
                ):
                    targets.append(
                        validated_candidate(
                            aggregate,
                            validated_measurements,
                            "sole aggregate measurement",
                        )
                    )
    else:
        # A portable standalone bundle has no per-image HDF authority. Its
        # clean master archive is therefore the authoritative table source.
        for path in (layout.master_parquet, layout.master_csv):
            if path.exists() or path.is_symlink():
                targets.append(
                    validated_candidate(path, deliverables_root, "master")
                )
    return tuple(dict.fromkeys(targets))


def _bundle_target_is_mixed_table(layout: BundleLayout, path: Path) -> bool:
    """Return whether a bundle table also carries non-metadata measurements."""
    if path.suffix.lower() not in {".csv", ".parquet"}:
        return False
    if layout.output_root is None:
        return True
    results_root = _absolute_path(layout.output_root) / "results"
    try:
        path.relative_to(results_root)
    except ValueError:
        return False
    return True


def _report_from_targets(
    source: str, targets: tuple[MetadataMigrationTarget, ...]
) -> MetadataMigrationReport:
    conflicts = tuple(
        conflict for target in targets for conflict in target.conflicts
    )
    if conflicts:
        status: MigrationStatus = "blocked"
    elif any(target.status == "migratable" for target in targets):
        status = "migratable"
    else:
        status = "compatible"
    plan_data = [
        {
            "path": target.path,
            "kind": target.kind,
            "fingerprint": target.source_fingerprint,
            "header_map": target.proposed_header_map,
            "needs_metadata_marker": target.needs_metadata_marker,
            "hdf_snapshot_fingerprint": target.hdf_snapshot_fingerprint,
            "conflicts": target.conflicts,
            "mixed_table": target.mixed_table,
        }
        for target in targets
    ]
    plan_fingerprint = _sha256_bytes(
        json.dumps(plan_data, sort_keys=True, separators=(",", ":")).encode()
    )
    source_fingerprint = _sha256_bytes(
        json.dumps(
            [(target.path, target.source_fingerprint) for target in targets],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    return MetadataMigrationReport(
        source=source,
        status=status,
        source_fingerprint=source_fingerprint,
        plan_fingerprint=plan_fingerprint,
        targets=targets,
        conflicts=conflicts,
    )


def preflight_metadata_schema(source: Any) -> MetadataMigrationReport:
    """Inspect a frame, supported file, or bundle without changing it.

    Args:
        source: pandas/Polars frame, supported file path, run-output path,
            standalone deliverables path, or resolved :class:`BundleLayout`.

    Returns:
        Immutable migration plan and compatibility status.
    """
    module = type(source).__module__.split(".", maxsplit=1)[0]
    if isinstance(source, pd.DataFrame) or module == "polars":
        target = _preflight_frame(source, source=f"<{module}-frame>")
        return _report_from_targets(target.path, (target,))
    if isinstance(source, BundleLayout):
        layout = source
        targets = tuple(
            _preflight_file(
                path,
                mixed_table=_bundle_target_is_mixed_table(layout, path),
            )
            for path in _discover_bundle_targets(layout)
        )
        return _report_from_targets(str(layout.deliverables_base), targets)
    path = _require_safe_migration_path(
        source, role="Metadata preflight source"
    )
    if path.is_file():
        target = _preflight_file(path)
        return _report_from_targets(str(path), (target,))
    layout = BundleLayout.detect(path)
    targets = tuple(
        _preflight_file(
            item,
            mixed_table=_bundle_target_is_mixed_table(layout, item),
        )
        for item in _discover_bundle_targets(layout)
    )
    return _report_from_targets(str(path), targets)


def _receipt_dir(source: Path, *, bundle: bool) -> Path:
    if bundle:
        return source / ".phenotypic" / "metadata_migration"
    return source.parent / ".metadata_migration"


def _receipt_path(
    source: Path, plan_fingerprint: str, *, bundle: bool
) -> Path:
    digest = plan_fingerprint.removeprefix("sha256:")[:16]
    return (
        _receipt_dir(source, bundle=bundle) / f"metadata-schema-{digest}.json"
    )


def _new_receipt(
    report: MetadataMigrationReport, *, bundle_root: Path | None
) -> dict[str, Any]:
    return {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "scope": "bundle" if bundle_root is not None else "file",
        "bundle_root": str(bundle_root) if bundle_root is not None else None,
        "state": "prepared",
        "source": report.source,
        "source_fingerprint": report.source_fingerprint,
        "plan_fingerprint": report.plan_fingerprint,
        "targets": [
            {
                **asdict(target),
                "state": "skipped"
                if target.status == "compatible"
                else "pending",
                "post_fingerprint": None,
                "rollback_fingerprint": None,
                "temp_path": None,
                "backup_path": None,
                "hdf_snapshot": None,
            }
            for target in report.targets
        ],
    }


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path = _require_safe_migration_path(path, role="Migration receipt")
    _ensure_directory_durable(path.parent)
    atomic_write_json(path, dict(receipt), sort_keys=True)
    _fsync_directory(path.parent)


def _json_safe_hdf_item(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__hdf_bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, np.generic):
        return _json_safe_hdf_item(value.item())
    if isinstance(value, list):
        return [_json_safe_hdf_item(item) for item in value]
    return value


def _restore_hdf_item(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__hdf_bytes__"}:
        return base64.b64decode(value["__hdf_bytes__"])
    if isinstance(value, list):
        return [_restore_hdf_item(item) for item in value]
    return value


def _encode_hdf_attr(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.kind in {"O", "U"}:
        return {
            "encoding": "list",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "value": _json_safe_hdf_item(array.tolist()),
        }
    return {
        "encoding": "bytes",
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "value": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def _decode_hdf_attr(payload: Mapping[str, Any]) -> Any:
    dtype: np.dtype[Any] = np.dtype(str(payload["dtype"]))
    shape = tuple(int(size) for size in payload["shape"])
    if payload["encoding"] == "list":
        array = np.asarray(
            _restore_hdf_item(payload["value"]), dtype=dtype
        ).reshape(shape)
    else:
        raw = base64.b64decode(str(payload["value"]))
        array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return array.item() if array.shape == () else array


def _decoded_snapshot_attr(payload: Any) -> Any:
    """Decode one strictly shaped HDF receipt attribute."""
    if not isinstance(payload, Mapping):
        raise ValueError("Invalid HDF attribute snapshot")
    if set(payload) != {"encoding", "dtype", "shape", "value"}:
        raise ValueError("Incomplete HDF attribute snapshot")
    if payload.get("encoding") not in {"bytes", "list"}:
        raise ValueError("Invalid HDF attribute snapshot encoding")
    shape = payload.get("shape")
    if not isinstance(shape, list) or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 0
        for size in shape
    ):
        raise ValueError("Invalid HDF attribute snapshot shape")
    try:
        return _decode_hdf_attr(payload)
    except Exception as exc:
        raise ValueError("Invalid HDF attribute snapshot value") from exc


def _validate_hdf_snapshot_semantics(
    path: Path,
    snapshot: Any,
    target: MetadataMigrationTarget,
    *,
    phase: Literal["original", "migrated"],
) -> None:
    """Validate compact rollback evidence against the HDF metadata topology."""
    import h5py  # type: ignore[import-untyped]

    if not isinstance(snapshot, list):
        raise ValueError("HDF migration receipt lacks a snapshot")
    if (
        target.hdf_snapshot_fingerprint is None
        or _hdf_snapshot_fingerprint(snapshot)
        != target.hdf_snapshot_fingerprint
    ):
        raise ValueError("HDF snapshot does not match its preflight binding")
    planned = target.proposed_header_map
    if len(set(planned)) != len(planned) or tuple(sorted(planned)) != planned:
        raise ValueError("HDF migration header map must be unique and sorted")
    planned_by_source: dict[str, str] = {}
    for source, canonical in planned:
        if (
            source == canonical
            or _stored_hdf_header_target(source) != canonical
            or source in planned_by_source
        ):
            raise ValueError("HDF migration header map is not canonical")
        planned_by_source[source] = canonical

    attr_records: dict[str, Mapping[str, Any]] = {}
    marker_records: dict[str, Mapping[str, Any]] = {}
    for raw_record in snapshot:
        if not isinstance(raw_record, Mapping):
            raise ValueError("Invalid HDF snapshot record")
        group_name = raw_record.get("group")
        if not isinstance(group_name, str) or not group_name.startswith("/"):
            raise ValueError("Invalid HDF snapshot group")
        if raw_record.get("marker") is True:
            if set(raw_record) != {
                "group",
                "marker",
                "marker_existed",
                "marker_value",
            }:
                raise ValueError("Incomplete HDF marker snapshot")
            if group_name in marker_records:
                raise ValueError("Duplicate HDF marker snapshot")
            marker_existed = raw_record.get("marker_existed")
            if not isinstance(marker_existed, bool):
                raise ValueError("Invalid HDF marker snapshot state")
            marker_value = raw_record.get("marker_value")
            if marker_existed:
                _decoded_snapshot_attr(marker_value)
            elif marker_value is not None:
                raise ValueError("Absent HDF marker cannot carry a value")
            marker_records[group_name] = raw_record
            continue
        if set(raw_record) != {"group", "attributes", "affected"}:
            raise ValueError("Incomplete HDF metadata attribute snapshot")
        if group_name in attr_records:
            raise ValueError("Duplicate HDF metadata group snapshot")
        attributes = raw_record.get("attributes")
        affected = raw_record.get("affected")
        if not isinstance(attributes, Mapping) or not all(
            isinstance(key, str) for key in attributes
        ):
            raise ValueError("Invalid HDF metadata attribute snapshot")
        if (
            not isinstance(affected, list)
            or not all(isinstance(key, str) for key in affected)
            or affected != sorted(set(affected))
        ):
            raise ValueError(
                "HDF affected attributes must be unique and sorted"
            )
        represented = {
            source: canonical
            for source, canonical in planned
            if source in attributes
        }
        if not represented:
            raise ValueError(
                "HDF snapshot group contains no planned legacy header"
            )
        expected_affected = sorted(
            {key for pair in represented.items() for key in pair}
        )
        if affected != expected_affected or not set(attributes).issubset(
            affected
        ):
            raise ValueError(
                "HDF snapshot attributes exceed the planned header map"
            )
        for source, canonical in represented.items():
            source_value = _decoded_snapshot_attr(attributes[source])
            if canonical in attributes and not _hdf_values_equal(
                source_value, _decoded_snapshot_attr(attributes[canonical])
            ):
                raise ValueError(
                    "HDF snapshot contains conflicting convergent aliases"
                )
        for payload in attributes.values():
            _decoded_snapshot_attr(payload)
        attr_records[group_name] = raw_record

    represented_sources = {
        source
        for record in attr_records.values()
        for source in planned_by_source
        if source in cast(Mapping[str, Any], record["attributes"])
    }
    if represented_sources != set(planned_by_source):
        raise ValueError("HDF snapshot does not cover the planned header map")

    with h5py.File(path, "r") as handle:
        metadata_groups: dict[str, Any] = {}

        def collect(_name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Group) and _is_metadata_attr_group(obj):
                metadata_groups[obj.name] = obj

        handle.visititems(collect)
        expected_roots = {
            _metadata_root_for_attr_group(group).name
            for group in metadata_groups.values()
        }
        if set(marker_records) != expected_roots:
            raise ValueError(
                "HDF marker snapshots do not match metadata roots"
            )
        if not set(attr_records).issubset(metadata_groups):
            raise ValueError(
                "HDF snapshot group is not a metadata attribute group"
            )

        for group_name, record in attr_records.items():
            group = metadata_groups[group_name]
            attributes = cast(Mapping[str, Any], record["attributes"])
            affected = cast(list[str], record["affected"])
            if phase == "original":
                for key in affected:
                    if (key in group.attrs) != (key in attributes):
                        raise ValueError(
                            "HDF original attribute state is incomplete"
                        )
                    if key in attributes and not _hdf_values_equal(
                        group.attrs[key],
                        _decoded_snapshot_attr(attributes[key]),
                    ):
                        raise ValueError(
                            "HDF original attribute snapshot does not match"
                        )
            else:
                for source in planned_by_source:
                    if source not in attributes:
                        continue
                    canonical = planned_by_source[source]
                    if source in group.attrs or canonical not in group.attrs:
                        raise ValueError(
                            "HDF migrated attribute state does not match"
                        )
                    if not _hdf_values_equal(
                        group.attrs[canonical],
                        _decoded_snapshot_attr(attributes[source]),
                    ):
                        raise ValueError(
                            "HDF migrated attribute value does not match"
                        )

        for root_name, record in marker_records.items():
            root = handle[root_name]
            marker_existed = cast(bool, record["marker_existed"])
            marker_value = record["marker_value"]
            if phase == "migrated":
                try:
                    marker = int(root.attrs[_METADATA_SCHEMA_ATTR])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "HDF migrated metadata marker is invalid"
                    ) from exc
                if marker != _FLAT_METADATA_SCHEMA_VERSION:
                    raise ValueError("HDF migrated metadata marker is invalid")
            else:
                if (_METADATA_SCHEMA_ATTR in root.attrs) != marker_existed:
                    raise ValueError(
                        "HDF original metadata marker state is incomplete"
                    )
                if marker_existed and not _hdf_values_equal(
                    root.attrs[_METADATA_SCHEMA_ATTR],
                    _decoded_snapshot_attr(marker_value),
                ):
                    raise ValueError(
                        "HDF original metadata marker does not match"
                    )


def _hdf_inventory(
    path: Path, excluded_attrs: set[tuple[str, str]]
) -> tuple[dict[str, str], dict[str, str]]:
    import h5py  # type: ignore[import-untyped]

    datasets: dict[str, str] = {}
    attrs: dict[str, str] = {}
    with h5py.File(path, "r") as handle:

        def inspect(name: str, obj: Any) -> None:
            object_path = f"/{name}" if name else "/"
            if isinstance(obj, h5py.Dataset):
                digest = hashlib.sha256()
                digest.update(obj.dtype.str.encode())
                digest.update(json.dumps(obj.shape).encode())
                value = obj[()]
                if np.asarray(value).dtype.kind == "O":
                    digest.update(repr(value).encode())
                else:
                    digest.update(np.asarray(value).tobytes())
                datasets[object_path] = digest.hexdigest()
            for key in obj.attrs:
                if (object_path, str(key)) in excluded_attrs:
                    continue
                attrs[f"{object_path}@{key}"] = json.dumps(
                    _encode_hdf_attr(obj.attrs[key]), sort_keys=True
                )

        inspect("", handle)
        handle.visititems(inspect)
    return datasets, attrs


def _read_hdf_rollback_snapshot(path: Path) -> list[dict[str, Any]]:
    """Capture the exact compact metadata state needed for HDF rollback."""
    import h5py  # type: ignore[import-untyped]

    snapshot: list[dict[str, Any]] = []
    roots: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        groups: list[Any] = []

        def collect(_name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Group) and _is_metadata_attr_group(obj):
                groups.append(obj)

        handle.visititems(collect)
        for group in groups:
            root = _metadata_root_for_attr_group(group)
            roots[root.name] = root
            mappings = [
                (source_header, _stored_hdf_header_target(source_header))
                for source_header in list(group.attrs)
                if _stored_hdf_header_target(source_header) != source_header
            ]
            affected = {header for pair in mappings for header in pair}
            if affected:
                snapshot.append(
                    {
                        "group": group.name,
                        "attributes": {
                            key: _encode_hdf_attr(group.attrs[key])
                            for key in affected
                            if key in group.attrs
                        },
                        "affected": sorted(affected),
                    }
                )
        for root in roots.values():
            marker_existed = _METADATA_SCHEMA_ATTR in root.attrs
            snapshot.append(
                {
                    "group": root.name,
                    "marker": True,
                    "marker_existed": marker_existed,
                    "marker_value": (
                        _encode_hdf_attr(root.attrs[_METADATA_SCHEMA_ATTR])
                        if marker_existed
                        else None
                    ),
                }
            )
    return snapshot


def _hdf_snapshot_fingerprint(snapshot: list[dict[str, Any]]) -> str:
    """Bind exact group topology, attribute presence/values, and markers."""
    return _sha256_bytes(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    )


def _migrate_hdf_copy(source: Path, temp: Path) -> list[dict[str, Any]]:
    import h5py  # type: ignore[import-untyped]

    snapshot = _read_hdf_rollback_snapshot(source)
    shutil.copy2(source, temp)
    excluded: set[tuple[str, str]] = {
        (str(record["group"]), str(key))
        for record in snapshot
        for key in (
            [_METADATA_SCHEMA_ATTR]
            if record.get("marker")
            else record["affected"]
        )
    }
    roots: dict[str, Any] = {}
    with h5py.File(temp, "r+") as handle:
        groups: list[Any] = []

        def collect(_name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Group) and _is_metadata_attr_group(obj):
                groups.append(obj)

        handle.visititems(collect)
        for group in groups:
            root = _metadata_root_for_attr_group(group)
            roots[root.name] = root
            mappings = [
                (source_header, _stored_hdf_header_target(source_header))
                for source_header in list(group.attrs)
                if _stored_hdf_header_target(source_header) != source_header
            ]
            for source_header, canonical in mappings:
                if canonical not in group.attrs:
                    group.attrs[canonical] = group.attrs[source_header]
                del group.attrs[source_header]
        for root in roots.values():
            root.attrs[_METADATA_SCHEMA_ATTR] = _FLAT_METADATA_SCHEMA_VERSION
        handle.flush()
    before_datasets, before_attrs = _hdf_inventory(source, excluded)
    after_datasets, after_attrs = _hdf_inventory(temp, excluded)
    if before_datasets != after_datasets or before_attrs != after_attrs:
        raise ValueError(
            f"HDF validation failed for {source}; non-target content changed"
        )
    return snapshot


def _write_migrated_copy(
    target: Mapping[str, Any], temp: Path
) -> dict[str, Any]:
    source = Path(str(target["path"]))
    kind = cast(TargetKind, str(target["kind"]))
    if kind in {"csv", "parquet"}:
        source_frame = _load_table(source, kind)
        frame = (
            _normalize_mixed_table(source_frame)
            if bool(target.get("mixed_table"))
            else normalize_metadata_columns(source_frame)
        )
        if kind == "csv":
            frame.to_csv(temp, index=False)
        else:
            frame.to_parquet(temp, index=False)
        shutil.copystat(source, temp)
        return {}
    if kind == "json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        normalized, _ = _normalize_json_value(payload)
        temp.write_text(
            json.dumps(
                normalized, indent=2, sort_keys=False, ensure_ascii=False
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.copystat(source, temp)
        return {}
    snapshot = _migrate_hdf_copy(source, temp)
    if _hdf_snapshot_fingerprint(snapshot) != target.get(
        "hdf_snapshot_fingerprint"
    ):
        raise ValueError(
            f"HDF metadata state changed after preflight: {source}"
        )
    return {"hdf_snapshot": snapshot}


def _copy_backup(
    source: Path, receipt_path: Path, *, source_fingerprint: str
) -> Path:
    source = _require_safe_migration_path(
        source, role="Migration backup source"
    )
    receipt_path = _require_safe_migration_path(
        receipt_path, role="Migration receipt"
    )
    backup_dir = receipt_path.parent / "backups"
    _require_safe_migration_path(
        backup_dir, role="Migration backup directory", root=receipt_path.parent
    )
    _ensure_directory_durable(backup_dir)
    digest = source_fingerprint.removeprefix("sha256:")[:16]
    backup = backup_dir / f"{source.name}.{digest}.bak"
    if backup.exists():
        if not backup.is_file() or backup.is_symlink():
            raise ValueError(
                f"Migration backup is not a regular file: {backup}"
            )
        if file_fingerprint(backup) != source_fingerprint:
            raise ValueError(
                f"Existing migration backup has wrong fingerprint: {backup}"
            )
        return backup
    handle = tempfile.NamedTemporaryFile(
        dir=backup_dir,
        prefix=f".{backup.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp = Path(handle.name)
    handle.close()
    try:
        shutil.copy2(source, temp)
        _fsync_file(temp)
        if file_fingerprint(temp) != source_fingerprint:
            raise ValueError(
                f"Prepared migration backup has wrong fingerprint: {temp}"
            )
        os.replace(temp, backup)
        _fsync_directory(backup_dir)
        if file_fingerprint(backup) != source_fingerprint:
            raise ValueError(
                f"Published migration backup has wrong fingerprint: {backup}"
            )
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return backup


def _new_temp_path(source: Path) -> Path:
    source = _require_safe_migration_path(source, role="Migration temp source")
    handle = tempfile.NamedTemporaryFile(
        dir=source.parent,
        prefix=f".{source.name}.metadata-",
        suffix=".tmp",
        delete=False,
    )
    path = Path(handle.name)
    handle.close()
    return path


def _fsync_file(path: Path) -> None:
    """Flush a prepared file before it becomes eligible for publication."""
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _ensure_directory_durable(path: Path) -> None:
    """Create a directory chain and durably publish each new entry."""
    path = _require_safe_migration_path(path, role="Migration directory")
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if cursor.is_symlink() or not cursor.is_dir():
        raise ValueError(f"Unsafe migration directory ancestor: {cursor}")
    for directory in reversed(missing):
        directory.mkdir()
        _fsync_directory(directory)
        _fsync_directory(directory.parent)
    check = path
    while check != cursor:
        if check.is_symlink():
            raise ValueError(
                f"Migration directory cannot be a symlink: {check}"
            )
        check = check.parent


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes where POSIX supports directory fsync."""
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_temp(temp: Path, target: Path) -> None:
    """Atomically publish one prepared migration target."""
    os.replace(temp, target)
    _fsync_directory(target.parent)


def _receipt_target(record: Mapping[str, Any]) -> MetadataMigrationTarget:
    """Parse and type-check the immutable planning fields of one receipt target."""
    kind = str(record.get("kind"))
    status = str(record.get("status"))
    if kind not in {"csv", "parquet", "json", "hdf"}:
        raise ValueError(f"Invalid migration receipt target kind: {kind!r}")
    if status not in {"compatible", "migratable"}:
        raise ValueError(
            f"Invalid migration receipt target status: {status!r}"
        )
    raw_map = record.get("proposed_header_map", [])
    if not isinstance(raw_map, (list, tuple)) or any(
        not isinstance(pair, (list, tuple))
        or len(pair) != 2
        or not all(isinstance(item, str) for item in pair)
        for pair in raw_map
    ):
        raise ValueError("Invalid migration receipt header map")
    raw_conflicts = record.get("conflicts", [])
    if not isinstance(raw_conflicts, (list, tuple)) or not all(
        isinstance(item, str) for item in raw_conflicts
    ):
        raise ValueError("Invalid migration receipt conflicts")
    fingerprint = record.get("source_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.startswith(
        "sha256:"
    ):
        raise ValueError("Invalid migration receipt source fingerprint")
    hdf_snapshot_fingerprint = record.get("hdf_snapshot_fingerprint")
    if hdf_snapshot_fingerprint is not None and (
        not isinstance(hdf_snapshot_fingerprint, str)
        or not hdf_snapshot_fingerprint.startswith("sha256:")
    ):
        raise ValueError("Invalid HDF snapshot preflight binding")
    if kind != "hdf" and hdf_snapshot_fingerprint is not None:
        raise ValueError("Non-HDF target cannot carry an HDF snapshot binding")
    if (
        kind == "hdf"
        and status == "migratable"
        and hdf_snapshot_fingerprint is None
    ):
        raise ValueError("Migratable HDF target lacks its snapshot binding")
    return MetadataMigrationTarget(
        path=str(record.get("path")),
        kind=cast(TargetKind, kind),
        status=cast(MigrationStatus, status),
        source_fingerprint=fingerprint,
        proposed_header_map=tuple(
            (str(pair[0]), str(pair[1])) for pair in raw_map
        ),
        needs_metadata_marker=bool(record.get("needs_metadata_marker", False)),
        hdf_snapshot_fingerprint=cast(str | None, hdf_snapshot_fingerprint),
        conflicts=tuple(raw_conflicts),
        mixed_table=bool(record.get("mixed_table", False)),
    )


def _validate_receipt(
    receipt_path: Path,
    receipt: Mapping[str, Any],
    *,
    expected_plan_fingerprint: str | None = None,
) -> None:
    """Strictly validate a receipt and every path before any mutation."""
    if receipt.get("schema_version") != _RECEIPT_SCHEMA_VERSION:
        raise ValueError("Unsupported metadata migration receipt schema")
    scope = receipt.get("scope")
    if scope not in {"file", "bundle"}:
        raise ValueError("Invalid metadata migration receipt scope")
    if receipt.get("state") not in {
        "prepared",
        "applied",
        "failed",
        "rolled_back",
    }:
        raise ValueError("Invalid metadata migration receipt state")
    plan_fingerprint = receipt.get("plan_fingerprint")
    source_fingerprint = receipt.get("source_fingerprint")
    if not isinstance(
        plan_fingerprint, str
    ) or not plan_fingerprint.startswith("sha256:"):
        raise ValueError("Invalid metadata migration plan fingerprint")
    if not isinstance(
        source_fingerprint, str
    ) or not source_fingerprint.startswith("sha256:"):
        raise ValueError("Invalid metadata migration source fingerprint")
    if (
        expected_plan_fingerprint is not None
        and plan_fingerprint != expected_plan_fingerprint
    ):
        raise ValueError(
            "Migration receipt plan fingerprint does not match request"
        )

    source_text = receipt.get("source")
    if not isinstance(source_text, str):
        raise ValueError("Invalid metadata migration receipt source")
    expected_paths: tuple[Path, ...]
    if scope == "file":
        source = _require_safe_migration_path(
            source_text, role="Migration source"
        )
        if receipt.get("bundle_root") is not None:
            raise ValueError(
                "File migration receipt cannot declare a bundle root"
            )
        expected_paths = (source,)
        receipt_source = str(source)
        expected_receipt = _receipt_path(
            source, plan_fingerprint, bundle=False
        )
    else:
        raw_root = receipt.get("bundle_root")
        if not isinstance(raw_root, str):
            raise ValueError("Bundle migration receipt is missing its root")
        root = _require_safe_migration_path(
            raw_root, role="Migration bundle root"
        )
        full_deliverables = _require_safe_migration_path(
            deliverables_dir(root),
            role="Migration bundle deliverables",
            root=root,
        )
        if source_text == str(full_deliverables):
            layout = BundleLayout(
                deliverables_base=full_deliverables, output_root=root
            )
        elif source_text == str(root):
            layout = BundleLayout(deliverables_base=root, output_root=None)
        else:
            raise ValueError(
                "Bundle receipt source is not rooted in its bundle"
            )
        expected_paths = _discover_bundle_targets(layout)
        receipt_source = str(layout.deliverables_base)
        expected_receipt = _receipt_path(root, plan_fingerprint, bundle=True)
    if source_text != receipt_source:
        raise ValueError(
            "Migration receipt source does not match its resolved scope"
        )
    journal_root = root if scope == "bundle" else source.parent
    receipt_path = _require_safe_migration_path(
        receipt_path,
        role="Migration receipt",
        root=journal_root,
    )
    expected_receipt = _require_safe_migration_path(
        expected_receipt,
        role="Expected migration receipt",
        root=journal_root,
    )
    if receipt_path != expected_receipt:
        raise ValueError(
            "Migration receipt is outside its authoritative journal"
        )

    raw_targets = receipt.get("targets")
    if not isinstance(raw_targets, list):
        raise ValueError("Invalid metadata migration receipt targets")
    targets = tuple(_receipt_target(record) for record in raw_targets)
    target_paths = tuple(
        _require_safe_migration_path(
            target.path,
            role="Migration target",
            root=(root if scope == "bundle" else source.parent),
        )
        for target in targets
    )
    if target_paths != expected_paths:
        raise ValueError("Migration receipt target set is not authoritative")
    if len(set(target_paths)) != len(target_paths):
        raise ValueError("Migration receipt contains duplicate targets")
    for target, expected_path in zip(targets, expected_paths, strict=True):
        if target.path != str(expected_path):
            raise ValueError("Migration receipt target path is not canonical")
        if target.kind != _kind_for_file(expected_path):
            raise ValueError(
                "Migration receipt target kind does not match its path"
            )

    rebuilt = _report_from_targets(source_text, targets)
    if rebuilt.plan_fingerprint != plan_fingerprint:
        raise ValueError("Migration receipt plan content has been altered")
    if rebuilt.source_fingerprint != source_fingerprint:
        raise ValueError(
            "Migration receipt source fingerprint has been altered"
        )

    backups_dir = receipt_path.parent / "backups"
    for record, target, target_path in zip(
        raw_targets, targets, target_paths, strict=True
    ):
        state = record.get("state")
        if state not in {
            "pending",
            "prepared",
            "applied",
            "skipped",
            "rolled_back",
        }:
            raise ValueError("Invalid migration receipt target state")
        if state == "skipped" and record.get("status") != "compatible":
            raise ValueError(
                "Only compatible migration targets may be skipped"
            )
        if state != "skipped" and record.get("status") != "migratable":
            raise ValueError(
                "Compatible migration targets must remain skipped"
            )
        source_fp = str(record["source_fingerprint"])
        post_fp = record.get("post_fingerprint")
        if post_fp is not None and (
            not isinstance(post_fp, str) or not post_fp.startswith("sha256:")
        ):
            raise ValueError("Invalid migration receipt post fingerprint")
        if state in {"prepared", "applied"} and post_fp is None:
            raise ValueError(
                "Prepared migration target lacks a post fingerprint"
            )
        rollback_fp = record.get("rollback_fingerprint")
        if rollback_fp is not None and (
            not isinstance(rollback_fp, str)
            or not rollback_fp.startswith("sha256:")
        ):
            raise ValueError("Invalid migration receipt rollback fingerprint")
        if target.kind != "hdf" and rollback_fp is not None:
            raise ValueError(
                "Non-HDF target cannot carry a rollback fingerprint"
            )
        if target.kind == "hdf":
            if state == "rolled_back" and rollback_fp is None:
                raise ValueError(
                    "Rolled-back HDF target lacks its fingerprint"
                )
            if rollback_fp is not None and state not in {
                "pending",
                "prepared",
                "rolled_back",
            }:
                raise ValueError(
                    "HDF rollback fingerprint has an invalid lifecycle state"
                )

        backup_text = record.get("backup_path")
        if backup_text is not None:
            backup = _require_safe_migration_path(
                str(backup_text),
                role="Migration backup",
                root=backups_dir,
            )
            if backup.parent != backups_dir:
                raise ValueError(
                    "Migration backup is outside the receipt backup directory"
                )
            if record.get("kind") == "hdf":
                raise ValueError(
                    "HDF migration receipts cannot contain file backups"
                )
            if not backup.is_file() or file_fingerprint(backup) != source_fp:
                raise ValueError(
                    "Migration backup is missing or has the wrong fingerprint"
                )
        elif (
            state in {"prepared", "applied", "rolled_back"}
            and record.get("kind") != "hdf"
        ):
            raise ValueError("Prepared migration target lacks its backup")

        temp_text = record.get("temp_path")
        temp: Path | None = None
        if temp_text is not None:
            temp = _require_safe_migration_path(
                str(temp_text),
                role="Migration temp",
                root=target_path.parent,
            )
            expected_prefix = f".{target_path.name}.metadata-"
            if (
                temp.parent != target_path.parent
                or not temp.name.startswith(expected_prefix)
                or temp.suffix != ".tmp"
            ):
                raise ValueError("Prepared migration temp path is unsafe")
            if temp.exists() and (
                not temp.is_file()
                or post_fp is None
                or file_fingerprint(temp) != post_fp
            ):
                raise ValueError(
                    "Prepared migration temp has the wrong fingerprint"
                )
        if state == "prepared" and temp is None:
            raise ValueError("Prepared migration target lacks its temp path")
        if state not in {"pending", "prepared"} and temp is not None:
            raise ValueError("Published migration target retains a temp path")

        current_fp = file_fingerprint(target_path)
        allowed_current = {source_fp}
        if rollback_fp is not None and state in {
            "pending",
            "prepared",
            "rolled_back",
        }:
            allowed_current.add(rollback_fp)
        if state in {"prepared", "applied"} and post_fp is not None:
            allowed_current.add(post_fp)
        if state == "applied":
            allowed_current = {cast(str, post_fp)}
        if state == "rolled_back":
            allowed_current = (
                {cast(str, rollback_fp)}
                if target.kind == "hdf"
                else {source_fp}
            )
        if current_fp not in allowed_current:
            raise ValueError(
                f"Migration receipt target fingerprint changed: {target_path}"
            )
        if target.kind == "hdf":
            snapshot = record.get("hdf_snapshot")
            if state == "skipped" or (
                state == "pending" and rollback_fp is None
            ):
                if snapshot is not None:
                    raise ValueError(
                        "Unprepared HDF migration target has a snapshot"
                    )
            else:
                phase: Literal["original", "migrated"] = (
                    "original"
                    if current_fp in {source_fp, rollback_fp}
                    else "migrated"
                )
                _validate_hdf_snapshot_semantics(
                    target_path,
                    snapshot,
                    target,
                    phase=phase,
                )
                if state == "prepared" and temp is not None and temp.exists():
                    _validate_hdf_snapshot_semantics(
                        temp,
                        snapshot,
                        target,
                        phase="migrated",
                    )
        elif record.get("hdf_snapshot") is not None:
            raise ValueError(
                "Non-HDF migration target contains an HDF snapshot"
            )


def validated_published_metadata_migration_targets(
    receipt_path: str | Path,
) -> tuple[tuple[Path, str, str], ...]:
    """Return receipt-certified artifact fingerprint transitions.

    A target is returned only after the complete receipt and its current disk
    state pass the migration engine's normal validation. ``prepared`` targets
    whose atomic replacement reached disk before the receipt state update are
    included when their current bytes match the prepared post fingerprint.

    Args:
        receipt_path: Durable metadata-migration receipt to validate.

    Returns:
        Tuples of ``(path, source_fingerprint, post_fingerprint)`` for
        published target transitions.

    Raises:
        OSError: The receipt or one of its targets cannot be read.
        ValueError: The receipt, its authority scope, or current target bytes
            fail validation.
    """
    path = _require_safe_migration_path(
        receipt_path, role="Published migration receipt"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("Metadata migration receipt must be a JSON object")
    _validate_receipt(path, receipt)

    transitions: list[tuple[Path, str, str]] = []
    for record in receipt["targets"]:
        if record.get("state") not in {"prepared", "applied"}:
            continue
        post_fingerprint = record.get("post_fingerprint")
        if not isinstance(post_fingerprint, str):
            continue
        target = Path(str(record["path"]))
        if file_fingerprint(target) != post_fingerprint:
            continue
        transitions.append(
            (
                target,
                str(record["source_fingerprint"]),
                post_fingerprint,
            )
        )
    return tuple(transitions)


def _prepare_receipt_target(
    target: dict[str, Any], receipt_path: Path
) -> None:
    source = Path(target["path"])
    accepted_source_fingerprints = {str(target["source_fingerprint"])}
    rollback_fingerprint = target.get("rollback_fingerprint")
    if isinstance(rollback_fingerprint, str):
        accepted_source_fingerprints.add(rollback_fingerprint)
    if file_fingerprint(source) not in accepted_source_fingerprints:
        raise ValueError(f"Source changed after preflight: {source}")
    temp = _new_temp_path(source)
    try:
        extra = _write_migrated_copy(target, temp)
        _fsync_file(temp)
        post_fingerprint = file_fingerprint(temp)
        target["temp_path"] = str(temp)
        target["post_fingerprint"] = post_fingerprint
        target.update(extra)
        if target["kind"] != "hdf":
            target["backup_path"] = str(
                _copy_backup(
                    source,
                    receipt_path,
                    source_fingerprint=str(target["source_fingerprint"]),
                )
            )
        target["state"] = "prepared"
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _receipt_validation_failure(
    receipt_path: Path, receipt: Mapping[str, Any], exc: Exception
) -> MetadataMigrationResult:
    """Return a non-mutating failure for an untrusted receipt."""
    return MetadataMigrationResult(
        status="failed",
        source=str(receipt.get("source", receipt_path)),
        source_fingerprint=str(receipt.get("source_fingerprint", "")),
        resulting_fingerprint=None,
        plan_fingerprint=str(receipt.get("plan_fingerprint", "")),
        receipt_path=receipt_path,
        blocked_targets=(str(exc),),
        conflicts=(str(exc),),
    )


def _apply_receipt(
    receipt_path: Path,
    receipt: dict[str, Any],
    *,
    expected_plan_fingerprint: str | None = None,
) -> MetadataMigrationResult:
    try:
        _validate_receipt(
            receipt_path,
            receipt,
            expected_plan_fingerprint=expected_plan_fingerprint,
        )
    except Exception as exc:
        return _receipt_validation_failure(receipt_path, receipt, exc)
    migrated: list[str] = []
    skipped: list[str] = []
    try:
        for target in receipt["targets"]:
            path = Path(target["path"])
            state = target["state"]
            if state == "skipped":
                if file_fingerprint(path) != target["source_fingerprint"]:
                    raise ValueError(
                        f"Skipped migration target changed: {path}"
                    )
                skipped.append(str(path))
                continue
            if state == "applied":
                if file_fingerprint(path) != target["post_fingerprint"]:
                    raise ValueError(
                        f"Applied migration target changed: {path}"
                    )
                migrated.append(str(path))
                continue
            if state == "rolled_back":
                expected_rollback = (
                    target.get("rollback_fingerprint")
                    if target["kind"] == "hdf"
                    else target["source_fingerprint"]
                )
                if file_fingerprint(path) != expected_rollback:
                    raise ValueError(
                        f"Rolled-back migration target changed: {path}"
                    )
                target["state"] = "pending"
                target["post_fingerprint"] = None
                target["temp_path"] = None
                if target["kind"] != "hdf":
                    target["hdf_snapshot"] = None
                _write_receipt(receipt_path, receipt)
            if state == "prepared":
                current = file_fingerprint(path)
                if current == target["post_fingerprint"]:
                    target["state"] = "applied"
                    target["temp_path"] = None
                    target["rollback_fingerprint"] = None
                    _write_receipt(receipt_path, receipt)
                    migrated.append(str(path))
                    continue
                if current != target["source_fingerprint"]:
                    raise ValueError(
                        f"Prepared migration target changed: {path}"
                    )
                temp = Path(target["temp_path"])
                if (
                    not temp.is_file()
                    or file_fingerprint(temp) != target["post_fingerprint"]
                ):
                    target["state"] = "pending"
            if target["state"] == "pending":
                _prepare_receipt_target(target, receipt_path)
                _write_receipt(receipt_path, receipt)
            temp = Path(target["temp_path"])
            accepted_prepublication = {str(target["source_fingerprint"])}
            if isinstance(target.get("rollback_fingerprint"), str):
                accepted_prepublication.add(
                    str(target["rollback_fingerprint"])
                )
            if file_fingerprint(path) not in accepted_prepublication:
                raise ValueError(
                    f"Migration target changed before publication: {path}"
                )
            _publish_temp(temp, path)
            if file_fingerprint(path) != target["post_fingerprint"]:
                raise ValueError(
                    f"Published migration target failed validation: {path}"
                )
            target["state"] = "applied"
            target["temp_path"] = None
            target["rollback_fingerprint"] = None
            _write_receipt(receipt_path, receipt)
            migrated.append(str(path))
        receipt["state"] = "applied"
        _write_receipt(receipt_path, receipt)
        result_fingerprint = _sha256_bytes(
            json.dumps(
                [
                    (
                        target["path"],
                        target.get("post_fingerprint")
                        or target["source_fingerprint"],
                    )
                    for target in receipt["targets"]
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        return MetadataMigrationResult(
            status="applied",
            source=receipt["source"],
            source_fingerprint=receipt["source_fingerprint"],
            resulting_fingerprint=result_fingerprint,
            plan_fingerprint=receipt["plan_fingerprint"],
            receipt_path=receipt_path,
            migrated_targets=tuple(migrated),
            skipped_targets=tuple(skipped),
        )
    except Exception as exc:
        receipt["state"] = "failed"
        receipt["failure"] = str(exc)
        _write_receipt(receipt_path, receipt)
        return MetadataMigrationResult(
            status="failed",
            source=receipt["source"],
            source_fingerprint=receipt["source_fingerprint"],
            resulting_fingerprint=None,
            plan_fingerprint=receipt["plan_fingerprint"],
            receipt_path=receipt_path,
            migrated_targets=tuple(migrated),
            skipped_targets=tuple(skipped),
            blocked_targets=(str(exc),),
            conflicts=(str(exc),),
        )


def _blocked_result(
    report: MetadataMigrationReport,
) -> MetadataMigrationResult:
    return MetadataMigrationResult(
        status="blocked",
        source=report.source,
        source_fingerprint=report.source_fingerprint,
        resulting_fingerprint=None,
        plan_fingerprint=report.plan_fingerprint,
        receipt_path=None,
        blocked_targets=tuple(
            target.path
            for target in report.targets
            if target.status == "blocked"
        ),
        conflicts=report.conflicts,
    )


def _compatible_result(
    report: MetadataMigrationReport,
) -> MetadataMigrationResult:
    return MetadataMigrationResult(
        status="compatible",
        source=report.source,
        source_fingerprint=report.source_fingerprint,
        resulting_fingerprint=report.source_fingerprint,
        plan_fingerprint=report.plan_fingerprint,
        receipt_path=None,
        skipped_targets=tuple(target.path for target in report.targets),
    )


def _find_file_receipt(
    source: Path, expected_source_fingerprint: str
) -> tuple[Path, dict[str, Any]] | None:
    """Find a validated single-file receipt after the target was replaced.

    A post-replace crash changes the current preflight fingerprint and therefore
    its plan-derived receipt name. The immutable original target path and source
    fingerprint stored in the prepared journal are the stable resume identity.
    """
    receipt_dir = _receipt_dir(source, bundle=False)
    _require_safe_migration_path(
        receipt_dir, role="Migration receipt directory", root=source.parent
    )
    if not receipt_dir.is_dir():
        return None
    for candidate in sorted(receipt_dir.glob("metadata-schema-*.json")):
        try:
            receipt = json.loads(candidate.read_text(encoding="utf-8"))
            targets = receipt.get("targets", [])
            if (
                receipt.get("schema_version") != _RECEIPT_SCHEMA_VERSION
                or len(targets) != 1
                or Path(str(targets[0]["path"])).resolve() != source
            ):
                continue
            accepted_fingerprints = {
                str(targets[0].get("source_fingerprint")),
                str(receipt.get("source_fingerprint")),
                str(receipt.get("plan_fingerprint")),
            }
            if expected_source_fingerprint not in accepted_fingerprints:
                continue
            if targets[0].get("state") not in {
                "pending",
                "prepared",
                "applied",
                "skipped",
                "rolled_back",
            }:
                continue
            _validate_receipt(candidate, receipt)
            return candidate, receipt
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            continue
    return None


def migrate_metadata_file(
    path: str | Path, *, expected_source_fingerprint: str
) -> MetadataMigrationResult:
    """Copy-on-write migrate one supported file after optimistic preflight."""
    source = _require_safe_migration_path(path, role="Migration source")
    resumable = _find_file_receipt(source, expected_source_fingerprint)
    if resumable is not None:
        return _apply_receipt(*resumable)
    report = preflight_metadata_schema(source)
    receipt_path = _receipt_path(source, report.plan_fingerprint, bundle=False)
    target_fingerprint = report.targets[0].source_fingerprint
    if expected_source_fingerprint not in {
        target_fingerprint,
        report.source_fingerprint,
    }:
        mismatch = MetadataMigrationReport(
            source=report.source,
            status="blocked",
            source_fingerprint=report.source_fingerprint,
            plan_fingerprint=report.plan_fingerprint,
            targets=report.targets,
            conflicts=("Source fingerprint does not match preflight",),
        )
        return _blocked_result(mismatch)
    if report.status == "blocked":
        return _blocked_result(report)
    if report.status == "compatible":
        return _compatible_result(report)
    receipt = _new_receipt(report, bundle_root=None)
    _write_receipt(receipt_path, receipt)
    return _apply_receipt(receipt_path, receipt)


def _resolve_bundle(
    source: str | Path | BundleLayout,
) -> tuple[BundleLayout, Path]:
    if isinstance(source, BundleLayout):
        layout = source
        root = layout.output_root or layout.deliverables_base
        return layout, _require_safe_migration_path(
            root, role="Migration bundle root"
        )
    requested = _require_safe_migration_path(
        source, role="Migration bundle source"
    )
    layout = BundleLayout.detect(requested)
    root = layout.output_root or layout.deliverables_base
    return layout, _require_safe_migration_path(
        root, role="Migration bundle root"
    )


def migrate_metadata_bundle(
    source: str | Path | BundleLayout, *, expected_plan_fingerprint: str
) -> MetadataMigrationResult:
    """Migrate authoritative sources in a full or standalone bundle."""
    layout, root = _resolve_bundle(source)
    requested_receipt = _receipt_path(
        root, expected_plan_fingerprint, bundle=True
    )
    _require_safe_migration_path(
        requested_receipt,
        role="Migration bundle receipt",
        root=root,
    )
    if requested_receipt.is_file():
        receipt = json.loads(requested_receipt.read_text(encoding="utf-8"))
        return _apply_receipt(
            requested_receipt,
            receipt,
            expected_plan_fingerprint=expected_plan_fingerprint,
        )
    report = preflight_metadata_schema(layout)
    if expected_plan_fingerprint != report.plan_fingerprint:
        mismatch = MetadataMigrationReport(
            source=report.source,
            status="blocked",
            source_fingerprint=report.source_fingerprint,
            plan_fingerprint=report.plan_fingerprint,
            targets=report.targets,
            conflicts=("Migration plan fingerprint does not match preflight",),
        )
        return _blocked_result(mismatch)
    if report.status == "blocked":
        return _blocked_result(report)
    if report.status == "compatible":
        return _compatible_result(report)
    receipt_path = _receipt_path(root, report.plan_fingerprint, bundle=True)
    receipt = _new_receipt(report, bundle_root=root)
    _write_receipt(receipt_path, receipt)
    return _apply_receipt(
        receipt_path,
        receipt,
        expected_plan_fingerprint=expected_plan_fingerprint,
    )


def _rollback_hdf(
    source: Path,
    snapshot: list[dict[str, Any]],
    target: MetadataMigrationTarget,
) -> None:
    import h5py  # type: ignore[import-untyped]

    temp = _new_temp_path(source)
    try:
        shutil.copy2(source, temp)
        excluded: set[tuple[str, str]] = set()
        with h5py.File(temp, "r+") as handle:
            for record in snapshot:
                group = handle[record["group"]]
                if record.get("marker"):
                    excluded.add((group.name, _METADATA_SCHEMA_ATTR))
                    if record["marker_existed"]:
                        group.attrs[_METADATA_SCHEMA_ATTR] = _decode_hdf_attr(
                            record["marker_value"]
                        )
                    elif _METADATA_SCHEMA_ATTR in group.attrs:
                        del group.attrs[_METADATA_SCHEMA_ATTR]
                    continue
                affected = [str(key) for key in record["affected"]]
                excluded.update((group.name, key) for key in affected)
                for key in affected:
                    if key in group.attrs:
                        del group.attrs[key]
                for key, value in record["attributes"].items():
                    group.attrs[key] = _decode_hdf_attr(value)
            handle.flush()
        before_datasets, before_attrs = _hdf_inventory(source, excluded)
        after_datasets, after_attrs = _hdf_inventory(temp, excluded)
        if before_datasets != after_datasets or before_attrs != after_attrs:
            raise ValueError(f"HDF rollback validation failed for {source}")
        _validate_hdf_snapshot_semantics(
            temp,
            snapshot,
            target,
            phase="original",
        )
        _fsync_file(temp)
        _publish_temp(temp, source)
        _validate_hdf_snapshot_semantics(
            source,
            snapshot,
            target,
            phase="original",
        )
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def rollback_metadata_migration(
    receipt_path: str | Path,
) -> MetadataMigrationResult:
    """Restore every applied target recorded by a migration receipt."""
    try:
        path = _require_safe_migration_path(
            receipt_path, role="Migration rollback receipt"
        )
    except Exception as exc:
        unsafe_path = _absolute_path(receipt_path)
        return _receipt_validation_failure(unsafe_path, {}, exc)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    try:
        _validate_receipt(path, receipt)
    except Exception as exc:
        return _receipt_validation_failure(path, receipt, exc)
    rolled_back: list[str] = []
    try:
        for target in reversed(receipt["targets"]):
            if target["state"] in {"skipped", "rolled_back", "pending"}:
                continue
            source = Path(target["path"])
            current = file_fingerprint(source)
            prepared_origin_fingerprint = (
                target.get("rollback_fingerprint")
                if target["kind"] == "hdf"
                and isinstance(target.get("rollback_fingerprint"), str)
                else target["source_fingerprint"]
            )
            if (
                target["state"] == "prepared"
                and current == prepared_origin_fingerprint
            ):
                temp_path = target.get("temp_path")
                if temp_path:
                    Path(temp_path).unlink(missing_ok=True)
                target["state"] = "rolled_back"
                target["temp_path"] = None
                if target["kind"] == "hdf":
                    target["rollback_fingerprint"] = current
                _write_receipt(path, receipt)
                continue
            if current != target["post_fingerprint"]:
                raise ValueError(
                    f"Cannot rollback changed migration target: {source}"
                )
            if target["kind"] == "hdf":
                _rollback_hdf(
                    source,
                    target["hdf_snapshot"] or [],
                    _receipt_target(target),
                )
                target["rollback_fingerprint"] = file_fingerprint(source)
            else:
                backup = Path(target["backup_path"])
                if (
                    not backup.is_file()
                    or file_fingerprint(backup) != target["source_fingerprint"]
                ):
                    raise ValueError(
                        f"Migration backup is missing or changed: {backup}"
                    )
                temp = _new_temp_path(source)
                try:
                    shutil.copy2(backup, temp)
                    _fsync_file(temp)
                    _publish_temp(temp, source)
                except BaseException:
                    temp.unlink(missing_ok=True)
                    raise
            if (
                target["kind"] != "hdf"
                and file_fingerprint(source) != target["source_fingerprint"]
            ):
                raise ValueError(f"Rollback fingerprint mismatch: {source}")
            target["state"] = "rolled_back"
            rolled_back.append(str(source))
            _write_receipt(path, receipt)
        receipt["state"] = "rolled_back"
        _write_receipt(path, receipt)
        return MetadataMigrationResult(
            status="rolled_back",
            source=receipt["source"],
            source_fingerprint=receipt["source_fingerprint"],
            resulting_fingerprint=_sha256_bytes(
                json.dumps(
                    [
                        (
                            target["path"],
                            file_fingerprint(Path(target["path"])),
                        )
                        for target in receipt["targets"]
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ),
            plan_fingerprint=receipt["plan_fingerprint"],
            receipt_path=path,
            migrated_targets=tuple(reversed(rolled_back)),
        )
    except Exception as exc:
        receipt["state"] = "failed"
        receipt["failure"] = str(exc)
        _write_receipt(path, receipt)
        return MetadataMigrationResult(
            status="failed",
            source=receipt["source"],
            source_fingerprint=receipt["source_fingerprint"],
            resulting_fingerprint=None,
            plan_fingerprint=receipt["plan_fingerprint"],
            receipt_path=path,
            migrated_targets=tuple(reversed(rolled_back)),
            blocked_targets=(str(exc),),
            conflicts=(str(exc),),
        )


__all__ = [
    "MetadataMigrationReport",
    "MetadataMigrationResult",
    "MetadataMigrationTarget",
    "migrate_metadata_bundle",
    "migrate_metadata_file",
    "preflight_metadata_schema",
    "rollback_metadata_migration",
]
