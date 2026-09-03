"""Explicit, root-only migration of persisted image provenance journals."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping

from phenotypic._core._provenance import (
    provenance_basename,
    validate_provenance_journal,
)
from phenotypic.sdk_ import CommitGuard, atomic_write_json

from ._cli_directory_scanner import _is_store_dir, scan_directory_structure
from ._cli_migrate_manifest import discover_migration_tasks
from phenotypic.sdk_.ngff_ import PhenotypicAttr, STORE_ROOT_JSON


_V1_JOURNAL_FIELDS = frozenset(
    {"schema_version", "status", "pipeline", "retry_base_length", "operations"}
)


@dataclass(frozen=True)
class ProvenanceMigrationTarget:
    """A classified migration target and its non-recursive store inventory."""

    kind: Literal["full_run", "direct_store", "process_tree"]
    root: Path
    stores: tuple[Path, ...]


@dataclass(frozen=True)
class ProvenanceUpgradeResult:
    """Outcome of inspecting and optionally upgrading one store root."""

    store_path: Path
    schema_before: int | None
    upgraded: bool


def _legacy_pipeline(value: object) -> dict[str, str] | None:
    """Return one canonical v2 pipeline identity from a v1 value."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"source_path", "sha256"}:
        raise ValueError("malformed provenance schema v1 pipeline identity")
    source_path = value["source_path"]
    digest = value["sha256"]
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("malformed provenance schema v1 pipeline source_path")
    basename = provenance_basename(source_path)
    if basename is None:
        raise ValueError("malformed provenance schema v1 pipeline source_path")
    return {"source_path": basename, "sha256": digest}


def _historical_version(
    operations: list[dict[str, Any]], phenotypic_block: Mapping[str, Any]
) -> str | None:
    """Recover a historical release without substituting this migrator."""
    if operations:
        first = operations[0].get("phenotypic_version")
        if isinstance(first, str) and first:
            return first
    root_version = phenotypic_block.get(PhenotypicAttr.PHENOTYPIC_VERSION)
    return root_version if isinstance(root_version, str) and root_version else None


def _upgrade_v1_journal(
    journal: Mapping[str, Any], phenotypic_block: Mapping[str, Any]
) -> dict[str, Any]:
    """Convert the exact persisted v1 shape to one legacy application."""
    if set(journal) != _V1_JOURNAL_FIELDS:
        raise ValueError("malformed provenance schema v1 journal fields")
    operations = journal["operations"]
    if not isinstance(operations, list) or not all(
        isinstance(operation, dict) for operation in operations
    ):
        raise ValueError("malformed provenance schema v1 operations")
    converted = {
        "schema_version": 2,
        "status": journal["status"],
        "original_filename": None,
        "applications": [
            {
                "sequence": 1,
                "kind": "legacy",
                "phenotypic_version": _historical_version(
                    operations, phenotypic_block
                ),
                "input_filename": None,
                "status": journal["status"],
                "pipeline": _legacy_pipeline(journal["pipeline"]),
                "retry_base_length": journal["retry_base_length"],
                "operations": deepcopy(operations),
            }
        ],
    }
    validate_provenance_journal(converted)
    return converted


def upgrade_store_provenance(
    store_path: Path,
    *,
    dry_run: bool = False,
    commit_guard: CommitGuard | None = None,
) -> ProvenanceUpgradeResult:
    """Inspect one root once and atomically upgrade only schema-v1 provenance."""
    store_path = Path(store_path)
    root_path = store_path / STORE_ROOT_JSON
    payload = json.loads(root_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"malformed Zarr store root: {store_path}")
    attributes = payload.get("attributes")
    if not isinstance(attributes, Mapping):
        return ProvenanceUpgradeResult(store_path, None, False)
    phenotypic_block = attributes.get(PhenotypicAttr.ROOT)
    if not isinstance(phenotypic_block, dict):
        return ProvenanceUpgradeResult(store_path, None, False)
    if PhenotypicAttr.PROVENANCE not in phenotypic_block:
        return ProvenanceUpgradeResult(store_path, None, False)
    journal = phenotypic_block[PhenotypicAttr.PROVENANCE]
    if not isinstance(journal, Mapping) or "schema_version" not in journal:
        raise ValueError(f"malformed PhenoTypic provenance root: {store_path}")
    schema_before = journal["schema_version"]
    if schema_before == 2:
        validate_provenance_journal(journal)
        return ProvenanceUpgradeResult(store_path, schema_before, False)
    if schema_before != 1:
        raise ValueError(
            f"unsupported provenance schema version {schema_before!r}; expected 1 or 2"
        )
    converted = _upgrade_v1_journal(journal, phenotypic_block)
    if not dry_run:
        phenotypic_block[PhenotypicAttr.PROVENANCE] = converted
        atomic_write_json(root_path, payload, commit_guard=commit_guard)
    return ProvenanceUpgradeResult(store_path, schema_before, True)


def _full_run_stores(root: Path) -> tuple[Path, ...]:
    """Return existing full-run stores through the canonical manifest scan."""
    if not (root / "results").is_dir():
        return ()
    return tuple(
        sorted(
            {
                task.store_path
                for task in discover_migration_tasks(root)
                if _is_store_dir(task.store_path)
                and (task.store_path / STORE_ROOT_JSON).is_file()
            }
        )
    )


def _process_tree_stores(root: Path) -> tuple[Path, ...]:
    """Return process-output stores without enumerating any store member."""
    try:
        datasets = scan_directory_structure(root)
    except (FileNotFoundError, ValueError):
        return ()
    return tuple(
        sorted(
            {
                path.resolve()
                for paths in datasets.values()
                for path in paths
                if _is_store_dir(path)
                and (path / STORE_ROOT_JSON).is_file()
            }
        )
    )


def provenance_migration_lifecycle_root(
    target: ProvenanceMigrationTarget,
) -> Path:
    """Return the fenced lifecycle root, externalizing direct-store state."""
    if target.kind != "direct_store":
        return target.root
    digest = hashlib.sha256(str(target.root).encode("utf-8")).hexdigest()
    return (
        target.root.parent
        / ".phenotypic"
        / "migration_targets"
        / digest
    )


def classify_provenance_migration_target(
    target: Path,
) -> ProvenanceMigrationTarget:
    """Classify a direct store, full run, or process-output tree by layout."""
    root = Path(target).resolve()
    if _is_store_dir(root) and (root / STORE_ROOT_JSON).is_file():
        return ProvenanceMigrationTarget("direct_store", root, (root,))
    if not root.is_dir():
        raise ValueError(f"provenance migration target is not a directory: {root}")
    has_full_layout = (root / "results").is_dir()
    full_stores = _full_run_stores(root) if has_full_layout else ()
    process_stores = _process_tree_stores(root)
    if has_full_layout and process_stores:
        raise ValueError(
            f"ambiguous provenance migration target has full-run and process stores: {root}"
        )
    if has_full_layout:
        return ProvenanceMigrationTarget("full_run", root, full_stores)
    if process_stores:
        return ProvenanceMigrationTarget("process_tree", root, process_stores)
    raise ValueError(
        f"no PhenoTypic OME-Zarr stores found for provenance migration: {root}"
    )


def _upgrade_outcome(
    store: Path,
    *,
    dry_run: bool,
    commit_guard: CommitGuard | None,
) -> tuple[ProvenanceUpgradeResult | None, tuple[Path, str] | None]:
    """Isolate one store failure from the rest of a local migration."""
    try:
        return (
            upgrade_store_provenance(
                store, dry_run=dry_run, commit_guard=commit_guard
            ),
            None,
        )
    except Exception as exc:  # noqa: BLE001 - one malformed store is reportable
        return None, (store, f"{type(exc).__name__}: {exc}")


def execute_provenance_migration(
    target: ProvenanceMigrationTarget,
    *,
    n_jobs: int,
    dry_run: bool,
    commit_guard: CommitGuard | None = None,
) -> tuple[tuple[ProvenanceUpgradeResult, ...], tuple[tuple[Path, str], ...]]:
    """Upgrade one classified inventory through the existing joblib pattern."""
    if n_jobs > 1 and len(target.stores) > 1:
        from joblib import Parallel, delayed

        outcomes = Parallel(n_jobs=n_jobs)(
            delayed(_upgrade_outcome)(
                store, dry_run=dry_run, commit_guard=commit_guard
            )
            for store in target.stores
        )
    else:
        outcomes = [
            _upgrade_outcome(
                store, dry_run=dry_run, commit_guard=commit_guard
            )
            for store in target.stores
        ]
    results = tuple(result for result, _ in outcomes if result is not None)
    failures = tuple(failure for _, failure in outcomes if failure is not None)
    return results, failures
