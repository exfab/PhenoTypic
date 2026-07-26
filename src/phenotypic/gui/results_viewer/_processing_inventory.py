"""External persistent inventory cache for immutable processing products."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from phenotypic.gui.results_viewer._discovery_contracts import (
    DiscoveryPhase,
    OutputDiscoveryCancellation,
    OutputDiscoveryProgressCallback,
    report_discovery_progress,
)
from phenotypic.gui.results_viewer._output_consistency import (
    OutputConsistencyReport,
)
from phenotypic.sdk_ import (
    DIR_OVERLAYS,
    DIR_RESULTS,
    BundleLayout,
    source_cache_key,
)

_CACHE_SCHEMA_VERSION = 2
_PROGRESS_INTERVAL = 256
logger = logging.getLogger(__name__)

InventoryEntryKind = Literal["file", "directory", "missing"]


@dataclass(frozen=True)
class ProcessingInventoryEntry:
    """Path/type/size/time identity of one immutable processing product."""

    relative_path: str
    kind: InventoryEntryKind
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class ProcessingInventory:
    """One verified processing-product inventory."""

    entries: tuple[ProcessingInventoryEntry, ...]
    fingerprint: str
    cache_hit: bool


def processing_inventory_cache_path(
    source_root: Path,
    *,
    cache_root: Path,
) -> Path:
    """Return the stable external inventory-cache path for one output."""
    source = Path(source_root).resolve()
    owner = Path(cache_root).resolve()
    if owner == source or owner.is_relative_to(source):
        raise ValueError(
            "Viewer cache root must be external to the selected output: "
            f"cache_root={owner!s}, output={source!s}"
    )
    key = source_cache_key(source, "processing-inventory-v1")
    return owner / f"{key}.processing-inventory.json"


def load_or_scan_processing_inventory(
    layout: BundleLayout,
    *,
    source_root: Path,
    cache_root: Path,
    consistency: OutputConsistencyReport,
    cancellation: OutputDiscoveryCancellation,
    progress: OutputDiscoveryProgressCallback | None,
) -> ProcessingInventory:
    """Reuse a valid terminal inventory or scan a fresh read-only snapshot.

    Persistent reuse is deliberately restricted to coherent terminal outputs.
    Active, incomplete, and contradictory outputs are scanned on every
    discovery and never update a persistent cache record.
    """
    cache_path = processing_inventory_cache_path(
        source_root,
        cache_root=cache_root,
    )
    if consistency.cache_reusable:
        cached = _load_cached_inventory(
            cache_path,
            source_root=source_root,
            consistency=consistency,
        )
        if cached is not None and _inventory_is_current(
            cached,
            source_root=source_root,
            cancellation=cancellation,
            progress=progress,
            phase="inventory",
        ):
            report_discovery_progress(
                progress,
                phase="inventory",
                detail="Reused unchanged terminal processing inventory.",
                completed=len(cached.entries),
                total=len(cached.entries),
                cache_hit=True,
            )
            return ProcessingInventory(
                entries=cached.entries,
                fingerprint=cached.fingerprint,
                cache_hit=True,
            )

    entries = _scan_processing_inventory(
        layout,
        source_root=source_root,
        cancellation=cancellation,
        progress=progress,
    )
    inventory = ProcessingInventory(
        entries=entries,
        fingerprint=_inventory_fingerprint(entries),
        cache_hit=False,
    )
    cancellation.raise_if_cancelled()
    if consistency.cache_reusable:
        try:
            _persist_inventory(
                cache_path,
                source_root=source_root,
                consistency=consistency,
                inventory=inventory,
            )
        except OSError:
            logger.warning(
                "Could not persist external processing inventory at %s",
                cache_path,
                exc_info=True,
            )
    return inventory


def inventory_is_current(
    inventory: ProcessingInventory,
    *,
    source_root: Path,
    cancellation: OutputDiscoveryCancellation,
    progress: OutputDiscoveryProgressCallback | None,
) -> bool:
    """Return whether every inventoried path retains its captured metadata."""
    return _inventory_is_current(
        inventory,
        source_root=source_root,
        cancellation=cancellation,
        progress=progress,
        phase="verifying",
    )


def _scan_processing_inventory(
    layout: BundleLayout,
    *,
    source_root: Path,
    cancellation: OutputDiscoveryCancellation,
    progress: OutputDiscoveryProgressCallback | None,
) -> tuple[ProcessingInventoryEntry, ...]:
    """Stat processing-product trees without reading their file contents."""
    candidates: dict[str, Path] = {}

    def _add(path: Path) -> None:
        try:
            relative = path.resolve(strict=False).relative_to(
                source_root.resolve()
            )
        except ValueError:
            relative = Path(os.path.relpath(path, source_root))
        candidates[relative.as_posix()] = path

    _add(layout.master_parquet)
    report_discovery_progress(
        progress,
        phase="inventory",
        detail="Scanning processing-product paths.",
    )
    overlays_root = layout.deliverables_base / DIR_OVERLAYS
    _add(overlays_root)
    if overlays_root.exists():
        for path in overlays_root.rglob("*"):
            cancellation.raise_if_cancelled()
            _add(path)
    results_root = (
        layout.output_root / DIR_RESULTS
        if layout.output_root is not None
        else None
    )
    if results_root is not None:
        _add(results_root)
    if results_root is not None and results_root.is_dir():
        for path in results_root.rglob("*"):
            cancellation.raise_if_cancelled()
            _add(path)

    entries: list[ProcessingInventoryEntry] = []
    for index, (relative_path, path) in enumerate(
        sorted(candidates.items()),
        start=1,
    ):
        cancellation.raise_if_cancelled()
        try:
            stat = path.stat()
        except OSError:
            entries.append(
                ProcessingInventoryEntry(
                    relative_path=relative_path,
                    kind="missing",
                    size=0,
                    mtime_ns=0,
                    ctime_ns=0,
                )
            )
            continue
        kind: InventoryEntryKind = (
            "directory" if path.is_dir() else "file"
        )
        entries.append(
            ProcessingInventoryEntry(
                relative_path=relative_path,
                kind=kind,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns,
            )
        )
        if index % _PROGRESS_INTERVAL == 0:
            report_discovery_progress(
                progress,
                phase="inventory",
                detail="Recording processing-product metadata.",
                completed=index,
                total=len(candidates),
            )
    report_discovery_progress(
        progress,
        phase="inventory",
        detail="Processing inventory captured.",
        completed=len(entries),
        total=len(entries),
    )
    return tuple(entries)


def _inventory_is_current(
    inventory: ProcessingInventory,
    *,
    source_root: Path,
    cancellation: OutputDiscoveryCancellation,
    progress: OutputDiscoveryProgressCallback | None,
    phase: DiscoveryPhase,
) -> bool:
    total = len(inventory.entries)
    for index, entry in enumerate(inventory.entries, start=1):
        cancellation.raise_if_cancelled()
        path = source_root / entry.relative_path
        try:
            stat = path.stat()
        except OSError:
            if entry.kind == "missing":
                continue
            return False
        if entry.kind == "missing":
            return False
        kind: InventoryEntryKind = (
            "directory" if path.is_dir() else "file"
        )
        if (
            kind != entry.kind
            or stat.st_size != entry.size
            or stat.st_mtime_ns != entry.mtime_ns
            or stat.st_ctime_ns != entry.ctime_ns
        ):
            return False
        if index % _PROGRESS_INTERVAL == 0:
            report_discovery_progress(
                progress,
                phase=phase,
                detail="Verifying processing inventory.",
                completed=index,
                total=total,
                cache_hit=inventory.cache_hit,
            )
    return True


def _inventory_fingerprint(
    entries: tuple[ProcessingInventoryEntry, ...],
) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        payload = (
            f"{entry.relative_path}\0{entry.kind}\0"
            f"{entry.size}\0{entry.mtime_ns}\0{entry.ctime_ns}\n"
        ).encode("utf-8")
        digest.update(payload)
    return f"sha256:{digest.hexdigest()}"


def _load_cached_inventory(
    cache_path: Path,
    *,
    source_root: Path,
    consistency: OutputConsistencyReport,
) -> ProcessingInventory | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != _CACHE_SCHEMA_VERSION
            or payload.get("source_root") != str(source_root.resolve())
            or payload.get("evidence_fingerprint")
            != consistency.evidence_fingerprint
        ):
            return None
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            return None
        entries = tuple(
            ProcessingInventoryEntry(
                relative_path=str(raw["relative_path"]),
                kind=_entry_kind(raw["kind"]),
                size=int(raw["size"]),
                mtime_ns=int(raw["mtime_ns"]),
                ctime_ns=int(raw["ctime_ns"]),
            )
            for raw in raw_entries
            if isinstance(raw, dict)
        )
        if len(entries) != len(raw_entries):
            return None
        fingerprint = payload.get("fingerprint")
        if (
            not isinstance(fingerprint, str)
            or fingerprint != _inventory_fingerprint(entries)
        ):
            return None
    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
        return None
    return ProcessingInventory(
        entries=entries,
        fingerprint=fingerprint,
        cache_hit=True,
    )


def _entry_kind(value: object) -> InventoryEntryKind:
    if value == "file":
        return "file"
    if value == "directory":
        return "directory"
    if value == "missing":
        return "missing"
    raise ValueError(f"Unsupported inventory entry kind: {value!r}")


def _persist_inventory(
    cache_path: Path,
    *,
    source_root: Path,
    consistency: OutputConsistencyReport,
    inventory: ProcessingInventory,
) -> None:
    """Atomically publish one external cache record."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "source_root": str(source_root.resolve()),
        "evidence_fingerprint": consistency.evidence_fingerprint,
        "fingerprint": inventory.fingerprint,
        "entries": [asdict(entry) for entry in inventory.entries],
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temp_name).replace(cache_path)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
