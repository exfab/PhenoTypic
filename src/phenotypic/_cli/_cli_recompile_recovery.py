"""Crash-recovery evidence and locking for per-store recompile mutation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from phenotypic.sdk_ import (
    DIR_RESULTS,
    DIR_ZARR,
    MEASUREMENT_TABLE_RELATIVE_PATH,
    STORE_SUFFIX,
    PreparedEmbeddedMeasurementTable,
    atomic_write_json,
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


def marker_claims_measurement_authority(marker_path: Path) -> bool:
    """Return whether a marker declares an embedded measurement artifact."""
    try:
        marker = json.loads(Path(marker_path).read_text(encoding="utf-8"))
        artifacts = marker.get("artifacts")
    except (OSError, AttributeError, json.JSONDecodeError):
        return False
    return isinstance(artifacts, dict) and "measurements" in artifacts


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
    prior_staged: Path | None = None
    if transition_authorized:
        prior = json.loads(
            recompile_table_transition_path(
                output_root, dataset_name, stem
            ).read_text(encoding="utf-8")
        )
        prior_staged = (output_root / str(prior["prepared_path"])).resolve()
        prior_staged.relative_to(output_root)

    root = _transition_root(output_root, dataset_name)
    staged = root / f"{stem}.{uuid4().hex}.parquet"
    _write_validated_parquet(staged, prepared)
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
        "prepared_path": staged.relative_to(output_root).as_posix(),
        "prepared_size": staged.stat().st_size,
        "prepared_sha256": _sha256(staged),
    }
    atomic_write_json(
        recompile_table_transition_path(output_root, dataset_name, stem),
        transition,
    )
    if prior_staged is not None and prior_staged != staged:
        prior_staged.unlink(missing_ok=True)
    return staged


def clear_recompile_table_transition(
    output_dir: Path, dataset_name: str, stem: str
) -> None:
    """Remove transition evidence after marker-last publication succeeds."""
    path = recompile_table_transition_path(output_dir, dataset_name, stem)
    try:
        transition = json.loads(path.read_text(encoding="utf-8"))
        prepared = transition.get("prepared_path")
        if isinstance(prepared, str):
            staged = (Path(output_dir).resolve() / prepared).resolve()
            staged.relative_to(Path(output_dir).resolve())
            staged.unlink(missing_ok=True)
    except (OSError, AttributeError, ValueError, json.JSONDecodeError):
        pass
    path.unlink(missing_ok=True)


def recoverable_recompile_table_transition(
    output_dir: Path,
    dataset_name: str,
    stem: str,
    store_path: Path,
) -> bool:
    """Return whether durable evidence exactly authorizes current table bytes."""
    output_root = Path(output_dir).resolve()
    transition_path = recompile_table_transition_path(
        output_root, dataset_name, stem
    )
    try:
        transition = json.loads(transition_path.read_text(encoding="utf-8"))
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
        prepared_relative = transition["prepared_path"]
        if not isinstance(prepared_relative, str):
            return False
        prepared = (output_root / prepared_relative).resolve(strict=True)
        prepared.relative_to(output_root)
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
            or transition.get("prepared_size") != prepared.stat().st_size
            or transition.get("prepared_sha256") != _sha256(prepared)
            or store != canonical_store
            or not _marker_allows_table_transition(
                output_root, dataset_name, stem, marker, table
            )
            or not _valid_embedded_measurement_contract(store)
        ):
            return False
        current_table = pq.read_table(table)
        prepared_table = pq.read_table(prepared)
        return current_table.equals(prepared_table, check_metadata=True)
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
        root = _transition_root(output_root, dataset_name)
        for transition_path in sorted(root.glob("*.json")):
            stem = transition_path.stem
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
    "recoverable_recompile_measurement_sources",
    "recoverable_recompile_table_transition",
    "recompile_store_lock_path",
    "recompile_table_transition_path",
]
