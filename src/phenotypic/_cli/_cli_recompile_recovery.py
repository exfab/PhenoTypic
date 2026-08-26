"""Crash-recovery evidence and locking for per-store recompile mutation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4


from phenotypic.sdk_ import (
    DIR_RESULTS,
    DIR_ZARR,
    MEASUREMENT_TABLE_RELATIVE_PATH,
    STORE_SUFFIX,
    CommitGuard,
    PreparedEmbeddedMeasurementTable,
    atomic_write_bytes,
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
        or (output_root / relative).resolve() != table_path.resolve(strict=True)
        or descriptor.get("kind", ARTIFACT_KIND_FILE) != ARTIFACT_KIND_FILE
        or not isinstance(size, int)
        or size < 0
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
    ):
        raise ValueError("Marker measurement descriptor is invalid")
    return size, sha256


def _validated_transition_staging_path(
    output_root: Path,
    dataset_name: str,
    stem: str,
    transition: dict[str, Any],
    table_path: Path,
) -> Path:
    """Return a canonical, private staged payload or raise."""
    relative = transition.get("prepared_path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Transition prepared path is not relative")
    staging_root = _transition_root(output_root, dataset_name)
    if (
        staging_root.is_symlink()
        or staging_root.resolve(strict=True) != staging_root
    ):
        raise ValueError("Transition staging directory is not canonical")
    candidate = output_root / relative
    if (
        candidate.parent != staging_root
        or candidate.is_symlink()
        or not candidate.is_file()
        or candidate.resolve(strict=True) != candidate
        or candidate == table_path
        or candidate.samefile(table_path)
        or re.fullmatch(
            rf"{re.escape(stem)}\.[0-9a-f]{{32}}\.parquet",
            candidate.name,
        )
        is None
    ):
        raise ValueError("Transition prepared payload is not canonical")
    return candidate


def _cleanup_orphan_staging_payloads(
    staging_root: Path,
    stem: str,
    *,
    keep: Path,
) -> None:
    """Remove only canonical same-stem staging files after journal rotation."""
    pattern = re.compile(rf"{re.escape(stem)}\.[0-9a-f]{{32}}\.parquet")
    for candidate in staging_root.glob(f"{stem}.*.parquet"):
        if (
            candidate == keep
            or candidate.is_symlink()
            or not candidate.is_file()
            or candidate.parent != staging_root
            or pattern.fullmatch(candidate.name) is None
        ):
            continue
        candidate.unlink()


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
        "prior_table_size": prior_table_size,
        "prior_table_sha256": prior_table_sha256,
        "prepared_path": staged.relative_to(output_root).as_posix(),
        "prepared_size": staged.stat().st_size,
        "prepared_sha256": _sha256(staged),
    }
    atomic_write_json(
        recompile_table_transition_path(output_root, dataset_name, stem),
        transition,
    )
    _cleanup_orphan_staging_payloads(root, stem, keep=staged)
    return staged


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
    canonical_store = zarr_store_path(
        output_root, dataset_name, stem
    ).resolve(strict=True)
    if store != canonical_store:
        raise RuntimeError("Transition store is not canonical")
    table = store / MEASUREMENT_TABLE_RELATIVE_PATH
    transition_path = recompile_table_transition_path(
        output_root, dataset_name, stem
    )
    try:
        transition = json.loads(transition_path.read_text(encoding="utf-8"))
        marker_path = image_completion_marker_path(
            output_root, dataset_name, stem
        )
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        prior_size, prior_sha256 = _marker_measurement_fingerprint(
            output_root,
            marker,
            table,
        )
        staged = _validated_transition_staging_path(
            output_root,
            dataset_name,
            stem,
            transition,
            table,
        )
        intended_size = transition.get("prepared_size")
        intended_sha256 = transition.get("prepared_sha256")
        if (
            staged != Path(staged_path)
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
            or staged.stat().st_size != intended_size
            or _sha256(staged) != intended_sha256
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
            return table
        if current != prior:
            raise RuntimeError(
                "Canonical table matches neither prior nor intended transition"
            )
        staged_bytes = staged.read_bytes()

        def _validate_immediately_before_replace() -> None:
            if _fingerprint(staged) != intended or _fingerprint(table) != prior:
                raise RuntimeError(
                    "Recompile transition changed before table promotion"
                )

        atomic_write_bytes(
            table,
            staged_bytes,
            pre_replace=_validate_immediately_before_replace,
            commit_guard=commit_guard,
        )
        if (
            _fingerprint(table) != intended
            or not _valid_embedded_measurement_contract(store)
        ):
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
    """Remove only canonical transition staging after marker-last publication."""
    output_root = Path(output_dir).resolve()
    path = recompile_table_transition_path(output_root, dataset_name, stem)
    try:
        transition = json.loads(path.read_text(encoding="utf-8"))
        store = zarr_store_path(
            output_root, dataset_name, stem
        ).resolve(strict=True)
        staged = _validated_transition_staging_path(
            output_root,
            dataset_name,
            stem,
            transition,
            store / MEASUREMENT_TABLE_RELATIVE_PATH,
        )
        staged.unlink()
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
        prior_table_size, prior_table_sha256 = _marker_measurement_fingerprint(
            output_root,
            marker,
            table,
        )
        prepared = _validated_transition_staging_path(
            output_root,
            dataset_name,
            stem,
            transition,
            table,
        )
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
            or transition.get("prepared_size") != prepared.stat().st_size
            or transition.get("prepared_sha256") != _sha256(prepared)
            or transition.get("prepared_size") != table.stat().st_size
            or transition.get("prepared_sha256") != _sha256(table)
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
    "promote_recompile_table_transition",
    "recoverable_recompile_measurement_sources",
    "recoverable_recompile_table_transition",
    "recompile_store_lock_path",
    "recompile_table_transition_path",
]
