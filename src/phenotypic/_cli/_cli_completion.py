"""Marker-last per-image success publication and validation."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from uuid import uuid4

from phenotypic.sdk_ import (
    aggregate_publication_marker_path,
    atomic_write_json,
    image_completion_marker_path,
    master_measurements_csv_path,
    master_measurements_parquet_path,
    measurements_csv_path,
    measurements_parquet_path,
    run_completion_marker_path,
)

SUCCESS_MARKER_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_image_success(
    output_dir: Path,
    *,
    work_id: str,
    dataset: str,
    relative_image_path: str,
    image_stem: str,
    mode: str,
    attempt_id: str,
    lifecycle_epoch: str,
    artifacts: Mapping[str, Path],
) -> Path:
    """Validate artifacts and atomically publish the image marker last."""
    if os.environ.get("SLURM_JOB_ID"):
        from ._cli_slurm_lifecycle import load_slurm_lifecycle

        lifecycle = load_slurm_lifecycle(output_dir)
        if lifecycle is not None and (
            lifecycle.get("generation") != lifecycle_epoch
            or lifecycle.get("active") is not True
        ):
            raise RuntimeError(
                "Cannot publish image success for a stale SLURM lifecycle"
            )
    descriptors: dict[str, dict[str, object]] = {}
    output_root = output_dir.resolve()
    for name, artifact in artifacts.items():
        resolved = artifact.resolve(strict=True)
        try:
            relative = resolved.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(f"Artifact escapes output root: {artifact}") from exc
        descriptors[name] = {
            "path": relative.as_posix(),
            "size": resolved.stat().st_size,
            "sha256": _sha256(resolved),
        }
    marker = {
        "version": SUCCESS_MARKER_VERSION,
        "work_id": work_id,
        "dataset": dataset,
        "relative_image_path": Path(relative_image_path).as_posix(),
        "image_stem": image_stem,
        "mode": mode,
        "attempt_id": attempt_id,
        "lifecycle_epoch": lifecycle_epoch,
        "artifacts": descriptors,
        "completed_at": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
    }
    marker_path = image_completion_marker_path(output_dir, dataset, image_stem)
    atomic_write_json(marker_path, marker)
    return marker_path


def valid_image_success(
    output_dir: Path,
    *,
    dataset: str,
    image_stem: str,
    work_id: str,
) -> bool:
    """Return whether the marker and every declared artifact match disk."""
    marker_path = image_completion_marker_path(output_dir, dataset, image_stem)
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("version") != SUCCESS_MARKER_VERSION
            or marker.get("work_id") != work_id
            or marker.get("dataset") != dataset
            or marker.get("image_stem") != image_stem
        ):
            return False
        artifacts = marker.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            return False
        output_root = output_dir.resolve()
        for descriptor in artifacts.values():
            if not isinstance(descriptor, dict):
                return False
            relative = descriptor.get("path")
            if not isinstance(relative, str):
                return False
            artifact = (output_root / relative).resolve()
            artifact.relative_to(output_root)
            if (
                not artifact.is_file()
                or artifact.stat().st_size != descriptor.get("size")
                or _sha256(artifact) != descriptor.get("sha256")
            ):
                return False
    except (OSError, ValueError, json.JSONDecodeError, AttributeError):
        return False
    return True


def current_success_counts(output_dir: Path) -> tuple[int, int] | None:
    """Return marker-validated ``(successful, total)`` for the current state.

    ``None`` identifies a legacy state that does not require general image
    success markers. Callers may retain their schema-2 compatibility path in
    that case, but schema-3 completion never depends on a manifest.
    """
    from ._cli_state_management import load_processing_state

    try:
        state = load_processing_state(output_dir)
    except (KeyError, TypeError, ValueError):
        return None
    if state is None or not state.config.get("success_markers_required", False):
        return None
    raw_work_ids = state.config.get("work_ids")
    if not isinstance(raw_work_ids, dict):
        return (0, sum(len(item.initial_images) for item in state.datasets.values()))

    successful = 0
    total = 0
    for dataset, raw_images in raw_work_ids.items():
        if not isinstance(dataset, str) or not isinstance(raw_images, dict):
            continue
        for image_name, work_id in raw_images.items():
            if not isinstance(image_name, str) or not isinstance(work_id, str):
                continue
            total += 1
            if valid_image_success(
                output_dir,
                dataset=dataset,
                image_stem=Path(image_name).stem,
                work_id=work_id,
            ):
                successful += 1
    return successful, total


def _current_success_work_ids(output_dir: Path, work_ids: object) -> list[str]:
    """Return sorted current work IDs backed by valid success markers."""
    successful: list[str] = []
    if not isinstance(work_ids, dict):
        return successful
    for dataset, raw_images in work_ids.items():
        if not isinstance(dataset, str) or not isinstance(raw_images, dict):
            continue
        for image_name, work_id in raw_images.items():
            if (
                isinstance(image_name, str)
                and isinstance(work_id, str)
                and valid_image_success(
                    output_dir,
                    dataset=dataset,
                    image_stem=Path(image_name).stem,
                    work_id=work_id,
                )
            ):
                successful.append(work_id)
    return sorted(successful)


def current_aggregate_is_current(output_dir: Path) -> bool | None:
    """Return whether aggregate evidence matches every current success.

    ``None`` identifies a legacy state. A valid partial aggregate is current
    when its source set is exactly the current marker-authorized success set,
    even though the whole run may remain terminal-incomplete.
    """
    from ._cli_state_management import load_processing_state

    try:
        state = load_processing_state(output_dir)
    except (KeyError, TypeError, ValueError):
        return None
    if state is None or not state.config.get("success_markers_required", False):
        return None
    if state.config.get("process_only_layer"):
        return True
    aggregate = valid_aggregate_snapshot(output_dir)
    if aggregate is None:
        return False
    work_ids = state.config.get("work_ids", {})
    expected_finalization = _canonical_digest(
        {
            "metadata_sha256": state.config.get("metadata_sha256"),
            "include_dataset_column": state.config.get(
                "include_dataset_column"
            ),
            "no_qc": state.config.get("no_qc", False),
        }
    )
    successful_work_ids = _current_success_work_ids(output_dir, work_ids)
    return (
        aggregate.get("inventory_digest") == _canonical_digest(work_ids)
        and aggregate.get("finalization_input_digest") == expected_finalization
        and aggregate.get("scientific_config_digest")
        == state.config.get("pipeline_sha256")
        and aggregate.get("source_set_digest")
        == _canonical_digest(successful_work_ids)
        and aggregate.get("source_image_count") == len(successful_work_ids)
    )


def current_run_is_complete(output_dir: Path) -> bool | None:
    """Return marker-derived current completion, or ``None`` for legacy state."""
    counts = current_success_counts(output_dir)
    if counts is None:
        return None
    successful, total = counts
    if successful != total:
        return False
    from ._cli_state_management import load_processing_state

    try:
        state = load_processing_state(output_dir)
    except (KeyError, TypeError, ValueError):
        return None
    if state is not None and state.config.get("process_only_layer"):
        return True
    return current_aggregate_is_current(output_dir) is True


def authorized_measurement_sources(
    output_dir: Path,
) -> dict[Path, str] | None:
    """Return current marker-authorized measurement Parquets by dataset.

    ``None`` requests the legacy source-discovery path. An empty mapping is a
    valid schema-3 result meaning that no successful measurements exist yet.
    """
    from ._cli_state_management import load_processing_state

    try:
        state = load_processing_state(output_dir)
    except (KeyError, TypeError, ValueError):
        return None
    if state is None or not state.config.get("success_markers_required", False):
        return None
    raw_work_ids = state.config.get("work_ids")
    if not isinstance(raw_work_ids, dict):
        return {}

    sources: dict[Path, str] = {}
    output_root = output_dir.resolve()
    for dataset, raw_images in raw_work_ids.items():
        if not isinstance(dataset, str) or not isinstance(raw_images, dict):
            continue
        for image_name, work_id in raw_images.items():
            if not isinstance(image_name, str) or not isinstance(work_id, str):
                continue
            stem = Path(image_name).stem
            if not valid_image_success(
                output_dir,
                dataset=dataset,
                image_stem=stem,
                work_id=work_id,
            ):
                continue
            try:
                marker = json.loads(
                    image_completion_marker_path(
                        output_dir, dataset, stem
                    ).read_text(encoding="utf-8")
                )
                descriptor = marker["artifacts"]["measurements"]
                relative = descriptor["path"]
                if not isinstance(relative, str):
                    continue
                source = (output_root / relative).resolve()
                source.relative_to(output_root)
            except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            sources[source] = dataset
    return sources


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def publish_aggregate_snapshot(output_dir: Path) -> Path:
    """Publish marker-last integrity evidence for the canonical core snapshot."""
    from ._cli_state_management import load_processing_state

    state = load_processing_state(output_dir)
    if state is None or not state.config.get("success_markers_required", False):
        raise RuntimeError("Aggregate marker publication requires current state")
    counts = current_success_counts(output_dir)
    if counts is None or counts[0] == 0:
        raise RuntimeError("No marker-authorized measurements to publish")

    required_paths = {
        "master_csv": master_measurements_csv_path(output_dir),
        "master_parquet": master_measurements_parquet_path(output_dir),
        "measurements_csv": measurements_csv_path(output_dir),
        "measurements_parquet": measurements_parquet_path(output_dir),
    }
    output_root = output_dir.resolve()
    descriptors: dict[str, dict[str, object]] = {}
    for name, path in required_paths.items():
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(output_root)
        descriptors[name] = {
            "path": relative.as_posix(),
            "size": resolved.stat().st_size,
            "sha256": _sha256(resolved),
        }

    work_ids = state.config.get("work_ids", {})
    source_work_ids = _current_success_work_ids(output_dir, work_ids)
    marker = {
        "version": 1,
        "publication_id": uuid4().hex,
        "processing_generation": state.config.get("processing_generation"),
        "inventory_digest": _canonical_digest(work_ids),
        "finalization_input_digest": _canonical_digest(
            {
                "metadata_sha256": state.config.get("metadata_sha256"),
                "include_dataset_column": state.config.get(
                    "include_dataset_column"
                ),
                "no_qc": state.config.get("no_qc", False),
            }
        ),
        "scientific_config_digest": state.config.get("pipeline_sha256"),
        "source_set_digest": _canonical_digest(sorted(source_work_ids)),
        "source_image_count": len(source_work_ids),
        "required_outputs": descriptors,
        "published_at": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
    }
    path = aggregate_publication_marker_path(output_dir)
    atomic_write_json(path, marker)
    return path


def valid_aggregate_snapshot(output_dir: Path) -> dict[str, object] | None:
    """Return a valid aggregate marker, rejecting any mixed core snapshot."""
    path = aggregate_publication_marker_path(output_dir)
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(marker, dict) or marker.get("version") != 1:
            return None
        outputs = marker.get("required_outputs")
        if not isinstance(outputs, dict) or not outputs:
            return None
        output_root = output_dir.resolve()
        for descriptor in outputs.values():
            if not isinstance(descriptor, dict):
                return None
            relative = descriptor.get("path")
            if not isinstance(relative, str):
                return None
            artifact = (output_root / relative).resolve()
            artifact.relative_to(output_root)
            if (
                not artifact.is_file()
                or artifact.stat().st_size != descriptor.get("size")
                or _sha256(artifact) != descriptor.get("sha256")
            ):
                return None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return marker


def publish_run_completion_evidence(
    output_dir: Path,
    *,
    execution_epoch: str,
    gui_record_generation: str | None = None,
) -> Path:
    """Publish all-success run evidence, idempotently for a no-op run."""
    from ._cli_state_management import load_processing_state

    completion = current_run_is_complete(output_dir)
    state = load_processing_state(output_dir)
    if completion is None or state is None or not state.config.get(
        "success_markers_required", False
    ):
        path = run_completion_marker_path(output_dir)
        atomic_write_json(
            path,
            {
                "schema_version": 1,
                "generation": gui_record_generation or execution_epoch,
                "execution_epoch": execution_epoch,
                "mode": (
                    "local"
                    if gui_record_generation is not None or execution_epoch == "local"
                    else "slurm"
                ),
                "status": "complete",
                "finalizer_succeeded": True,
                "completed_at": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
            },
        )
        return path
    if completion is not True:
        raise RuntimeError("Current run does not have complete publication evidence")
    aggregate = (
        None
        if state.config.get("process_only_layer")
        else valid_aggregate_snapshot(output_dir)
    )
    work_ids = state.config.get("work_ids", {})
    payload = {
        "version": 2,
        "processing_generation": state.config.get("processing_generation"),
        "inventory_digest": _canonical_digest(work_ids),
        "finalization_input_digest": (
            aggregate.get("finalization_input_digest")
            if aggregate is not None
            else _canonical_digest({"process_only_layer": state.config.get("process_only_layer")})
        ),
        "scientific_config_digest": state.config.get("pipeline_sha256"),
        "publication_id": (
            aggregate.get("publication_id") if aggregate is not None else None
        ),
        "execution_epoch": execution_epoch,
        "gui_record_generation": gui_record_generation,
        "generation": gui_record_generation or execution_epoch,
        "mode": (
            "local"
            if gui_record_generation is not None or execution_epoch == "local"
            else "slurm"
        ),
        "status": "complete",
        "finalizer_succeeded": True,
        "completed_at": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
    }
    path = run_completion_marker_path(output_dir)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        existing = None
    stable_keys = (
        "version",
        "inventory_digest",
        "finalization_input_digest",
        "scientific_config_digest",
        "publication_id",
        "status",
    )
    if isinstance(existing, dict) and all(
        existing.get(key) == payload.get(key) for key in stable_keys
    ):
        return path
    atomic_write_json(path, payload)
    return path


def valid_run_completion(output_dir: Path) -> dict[str, object] | None:
    """Return current all-success run evidence, rejecting stale markers."""
    from ._cli_state_management import load_processing_state

    path = run_completion_marker_path(output_dir)
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(marker, dict) or marker.get("status") != "complete":
        return None
    try:
        state = load_processing_state(output_dir)
    except (KeyError, TypeError, ValueError):
        return None
    if state is None or not state.config.get("success_markers_required", False):
        return marker if marker.get("finalizer_succeeded") is True else None
    if marker.get("version") != 2 or current_run_is_complete(output_dir) is not True:
        return None
    work_ids = state.config.get("work_ids", {})
    expected: dict[str, object] = {
        "inventory_digest": _canonical_digest(work_ids),
        "scientific_config_digest": state.config.get("pipeline_sha256"),
    }
    if state.config.get("process_only_layer"):
        expected["publication_id"] = None
        expected["finalization_input_digest"] = _canonical_digest(
            {"process_only_layer": state.config.get("process_only_layer")}
        )
    else:
        aggregate = valid_aggregate_snapshot(output_dir)
        if aggregate is None:
            return None
        expected["publication_id"] = aggregate.get("publication_id")
        expected["finalization_input_digest"] = aggregate.get(
            "finalization_input_digest"
        )
    if any(marker.get(key) != value for key, value in expected.items()):
        return None
    return marker
