"""Per-store embedded measurement-table rewrites for recompile."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from phenotypic.sdk_ import (
    CommitGuard,
    DIR_IMAGE_COMPLETE,
    MEASUREMENT_TABLE_RELATIVE_PATH,
    PhenotypicAttr,
    STORE_SUFFIX,
    progress_dir,
    read_phenotypic_attributes,
    store_stem,
)

from ._cli_completion import (
    authorized_measurement_sources,
    publish_image_success,
    valid_image_success,
)
from ._embedded_measurement_tables import prepare_embedded_measurement_table
from ._cli_recompile_recovery import (
    _fsync_recompile_directory,
    assert_no_unrecoverable_measurement_authority,
    begin_recompile_table_transition,
    clear_recompile_table_transition,
    recoverable_recompile_measurement_sources,
    promote_recompile_table_transition,
    recompile_store_lock_path,
)
from phenotypic.sdk_._file_locking import exclusive_path_lock


def _marker_artifacts(output_dir: Path, marker: dict) -> dict[str, Path]:
    """Resolve the existing marker's artifacts below its output root."""
    raw = marker.get("artifacts")
    if not isinstance(raw, dict):
        raise ValueError("Image completion marker has no artifact mapping")
    artifacts: dict[str, Path] = {}
    output_root = Path(output_dir).resolve()
    for name, descriptor in raw.items():
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise ValueError("Image completion marker has invalid artifacts")
        relative = descriptor.get("path")
        if not isinstance(relative, str):
            raise ValueError("Image completion marker artifact has no path")
        resolved = (output_root / relative).resolve()
        resolved.relative_to(output_root)
        artifacts[name] = resolved
    return artifacts


def _republish_table_marker(
    output_dir: Path,
    marker_path: Path,
    *,
    commit_guard: CommitGuard | None,
    lifecycle_epoch: str | None = None,
) -> None:
    """Rehash all existing artifacts and publish the marker last."""
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    publish_image_success(
        output_dir,
        work_id=str(marker["work_id"]),
        dataset=str(marker["dataset"]),
        relative_image_path=str(marker["relative_image_path"]),
        image_stem=str(marker["image_stem"]),
        mode=str(marker["mode"]),
        attempt_id=str(marker["attempt_id"]),
        lifecycle_epoch=(
            lifecycle_epoch
            if lifecycle_epoch is not None
            else str(marker["lifecycle_epoch"])
        ),
        artifacts=_marker_artifacts(output_dir, marker),
        commit_guard=commit_guard,
    )


def _replace_and_republish_table(
    output_dir: Path,
    dataset: str,
    store_path: Path,
    prepared: object,
    *,
    commit_guard: CommitGuard | None,
    lifecycle_epoch: str | None,
) -> None:
    """Journal, replace, marker-publish, and clear under one store lock."""
    from phenotypic.sdk_ import image_record_path
    from phenotypic.sdk_._measurement_tables import (
        PreparedEmbeddedMeasurementTable,
    )

    if not isinstance(prepared, PreparedEmbeddedMeasurementTable):
        raise TypeError(
            "Recompile table preparation returned an invalid payload"
        )
    stem = store_stem(store_path)
    with exclusive_path_lock(
        recompile_store_lock_path(output_dir, dataset, stem),
        timeout=60.0,
    ):
        staged = begin_recompile_table_transition(
            output_dir,
            dataset,
            stem,
            store_path,
            prepared,
        )
        promote_recompile_table_transition(
            output_dir,
            dataset,
            stem,
            store_path,
            staged,
            commit_guard=commit_guard,
        )
        # The record, for the same reason as the measure path -- see
        # `_cli_process_single`. This is the recompile half of the same
        # defect: after D1 the legacy marker is absent on a forward tree, so
        # `_republish_table_marker` would read a file that is not there.
        record_path = image_record_path(output_dir, dataset, stem)
        _republish_table_marker(
            output_dir,
            record_path,
            commit_guard=commit_guard,
            lifecycle_epoch=lifecycle_epoch,
        )
        _fsync_recompile_directory(record_path.parent)
        clear_recompile_table_transition(output_dir, dataset, stem)


def _standalone_marker_sources(output_dir: Path) -> dict[Path, str]:
    """Discover valid embedded authority when no processing state is present."""
    sources: dict[Path, str] = {}
    marker_root = progress_dir(output_dir) / DIR_IMAGE_COMPLETE
    for marker_path in sorted(marker_root.glob("*/*.json")):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            dataset = str(marker["dataset"])
            stem = str(marker["image_stem"])
            work_id = str(marker["work_id"])
            descriptor = marker["artifacts"]["measurements"]
            relative = descriptor["path"]
            table_path = (output_dir / str(relative)).resolve()
            if not valid_image_success(
                output_dir,
                dataset=dataset,
                image_stem=stem,
                work_id=work_id,
            ):
                continue
            if tuple(table_path.parts[-3:]) != (
                MEASUREMENT_TABLE_RELATIVE_PATH.parts
            ):
                continue
        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
            json.JSONDecodeError,
        ):
            continue
        sources[table_path] = dataset
    return sources


def recompile_embedded_measurement_table(
    output_dir: Path,
    table_path: Path,
    dataset: str,
    metadata_csv: Path | None,
    *,
    commit_guard: CommitGuard | None = None,
    lifecycle_epoch: str | None = None,
) -> None:
    """Rewrite one authorized embedded table and republish its marker last."""

    table_path = Path(table_path)
    if tuple(table_path.parts[-3:]) != MEASUREMENT_TABLE_RELATIVE_PATH.parts:
        raise RuntimeError(
            "Current-schema recompile requires embedded measurement tables; "
            "run --mode migrate"
        )
    store_path = table_path.parents[2]
    attrs = read_phenotypic_attributes(store_path)
    descriptor = attrs.get(PhenotypicAttr.TABLES, {}).get("measurements")
    if not isinstance(descriptor, dict):
        raise ValueError(f"Store lacks measurement descriptor: {store_path}")
    raw_baseline = descriptor.get("measurement_columns")
    if not isinstance(raw_baseline, list) or not all(
        isinstance(column, str) for column in raw_baseline
    ):
        raise ValueError(
            f"Store has invalid measurement baseline: {store_path}"
        )
    payload = pq.read_table(table_path).to_pandas()
    missing = [
        column for column in raw_baseline if column not in payload.columns
    ]
    if missing:
        raise ValueError(
            f"Embedded table cannot project its baseline {missing}: {table_path}"
        )
    prepared = prepare_embedded_measurement_table(
        payload.loc[:, raw_baseline], metadata_csv
    )
    _replace_and_republish_table(
        output_dir,
        dataset,
        store_path,
        prepared,
        commit_guard=commit_guard,
        lifecycle_epoch=lifecycle_epoch,
    )


def recompile_embedded_measurement_tables(
    output_dir: Path,
    metadata_csv: Path | None,
    *,
    commit_guard: CommitGuard | None = None,
    lifecycle_epoch: str | None = None,
) -> int:
    """Project, rejoin, replace, and marker-publish every authorized table.

    A legacy-only run is refused with a migration remedy. Once the first store
    is rewritten, any interruption leaves mixed Parquet generations; aggregate
    publication independently rejects that state until a retry converges.
    """

    output_dir = Path(output_dir)
    authorized = authorized_measurement_sources(output_dir)
    dataset_names = (
        sorted(
            path.name
            for path in (output_dir / "results").iterdir()
            if path.is_dir()
        )
        if (output_dir / "results").is_dir()
        else []
    )
    recovery_sources = recoverable_recompile_measurement_sources(
        output_dir, dataset_names
    )
    if authorized is None:
        marker_sources = _standalone_marker_sources(output_dir)
        if marker_sources or recovery_sources:
            authorized = {**marker_sources, **recovery_sources}
        else:
            legacy = sorted(
                (output_dir / "results").glob("*/measurements/*.parquet")
            )
            image_sources = list(
                (output_dir / "results").glob("*/hdf/*.h5")
            ) + list((output_dir / "results").glob(f"*/zarr/*{STORE_SUFFIX}"))
            if legacy and image_sources:
                raise RuntimeError(
                    "Legacy external measurement Parquets require --mode migrate "
                    "before recompile"
                )
            return 0

    authorized = {**authorized, **recovery_sources}
    assert_no_unrecoverable_measurement_authority(
        output_dir,
        dataset_names,
        set(authorized),
    )
    changed = 0
    for table_path, dataset in sorted(
        authorized.items(), key=lambda item: str(item[0])
    ):
        recompile_embedded_measurement_table(
            output_dir,
            table_path,
            dataset,
            metadata_csv,
            commit_guard=commit_guard,
            lifecycle_epoch=lifecycle_epoch,
        )
        changed += 1
    return changed


__all__ = [
    "recompile_embedded_measurement_table",
    "recompile_embedded_measurement_tables",
]
