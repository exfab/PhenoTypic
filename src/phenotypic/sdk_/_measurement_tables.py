"""Embedded per-image object-measurement table payloads."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from ._atomic_io import (
    CommitGuard,
    PARQUET_WRITE_OPTIONS,
    atomic_write_json,
    atomic_write_with_writer,
)

JoinStatus = Literal["not_requested", "joined", "no_common_keys"]


@dataclass(frozen=True)
class PreparedEmbeddedMeasurementTable:
    """Joined payload plus stable provenance recorded with its Parquet file."""

    frame: pd.DataFrame
    measurement_columns: tuple[str, ...]
    join_status: JoinStatus
    join_keys: tuple[str, ...]
    metadata_snapshot_sha256: str

    def parquet_metadata(self) -> dict[bytes, bytes]:
        """Return replaceable join provenance as Arrow schema metadata."""
        from . import ngff_

        keys = ngff_.EMBEDDED_MEASUREMENT_PARQUET_METADATA_KEYS
        values = {
            keys.JOIN_STATUS: self.join_status,
            keys.JOIN_KIND: "right",
            keys.JOIN_LEFT: "metadata",
            keys.JOIN_RIGHT: "measurements",
            keys.JOIN_KEYS: json.dumps(
                list(self.join_keys), separators=(",", ":")
            ),
            keys.METADATA_SNAPSHOT_SHA256: self.metadata_snapshot_sha256,
            keys.MEASUREMENT_COLUMNS: json.dumps(
                list(self.measurement_columns), separators=(",", ":")
            ),
        }
        return {key.encode(): value.encode() for key, value in values.items()}


def _write_validated_parquet(
    payload: Path,
    table: PreparedEmbeddedMeasurementTable,
    *,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Atomically write and validate a prepared Arrow table."""

    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    def _write(temp_path: str) -> None:
        arrow_table = pa.Table.from_pandas(table.frame, preserve_index=False)
        arrow_table = arrow_table.replace_schema_metadata(
            table.parquet_metadata()
        )
        pq.write_table(arrow_table, temp_path, **PARQUET_WRITE_OPTIONS)
        reread = pq.read_table(temp_path)
        if reread.column_names != list(table.frame.columns):
            raise RuntimeError(
                "Embedded measurement table failed schema validation"
            )

    atomic_write_with_writer(
        payload,
        _write,
        commit_guard=commit_guard,
        temp_suffix=".parquet.tmp",
    )


def write_embedded_measurement_table(
    store_part: Path,
    table: PreparedEmbeddedMeasurementTable,
) -> Path:
    """Write the prepared Parquet payload and its two Zarr v3 groups."""
    from . import ngff_

    payload = Path(store_part) / ngff_.MEASUREMENT_TABLE_RELATIVE_PATH
    payload.parent.mkdir(parents=True, exist_ok=True)
    group_document = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {},
    }
    atomic_write_json(
        Path(store_part) / ngff_.TABLES_GROUP / ngff_.STORE_ROOT_JSON,
        group_document,
    )
    atomic_write_json(payload.parent / ngff_.STORE_ROOT_JSON, group_document)
    _write_validated_parquet(payload, table)
    return payload


def build_measurement_table_descriptor(
    table: PreparedEmbeddedMeasurementTable,
    *,
    objmap_target: str,
) -> dict[str, object]:
    """Build the stable root descriptor for one embedded measurement table."""
    from phenotypic.schema import OBJECT

    from . import ngff_

    return {
        "schema_version": ngff_.MEASUREMENT_TABLE_SCHEMA_VERSION,
        "type": "object_measurements",
        "format": "parquet",
        "path": ngff_.MEASUREMENT_TABLE_RELATIVE_PATH.as_posix(),
        "measurement_columns": list(table.measurement_columns),
        "target": {
            "column": str(OBJECT.LABEL),
            "path": objmap_target,
        },
    }


def _valid_embedded_measurement_contract(store_path: Path) -> bool:
    """Return whether a store's table payload, groups, and descriptor agree."""
    from phenotypic.schema import OBJECT

    from . import ngff_

    try:
        store = Path(store_path)
        root = json.loads(
            (store / ngff_.STORE_ROOT_JSON).read_text(encoding="utf-8")
        )
        phenotypic = root["attributes"][ngff_.PhenotypicAttr.ROOT]
        descriptor = phenotypic[ngff_.PhenotypicAttr.TABLES][
            ngff_.MEASUREMENT_TABLE_GROUP
        ]
        columns = descriptor["measurement_columns"]
        objmap_target = phenotypic[ngff_.PhenotypicAttr.LABELS][
            ngff_.OBJMAP_LABEL
        ]
        if not isinstance(columns, list) or not all(
            isinstance(column, str) for column in columns
        ):
            return False
        if descriptor != {
            "schema_version": ngff_.MEASUREMENT_TABLE_SCHEMA_VERSION,
            "type": "object_measurements",
            "format": "parquet",
            "path": ngff_.MEASUREMENT_TABLE_RELATIVE_PATH.as_posix(),
            "measurement_columns": columns,
            "target": {
                "column": str(OBJECT.LABEL),
                "path": objmap_target,
            },
        }:
            return False

        expected_group = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {},
        }
        for group in (
            store / ngff_.TABLES_GROUP,
            store / ngff_.TABLES_GROUP / ngff_.MEASUREMENT_TABLE_GROUP,
        ):
            document = json.loads(
                (group / ngff_.STORE_ROOT_JSON).read_text(encoding="utf-8")
            )
            if document != expected_group:
                return False

        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        payload = store / ngff_.MEASUREMENT_TABLE_RELATIVE_PATH
        table = pq.read_table(payload)
        if any(column not in table.column_names for column in columns):
            return False
        metadata = table.schema.metadata or {}
        keys = ngff_.EMBEDDED_MEASUREMENT_PARQUET_METADATA_KEYS
        required = tuple(key.encode() for key in keys)
        if any(key not in metadata for key in required):
            return False
        join_status = metadata[keys.JOIN_STATUS.encode()].decode()
        if join_status not in {
            "not_requested",
            "joined",
            "no_common_keys",
        }:
            return False
        if (
            metadata[keys.JOIN_KIND.encode()] != b"right"
            or metadata[keys.JOIN_LEFT.encode()] != b"metadata"
            or metadata[keys.JOIN_RIGHT.encode()] != b"measurements"
        ):
            return False
        join_keys = json.loads(metadata[keys.JOIN_KEYS.encode()].decode())
        recorded_columns = json.loads(
            metadata[keys.MEASUREMENT_COLUMNS.encode()].decode()
        )
        if not isinstance(join_keys, list) or not all(
            isinstance(key, str) for key in join_keys
        ):
            return False
        digest = metadata[keys.METADATA_SNAPSHOT_SHA256.encode()].decode()
        if join_status == "not_requested":
            if join_keys or digest:
                return False
        else:
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                return False
            if join_status == "joined" and not join_keys:
                return False
            if join_status == "no_common_keys" and join_keys:
                return False
        return recorded_columns == columns
    except Exception:
        return False


def _clone_file_without_pixel_rewrite(source: str, target: str) -> str:
    """Hard-link an existing store file, falling back to a byte copy."""
    try:
        os.link(source, target)
        return target
    except OSError:
        return shutil.copy2(source, target)


def replace_embedded_measurement_table(
    store_path: Path,
    table: PreparedEmbeddedMeasurementTable,
    *,
    objmap_target: str | None = None,
    durable: bool | None = None,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Replace one store's authoritative table without recomputing pixel arrays.

    A descriptor-compatible replacement validates a same-directory temporary
    Parquet before one atomic file replacement. A descriptor change uses a
    complete root-last store transaction whose unchanged files are hard-linked
    where the platform permits.
    """
    from . import ngff_

    store_path = Path(store_path)
    root_path = store_path / ngff_.STORE_ROOT_JSON
    root_document = json.loads(root_path.read_text(encoding="utf-8"))
    phenotypic = root_document["attributes"][ngff_.PhenotypicAttr.ROOT]
    current = phenotypic.get(ngff_.PhenotypicAttr.TABLES, {}).get(
        ngff_.MEASUREMENT_TABLE_GROUP
    )
    if objmap_target is None:
        if isinstance(current, dict):
            target = current.get("target", {})
            if isinstance(target, dict):
                candidate = target.get("path")
                if isinstance(candidate, str):
                    objmap_target = candidate
        if objmap_target is None:
            labels = phenotypic.get(ngff_.PhenotypicAttr.LABELS, {})
            candidate = labels.get(ngff_.OBJMAP_LABEL)
            if isinstance(candidate, str):
                objmap_target = candidate
    if objmap_target is None:
        raise ValueError("OME-Zarr store does not declare an objmap target")

    descriptor = build_measurement_table_descriptor(
        table, objmap_target=objmap_target
    )
    payload = store_path / ngff_.MEASUREMENT_TABLE_RELATIVE_PATH
    if (
        current == descriptor
        and payload.is_file()
        and _valid_embedded_measurement_contract(store_path)
    ):
        _write_validated_parquet(payload, table, commit_guard=commit_guard)
        return payload

    part = ngff_.new_part_path(store_path)
    try:
        shutil.copytree(
            store_path,
            part,
            copy_function=_clone_file_without_pixel_rewrite,
        )
        # A transaction is readable only after its refreshed root is written last.
        (part / ngff_.STORE_ROOT_JSON).unlink()
        write_embedded_measurement_table(part, table)
        phenotypic.setdefault(ngff_.PhenotypicAttr.TABLES, {})[
            ngff_.MEASUREMENT_TABLE_GROUP
        ] = descriptor
        atomic_write_json(part / ngff_.STORE_ROOT_JSON, root_document)
        ngff_.promote_store(
            part,
            store_path,
            fsync=ngff_.durable_writes_enabled(durable),
            commit_guard=commit_guard,
        )
    except BaseException:
        if part.exists():
            shutil.rmtree(part, ignore_errors=True)
        raise
    return store_path / ngff_.MEASUREMENT_TABLE_RELATIVE_PATH
