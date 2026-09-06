"""Embedded per-image object-measurement table payloads."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
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


@dataclass(frozen=True)
class PreparedImageTables:
    """One image's measurement rows and, separately, its user metadata rows.

    Spec §7.1-7.2's inversion. ``measurements`` is the **pre-join baseline** --
    intrinsic identity (``Metadata_ImageFile``, ``Metadata_Dataset``, the
    object label) plus the measured columns, and nothing that came from
    ``--metadata``. ``metadata`` is the projection of the run's metadata
    snapshot onto the join keys this image actually carries, or ``None`` when
    there is no such projection to make.

    The join provenance below describes the **metadata** table, not the
    measurements table. After the inversion the measurements table carries no
    join at all, so its own recorded triple is
    ``not_requested`` / ``[]`` / ``""`` -- which is what
    :meth:`measurements_payload` builds, and the one place that rule lives.

    Attributes:
        measurements: The unjoined baseline, exactly the projection
            ``measurement_columns`` already named.
        metadata: User metadata rows for this image's keys, or ``None`` when
            ``join_status`` is ``not_requested`` or ``no_common_keys``.
        measurement_columns: Baseline column names, in writer order.
        join_status: Whether a metadata join was requested, possible, or done.
        join_keys: The common columns the metadata table is keyed on.
        metadata_snapshot_sha256: Digest of the ``metadata.csv`` snapshot this
            image was prepared against; ``""`` when none was supplied.
    """

    measurements: pd.DataFrame
    metadata: pd.DataFrame | None
    measurement_columns: tuple[str, ...]
    join_status: JoinStatus
    join_keys: tuple[str, ...]
    metadata_snapshot_sha256: str

    def measurements_payload(self) -> PreparedEmbeddedMeasurementTable:
        """Return the measurements table's own payload, carrying no join.

        The triple is ``not_requested`` / ``()`` / ``""`` unconditionally,
        because after the inversion the statement "this file is the result of
        a join" is false of ``tables/measurements/table.parquet`` on every
        store. That is also exactly the shape
        :func:`_valid_embedded_measurement_contract` already accepts, so the
        contract needs no change for this file.
        """
        return PreparedEmbeddedMeasurementTable(
            frame=self.measurements,
            measurement_columns=self.measurement_columns,
            join_status="not_requested",
            join_keys=(),
            metadata_snapshot_sha256="",
        )

    def metadata_parquet_metadata(self) -> dict[bytes, bytes]:
        """Return the metadata table's own join provenance as Arrow metadata.

        The same key spellings the measurement table uses -- one home for the
        names -- minus ``measurement_columns``, which says nothing about this
        file. This is what makes ``pht-metadata.parquet`` self-describing to a
        reader who has only the Parquet: it names the keys it is joined on,
        the direction of the join, and the snapshot it came from.
        """
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
        }
        return {key.encode(): value.encode() for key, value in values.items()}


#: Zarr v3 group document for every intermediate ``tables/*`` group. Written
#: for each group a table lives under, and compared **by exact equality** by
#: :func:`_valid_embedded_measurement_contract`, so a missing one is a
#: contract failure rather than a cosmetic omission.
_TABLES_GROUP_DOCUMENT: dict[str, object] = {
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {},
}


def _write_validated_frame(
    payload: Path,
    frame: pd.DataFrame,
    schema_metadata: dict[bytes, bytes],
    *,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Atomically write and validate one prepared Arrow table."""

    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    def _write(temp_path: str) -> None:
        arrow_table = pa.Table.from_pandas(frame, preserve_index=False)
        arrow_table = arrow_table.replace_schema_metadata(schema_metadata)
        pq.write_table(arrow_table, temp_path, **PARQUET_WRITE_OPTIONS)
        reread = pq.read_table(temp_path)
        if reread.column_names != list(frame.columns):
            raise RuntimeError(
                "Embedded measurement table failed schema validation"
            )

    atomic_write_with_writer(
        payload,
        _write,
        commit_guard=commit_guard,
        temp_suffix=".parquet.tmp",
    )


def _write_validated_parquet(
    payload: Path,
    table: PreparedEmbeddedMeasurementTable,
    *,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Atomically write and validate a prepared Arrow table."""
    _write_validated_frame(
        payload,
        table.frame,
        table.parquet_metadata(),
        commit_guard=commit_guard,
    )


def write_embedded_measurement_table(
    store_part: Path,
    table: PreparedEmbeddedMeasurementTable,
) -> Path:
    """Write the prepared Parquet payload and its two Zarr v3 groups."""
    from . import ngff_

    payload = Path(store_part) / ngff_.MEASUREMENT_TABLE_RELATIVE_PATH
    payload.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        Path(store_part) / ngff_.TABLES_GROUP / ngff_.STORE_ROOT_JSON,
        _TABLES_GROUP_DOCUMENT,
    )
    atomic_write_json(
        payload.parent / ngff_.STORE_ROOT_JSON, _TABLES_GROUP_DOCUMENT
    )
    _write_validated_parquet(payload, table)
    return payload


def write_metadata_table(
    store_part: Path,
    tables: PreparedImageTables,
) -> Path:
    """Write ``tables/metadata/pht-metadata.parquet`` and its Zarr v3 group.

    The metadata analogue of :func:`write_embedded_measurement_table`, and
    called from the same place: the store's **own** ``.part``, before the root
    ``zarr.json``. That is what makes D-A's backfill step unnecessary -- no
    path writes into a store that already carries a content proof.

    The payload is self-describing from the file alone (spec §7.2): its
    Parquet key/value metadata names the join keys, the join kind, and the
    metadata snapshot it was projected from, so a third party who opens only
    this Parquet can still say what it is and how it attaches.

    Args:
        store_part: An **unpromoted** ``*.ome.zarr.part`` directory.
        tables: The split payload. ``tables.metadata`` must not be ``None``.

    Returns:
        The written Parquet path.

    Raises:
        ValueError: If *tables* carries no metadata frame.
    """
    from . import ngff_

    if tables.metadata is None:
        raise ValueError("PreparedImageTables carries no metadata table")
    payload = Path(store_part) / ngff_.METADATA_TABLE_RELATIVE_PATH
    payload.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        Path(store_part) / ngff_.TABLES_GROUP / ngff_.STORE_ROOT_JSON,
        _TABLES_GROUP_DOCUMENT,
    )
    atomic_write_json(
        payload.parent / ngff_.STORE_ROOT_JSON, _TABLES_GROUP_DOCUMENT
    )
    _write_validated_frame(
        payload, tables.metadata, tables.metadata_parquet_metadata()
    )
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


def build_metadata_table_descriptor(
    tables: PreparedImageTables,
) -> dict[str, object]:
    """Build the stable root descriptor for one embedded metadata table.

    It records **what the file is**, not how it joins: the join keys are on
    the Parquet's own key/value metadata and in the root's ``metadata_table``
    block, and repeating them a third time here would be a third home for one
    fact.
    """
    from . import ngff_

    if tables.metadata is None:
        raise ValueError("PreparedImageTables carries no metadata table")
    return {
        "schema_version": ngff_.METADATA_TABLE_SCHEMA_VERSION,
        "type": "image_metadata",
        "format": "parquet",
        "path": ngff_.METADATA_TABLE_RELATIVE_PATH.as_posix(),
        "metadata_columns": [
            str(column) for column in tables.metadata.columns
        ],
    }


def build_metadata_table_block(
    tables: PreparedImageTables,
) -> dict[str, object] | None:
    """Return the root's ``metadata_table`` block, or ``None`` to omit it.

    D-A: a store keeps the metadata snapshot it was built against and says
    which one, so ``resolve_run_state`` can **derive** the divergence advisory
    instead of tracking a backfill stage.

    ``None`` for ``not_requested`` and only for it (H2). The advisory fires
    when a store's recorded snapshot is neither ``None`` nor the run's current
    ``metadata_sha256``; a metadata-free run has ``metadata_sha256 = None``
    and a producer digest of ``""``, so writing the block unconditionally
    would report **every** store of **every** metadata-free run as diverged --
    and an advisory that is always on teaches people to ignore the one that
    will matter. ``no_common_keys`` *does* get the block: a snapshot was
    supplied and matched nothing, there is no table to write, but a later edit
    to that snapshot is exactly the divergence worth surfacing.

    **The block carries the digest and nothing else (user ruling,
    2026-09-06).** An earlier draft of the plan also put ``join_keys`` and
    ``join_kind`` here. Those belong to the metadata table's own Parquet
    key/value metadata, where every reader that needs them is already opening
    the file. Recording them here as well would be two homes for one fact
    **with nothing comparing them** -- which is what separates this from the
    aggregate/run proof duplication, where two independently derived values
    are each checked against live state and the comparison *is* the point.
    """
    from . import ngff_

    if tables.join_status == "not_requested":
        return None
    return {
        ngff_.PhenotypicAttr.SNAPSHOT_SHA256: tables.metadata_snapshot_sha256
    }


def build_image_tables_attributes(
    tables: PreparedImageTables,
    *,
    objmap_target: str,
) -> dict[str, object]:
    """Return the ``attributes.phenotypic`` fragment two tables imply.

    A **total** description, not an increment: the caller applies it with
    :func:`apply_image_tables_attributes`, which removes whatever the fragment
    omits. See that function for why the difference is load-bearing.
    """
    from . import ngff_

    descriptors: dict[str, object] = {
        ngff_.MEASUREMENT_TABLE_GROUP: build_measurement_table_descriptor(
            tables.measurements_payload(), objmap_target=objmap_target
        )
    }
    if tables.metadata is not None:
        descriptors[ngff_.METADATA_TABLE_GROUP] = (
            build_metadata_table_descriptor(tables)
        )
    fragment: dict[str, object] = {ngff_.PhenotypicAttr.TABLES: descriptors}
    block = build_metadata_table_block(tables)
    if block is not None:
        fragment[ngff_.PhenotypicAttr.METADATA_TABLE] = block
    return fragment


def apply_image_tables_attributes(
    phenotypic: dict[str, object],
    fragment: dict[str, object],
) -> None:
    """Make *phenotypic*'s table attributes equal *fragment*, removals included.

    **Stated as a total function of the payload, never as an append.** A store
    built with ``--metadata`` and then re-measured **without** it must LOSE
    its ``metadata_table`` block and its ``tables.metadata`` descriptor. An
    append-only refresh leaves the stale digest in place,
    ``_store_metadata_snapshot`` keeps returning it, the run's
    ``metadata_sha256`` is now ``None``, and the divergence advisory fires on
    every such store forever -- H2's failure arriving through the measure path
    instead of the promote path. That is invisible until someone re-measures
    without ``--metadata``, which is why the removal is the rule and not a
    special case.
    """
    from . import ngff_

    phenotypic[ngff_.PhenotypicAttr.TABLES] = fragment[
        ngff_.PhenotypicAttr.TABLES
    ]
    if ngff_.PhenotypicAttr.METADATA_TABLE in fragment:
        phenotypic[ngff_.PhenotypicAttr.METADATA_TABLE] = fragment[
            ngff_.PhenotypicAttr.METADATA_TABLE
        ]
    else:
        phenotypic.pop(ngff_.PhenotypicAttr.METADATA_TABLE, None)


def write_image_tables(
    store_part: Path,
    tables: PreparedImageTables,
    *,
    objmap_target: str,
) -> dict[str, object]:
    """Write both of one image's tables into a part and return the fragment.

    Args:
        store_part: An **unpromoted** ``*.ome.zarr.part`` directory. Both
            tables land here, before the root ``zarr.json`` (D-A).
        tables: The split payload.
        objmap_target: Store-relative path of the label image the measurement
            table's ``Object_Label`` column indexes.

    Returns:
        The ``attributes.phenotypic`` fragment to apply to the root document
        with :func:`apply_image_tables_attributes`.
    """
    write_embedded_measurement_table(
        store_part, tables.measurements_payload()
    )
    if tables.metadata is not None:
        write_metadata_table(store_part, tables)
    return build_image_tables_attributes(tables, objmap_target=objmap_target)


def _valid_metadata_table_contract(
    store: Path, phenotypic: dict[str, object]
) -> bool:
    """Return whether a store's metadata table, group, and descriptor agree.

    **Conditioned on the descriptor being declared**, and deliberately so: a
    store written before the inversion has no ``tables.metadata`` entry and no
    ``tables/metadata/`` group, and is not thereby invalid -- it is a store
    from before the table existed. Only a store that *claims* the table is
    held to it (M2).
    """
    from . import ngff_

    tables = phenotypic.get(ngff_.PhenotypicAttr.TABLES)
    if not isinstance(tables, dict):
        return False
    descriptor = tables.get(ngff_.METADATA_TABLE_GROUP)
    if descriptor is None:
        return True
    if not isinstance(descriptor, dict):
        return False
    columns = descriptor.get("metadata_columns")
    if not isinstance(columns, list) or not all(
        isinstance(column, str) for column in columns
    ):
        return False
    if descriptor != {
        "schema_version": ngff_.METADATA_TABLE_SCHEMA_VERSION,
        "type": "image_metadata",
        "format": "parquet",
        "path": ngff_.METADATA_TABLE_RELATIVE_PATH.as_posix(),
        "metadata_columns": columns,
    }:
        return False

    group = store / ngff_.TABLES_GROUP / ngff_.METADATA_TABLE_GROUP
    document = json.loads(
        (group / ngff_.STORE_ROOT_JSON).read_text(encoding="utf-8")
    )
    if document != _TABLES_GROUP_DOCUMENT:
        return False

    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    payload = store / ngff_.METADATA_TABLE_RELATIVE_PATH
    table = pq.read_table(payload)
    if table.column_names != columns:
        return False
    metadata = table.schema.metadata or {}
    keys = ngff_.EMBEDDED_MEASUREMENT_PARQUET_METADATA_KEYS
    # Every shared key EXCEPT `measurement_columns`, which describes the other
    # table. `metadata_parquet_metadata` writes exactly this set.
    required = tuple(
        key.encode() for key in keys if key != keys.MEASUREMENT_COLUMNS
    )
    if any(key not in metadata for key in required):
        return False
    if metadata[keys.JOIN_STATUS.encode()] != b"joined":
        return False
    join_keys = json.loads(metadata[keys.JOIN_KEYS.encode()].decode())
    if not isinstance(join_keys, list) or not join_keys:
        return False
    if any(key not in columns for key in join_keys):
        return False
    digest = metadata[keys.METADATA_SNAPSHOT_SHA256.encode()].decode()
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


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

        expected_group = _TABLES_GROUP_DOCUMENT
        for group in (
            store / ngff_.TABLES_GROUP,
            store / ngff_.TABLES_GROUP / ngff_.MEASUREMENT_TABLE_GROUP,
        ):
            document = json.loads(
                (group / ngff_.STORE_ROOT_JSON).read_text(encoding="utf-8")
            )
            if document != expected_group:
                return False

        if not _valid_metadata_table_contract(store, phenotypic):
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


def _resolve_objmap_target(
    phenotypic: dict[str, object], objmap_target: str | None
) -> str:
    """Return the label image the measurement table's join column indexes."""
    from . import ngff_

    if objmap_target is not None:
        return objmap_target
    tables = phenotypic.get(ngff_.PhenotypicAttr.TABLES)
    current = (
        tables.get(ngff_.MEASUREMENT_TABLE_GROUP)
        if isinstance(tables, dict)
        else None
    )
    if isinstance(current, dict):
        target = current.get("target")
        if isinstance(target, dict) and isinstance(target.get("path"), str):
            return str(target["path"])
    labels = phenotypic.get(ngff_.PhenotypicAttr.LABELS)
    if isinstance(labels, dict):
        candidate = labels.get(ngff_.OBJMAP_LABEL)
        if isinstance(candidate, str):
            return candidate
    raise ValueError("OME-Zarr store does not declare an objmap target")


def _rewrite_store_tables(
    store_path: Path,
    *,
    plan: Callable[[dict[str, object]], Callable[[Path], None]],
    durable: bool | None,
    commit_guard: CommitGuard | None,
) -> Path:
    """Re-promote one store with refreshed embedded tables, root written last.

    *plan* is handed the store's ``attributes.phenotypic`` block. It validates
    and **mutates** it -- before any part exists, so a refusal costs nothing --
    and returns the writer that populates the part.

    **There is no in-place fast path, and its removal is the point (CAN-3 /
    C5).** The branch this replaces fired whenever the measurement
    descriptor was unchanged and rewrote ``table.parquet`` *inside the
    promoted store*, with no part and no root rewrite. Two things broke
    silently: the per-image record's store digest still matched, so the proof
    certified content that had changed underneath it; and the root's recorded
    metadata snapshot was never refreshed, so the divergence advisory read a
    stale value. After the inversion the descriptor is a pure function of the
    measurement schema and the objmap target, so *every* metadata-driven
    re-measure would take that branch.

    The cost is real and stated rather than hidden: the copytree/hardlink
    re-promote now runs on **every** ``--mode measure``, not only on a
    descriptor change. If ``--mode measure`` on a large tree becomes slow,
    this is the reason.
    """
    from . import ngff_

    store_path = Path(store_path)
    root_path = store_path / ngff_.STORE_ROOT_JSON
    root_document = json.loads(root_path.read_text(encoding="utf-8"))
    phenotypic = root_document["attributes"][ngff_.PhenotypicAttr.ROOT]
    populate = plan(phenotypic)

    part = ngff_.new_part_path(store_path)
    try:
        shutil.copytree(
            store_path,
            part,
            copy_function=_clone_file_without_pixel_rewrite,
        )
        # A transaction is readable only after its refreshed root is written last.
        (part / ngff_.STORE_ROOT_JSON).unlink()
        # The copied `tables/` tree goes with it: a store that LOSES its
        # metadata table must lose the file too, not only the descriptor.
        # These are hard links into the promoted store, so unlinking them in
        # the part leaves the live store untouched.
        shutil.rmtree(part / ngff_.TABLES_GROUP, ignore_errors=True)
        populate(part)
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
    return store_path


def replace_image_tables(
    store_path: Path,
    tables: PreparedImageTables,
    *,
    objmap_target: str | None = None,
    durable: bool | None = None,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Refresh one store's measurement AND metadata tables together.

    The ``--mode measure`` analogue of the promote-time writer: both tables
    and the root's ``metadata_table`` block move as one root-last
    transaction, so the store never certifies a table it does not have or a
    snapshot it was not built against.

    Args:
        store_path: A promoted ``*.ome.zarr`` store.
        tables: The split payload to write.
        objmap_target: Store-relative path of the label image the measurement
            table indexes. ``None`` reads it from the store.
        durable: ``fsync`` before promoting. ``None`` auto-detects SLURM.
        commit_guard: Publication guard, checked at the commit point.

    Returns:
        The store path.

    Raises:
        ValueError: If the store declares no objmap target.
    """

    def _plan(phenotypic: dict[str, object]) -> Callable[[Path], None]:
        target = _resolve_objmap_target(phenotypic, objmap_target)

        def _populate(part: Path) -> None:
            # `write_image_tables` returns the fragment describing exactly the
            # files it just wrote, so the root cannot certify a table the part
            # does not have -- and `apply_` REMOVES what the fragment omits,
            # which is how a store that loses its metadata loses the block.
            apply_image_tables_attributes(
                phenotypic,
                write_image_tables(part, tables, objmap_target=target),
            )

        return _populate

    return _rewrite_store_tables(
        store_path,
        plan=_plan,
        durable=durable,
        commit_guard=commit_guard,
    )


def replace_embedded_measurement_table(
    store_path: Path,
    table: PreparedEmbeddedMeasurementTable,
    *,
    objmap_target: str | None = None,
    durable: bool | None = None,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Replace one store's authoritative table without recomputing pixel arrays.

    A complete root-last store transaction whose unchanged files are
    hard-linked where the platform permits.

    **Superseded by** :func:`replace_image_tables` on every forward path. It
    survives only for the consumers that still read and rewrite
    *pre-inversion* stores byte-exactly -- ``--mode migrate`` and ``--mode
    recompile``, whose reclaim authority compares a store's bytes against a
    joined payload. **Retire it** with the last of those call sites. It
    deliberately leaves ``metadata_table`` and ``tables.metadata`` exactly as
    it found them: it has no metadata payload to describe, and clearing a
    block it knows nothing about would be a guess.
    """
    from . import ngff_

    def _plan(phenotypic: dict[str, object]) -> Callable[[Path], None]:
        target = _resolve_objmap_target(phenotypic, objmap_target)
        descriptor = build_measurement_table_descriptor(
            table, objmap_target=target
        )
        existing = phenotypic.get(ngff_.PhenotypicAttr.TABLES)
        declared = dict(existing) if isinstance(existing, dict) else {}
        declared[ngff_.MEASUREMENT_TABLE_GROUP] = descriptor
        phenotypic[ngff_.PhenotypicAttr.TABLES] = declared

        def _populate(part: Path) -> None:
            write_embedded_measurement_table(part, table)
            if ngff_.METADATA_TABLE_GROUP not in declared:
                return
            # The `tables/` tree was cleared along with the measurement
            # table, so a metadata table this store still DECLARES has to be
            # carried across, or the promoted store would certify a file it
            # no longer has.
            source = Path(store_path) / ngff_.METADATA_TABLE_RELATIVE_PATH
            destination = part / ngff_.METADATA_TABLE_RELATIVE_PATH
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                destination.parent / ngff_.STORE_ROOT_JSON,
                _TABLES_GROUP_DOCUMENT,
            )
            _clone_file_without_pixel_rewrite(str(source), str(destination))

        return _populate

    _rewrite_store_tables(
        store_path,
        plan=_plan,
        durable=durable,
        commit_guard=commit_guard,
    )
    return Path(store_path) / ngff_.MEASUREMENT_TABLE_RELATIVE_PATH


def read_embedded_measurement_descriptor(
    store_path: Path,
) -> dict[str, object]:
    """Return one store's ``tables.measurements`` descriptor.

    The descriptor is the store's own account of its embedded table: the
    payload path, the ``measurement_columns`` list, and the ``target``
    naming the join column and the label image it indexes. Reading it costs
    one small JSON parse and **never** opens the Parquet payload, so a
    caller that only needs the column list -- a column picker, say -- pays
    nothing for the ~130 columns it does not want.

    Args:
        store_path: Path to a ``*.ome.zarr`` directory.

    Returns:
        The descriptor mapping, exactly as
        :func:`build_measurement_table_descriptor` wrote it.

    Raises:
        OSError: If the store's root ``zarr.json`` does not exist.
        KeyError: If the root carries no ``phenotypic`` block, or the block
            declares no ``tables.measurements`` descriptor. **An absent
            descriptor is a normal state**, not a fault: a ``--mode
            process`` run never measures, and a store written before
            embedded tables has none.
        ValueError: If the store's ``store_schema_version`` is not this
            build's -- the same refusal every other content reader makes.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from phenotypic import GridImage
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> img = GridImage(load_synth_yeast_plate())
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     store = img.save2zarr(Path(tmp) / 'plate.ome.zarr')
        ...     try:
        ...         read_embedded_measurement_descriptor(store)
        ...     except KeyError:
        ...         print('no embedded table')
        no embedded table
    """
    from . import ngff_

    store_path = Path(store_path)
    # Gated by VALUE, not read raw: a store written by a newer build must be
    # refused here exactly as it is on every other path that decodes store
    # content, or this reader becomes the one place a future schema slips
    # through under today's semantics.
    block = ngff_.require_readable_store(store_path)
    descriptor = block.get(ngff_.PhenotypicAttr.TABLES, {}).get(
        ngff_.MEASUREMENT_TABLE_GROUP
    )
    if not isinstance(descriptor, dict):
        raise KeyError(
            f"OME-Zarr store declares no embedded measurement table: "
            f"{store_path}"
        )
    return descriptor


def embedded_measurement_columns(store_path: Path) -> tuple[str, ...]:
    """Return the column names one store's embedded table carries.

    The store enumerates its own columns in the descriptor, so this is the
    authoritative allow-list for anything that projects a single column --
    and it is what makes a column name a *closed* value set rather than a
    free-text parameter that reaches the filesystem.

    Args:
        store_path: Path to a ``*.ome.zarr`` directory.

    Returns:
        The declared column names, in the order the writer recorded them.

    Raises:
        OSError: If the store's root ``zarr.json`` does not exist.
        KeyError: If the store declares no measurement-table descriptor.
    """
    descriptor = read_embedded_measurement_descriptor(store_path)
    columns = descriptor.get("measurement_columns")
    if not isinstance(columns, list):
        raise KeyError(
            f"Measurement-table descriptor carries no column list: "
            f"{Path(store_path)}"
        )
    return tuple(str(column) for column in columns)


def read_embedded_measurement_column(
    store_path: Path, column: str
) -> dict[int, float | None]:
    """Project one measurement column out of a store's embedded table.

    Returns the column keyed by the descriptor's own ``target.column`` --
    ``Object_Label`` -- rather than by a positional index or an assumed key
    name, so the value a caller paints onto a colony is the value measured
    for *that* object. The join key is read from the store; it is never
    assumed.

    ``column`` is checked against
    :func:`embedded_measurement_columns` **before** the Parquet is opened.
    A name the store does not declare therefore never reaches the
    filesystem, which is what lets a request-facing caller pass a
    user-supplied name through without it becoming a probe.

    Only two of the table's ~130 columns are read. Parquet is columnar, so
    the other 128 are never decoded -- that is what makes a per-request
    projection affordable.

    Args:
        store_path: Path to a ``*.ome.zarr`` directory.
        column: Name of the column to project. Must appear in the store's
            declared ``measurement_columns``.

    Returns:
        A mapping ``{object_label: value}``. A null cell maps to ``None``.

    Raises:
        OSError: If the store's root ``zarr.json`` does not exist.
        KeyError: If the store declares no measurement-table descriptor.
        ValueError: If *column* is not one the store declares.
        TypeError: If *column* holds values that are not numbers -- a
            colour hex string, say. Measurement display scales them, and a
            silent ``None`` would hide the mismatch.
    """
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    from . import ngff_

    store_path = Path(store_path)
    descriptor = read_embedded_measurement_descriptor(store_path)
    declared = descriptor.get("measurement_columns")
    if not isinstance(declared, list) or column not in declared:
        raise ValueError(
            f"Column {column!r} is not declared by the embedded measurement "
            f"table at {store_path}"
        )
    target = descriptor.get("target")
    if not isinstance(target, dict) or not isinstance(
        target.get("column"), str
    ):
        raise KeyError(
            f"Measurement-table descriptor names no target column: "
            f"{store_path}"
        )
    join_column = str(target["column"])

    payload = store_path / ngff_.MEASUREMENT_TABLE_RELATIVE_PATH
    projection = [join_column]
    if column != join_column:
        projection.append(column)
    table = pq.read_table(payload, columns=projection)

    labels = table.column(join_column).to_pylist()
    values = table.column(column).to_pylist()
    projected: dict[int, float | None] = {}
    for label, value in zip(labels, values, strict=True):
        if label is None:
            continue
        if value is None:
            projected[int(label)] = None
            continue
        try:
            projected[int(label)] = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Column {column!r} of {store_path} is not numeric "
                f"(value {value!r} for label {label!r})"
            ) from exc
    return projected
