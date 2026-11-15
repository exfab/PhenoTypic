"""Embedded per-image object-measurement table payloads."""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

import pandas as pd

from ._atomic_io import (
    CommitGuard,
    PARQUET_WRITE_OPTIONS,
    atomic_write_json,
    atomic_write_with_writer,
)

if TYPE_CHECKING:
    import numpy as np

    from ._image_figures import StoredFigures
    from ._table_runs import TableRun

logger = logging.getLogger(__name__)

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


def _group_document(attributes: Mapping[str, object]) -> dict[str, object]:
    """A Zarr v3 group document carrying *attributes*."""
    return {**_TABLES_GROUP_DOCUMENT, "attributes": dict(attributes)}


def _ensure_group(
    directory: Path, document: Mapping[str, object] = _TABLES_GROUP_DOCUMENT
) -> None:
    """Make *directory* a Zarr v3 group: create it and write its ``zarr.json``."""
    from . import ngff_

    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / ngff_.STORE_ROOT_JSON, dict(document))


def ngff_feature_table_attrs(objmap_rel_path: str) -> dict[str, object]:
    """The ngio feature-table attributes on a run's ``measurements/`` group (spec D4).

    They let an ngio-style reader recognise the Parquet as an object table
    indexed by ``Object_Label`` over the objmap label. Harmless to every other
    reader; the layout is never flattened to suit ngio.

    Args:
        objmap_rel_path: The objmap label's path **relative to the table
            group's parent** -- ``tables/<run>/`` -- which is the convention
            the fractal/ngio table specification uses
            (``"../labels/<name>"`` from ``<image>/tables/``). For a run folder
            one level down that is ``"../../rgb/labels/objmap"``.

    Returns:
        The group's ``attributes`` mapping.
    """
    from phenotypic.schema import OBJECT

    label = str(OBJECT.LABEL)
    return {
        "type": "feature_table",
        "table_version": "1",
        "region": {"path": objmap_rel_path},
        "backend": "parquet",
        "index_key": label,
        "index_type": "int",
        "instance_key": label,
    }


def _objmap_region_path(run_id: str, objmap_target: str) -> str:
    """*objmap_target* (store-relative) relative to ``tables/<run_id>/``."""
    import posixpath

    from ._table_runs import table_run_relative_path

    return posixpath.relpath(
        objmap_target, start=table_run_relative_path(run_id).as_posix()
    )


def _measurement_group_document(
    run_id: str, objmap_target: str
) -> dict[str, object]:
    """The exact ``zarr.json`` of ``tables/<run_id>/measurements/``."""
    return _group_document(
        ngff_feature_table_attrs(_objmap_region_path(run_id, objmap_target))
    )


def array_digest(array: np.ndarray) -> str:
    """sha256 over an array's dtype string, shape and C-order bytes.

    The binding digest of spec D7: computed from the in-memory array at write
    time, never by re-reading the store, and identical for one set of values
    whatever the array's memory layout (a Fortran-ordered copy digests the
    same). The dtype is part of it, so the same labels as ``uint16`` and
    ``uint32`` differ: pass the array exactly as the store writes it.
    """
    import hashlib

    import numpy as np

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(
        f"{contiguous.dtype.str}|{list(contiguous.shape)}|".encode("ascii")
    )
    digest.update(memoryview(contiguous).cast("B") if contiguous.size else b"")
    return digest.hexdigest()


def objmap_digest(objmap: np.ndarray) -> str:
    """The digest a crop embedding entry binds to: :func:`array_digest` of the objmap.

    *objmap* is the level-0 label array as the store writes it. The writer
    does not cast it (``_write_series`` writes ``image.objmap[:]``'s own
    dtype, ``uint16``), so one set of labels has one digest.
    """
    return array_digest(objmap)


def refresh_embedding_staleness(
    runs: Mapping[str, Mapping[str, object]],
    *,
    objmap_sha256: str | None,
    array_sha256: Mapping[str, str],
) -> None:
    """Set every embedding entry's ``stale`` from its ``binds`` (spec D7), in place.

    **Recomputed on every write, never latched.** A staged run's Stage 1
    writes an all-zeros objmap, so its crop entries read stale; Stage 3's
    identical objmap clears the flag again. The files are kept either way.

    - **Crops** (``binds.objmap_sha256``): stale when it differs from
      *objmap_sha256*, the objmap being written; ``None`` (no objmap written)
      is stale.
    - **Grid** (``binds.array_sha256``): stale when it differs from
      ``array_sha256[entry.input.series]``, the level-0 array being written;
      a series not being written is stale.
    - An entry with neither binding cannot be verified and reads stale.

    Args:
        runs: ``{run_id: run_entry}``; each entry's ``embeddings`` mapping is
            updated in place.
        objmap_sha256: :func:`objmap_digest` of the objmap being written.
        array_sha256: :func:`array_digest` of each level-0 series being
            written, keyed by series name.
    """
    from ._table_runs import EMBEDDINGS_KEY

    for entry in runs.values():
        embeddings = entry.get(EMBEDDINGS_KEY)
        if not isinstance(embeddings, dict):
            continue
        for embedding in embeddings.values():
            if not isinstance(embedding, dict):
                continue
            binds = embedding.get("binds")
            binds = binds if isinstance(binds, Mapping) else {}
            if "objmap_sha256" in binds:
                current: str | None = objmap_sha256
                bound = binds["objmap_sha256"]
            elif "array_sha256" in binds:
                source = embedding.get("input")
                series = source.get("series") if isinstance(source, Mapping) else None
                current = array_sha256.get(series) if isinstance(series, str) else None
                bound = binds["array_sha256"]
            else:
                embedding["stale"] = True
                continue
            embedding["stale"] = current is None or bound != current


def grid_bound_series(runs: Mapping[str, Mapping[str, object]]) -> set[str]:
    """The series a grid embedding entry in *runs* binds to (those worth digesting)."""
    from ._table_runs import EMBEDDINGS_KEY

    series: set[str] = set()
    for entry in runs.values():
        embeddings = entry.get(EMBEDDINGS_KEY)
        if not isinstance(embeddings, Mapping):
            continue
        for embedding in embeddings.values():
            if not isinstance(embedding, Mapping):
                continue
            binds = embedding.get("binds")
            source = embedding.get("input")
            if (
                isinstance(binds, Mapping)
                and "array_sha256" in binds
                and isinstance(source, Mapping)
                and isinstance(source.get("series"), str)
            ):
                series.add(str(source["series"]))
    return series


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
    *,
    run_id: str,
    objmap_target: str,
) -> Path:
    """Write run *run_id*'s measurement Parquet and the groups above it (spec D1, D4).

    Writes ``tables/<run_id>/measurements/table.parquet`` with every level a
    Zarr v3 group: ``tables/`` and ``tables/<run_id>/`` plain, and
    ``measurements/`` carrying :func:`ngff_feature_table_attrs`.

    Args:
        store_part: An **unpromoted** ``*.ome.zarr.part`` directory.
        table: The prepared payload.
        run_id: The run folder (:func:`~phenotypic.sdk_.choose_table_run_id`).
        objmap_target: Store-relative path of the objmap label the table's
            ``Object_Label`` column indexes.

    Returns:
        The written Parquet path.
    """
    from . import ngff_
    from ._table_runs import measurement_table_relative_path

    payload = Path(store_part) / measurement_table_relative_path(run_id)
    _ensure_group(Path(store_part) / ngff_.TABLES_GROUP)
    _ensure_group(payload.parent.parent)
    _ensure_group(
        payload.parent, _measurement_group_document(run_id, objmap_target)
    )
    _write_validated_parquet(payload, table)
    return payload


def write_metadata_table(
    store_part: Path,
    tables: PreparedImageTables,
    *,
    run_id: str,
) -> Path:
    """Write ``tables/<run_id>/metadata/pht-metadata.parquet`` and its groups.

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
        run_id: The run folder the table belongs to.

    Returns:
        The written Parquet path.

    Raises:
        ValueError: If *tables* carries no metadata frame.
    """
    from . import ngff_
    from ._table_runs import metadata_table_relative_path

    if tables.metadata is None:
        raise ValueError("PreparedImageTables carries no metadata table")
    payload = Path(store_part) / metadata_table_relative_path(run_id)
    _ensure_group(Path(store_part) / ngff_.TABLES_GROUP)
    _ensure_group(payload.parent.parent)
    _ensure_group(payload.parent)
    _write_validated_frame(
        payload, tables.metadata, tables.metadata_parquet_metadata()
    )
    return payload


def build_measurement_table_descriptor(
    table: PreparedEmbeddedMeasurementTable,
    *,
    objmap_target: str,
    run_id: str | None = None,
) -> dict[str, object]:
    """Build the stable descriptor for one embedded measurement table.

    Args:
        table: The prepared payload.
        objmap_target: Store-relative path of the objmap label.
        run_id: The run folder the table lives in. ``None`` describes the
            **v1 flat** ``tables/measurements/table.parquet``, which only the
            legacy :func:`replace_embedded_measurement_table` still writes.
    """
    from phenotypic.schema import OBJECT

    from . import ngff_
    from ._table_runs import measurement_table_relative_path

    path = (
        ngff_.MEASUREMENT_TABLE_RELATIVE_PATH.as_posix()
        if run_id is None
        else measurement_table_relative_path(run_id).as_posix()
    )
    return {
        "schema_version": ngff_.MEASUREMENT_TABLE_SCHEMA_VERSION,
        "type": "object_measurements",
        "format": "parquet",
        "path": path,
        "measurement_columns": list(table.measurement_columns),
        "target": {
            "column": str(OBJECT.LABEL),
            "path": objmap_target,
        },
    }


def build_metadata_table_descriptor(
    tables: PreparedImageTables,
    *,
    run_id: str | None = None,
) -> dict[str, object]:
    """Build the stable descriptor for one embedded metadata table.

    It records **what the file is**, not how it joins: the join keys are on
    the Parquet's own key/value metadata and in the run's ``metadata_table``
    block, and repeating them a third time here would be a third home for one
    fact.

    Args:
        tables: The split payload; ``tables.metadata`` must not be ``None``.
        run_id: The run folder the table lives in; ``None`` describes the v1
            flat ``tables/metadata/pht-metadata.parquet``.
    """
    from . import ngff_
    from ._table_runs import metadata_table_relative_path

    if tables.metadata is None:
        raise ValueError("PreparedImageTables carries no metadata table")
    path = (
        ngff_.METADATA_TABLE_RELATIVE_PATH.as_posix()
        if run_id is None
        else metadata_table_relative_path(run_id).as_posix()
    )
    return {
        "schema_version": ngff_.METADATA_TABLE_SCHEMA_VERSION,
        "type": "image_metadata",
        "format": "parquet",
        "path": path,
        "metadata_columns": [
            str(column) for column in tables.metadata.columns
        ],
    }


def build_metadata_table_block(
    tables: PreparedImageTables,
) -> dict[str, object] | None:
    """Return a run's ``metadata_table`` block, or ``None`` to omit it.

    D-A: a store keeps the metadata snapshot it was built against and says
    which one, so ``resolve_run_state`` can **derive** the divergence advisory
    instead of tracking a backfill stage. Since #281 P3 (spec D3) the block
    lives in its run entry, ``tables.runs.<run>.metadata_table``, not at the
    root.

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
    run_id: str,
) -> dict[str, object]:
    """Return the table fields of run *run_id*'s entry that *tables* implies.

    ``measurements``, ``metadata`` when there is a metadata frame, and
    ``metadata_table`` unless the join was ``not_requested``. A **total**
    description of the run's tables, not an increment: a run re-measured
    without ``--metadata`` loses its ``metadata`` descriptor and its
    ``metadata_table`` block, because the run entry is rebuilt from this
    fragment rather than updated by it (see :func:`apply_image_tables_attributes`).
    """
    from . import ngff_

    fields: dict[str, object] = {
        ngff_.MEASUREMENT_TABLE_GROUP: build_measurement_table_descriptor(
            tables.measurements_payload(),
            objmap_target=objmap_target,
            run_id=run_id,
        )
    }
    if tables.metadata is not None:
        fields[ngff_.METADATA_TABLE_GROUP] = build_metadata_table_descriptor(
            tables, run_id=run_id
        )
    block = build_metadata_table_block(tables)
    if block is not None:
        fields[ngff_.PhenotypicAttr.METADATA_TABLE] = block
    return fields


def write_run_entry(
    part: Path,
    run_id: str,
    *,
    order: int,
    pipeline_sha256: str | None,
    measurements: PreparedImageTables | None = None,
    objmap_target: str | None = None,
    embeddings: Mapping[str, Mapping] | None = None,
    objmap_sha256: str | None = None,
    array_sha256: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Write run *run_id*'s tables into *part* and return its descriptor entry.

    The single owner of a run entry (spec D3): every writer that produces a
    run's tables -- the store save, the measure rewrite, the migrate import --
    goes through here, so a run folder's files and the entry describing them
    cannot disagree. Every subtable is optional, so a store holding embeddings
    and no measurements (a process store) has an entry too.

    The folder must not already exist in *part*: a caller replacing a run
    removes the part's hard-linked copy first, so every file is a new inode.

    Args:
        part: An **unpromoted** ``*.ome.zarr.part`` directory.
        run_id: The run folder, ``<order>-<sha12>`` or ``<order>-unknown``.
        order: The run's call-order number; must be *run_id*'s.
        pipeline_sha256: The run's pipeline digest, or ``None``; must agree
            with *run_id*.
        measurements: The run's measurement and metadata tables, or ``None``.
        objmap_target: Store-relative objmap label path; required with
            *measurements*.
        embeddings: Already-written embedding entries, keyed
            ``"<model>/<region>"``. Their files must already be in *part*
            under this run's folder. (#281 P4 adds the payload writer.)
        objmap_sha256: :func:`objmap_digest` of the objmap this write
            certifies; with *array_sha256*, recomputes each embedding's
            ``stale`` (D7). Both ``None`` leaves ``stale`` as given.
        array_sha256: :func:`array_digest` of each level-0 series written,
            keyed by series name.

    Returns:
        The run entry: ``order``, ``pipeline_sha256``, and the descriptors of
        the tables written.

    Raises:
        ValueError: If *run_id* disagrees with *order* or *pipeline_sha256*,
            or *measurements* is given without *objmap_target*.
    """
    from . import ngff_
    from ._table_runs import (
        EMBEDDINGS_KEY,
        ORDER_KEY,
        PIPELINE_SHA256_KEY,
        table_run_id,
    )

    if table_run_id(order, pipeline_sha256) != run_id:
        raise ValueError(
            f"run id {run_id!r} is not order {order} of pipeline "
            f"{pipeline_sha256!r}"
        )
    part = Path(part)
    _ensure_group(part / ngff_.TABLES_GROUP)
    _ensure_group(part / ngff_.TABLES_GROUP / run_id)
    entry: dict[str, object] = {
        ORDER_KEY: order,
        PIPELINE_SHA256_KEY: pipeline_sha256,
    }
    if measurements is not None:
        if objmap_target is None:
            raise ValueError("a run's measurements need an objmap_target")
        write_embedded_measurement_table(
            part,
            measurements.measurements_payload(),
            run_id=run_id,
            objmap_target=objmap_target,
        )
        if measurements.metadata is not None:
            write_metadata_table(part, measurements, run_id=run_id)
        entry.update(
            build_image_tables_attributes(
                measurements, objmap_target=objmap_target, run_id=run_id
            )
        )
    if embeddings:
        entry[EMBEDDINGS_KEY] = {
            key: deepcopy(dict(value)) for key, value in embeddings.items()
        }
        if objmap_sha256 is not None or array_sha256 is not None:
            refresh_embedding_staleness(
                {run_id: entry},
                objmap_sha256=objmap_sha256,
                array_sha256=array_sha256 or {},
            )
    return entry


def write_image_tables(
    store_part: Path,
    tables: PreparedImageTables,
    *,
    objmap_target: str,
    run_id: str,
    order: int,
    pipeline_sha256: str | None,
) -> dict[str, object]:
    """Write one image's measurement and metadata tables as run *run_id*.

    A thin delegate to :func:`write_run_entry` for the common case of a run
    whose only tables are its measurements.

    Args:
        store_part: An **unpromoted** ``*.ome.zarr.part`` directory. Both
            tables land here, before the root ``zarr.json`` (D-A).
        tables: The split payload.
        objmap_target: Store-relative path of the label image the measurement
            table's ``Object_Label`` column indexes.
        run_id: The run folder.
        order: The run's call-order number.
        pipeline_sha256: The run's pipeline digest, or ``None``.

    Returns:
        The run entry, to pass to :func:`apply_image_tables_attributes`.
    """
    return write_run_entry(
        store_part,
        run_id,
        order=order,
        pipeline_sha256=pipeline_sha256,
        measurements=tables,
        objmap_target=objmap_target,
    )


def apply_image_tables_attributes(
    phenotypic: dict[str, object],
    runs: Mapping[str, Mapping[str, object]],
    *,
    upstream: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    """Make *phenotypic*'s ``tables`` the v2 descriptor of exactly *runs* and *upstream*.

    **Stated as a total function of the payload, never as an append.** The
    caller passes every run the store keeps -- the carried ones and its own
    -- and nothing else survives. That is what lets a run re-measured
    **without** ``--metadata`` lose its ``metadata`` descriptor and its
    ``metadata_table`` block (H2): an append-only refresh would leave the
    stale digest in place and the divergence advisory would fire on that
    store forever.

    The root-level ``metadata_table`` block of the v1 layout is removed: it
    moved into each run entry (spec D3). With no runs and no upstream the
    ``tables`` key is removed too, so a table-less store declares nothing.

    Runs are emitted by ascending ``order`` so two identical writes produce
    byte-identical roots.
    """
    from . import ngff_
    from ._table_runs import (
        ORDER_KEY,
        RUNS_KEY,
        SCHEMA_VERSION_KEY,
        UPSTREAM_KEY,
    )

    phenotypic.pop(ngff_.PhenotypicAttr.METADATA_TABLE, None)
    upstream = dict(upstream or {})
    if not runs and not upstream:
        phenotypic.pop(ngff_.PhenotypicAttr.TABLES, None)
        return
    def _order(item: tuple[str, Mapping[str, object]]) -> int:
        order = item[1].get(ORDER_KEY)
        return order if isinstance(order, int) else 0

    ordered = sorted(runs.items(), key=_order)
    phenotypic[ngff_.PhenotypicAttr.TABLES] = {
        SCHEMA_VERSION_KEY: ngff_.TABLES_SCHEMA_VERSION,
        RUNS_KEY: {run_id: dict(entry) for run_id, entry in ordered},
        UPSTREAM_KEY: {key: dict(value) for key, value in sorted(upstream.items())},
    }


def _link_tree(source: Path, target: Path) -> None:
    """Hard-link every file under *source* to the same place under *target*.

    The files are never written through: a writer that replaces one first
    unlinks the part's copy, and :func:`atomic_write_json` replaces by rename.
    """
    from . import ngff_

    root = ngff_.long_path(source)
    for directory, _subdirectories, filenames in os.walk(root):
        destination = Path(ngff_.long_path(target)) / os.path.relpath(
            directory, root
        )
        destination.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            _clone_file_without_pixel_rewrite(
                str(Path(directory) / filename), str(destination / filename)
            )


def _replaced_store_block(final: Path) -> Mapping[str, object] | None:
    """The ``attributes.phenotypic`` block of the store a save replaces, or ``None``.

    ``None`` -- nothing to carry, and nothing to number a new folder after --
    when there is no store at *final*, when its root is unparseable (logged),
    or when the root has no ``phenotypic`` block: a third-party or half-built
    store that a save (``--mode migrate`` rebuilding an invalid store from its
    HDF source, say) replaces whole, as it always has.
    """
    from . import ngff_

    try:
        block = ngff_.read_phenotypic_attributes(Path(final))
    except (FileNotFoundError, NotADirectoryError, KeyError):
        return None
    except ValueError as exc:  # json.JSONDecodeError included
        logger.warning("Cannot carry tables from %s: %s", final, exc)
        return None
    return block if isinstance(block, Mapping) else None


def carry_table_runs(
    final: Path, part: Path, *, exclude: str | None
) -> dict[str, dict[str, dict[str, object]]]:
    """Carry every run folder of *final* but *exclude* into *part* (spec D5).

    Used when a store is rewritten from scratch over an existing one (a
    ``save2zarr`` over an existing store: ``--restart``, Stage 3 over
    Stage 1, a re-derived process store). The other runs' files are
    hard-linked (or copied) across byte for byte, with their descriptor
    entries unchanged, and the ``upstream`` block comes with them.

    *exclude* is the run the caller writes itself in this transaction. A
    write that produces no tables of its own passes ``None`` and so carries
    **every** folder, its own pipeline's included -- the rule figures follow.

    A carried run whose folder is missing on disk keeps its entry, with a
    warning: dropping it would wipe history, and a reader that opens the file
    finds it absent rather than silently gone from the descriptor.

    A store in the **v1 flat** layout carries nothing, with a warning naming
    ``--mode migrate``: its tables are not run folders, and relabelling them
    as one would be a guess. Forward CLI modes refuse such a store before
    they reach here; this is the notebook path, which today's save already
    behaves like (a save never carried v1 tables).

    Args:
        final: The store being replaced. Absent or unreadable carries nothing.
        part: The unpromoted part being written.
        exclude: The run folder the caller writes itself, or ``None``.

    Returns:
        ``{"runs": {run_id: entry}, "upstream": {...}}`` of what was carried.

    Raises:
        ValueError: If *final*'s tables descriptor is corrupt or a version
            this build does not read. A store is never rewritten over a
            descriptor it cannot interpret.
    """
    from . import ngff_
    from ._table_runs import (
        RUNS_KEY,
        UPSTREAM_KEY,
        table_runs_of_block,
        tables_schema_of_block,
    )

    carried: dict[str, dict[str, dict[str, object]]] = {
        RUNS_KEY: {},
        UPSTREAM_KEY: {},
    }
    block = _replaced_store_block(final)
    if block is None:
        return carried
    schema = tables_schema_of_block(block)
    if schema == 1:
        logger.warning(
            "Not carrying the v1 flat tables of %s into the rewritten store: "
            "convert the tree with `python -m phenotypic --mode migrate` first",
            final,
        )
    if schema != 2:
        return carried
    tables = block[ngff_.PhenotypicAttr.TABLES]
    assert isinstance(tables, Mapping)
    for run in table_runs_of_block(Path(final), block):
        if run.run_id == exclude:
            continue
        if run.folder.is_dir():
            _link_tree(run.folder, Path(part) / ngff_.TABLES_GROUP / run.run_id)
        else:
            logger.warning(
                "Carrying the entry of table run %s although %s is missing",
                run.run_id,
                run.folder,
            )
        carried[RUNS_KEY][run.run_id] = deepcopy(dict(run.descriptor))
    upstream = tables.get(UPSTREAM_KEY)
    if isinstance(upstream, Mapping):
        carried[UPSTREAM_KEY] = deepcopy(dict(upstream))
    if carried[RUNS_KEY]:
        _ensure_group(Path(part) / ngff_.TABLES_GROUP)
    return carried


def _valid_metadata_table_contract(
    store: Path,
    container: Mapping[str, object],
    *,
    run_id: str | None = None,
) -> bool:
    """Return whether a store's metadata table, group, and descriptor agree.

    *container* holds the ``metadata`` descriptor: the run entry of *run_id*
    on a v2 store, the v1 ``tables`` mapping when *run_id* is ``None``.

    **Conditioned on the descriptor being declared**, and deliberately so: a
    store written before the inversion has no ``metadata`` descriptor and no
    metadata group, and is not thereby invalid -- it is a store from before
    the table existed. Only a store that *claims* the table is held to it
    (M2).
    """
    from . import ngff_
    from ._table_runs import metadata_table_relative_path

    descriptor = container.get(ngff_.METADATA_TABLE_GROUP)
    if descriptor is None:
        return True
    if not isinstance(descriptor, dict):
        return False
    columns = descriptor.get("metadata_columns")
    if not isinstance(columns, list) or not all(
        isinstance(column, str) for column in columns
    ):
        return False
    relative = (
        ngff_.METADATA_TABLE_RELATIVE_PATH.as_posix()
        if run_id is None
        else metadata_table_relative_path(run_id).as_posix()
    )
    if descriptor != {
        "schema_version": ngff_.METADATA_TABLE_SCHEMA_VERSION,
        "type": "image_metadata",
        "format": "parquet",
        "path": relative,
        "metadata_columns": columns,
    }:
        return False

    payload = store / relative
    document = json.loads(
        (payload.parent / ngff_.STORE_ROOT_JSON).read_text(encoding="utf-8")
    )
    if document != _TABLES_GROUP_DOCUMENT:
        return False

    import pyarrow.parquet as pq  # type: ignore[import-untyped]

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


def _valid_measurement_payload(payload: Path, columns: list[str]) -> bool:
    """Return whether a measurement Parquet carries *columns* and valid join provenance."""
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    from . import ngff_

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


def _valid_measurement_descriptor(
    descriptor: object, *, relative_path: str, objmap_target: object
) -> list[str] | None:
    """The descriptor's ``measurement_columns`` when it is exactly the stable form, else ``None``."""
    from phenotypic.schema import OBJECT

    from . import ngff_

    if not isinstance(descriptor, dict):
        return None
    columns = descriptor.get("measurement_columns")
    if not isinstance(columns, list) or not all(
        isinstance(column, str) for column in columns
    ):
        return None
    if descriptor != {
        "schema_version": ngff_.MEASUREMENT_TABLE_SCHEMA_VERSION,
        "type": "object_measurements",
        "format": "parquet",
        "path": relative_path,
        "measurement_columns": columns,
        "target": {
            "column": str(OBJECT.LABEL),
            "path": objmap_target,
        },
    }:
        return None
    return columns


def _group_document_is(directory: Path, expected: Mapping[str, object]) -> bool:
    from . import ngff_

    document = json.loads(
        (directory / ngff_.STORE_ROOT_JSON).read_text(encoding="utf-8")
    )
    return document == expected


def _valid_v1_embedded_measurement_contract(
    store: Path, phenotypic: Mapping[str, object]
) -> bool:
    """The contract of a **v1 flat** store: ``tables/measurements/table.parquet``."""
    from . import ngff_

    tables = phenotypic[ngff_.PhenotypicAttr.TABLES]
    assert isinstance(tables, Mapping)
    objmap_target = phenotypic[ngff_.PhenotypicAttr.LABELS][ngff_.OBJMAP_LABEL]  # type: ignore[index]
    columns = _valid_measurement_descriptor(
        tables[ngff_.MEASUREMENT_TABLE_GROUP],
        relative_path=ngff_.MEASUREMENT_TABLE_RELATIVE_PATH.as_posix(),
        objmap_target=objmap_target,
    )
    if columns is None:
        return False
    for group in (
        store / ngff_.TABLES_GROUP,
        store / ngff_.TABLES_GROUP / ngff_.MEASUREMENT_TABLE_GROUP,
    ):
        if not _group_document_is(group, _TABLES_GROUP_DOCUMENT):
            return False
    if not _valid_metadata_table_contract(store, tables):
        return False
    return _valid_measurement_payload(
        store / ngff_.MEASUREMENT_TABLE_RELATIVE_PATH, columns
    )


def _valid_v2_embedded_measurement_contract(
    store: Path, phenotypic: Mapping[str, object], run_id: str | None
) -> bool:
    """The contract of run *run_id* (default: the latest with measurements) of a v2 store."""
    from . import ngff_
    from ._table_runs import (
        measurement_table_relative_path,
        table_runs_of_block,
    )

    runs = table_runs_of_block(store, phenotypic)
    if run_id is None:
        run = next((r for r in reversed(runs) if r.has_measurements), None)
    else:
        run = next((r for r in runs if r.run_id == run_id), None)
    if run is None or not run.has_measurements:
        return False
    objmap_target = phenotypic[ngff_.PhenotypicAttr.LABELS][ngff_.OBJMAP_LABEL]  # type: ignore[index]
    columns = _valid_measurement_descriptor(
        run.descriptor[ngff_.MEASUREMENT_TABLE_GROUP],
        relative_path=measurement_table_relative_path(run.run_id).as_posix(),
        objmap_target=objmap_target,
    )
    if columns is None or not isinstance(objmap_target, str):
        return False
    if not _group_document_is(store / ngff_.TABLES_GROUP, _TABLES_GROUP_DOCUMENT):
        return False
    if not _group_document_is(run.folder, _TABLES_GROUP_DOCUMENT):
        return False
    if not _group_document_is(
        run.folder / ngff_.MEASUREMENT_TABLE_GROUP,
        _measurement_group_document(run.run_id, objmap_target),
    ):
        return False
    if not _valid_metadata_table_contract(store, run.descriptor, run_id=run.run_id):
        return False
    return _valid_measurement_payload(run.measurements_path(), columns)


def _valid_embedded_measurement_contract(
    store_path: Path, run_id: str | None = None
) -> bool:
    """Return whether a store's table payload, groups, and descriptor agree.

    A v2 store is checked at run *run_id*, or by default at its latest run
    that holds ``measurements`` (spec D6a's record-less reader default). A v1
    flat store -- which only ``--mode migrate`` still reads -- is checked at
    ``tables/measurements/``. A store with no tables, a corrupt descriptor or
    any unreadable piece is not valid.
    """
    from . import ngff_
    from ._table_runs import tables_schema_of_block

    try:
        store = Path(store_path)
        root = json.loads(
            (store / ngff_.STORE_ROOT_JSON).read_text(encoding="utf-8")
        )
        phenotypic = root["attributes"][ngff_.PhenotypicAttr.ROOT]
        schema = tables_schema_of_block(phenotypic)
        if schema == 1:
            return _valid_v1_embedded_measurement_contract(store, phenotypic)
        if schema == 2:
            return _valid_v2_embedded_measurement_contract(
                store, phenotypic, run_id
            )
        return False
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
    clear_figure_run: str | None = None,
    clear_table_run: str | Callable[[], str | None] | None = None,
) -> Path:
    """Re-promote one store with refreshed embedded tables, root written last.

    *plan* is handed the store's ``attributes.phenotypic`` block. It validates
    and **mutates** it -- before any part exists, so a refusal costs nothing --
    and returns the writer that populates the part.

    *clear_table_run* drops the part's copy of that one run folder,
    ``tables/<run>/``, before *plan*'s writer rewrites it (spec D5); every
    other run folder is carried. It may name a subfolder,
    ``<run>/measurements``, to rewrite only that table of the run. ``None``
    drops the whole ``tables/`` tree, which only the v1 -> v2 conversion
    wants (it rebuilds the run folder from the flat one). A callable is resolved after
    *plan* has run, so a plan that chooses the run from the root it is handed
    can name it without a second read of the root.

    *clear_figure_run* drops the part's copy of that one run folder,
    ``figures/<run>/``, before *plan*'s writer rewrites it (spec 2026-09-22
    §1a). Every other run folder is carried across unchanged, as are all of
    them when it is ``None``.

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
    if callable(clear_table_run):
        clear_table_run = clear_table_run()

    part = ngff_.new_part_path(store_path)
    try:
        shutil.copytree(
            store_path,
            part,
            copy_function=_clone_file_without_pixel_rewrite,
        )
        # A transaction is readable only after its refreshed root is written last.
        (part / ngff_.STORE_ROOT_JSON).unlink()
        # The copied tables go with it: a run that LOSES its metadata table
        # must lose the file too, not only the descriptor, and a run being
        # rewritten must get new inodes (spec D5/D6). These are hard links
        # into the promoted store, so unlinking them in the part leaves the
        # live store untouched. Only that one run folder goes; every other
        # run is carried as it is.
        if clear_table_run is None:
            shutil.rmtree(part / ngff_.TABLES_GROUP, ignore_errors=True)
        else:
            shutil.rmtree(
                ngff_.long_path(part / ngff_.TABLES_GROUP / clear_table_run),
                ignore_errors=True,
            )
        if clear_figure_run is not None:
            # The copied figure files are HARD LINKS into the live store, like
            # everything copytree cloned above. Removing this run's folder
            # here means it is rewritten as new files; writing through a link
            # would change the published store before its new root exists.
            # Other runs' folders are never touched (spec §1a).
            shutil.rmtree(
                ngff_.long_path(part / ngff_.FIGURES_GROUP / clear_figure_run),
                ignore_errors=True,
            )
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
    pipeline_sha256: str | None,
    figures: StoredFigures | None = None,
    objmap_target: str | None = None,
    durable: bool | None = None,
    commit_guard: CommitGuard | None = None,
) -> TableRun:
    """Refresh the measurement AND metadata tables of one pipeline's run folder.

    The ``--mode measure`` analogue of the promote-time writer: both tables
    and the run's ``metadata_table`` block move as one root-last
    transaction, so the store never certifies a table it does not have or a
    snapshot it was not built against. When *figures* is given, its run
    folder is written in the same transaction.

    **Which folder** (spec D6): the one *pipeline_sha256* owns, chosen by
    :func:`~phenotypic.sdk_._table_runs.choose_table_run_id`. A store that
    already has a folder for this pipeline gets it replaced, keeping its
    number, its files rewritten as new inodes and no row of the previous
    write surviving. Otherwise a new folder takes the next number. ``None``
    replaces the highest-numbered ``-unknown`` folder, or adds one. Every
    other run folder, and the ``upstream`` block, is carried as hard links
    (spec D5), byte for byte.

    The rewrite writes no objmap, so it recomputes no embedding's ``stale``
    flag (spec D7): a carried run keeps its entry exactly. The replaced run's
    entry is rebuilt from *tables* alone -- a **total** description, as
    :func:`build_image_tables_attributes` states -- so an embedding entry it
    held is not carried into the rewrite. (#281 P4 adds ``embeddings=`` here.)

    A store still in the **v1 flat** layout is rewritten whole: its flat
    files go, and the new folder, numbered 1, is the store's only one.
    Forward CLI modes refuse such a store before they reach here (#281 P3
    Task 5).

    Args:
        store_path: A promoted ``*.ome.zarr`` store.
        tables: The split payload to write.
        pipeline_sha256: The sha256 of the pipeline file whose measurers
            produced *tables* -- for ``--mode measure``, the file measure
            mode runs, **not** the one the store's journal names (measure
            mode opens no journal application). ``None`` when no digest is
            known; one is never fabricated (spec D2).
        figures: One run's per-image figures. They replace that run's folder
            and descriptor entry only; every other run is carried across
            unchanged (spec 2026-09-22 §1a). ``None`` adds no run and
            touches no figure, and so does a store whose figures descriptor
            has a ``schema_version`` this writer does not know.
        objmap_target: Store-relative path of the label image the measurement
            table indexes. ``None`` reads it from the store.
        durable: ``fsync`` before promoting. ``None`` auto-detects SLURM.
        commit_guard: Publication guard, checked at the commit point.

    Returns:
        The run folder written, as the promoted root declares it. Its
        :meth:`~phenotypic.sdk_.TableRun.measurements_path` is the table a
        per-image record certifies after the rewrite (spec D6a).

    Raises:
        ValueError: If the store declares no objmap target, or its tables
            descriptor is corrupt or a version this build does not read.
    """

    from . import ngff_
    from ._table_runs import (
        RUNS_KEY,
        UPSTREAM_KEY,
        TABLE_RUN_ID_RE,
        TableRun,
        choose_table_run_id,
        tables_schema_of_block,
    )

    selected: dict[str, object] = {}

    def _plan(phenotypic: dict[str, object]) -> Callable[[Path], None]:
        target = _resolve_objmap_target(phenotypic, objmap_target)
        schema = tables_schema_of_block(phenotypic)
        # A v1 flat store numbers as an empty one: its flat tables are not
        # run folders, and the rewrite replaces them whole.
        run_id = choose_table_run_id(
            phenotypic if schema == 2 else {}, pipeline_sha256=pipeline_sha256
        )
        selected["clear"] = run_id if schema == 2 else None
        tables_block = phenotypic.get(ngff_.PhenotypicAttr.TABLES)
        kept = (
            {
                key: dict(value)
                for key, value in tables_block.get(RUNS_KEY, {}).items()
                if key != run_id
            }
            if schema == 2 and isinstance(tables_block, Mapping)
            else {}
        )
        upstream = (
            dict(tables_block.get(UPSTREAM_KEY, {}))
            if schema == 2 and isinstance(tables_block, Mapping)
            else {}
        )
        match = TABLE_RUN_ID_RE.match(run_id)
        assert match is not None
        order = int(match["order"])

        def _populate(part: Path) -> None:
            # `write_run_entry` returns the entry describing exactly the files
            # it just wrote, so the root cannot certify a table the part does
            # not have -- and `apply_` keeps only the runs it is given, which
            # is how a run that loses its metadata loses the descriptor.
            own = write_run_entry(
                part,
                run_id,
                order=order,
                pipeline_sha256=pipeline_sha256,
                measurements=tables,
                objmap_target=target,
            )
            apply_image_tables_attributes(
                phenotypic, {**kept, run_id: own}, upstream=upstream
            )
            selected["run"] = TableRun(
                store=Path(store_path),
                run_id=run_id,
                order=order,
                pipeline_sha256=pipeline_sha256,
                descriptor=deepcopy(own),
            )
            if figures is None:
                return
            from ._image_figures import (
                apply_image_figures_attributes,
                write_image_figures,
            )

            apply_image_figures_attributes(
                phenotypic, write_image_figures(part, figures)
            )

        return _populate

    if figures is not None:
        from ._image_figures import known_figures_schema, read_image_figures_descriptor

        descriptor = read_image_figures_descriptor(Path(store_path))
        if not known_figures_schema(descriptor):
            # Clearing "this run's folder" in a layout this writer cannot read
            # could delete anything; the store's figures stay as they are.
            logger.warning(
                "Adding no figure run to %s: its figures schema_version %r is "
                "not one this writer knows (%r)",
                store_path, descriptor.get("schema_version"),
                sorted(ngff_.READABLE_FIGURES_SCHEMA_VERSIONS),
            )
            figures = None

    def _clear() -> str | None:
        clear = selected["clear"]
        assert clear is None or isinstance(clear, str)
        return clear

    _rewrite_store_tables(
        store_path,
        plan=_plan,
        durable=durable,
        commit_guard=commit_guard,
        clear_figure_run=figures.run.run_id if figures is not None else None,
        clear_table_run=_clear,
    )
    run = selected["run"]
    assert isinstance(run, TableRun)
    return run


def _v1_run_entry(
    phenotypic: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    """The run id and v2 run entry a **v1 flat** *phenotypic* block converts to (spec D10).

    The folder is ``1-<sha12>`` for
    :func:`~phenotypic.sdk_._table_runs.v1_table_pipeline_sha`, or
    ``1-unknown``: a v1 store has no runs and no upstream, so its one folder
    is numbered 1. The entry keeps each v1 descriptor as it is, with ``path``
    repointed at the run folder, and takes the root-level ``metadata_table``
    block into the run (spec D3).

    Raises:
        ValueError: If the v1 ``tables`` mapping declares anything but
            ``measurements`` and ``metadata``, or a descriptor that is not a
            mapping. A table this converter does not know is refused rather
            than dropped.
    """
    from . import ngff_
    from ._table_runs import (
        ORDER_KEY,
        PIPELINE_SHA256_KEY,
        measurement_table_relative_path,
        metadata_table_relative_path,
        table_run_id,
        v1_table_pipeline_sha,
    )

    tables = phenotypic.get(ngff_.PhenotypicAttr.TABLES)
    assert isinstance(tables, Mapping)
    unknown = sorted(
        set(tables) - {ngff_.MEASUREMENT_TABLE_GROUP, ngff_.METADATA_TABLE_GROUP}
    )
    if unknown:
        raise ValueError(
            f"v1 tables descriptor declares {unknown!r}, which the v1 -> v2 "
            "conversion does not know"
        )
    sha = v1_table_pipeline_sha(phenotypic)
    run_id = table_run_id(1, sha)
    entry: dict[str, object] = {ORDER_KEY: 1, PIPELINE_SHA256_KEY: sha}
    for group, relative in (
        (ngff_.MEASUREMENT_TABLE_GROUP, measurement_table_relative_path),
        (ngff_.METADATA_TABLE_GROUP, metadata_table_relative_path),
    ):
        descriptor = tables.get(group)
        if descriptor is None:
            continue
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"v1 tables descriptor {group!r} is not a mapping")
        entry[group] = {
            **deepcopy(dict(descriptor)),
            "path": relative(run_id).as_posix(),
        }
    block = phenotypic.get(ngff_.PhenotypicAttr.METADATA_TABLE)
    if isinstance(block, Mapping):
        entry[ngff_.PhenotypicAttr.METADATA_TABLE] = deepcopy(dict(block))
    return run_id, entry


def _move_v1_payloads(
    store: Path,
    part: Path,
    run_id: str,
    entry: Mapping[str, object],
    *,
    objmap_target: str,
    skip_measurements: bool = False,
) -> None:
    """Hard-link a v1 store's flat payloads into run *run_id* of *part*.

    The payloads come from the **live** store at the v1 layout's paths
    (never the descriptor's ``path`` string), and each group above them gets
    its v2 document: the ``measurements/`` group carries the ngio
    feature-table attributes (spec D4). Bytes are linked, never rewritten.
    """
    from . import ngff_
    from ._table_runs import (
        measurement_table_relative_path,
        metadata_table_relative_path,
    )

    _ensure_group(part / ngff_.TABLES_GROUP)
    _ensure_group(part / ngff_.TABLES_GROUP / run_id)
    moves: list[tuple[Path, PurePosixPath, Mapping[str, object]]] = []
    if ngff_.MEASUREMENT_TABLE_GROUP in entry and not skip_measurements:
        moves.append(
            (
                ngff_.MEASUREMENT_TABLE_RELATIVE_PATH,
                measurement_table_relative_path(run_id),
                _measurement_group_document(run_id, objmap_target),
            )
        )
    if ngff_.METADATA_TABLE_GROUP in entry:
        moves.append(
            (
                ngff_.METADATA_TABLE_RELATIVE_PATH,
                metadata_table_relative_path(run_id),
                _TABLES_GROUP_DOCUMENT,
            )
        )
    for source, destination, document in moves:
        target = part / destination
        _ensure_group(target.parent, document)
        _clone_file_without_pixel_rewrite(str(Path(store) / source), str(target))


def convert_tables_v1_to_v2(
    store: Path,
    *,
    durable: bool | None,
    commit_guard: CommitGuard | None = None,
) -> TableRun | None:
    """Convert a **v1 flat** store's tables into run folder ``1-<sha12>`` (spec D10).

    ``tables/measurements`` and ``tables/metadata`` move into
    ``tables/1-<sha12>/`` (or ``1-unknown``, ``pipeline_sha256: null``, when
    :func:`~phenotypic.sdk_._table_runs.v1_table_pipeline_sha` finds no
    digest -- one is never fabricated). The payload bytes are hard-linked,
    not rewritten, and the descriptors are kept as they are with their
    ``path`` repointed; the root-level ``metadata_table`` block moves into the
    run (spec D3). It is a root-last re-promote through
    :func:`_rewrite_store_tables`, and pixel arrays are never rewritten.

    Only ``--mode migrate`` calls this. Converting changes the store root, so
    the caller republishes the image's record and then the run proof.

    Args:
        store: A promoted ``*.ome.zarr`` store.
        durable: ``fsync`` before promoting. ``None`` auto-detects SLURM.
        commit_guard: Publication guard, checked at the commit point.

    Returns:
        The run folder written, or ``None`` when *store* is not in the v1
        layout (already v2, no tables at all, or no ``phenotypic`` block).

    Raises:
        FileNotFoundError: If the store has no root, or a declared v1
            payload is missing.
        ValueError: If the tables descriptor is corrupt, a newer version, or
            a v1 mapping holding a table this converter does not know.
    """
    from . import ngff_
    from ._table_runs import TableRun, tables_schema_of_block

    store = Path(store)
    try:
        block = ngff_.read_phenotypic_attributes(store)
    except KeyError:
        # No `phenotypic` block (a third-party store): no PhenoTypic tables.
        return None
    if tables_schema_of_block(block) != 1:
        return None
    selected: dict[str, TableRun] = {}

    def _plan(phenotypic: dict[str, object]) -> Callable[[Path], None]:
        if tables_schema_of_block(phenotypic) != 1:
            raise ValueError(f"{store} is no longer in the v1 tables layout")
        run_id, entry = _v1_run_entry(phenotypic)
        target = _resolve_objmap_target(phenotypic, None)
        sha = entry["pipeline_sha256"]

        def _populate(part: Path) -> None:
            _move_v1_payloads(store, part, run_id, entry, objmap_target=target)
            apply_image_tables_attributes(phenotypic, {run_id: entry}, upstream={})
            selected["run"] = TableRun(
                store=store,
                run_id=run_id,
                order=1,
                pipeline_sha256=sha if isinstance(sha, str) else None,
                descriptor=deepcopy(entry),
            )

        return _populate

    # `None` clears the part's whole `tables/` tree: the flat folders go, and
    # the run folder is rebuilt from links into the live store.
    _rewrite_store_tables(
        store,
        plan=_plan,
        durable=durable,
        commit_guard=commit_guard,
        clear_table_run=None,
    )
    return selected["run"]


def replace_embedded_measurement_table(
    store_path: Path,
    table: PreparedEmbeddedMeasurementTable,
    *,
    objmap_target: str | None = None,
    durable: bool | None = None,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Replace one store's measurement table without recomputing pixel arrays.

    A complete root-last store transaction whose unchanged files are
    hard-linked where the platform permits.

    **Superseded by** :func:`replace_image_tables` on every forward path. It
    survives for ``--mode migrate``, which imports legacy per-image Parquets
    into converted stores and whose reclaim authority compares a store's
    bytes against that payload. It deliberately leaves the run's ``metadata``
    table and ``metadata_table`` block exactly as it found them: it has no
    metadata payload to describe, and clearing a block it knows nothing about
    would be a guess.

    **It writes the v2 layout** (#281 spec D10), so migrate never leaves a
    tree mixed:

    - the run is the one the store's pipeline owns
      (:func:`~phenotypic.sdk_._table_runs.import_table_run_id`:
      :func:`~phenotypic.sdk_._table_runs.choose_table_run_id` for
      :func:`~phenotypic.sdk_._table_runs.v1_table_pipeline_sha`, the rule
      that names a converted v1 folder). Migrate's checks of the import ask
      the same function, so they judge the run it wrote (review M2). An existing folder for it has its
      ``measurements/`` rewritten as new inodes and keeps everything else;
      otherwise a new folder takes the next number. Every other run folder,
      and the ``upstream`` block, is carried byte for byte;
    - a store still in the **v1 flat** layout is converted in the same
      transaction (as :func:`convert_tables_v1_to_v2` does), with its flat
      metadata table carried into the run and the new measurements in place
      of the flat ones.

    Returns:
        The written measurements Parquet path: the run it wrote, at
        ``tables/<run>/measurements/table.parquet``.
    """
    from . import ngff_
    from ._table_runs import (
        ORDER_KEY,
        PIPELINE_SHA256_KEY,
        RUNS_KEY,
        TABLE_RUN_ID_RE,
        UPSTREAM_KEY,
        import_table_run_id,
        measurement_table_relative_path,
        tables_schema_of_block,
        v1_table_pipeline_sha,
    )

    store_path = Path(store_path)
    selected: dict[str, str | None] = {}

    def _plan(phenotypic: dict[str, object]) -> Callable[[Path], None]:
        target = _resolve_objmap_target(phenotypic, objmap_target)
        schema = tables_schema_of_block(phenotypic)
        kept: dict[str, dict[str, object]] = {}
        upstream: dict[str, Mapping[str, object]] = {}
        if schema == 1:
            run_id, entry = _v1_run_entry(phenotypic)
            # The whole flat `tables/` tree goes; the run folder is rebuilt.
            selected["clear"] = None
        else:
            sha = v1_table_pipeline_sha(phenotypic)
            run_id = import_table_run_id(phenotypic)
            tables_block = phenotypic.get(ngff_.PhenotypicAttr.TABLES)
            if schema == 2 and isinstance(tables_block, Mapping):
                kept = {
                    key: dict(value)
                    for key, value in tables_block.get(RUNS_KEY, {}).items()
                }
                upstream = dict(tables_block.get(UPSTREAM_KEY, {}))
            match = TABLE_RUN_ID_RE.match(run_id)
            assert match is not None
            owned = kept.pop(run_id, None)
            entry = (
                deepcopy(owned)
                if owned is not None
                else {ORDER_KEY: int(match["order"]), PIPELINE_SHA256_KEY: sha}
            )
            # On a v2 store only this run's `measurements/` is rewritten (new
            # inodes); its metadata and embeddings, and every other run, are
            # carried. A store with no tables descriptor declares nothing
            # under `tables/` (a stranded payload, say), so all of it goes.
            selected["clear"] = (
                f"{run_id}/{ngff_.MEASUREMENT_TABLE_GROUP}" if schema == 2 else None
            )
        entry[ngff_.MEASUREMENT_TABLE_GROUP] = build_measurement_table_descriptor(
            table, objmap_target=target, run_id=run_id
        )
        selected["run_id"] = run_id

        def _populate(part: Path) -> None:
            if schema == 1:
                _move_v1_payloads(
                    store_path,
                    part,
                    run_id,
                    entry,
                    objmap_target=target,
                    skip_measurements=True,
                )
            _ensure_group(part / ngff_.TABLES_GROUP)
            _ensure_group(part / ngff_.TABLES_GROUP / run_id)
            write_embedded_measurement_table(
                part, table, run_id=run_id, objmap_target=target
            )
            apply_image_tables_attributes(
                phenotypic, {**kept, run_id: entry}, upstream=upstream
            )

        return _populate

    def _clear() -> str | None:
        return selected["clear"]

    _rewrite_store_tables(
        store_path,
        plan=_plan,
        durable=durable,
        commit_guard=commit_guard,
        clear_table_run=_clear,
    )
    run_id = selected["run_id"]
    assert run_id is not None
    return store_path / measurement_table_relative_path(run_id)


def read_embedded_measurement_descriptor(
    store_path: Path,
) -> dict[str, object]:
    """Return one store's measurement-table descriptor, from its latest measuring run.

    A store holds one run folder per measuring pipeline (#281 spec D1). This
    reader holds no per-image record, so it reads spec D6a's default: the
    ``measurements`` descriptor of
    :func:`~phenotypic.sdk_._table_runs.latest_measurement_run`, the
    highest-numbered run that holds ``measurements``. A newer run without
    measurements (a grid-only process folder) is skipped. A store still in
    the v1 flat layout (tables schema 1, before ``--mode migrate``) reads
    its one flat ``tables.measurements`` descriptor.

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
        KeyError: If the root carries no ``phenotypic`` block, or no run
            of the store declares ``measurements``. **An absent descriptor
            is a normal state**, not a fault: a ``--mode process`` run never
            measures, and a store written before embedded tables has none.
        ValueError: If the store's ``store_schema_version`` is not this
            build's -- the same refusal every other content reader makes --
            or its tables descriptor is corrupt or of an unknown version.

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
    # Gated by VALUE, not read raw: `latest_measurement_run` reads the root
    # through `require_readable_store`, so a store written by a newer build
    # is refused here exactly as it is on every other path that decodes store
    # content, rather than this reader becoming the one place a future
    # schema slips through under today's semantics.
    return _latest_measurement_descriptor(Path(store_path))[1]


def _latest_measurement_descriptor(
    store_path: Path,
) -> tuple[Path, dict[str, object]]:
    """The record-less default table (spec D6a): its payload and its descriptor.

    The latest run that holds ``measurements``, or, for a store still in the
    v1 flat layout, its one flat table
    (:func:`~phenotypic.sdk_._table_runs.declared_measurement_tables`, keyed
    on the tables schema). Read-only: a v1 store is read, never written.

    Raises:
        KeyError: If the store declares no measurement table.
    """
    from . import ngff_
    from ._table_runs import declared_measurement_tables

    declared = declared_measurement_tables(
        store_path, ngff_.require_readable_store(store_path)
    )
    if not declared:
        raise KeyError(
            f"OME-Zarr store declares no embedded measurement table: "
            f"{store_path}"
        )
    table = declared[-1]
    return store_path / table.relative_path, dict(table.descriptor)


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

    Reads the same table as :func:`read_embedded_measurement_descriptor`:
    the store's latest run that holds ``measurements`` (spec D6a), or a v1
    store's flat table.

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


    store_path = Path(store_path)
    payload, descriptor = _latest_measurement_descriptor(store_path)
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

    # `payload` is the same table's, so the columns checked above and the
    # file read below come from one run folder. It is derived from the
    # layout, never from the descriptor's own `path` string, so a
    # request-facing caller cannot be pointed outside the store by a
    # hand-edited root.
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
