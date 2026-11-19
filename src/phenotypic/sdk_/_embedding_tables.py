"""Embedding tables of a table run folder (#281 spec D1, D3, D7, D9; E1, E2).

An embedding belongs to the pipeline that produced it, so its table lives in
that pipeline's run folder::

    tables/<n>-<sha12>/embeddings/<model>/crops/table.parquet   Object_Label | embedding
    tables/<n>-<sha12>/embeddings/<model>/grid/table.parquet    Grid_* keys  | embedding
    tables/<n>-<sha12>/embeddings/<model>/grid-<name>/table.parquet   a named grid, as grid

:class:`EmbeddingPayload` is what :meth:`ImagePipeline.embed` returns and a
store writer consumes: one table's frame plus the descriptor fields the
producer knows (model, vector length, what it read, what it is bound to).
:func:`write_embedding_tables` writes payloads into an unpromoted part and
returns the run entry's ``embeddings`` block; :func:`read_embedding_table`
reads one back, verified.

**The vector column is Arrow ``FixedSizeList<float16>[D]``**, built directly
with an explicit schema. It never goes through ``pa.Table.from_pandas``, which
yields a variable-length ``list<halffloat>`` and pandas schema metadata, and it
never enters the master, the mirror, a split or any TSV (spec D1, E5).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa  # type: ignore[import-untyped]

#: The vector column of every embedding table (spec D1, E5).
EMBEDDING_COLUMN: Final[str] = "embedding"

#: The ``int16`` column of a run's embedding deliverable naming each row's
#: entry descriptor: a 0-based index into that key's ``descriptors`` list in
#: ``deliverables/embeddings/manifest.json`` (#281 P4 review Q1). Never in a
#: store's own table.
DESCRIPTOR_INDEX_COLUMN: Final[str] = "descriptor_index"

#: ``files[].media_type`` of an embedding table (spec D3).
PARQUET_MEDIA_TYPE: Final[str] = "application/vnd.apache.parquet"

#: ``type`` attribute of an embedding table group: a prefixed extension type in
#: RFC-8's style (spec D4).
EMBEDDING_TABLE_TYPE: Final[str] = "phenotypic:embedding_table"

#: The descriptor fields a producer must supply (spec D3); the writer adds
#: ``stale`` and ``files``.
PAYLOAD_DESCRIPTOR_FIELDS: Final[frozenset[str]] = frozenset(
    {"model", "dim", "dtype", "pooling", "key", "input", "producer", "binds"}
)


@dataclass(frozen=True)
class EmbeddingPayload:
    """One embedding table a pipeline produced for one image, not yet written.

    Attributes:
        key: ``"<model>/<region>"``, the entry's key in the run's
            ``embeddings`` block and its folder under ``embeddings/``; the
            region is ``crops``, ``grid`` or ``grid-<name>``.
        frame: The table: its key columns (``Object_Label`` for crops, the
            ``Grid_*`` keys for a grid) and ``embedding``, one 1-D
            ``float16`` vector per row.
        descriptor: The entry fields the producer knows (spec D3): ``model``,
            ``dim``, ``dtype``, ``pooling``, ``key`` (the index column),
            ``input`` (``{series, level}``), ``producer``
            (``{class, params_sha256}``) and ``binds`` -- for crops
            ``{objmap_path, objmap_sha256}``, for a grid
            ``{array_sha256, grid_sha256}``, digests of the image the vectors
            were computed from. The writer adds ``stale`` and ``files``.

    Raises:
        ValueError: If *key* is not ``"<model>/<region>"``
            (:func:`~phenotypic.sdk_._table_runs.split_embedding_key`), or
            *descriptor* lacks a required field.
    """

    key: str
    frame: pd.DataFrame
    descriptor: Mapping[str, object]

    def __post_init__(self) -> None:
        from ._table_runs import split_embedding_key

        split_embedding_key(self.key)
        missing = PAYLOAD_DESCRIPTOR_FIELDS - set(self.descriptor)
        if missing:
            raise ValueError(
                f"embedding payload {self.key!r} lacks descriptor fields {sorted(missing)}"
            )

    @property
    def model(self) -> str:
        """The model folder, ``<model>`` of :attr:`key`."""
        return self.key.partition("/")[0]

    @property
    def region(self) -> str:
        """``"crops"``, ``"grid"`` or ``"grid-<name>"``, the region folder of :attr:`key`."""
        return self.key.partition("/")[2]

    @property
    def dim(self) -> int:
        """The vector length, ``descriptor["dim"]``."""
        return int(self.descriptor["dim"])  # type: ignore[call-overload]


class EmbeddingTableUnreadableError(RuntimeError):
    """An embedding entry cannot be read: stale, missing, or not the bytes it declares.

    Spec D7/D9: a stale entry, or missing or corrupted bytes, make **that
    entry** unreadable. They never revoke the run's measurement authority,
    because embeddings are additive.

    Attributes:
        store: The store.
        key: The entry's ``"<model>/<region>"`` key.
        reason: Why it cannot be read.
    """

    def __init__(self, store: Path, key: str, reason: str) -> None:
        self.store, self.key, self.reason = Path(store), key, reason
        super().__init__(f"embedding table {key!r} of {store} is unreadable: {reason}")

    def __reduce__(self):
        """Rebuild from the constructor's arguments (pickling across workers)."""
        return (type(self), (self.store, self.key, self.reason))


def _vector_type(dim: int) -> pa.DataType:
    import pyarrow as pa  # type: ignore[import-untyped]

    return pa.list_(pa.float16(), dim)


def embedding_table_group_attrs(index_key: str) -> dict[str, object]:
    """The ``attributes`` of an embedding table group (spec D4)."""
    return {
        "type": EMBEDDING_TABLE_TYPE,
        "backend": "parquet",
        "index_key": index_key,
    }


def embedding_arrow_table(payload: EmbeddingPayload) -> pa.Table:
    """*payload*'s frame as an Arrow table under an explicit schema.

    Key columns are non-null ``int64``, in frame order; ``embedding`` is the
    last column, ``FixedSizeList<float16>[dim]``. The explicit schema, and no
    pandas metadata, is what makes identical payloads write identical bytes.

    Raises:
        ValueError: If the frame has no ``embedding`` column, lacks the
            descriptor's ``key`` column, has a non-integer key column, or a
            vector whose length is not ``dim``.
    """
    import numpy as np
    import pyarrow as pa  # type: ignore[import-untyped]

    frame = payload.frame
    if EMBEDDING_COLUMN not in frame.columns:
        raise ValueError(f"embedding payload {payload.key!r} has no {EMBEDDING_COLUMN!r} column")
    keys = [str(column) for column in frame.columns if column != EMBEDDING_COLUMN]
    if payload.descriptor["key"] not in keys:
        raise ValueError(
            f"embedding payload {payload.key!r} lacks its key column "
            f"{payload.descriptor['key']!r}"
        )
    dim = payload.dim
    cells = frame[EMBEDDING_COLUMN].to_list()
    vectors = (
        np.stack([np.asarray(cell, dtype=np.float16).reshape(-1) for cell in cells])
        if cells
        else np.zeros((0, dim), dtype=np.float16)
    )
    if vectors.shape != (len(frame), dim):
        raise ValueError(
            f"embedding payload {payload.key!r} has vectors of shape "
            f"{vectors.shape}; its descriptor says dim={dim}"
        )
    fields = []
    arrays = []
    for column in keys:
        values = frame[column].to_numpy()
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(
                f"embedding payload {payload.key!r}: key column {column!r} is "
                f"{values.dtype}, not an integer"
            )
        fields.append(pa.field(column, pa.int64(), nullable=False))
        arrays.append(pa.array(values.astype(np.int64), type=pa.int64()))
    fields.append(pa.field(EMBEDDING_COLUMN, _vector_type(dim), nullable=False))
    arrays.append(
        pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(vectors).reshape(-1), type=pa.float16()), dim
        )
    )
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_embedding_parquet(path: Path, table: pa.Table, dim: int) -> None:
    """Atomically write *table*; reread it and check the vector type before it lands."""
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    from ._atomic_io import PARQUET_WRITE_OPTIONS, atomic_write_with_writer

    def _write(temp_path: str) -> None:
        pq.write_table(table, temp_path, **PARQUET_WRITE_OPTIONS)
        reread = pq.read_table(temp_path)
        if (
            reread.column_names != table.column_names
            or reread.num_rows != table.num_rows
            or reread.schema.field(EMBEDDING_COLUMN).type != _vector_type(dim)
        ):
            raise RuntimeError(
                f"embedding table {path} failed schema validation: read back "
                f"{reread.schema}"
            )

    atomic_write_with_writer(path, _write, temp_suffix=".parquet.tmp")


def write_embedding_tables(
    part: Path,
    run_id: str,
    payloads: Mapping[str, EmbeddingPayload],
    *,
    objmap_sha256: str | None,
    array_sha256: Mapping[str, str] | None,
) -> dict[str, dict]:
    """Write *payloads* into run *run_id* of *part*; return its ``embeddings`` block (spec D3).

    Each payload becomes ``tables/<run_id>/embeddings/<model>/<region>/table.parquet``,
    every level a Zarr v3 group and the table group carrying
    :func:`embedding_table_group_attrs` (spec D4). Its entry is the payload's
    descriptor plus ``stale: false`` and ``files: [{path, sha256,
    media_type}]``; the file's sha256 lives in the root, so the completion
    marker covers it (D9).

    ``stale`` is then **recomputed** from the entry's ``binds`` (spec D7):
    the binds name the image the vectors were computed from, and
    *objmap_sha256* / *array_sha256* the store being written, so a payload
    computed from another objmap is written stale. When both are ``None`` --
    a write that writes no objmap and no array, the measure rewrite -- the
    entries stay ``stale: false``, bound to the image the caller embedded.

    Args:
        part: An **unpromoted** ``*.ome.zarr.part`` directory.
        run_id: The run folder. Its folder must not hold these tables yet.
        payloads: ``{"<model>/<region>": EmbeddingPayload}``; each mapping key
            must equal its payload's :attr:`~EmbeddingPayload.key`.
        objmap_sha256: :func:`~phenotypic.sdk_._measurement_tables.objmap_digest`
            of the objmap this write certifies, or ``None``.
        array_sha256: :func:`~phenotypic.sdk_._measurement_tables.array_digest`
            of each level-0 series written, keyed by ``input.series``, or
            ``None``.

    Returns:
        ``{key: entry}``, in *payloads* order.

    Raises:
        ValueError: If a mapping key and its payload's key differ, or a
            payload's frame does not fit its descriptor.
    """
    from . import ngff_
    from ._measurement_tables import (
        _ensure_group,
        _group_document,
        refresh_embedding_staleness,
    )
    from ._table_runs import EMBEDDINGS_KEY, embedding_table_relative_path

    part = Path(part)
    block: dict[str, dict] = {}
    for key, payload in payloads.items():
        if key != payload.key:
            raise ValueError(
                f"embedding payload {payload.key!r} given under the key {key!r}"
            )
        relative = embedding_table_relative_path(
            run_id, payload.model, payload.region  # type: ignore[arg-type]
        )
        path = part / relative
        table = embedding_arrow_table(payload)
        _ensure_group(part / ngff_.TABLES_GROUP)
        _ensure_group(part / ngff_.TABLES_GROUP / run_id)
        _ensure_group(path.parent.parent.parent)  # embeddings/
        _ensure_group(path.parent.parent)  # embeddings/<model>/
        _ensure_group(
            path.parent,
            _group_document(embedding_table_group_attrs(str(payload.descriptor["key"]))),
        )
        _write_embedding_parquet(path, table, payload.dim)
        entry = deepcopy(dict(payload.descriptor))
        entry["stale"] = False
        entry["files"] = [
            {
                "path": relative.as_posix(),
                "sha256": _file_sha256(path),
                "media_type": PARQUET_MEDIA_TYPE,
            }
        ]
        block[key] = entry
    if objmap_sha256 is not None or array_sha256 is not None:
        refresh_embedding_staleness(
            {run_id: {EMBEDDINGS_KEY: block}},
            objmap_sha256=objmap_sha256,
            array_sha256=array_sha256 or {},
        )
    return block


def read_embedding_table(
    store: Path, key: str, *, run_id: str | None = None
) -> pa.Table:
    """Read embedding table *key* of *store*, verified against its entry.

    Which run (spec D6a, for a reader without a record): *run_id* when given,
    otherwise the highest-numbered run whose ``embeddings`` declares *key*.

    Args:
        store: A promoted ``*.ome.zarr`` store.
        key: ``"<model>/<region>"``.
        run_id: Read this run folder's entry instead of the latest one.

    Returns:
        The table, whose ``embedding`` column is ``FixedSizeList<float16>[dim]``.

    Raises:
        KeyError: If no run (or not *run_id*) declares *key*.
        EmbeddingTableUnreadableError: If the entry is stale, its file is
            missing, its bytes do not match the recorded sha256, or the file
            is not the table the entry describes. Embeddings are additive, so
            this never touches the run's measurements (spec D9).
    """
    from ._table_runs import EMBEDDINGS_KEY, list_table_runs

    store = Path(store)
    runs = [
        run
        for run in list_table_runs(store)
        if (run_id is None or run.run_id == run_id)
        and isinstance(run.descriptor.get(EMBEDDINGS_KEY), Mapping)
        and key in run.descriptor[EMBEDDINGS_KEY]  # type: ignore[operator]
    ]
    if not runs:
        where = f"run {run_id!r} of {store}" if run_id is not None else str(store)
        raise KeyError(f"{where} declares no embedding table {key!r}")
    return read_embedding_entry(
        store, key, runs[-1].descriptor[EMBEDDINGS_KEY][key]  # type: ignore[index]
    )


def embedding_entry_is_stale(entry: Mapping[str, object]) -> bool:
    """Whether an embedding entry is stale: its ``stale`` is anything but ``False``.

    The one predicate (spec D7). A writer records ``stale: false`` or
    ``stale: true``, so an entry with no ``stale`` key, or a ``null``, was
    not written by this build's writer and describes nothing a reader can
    trust: it reads as stale, the same answer :func:`read_embedding_entry`
    and the deliverables aggregation both give.
    """
    return entry.get("stale") is not False


def _entry_file_path(store: Path, relative: str) -> Path | None:
    """*store* joined with *relative*, or ``None`` when it would leave the store.

    The check is **lexical**, on the unresolved paths, never through
    ``Path.resolve``: a store under a symlinked ``results/`` is still inside
    itself, and a file that is a symlink out of the store (DVC's
    ``cache.type=symlink``) is still that store's file. Only an absolute
    *relative* or one that climbs out with ``..`` is refused.
    """
    import os

    if os.path.isabs(relative):
        return None
    joined = os.path.normpath(os.path.join(os.path.abspath(store), relative))
    root = os.path.normpath(os.path.abspath(store))
    if os.path.commonpath([root, joined]) != root or joined == root:
        return None
    return Path(joined)


def read_embedding_entry(
    store: Path, key: str, entry: object
) -> pa.Table:
    """Read the table *entry* describes, verified, with one read of its bytes.

    :func:`read_embedding_table` once it has found the entry, and the
    deliverables aggregation, which holds each certified run's entry already.
    The file is read once, hashed, and parsed from those bytes, so the table
    returned is the one whose sha256 was checked.

    Args:
        store: A promoted ``*.ome.zarr`` store.
        key: The entry's ``"<model>/<region>"`` key, for messages.
        entry: The entry, from the store root's ``embeddings`` block.

    Returns:
        The table, whose ``embedding`` column is ``FixedSizeList<float16>[dim]``.

    Raises:
        EmbeddingTableUnreadableError: If the entry is not a mapping or is
            stale (:func:`embedding_entry_is_stale`), its file is missing or
            outside the store, its bytes do not match the recorded sha256,
            or the file is not the table the entry describes.
    """
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    store = Path(store)
    if not isinstance(entry, Mapping):
        raise EmbeddingTableUnreadableError(store, key, "its entry is not a mapping")
    if embedding_entry_is_stale(entry):
        raise EmbeddingTableUnreadableError(
            store, key,
            "the entry is stale: the store no longer holds the objmap or array "
            "its vectors were computed from",
        )
    files = entry.get("files")
    if (
        not isinstance(files, list)
        or not files
        or not isinstance(files[0], Mapping)
        or not isinstance(files[0].get("path"), str)
        or not isinstance(files[0].get("sha256"), str)
    ):
        raise EmbeddingTableUnreadableError(store, key, "the entry lists no file")
    relative = str(files[0]["path"])
    path = _entry_file_path(store, relative)
    if path is None:
        raise EmbeddingTableUnreadableError(
            store, key, f"its file {relative} lies outside the store"
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise EmbeddingTableUnreadableError(
            store, key, f"its file {relative} is missing ({exc.strerror or exc})"
        ) from exc
    if hashlib.sha256(data).hexdigest() != files[0]["sha256"]:
        raise EmbeddingTableUnreadableError(
            store, key, f"{relative} does not match its recorded sha256"
        )
    try:
        table = pq.read_table(pa.BufferReader(data))
    except (pa.ArrowException, OSError) as exc:
        raise EmbeddingTableUnreadableError(store, key, f"{relative}: {exc}") from exc
    dim = entry.get("dim")
    if (
        EMBEDDING_COLUMN not in table.column_names
        or not isinstance(dim, int)
        or table.schema.field(EMBEDDING_COLUMN).type != _vector_type(dim)
    ):
        raise EmbeddingTableUnreadableError(
            store, key, f"{relative} is not a FixedSizeList<float16>[{dim}] table"
        )
    return table
