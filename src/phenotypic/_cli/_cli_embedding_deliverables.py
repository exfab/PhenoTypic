"""The run's embedding deliverables (#281 spec E5, D6a, D7, D9).

``finalize_run`` aggregates each image's embedding tables, as it aggregates
its measurement table, from **the run folder the image's record certifies**
(spec D6a), into one file per entry key::

    deliverables/embeddings/<model>/<crops|grid>.parquet
    deliverables/embeddings/manifest.json

Each file holds the master's intrinsic identity (``Metadata_Dataset``,
``Metadata_ImageName``) plus the table's own key columns and ``embedding``,
so a row joins to the master on ``(Metadata_Dataset, Metadata_ImageName,
Object_Label)``. These files are Parquet only: never joined to metadata,
never in the master, the mirror, a split or any TSV (spec E5).

**Streamed**, one ``pq.ParquetWriter`` per output file and one row group
per store, so memory holds one store's table at a time: over a 33k-store run
the vectors alone are about 5 GB of float16.

**Two failure rules, and why they differ** (spec D7, D9, E5):

- A **stale** entry -- its store no longer holds the objmap or array its
  vectors were computed from -- is skipped with a warning naming the store,
  and counted. It describes nothing the run certifies, so leaving it out is
  the correct output rather than a degraded one.
- An entry that is **not stale but cannot be read** (its file missing, or
  not the bytes its sha256 records) fails finalization, as a master that
  cannot be built does. It never revokes measurement authority: the image's
  record still verifies and ``resolve_run_state`` still reads it
  ``verified`` (D9), so the remedy is a same-pipeline ``--mode measure``,
  which rewrites the embeddings without detecting again.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa  # type: ignore[import-untyped]

    from phenotypic.sdk_ import CommitGuard, TableRun

logger = logging.getLogger(__name__)

#: ``deliverables/embeddings/manifest.json``'s own format version.
EMBEDDINGS_MANIFEST_SCHEMA_VERSION: Final[int] = 1

#: Entry fields that describe one store's file rather than the table's kind;
#: the manifest's ``descriptors`` leave them out (spec D3).
_PER_STORE_FIELDS: Final[frozenset[str]] = frozenset({"binds", "files", "stale"})


def _entries(run: "TableRun") -> Mapping[str, object]:
    """*run*'s ``embeddings`` block, or an empty mapping."""
    from phenotypic.sdk_._table_runs import EMBEDDINGS_KEY

    block = run.descriptor.get(EMBEDDINGS_KEY)
    return block if isinstance(block, Mapping) else {}


def _store_identity(output_dir: Path, store: Path) -> tuple[str, str, str]:
    """``(store relative to output_dir, dataset, image name)`` of a run's store.

    The image name is the store's stem, the name the master gives a row read
    from an embedded table (``add_metadata_image_name_from_filename``).

    Raises:
        ValueError: If *store* is not ``results/<ds>/zarr/<stem>.ome.zarr``
            under *output_dir*.
    """
    from phenotypic.sdk_ import store_stem, zarr_store_path

    resolved = Path(store).resolve()
    relative = resolved.relative_to(Path(output_dir).resolve())
    stem = store_stem(resolved)
    dataset = relative.parts[1] if len(relative.parts) == 4 else ""
    if not dataset or zarr_store_path(output_dir, dataset, stem).resolve() != resolved:
        raise ValueError(
            f"{store} is not a store of this run: expected "
            "results/<dataset>/zarr/<stem>.ome.zarr under the output"
        )
    return relative.as_posix(), dataset, stem


def _with_identity(table: "pa.Table", dataset: str, image: str) -> "pa.Table":
    """Prepend ``Metadata_Dataset`` and ``Metadata_ImageName`` to *table*."""
    import pyarrow as pa  # type: ignore[import-untyped]

    from phenotypic.schema import EXPERIMENT, IMAGE

    table = table.replace_schema_metadata(None)
    rows = table.num_rows
    for name, value in ((str(IMAGE.IMAGE_NAME), image), (str(EXPERIMENT.DATASET), dataset)):
        table = table.add_column(
            0,
            pa.field(name, pa.string(), nullable=False),
            pa.array([value] * rows, type=pa.string()),
        )
    return table


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unreadable(exc: Exception, key: str, relative: str) -> RuntimeError:
    return RuntimeError(
        f"{exc}. Measurement authority is unaffected (embeddings are additive), "
        f"but finalization cannot publish deliverables/embeddings/{key}.parquet "
        f"without {relative}. Re-measure the image with the same pipeline "
        "(--mode measure) to rewrite its embeddings, then finalize again."
    )


def _publish_key(
    output_dir: Path,
    key: str,
    runs: Sequence["TableRun"],
    identities: Mapping[Path, tuple[str, str, str]],
    *,
    commit_guard: "CommitGuard | None",
) -> tuple[dict[str, int], dict[str, object], Path | None]:
    """Stream entry *key* of every run into its deliverable.

    Returns:
        ``({"rows", "stale_skipped"}, manifest entry, published path or None)``.
    """
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    from phenotypic.sdk_ import embeddings_dir, read_embedding_table
    from phenotypic.sdk_._atomic_io import PARQUET_WRITE_OPTIONS, atomic_write_with_writer
    from phenotypic.sdk_._embedding_tables import EmbeddingTableUnreadableError

    model, _, region = key.partition("/")
    target = embeddings_dir(output_dir) / model / f"{region}.parquet"
    readable: list["TableRun"] = []
    stale: list[str] = []
    descriptors: dict[str, object] = {}
    for run in runs:
        entry = _entries(run).get(key)
        if not isinstance(entry, Mapping):
            continue
        relative = identities[run.store][0]
        if entry.get("stale") is True:
            logger.warning(
                "Skipping the stale embedding table %r of %s: the store no "
                "longer holds the objmap or array its vectors were computed "
                "from (spec D7). Re-measure the image to refresh it.",
                key,
                relative,
            )
            stale.append(relative)
            continue
        readable.append(run)
        shared = {name: value for name, value in entry.items() if name not in _PER_STORE_FIELDS}
        descriptors[_canonical(shared)] = shared

    rows = 0

    def _write(temp_path: str) -> None:
        nonlocal rows
        writer: pq.ParquetWriter | None = None
        try:
            for run in readable:
                relative, dataset, image = identities[run.store]
                try:
                    table = read_embedding_table(run.store, key, run_id=run.run_id)
                except EmbeddingTableUnreadableError as exc:
                    raise _unreadable(exc, key, relative) from exc
                table = _with_identity(table, dataset, image)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_path,
                        table.schema,
                        compression=PARQUET_WRITE_OPTIONS["compression"],
                        compression_level=PARQUET_WRITE_OPTIONS["compression_level"],
                    )
                elif not table.schema.equals(writer.schema, check_metadata=False):
                    raise RuntimeError(
                        f"embedding table {key!r} of {relative} has schema "
                        f"{table.schema}, unlike the other stores' {writer.schema}; "
                        "one deliverable cannot hold both"
                    )
                # One row group per store: memory is one store's table.
                writer.write_table(table, row_group_size=max(table.num_rows, 1))
                rows += table.num_rows
        finally:
            if writer is not None:
                writer.close()

    published: Path | None = None
    if readable:
        atomic_write_with_writer(
            target, _write, commit_guard=commit_guard, temp_suffix=".parquet.tmp"
        )
        published = target
    manifest_entry: dict[str, object] = {
        "path": (
            published.relative_to(embeddings_dir(output_dir).parent).as_posix()
            if published is not None
            else None
        ),
        "rows": rows,
        "sha256": _file_sha256(published) if published is not None else None,
        "stores": len(readable),
        "stale_skipped": sorted(stale),
        "descriptors": [descriptors[name] for name in sorted(descriptors)],
    }
    return {"rows": rows, "stale_skipped": len(stale)}, manifest_entry, published


def _remove_unpublished(
    output_dir: Path, published: set[Path], *, commit_guard: "CommitGuard | None"
) -> None:
    """Delete a previous finalization's deliverables this one did not publish."""
    from phenotypic.sdk_ import embeddings_dir
    from phenotypic.sdk_._atomic_io import publication_commit

    root = embeddings_dir(output_dir)
    if not root.is_dir():
        return
    with publication_commit(commit_guard):
        if not published:
            shutil.rmtree(root)
            return
        for path in sorted(root.rglob("*.parquet")):
            if path not in published:
                path.unlink()
        for directory in sorted(
            (p for p in root.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)
        ):
            if not any(directory.iterdir()):
                directory.rmdir()


def aggregate_embedding_tables(
    output_dir: Path,
    runs: Sequence["TableRun"],
    *,
    commit_guard: "CommitGuard | None" = None,
) -> dict[str, dict[str, int]]:
    """Publish ``deliverables/embeddings/`` from each image's certified run (spec E5).

    For each entry key any run declares, the rows of every run's table are
    streamed, store by store in store-path order, into
    ``deliverables/embeddings/<model>/<crops|grid>.parquet`` with
    ``Metadata_Dataset`` and ``Metadata_ImageName`` prepended. Then
    ``manifest.json`` records, per key, the file's path, row count and
    sha256, how many stores it holds, the stores skipped as stale, and the
    distinct entry descriptors (each store's ``binds``, ``files`` and
    ``stale`` left out). No wall-clock value is written, so identical runs
    give identical bytes (spec C11).

    A previous finalization's files that this one does not publish are
    deleted, and with no entry at all the folder goes, so the deliverables
    always describe the folders this finalization certifies (D6a).

    Args:
        output_dir: Run output root.
        runs: Each image's certified run folder (spec D6a): the run its record
            certifies, **never** one chosen by pipeline sha. A run without
            embeddings contributes nothing.
        commit_guard: Publication guard threaded to every write and delete.

    Returns:
        ``{key: {"rows": n, "stale_skipped": k}}``, sorted by key: what
        :func:`~phenotypic.sdk_._run_metrics.build_run_metrics` takes as
        ``embeddings``. Empty when no run has embeddings.

    Raises:
        RuntimeError: If a non-stale entry cannot be read (its message names
            the store and the ``--mode measure`` remedy), or two stores'
            tables for one key have different schemas.
        ValueError: If a run's store is not a store of this run.
    """
    from phenotypic.sdk_ import embeddings_manifest_path
    from phenotypic.sdk_._run_metrics import write_canonical_json

    output_dir = Path(output_dir)
    with_embeddings = [run for run in runs if _entries(run)]
    identities = {run.store: _store_identity(output_dir, run.store) for run in with_embeddings}
    ordered = sorted(with_embeddings, key=lambda run: identities[run.store][0])
    keys = sorted({str(key) for run in ordered for key in _entries(run)})

    counts: dict[str, dict[str, int]] = {}
    manifest: dict[str, object] = {}
    published: set[Path] = set()
    for key in keys:
        counts[key], manifest[key], path = _publish_key(
            output_dir, key, ordered, identities, commit_guard=commit_guard
        )
        if path is not None:
            published.add(path)
    if keys:
        write_canonical_json(
            embeddings_manifest_path(output_dir),
            {"schema_version": EMBEDDINGS_MANIFEST_SCHEMA_VERSION, "entries": manifest},
            commit_guard=commit_guard,
        )
        published.add(embeddings_manifest_path(output_dir))
        _remove_unpublished(output_dir, published, commit_guard=commit_guard)
    else:
        _remove_unpublished(output_dir, set(), commit_guard=commit_guard)
    return counts


def embedding_table_schemas(output_dir: Path) -> dict[str, dict[str, object]]:
    """Describe each published embedding deliverable for ``schema.json`` (spec C12).

    Read back from what :func:`aggregate_embedding_tables` just published:
    the manifest names the files and their producer, and each file's footer
    gives its columns. Key and identity columns are rendered as polars
    dtypes, as every other ``schema.json`` table is.

    Returns:
        ``{"embeddings/<model>/<region>": {"columns", "dim", "model",
        "producer"}}``, the shape
        :func:`~phenotypic.util._measurement_outputs.build_output_schema`
        takes as ``embedding_tables``. Empty when nothing is published.
    """
    import polars as pl
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    from phenotypic.sdk_ import deliverables_dir, embeddings_manifest_path
    from phenotypic.sdk_._embedding_tables import EMBEDDING_COLUMN

    manifest_path = embeddings_manifest_path(output_dir)
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tables: dict[str, dict[str, object]] = {}
    for key, entry in sorted(manifest.get("entries", {}).items()):
        if entry.get("path") is None or not entry.get("descriptors"):
            continue
        path = deliverables_dir(output_dir) / entry["path"]
        schema = pq.read_schema(path)
        frame = pl.from_arrow(schema.empty_table().drop_columns([EMBEDDING_COLUMN]))
        assert isinstance(frame, pl.DataFrame)
        keys = frame.schema
        columns: dict[str, object] = {}
        for name in schema.names:
            columns[name] = EMBEDDING_COLUMN if name == EMBEDDING_COLUMN else keys[name]
        descriptor = entry["descriptors"][0]
        tables[f"embeddings/{key}"] = {
            "columns": columns,
            "dim": schema.field(EMBEDDING_COLUMN).type.list_size,
            "model": key.partition("/")[0],
            "producer": descriptor["producer"]["class"],
        }
    return tables


__all__ = [
    "EMBEDDINGS_MANIFEST_SCHEMA_VERSION",
    "aggregate_embedding_tables",
    "embedding_table_schemas",
]
