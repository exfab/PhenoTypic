"""Polars-native aggregation for per-image Parquet measurement files.

Reads and concatenates per-image measurement Parquet files into a single
Polars DataFrame. This replaces the former DuckDB-based aggregator: a single
multithreaded ``pl.read_parquet`` over all files is ~6-7x faster and ~4x
lighter on peak memory than reading via DuckDB and converting through Arrow
(measured on a 7.9k-file / 356k-row run), and keeps the whole compilation hot
path on one engine. That single-engine path matters on the cluster, where the
``polars-lts-cpu`` build (shipped by default for pre-AVX2 nodes) must cover
every step.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

from phenotypic.schema import EXPERIMENT

from ._metadata_join import normalize_measurement_metadata_columns

logger = logging.getLogger(__name__)

# Virtual source-path column. Named ``filename`` so callers that derive
# ``Metadata_ImageName`` from it (and drop it afterwards) are unchanged from
# the previous DuckDB reader, which exposed the same column.
SOURCE_PATH_COLUMN = "filename"


def _source_path_key(path: object) -> str:
    """Normalize source-path spellings for file-to-dataset lookups."""
    return str(path).replace("\\", "/")


def aggregate_parquet_files(
    file_paths: list[Path],
    path_to_dataset: dict[Path, str],
    include_dataset_column: bool = True,
    keep_filename: bool = False,
) -> "pl.DataFrame | None":
    """Read and concatenate Parquet measurement files into one Polars frame.

    A single multithreaded :func:`polars.read_parquet` reads every file and
    records each row's source path in a ``filename`` column (mirroring the
    virtual column the previous DuckDB reader exposed). Files with
    heterogeneous schemas fall back to a per-file ``diagonal_relaxed`` concat,
    preserving the schema-tolerant ``UNION ALL BY NAME`` behaviour of the old
    reader.

    Args:
        file_paths: Measurement file paths (``.parquet``).
        path_to_dataset: Maps each file path to its dataset name string.
        include_dataset_column: Whether to add a ``Metadata_Dataset`` column
            derived from the file-to-dataset mapping. Skipped when the data
            already carries that column.
        keep_filename: If ``True``, retain the ``filename`` source-path column
            in the output. Useful when callers need to derive per-file
            metadata (e.g. ``Metadata_ImageName``).

    Returns:
        A single Polars DataFrame with all measurements concatenated, or
        ``None`` if no files could be read.
    """
    import polars as pl

    if not file_paths:
        logger.warning("No measurement files provided to aggregate.")
        return None

    parquet_files: list[Path] = []
    for p in file_paths:
        if p.suffix.lower() == ".parquet":
            parquet_files.append(p)
        else:
            logger.warning("Skipping unsupported file type: %s", p)
    if not parquet_files:
        logger.warning("No .parquet files found in the input.")
        return None

    paths_str = [str(p) for p in parquet_files]
    try:
        # Fast path: one multithreaded read, source path recorded per row.
        # rechunk() consolidates the per-file chunks into one contiguous block
        # so downstream writes/compression are not penalised for fragmentation.
        df = pl.read_parquet(
            paths_str, include_file_paths=SOURCE_PATH_COLUMN
        ).rechunk()
    except Exception:
        # Schema-heterogeneous fallback: union every column across files.
        logger.debug(
            "Uniform read failed; falling back to diagonal_relaxed concat.",
            exc_info=True,
        )
        frames: list[pl.DataFrame] = []
        for p in parquet_files:
            try:
                frames.append(
                    pl.read_parquet(str(p)).with_columns(
                        pl.lit(str(p)).alias(SOURCE_PATH_COLUMN)
                    )
                )
            except Exception:
                logger.warning("Failed to read %s", p, exc_info=True)
        if not frames:
            return None
        df = pl.concat(frames, how="diagonal_relaxed").rechunk()

    return _finish_aggregate(
        df,
        path_to_dataset,
        include_dataset_column=include_dataset_column,
        keep_filename=keep_filename,
        n_files=len(parquet_files),
    )


def _finish_aggregate(
    df: "pl.DataFrame",
    path_to_dataset: dict[Path, str],
    *,
    include_dataset_column: bool,
    keep_filename: bool,
    n_files: int,
) -> "pl.DataFrame":
    """Normalize metadata headers, insert ``Metadata_Dataset``, drop ``filename``.

    Shared by both aggregators, so an embedded-table master and a legacy one
    get their dataset column and source-path handling from one place.
    """
    import polars as pl

    df = normalize_measurement_metadata_columns(df)

    if (
        include_dataset_column
        and path_to_dataset
        and str(EXPERIMENT.DATASET) not in df.columns
    ):
        mapping = {
            _source_path_key(p): name for p, name in path_to_dataset.items()
        }
        df = df.with_columns(
            pl.col(SOURCE_PATH_COLUMN)
            .str.replace_all(r"\\", "/")
            .replace_strict(mapping, default=None)
            .alias(str(EXPERIMENT.DATASET))
        )

    if not keep_filename and SOURCE_PATH_COLUMN in df.columns:
        df = df.drop(SOURCE_PATH_COLUMN)

    logger.info("Aggregated %d rows from %d files.", df.height, n_files)
    return df


# ---------------------------------------------------------------------------
# Embedded tables: read each one as its own store declares it (P7 Task 4)
# ---------------------------------------------------------------------------


def is_embedded_measurement_table(path: Path | str) -> bool:
    """Return whether *path* names a store's embedded measurement table.

    Authority is a property of the path shape --
    ``<store>/tables/measurements/table.parquet`` -- so this opens nothing.
    """
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH

    suffix = MEASUREMENT_TABLE_RELATIVE_PATH.parts
    return tuple(Path(path).parts[-len(suffix) :]) == suffix


def project_embedded_measurement_table(
    table_path: Path, *, read_path: Path | None = None
) -> "pl.DataFrame | None":
    """Read one store's embedded table, projected onto its own descriptor.

    **Why this exists.** A pre-inversion store's ``table.parquet`` is the
    per-image metadata JOIN: the descriptor's ``measurement_columns`` plus
    every user metadata column, with each measured row repeated once per
    matching metadata row. ``--mode migrate`` deliberately leaves those tables
    alone (D-A). Aggregated as-is they make the master v1-shaped, and the
    global join in ``finalize_post_master_outputs`` then joins a second time
    on the user columns -- which, on a real 6,657-image migration, dropped
    every measured row from the mirror. So the normalization happens here, at
    read, and applies to every store: for a post-inversion table it is the
    identity.

    Three steps, in order:

    1. **Skip a store that declares no table** (Step 0b).
       :func:`~phenotypic.sdk_._measurement_tables.read_embedded_measurement_descriptor`
       documents an absent descriptor as a normal state, so it excludes the
       store with an advisory rather than failing the whole finalization.
    2. **Project** onto ``measurement_columns``, in the descriptor's order.
    3. **Collapse join fan-out** (CAN-10(a)), only when the table's Parquet
       records ``phenotypic.join.status == "joined"``. Rows sharing the
       descriptor's ``target.column`` collapse only when they are identical
       across every projected column: a label identifies one object within
       one image, so identical rows under one label are copies of that
       object. Rows that share a label and disagree are distinct objects, and
       collapsing them would silently drop one -- that store is excluded with
       an advisory instead.

    An excluded store is absent from the master, and callers leave it out of
    the source set the aggregate proof certifies, so the run degrades toward
    ``incomplete`` rather than certifying an image it does not carry.

    Args:
        table_path: The store's ``tables/measurements/table.parquet``. The
            store root, and so its descriptor, is derived from it.
        read_path: Where to read the Parquet bytes from when a staged copy
            stands in for *table_path*. The descriptor is always read from
            the store itself.

    Returns:
        The projected frame, or ``None`` when the store is excluded -- in
        which case a warning naming the store has been logged.

    Raises:
        ValueError: The store's schema version is not this build's.
        polars.exceptions.ColumnNotFoundError: The table lacks a column its
            own descriptor declares -- a contract violation, not a normal
            state, and not something to aggregate around.
    """
    import polars as pl
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH
    from phenotypic.sdk_._measurement_tables import (
        read_embedded_measurement_descriptor,
    )
    from phenotypic.sdk_.ngff_ import EMBEDDED_MEASUREMENT_PARQUET_METADATA_KEYS

    table_path = Path(table_path)
    store = table_path.parents[len(MEASUREMENT_TABLE_RELATIVE_PATH.parts) - 1]
    payload = table_path if read_path is None else Path(read_path)

    try:
        descriptor = read_embedded_measurement_descriptor(store)
    except KeyError:
        logger.warning(
            "Excluding %s from the master: the store declares no embedded "
            "measurement table, so there is no column list to project its "
            "table onto. The run reads as incomplete until the store is "
            "re-measured.",
            store,
        )
        return None
    # The same list `embedded_measurement_columns` returns, read from the
    # descriptor already in hand rather than by parsing the root a second time
    # per store.
    columns = descriptor.get("measurement_columns")
    if not isinstance(columns, list) or not all(
        isinstance(column, str) for column in columns
    ):
        logger.warning(
            "Excluding %s from the master: its measurement-table descriptor "
            "carries no column list. The run reads as incomplete until the "
            "store is re-measured.",
            store,
        )
        return None

    frame = pl.read_parquet(payload, columns=columns)

    target = descriptor.get("target")
    label = target.get("column") if isinstance(target, dict) else None
    if not isinstance(label, str) or label not in frame.columns:
        label = None
    repeated = (
        frame.get_column(label).is_duplicated().any()
        if label is not None
        else frame.is_duplicated().any()
    )
    if not repeated:
        return frame

    keys = EMBEDDED_MEASUREMENT_PARQUET_METADATA_KEYS
    metadata = pq.read_schema(payload).metadata or {}
    if metadata.get(keys.JOIN_STATUS.encode()) != b"joined":
        # Repetition the join did not create is not this rule's business.
        return frame
    if label is None:
        logger.warning(
            "Excluding %s from the master: its metadata-joined table repeats "
            "rows but declares no target column, so they cannot be proved "
            "copies of one object. The run reads as incomplete until the "
            "store is re-measured.",
            store,
        )
        return None

    collapsed = frame.unique(maintain_order=True)
    if collapsed.get_column(label).is_duplicated().any():
        logger.warning(
            "Excluding %s from the master: its metadata-joined table holds "
            "rows that share %s but differ in their measurements. They are "
            "distinct objects rather than join fan-out, and collapsing them "
            "would drop one. The run reads as incomplete until the store is "
            "re-measured.",
            store,
            label,
        )
        return None
    logger.info(
        "Collapsed %d metadata-join fan-out rows to %d objects in %s",
        frame.height,
        collapsed.height,
        store,
    )
    return collapsed


def _warn_on_dtype_disagreement(frames: list["pl.DataFrame"]) -> None:
    """Advise when the projected tables disagree on a column's dtype.

    CAN-10(b), ruled R4 (2026-09-10). ``_restore_join_key_dtypes`` can leave a
    legacy store's join key as ``String`` where other stores carry ``Int64``,
    and ``diagonal_relaxed`` then widens the whole column to ``String``. The
    plan required that never happen silently; the probe showed the widening is
    harmless to the join, because ``join_metadata`` casts every key to
    ``String`` on both sides and the drifted values render identically. So it
    is advised rather than repaired.

    Called at **both** concats: inside each shard (and on the direct path),
    and over the shard frames where ``build_master_frame`` merges them --
    drift between stores that landed in different shards is visible only at
    the merge.

    **Logging changes no data, and that is all it guarantees.** It does not
    make the master independent of the shard count. ``diagonal_relaxed``
    widens pairwise, and pairwise widening is not associative in how it
    renders values: when a column's dtypes include ``Int64``, ``Float64`` and
    ``String`` and they split across a shard boundary, an integer reaches
    ``String`` as ``"7"`` in one grouping and through ``Float64`` as ``"7.0"``
    in the other (Task 4 review, MF-3). That K-dependence is inherited from
    ``diagonal_relaxed`` and predates this projection -- which, by removing
    the user columns, narrows it -- and it is recorded as follow-up FU-8.
    The aggregate proof is unaffected: it certifies a source set, not bytes.
    """
    dtypes: dict[str, set[str]] = {}
    for frame in frames:
        for name, dtype in frame.schema.items():
            dtypes.setdefault(name, set()).add(str(dtype))
    for name, seen in dtypes.items():
        if name != SOURCE_PATH_COLUMN and len(seen) > 1:
            logger.warning(
                "Column %s has dtypes %s across the measurement frames being "
                "concatenated; concatenation widens it to their common "
                "supertype.",
                name,
                sorted(seen),
            )


def aggregate_embedded_measurement_tables(
    sources: dict[Path, str],
    *,
    include_dataset_column: bool = True,
    read_paths: dict[Path, Path] | None = None,
) -> tuple["pl.DataFrame | None", dict[Path, str]]:
    """Concatenate embedded tables, each projected by :func:`project_embedded_measurement_table`.

    The ``filename`` column records each table's **store** path, even when
    the bytes came from a staged copy, so identity recovery and the
    ``Metadata_Dataset`` lookup see the path the source set names.

    Args:
        sources: Embedded table path -> dataset, in aggregation order.
        include_dataset_column: Whether to insert ``Metadata_Dataset`` when
            the tables lack it.
        read_paths: Staged copies to read in place of each table.

    Returns:
        ``(frame, aggregated)``. ``aggregated`` is *sources* minus every store
        the projection excluded -- the set the master was built from, which
        is what an aggregate proof may certify. ``frame`` is ``None`` when
        nothing was aggregated; it keeps its ``filename`` column.
    """
    import polars as pl

    frames: list[pl.DataFrame] = []
    aggregated: dict[Path, str] = {}
    for table_path, dataset in sources.items():
        frame = project_embedded_measurement_table(
            table_path,
            read_path=None if read_paths is None else read_paths.get(table_path),
        )
        if frame is None:
            continue
        frames.append(
            frame.with_columns(pl.lit(str(table_path)).alias(SOURCE_PATH_COLUMN))
        )
        aggregated[table_path] = dataset
    if not frames:
        logger.warning(
            "No embedded measurement table could be aggregated from %d "
            "source(s).",
            len(sources),
        )
        return None, aggregated

    _warn_on_dtype_disagreement(frames)
    df = pl.concat(frames, how="diagonal_relaxed").rechunk()
    return (
        _finish_aggregate(
            df,
            aggregated,
            include_dataset_column=include_dataset_column,
            keep_filename=True,
            n_files=len(frames),
        ),
        aggregated,
    )


def aggregate_measurement_sources(
    sources: dict[Path, str],
    *,
    include_dataset_column: bool = True,
) -> tuple["pl.DataFrame | None", dict[Path, str]]:
    """Aggregate one set of measurement sources for a shard.

    Embedded tables go through :func:`aggregate_embedded_measurement_tables`,
    anything else through :func:`aggregate_parquet_files` unchanged. The
    choice is all-or-nothing because a mixed authority set is refused
    upstream (``refuse_mixed_measurement_authority``).

    Returns:
        ``(frame, aggregated)`` -- the frame with its ``filename`` column, and
        the sources it was built from.
    """
    if sources and all(is_embedded_measurement_table(path) for path in sources):
        return aggregate_embedded_measurement_tables(
            sources, include_dataset_column=include_dataset_column
        )
    frame = aggregate_parquet_files(
        file_paths=list(sources),
        path_to_dataset=sources,
        include_dataset_column=include_dataset_column,
        keep_filename=True,
    )
    return frame, dict(sources)
