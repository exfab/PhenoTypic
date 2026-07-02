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

from phenotypic.schema import EXPERIMENT_METADATA

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

    if (
        include_dataset_column
        and path_to_dataset
        and str(EXPERIMENT_METADATA.DATASET) not in df.columns
    ):
        mapping = {_source_path_key(p): name for p, name in path_to_dataset.items()}
        df = df.with_columns(
            pl.col(SOURCE_PATH_COLUMN)
            .str.replace_all(r"\\", "/")
            .replace_strict(mapping, default=None)
            .alias(str(EXPERIMENT_METADATA.DATASET))
        )

    if not keep_filename and SOURCE_PATH_COLUMN in df.columns:
        df = df.drop(SOURCE_PATH_COLUMN)

    logger.info(
        "Aggregated %d rows from %d files.", df.height, len(parquet_files)
    )
    return df
