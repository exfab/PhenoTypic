"""DuckDB-based aggregation for Parquet measurement files.

Uses DuckDB's in-memory SQL engine to consolidate per-image measurement
files into a single Polars DataFrame, respecting SLURM resource limits
when running on the cluster.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from phenotypic.tools_ import EnvVar

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)

# Accepts bare integers or integers with a DuckDB-compatible unit suffix.
_MEM_LIMIT_RE = re.compile(r"^\d+(?:[KMGT]B?)?$", re.IGNORECASE)


def duckdb_aggregate(
    file_paths: list[Path],
    path_to_dataset: dict[Path, str],
    include_dataset_column: bool = True,
    keep_filename: bool = False,
) -> "pl.DataFrame | None":
    """Read and consolidate Parquet measurement files via DuckDB.

    Parquet files are read natively by DuckDB with parallel I/O and
    combined with ``UNION ALL BY NAME`` for schema-tolerant concatenation.

    Args:
        file_paths: Measurement file paths (``.parquet``).
        path_to_dataset: Maps each file path to its dataset name string.
        include_dataset_column: Whether to add a ``Metadata_Dataset``
            column derived from the file-to-dataset mapping.
        keep_filename: If ``True``, retain the DuckDB ``filename``
            virtual column in the output.  Useful when callers need
            to derive per-file metadata (e.g. ``Metadata_ImageFile``).

    Returns:
        A single Polars DataFrame with all measurements concatenated,
        or ``None`` if no files could be read.
    """
    import duckdb
    import polars as pl

    if not file_paths:
        logger.warning("No measurement files provided to aggregate.")
        return None

    parquet_files: list[str] = []
    for p in file_paths:
        if p.suffix.lower() == ".parquet":
            parquet_files.append(str(p))
        else:
            logger.warning("Skipping unsupported file type: %s", p)

    if not parquet_files:
        logger.warning("No .parquet files found in the input.")
        return None

    conn = duckdb.connect()
    try:
        _configure_connection(conn)

        # Escape single quotes in paths for SQL safety (paths come from
        # filesystem globs, not user input, but may contain apostrophes
        # in directory names like "O'Brien_lab").
        def _sql_str(s: str) -> str:
            return s.replace("'", "''")

        pq_list = ", ".join(f"'{_sql_str(f)}'" for f in parquet_files)
        base_query = (
            f"SELECT * FROM read_parquet([{pq_list}], "
            f"union_by_name=true, filename=true)"
        )

        # Build dataset mapping via a temp table (safe from injection).
        if include_dataset_column and path_to_dataset:
            mapping = [
                (str(path), name) for path, name in path_to_dataset.items()
            ]
            conn.execute(
                "CREATE TEMP TABLE _ds_map(path VARCHAR, dataset VARCHAR)"
            )
            conn.executemany(
                "INSERT INTO _ds_map VALUES (?, ?)", mapping
            )

            # Check the first file's schema to see if column exists.
            has_col = conn.execute(
                "SELECT COUNT(*) FROM parquet_schema($f) "
                "WHERE name = 'Metadata_Dataset'",
                {"f": parquet_files[0]},
            ).fetchone()[0] > 0

            if has_col:
                logger.debug(
                    "Metadata_Dataset column already present; skipping."
                )
                query = base_query
            else:
                query = (
                    "SELECT t.*, m.dataset AS \"Metadata_Dataset\" "
                    f"FROM ({base_query}) AS t "
                    "LEFT JOIN _ds_map m ON t.filename = m.path"
                )
        else:
            query = base_query

        arrow_table = conn.execute(query).arrow()
        result = pl.from_arrow(arrow_table)

        if not keep_filename and "filename" in result.columns:
            result = result.drop("filename")

        logger.info(
            "Aggregated %d rows from %d files.",
            result.height,
            len(parquet_files),
        )
        return result

    except Exception:
        logger.exception("DuckDB aggregation failed.")
        return None
    finally:
        conn.close()


def _configure_connection(conn: object) -> None:
    """Apply SLURM-aware resource limits to a DuckDB connection."""
    mem_limit = os.environ.get(EnvVar.SLURM_MEM_PER_NODE, "")
    if mem_limit.isdigit():
        # SLURM_MEM_PER_NODE is in megabytes when set by Slurm.
        mem_limit = f"{mem_limit}MB"
    if not (mem_limit and _MEM_LIMIT_RE.match(mem_limit)):
        mem_limit = "4GB"
    conn.execute(f"SET memory_limit = '{mem_limit}'")

    temp_dir = os.environ.get(EnvVar.SCRATCH, "/tmp").replace("'", "''")
    conn.execute(f"SET temp_directory = '{temp_dir}'")

    try:
        threads = int(os.environ.get(EnvVar.SLURM_CPUS_PER_TASK, "4"))
    except ValueError:
        threads = 4
    conn.execute(f"SET threads = {threads}")

    conn.execute("SET preserve_insertion_order = false")
