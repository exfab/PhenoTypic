"""Shared utilities for analysis plugin data preparation.

Provides JSON serialisation, columnar conversion, atomic file writing,
partitioning, stratified sampling, and column selection helpers that
plugins use in their ``prepare_data`` methods.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import polars as pl

from phenotypic.schema import EXPERIMENT_METADATA
from phenotypic.sdk_ import metadata_category_prefixes

from .._cli_output_manager import _atomic_write

# Prefix priority for column selection in scatter data.  Derived from the
# centralized helper so the list tracks the metadata namespace automatically
# (today ``Metadata_``; after B2 flip: ``MetadataGenetic_`` etc.).
SCATTER_PREFIX_PRIORITY = (*metadata_category_prefixes(), "Grid_", "Shape_", "Intensity_", "Color_")

# Column used to identify datasets throughout analysis plugins.
DATASET_COL = str(EXPERIMENT_METADATA.DATASET)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def sanitize_for_json(value: Any) -> Any:
    """Convert NaN, Inf, and -Inf to ``None`` for JSON compatibility.

    Args:
        value: A scalar value.

    Returns:
        The original value, or ``None`` if it is not JSON-representable.
    """
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def to_columnar(
    df: pl.DataFrame,
    total_rows: int,
    sampled: bool,
) -> dict:
    """Convert a DataFrame to columnar JSON-serialisable format.

    Args:
        df: DataFrame to convert.
        total_rows: Original row count (before any sampling).
        sampled: Whether the data was down-sampled.

    Returns:
        Dict with ``columns``, ``data``, ``total_rows``, ``sampled``.
    """
    columns = df.columns
    data: Dict[str, list] = {}

    for col in columns:
        series = df[col]
        if series.dtype.is_float():
            data[col] = [sanitize_for_json(v) for v in series.to_list()]
        else:
            data[col] = series.to_list()

    return {
        "columns": columns,
        "data": data,
        "total_rows": total_rows,
        "sampled": sampled,
    }


def write_json_atomic(payload: dict, target_path: Path) -> None:
    """Write *payload* as JSON to *target_path* atomically.

    Uses a temporary file in the same directory followed by
    :func:`os.replace` to avoid partial reads.

    Args:
        payload: JSON-serialisable dict to write.
        target_path: Destination file path.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fd = tempfile.NamedTemporaryFile(
        mode="w",
        dir=target_path.parent,
        prefix=f".{target_path.stem}_",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    )
    try:
        json.dump(payload, fd, indent=2, ensure_ascii=False)
        fd.write("\n")
        fd.flush()
        os.fsync(fd.fileno())
        fd.close()
        os.replace(fd.name, target_path)
    except BaseException:
        fd.close()
        try:
            os.unlink(fd.name)
        except OSError:
            pass
        raise


def write_parquet_sidecar(merged_df: pl.DataFrame, target_path: Path) -> None:
    """Write the merged measurement DataFrame as a Parquet sidecar file.

    Writes with zstd compression using an atomic write pattern
    (tempfile + :func:`os.replace`).

    Args:
        merged_df: Merged measurement DataFrame (Polars).
        target_path: Destination file path for the Parquet sidecar.
    """
    _atomic_write(
        target_path,
        lambda p: merged_df.write_parquet(p, compression="zstd", compression_level=3),
    )


# ---------------------------------------------------------------------------
# Partitioning helper
# ---------------------------------------------------------------------------


def partition_by_dataset(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """Partition *df* by ``Metadata_Dataset``, returning a flat string-keyed dict.

    Falls back to a single ``"all"`` group when the column is absent.
    Handles the tuple-key format that ``partition_by(as_dict=True)``
    returns in Polars.

    Args:
        df: DataFrame to partition.

    Returns:
        Mapping from dataset name to its sub-DataFrame.
    """
    if DATASET_COL not in df.columns:
        return {"all": df}

    raw = df.partition_by(DATASET_COL, as_dict=True)
    return {
        str(k[0]) if isinstance(k, tuple) else str(k): v
        for k, v in raw.items()
    }


# ---------------------------------------------------------------------------
# Sampling and column selection
# ---------------------------------------------------------------------------


def stratified_sample(
    df: pl.DataFrame,
    max_rows: int,
    seed: int = 42,
) -> pl.DataFrame:
    """Stratified proportional sample by ``Metadata_Dataset``.

    When ``Metadata_Dataset`` is not present, falls back to a simple
    random sample.

    Args:
        df: DataFrame to sample from.
        max_rows: Target number of rows.
        seed: Random seed for reproducibility.

    Returns:
        Sampled DataFrame (or the original if it already fits).
    """
    if df.height <= max_rows:
        return df

    if DATASET_COL not in df.columns:
        return df.sample(n=max_rows, seed=seed)

    # Proportional allocation per dataset.
    groups = partition_by_dataset(df)
    total = df.height
    sampled_parts: List[pl.DataFrame] = []

    for ds_name in sorted(groups.keys()):
        group_df = groups[ds_name]
        proportion = group_df.height / total
        n_sample = max(1, int(round(proportion * max_rows)))
        n_sample = min(n_sample, group_df.height)
        sampled_parts.append(group_df.sample(n=n_sample, seed=seed))

    result = pl.concat(sampled_parts)

    # Trim to exact max_rows if rounding pushed us over.
    if result.height > max_rows:
        result = result.sample(n=max_rows, seed=seed)

    return result


def select_scatter_columns(all_columns: List[str], max_cols: int = 25) -> List[str]:
    """Select up to *max_cols* columns for scatter data, prioritised by prefix.

    Excludes columns starting with ``Texture`` (too numerous).
    Prioritises columns by prefix in this order: Metadata, Grid,
    Shape, Intensity, Color, then remaining non-Texture columns.

    Args:
        all_columns: Full list of column names.
        max_cols: Maximum number of columns to return.

    Returns:
        Ordered list of selected column names.
    """
    selected: List[str] = []
    used: set[str] = set()

    # Pass 1: pick columns by prefix priority.
    for prefix in SCATTER_PREFIX_PRIORITY:
        for col in all_columns:
            if col.startswith(prefix) and col not in used:
                selected.append(col)
                used.add(col)

    # Pass 2: add remaining non-Texture columns.
    for col in all_columns:
        if col not in used and not col.startswith("Texture"):
            selected.append(col)
            used.add(col)

    return selected[:max_cols]
