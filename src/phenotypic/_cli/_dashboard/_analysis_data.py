"""Sidecar data preparation for the Analysis tab.

Scans per-image measurement Parquet files, optionally left-joins metadata,
and writes JSON sidecar files that the dashboard's Analysis tab
reads via ``fetch()``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from .._cli_output_manager import _atomic_write
from .._cli_utils import scan_parquets

logger = logging.getLogger(__name__)

# Prefix priority for column selection in scatter data.
_SCATTER_PREFIX_PRIORITY = ("Metadata_", "Grid_", "Shape_", "Intensity_", "Color_")

# Column used to identify datasets throughout this module.
_DATASET_COL = "Metadata_Dataset"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_analysis_sidecar(
    output_dir: Path,
    metadata_csv: Optional[Path] = None,
) -> None:
    """Write JSON sidecar files for the dashboard Analysis tab.

    Discovers datasets under ``results/``, loads and merges measurement
    Parquet files, and writes four JSON files into ``progress/``:

    * ``analysis_scatter.json`` -- columnar data for scatter plots
    * ``analysis_table.json`` -- columnar data for the raw-data table
    * ``analysis_stats.json`` -- per-dataset descriptive statistics
    * ``overlay_manifest.json`` -- dataset-to-overlay-image mapping

    Also writes ``progress/analysis_full.parquet`` as a local-mode
    sidecar (single-file approach, no chunking).

    All writes are atomic (tempfile + :func:`os.replace`).

    Args:
        output_dir: Root output directory (contains ``results/`` and
            ``progress/``).
        metadata_csv: Optional path to an external metadata CSV for
            left-joining onto measurements.
    """
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)

    merged_df = _load_and_merge(output_dir, metadata_csv)

    # Overlay manifest is independent of measurement data.
    overlay_manifest = _prepare_overlay_manifest(output_dir)
    _write_json_atomic(overlay_manifest, progress_dir / "overlay_manifest.json")

    if merged_df is None:
        logger.info("No measurement data found; skipping analysis sidecar files")
        return

    scatter_data = _prepare_scatter_data(merged_df)
    table_data = _prepare_table_data(merged_df)
    summary_stats = _prepare_summary_stats(merged_df)

    _write_json_atomic(scatter_data, progress_dir / "analysis_scatter.json")
    _write_json_atomic(table_data, progress_dir / "analysis_table.json")
    _write_json_atomic(summary_stats, progress_dir / "analysis_stats.json")

    # Write local-mode Parquet sidecar (single-file, no chunking).
    _write_parquet_sidecar(merged_df, progress_dir / "analysis_full.parquet")

    logger.debug(
        "Wrote analysis sidecar files (%d rows, %d columns)",
        merged_df.height,
        merged_df.width,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_and_merge(
    output_dir: Path,
    metadata_csv: Optional[Path],
) -> Optional[pl.DataFrame]:
    """Scan measurement Parquet files, concatenate, and optionally join metadata.

    Walks ``results/*/measurements/*.parquet``, adds ``Metadata_Dataset``
    and ``Metadata_ImageFile`` columns, and performs a left-join with
    *metadata_csv* when provided.

    Args:
        output_dir: Root output directory (contains ``results/``).
        metadata_csv: Optional external metadata CSV path.

    Returns:
        Merged DataFrame, or ``None`` if no measurements were found.
    """
    results_dir = output_dir / "results"
    if not results_dir.is_dir():
        return None

    # Discover dataset names from subdirectories.
    dataset_names = sorted(
        d.name for d in results_dir.iterdir() if d.is_dir()
    )
    if not dataset_names:
        return None

    all_measurements: List[pl.DataFrame] = []
    n_skipped = 0

    # Collect all Parquet paths, then stream-read in one batch.
    path_to_dataset: Dict[Path, str] = {}
    for dataset_name in dataset_names:
        dataset_meas_dir = results_dir / dataset_name / "measurements"
        if not dataset_meas_dir.is_dir():
            continue
        for parquet_file in sorted(dataset_meas_dir.glob("*.parquet")):
            path_to_dataset[parquet_file] = dataset_name

    lazy_frames = scan_parquets(list(path_to_dataset.keys()))

    for pq_path, lf in lazy_frames.items():
        dataset_name = path_to_dataset[pq_path]
        try:
            df = lf.collect()
            if _DATASET_COL not in df.columns:
                df = df.insert_column(
                    0, pl.lit(dataset_name).alias(_DATASET_COL)
                )
            if "Metadata_ImageFile" not in df.columns:
                df = df.insert_column(
                    min(1, df.width),
                    pl.lit(pq_path.stem).alias("Metadata_ImageFile"),
                )
            all_measurements.append(df)
        except Exception as e:
            logger.warning(
                "Failed to read %s: %s: %s",
                pq_path,
                type(e).__name__,
                e,
            )
            n_skipped += 1

    if n_skipped:
        logger.warning("Skipped %d Parquet file(s) due to read errors", n_skipped)

    if not all_measurements:
        return None

    try:
        master_df = pl.concat(all_measurements, how="diagonal_relaxed")
    except Exception as e:
        logger.error("Failed to concatenate measurements: %s", e)
        return None

    # Join external metadata if provided.
    if metadata_csv is not None:
        try:
            metadata_df = pl.read_csv(metadata_csv)
            common = [
                c for c in master_df.columns if c in metadata_df.columns
            ]
            if not common:
                logger.warning(
                    "Metadata CSV has no columns in common with measurements "
                    "-- skipping join"
                )
            else:
                logger.info("Joining metadata on columns: %s", common)
                # Cast join keys to string for consistent matching.
                master_df = master_df.with_columns(
                    pl.col(c).cast(pl.String) for c in common
                )
                metadata_df = metadata_df.with_columns(
                    pl.col(c).cast(pl.String) for c in common
                )
                n_rows_before = master_df.height
                master_df = master_df.join(
                    metadata_df, on=common, how="left"
                )
                if master_df.height > n_rows_before:
                    logger.warning(
                        "Metadata join increased row count from %d to %d -- "
                        "metadata CSV likely has duplicate keys on columns %s",
                        n_rows_before,
                        master_df.height,
                        common,
                    )
        except Exception as e:
            logger.warning(
                "Failed to read/join metadata CSV %s: %s: %s",
                metadata_csv,
                type(e).__name__,
                e,
            )

    return master_df


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------


def _prepare_scatter_data(df: pl.DataFrame, max_rows: int = 10_000) -> dict:
    """Prepare columnar data for scatter plots.

    Selects up to 25 key columns (prioritised by prefix) and optionally
    down-samples rows via stratified sampling.

    Args:
        df: Merged measurement DataFrame.
        max_rows: Maximum rows to include; excess rows are stratified-sampled.

    Returns:
        JSON-serializable dict with ``columns``, ``data``, ``total_rows``,
        and ``sampled`` keys.
    """
    columns = _select_scatter_columns(df.columns)
    sub = df.select(columns)

    total_rows = sub.height
    sampled = total_rows > max_rows
    if sampled:
        sub = _stratified_sample(sub, max_rows)

    return _to_columnar(sub, total_rows, sampled)


def _prepare_table_data(df: pl.DataFrame, max_rows: int = 5_000) -> dict:
    """Prepare columnar data for the raw-data table.

    Includes all columns.  Rows are stratified-sampled when they
    exceed *max_rows*.

    Args:
        df: Merged measurement DataFrame.
        max_rows: Maximum rows to include.

    Returns:
        JSON-serializable columnar dict.
    """
    total_rows = df.height
    sampled = total_rows > max_rows
    if sampled:
        df = _stratified_sample(df, max_rows)

    return _to_columnar(df, total_rows, sampled)


def _prepare_summary_stats(df: pl.DataFrame) -> dict:
    """Compute per-dataset descriptive statistics on the full dataset.

    Groups by ``Metadata_Dataset`` and computes count, mean, std, min,
    max, median, and coefficient of variation for each numeric column.

    Args:
        df: Merged measurement DataFrame.

    Returns:
        JSON-serializable dict with ``datasets`` and ``column_groups``.
    """
    numeric_cols = [
        c for c in df.columns if df[c].dtype.is_numeric()
    ]

    # Build column groups by splitting on the first underscore.
    column_groups: Dict[str, List[str]] = {}
    for col in numeric_cols:
        if "_" in col:
            group = col.split("_", 1)[0]
        else:
            group = col
        column_groups.setdefault(group, []).append(col)

    groups = _partition_by_dataset(df)

    datasets: Dict[str, dict] = {}
    for ds_name, group_df in sorted(groups.items()):
        col_stats: Dict[str, dict] = {}
        for col in numeric_cols:
            series = group_df[col]
            count = int(series.drop_nulls().len())
            mean = series.mean() if count else None
            std = series.std() if count else None
            col_min = series.min() if count else None
            col_max = series.max() if count else None
            median = series.median() if count else None

            if mean is not None and std is not None and mean != 0:
                cv = std / abs(mean) * 100
            else:
                cv = None

            col_stats[col] = {
                "count": count,
                "mean": _sanitize_for_json(mean),
                "std": _sanitize_for_json(std),
                "min": _sanitize_for_json(col_min),
                "max": _sanitize_for_json(col_max),
                "median": _sanitize_for_json(median),
                "cv": _sanitize_for_json(cv),
            }

        datasets[str(ds_name)] = {"columns": col_stats}

    return {
        "datasets": datasets,
        "column_groups": column_groups,
    }


def _prepare_overlay_manifest(output_dir: Path) -> dict:
    """Build dataset-to-overlay-image mapping from overlay PNGs.

    Scans ``results/*/overlays/*.png`` and groups filenames by dataset.

    Args:
        output_dir: Root output directory.

    Returns:
        Dict with a ``datasets`` key mapping dataset names to sorted
        lists of overlay image filenames.
    """
    results_dir = output_dir / "results"
    datasets: Dict[str, List[str]] = {}

    if not results_dir.is_dir():
        return {"datasets": datasets}

    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        overlay_dir = dataset_dir / "overlays"
        if not overlay_dir.is_dir():
            continue
        png_files = sorted(f.name for f in overlay_dir.glob("*.png"))
        if png_files:
            datasets[dataset_dir.name] = png_files

    return {"datasets": datasets}


# ---------------------------------------------------------------------------
# Partitioning helper
# ---------------------------------------------------------------------------


def _partition_by_dataset(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """Partition *df* by ``Metadata_Dataset``, returning a flat string-keyed dict.

    Falls back to a single ``"all"`` group when the column is absent.
    Handles the tuple-key format that ``partition_by(as_dict=True)``
    returns in Polars.

    Args:
        df: DataFrame to partition.

    Returns:
        Mapping from dataset name to its sub-DataFrame.
    """
    if _DATASET_COL not in df.columns:
        return {"all": df}

    raw = df.partition_by(_DATASET_COL, as_dict=True)
    return {
        str(k[0]) if isinstance(k, tuple) else str(k): v
        for k, v in raw.items()
    }


# ---------------------------------------------------------------------------
# Sampling and column selection
# ---------------------------------------------------------------------------


def _stratified_sample(
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

    if _DATASET_COL not in df.columns:
        return df.sample(n=max_rows, seed=seed)

    # Proportional allocation per dataset.
    groups = _partition_by_dataset(df)
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


def _select_scatter_columns(all_columns: List[str], max_cols: int = 25) -> List[str]:
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
    for prefix in _SCATTER_PREFIX_PRIORITY:
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


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _sanitize_for_json(value: Any) -> Any:
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


def _to_columnar(
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
            data[col] = [_sanitize_for_json(v) for v in series.to_list()]
        else:
            data[col] = series.to_list()

    return {
        "columns": columns,
        "data": data,
        "total_rows": total_rows,
        "sampled": sampled,
    }


def _write_json_atomic(payload: dict, target_path: Path) -> None:
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


def _write_parquet_sidecar(merged_df: pl.DataFrame, target_path: Path) -> None:
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
