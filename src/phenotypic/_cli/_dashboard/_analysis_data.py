"""Sidecar data preparation for the Analysis tab.

Scans per-image measurement CSVs, optionally left-joins metadata,
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

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Prefix priority for column selection in scatter data.
_SCATTER_PREFIX_PRIORITY = ("Metadata_", "Grid_", "Shape_", "Intensity_", "Color_")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_analysis_sidecar(
    output_dir: Path,
    metadata_csv: Optional[Path] = None,
) -> None:
    """Write JSON sidecar files for the dashboard Analysis tab.

    Discovers datasets under ``results/``, loads and merges measurement
    CSVs, and writes four JSON files into ``progress/``:

    * ``analysis_scatter.json`` -- columnar data for scatter plots
    * ``analysis_table.json`` -- columnar data for the raw-data table
    * ``analysis_stats.json`` -- per-dataset descriptive statistics
    * ``overlay_manifest.json`` -- dataset-to-overlay-image mapping

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

    logger.debug(
        "Wrote analysis sidecar files (%d rows, %d columns)",
        len(merged_df),
        len(merged_df.columns),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_and_merge(
    output_dir: Path,
    metadata_csv: Optional[Path],
) -> Optional[pd.DataFrame]:
    """Scan measurement CSVs, concatenate, and optionally join metadata.

    Walks ``results/*/measurements/*.csv``, adds ``Metadata_Dataset``
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

    all_measurements: List[pd.DataFrame] = []
    n_skipped = 0

    for dataset_name in dataset_names:
        dataset_meas_dir = results_dir / dataset_name / "measurements"
        if not dataset_meas_dir.is_dir():
            continue

        for csv_file in sorted(dataset_meas_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file)
                if "Metadata_Dataset" not in df.columns:
                    df = df.copy()
                    df.insert(0, "Metadata_Dataset", dataset_name)
                if "Metadata_ImageFile" not in df.columns:
                    df.insert(
                        min(1, len(df.columns)),
                        "Metadata_ImageFile",
                        csv_file.stem,
                    )
                all_measurements.append(df)
            except Exception as e:
                logger.warning(
                    "Failed to read %s: %s: %s",
                    csv_file,
                    type(e).__name__,
                    e,
                )
                n_skipped += 1

    if n_skipped:
        logger.warning("Skipped %d CSV file(s) due to read errors", n_skipped)

    if not all_measurements:
        return None

    try:
        master_df = pd.concat(all_measurements, axis=0, ignore_index=True)
    except Exception as e:
        logger.error("Failed to concatenate measurements: %s", e)
        return None

    # Join external metadata if provided.
    if metadata_csv is not None:
        try:
            metadata_df = pd.read_csv(metadata_csv)
            common = list(set(master_df.columns) & set(metadata_df.columns))
            if not common:
                logger.warning(
                    "Metadata CSV has no columns in common with measurements "
                    "-- skipping join"
                )
            else:
                logger.info("Joining metadata on columns: %s", common)
                for col in common:
                    master_df[col] = master_df[col].astype(str)
                    metadata_df[col] = metadata_df[col].astype(str)
                n_rows_before = len(master_df)
                master_df = master_df.merge(metadata_df, on=common, how="left")
                if len(master_df) > n_rows_before:
                    logger.warning(
                        "Metadata join increased row count from %d to %d -- "
                        "metadata CSV likely has duplicate keys on columns %s",
                        n_rows_before,
                        len(master_df),
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


def _prepare_scatter_data(df: pd.DataFrame, max_rows: int = 10_000) -> dict:
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
    columns = _select_scatter_columns(df.columns.tolist())
    sub = df[columns]

    total_rows = len(sub)
    sampled = total_rows > max_rows
    if sampled:
        sub = _stratified_sample(sub, max_rows)

    return _to_columnar(sub, total_rows, sampled)


def _prepare_table_data(df: pd.DataFrame, max_rows: int = 5_000) -> dict:
    """Prepare columnar data for the raw-data table.

    Includes all columns.  Rows are stratified-sampled when they
    exceed *max_rows*.

    Args:
        df: Merged measurement DataFrame.
        max_rows: Maximum rows to include.

    Returns:
        JSON-serializable columnar dict.
    """
    total_rows = len(df)
    sampled = total_rows > max_rows
    if sampled:
        df = _stratified_sample(df, max_rows)

    return _to_columnar(df, total_rows, sampled)


def _prepare_summary_stats(df: pd.DataFrame) -> dict:
    """Compute per-dataset descriptive statistics on the full dataset.

    Groups by ``Metadata_Dataset`` and computes count, mean, std, min,
    max, median, and coefficient of variation for each numeric column.

    Args:
        df: Merged measurement DataFrame.

    Returns:
        JSON-serializable dict with ``datasets`` and ``column_groups``.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Build column groups by splitting on the first underscore.
    column_groups: Dict[str, List[str]] = {}
    for col in numeric_cols:
        if "_" in col:
            group = col.split("_", 1)[0]
        else:
            group = col
        column_groups.setdefault(group, []).append(col)

    # Group by dataset (fall back to a single group if column is absent).
    if "Metadata_Dataset" in df.columns:
        grouped = df.groupby("Metadata_Dataset", sort=True)
    else:
        grouped = [("all", df)]

    datasets: Dict[str, dict] = {}
    for ds_name, group_df in grouped:
        col_stats: Dict[str, dict] = {}
        for col in numeric_cols:
            series = group_df[col]
            count = int(series.count())
            mean = float(series.mean()) if count else None
            std = float(series.std()) if count else None
            col_min = float(series.min()) if count else None
            col_max = float(series.max()) if count else None
            median = float(series.median()) if count else None

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
# Sampling and column selection
# ---------------------------------------------------------------------------


def _stratified_sample(
    df: pd.DataFrame,
    max_rows: int,
    seed: int = 42,
) -> pd.DataFrame:
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
    if len(df) <= max_rows:
        return df

    if "Metadata_Dataset" not in df.columns:
        return df.sample(n=max_rows, random_state=seed)

    # Proportional allocation per dataset.
    grouped = df.groupby("Metadata_Dataset", sort=True)
    total = len(df)
    sampled_parts: List[pd.DataFrame] = []

    for ds_name, group_df in grouped:
        proportion = len(group_df) / total
        n_sample = max(1, int(round(proportion * max_rows)))
        n_sample = min(n_sample, len(group_df))
        sampled_parts.append(group_df.sample(n=n_sample, random_state=seed))

    result = pd.concat(sampled_parts, ignore_index=True)

    # Trim to exact max_rows if rounding pushed us over.
    if len(result) > max_rows:
        result = result.sample(n=max_rows, random_state=seed)

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
    used: set = set()

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
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
    except TypeError:
        pass
    return value


def _to_columnar(
    df: pd.DataFrame,
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
    columns = df.columns.tolist()
    data: Dict[str, list] = {}

    for col in columns:
        series = df[col]
        kind = series.dtype.kind
        raw = series.tolist()

        if kind == "f":
            # Float columns: sanitize NaN/Inf and convert numpy scalars.
            data[col] = [
                _sanitize_for_json(v.item() if isinstance(v, np.floating) else v)
                for v in raw
            ]
        elif kind == "O":
            # Object columns may contain mixed types; sanitise NaN.
            data[col] = [
                None if isinstance(v, float) and math.isnan(v) else v
                for v in raw
            ]
        else:
            # Integer/bool/other: convert numpy scalars to Python natives.
            data[col] = [
                v.item() if isinstance(v, (np.integer, np.bool_)) else v
                for v in raw
            ]

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
