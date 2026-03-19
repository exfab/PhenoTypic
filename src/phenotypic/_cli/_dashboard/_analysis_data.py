"""Sidecar data preparation for the Analysis tab.

Scans per-image measurement Parquet files, optionally left-joins metadata,
and dispatches to registered analysis plugins to write their sidecar files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

from .._cli_utils import scan_parquets
from ._analysis_helpers import DATASET_COL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_analysis_sidecar(
    output_dir: Path,
    metadata_csv: Optional[Path] = None,
) -> None:
    """Write JSON sidecar files for the dashboard Analysis tab.

    Discovers datasets under ``results/``, loads and merges measurement
    Parquet files, builds an :class:`AnalysisPrepareContext`, and
    dispatches to each registered plugin's ``prepare_data`` method.

    All writes are atomic (tempfile + :func:`os.replace`).

    Args:
        output_dir: Root output directory (contains ``results/`` and
            ``progress/``).
        metadata_csv: Optional path to an external metadata CSV for
            left-joining onto measurements.
    """
    from phenotypic.tools_.register import AnalysisPluginRegistry

    from ._analysis._prepare_context import AnalysisPrepareContext

    # Trigger registration of all plugins.
    from ._analysis import _image_viewer, _raw_table, _scatter_plot, _summary_stats  # noqa: F401

    progress_dir = output_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)

    merged_df = _load_and_merge(output_dir, metadata_csv)

    ctx = AnalysisPrepareContext(
        output_dir=output_dir,
        progress_dir=progress_dir,
        merged_df=merged_df,
    )

    for name in AnalysisPluginRegistry.available():
        plugin = AnalysisPluginRegistry.get(name)()
        try:
            plugin.prepare_data(ctx)
        except Exception:
            logger.exception("Plugin %r failed during prepare_data", name)

    if merged_df is not None:
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
            if DATASET_COL not in df.columns:
                df = df.insert_column(
                    0, pl.lit(dataset_name).alias(DATASET_COL)
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
                    metadata_df, on=common, how="inner"
                )
                if master_df.height > n_rows_before:
                    logger.warning(
                        "Metadata join increased row count from %d to %d -- "
                        "metadata CSV likely has duplicate keys on columns %s",
                        n_rows_before,
                        master_df.height,
                        common,
                    )
                n_dropped = n_rows_before - master_df.height
                if n_dropped > 0:
                    logger.warning(
                        "Metadata inner join dropped %d/%d measurement rows "
                        "with no matching metadata on columns %s",
                        n_dropped,
                        n_rows_before,
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
