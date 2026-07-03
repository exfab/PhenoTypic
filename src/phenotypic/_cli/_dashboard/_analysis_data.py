"""Sidecar data preparation for the Analysis tab.

Scans per-image measurement Parquet files, optionally inner-joins metadata,
and dispatches to registered analysis plugins to write their sidecar files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import polars as pl

from .._measurement_sources import (
    add_metadata_image_name_from_filename,
    discover_measurement_sources,
    measurement_sources_by_path,
)
from .._cli_output_manager import join_metadata
from .._cli_parquet_agg import aggregate_parquet_files
from phenotypic.sdk_ import (
    progress_dir,
)

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
            inner-joining onto measurements.
    """
    from phenotypic.sdk_.register import AnalysisPluginRegistry

    from ._analysis._prepare_context import AnalysisPrepareContext

    # Trigger registration of all plugins.
    from ._analysis import _image_viewer, _raw_table, _scatter_plot, _summary_stats  # noqa: F401

    prog_dir = progress_dir(output_dir)
    prog_dir.mkdir(parents=True, exist_ok=True)

    merged_df = _load_and_merge(output_dir, metadata_csv)

    ctx = AnalysisPrepareContext(
        output_dir=output_dir,
        progress_dir=prog_dir,
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
    """Scan Parquet measurement files, aggregate with polars, and optionally join metadata.

    Walks ``results/*/measurements/`` for ``.parquet`` files.  Uses
    :func:`aggregate_parquet_files` for efficient in-memory concatenation,
    adds ``Metadata_Dataset`` and ``Metadata_ImageName`` columns, and
    performs an inner-join with *metadata_csv* when provided.

    Args:
        output_dir: Root output directory (contains ``results/``).
        metadata_csv: Optional external metadata CSV path.

    Returns:
        Merged DataFrame, or ``None`` if no measurements were found.
    """
    path_to_dataset = measurement_sources_by_path(
        discover_measurement_sources(output_dir)
    )

    if not path_to_dataset:
        return None

    # -- Polars aggregation --------------------------------------------
    master_df = aggregate_parquet_files(
        file_paths=list(path_to_dataset.keys()),
        path_to_dataset=path_to_dataset,
        include_dataset_column=True,
        keep_filename=True,
    )

    if master_df is None:
        return None

    master_df = add_metadata_image_name_from_filename(master_df)

    # -- Join external metadata if provided ----------------------------
    if metadata_csv is not None:
        try:
            master_df = join_metadata(master_df, metadata_csv)
        except Exception as e:
            logger.warning(
                "Failed to read/join metadata CSV %s: %s: %s",
                metadata_csv,
                type(e).__name__,
                e,
            )

    return master_df
