"""Context dataclass passed to analysis plugin ``prepare_data`` methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl


@dataclass(frozen=True)
class AnalysisPrepareContext:
    """Immutable context for plugin data preparation.

    Attributes:
        output_dir: Root output directory (contains ``results/`` and
            ``progress/``).
        progress_dir: Directory for writing sidecar JSON/Parquet files.
        merged_df: Merged measurement DataFrame, or ``None`` if no
            measurement data was found.
    """

    output_dir: Path
    progress_dir: Path
    merged_df: Optional[pl.DataFrame]
