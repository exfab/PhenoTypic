"""Lazy column-name cache for the analysis sub-app's column-aware widgets.

The analysis page renders column-name parameters (``on``, ``groupby``,
``time_label``, …) as dropdowns populated from the live measurements
schema on disk. Reading the schema is cheap when going via the parquet
footer, but cheap-times-N becomes noticeable when filter/model stacks
rebuild on every keystroke. This cache reads each source file once per
mtime so subsequent calls are O(1).

Resolution order is parquet-first then CSV fallback. A missing file
returns an empty list and the GUI surfaces a tooltip; the page does
not raise so the analysis sub-app stays usable on partially-seeded
output roots.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from phenotypic.gui._config import (
    MASTER_MEASUREMENTS_CSV,
    MASTER_MEASUREMENTS_PARQUET,
    MEASUREMENTS_CSV,
    MEASUREMENTS_PARQUET,
)
from phenotypic.sdk_ import deliverables_dir

if TYPE_CHECKING:
    from phenotypic.sdk_ import ColumnSource

logger = logging.getLogger(__name__)


#: ``source -> (parquet filename, csv filename)``. Resolution always
#: prefers the parquet footer; the CSV mirror is the no-pyarrow fallback.
_FILES_BY_SOURCE: "dict[ColumnSource, tuple[str, str]]" = {
    "measurements": (MEASUREMENTS_PARQUET, MEASUREMENTS_CSV),
    "master_measurements": (MASTER_MEASUREMENTS_PARQUET, MASTER_MEASUREMENTS_CSV),
}


@dataclass
class MeasurementSchema:
    """Lazy cache of column-name lists keyed by source + mtime.

    Attributes:
        output_root: Path to the CLI output directory whose
            ``deliverables/`` subdirectory holds
            ``measurements.{parquet,csv}`` and / or
            ``master_measurements.{parquet,csv}``.
    """

    output_root: Path
    #: ``source -> (sentinel mtime_ns, columns)``. The sentinel is the
    #: highest mtime observed across the parquet+csv pair so a CSV-only
    #: refresh still invalidates the cache.
    _cache: dict[str, tuple[int, list[str]]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def columns_for(self, source: "ColumnSource | str") -> list[str]:
        """Return the column list for ``source``.

        Uses the parquet footer when available and falls back to a
        zero-row CSV scan. Missing files return ``[]`` so the dropdown
        renders blank rather than crashing the page.

        Concurrency: the lock spans both the cache lookup AND the
        ``_read_columns`` call so concurrent readers serialize while a
        miss is materializing. Reads are footer-only (a few ms in
        practice) so the contention window is small. Two readers that
        both observe the same mtime before either enters the lock
        agree on the same cached entry — the second hits the cache
        the first thread just stored.

        Args:
            source: ``"measurements"`` or ``"master_measurements"``.

        Returns:
            List of column names in file order. Empty when neither the
            parquet nor the CSV mirror exists.
        """
        source_str = str(source)
        # Cast through `Any`-compatible str lookup since the typed dict only
        # accepts `ColumnSource`; users can pass either form per the signature.
        files = _FILES_BY_SOURCE.get(source_str)  # type: ignore[arg-type]
        if files is None:
            logger.warning("Unknown column source %r; returning []", source_str)
            return []

        deliverables = deliverables_dir(self.output_root)
        parquet_path = deliverables / files[0]
        csv_path = deliverables / files[1]
        sentinel = _max_mtime_ns(parquet_path, csv_path)

        with self._lock:
            cached = self._cache.get(source_str)
            if cached is not None and cached[0] == sentinel:
                return cached[1]
            columns = _read_columns(parquet_path, csv_path)
            self._cache[source_str] = (sentinel, columns)
            return columns

    def invalidate(self) -> None:
        """Drop every cached entry. Forces a fresh read on the next call."""
        with self._lock:
            self._cache.clear()


def _max_mtime_ns(*paths: Path) -> int:
    """Return the highest ``stat().st_mtime_ns`` of the existing paths.

    Returns ``-1`` when none of the paths exist, which makes a missing
    pair distinguishable from any real (non-negative) mtime.
    """
    mtimes = []
    for p in paths:
        try:
            mtimes.append(p.stat().st_mtime_ns)
        except FileNotFoundError:
            continue
    return max(mtimes, default=-1)


def _read_columns(parquet_path: Path, csv_path: Path) -> list[str]:
    """Read the column list from parquet (preferred) or CSV (fallback)."""
    if parquet_path.exists():
        try:
            return pl.scan_parquet(parquet_path).collect_schema().names()
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to scan parquet %s; falling back to CSV",
                parquet_path,
                exc_info=True,
            )

    if csv_path.exists():
        try:
            return pl.scan_csv(csv_path, n_rows=0).collect_schema().names()
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to scan CSV %s",
                csv_path,
                exc_info=True,
            )

    return []


__all__ = ["MeasurementSchema"]
