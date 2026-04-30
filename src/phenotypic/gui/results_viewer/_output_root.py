"""Output-root discovery and read-only access for the results viewer.

The results viewer consumes a CLI output directory produced by
``python -m phenotypic`` and never mutates it (other than its own tile
cache under ``.viewer_cache/``). This module locates the master
measurements parquet, validates the expected layout, and exposes a
small set of helpers used by the rest of the viewer package.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import polars as pl

logger = logging.getLogger(__name__)

_MASTER_FILENAME = "master_measurements.parquet"
_RESULTS_DIRNAME = "results"
_OVERLAYS_DIRNAME = "overlays"
_CACHE_RELATIVE = Path(".viewer_cache") / "dzi"
_PIPELINE_JSON = "pipeline.json"
_REQUIRED_COLUMNS = ("Metadata_Dataset", "Metadata_ImageFile")


@dataclass(frozen=True)
class OutputRoot:
    """Validated, read-only handle on a PhenoTypic CLI output directory.

    The dataclass aggregates all viewer-relevant artefacts of a single
    output run: the master measurements DataFrame (one row per object),
    a per-column inventory of unique string values for the filter
    sidebar, the on-disk DZI tile cache directory, and an optional
    one-line pipeline label parsed from ``pipeline.json``.

    Attributes:
        root: Absolute path of the output directory.
        master_df: Master measurements DataFrame loaded from
            ``master_measurements.parquet``. Each row is a single
            measured object.
        column_value_sets: Mapping from column name to the sorted
            list of unique string values found in that column. Used
            to populate filter dropdowns; nulls are dropped.
        cache_dir: Path to ``<root>/.viewer_cache/dzi``. Created on
            discovery if missing.
        pipeline_summary: One-line label parsed from
            ``<root>/pipeline.json`` (typically the pipeline ``name``)
            or ``None`` if the file is absent or unparseable.
    """

    root: Path
    master_df: pl.DataFrame
    column_value_sets: Mapping[str, list[str]]
    cache_dir: Path
    pipeline_summary: str | None

    @classmethod
    def discover(cls, root: Path) -> OutputRoot:
        """Validate the output layout and assemble an ``OutputRoot``.

        The expected layout is:

            <root>/master_measurements.parquet
            <root>/results/<dataset>/overlays/<image_stem>.png
            <root>/pipeline.json                  # optional

        Args:
            root: Path to a CLI output directory.

        Returns:
            A populated, frozen ``OutputRoot``.

        Raises:
            FileNotFoundError: If ``master_measurements.parquet`` is
                missing, or if the ``results/`` directory is missing
                or contains no datasets with an ``overlays`` subdir.
            ValueError: If the master DataFrame lacks either
                ``Metadata_Dataset`` or ``Metadata_ImageFile``.
        """
        root = Path(root).resolve()

        master_path = root / _MASTER_FILENAME
        if not master_path.is_file():
            raise FileNotFoundError(
                f"Master measurements parquet not found at {master_path!s}. "
                "Run `python -m phenotypic` to produce a CLI output directory "
                "before launching the results viewer."
            )

        logger.info("Loading master measurements from %s", master_path)
        master_df = pl.read_parquet(master_path)

        missing = [c for c in _REQUIRED_COLUMNS if c not in master_df.columns]
        if missing:
            raise ValueError(
                "Master measurements parquet is missing required column(s) "
                f"{missing!r}. Expected columns include {list(_REQUIRED_COLUMNS)!r}; "
                f"found columns: {master_df.columns!r}."
            )

        results_dir = root / _RESULTS_DIRNAME
        if not results_dir.is_dir():
            raise FileNotFoundError(
                f"Expected results directory not found at {results_dir!s}. "
                "The viewer requires a layout of "
                "<root>/results/<dataset>/overlays/<image_stem>.png produced "
                "by `python -m phenotypic`."
            )

        datasets = sorted(
            entry.name
            for entry in results_dir.iterdir()
            if entry.is_dir() and (entry / _OVERLAYS_DIRNAME).is_dir()
        )
        if not datasets:
            raise FileNotFoundError(
                f"No dataset overlays directories found under {results_dir!s}. "
                "Expected at least one <root>/results/<dataset>/overlays/ "
                "directory with overlay PNGs from `python -m phenotypic`."
            )
        logger.info("Discovered datasets: %s", datasets)

        column_value_sets = _build_column_value_sets(master_df)

        cache_dir = root / _CACHE_RELATIVE
        cache_dir.mkdir(parents=True, exist_ok=True)

        pipeline_summary = _read_pipeline_summary(root / _PIPELINE_JSON)

        return cls(
            root=root,
            master_df=master_df,
            column_value_sets=column_value_sets,
            cache_dir=cache_dir,
            pipeline_summary=pipeline_summary,
        )

    def overlay_path(self, dataset: str, stem: str) -> Path:
        """Return the absolute path of an overlay PNG.

        Args:
            dataset: Dataset name (matches ``Metadata_Dataset``).
            stem: Image stem (matches ``Metadata_ImageFile`` minus
                its extension).

        Returns:
            ``<root>/results/<dataset>/overlays/<stem>.png``. The
            returned path is not checked for existence; use
            :meth:`has_overlay` for that.
        """
        return self.root / _RESULTS_DIRNAME / dataset / _OVERLAYS_DIRNAME / f"{stem}.png"

    def has_overlay(self, dataset: str, stem: str) -> bool:
        """Return ``True`` if the overlay PNG exists on disk.

        Args:
            dataset: Dataset name.
            stem: Image stem.

        Returns:
            ``True`` if :meth:`overlay_path` resolves to a regular
            file, ``False`` otherwise.
        """
        return self.overlay_path(dataset, stem).is_file()

    def image_pairs(self, df: pl.DataFrame) -> list[tuple[str, str]]:
        """Extract unique ``(dataset, image_stem)`` pairs from a frame.

        Both columns are cast to ``pl.String`` and rows containing
        nulls in either column are dropped before deduplication. The
        result is sorted lexicographically so the picker order is
        deterministic.

        Args:
            df: Any DataFrame that has the ``Metadata_Dataset`` and
                ``Metadata_ImageFile`` columns (typically a filtered
                slice of :attr:`master_df`).

        Returns:
            Sorted list of unique ``(dataset, stem)`` tuples.
        """
        pairs_df = (
            df.select(
                pl.col("Metadata_Dataset").cast(pl.String),
                pl.col("Metadata_ImageFile").cast(pl.String),
            )
            .drop_nulls()
            .unique()
            .sort(["Metadata_Dataset", "Metadata_ImageFile"])
        )
        return [
            (dataset, stem)
            for dataset, stem in zip(
                pairs_df.get_column("Metadata_Dataset").to_list(),
                pairs_df.get_column("Metadata_ImageFile").to_list(),
                strict=True,
            )
        ]


def _build_column_value_sets(df: pl.DataFrame) -> Mapping[str, list[str]]:
    """Compute sorted unique string values for every column."""
    value_sets: dict[str, list[str]] = {}
    for column in df.columns:
        unique_strings = (
            df.get_column(column)
            .cast(pl.String)
            .drop_nulls()
            .unique()
            .sort()
            .to_list()
        )
        value_sets[column] = unique_strings
    return MappingProxyType(value_sets)


def _read_pipeline_summary(pipeline_json: Path) -> str | None:
    """Best-effort one-line label from ``pipeline.json``.

    Returns ``None`` on any read or parse error rather than raising,
    so a malformed pipeline manifest never blocks the viewer from
    booting.
    """
    if not pipeline_json.is_file():
        return None
    try:
        with pipeline_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key in ("name", "class_name"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return _PIPELINE_JSON
    except Exception:
        logger.debug("Failed to parse %s for pipeline_summary", pipeline_json, exc_info=True)
        return None
