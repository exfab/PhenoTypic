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

import polars as pl

from phenotypic.gui._config import (
    DIR_MEASUREMENTS,
    VIEWER_CACHE_DIRNAME,
)
from phenotypic.gui.results_viewer._filtered_state import KEY_DATASET, KEY_IMAGE_FILE
from phenotypic.sdk_ import (
    DIR_OVERLAYS,
    BundleLayout,
    migrate_legacy_qc,
)

logger = logging.getLogger(__name__)

#: The post-applied mirror seeded by ``_seed_measurements``. Preferred
#: over the master archive (``master_measurements.parquet``) for the
#: viewer's display frame because it reflects whatever ``PostMeasurement``
#: ops the user configured.
_CACHE_RELATIVE = Path(VIEWER_CACHE_DIRNAME) / "dzi"
_IMAGENAME_COL = "Metadata_ImageName"


@dataclass(frozen=True)
class OutputRoot:
    """Validated, read-only handle on a PhenoTypic CLI output directory.

    The dataclass aggregates all viewer-relevant artefacts of a single
    output run: the master measurements DataFrame (one row per object),
    a per-column inventory of unique string values for the filter
    sidebar, the on-disk DZI tile cache directory, and an optional
    one-line pipeline label parsed from ``pipeline.json``.

    Attributes:
        root: Absolute path of the run output directory for a full run, or
            the deliverables folder itself for a standalone bundle (where
            ``layout.output_root is None``). Prefer routing path resolution
            through :attr:`layout` rather than re-joining ``root``.
        layout: The resolved :class:`~phenotypic.sdk_.BundleLayout` topology
            (deliverables base + optional output root) backing this handle.
            All deliverables/qc/error path resolution goes through it.
        master_df: The viewer's display frame — the post-applied
            ``measurements.parquet`` mirror when present (so the filter
            sidebar reflects post ops), falling back to the clean master
            mid-run. Curation removes labeled rows from this mirror, so it is
            NOT a complete object set after curation.
        clean_master_df: The clean, pre-post ``master_measurements.parquet``
            frame — the full object set, including objects the curated mirror
            removed. Curation re-keying and the Error-analysis tab read this so
            labeled objects (which the mirror drops) remain resolvable across a
            viewer reload.
        column_value_sets: Mapping from column name to the sorted
            list of unique string values found in that column. Used
            to populate filter dropdowns; nulls are dropped.
        cache_dir: Path to ``<root>/.viewer_cache/dzi``. Created on
            discovery if missing.
        pipeline_summary: One-line label parsed from
            ``<root>/deliverables/pipeline.json`` (typically the
            pipeline ``name``) or ``None`` if the file is absent or
            unparseable.
    """

    root: Path
    layout: BundleLayout
    master_df: pl.DataFrame
    clean_master_df: pl.DataFrame
    column_value_sets: Mapping[str, list[str]]
    cache_dir: Path
    pipeline_summary: str | None
    #: Snapshot of ``(dataset, stem)`` pairs that have an overlay PNG on
    #: disk at discovery time. Used as an O(1) replacement for the
    #: :meth:`has_overlay` per-call ``stat`` so picker callbacks don't
    #: hit the filesystem on every render.
    overlay_index: frozenset[tuple[str, str]]

    @classmethod
    def discover(cls, root: Path) -> OutputRoot:
        """Classify the on-disk topology and assemble an ``OutputRoot``.

        Accepts either a full ``python -m phenotypic`` output directory
        (with per-image ``results/`` + machine state) **or** a standalone,
        portable ``deliverables/`` bundle (master + mirror + overlays only).
        :class:`~phenotypic.sdk_.BundleLayout` resolves which, and every
        deliverables/qc/error path is anchored on it so the standalone bundle
        is self-contained::

            <deliverables>/master_measurements.parquet
            <deliverables>/overlays/<dataset>/<image_stem>.png
            <deliverables>/pipeline.json          # optional
            <output_root>/results/<dataset>/...    # full runs only

        Args:
            root: Path to a CLI output directory or a deliverables bundle.

        Returns:
            A populated, frozen ``OutputRoot``.

        Raises:
            FileNotFoundError: If neither a deliverables bundle nor a run
                output directory can be located at ``root`` (no
                ``master_measurements.parquet``), or no datasets are found.
            ValueError: If the master DataFrame lacks ``Metadata_ImageFile``
                (or the ``Metadata_ImageName`` fallback), or lacks
                ``Metadata_Dataset`` with no ``results/`` to recover it from.
        """
        layout = BundleLayout.detect(Path(root))
        if layout.output_root is not None:
            migrate_legacy_qc(layout.output_root)

        master_path = layout.master_parquet
        if not master_path.is_file():
            raise FileNotFoundError(
                f"Master measurements parquet not found at {master_path!s}. "
                "Point the viewer at a `python -m phenotypic` output dir or a "
                "deliverables/ bundle."
            )

        # Prefer the post-applied mirror seeded by ``_seed_measurements``
        # so filter sidebars, image picker, and column-value sets reflect
        # whatever ``PostMeasurement`` ops the user configured. The clean
        # (pre-post) master is the FULL object set — including objects the
        # curated mirror removes — read so curation re-keying + the Error tab
        # keep labels resolvable across a viewer reload. Falls back to master
        # mid-run / on legacy outputs without the mirror.
        clean_master_df = pl.read_parquet(master_path)

        mirror_path = layout.mirror_parquet
        if mirror_path.is_file():
            logger.info(
                "Loading post-applied measurements mirror from %s", mirror_path
            )
            master_df = pl.read_parquet(mirror_path)
        else:
            logger.info(
                "Mirror %s not found; loading clean master from %s",
                mirror_path,
                master_path,
            )
            master_df = clean_master_df

        # Datasets are data-driven: the master frame is authoritative, unioned
        # with overlay subdirs (and results/ when present) to catch a dataset
        # that has overlays but no surviving rows.
        datasets = _discover_datasets(master_df, layout)
        if not datasets:
            raise FileNotFoundError(
                f"No datasets found in {layout.deliverables_base!s}. Expected a "
                "Metadata_Dataset column or deliverables/overlays/<dataset>/ dirs."
            )

        master_df = _ensure_required_columns(master_df, layout, datasets)
        clean_master_df = _ensure_required_columns(clean_master_df, layout, datasets)

        datasets_with_overlays = [
            ds for ds in datasets if layout.overlays_dir(ds).is_dir()
        ]
        if datasets_with_overlays:
            logger.info(
                "Discovered datasets: %s (with overlays: %s)",
                datasets,
                datasets_with_overlays,
            )
        else:
            logger.warning(
                "Discovered datasets: %s — none have a deliverables/overlays/<dataset>/ "
                "directory. Image picker entries will be disabled. Re-run with "
                "`--save-overlays` to enable pixel-level viewing.",
                datasets,
            )

        column_value_sets = _build_column_value_sets(master_df)

        # Cache root: the output root for full runs, else inside the bundle.
        # Read-only-mount fallback (spec Section 2 / review C5): if the chosen
        # location is not writable, fall back to a per-session temp dir keyed
        # by the bundle path so the viewer still boots off a read-only mount.
        cache_root = (
            layout.output_root
            if layout.output_root is not None
            else layout.deliverables_base
        )
        cache_dir = cache_root / _CACHE_RELATIVE
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            import hashlib
            import tempfile

            key = hashlib.sha1(
                str(layout.deliverables_base).encode()
            ).hexdigest()[:12]
            cache_dir = (
                Path(tempfile.gettempdir())
                / f"phenotypic-viewer-{key}"
                / _CACHE_RELATIVE
            )
            cache_dir.mkdir(parents=True, exist_ok=True)

        pipeline_summary = _read_pipeline_summary(layout.pipeline_config_path)
        overlay_index = _scan_overlay_index(layout, datasets_with_overlays)

        return cls(
            root=(
                layout.deliverables_base
                if layout.output_root is None
                else layout.output_root
            ),
            layout=layout,
            master_df=master_df,
            clean_master_df=clean_master_df,
            column_value_sets=column_value_sets,
            cache_dir=cache_dir,
            pipeline_summary=pipeline_summary,
            overlay_index=overlay_index,
        )

    @property
    def has_results(self) -> bool:
        """Whether per-image ``results/`` are available (full run, not a bundle)."""
        return self.layout.has_results

    def hdf_path(self, dataset: str, stem: str) -> Path | None:
        """Full-res per-image HDF path, or ``None`` for a standalone bundle."""
        return self.layout.hdf_path(dataset, stem)

    @property
    def results_dir(self) -> Path | None:
        """Path to ``results/``, or ``None`` for a standalone bundle.

        Callers MUST guard for ``None`` before joining a dataset onto it.
        """
        return self.layout.results_dir

    @property
    def viewer_cache_dir(self) -> Path:
        """The cache *root* (parent of the ``dzi/`` subdir held by :attr:`cache_dir`).

        Thumb routes and any non-DZI cache write under here so the standalone
        path stays inside the bundle (review Q2 — distinct from
        ``cache_dir = .../dzi``).
        """
        return self.cache_dir.parent

    def overlay_path(self, dataset: str, stem: str) -> Path:
        """Return the absolute path of an overlay PNG.

        Args:
            dataset: Dataset name (matches ``Metadata_Dataset``).
            stem: Image stem (matches ``Metadata_ImageFile`` minus
                its extension).

        Returns:
            ``<deliverables>/overlays/<dataset>/<stem>.png``. The returned
            path is not checked for existence; use :meth:`has_overlay`.
        """
        return self.layout.overlay_path(dataset, stem)

    def has_overlay(self, dataset: str, stem: str) -> bool:
        """Return ``True`` if the overlay PNG exists on disk.

        Backed by a frozenset snapshot taken at discovery time, so this
        is O(1) regardless of how many overlays exist. Overlays added
        after the viewer launches are not visible until restart; that is
        acceptable for an interactive review tool.

        Args:
            dataset: Dataset name.
            stem: Image stem.

        Returns:
            ``True`` if the overlay PNG was present at discovery time.
        """
        return (dataset, stem) in self.overlay_index

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
                pl.col(KEY_DATASET).cast(pl.String),
                pl.col(KEY_IMAGE_FILE).cast(pl.String),
            )
            .drop_nulls()
            .unique()
            .sort([KEY_DATASET, KEY_IMAGE_FILE])
        )
        return [
            (dataset, stem)
            for dataset, stem in zip(
                pairs_df.get_column(KEY_DATASET).to_list(),
                pairs_df.get_column(KEY_IMAGE_FILE).to_list(),
                strict=True,
            )
        ]

    def is_numeric_column(self, column: str) -> bool:
        """Return ``True`` if ``column`` can be filtered numerically.

        True when the column's polars dtype is numeric (covers every
        ``Size_*`` / ``Shape_*`` / ``Intensity_*`` measurement column for
        free), or when its filter value-set parses entirely as floats
        (covers numeric-valued string metadata like ``Metadata_Time``).
        Unknown columns return ``False``. Drives the Range/Compare gate in
        the filter sidebar.
        """
        if column not in self.master_df.columns:
            return False
        if self.master_df.schema[column].is_numeric():
            return True
        return _all_parse_as_float(self.column_value_sets.get(column, []))


def _discover_datasets(master_df: pl.DataFrame, layout: BundleLayout) -> list[str]:
    """Enumerate dataset names from the master frame, overlays, and results/.

    Datasets are data-driven: the master frame's ``Metadata_Dataset`` values
    are authoritative, unioned with the ``overlays/<dataset>/`` subdirs (a
    standalone bundle has no ``results/``) and the ``results/<dataset>/`` subdirs
    when present (full run), so a dataset with overlays but no surviving rows is
    still discovered.

    Args:
        master_df: The loaded display frame.
        layout: Resolved bundle topology.

    Returns:
        Sorted unique dataset names.
    """
    names: set[str] = set()
    if KEY_DATASET in master_df.columns:
        names.update(
            str(v)
            for v in master_df.get_column(KEY_DATASET).drop_nulls().unique().to_list()
        )
    overlays_root = layout.deliverables_base / DIR_OVERLAYS
    if overlays_root.is_dir():
        names.update(e.name for e in overlays_root.iterdir() if e.is_dir())
    if layout.results_dir is not None:
        names.update(e.name for e in layout.results_dir.iterdir() if e.is_dir())
    return sorted(names)


def _ensure_required_columns(
    df: pl.DataFrame,
    layout: BundleLayout,
    datasets: list[str],
) -> pl.DataFrame:
    """Backfill ``Metadata_Dataset`` and ``Metadata_ImageFile`` if missing.

    Real-world masters produced by older runs or by aggregators that
    skip ``include_dataset_column`` may lack one or both of these
    columns. The dataset is recoverable from the on-disk layout
    (``results/<dataset>/measurements/<stem>.parquet``) when a full-run
    ``results/`` is present; the image stem can fall back to
    ``Metadata_ImageName`` when present.

    Args:
        df: Loaded master DataFrame.
        layout: Resolved bundle topology (``layout.results_dir`` is ``None``
            for a standalone bundle, which makes dataset backfill impossible).
        datasets: Dataset directory names already discovered.

    Returns:
        The DataFrame with both required columns guaranteed present.

    Raises:
        ValueError: If neither column can be derived (e.g. a standalone bundle
            whose master lacks ``Metadata_Dataset``).
    """
    if KEY_IMAGE_FILE not in df.columns and _IMAGENAME_COL in df.columns:
        logger.info(
            "Master lacks %s; aliasing %s as the image stem column.",
            KEY_IMAGE_FILE,
            _IMAGENAME_COL,
        )
        df = df.with_columns(
            pl.col(_IMAGENAME_COL).cast(pl.String).alias(KEY_IMAGE_FILE)
        )

    if KEY_IMAGE_FILE not in df.columns:
        raise ValueError(
            f"Master measurements parquet is missing column {KEY_IMAGE_FILE!r} "
            f"(and the {_IMAGENAME_COL!r} fallback). Re-run `python -m phenotypic` "
            f"with the current version to regenerate the master."
        )

    if KEY_DATASET in df.columns:
        return df

    if layout.results_dir is None:
        raise ValueError(
            "Master measurements parquet is missing column 'Metadata_Dataset' and "
            "this is a standalone deliverables bundle (no results/ to recover it "
            "from). Recompile the run with the current version: "
            "`python -m phenotypic --mode recompile --output <dir>`."
        )
    results_root = layout.results_dir

    stem_to_dataset: dict[str, str] = {}
    collisions: set[str] = set()
    for dataset in datasets:
        meas_dir = results_root / dataset / DIR_MEASUREMENTS
        if not meas_dir.is_dir():
            continue
        for entry in meas_dir.iterdir():
            if entry.suffix != ".parquet":
                continue
            stem = entry.stem
            existing = stem_to_dataset.get(stem)
            if existing is not None and existing != dataset:
                collisions.add(stem)
                continue
            stem_to_dataset[stem] = dataset

    if not stem_to_dataset:
        raise ValueError(
            f"Master measurements parquet is missing column {KEY_DATASET!r} "
            "and no per-image parquets were found under "
            f"{results_root!s}/<dataset>/measurements/ to recover it from."
        )

    if collisions:
        logger.warning(
            "%d image stems appear in multiple dataset directories "
            "(%s). The first-seen dataset wins; expect ambiguous filtering.",
            len(collisions),
            sorted(collisions)[:5],
        )

    logger.info(
        "Backfilling %s from filesystem layout (%d stems mapped across %d datasets).",
        KEY_DATASET,
        len(stem_to_dataset),
        len(datasets),
    )
    mapping_df = pl.DataFrame(
        {
            KEY_IMAGE_FILE: list(stem_to_dataset.keys()),
            KEY_DATASET: list(stem_to_dataset.values()),
        }
    )
    df = df.with_columns(pl.col(KEY_IMAGE_FILE).cast(pl.String))
    enriched = df.join(mapping_df, on=KEY_IMAGE_FILE, how="left")

    null_count = enriched.get_column(KEY_DATASET).null_count()
    if null_count:
        logger.warning(
            "%d/%d master rows could not be linked to a dataset directory and "
            "will be excluded from filter results.",
            null_count,
            enriched.height,
        )
    return enriched


def _scan_overlay_index(
    layout: BundleLayout, datasets_with_overlays: list[str]
) -> frozenset[tuple[str, str]]:
    """Snapshot every ``(dataset, stem)`` whose overlay PNG exists on disk.

    Args:
        layout: The resolved bundle topology (overlays anchor on its
            deliverables base).
        datasets_with_overlays: Dataset names known to have a
            ``deliverables/overlays/<dataset>/`` directory; pre-filtered
            by the discovery scan to avoid an extra ``is_dir`` check.

    Returns:
        Frozen set of ``(dataset, stem)`` tuples; the stem is the PNG
        filename minus its ``.png`` suffix.
    """
    pairs: set[tuple[str, str]] = set()
    for dataset in datasets_with_overlays:
        for entry in layout.overlays_dir(dataset).iterdir():
            if entry.suffix.lower() == ".png" and entry.is_file():
                pairs.add((dataset, entry.stem))
    return frozenset(pairs)


_METADATA_PREFIX = "Metadata_"


def _all_parse_as_float(values: list[str]) -> bool:
    """Return True iff ``values`` is non-empty and every entry parses as float.

    Used to decide whether a column's filter options should be sorted
    numerically (``"2"`` before ``"10"``) and whether a column is
    range/compare-eligible. Empty input returns ``False`` (nothing to
    sort numerically).
    """
    if not values:
        return False
    for value in values:
        try:
            float(value)
        except (TypeError, ValueError):
            return False
    return True


class _LazyColumnValueSets(Mapping[str, list[str]]):
    """Sorted unique string values per column, computed on first access.

    Eagerly materialises ``Metadata_*`` columns (always small, always
    surfaced in the filter sidebar) and defers everything else until a
    callback actually asks for it. Avoids the multi-MB string allocation
    that would otherwise happen at boot for wide masters with hundreds
    of high-cardinality measurement columns.
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df
        self._cache: dict[str, list[str]] = {}
        for column in df.columns:
            if column.startswith(_METADATA_PREFIX):
                self._cache[column] = self._compute(column)

    def _compute(self, column: str) -> list[str]:
        values = (
            self._df.get_column(column)
            .cast(pl.String)
            .drop_nulls()
            .unique()
            .to_list()
        )
        if _all_parse_as_float(values):
            return sorted(values, key=float)
        return sorted(values)

    def __getitem__(self, column: str) -> list[str]:
        if column not in self._df.columns:
            raise KeyError(column)
        cached = self._cache.get(column)
        if cached is None:
            cached = self._compute(column)
            self._cache[column] = cached
        return cached

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._df.columns)

    def __len__(self) -> int:
        return len(self._df.columns)


def _build_column_value_sets(df: pl.DataFrame) -> Mapping[str, list[str]]:
    """Return a mapping from column name to sorted unique string values."""
    return _LazyColumnValueSets(df)


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
        return None
    except Exception:
        logger.debug("Failed to parse %s for pipeline_summary", pipeline_json, exc_info=True)
        return None
