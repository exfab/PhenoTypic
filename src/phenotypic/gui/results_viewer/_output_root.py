"""Output-root discovery and read-only access for the results viewer.

The results viewer consumes a CLI output directory produced by
``python -m phenotypic`` and never mutates it. This module locates the master
measurements parquet, validates the expected layout, and exposes a
small set of helpers used by the rest of the viewer package.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from phenotypic.gui._config import (
    DIR_MEASUREMENTS,
    SANDBOX_GUI_DIRNAME,
)
from phenotypic.gui.results_viewer._filtered_state import (
    KEY_DATASET,
    KEY_IMAGE_FILE,
)
from phenotypic.sdk_ import (
    DIR_OVERLAYS,
    BundleLayout,
    gui_launch_owner_path,
    is_metadata_header,
    paths_fingerprint,
    source_cache_key,
)

logger = logging.getLogger(__name__)

#: The post-applied mirror seeded by ``_seed_measurements``. Preferred
#: over the master archive (``master_measurements.parquet``) for the
#: viewer's display frame because it reflects whatever ``PostMeasurement``
#: ops the user configured.
_EXTERNAL_VIEWER_CACHE_SUBDIR = "viewer_cache"
_SNAPSHOT_READ_ATTEMPTS = 2
# Legacy master column shim. Post category-flip the canonical image-stem column
# is ``str(METADATA.IMAGE_NAME) == "MetadataImage_ImageName"`` (== KEY_IMAGE_FILE).
# Masters written before the flip used the pre-namespace ``Metadata_ImageName``;
# this literal recognizes those old masters so they still load (aliased below to
# the canonical column). Keep the literal — it is a legacy recognizer, not a live
# column name.
_IMAGENAME_COL = "Metadata_ImageName"


class OutputSnapshotChangedError(RuntimeError):
    """Raised when source files cannot be read as one stable revision."""


@dataclass(frozen=True)
class OutputSnapshotDescriptor:
    """Fingerprints that define one coherent Results binding.

    ``processing_fingerprint`` covers immutable processing products used to
    render image content. It is the cache identity and the only fingerprint
    that can invalidate an already-bound tile request.

    ``consumed_state_fingerprint`` covers mutable state read while constructing
    the Results and Analysis sessions: the measurements mirror, pipeline
    recipe, curation labels, custom categories, QC database, and QC review
    state. Changes there are incorporated by an explicit Refresh. They do not
    invalidate image tiles in the current session, including when the current
    GUI session itself wrote the state.

    Attributes:
        processing_fingerprint: Content identity of stable processing outputs.
        consumed_state_fingerprint: Content identity of refresh-owned state.
        captured_at: UTC time at which both fingerprints were verified.
        active_run: Whether a nonterminal GUI launch owner existed when the
            descriptor was captured.
    """

    processing_fingerprint: str
    consumed_state_fingerprint: str
    captured_at: datetime
    active_run: bool


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
        cache_dir: External DZI cache path under the explicitly supplied cache
            root. Discovery only computes this path; tile routes create it on
            an explicit request.
        snapshot: Stable-processing and refresh-consumed fingerprints captured
            around the complete discovery read.
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
    snapshot: OutputSnapshotDescriptor
    pipeline_summary: str | None
    #: Snapshot of ``(dataset, stem)`` pairs that have an overlay PNG on
    #: disk at discovery time. Used as an O(1) replacement for the
    #: :meth:`has_overlay` per-call ``stat`` so picker callbacks don't
    #: hit the filesystem on every render.
    overlay_index: frozenset[tuple[str, str]]

    @classmethod
    def discover(
        cls,
        root: Path,
        *,
        cache_root: Path,
    ) -> OutputRoot:
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
            cache_root: Explicit external directory that owns viewer caches,
                normally ``sandbox_viewer_cache_root(sandbox.root)``. The
                directory is not created during discovery and must not be
                inside the selected output.

        Returns:
            A populated, frozen ``OutputRoot``.

        Raises:
            FileNotFoundError: If neither a deliverables bundle nor a run
                output directory can be located at ``root`` (no
                ``master_measurements.parquet``), or no datasets are found.
            ValueError: If the master DataFrame lacks ``Metadata_ImageName``
                (or the ``Metadata_ImageName`` fallback), or lacks
                ``Metadata_Dataset`` with no ``results/`` to recover it from.
            OutputSnapshotChangedError: If two complete discovery attempts
                both observe source files changing between pre/post checks.
        """
        layout = BundleLayout.detect(Path(root))
        last_change: OutputSnapshotChangedError | None = None
        for attempt in range(_SNAPSHOT_READ_ATTEMPTS):
            try:
                return cls._discover_snapshot(
                    layout,
                    cache_root=cache_root,
                )
            except OutputSnapshotChangedError as exc:
                last_change = exc
            except OSError:
                last_change = OutputSnapshotChangedError(
                    "Output changed while discovery was reading source files."
                )
            if last_change is not None:
                if attempt + 1 == _SNAPSHOT_READ_ATTEMPTS:
                    raise last_change
                logger.info(
                    "Output changed during discovery; retrying one complete read."
                )
        raise AssertionError("snapshot retry loop did not return or raise")

    @classmethod
    def _discover_snapshot(
        cls,
        layout: BundleLayout,
        *,
        cache_root: Path,
    ) -> OutputRoot:
        """Read and verify one complete source snapshot."""
        source_root = (
            layout.deliverables_base
            if layout.output_root is None
            else layout.output_root
        )
        try:
            source_fingerprint, consumed_state_fingerprint = (
                _snapshot_fingerprints(layout, source_root=source_root)
            )
        except OSError as exc:
            raise OutputSnapshotChangedError(
                "Output changed while the pre-read fingerprint was captured."
            ) from exc

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
                "MetadataExperiment_Dataset column or "
                "deliverables/overlays/<dataset>/ dirs."
            )

        master_df = _ensure_required_columns(master_df, layout, datasets)
        clean_master_df = _ensure_required_columns(
            clean_master_df, layout, datasets
        )

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

        pipeline_summary = _read_pipeline_summary(layout.pipeline_config_path)
        overlay_index = _scan_overlay_index(layout, datasets_with_overlays)
        try:
            verified_fingerprint, verified_consumed_state = (
                _snapshot_fingerprints(layout, source_root=source_root)
            )
        except OSError as exc:
            raise OutputSnapshotChangedError(
                "Output changed while the post-read fingerprint was captured."
            ) from exc
        if verified_fingerprint != source_fingerprint:
            raise OutputSnapshotChangedError(
                "Output changed during discovery; refresh after the writer settles."
            )
        if verified_consumed_state != consumed_state_fingerprint:
            raise OutputSnapshotChangedError(
                "Viewer state changed during discovery; refresh after the writer settles."
            )
        cache_dir = _external_cache_dir(
            source_root,
            source_fingerprint=verified_fingerprint,
            cache_root=cache_root,
        )
        snapshot = OutputSnapshotDescriptor(
            processing_fingerprint=verified_fingerprint,
            consumed_state_fingerprint=verified_consumed_state,
            captured_at=datetime.now(timezone.utc),
            active_run=_active_run_snapshot(layout),
        )

        return cls(
            root=source_root,
            layout=layout,
            master_df=master_df,
            clean_master_df=clean_master_df,
            column_value_sets=column_value_sets,
            cache_dir=cache_dir,
            snapshot=snapshot,
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
        """External cache root containing DZI and thumbnail subdirectories."""
        return self.cache_dir.parent

    @property
    def source_fingerprint(self) -> str:
        """Backward-compatible stable processing fingerprint."""
        return self.snapshot.processing_fingerprint

    @property
    def consumed_state_fingerprint(self) -> str:
        """Fingerprint of state incorporated by an explicit Refresh."""
        return self.snapshot.consumed_state_fingerprint

    def snapshot_is_current(self) -> bool:
        """Return whether stable image-processing sources match this binding.

        Mutable GUI-owned state is intentionally excluded. Curation, QC review,
        and Analysis actions must not turn valid tile requests into HTTP 409
        responses within the same session.
        """
        try:
            current = paths_fingerprint(
                _processing_snapshot_paths(self.layout),
                root=self.root,
            )
        except OSError:
            return False
        return current == self.source_fingerprint

    def refresh_state_is_current(self) -> bool:
        """Return whether an explicit Refresh would consume the same state."""
        try:
            current = paths_fingerprint(
                _consumed_state_snapshot_paths(self.layout),
                root=self.root,
            )
        except OSError:
            return False
        return current == self.consumed_state_fingerprint

    def active_run_is_currently_running(self) -> bool:
        """Return whether the captured output still has a nonterminal owner."""
        return _active_run_snapshot(self.layout)

    def mutation_snapshot_is_safe(self) -> bool:
        """Return whether mutations still target this processing generation.

        GUI-owned consumed state can advance within a session. Its individual
        writers retain their own mtime/fingerprint CAS guards.
        """
        return (
            not self.active_run_is_currently_running()
            and self.snapshot_is_current()
        )

    def require_session_snapshot_current(self, *, context: str) -> None:
        """Reject construction that would mix any source generations.

        Args:
            context: Reader-facing construction phase included in the error.

        Raises:
            OutputSnapshotChangedError: If processing products or consumed
                Results and Analysis state differ from discovery.
        """
        if not (
            self.snapshot_is_current()
            and self.refresh_state_is_current()
        ):
            raise OutputSnapshotChangedError(
                f"{context} processing or consumed state changed after "
                "output discovery; "
                "refresh the shared Results and Analysis snapshot."
            )

    def overlay_path(self, dataset: str, stem: str) -> Path:
        """Return the absolute path of an overlay PNG.

        Args:
            dataset: Dataset name (matches ``Metadata_Dataset``).
            stem: Image stem (matches ``Metadata_ImageName`` minus
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

    def has_image_source(self, dataset: str, stem: str) -> bool:
        """Return ``True`` when a crop/DZI source exists for this image."""
        return self.hdf_path(dataset, stem) is not None or self.has_overlay(
            dataset, stem
        )

    def image_pairs(self, df: pl.DataFrame) -> list[tuple[str, str]]:
        """Extract unique ``(dataset, image_stem)`` pairs from a frame.

        Both columns are cast to ``pl.String`` and rows containing
        nulls in either column are dropped before deduplication. The
        result is sorted lexicographically so the picker order is
        deterministic.

        Args:
            df: Any DataFrame that has the ``Metadata_Dataset`` and
                ``Metadata_ImageName`` columns (typically a filtered
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


def _external_cache_dir(
    source_root: Path,
    *,
    source_fingerprint: str,
    cache_root: Path,
) -> Path:
    """Return a pure external DZI cache path for a source snapshot.

    The returned path is not created. This is what keeps
    :meth:`OutputRoot.discover` source-preserving.
    """
    owner = Path(cache_root).resolve()
    source = Path(source_root).resolve()
    if owner == source or owner.is_relative_to(source):
        raise ValueError(
            "Viewer cache root must be external to the selected output: "
            f"cache_root={owner!s}, output={source!s}"
        )
    key = source_cache_key(source_root, source_fingerprint)
    return owner / key / "dzi"


def sandbox_viewer_cache_root(sandbox_root: Path) -> Path:
    """Return the pure viewer-cache owner for a configured GUI sandbox.

    The path is not created. Shell and standalone launchers must pass this
    result (or another explicit external cache owner) to
    :meth:`OutputRoot.discover`.
    """
    return (
        Path(sandbox_root).resolve()
        / SANDBOX_GUI_DIRNAME
        / _EXTERNAL_VIEWER_CACHE_SUBDIR
    )


def user_viewer_cache_root() -> Path:
    """Return the deterministic per-user cache owner for standalone apps.

    The path is pure and is not created here. Linux honors
    ``XDG_CACHE_HOME``; Windows honors ``LOCALAPPDATA``; macOS follows its
    standard ``~/Library/Caches`` location. The fallback remains in the
    user's cache area and never depends on the selected output directory.
    """
    home = Path.home()
    if sys.platform == "win32":
        base = Path(
            os.environ.get(
                "LOCALAPPDATA",
                str(home / "AppData" / "Local"),
            )
        )
    elif sys.platform == "darwin":
        base = home / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(home / ".cache")))
    return base / "phenotypic" / "gui" / _EXTERNAL_VIEWER_CACHE_SUBDIR


def _snapshot_fingerprints(
    layout: BundleLayout,
    *,
    source_root: Path,
) -> tuple[str, str]:
    """Capture stable-processing and refresh-consumed fingerprints."""
    return (
        paths_fingerprint(
            _processing_snapshot_paths(layout),
            root=source_root,
        ),
        paths_fingerprint(
            _consumed_state_snapshot_paths(layout),
            root=source_root,
        ),
    )


def _active_run_snapshot(layout: BundleLayout) -> bool:
    """Return whether discovery captured a nonterminal GUI-owned run.

    This is descriptive snapshot metadata, not a mutation gate. Missing,
    malformed, and historical owner records are treated as inactive.
    """
    if layout.output_root is None:
        return False
    owner_path = gui_launch_owner_path(layout.output_root)
    try:
        payload = json.loads(owner_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    status = payload.get("status")
    return isinstance(status, str) and status in {
        "queued",
        "submitting",
        "running",
        "reconciling",
        "cancelling",
    }


def _processing_snapshot_paths(layout: BundleLayout) -> tuple[Path, ...]:
    """Return stable processing products that define image read binding."""
    paths: list[Path] = [
        layout.master_parquet,
    ]
    overlays_root = layout.deliverables_base / DIR_OVERLAYS
    if overlays_root.is_dir():
        paths.append(overlays_root)
        paths.extend(
            path
            for path in overlays_root.rglob("*")
            if path.is_file() or path.is_dir()
        )
    if layout.results_dir is not None:
        paths.append(layout.results_dir)
        paths.extend(
            path
            for path in layout.results_dir.iterdir()
            if path.is_dir()
        )
        paths.extend(
            path
            for path in layout.results_dir.rglob("*.h5")
            if path.is_file()
        )
        paths.extend(
            path
            for path in layout.results_dir.glob(
                f"*/{DIR_MEASUREMENTS}/*.parquet"
            )
            if path.is_file()
        )
    return tuple(paths)


def _consumed_state_snapshot_paths(layout: BundleLayout) -> tuple[Path, ...]:
    """Return mutable state atomically incorporated by explicit Refresh."""
    return (
        layout.mirror_parquet,
        layout.mirror_csv,
        layout.resolved_pipeline_config_path,
        layout.curation_labels_parquet,
        layout.custom_categories_json,
        layout.qc_duckdb,
        layout.qc_review_state_path,
    )


def _discover_datasets(
    master_df: pl.DataFrame, layout: BundleLayout
) -> list[str]:
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
            for v in master_df.get_column(KEY_DATASET)
            .drop_nulls()
            .unique()
            .to_list()
        )
    overlays_root = layout.deliverables_base / DIR_OVERLAYS
    if overlays_root.is_dir():
        names.update(e.name for e in overlays_root.iterdir() if e.is_dir())
    if layout.results_dir is not None:
        names.update(
            e.name for e in layout.results_dir.iterdir() if e.is_dir()
        )
    return sorted(names)


def _ensure_required_columns(
    df: pl.DataFrame,
    layout: BundleLayout,
    datasets: list[str],
) -> pl.DataFrame:
    """Backfill the dataset and image-stem key columns if missing.

    Real-world masters produced by older runs or by aggregators that
    skip ``include_dataset_column`` may lack one or both of these
    columns. The dataset is recoverable from the on-disk layout
    (``results/<dataset>/measurements/<stem>.parquet``) when a full-run
    ``results/`` is present; the image stem (``KEY_IMAGE_FILE`` ==
    ``MetadataImage_ImageName`` post category-flip) falls back to the
    pre-flip legacy ``Metadata_ImageName`` column (``_IMAGENAME_COL``)
    when present, so masters written before the namespace migration still
    load.

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
            f"Master measurements parquet is missing column {KEY_DATASET!r} and "
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

    Eagerly materialises metadata-family columns (always small, always
    surfaced in the filter sidebar) and defers everything else until a
    callback actually asks for it. Avoids the multi-MB string allocation
    that would otherwise happen at boot for wide masters with hundreds
    of high-cardinality measurement columns.
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df
        self._cache: dict[str, list[str]] = {}
        for column in df.columns:
            if is_metadata_header(column):
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
        logger.debug(
            "Failed to parse %s for pipeline_summary",
            pipeline_json,
            exc_info=True,
        )
        return None
