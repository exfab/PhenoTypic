"""
Output file organization and management for the PhenoTypic CLI.

This module handles all output file creation, directory structure management,
and saving of image layers, measurements, and overlays with comprehensive
error logging to prevent silent data loss.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic._core._image_pipeline import ImagePipeline

from ._cli_types import Dataset
from ._cli_duckdb_agg import duckdb_aggregate

logger = logging.getLogger(__name__)


def _atomic_write(target: Path, write_func: Callable[[str], None]) -> None:
    """Write to *target* atomically via a temp file and ``os.replace``.

    Args:
        target: Final destination path.
        write_func: Callable that writes content to a given file path string.

    Raises:
        Any exception from *write_func* after cleaning up the temp file.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[str] = None
    try:
        fd = tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.stem}_",
            suffix=".tmp",
            delete=False,
        )
        tmp_path = fd.name
        fd.close()
        write_func(tmp_path)
        with open(tmp_path, "r+b") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def join_metadata(df: "pl.DataFrame", metadata_csv: Path) -> "pl.DataFrame":
    """Join external metadata CSV onto a measurements DataFrame.

    Identifies columns common to both the measurements and metadata,
    casts them to ``String`` for a safe join, and performs an inner join
    with the metadata on the left.  Only rows present in both DataFrames
    survive.  Warns if the row count increases (duplicate metadata keys)
    or decreases (measurement rows with no matching metadata).

    Args:
        df: Measurements DataFrame (must have columns to join on).
        metadata_csv: Path to the metadata CSV file.

    Returns:
        DataFrame with metadata columns first, then measurement columns.
    """
    metadata_df = pl.read_csv(metadata_csv)
    common = list(set(df.columns) & set(metadata_df.columns))
    if not common:
        logger.warning(
            "Metadata CSV has no columns in common with measurements — skipping join"
        )
        return df

    logger.info("Joining metadata on columns: %s", common)
    df = df.with_columns(pl.col(col).cast(pl.String) for col in common)
    metadata_df = metadata_df.with_columns(
        pl.col(col).cast(pl.String) for col in common
    )
    n_rows_before = df.height
    n_cols_before = len(df.columns)
    df = metadata_df.join(df, on=common, how="inner")
    n_new_cols = len(df.columns) - n_cols_before
    if df.height > n_rows_before:
        logger.warning(
            "Metadata join increased row count from %d to %d — "
            "metadata CSV likely has duplicate keys on columns %s. "
            "Verify your metadata CSV has unique values on join columns.",
            n_rows_before,
            df.height,
            common,
        )
    n_dropped = n_rows_before - df.height
    if n_dropped > 0:
        logger.warning(
            "Metadata inner join dropped %d/%d measurement rows "
            "with no matching metadata on columns %s",
            n_dropped,
            n_rows_before,
            common,
        )
    logger.info(
        "Metadata join: added %d columns, %d/%d rows matched",
        n_new_cols,
        df.height,
        n_rows_before,
    )
    return df


def _scratch_dest_name(pq: Path) -> str:
    """Build a collision-safe filename for a parquet staged to $SCRATCH."""
    return f"{pq.parent.parent.name}_{pq.name}"


def _stage_to_scratch(parquet_files: List[Path]) -> Optional[Path]:
    """Copy parquet files to $SCRATCH for faster reading.

    Creates a staging directory using SLURM job/task IDs to avoid
    collisions when multiple aggregation tasks run on the same node.

    Args:
        parquet_files: Paths to copy.

    Returns:
        Path to staging directory, or ``None`` if $SCRATCH is unavailable.
    """
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        return None

    scratch_path = Path(scratch)
    if not scratch_path.is_dir():
        return None

    job_id = os.environ.get("SLURM_JOB_ID", "")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    if job_id and task_id:
        suffix = f"{job_id}_{task_id}"
    elif job_id:
        suffix = job_id
    else:
        suffix = str(os.getpid())

    staging_dir = scratch_path / f".phenotypic_stage_{suffix}"
    try:
        staging_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    try:
        for pq in parquet_files:
            shutil.copy2(pq, staging_dir / _scratch_dest_name(pq))
    except Exception:
        _cleanup_scratch(staging_dir)
        return None

    return staging_dir


def _remap_to_scratch(
    path_to_dataset: Dict[Path, str], scratch_dir: Path
) -> Dict[Path, str]:
    """Remap GPFS paths to their $SCRATCH copies.

    Args:
        path_to_dataset: Original path to dataset name mapping.
        scratch_dir: Staging directory on $SCRATCH.

    Returns:
        New mapping with paths pointing to scratch copies.
    """
    remapped: Dict[Path, str] = {}
    for original_path, dataset_name in path_to_dataset.items():
        remapped[scratch_dir / _scratch_dest_name(original_path)] = dataset_name
    return remapped


def _cleanup_scratch(staging_dir: Path) -> None:
    """Remove staging directory with error suppression."""
    try:
        shutil.rmtree(staging_dir)
    except Exception:
        pass


def _collect_feature_headers(
    pipeline: "ImagePipeline",
) -> Dict[str, List[str]]:
    """Map each ``MeasureFeatures`` operation key to its output column headers.

    Reads the pipeline's ``_meas`` dict and, for every measurer, gathers the
    prefixed headers declared by its ``MeasurementInfo`` enum(s). Handles
    both the singular ``_measurement_infoclass`` and plural
    ``_measurement_infoclasses`` (used by :class:`MeasureColor` to cover
    multiple color spaces). Measurers exposing neither attribute are
    skipped with a debug log — their columns will just remain in the
    master file without a dedicated split.

    Args:
        pipeline: An :class:`ImagePipeline` containing the ``MeasureFeatures``
            operations that produced the aggregated master measurements.

    Returns:
        Ordered mapping of ``_meas`` key → list of prefixed header strings
        (e.g. ``"MeasureSize" → ["Size_Area", "Size_IntegratedIntensity"]``).
        Keys with no discoverable headers are omitted.
    """
    headers_by_key: Dict[str, List[str]] = {}
    for key, measurer in pipeline._meas.items():
        infoclasses = []
        single = getattr(measurer, "_measurement_infoclass", None)
        if single is not None:
            infoclasses.append(single)
        plural = getattr(measurer, "_measurement_infoclasses", None)
        if plural:
            infoclasses.extend(plural)

        if not infoclasses:
            logger.debug(
                "Skipping measurer %r in split: no _measurement_infoclass(es) exposed",
                key,
            )
            continue

        headers: List[str] = []
        for info in infoclasses:
            try:
                headers.extend(info.get_headers())
            except Exception:
                logger.debug(
                    "Failed to read headers from %r on measurer %r",
                    info,
                    key,
                    exc_info=True,
                )
        if headers:
            headers_by_key[key] = headers
    return headers_by_key


def _load_pipeline_from_output_dir(
    output_dir: Path,
) -> Optional["ImagePipeline"]:
    """Recover the pipeline used for a run from the files left in *output_dir*.

    Prefers the canonical ``<output>/pipeline.json`` written by
    :func:`_persist_pipeline_to_output_dir` during aggregate finalize.
    Falls back to the legacy lookup via ``processing_state.json`` ->
    ``output_dir / <original-name>.json`` so older outputs created before
    this branch keep working. Returns ``None`` on any failure so callers
    can skip downstream steps cleanly.
    """
    from phenotypic._core._image_pipeline import ImagePipeline

    canonical = output_dir / "pipeline.json"
    if canonical.exists():
        try:
            return ImagePipeline.from_json(canonical)
        except Exception:
            logger.warning(
                "Could not load canonical pipeline.json from %s",
                output_dir,
                exc_info=True,
            )
            # Fall through to legacy lookup.

    state_path = output_dir / "processing_state.json"
    if not state_path.exists():
        return None
    try:
        state_dict = json.loads(state_path.read_text(encoding="utf-8"))
        original = state_dict.get("pipeline_path")
        if not original:
            return None
        candidate = output_dir / Path(original).name
        if not candidate.exists():
            return None
        return ImagePipeline.from_json(candidate)
    except Exception:
        logger.warning(
            "Could not load pipeline from %s for per-feature split",
            output_dir,
            exc_info=True,
        )
        return None


def _persist_pipeline_to_output_dir(
    output_dir: Path,
    pipeline: "ImagePipeline",
) -> Optional[Path]:
    """Atomically write a copy of *pipeline*'s JSON to ``<output>/pipeline.json``.

    The canonical pipeline JSON is the source of truth for analysis recipes
    and reproducibility — it captures filters/model alongside the
    operations/measurements/post chain. The analysis GUI reads from and
    writes back to this file; the CLI seeds it on every aggregate so the
    file always reflects the most recent run.

    Args:
        output_dir: Output root directory (the parent that holds
            ``master_measurements.parquet``).
        pipeline: The pipeline whose configuration to persist.

    Returns:
        Path to the written ``pipeline.json``, or ``None`` if the write
        failed. Failure is logged at WARNING; the caller does not need to
        handle the exception.
    """
    target = output_dir / "pipeline.json"

    def _write(p: str) -> None:
        Path(p).write_text(pipeline.to_json() or "")

    try:
        _atomic_write(target, _write)
        return target
    except Exception:
        logger.warning(
            "Failed to persist canonical pipeline.json to %s",
            output_dir,
            exc_info=True,
        )
        return None


def _emit_analysis_outputs(
    output_dir: Path,
    df: "pl.DataFrame",
    pipeline: "ImagePipeline",
) -> Optional[tuple[Path, int]]:
    """Run ``pipeline.analyze`` and atomic-write ``analysis.{csv,parquet}``.

    No-op when the pipeline has no model configured (``pipeline.get_model()``
    returns ``None``); the auto-emit trigger is the presence of an analysis
    endpoint in the pipeline JSON, not a CLI flag. Failure is non-fatal —
    the master/measurement outputs are not affected.

    Args:
        output_dir: Output root directory.
        df: The frame to analyze (polars). The CLI passes the
            post-applied frame seeded into ``measurements.parquet`` here
            via :func:`finalize_post_master_outputs`, not the clean
            ``master_measurements.parquet`` archive — callers wiring up
            new entry points are responsible for applying post (if any)
            before calling this.
        pipeline: Pipeline whose ``model`` (and optional ``filters``)
            define the analysis chain.

    Returns:
        ``(analysis_parquet_path, row_count)`` on success — the GUI's
        run console reads the row count without re-decoding the parquet.
        ``None`` when no model is configured or the analysis raised;
        failure is logged at WARNING.
    """
    if pipeline.get_model() is None:
        logger.debug(
            "Pipeline has no analysis model configured; skipping "
            "analysis.{csv,parquet}",
        )
        return None

    try:
        fit_pd = pipeline.analyze(df.to_pandas())
        fit_pl = pl.from_pandas(fit_pd)
    except Exception:
        logger.warning(
            "Analysis chain raised; master measurements still written",
            exc_info=True,
        )
        return None

    csv_path = output_dir / "analysis.csv"
    pq_path = output_dir / "analysis.parquet"

    try:
        _atomic_write(csv_path, fit_pl.write_csv)
    except Exception:
        logger.warning("Failed to write analysis.csv")
        return None

    try:
        _atomic_write(
            pq_path,
            lambda p: fit_pl.write_parquet(
                p, compression="zstd", compression_level=3
            ),
        )
    except Exception:
        logger.warning(
            "Failed to write analysis.parquet (analysis.csv was written)"
        )
        return None

    logger.info(
        "Wrote analysis (%d rows x %d cols) to %s",
        fit_pl.height,
        fit_pl.width,
        pq_path.name,
    )
    return pq_path, fit_pl.height


def _apply_post_to_master(
    master_df: "pl.DataFrame",
    pipeline: Optional["ImagePipeline"],
) -> "pl.DataFrame":
    """Run ``pipeline._post`` over a copy of *master_df* and return the result.

    The CLI keeps ``master_measurements.{csv,parquet}`` as a clean, post-free
    archive (because per-image runs now use ``apply_post=False``). This helper
    applies any configured :class:`PostMeasurement` ops to a pandas copy of
    the aggregated master so the post-applied frame can be written into
    ``measurements.{csv,parquet}`` and fed to :meth:`ImagePipeline.analyze`.

    No-op cases — returns *master_df* unchanged:
        * *pipeline* is ``None`` (could not be recovered for the SLURM sentinel).
        * The pipeline has no post ops configured.
        * Any post op raises (logged at WARNING; the master remains authoritative).
    """
    if pipeline is None:
        return master_df
    post_ops = pipeline.get_post()
    if not post_ops:
        return master_df

    try:
        df_pd = master_df.to_pandas()
        for key, post_op in post_ops.items():
            logger.debug("Running post-measurement transform on master: %s", key)
            df_pd = post_op.apply(df_pd)
        return pl.from_pandas(df_pd)
    except Exception:
        logger.warning(
            "Post-measurement transform raised during aggregation; "
            "seeding clean master into measurements.{csv,parquet} instead",
            exc_info=True,
        )
        return master_df


def _seed_measurements(output_dir: Path, master_df: "pl.DataFrame") -> None:
    """Atomically write ``measurements.{csv,parquet}`` as a fresh master copy.

    The GUI's results viewer mutates these mirrors in place when users curate
    colonies; re-runs of the CLI (forward, ``--measure``, ``--recompile``)
    intentionally reset them by calling this helper after the master is
    written. Failures of either write are logged at WARNING and do not raise
    — the master output is preserved as the authoritative source.
    """
    from phenotypic.gui._config import MEASUREMENTS_CSV, MEASUREMENTS_PARQUET

    try:
        _atomic_write(output_dir / MEASUREMENTS_CSV, master_df.write_csv)
    except Exception:
        logger.warning("Failed to seed measurements.csv (master was saved)")

    try:
        _atomic_write(
            output_dir / MEASUREMENTS_PARQUET,
            lambda p: master_df.write_parquet(
                p, compression="zstd", compression_level=3
            ),
        )
    except Exception:
        logger.warning(
            "Failed to seed measurements.parquet (master was saved)"
        )


def finalize_post_master_outputs(
    output_dir: Path,
    master_df: "pl.DataFrame",
    pipeline: Optional["ImagePipeline"],
) -> "pl.DataFrame":
    """Run every CLI side effect that follows a freshly written master file.

    This is the single canonical entry point for the work that happens after
    ``master_measurements.{csv,parquet}`` lands on disk. Every code path that
    writes the master — :func:`aggregate_measurements`, the recompile
    sentinel, future re-aggregators — should call this so the post-applied
    ``measurements.{csv,parquet}`` mirror, the persisted ``pipeline.json``,
    the analysis output, and the per-feature splits stay in lock-step.

    The order is:

    1. Apply ``pipeline._post`` to a copy of *master_df* via
       :func:`_apply_post_to_master`. The resulting ``post_df`` is what the
       GUI viewer/curation layer reads from ``measurements.{csv,parquet}``.
       When *pipeline* is ``None`` or has no post ops, ``post_df`` is the
       clean master unchanged.
    2. :func:`_seed_measurements` writes the post-applied frame.
    3. When *pipeline* is provided: persist ``pipeline.json``, run
       :func:`_emit_analysis_outputs` against ``post_df`` (so analysis sees
       post-applied data), and split the post-applied frame into
       per-feature spreadsheets so users see post-derived columns
       (e.g. ``Metadata_Strain`` from ``ExpandMetadata``) in
       ``measurements_by_feature/<feature>.{csv,parquet}``.

    Failures inside split or analysis are logged at WARNING and never raise;
    the master files remain authoritative regardless of what happens here.

    Args:
        output_dir: Output root that already contains
            ``master_measurements.{csv,parquet}``.
        master_df: The clean (post-free) aggregated master.
        pipeline: Recovered pipeline, or ``None`` when it can't be
            located (the SLURM sentinel may run before any pipeline.json
            is persisted).

    Returns:
        The post-applied frame. Callers that run additional side effects
        downstream (e.g. analysis-plugin dispatch in the recompile
        worker) should pass this to those steps so plugins see the same
        post-applied data the GUI viewer and analysis chain see, rather
        than the clean master. Equal to *master_df* when no post ops
        are configured.
    """
    post_df = _apply_post_to_master(master_df, pipeline)
    _seed_measurements(output_dir, post_df)

    if pipeline is None:
        logger.warning(
            "Pipeline not available — skipping per-feature split, analysis, "
            "and pipeline.json persistence (master files still written to %s)",
            output_dir,
        )
        return post_df

    _persist_pipeline_to_output_dir(output_dir, pipeline)
    _emit_analysis_outputs(output_dir, post_df, pipeline)

    try:
        # Splits operate on the post-applied frame so per-feature
        # spreadsheets match what the GUI viewer reads from
        # measurements.{csv,parquet}. The clean master_measurements.*
        # remains the archival source of truth.
        split_master_by_feature(post_df, output_dir, pipeline)
    except Exception:
        logger.warning(
            "Per-feature measurement split failed (master files still written)",
            exc_info=True,
        )

    return post_df


def split_master_by_feature(
    master_df: "pl.DataFrame",
    output_dir: Path,
    pipeline: "ImagePipeline",
) -> Dict[str, Path]:
    """Write one CSV + Parquet per ``MeasureFeatures`` into *output_dir*.

    Creates ``output_dir/measurements_by_feature/`` and, for every
    measurer key returned by :func:`_collect_feature_headers`, emits a
    spreadsheet containing all non-feature columns (metadata, object
    label, grid info, joined external metadata) alongside only that
    measurer's columns. A feature whose expected columns are entirely
    absent from the master (e.g. its operation failed for every image)
    is skipped.

    Args:
        master_df: Aggregated master measurements.
        output_dir: Base output directory; the ``measurements_by_feature/``
            subdirectory is created if missing.
        pipeline: Pipeline whose ``_meas`` dict defines the split.

    Returns:
        Mapping of ``_meas`` key → path to the emitted CSV. Empty if
        nothing could be split.
    """
    headers_by_key = _collect_feature_headers(pipeline)
    if not headers_by_key:
        logger.info("No MeasureFeatures exposed headers -- skipping split")
        return {}

    all_feature_cols: set[str] = set()
    for cols in headers_by_key.values():
        all_feature_cols.update(cols)

    non_feature_cols = [c for c in master_df.columns if c not in all_feature_cols]

    split_dir = output_dir / "measurements_by_feature"
    split_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}
    for key, headers in headers_by_key.items():
        present = [c for c in headers if c in master_df.columns]
        if not present:
            logger.debug(
                "Skipping split for %r: none of its columns are in master", key
            )
            continue

        subset = master_df.select(non_feature_cols + present)
        csv_path = split_dir / f"{key}.csv"
        pq_path = split_dir / f"{key}.parquet"

        def _write_parquet(path: str, _subset: "pl.DataFrame" = subset) -> None:
            _subset.write_parquet(path, compression="zstd", compression_level=3)

        try:
            _atomic_write(csv_path, subset.write_csv)
        except Exception:
            logger.warning("Failed to write split CSV for %r", key, exc_info=True)
            continue

        try:
            _atomic_write(pq_path, _write_parquet)
        except Exception:
            logger.warning(
                "Failed to write split Parquet for %r (CSV was saved)",
                key,
                exc_info=True,
            )

        written[key] = csv_path
        logger.info(
            "Split %r: %d rows x %d cols -> %s",
            key,
            subset.height,
            subset.width,
            csv_path.name,
        )

    return written


def aggregate_measurements(
    output_dir: Path,
    dataset_names: List[str],
    include_dataset_column: bool = True,
    metadata_csv: Optional[Path] = None,
    pipeline: Optional["ImagePipeline"] = None,
) -> Optional[Path]:
    """Aggregate per-image Parquet files into a master CSV via DuckDB.

    Scans ``results/{name}/measurements/`` for each dataset, looking for
    Parquet (``.parquet``) files.  Prefers pre-aggregated
    ``_dataset_aggregated.parquet`` files when available, falling back to
    individual per-image files.

    Uses :func:`duckdb_aggregate` for efficient in-memory concatenation
    and writes both ``master_measurements.csv`` and
    ``master_measurements.parquet`` to *output_dir* using atomic writes.

    When ``$SCRATCH`` is available (node-local SSD), files are staged
    there first to avoid GPFS metadata overhead.

    Works without an :class:`OutputManager` instance so it can be called
    from the SLURM sentinel job.

    Args:
        output_dir: Base output directory (contains ``results/``).
        dataset_names: Names of datasets to scan.
        include_dataset_column: Whether to insert ``Metadata_Dataset``
            into each file that lacks it.
        metadata_csv: Optional path to an external CSV file. When
            provided, shared columns are used as join keys for an inner
            join with metadata on the left.  Only measurement rows that
            match the metadata are kept.
        pipeline: Optional :class:`ImagePipeline` whose ``MeasureFeatures``
            operations define how to split the aggregated master into
            per-feature sub-spreadsheets.  When omitted, the pipeline is
            recovered from ``processing_state.json`` / the pipeline JSON
            copy in *output_dir*; if it cannot be recovered, the split
            step is skipped and a warning is logged (the master files are
            still written).

    Returns:
        Path to ``master_measurements.csv``, or ``None`` if no
        measurements were found.

    Side effects:
        Delegates the post-master work to
        :func:`finalize_post_master_outputs`, which always seeds
        ``measurements.{csv,parquet}`` and — when a pipeline is
        available — persists ``pipeline.json``, runs the analysis chain
        into ``analysis.{csv,parquet}``, and writes per-feature
        sub-spreadsheets into ``output_dir/measurements_by_feature/``
        (one file per :class:`MeasureFeatures` in ``pipeline._meas``).

        ``master_measurements.{csv,parquet}`` are intentionally a clean
        (pre-post) archive of what the per-image runs measured, while
        ``measurements.{csv,parquet}`` carry the post-applied frame that
        the GUI viewer reads/curates. The two diverge whenever the
        pipeline has any :class:`PostMeasurement` op configured. Split
        and analysis failures never change the return value.
    """
    results_dir = output_dir / "results"

    # -- File discovery ------------------------------------------------
    path_to_dataset: Dict[Path, str] = {}
    for dataset_name in dataset_names:
        meas_dir = results_dir / dataset_name / "measurements"
        if not meas_dir.is_dir():
            continue
        # Prefer pre-aggregated file
        agg_parquet = meas_dir / "_dataset_aggregated.parquet"
        if agg_parquet.exists():
            path_to_dataset[agg_parquet] = dataset_name
        else:
            for pq in sorted(meas_dir.glob("*.parquet")):
                if not pq.name.startswith("_"):
                    path_to_dataset[pq] = dataset_name

    # -- Stage to $SCRATCH ---------------------------------------------
    scratch_dir = _stage_to_scratch(list(path_to_dataset.keys()))
    if scratch_dir is not None:
        active_mapping = _remap_to_scratch(path_to_dataset, scratch_dir)
    else:
        active_mapping = path_to_dataset

    # -- DuckDB aggregation --------------------------------------------
    master_df = duckdb_aggregate(
        file_paths=list(active_mapping.keys()),
        path_to_dataset=active_mapping,
        include_dataset_column=include_dataset_column,
        keep_filename=True,
    )

    if scratch_dir is not None:
        _cleanup_scratch(scratch_dir)

    if master_df is None:
        logger.warning("No valid measurements found for aggregation")
        return None

    # Derive Metadata_ImageFile for the dashboard image viewer, then drop filename.
    if "Metadata_ImageFile" not in master_df.columns and "filename" in master_df.columns:
        master_df = master_df.with_columns(
            pl.col("filename").str.extract(r"([^/\\]+)\.[^.]+$", 1).alias("Metadata_ImageFile")
        )
    if "filename" in master_df.columns:
        master_df = master_df.drop("filename")

    # -- Join metadata -------------------------------------------------
    if metadata_csv is not None:
        try:
            master_df = join_metadata(master_df, metadata_csv)
        except Exception as e:
            logger.warning("Failed to join metadata CSV: %s: %s", type(e).__name__, e)

    # -- Write master CSV and Parquet ----------------------------------
    master_csv_path = output_dir / "master_measurements.csv"
    master_pq_path = output_dir / "master_measurements.parquet"

    try:
        _atomic_write(master_csv_path, master_df.write_csv)
    except Exception:
        logger.error("Failed to save master CSV")
        return None

    try:
        _atomic_write(
            master_pq_path,
            lambda p: master_df.write_parquet(
                p, compression="zstd", compression_level=3
            ),
        )
    except Exception:
        logger.warning("Failed to save master Parquet (CSV was saved)")

    logger.info(
        "Aggregated %d rows x %d cols into %s",
        master_df.height,
        master_df.width,
        master_csv_path.name,
    )

    resolved_pipeline = pipeline if pipeline is not None else _load_pipeline_from_output_dir(output_dir)
    finalize_post_master_outputs(output_dir, master_df, resolved_pipeline)

    return master_csv_path


class OutputManager:
    """
    Manages all output file creation and organization for CLI processing.

    Handles directory structure creation, output path resolution, and saving
    of measurements, overlays, and optional image layers (rgb, gray, masks, etc.).
    """

    def __init__(
        self,
        base_dir: Path,
        save_layers: Dict[str, bool],
        extensions: Dict[str, str],
        include_dataset_column: bool = True,
        overlay_alpha: float = 0.3,
        save_overlays: bool = True,
    ):
        """
        Initialize OutputManager.

        Args:
            base_dir: Base output directory for all results
            save_layers: Which layers to save. For forward runs produced by
                :meth:`from_config`, the only active key is ``"hdf"``; any
                legacy keys (``"rgb"``, ``"gray"``, ``"detect_mat"``,
                ``"objmap"``) are accepted but ignored for directory creation.
            extensions: File extensions for each layer {"hdf": ".h5", ...}
            include_dataset_column: Whether to add Metadata_Dataset column to measurements (default: True)
            overlay_alpha: Alpha transparency for label overlay (0.0-1.0, default: 0.3)
            save_overlays: If True, ``create_structure`` provisions an
                ``overlays/`` directory per dataset and workers will save
                a PNG overlay per image. Defaults to True; set False only
                for ``--measure`` reruns that should not regenerate overlays.
        """
        self.base_dir = Path(base_dir)
        self.save_layers = save_layers
        self.extensions = extensions
        self.include_dataset_column = include_dataset_column
        self.overlay_alpha = overlay_alpha
        self.save_overlays = save_overlays

        # Results directory for dataset outputs (images, measurements, overlays)
        self.results_dir = self.base_dir / "results"

        # Logs directory (always at root level)
        self.logs_dir = self.base_dir / "logs"

    @classmethod
    def from_config(
        cls,
        base_dir: Path,
        ext: str,
        include_dataset_column: bool = True,
        overlay_alpha: float = 0.3,
        save_overlays: bool = True,
    ) -> "OutputManager":
        """Create an OutputManager configured for HDF-centric forward runs.

        Forward runs now write a single ``.h5`` per image under
        ``results/<ds>/hdf/`` plus the parquet measurements and an
        overlay PNG. The ``ext`` argument is retained for backward
        compatibility with callers that still construct overlay
        filenames via :meth:`get_output_path`.

        Args:
            base_dir: Base output directory.
            ext: Extension retained for overlay PNG / legacy call sites;
                no longer the forward-run image-layer switch.
            include_dataset_column: Add Metadata_Dataset to measurements.
            overlay_alpha: Alpha for overlay compositing.
            save_overlays: If True (default), provision ``overlays/`` per
                dataset and save an overlay per image. Pass False only
                for ``--measure`` reruns that should not regenerate
                overlays.
        """
        return cls(
            base_dir=base_dir,
            save_layers={"hdf": True},
            extensions={"hdf": ".h5"},
            include_dataset_column=include_dataset_column,
            overlay_alpha=overlay_alpha,
            save_overlays=save_overlays,
        )

    def create_structure(self, datasets: List[Dataset]) -> None:
        """
        Create complete output directory structure.

        Always creates dataset-first structure with each dataset in its own
        folder.  Forward runs provision ``measurements/``, ``hdf/``, and
        ``overlays/`` for every dataset; ``overlays/`` is skipped only
        when :attr:`save_overlays` is False (e.g. ``--measure`` reruns).

        Args:
            datasets: List of datasets to create directories for
        """
        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create results directory for dataset outputs
        self.results_dir.mkdir(exist_ok=True)

        # Create logs directory at root level
        self.logs_dir.mkdir(exist_ok=True)
        (self.logs_dir / "slurm").mkdir(exist_ok=True)

        # Create dataset folders with subdirectories under results/
        for dataset in datasets:
            dataset_dir = self.results_dir / dataset.name
            dataset_dir.mkdir(exist_ok=True)

            (dataset_dir / "measurements").mkdir(exist_ok=True)
            (dataset_dir / "hdf").mkdir(exist_ok=True)
            if self.save_overlays:
                (dataset_dir / "overlays").mkdir(exist_ok=True)

    def get_output_path(
        self,
        dataset_name: str,
        layer: str,
        image_stem: str
    ) -> Path:
        """
        Get the output path for a specific file.

        Args:
            dataset_name: Dataset name (e.g., "single_image", directory name, or subdirectory name)
            layer: Layer type ("measurements", "overlays", "hdf", or a
                legacy per-layer key declared in ``save_layers``).
            image_stem: Image filename without extension

        Returns:
            Complete output path for the file
        """
        # Determine extension
        if layer == "measurements":
            ext = ".parquet"
        elif layer == "overlays":
            ext = ".png"
        elif layer == "hdf":
            ext = self.extensions.get("hdf", ".h5")
        else:
            if not self.save_layers.get(layer):
                raise ValueError(f"Layer '{layer}' is not enabled")
            ext = self.extensions.get(layer, ".png")

        # Always use: results/dataset/layer/file
        return self.results_dir / dataset_name / layer / f"{image_stem}{ext}"

    def save_measurements(
        self,
        measurements: pd.DataFrame,
        dataset_name: str,
        image_stem: str
    ) -> Path:
        """
        Save measurements as a Parquet file for a single image.

        Args:
            measurements: DataFrame with measurement data
            dataset_name: Dataset name
            image_stem: Image filename without extension

        Returns:
            Path where measurements were saved
        """
        # Add dataset column if requested
        if self.include_dataset_column and "Metadata_Dataset" not in measurements.columns:
            measurements = measurements.copy()
            measurements.insert(0, "Metadata_Dataset", dataset_name)

        output_path = self.get_output_path(dataset_name, "measurements", image_stem)
        parquet_df = pl.from_pandas(measurements)

        _atomic_write(
            output_path,
            lambda p: parquet_df.write_parquet(
                p, compression="zstd", compression_level=3
            ),
        )

        return output_path

    def save_overlay(
        self,
        image: Image,
        dataset_name: str,
        image_stem: str
    ) -> Path:
        """
        Save overlay visualization for a single image.

        Uses full-resolution save_overlay() from the image accessor.
        Prefers RGB overlay if available, falls back to grayscale.

        Args:
            image: Image object with processing results
            dataset_name: Dataset name
            image_stem: Image filename without extension

        Returns:
            Path where overlay was saved
        """
        output_path = self.get_output_path(dataset_name, "overlays", image_stem)

        if not image.rgb.isempty():
            image.rgb.save_overlay(
                filepath=output_path,
                overlay_alpha=self.overlay_alpha
            )
        else:
            image.gray.save_overlay(
                filepath=output_path,
                overlay_alpha=self.overlay_alpha
            )

        return output_path

    def _save_layer_safely(
        self,
        layer_name: str,
        dataset_name: str,
        image_stem: str,
        save_func: Callable[[Path], None],
    ) -> Optional[Path]:
        """Safely save an image layer with error logging.

        Args:
            layer_name: Name of layer (e.g., "rgb", "gray").
            dataset_name: Dataset name.
            image_stem: Image filename stem.
            save_func: Function to call for saving (takes path as argument).

        Returns:
            Path if successful, None if failed.
        """
        try:
            path = self.get_output_path(dataset_name, layer_name, image_stem)
            save_func(path)
            return path
        except Exception as e:
            logger.warning(
                "Failed to save %s for %s/%s: %s: %s",
                layer_name,
                dataset_name,
                image_stem,
                type(e).__name__,
                e,
            )
            return None

    def save_image_hdf(
        self,
        image: "Image",
        dataset_name: str,
        image_stem: str,
    ) -> Optional[Path]:
        """Save processed image as HDF5 under ``results/<ds>/hdf/``.

        Writes atomically: ``image.save2hdf5`` writes to a temp file in the
        same directory, then :func:`os.replace` promotes it to the final
        path.  This mirrors :func:`_atomic_write`'s spirit, but h5py needs
        to own the file handle so we cannot feed it a buffer.

        Args:
            image: Image object with processing results.
            dataset_name: Dataset name.
            image_stem: Image filename without extension.

        Returns:
            Path where HDF5 was saved, or ``None`` if saving failed.
        """
        final_path = self.get_output_path(dataset_name, "hdf", image_stem)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = final_path.with_name(f".{final_path.name}.{os.getpid()}.part")
        try:
            image.save2hdf5(tmp_path)
            os.replace(tmp_path, final_path)
            logger.info(
                "Saved HDF5 for %s/%s", dataset_name, image_stem
            )
            return final_path
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            logger.warning(
                "Failed to save HDF5 for %s/%s: %s: %s",
                dataset_name,
                image_stem,
                type(e).__name__,
                e,
            )
            return None

    def save_image_layers(
        self,
        image: Image,
        dataset_name: str,
        image_stem: str,
    ) -> Dict[str, Path]:
        """Save all requested image layers (rgb, gray, detect_mat, objmap).

        .. deprecated::
            Use :meth:`save_image_hdf` instead. Forward runs now persist a
            single HDF5 per image under ``results/<ds>/hdf/``. This shim
            remains for downstream scripts that still call the old per-layer
            writer; it will be removed in a future release.

        Args:
            image: Image object with processing results.
            dataset_name: Dataset name.
            image_stem: Image filename without extension.

        Returns:
            Dictionary mapping layer names to saved paths (only successful saves).
        """
        warnings.warn(
            "save_image_layers is deprecated; use save_image_hdf instead",
            DeprecationWarning,
            stacklevel=2,
        )
        saved_paths: Dict[str, Path] = {}

        layer_accessors = {
            "rgb": image.rgb,
            "gray": image.gray,
            "detect_mat": image.detect_mat,
            "objmap": image.objmap,
        }

        for layer_name, accessor in layer_accessors.items():
            if not self.save_layers.get(layer_name) or accessor.isempty():
                continue
            path = self._save_layer_safely(
                layer_name,
                dataset_name,
                image_stem,
                lambda p, acc=accessor: acc.imsave(filepath=p),
            )
            if path:
                saved_paths[layer_name] = path

        return saved_paths

    def aggregate_master_csv(
        self,
        datasets: List[Dataset],
        metadata_csv: Optional[Path] = None,
        pipeline: Optional["ImagePipeline"] = None,
    ) -> Optional[Path]:
        """Aggregate per-image measurement Parquet files into master CSV.

        Args:
            datasets: List of all datasets processed.
            metadata_csv: Optional path to external CSV for inner-join
                on shared columns.
            pipeline: Optional in-memory pipeline used to split the
                aggregated master into per-feature sub-spreadsheets. See
                :func:`aggregate_measurements` for fallback behavior.

        Returns:
            Path to master_measurements.csv, or None if no measurements found.
        """
        return aggregate_measurements(
            output_dir=self.base_dir,
            dataset_names=[ds.name for ds in datasets],
            include_dataset_column=self.include_dataset_column,
            metadata_csv=metadata_csv,
            pipeline=pipeline,
        )
