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
import uuid
import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    TYPE_CHECKING,
)

import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic._core._image_pipeline import ImagePipeline
    from phenotypic.plotting import AnalysisResult

from ._cli_types import Dataset
from ._measurement_sources import (
    add_metadata_image_name_from_filename,
    discover_measurement_sources,
    measurement_sources_by_path,
)
from ._cli_parquet_agg import SOURCE_PATH_COLUMN, aggregate_parquet_files
from phenotypic.schema import EXPERIMENT_METADATA, METADATA, METADATA_MATCH
from phenotypic.util import split_measurements
from phenotypic.sdk_ import (
    analysis_manifest_path,
    DIR_RESULTS,
    DIR_MEASUREMENTS,
    DIR_HDF,
    PARQUET_WRITE_OPTIONS,
    EnvVar,
    atomic_write_with_writer,
    dataset_overlays_dir,
    deliverables_dir,
    master_measurements_csv_path,
    master_measurements_parquet_path,
    measurements_by_feature_dir,
    measurements_csv_path,
    measurements_parquet_path,
    metadata_csv_deliverable_path,
    logs_dir,
    pipeline_json_path,
    pipeline_publication_lock,
    qc_duckdb_path,
    resolve_pipeline_config_path,
    resolve_processing_state_path,
)

logger = logging.getLogger(__name__)


#: Pre-join marker column proving a row came from the measurements frame. It is
#: the only null that is *structurally* impossible on a matched row, so it — not
#: a null in some measurement column — is what identifies a phantom. A real row
#: may legitimately carry a null measurement, and ``join_metadata`` is generic
#: over frame shape (callers pass frames without ``Object_Label``). Dropped
#: before returning.
_MEAS_PRESENT_SENTINEL: Final = "__phenotypic_measurement_present"


def join_metadata(
    df: "pl.DataFrame",
    metadata_csv: Path,
    *,
    how: Literal["inner", "left"] = "inner",
) -> "pl.DataFrame":
    """Join external metadata CSV onto a measurements DataFrame.

    Identifies columns common to both the measurements and metadata, casts them
    to ``String`` for a safe join, and joins with the metadata frame on the
    **left**.

    ``how="inner"`` (the default) keeps only rows present in both frames.
    ``how="left"`` keeps every metadata row: a metadata key that matched no
    measured object survives as a **phantom row** — its join-key and metadata
    values are carried, every measurement/info column is null, and
    :attr:`~phenotypic.schema.METADATA_MATCH.METADATA_ONLY` (``QC_MetadataOnly``)
    is ``True``. Absence of a colony is data: a strain that failed to grow, or
    that detection missed, is exactly what the user needs to see. The flag column
    is emitted **only** under ``how="left"``, so the inner callers' output schema
    is unchanged.

    Note that a left join is asymmetric by design: it keeps *metadata*-unmatched
    rows but still drops *measurement*-unmatched rows, because measurements are
    the right frame.

    Bare (un-prefixed) metadata *attribute* columns are prefixed via
    :func:`~phenotypic.sdk_.ensure_metadata_prefix` (``Strain`` ->
    ``MetadataGenetic_Strain``; unknown ``Foo`` -> ``Metadata_Foo``) so the
    downstream column orderer recognizes them as front metadata, matching the
    pandas ``insert_metadata`` path. Join-key columns, and any supplied column
    that is already a canonical schema header (a measurement/info column such as
    ``Grid_RowNum`` / ``Shape_Area``, or an already-prefixed ``Metadata*`` header),
    keep their raw names — no ``Metadata_`` prefix is appended to them.

    Args:
        df: Measurements DataFrame (must have columns to join on).
        metadata_csv: Path to the metadata CSV file.
        how: Join strategy. ``"inner"`` (default) drops metadata rows that
            matched no measured object; ``"left"`` keeps them as phantom rows
            and adds the ``QC_MetadataOnly`` flag column. Keyword-only.

    Returns:
        DataFrame with metadata columns first, then measurement columns (plus
        ``QC_MetadataOnly`` when ``how="left"``). Row order follows the metadata
        frame.
    """
    metadata_df = pl.read_csv(metadata_csv)
    # sorted() — a set's iteration order is not stable across runs, which would
    # make the logged column order nondeterministic.
    common = sorted(set(df.columns) & set(metadata_df.columns))
    if not common:
        logger.warning(
            "Metadata CSV has no columns in common with measurements — skipping join"
        )
        return df

    logger.info("Joining metadata on columns: %s", common)
    # Prefix bare metadata attribute columns (non-join-key) so the mirror's
    # column orderer recognizes them as front metadata (``Strain`` ->
    # ``MetadataGenetic_Strain``; unknown ``Foo`` -> ``Metadata_Foo``), matching
    # the pandas ``insert_metadata`` path. Deliberately left alone:
    #   - join keys (they must keep their raw names so the join still matches);
    #   - any supplied column that already has a canonical schema identity — a
    #     measurement/info header (``Grid_RowNum``, ``Shape_Area``, ...) or an
    #     already-prefixed ``Metadata*`` header — which must NOT get a
    #     ``Metadata_`` prefix appended;
    #   - a prefix that would collide with an existing column.
    from phenotypic.schema import header_to_module
    from phenotypic.sdk_ import ensure_metadata_prefix

    known_headers = header_to_module()
    taken = set(df.columns) | set(metadata_df.columns)
    rename_map: dict[str, str] = {}
    for col in metadata_df.columns:
        if col in common or col in known_headers:
            continue
        prefixed = ensure_metadata_prefix(col)
        if prefixed != col and prefixed not in taken:
            rename_map[col] = prefixed
            taken.add(prefixed)
    if rename_map:
        logger.info("Prefixing bare metadata columns: %s", rename_map)
        metadata_df = metadata_df.rename(rename_map)

    df = df.with_columns(pl.col(col).cast(pl.String) for col in common)
    metadata_df = metadata_df.with_columns(
        pl.col(col).cast(pl.String) for col in common
    )
    n_rows_before = df.height
    n_cols_before = len(df.columns)
    flag = str(METADATA_MATCH.METADATA_ONLY)
    if how == "left":
        df = df.with_columns(pl.lit(True).alias(_MEAS_PRESENT_SENTINEL))
    out = metadata_df.join(df, on=common, how=how, maintain_order="left")
    if how == "left":
        out = out.with_columns(
            pl.col(_MEAS_PRESENT_SENTINEL).is_null().alias(flag)
        ).drop(_MEAS_PRESENT_SENTINEL)
    n_new_cols = len(out.columns) - n_cols_before

    # Three independently computed signals. Height arithmetic can no longer
    # infer any of them: under a left join the output height is driven by the
    # metadata frame, so a row-count delta conflates duplicates, drops, and
    # phantoms.
    #
    # (1) Duplicate metadata keys — asked of the metadata frame directly. Fan-out
    #     on the measurement side (one key -> many colonies) is the normal case
    #     and must never warn.
    n_unique_keys = metadata_df.n_unique(subset=common)
    if n_unique_keys < metadata_df.height:
        logger.warning(
            "Metadata CSV has duplicate keys on columns %s (%d unique keys "
            "across %d rows) — each duplicate fans the join out into extra "
            "rows. Verify your metadata CSV has unique values on join columns.",
            common,
            n_unique_keys,
            metadata_df.height,
        )

    # (2) Measurement rows that matched no metadata — real under both modes,
    #     since measurements are the right frame.
    n_dropped = df.join(
        metadata_df.select(common).unique(), on=common, how="anti"
    ).height
    if n_dropped > 0:
        logger.warning(
            "Metadata %s join dropped %d/%d measurement rows "
            "with no matching metadata on columns %s",
            how,
            n_dropped,
            n_rows_before,
            common,
        )

    # (3) Metadata rows that matched nothing — the point of the left join.
    if how == "left":
        n_phantom = int(out[flag].sum())
        if n_phantom:
            logger.warning(
                "%d/%d metadata rows matched no measured object on columns "
                "%s — these samples were not detected. They are kept in "
                "measurements.{csv,parquet} with null measurements and "
                "%s = true.",
                n_phantom,
                metadata_df.height,
                common,
                flag,
            )

    logger.info(
        "Metadata join: added %d columns, %d rows in (%d measurement rows, "
        "%d metadata rows)",
        n_new_cols,
        out.height,
        n_rows_before,
        metadata_df.height,
    )
    return out


def _scratch_dest_name(pq: Path) -> str:
    """Build a collision-safe filename for a parquet staged to $SCRATCH."""
    return f"{pq.parent.parent.name}_{pq.name}"


def _stage_to_scratch(parquet_files: List[Path]) -> Optional[Path]:
    """Copy parquet files to $SCRATCH for faster reading.

    Creates a per-process staging directory keyed on the SLURM job/task IDs
    *and the process id*, so concurrent aggregation processes that share a
    SLURM job id (e.g. pytest-xdist workers, or two aggregations launched on
    one node) never stage into — and ``shutil.rmtree`` on cleanup — the same
    directory. The PID alone guarantees uniqueness; the job/task ids are
    retained in the name for debuggability.

    Args:
        parquet_files: Paths to copy.

    Returns:
        Path to staging directory, or ``None`` if $SCRATCH is unavailable.
    """
    scratch = os.environ.get(EnvVar.SCRATCH)
    if not scratch:
        return None

    scratch_path = Path(scratch)
    if not scratch_path.is_dir():
        return None

    job_id = os.environ.get(EnvVar.SLURM_JOB_ID, "")
    task_id = os.environ.get(EnvVar.SLURM_ARRAY_TASK_ID, "")
    suffix = "_".join(
        part for part in (job_id, task_id, str(os.getpid())) if part
    )

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
        remapped[scratch_dir / _scratch_dest_name(original_path)] = (
            dataset_name
        )
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

    Prefers the canonical typed pipeline config written by
    :func:`_persist_pipeline_to_output_dir` during aggregate finalize.
    Falls back to the legacy lookup via ``processing_state.json`` ->
    ``output_dir / <original-name>.json`` so older outputs created before
    this branch keep working. Returns ``None`` on any failure so callers
    can skip downstream steps cleanly.
    """
    from phenotypic._core._image_pipeline import ImagePipeline

    canonical = resolve_pipeline_config_path(output_dir)
    if canonical.exists():
        try:
            return ImagePipeline.from_json(canonical)
        except Exception:
            logger.warning(
                "Could not load canonical pipeline config from %s",
                output_dir,
                exc_info=True,
            )
            # Fall through to legacy lookup.

    state_path = resolve_processing_state_path(output_dir)
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
    target = pipeline_json_path(output_dir)

    def _write(p: str) -> None:
        Path(p).write_text(pipeline.to_json() or "")

    try:
        with pipeline_publication_lock(target):
            atomic_write_with_writer(target, _write)
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
    *,
    deliverables_base: Optional[Path] = None,
    publication_guard: Optional[Callable[[], bool]] = None,
) -> Optional["AnalysisResult"]:
    """Run ``pipeline.analyze`` and publish class-named analysis artifacts.

    No-op when the pipeline has no model configured (``pipeline.get_model()``
    returns ``None``); the auto-emit trigger is the presence of an analysis
    endpoint in the pipeline JSON, not a CLI flag. Failure is non-fatal —
    the master/measurement outputs are not affected.

    Args:
        output_dir: Output root directory. Used to resolve the
            named analysis write location via
            ``deliverables_dir(output_dir)`` UNLESS ``deliverables_base`` is
            supplied.
        df: The frame to analyze (polars). The CLI passes the
            post-applied frame seeded into ``measurements.parquet`` here
            via :func:`finalize_post_master_outputs`, not the clean
            ``master_measurements.parquet`` archive — callers wiring up
            new entry points are responsible for applying post (if any)
            before calling this.
        pipeline: Pipeline whose ``model`` (and optional ``filters``)
            define the analysis chain.
        deliverables_base: When given, write named analysis artifacts
            directly into this folder instead of ``deliverables_dir(output_dir)``.
            The GUI analysis sub-app passes ``layout.deliverables_base`` here so a
            standalone deliverables bundle (where the viewer ``root`` IS the
            deliverables folder) does not double-join ``deliverables/``.
        publication_guard: Optional GUI compare-and-set guard rechecked inside
            the artifact lock after computation, immediately before canonical
            replacement, and after the manifest commit boundary. CLI callers
            omit it.

    Returns:
        Runtime result retaining the exact analyzed table and producer on
        success. ``None`` when no model is configured or publication failed.
    """
    model = pipeline.get_model()
    if model is None:
        logger.debug(
            "Pipeline has no analysis model configured; skipping "
            "named analysis artifacts",
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

    from phenotypic.plotting import (
        AnalysisManifest,
        AnalysisManifestEntry,
        AnalysisResult,
        file_sha256,
        publish_analysis_manifest_entry,
        read_analysis_manifest,
        recover_analysis_publication,
        write_analysis_publication_journal,
    )

    analysis_id = type(model).__name__
    base = (
        Path(deliverables_base)
        if deliverables_base is not None
        else deliverables_dir(output_dir)
    )
    from phenotypic.plotting._analysis_artifacts import (
        _analysis_publication_paths,
        write_analysis_manifest,
    )
    from phenotypic.sdk_._file_locking import exclusive_path_lock

    transaction = _analysis_publication_paths(
        base, analysis_id, uuid.uuid4().hex
    )
    paths = transaction.canonical
    with exclusive_path_lock(transaction.lock):
        if not _analysis_publication_guard_allows(publication_guard):
            logger.warning(
                "Analysis publication blocked because its source snapshot "
                "or active-owner state changed during computation."
            )
            return None
        staged_csv = transaction.staged_csv
        staged_parquet = transaction.staged_parquet
        backup_csv = transaction.backup_csv
        backup_parquet = transaction.backup_parquet
        try:
            recover_analysis_publication(base)
            previous_manifest = read_analysis_manifest(base)
            previous_manifest_existed = analysis_manifest_path(base).exists()
            fit_pl.write_csv(staged_csv)
            fit_pl.write_parquet(staged_parquet, **PARQUET_WRITE_OPTIONS)
            entry = AnalysisManifestEntry(
                producer_class=type(model).__name__,
                csv=paths.csv.name,
                parquet=paths.parquet.name,
                rows=len(fit_pd),
                columns=tuple(fit_pd.columns),
                csv_sha256=file_sha256(staged_csv),
                parquet_sha256=file_sha256(staged_parquet),
            )
            write_analysis_publication_journal(
                base,
                analysis_id=analysis_id,
                token=transaction.token,
                old_csv_exists=paths.csv.exists(),
                old_parquet_exists=paths.parquet.exists(),
                entry=entry,
            )
            if not _analysis_publication_guard_allows(publication_guard):
                logger.warning(
                    "Analysis publication blocked because its source snapshot "
                    "or active-owner state changed before replacement."
                )
                recover_analysis_publication(base)
                return None
            if paths.csv.exists():
                os.replace(paths.csv, backup_csv)
            if paths.parquet.exists():
                os.replace(paths.parquet, backup_parquet)
            os.replace(staged_csv, paths.csv)
            os.replace(staged_parquet, paths.parquet)
            publish_analysis_manifest_entry(base, analysis_id, entry)
            if not _analysis_publication_guard_allows(publication_guard):
                logger.warning(
                    "Analysis publication blocked because its source snapshot "
                    "or active-owner state changed at the commit boundary."
                )
                with exclusive_path_lock(base / ".analysis-manifest.lock"):
                    current_manifest = read_analysis_manifest(base)
                    if (
                        current_manifest is None
                        or current_manifest.analyses.get(analysis_id) != entry
                    ):
                        raise RuntimeError(
                            "analysis manifest changed before guarded rollback"
                        )
                    restored_analyses = dict(current_manifest.analyses)
                    previous_entry = (
                        previous_manifest.analyses.get(analysis_id)
                        if previous_manifest is not None
                        else None
                    )
                    if previous_entry is None:
                        restored_analyses.pop(analysis_id, None)
                    else:
                        restored_analyses[analysis_id] = previous_entry
                    if not previous_manifest_existed and not restored_analyses:
                        analysis_manifest_path(base).unlink(missing_ok=True)
                    else:
                        write_analysis_manifest(
                            base,
                            AnalysisManifest(analyses=restored_analyses),
                        )
                recover_analysis_publication(base)
                return None
            recover_analysis_publication(base)
        except Exception:
            try:
                recover_analysis_publication(base)
            except Exception:  # noqa: BLE001 - preserve journal for next recovery
                logger.error(
                    "Could not recover interrupted analysis publication for %s",
                    analysis_id,
                    exc_info=True,
                )
            logger.warning(
                "Failed to publish analysis artifacts for %s; restored the "
                "previous generation",
                analysis_id,
                exc_info=True,
            )
            return None
        finally:
            for temporary in (staged_csv, staged_parquet):
                temporary.unlink(missing_ok=True)

    logger.info(
        "Wrote analysis (%d rows x %d cols) to %s",
        fit_pl.height,
        fit_pl.width,
        paths.parquet.name,
    )
    return AnalysisResult(
        analysis_id=analysis_id,
        table=fit_pd,
        producer=model,
        artifacts=paths,
        manifest_entry=entry,
    )


def _analysis_publication_guard_allows(
    publication_guard: Optional[Callable[[], bool]],
) -> bool:
    """Treat a failing GUI compare-and-set guard as a publication conflict."""
    if publication_guard is None:
        return True
    try:
        return publication_guard()
    except Exception:
        logger.warning(
            "Analysis publication guard failed; blocking publication.",
            exc_info=True,
        )
        return False


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
            logger.debug(
                "Running post-measurement transform on master: %s", key
            )
            df_pd = post_op.apply(df_pd)
        out = pl.from_pandas(df_pd)
        # ``to_pandas()`` promotes an Int64 column that carries nulls (every
        # measurement column on a --metadata phantom row) to float64, and
        # ``from_pandas`` brings it back as Float64 — silently changing the
        # mirror's dtypes. Restore the master's integer dtypes. Its own
        # try/except: a restore failure must not fall into the outer handler,
        # which would discard all post output.
        try:
            casts = [
                pl.col(name).cast(dt)
                for name, dt in master_df.schema.items()
                if name in out.columns
                and dt.is_integer()
                and out.schema[name].is_float()
            ]
            if casts:
                out = out.with_columns(casts)
        except Exception:
            logger.warning(
                "Failed to restore integer dtypes after the post-measurement "
                "pandas round-trip; post output is kept as-is",
                exc_info=True,
            )
        return out
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
    colonies; re-runs of the CLI (forward, measure mode, recompile mode)
    intentionally reset them by calling this helper after the master is
    written. Failures of either write are logged at WARNING and do not raise
    — the master output is preserved as the authoritative source.
    """
    try:
        atomic_write_with_writer(
            measurements_csv_path(output_dir), master_df.write_csv
        )
    except Exception:
        logger.warning("Failed to seed measurements.csv (master was saved)")

    try:
        atomic_write_with_writer(
            measurements_parquet_path(output_dir),
            lambda p: master_df.write_parquet(p, **PARQUET_WRITE_OPTIONS),
        )
    except Exception:
        logger.warning(
            "Failed to seed measurements.parquet (master was saved)"
        )


#: Post-applied mirror columns that source the REMBI manifest's per-image
#: ``image_data`` block, mapped to the bare keys
#: :func:`phenotypic.sdk_._rembi_manifest.build_rembi_manifest` reads. Only the
#: columns the mirror carries today are listed; ``MetadataImage_UUID`` is private
#: and absent from the measurement frame, so it is naturally omitted.
_REMBI_IMAGE_META_COLUMNS: Final[dict[str, str]] = {
    str(METADATA.IMAGE_NAME): "ImageName",
    str(METADATA.BIT_DEPTH): "BitDepth",
    str(METADATA.IMAGE_TYPE): "ImageType",
    str(METADATA.UUID): "UUID",
}


def _image_metadata_from_mirror(mirror_df: "pl.DataFrame") -> list[dict]:
    """Distinct per-image metadata rows for the REMBI manifest's image_data.

    Folds the per-colony post-applied mirror down to one dict per distinct
    image, reading whichever of :data:`_REMBI_IMAGE_META_COLUMNS` the frame
    carries and mapping each to the manifest builder's bare key. Returns an
    empty list when none of those columns are present.

    ``--metadata`` phantom rows are excluded — they describe a sample that was
    never imaged, and REMBI is a publication manifest, so folding one in
    fabricates an image record that does not exist (an ``n_images`` of 3 for two
    captured plates).

    Two independent filters are needed, because a phantom's shape depends on the
    join key:

    * ``QC_MetadataOnly`` — the authoritative signal, and the only one that works
      for the **documented** per-image join. When the CSV joins on
      ``MetadataImage_ImageName``, the phantom *keeps* that key, so its image
      name is emphatically **not** null and a name-based filter sees nothing.
    * a null image name — covers the per-colony join (on ``Grid_RowNum`` /
      ``Grid_ColNum``), where ``MetadataImage_ImageName`` comes from the
      measurement side and so is null on a phantom.

    Reading the flag is correct *here* even though analysis/post ops must never
    branch on it: this is CLI-internal code, downstream of the join that creates
    it, and it degrades to the name filter alone when the column is absent.
    Filtering the name column specifically (rather than ``drop_nulls()``) keeps
    legitimate images that merely lack an optional field such as ``BitDepth``.
    """
    present = [c for c in _REMBI_IMAGE_META_COLUMNS if c in mirror_df.columns]
    if not present:
        return []
    real = mirror_df
    flag = str(METADATA_MATCH.METADATA_ONLY)
    if flag in real.columns:
        real = real.filter(~pl.col(flag).fill_null(False))
    distinct = real.select(present)
    name_col = str(METADATA.IMAGE_NAME)
    if name_col in present:
        distinct = distinct.filter(pl.col(name_col).is_not_null())
    distinct = distinct.unique()
    return [
        {_REMBI_IMAGE_META_COLUMNS[c]: record[c] for c in present}
        for record in distinct.iter_rows(named=True)
    ]


def finalize_post_master_outputs(
    output_dir: Path,
    master_df: "pl.DataFrame",
    pipeline: Optional["ImagePipeline"],
    metadata_csv: Optional[Path] = None,
    no_qc: bool = False,
    study_config: Optional[dict] = None,
) -> "pl.DataFrame":
    """Run every CLI side effect that follows a freshly written master file.

    This is the single canonical entry point for the work that happens after
    :data:`~phenotypic.sdk_.MASTER_MEASUREMENTS_PARQUET` (and its CSV
    counterpart) land on disk. Every code path that writes the master —
    :func:`aggregate_measurements`, the recompile sentinel, future
    re-aggregators — should call this so the post-applied
    :data:`~phenotypic.sdk_.MEASUREMENTS_PARQUET` mirror, the persisted
    :data:`~phenotypic.sdk_.PIPELINE_JSON`, the analysis output, and the
    per-feature splits stay in lock-step.

    The order is:

    1. If *metadata_csv* is provided, left-join its rows onto a copy of
       *master_df* via :func:`join_metadata` — so a metadata key that
       matched no measured object survives as a phantom row flagged
       ``QC_MetadataOnly = true``. The join lands here — not
       on the master archive on disk — so
       ``master_measurements.{csv,parquet}`` stays a clean, op-free
       archive of what the workers measured, while the working frame
       used for the mirror picks up the external metadata columns. The
       join runs **before** post so :class:`PostMeasurement` ops can
       reference joined columns (e.g.
       ``EdgeCorrector(groupby="MetadataGenetic_Strain")``).
    2. Apply ``pipeline._post`` to the (optionally metadata-joined)
       working frame via :func:`_apply_post_to_master`. The resulting
       ``post_df`` is what the GUI viewer/curation layer reads from
       :data:`~phenotypic.sdk_.MEASUREMENTS_CSV` /
       :data:`~phenotypic.sdk_.MEASUREMENTS_PARQUET`. When *pipeline*
       is ``None`` or has no post ops, ``post_df`` equals the working
       frame unchanged.
    3. :func:`_seed_measurements` writes the post-applied frame.
    4. Split ``post_df`` into per-feature spreadsheets based on the
       ``MeasurementInfo`` columns present in the frame, so users see
       post-derived columns (e.g. ``Metadata_Strain`` from
       ``ExpandMetadata``) in
       ``measurements_by_feature/<feature>.{csv,parquet}``.
    5. When *pipeline* is provided: persist ``pipeline.json``, run
       :func:`_emit_analysis_outputs` against ``post_df`` (so analysis
       sees both post-applied and metadata-joined data), and run QC when
       configured.

    Failures inside metadata-join, split, or analysis are logged at WARNING
    and never raise; the master files remain authoritative regardless of
    what happens here.

    Args:
        output_dir: Output root that already contains
            :data:`~phenotypic.sdk_.MASTER_MEASUREMENTS_PARQUET` (and the
            CSV counterpart).
        master_df: The clean (post-free, metadata-free) aggregated master.
        pipeline: Recovered pipeline, or ``None`` when it can't be
            located (the SLURM sentinel may run before any pipeline.json
            is persisted).
        metadata_csv: Optional external metadata CSV. When provided, its
            columns are left-joined onto an in-memory copy of *master_df*
            before post ops run, so :class:`PostMeasurement` operations
            can reference joined columns. Only the mirror (and its
            derivatives) carry external metadata — never the master
            archive on disk. Undetected metadata keys survive as phantom
            rows carrying ``QC_MetadataOnly = true``.
        no_qc: When ``True``, skip the QC compute step entirely (no ``qc/``
            artifact is written and the GUI review state is left as-is).
            When ``False`` (default), QC runs whenever the pipeline has a
            non-empty ``qc`` section: the GUI-owned
            :data:`~phenotypic.sdk_.QC_REVIEW_STATE_JSON` is cleared
            (a fresh CLI run resets review progress) and
            :func:`phenotypic.sdk_._qc_recipe._runner.run_qc` writes the ``qc/`` artifact from
            the post-applied + metadata-joined frame. QC failures are
            logged and never affect the authoritative master files.
        study_config: Optional REMBI Study-level fields (parsed ``--study``
            YAML). Forwarded to the best-effort REMBI manifest writer, where
            they override constant ``Metadata_*`` study columns folded into
            ``deliverables/rembi.yaml``. ``None`` (the default) emits the
            manifest with only the study columns the mirror carries.

    Returns:
        The post-applied frame (with external metadata joined when
        *metadata_csv* is provided). Callers that run additional side
        effects downstream (e.g. analysis-plugin dispatch in the recompile
        worker) should pass this to those steps so plugins see the same
        post-applied data the GUI viewer and analysis chain see, rather
        than the clean master. Equal to *master_df* when no post ops
        are configured and no metadata CSV is supplied.
    """
    from phenotypic.sdk_ import migrate_legacy_qc

    migrate_legacy_qc(output_dir)

    # Metadata join runs first so PostMeasurement ops can reference joined
    # columns (e.g. ``EdgeCorrector(groupby="MetadataGenetic_Strain")``). The
    # master archive on disk is already written by the caller and stays
    # clean — only this in-memory working frame picks up the join.
    working_df = master_df
    if metadata_csv is not None:
        try:
            working_df = join_metadata(master_df, metadata_csv, how="left")
        except Exception as e:
            logger.warning(
                "Failed to join metadata CSV: %s: %s", type(e).__name__, e
            )
    post_df = _apply_post_to_master(working_df, pipeline)
    # Reorder the mirror/splits/analysis frame to the canonical cluster contract
    # ([front metadata] -> [measurements] -> [MetadataImage_] -> [info block]),
    # the same helper the pandas per-image path uses. The clean master on disk is
    # untouched — only this in-memory working frame is reordered.
    from phenotypic.sdk_ import order_measurement_columns

    post_df = post_df.select(order_measurement_columns(post_df.columns))
    _seed_measurements(output_dir, post_df)

    # Best-effort copy of the --metadata source CSV → deliverables/metadata.csv,
    # so a portable, co-located copy of the FULL original mapping survives next to
    # the run (the mirror is not a substitute for it: the left join keeps every
    # metadata row but a later post op may filter rows away, and
    # job_metadata.json only holds an absolute path useless once results move off
    # the cluster). Content-only copy (shutil.copy, not copy2 — mtime is not
    # preserved by design); re-running/--recompile overwrites it with a fresh
    # copy, mirroring master_measurements.*. Guarded like the other finalize side
    # effects: a failure is logged and never raised (spec §8.3 / D6).
    if metadata_csv is not None:
        try:
            shutil.copy(
                metadata_csv, metadata_csv_deliverable_path(output_dir)
            )
        except Exception:
            logger.warning(
                "Failed to copy --metadata CSV to deliverables/metadata.csv "
                "(master/measurements still written)",
                exc_info=True,
            )

    # Always emit the REMBI run manifest (deliverables/rembi.yaml) from the
    # post-applied MIRROR — never the clean master — folding its per-colony rows
    # up to each REMBI module's scope. Best-effort like the metadata.csv copy:
    # ``write_rembi_manifest`` is internally guarded and no-ops when deliverables/
    # is absent, and the mirror→image-metadata derivation is wrapped so a forward
    # finalize is never blocked.
    try:
        from phenotypic.sdk_._rembi_manifest import write_rembi_manifest

        write_rembi_manifest(
            output_dir,
            post_df.to_pandas(),
            _image_metadata_from_mirror(post_df),
            study_config=study_config,
        )
    except Exception:
        logger.warning(
            "Failed to write REMBI manifest deliverables/rembi.yaml "
            "(master/measurements still written)",
            exc_info=True,
        )

    if pipeline is not None:
        from phenotypic.plotting import AnalysisRegistry, PlotCoordinator

        _persist_pipeline_to_output_dir(output_dir, pipeline)
        measurements_pd = post_df.to_pandas()
        coordinator = PlotCoordinator(pipeline, output_dir)
        registry = AnalysisRegistry(deliverables_dir(output_dir))
        coordinator.emit_measurements(measurements_pd)

        analysis_result = _emit_analysis_outputs(output_dir, post_df, pipeline)
        if analysis_result is not None:
            registry.register(
                analysis_result.analysis_id,
                analysis_result.table,
                producer=analysis_result.producer,
                artifacts=analysis_result.artifacts,
                manifest_entry=analysis_result.manifest_entry,
            )
        coordinator.emit_analyses(measurements_pd, registry)

        # QC compute + review-progress reset. A fresh CLI run is "a different
        # run", so the GUI-owned review_state.json is cleared regardless of
        # whether QC then recomputes (so stale review progress never carries
        # across a rerun). The ``qc/`` artifact is rewritten by ``run_qc`` only
        # when QC is enabled and configured; failures are isolated so the
        # authoritative master files are never affected.
        _reset_qc_review_state(output_dir)
        successful_qc: dict[str, Any] = {}
        if not no_qc and pipeline.get_qc():
            try:
                # Import the submodule directly (not ``phenotypic.qc``) so QC
                # compute is only pulled in on the path that needs it, keeping
                # the qc package __init__ free of an eager _runner import.
                from phenotypic.sdk_._qc_recipe._runner import run_qc

                successful = run_qc(measurements_pd, pipeline, output_dir)
                successful_qc = {
                    module.instance_id: module for module in successful
                }
            except Exception:
                logger.warning(
                    "QC compute failed (master/measurements still written)",
                    exc_info=True,
                )
        coordinator.emit_qc(
            measurements_pd,
            registry,
            successful_modules=successful_qc,
            qc_database=(
                qc_duckdb_path(output_dir)
                if successful_qc
                else None
            ),
        )
    else:
        logger.warning(
            "Pipeline not available — skipping analysis, QC, and "
            "pipeline.json persistence (master files still written to %s)",
            output_dir,
        )

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

    # Re-emit the durable error-triage deliverables (errors/* + error_analysis.*)
    # from the labels store, keyed off the CLEAN master (the same frame the GUI's
    # CurationLabels loads, so headless == live). No-op without a durable
    # qc/curation_labels.parquet. (spec §9)
    try:
        from phenotypic._cli._cli_error_outputs import (
            reemit_error_deliverables,
        )

        reemit_error_deliverables(output_dir, master_df)
    except Exception:  # defensive: a curation re-emit must never fail finalize
        logger.warning(
            "Failed to re-emit error-triage deliverables", exc_info=True
        )

    return post_df


def _reset_qc_review_state(output_dir: Path) -> None:
    """Delete ``qc/review_state.json`` if present (CLI rerun resets review).

    The GUI owns ``review_state.json`` (per-module review progress); a fresh
    CLI recompile/remeasure is a new run, so any prior review progress is
    cleared here. ``run_qc`` itself never touches this file, so the GUI's
    in-session recompute preserves progress — only the CLI finalize path
    resets it.

    Best-effort: a missing file is a no-op and a failed unlink is logged at
    WARNING rather than raising.

    Args:
        output_dir: Run output root.
    """
    from phenotypic.sdk_ import qc_review_state_path
    from phenotypic.sdk_._file_locking import exclusive_path_lock

    state_path = qc_review_state_path(output_dir)
    try:
        lock_path = state_path.with_name(f".{state_path.name}.lock")
        with exclusive_path_lock(lock_path):
            if not state_path.exists():
                return
            state_path.unlink()
            logger.debug("Reset QC review state at %s", state_path)
    except OSError:
        logger.warning(
            "Failed to reset QC review state at %s", state_path, exc_info=True
        )


def split_master_by_feature(
    master_df: "pl.DataFrame",
    output_dir: Path,
    pipeline: Optional["ImagePipeline"] = None,
) -> Dict[str, Path]:
    """Write one CSV + Parquet per recognized feature into *output_dir*.

    Creates ``output_dir/measurements_by_feature/`` and emits a spreadsheet
    for every producer key returned by
    :func:`phenotypic.util.split_measurements`. Each spreadsheet contains
    all non-feature columns (metadata, object label, grid info, joined
    external metadata) alongside only that producer's columns.

    Args:
        master_df: Aggregated master measurements.
        output_dir: Base output directory; the ``measurements_by_feature/``
            subdirectory is created if missing.
        pipeline: Retained for backward compatibility with legacy callers.
            The split is now derived dynamically from ``MeasurementInfo``
            columns in *master_df*.

    Returns:
        Mapping of producer key → path to the emitted CSV. Empty if nothing
        could be split.
    """
    del pipeline

    split_frames = split_measurements(master_df)
    if not split_frames:
        logger.info("No recognized MeasurementInfo columns -- skipping split")
        return {}

    split_dir = measurements_by_feature_dir(output_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}
    for key, subset_frame in split_frames.items():
        if isinstance(subset_frame, pd.DataFrame):
            subset = pl.from_pandas(subset_frame)
        else:
            subset = subset_frame
        csv_path = split_dir / f"{key}.csv"
        pq_path = split_dir / f"{key}.parquet"

        def _write_parquet(
            path: str, _subset: "pl.DataFrame" = subset
        ) -> None:
            _subset.write_parquet(path, **PARQUET_WRITE_OPTIONS)

        try:
            atomic_write_with_writer(csv_path, subset.write_csv)
        except Exception:
            logger.warning(
                "Failed to write split CSV for %r", key, exc_info=True
            )
            continue

        try:
            atomic_write_with_writer(pq_path, _write_parquet)
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
    no_qc: bool = False,
    study_config: Optional[dict] = None,
) -> Optional[Path]:
    """Aggregate per-image Parquet files into a master CSV via DuckDB.

    Scans ``results/{name}/measurements/`` for each dataset, looking for
    Parquet (``.parquet``) files.  Prefers pre-aggregated
    ``_dataset_aggregated.parquet`` files when available, falling back to
    individual per-image files.

    Uses :func:`aggregate_parquet_files` for efficient in-memory concatenation
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
        pipeline: Optional :class:`ImagePipeline` used for post, analysis,
            QC, and pipeline persistence.  When omitted, the pipeline is
            recovered from ``processing_state.json`` / the pipeline JSON
            copy in *output_dir*. Per-feature splits are derived from the
            aggregated frame's ``MeasurementInfo`` columns even when the
            pipeline cannot be recovered.
        no_qc: Forwarded to :func:`finalize_post_master_outputs` to skip
            the QC compute step. See that function for details.
        study_config: Optional REMBI Study-level fields (parsed ``--study``
            YAML) forwarded to :func:`finalize_post_master_outputs`, where they
            override constant ``Metadata_*`` study columns in the emitted
            ``deliverables/rembi.yaml``.

    Returns:
        Path to ``master_measurements.csv``, or ``None`` if no
        measurements were found.

    Side effects:
        Delegates the post-master work to
        :func:`finalize_post_master_outputs`, which always seeds
        ``measurements.{csv,parquet}``, writes per-feature sub-spreadsheets
        into ``output_dir/measurements_by_feature/``, and — when a pipeline
        is available — persists ``pipeline.json`` and runs the analysis chain
        into ``analysis.{csv,parquet}``.

        ``master_measurements.{csv,parquet}`` are intentionally a clean
        (pre-post) archive of what the per-image runs measured, while
        ``measurements.{csv,parquet}`` carry the post-applied frame that
        the GUI viewer reads/curates. The two diverge whenever the
        pipeline has any :class:`PostMeasurement` op configured. Split
        and analysis failures never change the return value.
    """
    path_to_dataset = measurement_sources_by_path(
        discover_measurement_sources(output_dir, dataset_names)
    )

    # -- Stage to $SCRATCH ---------------------------------------------
    scratch_dir = _stage_to_scratch(list(path_to_dataset.keys()))
    if scratch_dir is not None:
        active_mapping = _remap_to_scratch(path_to_dataset, scratch_dir)
    else:
        active_mapping = path_to_dataset

    # -- Polars aggregation --------------------------------------------
    master_df = aggregate_parquet_files(
        file_paths=list(active_mapping.keys()),
        path_to_dataset=active_mapping,
        include_dataset_column=include_dataset_column,
        keep_filename=True,
    )

    if scratch_dir is not None and master_df is not None:
        staged_to_original = {
            str(scratch_dir / _scratch_dest_name(original)).replace(
                "\\", "/"
            ): str(original)
            for original in path_to_dataset
        }
        master_df = master_df.with_columns(
            pl.col(SOURCE_PATH_COLUMN)
            .str.replace_all(r"\\", "/")
            .replace_strict(
                staged_to_original,
                default=pl.col(SOURCE_PATH_COLUMN),
            )
            .alias(SOURCE_PATH_COLUMN)
        )

    if scratch_dir is not None:
        _cleanup_scratch(scratch_dir)

    if master_df is None:
        logger.warning("No valid measurements found for aggregation")
        return None

    master_df = add_metadata_image_name_from_filename(master_df)

    # -- Write master CSV and Parquet ----------------------------------
    master_csv_path = master_measurements_csv_path(output_dir)
    master_pq_path = master_measurements_parquet_path(output_dir)

    try:
        atomic_write_with_writer(master_csv_path, master_df.write_csv)
    except Exception:
        logger.error("Failed to save master CSV")
        return None

    try:
        atomic_write_with_writer(
            master_pq_path,
            lambda p: master_df.write_parquet(p, **PARQUET_WRITE_OPTIONS),
        )
    except Exception:
        logger.warning("Failed to save master Parquet (CSV was saved)")

    logger.info(
        "Aggregated %d rows x %d cols into %s",
        master_df.height,
        master_df.width,
        master_csv_path.name,
    )

    resolved_pipeline = (
        pipeline
        if pipeline is not None
        else _load_pipeline_from_output_dir(output_dir)
    )
    finalize_post_master_outputs(
        output_dir,
        master_df,
        resolved_pipeline,
        metadata_csv=metadata_csv,
        no_qc=no_qc,
        study_config=study_config,
    )

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
                for measure-mode reruns that should not regenerate overlays.
        """
        self.base_dir = Path(base_dir)
        self.save_layers = save_layers
        self.extensions = extensions
        self.include_dataset_column = include_dataset_column
        self.overlay_alpha = overlay_alpha
        self.save_overlays = save_overlays

        # Results directory for dataset outputs (images, measurements, overlays)
        self.results_dir = self.base_dir / DIR_RESULTS

        # Logs directory in the hidden machine-state cache.
        self.logs_dir = logs_dir(self.base_dir)

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
                for measure-mode reruns that should not regenerate
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
        when :attr:`save_overlays` is False (e.g. measure-mode reruns).

        Args:
            datasets: List of datasets to create directories for
        """
        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create results directory for dataset outputs
        self.results_dir.mkdir(exist_ok=True)

        # Create logs directory in the hidden machine-state cache.
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "slurm").mkdir(parents=True, exist_ok=True)

        # Create dataset folders with subdirectories under results/
        for dataset in datasets:
            dataset_dir = self.results_dir / dataset.name
            dataset_dir.mkdir(exist_ok=True)

            (dataset_dir / DIR_MEASUREMENTS).mkdir(exist_ok=True)
            (dataset_dir / DIR_HDF).mkdir(exist_ok=True)
            if self.save_overlays:
                dataset_overlays_dir(self.base_dir, dataset.name).mkdir(
                    parents=True, exist_ok=True
                )

    def get_output_path(
        self, dataset_name: str, layer: str, image_stem: str
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
        # Overlays are a user-facing deliverable, not a per-image result:
        # route them to <base>/deliverables/overlays/<ds>/<stem>.png.
        if layer == "overlays":
            return (
                dataset_overlays_dir(self.base_dir, dataset_name)
                / f"{image_stem}.png"
            )

        # Determine extension
        if layer == "measurements":
            ext = ".parquet"
        elif layer == "hdf":
            ext = self.extensions.get("hdf", ".h5")
        else:
            if not self.save_layers.get(layer):
                raise ValueError(f"Layer '{layer}' is not enabled")
            ext = self.extensions.get(layer, ".png")

        # Always use: results/dataset/layer/file
        return self.results_dir / dataset_name / layer / f"{image_stem}{ext}"

    def save_measurements(
        self, measurements: pd.DataFrame, dataset_name: str, image_stem: str
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
        if (
            self.include_dataset_column
            and str(EXPERIMENT_METADATA.DATASET) not in measurements.columns
        ):
            measurements = measurements.copy()
            measurements.insert(
                0, str(EXPERIMENT_METADATA.DATASET), dataset_name
            )

        output_path = self.get_output_path(
            dataset_name, "measurements", image_stem
        )
        parquet_df = pl.from_pandas(measurements)

        atomic_write_with_writer(
            output_path,
            lambda p: parquet_df.write_parquet(p, **PARQUET_WRITE_OPTIONS),
        )

        return output_path

    def save_overlay(
        self, image: Image, dataset_name: str, image_stem: str
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
        output_path = self.get_output_path(
            dataset_name, "overlays", image_stem
        )

        if not image.rgb.isempty():
            image.rgb.save_overlay(
                filepath=output_path, overlay_alpha=self.overlay_alpha
            )
        else:
            image.gray.save_overlay(
                filepath=output_path, overlay_alpha=self.overlay_alpha
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
        path. This mirrors :func:`atomic_write_with_writer`'s spirit, but h5py
        needs to own the file handle so we cannot feed it a buffer.

        Args:
            image: Image object with processing results.
            dataset_name: Dataset name.
            image_stem: Image filename without extension.

        Returns:
            Path where HDF5 was saved, or ``None`` if saving failed.
        """
        final_path = self.get_output_path(dataset_name, "hdf", image_stem)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = final_path.with_name(
            f".{final_path.name}.{os.getpid()}.part"
        )
        try:
            image.save2hdf5(tmp_path)
            os.replace(tmp_path, final_path)
            logger.info("Saved HDF5 for %s/%s", dataset_name, image_stem)
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
        no_qc: bool = False,
        study_config: Optional[dict] = None,
    ) -> Optional[Path]:
        """Aggregate per-image measurement Parquet files into master CSV.

        Args:
            datasets: List of all datasets processed.
            metadata_csv: Optional path to external CSV left-joined onto
                the mirror on shared columns.
            pipeline: Optional in-memory pipeline used to split the
                aggregated master into per-feature sub-spreadsheets. See
                :func:`aggregate_measurements` for fallback behavior.
            no_qc: Forwarded to skip the QC compute step. See
                :func:`finalize_post_master_outputs`.
            study_config: Optional parsed ``--study`` YAML forwarded to the
                REMBI manifest writer in :func:`finalize_post_master_outputs`.

        Returns:
            Path to master_measurements.csv, or None if no measurements found.
        """
        return aggregate_measurements(
            output_dir=self.base_dir,
            dataset_names=[ds.name for ds in datasets],
            include_dataset_column=self.include_dataset_column,
            metadata_csv=metadata_csv,
            pipeline=pipeline,
            no_qc=no_qc,
            study_config=study_config,
        )
