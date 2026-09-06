"""The one aggregation + join + publish path (spec §7.4).

``full``, ``measure`` and ``recompile`` all finalize through
:func:`finalize_run`, so recompile becomes *"call finalize_run again"* rather
than a parallel implementation that has to be kept in sync with the forward
one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import polars as pl

    from phenotypic import ImagePipeline
    from phenotypic.sdk_._publication_guard import CommitGuard

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 -- source selection
# ---------------------------------------------------------------------------


def refuse_mixed_measurement_authority(paths: Sequence[Path]) -> None:
    """Refuse a tree holding both embedded tables and legacy external Parquets.

    **The surviving half of the retired ``_consistent_embedded_join_keys``**
    (H6). That function carried two independent guards: this one, and a
    *mixed metadata digests or join keys* refusal. Only the second is retired
    -- D-A deliberately manufactures mixed snapshot generations on the normal
    rolling-input path, and the join is now global, so per-store recorded keys
    became provenance rather than input. Nothing about that says the
    **authority** mixture became acceptable, and deleting the whole function
    would have removed this refusal with nothing named as replacing it.

    Cheaper than what it replaces, not merely equivalent: authority is a
    property of the *path shape*, so this opens no Parquet footers at all,
    where the retired function read a schema per store.

    Args:
        paths: The measurement sources this finalization would aggregate.

    Raises:
        ValueError: Some sources are embedded store tables and some are not.
    """
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH

    suffix = MEASUREMENT_TABLE_RELATIVE_PATH.parts
    embedded = [
        path
        for path in paths
        if tuple(Path(path).parts[-len(suffix) :]) == suffix
    ]
    if embedded and len(embedded) != len(paths):
        raise ValueError(
            "Cannot aggregate mixed embedded and legacy measurement authority"
        )


def select_measurement_sources(
    output_dir: Path,
    dataset_names: Sequence[str],
) -> tuple[dict[Path, str], bool]:
    """Return ``(path_to_dataset, authorized)`` for one finalization.

    Args:
        output_dir: Run output root.
        dataset_names: Datasets to scan on the legacy arm.

    Returns:
        The sources to aggregate, and whether they came from marker-authorized
        per-image records. ``False`` means the legacy arm produced them.
    """
    from ._cli_completion import authorized_measurement_sources
    from ._measurement_sources import (
        discover_measurement_sources,
        measurement_sources_by_path,
    )

    authorized_sources = authorized_measurement_sources(output_dir)
    if authorized_sources is not None:
        # Schema-3 terminal publication never trusts checkpoint aggregates or
        # an unmarked per-image Parquet merely because it exists.
        refuse_mixed_measurement_authority(list(authorized_sources))
        return authorized_sources, True

    # LEGACY AGGREGATION ARM -- DELETE WHEN: the schema gate is armed and
    # refuses legacy trees before they reach finalization (P7 Task 5 Step 1d
    # sets _schema_shape.SCHEMA_GATE_ARMED = True). Until then a pre-record
    # tree can still reach finalize_run, and its authority lives in legacy
    # external Parquets under results/<ds>/measurements/ rather than in
    # embedded tables. When that condition holds, this branch is unreachable
    # and goes with the rest of the legacy surface.
    #
    # INV-INPUTS (§7.5) is narrowed to the authorized arm above, and this is
    # why: `flush_trailing_measurements_if_chunked` MANUFACTURES
    # `_dataset_aggregated.parquet` from `chunks/`, and
    # `discover_measurement_sources` then prefers it. That is deliberate on
    # this arm -- it is the only shape in which the preference is meaningful
    # -- and pinned by
    # `test_the_legacy_arm_still_prefers_its_dataset_aggregate`.
    from ._cli_chunk_writer import flush_trailing_measurements_if_chunked

    flush_trailing_measurements_if_chunked(output_dir)
    return (
        measurement_sources_by_path(
            discover_measurement_sources(output_dir, dataset_names)
        ),
        False,
    )


# ---------------------------------------------------------------------------
# Step 2 -- concatenation
# ---------------------------------------------------------------------------


def build_master_frame(
    output_dir: Path,
    dataset_names: Sequence[str],
    *,
    include_dataset_column: bool = True,
    shard_paths: Sequence[Path] | None = None,
) -> tuple["pl.DataFrame | None", bool]:
    """Concatenate this invocation's measurement sources into the master frame.

    Args:
        output_dir: Run output root.
        dataset_names: Datasets to scan on the legacy arm.
        include_dataset_column: Whether to insert ``Metadata_Dataset`` into
            each source that lacks it.
        shard_paths: P5's fan-out hook. When supplied, these are concatenated
            instead of reading the sources directly.

    Returns:
        ``(master_df, authorized)``. ``master_df`` is ``None`` when no source
        could be read.
    """
    import polars as pl

    from ._cli_output_manager import (
        _cleanup_scratch,
        _remap_to_scratch,
        _scratch_dest_name,
        _stage_to_scratch,
    )
    from ._cli_parquet_agg import SOURCE_PATH_COLUMN, aggregate_parquet_files
    from ._measurement_sources import add_metadata_image_name_from_filename

    path_to_dataset, authorized = select_measurement_sources(
        output_dir, dataset_names
    )

    if shard_paths is not None:
        frames = [pl.read_parquet(path) for path in shard_paths]
        if not frames:
            return None, authorized
        return pl.concat(frames, how="diagonal_relaxed"), authorized

    # -- Stage to $SCRATCH ---------------------------------------------
    scratch_dir = _stage_to_scratch(list(path_to_dataset.keys()))
    active_mapping = (
        _remap_to_scratch(path_to_dataset, scratch_dir)
        if scratch_dir is not None
        else path_to_dataset
    )

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
        return None, authorized
    return add_metadata_image_name_from_filename(master_df), authorized


# ---------------------------------------------------------------------------
# §7.5 -- the intermediates a later invocation must not mistake for inputs
# ---------------------------------------------------------------------------


def _invalidate_finalization_intermediates(
    output_dir: Path, dataset_names: Sequence[str]
) -> None:
    """Remove the previous finalization's outputs and intermediates.

    §7.5: chunk parquets, the rolling ``analysis_full.parquet``, each
    dataset's ``_dataset_aggregated.parquet`` and the recompile measurement
    shards are *products* of a finalization, never inputs to one. Under a
    rolling input, reusing any of them silently omits images that arrived
    since the cache was built, or retains rows for an image whose content
    changed and therefore has a new ``work_id``.

    **The chunk manifest and chunk state go with the chunks**, as one unit. A
    manifest naming deleted chunks is worse than no manifest, and state
    saying files are already chunked while their chunks are gone would make
    the next checkpoint write a manifest describing only the newest images.
    Nothing outside ``_cli_chunk_writer`` reads any of the three
    (``_rebuild_combined`` is its own corruption fallback for
    ``analysis_full.parquet``, deleted here alongside them), so the unit is
    self-contained.

    Best-effort throughout: a removal failure is logged and never fails a
    finalization that already published its master and mirror.

    Args:
        output_dir: Run output root.
        dataset_names: Datasets whose aggregate to drop.
    """
    import shutil

    from phenotypic.sdk_ import (
        DATASET_AGGREGATED_PARQUET,
        DIR_MEASUREMENTS,
        DIR_RECOMPILE_SHARDS,
        DIR_RESULTS,
        analysis_full_parquet_path,
        chunk_manifest_path,
        chunk_state_path,
        chunks_dir,
        progress_dir,
        recompile_dir,
    )

    progress = progress_dir(output_dir)
    targets: list[Path] = [
        chunks_dir(progress),
        recompile_dir(progress) / DIR_RECOMPILE_SHARDS,
    ]
    files: list[Path] = [
        analysis_full_parquet_path(progress),
        chunk_manifest_path(output_dir),
        chunk_state_path(output_dir),
    ]
    files.extend(
        output_dir
        / DIR_RESULTS
        / str(dataset)
        / DIR_MEASUREMENTS
        / DATASET_AGGREGATED_PARQUET
        for dataset in dataset_names
    )

    for directory in targets:
        try:
            if directory.is_dir():
                shutil.rmtree(directory)
        except OSError:
            logger.warning(
                "Could not invalidate finalization intermediate %s",
                directory,
                exc_info=True,
            )
    for path in files:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "Could not invalidate finalization intermediate %s",
                path,
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# The one path
# ---------------------------------------------------------------------------


def finalize_run(
    output_dir: Path,
    *,
    dataset_names: Sequence[str],
    include_dataset_column: bool = True,
    pipeline: "ImagePipeline | None" = None,
    metadata_csv: Path | None = None,
    no_qc: bool = False,
    study_config: dict | None = None,
    shard_paths: Sequence[Path] | None = None,
    commit_guard: "CommitGuard | None" = None,
) -> Path | None:
    """Aggregate, join, publish -- one path for ``full``, ``measure`` and ``recompile``.

    Six steps (spec §7.4, minus the backfill D-A cut):

    1. select marker-authorized embedded measurement tables
    2. concat  ->  ``master_measurements.parquet``   (un-joined, D8: no CSV)
    3. join metadata + append metadata-only phantoms + apply post ops
    4. write  ->  ``deliverables/measurements.{parquet,csv}``
    5. persist ``pipeline.json``, analysis outputs, per-feature splits
    6. publish the aggregate proof

    Steps 3-5 are :func:`~phenotypic._cli._cli_output_manager.finalize_post_master_outputs`,
    which owns them and three further un-numbered side effects (legacy-QC
    migration, canonical column ordering, and the REMBI manifest).

    INVARIANT (INV-INPUTS, §7.5) -- **step 1 selects exactly the
    marker-authorized embedded measurement tables.** It never reads a prior
    master, chunk parquet, measurement shard, ``analysis_full.parquet`` or
    ``_dataset_aggregated.parquet`` as an aggregation input. Those are
    outputs and intermediates of a PREVIOUS finalization; under a rolling
    input, reusing one silently omits images that arrived since, or retains
    rows for an image whose content changed and therefore has a new
    ``work_id``. Master is a pure function of the currently authorized
    embedded tables -- which is the derivability property this whole design
    is for.

    **The invariant is narrowed to the authorized arm.** A legacy tree can
    still reach finalization until P7 arms the schema gate, and on that arm
    ``_dataset_aggregated.parquet`` is legitimate authority -- see
    :func:`select_measurement_sources`.

    ``shard_paths`` is P5's fan-out hook: when supplied, step 2 merges those
    instead of reading the tables directly. It does not weaken INV-INPUTS,
    because the shards were themselves produced from authorized embedded
    tables **in this invocation**, and ``measurement_shards/`` is emptied
    when fan-out begins, so a prior run's shards can never be merged.

    Args:
        output_dir: Run output root.
        dataset_names: Datasets to finalize.
        include_dataset_column: Whether to insert ``Metadata_Dataset`` into
            sources that lack it. Threaded end to end -- the recompile SLURM
            finalizer task serialises it at both ends.
        pipeline: Recovered pipeline. ``None`` recovers it from the output
            directory.
        metadata_csv: The run's effective metadata snapshot. The join happens
            **here**, once, against the whole master.
        no_qc: Skip the QC compute step.
        study_config: REMBI Study-level fields forwarded to the manifest.
        shard_paths: P5's pre-merged measurement shards.
        commit_guard: Publication guard threaded to every terminal write.

    Returns:
        Path to ``master_measurements.parquet``, or ``None`` when no
        measurement source could be read.
    """
    from phenotypic.sdk_ import (
        PARQUET_WRITE_OPTIONS,
        atomic_write_with_writer,
        master_measurements_parquet_path,
    )

    from ._cli_output_manager import (
        _guarded_terminal_best_effort,
        _load_pipeline_from_output_dir,
        finalize_post_master_outputs,
    )

    output_dir = Path(output_dir)
    master_df, authorized = build_master_frame(
        output_dir,
        dataset_names,
        include_dataset_column=include_dataset_column,
        shard_paths=shard_paths,
    )
    if master_df is None:
        logger.warning("No valid measurements found for aggregation")
        return None

    master_path = master_measurements_parquet_path(output_dir)

    def write_master_parquet() -> bool:
        atomic_write_with_writer(
            master_path,
            lambda p: master_df.write_parquet(p, **PARQUET_WRITE_OPTIONS),
        )
        return True

    # D8: the Parquet is the master, so the Parquet write is the GATE. Before
    # D8 the CSV held that role and the Parquet was best-effort ("CSV was
    # saved"); leaving the roles as they were would let a run report success
    # having written no master at all.
    master_saved = _guarded_terminal_best_effort(
        commit_guard,
        write_master_parquet,
        warning="Failed to save master Parquet",
        default=False,
    )
    if not master_saved:
        return None

    logger.info(
        "Aggregated %d rows x %d cols into %s",
        master_df.height,
        master_df.width,
        master_path.name,
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
        commit_guard=commit_guard,
    )

    if authorized:
        from ._cli_completion import publish_aggregate_snapshot

        publish_aggregate_snapshot(output_dir, commit_guard=commit_guard)
        # AFTER the proof, so "invalidate on success" is literal. Publication
        # can still raise here -- a tree with no current state, an artifact
        # that moved -- and invalidating first would destroy the previous
        # finalization's intermediates on behalf of one that did not complete.
        _invalidate_finalization_intermediates(output_dir, dataset_names)

    return master_path
