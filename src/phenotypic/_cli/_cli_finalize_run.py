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
    from collections.abc import Mapping

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
    from ._cli_parquet_agg import is_embedded_measurement_table

    embedded = [path for path in paths if is_embedded_measurement_table(path)]
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


def _merge_measurement_shards(
    frames: "list[pl.DataFrame]",
) -> "pl.DataFrame | None":
    """Concatenate pre-projected shards, or ``None`` when no shard merged a source.

    **All-empty is ``None``, not an empty master** (Task 4 review, MF-2). A
    shard whose every source the projection excluded writes the zero-column
    empty sentinel, so its FILE still counts toward the carried K. When every
    shard is that sentinel nothing was aggregated, and the outcome must be
    the direct path's: ``finalize_run`` returns ``None`` and publishes no
    master, mirror or proof. Concatenating the sentinels instead produced a
    0x0 master that overwrote the previous one, and an aggregate proof
    certifying zero images while the run's live successes were not zero. A
    shard that merged stores holding zero rows still carries their columns,
    so it is not mistaken for the sentinel.

    The dtype advisory runs here as well as inside each shard (MF-1): drift
    between stores that landed in different shards is visible only at this
    concat.
    """
    import polars as pl

    from ._cli_parquet_agg import _warn_on_dtype_disagreement

    if all(frame.width == 0 for frame in frames):
        return None
    _warn_on_dtype_disagreement(frames)
    return pl.concat(frames, how="diagonal_relaxed")


def build_master_frame(
    output_dir: Path,
    dataset_names: Sequence[str],
    *,
    include_dataset_column: bool = True,
    shard_paths: Sequence[Path] | None = None,
) -> tuple["pl.DataFrame | None", bool, dict[Path, str]]:
    """Concatenate this invocation's measurement sources into the master frame.

    Args:
        output_dir: Run output root.
        dataset_names: Datasets to scan on the legacy arm.
        include_dataset_column: Whether to insert ``Metadata_Dataset`` into
            each source that lacks it.
        shard_paths: P5's fan-out hook. When supplied, these are concatenated
            instead of reading the sources directly.

    Returns:
        ``(master_df, authorized, path_to_dataset)``. ``master_df`` is ``None``
        when no source could be read.

        The third element is **the source set the master was built from**,
        returned rather than re-derived by the caller. ``finalize_run`` needs
        it to publish an aggregate proof that describes the master it just
        wrote; asking ``authorized_measurement_sources`` a second time would
        make the proof and the master answers to two different questions
        asked at two different moments, which is flow-r3 C2 exactly. On the
        embedded-table arm it is the selected set **minus every store the
        projection excluded** (P7 Task 4), so the proof never certifies an
        image the master does not carry.

        **On the** ``shard_paths`` **branch it is the full selection**, not
        the merged set: the shards applied the projection, and which stores
        they excluded is not visible here. Both shard producers therefore
        hand the merged set to ``finalize_run`` separately, as
        ``planned_work_ids`` -- the P5 fan-out through
        ``resolve_finalizer_shard_inputs``, and recompile's SLURM finalizer
        through the ``source_work_ids`` its measurement statuses record. The
        earlier KNOWN GAP -- recompile passing none, so a store its shards
        excluded was still certified -- is closed.
    """
    import polars as pl

    from ._cli_output_manager import (
        _cleanup_scratch,
        _scratch_dest_name,
        _stage_to_scratch,
    )
    from ._cli_parquet_agg import (
        SOURCE_PATH_COLUMN,
        aggregate_embedded_measurement_tables,
        aggregate_parquet_files,
        is_embedded_measurement_table,
    )
    from ._measurement_sources import add_metadata_image_name_from_filename

    path_to_dataset, authorized = select_measurement_sources(
        output_dir, dataset_names
    )

    if shard_paths is not None:
        # Concatenation in the ORDER GIVEN -- no sort here, deliberately. The
        # master's row order is therefore whatever `shard_paths` order is, and
        # the byte-identity the aggregate proof rests on is inherited from two
        # upstream properties rather than established here: `collect_shard_
        # paths` sorts by zero-padded shard id, and `shard_sources` gives each
        # shard a CONTIGUOUS block of the sorted sources. Sorting here instead
        # would also produce a correct master, and was rejected: it would put
        # the guarantee in the merge, so any other route that merges shards --
        # recompile hands its own shards to this same function -- would have
        # to re-establish it.
        frames = [pl.read_parquet(path) for path in shard_paths]
        return _merge_measurement_shards(frames), authorized, path_to_dataset

    # -- Stage to $SCRATCH: BOTH arms, including the embedded one ------
    #
    # It is worth staging on the embedded arm even though the copy looks
    # redundant there, and this was **measured, not reasoned** -- the
    # reasoning got it backwards. The argument for dropping it was that
    # `project_embedded_measurement_table` reads each store's descriptor
    # from the store regardless, so the copy cannot avoid the per-store GPFS
    # round trip it exists to amortize, and every table is then read twice.
    # All of that is true and the conclusion still does not follow: one bulk
    # multithreaded copy of many small files beats reading them individually
    # by far more than the second read costs.
    #
    # Cold, over this project's 6,529-store run (527 MB of tables), one
    # configuration per freshly-allocated node, two independent node pairs:
    #
    #     staged    323.9 s (i07) / 325.1 s (i08)   [copy 156-161 s of that]
    #     unstaged  397.3 s (i06) / 410.3 s (i09)
    #
    # ~20% faster cold, reproduced. WARM the two are a wash (~46 s either
    # way), which is why an interleaved benchmark says "staging is overhead"
    # -- its first run warms the page cache for every run after it. A
    # finalizer reading a just-written run is the cold case.
    scratch_dir = _stage_to_scratch(list(path_to_dataset.keys()))
    # Original source -> its staged copy, or `None` when nothing was staged.
    staged = (
        None
        if scratch_dir is None
        else {
            original: scratch_dir / _scratch_dest_name(original)
            for original in path_to_dataset
        }
    )

    # `finally`, because the projection raises by design (a newer store
    # schema, a table missing a column its descriptor declares), and staging
    # has already copied every table by then.
    try:
        if path_to_dataset and all(
            is_embedded_measurement_table(path) for path in path_to_dataset
        ):
            # P7 Task 4: project each table onto its own store's descriptor.
            # Excluded stores leave the returned source set too. Bytes come
            # from the staged copies; descriptors from the stores.
            master_df, path_to_dataset = aggregate_embedded_measurement_tables(
                path_to_dataset,
                include_dataset_column=include_dataset_column,
                read_paths=staged,
            )
        else:
            active_mapping = (
                path_to_dataset
                if staged is None
                else {
                    staged[original]: dataset
                    for original, dataset in path_to_dataset.items()
                }
            )

            master_df = aggregate_parquet_files(
                file_paths=list(active_mapping.keys()),
                path_to_dataset=active_mapping,
                include_dataset_column=include_dataset_column,
                keep_filename=True,
            )

            if staged is not None and master_df is not None:
                staged_to_original = {
                    str(copy).replace("\\", "/"): str(original)
                    for original, copy in staged.items()
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
    finally:
        if scratch_dir is not None:
            _cleanup_scratch(scratch_dir)

    if master_df is None:
        return None, authorized, path_to_dataset
    return (
        add_metadata_image_name_from_filename(master_df),
        authorized,
        path_to_dataset,
    )


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
# #281 spec C4, C12 -- the run summary and column schema
# ---------------------------------------------------------------------------


def _aggregated_pipeline_digests(
    aggregated_sources: "dict[Path, str]",
) -> list[str | None]:
    """Return the producing pipeline's digest for each aggregated table.

    **Every entry is** ``None`` **for a v1 flat table**, deliberately. A v1
    store's ``tables/measurements/table.parquet`` records no producing
    pipeline: the record names only the path, the descriptor has no pipeline
    field, and a table rewritten by ``--mode measure`` was never journaled
    (spec D10). The run config's digest is not a substitute -- it still names
    the full run's pipeline after ``--mode measure`` (spec C4, D6a) -- and
    neither is the finalizing invocation's own pipeline, which a continuation
    or a recompile did not measure every aggregated table with.

    P3 T4 replaces the body with ``certified_table_run(...).pipeline_sha256``
    per source, the run folder each image's record certifies (spec D6a);
    :func:`~phenotypic.sdk_._run_metrics.shared_pipeline_sha256` then reports
    the digest they share.

    Args:
        aggregated_sources: The sources the master was built from.

    Returns:
        One digest per source, ``None`` where it is not recorded.
    """
    return [None for _ in aggregated_sources]


def _accepted_work_ids(output_dir: Path) -> set[str] | None:
    """Return the run's accepted inventory, or ``None`` when no state names one."""
    from ._cli_state_management import load_processing_state

    try:
        state = load_processing_state(output_dir)
    except (KeyError, TypeError, ValueError):
        return None
    if state is None:
        return None
    raw = state.config.get("work_ids")
    if not isinstance(raw, dict):
        return None
    return {
        work_id
        for images in raw.values()
        if isinstance(images, dict)
        for work_id in images.values()
        if isinstance(work_id, str)
    }


def _published_table_schemas(
    output_dir: Path,
    *,
    master_df: "pl.DataFrame",
    mirror_df: "pl.DataFrame",
) -> "dict[str, Mapping[str, pl.DataType]]":
    """Return ``column -> dtype`` for every table finalization published.

    The master and the mirror are the frames this finalization just wrote,
    so their schemas come from the frames. The per-feature and per-category
    splits and the named analyses are not held here, so they come from the
    Parquet footers those writers published (a footer read opens no row
    group and writes nothing). Tables are keyed by their
    ``deliverables/``-relative path without the suffix, which names the
    ``.parquet`` and its ``.tsv`` twin alike: ``master_measurements``,
    ``measurements``, ``measurements_by_feature/<measurer>``,
    ``measurements_by_category/<label>``, and each named analysis id from
    ``analysis_manifest.json``.
    """
    import polars as pl

    from phenotypic.plotting._pipeline._analysis_artifacts import (
        read_analysis_manifest,
    )
    from phenotypic.sdk_ import (
        deliverables_dir,
        measurements_by_category_dir,
        measurements_by_feature_dir,
    )

    deliverables = deliverables_dir(output_dir)
    schemas: dict[str, Mapping[str, pl.DataType]] = {
        "master_measurements": master_df.schema,
        "measurements": mirror_df.schema,
    }
    paths: dict[str, Path] = {}
    for split_dir in (
        measurements_by_feature_dir(output_dir),
        measurements_by_category_dir(output_dir),
    ):
        if split_dir.is_dir():
            for path in sorted(split_dir.glob("*.parquet")):
                paths[path.relative_to(deliverables).with_suffix("").as_posix()] = path
    manifest = read_analysis_manifest(deliverables)
    if manifest is not None:
        for analysis_id, entry in sorted(manifest.analyses.items()):
            paths[analysis_id] = deliverables / entry.parquet
    schemas.update(
        (name, pl.read_parquet_schema(path)) for name, path in paths.items()
    )
    return dict(sorted(schemas.items()))


def publish_run_summaries(
    output_dir: Path,
    *,
    master_df: "pl.DataFrame",
    mirror_df: "pl.DataFrame",
    aggregated_sources: "dict[Path, str]",
    source_work_ids: Sequence[str] | None,
    commit_guard: "CommitGuard | None" = None,
    embeddings: "Mapping[str, Mapping[str, int]] | None" = None,
    measure_failed: int = 0,
) -> tuple[Path, Path]:
    """Write ``deliverables/metrics.json`` and ``deliverables/schema.json``.

    Every count comes from what finalization already holds, never from a deep
    ``resolve_run_state``, which would run an O(N) verification and write the
    verification cache inside the finalizer:

    - ``accepted``: the state's ``work_ids`` (the aggregated sources when no
      state names an inventory, as on the legacy arm);
    - ``succeeded``: the source set the master was built from;
    - ``failed``: accepted work ids in the terminal-failure journal that did
      not succeed, the same precedence ``resolve_run_state`` gives a verified
      image over a recorded failure;
    - on a ``--mode measure`` finalization, ``measure_failed``: the images
      whose re-measure failed in this invocation are added to ``failed`` and
      taken from ``succeeded`` (#281 P2 C11d). Measure records are keyed by
      measure work ids, outside the accepted inventory, so the line above
      never sees them, and the exit code (``3``) would otherwise disagree
      with ``failed: 0``. **Such an image's previous table is usually still
      aggregated:** a re-measure that fails before it replaces the table
      leaves the store and its record untouched, so the image stays
      marker-authorized, and the master, the mirror and ``objects.measured``
      carry its earlier measurements. It is counted failed because it did not
      succeed in this run, not because its rows are absent. (One that fails
      after the replace leaves its record stale, so it is not aggregated and
      the subtraction then also takes it from ``succeeded``, undercounting
      that by one; ``succeeded`` is floored at zero);
    - ``objects.measured`` and ``metadata.phantoms``: the master's rows and
      the mirror's ``QC_MetadataOnly`` rows.

    Neither file is read back by PhenoTypic, so neither is tracked state.

    Args:
        output_dir: Run output root.
        master_df: The master just written.
        mirror_df: The post-applied mirror just written.
        aggregated_sources: The sources the master was built from.
        source_work_ids: Their work ids, on the authorized arm; ``None`` on
            the legacy arm, where the sources are counted instead.
        commit_guard: Publication guard threaded to both writes.
        embeddings: Per embedding entry id, ``{"rows", "stale_skipped"}``.
            The P4 embedding deliverables pass it; ``None`` omits the key.
        measure_failed: How many images' ``--mode measure`` re-measure
            failed in this finalization's invocation; ``0`` for any other run.

    Returns:
        ``(metrics_path, schema_path)``.
    """
    from phenotypic.schema import METADATA_MATCH
    from phenotypic.sdk_ import metrics_json_path, schema_json_path
    from phenotypic.sdk_._run_metrics import (
        build_run_metrics,
        shared_pipeline_sha256,
        write_canonical_json,
    )
    from phenotypic.util._measurement_outputs import build_output_schema

    from ._cli_failure_tracker import terminal_failure_index

    succeeded_ids = set(source_work_ids) if source_work_ids is not None else None
    succeeded = (
        len(succeeded_ids)
        if succeeded_ids is not None
        else len(aggregated_sources)
    )
    accepted_ids = _accepted_work_ids(output_dir)
    if accepted_ids is None:
        accepted = succeeded
        failed = 0
    else:
        accepted = len(accepted_ids)
        failed = len(
            (set(terminal_failure_index(output_dir)) & accepted_ids)
            - (succeeded_ids or set())
        )
    failed += measure_failed
    succeeded = max(0, succeeded - measure_failed)

    phantom_flag = str(METADATA_MATCH.METADATA_ONLY)
    phantoms = (
        int(mirror_df.get_column(phantom_flag).fill_null(False).sum())
        if phantom_flag in mirror_df.columns
        else 0
    )

    import phenotypic

    metrics = build_run_metrics(
        phenotypic_version=phenotypic.__version__,
        # P3 T4 feeds certified_table_run(...).pipeline_sha256 here
        pipeline_sha256=shared_pipeline_sha256(
            _aggregated_pipeline_digests(aggregated_sources)
        ),
        accepted=accepted,
        succeeded=succeeded,
        failed=failed,
        measured_objects=master_df.height,
        metadata_phantoms=phantoms,
        embeddings=embeddings,
    )
    schema = build_output_schema(
        _published_table_schemas(
            output_dir, master_df=master_df, mirror_df=mirror_df
        )
    )

    metrics_path = metrics_json_path(output_dir)
    schema_path = schema_json_path(output_dir)
    write_canonical_json(schema_path, schema, commit_guard=commit_guard)
    write_canonical_json(metrics_path, metrics, commit_guard=commit_guard)
    return metrics_path, schema_path


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
    planned_work_ids: Sequence[str] | None = None,
    commit_guard: "CommitGuard | None" = None,
    execution_epoch: str | None = None,
    measure_failed: int | None = None,
) -> Path | None:
    """Aggregate, join, publish -- one path for ``full``, ``measure`` and ``recompile``.

    Six steps (spec §7.4, minus the backfill D-A cut):

    1. select marker-authorized embedded measurement tables
    2. concat  ->  ``master_measurements.parquet``   (un-joined, D8: no CSV)
    3. join metadata + append metadata-only phantoms + apply post ops
    4. write  ->  ``deliverables/measurements.{parquet,csv}``
    5. persist ``pipeline.json``, analysis outputs, per-feature splits
    6. write ``deliverables/metrics.json`` and ``deliverables/schema.json``
       (:func:`publish_run_summaries`, #281 spec C4/C12)
    7. publish the aggregate proof, which declares the files of steps 2-6

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
    tables **in this invocation**, and ``aggregation_shards/<epoch>/`` is
    emptied by
    :func:`~phenotypic._cli._cli_finalize_fanout.begin_aggregation_fanout`
    when fan-out begins -- at submission on SLURM, at driver start locally,
    and never at merge time -- so a prior run's shards can never be merged.

    The directory is ``aggregation_shards``, not ``measurement_shards``: that
    name is :data:`~phenotypic.sdk_.DIR_RECOMPILE_SHARDS` and already denotes
    two different live paths.

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
        planned_work_ids: The source set the shards recorded merging, from
            :func:`~phenotypic._cli._cli_finalize_fanout.resolve_finalizer_shard_inputs`.
            ``None`` on the non-fan-out path, where the set is taken from the
            sources this call itself selected -- the same moment the master
            was built from, so the proof still describes the master.
        commit_guard: Publication guard threaded to every terminal write.
        execution_epoch: The SLURM generation this finalizer runs under,
            stamped into the aggregate proof so ``phenotypic-wait`` can tell
            this submission's publication from an earlier one (#281 spec C3).
            ``None`` for a local run.
        measure_failed: A local ``--mode measure`` invocation's own failure
            count, for ``metrics.json`` (#281 P2 C11d, spec C3): the count its
            exit code is computed from. A local run mints no epoch for it,
            because a local measure journals under the forward run's
            content-derived generation, which every measure invocation
            shares. ``None`` otherwise; when ``execution_epoch`` is given (a
            SLURM finalizer) it is then the measure rows journalled under that
            generation, counted by the helper ``phenotypic-wait`` uses
            (:func:`~phenotypic._cli._cli_failure_tracker.terminal_failure_work_ids_under_epoch`).
            A forward SLURM generation has no measure rows, so that count is
            ``0`` for it.

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

    from ._cli_utils import logged_step

    output_dir = Path(output_dir)
    with logged_step(logger, "finalize_run: build master frame"):
        master_df, authorized, aggregated_sources = build_master_frame(
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
    with logged_step(logger, "finalize_run: write master parquet"):
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
    with logged_step(
        logger, "finalize_run: post-master outputs (mirror, splits, plots, QC)"
    ):
        mirror_df = finalize_post_master_outputs(
            output_dir,
            master_df,
            resolved_pipeline,
            metadata_csv=metadata_csv,
            no_qc=no_qc,
            study_config=study_config,
            commit_guard=commit_guard,
        )

    source_work_ids: list[str] | None = None
    if authorized:
        from ._cli_finalize_fanout import _work_ids_for_sources

        # The proof describes the master that was just written, and says so by
        # being GIVEN that master's source set rather than re-deriving one.
        # On the fan-out path the shards recorded what they merged; otherwise
        # it is the set this call selected a few lines above. Either way it is
        # the set the master came from, at the moment the master came from it.
        source_work_ids = (
            list(planned_work_ids)
            if planned_work_ids is not None
            else _work_ids_for_sources(output_dir, aggregated_sources)
        )

    # After the mirror and splits, whose columns schema.json describes, and
    # BEFORE the proof, which declares both files (#281 spec C4, C12).
    if measure_failed is None and execution_epoch is not None:
        from ._cli_failure_tracker import (
            MEASURE_FAILURE_STAGE,
            terminal_failure_work_ids_under_epoch,
        )

        # The measure stage only: a forward generation's own rows are already
        # counted through the accepted inventory above.
        measure_failed = len(
            terminal_failure_work_ids_under_epoch(
                output_dir, execution_epoch, stage=MEASURE_FAILURE_STAGE
            )
        )
    with logged_step(logger, "finalize_run: publish metrics.json and schema.json"):
        publish_run_summaries(
            output_dir,
            master_df=master_df,
            mirror_df=mirror_df,
            aggregated_sources=aggregated_sources,
            source_work_ids=source_work_ids,
            commit_guard=commit_guard,
            measure_failed=measure_failed or 0,
        )

    if authorized:
        from ._cli_completion import publish_aggregate_snapshot

        assert source_work_ids is not None
        with logged_step(logger, "finalize_run: publish aggregate proof"):
            publish_aggregate_snapshot(
                output_dir,
                source_work_ids=source_work_ids,
                commit_guard=commit_guard,
                execution_epoch=execution_epoch,
                require_run_summaries=True,
            )
        # AFTER the proof, so "invalidate on success" is literal. Publication
        # can still raise here -- a tree with no current state, an artifact
        # that moved -- and invalidating first would destroy the previous
        # finalization's intermediates on behalf of one that did not complete.
        with logged_step(
            logger, "finalize_run: invalidate finalization intermediates"
        ):
            _invalidate_finalization_intermediates(output_dir, dataset_names)

    return master_path
