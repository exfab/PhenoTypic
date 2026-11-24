"""Artifact-defined resume planning for staged GPU pipelines."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

from phenotypic.sdk_ import (
    CommitGuard,
    dataset_measurements_dir,
    journal_measurement_table,
    progress_dir,
    source_image_stem,
    zarr_store_path,
)
from phenotypic.sdk_._image_record import STAGE_STAGE3, read_image_record
from phenotypic.sdk_.ngff_ import (  # noqa: F401 -- public re-export
    PhenotypicAttr,
    read_phenotypic_attributes,
    valid_staged_store,
)

from ._cli_image_record import _now, consume_stage, record_stage
from ._cli_process_only import process_only_output_path
from ._cli_stage2_token import (
    delete_stage2_raw,
    delete_stage2_token,
    relocate_legacy_stage2_signal,
    stage2_raw_path,
    stage2_token_exists,
)
from ._cli_types import Dataset

#: ``"stage4"`` (#282 P4 T4): Stages 1-3 are done and the record is not, in a
#: run that defers its embeddings to Stage 4 (spec §4.6).
ResumeStage = Literal["stage1", "stage2", "stage3", "stage4", "complete"]
_STAGE_ORDER: dict[ResumeStage, int] = {
    "stage1": 0,
    "stage2": 1,
    "stage3": 2,
    "stage4": 3,
    "complete": 4,
}


@dataclass(frozen=True)
class StagedResumeItem:
    """One image and the earliest staged operation it still requires."""

    dataset: str
    image: Path
    stage: ResumeStage


@dataclass(frozen=True)
class StagedResumePlan:
    """Artifact-derived worklist and global initial stage for a resume run."""

    datasets: list[Dataset]
    items: tuple[StagedResumeItem, ...]
    initial_stage: ResumeStage

    @property
    def counts(self) -> dict[ResumeStage, int]:
        """Return image counts grouped by required stage."""
        counts: dict[ResumeStage, int] = dict.fromkeys(_STAGE_ORDER, 0)
        for item in self.items:
            counts[item.stage] += 1
        return counts


def pipeline_content_digest(pipeline_path: Path) -> str:
    """Return a SHA-256 digest of the serialized pipeline definition."""
    return hashlib.sha256(Path(pipeline_path).read_bytes()).hexdigest()


def valid_stage1_store(path: Path) -> bool:
    """Return whether a structurally valid store finished Stage 1.

    A decoded checkpoint is intentionally a valid OME-Zarr store so hard
    interruption never loses the source image. Its provenance lifecycle is
    still ``in_progress``, however, and therefore it must not be consumed by
    Stage 2 or mistaken for a resumable Stage-1 result. Stores created before
    provenance was introduced remain eligible through the missing-journal
    compatibility branch.
    """
    if not valid_staged_store(path):
        return False
    try:
        journal = read_phenotypic_attributes(path).get(
            PhenotypicAttr.PROVENANCE
        )
    except (OSError, KeyError, ValueError, TypeError, AttributeError):
        return False
    if journal is None:
        return True
    if not isinstance(journal, Mapping):
        return False
    return journal.get("status") in {"staged", "complete"}


def _staged_store_has_work_id(path: Path, work_id: str) -> bool:
    """Return whether a structurally valid store carries ``work_id``."""
    if not valid_staged_store(path):
        return False
    try:
        block = read_phenotypic_attributes(path)
        return block.get(PhenotypicAttr.WORK_ID) == work_id
    except (OSError, KeyError, ValueError, TypeError, AttributeError):
        return False


def staged_store_matches_work_id(path: Path, work_id: str) -> bool:
    """Return whether a valid staged store is bound to ``work_id``.

    Replaces ``staged_hdf_matches_work_id``. The work id lives in
    ``attributes.phenotypic.work_id``, written at store-build time -- never
    patched in afterwards, because the root ``zarr.json`` is written last.

    Args:
        path: Candidate ``*.ome.zarr`` directory.
        work_id: The work id this run expects the store to carry.

    Returns:
        ``True`` only for a valid staged store bound to *work_id*.
    """
    return valid_stage1_store(path) and _staged_store_has_work_id(
        path, work_id
    )


def stage3_completion_exists(
    output_dir: Path, dataset: str, image_stem: str
) -> bool:
    """Return whether Stage 3 fully published all required image artifacts.

    Spec §6.1: the fact moved from its own ``stage3_complete/<ds>/<stem>.json``
    tree into ``stages.stage3`` of the per-image record. **The name is kept on
    purpose** -- six modules import it, the SLURM observer among them, and
    renaming it is P6's job rather than this collapse's.

    This membership test is the single implementation of "Stage 3 was
    reported", and deliberately does **not** gain a wrapper in ``sdk_``. The
    check already has exactly one home -- this function, which every consumer
    already routes through -- so a second helper called only from here would be
    indirection rather than deduplication. It reads through
    :func:`read_image_record`, so an absent or unparseable record degrades to
    "not reported", which re-runs a stage rather than skipping one on the
    strength of a file nobody could parse.
    """
    record = read_image_record(output_dir, dataset, image_stem)
    if record is None:
        return False
    stages = record.get("stages")
    return isinstance(stages, dict) and STAGE_STAGE3 in stages


def stages_1_to_3_done(
    output_dir: Path, dataset: str, stem: str, work_id: str
) -> bool:
    """Whether one image of a run that embeds on the GPU has finished Stages 1-3.

    **The one "done" predicate under deferral** (#282 plan review C1). When
    ``defers_embeddings`` holds, Stage 3 (or the single-pass core) writes the
    store with its measurements and ``stages.stage3``, and writes no record and
    no embeddings; Stage 4 owns the image from there. So every recovery
    shortcut that used to publish such an image returns early on this and
    publishes nothing, and every stage loop skips it: publishing would certify
    an image with no embeddings, and re-running a stage would re-detect or
    re-measure it, with its Stage-2 signal already consumed.

    All three conjuncts are needed:

    - ``stages.stage3`` in the record (:func:`stage3_completion_exists`): the
      core reached its last write. A core that re-runs drops it first.
    - a store bound to *work_id* (:func:`staged_store_matches_work_id`): the
      store is this configuration's, not a previous one's.
    - a measurement table in the store's journal run
      (``journal_measurement_table``): the final save, not Stage 1's, is on
      disk.

    It says nothing about Stage 4: an image whose record is valid is both
    done here and finished (``valid_image_success``).

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        stem: Image stem.
        work_id: The image's current work id.
    """
    if not stage3_completion_exists(output_dir, dataset, stem):
        return False
    store = zarr_store_path(output_dir, dataset, stem)
    return (
        staged_store_matches_work_id(store, work_id)
        and journal_measurement_table(store) is not None
    )


Stage4Status = Literal["done", "pending", "not_ready"]


# ---------------------------------------------------------------------------
# The measure-done marker (#282 U18)
# ---------------------------------------------------------------------------


def measure_done_marker_path(
    output_dir: Path, epoch: str, dataset: str, stem: str
) -> Path:
    """``<output>/.phenotypic/progress/measure_done/<epoch>/<ds>/<stem>.json`` (U18).

    The segment is :data:`~phenotypic.sdk_.DIR_MEASURE_DONE`. One file per
    image per SLURM ``--mode measure`` pass, under ``.phenotypic/`` beside
    the other per-image progress state, never inside the store.
    """
    from phenotypic.sdk_ import DIR_MEASURE_DONE

    return progress_dir(output_dir) / DIR_MEASURE_DONE / epoch / dataset / f"{stem}.json"


def write_measure_done_marker(
    output_dir: Path,
    dataset: str,
    stem: str,
    *,
    epoch: str,
    work_id: str,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Atomically record that this pass's measure step rewrote *stem*'s table (U18).

    Written by ``run_measure_step`` after the measure core returns, under the
    epoch's *commit_guard*. Its absence is the evidence of a killed step: a
    same-pipeline measure core leaves the forward run folder in place, so the
    store alone cannot tell a finished step from one killed before its
    write. Every reader asks for it under the pass's own epoch, so a later
    pass ignores an earlier pass's markers.
    """
    import json

    from phenotypic.sdk_ import atomic_write_with_writer

    final = measure_done_marker_path(output_dir, epoch, dataset, stem)
    payload = json.dumps({"epoch": epoch, "work_id": work_id, "at": _now()})

    def _write(path: str) -> None:
        Path(path).write_text(payload, encoding="utf-8")

    atomic_write_with_writer(final, _write, commit_guard=commit_guard)
    return final


def measure_done_marker_exists(
    output_dir: Path, epoch: str, dataset: str, stem: str
) -> bool:
    """Whether this pass's measure step finished *stem* (:func:`write_measure_done_marker`)."""
    return measure_done_marker_path(output_dir, epoch, dataset, stem).is_file()


def clear_measure_done_markers(output_dir: Path) -> None:
    """Remove every pass's measure-done markers: a new measure submission's first write (U18).

    An earlier pass's markers are never read again (every reader names its
    own epoch), so they are only disk; the submitter clears them before it
    publishes the new epoch.
    """
    import shutil

    from phenotypic.sdk_ import DIR_MEASURE_DONE

    shutil.rmtree(progress_dir(output_dir) / DIR_MEASURE_DONE, ignore_errors=True)


def stage4_status(
    output_dir: Path,
    dataset: str,
    stem: str,
    work_id: str,
    *,
    republish: bool = False,
    pipeline_sha256: str | None = None,
    epoch: str | None = None,
) -> Stage4Status:
    """Where one image of a run that embeds on the GPU stands with Stage 4 (spec §4.6).

    - ``"done"``: its record is valid (``valid_image_success``), so Stage 4
      has published it, or nothing is left for Stage 4 to do;
    - ``"pending"``: it finished Stages 1-3 (:func:`stages_1_to_3_done`) and
      has no valid record, so Stage 4 owns it;
    - ``"not_ready"``: neither; Stages 1-3 have work left (or failed).

    **Republish mode** (``--mode measure``, #282 P6 T1) asks of the measure
    pipeline's run folder instead (:func:`_republish_stage4_status`): the
    record already exists, and its ``work_id`` is the forward one, never
    *work_id* (the measure work id).

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        stem: Image stem.
        work_id: The image's current work id (unused in republish mode).
        republish: Measure mode's republish Stage 4.
        pipeline_sha256: The measure pipeline file's sha256, whose run
            folder republish mode asks about; ``None`` outside it.
        epoch: Republish mode: the SLURM generation of the measure pass,
            which a republished record is stamped with; ``None`` locally.
    """
    if republish:
        return _republish_stage4_status(output_dir, dataset, stem, pipeline_sha256, epoch)
    from ._cli_completion import valid_image_success

    if valid_image_success(
        output_dir, dataset=dataset, image_stem=stem, work_id=work_id
    ):
        return "done"
    if stages_1_to_3_done(output_dir, dataset, stem, work_id):
        return "pending"
    return "not_ready"


def _republish_stage4_status(
    output_dir: Path,
    dataset: str,
    stem: str,
    pipeline_sha256: str | None,
    epoch: str | None,
) -> Stage4Status:
    """:func:`stage4_status` in republish mode.

    - ``"not_ready"``: the image's record has no ``artifacts`` (never
      forward-published); or, on SLURM (*epoch* given), this pass's measure
      step left no measure-done marker for it (U18,
      :func:`write_measure_done_marker`: the step was killed, or failed);
      or its store has no run folder of the measure pipeline declaring a
      measurement table (the measure core has not written one);
    - ``"done"``: **this pass's** Stage 4 republished it: the record
      verifies against the store as it is (``valid_image_success`` under
      the record's own ``work_id``), certifies that folder's table, and
      carries *epoch* as its ``lifecycle_epoch``;
    - ``"pending"``: otherwise.

    **The epoch is what makes "done" mean re-embedded.** A record that
    merely verifies does not: a same-pipeline measure core that carries
    the embeddings forward writes a root byte-identical to the one the
    forward record certified (the carried entries keep their sha256, and a
    measurements descriptor holds no content digest), so the forward
    record still verifies before any re-embedding. A republish stamps the
    SLURM generation; the measure core never writes the record. Locally
    (*epoch* ``None``) the record keeps its own epoch, so nothing is ever
    ``"done"``: a local Stage 4 takes exactly the stores its invocation
    measured.
    """
    from phenotypic.sdk_ import image_record_path, ngff_, zarr_store_path
    from phenotypic.sdk_._image_record import read_image_record
    from phenotypic.sdk_._table_runs import _run_of_block, owned_table_run_id

    from ._cli_completion import valid_image_success
    from ._cli_recompile_recovery import certified_table_run

    record = read_image_record(output_dir, dataset, stem)
    artifacts = record.get("artifacts") if record is not None else None
    if not isinstance(artifacts, dict) or not artifacts:
        return "not_ready"
    if epoch is not None and not measure_done_marker_exists(output_dir, epoch, dataset, stem):
        # U18: this pass never finished measuring it. With the forward
        # pipeline its folder exists from the forward run, so the folder
        # cannot say so; Stage 4 must not certify the un-re-measured table.
        return "not_ready"
    store = zarr_store_path(output_dir, dataset, stem)
    try:
        block = ngff_.read_phenotypic_attributes(store)
    except (OSError, KeyError, ValueError):
        return "not_ready"
    run_id = owned_table_run_id(block, pipeline_sha256=pipeline_sha256)
    run = None if run_id is None else _run_of_block(store, block, run_id)
    if run is None or not run.has_measurements:
        return "not_ready"
    work_id = record.get("work_id")
    certified = certified_table_run(
        output_dir, image_record_path(output_dir, dataset, stem)
    )
    if (
        epoch is not None
        and record.get("lifecycle_epoch") == epoch
        and isinstance(work_id, str)
        and valid_image_success(
            output_dir, dataset=dataset, image_stem=stem, work_id=work_id
        )
        and certified is not None
        and certified.run_id == run.run_id
    ):
        return "done"
    return "pending"


def write_stage3_completion_marker(
    output_dir: Path,
    dataset: str,
    image_name: str,
    image_stem: str,
    *,
    legacy_migration: bool = False,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Atomically record complete Stage 3 publication for one image.

    Writes ``stages.stage3`` through :func:`record_stage`, which merges rather
    than replaces -- Stage 2 and Stage 3 run in different jobs minutes apart
    and neither may erase the other's entry.

    ``legacy_migration`` lands **in the stage payload** (plan Task 1 Step 3b,
    which required a decision rather than a translation). A stage recorded by
    migration is a fact about that stage, ``stages`` entries are an open map,
    and the alternative -- moving the call site to P7 -- is not available:
    ``migrate_legacy_stage3_markers`` is called from the ordinary staged resume
    path at ``phenotypicCLI.py:2552``, not only from ``--mode migrate``.

    ``image_name`` is retained in the payload because it is the one fact a
    record built by :func:`record_stage` alone cannot reconstruct -- the stem
    is not the filename, and no ``relative_image_path`` exists until the image
    is published.
    """
    return record_stage(
        output_dir,
        dataset,
        image_stem,
        STAGE_STAGE3,
        {
            "at": _now(),
            "image_name": image_name,
            "legacy_migration": legacy_migration,
        },
        commit_guard=commit_guard,
    )


def remove_stage3_completion_marker(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    *,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Remove a terminal marker before regenerating upstream artifacts.

    Consumes only the ``stage3`` entry and leaves the rest of the record
    standing, which is what the deleted ``unlink`` did: the stage-3 marker and
    the per-image completion marker were separate files, and removing one never
    removed the other.
    """
    consume_stage(
        output_dir,
        dataset,
        image_stem,
        STAGE_STAGE3,
        commit_guard=commit_guard,
    )


def drop_stages_1_to_3_done(
    output_dir: Path,
    dataset: str,
    stem: str,
    *,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Undo :func:`stages_1_to_3_done` before a core re-runs one image (#282 P4 T2).

    Removes ``stages.stage3`` first, then the image's Stage-4 signals. A core
    that defers its embeddings calls this before its first write, so a crash
    anywhere inside it leaves an image that is not done, never one that a
    later invocation would hand to Stage 4 over a half-rewritten store; and
    signals computed against the store being replaced are gone before it is.
    """
    from ._cli_stage4_signal import delete_stage4_signals

    remove_stage3_completion_marker(
        output_dir, dataset, stem, commit_guard=commit_guard
    )
    delete_stage4_signals(output_dir, dataset, stem, commit_guard=commit_guard)


def classify_staged_image(
    *,
    output_dir: Path,
    dataset: str,
    image: Path,
    slot: str,
    input_root: Path,
    process_only_layer: str | None,
    markers_required: bool,
    expected_work_id: str | None = None,
    defers_embeddings: bool = False,
) -> ResumeStage:
    """Return the earliest stage required by one image's durable artifacts.

    With *defers_embeddings* (the run's ``defers_embeddings``, #282 P4 T4), an
    image that finished Stages 1-3 for *expected_work_id* -- ``stages.stage3``,
    a store bound to the work id and a measurement table, the conjuncts of
    :func:`stages_1_to_3_done` -- and has no valid record is ``"stage4"``:
    only Stage 4 is left, and no earlier stage may re-run it (plan review C1).
    Without it the same image is ``"stage3"``, as before.

    ``slot`` is an ordinary parameter, handed down by the caller, because this
    is a **pure classifier**: it has no ``StagePlan`` to derive one from and it
    must not acquire one. Relocation of a pre-slot-keying signal happens in
    :func:`build_staged_resume_plan`, *before* this is called -- a classifier
    that moved files would be a different thing, and moving them afterwards
    would be too late to change the verdict.
    """
    stem = source_image_stem(image)
    if expected_work_id is not None:
        from ._cli_completion import valid_image_success

        if valid_image_success(
            output_dir,
            dataset=dataset,
            image_stem=stem,
            work_id=expected_work_id,
        ):
            return "complete"

    if process_only_layer == "objmap":
        terminal = process_only_output_path(
            output_dir, image, input_root, "objmap", fmt="tiff"
        )
        if terminal.is_file() and expected_work_id is None:
            return "complete"

    store = zarr_store_path(output_dir, dataset, stem)
    stage2_done = stage2_token_exists(output_dir, dataset, stem, slot)
    if expected_work_id is not None:
        store_valid = staged_store_matches_work_id(store, expected_work_id)
        stage2_store_valid = _staged_store_has_work_id(store, expected_work_id)
    else:
        store_valid = valid_stage1_store(store)
        stage2_store_valid = valid_staged_store(store)
    # Stage 3 deliberately marks the root journal failed/in_progress before
    # publication. A retained Stage-2 token proves this same store previously
    # completed Stage 1, so it remains eligible for Stage 3 retry (or Stage 2
    # regeneration when the token's raw sidecar is missing).
    if not store_valid and not (stage2_done and stage2_store_valid):
        return "stage1"

    if (
        process_only_layer is None
        and expected_work_id is None
        and stage3_completion_exists(output_dir, dataset, stem)
    ):
        return "complete"
    if (
        process_only_layer is None
        and expected_work_id is not None
        and stage3_completion_exists(output_dir, dataset, stem)
        and journal_measurement_table(store) is not None
    ):
        # `store_valid` above is `staged_store_matches_work_id`, so this is
        # `stages_1_to_3_done` exactly.
        return "stage4" if defers_embeddings else "stage3"

    if (
        process_only_layer is None
        and not markers_required
        and journal_measurement_table(store) is not None
        and not stage2_done
    ):
        return "complete"

    # An explicit branch, NOT `stage2_done and raw.is_file()` (ledger FLOW-40).
    # The token is only a flag; Stage 3's real INPUT is the raw .npy. Without
    # this, a token-present/raw-missing image classifies "stage3" forever: the
    # worker reports a missing prereq rather than a scientific failure -- an
    # improvement -- but nothing ever routes it back to Stage 2, so it cannot
    # recover.
    #
    # It must NOT be folded into `stage2_done`, because `not stage2_done` is a
    # conjunct of the "complete" branch above: ANDing the raw in would flip a
    # token-present/raw-missing image that has a parquet all the way to
    # "complete".
    if (
        stage2_done
        and not stage2_raw_path(output_dir, dataset, stem, slot).is_file()
    ):
        return "stage2"

    return "stage3" if stage2_done else "stage2"


def build_staged_resume_plan(
    *,
    datasets: Sequence[Dataset],
    output_dir: Path,
    slot: str,
    input_root: Path,
    process_only_layer: str | None,
    markers_required: bool,
    work_ids: Mapping[str, Mapping[str, str]] | None = None,
    defers_embeddings: bool = False,
) -> StagedResumePlan:
    """Build a filtered worklist and earliest global resume stage.

    *defers_embeddings* is passed to :func:`classify_staged_image` for every
    image, so an image waiting only on Stage 4 is ``"stage4"`` and stays in
    the worklist (it is not ``"complete"``).

    Relocates each image's pre-slot-keying Stage-2 signal **before**
    classifying it. This is the single funnel every resume passes through and
    the last point at which a relocation can still change the verdict: once
    :func:`classify_staged_image` has returned ``"stage2"`` for an image whose
    signal is sitting at the legacy path, a full-dataset GPU sweep is already
    scheduled on the scarcest resource in the cluster.

    (Plan A named ``clear_downstream_artifacts_for_stage1``'s neighbourhood as
    the call site. That is too late by construction -- that function *deletes*
    the Stage-2 signal when Stage 1 re-runs.)
    """
    items: list[StagedResumeItem] = []
    pending_by_dataset: dict[str, list[Path]] = {}
    source_by_dataset = {dataset.name: dataset for dataset in datasets}
    for dataset in datasets:
        for image in dataset.images:
            expected_work_id = (
                work_ids.get(dataset.name, {}).get(image.name)
                if work_ids is not None
                else None
            )
            relocate_legacy_stage2_signal(
                output_dir, dataset.name, source_image_stem(image), slot
            )
            stage = classify_staged_image(
                output_dir=output_dir,
                dataset=dataset.name,
                image=image,
                slot=slot,
                input_root=input_root,
                process_only_layer=process_only_layer,
                markers_required=markers_required,
                expected_work_id=expected_work_id,
                defers_embeddings=defers_embeddings,
            )
            items.append(StagedResumeItem(dataset.name, image, stage))
            if stage != "complete":
                pending_by_dataset.setdefault(dataset.name, []).append(image)

    pending = [
        Dataset(
            name=name,
            images=images,
            input_dir=source_by_dataset[name].input_dir,
            output_dir=source_by_dataset[name].output_dir,
        )
        for name, images in pending_by_dataset.items()
    ]
    incomplete = [item.stage for item in items if item.stage != "complete"]
    initial: ResumeStage = (
        min(incomplete, key=_STAGE_ORDER.__getitem__)
        if incomplete
        else "complete"
    )
    return StagedResumePlan(pending, tuple(items), initial)


def migrate_legacy_stage3_markers(
    output_dir: Path, plan: StagedResumePlan
) -> int:
    """Create markers for completed embedded or explicit legacy tables."""
    migrated = 0
    for item in plan.items:
        if item.stage != "complete":
            continue
        measurement_table = journal_measurement_table(
            zarr_store_path(
                output_dir, item.dataset, source_image_stem(item.image)
            )
        )
        legacy_table = (
            dataset_measurements_dir(output_dir, item.dataset)
            / f"{source_image_stem(item.image)}.parquet"
        )
        if (
            measurement_table is None and not legacy_table.is_file()
        ) or stage3_completion_exists(
            output_dir, item.dataset, source_image_stem(item.image)
        ):
            continue
        write_stage3_completion_marker(
            output_dir,
            item.dataset,
            item.image.name,
            source_image_stem(item.image),
            legacy_migration=True,
        )
        migrated += 1
    return migrated


def clear_downstream_artifacts_for_stage1(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    slot: str,
    *,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Discard artifacts that cannot survive regeneration of the staged store.

    Deletes nothing else. The store itself is left alone: Stage 1's promote
    replaces it atomically, so removing it here would only open a window in
    which the image is absent and destroy the fallback if Stage 1 then fails
    (OPEN-QUESTIONS **D13**).

    Token first, then the raw array -- deleting the raw and leaving the token
    makes the next Stage 3 replay into a ``FileNotFoundError``, while the
    reverse merely orphans a ``.npy`` that Stage 2 overwrites.

    The image's Stage-4 signals go last (#282 P4 T4, spec §4.6): they were
    computed against the store Stage 1 is about to replace, so none of them
    can be valid at the new root.
    """
    from ._cli_stage4_signal import delete_stage4_signals

    delete_stage2_token(
        output_dir, dataset, image_stem, slot, commit_guard=commit_guard
    )
    delete_stage2_raw(
        output_dir, dataset, image_stem, slot, commit_guard=commit_guard
    )
    remove_stage3_completion_marker(
        output_dir,
        dataset,
        image_stem,
        commit_guard=commit_guard,
    )
    delete_stage4_signals(
        output_dir, dataset, image_stem, commit_guard=commit_guard
    )


def reconcile_stage3_publications(
    output_dir: Path,
    inventory: Mapping[str, Sequence[str]],
    slot: str,
    *,
    namespace: str,
) -> int:
    """Consume completed Stage-2 signals and quarantine unmarked parquets.

    A parquet is only eligible for aggregation after its terminal Stage 3
    marker exists. Unmarked parquets are preserved outside the aggregation
    tree so a later resume can safely republish them.
    """
    moved = 0
    quarantine = progress_dir(output_dir) / "unpublished_stage3" / namespace
    from ._cli_completion import valid_image_success
    from ._cli_state_management import load_processing_state

    state = load_processing_state(output_dir)
    work_ids = state.config.get("work_ids", {}) if state is not None else {}
    for dataset, image_names in inventory.items():
        for image_name in image_names:
            stem = source_image_stem(Path(image_name))
            dataset_work_ids = (
                work_ids.get(dataset, {}) if isinstance(work_ids, dict) else {}
            )
            work_id = (
                dataset_work_ids.get(image_name)
                if isinstance(dataset_work_ids, dict)
                else None
            )
            generally_complete = bool(
                isinstance(work_id, str)
                and valid_image_success(
                    output_dir,
                    dataset=dataset,
                    image_stem=stem,
                    work_id=work_id,
                )
            )
            if generally_complete or stage3_completion_exists(
                output_dir, dataset, stem
            ):
                # Token first, then the raw array -- same ordering rule as
                # every other consumption site.
                delete_stage2_token(output_dir, dataset, stem, slot)
                delete_stage2_raw(output_dir, dataset, stem, slot)
                continue
            parquet = (
                dataset_measurements_dir(output_dir, dataset)
                / f"{stem}.parquet"
            )
            if not parquet.is_file():
                continue
            destination = quarantine / dataset / parquet.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            parquet.replace(destination)
            moved += 1
    return moved


__all__ = [
    "ResumeStage",
    "StagedResumeItem",
    "StagedResumePlan",
    "build_staged_resume_plan",
    "classify_staged_image",
    "clear_downstream_artifacts_for_stage1",
    "drop_stages_1_to_3_done",
    "migrate_legacy_stage3_markers",
    "pipeline_content_digest",
    "reconcile_stage3_publications",
    "remove_stage3_completion_marker",
    "stage3_completion_exists",
    "staged_store_matches_work_id",
    "stages_1_to_3_done",
    "valid_stage1_store",
    "valid_staged_store",
    "write_stage3_completion_marker",
]
