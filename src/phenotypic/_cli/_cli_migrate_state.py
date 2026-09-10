"""Convert a legacy tree's machine state into the consolidated schema.

Spec §11.1. This module is the write half of ``--mode migrate``: it reads the
shapes P1--P6 stopped writing and produces the ones they read.

**Two legacy per-image trees, not three.** ``image_complete/`` becomes
``stages.measured`` and ``stage3_complete/`` becomes ``stages.stage3``. The
segments live in :data:`LEGACY_MARKER_SEGMENTS` rather than being spelled at
each use, and that tuple mirrors the schema gate's own loop
(``sdk_/_schema_shape.py``) so detection and conversion cannot disagree about
what a legacy tree *is*.

⛔ **``stage2_done/`` is not one of them, and the distinction is the verb.**
That tree holds a **consumable token**: Stage 3 replays the raw array,
measures, re-promotes the store and then ``unlink``s the token. It is
"Retained, not collapsed" (U-9, ``sdk_/_io_constants.py``'s comment on
``DIR_STAGE2_DONE``), and the schema gate deliberately does **not** fire on
it -- firing would classify every modern staged-GPU run ``CONVERT`` and strand
it.

So this module **reads** ``stage2_done/`` to enrich a record it is already
writing, and **never renames, consumes, unlinks or moves it**. Read-and-leave
is the whole rule. Renaming it aside -- into the ``legacy-v2/`` tree where
Task 5 guarantees nothing reads it -- would orphan every un-consumed Stage-2
result for any staged run live across the migrate, and ``--mode migrate``
would report success. That is data loss, not a failing test, and it is the
one mistake in this file that a passing suite would not catch.

**Descriptors are copied verbatim, never re-derived.** A migration that
recomputed ``artifacts`` from whatever is on disk now would certify a
corrupted artifact as sound, turning a format change into a laundering step.

**Plan and apply are separate on purpose.** :func:`plan_per_image_records`
performs no writes at all, so ``--dry-run`` can render exactly what a real
conversion would do by calling it alone. Composing them in
:func:`convert_per_image_markers` keeps the ordinary path one call.

The rename-aside of the two converted trees is **not here**: it is one
primitive shared with ``processing_state.json`` and the master, with a
collision rule and a ``--revert`` path, and it belongs with the task that owns
rollback.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from phenotypic.sdk_ import (
    DIR_IMAGE_COMPLETE,
    DIR_STAGE2_DONE,
    ProcessingStateKey,
    atomic_write_json,
    image_record_path,
    progress_dir,
    read_image_record,
    resolve_processing_state_path,
    source_image_stem,
)
from ._cli_identity import derive_processing_generation
from phenotypic.sdk_._image_record import (
    PROVENANCE_MIGRATED,
    RECORD_VERSION,
    STAGE_MEASURED,
    STAGE_STAGE2,
    STAGE_STAGE3,
)

__all__ = [
    "LEGACY_MARKER_SEGMENTS",
    "PlannedRecord",
    "PlannedState",
    "apply_per_image_records",
    "apply_processing_state",
    "convert_per_image_markers",
    "convert_processing_state",
    "plan_per_image_records",
    "plan_processing_state",
]

#: ``stage3_complete`` is module-private in :mod:`~phenotypic.sdk_._schema_shape`
#: -- P3 deleted the tree, so promoting its segment to a public constant would
#: have added a name the change was about to remove. It is restated here rather
#: than imported through the underscore, and
#: ``test_the_segments_match_the_schema_gate`` binds the two so they cannot
#: drift.
_DIR_STAGE3_COMPLETE: Final[str] = "stage3_complete"

#: The two per-image trees migrate converts, paired with the stage each
#: becomes. **Not** ``stage2_done/`` -- see the module docstring.
LEGACY_MARKER_SEGMENTS: Final[tuple[tuple[str, str], ...]] = (
    (DIR_IMAGE_COMPLETE, STAGE_MEASURED),
    (_DIR_STAGE3_COMPLETE, STAGE_STAGE3),
)


@dataclass(frozen=True)
class PlannedRecord:
    """One image's conversion, computed and not yet written.

    Carries ``sources`` so a dry run can name the files it read and a caller
    can rename exactly those aside afterwards, rather than re-deriving the
    set from a second walk that might not agree with the first.
    """

    dataset: str
    image_stem: str
    stages: Mapping[str, Mapping[str, object]]
    artifacts: Mapping[str, object]
    identity: Mapping[str, object]
    sources: tuple[Path, ...]


def _read_json(path: Path) -> Mapping[str, object] | None:
    """Return one JSON object, or ``None`` for anything unreadable.

    Never raises: a truncated marker in a tree of thousands must not abort a
    migration partway, leaving a tree that is neither shape.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _walk_segment(output_dir: Path, segment: str) -> Iterator[tuple[str, str, Path]]:
    """Yield ``(dataset, image_stem, path)`` for one legacy tree."""
    root = progress_dir(output_dir) / segment
    if not root.is_dir():
        return
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for marker in sorted(dataset_dir.glob("*.json")):
            yield dataset_dir.name, marker.stem, marker


def _stage2_entry(
    output_dir: Path, dataset: str, image_stem: str
) -> Mapping[str, object] | None:
    """Return a ``stages.stage2`` entry from the token, **without touching it**.

    An image with a Stage-2 token and no completion marker is a real
    interrupted state and worth recording -- that instinct is why an earlier
    draft listed ``stage2_done/`` beside the two marker trees. What was wrong
    was the verb. This reads the token and leaves it exactly where Stage 3
    expects to find and ``unlink`` it.

    **If you are about to add a rename, move or unlink of this path: don't.**
    A staged run live across the migrate would lose every un-consumed Stage-2
    result, silently, with migrate reporting success.
    """
    token = progress_dir(output_dir) / DIR_STAGE2_DONE / dataset / f"{image_stem}.json"
    payload = _read_json(token)
    if payload is None:
        return None
    return {"at": payload.get("completed_at"), "legacy_migration": True}


def plan_per_image_records(output_dir: Path) -> tuple[PlannedRecord, ...]:
    """Enumerate the records a conversion would write. **Writes nothing.**

    The union of the two legacy trees, not just ``image_complete/``: a
    stage-3 marker with no completion marker is a run that died between
    publishing Stage 3 and publishing completion, which an
    ``image_complete/``-only walk drops on the floor.

    Args:
        output_dir: Run output root.

    Returns:
        One :class:`PlannedRecord` per image found in either tree, ordered by
        ``(dataset, image_stem)`` so a dry run's output is stable.
    """
    found: dict[tuple[str, str], dict[str, object]] = {}
    for segment, stage in LEGACY_MARKER_SEGMENTS:
        for dataset, stem, path in _walk_segment(output_dir, segment):
            payload = _read_json(path)
            if payload is None:
                continue
            entry = found.setdefault(
                (dataset, stem),
                {"stages": {}, "artifacts": {}, "identity": {}, "sources": []},
            )
            entry["stages"][stage] = {  # type: ignore[index]
                "at": payload.get("completed_at"),
                "legacy_migration": True,
            }
            entry["sources"].append(path)  # type: ignore[union-attr]
            # Verbatim. Re-deriving these from the store would certify
            # whatever is there now.
            artifacts = payload.get("artifacts")
            if isinstance(artifacts, dict) and artifacts:
                entry["artifacts"] = artifacts
            for key in ("work_id", "relative_image_path", "mode", "attempt_id"):
                if key in payload and key not in entry["identity"]:  # type: ignore[operator]
                    entry["identity"][key] = payload[key]  # type: ignore[index]

    planned: list[PlannedRecord] = []
    for (dataset, stem) in sorted(found):
        entry = found[(dataset, stem)]
        stages = dict(entry["stages"])  # type: ignore[arg-type]
        stage2 = _stage2_entry(output_dir, dataset, stem)
        if stage2 is not None:
            stages[STAGE_STAGE2] = stage2
        planned.append(
            PlannedRecord(
                dataset=dataset,
                image_stem=stem,
                stages=stages,
                artifacts=dict(entry["artifacts"]),  # type: ignore[arg-type]
                identity=dict(entry["identity"]),  # type: ignore[arg-type]
                sources=tuple(entry["sources"]),  # type: ignore[arg-type]
            )
        )
    return tuple(planned)


def _stage_timestamp(entry: object) -> str:
    """Return a stage entry's ``at``, or ``""`` when it has none.

    ``""`` sorts before every real ISO-8601 timestamp, so an entry with no
    ``at`` loses a collision against one that has a time -- which is the
    direction that keeps information rather than discarding it.
    """
    if not isinstance(entry, Mapping):
        return ""
    at = entry.get("at")
    return at if isinstance(at, str) else ""


def _merge_stages(
    existing: object, converted: Mapping[str, Mapping[str, object]]
) -> dict[str, Mapping[str, object]]:
    """Union two stage maps, keeping the later entry on a key collision.

    **CAN-13's second half, and it is not "legacy wins".** An old-build SLURM
    array and the forward path can both write the same stage for the same
    image during the coexistence window, so a blind ``update`` in either
    direction discards a real entry. The rule is the later ``completed_at``,
    which is why this compares rather than overwrites.

    An earlier draft of this module did ``merged.update(converted)`` and would
    have replaced a forward ``stage3`` with the legacy one every time -- the
    test that catches it plants only ``image_complete/``, so the collision
    never arose and the suite stayed green.
    """
    merged: dict[str, Mapping[str, object]] = {
        key: value
        for key, value in (existing or {}).items()
        if isinstance(existing, Mapping)
    }
    for stage, entry in converted.items():
        current = merged.get(stage)
        if current is None or _stage_timestamp(entry) > _stage_timestamp(current):
            merged[stage] = entry
    return merged


def apply_per_image_records(
    output_dir: Path, planned: tuple[PlannedRecord, ...]
) -> int:
    """Write each planned record, merging into any record already there.

    **Merge, never overwrite** (CAN-13). The both-shapes-present case is real:
    an old-build SLURM array holds the old schema for its whole lifetime -- up
    to 30 days -- and keeps writing the legacy trees after a partial migrate.
    Replacing the record would discard the newer stages the forward path
    wrote in the meantime.

    Idempotence follows from the same merge: a second pass recomputes the same
    entries and writes the same bytes.

    Args:
        output_dir: Run output root.
        planned: The output of :func:`plan_per_image_records`.

    Returns:
        The number of records written.
    """
    for item in planned:
        existing = read_image_record(output_dir, item.dataset, item.image_stem)
        record: dict[str, object] = dict(existing) if existing else {}

        merged_stages = _merge_stages(record.get("stages"), item.stages)

        record.setdefault("version", RECORD_VERSION)
        record.setdefault("dataset", item.dataset)
        record.setdefault("image_stem", item.image_stem)
        for key, value in item.identity.items():
            record.setdefault(key, value)
        if item.artifacts and not record.get("artifacts"):
            record["artifacts"] = dict(item.artifacts)
        record["stages"] = merged_stages
        # A migrated record must say so: `marker_rejection` skips the
        # `work_id` fence only for records that declare `migrated`, and a
        # legacy tree's work_id need not match the one a forward run mints.
        record["provenance"] = PROVENANCE_MIGRATED

        atomic_write_json(
            image_record_path(output_dir, item.dataset, item.image_stem), record
        )
    return len(planned)


def convert_per_image_markers(output_dir: Path) -> int:
    """Convert both legacy per-image trees into per-image records.

    Plans every record first and only then writes -- marker-last, applied to
    the migration itself. A conversion that deleted as it went and then died
    would leave a tree that is neither shape.

    The two converted trees are **left in place**; renaming them aside is one
    shared primitive with a collision rule and a ``--revert`` path, and it
    belongs with rollback rather than here.

    Args:
        output_dir: Run output root.

    Returns:
        The number of records written.
    """
    return apply_per_image_records(output_dir, plan_per_image_records(output_dir))


# ---------------------------------------------------------------------------
# processing_state.json (Task 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedState:
    """The converted ``processing_state.json``, computed and not yet written.

    ``retained_failures`` is named rather than left implicit in "whatever was
    not deleted": it is the set **Task 2b** consumes into
    ``terminal_failures.jsonl``, and a consumer that re-derived it from a
    second read of the file could disagree with the pass that decided to keep
    it.
    """

    config: Mapping[str, object]
    datasets: Mapping[str, Mapping[str, object]]
    retained_failures: Mapping[str, tuple[str, ...]]


#: ``datasets.<ds>.started`` is a **dead key**: it is dropped unconditionally.
#:
#: Measured, not assumed. ``save_processing_state`` stopped writing it in P5,
#: and an AST-free enumeration of every reader --
#: ``grep -rn "ProcessingStateKey.STARTED" src/`` -- returns exactly this line.
#: Every other ``"started"`` in the CLI is an **event-log** status
#: (``append_event(..., "started")``), not a read of this key. The
#: fallback that reconstructs a ``DatasetState`` without an event log
#: (``_cli_state_management.py:183-187``) reads ``completed``, ``failed``,
#: ``errors`` and ``initial_images`` -- and **not** ``started``.
#:
#: So dropping it is removing a key with no writer and no reader, which needs
#: no condition at all.
_DEAD_DATASET_KEYS: Final[tuple[str, ...]] = (ProcessingStateKey.STARTED,)


def _completed_is_fully_consumed(
    output_dir: Path, dataset: str, completed: object
) -> bool:
    """Return whether every image ``completed`` names now has a record.

    **Per stem, not per dataset**, and the difference is data loss. On a tree
    with no event log the fallback above reads ``completed`` from this file and
    has nowhere else to get it, so the key is the only surviving statement that
    those images finished. A "does this dataset have *any* record" test would
    drop it on a dataset holding ``completed: ["a.tif"]`` and a record for
    ``b.tif`` only -- deleting the account of ``a`` on the strength of ``b``.

    Reads through :func:`~phenotypic.sdk_.read_image_record`, the same reader
    :func:`apply_per_image_records` writes through, so this cannot disagree
    with the pass that produced the records.

    Args:
        output_dir: Run output root.
        dataset: The dataset whose key is being considered.
        completed: The raw ``datasets.<ds>.completed`` value.

    Returns:
        ``True`` when every named image has a record -- and for an empty or
        absent list, which names nothing left to lose. ``False`` for a value
        that is not a list, which is a shape this build did not write and must
        not silently discard.
    """
    if completed is None:
        return True
    if not isinstance(completed, list):
        return False
    return all(
        read_image_record(output_dir, dataset, source_image_stem(Path(str(name))))
        is not None
        for name in completed
    )


def plan_processing_state(output_dir: Path) -> PlannedState | None:
    """Compute the v3 ``processing_state.json``, writing nothing.

    **Writes nothing at all**, so ``--dry-run`` (Task 5) can render exactly
    what a conversion would do by calling this alone. Deliberately not
    ``load_processing_state``: that reader calls ``migrate_legacy_machine_state``
    (``_cli_state_management.py:133``) -- a **write** -- and subscripts
    ``state_dict[ProcessingStateKey.VERSION]`` unguarded (``:192``), so a
    planner routed through it would mutate the tree it is only supposed to be
    describing and raise on the malformed one it most needs to survive.

    What changes, and what does not:

    * ``config.processing_generation`` becomes **content-derived** (spec §5.1,
      D3). The legacy value is a ``uuid4()`` no other process can re-derive,
      so a cold SLURM worker cannot fence itself against a run it never saw.
      ``per_image_config`` is ``None`` -- migrate is the caller
      :func:`~phenotypic._cli._cli_identity.derive_processing_generation`
      names for that, because a converted tree never recorded the per-image
      configuration and U-10's rule is to mark what cannot be recovered
      rather than fabricate it.
    * ``config.restart_epoch`` becomes ``0`` when absent. Its absence
      alongside ``work_ids`` is the schema gate's signal 4.
    * ``config.work_ids`` is **copied verbatim** (D-C). Re-minting them would
      invalidate every record migrate just converted, and a tree
      half-migrated across that boundary cannot be recovered without the
      original config.
    * ``datasets.<ds>.started`` is dropped **unconditionally** -- it has no
      writer and no reader left (:data:`_DEAD_DATASET_KEYS`).
    * ``datasets.<ds>.completed`` is dropped only when **every image it names
      has a record** (:func:`_completed_is_fully_consumed`). Per stem rather
      than per dataset: on a tree with no event log this key is the only
      surviving statement that those images finished.
    * ``datasets.<ds>.failed`` is always retained, and surfaced in
      ``retained_failures``. ``errors`` is untouched -- the same fallback
      reads it.

    Args:
        output_dir: Run output root. May be any directory.

    Returns:
        The planned state, or ``None`` when there is no readable state to
        convert -- an absent file, or one that is unparseable or not an
        object. An unreadable state is *unreadable*, not *legacy*: the schema
        gate gives it its own verdict rather than ``CONVERT`` because migrate
        cannot repair it, and synthesising a v3 block over it here would
        destroy what a human could still recover by hand.
    """
    raw = _read_json(resolve_processing_state_path(output_dir))
    if raw is None:
        return None
    config_raw = raw.get(ProcessingStateKey.CONFIG)
    config = dict(config_raw) if isinstance(config_raw, Mapping) else {}

    restart_epoch = config.get("restart_epoch")
    if not isinstance(restart_epoch, int) or isinstance(restart_epoch, bool):
        restart_epoch = 0
    pipeline_sha256 = config.get("pipeline_sha256")
    config["restart_epoch"] = restart_epoch
    config["processing_generation"] = derive_processing_generation(
        pipeline_sha256=(
            pipeline_sha256 if isinstance(pipeline_sha256, str) else None
        ),
        per_image_config=None,
        restart_epoch=restart_epoch,
    )

    datasets_raw = raw.get(ProcessingStateKey.DATASETS)
    datasets_in = (
        datasets_raw if isinstance(datasets_raw, Mapping) else {}
    )
    datasets: dict[str, Mapping[str, object]] = {}
    retained: dict[str, tuple[str, ...]] = {}
    for name, entry in datasets_in.items():
        if not isinstance(entry, Mapping):
            datasets[str(name)] = entry  # type: ignore[assignment]
            continue
        converted = dict(entry)
        for key in _DEAD_DATASET_KEYS:
            converted.pop(key, None)
        if _completed_is_fully_consumed(
            output_dir, str(name), converted.get(ProcessingStateKey.COMPLETED)
        ):
            converted.pop(ProcessingStateKey.COMPLETED, None)
        failed = converted.get(ProcessingStateKey.FAILED)
        if isinstance(failed, list) and failed:
            retained[str(name)] = tuple(str(item) for item in failed)
        datasets[str(name)] = converted

    return PlannedState(
        config=config, datasets=datasets, retained_failures=retained
    )


def apply_processing_state(output_dir: Path, planned: PlannedState) -> bool:
    """Write the planned state over the file it was planned from.

    Takes the plan as **data**, so plan and apply cannot disagree about what
    the conversion is -- the same seam
    :func:`apply_per_image_records` uses.

    Idempotent: every component of the new ``config`` is a function of the
    tree, so a second pass computes the same block and writes the same bytes.

    Args:
        output_dir: Run output root.
        planned: The output of :func:`plan_processing_state`.

    Returns:
        ``True`` when the state was rewritten.
    """
    path = resolve_processing_state_path(output_dir)
    raw = _read_json(path)
    if raw is None:
        return False
    payload = dict(raw)
    payload[ProcessingStateKey.CONFIG] = dict(planned.config)
    payload[ProcessingStateKey.DATASETS] = {
        name: dict(entry) if isinstance(entry, Mapping) else entry
        for name, entry in planned.datasets.items()
    }
    atomic_write_json(path, payload)
    return True


def convert_processing_state(output_dir: Path) -> bool:
    """Convert ``processing_state.json`` to the v3 schema.

    Unconditional at the call site and a cheap no-op on a tree that has no
    state -- the same shape as :func:`convert_per_image_markers`, so the
    migrator needs no guard around either.

    Args:
        output_dir: Run output root.

    Returns:
        ``True`` when the state was rewritten.
    """
    planned = plan_processing_state(output_dir)
    if planned is None:
        return False
    return apply_processing_state(output_dir, planned)
