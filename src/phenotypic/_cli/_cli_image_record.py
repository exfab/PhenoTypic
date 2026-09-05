"""The per-image record's writers (spec §6.1).

**Writers here, readers in :mod:`phenotypic.sdk_._image_record`.** That module
carries the reasoning for the split; the short form is that P6 Task 0 moves
``valid_image_success`` into ``sdk_``, which INV-LAYER forbids from importing
this package.

**No lock, and that is a ruling rather than an omission (CAN-6).** These
writers do read-merge-``atomic_write_json``. `atomic_write_json` is a
temp-write plus ``os.replace``, so a crash mid-write leaves the previous
record intact -- but it is not a compare-and-swap, and nothing here re-reads
under exclusion. What makes that safe is **INV-ONEWRITER: at most one process
writes a given image's record at a time**, enforced by

* **disjoint work partitioning** -- each SLURM array task owns a disjoint
  image list, and the local pool partitions the same way, so one image maps to
  one task maps to one writer; and
* **the controller's refusal to submit over a live array** --
  ``_cli_staged_controller.py`` returns early unless
  ``scheduler_job_is_active(active_job_id)`` is ``False``, and note it tests
  ``is not False``, so an *unknown* scheduler state also blocks. That
  fail-safe is load-bearing: an optimization treating unknown as inactive
  breaks INV-ONEWRITER silently.

**Stage sequencing is NOT the proof**, though it reads like one. "Stage 2
cannot start before stage 1" is a claim about logical ordering, not process
concurrency -- two processes can both believe they own stage 3 of one image.
Cite the two mechanisms above; sequencing alone is unfalsifiable.

The third writer class -- ``--mode migrate``'s whole-tree sweeps -- is covered
by **mode exclusivity** rather than partitioning, since there is nothing to
partition. That is the mechanism whose precondition a later edit can remove:
the day a sweep writer is called from a normal run, INV-ONEWRITER is false and
nothing in the partitioning argument notices.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from phenotypic.sdk_ import CommitGuard, atomic_write_json, image_record_path
from phenotypic.sdk_._image_record import (
    PROVENANCE_FORWARD,
    RECORD_VERSION,
    read_image_record,
)

__all__ = [
    "consume_stage",
    "publish_image_record",
    "record_stage",
]


def _now() -> str:
    """Return a UTC timestamp at millisecond resolution."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _existing_stages(
    output_dir: Path, dataset: str, image_stem: str
) -> dict[str, object]:
    """Return the stages already on disk for one image, or an empty map.

    Reads through :func:`read_image_record`, so an unreadable record degrades
    to "no stages recorded" rather than raising -- which is the same answer a
    caller would get from a tree where nothing has run yet, and the safe one:
    a stage is then re-run rather than skipped on the strength of a file
    nobody could parse.
    """
    record = read_image_record(output_dir, dataset, image_stem)
    if record is None:
        return {}
    stages = record.get("stages")
    return dict(stages) if isinstance(stages, dict) else {}


def publish_image_record(
    output_dir: Path,
    *,
    work_id: str,
    dataset: str,
    image_stem: str,
    relative_image_path: str,
    mode: str,
    stages: Mapping[str, Mapping[str, object]],
    artifacts: Mapping[str, Path],
    attempt_id: str,
    lifecycle_epoch: str,
    provenance: str = PROVENANCE_FORWARD,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Certify one image's artifacts and publish its record last.

    **``stages`` is a contribution, not a replacement (CAN-6 rule 1).** It is
    unioned with whatever is on disk. Replacing the map is what the three
    collapsed marker trees made impossible and this one file makes easy: the
    SLURM Stage-3 worker calls this *before* recording stage 3, so a replacing
    write would drop ``stage1`` and ``stage2`` and leave the image looking
    unprocessed to the next resume.

    ``lifecycle_epoch``, **not** ``scheduler_epoch``. §5.1's five-token
    collapse was withdrawn, and this value already has exactly one on-disk
    name -- it is what ``publish_image_success`` writes into every image
    marker. Giving a new artifact a second spelling is that collapse arriving
    from the other direction.

    Args:
        output_dir: Run output root.
        work_id: The content-derived identity this record certifies.
        dataset: Dataset name.
        image_stem: Source image stem.
        relative_image_path: The source image's path relative to the input
            root, retained so a reader can name the input without a scan.
        mode: ``"full"``, ``"process"`` or ``"measure"``.
        stages: Stage entries to merge in, keyed by the ``STAGE_*``
            constants. An unknown key is accepted -- ``stages`` is an open map
            (§6.1) -- so a future stage is additive rather than a schema
            break.
        artifacts: Artifact paths by role. Each is resolved ``strict=True``
            and described by kind, so a role naming a nonexistent path fails
            here rather than certifying it.
        attempt_id: The scheduler attempt that produced this work.
        lifecycle_epoch: The lifecycle generation current at publication.
        provenance: ``PROVENANCE_FORWARD`` for every normal run. Only
            ``--mode migrate`` writes ``PROVENANCE_MIGRATED``, and any forward
            run that rewrites this record restores ``"forward"`` by taking the
            default -- which is what makes U-10 self-limiting rather than a
            permanent hole.
        commit_guard: Publication guard threaded to the atomic replace.

    Returns:
        The written record's path.

    Raises:
        OSError: An artifact path does not resolve, or the write fails.
    """
    from ._cli_completion import _artifact_descriptor

    output_root = output_dir.resolve()
    descriptors: dict[str, dict[str, object]] = {}
    for name, artifact in artifacts.items():
        resolved = Path(artifact).resolve(strict=True)
        descriptors[name] = _artifact_descriptor(
            resolved, resolved.relative_to(output_root)
        )

    merged = _existing_stages(output_dir, dataset, image_stem)
    merged.update({str(key): dict(value) for key, value in stages.items()})

    record = {
        "version": RECORD_VERSION,
        "work_id": work_id,
        "dataset": dataset,
        "image_stem": image_stem,
        "relative_image_path": relative_image_path,
        "mode": mode,
        "provenance": provenance,
        "stages": merged,
        "artifacts": descriptors,
        "attempt_id": attempt_id,
        "lifecycle_epoch": lifecycle_epoch,
        "completed_at": _now(),
    }
    path = image_record_path(output_dir, dataset, image_stem)
    atomic_write_json(path, record, commit_guard=commit_guard)
    return path


def record_stage(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    stage: str,
    payload: Mapping[str, object],
    *,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Merge one stage entry into an image's record, leaving the rest alone.

    The three collapsed trees were independently writable and must stay so:
    Stage 2 and Stage 3 run in different jobs, on different nodes, minutes
    apart, and neither may erase the other's entry.

    Writes a **partial** record when none exists yet -- ``stages`` and the
    identity fields this call knows, and no ``artifacts``. That is deliberate:
    a stage can complete long before an image has anything to certify, and a
    reader distinguishes the two by ``artifacts`` being absent rather than by
    the record being absent.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        image_stem: Source image stem.
        stage: A ``STAGE_*`` constant, or any future stage name.
        payload: The stage entry, conventionally carrying at least ``at``.
        commit_guard: Publication guard threaded to the atomic replace.

    Returns:
        The written record's path.
    """
    existing = read_image_record(output_dir, dataset, image_stem)
    record: dict[str, object] = dict(existing) if existing else {}
    stages = record.get("stages")
    merged = dict(stages) if isinstance(stages, dict) else {}
    merged[stage] = dict(payload)

    record.setdefault("version", RECORD_VERSION)
    record.setdefault("dataset", dataset)
    record.setdefault("image_stem", image_stem)
    record["stages"] = merged
    path = image_record_path(output_dir, dataset, image_stem)
    atomic_write_json(path, record, commit_guard=commit_guard)
    return path


def consume_stage(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    stage: str,
    *,
    commit_guard: CommitGuard | None = None,
) -> bool:
    """Remove one stage entry, returning whether it was there to remove.

    **Idempotent, and the return value is the whole interface.** Consuming a
    stage twice is not an error -- a retried worker must be able to clean up
    after a predecessor that already did -- so the bool says what happened
    rather than an exception saying it did not.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        image_stem: Source image stem.
        stage: The stage entry to remove.
        commit_guard: Publication guard threaded to the atomic replace.

    Returns:
        ``True`` when the entry existed and was removed, ``False`` when there
        was nothing to remove -- including when the record is absent or
        unreadable.
    """
    record = read_image_record(output_dir, dataset, image_stem)
    if record is None:
        return False
    stages = record.get("stages")
    if not isinstance(stages, dict) or stage not in stages:
        return False

    remaining = {key: value for key, value in stages.items() if key != stage}
    updated = dict(record)
    updated["stages"] = remaining
    atomic_write_json(
        image_record_path(output_dir, dataset, image_stem),
        updated,
        commit_guard=commit_guard,
    )
    return True
