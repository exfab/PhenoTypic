"""Consumable Stage-2 completion token for the staged GPU engine.

Replaces the ``.npy`` objmap sidecar. Stage 2 retains its **raw** detector
output under ``stage2_raw/`` and drops this token; Stage 3 replays the raw
array and consumes both, exactly as it used to consume the sidecar. Stage 2
does **not** write into the promoted store -- only the final store needs
third-party interop, and an in-store write would be visible to the uncached
crop route as raw pre-``drop_frame_background`` labels.

Both halves are keyed by **detector slot** as well as by image
(:func:`detector_slot`). At ``N == 1`` that is one extra directory level and
no behavioural difference; it exists so that supporting more than one
``GpuDetector`` is an additive feature rather than a migration of the on-disk
layout of a signal a 33,923-image run depends on (nested-staging spec §4.4,
§13). Every path helper here therefore takes ``slot``, with **no default** --
a default would let a caller reach a shared path without being handed a slot,
which is precisely the structural hole the keying closes.

The token is deliberately **not** NGFF metadata. Using ``ome.labels`` as the
"Stage 2 done" signal is not an exact replacement for the objmap sidecar's
existence probe, and would break resume in two ways:

* The old signal was **consumable** -- Stage 3 deleted the sidecar at the end,
  and the resume planner's ``"complete"`` branch tests its **absence**. A
  durable labels list makes that conjunct permanently false, so ``"complete"``
  never fires and every finished image is reprocessed. It also silently
  disables ``migrate_legacy_stage3_markers``.
* The labels list is not the only discovery path: ``zarr.Group.members()``
  enumerates children by store listing and returns a partially written
  ``objmap``, which reads as a mix of real labels and ``fill_value``. NGFF only
  says label images SHOULD be listed; it grants no exclusivity.

Consequently, NGFF metadata never carries resume state. Resume state lives in
``.phenotypic/progress/``, where the rest of it already lives.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from phenotypic.sdk_ import (
    CommitGuard,
    DIR_STAGE2_DONE,
    atomic_write_with_writer,
    progress_dir,
    publication_commit,
)
from phenotypic.sdk_._image_record import STAGE_STAGE2  # noqa: F401

# ``STAGE_STAGE2`` is imported but not called here, and that is the point of
# CAN-27 rather than an oversight. This module is the Stage-2 vocabulary, and
# the stage it signals must have exactly ONE spelling in the tree; a future
# writer added here reaches for the constant already in scope instead of typing
# `"stage2"` a second time. `test_the_stage_names_come_from_one_shared_constant`
# asserts `_cli_stage2_token.STAGE_STAGE2 is STAGE_STAGE2` -- `is`, not `==`,
# because two modules that happen to spell the same string identically is
# precisely the state the constant exists to make unrepresentable.


# ---------------------------------------------------------------------------
# The detector slot
# ---------------------------------------------------------------------------

_UNSAFE = re.compile(r"[^A-Za-z0-9]+")


def detector_slot(gpu_path: Sequence[str]) -> str:
    """A filesystem-safe, collision-proof id for one staged detector.

    Readable half: each path segment with runs of non-alphanumerics collapsed
    to ``-``, joined by ``__``. Safe half: 8 hex characters of the EXACT path,
    so two paths that sanitise alike (``ops[0]`` and ``ops-0``) never collide.

    The input is ``StagePlan.gpu_path``, which is also the detector's recorded
    ``pipeline_step_path`` -- one addressing scheme, not two::

        ("CompositeDetector", "ops[0]") -> "CompositeDetector__ops-0__<8 hex>"

    Args:
        gpu_path: Tree path to the detector, addressed from the root pipeline.

    Returns:
        The slot id.
    """
    exact = "/".join(gpu_path)
    readable = "__".join(_UNSAFE.sub("-", part).strip("-") for part in gpu_path)
    digest = hashlib.sha256(exact.encode("utf-8")).hexdigest()[:8]
    return f"{readable}__{digest}"


def staged_detector_slot(pipeline_path: Path | str) -> str:
    """The slot for a serialized pipeline's single ``GpuDetector``.

    The adapter for the handful of sites that hold a **pipeline path** but no
    :class:`~phenotypic._cli._cli_pipeline_split.StagePlan` -- the SLURM
    script generator, the checkpoint handler's reconciliation, and the CLI's
    continuation preflight. Everywhere a ``StagePlan`` is already in scope,
    call ``detector_slot(plan.gpu_path)`` directly rather than re-loading the
    pipeline.

    The splitter is imported lazily: ``_cli_pipeline_split`` imports
    ``_cli_validation``, and this module is imported by the resume layer that
    both of those sit above.

    Args:
        pipeline_path: Path to the serialized pipeline JSON.

    Returns:
        The slot id for its ``GpuDetector``.

    Raises:
        ValueError: The pipeline has no ``GpuDetector``, or has one the staged
            engine cannot drive.
    """
    from phenotypic import ImagePipeline

    from ._cli_pipeline_split import split_pipeline_at_gpu

    plan = split_pipeline_at_gpu(ImagePipeline.from_json(Path(pipeline_path)))
    return detector_slot(plan.gpu_path)


# ---------------------------------------------------------------------------
# The consumable token
# ---------------------------------------------------------------------------


def stage2_token_path(
    output_dir: Path, dataset: str, image_stem: str, slot: str
) -> Path:
    """``<output>/.phenotypic/progress/stage2_done/<dataset>/<slot>/<stem>.json``.

    The ``stage2_done`` segment comes from
    :data:`~phenotypic.sdk_.DIR_STAGE2_DONE` rather than a module-private
    literal. This tree is **retained** by the §6.1 collapse (U-9) -- unlike
    ``stage3_complete/``, whose segment stayed private because P3 removes it --
    so the name is a durable layout fact with a second reader in the schema
    gate, which must keep *not* firing on it. The gate tests ``is_dir()`` on
    two named segments and never walks depth, so the ``<slot>`` level is
    invisible to it.
    """
    return (
        progress_dir(output_dir)
        / DIR_STAGE2_DONE
        / dataset
        / slot
        / f"{image_stem}.json"
    )


def write_stage2_token(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    slot: str,
    *,
    objmap_shape: tuple[int, int],
    detector_duration_seconds: float = 0.0,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Atomically record that Stage 2 finished this image, for this detector.

    Carries the objmap shape and detector compute duration. It omits ``work_id``,
    which ``stage2_detect_core`` has no parameter for and which could therefore
    only ever be ``None`` (ledger **FLOW-20**). The work-id conjunct that
    matters is read off the store by ``staged_store_matches_work_id``.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        image_stem: Image stem.
        slot: Detector slot from :func:`detector_slot`.
        objmap_shape: Level-0 ``(y, x)`` extent of the detected objmap.
        detector_duration_seconds: Stage-2 detector compute time, excluding
            queueing and later Stage-3 merge work.

    Returns:
        The token path.
    """
    final = stage2_token_path(output_dir, dataset, image_stem, slot)
    # ``int()`` rather than passing the tuple through: a shape derived from a
    # numpy expression holds ``np.int64``, which ``json.dumps`` refuses.
    payload = {
        "objmap_shape": [int(objmap_shape[0]), int(objmap_shape[1])],
        "detector_duration_seconds": float(detector_duration_seconds),
    }

    def _write(path: str) -> None:
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    atomic_write_with_writer(final, _write, commit_guard=commit_guard)
    return final


def stage2_token_exists(
    output_dir: Path, dataset: str, image_stem: str, slot: str
) -> bool:
    """Return whether Stage 2 has finished and Stage 3 has not yet consumed."""
    return stage2_token_path(output_dir, dataset, image_stem, slot).is_file()


def read_stage2_token(
    output_dir: Path, dataset: str, image_stem: str, slot: str
) -> dict:
    """Read the token payload.

    Raises:
        FileNotFoundError: If the token does not exist.
    """
    return json.loads(
        stage2_token_path(output_dir, dataset, image_stem, slot).read_text(
            encoding="utf-8"
        )
    )


def delete_stage2_token(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    slot: str,
    *,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Consume the token. Idempotent, mirroring the old sidecar delete."""
    with publication_commit(commit_guard):
        stage2_token_path(output_dir, dataset, image_stem, slot).unlink(
            missing_ok=True
        )


def find_stage2_token(
    output_dir: Path, dataset: str, image_stem: str
) -> Path | None:
    """Locate an image's Stage-2 token **without knowing its slot**.

    The one legitimate slot-free read, and it is explicit rather than a
    defaulted parameter. ``--mode migrate`` has no pipeline and therefore no
    ``StagePlan``, but still has to notice an interrupted Stage-2 state so it
    can record it (``_cli_migrate_state._stage2_entry``). Without this, a
    hand-built ``<ds>/<stem>.json`` path returns ``None`` for every modern
    token and migrate silently stops recording that state.

    Searches the legacy pre-slot-keying path first, then each slot directory.
    **Reads only** -- it never moves, renames or unlinks the token, because a
    staged run live across the migrate would otherwise lose every un-consumed
    Stage-2 result.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        image_stem: Image stem.

    Returns:
        The token path, or ``None`` when this image has no token in any slot.
    """
    base = progress_dir(output_dir) / DIR_STAGE2_DONE / dataset
    legacy = base / f"{image_stem}.json"
    if legacy.is_file():
        return legacy
    if not base.is_dir():
        return None
    for slot_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        candidate = slot_dir / f"{image_stem}.json"
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# The retained raw detector output
# ---------------------------------------------------------------------------

_STAGE2_RAW_DIR = "stage2_raw"


def stage2_raw_path(
    output_dir: Path, dataset: str, image_stem: str, slot: str
) -> Path:
    """``<output>/.phenotypic/progress/stage2_raw/<dataset>/<slot>/<stem>.npy``."""
    return (
        progress_dir(output_dir)
        / _STAGE2_RAW_DIR
        / dataset
        / slot
        / f"{image_stem}.npy"
    )


def write_stage2_raw(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    array: np.ndarray,
    slot: str,
    *,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Atomically retain Stage 2's **raw** detector output for Stage 3 to replay.

    This is what makes Stage 3 idempotent under retry. Stage 3 re-promotes the
    store over its own objmap, so the store cannot serve as its own input a
    second time: on a replay ``_write_object_output`` would run again on
    already-refined labels, and ``drop_frame_background`` would zero whichever
    real colony touches the frame most -- silently, once per retry.

    Written before the token, so a crash between them leaves no token and
    Stage 2 simply re-runs. That ordering is the module-wide rule: **write raw
    then token, delete token then raw**, so the only reachable intermediate
    state is "no token, orphan raw".

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        image_stem: Image stem.
        array: The raw detector labels, stored dtype-exact.
        slot: Detector slot from :func:`detector_slot`.

    Returns:
        The retained array's path.
    """
    final = stage2_raw_path(output_dir, dataset, image_stem, slot)

    def _write(path: str) -> None:
        with open(path, "wb") as handle:
            np.save(handle, array)

    atomic_write_with_writer(final, _write, commit_guard=commit_guard)
    return final


def stage2_result_replayable(
    output_dir: Path, dataset: str, image_stem: str, slot: str
) -> bool:
    """Return whether Stage 3 can actually replay this image's Stage-2 result.

    **Both halves, never the token alone.** The token is only a *flag*; Stage
    3's real input is the retained raw ``.npy``. A token-present/raw-missing
    state -- a partial cleanup, a truncated copy -- makes
    :func:`load_stage2_raw` raise an uncaught ``FileNotFoundError`` inside
    ``stage_event``, which is reported as a terminal **scientific** failure
    instead of a missing prerequisite. Combined with
    ``classify_staged_image``'s token-present/raw-missing branch, such an image
    is otherwise permanently unreachable (ledger **FLOW-17**, extended by
    **M7**).

    One function so the five probe sites cannot drift: the local strategy's
    Stage-2 filter, its Stage-3 gate, its ``--layer objmap`` gate, the SLURM
    shard worker's candidate filter, and the recovery controller's
    already-done skip.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        image_stem: Image stem.
        slot: Detector slot from :func:`detector_slot`.

    Returns:
        ``True`` only when the token **and** the raw array are both present.
    """
    return (
        stage2_token_exists(output_dir, dataset, image_stem, slot)
        and stage2_raw_path(output_dir, dataset, image_stem, slot).is_file()
    )


def load_stage2_raw(
    output_dir: Path, dataset: str, image_stem: str, slot: str
) -> np.ndarray:
    """Load the retained raw detector output.

    Raises:
        FileNotFoundError: If Stage 2 did not retain one.
    """
    return np.load(stage2_raw_path(output_dir, dataset, image_stem, slot))


def delete_stage2_raw(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    slot: str,
    *,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Consume the raw array. Idempotent; always paired with the token."""
    with publication_commit(commit_guard):
        stage2_raw_path(output_dir, dataset, image_stem, slot).unlink(
            missing_ok=True
        )


# ---------------------------------------------------------------------------
# One-time relocation of a pre-slot-keying signal
# ---------------------------------------------------------------------------


def _legacy_stage2_raw_path(
    output_dir: Path, dataset: str, image_stem: str
) -> Path:
    """The pre-slot-keying raw path: no ``<slot>`` level."""
    return (
        progress_dir(output_dir) / _STAGE2_RAW_DIR / dataset / f"{image_stem}.npy"
    )


def _legacy_stage2_token_path(
    output_dir: Path, dataset: str, image_stem: str
) -> Path:
    """The pre-slot-keying token path: no ``<slot>`` level."""
    return (
        progress_dir(output_dir) / DIR_STAGE2_DONE / dataset / f"{image_stem}.json"
    )


def relocate_legacy_stage2_signal(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    slot: str,
    *,
    slot_count: int = 1,
) -> bool:
    """Move a pre-slot-keying signal into its slot directory. Returns moved?

    A staged run interrupted before slot keying and resumed after it would
    otherwise not find its signals and recompute Stage 2 -- correct, but paid
    in GPU time on the scarcest resource in the cluster.

    **Moves BOTH halves.** The Stage-2 *signal* is two files -- the retained
    raw and the consumable token -- and :func:`stage2_result_replayable`
    requires both. Moving only the raw leaves the slot-keyed token absent, so
    the predicate stays ``False``, Stage 2 recomputes anyway, and the tree is
    half-migrated: the exact cost this helper exists to avoid, plus a mess.

    Moves the **raw first, then the token**, mirroring the write order, so an
    interrupted relocation can never leave a slot-keyed token with no
    slot-keyed raw. The only reachable intermediate state stays "no token,
    orphan raw".

    Never overwrites a current signal: if either slot-keyed path already
    exists, the legacy file is stale and the current one wins.

    ``slot_count`` is how many slots the caller's plan has, not a count this
    function can discover -- ``StagePlan`` has no ``slots`` attribute, and
    under the shipped ``N == 1`` model a plan has exactly one slot by
    construction. When a caller declares more than one, this **refuses**: a
    per-image legacy signal is one detector's output and cannot be attributed
    to one of several slots, and guessing would hand detector B the mask
    detector A produced. Refusing costs a recompute; guessing costs a wrong
    answer.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        image_stem: Image stem.
        slot: The single detector slot to relocate into.
        slot_count: Number of slots the caller's plan carries. Anything other
            than 1 refuses.

    Returns:
        ``True`` if at least one half was moved, ``False`` otherwise.
    """
    legacy_raw = _legacy_stage2_raw_path(output_dir, dataset, image_stem)
    legacy_token = _legacy_stage2_token_path(output_dir, dataset, image_stem)
    if not legacy_raw.is_file() and not legacy_token.is_file():
        return False
    if slot_count != 1:
        return False

    current_raw = stage2_raw_path(output_dir, dataset, image_stem, slot)
    current_token = stage2_token_path(output_dir, dataset, image_stem, slot)
    if current_raw.exists() or current_token.exists():
        return False

    moved = False
    # Raw first, then the token -- the write order, for the reason above.
    if legacy_raw.is_file():
        current_raw.parent.mkdir(parents=True, exist_ok=True)
        os.replace(legacy_raw, current_raw)
        moved = True
    if legacy_token.is_file():
        current_token.parent.mkdir(parents=True, exist_ok=True)
        os.replace(legacy_token, current_token)
        moved = True
    return moved
