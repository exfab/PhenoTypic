"""Consumable Stage-2 completion token for the staged GPU engine.

Replaces the ``.npy`` objmap sidecar. Stage 2 retains its **raw** detector
output under ``stage2_raw/`` and drops this token; Stage 3 replays the raw
array and consumes both, exactly as it used to consume the sidecar. Stage 2
does **not** write into the promoted store -- only the final store needs
third-party interop, and an in-store write would be visible to the uncached
crop route as raw pre-``drop_frame_background`` labels.

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

import json
from pathlib import Path

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


def stage2_token_path(output_dir: Path, dataset: str, image_stem: str) -> Path:
    """``<output>/.phenotypic/progress/stage2_done/<dataset>/<stem>.json``.

    The segment comes from :data:`~phenotypic.sdk_.DIR_STAGE2_DONE` rather than
    a module-private literal. This tree is **retained** by the §6.1 collapse
    (U-9) -- unlike ``stage3_complete/``, whose segment stayed private because
    P3 removes it -- so the name is a durable layout fact with a second reader
    in the schema gate, which must keep *not* firing on it.
    """
    return (
        progress_dir(output_dir) / DIR_STAGE2_DONE / dataset / f"{image_stem}.json"
    )


def write_stage2_token(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    *,
    objmap_shape: tuple[int, int],
    detector_duration_seconds: float = 0.0,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Atomically record that Stage 2 finished this image.

    Carries the objmap shape and detector compute duration. It omits ``work_id``,
    which ``stage2_detect_core`` has no parameter for and which could therefore
    only ever be ``None`` (ledger **FLOW-20**). The work-id conjunct that
    matters is read off the store by ``staged_store_matches_work_id``.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        image_stem: Image stem.
        objmap_shape: Level-0 ``(y, x)`` extent of the detected objmap.
        detector_duration_seconds: Stage-2 detector compute time, excluding
            queueing and later Stage-3 merge work.

    Returns:
        The token path.
    """
    final = stage2_token_path(output_dir, dataset, image_stem)
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


def stage2_token_exists(output_dir: Path, dataset: str, image_stem: str) -> bool:
    """Return whether Stage 2 has finished and Stage 3 has not yet consumed."""
    return stage2_token_path(output_dir, dataset, image_stem).is_file()


def read_stage2_token(output_dir: Path, dataset: str, image_stem: str) -> dict:
    """Read the token payload.

    Raises:
        FileNotFoundError: If the token does not exist.
    """
    return json.loads(
        stage2_token_path(output_dir, dataset, image_stem).read_text(encoding="utf-8")
    )


def delete_stage2_token(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    *,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Consume the token. Idempotent, mirroring the old sidecar delete."""
    with publication_commit(commit_guard):
        stage2_token_path(output_dir, dataset, image_stem).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The retained raw detector output
# ---------------------------------------------------------------------------

_STAGE2_RAW_DIR = "stage2_raw"


def stage2_raw_path(output_dir: Path, dataset: str, image_stem: str) -> Path:
    """``<output>/.phenotypic/progress/stage2_raw/<dataset>/<stem>.npy``."""
    return progress_dir(output_dir) / _STAGE2_RAW_DIR / dataset / f"{image_stem}.npy"


def write_stage2_raw(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    array: np.ndarray,
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
    Stage 2 simply re-runs.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        image_stem: Image stem.
        array: The raw detector labels, stored dtype-exact.

    Returns:
        The retained array's path.
    """
    final = stage2_raw_path(output_dir, dataset, image_stem)

    def _write(path: str) -> None:
        with open(path, "wb") as handle:
            np.save(handle, array)

    atomic_write_with_writer(final, _write, commit_guard=commit_guard)
    return final


def stage2_result_replayable(
    output_dir: Path, dataset: str, image_stem: str
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

    Returns:
        ``True`` only when the token **and** the raw array are both present.
    """
    return (
        stage2_token_exists(output_dir, dataset, image_stem)
        and stage2_raw_path(output_dir, dataset, image_stem).is_file()
    )


def load_stage2_raw(output_dir: Path, dataset: str, image_stem: str) -> np.ndarray:
    """Load the retained raw detector output.

    Raises:
        FileNotFoundError: If Stage 2 did not retain one.
    """
    return np.load(stage2_raw_path(output_dir, dataset, image_stem))


def delete_stage2_raw(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    *,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Consume the raw array. Idempotent; always paired with the token."""
    with publication_commit(commit_guard):
        stage2_raw_path(output_dir, dataset, image_stem).unlink(missing_ok=True)
